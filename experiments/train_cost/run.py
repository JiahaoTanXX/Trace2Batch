# Copyright (c) 2026 Jiahao Tan
# pyright: reportMissingImports=false

"""P3-A: 在相同监督目标与更新预算下测量两种样本表示的训练成本。

数据在 CPU 上预先准备; 不运行在线 rollout, 不提交 Kaggle 任务。
--check-only 不导入训练依赖、不加载权重, 可用于提交前预检。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

MODEL_REPO = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
# "sdpa" 是第一、二阶段的路径; "sdpa_repeat_kv" (第三阶段) 显式复制 KV 头并强制 memory-efficient 内核。
ATTENTION_CHOICES = ("sdpa", "sdpa_repeat_kv")


def atomic_write(path: Path, report: dict[str, Any]) -> None:
    """先编码再替换, 禁止 NaN/Infinity, 保留上次成功写入的结果。"""
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


# Static loss scale for the gradient check, like the GradScaler used by the measured updates; without it,
# fp16 gradients underflow on long rows. Scales from 256 to 65536 behave the same.
GATE_LOSS_SCALE = 4096.0


def _versions() -> dict[str, Any]:
    packages: dict[str, Any] = {"python": platform.python_version(), "platform": platform.system()}
    for package in ("torch", "transformers", "peft", "accelerate", "safetensors"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return packages


def load_model(
    device: str, *, seed: int, gradient_checkpointing: bool, cpu_threads: int, attention: str = "sdpa"
) -> tuple[Any, dict[str, Any]]:
    """仅加载固定的小模型; CUDA 不可用时在下载前失败。"""
    import torch

    if attention not in ATTENTION_CHOICES:
        raise ValueError(f"attention 必须是 {ATTENTION_CHOICES} 之一。")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用, 停止而不下载模型。")
    if device == "cpu":
        torch.set_num_threads(cpu_threads)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    if attention != "sdpa":
        from experiments.train_cost import attention as long_attention

        long_attention.register()

    cache = Path(os.environ.get("TRACE2BATCH_HF_CACHE", Path.home() / ".cache" / "trace2batch-hf" / "hub"))
    # 自动模型的第三方类型声明不能完整表达 PEFT 和 torch 的训练接口。
    base: Any = cast(Any, AutoModelForCausalLM).from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        cache_dir=cache,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation=attention,
    )
    base.config.use_cache = False
    base.config.attention_dropout = 0.0
    model: Any = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    )
    model.to(device)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if not trainable:
        raise RuntimeError("没有可训练 LoRA 参数。")
    info = {
        "trainable_parameters": trainable,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dtype": "float16" if device == "cuda" else "float32",
        "attention_implementation": model.config._attn_implementation,
        "gradient_checkpointing": gradient_checkpointing,
        "dropout": 0.0,
        "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"], "bias": "none"},
        "vocab_size": model.config.vocab_size,
    }
    if device == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name()
        info["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        info["cuda_version"] = torch.version.cuda
    return model, info


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--time-budget-seconds", type=float, default=2400.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-rollouts", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--diagnose-gradients",
        action="store_true",
        help="诊断模式: 对每一对样本在多个 loss scale 下比较梯度, 不做计时, 核对结果不作为停止条件。",
    )
    parser.add_argument("--loss-scales", default="1,256,4096,65536", help="诊断模式使用的 loss scale, 逗号分隔")
    parser.add_argument("--groups", default=None, help="只运行这些分组 (逗号分隔, 例如 dataset|lo-hi|k8)")
    parser.add_argument(
        "--attention", choices=ATTENTION_CHOICES, default="sdpa", help="长上下文用 sdpa_repeat_kv (第三阶段)"
    )
    parser.add_argument(
        "--policy-logits-only",
        action="store_true",
        help="只在监督位置计算 logits; loss 与梯度不变, 长上下文时省下全序列 logits 的显存",
    )
    return parser


def _diagnose(
    report: dict[str, Any],
    output: Path,
    grouped: dict[str, list[Any]],
    model: Any,
    args: argparse.Namespace,
    loss_scales: tuple[float, ...],
    deadline: float,
) -> None:
    """诊断每一对样本的梯度一致性; 不计时, 也不因核对未通过而停止。"""
    from experiments.train_cost import engine

    report["gradient_diagnostics"] = {}
    report["loss_scales"] = list(loss_scales)
    cuda = args.device == "cuda"
    torch: Any = None
    if cuda:
        import torch
    for cell, cell_pairs in grouped.items():
        entries = report["gradient_diagnostics"].setdefault(cell, [])
        for pair in cell_pairs:
            if time.monotonic() >= deadline:
                raise TimeoutError("时间预算已到, 不再诊断新的样本。")
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            entry = engine.gradient_diagnostics(
                model,
                pair,
                device=args.device,
                loss_scales=loss_scales,
                rtol=0.05 if cuda else 1e-3,
                atol=1e-3 if cuda else 1e-5,
                deadline=deadline,
                policy_logits_only=args.policy_logits_only,
            )
            entry["seconds"] = time.monotonic() - started
            entry["turns"] = len(pair.transition)
            entry["final_context_tokens"] = len(pair.transition[-1].prompt_token_ids) + len(
                pair.transition[-1].response_token_ids
            )
            if cuda:
                entry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                entry["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            entries.append(entry)
            atomic_write(output, report)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    for name in ("time_budget_seconds", "learning_rate"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} 必须是有限正数。")
    if args.repeats < 1 or args.warmup_rollouts < 0 or args.cpu_threads < 1:
        parser.error("repeats/cpu_threads 必须为正数, warmup-rollouts 不小于 0。")
    try:
        loss_scales = tuple(float(item) for item in args.loss_scales.split(","))
    except ValueError:
        loss_scales = ()
    if not loss_scales or not all(math.isfinite(scale) and scale > 0 for scale in loss_scales):
        parser.error("--loss-scales 必须是逗号分隔的有限正数。")
    from experiments.train_cost.data import load_pairs

    started = time.monotonic()
    deadline = started + args.time_budget_seconds
    pairs, prepared = load_pairs(args.data)
    if args.groups is not None:
        wanted = args.groups.split(",")
        missing = sorted(set(wanted) - {pair.group for pair in pairs})
        if missing:
            parser.error(f"--groups 中的分组不在数据里: {missing}")
        pairs = [pair for pair in pairs if pair.group in wanted]
    model_spec = prepared.get("model", {})
    if model_spec.get("repo") != MODEL_REPO or model_spec.get("revision") != MODEL_REVISION:
        raise ValueError("模型 model 必须是已固定 revision 的 Qwen3-0.6B。")
    if not pairs:
        raise ValueError("没有可用配对样本。")
    output = args.out / "results.json"
    if output.exists():
        raise FileExistsError("结果已存在, 请换一个输出目录, 避免覆盖先前实验。")
    # One verification and one AB/BA benchmark per cell: the dataset, or dataset|stratum.
    grouped: dict[str, list[Any]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.group].append(pair)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "checked" if args.check_only else "initializing",
        "complete": False,
        "mode": "gradient_diagnostics" if args.diagnose_gradients else "benchmark",
        "model": model_spec,
        "prepared_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "data_settings": prepared.get("settings", {}),
        "selection": prepared.get("selection", {}),
        "data_summary": {
            "rollouts": len(pairs),
            "policy_tokens": sum(pair.policy_tokens for pair in pairs),
            "datasets": dict(Counter(pair.dataset for pair in pairs)),
            "cells": {key: len(value) for key, value in grouped.items()},
        },
        "training_settings": {
            "device": args.device,
            "seed": args.seed,
            "repeats": args.repeats,
            "warmup_rollouts": args.warmup_rollouts,
            "learning_rate": args.learning_rate,
            "cpu_threads": args.cpu_threads if args.device == "cpu" else None,
            "gradient_checkpointing": args.gradient_checkpointing,
            "time_budget_seconds": args.time_budget_seconds,
            "attention": args.attention,
            "policy_logits_only": args.policy_logits_only,
            "groups": args.groups,
            "microbatch": 1,
            "optimizer": "AdamW",
            "optimizer_steps_per_rollout": 1,
            "loss_normalization": "per_rollout_policy_token",
        },
        "versions": _versions(),
        "verification": {},
        "progress": [],
        "benchmark": {},
    }
    atomic_write(output, report)
    if args.check_only:
        print(json.dumps({"status": report["status"], **report["data_summary"]}, ensure_ascii=False))
        return

    def stop_requested(signum: int, frame: Any) -> None:
        raise TimeoutError(f"收到停止信号 {signum}, 保存已完成的测量。")

    previous_sigterm = signal.signal(signal.SIGTERM, stop_requested)
    try:
        model, info = load_model(
            args.device,
            seed=args.seed,
            gradient_checkpointing=args.gradient_checkpointing,
            cpu_threads=args.cpu_threads,
            attention=args.attention,
        )
        report["model_runtime"] = info
        report["model_load_seconds"] = time.monotonic() - started
        for pair in pairs:
            for row in pair.trajectory:
                if any(token >= info["vocab_size"] for token in row.prompt_token_ids + row.response_token_ids):
                    raise ValueError("prepared 数据中 token id 超出模型词表。")
        from experiments.train_cost.engine import benchmark, verify_equivalence

        report["status"] = "running"
        atomic_write(output, report)
        if args.diagnose_gradients:
            _diagnose(report, output, grouped, model, args, loss_scales, deadline)
        benchmark_cells = {} if args.diagnose_gradients else grouped
        for cell, cell_pairs in benchmark_cells.items():
            if time.monotonic() >= deadline:
                raise TimeoutError("时间预算已到, 不再启动新的分组。")
            check = verify_equivalence(
                model,
                cell_pairs[0],
                device=args.device,
                rtol=0.05 if args.device == "cuda" else 1e-3,
                atol=1e-3 if args.device == "cuda" else 1e-5,
                deadline=deadline,
                loss_scale=GATE_LOSS_SCALE,
                policy_logits_only=args.policy_logits_only,
            )
            report["verification"][cell] = check
            atomic_write(output, report)
            if not check["passed"]:
                raise RuntimeError(f"{cell}: loss/梯度一致性核对失败, 停止计时。")

            def save_record(record: dict[str, Any], cell: str = cell) -> None:
                report["progress"].append({"cell": cell, **record})
                report["elapsed_seconds"] = time.monotonic() - started
                atomic_write(output, report)

            result = benchmark(
                model,
                cell_pairs,
                device=args.device,
                repeats=args.repeats,
                warmup_rollouts=args.warmup_rollouts,
                learning_rate=args.learning_rate,
                deadline=deadline,
                on_record=save_record,
                policy_logits_only=args.policy_logits_only,
            )
            report["benchmark"][cell] = result
            atomic_write(output, report)
            if not result["complete"]:
                if result.get("status") in {"stopped", "timeout", "deadline_exceeded", "interrupted"}:
                    raise TimeoutError(f"{cell}: 仅完成部分测量。")
                raise RuntimeError(f"{cell}: 测量未完整完成 ({result.get('status')})。")
        report["status"] = "complete"
        report["complete"] = True
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "stopped" if isinstance(error, TimeoutError | KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["complete"] = False
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_write(output, report)
        signal.signal(signal.SIGTERM, previous_sigterm)
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main(sys.argv[1:])
