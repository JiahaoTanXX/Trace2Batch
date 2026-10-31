# Copyright (c) 2026 Jiahao Tan
# pyright: reportMissingImports=false

"""P3-B: 在多轮工具调用环境上做 GRPO(Kaggle 2x T4: 一张卡跑 vLLM 采样, 一张卡做 LoRA 训练)。

每一步: 生成训练任务 -> 每个任务采样 G 条轨迹(TITO 或服务端重渲染) -> 组内优势 -> 一次 on-policy 更新 ->
保存 adapter 并热加载到 vLLM(/v1/load_lora_adapter)-> 下一步用新 adapter 采样。每步写一行 steps.jsonl,
评测写 evals.jsonl, 中途被中断也不丢失已完成的步。

    # 基础模型在各难度上的成功率(只需要 vLLM 服务)
    python -m experiments.grpo.run baseline --base-url http://127.0.0.1:8000 --out results/baseline
    # GRPO 训练(vLLM 需 --enable-lora 且设置 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True)
    python -m experiments.grpo.run train --driver tito --steps 60 --out results/tito_seed0
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from experiments.grpo.env import Task, make_task
from experiments.grpo.grpo import WeightedRow, group_advantages, rollout_rows, weighted_rows

EVAL_SEED_BASE = 10**9
LEARNABLE = (0.1, 0.7)


def train_tasks(step: int, count: int, difficulties: Sequence[int], seed: int) -> list[Task]:
    """第 step 步的训练任务; 种子空间 [(seed+1)*1e6, ...) 与评测集(>= 1e9)不重叠。"""
    rng = random.Random(f"grpo-train-{seed}-{step}")
    return [make_task((seed + 1) * 10**6 + step * 1000 + i, rng.choice(list(difficulties))) for i in range(count)]


def eval_tasks(count: int, difficulties: Sequence[int]) -> list[Task]:
    """固定的 held-out 评测集, 各难度轮流出现。"""
    return [make_task(EVAL_SEED_BASE + i, difficulties[i % len(difficulties)]) for i in range(count)]


def choose_difficulties(success: dict[int, float]) -> list[int]:
    """事先固定的规则: 取基础模型成功率落在 [0.1, 0.7] 的难度; 没有时取最接近 0.4 的一个。"""
    chosen = [k for k, rate in sorted(success.items()) if LEARNABLE[0] <= rate <= LEARNABLE[1]]
    return chosen or [min(success, key=lambda k: abs(success[k] - 0.4))]


def flat_logprob_gaps(
    training: Sequence[Sequence[float]], sampled: Sequence[Sequence[float] | None]
) -> list[float] | None:
    """训练端与 vLLM 对同一批采样 token 的 |log π 差|; 两边按出现顺序展平后长度必须相同, 否则返回 None。"""
    if any(values is None for values in sampled):
        return None
    left = [v for values in training for v in values]
    right = [v for values in sampled for v in values or []]
    if len(left) != len(right):
        return None
    return [round(abs(a - b), 6) for a, b in zip(left, right, strict=True)]


def _gap_summary(gaps: list[float]) -> dict[str, Any]:
    if not gaps:
        return {"n": 0}
    ordered = sorted(gaps)
    return {
        "n": len(gaps),
        "mean": round(statistics.fmean(gaps), 6),
        "median": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
    }


def answer_summary(tasks: Sequence[Task], results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """按答案类型(总价 / 缺货)分别统计成功率, 并统计不查询就回答的 "INSUFFICIENT" 猜测与没有回答的情况。"""
    by_kind: dict[str, list[float]] = {}
    for task, result in zip(tasks, results, strict=True):
        by_kind.setdefault(task.kind, []).append(result["reward"])
    answers = [result.get("answer") for result in results]
    return {
        "success_by_kind": {k: round(statistics.fmean(v), 6) for k, v in sorted(by_kind.items())},
        "tasks_by_kind": {k: len(v) for k, v in sorted(by_kind.items())},
        "bare_insufficient_answers": sum(answer == "INSUFFICIENT" for answer in answers),
        "no_answer": sum(answer is None for answer in answers),
    }


class Trainer(Protocol):
    def update(self, rows: Sequence[WeightedRow]) -> dict[str, Any]: ...
    def compare(self, trajectory: Sequence[WeightedRow], transition: Sequence[WeightedRow]) -> dict[str, Any]: ...
    def logprobs(self, rows: Sequence[WeightedRow]) -> list[float]: ...


def _step_record(step: int, rollouts: list[dict[str, Any]], rows: list[WeightedRow]) -> dict[str, Any]:
    turns = [t for r in rollouts for t in r["turns"]]
    rewards = [r["reward"] for r in rollouts]
    return {
        "step": step,
        "rollouts": len(rollouts),
        "reward_mean": round(statistics.fmean(rewards), 6),
        "turns_mean": round(len(turns) / len(rollouts), 4),
        "self_retoken_rate": round(sum(t["self_retoken"] for t in turns) / len(turns), 6) if turns else None,
        "self_retoken_turns": sum(t["self_retoken"] for t in turns),
        "turns": len(turns),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rollouts)),
        "rows": len(rows),
        "row_tokens": sum(len(w.row.prompt_token_ids) + len(w.row.response_token_ids) for w in rows),
        "policy_tokens": sum(sum(w.row.response_mask) for w in rows),
    }


def run_training(
    *,
    steps: int,
    tasks_for_step: Callable[[int], list[Task]],
    rollout: Callable[[Task, int, str], dict[str, Any]],
    trainer: Trainer,
    publish: Callable[[int], str],
    evaluate: Callable[[str, int], dict[str, Any]],
    log: Callable[[str, dict[str, Any]], None],
    base_model: str,
    group_size: int,
    concurrency: int,
    eval_every: int,
    compare_every: int,
    logprob_rollouts: int,
    start_step: int = 0,
    deadline: float | None = None,
) -> None:
    """GRPO 主循环。第 step 步用上一步发布的模型采样; 第 0 步用基础模型(LoRA 初始为零, 与基础模型相同)。"""
    model = base_model if start_step == 0 else f"step{start_step}"
    completed = start_step
    for step in range(start_step, steps):
        if deadline is not None and time.monotonic() >= deadline:
            # 停止前已完成 step 次更新; 仍然对当前模型做最终评测(pilot2 曾因此缺少最终评测)。
            log("stopped", {"step": step, "reason": "time_budget"})
            break
        if step % eval_every == 0:
            log("eval", {"step": step, "model": model, **evaluate(model, step)})
        started = time.perf_counter()
        tasks = tasks_for_step(step)
        jobs = [(task, step * 100_000 + i * 100 + g) for i, task in enumerate(tasks) for g in range(group_size)]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rollouts = list(pool.map(lambda job, current=model: rollout(job[0], job[1], current), jobs))
        rollout_seconds = time.perf_counter() - started
        advantages: list[float] = []
        for i in range(len(tasks)):
            group = rollouts[i * group_size : (i + 1) * group_size]
            advantages += group_advantages([r["reward"] for r in group])
        events = [r["events"] for r in rollouts]
        rows = weighted_rows(events, advantages, "trajectory")
        record = _step_record(step, rollouts, rows)
        record["reward_by_kind"] = answer_summary([task for task, _ in jobs], rollouts)["success_by_kind"]
        record["rollout_seconds"] = round(rollout_seconds, 3)
        record["groups_with_signal"] = sum(
            any(a != 0 for a in advantages[i * group_size : (i + 1) * group_size]) for i in range(len(tasks))
        )
        gaps: list[float] = []
        for r in rollouts[:logprob_rollouts]:
            own = [WeightedRow(row, 1.0) for row in rollout_rows(r["events"], "trajectory")]
            values = flat_logprob_gaps([trainer.logprobs(own)], [[v for t in r["sampled_logprobs"] for v in t or []]])
            if values is not None:
                gaps += values
        record["logprob_gap"] = _gap_summary(gaps)
        if step % compare_every == 0:
            record["compare"] = trainer.compare(rows, weighted_rows(events, advantages, "transition"))
        record["update"] = trainer.update(rows)
        model = publish(step + 1)
        record["step_seconds"] = round(time.perf_counter() - started, 3)
        log("step", record)
        completed = step + 1
    log("eval", {"step": completed, "model": model, **evaluate(model, completed)})


# ---------------------------------------------------------------- GPU 部分(只在命令行入口中使用)


class TorchTrainer:
    """LoRA + fp16 autocast + GradScaler + 梯度检查点; 每次 update 只调用一次 optimizer.step()。"""

    def __init__(self, model: Any, *, lr: float, device: str) -> None:
        import torch

        from experiments.grpo.grpo import compare_representations, grpo_update, policy_token_logprobs

        self._update, self._compare, self._logprobs = grpo_update, compare_representations, policy_token_logprobs
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        self.scaler = torch.amp.GradScaler("cuda") if device.startswith("cuda") else None

    def update(self, rows: Sequence[WeightedRow]) -> dict[str, Any]:
        return self._update(self.model, self.optimizer, self.scaler, rows, device=self.device)

    def compare(self, trajectory: Sequence[WeightedRow], transition: Sequence[WeightedRow]) -> dict[str, Any]:
        return self._compare(self.model, trajectory, transition, device=self.device)

    def logprobs(self, rows: Sequence[WeightedRow]) -> list[float]:
        return [v for w in rows for v in self._logprobs(self.model, w.row, device=self.device)]

    def save(self, path: Path) -> None:
        self.model.save_pretrained(str(path))


def load_lora_model(repo: str, revision: str, *, rank: int, device: str, cache_dir: Any) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, cache_dir=cache_dir, dtype=torch.float32)
    except TypeError:  # transformers < 4.56 只认 torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, cache_dir=cache_dir, torch_dtype=torch.float32
        )
    return wrap_lora(model, rank=rank, device=device)


def wrap_lora(model: Any, *, rank: int, device: str) -> Any:
    """fp32 基础权重 + LoRA(q/k/v/o); 关闭 KV cache, 开启梯度检查点, 训练模式。"""
    from peft import LoraConfig, TaskType, get_peft_model

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, config).to(device)
    model.train()
    return model


def post_text(base_url: str, path: str, payload: dict[str, Any]) -> tuple[int, str]:
    """LoRA 管理接口返回纯文本(不是 JSON)。"""
    request = urllib.request.Request(
        base_url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, response.read().decode(errors="replace")[:500]
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace")[:500]


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _client(base_url: str, model_key: str):
    """vLLM 客户端、分词器与 rollout 函数; 只在命令行入口中构造(需要 transformers)。"""
    from transformers import AutoTokenizer

    from experiments.grpo.rollout import rollout_env
    from experiments.template_prefix.run import CACHE_DIR
    from experiments.tito_vllm.run import ROOT, V2_TEMPLATES, VllmServer
    from experiments.vllm_sampling.run import MODELS, RL_SAMPLING, http_post

    served = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(served.repo, revision=served.revision, cache_dir=CACHE_DIR)
    stop_ids = [tokenizer.convert_tokens_to_ids(token) for token in ("<|im_end|>", "<|endoftext|>")]
    v2 = (ROOT / V2_TEMPLATES[model_key]).read_text(encoding="utf-8")
    post = http_post(base_url)

    def decode(ids: Sequence[int]) -> str:
        return tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    def make_rollout(driver: str, max_turns: int, max_tokens: int) -> Callable[[Task, int, str], dict[str, Any]]:
        def rollout(task: Task, seed: int, model: str) -> dict[str, Any]:
            server = VllmServer(post, model, served.chat_template_kwargs, {"official": None, "v2": v2}, "v2")
            try:
                return rollout_env(
                    server, task, driver=driver, max_turns=max_turns, seed=seed, sampling=RL_SAMPLING,
                    max_tokens=max_tokens, decode=decode, encode=encode, stop_ids=stop_ids, logprobs=True,
                )  # fmt: skip
            except Exception as error:  # 单条 rollout 的异常记为 0 奖励并保留原因, 不终止训练。
                message = f"{type(error).__name__}: {error}"[:300]
                return {"task_id": task.task_id, "reward": 0.0, "answer": None, "stop_reason": f"exception:{message}",
                        "turns": [], "events": [], "sampled_logprobs": []}  # fmt: skip

        return rollout

    return served, make_rollout, CACHE_DIR


def _evaluate_fn(rollout: Callable[[Task, int, str], dict[str, Any]], tasks: list[Task], concurrency: int):
    def evaluate(model: str, step: int) -> dict[str, Any]:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda pair: rollout(pair[1], EVAL_SEED_BASE + pair[0], model), enumerate(tasks)))
        by_level: dict[int, list[float]] = {}
        for task, result in zip(tasks, results, strict=True):
            by_level.setdefault(len(task.order), []).append(result["reward"])
        turns = [t for r in results for t in r["turns"]]
        return {
            "tasks": len(tasks),
            "success": round(statistics.fmean(r["reward"] for r in results), 6),
            **answer_summary(tasks, results),
            "answers": [r.get("answer") for r in results],
            "turns_mean": round(sum(len(r["turns"]) for r in results) / len(results), 4),
            "success_by_difficulty": {k: round(statistics.fmean(v), 6) for k, v in sorted(by_level.items())},
            "rewards": [r["reward"] for r in results],
            "self_retoken_rate": round(sum(t["self_retoken"] for t in turns) / len(turns), 6) if turns else None,
            "stop_reasons": dict(Counter(r["stop_reason"].split(":")[0] for r in results)),
            "seconds": round(time.perf_counter() - started, 1),
        }

    return evaluate


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("baseline", "train"))
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--driver", choices=("tito", "rerender"), default="tito")
    parser.add_argument("--difficulties", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--baseline-tasks", type=int, default=50, help="baseline 模式下每个难度的任务数")
    parser.add_argument("--choose-from", type=Path, default=None, help="按 baseline 结果和事先固定的规则选择难度")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--tasks-per-step", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--eval-tasks", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--compare-every", type=int, default=10)
    parser.add_argument("--logprob-rollouts", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--adapter-dir", type=Path, default=Path("/tmp/grpo-adapters"))
    parser.add_argument("--adapter-prefix", default="", help="同一服务端上依次运行多次训练时, 区分各次的 adapter 名称")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--time-budget-seconds", type=float, default=None)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    served, make_rollout, cache_dir = _client(args.base_url, args.model)
    rollout = make_rollout(args.driver, args.max_turns, args.max_tokens)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out / "settings.json").write_text(json.dumps(settings, indent=1, sort_keys=True), encoding="utf-8")

    if args.mode == "baseline":
        tasks = [
            make_task(EVAL_SEED_BASE + 10**6 + i, k) for k in args.difficulties for i in range(args.baseline_tasks)
        ]
        result = _evaluate_fn(rollout, tasks, args.concurrency)(served.key, 0)
        result["chosen_difficulties"] = choose_difficulties(
            {int(k): v for k, v in result["success_by_difficulty"].items()}
        )
        (args.out / "baseline.json").write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
        shown = {k: v for k, v in result.items() if k not in ("rewards", "answers")}
        print(json.dumps(shown, ensure_ascii=False), flush=True)
        return

    difficulties = args.difficulties
    if args.choose_from is not None:
        difficulties = json.loads(args.choose_from.read_text(encoding="utf-8"))["chosen_difficulties"]
    model = load_lora_model(served.repo, served.revision, rank=args.lora_rank, device=args.train_device,
                            cache_dir=cache_dir)  # fmt: skip
    trainer = TorchTrainer(model, lr=args.lr, device=args.train_device)
    held_out = eval_tasks(args.eval_tasks, difficulties)
    steps_log, evals_log = args.out / "steps.jsonl", args.out / "evals.jsonl"

    def publish(step: int) -> str:
        name = f"{args.adapter_prefix}step{step}"
        path = args.adapter_dir / name
        trainer.save(path)
        status, body = post_text(args.base_url, "/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
        if status != 200:
            raise RuntimeError(f"load_lora_adapter failed: {status} {body}")
        if step > 1:
            post_text(args.base_url, "/v1/unload_lora_adapter", {"lora_name": f"{args.adapter_prefix}step{step - 1}"})
        return name

    def log(kind: str, record: dict[str, Any]) -> None:
        _append(evals_log if kind == "eval" else steps_log, {"kind": kind, **record})
        shown = {k: v for k, v in record.items() if k not in ("rewards", "answers")}
        print(json.dumps({"kind": kind, **shown}), flush=True)

    deadline = None if args.time_budget_seconds is None else time.monotonic() + args.time_budget_seconds
    run_training(
        steps=args.steps,
        tasks_for_step=lambda step: train_tasks(step, args.tasks_per_step, difficulties, args.seed),
        rollout=rollout,
        trainer=trainer,
        publish=publish,
        evaluate=_evaluate_fn(rollout, held_out, args.concurrency),
        log=log,
        base_model=served.key,
        group_size=args.group_size,
        concurrency=args.concurrency,
        eval_every=args.eval_every,
        compare_every=args.compare_every,
        logprob_rollouts=args.logprob_rollouts,
        deadline=deadline,
    )


if __name__ == "__main__":
    main()
