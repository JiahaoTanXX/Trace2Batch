# Copyright (c) 2026 Jiahao Tan

"""验证前缀稳定的模板补丁(Qwen3 / gpt-oss), 并与官方模板的结果对照。

复用前缀实验的数据、采集方式(B / C)和归因流程, 只把 chat template 换成
`templates/` 下的补丁版本。另外统计补丁让多少轮 prompt 与官方模板不同, 用来衡量
"偏离官方用法"的程度: 补丁要求 rollout 与训练都改用补丁模板, 模型看到的历史与官方推荐不同。

    python -m experiments.template_patch.run
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments.template_prefix.run import (
    DATASETS,
    TOKENIZER_CONFIGS,
    Conversation,
    Renderer,
    TokenizerConfig,
    load_conversations,
    markdown_table,
    run_config,
)

RESULTS_DIR = Path(__file__).with_name("results")
OFFICIAL_RESULTS = Path(__file__).resolve().parents[1] / "template_prefix" / "results" / "results.json"
PATCHES = {
    "qwen3_no_think": "experiments/template_patch/templates/qwen3_prefix_stable.jinja",
    "qwen3_think_keep[synthetic]": "experiments/template_patch/templates/qwen3_prefix_stable.jinja",
    "qwen3_think_drop[synthetic]": "experiments/template_patch/templates/qwen3_prefix_stable.jinja",
    "gpt-oss": "experiments/template_patch/templates/gpt_oss_prefix_stable.jinja",
}
# 补丁 v2: 在 v1 基础上, content 已以换行结尾时不再在工具调用前插入换行; 另为 Qwen2.5 官方模板做同样修改。
_QWEN3_V2 = "experiments/template_patch/templates/qwen3_prefix_stable_v2.jinja"
PATCHES_V2 = {
    "qwen2.5": "experiments/template_patch/templates/qwen2.5_tool_newline.jinja",
    "qwen3_no_think": _QWEN3_V2,
    "qwen3_think_keep[synthetic]": _QWEN3_V2,
    "qwen3_think_drop[synthetic]": _QWEN3_V2,
}
PATCH_SETS = {"v1": (PATCHES, "+patch"), "v2": (PATCHES_V2, "+patch2")}


def patched_configs(patch_set: str = "v1") -> list[tuple[TokenizerConfig, TokenizerConfig]]:
    """(官方配置, 补丁配置) 对; 补丁配置名加 "+patch"(v1)或 "+patch2"(v2)。"""
    patches, suffix = PATCH_SETS[patch_set]
    pairs = []
    for config in TOKENIZER_CONFIGS:
        if config.name in patches:
            pairs.append((config, replace(config, name=f"{config.name}{suffix}", chat_template=patches[config.name])))
    return pairs


def prompt_shift(
    conversations: list[Conversation], official: Renderer, patched: Renderer, max_turns: int
) -> dict[str, int]:
    """补丁模板与官方模板渲染出的每轮 prompt 是否相同(只比较文本, 历史按"回传思考内容"的配置构造)。"""
    from experiments.template_prefix.run import SYNTHETIC_REASONING

    config = patched.config
    turns = differing = 0
    for conversation in conversations:
        messages = conversation.messages
        if config.synthetic_reasoning and config.keep_reasoning_in_history:
            messages = [
                {**m, "reasoning_content": SYNTHETIC_REASONING} if m["role"] == "assistant" else m for m in messages
            ]
        positions = [i for i, m in enumerate(messages) if m["role"] == "assistant"][:max_turns]
        for position in positions:
            try:
                left = official.render(messages[:position], conversation.tools, generation_prompt=True)
                right = patched.render(messages[:position], conversation.tools, generation_prompt=True)
            except Exception:  # 与 run_config 一致: 模板报错的对话跳过
                break
            turns += 1
            differing += int(left != right)
    return {"turns": turns, "prompt_differs_from_official": differing}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--patch-set", choices=sorted(PATCH_SETS), default="v1")
    parser.add_argument("--out", type=Path, default=None, help="默认 v1 为 results/, v2 为 results_v2/")
    args = parser.parse_args(argv)

    pairs = patched_configs(args.patch_set)
    patches = PATCH_SETS[args.patch_set][0]
    args.out = args.out or (RESULTS_DIR if args.patch_set == "v1" else RESULTS_DIR.with_name("results_v2"))
    renderers = {config.name: Renderer(config) for pair in pairs for config in pair}
    args.out.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, Any]] = []
    shifts: dict[str, dict[str, dict[str, int]]] = {}
    for spec in DATASETS:
        conversations, _ = load_conversations(spec, args.samples, args.seed)
        for official, patched in pairs:
            started = time.perf_counter()
            per_mode = run_config(conversations, renderers[patched.name], args.max_turns)
            shift = prompt_shift(conversations, renderers[official.name], renderers[patched.name], args.max_turns)
            shifts.setdefault(spec.name, {})[patched.name] = shift
            print(f"{spec.name} / {patched.name}: {time.perf_counter() - started:.1f}s shift={shift}", flush=True)
            for mode, (summary, examples) in per_mode.items():
                print(f"  {mode}: {summary}", flush=True)
                cells.append(
                    {
                        "dataset": spec.name,
                        "tokenizer": patched.name,
                        "mode": mode,
                        "summary": summary,
                        "examples": examples,
                    }
                )

    official_cells = json.loads(OFFICIAL_RESULTS.read_text(encoding="utf-8"))["cells"]
    report = {
        "schema_version": 1,
        "settings": {
            "samples_per_dataset": args.samples,
            "seed": args.seed,
            "max_turns": args.max_turns,
            "patch_set": args.patch_set,
            "patches": patches,
        },
        "cells": cells,
        "prompt_shift": shifts,
        "official_reference": [
            c for c in official_cells if c["tokenizer"] in patches and c["mode"] == "B_vllm_token_ids"
        ],
    }
    (args.out / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    (args.out / "summary.md").write_text(markdown_table(cells), encoding="utf-8")
    print(markdown_table(cells))


if __name__ == "__main__":
    main()
