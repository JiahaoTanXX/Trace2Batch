# Copyright (c) 2026 Jiahao Tan
# pyright: reportMissingImports=false

"""在规范输出上验证 token-in-token-out 的实现。

对前缀实验的 900 条轨迹, 用 B 方式的规范生成作为 response, 再用 trace2batch.tito 和
qwen_observation_ids 按 TITO 方式构造逐轮事件, 与服务端重渲染(B 方式)的事件逐轮比较:

- 模板前缀稳定时(Qwen2.5 官方模板、Qwen3 补丁模板), TITO 应与重渲染逐 token 相同,
  用来证明观察消息的分词与服务端渲染完全对齐;
- 模板会改写历史时(Qwen3 官方模板), 两者在改写处分歧, 但 TITO 事件依然 100% 合并。

    python -m experiments.tito.run
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments.template_prefix.run import (
    DATASETS,
    TOKENIZER_CONFIGS,
    Conversation,
    Renderer,
    build_turn_events,
    load_conversations,
)
from experiments.tito.encoder import qwen_observation_ids
from trace2batch.core import build_samples
from trace2batch.tito import TokenInTokenOut

RESULTS_DIR = Path(__file__).with_name("results")
QWEN3_PATCH = "experiments/template_patch/templates/qwen3_prefix_stable.jinja"


def configs() -> list:
    by_name = {config.name: config for config in TOKENIZER_CONFIGS}
    return [
        by_name["qwen2.5"],
        by_name["qwen3_no_think"],
        replace(by_name["qwen3_no_think"], name="qwen3_no_think+patch", chat_template=QWEN3_PATCH),
    ]


def tito_events(conversation: Conversation, renderer: Renderer, rerender: list[dict[str, Any]], max_turns: int):
    """用重渲染事件中的首轮 prompt 与规范 response, 按 TITO 方式重建逐轮事件。"""
    messages = conversation.messages
    positions = [i for i, m in enumerate(messages) if m["role"] == "assistant"][:max_turns]
    tito = TokenInTokenOut(rerender[0]["data"]["prompt_token_ids"])
    for turn, position in enumerate(positions):
        tito.record_response(rerender[turn]["data"]["response_token_ids"])
        if turn + 1 < len(positions):
            observations = messages[position + 1 : positions[turn + 1]]
            tito.add_observation(qwen_observation_ids(renderer, observations, conversation.tools))
    return tito.events()


def evaluate(conversations: list[Conversation], renderer: Renderer, max_turns: int) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for conversation in conversations:
        rerender = build_turn_events(conversation, renderer, max_turns)["B_vllm_token_ids"].events
        events = tito_events(conversation, renderer, rerender, max_turns)
        same = [
            a["data"]["prompt_token_ids"] == b["data"]["prompt_token_ids"]
            for a, b in zip(events, rerender, strict=True)
        ]
        tito_result = build_samples(events, mode="trajectory", max_prompt_length=10**9, max_response_length=10**9)
        rerender_result = build_samples(rerender, mode="trajectory", max_prompt_length=10**9, max_response_length=10**9)
        totals["trajectories"] += 1
        totals["turns"] += len(events)
        totals["prompts_identical_to_rerender"] += sum(same)
        totals["trajectories_identical_to_rerender"] += int(all(same))
        totals["merge_attempts"] += max(len(events) - 1, 0)
        totals["tito_merge_mismatches"] += tito_result.diagnostics["merge_mismatches"]
        totals["rerender_merge_mismatches"] += rerender_result.diagnostics["merge_mismatches"]
        totals["tito_policy_tokens"] += tito_result.summary()["policy_tokens"]
        totals["rerender_policy_tokens"] += rerender_result.summary()["policy_tokens"]
    return dict(totals)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    renderers = {config.name: Renderer(config) for config in configs()}
    results: dict[str, dict[str, dict[str, int]]] = {}
    for spec in DATASETS:
        conversations, _ = load_conversations(spec, args.samples, args.seed)
        for name, renderer in renderers.items():
            started = time.perf_counter()
            results.setdefault(spec.name, {})[name] = evaluate(conversations, renderer, args.max_turns)
            print(
                f"{spec.name} / {name}: {results[spec.name][name]} ({time.perf_counter() - started:.0f}s)", flush=True
            )

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "settings": {"samples_per_dataset": args.samples, "seed": args.seed, "max_turns": args.max_turns},
        "results": results,
    }
    (args.out / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "| 数据集 | 模板 | 轮次 | 与重渲染逐 token 相同的 prompt | TITO 合并失败 | 重渲染合并失败 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset, by_config in results.items():
        for name, r in by_config.items():
            lines.append(
                f"| {dataset} | {name} | {r['turns']} | {r['prompts_identical_to_rerender']} "
                f"| {r['tito_merge_mismatches']} | {r['rerender_merge_mismatches']} |"
            )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
