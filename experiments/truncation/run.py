# Copyright (c) 2026 Jiahao Tan

"""真实轨迹长度分布下三种截断策略(truncate / split_on_budget / drop)的数据层面对比。

另含两组补充实验:
- 预算分配(方案 1): 同一单行总长 T 下, 对比"prompt / response 各 T/2 的固定上限"与
  "prompt + response <= T 的动态分配"(max_total_length)。
- 共享前缀成本(方案 2): 除行 token 总量外, 统计相邻行共享前缀只算一次时的去重 token 数。

复用 template_prefix 实验的同一批样本(3 个数据集各 300 条, 种子 0)和 Qwen2.5 tokenizer 的
B 方式(vLLM token id)事件; 该组合 100% 前缀合并, 所以 trajectory 行长度就是完整轨迹长度。
每组 (max_prompt_length, max_response_length, 策略) 只调用 build_samples, 不涉及训练。

    python -m experiments.truncation.run
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from experiments.template_prefix.run import (
    DATASETS,
    TOKENIZER_CONFIGS,
    Renderer,
    build_turn_events,
    load_conversations,
)
from trace2batch.core import TRUNCATION_STRATEGIES, build_samples, shared_prefix_lengths

RESULTS_DIR = Path(__file__).with_name("results")
TOKENIZER = "qwen2.5"
MODE = "B_vllm_token_ids"
PROMPT_LIMITS = (8192, 32768)
RESPONSE_LIMITS = (4096, 8192, 16384, 32768)
# 方案 1: 单行总长预算 T; fixed = prompt / response 各 T/2, dynamic = 两者上限都是 T 且总长 <= T。
TOTAL_BUDGETS = (8192, 16384, 32768)
ALLOCATIONS = ("fixed_half", "dynamic_total")


@dataclass(frozen=True)
class Trajectory:
    sample_id: str
    events: list[dict[str, Any]]
    response_lengths: list[int]
    observation_lengths: list[int]  # 第 0 轮为 0


def _trajectory(sample_id: str, events: list[dict[str, Any]]) -> Trajectory:
    responses = [len(e["data"]["response_token_ids"]) for e in events]
    observations = [0]
    for previous, current in pairwise(events):
        context = previous["data"]["prompt_token_ids"] + previous["data"]["response_token_ids"]
        prompt = current["data"]["prompt_token_ids"]
        if prompt[: len(context)] != context:
            raise RuntimeError(f"{sample_id}: 截断实验要求事件 100% 前缀合并")
        observations.append(len(prompt) - len(context))
    return Trajectory(sample_id, events, responses, observations)


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def length_stats(trajectories: list[Trajectory]) -> dict[str, Any]:
    """未截断时合并 response 长度(生成 + 观察)与首轮 prompt 长度的分布。"""
    merged = [sum(t.response_lengths) + sum(t.observation_lengths) for t in trajectories]
    first_prompt = [len(t.events[0]["data"]["prompt_token_ids"]) for t in trajectories]
    policy = [sum(t.response_lengths) for t in trajectories]
    return {
        "trajectories": len(trajectories),
        "turns_mean": statistics.mean(len(t.events) for t in trajectories),
        "merged_response_p50": _percentile(merged, 0.5),
        "merged_response_p90": _percentile(merged, 0.9),
        "merged_response_max": max(merged),
        "policy_fraction_of_merged": sum(policy) / sum(merged),
        "first_prompt_p50": _percentile(first_prompt, 0.5),
        "first_prompt_max": max(first_prompt),
    }


def evaluate(
    trajectories: list[Trajectory],
    max_prompt: int,
    max_response: int,
    strategy: str,
    max_total: int | None = None,
) -> dict[str, Any]:
    # 预先放入计数键: 没有任何可用行时(例如 drop)也要输出 0。
    totals: Counter[str] = Counter(dict.fromkeys(("usable_row_tokens", "usable_shared_prefix_tokens"), 0))
    for trajectory in trajectories:
        result = build_samples(
            trajectory.events,
            mode="trajectory",
            max_prompt_length=max_prompt,
            max_response_length=max_response,
            max_total_length=max_total,
            truncation_strategy=strategy,
        )
        summary = result.summary()
        last_turn = len(trajectory.events) - 1
        final_kept = False
        usable = [row for row in result.rows if not row.is_drop]
        # 方案 2: 相邻可用行的共享前缀只算一次。
        totals["usable_shared_prefix_tokens"] += sum(shared_prefix_lengths(usable))
        for row in usable:
            totals["usable_row_tokens"] += len(row.prompt_token_ids) + len(row.response_token_ids)
            if row.response_mask[-1] == 0:
                totals["observation_tail_rows"] += 1
            if row.source_turn_indices[-1] == last_turn:
                first = row.source_turn_indices[0]
                full_length = sum(trajectory.response_lengths[first : last_turn + 1]) + sum(
                    trajectory.observation_lengths[first + 1 : last_turn + 1]
                )
                final_kept = len(row.response_token_ids) == full_length
        totals["trajectories"] += 1
        totals["available_policy_tokens"] += sum(trajectory.response_lengths)
        totals["policy_tokens"] += summary["policy_tokens"]
        totals["usable_policy_tokens"] += summary["usable_policy_tokens"]
        totals["rows"] += summary["n_rows"]
        totals["usable_rows"] += summary["usable_rows"]
        totals["marked_drop_rows"] += summary["marked_drop_rows"]
        totals["truncated_response_rows"] += summary["truncated_response_rows"]
        totals["budget_splits"] += summary["budget_splits"]
        totals["dropped_over_budget_rows"] += summary["dropped_over_budget_rows"]
        totals["final_generation_lost"] += int(not final_kept)
        totals["no_usable_row_trajectories"] += int(summary["usable_rows"] == 0)
    return dict(totals)


def _pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}%" if denominator else "n/a"


def markdown_table(cells: list[dict[str, Any]]) -> str:
    lines = [
        "| 数据集 | max_prompt | max_response | 策略 | 行数/轨迹 | policy 保留率(可用行) | 最终生成丢失 "
        "| 行尾仅观察 | 过长 prompt 行 | 无可用行轨迹 | 行 token 总量 | 去重后 token |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        m = cell["metrics"]
        budget = (
            f"{cell['max_prompt_length']} | {cell['max_response_length']}"
            if "allocation" not in cell
            else f"T={cell['total_budget']} | {cell['allocation']}"
        )
        lines.append(
            f"| {cell['dataset']} | {budget} | {cell['strategy']} "
            f"| {m['rows'] / m['trajectories']:.2f} "
            f"| {_pct(m['usable_policy_tokens'], m['available_policy_tokens'])} "
            f"| {_pct(m['final_generation_lost'], m['trajectories'])} "
            f"| {_pct(m.get('observation_tail_rows', 0), m['usable_rows'])} "
            f"| {m['marked_drop_rows']} "
            f"| {_pct(m['no_usable_row_trajectories'], m['trajectories'])} "
            f"| {m['usable_row_tokens']:,} | {m['usable_row_tokens'] - m['usable_shared_prefix_tokens']:,} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    config = next(item for item in TOKENIZER_CONFIGS if item.name == TOKENIZER)
    renderer = Renderer(config)
    args.out.mkdir(parents=True, exist_ok=True)
    distributions: dict[str, Any] = {}
    cells: list[dict[str, Any]] = []
    allocation_cells: list[dict[str, Any]] = []
    for spec in DATASETS:
        started = time.perf_counter()
        conversations, _ = load_conversations(spec, args.samples, args.seed)
        trajectories = [
            _trajectory(c.sample_id, build_turn_events(c, renderer, args.max_turns)[MODE].events) for c in conversations
        ]
        distributions[spec.name] = length_stats(trajectories)
        print(f"{spec.name}: events built in {time.perf_counter() - started:.1f}s; {distributions[spec.name]}")
        for max_prompt in PROMPT_LIMITS:
            for max_response in RESPONSE_LIMITS:
                for strategy in TRUNCATION_STRATEGIES:
                    cells.append(
                        {
                            "dataset": spec.name,
                            "max_prompt_length": max_prompt,
                            "max_response_length": max_response,
                            "strategy": strategy,
                            "metrics": evaluate(trajectories, max_prompt, max_response, strategy),
                        }
                    )
        for total in TOTAL_BUDGETS:
            for allocation in ALLOCATIONS:
                limits = (total // 2, total // 2, None) if allocation == "fixed_half" else (total, total, total)
                for strategy in TRUNCATION_STRATEGIES:
                    allocation_cells.append(
                        {
                            "dataset": spec.name,
                            "total_budget": total,
                            "allocation": allocation,
                            "max_prompt_length": limits[0],
                            "max_response_length": limits[1],
                            "max_total_length": limits[2],
                            "strategy": strategy,
                            "metrics": evaluate(trajectories, *limits[:2], strategy, max_total=limits[2]),
                        }
                    )

    report = {
        "schema_version": 1,
        "settings": {
            "samples_per_dataset": args.samples,
            "seed": args.seed,
            "max_turns": args.max_turns,
            "tokenizer": config.__dict__,
            "event_mode": MODE,
            "prompt_limits": PROMPT_LIMITS,
            "response_limits": RESPONSE_LIMITS,
            "total_budgets": TOTAL_BUDGETS,
            "allocations": ALLOCATIONS,
        },
        "length_distributions": distributions,
        "cells": cells,
        "allocation_cells": allocation_cells,
    }
    (args.out / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    (args.out / "summary.md").write_text(markdown_table(cells), encoding="utf-8")
    (args.out / "summary_allocation.md").write_text(markdown_table(allocation_cells), encoding="utf-8")
    print(markdown_table(allocation_cells))


if __name__ == "__main__":
    main()
