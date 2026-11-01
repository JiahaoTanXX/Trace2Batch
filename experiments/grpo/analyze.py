# Copyright (c) 2026 Jiahao Tan

"""P3-B: 按事先固定的规则汇总 4 次 GRPO 运行 (TITO 和重渲染, 各 2 个种子)。

只读取 results/<driver>_seed<k>/ 下的 steps.jsonl 和 evals.jsonl, 只用标准库:

- 学习效果: 第 60 步相对第 0 步的 held-out 成功率提升, 按评测任务 bootstrap 95% 置信区间, 总体和按答案类型;
- Q2 成本: 训练行数、参与更新的 token、训练更新与 rollout 的耗时;
- Q3: 每轮非规范切分率对步数的斜率; 两个 TITO 种子的斜率都 > 0 且都大于两个重渲染种子时, 判定为"强化";
- Q1: 每次梯度对照中, 逐轮与合并表示的耗时比和 token 比。

    python -m experiments.grpo.analyze --results experiments/grpo/results --out experiments/grpo/results/summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from experiments.grpo.run import eval_tasks

DRIVERS = ("tito", "rerender")
SEEDS = (0, 1)
BOOTSTRAP_SAMPLES = 10_000


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def slope(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum((x - mx) ** 2 for x in xs)


def bootstrap_gain(before: list[float], after: list[float], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = 0) -> dict:
    """配对 bootstrap: 按评测任务重采样, 返回平均提升及其 95% 置信区间。"""
    if len(before) != len(after) or not before:
        raise ValueError("bootstrap needs two nonempty paired lists")
    rng = random.Random(seed)
    n = len(before)
    draws = sorted(sum(after[i] - before[i] for i in (rng.randrange(n) for _ in range(n))) / n for _ in range(samples))
    return {
        "gain": sum(a - b for a, b in zip(after, before, strict=True)) / n,
        "ci95": [draws[int(0.025 * samples)], draws[int(0.975 * samples)]],
    }


def run_summary(directory: Path) -> dict[str, Any]:
    steps = [record for record in _read_jsonl(directory / "steps.jsonl") if record.get("kind") == "step"]
    evals = _read_jsonl(directory / "evals.jsonl")
    first, last = evals[0], evals[-1]
    tasks = eval_tasks(count=len(first["rewards"]), difficulties=[3])
    learning = {"all": bootstrap_gain(first["rewards"], last["rewards"])}
    for kind in ("total", "insufficient"):
        index = [i for i, task in enumerate(tasks) if task.kind == kind]
        learning[kind] = bootstrap_gain([first["rewards"][i] for i in index], [last["rewards"][i] for i in index])
    stop_reasons: dict[str, int] = {}
    for record in steps:
        for reason, count in record["stop_reasons"].items():
            stop_reasons[reason] = stop_reasons.get(reason, 0) + count
    compares = [record for record in steps if record.get("compare") and record["compare"]["cosine"] is not None]
    rates = [record["self_retoken_rate"] for record in steps]
    return {
        "steps": len(steps),
        "rollouts": sum(record["rollouts"] for record in steps),
        "stop_reasons": stop_reasons,
        "evals": [{"step": e["step"], "success": e["success"], "success_by_kind": e["success_by_kind"]} for e in evals],
        "tasks_by_kind": last["tasks_by_kind"],
        "learning": learning,
        "cost": {
            "rows_per_rollout": sum(r["rows"] for r in steps) / sum(r["rollouts"] for r in steps),
            "row_tokens": sum(r["row_tokens"] for r in steps),
            "update_tokens": sum(r["update"]["processed_tokens"] for r in steps),
            "update_seconds": sum(r["update"]["seconds"] for r in steps),
            "rollout_seconds": sum(r["rollout_seconds"] for r in steps),
            "step_seconds": sum(r["step_seconds"] for r in steps),
        },
        "self_retoken": {
            "slope_per_step": slope([float(r["step"]) for r in steps], rates),
            "first10_mean": statistics.fmean(rates[:10]),
            "last10_mean": statistics.fmean(rates[-10:]),
        },
        "gradient_compare": [
            {
                "step": r["step"],
                "cosine": r["compare"]["cosine"],
                "relative_l2": r["compare"]["relative_l2"],
                "time_ratio": r["compare"]["transition_seconds"] / r["compare"]["trajectory_seconds"],
                "token_ratio": r["compare"]["transition_tokens"] / r["compare"]["trajectory_tokens"],
            }
            for r in compares
        ],
        "logprob_gap": {
            "median_of_step_medians": statistics.median(r["logprob_gap"]["median"] for r in steps),
            "max": max(r["logprob_gap"]["max"] for r in steps),
        },
    }


def summarize(results: Path) -> dict[str, Any]:
    runs = {
        f"{driver}_seed{seed}": run_summary(results / f"{driver}_seed{seed}") for driver in DRIVERS for seed in SEEDS
    }
    tito = [runs[f"tito_seed{seed}"]["self_retoken"]["slope_per_step"] for seed in SEEDS]
    rerender = [runs[f"rerender_seed{seed}"]["self_retoken"]["slope_per_step"] for seed in SEEDS]
    reinforced = all(value > 0 for value in tito) and min(tito) > max(rerender)
    learned = {
        name: {kind: run["learning"][kind]["ci95"][0] > 0 for kind in ("all", "total", "insufficient")}
        for name, run in runs.items()
    }
    return {
        "runs": runs,
        "rules": {
            "learning_ci_above_zero": learned,
            "q3_slopes": {"tito": tito, "rerender": rerender},
            "q3_verdict": "TITO reinforces non-canonical splits during training"
            if reinforced
            else "no reinforcement observed within 60 steps",
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    text = json.dumps(summarize(args.results), ensure_ascii=False, indent=1) + "\n"
    if args.out is None:
        print(text, end="")
    else:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
