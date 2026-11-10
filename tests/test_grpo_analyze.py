# Copyright (c) 2026 Jiahao Tan

"""P3-B 汇总脚本的离线测试: 用合成的 steps/evals 检查 bootstrap、斜率和 Q3 判定规则。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.grpo.analyze import bootstrap_gain, slope, summarize


def _step(step: int, rate: float) -> dict:
    return {
        "kind": "step",
        "step": step,
        "rollouts": 4,
        "rows": 4,
        "row_tokens": 100,
        "rollout_seconds": 1.0,
        "step_seconds": 2.0,
        "stop_reasons": {"answered": 4},
        "self_retoken_rate": rate,
        "logprob_gap": {"median": 0.0, "max": 0.1},
        "update": {"processed_tokens": 80, "seconds": 0.5},
        "compare": {
            "cosine": 0.9999,
            "relative_l2": 0.01,
            "transition_seconds": 2.0,
            "trajectory_seconds": 1.0,
            "transition_tokens": 300,
            "trajectory_tokens": 100,
        },
    }


def _write_run(directory: Path, rates: list[float], final: float = 1.0, textlog: bool = False) -> None:
    directory.mkdir(parents=True)
    steps = [_step(i, rate) for i, rate in enumerate(rates)]
    if textlog:
        for step in steps:
            step["textlog"] = {"causes": {"match": 9, "retoken": 1}, "tokens": 100, "differing_tokens": 2}
            step["logprob_gap_mismatched_turns"] = {"n": 1, "median": 0.5, "max": 0.5}
    (directory / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    evals = [
        {"step": 0, "success": 0.0, "success_by_kind": {}, "rewards": [0.0] * 6, "tasks_by_kind": {}},
        {"step": len(rates), "success": final, "success_by_kind": {}, "rewards": [final] * 6, "tasks_by_kind": {}},
    ]
    (directory / "evals.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evals))


def test_bootstrap_is_paired_and_reproducible() -> None:
    result = bootstrap_gain([0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], samples=2000)
    assert result["gain"] == 0.5
    assert 0.0 <= result["ci95"][0] <= 0.5 <= result["ci95"][1] <= 1.0
    assert result == bootstrap_gain([0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], samples=2000)
    with pytest.raises(ValueError):
        bootstrap_gain([0.0], [0.0, 1.0])


def test_slope_is_least_squares() -> None:
    assert slope([0.0, 1.0, 2.0], [1.0, 3.0, 5.0]) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("tito", "rerender", "verdict"),
    [
        ([0.1] * 12, [0.1] * 12, "no reinforcement observed within 60 steps"),
        ([0.01 * i for i in range(12)], [0.1] * 12, "TITO reinforces non-canonical splits during training"),
    ],
)
def test_q3_rule_needs_both_tito_slopes_positive_and_above_rerender(
    tmp_path: Path, tito: list[float], rerender: list[float], verdict: str
) -> None:
    for seed in (0, 1):
        _write_run(tmp_path / f"tito_seed{seed}", tito)
        _write_run(tmp_path / f"rerender_seed{seed}", rerender)
    summary = summarize(tmp_path)
    assert summary["rules"]["q3_verdict"] == verdict
    run = summary["runs"]["tito_seed0"]
    assert run["learning"]["all"]["gain"] == 1.0 and run["cost"]["rows_per_rollout"] == 1.0
    assert run["gradient_compare"][0]["time_ratio"] == 2.0 and run["gradient_compare"][0]["token_ratio"] == 3.0


@pytest.mark.parametrize(
    ("final", "verdict"),
    [
        (0.0, "text-log retokenization hurts learning"),
        (1.0, "no detectable difference within 60 steps"),
    ],
)
def test_arm_d_rules_compare_final_success_with_c_and_split_rate_slopes(
    tmp_path: Path, final: float, verdict: str
) -> None:
    for seed in (0, 1):
        _write_run(tmp_path / f"tito_seed{seed}", [0.1] * 12)
        _write_run(tmp_path / f"rerender_seed{seed}", [0.1] * 12)
        _write_run(tmp_path / f"textlog_seed{seed}", [0.1 - 0.005 * i for i in range(12)], final, textlog=True)
    summary = summarize(tmp_path)
    assert summary["rules"]["q4_verdict"] == verdict
    assert summary["rules"]["q6_verdict"] == "non-canonical splits fall faster under D"
    textlog = summary["runs"]["textlog_seed0"]["textlog"]
    assert textlog["mismatched_turn_fraction"] == pytest.approx(0.1)
    assert textlog["differing_token_fraction"] == pytest.approx(0.02)
    assert textlog["mismatched_turn_logprob_gap"]["turns"] == 12


def test_arm_d_rules_are_skipped_until_both_seeds_exist(tmp_path: Path) -> None:
    for seed in (0, 1):
        _write_run(tmp_path / f"tito_seed{seed}", [0.1] * 12)
        _write_run(tmp_path / f"rerender_seed{seed}", [0.1] * 12)
    _write_run(tmp_path / "textlog_seed0", [0.1] * 12, textlog=True)
    summary = summarize(tmp_path)
    assert "q4_verdict" not in summary["rules"] and "textlog_seed0" not in summary["runs"]
