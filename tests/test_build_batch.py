# Copyright (c) 2026 Jiahao Tan

from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.test_trace2batch import SCENARIOS
from trace2batch.core import build_batch, build_samples


def _scenario(scenario_id: str) -> list[dict[str, Any]]:
    return next(item for item in SCENARIOS if item["id"] == scenario_id)["events"]


@pytest.mark.parametrize("mode", ["transition", "trajectory"])
def test_batch_rows_equal_concatenated_single_rollout_rows(mode: str) -> None:
    rollouts = {
        "r-three": _scenario("three_turn_tool"),
        "r-mismatch": _scenario("prefix_mismatch"),
        "r-single": _scenario("single_turn"),
    }
    original = copy.deepcopy(rollouts)
    batch = build_batch(rollouts, mode=mode, max_prompt_length=16, max_response_length=16)
    expected = [
        (rollout_id, row)
        for rollout_id, events in rollouts.items()
        for row in build_samples(events, mode=mode, max_prompt_length=16, max_response_length=16).rows
    ]
    assert [(item.rollout_id, item.row) for item in batch.rows] == expected
    assert rollouts == original


def test_rollouts_are_never_merged_across_ids() -> None:
    # 第二个 rollout 的 prompt 恰好延续第一个 rollout 的上下文, 也不能合并。
    first = [{"event_type": "model_request", "data": {"prompt_token_ids": [1], "response_token_ids": [10]}}]
    second = [{"event_type": "model_request", "data": {"prompt_token_ids": [1, 10, 90], "response_token_ids": [11]}}]
    batch = build_batch({"a": first, "b": second}, mode="trajectory", max_prompt_length=16, max_response_length=16)
    assert [item.rollout_id for item in batch.rows] == ["a", "b"]
    assert batch.summary()["merge_mismatches"] == 0


def test_batch_summary_aggregates_rollout_level_counts() -> None:
    rollouts = {
        "merged": _scenario("three_turn_tool"),
        "unmerged": _scenario("prefix_mismatch"),
        "no-reward": [{"event_type": "model_request", "data": {"prompt_token_ids": [1], "response_token_ids": [10]}}],
        "filtered": [{"event_type": "model_request", "data": {"status": "error"}}],
    }
    batch = build_batch(rollouts, mode="trajectory", max_prompt_length=16, max_response_length=16)
    summary = batch.summary()
    singles = [r.summary() for r in batch.per_rollout.values()]
    assert summary["n_rollouts"] == 4
    assert summary["n_rows"] == sum(s["n_rows"] for s in singles) == len(batch.rows)
    assert summary["policy_tokens"] == sum(s["policy_tokens"] for s in singles)
    assert summary["rollouts_with_reward"] == sum(s["has_reward"] for s in singles)
    assert summary["unmerged_rollouts"] == 1
    assert summary["rollouts_without_usable_rows"] == 1
    assert "has_reward" not in summary
    assert batch.to_dict()["rows"][0]["rollout_id"] == "merged"


def test_batch_passes_length_and_fillna_options() -> None:
    events = [{"event_type": "model_request", "data": {"prompt_token_ids": [1, 2], "response_token_ids": [10, 11, 12]}}]
    batch = build_batch(
        {"r": events},
        mode="transition",
        max_prompt_length=16,
        max_response_length=16,
        max_total_length=4,
        reward_fillna_value=-1.0,
    )
    assert batch.rows[0].row.response_token_ids == [10, 11]
    assert batch.rows[0].row.reward == -1.0


@pytest.mark.parametrize("rollouts", [[[]], {"": []}, {1: []}])
def test_invalid_rollout_container_or_id_is_rejected(rollouts: Any) -> None:
    with pytest.raises(ValueError):
        build_batch(rollouts, mode="transition", max_prompt_length=16, max_response_length=16)
