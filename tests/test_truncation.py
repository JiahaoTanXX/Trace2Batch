# Copyright (c) 2026 Jiahao Tan

from __future__ import annotations

from typing import Any

import pytest

from tests.test_trace2batch import SCENARIOS
from trace2batch.core import build_batch, build_samples

# three_turn_tool 合并后: response [10, 90, 11, 91, 92, 12, 13], mask [1, 0, 1, 0, 0, 1, 1]
THREE_TURN = next(item for item in SCENARIOS if item["id"] == "three_turn_tool")["events"]


def _build(strategy: str, max_response_length: int, **kwargs: Any):
    return build_samples(
        THREE_TURN,
        mode="trajectory",
        max_prompt_length=64,
        max_response_length=max_response_length,
        truncation_strategy=strategy,
        **kwargs,
    )


def test_default_strategy_is_truncate() -> None:
    assert _build("truncate", 3) == build_samples(
        THREE_TURN, mode="trajectory", max_prompt_length=64, max_response_length=3
    )


def test_truncate_cuts_tail_generation() -> None:
    result = _build("truncate", 3)
    assert [row.response_token_ids for row in result.rows] == [[10, 90, 11]]
    assert [row.response_mask for row in result.rows] == [[1, 0, 1]]
    assert result.summary()["policy_tokens"] == 2
    assert result.diagnostics["budget_splits"] == 0


def test_split_on_budget_starts_new_row_at_turn_boundary() -> None:
    result = _build("split_on_budget", 3)
    assert [row.prompt_token_ids for row in result.rows] == [[1], [1, 10, 90, 11, 91, 92]]
    assert [row.response_token_ids for row in result.rows] == [[10, 90, 11], [12, 13]]
    assert [row.response_mask for row in result.rows] == [[1, 0, 1], [1, 1]]
    assert [row.source_turn_indices for row in result.rows] == [[0, 1], [2]]
    assert result.diagnostics["budget_splits"] == 1
    assert result.diagnostics["truncated_response_rows"] == 0
    assert result.diagnostics["merge_mismatches"] == 0


@pytest.mark.parametrize("budget", range(2, 9))
def test_split_keeps_every_policy_token_when_each_turn_fits(budget: int) -> None:
    # 单轮生成最长 2 个 token, 预算 >= 2 时 split 不丢失任何 policy token。
    result = _build("split_on_budget", budget)
    assert result.summary()["policy_tokens"] == 4
    for row in result.rows:
        assert len(row.response_token_ids) == len(row.response_mask) <= budget
        assert row.response_mask[-1] == 1  # 行尾不会只剩观察 token


def test_split_still_truncates_single_oversized_generation() -> None:
    events = [{"event_type": "model_request", "data": {"prompt_token_ids": [1], "response_token_ids": [10, 11, 12]}}]
    result = build_samples(
        events, mode="trajectory", max_prompt_length=8, max_response_length=2, truncation_strategy="split_on_budget"
    )
    assert [row.response_token_ids for row in result.rows] == [[10, 11]]
    assert result.diagnostics["truncated_response_rows"] == 1


def test_split_respects_total_length_of_longer_new_prompt() -> None:
    # 新行 prompt 长 6, max_total_length=5 时响应上限为 0, 该行没有 policy token 被跳过。
    result = _build("split_on_budget", 16, max_total_length=5)
    assert [row.response_token_ids for row in result.rows] == [[10, 90, 11]]
    assert result.diagnostics["budget_splits"] == 1
    assert result.diagnostics["skipped_zero_policy_rows"] == 1


def test_drop_discards_over_budget_rows() -> None:
    result = _build("drop", 3)
    assert result.rows == []
    assert result.diagnostics["dropped_over_budget_rows"] == 1
    assert result.zero_usable_rows_reason == "all_rows_dropped_over_budget"
    assert _build("drop", 7).summary()["policy_tokens"] == 4


def test_strategies_equal_when_within_budget() -> None:
    results = [_build(strategy, 7) for strategy in ("truncate", "split_on_budget", "drop")]
    assert results[0].rows == results[1].rows == results[2].rows


def test_transition_split_equals_truncate() -> None:
    kwargs: dict[str, Any] = {"mode": "transition", "max_prompt_length": 64, "max_response_length": 1}
    truncated = build_samples(THREE_TURN, **kwargs)
    split = build_samples(THREE_TURN, truncation_strategy="split_on_budget", **kwargs)
    assert truncated.rows == split.rows


def test_build_batch_passes_strategy() -> None:
    batch = build_batch(
        {"r": THREE_TURN},
        mode="trajectory",
        max_prompt_length=64,
        max_response_length=3,
        truncation_strategy="split_on_budget",
    )
    assert batch.summary()["budget_splits"] == 1


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="truncation_strategy"):
        _build("keep_head", 3)
