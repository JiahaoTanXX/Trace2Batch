# Copyright (c) Microsoft. All rights reserved.
# Derived from microsoft/agent-lightning@69f5a40 agentlightning/verl/rollout_adapter.py (MIT).
# Modifications Copyright (c) 2026 Jiahao Tan.

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from trace2batch.core import build_samples

SCENARIOS: list[dict[str, Any]] = json.loads(
    (Path(__file__).resolve().parents[1] / "examples/trace2batch/scenarios.json").read_text(encoding="utf-8")
)["scenarios"]


def _events(response: list[int] | None = None) -> list[dict[str, Any]]:
    return [
        {
            "event_type": "model_request",
            "data": {"prompt_token_ids": [1], "response_token_ids": response if response is not None else [10]},
        }
    ]


@pytest.mark.parametrize("mode", ["transition", "trajectory"])
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario["id"])
def test_hand_calculated_scenarios(scenario: dict[str, Any], mode: str) -> None:
    original = copy.deepcopy(scenario["events"])
    result = build_samples(
        scenario["events"],
        mode=mode,
        max_prompt_length=scenario["max_prompt_length"],
        max_response_length=scenario["max_response_length"],
    )
    summary = result.summary()
    for key, expected in scenario["expected"][mode].items():
        assert summary[key] == expected
    assert result.zero_usable_rows_reason == scenario.get("expected_reason", {}).get(mode)
    assert scenario["events"] == original
    assert summary["model_requests"] == sum(
        summary[key]
        for key in ("kept_requests", "skipped_status_error", "skipped_http_error", "skipped_empty_response")
    )
    for row in result.rows:
        assert len(row.response_token_ids) == len(row.response_mask)
        assert set(row.response_mask) <= {0, 1}
        assert sum(row.response_mask) > 0
        assert len(row.prompt_token_ids) <= scenario["max_prompt_length"]
        assert len(row.response_token_ids) <= scenario["max_response_length"]


def test_exact_tool_observation_mask_and_source_turns() -> None:
    scenario = next(item for item in SCENARIOS if item["id"] == "three_turn_tool")
    result = build_samples(scenario["events"], mode="trajectory", max_prompt_length=16, max_response_length=16)
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.prompt_token_ids == [1]
    assert row.response_token_ids == [10, 90, 11, 91, 92, 12, 13]
    assert row.response_mask == [1, 0, 1, 0, 0, 1, 1]
    assert row.source_turn_indices == [0, 1, 2]
    assert row.reward == 1.0


def test_exact_truncation_keeps_mask_aligned() -> None:
    scenario = next(item for item in SCENARIOS if item["id"] == "response_truncated")
    result = build_samples(scenario["events"], mode="trajectory", max_prompt_length=16, max_response_length=3)
    assert result.rows[0].response_token_ids == [10, 11, 90]
    assert result.rows[0].response_mask == [1, 1, 0]


def test_prefix_mismatch_starts_new_group() -> None:
    scenario = next(item for item in SCENARIOS if item["id"] == "prefix_mismatch")
    result = build_samples(scenario["events"], mode="trajectory", max_prompt_length=16, max_response_length=16)
    assert [row.prompt_token_ids for row in result.rows] == [[1], [2, 90]]
    assert [row.response_token_ids for row in result.rows] == [[10, 11], [12]]
    assert [row.source_turn_indices for row in result.rows] == [[0], [1]]


def test_shorter_prefix_is_not_continuous() -> None:
    events = _events([10, 11]) + _events([12])
    result = build_samples(events, mode="trajectory", max_prompt_length=16, max_response_length=16)
    assert len(result.rows) == 2
    assert result.diagnostics["merge_mismatches"] == 1


def test_transition_does_not_include_observations_in_response() -> None:
    scenario = next(item for item in SCENARIOS if item["id"] == "continuous_tool")
    result = build_samples(scenario["events"], mode="transition", max_prompt_length=16, max_response_length=16)
    assert [row.response_token_ids for row in result.rows] == [[10, 11], [12]]
    assert result.rows[1].prompt_token_ids == [1, 10, 11, 90, 91]
    assert result.rows[1].response_mask == [1]


@pytest.mark.parametrize("value,expected", [(0, 0.0), (0.5, 0.5), (None, None), (False, None), ("1", None)])
def test_last_reward_and_zero_vs_missing(value: Any, expected: float | None) -> None:
    events = [
        *_events(),
        {"event_type": "reward", "data": {"value": 1}},
        {"event_type": "reward", "data": {"value": value}},
    ]
    result = build_samples(events, mode="transition", max_prompt_length=16, max_response_length=16)
    assert result.final_reward == expected
    assert result.summary()["has_reward"] == int(expected is not None)
    assert result.rows[0].reward == (expected if expected is not None else 0.0)


def test_prompt_marking_does_not_mean_zero_output_rows() -> None:
    events = [{"event_type": "model_request", "data": {"prompt_token_ids": [1, 2, 3], "response_token_ids": [10]}}]
    result = build_samples(events, mode="transition", max_prompt_length=2, max_response_length=16)
    assert result.rows[0].prompt_token_ids == [1, 2]
    assert result.rows[0].is_drop is True
    assert result.summary()["n_rows"] == 1
    assert result.summary()["usable_rows"] == 0


def test_mutating_output_does_not_mutate_input() -> None:
    events = _events()
    original = copy.deepcopy(events)
    result = build_samples(events, mode="trajectory", max_prompt_length=16, max_response_length=16)
    result.rows[0].response_token_ids.append(99)
    assert events == original


@pytest.mark.parametrize("limit", [-1, True, 1.5])
@pytest.mark.parametrize("key", ["max_prompt_length", "max_response_length"])
def test_invalid_limits_are_rejected(key: str, limit: Any) -> None:
    limits = {"max_prompt_length": 16, "max_response_length": 16, key: limit}
    with pytest.raises(ValueError, match="必须是非负整数"):
        build_samples(_events(), mode="transition", **limits)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        build_samples(_events(), mode="unknown", max_prompt_length=16, max_response_length=16)


@pytest.mark.parametrize("tokens", [[True], [1.5], "10"])
def test_synthetic_token_schema_is_explicit(tokens: Any) -> None:
    events = [{"event_type": "model_request", "data": {"response_token_ids": tokens}}]
    with pytest.raises(ValueError, match="整数列表"):
        build_samples(events, mode="transition", max_prompt_length=16, max_response_length=16)
