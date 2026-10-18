# Copyright (c) 2026 Jiahao Tan

from __future__ import annotations

import pytest

from trace2batch.core import build_samples
from trace2batch.tito import TokenInTokenOut


def _three_turns() -> TokenInTokenOut:
    tito = TokenInTokenOut([1])
    tito.record_response([10])
    tito.add_observation([90])
    tito.record_response([11])
    tito.add_observation([91, 92])
    tito.record_response([12, 13])
    return tito


def test_prompts_extend_previous_context() -> None:
    events = _three_turns().events()
    assert [e["data"]["prompt_token_ids"] for e in events] == [[1], [1, 10, 90], [1, 10, 90, 11, 91, 92]]
    assert [e["data"]["response_token_ids"] for e in events] == [[10], [11], [12, 13]]


def test_events_always_merge_into_one_trajectory_row() -> None:
    result = build_samples(_three_turns().events(), mode="trajectory", max_prompt_length=64, max_response_length=64)
    assert result.diagnostics["merge_mismatches"] == 0
    assert len(result.rows) == 1
    assert result.rows[0].response_token_ids == [10, 90, 11, 91, 92, 12, 13]
    assert result.rows[0].response_mask == [1, 0, 1, 0, 0, 1, 1]


def test_non_canonical_response_tokens_are_kept_verbatim() -> None:
    # 模型采样出的切分(例如 "-t" + "xt")原样保留, 不会因为重分词而改变。
    tito = TokenInTokenOut([1, 2])
    tito.record_response([501, 502])
    tito.add_observation([3])
    assert tito.prompt_ids == [1, 2, 501, 502, 3]


def test_prompt_ids_and_events_are_copies() -> None:
    tito = _three_turns()
    tito.prompt_ids.append(999)
    tito.events()[0]["data"]["prompt_token_ids"].append(999)
    assert tito.prompt_ids[-1] == 13
    assert tito.events()[0]["data"]["prompt_token_ids"] == [1]


def test_consecutive_responses_without_observation() -> None:
    tito = TokenInTokenOut([1])
    tito.record_response([10])
    tito.record_response([11])
    assert [e["data"]["prompt_token_ids"] for e in tito.events()] == [[1], [1, 10]]


@pytest.mark.parametrize("bad", ["abc", [1.5], [True]])
def test_non_integer_ids_are_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="整数序列"):
        TokenInTokenOut(bad)  # type: ignore[arg-type]
