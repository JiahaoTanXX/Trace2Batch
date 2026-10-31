# Copyright (c) 2026 Jiahao Tan

"""P3-B GRPO 的纯 Python 部分: 组内优势与训练行的构建(不需要 torch)。"""

from __future__ import annotations

from experiments.grpo.grpo import WeightedRow, group_advantages, rollout_rows, weighted_rows

EVENTS = [
    {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2], "response_token_ids": [3, 4]}},
    {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2, 3, 4, 8, 9], "response_token_ids": [5]}},
]


def test_group_advantages_subtract_the_group_mean_without_std_scaling() -> None:
    assert group_advantages([1.0, 0.0, 0.0, 1.0]) == [0.5, -0.5, -0.5, 0.5]
    assert group_advantages([1.0, 0.0, 0.0, 0.0]) == [0.75, -0.25, -0.25, -0.25]
    # 全对或全错的组没有学习信号。
    assert group_advantages([1.0, 1.0]) == [0.0, 0.0]
    assert group_advantages([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_rollout_rows_merge_a_prefix_consistent_trajectory_into_one_row() -> None:
    merged = rollout_rows(EVENTS, "trajectory")
    assert len(merged) == 1
    assert merged[0].prompt_token_ids == [1, 2]
    assert merged[0].response_token_ids == [3, 4, 8, 9, 5]
    assert merged[0].response_mask == [1, 1, 0, 0, 1]
    per_turn = rollout_rows(EVENTS, "transition")
    assert [r.response_token_ids for r in per_turn] == [[3, 4], [5]]


def test_rollout_rows_split_where_the_prefix_breaks() -> None:
    broken = [
        EVENTS[0],
        {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2, 7, 8, 9], "response_token_ids": [5]}},
    ]
    assert len(rollout_rows(broken, "trajectory")) == 2


def test_weighted_rows_give_every_row_of_a_rollout_the_same_advantage() -> None:
    rows = weighted_rows([EVENTS, EVENTS[:1]], [0.5, -0.5], "transition")
    assert [w.advantage for w in rows] == [0.5, 0.5, -0.5]
    assert all(isinstance(w, WeightedRow) for w in rows)
