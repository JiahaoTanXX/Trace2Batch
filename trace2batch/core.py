# Copyright (c) Microsoft. All rights reserved.
# Derived from microsoft/agent-lightning@69f5a40 agentlightning/verl/rollout_adapter.py (MIT).
# Modifications Copyright (c) 2026 Jiahao Tan.

"""构建多轮轨迹样本, 并统计过滤、mask 与长度截断结果。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SampleRow:
    """未填充的实验行; response_mask 只覆盖 response, 不是 attention mask。"""

    prompt_token_ids: list[int]
    response_token_ids: list[int]
    response_mask: list[int]
    source_turn_indices: list[int]
    reward: float
    is_drop: bool


@dataclass(frozen=True)
class BuildResult:
    """参考实验的行、互斥过滤计数及无可用行原因。"""

    mode: str
    rows: list[SampleRow]
    diagnostics: dict[str, int]
    final_reward: float | None
    zero_usable_rows_reason: str | None

    def summary(self) -> dict[str, int]:
        """区分输出行与排除过长 prompt 后的参考可用行; 不代表真实优化量。"""
        usable = [row for row in self.rows if not row.is_drop]
        return {
            **self.diagnostics,
            "n_rows": len(self.rows),
            "marked_drop_rows": len(self.rows) - len(usable),
            "usable_rows": len(usable),
            "policy_tokens": sum(sum(row.response_mask) for row in self.rows),
            "usable_policy_tokens": sum(sum(row.response_mask) for row in usable),
            "observation_tokens": sum(len(row.response_mask) - sum(row.response_mask) for row in self.rows),
            "has_reward": int(self.final_reward is not None),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "summary": self.summary()}


def _token_ids(data: Mapping[str, Any], name: str) -> list[int]:
    value = data.get(name, [])
    if not isinstance(value, list) or any(type(token) is not int for token in value):
        raise ValueError(f"合成事件的 {name} 必须是整数列表。")
    return list(value)


def _request_reason(data: Mapping[str, Any]) -> str:
    """按状态、HTTP 错误和空响应的优先级进行互斥过滤。"""
    if data.get("status") == "error":
        return "status_error"
    status = data.get("http_status")
    if isinstance(status, int) and status >= 400:
        return "http_error"
    if not data.get("response_token_ids"):
        return "empty_response"
    return "kept"


def build_samples(
    events: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    max_prompt_length: int,
    max_response_length: int,
) -> BuildResult:
    """从单个合成 rollout 的已投影 token 事件构建 transition/trajectory 行。

    文本轨迹处理: 互斥过滤、最后 reward、前缀合并、
    观察 token mask=0、响应截断、过长 prompt 标记。没有 Gateway 投影/去重、
    padding、图像、log_probs、优势/loss、PPO 行筛选或模型参数更新。
    """
    if mode not in {"transition", "trajectory"}:
        raise ValueError("mode 只支持 transition 或 trajectory。")
    for name, limit in (("max_prompt_length", max_prompt_length), ("max_response_length", max_response_length)):
        if type(limit) is not int or limit < 0:
            raise ValueError(f"{name} 必须是非负整数。")

    counters: dict[str, int] = dict.fromkeys(
        (
            "model_requests",
            "kept_requests",
            "skipped_status_error",
            "skipped_http_error",
            "skipped_empty_response",
            "truncated_response_rows",
            "skipped_zero_policy_rows",
            "merge_mismatches",
        ),
        0,
    )
    turns: list[tuple[list[int], list[int]]] = []
    final_reward: float | None = None
    for event in events:
        data = event.get("data", {})
        if not isinstance(data, Mapping):
            raise ValueError("合成事件的 data 必须是对象。")
        if event.get("event_type") == "reward":
            value = data.get("value")
            final_reward = float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
        elif event.get("event_type") == "model_request":
            counters["model_requests"] += 1
            reason = _request_reason(data)
            if reason != "kept":
                counters[f"skipped_{reason}"] += 1
                continue
            turns.append((_token_ids(data, "prompt_token_ids"), _token_ids(data, "response_token_ids")))
            counters["kept_requests"] += 1

    rows: list[SampleRow] = []

    def emit(prompt: list[int], response: list[int], mask: list[int], indices: list[int]) -> None:
        if len(response) > max_response_length:
            counters["truncated_response_rows"] += 1
        truncated_response = response[:max_response_length]
        truncated_mask = mask[:max_response_length]
        if sum(truncated_mask) == 0:
            counters["skipped_zero_policy_rows"] += 1
            return
        rows.append(
            SampleRow(
                prompt_token_ids=prompt[:max_prompt_length],
                response_token_ids=truncated_response,
                response_mask=truncated_mask,
                source_turn_indices=list(indices),
                reward=final_reward if final_reward is not None else 0.0,
                is_drop=len(prompt) > max_prompt_length,
            )
        )

    if mode == "transition":
        for index, (prompt, response) in enumerate(turns):
            emit(prompt, response, [1] * len(response), [index])
    elif turns:
        prompt, response = (list(ids) for ids in turns[0])
        context = prompt + response
        mask = [1] * len(response)
        indices = [0]
        for index, (next_prompt, next_response) in enumerate(turns[1:], start=1):
            if next_prompt[: len(context)] == context:
                observation = next_prompt[len(context) :]
                response.extend(observation)
                response.extend(next_response)
                mask.extend([0] * len(observation))
                mask.extend([1] * len(next_response))
                indices.append(index)
            else:
                counters["merge_mismatches"] += 1
                emit(prompt, response, mask, indices)
                prompt, response = list(next_prompt), list(next_response)
                mask = [1] * len(response)
                indices = [index]
            context = next_prompt + next_response
        emit(prompt, response, mask, indices)

    usable_rows = sum(not row.is_drop for row in rows)
    reason = None
    if not usable_rows:
        if not counters["model_requests"]:
            reason = "no_model_requests"
        elif not turns:
            reason = "all_model_requests_filtered"
        elif not rows:
            reason = "no_policy_tokens_after_truncation"
        else:
            reason = "all_rows_marked_drop"
    return BuildResult(mode, rows, counters, final_reward, reason)
