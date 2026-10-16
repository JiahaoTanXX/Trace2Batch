# Copyright (c) Microsoft. All rights reserved.
# Derived from microsoft/agent-lightning@69f5a40 agentlightning/verl/rollout_adapter.py (MIT).
# Modifications Copyright (c) 2026 Jiahao Tan.

"""构建多轮轨迹样本, 并统计过滤、mask 与长度截断结果。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

TRUNCATION_STRATEGIES = ("truncate", "split_on_budget", "drop")


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
    max_total_length: int | None = None,
    reward_fillna_value: float = 0.0,
    truncation_strategy: str = "truncate",
) -> BuildResult:
    """从单个合成 rollout 的已投影 token 事件构建 transition/trajectory 行。

    文本轨迹处理: 互斥过滤、最后 reward、前缀合并、
    观察 token mask=0、响应截断、过长 prompt 标记。没有 Gateway 投影/去重、
    padding、图像、log_probs、优势/loss、PPO 行筛选或模型参数更新。

    max_total_length 与上游 RolloutAdapter 一致: 响应上限取
    min(max_response_length, max_total_length - 截断后 prompt 长度), 不小于 0。
    reward_fillna_value 只用于缺失奖励行的 reward; final_reward 仍为 None。

    truncation_strategy 决定行的 response 超出上限时的处理:
    - "truncate"(默认, 原有行为): 截掉尾部, mask 同步截断。
    - "split_on_budget": trajectory 合并时, 若并入下一轮会超出上限, 就在轮次边界结束当前行,
      下一轮另起一行, prompt 为到此为止的完整上下文; 单轮生成本身超限时仍截断。
      transition 模式下每行只有一轮, 等同 truncate。
    - "drop": 超出上限的行整行丢弃。
    """
    if mode not in {"transition", "trajectory"}:
        raise ValueError("mode 只支持 transition 或 trajectory。")
    for name, limit in (("max_prompt_length", max_prompt_length), ("max_response_length", max_response_length)):
        if type(limit) is not int or limit < 0:
            raise ValueError(f"{name} 必须是非负整数。")
    if max_total_length is not None and (type(max_total_length) is not int or max_total_length < 0):
        raise ValueError("max_total_length 必须是非负整数或 None。")
    if isinstance(reward_fillna_value, bool) or not isinstance(reward_fillna_value, int | float):
        raise ValueError("reward_fillna_value 必须是数值。")
    if truncation_strategy not in TRUNCATION_STRATEGIES:
        raise ValueError(f"truncation_strategy 只支持 {', '.join(TRUNCATION_STRATEGIES)}。")

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
            "budget_splits",
            "dropped_over_budget_rows",
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

    def response_limit_for(prompt: list[int]) -> int:
        limit = max_response_length
        if max_total_length is not None:
            limit = min(limit, max(max_total_length - len(prompt[:max_prompt_length]), 0))
        return limit

    def emit(prompt: list[int], response: list[int], mask: list[int], indices: list[int]) -> None:
        truncated_prompt = prompt[:max_prompt_length]
        response_limit = response_limit_for(prompt)
        if len(response) > response_limit:
            if truncation_strategy == "drop":
                counters["dropped_over_budget_rows"] += 1
                return
            counters["truncated_response_rows"] += 1
        truncated_response = response[:response_limit]
        truncated_mask = mask[:response_limit]
        if sum(truncated_mask) == 0:
            counters["skipped_zero_policy_rows"] += 1
            return
        rows.append(
            SampleRow(
                prompt_token_ids=truncated_prompt,
                response_token_ids=truncated_response,
                response_mask=truncated_mask,
                source_turn_indices=list(indices),
                reward=final_reward if final_reward is not None else float(reward_fillna_value),
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
            observation = next_prompt[len(context) :]
            prefix_ok = next_prompt[: len(context)] == context
            over_budget = truncation_strategy == "split_on_budget" and len(response) + len(observation) + len(
                next_response
            ) > response_limit_for(prompt)
            if prefix_ok and not over_budget:
                response.extend(observation)
                response.extend(next_response)
                mask.extend([0] * len(observation))
                mask.extend([1] * len(next_response))
                indices.append(index)
            else:
                counters["merge_mismatches" if not prefix_ok else "budget_splits"] += 1
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
        elif not rows and counters["dropped_over_budget_rows"]:
            reason = "all_rows_dropped_over_budget"
        elif not rows:
            reason = "no_policy_tokens_after_truncation"
        else:
            reason = "all_rows_marked_drop"
    return BuildResult(mode, rows, counters, final_reward, reason)


@dataclass(frozen=True)
class BatchRow:
    """带来源 rollout_id 的样本行。"""

    rollout_id: str
    row: SampleRow


@dataclass(frozen=True)
class BatchResult:
    """多个 rollout 的行(按输入顺序)、逐 rollout 结果与汇总诊断。"""

    mode: str
    rows: list[BatchRow]
    per_rollout: dict[str, BuildResult]

    def summary(self) -> dict[str, int]:
        """逐 rollout 的 summary 求和, 再加 rollout 级计数; 对应上游 n_rollouts_w_reward 等指标。"""
        totals: dict[str, int] = {}
        for result in self.per_rollout.values():
            for key, value in result.summary().items():
                totals[key] = totals.get(key, 0) + value
        totals.pop("has_reward", None)
        results = list(self.per_rollout.values())
        return {
            **totals,
            "n_rollouts": len(results),
            "rollouts_with_reward": sum(result.final_reward is not None for result in results),
            "rollouts_without_usable_rows": sum(result.zero_usable_rows_reason is not None for result in results),
            "unmerged_rollouts": sum(
                result.mode == "trajectory" and result.diagnostics["merge_mismatches"] > 0 for result in results
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rows": [{"rollout_id": item.rollout_id, **asdict(item.row)} for item in self.rows],
            "per_rollout": {rollout_id: result.to_dict() for rollout_id, result in self.per_rollout.items()},
            "summary": self.summary(),
        }


def build_batch(
    rollouts: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    mode: str,
    max_prompt_length: int,
    max_response_length: int,
    max_total_length: int | None = None,
    reward_fillna_value: float = 0.0,
    truncation_strategy: str = "truncate",
) -> BatchResult:
    """对多个 rollout 逐个调用 build_samples, 行按输入映射顺序拼接并标注 rollout_id。

    rollout 之间互不合并, 与上游按 rollout 分组聚合一致; 仍不做 padding。
    """
    if not isinstance(rollouts, Mapping):
        raise ValueError("rollouts 必须是 rollout_id 到事件列表的映射。")
    rows: list[BatchRow] = []
    per_rollout: dict[str, BuildResult] = {}
    for rollout_id, events in rollouts.items():
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ValueError("rollout_id 必须是非空字符串。")
        result = build_samples(
            events,
            mode=mode,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            max_total_length=max_total_length,
            reward_fillna_value=reward_fillna_value,
            truncation_strategy=truncation_strategy,
        )
        per_rollout[rollout_id] = result
        rows.extend(BatchRow(rollout_id, row) for row in result.rows)
    return BatchResult(mode, rows, per_rollout)
