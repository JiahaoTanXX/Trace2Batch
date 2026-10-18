# Copyright (c) 2026 Jiahao Tan

"""token-in-token-out(TITO)轨迹记录器; 只用标准库。

服务端每轮重渲染 chat template 再分词时, 下一轮 prompt 可能不再以上一轮的
prompt + response 开头(模板改写、工具调用重新格式化、非规范切分), trajectory 合并因此失败。
TITO 的做法是: 历史一律沿用已有的 token id, 只对新增消息分词并追加在末尾, 所以每一轮的
prompt 都按构造以上一轮上下文开头, 也与模型在 rollout 中实际看到的 token 完全一致。

新增消息如何分词与模板有关, 由调用方提供(见 experiments/tito)。本模块只负责记账,
产出可直接交给 build_samples 的 model_request 事件。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class TokenInTokenOut:
    """按 TITO 方式维护一条轨迹的 token 上下文。

    用法: 用首轮 prompt 初始化; 每次生成后 record_response; 每次环境返回后 add_observation;
    任何时候 prompt_ids 都是下一次生成应使用的 prompt。
    """

    def __init__(self, initial_prompt_ids: Sequence[int]) -> None:
        self._context = _int_list(initial_prompt_ids, "initial_prompt_ids")
        self._events: list[dict[str, Any]] = []

    @property
    def prompt_ids(self) -> list[int]:
        """下一次生成的 prompt: 迄今为止的完整上下文(副本)。"""
        return list(self._context)

    def record_response(self, response_ids: Sequence[int]) -> None:
        """记录一次生成: 产出一个 model_request 事件, 并把生成的 token 原样接入上下文。"""
        response = _int_list(response_ids, "response_ids")
        self._events.append(
            {
                "event_type": "model_request",
                "data": {"prompt_token_ids": list(self._context), "response_token_ids": response},
            }
        )
        self._context.extend(response)

    def add_observation(self, observation_ids: Sequence[int]) -> None:
        """追加新消息(工具结果、用户消息以及下一轮的生成提示)的 token, 历史保持不变。"""
        self._context.extend(_int_list(observation_ids, "observation_ids"))

    def events(self) -> list[dict[str, Any]]:
        """迄今为止的 model_request 事件(副本), 可直接交给 build_samples。"""
        return [
            {"event_type": e["event_type"], "data": {key: list(ids) for key, ids in e["data"].items()}}
            for e in self._events
        ]


def _int_list(values: Sequence[int], name: str) -> list[int]:
    if isinstance(values, str | bytes) or any(type(value) is not int for value in values):
        raise ValueError(f"{name} 必须是整数序列。")
    return list(values)
