# Copyright (c) 2026 Jiahao Tan

"""Qwen 系模板下, token-in-token-out 中"新消息"的分词方式。

TITO 只对新增消息分词, 但新消息的渲染与模板有关(角色标签、工具结果的分组、
下一轮的生成提示)。这里把新消息接在一段占位对话(一条 user + 一条 assistant)之后渲染,
取占位 assistant 的 <|im_end|> 之后的全部文本再分词。这样:
- 新消息之前的换行(属于上一条消息的结尾)也包含在内, 与服务端渲染逐 token 对齐;
- 工具结果是否需要新开 <|im_start|>user 块, 取决于前一条消息的角色, 占位 assistant 与真实情况一致。

只适用于"每条助手消息以 <|im_end|> 结束"的 Qwen 系模板(Qwen2.5、Qwen3 及其补丁)。
"""

from __future__ import annotations

from typing import Any

from experiments.template_prefix.run import Renderer

_PLACEHOLDER: list[dict[str, Any]] = [
    {"role": "user", "content": "."},
    {"role": "assistant", "content": "."},
]


def qwen_observation_ids(renderer: Renderer, observations: list[dict[str, Any]], tools: Any) -> list[int]:
    """新消息(观察 + 下一轮生成提示)的 token, 用于接在 TITO 上下文之后。"""
    text = renderer.render([*_PLACEHOLDER, *observations], tools, generation_prompt=True)
    start = text.index("<|im_start|>assistant")
    end = text.index("<|im_end|>", start) + len("<|im_end|>")
    return renderer.encode(text[end:])
