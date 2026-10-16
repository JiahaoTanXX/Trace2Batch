# Copyright (c) 2026 Jiahao Tan

"""trajectory 前缀不匹配的定位与归因; 只用标准库, 解码通过回调注入。

上游 agent-lightning(@69f5a40, rollout_adapter.py 第 602-605 行)记录不匹配时,
template_mismatch / retoken_mismatch 恒为 False, others_mismatch 恒为 True。
这里给出三类原因的判定规则:

- retoken_mismatch: 解码后的文本仍是前缀, 只有 token id 切分不同。
- template_mismatch: 文本不同, 且每个差异片段都满足其一: 与模板标记(思考块、特殊 token、
  工具调用标签)重叠; 只是空白字符增删; 落在工具调用块内且两侧 JSON 语义相同。
- others_mismatch: 其余情况, 例如历史内容本身被改写。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

MismatchKind = Literal["template_mismatch", "retoken_mismatch", "others_mismatch"]

DEFAULT_TEMPLATE_MARKERS: tuple[str, ...] = (
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
)
# 找不到 prev 尾部的结尾锚点时, next 文本在 prev 尾部长度之外额外取的窗口。
_NEXT_WINDOW_SLACK = 256
# 用 prev 尾部最后这些长度的字符在 next 中定位对应区间的结尾。
_ANCHOR_LENGTHS = (32, 16, 8)


@dataclass(frozen=True)
class MismatchReport:
    """一次不匹配的原因、分歧位置与前后文本片段(snippet 为分歧点前后各 context_chars 个字符)。"""

    kind: MismatchKind
    token_index: int
    char_index: int
    prev_snippet: str
    next_snippet: str
    changed_segments: list[tuple[str, str]]


def first_divergence(prev_context_ids: Sequence[int], next_prompt_ids: Sequence[int]) -> int | None:
    """返回 next_prompt_ids 第一次偏离 prev_context_ids 的位置; 完整前缀时返回 None。

    next 比 prev 短且是 prev 的前缀时, 返回 len(next_prompt_ids)。
    """
    for index, (prev_id, next_id) in enumerate(zip(prev_context_ids, next_prompt_ids, strict=False)):
        if prev_id != next_id:
            return index
    if len(next_prompt_ids) < len(prev_context_ids):
        return len(next_prompt_ids)
    return None


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _changed_spans(prev_tail: str, next_window: str) -> list[tuple[int, int, int, int]]:
    """prev 尾部与 next 对应窗口之间的差异区间 (i1, i2, j1, j2)。"""
    matcher = SequenceMatcher(None, prev_tail, next_window, autojunk=False)
    spans: list[tuple[int, int, int, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # prev 尾部被完全覆盖之后的插入是正常延续(观察 / 生成提示), 不算差异。
        if tag == "equal" or (tag == "insert" and i1 == len(prev_tail)):
            continue
        spans.append((i1, i2, j1, j2))
    return spans


def _next_window(prev_tail: str, next_text: str, start: int) -> str:
    """next 中与 prev 尾部对应的区间: 截到 prev 尾部结尾片段第一次出现的位置。

    窗口过长时, 差异对齐可能把被删的模板片段错配到后面的生成提示上。
    结尾片段从长到短尝试, 因为最后几个字符之前也可能被改写。
    """
    for length in _ANCHOR_LENGTHS:
        anchor = prev_tail[-length:]
        position = next_text.find(anchor, start)
        if position != -1:
            return next_text[start : position + len(anchor)]
    return next_text[start : start + len(prev_tail) + _NEXT_WINDOW_SLACK]


def _enclosing_block(text: str, position: int, open_tag: str, close_tag: str) -> str | None:
    """返回包含 position 的 open_tag...close_tag 内部文本; 不在块内时返回 None。"""
    start = text.rfind(open_tag, 0, position + 1)
    if start == -1:
        return None
    end = text.find(close_tag, start)
    if end == -1 or end < position:
        return None
    return text[start + len(open_tag) : end]


def _same_tool_call_json(prev_text: str, prev_pos: int, next_text: str, next_pos: int) -> bool:
    """差异落在工具调用块内, 且两侧块内 JSON 语义相同(只是序列化方式不同)。"""
    prev_block = _enclosing_block(prev_text, prev_pos, "<tool_call>", "</tool_call>")
    next_block = _enclosing_block(next_text, next_pos, "<tool_call>", "</tool_call>")
    if prev_block is None or next_block is None:
        return False
    try:
        return json.loads(prev_block) == json.loads(next_block)
    except ValueError:
        return False


def _segment_is_template(segment: str, full_text: str, offset: int, markers: Sequence[str]) -> bool:
    """片段只含空白, 或与某个模板标记的出现位置重叠。"""
    if not segment.strip():
        return True  # 也覆盖空片段: 纯插入或纯删除的另一侧
    start, end = offset, offset + len(segment)
    for marker in markers:
        search_from = max(0, start - len(marker) + 1)
        position = full_text.find(marker, search_from)
        while position != -1 and position < end:
            if position + len(marker) > start:
                return True
            position = full_text.find(marker, position + 1)
    return False


def explain_mismatch(
    prev_ids: Sequence[int],
    next_ids: Sequence[int],
    decode: Callable[[Sequence[int]], str],
    *,
    markers: Sequence[str] = DEFAULT_TEMPLATE_MARKERS,
    context_chars: int = 80,
) -> MismatchReport | None:
    """定位并归因 next_ids 不以 prev_ids 为前缀的原因; 前缀成立时返回 None。

    decode 必须保留特殊 token(例如 skip_special_tokens=False), 否则无法识别特殊 token 增减。
    """
    token_index = first_divergence(prev_ids, next_ids)
    if token_index is None:
        return None
    prev_text = decode(prev_ids)
    next_text = decode(next_ids)
    char_index = _common_prefix_length(prev_text, next_text)

    if char_index == len(prev_text):
        kind: MismatchKind = "retoken_mismatch"
        segments: list[tuple[str, str]] = []
    else:
        prev_tail = prev_text[char_index:]
        next_window = _next_window(prev_tail, next_text, char_index)
        spans = _changed_spans(prev_tail, next_window)
        segments = [(prev_tail[i1:i2], next_window[j1:j2]) for i1, i2, j1, j2 in spans]
        template = all(
            (
                _segment_is_template(prev_tail[i1:i2], prev_text, char_index + i1, markers)
                and _segment_is_template(next_window[j1:j2], next_text, char_index + j1, markers)
            )
            or _same_tool_call_json(prev_text, char_index + i1, next_text, char_index + j1)
            for i1, i2, j1, j2 in spans
        )
        kind = "template_mismatch" if template and segments else "others_mismatch"

    return MismatchReport(
        kind=kind,
        token_index=token_index,
        char_index=char_index,
        prev_snippet=prev_text[max(0, char_index - context_chars) : char_index + context_chars],
        next_snippet=next_text[max(0, char_index - context_chars) : char_index + context_chars],
        changed_segments=segments,
    )


def classify_mismatch(
    prev_ids: Sequence[int],
    next_ids: Sequence[int],
    decode: Callable[[Sequence[int]], str],
    *,
    markers: Sequence[str] = DEFAULT_TEMPLATE_MARKERS,
) -> MismatchKind:
    """返回不匹配原因; 调用方应只在前缀不成立时调用, 前缀成立会抛出 ValueError。"""
    report = explain_mismatch(prev_ids, next_ids, decode, markers=markers)
    if report is None:
        raise ValueError("next_ids 以 prev_ids 为前缀, 不存在不匹配。")
    return report.kind
