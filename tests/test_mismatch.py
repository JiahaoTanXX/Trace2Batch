# Copyright (c) 2026 Jiahao Tan

from __future__ import annotations

from collections.abc import Sequence

import pytest

from trace2batch.core import build_samples
from trace2batch.mismatch import classify_mismatch, explain_mismatch, first_divergence

# 手工词表: 多个 id 可以解码成相同文本, 用来构造"文本一致、切分不同"的重分词情况。
VOCAB = {
    1: "<|im_start|>",
    2: "<|im_end|>",
    3: "user\n",
    4: "assistant\n",
    5: "<think>\n\n</think>\n\n",
    6: "\n",
    10: "Hello",
    11: " world",
    12: "Hello world",  # = 10 + 11 的另一种切分
    13: "ls",
    14: "<tool_call>",
    15: "</tool_call>",
    16: '{"cmd": "ls"}',
    17: '{"cmd":"ls"}',  # 工具参数 JSON 序列化方式改变
    18: "Goodbye",
    19: "  ",
    20: "<tool_response>out</tool_response>",
    21: '{"cmd": "pwd"}',
    22: "<think>",
    23: "</think>",
    24: "Plan first.",
}


def decode(ids: Sequence[int]) -> str:
    return "".join(VOCAB[token] for token in ids)


USER = [1, 3, 13, 2, 6]


def test_first_divergence_positions() -> None:
    assert first_divergence([1, 2], [1, 2, 3]) is None
    assert first_divergence([1, 2], [1, 2]) is None
    assert first_divergence([1, 2, 3], [1, 9, 3, 4]) == 1
    assert first_divergence([1, 2, 3], [1, 2]) == 2
    assert first_divergence([], [5]) is None


def test_prefix_holds_returns_none_and_classify_rejects_it() -> None:
    prev = [*USER, 1, 4, 10, 2]
    assert explain_mismatch(prev, [*prev, 6, 1, 3], decode) is None
    with pytest.raises(ValueError, match="前缀"):
        classify_mismatch(prev, [*prev, 6], decode)


def test_same_text_different_split_is_retoken() -> None:
    prev = [*USER, 1, 4, 10, 11, 2]
    nxt = [*USER, 1, 4, 12, 2, 6, 1, 3, 13, 2, 6]
    report = explain_mismatch(prev, nxt, decode)
    assert report is not None
    assert report.kind == "retoken_mismatch"
    assert report.token_index == len(USER) + 2
    assert report.changed_segments == []


def test_dropped_empty_think_block_is_template() -> None:
    # Qwen3 enable_thinking=False: 生成提示带空思考块, 下一轮重渲染历史时被删除。
    prev = [*USER, 1, 4, 5, 10, 2]
    nxt = [*USER, 1, 4, 10, 2, 6, 20, 6, 1, 4, 5]
    report = explain_mismatch(prev, nxt, decode)
    assert report is not None
    assert report.kind == "template_mismatch"
    assert report.changed_segments == [("<think>\n\n</think>\n\n", "")]
    assert "<think>" in report.prev_snippet
    assert report.next_snippet.startswith(decode(USER))


def test_added_special_token_is_template() -> None:
    prev = [*USER, 1, 4, 10]  # 生成未以 <|im_end|> 结束
    nxt = [*USER, 1, 4, 2, 10, 2, 6]
    assert classify_mismatch(prev, nxt, decode) == "template_mismatch"


def test_whitespace_only_rewrite_is_template() -> None:
    prev = [*USER, 1, 4, 19, 10, 2]
    nxt = [*USER, 1, 4, 10, 2, 6, 1, 3]
    assert classify_mismatch(prev, nxt, decode) == "template_mismatch"


def test_tool_call_json_reserialization_is_template() -> None:
    # 工具参数被解析后重新序列化: 文本不同但 JSON 语义相同, 属于工具调用格式变化。
    prev = [*USER, 1, 4, 14, 16, 15, 2]
    nxt = [*USER, 1, 4, 14, 17, 15, 2, 6]
    report = explain_mismatch(prev, nxt, decode)
    assert report is not None
    assert report.kind == "template_mismatch"


def test_tool_call_with_different_arguments_is_others() -> None:
    prev = [*USER, 1, 4, 14, 16, 15, 2]
    nxt = [*USER, 1, 4, 14, 21, 15, 2, 6]
    assert classify_mismatch(prev, nxt, decode) == "others_mismatch"


def test_rewritten_history_content_is_others() -> None:
    prev = [*USER, 1, 4, 10, 2]
    nxt = [*USER, 1, 4, 18, 2, 6]
    assert classify_mismatch(prev, nxt, decode) == "others_mismatch"


def test_custom_markers_change_attribution() -> None:
    prev = [*USER, 1, 4, 10, 2]
    nxt = [*USER, 1, 4, 18, 2, 6]
    assert classify_mismatch(prev, nxt, decode, markers=["Hello", "Goodbye"]) == "template_mismatch"


def test_classifier_attributes_build_samples_mismatch() -> None:
    # trajectory 合并失败时, 用上一组上下文与下一轮 prompt 归因。
    first_prompt = [*USER, 1, 4, 5]
    first_response = [10, 2]
    second_prompt = [*USER, 1, 4, 10, 2, 6, 20, 6, 1, 4, 5]
    events = [
        {
            "event_type": "model_request",
            "data": {"prompt_token_ids": first_prompt, "response_token_ids": first_response},
        },
        {"event_type": "model_request", "data": {"prompt_token_ids": second_prompt, "response_token_ids": [18, 2]}},
    ]
    result = build_samples(events, mode="trajectory", max_prompt_length=64, max_response_length=64)
    assert result.diagnostics["merge_mismatches"] == 1
    assert classify_mismatch(first_prompt + first_response, second_prompt, decode) == "template_mismatch"


def test_dropped_think_block_before_tool_call_has_single_segment() -> None:
    # 逐字符对齐会把 "<think>..." 与后面的 "<tool_call>" 错配成多个碎片, 误判为 others。
    prev = [*USER, 1, 4, 5, 14, 16, 15, 2]
    nxt = [*USER, 1, 4, 14, 16, 15, 2, 6, 20, 6, 1, 4, 5, 14]
    report = explain_mismatch(prev, nxt, decode)
    assert report is not None
    assert report.kind == "template_mismatch"
    assert report.changed_segments == [("<think>\n\n</think>\n\n", "")]


def test_emptied_reasoning_between_kept_think_tags_is_template() -> None:
    # GLM-4.5: 历史中保留 <think></think> 标签, 只清空思考内容。
    prev = [*USER, 1, 4, 22, 24, 23, 10, 2]
    nxt = [*USER, 1, 4, 22, 23, 10, 2, 6]
    report = explain_mismatch(prev, nxt, decode)
    assert report is not None
    assert report.kind == "template_mismatch"
    assert report.changed_segments == [("Plan first.", "")]


def test_text_change_outside_think_block_is_still_others() -> None:
    prev = [*USER, 1, 4, 22, 24, 23, 10, 2]
    nxt = [*USER, 1, 4, 22, 24, 23, 18, 2, 6]
    assert classify_mismatch(prev, nxt, decode) == "others_mismatch"
