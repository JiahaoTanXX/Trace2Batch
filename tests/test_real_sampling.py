# Copyright (c) 2026 Jiahao Tan

"""真实采样实验中 vLLM Hermes 解析逻辑的离线测试(不加载模型)。"""

from __future__ import annotations

from experiments.real_sampling.run import parse_hermes


def test_plain_text_becomes_content() -> None:
    message, parsed = parse_hermes("I will look at the file.")
    assert parsed is False
    assert message == {"role": "assistant", "content": "I will look at the file."}


def test_tool_call_is_parsed_with_leading_content() -> None:
    message, parsed = parse_hermes(
        'Let me check.\n<tool_call>\n{"name": "bash", "arguments": {"cmd": "ls"}}\n</tool_call>'
    )
    assert parsed is True
    assert message["content"] == "Let me check.\n"
    assert message["tool_calls"] == [{"type": "function", "function": {"name": "bash", "arguments": {"cmd": "ls"}}}]


def test_empty_leading_content_is_none() -> None:
    message, _ = parse_hermes('<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>')
    assert message["content"] is None


def test_multiple_and_unterminated_calls() -> None:
    text = '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call><tool_call>{"name": "b", "arguments": {}}'
    message, parsed = parse_hermes(text)
    assert parsed is True
    assert [call["function"]["name"] for call in message["tool_calls"]] == ["a", "b"]


def test_malformed_json_falls_back_to_content() -> None:
    text = '<tool_call>{"name": "a", "arguments": {x: 1}}</tool_call>'
    message, parsed = parse_hermes(text)
    assert parsed is False
    assert message == {"role": "assistant", "content": text}


def test_string_arguments_stay_strings() -> None:
    # 模型把 arguments 写成字符串时, json.dumps 后再 json.loads 仍是字符串, 模板会把它渲染成带引号的 JSON 字符串。
    message, _ = parse_hermes('<tool_call>{"name": "a", "arguments": "{\\"x\\": 1}"}</tool_call>')
    assert message["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
