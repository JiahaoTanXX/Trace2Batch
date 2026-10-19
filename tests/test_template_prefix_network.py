# Copyright (c) 2026 Jiahao Tan

"""用真实 Qwen tokenizer 复现前缀不匹配; 需要联网与 [real] 依赖, 默认跳过。"""

from __future__ import annotations

import pytest

from trace2batch.core import build_samples

pytestmark = pytest.mark.network

CONVERSATION_MESSAGES = [
    {"role": "user", "content": "List the files in the repository."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}}],
    },
    {"role": "tool", "content": "README.md\nsetup.py"},
    {"role": "assistant", "content": "The repository contains README.md and setup.py."},
]


def _run(config_name: str) -> tuple[dict[str, int], list[dict[str, object]]]:
    pytest.importorskip("transformers")
    from experiments.template_prefix.run import (
        TOKENIZER_CONFIGS,
        Conversation,
        Renderer,
        attribute_mismatches,
        build_turn_events,
    )

    config = next(item for item in TOKENIZER_CONFIGS if item.name == config_name)
    renderer = Renderer(config)
    built = build_turn_events(Conversation("network-fixture", CONVERSATION_MESSAGES, None), renderer, max_turns=10)
    events = built["B_vllm_token_ids"].events
    result = build_samples(events, mode="trajectory", max_prompt_length=10**6, max_response_length=10**6)
    return result.diagnostics, list(attribute_mismatches(events, renderer.decode))


def test_qwen3_no_think_drops_empty_think_block_as_template_mismatch() -> None:
    diagnostics, reports = _run("qwen3_no_think")
    assert diagnostics["merge_mismatches"] == 1
    assert [report["kind"] for report in reports] == ["template_mismatch"]
    assert reports[0]["changed_segments"] == [("<think>\n\n</think>\n\n", "")]


def test_qwen25_same_conversation_merges() -> None:
    diagnostics, reports = _run("qwen2.5")
    assert diagnostics["merge_mismatches"] == 0
    assert reports == []


def test_gpt_oss_final_answer_end_token_changes_in_history() -> None:
    # gpt-oss: 生成以 <|return|> 结束, 进入历史后被渲染为 <|end|>。
    pytest.importorskip("transformers")
    from experiments.template_prefix.run import (
        TOKENIZER_CONFIGS,
        Conversation,
        Renderer,
        attribute_mismatches,
        build_turn_events,
    )

    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3 + 3?"},
        {"role": "assistant", "content": "6"},
    ]
    renderer = Renderer(next(item for item in TOKENIZER_CONFIGS if item.name == "gpt-oss"))
    built = build_turn_events(Conversation("gpt-oss-fixture", messages, None), renderer, max_turns=10)
    reports = list(attribute_mismatches(built["B_vllm_token_ids"].events, renderer.decode, renderer.markers))
    assert [report["kind"] for report in reports] == ["template_mismatch"]
    assert reports[0]["changed_segments"][0][0] == "<|return|>"


@pytest.mark.parametrize(
    ("config_name", "chat_template", "expect_prefix"),
    [
        ("qwen2.5", None, False),
        ("qwen2.5", "experiments/template_patch/templates/qwen2.5_tool_newline.jinja", True),
        ("qwen3_no_think", "experiments/template_patch/templates/qwen3_prefix_stable.jinja", False),
        ("qwen3_no_think", "experiments/template_patch/templates/qwen3_prefix_stable_v2.jinja", True),
    ],
)
def test_parser_kept_newline_before_tool_call(config_name: str, chat_template: str | None, expect_prefix: bool) -> None:
    # 模型按规范格式输出 "正文\n<tool_call>...", Hermes 解析器把 "\n" 留在 content 末尾;
    # 官方模板(及补丁 v1)再插入一个换行导致前缀断开, 补丁 v2 不再插入。
    pytest.importorskip("transformers")
    from dataclasses import replace

    from experiments.real_sampling.run import parse_hermes
    from experiments.template_prefix.run import TOKENIZER_CONFIGS, Renderer

    config = next(item for item in TOKENIZER_CONFIGS if item.name == config_name)
    renderer = Renderer(replace(config, chat_template=chat_template))
    history = [{"role": "user", "content": "List the files."}]
    prompt = renderer.render(history, None, generation_prompt=True)
    output = 'Let me check.\n<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>'
    assistant, parsed = parse_hermes(output)
    assert parsed and assistant["content"] == "Let me check.\n"
    following = renderer.render(
        [*history, assistant, {"role": "tool", "content": "a.py"}], None, generation_prompt=True
    )
    assert following.startswith(prompt + output + "<|im_end|>") is expect_prefix
