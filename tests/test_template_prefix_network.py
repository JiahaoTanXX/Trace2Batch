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
