# Copyright (c) 2026 Jiahao Tan

"""vLLM 真实采样客户端的离线测试: 用假服务端代替 HTTP, 不需要 GPU 或网络。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from experiments.real_sampling.run import Turn
from experiments.vllm_sampling.run import (
    MODELS,
    RL_SAMPLING,
    assistant_for_history,
    derive_tools,
    history_for_api,
    observations_for_history,
    process_turn,
    sample_dataset,
)

VOCAB = {1: "<|im_start|>", 2: "<|im_end|>", 3: "user\n", 4: "assistant\n", 5: "hi", 6: "\n", 7: "ok", 8: "o", 9: "k"}


def decode(ids: Sequence[int]) -> str:
    return "".join(VOCAB[i] for i in ids)


def encode(text: str) -> list[int]:
    # 贪心最长匹配, 足以区分 "ok" 与 "o" + "k"。
    by_text = sorted(VOCAB.items(), key=lambda item: -len(item[1]))
    ids: list[int] = []
    while text:
        token, piece = next((i, p) for i, p in by_text if text.startswith(p))
        ids.append(token)
        text = text[len(piece) :]
    return ids


TURN = Turn("s1", 1, [{"role": "user", "content": "hi"}], [{"role": "user", "content": "hi"}], None)
PROMPT = [1, 3, 5, 2, 6, 1, 4]


def fake_server(response_ids: list[int], next_prompt: list[int]):
    calls: list[dict[str, Any]] = []

    def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        if payload["max_tokens"] == 1:
            return {
                "prompt_token_ids": next_prompt,
                "choices": [{"token_ids": [7], "message": {}, "finish_reason": "length"}],
            }
        return {
            "prompt_token_ids": PROMPT,
            "choices": [
                {"token_ids": response_ids, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
        }

    return post, calls


def _run(response_ids: list[int], next_prompt: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    post, calls = fake_server(response_ids, next_prompt)
    record = process_turn(TURN, 7, post, MODELS["qwen2.5-1.5b"], RL_SAMPLING, 64, decode, encode, ["<|im_end|>"])
    return record, calls


def test_prefix_ok_when_next_prompt_extends_context() -> None:
    record, calls = _run([7, 2], [*PROMPT, 7, 2, 6, 1, 3, 5, 2, 6, 1, 4])
    assert record["prefix_ok"] is True
    assert record["finished"] is True
    assert calls[0]["seed"] == 7 and calls[0]["return_token_ids"] is True
    # 第二次请求把服务端返回的 assistant 消息写回历史。
    assert calls[1]["messages"][1] == {"role": "assistant", "content": "ok"}


def test_non_canonical_split_is_retoken() -> None:
    record, _ = _run([8, 9, 2], [*PROMPT, 7, 2, 6, 1, 3])
    assert record["prefix_ok"] is False
    assert record["kind"] == "retoken_mismatch"
    assert record["self_retoken_mismatch"] is True


def test_request_error_is_reported() -> None:
    def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"error": {"status": 400, "message": "too long"}}

    record = process_turn(TURN, 0, post, MODELS["qwen2.5-1.5b"], RL_SAMPLING, 64, decode, encode, [])
    assert record["error"]["status"] == 400


def test_tool_call_ids_follow_returned_calls() -> None:
    assistant = assistant_for_history(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}}],
        }
    )
    observations = observations_for_history(
        [{"role": "tool", "content": "x"}, {"role": "tool", "content": "y"}, {"role": "user", "content": "z"}],
        assistant,
    )
    assert [o.get("tool_call_id") for o in observations] == ["c1", "call_1", None]


def test_sample_dataset_keeps_candidate_order_and_counts_errors() -> None:
    turns = [Turn(f"s{i}", i, [], [], None) for i in range(10)]

    def worker(turn: Turn, seed: int) -> dict[str, Any]:
        if turn.position % 3 == 0:
            return {"error": {"status": 400}}
        return {"sample_id": turn.sample_id, "seed": seed}

    records, errors, examples, stopped = sample_dataset(turns, target=4, concurrency=3, seed=100, worker=worker)
    assert stopped is False
    assert [r["sample_id"] for r in records] == ["s1", "s2", "s4", "s5"]
    assert [r["seed"] for r in records] == [101, 102, 104, 105]
    assert errors == {"400:other": 2}
    assert examples == ["", ""]


def test_history_for_api_uses_openai_format() -> None:
    history = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "a", "arguments": {"x": 1}}}],
        },
        {"role": "tool", "content": "r"},
    ]
    converted = history_for_api(history)
    call = converted[1]["tool_calls"][0]
    assert call["function"]["arguments"] == '{"x": 1}'
    assert converted[2]["tool_call_id"] == call["id"] == "call_1_0"


def test_tool_calls_finish_reason_counts_as_finished() -> None:
    def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if payload["max_tokens"] == 1:
            return {"prompt_token_ids": [*PROMPT, 7, 2, 6], "choices": [{"token_ids": [7], "message": {}}]}
        message = {"role": "assistant", "content": "ok"}
        return {
            "prompt_token_ids": PROMPT,
            "choices": [{"token_ids": [7, 2], "message": message, "finish_reason": "tool_calls"}],
        }

    record = process_turn(TURN, 0, post, MODELS["qwen2.5-1.5b"], RL_SAMPLING, 64, decode, encode, [])
    assert record["finished"] is True


def test_derive_tools_from_history() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "ls"}}}],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "edit", "arguments": {"path": "a", "new": "b"}}}],
        },
    ]
    tools = derive_tools(messages)
    assert tools is not None
    assert [t["function"]["name"] for t in tools] == ["bash", "edit"]
    assert sorted(tools[1]["function"]["parameters"]["properties"]) == ["new", "path"]
    assert derive_tools([{"role": "user", "content": "hi"}]) is None


def test_deadline_stops_starting_new_batches() -> None:
    turns = [Turn(f"s{i}", i, [], [], None) for i in range(20)]
    ticks = iter(range(100))

    def worker(turn: Turn, seed: int) -> dict[str, Any]:
        return {"sample_id": turn.sample_id}

    # clock 每批前前进 1; deadline=2 时只运行两批(每批 3 个)。
    records, _, _, stopped = sample_dataset(
        turns, target=20, concurrency=3, seed=0, worker=worker, deadline=2, clock=lambda: next(ticks)
    )
    assert [r["sample_id"] for r in records] == ["s0", "s1", "s2", "s3", "s4", "s5"]
    assert stopped is True


def test_patch2_configs_serve_the_v2_templates_on_the_pinned_models() -> None:
    from pathlib import Path

    from experiments.template_patch.run import PATCHES_V2

    root = Path(__file__).resolve().parents[1]
    pairs = {
        "qwen2.5-1.5b+patch2": ("qwen2.5-1.5b", PATCHES_V2["qwen2.5"]),
        "qwen3-0.6b+patch2": ("qwen3-0.6b", PATCHES_V2["qwen3_no_think"]),
    }
    for key, (base_key, template) in pairs.items():
        served, base = MODELS[key], MODELS[base_key]
        assert served.key == key
        assert (served.repo, served.revision) == (base.repo, base.revision)
        assert served.chat_template_kwargs == base.chat_template_kwargs
        assert served.recommended == base.recommended
        assert served.chat_template == template and (root / template).is_file()


def test_qwen3_patch2_requests_disable_thinking() -> None:
    payloads: list[dict[str, Any]] = []

    def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        payloads.append(payload)
        return {"error": {"status": 400, "message": "fixture stops after the first request"}}

    process_turn(TURN, 0, post, MODELS["qwen3-0.6b+patch2"], RL_SAMPLING, 64, decode, encode, [])
    assert payloads[0]["model"] == "qwen3-0.6b+patch2"
    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
