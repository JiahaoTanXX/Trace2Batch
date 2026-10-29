# Copyright (c) 2026 Jiahao Tan

"""TITO 真实 vLLM 客户端的离线测试: 假服务端用一个玩具 Qwen 式模板和分词器代替 vLLM, 不需要 GPU 或网络。

玩具模板复现两类已知失败:
- official: 工具调用前总是插入换行, content 已以换行结尾时叠加成两个换行(template_mismatch);
- 模型采样出非规范切分("o" + "k"), 重渲染再分词后变成 "ok"(retoken_mismatch)。
v2 模板只修复前者; TITO 两者都不受影响。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from experiments.template_prefix.run import Conversation
from experiments.tito_vllm.run import (
    VllmServer,
    observation_ids,
    rollout,
    summarize,
    trajectory_stats,
)

SPECIAL = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
PIECES = [*SPECIAL, "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "ok"]


class ToyTokenizer:
    """特殊 token 和若干多字符片段各占一个 id, 其余逐字符; 编码用贪心最长匹配。"""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {}
        for piece in PIECES:
            self.id(piece)

    def id(self, piece: str) -> int:
        return self.ids.setdefault(piece, len(self.ids) + 1)

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        while text:
            piece = next((p for p in sorted(PIECES, key=len, reverse=True) if text.startswith(p)), text[0])
            out.append(self.id(piece))
            text = text[len(piece) :]
        return out

    def decode(self, ids: Sequence[int]) -> str:
        reverse = {v: k for k, v in self.ids.items()}
        return "".join(reverse[i] for i in ids)


TOK = ToyTokenizer()


def render(messages: list[dict[str, Any]], template: str, generation_prompt: bool) -> str:
    text = ""
    for message in messages:
        role = "user" if message["role"] == "tool" else message["role"]
        body = message.get("content") or ""
        if message["role"] == "tool":
            body = f"<tool_response>{body}</tool_response>"
        for call in message.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            newline = "\n" if template == "official" or not body.endswith("\n") else ""
            payload = json.dumps({"name": call["function"]["name"], "arguments": arguments}, separators=(",", ":"))
            body += f"{newline}<tool_call>{payload}</tool_call>"
        text += f"<|im_start|>{role}\n{body}<|im_end|>\n"
    return text + ("<|im_start|>assistant\n" if generation_prompt else "")


V2_TEMPLATE_TEXT = "{# toy v2 #}"


def fake_post(samples: list[list[str]], finish: list[str] | None = None, calls: list[tuple[str, dict]] | None = None):
    """samples[k] 是第 k 次 /v1/completions 返回的 token 片段(可以是非规范切分)。"""
    finish = finish or ["stop"] * len(samples)
    state = {"next": 0}

    def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if calls is not None:
            calls.append((path, payload))
        if path == "/tokenize":
            if "prompt" in payload:
                return {"tokens": TOK.encode(payload["prompt"])}
            template = "v2" if payload.get("chat_template") == V2_TEMPLATE_TEXT else "official"
            return {"tokens": TOK.encode(render(payload["messages"], template, payload["add_generation_prompt"]))}
        if path == "/v1/completions":
            k = state["next"]
            state["next"] += 1
            ids = [TOK.id(piece) for piece in samples[k]]
            return {"choices": [{"token_ids": ids, "finish_reason": finish[k], "text": TOK.decode(ids)}]}
        raise AssertionError(path)

    return post


CALL = '<tool_call>{"name":"ls","arguments":{"path":"."}}</tool_call>'
CONVERSATION = Conversation(
    "s1",
    [
        {"role": "user", "content": "fix"},
        {"role": "assistant", "content": "look"},
        {"role": "tool", "content": "a.py"},
        {"role": "assistant", "content": "edit"},
        {"role": "tool", "content": "done"},
        {"role": "assistant", "content": "finish"},
    ],
    None,
)


def server(post) -> VllmServer:
    return VllmServer(post, "toy", {}, {"official": None, "v2": V2_TEMPLATE_TEXT}, tito_pipeline="v2")


def pieces(text: str) -> list[str]:
    return [TOK.decode([i]) for i in TOK.encode(text)]


def run(samples: list[list[str]], finish: list[str] | None = None, max_turns: int = 8) -> dict[str, Any]:
    return rollout(
        server(fake_post(samples, finish)),
        CONVERSATION,
        None,
        max_turns=max_turns,
        seed=0,
        sampling={"temperature": 1.0},
        max_tokens=64,
        decode=TOK.decode,
        markers=SPECIAL,
        stop_ids=[TOK.id("<|im_end|>"), TOK.id("<|endoftext|>")],
    )


def test_observation_ids_start_with_the_newline_after_the_previous_im_end() -> None:
    ids = observation_ids(server(fake_post([])), [{"role": "tool", "content": "a.py"}], None)
    expected = "\n<|im_start|>user\n<tool_response>a.py</tool_response><|im_end|>\n<|im_start|>assistant\n"
    assert TOK.decode(ids) == expected


def test_tito_merges_every_turn_while_rerender_fails_on_newline_and_retoken() -> None:
    samples = [
        [*pieces("look.\n" + CALL), "<|im_end|>"],  # 规范输出, content 以换行结尾
        ["o", "k", "<|im_end|>"],  # 非规范切分
        [*pieces("bye"), "<|im_end|>"],
    ]
    result = run(samples)

    assert [t["finish_reason"] for t in result["turns"]] == ["stop"] * 3
    assert result["stop_reason"] == "max_turns_or_end"
    # official: 换行叠加(第 1 轮后)和非规范切分(第 2 轮后); v2 只剩后者。
    assert [t["official_kind"] for t in result["turns"][:2]] == ["template_mismatch", "retoken_mismatch"]
    assert [t["v2_kind"] for t in result["turns"][:2]] == [None, "retoken_mismatch"]
    # TITO 的 prompt 在改写发生之前与 v2 重渲染逐 token 相同。
    assert [t["tito_equals_v2"] for t in result["turns"]] == [True, True, False]

    stats = result["pipelines"]
    assert stats["tito"]["merge_mismatches"] == 0 and stats["tito"]["n_rows"] == 1
    assert stats["official"]["merge_mismatches"] == 2 and stats["official"]["n_rows"] == 3
    assert stats["v2"]["merge_mismatches"] == 1 and stats["v2"]["n_rows"] == 2
    assert stats["official"]["total_tokens"] > stats["v2"]["total_tokens"] > stats["tito"]["total_tokens"]
    # 三条链路的 policy token 相同: 采样完全一样, 只是行的组织方式不同。
    assert len({s["policy_tokens"] for s in stats.values()}) == 1


def test_tito_feeds_its_own_context_back_as_the_prompt() -> None:
    calls: list[tuple[str, dict]] = []
    samples = [["o", "k", "<|im_end|>"], [*pieces("bye"), "<|im_end|>"]]
    rollout(
        server(fake_post(samples, calls=calls)),
        CONVERSATION,
        None,
        max_turns=2,
        seed=5,
        sampling={"temperature": 1.0},
        max_tokens=64,
        decode=TOK.decode,
        markers=SPECIAL,
        stop_ids=[TOK.id("<|im_end|>")],
    )
    prompts = [payload for path, payload in calls if path == "/v1/completions"]
    first, second = prompts[0]["prompt"], prompts[1]["prompt"]
    # 第二轮的 prompt 原样保留模型采样的 "o" + "k", 而不是重新分词成 "ok"。
    assert second[: len(first) + 3] == [*first, TOK.id("o"), TOK.id("k"), TOK.id("<|im_end|>")]
    assert [p["seed"] for p in prompts] == [5, 6]
    assert all(p["return_token_ids"] and p["skip_special_tokens"] is False for p in prompts)


def test_unfinished_turn_ends_the_rollout() -> None:
    result = run([[*pieces("lo")], [*pieces("bye"), "<|im_end|>"]], finish=["length", "stop"])
    assert len(result["turns"]) == 1
    assert result["stop_reason"] == "unfinished"
    assert result["pipelines"]["tito"]["n_rows"] == 1


def test_missing_stop_token_is_appended_and_counted() -> None:
    result = run([[*pieces("look")], [*pieces("bye")], [*pieces("end")]])
    assert [t["stop_token_appended"] for t in result["turns"]] == [True, True, True]
    assert result["pipelines"]["tito"]["merge_mismatches"] == 0


def test_trajectory_stats_counts_rows_and_tokens() -> None:
    events = [
        {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2], "response_token_ids": [3]}},
        {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2, 3, 4], "response_token_ids": [5]}},
    ]
    stats = trajectory_stats(events)
    assert stats["n_rows"] == 1 and stats["merge_mismatches"] == 0
    assert stats["total_tokens"] == 5 and stats["policy_tokens"] == 2


def test_summarize_reports_fully_mergeable_trajectories_and_token_ratio() -> None:
    first = run([[*pieces("look.\n" + CALL), "<|im_end|>"], ["o", "k", "<|im_end|>"], [*pieces("bye"), "<|im_end|>"]])
    second = run([[*pieces("a"), "<|im_end|>"], [*pieces("b"), "<|im_end|>"], [*pieces("c"), "<|im_end|>"]])
    summary = summarize([first, second])
    assert summary["trajectories"] == 2 and summary["turns"] == 6
    assert summary["fully_mergeable"] == {"tito": 2, "official": 1, "v2": 1}
    assert summary["tito_equals_v2_turns"] == 5
    assert summary["token_ratio_vs_tito"]["tito"] == 1.0
    assert summary["token_ratio_vs_tito"]["official"] > summary["token_ratio_vs_tito"]["v2"] > 1.0


def with_chat(post, message: dict[str, Any], same_sample: bool = True):
    """在假服务端上加 /v1/chat/completions: 返回最近一次 completions 的采样(或一个不同的采样)和给定的解析结果。"""
    last: dict[str, Any] = {}

    def wrapped(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path == "/v1/chat/completions":
            ids = last["ids"] if same_sample else [TOK.id("x")]
            return {"choices": [{"token_ids": ids, "message": message, "finish_reason": "stop"}]}
        response = post(path, payload)
        if path == "/v1/completions":
            last["ids"] = response["choices"][0]["token_ids"]
        return response

    return wrapped


def check_first_turn(message: dict[str, Any], same_sample: bool = True) -> dict[str, Any]:
    post = with_chat(fake_post([[*pieces("look.\n" + CALL), "<|im_end|>"]]), message, same_sample)
    result = rollout(
        server(post),
        CONVERSATION,
        None,
        max_turns=1,
        seed=0,
        sampling={},
        max_tokens=64,
        decode=TOK.decode,
        markers=SPECIAL,
        stop_ids=[TOK.id("<|im_end|>")],
        parser_check=True,
    )
    return result["turns"][0]["parser_check"]


def test_parser_check_compares_client_and_server_parsing_on_the_same_sample() -> None:
    call = {"function": {"name": "ls", "arguments": '{"path": "."}'}}
    assert check_first_turn({"content": "look.\n", "tool_calls": [call]}) == {"same_sample": True, "match": True}
    # 服务端把换行去掉时, 客户端解析与之不一致, 必须报告出来。
    assert check_first_turn({"content": "look.", "tool_calls": [call]}) == {"same_sample": True, "match": False}
    assert check_first_turn({"content": "x"}, same_sample=False) == {"same_sample": False, "match": None}


def test_context_length_error_ends_the_rollout() -> None:
    inner = fake_post([[*pieces("a"), "<|im_end|>"]])

    def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path == "/v1/completions" and len(payload["prompt"]) > 30:
            return {"error": {"status": 400, "message": "This model's maximum context length is 32768 tokens."}}
        return inner(path, payload)

    result = rollout(
        server(post),
        CONVERSATION,
        None,
        max_turns=8,
        seed=0,
        sampling={},
        max_tokens=64,
        decode=TOK.decode,
        markers=SPECIAL,
        stop_ids=[TOK.id("<|im_end|>")],
    )
    assert len(result["turns"]) == 1
    assert result["stop_reason"] == "request_error:400:context_length"


def test_retoken_diagnostics_locate_the_split_inside_the_sampled_response() -> None:
    samples = [["o", "k", "<|im_end|>"], [*pieces("bye"), "<|im_end|>"]]
    result = rollout(
        server(fake_post(samples)),
        CONVERSATION,
        None,
        max_turns=2,
        seed=0,
        sampling={},
        max_tokens=64,
        decode=TOK.decode,
        encode=TOK.encode,
        markers=SPECIAL,
        stop_ids=[TOK.id("<|im_end|>")],
    )
    first, second = result["turns"]
    # 模型自己采样出非规范切分; 重渲染的分歧点落在本轮采样的 response 内。
    assert first["self_retoken"] is True and second["self_retoken"] is False
    assert first["v2_kind"] == "retoken_mismatch" and first["v2_in_response"] is True
    assert first["official_in_response"] is True


def test_summarize_counts_self_retoken_turns_and_where_retoken_failures_occur() -> None:
    samples = [["o", "k", "<|im_end|>"], [*pieces("bye"), "<|im_end|>"]]
    result = rollout(
        server(fake_post(samples)),
        CONVERSATION,
        None,
        max_turns=2,
        seed=0,
        sampling={},
        max_tokens=64,
        decode=TOK.decode,
        encode=TOK.encode,
        markers=SPECIAL,
        stop_ids=[TOK.id("<|im_end|>")],
    )
    summary = summarize([result])
    assert summary["self_retoken_turns"] == 1
    assert summary["retoken_in_response"] == {"official": 1, "v2": 1}
