# Copyright (c) 2026 Jiahao Tan

"""P3-B 环境 rollout 的离线测试: 假 vLLM 服务端(玩具模板与分词器)+ 真实环境, 不需要 GPU。"""

from __future__ import annotations

import json
from typing import Any

from experiments.grpo.env import make_task
from experiments.grpo.grpo import rollout_rows
from experiments.grpo.rollout import rollout_env
from experiments.real_sampling.run import parse_hermes
from experiments.tito_vllm.run import VllmServer
from tests.test_tito_vllm import TOK, V2_TEMPLATE_TEXT, pieces, render

TASK = next(make_task(seed, 1) for seed in range(100) if make_task(seed, 1).kind == "total")
ITEM = next(iter(TASK.order))
# 紧凑 JSON: 玩具模板重新渲染工具调用时也输出紧凑 JSON, 规范输出才能保持前缀。
LOOKUP = f'<tool_call>{{"name":"lookup","arguments":{{"item":"{ITEM}"}}}}</tool_call>'
STOP = [TOK.id("<|im_end|>"), TOK.id("<|endoftext|>")]


def fake_server(samples: list[list[str]], finish: list[str] | None = None, calls: list[tuple[str, dict]] | None = None):
    """同时实现 /tokenize、/v1/completions 和 /v1/chat/completions; 第 k 次生成返回 samples[k]。"""
    finish = finish or ["stop"] * len(samples)
    state = {"next": 0}

    def generate() -> tuple[list[int], str]:
        k = state["next"]
        state["next"] += 1
        return [TOK.id(piece) for piece in samples[k]], finish[k]

    def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if calls is not None:
            calls.append((path, payload))
        template = "v2" if payload.get("chat_template") == V2_TEMPLATE_TEXT else "official"
        if path == "/tokenize":
            return {"tokens": TOK.encode(render(payload["messages"], template, payload["add_generation_prompt"]))}
        ids, reason = generate()
        logprobs = [-0.1 * (i + 1) for i in range(len(ids))]
        if path == "/v1/completions":
            choice = {"token_ids": ids, "finish_reason": reason, "logprobs": {"token_logprobs": logprobs}}
            return {"choices": [choice]}
        assert path == "/v1/chat/completions"
        prompt = TOK.encode(render(payload["messages"], template, True))
        body = ids[:-1] if ids and ids[-1] in STOP else ids
        parsed, has_calls = parse_hermes(TOK.decode(body))
        message: dict[str, Any] = {"role": "assistant", "content": parsed.get("content")}
        if has_calls:
            message["tool_calls"] = [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": c["function"]["name"], "arguments": json.dumps(c["function"]["arguments"])},
                }
                for i, c in enumerate(parsed["tool_calls"])
            ]
            reason = "tool_calls" if reason == "stop" else reason
        content = [{"logprob": value} for value in logprobs]
        choice = {"token_ids": ids, "message": message, "finish_reason": reason, "logprobs": {"content": content}}
        return {"prompt_token_ids": prompt, "choices": [choice]}

    return post


def run(driver: str, samples: list[list[str]], finish: list[str] | None = None, max_turns: int = 6, calls=None):
    server = VllmServer(
        fake_server(samples, finish, calls), "toy", {}, {"official": None, "v2": V2_TEMPLATE_TEXT}, "v2"
    )
    return rollout_env(
        server,
        TASK,
        driver=driver,
        max_turns=max_turns,
        seed=3,
        sampling={"temperature": 1.0},
        max_tokens=64,
        decode=TOK.decode,
        encode=TOK.encode,
        stop_ids=STOP,
        logprobs=True,
    )


SOLVED = [[*pieces(LOOKUP), "<|im_end|>"], [*pieces(f"ANSWER: {TASK.answer}"), "<|im_end|>"]]


def test_tito_rollout_executes_tools_and_scores_the_final_answer() -> None:
    calls: list[tuple[str, dict]] = []
    result = run("tito", SOLVED, calls=calls)
    assert result["reward"] == 1.0 and result["answer"] == TASK.answer
    assert result["stop_reason"] == "answered" and len(result["turns"]) == 2
    assert result["turns"][0]["tool_calls"] == 1
    # 第二轮的 prompt 中有环境返回的真实工具结果(而不是数据集观察)。
    completions = [payload for path, payload in calls if path == "/v1/completions"]
    second_prompt = TOK.decode(completions[1]["prompt"])
    price, stock = TASK.inventory[ITEM]
    assert json.dumps({"item": ITEM, "price": price, "stock": stock}) in second_prompt
    assert [c["seed"] for c in completions] == [3, 4]
    # TITO: 整条轨迹合并为 1 行; 每轮采样 token 的 vLLM logprob 都记录下来。
    assert len(rollout_rows(result["events"], "trajectory")) == 1
    assert [len(t) for t in result["sampled_logprobs"]] == [len(SOLVED[0]), len(SOLVED[1])]


def test_rerender_rollout_uses_server_parsing_and_server_prompts() -> None:
    calls: list[tuple[str, dict]] = []
    result = run("rerender", SOLVED, calls=calls)
    assert result["reward"] == 1.0 and len(result["turns"]) == 2
    chats = [payload for path, payload in calls if path == "/v1/chat/completions"]
    assert len(chats) == 2 and all(c["chat_template"] == V2_TEMPLATE_TEXT for c in chats)
    # 服务端解析出的工具调用(带 id)写回历史, 工具结果按 tool_call_id 对应。
    assert chats[1]["messages"][-1]["role"] == "tool" and chats[1]["messages"][-1]["tool_call_id"] == "c0"
    # 规范输出且 v2 模板下前缀成立, 合并为 1 行。
    assert len(rollout_rows(result["events"], "trajectory")) == 1


def test_non_canonical_split_is_recorded_and_only_rerender_rows_split() -> None:
    # 先调用工具, 再在回答前采样出非规范切分 "o" + "k"; 之后再调用一次工具, 让第二轮也做合并检查。
    samples = [
        [*pieces(LOOKUP), "<|im_end|>"],
        ["o", "k", *pieces(LOOKUP), "<|im_end|>"],
        [*pieces(f"ANSWER: {TASK.answer}"), "<|im_end|>"],
    ]
    tito = run("tito", samples)
    assert [t["self_retoken"] for t in tito["turns"]] == [False, True, False]
    assert tito["reward"] == 1.0 and len(rollout_rows(tito["events"], "trajectory")) == 1
    rerender = run("rerender", samples)
    assert rerender["reward"] == 1.0 and len(rollout_rows(rerender["events"], "trajectory")) == 2


def test_missing_answer_unfinished_turn_and_turn_limit_all_score_zero() -> None:
    no_answer = run("tito", [[*pieces("I am not sure."), "<|im_end|>"]])
    assert no_answer["reward"] == 0.0 and no_answer["stop_reason"] == "answered" and no_answer["answer"] is None
    unfinished = run("tito", [[*pieces("ANSWER: 1")]], finish=["length"])
    assert unfinished["reward"] == 0.0 and unfinished["stop_reason"] == "unfinished"
    looping = run("tito", [[*pieces(LOOKUP), "<|im_end|>"]] * 3, max_turns=3)
    assert looping["reward"] == 0.0 and looping["stop_reason"] == "max_turns" and len(looping["turns"]) == 3


def test_wrong_answer_scores_zero() -> None:
    wrong = run("tito", [[*pieces(LOOKUP), "<|im_end|>"], [*pieces("ANSWER: 0.01"), "<|im_end|>"]])
    assert wrong["reward"] == 0.0 and wrong["answer"] == "0.01"
