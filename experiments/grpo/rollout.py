# Copyright (c) 2026 Jiahao Tan

"""P3-B 在真实 vLLM 服务端上的多轮环境 rollout, 两种驱动方式与 experiments.tito_vllm 相同。

- tito: /v1/completions 直接接收 TITO 上下文的 token id; 客户端按 Hermes 规则解析工具调用; 工具结果由服务端
  /tokenize 渲染后追加到上下文。训练行为整条轨迹的原始 token。
- rerender: /v1/chat/completions(v2 模板), 服务端解析出的消息写回历史, 每轮重新渲染。训练行取服务端返回的
  prompt token 与采样 token, 前缀断开处拆行。

工具结果来自环境(不是数据集), 因此 rollout 是同策略的。回答、未正常结束和超过轮数都会结束 rollout。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from typing import Any

from experiments.grpo.env import TOOLS, Task, final_answer, initial_messages, reward, tool_response
from experiments.real_sampling.run import parse_hermes
from experiments.tito_vllm.run import VllmServer, _error_reason, observation_ids
from experiments.vllm_sampling.run import assistant_for_history
from trace2batch.tito import TokenInTokenOut


def _sampled_logprobs(choice: dict[str, Any]) -> list[float] | None:
    """两种接口返回 logprob 的格式不同: completions 为 token_logprobs, chat 为 content[i].logprob。"""
    logprobs = choice.get("logprobs") or {}
    if "token_logprobs" in logprobs:
        return [float(v) for v in logprobs["token_logprobs"]]
    if "content" in logprobs:
        return [float(item["logprob"]) for item in logprobs["content"] or []]
    return None


def _server_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """服务端返回的工具调用: arguments 是 JSON 字符串, 解析失败时交给环境报错(不抛异常)。"""
    calls = []
    for call in message.get("tool_calls") or []:
        try:
            arguments = json.loads(call["function"]["arguments"])
        except (TypeError, ValueError):
            arguments = call["function"]["arguments"]
        calls.append({"name": call["function"]["name"], "arguments": arguments})
    return calls


def rollout_env(
    server: VllmServer,
    task: Task,
    *,
    driver: str,
    max_turns: int,
    seed: int,
    sampling: dict[str, float],
    max_tokens: int,
    decode: Callable[[Sequence[int]], str],
    encode: Callable[[str], list[int]],
    stop_ids: Sequence[int],
    logprobs: bool = False,
) -> dict[str, Any]:
    if driver not in ("tito", "rerender"):
        raise ValueError("driver must be tito or rerender")
    started = time.perf_counter()
    messages = initial_messages(task)
    tito = TokenInTokenOut(server.tokenize(messages, TOOLS, server.tito_pipeline)) if driver == "tito" else None
    events: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    sampled_logprobs: list[list[float] | None] = []
    answer: str | None = None
    stop_reason = "max_turns"
    for k in range(max_turns):
        if tito is not None:
            prompt = tito.prompt_ids
            response = server.complete(prompt, seed + k, sampling, max_tokens, stop_ids, logprobs=logprobs)
        else:
            response = server.chat(messages, TOOLS, seed + k, sampling, max_tokens, logprobs=logprobs)
            prompt = []
        if "error" in response:
            stop_reason = _error_reason(response["error"])
            break
        choice = response["choices"][0]
        if tito is None:
            prompt = list(response.get("prompt_token_ids") or choice.get("prompt_token_ids") or [])
        sampled = list(choice.get("token_ids") or [])
        finished = choice.get("finish_reason") in ("stop", "tool_calls")
        appended = finished and (not sampled or sampled[-1] not in stop_ids)
        if appended:
            # 正常结束但返回的 token 不含停止 token 时补上, 否则下一轮会接在未闭合的消息后面。
            sampled.append(stop_ids[0])
        body = sampled[:-1] if finished else sampled
        if tito is not None:
            assistant, _ = parse_hermes(decode(body))
            calls = [dict(call["function"]) for call in assistant.get("tool_calls") or []]
            tito.record_response(sampled)
        else:
            assistant = assistant_for_history(choice["message"])
            calls = _server_calls(choice["message"])
            events.append(
                {"event_type": "model_request", "data": {"prompt_token_ids": prompt, "response_token_ids": sampled}}
            )
        sampled_logprobs.append(_sampled_logprobs(choice))
        turns.append(
            {
                "turn": k,
                "prompt_tokens": len(prompt),
                "sampled_tokens": len(sampled),
                "finish_reason": choice.get("finish_reason"),
                "stop_token_appended": appended,
                "tool_calls": len(calls),
                "self_retoken": encode(decode(body)) != body,
            }
        )
        if not finished:
            stop_reason = "unfinished"
            break
        if not calls:
            answer = final_answer(assistant.get("content"))
            stop_reason = "answered"
            break
        observations = [{"role": "tool", "content": tool_response(task, call)} for call in calls]
        messages = [*messages, assistant, *observations]
        if tito is not None:
            tito.add_observation(observation_ids(server, observations, TOOLS))
    return {
        "task_id": task.task_id,
        "driver": driver,
        "reward": reward(task, answer),
        "answer": answer,
        "stop_reason": stop_reason,
        "turns": turns,
        "events": tito.events() if tito is not None else events,
        "sampled_logprobs": sampled_logprobs,
        "seconds": round(time.perf_counter() - started, 3),
    }
