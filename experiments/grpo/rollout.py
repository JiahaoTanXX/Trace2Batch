# Copyright (c) 2026 Jiahao Tan

"""P3-B 在真实 vLLM 服务端上的多轮环境 rollout, 两种驱动方式与 experiments.tito_vllm 相同。

- tito: /v1/completions 直接接收 TITO 上下文的 token id; 客户端按 Hermes 规则解析工具调用; 工具结果由服务端
  /tokenize 渲染后追加到上下文。训练行为整条轨迹的原始 token。
- rerender: /v1/chat/completions(v2 模板), 服务端解析出的消息写回历史, 每轮重新渲染。训练行取服务端返回的
  prompt token 与采样 token, 前缀断开处拆行。
- textlog(D 组): 采样与 rerender 完全相同; 训练行模拟"只保存文本日志": rollout 结束后用 /tokenize 按 v2 模板
  渲染每条助手消息, 训练 token 是重新分词的结果, 不一定是采样的 token。见 textlog_events。

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


def textlog_events(
    server: VllmServer,
    messages: list[dict[str, Any]],
    assistant_positions: Sequence[int],
    prompts: Sequence[list[int]],
    sampled: Sequence[list[int]],
    finished: Sequence[bool],
    *,
    decode: Callable[[Sequence[int]], str],
    encode: Callable[[str], list[int]],
    stop_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """从文本日志重建训练事件, 并逐轮比较训练 token 与采样 token。

    第 i 轮的 prompt 就是服务端为这一轮渲染的 prompt(与 C 组相同); response 是"到第 i 条助手消息为止"的渲染
    减去 prompt 的部分, 截到 <|im_end|> 为止(其后的换行属于下一条消息)。未正常结束的最后一轮没有完整的消息,
    只能把它的文本重新分词。比较结果: match 相同; retoken 文本相同但切分不同; text 文本不同(例如工具调用 JSON
    被重新格式化); prefix_failed 渲染结果不以 prompt 开头(按文本日志的做法这不应发生)。
    """
    events: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for i, prompt in enumerate(prompts):
        cause = None
        if finished[i]:
            rendered = server.tokenize(messages[: assistant_positions[i] + 1], TOOLS, server.tito_pipeline, False)
            if rendered[: len(prompt)] != prompt or stop_id not in rendered[len(prompt) :]:
                cause = "prefix_failed"
                response = [*encode(decode(sampled[i][:-1])), stop_id]
            else:
                tail = rendered[len(prompt) :]
                response = tail[: tail.index(stop_id) + 1]
        else:
            response = encode(decode(sampled[i]))
        if cause is None:
            if response == sampled[i]:
                cause = "match"
            elif decode(response) == decode(sampled[i]):
                cause = "retoken"
            else:
                cause = "text"
        differing = sum(a != b for a, b in zip(response, sampled[i], strict=False)) + abs(
            len(response) - len(sampled[i])
        )
        checks.append({"cause": cause, "tokens": len(response), "differing_tokens": differing})
        events.append(
            {"event_type": "model_request", "data": {"prompt_token_ids": prompt, "response_token_ids": response}}
        )
    return events, checks


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
    if driver not in ("tito", "rerender", "textlog"):
        raise ValueError("driver must be tito, rerender or textlog")
    started = time.perf_counter()
    messages = initial_messages(task)
    tito = TokenInTokenOut(server.tokenize(messages, TOOLS, server.tito_pipeline)) if driver == "tito" else None
    events: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    sampled_logprobs: list[list[float] | None] = []
    answer: str | None = None
    stop_reason = "max_turns"
    # textlog: 每轮的 prompt、采样 token、是否正常结束, 以及助手消息在 messages 中的位置。
    log_prompts: list[list[int]] = []
    log_sampled: list[list[int]] = []
    log_finished: list[bool] = []
    assistant_positions: list[int] = []
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
            log_prompts.append(prompt)
            log_sampled.append(sampled)
            log_finished.append(finished)
            assistant_positions.append(len(messages))
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
            messages = [*messages, assistant]
            break
        observations = [{"role": "tool", "content": tool_response(task, call)} for call in calls]
        messages = [*messages, assistant, *observations]
        if tito is not None:
            tito.add_observation(observation_ids(server, observations, TOOLS))
    if driver == "textlog" and log_prompts:
        events, checks = textlog_events(
            server,
            messages,
            assistant_positions,
            log_prompts,
            log_sampled,
            log_finished,
            decode=decode,
            encode=encode,
            stop_id=stop_ids[0],
        )
        for turn, check in zip(turns, checks, strict=True):
            turn["textlog"] = check
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
