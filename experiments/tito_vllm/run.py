# Copyright (c) 2026 Jiahao Tan
# pyright: reportMissingImports=false

"""在真实 vLLM 服务端上做多轮 rollout, 配对比较 TITO 与服务端重渲染两条采集链路(GPU 版本)。

experiments/vllm_sampling 只采样单个轮次; 这里从每条轨迹的第一个助手轮次开始连续采样最多
max_turns 轮, 每轮都由模型生成, 观察沿用数据集。rollout 由 TITO 驱动:

1. POST /v1/completions, prompt 直接是 TITO 上下文的 token id, return_token_ids=True;
2. 按 vLLM Hermes 规则把采样文本解析成 assistant 消息, 接上数据集观察;
3. 观察的 token 由服务端 /tokenize 渲染得到(接在上一条 <|im_end|> 之后), 追加到 TITO 上下文;
4. 同一份采样再交给服务端重渲染: 用 /tokenize 分别按官方模板和补丁 v2 渲染下一轮 prompt,
   检查上一轮 "prompt + 采样 token" 是否仍是前缀, 并用归因器分类。

三条链路用的是完全相同的采样 token, 天然配对。每条链路的 model_request 事件交给
build_samples(trajectory), 比较合并失败次数、训练行数与 token 总量。

服务端以官方模板启动(不传 --chat-template), v2 模板随请求的 chat_template 字段发送,
因此需要 --trust-request-chat-template。

--driver rerender 改由服务端重渲染驱动(/v1/chat/completions + v2 模板, 服务端解析出的消息写回历史),
实测重渲染链路的合并结果; 种子规则与 TITO 驱动相同, 用来比较两种驱动方式下非规范切分的出现情况。

    python -m experiments.tito_vllm.run --model qwen3-0.6b --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.real_sampling.run import parse_hermes
from experiments.template_prefix.run import DATASETS, Conversation, load_conversations
from experiments.vllm_sampling.run import (
    MODELS,
    QWEN3_PATCH_V2,
    QWEN25_PATCH_V2,
    RL_SAMPLING,
    Post,
    assistant_for_history,
    derive_tools,
    history_for_api,
    http_post,
)
from trace2batch.core import build_samples
from trace2batch.mismatch import explain_mismatch
from trace2batch.tito import TokenInTokenOut

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).with_name("results")
# 服务端用官方模板; v2 是与 vllm_sampling 中 +patch2 配置相同的模板。
V2_TEMPLATES = {"qwen2.5-1.5b": QWEN25_PATCH_V2, "qwen3-0.6b": QWEN3_PATCH_V2}
PIPELINES = ("official", "v2")
_PLACEHOLDER: list[dict[str, Any]] = [{"role": "user", "content": "."}, {"role": "assistant", "content": "."}]


class VllmServer:
    """一个 vLLM 服务端的三个接口; templates 把链路名映射到模板文本(None 表示服务端默认模板)。"""

    def __init__(
        self,
        post: Post,
        model: str,
        chat_template_kwargs: dict[str, Any],
        templates: dict[str, str | None],
        tito_pipeline: str,
    ) -> None:
        self.post = post
        self.model = model
        self.chat_template_kwargs = chat_template_kwargs
        self.templates = templates
        self.tito_pipeline = tito_pipeline

    def _chat_fields(self, messages: list[dict[str, Any]], tools: Any, pipeline: str) -> dict[str, Any]:
        template = self.templates[pipeline]
        return {
            "model": self.model,
            "messages": history_for_api(messages),
            "chat_template_kwargs": self.chat_template_kwargs,
            **({"tools": tools} if tools else {}),
            **({"chat_template": template} if template is not None else {}),
        }

    def tokenize(
        self, messages: list[dict[str, Any]], tools: Any, pipeline: str, generation_prompt: bool = True
    ) -> list[int]:
        response = self.post(
            "/tokenize", {**self._chat_fields(messages, tools, pipeline), "add_generation_prompt": generation_prompt}
        )
        if "error" in response:
            raise RuntimeError(f"/tokenize failed: {response['error']}")
        return list(response["tokens"])

    def complete(
        self,
        prompt_ids: list[int],
        seed: int,
        sampling: dict[str, float],
        max_tokens: int,
        stop_ids: Sequence[int],
        logprobs: bool = False,
    ) -> dict[str, Any]:
        """logprobs=True 时额外返回每个采样 token 的 logprob(choices[0].logprobs.token_logprobs)。"""
        return self.post(
            "/v1/completions",
            {
                "model": self.model,
                "prompt": prompt_ids,
                "max_tokens": max_tokens,
                "seed": seed,
                **sampling,
                "return_token_ids": True,
                "skip_special_tokens": False,
                "stop_token_ids": list(stop_ids),
                **({"logprobs": 0} if logprobs else {}),
            },
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        seed: int,
        sampling: dict[str, float],
        max_tokens: int,
        logprobs: bool = False,
    ) -> dict[str, Any]:
        """logprobs=True 时额外返回每个采样 token 的 logprob(choices[0].logprobs.content[i].logprob)。"""
        fields = self._chat_fields(messages, tools, self.tito_pipeline)
        return self.post(
            "/v1/chat/completions",
            {
                **fields,
                **sampling,
                "max_tokens": max_tokens,
                "seed": seed,
                "return_token_ids": True,
                **({"logprobs": True, "top_logprobs": 0} if logprobs else {}),
            },
        )


def observation_ids(server: VllmServer, observations: list[dict[str, Any]], tools: Any) -> list[int]:
    """新消息(观察 + 下一轮生成提示)的 token: 接在占位 assistant 的 <|im_end|> 之后的部分。

    占位对话单独渲染(不加生成提示)时以 "<|im_end|>\\n" 结尾; 取它最后一个 token 之前的长度作切点,
    这样上一条消息结尾的换行也归入新消息, 与服务端完整渲染逐 token 对齐(见 experiments.tito.encoder)。
    """
    pipeline = server.tito_pipeline
    base = server.tokenize(_PLACEHOLDER, tools, pipeline, generation_prompt=False)
    full = server.tokenize([*_PLACEHOLDER, *observations], tools, pipeline)
    cut = len(base) - 1
    if full[:cut] != base[:cut]:
        raise RuntimeError("placeholder rendering is not a prefix of the observation rendering")
    return full[cut:]


def trajectory_stats(events: list[dict[str, Any]]) -> dict[str, int]:
    """一条 rollout 按 trajectory 方式构建后的合并失败、行数与 token 总量(不截断)。"""
    result = build_samples(events, mode="trajectory", max_prompt_length=10**9, max_response_length=10**9)
    summary = result.summary()
    return {
        "merge_mismatches": summary["merge_mismatches"],
        "n_rows": summary["n_rows"],
        "policy_tokens": summary["policy_tokens"],
        "total_tokens": sum(len(row.prompt_token_ids) + len(row.response_token_ids) for row in result.rows),
    }


def _error_reason(error: dict[str, Any]) -> str:
    message = str(error.get("message", ""))
    reason = "context_length" if "context length" in message or "maximum" in message else "other"
    return f"request_error:{error.get('status')}:{reason}"


def _check_parser(
    server: VllmServer,
    history: list[dict[str, Any]],
    tools: Any,
    seed: int,
    sampling: dict[str, float],
    max_tokens: int,
    body: list[int],
    parsed: dict[str, Any],
    stop_ids: Sequence[int],
    decode: Callable[[Sequence[int]], str],
) -> dict[str, Any]:
    """第一轮的 prompt 与 /v1/chat/completions 完全相同; 同种子再采样一次, 采样一致时比较两边的解析结果。

    不一致时保存采样文本(前 2,000 个字符)与两边的解析结果, 便于事后归因。
    """
    response = server.chat(history, tools, seed, sampling, max_tokens)
    if "error" in response:
        return {"same_sample": None, "match": None}
    choice = response["choices"][0]
    ids = list(choice.get("token_ids") or [])
    if ids and ids[-1] in stop_ids:
        ids = ids[:-1]
    if ids != body:
        return {"same_sample": False, "match": None}
    message = choice["message"]
    server_calls = [
        {"name": call["function"]["name"], "arguments": json.loads(call["function"]["arguments"])}
        for call in message.get("tool_calls") or []
    ]
    client_calls = [
        {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
        for call in parsed.get("tool_calls") or []
    ]
    match = (message.get("content") or None) == (parsed.get("content") or None) and server_calls == client_calls
    if match:
        return {"same_sample": True, "match": True}
    return {"same_sample": True, "match": False, "text": decode(body)[:2000], "client": parsed, "server": message}


def rollout(
    server: VllmServer,
    conversation: Conversation,
    tools: Any,
    *,
    max_turns: int,
    seed: int,
    sampling: dict[str, float],
    max_tokens: int,
    decode: Callable[[Sequence[int]], str],
    markers: Sequence[str],
    stop_ids: Sequence[int],
    parser_check: bool = False,
    encode: Callable[[str], list[int]] | None = None,
) -> dict[str, Any]:
    """从第一个助手轮次开始, 由 TITO 驱动连续采样; 同时记录两条重渲染链路在同一份采样上的结果。"""
    started = time.perf_counter()
    messages = conversation.messages
    positions = [i for i, m in enumerate(messages) if m["role"] == "assistant"][:max_turns]
    history = list(messages[: positions[0]])
    tito = TokenInTokenOut(server.tokenize(history, tools, server.tito_pipeline))
    rerender = {name: server.tokenize(history, tools, name) for name in PIPELINES}
    events: dict[str, list[dict[str, Any]]] = {name: [] for name in PIPELINES}
    turns: list[dict[str, Any]] = []
    stop_reason = "max_turns_or_end"
    for k, position in enumerate(positions):
        turn_started = time.perf_counter()
        prompt = tito.prompt_ids
        response = server.complete(prompt, seed + k, sampling, max_tokens, stop_ids)
        if "error" in response:
            stop_reason = _error_reason(response["error"])
            break
        choice = response["choices"][0]
        sampled = list(choice.get("token_ids") or [])
        finished = choice.get("finish_reason") == "stop"
        appended = finished and (not sampled or sampled[-1] not in stop_ids)
        if appended:
            # 正常结束但返回的 token 不含停止 token 时补上, 否则下一轮会接在未闭合的消息后面。
            sampled.append(stop_ids[0])
        body = sampled[:-1] if finished else sampled
        parsed, _ = parse_hermes(decode(body))
        tito.record_response(sampled)
        for name in PIPELINES:
            events[name].append(
                {
                    "event_type": "model_request",
                    "data": {"prompt_token_ids": rerender[name], "response_token_ids": sampled},
                }
            )
        divergence = next((i for i, (a, b) in enumerate(zip(prompt, rerender["v2"], strict=False)) if a != b), None)
        record: dict[str, Any] = {
            "turn": k,
            "prompt_tokens": len(prompt),
            "sampled_tokens": len(sampled),
            "finish_reason": choice.get("finish_reason"),
            "stop_token_appended": appended,
            "ends_with_endoftext": finished and decode(sampled[-1:]) == "<|endoftext|>",
            "tool_calls": len(parsed.get("tool_calls") or []),
            # 采样本身是否为非规范切分(重新分词后不同); 没有本地分词器时记为 None。
            "self_retoken": None if encode is None else encode(decode(body)) != body,
            "tito_equals_v2": prompt == rerender["v2"],
            "tito_v2_divergence": divergence,
            "checked": False,
            **{f"{name}_kind": None for name in PIPELINES},
        }
        if parser_check and k == 0:
            record["parser_check"] = _check_parser(
                server, history, tools, seed, sampling, max_tokens, body, parsed, stop_ids, decode
            )
        turns.append(record)
        if not finished:
            stop_reason = "unfinished"
            break
        if k + 1 >= len(positions):
            break
        observations = messages[position + 1 : positions[k + 1]]
        history = [*history, parsed, *observations]
        tito.add_observation(observation_ids(server, observations, tools))
        record["checked"] = True
        for name in PIPELINES:
            next_prompt = server.tokenize(history, tools, name)
            report = explain_mismatch(rerender[name] + sampled, next_prompt, decode, markers=markers)
            record[f"{name}_kind"] = None if report is None else report.kind
            record[f"{name}_segments"] = [] if report is None else report.changed_segments[:4]
            # 分歧点落在本轮采样的 response 内(而不是更早的 prompt 部分)。
            record[f"{name}_in_response"] = None if report is None else report.token_index >= len(rerender[name])
            rerender[name] = next_prompt
        record["seconds"] = round(time.perf_counter() - turn_started, 3)
    pipelines = {"tito": trajectory_stats(tito.events())}
    pipelines.update({name: trajectory_stats(events[name]) for name in PIPELINES})
    return {
        "sample_id": conversation.sample_id,
        "stop_reason": stop_reason,
        "turns": turns,
        "pipelines": pipelines,
        "seconds": round(time.perf_counter() - started, 3),
    }


def summarize(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [t for r in rollouts for t in r["turns"]]
    checked = [t for t in turns if t["checked"]]
    names = ("tito", *PIPELINES)
    totals = {name: sum(r["pipelines"][name]["total_tokens"] for r in rollouts) for name in names}
    return {
        "trajectories": len(rollouts),
        "turns": len(turns),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rollouts)),
        "with_tool_calls": sum(t["tool_calls"] > 0 for t in turns),
        "stop_token_appended": sum(t["stop_token_appended"] for t in turns),
        "ends_with_endoftext": sum(t["ends_with_endoftext"] for t in turns),
        "prompt_tokens_max": max((t["prompt_tokens"] for t in turns), default=0),
        "tito_equals_v2_turns": sum(t["tito_equals_v2"] for t in turns),
        "self_retoken_turns": sum(t.get("self_retoken") is True for t in turns),
        "retoken_in_response": {
            name: sum(t[f"{name}_kind"] == "retoken_mismatch" and t.get(f"{name}_in_response") is True for t in checked)
            for name in PIPELINES
        },
        "merge_checks": len(checked),
        "turn_kinds": {name: dict(Counter(t[f"{name}_kind"] or "prefix_ok" for t in checked)) for name in PIPELINES},
        "fully_mergeable": {
            name: sum(r["pipelines"][name]["merge_mismatches"] == 0 for r in rollouts) for name in names
        },
        "rows": {name: sum(r["pipelines"][name]["n_rows"] for r in rollouts) for name in names},
        "total_tokens": totals,
        "policy_tokens": sum(r["pipelines"]["tito"]["policy_tokens"] for r in rollouts),
        "token_ratio_vs_tito": {
            name: round(totals[name] / totals["tito"], 4) if totals["tito"] else None for name in names
        },
        "parser_check": _parser_outcomes(turns),
    }


def _parser_outcomes(turns: list[dict[str, Any]]) -> dict[str, int]:
    checks = [t["parser_check"] for t in turns if "parser_check" in t]
    return dict(Counter(f"same_sample={c['same_sample']},match={c['match']}" for c in checks))


def rollout_rerender(
    server: VllmServer,
    conversation: Conversation,
    tools: Any,
    *,
    max_turns: int,
    seed: int,
    sampling: dict[str, float],
    max_tokens: int,
    decode: Callable[[Sequence[int]], str],
    markers: Sequence[str],
    stop_ids: Sequence[int],
    encode: Callable[[str], list[int]] | None = None,
) -> dict[str, Any]:
    """由服务端重渲染驱动的 rollout: 每轮请求 /v1/chat/completions(v2 模板), 把服务端解析出的消息写回历史。

    第 k 轮的合并检查要等第 k+1 轮的请求返回服务端渲染的 prompt 后才能做; 最后一轮和下一次请求失败的
    轮次不做检查(checked=False)。
    """
    started = time.perf_counter()
    messages = conversation.messages
    positions = [i for i, m in enumerate(messages) if m["role"] == "assistant"][:max_turns]
    history = list(messages[: positions[0]])
    events: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    previous: tuple[dict[str, Any], list[int]] | None = None  # (上一轮的记录, 上一轮的 prompt + 采样)
    stop_reason = "max_turns_or_end"
    for k, position in enumerate(positions):
        response = server.chat(history, tools, seed + k, sampling, max_tokens)
        if "error" in response:
            stop_reason = _error_reason(response["error"])
            break
        choice = response["choices"][0]
        prompt = list(response.get("prompt_token_ids") or choice.get("prompt_token_ids") or [])
        if previous is not None:
            record, context = previous
            report = explain_mismatch(context, prompt, decode, markers=markers)
            record["checked"] = True
            record["kind"] = None if report is None else report.kind
            record["segments"] = [] if report is None else report.changed_segments[:4]
            record["in_response"] = None if report is None else report.token_index >= record["prompt_tokens"]
        sampled = list(choice.get("token_ids") or [])
        finished = choice.get("finish_reason") in ("stop", "tool_calls")
        appended = finished and (not sampled or sampled[-1] not in stop_ids)
        if appended:
            sampled.append(stop_ids[0])
        body = sampled[:-1] if finished else sampled
        assistant = assistant_for_history(choice["message"])
        events.append(
            {"event_type": "model_request", "data": {"prompt_token_ids": prompt, "response_token_ids": sampled}}
        )
        record = {
            "turn": k,
            "prompt_tokens": len(prompt),
            "sampled_tokens": len(sampled),
            "finish_reason": choice.get("finish_reason"),
            "stop_token_appended": appended,
            "tool_calls": len(assistant.get("tool_calls") or []),
            "self_retoken": None if encode is None else encode(decode(body)) != body,
            "checked": False,
            "kind": None,
        }
        turns.append(record)
        if not finished:
            stop_reason = "unfinished"
            break
        if k + 1 >= len(positions):
            break
        history = [*history, assistant, *messages[position + 1 : positions[k + 1]]]
        previous = (record, prompt + sampled)
    return {
        "sample_id": conversation.sample_id,
        "driver": "rerender",
        "stop_reason": stop_reason,
        "turns": turns,
        "pipelines": {"rerender": trajectory_stats(events)},
        "seconds": round(time.perf_counter() - started, 3),
    }


def summarize_rerender(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [t for r in rollouts for t in r["turns"]]
    checked = [t for t in turns if t["checked"]]
    stats = [r["pipelines"]["rerender"] for r in rollouts]
    return {
        "trajectories": len(rollouts),
        "turns": len(turns),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rollouts)),
        "with_tool_calls": sum(t["tool_calls"] > 0 for t in turns),
        "stop_token_appended": sum(t["stop_token_appended"] for t in turns),
        "prompt_tokens_max": max((t["prompt_tokens"] for t in turns), default=0),
        "self_retoken_turns": sum(t.get("self_retoken") is True for t in turns),
        "merge_checks": len(checked),
        "turn_kinds": dict(Counter(t["kind"] or "prefix_ok" for t in checked)),
        "retoken_in_response": sum(t["kind"] == "retoken_mismatch" and t.get("in_response") is True for t in checked),
        "fully_mergeable": sum(s["merge_mismatches"] == 0 for s in stats),
        "rows": sum(s["n_rows"] for s in stats),
        "total_tokens": sum(s["total_tokens"] for s in stats),
        "policy_tokens": sum(s["policy_tokens"] for s in stats),
    }


def dataset_deadlines(remaining_seconds: float, names: Sequence[str], weights: dict[str, float] | None) -> float:
    """names[0] 这个数据集可用的秒数: 剩余时间按尚未运行的数据集的权重分配, 没有权重时平均分配。"""
    if not weights:
        return remaining_seconds / len(names)
    return remaining_seconds * weights[names[0]] / sum(weights[name] for name in names)


def run_dataset(
    conversations: list[Conversation],
    worker: Callable[[Conversation, int], dict[str, Any]],
    concurrency: int,
    deadline: float | None,
    first_index: int = 0,
) -> tuple[list[dict[str, Any]], bool]:
    """按固定顺序分批并发 rollout; deadline 到达后不再开始新的批次。返回 (结果, 是否因时间上限提前停止)。

    worker 收到的是轨迹在完整选择顺序中的序号(first_index 起), 补跑时种子与原运行一致。
    """
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for start in range(0, len(conversations), concurrency):
            if deadline is not None and time.monotonic() >= deadline:
                return results, True
            batch = conversations[start : start + concurrency]
            results += pool.map(worker, batch, range(first_index + start, first_index + start + len(batch)))
    return results, False


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(V2_TEMPLATES), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--trajectories-per-dataset", type=int, default=40)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--parser-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--time-budget-seconds", type=float, default=None)
    parser.add_argument("--driver", choices=("tito", "rerender"), default="tito")
    parser.add_argument("--datasets", nargs="+", choices=[spec.name for spec in DATASETS], default=None)
    parser.add_argument(
        "--start-index", type=int, default=0, help="从完整选择顺序中的第几条轨迹开始(补跑时使用), 种子规则不变"
    )
    parser.add_argument(
        "--dataset-weights",
        type=lambda text: {k: float(v) for k, v in (item.split("=") for item in text.split(","))},
        default=None,
        help="按数据集分配时间预算, 例如 swe_smith_tool=0.62,nebius_swe_agent=0.25,toucan_qwen3=0.13",
    )
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from experiments.template_prefix.run import CACHE_DIR, DEFAULT_TEMPLATE_MARKERS

    model = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model.repo, revision=model.revision, cache_dir=CACHE_DIR)
    markers = tuple(
        dict.fromkeys([*DEFAULT_TEMPLATE_MARKERS, *(t.content for t in tokenizer.added_tokens_decoder.values())])
    )
    stop_ids = [tokenizer.convert_tokens_to_ids(token) for token in ("<|im_end|>", "<|endoftext|>")]

    def decode(ids: Sequence[int]) -> str:
        return tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    v2_text = (ROOT / V2_TEMPLATES[args.model]).read_text(encoding="utf-8")
    server = VllmServer(
        http_post(args.base_url), model.key, model.chat_template_kwargs, {"official": None, "v2": v2_text}, "v2"
    )
    out: Path = args.out or RESULTS_DIR / f"{args.model}@{args.driver}"
    out.mkdir(parents=True, exist_ok=True)
    specs = [spec for spec in DATASETS if args.datasets is None or spec.name in args.datasets]
    summaries: dict[str, Any] = {}
    all_rollouts: dict[str, list[dict[str, Any]]] = {}
    job_started = time.monotonic()
    for position, spec in enumerate(specs):
        conversations, _ = load_conversations(spec, args.samples, args.seed)
        multi_turn = [c for c in conversations if sum(m["role"] == "assistant" for m in c.messages) >= 2]
        chosen = multi_turn[args.start_index : args.trajectories_per_dataset]

        def worker(conversation: Conversation, index: int) -> dict[str, Any]:
            tools = conversation.tools or derive_tools(conversation.messages)
            common: dict[str, Any] = {
                "max_turns": args.max_turns,
                "seed": args.seed + 1000 * index,
                "sampling": RL_SAMPLING,
                "max_tokens": args.max_tokens,
                "decode": decode,
                "encode": encode,
                "markers": markers,
                "stop_ids": stop_ids,
            }
            try:
                if args.driver == "rerender":
                    return {"index": index, **rollout_rerender(server, conversation, tools, **common)}
                return {
                    "index": index,
                    **rollout(server, conversation, tools, parser_check=args.parser_check, **common),
                }
            except Exception as error:  # 单条轨迹失败不应终止整个任务; 记录后在汇总中排除。
                return {
                    "index": index,
                    "sample_id": conversation.sample_id,
                    "failed": f"{type(error).__name__}: {error}"[:500],
                }

        deadline = None
        if args.time_budget_seconds is not None:
            remaining = max(args.time_budget_seconds - (time.monotonic() - job_started), 0)
            names = [s.name for s in specs[position:]]
            deadline = time.monotonic() + dataset_deadlines(remaining, names, args.dataset_weights)
        started = time.perf_counter()
        results, stopped = run_dataset(chosen, worker, args.concurrency, deadline, first_index=args.start_index)
        rollouts = [r for r in results if "failed" not in r]
        failures = [r for r in results if "failed" in r]
        summaries[spec.name] = {
            **(summarize_rerender(rollouts) if args.driver == "rerender" else summarize(rollouts)),
            "failed_rollouts": len(failures),
            "failure_examples": [r["failed"] for r in failures[:3]],
            "seconds": round(time.perf_counter() - started, 1),
            "stopped_by_time_budget": stopped,
        }
        all_rollouts[spec.name] = results
        print(f"{spec.name}: {json.dumps(summaries[spec.name], ensure_ascii=False)}", flush=True)
        report = {
            "schema_version": 1,
            "settings": {
                "model": model.__dict__,
                "v2_template": V2_TEMPLATES[args.model],
                "sampling": RL_SAMPLING,
                **{k: v for k, v in vars(args).items() if k not in ("out", "base_url")},
            },
            "complete": position == len(specs) - 1,
            "summaries": summaries,
            "rollouts": all_rollouts,
        }
        (out / "results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
