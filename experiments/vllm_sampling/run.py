# Copyright (c) 2026 Jiahao Tan
# pyright: reportMissingImports=false

"""用真实 vLLM 服务端做真实采样的前缀一致性实验(GPU 版本)。

与 experiments/real_sampling(本地 transformers 模拟 vLLM)的区别: 这里直接请求一个
`vllm serve ... --enable-auto-tool-choice --tool-call-parser hermes` 服务, 链路与
agent-lightning 训练代理完全相同:

1. 第 k 轮: POST /v1/chat/completions(数据集真实历史, return_token_ids=True), 得到
   prompt_token_ids、采样 token_ids 和服务端解析出的 assistant 消息;
2. 第 k+1 轮: 把服务端返回的 assistant 消息原样写回历史, 接上数据集中的观察消息, 再请求一次
   (max_tokens=1), 得到服务端渲染的下一轮 prompt_token_ids;
3. 检查"第 k 轮 prompt + 生成 token"是否是下一轮 prompt 的前缀, 并用归因器分类。

请求并发发送; 结果按候选轮次的固定顺序取前 N 个, 与并发度无关。

    python -m experiments.vllm_sampling.run --model qwen2.5-1.5b --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from experiments.real_sampling.run import Turn, candidate_turns
from experiments.template_prefix.run import DATASETS, KINDS, load_conversations
from trace2batch.mismatch import explain_mismatch

RESULTS_DIR = Path(__file__).with_name("results")
EXAMPLES_PER_KIND = 5
Post = Callable[[str, dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ServedModel:
    """一个 vLLM 服务配置; served_name 是 /v1/chat/completions 请求里的 model 字段。"""

    key: str
    repo: str
    revision: str
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    chat_template: str | None = None  # 相对仓库根目录, 启动服务时传给 --chat-template
    # 模型自己推荐的采样设置(Qwen2.5 取 generation_config; Qwen3 取模型卡中非思考模式的建议)。
    recommended: dict[str, float] = field(default_factory=dict)


QWEN3_PATCH = "experiments/template_patch/templates/qwen3_prefix_stable.jinja"
# 补丁 v2: content 已以换行结尾时不再在工具调用前插入换行(与 experiments.template_patch 的 PATCHES_V2 相同)。
QWEN3_PATCH_V2 = "experiments/template_patch/templates/qwen3_prefix_stable_v2.jinja"
QWEN25_PATCH_V2 = "experiments/template_patch/templates/qwen2.5_tool_newline.jinja"
_QWEN25_RECOMMENDED = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.1}
_QWEN3_NO_THINK_RECOMMENDED = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
_QWEN3 = ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca")
MODELS: dict[str, ServedModel] = {
    "qwen2.5-1.5b": ServedModel(
        "qwen2.5-1.5b",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        recommended=_QWEN25_RECOMMENDED,
    ),
    "qwen2.5-0.5b": ServedModel(
        "qwen2.5-0.5b",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775",
        recommended=_QWEN25_RECOMMENDED,
    ),
    "qwen3-0.6b": ServedModel(
        "qwen3-0.6b", *_QWEN3, {"enable_thinking": False}, recommended=_QWEN3_NO_THINK_RECOMMENDED
    ),
    "qwen3-0.6b+patch": ServedModel(
        "qwen3-0.6b+patch", *_QWEN3, {"enable_thinking": False}, QWEN3_PATCH, _QWEN3_NO_THINK_RECOMMENDED
    ),
    "qwen2.5-1.5b+patch2": ServedModel(
        "qwen2.5-1.5b+patch2",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        chat_template=QWEN25_PATCH_V2,
        recommended=_QWEN25_RECOMMENDED,
    ),
    "qwen3-0.6b+patch2": ServedModel(
        "qwen3-0.6b+patch2", *_QWEN3, {"enable_thinking": False}, QWEN3_PATCH_V2, _QWEN3_NO_THINK_RECOMMENDED
    ),
}
# RL 训练常用的纯采样; recommended 使用 ServedModel.recommended。
RL_SAMPLING: dict[str, float] = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0}


def http_post(base_url: str) -> Post:
    def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            base_url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            return {"error": {"status": error.code, "message": error.read().decode(errors="replace")[:500]}}

    return post


def assistant_for_history(message: dict[str, Any]) -> dict[str, Any]:
    """服务端返回的 assistant 消息, 按 OpenAI 客户端的习惯写回历史(只保留 role / content / tool_calls)。"""
    result: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        result["tool_calls"] = [
            {"id": call["id"], "type": "function", "function": dict(call["function"])} for call in message["tool_calls"]
        ]
    return result


def observations_for_history(observations: list[dict[str, Any]], assistant: dict[str, Any]) -> list[dict[str, Any]]:
    """数据集观察消息; tool 消息补上 tool_call_id(OpenAI 接口要求), 优先使用本轮返回的调用 id。"""
    call_ids = [call["id"] for call in assistant.get("tool_calls") or []]
    result = []
    tool_index = 0
    for message in observations:
        if message["role"] == "tool":
            call_id = call_ids[tool_index] if tool_index < len(call_ids) else f"call_{tool_index}"
            result.append({"role": "tool", "content": message["content"], "tool_call_id": call_id})
            tool_index += 1
        else:
            result.append({"role": message["role"], "content": message["content"]})
    return result


def derive_tools(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """数据集没有工具定义时(SWE-smith), 用对话中出现过的工具名和参数名构造近似定义。

    没有工具定义时 vLLM 不会解析工具调用, 模型输出只会被当作普通文本; 近似定义让解析链路生效。
    参数类型一律写成 string, 不标必填。
    """
    arguments: dict[str, dict[str, None]] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            names = arguments.setdefault(call["function"]["name"], {})
            raw = call["function"]["arguments"]
            if isinstance(raw, dict):
                names.update(dict.fromkeys(raw))
    if not arguments:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Tool `{name}` (schema derived from the dataset trajectory).",
                "parameters": {"type": "object", "properties": {arg: {"type": "string"} for arg in args}},
            },
        }
        for name, args in arguments.items()
    ]


def history_for_api(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把数据集历史转成 OpenAI 接口要求的格式, 与真实 agent 客户端回传给服务端的一致。

    规范化后的数据集里, 工具参数是对象、工具调用没有 id、tool 消息没有 tool_call_id;
    OpenAI 接口要求 arguments 为 JSON 字符串、tool 消息带 tool_call_id。pilot 第 1 版直接发送
    规范化后的历史, SWE-smith 的大部分请求因此被拒(HTTP 400)。服务端收到后会把参数解析回对象,
    所以渲染出的 prompt 与前缀实验中的处理一致。
    """
    result: list[dict[str, Any]] = []
    pending_ids: list[str] = []
    for index, message in enumerate(history):
        if message["role"] == "assistant" and message.get("tool_calls"):
            calls = []
            for number, call in enumerate(message["tool_calls"]):
                arguments = call["function"]["arguments"]
                calls.append(
                    {
                        "id": call.get("id") or f"call_{index}_{number}",
                        "type": "function",
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": arguments
                            if isinstance(arguments, str)
                            else json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                )
            pending_ids = [call["id"] for call in calls]
            result.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
        elif message["role"] == "tool":
            call_id = pending_ids.pop(0) if pending_ids else f"call_{index}"
            result.append({"role": "tool", "content": message["content"], "tool_call_id": call_id})
        else:
            result.append({"role": message["role"], "content": message["content"]})
    return result


def _prompt_ids(response: dict[str, Any]) -> list[int]:
    return list(response.get("prompt_token_ids") or response["choices"][0].get("prompt_token_ids") or [])


def process_turn(
    turn: Turn,
    seed: int,
    post: Post,
    model: ServedModel,
    sampling: dict[str, float],
    max_tokens: int,
    decode: Callable[[Sequence[int]], str],
    encode: Callable[[str], list[int]],
    markers: Sequence[str],
) -> dict[str, Any]:
    """按 vLLM 链路处理一个轮次; 请求失败(例如 prompt 超出 max_model_len)时返回带 error 的记录。"""
    base = {
        "model": model.key,
        "return_token_ids": True,
        "chat_template_kwargs": model.chat_template_kwargs,
        **({"tools": turn.tools} if turn.tools else {}),
    }
    started = time.perf_counter()
    history = history_for_api(turn.history)
    first = post(
        "/v1/chat/completions", {**base, **sampling, "messages": history, "max_tokens": max_tokens, "seed": seed}
    )
    if "error" in first:
        return {"sample_id": turn.sample_id, "position": turn.position, "error": first["error"]}
    choice = first["choices"][0]
    prompt_ids = _prompt_ids(first)
    sampled = list(choice.get("token_ids") or [])
    assistant = assistant_for_history(choice["message"])
    follow = [*history, assistant, *observations_for_history(turn.observations, assistant)]
    second = post("/v1/chat/completions", {**base, "messages": follow, "max_tokens": 1, "temperature": 0.0})
    if "error" in second:
        return {"sample_id": turn.sample_id, "position": turn.position, "error": second["error"]}
    next_ids = _prompt_ids(second)
    context = prompt_ids + sampled
    report = explain_mismatch(context, next_ids, decode, markers=markers)
    # 以工具调用结束时 finish_reason 为 "tool_calls"(pilot 第 1 版误把它算作未结束)。
    finished = choice.get("finish_reason") in ("stop", "tool_calls")
    body = sampled[:-1] if finished and sampled else sampled
    return {
        "sample_id": turn.sample_id,
        "position": turn.position,
        "prompt_tokens": len(prompt_ids),
        "sampled_tokens": len(sampled),
        "finished": finished,
        "finish_reason": choice.get("finish_reason"),
        "tool_calls": len(assistant.get("tool_calls") or []),
        "self_retoken_mismatch": encode(decode(body)) != body,
        "prefix_ok": report is None,
        "kind": None if report is None else report.kind,
        "token_index": None if report is None else report.token_index,
        "changed_segments": [] if report is None else report.changed_segments[:4],
        "prev_snippet": None if report is None else report.prev_snippet,
        "next_snippet": None if report is None else report.next_snippet,
        "content_head": (choice["message"].get("content") or "")[:300],
        "seconds": round(time.perf_counter() - started, 3),
    }


def sample_dataset(
    candidates: list[Turn],
    target: int,
    concurrency: int,
    seed: int,
    worker: Callable[[Turn, int], dict[str, Any]],
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[list[dict[str, Any]], Counter[str], list[str], bool]:
    """按候选顺序分批并发处理, 取前 target 个成功的轮次; 失败按原因计数。

    deadline(clock 的时刻)到达后不再开始新的批次, 已发出的批次会等它完成;
    最后一个返回值表示是否因时间上限提前停止。
    """
    records: list[dict[str, Any]] = []
    errors: Counter[str] = Counter()
    error_examples: list[str] = []
    index = 0
    stopped = False
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        while len(records) < target and index < len(candidates):
            if deadline is not None and clock() >= deadline:
                stopped = True
                break
            batch = candidates[index : index + concurrency]
            results = list(pool.map(worker, batch, [seed + index + offset for offset in range(len(batch))]))
            index += len(batch)
            for result in results:
                if "error" in result:
                    message = str(result["error"].get("message", ""))
                    reason = "context_length" if "context length" in message or "maximum" in message else "other"
                    errors[f"{result['error'].get('status')}:{reason}"] += 1
                    if reason == "other" and len(error_examples) < 3:
                        error_examples.append(message[:300])
                elif len(records) < target:
                    records.append(result)
    return records, errors, error_examples, stopped


def summarize(records: list[dict[str, Any]], errors: Counter[str]) -> dict[str, Any]:
    kinds = Counter(r["kind"] for r in records if r["kind"])
    finished = [r for r in records if r["finished"]]
    return {
        "turns": len(records),
        "finished": len(finished),
        "with_tool_calls": sum(r["tool_calls"] > 0 for r in records),
        "self_retoken_mismatch": sum(r["self_retoken_mismatch"] for r in records),
        "prefix_ok": sum(r["prefix_ok"] for r in records),
        "prefix_ok_finished": sum(r["prefix_ok"] for r in finished),
        "prompt_tokens_max": max((r["prompt_tokens"] for r in records), default=0),
        "request_errors": dict(errors),
        **{kind: kinds.get(kind, 0) for kind in KINDS},
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--sampling", choices=("rl", "recommended"), default="rl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--turns-per-dataset", type=int, default=300)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=None,
        help="整个任务的时间上限; 每个数据集分得剩余时间的平均份额, 到点后不再开始新批次",
    )
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from experiments.template_prefix.run import CACHE_DIR, DEFAULT_TEMPLATE_MARKERS

    model = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model.repo, revision=model.revision, cache_dir=CACHE_DIR)
    markers = tuple(
        dict.fromkeys([*DEFAULT_TEMPLATE_MARKERS, *(t.content for t in tokenizer.added_tokens_decoder.values())])
    )

    def decode(ids: Sequence[int]) -> str:
        return tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    post = http_post(args.base_url)
    sampling = RL_SAMPLING if args.sampling == "rl" else model.recommended
    out: Path = args.out or RESULTS_DIR / f"{args.model}@{args.sampling}"
    out.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    all_records: dict[str, list[dict[str, Any]]] = {}
    job_started = time.monotonic()
    for position, spec in enumerate(DATASETS):
        conversations, _ = load_conversations(spec, args.samples, args.seed)
        candidates = []
        for conversation in conversations:
            tools = conversation.tools or derive_tools(conversation.messages)
            candidates += [replace(t, tools=tools) for t in candidate_turns(conversation, args.max_turns)]
        random.Random(args.seed).shuffle(candidates)

        def worker(turn: Turn, seed: int) -> dict[str, Any]:
            return process_turn(turn, seed, post, model, sampling, args.max_tokens, decode, encode, markers)

        deadline = None
        if args.time_budget_seconds is not None:
            remaining = args.time_budget_seconds - (time.monotonic() - job_started)
            deadline = time.monotonic() + max(remaining, 0) / (len(DATASETS) - position)
        started = time.perf_counter()
        records, errors, error_examples, stopped = sample_dataset(
            candidates, args.turns_per_dataset, args.concurrency, args.seed, worker, deadline
        )
        elapsed = time.perf_counter() - started
        summaries[spec.name] = {
            **summarize(records, errors),
            "seconds": round(elapsed, 1),
            "stopped_by_time_budget": stopped,
            "request_error_examples": error_examples,
        }
        all_records[spec.name] = records
        print(f"{spec.name}: {summaries[spec.name]}", flush=True)
        # 每个数据集完成后就写盘: 即使任务之后被强制终止, 已完成的数据集也不会丢失。
        write_report(out, model, args, sampling, summaries, all_records, complete=position == len(DATASETS) - 1)
    print(json.dumps(summaries, ensure_ascii=False, indent=1))


def write_report(
    out: Path,
    model: ServedModel,
    args: argparse.Namespace,
    sampling: dict[str, float],
    summaries: dict[str, Any],
    all_records: dict[str, list[dict[str, Any]]],
    complete: bool,
) -> None:
    report = {
        "schema_version": 1,
        "settings": {
            "model": model.__dict__,
            "sampling": {"preset": args.sampling, **sampling},
            "turns_per_dataset": args.turns_per_dataset,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "time_budget_seconds": args.time_budget_seconds,
        },
        "complete": complete,
        "summaries": summaries,
        "examples": {
            name: {kind: [r for r in records if r["kind"] == kind][:EXAMPLES_PER_KIND] for kind in KINDS}
            for name, records in all_records.items()
        },
        "records": all_records,
    }
    (out / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
