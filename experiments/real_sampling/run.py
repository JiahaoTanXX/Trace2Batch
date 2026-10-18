# Copyright (c) 2026 Jiahao Tan
# pyright: reportMissingImports=false
# torch / transformers 属于可选依赖, CI 中不安装。

"""用真实模型采样验证 trajectory 前缀一致性(前缀实验中 B 方式的真实版本)。

前缀实验里的 response 是"模板认可的规范输出", 测不到真实采样带来的问题。这里对数据集中
抽出的每个轮次:

1. 用数据集的真实历史 messages[:k] 渲染 prompt(与 vLLM 相同: chat template + 分词);
2. 让真实模型在该 prompt 上采样生成这一轮(temperature=1.0, 训练时的设置), 得到采样 token id;
3. 按 vLLM Hermes 工具解析器的逻辑把生成文本解析成 assistant 消息(content + tool_calls),
   接上数据集里该轮之后的观察消息, 重新渲染下一轮 prompt;
4. 检查"prompt + 采样 token"是否是下一轮 prompt 的前缀, 不是则用归因器分类。

不需要工具执行环境: 下一轮观察直接取数据集原有内容, 只用于让该轮助手消息进入历史。
本脚本在本地 CPU / MPS 上运行小模型; 大规模版本计划在 GPU 上用 vLLM 复现。

    python -m experiments.real_sampling.run --model qwen2.5-0.5b --turns-per-dataset 100
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.template_prefix.run import (
    CACHE_DIR,
    DATASETS,
    KINDS,
    QWEN_END,
    Conversation,
    Renderer,
    TokenizerConfig,
    load_conversations,
)
from experiments.tito.encoder import qwen_observation_ids
from trace2batch.mismatch import explain_mismatch

RESULTS_DIR = Path(__file__).with_name("results")
_QWEN25_05B = ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", "Qwen")
_QWEN3_06B = ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", "Qwen")
_NO_THINK = {"enable_thinking": False}
MODELS: dict[str, TokenizerConfig] = {
    "qwen2.5-0.5b": TokenizerConfig("qwen2.5-0.5b-instruct", *_QWEN25_05B, QWEN_END),
    # Qwen3 关闭思考: 官方模板 vs 前缀稳定补丁(experiments/template_patch)。
    "qwen3-0.6b": TokenizerConfig("qwen3-0.6b_no_think", *_QWEN3_06B, QWEN_END, template_kwargs=_NO_THINK),
    "qwen3-0.6b+patch": TokenizerConfig(
        "qwen3-0.6b_no_think+patch",
        *_QWEN3_06B,
        QWEN_END,
        template_kwargs=_NO_THINK,
        chat_template="experiments/template_patch/templates/qwen3_prefix_stable.jinja",
    ),
}
# vLLM Hermes 解析器(vllm@da9134b vllm/tool_parsers/hermes_tool_parser.py)使用的正则。
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
EXAMPLES_PER_KIND = 5
# rl: RL 训练常用的纯采样; recommended: 模型 generation_config 中的推荐设置(Qwen2.5-0.5B 为 T=0.7、
# top_p=0.8、top_k=20、repetition_penalty=1.1), 用来检验结论是否只是高温采样的产物。
RL_SAMPLING: dict[str, float] = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}


# ---------------------------------------------------------------- vLLM 行为模拟


def parse_hermes(model_output: str) -> tuple[dict[str, Any], bool]:
    """按 vLLM Hermes 工具解析器把生成文本变成客户端写回历史的 assistant 消息。

    返回 (消息, 是否解析出工具调用)。与 vLLM 一致:
    - 文本不含 <tool_call> 时, 整段作为 content;
    - 否则逐个 json.loads, arguments 用 json.dumps(ensure_ascii=False) 转成字符串,
      content 为第一个 <tool_call> 之前的文本(为空则为 None);
    - 任一调用解析失败时, 整段作为 content, 不含工具调用。
    下一次请求时 vLLM 的 _postprocess_messages 会把 arguments 字符串再解析成对象, 这里一并完成。
    """
    if "<tool_call>" not in model_output:
        return {"role": "assistant", "content": model_output}, False
    try:
        raw_calls = [json.loads(first or second) for first, second in _TOOL_CALL_RE.findall(model_output)]
        calls = [
            {
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.loads(json.dumps(call["arguments"], ensure_ascii=False)),
                },
            }
            for call in raw_calls
        ]
    except (ValueError, KeyError, TypeError):
        return {"role": "assistant", "content": model_output}, False
    content = model_output[: model_output.find("<tool_call>")]
    return {"role": "assistant", "content": content or None, "tool_calls": calls}, True


# ---------------------------------------------------------------- 轮次选择


@dataclass(frozen=True)
class Turn:
    sample_id: str
    position: int
    history: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    tools: Any


def candidate_turns(conversation: Conversation, max_turns: int) -> list[Turn]:
    """该对话中后面紧跟至少一条观察消息的助手轮次(前 max_turns 个)。"""
    messages = conversation.messages
    positions = [i for i, m in enumerate(messages) if m["role"] == "assistant"][:max_turns]
    turns: list[Turn] = []
    for position in positions:
        observations: list[dict[str, Any]] = []
        for message in messages[position + 1 :]:
            if message["role"] == "assistant":
                break
            observations.append(message)
        if observations:
            turns.append(Turn(conversation.sample_id, position, messages[:position], observations, conversation.tools))
    return turns


# ---------------------------------------------------------------- 采样与检查


class Sampler:
    def __init__(self, renderer: Renderer, device: str) -> None:
        config = renderer.config
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.renderer = renderer
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            config.repo, revision=config.revision, cache_dir=CACHE_DIR, dtype=torch.float32, trust_remote_code=False
        ).to(device)
        self.model.eval()
        tokenizer = renderer.tokenizer
        self.eos_ids = [tokenizer.convert_tokens_to_ids(token) for token in ("<|im_end|>", "<|endoftext|>")]
        defaults = self.model.generation_config
        self.recommended: dict[str, float] = {
            "temperature": defaults.temperature or 1.0,
            "top_p": defaults.top_p or 1.0,
            "top_k": defaults.top_k or 0,
            "repetition_penalty": defaults.repetition_penalty or 1.0,
        }
        self.params: dict[str, float] = dict(RL_SAMPLING)

    def sample(self, prompt_ids: list[int], seed: int, max_new_tokens: int) -> list[int]:
        torch = self.torch
        torch.manual_seed(seed)
        input_ids = torch.tensor([prompt_ids], device=self.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=True,
                temperature=self.params["temperature"],
                top_p=self.params["top_p"],
                top_k=int(self.params["top_k"]),
                repetition_penalty=self.params["repetition_penalty"],
                max_new_tokens=max_new_tokens,
                eos_token_id=self.eos_ids,
                pad_token_id=self.eos_ids[-1],
            )
        return output[0, len(prompt_ids) :].tolist()


def check_turn(
    turn: Turn, renderer: Renderer, prompt_ids: list[int], sampled: list[int], eos_ids: Sequence[int]
) -> dict[str, Any]:
    """对一个已采样轮次做 vLLM 链路下的前缀检查与自重分词检查。"""
    finished = bool(sampled) and sampled[-1] in eos_ids
    body = sampled[:-1] if finished else sampled
    # vLLM 默认 skip_special_tokens=True; <tool_call> 在 Qwen2.5 中不是 special token, 会保留。
    model_output = renderer.tokenizer.decode(body, skip_special_tokens=True)
    assistant, has_tool_call = parse_hermes(model_output)
    next_ids = renderer.encode(
        renderer.render([*turn.history, assistant, *turn.observations], turn.tools, generation_prompt=True)
    )
    context = prompt_ids + sampled
    report = explain_mismatch(context, next_ids, renderer.decode, markers=renderer.markers)
    retokenized = renderer.encode(renderer.decode(body))
    # token-in-token-out: 沿用 prompt + 采样 token, 只追加新消息的 token(按构造 100% 合并)。
    tito_next = context + qwen_observation_ids(renderer, turn.observations, turn.tools)
    return {
        "sample_id": turn.sample_id,
        "position": turn.position,
        "prompt_tokens": len(prompt_ids),
        "sampled_tokens": len(sampled),
        "finished": finished,
        "emitted_tool_call_tag": "<tool_call>" in model_output,
        "parsed_tool_call": has_tool_call,
        "self_retoken_mismatch": retokenized != body,
        "prefix_ok": report is None,
        "kind": None if report is None else report.kind,
        "token_index": None if report is None else report.token_index,
        "changed_segments": [] if report is None else report.changed_segments[:4],
        "prev_snippet": None if report is None else report.prev_snippet,
        "next_snippet": None if report is None else report.next_snippet,
        "model_output_head": model_output[:300],
        "tito_equals_rerender": tito_next == next_ids,
        "tito_text_equals_rerender": renderer.decode(tito_next) == renderer.decode(next_ids),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = Counter(r["kind"] for r in records if r["kind"])
    unfinished = [r for r in records if not r["finished"]]
    return {
        "turns": len(records),
        "finished": sum(r["finished"] for r in records),
        "emitted_tool_call_tag": sum(r["emitted_tool_call_tag"] for r in records),
        "parsed_tool_call": sum(r["parsed_tool_call"] for r in records),
        "tool_call_parse_failed": sum(r["emitted_tool_call_tag"] and not r["parsed_tool_call"] for r in records),
        "self_retoken_mismatch": sum(r["self_retoken_mismatch"] for r in records),
        "prefix_ok": sum(r["prefix_ok"] for r in records),
        "prefix_ok_finished_only": sum(r["prefix_ok"] for r in records if r["finished"]),
        "unfinished_prefix_ok": sum(r["prefix_ok"] for r in unfinished),
        **{kind: kinds.get(kind, 0) for kind in KINDS},
    }


def _pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}%" if denominator else "n/a"


def markdown_table(summaries: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| 数据集 | 采样轮次 | 正常结束 | 输出工具调用标签 | 工具调用解析失败 | 自重分词不一致 | 前缀成立 "
        "| template | retoken | others |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, s in summaries.items():
        n = s["turns"]
        lines.append(
            f"| {dataset} | {n} | {_pct(s['finished'], n)} | {_pct(s['emitted_tool_call_tag'], n)} "
            f"| {s['tool_call_parse_failed']} | {_pct(s['self_retoken_mismatch'], n)} | {_pct(s['prefix_ok'], n)} | "
            + " | ".join(str(s[kind]) for kind in KINDS)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--turns-per-dataset", type=int, default=60)
    parser.add_argument("--samples", type=int, default=300, help="与前缀实验相同的对话抽样数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096, help="CPU 上跳过更长的 prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sampling", choices=("rl", "recommended"), default="rl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", choices=sorted(MODELS), default="qwen2.5-0.5b")
    parser.add_argument("--out", type=Path, default=None, help="默认 results/<model>/")
    args = parser.parse_args(argv)
    model = MODELS[args.model]
    suffix = "" if args.sampling == "rl" else f"@{args.sampling}"
    out: Path = args.out or RESULTS_DIR / f"{args.model}{suffix}"

    import torch
    import transformers

    renderer = Renderer(model)
    sampler = Sampler(renderer, args.device)
    if args.sampling == "recommended":
        sampler.params = dict(sampler.recommended)
    out.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    all_records: dict[str, list[dict[str, Any]]] = {}
    for spec in DATASETS:
        conversations, _ = load_conversations(spec, args.samples, args.seed)
        candidates = [turn for conversation in conversations for turn in candidate_turns(conversation, args.max_turns)]
        random.Random(args.seed).shuffle(candidates)
        records: list[dict[str, Any]] = []
        skipped_long = 0
        started = time.perf_counter()
        for turn in candidates:
            if len(records) == args.turns_per_dataset:
                break
            prompt_ids = renderer.encode(renderer.render(turn.history, turn.tools, generation_prompt=True))
            if len(prompt_ids) > args.max_prompt_tokens:
                skipped_long += 1
                continue
            sampled = sampler.sample(prompt_ids, seed=args.seed + len(records), max_new_tokens=args.max_new_tokens)
            records.append(check_turn(turn, renderer, prompt_ids, sampled, sampler.eos_ids))
        summaries[spec.name] = {**summarize(records), "skipped_long_prompts": skipped_long}
        all_records[spec.name] = records
        print(f"{spec.name}: {summaries[spec.name]} ({time.perf_counter() - started:.0f}s)", flush=True)

    examples = {
        name: {kind: [r for r in records if r["kind"] == kind][:EXAMPLES_PER_KIND] for kind in KINDS}
        for name, records in all_records.items()
    }
    report = {
        "schema_version": 1,
        "settings": {
            "model": model.__dict__,
            "device": args.device,
            "turns_per_dataset": args.turns_per_dataset,
            "seed": args.seed,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "sampling": {"preset": args.sampling, **sampler.params},
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "summaries": summaries,
        "examples": examples,
        "records": all_records,
    }
    (out / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    (out / "summary.md").write_text(markdown_table(summaries), encoding="utf-8")
    print(markdown_table(summaries))


if __name__ == "__main__":
    main()
