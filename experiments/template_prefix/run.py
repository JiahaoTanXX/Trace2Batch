# Copyright (c) 2026 Jiahao Tan
# pyright: reportMissingImports=false
# transformers / pyarrow / huggingface_hub 属于可选依赖 [real], CI 中不安装。

"""真实 chat template 下的 trajectory 前缀一致性实验。

对每条轨迹、每个 tokenizer 配置, 按三种采集方式构造逐轮 (prompt_ids, response_ids) 事件,
喂给 build_samples(mode="trajectory"), 再用 trace2batch.mismatch 归因每次合并失败。

- A passthrough: 框架直接拼接 token, 新一轮 prompt = 上一轮上下文 + 新消息的 token(对照组)。
- B vllm_token_ids: 模拟 agent-lightning 代理 + vLLM return_token_ids。每轮 prompt 由服务端
  对完整 messages 重新渲染并分词; response 取"该助手消息作为最后一条时的渲染"在上下文中的
  token 切片, 即模板认可的规范生成。只暴露模板改写, 不引入切分差异。
- C retokenized_text: 与 B 相同, 但 response 由生成文本单独分词, 模拟只拿到文本、再分词的代理。

依赖 [real] 可选依赖(transformers、pyarrow); 只读取本地缓存中的 tokenizer 文件与数据分片,
不加载模型权重, 不执行数据或 tokenizer 仓库中的代码。运行:

    python -m experiments.template_prefix.run
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from trace2batch.core import build_samples
from trace2batch.mismatch import DEFAULT_TEMPLATE_MARKERS, MismatchKind, explain_mismatch

RESULTS_DIR = Path(__file__).with_name("results")
REPO_ROOT = Path(__file__).resolve().parents[2]
# 下载的 tokenizer / 数据分片属于不可信数据, 放在独立缓存目录。显式传 cache_dir,
# 不依赖 HF_HOME: 测试等场景可能在本模块之前就导入了 huggingface_hub。
CACHE_DIR = Path(os.environ.get("TRACE2BATCH_HF_CACHE", Path.home() / ".cache" / "trace2batch-hf" / "hub"))
KINDS: tuple[MismatchKind, ...] = ("template_mismatch", "retoken_mismatch", "others_mismatch")
MODES = ("A_passthrough", "B_vllm_token_ids", "C_retokenized_text")
# 合成思考内容: 数据集没有独立的思考字段, 只在配置名带 [synthetic] 的配置中注入。
SYNTHETIC_REASONING = "Let me reason about the next step before acting."
EXAMPLES_PER_KIND = 3
# gpt-oss 模板把 strftime_now() 的当前日期写进 system 消息; 固定日期保证结果可复现。
FIXED_TEMPLATE_DATE = date(2026, 10, 16)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo: str
    revision: str
    shard: str
    license: str


@dataclass(frozen=True)
class TokenizerConfig:
    name: str
    repo: str
    revision: str
    family: str
    # 生成在这些标记处停止(包含标记本身); 空表示模板不在助手消息末尾写结束标记。
    end_markers: tuple[str, ...]
    template_kwargs: dict[str, Any] = field(default_factory=dict)
    synthetic_reasoning: bool = False
    keep_reasoning_in_history: bool = False
    # DeepSeek-V3.1 模板用字符串拼接工具参数, 需要 JSON 字符串而不是 vLLM 解析出的对象。
    string_tool_arguments: bool = False
    # A 直通依赖"每条助手消息渲染一个 <|im_start|>assistant"的 Qwen 结构, 只对 Qwen 计算。
    passthrough: bool = False
    # 替换 tokenizer 自带模板的 Jinja 文件(相对仓库根目录), 用于验证模板补丁; None 表示用官方模板。
    chat_template: str | None = None


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "swe_smith_tool",
        "SWE-bench/SWE-smith-trajectories",
        "08e109b4a59eaeebf80e4675cd125d42e7ac99a4",
        "data/tool-00004-of-00008.parquet",
        "MIT",
    ),
    DatasetSpec(
        "toucan_qwen3",
        "Agent-Ark/Toucan-1.5M",
        "0df3cf37f2abefb380370cfb02eabea2a35ae782",
        "Qwen3/train-00009-of-00044.parquet",
        "Apache-2.0",
    ),
    DatasetSpec(
        "nebius_swe_agent",
        "nebius/SWE-agent-trajectories",
        "68195a1450865274106246d0d0296a1d6807b88e",
        "data/train-00006-of-00012.parquet",
        "CC-BY-4.0",
    ),
)

QWEN25 = ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", "Qwen")
QWEN3 = ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "Qwen")
DEEPSEEK = ("deepseek-ai/DeepSeek-V3.1", "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db", "DeepSeek")
GLM = ("zai-org/GLM-4.5", "cbb2c7cfb52fa128a9660cb1a7a78e017899e115", "GLM")
GPT_OSS = ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee", "gpt-oss")
QWEN_END = ("<|im_end|>",)
TOKENIZER_CONFIGS: tuple[TokenizerConfig, ...] = (
    TokenizerConfig("qwen2.5", *QWEN25, QWEN_END, passthrough=True),
    TokenizerConfig("qwen3_no_think", *QWEN3, QWEN_END, template_kwargs={"enable_thinking": False}, passthrough=True),
    TokenizerConfig(
        "qwen3_think_keep[synthetic]",
        *QWEN3,
        QWEN_END,
        synthetic_reasoning=True,
        keep_reasoning_in_history=True,
        passthrough=True,
    ),
    TokenizerConfig(
        "qwen3_think_drop[synthetic]",
        *QWEN3,
        QWEN_END,
        synthetic_reasoning=True,
        keep_reasoning_in_history=False,
        passthrough=True,
    ),
    # DeepSeek 的结束 token 本身由全角竖线组成。
    TokenizerConfig("deepseek_v3.1", *DEEPSEEK, ("<｜end▁of▁sentence｜>",), string_tool_arguments=True),  # noqa: RUF001
    TokenizerConfig("glm4.5_no_think", *GLM, (), template_kwargs={"enable_thinking": False}),
    TokenizerConfig("glm4.5_think_keep[synthetic]", *GLM, (), synthetic_reasoning=True, keep_reasoning_in_history=True),
    TokenizerConfig(
        "glm4.5_think_drop[synthetic]", *GLM, (), synthetic_reasoning=True, keep_reasoning_in_history=False
    ),
    TokenizerConfig("gpt-oss", *GPT_OSS, ("<|return|>", "<|call|>")),
)


# ---------------------------------------------------------------- 数据加载与规范化


@dataclass(frozen=True)
class Conversation:
    sample_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None


def _text(content: Any) -> str:
    """SWE-smith 的 content 可能是 [{'type': 'text', 'text': ...}] 列表。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def _arguments(raw: Any) -> Any:
    """与 vLLM 一致: 渲染前把 JSON 字符串形式的工具参数解析成对象。"""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


def _tool_call(name: str, arguments: Any) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "arguments": _arguments(arguments)}}


def _swe_smith(row: dict[str, Any]) -> Conversation | None:
    raw = json.loads(row["messages"])
    if not any("tool_calls" in message for message in raw):
        return None  # 该分片中约 3/4 是 XML 文本动作, 只取结构化工具调用的轨迹
    messages: list[dict[str, Any]] = []
    for message in raw:
        role = message["role"]
        item: dict[str, Any] = {"role": role, "content": _text(message.get("content"))}
        if role == "assistant" and message.get("tool_calls"):
            calls = message["tool_calls"]
            if isinstance(calls, str):
                calls = ast.literal_eval(calls)
            item["tool_calls"] = [_tool_call(c["function"]["name"], c["function"]["arguments"]) for c in calls]
        messages.append(item)
    return Conversation(row["traj_id"], messages, None)


def _toucan(row: dict[str, Any]) -> Conversation | None:
    raw = json.loads(row["messages"])
    messages: list[dict[str, Any]] = []
    for message in raw:
        role = message["role"]
        if role == "system" and message["content"].startswith("# Tools"):
            continue  # 系统消息是预先渲染好的工具提示, 改为通过 tools 参数交给模板渲染
        if role == "function":
            messages.append({"role": "tool", "content": _text(message.get("content"))})
            continue
        item: dict[str, Any] = {"role": role, "content": _text(message.get("content"))}
        call = message.get("function_call")
        if role == "assistant" and call:
            if isinstance(call, str):
                call = ast.literal_eval(call)
            item["tool_calls"] = [_tool_call(call["name"], call["arguments"])]
        messages.append(item)
    return Conversation(row["uuid"], messages, json.loads(row["available_tools"]))


def _nebius(row: dict[str, Any], index: int) -> Conversation | None:
    roles = {"system": "system", "user": "user", "ai": "assistant"}
    messages: list[dict[str, Any]] = []
    for message in row["trajectory"]:
        role = roles[message["role"]]
        content = message["system_prompt"] if role == "system" else message["text"]
        messages.append({"role": role, "content": content or ""})
    # instance_id 会在不同模型 / 尝试间重复, 用分片内行号保证唯一。
    return Conversation(f"{index}:{row['instance_id']}", messages, None)


def _merge_consecutive_assistants(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """连续的助手消息合并为一条(内容拼接、tool_calls 串联)。

    Toucan 把并行工具调用存成多条相邻助手消息; 真实模型会在同一次生成中输出它们,
    OpenAI 格式也用一条消息里的多个 tool_calls 表示。
    """
    merged: list[dict[str, Any]] = []
    for message in messages:
        if merged and message["role"] == "assistant" and merged[-1]["role"] == "assistant":
            previous = merged[-1]
            content = "\n".join(part for part in (previous["content"], message["content"]) if part)
            calls = [*previous.get("tool_calls", []), *message.get("tool_calls", [])]
            merged[-1] = {"role": "assistant", "content": content, **({"tool_calls": calls} if calls else {})}
        else:
            merged.append(message)
    return merged


def _shard_path(spec: DatasetSpec) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(spec.repo, spec.shard, repo_type="dataset", revision=spec.revision, cache_dir=CACHE_DIR)


def load_conversations(spec: DatasetSpec, samples: int, seed: int) -> tuple[list[Conversation], int]:
    """读取分片, 规范化为 OpenAI 风格 messages, 按固定种子抽样; 返回样本与可用总数。"""
    import pyarrow.parquet as pq

    rows = pq.read_table(_shard_path(spec)).to_pylist()
    conversations: list[Conversation] = []
    for index, row in enumerate(rows):
        if spec.name == "swe_smith_tool":
            conversation = _swe_smith(row)
        elif spec.name == "toucan_qwen3":
            conversation = _toucan(row)
        else:
            conversation = _nebius(row, index)
        if conversation is not None:
            conversation = Conversation(
                conversation.sample_id, _merge_consecutive_assistants(conversation.messages), conversation.tools
            )
        if conversation is not None and sum(m["role"] == "assistant" for m in conversation.messages) >= 2:
            conversations.append(conversation)
    chosen = sorted(random.Random(seed).sample(range(len(conversations)), min(samples, len(conversations))))
    return [conversations[index] for index in chosen], len(conversations)


# ---------------------------------------------------------------- 逐轮事件构造


class Renderer:
    """包装真实 tokenizer: 渲染 chat template、分词与保留特殊 token 的解码。"""

    def __init__(self, config: TokenizerConfig) -> None:
        from transformers import AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.repo, revision=config.revision, trust_remote_code=False, cache_dir=CACHE_DIR
        )
        # 分类器的模板标记: 通用标记 + 该 tokenizer 自己的特殊 / 新增 token。
        added = [token.content for token in self.tokenizer.added_tokens_decoder.values()]
        self.markers: tuple[str, ...] = tuple(dict.fromkeys([*DEFAULT_TEMPLATE_MARKERS, *added]))
        self.chat_template: str | None = None
        if config.chat_template is not None:
            self.chat_template = (REPO_ROOT / config.chat_template).read_text(encoding="utf-8")

    def render(self, messages: list[dict[str, Any]], tools: Any, *, generation_prompt: bool) -> str:
        if self.config.string_tool_arguments:
            messages = [_stringify_tool_arguments(message) for message in messages]
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=generation_prompt,
            tokenize=False,
            chat_template=self.chat_template,
            strftime_now=FIXED_TEMPLATE_DATE.strftime,
            **self.config.template_kwargs,
        )

    def encode(self, text: str) -> list[int]:
        # vLLM 对已渲染的 chat prompt 使用 add_special_tokens=False。
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _stringify_tool_arguments(message: dict[str, Any]) -> dict[str, Any]:
    calls = message.get("tool_calls")
    if not calls:
        return message
    converted = []
    for call in calls:
        arguments = call["function"]["arguments"]
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        converted.append({**call, "function": {**call["function"], "arguments": arguments}})
    return {**message, "tool_calls": converted}


def _trim_after_end(text: str, end_markers: tuple[str, ...]) -> str:
    """生成在最早出现的结束标记处停止; 模板在其后追加的内容属于下一轮 prompt。"""
    ends = [text.find(marker) + len(marker) for marker in end_markers if marker in text]
    return text[: min(ends)] if ends else text


def _generation_text(prompt_text: str, full_text: str, history_text: str) -> str | None:
    """该助手消息作为最后一条消息时, 模板渲染出的生成部分。

    通常 full_text 以 prompt_text 开头, 生成就是剩余部分。若追加这条消息会改写更早的历史
    (例如 gpt-oss 在出现最终回答后删除之前工具调用的 analysis), 就用生成提示串最后一次出现的
    位置定位; 仍无法定位时返回 None。
    """
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :]
    generation_prompt = prompt_text[len(history_text) :] if prompt_text.startswith(history_text) else ""
    position = full_text.rfind(generation_prompt) if generation_prompt else -1
    return full_text[position + len(generation_prompt) :] if position != -1 else None


def _text_after_assistant(prompt_text: str, position: int, messages: list[dict[str, Any]]) -> str:
    """prompt 文本中最后一条助手消息(位于 position 之前)的 <|im_end|> 之后的部分。

    依赖 Qwen 系模板的结构: 每条助手消息恰好渲染一个 "<|im_start|>assistant"。
    """
    count = sum(m["role"] == "assistant" for m in messages[:position])
    start = -1
    for _ in range(count):
        start = prompt_text.index("<|im_start|>assistant", start + 1)
    end = prompt_text.index("<|im_end|>", start)
    return prompt_text[end + len("<|im_end|>") :]


@dataclass
class TurnBuild:
    events: list[dict[str, Any]]
    generation_not_token_prefix: int = 0
    generation_not_text_prefix: int = 0
    generation_unresolved: int = 0


def build_turn_events(conversation: Conversation, renderer: Renderer, max_turns: int) -> dict[str, TurnBuild]:
    """把一条对话转成三种采集方式下的逐轮 model_request 事件; 三者共享每轮的渲染与分词。"""
    config = renderer.config
    messages = conversation.messages
    assistant_positions = [i for i, m in enumerate(messages) if m["role"] == "assistant"][:max_turns]

    def with_reasoning(message: dict[str, Any]) -> dict[str, Any]:
        return {**message, "reasoning_content": SYNTHETIC_REASONING} if config.synthetic_reasoning else message

    def history(end: int) -> list[dict[str, Any]]:
        """客户端回传给服务端的历史; think_drop 时不回传 reasoning_content。"""
        keep = config.synthetic_reasoning and config.keep_reasoning_in_history
        return [with_reasoning(m) if keep and m["role"] == "assistant" else m for m in messages[:end]]

    modes = MODES if config.passthrough else tuple(mode for mode in MODES if mode != "A_passthrough")
    builds = {mode: TurnBuild(events=[]) for mode in modes}
    passthrough_context: list[int] = []
    for turn, position in enumerate(assistant_positions):
        prompt_text = renderer.render(history(position), conversation.tools, generation_prompt=True)
        full_text = renderer.render(
            [*history(position), with_reasoning(messages[position])], conversation.tools, generation_prompt=False
        )
        not_text_prefix = int(not full_text.startswith(prompt_text))
        located = _generation_text(
            prompt_text,
            full_text,
            renderer.render(history(position), conversation.tools, generation_prompt=False) if not_text_prefix else "",
        )
        unresolved = int(located is None)
        generation_text = _trim_after_end(located if located is not None else "", config.end_markers)

        prompt_ids = renderer.encode(prompt_text)
        joint = renderer.encode(prompt_text + generation_text)
        not_token_prefix = int(joint[: len(prompt_ids)] != prompt_ids)
        canonical_response = joint[len(prompt_ids) :]
        per_mode = {
            "B_vllm_token_ids": (prompt_ids, canonical_response),
            "C_retokenized_text": (prompt_ids, renderer.encode(generation_text)),
        }
        if config.passthrough:
            if turn == 0:
                passthrough_prompt = prompt_ids
            else:
                # 只对新增消息(上一条助手消息之后的文本)分词, 接在已有 token 后面。
                passthrough_prompt = passthrough_context + renderer.encode(
                    _text_after_assistant(prompt_text, position, messages)
                )
            passthrough_context = passthrough_prompt + canonical_response
            per_mode["A_passthrough"] = (passthrough_prompt, canonical_response)

        for mode, (prompt, response) in per_mode.items():
            build = builds[mode]
            build.events.append(
                {"event_type": "model_request", "data": {"prompt_token_ids": prompt, "response_token_ids": response}}
            )
            build.generation_not_text_prefix += not_text_prefix
            build.generation_not_token_prefix += not_token_prefix
            build.generation_unresolved += unresolved
    return builds


# ---------------------------------------------------------------- 合并与归因


def attribute_mismatches(
    events: list[dict[str, Any]],
    decode: Callable[[Sequence[int]], str],
    markers: Sequence[str] = DEFAULT_TEMPLATE_MARKERS,
) -> Iterator[dict[str, Any]]:
    """复现 build_samples 的合并循环, 对每次前缀失败给出原因报告。"""
    context: list[int] | None = None
    for turn, event in enumerate(events):
        prompt = event["data"]["prompt_token_ids"]
        response = event["data"]["response_token_ids"]
        if context is not None and prompt[: len(context)] != context:
            report = explain_mismatch(context, prompt, decode, markers=markers)
            assert report is not None
            yield {"turn": turn, **report.__dict__}
        context = prompt + response


def run_config(
    conversations: list[Conversation], renderer: Renderer, max_turns: int
) -> dict[str, tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]]:
    """对一个 tokenizer 配置跑完所有对话, 返回每种采集方式的汇总与典型案例。

    模板对某条对话渲染报错时(例如工具参数类型不符合模板预期), 该对话计入 render_errors 并跳过。
    """
    kinds: dict[str, Counter[str]] = {}
    totals: dict[str, Counter[str]] = {}
    examples: dict[str, dict[str, list[dict[str, Any]]]] = {}
    render_errors: Counter[str] = Counter()
    for conversation in conversations:
        try:
            builds = build_turn_events(conversation, renderer, max_turns)
        except Exception as error:  # 模板内的 raise_exception / 类型错误
            render_errors[f"{type(error).__name__}: {str(error)[:120]}"] += 1
            continue
        for mode, built in builds.items():
            kinds.setdefault(mode, Counter())
            total = totals.setdefault(mode, Counter())
            mode_examples = examples.setdefault(mode, {kind: [] for kind in KINDS})
            result = build_samples(built.events, mode="trajectory", max_prompt_length=10**9, max_response_length=10**9)
            reports = list(attribute_mismatches(built.events, renderer.decode, renderer.markers))
            if len(reports) != result.diagnostics["merge_mismatches"]:
                raise RuntimeError(f"{conversation.sample_id}: 归因次数与 build_samples 不一致")
            total["trajectories"] += 1
            total["turns"] += len(built.events)
            total["merge_attempts"] += max(len(built.events) - 1, 0)
            total["merge_mismatches"] += len(reports)
            total["rows"] += len(result.rows)
            total["fully_merged_trajectories"] += int(len(result.rows) == 1)
            # 行内 prompt + response token 总量: 合并失败时每行都重复携带历史前缀。
            total["row_tokens"] += sum(len(r.prompt_token_ids) + len(r.response_token_ids) for r in result.rows)
            total["policy_tokens"] += sum(sum(r.response_mask) for r in result.rows)
            total["generation_not_token_prefix"] += built.generation_not_token_prefix
            total["generation_not_text_prefix"] += built.generation_not_text_prefix
            total["generation_unresolved"] += built.generation_unresolved
            for report in reports:
                kinds[mode][report["kind"]] += 1
                if len(mode_examples[report["kind"]]) < EXAMPLES_PER_KIND:
                    mode_examples[report["kind"]].append({"sample_id": conversation.sample_id, **report})
    output: dict[str, tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]] = {}
    for mode in totals:
        summary: dict[str, Any] = dict(totals[mode])
        summary.update({kind: kinds[mode][kind] for kind in KINDS})
        summary["render_errors"] = sum(render_errors.values())
        summary["render_error_types"] = dict(render_errors)
        output[mode] = (summary, examples[mode])
    return output


# ---------------------------------------------------------------- 输出


def _pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}%" if denominator else "n/a"


def markdown_table(cells: list[dict[str, Any]]) -> str:
    lines = [
        "| 数据集 | tokenizer 配置 | 采集方式 | 轨迹数 | 合并尝试 | 合并成功率 | 完整合并轨迹 | 行数/轨迹 "
        "| template | retoken | others |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        s = cell["summary"]
        failures = s["merge_mismatches"]
        lines.append(
            f"| {cell['dataset']} | {cell['tokenizer']} | {cell['mode']} | {s['trajectories']} | {s['merge_attempts']} "
            f"| {_pct(s['merge_attempts'] - failures, s['merge_attempts'])} "
            f"| {_pct(s['fully_merged_trajectories'], s['trajectories'])} "
            f"| {s['rows'] / s['trajectories']:.2f} | "
            + " | ".join(f"{s[kind]} ({_pct(s[kind], failures)})" for kind in KINDS)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=300, help="每个数据集抽样的轨迹数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=50, help="每条轨迹最多使用的助手轮数")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    import transformers

    args.out.mkdir(parents=True, exist_ok=True)
    renderers = {config.name: Renderer(config) for config in TOKENIZER_CONFIGS}
    cells: list[dict[str, Any]] = []
    for spec in DATASETS:
        conversations, eligible = load_conversations(spec, args.samples, args.seed)
        sample_ids = [conversation.sample_id for conversation in conversations]
        for config in TOKENIZER_CONFIGS:
            started = time.perf_counter()
            per_mode = run_config(conversations, renderers[config.name], args.max_turns)
            print(f"{spec.name} / {config.name}: {time.perf_counter() - started:.1f}s", flush=True)
            for mode, (summary, examples) in per_mode.items():
                print(f"  {mode}: {summary}", flush=True)
                cells.append(
                    {
                        "dataset": spec.name,
                        "tokenizer": config.name,
                        "mode": mode,
                        "summary": summary,
                        "examples": examples,
                    }
                )
        (args.out / f"samples_{spec.name}.json").write_text(
            json.dumps(
                {**spec.__dict__, "eligible_in_shard": eligible, "seed": args.seed, "sample_ids": sample_ids},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

    report = {
        "schema_version": 1,
        "settings": {
            "samples_per_dataset": args.samples,
            "seed": args.seed,
            "max_turns": args.max_turns,
            "synthetic_reasoning": SYNTHETIC_REASONING,
            "transformers": transformers.__version__,
            "tokenizers": [config.__dict__ for config in TOKENIZER_CONFIGS],
            "datasets": [spec.__dict__ for spec in DATASETS],
        },
        "cells": cells,
    }
    (args.out / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    (args.out / "summary.md").write_text(markdown_table(cells), encoding="utf-8")
    print(markdown_table(cells))


if __name__ == "__main__":
    main()
