# Trace2Batch

**Agentic RL 多轮轨迹数据的一致性诊断工具与实证研究。**

多轮 Agent 做 RL 训练时，常见的做法是把一整条轨迹合并成一个训练样本（`trajectory` 聚合），省掉重复历史前缀的计算。合并的前提是：下一轮 prompt 的 token 必须以上一轮的上下文为前缀。训练指标只会告诉你合并失败了多少次，不会告诉你为什么。Trace2Batch 回答的正是这个问题：**在真实的 chat template 下，这个前提到底会不会被悄悄破坏？被谁破坏？代价多大？**

## 核心发现

用 **4 个模型家族**（Qwen、DeepSeek、GLM、gpt-oss）的 5 个真实 tokenizer、9 种模板配置，以及 3 个公开 Agent 轨迹数据集的 900 条轨迹，模拟"OpenAI 兼容代理 + vLLM `return_token_ids`"这条常见的采集链路，共 107,848 次合并尝试：

| 配置 | SWE-smith | Toucan | SWE-agent（nebius） | 失败根因 |
|---|---|---|---|---|
| Qwen2.5 / DeepSeek-V3.1 / GLM-4.5 关闭思考 | 100% | 100% | 100% | —（前缀稳定的模板） |
| **Qwen3 关闭思考** | **0%** | **0%** | **0%** | 历史中删除空 `<think>` 块 |
| Qwen3 / GLM-4.5 思考，回传 reasoning [合成] | 100% | 100% | **0%** | 工具观察以 `user` 角色出现，历史思考仍被删除或清空 |
| Qwen3 / GLM-4.5 思考，不回传 reasoning [合成] | **0%** | **0%** | **0%** | 历史中思考被删除或清空 |
| **gpt-oss** | 100% | 100% | **0%** | 生成结束 token `<|return|>` 在历史中变为 `<|end|>` |

（表中为合并成功率。）

- 54,391 次合并失败**全部被归因为 `template_mismatch`**，差异片段逐条可查。
- 合并失败时，每一轮都失败，样本退化成逐轮，每行都要重复携带完整历史。不截断时，行内 token 总量膨胀 **15.75×**（SWE-smith）、**14.75×**（SWE-agent）、**1.75×**（Toucan）（Qwen3 关闭思考，对照组为 token 直通）。
- 渲染时还发现模板对数据格式很脆弱：gpt-oss 模板遇到 `description` 为 `null` 的工具定义会报错，GLM-4.5 模板遇到不合法的 JSON 参数会报错；gpt-oss 每条消息只渲染第一个工具调用。

结论：`trajectory` 聚合能否生效，取决于"模板 × 思考内容是否回传 × 工具观察的消息角色 × 是否出现最终回答"的组合，而不是数据本身。配置不对时，训练数据会退化成逐轮样本，而现有指标只记录失败次数，不给出原因。

据此向 microsoft/agent-lightning 提交了 Issue [#645](https://github.com/microsoft/agent-lightning/issues/645)，提议在训练指标中按原因归因合并失败。完整报告见 [REPORT.md](experiments/template_prefix/REPORT.md)。

## 发现二：真正的约束是单行总长，而不是截断方式

同一批真实轨迹的合并 response 中，**约 80% 是工具观察**，长度预算主要被观察占用。超出预算时，默认的尾部截断会丢掉最接近奖励的后期生成。

为此新增 `truncation_strategy` 参数（`truncate` / `split_on_budget` / `drop`），并做了两组对比：

**1. 在相同的单行总长下，把 prompt / response 固定对半分改成动态分配，默认截断的保留率显著提高**（`max_total_length`）：

| 数据集 | 单行总长 | policy token 保留率（固定对半 → 动态分配） | 最后一次生成丢失的轨迹（固定 → 动态） |
|---|---:|---|---|
| SWE-smith | 16,384 | 26.3% → **54.6%** | 75.0% → **41.3%** |
| SWE-smith | 32,768 | 57.2% → **83.0%** | 36.7% → **14.3%** |
| SWE-agent | 16,384 | 55.0% → **73.5%** | 54.0% → **36.3%** |
| SWE-agent | 32,768 | 77.8% → **99.4%** | 31.7% → **2.0%** |

**2. 在相同总长下，按轮次切分（split）保留的 policy token 从不多于截断。** 这与推导一致：位置超过单行总长的 token，任何一行都装不下。split 的价值是行在轮次边界结束（"行尾只剩观察"的比例为 0%）。它在更大单行总长下的 token 膨胀中，62.6%–67.7% 是重复前缀；按共享前缀只算一次计算，每个保留的 policy token 需要 6.4–6.5 个 token，低于或接近截断的 6.6–17.4。

这些都是数据层面的对比，没有做训练验证。详见 [截断实验报告](experiments/truncation/REPORT.md)。

## 项目亮点

- **不匹配归因器**（`trace2batch/mismatch.py`）：定位第一个分歧 token，把失败归为 `template_mismatch` / `retoken_mismatch` / `others_mismatch` 三类，并输出分歧点前后的文本片段供人工核查。只用 Python 标准库，解码函数以回调注入，可以接入任何 tokenizer。
- **可复现的实证研究**（`experiments/template_prefix/`）：数据集和 tokenizer 的 revision 全部固定，抽样种子固定，样本 id 落盘，一条命令即可重跑。重复运行的合并与归因结果完全一致。归因次数与样本构建的不匹配计数逐条交叉校验。
- **三种采集方式对照**：token 直通（对照组）、vLLM token id（每轮服务端重渲染）、文本重分词，可以分离模板改写与分词边界两类因素。
- **样本构建引擎**（`trace2batch/core.py`）：把模型请求、工具观察、奖励事件转成 RL 样本行（prompt / response / policy mask / reward），支持 `transition` 与 `trajectory` 两种表示、三种截断策略、多 rollout 批处理、总长预算、缺失奖励填充、请求过滤计数和"无可用行"原因诊断。零运行时依赖。
- **工程质量**：103 个默认离线测试，加上 3 个使用真实 tokenizer 的联网测试；ruff + pyright 静态检查；GitHub Actions 在 Python 3.12 / 3.13 上持续集成。

## 研究设置一览

| 维度 | 内容 |
|---|---|
| 数据 | SWE-smith（结构化工具调用）、Toucan-1.5M Qwen3 子集（MCP 工具调用）、nebius SWE-agent（文本动作），各 300 条 |
| Tokenizer | Qwen2.5-7B-Instruct、Qwen3-8B、DeepSeek-V3.1、GLM-4.5、gpt-oss-20b，共 9 种配置（含合成思考内容的配置已明确标注） |
| 采集方式 | A token 直通 / B vLLM token id / C 文本重分词 |
| 规模 | 66 个实验单元，每种配置约 11,984 次合并尝试，共 107,848 次 |
| 资源 | 纯 CPU，只下载 tokenizer 文件，不需要模型权重 |

典型分歧片段（Qwen3，`enable_thinking=False`）：

```text
上一轮上下文: …<|im_start|>assistant\n<think>\n\n</think>\n\nI'll help you implement the necessary changes…
下一轮 prompt: …<|im_start|>assistant\nI'll help you implement the necessary changes…
归因: template_mismatch，差异 = 删除 "<think>\n\n</think>\n\n"
```

已知局限：没有真实模型采样，所以测不到非规范 token 切分引起的重分词问题；没有覆盖 Llama、Mistral 等家族。详见报告第 6 节。

## 快速开始

需要 Python >= 3.12。

```bash
pip install -e ".[dev]"
python -m examples.trace2batch.run            # 12 个带手算期望的合成场景

pip install -e ".[real]"
python -m experiments.template_prefix.run     # 前缀一致性研究（首次运行会下载 tokenizer 与数据分片）
python -m experiments.truncation.run          # 截断策略、预算分配与共享前缀成本
```

样本构建：

```python
from trace2batch.core import build_batch, build_samples

events = [{"event_type": "model_request", "data": {"prompt_token_ids": [1], "response_token_ids": [10, 11]}}]
result = build_samples(events, mode="trajectory", max_prompt_length=16, max_response_length=16)
print(result.summary())

batch = build_batch({"r1": events, "r2": events}, mode="trajectory", max_prompt_length=16, max_response_length=16)
print([item.rollout_id for item in batch.rows], batch.summary()["n_rollouts"])
```

不匹配归因：

```python
from trace2batch.mismatch import explain_mismatch

report = explain_mismatch(
    prev_context_ids, next_prompt_ids, decode=lambda ids: tok.decode(ids, skip_special_tokens=False)
)
print(report.kind, report.changed_segments)
```

## 样本构建语义

| 功能 | 实现方式 |
|---|---|
| 逐轮表示 `transition` | 每次有效模型响应形成一行，生成 token 的 mask 为 1 |
| 连续轨迹表示 `trajectory` | 按完整 token 前缀合并；前缀不连续时分行 |
| 工具观察 | 作为上下文保留在合并响应中，mask 为 0 |
| 长度边界 | 响应与 mask 同步截断；可选 `max_total_length` 总长预算；过长 prompt 截断并标记丢弃 |
| 截断策略 | `truncate`（默认）/ `split_on_budget`（按轮次边界另起一行）/ `drop`（整行丢弃） |
| 批处理 `build_batch` | 每行带 `rollout_id`，rollout 之间不合并，汇总 rollout 级诊断 |
| 奖励 | 取最后一个 reward，区分零奖励与缺失奖励，缺失时按 `reward_fillna_value` 填充 |

三轮工具调用合并后的 response 与 mask：

```text
response: [10, 90, 11, 91, 92, 12, 13]
mask:     [ 1,  0,  1,  0,  0,  1,  1]
```

## 测试

```bash
python -m pytest -q                 # 默认离线
python -m pytest -q --run-network   # 额外运行真实 tokenizer 测试
ruff check . && ruff format --check . && pyright
```

## 目录结构

```text
Trace2Batch/
├── trace2batch/
│   ├── core.py                 # 样本构建、批处理与诊断
│   └── mismatch.py             # 前缀不匹配定位与原因归因
├── experiments/template_prefix/
│   ├── run.py                  # 真实 tokenizer 前缀一致性实验
│   ├── DATA.md / REPORT.md     # 数据说明与研究报告
│   └── results/                # 结果 JSON、汇总表与样本 id
├── experiments/truncation/
│   ├── run.py                  # 截断策略对比实验
│   ├── REPORT.md               # 截断实验报告
│   └── results/
├── examples/trace2batch/       # 合成场景与手算期望
├── tests/                      # 离线测试 + network 测试
├── drafts/upstream_issue.md    # 已提交上游 Issue 的原文
└── .github/workflows/ci.yml
```

## English summary

Trace2Batch is a diagnostic toolkit and empirical study of multi-turn agent trajectory data for agentic RL. Using 5 real tokenizers from 4 model families (Qwen, DeepSeek, GLM, gpt-oss; 9 template configurations) and 900 public agent trajectories, it simulates the OpenAI-compatible proxy + vLLM `return_token_ids` collection path over 107,848 merge attempts. All 54,391 failures are attributed to chat-template rewrites: Qwen3 with `enable_thinking=False` drops the empty `<think>` block from history (0% merged), Qwen3/GLM-4.5 thinking modes lose history reasoning unless it is sent back and observations use the `tool` role, and gpt-oss rewrites `<|return|>` to `<|end|>` after a final answer. Qwen2.5, DeepSeek-V3.1 and GLM-4.5 without thinking merge 100%. When merging fails, rows degrade to one per turn and uncapped token volume grows up to 15.75×. A second study shows that about 80% of the merged response budget is spent on tool observations, and that the binding constraint is the per-row total length. At the same total length, allocating the prompt/response budget dynamically instead of splitting it in half raises policy-token retention under default truncation from 26.3% to 54.6% on SWE-smith (16k total). Splitting at turn boundaries never retains more policy tokens at equal total length, and 62.6%–67.7% of its token overhead is shared prefix. The project ships a dependency-free mismatch classifier (template / retoken / others), a reproducible experiment pipeline, and a zero-dependency sample builder. The findings were reported upstream as microsoft/agent-lightning#645.

## 许可证

MIT，见 [LICENSE](LICENSE)。`trace2batch/core.py` 等部分文件派生自 MIT 许可的开源代码，原版权声明保留在文件头和 LICENSE 中。
