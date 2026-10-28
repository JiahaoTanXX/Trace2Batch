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

据此向 microsoft/agent-lightning 提交了 Issue [#645](https://github.com/microsoft/agent-lightning/issues/645)，提议在训练指标中按原因归因合并失败。

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

这些都是数据层面的对比，没有做训练验证。

## 发现三：真实采样下，"模板前缀稳定"也不够

上面的结论基于模板认可的规范输出。为了补上这个盲区，让 Qwen2.5-0.5B-Instruct 在真实历史上**实际采样**生成每一轮（temperature=1.0），再按 vLLM Hermes 工具解析器解析、重渲染下一轮 prompt。共 300 个轮次，本地 CPU 运行，可复现：

- 在规范输出下 100% 合并的 Qwen2.5 模板，真实采样后，**正常结束的 169 个轮次中只有 60.4% 前缀成立**。
- 主要原因是**非规范 token 切分**（32.5%）：模型采样出 `-t` + `xt`，而同一段文本重新分词得到的是 `-txt`。
- 其次是模板层面的差异（6.5%）：模型把工具调用 JSON 写成带缩进的多行格式而模板重新渲染成紧凑格式、以 `<|endoftext|>` 而非 `<|im_end|>` 结束、正文与模板的换行叠加等。
- 所以只要采用"服务端每轮重渲染 + 重分词"的链路，就算模板本身前缀稳定，trajectory 聚合也会频繁退化；要稳定合并，需要 token-in-token-out。

- **温度消融**：改用模型推荐的采样设置（T=0.7 等）后，前缀成立率升到 88.7%，retoken 降到 4.7%；但模板层面的失败基本不变，两种设置下都约为 6.5%。非规范切分主要是高温采样的问题，工具调用重新格式化则与温度无关。

- **真实 vLLM 服务端（GPU）**：在 Kaggle 的 T4 上，直接请求 `vllm serve`（Hermes 工具解析、`return_token_ids`），prompt 最长 32k，每种配置约 300 个轮次：
  - Qwen2.5-1.5B 前缀成立 **83.3%**（250/300）；
  - Qwen3-0.6B + 补丁 **74.4%**（212/285）；
  - **最主要的失败是"工具调用前插入换行"**（模型按规范格式输出 `正文\n<tool_call>`，Hermes 解析器把末尾的 `\n` 保留在 content 里，模板重新渲染时又插入一个 `\n`，变成 `\n\n`），分别占失败的 44% 和 85%；
  - 1.5B 模型的非规范切分只占 6.3%。
  - 换上补丁 v2 后在相同设置下重跑（见下一节），这类换行失败降为 0。


## 从诊断到修复：模板补丁与 token-in-token-out

针对 Qwen3 和 gpt-oss，各编写了一个**前缀稳定的模板补丁**（`experiments/template_patch/templates/`）：历史消息按生成时的样子渲染。补丁都是官方模板的最小改动。

| 验证 | 官方模板 | 补丁模板 |
|---|---:|---:|
| 规范输出：Qwen3 关闭思考，3 个数据集 | 0% | **100%** |
| 规范输出：gpt-oss，SWE-agent | 0% | **100%** |
| 真实采样：Qwen3-0.6B 关闭思考，150 个轮次 | 0% | **78.7%**（v1）/ **90.0%**（v2） |
| 真实 vLLM：Qwen2.5-1.5B，300 个轮次 | 83.3% | **90.7%**（v2） |
| 真实 vLLM：Qwen3-0.6B 关闭思考，约 300 个轮次 | — | **74.4%**（v1，285 轮）/ **94.7%**（v2，300 轮） |

- 规范输出下，补丁后的训练行与 token-in-token-out **逐数据集完全一致**。
- 提供只用标准库的 **token-in-token-out 参考实现**（`trace2batch/tito.py`）。在 900 条轨迹上，它与服务端重渲染逐 token 相同（12,884 / 12,884 个 prompt），模板改写历史时依然 100% 合并。
- 真实采样下剩余的 21.3%，来自工具调用被重新格式化（例如 `-74.0060` 经 JSON 解析后变成 `-74.006`）和非规范 token 切分。这两类**只有 token-in-token-out 能消除**。
- **补丁 v2**：解析器保留的换行与模板插入的换行叠加，是真实 vLLM 下最主要的失败。v2 只在 content 不以换行结尾时才插入换行，在 CPU 真实采样中消除了全部此类失败：Qwen3-0.6B 前缀成立率 78.3% → **90.2%**，Qwen2.5-0.5B 88.7% → **94.8%**；规范输出仍 100% 合并。在 Kaggle T4 的真实 vLLM 上，以相同设置重跑，换行叠加失败从 23 / 64 个降为 **0**：Qwen2.5-1.5B 83.3% → **90.7%**，Qwen3-0.6B 74.4% → **94.7%**。按轮次配对，Qwen2.5 有 25 个轮次由失败变为成立、3 个反之（采样不同，属于 fp16 运行间差异）；Qwen3 分别为 58 个和 1 个。剩余失败主要是非规范切分和工具调用重新格式化。
- 补丁有意偏离了 Qwen3 官方"历史中不保留思考"的建议，对模型效果的影响尚未验证。


## P3-A：训练成本基准

新增 [训练成本实验](experiments/train_cost/README.md)：从同一份前缀稳定的 token 事件构建逐轮和合并样本，逐目标核对完整上下文；采用相同 LoRA 初始化、每 rollout 一次更新、policy token 归一化和 AB/BA 顺序，测量真实前向、反向及更新成本。提供本地数据预检、限时运行和独立 Kaggle 提交包。

## 项目亮点

- **不匹配归因器**（`trace2batch/mismatch.py`）：定位第一个分歧 token，把失败归为 `template_mismatch` / `retoken_mismatch` / `others_mismatch` 三类，并输出分歧点前后的文本片段供人工核查。只用 Python 标准库，解码函数以回调注入，可以接入任何 tokenizer。
- **可复现的实证研究**（`experiments/template_prefix/`）：数据集和 tokenizer 的 revision 全部固定，抽样种子固定，样本 id 落盘，一条命令即可重跑。重复运行的合并与归因结果完全一致。归因次数与样本构建的不匹配计数逐条交叉校验。
- **三种采集方式对照**：token 直通（对照组）、vLLM token id（每轮服务端重渲染）、文本重分词，可以分离模板改写与分词边界两类因素。
- **样本构建引擎**（`trace2batch/core.py`）：把模型请求、工具观察、奖励事件转成 RL 样本行（prompt / response / policy mask / reward），支持 `transition` 与 `trajectory` 两种表示、三种截断策略、多 rollout 批处理、总长预算、缺失奖励填充、请求过滤计数和"无可用行"原因诊断。零运行时依赖。
- **工程质量**：默认离线测试和可选真实 tokenizer 测试；ruff + pyright 静态检查；GitHub Actions 在 Python 3.12 / 3.13 上持续集成。

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

已知局限：这组实验用的是模板规范输出，真实采样带来的问题见"发现三"；没有覆盖 Llama、Mistral 等家族。

完整的实验报告与结果数据将在项目完成后整理发布；每个实验都可以用下方「快速开始」中的命令在本地重跑，结果写入各实验目录的 `results/`。

## 快速开始

需要 Python >= 3.12。

```bash
pip install -e ".[dev]"
python -m examples.trace2batch.run            # 12 个带手算期望的合成场景

pip install -e ".[real]"
python -m experiments.template_prefix.run     # 前缀一致性研究（首次运行会下载 tokenizer 与数据分片）
python -m experiments.truncation.run          # 截断策略、预算分配与共享前缀成本

pip install -e ".[sampling]"
python -m experiments.real_sampling.run       # 真实模型采样（下载约 1 GB 的 0.5B 模型）
python -m experiments.template_patch.run      # 模板补丁验证
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
│   ├── mismatch.py             # 前缀不匹配定位与原因归因
│   └── tito.py                 # token-in-token-out 轨迹记录器
├── experiments/template_prefix/
│   ├── run.py                  # 真实 tokenizer 前缀一致性实验
│   └── DATA.md                 # 数据说明
├── experiments/real_sampling/  # 真实模型采样实验
├── experiments/template_patch/ # 前缀稳定模板补丁与验证实验
├── experiments/tito/           # TITO 观察分词与规范输出验证
├── experiments/vllm_sampling/  # 真实 vLLM 服务端采样客户端（GPU）
├── experiments/train_cost/    # P3-A 数据配对、masked CE 训练成本与限时提交包
├── experiments/truncation/
│   └── run.py                  # 截断策略对比实验
├── examples/trace2batch/       # 合成场景与手算期望
├── tests/                      # 离线测试 + network 测试
└── .github/workflows/ci.yml
```

## English summary

Trace2Batch is a diagnostic toolkit and empirical study of multi-turn agent trajectory data for agentic RL. Using 5 real tokenizers from 4 model families (Qwen, DeepSeek, GLM, gpt-oss; 9 template configurations) and 900 public agent trajectories, it simulates the OpenAI-compatible proxy + vLLM `return_token_ids` collection path over 107,848 merge attempts. All 54,391 failures are attributed to chat-template rewrites: Qwen3 with `enable_thinking=False` drops the empty `<think>` block from history (0% merged), Qwen3/GLM-4.5 thinking modes lose history reasoning unless it is sent back and observations use the `tool` role, and gpt-oss rewrites `<|return|>` to `<|end|>` after a final answer. Qwen2.5, DeepSeek-V3.1 and GLM-4.5 without thinking merge 100%. When merging fails, rows degrade to one per turn and uncapped token volume grows up to 15.75×. A follow-up with real sampling (Qwen2.5-0.5B-Instruct, T=1.0, 300 turns, vLLM Hermes parsing) shows that even a prefix-stable template breaks in practice: only 60.4% of finished turns keep the prefix, mainly because of non-canonical token splits (32.5%). Minimal prefix-stable template patches for Qwen3 and gpt-oss restore merging from 0% to 100% on canonical outputs, and from 0% to 78.7% under real Qwen3-0.6B sampling. A v2 patch that stops stacking a template newline on the newline the Hermes parser keeps removes every such failure under a real vLLM server (prefix kept: Qwen2.5-1.5B 83.3% → 90.7%, Qwen3-0.6B 74.4% → 94.7%); the remainder (tool-call reformatting such as -74.0060 → -74.006, and non-canonical splits) can only be removed by token-in-token-out. A second study shows that about 80% of the merged response budget is spent on tool observations, and that the binding constraint is the per-row total length. At the same total length, allocating the prompt/response budget dynamically instead of splitting it in half raises policy-token retention under default truncation from 26.3% to 54.6% on SWE-smith (16k total). Splitting at turn boundaries never retains more policy tokens at equal total length, and 62.6%–67.7% of its token overhead is shared prefix. The project ships a dependency-free mismatch classifier (template / retoken / others), a reproducible experiment pipeline, and a zero-dependency sample builder. The findings were reported upstream as microsoft/agent-lightning#645.

## 许可证

MIT，见 [LICENSE](LICENSE)。`trace2batch/core.py` 等部分文件派生自 MIT 许可的开源代码，原版权声明保留在文件头和 LICENSE 中。
