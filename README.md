# Trace2Batch

中文 | [English](README.en.md)

[![CI](https://github.com/JiahaoTanXX/Trace2Batch/actions/workflows/ci.yml/badge.svg)](https://github.com/JiahaoTanXX/Trace2Batch/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Agentic RL 多轮轨迹数据的一致性诊断、修复与实证研究。** 研究范围从 chat template、vLLM 服务端一直到训练成本，用 900 条公开 Agent 轨迹、超过 10 万次合并尝试和真实 GPU 运行，回答每个采用 trajectory 聚合的多轮 Agent RL 训练栈都要面对的问题：

> 把一整条轨迹合并成一个训练样本（`trajectory` 聚合）能省掉大量重复计算，前提是下一轮 prompt 的 token 必须以上一轮上下文为前缀。**这个前提在真实链路里会不会被悄悄破坏？被谁破坏？破坏后训练要多付多少代价？怎么彻底修好？**

常见的训练框架（例如 agent-lightning）只报告“合并失败了多少次”。Trace2Batch 能定位到第一个分歧 token，说清楚失败的原因，并给出经过真实服务端验证的修复方案。

## 主要结果

| | 结果 |
|---|---|
| 🔍 **诊断** | 107,848 次合并尝试中有 54,391 次失败，**全部**定位到 chat template 对历史的改写。Qwen3 关闭思考时合并率为 **0%**，训练 token 膨胀到 **15.75×** |
| 🧪 **真实采样** | 即使模板本身前缀稳定，真实采样下也只有 60.4% 的轮次前缀成立，主要原因是非规范 token 切分 |
| 🩹 **模板补丁** | 前缀稳定补丁让合并率从 **0% → 100%**（规范输出）；补丁 v2 在真实 vLLM 上消除了全部换行叠加失败，前缀成立率 74.4% → **94.7%** |
| ✅ **TITO 根治** | 真实 vLLM 多轮 rollout 中，token-in-token-out 让 **528 / 528** 条轨迹完整合并；服务端重渲染只有 63.6% / 77.3% |
| 🔬 **新发现** | TITO 并非零代价：模型会延续自己在上下文中的非规范切分。此前已出现过非规范切分的轨迹里，后续再出现的比例比重渲染驱动高 **10.5 / 24.8 个百分点**（95% 置信区间都大于 0） |
| ⏱️ **训练代价** | 真实 LoRA 训练中，合并失败让训练慢 **1.5–4.2×**，8 轮轨迹达 **4.0–5.2×**，差距几乎完全由 token 膨胀决定（r = 0.987） |
| ✂️ **截断** | 同一单行总长下，改用动态预算分配后，policy token 保留率从 26.3% 提升到 **54.6%** |
| 🌐 **上游反馈** | 向 microsoft/agent-lightning 提交 Issue [#645](https://github.com/microsoft/agent-lightning/issues/645)，维护者确认内部实验中也观察到同样的现象 |

## 1. 诊断：chat template 在悄悄破坏前缀

用 **4 个模型家族**（Qwen、DeepSeek、GLM、gpt-oss）的 5 个真实 tokenizer、9 种模板配置，以及 3 个公开 Agent 轨迹数据集的 900 条轨迹，模拟“OpenAI 兼容代理 + vLLM `return_token_ids`”这条主流采集链路，共 **107,848 次合并尝试**：

| 配置 | SWE-smith | Toucan | SWE-agent（nebius） | 失败根因 |
|---|---|---|---|---|
| Qwen2.5 / DeepSeek-V3.1 / GLM-4.5 关闭思考 | 100% | 100% | 100% | —（前缀稳定的模板） |
| **Qwen3 关闭思考** | **0%** | **0%** | **0%** | 历史中删除空 `<think>` 块 |
| Qwen3 / GLM-4.5 思考，回传 reasoning [合成] | 100% | 100% | **0%** | 工具观察以 `user` 角色出现，历史思考仍被删除或清空 |
| Qwen3 / GLM-4.5 思考，不回传 reasoning [合成] | **0%** | **0%** | **0%** | 历史中思考被删除或清空 |
| **gpt-oss** | 100% | 100% | **0%** | 生成结束 token `<|return|>` 在历史中变为 `<|end|>` |

（表中为合并成功率。）

- **54,391 次失败 100% 归因为 `template_mismatch`**，每一处差异片段都可以逐条查看。
- 合并一旦失败，就是每一轮都失败，样本退化成逐轮，每行都要重复携带完整历史。不截断时，行内 token 总量膨胀到 **15.75×**（SWE-smith）、**14.75×**（SWE-agent）、**1.75×**（Toucan）。
- 另外还发现模板对数据格式很脆弱：gpt-oss 模板遇到 `description` 为 `null` 的工具定义会报错，GLM-4.5 模板遇到不合法的 JSON 参数会报错，gpt-oss 每条消息只渲染第一个工具调用。

**结论**：`trajectory` 聚合能否生效，取决于“模板 × 思考内容是否回传 × 工具观察的消息角色 × 是否出现最终回答”的组合，而不是数据本身。配置一旦不对，训练数据就会悄悄退化成逐轮样本，而现有指标看不出原因。

典型分歧片段（Qwen3，`enable_thinking=False`）：

```text
上一轮上下文: …<|im_start|>assistant\n<think>\n\n</think>\n\nI'll help you implement the necessary changes…
下一轮 prompt: …<|im_start|>assistant\nI'll help you implement the necessary changes…
归因: template_mismatch，差异 = 删除 "<think>\n\n</think>\n\n"
```

## 2. 真实采样：“模板前缀稳定”也不够

规范输出只是理想情况。让模型在真实历史上**实际采样**每一轮，再按 vLLM Hermes 工具解析器解析、重渲染下一轮 prompt：

- **CPU 复现**（Qwen2.5-0.5B-Instruct，T=1.0，300 个轮次）：在规范输出下 100% 合并的 Qwen2.5 模板，真实采样后只有 **60.4%** 的正常结束轮次前缀成立。
  - 主因是**非规范 token 切分**（32.5%）：模型采样出 `-t` + `xt`，而同一段文本重新分词得到的是 `-txt`。
  - 其次是模板层面的改写（6.5%）：工具调用 JSON 被重新格式化、以 `<|endoftext|>` 结束、换行叠加。
  - **温度消融**：换成推荐采样设置（T=0.7）后，前缀成立率升到 88.7%，但模板层面的失败基本不变（约 6.5%）。非规范切分主要是高温采样的问题，重新格式化则与温度无关。
- **真实 vLLM 服务端**（Kaggle T4，`vllm serve` + Hermes 解析 + `return_token_ids`，prompt 最长 32k）：Qwen2.5-1.5B 前缀成立率 83.3%（250/300），Qwen3-0.6B + 补丁 v1 为 74.4%（212/285）。
  - **最主要的失败是一个解析器与模板之间的交互缺陷**：模型按规范格式输出 `正文\n<tool_call>`，Hermes 解析器把末尾的 `\n` 保留在 content 里，模板重新渲染时又插入一个 `\n`，叠加成 `\n\n`。它分别占失败的 44% 和 85%。

**结论**：只要采用“服务端每轮重渲染 + 重新分词”的链路，就算模板本身前缀稳定，trajectory 聚合也会频繁退化。

## 3. 修复：前缀稳定的模板补丁

为 Qwen3、gpt-oss 和 Qwen2.5 编写了**前缀稳定的模板补丁**（`experiments/template_patch/templates/`），全部是官方模板的最小改动：历史消息按生成时的样子渲染。补丁 v2 只在 content 不以换行结尾时才插入换行，修复了上面的换行叠加缺陷。

| 验证 | 官方模板 | 补丁模板 |
|---|---:|---:|
| 规范输出：Qwen3 关闭思考，3 个数据集 | 0% | **100%** |
| 规范输出：gpt-oss，SWE-agent | 0% | **100%** |
| 真实采样（CPU）：Qwen3-0.6B 关闭思考，150 个轮次 | 0% | 78.7%（v1）→ **90.0%**（v2） |
| 真实 vLLM：Qwen2.5-1.5B，300 个轮次 | 83.3% | **90.7%**（v2） |
| 真实 vLLM：Qwen3-0.6B 关闭思考 | — | 74.4%（v1，285 轮）→ **94.7%**（v2，300 轮） |

- 真实 vLLM 下，补丁 v2 让换行叠加失败从 23 / 64 次降为 **0**。逐轮配对后，Qwen2.5 有 25 个轮次由失败变为成立、3 个反之（这 3 个都是两次运行的采样本身不同），Qwen3 分别为 58 个和 1 个。
- 规范输出下，补丁后的训练行与 token-in-token-out **逐数据集完全一致**。
- 剩余的失败（非规范切分、`-74.0060` → `-74.006` 这类工具调用重新格式化）只要服务端还在重渲染，就无法通过模板消除。

## 4. 根治：TITO 在真实 vLLM 多轮 rollout 上端到端验证

token-in-token-out（TITO）的做法是历史一律沿用已有的 token id，只对新增消息分词。项目提供只用标准库的参考实现（`trace2batch/tito.py`），并在 **真实 vLLM 0.32 服务端**上做了端到端验证（`experiments/tito_vllm/`）：

- **rollout 由 TITO 驱动**：`/v1/completions` 直接接收 token id，观察 token 由服务端 `/tokenize` 渲染。每条轨迹从第一个助手轮次开始，最多 8 轮，每轮由模型采样（T=1.0）。
- **配对比较**：同一份采样再交给“服务端重渲染 + 官方模板”和“服务端重渲染 + 补丁 v2”两条链路，三条链路逐轮比较，排除了运行间随机性的干扰。

| 整条轨迹可完整合并 | **TITO** | 重渲染 + 官方模板 | 重渲染 + 补丁 v2 |
|---|---:|---:|---:|
| Qwen2.5-1.5B（264 条轨迹，1,360 轮） | **100%** | 62.1% | 63.6% |
| Qwen3-0.6B 关闭思考（264 条轨迹，1,482 轮） | **100%** | 0% | 77.3% |

- **单轮 85–89% 的前缀成立率，到了多轮就会累积成大量整条轨迹失败。** 只有 TITO 能让每条轨迹稳定地合并成 1 个训练行。
- 重渲染链路每失败一次，就多出一个携带完整历史的训练行。训练 token 总量为 TITO 的 **1.07–1.80×**（补丁 v2），Qwen3 官方模板最高 **5.0×**。
- 归因非常干净：补丁 v2 下剩余的 retoken 失败，分歧点**全部**落在模型本轮的采样内（Qwen2.5 154 / 154，Qwen3 121 / 121）。
- 客户端的 Hermes 解析经过逐条核对：两次采样相同的 475 次中，474 次与 vLLM 自带的解析结果一致。
- 此前的离线验证：在 900 条轨迹的规范输出上，TITO 与服务端重渲染逐 token 相同（12,884 / 12,884 个 prompt）。
- **推断经过实测验证**：在相同的轨迹上改由服务端重渲染真实驱动 rollout，实测的整条轨迹完整合并率为 63.0%（Qwen2.5，300 条）和 77.3%（Qwen3，264 条），与上表的配对推断相差不超过 2 个百分点。补齐 SWE-smith 后，TITO 在 Qwen2.5 的 300 / 300、Qwen3 的 299 / 299 条轨迹上全部完整合并。

### TITO 的代价：模型会延续自己的非规范切分

全量运行中有一个现象：在已出现过非规范切分的轨迹里，后续轮次再出现的比例明显更高。它可能来自内容本身，也可能是因为 TITO 把模型自己的非规范 token 留在了上下文里。为了区分这两种解释，我们在相同的轨迹、相同的种子规则下，又用服务端重渲染驱动跑了一遍。重渲染每轮都会把上下文重新规范分词，模型看不到自己的非规范 token。判定规则在运行前就已固定。

| 每轮非规范切分率 | 此前未出现 | 此前已出现 |
|---|---:|---:|
| Qwen2.5-1.5B，TITO 驱动 | 11.5% | **25.7%** |
| Qwen2.5-1.5B，重渲染驱动 | 11.0% | 15.2% |
| Qwen3-0.6B，TITO 驱动 | 4.9% | **29.2%** |
| Qwen3-0.6B，重渲染驱动 | 4.7% | 4.4% |

- “此前已出现”条件下，TITO 减重渲染的差值为 **+10.5 个百分点 [+4.6, +16.6]**（Qwen2.5）和 **+24.8 个百分点 [+14.5, +35.7]**（Qwen3）。这是按轨迹 bootstrap 得到的 95% 置信区间，两个都整体大于 0。
- 对照组干净：“此前未出现”条件下，两种驱动方式的比率几乎相同（差值为 +0.4 和 +0.1 个百分点，置信区间都包含 0）。Qwen3 在重渲染驱动下完全没有聚集（4.7% → 4.4%）。
- **结论**：TITO 保证了训练与推理逐 token 一致，代价是模型会看到并延续自己的非规范切分，采样分布因此改变。重渲染没有这个效应，但会带来训练与推理不一致和合并失败。本项目在真实服务端上量化了多轮 Agent RL 采集链路中的这一权衡；它在 RL 训练中会不会被学到、会不会放大，是下一步值得研究的问题。

## 5. 代价：合并失败让训练慢多少（P3-A）

[训练成本基准](experiments/train_cost/README.md)在真实训练循环中测量两种表示的成本（Kaggle T4，Qwen3-0.6B + LoRA，masked CE）：

- 两种表示来自**同一份** token 事件，逐个监督目标核对完整上下文；
- 正式计时前，先核对两种表示的 loss 和全部可训练参数的梯度是否一致；
- 采用相同的 LoRA 初始化、每条 rollout 只更新一次、按 policy token 归一化，以及 AB/BA 交替顺序。

- 退化成逐轮表示时，训练耗时是合并表示的 **1.5–4.2×**（按格），单条 rollout 最高 **5.1×**。
- **轮数是决定性因素**：长度固定时，加速比随轮数单调上升，从 2 轮的约 1.0–1.7× 到 8 轮的约 **4.0–5.2×**，两批独立样本结果一致。
- 加速比与处理 token 的比值高度相关（r = **0.987**），实际约为 token 比的 0.82 倍。
- 两种表示的峰值显存基本相同：合并省下的是**时间**，而不是显存。

## 6. 截断：真正的约束是单行总长

合并后的 response 中**约 80% 是工具观察**，长度预算主要被观察占满；超出时，默认的尾部截断会丢掉离奖励最近的后期生成。为此新增了 `truncation_strategy`（`truncate` / `split_on_budget` / `drop`）和 `max_total_length`：

| 数据集 | 单行总长 | policy token 保留率（固定对半 → 动态分配） | 最后一次生成丢失的轨迹（固定 → 动态） |
|---|---:|---|---|
| SWE-smith | 16,384 | 26.3% → **54.6%** | 75.0% → **41.3%** |
| SWE-smith | 32,768 | 57.2% → **83.0%** | 36.7% → **14.3%** |
| SWE-agent | 16,384 | 55.0% → **73.5%** | 54.0% → **36.3%** |
| SWE-agent | 32,768 | 77.8% → **99.4%** | 31.7% → **2.0%** |

- 推导并验证：在相同总长下，按轮次切分保留的 policy token **从不多于**截断，因为位置超过单行总长的 token，任何一行都装不下。
- 切分的价值在于行在轮次边界结束（“行尾只剩观察”的比例为 0%）。它带来的 token 膨胀中 62.6%–67.7% 是重复前缀；如果共享前缀只算一次，每个保留的 policy token 只需 6.4–6.5 个 token，低于或接近截断的 6.6–17.4。

## 上游反馈

基于第 1 节的结果，向 [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning) 提交了 Issue [#645](https://github.com/microsoft/agent-lightning/issues/645)，指出合并失败的原因字段恒为 `False`，并附上复现数据与归因方案。维护者回复确认，他们在早期实验中也观察到了同样的现象，并因此在 coding agent 示例中提供了专门的 chat template。

## 工具箱

- **不匹配归因器**（`trace2batch/mismatch.py`）：定位第一个分歧 token，把失败归为 `template_mismatch` / `retoken_mismatch` / `others_mismatch`，并输出分歧点前后的文本片段供人工核查。只用 Python 标准库；解码函数以回调注入，可以接入任何 tokenizer。
- **TITO 记录器**（`trace2batch/tito.py`）：零依赖，产出的事件可以直接交给样本构建器。
- **样本构建引擎**（`trace2batch/core.py`）：把模型请求、工具观察、奖励事件转成 RL 样本行（prompt / response / policy mask / reward），支持：
  - `transition` / `trajectory` 两种表示和三种截断策略；
  - 多 rollout 批处理和总长预算；
  - 缺失奖励填充和“无可用行”原因诊断。

  零运行时依赖。
- **真实服务端客户端**（`experiments/vllm_sampling/`、`experiments/tito_vllm/`）：直接对接 `vllm serve`，覆盖单轮采样，以及多轮 TITO 与重渲染的配对比较。
- **训练成本基准**（`experiments/train_cost/`）：先核对梯度一致，再计时的真实 LoRA 训练循环，附限时运行和 Kaggle 提交流程。

## 实验设置与可复现性

| 维度 | 内容 |
|---|---|
| 数据 | SWE-smith（结构化工具调用）、Toucan-1.5M Qwen3 子集（MCP 工具调用）、nebius SWE-agent（文本动作） |
| Tokenizer | Qwen2.5、Qwen3、DeepSeek-V3.1、GLM-4.5、gpt-oss，共 9 种模板配置（合成思考内容的配置已明确标注） |
| 采集方式 | token 直通（对照组）/ vLLM token id（每轮服务端重渲染）/ 文本重分词，可以分离模板改写与分词边界两类因素 |
| 真实模型 | Qwen2.5-0.5B / 1.5B-Instruct、Qwen3-0.6B，revision 固定 |
| 硬件 | 诊断与截断实验只用 CPU；真实服务端与训练实验用 Kaggle T4 |

- 数据集与 tokenizer 的 revision、抽样种子全部固定，样本 id 落盘，一条命令即可重跑；重复运行的合并与归因结果完全一致。
- 归因次数与样本构建的不匹配计数逐条交叉校验。
- 默认离线测试 + 可选真实 tokenizer 测试；ruff + pyright 静态检查；GitHub Actions 在 Python 3.12 / 3.13 上做持续集成。
- 完整的实验报告与结果数据将在项目完成后整理发布；每个实验都可以用下方的命令在本地重跑，结果写入各实验目录下的 `results/`。

## 局限

- 第 1 节用的是模板规范输出，没有覆盖 Llama、Mistral 等模型家族。
- 真实采样实验只用了 0.5B–1.5B 的小模型，在 T4 上以 fp16 运行，每个配置只运行一次。
- 多轮 rollout 中的工具观察来自数据集，属于离策略。非规范切分聚集的比较是观察性的：两种驱动方式的采样在第一个分歧点之后就不再配对。Qwen3 的重渲染驱动在 SWE-smith 上只完成了 64 条轨迹。
- 补丁有意偏离了 Qwen3 官方“历史中不保留思考”的建议，对模型效果的影响尚未验证。
- 训练成本基准是 masked CE + LoRA 的参考训练循环，不是 GRPO 或全参数训练；截断实验是数据层面的对比。

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

# 需要一个已启动的 vLLM 服务（GPU；vllm serve ... --enable-auto-tool-choice --tool-call-parser hermes）
python -m experiments.vllm_sampling.run --model qwen3-0.6b+patch2 --base-url http://127.0.0.1:8000   # 单轮采样
python -m experiments.tito_vllm.run --model qwen3-0.6b --base-url http://127.0.0.1:8000             # 多轮 TITO（服务端另加 --trust-request-chat-template）
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

TITO 记录：

```python
from trace2batch.tito import TokenInTokenOut

tito = TokenInTokenOut(initial_prompt_ids)
tito.record_response(sampled_ids)  # 生成的 token 原样接入上下文
tito.add_observation(observation_ids)  # 只对新增消息分词
events = tito.events()  # 可直接交给 build_samples
```

## 样本构建语义

| 功能 | 实现方式 |
|---|---|
| 逐轮表示 `transition` | 每次有效模型响应形成一行，生成 token 的 mask 为 1 |
| 连续轨迹表示 `trajectory` | 按完整 token 前缀合并；前缀不连续时分行 |
| 工具观察 | 作为上下文保留在合并响应中，mask 为 0 |
| 长度边界 | 响应与 mask 同步截断；可选 `max_total_length` 总长预算；过长的 prompt 截断并标记丢弃 |
| 截断策略 | `truncate`（默认）/ `split_on_budget`（按轮次边界另起一行）/ `drop`（整行丢弃） |
| 批处理 `build_batch` | 每行带 `rollout_id`，rollout 之间不合并，并汇总 rollout 级诊断 |
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
├── experiments/real_sampling/  # 真实模型采样实验（CPU）
├── experiments/template_patch/ # 前缀稳定模板补丁与验证实验
├── experiments/tito/           # TITO 观察分词与规范输出验证
├── experiments/vllm_sampling/  # 真实 vLLM 服务端采样客户端（GPU）
├── experiments/tito_vllm/      # 真实 vLLM 多轮 rollout：TITO 与重渲染配对比较（GPU）
├── experiments/train_cost/     # P3-A 训练成本基准：数据配对、masked CE 与限时运行
├── experiments/truncation/
│   └── run.py                  # 截断策略对比实验
├── examples/trace2batch/       # 合成场景与手算期望
├── tests/                      # 离线测试 + network 测试
└── .github/workflows/ci.yml
```

## 许可证

MIT，见 [LICENSE](LICENSE)。`trace2batch/core.py` 等部分文件派生自 MIT 许可的开源代码，原版权声明保留在文件头和 LICENSE 中。
