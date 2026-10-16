# Trace2Batch

**Agentic RL 多轮轨迹数据的一致性诊断工具与实证研究。**

多轮 Agent 做 RL 训练时，常见的做法是把一整条轨迹合并成一个训练样本（`trajectory` 聚合），省掉重复历史前缀的计算。合并的前提是：下一轮 prompt 的 token 必须以上一轮的上下文为前缀。训练指标只会告诉你合并失败了多少次，不会告诉你为什么。Trace2Batch 回答的正是这个问题：**在真实的 chat template 下，这个前提到底会不会被悄悄破坏？被谁破坏？代价多大？**

## 核心发现

用真实的 Qwen2.5 / Qwen3 tokenizer、3 个公开 Agent 轨迹数据集的 900 条轨迹，模拟"OpenAI 兼容代理 + vLLM `return_token_ids`"这条常见的采集链路，每种配置跑 11,984 次合并尝试：

| 发现 | 数据 |
|---|---|
| Qwen3 关闭思考模式（`enable_thinking=False`），轨迹合并**全部失败** | 11,984 / 11,984 次失败，合并成功率 **0%** |
| 失败根因定位到模板：历史轮次里的空 `<think>` 块被模板删除 | 全部失败归因为 `template_mismatch`，差异片段逐条可查 |
| 合并失败后，每行都要重复携带完整历史 | 不截断时行内 token 总量膨胀 **15.75×**（SWE-smith）/ **14.75×**（SWE-agent）/ **1.75×**（Toucan） |
| 即使回传 `reasoning_content`，工具观察以 `user` 角色返回时仍然全部失败 | nebius SWE-agent 数据集 6,147 / 6,147 次失败 |
| 同样的数据换成 Qwen2.5 模板 | 3 个数据集 **100%** 合并 |

结论：`trajectory` 聚合能否生效，取决于"模板配置 × 思考内容是否回传 × 工具观察的消息角色"这三者的组合，而不是数据本身。配置不对时，训练数据会退化成逐轮样本，而现有指标只记录失败次数，不给出原因。

据此向 microsoft/agent-lightning 提交了 Issue [#645](https://github.com/microsoft/agent-lightning/issues/645)，提议在训练指标中按原因归因合并失败。完整报告见 [REPORT.md](experiments/template_prefix/REPORT.md)。

## 项目亮点

- **不匹配归因器**（`trace2batch/mismatch.py`）：定位第一个分歧 token，把失败归为 `template_mismatch` / `retoken_mismatch` / `others_mismatch` 三类，并输出分歧点前后的文本片段供人工核查。只用 Python 标准库，解码函数以回调注入，可以接入任何 tokenizer。
- **可复现的实证研究**（`experiments/template_prefix/`）：数据集和 tokenizer 的 revision 全部固定，抽样种子固定，样本 id 落盘，一条命令即可重跑。两次完整运行的合并与归因结果完全一致。归因次数与样本构建的不匹配计数逐条交叉校验。
- **三种采集方式对照**：token 直通（对照组）、vLLM token id（每轮服务端重渲染）、文本重分词，可以分离模板改写与分词边界两类因素。
- **样本构建引擎**（`trace2batch/core.py`）：把模型请求、工具观察、奖励事件转成 RL 样本行（prompt / response / policy mask / reward），支持 `transition` 与 `trajectory` 两种表示、多 rollout 批处理、总长预算、缺失奖励填充、请求过滤计数和"无可用行"原因诊断。零运行时依赖。
- **工程质量**：81 个默认离线测试，加上 2 个使用真实 tokenizer 的联网测试；ruff + pyright 静态检查；GitHub Actions 在 Python 3.12 / 3.13 上持续集成。

## 研究设置一览

| 维度 | 内容 |
|---|---|
| 数据 | SWE-smith（结构化工具调用）、Toucan-1.5M Qwen3 子集（MCP 工具调用）、nebius SWE-agent（文本动作），各 300 条 |
| Tokenizer | Qwen2.5-7B-Instruct；Qwen3-8B 分 3 种思考配置（其中 2 种注入合成思考内容，已明确标注） |
| 采集方式 | A token 直通 / B vLLM token id / C 文本重分词 |
| 规模 | 36 个实验单元，每种 tokenizer 配置 11,984 次合并尝试 |
| 资源 | 纯 CPU，只下载 tokenizer 文件，不需要模型权重 |

典型分歧片段（Qwen3，`enable_thinking=False`）：

```text
上一轮上下文: …<|im_start|>assistant\n<think>\n\n</think>\n\nI'll help you implement the necessary changes…
下一轮 prompt: …<|im_start|>assistant\nI'll help you implement the necessary changes…
归因: template_mismatch，差异 = 删除 "<think>\n\n</think>\n\n"
```

已知局限：没有真实模型采样，所以测不到非规范 token 切分引起的重分词问题；只覆盖 Qwen 系模板。详见报告第 6 节。

## 快速开始

需要 Python >= 3.12。

```bash
pip install -e ".[dev]"
python -m examples.trace2batch.run            # 12 个带手算期望的合成场景

pip install -e ".[real]"
python -m experiments.template_prefix.run     # 真实 tokenizer 研究（首次运行会下载 tokenizer 与数据分片）
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
├── examples/trace2batch/       # 合成场景与手算期望
├── tests/                      # 离线测试 + network 测试
├── drafts/upstream_issue.md    # 已提交上游 Issue 的原文
└── .github/workflows/ci.yml
```

## English summary

Trace2Batch is a diagnostic toolkit and empirical study of multi-turn agent trajectory data for agentic RL. Using real Qwen2.5 / Qwen3 tokenizers and 900 public agent trajectories, it simulates the OpenAI-compatible proxy + vLLM `return_token_ids` collection path. It shows that with Qwen3 and `enable_thinking=False`, all 11,984 trajectory merge attempts fail, because the chat template drops the empty `<think>` block from history. Training rows then degrade to one per turn, and uncapped token volume grows up to 15.75×. Qwen2.5 merges 100%. The project ships a dependency-free mismatch classifier (template / retoken / others), a reproducible experiment pipeline, and a zero-dependency sample builder. The findings were reported upstream as microsoft/agent-lightning#645.

## 许可证

MIT，见 [LICENSE](LICENSE)。`trace2batch/core.py` 等部分文件派生自 MIT 许可的开源代码，原版权声明保留在文件头和 LICENSE 中。
