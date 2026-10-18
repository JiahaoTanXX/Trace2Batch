# 前缀稳定的模板补丁，与 token-in-token-out 的对照

> 数字来自两次完整运行，均可复现：
> - 模板层面：`python -m experiments.template_patch.run`，结果在 `results/`。
> - 真实采样：`python -m experiments.real_sampling.run --model qwen3-0.6b` 和 `--model qwen3-0.6b+patch`（各 `--turns-per-dataset 50`），结果在 `../real_sampling/results/qwen3-0.6b*/`。
>
> 数据、采集方式和归因流程与前缀实验相同，见 [`../template_prefix/REPORT.md`](../template_prefix/REPORT.md)。

## 1. 问题与思路

前缀实验发现，trajectory 合并失败全部来自模板改写历史：

- Qwen3 删除历史中的思考块；
- gpt-oss 把生成时的 `<|return|>` 改写为 `<|end|>`，并在出现最终回答后删除之前工具调用的 analysis。

思路是：**让模板对历史消息的渲染，与这条消息生成时的样子完全一致**。补丁只做最小改动，见 `templates/`，都保留了原模板的来源和 Apache-2.0 版权说明：

| 补丁 | 改动 |
|---|---|
| `qwen3_prefix_stable.jinja` | 历史助手消息始终渲染 `<think>\n{思考}\n</think>\n\n` + 正文，不再根据"是否位于最后一次用户提问之后"删除；没有思考内容时渲染空块，与 `enable_thinking=False` 的生成提示一致 |
| `gpt_oss_prefix_stable.jinja` | 不再删除工具调用轮次的 analysis；历史中的最终回答以 `<|return|>` 结尾，与生成时一致 |

## 2. 模板层面验证（规范输出，B 方式）

| 配置 | 数据集 | 官方模板合并率 | 补丁后合并率 | 补丁使 prompt 与官方不同的轮次 |
|---|---|---:|---:|---:|
| qwen3_no_think | SWE-smith / Toucan / nebius | 0% / 0% / 0% | **100% / 100% / 100%** | 5,416/5,716 · 421/721 · 6,147/6,447 |
| qwen3_think_keep [合成] | SWE-smith / Toucan / nebius | 100% / 100% / 0% | 100% / 100% / **100%** | 0 · 0 · 6,147/6,447 |
| qwen3_think_drop [合成] | SWE-smith / Toucan / nebius | 0% / 0% / 0% | 0% / 0% / 0% | 5,416/5,716 · 421/721 · 6,147/6,447 |
| gpt-oss | SWE-smith / Toucan / nebius | 100% / 100% / 0% | 100% / 100% / **100%** | 0 · 0 · 6,147/6,447 |

- 补丁修复了所有"模板删除或改写了客户端回传内容"的情况。C 方式与 B 方式结果完全相同。
- **think_drop 仍是 0%**：客户端根本没有把思考内容回传，模板无法凭空恢复。这需要修改客户端，而不是模板。
- **补丁只在需要的地方改变 prompt**：官方模板本来就能合并的组合（例如 SWE-smith 上的 think_keep 和 gpt-oss），补丁渲染出的 prompt 与官方完全相同。改变的轮次恰好是"除每条轨迹第一轮以外的所有轮次"。
- **一致性检查**：补丁后 qwen3_no_think 的行 token 总量（SWE-smith 5,684,733、Toucan 856,045、nebius 4,557,486），与前缀实验中 token 直通（A 方式）的对照组**逐数据集完全相等**。也就是说，在规范输出下，"补丁模板 + 服务端重渲染"与 token-in-token-out 产生完全相同的训练行。

## 3. 真实采样验证（Qwen3-0.6B，关闭思考）

在 3 个数据集各 50 个真实轮次上，用 Qwen3-0.6B（Apache-2.0，revision `c1899de`）以 `enable_thinking=False`、T=1.0 采样，流程同真实采样实验。官方模板和补丁模板各跑一次，轮次相同。

| 数据集 | 官方模板：前缀成立 | 补丁模板：前缀成立 | 补丁后剩余失败 |
|---|---:|---:|---|
| SWE-smith | 0/50 | **28/50（56%）** | template 17、retoken 5 |
| Toucan | 0/50 | **45/50（90%）** | template 3、retoken 2 |
| nebius | 0/50 | **45/50（90%）** | retoken 5 |
| 合计 | **0/150（0%）** | **118/150（78.7%）** | template 20、retoken 12 |

官方模板下 150 个轮次全部失败，且全部归为 template：空思考块被删除，在任何切分差异出现之前前缀就已经断开。补丁之后剩下的失败已经与思考块无关：

| 剩余原因 | 次数 | 例子 |
|---|---:|---|
| 工具调用前的换行（template） | 17 | 模型输出 `正文<tool_call>`，模板重新渲染时在正文和工具调用之间插入 `\n` |
| JSON 数字被规范化（template） | 2 | 模型写 `"longitude": -74.0060`、`"b": 28.50`，解析器 `json.loads` 再 `json.dumps` 后变成 `-74.006`、`28.5` |
| 正文中出现 `</think>`（template） | 1 | 关闭思考时模型仍在正文里写出 `</think>`，模板按 `</think>` 切分，丢掉了它前面的文本 |
| 非规范 token 切分（retoken） | 12 | 采样出的 token 序列与同一段文本的规范分词不同 |

## 4. 三种做法的对照

| 做法 | 模板改写（思考块、结束 token） | 工具调用重新格式化（换行、JSON 数字） | 非规范 token 切分 | 代价 |
|---|---|---|---|---|
| 官方模板 + 服务端重渲染（上游现状） | ✗ | ✗ | ✗ | — |
| 补丁模板 + 服务端重渲染 | ✓ | ✗ | ✗ | rollout 和训练都要换模板，并偏离官方推荐（见第 5 节） |
| **token-in-token-out**（沿用上一轮的 token id，只对新消息分词） | ✓ | ✓ | ✓ | 需要框架在 rollout 侧维护 token 序列，不能只依赖 OpenAI 兼容的文本接口 |

- token-in-token-out 下，下一轮 prompt 的定义就是"上一轮上下文 + 新消息的 token"，所以**按构造 100% 合并**，不需要测量。真正的问题在于：它让训练行与模型在 rollout 中实际看到的 token 完全一致；而重渲染链路下，模型在下一轮看到的是重新分词后的上下文，两者本来就不同。
- 实测：Qwen3-0.6B 加补丁后，合并率从 0% 提升到 78.7%。剩下的 21.3% 来自工具调用重新格式化（13.3%）和非规范切分（8.0%），这两类**只有 token-in-token-out 能消除**。
- 结论：模板补丁适合"只能用 OpenAI 兼容接口"时的低成本缓解；要真正稳定合并，应该采用 token-in-token-out。

## 5. 局限性与风险

- **偏离官方推荐。** Qwen3 模型卡明确建议历史中不包含思考内容（"No Thinking Content in History"）。补丁有意偏离这一点，换取 rollout 与训练的一致性。gpt-oss 补丁则把 harmony 格式中"不应作为输入"的 `<|return|>` 放进了历史。两者都要求 rollout（例如 vLLM 的 `--chat-template`）和训练使用同一个补丁模板。**对模型效果的影响没有验证**，需要 GPU 上的训练实验。
- 真实采样只用了 0.6B 模型、T=1.0，每个数据集 50 个轮次，并跳过了 prompt 超过 4,096 token 的轮次。剩余失败的比例不能直接推广到大模型。
- gpt-oss 补丁只做了模板层面的验证，没有真实采样。没有采用 gpt-oss-20b，是因为它超出了 1.5B 的权重下载范围。
- think_drop 场景需要修改客户端，补丁无能为力。
