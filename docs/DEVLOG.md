# 开发日志

每次改动记录**做了什么、为什么、怎么做**，以及对应的实验结果，方便以后回溯和寻找新的优化点。所有数字都来自真实运行，并给出结果文件路径。按时间倒序排列，最新的在最上面。

条目格式：

```text
## 日期 · 标题
- 提交：<commit>
- 是什么 / 为什么 / 怎么做
- 结果：真实数字 + 结果文件
- 待跟进：发现的新问题或后续可以优化的点
```

---

## 2026-10-17 · 真实模型采样的前缀一致性（本地 CPU 先导实验）

- **提交**：`a47b6d2`，以及本条目对应的结果与文档提交
- **是什么**：新增 `experiments/real_sampling/`。让 Qwen2.5-0.5B-Instruct 在数据集的真实历史上实际采样生成每一轮，按 vLLM Hermes 工具解析器解析成 assistant 消息，再重渲染下一轮 prompt，检查前缀并归因。
- **为什么**：前缀实验用的是模板规范输出，测不到非规范 token 切分和工具调用重新格式化，`retoken = 0` 是方法上的盲区。这是项目最大的短板："只是模拟"。用户于 2026-10-17 同意下载 0.5B–1.5B 小模型的权重。
- **怎么做**：
  - 采样参数为 temperature=1.0、top_p=1.0、top_k=0，最多 512 个新 token；每个轮次的种子固定，CPU 上能逐 token 复现。
  - 3 个数据集各 100 个轮次，跳过 prompt 超过 4,096 token 的轮次。
  - 解析逻辑对照 vllm@`da9134b` 的 `hermes_tool_parser.py` 实现，有离线单元测试。
  - 新增可选依赖 `[sampling]`（torch）。
  - 总耗时：CPU 约 79 分钟。MPS 约快 1.7 倍，但不保证可复现，所以没有用。
- **结果**（`experiments/real_sampling/results/`）：
  - 正常结束的 169 个轮次中，前缀成立 102 个（60.4%）；retoken 55 个（32.5%），template 11 个（6.5%），others 1 个。
  - 全部 300 个轮次中，前缀成立 111 个（37.0%）。被截断的 131 个轮次中，有 121 个出现 retoken（92.4%）。
  - retoken 实例（用相同种子重新采样复现）：采样得到 `-t` `xt`，规范分词是 `-txt`。
  - template 的来源：工具调用 JSON 带缩进被重新格式化为紧凑格式、换行叠加（8 个）；以 `<|endoftext|>` 结束（3 个）；正文中的 `<|im_start|>` 被解码丢掉（1 个）。others 是解析器丢掉了多余的 `description` 字段。
  - "自重分词不一致"与 retoken 归因 300/300 吻合，可以交叉验证归因器。
- **待跟进**：
  - 0.5B 在 T=1.0 下质量低，比例不能推广到大模型，需要在 Kaggle GPU 上用 vLLM 和 1.5B 或更大的模型复现。
  - 加一组模型推荐采样设置（T=0.7 等）的对比。
  - 覆盖其他家族的模板。
  - 去掉 4,096 token 的 prompt 限制，减少选择偏差。

## 2026-10-16 · P2 方案 1（预算分配）与方案 2（共享前缀成本），并更正 P2 结论

- **提交**：`b7f9c7c`、`3b02a36`，以及本条目对应的结果与文档提交
- **是什么**：
  - 方案 1：在同一单行总长 T ∈ {8k, 16k, 32k} 下，对比 prompt / response 各 T/2 的固定上限，与 `max_total_length = T` 的动态分配。
  - 方案 2：新增 `trace2batch.core.shared_prefix_lengths`，统计相邻行的共享前缀，得到"共享前缀只算一次"时的去重 token 数。
- **为什么**：P2 发现 `split_on_budget` 有两个代价，一是 prompt 预算紧时失效（问题 B2），二是 token 膨胀 1.7–8.8 倍（问题 B1）。方案 1 针对 B2，方案 2 针对 B1。
- **怎么做**：
  - 截断实验新增 54 个单元：3 个数据集 × 3 个 T × 2 种分配 × 3 种策略。
  - 所有单元都统计行 token 总量和去重后的 token 数。
  - 修复一个小 bug：没有可用行时计数键缺失，导致汇总表报错。
- **结果**（`experiments/truncation/results/summary_allocation.md`、`results.json`）：
  - **更正**：P2 中 split 的大幅提升（例如 SWE-smith 保留率从 9.4% 到 84.7%），主要来自该设置下单行允许的总长更大（prompt 32k + response 4k）。**在相同的单行总长下，split 保留的 policy token 从不多于 truncate**：动态分配时两者完全相同，固定分配时持平或略低。这与推导一致：位置 ≥ T 的 token，任何一行都装不下。
  - 方案 1：动态分配让 truncate 的保留率提高到 1.02–2.38 倍。例如 SWE-smith T=16k 时从 26.3% 到 54.6%，最终生成丢失从 75.0% 到 41.3%；nebius T=32k 时从 77.8% 到 99.4%。
  - 方案 2：在 prompt 32k、response 4k 的设置下，split 行 token 的 62.6%（SWE-smith）和 67.7%（nebius）是重复前缀；去重后的倍数从 8.82× 降到 3.29×，从 7.77× 降到 2.51×。每个保留的 policy token 需要的 token 数：truncate 17.4 / 7.4，split 去重后 6.4 / 6.5。
  - split 剩下的优势：行在轮次边界结束，"行尾仅观察"为 0%。
- **待跟进**：
  - 超过单行总长的轨迹，只有修改上下文（方案 3：压缩观察）才能保留，需要评估 off-policy 风险。
  - 动态分配要求训练端按总长打包序列（如 verl 的 remove-padding）。
  - 去重收益需要训练端支持共享前缀计算。
  - 都需要 GPU 训练来验证（P3）。

## 2026-10-16 · 前缀实验扩展到 4 个模型家族

- **提交**：`88fb8e3`、`c590fa2`、`2a37f3d`、`f4e4665`，以及本条目对应的结果与文档提交
- **是什么**：在 Qwen2.5 / Qwen3 之外，新增 DeepSeek-V3.1、GLM-4.5、gpt-oss-20b 三个家族的 tokenizer，共 9 种模板配置。
- **为什么**：P1 只覆盖 Qwen 系，是报告里最大的局限。不同家族的模板对历史的处理差别很大，需要验证结论是否普遍成立。
- **怎么做**：
  - 每个配置声明自己的结束标记。
  - 追加消息会改写更早的历史时（gpt-oss），按生成提示串定位生成部分。
  - DeepSeek-V3.1 传字符串形式的工具参数；gpt-oss 的模板日期固定为 2026-10-16。
  - 归因器的模板标记加入各 tokenizer 的特殊 token。
  - 修复缓存目录 bug，改为显式传入 `cache_dir`。
  - 修复两个归因器误判：一是差异位于 `<think>…</think>` 块内（GLM 保留标签、只清空内容），二是结尾片段在一条消息里重复出现时，对齐窗口截到了错误位置（nebius × GLM 思考配置中有 1 次被误判为 others）。
- **结果**（`experiments/template_prefix/results/`，修复后的完整运行）：
  - B 方式在 107,848 次合并尝试中失败 54,391 次，**全部归为 template**，retoken 和 others 都是 0；C 与 B 完全相同。
  - 前缀稳定、3 个数据集都 100% 合并：Qwen2.5、DeepSeek-V3.1、GLM-4.5 关闭思考。
  - 全部失败：
    - Qwen3 关闭思考（3 个数据集），原因是空 `<think>` 块被删除；
    - Qwen3 / GLM-4.5 思考模式不回传 reasoning（合成，3 个数据集）；
    - Qwen3 / GLM-4.5 思考模式回传 reasoning 但观察用 `user` 角色（合成，nebius）；
    - gpt-oss × nebius，原因是 `<|return|>` 在历史中变为 `<|end|>`。
  - 合并失败时样本退化成逐轮。GLM-4.5 × SWE-smith 的 think_drop 行 token 总量为 think_keep 的 15.67 倍。
  - Qwen 部分的 36 个单元，与扩展前的运行逐项一致。
  - 渲染报错（对话被跳过）：gpt-oss 有 2 条 Toucan 对话（工具 `description` 为 `null`），GLM-4.5 有 1 条（工具参数不是合法 JSON）。
- **待跟进**：
  - DeepSeek-V3.1 的思考模式没有覆盖：模板无法渲染该模式下的生成。
  - Llama（需要申请权限）、Mistral 未覆盖。
  - Issue #645 的多家族补充评论：草稿已写好（本地 `drafts/issue_645_followup.md`，未提交），用户审核通过，但决定暂不发布（2026-10-16），等维护者回复后再考虑。

## 2026-10-15 · P2：截断策略量化

- **提交**：`c2aafde`、`5672657`、`63ecb9b`、`739de2f`
- **是什么**：`build_samples` / `build_batch` 新增 `truncation_strategy`，取值 `truncate`（默认）、`split_on_budget`、`drop`，并做了 72 组对比实验。
- **为什么**：合并后的 response 超出预算时，默认截掉尾部，而尾部正好是最接近奖励的后期生成，存在信用分配错位。
- **怎么做**：
  - `split_on_budget`：并入下一轮会超限时，在轮次边界另起一行，新行的 prompt 是完整上下文；单轮生成本身超限时仍截断。
  - 实验复用 P1 的样本，使用 Qwen2.5 + B 方式（这个组合 100% 合并），网格为 prompt {8k, 32k} × response {4k, 8k, 16k, 32k} × 3 种策略。
- **结果**（`experiments/truncation/results/`）：
  - 合并 response 中约 80% 是工具观察：生成 token 的占比分别为 SWE-smith 16.7%、nebius 17.9%、Toucan 20.7%。
  - SWE-smith，response 4k：truncate 只保留 9.4% 的生成 token，95.7% 的轨迹丢失最后一次生成，86.7% 的行结尾只剩观察。
  - prompt 32k 时，split 的保留率：SWE-smith 84.7%–86.0%，nebius ≥ 99.6%；最终生成丢失 SWE-smith ≤ 13.3%、nebius ≤ 1.7%。代价是行 token 总量为 truncate 的 1.71–8.82 倍（response ≤ 16k）。
  - prompt 8k 时，split 产生大量 `is_drop` 行；response ≥ 8k 时保留率与 truncate 持平。
  - drop 在所有设置下最差，SWE-smith 在 4k 时 95.7% 的轨迹没有可用行。
- **待跟进**（据此提出方案 1、2）：split 的两个代价，一是 prompt 预算紧时失效，二是计算量膨胀。问题 A（观察挤占预算）的根源可以考虑压缩观察，但存在 off-policy 风险。问题 D（信用分配）需要训练验证。

## 2026-10-15 · 上游 Issue 与 README 重写

- **提交**：`46f4ec7`、`ba148c2`
- **是什么**：审核后原文提交 [microsoft/agent-lightning#645](https://github.com/microsoft/agent-lightning/issues/645)；README 改为以研究发现为主线。
- **为什么**：把 P1 的发现反馈给上游，同时提升项目首页的表达力度。
- **结果**：Issue 已提交，截至 2026-10-16 尚无维护者回复。

## 2026-10-15 · P1：真实 chat template 前缀一致性研究

- **提交**：`0b7ef32`、`8a86a1b`、`5200a8d`、`f1a076e`、`c50ab9c`、`7e7b68a`、`2871fa1`
- **是什么**：新增零依赖归因器 `trace2batch/mismatch.py`（template / retoken / others 三类）和实验 `experiments/template_prefix/`。
- **为什么**：上游 `rollout_adapter.py` 第 602–605 行把 `template_mismatch` / `retoken_mismatch` 固定写成 False，合并失败的原因无从得知。
- **怎么做**：
  - 3 个公开数据集各 300 条轨迹，Qwen2.5 和 Qwen3（3 种思考配置）。
  - 三种采集方式：A token 直通（对照组）、B 模拟 vLLM `return_token_ids`、C 文本重分词。
  - 归因次数与 `build_samples` 的不匹配计数逐条交叉校验。
  - 修过一个 bug：按字符对齐时，被删的思考块会和后面的 `<tool_call>` 错配成碎片，改为按标记 / 单词粒度对齐。
- **结果**（`experiments/template_prefix/results/`）：
  - Qwen2.5 100% 合并。
  - Qwen3 `enable_thinking=False` 时，11,984 次合并全部失败，全部归为 template；不截断时行 token 总量膨胀 15.75×（SWE-smith）、14.75×（nebius）、1.75×（Toucan）。
  - 思考模式下不回传 reasoning 时全部失败；回传时，nebius（观察用 `user` 角色）仍然全部失败。
  - 两次完整运行的合并计数一致。
- **待跟进**：retoken 类没有观测到，因为没有真实模型采样；P1 当时只覆盖 Qwen，已于 2026-10-16 扩展。

## 2026-10-15 · P0：仓库整改

- **提交**：`99017e4`、`84da424`、`e79e4d5`、`4d75a8d`、`df70da3`、`a1784f8`、`5a2ba3c`、`ac83c53`
- **是什么**：
  - 补 `[project]` 元数据和可选依赖。
  - 新增 CI（3.12 / 3.13）。
  - 补全版权和来源说明。
  - 新增 `max_total_length`、`reward_fillna_value`、`build_batch`。
  - 删除低价值断言。
- **为什么**：让来源、许可证和工程规范站得住，为后续研究打基础。
- **结果**：测试从 51 项增加到 69 项，原有的行为测试不变；CI 在 GitHub 上通过。
