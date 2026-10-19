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

## 2026-10-19 · 更正："工具调用前换行"的机制

- **是什么**：重新检查 GPU 全量结果中全部 84 个"工具调用前多出换行"的案例（Qwen2.5 22 个、Qwen3 62 个）。**模型在每一个案例里都按规范格式在 `<tool_call>` 前写了 `\n`**。CPU 真实采样中的同类案例（17、18、3、14 个）也是如此。
- **更正**：此前报告、README、开发日志写的是"模型输出 `正文<tool_call>`（没写换行），模板插入了换行"，这是错的。真实机制是：Hermes 解析器把 `<tool_call>` 之前的文本连同末尾的 `\n` 一起作为 content；Qwen 系模板重新渲染时又插入一个 `\n`，叠加成 `\n\n`。**即使模型输出完全规范，这个问题也会出现**，它是解析器与模板之间的交互缺陷。
- **怎么做**：修改所有错误描述的位置（README、`experiments/vllm_sampling/REPORT.md`、`experiments/template_patch/REPORT.md`、本日志），并标注更正日期。
- **影响**：补丁第 2 版的设计随之调整：只在 content 不以 `\n` 结尾时才插入换行，这样既不改变官方格式下的渲染，又能修复观察到的全部案例。

## 2026-10-19 · Kaggle 真实 vLLM 采样全量（方案 A）

- **提交**：`768deae`（时间上限与逐数据集写盘；Kaggle 上按这个 commit 克隆），以及本条目对应的结果与文档提交
- **是什么**：方案 A，用户已同意。两块 T4 并行：Qwen2.5-1.5B-Instruct（GPU 0）和 Qwen3-0.6B + 补丁（GPU 1），T=1.0，每个数据集 100 个轮次。每个任务设时间上限：客户端 3.25 小时后不再开始新批次，驱动 3.5 小时后强制终止。
- **为什么**：用真实 vLLM 服务端、更大的模型和 32k 的 prompt 上限，验证本地 CPU 小模型实验的结论。
- **结果**（`experiments/vllm_sampling/results/`，报告见 `experiments/vllm_sampling/REPORT.md`；Kaggle 驱动与日志只在本地 `kaggle/results/full_v1/`）：
  - 消耗 **2.67 GPU 小时**（在约 3.6 小时的上限内），累计 4.35 h，本周剩余 25.65 h。
  - Qwen2.5-1.5B：前缀成立 250/300（83.3%）。SWE-smith 63/100、Toucan 95/100、nebius 92/100。
  - Qwen3-0.6B + 补丁：212/285（74.4%）。SWE-smith 22/85（到达时间上限，只完成 85 个）、Toucan 94/100、nebius 96/100。
  - 失败原因：
    - "工具调用前插入换行"：Qwen2.5 22/50，Qwen3 62/73；
    - 以 `<|endoftext|>` 结束：Qwen2.5 4 次；
    - 工具调用 JSON 重新格式化：3 次和 4 次；
    - 其他（补上 `</tool_call>`、`\r\n` → `\n`）：2 次和 1 次；
    - retoken：19 次（6.3%）和 6 次（2.1%）。
  - 结论：真实服务端链路上，即使模板前缀稳定，仍有 16.7%–25.6% 的轮次合并失败；主要原因是工具调用前的换行。1.5B 模型的非规范切分远少于本地 0.5B 实验。
- **待跟进**：
  - 补丁第 2 版：不再在 content 与工具调用之间插入换行，先在 CPU 上验证；
  - 简历与 README 的数字同步更新；
  - 如有余额，可以加推荐采样设置或更多模型。

## 2026-10-18 · Kaggle 真实 vLLM 采样 pilot 第 2 版

- **提交**：`c505388`（Kaggle 上按这个 commit 克隆）
- **是什么**：两块 T4 并行：GPU 0 跑 Qwen2.5-1.5B-Instruct，GPU 1 跑 Qwen3-0.6B + 补丁模板（vLLM `--chat-template`）。都是 T=1.0、每个数据集 20 个轮次，用来验证 pilot 第 1 版的修复并测量吞吐。
- **结果**（本地 `kaggle/results/pilot_v2/`，不推送）：
  - 修复生效：SWE-smith 请求正常（只有 3 个 prompt 超过 32,768，被拒），工具调用被解析（SWE-smith 7/20 和 17/20，Toucan 14/20 和 15/20），所有轮次都正常结束。
  - Qwen2.5-1.5B：前缀成立 49/60（81.7%）。
    - SWE-smith 13/20：template 6 次（全部是"工具调用前插入换行"），retoken 1 次；
    - Toucan 19/20：retoken 1 次；
    - nebius 17/20：retoken 3 次。
  - Qwen3-0.6B + 补丁：前缀成立 42/60（70.0%）。
    - SWE-smith 4/20：template 16 次，全部是工具调用前插入换行；
    - Toucan 19/20：JSON 数字规范化 1 次；
    - nebius 19/20：retoken 1 次。
  - **新发现**：在真实 vLLM 链路上，SWE-smith 的主要失败是"工具调用前多出一个换行"（2026-10-19 更正机制：模型按规范格式输出 `正文\n<tool_call>`，Hermes 解析器把末尾的 `\n` 保留在 content 里，模板重新渲染时又插入一个 `\n`，变成 `\n\n`；此前误写为模型没有写换行），占 34 次失败中的 22 次。1.5B 模型在 T=1.0 下 retoken 很少（5/60）。
  - **吞吐与额度**：环境准备 178 s；客户端 Qwen2.5 用了 2,182 s，Qwen3 用了 3,347 s（约 36 s/轮和 56 s/轮，SWE-smith 的长 prompt 占了大头，中位数约 4,100 token）。本次消耗 **1.03 GPU 小时，超出提交前估计的约 0.5 小时**，累计 1.68 h。
- **待跟进**：
  - 全量运行需要缩小规模并设置硬性时间上限。
  - 补丁可以进一步取消"工具调用前插入换行"：解析器保留正文原样（包括结尾换行），不再插入换行可以让两种写法都保持一致。需要先在 CPU 上验证。

## 2026-10-18 · Kaggle 真实 vLLM 采样 pilot 第 1 版（发现 3 个问题）

- **提交**：`6157387`（客户端，Kaggle 上按这个 commit 克隆），以及本条目对应的修复提交
- **是什么**：`experiments/vllm_sampling/run.py` 直接请求 `vllm serve`（Hermes 工具解析、`return_token_ids`）：把服务端返回的 assistant 消息写回历史、接上数据集观察后再请求一次，取服务端渲染的下一轮 prompt 做前缀检查。pilot 设置为 Qwen2.5-1.5B-Instruct、T=1.0、每个数据集 30 个轮次、单块 T4、并发 16。
- **为什么**：用真实服务端链路、更大的模型和不跳过长 prompt 的设置，复现本地 CPU 的结论。
- **结果**（Kaggle 输出保存在本地 `kaggle/results/pilot_v1/`，不推送）：
  - 整条链路跑通。环境准备 71 s，服务就绪 160 s，客户端 1,252 s。本次 pilot 消耗约 0.42 GPU 小时，累计 0.65 h。
  - nebius：30 个轮次全部正常结束，prompt 最长 32,137 token；前缀成立 26/30，4 次失败都是 retoken。
  - **问题 1**：SWE-smith 有 449 次请求被拒（HTTP 400）。原因是发送的历史不符合 OpenAI 接口格式：工具参数是对象而不是 JSON 字符串，tool 消息没有 tool_call_id。只有历史中没有工具调用的前段轮次成功（prompt 最长只有 1,730 token）。
  - **问题 2**：以工具调用结束时 `finish_reason` 为 `tool_calls`，被误算成未结束（Toucan 只显示 5/30 正常结束）。
  - **问题 3**：SWE-smith 数据没有工具定义，vLLM 不解析工具调用（`with_tool_calls` 为 0）。
  - **吞吐**：约 14 s/轮，nebius 的长 prompt 占了大部分时间（30 个轮次用 832 s）。之前估计的全量 1.5–3 GPU 小时明显偏低，按这个速度全量会超过 10 小时，需要缩小规模。
- **修复**：
  - `history_for_api` 把历史转成 OpenAI 格式（参数转字符串，补 id 和 tool_call_id）。
  - `finish_reason` 为 `tool_calls` 时也算正常结束。
  - `derive_tools` 用对话中出现过的工具名和参数名构造近似工具定义。
  - 请求错误按"超长 / 其他"分类，并保存错误信息样例。
- **待跟进**：用 pilot 第 2 版验证修复（两块 GPU 并行，同时验证 Qwen3 补丁模板在 vLLM `--chat-template` 下的表现），并据此确定全量规模。

## 2026-10-18 · token-in-token-out 参考实现（B）与采样温度消融（C）

- **提交**：`8dc48b6`、`902bdb9`，以及本条目对应的结果与文档提交
- **是什么**：
  - B：`trace2batch/tito.py`，只用标准库的 TITO 轨迹记录器；`experiments/tito/encoder.py`，Qwen 系的新消息分词；`experiments/tito/run.py`，在规范输出上验证。
  - C：真实采样新增 `--sampling recommended`（取模型的 generation_config）。
- **为什么**：
  - 模板补丁报告的结论是"只有 token-in-token-out 能消除所有失败"，需要一个可运行、经过验证的实现来支撑。
  - 真实采样中 retoken 比例很高，需要回答"这是不是高温采样的产物"。
- **怎么做**：
  - 新消息分词：把新消息接在占位对话之后渲染，取占位 assistant 的 `<|im_end|>` 之后的文本。
  - 真实采样中额外记录 TITO 上下文与重渲染是否逐 token、逐文本相同。
  - 推荐设置在相同的轮次和种子上重跑：Qwen2.5-0.5B 300 个轮次，Qwen3-0.6B + 补丁 150 个轮次。
- **结果**（`experiments/tito/results/`、`experiments/real_sampling/results/*@recommended/`）：
  - TITO 规范输出验证：Qwen2.5 和 Qwen3 补丁下，12,884 / 12,884 个 prompt 与重渲染逐 token 相同；Qwen3 官方模板下只有 900 个相同（每条轨迹的第一轮）。TITO 合并失败在所有配置下都是 0。
  - 温度消融（Qwen2.5-0.5B，正常结束口径）：T=1.0 时前缀成立 60.4%、retoken 32.5%、template 6.5%；推荐设置下分别为 88.7%、4.7%、6.6%。非规范切分对温度敏感，模板层面的失败与温度无关。
  - Qwen3-0.6B + 补丁：T=1.0 时 78.3%，generation_config 默认设置下 83.2%；retoken 11 → 3，template 20 → 21。注意：这里用的是思考模式的默认值，不是非思考模式的建议。
  - 交叉验证：TITO 与重渲染逐 token 相同的轮次，恰好等于前缀成立的轮次（188 和 119）；文本相同但 token 不同的轮次，恰好等于 retoken 的数量（10 和 3）。
- **待跟进**：
  - 生成被截断时，TITO 缺少 `<|im_end|>`，需要在框架里单独处理。
  - 采样温度消融还没有覆盖 GPU 上的大模型，Kaggle 全量实验中会包含。

## 2026-10-18 · Kaggle T4 上的 vLLM 可行性检查

- **提交**：本条目对应的提交
- **是什么**：本地脚本 `kaggle/vllm_feasibility.py`（不推送到 GitHub）在 Kaggle 免费的 T4 上安装 vLLM，启动带 Hermes 工具解析的 OpenAI 兼容服务，检查 `return_token_ids` 与下一轮前缀。这是 GPU 上大规模真实采样实验的前置检查。
- **为什么**：本地 CPU 只能跑 0.5B 级模型、每个数据集 100 个轮次，还要跳过长 prompt。要真正复现上游链路（真实的 vLLM 服务端），需要 GPU。
- **过程**（用户已同意提交私有 notebook，账号有 30 GPU 小时/周）：
  - 第 1 次失败：Kaggle 预装的 torchaudio 是按 CUDA 12.8 编译的，vLLM 自带的 PyTorch 是 CUDA 13.0，导致启动报错。
  - 第 2 次：我的修复脚本没有生效就推送了，结果同样失败，浪费约 5 分钟 GPU。
  - 第 3 次成功：把 vLLM 装进独立的 uv 虚拟环境。
  - 三次合计用了 0.23 GPU 小时。
- **结果**（本地 `kaggle/results/feasibility_v3.json`；按用户要求，Kaggle 相关脚本与结果不推送到 GitHub）：
  - vLLM 0.32.0 在 T4（计算能力 7.5，fp16）上可用：安装 141 s，服务就绪 125 s。
  - 响应中包含 `prompt_token_ids` 和 `token_ids`，Hermes 解析出了结构化的 `tool_calls`，下一轮前缀成立。首个请求用时 0.79 s（172 个 prompt token、19 个生成 token）。
- **修正与待跟进**：
  - 第 1 次和第 3 次装到的 vLLM 版本不同（0.31.0 / 0.32.0），已固定为 `vllm==0.32.0`。
  - 虚拟环境原本放在 `/kaggle/working`，被保存成了输出，导致下载卡住，已改到 `/tmp`。
  - 下一步：在 GPU 上用真实 vLLM 跑大规模真实采样（更大的模型、不跳过长 prompt）。

## 2026-10-18 · 前缀稳定的模板补丁（Qwen3 / gpt-oss）与 token-in-token-out 对照

- **提交**：`37c01bb`、`b81e5c8`、`b568581`，以及本条目对应的结果与文档提交
- **是什么**：在官方模板上做最小改动，编写两个补丁（`experiments/template_patch/templates/`），让历史消息按生成时的样子渲染：
  - Qwen3：始终保留思考块，没有内容时渲染空块；
  - gpt-oss：保留工具调用的 analysis，历史中的最终回答以 `<|return|>` 结尾。
  同时在规范输出和真实采样两个层面验证，并与 token-in-token-out 对照。
- **为什么**：项目此前只有诊断，没有修复。这一步把"发现问题"推进到"解决问题"，并回答模板层面的修复能走多远。
- **怎么做**：
  - `TokenizerConfig` 新增 `chat_template` 字段，可以用补丁模板替换官方模板。
  - 新增 `experiments/template_patch/run.py`，复用前缀实验流程，并统计补丁让多少轮 prompt 偏离官方模板。
  - 真实采样脚本支持 `--model`；下载 Qwen3-0.6B（Apache-2.0，`c1899de`），在关闭思考的情况下，用官方模板和补丁模板各采样 150 个轮次（CPU，可复现）。
- **结果**（`experiments/template_patch/results/`、`experiments/real_sampling/results/qwen3-0.6b*/`）：
  - 规范输出：Qwen3 关闭思考 0% → 100%（3 个数据集）；nebius 上 Qwen3 思考配置（回传 reasoning）和 gpt-oss 都是 0% → 100%；不回传 reasoning 时仍为 0%，需要修改客户端。
  - 补丁只改变需要改变的 prompt（官方模板能合并的组合，补丁与官方完全相同）。补丁后 Qwen3 关闭思考的行 token 总量与 token 直通对照逐数据集相等。
  - 真实采样（Qwen3-0.6B）：官方模板 0/150；补丁模板 118/150（78.7%）。剩余失败中：
    - 工具调用前的换行 17 次；
    - JSON 数字规范化 2 次（`-74.0060` → `-74.006`、`28.50` → `28.5`）；
    - 正文中出现 `</think>` 1 次；
    - 非规范切分 12 次。
  - 结论：模板补丁可以消除模板改写，但工具调用重新格式化和非规范切分只有 token-in-token-out 能消除。
- **待跟进**：
  - 补丁偏离了 Qwen3 官方"历史中不保留思考"的建议，对模型效果的影响需要训练验证。
  - gpt-oss 补丁没有做真实采样（模型超出 1.5B 的权重范围）。
  - 可以考虑把 token-in-token-out 作为上游建议的一部分。

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
- **结果**（`experiments/real_sampling/results/qwen2.5-0.5b/`）：
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
