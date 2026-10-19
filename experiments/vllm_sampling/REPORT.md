# 真实 vLLM 服务端采样下的前缀一致性（GPU）

> 数字来自 `experiments/vllm_sampling/run.py` 在 Kaggle 免费 GPU（2 块 Tesla T4）上的一次完整运行，结果在 `results/<模型>@rl/results.json`。用于提交和调度的 Kaggle 驱动脚本只保留在本地，不在仓库中。

## 1. 与此前实验的区别

| | 前缀实验 | 本地真实采样 | **本实验** |
|---|---|---|---|
| response 来源 | 模板认可的规范输出 | transformers 在 CPU 上采样 | **真实 `vllm serve` 采样** |
| 工具解析 / 模板渲染 | 本地模拟 vLLM | 本地模拟 vLLM | **vLLM 服务端（Hermes 解析器、`return_token_ids`）** |
| 模型 | 仅 tokenizer | 0.5B / 0.6B | **Qwen2.5-1.5B-Instruct、Qwen3-0.6B + 补丁** |
| prompt 长度 | 不限 | 跳过超过 4,096 token 的 | **上限 32,768（max-model-len）** |

这是与 agent-lightning 训练代理完全相同的链路：每一轮都由服务端对完整历史重新渲染 chat template 并分词。

## 2. 设置

- **服务端**：vLLM 0.32.0，fp16，`--max-model-len 32768`，`--enable-auto-tool-choice --tool-call-parser hermes`；Qwen3 用 `--chat-template` 加载前缀稳定补丁（`experiments/template_patch/templates/qwen3_prefix_stable.jinja`），并设置 `enable_thinking=False`。
- **每个轮次的流程**：
  1. 第 k 轮：数据集真实历史按 OpenAI 格式发送，T=1.0、top_p=1.0、不限 top_k，`max_tokens=1024`，每个轮次的种子固定；
  2. 把服务端返回的 assistant 消息原样写回历史，接上数据集中的观察消息，再请求一次（`max_tokens=1`），拿到服务端渲染的下一轮 prompt；
  3. 检查"第 k 轮 prompt + 生成 token"是否是下一轮 prompt 的前缀，不是就归因。
- **轮次选择**：与此前实验相同的 3 个数据集、相同的 300 条对话，从中选"后面紧跟观察消息"的助手轮次，打乱顺序（种子 0）后依次取，每个数据集 100 个。超过 32,768 token 被服务端拒绝的轮次跳过并计数。
- **工具定义**：Toucan 用数据集自带的定义；SWE-smith 数据集没有工具定义，用对话中出现过的工具名和参数名构造近似定义（参数一律为 string），让服务端的工具解析生效；nebius 是文本动作，不提供工具。
- **资源**：Kaggle 2 块 T4 并行，两个模型各占一块，并发 16。每个任务设了时间上限（3.25 小时后不再开始新批次）。本次运行消耗 2.67 GPU 小时。

## 3. 结果

| 模型（T=1.0） | SWE-smith | Toucan | nebius | 合计前缀成立 |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 63/100 | 95/100 | 92/100 | **250/300（83.3%）** |
| Qwen3-0.6B + 补丁（关闭思考） | 22/85 | 94/100 | 96/100 | **212/285（74.4%）** |

Qwen3 在 SWE-smith 上到达时间上限，只完成了 85 个轮次。两个模型各有 12 和 13 个轮次因 prompt 超过 32,768 token 被跳过：Qwen2.5 在 SWE-smith 11 个、nebius 1 个；Qwen3 在 SWE-smith 11 个、nebius 2 个。

**失败原因**（按归因器分类，并人工归纳 template 的具体来源）：

| 原因 | Qwen2.5-1.5B | Qwen3-0.6B + 补丁 |
|---|---:|---:|
| **template：工具调用前插入换行**（模型按规范格式输出 `正文\n<tool_call>`，Hermes 解析器把末尾的 `\n` 保留在 content 里，模板重新渲染时又插入一个 `\n`，变成 `\n\n`） | **22** | **62** |
| template：以 `<|endoftext|>` 而非 `<|im_end|>` 结束 | 4 | 0 |
| template：工具调用 JSON 被重新格式化（多行缩进变成紧凑格式、`28.50` → `28.5`、`-74.0060` → `-74.006` 等） | 3 | 4 |
| template：其他（模型没有闭合工具调用、模板补上 `</tool_call>`；`\r\n` → `\n`） | 2 | 1 |
| retoken：非规范 token 切分 | 19（6.3%） | 6（2.1%） |
| others | 0 | 0 |
| **合计失败** | 50 | 73 |

## 4. 结论

1. **在真实服务端链路上，即使模板本身前缀稳定（Qwen2.5，或加了补丁的 Qwen3），trajectory 合并仍会在 16.7%–25.6% 的轮次上失败。** 这与前缀实验中"规范输出下 100% 合并"形成对比：差异全部来自真实模型的输出与模板规范格式之间的不一致。
2. **最主要的原因是"工具调用前插入换行"**，分别占 Qwen2.5 失败的 44%、Qwen3 失败的 85%，集中在工具调用频繁的 SWE-smith 上。机制是：在全部 84 个案例中，模型都按规范格式在 `<tool_call>` 前写了 `\n`。Hermes 解析器把 `<tool_call>` 之前的文本（包括这个 `\n`）原样作为 `content`；Qwen 系模板重新渲染时又在 content 和工具调用之间加一个 `\n`，两者叠加成 `\n\n`。也就是说，**即使模型输出完全规范，这个问题也会出现**，它是解析器与模板之间的交互缺陷。（2026-10-19 更正：此前误写为"模型没有写这个换行"。）
3. **在 1.5B 规模、真实 vLLM 下，非规范切分比本地 0.5B 实验少得多**：6.3%，本地 0.5B 是 32.5%（正常结束口径，transformers 采样）。本地实验的高比例主要来自小模型高温采样时的低质量输出，见 `experiments/real_sampling/REPORT.md` 中的温度消融。
4. **对修复的含义**：
   - 换行问题属于模板层面，可以继续改进补丁：content 已经以 `\n` 结尾时，不再插入换行。
   - JSON 重新格式化、结束 token 不同、非规范切分，只能靠 token-in-token-out 消除（见 `experiments/template_patch/REPORT.md`）。

## 5. 局限性

- 只覆盖 2 个模型（1.5B 和 0.6B，权重下载范围限制在 1.5B 以内）和 T=1.0 一种采样设置。
- SWE-smith 的工具定义是从对话中构造的近似定义，与原始 agent 实际使用的定义不同，可能影响模型输出工具调用的方式和频率。
- 观察消息取自数据集，不是模型实际动作的执行结果；这只影响下一轮之后的内容，不影响本轮的前缀检查。
- 服务端批处理时，fp16 数值可能随批次组成而变化，固定种子也不能保证逐 token 复现；结果是一次运行的统计。
- Qwen3 在 SWE-smith 上因时间上限只完成了 85 个轮次；超过 32,768 token 的轮次被跳过，样本略偏向较短的 prompt。
