# 真实模型采样下的前缀一致性（本地 CPU 小规模版）

> 所有数字来自 `python -m experiments.real_sampling.run --model qwen2.5-0.5b --turns-per-dataset 100` 的一次完整运行（CPU，可复现），原始结果在 `results/qwen2.5-0.5b/results.json` 和 `results/qwen2.5-0.5b/summary.md`。这是小规模先导实验，大规模版本计划在 GPU 上用 vLLM 复现。

## 1. 为什么要做

前缀一致性实验（[`../template_prefix/REPORT.md`](../template_prefix/REPORT.md)）里，B 方式的 response 是"模板认可的规范输出"，存在两个盲区：

- **非规范 token 切分**：模型采样出的 token 序列，不一定等于对同一段文本重新分词的结果。
- **工具调用重新格式化**：模型自己写出的工具调用文本，经工具解析器解析、模板重新渲染后，可能与原文不同。

所以那次实验中 `retoken_mismatch = 0`，Qwen2.5 也 100% 合并。这里用真实模型采样来补上这两个盲区。

## 2. 设置

- **模型**：[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)（Apache-2.0，revision `7ae5576`），float32，transformers 5.20.0 + torch 2.14.1，运行在 CPU 上。
- **采样参数**：temperature=1.0、top_p=1.0、top_k=0（纯采样，RL 训练中常用的设置；该模型推荐的推理设置是 T=0.7、top_p=0.8、top_k=20）。停止 token 用模型 `generation_config` 中的 `<|im_end|>` 和 `<|endoftext|>`，`max_new_tokens=512`。每个轮次的随机种子固定，用同样的种子重新采样能逐 token 复现。
- **轮次选择**：与前缀实验相同的 3 个数据集、相同的 300 条对话（种子 0）。从中选"后面紧跟观察消息"的助手轮次，打乱后依次取，跳过 prompt 超过 4,096 token 的轮次（CPU 限制），每个数据集取 100 个，共 300 个。
- **每个轮次的流程**（对应上游的"代理 + vLLM `return_token_ids`"链路）：
  1. 用数据集的真实历史渲染 prompt 并分词。
  2. 模型采样生成这一轮，记录采样到的 token id（含停止 token）。
  3. 把生成文本按 vLLM 默认的 `skip_special_tokens=True` 解码，再按 **vLLM Hermes 工具解析器**（vllm@`da9134b`，`vllm/tool_parsers/hermes_tool_parser.py`）解析成 assistant 消息：`content` 加 `tool_calls`；参数先 `json.dumps`，下一次请求时再由 `_postprocess_messages` 解析回对象。
  4. 接上数据集中该轮之后的观察消息，重新渲染下一轮 prompt。观察内容取自数据集，与模型实际选择的动作无关；它只用来让这一轮助手消息进入历史。
  5. 检查"prompt + 采样 token"是否是下一轮 prompt 的前缀，不是就用归因器分类。另外单独检查"自重分词"：对采样文本重新分词，结果是否等于采样 token。

## 3. 结果

| 数据集 | 采样轮次 | 正常结束 | 输出了工具调用标签 | 工具调用解析失败 | 自重分词不一致 | 前缀成立 | template | retoken | others |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SWE-smith | 100 | 33 | 2 | 2 | 82 | **17** | 2 | 81 | 0 |
| Toucan | 100 | 85 | 46 | 8 | 27 | **64** | 8 | 27 | 1 |
| nebius | 100 | 51 | 0 | 0 | 69 | **30** | 2 | 68 | 0 |

很多生成在 512 token 时被截断：SWE-smith 67 个，nebius 49 个。小模型在 T=1.0 下容易越写越散，被截断的生成质量明显更差。所以下面**以正常结束的 169 个轮次为主要口径**：

| 口径 | 轮次 | 前缀成立 | retoken | template | others |
|---|---:|---:|---:|---:|---:|
| 只看正常结束的轮次 | 169 | **102（60.4%）** | 55（32.5%） | 11（6.5%） | 1（0.6%） |
| 全部轮次 | 300 | 111（37.0%） | 176（58.7%） | 12（4.0%） | 1（0.3%） |

正常结束的轮次中，按数据集的前缀成立率分别是：SWE-smith 13/33（39.4%），Toucan 62/85（72.9%），nebius 27/51（52.9%）。

**对比**：同一个 tokenizer 家族（Qwen2.5）在模板规范输出下 100% 合并。也就是说，上面的所有失败**都来自真实采样**。在 trajectory 聚合中，每次合并都要求前缀成立。以正常结束轮次的 60.4% 为例，一条有十几轮的轨迹几乎不可能完整合并。

## 4. 失败原因

**(1) 非规范 token 切分（retoken，主要原因）。** 采样出的 token 序列和对同一段文本重新分词的结果不同。下面是用相同种子重新采样复现的 Toucan 案例，展示分歧点附近的 token：

| 采样 token | 规范分词 |
|---|---|
| `desk` `-t` `xt` `-f` | `desk` `-txt` `-f` `ilename` |
| `<` `TOOLS` `>\n` | `<T` `O` `OLS` |
| `anc` `en` `debug` `ged` | `anc` `ende` `bug` `ged` |
| `Re` `dea` `fl` | `Red` `ea` `fl` |

"自重分词不一致"这项检查与 retoken 归因在 300 个轮次中完全吻合：0 个不一致，可以作为归因器的交叉验证。被截断的 131 个轮次中有 121 个（92.4%）出现 retoken，正常结束的 169 个中是 55 个（32.5%），说明它和生成质量强相关。

**(2) 工具调用重新格式化（template，Toucan 上 8 个）。** 模型写出的工具调用 JSON 带换行和缩进，模板用紧凑的 `tojson` 重新渲染；或者正文结尾的换行与模板自己加的换行叠在一起，`\n` 变成 `\n\n`。归因器把这些归为 template（空白差异，或 JSON 语义相同）。

**(3) 结束 token 不同（template，3 个）。** 模型以 `<|endoftext|>` 结束（这也是 `generation_config` 中的停止 token），而历史中这条消息总是以 `<|im_end|>` 结尾。

**(4) 解码丢掉的特殊 token（template，1 个）。** 模型在正文中生成了 `<|im_start|>`，`skip_special_tokens=True` 解码时被丢掉，历史中就没有了。

**(5) 解析器丢掉的字段（others，1 个）。** 模型在工具调用 JSON 里多写了 `"description"` 字段，Hermes 解析器只保留 `name` 和 `arguments`，重新渲染后这段内容消失。

另外有 10 次工具调用的 JSON 不合法、解析失败（SWE-smith 2 次，Toucan 8 次）。按 vLLM 的行为，整段文本作为 `content` 原样保留，所以这本身不会破坏前缀。

## 5. 结论

1. **真实采样会让"模拟中 100% 合并"的模板大量合并失败。** 以 Qwen2.5 为例，正常结束的轮次中只有 60.4% 能合并；主要原因是非规范 token 切分（占正常结束轮次的 32.5%），其次是工具调用重新格式化、结束 token 不同等模板层面的差异（6.5%）。
2. 这说明 `retoken_mismatch` 在真实训练中**不是可以忽略的类别**，前缀实验中观测到 0 是模拟方法的盲区。上游 Issue 中提议的 retoken / template 分类在真实采样下都有实际意义。
3. 对训练框架的含义：只要使用"服务端每轮重渲染 + 重分词"的链路，就算模板前缀稳定，非规范切分也会让 trajectory 聚合频繁退化。要稳定合并，需要 token-in-token-out（直接沿用上一轮的 token，而不是重渲染文本），也就是前缀实验中 A 方式的链路。

## 6. 局限性

- **模型很小（0.5B）**，在 T=1.0 纯采样下生成质量低：169/300 正常结束，被截断的生成更容易出现非规范切分。更大的模型或更低的温度，比例很可能更低。**这里的比例不能直接推广到大模型**，需要在 GPU 上用更大的模型复现。
- **只覆盖 Qwen2.5 一个模板**，没有测其他家族在真实采样下的表现。
- **选择偏差**：CPU 上跳过了 prompt 超过 4,096 token 的轮次（SWE-smith 跳过 485 个候选，nebius 跳过 257 个），所以样本偏向轨迹前段（平均位置：SWE-smith 第 4.7 条消息，nebius 第 8.3 条，Toucan 第 2.4 条）。
- **用 transformers 的 `generate` 模拟 vLLM**：采样语义相同，但没有经过真实的 vLLM 服务端。
- **观察内容取自数据集**，不是模型实际动作的执行结果。这不影响这一轮的前缀检查，但不代表完整的真实轨迹。
- **规模**：每个数据集 100 个轮次。
