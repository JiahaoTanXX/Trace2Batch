# 真实 chat template 下的 trajectory 前缀一致性

> 所有数字都来自 `python -m experiments.template_prefix.run` 的一次完整运行，原始结果在 `results/results.json` 和 `results/summary.md`。数据与版本见 [DATA.md](DATA.md)。

## 1. 研究问题

agent-lightning 的 `trajectory` 聚合（`agentlightning/verl/rollout_adapter.py`，@`69f5a40`）只有在"下一轮 prompt 的 token 以上一轮 prompt + response 的 token 为前缀"时，才把多轮合并成一行；否则拆成多行，并记录一条不匹配。上游为不匹配定义了 `template_mismatch`、`retoken_mismatch`、`others_mismatch` 三个字段，但第 602–605 行把它们固定写成 `False / False / True`，注释是 `Token-prefix failures are classified as other mismatches.`

本实验回答两个问题：

1. 在真实 tokenizer 和 chat template 下，按上游实际的采集链路（OpenAI 兼容代理 + vLLM `return_token_ids`），多轮工具调用轨迹的合并会失败多少次？
2. 这些失败分别属于哪一类原因？

## 2. 设置

- **数据**：3 个公开数据集，各取一个分片，用固定种子抽 300 条轨迹，共 900 条。
  - SWE-smith（结构化 `tool_calls`）
  - Toucan-1.5M 的 Qwen3 子集（MCP 工具调用）
  - nebius SWE-agent（文本动作，工具观察以 `user` 角色出现）
- **Tokenizer**：Qwen2.5-7B-Instruct、Qwen3-8B，只下载 tokenizer 文件，revision 固定。Qwen3 分 3 种配置：
  - `no_think`：`enable_thinking=False`，不注入任何内容
  - `think_keep[synthetic]`：注入合成思考内容，客户端在历史里回传 `reasoning_content`
  - `think_drop[synthetic]`：注入合成思考内容，客户端不回传
- **三种采集方式**：
  - **A 直通**（对照组）：框架直接拼接 token，只对新消息分词。
  - **B vLLM token id**：每轮由服务端对完整 messages 重新渲染并分词作为 prompt；response 是该助手消息在上下文中的规范 token。这对应上游的实际链路（`server/proxy.py` 在训练模式下设置 `return_token_ids=True`）。工具参数在渲染前解析成对象，与 vLLM 的 `_postprocess_messages` 一致。
  - **C 文本重分词**：与 B 相同，但 response 由生成文本单独分词。与 B 对比，可以单独看出边界切分差异的影响。
- 每轮事件喂给 `trace2batch.core.build_samples(mode="trajectory")`，再用 `trace2batch.mismatch.explain_mismatch` 对每一次失败归因。归因次数与 `build_samples` 的 `merge_mismatches` 逐条轨迹核对一致，不一致时脚本直接报错。
- **规模**：每种配置下，SWE-smith 有 5,416 次合并尝试，Toucan 有 421 次，nebius 有 6,147 次，合计 11,984 次；4 个 tokenizer 配置 × 3 种采集方式共 36 个单元。

## 3. 结果

A 直通在全部 12 个"数据集 × tokenizer 配置"组合下合并成功率都是 100%，说明实验框架本身没有引入不匹配。B 和 C 在所有组合下结果完全相同，下表只列 B：

| 数据集（合并尝试） | qwen2.5 | qwen3_no_think | qwen3_think_keep [synthetic] | qwen3_think_drop [synthetic] |
|---|---|---|---|---|
| SWE-smith（5,416） | 100% 合并 | **0% 合并**，5,416 次 template | 100% 合并 | **0% 合并**，5,416 次 template |
| Toucan（421） | 100% 合并 | **0% 合并**，421 次 template | 100% 合并 | **0% 合并**，421 次 template |
| nebius（6,147） | 100% 合并 | **0% 合并**，6,147 次 template | **0% 合并**，6,147 次 template | **0% 合并**，6,147 次 template |

- 所有不匹配都被归为 `template_mismatch`，`retoken_mismatch` 和 `others_mismatch` 都是 0。
- 只要不匹配，就是**每一轮都不匹配**：`trajectory` 模式的行数完全退化成轮数。平均每条轨迹的行数：SWE-smith 19.05，Toucan 2.40，nebius 21.49。

### token 量膨胀

合并失败时，每一行都要重新携带完整的历史 prompt。下表统计所有行的 prompt + response token 总量（`row_tokens`），以同一 tokenizer 配置下的 A 直通（全部合并）作为对照组。统计时长度上限设为不限，所以这是"不截断时的数据量"。实际训练中会被 `max_prompt_length` / `max_response_length` 截断，那时的影响会表现为截断，而不是这个倍数。

| 数据集 | tokenizer 配置 | 行数（B / A） | row_tokens（B） | row_tokens（A） | 倍数 |
|---|---|---:|---:|---:|---:|
| SWE-smith | qwen3_no_think | 5,716 / 300 | 89,554,881 | 5,684,733 | 15.75× |
| SWE-smith | qwen3_think_drop [synthetic] | 5,716 / 300 | 89,612,041 | 5,741,893 | 15.61× |
| Toucan | qwen3_no_think | 721 / 300 | 1,497,643 | 856,045 | 1.75× |
| Toucan | qwen3_think_drop [synthetic] | 721 / 300 | 1,504,853 | 863,255 | 1.74× |
| nebius | qwen3_no_think | 6,447 / 300 | 67,229,517 | 4,557,486 | 14.75× |
| nebius | qwen3_think_keep [synthetic] | 6,447 / 300 | 67,293,987 | 4,621,956 | 14.56× |
| nebius | qwen3_think_drop [synthetic] | 6,447 / 300 | 67,293,987 | 4,621,956 | 14.56× |

在所有组合下，B 与 A 的 policy token（mask=1）总数完全相等，例如 SWE-smith × qwen3_no_think 两者都是 904,497。也就是说，合并失败**不会丢失训练信号**，代价是同样的 policy token 要搭配重复的历史 prefix 处理很多次，同时失去 trajectory 级别的整体奖励分配。合并成功的组合中，B 与 A 的 row_tokens 完全相同，倍数为 1.00×。

完整的 36 行结果表见 [`results/summary.md`](results/summary.md)。

## 4. 机制：结果为什么是 0% 或 100%

chat template 是确定性的，所以结果取决于"模板配置 × 消息角色约定"的组合，而不是数据分布。每一次失败都是同一个差异：Qwen3 模板在重渲染历史时删掉了助手消息开头的思考块。

Qwen3-8B 模板（revision `b968826`）对历史助手消息的处理：

1. 先找出 `last_query_index`，也就是最后一条"真正的用户提问"，即不是 `<tool_response>` 包裹的 `user` 消息。
2. 位于 `last_query_index` 之后的助手消息，只有在 `loop.last`（它是最后一条消息）或带有非空 `reasoning_content` 时，才渲染 `<think>…</think>` 块；其他助手消息一律不渲染。

对应三种失败：

- **`no_think`**：`enable_thinking=False` 时，生成提示末尾是 `<think>\n\n</think>\n\n`，模型从这里开始生成。下一轮，这条助手消息不再是最后一条，又没有 `reasoning_content`，空思考块就被删掉了。不需要任何合成数据就会触发。
- **`think_drop`**：生成时带思考块，但客户端没有把 `reasoning_content` 回传给服务端（例如只把 `content` 和 `tool_calls` 写回历史），下一轮思考块随之消失。本实验没有调查各个 agent 框架实际是否回传。
- **`think_keep` 在 nebius 上失败**：nebius 把工具观察放在 `user` 角色里，每一轮都会把 `last_query_index` 推到最新一条观察，于是之前的助手消息都落在"最后一次提问"之前，即使回传了 `reasoning_content`，思考块也会被删掉。SWE-smith 和 Toucan 用的是 `tool` 角色，所以 `think_keep` 能完整合并。

Qwen2.5 模板对助手消息的渲染与位置无关，所以 B 和 C 都 100% 合并。

## 5. 典型案例

以下是归因器输出的分歧片段（`prev` 是上一轮上下文，`next` 是下一轮 prompt，从分歧点向前截取），全部来自 `results/results.json` 的 `examples` 字段。

**案例 1：SWE-smith × qwen3_no_think × B**（`getmoto__moto.694ce1f4.pr_8342.mz3vj24e`，第 1 轮，分歧在第 1,991 个 token）

```text
prev: …<|im_start|>assistant\n<think>\n\n</think>\n\nI'll help you implement the necessary changes…
next: …<|im_start|>assistant\nI'll help you implement the necessary changes…
差异: 删除 "<think>\n\n</think>\n\n"
```

**案例 2：Toucan × qwen3_think_drop[synthetic] × B**（`5e4b21a0-885b-5cf0-9697-05b24116a47c`，第 1 轮，第 366 个 token）

```text
prev: …<|im_start|>assistant\n<think>\nLet me reason about the next step before acting.\n</think>\n\n<tool_call>\n{…
next: …<|im_start|>assistant\n<tool_call>\n{"name": "weather-query-server-get_forecast", …
差异: 删除整个思考块
```

**案例 3：nebius × qwen3_think_keep[synthetic] × B**（`9:Stratoscale__skipper-164`，第 1 轮，第 2,234 个 token）

```text
prev: …bash-$<|im_end|>\n<|im_start|>assistant\n<think>\nLet me reason about the next step before acting.\n</think>\n\nTo start addr…
next: …bash-$<|im_end|>\n<|im_start|>assistant\nTo start addressing the issue…
差异: 删除思考块；客户端回传了 reasoning_content，但工具观察以 user 角色出现，模板仍然删除
```

**案例 4：真实 tokenizer 的最小复现**（`tests/test_template_prefix_network.py`）：一次工具调用加一条最终回答，在 qwen3_no_think 下 B 方式产生 1 次 `template_mismatch`，差异片段是 `("<think>\n\n</think>\n\n", "")`；同一段对话在 qwen2.5 下合并成功。

## 6. 局限性

- **没有真实模型采样。** B 和 C 中的 response 是"模板认可的规范输出"。真实模型可能生成非规范的 token 切分，或者与模板 `tojson` 不同的工具调用 JSON 格式（经 vLLM 工具解析器再重新序列化之后就会不同），这两类都会造成不匹配，本实验**测不到**。因此 `retoken_mismatch = 0` 只说明 Qwen tokenizer 在"规范生成 + 单独分词"下边界稳定，**不能**推出真实训练里不存在重分词问题。
- **思考内容是合成的。** 三个数据集都没有独立的思考字段。`think_keep` 和 `think_drop` 注入的是一段固定文本；这两个配置只能说明"有没有思考内容、是否回传"这一机制，不能代表真实思考内容的长度分布。
- **只覆盖 2 个 tokenizer，都是 Qwen 系。** 其他家族（DeepSeek、GLM、gpt-oss 等）的模板行为没有测。`A_passthrough` 依赖 Qwen 模板"每条助手消息渲染一个 `<|im_start|>assistant`"这一结构。
- **规模**：每个数据集只用了一个分片里的 300 条轨迹，每条最多 50 轮。由于模板是确定性的，结论对抽样不敏感，但数据角色约定（`tool` 还是 `user`）会直接改变结果，其他数据集需要单独验证。
- **分类器是启发式的。** `template_mismatch` 的判定依赖一份模板标记列表、纯空白差异和工具调用 JSON 等价三条规则。本实验中所有失败的差异片段都是完整的思考块，属于明确情况；更复杂的改写可能被分到 `others`。

## 7. 对上游的建议

1. **填上归因字段。** 在 `merge_mismatch_rows` 里，用"分歧点 + 解码文本比较"判定 `retoken_mismatch`（文本仍是前缀，只有 id 不同）和 `template_mismatch`（差异只落在模板标记、空白或等价的工具调用 JSON 上）。上游已经在调用 `_decode_trace_text`，额外开销只是一次差异比对，并且只对最多 100 条上传到 wandb 的样本进行。
2. **在文档中说明 Qwen3 的合并前提。** 要让 `trajectory` 聚合真正合并，需要同时满足：开启思考模式、客户端回传 `reasoning_content`、工具观察使用 `tool` 角色。否则 `trajectory` 会悄悄退化成逐轮行，并带来上面那样的 token 量膨胀。
3. **拆行本身是正确的保守行为。** 不匹配时，模型在下一轮实际看到的就是被删掉思考块的 prompt。强行合并会让训练上下文与推理时的上下文不一致。所以建议是"让不匹配可见、可归因"，而不是放宽前缀检查。
4. **可选**：在训练指标里增加按原因分类的不匹配计数，例如 `training/n_trace_merge_mismatch_template`，方便用户发现自己的模板配置不适合 `trajectory` 聚合。
