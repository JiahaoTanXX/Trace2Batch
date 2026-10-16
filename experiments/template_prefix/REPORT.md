# 真实 chat template 下的 trajectory 前缀一致性

> 所有数字都来自 `python -m experiments.template_prefix.run` 的一次完整运行，原始结果在 `results/results.json` 和 `results/summary.md`。数据与版本见 [DATA.md](DATA.md)。
>
> **2026-10-16 更新**：从 Qwen 系的 2 个 tokenizer（4 种配置）扩展到 4 个模型家族的 5 个 tokenizer（9 种配置），新增 DeepSeek-V3.1、GLM-4.5、gpt-oss-20b。Qwen 部分的数字与扩展前的两次运行完全一致。

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
- **Tokenizer**：4 个模型家族共 5 个 tokenizer，只下载 tokenizer 和配置文件，revision 固定，共 9 种配置：
  - Qwen2.5-7B-Instruct：`qwen2.5`
  - Qwen3-8B 与 GLM-4.5 各 3 种：
    - `no_think`：`enable_thinking=False`，不注入任何内容
    - `think_keep[synthetic]`：注入合成思考内容，客户端在历史里回传 `reasoning_content`
    - `think_drop[synthetic]`：注入合成思考内容，客户端不回传
  - DeepSeek-V3.1：`deepseek_v3.1`（默认非思考模式）
  - gpt-oss-20b：`gpt-oss`（默认 `reasoning_effort=medium`）
  - 各家族的结束标记、工具参数格式等处理细节见 DATA.md。
- **三种采集方式**：
  - **A 直通**（对照组）：框架直接拼接 token，只对新消息分词。依赖 Qwen 模板结构，只对 Qwen 的 4 种配置计算。
  - **B vLLM token id**：每轮由服务端对完整 messages 重新渲染并分词作为 prompt；response 是该助手消息在上下文中的规范 token。这对应上游的实际链路（`server/proxy.py` 在训练模式下设置 `return_token_ids=True`）。工具参数在渲染前解析成对象，与 vLLM 的 `_postprocess_messages` 一致。
  - **C 文本重分词**：与 B 相同，但 response 由生成文本单独分词。与 B 对比，可以单独看出边界切分差异的影响。
- 每轮事件喂给 `trace2batch.core.build_samples(mode="trajectory")`，再用 `trace2batch.mismatch.explain_mismatch` 对每一次失败归因。归因次数与 `build_samples` 的 `merge_mismatches` 逐条轨迹核对一致，不一致时脚本直接报错。
- **规模**：每种配置下，SWE-smith 有 5,416 次合并尝试，Toucan 有 421 次，nebius 有 6,147 次，合计 11,984 次。例外：GLM-4.5 和 gpt-oss 的模板分别在 1 条和 2 条 Toucan 对话上渲染报错，这些对话被跳过，见第 6 节。9 种配置的 B 方式共 107,848 次合并尝试；全部单元共 66 个（Qwen 4 种配置 × 3 个数据集 × 3 种方式 = 36，其余 5 种配置 × 3 × 2 = 30）。

## 3. 结果

A 直通在 Qwen 的全部 12 个"数据集 × 配置"组合下合并成功率都是 100%，说明实验框架本身没有引入不匹配。B 和 C 在所有 27 个组合下结果完全相同，下表只列 B：

| 配置 | SWE-smith（5,416） | Toucan（421；GLM / gpt-oss 为 419） | nebius（6,147） |
|---|---|---|---|
| qwen2.5 | 100% | 100% | 100% |
| qwen3_no_think | **0%** | **0%** | **0%** |
| qwen3_think_keep [synthetic] | 100% | 100% | **0%** |
| qwen3_think_drop [synthetic] | **0%** | **0%** | **0%** |
| deepseek_v3.1 | 100% | 100% | 100% |
| glm4.5_no_think | 100% | 100% | 100% |
| glm4.5_think_keep [synthetic] | 100% | 100% | **0%** |
| glm4.5_think_drop [synthetic] | **0%** | **0%** | **0%** |
| gpt-oss | 100% | 100% | **0%** |

（表中为合并成功率。）

- B 方式在 107,848 次合并尝试中共失败 54,391 次，**全部被归为 `template_mismatch`**，`retoken_mismatch` 和 `others_mismatch` 都是 0。
- 只要不匹配，就是**每一轮都不匹配**：`trajectory` 模式的行数完全退化成轮数。平均每条轨迹的行数：SWE-smith 19.05，Toucan 2.40，nebius 21.49。
- 只看不注入合成数据的 5 种配置（qwen2.5、qwen3_no_think、deepseek_v3.1、glm4.5_no_think、gpt-oss），15 个"数据集 × 配置"组合中有 4 个全部失败：qwen3_no_think 的 3 个，以及 gpt-oss × nebius。

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

其他家族没有 A 对照组，但规律相同。例如 GLM-4.5 在 SWE-smith 上，think_drop（全部失败）的 row_tokens 为 81,858,276，是同样内容、全部合并的 think_keep（5,223,525）的 15.67 倍。

在 Qwen 的所有组合下，B 与 A 的 policy token（mask=1）总数完全相等，例如 SWE-smith × qwen3_no_think 两者都是 904,497。也就是说，合并失败**不会丢失训练信号**，代价是同样的 policy token 要搭配重复的历史 prefix 处理很多次，同时失去 trajectory 级别的整体奖励分配。合并成功的组合中，B 与 A 的 row_tokens 完全相同，倍数为 1.00×。

完整结果表见 [`results/summary.md`](results/summary.md)。

## 4. 机制：结果为什么是 0% 或 100%

chat template 是确定性的，所以结果取决于"模板 × 配置 × 消息角色约定"的组合，而不是数据分布。所有失败可以归结为三种模板行为：

**(1) 历史中删除或清空思考内容（Qwen3、GLM-4.5）。** 两个模板都会先找出最后一条真正的用户提问（`last_query_index`）。只有位于它之后、并且带有思考内容的助手消息，才在历史中保留思考：

- Qwen3（revision `b968826`）：不满足条件时，整个 `<think>…</think>` 块被删除。另外，`enable_thinking=False` 时生成提示末尾是 `<think>\n\n</think>\n\n`，进入历史后这个空块也会被删除，不需要任何合成数据就会触发。
- GLM-4.5（revision `cbb2c7c`）：不满足条件时，渲染为空的 `<think></think>`。这正是 `enable_thinking=False` 时生成提示的形式，所以 GLM 关闭思考时前缀稳定、100% 合并；开启思考时，思考内容被清空，标签保留。
- 两者共同的失败模式：
  - **客户端不回传 `reasoning_content`**（think_drop），思考在下一轮消失。
  - **工具观察以 `user` 角色出现**（nebius），每一轮都会把 `last_query_index` 推到最新一条观察，之前的助手消息都落在"最后一次提问"之前，即使回传了思考内容也会被删除。SWE-smith 和 Toucan 用 `tool` 角色，所以 think_keep 能完整合并。

**(2) 生成结束 token 在历史中被改写（gpt-oss）。** gpt-oss 模板（revision `6cee5e8`）中，作为最后一条消息的最终回答以 `<|return|>` 结尾（模型生成时的停止 token），进入历史后被渲染为 `<|end|>`。所以只要一条最终回答之后还有下一轮，前缀就会断开。nebius 中每条助手消息都是最终回答，100% 失败。SWE-smith 和 Toucan 的中间轮次都是工具调用，以 `<|call|>` 结尾、历史中不变，最终回答只出现在最后一轮，所以 100% 合并。

**(3) 前缀稳定的模板（Qwen2.5、DeepSeek-V3.1、GLM-4.5 关闭思考）。** 助手消息的渲染与它在对话中的位置无关，B 和 C 都 100% 合并。DeepSeek-V3.1 在非思考模式下，生成提示与历史中都使用 `<｜Assistant｜></think>`，保持一致。

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

**案例 4：SWE-smith × glm4.5_think_drop[synthetic] × B**（`getmoto__moto.694ce1f4.pr_8342.mz3vj24e`，第 1 轮，第 1,883 个 token）

```text
prev: …<|assistant|>\n<think>Let me reason about the next step before acting.</think>\nI'll help you implement…
next: …<|assistant|>\n<think></think>\nI'll help you implement the necessary changes…
差异: 思考内容被清空，标签保留
```

**案例 5：nebius × gpt-oss × B**（`9:Stratoscale__skipper-164`，第 1 轮，第 2,387 个 token）

```text
prev: …skipper --version\n```<|return|>
next: …skipper --version\n```<|end|><|start|>user<|message|>Error: no such option: --version…
差异: 生成结束 token <|return|> 在历史中变为 <|end|>
```

**案例 6：真实 tokenizer 的最小复现**（`tests/test_template_prefix_network.py`，默认跳过，`--run-network` 启用）：
- 一次工具调用加一条最终回答：在 qwen3_no_think 下 B 方式产生 1 次 `template_mismatch`，差异片段是 `("<think>\n\n</think>\n\n", "")`；同一段对话在 qwen2.5 下合并成功。
- 两轮问答：在 gpt-oss 下产生 1 次 `template_mismatch`，差异从 `<|return|>` 开始。

## 6. 局限性

- **没有真实模型采样。** B 和 C 中的 response 是"模板认可的规范输出"。真实模型可能生成非规范的 token 切分，或者与模板 `tojson` 不同的工具调用 JSON 格式（经 vLLM 工具解析器再重新序列化之后就会不同），这两类都会造成不匹配，本实验**测不到**。因此 `retoken_mismatch = 0` 只说明 Qwen tokenizer 在"规范生成 + 单独分词"下边界稳定，**不能**推出真实训练里不存在重分词问题。
- **思考内容是合成的。** 三个数据集都没有独立的思考字段。`think_keep` 和 `think_drop` 注入的是一段固定文本；这两个配置只能说明"有没有思考内容、是否回传"这一机制，不能代表真实思考内容的长度分布。
- **覆盖 4 个家族、5 个 tokenizer**，没有覆盖 Llama（需要申请权限）、Mistral、Kimi 等。
  - `A_passthrough` 依赖 Qwen 模板的结构，只对 Qwen 计算。
  - DeepSeek-V3.1 只测了非思考模式：它的模板在历史中总是去掉思考内容，也无法渲染思考模式下的生成，按本实验的方法构造不出该模式的生成。
- **模板渲染报错**：
  - gpt-oss 有 2 条 Toucan 对话报错，原因是工具 `description` 为 `null`。
  - GLM-4.5 有 1 条 Toucan 对话报错，原因是工具参数不是合法 JSON。
  - 这些对话被跳过。它们本身反映了模板对数据格式的脆弱性，真实训练中也会导致请求失败。
- **数据完整性问题**：gpt-oss 模板每条消息只渲染第一个工具调用，并行调用会被静默丢掉。这不会造成前缀不匹配，所以不在合并成功率里体现。
- **规模**：每个数据集只用了一个分片里的 300 条轨迹，每条最多 50 轮。由于模板是确定性的，结论对抽样不敏感，但数据角色约定（`tool` 还是 `user`）会直接改变结果，其他数据集需要单独验证。
- **分类器是启发式的。** `template_mismatch` 的判定依赖四条规则：与模板标记（含各 tokenizer 的特殊 token）重叠、纯空白差异、工具调用 JSON 等价、差异位于思考块内。扩展过程中修过两个误判：GLM 清空思考内容但保留标签，以及同一段文本重复出现导致对齐窗口错位。修复后本次运行 `others_mismatch` 为 0，但更复杂的改写仍可能被分到 `others`。

## 7. 对上游的建议

以下建议已整理为上游 Issue [microsoft/agent-lightning#645](https://github.com/microsoft/agent-lightning/issues/645)（原文见 [`drafts/upstream_issue.md`](../../drafts/upstream_issue.md)）。

1. **填上归因字段。** 在 `merge_mismatch_rows` 里，用"分歧点 + 解码文本比较"判定 `retoken_mismatch`（文本仍是前缀，只有 id 不同）和 `template_mismatch`（差异只落在模板标记、空白或等价的工具调用 JSON 上）。上游已经在调用 `_decode_trace_text`，额外开销只是一次差异比对，并且只对最多 100 条上传到 wandb 的样本进行。
2. **在文档中说明不同模板的合并前提。**
   - Qwen3 需要同时满足：开启思考模式、客户端回传 `reasoning_content`、工具观察使用 `tool` 角色。
   - GLM-4.5 关闭思考即可稳定合并；开启思考时，条件与 Qwen3 相同。
   - gpt-oss 在出现最终回答之后的轮次无法合并，因为 `<|return|>` 会变成 `<|end|>`。
   - 不满足条件时，`trajectory` 会悄悄退化成逐轮行，并带来上面那样的 token 量膨胀。
3. **拆行本身是正确的保守行为。** 不匹配时，模型在下一轮实际看到的就是被删掉思考块的 prompt。强行合并会让训练上下文与推理时的上下文不一致。所以建议是"让不匹配可见、可归因"，而不是放宽前缀检查。
4. **可选**：在训练指标里增加按原因分类的不匹配计数，例如 `training/n_trace_merge_mismatch_template`，方便用户发现自己的模板配置不适合 `trajectory` 聚合。
