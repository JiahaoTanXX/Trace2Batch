# 固定轨迹实验说明

本目录提供 12 个合成场景、两种样本表示及手算期望核验。运行命令：

```bash
python -m examples.trace2batch.run
```

## 输入

`scenarios.json` 的每个场景包含：

- `id`、`description`：标识与说明。
- `events`：单次 rollout 的已投影 token 事件。
- `max_prompt_length`、`max_response_length`：非负整数长度预算。
- `expected`：两种模式的手算指标期望。
- `expected_reason`：可选的无可用行原因期望。

模型事件使用 `event_type=model_request`，其 `data` 包含 `prompt_token_ids`、`response_token_ids`，并可提供 `status`、`http_status`。奖励事件使用 `event_type=reward` 与 `data.value`。

所有 token 均为人工整数符号；输入已投影，不负责原始模型请求的解析、去重或采集。

## 处理流程

1. 请求按状态错误、HTTP 错误和空响应的优先级互斥过滤；提取最后一个 reward。
2. `transition` 按有效轮次分行；`trajectory` 根据完整 token 前缀合并，观察 mask=0，生成 mask=1。
3. 分组后同步截断响应与 mask；没有 policy token 的行不输出，过长 prompt 截断并标记 `is_drop`。
4. 汇总计数并核对手算期望，输出确定性 JSON。

## 输出字段

报告含 `project_name=Trace2Batch`、`implementation=stdlib_cpu_reference`、`synthetic=true`、`real_training=false`、场景数与结果数。

每个场景-模式结果含 `rows`、`diagnostics`、`summary`、`final_reward` 与 `zero_usable_rows_reason`。

样本行：

| 字段 | 含义 |
|---|---|
| `prompt_token_ids` | 截断后的 prompt |
| `response_token_ids` | 未填充、截断后的 response |
| `response_mask` | 仅覆盖 response 的 0/1 policy mask，不是 attention mask |
| `source_turn_indices` | 过滤后有效模型事件的序号 |
| `reward` | 最后 reward，缺失时填零 |
| `is_drop` | 原始 prompt 是否超过上限 |

`has_reward` 保留奖励存在性，因此实际零奖励与缺失奖励不会混淆。

## 指标与诊断

- `n_rows`：所有输出行；`usable_rows`：排除 `is_drop` 后的参考可用行。
- `policy_tokens`：所有输出行的 mask=1 数量；`usable_policy_tokens`：参考可用行的相应数量。
- `observation_tokens`：所有输出行中 mask=0 的观察 token 数量。
- 请求过滤、响应截断、零 policy 行与前缀不匹配分别计数。

无参考可用行时，返回以下互斥原因之一：

```text
no_model_requests
all_model_requests_filtered
no_policy_tokens_after_truncation
all_rows_marked_drop
```

输出行不等于可用行；可用 token 数不等于实际训练收益。实验不构造 tensor、padding、attention mask 或实际 loss，也不运行模型训练。
