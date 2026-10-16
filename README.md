# Trace2Batch

多轮 Agent 轨迹的数据表示与样本构建实验。

Trace2Batch 把模型响应、工具观察和奖励事件转换为可检查的 RL 样本行（prompt / response / policy mask / reward），对比逐轮 `transition` 与连续轨迹 `trajectory` 两种表示。核心逻辑参考 [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning) 的轨迹聚合实现（见下文"与上游的关系"），用 Python 标准库重写为可独立测试的参考实现，不需要 GPU 或外部模型服务。

## 与上游的关系

核心算法**不是原创**，参考自 agent-lightning 的 [`agentlightning/verl/rollout_adapter.py`](https://github.com/microsoft/agent-lightning/blob/69f5a40/agentlightning/verl/rollout_adapter.py)（MIT，commit `69f5a40`，2026-10-09）。派生文件保留了 Microsoft 的版权声明并注明来源，见 `LICENSE` 与各文件头。

| Trace2Batch（`trace2batch/core.py`） | 上游 `rollout_adapter.py`（@69f5a40） | 差异 |
|---|---|---|
| `build_samples` 内的 `emit()` | `append_training_row()`（约第 455 行） | 上游还处理 log_probs 对齐、padding、routed experts、多模态输入 |
| `trajectory` 前缀合并 | 第 540–650 行的合并循环，`ids_startswith`（第 46 行） | 逻辑基本一致 |
| `merge_mismatches` 计数 | `merge_mismatch_rows`（约第 595 行） | 上游会把不匹配样本上传到 wandb；两者都不区分不匹配原因 |
| 响应与 mask 同步截断 | `response_mask[:response_limit]` | 上游本来就这样做 |
| `max_total_length` | 同名参数（第 477–478 行） | 语义一致：响应上限取 `min(max_response_length, max_total_length - 截断后 prompt 长度)` |
| `is_drop` 标记过长 prompt | 同名逻辑 | 一致 |
| `reward_fillna_value`、零奖励 / 缺失奖励区分 | `_fillna_reward()`（第 896 行）、`sample_with_reward_count` | 语义一致 |
| `build_batch` 的 rollout 级汇总 | `n_rollouts_w_reward`、`n_unmerged_rollouts` 等指标 | 上游按 `(step, sample_idx)` 排序；这里保持输入顺序 |

**Trace2Batch 自己新增的部分：**

- 请求过滤的互斥计数（状态错误 / HTTP 错误 / 空响应）。上游在更早的 triplet 提取阶段处理这一步，这里把它整理进同一个流程。
- 四类"无可用行"原因诊断，区分输出行、标记丢弃行与参考可用行。
- 12 个带手算期望的合成场景，以及输入不变性、断网执行、跨进程确定性等测试。

## 核心功能

| 功能 | 实现方式 |
|---|---|
| 逐轮表示 `transition` | 每次有效模型响应形成一行，生成 token 的 mask 为 1 |
| 连续轨迹表示 `trajectory` | 按完整 token 前缀合并；前缀不连续时分行 |
| 工具观察处理 | 保留观察作为上下文，合并响应中的观察 mask 为 0 |
| 长度边界 | 响应与 mask 同步截断；可选 `max_total_length` 总长预算；过长 prompt 截断并标记丢弃 |
| 批处理 `build_batch` | 多个 rollout 逐个构建，行带 `rollout_id`，rollout 之间不合并，汇总诊断 |
| 请求过滤 | 按状态错误、HTTP 错误、空响应的优先级互斥计数 |
| 奖励与诊断 | 提取最后 reward，保留零奖励与缺失奖励的区别，缺失时按 `reward_fillna_value` 填充 |

## 实验结果

12 个固定场景分别执行两种表示，产生 24 组结果；每组结果均核对手算期望。以下是实际输出：

| 场景 | 逐轮表示：行数 / policy token | 连续轨迹：行数 / policy token | 验证结论 |
|---|---|---|---|
| 两轮连续工具调用 | 2 / 3 | 1 / 3 | 2 个观察 token 保留，mask 为 0 |
| 三轮连续工具调用 | 3 / 4 | 1 / 4 | 合并后保留 4 个生成 token 与 3 个观察 token |
| 前缀不匹配 | 2 / 3 | 2 / 3 | 不强行合并，记录一次不匹配 |
| 响应上限为 3 | 2 / 3 | 1 / 2 | 观察占用合并响应预算，截断可能丢失后续生成 token |

三轮轨迹的合并结果：

```text
response: [10, 90, 11, 91, 92, 12, 13]
mask:     [ 1,  0,  1,  0,  0,  1,  1]
```

实验数据为人工构造的 token 符号，不对应真实 tokenizer 词表。上述结果用于验证数据表示，不作为模型训练效果或性能提升的证据。

## 研究：真实 chat template 下的前缀一致性

上游在 trajectory 合并失败时，`template_mismatch` / `retoken_mismatch` 恒为 `False`，不区分原因。`trace2batch/mismatch.py` 提供一个零依赖的归因器（retoken / template / others）。`experiments/template_prefix/` 用真实 tokenizer（Qwen2.5-7B-Instruct、Qwen3-8B，只下载 tokenizer 文件）和 3 个公开 Agent 轨迹数据集（各 300 条，共 900 条），模拟上游"代理 + vLLM `return_token_ids`"的采集链路：

- Qwen2.5：三个数据集都 100% 合并。
- Qwen3 `enable_thinking=False`：11,984 次合并尝试**全部失败**，全部归因为 `template_mismatch`（历史中的空思考块被模板删除）；行数退化为轮数，不截断时行内 token 总量为对照组的 1.75×–15.75×。
- Qwen3 思考模式（思考内容为合成注入）：客户端不回传 `reasoning_content` 时全部失败；回传时，工具观察用 `tool` 角色的数据集能合并，用 `user` 角色的 nebius 数据集仍然全部失败。
- 局限：没有真实模型采样，测不到非规范切分导致的 `retoken_mismatch`；只覆盖 Qwen 系模板。

详见 [REPORT.md](experiments/template_prefix/REPORT.md) 和 [DATA.md](experiments/template_prefix/DATA.md)。已向上游提交 Issue [microsoft/agent-lightning#645](https://github.com/microsoft/agent-lightning/issues/645)，等待维护者回应。复现：`pip install -e ".[real]" && python -m experiments.template_prefix.run`。

## 安装与运行

需要 Python >=3.12。核心模块没有运行时依赖：

```bash
pip install -e ".[dev]"
python -m examples.trace2batch.run
```

程序读取固定场景并输出 JSON，包含样本行、过滤计数、policy/观察 token 数、奖励状态与无可用行原因。手算期望不匹配时失败退出。报告标记 `synthetic=true`、`real_training=false`。

核心接口：

```python
from trace2batch.core import build_batch, build_samples

events = [{"event_type": "model_request", "data": {"prompt_token_ids": [1], "response_token_ids": [10, 11]}}]
result = build_samples(events, mode="trajectory", max_prompt_length=16, max_response_length=16)
print(result.summary())

batch = build_batch({"r1": events, "r2": events}, mode="trajectory", max_prompt_length=16, max_response_length=16)
print([item.rollout_id for item in batch.rows], batch.summary()["n_rollouts"])
```

## 测试

```bash
python -m pytest -q
ruff check .
ruff format --check .
pyright
```

需要联网的测试用 `@pytest.mark.network` 标记，默认跳过，`python -m pytest --run-network` 启用。GitHub Actions 在 Python 3.12 / 3.13 上运行上述检查。

## 目录结构

```text
Trace2Batch/
├── trace2batch/
│   ├── __init__.py
│   ├── core.py                 # 样本构建、批处理与诊断
│   └── mismatch.py             # 前缀不匹配定位与原因归因
├── examples/trace2batch/
│   ├── run.py                  # 固定实验入口
│   ├── scenarios.json          # 合成事件与手算期望
│   └── README.md               # 输入、输出与字段说明
├── experiments/template_prefix/
│   ├── run.py                  # 真实 tokenizer 前缀一致性实验
│   ├── DATA.md / REPORT.md     # 数据说明与研究报告
│   └── results/                # 结果 JSON、汇总表与样本 id
├── tests/
│   ├── test_trace2batch.py      # 核心逻辑与边界测试
│   ├── test_build_batch.py      # 批处理测试
│   ├── test_mismatch.py         # 归因器测试（手工 decode 夹具）
│   ├── test_template_prefix_network.py  # 真实 tokenizer 复现（network，默认跳过）
│   └── test_trace2batch_cli.py  # 入口、断网与确定性测试
├── .github/workflows/ci.yml
├── pyproject.toml
└── LICENSE                     # MIT
```

## 实现边界

核心模块实现的是未填充样本的数据处理与参考验证，不包括模型调用、log_probs、padding、优势/loss 计算、参数更新或部署。

## English summary

Trace2Batch is a dependency-free, testable reference reimplementation of the trajectory aggregation logic in microsoft/agent-lightning (`agentlightning/verl/rollout_adapter.py` @ `69f5a40`, MIT). It turns multi-turn agent events into RL sample rows (prompt / response / policy mask / reward) in `transition` or `trajectory` mode, with upstream-compatible `max_total_length` and `reward_fillna_value`, multi-rollout batching, and filter / zero-usable-row diagnostics. The core algorithm is derived from upstream; see the mapping table above for what is new. It also adds a dependency-free prefix-mismatch classifier (retoken / template / others) and a study with real Qwen2.5 / Qwen3 tokenizers on 900 public agent trajectories: under Qwen3 with `enable_thinking=False`, all 11,984 merge attempts fail as template mismatches (the empty think block is dropped from history), while Qwen2.5 merges 100%. See `experiments/template_prefix/REPORT.md`.
