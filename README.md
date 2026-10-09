# Trace2Batch

多轮 Agent 轨迹的数据表示与样本构建实验。

这是一个聚焦训练数据语义的个人技术项目：将模型响应、工具观察和奖励事件转换为可检查的样本行，对比逐轮与连续轨迹两种表示，并用固定输入、手算期望和自动化测试验证边界行为。运行仅使用 Python 标准库，不需要 GPU 或外部模型服务。

## 项目工作

- 实现样本构建模块，将请求过滤、奖励提取、轨迹分组、policy mask 和长度截断组织为可独立运行的数据处理流程。
- 设计 12 个合成场景，覆盖连续工具调用、前缀不匹配、错误请求、响应截断、过长 prompt、零响应预算及缺失奖励。
- 实现互斥过滤计数和四类无可用行诊断，区分输出行、标记丢弃行与参考可用行。
- 编写 51 项自动化测试，验证精确 token/mask 序列、输入不变性、错误期望失败、断网执行和跨进程结果一致性。

## 核心功能

| 功能 | 实现方式 |
|---|---|
| 逐轮表示 `transition` | 每次有效模型响应形成一行，生成 token 的 mask 为 1 |
| 连续轨迹表示 `trajectory` | 按完整 token 前缀合并；前缀不连续时分行 |
| 工具观察处理 | 保留观察作为上下文，合并响应中的观察 mask 为 0 |
| 长度边界 | 响应与 mask 同步截断；过长 prompt 截断并标记丢弃 |
| 请求过滤 | 按状态错误、HTTP 错误、空响应的优先级互斥计数 |
| 奖励与诊断 | 提取最后 reward，保留零奖励与缺失奖励的区别，解释无可用行原因 |

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

## 快速运行

需要 Python >=3.12，在项目根目录执行：

```bash
python -m examples.trace2batch.run
```

程序读取固定场景并输出 JSON，包含样本行、过滤计数、policy/观察 token 数、奖励状态与无可用行原因。手算期望不匹配时失败退出。报告标记 `synthetic=true`、`real_training=false`。

核心接口：

```python
from trace2batch.core import build_samples

result = build_samples(
    [{"event_type": "model_request", "data": {
        "prompt_token_ids": [1], "response_token_ids": [10, 11]
    }}],
    mode="trajectory",
    max_prompt_length=16,
    max_response_length=16,
)
print(result.summary())
```

## 测试与技术栈

运行：Python 标准库。开发验证：pytest、Ruff、Pyright。

在已有对应开发工具的环境中执行：

```bash
python -m pytest tests -q
ruff check .
ruff format --check .
pyright
```

本地验证：51 项测试通过，格式与类型检查通过；关闭第三方包加载后，固定实验仍可运行。

## 目录结构

```text
Trace2Batch/
├── trace2batch/
│   ├── __init__.py
│   └── core.py                 # 样本构建与诊断
├── examples/trace2batch/
│   ├── run.py                  # 固定实验入口
│   ├── scenarios.json          # 合成事件与手算期望
│   └── README.md               # 输入、输出与字段说明
├── tests/
│   ├── test_trace2batch.py      # 核心逻辑与边界测试
│   └── test_trace2batch_cli.py  # 入口、断网与确定性测试
├── pyproject.toml              # 开发检查配置
└── LICENSE                     # MIT 许可证
```

## 实现边界

项目实现的是未填充样本的数据处理与参考验证，不包括模型调用、优势/loss 计算、参数更新或部署。现有许可证与源码版权声明予以保留。
