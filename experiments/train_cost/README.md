# P3-A：逐轮与合并样本的真实训练成本

本实验比较同一份 token 事件在 `transition` 与 `trajectory` 表示下的训练成本。执行真实前向、反向传播和 LoRA 参数更新；目标是 masked cross-entropy，不是 GRPO。下文说明公平对照的做法、本地准备、训练输出和 Kaggle 提交流程。

## 公平对照

- 使用固定 revision 的 Qwen3-0.6B、关闭思考和 v2 前缀稳定模板，先生成一份规范 token 事件，再构建两种表示。
- 每个监督目标的 token ID 与此前完整上下文逐一相等；只比较 token 总数不够。
- prompt 和工具观察只作为上下文。正确做 next-token shift，只计算模型生成 token 的 CE。
- 无 padding，microbatch=1；每条 rollout 的所有行累积梯度，loss 按该 rollout 的 policy token 数归一化，只更新一次。
- 每个条件都从相同的初始 LoRA 参数、全新 AdamW 与 GradScaler 开始。每个条件预热后再次重置；默认两次重复采用 AB/BA 顺序。
- LoRA：`r=8`、`alpha=16`、`dropout=0`、`q_proj/v_proj`、`bias=none`；关闭其他 dropout 和 KV cache，默认梯度检查点开启。CUDA fp16，CPU float32。
- 正式计时前，核对每个分组首个配对样本的 loss 和所有可训练参数梯度。核对使用 training 模式（Hugging Face 只在 training 模式下启用梯度检查点；所有 dropout 必须为 0），并像训练一样做 loss scaling（固定 4096），存储梯度前再除回去。通过条件为：loss 相对误差 ≤1e-3、梯度相对 L2 误差 ≤0.02、余弦 ≥0.9998，且梯度存在性一致。阈值介于 T4 上 fp16 的数值噪声与单个上下文 token 被改动时的梯度差异之间。逐元素 `allclose` 结果只作诊断记录。

每 rollout 等权，而不是把不同 rollout 长度混为相同 token 权重。这一规则在两种表示中一致；正式研究需明确该目标与实际训练栈是否相符。

## 本地准备和预检

从仓库根目录执行；只运行此阶段不使用 GPU。`prepare` 强制读取已有缓存，缺少固定 revision 的数据或 tokenizer 就报错，不加载模型权重。

```bash
python -m pip install -e ".[dev,train]"
python -m experiments.train_cost.prepare --out kaggle/p3_inputs/prepared.json
python -m experiments.train_cost.run --data kaggle/p3_inputs/prepared.json --out kaggle/p3_inputs/check --check-only
```

默认（pilot）每个数据集最多选择 4 条、每条最多前 4 个 assistant 轮次、完整上下文总长不超过 4096。这里取的是轨迹前缀片段，不能称作原轨迹的完整训练。前缀内任何行都不截断；超长、非法或无法保持前缀的候选显式跳过并计数。没有足够样本就如实报告，不能偷偷减少行长度。

按长度分层时，每条候选取层内最长的完整轮次前缀（至少 2 轮），从高层到低层按固定种子顺序挑选，每条候选只用一次；`--cells` 只决定输出哪些格，不影响选择：

```bash
python -m experiments.train_cost.prepare --out all.json --strata 0-2048,2048-4096,4096-6144 \
  --samples 6 --min-turns 2 --max-turns 20
```

每条 rollout 记录 `stratum`，加载时会重新核验最终完整上下文确实落在该层内；核对与计时按“数据集|层”分组。

输出包含样本 ID、逐条筛选原因、固定数据 revision 与分片 hash、模板/tokenizer hash、准备库版本。`run --check-only` 重新审计这些事件，不导入 torch/Transformers/PEFT，不下载权重，不覆盖已有结果。

## 训练和输出

```bash
python -m experiments.train_cost.run --data kaggle/p3_inputs/prepared.json --out kaggle/results/p3_pilot --device cuda --time-budget-seconds 2400
```

训练依赖可选、按需导入。模型与 revision 固定为 `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`，加载 `trust_remote_code=False`。本地 CPU 可用 `--device cpu` 做小样本 smoke；它不能预测 T4 的速度或显存。

`results.json` 原子写入启动、校验、每个条件和最终状态；每个条件还记录各条 rollout 的耗时（CUDA event，不额外同步，条件总时长仍按墙钟计）。`--diagnose-gradients` 只在多个 loss scale 下比较每一对样本的梯度（`--loss-scales`），不计时，也不因核对未通过而停止。记录数据 SHA256、训练配置、实际依赖版本、模型加载耗时、各条件实际完成的 rollout/行/token/更新次数、计时、有效 policy tokens/s、处理 tokens/s 和 CUDA 峰值 allocated/reserved 显存。CPU 显存字段为 `null`。

计时包含输入张量构造、前向、masked loss、反向、有限性检查和 AdamW 更新；不包含模型加载、预热、初始化参数恢复和新优化器创建。CUDA 前后同步。有限性检查会产生同步开销，因此这是带校验的参考训练循环，不代表所有生产训练栈。

只有所有重复的两种条件均完整完成时，才给出 `speed_ratio = transition_seconds / trajectory_seconds`。大于 1 表示该循环下合并更快。超时、OOM、非有限 loss/梯度、单边完成都必须报告；它们的比值为 `null`，不能作为完整实验结论。

## Kaggle 驱动与提交前检查

私人模板是本地 `kaggle/train_cost_driver.py`。用下列命令准备提交包；账号标识只放在本地 metadata。

```bash
python -m experiments.train_cost.bundle --data kaggle/p3_inputs/prepared.json --out kaggle/p3_pilot --owner YOUR_KAGGLE_ID --driver kaggle/train_cost_driver.py
```

规模上限是显式的配置档：`--profile pilot`（默认：≤4096、≤12 条、每个数据集 ≤4 条、≤4 轮）或 `--profile stratified`（≤6144、每个包 ≤48 条、每格 ≤6 条、≤20 轮，并要求每条都有 stratum）。`--session-limit-minutes`（10–720 的整数）和 `--run-mode {benchmark,diagnose}` 在打包时写入驱动，提交时的超时必须与写入值相同。

包内只有 `pilot.py` 和私有 `kernel-metadata.json`。`pilot.py` 内嵌允许清单中的源码及 prepared JSON，并记录压缩包 SHA256；不依赖尚未推送的 GitHub 代码，不包含账号凭据、缓存、虚拟环境或其他 Kaggle 文件。运行前校验大小、哈希、路径、源码语法、数据配对及模型限制。

节约 GPU 时间的措施：

1. 在安装依赖、下载权重之前检查 CUDA、预装 torch 和数据。
2. 只跑一次，失败不自动重试，也不自动重提任务。
3. 独立虚拟环境复用预装 torch。pip 先 dry-run，若计划安装 torch 或 vLLM 则停止；不安装 vLLM。
4. 固定 Transformers 5.20.0、PEFT 0.21.2 和 torchao 0.17.0。Kaggle 镜像预装的 torchao 0.10.0 会让 PEFT 0.21.2 在注入 LoRA 时报错，而 0.17.0 是官方标明兼容 torch 2.11 的版本。安装后先在微型随机模型上离线做一次 LoRA 前向和反向，再下载真实权重。记录实际导入的版本，以及被遮蔽的系统副本。`pip check` 只对运行依赖闭包内的冲突判失败。
5. 每个安装、探测和训练子进程都有上限；驱动总硬界早于打包写入的会话上限，训练软停止留出写盘时间。虚拟环境用 `--without-pip` 创建并复用系统 pip（Kaggle 的 Python 3.13 没有 ensurepip）。
6. 虚拟环境、解包源码和下载缓存放临时目录；需要保存的日志、JSON 放 `/kaggle/working`。
7. pilot 使用 GPU 0 独立计时。即使会话提供两张卡，也先不启动未经验证的并行训练。

`driver-status.json` 保存驱动的阶段、总时间、失败原因；`driver.log` 保存子进程日志，`dependency-versions.json` 保存实际环境。读取驱动状态及训练结果两者，不能仅凭退出码宣称完成。

用户审阅配置、样本规模、包哈希和时间预算后，才可提交；提交时的超时等于包内写入的会话上限。运行前后查询额度，按实际扣减核算成本。

## 检查与局限

```bash
python -m pytest -q
ruff check .
ruff format --check .
pyright --pythonpath .venv/bin/python
```

无 torch 的 `[dev]` 环境会跳过真实 torch 单元测试，数据/CLI/打包/驱动测试仍运行；安装 `[train]` 的环境需要运行全部训练测试。需要 tokenizer 的既有测试仍由 `--run-network` 控制。

这是带校验的参考训练循环上的成本基准：结果只适用于所测的模型、硬件、精度和样本设置；样本是轨迹前缀片段，按层选样时轮数与数据集、层可能混杂；不能外推到更长上下文、其他模型、全参数或多卡训练。约 8k 以上需要改用不物化注意力矩阵的路径。v2 对模型效果的影响、完整 RL 学习效果仍需独立研究。

训练接口参考：[PEFT Quicktour](https://huggingface.co/docs/peft/main/quicktour)、[LoRA 配置](https://huggingface.co/docs/peft/main/package_reference/lora)。
