# 数据与 tokenizer 说明

所有文件只从 Hugging Face 下载到本地缓存（默认 `~/.cache/trace2batch-hf`），不提交进仓库，也不执行其中的任何代码（`trust_remote_code=False`）。只下载 tokenizer 文件和模型的 `config.json`，不下载模型权重。每个数据集只下载一个 parquet 分片，在分片内用固定种子抽样。抽中的样本 id 保存在 `results/samples_<数据集>.json`。

## 数据集

| 名称（脚本内） | 仓库 | revision | 使用的分片 | 许可证 | 分片内可用轨迹 | 抽样 |
|---|---|---|---|---|---:|---:|
| `swe_smith_tool` | [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | `08e109b4a59eaeebf80e4675cd125d42e7ac99a4` | `data/tool-00004-of-00008.parquet`（3,012 行） | MIT | 785 | 300 |
| `toucan_qwen3` | [Agent-Ark/Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) | `0df3cf37f2abefb380370cfb02eabea2a35ae782` | `Qwen3/train-00009-of-00044.parquet`（12,537 行） | Apache-2.0 | 12,537 | 300 |
| `nebius_swe_agent` | [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | `68195a1450865274106246d0d0296a1d6807b88e` | `data/train-00006-of-00012.parquet`（6,670 行） | CC-BY-4.0（需署名：Nebius） | 6,664 | 300 |

"可用轨迹"指规范化之后至少有 2 条助手消息的轨迹，只有这样才存在合并尝试。抽样方式：`random.Random(0).sample`，然后按分片内顺序排列。分片选的是体积较小的那个，没有其他依据。

### 字段格式与规范化

规范化之后统一为 OpenAI 风格的 `messages`（`system` / `user` / `assistant` / `tool`）。工具参数在渲染前解析成对象，与 vLLM 的行为一致（`vllm/renderers/chat_utils/chat_message.py` 中的 `_postprocess_messages`，vllm@`3ddc670`）。

- **SWE-smith（`tool` split）**：`messages` 是 JSON 字符串。该分片里只有 785 / 3,012 行带结构化的 `tool_calls`，其余是写在文本里的 XML 动作，这部分**只取前者**。`content` 可能是 `[{"type": "text", ...}]` 列表，规范化时拼接成字符串。这 785 条轨迹全部由 claude-3-5-sonnet-20241022 生成，数据中不含工具 schema，所以渲染时不传 `tools`。
- **Toucan（`Qwen3` 配置）**：每条轨迹只有一轮用户提问，工具调用用旧式 `function_call` 字段，工具结果的角色是 `function`。数据里的 system 消息是预先渲染好的工具提示，因此**丢弃它**，改为把 `available_tools` 作为 `tools` 交给模板渲染。并行工具调用在数据中存成相邻的多条助手消息，规范化时**合并为一条**带多个 `tool_calls` 的消息，因为真实模型会在同一次生成里输出它们。
- **nebius SWE-agent**：`trajectory` 里的角色是 `system` / `user` / `ai`。动作写在文本里，工具观察以 `user` 角色出现，**没有结构化工具调用**。轨迹由 SWE-agent-LLaMA 8B / 70B / 405B 生成。`instance_id` 会在不同尝试之间重复，所以样本 id 用 `分片行号:instance_id`。
- 每条轨迹最多使用前 50 条助手消息（`--max-turns 50`）。

三个数据集都**没有独立的思考内容字段**。Qwen3 的 `think_keep` / `think_drop` 配置统一注入一段固定的合成思考文本（见 `run.py` 中的 `SYNTHETIC_REASONING`），配置名带 `[synthetic]` 标记。

## Tokenizer

| 仓库 | revision | 许可证 | 配置 |
|---|---|---|---|
| [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | `a09a35458c702b33eeacc393d103063234e8bc28` | Apache-2.0 | `qwen2.5` |
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `b968826d9c46dd6066d109eabc6255188de91218` | Apache-2.0 | `qwen3_no_think`（`enable_thinking=False`）、`qwen3_think_keep[synthetic]`、`qwen3_think_drop[synthetic]` |

运行环境：Python 3.12.13、transformers 5.20.0、tokenizers 0.23.3，没有安装 PyTorch。

## 复现

```bash
pip install -e ".[real]"
python -m experiments.template_prefix.run   # 首次运行会下载上述分片和 tokenizer 文件
```
