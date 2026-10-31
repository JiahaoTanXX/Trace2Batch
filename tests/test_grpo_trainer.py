# Copyright (c) 2026 Jiahao Tan

"""P3-B 训练端的 CPU 冒烟测试: 极小的随机 Qwen3 + peft LoRA, 不需要 GPU 或下载权重。"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from experiments.grpo.grpo import WeightedRow, rollout_rows  # noqa: E402
from experiments.grpo.run import TorchTrainer, wrap_lora  # noqa: E402

EVENTS = [
    {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2, 3], "response_token_ids": [4, 5]}},
    {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2, 3, 4, 5, 9, 10], "response_token_ids": [6]}},
]


def tiny_trainer(lr: float = 1e-2) -> TorchTrainer:
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
    )  # fmt: skip
    model = wrap_lora(transformers.Qwen3ForCausalLM(config), rank=4, device="cpu")
    return TorchTrainer(model, lr=lr, device="cpu")


def test_lora_wrapper_trains_only_adapter_weights_in_train_mode() -> None:
    trainer = tiny_trainer()
    trainable = [name for name, p in trainer.model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)
    assert trainer.model.training and trainer.model.config.use_cache is False


def test_update_raises_the_logprob_of_a_positively_rewarded_rollout() -> None:
    trainer = tiny_trainer()
    rows = [WeightedRow(row, 1.0) for row in rollout_rows(EVENTS, "trajectory")]
    # LoRA B 初始化为 0: 第一步之后 adapter 才偏离基础模型; 连续更新几次再比较。
    before = sum(trainer.logprobs(rows))
    metrics: dict = {}
    for _ in range(5):
        metrics = trainer.update(rows)
    assert metrics["optimizer_step"] is True and metrics["policy_tokens"] == 3
    assert sum(trainer.logprobs(rows)) > before


def test_compare_reports_identical_gradients_for_both_representations() -> None:
    trainer = tiny_trainer()
    trainer.update([WeightedRow(row, 1.0) for row in rollout_rows(EVENTS, "trajectory")])  # 让 LoRA B 不为 0
    result = trainer.compare(
        [WeightedRow(row, 0.5) for row in rollout_rows(EVENTS, "trajectory")],
        [WeightedRow(row, 0.5) for row in rollout_rows(EVENTS, "transition")],
    )
    assert result["trajectory_rows"] == 1 and result["transition_rows"] == 2
    assert result["relative_l2"] < 1e-5 and result["cosine"] > 1 - 1e-6


def test_save_writes_a_peft_adapter(tmp_path: Path) -> None:
    trainer = tiny_trainer()
    trainer.save(tmp_path / "step1")
    files = {p.name for p in (tmp_path / "step1").iterdir()}
    assert {"adapter_config.json", "adapter_model.safetensors"} <= files
