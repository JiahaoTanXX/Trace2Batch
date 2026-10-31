# Copyright (c) 2026 Jiahao Tan

"""P3-B GRPO 更新步骤的 CPU 测试: 用一个极小的因果语言模型, 不需要 GPU 或下载权重。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from experiments.grpo.grpo import (  # noqa: E402
    WeightedRow,
    compare_representations,
    grpo_update,
    policy_token_logprobs,
    rollout_rows,
)
from trace2batch.core import SampleRow  # noqa: E402


class TinyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.embedding = torch.nn.Embedding(13, 5)
        self.projection = torch.nn.Linear(5, 13)

    def forward(self, input_ids):
        hidden = self.embedding(input_ids).cumsum(dim=1) * 0.3
        return SimpleNamespace(logits=self.projection(hidden))


ROW = SampleRow([1, 2], [3, 4, 8, 9, 5], [1, 1, 0, 0, 1], [0, 1], 1.0, False)


def total_logprob(model, row) -> float:
    with torch.no_grad():
        return sum(policy_token_logprobs(model, row, device="cpu"))


def step(advantage: float) -> tuple[float, float, dict]:
    model = TinyCausalLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    before = total_logprob(model, ROW)
    metrics = grpo_update(model, optimizer, None, [WeightedRow(ROW, advantage)], device="cpu")
    return before, total_logprob(model, ROW), metrics


def test_positive_advantage_raises_and_negative_lowers_the_policy_logprob() -> None:
    before, after, metrics = step(1.0)
    assert after > before
    assert metrics["rows"] == 1 and metrics["policy_tokens"] == 3 and metrics["optimizer_step"] is True
    before, after, _ = step(-1.0)
    assert after < before


def test_zero_advantage_batch_skips_the_update() -> None:
    model = TinyCausalLM()
    weights = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    metrics = grpo_update(model, optimizer, None, [WeightedRow(ROW, 0.0)], device="cpu")
    assert metrics["optimizer_step"] is False and metrics["rows"] == 0
    assert all(torch.equal(a, b) for a, b in zip(weights, model.parameters(), strict=True))


def test_policy_token_logprobs_cover_only_policy_tokens_and_match_cross_entropy() -> None:
    model = TinyCausalLM()
    values = policy_token_logprobs(model, ROW, device="cpu")
    assert len(values) == 3 and all(v < 0 for v in values)
    logits = model(torch.tensor([[1, 2, 3, 4, 8, 9, 5]])).logits[0]
    expected = torch.log_softmax(logits, dim=-1)[[1, 2, 5], [3, 4, 5]]
    torch.testing.assert_close(torch.tensor(values), expected.detach())


def test_trajectory_and_transition_rows_give_the_same_gradient() -> None:
    events = [
        {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2], "response_token_ids": [3, 4]}},
        {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2, 3, 4, 8, 9], "response_token_ids": [5]}},
    ]
    trajectory = [WeightedRow(r, 0.5) for r in rollout_rows(events, "trajectory")]
    transition = [WeightedRow(r, 0.5) for r in rollout_rows(events, "transition")]
    model = TinyCausalLM()
    result = compare_representations(model, trajectory, transition, device="cpu")
    assert result["relative_l2"] < 1e-6 and result["cosine"] > 1 - 1e-6
    assert result["trajectory_tokens"] < result["transition_tokens"]
    # 比较之后不留下梯度, 不影响随后的正式更新。
    assert all(p.grad is None for p in model.parameters())


def test_compare_divides_out_the_fixed_loss_scale() -> None:
    """GPU 上 fp16 反向需要固定的 loss scale 防止下溢(pilot 第 5 步因此误报不一致); 缩放必须被完全除回去。"""
    events = [
        {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2], "response_token_ids": [3, 4]}},
        {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2, 3, 4, 8, 9], "response_token_ids": [5]}},
    ]
    trajectory = [WeightedRow(r, 0.5) for r in rollout_rows(events, "trajectory")]
    transition = [WeightedRow(r, 0.5) for r in rollout_rows(events, "transition")]
    plain = compare_representations(TinyCausalLM(), trajectory, transition, device="cpu", loss_scale=1.0)
    scaled = compare_representations(TinyCausalLM(), trajectory, transition, device="cpu", loss_scale=4096.0)
    assert abs(plain["trajectory_grad_norm"] - scaled["trajectory_grad_norm"]) < 1e-9 * max(
        1.0, plain["trajectory_grad_norm"]
    )
    assert scaled["relative_l2"] < 1e-6 and scaled["loss_scale"] == 4096.0
