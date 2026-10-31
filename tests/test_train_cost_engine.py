"""CPU numerical checks for the optional training-cost engine; never download models."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from experiments.train_cost.data import PairedRollout
from trace2batch.core import SampleRow

# Optional even in the default suite: core users do not need the training extra.
torch = pytest.importorskip("torch")


class TinyCausalLM(torch.nn.Module):
    """A real causal language model with a trainable prefix-dependent objective."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.embedding = torch.nn.Embedding(13, 5)
        self.projection = torch.nn.Linear(5, 13)
        self.register_buffer("prefix_scale", torch.tensor(0.3))

    def forward(self, input_ids):
        hidden = self.embedding(input_ids).cumsum(dim=1) * self.prefix_scale
        return SimpleNamespace(logits=self.projection(hidden))


def row(prompt, response, mask):
    return SampleRow(prompt, response, mask, [0], 0.0, False)


def paired(final_token=5):
    # All three policy targets see exactly the same complete prefix in both modes.
    transition = [row([1, 2], [3, 4], [1, 1]), row([1, 2, 3, 4, 8, 9], [final_token], [1])]
    trajectory = [row([1, 2], [3, 4, 8, 9, final_token], [1, 1, 0, 0, 1])]
    return PairedRollout(sample_id="tiny", dataset="fixture", transition=transition, trajectory=trajectory)


def engine():
    from experiments.train_cost import engine as implementation

    return implementation


def test_policy_loss_shifts_targets_and_masks_prompt_and_observation():
    model = TinyCausalLM()
    item = paired().trajectory[0]
    logits = model(torch.tensor([[1, 2, 3, 4, 8, 9, 5]])).logits
    # Hand-picked causal predecessors: first policy follows prompt, last follows observation.
    expected = torch.nn.functional.cross_entropy(logits[0, [1, 2, 5]], torch.tensor([3, 4, 5]), reduction="sum")
    actual = engine().policy_loss(model, item, device="cpu")
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert model.embedding.weight.grad is not None
    assert torch.isfinite(model.embedding.weight.grad).all()


def test_verify_compares_normalized_losses_and_trainable_gradients_without_side_effects():
    model = TinyCausalLM().train()
    model.embedding.weight.requires_grad_(False)
    model.projection.bias.grad = torch.full_like(model.projection.bias, 0.25)
    before = copy.deepcopy(model.state_dict())
    original_grad = model.projection.bias.grad.clone()
    reference_logits = model(torch.tensor([[1, 2, 3, 4, 8, 9, 5]])).logits
    reference_loss = (
        torch.nn.functional.cross_entropy(reference_logits[0, [1, 2, 5]], torch.tensor([3, 4, 5]), reduction="sum") / 3
    )
    result = engine().verify_equivalence(model, paired(), device="cpu", rtol=1e-5, atol=1e-6)
    assert result["passed"]
    assert result["policy_tokens"] == 3
    assert result["transition_loss"] == pytest.approx(result["trajectory_loss"], rel=1e-5)
    assert result["transition_loss"] == pytest.approx(reference_loss.item(), rel=1e-5)
    assert result["max_gradient_abs_error"] < 1e-6
    assert model.training
    torch.testing.assert_close(model.projection.bias.grad, original_grad)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_verify_detects_same_count_with_wrong_target_or_context():
    pair = paired()
    pair = replace(pair, trajectory=[row([1, 7], [3, 4, 8, 9, 5], [1, 1, 0, 0, 1])])
    result = engine().verify_equivalence(TinyCausalLM(), pair, device="cpu", rtol=1e-7, atol=1e-8)
    assert not result["passed"]
    assert result["max_gradient_abs_error"] > 0


def test_benchmark_one_update_per_rollout_and_ab_ba_start_from_identical_parameters():
    model = TinyCausalLM()
    initial = copy.deepcopy(model.state_dict())
    records = []
    starts = []
    final_parameters = []
    original_forward = model.forward

    reference = TinyCausalLM()
    optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4)
    # Independent direct CE reference: targets and causal positions are hand selected.
    for final_token in [5, 6]:
        optimizer.zero_grad(set_to_none=True)
        logits = reference(torch.tensor([[1, 2, 3, 4, 8, 9, final_token]])).logits
        loss = (
            torch.nn.functional.cross_entropy(logits[0, [1, 2, 5]], torch.tensor([3, 4, final_token]), reduction="sum")
            / 3
        )
        loss.backward()
        optimizer.step()

    def capture_record(record):
        records.append(record)
        final_parameters.append(copy.deepcopy(model.state_dict()))

    def tracked_forward(input_ids):
        starts.append(model.projection.weight.detach().clone())
        return original_forward(input_ids)

    model.forward = tracked_forward
    result = engine().benchmark(
        model, [paired(), paired(6)], device="cpu", repeats=2, warmup_rollouts=1, on_record=capture_record
    )
    assert result["complete"]
    assert result["status"] == "complete"
    assert result["speed_ratio"] > 0
    assert [(record["repeat"], record["mode"]) for record in records] == [
        (0, "transition"),
        (0, "trajectory"),
        (1, "trajectory"),
        (1, "transition"),
    ]
    assert [record["optimizer_steps"] for record in records] == [2, 2, 2, 2]
    assert [record["policy_tokens"] for record in records] == [6, 6, 6, 6]
    assert [record["processed_tokens"] for record in records] == [22, 14, 14, 22]
    assert all(record["rollouts"] == 2 and record["complete"] for record in records)
    # A fresh first rollout is seen at every warmup and measured condition start.
    for offset in [0, 2, 6, 7, 9, 10, 12, 14]:
        torch.testing.assert_close(starts[offset], initial["projection.weight"])
    for measured in final_parameters:
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(measured[name], value)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name])


def test_benchmark_partial_deadline_record_never_yields_a_speed_ratio():
    model = TinyCausalLM()
    initial = copy.deepcopy(model.state_dict())
    records = []
    result = engine().benchmark(
        model,
        [paired()],
        device="cpu",
        repeats=2,
        warmup_rollouts=0,
        deadline=time.monotonic() - 1,
        on_record=records.append,
    )
    assert not result["complete"]
    assert result["status"] == "deadline_exceeded"
    assert result["speed_ratio"] is None
    assert len(records) == 1
    assert not records[0]["complete"]
    assert records[0]["optimizer_steps"] == 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name])


def test_verify_rejects_nonfinite_objective_and_restores_existing_gradients():
    model = TinyCausalLM()
    with torch.no_grad():
        model.projection.weight[0, 0] = math.nan
    model.projection.bias.grad = torch.ones_like(model.projection.bias)
    with pytest.raises(FloatingPointError, match="finite"):
        engine().verify_equivalence(model, paired(), device="cpu", rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(model.projection.bias.grad, torch.ones_like(model.projection.bias))


def test_benchmark_nonfinite_loss_is_incomplete_and_does_not_update():
    model = TinyCausalLM()
    original_forward = model.forward

    def nonfinite_forward(input_ids):
        result = original_forward(input_ids)
        return SimpleNamespace(logits=result.logits * math.nan)

    model.forward = nonfinite_forward
    initial = copy.deepcopy(model.state_dict())
    records = []
    result = engine().benchmark(model, [paired()], device="cpu", repeats=2, warmup_rollouts=0, on_record=records.append)
    assert not result["complete"]
    assert result["status"] == "nonfinite"
    assert result["speed_ratio"] is None
    assert records[0]["optimizer_steps"] == 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name])


def test_policy_loss_only_converts_supervised_logits_to_full_precision():
    model = TinyCausalLM().half()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True, record_shapes=True
    ) as profile:
        loss = engine().policy_loss(model, paired().trajectory[0], device="cpu")
    assert loss.dtype == torch.float32
    # Only three supervised positions need fp32 vocabulary logits. Casting all six
    # predecessors would allocate twice as much, and real prompt contexts are longer.
    cast_allocations = [event.cpu_memory_usage for event in profile.events() if event.name == "aten::_to_copy"]
    assert cast_allocations
    assert max(cast_allocations) <= 3 * 13 * 4


def test_deadline_between_conditions_keeps_only_one_complete_side(monkeypatch):
    implementation = engine()
    clock = [10.0]
    monkeypatch.setattr(implementation.time, "monotonic", lambda: clock[0])
    records = []

    def after_first(record):
        records.append(record)
        clock[0] = 12.0

    result = implementation.benchmark(
        TinyCausalLM(),
        [paired()],
        device="cpu",
        repeats=2,
        warmup_rollouts=0,
        deadline=11.0,
        on_record=after_first,
    )
    assert [record["complete"] for record in records] == [True, False]
    assert records[0]["optimizer_steps"] == 1
    assert records[1]["optimizer_steps"] == 0
    assert result["speed_ratio"] is None
    assert result["status"] == "deadline_exceeded"


@pytest.mark.parametrize(
    "failure,status", [(KeyboardInterrupt(), "interrupted"), (RuntimeError("out of memory"), "oom")]
)
def test_interrupt_and_oom_preserve_partial_record_and_restore_parameters(failure, status):
    model = TinyCausalLM().eval()
    original_forward = model.forward
    initial = copy.deepcopy(model.state_dict())
    calls = 0

    def interrupted_forward(input_ids):
        nonlocal calls
        calls += 1
        # First transition row completes backward; the second aborts the rollout.
        if calls == 2:
            raise failure
        return original_forward(input_ids)

    model.forward = interrupted_forward
    records = []
    result = engine().benchmark(model, [paired()], device="cpu", warmup_rollouts=0, on_record=records.append)
    assert not result["complete"]
    assert result["status"] == status
    assert result["speed_ratio"] is None
    assert records[0]["processed_tokens"] == 4
    assert records[0]["policy_tokens"] == 2
    assert records[0]["optimizer_steps"] == 0
    assert not model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name])


def test_nonfinite_gradient_blocks_optimizer_even_with_finite_loss():
    model = TinyCausalLM()
    initial = copy.deepcopy(model.state_dict())
    handle = model.projection.weight.register_hook(lambda gradient: gradient * math.nan)
    try:
        result = engine().benchmark(model, [paired()], device="cpu", warmup_rollouts=0)
    finally:
        handle.remove()
    assert result["status"] == "nonfinite"
    assert result["records"][0]["optimizer_steps"] == 0
    assert result["speed_ratio"] is None
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name])


def test_gradient_presence_mismatch_can_be_saved_in_strict_json():
    class ConditionalParameterLM(TinyCausalLM):
        def __init__(self):
            super().__init__()
            self.extra_bias = torch.nn.Parameter(torch.zeros(13))

        def forward(self, input_ids):
            result = super().forward(input_ids)
            if input_ids.shape[1] == 4:
                return SimpleNamespace(logits=result.logits + self.extra_bias)
            return result

    result = engine().verify_equivalence(ConditionalParameterLM(), paired(), device="cpu", rtol=1e-4, atol=1e-6)
    assert not result["passed"]
    # The runner writes allow_nan=False; mismatch diagnostics must survive that write.
    json.dumps(result, allow_nan=False)


def test_interrupt_during_final_synchronization_still_saves_incomplete_record(monkeypatch):
    implementation = engine()
    original_sync = implementation._sync
    calls = 0

    def interrupt_final_sync(torch_module, device):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise KeyboardInterrupt()
        original_sync(torch_module, device)

    monkeypatch.setattr(implementation, "_sync", interrupt_final_sync)
    records = []
    try:
        result = implementation.benchmark(
            TinyCausalLM(), [paired()], device="cpu", warmup_rollouts=0, on_record=records.append
        )
    except KeyboardInterrupt:
        pytest.fail("final synchronization interruption escaped before on_record saved partial results")
    assert not result["complete"]
    assert result["status"] == "interrupted"
    assert len(records) == 1
    assert not records[0]["complete"]
    assert result["speed_ratio"] is None


class ModeRecordingLM(TinyCausalLM):
    def __init__(self):
        super().__init__()
        self.seen_modes: list[bool] = []

    def forward(self, input_ids):
        self.seen_modes.append(self.training)
        return super().forward(input_ids)


def test_verify_runs_in_training_mode_and_restores_caller_mode():
    # Hugging Face only applies gradient checkpointing to modules in training mode.
    model = ModeRecordingLM().eval()
    result = engine().verify_equivalence(model, paired(), device="cpu", rtol=1e-5, atol=1e-6)
    assert result["passed"]
    assert model.seen_modes and all(model.seen_modes)
    assert not model.training


@pytest.mark.parametrize("where", ["module", "attention_attribute"])
def test_verify_rejects_active_dropout(where):
    model = TinyCausalLM().eval()
    if where == "module":
        model.dropout = torch.nn.Dropout(0.1)
    else:
        model.projection.attention_dropout = 0.1
    with pytest.raises(ValueError, match="dropout"):
        engine().verify_equivalence(model, paired(), device="cpu", rtol=1e-5, atol=1e-6)
    assert not model.training


def test_verify_keeps_hf_gradient_checkpointing_active():
    """Version 3 ran out of CUDA memory here: eval mode silently disabled checkpointing in every layer."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
    )
    config.use_cache = False
    model = transformers.Qwen3ForCausalLM(config)
    model.enable_input_require_grads()
    prompt = [1 + index % 60 for index in range(240)]
    pair = PairedRollout(
        sample_id="long",
        dataset="fixture",
        transition=[row(prompt, [3, 4], [1, 1]), row([*prompt, 3, 4, 8, 9], [5], [1])],
        trajectory=[row(prompt, [3, 4, 8, 9, 5], [1, 1, 0, 0, 1])],
    )
    weights = {parameter.untyped_storage().data_ptr() for parameter in model.parameters()}

    def saved_bytes() -> tuple[int, dict]:
        seen: dict[int, int] = {}

        def pack(tensor):
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in weights:
                seen[storage.data_ptr()] = max(seen.get(storage.data_ptr(), 0), storage.nbytes())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            result = engine().verify_equivalence(model, pair, device="cpu", rtol=1e-4, atol=1e-5)
        return sum(seen.values()), result

    model.gradient_checkpointing_disable()
    without, plain = saved_bytes()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.eval()
    with_checkpointing, checked = saved_bytes()
    assert plain["passed"] and checked["passed"]
    assert checked["transition_loss"] == pytest.approx(plain["transition_loss"], rel=1e-5)
    assert with_checkpointing * 3 < without, (with_checkpointing, without)
    assert not model.training


def test_benchmark_records_per_rollout_timing_without_changing_totals():
    pairs = [paired(), replace(paired(6), sample_id="second")]
    result = engine().benchmark(TinyCausalLM(), pairs, device="cpu", repeats=2, warmup_rollouts=1)
    assert result["complete"]
    for record in result["records"]:
        assert record["rollout_ids"] == ["tiny", "second"]
        assert len(record["rollout_seconds"]) == record["rollouts"] == 2
        assert all(seconds >= 0 for seconds in record["rollout_seconds"])
        assert sum(record["rollout_seconds"]) <= record["seconds"] + 1e-6


def test_verify_reports_scale_aware_gradient_metrics():
    same = engine().verify_equivalence(TinyCausalLM(), paired(), device="cpu", rtol=1e-5, atol=1e-6)
    metrics = same["gradient_metrics"]
    assert metrics["transition_grad_norm"] > 0 and metrics["trajectory_grad_norm"] > 0
    assert metrics["relative_l2_error"] < 1e-6
    assert metrics["cosine"] == pytest.approx(1.0, abs=1e-9)
    assert metrics["elements_outside_tolerance"] == 0 and metrics["elements"] > 0
    assert 0 <= metrics["transition_zero_fraction"] <= 1
    wrong = replace(paired(), trajectory=[row([1, 7], [3, 4, 8, 9, 5], [1, 1, 0, 0, 1])])
    differ = engine().verify_equivalence(TinyCausalLM(), wrong, device="cpu", rtol=1e-5, atol=1e-6)["gradient_metrics"]
    assert differ["relative_l2_error"] > 1e-2
    assert differ["cosine"] < 0.9999
    assert differ["elements_outside_tolerance"] > 0


def test_loss_scaling_is_undone_before_gradients_are_compared():
    plain = engine().verify_equivalence(TinyCausalLM(), paired(), device="cpu", rtol=1e-5, atol=1e-6)
    scaled = engine().verify_equivalence(
        TinyCausalLM(), paired(), device="cpu", rtol=1e-5, atol=1e-6, loss_scale=1024.0
    )
    assert scaled["loss_scale"] == 1024.0 and plain["loss_scale"] == 1.0
    assert scaled["transition_loss"] == pytest.approx(plain["transition_loss"], rel=1e-7)
    for key in ("transition_grad_norm", "trajectory_grad_norm"):
        assert scaled["gradient_metrics"][key] == pytest.approx(plain["gradient_metrics"][key], rel=1e-6)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="loss_scale"):
            engine().verify_equivalence(TinyCausalLM(), paired(), device="cpu", rtol=1e-5, atol=1e-6, loss_scale=bad)


def test_gradient_diagnostics_records_overflow_and_restores_the_model():
    model = TinyCausalLM().eval()
    before = copy.deepcopy(model.state_dict())
    report = engine().gradient_diagnostics(model, paired(), device="cpu", loss_scales=(1.0, 1e39), rtol=1e-5, atol=1e-6)
    assert report["sample_id"] == "tiny" and report["group"] == "fixture"
    first, overflow = report["results"]
    assert first["loss_scale"] == 1.0 and first["status"] == "ok" and first["passed"]
    assert overflow["loss_scale"] == 1e39 and overflow["status"] == "nonfinite"
    assert not model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_gradient_criterion_separates_measured_fp16_noise_from_semantic_errors():
    judge = engine().judge_gradients
    # fp16 noise of the size seen on long rows with loss scaling passes, including the worst measured T4 gate
    # (stage-2 run C, 5069 tokens: loss 2.3e-4, relative L2 0.0242, cosine 0.999709; fp32 agrees to 4e-6).
    assert judge(loss_relative_error=5e-4, relative_l2_error=0.015, cosine=0.9999, presence_mismatches=0)["passed"]
    assert judge(loss_relative_error=2.3e-4, relative_l2_error=0.0242, cosine=0.999709, presence_mismatches=0)["passed"]
    # One changed context token (fp32 reference: loss 3.0e-3, relative L2 0.044, cosine 0.99905) now fails only on
    # the loss threshold; the gradient thresholds sit above it and catch only larger errors.
    changed = judge(loss_relative_error=3.0e-3, relative_l2_error=0.044, cosine=0.99905, presence_mismatches=0)
    assert not changed["passed"] and not changed["loss_close"] and changed["gradients_close"]
    # One changed target token fails on every metric.
    target = judge(loss_relative_error=0.11, relative_l2_error=0.156, cosine=0.988, presence_mismatches=0)
    assert not target["passed"] and not target["loss_close"] and not target["gradients_close"]
    # Each threshold rejects on its own.
    assert not judge(loss_relative_error=1e-5, relative_l2_error=0.06, cosine=0.99999, presence_mismatches=0)["passed"]
    assert not judge(loss_relative_error=1e-5, relative_l2_error=0.01, cosine=0.9985, presence_mismatches=0)["passed"]
    assert not judge(loss_relative_error=2e-3, relative_l2_error=0.001, cosine=0.999999, presence_mismatches=0)[
        "passed"
    ]
    assert not judge(loss_relative_error=0.0, relative_l2_error=0.0, cosine=1.0, presence_mismatches=1)["passed"]
    assert not judge(loss_relative_error=0.0, relative_l2_error=None, cosine=None, presence_mismatches=0)["passed"]


def test_verify_reports_criterion_and_keeps_elementwise_check_as_diagnostic():
    result = engine().verify_equivalence(TinyCausalLM(), paired(), device="cpu", rtol=1e-5, atol=1e-6)
    assert result["passed"] and result["gradients_close"]
    assert set(result["criterion"]) == {"max_loss_relative_error", "max_relative_l2_error", "min_cosine"}
    assert "elementwise_close" in result


def _tiny_qwen3(attention="sdpa"):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
    )
    config.use_cache = False
    model = transformers.Qwen3ForCausalLM(config)
    model.config._attn_implementation = attention
    return model


def _long_pair():
    prompt = [1 + index % 60 for index in range(120)]
    return PairedRollout(
        sample_id="long",
        dataset="fixture",
        transition=[row(prompt, [3, 4], [1, 1]), row([*prompt, 3, 4, 8, 9], [5], [1])],
        trajectory=[row(prompt, [3, 4, 8, 9, 5], [1, 1, 0, 0, 1])],
    )


def test_policy_logits_only_gives_the_same_loss_and_gradients():
    """第三阶段: 只在监督位置计算 logits, loss 和梯度必须与全序列 logits 相同。"""
    model = _tiny_qwen3()
    item = _long_pair().trajectory[0]
    results = []
    for selective in (False, True):
        model.zero_grad()
        loss = engine().policy_loss(model, item, device="cpu", policy_logits_only=selective)
        loss.backward()
        results.append((loss.detach(), [parameter.grad.clone() for parameter in model.parameters()]))
    torch.testing.assert_close(results[0][0], results[1][0])
    for full, selected in zip(results[0][1], results[1][1], strict=True):
        torch.testing.assert_close(full, selected)


def test_policy_logits_only_runs_the_lm_head_on_supervised_positions_only():
    model = _tiny_qwen3()
    shapes = []
    handle = model.lm_head.register_forward_hook(lambda module, inputs, output: shapes.append(tuple(output.shape)))
    try:
        engine().policy_loss(model, _long_pair().trajectory[0], device="cpu", policy_logits_only=True)
        engine().policy_loss(model, _long_pair().trajectory[0], device="cpu")
    finally:
        handle.remove()
    # 3 个监督 token 对应 3 个前驱位置; 默认路径对全部 125 个位置计算 logits。
    assert shapes == [(1, 3, 64), (1, 125, 64)]


def test_verify_and_benchmark_pass_the_logits_option_through():
    model = _tiny_qwen3()
    check = engine().verify_equivalence(
        model, _long_pair(), device="cpu", rtol=1e-4, atol=1e-5, policy_logits_only=True
    )
    assert check["passed"] and check["policy_logits_only"]
    assert set(check["seconds"]) == {"transition", "trajectory"}
    result = engine().benchmark(
        model, [_long_pair()], device="cpu", repeats=2, warmup_rollouts=0, policy_logits_only=True
    )
    assert result["complete"]


def test_repeat_kv_attention_matches_sdpa_and_is_used():
    from experiments.train_cost import attention

    name = attention.register()
    reference = _tiny_qwen3()
    candidate = _tiny_qwen3(name)
    candidate.load_state_dict(reference.state_dict())
    masks = []
    original = attention.repeat_kv_attention

    def spy(*args, **kwargs):
        masks.append(args[4])
        return original(*args, **kwargs)

    from transformers import AttentionInterface

    AttentionInterface.register(name, spy)
    try:
        ids = torch.tensor([[1 + index % 60 for index in range(50)]])
        torch.testing.assert_close(candidate(input_ids=ids).logits, reference(input_ids=ids).logits)
        check = engine().verify_equivalence(candidate, _long_pair(), device="cpu", rtol=1e-4, atol=1e-5)
    finally:
        attention.register()
    assert check["passed"]
    # 每层都走自定义路径, 而且单行输入不构造 mask (因果关系由 is_causal 表达)。
    assert masks and all(mask is None for mask in masks)


def test_repeat_kv_attention_rejects_an_explicit_mask():
    from experiments.train_cost import attention

    query = torch.zeros(1, 4, 3, 8)
    key = torch.zeros(1, 2, 3, 8)
    with pytest.raises(ValueError, match="attention mask"):
        attention.repeat_kv_attention(SimpleNamespace(num_key_value_groups=2), query, key, key, torch.zeros(1, 1, 3, 3))
