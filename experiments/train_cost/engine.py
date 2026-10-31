# Copyright (c) 2026 Jiahao Tan
# pyright: reportMissingImports=false

"""Optional masked-CE training measurements, with one update per complete rollout.

Torch is imported only when an entry point is called. A benchmark restores its
caller's trainable parameters, gradients and module training modes on every exit.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Any

from trace2batch.core import SampleRow

_MODES = ("transition", "trajectory")
# Pass criterion for the transition/trajectory gradient check on a T4 in fp16. The gradient thresholds sit above
# the fp16 noise measured with loss scaling (up to relative L2 0.024 at ~5k tokens, while fp32 agrees to 4e-6);
# the loss threshold still rejects a single changed context token (fp32 loss error 3e-3). Exact equivalence is
# checked separately in fp32 on CPU before submission.
GRADIENT_CRITERION = {"max_loss_relative_error": 1e-3, "max_relative_l2_error": 0.05, "min_cosine": 0.999}


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Training requires the optional torch dependency.") from error
    return torch


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Training deadline exceeded.")


def _device(torch: Any, device: Any) -> Any:
    result = torch.device(device)
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("Training-cost measurements support only cpu or cuda.")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return result


def _policy_count(row: SampleRow) -> int:
    if row.is_drop or not row.prompt_token_ids or not row.response_token_ids:
        raise ValueError("Training rows need a nonempty prompt and response and cannot be marked drop.")
    if len(row.response_token_ids) != len(row.response_mask) or any(mask not in (0, 1) for mask in row.response_mask):
        raise ValueError("response_mask must contain one binary value for each response token.")
    if any(type(token) is not int or token < 0 for token in row.prompt_token_ids + row.response_token_ids):
        raise ValueError("Token IDs must be nonnegative integers.")
    count = sum(row.response_mask)
    if count <= 0:
        raise ValueError("Training rows need supervised policy tokens.")
    return count


def policy_loss(model: Any, row: SampleRow, *, device: Any, policy_logits_only: bool = False) -> Any:
    """Return CE sum over policy response targets using their causal predecessors.

    The full prompt and observations remain model context. Their target labels are
    ignored, while response policy tokens (including the first) are shifted by one.
    The caller owns autocast and backward so rows can be accumulated without padding.

    `policy_logits_only` passes the supervised positions as `logits_to_keep`, so the LM
    head runs only there instead of over the whole row (about 4.6 GiB of fp16 logits at
    16k tokens). The loss and gradients are the same; only the cost changes.
    """
    torch = _torch()
    _policy_count(row)
    tokens = row.prompt_token_ids + row.response_token_ids
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    labels = torch.tensor(
        [
            [-100] * len(row.prompt_token_ids)
            + [token if mask else -100 for token, mask in zip(row.response_token_ids, row.response_mask, strict=True)]
        ],
        dtype=torch.long,
        device=device,
    )
    # Gather before converting: prompt/observation vocab logits need no fp32 copy.
    # Explicit shifting is independent of each model family's built-in label handling.
    shifted_labels = labels[:, 1:]
    supervised = shifted_labels != -100
    if policy_logits_only:
        # Position i predicts token i + 1, so the kept logits are exactly the supervised predecessors.
        positions = supervised[0].nonzero().flatten()
        logits = model(input_ids=input_ids, logits_to_keep=positions).logits[0]
        return torch.nn.functional.cross_entropy(logits.float(), shifted_labels[supervised], reduction="sum")
    logits = model(input_ids=input_ids).logits
    return torch.nn.functional.cross_entropy(
        logits[:, :-1, :][supervised].float(), shifted_labels[supervised], reduction="sum"
    )


def _parameters(model: Any) -> list[Any]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The model has no trainable parameters.")
    return parameters


def _snapshot(model: Any, parameters: list[Any]) -> tuple[list[Any], list[Any], list[tuple[Any, bool]]]:
    return (
        [parameter.detach().cpu().clone() for parameter in parameters],
        [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters],
        [(module, module.training) for module in model.modules()],
    )


def _restore_parameters(torch: Any, parameters: list[Any], values: list[Any]) -> None:
    with torch.no_grad():
        for parameter, value in zip(parameters, values, strict=True):
            parameter.copy_(value)
        for parameter in parameters:
            parameter.grad = None


def _restore(
    torch: Any,
    parameters: list[Any],
    snapshot: tuple[list[Any], list[Any], list[tuple[Any, bool]]],
) -> None:
    values, gradients, modes = snapshot
    _restore_parameters(torch, parameters, values)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient
    for module, training in modes:
        module.training = training


def _finite(torch: Any, value: Any, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"Nonfinite {name}; no valid training measurement is available.")


def _finite_gradients(torch: Any, parameters: list[Any]) -> None:
    for parameter in parameters:
        if parameter.grad is not None:
            _finite(torch, parameter.grad, "gradient")


def _require_no_dropout(torch: Any, model: Any) -> None:
    for module in model.modules():
        rates = [module.p] if isinstance(module, torch.nn.modules.dropout._DropoutNd) else []
        rates.append(getattr(module, "attention_dropout", 0.0))
        if any(isinstance(rate, (int, float)) and rate > 0 for rate in rates):
            raise ValueError("Gradient verification runs in training mode and requires all dropout to be disabled.")


def _pair_tokens(pair: Any) -> int:
    counts = [sum(_policy_count(row) for row in getattr(pair, mode)) for mode in _MODES]
    if not all(getattr(pair, mode) for mode in _MODES) or counts[0] != counts[1] or counts[0] != pair.policy_tokens:
        raise ValueError("Paired representations must contain equal nonzero policy token counts.")
    return counts[0]


def _autocast(torch: Any, device: Any) -> Any:
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def verify_equivalence(
    model: Any,
    pair: Any,
    *,
    device: Any,
    rtol: float,
    atol: float,
    deadline: float | None = None,
    loss_scale: float = 1.0,
    policy_logits_only: bool = False,
) -> dict[str, Any]:
    """Compare normalized CE and every trainable gradient without an optimizer step.

    Verification uses training mode, like the measured updates, so gradient
    checkpointing stays active (Hugging Face skips it in eval mode, which kept every
    layer's activations and ran out of memory on a 16 GB T4). Reproducibility comes
    from requiring all dropout to be disabled instead. CUDA fp16 autocast matches the
    measured forward precision. Losses and stored gradients are float32.

    `loss_scale` multiplies each backward like a static GradScaler (fp16 gradients
    below the half-precision range otherwise flush to zero); stored gradients are
    divided by it again, so reported losses and gradients are unscaled.
    """
    torch = _torch()
    device = _device(torch, device)
    if not math.isfinite(rtol) or not math.isfinite(atol) or min(rtol, atol) < 0:
        raise ValueError("Equivalence tolerances must be finite and nonnegative.")
    if not isinstance(loss_scale, int | float) or not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError("loss_scale must be a finite positive number.")
    policy_tokens = _pair_tokens(pair)
    parameters = _parameters(model)
    _require_no_dropout(torch, model)
    snapshot = _snapshot(model, parameters)
    losses: dict[str, float] = {}
    gradients: dict[str, list[Any]] = {}
    seconds: dict[str, float] = {}
    try:
        model.train()
        for mode in _MODES:
            _check_deadline(deadline)
            for parameter in parameters:
                parameter.grad = None
            total_loss = 0.0
            _sync(torch, device)
            started = time.monotonic()
            for row in getattr(pair, mode):
                _check_deadline(deadline)
                with _autocast(torch, device):
                    loss = policy_loss(model, row, device=device, policy_logits_only=policy_logits_only) / policy_tokens
                _finite(torch, loss, "loss")
                _check_deadline(deadline)
                (loss * loss_scale).backward()
                _check_deadline(deadline)
                total_loss += float(loss.detach().item())
            _sync(torch, device)
            # Forward + backward only (no optimizer step); a cost probe, not the benchmark's timing.
            seconds[mode] = time.monotonic() - started
            _finite_gradients(torch, parameters)
            losses[mode] = total_loss
            gradients[mode] = [
                None if parameter.grad is None else parameter.grad.detach().float().cpu().div(loss_scale)
                for parameter in parameters
            ]
        max_error = 0.0
        presence_mismatches = 0
        elementwise_close = True
        for left, right in zip(gradients["transition"], gradients["trajectory"], strict=True):
            if left is None or right is None:
                elementwise_close = elementwise_close and left is None and right is None
                if (left is None) != (right is None):
                    presence_mismatches += 1
            else:
                max_error = max(max_error, float((left.double() - right.double()).abs().max().item()))
                elementwise_close = elementwise_close and bool(torch.allclose(left, right, rtol=rtol, atol=atol))
        metrics = _gradient_metrics(torch, gradients["transition"], gradients["trajectory"], rtol, atol)
        loss_error = abs(losses["transition"] - losses["trajectory"])
        verdict = judge_gradients(
            loss_relative_error=loss_error / max(abs(losses["trajectory"]), 1e-12),
            relative_l2_error=metrics.get("relative_l2_error"),
            cosine=metrics.get("cosine"),
            presence_mismatches=presence_mismatches,
        )
        _check_deadline(deadline)
        return {
            "sample_id": pair.sample_id,
            "dataset": pair.dataset,
            "policy_tokens": policy_tokens,
            "transition_loss": losses["transition"],
            "trajectory_loss": losses["trajectory"],
            "loss_abs_error": loss_error,
            "max_gradient_abs_error": None if presence_mismatches else max_error,
            "gradient_presence_mismatches": presence_mismatches,
            **verdict,
            # Diagnostic only: element-wise allclose is too brittle for over a million fp16 gradients.
            "elementwise_close": elementwise_close,
            "rtol": rtol,
            "atol": atol,
            "loss_scale": float(loss_scale),
            "gradient_metrics": metrics,
            "seconds": seconds,
            "policy_logits_only": policy_logits_only,
        }
    finally:
        _restore(torch, parameters, snapshot)


def judge_gradients(
    *,
    loss_relative_error: float,
    relative_l2_error: float | None,
    cosine: float | None,
    presence_mismatches: int,
) -> dict[str, Any]:
    """Apply GRADIENT_CRITERION; missing metrics or mismatched gradient presence never pass."""
    loss_close = loss_relative_error <= GRADIENT_CRITERION["max_loss_relative_error"]
    gradients_close = (
        presence_mismatches == 0
        and relative_l2_error is not None
        and cosine is not None
        and relative_l2_error <= GRADIENT_CRITERION["max_relative_l2_error"]
        and cosine >= GRADIENT_CRITERION["min_cosine"]
    )
    return {
        "loss_close": loss_close,
        "gradients_close": gradients_close,
        "passed": loss_close and gradients_close,
        "criterion": dict(GRADIENT_CRITERION),
    }


def _gradient_metrics(torch: Any, left: list[Any], right: list[Any], rtol: float, atol: float) -> dict[str, Any]:
    """Scale-aware comparison over all trainable gradients (missing gradients count as zeros)."""
    pairs = [(a, b) for a, b in zip(left, right, strict=True) if a is not None or b is not None]
    if not pairs:
        return {"elements": 0}
    first = torch.cat([(a if a is not None else torch.zeros_like(b)).double().flatten() for a, b in pairs])
    second = torch.cat([(b if b is not None else torch.zeros_like(a)).double().flatten() for a, b in pairs])
    difference = first - second
    left_norm, right_norm = float(first.norm()), float(second.norm())
    denominator = left_norm * right_norm
    return {
        "elements": int(first.numel()),
        "transition_grad_norm": left_norm,
        "trajectory_grad_norm": right_norm,
        "difference_norm": float(difference.norm()),
        "relative_l2_error": float(difference.norm()) / right_norm if right_norm > 0 else None,
        "cosine": float(first.dot(second)) / denominator if denominator > 0 else None,
        "elements_outside_tolerance": int((difference.abs() > atol + rtol * second.abs()).sum()),
        "transition_zero_fraction": float((first == 0).double().mean()),
        "trajectory_zero_fraction": float((second == 0).double().mean()),
    }


def gradient_diagnostics(
    model: Any,
    pair: Any,
    *,
    device: Any,
    loss_scales: Sequence[float],
    rtol: float,
    atol: float,
    deadline: float | None = None,
    policy_logits_only: bool = False,
) -> dict[str, Any]:
    """Run the gradient check once per loss scale; overflow is recorded, never retried or raised."""
    results: list[dict[str, Any]] = []
    for scale in loss_scales:
        try:
            result = verify_equivalence(
                model,
                pair,
                device=device,
                rtol=rtol,
                atol=atol,
                deadline=deadline,
                loss_scale=scale,
                policy_logits_only=policy_logits_only,
            )
            results.append({"status": "ok", **result})
        except FloatingPointError as error:
            results.append(
                {"status": "nonfinite", "loss_scale": float(scale), "error": f"{type(error).__name__}: {error}"}
            )
    return {"sample_id": pair.sample_id, "dataset": pair.dataset, "group": pair.group, "results": results}


def _sync(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mark(torch: Any, device: Any) -> Any:
    """A per-rollout timestamp that never forces a host synchronization."""
    if device.type != "cuda":
        return time.monotonic()
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _between(device: Any, begin: Any, end: Any) -> float:
    # CUDA events are read only after the condition's final synchronization.
    return end - begin if device.type != "cuda" else begin.elapsed_time(end) / 1000


def _train_rollout(
    torch: Any,
    model: Any,
    pair: Any,
    mode: str,
    *,
    device: Any,
    parameters: list[Any],
    optimizer: Any,
    scaler: Any,
    deadline: float | None,
    record: dict[str, Any],
    policy_logits_only: bool = False,
) -> None:
    _check_deadline(deadline)
    optimizer.zero_grad(set_to_none=True)
    policy_tokens = pair.policy_tokens
    for row in getattr(pair, mode):
        _check_deadline(deadline)
        with _autocast(torch, device):
            loss = policy_loss(model, row, device=device, policy_logits_only=policy_logits_only) / policy_tokens
        record["processed_tokens"] += len(row.prompt_token_ids) + len(row.response_token_ids)
        record["rows"] += 1
        _finite(torch, loss, "loss")
        _check_deadline(deadline)
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
        record["policy_tokens"] += sum(row.response_mask)
        _check_deadline(deadline)
    if scaler is not None:
        scaler.unscale_(optimizer)
    _finite_gradients(torch, parameters)
    _check_deadline(deadline)
    if scaler is None:
        optimizer.step()
    else:
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            raise FloatingPointError("Nonfinite gradients caused fp16 optimizer overflow.")
    record["optimizer_steps"] += 1
    record["rollouts"] += 1
    _finite_gradients(torch, parameters)
    for parameter in parameters:
        _finite(torch, parameter, "parameter after optimizer step")
    _check_deadline(deadline)


def _counters() -> dict[str, int]:
    return {"policy_tokens": 0, "processed_tokens": 0, "rows": 0, "rollouts": 0, "optimizer_steps": 0}


def _failure_status(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "deadline_exceeded"
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    if isinstance(error, FloatingPointError):
        return "nonfinite"
    if isinstance(error, RuntimeError) and "out of memory" in str(error).lower():
        return "oom"
    return "error"


def benchmark(
    model: Any,
    pairs: Sequence[Any],
    *,
    device: Any,
    repeats: int = 2,
    warmup_rollouts: int = 1,
    learning_rate: float = 1e-4,
    deadline: float | None = None,
    on_record: Callable[[dict[str, Any]], None] | None = None,
    policy_logits_only: bool = False,
) -> dict[str, Any]:
    """Measure AB/BA conditions; only fully completed pairs yield a speed ratio.

    Each condition has its own warmup, then resets trainable parameters and creates
    a fresh AdamW and GradScaler before timing. The ratio is transition seconds /
    trajectory seconds over all repeats. Partial records count successful forward
    input tokens and backward policy tokens, even if that rollout did not update.
    """
    torch = _torch()
    device = _device(torch, device)
    if type(repeats) is not int or repeats < 1 or type(warmup_rollouts) is not int or warmup_rollouts < 0:
        raise ValueError("repeats must be positive and warmup_rollouts nonnegative integers.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    pairs = list(pairs)
    if not pairs:
        raise ValueError("At least one paired rollout is required.")
    for pair in pairs:
        _pair_tokens(pair)
    parameters = _parameters(model)
    snapshot = _snapshot(model, parameters)
    records: list[dict[str, Any]] = []
    status = "complete"

    def fresh_optimizer() -> tuple[Any, Any]:
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        return optimizer, scaler

    try:
        model.train()
        for repeat in range(repeats):
            order = _MODES if repeat % 2 == 0 else _MODES[::-1]
            for mode in order:
                record: dict[str, Any] = {
                    "repeat": repeat,
                    "mode": mode,
                    "complete": False,
                    "status": "running",
                    "phase": "warmup",
                    "seconds": 0.0,
                    "peak_allocated_bytes": None,
                    "peak_reserved_bytes": None,
                    **_counters(),
                }
                started: float | None = None
                try:
                    _check_deadline(deadline)
                    _restore_parameters(torch, parameters, snapshot[0])
                    optimizer, scaler = fresh_optimizer()
                    warmup_record: dict[str, Any] = _counters()
                    for pair in pairs[:warmup_rollouts]:
                        _train_rollout(
                            torch,
                            model,
                            pair,
                            mode,
                            device=device,
                            parameters=parameters,
                            optimizer=optimizer,
                            scaler=scaler,
                            deadline=deadline,
                            record=warmup_record,
                            policy_logits_only=policy_logits_only,
                        )
                    _sync(torch, device)
                    _check_deadline(deadline)
                    _restore_parameters(torch, parameters, snapshot[0])
                    # Destroy warmup optimizer state before allocating the measured one.
                    del optimizer, scaler
                    optimizer, scaler = fresh_optimizer()
                    _sync(torch, device)
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    _check_deadline(deadline)
                    record["phase"] = "measurement"
                    marks: list[tuple[Any, Any]] = []
                    started = time.monotonic()
                    for pair in pairs:
                        begin = _mark(torch, device)
                        _train_rollout(
                            torch,
                            model,
                            pair,
                            mode,
                            device=device,
                            parameters=parameters,
                            optimizer=optimizer,
                            scaler=scaler,
                            deadline=deadline,
                            record=record,
                            policy_logits_only=policy_logits_only,
                        )
                        marks.append((begin, _mark(torch, device)))
                    _sync(torch, device)
                    _check_deadline(deadline)
                    # Diagnostic per-rollout split; the condition total below stays the wall-clock measure.
                    record["rollout_ids"] = [pair.sample_id for pair in pairs]
                    record["rollout_seconds"] = [_between(device, begin, end) for begin, end in marks]
                    record["complete"] = True
                    record["status"] = "complete"
                except (Exception, KeyboardInterrupt) as error:
                    record["status"] = _failure_status(error)
                    record["error"] = f"{type(error).__name__}: {error}"
                    status = record["status"]
                finally:
                    # CUDA completion, rather than submission time, bounds elapsed time.
                    try:
                        _sync(torch, device)
                    except (Exception, KeyboardInterrupt) as error:
                        record["complete"] = False
                        record["status"] = status = _failure_status(error)
                        record["error"] = f"{type(error).__name__}: {error}"
                    if started is not None:
                        record["seconds"] = time.monotonic() - started
                    if device.type == "cuda":
                        record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                        record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                    seconds = record["seconds"]
                    record["policy_tokens_per_second"] = record["policy_tokens"] / seconds if seconds > 0 else None
                    record["processed_tokens_per_second"] = (
                        record["processed_tokens"] / seconds if seconds > 0 else None
                    )
                records.append(record)
                if on_record is not None:
                    on_record(dict(record))
                if not record["complete"]:
                    break
            if status != "complete":
                break
        complete = len(records) == repeats * 2 and all(record["complete"] for record in records)
        totals = {mode: sum(record["seconds"] for record in records if record["mode"] == mode) for mode in _MODES}
        return {
            "records": records,
            "complete": complete,
            "status": status,
            "speed_ratio": totals["transition"] / totals["trajectory"]
            if complete and totals["trajectory"] > 0
            else None,
            "speed_ratio_definition": "transition_seconds / trajectory_seconds",
            "seconds_by_mode": totals,
            "repeats": repeats,
            "rollouts_per_condition": len(pairs),
            "policy_tokens_per_condition": sum(pair.policy_tokens for pair in pairs),
        }
    finally:
        _restore(torch, parameters, snapshot)
