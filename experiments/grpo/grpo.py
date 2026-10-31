# Copyright (c) 2026 Jiahao Tan
# pyright: reportMissingImports=false

"""P3-B 的 GRPO 核心: 组内优势、训练行、一次 on-policy 更新与诊断。torch 只在调用时导入。

损失是每个 policy token 的 -A·log π, 按整批的 policy token 总数归一化; 工具观察与 prompt 不参与损失。
一条轨迹的所有训练行共享该轨迹的优势。每批只更新一次, 采样与训练使用同一组参数, 因此不需要比率裁剪。
masked CE 复用 P3-A 的 policy_loss(已核对 next-token 对齐与掩码), 优势只是它的权重。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from experiments.train_cost.engine import (
    _autocast,
    _finite,
    _finite_gradients,
    _gradient_metrics,
    _torch,
    policy_loss,
)
from trace2batch.core import SampleRow, build_samples

MAX_GRAD_NORM = 1.0
# 与 P3-A 梯度核对相同的固定 loss scale: GPU 上 fp16 反向时防止小梯度下溢, 比较前除回去。
COMPARE_LOSS_SCALE = 4096.0


@dataclass(frozen=True)
class WeightedRow:
    row: SampleRow
    advantage: float


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """组内奖励减组均值; 不除以标准差, 以免全对或全错之外的小方差组被放大。"""
    mean = sum(rewards) / len(rewards)
    return [reward - mean for reward in rewards]


def rollout_rows(events: list[dict[str, Any]], mode: str) -> list[SampleRow]:
    """一条 rollout 的训练行(不截断); trajectory 在前缀断开处拆行, 与 trace2batch.core 的语义相同。"""
    result = build_samples(events, mode=mode, max_prompt_length=10**9, max_response_length=10**9)
    return [row for row in result.rows if not row.is_drop and sum(row.response_mask) > 0]


def weighted_rows(
    rollouts: Sequence[list[dict[str, Any]]], advantages: Sequence[float], mode: str
) -> list[WeightedRow]:
    return [
        WeightedRow(row, advantage)
        for events, advantage in zip(rollouts, advantages, strict=True)
        for row in rollout_rows(events, mode)
    ]


def _parameters(model: Any) -> list[Any]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _backward(
    torch: Any, model: Any, rows: Sequence[WeightedRow], scaler: Any, *, device: Any, loss_scale: float = 1.0
) -> dict[str, Any]:
    """按优势加权累积梯度; 优势为 0 的行不计算(没有梯度贡献), 但其 policy token 计入归一化分母。

    loss_scale 只在没有 GradScaler 时使用(比较梯度时), 调用方负责把梯度除回去。
    """
    total = sum(sum(w.row.response_mask) for w in rows)
    metrics = {"rows": 0, "processed_tokens": 0, "policy_tokens": 0, "loss": 0.0}
    for weighted in rows:
        if weighted.advantage == 0:
            continue
        with _autocast(torch, device):
            loss = weighted.advantage * policy_loss(model, weighted.row, device=device) / total
        _finite(torch, loss, "loss")
        (loss * loss_scale if scaler is None else scaler.scale(loss)).backward()
        metrics["rows"] += 1
        metrics["processed_tokens"] += len(weighted.row.prompt_token_ids) + len(weighted.row.response_token_ids)
        metrics["policy_tokens"] += sum(weighted.row.response_mask)
        metrics["loss"] += float(loss.detach())
    return metrics


def grpo_update(model: Any, optimizer: Any, scaler: Any, rows: Sequence[WeightedRow], *, device: Any) -> dict[str, Any]:
    """一次 on-policy 更新: 累积整批梯度, 裁剪到 MAX_GRAD_NORM, 只调用一次 optimizer.step()。"""
    torch = _torch()
    device = torch.device(device)
    parameters = _parameters(model)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    metrics: dict[str, Any] = _backward(torch, model, rows, scaler, device=device)
    metrics["optimizer_step"] = False
    if metrics["rows"] == 0:
        return {**metrics, "seconds": round(time.perf_counter() - started, 3)}
    if scaler is not None:
        scaler.unscale_(optimizer)
    _finite_gradients(torch, parameters)
    metrics["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM))
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    metrics["optimizer_step"] = True
    if device.type == "cuda":
        torch.cuda.synchronize()
    metrics["seconds"] = round(time.perf_counter() - started, 3)
    return metrics


def policy_token_logprobs(model: Any, row: SampleRow, *, device: Any) -> list[float]:
    """训练端对一行中每个 policy token 的 log π(按出现顺序), 用于与 vLLM 返回的 logprob 比较。"""
    torch = _torch()
    device = torch.device(device)
    tokens = row.prompt_token_ids + row.response_token_ids
    with torch.no_grad(), _autocast(torch, device):
        logits = model(input_ids=torch.tensor([tokens], dtype=torch.long, device=device)).logits[0].float()
    logprobs = torch.log_softmax(logits[:-1], dim=-1)
    offset = len(row.prompt_token_ids)
    return [
        float(logprobs[offset + index - 1, token])
        for index, (token, mask) in enumerate(zip(row.response_token_ids, row.response_mask, strict=True))
        if mask
    ]


def compare_representations(
    model: Any,
    trajectory: Sequence[WeightedRow],
    transition: Sequence[WeightedRow],
    *,
    device: Any,
    loss_scale: float = COMPARE_LOSS_SCALE,
) -> dict[str, Any]:
    """同一批数据分别按 trajectory 与 transition 累积梯度(不更新), 比较耗时与梯度; 结束时清空梯度。"""
    torch = _torch()
    device = torch.device(device)
    parameters = _parameters(model)
    results: dict[str, Any] = {}
    gradients: dict[str, list[Any]] = {}
    for name, rows in (("trajectory", trajectory), ("transition", transition)):
        for parameter in parameters:
            parameter.grad = None
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        metrics = _backward(torch, model, rows, None, device=device, loss_scale=loss_scale)
        if device.type == "cuda":
            torch.cuda.synchronize()
        results[f"{name}_seconds"] = round(time.perf_counter() - started, 3)
        results[f"{name}_rows"] = metrics["rows"]
        results[f"{name}_tokens"] = metrics["processed_tokens"]
        gradients[name] = [None if p.grad is None else p.grad.detach().float() / loss_scale for p in parameters]
    for parameter in parameters:
        parameter.grad = None
    metrics = _gradient_metrics(torch, gradients["transition"], gradients["trajectory"], rtol=1e-3, atol=1e-6)
    results["loss_scale"] = loss_scale
    results["trajectory_grad_norm"] = metrics.get("trajectory_grad_norm")
    results["relative_l2"] = metrics.get("relative_l2_error")
    results["cosine"] = metrics.get("cosine")
    return results
