# Copyright (c) 2026 Jiahao Tan

"""P3-B 训练循环编排的离线测试: rollout、训练器和 adapter 发布都用假实现, 不需要 torch 或 GPU。"""

from __future__ import annotations

from typing import Any

from experiments.grpo.env import make_task
from experiments.grpo.run import (
    choose_difficulties,
    eval_tasks,
    flat_logprob_gaps,
    run_training,
    train_tasks,
)

EVENTS = [
    {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2], "response_token_ids": [3, 4]}},
    {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2, 3, 4, 8, 9], "response_token_ids": [5]}},
]


def test_train_and_eval_tasks_are_deterministic_and_disjoint() -> None:
    first = train_tasks(step=3, count=4, difficulties=[1, 2], seed=0)
    assert first == train_tasks(step=3, count=4, difficulties=[1, 2], seed=0)
    assert first != train_tasks(step=4, count=4, difficulties=[1, 2], seed=0)
    assert {len(t.order) for t in first} <= {1, 2}
    held_out = eval_tasks(count=6, difficulties=[1, 2])
    assert [len(t.order) for t in held_out] == [1, 2, 1, 2, 1, 2]
    train_ids = {t.task_id for step in range(50) for t in train_tasks(step=step, count=4, difficulties=[1, 2], seed=0)}
    assert not train_ids & {t.task_id for t in eval_tasks(count=200, difficulties=[1, 2])}


def test_choose_difficulties_keeps_learnable_levels_or_the_closest_one() -> None:
    assert choose_difficulties({1: 0.9, 2: 0.6, 3: 0.3, 4: 0.05}) == [2, 3]
    # 没有落在 [0.1, 0.7] 的难度时, 取成功率最接近 0.4 的一个。
    assert choose_difficulties({1: 0.95, 2: 0.75, 3: 0.0}) == [2]


def test_flat_logprob_gaps_align_training_and_sampling_logprobs() -> None:
    assert flat_logprob_gaps([[-0.5, -1.0], [-2.0]], [[-0.4], [-1.1, -2.0]]) == [0.1, 0.1, 0.0]
    assert flat_logprob_gaps([[-0.5]], [[-0.5, -1.0]]) is None
    assert flat_logprob_gaps([[-0.5]], [[-0.5], None]) is None


class FakeTrainer:
    def __init__(self) -> None:
        self.updates: list[list[float]] = []
        self.compares = 0

    def update(self, rows) -> dict[str, Any]:
        self.updates.append([w.advantage for w in rows])
        return {"optimizer_step": any(w.advantage for w in rows), "rows": len(rows)}

    def compare(self, trajectory, transition) -> dict[str, Any]:
        self.compares += 1
        return {"trajectory_rows": len(trajectory), "transition_rows": len(transition)}

    def logprobs(self, rows) -> list[float]:
        return [-0.1] * sum(sum(w.row.response_mask) for w in rows)


def test_run_training_publishes_each_step_and_uses_the_new_adapter() -> None:
    used: list[str] = []
    logs: list[tuple[str, dict]] = []
    trainer = FakeTrainer()

    def rollout(task, seed: int, model: str) -> dict[str, Any]:
        used.append(model)
        win = seed % 2 == 0  # 组内一半成功, 一半失败
        return {
            "task_id": task.task_id,
            "reward": float(win),
            "stop_reason": "answered",
            "turns": [{"self_retoken": False, "tool_calls": 1}, {"self_retoken": not win, "tool_calls": 0}],
            "events": EVENTS,
            "sampled_logprobs": [[-0.1, -0.1], [-0.1]],
        }

    published: list[int] = []

    def publish(step: int) -> str:
        published.append(step)
        return f"step{step}"

    evaluations: list[tuple[str, int]] = []

    def evaluate(model: str, step: int) -> dict[str, Any]:
        evaluations.append((model, step))
        return {"success": 0.5}

    run_training(
        steps=3,
        tasks_for_step=lambda step: [make_task(step * 10 + i, 1) for i in range(2)],
        rollout=rollout,
        trainer=trainer,
        publish=publish,
        evaluate=evaluate,
        log=lambda kind, record: logs.append((kind, record)),
        base_model="base",
        group_size=4,
        concurrency=4,
        eval_every=2,
        compare_every=2,
        logprob_rollouts=2,
    )
    # 第 0 步用基础模型, 之后每一步都用上一步发布的 adapter。
    assert used[:8] == ["base"] * 8 and set(used[8:16]) == {"step1"} and set(used[16:]) == {"step2"}
    assert published == [1, 2, 3]
    # 第 0、2 步以及训练结束后评测, 评测用当时的模型。
    assert evaluations == [("base", 0), ("step2", 2), ("step3", 3)]
    assert trainer.compares == 2  # 第 0 和第 2 步
    assert all(abs(sum(a)) < 1e-9 and any(a) for a in trainer.updates)
    steps = [record for kind, record in logs if kind == "step"]
    assert [s["step"] for s in steps] == [0, 1, 2]
    assert steps[0]["reward_mean"] == 0.5 and steps[0]["self_retoken_rate"] == 0.25
    assert steps[0]["logprob_gap"]["n"] == 6 and abs(steps[0]["logprob_gap"]["max"]) < 1e-9
    assert "compare" in steps[0] and "compare" not in steps[1]
    # 按答案类型分别统计奖励, 防止某一类答案的捷径掩盖另一类的变化(pilot 的教训)。
    kinds = {make_task(i, 1).kind for i in range(2)}
    assert set(steps[0]["reward_by_kind"]) == kinds


def test_answer_summary_counts_kinds_and_bare_insufficient_guesses() -> None:
    from experiments.grpo.run import answer_summary

    total = next(make_task(s, 1) for s in range(100) if make_task(s, 1).kind == "total")
    short = next(make_task(s, 1) for s in range(100) if make_task(s, 1).kind == "insufficient")
    results = [
        {"reward": 1.0, "answer": total.answer},
        {"reward": 0.0, "answer": "INSUFFICIENT"},
        {"reward": 1.0, "answer": short.answer},
        {"reward": 0.0, "answer": None},
    ]
    summary = answer_summary([total, total, short, short], results)
    assert summary["success_by_kind"] == {"insufficient": 0.5, "total": 0.5}
    assert summary["tasks_by_kind"] == {"insufficient": 2, "total": 2}
    assert summary["bare_insufficient_answers"] == 1 and summary["no_answer"] == 1


def test_time_budget_stop_still_runs_the_final_evaluation() -> None:
    """pilot2 在第 19 步因时间预算停止, 结果缺少最终评测; 停止时必须用当前模型评测一次。"""
    logs: list[tuple[str, dict]] = []
    clock = {"now": 0.0}

    def rollout(task, seed: int, model: str) -> dict[str, Any]:
        clock["now"] += 1.0
        return {
            "task_id": task.task_id,
            "reward": float(seed % 2 == 0),
            "stop_reason": "answered",
            "turns": [{"self_retoken": False, "tool_calls": 0}],
            "events": EVENTS[:1],
            "sampled_logprobs": [[-0.1, -0.1]],
        }

    import experiments.grpo.run as run_module

    original = run_module.time.monotonic
    run_module.time.monotonic = lambda: clock["now"]
    try:
        run_training(
            steps=10,
            tasks_for_step=lambda step: [make_task(step, 1)],
            rollout=rollout,
            trainer=FakeTrainer(),
            publish=lambda step: f"step{step}",
            evaluate=lambda model, step: {"success": 0.0},
            log=lambda kind, record: logs.append((kind, record)),
            base_model="base",
            group_size=2,
            concurrency=1,
            eval_every=100,
            compare_every=100,
            logprob_rollouts=0,
            deadline=5.0,
        )
    finally:
        run_module.time.monotonic = original
    kinds = [kind for kind, _ in logs]
    assert "stopped" in kinds and kinds[-1] == "eval"
    final = logs[-1][1]
    assert final["model"] == f"step{final['step']}" and final["step"] < 10
