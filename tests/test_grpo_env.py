# Copyright (c) 2026 Jiahao Tan

"""P3-B 多轮工具调用环境的离线测试: 任务生成确定、工具执行安全、奖励判定严格。"""

from __future__ import annotations

import json

import pytest

from experiments.grpo.env import (
    TOOLS,
    final_answer,
    initial_messages,
    make_task,
    reward,
    safe_calculate,
    tool_response,
)


def test_make_task_is_deterministic_and_answer_matches_the_inventory() -> None:
    first, second = make_task(7, 3), make_task(7, 3)
    assert first == second
    assert len(first.order) == 3 and set(first.order) <= set(first.inventory)
    assert make_task(8, 3) != first
    for seed in range(50):
        task = make_task(seed, 3)
        short = [item for item, qty in task.order.items() if task.inventory[item][1] < qty]
        if task.kind == "total":
            total = sum(qty * task.inventory[item][0] for item, qty in task.order.items())
            assert not short and abs(float(task.answer) - total) < 1e-9
        else:
            # 缺货答案必须写出缺货物品和它的实际库存: 不查询就无法作答。
            assert len(short) == 1 and task.answer == f"INSUFFICIENT {short[0]} {task.inventory[short[0]][1]}"


def test_tasks_cover_both_answer_kinds() -> None:
    assert {make_task(seed, 2).kind for seed in range(200)} == {"total", "insufficient"}


def test_lookup_returns_price_and_stock_or_an_error() -> None:
    task = make_task(3, 2)
    item = next(iter(task.order))
    result = json.loads(tool_response(task, {"name": "lookup", "arguments": {"item": item}}))
    assert result == {"item": item, "price": task.inventory[item][0], "stock": task.inventory[item][1]}
    assert "error" in json.loads(tool_response(task, {"name": "lookup", "arguments": {"item": "no-such-item"}}))
    assert "error" in json.loads(tool_response(task, {"name": "delete_all", "arguments": {}}))
    assert "error" in json.loads(tool_response(task, {"name": "lookup", "arguments": "not a dict"}))


def test_calculate_accepts_only_arithmetic() -> None:
    assert safe_calculate("3 * 2.5 + (4 - 1) / 2") == "9"
    assert safe_calculate("-2 * 3") == "-6"
    assert safe_calculate("0.1 + 0.2") == "0.3"
    for bad in ("__import__('os')", "2 ** 100000", "a + 1", "1 / 0", "[1, 2]", "1 if 1 else 2"):
        with pytest.raises(ValueError):
            safe_calculate(bad)
    task = make_task(1, 1)
    assert json.loads(tool_response(task, {"name": "calculate", "arguments": {"expression": "1 + 2"}})) == {
        "result": "3"
    }
    assert "error" in json.loads(tool_response(task, {"name": "calculate", "arguments": {"expression": "x"}}))


def test_final_answer_parses_the_last_answer_line() -> None:
    assert final_answer("Let me add.\nANSWER: 12.50") == "12.50"
    assert final_answer("ANSWER: insufficient Apple 2") == "INSUFFICIENT apple 2"
    assert final_answer("ANSWER: INSUFFICIENT") == "INSUFFICIENT"
    assert final_answer("The total is 12.5") is None
    assert final_answer(None) is None


def test_reward_compares_numbers_exactly_after_normalization() -> None:
    task = next(make_task(s, 2) for s in range(100) if make_task(s, 2).kind == "total")
    value = float(task.answer)
    assert reward(task, f"{value:.2f}") == 1.0
    assert reward(task, f"{value + 0.01:.2f}") == 0.0
    assert reward(task, "INSUFFICIENT") == 0.0
    assert reward(task, None) == 0.0
    assert reward(task, "12abc") == 0.0


def test_shortage_answers_need_the_item_and_its_stock_so_guessing_scores_zero() -> None:
    task = next(make_task(s, 2) for s in range(100) if make_task(s, 2).kind == "insufficient")
    _, item, stock = task.answer.split()
    other = next(name for name in task.order if name != item)
    assert reward(task, final_answer(f"ANSWER: INSUFFICIENT {item} {stock}")) == 1.0
    assert reward(task, final_answer(f"ANSWER: insufficient {item.upper()} {stock}")) == 1.0
    # 以前的捷径: 不查询就直接回答 INSUFFICIENT。现在得 0 分。
    assert reward(task, final_answer("ANSWER: INSUFFICIENT")) == 0.0
    assert reward(task, final_answer(f"ANSWER: INSUFFICIENT {item} {int(stock) + 1}")) == 0.0
    assert reward(task, final_answer(f"ANSWER: INSUFFICIENT {other} {stock}")) == 0.0
    assert reward(task, final_answer("ANSWER: 12.5")) == 0.0


def test_initial_messages_state_the_answer_format_and_tools_match_hermes_schema() -> None:
    messages = initial_messages(make_task(2, 2))
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "ANSWER:" in messages[0]["content"] and "INSUFFICIENT <item> <stock>" in messages[0]["content"]
    assert {tool["function"]["name"] for tool in TOOLS} == {"lookup", "calculate"}
    assert all(tool["type"] == "function" and "parameters" in tool["function"] for tool in TOOLS)
