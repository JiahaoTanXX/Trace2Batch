# Copyright (c) 2026 Jiahao Tan

"""P3-B 的多轮工具调用环境: 确定性、可自动判分, 只用标准库。

每个任务有一张随机生成的库存表(物品 -> 单价、库存)和一个订单。模型需要先用 lookup 查询单价和库存,
可以用 calculate 做四则运算, 最后回复一行 "ANSWER: <总价>"; 有物品库存不足时回答
"ANSWER: INSUFFICIENT <物品> <实际库存>"。奖励是 0/1: 最终回答与标准答案在规范化后完全相同才得 1。
价格以分为单位存储, 标准答案没有浮点误差。

缺货答案必须写出物品和实际库存: pilot 中只写 "INSUFFICIENT" 就能拿到 30% 的奖励(基础模型不调用工具、
直接猜测), 这个捷径让成功率失去意义。现在不查询就无法给出缺货答案。
"""

from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext
from typing import Any

ITEMS = (
    "apple", "bolt", "candle", "drill", "envelope", "flask", "glove", "hammer", "ink", "jar",
    "kettle", "lamp", "magnet", "needle", "oven", "pencil", "quilt", "ruler", "stapler", "towel",
    "umbrella", "valve", "wrench", "yarn", "zipper", "brush", "cable", "funnel", "hinge", "ladder",
)  # fmt: skip
DISTRACTORS = 3
SHORTAGE_PROBABILITY = 0.3
MAX_ABS_RESULT = Decimal(10) ** 12
MAX_EXPRESSION_NODES = 64

SYSTEM_PROMPT = (
    "You are a purchasing assistant. Use the lookup tool to get each item's price and stock, and the calculate "
    "tool for arithmetic. When you know the result, reply without calling a tool and end with a line "
    "'ANSWER: <total cost>' (a number). If an item's stock is below the ordered quantity, end instead with "
    "'ANSWER: INSUFFICIENT <item> <stock>', giving that item and its current stock."
)
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up the unit price and the current stock of an item.",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string", "description": "Item name, e.g. apple"}},
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression with + - * / and parentheses.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "e.g. 3 * 2.5 + 4"}},
                "required": ["expression"],
            },
        },
    },
]
_ANSWER_RE = re.compile(r"^\s*ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Task:
    task_id: str
    inventory: dict[str, tuple[float, int]]  # 物品 -> (单价, 库存)
    order: dict[str, int]  # 物品 -> 数量
    answer: str  # 规范化后的总价, 或 "INSUFFICIENT <物品> <实际库存>"

    @property
    def kind(self) -> str:
        return "insufficient" if self.answer.startswith("INSUFFICIENT") else "total"

    @property
    def question(self) -> str:
        parts = [f"{qty} {item}" for item, qty in self.order.items()]
        return f"I want to buy {', '.join(parts)}. What is the total cost?"


def _format_cents(cents: int) -> str:
    return format((Decimal(cents) / 100).normalize(), "f")


def make_task(seed: int, n_items: int) -> Task:
    """由 (seed, n_items) 唯一确定的任务; n_items 是订单中的物品数, 决定需要的工具调用轮数。"""
    rng = random.Random(f"grpo-task-{seed}-{n_items}")
    names = rng.sample(ITEMS, n_items + DISTRACTORS)
    order = {name: rng.randint(1, 5) for name in names[:n_items]}
    cents = {name: rng.randint(1, 80) * 25 for name in names}
    stock = {name: rng.randint(0, 9) for name in names}
    short = rng.choice(names[:n_items]) if rng.random() < SHORTAGE_PROBABILITY else None
    for name, qty in order.items():
        # 只有被选中的物品库存不足, 其余物品库存都够; 这样答案只取决于是否发生了一次缺货。
        stock[name] = rng.randint(0, qty - 1) if name == short else max(stock[name], qty)
    inventory = {name: (cents[name] / 100, stock[name]) for name in sorted(names)}
    total = _format_cents(sum(qty * cents[name] for name, qty in order.items()))
    answer = f"INSUFFICIENT {short} {stock[short]}" if short else total
    return Task(f"{seed}-{n_items}", inventory, order, answer)


def initial_messages(task: Task) -> list[dict[str, Any]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task.question}]


def _evaluate(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        value = _evaluate(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub | ast.Mult | ast.Div):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def safe_calculate(expression: str) -> str:
    """只接受数字、+ - * /、括号和正负号; 其余一律拒绝。用 Decimal 计算, 0.1 + 0.2 得到 0.3。"""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid expression") from error
    if sum(1 for _ in ast.walk(tree)) > MAX_EXPRESSION_NODES:
        raise ValueError("expression too long")
    try:
        with localcontext() as context:
            context.prec = 28
            context.traps[DivisionByZero] = True
            value = _evaluate(tree)
    except (DivisionByZero, InvalidOperation) as error:
        raise ValueError("arithmetic error") from error
    if abs(value) > MAX_ABS_RESULT:
        raise ValueError("result too large")
    return format(value.quantize(Decimal("1e-6")).normalize(), "f")


def tool_response(task: Task, call: dict[str, Any]) -> str:
    """执行一次工具调用, 返回 JSON 字符串; 任何非法调用都返回 {"error": ...}, 不抛异常。"""
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return json.dumps({"error": "arguments must be a JSON object"})
    if call.get("name") == "lookup":
        item = str(arguments.get("item", "")).strip().lower()
        if item not in task.inventory:
            return json.dumps({"error": f"unknown item: {item}"})
        price, stock = task.inventory[item]
        return json.dumps({"item": item, "price": price, "stock": stock})
    if call.get("name") == "calculate":
        try:
            return json.dumps({"result": safe_calculate(str(arguments.get("expression", "")))})
        except ValueError as error:
            return json.dumps({"error": str(error)})
    return json.dumps({"error": f"unknown tool: {call.get('name')}"})


def final_answer(content: str | None) -> str | None:
    """最后一行 "ANSWER: ..." 的值; 缺货答案规范化为 "INSUFFICIENT <小写物品> <其余部分>"; 没有时返回 None。"""
    matches = _ANSWER_RE.findall(content or "")
    if not matches:
        return None
    value = matches[-1].strip()
    words = value.split()
    if words and words[0].upper() == "INSUFFICIENT":
        return " ".join(["INSUFFICIENT", *(w.lower() if i == 0 else w for i, w in enumerate(words[1:]))])
    return value


def reward(task: Task, answer: str | None) -> float:
    if answer is None:
        return 0.0
    if task.kind == "insufficient":
        return float(answer == task.answer)
    if answer.startswith("INSUFFICIENT"):
        return 0.0
    try:
        return float(Decimal(answer.replace("$", "").replace(",", "")) == Decimal(task.answer))
    except InvalidOperation:
        return 0.0
