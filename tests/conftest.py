# Copyright (c) 2026 Jiahao Tan

"""默认跳过 network 测试; 传入 --run-network 才会联网。"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-network", action="store_true", default=False, help="运行需要联网的测试")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="需要 --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
