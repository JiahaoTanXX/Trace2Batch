# Copyright (c) Microsoft. All rights reserved.
# Derived from microsoft/agent-lightning@69f5a40 agentlightning/verl/rollout_adapter.py (MIT).
# Modifications Copyright (c) 2026 Jiahao Tan.

from __future__ import annotations

import copy
import json
import runpy
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from examples.trace2batch.run import run_scenarios
from tests.test_trace2batch import SCENARIOS

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_report_is_deterministic_and_marks_reference_scope() -> None:
    report = run_scenarios(SCENARIOS)
    assert report == run_scenarios(SCENARIOS)
    assert report["synthetic"] is True
    assert report["real_training"] is False
    assert report["implementation"] == "stdlib_cpu_reference"
    assert report["project_name"] == "Trace2Batch"
    assert "repository_version" not in report
    assert report["scenario_count"] == 12
    assert report["comparison_count"] == 24
    assert all(result["expectations_passed"] for result in report["results"])


def test_bad_manual_expectation_fails_instead_of_printing_success() -> None:
    scenarios = copy.deepcopy(SCENARIOS[:1])
    scenarios[0]["expected"]["transition"]["n_rows"] = 99
    with pytest.raises(ValueError, match="不符合手算期望"):
        run_scenarios(scenarios)


def test_bad_reason_expectation_is_detected() -> None:
    scenarios = copy.deepcopy(SCENARIOS[:1])
    scenarios[0]["expected_reason"] = {"transition": "wrong"}
    with pytest.raises(ValueError, match="无可用行原因"):
        run_scenarios(scenarios)


def test_run_entrypoint_makes_no_network_calls(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    def reject_network(*args, **kwargs):
        raise AssertionError("本地参考实验不得建立网络连接")

    monkeypatch.setattr(socket, "socket", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    # Import-free execution avoids runpy's warning about an already imported CLI module.
    runpy.run_path(str(REPO_ROOT / "examples/trace2batch/run.py"), run_name="__main__")
    report = json.loads(capsys.readouterr().out)
    assert report["comparison_count"] == 24


def test_documented_module_command_emits_repeatable_json() -> None:
    command = [sys.executable, "-m", "examples.trace2batch.run"]
    first = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    second = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == ""
    assert json.loads(first.stdout)["scenario_count"] == 12
