# Copyright (c) 2026 Jiahao Tan

"""训练成本入口的预检和结果持久化测试, 不加载模型。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.train_cost.run import atomic_write, main


def prepared(path: Path, *, repo: str = "Qwen/Qwen3-0.6B") -> None:
    events = [
        {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2], "response_token_ids": [3]}},
        {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2, 3, 4], "response_token_ids": [5, 6]}},
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {"repo": repo, "revision": "c1899de289a04d12100db370d81485cdf75e47ca"},
                "settings": {"max_total_length": 32},
                "selection": {"fixture": {"selected": 1}},
                "rollouts": [{"dataset": "fixture", "sample_id": "r1", "events": events}],
            }
        )
    )


def test_check_only_validates_without_importing_optional_training_dependencies(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    prepared(data)
    out = tmp_path / "check"
    code = (
        "import sys; from experiments.train_cost.run import main; "
        f"main(['--data', {str(data)!r}, '--out', {str(out)!r}, '--check-only']); "
        "assert not any(n in sys.modules for n in ('torch', 'transformers', 'peft'))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((out / "results.json").read_text())
    assert report["status"] == "checked"
    assert report["data_summary"]["rollouts"] == 1
    assert report["data_summary"]["policy_tokens"] == 3
    assert report["benchmark"] == {}


@pytest.mark.parametrize(
    "option,value", [("--time-budget-seconds", "0"), ("--repeats", "0"), ("--learning-rate", "nan")]
)
def test_invalid_budget_rejected_before_model_loading(tmp_path: Path, option: str, value: str) -> None:
    data = tmp_path / "data.json"
    prepared(data)
    with pytest.raises(SystemExit):
        main(["--data", str(data), "--out", str(tmp_path / "out"), option, value, "--check-only"])


def test_disallowed_model_rejected_before_model_loading(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    prepared(data, repo="other/large-model")
    with pytest.raises(ValueError, match=r"model|模型"):
        main(["--data", str(data), "--out", str(tmp_path / "out"), "--check-only"])
    assert not (tmp_path / "out" / "results.json").exists()


def test_refuse_to_overwrite_previous_results(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    prepared(data)
    out = tmp_path / "out"
    out.mkdir()
    (out / "results.json").write_text('{"status":"old"}')
    with pytest.raises(FileExistsError):
        main(["--data", str(data), "--out", str(out), "--check-only"])
    assert json.loads((out / "results.json").read_text()) == {"status": "old"}


def test_atomic_write_rejects_non_finite_json_and_preserves_previous_result(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    atomic_write(path, {"status": "initializing"})
    with pytest.raises(ValueError):
        atomic_write(path, {"loss": float("nan")})
    assert json.loads(path.read_text()) == {"status": "initializing"}


def test_runtime_failure_is_saved_before_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data.json"
    prepared(data)

    def unavailable(*args: object, **kwargs: object) -> object:
        raise RuntimeError("CUDA preflight fixture failure")

    monkeypatch.setattr("experiments.train_cost.run.load_model", unavailable)
    with pytest.raises(RuntimeError, match="fixture failure"):
        main(["--data", str(data), "--out", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out" / "results.json").read_text())
    assert report["status"] == "failed"
    assert "fixture failure" in report["error"]
    assert report["complete"] is False


def stratified(path: Path) -> None:
    short = [{"event_type": "model_request", "data": {"prompt_token_ids": [1, 2], "response_token_ids": [3]}}]
    long = [
        {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2], "response_token_ids": [3]}},
        {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2, 3, 4], "response_token_ids": [5, 6]}},
    ]
    rollouts = [
        {"dataset": "fixture", "sample_id": "short", "stratum": "0-4", "events": short},
        {"dataset": "fixture", "sample_id": "long", "stratum": "4-8", "events": long},
        {"dataset": "other", "sample_id": "long2", "stratum": "4-8", "events": long},
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {"repo": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"},
                "settings": {"max_total_length": 8},
                "selection": {"fixture": {"selected": 2}, "other": {"selected": 1}},
                "rollouts": rollouts,
            }
        )
    )


def test_check_only_reports_cells(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    stratified(data)
    main(["--data", str(data), "--out", str(tmp_path / "out"), "--check-only"])
    summary = json.loads((tmp_path / "out" / "results.json").read_text())["data_summary"]
    assert summary["datasets"] == {"fixture": 2, "other": 1}
    assert summary["cells"] == {"fixture|0-4": 1, "fixture|4-8": 1, "other|4-8": 1}


def test_verification_and_benchmark_run_per_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data.json"
    stratified(data)
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr("experiments.train_cost.run.load_model", lambda *a, **k: (object(), {"vocab_size": 100}))

    def verify(model: object, pair: object, **kwargs: object) -> dict:
        calls.append(("verify", [pair.group]))  # type: ignore[attr-defined]
        return {"passed": True}

    def bench(model: object, pairs: list, **kwargs: object) -> dict:
        calls.append(("benchmark", [pair.group for pair in pairs]))
        return {"complete": True, "status": "complete"}

    monkeypatch.setattr("experiments.train_cost.engine.verify_equivalence", verify)
    monkeypatch.setattr("experiments.train_cost.engine.benchmark", bench)
    main(["--data", str(data), "--out", str(tmp_path / "out"), "--device", "cpu"])
    report = json.loads((tmp_path / "out" / "results.json").read_text())
    cells = ["fixture|0-4", "fixture|4-8", "other|4-8"]
    assert sorted(report["verification"]) == cells and sorted(report["benchmark"]) == cells
    assert [groups for kind, groups in calls if kind == "benchmark"] == [[cell] for cell in cells]


def test_diagnose_mode_checks_every_pair_and_never_benchmarks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data.json"
    stratified(data)
    seen: list[tuple[str, tuple]] = []
    monkeypatch.setattr("experiments.train_cost.run.load_model", lambda *a, **k: (object(), {"vocab_size": 100}))

    def diagnose(model: object, pair: object, *, loss_scales: tuple, **kwargs: object) -> dict:
        seen.append((pair.sample_id, tuple(loss_scales)))  # type: ignore[attr-defined]
        return {"sample_id": pair.sample_id, "group": pair.group, "results": []}  # type: ignore[attr-defined]

    def forbidden(*args: object, **kwargs: object) -> dict:
        raise AssertionError("diagnose mode must not run the gate or the benchmark")

    monkeypatch.setattr("experiments.train_cost.engine.gradient_diagnostics", diagnose)
    monkeypatch.setattr("experiments.train_cost.engine.verify_equivalence", forbidden)
    monkeypatch.setattr("experiments.train_cost.engine.benchmark", forbidden)
    out = tmp_path / "out"
    main(["--data", str(data), "--out", str(out), "--device", "cpu", "--diagnose-gradients", "--loss-scales", "1,4096"])
    report = json.loads((out / "results.json").read_text())
    assert report["status"] == "complete" and report["mode"] == "gradient_diagnostics"
    assert sorted(seen) == [("long", (1.0, 4096.0)), ("long2", (1.0, 4096.0)), ("short", (1.0, 4096.0))]
    assert sorted(report["gradient_diagnostics"]) == ["fixture|0-4", "fixture|4-8", "other|4-8"]
    assert report["benchmark"] == {} and report["verification"] == {}


@pytest.mark.parametrize("scales", ["0,1", "1,-4", "1,x", "", "1,inf"])
def test_invalid_loss_scales_rejected_before_model_loading(
    tmp_path: Path, scales: str, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "data.json"
    stratified(data)
    with pytest.raises(SystemExit):
        main(["--data", str(data), "--out", str(tmp_path / "out"), "--diagnose-gradients", "--loss-scales", scales])
    message = capsys.readouterr().err
    assert "--loss-scales" in message and "unrecognized" not in message
    assert not (tmp_path / "out").exists()


def test_gate_uses_training_like_loss_scaling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data.json"
    stratified(data)
    scales: list[object] = []
    monkeypatch.setattr("experiments.train_cost.run.load_model", lambda *a, **k: (object(), {"vocab_size": 100}))

    def verify(model: object, pair: object, **kwargs: object) -> dict:
        scales.append(kwargs.get("loss_scale"))
        return {"passed": True}

    monkeypatch.setattr("experiments.train_cost.engine.verify_equivalence", verify)
    monkeypatch.setattr(
        "experiments.train_cost.engine.benchmark", lambda *a, **k: {"complete": True, "status": "complete"}
    )
    main(["--data", str(data), "--out", str(tmp_path / "out"), "--device", "cpu"])
    assert scales == [4096.0, 4096.0, 4096.0]


def test_long_context_options_reach_the_model_and_every_engine_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data.json"
    stratified(data)
    loaded: list[object] = []
    seen: list[object] = []

    def load(*args: object, **kwargs: object) -> tuple[object, dict]:
        loaded.append(kwargs.get("attention"))
        return object(), {"vocab_size": 100}

    def verify(model: object, pair: object, **kwargs: object) -> dict:
        seen.append(kwargs.get("policy_logits_only"))
        return {"passed": True}

    def bench(model: object, pairs: list, **kwargs: object) -> dict:
        seen.append(kwargs.get("policy_logits_only"))
        return {"complete": True, "status": "complete"}

    monkeypatch.setattr("experiments.train_cost.run.load_model", load)
    monkeypatch.setattr("experiments.train_cost.engine.verify_equivalence", verify)
    monkeypatch.setattr("experiments.train_cost.engine.benchmark", bench)
    out = tmp_path / "out"
    options = ["--attention", "sdpa_repeat_kv", "--policy-logits-only"]
    main(["--data", str(data), "--out", str(out), "--device", "cpu", *options])
    report = json.loads((out / "results.json").read_text())
    assert loaded == ["sdpa_repeat_kv"] and seen == [True] * 6
    assert report["training_settings"]["attention"] == "sdpa_repeat_kv"
    assert report["training_settings"]["policy_logits_only"] is True


def test_default_settings_keep_the_stage_one_and_two_path(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    stratified(data)
    main(["--data", str(data), "--out", str(tmp_path / "out"), "--check-only"])
    settings = json.loads((tmp_path / "out" / "results.json").read_text())["training_settings"]
    assert settings["attention"] == "sdpa" and settings["policy_logits_only"] is False


def test_groups_option_runs_only_the_listed_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data.json"
    stratified(data)
    benchmarked: list[list[str]] = []
    monkeypatch.setattr("experiments.train_cost.run.load_model", lambda *a, **k: (object(), {"vocab_size": 100}))
    monkeypatch.setattr("experiments.train_cost.engine.verify_equivalence", lambda *a, **k: {"passed": True})

    def bench(model: object, pairs: list, **kwargs: object) -> dict:
        benchmarked.append([pair.group for pair in pairs])
        return {"complete": True, "status": "complete"}

    monkeypatch.setattr("experiments.train_cost.engine.benchmark", bench)
    out = tmp_path / "out"
    main(["--data", str(data), "--out", str(out), "--device", "cpu", "--groups", "fixture|4-8,other|4-8"])
    report = json.loads((out / "results.json").read_text())
    assert benchmarked == [["fixture|4-8"], ["other|4-8"]]
    assert report["data_summary"]["rollouts"] == 2 and report["training_settings"]["groups"] == "fixture|4-8,other|4-8"


def test_unknown_group_is_rejected_before_model_loading(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    stratified(data)
    with pytest.raises(SystemExit):
        main(["--data", str(data), "--out", str(tmp_path / "out"), "--groups", "missing|0-4"])
    assert not (tmp_path / "out").exists()
