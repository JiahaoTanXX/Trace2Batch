"""Offline safety checks for the self-contained training-cost submission bundle."""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
import zipfile
from collections import Counter
from contextlib import suppress
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "kaggle" / "train_cost_driver.py"
# The Kaggle driver template is private and kept out of the repository (kaggle/ is excluded locally),
# so these tests run only where it exists; clean clones and CI skip them explicitly.
pytestmark = pytest.mark.skipif(not DRIVER.exists(), reason="private Kaggle driver template is not in the repository")
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SOURCES = (
    "trace2batch/__init__.py",
    "trace2batch/core.py",
    "trace2batch/mismatch.py",
    "trace2batch/tito.py",
    "experiments/__init__.py",
    "experiments/train_cost/__init__.py",
    "experiments/train_cost/data.py",
    "experiments/train_cost/engine.py",
    "experiments/train_cost/run.py",
)


def bundle_module() -> ModuleType:
    try:
        return importlib.import_module("experiments.train_cost.bundle")
    except ModuleNotFoundError:
        pytest.fail("The safe bundle builder has not been implemented")


def driver_module(path: Path = DRIVER) -> ModuleType:
    if not path.exists():
        pytest.fail("The standalone driver has not been implemented")
    spec = importlib.util.spec_from_file_location("train_cost_test_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def prepared() -> dict:
    return {
        "schema_version": 1,
        "model": {"repo": "Qwen/Qwen3-0.6B", "revision": REVISION},
        "settings": {"max_total_length": 4096},
        "selection": {"toy": {"selected": 1}},
        "rollouts": [
            {
                "dataset": "toy",
                "sample_id": "a",
                "events": [
                    {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2], "response_token_ids": [3]}},
                    {
                        "event_type": "model_request",
                        "data": {"prompt_token_ids": [1, 2, 3, 4], "response_token_ids": [5]},
                    },
                ],
            }
        ],
    }


@pytest.fixture
def bundle_input(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "repo"
    for filename in SOURCES:
        path = source_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""An allowed Python source."""\n', encoding="utf-8")
    for filename in ("kaggle/private.py", ".venv/token", "private/kaggle.json", "trace2batch/secret.py"):
        path = source_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SHOULD_NEVER_ENTER_THE_PAYLOAD", encoding="utf-8")
    data = tmp_path / "prepared.json"
    data.write_text(json.dumps(prepared()), encoding="utf-8")
    return source_root, data


def build(tmp_path: Path, bundle_input: tuple[Path, Path]) -> tuple[dict, ModuleType]:
    source_root, data = bundle_input
    result = bundle_module().build_bundle(
        data, tmp_path / "out", owner="fixture-owner", driver_path=DRIVER, source_root=source_root
    )
    return result, driver_module(tmp_path / "out" / "pilot.py")


def test_bundle_contains_only_allowed_sources_and_data(tmp_path: Path, bundle_input: tuple[Path, Path]) -> None:
    result, module = build(tmp_path, bundle_input)
    payload = base64.b64decode(module.PAYLOAD_B64, validate=True)
    assert hashlib.sha256(payload).hexdigest() == result["payload_sha256"]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {*SOURCES, "data.json"}
        assert b"SHOULD_NEVER_ENTER_THE_PAYLOAD" not in b"".join(archive.read(name) for name in archive.namelist())
    metadata = json.loads((tmp_path / "out" / "kernel-metadata.json").read_text())
    assert metadata["id"] == "fixture-owner/trace2batch-p3-pilot"
    title_slug = re.sub(r"[^a-z0-9]+", "-", metadata["title"].lower()).strip("-")
    assert metadata["id"].split("/")[1] == title_slug
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["kernel_type"] == "script"
    assert metadata["code_file"] == "pilot.py"
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == ["kernel-metadata.json", "pilot.py"]


@pytest.mark.parametrize("placement", ["root", "model", "event"])
def test_bundle_rejects_unknown_prepared_fields(
    tmp_path: Path, bundle_input: tuple[Path, Path], placement: str
) -> None:
    source_root, data = bundle_input
    value = prepared()
    if placement == "root":
        value["api_key"] = "FAKE_CREDENTIAL_SENTINEL"
    elif placement == "model":
        value["model"]["api_key"] = "FAKE_CREDENTIAL_SENTINEL"
    else:
        value["rollouts"][0]["events"][0]["data"]["headers"] = {"Authorization": "FAKE_CREDENTIAL_SENTINEL"}
    data.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="field"):
        bundle_module().build_bundle(
            data, tmp_path / "out", owner="fixture-owner", driver_path=DRIVER, source_root=source_root
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["model", "revision", "empty", "length"])
def test_bundle_rejects_unsafe_configuration_before_writing(
    tmp_path: Path,
    bundle_input: tuple[Path, Path],
    change: str,
) -> None:
    source_root, data = bundle_input
    value = prepared()
    if change == "model":
        value["model"]["repo"] = "other/large-model"
    elif change == "revision":
        value["model"]["revision"] = "main"
    elif change == "empty":
        value["rollouts"] = []
    else:
        value["settings"]["max_total_length"] = 8192
    data.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        bundle_module().build_bundle(
            data, tmp_path / "out", owner="fixture-owner", driver_path=DRIVER, source_root=source_root
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["symlink", "syntax", "oversize", "missing"])
def test_bundle_rejects_invalid_allowlisted_sources(
    tmp_path: Path,
    bundle_input: tuple[Path, Path],
    change: str,
) -> None:
    source_root, data = bundle_input
    source = source_root / "trace2batch" / "core.py"
    if change == "symlink":
        source.unlink()
        source.symlink_to(data)
    elif change == "syntax":
        source.write_text("def broken(\n")
    elif change == "oversize":
        source.write_text("#" * (2 * 1024 * 1024))
    else:
        source.unlink()
    with pytest.raises(ValueError):
        bundle_module().build_bundle(
            data, tmp_path / "out", owner="fixture-owner", driver_path=DRIVER, source_root=source_root
        )
    assert not (tmp_path / "out").exists()


def test_bundle_rejects_invalid_owner(tmp_path: Path, bundle_input: tuple[Path, Path]) -> None:
    source_root, data = bundle_input
    with pytest.raises(ValueError):
        bundle_module().build_bundle(
            data, tmp_path / "out", owner="../private", driver_path=DRIVER, source_root=source_root
        )


@pytest.mark.parametrize("change", ["hash", "path", "duplicate", "oversize"])
def test_driver_rejects_tampered_or_unsafe_payload(
    tmp_path: Path,
    bundle_input: tuple[Path, Path],
    change: str,
) -> None:
    _, module = build(tmp_path, bundle_input)
    payload = base64.b64decode(module.PAYLOAD_B64)
    if change != "hash":
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if change == "path":
                archive.writestr("../escape.py", "print('unsafe')")
            elif change == "duplicate":
                with pytest.warns(UserWarning):
                    archive.writestr("data.json", "{}")
                    archive.writestr("data.json", "{}")
            else:
                archive.writestr("data.json", "x" * (9 * 1024 * 1024))
        payload = stream.getvalue()
    digest = "0" * 64 if change == "hash" else hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError):
        module.unpack_payload(base64.b64encode(payload).decode(), digest, tmp_path / "unpacked")
    assert not (tmp_path / "escape.py").exists()


class FakeCommands:
    def __init__(
        self,
        *,
        no_gpu: bool = False,
        install_timeout: bool = False,
        train_timeout: bool = False,
        upgrade_torch: bool = False,
        wrong_dependencies: bool = False,
        invalid_data: bool = False,
        kaggle_ensurepip: bool = False,
        no_pip: bool = False,
        venv_is_base: bool = False,
        pip_check_output: str | None = None,
        lora_preflight_fails: bool = False,
    ) -> None:
        self.lora_preflight_fails = lora_preflight_fails
        self.pip_check_output = pip_check_output
        self.kaggle_ensurepip = kaggle_ensurepip
        self.no_pip = no_pip
        self.venv_is_base = venv_is_base
        self.no_gpu = no_gpu
        self.install_timeout = install_timeout
        self.train_timeout = train_timeout
        self.upgrade_torch = upgrade_torch
        self.wrong_dependencies = wrong_dependencies
        self.invalid_data = invalid_data
        self.commands: list[tuple[list[str], float, dict]] = []

    def __call__(self, command: list[str], *, timeout: float, log_path: Path, env: dict) -> subprocess.CompletedProcess:
        self.commands.append((command, timeout, env))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log:
            log.write("stage invoked\n")
        if "venv" in command and "--without-pip" not in command and self.kaggle_ensurepip:
            # Mirrors Version 1: venv hides ensurepip's own output behind the outer CalledProcessError.
            return subprocess.CompletedProcess(command, 1, stdout="Error: ... ensurepip ... exit status 1.")
        if "ensurepip" in command:
            with log_path.open("a") as log:
                log.write("ensurepip fixture detail\n")
            return subprocess.CompletedProcess(command, int(self.kaggle_ensurepip), stdout="")
        if "-c" in command:
            script = command[command.index("-c") + 1]
            if "load_pairs" in script and self.invalid_data:
                return subprocess.CompletedProcess(command, 1, stdout="invalid data")
            if "lora_preflight" in script:
                if self.lora_preflight_fails:
                    return subprocess.CompletedProcess(command, 1, stdout="ImportError: incompatible torchao")
                report = {
                    "lora_preflight": True,
                    "trainable_parameters": 512,
                    "loss_finite": True,
                    "grads_finite": True,
                }
                return subprocess.CompletedProcess(command, 0, stdout=json.dumps(report))
            if "runtime_closure" in script:
                closure = ["torch", "transformers", "peft", "tokenizers", "huggingface-hub"]
                return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"runtime_closure": closure}))
            if "venv_pip" in script:
                venv = Path(command[0]).parents[1]
                report = {
                    "venv_pip": True,
                    "prefix": str(venv),
                    "base_prefix": "/usr" if not self.venv_is_base else str(venv),
                    "purelib": str(venv / "lib" / "site-packages"),
                    "pip_version": None if self.no_pip else "25.3",
                    "pip_file": None if self.no_pip else "/usr/lib/pip/__init__.py",
                }
                return subprocess.CompletedProcess(command, 0, stdout=json.dumps(report))
            return subprocess.CompletedProcess(
                command,
                int(self.no_gpu),
                stdout=json.dumps(
                    {
                        "torch_version": "2.9.0+cu128",
                        "cuda": not self.no_gpu,
                        "requested_versions": {
                            "transformers": "5.20.0",
                            "peft": "wrong" if self.wrong_dependencies else "0.21.2",
                            "torchao": "0.17.0",
                        },
                    }
                ),
            )
        if command[-2:] == ["pip", "check"] and self.pip_check_output is not None:
            return subprocess.CompletedProcess(command, 1, stdout=f"$ pip check\n{self.pip_check_output}\n")
        if "--dry-run" in command:
            plan = (
                {"install": [{"metadata": {"name": "torch", "version": "2.10.0"}}]}
                if self.upgrade_torch
                else {"install": []}
            )
            Path(command[command.index("--report") + 1]).write_text(json.dumps(plan))
        if "install" in command and "--dry-run" not in command and self.install_timeout:
            raise subprocess.TimeoutExpired(command, timeout)
        if "experiments.train_cost.run" in command and self.train_timeout:
            raise subprocess.TimeoutExpired(command, timeout)
        return subprocess.CompletedProcess(command, 0, stdout="")


def test_driver_stops_before_install_without_gpu(tmp_path: Path, bundle_input: tuple[Path, Path]) -> None:
    _, module = build(tmp_path, bundle_input)
    commands = FakeCommands(no_gpu=True)
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "error"
    assert len(commands.commands) == 1
    assert not any("install" in command for command, _, _ in commands.commands)
    saved = json.loads((tmp_path / "working" / "driver-status.json").read_text())
    assert saved["status"] == "error"


@pytest.mark.parametrize("stage", ["install", "training"])
def test_driver_records_timeouts_without_retries(
    tmp_path: Path,
    bundle_input: tuple[Path, Path],
    stage: str,
) -> None:
    _, module = build(tmp_path, bundle_input)
    commands = FakeCommands(install_timeout=stage == "install", train_timeout=stage == "training")
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "timeout"
    launched = [command for command, _, _ in commands.commands if "experiments.train_cost.run" in command]
    assert len(launched) == int(stage == "training")
    assert (tmp_path / "working" / "driver.log").exists()
    saved = json.loads((tmp_path / "working" / "driver-status.json").read_text())
    assert saved["status"] == "timeout"


def test_driver_refuses_resolver_torch_replacement(tmp_path: Path, bundle_input: tuple[Path, Path]) -> None:
    _, module = build(tmp_path, bundle_input)
    commands = FakeCommands(upgrade_torch=True)
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "error"
    assert "torch" in status["error"].lower()
    assert not any("install" in command and "--dry-run" not in command for command, _, _ in commands.commands)


def test_driver_keeps_one_gpu_and_training_write_reserve(tmp_path: Path, bundle_input: tuple[Path, Path]) -> None:
    _, module = build(tmp_path, bundle_input)
    commands = FakeCommands()
    status = module.run_driver(
        output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands, clock=lambda: 10.0
    )
    assert status["status"] == "complete"
    train = [
        (command, timeout, env)
        for command, timeout, env in commands.commands
        if "experiments.train_cost.run" in command
    ]
    assert len(train) == 1
    command, timeout, env = train[0]
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--repeats") + 1] == "2"
    soft_budget = float(command[command.index("--time-budget-seconds") + 1])
    assert soft_budget <= 3360
    assert timeout - soft_budget >= 180
    assert timeout < 3600
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert all(0 < timeout < 3600 for _, timeout, _ in commands.commands)
    installs = [command for command, _, _ in commands.commands if "install" in command]
    assert all("transformers==5.20.0" in command and "peft==0.21.2" in command for command in installs)
    assert all("--constraint" in command for command in installs)
    assert not any(arg.startswith(("torch==", "vllm")) for command in installs for arg in command)
    assert any("--system-site-packages" in command for command, _, _ in commands.commands)
    assert (tmp_path / "working" / "dependency-versions.json").exists()


def test_driver_explicit_model_cache_is_inside_temporary_hf_home(
    tmp_path: Path, bundle_input: tuple[Path, Path]
) -> None:
    _, module = build(tmp_path, bundle_input)
    commands = FakeCommands()
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "complete"
    for _, _, env in commands.commands:
        assert env.get("TRACE2BATCH_HF_CACHE") == str(Path(env["HF_HOME"]) / "hub")


def test_process_timeout_stops_child_and_preserves_log(tmp_path: Path) -> None:
    module = driver_module()
    log = tmp_path / "log.txt"
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        module.run_process(
            [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"],
            timeout=0.1,
            log_path=log,
            env={},
        )
    assert time.monotonic() - started < 3
    assert "started" in log.read_text()


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX process groups")
def test_timeout_kills_sigterm_ignoring_child_after_leader_exits(tmp_path: Path) -> None:
    module = driver_module()
    pidfile = tmp_path / "child.pid"
    heartbeat = tmp_path / "child.heartbeat"
    child = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(pidfile)!r}).write_text(str(os.getpid()));\n"
        f"while True:\n Path({str(heartbeat)!r}).write_text(str(time.monotonic())); time.sleep(0.02)"
    )
    leader = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
        "time.sleep(60)"
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            module.run_process([sys.executable, "-c", leader], timeout=1, log_path=tmp_path / "log", env={})
        assert pidfile.exists(), "Child fixture must start before the timeout"
        assert heartbeat.exists()
        time.sleep(0.05)
        stopped_at = heartbeat.read_text()
        time.sleep(0.15)
        assert heartbeat.read_text() == stopped_at, "Child is still executing after the leader was terminated"
    finally:
        if pidfile.exists():
            with suppress(ProcessLookupError):
                os.kill(int(pidfile.read_text()), signal.SIGKILL)


def test_driver_accepts_preparation_provenance(tmp_path: Path, bundle_input: tuple[Path, Path]) -> None:
    _, data = bundle_input
    value = prepared()
    value["model"]["template_sha256"] = "a" * 64
    value["model"]["tokenizer_class"] = "Qwen2TokenizerFast"
    data.write_text(json.dumps(value))
    _, module = build(tmp_path, bundle_input)
    data_path = module.unpack_payload(module.PAYLOAD_B64, module.PAYLOAD_SHA256, tmp_path / "source")
    assert json.loads(data_path.read_text())["model"]["template_sha256"] == "a" * 64


def test_process_output_cannot_reuse_json_from_earlier_stage(tmp_path: Path) -> None:
    module = driver_module()
    log = tmp_path / "log.txt"
    log.write_text('{"torch_version":"stale", "cuda":true}\n')
    result = module.run_process([sys.executable, "-c", "pass"], timeout=2, log_path=log, env={})
    with pytest.raises(ValueError, match="JSON"):
        module._json_output(result)


def test_driver_does_not_forward_private_credentials(
    tmp_path: Path,
    bundle_input: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, module = build(tmp_path, bundle_input)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fixture-not-a-real-token")
    monkeypatch.setenv("HF_TOKEN", "fixture-not-a-real-token")
    commands = FakeCommands()
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "complete"
    assert all("KAGGLE_API_TOKEN" not in env and "HF_TOKEN" not in env for _, _, env in commands.commands)


def test_driver_reaudits_data_before_environment_install(tmp_path: Path, bundle_input: tuple[Path, Path]) -> None:
    _, module = build(tmp_path, bundle_input)
    commands = FakeCommands(invalid_data=True)
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "error"
    assert status["stage"] == "data_preflight"
    assert not any("install" in command for command, _, _ in commands.commands)


def test_driver_checks_actual_pinned_dependencies(tmp_path: Path, bundle_input: tuple[Path, Path]) -> None:
    _, module = build(tmp_path, bundle_input)
    commands = FakeCommands(wrong_dependencies=True)
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "error"
    assert not any("experiments.train_cost.run" in command for command, _, _ in commands.commands)


def test_driver_environment_does_not_depend_on_ensurepip(tmp_path: Path, bundle_input: tuple[Path, Path]) -> None:
    _, module = build(tmp_path, bundle_input)
    commands = FakeCommands(kaggle_ensurepip=True)
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "complete", status.get("error")
    venvs = [command for command, _, _ in commands.commands if command[1:3] == ["-m", "venv"]]
    assert venvs and all("--system-site-packages" in command and "--without-pip" in command for command in venvs)
    # The diagnostic ensurepip probe runs once, its failure is recorded, and it is never retried.
    probes = [command for command, _, _ in commands.commands if "ensurepip" in command]
    assert len(probes) == 1
    assert status["ensurepip_probe"]["ensurepip_returncode"] == 1
    assert "ensurepip fixture detail" in (tmp_path / "working" / "driver.log").read_text()
    installs = [command for command, _, _ in commands.commands if "install" in command]
    assert installs and all(Path(command[0]).parents[1].name == "venv" for command in installs)


@pytest.mark.parametrize("problem", ["no_pip", "venv_is_base"])
def test_driver_stops_before_install_without_usable_venv_pip(
    tmp_path: Path, bundle_input: tuple[Path, Path], problem: str
) -> None:
    _, module = build(tmp_path, bundle_input)
    commands = FakeCommands(no_pip=problem == "no_pip", venv_is_base=problem == "venv_is_base")
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "error"
    assert status["stage"] == "pip_bootstrap"
    assert not any("install" in command for command, _, _ in commands.commands)
    assert not any("experiments.train_cost.run" in command for command, _, _ in commands.commands)


def test_driver_records_interpreter_diagnostics_without_values(
    tmp_path: Path,
    bundle_input: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, module = build(tmp_path, bundle_input)
    monkeypatch.setenv("PIP_INDEX_URL", "https://user:fixture-not-a-real-secret@example.invalid/simple")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fixture-not-a-real-token")
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=FakeCommands())
    diagnostics = status["environment_diagnostics"]
    assert diagnostics["python_version"] == sys.version
    assert diagnostics["executable"] == sys.executable
    assert "PIP_INDEX_URL" in diagnostics["environment_names"]
    assert "KAGGLE_API_TOKEN" not in diagnostics["environment_names"]
    saved = (tmp_path / "working" / "driver-status.json").read_text() + (
        tmp_path / "working" / "driver.log"
    ).read_text()
    assert "fixture-not-a-real" not in saved


def test_real_venv_pip_probe_matches_interpreter(tmp_path: Path) -> None:
    """Runs the real venv creation and probe; the verdict must match whether this interpreter has pip."""
    module = driver_module()
    venv = tmp_path / "venv"
    log = tmp_path / "log.txt"
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1"}
    created = module.run_process(
        [sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", str(venv)],
        timeout=60,
        log_path=log,
        env=env,
    )
    assert created.returncode == 0, log.read_text()
    probe = module.run_process(
        [str(venv / "bin" / "python"), "-c", module.VENV_PIP_PROBE], timeout=60, log_path=log, env=env
    )
    assert probe.returncode == 0, log.read_text()
    report = module._json_output(probe)
    # Ground truth independent of the probe: a venv sees the *base* interpreter's site-packages.
    has_pip = subprocess.run([str(venv / "bin" / "python"), "-c", "import pip"], env=env, check=False).returncode == 0
    if not has_pip:
        with pytest.raises(ValueError, match="pip"):
            module.check_venv_pip(report, venv)
    else:
        checked = module.check_venv_pip(report, venv)
        assert Path(checked["purelib"]).resolve().is_relative_to(venv.resolve())


@pytest.mark.parametrize("change", ["total", "per_dataset", "turns"])
def test_bundle_refuses_pilot_scale_excess(
    tmp_path: Path,
    bundle_input: tuple[Path, Path],
    change: str,
) -> None:
    source_root, data = bundle_input
    value = prepared()
    if change == "turns":
        events = []
        context = [1, 2]
        for turn in range(5):
            events.append(
                {
                    "event_type": "model_request",
                    "data": {
                        "prompt_token_ids": list(context),
                        "response_token_ids": [10 + turn],
                    },
                }
            )
            context += [10 + turn, 20 + turn]
        value["rollouts"][0]["events"] = events
    else:
        count = 13 if change == "total" else 5
        events = value["rollouts"][0]["events"]
        value["rollouts"] = [
            {"dataset": f"set-{i}" if change == "total" else "toy", "sample_id": str(i), "events": events}
            for i in range(count)
        ]
        value["selection"] = (
            {f"set-{i}": {"selected": 1} for i in range(count)}
            if change == "total"
            else {
                "toy": {"selected": count},
            }
        )
    data.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="pilot"):
        bundle_module().build_bundle(
            data, tmp_path / "out", owner="fixture-owner", driver_path=DRIVER, source_root=source_root
        )
    assert not (tmp_path / "out").exists()


def test_bundle_accepts_directory_alias_for_prepared_file(
    tmp_path: Path,
    bundle_input: tuple[Path, Path],
) -> None:
    source_root, data = bundle_input
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "template.py").write_text(DRIVER.read_text())
    result = bundle_module().build_bundle(
        alias / data.name,
        tmp_path / "out",
        owner="fixture-owner",
        driver_path=alias / "template.py",
        source_root=source_root,
    )
    assert result["payload_bytes"] > 0


def test_driver_records_unrelated_system_conflicts_without_stopping(
    tmp_path: Path, bundle_input: tuple[Path, Path]
) -> None:
    _, module = build(tmp_path, bundle_input)
    line = "streamlit 1.51.0 has requirement protobuf<7,>=3.20, but you have protobuf 7.34.1."
    commands = FakeCommands(pip_check_output=line)
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "complete", status.get("error")
    assert status["dependency_check"]["unrelated_system_conflicts"] == [line]
    assert status["dependency_check"]["runtime_conflicts"] == []


@pytest.mark.parametrize(
    "output",
    [
        "transformers 5.20.0 has requirement tokenizers<0.24, but you have tokenizers 0.25.0.",
        "Huggingface_Hub 2.2.0 requires httpx2, which is not installed.",
        "pip check produced an unrecognised failure",
    ],
)
def test_driver_stops_on_runtime_dependency_conflict(
    tmp_path: Path, bundle_input: tuple[Path, Path], output: str
) -> None:
    _, module = build(tmp_path, bundle_input)
    commands = FakeCommands(pip_check_output=output)
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "error"
    assert status["stage"] == "dependency_check"
    assert not any("experiments.train_cost.run" in command for command, _, _ in commands.commands)


def test_driver_runs_offline_lora_preflight_before_training(tmp_path: Path, bundle_input: tuple[Path, Path]) -> None:
    _, module = build(tmp_path, bundle_input)
    commands = FakeCommands()
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "complete", status.get("error")
    order = [
        index
        for index, (command, _, _) in enumerate(commands.commands)
        if "experiments.train_cost.run" in command or ("-c" in command and "lora_preflight" in command[-1])
    ]
    assert len(order) == 2 and "lora_preflight" in commands.commands[order[0]][0][-1]
    _, timeout, env = commands.commands[order[0]]
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert 0 < timeout <= 180
    assert status["lora_preflight"]["trainable_parameters"] > 0


def test_driver_stops_before_model_download_when_lora_preflight_fails(
    tmp_path: Path, bundle_input: tuple[Path, Path]
) -> None:
    _, module = build(tmp_path, bundle_input)
    commands = FakeCommands(lora_preflight_fails=True)
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "error"
    assert status["stage"] == "lora_preflight"
    assert not any("experiments.train_cost.run" in command for command, _, _ in commands.commands)


def test_driver_pins_torchao_compatible_with_peft(tmp_path: Path, bundle_input: tuple[Path, Path]) -> None:
    _, module = build(tmp_path, bundle_input)
    commands = FakeCommands()
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "complete", status.get("error")
    installs = [command for command, _, _ in commands.commands if "install" in command]
    assert installs and all("torchao==0.17.0" in command for command in installs)


def test_lora_preflight_matches_run_configuration() -> None:
    module = driver_module()
    run_source = (ROOT / "experiments" / "train_cost" / "run.py").read_text()
    for fragment in ("r=8", "lora_alpha=16", "lora_dropout=0.0", 'target_modules=["q_proj", "v_proj"]', 'bias="none"'):
        assert fragment in run_source
        assert fragment in module.LORA_PREFLIGHT


def test_real_lora_preflight_runs_without_network(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    module = driver_module()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HOME": str(tmp_path),
    }
    result = module.run_process(
        [sys.executable, "-c", module.LORA_PREFLIGHT], timeout=120, log_path=tmp_path / "log", env=env
    )
    assert result.returncode == 0, (tmp_path / "log").read_text()[-2000:]
    report = module.check_lora_preflight(module._json_output(result))
    assert report["trainable_parameters"] > 0


def test_real_version_probe_reports_the_importable_distribution(tmp_path: Path) -> None:
    """Shadowing as in a --system-site-packages venv: the first path entry wins, later copies are listed apart."""
    module = driver_module()
    for folder, version in (("venv", "2.0"), ("system", "1.0")):
        info = tmp_path / folder / f"shadowfixture-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: shadowfixture\nVersion: {version}\n")
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": f"{tmp_path / 'venv'}:{tmp_path / 'system'}"}
    script = module.VERSION_PROBE.replace("for n in REQUESTED", "for n in ()")
    result = module.run_process([sys.executable, "-c", script], timeout=120, log_path=tmp_path / "log", env=env)
    assert result.returncode == 0, (tmp_path / "log").read_text()[-2000:]
    report = module._json_output(result)
    assert report["versions"]["shadowfixture"] == "2.0"
    assert report["shadowed_versions"]["shadowfixture"] == ["1.0"]


def turn_events(contexts: list[int]) -> list[dict]:
    events, context = [], []
    for turn, target in enumerate(contexts):
        prompt = context + [7] * (target - 1 - len(context))
        events.append(
            {"event_type": "model_request", "data": {"prompt_token_ids": prompt, "response_token_ids": [20 + turn]}}
        )
        context = [*prompt, 20 + turn]
    return events


def stratified(rollouts: list[tuple[str, str, str, list[int]]], *, max_total_length: int = 6144) -> dict:
    value = prepared()
    value["settings"] = {
        "max_total_length": max_total_length,
        "strata": ["0-2048", "2048-4096", "4096-6144"],
        "min_turns": 2,
        "cells": None,
        "selection_order": "highest stratum first; longest whole-turn prefix; each candidate once",
    }
    value["rollouts"] = [
        {"dataset": dataset, "sample_id": sample_id, "stratum": stratum, "events": turn_events(contexts)}
        for dataset, sample_id, stratum, contexts in rollouts
    ]
    counts = Counter(dataset for dataset, *_ in rollouts)
    value["selection"] = {
        name: {
            "requested_per_cell": 6,
            "selected": count,
            "strata": {"4096-6144": {"chosen": count, "emitted": True}},
            "rejected": {},
            "decisions": [
                {"sample_id": "x", "selected": True, "stratum": "4096-6144", "turns": 2, "context_tokens": 5000}
            ],
        }
        for name, count in counts.items()
    }
    return value


def build_stratified(tmp_path: Path, bundle_input: tuple[Path, Path], value: dict, **kwargs) -> tuple[dict, ModuleType]:
    source_root, data = bundle_input
    data.write_text(json.dumps(value))
    result = bundle_module().build_bundle(
        data, tmp_path / "out", owner="fixture-owner", driver_path=DRIVER, source_root=source_root, **kwargs
    )
    return result, driver_module(tmp_path / "out" / "pilot.py")


def test_stratified_profile_keeps_strata_and_bakes_the_session_limit(
    tmp_path: Path, bundle_input: tuple[Path, Path]
) -> None:
    value = stratified([("toy", str(i), "4096-6144", [3000, 6000]) for i in range(6)])
    result, module = build_stratified(tmp_path, bundle_input, value, profile="stratified", session_limit_minutes=45)
    assert result["profile"] == "stratified" and result["session_limit_seconds"] == 2700
    assert module.SESSION_LIMIT_SECONDS == 2700
    data_path = module.unpack_payload(module.PAYLOAD_B64, module.PAYLOAD_SHA256, tmp_path / "source")
    packaged = json.loads(data_path.read_text())
    assert {rollout["stratum"] for rollout in packaged["rollouts"]} == {"4096-6144"}
    assert packaged["settings"]["strata"] == ["0-2048", "2048-4096", "4096-6144"]
    metadata = json.loads((tmp_path / "out" / "kernel-metadata.json").read_text())
    assert metadata["id"] == "fixture-owner/trace2batch-p3-stratified"
    assert metadata["id"].split("/")[1] == re.sub(r"[^a-z0-9]+", "-", metadata["title"].lower()).strip("-")


def test_pilot_profile_still_defaults_to_one_hour(tmp_path: Path, bundle_input: tuple[Path, Path]) -> None:
    result, module = build(tmp_path, bundle_input)
    assert result["profile"] == "pilot" and result["session_limit_seconds"] == 3600
    assert module.SESSION_LIMIT_SECONDS == 3600


@pytest.mark.parametrize("change", ["context", "per_cell", "total", "turns", "no_stratum"])
def test_stratified_profile_refuses_scale_excess(tmp_path: Path, bundle_input: tuple[Path, Path], change: str) -> None:
    rows = [("toy", str(i), "4096-6144", [3000, 6000]) for i in range(6)]
    if change == "per_cell":
        rows.append(("toy", "7", "4096-6144", [3000, 6000]))
    elif change == "total":
        rows = [(f"set-{i // 6}", str(i), "4096-6144", [3000, 6000]) for i in range(49)]
    elif change == "turns":
        rows[0] = ("toy", "0", "4096-6144", [1000 + 200 * turn for turn in range(21)])
    value = stratified(rows, max_total_length=6400 if change == "context" else 6144)
    if change == "no_stratum":
        del value["rollouts"][0]["stratum"]
    with pytest.raises(ValueError, match="stratified"):
        build_stratified(tmp_path, bundle_input, value, profile="stratified", session_limit_minutes=45)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("minutes", [0, 721, 45.5, "45"])
def test_session_limit_must_be_whole_minutes_within_kaggle_bounds(
    tmp_path: Path, bundle_input: tuple[Path, Path], minutes: object
) -> None:
    value = stratified([("toy", "0", "4096-6144", [3000, 6000])])
    with pytest.raises(ValueError, match="session"):
        build_stratified(tmp_path, bundle_input, value, profile="stratified", session_limit_minutes=minutes)


def test_packaged_data_size_is_still_bounded(
    tmp_path: Path, bundle_input: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bundle_module(), "MAX_DATA_BYTES", 1000)
    value = stratified([("toy", "0", "4096-6144", [3000, 6000])])
    with pytest.raises(ValueError, match="packaged data"):
        build_stratified(tmp_path, bundle_input, value, profile="stratified", session_limit_minutes=45)


def test_driver_sets_allocator_and_accepts_only_bounded_context(
    tmp_path: Path, bundle_input: tuple[Path, Path]
) -> None:
    value = stratified([("toy", "0", "4096-6144", [3000, 6000])])
    _, module = build_stratified(tmp_path, bundle_input, value, profile="stratified", session_limit_minutes=45)
    commands = FakeCommands()
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "complete", status.get("error")
    assert all(env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True" for _, _, env in commands.commands)
    assert all(timeout <= 2700 - 60 for _, timeout, _ in commands.commands)
    train = next(command for command, _, _ in commands.commands if "experiments.train_cost.run" in command)
    assert int(train[train.index("--time-budget-seconds") + 1]) <= 2700 - 60 - 180


def test_diagnose_run_mode_is_baked_and_passed_to_training(tmp_path: Path, bundle_input: tuple[Path, Path]) -> None:
    value = stratified([("toy", "0", "4096-6144", [3000, 6000])])
    result, module = build_stratified(
        tmp_path, bundle_input, value, profile="stratified", session_limit_minutes=30, run_mode="diagnose"
    )
    assert result["run_mode"] == "diagnose" and module.RUN_MODE == "diagnose"
    commands = FakeCommands()
    status = module.run_driver(output_dir=tmp_path / "working", temp_root=tmp_path, command_runner=commands)
    assert status["status"] == "complete", status.get("error")
    train = next(command for command, _, _ in commands.commands if "experiments.train_cost.run" in command)
    assert "--diagnose-gradients" in train


def test_benchmark_is_the_default_run_mode(tmp_path: Path, bundle_input: tuple[Path, Path]) -> None:
    result, module = build(tmp_path, bundle_input)
    assert result["run_mode"] == "benchmark" and module.RUN_MODE == "benchmark"
    with pytest.raises(ValueError, match="run_mode"):
        build_stratified(
            tmp_path / "other",
            bundle_input,
            stratified([("toy", "0", "4096-6144", [3000, 6000])]),
            profile="stratified",
            session_limit_minutes=30,
            run_mode="train",
        )


def test_stratified_profile_keeps_turns_fields(tmp_path: Path, bundle_input: tuple[Path, Path]) -> None:
    value = stratified([("toy", str(i), "4096-6144", [3000, 6000]) for i in range(2)])
    for rollout in value["rollouts"]:
        rollout["turns"] = 2
    value["settings"].update({"window": "4096-6144", "turns": [2], "seed": 1, "extra_candidates": 1000})
    _, module = build_stratified(tmp_path, bundle_input, value, profile="stratified", session_limit_minutes=45)
    data_path = module.unpack_payload(module.PAYLOAD_B64, module.PAYLOAD_SHA256, tmp_path / "src")
    packaged = json.loads(data_path.read_text())
    assert {rollout["turns"] for rollout in packaged["rollouts"]} == {2}
    assert packaged["settings"]["turns"] == [2] and packaged["settings"]["seed"] == 1


@pytest.mark.parametrize(("megabytes", "accepted"), [(5, True), (9, False)])
def test_driver_bounds_packaged_data_at_eight_mib(tmp_path: Path, megabytes: int, accepted: bool) -> None:
    module = driver_module()
    value = prepared()
    value["padding"] = "x" * (megabytes * 1024 * 1024)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.writestr(name, '"""An allowed Python source."""\n')
        archive.writestr("data.json", json.dumps(value))
    payload = stream.getvalue()
    arguments = (base64.b64encode(payload).decode(), hashlib.sha256(payload).hexdigest(), tmp_path / "unpacked")
    if accepted:
        assert module.unpack_payload(*arguments).exists()
    else:
        with pytest.raises(ValueError):
            module.unpack_payload(*arguments)


def test_bundle_accepts_packaged_data_between_four_and_eight_mib(
    tmp_path: Path, bundle_input: tuple[Path, Path]
) -> None:
    value = stratified([("toy", "0", "4096-6144", [3000, 6000])])
    decision = {"sample_id": "x" * 200, "selected": False, "reason": "y" * 400}
    value["selection"]["toy"]["decisions"] = [decision] * 7000  # about 4.6 MiB once packaged
    result, _ = build_stratified(tmp_path, bundle_input, value, profile="stratified", session_limit_minutes=45)
    assert result["payload_bytes"] < 8 * 1024 * 1024
