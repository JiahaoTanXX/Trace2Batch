# Copyright (c) 2026 Jiahao Tan
"""Build a self-contained private pilot without touching Kaggle or credentials."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_FILES = (
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
MODEL_REPO = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MAX_SOURCE_BYTES = 1024 * 1024
MAX_INPUT_BYTES = 16 * 1024 * 1024
# Token-ID JSON compresses about 20x; 8 MiB uncompressed still stays far below the payload limit.
MAX_DATA_BYTES = 8 * 1024 * 1024
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
SESSION_LINE = "SESSION_LIMIT_SECONDS = 3600\n"
RUN_MODE_LINE = 'RUN_MODE = "benchmark"\n'
RUN_MODES = ("benchmark", "diagnose")
# Explicit scale limits; each is a reviewed experiment, never an implicit default.
PROFILES: dict[str, dict[str, Any]] = {
    "pilot": {
        "slug": "trace2batch-p3-pilot",
        "title": "Trace2Batch P3 Pilot",
        "max_context": 4096,
        "max_rollouts": 12,
        "max_per_group": 4,
        "max_turns": 4,
        "strata": False,
    },
    "stratified": {
        "slug": "trace2batch-p3-stratified",
        "title": "Trace2Batch P3 Stratified",
        "max_context": 6144,
        "max_rollouts": 48,
        "max_per_group": 6,
        "max_turns": 20,
        "strata": True,
    },
}


def _read_file(path: Path, limit: int) -> bytes:
    if not path.is_file() or path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError(f"Expected a regular file without symlinks: {path.name}")
    if path.stat().st_size > limit:
        raise ValueError(f"File exceeds the size limit: {path.name}")
    return path.read_bytes()


def _python_source(path: Path) -> bytes:
    content = _read_file(path, MAX_SOURCE_BYTES)
    try:
        ast.parse(content.decode("utf-8"), filename=path.name)
    except (SyntaxError, UnicodeError) as exc:
        raise ValueError(f"Invalid UTF-8 Python source: {path.name}") from exc
    return content


def _fields(value: Any, allowed: str, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - set(allowed.split()):
        raise ValueError(f"Unknown or invalid prepared fields in {where}")
    return dict(value)


def _public_value(value: Any) -> Any:
    if isinstance(value, str):
        if "KGAT_" in value:
            raise ValueError("Credential-like value in prepared fields")
    elif isinstance(value, list):
        return [_public_value(item) for item in value]
    elif value is not None and (type(value) not in (bool, int, float) or not math.isfinite(value)):
        raise ValueError("Prepared scalar fields cannot contain objects or nonfinite values")
    return value


def _public_prepared(document: dict[str, Any]) -> dict[str, Any]:
    """Project the known schema; unknown nested fields cannot ride along in JSON."""
    result = _fields(document, "schema_version model settings selection rollouts", "root")
    model = _fields(
        result["model"],
        "repo revision name family end_markers template_kwargs synthetic_reasoning keep_reasoning_in_history "
        "string_tool_arguments passthrough chat_template template_sha256 tokenizer_sha256 tokenizer_class "
        "transformers_version tokenizers_version",
        "model",
    )
    kwargs = _fields(model.pop("template_kwargs", {}), "enable_thinking", "model.template_kwargs")
    result["model"] = {key: _public_value(value) for key, value in model.items()}
    result["model"]["template_kwargs"] = {key: _public_value(value) for key, value in kwargs.items()}
    settings = _fields(
        result["settings"],
        "max_total_length max_turns samples_per_dataset candidate_pool_per_dataset seed event_mode "
        "local_files_only fixed_template_date datasets strata min_turns cells selection_order "
        "window turns extra_candidates excluded_inputs",
        "settings",
    )
    datasets = settings.pop("datasets", [])
    excluded_inputs = settings.pop("excluded_inputs", None)
    result["settings"] = {key: _public_value(value) for key, value in settings.items()}
    if excluded_inputs is not None:
        result["settings"]["excluded_inputs"] = [
            {key: _public_value(value) for key, value in _fields(item, "name sha256", "excluded input").items()}
            for item in excluded_inputs
        ]
    result["settings"]["datasets"] = [
        {
            key: _public_value(value)
            for key, value in _fields(
                item, "name repo revision shard license shard_sha256", "dataset provenance"
            ).items()
        }
        for item in datasets
    ]
    selection = {}
    for name, report in result["selection"].items():
        _public_value(name)
        item = _fields(
            report,
            "requested requested_per_cell selected scanned candidate_pool eligible_in_shard rejected decisions strata "
            "cells extra_candidates excluded",
            "selection",
        )
        rejected = item.pop("rejected", {})
        decisions = item.pop("decisions", [])
        strata = item.pop("strata", None)
        cells = item.pop("cells", None)
        selection[name] = {key: _public_value(value) for key, value in item.items()}
        if cells is not None:
            selection[name]["cells"] = {
                _public_value(cell): _public_value(count)
                for cell, count in _fields(cells, " ".join(cells), "cells").items()
            }
        if strata is not None:
            selection[name]["strata"] = {
                _public_value(stratum): {
                    key: _public_value(value) for key, value in _fields(entry, "chosen emitted", "strata").items()
                }
                for stratum, entry in _fields(strata, " ".join(strata), "strata").items()
            }
        selection[name]["rejected"] = {_public_value(key): _public_value(value) for key, value in rejected.items()}
        selection[name]["decisions"] = [
            {
                key: _public_value(value)
                for key, value in _fields(
                    decision,
                    "sample_id selected turns policy_tokens context_tokens reason stratum",
                    "selection decision",
                ).items()
            }
            for decision in decisions
        ]
    result["selection"] = selection
    rollouts = []
    for raw in result["rollouts"]:
        rollout = _fields(raw, "dataset sample_id stratum turns events", "rollout")
        events = []
        for raw_event in rollout["events"]:
            event = _fields(raw_event, "event_type data", "event")
            data = _fields(
                event["data"],
                "value"
                if event["event_type"] == "reward"
                else "prompt_token_ids response_token_ids status http_status",
                "event.data",
            )
            events.append({"event_type": event["event_type"], "data": {k: _public_value(v) for k, v in data.items()}})
        projected = {"dataset": _public_value(rollout["dataset"]), "sample_id": _public_value(rollout["sample_id"])}
        for optional in ("stratum", "turns"):
            if optional in rollout:
                projected[optional] = _public_value(rollout[optional])
        rollouts.append({**projected, "events": events})
    result["rollouts"] = rollouts
    return result


def _prepared_data(path: Path, profile: str) -> bytes:
    limits = PROFILES[profile]
    path = path.parent.resolve() / path.name
    content = _read_file(path.absolute(), MAX_INPUT_BYTES)
    try:
        value = json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid prepared JSON") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Prepared JSON must have schema_version 1")
    model = value.get("model")
    if not isinstance(model, dict) or model.get("repo") != MODEL_REPO or model.get("revision") != MODEL_REVISION:
        raise ValueError(f"The {profile} requires the fixed Qwen3-0.6B model revision")
    settings = value.get("settings")
    if not isinstance(settings, dict) or type(settings.get("max_total_length")) is not int:
        raise ValueError("Prepared JSON needs an integer max_total_length")
    if not 1 <= settings["max_total_length"] <= limits["max_context"]:
        raise ValueError(f"The {profile} context limit must be between 1 and {limits['max_context']}")
    rollouts = value.get("rollouts")
    if not isinstance(rollouts, list) or not rollouts:
        raise ValueError(f"The {profile} requires nonempty prepared rollouts")
    if limits["strata"] and not all(isinstance(row, dict) and "stratum" in row for row in rollouts):
        raise ValueError("The stratified profile requires a stratum on every rollout")
    from experiments.train_cost.data import load_pairs

    pairs, _ = load_pairs(path)
    if not pairs:
        raise ValueError("Prepared data has no valid paired rollouts")
    per_group = max(Counter(pair.group for pair in pairs).values())
    if len(pairs) > limits["max_rollouts"] or per_group > limits["max_per_group"]:
        raise ValueError(
            f"The {profile} permits at most {limits['max_rollouts']} rollouts and {limits['max_per_group']} per group"
        )
    if any(
        sum(event.get("event_type") == "model_request" for event in row["events"]) > limits["max_turns"]
        for row in rollouts
    ):
        raise ValueError(f"The {profile} permits at most {limits['max_turns']} model requests per rollout")
    packaged = json.dumps(_public_prepared(value), ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    if len(packaged) > MAX_DATA_BYTES:
        raise ValueError("The packaged data exceeds the size limit")
    return packaged


def build_bundle(
    data_path: str | Path,
    out_dir: str | Path,
    *,
    owner: str,
    driver_path: str | Path,
    source_root: str | Path | None = None,
    profile: str = "pilot",
    session_limit_minutes: int = 60,
    run_mode: str = "benchmark",
) -> dict[str, Any]:
    """Validate local inputs, then write exactly pilot.py and private metadata."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,49}", owner):
        raise ValueError("owner must be an account slug")
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}")
    # Kaggle sessions run for at most 12 hours; the cap is fixed per bundle and recorded in it.
    if type(session_limit_minutes) is not int or not 10 <= session_limit_minutes <= 720:
        raise ValueError("session_limit_minutes must be a whole number of minutes from 10 to 720")
    session_limit_seconds = session_limit_minutes * 60
    if run_mode not in RUN_MODES:
        raise ValueError(f"run_mode must be one of {RUN_MODES}")
    root = Path(source_root).resolve() if source_root is not None else Path(__file__).resolve().parents[2]
    contents = {name: _python_source(root / name) for name in SOURCE_FILES}
    contents["data.json"] = _prepared_data(Path(data_path), profile)
    driver = Path(driver_path)
    template = _python_source(driver.parent.resolve() / driver.name).decode("utf-8")
    markers = ("__TRACE2BATCH_PAYLOAD_B64__", "__TRACE2BATCH_PAYLOAD_SHA256__")
    if (
        any(template.count(marker) != 1 for marker in markers)
        or template.count(SESSION_LINE) != 1
        or template.count(RUN_MODE_LINE) != 1
    ):
        raise ValueError("Driver template must contain each payload marker, session limit and run mode exactly once")
    template = template.replace(SESSION_LINE, f"SESSION_LIMIT_SECONDS = {session_limit_seconds}\n")
    template = template.replace(RUN_MODE_LINE, f'RUN_MODE = "{run_mode}"\n')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in contents.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    payload = stream.getvalue()
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("Compressed payload exceeds the size limit")
    digest = hashlib.sha256(payload).hexdigest()
    script = template.replace(markers[0], base64.b64encode(payload).decode("ascii")).replace(markers[1], digest)
    ast.parse(script, filename="pilot.py")
    metadata = {
        "id": f"{owner}/{PROFILES[profile]['slug']}",
        "title": PROFILES[profile]["title"],
        "code_file": "pilot.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    output = Path(out_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "pilot.py").write_text(script, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "payload_sha256": digest,
        "payload_bytes": len(payload),
        "files": sorted(contents),
        "out": str(output),
        "profile": profile,
        "session_limit_seconds": session_limit_seconds,
        "run_mode": run_mode,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="pilot")
    parser.add_argument("--session-limit-minutes", type=int, default=60)
    parser.add_argument("--run-mode", choices=RUN_MODES, default="benchmark")
    args = parser.parse_args(argv)
    try:
        result = build_bundle(
            args.data,
            args.out,
            owner=args.owner,
            driver_path=args.driver,
            profile=args.profile,
            session_limit_minutes=args.session_limit_minutes,
            run_mode=args.run_mode,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
