# Copyright (c) 2026 Jiahao Tan

"""Strict, loss-preserving rollout pairing without optional dependencies."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import zip_longest
from pathlib import Path
from typing import Any

from trace2batch.core import SampleRow, build_samples


@dataclass(frozen=True)
class PairedRollout:
    sample_id: str
    dataset: str
    transition: list[SampleRow]
    trajectory: list[SampleRow]
    stratum: str | None = None
    turns: int | None = None

    @property
    def policy_tokens(self) -> int:
        return sum(sum(row.response_mask) for row in self.transition)

    @property
    def group(self) -> str:
        """Verification and timing unit: the dataset, then |stratum and |k<turns> when the data records them."""
        parts = [self.dataset]
        if self.stratum is not None:
            parts.append(self.stratum)
        if self.turns is not None:
            parts.append(f"k{self.turns}")
        return "|".join(parts)


def parse_stratum(value: Any) -> tuple[int, int]:
    """A stratum "lo-hi" holds rollouts whose final complete context is in (lo, hi]."""
    if not isinstance(value, str) or not re.fullmatch(r"(0|[1-9][0-9]*)-[1-9][0-9]*", value):
        raise ValueError("stratum must look like 'lo-hi' with nonnegative integers")
    low, high = (int(part) for part in value.split("-"))
    if low >= high:
        raise ValueError("stratum lower bound must be below its upper bound")
    return low, high


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _ids(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value or any(type(token) is not int or token < 0 for token in value):
        raise ValueError(f"{name} must be a nonempty list of nonnegative integer tokens")
    return value


def _targets(rows: Sequence[SampleRow], max_total_length: int | None) -> Iterator[tuple[int, tuple[int, ...]]]:
    if not rows:
        raise ValueError("a representation has no rows or policy tokens")
    for row in rows:
        prompt = _ids(row.prompt_token_ids, "prompt_token_ids")
        response = _ids(row.response_token_ids, "response_token_ids")
        mask = row.response_mask
        if (
            not isinstance(mask, list)
            or len(mask) != len(response)
            or any(type(flag) is not int or flag not in (0, 1) for flag in mask)
            or not any(mask)
        ):
            raise ValueError("response_mask must contain one binary flag per token and a policy target")
        if row.is_drop:
            raise ValueError("dropped rows are forbidden")
        if max_total_length is not None and len(prompt) + len(response) > max_total_length:
            raise ValueError("complete row exceeds max_total_length; truncation is forbidden")
        context = list(prompt)
        for target, supervised in zip(response, mask, strict=True):
            if supervised:
                yield target, tuple(context)
            context.append(target)


def audit_pair(pair: PairedRollout, *, max_total_length: int | None = None) -> None:
    """Compare every target ID and all preceding tokens, in rollout order."""
    if max_total_length is not None:
        _positive_int(max_total_length, "max_total_length")
    compared = zip_longest(_targets(pair.transition, max_total_length), _targets(pair.trajectory, max_total_length))
    for position, (left, right) in enumerate(compared):
        if left is None or right is None:
            raise ValueError("representations have different policy target counts")
        left_id, left_context = left
        right_id, right_context = right
        if left_id != right_id:
            raise ValueError(f"policy target ID differs at supervised token {position}")
        if left_context != right_context:
            raise ValueError(f"full token context differs at supervised token {position}")


def pair_events(
    events: Sequence[Mapping[str, Any]], *, sample_id: str, dataset: str, max_total_length: int
) -> PairedRollout:
    """Build both representations after rejecting any lossy or invalid input."""
    _nonempty_text(sample_id, "sample_id")
    _nonempty_text(dataset, "dataset")
    _positive_int(max_total_length, "max_total_length")
    if not isinstance(events, list | tuple):
        raise ValueError("events must be a sequence of event objects")
    previous: list[int] | None = None
    requests = 0
    for event in events:
        if not isinstance(event, Mapping) or not isinstance(event.get("data"), Mapping):
            raise ValueError("each event must contain a data object")
        data = event["data"]
        kind = event.get("event_type")
        if kind == "reward":
            continue
        if kind != "model_request":
            raise ValueError("only model_request and reward events are supported")
        prompt = _ids(data.get("prompt_token_ids"), "prompt_token_ids")
        response = _ids(data.get("response_token_ids"), "response_token_ids")
        if data.get("status") == "error" or (isinstance(data.get("http_status"), int) and data["http_status"] >= 400):
            raise ValueError("failed model requests cannot be filtered from a benchmark rollout")
        if previous is not None and prompt[: len(previous)] != previous:
            raise ValueError(f"broken prefix before model request {requests}")
        if len(prompt) + len(response) > max_total_length:
            raise ValueError("complete context exceeds max_total_length; truncation is forbidden")
        previous = prompt + response
        requests += 1
    if not requests:
        raise ValueError("rollout has no model requests or policy tokens")
    built = [
        build_samples(
            events,
            mode=mode,
            max_prompt_length=max_total_length,
            max_response_length=max_total_length,
            max_total_length=max_total_length,
        )
        for mode in ("transition", "trajectory")
    ]
    for result in built:
        if result.zero_usable_rows_reason or any(
            value for name, value in result.diagnostics.items() if name not in ("model_requests", "kept_requests")
        ):
            raise ValueError("building representations filtered, truncated, split, or dropped a row")
    pair = PairedRollout(sample_id, dataset, built[0].rows, built[1].rows)
    audit_pair(pair, max_total_length=max_total_length)
    return pair


def load_pairs(path: str | Path) -> tuple[list[PairedRollout], dict[str, Any]]:
    """Load and re-audit schema v1 prepared events; never trust saved row counts."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int:
        raise ValueError("prepared data must be a schema_version=1 object")
    if document["schema_version"] != 1:
        raise ValueError("unsupported prepared schema_version")
    model = document.get("model")
    settings = document.get("settings")
    selection = document.get("selection")
    if not isinstance(model, dict) or not isinstance(settings, dict) or not isinstance(selection, dict):
        raise ValueError("prepared model, settings, and selection must be objects")
    _nonempty_text(model.get("repo"), "model.repo")
    revision = _nonempty_text(model.get("revision"), "model.revision")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("model.revision must pin a full commit hash")
    budget = _positive_int(settings.get("max_total_length"), "settings.max_total_length")
    rollouts = document.get("rollouts")
    if not isinstance(rollouts, list) or not rollouts:
        raise ValueError("prepared file must contain at least one complete rollout")
    pairs: list[PairedRollout] = []
    identities: set[tuple[str, str]] = set()
    for rollout in rollouts:
        if not isinstance(rollout, dict):
            raise ValueError("each prepared rollout must be an object")
        dataset = _nonempty_text(rollout.get("dataset"), "dataset")
        sample_id = _nonempty_text(rollout.get("sample_id"), "sample_id")
        identity = dataset, sample_id
        if identity in identities:
            raise ValueError(f"duplicate rollout identity: {identity}")
        identities.add(identity)
        events = rollout.get("events")
        if not isinstance(events, list):
            raise ValueError("prepared rollout events must be a list")
        pair = pair_events(events, sample_id=sample_id, dataset=dataset, max_total_length=budget)
        if "stratum" in rollout:
            low, high = parse_stratum(rollout["stratum"])
            final = pair.transition[-1]
            if not low < len(final.prompt_token_ids) + len(final.response_token_ids) <= high:
                raise ValueError(f"final complete context of {identity} lies outside its stratum")
            pair = replace(pair, stratum=rollout["stratum"])
        if "turns" in rollout:
            turns = rollout["turns"]
            if type(turns) is not int or turns != len(pair.transition):
                raise ValueError(f"turns of {identity} must equal its number of model requests")
            pair = replace(pair, turns=turns)
        pairs.append(pair)
    counts = Counter(pair.dataset for pair in pairs)
    if set(counts) - set(selection):
        raise ValueError("selection report is missing a selected dataset")
    for name, counts_report in selection.items():
        if not isinstance(counts_report, dict):
            raise ValueError("selection entries must contain dataset counts")
        if "selected" in counts_report and (
            type(counts_report["selected"]) is not int or counts_report["selected"] != counts[name]
        ):
            raise ValueError(f"selection count differs from audited rollouts for {name}")
    return pairs, document
