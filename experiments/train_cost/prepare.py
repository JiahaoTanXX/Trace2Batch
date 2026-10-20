# Copyright (c) 2026 Jiahao Tan
# pyright: reportMissingImports=false

"""Prepare fixed-revision canonical token events using local caches only.

    python -m experiments.train_cost.prepare --out prepared.json

No model is loaded. Missing tokenizer or dataset cache entries are errors; this
entry point disables Hub network access and implicit credential lookup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from typing import Any

from experiments.train_cost.data import pair_events, parse_stratum


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _context_after(event: dict[str, Any]) -> int:
    return len(event["data"]["prompt_token_ids"]) + len(event["data"]["response_token_ids"])


def select_strata(
    candidates: Sequence[tuple[str, list[dict[str, Any]]]],
    strata: Sequence[tuple[int, int]],
    *,
    samples: int,
    min_turns: int,
) -> tuple[dict[str, list[tuple[str, list[dict[str, Any]]]]], list[dict[str, Any]]]:
    """Pick up to `samples` candidates per stratum, in candidate (fixed seed) order.

    A candidate contributes its longest whole-turn prefix whose final complete context
    lies in (lo, hi] and that has at least `min_turns` turns; no turn is ever cut.
    Higher strata are filled first, because long prefixes are scarcer, and each
    candidate is used at most once so strata never share a trajectory.
    """
    chosen: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {f"{low}-{high}": [] for low, high in strata}
    used: set[str] = set()
    decisions: list[dict[str, Any]] = []
    for low, high in sorted(strata, reverse=True):
        name = f"{low}-{high}"
        for sample_id, events in candidates:
            if len(chosen[name]) == samples:
                break
            if sample_id in used:
                continue
            turns = [k for k in range(min_turns, len(events) + 1) if low < _context_after(events[k - 1]) <= high]
            if turns:
                used.add(sample_id)
                chosen[name].append((sample_id, events[: turns[-1]]))
                decisions.append(
                    {
                        "sample_id": sample_id,
                        "selected": True,
                        "stratum": name,
                        "turns": turns[-1],
                        "context_tokens": _context_after(events[turns[-1] - 1]),
                    }
                )
    for sample_id, _ in candidates:
        if sample_id not in used:
            decisions.append({"sample_id": sample_id, "selected": False, "reason": "no qualifying prefix or unused"})
    return chosen, decisions


def parse_options(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=None, help="fixed dataset names; default all three")
    parser.add_argument("--samples", type=_positive, default=4, help="maximum selected rollouts per dataset or cell")
    parser.add_argument("--max-turns", type=_positive, default=4, help="use only the first assistant turns")
    parser.add_argument("--max-total-length", type=_positive, default=4096)
    parser.add_argument("--candidates", type=_positive, default=300, help="fixed-seed candidate pool per dataset")
    parser.add_argument("--strata", default=None, help="comma-separated lo-hi context strata, e.g. 0-2048,2048-4096")
    parser.add_argument("--min-turns", type=int, default=2, help="stratified mode: fewest turns per rollout")
    parser.add_argument("--cells", default=None, help="stratified mode: comma-separated dataset|lo-hi to emit")
    args = parser.parse_args(argv)
    cells_text = args.cells
    if args.strata is None:
        if cells_text is not None:
            parser.error("--cells requires --strata")
        args.cells = None
        return args
    try:
        strata = [parse_stratum(item.strip()) for item in args.strata.split(",")]
    except ValueError as error:
        parser.error(f"--strata: {error}")
    if any(previous[1] > current[0] for previous, current in pairwise(strata)):
        parser.error("--strata must be sorted and non-overlapping")
    if args.min_turns < 1:
        parser.error("--min-turns must be positive")
    args.strata = strata
    if cells_text is not None:
        from experiments.template_prefix.run import DATASETS

        names = {spec.name for spec in DATASETS}
        known = {f"{low}-{high}" for low, high in strata}
        cells = set()
        for item in cells_text.split(","):
            dataset, _, stratum = item.strip().partition("|")
            if dataset not in names or stratum not in known:
                parser.error(f"--cells entry {item!r} must be dataset|lo-hi from the known datasets and --strata")
            cells.add((dataset, stratum))
        args.cells = cells
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_options(argv)
    parser = argparse.ArgumentParser(prog="prepare")

    # Set these before importing optional packages. Hub also consults the
    # explicit constant below if it was imported previously by an API caller.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    from huggingface_hub import constants, hf_hub_download

    constants.HF_HUB_OFFLINE = True
    constants.HF_HUB_DISABLE_IMPLICIT_TOKEN = True
    from experiments.real_sampling.run import MODELS
    from experiments.template_prefix.run import (
        CACHE_DIR,
        DATASETS,
        FIXED_TEMPLATE_DATE,
        Renderer,
        build_turn_events,
        load_conversations,
    )

    names = {spec.name for spec in DATASETS}
    if args.datasets is not None and (set(args.datasets) - names or len(args.datasets) != len(set(args.datasets))):
        parser.error(f"--datasets must contain unique names from {sorted(names)}")
    specs = [spec for spec in DATASETS if args.datasets is None or spec.name in args.datasets]
    config = MODELS["qwen3-0.6b+patch2"]
    renderer = Renderer(config)
    template = renderer.chat_template or renderer.tokenizer.chat_template
    tokenizer_text = renderer.tokenizer.backend_tokenizer.to_str()
    selected: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    dataset_versions: list[dict[str, Any]] = []
    for spec in specs:
        # Explicit cache preflight gives a clear missing-data error before the
        # shared loader runs; the offline environment forbids its network path.
        shard = Path(
            hf_hub_download(
                spec.repo,
                spec.shard,
                repo_type="dataset",
                revision=spec.revision,
                cache_dir=CACHE_DIR,
                local_files_only=True,
                token=False,
            )
        )
        conversations, eligible = load_conversations(spec, args.candidates, 0)
        rejected: Counter[str] = Counter()
        decisions: list[dict[str, Any]] = []
        kept = 0
        if args.strata is not None:
            candidates: list[tuple[str, list[dict[str, Any]]]] = []
            for conversation in conversations:
                try:
                    built = build_turn_events(conversation, renderer, args.max_turns)["B_vllm_token_ids"]
                    if (
                        built.generation_unresolved
                        or built.generation_not_token_prefix
                        or built.generation_not_text_prefix
                    ):
                        raise ValueError("generation prefix could not be preserved")
                except Exception as error:  # Reported, never shortened or repaired.
                    reason = f"{type(error).__name__}: {str(error)[:180]}"
                    rejected[reason] += 1
                    decisions.append({"sample_id": conversation.sample_id, "selected": False, "reason": reason})
                    continue
                candidates.append((conversation.sample_id, built.events))
            chosen, choices = select_strata(candidates, args.strata, samples=args.samples, min_turns=args.min_turns)
            decisions.extend(choices)
            strata_report: dict[str, Any] = {}
            for name, items in chosen.items():
                emitted = args.cells is None or (spec.name, name) in args.cells
                for sample_id, events in items:
                    # Re-audit with the stratum's own budget before anything is written.
                    pair_events(events, sample_id=sample_id, dataset=spec.name, max_total_length=parse_stratum(name)[1])
                    if emitted:
                        rollout = {"dataset": spec.name, "sample_id": sample_id, "stratum": name, "events": events}
                        selected.append(rollout)
                        kept += 1
                strata_report[name] = {"chosen": len(items), "emitted": emitted}
            selection[spec.name] = {
                "requested_per_cell": args.samples,
                "selected": kept,
                "strata": strata_report,
                "candidate_pool": len(conversations),
                "eligible_in_shard": eligible,
                "rejected": dict(rejected),
                "decisions": decisions,
            }
            with shard.open("rb") as stream:
                shard_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            dataset_versions.append({**asdict(spec), "shard_sha256": shard_sha256})
            print(f"{spec.name}: {strata_report}, emitted={kept}, rejected={dict(rejected)}", flush=True)
            continue
        for conversation in conversations:
            if kept == args.samples:
                break
            try:
                built = build_turn_events(conversation, renderer, args.max_turns)["B_vllm_token_ids"]
                if built.generation_unresolved or built.generation_not_token_prefix or built.generation_not_text_prefix:
                    raise ValueError("generation prefix could not be preserved")
                pair = pair_events(
                    built.events,
                    sample_id=conversation.sample_id,
                    dataset=spec.name,
                    max_total_length=args.max_total_length,
                )
            except Exception as error:  # A malformed conversation must be reported, never shortened.
                reason = f"{type(error).__name__}: {str(error)[:180]}"
                rejected[reason] += 1
                decisions.append({"sample_id": conversation.sample_id, "selected": False, "reason": reason})
                continue
            selected.append({"dataset": spec.name, "sample_id": conversation.sample_id, "events": built.events})
            decisions.append(
                {
                    "sample_id": conversation.sample_id,
                    "selected": True,
                    "turns": len(pair.transition),
                    "policy_tokens": pair.policy_tokens,
                    "context_tokens": len(pair.trajectory[0].prompt_token_ids)
                    + len(pair.trajectory[0].response_token_ids),
                }
            )
            kept += 1
        selection[spec.name] = {
            "requested": args.samples,
            "selected": kept,
            "scanned": len(decisions),
            "candidate_pool": len(conversations),
            "eligible_in_shard": eligible,
            "rejected": dict(rejected),
            "decisions": decisions,
        }
        with shard.open("rb") as stream:
            shard_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        dataset_versions.append({**asdict(spec), "shard_sha256": shard_sha256})
        print(
            f"{spec.name}: selected={kept}/{args.samples}, scanned={len(decisions)}, rejected={dict(rejected)}",
            flush=True,
        )
    report = {
        "schema_version": 1,
        "model": {
            **asdict(config),
            "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(tokenizer_text.encode()).hexdigest(),
            "tokenizer_class": type(renderer.tokenizer).__name__,
            "transformers_version": version("transformers"),
            "tokenizers_version": version("tokenizers"),
        },
        "settings": {
            "max_total_length": args.max_total_length if args.strata is None else args.strata[-1][1],
            "max_turns": args.max_turns,
            "samples_per_dataset": args.samples,
            "candidate_pool_per_dataset": args.candidates,
            "seed": 0,
            "event_mode": "B_vllm_token_ids",
            "local_files_only": True,
            "fixed_template_date": FIXED_TEMPLATE_DATE.isoformat(),
            "datasets": dataset_versions,
            **(
                {}
                if args.strata is None
                else {
                    "strata": [f"{low}-{high}" for low, high in args.strata],
                    "min_turns": args.min_turns,
                    "cells": None if args.cells is None else sorted(f"{d}|{s}" for d, s in args.cells),
                    "selection_order": "highest stratum first; longest whole-turn prefix; each candidate once",
                }
            ),
        },
        "selection": selection,
        "rollouts": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    temporary.replace(args.out)
    if not selected:
        raise SystemExit(f"No complete rollout fit the limits; selection report saved to {args.out}")
    print(f"Prepared {len(selected)} audited rollouts: {args.out}", flush=True)


if __name__ == "__main__":
    main()
