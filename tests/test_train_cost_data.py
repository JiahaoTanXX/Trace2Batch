# Copyright (c) 2026 Jiahao Tan

from __future__ import annotations

import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from trace2batch.core import SampleRow


def _data() -> Any:
    return importlib.import_module("experiments.train_cost.data")


def _events() -> list[dict[str, Any]]:
    return [
        {"event_type": "model_request", "data": {"prompt_token_ids": [1, 2], "response_token_ids": [10, 11]}},
        {
            "event_type": "model_request",
            "data": {"prompt_token_ids": [1, 2, 10, 11, 90], "response_token_ids": [12]},
        },
    ]


def _prepared() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": {"repo": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"},
        "settings": {"max_total_length": 6},
        "selection": {"toy": {"selected": 1}},
        "rollouts": [{"sample_id": "original-id", "dataset": "toy", "events": _events()}],
    }


def test_pair_retains_all_policy_targets_and_masks_observations() -> None:
    pair = _data().pair_events(_events(), sample_id="original-id", dataset="toy", max_total_length=6)
    assert (pair.sample_id, pair.dataset, pair.policy_tokens) == ("original-id", "toy", 3)
    assert [row.response_token_ids for row in pair.transition] == [[10, 11], [12]]
    assert len(pair.trajectory) == 1
    assert pair.trajectory[0].prompt_token_ids == [1, 2]
    assert pair.trajectory[0].response_token_ids == [10, 11, 90, 12]
    assert pair.trajectory[0].response_mask == [1, 1, 0, 1]
    assert pair.trajectory[0].source_turn_indices == [0, 1]


def test_broken_prefix_is_rejected_even_when_policy_count_matches() -> None:
    events = _events()
    events[1]["data"]["prompt_token_ids"][0] = 99
    with pytest.raises(ValueError, match=r"prefix|前缀"):
        _data().pair_events(events, sample_id="a", dataset="toy", max_total_length=64)


def test_audit_compares_full_context_not_only_counts_or_targets() -> None:
    data = _data()
    pair = data.pair_events(_events(), sample_id="a", dataset="toy", max_total_length=6)
    original = pair.trajectory[0]
    changed = SampleRow([9, 2], original.response_token_ids, original.response_mask, [0, 1], 0.0, False)
    with pytest.raises(ValueError, match=r"context|上下文"):
        data.audit_pair(data.PairedRollout("a", "toy", pair.transition, [changed]))


def test_audit_compares_target_ids_as_well_as_context() -> None:
    data = _data()
    pair = data.pair_events(_events(), sample_id="a", dataset="toy", max_total_length=6)
    original = pair.trajectory[0]
    changed = SampleRow([1, 2], [10, 11, 90, 99], original.response_mask, [0, 1], 0.0, False)
    with pytest.raises(ValueError, match=r"target|目标"):
        data.audit_pair(data.PairedRollout("a", "toy", pair.transition, [changed]))


@pytest.mark.parametrize(
    "field,bad",
    [
        ("prompt_token_ids", []),
        ("response_token_ids", []),
        ("prompt_token_ids", [-1]),
        ("response_token_ids", [True]),
        ("response_token_ids", [1.0]),
        ("prompt_token_ids", "12"),
    ],
)
def test_invalid_token_data_cannot_be_filtered_or_truncated(field: str, bad: Any) -> None:
    events = _events()
    events[0]["data"][field] = bad
    with pytest.raises(ValueError):
        _data().pair_events(events, sample_id="a", dataset="toy", max_total_length=64)


@pytest.mark.parametrize("change", ["status", "http", "overlength", "no_requests", "empty_event"])
def test_pair_rejects_any_skipped_or_incomplete_row(change: str) -> None:
    events = _events()
    budget = 64
    if change == "status":
        events[0]["data"]["status"] = "error"
    elif change == "http":
        events[0]["data"]["http_status"] = 500
    elif change == "overlength":
        budget = 5
    elif change == "no_requests":
        events = []
    else:
        events = [{}]
    with pytest.raises(ValueError):
        _data().pair_events(events, sample_id="a", dataset="toy", max_total_length=budget)


def test_load_pairs_reaudits_prepared_file_and_retains_metadata(tmp_path: Path) -> None:
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(_prepared()), encoding="utf-8")
    pairs, report = _data().load_pairs(path)
    assert pairs[0].sample_id == "original-id"
    assert pairs[0].policy_tokens == 3
    assert report["selection"]["toy"]["selected"] == 1
    assert report["model"]["revision"] == "c1899de289a04d12100db370d81485cdf75e47ca"
    broken = _prepared()
    broken["rollouts"][0]["events"][1]["data"]["prompt_token_ids"][1] = 99
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match=r"prefix|前缀"):
        _data().load_pairs(path)


@pytest.mark.parametrize("change", ["schema", "budget", "duplicate", "count", "unversioned", "no_rollouts"])
def test_prepared_file_validation_is_not_bypassed(tmp_path: Path, change: str) -> None:
    document = _prepared()
    if change == "schema":
        document["schema_version"] = 2
    elif change == "budget":
        document["settings"]["max_total_length"] = 5
    elif change == "duplicate":
        document["rollouts"].append(copy.deepcopy(document["rollouts"][0]))
    elif change == "count":
        document["selection"]["toy"]["selected"] = 2
    elif change == "unversioned":
        document["model"]["revision"] = "main"
    else:
        document["rollouts"] = []
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        _data().load_pairs(path)


def test_prepare_cli_help_does_not_import_optional_training_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "-S", "-m", "experiments.train_cost.prepare", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--out" in result.stdout


def test_audit_streams_contexts_with_bounded_memory() -> None:
    import tracemalloc

    data = _data()
    row = SampleRow([1] * 2000, [2] * 1000, [1] * 1000, [0], 0.0, False)
    pair = data.PairedRollout("a", "toy", [row], [row])
    tracemalloc.start()
    try:
        data.audit_pair(pair)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Materializing one context tuple per target costs tens of MB for this
    # 3000-token fixture; two streaming iterators need only a few contexts.
    assert peak < 2_000_000


def test_prepare_enforces_offline_mode_for_an_already_imported_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import ModuleType, SimpleNamespace

    prepare = importlib.import_module("experiments.train_cost.prepare")
    hub = ModuleType("huggingface_hub")
    constants = SimpleNamespace(HF_HUB_OFFLINE=False, HF_HUB_DISABLE_IMPLICIT_TOKEN=False)
    hub.constants = constants  # type: ignore[attr-defined]
    hub.hf_hub_download = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN"):
        monkeypatch.setenv(name, "0")
    with pytest.raises(SystemExit) as stopped:
        prepare.main(["--out", str(tmp_path / "unused.json"), "--datasets", "unknown"])
    assert stopped.value.code == 2
    assert constants.HF_HUB_OFFLINE is True
    assert constants.HF_HUB_DISABLE_IMPLICIT_TOKEN is True


def test_stratum_is_reaudited_against_the_final_complete_context(tmp_path: Path) -> None:
    # The fixture's final complete context is 6 tokens (prompt 5 + response 1).
    document = _prepared()
    document["rollouts"][0]["stratum"] = "4-6"
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    pairs, _ = _data().load_pairs(path)
    assert pairs[0].stratum == "4-6"
    assert pairs[0].group == "toy|4-6"
    for bad in ("6-8", "0-5", "x", "6-4", "4-6-8", "04-6", 7):
        document["rollouts"][0]["stratum"] = bad
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match="stratum"):
            _data().load_pairs(path)


def test_group_without_stratum_is_the_dataset(tmp_path: Path) -> None:
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(_prepared()), encoding="utf-8")
    pairs, _ = _data().load_pairs(path)
    assert pairs[0].stratum is None
    assert pairs[0].group == "toy"


def _turns(contexts: list[int]) -> list[dict[str, Any]]:
    """Prefix-consistent events whose complete context after turn k is contexts[k - 1]."""
    events: list[dict[str, Any]] = []
    context: list[int] = []
    for turn, target in enumerate(contexts):
        prompt = context + [7] * (target - 1 - len(context))
        response = [20 + turn]
        events.append(
            {"event_type": "model_request", "data": {"prompt_token_ids": prompt, "response_token_ids": response}}
        )
        context = prompt + response
    return events


def test_stratified_selection_takes_longest_prefix_highest_stratum_first() -> None:
    prepare = importlib.import_module("experiments.train_cost.prepare")
    candidates = [
        ("a", _turns([300, 700, 1500])),
        ("b", _turns([200, 500])),
        ("c", _turns([900, 1200, 2500, 3000])),
        ("d", _turns([100, 150])),
        ("e", _turns([1900, 2100])),  # only a one-turn prefix fits either stratum
    ]
    chosen, decisions = prepare.select_strata(candidates, [(0, 1000), (1000, 2000)], samples=2, min_turns=2)
    assert [(sample_id, len(events)) for sample_id, events in chosen["1000-2000"]] == [("a", 3), ("c", 2)]
    assert [(sample_id, len(events)) for sample_id, events in chosen["0-1000"]] == [("b", 2), ("d", 2)]
    assert chosen["1000-2000"][0][1] == candidates[0][1][:3]
    assert {item["sample_id"] for item in decisions if not item["selected"]} >= {"e"}


@pytest.mark.parametrize(
    "arguments",
    [
        ["--strata", "0-2048,1024-4096"],
        ["--strata", "2048-4096,0-2048"],
        ["--strata", "0-2048", "--cells", "toucan_qwen3|2048-4096"],
        ["--strata", "0-2048", "--cells", "unknown|0-2048"],
        ["--cells", "toucan_qwen3|0-2048"],
        ["--strata", "0-2048", "--min-turns", "0"],
    ],
)
def test_prepare_rejects_invalid_strata_before_loading_anything(tmp_path: Path, arguments: list[str]) -> None:
    prepare = importlib.import_module("experiments.train_cost.prepare")
    with pytest.raises(SystemExit) as stopped:
        prepare.parse_options(["--out", str(tmp_path / "unused.json"), *arguments])
    assert stopped.value.code == 2
    assert not (tmp_path / "unused.json").exists()


def test_prepare_parses_strata_and_cells(tmp_path: Path) -> None:
    prepare = importlib.import_module("experiments.train_cost.prepare")
    options = prepare.parse_options(
        [
            "--out",
            str(tmp_path / "out.json"),
            "--strata",
            "0-2048,2048-4096,4096-6144",
            "--cells",
            "nebius_swe_agent|4096-6144",
            "--samples",
            "6",
        ]
    )
    assert options.strata == [(0, 2048), (2048, 4096), (4096, 6144)]
    assert options.cells == {("nebius_swe_agent", "4096-6144")}
    assert options.min_turns == 2
    pilot = prepare.parse_options(["--out", str(tmp_path / "out.json")])
    assert pilot.strata is None and pilot.cells is None


def test_turns_field_is_reaudited_and_joins_the_group(tmp_path: Path) -> None:
    document = _prepared()  # the fixture rollout has 2 model requests and a final context of 6 tokens
    document["rollouts"][0].update({"stratum": "4-6", "turns": 2})
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    pairs, _ = _data().load_pairs(path)
    assert pairs[0].turns == 2
    assert pairs[0].group == "toy|4-6|k2"
    for bad in (3, 0, "2", 2.0, True):
        document["rollouts"][0]["turns"] = bad
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match="turns"):
            _data().load_pairs(path)


def test_turns_selection_takes_exact_prefixes_highest_turns_first_and_skips_excluded() -> None:
    prepare = importlib.import_module("experiments.train_cost.prepare")
    candidates = [
        ("a", _turns([500, 1500, 2500, 3500])),  # k=4 ends in (3000, 4000]
        ("b", _turns([3200, 3600])),  # k=1 and k=2 both inside; one-turn prefixes are never used
        ("x", _turns([400, 900, 1300, 3100])),  # excluded
        ("c", _turns([1000, 3100, 3300, 3700])),  # k=2, 3 and 4 inside; taken once, at the highest needed k
        ("d", _turns([3050, 3900])),
        ("e", _turns([200, 300, 600, 900])),  # never inside the window
    ]
    chosen, decisions = prepare.select_turns(candidates, (3000, 4000), (2, 4), samples=2, excluded={"x"})
    assert [(sample_id, len(events)) for sample_id, events in chosen[4]] == [("a", 4), ("c", 4)]
    assert [(sample_id, len(events)) for sample_id, events in chosen[2]] == [("b", 2), ("d", 2)]
    assert chosen[4][0][1] == candidates[0][1][:4]
    reasons = {item["sample_id"]: item for item in decisions}
    assert reasons["x"]["selected"] is False and "excluded" in reasons["x"]["reason"]
    assert reasons["e"]["selected"] is False
    assert reasons["a"]["turns"] == 4 and reasons["a"]["context_tokens"] == 3500


def test_extra_candidates_follow_the_seeded_pool_in_shard_order() -> None:
    prepare = importlib.import_module("experiments.train_cost.prepare")
    pool = ["s7", "s2", "s5"]
    shard = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    assert prepare.extend_candidates(pool, shard, extra=2, key=lambda item: item) == ["s7", "s2", "s5", "s1", "s3"]
    assert prepare.extend_candidates(pool, shard, extra=0, key=lambda item: item) == pool
    assert prepare.extend_candidates(pool, shard, extra=99, key=lambda item: item) == [*pool, "s1", "s3", "s4", "s6"]


def test_excluded_rollouts_are_read_from_prepared_files(tmp_path: Path) -> None:
    prepare = importlib.import_module("experiments.train_cost.prepare")
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    rollouts = [{"dataset": "d1", "sample_id": "x"}, {"dataset": "d2", "sample_id": "y"}]
    first.write_text(json.dumps({"rollouts": rollouts}))
    second.write_text(json.dumps({"rollouts": [{"dataset": "d1", "sample_id": "z"}]}))
    assert prepare.read_excluded([first, second]) == {("d1", "x"), ("d2", "y"), ("d1", "z")}


@pytest.mark.parametrize(
    "arguments",
    [
        ["--turns", "2,4"],
        ["--window", "3072-4096"],
        ["--turns", "2,4", "--window", "3072-4096", "--strata", "0-2048"],
        ["--turns", "2,2", "--window", "3072-4096"],
        ["--turns", "0,4", "--window", "3072-4096"],
        ["--turns", "2,25", "--window", "3072-4096"],
        ["--turns", "2,4", "--window", "4096-3072"],
        ["--seed", "-1"],
        ["--extra-candidates", "-5"],
    ],
)
def test_prepare_rejects_invalid_turns_options(
    tmp_path: Path, arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    prepare = importlib.import_module("experiments.train_cost.prepare")
    with pytest.raises(SystemExit) as stopped:
        prepare.parse_options(["--out", str(tmp_path / "unused.json"), *arguments])
    assert stopped.value.code == 2
    assert "unrecognized" not in capsys.readouterr().err


def test_prepare_parses_turns_options(tmp_path: Path) -> None:
    prepare = importlib.import_module("experiments.train_cost.prepare")
    options = prepare.parse_options(
        [
            "--out",
            str(tmp_path / "o.json"),
            "--turns",
            "8,2,6,4",
            "--window",
            "3072-4096",
            "--seed",
            "1",
            "--exclude",
            str(tmp_path / "a.json"),
            "--extra-candidates",
            "1000",
        ]
    )
    assert options.turns == [2, 4, 6, 8] and options.window == (3072, 4096)
    assert options.seed == 1 and options.extra_candidates == 1000 and options.exclude == [tmp_path / "a.json"]
    assert options.max_turns >= 8
    default = prepare.parse_options(["--out", str(tmp_path / "o.json")])
    assert default.turns is None and default.seed == 0 and default.extra_candidates == 0 and default.exclude == []
