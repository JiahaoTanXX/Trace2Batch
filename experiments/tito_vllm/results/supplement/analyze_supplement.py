"""Non-canonical split clustering: TITO-driven (full run plus the SWE-smith fill-in) vs re-render-driven rollouts.

Run from the repository root: python experiments/tito_vllm/results/supplement/analyze_supplement.py
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL = HERE.parent
DATASETS = ("swe_smith_tool", "toucan_qwen3", "nebius_swe_agent")


def load(path: Path) -> dict:
    return json.loads(path.read_text())["rollouts"]


def counts(rollout: dict) -> list[int]:
    """Turns and self-retoken turns before / after the first self-retoken in the trajectory."""
    seen = False
    result = [0, 0, 0, 0]  # before_n, before_k, after_n, after_k
    for turn in rollout["turns"]:
        offset = 2 if seen else 0
        result[offset] += 1
        result[offset + 1] += turn["self_retoken"] is True
        if turn["self_retoken"]:
            seen = True
    return result


def rates(table: dict, keys: list) -> tuple[list[int], float, float]:
    sums = [sum(table[key][j] for key in keys) for j in range(4)]
    before = sums[1] / sums[0] if sums[0] else float("nan")
    after = sums[3] / sums[2] if sums[2] else float("nan")
    return sums, before, after


def late(rollouts: dict, keys: list) -> tuple[int, int]:
    """Self-retoken turns among turns >= 1."""
    n = k = 0
    for key in keys:
        for turn in rollouts[key]["turns"]:
            if turn["turn"] >= 1:
                n += 1
                k += turn["self_retoken"] is True
    return k, n


def merged(rollout: dict, pipeline: str) -> bool:
    return rollout["pipelines"][pipeline]["merge_mismatches"] == 0


def tokens(rollout: dict, pipeline: str) -> int:
    return rollout["pipelines"][pipeline]["total_tokens"]


def analyze(model: str) -> None:
    tito = {}
    for dataset, rollouts in load(FULL / model / "results.json").items():
        for index, rollout in enumerate(rollouts):
            if "failed" not in rollout:
                tito[(dataset, index)] = rollout
    for dataset, rollouts in load(HERE / f"{model}@tito_swe_smith_64_99" / "results.json").items():
        for rollout in rollouts:
            if "failed" not in rollout:
                tito[(dataset, rollout["index"])] = rollout
    rerender = {
        (dataset, rollout["index"]): rollout
        for dataset, rollouts in load(HERE / f"{model}@rerender" / "results.json").items()
        for rollout in rollouts
        if "failed" not in rollout
    }
    common = sorted(set(tito) & set(rerender))
    tito_counts = {key: counts(tito[key]) for key in common}
    rerender_counts = {key: counts(rerender[key]) for key in common}
    sums_t, before_t, after_t = rates(tito_counts, common)
    sums_r, before_r, after_r = rates(rerender_counts, common)
    rng = random.Random(0)
    diffs, diffs_before = [], []
    for _ in range(10000):
        keys = [common[rng.randrange(len(common))] for _ in common]
        _, b1, a1 = rates(tito_counts, keys)
        _, b2, a2 = rates(rerender_counts, keys)
        diffs.append(a1 - a2)
        diffs_before.append(b1 - b2)
    diffs.sort()
    diffs_before.sort()
    ci, ci_before = (diffs[249], diffs[9749]), (diffs_before[249], diffs_before[9749])
    measured = sum(merged(rerender[key], "rerender") for key in common)
    inferred = sum(merged(tito[key], "v2") for key in common)
    tokens_tito = sum(tokens(tito[key], "tito") for key in common)
    n = len(common)
    print(f"== {model}: common trajectories {n} (tito {len(tito)}, rerender {len(rerender)})")
    print(f"   TITO     before {sums_t[1]}/{sums_t[0]}={before_t:.3f}  after {sums_t[3]}/{sums_t[2]}={after_t:.3f}")
    print(f"   rerender before {sums_r[1]}/{sums_r[0]}={before_r:.3f}  after {sums_r[3]}/{sums_r[2]}={after_r:.3f}")
    print(f"   diff(after) TITO-rerender = {after_t - after_r:+.3f}, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"   diff(before) = {before_t - before_r:+.3f}, 95% CI [{ci_before[0]:+.3f}, {ci_before[1]:+.3f}]")
    late_t, late_r = late(tito, common), late(rerender, common)
    print(f"   late-turn self-retoken  TITO {late_t[0]}/{late_t[1]}  rerender {late_r[0]}/{late_r[1]}")
    print(
        f"   fully mergeable: rerender measured {measured}/{n}={measured / n:.3f}; "
        f"v2 inferred {inferred}/{n}={inferred / n:.3f}"
    )
    tokens_rerender = sum(tokens(rerender[key], "rerender") for key in common)
    tokens_v2 = sum(tokens(tito[key], "v2") for key in common)
    print(
        f"   tokens vs TITO: measured rerender {tokens_rerender / tokens_tito:.3f}x, "
        f"inferred v2 {tokens_v2 / tokens_tito:.3f}x"
    )
    per = {}
    for dataset in DATASETS:
        keys = [key for key in common if key[0] == dataset]
        per[dataset] = (
            len(keys),
            sum(merged(rerender[key], "rerender") for key in keys),
            sum(merged(tito[key], "v2") for key in keys),
            round(sum(tokens(rerender[k], "rerender") for k in keys) / sum(tokens(tito[k], "tito") for k in keys), 3),
        )
    print("   per dataset (n, measured, inferred, measured token ratio):", per)
    print(
        "   TITO fully mergeable on all its trajectories:",
        sum(merged(rollout, "tito") for rollout in tito.values()),
        "/",
        len(tito),
    )


if __name__ == "__main__":
    for name in ("qwen2.5-1.5b", "qwen3-0.6b"):
        analyze(name)
