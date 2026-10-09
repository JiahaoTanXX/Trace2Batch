# Copyright (c) Microsoft. All rights reserved.

"""在本地执行固定合成场景, 核对手算期望并输出确定性 JSON。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trace2batch.core import build_samples


def run_scenarios(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        for mode in ("transition", "trajectory"):
            result = build_samples(
                scenario["events"],
                mode=mode,
                max_prompt_length=scenario["max_prompt_length"],
                max_response_length=scenario["max_response_length"],
            )
            actual = result.summary()
            for key, expected in scenario["expected"][mode].items():
                if actual.get(key) != expected:
                    raise ValueError(f"场景 {scenario['id']} / {mode} 的 {key} 不符合手算期望。")
            expected_reason = scenario.get("expected_reason", {}).get(mode)
            if result.zero_usable_rows_reason != expected_reason:
                raise ValueError(f"场景 {scenario['id']} / {mode} 的无可用行原因不符合期望。")
            results.append(
                {
                    "scenario_id": scenario["id"],
                    "description": scenario["description"],
                    "expectations_passed": True,
                    **result.to_dict(),
                }
            )
    return {
        "schema_version": 1,
        "implementation": "stdlib_cpu_reference",
        "synthetic": True,
        "real_training": False,
        "project_name": "Trace2Batch",
        "scenario_count": len(scenarios),
        "comparison_count": len(results),
        "results": results,
    }


def main() -> None:
    fixture = json.loads(Path(__file__).with_name("scenarios.json").read_text(encoding="utf-8"))
    if fixture.get("synthetic") is not True:
        raise ValueError("固定场景必须明确标记为 synthetic。")
    report = run_scenarios(fixture["scenarios"])
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
