"""Rebuild labs/results/summary.json from the raw lab outputs."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "labs" / "results"
LAB_FILES = {
    "lab1": "lab1_falsification.json",
    "lab2": "lab2_locality.json",
    "lab3": "lab3_reproducibility.json",
    "lab4": "lab4_calibration.json",
    "lab5": "lab5_null_market.json",
}


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    raw = {key: load(name) for key, name in LAB_FILES.items()}
    lab1 = raw["lab1"]
    lab4 = raw["lab4"]
    lab4_datasets = {
        name: {
            "accuracy_rank": value["accuracy_rank"],
            "ece_rank": value["ece_rank"],
            "rank_correlation": value["rank_correlation"],
            "friedman_ece": value["friedman_ece"],
        }
        for name, value in lab4["per_dataset"].items()
    }
    labs = {
        "lab1": {
            "killed": lab1["killed"],
            "n_probes": lab1["n_probes"],
            "coverage": lab1["coverage"],
            "numerical_coverage": lab1["numerical_coverage"],
            "protocol_coverage": lab1["protocol_coverage"],
            "survivors": lab1["survivors"],
            "hypothesis_tests": lab1["hypothesis_tests"],
            "hac_envelope": lab1["exploratory_hac_envelope"],
        },
        "lab2": {
            "by_lag": raw["lab2"]["by_lag"],
            "hypothesis_tests": raw["lab2"]["hypothesis_tests"],
        },
        "lab3": {
            "by_workload": raw["lab3"]["by_workload"],
            "hypothesis_tests": raw["lab3"]["hypothesis_tests"],
        },
        "lab4": {
            "hypothesis_tests": lab4["hypothesis_tests"],
            "significant_pairs": lab4["significant_pairs"],
            "closest_holm_p": min(value["p_holm"] for value in lab4["all_pairs_holm"].values()),
            "per_dataset": lab4_datasets,
        },
        "lab5": {
            "aggregate": raw["lab5"]["aggregate"],
            "hypothesis_tests": raw["lab5"]["hypothesis_tests"],
        },
    }
    tests = [value for lab in labs.values() for value in lab["hypothesis_tests"].values()]
    summary = {
        "run_date": date.today().isoformat(),
        "labs": labs,
        "hypothesis_tests_passed": sum(bool(value) for value in tests),
        "hypothesis_tests_total": len(tests),
        "novelty_status": "leads only; no invention claimed without literature review and replication",
    }
    path = RESULTS / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {path.relative_to(ROOT)} "
        f"({summary['hypothesis_tests_passed']}/{summary['hypothesis_tests_total']} tests passed)"
    )


if __name__ == "__main__":
    main()
