"""Lab 3: measure the reproducibility envelope across BLAS/torch thread counts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "labs" / "results" / "lab3_reproducibility.json"
WORKER = ROOT / "labs" / "_repro_worker.py"
THREADS = (1, 2, 4, 8)
REPEATS = 2


def run(workload: str, threads: int, seed: int) -> dict:
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = str(threads)
    env["NUMEXPR_NUM_THREADS"] = str(threads)
    cmd = [
        sys.executable,
        str(WORKER),
        "--workload",
        workload,
        "--threads",
        str(threads),
        "--seed",
        str(seed),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> dict:
    rows = []
    for workload in ("numpy", "torch"):
        for threads in THREADS:
            for repeat in range(REPEATS):
                record = run(workload, threads, seed=3)
                record["repeat"] = repeat
                rows.append(record)
                print(
                    f"{workload} threads={threads} repeat={repeat} "
                    f"loss={record['loss']:.8f} hash={record['param_hash'][:10]}"
                )

    by_workload = {}
    for workload in ("numpy", "torch"):
        subset = [r for r in rows if r["workload"] == workload]
        hashes = {r["param_hash"] for r in subset}
        losses = np.array([r["loss"] for r in subset], dtype=float)
        by_workload[workload] = {
            "unique_hashes": len(hashes),
            "bitwise_stable": bool(len(hashes) == 1),
            "loss_min": float(losses.min()),
            "loss_max": float(losses.max()),
            "loss_range": float(losses.max() - losses.min()),
            "param_norm_range": float(
                max(r["param_norm"] for r in subset) - min(r["param_norm"] for r in subset)
            ),
        }

    payload = {
        "lab": "lab3_reproducibility",
        "config": {"threads": THREADS, "repeats": REPEATS, "seed": 3},
        "rows": rows,
        "by_workload": by_workload,
        "hypothesis_tests": {
            "H9a_any_nonidentical_hash": bool(
                any(not v["bitwise_stable"] for v in by_workload.values())
            ),
            "H9b_loss_range_below_1e_3": bool(
                all(v["loss_range"] < 1e-3 for v in by_workload.values())
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(by_workload, indent=2))
    return payload


if __name__ == "__main__":
    main()
