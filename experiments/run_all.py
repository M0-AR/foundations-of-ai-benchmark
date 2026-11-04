"""Run every experiment in order and assemble results/summary.json."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [
    "e1_calculus_autograd.py",
    "e2_activations.py",
    "e3_attention.py",
    "e4_gpt.py",
    "e5_market.py",
    "e6_reproducibility.py",
]
RESULT_FILES = {
    "E1": "e1_calculus_autograd.json",
    "E2": "e2_activations.json",
    "E3": "e3_attention.json",
    "E4": "e4_gpt.json",
    "E5": "e5_market.json",
    "E6": "e6_reproducibility.json",
}


def main() -> int:
    assemble_only = "--assemble-only" in sys.argv[1:]
    timings: dict[str, float] = {}
    if assemble_only:
        path = ROOT / "results" / "summary.json"
        if path.exists():
            timings = json.loads(path.read_text(encoding="utf-8")).get("timings_seconds", {})
    else:
        for script in EXPERIMENTS:
            start = time.perf_counter()
            print(f"\n=== {script} ===", flush=True)
            result = subprocess.run(
                [sys.executable, str(ROOT / "experiments" / script)], check=False
            )
            timings[script] = time.perf_counter() - start
            if result.returncode != 0:
                print(f"FAILED: {script} (exit {result.returncode})", file=sys.stderr)
                return result.returncode
    summary = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": "config/preregistration.yaml",
        "experiments": {},
        "timings_seconds": timings,
    }
    all_hypotheses = True
    for key, filename in RESULT_FILES.items():
        path = ROOT / "results" / filename
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary["experiments"][key] = payload
        all_hypotheses = all_hypotheses and bool(payload.get("all_passed"))
    summary["all_hypotheses_supported"] = bool(all_hypotheses)
    summary["all_experiments_completed"] = True
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAll hypotheses supported: {all_hypotheses}")
    print("All experiments completed: True")
    print(f"Wrote {ROOT / 'results' / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
