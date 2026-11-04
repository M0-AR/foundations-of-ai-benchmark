# Reproducibility guide

## 1. What is pinned

| Item | Mechanism |
|---|---|
| Python | `>=3.11`; the Docker image uses 3.12 |
| Dependencies | `requirements.txt` and `pyproject.toml` |
| Text corpus | SHA256 checked on every E4 run |
| Tabular data | scikit-learn version plus matrix hash in E2 |
| Market data | raw response hash, timestamp, URL, and cached CSV |
| Seeds | fixed in `config/experiment.yaml` |
| Hypotheses | frozen in `config/preregistration.yaml` |
| Results | JSON with environment manifest and git commit when available |
| Manuscript | regenerated from JSON by `scripts/make_report.py` |

## 2. Local workflow

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
python experiments/run_all.py
python scripts/make_report.py
python scripts/check_results.py
```

`run_all.py` executes E1–E6 in order and writes `results/summary.json`.
E4 is the slow step (about 12 minutes on a 28-core CPU because PyTorch uses
multiple threads). E5 requires outbound HTTPS to Yahoo Finance.

## 3. Docker workflow

```bash
docker compose build
docker compose run --rm test
docker compose run --rm bench
docker compose run --rm report
docker compose run --rm check
```

`bench` mounts `results/`, `data/`, and `paper/` so the artifacts survive the
container. `report` mounts the repository root so the generated `README.md`
is written back to the host. If the host is offline, E5 fails; the other
experiments still run when invoked individually.

## 4. Determinism

- Python, NumPy, and PyTorch seeds are set before every stochastic stage.
- E6 retrains the same model twice and compares parameter bytes.
- CPU floating-point results can still differ across BLAS implementations,
  PyTorch versions, thread counts, and CPU architectures.
- Market results are not expected to be byte-identical across retrieval
  dates; the snapshot hash defines the sample.
- The transformer validation batches are fixed, so the BPC endpoint is stable
  for a fixed environment and seed.

## 5. Verifying the artifact

`scripts/check_results.py` checks:

1. All six result files exist and contain environment, metrics, and
   hypothesis-test fields.
2. E2 has exactly 30 pre-registered all-pairs tests.
3. E4 has exactly three seed runs.
4. E5 has exactly five symbols.
5. `summary.json` marks all experiments complete and contains all six.
6. `headline.json` covers six hypotheses.
7. Every file in the E6 manifest exists and matches its SHA256.
8. `README.md` exists and contains no unresolved `{{PLACEHOLDER}}` tokens.

To inspect the manifest manually:

```bash
python - <<'PY'
import json, hashlib
from pathlib import Path
root = Path(".")
manifest = json.loads(Path("results/e6_reproducibility.json").read_text())["metrics"]["manifest"]
for rel, rec in sorted(manifest.items()):
    digest = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    print(("OK " if digest == rec["sha256"] else "BAD"), rel)
PY
```

## 6. Regenerating the manuscript

```bash
python scripts/make_report.py
```

The script reads only `results/*.json`, writes five figures to
`paper/figures/`, writes `results/headline.json`, and renders
`docs/README.template.md` into `README.md` and `results/report.md`. It aborts
if any placeholder remains. Do not edit `README.md` by hand; edit the
template or the result files.

## 7. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `Tiny Shakespeare checksum mismatch` | corrupt or truncated download | delete the local file and rerun |
| `Yahoo fetch failed` | network, rate limit, or provider change | retry later; check `data/raw/market/` |
| `missing result file` in report | `run_all.py` not finished | run the missing experiment |
| `unresolved README placeholders` | template uses an unknown key | add the key to `make_report.py` |
| `manifest hash mismatch` | source or result edited after E6 | rerun E6, then the report |
| E4 much slower than expected | thread contention with other jobs | run it alone or set `OMP_NUM_THREADS` |

## 8. Reporting deviations

Append to the `deviations` list in `config/preregistration.yaml`; never edit a
frozen hypothesis. Rerun E6 after any change so the hash manifest and the
report reflect the final state. The manuscript's Appendix A mirrors the
deviations list.
