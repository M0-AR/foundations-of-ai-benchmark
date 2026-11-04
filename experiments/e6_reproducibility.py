"""E6: seed-selection instability and byte-level reproducibility audit."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from fbench.autograd import expression  # noqa: E402
from fbench.data import load_tabular  # noqa: E402
from fbench.nn import MLPClassifier  # noqa: E402
from fbench.provenance import DATA, env_manifest, sha256_file, utc_now, write_json  # noqa: E402
from fbench.stats import kendall_tau  # noqa: E402
from fbench.utils import set_seed  # noqa: E402

CFG = ROOT / "config" / "experiment.yaml"


def main() -> dict:
    with open(CFG, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    e2_path = ROOT / "results" / "e2_activations.json"
    if not e2_path.exists():
        raise FileNotFoundError("E6 requires E2 results; run experiments/e2_activations.py first")
    e2 = json.loads(e2_path.read_text(encoding="utf-8"))
    datasets = cfg["data"]["tabular_datasets"]
    activations = list(cfg["e2_activations"]["activations"])
    seeds = [int(s) for s in cfg["e2_activations"]["seeds"]]

    stability = {}
    for dataset in datasets:
        raw = [r for r in e2["metrics"]["raw_runs"] if r["dataset"] == dataset]
        by_activation = {
            a: sorted([r for r in raw if r["activation"] == a], key=lambda r: r["seed"])
            for a in activations
        }
        split = len(seeds) // 2
        first = seeds[:split]
        second = seeds[split:]
        rank_first = {
            a: float(np.mean([r["test_accuracy"] for r in by_activation[a] if r["seed"] in first]))
            for a in activations
        }
        rank_second = {
            a: float(np.mean([r["test_accuracy"] for r in by_activation[a] if r["seed"] in second]))
            for a in activations
        }
        order_first = sorted(activations, key=lambda a: rank_first[a], reverse=True)
        order_second = sorted(activations, key=lambda a: rank_second[a], reverse=True)
        best_first = order_first[0]
        best_second = order_second[0]
        # regret: accuracy lost on the second half by choosing on the first half
        regret = rank_second[best_second] - rank_second[best_first]
        stability[dataset] = {
            "first_half_seeds": first,
            "second_half_seeds": second,
            "ranking_first_half": order_first,
            "ranking_second_half": order_second,
            "kendall_tau": kendall_tau(
                [rank_first[a] for a in activations], [rank_second[a] for a in activations]
            ),
            "selected_on_first_half": best_first,
            "best_on_second_half": best_second,
            "selection_regret_accuracy": float(regret),
            "activation_mean_accuracy": {"first_half": rank_first, "second_half": rank_second},
        }

    # seed effects across datasets: does a lucky seed transfer?
    cross_dataset = {}
    for i, d1 in enumerate(datasets):
        for d2 in datasets[i + 1 :]:
            taus = []
            for activation in activations:
                a = _seed_accuracies(e2["metrics"]["raw_runs"], d1, activation)
                b = _seed_accuracies(e2["metrics"]["raw_runs"], d2, activation)
                if len(a) == len(b) and len(a) > 2:
                    taus.append(kendall_tau(a, b))
            cross_dataset[f"{d1}__{d2}"] = {
                "mean_kendall_tau": float(np.mean(taus)) if taus else float("nan"),
                "per_activation": taus,
            }

    # ------------------------------------------------ deterministic rerun check
    deterministic = _determinism_check(cfg)

    # ---------------------------------------------------------- file manifest
    manifest = _build_manifest()
    payload = {
        "experiment": "E6_reproducibility",
        "generated_utc": utc_now(),
        "env": env_manifest(),
        "inputs": {
            "e2_sha256": sha256_file(e2_path),
            "script_sha256": sha256_file(__file__),
        },
        "metrics": {
            "seed_stability": stability,
            "cross_dataset_seed_transfer": cross_dataset,
            "deterministic_rerun": deterministic,
            "manifest": manifest,
            "config_sha256": sha256_file(CFG),
            "preregistration_sha256": sha256_file(ROOT / "config" / "preregistration.yaml"),
        },
        "hypothesis_tests": {
            "deterministic_rerun_byte_identical": deterministic["byte_identical"],
        },
        "all_passed": bool(deterministic["byte_identical"]),
    }
    write_json(ROOT / "results" / "e6_reproducibility.json", payload)
    tau = np.mean([v["kendall_tau"] for v in stability.values()])
    print(
        f"E6: mean split-half tau={tau:.3f} "
        f"deterministic={deterministic['byte_identical']} pass={payload['all_passed']}"
    )
    return payload


def _seed_accuracies(raw: list[dict], dataset: str, activation: str) -> list[float]:
    rows = [r for r in raw if r["dataset"] == dataset and r["activation"] == activation]
    return [r["test_accuracy"] for r in sorted(rows, key=lambda r: r["seed"])]


def _determinism_check(cfg: dict) -> dict:
    dataset = load_tabular("breast_cancer")
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, dataset.y, test_size=0.3, random_state=42, stratify=dataset.y
    )
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-12
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    hashes = []
    accuracies = []
    for _ in range(2):
        set_seed(0)
        model = MLPClassifier(X_train.shape[1], 64, 2, activation="tanh", seed=0)
        model.fit(X_train, y_train, epochs=30, lr=0.01, batch_size=128, seed=0)
        acc = float(np.mean(model.predict(X_test) == y_test))
        hashes.append(hashlib.sha256(model.get_flat_params().tobytes()).hexdigest())
        accuracies.append(acc)
    set_seed(0)
    out1 = expression(0.3, -0.7)
    out1.backward()
    set_seed(0)
    out2 = expression(0.3, -0.7)
    out2.backward()
    autograd_identical = out1.data == out2.data and out1.grad == out2.grad
    return {
        "parameter_hashes": hashes,
        "accuracies": accuracies,
        "byte_identical": hashes[0] == hashes[1],
        "accuracy_identical": accuracies[0] == accuracies[1],
        "autograd_identical": bool(autograd_identical),
    }


def _build_manifest() -> dict:
    excluded_results = {"e6_reproducibility.json", "summary.json", "headline.json"}
    files = sorted(
        [p for p in (ROOT / "results").glob("*.json") if p.name not in excluded_results]
        + list((ROOT / "config").glob("*.yaml"))
        + list((ROOT / "src" / "fbench").glob("*.py"))
        + list((ROOT / "experiments").glob("*.py"))
        + list((ROOT / "scripts").glob("*.py"))
        + list((ROOT / "tests").glob("*.py"))
        + list((ROOT / "labs").glob("*.py"))
        + list((ROOT / "labs").glob("*.md"))
        + list((ROOT / "labs" / "results").glob("*.json"))
        + list((DATA / "public").glob("*"))
    )
    return {
        str(p.relative_to(ROOT)): {
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }
        for p in files
        if p.is_file()
    }


if __name__ == "__main__":
    main()
