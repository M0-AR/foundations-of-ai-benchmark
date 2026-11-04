"""E2: do activation-function differences survive seed noise on real tabular data?"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from fbench.data import load_tabular  # noqa: E402
from fbench.nn import MLPClassifier, cross_entropy  # noqa: E402
from fbench.provenance import env_manifest, sha256_file, utc_now, write_json  # noqa: E402
from fbench.stats import (  # noqa: E402
    cohen_dz,
    holm,
    kendall_tau,
    paired_wilcoxon,
    probability_of_superiority,
)
from fbench.utils import set_seed  # noqa: E402

CFG = ROOT / "config" / "experiment.yaml"


def main() -> dict:
    with open(CFG, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    spec = cfg["e2_activations"]
    datasets = cfg["data"]["tabular_datasets"]
    activations = list(spec["activations"])
    seeds = [int(s) for s in spec["seeds"]]

    raw: list[dict] = []
    per_dataset: dict[str, dict] = {}
    for dataset_name in datasets:
        dataset = load_tabular(dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X,
            dataset.y,
            test_size=float(spec["test_fraction"]),
            random_state=42,
            stratify=dataset.y,
        )
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-12
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        n_classes = int(len(np.unique(dataset.y)))
        print(f"E2 {dataset_name}: train={X_train.shape} test={X_test.shape}")
        for activation in activations:
            for seed in seeds:
                set_seed(seed)
                model = MLPClassifier(
                    n_features=X_train.shape[1],
                    hidden=int(spec["hidden_units"]),
                    n_classes=n_classes,
                    activation=activation,
                    seed=seed,
                )
                start = time.perf_counter()
                model.fit(
                    X_train,
                    y_train,
                    epochs=int(spec["epochs"]),
                    lr=float(spec["learning_rate"]),
                    batch_size=int(spec["batch_size"]),
                    seed=seed,
                )
                elapsed = time.perf_counter() - start
                probs = model.predict_proba(X_test)
                acc = float(np.mean(probs.argmax(axis=1) == y_test))
                ce = cross_entropy(probs, y_test)
                raw.append(
                    {
                        "dataset": dataset_name,
                        "activation": activation,
                        "seed": seed,
                        "test_accuracy": acc,
                        "test_cross_entropy": ce,
                        "train_seconds": elapsed,
                        "n_train": int(len(y_train)),
                        "n_test": int(len(y_test)),
                    }
                )
        per_dataset[dataset_name] = analyse_dataset(dataset_name, activations, seeds, raw)

    # ------------------------------------------------------- cross-dataset view
    overall_pairs = {}
    for dataset in datasets:
        overall_pairs[dataset] = per_dataset[dataset]["pairwise_vs_relu"]
    all_p = {
        f"{dataset}:{pair}": value["p_value"]
        for dataset, pairs in overall_pairs.items()
        for pair, value in pairs.items()
    }
    overall_holm = holm(all_p)

    # H4 speaks of "at least one pair", so the primary family is all ten
    # unordered pairs per dataset (30 tests), Holm-corrected as one family.
    all_pairs: dict[str, dict] = {}
    all_pair_p: dict[str, float] = {}
    for dataset in datasets:
        rows = [r for r in raw if r["dataset"] == dataset]
        for left, right in itertools.combinations(activations, 2):
            va = _seed_values(rows, left)
            vb = _seed_values(rows, right)
            test = paired_wilcoxon(va, vb)
            key = f"{dataset}:{left}_vs_{right}"
            all_pairs[key] = {
                "dataset": dataset,
                "pair": f"{left}_vs_{right}",
                "p_value": test["p_value"],
                "cohen_dz": cohen_dz(va, vb),
                "mean_difference": float(np.mean(np.array(va) - np.array(vb))),
                "probability_of_superiority": probability_of_superiority(va, vb),
                "n": test["n"],
            }
            all_pair_p[key] = test["p_value"]
    all_pair_holm = holm(all_pair_p)
    for key, adjusted in all_pair_holm.items():
        all_pairs[key]["p_holm"] = adjusted["p_holm"]
        all_pairs[key]["reject"] = adjusted["reject"]
    any_significant = any(
        value["p_holm"] < 0.05 and np.isfinite(value["cohen_dz"]) and abs(value["cohen_dz"]) >= 0.5
        for value in all_pairs.values()
    )

    metrics = {
        "raw_runs": raw,
        "per_dataset": per_dataset,
        "overall_pairwise_holm": overall_holm,
        "all_pairs_holm": all_pairs,
        "analysis_note": (
            "Primary H4 family: all 10 unordered activation pairs per dataset, "
            "Holm-corrected across 30 tests. The per-dataset vs-ReLU contrasts "
            "are retained as a secondary view."
        ),
        "max_abs_effect_size": float(
            np.nanmax([abs(value["cohen_dz"]) for value in all_pairs.values()])
        ),
    }
    thresholds = {
        "H4_any_significant_pair_with_medium_effect": bool(any_significant),
    }
    payload = {
        "experiment": "E2_activations",
        "generated_utc": utc_now(),
        "env": env_manifest(),
        "config": spec,
        "inputs": {
            "datasets": datasets,
            "script_sha256": sha256_file(__file__),
        },
        "metrics": metrics,
        "hypothesis_tests": thresholds,
        "all_passed": bool(all(thresholds.values())),
    }
    write_json(ROOT / "results" / "e2_activations.json", payload)
    _write_csv(raw, ROOT / "results" / "e2_activations_runs.csv")
    print(f"E2 pass={payload['all_passed']} effect={metrics['max_abs_effect_size']:.3f}")
    return payload


def analyse_dataset(
    dataset: str, activations: list[str], seeds: list[int], raw: list[dict]
) -> dict:
    rows = [r for r in raw if r["dataset"] == dataset]
    by_activation = {a: [] for a in activations}
    for row in rows:
        by_activation[row["activation"]].append(row)
    summary = {}
    for activation in activations:
        values = sorted(by_activation[activation], key=lambda r: r["seed"])
        acc = np.array([r["test_accuracy"] for r in values])
        ce = np.array([r["test_cross_entropy"] for r in values])
        summary[activation] = {
            "accuracy_mean": float(acc.mean()),
            "accuracy_std": float(acc.std(ddof=1)),
            "accuracy_median": float(np.median(acc)),
            "cross_entropy_mean": float(ce.mean()),
            "cross_entropy_std": float(ce.std(ddof=1)),
            "train_seconds_mean": float(np.mean([r["train_seconds"] for r in values])),
        }
    groups = [
        [r["test_accuracy"] for r in sorted(by_activation[a], key=lambda r: r["seed"])]
        for a in activations
    ]
    friedman = stats.friedmanchisquare(*groups)
    pairwise = {}
    effect_sizes = {}
    reference = "relu"
    for activation in activations:
        if activation == reference:
            continue
        a = [r["test_accuracy"] for r in sorted(by_activation[activation], key=lambda r: r["seed"])]
        b = [r["test_accuracy"] for r in sorted(by_activation[reference], key=lambda r: r["seed"])]
        test = paired_wilcoxon(a, b)
        pairwise[f"{activation}_vs_{reference}"] = test
        effect_sizes[f"{activation}_vs_{reference}"] = {
            "cohen_dz": cohen_dz(a, b),
            "probability_of_superiority": probability_of_superiority(a, b),
            "mean_difference": float(np.mean(np.array(a) - np.array(b))),
        }
    adjusted = holm({k: v["p_value"] for k, v in pairwise.items()})
    for key, value in adjusted.items():
        pairwise[key]["p_holm"] = value["p_holm"]
        pairwise[key]["reject"] = value["reject"]
    rank_stability = _rank_stability(activations, by_activation, seeds)
    return {
        "summary": summary,
        "friedman": {
            "statistic": float(friedman.statistic),
            "p_value": float(friedman.pvalue),
        },
        "pairwise_vs_relu": pairwise,
        "effect_sizes": effect_sizes,
        "rank_stability": rank_stability,
    }


def _seed_values(rows: list[dict], activation: str) -> list[float]:
    ordered = sorted([r for r in rows if r["activation"] == activation], key=lambda r: r["seed"])
    return [r["test_accuracy"] for r in ordered]


def _rank_stability(
    activations: list[str], by_activation: dict[str, list[dict]], seeds: list[int]
) -> dict:
    acc = {
        a: np.array([r["test_accuracy"] for r in sorted(by_activation[a], key=lambda r: r["seed"])])
        for a in activations
    }
    rng = np.random.default_rng(7)
    best_counts = {a: 0 for a in activations}
    tau_values = []
    n_boot = 5000
    for _ in range(n_boot):
        idx = rng.integers(0, len(seeds), size=len(seeds))
        means = {a: float(acc[a][idx].mean()) for a in activations}
        best = max(means, key=means.get)
        best_counts[best] += 1
    for _ in range(2000):
        perm = rng.permutation(len(seeds))
        first, second = perm[: len(seeds) // 2], perm[len(seeds) // 2 :]
        first_rank = [float(acc[a][first].mean()) for a in activations]
        second_rank = [float(acc[a][second].mean()) for a in activations]
        tau_values.append(kendall_tau(first_rank, second_rank))
    return {
        "bootstrap_probability_best": {a: best_counts[a] / n_boot for a in activations},
        "split_half_kendall_tau_mean": float(np.mean(tau_values)),
        "split_half_kendall_tau_p05": float(np.quantile(tau_values, 0.05)),
        "split_half_kendall_tau_p95": float(np.quantile(tau_values, 0.95)),
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    import csv

    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
