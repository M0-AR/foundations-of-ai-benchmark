"""Lab 4: activation choice under flat accuracy, measured by calibration."""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fbench.data import load_tabular  # noqa: E402
from fbench.nn import MLPClassifier, cross_entropy  # noqa: E402
from fbench.stats import cohen_dz, holm, paired_wilcoxon  # noqa: E402
from fbench.utils import set_seed  # noqa: E402

CFG = ROOT / "config" / "experiment.yaml"
OUT = ROOT / "labs" / "results" / "lab4_calibration.json"


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence >= low) & (confidence < high)
        if high == 1.0:
            mask = (confidence >= low) & (confidence <= high)
        if not np.any(mask):
            continue
        total += float(np.mean(mask)) * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return total


def main() -> dict:
    with open(CFG, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    spec = cfg["e2_activations"]
    datasets = cfg["data"]["tabular_datasets"]
    activations = list(spec["activations"])
    seeds = [int(s) for s in spec["seeds"]]

    raw = []
    for dataset_name in datasets:
        dataset = load_tabular(dataset_name)
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.X,
            dataset.y,
            test_size=float(spec["test_fraction"]),
            random_state=42,
            stratify=dataset.y,
        )
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True) + 1e-12
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        n_classes = int(len(np.unique(dataset.y)))
        print(f"lab4 {dataset_name}: train={x_train.shape} test={x_test.shape}")
        for activation in activations:
            for seed in seeds:
                set_seed(seed)
                model = MLPClassifier(
                    n_features=x_train.shape[1],
                    hidden=int(spec["hidden_units"]),
                    n_classes=n_classes,
                    activation=activation,
                    seed=seed,
                )
                start = time.perf_counter()
                model.fit(
                    x_train,
                    y_train,
                    epochs=int(spec["epochs"]),
                    lr=float(spec["learning_rate"]),
                    batch_size=int(spec["batch_size"]),
                    seed=seed,
                )
                probs = model.predict_proba(x_test)
                raw.append(
                    {
                        "dataset": dataset_name,
                        "activation": activation,
                        "seed": seed,
                        "test_accuracy": float(np.mean(probs.argmax(axis=1) == y_test)),
                        "test_nll": cross_entropy(probs, y_test),
                        "test_ece_15": ece(probs, y_test),
                        "train_seconds": time.perf_counter() - start,
                    }
                )

    per_dataset = {}
    all_pair_p = {}
    all_pairs = {}
    for dataset_name in datasets:
        rows = [r for r in raw if r["dataset"] == dataset_name]
        summary = {}
        for activation in activations:
            values = [r for r in rows if r["activation"] == activation]
            summary[activation] = {
                "accuracy_mean": float(np.mean([r["test_accuracy"] for r in values])),
                "nll_mean": float(np.mean([r["test_nll"] for r in values])),
                "ece_mean": float(np.mean([r["test_ece_15"] for r in values])),
                "ece_std": float(np.std([r["test_ece_15"] for r in values], ddof=1)),
            }
        groups = [
            [
                r["test_ece_15"]
                for r in sorted(rows, key=lambda r: r["seed"])
                if r["activation"] == a
            ]
            for a in activations
        ]
        friedman = stats.friedmanchisquare(*groups)
        for left, right in itertools.combinations(activations, 2):
            a = [
                r["test_ece_15"]
                for r in sorted(rows, key=lambda r: r["seed"])
                if r["activation"] == left
            ]
            b = [
                r["test_ece_15"]
                for r in sorted(rows, key=lambda r: r["seed"])
                if r["activation"] == right
            ]
            test = paired_wilcoxon(a, b)
            key = f"{dataset_name}:{left}_vs_{right}"
            all_pairs[key] = {
                "dataset": dataset_name,
                "pair": f"{left}_vs_{right}",
                "p_value": test["p_value"],
                "cohen_dz": cohen_dz(a, b),
                "mean_difference": float(np.mean(np.array(a) - np.array(b))),
                "n": test["n"],
            }
            all_pair_p[key] = test["p_value"]
        accuracy_rank = sorted(activations, key=lambda a: summary[a]["accuracy_mean"], reverse=True)
        ece_rank = sorted(activations, key=lambda a: summary[a]["ece_mean"])
        per_dataset[dataset_name] = {
            "summary": summary,
            "friedman_ece": {
                "statistic": float(friedman.statistic),
                "p_value": float(friedman.pvalue),
            },
            "accuracy_rank": accuracy_rank,
            "ece_rank": ece_rank,
            "rank_correlation": float(
                stats.spearmanr(
                    [accuracy_rank.index(a) for a in activations],
                    [ece_rank.index(a) for a in activations],
                ).statistic
            ),
        }

    adjusted = holm(all_pair_p)
    for key, value in adjusted.items():
        all_pairs[key]["p_holm"] = value["p_holm"]
        all_pairs[key]["reject"] = value["reject"]
    any_significant = any(v["reject"] for v in all_pairs.values())
    rank_changed = any(
        per_dataset[d]["accuracy_rank"] != per_dataset[d]["ece_rank"] for d in datasets
    )
    payload = {
        "lab": "lab4_calibration",
        "config": spec,
        "raw": raw,
        "per_dataset": per_dataset,
        "all_pairs_holm": all_pairs,
        "hypothesis_tests": {
            "H10a_any_ece_pair_holm_significant": bool(any_significant),
            "H10b_accuracy_rank_differs_from_ece_rank": bool(rank_changed),
        },
        "significant_pairs": [k for k, v in all_pairs.items() if v["reject"]],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["hypothesis_tests"], indent=2))
    print(json.dumps({d: per_dataset[d]["summary"] for d in datasets}, indent=2))
    return payload


if __name__ == "__main__":
    main()
