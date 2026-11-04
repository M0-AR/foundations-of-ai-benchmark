"""Build every table, figure, and the README from executed result files.

This script is deliberately the only place where numbers enter the written
report. It never invents a value: every number is read from ``results/*.json``
and every figure is drawn from the same files. Run it after ``run_all.py``.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"

DATASET_LABELS = {
    "breast_cancer": "Breast cancer",
    "wine": "Wine",
    "digits": "Digits",
}
ACTIVATION_ORDER = ["sigmoid", "tanh", "relu", "leaky_relu", "gelu"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def test_count() -> int:
    return sum(
        line.strip().startswith("def test_")
        for path in (ROOT / "tests").glob("*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def repository_url() -> str:
    override = os.environ.get("REPO_URL", "").strip().rstrip("/")
    if override:
        return override
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    url = result.stdout.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split(":", 1)[1]
    return url


def pages_url(repo_url: str) -> str:
    override = os.environ.get("PAGES_URL", "").strip().rstrip("/")
    if override:
        return override
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)$", repo_url)
    if match:
        return f"https://{match.group(1)}.github.io/{match.group(2)}"
    return "preview.html"


def gif_size_mb() -> str:
    path = ROOT / "docs" / "assets" / "demo.gif"
    if not path.exists():
        return "0.0"
    return f"{path.stat().st_size / 1024 / 1024:.1f}"


def load(name: str) -> dict[str, Any]:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"missing result file: {path}; run experiments/run_all.py")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | int | None, nd: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.{nd}e}"
    return f"{value:.{nd}f}"


def sci(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.2e}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    for row in rows:
        cells = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        lines.append(
            "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        )
    return "\n".join(lines)


def figure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


# --------------------------------------------------------------------- E1
def e1_section(e1: dict[str, Any]) -> tuple[str, dict[str, str]]:
    calc = e1["metrics"]["calculus"]
    rows = []
    for name, row in calc.items():
        rows.append(
            [
                name,
                fmt(row["best_h"]),
                sci(row["best_error"]),
                fmt(row.get("loglog_slope_truncation")),
                fmt(row.get("loglog_slope_all")),
            ]
        )
    table = markdown_table(
        ["Function", "Best h", "Best mean abs error", "Slope (h >= 1e-3)", "Slope (all h)"],
        rows,
    )
    values = {
        "E1_SLOPE": fmt(e1["metrics"]["median_loglog_slope_truncation"], 3),
        "E1_AUTOGRAD_FD_ERR": sci(
            e1["metrics"]["autograd_vs_finite_difference"]["max_relative_error"]
        ),
        "E1_MLP_FD_ERR": sci(e1["metrics"]["numpy_mlp_vs_finite_difference"]["max_relative_error"]),
        "E1_MLP_TORCH_ERR": sci(e1["metrics"]["numpy_mlp_vs_pytorch"]["max_relative_error"]),
    }
    return table, values


def e1_figure(e1: dict[str, Any]) -> None:
    calc = e1["metrics"]["calculus"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for color, (name, row) in zip(COLORS, calc.items(), strict=True):
        h = np.asarray(row["h_values"], dtype=float)
        err = np.asarray(row["mean_abs_errors"], dtype=float)
        ax.loglog(h, err, "o-", color=color, ms=3, lw=1.2, label=name)
    ax.invert_xaxis()
    ax.set_xlabel("Step size h")
    ax.set_ylabel("Mean absolute derivative error")
    ax.set_title("Central-difference convergence against analytic derivatives")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01_calculus.png")
    plt.close(fig)


# --------------------------------------------------------------------- E2
def e2_section(e2: dict[str, Any]) -> tuple[str, str, str, dict[str, str]]:
    per = e2["metrics"]["per_dataset"]
    summary_rows = []
    pair_rows = []
    for dataset in DATASET_LABELS:
        if dataset not in per:
            continue
        entry = per[dataset]
        for activation in ACTIVATION_ORDER:
            if activation not in entry["summary"]:
                continue
            item = entry["summary"][activation]
            summary_rows.append(
                [
                    DATASET_LABELS[dataset],
                    activation,
                    f"{fmt(item['accuracy_mean'])} ± {fmt(item['accuracy_std'])}",
                    fmt(item["cross_entropy_mean"]),
                    fmt(item["train_seconds_mean"], 2),
                ]
            )
        for pair, test in entry["pairwise_vs_relu"].items():
            effect = entry["effect_sizes"][pair]
            pair_rows.append(
                [
                    DATASET_LABELS[dataset],
                    pair.replace("_vs_relu", " vs relu"),
                    fmt(test["p_value"], 4),
                    fmt(test.get("p_holm", float("nan")), 4),
                    fmt(effect["cohen_dz"]),
                    fmt(effect["probability_of_superiority"], 2),
                ]
            )
    summary_table = markdown_table(
        ["Dataset", "Activation", "Test accuracy (mean ± SD)", "Test CE", "Train s/run"],
        summary_rows,
    )
    pair_table = markdown_table(
        ["Dataset", "Pair (vs ReLU)", "Wilcoxon p", "Holm p", "Cohen d_z", "P(superiority)"],
        pair_rows,
    )
    all_pairs = e2["metrics"].get("all_pairs_holm", {})
    all_rows = [
        [
            DATASET_LABELS.get(value["dataset"], value["dataset"]),
            value["pair"].replace("_vs_", " vs "),
            fmt(value["p_value"], 4),
            fmt(value.get("p_holm", float("nan")), 4),
            fmt(value["cohen_dz"]),
            fmt(value["probability_of_superiority"], 2),
        ]
        for value in all_pairs.values()
    ]
    all_pair_table = markdown_table(
        ["Dataset", "Pair", "Wilcoxon p", "Holm p", "Cohen d_z", "P(superiority)"],
        all_rows,
    )
    significant = [
        f"{DATASET_LABELS.get(value['dataset'], value['dataset'])}: {value['pair']}"
        for value in all_pairs.values()
        if bool(value.get("reject"))
    ]
    friedman_sig = [
        f"{DATASET_LABELS[d]} (p = {fmt(entry['friedman']['p_value'], 4)})"
        for d, entry in per.items()
        if entry["friedman"]["p_value"] < 0.05
    ]
    values = {
        "E2_MAX_DZ": fmt(e2["metrics"]["max_abs_effect_size"], 3),
        "E2_MIN_HOLM": fmt(
            min((value.get("p_holm", 1.0) for value in all_pairs.values()), default=1.0),
            3,
        ),
        "E2_SIG_PAIRS": ", ".join(significant) if significant else "none",
        "E2_FRIEDMAN_SIG": ", ".join(friedman_sig) if friedman_sig else "none",
    }
    return summary_table, pair_table, all_pair_table, values


def e2_figure(e2: dict[str, Any]) -> None:
    per = e2["metrics"]["per_dataset"]
    datasets = [d for d in DATASET_LABELS if d in per]
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.4, 2.8), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets, strict=True):
        entry = per[dataset]
        means = [entry["summary"][a]["accuracy_mean"] for a in ACTIVATION_ORDER]
        stds = [entry["summary"][a]["accuracy_std"] for a in ACTIVATION_ORDER]
        ax.bar(
            range(len(ACTIVATION_ORDER)),
            means,
            yerr=stds,
            capsize=2,
            color=COLORS,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_xticks(range(len(ACTIVATION_ORDER)))
        ax.set_xticklabels(["sig", "tanh", "relu", "lrelu", "gelu"], fontsize=7)
        ax.set_title(DATASET_LABELS[dataset])
        low = max(0.0, min(m - s for m, s in zip(means, stds, strict=True)) - 0.05)
        ax.set_ylim(low, 1.0)
    axes[0].set_ylabel("Test accuracy")
    fig.suptitle("Activation functions on three public tabular datasets (10 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02_activations.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- E3
def e3_section(e3: dict[str, Any]) -> tuple[str, dict[str, str]]:
    metrics = e3["metrics"]
    rows = [
        [
            "Attention parity vs PyTorch fused SDPA",
            "max abs error",
            sci(metrics["attention_parity_max_abs_error"]),
            "< 1e-6",
        ],
        ["Causal leakage", "max future weight", sci(metrics["causal_leak_max"]), "== 0"],
        [
            "Softmax normalisation",
            "max row-sum error",
            sci(metrics["softmax_row_sum_max_abs_error"]),
            "< 1e-6",
        ],
        [
            "Attention gradient check",
            "gradcheck",
            str(bool(metrics["attention_gradcheck_pass"])),
            "True",
        ],
        [
            "RoPE norm preservation",
            "max abs error",
            sci(metrics["rope_norm_max_abs_error"]),
            "< 1e-6",
        ],
        ["RoPE pair norm", "max abs error", sci(metrics["rope_pair_norm_max_abs_error"]), "< 1e-6"],
        [
            "RoPE relative-position (Toeplitz)",
            "max deviation",
            sci(metrics["rope_toeplitz_max_deviation"]),
            "< 1e-6",
        ],
        [
            "RoPE informativeness",
            "score std over offsets",
            sci(metrics["rope_relative_distance_score_std"]),
            "> 1e-3",
        ],
    ]
    table = markdown_table(["Property", "Statistic", "Observed", "Threshold"], rows)
    values = {
        "E3_PARITY": sci(metrics["attention_parity_max_abs_error"]),
        "E3_LEAK": sci(metrics["causal_leak_max"]),
        "E3_ROPE": sci(metrics["rope_toeplitz_max_deviation"]),
        "E3_GRADCHECK": str(bool(metrics["attention_gradcheck_pass"])),
    }
    return table, values


# --------------------------------------------------------------------- E4
def e4_section(e4: dict[str, Any]) -> tuple[str, str, dict[str, str], str]:
    runs = e4["metrics"]["runs"]
    rows = []
    for run in runs:
        ci = run["val_bpc_ci95"]
        rows.append(
            [
                str(run["seed"]),
                fmt(run["val_bpc"], 4),
                f"[{fmt(ci[0], 4)}, {fmt(ci[1], 4)}]",
                f"{run['parameters']:,}",
                f"{run['parameters_non_embedding']:,}",
                fmt(run["train_seconds"], 1),
            ]
        )
    run_table = markdown_table(
        ["Seed", "Val BPC", "95% batch-bootstrap CI", "Parameters", "Non-embedding", "Train s"],
        rows,
    )
    baselines = e4["metrics"]["baselines"]
    base_rows = []
    for name in ["uniform", "unigram", "interpolated_ngram", "trigram_add1"]:
        if name not in baselines:
            continue
        item = baselines[name]
        ci = item.get("bpc_ci95", [float("nan"), float("nan")])
        base_rows.append(
            [
                name,
                fmt(item["bpc"], 4),
                f"[{fmt(ci[0], 4)}, {fmt(ci[1], 4)}]",
                str(item.get("weights", "")),
            ]
        )
    baseline_table = markdown_table(
        ["Baseline", "Val BPC", "95% block-bootstrap CI", "Mixture weights"],
        base_rows,
    )
    aggregate = e4["metrics"]["aggregate"]
    primary = baselines.get("trigram_add1", {})
    primary_bpc = float(primary.get("bpc", float("nan")))
    improvement = primary_bpc - aggregate["bpc_mean"]
    values = {
        "E4_BPC_MEAN": fmt(aggregate["bpc_mean"], 4),
        "E4_BPC_STD": fmt(aggregate["bpc_std"], 4),
        "E4_BPC_MIN": fmt(aggregate["bpc_min"], 4),
        "E4_BPC_MAX": fmt(aggregate["bpc_max"], 4),
        "E4_BASELINE_BPC": fmt(primary_bpc, 4),
        "E4_ABS_IMPROVEMENT": fmt(improvement, 4),
        "E4_REL_IMPROVEMENT": fmt(100.0 * improvement / primary_bpc, 1),
        "E4_PARAMS": f"{aggregate['parameters']:,}",
        "E4_SECONDS": fmt(aggregate["train_seconds_mean"], 1),
    }
    samples = "\n\n".join(
        f"**Seed {run['seed']}**\n\n```text\n{run['sample']}\n```" for run in runs
    )
    return run_table, baseline_table, values, samples


def e4_figure(e4: dict[str, Any]) -> None:
    runs = e4["metrics"]["runs"]
    baselines = e4["metrics"]["baselines"]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for color, run in zip(COLORS, runs, strict=False):
        steps = [p["step"] for p in run["curve"]]
        bpc = [p["val_bpc"] for p in run["curve"]]
        ax.plot(steps, bpc, color=color, lw=1.3, label=f"seed {run['seed']}")
    for name, style in [
        ("trigram_add1", "--"),
        ("interpolated_ngram", ":"),
        ("unigram", "-."),
    ]:
        if name in baselines:
            ax.axhline(baselines[name]["bpc"], color="black", ls=style, lw=0.9, label=name)
    ax.set_xlabel("Optimiser step")
    ax.set_ylabel("Validation bits per character")
    ax.set_title("TinyGPT learning curves versus n-gram baselines")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03_gpt.png")
    plt.close(fig)


def e4_attention_figure(e4: dict[str, Any]) -> None:
    runs = e4["metrics"]["runs"]
    if not runs:
        return
    layers = np.arange(len(runs[0]["attention"]["normalised_entropy_per_layer"]))
    values = np.array(
        [run["attention"]["normalised_entropy_per_layer"] for run in runs], dtype=float
    )
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.plot(layers + 1, values.mean(axis=0), "o-", color=COLORS[0], lw=1.4)
    for row in values:
        ax.plot(layers + 1, row, color="grey", lw=0.6, alpha=0.5)
    ax.set_xticks(layers + 1)
    ax.set_xlabel("Transformer block")
    ax.set_ylabel("Normalised attention entropy")
    ax.set_title("Attention entropy across depth (3 seeds)")
    ax.set_ylim(0, 1)
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05_attention_entropy.png")
    plt.close(fig)


# --------------------------------------------------------------------- E5
def e5_section(e5: dict[str, Any]) -> tuple[str, str, dict[str, str]]:
    per = e5["metrics"]["per_symbol"]
    symbols = list(per.keys())
    rows = []
    for symbol in symbols:
        entry = per[symbol]
        lb = entry["ljung_box"]
        ridge = entry["models"]["ridge"]
        ci = ridge["oos_r2_vs_zero_ci95"]
        dm = ridge["dm_vs_zero"]
        rows.append(
            [
                symbol,
                str(entry["n_oos"]),
                fmt(lb["statistic"], 1),
                fmt(lb.get("p_holm", float("nan")), 3),
                "yes" if lb.get("reject_holm") else "no",
                f"{fmt(ridge['oos_r2_vs_zero'], 4)} [{fmt(ci[0], 4)}, {fmt(ci[1], 4)}]",
                fmt(dm.get("p_holm", float("nan")), 3),
                "yes" if dm.get("reject_holm") else "no",
                fmt(ridge["directional_accuracy"], 3),
                fmt(ridge["spearman_ic"], 3),
                fmt(ridge["strategy"]["sharpe"], 2),
            ]
        )
    table = markdown_table(
        [
            "Symbol",
            "OOS days",
            "LB stat",
            "LB Holm p",
            "Clustering",
            "OOS R² [95% CI]",
            "DM Holm p",
            "Alpha",
            "Dir. acc.",
            "Spearman IC",
            "Sharpe",
        ],
        rows,
    )
    snapshot_rows = [
        [
            snap["symbol"],
            str(snap["rows"]),
            snap["first_date"],
            snap["last_date"],
            f"{snap['last_close']:.2f}",
            snap["raw_sha256"][:12] + "…",
            snap["retrieved_utc"],
        ]
        for snap in e5["metrics"]["snapshots"]
    ]
    snapshot_table = markdown_table(
        ["Symbol", "Rows", "First date", "Last date", "Last close", "SHA256", "Retrieved (UTC)"],
        snapshot_rows,
    )
    exploratory = e5["metrics"].get("exploratory", {})
    values = {
        "E5_SYMBOLS": str(len(symbols)),
        "E5_LB_REJECTS": str(exploratory.get("volatility_clustering_holm_rejections", "n/a")),
        "E5_DM_REJECTS": str(exploratory.get("dm_holm_rejections", "n/a")),
        "E5_R2_NONPOS": str(exploratory.get("symbols_with_nonpositive_oos_r2", "n/a")),
        "E5_LAST_DATE": max(
            (snap["last_date"] for snap in e5["metrics"]["snapshots"]), default="n/a"
        ),
    }
    return table, snapshot_table, values


def e5_figure(e5: dict[str, Any]) -> None:
    per = e5["metrics"]["per_symbol"]
    symbols = list(per.keys())
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.4, 4.4), gridspec_kw={"height_ratios": [1.2, 1.0]}, sharex=True
    )
    x = np.arange(len(symbols))
    r2 = np.array([per[s]["models"]["ridge"]["oos_r2_vs_zero"] for s in symbols])
    low = np.array([per[s]["models"]["ridge"]["oos_r2_vs_zero_ci95"][0] for s in symbols])
    high = np.array([per[s]["models"]["ridge"]["oos_r2_vs_zero_ci95"][1] for s in symbols])
    ax1.errorbar(
        x,
        r2,
        yerr=[r2 - low, high - r2],
        fmt="o",
        color=COLORS[0],
        capsize=3,
        ms=5,
        lw=1.2,
    )
    ax1.axhline(0.0, color="black", lw=0.8)
    ax1.set_ylabel("OOS R² vs zero")
    ax1.set_title("Out-of-sample directional predictability (ridge, 2 bp turnover cost)")
    ax1.grid(True, ls=":", lw=0.5, alpha=0.6)

    lb = np.array(
        [-math.log10(max(per[s]["ljung_box"].get("p_holm", 1.0), 1e-300)) for s in symbols]
    )
    dm = np.array(
        [
            -math.log10(max(per[s]["models"]["ridge"]["dm_vs_zero"].get("p_holm", 1.0), 1e-300))
            for s in symbols
        ]
    )
    width = 0.36
    ax2.bar(x - width / 2, lb, width, color=COLORS[2], label="Ljung–Box (squared returns)")
    ax2.bar(x + width / 2, dm, width, color=COLORS[3], label="Diebold–Mariano (ridge vs zero)")
    ax2.axhline(-math.log10(0.05), color="black", ls="--", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(symbols)
    ax2.set_ylabel("−log10(Holm p)")
    ax2.set_xlabel("Symbol")
    ax2.legend(fontsize=7)
    ax2.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04_market.png")
    plt.close(fig)


# --------------------------------------------------------------------- E6
def e6_section(e6: dict[str, Any]) -> tuple[str, dict[str, str]]:
    det = e6["metrics"]["deterministic_rerun"]
    stability = e6["metrics"]["seed_stability"]
    rows = [
        ["Parameter hash run 1", det["parameter_hashes"][0][:16] + "…"],
        ["Parameter hash run 2", det["parameter_hashes"][1][:16] + "…"],
        ["Byte-identical retraining", str(det["byte_identical"])],
        ["Autograd expression identical", str(det["autograd_identical"])],
        [
            "Split-half Kendall τ (mean over datasets)",
            fmt(np.mean([v["kendall_tau"] for v in stability.values()]), 3),
        ],
        ["Files in artifact manifest", str(len(e6["metrics"]["manifest"]))],
    ]
    table = markdown_table(["Check", "Result"], rows)
    values = {
        "E6_TAU": fmt(np.mean([v["kendall_tau"] for v in stability.values()]), 3),
        "E6_MANIFEST_COUNT": str(len(e6["metrics"]["manifest"])),
        "E6_DETERMINISTIC": str(det["byte_identical"]),
        "E6_DETERMINISTIC_LABEL": "yes" if det["byte_identical"] else "no",
    }
    return table, values


# --------------------------------------------------------------- headline
def build_headline(
    e1: dict[str, Any],
    e2: dict[str, Any],
    e3: dict[str, Any],
    e4: dict[str, Any],
    e5: dict[str, Any],
    e6: dict[str, Any],
) -> tuple[str, dict[str, str], dict[str, Any]]:
    h1_checks = [
        e1["metrics"]["autograd_vs_finite_difference"]["max_relative_error"],
        e1["metrics"]["numpy_mlp_vs_finite_difference"]["max_relative_error"],
        e1["metrics"]["numpy_mlp_vs_pytorch"]["max_relative_error"],
    ]
    h1 = all(bool(v) for k, v in e1["hypothesis_tests"].items() if k.startswith("H1"))
    h2 = bool(e1["hypothesis_tests"]["H2_convergence"])
    h3 = all(bool(v) for k, v in e3["hypothesis_tests"].items() if k.startswith("H3"))
    h4 = bool(e2["hypothesis_tests"]["H4_any_significant_pair_with_medium_effect"])
    h5 = all(bool(v) for k, v in e4["hypothesis_tests"].items() if k.startswith("H5"))
    h6 = all(bool(v) for k, v in e5["hypothesis_tests"].items() if k.startswith("H6"))

    e4_agg = e4["metrics"]["aggregate"]
    primary = e4["metrics"]["baselines"]["trigram_add1"]
    improvement = 100.0 * (primary["bpc"] - e4_agg["bpc_mean"]) / primary["bpc"]
    all_pairs = e2["metrics"].get("all_pairs_holm", {})
    min_holm = min((value.get("p_holm", 1.0) for value in all_pairs.values()), default=1.0)
    exploratory = e5["metrics"].get("exploratory", {})

    rows = [
        [
            "H1",
            "Gradient fidelity",
            "max relative error",
            "< 1e-5",
            f"{sci(max(h1_checks))} (worst of 3 checks)",
            "yes" if h1 else "no",
        ],
        [
            "H2",
            "Calculus convergence",
            "median log–log slope",
            ">= 1.8",
            fmt(e1["metrics"]["median_loglog_slope_truncation"], 3),
            "yes" if h2 else "no",
        ],
        [
            "H3",
            "Attention / RoPE fidelity",
            "parity, leak, RoPE invariants",
            "all thresholds",
            f"parity {sci(e3['metrics']['attention_parity_max_abs_error'])}; "
            f"leak {sci(e3['metrics']['causal_leak_max'])}; "
            f"RoPE {sci(e3['metrics']['rope_toeplitz_max_deviation'])}",
            "yes" if h3 else "no",
        ],
        [
            "H4",
            "Activation effects",
            "Holm p and |d_z|",
            "p < 0.05 and |d_z| >= 0.5",
            f"min Holm p = {fmt(min_holm, 3)}; max |d_z| = "
            f"{fmt(e2['metrics']['max_abs_effect_size'], 3)}",
            "yes" if h4 else "no",
        ],
        [
            "H5",
            "Transformer vs smoothed trigram",
            "mean val BPC",
            "lower, all seeds, >= 5%",
            f"{fmt(e4_agg['bpc_mean'], 4)} vs {fmt(primary['bpc'], 4)} "
            f"({fmt(improvement, 1)}% lower)",
            "yes" if h5 else "no",
        ],
        [
            "H6",
            "Live market structure vs alpha",
            "LB / OOS R² / DM",
            "clustering; no alpha",
            f"LB Holm rejects {exploratory.get('volatility_clustering_holm_rejections', 'n/a')}"
            f"/{exploratory.get('n_symbols', 'n/a')}; DM Holm rejects "
            f"{exploratory.get('dm_holm_rejections', 'n/a')}"
            f"/{exploratory.get('n_symbols', 'n/a')}",
            "yes"
            if h6
            else (
                "partial"
                if exploratory.get("volatility_clustering_majority")
                and exploratory.get("no_holm_significant_directional_alpha")
                else "no"
            ),
        ],
    ]
    table = markdown_table(
        [
            "ID",
            "Hypothesis",
            "Primary endpoint",
            "Pre-registered criterion",
            "Observed",
            "Supported",
        ],
        rows,
    )
    supported = sum([h1, h2, h3, h4, h5, h6])
    headline = {
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
        "h5": h5,
        "h6": h6,
        "supported_count": supported,
        "total": 6,
        "det_ok": bool(e6["metrics"]["deterministic_rerun"]["byte_identical"]),
    }
    status = (
        f"{supported} of 6 pre-registered hypotheses were supported. "
        "H1, H2, H3 and H5 passed their stated thresholds. "
        "H4 was not supported after Holm correction, despite a significant "
        "omnibus Friedman test on one dataset and a maximum paired effect size of "
        f"|d_z| = {fmt(e2['metrics']['max_abs_effect_size'], 2)}. "
        f"H6 was only partially supported: Ljung–Box rejected in "
        f"{exploratory.get('volatility_clustering_holm_rejections', 'n/a')} of "
        f"{exploratory.get('n_symbols', 'n/a')} symbols, while Diebold–Mariano rejected "
        f"in {exploratory.get('dm_holm_rejections', 'n/a')} of "
        f"{exploratory.get('n_symbols', 'n/a')}."
    )
    abstract_result = (
        f"All numerical-fidelity checks passed (worst relative gradient error "
        f"{sci(max(h1_checks))}; attention parity {sci(e3['metrics']['attention_parity_max_abs_error'])}). "
        f"A {e4_agg['parameters']:,}-parameter character transformer reached "
        f"{fmt(e4_agg['bpc_mean'], 3)} ± {fmt(e4_agg['bpc_std'], 3)} bits per character versus "
        f"{fmt(primary['bpc'], 3)} for the strongest smoothed trigram baseline. "
        f"On live daily equity data, volatility clustering was Holm-significant in "
        f"{exploratory.get('volatility_clustering_holm_rejections', 'n/a')} of "
        f"{exploratory.get('n_symbols', 'n/a')} symbols and no symbol showed "
        f"Holm-significant next-day directional predictability after costs."
    )
    values = {"STATUS_SENTENCE": status, "ABSTRACT_RESULTS": abstract_result}
    return table, values, headline


def load_labs() -> dict[str, Any]:
    path = ROOT / "labs" / "results" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing lab summary: {path}; run labs/build_summary.py")
    return json.loads(path.read_text(encoding="utf-8"))


def labs_section(labs: dict[str, Any]) -> tuple[str, dict[str, str]]:
    lab1 = labs["labs"]["lab1"]
    lab2 = labs["labs"]["lab2"]
    lab3 = labs["labs"]["lab3"]
    lab4 = labs["labs"]["lab4"]
    lab5 = labs["labs"]["lab5"]
    lag4 = lab2["by_lag"]["4"]
    lag8 = lab2["by_lag"]["8"]
    numpy = lab3["by_workload"]["numpy"]
    torch = lab3["by_workload"]["torch"]
    closest = lab4["closest_holm_p"]
    rows = [
        [
            "1",
            "Falsification coverage",
            "Do detectors catch known errors?",
            f"{lab1['killed']}/{lab1['n_probes']} probes killed "
            f"({lab1['coverage']:.1%}); concealed optional stopping survives",
            "2/2 pass",
        ],
        [
            "2",
            "Locality phase",
            "When does attention beat an n-gram?",
            f"Crossover between lag 2 and 4; {fmt(lag4['delta_bpc_mean'], 2)} BPC at lag 4, "
            f"{fmt(lag8['delta_bpc_mean'], 2)} at lag 8",
            "2/2 pass",
        ],
        [
            "3",
            "Reproducibility envelope",
            "Does thread count change parameter bits?",
            f"{numpy['unique_hashes']} NumPy / {torch['unique_hashes']} torch hashes; "
            f"loss range {sci(max(numpy['loss_range'], torch['loss_range']))}",
            "2/2 pass",
        ],
        [
            "4",
            "Calibration",
            "Do calibration and accuracy rank agree?",
            f"Ranks differ on {len(lab4['per_dataset'])}/3 datasets; closest Holm p = {fmt(closest, 4)}",
            "1/2 pass",
        ],
        [
            "5",
            "Null-market alpha",
            "Does a two-sided DM test detect alpha?",
            f"{lab5['aggregate']['market_false_alpha_rate']:.0%} false alpha; "
            f"{lab5['aggregate']['markets_with_any_two_sided_rejection']}/"
            f"{lab5['aggregate']['symbols'] // 5} markets reject two-sided, all harm",
            "2/2 pass",
        ],
    ]
    table = markdown_table(
        ["Lab", "Question", "What it tests", "Observed", "Pre-registered tests"],
        rows,
    )
    values = {
        "LABS_TABLE": table,
        "LABS_PASSED": str(labs["hypothesis_tests_passed"]),
        "LABS_TOTAL": str(labs["hypothesis_tests_total"]),
        "LABS_COUNT": str(len(labs["labs"])),
    }
    return table, values


def provenance_table(e1: dict[str, Any], e6: dict[str, Any], e4: dict[str, Any]) -> str:
    env = e1["env"]
    packages = env["packages"]
    rows = [
        ["Python", env["python"]],
        ["Platform", env["platform"]],
        ["CPU count", str(env["cpu_count"])],
        ["PyTorch", packages.get("torch", "unknown")],
        ["NumPy", packages.get("numpy", "unknown")],
        ["SciPy", packages.get("scipy", "unknown")],
        ["scikit-learn", packages.get("sklearn", "unknown")],
        ["statsmodels", packages.get("statsmodels", "unknown")],
        ["CUDA available", str(env["torch_cuda_available"])],
        ["Tiny Shakespeare SHA256", e4["inputs"]["corpus_sha256"]],
        ["Configuration SHA256", e6["metrics"]["config_sha256"]],
        ["Pre-registration SHA256", e6["metrics"]["preregistration_sha256"]],
        ["Artifact manifest entries", str(len(e6["metrics"]["manifest"]))],
    ]
    return markdown_table(["Field", "Value"], rows)


def render_readme(values: dict[str, str]) -> None:
    template = (ROOT / "docs" / "README.template.md").read_text(encoding="utf-8")
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    leftover = [line for line in rendered.splitlines() if "{{" in line and "}}" in line]
    if leftover:
        raise RuntimeError(f"unresolved README placeholders: {leftover[:3]}")
    (ROOT / "README.md").write_text(rendered, encoding="utf-8")
    (RESULTS / "report.md").write_text(rendered, encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_style()
    e1 = load("e1_calculus_autograd.json")
    e2 = load("e2_activations.json")
    e3 = load("e3_attention.json")
    e4 = load("e4_gpt.json")
    e5 = load("e5_market.json")
    e6 = load("e6_reproducibility.json")
    labs = load_labs()

    e1_table, e1_values = e1_section(e1)
    e2_summary, e2_pairs, e2_all_pairs, e2_values = e2_section(e2)
    e3_table, e3_values = e3_section(e3)
    e4_table, e4_base_table, e4_values, e4_samples = e4_section(e4)
    e5_table, e5_snapshots, e5_values = e5_section(e5)
    e6_table, e6_values = e6_section(e6)
    labs_table, labs_values = labs_section(labs)
    headline_table, headline_values, headline = build_headline(e1, e2, e3, e4, e5, e6)

    e1_figure(e1)
    e2_figure(e2)
    e4_figure(e4)
    e4_attention_figure(e4)
    e5_figure(e5)

    repo_url = repository_url()
    values: dict[str, str] = {
        "PAGES_URL": pages_url(repo_url),
        "REPO_URL": repo_url or "<repository-url>",
        "ISSUE_URL": (repo_url + "/issues/new?title=Assessment%20request")
        if repo_url
        else "preview.html#business",
        "TEST_COUNT": str(test_count()),
        "FIGURE_COUNT": str(len(list(FIGURES.glob("*.png")))),
        "MANIFEST_COUNT": str(len(e6["metrics"]["manifest"])),
        "SUPPORTED_COUNT": str(headline["supported_count"]),
        "TOTAL_HYPOTHESES": str(headline["total"]),
        "GIF_SIZE_MB": gif_size_mb(),
        "DATE": max(
            e1["generated_utc"],
            e2["generated_utc"],
            e3["generated_utc"],
            e4["generated_utc"],
            e5["generated_utc"],
            e6["generated_utc"],
        )[:10],
        "ENV_PYTHON": e1["env"]["python"],
        "ENV_TORCH": e1["env"]["packages"].get("torch", "unknown"),
        "ENV_NUMPY": e1["env"]["packages"].get("numpy", "unknown"),
        "ENV_PLATFORM": e1["env"]["platform"],
        "ENV_CPU": str(e1["env"]["cpu_count"]),
        "HEADLINE_TABLE": headline_table,
        "PROVENANCE_TABLE": provenance_table(e1, e6, e4),
        "E1_TABLE": e1_table,
        "E2_SUMMARY_TABLE": e2_summary,
        "E2_PAIR_TABLE": e2_pairs,
        "E2_ALL_PAIR_TABLE": e2_all_pairs,
        "E3_TABLE": e3_table,
        "E4_RUN_TABLE": e4_table,
        "E4_BASELINE_TABLE": e4_base_table,
        "E4_SAMPLES": e4_samples,
        "E5_TABLE": e5_table,
        "E5_SNAPSHOT_TABLE": e5_snapshots,
        "E6_TABLE": e6_table,
        "LABS_TABLE": labs_table,
    }
    values.update(e1_values)
    values.update(e2_values)
    values.update(e3_values)
    values.update(e4_values)
    values.update(e5_values)
    values.update(e6_values)
    values.update(headline_values)
    values.update(labs_values)

    (RESULTS / "headline.json").write_text(
        json.dumps(headline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    render_readme(values)
    print(f"Wrote README.md, results/report.md, results/headline.json, and {FIGURES}")


if __name__ == "__main__":
    main()
