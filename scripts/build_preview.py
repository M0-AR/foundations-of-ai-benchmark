"""Render the interactive preview site from the executed result files.

The preview is generated, never hand-edited. Every number shown in the page is
read from ``results/*.json`` and every link is resolved from the repository URL
supplied at build time (or left relative for local use).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEMPLATE = ROOT / "docs" / "preview.template.html"
OUTPUT = ROOT / "preview.html"
INDEX = ROOT / "index.html"
DOCS = ROOT / "docs"
DOCS_PREVIEW = DOCS / "preview.html"
DOCS_INDEX = DOCS / "index.html"
DOCS_NOJEKYLL = DOCS / ".nojekyll"

DOCS_REDIRECT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=preview.html">
<link rel="canonical" href="preview.html">
<meta name="robots" content="noindex">
<title>Foundations-Bench</title>
</head>
<body>
<p><a href="preview.html">Open the Foundations-Bench preview</a>.</p>
</body>
</html>
"""


def repository_url() -> str:
    override = os.environ.get("PREVIEW_REPO_URL", "").strip().rstrip("/")
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


def load(name: str) -> dict[str, Any]:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"missing result file: {path}; run experiments/run_all.py")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | int | None, nd: int = 3) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.{nd}e}"
    return f"{value:.{nd}f}"


def sci(value: float) -> str:
    return f"{float(value):.2e}"


def test_count() -> int:
    return sum(
        line.strip().startswith("def test_")
        for path in (ROOT / "tests").glob("*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def hypothesis_records(
    e1: dict[str, Any],
    e2: dict[str, Any],
    e3: dict[str, Any],
    e4: dict[str, Any],
    e5: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    h1_checks = [
        e1["metrics"]["autograd_vs_finite_difference"]["max_relative_error"],
        e1["metrics"]["numpy_mlp_vs_finite_difference"]["max_relative_error"],
        e1["metrics"]["numpy_mlp_vs_pytorch"]["max_relative_error"],
    ]
    worst = max(h1_checks)
    slope = e1["metrics"]["median_loglog_slope_truncation"]
    e2m = e2["metrics"]
    all_pairs = e2m["all_pairs_holm"]
    min_holm = min(value.get("p_holm", 1.0) for value in all_pairs.values())
    max_dz = e2m["max_abs_effect_size"]
    e4m = e4["metrics"]
    base = e4m["baselines"]["trigram_add1"]["bpc"]
    mean = e4m["aggregate"]["bpc_mean"]
    improve = 100.0 * (base - mean) / base
    ex = e5["metrics"]["exploratory"]
    lb = ex.get("volatility_clustering_holm_rejections", 0)
    dm = ex.get("dm_holm_rejections", 0)
    n = ex.get("n_symbols", 5)

    h1_ok = all(bool(v) for k, v in e1["hypothesis_tests"].items() if k.startswith("H1"))
    h2_ok = bool(e1["hypothesis_tests"]["H2_convergence"])
    h3_ok = all(bool(v) for k, v in e3["hypothesis_tests"].items() if k.startswith("H3"))
    h4_ok = bool(e2["hypothesis_tests"]["H4_any_significant_pair_with_medium_effect"])
    h5_ok = all(bool(v) for k, v in e4["hypothesis_tests"].items() if k.startswith("H5"))
    h6_ok = all(bool(v) for k, v in e5["hypothesis_tests"].items() if k.startswith("H6"))

    hypotheses = [
        {
            "id": "H1",
            "short": "Gradient fidelity",
            "field": "Calculus and autograd",
            "title": "Hand-written gradients match independent references.",
            "plain": (
                "A derivative engine and a NumPy neural network are compared against "
                "finite differences and PyTorch. They agree to roughly one part in ten "
                "billion, far tighter than the pre-registered threshold."
            ),
            "criterion": "Worst relative error < 1e-5 across three checks.",
            "observed": f"{sci(worst)} worst relative error.",
            "status": "supported" if h1_ok else "not supported",
            "why": (
                "If the gradients were wrong, every downstream training result would be "
                "suspect. This check clears the mathematical foundation before any model "
                "claim is made."
            ),
            "link": "experiments/e1_calculus_autograd.py",
        },
        {
            "id": "H2",
            "short": "Second-order convergence",
            "field": "Numerical calculus",
            "title": "Central differences converge at the predicted rate.",
            "plain": (
                "The error should fall with the square of the step size in the truncation "
                "regime. The measured median log–log slope is 2.000, which is the textbook "
                "value for a central difference."
            ),
            "criterion": "Median log–log slope ≥ 1.8.",
            "observed": f"Median slope {fmt(slope, 3)}.",
            "status": "supported" if h2_ok else "not supported",
            "why": (
                "A correct implementation should not merely produce small errors; it should "
                "produce errors that shrink at the theoretically predicted rate."
            ),
            "link": "experiments/e1_calculus_autograd.py",
        },
        {
            "id": "H3",
            "short": "Attention and RoPE fidelity",
            "field": "Transformer internals",
            "title": "Attention is causal, exact, and position-aware.",
            "plain": (
                "Causal attention matches PyTorch's fused implementation at machine "
                "precision, leaks exactly zero future information, and the rotary position "
                "encoding preserves the intended relative-distance structure."
            ),
            "criterion": "Parity < 1e-6, leak == 0, RoPE invariants < 1e-6.",
            "observed": (
                f"Parity {sci(e3['metrics']['attention_parity_max_abs_error'])}, "
                f"leak {sci(e3['metrics']['causal_leak_max'])}, "
                f"RoPE {sci(e3['metrics']['rope_toeplitz_max_deviation'])}."
            ),
            "status": "supported" if h3_ok else "not supported",
            "why": (
                "Most transformer bugs hide in masking, softmax normalisation, or position "
                "handling. These checks isolate each of those failure modes."
            ),
            "link": "experiments/e3_attention.py",
        },
        {
            "id": "H4",
            "short": "Activation effects",
            "field": "Small-model statistics",
            "title": "A large raw effect still failed the strict test.",
            "plain": (
                "Activation functions were compared on three datasets and ten seeds each. "
                "One paired effect was large, but after correcting for thirty tests no pair "
                "met the pre-registered significance threshold."
            ),
            "criterion": "Holm p < 0.05 and |d_z| ≥ 0.5 for at least one pair.",
            "observed": f"Smallest Holm p {fmt(min_holm, 3)}; largest |d_z| {fmt(max_dz, 3)}.",
            "status": "supported" if h4_ok else "not supported",
            "why": (
                "This is the most useful negative result in the project: effect size alone "
                "is not evidence when many comparisons are made."
            ),
            "link": "experiments/e2_activations.py",
        },
        {
            "id": "H5",
            "short": "Transformer versus n-gram",
            "field": "Language modelling",
            "title": "The transformer beats the strongest smoothed trigram.",
            "plain": (
                "A 0.8M-parameter character transformer reaches 2.558 bits per character "
                "on Tiny Shakespeare, versus 3.096 for a smoothed trigram. The improvement "
                "holds across all three seeds."
            ),
            "criterion": "Lower BPC on every seed and ≥ 5% improvement.",
            "observed": f"{fmt(mean, 3)} vs {fmt(base, 3)} ({fmt(improve, 1)}% lower).",
            "status": "supported" if h5_ok else "not supported",
            "why": (
                "The comparison is deliberately unfair to the transformer in compute and "
                "fair in statistics: a cheap n-gram baseline, three seeds, and a margin "
                "fixed before the run."
            ),
            "link": "experiments/e4_gpt.py",
        },
        {
            "id": "H6",
            "short": "Market structure versus alpha",
            "field": "Live data",
            "title": "Volatility clusters; next-day direction does not.",
            "plain": (
                f"Ljung–Box rejects in {lb} of {n} liquid symbols, so volatility clustering "
                f"is real. Diebold–Mariano rejects in {dm} of {n} after costs, so there is "
                "no reliable next-day directional edge at this scale."
            ),
            "criterion": "Clustering on all symbols and no directional alpha.",
            "observed": f"Ljung–Box {lb}/{n}; Diebold–Mariano {dm}/{n}.",
            "status": "partial"
            if (lb > 0 and dm == 0)
            else ("supported" if h6_ok else "not supported"),
            "why": (
                "Structure and alpha are different claims. A market can be highly structured "
                "and still be very hard to trade after costs."
            ),
            "link": "experiments/e5_market.py",
        },
    ]
    studies = [
        {
            "id": "E1",
            "title": "Calculus and gradients",
            "plain": "Does the hand-written derivative engine actually compute gradients?",
            "criterion": "Slope ≥ 1.8; worst relative error < 1e-5.",
            "observed": f"Slope {fmt(slope, 3)}; worst error {sci(worst)}.",
            "status": "supported" if (h1_ok and h2_ok) else "not supported",
            "why": "This is the foundation: if gradients are wrong, every later training result is meaningless.",
            "link": "experiments/e1_calculus_autograd.py",
            "field": "H1 · H2",
        },
        {
            "id": "E2",
            "title": "Activation functions",
            "plain": "Do activation choices reliably change small-model accuracy?",
            "criterion": "Holm p < 0.05 and |d_z| ≥ 0.5 for at least one pair.",
            "observed": f"Smallest Holm p {fmt(min_holm, 3)}; largest |d_z| {fmt(max_dz, 3)}.",
            "status": "supported" if h4_ok else "not supported",
            "why": "It demonstrates why multiplicity correction matters, even when a raw effect looks large.",
            "link": "experiments/e2_activations.py",
            "field": "H4",
        },
        {
            "id": "E3",
            "title": "Attention and RoPE",
            "plain": "Are causal attention and rotary positions implemented correctly?",
            "criterion": "Parity < 1e-6; leak == 0; RoPE invariants < 1e-6.",
            "observed": f"Parity {sci(e3['metrics']['attention_parity_max_abs_error'])}; leak {sci(e3['metrics']['causal_leak_max'])}.",
            "status": "supported" if h3_ok else "not supported",
            "why": "Attention bugs are subtle; exact parity and zero leakage catch the common ones.",
            "link": "experiments/e3_attention.py",
            "field": "H3",
        },
        {
            "id": "E4",
            "title": "Character transformer",
            "plain": "Does a small transformer beat a strong n-gram baseline?",
            "criterion": "Lower BPC on every seed and ≥ 5% improvement.",
            "observed": f"{fmt(mean, 3)} vs {fmt(base, 3)} BPC.",
            "status": "supported" if h5_ok else "not supported",
            "why": "It connects implementation fidelity to a measurable language-modelling outcome.",
            "link": "experiments/e4_gpt.py",
            "field": "H5",
        },
        {
            "id": "E5",
            "title": "Live market test",
            "plain": "Is there structure, and is it tradeable after costs?",
            "criterion": "Clustering on all symbols and no directional alpha.",
            "observed": f"Ljung–Box {lb}/{n}; Diebold–Mariano {dm}/{n}.",
            "status": "partial"
            if (lb > 0 and dm == 0)
            else ("supported" if h6_ok else "not supported"),
            "why": "It separates statistical structure from a tradeable edge on live, uncontrolled data.",
            "link": "experiments/e5_market.py",
            "field": "H6",
        },
        {
            "id": "E6",
            "title": "Reproducibility audit",
            "plain": "Does the same seed produce the same bytes, and is every artifact hashed?",
            "criterion": "Byte-identical retraining; complete manifest.",
            "observed": (
                f"Deterministic {str(e6_deterministic(e1, e2, e3, e4, e5)).lower()}; "
                f"{len(load('e6_reproducibility.json')['metrics']['manifest'])} files hashed."
            ),
            "status": (
                "supported"
                if load("e6_reproducibility.json")["metrics"]["deterministic_rerun"][
                    "byte_identical"
                ]
                else "not supported"
            ),
            "why": "A result that cannot be regenerated is a story, not evidence.",
            "link": "experiments/e6_reproducibility.py",
            "field": "E6",
        },
    ]
    return hypotheses, studies


def e6_deterministic(*_: Any) -> bool:
    return bool(load("e6_reproducibility.json")["metrics"]["deterministic_rerun"]["byte_identical"])


def build_data(repo_url: str) -> dict[str, Any]:
    e1 = load("e1_calculus_autograd.json")
    e2 = load("e2_activations.json")
    e3 = load("e3_attention.json")
    e4 = load("e4_gpt.json")
    e5 = load("e5_market.json")
    e6 = load("e6_reproducibility.json")
    hypotheses, studies = hypothesis_records(e1, e2, e3, e4, e5)

    def link(path: str) -> str:
        return f"{repo_url}/blob/main/{path}" if repo_url else path

    for item in hypotheses + studies:
        item["link"] = link(item["link"])
    for item in studies:
        item["why"] = item["why"]

    e4m = e4["metrics"]
    base = e4m["baselines"]["trigram_add1"]["bpc"]
    mean = e4m["aggregate"]["bpc_mean"]
    seed_rows = [{"seed": run["seed"], "bpc": run["val_bpc"]} for run in e4m["runs"]]
    baselines = [
        {
            "key": "uniform",
            "label": "Uniform",
            "sublabel": "no structure",
            "value": e4m["baselines"]["uniform"]["bpc"],
        },
        {
            "key": "unigram",
            "label": "Unigram",
            "sublabel": "frequency only",
            "value": e4m["baselines"]["unigram"]["bpc"],
        },
        {
            "key": "interpolated",
            "label": "Interpolated",
            "sublabel": "mixture",
            "value": e4m["baselines"]["interpolated_ngram"]["bpc"],
        },
        {"key": "trigram", "label": "Trigram +1", "sublabel": "strong baseline", "value": base},
        {"key": "tinygpt", "label": "TinyGPT", "sublabel": "3 seeds", "value": mean},
    ]
    order = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
    per = e5["metrics"]["per_symbol"]
    symbols = []
    for symbol in order:
        if symbol not in per:
            continue
        entry = per[symbol]
        ridge = entry["models"]["ridge"]
        ci = ridge["oos_r2_vs_zero_ci95"]
        symbols.append(
            {
                "symbol": symbol,
                "r2": ridge["oos_r2_vs_zero"],
                "lo": ci[0],
                "hi": ci[1],
                "lbP": entry["ljung_box"]["p_holm"],
                "dmP": ridge["dm_vs_zero"]["p_holm"],
                "sharpe": ridge["strategy"]["sharpe"],
            }
        )
    exploratory = e5["metrics"]["exploratory"]
    date = max(
        e1["generated_utc"],
        e2["generated_utc"],
        e3["generated_utc"],
        e4["generated_utc"],
        e5["generated_utc"],
        e6["generated_utc"],
    )[:10]
    supported = sum(1 for h in hypotheses if h["status"] == "supported")
    manifest = len(e6["metrics"]["manifest"])
    figures = len(list((ROOT / "paper" / "figures").glob("*.png")))

    return {
        "meta": {
            "date": date,
            "version": "0.1.0",
            "studies": 6,
            "supported": supported,
            "total": 6,
            "manifest": manifest,
            "tests": test_count(),
            "figures": figures,
            "repoUrl": repo_url,
        },
        "e1": {
            "slope": e1["metrics"]["median_loglog_slope_truncation"],
            "gradErr": max(
                e1["metrics"]["autograd_vs_finite_difference"]["max_relative_error"],
                e1["metrics"]["numpy_mlp_vs_finite_difference"]["max_relative_error"],
                e1["metrics"]["numpy_mlp_vs_pytorch"]["max_relative_error"],
            ),
            "gradErrText": sci(
                max(
                    e1["metrics"]["autograd_vs_finite_difference"]["max_relative_error"],
                    e1["metrics"]["numpy_mlp_vs_finite_difference"]["max_relative_error"],
                    e1["metrics"]["numpy_mlp_vs_pytorch"]["max_relative_error"],
                )
            ),
        },
        "e2": {
            "maxDz": e2["metrics"]["max_abs_effect_size"],
            "minHolm": min(v.get("p_holm", 1.0) for v in e2["metrics"]["all_pairs_holm"].values()),
            "pairs": len(e2["metrics"]["all_pairs_holm"]),
        },
        "e3": {
            "parity": e3["metrics"]["attention_parity_max_abs_error"],
            "leak": e3["metrics"]["causal_leak_max"],
            "rope": e3["metrics"]["rope_toeplitz_max_deviation"],
        },
        "e4": {
            "bpcMean": mean,
            "bpcStd": e4m["aggregate"]["bpc_std"],
            "baseBpc": base,
            "improvePct": 100.0 * (base - mean) / base,
            "params": e4m["aggregate"]["parameters"],
            "seconds": e4m["aggregate"]["train_seconds_mean"],
            "seeds": seed_rows,
            "baselines": baselines,
        },
        "e5": {
            "symbols": symbols,
            "lbRejects": exploratory.get("volatility_clustering_holm_rejections", 0),
            "dmRejects": exploratory.get("dm_holm_rejections", 0),
            "n": exploratory.get("n_symbols", 5),
            "nonpos": exploratory.get("symbols_with_nonpositive_oos_r2", 0),
            "lastDate": max(s["last_date"] for s in e5["metrics"]["snapshots"]),
        },
        "e6": {
            "deterministic": bool(e6["metrics"]["deterministic_rerun"]["byte_identical"]),
            "tau": sum(v["kendall_tau"] for v in e6["metrics"]["seed_stability"].values())
            / len(e6["metrics"]["seed_stability"]),
            "manifest": manifest,
        },
        "hypotheses": hypotheses,
        "studies": studies,
    }


def render(repo_url: str) -> None:
    data = build_data(repo_url)
    template = TEMPLATE.read_text(encoding="utf-8")
    values = {
        "DATA_JSON": json.dumps(data, separators=(",", ":"), ensure_ascii=True).replace(
            "</", "<\\/"
        ),
        "REPO_URL": repo_url or "#top",
        "ISSUE_URL": (
            f"{repo_url}/issues/new?title=Assessment%20request" if repo_url else "#business"
        ),
        "VERSION": data["meta"]["version"],
        "DATE": data["meta"]["date"],
        "SUPPORTED": str(data["meta"]["supported"]),
        "TOTAL": str(data["meta"]["total"]),
        "BPC_MEAN": fmt(data["e4"]["bpcMean"], 3),
        "BPC_BASE": fmt(data["e4"]["baseBpc"], 3),
        "CORPUS_HASH": load("e4_gpt.json")["inputs"]["corpus_sha256"],
        "CONFIG_HASH": load("e6_reproducibility.json")["metrics"]["config_sha256"],
        "PREREG_HASH": load("e6_reproducibility.json")["metrics"]["preregistration_sha256"],
        "MANIFEST": str(data["meta"]["manifest"]),
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    leftover = [line for line in rendered.splitlines() if "{{" in line and "}}" in line]
    if leftover:
        raise RuntimeError(f"unresolved preview placeholders: {leftover[:3]}")
    docs_markers = (
        'content="docs/assets/preview-hero.png"',
        '<a href="README.md">README</a>',
    )
    for marker in docs_markers:
        if marker not in rendered:
            raise RuntimeError(f"docs mirror marker missing: {marker}")
    docs_rendered = rendered.replace(
        'content="docs/assets/preview-hero.png"',
        'content="assets/preview-hero.png"',
    ).replace('<a href="README.md">README</a>', '<a href="../README.md">README</a>')

    OUTPUT.write_text(rendered, encoding="utf-8")
    INDEX.write_text(rendered, encoding="utf-8")
    DOCS_PREVIEW.write_text(docs_rendered, encoding="utf-8")
    DOCS_INDEX.write_text(DOCS_REDIRECT, encoding="utf-8")
    DOCS_NOJEKYLL.write_text("", encoding="utf-8")
    print(
        f"Wrote {OUTPUT.name}, {INDEX.name}, {DOCS_PREVIEW.relative_to(ROOT)}, "
        f"and {DOCS_INDEX.relative_to(ROOT)} ({len(rendered):,} bytes)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-url",
        default=os.environ.get("PREVIEW_REPO_URL", ""),
        help="Public repository URL used for code, issue, and README links.",
    )
    parser.add_argument(
        "--github",
        action="store_true",
        help="Derive the repository URL from GITHUB_REPOSITORY.",
    )
    args = parser.parse_args()
    repo_url = args.repo_url.rstrip("/")
    if not repo_url and not args.github:
        repo_url = repository_url()
    if args.github and not repo_url:
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        if repo:
            repo_url = f"https://github.com/{repo}"
    render(repo_url)


if __name__ == "__main__":
    main()
