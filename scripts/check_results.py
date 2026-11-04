"""Validate result artifacts, provenance hashes, and the generated manuscript."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REQUIRED = {
    "E1": "e1_calculus_autograd.json",
    "E2": "e2_activations.json",
    "E3": "e3_attention.json",
    "E4": "e4_gpt.json",
    "E5": "e5_market.json",
    "E6": "e6_reproducibility.json",
}
EXPECTED_HYPOTHESES = {
    "E1": {"H1_autograd_fd", "H1_numpy_mlp_fd", "H1_numpy_mlp_torch", "H2_convergence"},
    "E2": {"H4_any_significant_pair_with_medium_effect"},
    "E3": {
        "H3_attention_parity",
        "H3_causal_leak",
        "H3_gradcheck",
        "H3_rope_informative",
        "H3_rope_norm",
        "H3_rope_toeplitz",
        "H3_softmax_normalised",
    },
    "E4": {
        "H5_transformer_beats_smoothed_trigram",
        "H5_improvement_at_least_5pct",
        "H5_seed_consistency",
    },
    "E5": {"H6_volatility_clustering", "H6_no_directional_alpha"},
    "E6": {"deterministic_rerun_byte_identical"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_finite(value: object, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SystemExit(f"non-finite primary endpoint at {path}: {value}")
    if isinstance(value, dict):
        for key, item in value.items():
            check_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_finite(item, f"{path}[{index}]")


def check_primary_endpoints(payloads: dict[str, dict]) -> None:
    e1 = payloads["E1"]["metrics"]
    for key in ("median_loglog_slope_truncation",):
        check_finite(e1[key], f"E1.{key}")
    for key in (
        "autograd_vs_finite_difference",
        "numpy_mlp_vs_finite_difference",
        "numpy_mlp_vs_pytorch",
    ):
        check_finite(e1[key]["max_relative_error"], f"E1.{key}")

    e2 = payloads["E2"]["metrics"]
    check_finite(e2["max_abs_effect_size"], "E2.max_abs_effect_size")
    for key, value in e2["all_pairs_holm"].items():
        check_finite(value["p_holm"], f"E2.all_pairs_holm.{key}.p_holm")
        check_finite(value["cohen_dz"], f"E2.all_pairs_holm.{key}.cohen_dz")

    for key, value in payloads["E3"]["metrics"].items():
        if isinstance(value, (int, float)):
            check_finite(value, f"E3.{key}")

    e4 = payloads["E4"]["metrics"]
    for key in ("bpc_mean", "bpc_std", "bpc_min", "bpc_max"):
        check_finite(e4["aggregate"][key], f"E4.aggregate.{key}")
    for name, value in e4["baselines"].items():
        check_finite(value["bpc"], f"E4.baselines.{name}.bpc")
    for run in e4["runs"]:
        check_finite(run["val_bpc"], f"E4.runs[{run['seed']}].val_bpc")

    e5 = payloads["E5"]["metrics"]
    for symbol, value in e5["per_symbol"].items():
        check_finite(value["ljung_box"]["p_holm"], f"E5.{symbol}.ljung_box.p_holm")
        ridge = value["models"]["ridge"]
        check_finite(ridge["oos_r2_vs_zero"], f"E5.{symbol}.ridge.oos_r2")
        check_finite(ridge["dm_vs_zero"]["p_holm"], f"E5.{symbol}.ridge.dm_p_holm")


README_REQUIRED = [
    "(#-beginner-guide--read-this-and-you-are-a-professional)",
    "You will know more than most interview candidates.",
    "Let's work this out in a step-by-step way to be sure we have the right answer.",
    "docs/assets/demo.gif",
    "docs/assets/preview-results.png",
    "docs/assets/preview-quiz.png",
    "docs/assets/preview-mobile.png",
]
FORBIDDEN_README = re.compile(
    r"\b(playwright|mcp|exa|duckduckgo|kaggle|ai-generated|headless browser|autonomous coding agent)\b",
    re.IGNORECASE,
)
SCREENSHOTS = [
    "preview-hero.png",
    "preview-results.png",
    "preview-learn.png",
    "preview-business.png",
    "preview-quiz.png",
    "preview-mobile.png",
]


def check_site_and_assets(readme_text: str) -> None:
    preview = ROOT / "preview.html"
    index = ROOT / "index.html"
    for path in (preview, index):
        if not path.exists():
            raise SystemExit(f"{path.name} is missing; run scripts/build_preview.py")
    preview_bytes = preview.read_bytes()
    if preview_bytes != index.read_bytes():
        raise SystemExit("preview.html and index.html differ; rerun scripts/build_preview.py")
    site = preview_bytes.decode("utf-8")
    leftover = [line for line in site.splitlines() if "{{" in line and "}}" in line]
    if leftover:
        raise SystemExit(f"{preview.name} has unresolved placeholders: {leftover[:3]}")
    for marker in ('id="quiz"', 'id="learn"', "docs/assets/preview-hero.png"):
        if marker not in site:
            raise SystemExit(f"{preview.name}: missing {marker}")

    for required in README_REQUIRED:
        if required not in readme_text:
            raise SystemExit(f"README.md: missing required text: {required}")
    forbidden = sorted(set(match.group(0) for match in FORBIDDEN_README.finditer(readme_text)))
    if forbidden:
        raise SystemExit(f"README.md: remove tool mentions: {forbidden}")
    for target in re.findall(r"\]\(([^)]+)\)", readme_text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = target.split("#", 1)[0]
        if path and not (ROOT / path).exists():
            raise SystemExit(f"README.md: broken local link: {target}")

    gif = ROOT / "docs" / "assets" / "demo.gif"
    if not gif.exists():
        raise SystemExit("docs/assets/demo.gif is missing; run scripts/make_demo.py")
    header = gif.read_bytes()[:6]
    if header not in (b"GIF87a", b"GIF89a"):
        raise SystemExit("docs/assets/demo.gif is not a GIF")
    size = gif.stat().st_size
    if not 100_000 <= size <= 6_000_000:
        raise SystemExit(f"docs/assets/demo.gif is {size} bytes; expected 100 KB to 6 MB")

    for name in SCREENSHOTS:
        path = ROOT / "docs" / "assets" / name
        if not path.exists() or path.stat().st_size < 10_000:
            raise SystemExit(f"missing or tiny screenshot: docs/assets/{name}")

    for name in (".nojekyll", "CITATION.cff", "docs/PUBLISHING.md"):
        if not (ROOT / name).exists():
            raise SystemExit(f"missing publishing file: {name}")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if "cff-version: 1.2.0" not in citation or "license: MIT" not in citation:
        raise SystemExit("CITATION.cff is missing required CFF fields")


def main() -> int:
    payloads = {}
    for key, name in REQUIRED.items():
        payload = load(name)
        for field in ("experiment", "generated_utc", "env", "metrics", "hypothesis_tests"):
            if field not in payload:
                raise SystemExit(f"{name}: missing field {field}")
        missing = EXPECTED_HYPOTHESES[key] - set(payload["hypothesis_tests"])
        if missing:
            raise SystemExit(f"{name}: missing hypothesis keys {sorted(missing)}")
        payloads[key] = payload

    check_primary_endpoints(payloads)

    if len(payloads["E2"]["metrics"]["all_pairs_holm"]) != 30:
        raise SystemExit("E2: expected 30 pre-registered all-pairs tests")
    if len(payloads["E4"]["metrics"]["runs"]) != 3:
        raise SystemExit("E4: expected 3 seeds")
    if len(payloads["E5"]["metrics"]["per_symbol"]) != 5:
        raise SystemExit("E5: expected 5 symbols")

    summary_path = RESULTS / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("all_experiments_completed"):
            raise SystemExit("summary.json: all_experiments_completed is not true")
        if set(summary.get("experiments", {})) != set(REQUIRED):
            raise SystemExit("summary.json: experiment set mismatch")

    headline_path = RESULTS / "headline.json"
    if headline_path.exists():
        headline = json.loads(headline_path.read_text(encoding="utf-8"))
        if headline.get("total") != 6:
            raise SystemExit("headline.json: total is not 6")

    e6 = payloads["E6"]["metrics"]
    for relative, record in e6["manifest"].items():
        path = ROOT / relative
        if not path.exists():
            raise SystemExit(f"manifest file missing: {relative}")
        if sha256(path) != record["sha256"]:
            raise SystemExit(f"manifest hash mismatch: {relative}")

    readme = ROOT / "README.md"
    if not readme.exists():
        raise SystemExit("README.md is missing; run scripts/make_report.py")
    text = readme.read_text(encoding="utf-8")
    leftover = [line for line in text.splitlines() if "{{" in line and "}}" in line]
    if leftover:
        raise SystemExit(f"README.md has unresolved placeholders: {leftover[:3]}")

    print(f"OK: {len(payloads)} result files, {len(e6['manifest'])} manifest entries")
    print(
        f"Hypotheses supported: {sum(bool(p['all_passed']) for p in payloads.values())}/6 runs passed"
    )
    check_site_and_assets(text)

    print(f"README bytes: {len(text):,}")
    print("Site: preview.html == index.html, placeholders resolved, demo assets present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
