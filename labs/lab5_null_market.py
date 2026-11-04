"""Lab 5: false-alpha rate on zero-drift GARCH markets with volatility clustering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.e5_market import _predict_ridge, build_features  # noqa: E402
from fbench.stats import diebold_mariano, holm, oos_r2  # noqa: E402

OUT = ROOT / "labs" / "results" / "lab5_null_market.json"
N_MARKETS = 50
N_SYMBOLS = 5
N_DAYS = 1250
OMEGA = 1e-6
ALPHA = 0.08
BETA = 0.90


def garch_returns(rng: np.random.Generator, n: int) -> np.ndarray:
    burn = 200
    sigma2 = OMEGA / (1.0 - ALPHA - BETA)
    shocks = rng.normal(size=n + burn)
    returns = np.empty(n + burn)
    for t in range(n + burn):
        sigma2 = OMEGA + ALPHA * (returns[t - 1] ** 2 if t else sigma2) + BETA * sigma2
        returns[t] = np.sqrt(sigma2) * shocks[t]
    return returns[burn:]


def make_frame(returns: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.concatenate(([close[0]], close[:-1]))
    noise = rng.uniform(0.0005, 0.004, size=len(close))
    high = np.maximum(open_, close) * (1.0 + noise)
    low = np.minimum(open_, close) * (1.0 - noise)
    volume = rng.lognormal(mean=14.0, sigma=0.4, size=len(close))
    return pd.DataFrame(
        {
            "date": pd.date_range("2019-01-01", periods=len(close), freq="B"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def analyse(returns: np.ndarray, rng: np.random.Generator) -> dict:
    frame = make_frame(returns, rng)
    lb = acorr_ljungbox(returns**2, lags=[10], return_df=True)
    x, y, _ = build_features(frame, lags=5, vol_window=20, momentum_window=20)
    pred = _predict_ridge(x, y, 0.6, 21, [0.01, 0.1, 1.0, 10.0])
    n_train = int(0.6 * len(y))
    y_oos, pred_oos = y[n_train:], pred[n_train:]
    residual = y_oos - pred_oos
    dm = diebold_mariano(residual**2, y_oos**2)
    return {
        "ljung_box_p": float(lb["lb_pvalue"].iloc[0]),
        "dm_p": float(dm["p_value"]),
        "dm_stat": float(dm["stat"]),
        "dm_mean_diff": float(dm["mean_diff"]),
        "oos_r2": float(oos_r2(y_oos, pred_oos)),
        "n_oos": int(len(y_oos)),
    }


def main() -> dict:
    rng = np.random.default_rng(2026)
    markets = []
    for market in range(N_MARKETS):
        symbols = [analyse(garch_returns(rng, N_DAYS), rng) for _ in range(N_SYMBOLS)]
        dm_holm = holm({f"s{i}": s["dm_p"] for i, s in enumerate(symbols)})
        for i, symbol in enumerate(symbols):
            symbol["dm_p_holm"] = dm_holm[f"s{i}"]["p_holm"]
            symbol["dm_reject_holm"] = bool(dm_holm[f"s{i}"]["reject"])
            symbol["dm_better_holm"] = bool(dm_holm[f"s{i}"]["reject"] and symbol["dm_stat"] < 0)
        markets.append(
            {
                "market": market,
                "symbols": symbols,
                "lb_rejections": int(sum(s["ljung_box_p"] < 0.05 for s in symbols)),
                "dm_rejections_holm": int(sum(s["dm_reject_holm"] for s in symbols)),
                "dm_better_holm": int(sum(s["dm_better_holm"] for s in symbols)),
            }
        )
        if (market + 1) % 10 == 0:
            print(f"lab5 market {market + 1}/{N_MARKETS}")

    all_symbols = [s for m in markets for s in m["symbols"]]
    lb_rate = float(np.mean([s["ljung_box_p"] < 0.05 for s in all_symbols]))
    false_alpha_rate = float(np.mean([m["dm_better_holm"] > 0 for m in markets]))
    any_rejections = int(sum(m["dm_better_holm"] > 0 for m in markets))
    any_two_sided = int(sum(m["dm_rejections_holm"] > 0 for m in markets))
    payload = {
        "lab": "lab5_null_market",
        "config": {
            "n_markets": N_MARKETS,
            "n_symbols": N_SYMBOLS,
            "n_days": N_DAYS,
            "omega": OMEGA,
            "alpha": ALPHA,
            "beta": BETA,
            "initial_train_fraction": 0.6,
            "refit_every": 21,
        },
        "market_summary": [
            {
                "market": m["market"],
                "lb_rejections": m["lb_rejections"],
                "dm_rejections_holm": m["dm_rejections_holm"],
                "dm_better_holm": m["dm_better_holm"],
            }
            for m in markets
        ],
        "aggregate": {
            "symbols": len(all_symbols),
            "volatility_clustering_rate": lb_rate,
            "market_false_alpha_rate": false_alpha_rate,
            "markets_with_any_significant_better_model": any_rejections,
            "markets_with_any_two_sided_rejection": any_two_sided,
            "mean_dm_rejections_per_market": float(
                np.mean([m["dm_rejections_holm"] for m in markets])
            ),
            "mean_dm_better_per_market": float(np.mean([m["dm_better_holm"] for m in markets])),
            "mean_oos_r2": float(np.mean([s["oos_r2"] for s in all_symbols])),
        },
        "hypothesis_tests": {
            "H11a_volatility_clustering_ge_0_80": bool(lb_rate >= 0.80),
            "H11b_false_alpha_le_0_10": bool(false_alpha_rate <= 0.10),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))
    print(json.dumps(payload["hypothesis_tests"], indent=2))
    return payload


if __name__ == "__main__":
    main()
