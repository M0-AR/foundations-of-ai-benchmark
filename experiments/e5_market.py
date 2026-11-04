"""E5: live daily equity data - volatility clustering versus directional alpha."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402

from fbench.data import cache_market_snapshot, fetch_yahoo_daily  # noqa: E402
from fbench.nn import LinearRegressor  # noqa: E402
from fbench.provenance import env_manifest, sha256_file, utc_now, write_json  # noqa: E402
from fbench.stats import (  # noqa: E402
    diebold_mariano,
    directional_accuracy,
    holm,
    max_drawdown,
    oos_r2,
    sharpe_ratio,
    spearman_ic,
)

CFG = ROOT / "config" / "experiment.yaml"


def main() -> dict:
    with open(CFG, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    spec = cfg["e5_market"]
    symbols = list(cfg["data"]["market_symbols"])

    per_symbol = {}
    snapshots = []
    for symbol in symbols:
        snapshot = fetch_yahoo_daily(
            symbol,
            range_=cfg["data"]["market_range"],
            interval=cfg["data"]["market_interval"],
        )
        cache_market_snapshot(snapshot)
        snapshots.append(
            {
                "symbol": symbol,
                "rows": int(len(snapshot.frame)),
                "first_date": str(snapshot.frame["date"].iloc[0].date()),
                "last_date": snapshot.last_date,
                "last_close": snapshot.last_close,
                "retrieved_utc": snapshot.retrieved_utc,
                "raw_sha256": snapshot.raw_sha256,
                "source_url": snapshot.source_url,
            }
        )
        print(
            f"E5 {symbol}: {len(snapshot.frame)} rows, last {snapshot.last_date} "
            f"close={snapshot.last_close:.2f}"
        )
        per_symbol[symbol] = analyse_symbol(snapshot.frame, spec)

    # ------------------------------------------------ Holm across symbols
    lb_p = {s: per_symbol[s]["ljung_box"]["p_value"] for s in symbols}
    lb_holm = holm(lb_p)
    dm_p = {s: per_symbol[s]["models"]["ridge"]["dm_vs_zero"]["p_value"] for s in symbols}
    dm_holm = holm(dm_p)
    for symbol in symbols:
        lb_entry = lb_holm.get(symbol, {"p_holm": 1.0, "reject": False})
        dm_entry = dm_holm.get(symbol, {"p_holm": 1.0, "reject": False})
        per_symbol[symbol]["ljung_box"]["p_holm"] = lb_entry["p_holm"]
        per_symbol[symbol]["ljung_box"]["reject_holm"] = lb_entry["reject"]
        per_symbol[symbol]["models"]["ridge"]["dm_vs_zero"]["p_holm"] = dm_entry["p_holm"]
        per_symbol[symbol]["models"]["ridge"]["dm_vs_zero"]["reject_holm"] = dm_entry["reject"]

    volatility_clustering = all(per_symbol[s]["ljung_box"]["reject_holm"] for s in symbols)
    no_directional_alpha = all(
        per_symbol[s]["models"]["ridge"]["oos_r2_vs_zero"] <= 0.0
        and not per_symbol[s]["models"]["ridge"]["dm_vs_zero"]["reject_holm"]
        for s in symbols
    )
    lb_rejections = sum(bool(per_symbol[s]["ljung_box"]["reject_holm"]) for s in symbols)
    dm_rejections = sum(
        bool(per_symbol[s]["models"]["ridge"]["dm_vs_zero"]["reject_holm"]) for s in symbols
    )
    nonpositive_r2 = sum(per_symbol[s]["models"]["ridge"]["oos_r2_vs_zero"] <= 0.0 for s in symbols)
    metrics = {
        "snapshots": snapshots,
        "per_symbol": per_symbol,
        "holm_ljung_box": lb_holm,
        "holm_dm_ridge": dm_holm,
        "volatility_clustering_all_symbols": bool(volatility_clustering),
        "no_directional_alpha_all_symbols": bool(no_directional_alpha),
        "exploratory": {
            "label": "post hoc descriptive summary; not used for the pre-registered test",
            "volatility_clustering_holm_rejections": int(lb_rejections),
            "volatility_clustering_majority": bool(lb_rejections > len(symbols) / 2),
            "dm_holm_rejections": int(dm_rejections),
            "no_holm_significant_directional_alpha": bool(dm_rejections == 0),
            "symbols_with_nonpositive_oos_r2": int(nonpositive_r2),
            "n_symbols": len(symbols),
        },
    }
    thresholds = {
        "H6_volatility_clustering": bool(volatility_clustering),
        "H6_no_directional_alpha": bool(no_directional_alpha),
    }
    payload = {
        "experiment": "E5_market",
        "generated_utc": utc_now(),
        "env": env_manifest(),
        "config": spec,
        "inputs": {
            "symbols": symbols,
            "range": cfg["data"]["market_range"],
            "interval": cfg["data"]["market_interval"],
            "script_sha256": sha256_file(__file__),
        },
        "metrics": metrics,
        "hypothesis_tests": thresholds,
        "all_passed": bool(all(thresholds.values())),
    }
    write_json(ROOT / "results" / "e5_market.json", payload)
    print(
        f"E5 vol_cluster={volatility_clustering} no_alpha={no_directional_alpha} "
        f"pass={payload['all_passed']}"
    )
    return payload


def analyse_symbol(frame: pd.DataFrame, spec: dict) -> dict:
    returns = np.diff(np.log(frame["close"].to_numpy(dtype=float)))
    x, y, dates = build_features(
        frame, int(spec["lags"]), int(spec["vol_window"]), int(spec["momentum_window"])
    )
    # Ljung-Box on squared returns (full sample, as in the pre-registration)
    lb = acorr_ljungbox(returns**2, lags=[int(spec["ljung_box_lags"])], return_df=True)
    ljung = {
        "statistic": float(lb["lb_stat"].iloc[0]),
        "p_value": float(lb["lb_pvalue"].iloc[0]),
        "lags": int(spec["ljung_box_lags"]),
        "n_returns": int(len(returns)),
    }
    cost = float(spec["cost_bps_per_turnover"]) / 10_000.0
    n_train = int(float(spec["initial_train_fraction"]) * len(y))
    models = {
        "zero": _predict_zero(x),
        "mean": _predict_mean(
            x, y, float(spec["initial_train_fraction"]), int(spec["refit_every"])
        ),
        "ridge": _predict_ridge(
            x,
            y,
            float(spec["initial_train_fraction"]),
            int(spec["refit_every"]),
            [float(a) for a in spec["ridge_alphas"]],
        ),
        "linear_gd": _predict_gd(
            x, y, float(spec["initial_train_fraction"]), int(spec["refit_every"]), spec
        ),
    }
    results = {}
    for name, pred in models.items():
        results[name] = evaluate(y[n_train:], pred[n_train:], cost, spec)
    # strategy returns use the ridge forecast as the primary model
    strategy = results["ridge"].pop("_strategy_returns")
    results["ridge"]["strategy"] = {
        "sharpe": sharpe_ratio(strategy),
        "max_drawdown": max_drawdown(strategy),
        "mean_daily_return": float(np.mean(strategy)),
        "annualised_return": float(np.mean(strategy) * 252),
        "annualised_volatility": float(np.std(strategy, ddof=1) * np.sqrt(252)),
    }
    return {
        "rows": int(len(frame)),
        "first_date": str(frame["date"].iloc[0].date()),
        "last_date": str(frame["date"].iloc[-1].date()),
        "last_close": float(frame["close"].iloc[-1]),
        "oos_start": str(dates.iloc[n_train].date()),
        "oos_end": str(dates.iloc[-1].date()),
        "n_oos": int(len(y) - n_train),
        "ljung_box": ljung,
        "feature_names": _feature_names(int(spec["lags"])),
        "models": results,
    }


def build_features(
    frame: pd.DataFrame, lags: int, vol_window: int, momentum_window: int
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    close = frame["close"].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    dates = frame["date"]
    logret = np.zeros_like(close)
    logret[1:] = np.diff(np.log(close))
    start = max(lags, vol_window, momentum_window) + 1
    rows = []
    targets = []
    target_dates = []
    for i in range(start, len(close) - 1):
        feats = [logret[i - j] for j in range(lags)]
        window = logret[i - vol_window + 1 : i + 1]
        vol = float(np.std(window, ddof=1))
        momentum = float(np.sum(logret[i - momentum_window + 1 : i + 1]))
        vol_window_values = volume[i - vol_window + 1 : i + 1]
        volume_z = float(
            (volume[i] - vol_window_values.mean()) / (vol_window_values.std(ddof=1) + 1e-12)
        )
        intraday_range = float((high[i] - low[i]) / close[i])
        feats.extend([vol, momentum, volume_z, intraday_range])
        rows.append(feats)
        targets.append(logret[i + 1])
        target_dates.append(dates.iloc[i + 1])
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        pd.Series(target_dates),
    )


def _feature_names(lags: int) -> list[str]:
    return [f"ret_lag_{j + 1}" for j in range(lags)] + [
        "vol_20",
        "momentum_20",
        "volume_z_20",
        "intraday_range",
    ]


def _split(x: np.ndarray, y: np.ndarray, initial_fraction: float, refit_every: int):
    n_train = int(initial_fraction * len(y))
    for start in range(n_train, len(y), refit_every):
        yield start, min(start + refit_every, len(y))


def _predict_zero(x: np.ndarray) -> np.ndarray:
    return np.zeros(len(x))


def _predict_mean(x: np.ndarray, y: np.ndarray, initial_fraction: float, refit: int) -> np.ndarray:
    pred = np.zeros(len(y))
    for start, end in _split(x, y, initial_fraction, refit):
        pred[start:end] = float(np.mean(y[:start]))
    return pred


def _predict_ridge(
    x: np.ndarray, y: np.ndarray, initial_fraction: float, refit: int, alphas: list[float]
) -> np.ndarray:
    pred = np.zeros(len(y))
    for start, end in _split(x, y, initial_fraction, refit):
        xt, yt = x[:start], y[:start]
        mean, std = xt.mean(axis=0), xt.std(axis=0) + 1e-12
        xt = (xt - mean) / std
        best_alpha, best_mse = alphas[0], np.inf
        val_start = max(int(0.8 * len(yt)), 40)
        x_tr, y_tr = xt[:val_start], yt[:val_start]
        x_val, y_val = xt[val_start:], yt[val_start:]
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            model.fit(x_tr, y_tr)
            mse = float(np.mean((model.predict(x_val) - y_val) ** 2))
            if mse < best_mse:
                best_alpha, best_mse = alpha, mse
        model = Ridge(alpha=best_alpha)
        model.fit(xt, yt)
        pred[start:end] = model.predict((x[start:end] - mean) / std)
    return pred


def _predict_gd(
    x: np.ndarray, y: np.ndarray, initial_fraction: float, refit: int, spec: dict
) -> np.ndarray:
    pred = np.zeros(len(y))
    for start, end in _split(x, y, initial_fraction, refit):
        xt, yt = x[:start], y[:start]
        mean, std = xt.mean(axis=0), xt.std(axis=0) + 1e-12
        xt = (xt - mean) / std
        model = LinearRegressor(xt.shape[1], seed=start)
        model.fit(
            xt,
            yt,
            epochs=int(spec["gd_epochs"]),
            lr=float(spec["gd_learning_rate"]),
            l2=float(spec["gd_l2"]),
        )
        pred[start:end] = model.predict((x[start:end] - mean) / std)
    return pred


def evaluate(y: np.ndarray, pred: np.ndarray, cost: float, spec: dict) -> dict:
    residual = y - pred
    ss_res = float(np.sum(residual**2))
    ss_zero = float(np.sum(y**2))
    r2_zero = 1.0 - ss_res / ss_zero if ss_zero > 0 else float("nan")
    ci = _block_bootstrap_r2(
        y, pred, block=21, n_resamples=int(spec["bootstrap_resamples"]), seed=23
    )
    dm = diebold_mariano(residual**2, y**2)
    position = np.sign(pred)
    turnover = np.abs(np.diff(np.concatenate(([0.0], position))))
    strategy = position * y - cost * turnover
    return {
        "oos_r2_vs_zero": float(r2_zero),
        "oos_r2_vs_zero_ci95": [ci["low"], ci["high"]],
        "oos_r2_vs_test_mean": oos_r2(y, pred),
        "directional_accuracy": directional_accuracy(y, pred),
        "spearman_ic": spearman_ic(y, pred),
        "dm_vs_zero": dm,
        "mean_squared_error": float(np.mean(residual**2)),
        "zero_forecast_mse": float(np.mean(y**2)),
        "_strategy_returns": strategy,
    }


def _block_bootstrap_r2(
    y: np.ndarray, pred: np.ndarray, block: int, n_resamples: int, seed: int
) -> dict:
    n = len(y)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        y_b, p_b = y[idx], pred[idx]
        denom = float(np.sum(y_b**2))
        draws[i] = 1.0 - float(np.sum((y_b - p_b) ** 2)) / denom if denom > 0 else np.nan
    return {
        "point": float(1.0 - np.sum((y - pred) ** 2) / np.sum(y**2)),
        "low": float(np.nanquantile(draws, 0.025)),
        "high": float(np.nanquantile(draws, 0.975)),
    }


if __name__ == "__main__":
    main()
