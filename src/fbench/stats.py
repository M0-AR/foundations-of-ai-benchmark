"""Statistical inference utilities used by every experiment."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: Iterable[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"point": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, data.size, size=(n_resamples, data.size))
    stats_dist = statistic(data[idx], axis=1)
    alpha = 1.0 - confidence
    return {
        "point": float(statistic(data)),
        "low": float(np.quantile(stats_dist, alpha / 2.0)),
        "high": float(np.quantile(stats_dist, 1.0 - alpha / 2.0)),
        "n": int(data.size),
    }


def paired_wilcoxon(a: Iterable[float], b: Iterable[float]) -> dict[str, float]:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    diff = x - y
    diff = diff[np.isfinite(diff)]
    if diff.size == 0 or np.allclose(diff, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "n": int(diff.size)}
    result = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": int(diff.size),
    }


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float]]:
    """Holm-Bonferroni adjustment, returned in the original key order."""
    items = [(k, float(v)) for k, v in p_values.items() if np.isfinite(v)]
    if not items:
        return {}
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, dict[str, float]] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)
        adjusted[key] = {
            "p_raw": p,
            "p_holm": running,
            "reject": float(running < alpha),
            "rank": float(rank + 1),
        }
    return adjusted


def cohen_dz(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    diff = x - y
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")
    if sd == 0.0 or not np.isfinite(sd):
        return 0.0 if diff.size and float(np.mean(diff)) == 0.0 else float("nan")
    return float(np.mean(diff) / sd)


def probability_of_superiority(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)[:, None]
    y = np.asarray(list(b), dtype=float)[None, :]
    return float(np.mean((x > y) + 0.5 * (x == y)))


def kendall_tau(a: Iterable[float], b: Iterable[float]) -> float:
    result = stats.kendalltau(np.asarray(list(a)), np.asarray(list(b)))
    value = float(result.statistic)
    return 0.0 if not np.isfinite(value) else value


def newey_west_var(x: np.ndarray, max_lag: int | None = None) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float("nan")
    if max_lag is None:
        max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    centered = x - x.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    var = gamma0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        var += 2.0 * (1.0 - lag / (max_lag + 1.0)) * gamma
    return max(var, 0.0) / n


def diebold_mariano(
    loss_model: Iterable[float], loss_benchmark: Iterable[float]
) -> dict[str, float]:
    """Diebold-Mariano test with Newey-West HAC variance; negative stat favours model."""
    d = np.asarray(list(loss_model), dtype=float) - np.asarray(list(loss_benchmark), dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 10:
        return {"stat": float("nan"), "p_value": float("nan"), "mean_diff": float("nan")}
    var = newey_west_var(d)
    stat = float(d.mean() / np.sqrt(var)) if var > 0 else float("nan")
    p = float(2.0 * stats.norm.sf(abs(stat))) if np.isfinite(stat) else float("nan")
    return {"stat": stat, "p_value": p, "mean_diff": float(d.mean()), "n": int(d.size)}


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    result = stats.spearmanr(y_true, y_pred)
    return float(result.statistic)


def sharpe_ratio(returns: np.ndarray, periods: int = 252) -> float:
    returns = np.asarray(returns, dtype=float)
    sd = float(np.std(returns, ddof=1)) if returns.size > 1 else float("nan")
    if sd == 0.0 or not np.isfinite(sd):
        return float("nan")
    return float(np.mean(returns) / sd * np.sqrt(periods))


def max_drawdown(returns: np.ndarray) -> float:
    equity = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def block_bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    block: int = 21,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Moving-block bootstrap for serially dependent series."""
    x = np.asarray(values, dtype=float)
    n = x.size
    if n < block * 2:
        return {"point": float(statistic(x)), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([x[s : s + block] for s in starts])[:n]
        draws[i] = statistic(sample)
    return {
        "point": float(statistic(x)),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
    }
