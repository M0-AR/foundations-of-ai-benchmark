"""Calculus utilities: analytic derivatives and central finite differences."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

FUNCTIONS: dict[str, tuple[Callable[[float], float], Callable[[float], float]]] = {
    "x_squared": (lambda x: x * x, lambda x: 2.0 * x),
    "x_cubed": (lambda x: x**3, lambda x: 3.0 * x * x),
    "linear_2x_plus_3": (lambda x: 2.0 * x + 3.0, lambda x: 2.0),
    "sin": (math.sin, math.cos),
    "exp": (math.exp, math.exp),
    "sigmoid": (
        lambda x: 1.0 / (1.0 + math.exp(-x)),
        lambda x: (1.0 / (1.0 + math.exp(-x))) * (1.0 - 1.0 / (1.0 + math.exp(-x))),
    ),
}


def central_difference(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


def convergence_table(h_values: list[float], points: list[float]) -> dict[str, dict[str, object]]:
    """Measure absolute error of central differences against analytic derivatives."""
    table: dict[str, dict[str, object]] = {}
    for name, (f, df) in FUNCTIONS.items():
        errors: list[float] = []
        for h in h_values:
            trial = [abs(central_difference(f, x, h) - df(x)) for x in points]
            errors.append(float(np.mean(trial)))
        best_idx = int(np.argmin(errors))
        mask = np.array(errors) > 0
        slope = float("nan")
        if mask.sum() >= 3:
            slope = float(
                np.polyfit(
                    np.log(np.array(h_values)[mask]),
                    np.log(np.array(errors)[mask]),
                    1,
                )[0]
            )
        table[name] = {
            "h_values": list(map(float, h_values)),
            "mean_abs_errors": errors,
            "best_h": float(h_values[best_idx]),
            "best_error": float(errors[best_idx]),
            "loglog_slope_all": slope,
            "loglog_slope_tail": _tail_slope(h_values, errors, n=4),
        }
    return table


def _tail_slope(h_values: list[float], errors: list[float], n: int = 4) -> float:
    h = np.asarray(h_values[:n], dtype=float)
    e = np.asarray(errors[:n], dtype=float)
    mask = e > 0
    if mask.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(h[mask]), np.log(e[mask]), 1)[0])
