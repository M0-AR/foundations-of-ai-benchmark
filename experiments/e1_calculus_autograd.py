"""E1: calculus convergence and gradient fidelity (finite differences, autograd, PyTorch)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fbench.autograd import expression  # noqa: E402
from fbench.calculus import convergence_table  # noqa: E402
from fbench.data import load_tabular  # noqa: E402
from fbench.nn import MLPClassifier, numerical_gradient_check  # noqa: E402
from fbench.provenance import env_manifest, sha256_file, utc_now, write_json  # noqa: E402
from fbench.utils import set_seed  # noqa: E402

CFG = ROOT / "config" / "experiment.yaml"
import yaml  # noqa: E402


def main() -> dict:
    with open(CFG, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    spec = cfg["e1_calculus"]
    set_seed(0)

    convergence = convergence_table(
        h_values=[float(h) for h in spec["h_values"]],
        points=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
    )
    truncation_slopes = []
    for row in convergence.values():
        h = np.asarray(row["h_values"])
        err = np.asarray(row["mean_abs_errors"])
        mask = h >= 1e-3
        slope = float(np.polyfit(np.log(h[mask]), np.log(err[mask]), 1)[0])
        row["loglog_slope_truncation"] = slope
        truncation_slopes.append(slope)
    median_slope = float(np.median(truncation_slopes))

    # ------------------------------------------------ scalar autograd fidelity
    rng = np.random.default_rng(2026)
    autograd_vs_fd = []
    autograd_vs_torch = []
    h = 1e-6
    for _ in range(int(spec["autograd_trials"])):
        a = float(rng.uniform(-1.5, 1.5))
        b = float(rng.uniform(-1.5, 1.5))
        out, x_val, y_val = expression(a, b, return_inputs=True)
        out.backward()
        ga, gb = x_val.grad, y_val.grad
        # finite differences of the same expression
        plus_a = expression(a + h, b)
        minus_a = expression(a - h, b)
        fd_a = (plus_a.data - minus_a.data) / (2 * h)
        plus_b = expression(a, b + h)
        minus_b = expression(a, b - h)
        fd_b = (plus_b.data - minus_b.data) / (2 * h)
        autograd_vs_fd.append(max(_rel(ga, fd_a), _rel(gb, fd_b)))
        # PyTorch reference
        ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
        z = (ta * tb + ta**2).tanh()
        positive = (z * z + 1.0) * (tb**3).exp()
        loss = positive.log() + (ta / (tb * tb + 1.0)).sigmoid()
        loss.backward()
        autograd_vs_torch.append(max(_rel(ga, float(ta.grad)), _rel(gb, float(tb.grad))))

    # --------------------------------------------------- NumPy MLP gradient check
    data = load_tabular("breast_cancer")
    X = data.X[:120]
    y = data.y[:120]
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-12
    X = (X - mean) / std
    mlp = MLPClassifier(n_features=X.shape[1], hidden=16, n_classes=2, activation="tanh", seed=7)
    fd_check = numerical_gradient_check(mlp, X, y, n_checks=20, eps=1e-5, seed=3)

    # PyTorch reference on exactly the same weights
    torch_mlp = _torch_mirror(mlp)
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    logits = Xt @ torch_mlp["W1"] + torch_mlp["b1"]
    a1 = torch.tanh(logits)
    out = a1 @ torch_mlp["W2"] + torch_mlp["b2"]
    loss = torch.nn.functional.cross_entropy(out, yt)
    params = list(torch_mlp.values())
    grads = torch.autograd.grad(loss, params)
    torch_flat = np.concatenate([g.detach().numpy().ravel() for g in grads])
    numpy_flat = mlp.flat_grads(X, y)
    mlp_vs_torch = {
        "max_abs_error": float(np.max(np.abs(torch_flat - numpy_flat))),
        "max_relative_error": float(
            np.max(np.abs(torch_flat - numpy_flat) / np.maximum(1.0, np.abs(torch_flat)))
        ),
        "cosine_similarity": float(
            np.dot(torch_flat, numpy_flat)
            / (np.linalg.norm(torch_flat) * np.linalg.norm(numpy_flat) + 1e-15)
        ),
    }

    metrics = {
        "calculus": convergence,
        "median_loglog_slope_truncation": median_slope,
        "autograd_vs_finite_difference": {
            "median_max_relative_error": float(np.median(autograd_vs_fd)),
            "p95_max_relative_error": float(np.quantile(autograd_vs_fd, 0.95)),
            "max_relative_error": float(np.max(autograd_vs_fd)),
            "trials": int(len(autograd_vs_fd)),
        },
        "autograd_vs_pytorch": {
            "median_max_relative_error": float(np.median(autograd_vs_torch)),
            "p95_max_relative_error": float(np.quantile(autograd_vs_torch, 0.95)),
            "max_relative_error": float(np.max(autograd_vs_torch)),
        },
        "numpy_mlp_vs_finite_difference": fd_check,
        "numpy_mlp_vs_pytorch": mlp_vs_torch,
    }
    thresholds = {
        "H1_autograd_fd": metrics["autograd_vs_finite_difference"]["max_relative_error"] < 1e-5,
        "H1_numpy_mlp_fd": fd_check["max_relative_error"] < 1e-5,
        "H1_numpy_mlp_torch": mlp_vs_torch["max_relative_error"] < 1e-5,
        "H2_convergence": median_slope >= 1.8,
    }
    payload = {
        "experiment": "E1_calculus_autograd",
        "generated_utc": utc_now(),
        "env": env_manifest(),
        "inputs": {
            "tabular_dataset": data.name,
            "tabular_sha256": data.sha256,
            "script_sha256": sha256_file(__file__),
        },
        "metrics": metrics,
        "hypothesis_tests": thresholds,
        "all_passed": bool(all(thresholds.values())),
    }
    write_json(ROOT / "results" / "e1_calculus_autograd.json", payload)
    print(
        f"E1: slope={median_slope:.3f} fd_err={fd_check['max_relative_error']:.2e} "
        f"torch_err={mlp_vs_torch['max_relative_error']:.2e} pass={payload['all_passed']}"
    )
    return payload


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(a) + abs(b))


def _torch_mirror(mlp: MLPClassifier) -> dict[str, torch.Tensor]:
    return {
        key: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for key, value in mlp.params.items()
    }


if __name__ == "__main__":
    main()
