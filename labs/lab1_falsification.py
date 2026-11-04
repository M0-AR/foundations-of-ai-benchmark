"""Lab 1: falsification coverage of targeted numerical and protocol detectors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fbench.stats import diebold_mariano, holm  # noqa: E402

OUT = ROOT / "labs" / "results" / "lab1_falsification.json"


def finite_difference(fn, x: float, h: float = 1e-6) -> float:
    return float((fn(x + h) - fn(x - h)) / (2.0 * h))


def derivative_probe(fn, correct, corrupted, points=(-0.7, -0.2, 0.3, 1.1)) -> dict:
    errors = []
    for x in points:
        truth = finite_difference(fn, x)
        scale = 1.0 + abs(truth)
        errors.append(abs(float(correct(x)) - truth) / scale)
        errors.append(abs(float(corrupted(x)) - truth) / scale)
    correct_error = max(errors[0::2])
    corrupted_error = max(errors[1::2])
    threshold = 1e-5
    return {
        "control_pass": bool(correct_error < threshold),
        "mutant_pass": bool(corrupted_error < threshold),
        "control_error": float(correct_error),
        "mutant_error": float(corrupted_error),
    }


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def attention_probe(variant: str) -> dict:
    rng = np.random.default_rng(7)
    t, d = 6, 8
    q = rng.normal(size=(t, d))
    k = rng.normal(size=(t, d))
    v = rng.normal(size=(t, d))
    mask = np.triu(np.ones((t, t), dtype=bool), k=1)
    scores = q @ k.T / np.sqrt(d)
    if variant == "no_scale":
        scores = q @ k.T
    if variant == "off_by_one":
        mask = np.triu(np.ones((t, t), dtype=bool), k=0)
    if variant == "no_mask":
        mask = np.zeros((t, t), dtype=bool)
    scores = np.where(mask, -np.inf, scores)
    axis = 0 if variant == "softmax_axis" else 1
    with np.errstate(invalid="ignore"):
        weights = np.exp(scores - np.max(scores, axis=axis, keepdims=True))
        weights /= np.sum(weights, axis=axis, keepdims=True)
    weights = np.nan_to_num(weights, nan=1.0 / weights.shape[axis])
    out = weights @ v
    reference_scores = q @ k.T / np.sqrt(d)
    reference_scores = np.where(
        np.triu(np.ones((t, t), dtype=bool), k=1), -np.inf, reference_scores
    )
    reference_weights = np.exp(reference_scores - np.max(reference_scores, axis=1, keepdims=True))
    reference_weights /= np.sum(reference_weights, axis=1, keepdims=True)
    reference = reference_weights @ v
    error = float(np.max(np.abs(out - reference)))
    return {
        "control_pass": True,
        "mutant_pass": bool(error < 1e-8),
        "mutant_error": error,
    }


def rope_rotate(x: np.ndarray, mode: str) -> np.ndarray:
    cos = np.cos(np.arange(x.shape[0])[:, None] * 0.1)
    sin = np.sin(np.arange(x.shape[0])[:, None] * 0.1)
    even, odd = x[:, 0::2], x[:, 1::2]
    if mode == "correct":
        return np.stack((even * cos - odd * sin, even * sin + odd * cos), axis=-1).reshape(x.shape)
    if mode == "sign":
        return np.stack((even * cos + odd * sin, -even * sin + odd * cos), axis=-1).reshape(x.shape)
    half = x.shape[1] // 2
    if mode == "half_split":
        first, second = x[:, :half], x[:, half:]
        return np.concatenate((first * cos - second * sin, first * sin + second * cos), axis=1)
    raise ValueError(mode)


def rope_probe(mode: str) -> dict:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(5, 8))
    reference = rope_rotate(x, "correct")
    out = rope_rotate(x, mode)
    error = float(np.max(np.abs(out - reference)))
    return {
        "control_pass": True,
        "mutant_pass": bool(error < 1e-8),
        "mutant_error": error,
    }


def holm_probe(variant: str) -> dict:
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    keys = [f"p{i}" for i in range(len(p))]
    correct = holm(dict(zip(keys, p, strict=True)))
    reference = np.array([correct[k]["p_holm"] for k in keys])
    if variant == "raw":
        adjusted = p
    elif variant == "no_monotonicity":
        order = np.argsort(p)
        adjusted = np.empty_like(p)
        for rank, index in enumerate(order):
            adjusted[index] = min(1.0, (len(p) - rank) * p[index])
    else:
        raise ValueError(variant)
    error = float(np.max(np.abs(adjusted - reference)))
    return {
        "control_pass": True,
        "mutant_pass": bool(error < 1e-12),
        "mutant_error": error,
    }


def oos_probe() -> dict:
    rng = np.random.default_rng(19)
    n_train, n_test, n_features = 50, 150, 20
    x = rng.normal(size=(n_train + n_test, n_features))
    y = rng.normal(size=n_train + n_test)
    coef = np.linalg.lstsq(x[:n_train], y[:n_train], rcond=None)[0]
    pred_oos = x[n_train:] @ coef
    pred_in = x[:n_train] @ coef
    r2_oos = 1.0 - np.sum((y[n_train:] - pred_oos) ** 2) / np.sum(y[n_train:] ** 2)
    r2_in = 1.0 - np.sum((y[:n_train] - pred_in) ** 2) / np.sum(y[:n_train] ** 2)
    return {
        "control_pass": bool(r2_oos <= 0.05),
        "mutant_pass": bool(r2_in <= 0.05),
        "r2_oos": float(r2_oos),
        "r2_in_sample": float(r2_in),
    }


def leakage_probe() -> dict:
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(23)
    n_train, n_test, n_features = 100, 100, 50
    x = rng.normal(size=(n_train + n_test, n_features))
    y = rng.integers(0, 2, size=n_train + n_test)
    control = LogisticRegression(max_iter=1000).fit(x[:n_train], y[:n_train])
    leak = LogisticRegression(max_iter=1000).fit(x, y)
    control_accuracy = float(np.mean(control.predict(x[n_train:]) == y[n_train:]))
    leak_accuracy = float(np.mean(leak.predict(x[n_train:]) == y[n_train:]))
    return {
        "control_pass": bool(control_accuracy <= 0.65),
        "mutant_pass": bool(leak_accuracy <= 0.65),
        "control_accuracy": control_accuracy,
        "test_accuracy": leak_accuracy,
    }


def cost_probe(variant: str) -> dict:
    rng = np.random.default_rng(29)
    returns = rng.normal(0.0, 0.01, size=500)
    positions = rng.choice([-1.0, 0.0, 1.0], size=500)
    turnover = np.abs(np.diff(np.concatenate(([0.0], positions))))
    gross = float(np.mean(positions * returns))
    net = gross - 0.0002 * float(np.mean(turnover))
    reported = gross if variant == "gross" else net
    return {
        "control_pass": bool(net < gross),
        "mutant_pass": bool(reported < gross),
        "gross": gross,
        "net": net,
        "reported": reported,
    }


def dm_probe() -> dict:
    rng = np.random.default_rng(31)
    n, reps, phi = 250, 120, 0.4
    hac_false = 0
    iid_false = 0
    for _ in range(reps):
        shocks = rng.normal(size=n)
        diff = np.empty(n)
        diff[0] = shocks[0]
        for t in range(1, n):
            diff[t] = phi * diff[t - 1] + shocks[t]
        hac_false += int(diebold_mariano(diff, np.zeros(n))["p_value"] < 0.05)
        iid_false += int(stats.ttest_1samp(diff, 0.0).pvalue < 0.05)
    hac_rate = hac_false / reps
    iid_rate = iid_false / reps
    return {
        "control_pass": bool(hac_rate <= 0.10),
        "mutant_pass": bool(iid_rate <= 0.10),
        "hac_false_positive_rate": float(hac_rate),
        "iid_false_positive_rate": float(iid_rate),
    }


def hac_envelope(reps: int = 200) -> dict:
    rng = np.random.default_rng(37)
    out = {}
    for phi in (0.0, 0.2, 0.4, 0.6, 0.8):
        hac_false = 0
        iid_false = 0
        for _ in range(reps):
            shocks = rng.normal(size=250)
            diff = np.empty(250)
            diff[0] = shocks[0]
            for t in range(1, 250):
                diff[t] = phi * diff[t - 1] + shocks[t]
            hac_false += int(diebold_mariano(diff, np.zeros(250))["p_value"] < 0.05)
            iid_false += int(stats.ttest_1samp(diff, 0.0).pvalue < 0.05)
        out[str(phi)] = {
            "hac_false_positive_rate": hac_false / reps,
            "iid_false_positive_rate": iid_false / reps,
        }
    return out


def protocol_audit(record: dict) -> list[str]:
    violations = []
    if record["reported_seeds"] != record["preregistered_seeds"]:
        violations.append("seed_count_mismatch")
    if record["reported_n"] != record["preregistered_n"] or record["peeked"]:
        violations.append("optional_stopping")
    if record["test_evaluations"] > 1:
        violations.append("test_set_reuse")
    if record["reported_family"] != record["preregistered_family"]:
        violations.append("multiplicity_family_mismatch")
    if record["reported_hypotheses"] != record["preregistered_hypotheses"]:
        violations.append("endpoint_omission")
    return violations


def protocol_probe(variant: str) -> dict:
    base = {
        "preregistered_seeds": 10,
        "reported_seeds": 10,
        "preregistered_n": 200,
        "reported_n": 200,
        "peeked": False,
        "test_evaluations": 1,
        "preregistered_family": "all_pairs_30",
        "reported_family": "all_pairs_30",
        "preregistered_hypotheses": ["H1", "H2", "H3", "H4", "H5", "H6"],
        "reported_hypotheses": ["H1", "H2", "H3", "H4", "H5", "H6"],
    }
    record = dict(base)
    if variant == "seed_cherry_pick":
        record["reported_seeds"] = 1
    elif variant == "optional_stopping_honest":
        record["reported_n"] = 120
        record["peeked"] = True
    elif variant == "optional_stopping_concealed":
        pass
    elif variant == "test_reuse":
        record["test_evaluations"] = 2
    elif variant == "family_switch":
        record["reported_family"] = "per_dataset"
    elif variant == "endpoint_drop":
        record["reported_hypotheses"] = ["H1", "H2", "H3", "H4", "H5"]
    else:
        raise ValueError(variant)
    violations = protocol_audit(record)
    return {
        "control_pass": True,
        "mutant_pass": bool(not violations),
        "violations": violations,
    }


def main() -> dict:
    probes = []
    probes.append(
        {
            "family": "autograd",
            "name": "mul_wrong_factor",
            **derivative_probe(lambda x: 2.3 * x, lambda x: 2.3, lambda x: 1.0 / 2.3),
        }
    )
    probes.append(
        {
            "family": "autograd",
            "name": "exp_missing_chain",
            **derivative_probe(np.exp, np.exp, lambda x: 1.0),
        }
    )
    probes.append(
        {
            "family": "autograd",
            "name": "sigmoid_wrong_sign",
            **derivative_probe(
                sigmoid,
                lambda x: sigmoid(x) * (1.0 - sigmoid(x)),
                lambda x: sigmoid(x) * (1.0 + sigmoid(x)),
            ),
        }
    )
    probes.append(
        {
            "family": "autograd",
            "name": "relu_all_ones",
            **derivative_probe(
                lambda x: max(0.0, x),
                lambda x: 1.0 if x > 0 else 0.0,
                lambda x: 1.0,
            ),
        }
    )
    probes.append(
        {
            "family": "autograd",
            "name": "power_forgot_exponent",
            **derivative_probe(lambda x: x**3, lambda x: 3.0 * x**2, lambda x: 3.0 * x),
        }
    )
    probes.append(
        {
            "family": "autograd",
            "name": "tanh_missing_square",
            **derivative_probe(
                np.tanh, lambda x: 1.0 - np.tanh(x) ** 2, lambda x: 1.0 - np.tanh(x)
            ),
        }
    )
    for variant in ("no_scale", "no_mask", "off_by_one", "softmax_axis"):
        probes.append({"family": "attention", "name": variant, **attention_probe(variant)})
    for variant in ("sign", "half_split"):
        probes.append({"family": "attention", "name": f"rope_{variant}", **rope_probe(variant)})
    for variant in ("raw", "no_monotonicity"):
        probes.append({"family": "inference", "name": f"holm_{variant}", **holm_probe(variant)})
    probes.append({"family": "inference", "name": "oos_in_sample", **oos_probe()})
    probes.append({"family": "inference", "name": "test_set_leakage", **leakage_probe()})
    probes.append({"family": "inference", "name": "cost_omitted", **cost_probe("gross")})
    probes.append({"family": "inference", "name": "dm_iid_variance", **dm_probe()})
    for variant in (
        "seed_cherry_pick",
        "optional_stopping_honest",
        "optional_stopping_concealed",
        "test_reuse",
        "family_switch",
        "endpoint_drop",
    ):
        probes.append({"family": "protocol", "name": variant, **protocol_probe(variant)})

    for probe in probes:
        probe["killed"] = bool(probe["control_pass"] and not probe["mutant_pass"])
        probe["valid"] = bool(probe["control_pass"])

    by_family: dict[str, dict] = {}
    for family in ("autograd", "attention", "inference", "protocol"):
        rows = [p for p in probes if p["family"] == family]
        killed = sum(p["killed"] for p in rows)
        by_family[family] = {
            "n": len(rows),
            "killed": killed,
            "coverage": killed / len(rows),
            "survivors": [p["name"] for p in rows if not p["killed"]],
        }
    n = len(probes)
    killed = sum(p["killed"] for p in probes)
    coverage = killed / n
    numerical = by_family["autograd"]["coverage"]
    numerical = (
        by_family["autograd"]["killed"]
        + by_family["attention"]["killed"]
        + by_family["inference"]["killed"]
    ) / 18
    payload = {
        "lab": "lab1_falsification",
        "n_probes": n,
        "killed": killed,
        "survived": n - killed,
        "coverage": coverage,
        "by_family": by_family,
        "hypothesis_tests": {
            "H7a_coverage_ge_0_80": bool(coverage >= 0.80),
            "H7b_numerical_ge_protocol": bool(numerical >= by_family["protocol"]["coverage"]),
        },
        "numerical_coverage": numerical,
        "protocol_coverage": by_family["protocol"]["coverage"],
        "survivors": [p["name"] for p in probes if not p["killed"]],
        "exploratory_hac_envelope": hac_envelope(),
        "probes": probes,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: payload[k] for k in ("killed", "n_probes", "coverage", "survivors")}, indent=2
        )
    )
    return payload


if __name__ == "__main__":
    main()
