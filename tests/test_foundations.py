"""Fast verification tests for the mathematical and model foundations."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fbench.autograd import expression
from fbench.calculus import convergence_table
from fbench.data import load_tabular, load_tinyshakespeare
from fbench.models import (
    TinyGPT,
    apply_rope,
    attention_math,
    reference_attention,
    rope_cos_sin,
)
from fbench.nn import MLPClassifier, numerical_gradient_check
from fbench.stats import bootstrap_ci, diebold_mariano, holm, paired_wilcoxon


def test_calculus_converges_second_order():
    table = convergence_table(
        h_values=[0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
        points=[-1.0, 0.5, 2.0],
    )
    slopes = []
    for row in table.values():
        h = np.asarray(row["h_values"])
        err = np.asarray(row["mean_abs_errors"])
        slopes.append(np.polyfit(np.log(h), np.log(err), 1)[0])
    assert np.median(slopes) > 1.8


def test_scalar_autograd_matches_finite_difference():
    h = 1e-6
    out, x, y = expression(0.37, -0.41, return_inputs=True)
    out.backward()
    fd_x = (expression(0.37 + h, -0.41).data - expression(0.37 - h, -0.41).data) / (2 * h)
    fd_y = (expression(0.37, -0.41 + h).data - expression(0.37, -0.41 - h).data) / (2 * h)
    assert x.grad == pytest.approx(fd_x, rel=1e-4, abs=1e-6)
    assert y.grad == pytest.approx(fd_y, rel=1e-4, abs=1e-6)


def test_numpy_mlp_gradients_match_finite_difference():
    data = load_tabular("breast_cancer")
    X, y = data.X[:80], data.y[:80]
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    model = MLPClassifier(X.shape[1], 12, 2, activation="tanh", seed=1)
    result = numerical_gradient_check(model, X, y, n_checks=12, eps=1e-5, seed=2)
    assert result["max_relative_error"] < 1e-5
    assert result["cosine_similarity"] > 0.999999


def test_attention_matches_pytorch_and_is_causal():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 7, 5, dtype=torch.float64)
    k = torch.randn(2, 3, 7, 5, dtype=torch.float64)
    v = torch.randn(2, 3, 7, 5, dtype=torch.float64)
    ours = attention_math(q, k, v)
    ref = reference_attention(q, k, v)
    assert torch.allclose(ours, ref, atol=1e-12)
    # changing a future token must not affect earlier outputs
    v2 = v.clone()
    v2[:, :, 5:, :] += 10.0
    assert torch.allclose(attention_math(q, k, v2)[:, :, :5], ours[:, :, :5], atol=1e-12)


def test_rope_preserves_norms_and_relative_structure():
    torch.manual_seed(0)
    x = torch.randn(2, 9, 8, dtype=torch.float64)
    cos, sin = rope_cos_sin(9, 8, dtype=torch.float64)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-12)
    q = torch.randn(8, dtype=torch.float64)
    k = torch.randn(8, dtype=torch.float64)
    q_seq, k_seq = q.repeat(9, 1), k.repeat(9, 1)
    scores = apply_rope(q_seq, cos, sin) @ apply_rope(k_seq, cos, sin).T
    # Toeplitz: score(m, n) depends only on |n - m|
    for m in range(9):
        for n in range(9):
            d = n - m
            freqs = 1.0 / (10_000 ** (torch.arange(0, 8, 2, dtype=torch.float64) / 8))
            angle = freqs * d
            c, s = angle.cos(), angle.sin()
            ke, ko = k[0::2], k[1::2]
            rk = torch.stack((ke * c - ko * s, ke * s + ko * c), dim=-1).flatten()
            assert scores[m, n].item() == pytest.approx(float(q @ rk), abs=1e-12)


def test_tiny_gpt_forward_and_causal_loss():
    torch.manual_seed(0)
    model = TinyGPT(vocab_size=17, n_layer=2, n_head=2, n_embd=16, block_size=8)
    idx = torch.randint(0, 17, (2, 8))
    targets = torch.randint(0, 17, (2, 8))
    logits, loss, _ = model(idx, targets)
    assert logits.shape == (2, 8, 17)
    assert math.isfinite(float(loss.detach()))
    idx2 = idx.clone()
    idx2[:, 5:] = (idx2[:, 5:] + 1) % 17
    logits2, _, _ = model(idx2)
    assert torch.allclose(logits[:, :5], logits2[:, :5], atol=1e-6)
    generated = model.generate(idx[:, :2], max_new_tokens=4)
    assert generated.shape == (2, 6)


def test_statistics_utilities():
    ci = bootstrap_ci(np.arange(20.0), statistic=np.mean, n_resamples=500, seed=0)
    assert ci["low"] < ci["point"] < ci["high"]
    assert paired_wilcoxon([1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0])["p_value"] < 0.05
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"]["reject"] and adjusted["c"]["p_holm"] <= 1.0
    rng = np.random.default_rng(0)
    benchmark = rng.uniform(1.0, 2.0, size=200)
    model_loss = benchmark - rng.uniform(0.2, 0.8, size=200)
    dm = diebold_mariano(model_loss, benchmark)
    assert dm["stat"] < 0  # model has lower loss than benchmark


def test_tiny_shakespeare_integrity():
    dataset = load_tinyshakespeare()
    assert len(dataset.text) == 1_115_394
    assert dataset.vocab_size == 65
    assert dataset.sha256 == "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
    assert len(dataset.train_ids) + len(dataset.val_ids) == len(dataset.text)
