"""E3: numerical and structural verification of attention, causality, and RoPE."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from fbench.models import (  # noqa: E402
    CausalSelfAttention,
    apply_rope,
    attention_math,
    reference_attention,
    rope_cos_sin,
    rope_frequencies,
)
from fbench.provenance import env_manifest, sha256_file, utc_now, write_json  # noqa: E402
from fbench.utils import set_seed  # noqa: E402

CFG = ROOT / "config" / "experiment.yaml"


def main() -> dict:
    with open(CFG, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    spec = cfg["e3_attention"]
    set_seed(0)

    # ------------------------------------------- fused vs explicit attention
    rng = torch.Generator().manual_seed(123)
    parity_errors = []
    for _ in range(int(spec["trials"])):
        batch, heads = int(spec["batch_size"]), int(spec["n_heads"])
        seq, dim = int(spec["seq_len"]), int(spec["head_dim"])
        q = torch.randn(batch, heads, seq, dim, generator=rng, dtype=torch.float64)
        k = torch.randn(batch, heads, seq, dim, generator=rng, dtype=torch.float64)
        v = torch.randn(batch, heads, seq, dim, generator=rng, dtype=torch.float64)
        ours = attention_math(q, k, v, causal=True)
        ref = reference_attention(q, k, v, causal=True)
        parity_errors.append(float((ours - ref).abs().max()))

    # ------------------------------------------------ module-level causality
    module = CausalSelfAttention(
        n_embd=int(spec["n_heads"]) * int(spec["head_dim"]),
        n_head=int(spec["n_heads"]),
        dropout=0.0,
        use_rope=True,
    ).double()
    x = torch.randn(
        int(spec["batch_size"]),
        int(spec["seq_len"]),
        module.qkv.in_features,
        generator=rng,
        dtype=torch.float64,
    )
    _, weights = module(x, return_attn=True)
    if weights is None:
        raise RuntimeError("attention weights missing")
    weights = weights.detach()
    future_mask = torch.triu(torch.ones_like(weights[0, 0], dtype=torch.bool), diagonal=1)
    causal_leak = float(weights[:, :, future_mask].abs().max())
    row_sums = weights.sum(dim=-1)
    row_sum_error = float((row_sums - 1.0).abs().max())

    # ------------------------------------------------ gradient check (double)
    q = torch.randn(2, 2, 5, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, 5, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 2, 5, 4, dtype=torch.float64, requires_grad=True)
    grad_ok = torch.autograd.gradcheck(
        lambda a, b, c: attention_math(a, b, c, causal=True),
        (q, k, v),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )

    # ------------------------------------------------------- RoPE properties
    seq_len, head_dim = 64, 32
    base_x = torch.randn(3, seq_len, head_dim, generator=rng, dtype=torch.float64)
    cos, sin = rope_cos_sin(seq_len, head_dim, dtype=torch.float64)
    rotated = apply_rope(base_x, cos, sin)
    norm_before = base_x.norm(dim=-1)
    norm_after = rotated.norm(dim=-1)
    rope_norm_error = float((norm_before - norm_after).abs().max())
    pair_before = base_x[..., 0::2].pow(2) + base_x[..., 1::2].pow(2)
    pair_after = rotated[..., 0::2].pow(2) + rotated[..., 1::2].pow(2)
    rope_pair_norm_error = float((pair_before - pair_after).abs().max())

    q_template = torch.randn(head_dim, generator=rng, dtype=torch.float64)
    k_template = torch.randn(head_dim, generator=rng, dtype=torch.float64)
    q_seq = q_template.unsqueeze(0).repeat(seq_len, 1)
    k_seq = k_template.unsqueeze(0).repeat(seq_len, 1)
    q_rot = apply_rope(q_seq, cos, sin)
    k_rot = apply_rope(k_seq, cos, sin)
    scores = q_rot @ k_rot.T
    # reference: score at offset d must equal <q, R_d k>
    freqs = rope_frequencies(head_dim, dtype=torch.float64)
    offsets = torch.arange(-(seq_len - 1), seq_len, dtype=torch.float64)
    angles = offsets[:, None] * freqs[None, :]
    ref_scores = []
    for d_idx in range(len(offsets)):
        c = angles[d_idx].cos()
        s = angles[d_idx].sin()
        qe, qo = q_template[0::2], q_template[1::2]
        ke, ko = k_template[0::2], k_template[1::2]
        # <q, R_d k> with R_d acting on adjacent pairs
        rotated_k_even = ke * c - ko * s
        rotated_k_odd = ke * s + ko * c
        dot = (qe * rotated_k_even + qo * rotated_k_odd).sum()
        ref_scores.append(dot)
    ref_scores = torch.stack(ref_scores)
    toeplitz_error = 0.0
    for m in range(seq_len):
        for n in range(seq_len):
            offset = n - m
            toeplitz_error = max(
                toeplitz_error,
                float((scores[m, n] - ref_scores[offset + seq_len - 1]).abs()),
            )

    # rotation must not collapse different relative distances
    distance_spread = float(torch.std(torch.stack([scores[0, d] for d in range(seq_len)])).item())

    metrics = {
        "attention_parity_max_abs_error": float(np.max(parity_errors)),
        "attention_parity_median_abs_error": float(np.median(parity_errors)),
        "causal_leak_max": causal_leak,
        "softmax_row_sum_max_abs_error": row_sum_error,
        "attention_gradcheck_pass": bool(grad_ok),
        "rope_norm_max_abs_error": rope_norm_error,
        "rope_pair_norm_max_abs_error": rope_pair_norm_error,
        "rope_toeplitz_max_deviation": toeplitz_error,
        "rope_relative_distance_score_std": distance_spread,
    }
    thresholds = {
        "H3_attention_parity": metrics["attention_parity_max_abs_error"] < 1e-6,
        "H3_causal_leak": causal_leak == 0.0,
        "H3_softmax_normalised": row_sum_error < 1e-6,
        "H3_gradcheck": bool(grad_ok),
        "H3_rope_norm": rope_norm_error < 1e-6,
        "H3_rope_toeplitz": toeplitz_error < 1e-6,
        "H3_rope_informative": distance_spread > 1e-3,
    }
    payload = {
        "experiment": "E3_attention",
        "generated_utc": utc_now(),
        "env": env_manifest(),
        "config": spec,
        "inputs": {"script_sha256": sha256_file(__file__)},
        "metrics": metrics,
        "hypothesis_tests": thresholds,
        "all_passed": bool(all(thresholds.values())),
    }
    write_json(ROOT / "results" / "e3_attention.json", payload)
    print(
        f"E3: parity={metrics['attention_parity_max_abs_error']:.2e} "
        f"leak={causal_leak:.1e} rope={toeplitz_error:.2e} pass={payload['all_passed']}"
    )
    return payload


if __name__ == "__main__":
    main()
