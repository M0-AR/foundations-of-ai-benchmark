"""Lab 2: does a transformer earn its keep only beyond the n-gram horizon?"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fbench.models import TinyGPT  # noqa: E402
from fbench.utils import set_seed  # noqa: E402

OUT = ROOT / "labs" / "results" / "lab2_locality.json"

VOCAB = 8
SEQ_LEN = 64
TRAIN_SEQS = 400
VAL_SEQS = 100
COPY_PROB = 0.85
LAGS = (1, 2, 4, 8)
NGRAM_ORDERS = (1, 2, 3)
SEEDS = (0, 1)
STEPS = 400
BATCH = 32
BLOCK = 32


def make_permutation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(VOCAB)
    while np.any(perm == np.arange(VOCAB)):
        perm = rng.permutation(VOCAB)
    return perm


def make_sequences(seed: int, n_seq: int, lag: int, perm: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.integers(0, VOCAB, size=(n_seq, SEQ_LEN))
    for t in range(lag, SEQ_LEN):
        copy = rng.random(n_seq) < COPY_PROB
        data[copy, t] = perm[data[copy, t - lag]]
    return data.astype(np.int64)


def sample_batch(data: np.ndarray, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    rows = rng.integers(0, len(data), size=BATCH)
    starts = rng.integers(0, SEQ_LEN - BLOCK, size=BATCH)
    x = np.stack(
        [data[row, start : start + BLOCK] for row, start in zip(rows, starts, strict=True)]
    )
    y = np.stack(
        [data[row, start + 1 : start + 1 + BLOCK] for row, start in zip(rows, starts, strict=True)]
    )
    return torch.from_numpy(x), torch.from_numpy(y)


def train_transformer(train: np.ndarray, seed: int) -> tuple[TinyGPT, float]:
    set_seed(seed)
    model = TinyGPT(
        vocab_size=VOCAB,
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=BLOCK,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed + 100)
    start = time.perf_counter()
    model.train()
    for _ in range(STEPS):
        x, y = sample_batch(train, rng)
        _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    elapsed = time.perf_counter() - start
    return model, elapsed


@torch.no_grad()
def transformer_bpc(model: TinyGPT, data: np.ndarray) -> float:
    model.eval()
    losses = []
    starts = (0, 8, 16, 24)
    for start in starts:
        x = torch.from_numpy(data[:, start : start + BLOCK])
        y = torch.from_numpy(data[:, start + 1 : start + 1 + BLOCK])
        _, loss, _ = model(x, y)
        losses.append(float(loss))
    return float(np.mean(losses) / math.log(2.0))


def ngram_bpc(train: np.ndarray, val: np.ndarray, order: int, alpha: float = 1.0) -> float:
    counts: dict[tuple[int, ...], np.ndarray] = defaultdict(
        lambda: np.zeros(VOCAB, dtype=np.float64)
    )
    for row in train:
        for t in range(order, SEQ_LEN):
            counts[tuple(int(v) for v in row[t - order : t])][int(row[t])] += 1.0
    total = 0.0
    n = 0
    for row in val:
        for t in range(order, SEQ_LEN):
            context = tuple(int(v) for v in row[t - order : t])
            row_counts = counts.get(context, np.zeros(VOCAB, dtype=np.float64))
            prob = (row_counts[int(row[t])] + alpha) / (row_counts.sum() + alpha * VOCAB)
            total -= math.log2(float(prob))
            n += 1
    return total / n


def main() -> dict:
    results = []
    for lag in LAGS:
        perm = make_permutation(500 + lag)
        train = make_sequences(1000 + lag, TRAIN_SEQS, lag, perm)
        val = make_sequences(2000 + lag, VAL_SEQS, lag, perm)
        ngram = {order: ngram_bpc(train, val, order) for order in NGRAM_ORDERS}
        best_order = min(ngram, key=ngram.get)
        for seed in SEEDS:
            model, elapsed = train_transformer(train, seed)
            bpc = transformer_bpc(model, val)
            result = {
                "lag": lag,
                "seed": seed,
                "transformer_bpc": bpc,
                "ngram_bpc": ngram,
                "best_ngram_bpc": ngram[best_order],
                "best_ngram_order": best_order,
                "delta_bpc": bpc - ngram[best_order],
                "train_seconds": elapsed,
            }
            results.append(result)
            print(
                f"lag={lag} seed={seed} transformer={bpc:.4f} "
                f"ngram={ngram[best_order]:.4f} delta={result['delta_bpc']:+.4f} "
                f"({elapsed:.1f}s)"
            )

    by_lag = {}
    for lag in LAGS:
        rows = [r for r in results if r["lag"] == lag]
        by_lag[str(lag)] = {
            "transformer_bpc_mean": float(np.mean([r["transformer_bpc"] for r in rows])),
            "best_ngram_bpc_mean": float(np.mean([r["best_ngram_bpc"] for r in rows])),
            "delta_bpc_mean": float(np.mean([r["delta_bpc"] for r in rows])),
            "delta_bpc_std": float(np.std([r["delta_bpc"] for r in rows], ddof=1)),
            "best_ngram_orders": [r["best_ngram_order"] for r in rows],
        }

    l1 = by_lag["1"]["delta_bpc_mean"]
    l8 = by_lag["8"]["delta_bpc_mean"]
    payload = {
        "lab": "lab2_locality",
        "config": {
            "vocab": VOCAB,
            "seq_len": SEQ_LEN,
            "train_seqs": TRAIN_SEQS,
            "val_seqs": VAL_SEQS,
            "copy_prob": COPY_PROB,
            "lags": LAGS,
            "ngram_orders": NGRAM_ORDERS,
            "seeds": SEEDS,
            "steps": STEPS,
            "batch": BATCH,
            "block": BLOCK,
        },
        "runs": results,
        "by_lag": by_lag,
        "hypothesis_tests": {
            "H8a_transformer_beats_ngram_at_lag_8": bool(l8 < 0.0),
            "H8b_advantage_larger_at_lag_8_than_lag_1": bool(l8 < l1),
        },
        "deltas": {"lag_1": l1, "lag_8": l8},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["by_lag"], indent=2))
    return payload


if __name__ == "__main__":
    main()
