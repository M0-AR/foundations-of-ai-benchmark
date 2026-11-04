"""Subprocess worker for Lab 3; environment is set by the parent before import."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def numpy_workload(seed: int) -> dict:
    from fbench.nn import LinearRegressor

    rng = np.random.default_rng(seed)
    n, p = 2000, 256
    x = rng.normal(size=(n, p))
    w = rng.normal(size=p)
    y = x @ w + rng.normal(scale=0.1, size=n)
    model = LinearRegressor(p, seed=seed)
    model.fit(x[:1500], y[:1500], epochs=200, lr=0.01, l2=1e-4)
    pred = model.predict(x[1500:])
    loss = float(np.mean((pred - y[1500:]) ** 2))
    params = np.concatenate((model.w, [model.b]))
    return {
        "workload": "numpy",
        "seed": seed,
        "loss": loss,
        "param_hash": digest(params),
        "param_norm": float(np.linalg.norm(params)),
    }


def torch_workload(seed: int, threads: int) -> dict:
    import torch

    from fbench.models import TinyGPT

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 32, size=(1000, 64)).astype(np.int64)
    model = TinyGPT(vocab_size=32, n_layer=2, n_head=2, n_embd=32, block_size=32, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    model.train()
    for _ in range(30):
        rows = rng.integers(0, len(data), size=16)
        starts = rng.integers(0, 32, size=16)
        x = torch.from_numpy(
            np.stack([data[r, s : s + 32] for r, s in zip(rows, starts, strict=True)])
        )
        y = torch.from_numpy(
            np.stack([data[r, s + 1 : s + 33] for r, s in zip(rows, starts, strict=True)])
        )
        _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        rows = np.arange(16)
        x = torch.from_numpy(data[rows, :32])
        y = torch.from_numpy(data[rows, 1:33])
        _, loss, _ = model(x, y)
    flat = torch.cat([p.detach().reshape(-1).float() for p in model.parameters()]).numpy()
    return {
        "workload": "torch",
        "seed": seed,
        "loss": float(loss),
        "param_hash": digest(flat),
        "param_norm": float(np.linalg.norm(flat)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=("numpy", "torch"), required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    np.random.seed(args.seed)
    if args.workload == "numpy":
        result = numpy_workload(args.seed)
    else:
        result = torch_workload(args.seed, args.threads)
    result["threads"] = args.threads
    print(json.dumps(result))


if __name__ == "__main__":
    main()
