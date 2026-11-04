"""E4: char-level TinyGPT versus tuned interpolated n-gram baselines on Tiny Shakespeare."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from fbench.data import get_batch, load_tinyshakespeare  # noqa: E402
from fbench.models import TinyGPT  # noqa: E402
from fbench.provenance import env_manifest, sha256_file, utc_now, write_json  # noqa: E402
from fbench.stats import bootstrap_ci  # noqa: E402
from fbench.utils import set_seed  # noqa: E402

CFG = ROOT / "config" / "experiment.yaml"


def main() -> dict:
    with open(CFG, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    spec = cfg["e4_gpt"]
    dataset = load_tinyshakespeare(cfg)
    vocab = dataset.vocab_size
    print(
        f"E4 corpus: {len(dataset.text):,} chars, vocab={vocab}, "
        f"train={len(dataset.train_ids):,} val={len(dataset.val_ids):,}"
    )

    baselines = ngram_baselines(dataset)
    print("E4 baselines: " + ", ".join(f"{k}={v['bpc']:.3f}" for k, v in baselines.items()))

    device = torch.device("cpu")
    seeds = [int(s) for s in spec["seeds"]]
    runs = []
    for seed in seeds:
        set_seed(seed)
        model = TinyGPT(
            vocab_size=vocab,
            n_layer=int(spec["n_layer"]),
            n_head=int(spec["n_head"]),
            n_embd=int(spec["n_embd"]),
            block_size=int(spec["block_size"]),
            dropout=float(spec["dropout"]),
        ).to(device)
        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=float(spec["learning_rate"]),
            weight_decay=float(spec["weight_decay"]),
        )
        rng = np.random.default_rng(seed)
        eval_batches = _fixed_eval_batches(
            dataset.val_ids,
            int(spec["batch_size"]),
            int(spec["block_size"]),
            int(spec["eval_batches"]),
            seed=999,
        )
        curve = []
        start = time.perf_counter()
        for step in range(1, int(spec["max_steps"]) + 1):
            x, y = get_batch(
                dataset.train_ids, int(spec["batch_size"]), int(spec["block_size"]), rng
            )
            logits, loss, _ = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            if step % int(spec["eval_interval"]) == 0 or step == 1:
                ev = evaluate(model, eval_batches, device)
                curve.append({"step": step, "val_ce": ev["ce"], "val_bpc": ev["bpc"]})
                print(
                    f"  seed {seed} step {step:4d} train_ce={loss.item():.4f} "
                    f"val_bpc={ev['bpc']:.4f}"
                )
        train_seconds = time.perf_counter() - start
        final = evaluate(model, eval_batches, device)
        attn = attention_diagnostics(model, eval_batches[0], device)
        generator = torch.Generator().manual_seed(seed)
        prompt = torch.tensor([[dataset.stoi[c] for c in "ROMEO: "]], dtype=torch.long)
        sample_ids = model.generate(
            prompt,
            max_new_tokens=int(spec["sample_chars"]),
            temperature=0.8,
            top_k=50,
            generator=generator,
        )[0].tolist()
        sample = "".join(dataset.itos[i] for i in sample_ids)
        runs.append(
            {
                "seed": seed,
                "val_ce": final["ce"],
                "val_bpc": final["bpc"],
                "val_bpc_ci95": final["bpc_ci95"],
                "per_batch_ce": final["per_batch_ce"],
                "curve": curve,
                "train_seconds": train_seconds,
                "parameters": model.num_parameters(),
                "parameters_non_embedding": model.num_parameters(non_embedding=True),
                "attention": attn,
                "sample": sample,
            }
        )
        print(
            f"E4 seed {seed}: bpc={final['bpc']:.4f} "
            f"ci={final['bpc_ci95'][0]:.4f}-{final['bpc_ci95'][1]:.4f} "
            f"time={train_seconds:.1f}s"
        )

    bpc_values = np.array([r["val_bpc"] for r in runs])
    mean_bpc = float(bpc_values.mean())
    primary = baselines["trigram_add1"]
    strongest = min(baselines.values(), key=lambda item: item["bpc"])
    improvement = float(primary["bpc"] - mean_bpc)
    thresholds = {
        "H5_transformer_beats_smoothed_trigram": mean_bpc < primary["bpc"],
        "H5_seed_consistency": bool(np.all(bpc_values < primary["bpc"])),
        "H5_improvement_at_least_5pct": improvement / primary["bpc"] >= 0.05,
    }
    metrics = {
        "runs": runs,
        "aggregate": {
            "bpc_mean": mean_bpc,
            "bpc_std": float(bpc_values.std(ddof=1)),
            "bpc_min": float(bpc_values.min()),
            "bpc_max": float(bpc_values.max()),
            "parameters": runs[0]["parameters"],
            "train_seconds_mean": float(np.mean([r["train_seconds"] for r in runs])),
        },
        "baselines": baselines,
        "primary_baseline": primary["name"],
        "primary_baseline_bpc": primary["bpc"],
        "strongest_baseline": strongest["name"],
        "strongest_baseline_bpc": strongest["bpc"],
        "absolute_bpc_improvement_vs_primary": improvement,
        "relative_bpc_improvement_vs_primary": improvement / primary["bpc"],
        "delta_bpc_vs_strongest": float(mean_bpc - strongest["bpc"]),
    }
    payload = {
        "experiment": "E4_gpt",
        "generated_utc": utc_now(),
        "env": env_manifest(),
        "config": spec,
        "inputs": {
            "corpus_sha256": dataset.sha256,
            "corpus_chars": len(dataset.text),
            "vocab_size": vocab,
            "script_sha256": sha256_file(__file__),
        },
        "metrics": metrics,
        "hypothesis_tests": thresholds,
        "all_passed": bool(all(thresholds.values())),
    }
    write_json(ROOT / "results" / "e4_gpt.json", payload)
    print(f"E4 pass={payload['all_passed']} mean_bpc={mean_bpc:.4f}")
    return payload


def _fixed_eval_batches(
    ids: np.ndarray, batch_size: int, block_size: int, n_batches: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [get_batch(ids, batch_size, block_size, rng) for _ in range(n_batches)]


@torch.no_grad()
def evaluate(
    model: TinyGPT, batches: list[tuple[np.ndarray, np.ndarray]], device: torch.device
) -> dict:
    model.eval()
    per_batch = []
    for x, y in batches:
        _, loss, _ = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
        per_batch.append(float(loss.item()))
    model.train()
    per_batch_arr = np.array(per_batch)
    ce = float(per_batch_arr.mean())
    ci = bootstrap_ci(per_batch_arr, statistic=np.mean, n_resamples=5000, seed=11)
    return {
        "ce": ce,
        "bpc": ce / math.log(2),
        "bpc_ci95": [ci["low"] / math.log(2), ci["high"] / math.log(2)],
        "per_batch_ce": per_batch,
    }


@torch.no_grad()
def attention_diagnostics(
    model: TinyGPT, batch: tuple[np.ndarray, np.ndarray], device: torch.device
) -> dict:
    model.eval()
    x, _ = batch
    _, _, weights = model(torch.from_numpy(x).to(device), return_attn=True)
    entropies = []
    max_weights = []
    for layer_weights in weights:
        # (B, H, T, T) -> normalised entropy per query position
        p = layer_weights.clamp_min(1e-12)
        entropy = -(p * p.log()).sum(dim=-1) / math.log(p.size(-1))
        entropies.append(float(entropy.mean().item()))
        max_weights.append(float(layer_weights.max(dim=-1).values.mean().item()))
    model.train()
    return {
        "normalised_entropy_per_layer": entropies,
        "mean_max_attention_weight_per_layer": max_weights,
        "normalised_entropy_mean": float(np.mean(entropies)),
    }


def ngram_baselines(dataset) -> dict:
    """Tuned Jelinek-Mercer interpolation of unigram/bigram/trigram models."""
    train = dataset.train_ids
    val = dataset.val_ids
    vocab = dataset.vocab_size
    # hold out the last 10% of train for weight tuning (never validation)
    cut = int(0.9 * len(train))
    fit_ids, tune_ids = train[:cut], train[cut:]

    counts1 = np.bincount(fit_ids, minlength=vocab).astype(np.float64)
    counts2 = (
        np.bincount(fit_ids[:-1] * vocab + fit_ids[1:], minlength=vocab * vocab)
        .astype(np.float64)
        .reshape(vocab, vocab)
    )
    counts3 = (
        np.bincount(
            (fit_ids[:-2] * vocab + fit_ids[1:-1]) * vocab + fit_ids[2:],
            minlength=vocab**3,
        )
        .astype(np.float64)
        .reshape(vocab, vocab, vocab)
    )

    def probs(ids: np.ndarray, weights: tuple[float, float, float]) -> np.ndarray:
        """Probability assigned to the observed next character at every position."""
        w1, w2, w3 = weights
        a, b, c = ids[:-2], ids[1:-1], ids[2:]
        n2 = counts2.sum(axis=1)[b]
        n3 = counts3.sum(axis=-1)[a, b]
        p1 = counts1[c] / counts1.sum()
        p2 = np.divide(
            counts2[b, c],
            n2,
            out=np.zeros(len(c), dtype=np.float64),
            where=n2 > 0,
        )
        p3 = np.divide(
            counts3[a, b, c],
            n3,
            out=np.zeros(len(c), dtype=np.float64),
            where=n3 > 0,
        )
        return w1 * p1 + w2 * p2 + w3 * p3

    best = None
    grid = np.arange(0.0, 1.0001, 0.1)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            w3 = max(w3, 0.0)
            if w1 + w2 + w3 <= 0:
                continue
            p = probs(tune_ids, (w1, w2, w3))
            bpc = float(-np.log2(np.clip(p, 1e-12, 1.0)).mean())
            if best is None or bpc < best["bpc"]:
                best = {"bpc": bpc, "weights": [float(w1), float(w2), float(w3)]}
    assert best is not None

    def evaluate_weights(ids: np.ndarray, weights) -> dict:
        p = probs(ids, weights)
        nll = -np.log2(np.clip(p, 1e-12, 1.0))
        block = 128
        n_blocks = len(nll) // block
        block_means = nll[: n_blocks * block].reshape(n_blocks, block).mean(axis=1)
        ci = bootstrap_ci(block_means, statistic=np.mean, n_resamples=5000, seed=13)
        return {
            "bpc": float(nll.mean()),
            "bpc_ci95": [float(ci["low"]), float(ci["high"])],
            "weights": list(best["weights"]),
        }

    return {
        "uniform": {
            "name": "uniform",
            "bpc": math.log2(vocab),
            "bpc_ci95": [math.log2(vocab), math.log2(vocab)],
            "weights": [1.0, 0.0, 0.0],
        },
        "unigram": {
            "name": "unigram",
            "bpc": float(-np.log2(np.clip(counts1[val] / counts1.sum(), 1e-12, 1.0)).mean()),
            "bpc_ci95": _unigram_ci(counts1, val),
            "weights": [1.0, 0.0, 0.0],
        },
        "trigram_add1": {
            "name": "trigram_add1",
            **_add_one_trigram(val, counts3),
        },
        "interpolated_ngram": {
            "name": "interpolated_ngram",
            **evaluate_weights(val, tuple(best["weights"])),
            "tuning_bpc": best["bpc"],
        },
    }


def _add_one_trigram(val: np.ndarray, counts3: np.ndarray) -> dict:
    vocab = counts3.shape[0]
    a, b, c = val[:-2], val[1:-1], val[2:]
    rows = counts3.reshape(-1, vocab)[a * vocab + b]
    totals = rows.sum(axis=1)
    p = (rows[np.arange(len(c)), c] + 1.0) / (totals + vocab)
    nll = -np.log2(np.clip(p, 1e-12, 1.0))
    block = 128
    n_blocks = len(nll) // block
    means = nll[: n_blocks * block].reshape(n_blocks, block).mean(axis=1)
    ci = bootstrap_ci(means, statistic=np.mean, n_resamples=5000, seed=19)
    return {
        "bpc": float(nll.mean()),
        "bpc_ci95": [float(ci["low"]), float(ci["high"])],
        "weights": [0.0, 0.0, 1.0],
        "smoothing": "add-one",
    }


def _unigram_ci(counts1: np.ndarray, val: np.ndarray) -> list[float]:
    nll = -np.log2(np.clip(counts1[val] / counts1.sum(), 1e-12, 1.0))
    block = 128
    n_blocks = len(nll) // block
    means = nll[: n_blocks * block].reshape(n_blocks, block).mean(axis=1)
    ci = bootstrap_ci(means, statistic=np.mean, n_resamples=5000, seed=17)
    return [float(ci["low"]), float(ci["high"])]


if __name__ == "__main__":
    main()
