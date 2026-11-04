"""Data acquisition and provenance for public text, tabular, and live market data."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml

from .provenance import DATA, ROOT, sha256_bytes, sha256_file, utc_now

CONFIG_PATH = ROOT / "config" / "experiment.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# --------------------------------------------------------------------- text
@dataclass
class CharDataset:
    text: str
    train_text: str
    val_text: str
    vocab: list[str]
    stoi: dict[str, int]
    itos: dict[int, str]
    train_ids: np.ndarray
    val_ids: np.ndarray
    sha256: str
    source: str

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def load_tinyshakespeare(cfg: dict[str, Any] | None = None) -> CharDataset:
    cfg = cfg or load_config()
    spec = cfg["data"]
    local = ROOT / spec["tinyshakespeare_local"]
    expected = spec["tinyshakespeare_sha256"]
    source = str(local)
    if local.exists() and sha256_file(local) == expected:
        payload = local.read_bytes()
    else:
        payload = _download(spec["tinyshakespeare_url"], expected)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(payload)
        source = spec["tinyshakespeare_url"]
    digest = sha256_bytes(payload)
    if digest != expected:
        raise RuntimeError(f"Tiny Shakespeare checksum mismatch: {digest} != {expected}")
    text = payload.decode("utf-8")
    split = int(0.9 * len(text))
    train_text, val_text = text[:split], text[split:]
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return CharDataset(
        text=text,
        train_text=train_text,
        val_text=val_text,
        vocab=vocab,
        stoi=stoi,
        itos=itos,
        train_ids=np.array([stoi[c] for c in train_text], dtype=np.int64),
        val_ids=np.array([stoi[c] for c in val_text], dtype=np.int64),
        sha256=digest,
        source=source,
    )


def _download(url: str, expected_sha: str | None = None, attempts: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            payload = response.content
            if expected_sha and sha256_bytes(payload) != expected_sha:
                raise RuntimeError("downloaded payload failed checksum")
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from last_error


def get_batch(
    ids: np.ndarray,
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    ix = rng.integers(0, len(ids) - block_size - 1, size=batch_size)
    x = np.stack([ids[i : i + block_size] for i in ix])
    y = np.stack([ids[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


# ------------------------------------------------------------------ tabular
@dataclass
class TabularDataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    description: str
    sha256: str


def load_tabular(name: str) -> TabularDataset:
    from sklearn import datasets

    loaders = {
        "breast_cancer": datasets.load_breast_cancer,
        "wine": datasets.load_wine,
        "digits": datasets.load_digits,
        "iris": datasets.load_iris,
    }
    if name not in loaders:
        raise ValueError(f"unsupported tabular dataset: {name}")
    bunch = loaders[name]()
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target, dtype=np.int64)
    digest = sha256_bytes(X.tobytes() + y.tobytes() + name.encode())
    return TabularDataset(
        name=name,
        X=X,
        y=y,
        feature_names=list(bunch.feature_names),
        target_names=[str(t) for t in bunch.target_names],
        description=str(bunch.DESCR).split("\n")[0],
        sha256=digest,
    )


# --------------------------------------------------------------- live market
@dataclass
class MarketSnapshot:
    symbol: str
    frame: pd.DataFrame
    raw_sha256: str
    retrieved_utc: str
    source_url: str
    last_close: float
    last_date: str


def fetch_yahoo_daily(
    symbol: str,
    range_: str = "5y",
    interval: str = "1d",
    attempts: int = 3,
) -> MarketSnapshot:
    """Fetch live daily OHLCV from Yahoo Finance with retries and provenance."""
    hosts = ["https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com"]
    last_error: Exception | None = None
    for attempt in range(attempts):
        for host in hosts:
            url = f"{host}/v8/finance/chart/{symbol}"
            params = {"range": range_, "interval": interval, "events": "div,splits"}
            try:
                response = requests.get(
                    url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30
                )
                response.raise_for_status()
                raw = response.content
                frame = _parse_yahoo_json(raw, symbol)
                return MarketSnapshot(
                    symbol=symbol,
                    frame=frame,
                    raw_sha256=sha256_bytes(raw),
                    retrieved_utc=utc_now(),
                    source_url=response.url,
                    last_close=float(frame["close"].iloc[-1]),
                    last_date=str(frame["date"].iloc[-1].date()),
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        time.sleep(5.0 * (attempt + 1))
    raise RuntimeError(f"Yahoo fetch failed for {symbol}") from last_error


def _parse_yahoo_json(raw: bytes, symbol: str) -> pd.DataFrame:
    import json

    payload = json.loads(raw)
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo error for {symbol}: {chart['error']}")
    result = chart["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York"),
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        }
    )
    frame = frame.dropna(subset=["close"]).reset_index(drop=True)
    if len(frame) < 300:
        raise RuntimeError(f"insufficient history for {symbol}: {len(frame)} rows")
    return frame


def cache_market_snapshot(snapshot: MarketSnapshot) -> Path:
    directory = DATA / "raw" / "market"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{snapshot.symbol}_{snapshot.retrieved_utc[:10]}.csv"
    snapshot.frame.to_csv(path, index=False)
    return path


def write_bytes(path: str | Path, payload: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def read_json_bytes(raw: bytes) -> dict[str, Any]:
    import json

    return json.load(io.BytesIO(raw))
