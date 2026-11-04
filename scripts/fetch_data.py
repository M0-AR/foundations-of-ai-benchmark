"""Download and verify every external input used by the benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fbench.data import (  # noqa: E402
    cache_market_snapshot,
    fetch_yahoo_daily,
    load_config,
    load_tinyshakespeare,
)
from fbench.provenance import sha256_file  # noqa: E402


def main() -> int:
    cfg = load_config()
    text = load_tinyshakespeare(cfg)
    path = ROOT / cfg["data"]["tinyshakespeare_local"]
    print(f"text: {len(text.text):,} chars, vocab={text.vocab_size}, sha256={text.sha256}")
    print(f"text: {path}")
    for symbol in cfg["data"]["market_symbols"]:
        snapshot = fetch_yahoo_daily(
            symbol,
            range_=cfg["data"]["market_range"],
            interval=cfg["data"]["market_interval"],
        )
        cached = cache_market_snapshot(snapshot)
        print(
            f"market: {symbol} rows={len(snapshot.frame)} "
            f"last={snapshot.last_date} close={snapshot.last_close:.2f} "
            f"sha256={snapshot.raw_sha256[:16]}... -> {cached.relative_to(ROOT)}"
        )
    print(f"config sha256={sha256_file(ROOT / 'config' / 'experiment.yaml')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
