# Data documentation

Every external input is pinned by a hash or by a retrieval timestamp. The
benchmark does not scrape HTML and does not use credentials.

## 1. Tiny Shakespeare

| Field | Value |
|---|---|
| Source | `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` |
| Local path | `data/public/tinyshakespeare.txt` |
| Size | 1,115,394 characters |
| Vocabulary | 65 symbols (byte-level characters after UTF-8 decoding) |
| SHA256 | `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` |
| Split | contiguous 90% train / 10% validation |

The loader verifies the checksum on every run and refuses to train on a
mismatched file. There is no separate test split; the validation split is
both monitored during training and used as the reported endpoint. This is a
deliberate limitation recorded in the manuscript.

The corpus is distributed with Karpathy's `char-rnn` repository and is a
concatenation of Shakespeare's plays. It is used only for character-level
language modelling.

## 2. Tabular datasets

| Dataset | Rows | Features | Classes | Loader |
|---|---:|---:|---:|---|
| Breast cancer | 569 | 30 | 2 | `sklearn.datasets.load_breast_cancer` |
| Wine | 178 | 13 | 3 | `sklearn.datasets.load_wine` |
| Digits | 1,797 | 64 | 10 | `sklearn.datasets.load_digits` |

The split is 70/30, stratified, `random_state=42`. Standardisation statistics
are estimated on the training split only. E2 never selects a model on the
test split. The loader also computes a SHA256 over the feature matrix, target
vector, and dataset name; it is recorded in the E2 artifact.

## 3. Live market data

| Field | Value |
|---|---|
| Endpoint | `https://query1.finance.yahoo.com/v8/finance/chart/{symbol}` (fallback `query2`) |
| Symbols | `SPY`, `QQQ`, `AAPL`, `MSFT`, `NVDA` |
| Range / interval | `5y` / `1d` |
| Fields | timestamp, open, high, low, close, volume |
| Adjustment | provider split adjustment; dividends not reinvested |
| Cache | `data/raw/market/{symbol}_{YYYY-MM-DD}.csv` |

Each E5 run stores the raw response SHA256, final URL, retrieval timestamp,
row count, first date, last date, and last close in `results/e5_market.json`.
The cache is written for inspection but the experiment always fetches fresh
data. A rerun on a later date will therefore see a different sample; this is
intended for external validity and is disclosed as a reproducibility
limitation.

### Leakage controls

- The target is the next day's log return, built by shifting close prices
  backwards.
- Features at day *t* use only information available at the close of day *t*.
- The initial 60% of rows is training only.
- Refits occur every 21 trading days and use data up to the refit date.
- The ridge penalty is selected inside each training window, never on the
  out-of-sample block.
- The cost model (2 basis points per unit turnover) is applied to the sign
  strategy before Sharpe and drawdown are computed.

### Known provider limitations

Yahoo Finance is not an exchange-grade feed. Missing sessions, provider-side
adjustments, and late corrections are outside the benchmark's control. The
snapshot hashes make the exact sample auditable, but they do not make the
provider authoritative.

## 4. Network policy

Only two network operations are performed:

1. Tiny Shakespeare download if the local checksum-pinned file is absent.
2. Yahoo Finance chart requests for E5 and `scripts/fetch_data.py`.

Everything else runs offline. The Docker image works without network access
if the corpus and market CSVs are mounted; E5 will fail rather than silently
reuse stale data.
