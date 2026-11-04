# Novelty Lab Pre-registration

This file was written before running any lab. It freezes the endpoints, splits,
seeds, and decision rules so that the lab results can be reported honestly,
including negative results. The labs live outside the E6 manifest so that
exploratory work does not silently change the public benchmark.

## Scope

The existing repository verifies known foundations. These labs ask whether the
verification stack can produce a new, falsifiable result rather than another
confirmation. Five labs are attempted; each is allowed to fail.

## Lab 1: Falsification Coverage

- Unit: 24 targeted probes, six per family: autograd derivatives, attention and
  RoPE invariants, statistical inference, and research-protocol metadata.
- A probe is killed only if the uncorrupted control passes the detector and the
  corrupted version fails it.
- Primary endpoint: overall kill rate.
- Pre-registered tests:
  - H7a: overall kill rate is at least 0.80.
  - H7b: the numerical-family kill rate is at least the protocol-family kill
    rate.
- Seeds: deterministic; no seed sweep.

## Lab 2: Locality Phase Diagram

- Synthetic language: vocabulary 8, sequence length 64, copy probability 0.85,
  dependency lag `L` in {1, 2, 4, 8}; 400 train sequences and 100 validation
  sequences per lag.
- Baselines: add-one-smoothed n-gram models with context lengths `k` in {1, 2, 3}.
- Model: 2-layer, 2-head, 32-dimensional TinyGPT, block size 32, 400 steps,
  batch 32, learning rate 3e-3, two seeds {0, 1}.
- Primary endpoint: `delta_bpc = transformer_bpc - best_ngram_bpc` on held-out
  sequences; negative values favour the transformer.
- Pre-registered tests:
  - H8a: at `L = 8`, the transformer has lower BPC than the best n-gram.
  - H8b: the transformer advantage is larger at `L = 8` than at `L = 1`.
- No tuning after seeing validation results. If the model fails to train, the
  result is reported as a failure, not silently retried with a new recipe.

## Lab 3: Reproducibility Envelope

- Workloads: a NumPy MLP trained by gradient descent and a TinyGPT trained with
  torch, both on deterministic synthetic data.
- Thread counts: 1, 2, 4, 8 for `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, and torch intra-op threads.
- Repeats: two per condition; worker runs in a fresh subprocess.
- Primary endpoint: parameter hash equality and loss range.
- Pre-registered tests:
  - H9a: at least one workload has a non-identical parameter hash across thread
    counts.
  - H9b: the loss range across thread counts is below 1e-3 for both workloads.

## Lab 4: Activation Calibration Under Flat Accuracy

- Data and model: the E2 protocol; breast cancer, wine, and digits; hidden 64;
  300 epochs; learning rate 0.01; batch 128; 30% test split; seeds 0-9.
- Activations: sigmoid, tanh, relu, leaky_relu, gelu.
- Endpoints: test accuracy, test NLL, and 15-bin expected calibration error
  (ECE).
- Pre-registered tests:
  - H10a: at least one activation pair differs in ECE after Holm correction
    across all 30 dataset-pair tests.
  - H10b: the activation ranking by accuracy is not identical to the ranking by
    ECE on at least one dataset.
- The accuracy result from E2 is not re-tested; only the calibration endpoint is
  new.

## Lab 5: Null-Market False Alpha

- Generator: zero-drift GARCH(1,1) with omega 1e-6, alpha 0.08, beta 0.90.
- Markets: 50 independent simulations, five symbols each, 1,250 daily returns.
- Model: walk-forward ridge with features from E5 (five return lags, 20-day
  volatility, 20-day momentum, volume z-score, intraday range), refit every 21
  days, 60% initial train, ridge alpha selected on the most recent 20% of the
  training window.
- Endpoints: Ljung-Box p-value on squared returns and Diebold-Mariano p-value
  versus a zero forecast, Holm-corrected across five symbols per market.
- Pre-registered tests:
  - H11a: volatility clustering is detected in at least 80% of simulated
    symbols.
  - H11b: the market-level false-alpha rate (any Holm-significant DM result) is
    at most 0.10.

## Reporting Rule

Each lab writes a JSON result under `labs/results/`. The final report must list
the pre-registered endpoint, the observed value, and whether the hypothesis
survived. No result may be described as a discovery if its pre-registered test
failed or if the interval includes the null.
