# Methods notes

This document records implementation details that are too low-level for the
manuscript but matter for replication.

## E1: calculus and automatic differentiation

- Central differences use `(f(x+h) - f(x-h)) / (2h)` in `float64`.
- The convergence slope is fit by ordinary least squares on `log(h)` versus
  `log(error)` for `h >= 1e-3`, where truncation error dominates.
- The scalar engine is a topological reverse-mode implementation. Each
  primitive stores local values and a vector–Jacobian product closure.
- The MLP gradient is derived by hand for cross-entropy, softmax, and `tanh`.
  It is compared with central differences at random coordinates and with
  PyTorch autograd on identical weights.
- Relative error denominator: `max(1, |analytic| + |numeric|)`.

## E2: activations

- One hidden layer, 64 units.
- Initialisation: Xavier for sigmoid/tanh, He for ReLU/leaky ReLU/GELU.
- Optimiser: Adam, learning rate 0.01, batch size 128, 300 epochs, L2 `1e-4`.
- Ten seeds per dataset × activation; the same seed is used for every
  activation on a given dataset, so tests are paired.
- Endpoint: test accuracy. Test cross-entropy is secondary.
- Effect size: paired Cohen's `d_z` on the ten seed differences.
- Multiplicity: Holm within the 30-test all-pairs family (primary) and within
  the 12-test vs-ReLU family (secondary).
- Stability: split-half Kendall τ over seeds and a bootstrap distribution of
  the selected-best activation.

## E3: attention and RoPE

- Attention: explicit masked softmax, scale `1/sqrt(head_dim)`, four heads,
  sequence length 16, batch 4.
- Parity target: `torch.nn.functional.scaled_dot_product_attention` in
  `float64`, 50 random trials.
- Causal leak: maximum attention weight assigned to a future position.
- RoPE: interleaved adjacent channel pairs, base 10,000. The test checks
  pair-norm preservation, total-norm preservation, Toeplitz structure of the
  relative-position score matrix, and that scores vary across offsets.
- Gradient check: `torch.autograd.gradcheck` in double precision.

## E4: character transformer

- Four blocks, four heads, 128 embedding dimensions, block size 128.
- Pre-RMSNorm, GELU feed-forward with 4× expansion, dropout 0.1.
- Tied input/output embeddings; learned absolute position embeddings are
  replaced by RoPE in the attention path.
- 798,464 total parameters; 790,144 non-embedding parameters.
- AdamW, learning rate `3e-4`, weight decay 0.01, gradient clipping 1.0.
- Batch size 32, 1,200 steps, three seeds.
- Validation: 40 fixed batches every 200 steps; the final validation is the
  endpoint.
- N-gram baselines are trained only on the training split:
  - uniform;
  - unigram with add-one smoothing;
  - trigram with add-one smoothing;
  - Jelinek–Mercer interpolation with weights tuned on the last 10% of the
    training split.
- Bits per character: cross-entropy divided by `ln 2`.
- Confidence intervals: block bootstrap over contiguous 128-character
  validation blocks.

## E5: market protocol

- Features: five lagged log returns, 20-day realised volatility, 20-day
  momentum, 20-day volume z-score, and the intraday high–low range.
- Target: next-day log return.
- Initial training fraction: 0.6. Refit every 21 trading days.
- Ridge alphas: `[0.01, 0.1, 1.0, 10.0]`, selected on the last 20% of each
  training window.
- Out-of-sample R² is computed against a zero forecast:
  `1 - SSE_model / SSE_zero`.
- Diebold–Mariano uses squared-error loss, HAC variance with lag 5, and a
  Holm correction across the five symbols.
- Ljung–Box uses lag 10 on squared returns and a Holm correction across the
  five symbols.
- Strategy: long if the forecast is positive, short if negative, with 2 bp
  cost per unit turnover. Sharpe is annualised by `sqrt(252)`.
- The strict H6 conjunction requires all five symbols to show clustering and
  none to show significant directional alpha. It is intentionally hard to
  satisfy.

## E6: reproducibility

- Deterministic check: train the same MLP twice from seed 0 and compare the
  SHA256 of the flattened parameter bytes.
- Seed stability: split seeds into halves, rank activations by mean test
  accuracy in each half, and compute Kendall τ.
- Cross-dataset transfer: for each dataset pair and activation, correlate seed
  accuracies; report the mean τ.
- Manifest: SHA256 and byte size of result JSON, configuration, source,
  experiment, script, test, and public data files. The audit's own output and
  generated report files are excluded to avoid self-reference.

## Statistical conventions

- Two-sided tests throughout.
- Holm–Bonferroni controls the family-wise error rate at 0.05.
- Bootstrap intervals are percentile intervals with 10,000 resamples for E2,
  2,000 moving-block resamples for E4, and 2,000 moving-block resamples for E5.
- `n/a` is used for undefined statistics. A primary endpoint may never be
  NaN or infinite. Degenerate secondary diagnostics (for example, a
  Diebold–Mariano comparison between two identical zero forecasts) are
  recorded as NaN and are never used in a hypothesis test.
- No result is removed after inspection. Exploratory analyses are labelled as
  exploratory.
