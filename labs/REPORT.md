# Novelty Lab Report

Run date: 2026-11-05. All endpoints and decision rules were frozen in
`labs/PREREGISTRATION.md` before execution. Raw outputs are in `labs/results/`.

## Bottom line

Five labs ran. Nine of ten pre-registered tests passed. The failure is itself
useful: activation calibration differences are visible in the omnibus test and
in raw pairwise tests, but no pairwise difference survives Holm correction.

The strongest new empirical findings are:

1. A locality phase transition: the transformer loses to an n-gram model when
   the dependency lag is inside the n-gram horizon, and wins by about 1.6-1.8
   bits per character when the lag is beyond it.
2. Thread-count changes alter parameter bits without materially changing loss.
   The reproducibility envelope is real, but small.
3. A two-sided Diebold-Mariano test is not an alpha test. On zero-drift GARCH
   markets, 34% of markets produce a significant two-sided rejection, but 0%
   favour the model; all rejections are evidence of harm.
4. A metadata-only protocol audit catches 5 of 6 overt violations but cannot
   detect concealed optional stopping.
5. The repository's default HAC lag under-covers at high persistence: the
   false-positive rate reaches 19.5% at AR(1) coefficient 0.8.

None of these is yet a publishable "invention" without a literature review and
independent replication. They are reproducible, falsifiable leads.

## Results

| Lab | Pre-registered endpoint | Observed | Status |
|---|---|---|---|
| 1 | Overall probe kill rate >= 0.80 | 23/24 = 0.958 | pass |
| 1 | Numerical kill rate >= protocol kill rate | 1.000 vs 0.833 | pass |
| 2 | Transformer beats n-gram at lag 8 | delta = -1.614 BPC | pass |
| 2 | Lag-8 advantage larger than lag-1 | -1.614 vs +0.017 | pass |
| 3 | Any non-identical parameter hash | NumPy 2 hashes, torch 4 hashes | pass |
| 3 | Loss range < 1e-3 | 0.0 and 2.38e-7 | pass |
| 4 | Any ECE pair survives Holm | none (closest p_holm = 0.0586) | fail |
| 4 | Accuracy rank differs from ECE rank | all three datasets | pass |
| 5 | Volatility clustering >= 80% | 100% of 250 symbols | pass |
| 5 | False-alpha rate <= 10% | 0% of 50 markets | pass |

## Lab 1: Falsification Coverage

24 probes, six per family. A probe counts as killed only when the clean control
passes the detector and the corrupted version fails it.

| Family | Killed | Coverage | Survivors |
|---|---:|---:|---|
| Autograd derivatives | 6/6 | 1.000 | none |
| Attention and RoPE | 6/6 | 1.000 | none |
| Statistical inference | 6/6 | 1.000 | none |
| Research protocol | 5/6 | 0.833 | concealed optional stopping |
| Overall | 23/24 | 0.958 | |

The protocol survivor is the important one. A record that reports the planned
sample size, planned seed count, one test evaluation, the planned multiplicity
family, and all planned endpoints is accepted even when the underlying analysis
stopped early and concealed it. A metadata audit cannot distinguish that record
from an honest one. This is not a bug in the detector; it is the boundary of
what documentation can prove.

Exploratory HAC envelope (not pre-registered): 200 null simulations per AR(1)
coefficient, 250 observations, 5% nominal level.

| AR(1) phi | HAC false positive | iid t-test false positive |
|---:|---:|---:|
| 0.0 | 0.080 | 0.075 |
| 0.2 | 0.075 | 0.130 |
| 0.4 | 0.100 | 0.215 |
| 0.6 | 0.105 | 0.305 |
| 0.8 | 0.195 | 0.485 |

The repository's `diebold_mariano` uses the default Newey-West lag rule. At high
persistence the test is anti-conservative. The E5 market conclusion is not
overturned by this lab, but the HAC lag should be treated as an explicit
sensitivity parameter rather than an implementation detail.

## Lab 2: Locality Phase Diagram

Two seeds per cell; 400 transformer steps per run. The n-gram baseline is
add-one smoothed and chooses the best context order on validation.

| Lag L | Transformer BPC | Best n-gram BPC | Delta | Best n-gram order |
|---:|---:|---:|---:|---:|
| 1 | 0.9069 | 0.8899 | +0.0170 | 1 |
| 2 | 1.0806 | 0.9453 | +0.1354 | 2 |
| 4 | 1.1841 | 2.9970 | -1.8129 | 1 |
| 8 | 1.3875 | 3.0019 | -1.6144 | 1 |

The crossover is between lag 2 and lag 4, exactly where the trigram baseline
stops being able to see the dependency. At lag 1 and 2 the n-gram is at least as
good; at lag 4 and 8 the transformer wins by more than 1.6 bits per character.
The advantage is largest at lag 4, not lag 8: the transformer itself degrades as
the dependency moves farther back. The phase boundary is therefore a joint
property of the baseline horizon and the model's effective attention horizon,
not a simple monotone function of lag.

## Lab 3: Reproducibility Envelope

Two workloads, four thread counts, two fresh subprocess repeats per condition.

| Workload | Unique parameter hashes | Loss range | Parameter-norm range |
|---|---:|---:|---:|
| NumPy MLP | 2 | 0.0 | 0.0 |
| Torch TinyGPT | 4 | 2.38e-7 | 0.0 |

Both workloads are bitwise unstable across thread counts. The functional effect
is negligible at this scale, but the result means that a hash-based
reproducibility check can fail while a metric-based check passes. The NumPy
divergence appears only at eight threads; the torch divergence appears at every
thread count above one in this run.

## Lab 4: Activation Calibration Under Flat Accuracy

The accuracy endpoint is the E2 result and is not re-tested here. The new
endpoint is 15-bin ECE.

| Dataset | Accuracy rank | ECE rank | Rank correlation | Friedman ECE p |
|---|---|---|---|---:|
| Breast cancer | tanh, relu, leaky_relu, gelu, sigmoid | tanh, sigmoid, leaky_relu, relu, gelu | 0.30 | 0.000139 |
| Wine | sigmoid, gelu, tanh, relu, leaky_relu | sigmoid, tanh, gelu, relu, leaky_relu | 0.90 | 0.648 |
| Digits | sigmoid, leaky_relu, tanh, relu, gelu | leaky_relu, tanh, sigmoid, relu, gelu | 0.70 | 0.823 |

Breast cancer shows a significant omnibus calibration difference, but the
smallest pairwise raw p-values (0.00195 for tanh vs gelu, relu vs gelu, and
leaky_relu vs gelu) become 0.0586 after Holm correction across 30 tests. The
pre-registered pairwise endpoint therefore fails. This is a near miss, not a
confirmed calibration effect. The rank reversal is real but is driven by small
differences and should not be over-interpreted.

## Lab 5: Null-Market False Alpha

50 independent zero-drift GARCH markets, five symbols each, 250 symbols total.

| Endpoint | Observed |
|---|---:|
| Volatility-clustering detection | 100.0% |
| Market-level false alpha (Holm, model better) | 0.0% |
| Market-level any two-sided DM rejection | 34.0% |
| Mean out-of-sample R2 vs zero | -0.0151 |

The null generator behaves as intended: volatility clusters, but there is no
directional signal. The walk-forward ridge model never produces a
Holm-significant improvement. However, 17 of 50 markets produce a two-sided
Diebold-Mariano rejection because the model is significantly worse than zero.
This is the sign trap: a two-sided rejection is not evidence of alpha, and a
"significant DM test" can be evidence of harm. The E5 wording should be read
with that distinction in mind.

## Reproducibility

Run the full exploratory suite and rebuild the machine-readable summary:

```bash
make labs
```

Equivalent direct commands:

```bash
python labs/lab1_falsification.py
python labs/lab2_locality.py
python labs/lab3_reproducibility.py
python labs/lab4_calibration.py
python labs/lab5_null_market.py
python labs/build_summary.py
```

These results are included in the public README, the interactive preview, and
the E6 SHA-256 artifact manifest. The six confirmatory studies and their 4/6
headline count are unchanged; the labs are reported separately.

Because the lab artifacts are hashed by E6, rerunning `make labs` invalidates
the old manifest. Refresh the provenance and generated pages before checking:

```bash
python experiments/e6_reproducibility.py
python experiments/run_all.py --assemble-only
python scripts/make_report.py
python scripts/build_preview.py
python scripts/check_results.py
```
