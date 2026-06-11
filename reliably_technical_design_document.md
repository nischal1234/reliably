# `reliably` — Technical Design & Implementation Document

**Status:** Implementation-ready · **Spec version:** 1.0 · **Audience:** the engineer building it (you)

> A framework-agnostic library that turns any model's probabilistic outputs into a publication-ready *reliability report* — calibration, uncertainty, discrimination, and representation quality — where **every metric carries a bootstrap confidence interval and every model comparison carries a significance test.**

This document is the single source of truth to begin implementation. It is divided into: **Part A — Mathematical foundations** (every metric, with formulas and algorithms), **Part B — Engineering specification** (architecture, API, repo, code, tests, CI, release), **Part C — Research & launch** (the empirical study, the paper, go-to-market). Read A and B before writing code; the math determines the interfaces.

---

# PART A — MATHEMATICAL FOUNDATIONS

## A.0 Notation

- Dataset of `N` samples, indexed `i`. For classification with `K` classes, the model emits a probability vector `p_i ∈ Δ^{K-1}` (the simplex). True label `y_i ∈ {1..K}`.
- "Top-label" confidence: `ĉ_i = max_k p_i[k]`, predicted class `ŷ_i = argmax_k p_i[k]`, correctness `a_i = 1[ŷ_i = y_i]`.
- For binary tasks we also use scalar score `s_i ∈ [0,1]` and label `y_i ∈ {0,1}`.
- `Φ` is the standard normal CDF, `Φ⁻¹` its inverse (probit).
- Indicator `1[·]`. Empirical mean `Ê[·]`.

All metrics below are defined so they can be computed from `(y_true, y_prob)` alone — this is what makes the library framework-agnostic.

---

## A.1 Calibration metrics

A model is **perfectly calibrated** if, among all samples predicted with confidence `c`, exactly fraction `c` are correct: `P(ŷ = y | ĉ = c) = c` for all `c`. Calibration error metrics quantify deviation from this.

### A.1.1 Reliability curve
Partition confidences into `M` bins `B_1..B_M`. For bin `B_m`:
- `conf(B_m) = (1/|B_m|) Σ_{i∈B_m} ĉ_i` (mean confidence)
- `acc(B_m)  = (1/|B_m|) Σ_{i∈B_m} a_i` (mean accuracy)

A reliability diagram plots `acc(B_m)` against `conf(B_m)`; the diagonal is perfect calibration.

### A.1.2 Expected Calibration Error (ECE) — equal-width binning
```
ECE = Σ_m (|B_m| / N) · | acc(B_m) − conf(B_m) |
```
Default `M = 15`, equal-width bins on `[0,1]`. **This is the most-used metric and the most flawed** (see A.1.6); compute it for compatibility but never report it alone.

### A.1.3 Adaptive ECE (equal-mass binning)
Same formula, but bin edges chosen so each bin holds `≈ N/M` samples (quantiles of the confidence distribution). Removes the empty/overfull-bin pathology of equal-width binning. Expose as `binning="adaptive"`.

### A.1.4 Maximum Calibration Error (MCE)
```
MCE = max_m | acc(B_m) − conf(B_m) |
```
Worst-case bin; useful for safety-critical settings.

### A.1.5 Classwise / marginal ECE
Top-label ECE ignores all but the predicted class. Classwise ECE averages a one-vs-rest calibration error over every class `k`:
```
cwECE = (1/K) Σ_k Σ_m (|B_{m,k}| / N) · | acc_k(B_{m,k}) − conf_k(B_{m,k}) |
```
where bin membership uses `p_i[k]` and "accuracy" is `1[y_i = k]`. Report this for multiclass models.

### A.1.6 Debiased ECE
Binned ECE is a **biased estimator**: with finite samples it is positive even for a perfectly calibrated model, and the bias *grows* as calibration improves and as `M` grows. Implement the bias-corrected estimator: within each bin, the plug-in squared gap `(acc−conf)²` overestimates the true squared gap by approximately the variance of the accuracy estimate `acc(B_m)(1−acc(B_m))/(|B_m|−1)`. For the L2 (squared) ECE:
```
ECE²_debiased = Σ_m (|B_m|/N) · [ (acc(B_m) − conf(B_m))²
                                   − acc(B_m)(1 − acc(B_m)) / (|B_m| − 1) ]
```
Clamp negative bin terms to 0. Expose as `debiased=True`.

### A.1.7 Kernel calibration error / SmoothECE (the principled default)
Binning is discontinuous and bin-count-dependent. The kernel approach replaces bins with a smooth kernel regression of correctness on confidence. Define a Gaussian-kernel reliability estimate
```
r̂(c) = Σ_i K_h(c − ĉ_i) · a_i  /  Σ_i K_h(c − ĉ_i),    K_h(u) = exp(−u² / 2h²)
```
and the smooth calibration error as the kernel-weighted mean absolute gap
```
smECE = Σ_i w_i · | r̂(ĉ_i) − ĉ_i |  /  Σ_i w_i ,   w_i = 1.
```
The bandwidth `h` is selected automatically (reflect at boundaries and minimize a smoothed risk, or rule-of-thumb `h ∝ N^(−1/5)`). **This is the recommended headline metric**: continuous, binning-free, and it yields a reliability diagram with a confidence band for free (A.1.8). Provide an option to delegate to the `relplot` algorithm for exact parity with the published SmoothECE.

### A.1.8 Reliability diagram with confidence band
The kernel estimate `r̂(c)` plotted over `c ∈ [0,1]`, with a band from the bootstrap (A.4) over the curve. This visual — a smooth curve with a shaded band that may or may not cross the diagonal — is the product's signature image.

---

## A.2 Proper scoring rules

### A.2.1 Brier score (+ Murphy decomposition)
Binary: `BS = (1/N) Σ_i (s_i − y_i)²`. Multiclass (one-hot `e_{y_i}`): `BS = (1/N) Σ_i ‖p_i − e_{y_i}‖²`.

The **Murphy / calibration–refinement decomposition** (binary, binned) separates calibration from sharpness:
```
BS =  Σ_m (|B_m|/N)(s̄_m − ȳ_m)²        # reliability (calibration)  ↓ better
    − Σ_m (|B_m|/N)(ȳ_m − ȳ)²          # resolution                ↑ better
    + ȳ(1 − ȳ)                          # uncertainty (irreducible)
```
where `s̄_m`, `ȳ_m` are bin mean score/label and `ȳ` is base rate. Report all three components.

### A.2.2 Negative log-likelihood (log loss)
`NLL = −(1/N) Σ_i log p_i[y_i]`. Clip probabilities to `[ε, 1−ε]` with `ε = 1e-12` for numerical safety. NLL is strictly proper and is the objective minimized by temperature scaling.

### A.2.3 (Optional) Ranked Probability Score
For ordinal labels, RPS penalizes by distance between predicted and true class; include in `v1.1`.

---

## A.3 Discrimination and the DeLong test

### A.3.1 AUROC
With `m` positives `{X_i}` and `n` negatives `{Y_j}`, define the kernel `ψ(x,y) = 1[x>y] + ½·1[x=y]`. Then
```
AUC = (1 / m·n) Σ_i Σ_j ψ(X_i, Y_j).
```

### A.3.2 DeLong variance (fast structural components)
Compute placement values:
```
V10(X_i) = (1/n) Σ_j ψ(X_i, Y_j),    V01(Y_j) = (1/m) Σ_i ψ(X_i, Y_j).
```
Then `AUC = mean_i V10_i = mean_j V01_j`, and
```
S10 = (1/(m−1)) Σ_i (V10_i − AUC)²,   S01 = (1/(n−1)) Σ_j (V01_j − AUC)²,
Var(AUC) = S10/m + S01/n.
```
Use the **midrank-based O(N log N)** computation (Sun & Xu, 2014) rather than the naïve O(N²) double sum — placement values equal normalized midranks, so they come from sorting.

### A.3.3 Comparing two correlated AUROCs (same test set)
For predictors `a, b` on the *same* samples, form the `2×2` covariance `S = S10/m + S01/n` from the stacked `V10`, `V01` vectors. The test statistic is
```
z = (AUC_a − AUC_b) / sqrt(S_aa + S_bb − 2·S_ab),    p = 2(1 − Φ(|z|)).
```
This is the canonical answer to "is model A's AUROC significantly higher than B's?" and is central to the library's identity.

---

## A.4 Uncertainty of the metrics themselves (the wedge)

Every metric `θ̂` (ECE, Brier, AUROC, MIG, …) is an estimate from a finite sample and must ship with an interval. Default method: **nonparametric bootstrap**; default `B = 2000` resamples; fixed RNG seed for reproducibility.

### A.4.1 Percentile CI
Resample indices with replacement `B` times, recompute `θ̂*_b`. The `(1−α)` percentile interval is `[Q_{α/2}, Q_{1−α/2}]` of `{θ̂*_b}`.

### A.4.2 BCa CI (bias-corrected and accelerated — default for skewed metrics)
- Bias correction: `z₀ = Φ⁻¹( #{θ̂*_b < θ̂} / B )`.
- Acceleration via jackknife (leave-one-out estimates `θ̂_(i)`, mean `θ̄`):
```
a = Σ_i (θ̄ − θ̂_(i))³  /  ( 6 · [ Σ_i (θ̄ − θ̂_(i))² ]^(3/2) ).
```
- Adjusted percentile endpoints (level `α`): with `z_α = Φ⁻¹(α)`,
```
α₁ = Φ( z₀ + (z₀ + z_{α/2})  / (1 − a(z₀ + z_{α/2})) )
α₂ = Φ( z₀ + (z₀ + z_{1−α/2})/ (1 − a(z₀ + z_{1−α/2})) )
```
report `[Q_{α₁}, Q_{α₂}]`. BCa corrects for both bias and skew and gives better coverage than percentile for metrics like ECE.

### A.4.3 Paired bootstrap difference test
For comparing two models on the same data without DeLong (works for *any* metric): resample sample-indices once per replicate, apply to **both** models, record `Δ_b = θ̂*_{a,b} − θ̂*_{b,b}`. Two-sided p-value via the bootstrap-hypothesis convention:
```
p = 2 · min( (1 + #{Δ_b ≤ 0}) / (B + 1),  (1 + #{Δ_b ≥ 0}) / (B + 1) ).
```
Report `Δ̂`, its CI, and `p`. This is the general-purpose companion to DeLong.

### A.4.4 Multiple-comparison correction
When comparing many models/metrics at once, expose Holm–Bonferroni (FWER) and Benjamini–Hochberg (FDR) adjustment; default Holm.

### A.4.5 Performance note
The bootstrap must be **vectorized**: pre-draw a `B×N` index matrix and compute all replicates with array ops, not a Python loop. For metrics that are means (Brier, NLL, ECE bin counts), reformulate as weighted sums over the index matrix. Optional Numba/torch backend for `B·N > 10⁸`.

---

## A.5 Recalibration

All methods fit on a held-out calibration split (never the test split used for reporting) or via cross-fitting. Each returns a `Calibrator` with `.transform(p) -> p_cal`.

### A.5.1 Temperature scaling (default, multiclass)
Single scalar `T>0` applied to logits `z`: `p_cal = softmax(z/T)`. Fit `T` by minimizing NLL on the calibration split (1-D convex problem; L-BFGS or golden-section). Preserves accuracy (argmax unchanged). Requires logits; if only probabilities are given, recover pseudo-logits as `log p` up to a constant.

### A.5.2 Vector / matrix scaling
Generalize `T` to a per-class vector or full `K×K` linear map on logits, fit by NLL. More expressive, can change accuracy, risks overfit on small calibration sets — gate behind `method="matrix"`.

### A.5.3 Platt scaling (binary)
Fit logistic `p_cal = σ(A·s + B)` by maximum likelihood on the calibration split.

### A.5.4 Isotonic regression
Nonparametric monotone fit via Pool-Adjacent-Violators (PAV). Flexible but step-like and data-hungry; expose `method="isotonic"`. Wrap `sklearn.isotonic.IsotonicRegression`.

### A.5.5 Beta calibration (binary)
Three-parameter family fixing isotonic/Platt edge biases. Fit logistic regression on features `[log s, −log(1−s)]`:
```
logit(p_cal) = c + a·log s − b·log(1 − s),    a, b ≥ 0.
```

### A.5.6 Histogram binning
Replace each bin's confidence with its empirical accuracy on the calibration split. Simple, nonparametric, discontinuous.

### A.5.7 Reporting recalibration
Always show before/after on the *test* split with CIs, and a paired test on the ECE/Brier improvement — so users see whether the recalibration gain is statistically real.

---

## A.6 Representation quality (the disentanglement arm, `reliably.repr`)

Given latent codes `z ∈ ℝ^{N×D}` and ground-truth generative factors `v ∈ ℝ^{N×K}`, measure how cleanly latents encode factors. The modern, maintained PyTorch successor to the abandoned `disentanglement_lib`.

### A.6.1 Mutual Information Gap (MIG)
Discretize each latent `z_j` into bins; estimate `I(z_j; v_k)` and factor entropy `H(v_k)`.
```
MIG = (1/K) Σ_k (1/H(v_k)) · ( I(z_{j1}; v_k) − I(z_{j2}; v_k) )
```
where `j1, j2` are the latents with the two highest MI for factor `k`. Higher = each factor captured dominantly by one latent.

### A.6.2 SAP (Separated Attribute Predictability)
Build a `D×K` score matrix `S` where `S_{jk}` is the predictive strength of latent `j` for factor `k` (R² via linear fit for continuous factors; balanced accuracy for discrete). `SAP = mean_k (top1_k − top2_k)` of each column.

### A.6.3 DCI (Disentanglement, Completeness, Informativeness)
Train an importance model (gradient-boosted trees or Lasso) predicting each factor from all latents → relative-importance matrix `R ∈ ℝ^{D×K}_{≥0}`.
- **Disentanglement** of latent `j`: `D_j = 1 − H_K(P_{j·})`, `P_{jk}=R_{jk}/Σ_k R_{jk}`, entropy base `K`. Overall `D = Σ_j ρ_j D_j`, weight `ρ_j = Σ_k R_{jk} / Σ R`.
- **Completeness** of factor `k`: `C_k = 1 − H_D(P̃_{·k})`, `P̃_{jk}=R_{jk}/Σ_j R_{jk}`, base `D`.
- **Informativeness**: prediction error (or accuracy) of the importance model per factor.

### A.6.4 FactorVAE metric
Normalize each latent by its global std. For many minibatches: fix one random factor, vary the rest, take the index of the **lowest-variance** normalized latent as a "vote" for that factor. Train a majority-vote classifier (vote-index → factor) and report held-out accuracy.

### A.6.5 Interventional Robustness Score (IRS)
Measures the maximum change in a matched latent under interventions on nuisance factors while the matched factor is held fixed (Suter et al., 2019). Implement faithfully to the reference; report as `[0,1]`.

### A.6.6 CIs for representation metrics
Bootstrap over samples (resample rows of `z, v`) to attach CIs — **no existing disentanglement library does this**, and it's directly reusable from A.4.

---

## A.7 LLM confidence specifics (`reliably.llm`, v1.0+)

LLM "confidence" comes in three forms; the arm ingests each and routes into the calibration machinery (A.1–A.4):
- **Verbalized**: the model states a number ("I'm 80% sure"). Parse to `[0,1]`.
- **Sequence probability / logit-based**: length-normalized token log-probs of the answer span.
- **Semantic entropy** (brief, v1.1): cluster sampled answers by meaning, entropy over clusters.

The arm **does not reimplement** LM-Polygraph / UQLM; it provides thin adapters that consume their outputs and add calibration metrics, CIs, and report cards. Headline LLM metric: top-label ECE + smECE on correctness vs. confidence, with overconfidence flag when mean confidence ≫ accuracy.

---

# PART B — ENGINEERING SPECIFICATION

## B.1 Design principles
1. **Numpy-core, framework-optional.** Public API accepts anything implementing the Python Array API (numpy, torch, jax via `__array_namespace__`), converting once at the boundary. No hard torch dependency in core.
2. **Every number has an interval; every comparison has a test.** Enforced at the type level: a bare float is never returned for a metric — always a `MetricResult`.
3. **Pure functions + thin objects.** Metrics are pure functions; `Report`/`Comparison` are immutable dataclasses wrapping them.
4. **Deterministic.** All stochastic ops take an explicit `rng`/`seed`; same inputs ⇒ same outputs.
5. **No network, no state, no telemetry.** Runs fully offline on the user's machine.
6. **Tested to a higher standard than the incumbents** — this is a differentiator, so coverage and statistical-correctness tests are first-class.

## B.2 Layered architecture

```
┌─────────────────────────────────────────────────────────┐
│ Facade API:  rb.evaluate · rb.compare · rb.recalibrate    │  (B.4)
├─────────────────────────────────────────────────────────┤
│ Report / Comparison / Calibrator   (immutable results)    │  (B.3)
├───────────────┬───────────────┬───────────────┬──────────┤
│ metrics       │ stats         │ recalibrate    │ repr/llm │  (A.1-A.7)
│ (calibration, │ (bootstrap,   │ (temperature,  │          │
│  scoring,     │  BCa, DeLong, │  isotonic,…)   │          │
│  AUROC)       │  paired test) │                │          │
├───────────────┴───────────────┴───────────────┴──────────┤
│ backend: array-API abstraction, RNG, validation, binning  │  (B.6)
├─────────────────────────────────────────────────────────┤
│ viz (matplotlib) · report (Jinja2 HTML/MD export)         │
└─────────────────────────────────────────────────────────┘
```

## B.3 Core data model

```python
# reliably/_core/results.py
from dataclasses import dataclass
from typing import Literal, Mapping

@dataclass(frozen=True, slots=True)
class CI:
    low: float
    high: float
    level: float = 0.95
    method: Literal["percentile", "bca", "analytic"] = "bca"

@dataclass(frozen=True, slots=True)
class MetricResult:
    name: str                 # e.g. "smECE"
    value: float              # point estimate
    ci: CI | None             # None only if CI explicitly disabled
    n: int                    # sample size it was computed on
    extra: Mapping[str, float] | None = None  # e.g. Brier decomposition parts
    def __str__(self) -> str:
        c = f" [{self.ci.low:.4f}, {self.ci.high:.4f}]" if self.ci else ""
        return f"{self.name}={self.value:.4f}{c}"

@dataclass(frozen=True, slots=True)
class Report:
    task: Literal["binary", "multiclass", "regression"]
    metrics: Mapping[str, MetricResult]
    n: int
    meta: Mapping[str, object]            # seed, n_bootstrap, binning, etc.
    # methods: summary(), reliability_diagram(), to_html(), to_markdown(), __getitem__

@dataclass(frozen=True, slots=True)
class ComparisonResult:
    metric: str
    delta: float                          # value_a - value_b
    ci: CI
    p_value: float
    test: Literal["delong", "paired_bootstrap"]
    significant: bool                     # p < alpha after correction
    correction: str | None
```

## B.4 Public API (frozen surface for v1.0)

```python
import reliably as rb

# ---- core evaluation -------------------------------------------------
rb.evaluate(
    y_true,                       # (N,) int labels  | (N,) {0,1} for binary
    y_prob,                       # (N,K) probs | (N,) score for binary
    *,
    task="auto",                  # "auto"|"binary"|"multiclass"
    metrics="default",            # or list[str] | "all"
    binning="adaptive",           # "equal_width"|"adaptive"
    n_bins=15,
    ci="bca",                     # "bca"|"percentile"|None
    n_bootstrap=2000,
    level=0.95,
    seed=0,
) -> Report

# ---- comparison ------------------------------------------------------
rb.compare(
    report_or_inputs_a,
    report_or_inputs_b,
    *,
    metric="auroc",
    test="auto",                  # "auto"→delong for auroc else paired_bootstrap
    correction="holm",            # None|"holm"|"bh"
    level=0.95,
    seed=0,
) -> ComparisonResult | list[ComparisonResult]

# ---- recalibration ---------------------------------------------------
cal = rb.recalibrate(y_prob_cal, y_cal, method="temperature")  # Calibrator
y_prob_test_cal = cal.transform(y_prob_test)

# ---- standalone metric functions (each returns MetricResult) ---------
rb.metrics.ece(y_true, y_prob, ...)
rb.metrics.smece(y_true, y_prob, ...)
rb.metrics.brier(y_true, y_prob, decompose=True, ...)
rb.metrics.nll(...); rb.metrics.auroc(...); rb.metrics.mce(...)

# ---- representation arm (v1.0) --------------------------------------
rb.repr.disentanglement(z, factors, metrics=("mig","sap","dci","factorvae","irs"))

# ---- llm arm (v1.0+) -------------------------------------------------
rb.llm.evaluate(answers, correct, confidence, kind="verbalized")

# ---- viz / export ----------------------------------------------------
report.reliability_diagram(ax=None, band=True)   # matplotlib Axes
report.to_html("card.html"); report.to_markdown()
```

Design rule: `evaluate` and `compare` accept **either** raw arrays **or** a pre-built `Report`, so comparison never recomputes if a report exists.

## B.5 Module responsibilities

| Module | Contents | Depends on |
|---|---|---|
| `reliably/_core/backend.py` | array-API namespace resolution, dtype/shape validation, RNG, binning utilities | — |
| `reliably/_core/results.py` | `CI`, `MetricResult`, `Report`, `ComparisonResult` | backend |
| `reliably/stats/bootstrap.py` | vectorized bootstrap, percentile + BCa, jackknife | backend |
| `reliably/stats/delong.py` | fast DeLong variance + 2-sample test | backend |
| `reliably/stats/tests.py` | paired bootstrap test, Holm/BH correction | bootstrap |
| `reliably/metrics/calibration.py` | ECE family, MCE, smECE, debiased, reliability curve | backend, bootstrap |
| `reliably/metrics/scoring.py` | Brier (+decomp), NLL | backend, bootstrap |
| `reliably/metrics/discrimination.py` | AUROC | backend, delong |
| `reliably/recalibrate/*.py` | temperature, platt, isotonic, beta, histogram, matrix | backend |
| `reliably/repr/*.py` | mig, sap, dci, factorvae, irs | backend, bootstrap |
| `reliably/llm/*.py` | adapters + verbalized parsing | metrics |
| `reliably/viz/diagrams.py` | reliability diagram, confidence histogram | metrics (matplotlib extra) |
| `reliably/report/render.py` | HTML/Markdown report cards (Jinja2) | results |
| `reliably/api.py` | `evaluate`, `compare`, `recalibrate` facade | everything |

## B.6 Backend / numerics contract
- Single conversion at API boundary to a canonical float64 numpy array (or the input's native namespace if `keep_backend=True`).
- Probabilities validated: finite, `≥0`, rows sum to `1±1e-4` (auto-normalize with a warning if not). Binary scores validated to `[0,1]`.
- `ε = 1e-12` clipping for logs.
- RNG: `numpy.random.default_rng(seed)`; the same generator threads through bootstrap and any stochastic metric.
- Determinism test in CI (B.10): two runs with same seed are bit-identical.

## B.7 Repository layout

```
reliably/
├── pyproject.toml
├── README.md
├── LICENSE                      # Apache-2.0
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── src/reliably/
│   ├── __init__.py              # re-exports public API + __version__
│   ├── api.py
│   ├── _core/{backend,results,validation}.py
│   ├── stats/{bootstrap,delong,tests}.py
│   ├── metrics/{calibration,scoring,discrimination}.py
│   ├── recalibrate/{base,temperature,platt,isotonic,beta,histogram,matrix}.py
│   ├── repr/{mig,sap,dci,factorvae,irs}.py
│   ├── llm/{evaluate,verbalized,adapters}.py
│   ├── viz/diagrams.py
│   ├── report/{render.py, templates/card.html.j2}
│   └── py.typed
├── tests/
│   ├── test_calibration.py
│   ├── test_delong.py           # parity vs pROC reference values
│   ├── test_bootstrap_coverage.py   # statistical correctness
│   ├── test_recalibrate.py
│   ├── test_repr.py
│   ├── test_determinism.py
│   └── property/test_metric_properties.py   # Hypothesis
├── benchmarks/                  # the empirical study (Part C)
│   ├── datasets.py
│   ├── run_study.py
│   └── results/
├── docs/                        # MkDocs Material
│   ├── index.md  quickstart.md  concepts.md  api/  examples/
└── .github/workflows/{ci.yml, release.yml}
```

## B.8 Key starter files

### `pyproject.toml`
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "reliably"
version = "0.1.0"
description = "Statistically rigorous model trust evaluation: calibration, uncertainty, and representation quality — with CIs and significance tests on every metric."
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
authors = [{ name = "YOUR NAME" }]
keywords = ["calibration", "uncertainty", "ece", "auroc", "delong", "bootstrap", "disentanglement", "trustworthy-ml"]
dependencies = ["numpy>=1.24", "scipy>=1.10"]

[project.optional-dependencies]
viz   = ["matplotlib>=3.7"]
report= ["jinja2>=3.1"]
sklearn = ["scikit-learn>=1.3"]   # isotonic, DCI importance models
torch = ["torch>=2.1"]            # optional GPU bootstrap / repr arm
llm   = ["lm-polygraph", "uqlm"]  # adapters only
all   = ["reliably[viz,report,sklearn,torch]"]
dev   = ["pytest", "pytest-cov", "hypothesis", "mypy", "ruff", "mkdocs-material", "mkdocstrings[python]"]

[project.urls]
Homepage = "https://github.com/YOU/reliably"
Documentation = "https://reliably.readthedocs.io"

[tool.ruff]
line-length = 100
[tool.mypy]
strict = true
[tool.pytest.ini_options]
addopts = "-q --cov=reliably --cov-report=term-missing"
```

### `src/reliably/stats/bootstrap.py` (the engine — abridged but correct)
```python
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from reliably._core.results import CI

def bootstrap_ci(
    estimator,                      # callable(idx: np.ndarray) -> float
    n: int,
    *,
    point: float,
    n_boot: int = 2000,
    level: float = 0.95,
    method: str = "bca",
    seed: int = 0,
) -> CI:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))          # B x N resample matrix
    boot = np.fromiter((estimator(idx[b]) for b in range(n_boot)),
                       dtype=float, count=n_boot)
    alpha = 1.0 - level
    if method == "percentile":
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        return CI(float(lo), float(hi), level, "percentile")

    # --- BCa ---
    z0 = norm.ppf((np.sum(boot < point) + 0.5) / (n_boot + 1))
    # jackknife for acceleration
    full = np.arange(n)
    jack = np.fromiter(
        (estimator(np.delete(full, i)) for i in range(n)),
        dtype=float, count=n,
    )
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5) + 1e-12
    a = num / den
    zL, zU = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    def adj(zq: float) -> float:
        return float(norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq))))
    lo, hi = np.quantile(boot, [adj(zL), adj(zU)])
    return CI(float(lo), float(hi), level, "bca")
```
> Note: the per-sample Python comprehension is the readable reference; the production path vectorizes the common "metric = weighted mean" case and only falls back to the loop for arbitrary estimators. The jackknife loop is O(N) calls — fine for N up to ~10^5; for larger N use the infinitesimal jackknife approximation (a `v1.1` task).

### `src/reliably/stats/delong.py` (fast DeLong, abridged)
```python
import numpy as np
from scipy.stats import norm

def _midrank(x):
    order = np.argsort(x)
    ranked = np.empty_like(order, dtype=float)
    sx = x[order]
    i, N = 0, len(x)
    while i < N:
        j = i
        while j < N and sx[j] == sx[i]:
            j += 1
        ranked[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranked

def delong_var_components(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    m, n = len(pos), len(neg)
    tx, ty = _midrank(pos), _midrank(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
    v10 = (tz[:m] - tx) / n
    v01 = 1.0 - (tz[m:] - ty) / m
    s10 = np.var(v10, ddof=1) / m
    s01 = np.var(v01, ddof=1) / n
    return auc, s10 + s01, v10, v01

def delong_test(scores_a, scores_b, labels):
    auc_a, var_a, v10a, v01a = delong_var_components(scores_a, labels)
    auc_b, var_b, v10b, v01b = delong_var_components(scores_b, labels)
    cov = (np.cov(v10a, v10b)[0, 1] / len(v10a)
           + np.cov(v01a, v01b)[0, 1] / len(v01a))
    se = np.sqrt(var_a + var_b - 2 * cov)
    z = (auc_a - auc_b) / se if se > 0 else 0.0
    return auc_a - auc_b, float(2 * (1 - norm.cdf(abs(z)))), se
```

### `tests/property/test_metric_properties.py` (Hypothesis)
```python
import numpy as np
from hypothesis import given, strategies as st, settings
import reliably as rb

@settings(max_examples=200, deadline=None)
@given(n=st.integers(50, 500), seed=st.integers(0, 10_000))
def test_perfect_calibration_has_low_ece(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)                       # confidence
    y = (rng.uniform(0, 1, n) < p).astype(int)     # calibrated by construction
    r = rb.metrics.ece(y, np.c_[1 - p, p], ci=None)
    assert 0.0 <= r.value <= 0.25                  # tightened by debias test

@given(n=st.integers(100, 1000), seed=st.integers(0, 10_000))
def test_auroc_in_unit_interval_and_ci_brackets(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n); s = rng.uniform(0, 1, n)
    r = rb.metrics.auroc(y, s, seed=seed)
    assert 0.0 <= r.value <= 1.0
    assert r.ci.low <= r.value <= r.ci.high
```

### `.github/workflows/ci.yml`
```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    strategy:
      matrix:
        python: ["3.10", "3.11", "3.12"]
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "${{ matrix.python }}" }
      - run: pip install -e ".[dev,all]"
      - run: ruff check .
      - run: mypy src
      - run: pytest --cov=reliably --cov-fail-under=90
```

## B.9 Performance & numerical targets
- `evaluate` on `N = 50k`, `K = 10`, `B = 2000`: target **< 3 s** on a laptop CPU via vectorized resampling.
- DeLong via midrank is `O(N log N)`; verify against naïve `O(N²)` on small N in tests.
- Provide `ci=None` fast path for point estimates only.
- Memory guard: the `B×N` index matrix at `B=2000, N=1e6` is 16 GB in int64 — chunk the bootstrap and/or downcast to int32 when `N < 2^31`.

## B.10 Testing strategy (a differentiator — over-invest here)
1. **Known-value tests.** Hand-computed ECE/Brier/AUROC on tiny fixtures; DeLong p-values matched to R's `pROC::roc.test` reference numbers committed as golden values.
2. **Property-based (Hypothesis).** Invariants: metrics in valid ranges; CI brackets point estimate; permutation invariance; `compare(a,a)` ⇒ `p≈1, Δ≈0`.
3. **Statistical-correctness simulation.** The headline test: simulate data where the true metric is known, build many 95% CIs, assert empirical coverage ∈ [0.93, 0.97]. **No incumbent ships this — putting it in CI is a marketing asset.**
4. **Cross-framework parity.** Same inputs as numpy / torch / jax arrays produce identical results within tolerance.
5. **Determinism.** Same seed ⇒ identical bytes.
6. **Coverage gate** ≥ 90%, enforced in CI.

## B.11 Release engineering
- SemVer; public API (B.4) frozen at `1.0`. Pre-1.0 may break with CHANGELOG notes.
- Trusted-publisher PyPI release via GitHub Actions on tag.
- `CITATION.cff` + Zenodo DOI on first tagged release (needed for the paper and JMLR MLOSS).
- Docs on Read the Docs (free for OSS), MkDocs Material + `mkdocstrings` auto-API.

---

# PART C — RESEARCH, PAPER & LAUNCH

## C.1 The empirical study (the launch hook) — `benchmarks/`
**Thesis to test:** *a large fraction of published "method A beats method B" calibration/UQ claims are not statistically significant once CIs and paired tests are applied.*

Protocol:
1. **Collect** 10–20 public, reproducible comparisons: standard image classifiers (CIFAR-10/100, ImageNet-val logits where available), tabular benchmarks, and a handful of LLM-confidence datasets. Use only released logits/predictions to avoid retraining cost (keeps it zero-GPU).
2. **Recompute** each comparison's headline metric (ECE/Brier/AUROC) through `reliably`, attaching BCa CIs and paired-bootstrap / DeLong p-values.
3. **Tabulate** the fraction of comparisons whose 95% CIs overlap / whose p > 0.05 after Holm correction.
4. **Report** honestly and conservatively; release every script in `benchmarks/run_study.py` for one-command reproduction.

Deliverable: a figure (forest plot of `Δ` with CIs) + a single sentence ("X of N published comparisons are not significant at α=0.05") that is the tweet, the HN title, and the paper's abstract.

## C.2 Companion paper outline (arXiv → JMLR MLOSS)
1. **Abstract** — the library + the empirical finding.
2. **Introduction** — reproducibility gap; binning bias of ECE; missing significance testing.
3. **Background** — calibration metrics, proper scoring, DeLong, bootstrap, disentanglement.
4. **The `reliably` library** — design, API, the significance-first abstraction, the statistical-coverage test suite.
5. **Empirical study** — C.1 results.
6. **Related work** — Uncertainty Toolbox, netcal, Torch-Uncertainty, LM-Polygraph, UQLM, disentanglement_lib; positioning table.
7. **Limitations & conclusion.**

Submit to arXiv first (citation handle + credibility), then JMLR MLOSS once a user community exists (stars/contributors), and target a NeurIPS/ICLR reproducibility or trustworthy-ML workshop.

## C.3 README structure (the landing page)
1. One-line pitch + animated GIF of the report card / reliability diagram with band.
2. `pip install reliably` in the first screen.
3. Five-line copy-paste example producing a report with CIs.
4. The "wow" snippet: `rb.compare(...) → "AUROC difference NOT significant (p=0.19)"`.
5. Comparison table vs. incumbents (maintenance status, CIs, significance tests, LLM, representation) — only `reliably` ticks the last four columns.
6. Links: docs, paper, examples, "good first issues."

## C.4 Launch sequence
1. **Pre-launch (weeks −3 to 0):** build in public — short progress posts on X and r/MachineLearning; seed "good first issue" labels; get 2–3 friends to try the API and give quotes.
2. **Day 0 (mid-week, ~9am PT):** arXiv preprint live → Show HN ("Show HN: reliably — calibration/UQ metrics with confidence intervals and significance tests") with a detailed founder comment (motivation, stack, limitations, the empirical finding) within 5 minutes → simultaneous r/MachineLearning "[P]" post leading with the empirical result → X thread with the forest-plot GIF.
3. **Days 1–7:** respond to every issue/comment within hours; ship adapters (sklearn `evaluate`, HF `evaluate`) to ride ecosystems; write a "how I built the bootstrap engine" follow-up post.
4. **Weeks 2–8:** ship the LLM and representation arms based on which gets more demand; submit MLOSS; propose a workshop talk.

## C.5 Build timeline (solo)
| Weeks | Deliverable |
|---|---|
| 1–3 | backend, results model, bootstrap+BCa, DeLong, calibration+scoring+AUROC metrics, known-value + property tests |
| 4–5 | recalibration (temperature, isotonic, platt, beta), viz (reliability diagram w/ band), HTML report card |
| 6–7 | empirical study (`benchmarks/`), statistical-coverage CI test, arXiv draft |
| 8 | docs site, README + GIF, packaging, ≥90% coverage, polish |
| 9 | launch (arXiv + Show HN + r/ML + X) |
| 10–16 | LLM arm, representation arm, MLOSS submission, integrations |

## C.6 Risk register
| Risk | Mitigation |
|---|---|
| "Yet another calibration lib" | Lead with significance-first wedge + empirical reproducibility result; position as a *reproducibility* tool. |
| Torch-Uncertainty overlap | Differentiate: it trains UQ methods; you evaluate any model's outputs post-hoc with stats. Ship an adapter so they compose. |
| LLM incumbents (UQLM, LM-Polygraph) | Integrate via adapters, don't reimplement; own the calibration+significance layer. |
| Solo maintenance | numpy-light core, ≥90% coverage, "good first issue" funnel, defer heavy arms until demand pulls them. |
| Empirical claim overstated | Conservative framing, release all reproduction code, frame as "comparisons lack reported uncertainty." |
| PyPI/name collision | Verify `reliably`/`trustcal` availability before first commit. |
| Weak launch (<150 stars wk1) | Re-pitch fully on the reproducibility angle; push the paper at a workshop. |

## C.7 Immediate next actions (day-one checklist)
1. Verify `reliably` (and fallback `trustcal`) free on PyPI + GitHub; create the repo with the B.7 layout.
2. Drop in `pyproject.toml`, CI, license (Apache-2.0), `CITATION.cff`, ruff/mypy config.
3. Implement `_core/backend.py` → `stats/bootstrap.py` → `stats/delong.py` first; they unblock everything.
4. Implement `metrics/calibration.py` (ECE, adaptive, debiased, smECE) + `scoring.py` + `discrimination.py`, each returning `MetricResult` with a CI.
5. Wire `api.evaluate` + `api.compare`; write the five-line README example and confirm it runs.
6. Stand up the statistical-coverage test (B.10 #3) — it is both correctness insurance and a launch talking point.

---

*End of document. Build A→B in order; C runs in parallel from week 6. The non-negotiable identity of this project, present from the first commit, is: every metric carries an interval, every comparison carries a test.*
