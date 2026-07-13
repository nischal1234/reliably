# reliably

**Statistically rigorous model reliability evaluation — every metric carries a confidence interval, every comparison carries a significance test.**

```
pip install reliably
```

---

## Five-line quick start

```python
import numpy as np
import reliably as rb

y_true = np.array([0, 1, 1, 0, 1, ...])  # your labels
y_prob = model.predict_proba(X)           # your probabilities

report = rb.evaluate(y_true, y_prob)
print(report.summary())
# Report(task=binary, n=1000)
#   ECE=0.0312 [0.0211, 0.0421]
#   smECE=0.0289 [0.0185, 0.0398]
#   Brier=0.1842 [0.1714, 0.1971]
#   NLL=0.5103 [0.4887, 0.5319]
#   AUROC=0.7841 [0.7512, 0.8170]
```

## The "wow" comparison

```python
result = rb.compare(report_a, report_b, y_true=y_true, metric="auroc")
if not result.significant:
    print(f"AUROC difference NOT significant (p={result.p_value:.2f})")
```

---

## Why reliably?

| Feature | reliably | netcal | Uncertainty Toolbox | Torch-Uncertainty |
|---------|----------|--------|--------------------|--------------------|
| Bootstrap CIs on every metric | ✅ | ❌ | Partial | ❌ |
| DeLong significance test | ✅ | ❌ | ❌ | ❌ |
| Paired bootstrap comparison | ✅ | ❌ | ❌ | ❌ |
| LLM confidence calibration | ✅ | ❌ | ❌ | ❌ |
| Representation quality (MIG, DCI…) | ✅ | ❌ | ❌ | ❌ |
| Statistical coverage test suite | ✅ | ❌ | ❌ | ❌ |
| Framework-agnostic (numpy/torch/jax) | ✅ | ✅ | ✅ | ✗ (torch only) |
| Actively maintained | ✅ | ❌ | Partial | ✅ |

---

## Metrics included

**Calibration**
- ECE (equal-width and adaptive binning)
- SmoothECE (kernel regression, recommended headline metric)
- Debiased ECE² (bias-corrected)
- Maximum Calibration Error (MCE)
- Classwise / marginal ECE

**Scoring rules**
- Brier score with Murphy decomposition (reliability + resolution + uncertainty)
- Negative Log-Likelihood

**Discrimination**
- AUROC with DeLong analytic CI and two-sample correlated test

**Recalibration**
- Temperature scaling, vector scaling, matrix scaling
- Platt scaling, isotonic regression, beta calibration, histogram binning

**Representation quality** (`reliably.repr`)
- MIG, SAP, DCI, FactorVAE, IRS — all with bootstrap CIs

**LLM confidence** (`reliably.llm`)
- Verbalized confidence parsing
- Sequence log-probability → confidence
- Adapters for LM-Polygraph and UQLM

---

## Installation

```bash
# Core (numpy + scipy only)
pip install reliably

# With visualization
pip install reliably[viz]

# With HTML reports
pip install reliably[report]

# With scikit-learn (isotonic regression, DCI)
pip install reliably[sklearn]

# Everything
pip install reliably[all]
```

---

## Links

- [Documentation](https://reliably.readthedocs.io)

---

## License

Apache-2.0
