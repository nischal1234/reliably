# Metrics Reference

Every metric returns a `MetricResult(value, ci, n)` — never a bare float.

## Calibration

| Name | Key | Description |
|------|-----|-------------|
| ECE | `"ece"` | Expected Calibration Error (equal-width bins) |
| Adaptive ECE | `"adaptive_ece"` | ECE with equal-mass bins |
| smECE | `"smece"` | Smooth ECE via kernel regression (recommended) |
| Debiased ECE | `"debiased_ece"` | Bias-corrected ECE² |
| MCE | `"mce"` | Maximum Calibration Error |
| cwECE | `"cwece"` | Classwise ECE (multiclass) |

## Scoring

| Name | Key | Description |
|------|-----|-------------|
| Brier | `"brier"` | Brier score with Murphy decomposition |
| NLL | `"nll"` | Negative log-likelihood |

## Discrimination

| Name | Key | Description |
|------|-----|-------------|
| AUROC | `"auroc"` | Area Under ROC curve with DeLong CI |

## Usage

```python
# Default metrics (binary)
report = rb.evaluate(y, p)

# Specific metrics
report = rb.evaluate(y, p, metrics=["ece", "auroc", "brier"])

# All metrics
report = rb.evaluate(y, p, metrics="all")
```
