# Quickstart

## Install

```bash
pip install "reliably[all]"
```

## Evaluate a model

```python
import numpy as np
import reliably as rb

rng = np.random.default_rng(0)
y_true = rng.integers(0, 2, 1000)
y_prob = rng.uniform(0, 1, 1000)

report = rb.evaluate(y_true, y_prob)
print(report.summary())
```

## Compare two models

```python
report_a = rb.evaluate(y_true, probs_a)
report_b = rb.evaluate(y_true, probs_b)

result = rb.compare(report_a, report_b, y_true=y_true, metric="auroc")
print(f"p-value: {result.p_value:.3f}, significant: {result.significant}")
```

## Recalibrate

```python
# Fit on a held-out calibration split
calibrator = rb.recalibrate(p_cal, y_cal, method="temperature")

# Apply to test set
p_calibrated = calibrator.transform(p_test)
report_after = rb.evaluate(y_test, p_calibrated)
```

## Export

```python
# HTML report (requires pip install reliably[report])
report.to_html("report.html")

# Markdown table
print(report.to_markdown())

# Reliability diagram (requires pip install reliably[viz])
ax = report.reliability_diagram(y_true, y_prob)
ax.figure.savefig("diagram.png", dpi=150)
```
