# reliably

**Statistically rigorous model reliability evaluation — every metric carries a confidence interval, every comparison carries a significance test.**

```
pip install reliably
```

## Five-line quick start

```python
import numpy as np
import reliably as rb

report = rb.evaluate(y_true, y_prob)
print(report.summary())
# Report(task=binary, n=1000)
#   ECE=0.0312 [0.0211, 0.0421]
#   smECE=0.0289 [0.0185, 0.0398]
#   Brier=0.1842 [0.1714, 0.1971]
#   NLL=0.5103 [0.4887, 0.5319]
#   AUROC=0.7841 [0.7512, 0.8170]

# Render to HTML
report.to_html("report.html")

# Plot reliability diagram
ax = report.reliability_diagram(y_true, y_prob)
```

## Why reliably?

| Feature | reliably | netcal | Uncertainty Toolbox |
|---------|----------|--------|---------------------|
| Bootstrap CIs on every metric | ✅ | ❌ | Partial |
| DeLong significance test | ✅ | ❌ | ❌ |
| Paired bootstrap comparison | ✅ | ❌ | ❌ |
| Representation quality (MIG, DCI…) | ✅ | ❌ | ❌ |
| Framework-agnostic (numpy/torch/jax) | ✅ | ✅ | ✅ |

## Installation

```bash
# Core only (numpy + scipy)
pip install reliably

# With visualization
pip install "reliably[viz]"

# With HTML reports
pip install "reliably[report]"

# Everything
pip install "reliably[all]"
```

## License

Apache 2.0
