# Recalibration

## Methods

| Method | Key | Notes |
|--------|-----|-------|
| Temperature scaling | `"temperature"` | Single scalar; preserves accuracy |
| Platt scaling | `"platt"` | Logistic regression on binary scores |
| Isotonic regression | `"isotonic"` | Non-parametric; requires scikit-learn |
| Beta calibration | `"beta"` | Beta CDF transform |
| Histogram binning | `"histogram"` | Non-parametric; step function |
| Vector scaling | `"vector"` | Per-class temperature (multiclass) |
| Matrix scaling | `"matrix"` | Full affine transform (multiclass) |

## Example

```python
import reliably as rb

# Fit on calibration split
cal = rb.recalibrate(p_cal, y_cal, method="temperature")

# Apply to test set
p_test_cal = cal.transform(p_test)

# Compare before/after
before = rb.evaluate(y_test, p_test, ci=None)
after  = rb.evaluate(y_test, p_test_cal, ci=None)

print("Before:", before["smECE"].value)
print("After: ", after["smECE"].value)
```
