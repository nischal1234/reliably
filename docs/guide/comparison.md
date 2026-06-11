# Model Comparison

## Statistical tests

- **DeLong test** — used automatically when `metric="auroc"`. Analytic, O(N log N).
- **Paired bootstrap** — used for all other metrics. Permutation-based.

## Multiple comparison correction

| Method | Key |
|--------|-----|
| Holm–Bonferroni (default) | `"holm"` |
| Benjamini–Hochberg | `"bh"` |
| None | `None` |

## Example

```python
import reliably as rb

report_a = rb.evaluate(y, probs_a)
report_b = rb.evaluate(y, probs_b)

# AUROC comparison via DeLong
result = rb.compare(report_a, report_b, y_true=y, metric="auroc")
print(f"ΔAUROC = {result.delta:+.4f}  p = {result.p_value:.3f}  significant = {result.significant}")

# ECE comparison via paired bootstrap
result = rb.compare(report_a, report_b, y_true=y, metric="ece")
```
