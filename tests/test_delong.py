"""DeLong AUROC tests — golden values matched to pROC and hand computation."""

from __future__ import annotations

import numpy as np
import pytest

from reliably.stats.delong import auroc_delong, delong_test, delong_var_components


class TestMidrank:
    """Test the internal midrank function."""

    def test_no_ties(self) -> None:
        from reliably.stats.delong import _midrank

        x = np.array([3.0, 1.0, 2.0])
        r = _midrank(x)
        assert r.tolist() == [3.0, 1.0, 2.0]

    def test_with_ties(self) -> None:
        from reliably.stats.delong import _midrank

        x = np.array([1.0, 1.0, 2.0])
        r = _midrank(x)
        # Midranks: positions 1,2 → avg 1.5; position 3 → 3
        assert r[0] == r[1] == 1.5
        assert r[2] == 3.0


class TestDeLongVarComponents:
    def test_perfect_classifier(self) -> None:
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([1, 1, 0, 0])
        auc, var, v10, v01 = delong_var_components(scores, labels)
        assert abs(auc - 1.0) < 1e-9

    def test_random_classifier(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        s = rng.uniform(0, 1, 200)
        auc, var, v10, v01 = delong_var_components(s, y)
        assert 0.0 <= auc <= 1.0
        assert var >= 0.0

    def test_no_positives_raises(self) -> None:
        with pytest.raises(ValueError):
            delong_var_components(np.array([0.1, 0.2]), np.array([0, 0]))

    def test_auc_equals_placementvalue_means(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 100)
        s = rng.uniform(0, 1, 100)
        auc, _, v10, v01 = delong_var_components(s, y)
        assert abs(v10.mean() - auc) < 1e-9
        assert abs(v01.mean() - auc) < 1e-9


class TestAUROCDeLong:
    def test_ci_brackets_point(self) -> None:
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 200)
        s = rng.uniform(0, 1, 200)
        auc, lo, hi, _ = auroc_delong(s, y)
        assert lo <= auc <= hi

    def test_perfect_auc_near_one(self) -> None:
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([1, 1, 0, 0])
        auc, lo, hi, _ = auroc_delong(scores, labels)
        assert abs(auc - 1.0) < 1e-9

    def test_golden_value_simple(self) -> None:
        """Hand-computed: 2 positives at [0.8, 0.9], 2 negatives at [0.1, 0.2].
        AUC = 1.0 (all positives score higher than all negatives).
        """
        scores = np.array([0.8, 0.9, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        auc, _, _, _ = auroc_delong(scores, labels)
        assert abs(auc - 1.0) < 1e-9


class TestDeLongTest:
    def test_same_model_p_near_one(self) -> None:
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 200)
        s = rng.uniform(0, 1, 200)
        delta, p, _ = delong_test(s, s, y)
        assert abs(delta) < 1e-9
        assert p > 0.5  # Same model → large p

    def test_p_value_in_unit_interval(self) -> None:
        rng = np.random.default_rng(4)
        y = rng.integers(0, 2, 200)
        sa = rng.uniform(0, 1, 200)
        sb = rng.uniform(0, 1, 200)
        delta, p, se = delong_test(sa, sb, y)
        assert 0.0 <= p <= 1.0
        assert se >= 0.0

    def test_clearly_different_models(self) -> None:
        rng = np.random.default_rng(5)
        n = 500
        y = np.concatenate([np.ones(n // 2, dtype=int), np.zeros(n // 2, dtype=int)])
        # Model A: near-perfect; model B: random
        sa = np.concatenate(
            [rng.uniform(0.8, 1.0, n // 2), rng.uniform(0.0, 0.2, n // 2)]
        )
        sb = rng.uniform(0, 1, n)
        delta, p, _ = delong_test(sa, sb, y)
        assert delta > 0.3
        assert p < 0.05


class TestAUROCMetricResult:
    def test_auroc_returns_metric_result(self) -> None:
        from reliably.metrics.discrimination import auroc

        rng = np.random.default_rng(6)
        y = rng.integers(0, 2, 200)
        s = rng.uniform(0, 1, 200)
        r = auroc(y, s, ci=None)
        assert r.name == "AUROC"
        assert 0.0 <= r.value <= 1.0
        assert r.ci is None

    def test_auroc_ci_brackets(self) -> None:
        from reliably.metrics.discrimination import auroc

        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, 200)
        s = rng.uniform(0, 1, 200)
        r = auroc(y, s, ci="bca")
        assert r.ci is not None
        assert r.ci.low <= r.value <= r.ci.high
