"""Tests for stats/tests.py, api.py, llm/, viz/, report/, and discrimination bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

import reliably as rb


# ---------------------------------------------------------------------------
# stats/tests.py
# ---------------------------------------------------------------------------

class TestHolmBonferroni:
    def test_all_significant(self) -> None:
        from reliably.stats.tests import holm_bonferroni
        result = holm_bonferroni([0.001, 0.002, 0.003], level=0.05)
        assert all(result)

    def test_none_significant(self) -> None:
        from reliably.stats.tests import holm_bonferroni
        result = holm_bonferroni([0.5, 0.6, 0.7], level=0.05)
        assert not any(result)

    def test_partial_significant(self) -> None:
        from reliably.stats.tests import holm_bonferroni
        # p=[0.01, 0.04, 0.2] at α=0.05:
        #   rank0: 0.01 ≤ 0.05/3 ≈ 0.0167 → reject
        #   rank1: 0.04 > 0.05/2 = 0.025  → fail, stop
        result = holm_bonferroni([0.01, 0.04, 0.2], level=0.05)
        assert result[0] is True
        assert result[1] is False
        assert result[2] is False

    def test_single_pvalue(self) -> None:
        from reliably.stats.tests import holm_bonferroni
        assert holm_bonferroni([0.03], level=0.05) == [True]
        assert holm_bonferroni([0.06], level=0.05) == [False]


class TestBenjaminiHochberg:
    def test_basic(self) -> None:
        from reliably.stats.tests import benjamini_hochberg
        result = benjamini_hochberg([0.01, 0.04, 0.2], level=0.05)
        assert result[0] is True

    def test_none_significant(self) -> None:
        from reliably.stats.tests import benjamini_hochberg
        result = benjamini_hochberg([0.4, 0.5, 0.6], level=0.05)
        assert not any(result)


class TestApplyCorrection:
    def test_none_correction(self) -> None:
        from reliably.stats.tests import apply_correction
        result = apply_correction([0.03, 0.06], None, level=0.05)
        assert result == [True, False]

    def test_holm_correction(self) -> None:
        from reliably.stats.tests import apply_correction
        result = apply_correction([0.01, 0.5], "holm")
        assert result[0] is True

    def test_bh_correction(self) -> None:
        from reliably.stats.tests import apply_correction
        result = apply_correction([0.01, 0.5], "bh")
        assert result[0] is True

    def test_unknown_correction_raises(self) -> None:
        from reliably.stats.tests import apply_correction
        with pytest.raises(ValueError):
            apply_correction([0.01], "unknown")


class TestPairedBootstrap:
    def test_same_estimator_p_near_one(self) -> None:
        from reliably.stats.tests import paired_bootstrap_test
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 200)
        delta, ci, p = paired_bootstrap_test(
            lambda idx: x[idx].mean(), lambda idx: x[idx].mean(),
            len(x), point_a=x.mean(), point_b=x.mean(), n_boot=200, seed=0
        )
        assert abs(delta) < 1e-9
        assert p > 0.1

    def test_different_estimators(self) -> None:
        from reliably.stats.tests import paired_bootstrap_test
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 300)
        y = x + 5.0  # clearly different
        delta, ci, p = paired_bootstrap_test(
            lambda idx: x[idx].mean(), lambda idx: y[idx].mean(),
            len(x), point_a=x.mean(), point_b=y.mean(), n_boot=200, seed=0
        )
        assert p < 0.01

    def test_ci_brackets_delta(self) -> None:
        from reliably.stats.tests import paired_bootstrap_test
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        delta, ci, p = paired_bootstrap_test(
            lambda idx: x[idx].mean(), lambda idx: y[idx].mean(),
            len(x), point_a=x.mean(), point_b=y.mean(), n_boot=200, seed=0
        )
        assert ci.low <= delta <= ci.high


# ---------------------------------------------------------------------------
# metrics/discrimination.py - bootstrap path
# ---------------------------------------------------------------------------

class TestAUROCBootstrap:
    def test_percentile_ci(self) -> None:
        from reliably.metrics.discrimination import auroc
        rng = np.random.default_rng(10)
        y = rng.integers(0, 2, 200)
        s = rng.uniform(0, 1, 200)
        r = auroc(y, s, ci="percentile", n_bootstrap=200, seed=0)
        assert r.ci is not None
        assert r.ci.method == "percentile"
        assert r.ci.low <= r.value <= r.ci.high

    def test_2d_input(self) -> None:
        from reliably.metrics.discrimination import auroc
        rng = np.random.default_rng(11)
        y = rng.integers(0, 2, 200)
        p = rng.uniform(0, 1, 200)
        p2d = np.stack([1 - p, p], axis=1)
        r1 = auroc(y, p, ci=None)
        r2 = auroc(y, p2d, ci=None)
        assert abs(r1.value - r2.value) < 1e-10


# ---------------------------------------------------------------------------
# LLM arm
# ---------------------------------------------------------------------------

class TestVerbalized:
    def test_percentage(self) -> None:
        from reliably.llm.verbalized import parse_verbalized_confidence
        assert abs(parse_verbalized_confidence("I am 85% sure.") - 0.85) < 1e-9

    def test_decimal(self) -> None:
        from reliably.llm.verbalized import parse_verbalized_confidence
        assert abs(parse_verbalized_confidence("confidence 0.75") - 0.75) < 1e-9

    def test_word(self) -> None:
        from reliably.llm.verbalized import parse_verbalized_confidence
        assert parse_verbalized_confidence("probably correct") == 0.75

    def test_none(self) -> None:
        from reliably.llm.verbalized import parse_verbalized_confidence
        assert parse_verbalized_confidence("no confidence here") is None

    def test_batch(self) -> None:
        from reliably.llm.verbalized import parse_verbalized_batch
        results = parse_verbalized_batch(["90%", "maybe", "unknown"])
        assert abs(results[0] - 0.9) < 1e-9
        assert results[1] == 0.5
        assert results[2] is None


class TestLLMAdapters:
    def test_logprobs_to_confidence(self) -> None:
        from reliably.llm.adapters import logprobs_to_confidence
        lp = np.array([-0.1, -0.2, -0.3])
        c = logprobs_to_confidence(lp)
        assert 0.0 < c[0] <= 1.0

    def test_from_lm_polygraph_dict(self) -> None:
        from reliably.llm.adapters import from_lm_polygraph
        result = from_lm_polygraph({"confidence": [0.8, 0.6]})
        assert "confidence" in result
        assert abs(result["confidence"][0] - 0.8) < 1e-9

    def test_from_uqlm_dict(self) -> None:
        from reliably.llm.adapters import from_uqlm
        result = from_uqlm({"probabilities": [0.9, 0.7]})
        assert "confidence" in result
        assert abs(result["confidence"][0] - 0.9) < 1e-9


class TestLLMEvaluate:
    def test_probability_kind(self) -> None:
        from reliably.llm.evaluate import evaluate
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 100).tolist()
        conf = rng.uniform(0, 1, 100).tolist()
        report = evaluate(["ans"] * 100, y, conf, kind="probability", ci=None)
        assert "smECE" in report.metrics

    def test_verbalized_kind(self) -> None:
        from reliably.llm.evaluate import evaluate
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 10).tolist()
        conf = ["80%", "probably", "certain", "60%", "maybe",
                "0.9", "unlikely", "75%", "maybe", "90%"]
        report = evaluate(["ans"] * 10, y, conf, kind="verbalized", ci=None)
        assert "Brier" in report.metrics


# ---------------------------------------------------------------------------
# viz/diagrams.py
# ---------------------------------------------------------------------------

class TestVizDiagrams:
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("matplotlib"),
        reason="matplotlib not installed",
    )
    def test_reliability_diagram(self) -> None:
        from reliably.viz.diagrams import reliability_diagram
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        p = rng.uniform(0, 1, 200)
        ax = reliability_diagram(y, p, band=False)
        assert ax is not None

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("matplotlib"),
        reason="matplotlib not installed",
    )
    def test_confidence_histogram(self) -> None:
        from reliably.viz.diagrams import confidence_histogram
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 200)
        ax = confidence_histogram(p)
        assert ax is not None


# ---------------------------------------------------------------------------
# report/render.py
# ---------------------------------------------------------------------------

class TestReportRender:
    def test_to_markdown(self) -> None:
        from reliably.report.render import to_markdown
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        p = rng.uniform(0, 1, 200)
        report = rb.evaluate(y, p, ci=None)
        md = to_markdown(report)
        assert "| Metric" in md
        assert "Brier" in md

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("jinja2"),
        reason="jinja2 not installed",
    )
    def test_to_html(self) -> None:
        from reliably.report.render import to_html
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        p = rng.uniform(0, 1, 200)
        report = rb.evaluate(y, p, ci=None)
        html = to_html(report)
        assert "<html" in html
        assert "Brier" in html


# ---------------------------------------------------------------------------
# api.py — compare and evaluate edge cases
# ---------------------------------------------------------------------------

class TestEvaluateAPI:
    def test_evaluate_all_metrics(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        p = rng.uniform(0, 1, 200)
        report = rb.evaluate(y, p, metrics="all", ci=None)
        assert len(report.metrics) > 0

    def test_evaluate_metric_list(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 200)
        p = rng.uniform(0, 1, 200)
        report = rb.evaluate(y, p, metrics=["brier", "nll"], ci=None)
        assert "Brier" in report.metrics
        assert "NLL" in report.metrics

    def test_evaluate_multiclass(self) -> None:
        rng = np.random.default_rng(2)
        y = rng.integers(0, 3, 200)
        p = rng.dirichlet([1, 1, 1], 200)
        report = rb.evaluate(y, p, ci=None)
        assert "Brier" in report.metrics

    def test_evaluate_with_ci(self) -> None:
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 200)
        p = rng.uniform(0, 1, 200)
        report = rb.evaluate(y, p, ci="bca", n_bootstrap=100, seed=0)
        for mr in report.metrics.values():
            assert mr.ci is not None
            assert mr.ci.low <= mr.value <= mr.ci.high

    def test_report_summary(self) -> None:
        rng = np.random.default_rng(4)
        y = rng.integers(0, 2, 200)
        p = rng.uniform(0, 1, 200)
        report = rb.evaluate(y, p, ci=None)
        s = report.summary()
        assert "Report" in s

    def test_report_getitem(self) -> None:
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, 200)
        p = rng.uniform(0, 1, 200)
        report = rb.evaluate(y, p, ci=None)
        assert report["Brier"].name == "Brier"


class TestCompareAPI:
    def test_compare_same_model_low_delta(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 300)
        p = rng.uniform(0, 1, 300)
        r_a = rb.evaluate(y, p, ci=None)
        r_b = rb.evaluate(y, p, ci=None)
        cr = rb.compare(r_a, r_b, y_true=y, metric="auroc")
        assert abs(cr.delta) < 1e-9

    def test_compare_returns_comparison_result(self) -> None:
        from reliably._core.results import ComparisonResult
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 300)
        p_a = rng.uniform(0, 1, 300)
        p_b = rng.uniform(0, 1, 300)
        r_a = rb.evaluate(y, p_a, ci=None)
        r_b = rb.evaluate(y, p_b, ci=None)
        cr = rb.compare(r_a, r_b, y_true=y)
        assert isinstance(cr, ComparisonResult)
        assert 0.0 <= cr.p_value <= 1.0

    def test_compare_with_raw_arrays_delong(self) -> None:
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 300)
        s_a = rng.uniform(0, 1, 300)
        s_b = rng.uniform(0, 1, 300)
        cr = rb.compare(s_a, s_b, y_true=y, metric="auroc", test="delong")
        assert 0.0 <= cr.p_value <= 1.0


# ---------------------------------------------------------------------------
# backend / validation
# ---------------------------------------------------------------------------

class TestBackend:
    def test_to_numpy_list(self) -> None:
        from reliably._core.backend import to_numpy
        arr = to_numpy([1, 2, 3])
        assert arr.dtype == np.float64

    def test_clip_probs(self) -> None:
        from reliably._core.backend import clip_probs
        result = clip_probs(np.array([0.0, 0.5, 1.0]))
        assert result[0] > 0
        assert result[2] < 1.0

    def test_adaptive_bins_monotone(self) -> None:
        from reliably._core.backend import adaptive_bins
        rng = np.random.default_rng(0)
        c = rng.uniform(0, 1, 200)
        edges = adaptive_bins(c, 10)
        assert np.all(np.diff(edges) >= 0)

    def test_softmax_sums_to_one(self) -> None:
        from reliably._core.backend import softmax
        z = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 2.0]])
        s = softmax(z)
        assert np.allclose(s.sum(axis=1), 1.0)


class TestValidation:
    def test_prepare_binary(self) -> None:
        from reliably._core.validation import prepare_inputs
        y = np.array([0, 1, 1])
        p = np.array([0.2, 0.8, 0.6])
        yt, yp, task = prepare_inputs(y, p)
        assert task == "binary"

    def test_prepare_multiclass(self) -> None:
        from reliably._core.validation import prepare_inputs
        y = np.array([0, 1, 2])
        p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
        yt, yp, task = prepare_inputs(y, p)
        assert task == "multiclass"

    def test_shape_mismatch_raises(self) -> None:
        from reliably._core.validation import prepare_inputs
        with pytest.raises(ValueError):
            prepare_inputs(np.array([0, 1]), np.array([0.5, 0.5, 0.5]))

    def test_auto_normalize(self) -> None:
        from reliably._core.validation import prepare_inputs
        import warnings
        y = np.array([0, 1])
        p = np.array([[0.4, 0.7], [0.3, 0.6]])
        with warnings.catch_warnings(record=True):
            yt, yp, task = prepare_inputs(y, p)
        assert np.allclose(yp.sum(axis=1), 1.0)
