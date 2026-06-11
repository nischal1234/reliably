"""Tests for report.to_html(), report.to_markdown(), reliability_diagram, and extras."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import reliably as rb
from reliably._core.results import Report
from reliably._core.validation import prepare_inputs
from reliably.report.render import to_html, to_markdown
from reliably.repr import disentanglement

if TYPE_CHECKING:
    SampleReport = tuple[Report, np.ndarray, np.ndarray]


@pytest.fixture()
def sample_report() -> tuple[Report, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 300)
    p = rng.uniform(0, 1, 300)
    report = rb.evaluate(y, p, ci=None)
    return report, y, p


class TestToHtml:
    def test_returns_html_string(self, sample_report: SampleReport) -> None:
        report, _, _ = sample_report
        html = report.to_html()
        assert isinstance(html, str)
        assert "<html" in html

    def test_standalone_function(self, sample_report: SampleReport) -> None:
        report, _, _ = sample_report
        assert "<html" in to_html(report)

    def test_writes_file(
        self,
        sample_report: SampleReport,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        report, _, _ = sample_report
        out = tmp_path / "report.html"  # type: ignore[operator]
        html = report.to_html(path=out)
        assert out.exists()
        assert out.read_text(encoding="utf-8") == html

    def test_contains_metric_names(self, sample_report: SampleReport) -> None:
        report, _, _ = sample_report
        html = report.to_html()
        assert "ECE" in html or "ece" in html.lower()

    def test_empty_report(self) -> None:
        r = Report(task="binary", metrics={}, n=100, meta={})
        assert "<html" in r.to_html()


class TestToMarkdown:
    def test_returns_markdown(self, sample_report: SampleReport) -> None:
        report, _, _ = sample_report
        md = report.to_markdown()
        assert isinstance(md, str)
        assert "| Metric" in md

    def test_standalone_function(self, sample_report: SampleReport) -> None:
        assert "| Metric" in to_markdown(sample_report[0])

    def test_contains_all_metrics(self, sample_report: SampleReport) -> None:
        report, _, _ = sample_report
        md = report.to_markdown()
        for name in report.metrics:
            assert name in md

    def test_no_ci_row(self) -> None:
        r = Report(task="binary", metrics={}, n=50, meta={})
        assert "N:** 50" in r.to_markdown()

    def test_extra_decomposition(self, sample_report: SampleReport) -> None:
        report, _, _ = sample_report
        md = report.to_markdown()
        if "Brier" in report.metrics and report.metrics["Brier"].extra:
            assert "reliability" in md or "uncertainty" in md


class TestReliabilityDiagram:
    def test_returns_axes(self, sample_report: SampleReport) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        report, y, p = sample_report
        ax = report.reliability_diagram(y, p, band=False)
        assert ax is not None
        plt.close("all")

    def test_no_band(self, sample_report: SampleReport) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        report, y, p = sample_report
        ax = report.reliability_diagram(y, p, band=False, n_bins=10)
        assert ax is not None
        plt.close("all")

    def test_existing_axes(self, sample_report: SampleReport) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        report, y, p = sample_report
        fig, ax = plt.subplots()
        result_ax = report.reliability_diagram(y, p, band=False, ax=ax)
        assert result_ax is ax
        plt.close("all")

    def test_multiclass(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 200)
        p = rng.dirichlet([1, 1, 1], 200)
        report = rb.evaluate(y, p, ci=None)
        ax = report.reliability_diagram(y, p, band=False)
        assert ax is not None
        plt.close("all")

    def test_confidence_histogram(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from reliably.viz.diagrams import confidence_histogram

        p = np.random.default_rng(1).uniform(0, 1, 200)
        ax = confidence_histogram(p)
        assert ax is not None
        plt.close("all")


class TestValidationEdgeCases:
    def test_2d_ytrue_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            prepare_inputs(np.zeros((3, 2)), np.zeros(3))

    def test_length_mismatch_1d(self) -> None:
        with pytest.raises(ValueError, match="length"):
            prepare_inputs(np.zeros(5), np.zeros(3))

    def test_length_mismatch_2d(self) -> None:
        with pytest.raises(ValueError, match="length"):
            prepare_inputs(np.zeros(5), np.zeros((3, 2)))

    def test_3d_yprob_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D or 2-D"):
            prepare_inputs(np.zeros(3), np.zeros((3, 2, 2)))

    def test_invalid_task_raises(self) -> None:
        with pytest.raises(ValueError, match="task"):
            prepare_inputs(np.zeros(5), np.zeros(5), task="regression")

    def test_explicit_binary_task(self) -> None:
        p = np.full((5, 2), 0.5)
        _, _, task = prepare_inputs(np.zeros(5, dtype=int), p, task="binary")
        assert task == "binary"

    def test_explicit_multiclass_task(self) -> None:
        p = np.full((5, 3), 1.0 / 3)
        _, _, task = prepare_inputs(np.zeros(5, dtype=int), p, task="multiclass")
        assert task == "multiclass"


class TestDisentanglementSuite:
    def test_mig_sap_dci(self) -> None:
        rng = np.random.default_rng(7)
        z = rng.normal(0, 1, (80, 4))
        f = rng.normal(0, 1, (80, 3))
        results = disentanglement(z, f, metrics=("mig", "sap", "dci"), ci=None)
        assert "mig" in results
        assert "sap" in results
        assert "dci" in results

    def test_factorvae_irs(self) -> None:
        rng = np.random.default_rng(8)
        z = rng.normal(0, 1, (80, 4))
        f = rng.integers(0, 3, (80, 3)).astype(float)
        results = disentanglement(z, f, metrics=("factorvae", "irs"), ci=None)
        assert "factorvae" in results
        assert "irs" in results


class TestCompareAPIPaths:
    def setup_method(self) -> None:
        rng = np.random.default_rng(99)
        self.y = rng.integers(0, 2, 200)
        self.pa = rng.uniform(0, 1, 200)
        self.pb = rng.uniform(0, 1, 200)

    def test_delong_with_raw_arrays(self) -> None:
        cr = rb.compare(self.pa, self.pb, metric="auroc", y_true=self.y)
        assert 0.0 <= cr.p_value <= 1.0
        assert cr.test == "delong"

    def test_compare_missing_ytrue_raises(self) -> None:
        with pytest.raises(ValueError, match="y_true"):
            rb.compare(self.pa, self.pb, metric="auroc")

    def test_compare_missing_ytrue_b_raises(self) -> None:
        ra = rb.evaluate(self.y, self.pa, ci=None)
        with pytest.raises(ValueError, match="y_true"):
            rb.compare(ra, self.pb, metric="auroc")

    def test_paired_bootstrap_raw_arrays(self) -> None:
        cr = rb.compare(self.pa, self.pb, metric="brier", y_true=self.y)
        assert 0.0 <= cr.p_value <= 1.0
        assert cr.test == "paired_bootstrap"

    def test_compare_missing_metric_raises(self) -> None:
        ra = rb.evaluate(self.y, self.pa, metrics=["ece"], ci=None)
        rb_ = rb.evaluate(self.y, self.pb, metrics=["ece"], ci=None)
        with pytest.raises(ValueError, match="not found"):
            rb.compare(ra, rb_, metric="auroc", y_true=self.y)

    def test_compare_reports_no_ytrue(self) -> None:
        ra = rb.evaluate(self.y, self.pa, metrics=["brier"], ci=None)
        rb_ = rb.evaluate(self.y, self.pb, metrics=["brier"], ci=None)
        cr = rb.compare(ra, rb_, metric="brier")
        assert isinstance(cr.delta, float)
