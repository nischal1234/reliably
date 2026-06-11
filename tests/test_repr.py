"""Tests for representation quality metrics."""

from __future__ import annotations

import numpy as np
import pytest


def _make_latents(n: int = 200, d: int = 4, k: int = 3, seed: int = 0) -> tuple[
    np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (n, d))
    f = rng.normal(0, 1, (n, k))
    return z, f


class TestMIG:
    def test_valid_range(self) -> None:
        from reliably.repr.mig import mig

        z, f = _make_latents()
        r = mig(z, f, ci=None)
        assert 0.0 <= r.value <= 1.0

    def test_name(self) -> None:
        from reliably.repr.mig import mig

        z, f = _make_latents()
        r = mig(z, f, ci=None)
        assert r.name == "MIG"

    def test_n_field(self) -> None:
        from reliably.repr.mig import mig

        z, f = _make_latents(n=150)
        r = mig(z, f, ci=None)
        assert r.n == 150

    def test_ci_brackets(self) -> None:
        from reliably.repr.mig import mig

        z, f = _make_latents()
        r = mig(z, f, ci="bca", n_bootstrap=50, seed=0)
        assert r.ci is not None
        assert r.ci.low <= r.value <= r.ci.high


class TestSAP:
    def test_valid_range(self) -> None:
        from reliably.repr.sap import sap

        z, f = _make_latents()
        r = sap(z, f, ci=None)
        assert r.value >= 0.0

    def test_name(self) -> None:
        from reliably.repr.sap import sap

        z, f = _make_latents()
        r = sap(z, f, ci=None)
        assert r.name == "SAP"


class TestDCI:
    def test_valid_range(self) -> None:
        from reliably.repr.dci import dci

        z, f = _make_latents(n=100)
        r = dci(z, f, ci=None)
        assert 0.0 <= r.value <= 1.0

    def test_extra_fields(self) -> None:
        from reliably.repr.dci import dci

        z, f = _make_latents(n=100)
        r = dci(z, f, ci=None)
        assert r.extra is not None
        assert "disentanglement" in r.extra
        assert "completeness" in r.extra
        assert "informativeness" in r.extra


class TestFactorVAE:
    def test_valid_range(self) -> None:
        from reliably.repr.factorvae import factorvae_metric

        z, f = _make_latents(n=300)
        r = factorvae_metric(z, f, n_votes=100, ci=None)
        assert 0.0 <= r.value <= 1.0

    def test_name(self) -> None:
        from reliably.repr.factorvae import factorvae_metric

        z, f = _make_latents(n=200)
        r = factorvae_metric(z, f, n_votes=50, ci=None)
        assert r.name == "FactorVAE"


class TestIRS:
    def test_valid_range(self) -> None:
        from reliably.repr.irs import irs

        z, f = _make_latents(n=300)
        r = irs(z, f, n_interventions=50, ci=None)
        assert 0.0 <= r.value <= 1.0

    def test_name(self) -> None:
        from reliably.repr.irs import irs

        z, f = _make_latents(n=200)
        r = irs(z, f, ci=None)
        assert r.name == "IRS"


class TestDisentanglementAPI:
    def test_returns_dict(self) -> None:
        from reliably.repr import disentanglement

        z, f = _make_latents(n=100)
        results = disentanglement(z, f, metrics=("mig",), ci=None)
        assert "mig" in results

    def test_multiple_metrics(self) -> None:
        from reliably.repr import disentanglement

        z, f = _make_latents(n=100)
        results = disentanglement(z, f, metrics=("mig", "sap"), ci=None)
        assert set(results.keys()) == {"mig", "sap"}
