# CLAUDE.md — Project Instructions for `reliably`

> This file is read automatically by Claude Code at the start of every session and by
> `claude-code-action` on every CI run. It is the contract. Follow it exactly.
> The full design is in `docs/reliably_technical_design_document.md` (the "TDD"). When this
> file and the TDD disagree, the TDD wins on *math/API*, this file wins on *process*.

## 1. What we are building (one sentence)
A framework-agnostic Python library that turns any model's probabilistic outputs into a
publication-ready reliability report — calibration, uncertainty, discrimination, and
representation quality — where **every metric carries a bootstrap confidence interval and
every model comparison carries a significance test.**

## 2. Non-negotiable invariants (never violate these)
1. A metric function NEVER returns a bare float. It returns a `MetricResult` (value + CI + n).
2. Every stochastic operation takes an explicit `seed`; same inputs + same seed ⇒ identical output (verified in `tests/test_determinism.py`).
3. Core (`metrics`, `stats`, `recalibrate`) depends ONLY on `numpy` + `scipy`. `torch`,
   `matplotlib`, `jinja2`, `scikit-learn` are OPTIONAL extras, imported lazily inside the
   functions that need them. Never add a hard dependency to the core.
4. No network calls, no file writes outside explicit export functions, no telemetry, ever.
5. Public API surface (TDD §B.4) is frozen at v1.0. Before v1.0 it may change, but any change
   must be recorded in `CHANGELOG.md`.
6. New/changed behaviour MUST ship with tests in the same PR. No PR merges with failing CI or
   coverage below 90%.

## 3. Architecture & layering (TDD §B.2, §B.5)
Build order, because each layer unblocks the next:
1. `_core/backend.py`, `_core/results.py`, `_core/validation.py`
2. `stats/bootstrap.py` (percentile + BCa + jackknife), `stats/delong.py`, `stats/tests.py`
3. `metrics/calibration.py`, `metrics/scoring.py`, `metrics/discrimination.py`
4. `recalibrate/*` (temperature → isotonic → platt → beta → histogram → matrix)
5. `api.py` (`evaluate`, `compare`, `recalibrate`)
6. `viz/diagrams.py`, `report/render.py`
7. `repr/*` (mig, sap, dci, factorvae, irs)
8. `llm/*` (adapters only — never reimplement LM-Polygraph/UQLM)

A module may only import from layers above it in this list. No upward or circular imports.

## 4. Math is authoritative in the TDD
Do not invent formulas. Implement exactly what `docs/reliably_technical_design_document.md`
Part A specifies. The two spots that need extra care because the summary glosses subtleties:
- **SmoothECE bandwidth selection (A.1.7)** — match the published `relplot` behaviour; add a
  test that compares against a small committed reference table.
- **IRS (A.6.5)** — implement faithfully to Suter et al. 2019; flag in the PR if uncertain
  rather than approximating silently.
When a formula is ambiguous, open a GitHub issue describing the ambiguity instead of guessing.

## 5. Coding standards
- Python 3.10+, full type hints, `from __future__ import annotations` at top of every module.
- `ruff` clean (line length 100) and `mypy --strict` clean before any commit.
- Docstrings: NumPy style, with a runnable `Examples` block for every public function.
- Prefer pure functions; dataclasses are `frozen=True, slots=True`.
- Numerical safety: clip probabilities to `[1e-12, 1-1e-12]` before any `log`.
- Vectorize the bootstrap (pre-draw a `B×N` index matrix); never loop in Python over B for
  mean-type metrics. See TDD §A.4.5.

## 6. Testing standards (TDD §B.10 — this is a project differentiator)
Every PR must keep all six test categories green:
1. Known-value tests (hand-computed fixtures; DeLong matched to R `pROC::roc.test` golden values).
2. Property-based tests with Hypothesis (ranges, CI brackets point estimate, permutation invariance).
3. Statistical-coverage simulation: build many 95% CIs on data with a known true metric; assert
   empirical coverage ∈ [0.93, 0.97]. This test is also a marketing claim — keep it passing.
4. Cross-framework parity (numpy/torch/jax inputs agree within tolerance).
5. Determinism (same seed ⇒ identical bytes).
6. Coverage ≥ 90%, enforced by `--cov-fail-under=90`.

## 7. Git & PR workflow (how the autonomous loop behaves)
- One issue → one branch → one PR. Branch name: `feat/<module>` or `fix/<short-slug>`.
- Commit messages: Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- Every PR description must list: what it implements, which TDD section, which tests were added,
  and any deviations from the spec.
- Do NOT merge your own PR if a test is failing or coverage dropped. If CI is red, fix it in the
  same PR before merging.
- Keep PRs small: ideally one module or one metric per PR. If a task is large, split it and open
  follow-up issues.
- Update `CHANGELOG.md` in every PR that changes behaviour.

## 8. What to do when blocked or uncertain
- Spec ambiguity → open a GitHub issue labelled `needs-decision`, describe options, do NOT guess.
- A test is impossible to satisfy → it usually means the implementation is wrong; fix the code,
  not the test, unless the test itself is provably incorrect (justify in the PR).
- A required external reference value is unavailable → mark the test `xfail` with a clear reason
  and open a `needs-reference` issue. Never delete a test to make CI pass.

## 9. Cost & safety guardrails for autonomous runs
- Stay within the task scope of the triggering issue. Do not refactor unrelated code.
- Never touch: `LICENSE`, `.github/workflows/release.yml`, anything under `.git/`, or repository
  secrets. Never run `git push --force` to `main`. Never delete branches other than your own.
- Respect branch protection on `main`; all changes go through PRs.
- If a single task would require more than ~15 tool iterations, stop and split it into issues.

## 10. Definition of done for v0.1 (the first shippable milestone)
- `pip install -e ".[dev,all]"` works on Linux/macOS/Windows, Python 3.10–3.12.
- `rb.evaluate(...)` returns a `Report` with ECE, adaptive-ECE, smECE, Brier(+decomp), NLL, AUROC,
  each with a BCa CI.
- `rb.compare(...)` returns a `ComparisonResult` with DeLong (AUROC) or paired-bootstrap (others),
  a p-value, and a `significant` flag.
- `rb.recalibrate(..., method="temperature"|"isotonic")` works and before/after is reportable.
- `report.reliability_diagram()` and `report.to_html()` work behind the `viz`/`report` extras.
- All six test categories pass; coverage ≥ 90%; `ruff` and `mypy --strict` clean.
- README has a 5-line runnable example and the incumbents comparison table.
