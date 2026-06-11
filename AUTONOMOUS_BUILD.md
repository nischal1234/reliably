# Autonomous Build Setup — `reliably` on GitHub

This is the operational guide for letting Claude build the library with minimal human
involvement. It pairs with `CLAUDE.md` (the rules Claude follows) and the TDD (the spec).

Honest scope: this automates ~90% of the *building*. The human gates in §5 are one-time and
unavoidable. Treat this as "an autonomous builder you check on once a day," not "fire and forget."

---

## 1. One-time human setup (you must do these — ~30 minutes)
1. Create a GitHub repo (private is fine to start). Add the three files from the deliverables:
   `CLAUDE.md` and `docs/reliably_technical_design_document.md`, plus this guide.
2. Get an Anthropic API key (console.anthropic.com) and set a monthly spend cap there.
3. Install the Claude GitHub App: easiest is to run `/install-github-app` from inside the
   Claude Code terminal, which installs the app, sets the `ANTHROPIC_API_KEY` repo secret, and
   drops a starter workflow. Manual alternative: install at github.com/apps/claude, then add
   `ANTHROPIC_API_KEY` under repo Settings → Secrets and variables → Actions.
4. Turn on branch protection for `main`: require PRs, require the CI check to pass before merge.
5. Verify the package name `reliably` (fallback `trustcal`) is free on PyPI and GitHub before the
   first commit.

Requirements on any machine where you run Claude Code interactively: Node.js 18+, Git 2.23+, and
the `gh` CLI installed (Claude uses it to open issues/PRs).

---

## 2. Three levels of autonomy (pick how hands-off you want to be)

**Level 1 — Issue-driven (recommended start).** You (or Claude) file one GitHub issue per module
from the TDD build order. Labeling an issue `claude` triggers the action; Claude implements it,
writes tests, and opens a PR. CI runs. If green, it auto-merges. You watch the PR feed.

**Level 2 — Self-seeding backlog.** A scheduled workflow asks Claude each morning to (a) look at
open issues and the TDD, (b) pick the next unbuilt module in dependency order, (c) implement it,
(d) open a PR, and (e) file the next issue. This is the closest practical thing to "beginning to
end" — it advances one module per run until v0.1 is done.

**Level 3 — Auto-fix loop.** On top of Level 2, a second workflow triggers on CI failure and asks
Claude to fix the failing PR in place. Combined, the repo builds and self-heals while you sleep.

Start at Level 1 for a week to calibrate trust and cost, then enable Levels 2–3.

---

## 3. Workflow files

### `.github/workflows/ci.yml`  (the gate everything passes through)
Use the CI from TDD §B.8 (matrix test + ruff + mypy + coverage gate). This is what makes
autonomy safe: bad code can't merge because CI is a required check.

### `.github/workflows/claude.yml`  (Level 1: respond to @claude and labeled issues)
```yaml
name: claude
on:
  issue_comment:
    types: [created]
  issues:
    types: [opened, labeled]
  pull_request_review_comment:
    types: [created]
jobs:
  claude:
    if: contains(github.event.comment.body, '@claude') || github.event.label.name == 'claude'
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      issues: write
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          # honour CLAUDE.md automatically; cap iterations for cost safety
          claude_args: "--max-turns 20"
```

### `.github/workflows/autobuild.yml`  (Level 2: self-seeding daily builder)
```yaml
name: autobuild
on:
  schedule:
    - cron: "0 8 * * 1-5"   # weekday mornings UTC
  workflow_dispatch: {}      # lets you trigger manually too
jobs:
  build-next:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      issues: write
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          claude_args: "--max-turns 30"
          prompt: |
            You are building the `reliably` library. Read CLAUDE.md and
            docs/reliably_technical_design_document.md.
            1. Determine the next unimplemented module in the build order (CLAUDE.md §3) by
               inspecting src/reliably/ and the open issues.
            2. Implement exactly that ONE module plus its tests (all six test categories that apply).
            3. Run ruff, mypy --strict, and pytest locally; fix until green and coverage >= 90%.
            4. Open a pull request describing what you built, which TDD section, and the tests added.
            5. Open a follow-up GitHub issue for the NEXT module so tomorrow's run has a target.
            Do not implement more than one module. Stay within scope. Obey all guardrails in
            CLAUDE.md §9. If the spec is ambiguous, open a `needs-decision` issue and stop.
```

### `.github/workflows/autofix.yml`  (Level 3: self-heal failing PRs)
```yaml
name: autofix
on:
  workflow_run:
    workflows: ["ci"]
    types: [completed]
jobs:
  fix:
    if: github.event.workflow_run.conclusion == 'failure'
    runs-on: ubuntu-latest
    permissions: { contents: write, pull-requests: write }
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0, ref: ${{ github.event.workflow_run.head_branch }} }
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          claude_args: "--max-turns 20"
          prompt: |
            CI failed on this branch. Read the failing job logs, reproduce locally, and fix the
            code (not the tests, unless a test is provably wrong — justify in the commit). Push the
            fix to this same branch. Keep changes minimal and within the PR's original scope.
```

---

## 4. Seeding the backlog (the kickstart)
After the workflows are in, create the first issue and label it `claude`:

> **Title:** Implement `_core` + `stats` foundation
> **Body:** Per CLAUDE.md §3 build order, implement `_core/backend.py`, `_core/results.py`,
> `_core/validation.py`, then `stats/bootstrap.py` (percentile + BCa + jackknife),
> `stats/delong.py`, `stats/tests.py`. Add known-value and property tests. Open a follow-up issue
> for the metrics layer. Label: `claude`.

From there, the `autobuild` schedule keeps it moving. You review/merge (or let auto-merge handle
green PRs) and skim the daily PR each morning.

Optional auto-merge: add a rule (GitHub native auto-merge, or a small step in `ci.yml`) that
merges a PR once all required checks pass and it carries an `auto-merge` label Claude applies.
Keep this OFF until you trust the output.

---

## 5. The human gates that stay manual (and why)
- **Accounts, API key, billing cap** — security; you own the credentials.
- **GitHub App install + branch protection** — one-time trust setup.
- **PyPI name + trusted-publisher config** — account ownership and identity.
- **The first real PyPI release tag** — keep a human approval on `release.yml` so a bad build can't
  ship to the world automatically.
- **The empirical study sign-off** — a human should read the result before it becomes a paper claim.
- **Launch posts (HN, Reddit, X)** — these platforms forbid automated posting and rely on a real
  account's standing. Claude can draft them; you post.

---

## 6. Cost & safety, concretely
- Set the spend cap in the Anthropic console; start low (e.g. enough for ~1 module/day) and raise it.
- `--max-turns` in each workflow bounds runaway loops.
- Branch protection + required CI means nothing untested reaches `main`.
- `CLAUDE.md §9` forbids force-push, secret access, and editing `release.yml`.
- Public-repo note: if you make the repo public, restrict who can trigger the action (the App and
  branch rules handle this) so strangers can't run up your bill via comments.

---

## 7. Daily human checklist (≈5 minutes)
1. Skim the overnight PR: does the diff match the issue, do tests look real (not deleted/weakened)?
2. Merge if green and sensible; otherwise comment `@claude <what to change>` and let it revise.
3. Glance at the API spend.
4. Once a week: read `CHANGELOG.md`, run the examples yourself, and decide if it's time to tag a release.
