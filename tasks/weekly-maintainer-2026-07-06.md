# Weekly Maintainer Digest — 2026-07-06

Automated maintenance pass for **the-llm-council**. Read-only scans + one gated
build. Nothing merged, released, tagged, or posted to public issues. All work is
on `claude/` branches for review.

---

## Executive Summary (3 lines)

- **Built:** Test coverage for untested resilience/provider modules (+114 tests, suite green at 493 passed) on `claude/feature-resilience-provider-tests-2026-07-06`.
- **Planned / deferred:** Version skew + 3 Dependabot security PRs and the Windows CLI fix (#54) — all *merge actions* the maintainer must action manually (this orchestrator never merges).
- **Because:** The two highest-RICE items were merge decisions off-limits to the builder, so value was shipped on the highest *buildable* candidate — the flagged untested modules.

---

## Step 1 — Repo Health

`HEALTH: watch` — main CI is green and at release tag v0.7.18, but there is a
cluster of unactioned maintenance debt.

- **PRs & branches:** 10 open PRs. Two aged unreviewed: **#36** (task adaptive
  protocol, 110d) and **#42** (codex isolated HOME, 60d). PR **#54** (Windows CLI
  arg-length fix) shows CI `action_required`.
- **CI / build:** main green across `test (3.10/3.11/3.12)`, `Build distribution`,
  `Publish to PyPI`. **Version skew:** `pyproject.toml` reads `0.7.17` while the
  published GitHub release is `v0.7.18`; `CLAUDE.md` header also says `0.7.17`.
- **Security (3 open Dependabot advisories):** `idna` (medium, CVE-2024-3651
  bypass), `pytest` (medium, tmpdir handling), `Pygments` (low, ReDoS). Prior
  high/medium alerts (urllib3, requests, pyasn1) already patched in #45.
- **Dependencies:** 5 open Dependabot PRs unmerged — incl. two `actions/*` PRs
  (#24, #25) ~128 days old.
- **Releases:** last release `v0.7.18` (2026-06-10, 26d ago); 0 commits on main
  since — main is exactly at the tag. Cadence healthy (v0.7.4→v0.7.18 in ~2 mo).
- **Drift:** 17 test files / 43 source files. Untested modules flagged:
  `engine/degradation.py`, `storage/summarize.py`, `providers/cli/codex.py`,
  `providers/cli/gemini.py`, `providers/vertex.py`. `mypy --strict` still ~78
  structural errors (known issue).

---

## Step 2 — Feature Candidates (RICE-ranked)

Scale: R/I/C 1–5, Effort 1–5 (higher = costlier). Final = (R × I × C) / E.

| Rank | Feature | R | I | C | E | Final | Flags |
|------|---------|---|---|---|---|-------|-------|
| 1 | Merge Dependabot security PRs + bump `pyproject` to 0.7.18 | 5 | 3 | 5 | 1 | **75.0** | — *(merge action)* |
| 2 | Windows CLI stdin fix — merge PR #54 | 3 | 5 | 5 | 1 | **75.0** | — *(merge action)* |
| 3 | **Test coverage for `degradation.py` + `vertex.py` + `cli/*.py`** | 3 | 4 | 5 | 2 | **30.0** | — ✅ built |
| 4 | Anthropic prompt-cache (`cache_control`) header support | 3 | 4 | 4 | 2 | 24.0 | — |
| 5 | Deterministic prompt-level caching (local disk) | 4 | 4 | 3 | 3 | 16.0 | new-deps |
| 6 | Adaptive Cost/Quality Routing | 4 | 5 | 3 | 4 | 15.0 | multi-run |
| 7 | Local observability / structured trace store | 3 | 4 | 2 | 4 | 6.0 | multi-run, speculative |
| 8 | `mypy --strict` structural clean-up (~78 errors) | 2 | 2 | 4 | 3 | 5.3 | — |

`TOP: Merge Dependabot security PRs + bump pyproject to 0.7.18 (Final: 75.0)`

---

## Step 3 — Value Gate

The two highest-Final items (#1, #2) are **merge actions**, not buildable
features: closing Dependabot PRs and merging an already-authored PR (#54). Hard
safety forbids this orchestrator from merging, and rebuilding existing PRs would
duplicate work. They are handed back to the maintainer as manual actions.

Gate applied to the highest **buildable** candidate, **#3** (Final 30.0):
Final ≥ 8 ✅ · unflagged ✅ · self-contained single-run change with tests ✅ →
**gate passed → build.**

---

## Step 4 — What Was Built

**Branch:** `claude/feature-resilience-provider-tests-2026-07-06` (pushed to
origin, commit `c1bb3ed`; no PR). Source untouched — tests only.

| New test file | Tests | Focus |
|---|---|---|
| `tests/test_degradation.py` | 34 | Every branch of `DegradationPolicy._determine_action`: non-retryable skip/abort/fallback + billing-URL surfacing, exponential backoff (1000→2000→4000→8000→cap 10000ms), per provider+phase retry bookkeeping, model-unavailable fallback/skip, max-retries paths, `DegradationReport`/`FailureEvent` serialization (200-char truncation), `create_default_policy` defaults. |
| `tests/test_cli_provider_parsing.py` | 53 | Codex + Gemini pure helpers: JSONL usage/message/error extraction, `_prepare_schema_for_codex` strict-schema closure, `_LiveCodexState` ingestion, `_build_command`, `_resolve_model`, Gemini `_extract_text_payload`/`_normalize_usage`/`_format_gemini_error`/`_extract_json_payload`. |
| `tests/test_summarize_tiers.py` | 27 | `TieredSummary.get_tier` (all 5 tiers), `_extract_gist/findings/actions/rationale` heuristics, AUDIT-tier artifact round-trip, `SummarizationResult.to_dict`, `get_total_tokens_saved`. |

- **Suite:** `pytest` → **493 passed** (baseline 379; **+114**). `ruff check` and
  `ruff format --check` clean on all new files.
- **Deliberately skipped:** `providers/vertex.py` — full `generate()` coverage
  needs heavy Google/Anthropic Vertex SDK mocking and it already has partial
  coverage in `test_providers.py`. Judged lower-value than depth on the other
  three; left for a dedicated pass. (Sanity floor respected — no padded tests.)

---

## Step 5 — Stale `claude/` Branch Cleanup List

Today is 2026-07-06. 7 branches past the 30-day threshold (work merged to main),
recommended for deletion by the maintainer:

| Branch | Age | Status |
|---|---|---|
| `claude/feature-markdown-output-2026-06-01` | 35d | stale — merged |
| `claude/weekly-maintainer-2026-06-01` | 35d | stale — merged |
| `claude/fix-ci-ruff-format-2026-06-01` | 35d | stale — merged |
| `claude/security-deps-2026-06-01` | 35d | stale — merged |
| `claude/gate-publish-on-ci-2026-06-01` | 35d | stale — merged |
| `claude/weekly-maintainer-2026-06-08` | 28d | stale — merged |
| `claude/weekly-maintainer-2026-06-15` | 21d | stale |
| `claude/feature-engine-tests-2026-06-22` | 14d | borderline |
| `claude/weekly-maintainer-2026-06-22` | 14d | borderline |
| `claude/feature-registry-tests-2026-06-29` | 7d | recent — keep |
| `claude/weekly-maintainer-2026-06-29` | 7d | recent — keep |

*(This orchestrator does not delete branches; cleanup is a maintainer action.)*

---

## Maintainer Action Items (not done by this pass — by design)

1. **Merge the 3 Dependabot security PRs** (idna, pytest, pygments) — closes all
   open advisories. Then merge/close the two 128-day-old `actions/*` PRs.
2. **Fix version skew:** bump `pyproject.toml` (and `CLAUDE.md` header) `0.7.17`
   → `0.7.18` to match the published release.
3. **Review PR #54** (Windows CLI fix — currently `action_required`).
4. **Triage aged PRs #36 (110d) and #42 (60d).**
5. **Review & merge** `claude/feature-resilience-provider-tests-2026-07-06`.
6. **Delete** the 7 stale `claude/` branches above.

---

## Notes

- **Slack:** no target channel is configured in the task (`#<channel>`
  placeholder unfilled), so the 3-line summary was **not** posted. Fill in a
  channel to enable this.
- **Threshold recalibration:** the RICE `Final` scale here is raw `(R×I×C)/E`
  (max 125), not the 1–10 the default threshold of 8 assumed. The gate held
  because the decisive constraint was *buildability under safety rules*, not the
  numeric cutoff. Recommend the scout normalize `Final` to 0–10 (or the gate
  threshold be raised to ~20 on the raw scale) so the cutoff is meaningful.
