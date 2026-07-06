"""Tests for tiered summarization internals not covered by test_storage.py.

Covers :class:`TieredSummary` tier rendering, the heuristic extraction
methods, serialization helpers, AUDIT-tier artifact wiring, and
``get_total_tokens_saved``.
"""

from __future__ import annotations

from llm_council.protocol.types import SummaryTier
from llm_council.storage import ArtifactStore, ArtifactType
from llm_council.storage.summarize import (
    TIER_CHAR_LIMITS,
    TIER_TOKEN_LIMITS,
    SummarizationResult,
    Summarizer,
    TieredSummary,
)


def _store(tmp_path) -> ArtifactStore:
    return ArtifactStore(
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "ledger.db",
    )


class TestTieredSummary:
    """Rendering a pre-built multi-tier summary per requested tier."""

    def _sample(self) -> TieredSummary:
        return TieredSummary(
            gist="One line gist.",
            findings=["finding a", "finding b"],
            actions=["do x", "do y"],
            rationale="because reasons",
            artifact_ref="artifact-123",
        )

    def test_gist_tier(self):
        assert self._sample().get_tier(SummaryTier.GIST) == "One line gist."

    def test_findings_tier(self):
        rendered = self._sample().get_tier(SummaryTier.FINDINGS)
        assert "Key findings:" in rendered
        assert "- finding a" in rendered
        assert "- finding b" in rendered
        assert "do x" not in rendered

    def test_actions_tier_includes_findings_and_actions(self):
        rendered = self._sample().get_tier(SummaryTier.ACTIONS)
        assert "Key findings:" in rendered
        assert "Actions:" in rendered
        assert "- do x" in rendered
        assert "because reasons" not in rendered

    def test_rationale_tier_includes_everything_but_ref(self):
        rendered = self._sample().get_tier(SummaryTier.RATIONALE)
        assert "Rationale:" in rendered
        assert "because reasons" in rendered
        assert "artifact-123" not in rendered

    def test_audit_tier_appends_artifact_ref(self):
        rendered = self._sample().get_tier(SummaryTier.AUDIT)
        assert "because reasons" in rendered
        assert "[Full details: artifact artifact-123]" in rendered

    def test_audit_tier_without_ref_has_no_note(self):
        summary = self._sample()
        summary.artifact_ref = None
        rendered = summary.get_tier(SummaryTier.AUDIT)
        assert "Full details" not in rendered

    def test_to_dict(self):
        data = self._sample().to_dict()
        assert data["gist"] == "One line gist."
        assert data["findings"] == ["finding a", "finding b"]
        assert data["actions"] == ["do x", "do y"]
        assert data["rationale"] == "because reasons"
        assert data["artifact_ref"] == "artifact-123"


class TestSummarizationResultSerialization:
    """`SummarizationResult.to_dict` shape."""

    def test_to_dict(self):
        result = SummarizationResult(
            tier=SummaryTier.ACTIONS,
            summary="s",
            token_estimate=10,
            original_tokens=100,
            tokens_saved=90,
            artifact_ref="ref-1",
            truncated=True,
        )
        data = result.to_dict()
        assert data["tier"] == "actions"
        assert data["summary"] == "s"
        assert data["token_estimate"] == 10
        assert data["original_tokens"] == 100
        assert data["tokens_saved"] == 90
        assert data["artifact_ref"] == "ref-1"
        assert data["truncated"] is True


class TestGistExtraction:
    """`_extract_gist` heuristics."""

    def test_finds_summary_line(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        content = "Intro paragraph.\nSummary: the system works well.\nMore detail."
        gist = summarizer._extract_gist(content, char_limit=200)
        assert gist == "the system works well"

    def test_finds_conclusion_line(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        content = "Body text here.\nConclusion: ship it now.\n"
        gist = summarizer._extract_gist(content, char_limit=200)
        assert gist == "ship it now"

    def test_falls_back_to_first_meaningful_line(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        content = "short\nThis is a meaningful first line with content."
        gist = summarizer._extract_gist(content, char_limit=200)
        assert gist == "This is a meaningful first line with content."

    def test_truncates_long_first_line(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        content = "X" * 500
        gist = summarizer._extract_gist(content, char_limit=50)
        assert gist.endswith("...")
        assert len(gist) == 50


class TestFindingsExtraction:
    """`_extract_findings` pulls bullets and numbered items."""

    def test_extracts_bullets(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        content = "- first important finding here\n- second important finding here\n"
        result = summarizer._extract_findings(content, char_limit=500)
        assert "Key findings:" in result
        assert "first important finding here" in result

    def test_extracts_numbered_items(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        content = "1. do the first thing carefully\n2. then do the second thing\n"
        result = summarizer._extract_findings(content, char_limit=500)
        assert "do the first thing carefully" in result

    def test_skips_trivial_short_items(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        content = "- ok\n- this one is long enough to keep in findings\n"
        result = summarizer._extract_findings(content, char_limit=500)
        assert "this one is long enough" in result
        # The trivial "ok" bullet is dropped.
        assert "- ok\n" not in result

    def test_falls_back_to_first_paragraph(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        content = "First paragraph with no bullets.\n\nSecond paragraph."
        result = summarizer._extract_findings(content, char_limit=500)
        assert result.startswith("First paragraph")


class TestActionsExtraction:
    """`_extract_actions` appends recommendation-style items."""

    def test_extracts_should_recommendations(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        content = (
            "- a solid finding worth keeping here.\nYou should refactor the retry loop soon.\n"
        )
        result = summarizer._extract_actions(content, char_limit=1000)
        assert "Actions:" in result
        assert "refactor the retry loop" in result

    def test_no_actions_returns_findings_only(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        content = "- a plain finding with enough length here.\n"
        result = summarizer._extract_actions(content, char_limit=1000)
        assert "Actions:" not in result
        assert "plain finding" in result


class TestRationaleExtraction:
    """`_extract_rationale` appends reasoning items."""

    def test_extracts_because_clause(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        content = (
            "- a finding worth keeping in the output.\n"
            "This matters because the retry budget is exhausted quickly.\n"
        )
        result = summarizer._extract_rationale(content, char_limit=2000)
        assert "Rationale:" in result
        assert "retry budget is exhausted quickly" in result

    def test_no_reasons_returns_actions_only(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        content = "- a finding worth keeping in the output here.\n"
        result = summarizer._extract_rationale(content, char_limit=2000)
        assert "Rationale:" not in result


class TestGenerateSummaryDispatch:
    """`_generate_summary` routes to the correct extractor per tier."""

    def test_audit_returns_full_content_within_limit(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        content = "short audit body"
        result = summarizer._generate_summary(content, SummaryTier.AUDIT, char_limit=1000)
        assert result == content

    def test_audit_truncates_over_limit(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        content = "A" * 100
        result = summarizer._generate_summary(content, SummaryTier.AUDIT, char_limit=40)
        assert len(result) == 40


class TestAuditTierArtifactWiring:
    """AUDIT-tier summaries store full content and append the artifact ref."""

    def test_audit_summary_stores_and_references_artifact(self, tmp_path):
        store = _store(tmp_path)
        summarizer = Summarizer(artifact_store=store, threshold_tokens=10)
        run = store.create_run(subagent="test", task="audit")

        # AUDIT tier only summarizes above ~10k tokens (>40k chars).
        content = "This is a fairly long body of content.\n" * 1200
        result = summarizer.summarize(
            content,
            SummaryTier.AUDIT,
            run_id=run.run_id,
            artifact_type=ArtifactType.SYNTHESIS,
            store_full=True,
        )

        assert result.artifact_ref is not None
        assert f"[Full details: artifact {result.artifact_ref}]" in result.summary
        # The stored artifact round-trips to the original content.
        assert store.get_artifact_content(result.artifact_ref) == content

    def test_summarize_without_run_id_skips_storage(self, tmp_path):
        store = _store(tmp_path)
        summarizer = Summarizer(artifact_store=store, threshold_tokens=10)

        content = "Large body of content that exceeds the tier limit.\n" * 50
        result = summarizer.summarize(content, SummaryTier.GIST, store_full=True)

        assert result.artifact_ref is None
        assert result.tokens_saved > 0


class TestGetTotalTokensSaved:
    """Aggregation helper across summarization results."""

    def test_sums_tokens_saved(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        results = {
            "p1": SummarizationResult(
                tier=SummaryTier.GIST,
                summary="s",
                token_estimate=5,
                original_tokens=100,
                tokens_saved=95,
            ),
            "p2": SummarizationResult(
                tier=SummaryTier.GIST,
                summary="s",
                token_estimate=5,
                original_tokens=50,
                tokens_saved=45,
            ),
        }
        assert summarizer.get_total_tokens_saved(results) == 140

    def test_empty_results(self, tmp_path):
        summarizer = Summarizer(artifact_store=_store(tmp_path))
        assert summarizer.get_total_tokens_saved({}) == 0


class TestTierLimits:
    """Char limits are derived from token limits (4 chars per token)."""

    def test_char_limits_are_four_times_token_limits(self):
        for tier, token_limit in TIER_TOKEN_LIMITS.items():
            assert TIER_CHAR_LIMITS[tier] == token_limit * 4
