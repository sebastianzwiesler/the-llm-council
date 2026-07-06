"""Focused tests for the graceful-degradation policy engine.

Complements the higher-level coverage in ``tests/test_engine.py`` by
exercising each branch of :class:`DegradationPolicy._determine_action`,
the exponential-backoff schedule, retry bookkeeping, and the report/event
serialization helpers.
"""

from __future__ import annotations

from llm_council.engine.degradation import (
    DegradationAction,
    DegradationDecision,
    DegradationPolicy,
    DegradationReport,
    FailureEvent,
    create_default_policy,
)
from llm_council.providers.base import ErrorType


class TestNonRetryableErrors:
    """Non-retryable errors (billing, auth, cli_not_found) must fail fast."""

    def test_auth_error_skips_when_providers_remain(self):
        policy = create_default_policy()

        decision = policy.decide(
            provider="openai",
            error="invalid_api_key",
            phase="drafts",
            remaining_providers=2,
        )

        assert decision.action == DegradationAction.SKIP
        assert "Authentication error" in decision.reason
        assert decision.billing_url is None

    def test_billing_error_includes_billing_url(self):
        policy = create_default_policy()

        decision = policy.decide(
            provider="openai",
            error="insufficient_quota",
            phase="drafts",
            remaining_providers=1,
        )

        assert decision.action == DegradationAction.SKIP
        assert decision.billing_url == "https://platform.openai.com/account/billing"
        assert "check" in decision.reason

    def test_non_retryable_prefers_fallback_over_skip(self):
        policy = DegradationPolicy(fallback_providers={"openai": "anthropic"})

        decision = policy.decide(
            provider="openai",
            error="insufficient_quota",
            phase="drafts",
            remaining_providers=2,
        )

        assert decision.action == DegradationAction.FALLBACK
        assert decision.fallback_provider == "anthropic"
        # Billing URL is still surfaced even when falling back.
        assert decision.billing_url == "https://platform.openai.com/account/billing"

    def test_non_retryable_aborts_in_critical_phase_with_no_providers(self):
        policy = create_default_policy()

        decision = policy.decide(
            provider="last",
            error="unauthorized",
            phase="synthesis",
            remaining_providers=0,
        )

        assert decision.action == DegradationAction.ABORT
        assert "Critical failure in synthesis" in decision.reason

    def test_non_retryable_skips_in_drafts_phase_with_no_providers(self):
        """The abort-on-critical rule only applies to critique/synthesis."""
        policy = create_default_policy()

        decision = policy.decide(
            provider="only",
            error="invalid_api_key",
            phase="drafts",
            remaining_providers=0,
        )

        assert decision.action == DegradationAction.SKIP


class TestRetryableErrors:
    """Rate-limit / timeout / network errors retry with backoff."""

    def test_rate_limit_retries_with_backoff(self):
        policy = create_default_policy(max_retries=3)

        decision = policy.decide("p", "429 too many requests", "drafts", 1)

        assert decision.action == DegradationAction.RETRY
        assert decision.retry_delay_ms == 1000  # base delay on first attempt

    def test_timeout_error_retries(self):
        policy = create_default_policy(max_retries=2)

        decision = policy.decide("p", "request timed out", "drafts", 1)

        assert decision.action == DegradationAction.RETRY

    def test_network_error_retries(self):
        policy = create_default_policy(max_retries=2)

        decision = policy.decide("p", "connection reset by peer", "drafts", 1)

        assert decision.action == DegradationAction.RETRY

    def test_exponential_backoff_schedule(self):
        """Delay should double per retry and cap at MAX_RETRY_DELAY_MS."""
        policy = create_default_policy(max_retries=10)

        delays = []
        for _ in range(5):
            decision = policy.decide("p", "rate limit", "drafts", 1)
            assert decision.action == DegradationAction.RETRY
            delays.append(decision.retry_delay_ms)

        # 1000, 2000, 4000, 8000, then capped at 10000
        assert delays == [1000, 2000, 4000, 8000, 10000]

    def test_retry_count_tracked_per_provider_and_phase(self):
        policy = create_default_policy(max_retries=2)

        # Retries for the same provider+phase accumulate independently
        policy.decide("p", "rate limit", "drafts", 1)
        d2 = policy.decide("p", "rate limit", "drafts", 1)
        assert d2.retry_delay_ms == 2000

        # A different phase starts fresh at the base delay.
        d_other = policy.decide("p", "rate limit", "critique", 1)
        assert d_other.retry_delay_ms == 1000

    def test_retryable_error_at_retry_limit_falls_through_to_skip(self):
        policy = create_default_policy(max_retries=1)

        first = policy.decide("p", "rate limit", "drafts", 2)
        assert first.action == DegradationAction.RETRY

        second = policy.decide("p", "rate limit", "drafts", 2)
        # Retries exhausted, providers still remain -> skip.
        assert second.action == DegradationAction.SKIP


class TestModelUnavailable:
    """MODEL_UNAVAILABLE is not retried; it falls back or skips."""

    def test_model_unavailable_uses_fallback(self):
        policy = DegradationPolicy(fallback_providers={"p": "backup"})

        decision = policy.decide("p", "model not found", "drafts", 1)

        assert decision.action == DegradationAction.FALLBACK
        assert decision.fallback_provider == "backup"
        assert "Model unavailable" in decision.reason

    def test_model_unavailable_skips_without_fallback(self):
        policy = create_default_policy()

        decision = policy.decide("p", "model not found", "drafts", 1)

        assert decision.action == DegradationAction.SKIP
        assert "no fallback" in decision.reason


class TestMaxRetriesExceeded:
    """Behavior once a provider has burned through its retry budget."""

    def _exhaust(self, policy: DegradationPolicy, phase: str, remaining: int) -> None:
        for _ in range(policy._max_retries):
            policy.decide("p", "rate limit", phase, remaining)

    def test_fallback_used_after_exhausting_retries(self):
        policy = DegradationPolicy(fallback_providers={"p": "backup"}, max_retries=1)

        policy.decide("p", "rate limit", "drafts", 2)  # first retry
        decision = policy.decide("p", "rate limit", "drafts", 2)

        assert decision.action == DegradationAction.FALLBACK
        assert decision.fallback_provider == "backup"

    def test_abort_when_exhausted_in_critical_phase(self):
        policy = create_default_policy(max_retries=1)

        policy.decide("p", "rate limit", "synthesis", 0)  # first retry
        decision = policy.decide("p", "rate limit", "synthesis", 0)

        assert decision.action == DegradationAction.ABORT
        assert "exhausted in synthesis" in decision.reason

    def test_abort_when_exhausted_in_drafts_and_abort_on_all(self):
        policy = DegradationPolicy(max_retries=1, abort_on_all_failures=True)

        policy.decide("p", "rate limit", "drafts", 0)  # first retry
        decision = policy.decide("p", "rate limit", "drafts", 0)

        assert decision.action == DegradationAction.ABORT
        assert "All providers exhausted" in decision.reason

    def test_skip_when_exhausted_but_providers_remain(self):
        policy = create_default_policy(max_retries=1)

        policy.decide("p", "rate limit", "drafts", 2)  # first retry
        decision = policy.decide("p", "rate limit", "drafts", 2)

        assert decision.action == DegradationAction.SKIP


class TestUnknownAndDefaultBranches:
    """Unknown errors get one retry, then continue/abort based on capacity."""

    def test_unknown_error_retries_once(self):
        policy = create_default_policy(max_retries=2)

        decision = policy.decide("p", "some totally opaque failure", "drafts", 2)

        assert decision.action == DegradationAction.RETRY
        assert decision.retry_delay_ms == DegradationPolicy.BASE_RETRY_DELAY_MS
        assert "Unknown error" in decision.reason

    def test_unknown_error_continues_after_single_retry(self):
        policy = create_default_policy(max_retries=2)

        policy.decide("p", "opaque failure", "drafts", 2)  # the one allowed retry
        decision = policy.decide("p", "opaque failure", "drafts", 2)

        assert decision.action == DegradationAction.CONTINUE
        assert "remaining provider" in decision.reason

    def test_unknown_error_aborts_when_below_minimum(self):
        policy = DegradationPolicy(
            max_retries=2,
            min_providers_required=1,
            abort_on_all_failures=True,
        )

        policy.decide("p", "opaque failure", "drafts", 0)  # the one allowed retry
        decision = policy.decide("p", "opaque failure", "drafts", 0)

        assert decision.action == DegradationAction.ABORT
        assert "Below minimum" in decision.reason

    def test_unknown_error_continues_when_abort_disabled(self):
        policy = DegradationPolicy(
            max_retries=2,
            min_providers_required=1,
            abort_on_all_failures=False,
        )

        policy.decide("p", "opaque failure", "drafts", 0)  # the one allowed retry
        decision = policy.decide("p", "opaque failure", "drafts", 0)

        assert decision.action == DegradationAction.CONTINUE
        assert "degraded capacity" in decision.reason

    def test_exception_input_is_stringified_and_classified(self):
        policy = create_default_policy()

        decision = policy.decide(
            provider="p",
            error=RuntimeError("rate limit exceeded"),
            phase="drafts",
            remaining_providers=1,
        )

        assert decision.action == DegradationAction.RETRY


class TestReportBookkeeping:
    """The policy records structured failure events as it decides."""

    def test_report_accumulates_and_categorizes(self):
        policy = create_default_policy(max_retries=2)

        policy.decide("p1", "insufficient_quota", "drafts", 2)  # skip
        policy.decide("p2", "rate limit", "drafts", 2)  # retry
        report = policy.get_report()

        assert len(report.failures) == 2
        assert "p1" in report.providers_skipped
        assert report.total_retries == 1

    def test_reset_clears_state_and_retry_counts(self):
        policy = create_default_policy(max_retries=2)

        policy.decide("p", "rate limit", "drafts", 1)
        policy.reset()

        # After reset the retry counter restarts from the base delay.
        decision = policy.decide("p", "rate limit", "drafts", 1)
        assert decision.retry_delay_ms == 1000
        assert len(policy.get_report().failures) == 1


class TestDegradationReport:
    """Direct unit tests for the report/event dataclasses."""

    def test_add_failure_updates_skips(self):
        report = DegradationReport()
        report.add_failure(
            FailureEvent(
                provider="p",
                phase="drafts",
                error_type=ErrorType.AUTH,
                error_message="bad key",
                action_taken=DegradationAction.SKIP,
            )
        )
        report.add_failure(
            FailureEvent(
                provider="p",  # duplicate provider should not double-count
                phase="critique",
                error_type=ErrorType.AUTH,
                error_message="bad key",
                action_taken=DegradationAction.SKIP,
            )
        )
        assert report.providers_skipped == ["p"]

    def test_add_failure_tracks_fallbacks_uniquely(self):
        report = DegradationReport()
        for _ in range(2):
            report.add_failure(
                FailureEvent(
                    provider="p",
                    phase="drafts",
                    error_type=ErrorType.MODEL_UNAVAILABLE,
                    error_message="gone",
                    action_taken=DegradationAction.FALLBACK,
                    fallback_provider="backup",
                )
            )
        assert report.fallbacks_used == ["backup"]

    def test_add_failure_sets_aborted(self):
        report = DegradationReport()
        report.add_failure(
            FailureEvent(
                provider="p",
                phase="synthesis",
                error_type=ErrorType.AUTH,
                error_message="x",
                action_taken=DegradationAction.ABORT,
            )
        )
        assert report.aborted is True

    def test_to_summary_empty(self):
        assert DegradationReport().to_summary() == "No degradation events"

    def test_to_summary_includes_all_sections(self):
        report = DegradationReport()
        report.add_failure(
            FailureEvent(
                provider="a",
                phase="drafts",
                error_type=ErrorType.AUTH,
                error_message="x",
                action_taken=DegradationAction.SKIP,
            )
        )
        report.add_failure(
            FailureEvent(
                provider="b",
                phase="drafts",
                error_type=ErrorType.MODEL_UNAVAILABLE,
                error_message="x",
                action_taken=DegradationAction.FALLBACK,
                fallback_provider="c",
            )
        )
        report.add_failure(
            FailureEvent(
                provider="d",
                phase="drafts",
                error_type=ErrorType.RATE_LIMIT,
                error_message="x",
                action_taken=DegradationAction.RETRY,
            )
        )
        report.add_failure(
            FailureEvent(
                provider="e",
                phase="synthesis",
                error_type=ErrorType.AUTH,
                error_message="x",
                action_taken=DegradationAction.ABORT,
            )
        )

        summary = report.to_summary()
        assert "4 failure(s)" in summary
        assert "Skipped: a" in summary
        assert "Fallbacks: c" in summary
        assert "Retries: 1" in summary
        assert "ABORTED" in summary

    def test_report_to_dict_roundtrip(self):
        report = DegradationReport()
        report.add_failure(
            FailureEvent(
                provider="p",
                phase="drafts",
                error_type=ErrorType.RATE_LIMIT,
                error_message="x" * 500,  # long message should be truncated in event dict
                action_taken=DegradationAction.RETRY,
            )
        )
        data = report.to_dict()

        assert data["total_retries"] == 1
        assert data["aborted"] is False
        assert len(data["failures"]) == 1
        assert len(data["failures"][0]["error_message"]) == 200


class TestFailureEvent:
    """Event serialization details."""

    def test_to_dict_truncates_message_and_serializes_enum(self):
        event = FailureEvent(
            provider="p",
            phase="drafts",
            error_type=ErrorType.BILLING,
            error_message="y" * 300,
            action_taken=DegradationAction.SKIP,
            retry_count=2,
            fallback_provider="backup",
        )
        data = event.to_dict()

        assert data["error_type"] == "billing"
        assert data["action_taken"] == "skip"
        assert data["retry_count"] == 2
        assert data["fallback_provider"] == "backup"
        assert len(data["error_message"]) == 200
        assert data["timestamp"]  # populated by default factory


class TestCreateDefaultPolicy:
    """Factory produces the documented defaults."""

    def test_defaults(self):
        policy = create_default_policy()
        assert policy._max_retries == 2
        assert policy._min_providers == 1
        assert policy._abort_on_all_failures is True
        assert policy._fallbacks == {}

    def test_overrides_are_applied(self):
        policy = create_default_policy(
            max_retries=5,
            fallback_providers={"a": "b"},
        )
        assert policy._max_retries == 5
        assert policy._fallbacks == {"a": "b"}


class TestDegradationDecisionDefaults:
    """Sanity checks on the decision dataclass defaults."""

    def test_decision_defaults(self):
        decision = DegradationDecision(
            action=DegradationAction.CONTINUE,
            reason="ok",
        )
        assert decision.retry_delay_ms == 0
        assert decision.fallback_provider is None
        assert decision.billing_url is None
        assert decision.should_log is True
