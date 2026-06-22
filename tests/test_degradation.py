"""Dedicated unit tests for ``llm_council.engine.degradation``.

Focuses on branch coverage of the degradation policy decision tree and the
report/event dataclasses. Complements (does not duplicate) the high-level
smoke tests in ``test_engine.py`` by exercising the less-trodden branches:
MODEL_UNAVAILABLE handling, the UNKNOWN-error single-retry path, CONTINUE
decisions, abort-on-minimum behavior, exponential backoff growth/cap, and
the ``FailureEvent``/``DegradationReport`` accumulation logic.
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


# ---------------------------------------------------------------------------
# FailureEvent
# ---------------------------------------------------------------------------
class TestFailureEvent:
    def test_defaults(self):
        event = FailureEvent(
            provider="p1",
            phase="drafts",
            error_type=ErrorType.RATE_LIMIT,
            error_message="rate limited",
            action_taken=DegradationAction.RETRY,
        )
        assert event.retry_count == 0
        assert event.fallback_provider is None
        # timestamp default factory produces a non-empty ISO string
        assert isinstance(event.timestamp, str)
        assert event.timestamp

    def test_to_dict_truncates_message(self):
        long_message = "x" * 500
        event = FailureEvent(
            provider="p1",
            phase="critique",
            error_type=ErrorType.AUTH,
            error_message=long_message,
            action_taken=DegradationAction.SKIP,
            retry_count=2,
            fallback_provider="backup",
        )
        data = event.to_dict()

        assert data["provider"] == "p1"
        assert data["phase"] == "critique"
        assert data["error_type"] == "auth"
        assert data["action_taken"] == "skip"
        assert data["retry_count"] == 2
        assert data["fallback_provider"] == "backup"
        # error message is truncated to 200 chars
        assert len(data["error_message"]) == 200


# ---------------------------------------------------------------------------
# DegradationReport
# ---------------------------------------------------------------------------
class TestDegradationReport:
    def _event(self, action: DegradationAction, **kwargs) -> FailureEvent:
        return FailureEvent(
            provider=kwargs.pop("provider", "p1"),
            phase=kwargs.pop("phase", "drafts"),
            error_type=kwargs.pop("error_type", ErrorType.UNKNOWN),
            error_message=kwargs.pop("error_message", "boom"),
            action_taken=action,
            **kwargs,
        )

    def test_empty_report_summary(self):
        report = DegradationReport()
        assert report.to_summary() == "No degradation events"

    def test_retry_increments_total(self):
        report = DegradationReport()
        report.add_failure(self._event(DegradationAction.RETRY))
        report.add_failure(self._event(DegradationAction.RETRY))
        assert report.total_retries == 2

    def test_skip_dedupes_provider(self):
        report = DegradationReport()
        report.add_failure(self._event(DegradationAction.SKIP, provider="dup"))
        report.add_failure(self._event(DegradationAction.SKIP, provider="dup"))
        assert report.providers_skipped == ["dup"]

    def test_fallback_dedupes_and_requires_name(self):
        report = DegradationReport()
        # Fallback without a fallback_provider should not be recorded.
        report.add_failure(self._event(DegradationAction.FALLBACK, fallback_provider=None))
        assert report.fallbacks_used == []

        report.add_failure(self._event(DegradationAction.FALLBACK, fallback_provider="backup"))
        report.add_failure(self._event(DegradationAction.FALLBACK, fallback_provider="backup"))
        assert report.fallbacks_used == ["backup"]

    def test_abort_sets_flag(self):
        report = DegradationReport()
        report.add_failure(self._event(DegradationAction.ABORT))
        assert report.aborted is True

    def test_continue_is_recorded_but_neutral(self):
        report = DegradationReport()
        report.add_failure(self._event(DegradationAction.CONTINUE))
        assert len(report.failures) == 1
        assert report.total_retries == 0
        assert report.providers_skipped == []
        assert report.fallbacks_used == []
        assert report.aborted is False

    def test_summary_includes_all_sections(self):
        report = DegradationReport()
        report.add_failure(self._event(DegradationAction.SKIP, provider="skipme"))
        report.add_failure(self._event(DegradationAction.FALLBACK, fallback_provider="fb"))
        report.add_failure(self._event(DegradationAction.RETRY))
        report.add_failure(self._event(DegradationAction.ABORT))

        summary = report.to_summary()
        assert "4 failure(s)" in summary
        assert "Skipped: skipme" in summary
        assert "Fallbacks: fb" in summary
        assert "Retries: 1" in summary
        assert "ABORTED" in summary

    def test_to_dict_round_trips_fields(self):
        report = DegradationReport()
        report.add_failure(self._event(DegradationAction.SKIP, provider="s1"))
        report.add_failure(self._event(DegradationAction.RETRY))

        data = report.to_dict()
        assert data["total_retries"] == 1
        assert data["providers_skipped"] == ["s1"]
        assert data["aborted"] is False
        assert len(data["failures"]) == 2
        assert all(isinstance(f, dict) for f in data["failures"])


# ---------------------------------------------------------------------------
# DegradationPolicy decision tree
# ---------------------------------------------------------------------------
class TestDegradationPolicyDecisions:
    def test_billing_error_with_fallback(self):
        policy = DegradationPolicy(fallback_providers={"primary": "backup"})
        decision = policy.decide("primary", "insufficient_quota", "drafts", 1)
        assert decision.action == DegradationAction.FALLBACK
        assert decision.fallback_provider == "backup"
        assert decision.billing_url is not None

    def test_billing_error_without_fallback_skips(self):
        policy = create_default_policy()
        decision = policy.decide("openai", "billing payment required", "drafts", 2)
        assert decision.action == DegradationAction.SKIP
        assert decision.billing_url == "https://platform.openai.com/account/billing"

    def test_auth_error_aborts_when_no_remaining_in_critical_phase(self):
        policy = create_default_policy()
        decision = policy.decide("solo", "unauthorized: invalid_api_key", "synthesis", 0)
        assert decision.action == DegradationAction.ABORT

    def test_auth_error_skips_in_noncritical_phase(self):
        policy = create_default_policy()
        decision = policy.decide("solo", "unauthorized: invalid_api_key", "drafts", 0)
        assert decision.action == DegradationAction.SKIP

    def test_model_unavailable_with_fallback(self):
        policy = DegradationPolicy(fallback_providers={"primary": "backup"})
        decision = policy.decide("primary", "model not found", "drafts", 1)
        assert decision.action == DegradationAction.FALLBACK
        assert decision.fallback_provider == "backup"

    def test_model_unavailable_without_fallback_skips(self):
        policy = create_default_policy()
        decision = policy.decide("primary", "model not found", "drafts", 1)
        assert decision.action == DegradationAction.SKIP

    def test_unknown_error_retries_once_then_continues(self):
        policy = create_default_policy(max_retries=2)
        # First unknown error gets a single retry.
        first = policy.decide("p", "some totally novel failure", "drafts", 2)
        assert first.action == DegradationAction.RETRY
        assert first.retry_delay_ms == DegradationPolicy.BASE_RETRY_DELAY_MS

        # Second time around, the unknown one-retry budget is spent so it
        # falls through to CONTINUE (remaining providers >= min required).
        second = policy.decide("p", "some totally novel failure", "drafts", 2)
        assert second.action == DegradationAction.CONTINUE

    def test_continue_when_enough_remaining(self):
        # An UNKNOWN error spends its single retry, then on the next call falls
        # through to CONTINUE because remaining providers >= min required.
        policy = create_default_policy(max_retries=2)
        novel = "some totally novel failure"
        assert policy.decide("p", novel, "drafts", 3).action == DegradationAction.RETRY
        decision = policy.decide("p", novel, "drafts", 3)
        assert decision.action == DegradationAction.CONTINUE
        assert "remaining" in decision.reason

    def test_abort_when_below_minimum_and_abort_enabled(self):
        # UNKNOWN error, retry budget spent, no remaining providers, and the
        # minimum-required threshold is unmet -> ABORT.
        policy = DegradationPolicy(
            max_retries=2,
            min_providers_required=2,
            abort_on_all_failures=True,
        )
        novel = "some totally novel failure"
        policy.decide("p", novel, "drafts", 0)  # consumes the single UNKNOWN retry
        decision = policy.decide("p", novel, "drafts", 0)
        assert decision.action == DegradationAction.ABORT

    def test_continue_degraded_when_abort_disabled(self):
        # Same as above but abort_on_all_failures=False -> CONTINUE degraded.
        policy = DegradationPolicy(
            max_retries=2,
            min_providers_required=2,
            abort_on_all_failures=False,
        )
        novel = "some totally novel failure"
        policy.decide("p", novel, "drafts", 0)  # consumes the single UNKNOWN retry
        decision = policy.decide("p", novel, "drafts", 0)
        assert decision.action == DegradationAction.CONTINUE
        assert "degraded" in decision.reason.lower()


class TestExponentialBackoff:
    def test_retry_delay_grows_then_caps(self):
        # Generous retry budget so each call keeps choosing RETRY.
        policy = create_default_policy(max_retries=10)
        delays = []
        for _ in range(6):
            decision = policy.decide("p", "rate_limit", "drafts", 2)
            assert decision.action == DegradationAction.RETRY
            delays.append(decision.retry_delay_ms)

        # First delay is the base; subsequent ones double until the cap.
        assert delays[0] == DegradationPolicy.BASE_RETRY_DELAY_MS
        assert delays[1] == DegradationPolicy.BASE_RETRY_DELAY_MS * 2
        # Strictly non-decreasing and never above the cap.
        for earlier, later in zip(delays, delays[1:], strict=False):
            assert later >= earlier
        assert max(delays) <= DegradationPolicy.MAX_RETRY_DELAY_MS
        # The final delay should have hit the cap given enough doublings.
        assert delays[-1] == DegradationPolicy.MAX_RETRY_DELAY_MS


class TestPolicyStateManagement:
    def test_retry_count_tracked_per_provider_phase(self):
        policy = create_default_policy(max_retries=2)
        # Two different phases keep independent retry counters.
        d_drafts = policy.decide("p", "rate_limit", "drafts", 2)
        d_critique = policy.decide("p", "rate_limit", "critique", 2)
        assert d_drafts.action == DegradationAction.RETRY
        assert d_critique.action == DegradationAction.RETRY
        # Both should have been first attempts (delay == base).
        assert d_drafts.retry_delay_ms == DegradationPolicy.BASE_RETRY_DELAY_MS
        assert d_critique.retry_delay_ms == DegradationPolicy.BASE_RETRY_DELAY_MS

    def test_reset_clears_retry_counts(self):
        policy = create_default_policy(max_retries=1)
        policy.decide("p", "rate_limit", "drafts", 2)  # retry #1
        # Without reset, next call would exhaust retries -> SKIP.
        policy.reset()
        decision = policy.decide("p", "rate_limit", "drafts", 2)
        assert decision.action == DegradationAction.RETRY

    def test_decide_returns_decision_type(self):
        policy = create_default_policy()
        decision = policy.decide("p", "rate_limit", "drafts", 2)
        assert isinstance(decision, DegradationDecision)

    def test_accepts_exception_object_as_error(self):
        policy = create_default_policy()
        decision = policy.decide("p", RuntimeError("rate limit exceeded"), "drafts", 2)
        assert decision.action == DegradationAction.RETRY


class TestCreateDefaultPolicy:
    def test_returns_policy_with_defaults(self):
        policy = create_default_policy()
        assert isinstance(policy, DegradationPolicy)
        assert isinstance(policy.get_report(), DegradationReport)

    def test_custom_fallbacks_passed_through(self):
        policy = create_default_policy(fallback_providers={"a": "b"})
        decision = policy.decide("a", "model not found", "drafts", 1)
        assert decision.action == DegradationAction.FALLBACK
        assert decision.fallback_provider == "b"
