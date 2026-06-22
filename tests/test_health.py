"""Dedicated unit tests for ``llm_council.engine.health``.

Focuses on branch coverage of the health-check flow and the report/status
dataclasses. Complements (does not duplicate) the high-level smoke tests in
``test_engine.py`` by exercising the less-trodden branches: exception-path
severity classification (AUTH/BILLING -> DOWN vs transient -> DEGRADED),
DoctorResult not-ok error-type classification, ``should_skip_provider``,
empty-input aggregation, the UNKNOWN status produced by ``check_all`` when a
check raises, cache TTL expiry, and ``ProviderHealth.to_dict`` serialization.
"""

from __future__ import annotations

import asyncio

import pytest

from llm_council.engine.health import (
    HealthChecker,
    HealthReport,
    HealthStatus,
    ProviderHealth,
    preflight_check,
)
from llm_council.providers.base import DoctorResult, ErrorType


class StubProvider:
    """Minimal provider stub exposing only ``doctor()``."""

    def __init__(self, result: DoctorResult):
        self._result = result

    async def doctor(self) -> DoctorResult:
        return self._result


class RaisingProvider:
    """Provider whose ``doctor()`` raises a chosen exception."""

    def __init__(self, exc: Exception):
        self._exc = exc

    async def doctor(self) -> DoctorResult:
        raise self._exc


# ---------------------------------------------------------------------------
# ProviderHealth
# ---------------------------------------------------------------------------
class TestProviderHealth:
    def test_is_usable_matrix(self):
        assert ProviderHealth("p", HealthStatus.OK).is_usable()
        assert ProviderHealth("p", HealthStatus.DEGRADED).is_usable()
        assert not ProviderHealth("p", HealthStatus.DOWN).is_usable()
        assert not ProviderHealth("p", HealthStatus.UNKNOWN).is_usable()

    def test_default_checked_at_is_set(self):
        health = ProviderHealth("p", HealthStatus.OK)
        assert isinstance(health.checked_at, str)
        assert health.checked_at


# ---------------------------------------------------------------------------
# HealthReport aggregation / serialization
# ---------------------------------------------------------------------------
class TestHealthReport:
    def test_get_usable_and_down_with_mixed_statuses(self):
        report = HealthReport(
            providers=[
                ProviderHealth("ok", HealthStatus.OK),
                ProviderHealth("degraded", HealthStatus.DEGRADED),
                ProviderHealth("down", HealthStatus.DOWN),
                ProviderHealth("unknown", HealthStatus.UNKNOWN),
            ],
            all_healthy=False,
            usable_count=2,
            total_count=4,
        )
        assert report.get_usable_providers() == ["ok", "degraded"]
        assert report.get_down_providers() == ["down"]

    def test_empty_report(self):
        report = HealthReport(providers=[], all_healthy=True, usable_count=0, total_count=0)
        assert report.get_usable_providers() == []
        assert report.get_down_providers() == []

    def test_to_dict_serializes_error_type(self):
        report = HealthReport(
            providers=[
                ProviderHealth("ok", HealthStatus.OK, latency_ms=12.0),
                ProviderHealth(
                    "auth",
                    HealthStatus.DOWN,
                    message="bad key",
                    error_type=ErrorType.AUTH,
                ),
            ],
            all_healthy=False,
            usable_count=1,
            total_count=2,
            check_duration_ms=42,
        )
        data = report.to_dict()
        assert data["all_healthy"] is False
        assert data["usable_count"] == 1
        assert data["total_count"] == 2
        assert data["check_duration_ms"] == 42

        by_name = {p["provider"]: p for p in data["providers"]}
        assert by_name["ok"]["status"] == "ok"
        assert by_name["ok"]["error_type"] is None
        assert by_name["auth"]["status"] == "down"
        assert by_name["auth"]["error_type"] == "auth"


# ---------------------------------------------------------------------------
# HealthChecker.check_provider branches
# ---------------------------------------------------------------------------
class TestCheckProviderBranches:
    @pytest.mark.asyncio
    async def test_ok_result(self):
        checker = HealthChecker()
        health = await checker.check_provider(
            "p", StubProvider(DoctorResult(ok=True, message="all good", latency_ms=5.0))
        )
        assert health.status == HealthStatus.OK
        assert health.message == "all good"
        assert health.latency_ms == 5.0

    @pytest.mark.asyncio
    async def test_ok_result_with_details(self):
        checker = HealthChecker()
        health = await checker.check_provider(
            "p",
            StubProvider(DoctorResult(ok=True, message="ok", details={"region": "us"})),
        )
        assert health.status == HealthStatus.OK
        assert health.details == {"region": "us"}

    @pytest.mark.asyncio
    async def test_not_ok_result_classifies_error(self):
        checker = HealthChecker()
        health = await checker.check_provider(
            "p", StubProvider(DoctorResult(ok=False, message="invalid_api_key"))
        )
        assert health.status == HealthStatus.DOWN
        assert health.error_type == ErrorType.AUTH

    @pytest.mark.asyncio
    async def test_not_ok_result_without_message(self):
        checker = HealthChecker()
        health = await checker.check_provider("p", StubProvider(DoctorResult(ok=False)))
        assert health.status == HealthStatus.DOWN
        assert health.message == "Health check failed"

    @pytest.mark.asyncio
    async def test_timeout_yields_degraded(self):
        class SlowProvider:
            async def doctor(self):
                await asyncio.sleep(5)
                return DoctorResult(ok=True)

        checker = HealthChecker(timeout=0.05)
        health = await checker.check_provider("slow", SlowProvider())
        assert health.status == HealthStatus.DEGRADED
        assert health.error_type == ErrorType.TIMEOUT

    @pytest.mark.asyncio
    async def test_exception_auth_is_down(self):
        checker = HealthChecker()
        health = await checker.check_provider(
            "p", RaisingProvider(RuntimeError("unauthorized invalid_api_key"))
        )
        assert health.status == HealthStatus.DOWN
        assert health.error_type == ErrorType.AUTH

    @pytest.mark.asyncio
    async def test_exception_billing_is_down(self):
        checker = HealthChecker()
        health = await checker.check_provider(
            "p", RaisingProvider(RuntimeError("insufficient_quota: billing"))
        )
        assert health.status == HealthStatus.DOWN
        assert health.error_type == ErrorType.BILLING

    @pytest.mark.asyncio
    async def test_exception_transient_is_degraded(self):
        checker = HealthChecker()
        health = await checker.check_provider(
            "p", RaisingProvider(RuntimeError("connection reset by peer"))
        )
        assert health.status == HealthStatus.DEGRADED
        assert health.error_type == ErrorType.NETWORK

    @pytest.mark.asyncio
    async def test_exception_message_is_truncated(self):
        checker = HealthChecker()
        long = "boom " * 100
        health = await checker.check_provider("p", RaisingProvider(RuntimeError(long)))
        # message is "Health check error: " + first 100 chars of error text
        assert health.message.startswith("Health check error: ")
        assert len(health.message) <= len("Health check error: ") + 100


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
class TestCaching:
    @pytest.mark.asyncio
    async def test_cache_hit_returns_stale_result(self):
        checker = HealthChecker()
        provider = StubProvider(DoctorResult(ok=True))
        first = await checker.check_provider("p", provider)

        # Swap behavior; cached value should still be returned.
        checker._cache["p"] = first
        second = await checker.check_provider("p", RaisingProvider(RuntimeError("auth")))
        assert second is first

    @pytest.mark.asyncio
    async def test_cache_expiry_triggers_recheck(self):
        checker = HealthChecker()
        checker._cache_ttl = 0.0  # force immediate expiry
        provider_ok = StubProvider(DoctorResult(ok=True))
        first = await checker.check_provider("p", provider_ok)
        assert first.status == HealthStatus.OK

        # With TTL=0 the cached entry is always considered stale.
        second = await checker.check_provider(
            "p", StubProvider(DoctorResult(ok=False, message="invalid_api_key"))
        )
        assert second.status == HealthStatus.DOWN

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        checker = HealthChecker()
        await checker.check_provider("p", StubProvider(DoctorResult(ok=True)))
        assert "p" in checker._cache
        checker.clear_cache()
        assert checker._cache == {}


# ---------------------------------------------------------------------------
# check_all aggregation
# ---------------------------------------------------------------------------
class TestCheckAll:
    @pytest.mark.asyncio
    async def test_empty_providers(self):
        checker = HealthChecker()
        report = await checker.check_all({})
        assert report.total_count == 0
        assert report.usable_count == 0
        # vacuous truth: all() over empty is True
        assert report.all_healthy is True
        assert report.get_usable_providers() == []

    @pytest.mark.asyncio
    async def test_all_failed(self):
        checker = HealthChecker()
        providers = {
            "a": StubProvider(DoctorResult(ok=False, message="invalid_api_key")),
            "b": StubProvider(DoctorResult(ok=False, message="invalid_api_key")),
        }
        report = await checker.check_all(providers)
        assert report.total_count == 2
        assert report.usable_count == 0
        assert not report.all_healthy
        assert set(report.get_down_providers()) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        # Short timeout so the slow provider degrades quickly.
        checker = HealthChecker(timeout=0.05)
        providers = {
            "ok": StubProvider(DoctorResult(ok=True)),
            "down": StubProvider(DoctorResult(ok=False, message="invalid_api_key")),
            "slow": _make_slow_provider(),
        }
        report = await checker.check_all(providers)
        assert report.total_count == 3
        # ok + degraded(slow timeout) are usable; down is not
        assert report.usable_count == 2
        assert "ok" in report.get_usable_providers()
        assert "slow" in report.get_usable_providers()
        assert report.get_down_providers() == ["down"]
        assert not report.all_healthy
        assert not report.all_healthy

    @pytest.mark.asyncio
    async def test_all_healthy_true_when_every_provider_ok(self):
        checker = HealthChecker()
        providers = {
            "a": StubProvider(DoctorResult(ok=True)),
            "b": StubProvider(DoctorResult(ok=True)),
        }
        report = await checker.check_all(providers)
        assert report.all_healthy is True
        assert report.usable_count == 2


def _make_slow_provider():
    """Build a provider that always times out for the default-ish checker."""

    class _Slow:
        async def doctor(self):
            await asyncio.sleep(30)
            return DoctorResult(ok=True)

    return _Slow()


# ---------------------------------------------------------------------------
# should_skip_provider
# ---------------------------------------------------------------------------
class TestShouldSkipProvider:
    def test_down_is_skipped(self):
        checker = HealthChecker()
        assert checker.should_skip_provider(ProviderHealth("p", HealthStatus.DOWN))

    def test_auth_error_is_skipped_even_if_degraded(self):
        checker = HealthChecker()
        health = ProviderHealth("p", HealthStatus.DEGRADED, error_type=ErrorType.AUTH)
        assert checker.should_skip_provider(health)

    def test_billing_error_is_skipped(self):
        checker = HealthChecker()
        health = ProviderHealth("p", HealthStatus.DEGRADED, error_type=ErrorType.BILLING)
        assert checker.should_skip_provider(health)

    def test_cli_not_found_is_skipped(self):
        checker = HealthChecker()
        health = ProviderHealth("p", HealthStatus.DEGRADED, error_type=ErrorType.CLI_NOT_FOUND)
        assert checker.should_skip_provider(health)

    def test_transient_degraded_is_not_skipped(self):
        checker = HealthChecker()
        health = ProviderHealth("p", HealthStatus.DEGRADED, error_type=ErrorType.NETWORK)
        assert not checker.should_skip_provider(health)

    def test_ok_is_not_skipped(self):
        checker = HealthChecker()
        assert not checker.should_skip_provider(ProviderHealth("p", HealthStatus.OK))


# ---------------------------------------------------------------------------
# preflight_check convenience
# ---------------------------------------------------------------------------
class TestPreflightCheck:
    @pytest.mark.asyncio
    async def test_filters_down_providers(self):
        providers = {
            "good": StubProvider(DoctorResult(ok=True)),
            "bad": StubProvider(DoctorResult(ok=False, message="invalid_api_key")),
        }
        usable, report = await preflight_check(providers, timeout=5.0)
        assert set(usable) == {"good"}
        assert report.usable_count == 1

    @pytest.mark.asyncio
    async def test_skip_on_failure_false_returns_all(self):
        providers = {
            "good": StubProvider(DoctorResult(ok=True)),
            "bad": StubProvider(DoctorResult(ok=False, message="invalid_api_key")),
        }
        usable, report = await preflight_check(providers, timeout=5.0, skip_on_failure=False)
        assert set(usable) == {"good", "bad"}
        assert report.usable_count == 1

    @pytest.mark.asyncio
    async def test_empty_input(self):
        usable, report = await preflight_check({}, timeout=1.0)
        assert usable == {}
        assert report.total_count == 0
