"""Tests for the pure parsing/command helpers of the CLI provider adapters.

These functions do the JSONL/JSON payload extraction and command assembly for
the Codex and Gemini CLI subprocess adapters. They are pure (no subprocess),
so they can be exercised directly without mocking ``asyncio.create_subprocess_exec``.
"""

from __future__ import annotations

import json

import pytest

from llm_council.providers.base import GenerateRequest, Message, StructuredOutputConfig
from llm_council.providers.cli.codex import (
    CodexCLIProvider,
    _extract_agent_message,
    _extract_error_message,
    _extract_usage_payload,
    _ingest_codex_stdout_line,
    _LiveCodexState,
    _prepare_schema_for_codex,
)
from llm_council.providers.cli.gemini_cli import (
    GeminiCLIProvider,
    _extract_json_payload,
    _extract_text_payload,
    _format_gemini_error,
    _normalize_usage,
)


def _jsonl(*payloads: dict) -> str:
    return "\n".join(json.dumps(p) for p in payloads)


class TestCodexUsageExtraction:
    """`_extract_usage_payload` reads token usage from turn.completed events."""

    def test_extracts_usage_and_sums_cached_input(self):
        stdout = _jsonl(
            {"type": "turn.started"},
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 20,
                    "output_tokens": 50,
                },
            },
        )
        usage = _extract_usage_payload(stdout)
        assert usage == {
            "prompt_tokens": 120,
            "completion_tokens": 50,
            "total_tokens": 170,
        }

    def test_returns_none_when_no_turn_completed(self):
        stdout = _jsonl({"type": "turn.started"})
        assert _extract_usage_payload(stdout) is None

    def test_returns_none_when_usage_missing(self):
        stdout = _jsonl({"type": "turn.completed"})
        assert _extract_usage_payload(stdout) is None

    def test_ignores_blank_and_non_json_lines(self):
        stdout = "\n".join(
            [
                "",
                "not json at all",
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 5, "output_tokens": 3},
                    }
                ),
            ]
        )
        usage = _extract_usage_payload(stdout)
        assert usage == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}

    def test_handles_null_token_fields(self):
        stdout = _jsonl(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": None, "output_tokens": None},
            }
        )
        assert _extract_usage_payload(stdout) == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


class TestCodexAgentMessageExtraction:
    """`_extract_agent_message` returns the last agent_message text."""

    def test_returns_last_agent_message(self):
        stdout = _jsonl(
            {"type": "item.completed", "item": {"type": "agent_message", "text": "first"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "second"}},
        )
        assert _extract_agent_message(stdout) == "second"

    def test_ignores_non_agent_items(self):
        stdout = _jsonl(
            {"type": "item.completed", "item": {"type": "tool_call", "text": "nope"}},
        )
        assert _extract_agent_message(stdout) == ""

    def test_empty_when_no_messages(self):
        assert _extract_agent_message("") == ""

    def test_ignores_malformed_lines(self):
        stdout = "garbage\n" + json.dumps(
            {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}}
        )
        assert _extract_agent_message(stdout) == "ok"


class TestCodexErrorMessageExtraction:
    """`_extract_error_message` surfaces error payloads from stdout JSONL."""

    def test_returns_error_message(self):
        stdout = _jsonl({"type": "error", "message": "rate limit hit"})
        assert _extract_error_message(stdout) == "rate limit hit"

    def test_returns_empty_without_error_event(self):
        stdout = _jsonl({"type": "turn.completed"})
        assert _extract_error_message(stdout) == ""

    def test_returns_last_error(self):
        stdout = _jsonl(
            {"type": "error", "message": "first"},
            {"type": "error", "message": "last"},
        )
        assert _extract_error_message(stdout) == "last"


class TestCodexSchemaPrep:
    """`_prepare_schema_for_codex` enforces strict structured-output rules."""

    def test_strips_meta_keys_and_closes_object(self):
        schema = {
            "$schema": "https://json-schema.org/draft-07/schema",
            "type": "object",
            "additionalProperties": True,
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        }
        result = _prepare_schema_for_codex(schema)

        assert "$schema" not in result
        assert result["additionalProperties"] is False
        # All properties are promoted to required.
        assert result["required"] == ["a"]

    def test_nested_objects_are_recursively_closed(self):
        schema = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {"inner": {"type": "string"}},
                },
            },
        }
        result = _prepare_schema_for_codex(schema)

        nested = result["properties"]["outer"]
        assert nested["additionalProperties"] is False
        assert nested["required"] == ["inner"]

    def test_array_of_objects_items_are_closed(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
            },
        }
        result = _prepare_schema_for_codex(schema)

        item_schema = result["properties"]["items"]["items"]
        assert item_schema["additionalProperties"] is False
        assert item_schema["required"] == ["name"]


class TestCodexLiveStateIngestion:
    """`_ingest_codex_stdout_line` mutates live subprocess state incrementally."""

    def test_turn_started_flag(self):
        state = _LiveCodexState()
        _ingest_codex_stdout_line(json.dumps({"type": "turn.started"}), state)
        assert state.saw_turn_started is True

    def test_agent_message_captured(self):
        state = _LiveCodexState()
        _ingest_codex_stdout_line(
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}),
            state,
        )
        assert state.agent_message == "hi"

    def test_turn_completed_records_usage(self):
        state = _LiveCodexState()
        _ingest_codex_stdout_line(
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 10, "output_tokens": 4},
                }
            ),
            state,
        )
        assert state.saw_turn_completed is True
        assert state.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        }

    def test_error_message_captured(self):
        state = _LiveCodexState()
        _ingest_codex_stdout_line(json.dumps({"type": "error", "message": "boom"}), state)
        assert state.error_message == "boom"

    def test_blank_and_invalid_lines_are_noops(self):
        state = _LiveCodexState()
        _ingest_codex_stdout_line("", state)
        _ingest_codex_stdout_line("not-json", state)
        assert state.agent_message == ""
        assert not state.saw_turn_started
        # The raw line is still buffered for stdout reconstruction.
        assert state.stdout_parts == ["", "not-json"]


class TestCodexBuildCommand:
    """`_build_command` assembles a safe argument list."""

    def test_prompt_and_flags_present(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        request = GenerateRequest(prompt="hello world")

        cmd = provider._build_command(request, model="gpt-5.4")

        assert cmd[0] == "/usr/local/bin/codex"
        assert cmd[1] == "exec"
        assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "gpt-5.4"
        assert cmd[-1] == "hello world"
        # Least-privilege sandbox flag from DEFAULT_FLAGS.
        assert "read-only" in cmd

    def test_messages_are_joined_from_user_turns(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        request = GenerateRequest(
            messages=[
                Message(role="system", content="ignored"),
                Message(role="user", content="one"),
                Message(role="user", content="two"),
            ]
        )

        cmd = provider._build_command(request, model="m")

        assert cmd[-1] == "one\n\ntwo"

    def test_schema_and_output_paths_are_wired(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        request = GenerateRequest(prompt="p")

        cmd = provider._build_command(
            request,
            model="m",
            output_last_message_path="/tmp/out.txt",
            output_schema_path="/tmp/schema.json",
        )

        assert "--output-schema" in cmd
        assert cmd[cmd.index("--output-schema") + 1] == "/tmp/schema.json"
        assert "-o" in cmd
        assert cmd[cmd.index("-o") + 1] == "/tmp/out.txt"

    def test_missing_cli_path_raises(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        provider._cli_path = None
        with pytest.raises(RuntimeError, match="Codex CLI not found"):
            provider._build_command(GenerateRequest(prompt="p"), model="m")


class TestCodexResolveModel:
    """`_resolve_model` normalizes `*-codex` names under ChatGPT auth."""

    @pytest.mark.asyncio
    async def test_non_codex_model_unchanged(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        request = GenerateRequest(prompt="p", model="gpt-5.4")
        assert await provider._resolve_model(request) == "gpt-5.4"

    @pytest.mark.asyncio
    async def test_codex_suffix_stripped_when_chatgpt_login(self, monkeypatch):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")

        async def fake_status(self):
            return "Logged in using ChatGPT"

        monkeypatch.setattr(CodexCLIProvider, "_login_status_text", fake_status)
        request = GenerateRequest(prompt="p", model="gpt-5.4-codex")

        assert await provider._resolve_model(request) == "gpt-5.4"

    @pytest.mark.asyncio
    async def test_codex_suffix_kept_when_not_chatgpt(self, monkeypatch):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")

        async def fake_status(self):
            return "Logged in using API key"

        monkeypatch.setattr(CodexCLIProvider, "_login_status_text", fake_status)
        request = GenerateRequest(prompt="p", model="gpt-5.4-codex")

        assert await provider._resolve_model(request) == "gpt-5.4-codex"


class TestCodexErrorDetails:
    """`_error_details` collapses stderr to the most relevant lines."""

    def test_prefers_error_prefixed_lines(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        stderr = "noise line\nERROR: something broke\nmore noise"
        assert provider._error_details(stderr) == "ERROR: something broke"

    def test_falls_back_to_last_lines(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        stderr = "\n".join(f"line{i}" for i in range(20))
        details = provider._error_details(stderr)
        # Keeps the trailing lines, drops the earliest ones.
        assert "line19" in details
        assert "line0\n" not in details

    def test_empty_stderr(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        assert provider._error_details("   ") == ""


class TestGeminiTextExtraction:
    """`_extract_text_payload` walks nested Gemini CLI JSON shapes."""

    def test_plain_string(self):
        assert _extract_text_payload("hello") == "hello"

    def test_none_returns_empty(self):
        assert _extract_text_payload(None) == ""

    def test_dict_with_text_key(self):
        assert _extract_text_payload({"text": "answer"}) == "answer"

    def test_dict_nested_response_key(self):
        assert _extract_text_payload({"response": {"text": "nested"}}) == "nested"

    def test_list_joins_non_empty_parts(self):
        payload = [{"text": "a"}, {"text": ""}, {"text": "b"}]
        assert _extract_text_payload(payload) == "a\nb"

    def test_unknown_shape_returns_empty(self):
        assert _extract_text_payload(42) == ""


class TestGeminiUsageNormalization:
    """`_normalize_usage` maps Gemini stat shapes to council usage."""

    def test_camel_case_fields(self):
        stats = {"inputTokenCount": 10, "outputTokenCount": 5, "totalTokenCount": 15}
        assert _normalize_usage(stats) == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_snake_case_fallback(self):
        stats = {"input_tokens": 3, "output_tokens": 2}
        usage = _normalize_usage(stats)
        assert usage == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }

    def test_total_derived_when_absent(self):
        stats = {"inputTokenCount": 4, "outputTokenCount": 6}
        assert _normalize_usage(stats)["total_tokens"] == 10

    def test_non_dict_returns_none(self):
        assert _normalize_usage("nope") is None


class TestGeminiErrorFormatting:
    """`_format_gemini_error` produces a stable string across shapes."""

    def test_string_passthrough(self):
        assert _format_gemini_error("boom") == "boom"

    def test_dict_message_key(self):
        assert _format_gemini_error({"message": "bad thing"}) == "bad thing"

    def test_dict_without_message_is_json(self):
        result = _format_gemini_error({"code": 500})
        assert json.loads(result) == {"code": 500}


class TestGeminiJsonExtraction:
    """`_extract_json_payload` tolerates a warning prelude before JSON."""

    def test_clean_json(self):
        assert _extract_json_payload('{"a": 1}') == {"a": 1}

    def test_json_after_warning_lines(self):
        stdout = 'WARN: deprecated flag\nignored line\n{"text": "ok"}'
        assert _extract_json_payload(stdout) == {"text": "ok"}

    def test_json_array_payload(self):
        assert _extract_json_payload("[1, 2, 3]") == [1, 2, 3]

    def test_empty_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json_payload("   ")

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json_payload("just some log text\nno json here")


class TestGeminiBuildCommand:
    """`_build_command` assembles the Gemini CLI argument list."""

    def test_prompt_model_and_format(self):
        provider = GeminiCLIProvider(cli_path="/opt/gemini")
        request = GenerateRequest(prompt="hi", model="gemini-3-pro")

        cmd = provider._build_command(request)

        assert cmd[0] == "/opt/gemini"
        assert cmd[cmd.index("-p") + 1] == "hi"
        assert cmd[cmd.index("-m") + 1] == "gemini-3-pro"
        assert cmd[cmd.index("--output-format") + 1] == "json"
        assert cmd[cmd.index("--approval-mode") + 1] == "default"

    def test_default_model_used_when_request_has_none(self):
        provider = GeminiCLIProvider(cli_path="/opt/gemini", default_model="gemini-x")
        cmd = provider._build_command(GenerateRequest(prompt="hi"))
        assert cmd[cmd.index("-m") + 1] == "gemini-x"

    def test_missing_cli_path_raises(self):
        provider = GeminiCLIProvider(cli_path="/opt/gemini")
        provider._cli_path = None
        with pytest.raises(RuntimeError, match="Gemini CLI not found"):
            provider._build_command(GenerateRequest(prompt="p"))

    def test_system_only_messages_yield_empty_prompt(self):
        """Only user turns feed the prompt; a system-only request builds an empty prompt."""
        provider = GeminiCLIProvider(cli_path="/opt/gemini")
        request = GenerateRequest(messages=[Message(role="system", content="only system")])
        cmd = provider._build_command(request)
        assert cmd[cmd.index("-p") + 1] == ""


class TestCodexStructuredOutputFlag:
    """Sanity check that structured output requests carry a schema through."""

    def test_request_accepts_structured_output(self):
        request = GenerateRequest(
            prompt="p",
            structured_output=StructuredOutputConfig(
                json_schema={"type": "object", "properties": {"x": {"type": "string"}}}
            ),
        )
        assert request.structured_output is not None
        prepared = _prepare_schema_for_codex(dict(request.structured_output.json_schema))
        assert prepared["additionalProperties"] is False
