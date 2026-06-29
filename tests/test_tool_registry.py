"""Tests for the declarative tool registry (registry/tool_registry.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_council.registry.tool_registry import (
    CapabilityPack,
    Tool,
    ToolParameter,
    ToolRegistry,
    get_tool_registry,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "tool_registry.yaml"


@pytest.fixture
def loaded_registry() -> ToolRegistry:
    """Return a fresh registry loaded from the bundled YAML config."""
    registry = ToolRegistry()
    registry.load_from_yaml(CONFIG_PATH)
    return registry


class TestToolParameter:
    """Tests for ToolParameter JSON schema conversion."""

    def test_to_json_schema_minimal(self):
        """A bare parameter yields just its type."""
        param = ToolParameter(name="query", type="string")
        assert param.to_json_schema() == {"type": "string"}

    def test_to_json_schema_with_description(self):
        """Description is included when provided."""
        param = ToolParameter(name="query", type="string", description="Search query")
        schema = param.to_json_schema()
        assert schema["type"] == "string"
        assert schema["description"] == "Search query"

    def test_to_json_schema_includes_non_none_default(self):
        """A non-None default is surfaced in the schema."""
        param = ToolParameter(name="limit", type="integer", default=5)
        assert param.to_json_schema()["default"] == 5

    def test_to_json_schema_omits_none_default(self):
        """A None default is omitted (the dataclass default)."""
        param = ToolParameter(name="topic", type="string")
        assert "default" not in param.to_json_schema()

    def test_required_defaults_to_true(self):
        """Parameters are required unless stated otherwise."""
        assert ToolParameter(name="x", type="string").required is True


class TestToolConversion:
    """Tests for Tool -> provider format conversion."""

    def _sample_tool(self) -> Tool:
        return Tool(
            name="web_search",
            description="Search the web",
            parameters=[
                ToolParameter(name="query", type="string", required=True),
                ToolParameter(name="limit", type="integer", required=False, default=5),
            ],
            roles=["drafter", "researcher"],
        )

    def test_to_openai_tool_shape(self):
        """OpenAI format nests under function with object parameters."""
        result = self._sample_tool().to_openai_tool()
        assert result["type"] == "function"
        assert result["function"]["name"] == "web_search"
        assert result["function"]["description"] == "Search the web"
        params = result["function"]["parameters"]
        assert params["type"] == "object"
        assert set(params["properties"]) == {"query", "limit"}

    def test_to_openai_tool_required_list(self):
        """Only required params appear in the required list."""
        result = self._sample_tool().to_openai_tool()
        assert result["function"]["parameters"]["required"] == ["query"]

    def test_to_anthropic_tool_shape(self):
        """Anthropic format uses input_schema at the top level."""
        result = self._sample_tool().to_anthropic_tool()
        assert result["name"] == "web_search"
        assert result["description"] == "Search the web"
        assert result["input_schema"]["type"] == "object"
        assert set(result["input_schema"]["properties"]) == {"query", "limit"}

    def test_to_anthropic_tool_required_list(self):
        """Only required params appear in the Anthropic required list."""
        result = self._sample_tool().to_anthropic_tool()
        assert result["input_schema"]["required"] == ["query"]

    def test_tool_with_no_parameters(self):
        """A tool without parameters yields empty properties and required."""
        tool = Tool(name="noop", description="does nothing")
        openai = tool.to_openai_tool()
        assert openai["function"]["parameters"]["properties"] == {}
        assert openai["function"]["parameters"]["required"] == []


class TestToolRegistryRegistration:
    """Tests for direct registration and lookup APIs."""

    def test_register_and_get_tool(self):
        """A registered tool is retrievable by name."""
        registry = ToolRegistry()
        tool = Tool(name="alpha", description="d")
        registry.register_tool(tool)
        assert registry.get_tool("alpha") is tool

    def test_get_tool_missing_returns_none(self):
        """Unknown tool names return None."""
        assert ToolRegistry().get_tool("does-not-exist") is None

    def test_get_all_tools(self):
        """get_all_tools returns every registered tool."""
        registry = ToolRegistry()
        registry.register_tool(Tool(name="a", description="d"))
        registry.register_tool(Tool(name="b", description="d"))
        names = {t.name for t in registry.get_all_tools()}
        assert names == {"a", "b"}

    def test_register_tool_overwrites_same_name(self):
        """Registering a tool with an existing name replaces it."""
        registry = ToolRegistry()
        registry.register_tool(Tool(name="a", description="first"))
        registry.register_tool(Tool(name="a", description="second"))
        assert registry.get_tool("a").description == "second"
        assert len(registry.get_all_tools()) == 1

    def test_register_and_get_capability_pack(self):
        """A registered capability pack is retrievable by name."""
        registry = ToolRegistry()
        pack = CapabilityPack(name="p", description="d")
        registry.register_capability_pack(pack)
        assert registry.get_capability_pack("p") is pack

    def test_get_capability_pack_missing_returns_none(self):
        """Unknown capability pack names return None."""
        assert ToolRegistry().get_capability_pack("missing") is None


class TestToolRegistryRoleFiltering:
    """Tests for role-based tool filtering."""

    def _registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register_tool(Tool(name="drafter_only", description="d", roles=["drafter"]))
        registry.register_tool(Tool(name="shared", description="d", roles=["drafter", "critic"]))
        registry.register_tool(Tool(name="universal", description="d", roles=[]))
        return registry

    def test_role_filter_includes_matching_and_roleless(self):
        """A role gets its tools plus any tool with no role restriction."""
        registry = self._registry()
        names = {t.name for t in registry.get_tools_for_role("drafter")}
        assert names == {"drafter_only", "shared", "universal"}

    def test_role_filter_excludes_non_matching(self):
        """A role does not get tools scoped to other roles."""
        registry = self._registry()
        names = {t.name for t in registry.get_tools_for_role("critic")}
        assert names == {"shared", "universal"}
        assert "drafter_only" not in names

    def test_roleless_tool_available_to_unknown_role(self):
        """A tool with empty roles is available to any role."""
        registry = self._registry()
        names = {t.name for t in registry.get_tools_for_role("nobody")}
        assert names == {"universal"}


class TestToolRegistryYamlLoading:
    """Tests for YAML config loading behavior."""

    def test_load_from_yaml_populates_tools(self, loaded_registry: ToolRegistry):
        """Loading the bundled config registers the expected tools."""
        names = {t.name for t in loaded_registry.get_all_tools()}
        assert {"web_search", "context7_lookup", "code_analysis", "read_file"} <= names

    def test_load_from_yaml_populates_capability_packs(self, loaded_registry: ToolRegistry):
        """Loading the bundled config registers capability packs."""
        packs = loaded_registry.list_capability_packs()
        assert "repo-analysis" in packs
        assert "docs-research" in packs

    def test_load_parses_parameter_metadata(self, loaded_registry: ToolRegistry):
        """Parameter default/type are parsed from YAML."""
        tool = loaded_registry.get_tool("web_search")
        assert tool is not None
        params = {p.name: p for p in tool.parameters}
        assert params["query"].required is True
        assert params["limit"].default == 5
        assert params["limit"].type == "integer"

    def test_load_required_defaults_true_when_key_absent(self, loaded_registry: ToolRegistry):
        """When a YAML param omits ``required``, it parses as required=True.

        Note: web_search.limit has a ``default`` but no ``required`` key, so the
        parser marks it required. This documents current behavior (see summary).
        """
        tool = loaded_registry.get_tool("web_search")
        assert tool is not None
        limit = next(p for p in tool.parameters if p.name == "limit")
        assert limit.required is True

    def test_load_explicit_required_false_is_honored(self, loaded_registry: ToolRegistry):
        """An explicit ``required: false`` in YAML is parsed as not required."""
        tool = loaded_registry.get_tool("context7_lookup")
        assert tool is not None
        topic = next(p for p in tool.parameters if p.name == "topic")
        assert topic.required is False

    def test_load_missing_file_is_noop(self, tmp_path: Path):
        """Loading a non-existent path logs a warning and registers nothing."""
        registry = ToolRegistry()
        registry.load_from_yaml(tmp_path / "nope.yaml")
        assert registry.get_all_tools() == []
        assert registry._loaded is False

    def test_load_empty_yaml(self, tmp_path: Path):
        """An empty YAML file loads cleanly with no tools."""
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("")
        registry = ToolRegistry()
        registry.load_from_yaml(cfg)
        assert registry.get_all_tools() == []
        assert registry._loaded is True

    def test_load_shorthand_parameter_type(self, tmp_path: Path):
        """Scalar parameter values are treated as the type shorthand."""
        cfg = tmp_path / "tools.yaml"
        cfg.write_text(
            "tools:\n  echo:\n    description: echo a value\n    parameters:\n      value: string\n"
        )
        registry = ToolRegistry()
        registry.load_from_yaml(cfg)
        tool = registry.get_tool("echo")
        assert tool is not None
        assert len(tool.parameters) == 1
        assert tool.parameters[0].name == "value"
        assert tool.parameters[0].type == "string"

    def test_load_invalid_yaml_does_not_raise(self, tmp_path: Path):
        """Malformed YAML is caught and logged, not raised."""
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("tools: [unterminated")
        registry = ToolRegistry()
        registry.load_from_yaml(cfg)  # should not raise
        assert registry.get_all_tools() == []


class TestToolRegistryEnsureLoaded:
    """Tests for lazy loading via ensure_loaded."""

    def test_ensure_loaded_with_explicit_path(self):
        """An explicit path is loaded on first ensure_loaded call."""
        registry = ToolRegistry()
        registry.ensure_loaded(CONFIG_PATH)
        assert registry.get_tool("web_search") is not None

    def test_ensure_loaded_is_idempotent(self, tmp_path: Path):
        """A second ensure_loaded does not reload once loaded."""
        cfg = tmp_path / "tools.yaml"
        cfg.write_text("tools:\n  a:\n    description: d\n")
        registry = ToolRegistry()
        registry.ensure_loaded(cfg)
        assert registry._loaded is True
        # Mutate the file; a reload would pick up "b", idempotency keeps only "a".
        cfg.write_text("tools:\n  b:\n    description: d\n")
        registry.ensure_loaded(cfg)
        names = {t.name for t in registry.get_all_tools()}
        assert names == {"a"}

    def test_ensure_loaded_no_default_config_stays_empty(self, monkeypatch):
        """With no discoverable default config, the registry stays empty."""
        from llm_council.registry import tool_registry as tr_module

        monkeypatch.setattr(tr_module, "DEFAULT_CONFIG_CANDIDATES", ())
        registry = ToolRegistry()
        registry.ensure_loaded()
        assert registry.get_all_tools() == []
        assert registry._loaded is False


class TestCapabilityPackResolution:
    """Tests for resolve_capability_tools."""

    def _registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register_tool(Tool(name="t1", description="d", roles=["drafter"]))
        registry.register_tool(Tool(name="t2", description="d", roles=["critic"]))
        registry.register_tool(Tool(name="t3", description="d", roles=[]))
        registry.register_capability_pack(
            CapabilityPack(name="pack", description="d", tools=["t1", "t2", "t3"])
        )
        return registry

    def test_resolve_all_tools_without_role(self):
        """Without a role filter, all referenced tools resolve."""
        registry = self._registry()
        tools = registry.resolve_capability_tools(["pack"])
        assert [t.name for t in tools] == ["t1", "t2", "t3"]

    def test_resolve_filters_tools_by_role(self):
        """Role filtering drops tools scoped to other roles."""
        registry = self._registry()
        tools = registry.resolve_capability_tools(["pack"], role="drafter")
        assert [t.name for t in tools] == ["t1", "t3"]

    def test_resolve_skips_missing_pack(self):
        """A non-existent pack name is skipped silently."""
        registry = self._registry()
        tools = registry.resolve_capability_tools(["nope"])
        assert tools == []

    def test_resolve_skips_missing_tool_reference(self):
        """A pack referencing an unregistered tool skips that reference."""
        registry = ToolRegistry()
        registry.register_tool(Tool(name="real", description="d"))
        registry.register_capability_pack(
            CapabilityPack(name="pack", description="d", tools=["real", "ghost"])
        )
        tools = registry.resolve_capability_tools(["pack"])
        assert [t.name for t in tools] == ["real"]

    def test_resolve_deduplicates_across_packs(self):
        """A tool referenced by multiple packs appears once."""
        registry = self._registry()
        registry.register_capability_pack(
            CapabilityPack(name="pack2", description="d", tools=["t1", "t3"])
        )
        tools = registry.resolve_capability_tools(["pack", "pack2"])
        assert [t.name for t in tools] == ["t1", "t2", "t3"]

    def test_resolve_pack_role_mismatch_skips_pack(self):
        """A pack scoped to other roles is skipped entirely for a mismatched role."""
        registry = self._registry()
        registry.register_capability_pack(
            CapabilityPack(name="critic_pack", description="d", tools=["t3"], roles=["critic"])
        )
        tools = registry.resolve_capability_tools(["critic_pack"], role="drafter")
        assert tools == []


class TestToolRegistryProviderConversion:
    """Tests for registry-level conversion to provider tool formats."""

    def test_to_openai_tools_all(self, loaded_registry: ToolRegistry):
        """Without a role, all tools convert to OpenAI format."""
        tools = loaded_registry.to_openai_tools()
        assert len(tools) == len(loaded_registry.get_all_tools())
        assert all(t["type"] == "function" for t in tools)

    def test_to_openai_tools_role_filtered(self, loaded_registry: ToolRegistry):
        """A role filter restricts the converted OpenAI tool set."""
        tools = loaded_registry.to_openai_tools(role="synthesizer")
        names = {t["function"]["name"] for t in tools}
        # create_checklist is scoped to planner+synthesizer; read_file is not.
        assert "create_checklist" in names
        assert "read_file" not in names

    def test_to_anthropic_tools_all(self, loaded_registry: ToolRegistry):
        """Without a role, all tools convert to Anthropic format."""
        tools = loaded_registry.to_anthropic_tools()
        assert len(tools) == len(loaded_registry.get_all_tools())
        assert all("input_schema" in t for t in tools)


class TestToolRegistrySingleton:
    """Tests for the singleton accessor."""

    def test_get_instance_returns_same_object(self):
        """get_instance returns the same instance across calls."""
        first = ToolRegistry.get_instance()
        second = ToolRegistry.get_instance()
        assert first is second

    def test_get_tool_registry_matches_get_instance(self):
        """The module-level accessor returns the singleton."""
        assert get_tool_registry() is ToolRegistry.get_instance()
