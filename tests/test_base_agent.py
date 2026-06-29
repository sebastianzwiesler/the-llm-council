"""Tests for the base agent abstraction (registry/base_agent.py).

These tests cover the behavior that actually exists today: construction and
wiring, system-prompt assembly, PASS detection, tool loading from the registry,
result/context dataclasses, and the agent factory. The concrete ``run()``
implementations currently return placeholder strings, so they are only asserted
against that documented placeholder behavior, not aspirational model calls.
"""

from __future__ import annotations

import pytest

from llm_council.registry.base_agent import (
    AGENT_CLASSES,
    COUNCIL_PROTOCOL,
    AgentContext,
    AgentResult,
    BaseAgent,
    CriticAgent,
    DrafterAgent,
    PlannerAgent,
    ResearcherAgent,
    SynthesizerAgent,
    create_agent,
)
from llm_council.registry.tool_registry import Tool


class TestAgentResult:
    """Tests for the AgentResult dataclass."""

    def test_defaults(self):
        """A result defaults to not-passed with empty collections."""
        result = AgentResult(content="hello")
        assert result.content == "hello"
        assert result.passed is False
        assert result.tool_calls == []
        assert result.usage == {}
        assert result.metadata == {}

    def test_has_content_true_for_real_content(self):
        """Substantive, non-passed content reports has_content True."""
        assert AgentResult(content="some output").has_content is True

    def test_has_content_false_when_passed(self):
        """A passed result is treated as having no content."""
        assert AgentResult(content="anything", passed=True).has_content is False

    def test_has_content_false_for_whitespace(self):
        """Whitespace-only content is not substantive."""
        assert AgentResult(content="   \n\t ").has_content is False


class TestAgentContext:
    """Tests for the AgentContext dataclass."""

    def test_defaults(self):
        """Context defaults round 1 with empty/None optional fields."""
        ctx = AgentContext(task="do the thing")
        assert ctx.task == "do the thing"
        assert ctx.previous_drafts == {}
        assert ctx.critique is None
        assert ctx.round_number == 1
        assert ctx.mode is None
        assert ctx.extra == {}

    def test_custom_fields(self):
        """Custom field values are stored as given."""
        ctx = AgentContext(
            task="t",
            previous_drafts={"drafter": "draft text"},
            critique="needs work",
            round_number=2,
            mode="impl",
            extra={"k": "v"},
        )
        assert ctx.previous_drafts == {"drafter": "draft text"}
        assert ctx.critique == "needs work"
        assert ctx.round_number == 2
        assert ctx.mode == "impl"
        assert ctx.extra == {"k": "v"}


class TestBaseAgentConstruction:
    """Tests for BaseAgent construction and basic wiring.

    BaseAgent is abstract; DrafterAgent is used as a minimal concrete subclass
    to exercise the shared base behavior.
    """

    def test_base_agent_is_abstract(self):
        """BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent(model="m", system_prompt="p")  # type: ignore[abstract]

    def test_construction_stores_attributes(self):
        """Constructor arguments are stored on the instance."""
        agent = DrafterAgent(model="gpt-test", system_prompt="base prompt")
        assert agent.model == "gpt-test"
        assert agent.max_iterations == 10
        assert agent.include_protocol is True

    def test_tools_default_empty(self):
        """Tools default to an empty list, not None."""
        agent = DrafterAgent(model="m", system_prompt="p")
        assert agent.tools == []

    def test_tools_passed_in(self):
        """Provided tools are exposed via the tools property."""
        tool = Tool(name="x", description="d")
        agent = DrafterAgent(model="m", system_prompt="p", tools=[tool])
        assert agent.tools == [tool]

    def test_custom_max_iterations(self):
        """A custom max_iterations is honored."""
        agent = DrafterAgent(model="m", system_prompt="p", max_iterations=3)
        assert agent.max_iterations == 3


class TestSystemPromptAssembly:
    """Tests for the system_prompt property composition."""

    def test_includes_base_prompt(self):
        """The base prompt is always present."""
        agent = DrafterAgent(model="m", system_prompt="BASE_TEXT")
        assert "BASE_TEXT" in agent.system_prompt

    def test_includes_protocol_by_default(self):
        """The Council Protocol is appended by default."""
        agent = DrafterAgent(model="m", system_prompt="BASE_TEXT")
        assert COUNCIL_PROTOCOL in agent.system_prompt

    def test_includes_role_prompt(self):
        """The subclass ROLE_PROMPT is appended."""
        agent = DrafterAgent(model="m", system_prompt="BASE_TEXT")
        assert "Your Role: Drafter" in agent.system_prompt

    def test_excludes_protocol_when_disabled(self):
        """include_protocol=False omits the Council Protocol."""
        agent = DrafterAgent(model="m", system_prompt="BASE_TEXT", include_protocol=False)
        assert COUNCIL_PROTOCOL not in agent.system_prompt
        assert "BASE_TEXT" in agent.system_prompt

    def test_order_base_then_protocol_then_role(self):
        """Sections appear in base -> protocol -> role order."""
        agent = DrafterAgent(model="m", system_prompt="BASE_TEXT")
        prompt = agent.system_prompt
        assert prompt.index("BASE_TEXT") < prompt.index(COUNCIL_PROTOCOL.strip())
        assert prompt.index(COUNCIL_PROTOCOL.strip()) < prompt.index("Your Role: Drafter")


class TestDetectPass:
    """Tests for the _detect_pass helper."""

    @pytest.mark.parametrize(
        "content",
        ["PASS", "pass", "  pass  ", "Pass", "**PASS**", "**PASS** with note"],
    )
    def test_detects_pass_variants(self, content: str):
        """Recognized PASS forms are detected (case/whitespace-insensitive)."""
        agent = DrafterAgent(model="m", system_prompt="p")
        assert agent._detect_pass(content) is True

    @pytest.mark.parametrize(
        "content",
        ["I pass on this", "no pass here", "passing the test", "real content"],
    )
    def test_non_pass_content(self, content: str):
        """Non-PASS content is not flagged."""
        agent = DrafterAgent(model="m", system_prompt="p")
        assert agent._detect_pass(content) is False


class TestAddToolsFromRegistry:
    """Tests for loading tools from the registry by role."""

    def test_add_tools_uses_role_name_by_default(self, monkeypatch):
        """Without an explicit role, the agent's ROLE_NAME is used."""
        from llm_council.registry import base_agent as ba_module

        captured: dict[str, str] = {}
        drafter_tool = Tool(name="drafter_tool", description="d", roles=["drafter"])

        class FakeRegistry:
            def ensure_loaded(self):
                pass

            def get_tools_for_role(self, role: str) -> list[Tool]:
                captured["role"] = role
                return [drafter_tool]

        monkeypatch.setattr(ba_module, "get_tool_registry", lambda: FakeRegistry())

        agent = DrafterAgent(model="m", system_prompt="p")
        agent.add_tools_from_registry()

        assert captured["role"] == "drafter"
        assert drafter_tool in agent.tools

    def test_add_tools_with_explicit_role(self, monkeypatch):
        """An explicit role overrides ROLE_NAME and extends existing tools."""
        from llm_council.registry import base_agent as ba_module

        captured: dict[str, str] = {}
        extra_tool = Tool(name="extra", description="d")
        existing_tool = Tool(name="existing", description="d")

        class FakeRegistry:
            def ensure_loaded(self):
                pass

            def get_tools_for_role(self, role: str) -> list[Tool]:
                captured["role"] = role
                return [extra_tool]

        monkeypatch.setattr(ba_module, "get_tool_registry", lambda: FakeRegistry())

        agent = DrafterAgent(model="m", system_prompt="p", tools=[existing_tool])
        agent.add_tools_from_registry(role="critic")

        assert captured["role"] == "critic"
        assert agent.tools == [existing_tool, extra_tool]


class TestConcreteAgentRoles:
    """Tests for the role wiring of the concrete agent subclasses."""

    @pytest.mark.parametrize(
        ("agent_cls", "role_name"),
        [
            (DrafterAgent, "drafter"),
            (CriticAgent, "critic"),
            (SynthesizerAgent, "synthesizer"),
            (PlannerAgent, "planner"),
            (ResearcherAgent, "researcher"),
        ],
    )
    def test_role_name_set(self, agent_cls, role_name):
        """Each concrete agent declares the expected ROLE_NAME."""
        assert role_name == agent_cls.ROLE_NAME

    @pytest.mark.parametrize(
        "agent_cls",
        [DrafterAgent, CriticAgent, SynthesizerAgent, PlannerAgent, ResearcherAgent],
    )
    def test_role_prompt_non_empty(self, agent_cls):
        """Each concrete agent provides a non-empty ROLE_PROMPT."""
        assert agent_cls.ROLE_PROMPT.strip() != ""


class TestConcreteAgentRun:
    """Tests for the (currently placeholder) run() implementations.

    NOTE: run() returns documented placeholder content today. These tests pin
    the real, existing contract (an AgentResult with role metadata), not any
    future model-backed behavior.
    """

    async def test_drafter_run_returns_result_with_role_metadata(self):
        """Drafter.run returns an AgentResult tagged with its role and mode."""
        agent = DrafterAgent(model="m", system_prompt="p")
        result = await agent.run(AgentContext(task="t", mode="impl"))
        assert isinstance(result, AgentResult)
        assert result.passed is False
        assert result.metadata["role"] == "drafter"
        assert result.metadata["mode"] == "impl"

    async def test_critic_run_role_metadata(self):
        """Critic.run reports the critic role in metadata."""
        agent = CriticAgent(model="m", system_prompt="p")
        result = await agent.run(AgentContext(task="t", mode="review"))
        assert result.metadata["role"] == "critic"
        assert result.metadata["mode"] == "review"

    async def test_synthesizer_run_role_metadata(self):
        """Synthesizer.run reports the synthesizer role in metadata."""
        agent = SynthesizerAgent(model="m", system_prompt="p")
        result = await agent.run(AgentContext(task="t"))
        assert result.metadata["role"] == "synthesizer"

    async def test_planner_run_role_metadata(self):
        """Planner.run reports the planner role and forwards mode."""
        agent = PlannerAgent(model="m", system_prompt="p")
        result = await agent.run(AgentContext(task="t", mode="plan"))
        assert result.metadata["role"] == "planner"
        assert result.metadata["mode"] == "plan"

    async def test_researcher_run_role_metadata(self):
        """Researcher.run reports the researcher role in metadata."""
        agent = ResearcherAgent(model="m", system_prompt="p")
        result = await agent.run(AgentContext(task="t"))
        assert result.metadata["role"] == "researcher"


class TestCreateAgentFactory:
    """Tests for the create_agent factory function."""

    @pytest.mark.parametrize(
        ("role", "expected_cls"),
        [
            ("drafter", DrafterAgent),
            ("critic", CriticAgent),
            ("synthesizer", SynthesizerAgent),
            ("planner", PlannerAgent),
            ("researcher", ResearcherAgent),
        ],
    )
    def test_creates_expected_subclass(self, role, expected_cls):
        """A known role maps to its concrete agent subclass."""
        agent = create_agent(role=role, model="m")
        assert isinstance(agent, expected_cls)

    def test_unknown_role_falls_back_to_base_agent(self):
        """An unknown role falls back to the BaseAgent class.

        BaseAgent is abstract, so instantiation raises TypeError. This pins the
        current factory behavior (no special-casing of unknown roles).
        """
        with pytest.raises(TypeError):
            create_agent(role="nonexistent", model="m")

    def test_passes_through_model_and_tools(self):
        """The factory forwards model and tools to the agent."""
        tool = Tool(name="x", description="d")
        agent = create_agent(role="drafter", model="my-model", tools=[tool])
        assert agent.model == "my-model"
        assert agent.tools == [tool]

    def test_agent_classes_mapping_complete(self):
        """The AGENT_CLASSES registry covers all five concrete roles."""
        assert set(AGENT_CLASSES) == {
            "drafter",
            "critic",
            "synthesizer",
            "planner",
            "researcher",
        }
