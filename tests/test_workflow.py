"""
Test suite for UDA-Hub.

Run from the solution/ directory:
    pytest tests/ -v

Prerequisites:
    - OPENAI_API_KEY must be set in .env or environment
    - Run both setup notebooks first so the databases exist
"""

import os
import sys
import json
import uuid
import logging
import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

# Ensure imports resolve from solution/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# ── helpers ──────────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UDAHUB_DB = os.path.join(_ROOT, "data", "core", "udahub.db")
_CULTPASS_DB = os.path.join(_ROOT, "data", "external", "cultpass.db")


def _db_exists(path: str) -> bool:
    return os.path.exists(path)


def _knowledge_count() -> int:
    if not _db_exists(_UDAHUB_DB):
        return 0
    conn = sqlite3.connect(_UDAHUB_DB)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM knowledge")
    n = c.fetchone()[0]
    conn.close()
    return n


# ── Data / DB tests ──────────────────────────────────────────────────────────

class TestDatabaseSetup:
    def test_udahub_db_exists(self):
        assert _db_exists(_UDAHUB_DB), (
            "udahub.db not found — run 02_core_db_setup.ipynb first"
        )

    def test_cultpass_db_exists(self):
        assert _db_exists(_CULTPASS_DB), (
            "cultpass.db not found — run 01_external_db_setup.ipynb first"
        )

    def test_knowledge_base_has_minimum_articles(self):
        count = _knowledge_count()
        assert count >= 14, f"Expected ≥14 articles, found {count}"

    def test_articles_jsonl_has_14_entries(self):
        path = os.path.join(_ROOT, "data", "external", "cultpass_articles.jsonl")
        with open(path, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) >= 14, f"Expected ≥14 lines, found {len(lines)}"

    def test_articles_have_required_fields(self):
        path = os.path.join(_ROOT, "data", "external", "cultpass_articles.jsonl")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    article = json.loads(line)
                    assert "title" in article
                    assert "content" in article
                    assert "tags" in article

    def test_udahub_db_has_required_tables(self):
        if not _db_exists(_UDAHUB_DB):
            pytest.skip("udahub.db not found")
        conn = sqlite3.connect(_UDAHUB_DB)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in c.fetchall()}
        conn.close()
        required = {"accounts", "users", "tickets", "ticket_metadata", "ticket_messages", "knowledge"}
        assert required.issubset(tables), f"Missing tables: {required - tables}"

    def test_cultpass_db_has_users(self):
        if not _db_exists(_CULTPASS_DB):
            pytest.skip("cultpass.db not found")
        conn = sqlite3.connect(_CULTPASS_DB)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users")
        assert c.fetchone()[0] > 0
        conn.close()


# ── Tool tests ────────────────────────────────────────────────────────────────

class TestKnowledgeTools:
    def test_search_login_returns_login_article(self):
        """Verify the login query retrieves the login-specific article, not an unrelated one."""
        if not _db_exists(_UDAHUB_DB):
            pytest.skip("udahub.db not found")
        from agentic.tools.knowledge_tools import search_knowledge_base
        result = search_knowledge_base.invoke({"query": "login password reset"})
        assert isinstance(result, str)
        assert len(result) > 10
        # The login article must appear — not just any article
        assert "Login" in result or "Password" in result or "login" in result.lower()
        # The escalation marker must NOT appear — this query has a clear match
        from agentic.tools.knowledge_tools import ESCALATION_MARKER
        assert ESCALATION_MARKER not in result

    def test_search_subscription_returns_subscription_article(self):
        """Verify cancel/subscription query retrieves the subscription management article."""
        if not _db_exists(_UDAHUB_DB):
            pytest.skip("udahub.db not found")
        from agentic.tools.knowledge_tools import search_knowledge_base, ESCALATION_MARKER
        result = search_knowledge_base.invoke({"query": "cancel subscription billing"})
        assert isinstance(result, str)
        assert "subscription" in result.lower() or "cancel" in result.lower() or "Subscription" in result
        assert ESCALATION_MARKER not in result

    def test_search_reservation_returns_booking_article(self):
        """Verify reservation query retrieves the booking article."""
        if not _db_exists(_UDAHUB_DB):
            pytest.skip("udahub.db not found")
        from agentic.tools.knowledge_tools import search_knowledge_base, ESCALATION_MARKER
        result = search_knowledge_base.invoke({"query": "reserve spot event booking"})
        assert isinstance(result, str)
        assert "reserv" in result.lower() or "booking" in result.lower() or "event" in result.lower()
        assert ESCALATION_MARKER not in result

    def test_search_no_match_returns_escalation_marker(self):
        """Completely unrelated query must return the escalation marker, not a fabricated answer."""
        if not _db_exists(_UDAHUB_DB):
            pytest.skip("udahub.db not found")
        from agentic.tools.knowledge_tools import search_knowledge_base, ESCALATION_MARKER
        result = search_knowledge_base.invoke({"query": "xyzzy frobnicator quux"})
        assert ESCALATION_MARKER in result

    def test_search_weak_query_returns_escalation_marker(self):
        """Query that produces only very low scores must not return articles."""
        if not _db_exists(_UDAHUB_DB):
            pytest.skip("udahub.db not found")
        from agentic.tools.knowledge_tools import search_knowledge_base, ESCALATION_MARKER, MIN_RELEVANCE_SCORE
        # Single-word query that might weakly match content but shouldn't pass the title/tag threshold
        result = search_knowledge_base.invoke({"query": "xyz123notaword"})
        assert ESCALATION_MARKER in result

    def test_min_relevance_score_constant_is_set(self):
        """MIN_RELEVANCE_SCORE must be defined and positive."""
        from agentic.tools.knowledge_tools import MIN_RELEVANCE_SCORE
        assert MIN_RELEVANCE_SCORE > 0.0


class TestAccountTools:
    def test_get_user_account_by_id(self):
        if not _db_exists(_CULTPASS_DB):
            pytest.skip("cultpass.db not found")
        from agentic.tools.account_tools import get_user_account
        # f556c0 = Bob Stone (not blocked)
        result = get_user_account.invoke({"user_identifier": "f556c0"})
        assert isinstance(result, str)
        assert "Bob Stone" in result or "f556c0" in result

    def test_get_user_account_not_found(self):
        if not _db_exists(_CULTPASS_DB):
            pytest.skip("cultpass.db not found")
        from agentic.tools.account_tools import get_user_account
        result = get_user_account.invoke({"user_identifier": "nonexistent-user-xyz"})
        assert "No user found" in result

    def test_get_user_reservations(self):
        if not _db_exists(_CULTPASS_DB):
            pytest.skip("cultpass.db not found")
        from agentic.tools.account_tools import get_user_reservations
        result = get_user_reservations.invoke({"user_identifier": "f556c0"})
        assert isinstance(result, str)

    def test_manage_subscription_invalid_action(self):
        if not _db_exists(_CULTPASS_DB):
            pytest.skip("cultpass.db not found")
        from agentic.tools.account_tools import manage_subscription
        result = manage_subscription.invoke({"user_identifier": "f556c0", "action": "delete"})
        assert "Invalid action" in result


# ── Agent unit tests (mocked LLM) ────────────────────────────────────────────

class TestClassifierAgent:
    def test_classifier_categorises_billing(self):
        from agentic.agents.classifier import classifier_node

        mock_output = MagicMock()
        mock_output.category = "billing"
        mock_output.urgency = "medium"
        mock_output.complexity = "simple"

        with patch("agentic.agents.classifier._llm") as mock_llm:
            mock_llm.invoke.return_value = mock_output
            state = {"messages": [HumanMessage(content="I was charged twice this month")]}
            result = classifier_node(state)

        assert result["classification"] == "billing"
        assert result["urgency"] == "medium"
        assert result["complexity"] == "simple"

    def test_classifier_categorises_technical(self):
        from agentic.agents.classifier import classifier_node

        mock_output = MagicMock()
        mock_output.category = "technical"
        mock_output.urgency = "high"
        mock_output.complexity = "simple"

        with patch("agentic.agents.classifier._llm") as mock_llm:
            mock_llm.invoke.return_value = mock_output
            state = {"messages": [HumanMessage(content="The app keeps crashing")]}
            result = classifier_node(state)

        assert result["classification"] == "technical"
        assert "complexity" in result

    def test_classifier_includes_ticket_metadata_in_prompt(self):
        """Classifier passes ticket_priority and ticket_age_hours to the LLM."""
        from agentic.agents.classifier import classifier_node

        mock_output = MagicMock()
        mock_output.category = "account"
        mock_output.urgency = "high"
        mock_output.complexity = "complex"

        with patch("agentic.agents.classifier._llm") as mock_llm:
            mock_llm.invoke.return_value = mock_output
            state = {
                "messages": [HumanMessage(content="I think my account was hacked")],
                "ticket_priority": "high",
                "ticket_age_hours": 48,
            }
            result = classifier_node(state)
            # Verify the LLM was called with metadata in the content
            call_args = mock_llm.invoke.call_args[0][0]
            human_msg = call_args[-1]
            assert "ticket_priority" in human_msg.content or "48" in human_msg.content

        assert result["classification"] == "account"
        assert result["urgency"] == "high"
        assert result["complexity"] == "complex"


class TestRoutingDecisions:
    """Verify that route_after_classifier produces the correct routing decision
    based on classification, urgency, complexity and ticket metadata."""

    def test_high_urgency_account_routes_to_direct_escalate(self):
        from agentic.workflow import route_after_classifier
        state = {"classification": "account", "urgency": "high", "complexity": "simple", "ticket_priority": "normal"}
        assert route_after_classifier(state) == "direct_escalate"

    def test_high_urgency_billing_routes_to_direct_escalate(self):
        from agentic.workflow import route_after_classifier
        state = {"classification": "billing", "urgency": "high", "complexity": "simple", "ticket_priority": "normal"}
        assert route_after_classifier(state) == "direct_escalate"

    def test_critical_ticket_priority_routes_to_direct_escalate(self):
        from agentic.workflow import route_after_classifier
        # Even low-urgency tickets with critical priority should escalate immediately
        state = {"classification": "general", "urgency": "low", "complexity": "simple", "ticket_priority": "critical"}
        assert route_after_classifier(state) == "direct_escalate"

    def test_high_urgency_complex_routes_to_direct_escalate(self):
        from agentic.workflow import route_after_classifier
        # High urgency + complex (any category) → escalate
        state = {"classification": "technical", "urgency": "high", "complexity": "complex", "ticket_priority": "normal"}
        assert route_after_classifier(state) == "direct_escalate"

    def test_low_urgency_technical_routes_to_resolve(self):
        from agentic.workflow import route_after_classifier
        state = {"classification": "technical", "urgency": "low", "complexity": "simple", "ticket_priority": "normal"}
        assert route_after_classifier(state) == "resolve"

    def test_medium_urgency_billing_routes_to_resolve(self):
        from agentic.workflow import route_after_classifier
        state = {"classification": "billing", "urgency": "medium", "complexity": "simple", "ticket_priority": "normal"}
        assert route_after_classifier(state) == "resolve"

    def test_high_urgency_technical_simple_routes_to_resolve(self):
        from agentic.workflow import route_after_classifier
        # High urgency technical but simple (e.g. app crash) → still try to resolve
        state = {"classification": "technical", "urgency": "high", "complexity": "simple", "ticket_priority": "normal"}
        assert route_after_classifier(state) == "resolve"

    def test_booking_always_routes_to_resolve(self):
        from agentic.workflow import route_after_classifier
        state = {"classification": "booking", "urgency": "medium", "complexity": "simple", "ticket_priority": "normal"}
        assert route_after_classifier(state) == "resolve"


class TestSupervisorAgent:
    def test_supervisor_routes_support_question(self):
        from agentic.agents.supervisor import supervisor_node

        mock_decision = MagicMock()
        mock_decision.action = "classify"
        mock_decision.direct_response = ""

        with patch("agentic.agents.supervisor._structured") as mock_llm:
            mock_llm.invoke.return_value = mock_decision
            state = {"messages": [HumanMessage(content="My subscription expired but I was still charged")]}
            result = supervisor_node(state)

        assert result["next_action"] == "classify"

    def test_supervisor_handles_greeting(self):
        from agentic.agents.supervisor import supervisor_node

        mock_decision = MagicMock()
        mock_decision.action = "end"
        mock_decision.direct_response = "Hello! How can I help you today?"

        with patch("agentic.agents.supervisor._structured") as mock_llm:
            mock_llm.invoke.return_value = mock_decision
            state = {"messages": [HumanMessage(content="Hi there!")]}
            result = supervisor_node(state)

        assert result["next_action"] == "end"
        assert len(result.get("messages", [])) == 1


# ── Log-capture helper ────────────────────────────────────────────────────────

class _LogCapture(logging.Handler):
    """Attaches to the root logger, collects all records that carry 'agent' extra."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def structured(self) -> list[dict]:
        """Return only records that have agent-level structure, as parsed dicts."""
        from agentic.logging_config import JsonFormatter
        fmt = JsonFormatter()
        out = []
        for r in self.records:
            if hasattr(r, "agent"):
                try:
                    out.append(json.loads(fmt.format(r)))
                except Exception:
                    pass
        return out

    def events_by(self, agent: str = None, event: str = None) -> list[dict]:
        return [
            r for r in self.structured()
            if (agent is None or r.get("agent") == agent)
            and (event is None or r.get("event") == event)
        ]

    @staticmethod
    def install() -> "_LogCapture":
        cap = _LogCapture()
        logging.getLogger().addHandler(cap)
        return cap

    def uninstall(self) -> None:
        logging.getLogger().removeHandler(self)


# ── Integration / end-to-end tests ───────────────────────────────────────────

@pytest.mark.integration
class TestEndToEnd:
    """Full workflow tests — require OPENAI_API_KEY and populated databases.

    Each test asserts on the complete agent state (next_action, retrieval_quality,
    confidence, classification) as well as the final message content, so failures
    show WHICH part of the pipeline produced the wrong output.
    """

    def setup_method(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        if not _db_exists(_UDAHUB_DB) or not _db_exists(_CULTPASS_DB):
            pytest.skip("Databases not set up")

    def _invoke(self, message: str, thread_id: str = None) -> dict:
        from agentic.workflow import orchestrator
        tid = thread_id or f"test-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": tid}}
        return orchestrator.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

    # ── Resolution scenarios ──────────────────────────────────────────────────

    def test_login_issue_resolves_with_kb_retrieval(self):
        """Password reset query: low-urgency technical issue — must reach Resolver, not Escalator."""
        result = self._invoke("I forgot my CultPass password. What are the steps to reset it?")

        # State assertions
        assert result.get("next_action") == "end", (
            f"Expected resolution, got next_action={result.get('next_action')!r}"
        )
        assert result.get("retrieval_quality") in ("high", "medium"), (
            f"Expected KB hit, got retrieval_quality={result.get('retrieval_quality')!r}"
        )
        assert result.get("confidence", 0) >= 0.4, (
            f"Low confidence on a clear KB question: {result.get('confidence')}"
        )
        assert result.get("classification") in ("technical", "account"), (
            f"Expected technical or account classification, got {result.get('classification')!r}"
        )

        # Response assertions
        last = result["messages"][-1]
        assert isinstance(last, AIMessage)
        content_lower = last.content.lower()
        assert len(last.content) > 50
        assert any(kw in content_lower for kw in ("login", "password", "reset", "forgot", "account")), (
            f"Response doesn't mention login/password: {last.content[:200]}"
        )

    def test_subscription_cancel_query_resolves_with_tool(self):
        """Subscription cancel query: must be resolved and include subscription guidance."""
        result = self._invoke("How do I cancel my CultPass subscription?")

        assert result.get("next_action") == "end", (
            f"Expected resolution, got next_action={result.get('next_action')!r}"
        )
        assert result.get("confidence", 0) >= 0.4

        last = result["messages"][-1]
        assert isinstance(last, AIMessage)
        content_lower = last.content.lower()
        assert any(kw in content_lower for kw in ("subscription", "cancel", "account", "pause")), (
            f"Response doesn't mention subscription: {last.content[:200]}"
        )

    def test_greeting_handled_directly_without_classification(self):
        """Greeting: Supervisor must handle it directly — no KB retrieval or classification."""
        result = self._invoke("Hello!")
        assert isinstance(result["messages"][-1], AIMessage)
        # Supervisor handles greetings directly — classification should not be set
        # (or defaults to empty string)
        assert result.get("classification", "") in ("", "general", None), (
            "Greeting should not trigger full classification pipeline"
        )

    # ── Escalation scenarios ──────────────────────────────────────────────────

    def test_unknown_topic_explicitly_escalates(self):
        """Completely off-topic query: must trigger Escalator, not just return a response."""
        result = self._invoke(
            "My blockchain NFT wallet integration with CultPass is broken and I need a refund for 50 transactions"
        )

        # Must have escalated — next_action or UDA ref in message
        final_msg = result["messages"][-1].content
        escalated_via_state = result.get("next_action") == "escalate"
        escalated_via_msg = "UDA-" in final_msg

        assert escalated_via_state or escalated_via_msg, (
            f"Expected escalation. next_action={result.get('next_action')!r}, "
            f"message={final_msg[:200]!r}"
        )
        assert len(final_msg) > 20

    def test_high_urgency_account_issue_escalates(self):
        """Hacked account (high urgency + account category): must direct-escalate."""
        result = self._invoke(
            "I think my account has been hacked — someone changed my email and I can't access it"
        )

        final_msg = result["messages"][-1].content
        assert "UDA-" in final_msg, (
            f"Escalation ticket reference missing from response: {final_msg[:300]!r}"
        )
        # Escalator always terminates — next_action is not set on its return
        # but the final message must contain an escalation reference
        content_lower = final_msg.lower()
        assert any(kw in content_lower for kw in ("escalat", "team", "specialist", "24 hour", "follow")), (
            f"Response does not sound like an escalation: {final_msg[:300]!r}"
        )

    def test_deterministic_escalation_on_no_kb_match(self):
        """Query with no KB-keyword overlap must hit the Escalator (low-confidence or marker)."""
        result = self._invoke(
            "I need help with my CultPass smart TV screen mirroring and Bluetooth speaker pairing"
        )

        final_msg = result["messages"][-1].content
        assert "UDA-" in final_msg or result.get("next_action") == "escalate", (
            f"Expected deterministic escalation. Message: {final_msg[:200]!r}"
        )

    # ── Multi-turn conversation ───────────────────────────────────────────────

    def test_multi_turn_conversation_accumulates_context(self):
        thread_id = f"test-multi-{uuid.uuid4().hex[:8]}"
        from agentic.workflow import orchestrator
        config = {"configurable": {"thread_id": thread_id}}

        r1 = orchestrator.invoke(
            {"messages": [HumanMessage(content="How do I reset my password?")]},
            config=config,
        )
        assert isinstance(r1["messages"][-1], AIMessage)
        assert r1.get("next_action") == "end"

        r2 = orchestrator.invoke(
            {"messages": [HumanMessage(content="What if I still don't receive the email?")]},
            config=config,
        )
        assert isinstance(r2["messages"][-1], AIMessage)
        # Thread must have accumulated messages from both turns
        assert len(r2["messages"]) > len(r1["messages"]), (
            "Thread history did not grow — short-term memory may be broken"
        )


# ── Complete ticket workflow + structured log capture ─────────────────────────

@pytest.mark.integration
class TestCompleteTicketWorkflow:
    """Demonstrate a complete ticket processing workflow with structured log capture.

    Each test runs the real orchestrator and verifies:
    1. The final agent state (classification, routing, resolution / escalation)
    2. That the expected structured log events were actually emitted
    """

    def setup_method(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        if not _db_exists(_UDAHUB_DB) or not _db_exists(_CULTPASS_DB):
            pytest.skip("Databases not set up")

    def _run_with_logs(self, message: str) -> tuple[dict, _LogCapture]:
        from agentic.workflow import orchestrator
        tid = f"test-wf-{uuid.uuid4().hex[:8]}"
        cap = _LogCapture.install()
        try:
            result = orchestrator.invoke(
                {"messages": [HumanMessage(content=message)]},
                {"configurable": {"thread_id": tid}},
            )
        finally:
            cap.uninstall()
        return result, cap

    def test_resolution_workflow_complete_pipeline(self):
        """
        Complete resolution pipeline:
          submit → Supervisor routes → Classifier labels → Resolver retrieves KB
          → generates answer → state: next_action=end, retrieval_quality=high/medium
          → structured logs show: supervisor route, classifier classify, resolver tool_call + resolve
        """
        result, cap = self._run_with_logs("I forgot my CultPass password. What are the steps to reset it?")

        # ── State assertions ──
        assert result.get("next_action") == "end", (
            f"Expected resolved ticket. next_action={result.get('next_action')!r}"
        )
        assert result.get("retrieval_quality") in ("high", "medium"), (
            f"Expected KB retrieval, got: {result.get('retrieval_quality')!r}"
        )
        assert result.get("confidence", 0) >= 0.4
        assert isinstance(result["messages"][-1], AIMessage)
        assert len(result["messages"][-1].content) > 50

        # ── Log assertions ──
        structured = cap.structured()
        assert structured, "No structured log records were emitted during workflow"

        agents_logged = {r["agent"] for r in structured}
        assert "supervisor" in agents_logged, f"Supervisor log missing. Agents logged: {agents_logged}"
        assert "resolver" in agents_logged, f"Resolver log missing. Agents logged: {agents_logged}"

        supervisor_routes = cap.events_by(agent="supervisor", event="route")
        assert supervisor_routes, "No supervisor routing event logged"
        assert supervisor_routes[0]["status"] == "success"

        resolver_resolves = cap.events_by(agent="resolver", event="resolve")
        assert resolver_resolves, "No resolver resolve event logged"
        assert resolver_resolves[0]["status"] == "success"

        tool_calls = cap.events_by(agent="resolver", event="tool_call")
        assert tool_calls, "No tool_call events logged — Resolver did not call any tools"
        tool_names = [t.get("tool_name") for t in tool_calls]
        assert "search_knowledge_base" in tool_names, (
            f"search_knowledge_base not called. Tools used: {tool_names}"
        )

        # Every structured log must have the required fields
        for record in structured:
            assert "timestamp" in record, f"Missing timestamp: {record}"
            assert "level" in record, f"Missing level: {record}"
            assert "agent" in record, f"Missing agent: {record}"
            assert "event" in record, f"Missing event: {record}"
            assert "status" in record, f"Missing status: {record}"

    def test_escalation_workflow_complete_pipeline(self):
        """
        Complete escalation pipeline:
          submit → Supervisor routes → Classifier labels high-urgency account issue
          → route_after_classifier direct-escalates → Escalator generates UDA ref
          → structured logs show: supervisor route, escalator escalate event
        """
        result, cap = self._run_with_logs(
            "Someone has taken over my account and I cannot get back in — this is urgent"
        )

        # ── State assertions ──
        final_msg = result["messages"][-1].content
        assert "UDA-" in final_msg, (
            f"Escalation UDA reference not found in response: {final_msg[:300]!r}"
        )
        content_lower = final_msg.lower()
        assert any(kw in content_lower for kw in ("escalat", "team", "specialist", "24 hour", "follow")), (
            f"Response does not look like an escalation message: {final_msg[:300]!r}"
        )

        # ── Log assertions ──
        structured = cap.structured()
        assert structured, "No structured log records emitted"

        supervisor_routes = cap.events_by(agent="supervisor", event="route")
        assert supervisor_routes, "Supervisor routing event not logged"

        escalator_events = cap.events_by(agent="escalator", event="escalate")
        assert escalator_events, (
            "Escalator event not logged — escalation path may not have been taken. "
            f"Agents in logs: {[r.get('agent') for r in structured]}"
        )
        assert escalator_events[0]["status"] == "escalated"

    def test_tool_integration_subscription_workflow(self):
        """
        Subscription query with user email: Resolver must reach the tool-calling phase
        and invoke at least one account tool (get_user_account or manage_subscription).
        search_knowledge_base is optional — the LLM may skip it when intent is unambiguous.
        """
        result, cap = self._run_with_logs(
            "My email is bob.stone@granite.com. Could you walk me through how to cancel my subscription?"
        )

        # ── State assertions ──
        assert result.get("next_action") == "end", (
            f"Expected resolution, got next_action={result.get('next_action')!r}"
        )
        last = result["messages"][-1]
        assert isinstance(last, AIMessage)
        assert len(last.content) > 30

        # ── Log assertions ──
        tool_calls = cap.events_by(agent="resolver", event="tool_call")
        assert tool_calls, "Resolver made no tool calls at all"
        tool_names = [t.get("tool_name", "") for t in tool_calls]

        # Account lookup must occur when an email is present in the message
        account_tools_called = any(
            t in tool_names for t in ("get_user_account", "manage_subscription")
        )
        assert account_tools_called, (
            f"No account tool called despite email in message. Tool calls: {tool_names}"
        )

        # Resolver must emit a successful resolve event
        resolve_events = cap.events_by(agent="resolver", event="resolve")
        assert resolve_events, "No resolver resolve event logged"
        assert resolve_events[0]["status"] == "success"

    def test_deterministic_escalation_logs_kb_failure(self):
        """
        Off-topic query: Resolver must log an escalation event — either via ESCALATION_MARKER
        detection (event='escalate') or via low-confidence LLM decision (event='resolve',
        status='escalated'). Both paths are valid; we verify the escalation audit trail exists.
        """
        result, cap = self._run_with_logs(
            "I need help configuring my CultPass interplanetary travel portal"
        )

        # ── Escalation must have occurred ──
        final_msg = result["messages"][-1].content
        assert "UDA-" in final_msg or result.get("next_action") == "escalate"

        # ── Log must show a resolver escalation event (either path) ──
        # Phase 2: ESCALATION_MARKER detected → event="escalate"
        # Phase 3: LLM low-confidence → event="resolve", status="escalated"
        resolver_escalated = (
            cap.events_by(agent="resolver", event="escalate")
            or [r for r in cap.events_by(agent="resolver", event="resolve")
                if r.get("status") == "escalated"]
        )
        assert resolver_escalated, (
            "No resolver escalation event logged (neither event='escalate' nor "
            "event='resolve' with status='escalated'). "
            f"Resolver events: {cap.events_by(agent='resolver')}"
        )
        assert resolver_escalated[0]["status"] == "escalated"


# ── Repository layer tests (DB-state verification) ────────────────────────────

class TestRepositoryLayer:
    """Verify that the repository functions read and write the database correctly.

    Uses a pytest fixture to restore the original subscription status after each
    test that modifies it, so tests remain independent of execution order.
    """

    @pytest.fixture(autouse=True)
    def _restore_bob_subscription(self):
        """Snapshot Bob Stone's subscription status and restore it after the test."""
        if not _db_exists(_CULTPASS_DB):
            yield
            return
        conn = sqlite3.connect(_CULTPASS_DB)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT status FROM subscriptions WHERE user_id = 'f556c0'"
        ).fetchone()
        original_status = row["status"] if row else None
        conn.close()
        yield
        # Restore
        if original_status is not None:
            conn = sqlite3.connect(_CULTPASS_DB)
            conn.execute(
                "UPDATE subscriptions SET status = ? WHERE user_id = 'f556c0'",
                (original_status,),
            )
            conn.commit()
            conn.close()

    @pytest.fixture()
    def restore_frank_subscription(self):
        """Snapshot Frank Ocean's subscription status and restore it after the test."""
        if not _db_exists(_CULTPASS_DB):
            yield
            return
        conn = sqlite3.connect(_CULTPASS_DB)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT status FROM subscriptions WHERE user_id = 'e6376d'"
        ).fetchone()
        original_status = row["status"] if row else None
        conn.close()
        yield
        if original_status is not None:
            conn = sqlite3.connect(_CULTPASS_DB)
            conn.execute(
                "UPDATE subscriptions SET status = ? WHERE user_id = 'e6376d'",
                (original_status,),
            )
            conn.commit()
            conn.close()

    def test_get_user_returns_correct_struct(self):
        if not _db_exists(_CULTPASS_DB):
            pytest.skip("cultpass.db not found")
        from data.repositories import get_user
        account = get_user(_CULTPASS_DB, "f556c0")
        assert account is not None
        assert account.user_id == "f556c0"
        assert account.full_name == "Bob Stone"
        assert account.is_blocked is False
        assert account.subscription_status is not None

    def test_get_user_by_email_returns_account(self):
        if not _db_exists(_CULTPASS_DB):
            pytest.skip("cultpass.db not found")
        from data.repositories import get_user, get_user_id_by_identifier
        # Lookup Bob's email first via user_id
        account = get_user(_CULTPASS_DB, "f556c0")
        assert account is not None
        # Now look up by email — should return the same user
        account_by_email = get_user(_CULTPASS_DB, account.email)
        assert account_by_email is not None
        assert account_by_email.user_id == "f556c0"

    def test_get_user_nonexistent_returns_none(self):
        if not _db_exists(_CULTPASS_DB):
            pytest.skip("cultpass.db not found")
        from data.repositories import get_user
        assert get_user(_CULTPASS_DB, "no-such-user") is None

    def test_get_reservations_returns_list(self):
        if not _db_exists(_CULTPASS_DB):
            pytest.skip("cultpass.db not found")
        from data.repositories import get_reservations
        # Charlie Green has reservations (seeded in setup notebook)
        reservations = get_reservations(_CULTPASS_DB, "a1b2c3")
        assert isinstance(reservations, list)

    def test_manage_subscription_cancel_updates_db(self):
        """Calling manage_subscription with action='cancel' must persist 'cancelled' in the DB."""
        if not _db_exists(_CULTPASS_DB):
            pytest.skip("cultpass.db not found")
        from agentic.tools.account_tools import manage_subscription

        result = manage_subscription.invoke({"user_identifier": "f556c0", "action": "cancel"})
        assert "cancel" in result.lower() or "cancelled" in result.lower()

        # Verify the change landed in the database
        conn = sqlite3.connect(_CULTPASS_DB)
        row = conn.execute(
            "SELECT status FROM subscriptions WHERE user_id = 'f556c0'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "cancelled", f"Expected 'cancelled', got '{row[0]}'"

    @pytest.mark.usefixtures("restore_frank_subscription")
    def test_manage_subscription_pause_updates_db(self):
        """Calling manage_subscription with action='pause' must persist 'paused' in the DB."""
        if not _db_exists(_CULTPASS_DB):
            pytest.skip("cultpass.db not found")
        from agentic.tools.account_tools import manage_subscription

        result = manage_subscription.invoke({"user_identifier": "e6376d", "action": "pause"})
        assert "pause" in result.lower() or "paused" in result.lower()

        conn = sqlite3.connect(_CULTPASS_DB)
        row = conn.execute(
            "SELECT status FROM subscriptions WHERE user_id = 'e6376d'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "paused", f"Expected 'paused', got '{row[0]}'"

    def test_update_subscription_no_user_returns_failure(self):
        """update_subscription on a missing user_id must return success=False."""
        if not _db_exists(_CULTPASS_DB):
            pytest.skip("cultpass.db not found")
        from data.repositories import update_subscription
        result = update_subscription(_CULTPASS_DB, "nonexistent-uid", "cancel")
        assert result.success is False
        assert "not found" in result.message.lower() or "no subscription" in result.message.lower()


# ── Agent-to-tool integration tests ──────────────────────────────────────────

class TestResolverToolIntegration:
    """Verify that the Resolver agent actually invokes account tools when needed.

    Both _llm_tools (Phase 1) and _llm_structured (Phase 3) are mocked so no
    real API key is required, and the tool spy does NOT call through to the DB
    so tests leave no side-effects.
    """

    def _make_structured_mock(self, confidence=0.8, needs_escalation=False, response="Done."):
        """Return a mock structured output object."""
        m = MagicMock()
        m.confidence = confidence
        m.needs_escalation = needs_escalation
        m.response = response
        return m

    def test_resolver_calls_manage_subscription_on_cancel_request(self):
        """When _llm_tools emits a cancel tool-call, _run_tools must receive it with action=cancel."""
        from langchain_core.messages import AIMessage as LCAIMessage, ToolMessage
        import agentic.agents.resolver as resolver_module
        from agentic.agents.resolver import resolver_node

        tool_call_msg = LCAIMessage(
            content="",
            tool_calls=[{
                "id": "tc_001",
                "name": "manage_subscription",
                "args": {"user_identifier": "f556c0", "action": "cancel"},
            }],
        )

        recorded: list[dict] = []

        def spy_run_tools(tool_calls):
            recorded.extend(tool_calls)
            # Return a success ToolMessage without touching the DB
            return [ToolMessage(content="Subscription cancelled.", tool_call_id=tc["id"]) for tc in tool_calls]

        state = {
            "messages": [HumanMessage(content="Please cancel subscription for user f556c0")],
            "classification": "billing",
            "urgency": "low",
            "complexity": "simple",
            "ticket_priority": "normal",
            "ticket_age_hours": 0,
            "confidence": 0.0,
            "retrieval_quality": "none",
            "next_action": "resolve",
            "long_term_context": "",
        }

        structured_mock = self._make_structured_mock(response="Subscription cancelled.")
        with patch("agentic.agents.resolver._llm_tools") as mock_llm_tools, \
             patch("agentic.agents.resolver._llm_structured") as mock_llm_struct, \
             patch.object(resolver_module, "_run_tools", side_effect=spy_run_tools):
            mock_llm_tools.invoke.return_value = tool_call_msg
            mock_llm_struct.invoke.return_value = structured_mock
            resolver_node(state)

        assert recorded, "_run_tools was never called — Resolver did not dispatch any tool"
        tool_names = [tc["name"] for tc in recorded]
        assert "manage_subscription" in tool_names, (
            f"Expected manage_subscription in tool calls, got: {tool_names}"
        )
        cancel_call = next(tc for tc in recorded if tc["name"] == "manage_subscription")
        assert cancel_call["args"]["action"] == "cancel"

    def test_resolver_calls_get_user_account_on_account_query(self):
        """When _llm_tools emits a get_user_account tool-call, _run_tools must receive it."""
        from langchain_core.messages import AIMessage as LCAIMessage, ToolMessage
        import agentic.agents.resolver as resolver_module
        from agentic.agents.resolver import resolver_node

        tool_call_msg = LCAIMessage(
            content="",
            tool_calls=[{
                "id": "tc_002",
                "name": "get_user_account",
                "args": {"user_identifier": "f556c0"},
            }],
        )

        recorded: list[dict] = []

        def spy_run_tools(tool_calls):
            recorded.extend(tool_calls)
            return [ToolMessage(content="**User:** Bob Stone\n**Account status:** Active", tool_call_id=tc["id"]) for tc in tool_calls]

        state = {
            "messages": [HumanMessage(content="What is the status of account f556c0?")],
            "classification": "account",
            "urgency": "low",
            "complexity": "simple",
            "ticket_priority": "normal",
            "ticket_age_hours": 0,
            "confidence": 0.0,
            "retrieval_quality": "none",
            "next_action": "resolve",
            "long_term_context": "",
        }

        structured_mock = self._make_structured_mock(response="Account is active.")
        with patch("agentic.agents.resolver._llm_tools") as mock_llm_tools, \
             patch("agentic.agents.resolver._llm_structured") as mock_llm_struct, \
             patch.object(resolver_module, "_run_tools", side_effect=spy_run_tools):
            mock_llm_tools.invoke.return_value = tool_call_msg
            mock_llm_struct.invoke.return_value = structured_mock
            resolver_node(state)

        assert recorded, "_run_tools was never called — Resolver did not dispatch any tool"
        tool_names = [tc["name"] for tc in recorded]
        assert "get_user_account" in tool_names, (
            f"Expected get_user_account in tool calls, got: {tool_names}"
        )


# ── Long-term memory tests ─────────────────────────────────────────────────────

class TestLongTermMemory:
    """Verify that the LTM repository saves and retrieves resolved-issue summaries correctly,
    and that the Resolver persists LTM on successful resolution."""

    @pytest.fixture(autouse=True)
    def _cleanup_test_ltm(self):
        """Remove any LTM rows written by this test class after each test."""
        yield
        if not _db_exists(_UDAHUB_DB):
            return
        conn = sqlite3.connect(_UDAHUB_DB)
        conn.execute(
            "DELETE FROM long_term_memory WHERE thread_id LIKE 'test-ltm-%'"
        )
        conn.commit()
        conn.close()

    def test_save_and_retrieve_memory(self):
        """save_memory → get_customer_memories round-trip for a known customer_id."""
        if not _db_exists(_UDAHUB_DB):
            pytest.skip("udahub.db not found")
        from data.repositories import save_memory, get_customer_memories

        thread_id = "test-ltm-001"
        customer_id = "bob.stone@granite.com"
        summary = "Resolved login issue: advised customer to use Forgot Password link."

        save_memory(_UDAHUB_DB, thread_id, customer_id, "technical", summary)

        memories = get_customer_memories(_UDAHUB_DB, customer_id, limit=5)
        assert any(summary in m for m in memories), (
            f"Saved summary not found in retrieved memories: {memories}"
        )

    def test_memories_are_customer_scoped(self):
        """Memories written for customer A must not appear when retrieving for customer B."""
        if not _db_exists(_UDAHUB_DB):
            pytest.skip("udahub.db not found")
        from data.repositories import save_memory, get_customer_memories

        save_memory(_UDAHUB_DB, "test-ltm-002", "alice@example.com", "billing", "Alice's billing issue resolved.")
        save_memory(_UDAHUB_DB, "test-ltm-003", "bob@example.com", "technical", "Bob's login issue resolved.")

        alice_memories = get_customer_memories(_UDAHUB_DB, "alice@example.com")
        bob_memories = get_customer_memories(_UDAHUB_DB, "bob@example.com")

        assert all("Alice" in m for m in alice_memories), f"Got unexpected memories for Alice: {alice_memories}"
        assert all("Bob" in m for m in bob_memories), f"Got unexpected memories for Bob: {bob_memories}"

    def test_extract_customer_id_finds_email(self):
        """extract_customer_id should extract an email from a message string."""
        from data.repositories import extract_customer_id
        text = "Hi, my email is bob.stone@granite.com and I can't log in"
        assert extract_customer_id(text) == "bob.stone@granite.com"

    def test_extract_customer_id_no_email(self):
        """extract_customer_id returns empty string when no email is present."""
        from data.repositories import extract_customer_id
        assert extract_customer_id("I can't log in, please help") == ""

    def test_resolver_saves_ltm_on_successful_resolution(self):
        """After a successful resolution that used an account tool, a LTM row must be written."""
        if not _db_exists(_UDAHUB_DB):
            pytest.skip("udahub.db not found")

        from langchain_core.messages import AIMessage as LCAIMessage, ToolMessage
        import agentic.agents.resolver as resolver_module
        from agentic.agents.resolver import resolver_node

        tool_call_msg = LCAIMessage(
            content="",
            tool_calls=[{
                "id": "tc_ltm_01",
                "name": "get_user_account",
                "args": {"user_identifier": "ltm-test@example.com"},
            }],
        )

        structured_mock = MagicMock()
        structured_mock.confidence = 0.85
        structured_mock.needs_escalation = False
        structured_mock.response = "Your account is active. No issues found."

        def spy_run_tools(tool_calls):
            return [ToolMessage(content="**User:** Test User\n**Account status:** Active", tool_call_id=tc["id"]) for tc in tool_calls]

        state = {
            "messages": [HumanMessage(content="Check account for ltm-test@example.com")],
            "classification": "account",
            "urgency": "low",
            "complexity": "simple",
            "ticket_priority": "normal",
            "ticket_age_hours": 0,
            "confidence": 0.0,
            "retrieval_quality": "none",
            "next_action": "resolve",
            "long_term_context": "",
        }
        config = {"configurable": {"thread_id": "test-ltm-resolver-001"}}

        with patch("agentic.agents.resolver._llm_tools") as mock_llm_tools, \
             patch("agentic.agents.resolver._llm_structured") as mock_llm_struct, \
             patch.object(resolver_module, "_run_tools", side_effect=spy_run_tools):
            mock_llm_tools.invoke.return_value = tool_call_msg
            mock_llm_struct.invoke.return_value = structured_mock
            result = resolver_node(state, config=config)

        assert result.get("next_action") == "end", f"Expected 'end', got: {result.get('next_action')}"

        # Verify the LTM row was written
        conn = sqlite3.connect(_UDAHUB_DB)
        rows = conn.execute(
            "SELECT content FROM long_term_memory WHERE thread_id = 'test-ltm-resolver-001'"
        ).fetchall()
        conn.close()

        assert rows, "No LTM row written by the resolver"
        import json
        data = json.loads(rows[0][0])
        assert data.get("customer_id") == "ltm-test@example.com"
        assert data.get("category") == "account"

    def test_supervisor_loads_ltm_into_state(self):
        """If LTM exists for a customer, supervisor_node must populate long_term_context."""
        if not _db_exists(_UDAHUB_DB):
            pytest.skip("udahub.db not found")

        from data.repositories import save_memory
        from agentic.agents.supervisor import supervisor_node

        # Pre-seed a memory for the customer
        customer_email = "returning@example.com"
        save_memory(
            _UDAHUB_DB,
            "test-ltm-prior-session",
            customer_email,
            "billing",
            "Resolved duplicate charge: refund of £15 was issued.",
        )

        state = {
            "messages": [HumanMessage(content=f"Hi, my email is {customer_email}, I have another billing question")],
            "long_term_context": "",
        }

        mock_decision = MagicMock()
        mock_decision.action = "classify"
        mock_decision.direct_response = ""

        with patch("agentic.agents.supervisor._structured") as mock_llm:
            mock_llm.invoke.return_value = mock_decision
            result = supervisor_node(state)

        ltm = result.get("long_term_context", "")
        assert ltm, "long_term_context was not populated by the supervisor"
        assert "returning@example.com" in ltm or "billing" in ltm or "refund" in ltm.lower(), (
            f"Expected LTM content not found in: {ltm!r}"
        )


# ── Task 6 — Agent state integration + error-handling tests ──────────────────

class TestAgentStateIntegration:
    """Verify that agent nodes return complete, correctly-typed state fields.

    These tests use mocked LLMs so no API key or network is required.
    """

    # ── Classifier output shape ───────────────────────────────────────────────

    def test_classifier_returns_all_required_state_fields(self):
        """classifier_node must return classification, urgency, AND complexity."""
        from agentic.agents.classifier import classifier_node

        mock_out = MagicMock()
        mock_out.category = "billing"
        mock_out.urgency = "medium"
        mock_out.complexity = "simple"

        with patch("agentic.agents.classifier._llm") as m:
            m.invoke.return_value = mock_out
            result = classifier_node({"messages": [HumanMessage("I was double-charged")]})

        assert result["classification"] == "billing"
        assert result["urgency"] == "medium"
        assert result["complexity"] == "simple"

    def test_classifier_fallback_on_llm_error(self):
        """If the LLM raises, classifier_node must return safe defaults, not crash."""
        from agentic.agents.classifier import classifier_node

        with patch("agentic.agents.classifier._llm") as m:
            m.invoke.side_effect = RuntimeError("LLM unavailable")
            result = classifier_node({"messages": [HumanMessage("Test")]})

        assert result["classification"] == "general"
        assert result["urgency"] == "medium"
        assert result["complexity"] == "simple"

    # ── Escalator output shape ────────────────────────────────────────────────

    def test_escalator_includes_ticket_reference(self):
        """escalator_node response must contain a UDA-* ticket reference."""
        from agentic.agents.escalator import escalator_node

        mock_response = MagicMock()
        mock_response.content = "We are escalating your case. Reference: UDA-BIL-MED."

        with patch("agentic.agents.escalator._llm") as m:
            m.invoke.return_value = mock_response
            state = {
                "messages": [HumanMessage("I have a billing dispute")],
                "classification": "billing",
                "urgency": "medium",
            }
            result = escalator_node(state)

        msgs = result.get("messages", [])
        assert msgs, "escalator_node returned no messages"
        combined = " ".join(m.content for m in msgs)
        assert "UDA-" in combined, f"Ticket reference missing from: {combined!r}"

    def test_escalator_fallback_on_llm_error(self):
        """If LLM raises, escalator_node must return the hardcoded fallback message."""
        from agentic.agents.escalator import escalator_node

        with patch("agentic.agents.escalator._llm") as m:
            m.invoke.side_effect = RuntimeError("timeout")
            state = {
                "messages": [HumanMessage("Help!")],
                "classification": "general",
                "urgency": "low",
            }
            result = escalator_node(state)

        msgs = result.get("messages", [])
        assert msgs, "escalator_node returned no messages in fallback"
        assert "UDA-" in msgs[0].content

    # ── Resolver state shape ──────────────────────────────────────────────────

    def test_resolver_sets_retrieval_quality_on_success(self):
        """After a successful KB search, retrieval_quality must be 'high'."""
        from langchain_core.messages import AIMessage as LCAIMessage, ToolMessage
        import agentic.agents.resolver as resolver_module
        from agentic.agents.resolver import resolver_node

        tool_call_msg = LCAIMessage(
            content="",
            tool_calls=[{"id": "t1", "name": "search_knowledge_base", "args": {"query": "login"}}],
        )
        structured_mock = MagicMock()
        structured_mock.confidence = 0.85
        structured_mock.needs_escalation = False
        structured_mock.response = "Use the Forgot Password link."

        def fake_run_tools(calls):
            return [ToolMessage(content="Login article content here.", tool_call_id=c["id"]) for c in calls]

        state = {
            "messages": [HumanMessage("I can't log in")],
            "classification": "technical", "urgency": "low",
            "complexity": "simple", "ticket_priority": "normal",
            "ticket_age_hours": 0, "confidence": 0.0,
            "retrieval_quality": "none", "next_action": "resolve",
            "long_term_context": "",
        }

        with patch("agentic.agents.resolver._llm_tools") as mock_lt, \
             patch("agentic.agents.resolver._llm_structured") as mock_ls, \
             patch.object(resolver_module, "_run_tools", side_effect=fake_run_tools):
            mock_lt.invoke.return_value = tool_call_msg
            mock_ls.invoke.return_value = structured_mock
            result = resolver_node(state)

        assert result["next_action"] == "end"
        assert result["retrieval_quality"] == "high"
        assert result["confidence"] == pytest.approx(0.85)

    def test_resolver_escalates_when_llm_requests_it(self):
        """When the structured output sets needs_escalation=True, next_action must be 'escalate'."""
        from langchain_core.messages import AIMessage as LCAIMessage, ToolMessage
        import agentic.agents.resolver as resolver_module
        from agentic.agents.resolver import resolver_node

        tool_call_msg = LCAIMessage(content="", tool_calls=[])
        structured_mock = MagicMock()
        structured_mock.confidence = 0.25
        structured_mock.needs_escalation = True
        structured_mock.response = "I cannot help with this."

        state = {
            "messages": [HumanMessage("Complex billing fraud")],
            "classification": "billing", "urgency": "high",
            "complexity": "complex", "ticket_priority": "normal",
            "ticket_age_hours": 0, "confidence": 0.0,
            "retrieval_quality": "none", "next_action": "resolve",
            "long_term_context": "",
        }

        with patch("agentic.agents.resolver._llm_tools") as mock_lt, \
             patch("agentic.agents.resolver._llm_structured") as mock_ls:
            mock_lt.invoke.return_value = tool_call_msg
            mock_ls.invoke.return_value = structured_mock
            result = resolver_node(state)

        assert result["next_action"] == "escalate"

    def test_resolver_fallback_on_llm_error(self):
        """If the LLM raises an exception, resolver_node must return next_action='escalate'."""
        from agentic.agents.resolver import resolver_node

        state = {
            "messages": [HumanMessage("Help me")],
            "classification": "general", "urgency": "low",
            "complexity": "simple", "ticket_priority": "normal",
            "ticket_age_hours": 0, "confidence": 0.0,
            "retrieval_quality": "none", "next_action": "resolve",
            "long_term_context": "",
        }

        with patch("agentic.agents.resolver._llm_tools") as m:
            m.invoke.side_effect = RuntimeError("connection timeout")
            result = resolver_node(state)

        assert result["next_action"] == "escalate"
        assert result["confidence"] == 0.0


class TestErrorHandling:
    """Verify that the system degrades gracefully when infrastructure is missing."""

    def test_account_tool_missing_db_returns_error_string(self):
        """get_user_account must return an error string, not raise, when DB path is invalid."""
        from agentic.tools.account_tools import get_user_account
        import agentic.tools.account_tools as at_module

        with patch.object(at_module, "_CULTPASS_DB", "/nonexistent/path/cultpass.db"):
            result = get_user_account.invoke({"user_identifier": "anyone"})

        assert isinstance(result, str)
        assert len(result) > 0
        # Must not crash — the tool should absorb the FileNotFoundError
        assert "Error" in result or "not found" in result.lower() or "Database" in result

    def test_knowledge_tool_missing_db_returns_escalation_marker(self):
        """search_knowledge_base must return the escalation marker when udahub.db is missing."""
        from agentic.tools.knowledge_tools import search_knowledge_base, ESCALATION_MARKER
        import agentic.tools.knowledge_tools as kt_module

        with patch.object(kt_module, "_UDAHUB_DB", "/nonexistent/path/udahub.db"):
            result = search_knowledge_base.invoke({"query": "login"})

        assert ESCALATION_MARKER in result

    def test_supervisor_fallback_on_llm_error(self):
        """supervisor_node must return next_action='classify' if the LLM raises."""
        from agentic.agents.supervisor import supervisor_node

        with patch("agentic.agents.supervisor._structured") as m:
            m.invoke.side_effect = RuntimeError("API down")
            result = supervisor_node({"messages": [HumanMessage("Hello")]})

        assert result["next_action"] == "classify"


class TestStructuredLogging:
    """Verify that the JsonFormatter produces valid, well-formed log records."""

    def test_json_formatter_produces_valid_json(self):
        """JsonFormatter.format() must return a parseable JSON string."""
        import json
        from agentic.logging_config import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="agentic.agents.resolver",
            level=logging.INFO,
            pathname="", lineno=0, msg="Tool invoked: get_user_account",
            args=(), exc_info=None,
        )
        record.agent = "resolver"
        record.event = "tool_call"
        record.tool_name = "get_user_account"
        record.status = "success"
        record.thread_id = "test-thread-001"

        output = formatter.format(record)
        data = json.loads(output)  # must not raise

        assert data["level"] == "INFO"
        assert data["agent"] == "resolver"
        assert data["event"] == "tool_call"
        assert data["tool_name"] == "get_user_account"
        assert data["status"] == "success"
        assert data["thread_id"] == "test-thread-001"
        assert "timestamp" in data
        assert "message" in data

    def test_json_formatter_omits_none_fields(self):
        """Fields not set via extra must not appear as null in the output."""
        import json
        from agentic.logging_config import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="agentic.agents.classifier",
            level=logging.INFO,
            pathname="", lineno=0, msg="Classified as billing",
            args=(), exc_info=None,
        )
        record.agent = "classifier"
        record.event = "classify"
        record.status = "success"
        # tool_name and thread_id NOT set

        output = formatter.format(record)
        data = json.loads(output)

        assert "tool_name" not in data   # omitted, not null
        assert "thread_id" not in data   # omitted, not null
        assert data["agent"] == "classifier"
