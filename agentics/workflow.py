"""Custom LangGraph workflow — build from scratch (no prebuilt graph)."""
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from agentic.agents.supervisor import supervisor_node, route_after_supervisor
from agentic.agents.classifier import classifier_node
from agentic.agents.resolver import resolver_node, route_after_resolver
from agentic.agents.escalator import escalator_node
import os as _os
from agentic.logging_config import configure_logging

_LOG_FILE = _os.path.join(_os.path.dirname(__file__), "..", "logs", "udahub.log")
configure_logging(log_file=_LOG_FILE)


class AgentState(TypedDict):
    # Conversation messages — new messages are appended via add_messages reducer
    messages: Annotated[list[BaseMessage], add_messages]

    # Set by the Classifier Agent
    classification: str   # billing | technical | account | booking | general
    urgency: str          # low | medium | high
    complexity: str       # simple | complex

    # Ticket metadata — supplied by the caller (channel, CRM, age, priority)
    ticket_priority: str  # normal | high | critical
    ticket_age_hours: int # hours since ticket was first opened

    # Set by the Resolver Agent
    confidence: float     # 0.0–1.0
    retrieval_quality: str  # high | medium | low | none — set after KB search

    # Routing signal written by Supervisor / Resolver
    next_action: str      # classify | end | escalate

    # Retrieved from the long-term memory store by the Supervisor
    long_term_context: str


def route_after_classifier(state: dict) -> str:
    """Route after classification based on category, urgency, complexity and ticket metadata.

    Routing rules (in priority order):
    1. critical ticket priority → immediate escalation regardless of category
    2. high urgency + account or billing → immediate escalation
       (blocked accounts, payment fraud, security breaches require a human)
    3. complex + high urgency (any category) → immediate escalation
    4. everything else → resolver
    """
    category = state.get("classification", "general")
    urgency = state.get("urgency", "low")
    complexity = state.get("complexity", "simple")
    ticket_priority = state.get("ticket_priority", "normal")

    if ticket_priority == "critical":
        return "direct_escalate"

    if urgency == "high" and category in ("account", "billing"):
        return "direct_escalate"

    if urgency == "high" and complexity == "complex":
        return "direct_escalate"

    return "resolve"


def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("resolver", resolver_node)
    graph.add_node("escalator", escalator_node)

    # ── Entry point ──────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # ── Edges ────────────────────────────────────────────────────────────────
    # Supervisor → classify (pipeline) OR → END (greetings / farewells)
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"classify": "classifier", "end": END},
    )

    # Classifier → direct escalation for urgent/critical tickets
    #            → resolver for everything else
    graph.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "direct_escalate": "escalator",  # skip resolver for urgent cases
            "resolve": "resolver",
        },
    )

    # Resolver → END (high confidence) OR → Escalator (low confidence)
    graph.add_conditional_edges(
        "resolver",
        route_after_resolver,
        {"end": END, "escalate": "escalator"},
    )

    # Escalator always terminates
    graph.add_edge("escalator", END)

    return graph


# Compile once with an in-memory checkpointer for short-term session memory.
# The same compiled graph is imported by 03_agentic_app and the tests.
_checkpointer = MemorySaver()
orchestrator = _build_graph().compile(checkpointer=_checkpointer)
