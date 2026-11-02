"""Classifier Agent — categorises the ticket, sets urgency and complexity."""
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic.llm import get_llm

logger = logging.getLogger(__name__)


class _Classification(BaseModel):
    category: Literal["billing", "technical", "account", "booking", "general"]
    urgency: Literal["low", "medium", "high"]
    complexity: Literal["simple", "complex"] = Field(
        description="simple = one clear issue answerable from KB; complex = multi-part, ambiguous, or requires human judgement"
    )
    summary: str = Field(description="One-sentence summary of the issue")


_llm = get_llm().with_structured_output(_Classification)

_SYSTEM = """You are the Classifier Agent for UDA-Hub customer support.

**Categories:**
- billing     — payment problems, subscription cost, charges, refunds, invoices
- technical   — app crashes, login errors, QR code issues, connectivity, bugs
- account     — profile, blocked account, password, contact details, preferences
- booking     — reservations, event cancellations, experience queries, premium access
- general     — general questions, feedback, anything else

**Urgency:**
- high   — account blocked, payment failure, security concern, data loss, fraud
- medium — feature not working, service disruption affecting usage
- low    — general questions, minor inconveniences, feedback

**Complexity:**
- simple  — single, well-defined issue that can be resolved using the knowledge base
- complex — multiple issues, ambiguous situation, suspected fraud, or clearly requires human review

**Ticket Metadata (if provided):**
- ticket_priority: normal | high | critical — set by support channel; escalate critical tickets immediately
- ticket_age_hours: how long the ticket has been open — older tickets may need priority handling

Use all available signals (message content + metadata) to classify accurately."""


def classifier_node(state: dict) -> dict:
    messages = state.get("messages", [])
    ticket_priority = state.get("ticket_priority", "normal")
    ticket_age_hours = state.get("ticket_age_hours", 0)

    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    if not last_human:
        return {"classification": "general", "urgency": "low", "complexity": "simple"}

    # Append ticket metadata to the message so the LLM can factor it in
    metadata_lines = []
    if ticket_priority != "normal":
        metadata_lines.append(f"ticket_priority: {ticket_priority}")
    if ticket_age_hours > 0:
        metadata_lines.append(f"ticket_age_hours: {ticket_age_hours}")

    content = last_human.content
    if metadata_lines:
        content += "\n\n[Ticket metadata: " + ", ".join(metadata_lines) + "]"

    try:
        result = _llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=content),
        ])
        logger.info(
            "Classified as %s / urgency=%s / complexity=%s",
            result.category, result.urgency, result.complexity,
            extra={"agent": "classifier", "event": "classify",
                   "status": "success"},
        )
        return {
            "classification": result.category,
            "urgency": result.urgency,
            "complexity": result.complexity,
        }
    except Exception as exc:
        logger.error(
            "Classifier error: %s", exc,
            extra={"agent": "classifier", "event": "classify", "status": "error"},
        )
        return {"classification": "general", "urgency": "medium", "complexity": "simple"}
