"""Supervisor Agent — entry point that routes each user message."""
import logging
import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from agentic.llm import get_llm
from data.repositories import get_customer_memories, extract_customer_id

logger = logging.getLogger(__name__)

_SOLUTION_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_UDAHUB_DB = os.path.join(_SOLUTION_ROOT, "data", "core", "udahub.db")


class _SupervisorDecision(BaseModel):
    action: Literal["classify", "end"]
    direct_response: str = ""


_structured = get_llm().with_structured_output(_SupervisorDecision)

_SYSTEM = """You are the Supervisor Agent for UDA-Hub, CultPass's intelligent support system.

For each incoming user message decide:

- **classify**: The user has a genuine support request (login issues, billing, reservations,
  subscription, app problems, account concerns, refunds, etc.). Route to the classification pipeline.

- **end**: The user is greeting, saying goodbye, sending small talk, or confirming the session is
  over. Provide a short, warm direct_response.

Be decisive. When in doubt, classify."""


def supervisor_node(state: dict, config: RunnableConfig = None) -> dict:
    messages = state.get("messages", [])
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "") if config else ""
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    if not last_human:
        return {"next_action": "end"}

    # ── Log incoming user query ───────────────────────────────────────────────
    logger.info(
        "User query: %s", last_human.content,
        extra={"agent": "supervisor", "event": "user_query",
               "status": "received", "thread_id": thread_id},
    )

    # ── Long-term memory retrieval ────────────────────────────────────────────
    # If we don't yet have context for this session, look up the customer's
    # prior resolved issues by scanning their message for an email address.
    long_term_context = state.get("long_term_context", "")
    if not long_term_context and os.path.exists(_UDAHUB_DB):
        customer_id = extract_customer_id(last_human.content)
        if customer_id:
            try:
                memories = get_customer_memories(_UDAHUB_DB, customer_id, limit=3)
                if memories:
                    long_term_context = (
                        f"Prior resolved issues for {customer_id}:\n"
                        + "\n".join(f"- {m}" for m in memories)
                    )
                    logger.info("LTM loaded for %s: %d memories", customer_id, len(memories))
            except Exception as exc:
                logger.warning("LTM retrieval failed: %s", exc)

    try:
        decision = _structured.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"User message: {last_human.content}"),
        ])
        logger.info(
            "Supervisor routed to %s", decision.action,
            extra={"agent": "supervisor", "event": "route",
                   "status": "success", "thread_id": thread_id},
        )

        result: dict = {"next_action": decision.action, "long_term_context": long_term_context}
        if decision.action == "end" and decision.direct_response:
            result["messages"] = [AIMessage(content=decision.direct_response)]
        return result

    except Exception as exc:
        logger.error(
            "Supervisor error: %s", exc,
            extra={"agent": "supervisor", "event": "route", "status": "error", "thread_id": thread_id},
        )
        return {"next_action": "classify", "long_term_context": long_term_context}


def route_after_supervisor(state: dict) -> str:
    return state.get("next_action", "classify")
