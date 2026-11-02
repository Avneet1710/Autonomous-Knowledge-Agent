"""Escalation Agent — handles cases the resolver cannot confidently resolve."""
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agentic.llm import get_llm

logger = logging.getLogger(__name__)

_llm = get_llm(temperature=0.3)

_SYSTEM = """\
You are the Escalation Agent for UDA-Hub, CultPass's support system.

The automated pipeline could not fully resolve this customer's issue.
Your response must:
1. Acknowledge the issue with empathy and sincerity.
2. Confirm that the case is being escalated to a human specialist.
3. Give a clear SLA: the customer will hear back within 24 hours.
4. Ask for any missing details that would help the specialist (e.g., user ID, email, screenshot).
5. End with the escalation reference: {ticket_ref}

Keep the tone warm, professional, and reassuring. Be concise (3–5 sentences)."""


def escalator_node(state: dict) -> dict:
    messages = state.get("messages", [])
    category = state.get("classification", "general")
    urgency = state.get("urgency", "medium")

    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    ticket_ref = f"UDA-{category[:3].upper()}-{urgency[:3].upper()}"

    try:
        system_msg = SystemMessage(content=_SYSTEM.format(ticket_ref=ticket_ref))
        user_msg = HumanMessage(
            content=last_human.content if last_human else "No message provided."
        )
        response = _llm.invoke([system_msg, user_msg])
        logger.info(
            "Escalated ticket %s (category=%s urgency=%s)", ticket_ref, category, urgency,
            extra={"agent": "escalator", "event": "escalate",
                   "status": "escalated"},
        )
        return {"messages": [AIMessage(content=response.content)]}

    except Exception as exc:
        logger.error(
            "Escalator error: %s", exc,
            extra={"agent": "escalator", "event": "escalate", "status": "error"},
        )
        fallback = (
            "I'm sorry I couldn't resolve this automatically. "
            f"Your case ({ticket_ref}) has been escalated to our support team, "
            "who will follow up within 24 hours. "
            "Please have your account email or user ID ready."
        )
        return {"messages": [AIMessage(content=fallback)]}
