"""Resolver Agent — searches the knowledge base, calls account tools, generates a response."""
import logging
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agentic.llm import get_llm
from agentic.tools.knowledge_tools import search_knowledge_base, ESCALATION_MARKER
from agentic.tools.account_tools import get_user_account, get_user_reservations, manage_subscription
from data.repositories import save_memory

logger = logging.getLogger(__name__)

_SOLUTION_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_UDAHUB_DB = os.path.join(_SOLUTION_ROOT, "data", "core", "udahub.db")

_TOOLS = [search_knowledge_base, get_user_account, get_user_reservations, manage_subscription]
_TOOL_MAP = {t.name: t for t in _TOOLS}

_llm_tools = get_llm().bind_tools(_TOOLS)


class _ResolverOutput(BaseModel):
    response: str = Field(description="The reply to send to the customer")
    confidence: float = Field(ge=0.0, le=1.0, description="How confident you are in this answer")
    needs_escalation: bool = Field(description="True if a human agent is required")


_llm_structured = get_llm().with_structured_output(_ResolverOutput)

_SYSTEM_TOOL = """\
You are the Resolver Agent for UDA-Hub, handling CultPass customer support.

Ticket category: {category} | Urgency: {urgency}
{ltm_note}

Your task:
1. Search the knowledge base for articles relevant to the customer's issue.
2. If the customer provides their user ID or email, look up their account details.
3. Use the information you find to prepare a helpful response.

Always call search_knowledge_base first. Call account tools only if the customer
has provided identifying information or explicitly asked about their account."""

_SYSTEM_GENERATE = """\
You are the Resolver Agent for UDA-Hub.

IMPORTANT: Your response MUST be grounded exclusively in the retrieved articles provided below.
- Do NOT use general knowledge, assumptions, or information not present in the retrieved articles.
- If the retrieved articles do not sufficiently answer the customer's question, say so honestly
  and set confidence below 0.4 and needs_escalation=True.
- Quoting or paraphrasing specific article guidance is encouraged.

Set **confidence** (0.0–1.0):
- 0.8–1.0  Retrieved article fully and directly covers the issue
- 0.5–0.79 Partial match — article is relevant but doesn't cover every aspect
- 0.2–0.49 Weak match — article is only loosely related; answer may not fully help
- 0.0–0.19 No relevant info retrieved — escalation required

Set **needs_escalation** to true when:
- confidence < 0.4
- The retrieved articles do not address the question
- The issue involves account-blocked status, security, or billing dispute
- The customer explicitly asks to speak to a human"""


def _run_tools(tool_calls: list) -> list[ToolMessage]:
    results = []
    for call in tool_calls:
        fn = _TOOL_MAP.get(call["name"])
        if fn:
            try:
                out = fn.invoke(call["args"])
                results.append(ToolMessage(content=str(out), tool_call_id=call["id"]))
            except Exception as exc:
                results.append(ToolMessage(content=f"Tool error: {exc}", tool_call_id=call["id"]))
    return results


def _extract_customer_id(tool_calls: list) -> str:
    """Pull user_identifier out of any account-tool call made during resolution."""
    account_tools = {"get_user_account", "get_user_reservations", "manage_subscription"}
    for call in tool_calls:
        if call.get("name") in account_tools:
            uid = call.get("args", {}).get("user_identifier", "")
            if uid:
                return uid
    return ""


def resolver_node(state: dict, config: RunnableConfig = None) -> dict:
    messages = state.get("messages", [])
    category = state.get("classification", "general")
    urgency = state.get("urgency", "medium")
    ltm = state.get("long_term_context", "")
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "") if config else ""

    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not last_human:
        return {"confidence": 0.0, "next_action": "escalate"}

    ltm_note = f"Prior context about this customer: {ltm}" if ltm else ""
    system_tool = SystemMessage(
        content=_SYSTEM_TOOL.format(category=category, urgency=urgency, ltm_note=ltm_note)
    )

    try:
        # Phase 1: let the LLM decide which tools to call
        tool_response = _llm_tools.invoke([system_tool, last_human])
        tool_results: list[ToolMessage] = []
        all_tool_calls: list[dict] = []

        if getattr(tool_response, "tool_calls", None):
            all_tool_calls = tool_response.tool_calls
            tool_results = _run_tools(all_tool_calls)
            for tc in all_tool_calls:
                logger.info(
                    "Tool invoked: %s", tc["name"],
                    extra={"agent": "resolver", "event": "tool_call",
                           "tool_name": tc["name"], "status": "success",
                           "thread_id": thread_id},
                )

        # Phase 2: deterministic escalation if KB returned no relevant articles
        # This fires before the LLM confidence check so the decision is objective,
        # not dependent on the LLM's self-reported confidence score.
        kb_failed = any(ESCALATION_MARKER in m.content for m in tool_results)
        retrieval_quality = "none" if kb_failed else ("high" if tool_results else "low")

        if kb_failed:
            logger.warning(
                "Resolver: KB returned no relevant articles — deterministic escalation",
                extra={"agent": "resolver", "event": "escalate",
                       "status": "escalated", "thread_id": thread_id},
            )
            return {
                "confidence": 0.0,
                "retrieval_quality": "none",
                "next_action": "escalate",
            }

        retrieved_text = (
            "\n\n".join(m.content for m in tool_results)
            if tool_results
            else "No tools were called; no external data retrieved."
        )

        # Phase 3: generate a structured final answer grounded in retrieved articles
        generate_messages = [
            SystemMessage(content=_SYSTEM_GENERATE),
            last_human,
            HumanMessage(
                content=f"Retrieved information:\n\n{retrieved_text}\n\nNow write the response."
            ),
        ]
        output = _llm_structured.invoke(generate_messages)

        status = "escalated" if (output.needs_escalation or output.confidence < 0.4) else "success"
        logger.info(
            "Resolver: confidence=%.2f, escalate=%s, retrieval_quality=%s",
            output.confidence, output.needs_escalation, retrieval_quality,
            extra={"agent": "resolver", "event": "resolve",
                   "status": status, "thread_id": thread_id},
        )

        if output.needs_escalation or output.confidence < 0.4:
            return {
                "confidence": output.confidence,
                "retrieval_quality": retrieval_quality,
                "next_action": "escalate",
            }

        # Phase 4: persist resolution to long-term memory
        customer_id = _extract_customer_id(all_tool_calls)
        if thread_id and customer_id and os.path.exists(_UDAHUB_DB):
            try:
                save_memory(
                    _UDAHUB_DB,
                    thread_id=thread_id,
                    customer_id=customer_id,
                    category=category,
                    summary=output.response[:250],
                )
                logger.info("LTM saved for customer %s (thread %s)", customer_id, thread_id)
            except Exception as exc:
                logger.warning("LTM save failed (non-fatal): %s", exc)

        return {
            "messages": [AIMessage(content=output.response)],
            "confidence": output.confidence,
            "retrieval_quality": retrieval_quality,
            "next_action": "end",
        }

    except Exception as exc:
        logger.error(
            "Resolver error: %s", exc,
            extra={"agent": "resolver", "event": "resolve",
                   "status": "error", "thread_id": thread_id},
        )
        return {"confidence": 0.0, "retrieval_quality": "none", "next_action": "escalate"}


def route_after_resolver(state: dict) -> str:
    return state.get("next_action", "end")
