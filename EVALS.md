# EVALS.md — Evaluating the UDA-Hub Agentic System

> What to evaluate, why it matters, when to run it, and how to measure it.

---

## Why Evaluate an Agentic System?

A regular software test checks whether a function returns the right output for a given input.
An agentic system is different — the "output" is a multi-step process where:
- An LLM makes decisions at each step
- Those decisions route the workflow differently
- Tools interact with real databases
- The final answer is natural language, not a deterministic value

This means traditional unit tests can confirm the *plumbing* works, but they cannot confirm the system *behaves correctly* in practice. Evaluation (evals) fills that gap: it measures whether the agent does the right thing, for the right reasons, reliably across many scenarios.

---

## Dimensions of Evaluation

UDA-Hub is evaluated across five dimensions:

| Dimension | Question it answers |
|---|---|
| **Routing correctness** | Did the right agent handle the right kind of ticket? |
| **Retrieval quality** | Did the KB search return the right articles? |
| **Response groundedness** | Is the answer based on retrieved knowledge, not hallucination? |
| **Resolution accuracy** | Was the issue actually resolved correctly? |
| **Escalation precision** | Did the system escalate when it should, and not escalate when it shouldn't? |

---

## 1. Routing Correctness

**What:** Verify that `route_after_classifier` sends each ticket to the correct node — resolver or escalator — based on category, urgency, complexity, and ticket priority.

**Why:** If a high-urgency account breach goes to the resolver instead of the escalator, a customer with a compromised account might receive a canned KB answer instead of immediate human help. Routing errors are silent and dangerous.

**When:** Run on every code change to `workflow.py` or `classifier.py`. These are fast, deterministic, LLM-free tests.

**How:**
```python
# Pure unit test — no LLM, no DB
from agentic.workflow import route_after_classifier

def test_blocked_account_escalates_immediately():
    state = {"classification": "account", "urgency": "high",
             "complexity": "simple", "ticket_priority": "normal"}
    assert route_after_classifier(state) == "direct_escalate"

def test_billing_question_goes_to_resolver():
    state = {"classification": "billing", "urgency": "medium",
             "complexity": "simple", "ticket_priority": "normal"}
    assert route_after_classifier(state) == "resolve"
```

**Metrics:** 100% of routing tests must pass. No tolerance for misrouting high-urgency tickets.

---

## 2. Retrieval Quality

**What:** Verify that `search_knowledge_base` returns the correct article for a given query, and returns the escalation marker when no relevant article exists.

**Why:** The entire resolution chain depends on retrieval. If the wrong article is returned, the resolver will give a wrong (but confident-sounding) answer. If no article is returned but the escalation marker is missing, the LLM will hallucinate an answer.

**When:** Run after any change to `knowledge_tools.py`, after adding new articles to the KB, or after changing `MIN_RELEVANCE_SCORE`.

**How — Article match tests:**
```python
result = search_knowledge_base.invoke({"query": "login password reset"})
assert "Login" in result or "Password" in result  # correct article returned
assert ESCALATION_MARKER not in result             # not a false negative
```

**How — No-match tests:**
```python
result = search_knowledge_base.invoke({"query": "xyzzy frobnicator quux"})
assert ESCALATION_MARKER in result  # escalation triggered deterministically
```

**Metrics:**
- Precision: what fraction of retrieved articles are relevant to the query?
- Recall: for each known query type, is the expected article in the top-3 results?
- False-negative rate: how often does the system fail to retrieve an article that exists?

**Target:** ≥90% precision and recall across the 14 article categories.

---

## 3. Response Groundedness

**What:** Verify that the Resolver's answer is derived from retrieved KB articles, not from the LLM's general training knowledge.

**Why:** An LLM can generate plausible-sounding but wrong answers from its training data. For a support system, a wrong but confident answer is worse than no answer — it misleads the customer. Grounded responses are verifiable; hallucinated ones are not.

**When:** Run on a curated eval set whenever the Resolver system prompt changes or the LLM model is updated.

**How — Automated grounding check:**
For each test case, retrieve the KB articles manually and compare the answer to article content:
```python
# 1. Run the resolver with a known query
result = orchestrator.invoke({"messages": [HumanMessage("How do I reset my password?")]}, config)

# 2. Retrieve the expected article independently
kb_result = search_knowledge_base.invoke({"query": "password reset"})

# 3. Check that key phrases from the article appear in the answer
assert "Forgot Password" in result["messages"][-1].content  # from the KB article
```

**How — LLM-as-judge (qualitative):**
Use a separate LLM call to evaluate: *"Does this response reference the provided article content, or does it introduce information not present in the articles?"*

```python
judge_prompt = f"""
Article: {kb_article}
Response: {agent_response}

Does the response stay within the information provided in the article?
Answer: YES or NO, then one sentence explanation.
"""
```

**Metrics:**
- Grounding rate: % of responses that reference article content
- Hallucination rate: % of responses that introduce facts not in retrieved articles

**Target:** ≥85% grounding rate; <10% hallucination rate.

---

## 4. Resolution Accuracy

**What:** Given a realistic customer support scenario, did the agent produce a response that actually solves (or correctly escalates) the customer's problem?

**Why:** This is the ultimate measure of system usefulness. A system can route correctly, retrieve the right article, and still write an unhelpful response. End-to-end accuracy measures the complete chain.

**When:** Run on a fixed eval set of 20–50 representative support scenarios before any release. These require a live LLM and are slower — run in CI on pull requests, not on every commit.

**How — Eval set structure (sample):**

| Scenario | Expected outcome | Expected classification |
|---|---|---|
| "I can't log in, password reset email not arriving" | resolved + login article cited | technical / medium |
| "I want to cancel my subscription" | resolved + manage_subscription called | account / low |
| "My account was hacked" | escalated | account / high |
| "What's included in my plan?" | resolved + subscription article cited | general / low |
| "Completely unrelated topic (xyzzy)" | escalated | any / low |

**How — Checking each scenario:**
```python
result = orchestrator.invoke({"messages": [HumanMessage(scenario)]}, config)
final_msg = result["messages"][-1].content

# For a "should resolve" case:
assert result["next_action"] == "end"
assert result["confidence"] >= 0.5
assert len(final_msg) > 50  # substantive answer

# For a "should escalate" case:
assert result["next_action"] == "escalate"
assert "UDA-" in final_msg  # ticket reference from Escalator
```

**Metrics:**
- Resolution rate: % of resolvable tickets that were resolved (not over-escalated)
- Escalation precision: % of escalated tickets that genuinely needed a human
- Correct-path rate: % of tickets that followed the expected workflow path

**Target:** ≥80% resolution rate on clearly answerable tickets; ≥95% escalation rate on high-urgency tickets.

---

## 5. Escalation Precision

**What:** Check that the system escalates exactly when it should — and does NOT escalate when it shouldn't.

**Why:** Two failure modes exist:
- **Under-escalation:** a complex or sensitive case is "resolved" by the AI when it needs a human → customer receives wrong guidance, no human follows up.
- **Over-escalation:** a simple case is escalated unnecessarily → increased support load, poor customer experience.

**When:** Run alongside the resolution accuracy eval set.

**How:**

```python
# Should escalate — account hacked, high urgency
result = orchestrator.invoke({"messages": [HumanMessage(
    "I think my account has been compromised and someone changed my email"
)]}, config)
assert result.get("next_action") == "escalate"
assert "UDA-" in result["messages"][-1].content  # escalation ticket ref present

# Should NOT escalate — simple FAQ question
result = orchestrator.invoke({"messages": [HumanMessage(
    "How many experiences can I attend per month?"
)]}, config)
assert result.get("next_action") == "end"
assert result.get("confidence", 0) >= 0.5
```

**Metrics:**
- False escalation rate: simple tickets incorrectly escalated
- Missed escalation rate: complex/urgent tickets not escalated

**Target:** <15% false escalation rate; <5% missed escalation rate for high-urgency tickets.

---

## 6. Memory Correctness

**What:** Verify that the short-term and long-term memory mechanisms work as expected.

**Why:** If memory is broken, the agent asks customers to repeat themselves (short-term failure) or gives generic responses to returning customers who have been helped before (long-term failure). Both destroy trust.

**When:** Run after any change to `workflow.py` checkpointing or `udahub.py` LongTermMemory model.

**How — Short-term (same session):**
```python
thread_id = "eval-memory-001"
config = {"configurable": {"thread_id": thread_id}}

# Turn 1
orchestrator.invoke({"messages": [HumanMessage("I can't log in")]}, config)

# Turn 2 — follow-up should be understood in context
r2 = orchestrator.invoke({"messages": [HumanMessage("What if the email doesn't arrive?")]}, config)
# Agent should reference "the reset email" without needing re-explanation
assert len(r2["messages"]) > 2  # history accumulated
```

**How — Long-term (cross-session):**

The LTM system has two parts: the Resolver saves on success; the Supervisor loads on the next session.

```python
import json

# ── Part A: verify Resolver saves LTM after success ──────────────────────────
from data.repositories import save_memory, get_customer_memories

save_memory(UDAHUB_DB, "session-001", "alice@example.com", "billing", "Resolved duplicate charge.")
memories = get_customer_memories(UDAHUB_DB, "alice@example.com")
assert any("duplicate charge" in m for m in memories)

# ── Part B: verify memories are customer-scoped ───────────────────────────────
save_memory(UDAHUB_DB, "session-002", "bob@example.com", "technical", "Resolved login issue.")
alice_mems = get_customer_memories(UDAHUB_DB, "alice@example.com")
bob_mems   = get_customer_memories(UDAHUB_DB, "bob@example.com")
assert not any("login" in m for m in alice_mems)  # Bob's memory stays with Bob

# ── Part C: verify Supervisor injects LTM into state ──────────────────────────
# Pre-seed a memory for a returning customer
save_memory(UDAHUB_DB, "prior-session", "returning@example.com", "billing", "Refund of £15 issued.")

state = {"messages": [HumanMessage("Hi, email is returning@example.com, billing question again")],
         "long_term_context": ""}

with patch("agentic.agents.supervisor._structured") as mock_llm:
    mock_llm.invoke.return_value = MagicMock(action="classify", direct_response="")
    result = supervisor_node(state)

assert result["long_term_context"]  # LTM was loaded and injected
assert "refund" in result["long_term_context"].lower() or "billing" in result["long_term_context"].lower()
```

---

## 7. Tool Integration

**What:** Verify that (a) the Resolver correctly dispatches the right tool call when the LLM produces one, and (b) tool invocations actually reach the database and persist state changes.

**Why:** Three silent failures are possible: the LLM decides not to call a tool at all, it calls the wrong tool, or it calls the right tool but the repository layer fails to persist the change. All three result in a response that appears correct but is based on stale or fabricated data.

**When:** Run after any change to `account_tools.py`, `knowledge_tools.py`, `resolver.py`, or anything in `data/repositories/`.

**How — Agent dispatches the correct tool (no real DB, no real API):**

The Resolver's `_run_tools` function is a plain Python function and can be patched directly. This tests the routing of tool calls without exercising the database.

```python
import agentic.agents.resolver as resolver_module
from langchain_core.messages import AIMessage, ToolMessage

tool_call_msg = AIMessage(
    content="",
    tool_calls=[{
        "id": "tc_001",
        "name": "manage_subscription",
        "args": {"user_identifier": "f556c0", "action": "cancel"},
    }],
)

recorded = []

def spy_run_tools(tool_calls):
    recorded.extend(tool_calls)
    return [ToolMessage(content="Subscription cancelled.", tool_call_id=tc["id"]) for tc in tool_calls]

with patch("agentic.agents.resolver._llm_tools") as mock_llm_tools, \
     patch("agentic.agents.resolver._llm_structured") as mock_llm_struct, \
     patch.object(resolver_module, "_run_tools", side_effect=spy_run_tools):
    mock_llm_tools.invoke.return_value = tool_call_msg
    mock_llm_struct.invoke.return_value = structured_mock
    resolver_node(state)

assert any(tc["name"] == "manage_subscription" for tc in recorded)
assert any(tc["args"]["action"] == "cancel" for tc in recorded)
```

**How — DB-state verification with restore fixture:**

Tools that modify the database are tested against the real `cultpass.db`. A pytest fixture snapshots the subscription status before the test and restores it after, keeping tests independent of execution order.

```python
@pytest.fixture(autouse=True)
def _restore_bob_subscription():
    conn = sqlite3.connect(CULTPASS_DB)
    row = conn.execute("SELECT status FROM subscriptions WHERE user_id='f556c0'").fetchone()
    original = row[0] if row else None
    conn.close()
    yield
    if original is not None:
        conn = sqlite3.connect(CULTPASS_DB)
        conn.execute("UPDATE subscriptions SET status=? WHERE user_id='f556c0'", (original,))
        conn.commit()
        conn.close()

def test_manage_subscription_cancel_updates_db():
    result = manage_subscription.invoke({"user_identifier": "f556c0", "action": "cancel"})
    assert "cancel" in result.lower()
    conn = sqlite3.connect(CULTPASS_DB)
    row = conn.execute("SELECT status FROM subscriptions WHERE user_id='f556c0'").fetchone()
    conn.close()
    assert row[0] == "cancelled"
```

**Why separate these two test types:** The dispatcher test runs without a DB or API key — it belongs on every commit. The DB-state test verifies the complete path from tool → repository → SQLite — it catches regressions in the repository layer.

---

## 10. End-to-End State Assertions

**What:** Run the complete orchestrator with a real LLM and verify the full agent *state* — not just the message text. Each integration test asserts on `next_action`, `retrieval_quality`, `confidence`, `classification`, and the presence of specific keywords or ticket references in the final message.

**Why:** A response that "looks right" can still be produced by the wrong code path. A login question that escalates instead of resolving, or a KB hit that registers `retrieval_quality="low"`, is a regression that text-length checks would miss entirely. State assertions catch these silent failures.

**When:** Every PR, against a populated database and a valid API key.

**How — Resolution path:**
```python
def test_login_issue_resolves_with_kb_retrieval(self):
    result = self._invoke("I can't log in to my CultPass account")

    assert result.get("next_action") == "end"
    assert result.get("retrieval_quality") in ("high", "medium")
    assert result.get("confidence", 0) >= 0.4
    assert result.get("classification") == "technical"

    content_lower = result["messages"][-1].content.lower()
    assert any(kw in content_lower for kw in ("login", "password", "reset", "forgot", "account"))
```

**How — Escalation path:**
```python
def test_unknown_topic_explicitly_escalates(self):
    result = self._invoke("My blockchain NFT wallet integration with CultPass is broken...")

    final_msg = result["messages"][-1].content
    escalated_via_state = result.get("next_action") == "escalate"
    escalated_via_msg   = "UDA-" in final_msg

    assert escalated_via_state or escalated_via_msg, (
        f"Expected escalation. next_action={result.get('next_action')!r}, "
        f"message={final_msg[:200]!r}"
    )
```

**State fields checked per test:**

| Test | `next_action` | `retrieval_quality` | `confidence` | `classification` | `UDA-` in message |
|---|---|---|---|---|---|
| Login resolves | `end` | `high`/`medium` | ≥ 0.4 | `technical` | No |
| Subscription resolves | `end` | — | ≥ 0.4 | — | No |
| Unknown topic escalates | `escalate` | — | — | — | Yes |
| Hacked account escalates | — | — | — | — | Yes |
| Multi-turn accumulates | `end` (both) | — | — | — | No |

---

## 11. Complete Ticket Workflow + Structured Log Capture

**What:** Run a real workflow, install a `_LogCapture` handler on the root logger before invocation, collect every structured log record, and assert that the expected agent/event/status combinations were actually emitted.

**Why:** Structured logs are the audit trail for every agent decision and tool call. If a resolver calls no tools, or an escalation produces no escalation log event, the pipeline is broken even if the final message appears correct. Log-capture tests verify the *interior* of the workflow, not just its output.

**How — `_LogCapture` helper:**
```python
class _LogCapture(logging.Handler):
    def emit(self, record):
        self.records.append(record)

    def structured(self):
        fmt = JsonFormatter()
        return [json.loads(fmt.format(r)) for r in self.records if hasattr(r, "agent")]

    def events_by(self, agent=None, event=None):
        return [r for r in self.structured()
                if (agent is None or r.get("agent") == agent)
                and (event is None or r.get("event") == event)]

    @staticmethod
    def install():
        cap = _LogCapture()
        logging.getLogger().addHandler(cap)
        return cap
```

**How — Resolution path log verification:**
```python
def test_resolution_workflow_complete_pipeline(self):
    result, cap = self._run_with_logs("I can't log in to my CultPass account")

    assert result.get("next_action") == "end"
    assert result.get("retrieval_quality") in ("high", "medium")

    supervisor_routes = cap.events_by(agent="supervisor", event="route")
    assert supervisor_routes and supervisor_routes[0]["status"] == "success"

    resolver_resolves = cap.events_by(agent="resolver", event="resolve")
    assert resolver_resolves and resolver_resolves[0]["status"] == "success"

    tool_names = [t.get("tool_name") for t in cap.events_by(agent="resolver", event="tool_call")]
    assert "search_knowledge_base" in tool_names

    for record in cap.structured():
        assert all(f in record for f in ("timestamp", "level", "agent", "event", "status"))
```

**How — Escalation path log verification:**
```python
def test_escalation_workflow_complete_pipeline(self):
    result, cap = self._run_with_logs(
        "Someone has taken over my account and I cannot get back in — this is urgent"
    )
    assert "UDA-" in result["messages"][-1].content

    escalator_events = cap.events_by(agent="escalator", event="escalate")
    assert escalator_events, "Escalator event not logged"
    assert escalator_events[0]["status"] == "escalated"
```

**Logs captured per workflow path:**

| Log event | `agent` | `event` | `status` | Triggered by |
|---|---|---|---|---|
| Supervisor routing decision | `supervisor` | `route` | `success` | Every ticket |
| Classifier labelling | `classifier` | `classify` | `success` | Support tickets |
| Resolver tool call | `resolver` | `tool_call` | `success` | Each tool invoked |
| Resolver resolution | `resolver` | `resolve` | `success` / `escalated` | After tool calls |
| Resolver KB-failure escalation | `resolver` | `escalate` | `escalated` | ESCALATION_MARKER detected |
| Escalator ticket created | `escalator` | `escalate` | `escalated` | All escalation paths |

---

## Eval Infrastructure Summary

| Eval type | Speed | LLM needed | DB needed | Run frequency |
|---|---|---|---|---|
| Routing unit tests | Fast (<5s) | No | No | Every commit |
| Retrieval unit tests | Fast (<10s) | No | Yes | Every commit |
| Tool dispatch (spy) | Fast (<5s) | No (mocked) | No | Every commit |
| Repository / DB-state | Fast (<30s) | No | Yes | Every commit |
| LTM save / retrieve | Fast (<10s) | No (mocked) | Yes | Every commit |
| Agent state shape | Fast (<5s) | No (mocked) | No | Every commit |
| Error-handling / fallback | Fast (<5s) | No (mocked) | Partial | Every commit |
| Structured logging | Fast (<5s) | No | No | Every commit |
| End-to-end state assertions | Slow (2–5 min) | Yes | Yes | Every PR |
| Complete workflow + log capture | Slow (5–10 min) | Yes | Yes | Every PR |
| Grounding (automated) | Medium (1–2 min) | Yes | Yes | Every PR |
| Resolution accuracy | Slow (5–10 min) | Yes | Yes | Every PR |
| Escalation precision | Slow (5–10 min) | Yes | Yes | Every PR |
| Memory correctness | Medium (2–3 min) | Yes | Yes | Weekly / on change |

---

## Recommended Eval Set Size

| Stage | Scenarios |
|---|---|
| Development (fast feedback) | 10–15 hand-crafted scenarios covering each category |
| Pre-release (thorough) | 50–100 scenarios including edge cases and adversarial inputs |
| Production monitoring | Sampled live tickets (with consent) reviewed weekly |

---

## 8. Agent State Integration

**What:** Verify that each agent node returns a complete, correctly-typed state dictionary — all required fields present, correct values, correct types — including fallback behaviour when the LLM or infrastructure fails.

**Why:** The LangGraph state is the contract between agents. A missing `retrieval_quality` field or a wrong `next_action` value silently corrupts downstream decisions. These tests catch integration-level regressions that pure unit tests on isolated functions miss.

**When:** Run after any change to agent node functions or `AgentState` TypedDict.

**How — State shape tests (no LLM, no DB):**
```python
# Classifier must always return all three fields
mock_out.category = "billing"; mock_out.urgency = "medium"; mock_out.complexity = "simple"
with patch("agentic.agents.classifier._llm") as m:
    m.invoke.return_value = mock_out
    result = classifier_node({"messages": [HumanMessage("double charge")]})
assert "classification" in result and "urgency" in result and "complexity" in result

# Resolver must tag retrieval_quality after tool call
result = resolver_node(state)  # with mocked LLM + fake tool results
assert result["retrieval_quality"] == "high"
assert result["confidence"] == pytest.approx(0.85)
```

**How — Fallback / error-handling tests:**
```python
# Classifier falls back to safe defaults on LLM error
with patch("agentic.agents.classifier._llm") as m:
    m.invoke.side_effect = RuntimeError("LLM unavailable")
    result = classifier_node({"messages": [HumanMessage("test")]})
assert result["classification"] == "general"   # safe default
assert result["next_action"] is None or "urgency" in result  # no crash

# Resolver falls back to escalate on LLM error
with patch("agentic.agents.resolver._llm_tools") as m:
    m.invoke.side_effect = RuntimeError("timeout")
    result = resolver_node(state)
assert result["next_action"] == "escalate"
assert result["confidence"] == 0.0

# Account tool returns error string (not raises) when DB missing
with patch.object(account_tools_module, "_CULTPASS_DB", "/missing/path.db"):
    result = get_user_account.invoke({"user_identifier": "anyone"})
assert "Error" in result or "not found" in result.lower()
```

**Metrics:** 100% of state-shape tests must pass. Zero test failures on error-path tests. No unhandled exceptions from any agent node under any tested failure condition.

---

## 9. Structured Logging

**What:** Verify that the `JsonFormatter` in `agentic/logging_config.py` produces valid, well-formed JSON log records with all required fields present and no `null`/absent fields for values that were not set. Also verify the persistent log file and the user-query log event.

**Why:** If the formatter is broken, monitoring dashboards and alerting rules fail silently — logs go out as garbage JSON or malformed strings and are dropped by the log aggregator. A test that parses the formatter's output catches this before deployment. The user-query event is equally important: without it, the log file records agent decisions but not the input that triggered them, making debugging and auditing harder.

**When:** Run after any change to `logging_config.py`, `supervisor.py`, or after adding/removing structured fields in any agent's log calls.

**How:**
```python
import json, logging
from agentic.logging_config import JsonFormatter

formatter = JsonFormatter()
record = logging.LogRecord(
    name="agentic.agents.resolver", level=logging.INFO,
    pathname="", lineno=0, msg="Tool invoked", args=(), exc_info=None,
)
record.agent = "resolver"
record.event = "tool_call"
record.tool_name = "get_user_account"
record.status = "success"
record.thread_id = "test-thread-001"

output = formatter.format(record)
data = json.loads(output)          # must not raise

assert data["agent"]     == "resolver"
assert data["event"]     == "tool_call"
assert data["tool_name"] == "get_user_account"
assert data["status"]    == "success"
assert data["thread_id"] == "test-thread-001"
assert "timestamp" in data
assert "tool_name" not in json.loads(formatter.format(minimal_record))  # omitted, not null
```

**How — file output and user-query event (integration):**
```python
# After a real workflow run, logs/udahub.log must exist and contain user_query events
import json, os
log_path = os.path.join("logs", "udahub.log")
assert os.path.exists(log_path), "Log file not created"

with open(log_path, encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

user_queries = [r for r in records if r.get("event") == "user_query"]
assert user_queries, "No user_query events found in log file"
assert user_queries[0]["status"] == "received"
assert len(user_queries[0]["message"]) > 10  # actual query text logged
```

**Fields every structured log record must contain:**

| Field | Type | Example |
|---|---|---|
| `timestamp` | ISO-8601 string | `"2026-11-02T10:42:33.801+00:00"` |
| `level` | string | `"INFO"` |
| `agent` | string | `"resolver"` |
| `event` | string | `"user_query"` / `"route"` / `"classify"` / `"tool_call"` / `"resolve"` / `"escalate"` |
| `status` | string | `"received"` / `"success"` / `"error"` / `"escalated"` |
| `thread_id` | string (when set) | `"session-abc123"` |
| `tool_name` | string (when set) | `"get_user_account"` |

**Log file location:** `logs/udahub.log` — created automatically at first run, appended on subsequent runs. Each line is one JSON object.

---

## 12. Integration Test Suite — All 11 Tests

The full integration suite runs with `pytest -m integration` and requires a valid API key and populated databases. Each test is tagged `@pytest.mark.integration` and skips automatically without credentials.

### Class: `TestEndToEnd` — 7 tests

These tests invoke the real orchestrator and assert on the **complete agent state**, not just message text.

---

#### `test_login_issue_resolves_with_kb_retrieval`

**Input:** `"I forgot my CultPass password. What are the steps to reset it?"`

**What it asserts:**

| Field | Expected |
|---|---|
| `next_action` | `"end"` |
| `retrieval_quality` | `"high"` or `"medium"` |
| `confidence` | ≥ 0.4 |
| `classification` | `"technical"` or `"account"` |
| Response content | contains `"login"` / `"password"` / `"reset"` / `"forgot"` / `"account"` |

**Why:** Verifies the full resolution pipeline — Supervisor routes to Classifier, Classifier labels as technical/low, `route_after_classifier` sends to Resolver, Resolver retrieves a KB article and generates a grounded answer with adequate confidence.

---

#### `test_subscription_cancel_query_resolves_with_tool`

**Input:** `"How do I cancel my CultPass subscription?"`

**What it asserts:** `next_action == "end"`, `confidence ≥ 0.4`, response mentions subscription/cancel/account/pause.

**Why:** Verifies a billing-category resolution path. Subscription queries must reach the Resolver (not direct-escalate), and the answer must reference the relevant KB article or tool output.

---

#### `test_greeting_handled_directly_without_classification`

**Input:** `"Hello!"`

**What it asserts:** Final message is an `AIMessage`; `classification` is empty, `"general"`, or `None`.

**Why:** The Supervisor must short-circuit the pipeline for greetings — no Classifier, no Resolver, no Escalator invoked. Tests the `route_after_supervisor → "end"` branch.

---

#### `test_unknown_topic_explicitly_escalates`

**Input:** `"My blockchain NFT wallet integration with CultPass is broken and I need a refund for 50 transactions"`

**What it asserts:** Either `next_action == "escalate"` OR `"UDA-"` appears in the final message (both accepted — direct escalation or resolver-escalation both produce a UDA ref).

**Why:** Confirms off-topic or unresolvable queries trigger the Escalator rather than a hallucinated answer. The reviewer specifically required this test to verify the Escalator is *invoked*, not merely that a response is returned.

---

#### `test_high_urgency_account_issue_escalates`

**Input:** `"I think my account has been hacked — someone changed my email and I can't access it"`

**What it asserts:** `"UDA-"` in response; response contains escalation language (`"escalat"` / `"team"` / `"specialist"` / `"24 hour"` / `"follow"`).

**Why:** Verifies the `route_after_classifier` rule: `urgency == "high"` + `category == "account"` → `direct_escalate`, bypassing the Resolver entirely. Security/account-breach issues must never be handled by a KB lookup alone.

---

#### `test_deterministic_escalation_on_no_kb_match`

**Input:** `"I need help with my CultPass smart TV screen mirroring and Bluetooth speaker pairing"`

**What it asserts:** Either `"UDA-"` in response OR `next_action == "escalate"`.

**Why:** A query with zero KB-keyword overlap (no billing/login/account/subscription/reservation terms) must escalate rather than return a fabricated answer. Tests the ESCALATION_MARKER path or low-confidence LLM path.

---

#### `test_multi_turn_conversation_accumulates_context`

**Input:** Turn 1: `"How do I reset my password?"` → Turn 2: `"What if I still don't receive the email?"` (same `thread_id`)

**What it asserts:** Both turns return `AIMessage`; Turn 1 `next_action == "end"`; Turn 2 message count is greater than Turn 1 (thread history accumulated).

**Why:** Verifies short-term memory (`MemorySaver` checkpointer). A follow-up question in the same thread must have access to the prior exchange without the customer repeating context.

---

### Class: `TestCompleteTicketWorkflow` — 4 tests

These tests use a `_LogCapture` handler to collect structured log records during the live workflow and assert that the expected agent/event/status combinations were emitted.

---

#### `test_resolution_workflow_complete_pipeline`

**Input:** `"I forgot my CultPass password. What are the steps to reset it?"`

**State assertions:** `next_action == "end"`, `retrieval_quality` in `("high", "medium")`, `confidence ≥ 0.4`.

**Log assertions:**

| Log event required | `agent` | `event` | `status` |
|---|---|---|---|
| Supervisor routed | `supervisor` | `route` | `success` |
| Resolver resolved | `resolver` | `resolve` | `success` |
| KB tool called | `resolver` | `tool_call` | `success` (tool_name = `search_knowledge_base`) |

Every captured record must also contain `timestamp`, `level`, `agent`, `event`, `status`.

**Why:** The most complete single-test verification of the system. Checks not just the output but that the correct agents fired, in the correct order, and emitted the correct log events.

---

#### `test_escalation_workflow_complete_pipeline`

**Input:** `"Someone has taken over my account and I cannot get back in — this is urgent"`

**State assertions:** `"UDA-"` in response; response contains escalation language.

**Log assertions:**

| Log event required | `agent` | `event` | `status` |
|---|---|---|---|
| Supervisor routed | `supervisor` | `route` | `success` |
| Escalator fired | `escalator` | `escalate` | `escalated` |

**Why:** Verifies the complete escalation audit trail. The Escalator must have been *invoked* (not just the output inspected) — confirmed by the `escalator/escalate` log event.

---

#### `test_tool_integration_subscription_workflow`

**Input:** `"My email is bob.stone@granite.com. Could you walk me through how to cancel my subscription?"`

**State assertions:** `next_action == "end"`.

**Log assertions:** At least one `resolver/tool_call` event; at least one tool name is `get_user_account` or `manage_subscription`; `resolver/resolve` event with `status == "success"`.

**Why:** When a customer's email appears in the message, the Resolver must perform an account lookup. Verifies that the email extraction, tool dispatch, and account-tool logging all work end-to-end with a real customer in `cultpass.db`.

---

#### `test_deterministic_escalation_logs_kb_failure`

**Input:** `"I need help configuring my CultPass interplanetary travel portal"`

**State assertions:** `"UDA-"` in response OR `next_action == "escalate"`.

**Log assertions:** At least one resolver escalation event — either `event="escalate"` (ESCALATION_MARKER path) or `event="resolve"` with `status="escalated"` (low-confidence LLM path).

**Why:** Verifies the escalation audit trail for the Resolver-initiated path. Unlike the direct-escalation tests (which bypass the Resolver), this confirms that the Resolver itself correctly emits a traceable escalation event when it cannot confidently answer.

---

### Running the integration suite

```bash
# All 11 integration tests
py -m pytest tests/test_workflow.py -m "integration" -v

# Unit tests only (no API key needed)
py -m pytest tests/test_workflow.py -m "not integration" -v

# Full suite (68 tests)
py -m pytest tests/test_workflow.py -v
```

After running, inspect `logs/udahub.log` for the complete structured JSON audit trail of every agent decision, tool call, and escalation event from the session.

---

## What NOT to Use as an Eval

- **Response length** — a long answer is not necessarily a correct one
- **Absence of errors** — no exception ≠ correct behaviour
- **LLM confidence alone** — the LLM may be confidently wrong; combine with retrieval signals
- **Single-turn tests for memory** — memory must be tested across turns and sessions to be meaningful
