# UDA-Hub — Project Document

> A plain-language explanation of what the project is, what was built, how it works, and why each agentic concept was chosen.

---

## 1. The Problem

**CultPass Card** is a subscription service that gives members access to cultural experiences — museums, concerts, film screenings, art exhibitions — a fixed number of times per month.

When members have problems (can't log in, confused about billing, want to cancel, can't book an experience), they contact customer support. Traditionally, a human agent reads the message, figures out what kind of problem it is, looks up the customer's account, searches a knowledge base for an answer, and either resolves the issue or escalates it to a specialist.

**The project goal:** build an AI system — UDA-Hub — that does this entire job automatically, step by step, just like a human support team would.

---

## 2. What Was Built

UDA-Hub is a **multi-agent customer support system**. Think of it like a small team of four AI specialists, each with a specific job, working together to handle any support message a CultPass member sends:

| Team member | Job |
|---|---|
| **Supervisor** | Reads the message first. Decides: is this a real support question, or just a greeting? Routes accordingly. |
| **Classifier** | Reads the support question and labels it — what type of problem is it? How urgent? |
| **Resolver** | Does the actual work. Looks up relevant articles, checks the customer's account, and writes a helpful answer. |
| **Escalator** | Steps in when the Resolver isn't confident enough. Writes an empathetic message telling the customer a human will follow up, and assigns a reference number. |

The system also has:
- **A knowledge base** — 14 support articles covering every common CultPass issue
- **Account tools** — look up a customer, check their reservations, pause or cancel a subscription
- **Memory** — the system remembers what was said earlier in a conversation so it doesn't ask the customer to repeat themselves

---

## 3. How It Works — Step by Step

Imagine a customer sends this message:

> *"I can't log in to my CultPass account. I've tried resetting my password but never received the email."*

Here is exactly what happens inside UDA-Hub:

### Step 1 — Supervisor reads the message
The Supervisor looks at the message and asks: *"Is this a support request?"* Yes — it's not a greeting or a joke. It decides to send it for classification.

### Step 2 — Classifier labels the problem
The Classifier reads the message and says: *"This is a **technical** problem, urgency **medium**."* It stores this label in the system's shared memory so every agent that follows knows what kind of problem they're dealing with.

### Step 3 — Resolver searches for an answer
The Resolver does two things:
1. **Searches the knowledge base** — it finds the article *"How to Handle Login Issues"* which covers exactly this scenario.
2. **Checks the customer's account** if they provided an email or ID — to see if their account is blocked.

It then writes a helpful answer based on what it found, and gives itself a **confidence score** between 0 and 1. If confidence is above 0.4, it sends the answer directly.

### Step 4 (only if needed) — Escalator takes over
If the Resolver's confidence is below 0.4 — meaning it couldn't find a good enough answer — the Escalator writes instead:

> *"We're sorry to hear you're having trouble. Your case (UDA-TEC-MED) has been passed to our support team, who will follow up within 24 hours."*

---

## 4. The Databases

The system uses two databases:

**CultPass database** — represents the real CultPass platform:
- 6 member accounts (some blocked, some active)
- 7 cultural experiences with dates and locations
- 6 subscriptions (basic and premium, active and cancelled)
- 8 reservations spread across non-blocked members

**UDA-Hub database** — the support system's own storage:
- 14 knowledge articles covering every common support topic
- Ticket and message history
- Long-term memory (for storing what was resolved in past conversations)

---

## 5. Agentic Concepts Used — And Why

This section explains every AI/agent concept used in the project, what it means, why it was used here, and when you'd use it in any other project.

---

### Concept 1 — Multi-Agent Systems

**What it means:** Instead of one big AI doing everything, you have several smaller AIs ("agents"), each responsible for one specific job.

**Why used here:** Customer support has distinct phases — understanding the problem, classifying it, finding an answer, escalating it. Each phase benefits from a focused agent rather than asking a single AI to do all of it at once. A single AI often confuses these steps or skips one.

**When to use:** Whenever a task has clearly distinct phases, or when different parts of the task need different "expertise." Also use it when one AI's output becomes another AI's input.

**When NOT to use:** For simple, single-step tasks where one AI call is enough (e.g., "translate this sentence").

---

### Concept 2 — Supervisor Pattern

**What it means:** One agent acts as the "boss" — it receives every message first, decides what kind of work it is, and routes it to the right specialist. It doesn't do the actual work itself.

**Why used here:** Not every message needs the full classification + resolution pipeline. A greeting like "Hello!" should be answered directly without running through the entire system. The Supervisor acts as a smart traffic controller.

**When to use:** When your system receives diverse inputs and different inputs need completely different handling paths. The Supervisor keeps the routing logic clean and centralised.

---

### Concept 3 — StateGraph (LangGraph)

**What it means:** A "graph" is a map of how agents connect to each other. LangGraph's StateGraph lets you draw this map in code — which agent goes first, which goes next, and under what conditions.

**Why used here:** The flow through UDA-Hub is not linear. Depending on what the Supervisor and Resolver decide, the message takes different paths. A StateGraph makes these branching paths explicit and testable.

**When to use:** Any time you have conditional, multi-step workflows where the next step depends on the result of the previous one. Think of it like a flowchart — if you'd draw a flowchart for the task, you need a StateGraph.

---

### Concept 4 — Shared State (TypedDict)

**What it means:** All agents in the workflow share a common "notepad" — a dictionary that holds information gathered so far: the messages exchanged, the classification, the urgency level, the confidence score, and so on.

**Why used here:** The Resolver needs to know what the Classifier decided. The Escalator needs to know what the Resolver attempted. Without shared state, each agent would start from scratch and lose all context.

**When to use:** In any multi-agent system where later agents need information produced by earlier agents. Shared state is the backbone of agent coordination.

---

### Concept 5 — Tool Calling

**What it means:** An AI agent can call external functions ("tools") mid-conversation — like looking up a database, calling an API, or running a calculation — and use the result in its response.

**Why used here:** The Resolver can't answer "What is my subscription status?" from memory alone — it needs to look it up in the real CultPass database. Tool calling lets it pause, fetch the real data, and then write an accurate answer.

Tools implemented:
- `search_knowledge_base` — find relevant support articles
- `get_user_account` — look up a customer's account and subscription
- `get_user_reservations` — see a customer's upcoming bookings
- `manage_subscription` — pause or cancel a subscription

**When to use:** Whenever the AI needs real-world data it doesn't have in its training — databases, APIs, files, calendars. Without tool calling, the AI makes things up ("hallucination"). Tool calling grounds its answers in facts.

---

### Concept 6 — RAG (Retrieval-Augmented Generation)

**What it means:** Instead of asking an AI to answer from its own memory (which may be wrong or outdated), you first *retrieve* relevant documents from your own knowledge base, then *give those documents to the AI* so it answers based on them.

**Why used here:** The knowledge base contains CultPass-specific support articles. The AI doesn't know CultPass's policies by default — it needs to read the right article first. RAG ensures answers are grounded in company policy, not guesswork.

How the retrieval works in UDA-Hub:
1. Break the customer's question into keywords
2. Score every article: title match = 3 points, tag match = 2 points, content match = 1 point
3. Return the top 3 articles above a minimum score
4. Give those articles to the Resolver as context

**When to use:** Any time you have a private or frequently-updated knowledge source (company docs, product manuals, legal policies, FAQs). RAG is the standard solution to "how do I make an AI answer from MY data, not its training?"

---

### Concept 7 — Structured Output (Pydantic)

**What it means:** Instead of asking an AI to return a freeform essay, you ask it to return data in a specific, predictable format — like a form with fields. Pydantic enforces that the fields are the right type.

**Why used here:** The Classifier must return exactly `{category: "billing", urgency: "high"}` — not a paragraph explaining its thinking. The Resolver must return `{response: "...", confidence: 0.8, needs_escalation: false}`. Structured output makes the AI's decisions machine-readable so the code can act on them.

**When to use:** Any time you need to extract a decision or data point from an AI response and use it programmatically. If you'd normally parse the AI's text to extract a value, use structured output instead.

---

### Concept 8 — Short-Term Memory (MemorySaver + thread_id)

**What it means:** Within a single conversation, the system remembers everything that was said. If a customer asks "I can't log in" and then follows up with "I still haven't received the reset email", the AI knows the follow-up is about the login problem without being told again.

**Why used here:** Customer support conversations are multi-turn by nature. Without memory, every message would be treated as a brand-new, isolated request.

**How it works:** LangGraph's `MemorySaver` stores the entire conversation state. Each session is identified by a `thread_id` — use the same thread_id to continue a conversation, use a different one to start fresh.

**When to use:** Any conversational AI application where the user might send follow-up messages, ask clarifying questions, or refer back to earlier parts of the conversation.

---

### Concept 9 — Long-Term Memory (SQLite persistence)

**What it means:** Unlike short-term memory (which is lost when the session ends), long-term memory saves important information to a database — so the AI remembers a returning customer even in a brand-new conversation.

**Why used here:** If a customer previously had their subscription cancelled and contacted support, that history should be available the next time they contact support — even days later. The AI shouldn't ask "have you contacted us before?" when it already knows.

**What is stored:** Resolved issue summaries (category + a 250-character digest of the resolver's answer), keyed by customer email or user_id, stored in the `long_term_memory` table of `udahub.db`.

**How it works end-to-end:**
1. **Save** — After the Resolver produces a successful answer (confidence ≥ 0.4, not escalated), it extracts the `user_identifier` from any account-tool call made during resolution and calls `save_memory()` in `data/repositories/ltm_repo.py`. The summary is stored as JSON in the `content` column.
2. **Retrieve** — At the start of each new session, the Supervisor scans the customer's message for an email address. If found, it calls `get_customer_memories()` to load prior resolved issues and injects them as `long_term_context` into the agent state. The Resolver's Phase 1 system prompt then includes this context so the LLM has background on the customer.

**When to use:** Any customer-facing system where personalization matters. Also useful in copilots, tutors, or personal assistants that should "know" the user over time.

---

### Concept 10 — Confidence Scoring and Escalation

**What it means:** The Resolver doesn't just produce an answer — it also rates how confident it is in that answer (0.0 = not confident at all, 1.0 = completely certain). If confidence falls below a threshold, the system automatically escalates to human support.

**Why used here:** Not all support questions can be answered from the knowledge base. A customer reporting account fraud or a complex billing dispute needs a human. Forcing the AI to answer everything would result in wrong or harmful responses.

**When to use:** Any AI system that operates in a high-stakes or sensitive domain (finance, healthcare, legal, customer support). The AI should know the limits of its own knowledge and hand off gracefully rather than guessing.

---

### Concept 11 — Deterministic Escalation Signals

**What it means:** Instead of always relying on the LLM to decide when to escalate (which it might get wrong), certain conditions trigger escalation automatically in code — before the LLM is even consulted.

**Why used here:** The LLM's confidence score is useful but not infallible — it might report high confidence on a hallucinated answer, or low confidence on a correct one. For the case where the knowledge base returns *no relevant articles*, we don't need the LLM's opinion. Escalating immediately is always the right call, and doing so in code makes it testable without mocking the LLM.

In UDA-Hub: the `search_knowledge_base` tool embeds an `<<ESCALATION_REQUIRED>>` marker in its response when no articles meet the relevance threshold. The Resolver detects this marker and escalates without making a second LLM call.

**When to use:** Any time there is an objective, unambiguous condition that means "a human is needed" — empty retrieval, a blocked account flag, a confirmed fraud signal from an external system. Objective conditions should be handled in code; subjective conditions (is the answer good enough?) can be handled by the LLM.

---

### Concept 12 — Repository / Database Abstraction Layer

**What it means:** Instead of letting tools (or any business logic) open database connections and run SQL directly, you create a dedicated "repository" module. The repository owns all database access; the tools simply call it and get back clean Python objects.

**Why used here:** The three account tools (`get_user_account`, `get_user_reservations`, `manage_subscription`) all needed to talk to the CultPass SQLite database. Without a repository layer, each tool embedded raw `sqlite3` calls — mixing business logic with infrastructure. That made the code hard to test (tests had to use the real DB every time), hard to change (swap SQLite → Postgres meant editing every tool), and fragile (no centralised error handling or connection management).

The repository layer introduced:
- `base.py` — a single `get_connection()` context manager that handles commit, rollback, and connection close automatically
- `account_repo.py` — `get_user()` returns a `UserAccount` dataclass, not a raw tuple
- `reservation_repo.py` — `get_reservations()` returns `list[Reservation]`
- `subscription_repo.py` — `update_subscription()` returns a `SubscriptionUpdate` dataclass with a `success` flag and a human-readable `message`

The tools now call these functions and format the results into strings for the LLM — no SQL anywhere in tool code.

**When to use:** Any time a tool, agent, or service needs to read or write a database. The repository pattern is especially valuable when:
- The same data is accessed by more than one tool
- You want to test business logic without a real database (mock the repo, not the DB driver)
- You may change the underlying database technology later
- You need centralised error handling (e.g., connection failures, missing rows)

**When NOT to use:** For truly one-off scripts or throw-away prototypes where the overhead of a separate module is more cost than benefit. Also unnecessary if you're using a full ORM (SQLAlchemy) that already provides this abstraction.

---

### Concept 13 — Structured Logging

**What it means:** Instead of printing plain text like `"Resolver called tool"`, each agent emits a structured JSON log record — a machine-readable dictionary that always contains the same fields: `timestamp`, `level`, `agent`, `event`, `tool_name`, `status`, `thread_id`, `message`. Every run also writes a persistent log file (`logs/udahub.log`) so the full audit trail is available after the session ends.

**Why used here:** In a multi-agent system, logs from four different agents intermingle. A plain text log like `"Escalated ticket UDA-BIL-MED"` is readable to a human but useless to a monitoring dashboard, a log-aggregation tool (DataDog, CloudWatch, Splunk), or an alert rule that needs to count escalations by category. Structured JSON logs make every field queryable without regex parsing. Persisting them to a file means you can replay, audit, or grep a full conversation history even after the Python process has exited.

How it works in UDA-Hub:
- `agentic/logging_config.py` defines `JsonFormatter` — a Python `logging.Formatter` subclass that formats every `LogRecord` as a single-line JSON object
- `configure_logging(log_file="logs/udahub.log")` attaches both a `StreamHandler` (stderr) and a `FileHandler` (the log file) to the root logger; `workflow.py` calls it once at startup
- The **first event logged is the user's query** (`event="user_query"`, `status="received"`) so the full input→output chain is traceable in a single file
- Each agent passes an `extra={}` dict to every log call with agent-specific fields: `agent`, `event`, `tool_name` (optional), `status`, `thread_id`
- The formatter picks these up from the `LogRecord` and includes them only when they are set (no `null` fields in the output)

Example — a complete password-reset workflow in `logs/udahub.log`:
```json
{"timestamp": "2026-11-02T10:42:33.801+00:00", "level": "INFO", "agent": "supervisor", "event": "user_query", "status": "received", "thread_id": "session-abc123", "message": "User query: I forgot my CultPass password"}
{"timestamp": "2026-11-02T10:42:34.120+00:00", "level": "INFO", "agent": "supervisor", "event": "route", "status": "success", "thread_id": "session-abc123", "message": "Supervisor routed to classify"}
{"timestamp": "2026-11-02T10:42:35.880+00:00", "level": "INFO", "agent": "classifier", "event": "classify", "status": "success", "message": "Classified as technical / urgency=low / complexity=simple"}
{"timestamp": "2026-11-02T10:42:37.210+00:00", "level": "INFO", "agent": "resolver", "event": "tool_call", "tool_name": "search_knowledge_base", "status": "success", "thread_id": "session-abc123", "message": "Tool invoked: search_knowledge_base"}
{"timestamp": "2026-11-02T10:42:39.560+00:00", "level": "INFO", "agent": "resolver", "event": "resolve", "status": "success", "thread_id": "session-abc123", "message": "Resolver: confidence=0.85, escalate=False, retrieval_quality=high"}
```

**When to use:** Any production AI system, API service, or multi-step pipeline where you need to debug, monitor, or alert on behaviour at scale. Structured logging is non-negotiable once you have more than one agent or more than one instance of your service.

**When NOT to use:** In a local development script or a notebook where human-readable output is more useful than machine-parseable JSON. You can omit the `log_file` argument to `configure_logging()` to suppress the file and keep logs on stderr only.

---

### Concept 14 — Agentic System Evaluation (Evals)

**What it means:** Deliberately testing that an AI agent behaves correctly across a range of scenarios — not just that it "runs without crashing", but that it routes correctly, retrieves the right information, answers based on evidence, and escalates when appropriate.

**Why used here:** A support agent that gives wrong answers confidently is more harmful than one that crashes. Standard unit tests catch code errors; evals catch *reasoning* errors. UDA-Hub has five evaluation dimensions: routing correctness, retrieval quality, response groundedness, resolution accuracy, and escalation precision. See [EVALS.md](EVALS.md) for the full evaluation framework.

**When to use:** Evals are essential for any AI system where the output is consequential — support, healthcare, finance, legal. The rule of thumb: if a wrong answer would cause harm or frustration, you need evals, not just tests. Fast deterministic evals run on every commit; slower LLM-in-the-loop evals run on every pull request.

---

---

## 6. Technology Choices

| What | Tool | Why |
|---|---|---|
| Agent orchestration | LangGraph StateGraph | Custom control over every routing decision; no black-box prebuilt agents |
| LLM | GPT-4o-mini via Vocareum | Fast, affordable, supports tool calling and structured output |
| Databases | SQLite via SQLAlchemy ORM | Simple, portable, no server needed — perfect for a course project |
| DB access layer | Repository pattern (dataclasses) | Decouples tools from SQL; enables unit testing without a real DB |
| Tool binding | LangChain `@tool` + `bind_tools()` | Standard way to give LLMs access to Python functions |
| Structured output | Pydantic + `with_structured_output()` | Guarantees machine-readable agent decisions |
| Memory | LangGraph MemorySaver + SQLite | Short-term via checkpointer, long-term via database |
| Testing | pytest | Standard Python testing; supports both unit and integration tests |

---

## 7. What the Project Demonstrates

By building UDA-Hub, the following skills were demonstrated:

- Designing a multi-agent system from scratch using a **Supervisor Pattern**
- Implementing a **conditional LangGraph workflow** with branching paths
- Grounding AI responses in a private knowledge base using **RAG**
- Enabling AI to take real actions (account lookups, subscription changes) using **tool calling**
- Making AI decisions machine-readable using **structured output**
- Maintaining conversation context using **short-term memory**
- Persisting customer history across sessions using **long-term memory**
- Routing based on urgency and category so high-priority issues get priority handling
- Knowing when to escalate using **confidence scoring** and **deterministic escalation signals**
- Separating database access into a **repository layer** so tools are testable and infrastructure-agnostic
- Persisting resolved-issue summaries to **long-term memory** and retrieving them for returning customers across sessions
- Emitting **structured JSON logs** from every agent with `agent`, `event`, `tool_name`, `status`, and `thread_id` fields
- Testing graceful **error handling** — tools return safe error strings or escalation markers when databases are unavailable, agents fall back to safe defaults when the LLM raises
- Evaluating agent behaviour beyond "did it return something" — using retrieval precision, grounding checks, and routing correctness tests
