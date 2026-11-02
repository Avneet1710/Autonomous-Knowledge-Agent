"""Long-term memory repository — cross-session resolved-issue storage in udahub.db."""
import json
import re
import uuid

from .base import get_connection

# Must match the account_id inserted by 02_core_db_setup.ipynb.
_ACCOUNT_ID = "cultpass"

# Regex that matches common email addresses in customer messages.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def save_memory(
    db_path: str,
    thread_id: str,
    customer_id: str,
    category: str,
    summary: str,
) -> None:
    """Persist a resolved-issue summary to long_term_memory."""
    content = json.dumps(
        {"customer_id": customer_id, "category": category, "summary": summary}
    )
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO long_term_memory
                (memory_id, thread_id, account_id, memory_type, content)
            VALUES (?, ?, ?, 'resolved_issue', ?)
            """,
            (uuid.uuid4().hex[:12], thread_id, _ACCOUNT_ID, content),
        )


def get_customer_memories(db_path: str, customer_id: str, limit: int = 3) -> list[str]:
    """Return the most recent resolved-issue summaries for a given customer_id."""
    with get_connection(db_path) as conn:
        # Over-fetch and filter in Python because customer_id lives inside JSON content.
        rows = conn.execute(
            """
            SELECT content FROM long_term_memory
            WHERE memory_type = 'resolved_issue'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit * 5,),
        ).fetchall()

    results: list[str] = []
    for row in rows:
        try:
            data = json.loads(row["content"])
            if data.get("customer_id") == customer_id:
                results.append(data.get("summary", ""))
                if len(results) >= limit:
                    break
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def extract_customer_id(text: str) -> str:
    """Extract an email address from a customer message, or return empty string."""
    match = _EMAIL_RE.search(text)
    return match.group(0) if match else ""
