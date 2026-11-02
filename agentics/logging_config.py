"""Structured JSON logging for UDA-Hub.

Every log record emitted by any agent includes:
  timestamp  — ISO-8601 UTC
  level      — DEBUG | INFO | WARNING | ERROR
  agent      — which agent produced the log (supervisor, classifier, resolver, escalator)
  event      — what happened (route, classify, tool_call, resolve, escalate, error, …)
  tool_name  — the LangChain tool invoked, when applicable
  status     — success | error | escalated | skipped
  thread_id  — LangGraph session identifier from config
  message    — human-readable description

Usage in agent modules:
    logger.info(
        "Tool invoked",
        extra={"agent": "resolver", "event": "tool_call",
               "tool_name": "get_user_account", "status": "success",
               "thread_id": thread_id},
    )

Call configure_logging() once at application start (workflow.py does this).
Tests that import agents without configuring logging will see plain text output,
which is fine — the formatter is applied only when a handler is attached.
"""

import json
import logging
import datetime


class JsonFormatter(logging.Formatter):
    """Format every LogRecord as a single-line JSON object."""

    # Fields pulled from the LogRecord's `extra` dict (or defaults)
    _EXTRA_FIELDS = ("agent", "event", "tool_name", "status", "thread_id")

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in self._EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO, log_file: str = None) -> None:
    """Attach the JsonFormatter to the root logger.

    Writes structured JSON to stderr (always) and, when log_file is given,
    to that file as well (one JSON object per line, appended).

    Safe to call multiple times — existing handlers are replaced so the
    formatter is not added twice when the module is reloaded in notebooks.
    """
    import os

    formatter = JsonFormatter()

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        handlers.append(file_handler)

    for h in handlers:
        h.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)

    # Suppress noisy third-party loggers
    for noisy in ("httpcore", "httpx", "openai", "langsmith"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
