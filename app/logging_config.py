"""
Structured JSON logging.

Every agent step logs a single JSON line with a shared request_id so a whole
query's trace can be reconstructed with `grep request_id`. No API keys or
secrets are ever logged (see redact()).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_SENSITIVE_KEYS = {"api_key", "openai_api_key", "hf_token", "authorization"}


def redact(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        k: ("***REDACTED***" if k.lower() in _SENSITIVE_KEYS else v)
        for k, v in payload.items()
    }


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(redact(extra))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    logger.info(message, extra={"extra_fields": fields})
