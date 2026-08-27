"""Structured JSON logging with conservative secret redaction."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any


def _secrets() -> tuple[str, ...]:
    names = {"BDL_API_KEY"}
    return tuple(v for n in names if (v := os.getenv(n)))


def _redact(value: Any) -> Any:
    secrets = _secrets()
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


class _JSONFormatter(logging.Formatter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": self.run_id,
            "message": _redact(record.getMessage()),
        }
        if record.exc_info:
            payload["exception"] = _redact(self.formatException(record.exc_info))
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def configure(run_id: str, level: str = "INFO") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter(run_id))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
