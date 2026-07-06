"""Stdlib-based JSON-line logger factory for PINN-AURA-MFP.

Usage::

    from src.utils.logging import get_logger
    log = get_logger(__name__)
    log.info("starting epoch", extra={"epoch": 1})
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_CONFIGURED: set[str] = set()
_ROOT_HANDLER_INSTALLED = False

# Standard ``LogRecord`` attributes we must not emit as payload fields.
_RESERVED: frozenset[str] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "taskName",
    }
)


class JsonLineFormatter(logging.Formatter):
    """Format each record as a single JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - stdlib override
        """Return the record serialised as a JSON line."""
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            try:
                json.dumps(value)
            except TypeError:
                value = repr(value)
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def _install_root_handler() -> None:
    global _ROOT_HANDLER_INSTALLED
    if _ROOT_HANDLER_INSTALLED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonLineFormatter())
    root = logging.getLogger("pinn_aura_mfp")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False
    _ROOT_HANDLER_INSTALLED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger attached to the JSON-line root handler.

    Args:
        name: Module name; typically ``__name__``.

    Returns:
        A configured :class:`logging.Logger`.
    """
    _install_root_handler()
    logger_name = f"pinn_aura_mfp.{name}"
    logger = logging.getLogger(logger_name)
    if logger_name not in _CONFIGURED:
        logger.setLevel(logging.INFO)
        _CONFIGURED.add(logger_name)
    return logger
