"""Structured audit logging for dose builds (D7.3).

Emits one JSONL record per dose / influence build with enough fields
that an operator can answer:

* Who triggered this build? (api_key_prefix, request_id)
* What did it touch? (geometry_id, beam_model_id, engine, modality)
* How did it go? (state, duration_s, dose_id, cache_hit, error)
* When? (timestamp, ISO-8601 UTC)

The audit log is **separate** from the regular loguru stderr stream
so it can be shipped to a log-aggregator (Splunk, Datadog, ELK)
without dragging in debug noise. The format is JSONL — one
``json.dumps`` per line — so it parses cleanly with ``jq`` and
streaming-tail aggregators.

Configuration: ``RADIARCH_AUDIT_LOG_PATH``. Empty (default) means
audit-only sink to stderr (alongside loguru). Set to a file path
for production.

Concurrency: file writes are atomic at the OS level for lines below
PIPE_BUF (4096 bytes on Linux). A typical audit record is ~300
bytes, so concurrent Celery workers can write to the same file
without locks. We don't fsync — operators who need fsync-per-line
durability should ship to a real log aggregator.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

from loguru import logger

from ..config import get_settings


_LOCK = threading.Lock()


@dataclass
class AuditEvent:
    """One audit record. All fields are JSON-serializable."""

    event_id: str
    timestamp: str
    event_type: str  # "dose.compute", "dose.influence", "dose.delete", ...
    state: str  # "started", "succeeded", "failed", "cache_hit"
    cache_key: Optional[str] = None
    geometry_id: Optional[str] = None
    beam_model_id: Optional[str] = None
    engine_name: Optional[str] = None
    engine_version: Optional[str] = None
    modality: Optional[str] = None
    scenario_count: Optional[int] = None
    nb_primaries: Optional[float] = None
    dose_id: Optional[str] = None
    influence_id: Optional[str] = None
    duration_s: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    api_key_prefix: Optional[str] = None
    request_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def emit(event: AuditEvent) -> None:
    """Write one audit record. Never raises (audit must not break the app)."""
    try:
        payload = {k: v for k, v in asdict(event).items() if v is not None and v != {}}
        line = json.dumps(payload, separators=(",", ":"), default=str)
    except Exception as exc:  # never trust an event
        logger.error(f"audit.emit: failed to serialize event: {exc}")
        return

    # Stderr sink (always). Loguru handles the rotation if it's
    # configured to a file.
    logger.bind(audit=True).info(line)

    # Optional file sink — JSONL.
    settings = get_settings()
    path = settings.audit_log_path
    if not path:
        return

    try:
        # Ensure directory exists; cheap idempotent op.
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with _LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:  # pragma: no cover
        # Don't let audit failures kill a dose build. Surface to the
        # regular log so ops can fix the file-perms issue.
        logger.error(f"audit.emit: file sink failed for {path}: {exc}")


def make_event(event_type: str, **kwargs) -> AuditEvent:
    """Construct an event with defaulted bookkeeping fields."""
    return AuditEvent(
        event_id=str(uuid.uuid4()),
        timestamp=_now_iso(),
        event_type=event_type,
        state=kwargs.pop("state", "started"),
        **kwargs,
    )


@contextmanager
def audit_span(event_type: str, **start_fields) -> Iterator[Dict[str, Any]]:
    """Context manager — emits a started + finished pair.

    Usage::

        with audit_span("dose.compute", geometry_id=g, beam_model_id=b) as ctx:
            result = service.compute_dose(...)
            ctx["dose_id"] = result.dose_id
            ctx["cache_hit"] = False

    On normal exit emits a "succeeded" event. On exception emits
    "failed" with the exception type + message. Either way the
    duration_s field is populated.
    """
    t0 = time.monotonic()
    start = make_event(event_type, state="started", **start_fields)
    emit(start)

    ctx: Dict[str, Any] = dict(start_fields)
    try:
        yield ctx
    except Exception as exc:
        end = make_event(
            event_type,
            state="failed",
            duration_s=round(time.monotonic() - t0, 4),
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
            **ctx,
        )
        emit(end)
        raise
    else:
        end = make_event(
            event_type,
            state=ctx.pop("state", "succeeded"),
            duration_s=round(time.monotonic() - t0, 4),
            **ctx,
        )
        emit(end)


__all__ = ["AuditEvent", "audit_span", "emit", "make_event"]
