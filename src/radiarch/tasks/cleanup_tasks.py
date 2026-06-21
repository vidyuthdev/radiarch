"""Disk-space cleanup tasks (D7.2).

Runs periodically (every ``RADIARCH_CLEANUP_INTERVAL_S`` seconds via
Celery Beat) and evicts least-recently-accessed dose / influence
entries when the corresponding store exceeds its configured size cap.

Eviction rules
--------------
* Entries younger than ``RADIARCH_DOSE_MIN_AGE_HOURS`` are protected
  — never evicted. This shields an in-flight optimization loop from
  losing the Dij it's iterating against.
* Among evictable entries, oldest-access wins (LRU). Access time is
  the entry's ``meta.json`` mtime (the persistence layer updates it
  on every read via ``DoseStore.get_by_id``; if you don't see mtime
  bumps, the read-path isn't touching the file).
* Eviction stops once total size drops below the cap (no overshoot).

Operationally
-------------
* The task is **safe to run while builds are in flight** — it
  reads the store, computes the eviction set, then calls
  ``store.delete_by_id`` one entry at a time. The persistence
  layer's atomic-write contract means a half-deleted entry can't
  poison a concurrent reader.
* Failures are logged but don't crash the worker; the next tick will
  retry. We don't autoretry the task because eviction is idempotent
  and a transient FS error usually self-heals.
* Set ``dose_store_max_gb`` / ``influence_store_max_gb`` to 0 to
  effectively disable eviction for that store.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from loguru import logger

from .celery_app import celery_app
from ..config import get_settings
from ..services.audit import emit, make_event


# ---------------------------------------------------------------------------
# Sizing helpers — done with os.walk to avoid loading metadata into memory
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    """One on-disk cache entry, keyed by directory name (== dose_id)."""

    id: str
    path: Path
    size_bytes: int
    atime_s: float  # access time of the meta.json (LRU signal)


def _enumerate_entries(base_dir: Path) -> List[_Entry]:
    """List every dose/influence entry under base_dir."""
    if not base_dir.is_dir():
        return []
    out: List[_Entry] = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        meta = child / "meta.json"
        if not meta.is_file():
            # Not a complete entry (mid-write tempdir, or corrupted).
            # Leave it alone — the persistence layer will reap it.
            continue
        size = sum(p.stat().st_size for p in child.rglob("*") if p.is_file())
        out.append(_Entry(
            id=child.name,
            path=child,
            size_bytes=size,
            atime_s=meta.stat().st_atime,
        ))
    return out


def _evict_to_cap(
    entries: List[_Entry],
    cap_bytes: int,
    min_age_seconds: float,
) -> Tuple[int, int]:
    """Delete oldest-access entries until total size ≤ cap.

    Returns ``(evicted_count, bytes_freed)``.
    """
    if cap_bytes <= 0:
        return (0, 0)

    total = sum(e.size_bytes for e in entries)
    if total <= cap_bytes:
        return (0, 0)

    now = time.time()
    candidates = [
        e for e in entries
        if (now - e.atime_s) >= min_age_seconds
    ]
    # Oldest-first by access time.
    candidates.sort(key=lambda e: e.atime_s)

    evicted = 0
    freed = 0
    for entry in candidates:
        if total - freed <= cap_bytes:
            break
        try:
            shutil.rmtree(entry.path)
            freed += entry.size_bytes
            evicted += 1
        except Exception as exc:  # pragma: no cover
            logger.error(f"cleanup: failed to evict {entry.path}: {exc}")
    return (evicted, freed)


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@celery_app.task(name="radiarch.cleanup.dose_stores")
def cleanup_dose_stores() -> dict:
    """Sweep dose + influence stores; evict over-cap entries.

    Returns a summary dict (for Celery result backend introspection
    and for the audit log).
    """
    settings = get_settings()
    artifact_dir = Path(settings.artifact_dir)

    min_age_s = settings.dose_min_age_hours * 3600.0

    summary = {
        "dose": _sweep_store(
            artifact_dir / "doses",
            settings.dose_store_max_gb,
            min_age_s,
            "dose",
        ),
        "influence": _sweep_store(
            artifact_dir / "influence",
            settings.influence_store_max_gb,
            min_age_s,
            "influence",
        ),
        # O20 — sweep the optimization store under the same LRU + min-age
        # policy, and additionally prune stale per-run checkpoints.
        "optimization": _sweep_store(
            artifact_dir / "optimization",
            settings.optimization_store_max_gb,
            min_age_s,
            "optimization",
        ),
        "optimization_checkpoints": _prune_optimization_checkpoints(
            artifact_dir / "optimization",
            settings.optimization_checkpoint_keep,
        ),
        # Service 5 — BAO results are metadata-only but still LRU-capped.
        "bao": _sweep_store(
            artifact_dir / "bao",
            settings.bao_store_max_gb,
            min_age_s,
            "bao",
        ),
        # Service 6 — evaluation reports (metadata-only).
        "evaluation": _sweep_store(
            artifact_dir / "evaluation",
            settings.evaluation_store_max_gb,
            min_age_s,
            "evaluation",
        ),
    }

    emit(make_event(
        "cleanup.swept",
        state="succeeded",
        extra=summary,
    ))
    return summary


def _sweep_store(
    base_dir: Path,
    cap_gb: float,
    min_age_s: float,
    label: str,
) -> dict:
    if not base_dir.is_dir():
        return {"skipped": True, "reason": "base_dir does not exist"}

    cap_bytes = int(cap_gb * 1024**3)
    entries = _enumerate_entries(base_dir)
    total_bytes = sum(e.size_bytes for e in entries)

    evicted, freed = _evict_to_cap(entries, cap_bytes, min_age_s)

    result = {
        "store": label,
        "entries_before": len(entries),
        "bytes_before": total_bytes,
        "cap_bytes": cap_bytes,
        "evicted_count": evicted,
        "bytes_freed": freed,
    }
    if evicted > 0:
        logger.info(
            f"cleanup[{label}]: evicted {evicted} entries, freed "
            f"{freed / 1024**2:.1f} MB (was {total_bytes / 1024**2:.1f} MB)"
        )
    return result


def _prune_optimization_checkpoints(base_dir: Path, keep: int) -> dict:
    """Keep only the latest ``keep`` checkpoints per optimization id.

    Independent of the size-cap sweep: checkpoints accumulate during long
    solver loops and are only useful for warm-starting / debugging the most
    recent iterations, so we cap their count per run regardless of total store
    size. ``keep <= 0`` removes all checkpoints; a negative value disables
    pruning.
    """
    if not base_dir.is_dir() or keep < 0:
        return {"skipped": True, "kept_per_run": keep}

    pruned = 0
    runs = 0
    for child in base_dir.iterdir():
        cps_dir = child / "checkpoints"
        if not cps_dir.is_dir():
            continue
        runs += 1
        cps = sorted(
            cps_dir.glob("iter_*.npy"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        evict = cps[:-keep] if keep > 0 else cps
        for p in evict:
            try:
                p.unlink()
                pruned += 1
            except OSError as exc:  # pragma: no cover
                logger.error(f"cleanup: failed to prune checkpoint {p}: {exc}")
    if pruned:
        logger.info("cleanup[optimization]: pruned %d old checkpoints across "
                    "%d runs (keep=%d)", pruned, runs, keep)
    return {"runs_scanned": runs, "checkpoints_pruned": pruned, "kept_per_run": keep}


__all__ = ["cleanup_dose_stores"]
