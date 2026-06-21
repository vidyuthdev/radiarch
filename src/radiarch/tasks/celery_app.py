import sys
import os

# Ensure /app is in sys.path so vendored packages like opentps can be imported
if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from celery import Celery

from ..config import get_settings

settings = get_settings()

celery_app = Celery(
    "radiarch",
    broker=settings.broker_url,
    backend=settings.result_backend,
    include=[
        "radiarch.tasks.plan_tasks",
        "radiarch.tasks.geometry_tasks",
        "radiarch.tasks.beam_model_tasks",
        # D7.1 fix — dose_tasks was missing from include, which meant
        # the broker never registered radiarch.dose.compute /
        # radiarch.dose.influence and async dose builds silently
        # disappeared in non-eager mode.
        "radiarch.tasks.dose_tasks",
        # Service 4 — async inverse-plan optimization.
        "radiarch.tasks.optimization_tasks",
        # Service 5 — async beam-angle optimization.
        "radiarch.tasks.bao_tasks",
        # Service 6 — async plan evaluation (DVH / indices / gamma).
        "radiarch.tasks.evaluation_tasks",
        "radiarch.tasks.cleanup_tasks",
    ],
)

# Per-task overrides — dose Monte-Carlo runs need longer time + memory
# limits than fast geometry/beam-model builds. Configured here rather
# than via decorator so they're discoverable in one place and tunable
# from settings.
celery_app.conf.task_routes = {
    "radiarch.dose.*": {"queue": "dose"},
    "radiarch.optimize.*": {"queue": "optimize"},
    "radiarch.bao.*": {"queue": "optimize"},
    "radiarch.evaluate.*": {"queue": "evaluate"},
    "radiarch.cleanup.*": {"queue": "maintenance"},
}

celery_app.conf.task_annotations = {
    "radiarch.dose.compute": {
        "soft_time_limit": settings.dose_soft_time_limit_s,
        "time_limit": settings.dose_hard_time_limit_s,
        "rate_limit": settings.dose_rate_limit,
    },
    "radiarch.dose.influence": {
        # Beamlet-mode builds are heavier — 2x the time budget.
        "soft_time_limit": settings.dose_soft_time_limit_s * 2,
        "time_limit": settings.dose_hard_time_limit_s * 2,
        "rate_limit": settings.dose_rate_limit,
    },
    "radiarch.optimize.run": {
        # Optimization is the heaviest stage: a full solver loop of Dij·w
        # matvecs, optionally over many robust scenarios.
        "soft_time_limit": settings.optimization_soft_time_limit_s,
        "time_limit": settings.optimization_hard_time_limit_s,
        "rate_limit": settings.optimize_rate_limit,
    },
    "radiarch.bao.run": {
        # BAO scores many angle sets, each a (short) fluence optimization —
        # the heaviest task overall. Give it the largest budget.
        "soft_time_limit": settings.bao_soft_time_limit_s,
        "time_limit": settings.bao_hard_time_limit_s,
        "rate_limit": settings.bao_rate_limit,
    },
    "radiarch.evaluate.run": {
        # Evaluation is read-only array analysis — light, except gamma on a
        # full grid. Modest budget; gamma dominates when enabled.
        "soft_time_limit": settings.evaluation_soft_time_limit_s,
        "time_limit": settings.evaluation_hard_time_limit_s,
        "rate_limit": settings.evaluate_rate_limit,
    },
}

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    task_track_started=True,
    # Production resilience (defaults — overridden per-task above)
    task_time_limit=settings.celery_default_time_limit_s,
    task_soft_time_limit=settings.celery_default_soft_time_limit_s,
    task_acks_late=True,           # ACK after completion, not before
    worker_prefetch_multiplier=1,  # One task at a time per worker
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    # D7.1 — per-worker memory cap. Worker recycles after exceeding
    # this many KB of RSS, which protects against MCsquare leaks
    # without killing in-flight tasks (Celery waits for current task
    # to finish, then exits cleanly).
    worker_max_memory_per_child=settings.celery_worker_max_memory_kb,
    # Recycle every N tasks too, belt-and-suspenders against leaks
    # in long-running workers.
    worker_max_tasks_per_child=settings.celery_worker_max_tasks,
)

# D7.2 — scheduled disk-space sweep. Runs every N minutes via Celery
# Beat (deploy with `celery -A radiarch.tasks.celery_app beat`).
celery_app.conf.beat_schedule = {
    "dose-store-cleanup": {
        "task": "radiarch.cleanup.dose_stores",
        "schedule": float(settings.cleanup_interval_s),
        "options": {"queue": "maintenance"},
    },
}

if settings.environment == "dev":
    # Run tasks synchronously during development to avoid separate worker requirement
    celery_app.conf.task_always_eager = True

