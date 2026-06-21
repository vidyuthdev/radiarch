from functools import lru_cache
from typing import List, Optional, Any

from pydantic import AnyUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Radiarch global configuration."""

    model_config = SettingsConfigDict(env_prefix="RADIARCH_", env_file=".env", env_file_encoding="utf-8")

    project_name: str = Field(default="Radiarch TPS Service")
    api_prefix: str = Field(default="/api/v1")
    environment: str = Field(default="dev")

    # Force synthetic planner (skips OpenTPS/MCsquare)
    force_synthetic: bool = Field(default=False, description="Use synthetic planner instead of OpenTPS")

    # Orthanc / PACS configuration
    orthanc_base_url: AnyUrl | str = Field(default="http://localhost:8042")
    orthanc_username: Optional[str] = Field(default=None)
    orthanc_password: Optional[str] = Field(default=None)
    orthanc_use_mock: bool = Field(default=True, description="Use fake Orthanc adapter for development")

    # Database and artifact storage
    database_url: str = Field(default="", description="Leave empty to use InMemoryStore; set to sqlite:///./radiarch.db or postgresql+psycopg://... for persistence")
    artifact_dir: str = Field(default="./data/artifacts")
    upload_dir: str = Field(
        default="",
        description=(
            "Where uploaded DICOM ZIPs get extracted to. "
            "Empty (the default) resolves to {artifact_dir}/uploads at runtime."
        ),
    )

    # Session TTL
    session_ttl: int = Field(default=3600, description="Session expiration in seconds")

    # DICOMweb STOW-RS notification (vendor-neutral; empty = disabled)
    dicomweb_url: str = Field(default="", description="DICOMweb STOW-RS URL for artifact push; empty = disabled")
    dicomweb_username: Optional[str] = Field(default=None)
    dicomweb_password: Optional[str] = Field(default=None)

    # Job queue
    broker_url: str = Field(default="redis://localhost:6379/0")
    result_backend: str = Field(default="redis://localhost:6379/1")

    # OpenTPS configuration (only needed when force_synthetic=false)
    opentps_data_root: str = Field(default="/data/opentps/testData")
    opentps_beam_library: str = Field(default="/data/opentps/beam-models")
    opentps_venv: str = Field(default="", description="Path to OpenTPS venv site-packages; empty = skip")

    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    # -------------------------------------------------------------------
    # D7.1 — Celery resource limits (per-task overrides for dose work)
    # -------------------------------------------------------------------
    celery_default_time_limit_s: int = Field(
        default=1800,
        description="Hard time limit for non-dose Celery tasks (seconds).",
    )
    celery_default_soft_time_limit_s: int = Field(
        default=1500,
        description="Soft time limit for non-dose Celery tasks (seconds).",
    )
    dose_soft_time_limit_s: int = Field(
        default=3600,
        description=(
            "Soft time limit (SoftTimeLimitExceeded) for dose compute "
            "tasks. Influence builds get 2x this. Bump for clinical-grid "
            "MCsquare runs."
        ),
    )
    dose_hard_time_limit_s: int = Field(
        default=4200,
        description=(
            "Hard SIGKILL time limit for dose compute tasks. Must be "
            "greater than dose_soft_time_limit_s by enough headroom "
            "for graceful cleanup."
        ),
    )
    dose_rate_limit: str = Field(
        default="6/m",
        description=(
            "Celery rate-limit string applied to dose tasks. Prevents "
            "a flood of MCsquare jobs from spiking CPU/IO."
        ),
    )
    celery_worker_max_memory_kb: int = Field(
        default=4_000_000,  # 4 GB
        description=(
            "Worker recycles after exceeding this RSS (KB). Protects "
            "against MCsquare/numpy memory leaks. Set to 0 to disable."
        ),
    )
    celery_worker_max_tasks: int = Field(
        default=50,
        description=(
            "Worker recycles after this many tasks. Belt-and-suspenders "
            "against slow memory growth. Set to 0 to disable."
        ),
    )

    # -------------------------------------------------------------------
    # O20 — Optimization Service (Service 4) resource limits + cleanup
    # -------------------------------------------------------------------
    optimization_soft_time_limit_s: int = Field(
        default=14400,  # 4h — optimization is the heaviest stage (many Dij·w)
        description="Soft time limit for optimization run tasks (seconds).",
    )
    optimization_hard_time_limit_s: int = Field(
        default=15000,
        description="Hard SIGKILL time limit for optimization run tasks.",
    )
    optimize_rate_limit: str = Field(
        default="2/m",
        description="Celery rate-limit string applied to optimization tasks.",
    )
    optimization_store_max_gb: float = Field(
        default=30.0,
        description="Soft cap on total optimization cache size (LRU eviction).",
    )
    optimization_checkpoint_keep: int = Field(
        default=5,
        description="Keep only the latest N checkpoints per optimization id.",
    )

    # -------------------------------------------------------------------
    # Service 5 — BAO (Beam Angle Optimization) resource limits + cleanup
    # -------------------------------------------------------------------
    bao_soft_time_limit_s: int = Field(
        default=28800,  # 8h — scores many angle sets, each a fluence opt
        description="Soft time limit for BAO run tasks (seconds).",
    )
    bao_hard_time_limit_s: int = Field(
        default=30000,
        description="Hard SIGKILL time limit for BAO run tasks.",
    )
    bao_rate_limit: str = Field(
        default="1/m",
        description="Celery rate-limit string applied to BAO tasks.",
    )
    bao_store_max_gb: float = Field(
        default=5.0,
        description="Soft cap on total BAO cache size (metadata only; small).",
    )

    # -------------------------------------------------------------------
    # Service 6 — Evaluation resource limits + cleanup
    # -------------------------------------------------------------------
    evaluation_soft_time_limit_s: int = Field(
        default=1800,
        description="Soft time limit for evaluation run tasks (gamma dominates).",
    )
    evaluation_hard_time_limit_s: int = Field(
        default=2400,
        description="Hard SIGKILL time limit for evaluation run tasks.",
    )
    evaluate_rate_limit: str = Field(
        default="30/m",
        description="Celery rate-limit string applied to evaluation tasks.",
    )
    evaluation_store_max_gb: float = Field(
        default=5.0,
        description="Soft cap on total evaluation cache size (metadata only).",
    )

    # -------------------------------------------------------------------
    # D7.2 — Disk-space + cleanup policy
    # -------------------------------------------------------------------
    cleanup_interval_s: int = Field(
        default=3600,
        description="Seconds between disk-cleanup Celery Beat ticks.",
    )
    dose_store_max_gb: float = Field(
        default=50.0,
        description=(
            "Soft cap on total dose cache size. When exceeded the "
            "cleanup task evicts least-recently-accessed entries."
        ),
    )
    influence_store_max_gb: float = Field(
        default=100.0,
        description="Same as dose_store_max_gb but for the Dij cache.",
    )
    dose_min_age_hours: int = Field(
        default=24,
        description=(
            "Don't evict dose/influence entries younger than this — "
            "protects an active optimization loop from losing its Dij."
        ),
    )

    # -------------------------------------------------------------------
    # D7.3 — Structured audit logging
    # -------------------------------------------------------------------
    audit_log_path: str = Field(
        default="",
        description=(
            "If set, append JSONL audit events here in addition to "
            "stderr. Empty disables file sink. One line per event."
        ),
    )

    # -------------------------------------------------------------------
    # D8.2 — Static API key (set empty to disable auth)
    # -------------------------------------------------------------------
    api_key: str = Field(
        default="",
        description=(
            "Static API key required in X-API-Key header for protected "
            "routes (dose + influence). Empty disables auth (dev only)."
        ),
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name carrying the API key.",
    )


@lru_cache
def get_settings() -> Settings:
    import sys
    import os
    import site

    settings = Settings()

    # Add OpenTPS venv site-packages to sys.path if configured
    venv_path = settings.opentps_venv
    if venv_path and os.path.exists(venv_path):
        site.addsitedir(venv_path)

    return settings
