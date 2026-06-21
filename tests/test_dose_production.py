"""Production-hardening tests for Service 3 (D6.7 + D7.x + D8.2).

Covers the new surface added by the D6.7-D8.3 batch:

* ``/dose/engines`` and ``/dose/engines/{name}`` health endpoints
* API-key auth dependency (allow / deny / disabled-when-empty)
* Audit-log emission (file sink + JSONL format)
* Disk cleanup task — LRU eviction respecting min-age protection

These tests run without OpenTPS / MCsquare — they exercise the
plumbing, not the physics. Real MCsquare integration is validated
by ``demo/compare_engines.py`` (D6.5).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """FastAPI test client with API key + audit log enabled."""
    monkeypatch.setenv("RADIARCH_API_KEY", "test-key-abc123")
    monkeypatch.setenv("RADIARCH_AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("RADIARCH_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RADIARCH_DATABASE_URL", "")
    monkeypatch.setenv("RADIARCH_BROKER_URL", "memory://")
    monkeypatch.setenv("RADIARCH_RESULT_BACKEND", "cache+memory://")
    # Force settings reload (lru_cache).
    from radiarch.config import get_settings
    get_settings.cache_clear()

    from radiarch.app import create_app
    return TestClient(create_app())


@pytest.fixture
def app_client_no_auth(tmp_path, monkeypatch):
    """FastAPI test client with API key DISABLED (empty)."""
    monkeypatch.setenv("RADIARCH_API_KEY", "")
    monkeypatch.setenv("RADIARCH_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RADIARCH_DATABASE_URL", "")
    monkeypatch.setenv("RADIARCH_BROKER_URL", "memory://")
    monkeypatch.setenv("RADIARCH_RESULT_BACKEND", "cache+memory://")
    from radiarch.config import get_settings
    get_settings.cache_clear()
    from radiarch.app import create_app
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# D6.7 — Engine health endpoints
# ---------------------------------------------------------------------------

class TestEngineHealthEndpoints:
    def test_list_engines_returns_known_engines(self, app_client):
        r = app_client.get(
            "/api/v1/dose/engines",
            headers={"X-API-Key": "test-key-abc123"},
        )
        assert r.status_code == 200
        names = {e["name"] for e in r.json()}
        # analytic + mcsquare + ccc all register at import time
        assert {"analytic", "mcsquare", "ccc"} <= names

    def test_engine_health_analytic_is_available(self, app_client):
        r = app_client.get(
            "/api/v1/dose/engines/analytic",
            headers={"X-API-Key": "test-key-abc123"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "analytic"
        assert body["registered"] is True
        # analytic has no opentps dependency → always available
        assert body["available"] is True

    def test_engine_health_unknown_returns_404(self, app_client):
        r = app_client.get(
            "/api/v1/dose/engines/does-not-exist",
            headers={"X-API-Key": "test-key-abc123"},
        )
        assert r.status_code == 404

    def test_mcsquare_health_includes_diagnostics(self, app_client):
        """Even when mcsquare can't run, the diagnostics block must be present."""
        r = app_client.get(
            "/api/v1/dose/engines/mcsquare",
            headers={"X-API-Key": "test-key-abc123"},
        )
        assert r.status_code == 200
        body = r.json()
        assert "diagnostics" in body
        # The diagnostics dict always has opentps_importable
        assert "opentps_importable" in body["diagnostics"]


# ---------------------------------------------------------------------------
# D8.2 — Static API key auth
# ---------------------------------------------------------------------------

class TestApiKeyAuth:
    def test_engines_endpoint_requires_key(self, app_client):
        r = app_client.get("/api/v1/dose/engines")  # no header
        assert r.status_code == 401
        assert "API key" in r.json()["detail"]

    def test_engines_endpoint_accepts_correct_key(self, app_client):
        r = app_client.get(
            "/api/v1/dose/engines",
            headers={"X-API-Key": "test-key-abc123"},
        )
        assert r.status_code == 200

    def test_wrong_key_returns_401(self, app_client):
        r = app_client.get(
            "/api/v1/dose/engines",
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 401

    def test_auth_disabled_when_key_empty(self, app_client_no_auth):
        """Empty RADIARCH_API_KEY → no auth check (dev convenience)."""
        r = app_client_no_auth.get("/api/v1/dose/engines")
        assert r.status_code == 200

    def test_401_response_includes_www_authenticate(self, app_client):
        r = app_client.get("/api/v1/dose/engines")
        assert r.status_code == 401
        assert "WWW-Authenticate" in r.headers
        assert "ApiKey" in r.headers["WWW-Authenticate"]

    def test_constant_time_comparison_used(self, app_client):
        """Both 'wrong key' and 'missing key' return the same 401 detail.

        Distinguishing them would leak whether the header was present.
        """
        r1 = app_client.get("/api/v1/dose/engines")
        r2 = app_client.get(
            "/api/v1/dose/engines",
            headers={"X-API-Key": "definitely-wrong"},
        )
        assert r1.status_code == r2.status_code == 401
        assert r1.json()["detail"] == r2.json()["detail"]


# ---------------------------------------------------------------------------
# D7.3 — Audit log
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_audit_emits_jsonl_to_file_sink(self, tmp_path, monkeypatch):
        from radiarch.services.audit import emit, make_event
        log_path = tmp_path / "audit.jsonl"
        monkeypatch.setenv("RADIARCH_AUDIT_LOG_PATH", str(log_path))
        from radiarch.config import get_settings
        get_settings.cache_clear()

        evt = make_event(
            "test.event",
            state="succeeded",
            geometry_id="g-1",
            engine_name="analytic",
        )
        emit(evt)

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["event_type"] == "test.event"
        assert parsed["state"] == "succeeded"
        assert parsed["geometry_id"] == "g-1"
        assert parsed["engine_name"] == "analytic"
        # Required bookkeeping fields
        assert "event_id" in parsed
        assert "timestamp" in parsed
        # None values are stripped
        assert "dose_id" not in parsed

    def test_audit_span_emits_started_and_succeeded(self, tmp_path, monkeypatch):
        from radiarch.services.audit import audit_span
        log_path = tmp_path / "audit.jsonl"
        monkeypatch.setenv("RADIARCH_AUDIT_LOG_PATH", str(log_path))
        from radiarch.config import get_settings
        get_settings.cache_clear()

        with audit_span("test.span", geometry_id="g-1") as ctx:
            ctx["dose_id"] = "d-99"

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2
        started = json.loads(lines[0])
        finished = json.loads(lines[1])
        assert started["state"] == "started"
        assert finished["state"] == "succeeded"
        assert finished["dose_id"] == "d-99"
        assert "duration_s" in finished

    def test_audit_span_captures_exception(self, tmp_path, monkeypatch):
        from radiarch.services.audit import audit_span
        log_path = tmp_path / "audit.jsonl"
        monkeypatch.setenv("RADIARCH_AUDIT_LOG_PATH", str(log_path))
        from radiarch.config import get_settings
        get_settings.cache_clear()

        with pytest.raises(RuntimeError):
            with audit_span("test.span") as ctx:
                raise RuntimeError("boom")

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2
        finished = json.loads(lines[1])
        assert finished["state"] == "failed"
        assert finished["error_type"] == "RuntimeError"
        assert "boom" in finished["error_message"]

    def test_audit_never_raises_when_file_unwritable(self, tmp_path, monkeypatch):
        """Audit must not break the app even if its sink is broken."""
        from radiarch.services.audit import emit, make_event
        # Point at a path that can't exist
        monkeypatch.setenv("RADIARCH_AUDIT_LOG_PATH", "/dev/null/no/such/path.jsonl")
        from radiarch.config import get_settings
        get_settings.cache_clear()

        # Should NOT raise
        emit(make_event("test.event", state="started"))


# ---------------------------------------------------------------------------
# D7.2 — Disk cleanup task
# ---------------------------------------------------------------------------

class TestCleanupTask:
    def _make_entry(self, base: Path, name: str, size_kb: int = 16, age_seconds: float = 0):
        """Create a dose-store-shaped directory of approximate size."""
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        # meta.json drives mtime → atime; create both
        meta = d / "meta.json"
        meta.write_text('{"dose_id": "' + name + '"}')
        # The actual cache blob
        blob = d / "dose.nii.gz"
        blob.write_bytes(b"\0" * (size_kb * 1024))
        # Backdate access time
        if age_seconds:
            mtime = time.time() - age_seconds
            os.utime(meta, (mtime, mtime))
            os.utime(blob, (mtime, mtime))
        return d

    def test_enumerate_skips_dirs_without_meta(self, tmp_path):
        from radiarch.tasks.cleanup_tasks import _enumerate_entries
        good = self._make_entry(tmp_path, "good-1")
        # half-written entry — no meta.json
        (tmp_path / "tempdir-1").mkdir()
        (tmp_path / "tempdir-1" / "dose.nii.gz").write_bytes(b"\0" * 1024)

        entries = _enumerate_entries(tmp_path)
        ids = {e.id for e in entries}
        assert ids == {"good-1"}

    def test_evict_to_cap_respects_min_age(self, tmp_path):
        from radiarch.tasks.cleanup_tasks import _enumerate_entries, _evict_to_cap

        # 3 entries × 100 KB each = 300 KB on disk. Cap to 150 KB.
        # But all 3 are "young" (age=0) — must not be evicted.
        for name in ("a", "b", "c"):
            self._make_entry(tmp_path, name, size_kb=100, age_seconds=0)

        entries = _enumerate_entries(tmp_path)
        evicted, freed = _evict_to_cap(
            entries,
            cap_bytes=150 * 1024,
            min_age_seconds=24 * 3600,
        )
        assert evicted == 0, "young entries must be protected"
        assert freed == 0
        # Files still present
        assert (tmp_path / "a" / "meta.json").is_file()

    def test_evict_to_cap_evicts_oldest_first(self, tmp_path):
        from radiarch.tasks.cleanup_tasks import _enumerate_entries, _evict_to_cap

        # All entries older than min-age (1 hour). Cap at 150 KB.
        # Should evict the *two* oldest to get under cap.
        self._make_entry(tmp_path, "old-1", size_kb=100, age_seconds=10000)
        self._make_entry(tmp_path, "mid-1", size_kb=100, age_seconds=5000)
        self._make_entry(tmp_path, "new-1", size_kb=100, age_seconds=3700)

        entries = _enumerate_entries(tmp_path)
        # Sum bytes that each entry actually has on disk (includes the
        # meta.json + any other files), so we don't hard-code 100 KB
        # and miss the meta sidecar.
        bytes_per_entry = {e.id: e.size_bytes for e in entries}

        evicted, freed = _evict_to_cap(
            entries,
            cap_bytes=150 * 1024,
            min_age_seconds=3600,
        )
        assert evicted == 2
        # Freed bytes = sum of the two oldest entries' on-disk sizes.
        # We can't predict it exactly (FS block size, meta.json size,
        # etc.) so assert on the sum of the two oldest as captured.
        expected_freed = bytes_per_entry["old-1"] + bytes_per_entry["mid-1"]
        assert freed == expected_freed
        # Newest survives
        assert (tmp_path / "new-1" / "meta.json").is_file()
        # Oldest is gone
        assert not (tmp_path / "old-1").exists()

    def test_evict_to_cap_is_noop_when_under_cap(self, tmp_path):
        from radiarch.tasks.cleanup_tasks import _enumerate_entries, _evict_to_cap

        self._make_entry(tmp_path, "small-1", size_kb=10, age_seconds=10000)
        entries = _enumerate_entries(tmp_path)
        evicted, freed = _evict_to_cap(
            entries,
            cap_bytes=1024 * 1024,  # 1 MB, way over
            min_age_seconds=3600,
        )
        assert evicted == 0
        assert freed == 0


# ---------------------------------------------------------------------------
# D6.2-D6.4 — MCsquare engine internals (unit-level, no OpenTPS required)
# ---------------------------------------------------------------------------

class TestMCsquareEngineUnit:
    def test_health_payload_shape(self):
        from radiarch.services.dose_engines.mcsquare import MCsquareEngine
        engine = MCsquareEngine()
        h = engine.health()
        assert h["name"] == "mcsquare"
        assert h["modalities"] == ["PROTON_PBS"]
        assert "available" in h
        assert "supports" in h
        assert h["supports"]["compute_dose"] is True
        assert h["supports"]["build_influence"] is True
        assert "diagnostics" in h

    def test_validate_rejects_wrong_modality(self):
        from radiarch.services.dose_engines.mcsquare import MCsquareEngine
        from types import SimpleNamespace
        engine = MCsquareEngine()
        # Build minimal stubs that satisfy the validate() attribute paths
        geometry = SimpleNamespace(density=__import__("numpy").zeros((4, 4, 4)), ct_image=object())
        modality = SimpleNamespace(value="PHOTON_IMRT")
        fluence = SimpleNamespace(total_count=10)
        beam_model_result = SimpleNamespace(modality=modality, fluence_elements=fluence)
        beam_model = SimpleNamespace(result=beam_model_result)
        issues = engine.validate(geometry, beam_model, {})
        assert any("PROTON_PBS" in issue for issue in issues)

    def test_validate_rejects_missing_ct_image(self):
        from radiarch.services.dose_engines.mcsquare import MCsquareEngine
        from types import SimpleNamespace
        engine = MCsquareEngine()
        geometry = SimpleNamespace(density=__import__("numpy").zeros((4, 4, 4)), ct_image=None)
        modality = SimpleNamespace(value="PROTON_PBS")
        fluence = SimpleNamespace(total_count=10)
        beam_model_result = SimpleNamespace(modality=modality, fluence_elements=fluence)
        beam_model = SimpleNamespace(result=beam_model_result)
        issues = engine.validate(geometry, beam_model, {})
        assert any("CTImage" in issue for issue in issues)

    def test_apply_weights_uses_flat_path_when_available(self):
        from radiarch.services.dose_engines.mcsquare import MCsquareEngine
        import numpy as np
        from types import SimpleNamespace

        # Plan exposes .spotMUs of the right size → flat path
        weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        plan = SimpleNamespace(spotMUs=np.zeros(4, dtype=np.float32))
        fluence = SimpleNamespace(total_count=4, per_beam=None)

        MCsquareEngine._apply_weights_to_plan(plan, fluence, weights)
        np.testing.assert_array_equal(plan.spotMUs, weights)


# ---------------------------------------------------------------------------
# Smoke: Celery app config exposes new task names + Beat schedule
# ---------------------------------------------------------------------------

class TestCeleryAppConfig:
    def test_dose_tasks_included(self):
        from radiarch.tasks.celery_app import celery_app
        # The bug we fixed — dose_tasks was missing from include
        assert "radiarch.tasks.dose_tasks" in celery_app.conf.include
        assert "radiarch.tasks.cleanup_tasks" in celery_app.conf.include

    def test_per_task_annotations_exist_for_dose(self):
        from radiarch.tasks.celery_app import celery_app
        annotations = celery_app.conf.task_annotations
        assert "radiarch.dose.compute" in annotations
        assert "radiarch.dose.influence" in annotations
        assert annotations["radiarch.dose.compute"]["soft_time_limit"] > 0

    def test_beat_schedule_includes_cleanup(self):
        from radiarch.tasks.celery_app import celery_app
        schedule = celery_app.conf.beat_schedule
        assert "dose-store-cleanup" in schedule
        entry = schedule["dose-store-cleanup"]
        assert entry["task"] == "radiarch.cleanup.dose_stores"
