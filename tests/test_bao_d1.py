"""Unit tests for the BAO Pydantic models (Service 5)."""

from __future__ import annotations

import pytest

from radiarch.models.bao import BAORunRequest, CandidateAngle
from radiarch.models.dose import EngineSpec
from radiarch.models.optimization import ObjectiveSpec


def _req(**over) -> BAORunRequest:
    kwargs = dict(
        geometry_id="g-1",
        dose_engine=EngineSpec(name="analytic"),
        objectives=[ObjectiveSpec(type="DUniform", structure_name="PTV",
                                  dose_gy=10.0, weight=1.0)],
        n_beams=2,
        angle_step_deg=90.0,
    )
    kwargs.update(over)
    return BAORunRequest(**kwargs)


def test_candidate_generation_from_step():
    cands = _req(angle_step_deg=90.0).resolve_candidates()
    assert [c.gantry_deg for c in cands] == [0.0, 90.0, 180.0, 270.0]


def test_candidate_generation_with_couch():
    cands = _req(angle_step_deg=180.0, couch_angles=[0.0, 30.0]).resolve_candidates()
    assert [(c.gantry_deg, c.couch_deg) for c in cands] == [
        (0.0, 0.0), (0.0, 30.0), (180.0, 0.0), (180.0, 30.0),
    ]


def test_explicit_candidates_passthrough():
    cands = [CandidateAngle(gantry_deg=10), CandidateAngle(gantry_deg=20)]
    assert _req(candidate_angles=cands, angle_step_deg=None).resolve_candidates() == cands


def test_requires_candidates_or_step():
    with pytest.raises(ValueError, match="candidate_angles or angle_step_deg"):
        BAORunRequest(geometry_id="g", dose_engine=EngineSpec(name="analytic"),
                      objectives=[ObjectiveSpec(type="DMin", structure_name="P",
                                                dose_gy=1, weight=1)],
                      n_beams=1)


def test_rejects_unknown_search():
    with pytest.raises(ValueError, match="search must be"):
        _req(search="bogus")


def test_rejects_unknown_scoring():
    with pytest.raises(ValueError, match="scoring must be"):
        _req(scoring="bogus")


def test_cache_key_stable_and_sensitive():
    base = _req()
    assert base.compute_cache_key() == _req().compute_cache_key()
    # plan_id is transient — must not change the key.
    assert _req(plan_id="p-1").compute_cache_key() == base.compute_cache_key()
    # Substantive fields change the key.
    assert _req(n_beams=3).compute_cache_key() != base.compute_cache_key()
    assert _req(angle_step_deg=45.0).compute_cache_key() != base.compute_cache_key()
    assert _req(search="top_k").compute_cache_key() != base.compute_cache_key()


def test_candidate_angle_bounds():
    with pytest.raises(ValueError):
        CandidateAngle(gantry_deg=360.0)  # must be < 360
    with pytest.raises(ValueError):
        CandidateAngle(gantry_deg=-1.0)
