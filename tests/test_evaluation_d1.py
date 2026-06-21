"""Unit tests for the Evaluation Pydantic models (Service 6)."""

from __future__ import annotations

import pytest

from radiarch.models.evaluation import EvaluationRequest, GammaSpec


def _req(**over) -> EvaluationRequest:
    kwargs = dict(dose_id="d-1", geometry_id="g-1", prescription_gy=60.0)
    kwargs.update(over)
    return EvaluationRequest(**kwargs)


def test_requires_a_dose_source():
    with pytest.raises(ValueError, match="dose_id or dose_ref_uri"):
        EvaluationRequest(geometry_id="g-1", prescription_gy=60.0)


def test_accepts_dose_ref_uri():
    r = EvaluationRequest(dose_ref_uri="file:///tmp/d.nii.gz",
                          geometry_id="g-1", prescription_gy=60.0)
    assert r.dose_ref_uri


def test_gamma_requires_reference():
    with pytest.raises(ValueError, match="reference_dose"):
        GammaSpec()


def test_cache_key_stable_and_sensitive():
    base = _req()
    assert base.compute_cache_key() == _req().compute_cache_key()
    # plan_id is transient.
    assert _req(plan_id="p").compute_cache_key() == base.compute_cache_key()
    assert _req(prescription_gy=70.0).compute_cache_key() != base.compute_cache_key()
    assert _req(target_structure="PTV").compute_cache_key() != base.compute_cache_key()
    # gamma changes the key.
    g = _req(gamma=GammaSpec(reference_dose_id="r-1"))
    assert g.compute_cache_key() != base.compute_cache_key()


def test_structures_order_independent_cache_key():
    a = _req(structures=["PTV", "OAR"])
    b = _req(structures=["OAR", "PTV"])
    assert a.compute_cache_key() == b.compute_cache_key()
