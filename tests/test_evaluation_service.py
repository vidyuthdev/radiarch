"""End-to-end tests for :class:`EvaluationService` (Service 6).

The geometry loader is stubbed (no SimpleITK geometry store), and the dose is
supplied as a ``file://`` NIfTI written to a temp dir, so the service runs fully
without the dose cache or a real geometry.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from radiarch.models.beam_model import (
    BeamModelResult, FluenceElementSet, Modality, PerBeamElements,
)
from radiarch.models.evaluation import EvaluationRequest, GammaSpec
from radiarch.models.geometry import CTMetadata, GeometryResult, GridSpec
from radiarch.services.dose import DoseService
from radiarch.services.dose_engines.protocol import GeometryBundle
from radiarch.services.evaluation import EvaluationService


def _geometry_bundle() -> GeometryBundle:
    nz, ny, nx = 6, 6, 6
    masks = np.zeros((nz, ny, nx), dtype=np.uint16)
    masks[2:4, 2:4, 2:4] = 1  # PTV
    masks[2:4, 4:6, 2:4] = 2  # OAR
    return GeometryBundle(
        result=GeometryResult(
            geometry_id="g-1", density_grid_uri="/tmp/d.nii.gz",
            structure_masks_uri="/tmp/m.nii.gz",
            structure_index={"PTV": 1, "OAR": 2},
            grid_spec=GridSpec(spacing_mm=(2, 2, 2), origin_mm=(0, 0, 0),
                               size=(nx, ny, nz)),
            frame_of_reference_uid="1.2.3",
            ct_metadata=CTMetadata(num_slices=nz), cache_key="g",
        ),
        density=np.ones((nz, ny, nx), dtype=np.float32),
        masks=masks, spacing_mm=(2.0, 2.0, 2.0),
    )


def _write_dose(path: Path, arr: np.ndarray) -> None:
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing([2.0, 2.0, 2.0])
    sitk.WriteImage(img, str(path), useCompression=True)


@pytest.fixture
def ctx(monkeypatch):
    tmp = tempfile.TemporaryDirectory()
    store_tmp = tempfile.TemporaryDirectory()
    dose_service = DoseService(dose_dir=store_tmp.name,
                              influence_dir=store_tmp.name)
    monkeypatch.setattr(dose_service, "_load_geometry",
                        lambda gid: _geometry_bundle())
    svc = EvaluationService(base_dir=tmp.name, dose_service=dose_service)

    # Dose: 60 Gy in the PTV, 10 Gy in the OAR, 0 elsewhere.
    dose = np.zeros((6, 6, 6), dtype=np.float32)
    dose[2:4, 2:4, 2:4] = 60.0
    dose[2:4, 4:6, 2:4] = 10.0
    dose_path = Path(tmp.name) / "dose.nii.gz"
    _write_dose(dose_path, dose)

    yield svc, f"file://{dose_path}", dose_path
    tmp.cleanup()
    store_tmp.cleanup()


def _req(dose_uri, **over) -> EvaluationRequest:
    kwargs = dict(dose_ref_uri=dose_uri, geometry_id="g-1",
                  prescription_gy=60.0, target_structure="PTV")
    kwargs.update(over)
    return EvaluationRequest(**kwargs)


def test_dvh_curves_for_all_structures(ctx):
    svc, dose_uri, _ = ctx
    result = svc.run(_req(dose_uri))
    names = {c.structure_name for c in result.dvh_curves}
    assert names == {"PTV", "OAR"}
    ptv = next(c for c in result.dvh_curves if c.structure_name == "PTV")
    assert ptv.metrics.mean_gy == pytest.approx(60.0)
    assert ptv.metrics.v_prescription_pct == pytest.approx(100.0)


def test_indices_for_target(ctx):
    svc, dose_uri, _ = ctx
    result = svc.run(_req(dose_uri))
    assert result.indices is not None
    assert result.indices.target_structure == "PTV"
    assert result.indices.coverage_pct == pytest.approx(100.0)
    assert result.indices.conformity_index == pytest.approx(1.0)  # PTV exactly == isodose
    assert result.indices.hotspot_gy == pytest.approx(60.0)


def test_structures_subset(ctx):
    svc, dose_uri, _ = ctx
    result = svc.run(_req(dose_uri, structures=["PTV"]))
    assert {c.structure_name for c in result.dvh_curves} == {"PTV"}


def test_gamma_against_self(ctx):
    svc, dose_uri, dose_path = ctx
    result = svc.run(_req(dose_uri,
                          gamma=GammaSpec(reference_dose_uri=f"file://{dose_path}")))
    assert result.gamma is not None
    assert result.gamma.pass_rate_pct == pytest.approx(100.0)


def test_cache_hit_same_id(ctx):
    svc, dose_uri, _ = ctx
    a = svc.run(_req(dose_uri))
    b = svc.run(_req(dose_uri))
    assert a.evaluation_id == b.evaluation_id


def test_unknown_structure_rejected(ctx):
    svc, dose_uri, _ = ctx
    with pytest.raises(ValueError, match="not in geometry"):
        svc.run(_req(dose_uri, structures=["NOPE"]))


def test_persisted_and_retrievable(ctx):
    svc, dose_uri, _ = ctx
    result = svc.run(_req(dose_uri))
    again = svc.store.get_by_id(result.evaluation_id)
    assert again is not None
    assert again.evaluation_id == result.evaluation_id
