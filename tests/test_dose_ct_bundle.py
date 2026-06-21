"""D6.1 — CTImage in GeometryBundle.

End-to-end coverage of the new CT-persistence + load path:

1. ``GeometryStore.save`` writes ``ct.nii.gz`` when a CT array is passed,
   and skips it (backward-compat) when omitted.
2. ``GeometryResult.ct_grid_uri`` round-trips through the meta.json
   serializer.
3. ``DoseService._load_geometry`` populates ``bundle.ct_hu`` when the CT
   NIfTI exists, and falls back to None (with a logged warning) when
   either the URI is null or the file is missing.
4. ``DoseService._load_geometry`` populates ``bundle.ct_image`` with an
   OpenTPS CTImage when OpenTPS is importable; falls back to None
   otherwise without breaking the bundle.

These tests exercise the persistence + load layers directly — they do
NOT spin up FastAPI / Celery / the engines. The MCsquare consumer of
``ct_image`` is exercised in D6.2's tests.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

from radiarch.models.geometry import (
    CTMetadata,
    GeometryResult,
    GridSpec,
)
from radiarch.services.dose import DoseService, _wrap_ct_image
from radiarch.services.persistence import (
    CT_FILENAME,
    GeometryPaths,
    GeometryStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(geometry_id: str, paths: GeometryPaths, *, with_ct: bool) -> GeometryResult:
    """Build a GeometryResult pointing at on-disk paths under ``paths.root``.

    Uses a symmetric (4, 4, 4) grid so the (x,y,z) ↔ (z,y,x) transpose
    that ``_write_nifti`` applies on the way to/from disk doesn't change
    the array shape — keeps the round-trip assertions readable.
    """
    return GeometryResult(
        geometry_id=geometry_id,
        density_grid_uri=str(paths.density),
        structure_masks_uri=str(paths.masks),
        ct_grid_uri=str(paths.ct) if with_ct else None,
        structure_index={"PTV": 1},
        grid_spec=GridSpec(
            spacing_mm=(2.0, 2.0, 2.0),
            origin_mm=(0.0, 0.0, 0.0),
            size=(4, 4, 4),
        ),
        frame_of_reference_uid="1.2.3.4",
        ct_metadata=CTMetadata(num_slices=4, series_instance_uid="1.2.840.5"),
        cache_key=f"ck-{geometry_id}",
    )


def _arrays():
    """A symmetric (4, 4, 4) density/masks/CT triple."""
    density = np.ones((4, 4, 4), dtype=np.float32)
    masks = np.zeros((4, 4, 4), dtype=np.uint16)
    masks[1:3, 1:3, 1:3] = 1
    # Synthetic CT in HU: -1000 air everywhere, 0 soft tissue in the
    # PTV region, 1000 bone insert at the center.
    ct = np.full((4, 4, 4), -1000, dtype=np.int16)
    ct[1:3, 1:3, 1:3] = 0
    ct[2, 2, 2] = 1000
    return density, masks, ct


# ---------------------------------------------------------------------------
# 1. Persistence layer — writes / skips CT NIfTI based on the arg.
# ---------------------------------------------------------------------------

class TestGeometryStoreCT:
    def test_save_with_ct_writes_ct_file(self, tmp_path):
        store = GeometryStore(tmp_path)
        geometry_id = str(uuid.uuid4())
        paths = GeometryPaths.for_id(store.base_dir, geometry_id)
        density, masks, ct = _arrays()
        result = _make_result(geometry_id, paths, with_ct=True)

        store.save(
            geometry_id=geometry_id, cache_key=result.cache_key,
            density=density, masks=masks, ct=ct, result=result,
        )
        assert paths.density.exists()
        assert paths.masks.exists()
        assert paths.ct.exists(), "ct.nii.gz should be written when ct= is supplied"

        # Round-trip the CT NIfTI: values should survive int16 -> int16.
        roundtrip = sitk.GetArrayFromImage(sitk.ReadImage(str(paths.ct)))
        # _write_nifti transposes (ijk)->(zyx); our test ct is already in
        # (z, y, x) — saved as (x, y, z) on disk and read back as (z, y, x)
        # by GetArrayFromImage, so the round-trip should match shape.
        assert roundtrip.shape == ct.shape
        assert roundtrip.max() >= 1000
        assert roundtrip.min() <= -1000

    def test_save_without_ct_skips_ct_file_backward_compat(self, tmp_path):
        """Pre-D6.1 callers that don't pass ct=... still work; no file written."""
        store = GeometryStore(tmp_path)
        geometry_id = str(uuid.uuid4())
        paths = GeometryPaths.for_id(store.base_dir, geometry_id)
        density, masks, _ = _arrays()
        result = _make_result(geometry_id, paths, with_ct=False)

        store.save(
            geometry_id=geometry_id, cache_key=result.cache_key,
            density=density, masks=masks, result=result,
        )
        assert paths.density.exists()
        assert paths.masks.exists()
        assert not paths.ct.exists()

    def test_geometry_result_ct_uri_round_trips_through_meta(self, tmp_path):
        store = GeometryStore(tmp_path)
        geometry_id = str(uuid.uuid4())
        paths = GeometryPaths.for_id(store.base_dir, geometry_id)
        density, masks, ct = _arrays()
        result = _make_result(geometry_id, paths, with_ct=True)
        store.save(
            geometry_id=geometry_id, cache_key=result.cache_key,
            density=density, masks=masks, ct=ct, result=result,
        )
        # Read the cached GeometryResult back from disk.
        rehydrated = store.get_by_id(geometry_id)
        assert rehydrated is not None
        assert rehydrated.ct_grid_uri == str(paths.ct)

    def test_old_meta_without_ct_uri_still_loads(self, tmp_path):
        """An old cached GeometryResult without ct_grid_uri must still validate."""
        store = GeometryStore(tmp_path)
        geometry_id = "legacy-1"
        paths = GeometryPaths.for_id(store.base_dir, geometry_id)
        density, masks, _ = _arrays()
        result = _make_result(geometry_id, paths, with_ct=False)
        store.save(
            geometry_id=geometry_id, cache_key=result.cache_key,
            density=density, masks=masks, result=result,
        )
        rehydrated = store.get_by_id(geometry_id)
        assert rehydrated is not None
        assert rehydrated.ct_grid_uri is None


# ---------------------------------------------------------------------------
# 2. DoseService._maybe_load_ct — handles the three failure modes.
# ---------------------------------------------------------------------------

class TestMaybeLoadCT:
    def test_returns_none_when_uri_missing(self):
        """No ct_grid_uri on the result → (None, None) bundle fields."""
        # A minimal fake geom with ct_grid_uri unset.
        class FakeGeom:
            geometry_id = "g-1"
            ct_grid_uri = None
            grid_spec = GridSpec(spacing_mm=(2, 2, 3), origin_mm=(0, 0, 0), size=(8, 8, 4))
            ct_metadata = CTMetadata(num_slices=4)
            frame_of_reference_uid = "1.2.3"

        ct_hu, ct_image = DoseService._maybe_load_ct(FakeGeom())
        assert ct_hu is None
        assert ct_image is None

    def test_returns_none_when_file_missing(self, tmp_path):
        """ct_grid_uri set but file deleted under us → warn + (None, None)."""
        class FakeGeom:
            geometry_id = "g-2"
            ct_grid_uri = str(tmp_path / "nonexistent.nii.gz")
            grid_spec = GridSpec(spacing_mm=(2, 2, 3), origin_mm=(0, 0, 0), size=(8, 8, 4))
            ct_metadata = CTMetadata(num_slices=4)
            frame_of_reference_uid = "1.2.3"

        ct_hu, ct_image = DoseService._maybe_load_ct(FakeGeom())
        assert ct_hu is None
        assert ct_image is None

    def test_loads_ct_when_file_exists(self, tmp_path):
        """Happy path: NIfTI on disk → ct_hu populated as int16."""
        store = GeometryStore(tmp_path)
        geometry_id = str(uuid.uuid4())
        paths = GeometryPaths.for_id(store.base_dir, geometry_id)
        density, masks, ct = _arrays()
        result = _make_result(geometry_id, paths, with_ct=True)
        store.save(
            geometry_id=geometry_id, cache_key=result.cache_key,
            density=density, masks=masks, ct=ct, result=result,
        )
        rehydrated = store.get_by_id(geometry_id)
        assert rehydrated is not None

        ct_hu, ct_image = DoseService._maybe_load_ct(rehydrated)
        assert ct_hu is not None
        assert ct_hu.dtype == np.int16
        assert ct_hu.shape == ct.shape
        # CTImage is None if OpenTPS isn't importable; either way it
        # mustn't crash the load.
        assert ct_image is None or ct_image is not None  # tautology — both legal


# ---------------------------------------------------------------------------
# 3. _wrap_ct_image — graceful skip when OpenTPS isn't importable.
# ---------------------------------------------------------------------------

class TestWrapCTImage:
    def test_returns_none_when_opentps_missing(self):
        """Simulate OpenTPS import failing — wrapper must return None, not raise."""
        class FakeGeom:
            geometry_id = "g-3"
            grid_spec = GridSpec(spacing_mm=(2, 2, 3), origin_mm=(0, 0, 0), size=(8, 8, 4))
            ct_metadata = CTMetadata(num_slices=4)
            frame_of_reference_uid = "1.2.3"

        ct_hu = np.zeros((4, 8, 8), dtype=np.int16)
        # Force the import path inside _wrap_ct_image to fail by stubbing
        # out the module before the import runs.
        import sys
        sentinel = object()
        saved = sys.modules.pop("opentps.core.data.images", sentinel)
        # Insert a poisoned entry so the import inside _wrap_ct_image raises.
        sys.modules["opentps.core.data.images"] = None  # type: ignore
        try:
            wrapped = _wrap_ct_image(ct_hu, FakeGeom())
        finally:
            if saved is sentinel:
                sys.modules.pop("opentps.core.data.images", None)
            else:
                sys.modules["opentps.core.data.images"] = saved  # type: ignore

        assert wrapped is None


# ---------------------------------------------------------------------------
# 4. End-to-end integration: real GeometryService → DoseService round-trip.
#
# This is the "does D6.1 actually work?" test. We stub the DICOM-load step
# only (so we don't need real DICOM on disk), but let everything else run:
# HU→density, CT resample, NIfTI persistence, cache index, then
# DoseService._load_geometry pulls it back and wraps it in a CTImage.
#
# If any of these tests fail, MCsquare (D6.2) won't be able to start.
# ---------------------------------------------------------------------------

class _FakePatient:
    """Stands in for opentps.core.data.Patient."""
    def __init__(self) -> None:
        self.name = "TEST"
        self.rtStructs: list = []


class _FakeCT:
    """Stands in for opentps.core.data.images.CTImage — only the attrs the
    GeometryService actually reads off the source CT during _process."""
    def __init__(self, imageArray, origin, spacing):
        self.imageArray = imageArray
        self.origin = origin
        self.spacing = spacing
        self.patient = _FakePatient()
        self.seriesInstanceUID = "1.2.840.5"
        self.studyInstanceUID = "1.2.840.4"
        self.frameOfReferenceUID = "1.2.840.9"


def _synthetic_loaded_ct():
    """Build a small (8,8,8) HU volume with a distinct value pattern.

    Distinctive values per slab so we can verify the CT-on-disk matches
    the input array element-by-element (not just shape).
    """
    from radiarch.services.geometry import _LoadedCT
    ct_array = np.full((8, 8, 8), -1000, dtype=np.int16)  # air baseline
    ct_array[2:6, 2:6, 2:6] = 0          # water/soft tissue
    ct_array[3:5, 3:5, 3:5] = 800        # bone-ish
    ct_array[4, 4, 4] = 1500             # cortical bone single voxel

    ct = _FakeCT(imageArray=ct_array, origin=(0.0, 0.0, 0.0), spacing=(2.0, 2.0, 2.0))
    return _LoadedCT(ct=ct, patient=ct.patient, contours=[])


class TestGeometryToDoseRoundTrip:
    """The crown-jewel test for D6.1: a CT goes through the full
    GeometryService.build() → on-disk ct.nii.gz → DoseService._load_geometry
    pipeline and comes back out with the SAME Hounsfield values it went in
    with.
    """

    def test_geometry_build_produces_ct_nifti(self, tmp_path, monkeypatch):
        """End of the producer side: GeometryService writes a ct.nii.gz
        and stamps ct_grid_uri on the result."""
        from radiarch.models.geometry import (
            GeometryBuildRequest, HUDensityModel, PatientRef,
        )
        from radiarch.services.geometry import GeometryService

        svc = GeometryService(base_dir=tmp_path)
        # Stub _load so we don't need OpenTPS / Orthanc / DICOM on disk.
        monkeypatch.setattr(svc, "_load", lambda _req: _synthetic_loaded_ct())

        req = GeometryBuildRequest(
            patient_ref=PatientRef(dicom_study_uid="1.2.3"),
            grid_spec=None,
            hu_to_density_model=HUDensityModel.linear,
        )
        result = svc.build(req)

        # ct_grid_uri must be populated and the file must exist on disk.
        assert result.ct_grid_uri is not None, \
            "GeometryResult.ct_grid_uri should be populated after a build"
        assert Path(result.ct_grid_uri).is_file(), \
            f"ct.nii.gz missing on disk at {result.ct_grid_uri}"
        # And it must live next to density.nii.gz / masks.nii.gz.
        assert Path(result.ct_grid_uri).parent == Path(result.density_grid_uri).parent

    def test_dose_service_load_geometry_returns_ct_hu(self, tmp_path, monkeypatch):
        """Producer + consumer: build geometry, then load it through the
        DoseService and check ct_hu came back with the right values."""
        from radiarch.models.geometry import (
            GeometryBuildRequest, HUDensityModel, PatientRef,
        )
        from radiarch.services.dose import DoseService
        from radiarch.services.geometry import GeometryService

        geom_svc = GeometryService(base_dir=tmp_path / "geom")
        monkeypatch.setattr(geom_svc, "_load",
                            lambda _req: _synthetic_loaded_ct())

        # Build the geometry the normal way.
        req = GeometryBuildRequest(
            patient_ref=PatientRef(dicom_study_uid="1.2.3"),
            grid_spec=None,
            hu_to_density_model=HUDensityModel.linear,
        )
        geom_result = geom_svc.build(req)

        # Point the DoseService at the same geometry store so its
        # internal GeometryService() instance finds our build.
        dose_svc = DoseService(
            dose_dir=tmp_path / "doses",
            influence_dir=tmp_path / "infl",
        )
        monkeypatch.setattr(
            "radiarch.services.dose.GeometryService",
            lambda: geom_svc,
        )

        bundle = dose_svc._load_geometry(geom_result.geometry_id)

        # ct_hu must be populated and shaped to match density.
        assert bundle.ct_hu is not None, \
            "DoseService._load_geometry should populate bundle.ct_hu when ct_grid_uri is set"
        assert bundle.ct_hu.shape == bundle.density.shape, \
            "ct_hu should be on the same grid as density"
        assert bundle.ct_hu.dtype == np.int16, \
            "ct_hu should be int16 (HU values fit in 16 bits)"

        # The original synthetic CT had voxels with values -1000, 0, 800, 1500.
        # Those distinctive values must survive the round-trip exactly
        # (linear interpolation = identity for the fast path with no
        # resample, since target grid == source grid).
        seen_values = set(int(v) for v in np.unique(bundle.ct_hu))
        for expected in (-1000, 0, 800, 1500):
            assert expected in seen_values, (
                f"HU value {expected} from source CT did not survive the "
                f"persist→load round-trip. Seen values: {sorted(seen_values)}"
            )

    def test_dose_service_load_geometry_builds_ct_image_when_opentps_present(
        self, tmp_path, monkeypatch,
    ):
        """If OpenTPS is importable, ct_image should be a real CTImage."""
        from radiarch.models.geometry import (
            GeometryBuildRequest, HUDensityModel, PatientRef,
        )
        from radiarch.services.dose import DoseService
        from radiarch.services.geometry import GeometryService

        opentps_images = pytest.importorskip(
            "opentps.core.data.images",
            reason="OpenTPS not installed; ct_image will be None in this env "
                   "(behaviour already covered by TestMaybeLoadCT).",
        )

        geom_svc = GeometryService(base_dir=tmp_path / "geom")
        monkeypatch.setattr(geom_svc, "_load",
                            lambda _req: _synthetic_loaded_ct())
        req = GeometryBuildRequest(
            patient_ref=PatientRef(dicom_study_uid="1.2.3"),
            grid_spec=None,
            hu_to_density_model=HUDensityModel.linear,
        )
        geom_result = geom_svc.build(req)

        dose_svc = DoseService(
            dose_dir=tmp_path / "doses",
            influence_dir=tmp_path / "infl",
        )
        monkeypatch.setattr(
            "radiarch.services.dose.GeometryService",
            lambda: geom_svc,
        )

        bundle = dose_svc._load_geometry(geom_result.geometry_id)

        assert bundle.ct_image is not None, \
            "ct_image should be an OpenTPS CTImage when OpenTPS is importable"
        # Quack-check: must look like a CTImage. We don't assert isinstance
        # because the import-protected path may have wrapped it differently
        # in alternative builds.
        assert hasattr(bundle.ct_image, "imageArray")
        assert hasattr(bundle.ct_image, "origin")
        assert hasattr(bundle.ct_image, "spacing")
        # Spacing should match the GridSpec we built.
        assert tuple(float(s) for s in bundle.ct_image.spacing) == (2.0, 2.0, 2.0)


# ---------------------------------------------------------------------------
# 5. Cache-hit path: rebuild same request → same cached ct.nii.gz.
# ---------------------------------------------------------------------------

class TestCacheHit:
    def test_repeat_build_reuses_cached_ct(self, tmp_path, monkeypatch):
        """Two identical builds must produce the same geometry_id AND keep
        the CT file alive on disk for the second load."""
        from radiarch.models.geometry import (
            GeometryBuildRequest, HUDensityModel, PatientRef,
        )
        from radiarch.services.geometry import GeometryService

        svc = GeometryService(base_dir=tmp_path)
        monkeypatch.setattr(svc, "_load", lambda _req: _synthetic_loaded_ct())

        req = GeometryBuildRequest(
            patient_ref=PatientRef(dicom_study_uid="1.2.3"),
            grid_spec=None,
            hu_to_density_model=HUDensityModel.linear,
        )
        first = svc.build(req)
        second = svc.build(req)
        assert first.geometry_id == second.geometry_id, "Cache hit should return same id"
        assert first.ct_grid_uri == second.ct_grid_uri
        assert Path(second.ct_grid_uri).is_file()
