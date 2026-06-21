"""Pydantic I/O models for the Geometry Service.

Service 1 converts raw clinical DICOM (CT + RTSTRUCT) into a
computation-ready voxel model consumable by downstream dose / optimization
services. This module defines the public request / response schemas.

See ``docs/tps_services_implementation_plan.md`` (Service 1) for the full
specification. Highlights:

  - ``GeometryBuildRequest`` — inputs: patient_ref, target GridSpec,
    HU→density model choice, optional structure-name aliasing.
  - ``GeometryResult`` — outputs: URIs to the density grid and multi-label
    mask volume, a structure_index name→label map, the realized GridSpec
    with its 4×4 affine, and a content-addressable ``cache_key``.

The cache_key is deliberately stable across runs so identical requests can
short-circuit to a cached geometry without rebuilding.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from .job import JobState


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HUDensityModel(str, Enum):
    """Selectable HU → mass density conversion models.

    SCHNEIDER       — Piecewise-linear Schneider-2000 calibration.
    STOICHIOMETRIC  — Vendored MCsquare CT calibration (mass density from
                      tissue-composition stoichiometry). Most accurate for
                      proton dose calculation.
    LINEAR          — Simple ρ = max(0, 1 + HU/1000). Fast, for tests and
                      synthetic mode.
    """

    schneider = "SCHNEIDER"
    stoichiometric = "STOICHIOMETRIC"
    linear = "LINEAR"


# ---------------------------------------------------------------------------
# GridSpec
# ---------------------------------------------------------------------------

class GridSpec(BaseModel):
    """Axis-aligned voxel grid in patient coordinates (LPS).

    ``spacing_mm`` is always required. Leaving ``origin_mm`` or ``size``
    null in an input *request* means "inherit from the source CT"; the
    service will fill them in before returning a GridSpec in a
    ``GeometryResult``.

    Axes correspond to the stored array layout (i, j, k). The ``affine``
    is derived from spacing + origin on demand — no rotational component
    is supported in v1 (rows/cols strictly aligned to patient axes).
    """

    spacing_mm: Tuple[float, float, float] = Field(
        ..., description="Voxel spacing in mm along (i, j, k)."
    )
    origin_mm: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="Origin (mm) of voxel (0,0,0) in patient LPS coordinates.",
    )
    size: Optional[Tuple[int, int, int]] = Field(
        default=None, description="Number of voxels along (i, j, k)."
    )

    # Populated on output only — derived from spacing + origin.
    affine: Optional[List[List[float]]] = Field(
        default=None,
        description="4×4 voxel-index → patient-LPS affine. Derived; set on output.",
    )

    @field_validator("spacing_mm")
    @classmethod
    def _spacing_positive(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if any(s <= 0 for s in v):
            raise ValueError(f"spacing_mm must be strictly positive, got {v}")
        return v

    @field_validator("size")
    @classmethod
    def _size_positive(cls, v: Optional[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
        if v is not None and any(s <= 0 for s in v):
            raise ValueError(f"size entries must be positive, got {v}")
        return v

    # ---- Helpers (not part of the public schema) --------------------------

    def compute_affine(self) -> List[List[float]]:
        """Build a 4×4 voxel-index → patient-LPS affine from spacing + origin.

        Requires both ``origin_mm`` and ``spacing_mm`` to be populated.
        """
        if self.origin_mm is None:
            raise ValueError("compute_affine requires origin_mm to be set")
        sx, sy, sz = self.spacing_mm
        ox, oy, oz = self.origin_mm
        return [
            [sx, 0.0, 0.0, ox],
            [0.0, sy, 0.0, oy],
            [0.0, 0.0, sz, oz],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def to_numpy_affine(self) -> np.ndarray:
        return np.asarray(self.compute_affine(), dtype=np.float64)

    def is_fully_specified(self) -> bool:
        return self.origin_mm is not None and self.size is not None


# ---------------------------------------------------------------------------
# Patient reference
# ---------------------------------------------------------------------------

class PatientRef(BaseModel):
    """Points the Geometry Service at a specific CT + RTSTRUCT pair.

    Two mutually-compatible ways to identify the source data:

    * ``dicom_study_uid`` (+ optional series UIDs) — pulls from a PACS
      via the configured Orthanc/DICOMweb adapter, or from the
      ``opentps_data_root`` fallback in dev mode.
    * ``upload_id`` — points at a previously-uploaded DICOM bundle
      sitting under ``{settings.upload_dir}/{upload_id}/``. Takes
      precedence when both are present.

    At least one of the two MUST be provided.
    """

    dicom_study_uid: Optional[str] = Field(
        default=None,
        description="DICOM Study Instance UID. Required unless upload_id is set.",
    )
    ct_series_uid: Optional[str] = Field(
        default=None,
        description="CT Series Instance UID. Null = auto-detect the primary CT.",
    )
    rtstruct_uid: Optional[str] = Field(
        default=None,
        description="RTSTRUCT Series Instance UID. Null = auto-detect.",
    )
    upload_id: Optional[str] = Field(
        default=None,
        description=(
            "Upload id returned by POST /uploads/dicom. When set, the "
            "geometry build reads CT + RTSTRUCT from the extracted upload "
            "directory instead of going to PACS."
        ),
    )

    @model_validator(mode="after")
    def _require_one_source(self) -> "PatientRef":
        if not self.dicom_study_uid and not self.upload_id:
            raise ValueError(
                "PatientRef requires either dicom_study_uid or upload_id."
            )
        return self


# ---------------------------------------------------------------------------
# Request / response
# ---------------------------------------------------------------------------

class GeometryBuildRequest(BaseModel):
    """Input payload for POST /api/v1/geometry/build."""

    patient_ref: PatientRef
    grid_spec: Optional[GridSpec] = Field(
        default=None,
        description="Target grid. Null = match the source CT grid exactly (fast path).",
    )
    hu_to_density_model: HUDensityModel = Field(
        default=HUDensityModel.stoichiometric,
        description="Which HU→density conversion to use.",
    )
    structure_name_map: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description=(
            "Canonical-name → list-of-aliases mapping. Case-insensitive matching. "
            'e.g. {"PTV": ["PTV_60", "PTV60"], "SpinalCord": ["Cord"]}'
        ),
    )
    data_root_override: Optional[str] = Field(
        default=None,
        description="Override RADIARCH_OPENTPS_DATA_ROOT for this build (dev/testing only).",
    )

    # ---- Cache key -------------------------------------------------------

    def compute_cache_key(self) -> str:
        """Deterministic sha256 over the inputs that affect the output.

        Excludes ``data_root_override`` (a developer convenience — the
        actual CT/RTSTRUCT UIDs fully determine the geometry content).
        """
        payload = {
            "study": self.patient_ref.dicom_study_uid,
            "ct": self.patient_ref.ct_series_uid,
            "rts": self.patient_ref.rtstruct_uid,
            "upload": self.patient_ref.upload_id,
            "grid": self.grid_spec.model_dump() if self.grid_spec else None,
            "hu_model": self.hu_to_density_model.value,
            "name_map": self._normalized_name_map(),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _normalized_name_map(self) -> Optional[Dict[str, List[str]]]:
        """Lowercase canonical name → sorted-lowercase alias list.

        Normalization makes the cache key invariant to stylistic
        differences that don't affect the output.
        """
        if not self.structure_name_map:
            return None
        return {
            k.lower(): sorted(a.lower() for a in v)
            for k, v in self.structure_name_map.items()
        }


class CTMetadata(BaseModel):
    """Minimal CT provenance carried through to downstream services."""

    patient_name: str = Field(default="ANONYMOUS")
    modality: str = Field(default="CT")
    num_slices: int = Field(..., ge=1)
    study_instance_uid: Optional[str] = None
    series_instance_uid: Optional[str] = None


class GeometryResult(BaseModel):
    """Output of a completed geometry build."""

    geometry_id: str
    density_grid_uri: str = Field(..., description="URI/path to the density volume (NIfTI).")
    structure_masks_uri: str = Field(
        ..., description="URI/path to the multi-label mask volume (NIfTI, uint16)."
    )
    ct_grid_uri: Optional[str] = Field(
        default=None,
        description=(
            "URI/path to the CT volume in Hounsfield Units (NIfTI, int16), "
            "resampled to ``grid_spec``. Optional — older geometries cached "
            "before CT persistence was added will have this set to null; "
            "engines that require a real CT (e.g. MCsquare) must handle the "
            "null case explicitly. Populated for all newly-built geometries."
        ),
    )
    structure_index: Dict[str, int] = Field(
        ..., description='Canonical name → integer label in the mask volume. 0 = background.'
    )
    grid_spec: GridSpec = Field(..., description="Grid the outputs were written on.")
    frame_of_reference_uid: str
    ct_metadata: CTMetadata
    cache_key: str

    created_at: Optional[datetime] = None

    @model_validator(mode="after")
    def _check_structure_index(self) -> "GeometryResult":
        if 0 in self.structure_index.values():
            raise ValueError("structure_index labels must be >= 1 (0 is reserved for background)")
        if len(set(self.structure_index.values())) != len(self.structure_index):
            raise ValueError("structure_index labels must be unique")
        return self


# ---------------------------------------------------------------------------
# Async job tracking
# ---------------------------------------------------------------------------

class GeometryStage(str, Enum):
    """Stages a :class:`GeometryService.build` call passes through.

    Reported via the ``stage`` field on ``GeometryJobStatus`` so clients
    polling ``GET /geometry/jobs/{job_id}`` can render a progress UI
    without waiting for the final result.
    """

    queued = "queued"
    loading_dicom = "loading_dicom"
    converting_hu = "converting_hu"
    rasterizing_contours = "rasterizing_contours"
    resampling = "resampling"
    persisting = "persisting"
    done = "done"


class GeometryJobStatus(BaseModel):
    """Tracks an async ``POST /geometry/build`` invocation.

    Unlike plan jobs this has no parent — it's identified by its own
    ``id`` and carries the ``cache_key`` so a second request for the same
    inputs can short-circuit to the cached geometry or reuse the
    in-flight job.
    """

    id: str
    cache_key: str
    state: JobState = JobState.queued
    progress: float = 0.0
    stage: Optional[GeometryStage] = GeometryStage.queued
    message: Optional[str] = None
    geometry_id: Optional[str] = None  # populated when state == succeeded
    eta_seconds: Optional[float] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
