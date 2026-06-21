"""``DoseService`` — the public entry point for Service 3.

Two public methods:

* :meth:`compute_dose` — geometry + beam model + weights + scenarios → dose.
* :meth:`build_influence` — geometry + beam model + (optional) scenario → Dij.

Each is content-addressable: identical inputs → same cache_key → reused
on-disk result, no engine call. The cache keys live in
:mod:`radiarch.models.dose`; the on-disk store is in
:mod:`radiarch.services.dose_persistence`.

The orchestrator is engine-agnostic — it dispatches through
:mod:`radiarch.services.dose_engines.registry`. Engine plugins receive
loaded *bundles* (density grid, masks, plan object) rather than raw ids,
so they don't depend on Radiarch's persistence layout.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import SimpleITK as sitk
from loguru import logger

from ..config import get_settings
from ..models.beam_model import Modality
from ..models.dose import (
    DoseComputeRequest,
    DoseResult,
    DoseStage,
    DoseStatistics,
    InfluenceBuildRequest,
    InfluenceResult,
    ScenarioDoseEntry,
    ScenarioSpec,
    WeightVector,
)
from .beam_persistence import BeamModelStore
from .dose_engines import (
    EngineRegistryError,
    get_engine,
)
from .dose_engines.protocol import (
    BeamModelBundle,
    DoseEnginePlugin,
    EngineParamError,
    EngineRuntimeError,
    EngineUnavailableError,
    GeometryBundle,
)
from .dose_persistence import (
    DOSE_FILENAME,
    DoseStore,
    InfluenceStore,
)
from .geometry import GeometryService
from .scenarios import expand_scenarios


# Stage + fraction + message — same shape as other services.
ProgressCallback = Callable[[DoseStage, float, str], None]


class DoseService:
    """Stateless orchestrator. Reused across requests; persistence on disk."""

    def __init__(self,
                 dose_dir: Optional[str | Path] = None,
                 influence_dir: Optional[str | Path] = None) -> None:
        if dose_dir is None or influence_dir is None:
            settings = get_settings()
            if dose_dir is None:
                dose_dir = Path(settings.artifact_dir) / "doses"
            if influence_dir is None:
                influence_dir = Path(settings.artifact_dir) / "influence"
        self.dose_store = DoseStore(dose_dir)
        self.influence_store = InfluenceStore(influence_dir)

    # -----------------------------------------------------------------
    # compute_dose
    # -----------------------------------------------------------------

    def compute_dose(
        self,
        request: DoseComputeRequest,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> DoseResult:
        on_progress = progress_callback or _noop_progress
        cache_key = request.compute_cache_key()

        cached = self.dose_store.lookup_by_cache_key(cache_key)
        if cached is not None:
            logger.info("Dose cache hit for key %s → %s", cache_key[:10], cached.dose_id)
            on_progress(DoseStage.done, 1.0, "cache hit")
            return cached

        on_progress(DoseStage.loading_geometry, 0.05, "Loading geometry")
        geometry = self._load_geometry(request.geometry_id)

        on_progress(DoseStage.loading_beam_model, 0.15, "Loading beam model")
        beam_model = self._load_beam_model(request.beam_model_id)

        # Cross-validate modality before any engine call.
        self._check_modality(request.engine.name, beam_model.result.modality)

        on_progress(DoseStage.validating_engine, 0.20, f"Validating engine {request.engine.name}")
        engine = get_engine(request.engine.name)
        issues = engine.validate(geometry, beam_model, request.engine.params)
        if issues:
            raise ValueError(
                f"Engine {request.engine.name} rejected the request: "
                + "; ".join(issues)
            )

        weights = self._materialize_weights(request.weights,
                                            beam_model.result.fluence_elements.total_count)

        # Scenario expansion (or nominal-only).
        if request.scenarios is not None:
            on_progress(DoseStage.expanding_scenarios, 0.25,
                        "Expanding scenarios")
            scenarios = expand_scenarios(request.scenarios)
        else:
            scenarios = [ScenarioSpec(name="nominal")]

        # Compute nominal dose (always first scenario).
        on_progress(DoseStage.computing_dose, 0.40,
                    f"Computing dose ({len(scenarios)} scenarios)")
        nominal_scenario = scenarios[0]
        nominal_result = engine.compute_dose(
            geometry, beam_model, weights,
            scenario=nominal_scenario, params=request.engine.params,
        )

        # Per-scenario doses for the rest.
        scenario_arrays: Dict[str, np.ndarray] = {}
        scenario_entries: List[ScenarioDoseEntry] = []
        if len(scenarios) > 1:
            for i, sc in enumerate(scenarios[1:], start=1):
                frac = 0.40 + 0.4 * (i / max(len(scenarios) - 1, 1))
                on_progress(DoseStage.computing_dose, frac,
                            f"Scenario {i}/{len(scenarios) - 1}: {sc.name}")
                arr = engine.compute_dose(
                    geometry, beam_model, weights,
                    scenario=sc, params=request.engine.params,
                ).dose
                h = sc.hash()
                scenario_arrays[h] = arr
                stats = _summary_stats(arr)
                scenario_entries.append(ScenarioDoseEntry(
                    scenario_name=sc.name,
                    scenario_hash=h,
                    dose_grid_uri="",  # filled in below once dose_id is known
                    statistics=stats,
                ))

        # Persist.
        on_progress(DoseStage.persisting, 0.90, "Writing dose volume")
        dose_id = str(uuid.uuid4())
        dose_root = self.dose_store.base_dir / dose_id
        dose_uri = str(dose_root / DOSE_FILENAME)
        # Patch each scenario_entry's uri now that we know dose_id.
        for entry in scenario_entries:
            entry.dose_grid_uri = str(dose_root / f"scenario_{entry.scenario_hash}.nii.gz")

        result = DoseResult(
            dose_id=dose_id,
            plan_id=request.plan_id,
            geometry_id=geometry.result.geometry_id,
            beam_model_id=beam_model.result.beam_model_id,
            modality=beam_model.result.modality,
            engine_name=engine.name,
            engine_version=engine.version,
            dose_grid_uri=dose_uri,
            statistics=_summary_stats(nominal_result.dose),
            scenario_doses=scenario_entries or None,
            cache_key=cache_key,
        )

        self.dose_store.save(
            dose_id=dose_id, cache_key=cache_key,
            nominal_dose=nominal_result.dose,
            spacing_mm=geometry.spacing_mm,
            scenario_doses=scenario_arrays or None,
            result=result,
        )

        on_progress(DoseStage.done, 1.0, f"dose_id={dose_id}")
        logger.info(
            "Dose %s built: engine=%s scenarios=%d max=%.3f Gy",
            dose_id, engine.name, len(scenarios),
            result.statistics.max_gy,
        )
        return result

    # -----------------------------------------------------------------
    # build_influence
    # -----------------------------------------------------------------

    def build_influence(
        self,
        request: InfluenceBuildRequest,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> InfluenceResult:
        on_progress = progress_callback or _noop_progress
        cache_key = request.compute_cache_key()

        cached = self.influence_store.lookup_by_cache_key(cache_key)
        if cached is not None:
            on_progress(DoseStage.done, 1.0, "cache hit")
            return cached

        on_progress(DoseStage.loading_geometry, 0.05, "Loading geometry")
        geometry = self._load_geometry(request.geometry_id)

        on_progress(DoseStage.loading_beam_model, 0.15, "Loading beam model")
        beam_model = self._load_beam_model(request.beam_model_id)

        self._check_modality(request.engine.name, beam_model.result.modality)

        on_progress(DoseStage.validating_engine, 0.20,
                    f"Validating engine {request.engine.name}")
        engine = get_engine(request.engine.name)
        issues = engine.validate(geometry, beam_model, request.engine.params)
        if issues:
            raise ValueError(
                f"Engine {request.engine.name} rejected the request: "
                + "; ".join(issues)
            )

        on_progress(DoseStage.computing_dose, 0.50, "Building Dij")
        inf = engine.build_influence(
            geometry, beam_model,
            scenario=request.scenario, params=request.engine.params,
        )

        influence_id = str(uuid.uuid4())
        influence_root = self.influence_store.base_dir / influence_id
        result = InfluenceResult(
            influence_id=influence_id,
            plan_id=request.plan_id,
            geometry_id=geometry.result.geometry_id,
            beam_model_id=beam_model.result.beam_model_id,
            modality=beam_model.result.modality,
            engine_name=engine.name,
            engine_version=engine.version,
            scenario=request.scenario,
            influence_uri=str(influence_root / "dij.npz"),
            n_voxels=inf.n_voxels,
            n_elements=inf.n_elements,
            nnz=inf.nnz,
            cache_key=cache_key,
        )

        on_progress(DoseStage.persisting, 0.90, "Writing Dij")
        self.influence_store.save(
            influence_id=influence_id, cache_key=cache_key,
            influence=inf, result=result,
        )
        on_progress(DoseStage.done, 1.0, f"influence_id={influence_id}")
        return result

    # -----------------------------------------------------------------
    # Loading — testability seams
    # -----------------------------------------------------------------

    def _load_geometry(self, geometry_id: str) -> GeometryBundle:
        """Look up the geometry; load density + masks (+ optional CT) NIfTI bundles.

        The CT is loaded from ``geom.ct_grid_uri`` when present. For
        geometries cached before D6.1 — or any future case where the CT
        file is missing — ``ct_hu`` and ``ct_image`` fall back to None
        and the engine is expected to surface a clean error if it needs
        them. ``ct_image`` is an OpenTPS ``CTImage`` wrapped around the
        HU array; when OpenTPS isn't importable we still populate
        ``ct_hu`` and leave ``ct_image=None`` so non-OpenTPS engines can
        still see the raw HU.
        """
        geom = GeometryService().store.get_by_id(geometry_id)
        if geom is None:
            raise ValueError(
                f"geometry_id {geometry_id!r} not found — build the geometry first."
            )
        density = sitk.GetArrayFromImage(sitk.ReadImage(geom.density_grid_uri))
        masks = sitk.GetArrayFromImage(sitk.ReadImage(geom.structure_masks_uri))
        spacing = tuple(geom.grid_spec.spacing_mm)

        ct_hu, ct_image = self._maybe_load_ct(geom)

        return GeometryBundle(
            result=geom,
            density=density.astype(np.float32),
            masks=masks.astype(np.uint16),
            spacing_mm=spacing,
            ct_hu=ct_hu,
            ct_image=ct_image,
        )

    @staticmethod
    def _maybe_load_ct(geom):
        """Read CT NIfTI + wrap in CTImage; returns (None, None) when unavailable.

        Three failure modes are tolerated silently:

        * ``ct_grid_uri`` is null (pre-D6.1 cached geometry).
        * The file on disk has been deleted out from under us.
        * OpenTPS isn't importable (e.g. lightweight test env).

        Read errors that aren't "file missing" are intentionally swallowed
        with a warning rather than raised — the bundle is still useful
        for engines that don't need a CT (analytic, future CCC).
        """
        uri = getattr(geom, "ct_grid_uri", None)
        if not uri:
            return None, None
        ct_path = Path(uri)
        if not ct_path.is_file():
            logger.warning(
                "Geometry %s declares ct_grid_uri=%s but the file is missing.",
                geom.geometry_id, uri,
            )
            return None, None
        try:
            ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path))).astype(np.int16)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("Failed to read CT NIfTI %s: %s", uri, exc)
            return None, None

        ct_image = _wrap_ct_image(ct_arr, geom)
        return ct_arr, ct_image

    def _load_beam_model(self, beam_model_id: str) -> BeamModelBundle:
        """Look up the beam model and unpickle its plan artifact."""
        settings = get_settings()
        bm_store = BeamModelStore(Path(settings.artifact_dir) / "beam_models")
        result = bm_store.get_by_id(beam_model_id)
        if result is None:
            raise ValueError(
                f"beam_model_id {beam_model_id!r} not found — "
                "build the beam model first."
            )
        plan: Any
        try:
            plan = bm_store.load_plan(beam_model_id)
        except FileNotFoundError:
            plan = None  # let the engine surface the missing artifact
        return BeamModelBundle(result=result, plan=plan)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _check_modality(engine_name: str, modality: Modality) -> None:
        """Cross-check engine.modalities against the beam-model modality."""
        try:
            engine = get_engine(engine_name)
        except EngineRegistryError as exc:
            raise ValueError(str(exc)) from exc
        if modality.value not in engine.modalities:
            raise ValueError(
                f"Engine {engine_name!r} does not support modality "
                f"{modality.value!r}; supports {engine.modalities}."
            )

    @staticmethod
    def _materialize_weights(weights: WeightVector, expected_length: int) -> np.ndarray:
        """Resolve inline values / URI into a numpy float32 array."""
        if weights.length != expected_length:
            raise ValueError(
                f"weights.length={weights.length} does not match beam-model "
                f"element count {expected_length}"
            )
        if weights.values is not None:
            return np.asarray(weights.values, dtype=np.float32)
        # URI mode — only file:// supported in v1.
        uri = weights.weights_uri or ""
        path = uri.removeprefix("file://") if uri.startswith("file://") else uri
        arr = np.load(path)
        return arr.astype(np.float32)


def _summary_stats(arr: np.ndarray) -> DoseStatistics:
    """Cheap summary stats for a dose array."""
    nonzero = arr[arr > 0]
    nz_count = int(nonzero.size)
    if nz_count == 0:
        return DoseStatistics(
            max_gy=0.0, mean_gy=0.0, p95_gy=0.0, nonzero_voxel_count=0,
        )
    return DoseStatistics(
        max_gy=float(arr.max()),
        mean_gy=float(arr.mean()),
        p95_gy=float(np.percentile(nonzero, 95)),
        nonzero_voxel_count=nz_count,
    )


def _noop_progress(stage: DoseStage, fraction: float, message: str) -> None:
    del stage, fraction, message


def _wrap_ct_image(ct_hu: np.ndarray, geom) -> Optional[Any]:
    """Wrap a HU array in an OpenTPS ``CTImage``, or return None.

    Returns None when OpenTPS isn't importable in this environment — the
    bundle still carries the raw ``ct_hu`` so non-OpenTPS engines (the
    analytic test engine, a future pure-numpy CCC) can use the HU
    directly. OpenTPS uses (x, y, z) array order; SimpleITK gave us
    (z, y, x), so we transpose on the way out.

    Provenance fields (frameOfReferenceUID, seriesInstanceUID) are
    pulled from the persisted GeometryResult.ct_metadata so the resulting
    CTImage matches the original DICOM series.
    """
    try:
        from opentps.core.data.images import CTImage
    except Exception:  # pragma: no cover — exercised on import-failure machines
        return None

    # SimpleITK → (z, y, x); OpenTPS expects (x, y, z).
    arr_ijk = np.transpose(np.asarray(ct_hu), (2, 1, 0)).astype(np.int16, copy=False)

    grid = geom.grid_spec
    spacing = tuple(float(s) for s in grid.spacing_mm)
    origin = tuple(float(o) for o in (grid.origin_mm or (0.0, 0.0, 0.0)))

    series_uid = ""
    try:
        series_uid = str(geom.ct_metadata.series_instance_uid or "")
    except Exception:  # pragma: no cover
        pass

    for_uid = str(getattr(geom, "frame_of_reference_uid", "") or "")

    return CTImage(
        imageArray=arr_ijk,
        name=f"CT[{geom.geometry_id}]",
        origin=origin,
        spacing=spacing,
        seriesInstanceUID=series_uid,
        frameOfReferenceUID=for_uid,
    )


__all__ = ["DoseService", "ProgressCallback"]
