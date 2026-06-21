"""MCsquare proton dose engine.

Adapter between Radiarch's :class:`DoseEnginePlugin` protocol and
OpenTPS's ``MCsquareDoseCalculator``. Covers the full nominal-dose
path including weight injection, scenario application, and graceful
degradation when MCsquare isn't importable.

The engine claims ``PROTON_PBS`` only. Beamlet-mode influence
(``build_influence``) is implemented in D8.1 via OpenTPS's
``computeBeamlets`` — see :func:`build_influence` below.

Design notes
------------
* **Graceful degradation.** Even when OpenTPS isn't importable in the
  current environment, the engine *registers itself* so the registry
  knows it exists. Every public method then raises
  :class:`EngineUnavailableError`, which the dispatch layer surfaces
  as HTTP 501. This keeps unit tests deterministic on machines that
  don't ship MCsquare binaries.
* **Weight application is hierarchy-aware.** OpenTPS plans don't
  guarantee a flat ``plan.spotMUs`` array — beams own layers and
  layers own spots, each with its own MU. We walk the structure and
  fan the weight vector out across ``plan.beams[*].layers[*].spotMUs``
  using :class:`FluenceElementSet` indexing from the beam model.
* **Scenarios mutate the *plan*, not the input geometry.** Density and
  range scales modify the CT calibration (via stopping-power
  multipliers); setup shift moves the plan isocenter. This is the
  TPS-correct way to model robustness perturbations — modifying
  ``geometry.density`` in-place (the previous behavior) corrupts the
  cached bundle for subsequent scenarios.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
from loguru import logger

from ...models.dose import ScenarioSpec
from .protocol import (
    BeamModelBundle,
    DoseEnginePlugin,
    EngineParamError,
    EngineRuntimeError,
    EngineUnavailableError,
    GeometryBundle,
    InfluenceData,
    NominalDose,
)
from .registry import register_engine


# ---------------------------------------------------------------------------
# OpenTPS availability probe
# ---------------------------------------------------------------------------

def _opentps_available() -> bool:
    """True when ``opentps.core`` can be imported in the running env."""
    try:
        import opentps.core  # noqa: F401
    except Exception:  # pragma: no cover — exercised on import-failure machines
        return False
    return True


def _opentps_diagnostics() -> dict:
    """Engine health-check payload (D6.7).

    Returns a JSON-serializable dict the ``/dose/engines`` route can
    surface so operators can see *why* the engine is down without
    digging into worker logs.
    """
    diag: dict = {"opentps_importable": False, "mcsquare_binary": None}
    try:
        import opentps.core  # noqa: F401
        diag["opentps_importable"] = True
    except Exception as exc:  # pragma: no cover
        diag["import_error"] = f"{type(exc).__name__}: {exc}"
        return diag

    try:
        from opentps.core.processing.doseCalculation.protons.mcsquareDoseCalculator import (
            MCsquareDoseCalculator,
        )
        calc = MCsquareDoseCalculator()
        bin_path = getattr(calc, "binPath", None) or getattr(calc, "_binPath", None)
        diag["mcsquare_binary"] = str(bin_path) if bin_path else "not introspectable"
    except Exception as exc:  # pragma: no cover
        diag["calculator_init_error"] = f"{type(exc).__name__}: {exc}"
    return diag


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass
class MCsquareEngine:
    """Proton dose engine backed by OpenTPS's MCsquare wrapper."""

    name: str = "mcsquare"
    version: str = "0.2.0"  # bumped for D6.2-D6.4 + D8.1
    modalities: List[str] = field(default_factory=lambda: ["PROTON_PBS"])

    # ----- diagnostics (D6.7) -----------------------------------------

    def health(self) -> dict:
        """Return a JSON-serializable engine health payload."""
        return {
            "name": self.name,
            "version": self.version,
            "modalities": list(self.modalities),
            "available": _opentps_available(),
            "supports": {
                "compute_dose": True,
                "build_influence": True,   # D8.1
                "compute_grad": False,     # future work
            },
            "diagnostics": _opentps_diagnostics(),
        }

    # ----- validation -------------------------------------------------

    def validate(self, geometry, beam_model, params: dict) -> List[str]:
        issues: List[str] = []
        if beam_model.result.modality.value != "PROTON_PBS":
            issues.append(
                f"mcsquare engine requires PROTON_PBS, got "
                f"{beam_model.result.modality.value}"
            )
        if geometry.density.ndim != 3:
            issues.append("geometry density must be 3D")
        if beam_model.result.fluence_elements.total_count < 1:
            issues.append("beam model has no fluence elements")
        # D6.1 — MCsquare needs a real OpenTPS CTImage. Surface during
        # validate so callers see 422, not 500 deep inside compute.
        if geometry.ct_image is None:
            issues.append(
                "mcsquare engine requires a CTImage on the geometry "
                "bundle; rebuild the geometry to populate ct_grid_uri "
                "(D6.1)."
            )
        return issues

    # ----- compute_dose (D6.2-D6.4) -----------------------------------

    def compute_dose(
        self,
        geometry: GeometryBundle,
        beam_model: BeamModelBundle,
        weights: np.ndarray,
        scenario: Optional[ScenarioSpec] = None,
        params: Optional[dict] = None,
    ) -> NominalDose:
        if not _opentps_available():
            raise EngineUnavailableError(
                "OpenTPS / MCsquare is not importable in this environment; "
                "use the analytic engine for tests, or install MCsquare to "
                "run the proton plugin."
            )

        expected = beam_model.result.fluence_elements.total_count
        if weights.shape != (expected,):
            raise EngineParamError(
                f"weights shape {weights.shape} != ({expected},)"
            )
        if geometry.ct_image is None:
            raise EngineUnavailableError(
                "MCsquare requires a CTImage on the geometry bundle. "
                "Rebuild the geometry to populate ct_grid_uri (D6.1)."
            )

        params = params or {}
        nb_primaries = float(params.get("nb_primaries", 1e4))

        # Clone plan + CT so this scenario doesn't poison the cached
        # bundle for the next scenario in the expansion loop.
        plan = self._clone_plan(beam_model.plan)
        ct_image = self._clone_ct(geometry.ct_image)

        # D6.3 — fan the weight vector across the plan's spot hierarchy.
        self._apply_weights_to_plan(plan, beam_model.result.fluence_elements, weights)

        # D6.4 — apply scenario perturbations to the *plan* (isocenter
        # shift) and *CT calibration* (density/range scales), never to
        # the input geometry bundle.
        ct_calibration = self._maybe_clone_calibration(
            getattr(beam_model, "ct_calibration", None)
            or getattr(geometry, "ct_calibration", None)
        )
        if scenario is not None:
            self._apply_scenario(plan, ct_image, ct_calibration, scenario)

        # Build + run MCsquare. Catch OpenTPS-specific failures and
        # re-raise as EngineRuntimeError so they bubble to the API as
        # a clean 500 with a typed message instead of an opaque trace.
        try:
            from opentps.core.processing.doseCalculation.protons.mcsquareDoseCalculator import (
                MCsquareDoseCalculator,
            )
        except Exception as exc:  # pragma: no cover
            raise EngineUnavailableError(f"MCsquare import failed: {exc}") from exc

        try:
            mc_calc = MCsquareDoseCalculator()
            mc_calc.nbPrimaries = nb_primaries
            # ct is passed as a *positional argument* to computeDose
            # in OpenTPS v3 — the calculator sets self.ct itself.
            # Setting mc_calc.ct externally is a no-op (and previously
            # masked the missing-arg bug).
            if ct_calibration is not None:
                # OpenTPS calls it ctCalibration in newer versions and
                # calibration in older ones — try both.
                if hasattr(mc_calc, "ctCalibration"):
                    mc_calc.ctCalibration = ct_calibration
                elif hasattr(mc_calc, "calibration"):
                    mc_calc.calibration = ct_calibration
            # Beam model (BDL) hand-off when provided.
            bdl = getattr(beam_model, "bdl", None)
            if bdl is not None and hasattr(mc_calc, "beamModel"):
                mc_calc.beamModel = bdl
            # Optional MCsquare runtime knobs (params= overrides).
            for attr, key in (
                ("statUncertainty", "stat_uncertainty"),
                ("nbThreads", "nb_threads"),
                ("doseGrid", "dose_grid"),
            ):
                if key in params and hasattr(mc_calc, attr):
                    setattr(mc_calc, attr, params[key])

            # computeDose(ct, plan) — ct is positional. See
            # opentps/core/processing/doseCalculation/protons/
            # mcsquareDoseCalculator.py:282.
            dose_image = mc_calc.computeDose(ct_image, plan)
        except (EngineParamError, EngineUnavailableError):
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("MCsquare dose computation failed")
            raise EngineRuntimeError(f"MCsquare failed: {exc}") from exc

        arr = np.asarray(dose_image.imageArray, dtype=np.float32)
        return NominalDose(dose=arr)

    # ----- build_influence (D8.1) -------------------------------------

    def build_influence(
        self,
        geometry: GeometryBundle,
        beam_model: BeamModelBundle,
        scenario: Optional[ScenarioSpec] = None,
        params: Optional[dict] = None,
    ) -> InfluenceData:
        """Beamlet-mode MCsquare → sparse CSR Dij.

        D8.1 — uses OpenTPS's ``computeBeamlets`` (or
        ``computeBeamletsByFluence`` depending on version) which
        returns a ``SparseBeamlets`` object. We re-pack into the
        CSR-friendly ``InfluenceData`` shape Radiarch persists.
        """
        if not _opentps_available():
            raise EngineUnavailableError(
                "MCsquare / OpenTPS unavailable; cannot build influence."
            )
        if geometry.ct_image is None:
            raise EngineUnavailableError(
                "MCsquare requires a CTImage on the geometry bundle."
            )

        params = params or {}
        nb_primaries = float(params.get("nb_primaries", 1e4))

        plan = self._clone_plan(beam_model.plan)
        ct_image = self._clone_ct(geometry.ct_image)
        ct_calibration = self._maybe_clone_calibration(
            getattr(beam_model, "ct_calibration", None)
            or getattr(geometry, "ct_calibration", None)
        )
        if scenario is not None:
            self._apply_scenario(plan, ct_image, ct_calibration, scenario)

        try:
            from opentps.core.processing.doseCalculation.protons.mcsquareDoseCalculator import (
                MCsquareDoseCalculator,
            )
        except Exception as exc:  # pragma: no cover
            raise EngineUnavailableError(f"MCsquare import failed: {exc}") from exc

        try:
            mc_calc = MCsquareDoseCalculator()
            mc_calc.nbPrimaries = nb_primaries
            # ct is positional to computeBeamlets in OpenTPS v3 —
            # don't set it as an attribute.
            if ct_calibration is not None and hasattr(mc_calc, "ctCalibration"):
                mc_calc.ctCalibration = ct_calibration
            bdl = getattr(beam_model, "bdl", None)
            if bdl is not None and hasattr(mc_calc, "beamModel"):
                mc_calc.beamModel = bdl

            # Different OpenTPS versions name the beamlet method
            # differently; try the most common ones in order.
            beamlet_method = (
                getattr(mc_calc, "computeBeamlets", None)
                or getattr(mc_calc, "computeBeamletsByFluence", None)
            )
            if beamlet_method is None:
                raise EngineUnavailableError(
                    "Installed OpenTPS lacks computeBeamlets — upgrade "
                    "to a build with beamlet-mode MCsquare to use "
                    "influence builds."
                )
            # computeBeamlets(ct, plan) — ct is positional. The type
            # hint says Sequence[CTImage] (for 4D dose), but for 3D
            # the impl handles a single CTImage too — it just does
            # `self.ct = ct` and uses it. See
            # mcsquareDoseCalculator.py:438.
            #
            # NOTE: this internally overrides plan.spotMUs to ones
            # (one beamlet per spot at unit weight), so any earlier
            # _apply_weights_to_plan call is correctly ignored —
            # weights get applied later via Dij @ w.
            sparse_beamlets = beamlet_method(ct_image, plan)
        except (EngineUnavailableError, EngineParamError):
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("MCsquare beamlet computation failed")
            raise EngineRuntimeError(f"MCsquare beamlet failed: {exc}") from exc

        # Convert OpenTPS SparseBeamlets → scipy CSR.
        try:
            csr = sparse_beamlets.toSparseMatrix().tocsr().astype(np.float32)
        except AttributeError:
            csr = sparse_beamlets.matrix.tocsr().astype(np.float32)

        n_voxels = csr.shape[0]
        n_elements = csr.shape[1]
        expected_elements = beam_model.result.fluence_elements.total_count
        if n_elements != expected_elements:
            raise EngineRuntimeError(
                f"MCsquare returned Dij with {n_elements} columns; "
                f"beam model has {expected_elements} fluence elements."
            )

        return InfluenceData(
            indptr=csr.indptr.astype(np.int64),
            indices=csr.indices.astype(np.int32),
            data=csr.data.astype(np.float32),
            n_voxels=n_voxels,
            n_elements=n_elements,
        )

    # ----- apply_influence (matvec) -----------------------------------

    def apply_influence(self, influence, weights, grid_shape):
        """Standard CSR matvec: dose = Dij @ w."""
        from scipy.sparse import csr_matrix
        csr = csr_matrix(
            (influence.data, influence.indices, influence.indptr),
            shape=(influence.n_voxels, influence.n_elements),
        )
        dose_flat = csr @ weights.astype(np.float32)
        return NominalDose(dose=dose_flat.reshape(grid_shape).astype(np.float32))

    def compute_grad(self, geometry, beam_model, weights, dL_dDose,
                     scenario=None, params=None):
        raise EngineUnavailableError(
            "mcsquare doesn't implement adjoint gradients. Build a Dij "
            "via build_influence() and compute the gradient as "
            "Dij.T @ dL_dDose in the optimizer."
        )

    # ----- internal helpers (D6.3, D6.4) ------------------------------

    @staticmethod
    def _clone_plan(plan: Any) -> Any:
        """Deep-copy the OpenTPS plan for per-scenario isolation."""
        if hasattr(plan, "copy") and callable(plan.copy):
            try:
                return plan.copy()
            except Exception:  # pragma: no cover
                pass
        return copy.deepcopy(plan)

    @staticmethod
    def _clone_ct(ct_image: Any) -> Any:
        if hasattr(ct_image, "copy") and callable(ct_image.copy):
            try:
                return ct_image.copy()
            except Exception:  # pragma: no cover
                pass
        return copy.deepcopy(ct_image)

    @staticmethod
    def _maybe_clone_calibration(cal: Any) -> Any:
        if cal is None:
            return None
        try:
            return copy.deepcopy(cal)
        except Exception:  # pragma: no cover
            return cal

    @staticmethod
    def _apply_weights_to_plan(
        plan: Any,
        fluence_elements: Any,
        weights: np.ndarray,
    ) -> None:
        """Push the weight vector into the per-spot MU fields of the plan.

        Strategy:

        1. If the plan exposes a flat ``spotMUs`` array sized to the
           full element count, set it directly — that's the fast path
           and matches OpenTPS's pre-built ``ProtonPlan`` layout.
        2. Otherwise, walk ``plan.beams[*].layers[*]`` and slice the
           weight vector using the beam model's
           ``FluenceElementSet.per_beam`` index ranges. This handles
           the case where layers have variable spot counts.
        """
        weights = weights.astype(np.float32)

        flat = getattr(plan, "spotMUs", None)
        if flat is not None and hasattr(flat, "size") and flat.size == weights.size:
            plan.spotMUs = weights
            return

        # Hierarchical fan-out path.
        beams = getattr(plan, "beams", None)
        if beams is None:
            raise EngineParamError(
                "OpenTPS plan exposes neither .spotMUs nor .beams — "
                "cannot apply weights. Plan type: " + type(plan).__name__
            )
        per_beam = getattr(fluence_elements, "per_beam", None)
        if per_beam is None or len(per_beam) != len(beams):
            raise EngineParamError(
                f"FluenceElementSet has {len(per_beam) if per_beam else 0} "
                f"beam entries; plan has {len(beams)}."
            )

        cursor = 0
        for beam_idx, beam in enumerate(beams):
            beam_meta = per_beam[beam_idx]
            layers = getattr(beam, "layers", None) or [beam]
            per_layer_counts = getattr(beam_meta, "per_layer", None) or [
                beam_meta.spot_count
            ]
            for layer, count in zip(layers, per_layer_counts):
                slice_ = weights[cursor : cursor + count]
                if hasattr(layer, "spotMUs"):
                    layer.spotMUs = slice_.copy()
                elif hasattr(layer, "spots"):
                    for spot, w in zip(layer.spots, slice_):
                        if hasattr(spot, "spotMU"):
                            spot.spotMU = float(w)
                        elif hasattr(spot, "MU"):
                            spot.MU = float(w)
                cursor += count

        if cursor != weights.size:
            raise EngineParamError(
                f"Weight fan-out covered {cursor} spots; expected "
                f"{weights.size}. Beam model / plan are out of sync."
            )

    @staticmethod
    def _apply_scenario(
        plan: Any,
        ct_image: Any,
        ct_calibration: Any,
        scenario: ScenarioSpec,
    ) -> None:
        """Apply a robustness perturbation to the cloned plan + CT.

        * ``setup_shift_mm`` → shift every beam's isocenter (mm in patient
          coords). Equivalent to moving the patient.
        * ``density_scale`` → multiplicative scale on density (via
          calibration if available, else direct CT mutation).
        * ``range_scale`` → R_perturbed / R_nominal; the
          stopping-power-equivalent of density^-1, so we apply
          1/range_scale to the calibration's density mapping.
        """
        # ---- setup shift (plan isocenter) ----------------------------
        if scenario.setup_shift_mm is not None:
            shift = np.asarray(scenario.setup_shift_mm, dtype=np.float32)
            if shift.shape != (3,):
                raise EngineParamError(
                    f"setup_shift_mm must be a 3-vector, got shape {shift.shape}"
                )
            for beam in getattr(plan, "beams", []) or []:
                iso = getattr(beam, "isocenterPosition", None)
                if iso is not None:
                    beam.isocenterPosition = np.asarray(iso, dtype=np.float32) + shift
                elif hasattr(beam, "isocenter"):
                    beam.isocenter = np.asarray(beam.isocenter, dtype=np.float32) + shift

        # ---- density / range scales (CT or calibration) --------------
        scale = 1.0
        if scenario.density_scale is not None:
            scale *= float(scenario.density_scale)
        if scenario.range_scale is not None:
            scale *= 1.0 / float(scenario.range_scale)

        if scale != 1.0:
            applied = False
            if ct_calibration is not None:
                for attr in ("hu2density", "_density", "densityValues"):
                    table = getattr(ct_calibration, attr, None)
                    if table is not None and hasattr(table, "__mul__"):
                        try:
                            setattr(ct_calibration, attr, table * scale)
                            applied = True
                            break
                        except Exception:  # pragma: no cover
                            continue
            if not applied and ct_image is not None and hasattr(ct_image, "imageArray"):
                arr = np.asarray(ct_image.imageArray, dtype=np.float32)
                # Density scale ≈ HU scaling for soft tissue. An
                # approximation, but matches what the analytic engine
                # does so cross-engine comparison stays meaningful.
                ct_image.imageArray = (arr * scale).astype(arr.dtype)


# Register on import — even when OpenTPS is unavailable, so the
# registry exposes the engine's existence (and surfaces a clean error).
register_engine(MCsquareEngine())


__all__ = ["MCsquareEngine", "_opentps_available", "_opentps_diagnostics"]
