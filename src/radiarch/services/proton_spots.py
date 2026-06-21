"""Generate a proton-PBS beam model from a geometry + beam set.

This is the adapter between Service 2's modality-neutral types
(:class:`BeamSetSpec`, :class:`DeliveryParams`, :class:`FluenceElementSet`)
and OpenTPS's ``ProtonPlanDesign.buildPlan()``. The function below is
intentionally the *only* place that knows how to call OpenTPS for proton
plan construction; the rest of the Beam Model Service treats spots as
opaque "fluence elements."

OpenTPS's ``buildPlan()`` returns a ``ProtonPlan`` whose ``beams`` list
contains ``PlanProtonBeam`` objects, each with ``layers`` (per energy)
that hold ``spots`` (per scanning position). We walk that structure to
populate :class:`PerBeamElements` with energy_layers + spots_per_layer.

The returned ``ProtonPlan`` is the artifact persisted by the Beam Model
Service (pickled). Downstream dose engines consume it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from loguru import logger

from ..models.beam_model import (
    BeamSetSpec,
    DeliveryParams,
    FluenceElementSet,
    Modality,
    PerBeamElements,
)
from .machine_model import ProtonMachineModel


@dataclass
class ProtonBuildResult:
    """Internal bundle — the FluenceElementSet plus the OpenTPS plan."""

    fluence_elements: FluenceElementSet
    plan: Any  # OpenTPS ProtonPlan (or test double)


def generate_proton_spots(
    ct: Any,
    patient: Any,
    target_contour: Any,
    machine_model: ProtonMachineModel,
    beam_set: BeamSetSpec,
    params: DeliveryParams,
    prescription_gy: float = 2.0,
) -> ProtonBuildResult:
    """Build a proton plan and summarize its fluence elements.

    Parameters
    ----------
    ct
        OpenTPS ``CTImage`` for the patient.
    patient
        OpenTPS ``Patient`` carrying the RTStructs.
    target_contour
        The target ``ROIContour`` to define the prescription against.
        May be ``None`` — the plan still builds but with no target mask,
        which exercises the "no target" warning path inside OpenTPS.
    machine_model
        Resolved :class:`ProtonMachineModel`.
    beam_set
        Geometric beam configuration.
    params
        :class:`DeliveryParams` carrying ``spot_spacing_mm`` and
        ``layer_spacing_mm``.
    prescription_gy
        Dose prescription, in Gy. Default 2.0 — clinical norm for one
        fraction. Used by ``defineTargetMaskAndPrescription``.

    Returns
    -------
    ProtonBuildResult
        ``fluence_elements`` for the cache + the raw ``ProtonPlan`` for
        persistence.
    """
    # Lazy import — keeps fast tests from pulling in OpenTPS.
    from opentps.core.data.plan import ProtonPlanDesign

    plan_design = ProtonPlanDesign()
    plan_design.ct = ct
    plan_design.patient = patient
    plan_design.calibration = machine_model.calibration

    # Convert BeamSetSpec → flat angle lists in beam order.
    plan_design.gantryAngles = [b.gantry_deg for b in beam_set.beams]
    plan_design.couchAngles = [b.couch_deg for b in beam_set.beams]

    if params.spot_spacing_mm is not None:
        plan_design.spotSpacing = params.spot_spacing_mm
    if params.layer_spacing_mm is not None:
        plan_design.layerSpacing = params.layer_spacing_mm

    if target_contour is not None:
        plan_design.defineTargetMaskAndPrescription(target_contour, prescription_gy)
    else:
        logger.warning(
            "No target contour supplied — proton plan will build without a "
            "prescription mask and may have zero spots."
        )

    logger.info(
        "Building proton plan: %d beams, spot_spacing=%s mm, layer_spacing=%s mm",
        len(beam_set.beams),
        plan_design.spotSpacing,
        plan_design.layerSpacing,
    )
    proton_plan = plan_design.buildPlan()

    fluence = _summarize_proton_plan(proton_plan, beam_set)
    return ProtonBuildResult(fluence_elements=fluence, plan=proton_plan)


# ---------------------------------------------------------------------------
# Plan introspection
# ---------------------------------------------------------------------------

def _summarize_proton_plan(plan: Any, beam_set: BeamSetSpec) -> FluenceElementSet:
    """Walk a built ``ProtonPlan`` and produce a :class:`FluenceElementSet`.

    OpenTPS internal layout (best-effort, defensive to API drift):
      plan.beams  -> [PlanProtonBeam]
        beam.layers  -> [PlanProtonLayer]
          layer.nominalEnergy : float (MeV)
          layer.spots         : list   (or .numberOfSpots / .scanSpotPositions)
    """
    per_beam: List[PerBeamElements] = []
    total = 0

    beams = list(getattr(plan, "beams", []) or [])
    # Align to BeamSetSpec by index when possible — OpenTPS preserves
    # construction order from gantryAngles.
    for idx, beam in enumerate(beams):
        beam_id = (
            beam_set.beams[idx].beam_id
            if idx < len(beam_set.beams)
            else f"beam_{idx}"
        )
        energies, spots = _walk_layers(beam)
        element_count = sum(spots)
        total += element_count
        per_beam.append(
            PerBeamElements(
                beam_id=beam_id,
                element_count=element_count,
                energy_layers=energies if energies else None,
                spots_per_layer=spots if spots else None,
            )
        )

    # If the plan ended up empty (no target / no spots placed), still
    # emit one PerBeamElements per beam in the request so downstream
    # code can find them by beam_id.
    if not per_beam:
        per_beam = [
            PerBeamElements(beam_id=b.beam_id, element_count=0)
            for b in beam_set.beams
        ]

    return FluenceElementSet(total_count=total, per_beam=per_beam)


def _walk_layers(beam: Any) -> Tuple[List[float], List[int]]:
    """Extract (energies, spots-per-layer) from one PlanProtonBeam.

    OpenTPS exposes spot lists under a few different attribute names
    across versions — try the most common shapes and fall back to zero
    if none match.
    """
    energies: List[float] = []
    spots: List[int] = []

    layers = list(getattr(beam, "layers", []) or [])
    for layer in layers:
        energy = (
            getattr(layer, "nominalEnergy", None)
            or getattr(layer, "energy", None)
            or 0.0
        )
        count = _count_spots(layer)
        # Skip only truly empty layer entries (no energy AND no count).
        if count == 0 and float(energy) == 0.0:
            continue
        energies.append(float(energy))
        spots.append(int(count))
    return energies, spots


def _count_spots(layer: Any) -> int:
    """Try several OpenTPS attribute paths to count spots in a layer.

    The canonical OpenTPS PlanProtonLayer exposes:
      * ``spotMUs`` — numpy array of monitor units (one per spot)
      * ``spotXY`` — list of (x, y) positions
      * ``numberOfSpots`` — @property returning len(spotXY)

    We check spotMUs first because it's what dose calculation actually
    consumes (a spot with MU=0 is still a spot but won't deposit dose).
    """
    # OpenTPS canonical path: spotMUs is the source of truth.
    mus = getattr(layer, "spotMUs", None)
    if mus is not None and hasattr(mus, "__len__"):
        n = len(mus)
        if n > 0:
            return n
    # spotXY positions (some OpenTPS code paths populate this first).
    xy = getattr(layer, "spotXY", None)
    if xy is not None and hasattr(xy, "__len__"):
        n = len(xy)
        if n > 0:
            return n
    # The @property — works on any OpenTPS version.
    n = getattr(layer, "numberOfSpots", None)
    if isinstance(n, int) and n > 0:
        return n
    # Legacy / test-double attribute names.
    spots = getattr(layer, "spots", None)
    if spots is not None and hasattr(spots, "__len__"):
        return len(spots)
    pos = getattr(layer, "scanSpotPositions", None)
    if pos is not None and hasattr(pos, "__len__"):
        return len(pos)
    return 0


__all__ = ["ProtonBuildResult", "generate_proton_spots"]
