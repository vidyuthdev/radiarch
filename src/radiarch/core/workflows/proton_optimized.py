"""Proton IMPT Optimized workflow — beamlet computation + L-BFGS-B optimization."""

from __future__ import annotations

import os
from typing import Any, Dict

from loguru import logger

from ._helpers import (
    PlannerError,
    build_gantry_angles,
    build_mc_calculator,
    build_objectives,
    compute_dvh,
    export_rtdose,
    find_body_roi,
    find_target_roi,
    load_ct_and_patient,
    setup_calibration,
    setup_sim_dir,
)
from ...models.plan import PlanDetail


def run(plan: PlanDetail) -> Dict[str, Any]:
    """Phase 8A: IMPT optimization pipeline."""
    logger.info("Starting IMPT optimization for plan %s", plan.id)

    from opentps.core.data.plan import ProtonPlanDesign
    from opentps.core.utils.programSettings import ProgramSettings
    from opentps.core.processing.planOptimization.planOptimization import IntensityModulationOptimizer

    sim_dir = setup_sim_dir(plan.id)
    ps = ProgramSettings()
    ps._config["dir"]["simulationFolder"] = sim_dir

    # 1. Load data
    ct, patient, _ = load_ct_and_patient()
    calibration, _ = setup_calibration()

    # 2. Plan design
    plan_design = ProtonPlanDesign()
    plan_design.calibration = calibration
    plan_design.ct = ct
    plan_design.patient = patient
    plan_design.gantryAngles = build_gantry_angles(plan.beam_count)
    plan_design.spotSpacing = plan.spot_spacing_mm
    plan_design.layerSpacing = plan.layer_spacing_mm

    target_roi = find_target_roi(patient)
    if target_roi:
        plan_design.defineTargetMaskAndPrescription(target_roi, plan.prescription_gy)
    else:
        logger.warning("No target ROI found, optimization may fail")

    logger.info("Building initial plan...")
    proton_plan = plan_design.buildPlan()

    # 3. Beamlets
    logger.info("Computing interaction matrix (beamlets)...")
    mc_calc = build_mc_calculator(ct, calibration, nb_primaries=plan.nb_primaries_beamlets)

    rois_for_calc = []
    if target_roi:
        rois_for_calc.append(target_roi)
    body_roi = find_body_roi(patient)
    if body_roi:
        rois_for_calc.append(body_roi)

    mc_calc.computeBeamlets(ct, proton_plan, roi=rois_for_calc)

    # 4. Objectives
    objectives = build_objectives(plan, patient, target_roi)
    proton_plan.planDesign.objectives = objectives

    # 5. Optimize
    logger.info("Running optimizer (%s)...", plan.optimization_method)
    solver = IntensityModulationOptimizer(
        plan=proton_plan,
        method=plan.optimization_method,
        maxiter=plan.max_iterations,
    )
    res = solver.optimize()
    logger.info("Optimization complete. Success: %s, Message: %s", res.success, res.message)

    # 6. Final dose
    logger.info("Computing final dose...")
    mc_calc.nbPrimaries = plan.nb_primaries_final
    dose_image = mc_calc.computeDose(ct, proton_plan)

    rtdose_path = os.path.join(sim_dir, "RTDOSE.dcm")
    export_rtdose(dose_image, ct, rtdose_path)

    return {
        "engine": "opentps_optimized",
        "beamCount": len(proton_plan.planDesign.gantryAngles),
        "maxDose": float(dose_image.imageArray.max()),
        "rtdosePath": rtdose_path,
        "simDir": sim_dir,
        "dvh": compute_dvh(dose_image, target_roi, ct) if target_roi else {},
        "optimization": {
            "success": bool(res.success),
            "iterations": int(res.nit),
            "final_cost": float(res.fun),
            # O18 — surface the solver method in qa_summary so the plan report
            # records which engine produced the weights, matching the
            # OptimizationService result contract (convergence + solver_method).
            "solver_method": plan.optimization_method,
        },
    }


# O18 note: the legacy plan path above runs OpenTPS' IntensityModulationOptimizer
# directly on OpenTPS objects (ct/patient/proton_plan). Re-routing it through
# radiarch.services.optimization.OptimizationService — which is engine-agnostic
# and keyed on geometry_id/beam_model_id — requires first building Services 1–2
# (Geometry, Beam Model) inside this workflow under the OpenTPS runtime. That
# refactor is deferred until the plan path migrates onto the microservice
# pipeline end-to-end; the OptimizationService is already wired for direct
# /api/v1/optimize/run use in the meantime.
