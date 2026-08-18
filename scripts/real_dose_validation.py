#!/usr/bin/env python3
"""Validate the full geometry -> beam-model -> dose pipeline on REAL DICOM.

This is the committed, reproducible version of the ad-hoc runs used to validate
Radiarch on real clinical CT (LCTSC, BREAST-DIAGNOSIS) and to exercise the real
MCsquare Monte Carlo proton engine on a real CT.

Unlike ``demo/show_dose.py`` (which hard-codes the bundled fantom path and
overrides ``RADIARCH_OPENTPS_DATA_ROOT`` at import), this script points every
stage at a caller-supplied DICOM directory, so it works on any real study —
and it sets the OpenTPS data root *before* importing radiarch, so the
beam-model build (which re-reads the patient for its target contour) finds the
same study the geometry was built from.

Engines
-------
* ``analytic`` — toy depth-falloff, pure numpy. Runs anywhere (macOS included).
  Validates the *plumbing* only; the dose distribution is not physical.
* ``mcsquare`` — real proton Monte Carlo. Linux/x86-64 only (use the Docker
  worker). Produces a genuine proton dose; keep ``--primaries`` low for a smoke
  test (statistically noisy but proves the engine runs on the CT).

Honesty
-------
This validates *ingestion + dose orchestration on real anatomy*. It does NOT
assert clinical dose accuracy: public RT structure sets here are OARs (no PTV),
so the plan aims at a fallback pseudo-target, weights are uniform (uncalibrated),
and a low primary count is deliberately noisy. Distinguish "it ran" from "it is
correct".

Examples
--------
    # Local plumbing check on a real CT+RTSTRUCT (macOS ok):
    python scripts/real_dose_validation.py \
        --data-root data/dicom/lctsc-s1-101 --engine analytic

    # Real proton dose in the Docker worker (Linux):
    #   docker cp scripts <worker>:/app/scripts   # if not baked in
    #   docker exec <worker> python scripts/real_dose_validation.py \
    #       --data-root /data/artifacts/lctsc_root --engine mcsquare --primaries 1e4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True,
                   help="Directory containing a DICOM CT series (+ optional RTSTRUCT).")
    p.add_argument("--engine", default="analytic",
                   choices=["analytic", "mcsquare"])
    p.add_argument("--hu-model", default="stoichiometric",
                   choices=["schneider", "stoichiometric", "linear"])
    p.add_argument("--primaries", type=float, default=1e4,
                   help="MCsquare primaries (ignored by analytic).")
    p.add_argument("--gantry", type=float, default=0.0)
    p.add_argument("--out", default=None, help="Artifact dir (default: <data-root>/../_validation_artifacts).")
    args = p.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        print(f"ERROR: data-root not found: {data_root}", file=sys.stderr)
        return 1
    out = Path(args.out).resolve() if args.out else (data_root.parent / "_validation_artifacts")

    # Set config BEFORE importing radiarch so get_settings() picks it up, and so
    # the beam-model build (which re-reads patient from opentps_data_root for its
    # target contour) sees the SAME study as the geometry build.
    os.environ["RADIARCH_OPENTPS_DATA_ROOT"] = str(data_root)
    os.environ["RADIARCH_ARTIFACT_DIR"] = str(out)
    os.environ.setdefault("RADIARCH_ORTHANC_USE_MOCK", "true")
    os.environ.setdefault("RADIARCH_FORCE_SYNTHETIC", "false")

    import numpy as np

    from radiarch.models.beam_model import (
        BeamModelBuildRequest, BeamSetSpec, BeamSpec, DeliveryParams, Modality,
    )
    from radiarch.models.dose import (
        DoseComputeRequest, EngineSpec, WeightVector,
    )
    from radiarch.models.geometry import (
        GeometryBuildRequest, HUDensityModel, PatientRef,
    )
    from radiarch.services.beam_model import BeamModelService
    from radiarch.services.dose import DoseService
    from radiarch.services.dose_engines import engine_health
    from radiarch.services.geometry import GeometryService

    print("=" * 64)
    print(f"  Real-data dose validation")
    print("=" * 64)
    print(f"  data-root : {data_root}")
    print(f"  engine    : {args.engine}")
    print(f"  hu-model  : {args.hu_model}")

    if args.engine == "mcsquare":
        h = engine_health("mcsquare")
        print(f"  mcsquare available: {h.get('available')}")
        if not h.get("available"):
            print("  ERROR: MCsquare not available here (needs OpenTPS + Linux binary; "
                  "run inside the Docker worker).", file=sys.stderr)
            return 2

    # --- Step 1: geometry from real DICOM -------------------------------
    geom = GeometryService().build(GeometryBuildRequest(
        patient_ref=PatientRef(dicom_study_uid=data_root.name),
        data_root_override=str(data_root),
        hu_to_density_model=HUDensityModel[args.hu_model],
    ))
    print(f"\n  [1] geometry_id : {geom.geometry_id}")
    print(f"      grid        : {geom.grid_spec.size}")
    print(f"      structures  : {list(geom.structure_index) or '(CT only)'}")

    # --- Step 2: proton beam model --------------------------------------
    bm = BeamModelService().build(BeamModelBuildRequest(
        geometry_id=geom.geometry_id,
        modality=Modality.proton_pbs,
        beam_set=BeamSetSpec(isocenter_mm=(0.0, 0.0, 0.0),
                             beams=[BeamSpec(beam_id="B1", gantry_deg=args.gantry)]),
        delivery_params=DeliveryParams(),
    ))
    n = bm.fluence_elements.total_count
    print(f"  [2] beam_model  : {bm.beam_model_id}  ({n} fluence elements)")

    # --- Step 3: dose ---------------------------------------------------
    params = {"nb_primaries": args.primaries} if args.engine == "mcsquare" else {}
    t0 = time.monotonic()
    dose = DoseService().compute_dose(DoseComputeRequest(
        geometry_id=geom.geometry_id,
        beam_model_id=bm.beam_model_id,
        engine=EngineSpec(name=args.engine, params=params),
        weights=WeightVector(length=n, values=[1.0] * n),
    ))
    dt = time.monotonic() - t0
    s = dose.statistics
    print(f"  [3] dose_id     : {dose.dose_id}   ({dt:.1f}s)")
    print(f"      max={s.max_gy:.3f} Gy  mean={s.mean_gy:.4f}  p95={s.p95_gy:.3f}  "
          f"nonzero={s.nonzero_voxel_count:,}")

    # --- Validation assertions (plumbing, not clinical accuracy) --------
    ok = True
    checks = [
        ("dose nonzero", s.nonzero_voxel_count > 0),
        ("max finite/positive", np.isfinite(s.max_gy) and s.max_gy > 0),
        ("mean finite/non-negative", np.isfinite(s.mean_gy) and s.mean_gy >= 0),
    ]
    print()
    for name, passed in checks:
        print(f"      {'PASS' if passed else 'FAIL'}  {name}")
        ok = ok and passed
    print("\n  RESULT:", "VALIDATED (ingestion + dose ran on real CT)" if ok else "FAILED")
    print("  Note: plumbing/physics-ran check only — NOT a clinical dose claim.")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
