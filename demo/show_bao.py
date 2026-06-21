"""Live demo of Service 5 — BAO (Beam Angle Optimization).

Runs Geometry → BAO end-to-end on the bundled SimpleFantom. BAO scores each
candidate angle set by building a beam model and running a short fluence
optimization (Service 4), then selects the best ``n_beams``.

Usage:
    python demo/show_bao.py
    python demo/show_bao.py --n-beams 3 --angle-step 45 --search greedy
    python demo/show_bao.py --search top_k --scoring-iters 20

Note: with the default analytic toy engine the per-angle scores are typically
flat (its kernel isn't target-aware — see show_optimization.py); use
`--engine mcsquare` for clinically meaningful angle selection.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

os.environ.setdefault("RADIARCH_ORTHANC_USE_MOCK", "true")
os.environ.setdefault("RADIARCH_DATABASE_URL", "")
os.environ.setdefault("RADIARCH_BROKER_URL", "memory://")
os.environ.setdefault("RADIARCH_RESULT_BACKEND", "cache+memory://")
os.environ.setdefault("RADIARCH_ARTIFACT_DIR", str(_REPO_ROOT / "data" / "artifacts"))
_TEST_DATA = (
    _REPO_ROOT / "tests" / "opentps" / "core" / "opentps-testData"
    / "SimpleFantomWithStruct"
)
os.environ.setdefault("RADIARCH_OPENTPS_DATA_ROOT", str(_TEST_DATA))

from radiarch.models.dose import EngineSpec  # noqa: E402
from radiarch.models.geometry import (  # noqa: E402
    GeometryBuildRequest, HUDensityModel, PatientRef,
)
from radiarch.models.optimization import ObjectiveSpec  # noqa: E402
from radiarch.models.bao import BAORunRequest  # noqa: E402
from radiarch.services.geometry import GeometryService  # noqa: E402
from radiarch.services.bao import BAOService  # noqa: E402

BAR = "─" * 64


def _h(label: str) -> None:
    print(f"\n{BAR}\n  {label}\n{BAR}")


def _row(key: str, value) -> None:
    print(f"  {key:<28} {value}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["analytic", "mcsquare"], default="analytic")
    p.add_argument("--n-beams", type=int, default=2)
    p.add_argument("--angle-step", type=float, default=90.0)
    p.add_argument("--search", choices=["greedy", "top_k"], default="greedy")
    p.add_argument("--scoring-iters", type=int, default=20)
    args = p.parse_args()

    _h("Radiarch BAO Service — Live Demo")
    _row("engine:", args.engine)
    _row("n_beams:", args.n_beams)
    _row("angle step (deg):", args.angle_step)
    _row("search:", args.search)

    if not _TEST_DATA.exists():
        print(f"ERROR: test data not found at {_TEST_DATA}", file=sys.stderr)
        sys.exit(1)

    _h("Step 1 — Geometry build")
    t0 = time.monotonic()
    geom = GeometryService().build(GeometryBuildRequest(
        patient_ref=PatientRef(dicom_study_uid="demo-study-001"),
        grid_spec=None, hu_to_density_model=HUDensityModel.stoichiometric,
    ))
    structures = list(geom.structure_index.keys())
    _row("geometry_id:", geom.geometry_id)
    _row("structures:", structures)
    _row("elapsed:", f"{(time.monotonic() - t0) * 1000:.1f} ms")

    _h(f"Step 2 — Beam-angle optimization ({args.search})")
    req = BAORunRequest(
        geometry_id=geom.geometry_id,
        dose_engine=EngineSpec(name=args.engine),
        objectives=[ObjectiveSpec(type="DUniform", structure_name=structures[0],
                                  dose_gy=10.0, weight=1.0)],
        n_beams=args.n_beams,
        angle_step_deg=args.angle_step,
        search=args.search,
        scoring_iterations=args.scoring_iters,
    )
    n_cand = len(req.resolve_candidates())
    _row("candidate angles:", n_cand)
    t0 = time.monotonic()
    result = BAOService().run(req)
    _row("bao_id:", result.bao_id)
    _row("selected angles:", [c.key() for c in result.selected_angles])
    _row("final score:", f"{result.final_score:.4g}")
    _row("beam_model_id:", result.beam_model_id)
    _row("elapsed:", f"{(time.monotonic() - t0):.1f} s")

    if result.per_angle_scores:
        _h("Per-angle scores")
        for s in sorted(result.per_angle_scores, key=lambda x: x.score):
            _row(f"gantry {s.gantry_deg:g}/{s.couch_deg:g}:", f"{s.score:.4g}")

    if result.selection_history:
        _h("Greedy selection history")
        for step in result.selection_history:
            _row(f"step {step.step}:",
                 f"+gantry {step.added_gantry_deg:g} → score {step.combined_score:.4g}")
    print()


if __name__ == "__main__":
    main()
