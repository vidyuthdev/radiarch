"""Live demo of Service 6 — Evaluation Service.

Runs Geometry → Beam Model → Dose → Evaluation on the bundled SimpleFantom and
prints a DVH + plan-quality report. Defaults to the analytic engine.

Usage:
    python demo/show_evaluation.py
    python demo/show_evaluation.py --prescription 20 --show
    python demo/show_evaluation.py --gamma          # gamma vs the same dose (sanity: 100%)
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

from radiarch.models.beam_model import (  # noqa: E402
    BeamModelBuildRequest, BeamSetSpec, BeamSpec, DeliveryParams, Modality,
)
from radiarch.models.dose import DoseComputeRequest, EngineSpec, WeightVector  # noqa: E402
from radiarch.models.evaluation import EvaluationRequest, GammaSpec  # noqa: E402
from radiarch.models.geometry import (  # noqa: E402
    GeometryBuildRequest, HUDensityModel, PatientRef,
)
from radiarch.services.beam_model import BeamModelService  # noqa: E402
from radiarch.services.dose import DoseService  # noqa: E402
from radiarch.services.evaluation import EvaluationService  # noqa: E402
from radiarch.services.geometry import GeometryService  # noqa: E402

BAR = "─" * 64


def _h(label: str) -> None:
    print(f"\n{BAR}\n  {label}\n{BAR}")


def _row(key: str, value) -> None:
    print(f"  {key:<28} {value}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["analytic", "mcsquare"], default="analytic")
    p.add_argument("--prescription", type=float, default=2.0)
    p.add_argument("--gamma", action="store_true",
                   help="Run a gamma comparison vs the same dose (sanity → 100%).")
    p.add_argument("--show", action="store_true", help="Render the DVH to preview.png.")
    args = p.parse_args()

    _h("Radiarch Evaluation Service — Live Demo")
    _row("engine:", args.engine)
    _row("prescription (Gy):", args.prescription)

    if not _TEST_DATA.exists():
        print(f"ERROR: test data not found at {_TEST_DATA}", file=sys.stderr)
        sys.exit(1)

    _h("Step 1 — Geometry build")
    geom = GeometryService().build(GeometryBuildRequest(
        patient_ref=PatientRef(dicom_study_uid="demo-study-001"),
        grid_spec=None, hu_to_density_model=HUDensityModel.stoichiometric,
    ))
    structures = list(geom.structure_index.keys())
    _row("geometry_id:", geom.geometry_id)
    _row("structures:", structures)

    _h("Step 2 — Beam-model + dose")
    bm = BeamModelService().build(BeamModelBuildRequest(
        geometry_id=geom.geometry_id, modality=Modality.proton_pbs,
        beam_set=BeamSetSpec(isocenter_mm=(0.0, 0.0, 0.0),
                             beams=[BeamSpec(beam_id="B1", gantry_deg=0.0)]),
        delivery_params=DeliveryParams(),
    ))
    n = bm.fluence_elements.total_count
    dose = DoseService().compute_dose(DoseComputeRequest(
        geometry_id=geom.geometry_id, beam_model_id=bm.beam_model_id,
        engine=EngineSpec(name=args.engine),
        weights=WeightVector(length=n, values=[1.0] * n),
    ))
    _row("dose_id:", dose.dose_id)
    _row("max dose (Gy):", f"{dose.statistics.max_gy:.3f}")

    _h("Step 3 — Evaluation")
    gamma_spec = GammaSpec(reference_dose_id=dose.dose_id) if args.gamma else None
    t0 = time.monotonic()
    result = EvaluationService().run(EvaluationRequest(
        dose_id=dose.dose_id, geometry_id=geom.geometry_id,
        prescription_gy=args.prescription, target_structure=structures[0],
        gamma=gamma_spec,
    ))
    _row("evaluation_id:", result.evaluation_id)
    _row("elapsed:", f"{(time.monotonic() - t0) * 1000:.1f} ms")

    _h("DVH summary")
    for c in result.dvh_curves:
        m = c.metrics
        _row(f"{c.structure_name}:",
             f"mean={m.mean_gy:.2f} max={m.max_gy:.2f} D95={m.d95_gy:.2f} "
             f"V_presc={m.v_prescription_pct:.0f}% vol={m.volume_cc:.1f}cc")

    if result.indices:
        _h("Plan indices (target)")
        _row("conformity index:", f"{result.indices.conformity_index:.3f}")
        _row("homogeneity index:", f"{result.indices.homogeneity_index:.3f}")
        _row("coverage:", f"{result.indices.coverage_pct:.1f}%")
        _row("hotspot (Gy):", f"{result.indices.hotspot_gy:.2f}")

    if result.gamma:
        _h("Gamma analysis")
        _row("criteria:", f"{result.gamma.dose_percent}%/{result.gamma.distance_mm}mm")
        _row("pass rate:", f"{result.gamma.pass_rate_pct:.1f}%")
        _row("mean gamma:", f"{result.gamma.mean_gamma:.3f}")

    if args.show:
        _show_dvh(result)
    else:
        print("\n  (pass --show to render the DVH)")
    print()


def _show_dvh(result) -> None:
    try:
        import matplotlib
        for backend in ("MacOSX", "TkAgg", "Qt5Agg", "Agg"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"\n  (--show needs matplotlib: {exc})")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in result.dvh_curves:
        ax.plot(c.dose_bins_gy, c.volume_pct, label=c.structure_name)
    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Volume (%)")
    ax.set_title(f"DVH — evaluation {result.evaluation_id[:8]}…")
    ax.legend()
    ax.grid(alpha=0.3)
    out = Path(os.environ["RADIARCH_ARTIFACT_DIR"]) / "evaluation" / \
        result.evaluation_id / "preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n  Saved preview → {out}")
    try:
        plt.show()
    except Exception as exc:
        print(f"  (GUI window not available: {exc})")


if __name__ == "__main__":
    main()
