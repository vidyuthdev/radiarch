"""Live demo of Service 4 — Optimization Service.

Runs Geometry → Beam Model → Optimization end-to-end and prints a convergence
summary. Defaults to the analytic engine so it runs without OpenTPS / MCsquare.

Usage:
    # 1. Bundled SimpleFantom + analytic engine (fastest)
    python demo/show_optimization.py
    python demo/show_optimization.py --show

    # 2. Custom objectives  (name=type:dose_gy, comma-separated)
    python demo/show_optimization.py --objectives ptv_dmin=60,oar_dmax=20

    # 3. Robust optimization
    python demo/show_optimization.py --robust 9 --aggregation WORST_CASE

    # 4. Different solver / iteration budget
    python demo/show_optimization.py --solver Adam --max-iters 500

    # 5. Real DICOM study
    python demo/show_optimization.py --upload /path/to/study.zip
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

os.environ.setdefault("RADIARCH_ORTHANC_USE_MOCK", "true")
os.environ.setdefault("RADIARCH_DATABASE_URL", "")
os.environ.setdefault("RADIARCH_BROKER_URL", "memory://")
os.environ.setdefault("RADIARCH_RESULT_BACKEND", "cache+memory://")
os.environ.setdefault("RADIARCH_DICOMWEB_URL", "")
os.environ.setdefault("RADIARCH_ARTIFACT_DIR", str(_REPO_ROOT / "data" / "artifacts"))
_TEST_DATA = (
    _REPO_ROOT / "tests" / "opentps" / "core" / "opentps-testData"
    / "SimpleFantomWithStruct"
)
os.environ.setdefault("RADIARCH_OPENTPS_DATA_ROOT", str(_TEST_DATA))


import numpy as np  # noqa: E402

from radiarch.models.beam_model import (  # noqa: E402
    BeamModelBuildRequest,
    BeamSetSpec,
    BeamSpec,
    DeliveryParams,
    Modality,
)
from radiarch.models.dose import EngineSpec, ScenarioSpec  # noqa: E402
from radiarch.models.geometry import (  # noqa: E402
    GeometryBuildRequest,
    HUDensityModel,
    PatientRef,
)
from radiarch.models.optimization import (  # noqa: E402
    ObjectiveSpec,
    OptimizationRunRequest,
    RobustnessSpec,
    SolverConfig,
)
from radiarch.services.beam_model import BeamModelService  # noqa: E402
from radiarch.services.geometry import GeometryService  # noqa: E402
from radiarch.services.optimization import OptimizationService  # noqa: E402

BAR = "─" * 64


def _h(label: str) -> None:
    print(f"\n{BAR}\n  {label}\n{BAR}")


def _row(key: str, value) -> None:
    print(f"  {key:<28} {value}")


def _ingest_upload_zip(zip_path: Path) -> str:
    import shutil
    import uuid
    import zipfile

    from radiarch.config import get_settings

    settings = get_settings()
    base = settings.upload_dir or str(Path(settings.artifact_dir) / "uploads")
    upload_root = Path(base).expanduser().resolve()
    upload_root.mkdir(parents=True, exist_ok=True)
    upload_id = str(uuid.uuid4())
    dest = upload_root / upload_id
    dest.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        dest_resolved = dest.resolve()
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if dest_resolved not in target.parents and target != dest_resolved:
                shutil.rmtree(dest, ignore_errors=True)
                raise ValueError(f"Refusing unsafe ZIP entry: {member.filename!r}")
        zf.extractall(dest)
    print(f"  extracted {sum(1 for p in dest.rglob('*.dcm'))} .dcm files into {dest}")
    return upload_id


def _parse_objectives(spec: Optional[str], structures: List[str]) -> List[ObjectiveSpec]:
    """Parse ``name=type:dose`` CSV, or fall back to sensible defaults.

    ``name`` matches (case-insensitively) a structure; ``type`` is one of the
    objective types (dmin/dmax/duniform); ``dose`` is the target in Gy. Example:
    ``ptv_dmin=60,oar_dmax=20`` → DMin on the PTV-like structure at 60 Gy and
    DMax on the OAR-like one at 20 Gy.
    """
    type_map = {"dmin": "DMin", "dmax": "DMax", "duniform": "DUniform"}
    if not spec:
        # Default: drive the first structure toward a uniform 10 Gy.
        target = structures[0]
        return [ObjectiveSpec(type="DUniform", structure_name=target,
                              dose_gy=10.0, weight=1.0)]
    out: List[ObjectiveSpec] = []
    for token in spec.split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        name_part, rhs = token.split("=", 1)
        kind, dose_s = (rhs.split(":", 1) if ":" in rhs else (name_part.split("_")[-1], rhs))
        kind = type_map.get(kind.lower().split("_")[-1], "DUniform")
        # Resolve the structure by fuzzy prefix match.
        key = name_part.lower().split("_")[0]
        match = next((s for s in structures if key in s.lower()), structures[0])
        out.append(ObjectiveSpec(type=kind, structure_name=match,
                                 dose_gy=float(dose_s), weight=1.0))
    return out or [ObjectiveSpec(type="DUniform", structure_name=structures[0],
                                 dose_gy=10.0, weight=1.0)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upload", type=str, default=None)
    p.add_argument("--engine", choices=["analytic", "mcsquare"], default="analytic")
    p.add_argument("--objectives", type=str, default=None,
                   help="CSV of name=type:dose_gy, e.g. ptv_dmin=60,oar_dmax=20")
    p.add_argument("--solver", choices=["L-BFGS-B", "Adam", "ProjectedGradient"],
                   default="L-BFGS-B")
    p.add_argument("--max-iters", type=int, default=200)
    p.add_argument("--robust", type=int, default=0,
                   help="Number of robustness scenarios (0 = nominal only).")
    p.add_argument("--aggregation", choices=["WORST_CASE", "EXPECTED", "CVAR"],
                   default="EXPECTED")
    p.add_argument("--show", action="store_true",
                   help="Render DVH + dose slice + convergence to preview.png.")
    args = p.parse_args()

    _h("Radiarch Optimization Service — Live Demo")
    _row("engine:", args.engine)
    _row("solver:", args.solver)
    _row("max iters:", args.max_iters)
    _row("robust scenarios:", args.robust)

    # ---- Geometry ------------------------------------------------------
    if args.upload:
        upload_zip = Path(args.upload).expanduser().resolve()
        if not upload_zip.is_file():
            print(f"ERROR: upload not found at {upload_zip}", file=sys.stderr)
            sys.exit(1)
        patient_ref = PatientRef(upload_id=_ingest_upload_zip(upload_zip))
    else:
        if not _TEST_DATA.exists():
            print(f"ERROR: test data not found at {_TEST_DATA}", file=sys.stderr)
            sys.exit(1)
        patient_ref = PatientRef(dicom_study_uid="demo-study-001")

    _h("Step 1 — Geometry build")
    t0 = time.monotonic()
    geom = GeometryService().build(GeometryBuildRequest(
        patient_ref=patient_ref, grid_spec=None,
        hu_to_density_model=HUDensityModel.stoichiometric,
    ))
    structures = list(geom.structure_index.keys())
    _row("geometry_id:", geom.geometry_id)
    _row("structures:", structures)
    _row("elapsed:", f"{(time.monotonic() - t0) * 1000:.1f} ms")

    # ---- Beam model ----------------------------------------------------
    _h("Step 2 — Beam-model build")
    t0 = time.monotonic()
    bm = BeamModelService().build(BeamModelBuildRequest(
        geometry_id=geom.geometry_id, modality=Modality.proton_pbs,
        beam_set=BeamSetSpec(isocenter_mm=(0.0, 0.0, 0.0),
                             beams=[BeamSpec(beam_id="B1", gantry_deg=0.0)]),
        delivery_params=DeliveryParams(),
    ))
    _row("beam_model_id:", bm.beam_model_id)
    _row("total elements:", bm.fluence_elements.total_count)
    _row("elapsed:", f"{(time.monotonic() - t0) * 1000:.1f} ms")

    # ---- Optimization --------------------------------------------------
    objectives = _parse_objectives(args.objectives, structures)
    robustness = RobustnessSpec()
    if args.robust > 0:
        scenarios = [
            ScenarioSpec(name=f"range_{i}", range_scale=1.0 + 0.03 * (i - args.robust // 2))
            for i in range(args.robust)
        ]
        robustness = RobustnessSpec(enabled=True, scenarios=scenarios,
                                    aggregation=args.aggregation)

    req = OptimizationRunRequest(
        geometry_id=geom.geometry_id,
        beam_model_id=bm.beam_model_id,
        dose_engine=EngineSpec(name=args.engine),
        objectives=objectives,
        solver=SolverConfig(method=args.solver, max_iterations=args.max_iters),
        robustness=robustness,
        checkpoint_interval=max(1, args.max_iters // 10),
    )

    _h("Step 3 — Optimization run (#1)")
    for o in objectives:
        _row("objective:", f"{o.type}({o.structure_name}, {o.dose_gy} Gy)")
    t0 = time.monotonic()
    result = OptimizationService().run(req)
    elapsed_1 = (time.monotonic() - t0) * 1000
    _row("optimization_id:", result.optimization_id)
    _row("solver success:", result.convergence.success)
    _row("iterations:", result.convergence.iterations)
    _row("final cost:", f"{result.convergence.final_cost:.4g}")
    _row("checkpoints:", len(result.checkpoints))
    if result.robust_stats:
        _row("worst-case cost:", f"{result.robust_stats.worst_case_metrics.get('worst_cost', 0):.4g}")
    _row("weights .npy:", result.weights_ref_uri)
    _row("dose .nii.gz:", result.dose_ref_uri)
    _row("elapsed:", f"{elapsed_1:.1f} ms")

    if args.engine == "analytic" and result.convergence.iterations == 0:
        print()
        print("  ⓘ 0 iterations: the analytic toy engine deposits a fixed")
        print("    depth-falloff kernel that is not target-aware and whose")
        print("    sparse Dij keeps only the highest-intensity voxels — so the")
        print("    structure may lie outside the kept set, giving a ~0 gradient.")
        print("    This exercises the full pipeline + caching; use")
        print("    `--engine mcsquare` for a clinically meaningful optimization.")

    _h("Step 3b — Optimization run (#2, cache hit)")
    t0 = time.monotonic()
    result2 = OptimizationService().run(req)
    elapsed_2 = max((time.monotonic() - t0) * 1000, 0.01)
    _row("optimization_id:", result2.optimization_id)
    _row("speedup:", f"{elapsed_1 / elapsed_2:.0f}×")
    if result2.optimization_id == result.optimization_id:
        print("  ✓ same optimization_id — cache hit confirmed")

    if args.show:
        _show_preview(result, geom)
    else:
        print("\n  (pass --show to render DVH + dose slice + convergence)")
    print()


def _show_preview(result, geom) -> None:
    try:
        import matplotlib
        for backend in ("MacOSX", "TkAgg", "Qt5Agg", "Agg"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
        import matplotlib.pyplot as plt
        import SimpleITK as sitk
    except ImportError as exc:
        print(f"\n  (--show needs matplotlib + SimpleITK: {exc})")
        return

    dose = sitk.GetArrayFromImage(sitk.ReadImage(result.dose_ref_uri))
    density = sitk.GetArrayFromImage(sitk.ReadImage(geom.density_grid_uri))
    masks = sitk.GetArrayFromImage(sitk.ReadImage(geom.structure_masks_uri))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1 — cumulative DVH per structure.
    for name, label in geom.structure_index.items():
        vals = dose[masks == label]
        if vals.size == 0:
            continue
        order = np.sort(vals)
        vol = 100.0 * (1.0 - np.arange(order.size) / max(order.size, 1))
        axes[0].plot(order, vol, label=name)
    axes[0].set_xlabel("Dose (Gy)")
    axes[0].set_ylabel("Volume (%)")
    axes[0].set_title("DVH")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Panel 2 — axial dose slice with highest deposition.
    per_slice = dose.sum(axis=(1, 2))
    z = int(np.argmax(per_slice)) if per_slice.max() > 0 else dose.shape[0] // 2
    nonzero = dose[dose > 0]
    vmax = float(np.percentile(nonzero, 99)) if nonzero.size else 1.0
    axes[1].imshow(density[z], cmap="gray")
    im = axes[1].imshow(dose[z], cmap="jet", alpha=0.55, vmin=0, vmax=max(vmax, 1e-6))
    axes[1].set_title(f"Axial dose z={z}")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], shrink=0.7, label="Dose (Gy)")

    # Panel 3 — convergence curve.
    ch = result.convergence.cost_history
    axes[2].plot(range(len(ch)), ch, marker=".")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Composite cost")
    axes[2].set_title("Convergence")
    axes[2].set_yscale("log" if min(ch) > 0 else "linear")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Optimization {result.optimization_id[:8]}…  "
                 f"final_cost={result.convergence.final_cost:.3g}  "
                 f"iters={result.convergence.iterations}", fontsize=11)

    out_path = Path(result.dose_ref_uri).parent / "preview.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n  Saved preview → {out_path}")
    try:
        plt.show()
    except Exception as exc:
        print(f"  (GUI window not available: {exc})")


if __name__ == "__main__":
    main()
