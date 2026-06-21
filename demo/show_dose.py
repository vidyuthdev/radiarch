"""Live demo of Service 3 — Dose Service.

Runs the Geometry Service → Beam Model Service → Dose Service pipeline
end-to-end and prints summary statistics. Defaults to the analytic
engine so the demo runs without OpenTPS / MCsquare; pass ``--engine
mcsquare`` to exercise the real proton engine (requires OpenTPS to be
importable).

Usage:
    # 1. Bundled SimpleFantom + analytic engine (fastest)
    python demo/show_dose.py
    python demo/show_dose.py --show

    # 2. With scenarios — exercises the robustness path
    python demo/show_dose.py --scenarios 5

    # 3. Real DICOM + MCsquare proton engine
    python demo/show_dose.py --upload /path/to/study.zip --engine mcsquare

    # 4. Build the influence matrix too
    python demo/show_dose.py --influence
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional


_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Mock Orthanc → service falls back to disk loading.
os.environ["RADIARCH_ORTHANC_USE_MOCK"] = "true"
os.environ["RADIARCH_DATABASE_URL"] = ""
os.environ["RADIARCH_BROKER_URL"] = "memory://"
os.environ["RADIARCH_RESULT_BACKEND"] = "cache+memory://"
os.environ["RADIARCH_DICOMWEB_URL"] = ""
os.environ["RADIARCH_ARTIFACT_DIR"] = str(_REPO_ROOT / "data" / "artifacts")
_TEST_DATA = (
    _REPO_ROOT / "tests" / "opentps" / "core" / "opentps-testData"
    / "SimpleFantomWithStruct"
)
os.environ["RADIARCH_OPENTPS_DATA_ROOT"] = str(_TEST_DATA)


import numpy as np  # noqa: E402

from radiarch.models.beam_model import (  # noqa: E402
    BeamModelBuildRequest,
    BeamSetSpec,
    BeamSpec,
    DeliveryParams,
    Modality,
)
from radiarch.models.dose import (  # noqa: E402
    DoseComputeRequest,
    EngineSpec,
    InfluenceBuildRequest,
    ScenarioSetSpec,
    WeightVector,
)
from radiarch.models.geometry import (  # noqa: E402
    GeometryBuildRequest,
    HUDensityModel,
    PatientRef,
)
from radiarch.services.beam_model import BeamModelService  # noqa: E402
from radiarch.services.dose import DoseService  # noqa: E402
from radiarch.services.geometry import GeometryService  # noqa: E402


BAR = "─" * 64


def _h(label: str) -> None:
    print(f"\n{BAR}\n  {label}\n{BAR}")


def _row(key: str, value) -> None:
    print(f"  {key:<28} {value}")


# ---------------------------------------------------------------------------
# Upload helper — same one used by show_geometry.py
# ---------------------------------------------------------------------------

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

    dcm_count = sum(1 for p in dest.rglob("*.dcm") if p.is_file())
    print(f"  extracted {dcm_count} .dcm files into {dest}")
    return upload_id


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upload", type=str, default=None,
                   help="ZIP of DICOM study to use instead of the bundled fixture.")
    p.add_argument("--engine", choices=["analytic", "mcsquare", "ccc"],
                   default="analytic")
    p.add_argument("--modality", choices=["proton", "photon"], default="proton",
                   help="Beam-model modality.")
    p.add_argument("--scenarios", type=int, default=0,
                   help="Number of robustness scenarios to evaluate (0 = nominal only).")
    p.add_argument("--influence", action="store_true",
                   help="Also build the influence matrix Dij.")
    p.add_argument("--show", action="store_true",
                   help="Render axial dose slice with structure overlay.")
    args = p.parse_args()

    _h("Radiarch Dose Service — Live Demo")
    _row("engine:", args.engine)
    _row("modality:", args.modality)
    _row("scenarios:", args.scenarios)
    _row("influence:", args.influence)

    # D6.6 — when the user explicitly asks for mcsquare, surface the
    # availability check up front rather than failing inside the
    # Celery task. Faster feedback loop, clearer error.
    if args.engine == "mcsquare":
        from radiarch.services.dose_engines import engine_health
        h = engine_health("mcsquare")
        if not h.get("available"):
            print()
            print("  ⚠ MCsquare engine is not available in this environment.")
            print(f"    diagnostics: {h.get('diagnostics', {})}")
            print("    Run `./scripts/install-dev.sh` or set RADIARCH_OPENTPS_VENV.")
            print("    Falling back to --engine analytic for this run.")
            args.engine = "analytic"

    # ---- Geometry ------------------------------------------------------
    if args.upload:
        upload_zip = Path(args.upload).expanduser().resolve()
        if not upload_zip.is_file():
            print(f"ERROR: upload not found at {upload_zip}", file=sys.stderr)
            sys.exit(1)
        upload_id = _ingest_upload_zip(upload_zip)
        patient_ref = PatientRef(upload_id=upload_id)
    else:
        if not _TEST_DATA.exists():
            print(f"ERROR: test data not found at {_TEST_DATA}", file=sys.stderr)
            sys.exit(1)
        patient_ref = PatientRef(dicom_study_uid="demo-study-001")

    geom_req = GeometryBuildRequest(
        patient_ref=patient_ref,
        grid_spec=None,
        hu_to_density_model=HUDensityModel.stoichiometric,
    )
    _h("Step 1 — Geometry build")
    t0 = time.monotonic()
    geom_result = GeometryService().build(geom_req)
    _row("geometry_id:", geom_result.geometry_id)
    _row("grid size:", geom_result.grid_spec.size)
    _row("structures:", list(geom_result.structure_index.keys()))
    _row("elapsed:", f"{(time.monotonic() - t0) * 1000:.1f} ms")

    # ---- Beam model ----------------------------------------------------
    modality = Modality.proton_pbs if args.modality == "proton" else Modality.photon_imrt
    bm_req = BeamModelBuildRequest(
        geometry_id=geom_result.geometry_id,
        modality=modality,
        beam_set=BeamSetSpec(
            isocenter_mm=(0.0, 0.0, 0.0),
            beams=[BeamSpec(beam_id="B1", gantry_deg=0.0)],
        ),
        delivery_params=DeliveryParams(),
    )
    _h("Step 2 — Beam-model build")
    t0 = time.monotonic()
    bm_result = BeamModelService().build(bm_req)
    _row("beam_model_id:", bm_result.beam_model_id)
    _row("total elements:", bm_result.fluence_elements.total_count)
    _row("elapsed:", f"{(time.monotonic() - t0) * 1000:.1f} ms")

    # ---- Dose compute --------------------------------------------------
    n = bm_result.fluence_elements.total_count
    # All-ones weights are the conventional "uniform delivery" baseline.
    weights = WeightVector(length=n, values=[1.0] * n)

    scenarios_spec = None
    if args.scenarios > 0:
        scenarios_spec = ScenarioSetSpec(
            setup_sigma_mm=3.0,
            range_sigma=0.035,
            count=args.scenarios,
        )

    dose_req = DoseComputeRequest(
        geometry_id=geom_result.geometry_id,
        beam_model_id=bm_result.beam_model_id,
        engine=EngineSpec(name=args.engine),
        weights=weights,
        scenarios=scenarios_spec,
    )

    _h("Step 3 — Dose compute (#1)")
    t0 = time.monotonic()
    dose_result = DoseService().compute_dose(dose_req)
    elapsed_1 = (time.monotonic() - t0) * 1000
    _row("dose_id:", dose_result.dose_id)
    _row("max dose (Gy):", f"{dose_result.statistics.max_gy:.3f}")
    _row("mean dose (Gy):", f"{dose_result.statistics.mean_gy:.3f}")
    _row("p95 dose (Gy):", f"{dose_result.statistics.p95_gy:.3f}")
    _row("nonzero voxels:", f"{dose_result.statistics.nonzero_voxel_count:,}")
    if dose_result.scenario_doses:
        _row("scenario doses:", len(dose_result.scenario_doses))
    _row("elapsed:", f"{elapsed_1:.1f} ms")

    _h("Step 3b — Dose compute (#2, cache hit)")
    t0 = time.monotonic()
    dose2 = DoseService().compute_dose(dose_req)
    elapsed_2 = max((time.monotonic() - t0) * 1000, 0.01)
    _row("dose_id:", dose2.dose_id)
    _row("elapsed:", f"{elapsed_2:.1f} ms")
    _row("speedup:", f"{elapsed_1 / elapsed_2:.0f}×")
    if dose2.dose_id == dose_result.dose_id:
        print("  ✓ same dose_id — cache hit confirmed")

    # ---- Optional influence --------------------------------------------
    if args.influence:
        # Heads-up on memory for clinical-scale grids.
        n_vox = int(np.prod(geom_result.grid_spec.size))
        n_elt = bm_result.fluence_elements.total_count
        rough_gb = (n_vox * n_elt * 4) / 1e9
        if rough_gb > 0.5:
            print(f"\n  Note: dense Dij would be ~{rough_gb:.1f} GB; "
                  f"analytic engine will cap active voxels.")
        inf_req = InfluenceBuildRequest(
            geometry_id=geom_result.geometry_id,
            beam_model_id=bm_result.beam_model_id,
            engine=EngineSpec(name=args.engine),
        )
        _h("Step 4 — Influence build")
        t0 = time.monotonic()
        inf_result = DoseService().build_influence(inf_req)
        _row("influence_id:", inf_result.influence_id)
        _row("rows × cols:", f"{inf_result.n_voxels:,} × {inf_result.n_elements}")
        _row("nnz:", f"{inf_result.nnz:,}")
        _row("density:", f"{inf_result.nnz / max(inf_result.n_voxels * inf_result.n_elements, 1):.4f}")
        _row("elapsed:", f"{(time.monotonic() - t0) * 1000:.1f} ms")

    if args.show:
        _show_axial_dose(dose_result, geom_result)
    else:
        print(f"\n  (pass --show to render axial dose slices)")
    print()


def _show_axial_dose(dose_result, geom_result) -> None:
    try:
        import matplotlib
        for backend in ("MacOSX", "TkAgg", "Qt5Agg"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
        import matplotlib.pyplot as plt
        import numpy as np
        import SimpleITK as sitk
    except ImportError as exc:
        print(f"\n  (--show needs matplotlib + SimpleITK: {exc})")
        return

    from pathlib import Path

    dose = sitk.GetArrayFromImage(sitk.ReadImage(dose_result.dose_grid_uri))
    density = sitk.GetArrayFromImage(sitk.ReadImage(geom_result.density_grid_uri))

    # Diagnostics — extremely useful when something looks wrong.
    nonzero = dose[dose > 0]
    print()
    print(f"  Dose array shape:  {dose.shape}")
    print(f"  Dose range:        min={dose.min():.3f}  max={dose.max():.3f}  "
          f"mean={dose.mean():.3f}")
    print(f"  Nonzero voxels:    {nonzero.size:,} / {dose.size:,} "
          f"({100 * nonzero.size / dose.size:.1f}%)")
    print(f"  Density range:     min={density.min():.3f}  max={density.max():.3f}")

    # Find the body's z-extent (slices with non-air density) so we don't
    # render slices that are entirely above/below the patient.
    body_mask_per_slice = (density > 0.1).sum(axis=(1, 2))
    body_slices = np.where(body_mask_per_slice > 50)[0]
    if body_slices.size == 0:
        print("  (no body voxels found — density looks empty)")
        return
    z_lo, z_hi = int(body_slices[0]), int(body_slices[-1])
    print(f"  Body z-extent:     slices {z_lo}..{z_hi}")

    # Pick the slice IN THE BODY with the most dose, not over the whole CT.
    per_slice_dose = dose[z_lo:z_hi + 1].sum(axis=(1, 2))
    if per_slice_dose.max() <= 0:
        # Nothing deposited inside the body — fall back to middle of body.
        z = (z_lo + z_hi) // 2
    else:
        z = z_lo + int(np.argmax(per_slice_dose))
    print(f"  Picked slice:      z={z}  "
          f"(slice dose max={dose[z].max():.3f}, "
          f"mean={dose[z].mean():.3f})")

    # Use a vmax that's clipped to the 99th percentile of nonzero dose
    # so a few hot voxels don't wash out the body.
    if nonzero.size > 0:
        vmax = float(np.percentile(nonzero, 99))
    else:
        vmax = 1.0
    vmax = max(vmax, dose[z].max() * 0.5, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Axial (z), coronal (y), sagittal (x) cuts — gives a fuller picture.
    y_mid = density.shape[1] // 2
    x_mid = density.shape[2] // 2

    axes[0].imshow(density[z], cmap="gray")
    im0 = axes[0].imshow(dose[z], cmap="jet", alpha=0.55, vmin=0, vmax=vmax)
    axes[0].set_title(f"Axial z={z}")

    axes[1].imshow(density[:, y_mid, :], cmap="gray", aspect="auto")
    axes[1].imshow(dose[:, y_mid, :], cmap="jet", alpha=0.55,
                   vmin=0, vmax=vmax, aspect="auto")
    axes[1].set_title(f"Coronal y={y_mid}")

    axes[2].imshow(density[:, :, x_mid], cmap="gray", aspect="auto")
    axes[2].imshow(dose[:, :, x_mid], cmap="jet", alpha=0.55,
                   vmin=0, vmax=vmax, aspect="auto")
    axes[2].set_title(f"Sagittal x={x_mid}")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im0, ax=axes, shrink=0.7, label="Dose (Gy)")
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Dose {dose_result.dose_id[:8]}…  engine={dose_result.engine_name}  "
        f"max={dose_result.statistics.max_gy:.1f} Gy  (display vmax={vmax:.1f})",
        fontsize=11,
    )

    out_path = Path(dose_result.dose_grid_uri).parent / "preview.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n  Saved preview → {out_path}")

    try:
        plt.show()
    except Exception as exc:
        print(f"  (GUI window not available: {exc})")


if __name__ == "__main__":
    main()
