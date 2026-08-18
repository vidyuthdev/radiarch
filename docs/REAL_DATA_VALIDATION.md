# Real-data validation — ingestion + MCsquare proton dose

Status of running Radiarch on **real 3D data** (NRRD + clinical DICOM), and the
first end-to-end run of the **real MCsquare Monte Carlo proton engine on a real
clinical CT**. Reproduced by `scripts/real_dose_validation.py` (dose) and
`scripts/nrrd_ingest_check.py` (NRRD ingest).

> **Integrity note.** Everything below validates *ingestion and dose
> orchestration on real anatomy*. It is **not** a clinical-accuracy claim. The
> public RT structure sets used here are OARs (no PTV/tumor), plans aim at a
> fallback pseudo-target with uniform (uncalibrated) weights, and MCsquare was
> run at a deliberately low, statistically-noisy primary count. "It ran" ≠ "it
> is correct."

## Data (all gitignored under `data/`)

| Source | Modality | Patients | Structures |
| --- | --- | --- | --- |
| Local NRRD (Slicer export) | CT `.nrrd` + `.seg.nrrd` | 1 pelvis (+4 CT-only) | urinary bladder, uterus |
| TCIA **LCTSC-Test-S1-101** | DICOM CT + RTSTRUCT | 1 thorax | SpinalCord, Lung_L, Lung_R, Heart, Esophagus (auto-seg, Plastimatch) |
| TCIA **BREAST-DIAGNOSIS** | DICOM CT (no RTSTRUCT) | 3 | — |

## 1. NRRD ingestion (committed: `adapters/nrrd_ingest.py`)

Pelvis CT + Slicer segmentation → geometry build:

- 512×512×144, HU sane, density finite/non-negative.
- Structures rasterized, physiologically plausible: **urinary bladder 146.6 mL**,
  **uterus 73.5 mL**.
- Deterministic (two independent builds byte-identical); fast-path masks exact.
- All 5 local NRRD volumes ingest, including an **oblique head CT** that is
  resampled to an axis-aligned grid on load.

## 2. DICOM ingestion (existing pipeline, first verified on real clinical CT)

**LCTSC-Test-S1-101** (CT + RTSTRUCT), stoichiometric HU→density:

- 512×512×130, spacing 0.977×0.977×3.0 mm, HU [−1000, 3017], finite.
- 5 structures, plausible volumes: Lung_R 2173.6 mL, Lung_L 1812.2 mL,
  Heart 756.3 mL, SpinalCord 72.7 mL, Esophagus 33.4 mL (R > L lung is correct).

**BREAST-DIAGNOSIS** — 3 CT-only patients, 512×512×207 CAP series each,
HU [−1024, 3071], finite. Confirms multi-patient CT ingestion breadth.

## 3. Real MCsquare Monte Carlo proton dose on real CT (Docker)

First run of the real proton engine on real patient data (prior MC runs used the
synthetic fantom). Ran `scripts/real_dose_validation.py --engine mcsquare` in the
Linux worker on LCTSC:

- MCsquare transported **10,080 primaries in ~12 s**; dose **finite,
  non-negative, 1.14M non-zero voxels**.
- The dose is a **coherent proton beam**: enters anterior, deposits into the
  chest, stops at depth near the SpinalCord fallback pseudo-target — visually
  consistent with proton physics.
- Engine health: MCsquare available, `MCsquare_linux` binary present/executable.

**Caveats (why this is a smoke test, not a dose):** ~81% statistical uncertainty
(1e4 primaries); `max ≈ 1680 Gy` is an uncalibrated low-statistics entrance hot
voxel; pseudo-target is an OAR (no tumor in LCTSC); uniform weights, no MU
normalization.

## Reproduce

Local plumbing check (analytic engine, macOS ok):

```bash
python scripts/real_dose_validation.py \
    --data-root data/dicom/<study-dir> --engine analytic
```

Real proton dose (Linux — the Docker worker):

```bash
docker compose build worker api && docker compose up -d
WORKER=$(docker compose ps -q worker)
# stage a DICOM study on the persistent volume:
docker cp <study-dir> "$WORKER":/data/artifacts/study
docker cp scripts "$WORKER":/app/scripts          # if not baked into the image
docker exec "$WORKER" python scripts/real_dose_validation.py \
    --data-root /data/artifacts/study --engine mcsquare --primaries 1e4
docker compose down                                # volumes persist
```

## Known gaps surfaced during validation

- **`demo/show_dose.py` hard-overrides `RADIARCH_OPENTPS_DATA_ROOT` /
  `RADIARCH_ARTIFACT_DIR` at import** to the bundled fantom paths, so it can't be
  pointed at real data with `-e`. `scripts/real_dose_validation.py` avoids this
  by setting config before import.
- **`BeamModelService._load_geometry` re-reads the patient from
  `opentps_data_root`** (for the target contour) instead of reusing the
  geometry's own source — so the upload/PACS flow needs the data root set too.
  Worth fixing so the beam model reuses the geometry that was already built.
