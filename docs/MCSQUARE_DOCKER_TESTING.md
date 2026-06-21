# Plan — Running & testing the MCsquare Monte Carlo engine via Docker

## Why this exists

MCsquare (the real proton Monte Carlo dose engine) **cannot run on the host Mac**:
the vendored binaries are Linux x86-64 only (`MCsquare_linux*`), and there is no
`MCsquare_mac`. It *can* run inside the project's Linux Docker stack. This plan
brings the stack up correctly and validates the engine end-to-end through all of
Services 3–6.

Status of the code under test: Services 3–6 are committed on
`feature/services-3-6` (544 tests green); the MCsquare engine drives all the way
to the binary invocation on macOS and only fails for lack of a mac binary. The
default CT-calibration/BDL fallback fix (`2b84cf3`) is required for the run to
work on Linux too.

## Two prerequisite fixes to the compose stack

These are needed before the Monte Carlo path works in Docker. (Pre-existing
latent issues, not introduced by Services 3–6.)

### 1. Worker must consume the per-service queues

`docker-compose.yml` runs the worker with no `-Q`, so it only listens to the
default `celery` queue — but tasks route to `dose` / `optimize` / `evaluate` /
`maintenance` (and production mode disables eager execution). Update the worker
command:

```yaml
  worker:
    command: >
      celery -A radiarch.tasks.celery_app worker --loglevel=info
      --concurrency=2 -Q celery,dose,optimize,evaluate,maintenance
```

(Or run dedicated workers per queue for isolation — optimization/BAO are heavy.)

### 2. Worker (and api) must run as linux/amd64 on Apple Silicon

The MCsquare binaries are x86-64 ELF. On an arm64 host the container must be
amd64 so they execute under Rosetta/QEMU:

```yaml
  worker:
    platform: linux/amd64
    build: { context: ., dockerfile: Dockerfile }
```

Confirm `MCsquare_linux*` are executable in the image (they're vendored under
`src/opentps/core/processing/doseCalculation/protons/MCsquare/`); add
`RUN chmod +x .../MCsquare/MCsquare_linux*` to the Dockerfile if exec perms are
lost in the build.

## Phase A — Bring up the stack

```bash
# Start Docker Desktop first (host): open -a Docker  (wait until `docker info` works)
cd radiarch
docker compose build worker api          # amd64 build — slow first time under emulation
docker compose up -d redis postgres orthanc
docker compose up -d api worker
docker compose ps                         # all healthy/running
docker compose logs -f worker | grep -i "celery@.*ready\|queues"   # confirm queues
```

## Phase B — Run DB migrations (new job tables)

The Optimization/BAO/Evaluation job tables ship as Alembic migrations
(`optimization_jobs`, `bao_jobs`, `evaluation_jobs`). Apply them:

```bash
docker compose exec api alembic -c src/alembic.ini upgrade head   # adjust path to alembic.ini
docker compose exec postgres psql -U radiarch -d radiarch -c "\dt" | grep -E "optimization_jobs|bao_jobs|evaluation_jobs"
```

## Phase C — Binary smoke test (does MCsquare actually execute?)

Fastest possible real run — one nominal dose, low primaries, bundled fantom:

```bash
docker compose exec worker python demo/show_dose.py --engine mcsquare
# Pass low primaries for speed if needed via the API params: {"nb_primaries": 1e4}
```

**Pass criterion:** completes with a non-zero `max dose (Gy)` and a dose NIfTI on
disk — i.e. `MCsquare_linux` ran (no `FileNotFoundError`, no "exec format error").
This is the gate; everything below assumes it passes.

## Phase D — V1 analytic-vs-MCsquare gamma (physics sanity)

```bash
docker compose exec worker python demo/compare_engines.py
```

**Pass criterion (per TASKS.md V1):** gamma pass rate ≥ 95% (3%/3mm) on the
bundled SimpleFantom. If < 95%, stop and debug the engine before trusting any
optimization built on top.

## Phase E — V4 Dij vs nominal-dose consistency (gate for optimization)

```bash
docker compose exec worker pytest tests/test_mcsquare_dij_consistency.py -v
```

**Pass criterion:** the MCsquare leg passes within 5% p95 lit-voxel agreement at
`nb_primaries=1e5` (the `TestMCsquareDijConsistency` class no longer auto-skips
once the binary runs). This proves `build_influence` (Dij) agrees with
`compute_dose`, which the optimizer relies on.

## Phase F — Service 4 (Optimization) on MCsquare

End-to-end inverse plan on real physics (this is the headline result):

```bash
docker compose exec worker python demo/show_optimization.py --engine mcsquare \
  --objectives ptv_dmin=60,oar_dmax=20 --solver L-BFGS-B --max-iters 100 --show
```

Or via the API (exercises the async Celery path + job polling):

```bash
KEY=$(grep RADIARCH_API_KEY .env | cut -d= -f2)
# POST /api/v1/optimize/run with engine "mcsquare"; poll /optimize/jobs/{id};
# GET /optimize/{id} → weights_ref_uri (.npy) + dose_ref_uri (.nii.gz),
# convergence.cost_history strictly decreasing.
```

**Pass criterion (TASKS.md DoD):** PTV target dose within ±5%, OAR < 30% of
target, monotonically decreasing cost, real `.npy` weights + `.nii.gz` dose.

## Phase G — Services 5 & 6 on MCsquare

```bash
# BAO: select beam angles using real per-angle MCsquare optimizations (slow —
# keep candidate sweep coarse + scoring_iterations small).
docker compose exec worker python demo/show_bao.py --engine mcsquare \
  --n-beams 2 --angle-step 90 --scoring-iters 10

# Evaluation: DVH + indices + gamma on the MCsquare dose from Phase F.
docker compose exec worker python demo/show_evaluation.py --engine mcsquare \
  --prescription 60 --gamma
```

**Pass criteria:** BAO returns a selected beam set with non-flat per-angle
scores (real physics → angles differ); Evaluation produces a clinically sensible
DVH (PTV coverage high, OAR sparing), Paddick CI in (0,1], and a gamma pass rate
near 100% when comparing a dose to itself.

## Phase H — Full async smoke through the API

```bash
API_BASE_URL=http://localhost:8000/api/v1 X_API_KEY=$KEY ./scripts/smoke_test.sh
# Extend smoke_test.sh with /optimize/run, /bao/run, /evaluate/run if not present.
docker compose exec api tail -50 /data/artifacts/audit.jsonl   # audit events per service
```

## Performance notes

- **x86-64 emulation on Apple Silicon is slow** (≈3–10× native). Keep
  `nb_primaries` low (1e4–1e5) for smoke/dev; raise to 1e6 only for
  production-grade gamma validation, ideally on a native Linux box.
- Optimization Dij build dominates wall-clock; BAO multiplies that by
  `n_candidates × n_beams`. Start with 2 beams / 4 candidates.
- Give the worker container generous memory (MCsquare ~2–4 GB/process on a
  512³ grid; the bundled fantom is small).

## Definition of done for "MCsquare verified"

- [ ] Phase C: binary executes, produces a dose
- [ ] Phase D: V1 gamma ≥ 95%
- [ ] Phase E: V4 Dij consistency passes
- [ ] Phase F: optimization produces a sensible, converging plan on MCsquare
- [ ] Phase G: BAO + Evaluation run on MCsquare output
- [ ] Phase H: async API flow + audit log green
