# Radiarch — Handoff Task List

**Status as of handoff:** Features 1–3 complete and production-hardened. Features 4–6 not started. 25 active tasks tracked here: 4 validation (V1–V4) + 21 implementation (O1–O21) for the Optimization Service.

> **Read this top-to-bottom before starting.** Sections "Conventions to follow" and "Module map" tell you the patterns already established — match them. The implementation tasks assume you'll mirror Feature 3's structure rather than invent new ones.

---

## 1. 60-second project context

**What Radiarch is.** A backend treatment-planning system (TPS) for proton + photon radiation oncology. FastAPI + Celery + Postgres + Redis + vendored OpenTPS/MCsquare. Designed as 6 composable microservices behind a single API:

| # | Service | Status |
|---|---|---|
| 1 | Geometry (DICOM → CT bundle + masks) | ✅ done |
| 2 | Beam Model (proton spots / photon beamlets) | ✅ done |
| 3 | Dose (engine-agnostic compute + Dij + scenarios) | ✅ done, production-hardened |
| 4 | **Optimization** (inverse plan, this handoff) | ⏳ pending |
| 5 | BAO (beam angle optimization) | ⏳ pending |
| 6 | Evaluation (DVH, indices, gamma, reports) | ⏳ pending |

**Where the action happens:**
- `src/radiarch/services/` — one file per service, plus `dose_engines/` plugin subdir
- `src/radiarch/models/` — Pydantic I/O contracts per service
- `src/radiarch/api/routes/` — FastAPI routers per service
- `src/radiarch/tasks/` — Celery tasks per service
- `tests/` — pytest, mirrors `services/` and `api/routes/` layout
- `demo/` — runnable end-to-end scripts (one per service)
- `docs/` — `PRODUCTION.md`, `adr/`
- `src/opentps/` — vendored OpenTPS Core, do not edit

**Existing test count:** 175 tests across 17 files, all green pre-handoff.

---

## 2. How to run

```bash
cd /Users/vidyuthashok/Desktop/radiarch-VAK/radiarch
source src/.venv/bin/activate

# Install / reinstall (regenerates editable-install finder cleanly)
./scripts/install-dev.sh

# Layout regression — fastest, always run first when in doubt
pytest tests/test_package_layout.py -v

# Feature 3 unit + e2e + production tests (~30s, no external deps)
pytest tests/test_dose_d1.py tests/test_dose_engines.py \
       tests/test_dose_service.py tests/test_dose_production.py \
       tests/test_dose_e2e.py -v

# V4 Dij-consistency regression (analytic part always, MCsquare part if available)
pytest tests/test_mcsquare_dij_consistency.py -v

# Full suite minus Redis-needing integration tests
pytest tests/ -v \
  --ignore=tests/test_dose_integration.py \
  --ignore=tests/test_api_e2e.py \
  --ignore=tests/test_opentps_integration.py \
  --ignore=tests/opentps
```

**`scripts/install-dev.sh` is mandatory** after adding any new package directory under `src/opentps/` or `src/radiarch/` — modern setuptools' strict editable install snapshots the package tree at install time and stale snapshots break vendored-package imports. The script purges build artifacts and reinstalls cleanly. See `tests/conftest.py` for the belt-and-suspenders sys.path guard.

---

## 3. Validation tasks (Feature 3) — V1–V4

These must run on a machine with MCsquare installed and a real clinical CT available. They gate Feature 4 — if MCsquare physics is wrong, the optimizer built on top will produce wrong plans.

### V1 — Analytic vs MCsquare on bundled SimpleFantom

```bash
python demo/compare_engines.py
```

**Expect:** gamma pass rate ≥95% (3%/3mm). Exit code 0 = pass. Document the gamma summary in the task notes. If <95%, do **not** proceed with V2/V3/V4 — debug the engine first.

### V2 — Real clinical CT

```bash
# Download a proton-relevant TCIA study per demo/README_DICOM.md
python demo/compare_engines.py --upload /path/to/study.zip
```

**Expect:** similar gamma agreement as V1 on heterogeneous anatomy. Engines diverging here while agreeing on V1 indicates a calibration / CT-conversion bug (likely in `radiarch.services.geometry` HU→density or in the MCsquare calibration table loader).

### V3 — Production endpoint smoke under realistic config

```bash
docker compose up -d
KEY=$(openssl rand -hex 32)
# Edit .env: RADIARCH_API_KEY=$KEY, RADIARCH_AUDIT_LOG_PATH=/data/artifacts/audit.jsonl
docker compose restart api worker
API_BASE_URL=http://localhost:8000/api/v1 X_API_KEY=$KEY ./scripts/smoke_test.sh

# Verify
curl -H "X-API-Key: $KEY" http://localhost:8000/api/v1/dose/engines/mcsquare | jq .
docker compose exec api tail -50 /data/artifacts/audit.jsonl
```

**Expect:** all 10 dose endpoints respond 200/202/401 as documented; audit JSONL accumulates one line per event; cleanup task runs within 1h and emits a `cleanup.swept` event.

### V4 — MCsquare Dij vs nominal dose ⚠️ critical gate for Feature 4

```bash
pytest tests/test_mcsquare_dij_consistency.py -v
```

`TestAnalyticDijConsistency` runs anywhere (always-on guard). `TestMCsquareDijConsistency` auto-skips without OpenTPS. The MCsquare leg must pass within 5% p95 lit-voxel agreement at `nb_primaries=1e5`. Tighter tolerance (1%) requires `nb_primaries=1e6` — bump and re-run if you want production-grade confidence.

**If V4 fails:** the engine's `compute_dose` and `build_influence` disagree. The optimizer (Feature 4) cannot be trusted on top of a broken Dij. Investigate before any O-task.

---

## 4. Feature 4 — Optimization Service (O1–O21)

**Goal:** Solve for optimal fluence weights `w*` given dose objectives and a dose engine. Inverse planning. Robust optimization supported via scenario aggregation.

**Reference spec:** the handoff spec the user provided defines the API shape (OptimizationRunRequest, OptimizationResult, etc.). Follow it.

**Dependency chain:**
- O1 (models) blocks O2, O14, O16
- O3 + O4 + O5 (objectives) block O9
- O6 + O7 + O8 (solvers) block O9
- O9 (service) blocks O10, O11, O14, O15
- O12 + O13 (robustness) extend O9
- O14 + O15 (API + Celery) block O16 + O17
- O18, O19, O20, O21 are integration / polish — do last

**Parallelizable batches:**
1. **First batch (independent foundations):** O1, O3, O6 — start these in parallel
2. **Second batch:** O2, O4, O5, O7, O8 — after O1/O3/O6 are done
3. **Core service:** O9 → O10 → O11
4. **Robustness:** O12 → O13
5. **API + Celery:** O14 → O15
6. **Tests:** O16 (API) + O17 (service)
7. **Integration:** O18, O19, O20, O21

### O1 — Pydantic models for Optimization Service

Create `src/radiarch/models/optimization.py` with:

- `ObjectiveSpec(type: Literal["DMin"|"DMax"|"DUniform"|"DVHMin"|"DVHMax"|"EUD"], structure_name: str, dose_gy: float, weight: float, volume_fraction: Optional[float])` — `volume_fraction` required for DVH types only (Pydantic validator)
- `ConstraintSpec(structure_name, type, op: Literal[">=", "<="], value_gy, weight)`
- `RegularizationConfig(fluence_smoothness: Optional[float], total_variation: Optional[float])`
- `SolverConfig(method: Literal["L-BFGS-B"|"Adam"|"ProjectedGradient"], max_iterations: int, convergence_tol: Optional[float], regularization: RegularizationConfig)`
- `RobustnessSpec(enabled: bool, scenarios: List[ScenarioSpec], aggregation: Literal["WORST_CASE"|"EXPECTED"|"CVAR"])` — reuse `ScenarioSpec` from `models/dose.py`
- `OptimizationRunRequest(plan_id, geometry_id, beam_model_id, dose_engine: EngineSpec, objectives: List[ObjectiveSpec], constraints: List[ConstraintSpec], solver: SolverConfig, init_weights_uri: Optional[str], robustness: RobustnessSpec, checkpoint_interval: Optional[int])`
- `ConvergenceInfo(success, iterations, final_cost, cost_history, constraint_violations)`
- `RobustStats(scenario_doses, worst_case_metrics)`
- `CheckpointInfo(iteration, weights_uri, cost)`
- `OptimizationResult(optimization_id, cache_key, weights_ref_uri, dose_ref_uri, convergence, robust_stats: Optional, compute_time_s, checkpoints, geometry_id, beam_model_id, engine_name, engine_version)`
- `OptimizationStage(Enum)`: `queued, loading, building_objective, optimizing, computing_final_dose, persisting, done`
- `OptimizationJobStatus(id, cache_key, state: JobState, progress: float, stage: OptimizationStage, message, optimization_id: Optional)`

Add `compute_cache_key()` to `OptimizationRunRequest` — SHA256 of: `{geometry_id, beam_model_id, engine.name, engine.version, engine.params, objectives_hash, constraints_hash, solver_config_hash, init_weights_hash, robustness_hash}`. Mirror `DoseComputeRequest.compute_cache_key` exactly for consistency.

**Tests:** `tests/test_optimization_d1.py` mirroring `tests/test_dose_d1.py` — model round-trip, cache-key stability, cache-key sensitivity to each field, validator rejections.

### O2 — OptimizationJob DB model + migration + store methods

- Add `OptimizationJob` table to `src/radiarch/core/db_models.py` (mirror `DoseJob` schema: `id`, `cache_key`, `state`, `progress`, `stage`, `message`, `optimization_id` nullable, `created_at`, `updated_at`)
- Alembic migration in `src/migrations/versions/`
- Store methods in `src/radiarch/core/store.py`: `create_optimization_job`, `get_optimization_job`, `update_optimization_job(id, **fields)`, `list_optimization_jobs`
- Tests in `tests/services/test_persistence.py` (extend existing file)

### O3 — Point-dose objectives (DMin, DMax, DUniform)

Create `src/radiarch/services/objectives.py`. Each objective is a callable returning `(loss: float, grad_wrt_dose: np.ndarray)` so the optimizer can sum gradients. Signature:

```python
class Objective(Protocol):
    name: str
    def __call__(self, dose: np.ndarray, mask: np.ndarray) -> Tuple[float, np.ndarray]: ...
```

- **DMin**: `loss = w * sum(max(0, d_target - d_i)^2 for i in mask)`; `grad[i] = -2 * w * max(0, d_target - d_i) * mask[i]`
- **DMax**: symmetric (penalize above target)
- **DUniform**: `loss = w * sum((d_i - d_target)^2 for i in mask)`; `grad[i] = 2 * w * (d_i - d_target) * mask[i]`

All differentiable, all per-voxel weighted by mask. Tests against hand-computed values on 4³ grid with simple masks.

### O4 — DVH and EUD objectives

Extend `objectives.py`:

- **DVHMin(structure, dose_gy, volume_fraction, weight)**: volume_fraction % of structure must receive ≥ dose_gy. Use sigmoid surrogate for differentiability: `H(d - d_target) ≈ sigmoid(k * (d - d_target))` with `k` large enough (~10/dose_gy) — penalize if `(1 - mean(H)) > volume_fraction`.
- **DVHMax**: symmetric.
- **EUD(structure, dose_gy, a, weight)**: generalized equivalent uniform dose per Niemierko. `EUD = (mean(d^a)) ^ (1/a)`. Penalize `|EUD - dose_gy|^2`. Tests: gEUD→mean when a=1, →min when a→-∞, →max when a→+∞.

### O5 — Constraints + regularization

- **Penalty-based hard constraints**: `(value - limit)^2` above limit (for `<=`) or below (for `>=`), multiplied by `weight`. Becomes a soft penalty in the composite objective.
- **fluence_smoothness regularizer**: `sum((w[i] - w[neighbor])^2)` where neighbors are defined by the beam model's spot grid (need to walk `beam_model.fluence_elements.per_beam[*].per_layer[*]` to find spatial neighbors). For v0.1, treat neighbors as adjacent indices within the same layer.
- **total_variation regularizer**: `sum(|w[i+1] - w[i]|)` — simpler 1D approximation in fluence-element order.

### O6 — Solver protocol + L-BFGS-B

Create `src/radiarch/services/optimization_solvers.py`. Define:

```python
class SolverPlugin(Protocol):
    name: str
    def run(
        self,
        cost_and_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]],
        w0: np.ndarray,
        max_iter: int,
        convergence_tol: float,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
    ) -> Tuple[np.ndarray, ConvergenceInfo]: ...
```

Implement `LBFGSBSolver` wrapping `scipy.optimize.minimize(method="L-BFGS-B", bounds=[(0, None)] * n, jac=True)`. Use the callback to record cost history and trigger checkpoints. Honor `convergence_tol` via `options={"ftol": tol}`.

### O7 — Adam solver

`AdamSolver` with configurable `learning_rate, beta1=0.9, beta2=0.999, eps=1e-8`. Clip `w = max(w, 0)` after each step. Faster than L-BFGS-B for very large weight vectors (>100k). Tests against L-BFGS-B on a convex toy problem — both should converge to the same answer ±1%.

### O8 — Projected gradient solver

Simpler baseline. Backtracking line search + non-negativity projection. Useful as a debugging engine when L-BFGS-B / Adam misbehave.

### O9 — OptimizationService class

Create `src/radiarch/services/optimization.py`. Mirror `DoseService` structure (`compute_dose` analog). Pseudocode:

```python
class OptimizationService:
    def __init__(self, base_dir: Optional[Path] = None): ...

    def run(
        self,
        request: OptimizationRunRequest,
        progress_callback: Optional[OptimizationProgressCallback] = None,
    ) -> OptimizationResult:
        # 1. Cache lookup by request.compute_cache_key()
        # 2. Load geometry + beam_model bundles (reuse DoseService loaders)
        # 3. Build / load Dij via DoseService.build_influence
        # 4. Build composite objective from request.objectives + constraints + regularizers
        # 5. Resolve solver from request.solver.method
        # 6. Initial weights: init_weights_uri or uniform-ones
        # 7. Solver.run(cost_and_grad, w0, callback=progress_callback + checkpoint_writer)
        # 8. Compute final dose via engine.apply_influence(Dij, w_final)
        # 9. Persist via OptimizationStore; emit audit events
        # 10. Return OptimizationResult
```

Hook the engine's `apply_influence` for the matvec — this is the *only* engine interaction during iteration. Engine-agnostic by construction.

### O10 — Gradient via Dij.T @ dL/dDose

The cost function is `L(w) = sum_i obj_i(D(w))` where `D(w) = Dij @ w`. The gradient is `dL/dw = Dij.T @ sum_i dobj_i/dD`. Implement this once in the service, not per objective. Verify with numerical gradient (finite differences) on a 16³ grid — relative error must be <1e-3.

### O11 — Checkpointing + warm start

- Save weights as `{artifact_dir}/optimization/{opt_id}/checkpoints/iter_{N}.npy` plus a `meta.json` entry every `checkpoint_interval` iterations
- Atomic writes (tempdir + `os.replace`)
- Support `init_weights_uri` parsing (`file://` only for v0.1) — validate length matches beam model's `fluence_elements.total_count`
- Test: round-trip save/load, reject mismatched length

### O12 — Scenario aggregation

Robust wrapper that takes `RobustnessSpec` and turns one cost evaluation into many:

- **WORST_CASE**: `L = max_s L_s`
- **EXPECTED**: `L = mean_s L_s`
- **CVaR (α=0.1 by default)**: `L = mean over top-10% worst L_s`

Each scenario gets its own perturbed Dij if `scenario.density_scale` or `range_scale` is set (these change the engine's Dij). Setup shifts (`scenario.setup_shift_mm`) can reuse the nominal Dij if the engine supports it — for v0.1, recompute per-scenario for correctness, optimize later.

### O13 — Robust gradient aggregation

Matches the cost aggregation:

- **WORST_CASE** → subgradient from argmax scenario only
- **EXPECTED** → mean of per-scenario gradients
- **CVaR** → mean over top-α scenarios only

Cache Dij per scenario class. Setup-shift-only scenarios *can* share the nominal Dij (just shift the dose grid). For v0.1, accept the recomputation cost.

### O14 — FastAPI routes with auth + audit

`src/radiarch/api/routes/optimization.py`. Mirror `dose.py` exactly:

```python
router = APIRouter(prefix="/optimize", tags=["optimization"],
                   dependencies=[Depends(api_key_auth)])
```

Endpoints:
- `POST /api/v1/optimize/run` → 200 cache hit / 202 async / 422 / 401
- `GET /api/v1/optimize/jobs/{job_id}` → OptimizationJobStatus
- `GET /api/v1/optimize/{opt_id}` → OptimizationResult
- `GET /api/v1/optimize/{opt_id}/weights` → stream weights .npy
- `GET /api/v1/optimize/{opt_id}/checkpoints` → list checkpoints
- `DELETE /api/v1/optimize/{opt_id}` → 204

Emit audit events for every state: `optimize.run` with `state=cache_hit | dispatched | started | succeeded | failed`. Use `make_event` / `emit` from `src/radiarch/services/audit.py` — same pattern as `routes/dose.py`.

Register the router in `src/radiarch/app.py`.

### O15 — Async Celery task

`src/radiarch/tasks/optimization_tasks.py`. Mirror `dose_tasks.py` pattern:

```python
@celery_app.task(name="radiarch.optimize.run", ...)
def run_optimization_job(job_id: str, request_payload: dict):
    # update_optimization_job(state=running, ...)
    # request = OptimizationRunRequest.model_validate(request_payload)
    # service = OptimizationService()
    # result = service.run(request, progress_callback=on_progress)
    # update_optimization_job(state=succeeded, optimization_id=result.optimization_id)
    # emit audit events on success / failure / SoftTimeLimitExceeded
```

Add to `celery_app.include` in `src/radiarch/tasks/celery_app.py`. Add per-task `task_annotations`:

```python
"radiarch.optimize.run": {
    "soft_time_limit": settings.optimization_soft_time_limit_s,  # add to config.py
    "time_limit": settings.optimization_hard_time_limit_s,
    "rate_limit": settings.optimize_rate_limit,
},
```

Route to dedicated `"optimize"` queue in `task_routes`.

### O16 — API tests

`tests/test_api_optimization.py`. Mirror `tests/test_api_dose.py` test inventory:

- Auth required on every route (401 without key)
- Cache hit returns 200 inline
- Cache miss dispatches 202 with `job_id`
- Job polling returns `OptimizationJobStatus`
- Result GET returns `OptimizationResult`
- 404 on unknown id
- 422 on bad ObjectiveSpec (e.g., DVHMin without volume_fraction)
- 422 on unknown engine
- 422 on init_weights_uri with wrong length
- Audit log captures the expected event sequence (use file-sink fixture from `test_dose_production.py`)

### O17 — OptimizationService unit + integration tests

`tests/test_optimization_service.py`:

- **Convergence**: single DMin objective on a 16³ water phantom, all-zero initial weights, must reduce cost monotonically and converge in <100 iterations
- **Composite**: DMin (target=60Gy in PTV) + DMax (avoid=20Gy in OAR) — verify final dose hits both
- **Robust aggregation**: WORST_CASE picks the worst scenario's cost at convergence (verify via per-scenario manual evaluation)
- **Checkpoint round-trip**: save → load → resume produces same trajectory
- **Warm start**: `init_weights_uri` of a converged run produces near-zero gradient on first iteration
- **Gradient correctness**: numerical-vs-analytic on 16³ grid, rel_err < 1e-3
- **Cache hit**: same request twice returns same optimization_id

Also write `tests/test_optimization_d1.py` for the Pydantic models.

### O18 — Wire into RadiarchPlanner

Update `src/radiarch/core/workflows/proton_optimized.py` to dispatch to `OptimizationService` instead of the inline `IntensityModulationOptimizer`. Keep `POST /api/v1/plans` working — it should produce the same outputs as before, just routed through the new service.

Update `plan.qa_summary` to include `final_cost`, `iterations`, `solver_method`.

### O19 — `demo/show_optimization.py`

Mirror `demo/show_dose.py` structure:

```python
python demo/show_optimization.py
python demo/show_optimization.py --upload /path/to/study.zip
python demo/show_optimization.py --objectives ptv_dmin=60,oar_dmax=20
python demo/show_optimization.py --robust 9 --aggregation WORST_CASE
python demo/show_optimization.py --solver Adam --max-iters 500
python demo/show_optimization.py --show   # render DVH + dose slice + convergence plot
```

Save `preview.png` (3-panel: DVH curve, axial dose, convergence). Reuse the bundled SimpleFantom for the default run.

### O20 — Production hardening

Add to `src/radiarch/config.py`:

```python
optimization_soft_time_limit_s: int = 14400  # 4h, optimization is heaviest
optimization_hard_time_limit_s: int = 15000
optimize_rate_limit: str = "2/m"
optimization_store_max_gb: float = 30.0
optimization_checkpoint_keep: int = 5  # keep latest N per opt_id
```

Wire into `tasks/celery_app.py` annotations. Extend `tasks/cleanup_tasks.py` to sweep `optimization/` under the same LRU+min-age policy as `doses/`. Add a sub-sweep that keeps only the latest N checkpoints per opt_id, evicting older ones.

### O21 — Docs

- Update `README.md` with `/api/v1/optimize` section + `show_optimization.py` usage
- Add **Optimization** section to `docs/PRODUCTION.md` (settings table, sizing rules — optimization workers want more CPU than dose workers because Dij @ w matvec dominates)
- Write `docs/adr/0002-optimization-solver-choice.md` documenting why L-BFGS-B is the default and when to pick Adam (>100k spots, very wide Dij, ill-conditioned Hessian)

---

## 5. Conventions to follow (mirror Feature 3)

Read these existing files before writing new code — they're the templates:

- **Models:** `src/radiarch/models/dose.py` (Pydantic schemas + `compute_cache_key`)
- **Persistence:** `src/radiarch/services/dose_persistence.py` (atomic writes, `_index.json`)
- **Service orchestrator:** `src/radiarch/services/dose.py` (progress callbacks, stage enum, cache lookup → compute → persist)
- **Engine protocol:** `src/radiarch/services/dose_engines/protocol.py` + `analytic.py` (full implementation reference)
- **Routes:** `src/radiarch/api/routes/dose.py` (auth dep at router level, 200/202/422/401 contract, audit emission)
- **Celery task:** `src/radiarch/tasks/dose_tasks.py` (state mirroring, progress callback, SoftTimeLimitExceeded handler, audit on success/failure)
- **Audit:** `src/radiarch/services/audit.py` (`make_event`, `emit`, `audit_span`)
- **Auth:** `src/radiarch/api/security.py` (constant-time compare, 401 + WWW-Authenticate header)
- **Tests:** `tests/test_dose_d1.py` (model unit tests), `tests/test_dose_service.py` (service tests), `tests/test_api_dose.py` (API tests), `tests/test_dose_e2e.py` (integration), `tests/test_dose_production.py` (auth/audit/cleanup)
- **Demo:** `demo/show_dose.py` + `demo/compare_engines.py`

**Style guardrails Claude Code should follow (matches what's in the codebase):**

- Type hints everywhere. Pydantic v2 syntax.
- Docstrings on every public class/function. Explain *why*, not what — the what is the code.
- Dataclasses for engine I/O (`NominalDose`, `InfluenceData`), Pydantic for API I/O.
- No bare `Exception` catches in service code unless re-raised as a typed engine exception (`EngineParamError`, `EngineRuntimeError`, `EngineUnavailableError`).
- Settings via `RADIARCH_*` env vars, configured in `src/radiarch/config.py`.
- Audit log emission is *never* a critical path — wrap in try/except, log to loguru if the sink fails.
- Atomic writes for any persistence: tempdir → `os.replace`, update index *last*.
- Cache keys are SHA256 of a normalized JSON representation. Include engine version, exclude transient fields like timestamps.
- Loguru for logging (not stdlib `logging`).

---

## 6. Module map (reference)

```
src/radiarch/
├── app.py                            # FastAPI factory (register new optimization router here)
├── config.py                         # Settings (add optimization_* settings here)
├── adapters/                         # Orthanc, DICOMweb — don't touch for Feature 4
├── api/
│   ├── security.py                   # api_key_auth dependency — reuse as-is
│   └── routes/
│       ├── dose.py                   # Pattern to mirror for optimization.py
│       └── optimization.py           # ← O14
├── core/
│   ├── db_models.py                  # Add OptimizationJob here (O2)
│   ├── store.py                      # Add optimization_job CRUD here (O2)
│   └── workflows/proton_optimized.py # ← O18, dispatch to OptimizationService
├── models/
│   ├── dose.py                       # ScenarioSpec lives here, reuse in optimization.py
│   └── optimization.py               # ← O1
├── services/
│   ├── dose.py                       # Pattern reference for OptimizationService
│   ├── dose_engines/                 # Engine protocol + analytic + mcsquare + ccc
│   ├── objectives.py                 # ← O3, O4, O5
│   ├── optimization.py               # ← O9, O10, O11
│   ├── optimization_solvers.py       # ← O6, O7, O8
│   ├── audit.py                      # Reuse make_event/emit/audit_span
│   ├── dose_persistence.py           # Pattern reference for optimization_persistence.py
│   └── scenarios.py                  # Reuse expand_scenarios for robustness
└── tasks/
    ├── celery_app.py                 # Add optimization_tasks to include + annotations (O15, O20)
    ├── cleanup_tasks.py              # Extend with optimization sweep (O20)
    ├── dose_tasks.py                 # Pattern to mirror for optimization_tasks.py
    └── optimization_tasks.py         # ← O15

src/migrations/versions/              # Alembic migrations — add OptimizationJob migration (O2)
demo/show_optimization.py             # ← O19
docs/PRODUCTION.md                    # ← O21 update
docs/adr/0002-optimization-solver-choice.md  # ← O21 new
tests/test_optimization_d1.py         # ← O1 tests
tests/test_optimization_service.py    # ← O9, O17
tests/test_api_optimization.py        # ← O14, O16
```

---

## 7. Recent context Claude Code should know

- **MCsquare engine bug just fixed**: in OpenTPS v3, `MCsquareDoseCalculator.computeDose(ct, plan)` and `computeBeamlets(ct, plan)` take `ct` as a **positional** argument (the method sets `self.ct = ct` internally). Setting `mc_calc.ct = ct_image` as an attribute then calling `computeDose(plan)` raises `TypeError: missing 1 required positional argument: 'ct'`. The fix is committed; just be aware if you touch `src/radiarch/services/dose_engines/mcsquare.py`.

- **Editable install hygiene**: any time you add a new `__init__.py` under `src/`, run `./scripts/install-dev.sh` or strict-mode pip will keep using the stale finder. The `tests/conftest.py` prepends `src/` to `sys.path` as a belt-and-suspenders guard so tests still work even with a stale install.

- **CCC photon engine is deliberately a stub** (`src/radiarch/services/dose_engines/ccc.py`). See `docs/adr/0001-ccc-photon-engine.md` — decision is to defer photon to v2. Feature 4 should focus on protons; photon objectives still get implemented in O3–O5 generically because they work on dose volumes regardless of how the dose was computed.

- **OpenTPS plan structure**: weights live in `plan.beams[*].layers[*].spotMUs` (or a flat `plan.spotMUs` for simple plans). The `_apply_weights_to_plan` helper in `src/radiarch/services/dose_engines/mcsquare.py` handles both; reuse this pattern if you ever need to round-trip weights into an OpenTPS plan from Feature 4.

- **Robust optimization gotcha**: `MCsquareDoseCalculator.computeBeamlets` internally sets `plan.spotMUs = np.ones(...)` before computing — so you build one beamlet per spot at unit MU, then apply weights later via `Dij @ w`. This is the right model for the optimizer. Don't try to pre-weight the plan before `build_influence`.

---

## 8. Definition of done

**Feature 4 ships when:**

1. All 21 O-tasks completed and tracked in pytest
2. Full test suite green (`pytest tests/ -v` minus the integration tests that need Redis)
3. `demo/show_optimization.py --robust 9` runs end-to-end on bundled SimpleFantom and produces a sensible DVH (PTV target dose ±5%, OAR dose < 30% of target)
4. `POST /api/v1/optimize/run` returns 202 with a job_id; polling returns succeeded; result has `weights_ref_uri` pointing at a real .npy file; `dose_ref_uri` pointing at a real .nii.gz; convergence shows monotonically decreasing cost
5. Smoke test (`./scripts/smoke_test.sh` extended for optimization) green against docker-compose stack
6. `docs/PRODUCTION.md` Optimization section + `0002-optimization-solver-choice.md` written
7. README updated

**Then start Feature 5 (BAO) or Feature 6 (Evaluation) per business priority.** Feature 6 (Evaluation) is the natural next step if you want a usable end-to-end TPS — it consumes the dose volume that Feature 4 produces and turns it into a clinician-readable report.

---

*Generated at handoff. If anything here is unclear or stale, check git log on the files referenced — the patterns are recent (last 2 weeks of commits) and the commit messages explain intent.*
