# Production Deployment — Radiarch TPS

This document covers what you need to know to run Radiarch in
production. It complements the README (which targets local dev) and
focuses on the things that bite you only at scale: resource limits,
auth, audit logging, disk hygiene, and the deployment topology.

> Audience: ops / DevOps engineer setting up the first production
> tenant. Assumes Linux + Docker + a reverse proxy (nginx / Caddy /
> Traefik) in front. Bare-metal Python installs are supported but not
> recommended.

## Reference topology

```
                        ┌─────────────────┐
                        │   Reverse proxy │  TLS termination, SSO
                        │  (nginx/Caddy)  │  X-Forwarded-* headers
                        └────────┬────────┘
                                 │ HTTPS
                  ┌──────────────┼──────────────┐
                  │              │              │
            ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
            │  api #1   │  │  api #2   │  │  api #N   │  uvicorn
            └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
                  │              │              │
                  └──────┬───────┴──────┬───────┘
                         │              │
                  ┌──────▼──────┐  ┌────▼────┐
                  │  postgres   │  │  redis  │  job state, broker
                  └─────────────┘  └────┬────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  │                                         │
        ┌─────────▼─────────┐                    ┌──────────▼─────────┐
        │  celery worker    │     queue="dose"   │  celery worker     │
        │  (default queue)  │                    │  (dose queue)      │
        │  fast tasks       │                    │  MCsquare runs     │
        └───────────────────┘                    └──────────┬─────────┘
                                                            │
                                                ┌───────────▼───────────┐
                                                │  celery beat (1x)     │
                                                │  cleanup scheduler    │
                                                └───────────────────────┘

       Shared filesystem (NFS / EBS / GCS-FUSE / Azure Files):
         /data/artifacts/  ← dose NIfTIs, Dij CSRs, geometry caches
         /data/uploads/    ← user-uploaded DICOM
```

Why this shape:

- **Stateless API workers** scale horizontally. They read jobs from
  Postgres and dispatch to Celery — nothing on local disk.
- **Two queues for Celery** (`default` and `dose`) lets you size the
  dose pool independently (fewer workers, more memory each) without
  starving fast geometry/beam-model jobs.
- **One Celery Beat instance** runs the cleanup tasks. Multiple beat
  instances will trigger duplicate sweeps — pin it to one node.
- **Shared filesystem** for artifacts. Object storage (S3, GCS) works
  too if you front it with a FUSE driver, but local FS is faster for
  the per-build read-then-write pattern.

## Environment variables — production checklist

Set these in your orchestrator (Docker `env_file`, k8s `ConfigMap` /
`Secret`, systemd `EnvironmentFile`). See `.env.example` for the
full list; what follows is what *must* change from defaults.

| Variable | Production value | Why |
|---|---|---|
| `RADIARCH_ENVIRONMENT` | `production` | Disables Celery eager mode. Required. |
| `RADIARCH_API_KEY` | High-entropy secret (e.g. `openssl rand -hex 32`) | D8.2. Empty disables auth — DO NOT ship empty. |
| `RADIARCH_DATABASE_URL` | `postgresql+psycopg://user:pass@host:5432/radiarch` | InMemoryStore loses every job on restart. |
| `RADIARCH_BROKER_URL` | `redis://redis-host:6379/0` (or RabbitMQ) | Same. |
| `RADIARCH_RESULT_BACKEND` | `redis://redis-host:6379/1` | Same. |
| `RADIARCH_ARTIFACT_DIR` | `/data/artifacts` (shared mount) | All workers must see the same path. |
| `RADIARCH_AUDIT_LOG_PATH` | `/var/log/radiarch/audit.jsonl` | D7.3. Ships to your log aggregator. |
| `RADIARCH_DOSE_STORE_MAX_GB` | Site-specific. Default 50 GB. | D7.2. Set to 70-80% of your dose volume. |
| `RADIARCH_INFLUENCE_STORE_MAX_GB` | Site-specific. Default 100 GB. | Dij matrices are big; size to your concurrent-optimization budget. |
| `RADIARCH_CELERY_WORKER_MAX_MEMORY_KB` | 4 GB for fast workers, 16 GB+ for dose workers | D7.1. Recycles leaked memory. |
| `RADIARCH_DOSE_SOFT_TIME_LIMIT_S` | Bench your worst clinical case + 30 % | Defaults work for SimpleFantom; clinical CT needs longer. |
| `RADIARCH_CORS_ORIGINS` | Explicit list of allowed origins | Default `*` is wrong for production. |

## Resource limits (D7.1)

The defaults in `config.py` are tuned for the bundled SimpleFantom +
analytic engine. Clinical MCsquare runs need more:

- **Soft time limit** = the time after which Celery raises
  `SoftTimeLimitExceeded` in your task. Lets the worker clean up
  partial files, write a `failed` job state, emit the audit event.
- **Hard time limit** = `SIGKILL`. Set ≥ soft + 60 s headroom.
- **Memory per worker** = `worker_max_memory_per_child`. Worker
  finishes its current task, then exits cleanly. Replacement is
  automatic.
- **Tasks per worker** = `worker_max_tasks_per_child`. Belt-and-
  suspenders against slow growth.
- **Rate limit** = `task_annotations[...]["rate_limit"]`. Prevents
  a flood of dose jobs from monopolizing the worker pool when
  someone scripts the API.

Sizing rule of thumb: one dose worker = 1 dedicated CPU + (2× peak
RSS of MCsquare for your CT size). MCsquare on a 512×512×200 grid
runs ~2-4 GB per process; budget 6-8 GB per worker.

## Optimization Service (Service 4)

The optimizer runs on its own `optimize` Celery queue. It is the heaviest
stage: a full solver loop of `Dij·w` matvecs, optionally over many robust
scenarios (each with its own `Dij`). Run optimization workers separately from
dose workers so a long inverse-plan run can't starve dose builds.

| Setting | Env var | Default | Notes |
|---|---|---|---|
| Soft time limit | `RADIARCH_OPTIMIZATION_SOFT_TIME_LIMIT_S` | 14400 (4h) | Heaviest stage; bump for large robust runs. |
| Hard time limit | `RADIARCH_OPTIMIZATION_HARD_TIME_LIMIT_S` | 15000 | Must exceed soft + cleanup headroom. |
| Rate limit | `RADIARCH_OPTIMIZE_RATE_LIMIT` | 2/m | Throttle scripted submissions. |
| Store cap | `RADIARCH_OPTIMIZATION_STORE_MAX_GB` | 30.0 | LRU + min-age eviction, same policy as `doses/`. |
| Checkpoints kept | `RADIARCH_OPTIMIZATION_CHECKPOINT_KEEP` | 5 | Latest N per `optimization_id`; older pruned each sweep. |

**Sizing:** optimization workers want **more CPU than dose workers** — the
`Dij·w` matvec (and the `Dijᵀ·dL/dD` transpose for the gradient) dominate the
loop and parallelize across BLAS threads. The Dij itself is loaded once per
scenario and reused every iteration, so RAM is dominated by `nnz(Dij) × 8 bytes
× n_scenarios`. Budget by Dij size, not by iteration count. Solver choice:
L-BFGS-B by default; Adam for >100k spots or ill-conditioned problems (see
`docs/adr/0002-optimization-solver-choice.md`).

## BAO Service (Service 5)

BAO shares the `optimize` Celery queue. It is the **heaviest** task in the
system: every candidate angle set it scores is itself a (short) fluence
optimization, so cost is `O(n_beams · n_candidates · scoring_iterations)`
matvecs. Keep the candidate sweep coarse (`angle_step_deg` 20–45°) and
`scoring_iterations` small for selection, then run a full-budget `/optimize/run`
on the winning beam model.

| Setting | Env var | Default | Notes |
|---|---|---|---|
| Soft time limit | `RADIARCH_BAO_SOFT_TIME_LIMIT_S` | 28800 (8h) | Largest budget — many inner optimizations. |
| Hard time limit | `RADIARCH_BAO_HARD_TIME_LIMIT_S` | 30000 | Must exceed soft. |
| Rate limit | `RADIARCH_BAO_RATE_LIMIT` | 1/m | Very conservative — one heavy run at a time. |
| Store cap | `RADIARCH_BAO_STORE_MAX_GB` | 5.0 | Results are metadata-only (small). |

**Sizing:** scale by `n_candidates × scoring_iterations`. Co-locate BAO with
optimization workers (same CPU profile) but expect a single BAO run to occupy a
worker for a long time — hence the `1/m` rate limit.

## Evaluation Service (Service 6)

Evaluation is read-only array analysis on its own `evaluate` Celery queue —
light and fast, except gamma analysis on a full grid (`O(N · window³)`; see
`docs/adr/0004-evaluation-metrics.md`). DVH + indices are sub-second on clinical
grids; budget for gamma only when it's requested.

| Setting | Env var | Default | Notes |
|---|---|---|---|
| Soft time limit | `RADIARCH_EVALUATION_SOFT_TIME_LIMIT_S` | 1800 | Gamma dominates; raise for large grids. |
| Hard time limit | `RADIARCH_EVALUATION_HARD_TIME_LIMIT_S` | 2400 | Must exceed soft. |
| Rate limit | `RADIARCH_EVALUATE_RATE_LIMIT` | 30/m | High — evaluation is cheap. |
| Store cap | `RADIARCH_EVALUATION_STORE_MAX_GB` | 5.0 | Reports are metadata-only. |

**Sizing:** evaluation workers are CPU-light; co-locate with the API or run a
small dedicated pool. The gamma window search is single-threaded numpy — if it
becomes the bottleneck on large grids, that's the documented swap point.

## Disk hygiene (D7.2)

The cleanup task (`radiarch.cleanup.dose_stores`) runs every
`RADIARCH_CLEANUP_INTERVAL_S` seconds (default: 1 h) and evicts
least-recently-accessed entries when a store exceeds its cap.
Entries younger than `RADIARCH_DOSE_MIN_AGE_HOURS` (default: 24 h)
are **protected** — never evicted — to shield an in-flight
optimization loop.

Set caps to 70-80 % of the underlying filesystem capacity. The task
sweeps both `doses/` and `influence/` independently, so a single
runaway optimization can't evict somebody else's nominal doses.

To monitor:

```bash
# Latest cleanup result
celery -A radiarch.tasks.celery_app inspect ping
celery -A radiarch.tasks.celery_app result <task_id>

# Or just grep the audit log
jq 'select(.event_type == "cleanup.swept")' < audit.jsonl | tail -1
```

## Auth (D8.2)

Static API key in the `X-API-Key` header, configured via
`RADIARCH_API_KEY`. Auth applies to all `/api/v1/dose/*` routes
when the key is configured. Other services (geometry, beam model)
are currently unprotected — front them with the reverse-proxy SSO
or add `Depends(api_key_auth)` to their routers.

Rotation: set the new key in the env, restart the API workers. The
audit log records the first six chars of the presented key so you
can correlate requests with rotated keys.

Future: swap to JWT/OAuth by replacing `api_key_auth` in
`src/radiarch/api/security.py`. Route decorators don't change.

## Audit logging (D7.3)

One JSONL line per dose-relevant event. Configure
`RADIARCH_AUDIT_LOG_PATH` to write to a file in addition to stderr.
Schema in `src/radiarch/services/audit.py` (`AuditEvent`).

Ship to your aggregator with anything that tails files
(filebeat, vector.dev, fluentd). Example vector config:

```toml
[sources.radiarch_audit]
type = "file"
include = ["/var/log/radiarch/audit.jsonl"]

[transforms.radiarch_audit_parsed]
type = "remap"
inputs = ["radiarch_audit"]
source = '. |= parse_json!(.message)'

[sinks.datadog_logs]
type = "datadog_logs"
inputs = ["radiarch_audit_parsed"]
default_api_key = "${DD_API_KEY}"
```

Useful queries:

```bash
# All failed dose builds in the last hour
jq 'select(.event_type=="dose.compute" and .state=="failed") |
    select(.timestamp > "'$(date -u -v-1H +%Y-%m-%dT%H:%M:%SZ)'")' < audit.jsonl

# Slowest builds today
jq 'select(.event_type=="dose.compute" and .state=="succeeded")
    | {dose_id, geometry_id, engine_name, duration_s}' < audit.jsonl \
  | jq -s 'sort_by(.duration_s) | reverse | .[0:10]'
```

## Backup + restore

- **Postgres** — backup as usual (`pg_dump`). Job rows are cheap
  to recreate (they're metadata + foreign keys into the artifact
  cache); the value is the audit trail.
- **`/data/artifacts/`** — backup or accept that on loss, builds
  will simply recompute from geometry + beam-model + weights. The
  cache-key index is regenerable by walking the directory.
- **`/data/uploads/`** — back up, these are originals.

## Health checks

- `GET /api/v1/info` — liveness (no DB, no Redis, just process up)
- `GET /api/v1/dose/engines` — engine availability snapshot
- `GET /api/v1/dose/engines/mcsquare` — MCsquare-specific
  diagnostics (binary path, OpenTPS importable)

Wire the first into your k8s `livenessProbe`, the second into your
monitoring (alert when `available: false`).

## Pre-launch checklist

- [ ] `RADIARCH_API_KEY` set to a 32-byte hex secret
- [ ] `RADIARCH_DATABASE_URL` points at a managed Postgres (not SQLite)
- [ ] `RADIARCH_BROKER_URL` / `RESULT_BACKEND` point at managed Redis
- [ ] `RADIARCH_ARTIFACT_DIR` is on a shared filesystem all workers can see
- [ ] `RADIARCH_AUDIT_LOG_PATH` set and aggregator collecting
- [ ] Disk caps (`*_STORE_MAX_GB`) sized to your filesystem
- [ ] Celery worker pool sized to your peak dose-job concurrency
- [ ] Celery Beat running on exactly **one** node
- [ ] Reverse proxy terminates TLS and forwards `X-API-Key`
- [ ] CORS origins narrowed from `*`
- [ ] `/api/v1/dose/engines/mcsquare` returns `available: true`
- [ ] Smoke test (`scripts/smoke_test.sh`) green against the prod URL
- [ ] Cleanup task runs in audit log within the first hour
- [ ] Backups configured for Postgres + uploads
- [ ] Runbook for: dose worker OOM, Postgres failover, Redis failover,
      MCsquare binary missing, audit log file full
