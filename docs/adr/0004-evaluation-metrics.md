# ADR 0004 — Evaluation metrics

**Status:** Accepted
**Date:** 2026-06-21
**Context:** Service 6 (Evaluation) — which plan-quality metrics to compute and how.

## Context

Evaluation turns a computed dose volume + structure masks into a
clinician-readable report. It is the read-only end of the pipeline (no engine, no
solver) and must produce the metrics a physician/physicist actually uses to
accept or reject a plan, in standard, citable forms.

## Decision

**DVH — cumulative.** For each structure we compute the cumulative DVH (fraction
of volume receiving ≥ each dose level) over `[0, max_dose]` bins, and extract the
standard scalars: mean/max/min, `D2/D50/D95/D98` (`Dx` = the (100−x) percentile),
`V_prescription`, and volume in cc. Cumulative (not differential) because that's
the form every `Dx`/`Vx` query reads off directly.

**Indices — ICRU-83 + Paddick.**
- **Homogeneity Index** `HI = (D2 − D98) / D50` (ICRU-83) — 0 is perfectly
  uniform. Robust to single-voxel outliers (uses D2/D98, not max/min).
- **Conformity Index** `CI = TV_PIV² / (TV · PIV)` (Paddick) — 1.0 is ideal;
  penalizes both target under-coverage and dose spillage into normal tissue.
  Chosen over the RTOG ratio because it is symmetric in those two failure modes.
- **Coverage** — fraction of target receiving ≥ prescription, and the global
  hotspot dose.

**Gamma — Low et al. 1998.** Combined dose-difference / distance-to-agreement
pass/fail per voxel vs a reference dose, with a low-dose threshold to exclude
noise and a global-vs-local normalization switch. Implemented as a direct
search-window minimization (box radius scaled to the DTA). This is `O(N ·
window³)` — fine for the bundled fantom and test grids; **for full clinical grids
it should be replaced with a kd-tree / GPU kernel** (the current implementation
caps the per-axis search radius so a pathological spacing can't blow up). This is
a deliberate v0.1 simplicity-over-speed choice, noted here so it's not mistaken
for production-grade gamma.

## Consequences

- All metrics are pure-numpy functions (`dvh.py`, `indices.py`, `gamma.py`),
  decoupled from persistence and engines, so they're unit-tested directly on
  small arrays with hand-checkable answers.
- Results are JSON-only (DVH curves serialize inline), so the store is light and
  the report is directly consumable by the OHIF DVH panel.
- Adding metrics (e.g. EUD readout, NTCP/TCP models, ICRU reference points) is a
  localized addition to the metric modules + the result model.
- The gamma implementation is the known scaling bottleneck; swapping it is
  isolated to `gamma.py` behind the same `gamma_index` signature.
