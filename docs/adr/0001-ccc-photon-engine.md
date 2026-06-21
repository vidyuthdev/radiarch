# ADR 0001 — Photon Dose Engine: Build, Buy, or Defer

| Field | Value |
|---|---|
| Status | **Accepted — Defer to v2 (D9), interim shim retained** |
| Date | 2026-06-05 |
| Deciders | Radiarch engineering, clinical advisory |
| Supersedes | — |
| Related | D8.3 task; `src/radiarch/services/dose_engines/ccc.py` |

## Context

Radiarch ships with two engines registered:

- `analytic` — deterministic toy physics, used by tests and as a
  baseline. Implements all five `DoseEnginePlugin` methods including
  influence + gradient.
- `mcsquare` — real Monte Carlo via vendored OpenTPS. Proton-only
  (`PROTON_PBS`).

The `ccc` (Collapsed-Cone Convolution) photon engine is currently a
**stub**: it registers, claims `PHOTON_IMRT`, but every method raises
`EngineUnavailableError`. A photon engine is required for IMRT /
VMAT workflows, which are 60-70 % of clinical demand.

This ADR decides what to do about it.

## Options

### Option A — Build it ourselves

Port one of the open-source CCC implementations into Radiarch:

| Project | License | Maturity | Bindings |
|---|---|---|---|
| **matRad** | GPL-3.0 | High; clinically used | MATLAB primary, partial Python |
| **PORTPY / portpy-core** | Apache-2.0 | Medium; research-grade | Native Python |
| **PyPlanScoring CCC** | MIT | Low; partial | Python, no dose-engine, scoring only |
| **deepCCC** (Mayo) | Apache-2.0 | High; clinically validated | C++ with Python wrapper |

**Effort estimate.** 8-12 engineer-weeks for a vendored, tested,
clinically-comparable photon CCC engine. Breakdown:

- 2 wk: licence due diligence + port skeleton
- 3 wk: kernel + density correction + ray trace
- 2 wk: parallelization (GPU optional)
- 2 wk: validation against TG-119 / TG-244 benchmarks
- 1-3 wk: clinical QA loop with our medical physicist

**Pros.** Single codebase, fully open, no licence runtime cost,
implements full `DoseEnginePlugin` protocol including beamlets +
gradient (advantage matRad gives us). Aligns with the proton MCsquare
pattern of vendoring upstream.

**Cons.** Photon CCC is much harder to get right than the proton
MC wrapper — kernel calibration is per-LINAC and clinical sign-off
requires medical physics time we don't currently have on call.
Long-tail of edge cases (heterogeneity, electronic disequilibrium)
that will surface after launch. Maintenance burden grows: every CCC
fix is now ours.

### Option B — Buy / integrate a vendor SDK

Wrap a commercial photon engine behind the same `DoseEnginePlugin`
protocol.

| Vendor | API surface | Pricing (rough) | Deployment |
|---|---|---|---|
| RaySearch RayStation SDK | Full TPS API | $$$ — site licence | On-prem, Windows |
| Varian Eclipse Scripting (ESAPI) | TPS-scoped | $$ — bundled w/ Eclipse | On-prem, Windows |
| Elekta Monaco SDK | Limited | $$ | On-prem |
| MIM Software | Limited | $ | Cloud + on-prem |

**Effort estimate.** 4-6 engineer-weeks for a working wrapper
against one vendor (assumes the customer already owns the SDK).
Most of the work is plumbing: marshalling our `BeamModelBundle` into
the vendor's plan format, watching for async result files, parsing
DICOM-RTDOSE back.

**Pros.** Fast to ship. Vendor owns the dose-engine correctness +
QA. Clinical sign-off is largely covered by the vendor's existing
510(k) / CE-MDR clearance for their TPS.

**Cons.** **Vendor lock per deployment** — every hospital uses
their own TPS, so we'd need N adapters. Licence costs are
customer-borne but raise procurement friction. SDK quality varies
wildly (RayStation is good, Elekta is rough). Most vendor SDKs are
Windows-only — kills our Linux container story.

### Option C — Defer; ship without photon, position for v2

Mark photon as out-of-scope for v1. Promote the proton MCsquare
engine to full production status. Ship Radiarch as a **proton TPS
service** with photon support landing in v2.

**Effort estimate.** Already done — `ccc.py` stub + `EngineUnavailableError`
behavior is the current state.

**Pros.** Fastest path to a shippable v1. Proton TPS is a real
market (every proton centre is desperate for better software).
Doesn't force a build-or-buy gamble before we have user data on
which workflows actually matter. Keeps the engine-protocol
contract honest: we *can* swap in a CCC backend later without
breaking API consumers, because the routes already negotiate engine
availability via `/dose/engines`.

**Cons.** Eliminates 60-70 % of clinical addressable market for v1.
Sales conversations get awkward ("can it do IMRT?" — "not yet").

## Decision

**Adopt Option C — defer photon to v2 (codenamed D9).** Keep the
`ccc.py` stub registered so the protocol stays exercised. Position
Radiarch v1 as a proton TPS service.

Rationale:

1. **MCsquare proton work is now production-ready** (D6/D7/D8.1
   complete). Shipping photon poorly would dilute that.
2. **We don't have a medical physicist on retainer** for clinical
   QA of a homegrown CCC implementation. Option A's 8-12 weeks
   doesn't include that loop.
3. **Vendor SDK fragmentation** makes Option B a per-customer
   engineering project, not a product. We don't want to be a
   systems integrator.
4. **The protocol holds.** A future `BuildCCCEngine` task can
   slot in behind the same `DoseEnginePlugin` contract; API
   consumers already handle 501 cleanly via the engine health
   endpoint.

## Consequences

**Immediate (this sprint).**

- No code changes; `ccc.py` stays as-is.
- Update the project's positioning materials to emphasize proton.
- Add a "photon support: planned for v2" note to the README and
  `/api/v1/info` response.

**Within 3 months.**

- Run clinical user research with two proton centres. Goal: confirm
  what photon workflows they'd actually want (IMRT? VMAT? both?
  re-irradiation only?).
- Spike a 1-week PoC against **one** of the Option A candidates
  (PORTPY most likely — Apache-2.0, native Python, smallest port).
  Goal: bound the real effort estimate.

**Within 6-9 months (D9 / v2).**

- Revisit this ADR with the spike results and the user-research
  findings. Re-decide between Option A (vendored open-source CCC)
  and Option B (commercial SDK), or split: ship open-source CCC for
  research customers, vendor-SDK adapters for clinical.

**Reversal cost if we're wrong.**

- *If we should have built (A) and didn't*: 8-12 weeks of catch-up
  whenever the photon market becomes critical. Cheap option to
  hold open.
- *If we should have bought (B) and didn't*: same, plus we already
  have the protocol — wrapping a vendor SDK is a few weeks once we
  pick one.

The defer decision is **fully reversible** because we've kept the
`DoseEnginePlugin` contract intact and registered the stub. The
only cost is the v1 positioning hit.

## References

- `src/radiarch/services/dose_engines/protocol.py` — engine
  contract that any photon implementation must satisfy.
- `src/radiarch/services/dose_engines/ccc.py` — current stub.
- TG-119 (commissioning) and TG-244 (independent verification) —
  the validation bar any clinical CCC must clear.
- AAPM MPPG 5.a — vendor TPS commissioning guidance (relevant for
  Option B due-diligence).
