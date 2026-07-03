---
class: trace-encoding
subsystem: trace unit held-level / skip-token span encoding
posture: encoding-artifact -- the underlying mechanism (cycle_beat, level rising/falling) is faithful; the disagreement is in how spans are *encoded* into trace frames
status: encoding layer CLOSED (2026-06-27); one residual falling-edge case documented-and-bounded
---

# Trace Held-Level Encoding Gaps

After the held-level re-architecture (`LOCK_STALL` / `PORT_*` / DMA
stall-starvation / core-stall family emit as B..E spans, decoded by upstream
`parse_trace`), the span *structure* is faithful. These are **timing/encoding
artifacts, not mechanism bugs**, and are stable Phoenix->Strix (DMA delivery
timing, not silicon-version-specific). Measured on
`vec_mul_trace_distribute_lateral` and `add_one_using_dma` (peano, interpreter
backend) vs real NPU1.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| Held-level falling edge, close-to-empty | a held span whose deassert lands on a cycle with no *other* concurrent frame cannot encode its falling edge in mode-0 (no empty frame); the close defers to the next frame / end-of-segment. HW closes the span at the deassert cycle. | `src/device/trace_unit/mod.rs` (`set_event_level` / `commit_pending_frame`, the `active==0` defer); [finding: 2026-06-08 falling-edge depends on concurrent levels](../superpowers/findings/2026-06-08-lock-stall-falling-edge-depends-on-concurrent-levels.md) | **DOCUMENTED.** Matches how HW closes a level span (no synthetic empty frame); the residual is only the *lone-level* case where no concurrent frame carries the close. With the per-cycle TRUE metronome the deferral is bounded to ~1 cycle. |
| Held-level count under-emission (DDR-fill / bursty delivery) | our stream/DMA delivery was *smoother* than HW's bursty ~1024-cycle bursts, so level events asserted fewer times (e.g. memtile `PORT_RUNNING` ports decoded 71/187/90/156 cycles instead of the hardware law `sum(PORT_RUNNING) == words == 64` per port). | `src/device/trace_unit/mod.rs` (`commit_pending_frame`, skip-token held-level encoding); spec [`2026-06-08-skip-token-held-level-encoding.md`](../superpowers/specs/2026-06-08-skip-token-held-level-encoding.md) | **CLOSED (2026-06-27).** Root cause was **not** DDR jitter (that narrative was a metric artifact -- the cadence tool counted trace *frame-records*, not *spans*; corrected to span-based `tools/port-span-cadence.py`). The real defect was the encoder: pairs of RUNNING spans merged across the deassert->reassert gap. Four encoder/decoder/model fixes (`70b25060` decoder two-phase deactivate/activate; `edf72f68` `Repeat(D-2)` for gap-opened lone holds; `52649b88` `cycle_beat` set once per external interface, not on both push+pop; `924ad3c0`+`8b82dabf` defer same-cycle level commits into the one-frame-per-cycle path) plus the two recv-path model fixes (see [`dma-stream-resources.md`](dma-stream-resources.md)) bring all four memtile ports to exactly 64. **Regression oracle: `sum(PORT_RUNNING) == words` per port.** All fixes are trace-only (data / cycle-counts / RUNNING-STALLED exclusivity unchanged). Full chain: finding [`2026-06-16-port-cadence-metric-was-frame-records.md`](../superpowers/findings/2026-06-16-port-cadence-metric-was-frame-records.md). |

## Core-stall family conversion (coverage note, not a confirmed disagreement)

The core stall family conversion (MEMORY_STALL / STREAM_STALL / CASCADE_STALL as
held levels) is **not exercised by `distribute_lateral`** -- its trace config does
not include events 23/24/25 -- so those three have unit coverage but **partial
HW-trace validation**. MEMORY_STALL is now HW-validated and divergent (a
core-timing gap, tracked in [`core-compute-timing.md`](core-compute-timing.md));
STREAM_STALL / CASCADE_STALL still await a cascade/stream-stall kernel capture.

Two related modeling notes (not HW-confirmed disagreements, hence not rows
above): mode-1 (EventPc) PC anchoring is dropped for stall held levels (the level
path carries no PC, matching the LOCK_STALL precedent), and the executor-level
*intra-bundle* structural bank conflict stays a bounded MEMORY_STALL pulse (it
resolves within one executor step, so a same-step rising+falling would encode as
a zero-width span).
