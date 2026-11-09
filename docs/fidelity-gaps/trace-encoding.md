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

| Gap | Model vs hardware | Where | Status / rationale | Obs. |
|-----|-------------------|-------|--------------------|------|
| Held-level falling edge, close-to-empty | a held span whose deassert lands on a cycle with no *other* concurrent frame cannot encode its falling edge in mode-0 (no empty frame); the close defers to the next frame / end-of-segment. HW closes the span at the deassert cycle. | `src/device/trace_unit/mod.rs` (`set_event_level` / `commit_pending_frame`, the `active==0` defer); finding: 2026-06-08 falling-edge depends on concurrent levels (`xdna-emu-priv: research/findings/2026-06-08-lock-stall-falling-edge-depends-on-concurrent-levels.md`) | **DOCUMENTED.** Matches how HW closes a level span (no synthetic empty frame); the residual is only the *lone-level* case where no concurrent frame carries the close. With the per-cycle TRUE metronome the deferral is bounded to ~1 cycle. | confirmed |
| Held-level count under-emission (DDR-fill / bursty delivery) | our stream/DMA delivery was *smoother* than HW's bursty ~1024-cycle bursts, so level events asserted fewer times (e.g. memtile `PORT_RUNNING` ports decoded 71/187/90/156 cycles instead of the hardware law `sum(PORT_RUNNING) == words == 64` per port). | `src/device/trace_unit/mod.rs` (`commit_pending_frame`, skip-token held-level encoding); spec `xdna-emu-priv: research/specs/2026-06-08-skip-token-held-level-encoding.md` | **CLOSED (2026-06-27).** Root cause was **not** DDR jitter (that narrative was a metric artifact -- the cadence tool counted trace *frame-records*, not *spans*; corrected to span-based `tools/port-span-cadence.py`). The real defect was the encoder: pairs of RUNNING spans merged across the deassert->reassert gap. Four encoder/decoder/model fixes (`e0ce540f` decoder two-phase deactivate/activate; `ba1b4a59`† `Repeat(D-2)` for gap-opened lone holds; `2f85d1c9` `cycle_beat` set once per external interface, not on both push+pop; `5311905c`†+`90ce9fa8`† defer same-cycle level commits into the one-frame-per-cycle path) plus the two recv-path model fixes (see [`dma-stream-resources.md`](dma-stream-resources.md)) bring all four memtile ports to exactly 64. **Regression oracle: `sum(PORT_RUNNING) == words` per port.** All fixes are trace-only (data / cycle-counts / RUNNING-STALLED exclusivity unchanged). Full chain: finding `xdna-emu-priv: research/findings/2026-06-16-port-cadence-metric-was-frame-records.md`. | confirmed |

† Commit pruned from public history in the 2026-10 provenance cleanup (its
file also held removed trace code); the change is folded into the next
version of that file. The original commit is in xdna-emu-priv's
`archive/public-pre-round3/master`.

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

## Found with the LOCK_STALL pulse probe (2026-11-09, silicon vs emulator, not yet fixed)

`build/experiments/lock-stall-pulse`, `tools/gen_lock_pulse_probe.py`. Two markers
(`event #0` / `event #1`) a known number of bundles apart, decoded with the in-tree decoder
(`ts` adds `1 + cycles` per frame, `soc` removes the `1`):

| Gap | Silicon | Emulator | Status |
|-----|---------|----------|--------|
| Cycles field of a pure-pulse frame | the idle cycles between the frames: markers 2, 3, 5, 22, 102, 302 bundles apart decode `ts` 2, 3, 5, 22, 102, 302 apart (`soc` one less) | the full elapsed cycles: `ts` one late per frame (3, 4, 6, 23, ...), `soc` exact | **OPEN.** `emit_event_frame(active, gap)` in the pure-pulse branch of `commit_pending_frame` passes the elapsed gap; the decoder adds its own implicit 1 on top. The comparator reads `soc`, where the emulator is exact and silicon drifts one cycle low per frame, so tiles with many frames shift against each other; the fix is a convention change for every pulse frame (a held-level hold has its own `skip_debt` bookkeeping), held back as a design choice of the encoder. |
| INSTR_LOCK_ACQUIRE_REQ / _RELEASE_REQ stamp | request bundle at cycle c: REQ at c+1 (refused acquire too: a core held across the trace start has no REQ for it) | stamped when the request completes: c+4, or the end of a hold | **OPEN.** Needs the REQ recorded one step after issue; mode-1 (EventPc) frames carry the fetch PC at the REQ, which at c+1 is the fourth bundle after the request (the PC the emulator takes at completion today), not known until then. |
| LOCK_STALL pulse of a request window | one 1-cycle pulse at c+4 | **FIXED** 2026-11-09 (releases pulsed already, acquires did not) | closed |
| Held level at the start of the trace | first frame, one cycle after the start event | **FIXED** 2026-11-09 (`level_signals`) | closed |
