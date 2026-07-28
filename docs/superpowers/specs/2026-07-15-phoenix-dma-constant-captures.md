# Phoenix DMA-Constant Captures -- Design Spec

**Status:** approved design (Maya, 2026-07-15). Next: implementation plan.

## Goal

Bank the Phoenix-only hardware measurements that pin three DMA/token constants
the emulator currently carries as inferred, un-derived, or unbounded -- **before
the Strix swap retires the Phoenix NPU** (a one-way door; these captures become
impossible afterward). Where the open-source spec (AM020) already pins a
constant, close it by derivation and skip the capture; where it does not, run a
generous HW sweep on the existing trace pipeline.

## Context

Three open fidelity-gap items are HW-empirical-gated on Phoenix:

1. **Memtile bank-access width / strided-channel parallelism** -- the 16-byte
   granule cap is applied to the memtile DMA on inference from the compute tile
   (`docs/fidelity-gaps/dma-stream-resources.md`, granule-cap row).
2. **MM2S egress FIFO depth** (`DMA_MM2S_EGRESS_FIFO_DEPTH = 12`) -- the one
   un-derived constant of the bank-arbitration arc, from AMD's device-char model
   (NPU1.json), unconstrained by any open-source source or capture.
3. **TCT token buffer depth** -- `src/device/dma/token.rs` models the completion-
   token backpressure as unbounded; aie-rt exposes only a 1-bit `STALLED_TCT`.

### Derive-from-toolchain checks already done (2026-07-15)

- **Memtile width + interleave: DERIVABLE.** AM020 ch.5 states 16 banks each
  128-bit wide (line 31/105), interleaved at 128-bit granularity wrapping every
  256B across 16 banks (line 137) -- which the emulator's memtile bank map
  `(addr>>4)&0xF` (`src/device/banking.rs:49`) already implements. A single
  channel "access[es] memory over a shared interface" (line 153), grounding the
  single-granule-per-cycle cap. **The residual is one interpretive corner** the
  doc does not state outright: whether a *strided* single channel serializes its
  four stream words across cycles (the cap) or gangs shared interfaces to write
  four banks in one cycle. Belt-and-suspenders HW confirmation chosen (Maya) to
  nail it while Phoenix exists.
- **TCT token buffer depth: NOT DERIVABLE.** AM020's Task-Completion-Tokens
  section (ch.1:189) describes the token as "a single stream word... routed back
  to the issuing controller through the standard stream-switch network" -- no
  buffer depth stated; the backpressure is the stream path itself. AM020 *does*
  pin the **input task queue** at four tasks/channel (ch.2, ch.5:51/65), a
  different resource. TCT stays HW-gated; capture is characterize-only.

### Bonus derivation (small, separate)

AM020 pins the **DMA input task queue** at 4 tasks/channel. Check whether the
emulator bounds it (or models it unbounded like the token buffer); if unbounded,
that is a cheap derive-from-toolchain fix independent of the captures.

## Guiding principle: more captures, not fewer

HW is the cheap oracle (microseconds per run); EMU is the slow part. Every probe
runs a **full sweep with replicates and controls**, not the minimal discriminator:

- >= 3 HW repeats per variant (`_r1/_r2/_r3`), pooled and medianed, as
  `bankdisc` does; add a multi-run noise campaign
  (`tools/multirun-trace-campaign.py`) where a variance claim rides on the result
  (egress depth, TCT depth).
- Sweep the independent variable across its full meaningful range, not two
  endpoints (strides, lock-dwell K, task sizes).
- Always include the pre-registered **validity control** (an `idle`/`apart`
  variant that must read the null result) and the free **grounding anchors**
  (`PERF_CNT_2`, `INSTR_EVENT_0/1`) to prove the trace window did not truncate.

## Shared infrastructure (reused as-is, do not rebuild)

The 6-stage trace pipeline in event-time mode 0 (the real trace unit -- NOT
`AIE_RW_ACCESS`, NOT perf-counter polling). Per experiment we write exactly two
Python files -- a generator modeled on `tools/experiments/bankdisc.py` and an
analyzer modeled on `bankdisc_measure.py`/`bankdisc_analyze.py` -- pick the event
list in `trace_config.json`, and drive it through `bridge-runner/build/
bridge-trace-runner` under `env -u XDNA_EMU` (unsetting `XDNA_EMU` targets the
real NPU). Stages 2-5 (trace-inject, compile, run, decode) are reused unchanged.

**Timebase traps** (do not relitigate; baked into every analyzer):
- Use `ts`, never `soc` (soc subtracts a physically-real +1).
- Level events (STALLED_LOCK, CONFLICT_*, BACKPRESSURE, TOKEN_STALL): compare
  **interval area** from the mode-0 B/E rebuild, never decoded record counts
  (held signals under-count ~2.2x).
- Discrete events (FINISHED_BD, FINISHED_TASK): the rising-edge `ts` is valid.
- `START_TASK`/`FINISHED_TASK` fire **zero** times for a self-looping single-BD
  chain -- bracket transfers with `FINISHED_BD` + `STALLED_LOCK` falling edge.

---

## Experiment A -- Memtile bank-width + strided-channel parallelism

**Constants pinned:** memtile `access_granule_bytes` (confirm 16B), and the
strided single-channel per-cycle bank-access count (1 vs up to 4).

### A0 -- Derivation update (no HW)

Upgrade three code comments and the gap-doc row from "unvalidated/inferred" to
AM020 ch.5 citations:
- `src/device/banking.rs:48` ("Unvalidated: preserve the previous flat
  interleave") -> AM020 ch.5:137 confirms 128-bit-granularity interleave across
  16 banks, wrap 256B.
- `src/device/banking.rs:63` (`access_granule_bytes` memtile arm) -> AM020
  ch.5:105 confirms 128-bit banks.
- `src/device/dma/engine/stepping.rs:261` (memtile cap "an INFERENCE") -> AM020
  ch.5:153 "shared interface" grounds the single-port cap; the strided corner is
  the sole residual, confirmed by A2.
- `docs/fidelity-gaps/dma-stream-resources.md` granule-cap row: INFERRED,
  HW-gated -> derived (width/interleave/port) + HW-confirmed (A2).

### A1 -- Confirm bank-access width (two-DMA contention)

Row-1 memtile kernel (no core), two DMA channels contending one physical bank on
pinned buffers, sweeping `MEM_TILE_CONFLICT_DM_BANK_0..15` (events 112-127; note
16 banks, not the compute tile's 8). Invert conflict area under the single-port
round-robin model (as `bankdisc_analyze.py` does) -> bytes/access. Expect 16.0 B,
cross-checking AM020. Include the `apart` control (contenders in different banks
-> conflict area ~0) and an `idle` control.

### A2 -- Strided single-channel parallelism (the open corner)

One memtile MM2S with a strided BD reading `word[i]` from `base + i*stride` so
each word lands in a fresh bank when `stride >= 16B`. Fixed N words per run,
sweep **stride** across the full range {contiguous 4B, 16B, 32B, 64B, 128B,
256B-wrap} and N across a couple of sizes. Measure `FINISHED_BD` span (bracketed
by the preceding `STALLED_LOCK` falling edge).

**Discriminator:** `span(strided) / span(contiguous)`, drain path common to both
so the difference isolates the memory port:
- **~4** -> the channel serializes (one 16B granule/cycle) -> the emulator's cap
  is correct; close the residual.
- **~1** -> the channel writes/reads 4 banks in one cycle -> the cap is wrong for
  the memtile; the fix is to lift the cap for `BankLayout::MemTile` (throughput
  only -- memtile DMA does not arbitrate).

Sanity: `MEM_TILE_CONFLICT_DM_BANK_n` ~0 (single channel, no contender).

**Deliverables:** `tools/experiments/memtile_bankwidth.py` (generator, variants
for A1 contention + A2 stride sweep), `tools/experiments/memtile_bankwidth_
measure.py` (spans + conflict areas) and `..._analyze.py` (width inversion +
strided ratio); finding `docs/superpowers/findings/2026-07-15-memtile-bank-
access-width.md`; the A0 comment/gap updates; constant fix only if A2 says ~1.

---

## Experiment B -- MM2S egress FIFO depth

**Constant pinned:** `DMA_MM2S_EGRESS_FIFO_DEPTH` (currently 12).

Compute-tile MM2S (where the constant lives), BD/lock structure that fills the
egress FIFO to capacity, then stalls the **memory-side fetch** on a lock while
the **stream side keeps draining**. Measure the delay from `MM2S_0_STALLED_LOCK`
onset to `MM2S_0_MEMORY_STARVATION` onset = **egress occupancy in beats at
stall-time** -- the exact mirror of the S2MM ingress finding
(`2026-07-14-dma-memory-pressure-event-semantics.md`), which pinned ingress at
~15-16 beats from "backpressure asserts +15" with zero variance over 39 windows.
Confirm the stream is draining (not itself backpressured) via
`MM2S_0_STREAM_BACKPRESSURE` ~0 during the window.

**Sweep (generous):** build on `producer_probe.py`'s K-dwell machinery -- sweep
the fill/dwell so the FIFO reaches full at a range of pre-stall offsets, and read
depth as the *maximum* stable STARVATION-onset delay across the sweep (the FIFO's
ceiling). Multi-run campaign for the variance claim (mirror the S2MM finding's
zero-variance standard). Controls: a cold-start window (STARVATION expected only
on the first empty fill) and a never-stalled window (STARVATION = 0).

**The crux -- escalate, do not abandon (Maya, 2026-07-15):** STARVATION fires
only if the FIFO is genuinely full at stall-time. This is one of the final
fidelity gaps, so if the naive mid-transfer lock-stall cannot top the FIFO (the
stream drains it as fast as the fetch fills it), get clever and *focus* -- do not
report BLOCKED and walk away. Escalation levers, cheap to try since HW is cheap:

1. **Decouple fill from drain.** Backpressure the *stream* side first (hold a
   downstream consumer so it does not accept) while the memory-side fetch fills
   the FIFO to its ceiling; then release the stream to drain while blocking the
   fetch. STARVATION onset delay from a known-full start = full depth. Cleanest.
2. **Starve the fetch harder.** Make the memory-side fetch rate ~0 for longer
   than the FIFO depth -- e.g. bank contention (a core hammering the MM2S source
   bank so it loses arbitration every cycle) or a slow/strided source BD whose
   fetch rate is below the stream drain rate, so the FIFO empties.
3. **Sweep the dwell and read the ceiling.** Vary the pre-stall dwell so the FIFO
   reaches a range of occupancies; the maximum stable stream-flow duration before
   STARVATION across the sweep is the depth.

The one hard line: still **no guessing or fitting a number.** If every escalation
genuinely fails to expose the depth, that itself is a characterized result worth
documenting -- but exhaust the levers first.

**Deliverables:** `tools/experiments/mm2s_egress_depth.py`,
`..._measure.py`; finding `docs/superpowers/findings/2026-07-15-mm2s-egress-fifo-
depth.md`; set `DMA_MM2S_EGRESS_FIFO_DEPTH` to the measured value (or record the
gap as pinned-with-caveat if BLOCKED) and update the gap-doc row.

---

## Experiment C -- TCT token buffer depth (characterize)

**Constant pinned (if clean):** the completion-token buffer depth behind
`Stalled_TCT`; `token.rs` models it unbounded, NPU1.json's 128 is unvalidated.

Multi-task BD chain (so `FINISHED_TASK` fires, unlike self-looping single-BD)
completing many small tasks while the **token-return stream route is throttled**
to a slow/backpressured consumer, so completion tokens back up. Count
`FINISHED_TASK` edges (or the outstanding-token count) before
`DMA_TASK_TOKEN_STALL` (event 102 memtile / 75 shim / 140 memtile-variant)
asserts = token buffer depth. Also read status-register bit[5] `Stalled_TCT` as a
cross-check.

**Sweep (generous):** vary task size (tokens per unit time) and the return-route
throttle across their range; look for a consistent outstanding-count at
TOKEN_STALL onset. Multi-run for stability.

**Risk (accepted, characterize-only):** controlling the token route's drain rate
is the hard part, and the result may characterize the mechanism (TOKEN_STALL
onset behavior) without pinning a crisp integer depth. That is an acceptable
outcome -- document the mechanism and any bound observed; do not force a number.

**Deliverables:** `tools/experiments/tct_token_depth.py`, `..._measure.py`;
finding `docs/superpowers/findings/2026-07-15-tct-token-buffer-depth.md`; bound
the token buffer in `token.rs` if pinned, else document the mechanism + observed
bound in the gap-doc row.

---

## Order & cross-cutting

**Order:** A -> B -> C (most-derivable/self-checking first; riskiest last, so two
solid numbers are banked before the fuzzy one). Plus the A0 derivation and the
input-task-queue=4 bonus check, which need no HW and can land first.

**Per experiment:** design generator+analyzer -> HW capture (full sweep +
replicates + controls) -> finding (the finding is the deliverable) -> fix the
constant / close the gap-doc row.

**HW discipline (from CLAUDE.md):** never run two HW suites concurrently; no
`xrt-smi` during a HW run; HW invocations use `env -u XDNA_EMU
XDNA_EMU_RUNTIME=release`; a single capture is cheap, run freely. Recover a
wedged NPU with `pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'` before
escalating.

## Out of scope

- The producer stall-count residual (deferred by design to the future
  variable-latency memory / RAM-sim model).
- Firmware-banked items (core reset Part 2, varway56, dispatch latency).
- BD reuse/pool wedge (won't-fix, non-monotonic).
- Any calibration/knob-tuning to hit a target number -- if a capture does not
  pin a constant cleanly, it is documented, not fitted.
