# A.1 Findings: cascade_flows EMU_TRACE_BUG (2026-04-25)

## Original bug (Phase E, 2026-04-23)

`docs/superpowers/plans/2026-04-23-phase-e-validation.md` recorded:
> `cascade_flows.chess`: HW captures 2 timestamped events (9-cycle span);
> EMU captures *nothing*. Classification: EMU_TRACE_BUG.

## Status today: **largely resolved by intervening work**

The traced MLIR generated on 2026-04-23 14:28 used the narrow 3-event
default (INSTR_VECTOR, INSTR_EVENT_0, INSTR_EVENT_1). Commit `13beeff`
("trace-inject: broader default event set matching mlir-aie upstream",
2026-04-23 15:08) widened the default to 8 events, including the
LOCK_*_REQ and *_STALL events that fire reliably on cascade tiles.

After force-rebuilding `aie-hw-cycles-traced.mlir` and rerunning
`emu-bridge-test.sh --with-cycle-diff -v cascade_flows --chess-only`
on 2026-04-25, the result is `DRIFT(ratio=0.01, diverge=7)` rather
than EMU_TRACE_BUG: EMU captures 5 events on the cascade-source tile.

## New picture (DRIFT, not bug)

| Tile | Role | HW events | EMU events |
|------|------|-----------|------------|
| (0,3) / HW(1,3) | cascade source (kernel1) | INSTR_VECTOR×2, LOCK_STALL×2, INSTR_LOCK_RELEASE_REQ×1 | INSTR_LOCK_ACQUIRE_REQ×1, INSTR_LOCK_RELEASE_REQ×1, MEMORY_STALL×3 |
| (1,3) / HW(2,3) | cascade middle (kernel2) | (none captured) | (none captured) |
| (1,2) / HW(2,2) | cascade sink (kernel3) | 4 unnamed events | (none captured) |

Cycles: HW span 6368, EMU span 61. EMU total emulator cycles `= 2256`
(`XDNA_EMU_STATUS: halt_reason=completed cycles=2256`).

## Root-cause analysis

**Three findings, all actionable but each is its own thread:**

### Finding 1: trace window is host-driven and fragile under EMU timing

Trace start/stop are broadcast events 15/14, fired by host control
packets. On HW, the host fires START → kernel runs ~6368 cycles → host
fires STOP. On EMU, the emulator outpaces wallclock, so the host fires
START and STOP nearly back-to-back. Result: the trace captures only the
small slice of EMU execution that overlaps the host's broadcast-fire
window.

This is *not* an EMU bug. It's a real consequence of the host-driven
trace window applied to a fast emulator. The fix surface is the
perfctr-based trace window (observability lead #2): use a tile-internal
PERF_CNT_0 fired N cycles after START to drive STOP, replacing the
host-fired BROADCAST_14.

Tracked under thread C.1 (event-bounded trace prototype) and A.2 (the
sweep + PC-anchored joining lets us cope with shorter windows).

### Finding 2: EMU and HW report disjoint events on the same tile

On t03, HW fires INSTR_VECTOR×2 and LOCK_STALL×2; EMU fires
INSTR_LOCK_ACQUIRE_REQ×1 and MEMORY_STALL×3. The
INSTR_LOCK_RELEASE_REQ×1 is the one shared event.

Two sub-questions:

- **INSTR_VECTOR**: does EMU's slot classifier mark `put_mcd(v32)` as
  vector-slot work? `put_mcd` is structurally a MOV-class instruction
  (vector-to-MCD-port move), and the VR_MV / Vector slot mapping in
  llvm-aie's TableGen would tell us whether it's classified as Vector
  or as Misc. If the latter, INSTR_VECTOR shouldn't fire — and if HW
  *does* fire it, that's the divergence.
- **MEMORY_STALL vs LOCK_STALL**: HW reports the wait as a lock stall
  (waiting on objFifo lock), EMU reports it as a memory stall. Worth
  checking which is canonical per AM025 §3 (event taxonomy).

Tracked: needs its own investigation. Not blocking: EMU still produces
trace data, the test still PASSes.

### Finding 3: t13 (cascade middle) silent on both sides

Neither HW nor EMU captures any events on t13 in this run. t13's kernel
is `func.call @extern_kernel2() : () -> () ; aie.end` with no acquire
or release ops. extern_kernel2 does `get_scd → upd_elem → put_mcd`,
which is a tight cascade-pipe. If the trace window doesn't overlap that
short execution, no events fire.

Not a bug. Documents that the default event set as currently configured
isn't a reliable per-tile presence signal.

## What this means for thread A

- **A.1 closed**: original EMU_TRACE_BUG no longer reproduces. Run
  artifacts in `build/bridge-test-results/20260425/cascade_flows.*`.
- **A.2 (thin defaults + sweep + PC-anchored joining)** would naturally
  surface findings 1 and 2 as per-(tile, event, PC) divergences and
  let us reason about them independent of the host-driven trace window.
- **A.5 (cycle accumulator + retire EMU_SECONDS_PER_CYCLE)** is needed
  before we can take ratio-based DRIFT classifications seriously.
- **C.1 (event-bounded trace prototype)** is the long-term fix for the
  short-window-on-EMU problem.

The DRIFT(0.01, diverge=7) classification is a useful signal but
shouldn't drive an EMU code change without first stabilising the trace
window (C.1) and confirming the cycle accumulator (A.5).
