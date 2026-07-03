# SP-4a: deadlock-full (EMU) vs flowing-equilibrium (HW) — consolidated data

**Date:** 2026-07-02  **Issue:** #140 (SP-4a cold-start fill-state)
**Status:** DATA CONSOLIDATION for joint analysis. Fable's pre-arm fabric-fill fix
(a) was implemented and proven a **no-op on the oracle** (still +1492). This doc
brings the decisive EMU + HW + toolchain data into one place so we can settle the
real mechanism before touching code again.
**Working evidence:** `build/experiments/sp4a-coupling-probe/` (gitignored):
`lockprobe/sp4a.log` (per-cycle lock/FSM probe, cap 9000), `postfix/` (oracle),
`dbg/` (terminal-map gate trace).

## The oracle (unchanged target)

Shim S2MM_0 first `STREAM_STARVATION` offset, relative to its `START_TASK` base:
- **HW: +13** (then relay-paced ~+64). Pipeline EMPTY at drain-start.
- **EMU: +1492** (then steady +66). Pipeline DEADLOCK-FULL at drain-start.
The whole gap is the first-starvation offset; steady-state cadence already matches.

## Kernel + lock map (lean `of_q0_lean.py`, runs at col 1)

`ProdCore(1,2) -> MemTile(1,1) -> ConsA(1,3) -> (relay) -> MemTile(1,1) -> shim(1,0)`.
All objectfifos DEPTH-2, object = 64 words (256 B). Drain = one 2048-word BD.

Two zero-copy links share a memtile buffer + lock pair each:
- `object_fifo_link(of_src,[of_d])`: memtile **src_cons** buf (locks 0=prod/free, 1=cons/full).
- `object_fifo_link([of_j],of_out)`: memtile **j_cons** buf (locks **2=prod/free, 3=cons/full**)
  — **this is the of_out send buffer**; the shim S2MM reads it.

Probe fields: `mt_srcP/C` = memtile locks 0/1, `mt_jP/C` = memtile locks 2/3
(the of_out buffer: **jP=free slots, jC=full slots ready to send**), `prod_srcP/C`
= producer locks 0/1, `consA_dP/C/jP/jC` = consA locks 0/1/2/3.

## EMU timeline (from `lockprobe/sp4a.log`, per-cycle)

| cy | event | of_out (j_cons) | of_out FSM | chain |
|----|-------|-----------------|-----------|-------|
| 0 | init | jP=2 jC=0 (empty) | AcquiringLock(67) — waiting for first full slot | all init |
| ~353 | first full of_out buffer arrives; MM2S **acquires it** | jP=0 jC=0 | AcquiringLock(acquired)→MemoryLatency→Transferring | filling |
| ~353→799 | MM2S reads 4 words into stream_out, **stalls** (shim un-armed, TREADY low) | jP=0 jC=1 | **Transferring(16/256) — STUCK** | freezing |
| **799 → 6779** | **TOTAL DEADLOCK, frozen ~6000cy** | **jP=0 jC=1** (2 slots occupied: 1 in MM2S transfer + 1 full) | Transferring(16/256) frozen | **every stage 0-free/1-full: prod srcC=1 blocked, mt srcC=1, consA dC=1, jC=1** |
| **6779** | shim S2MM arms (`START_TASK` base) | | Transferring(92/256) advancing | deadlock breaks |
| ~7202 | shim starts accepting (its own BD setup) | | | flood begins |
| 7346 | first of_out object drains | | DrainingEgress | cascade |
| 7346→~8271 | **~14 objects flood contiguously, one per ~65cy** (7346,7414,7479,7544,…) | jC oscillates 1↔0 refilling | Transferring/DrainingEgress cycling | whole chain cascades smoothly |
| **8271 (+1492)** | shim **first STREAM_STARVATION** | of_out finally runs dry | | production-paced thereafter |

**EMU essence:** the of_out MM2S faithfully waits for a full buffer (cy 353), acquires
ONE, reads 4 words into stream_out, then **stalls holding it** because the shim is
un-armed. The S2MM fills the 2nd slot. The 2-deep buffer is now full → backpressure
freezes the whole chain into **total deadlock for ~6000cy**. When the shim arms, the
deadlock-full chain **cascades smoothly** — ~14 objects back-to-back — and the shim
isn't starved until that backlog+production runs dry at **+1492**.

## HW picture (trace-free in-kernel-timer W1/W2 + decoded trace)

- Producer **flows**: 12 of_src releases pre-drain, with an **839cy stall at iter7**,
  then continues. NOT frozen at 2 — reaches a flowing equilibrium.
- ConsA flows: 5 of_j releases pre-drain, smooth.
- **of_out stays EMPTY** until the drain dispatches.
- of_out send-port (slot5) first span at **+1103** (not +0); shim starves **+13**
  (empty from +13, first object only arrives ~+1103), then relay-paced ~+64.
- The write32 sweep proved this is **window-length-independent**: extending the
  pre-drain window 5× (N=256, +28000cy) leaves HW empty every time. So HW's stable
  pre-drain state has **of_out empty**, indefinitely.

## Toolchain semantics (aie-rt / mlir-aie, agent-derived)

- No engagement gate/token anywhere. Interior channels CDO-started; shim S2MM drain
  is Idle until the runtime `npu_dma_memcpy_nd`.
- objectfifo lock: j_cons prod init=2 (free), cons init=0 (full). An MM2S BD acquires
  its cons lock, transfers to the stream, and **cannot complete/release while
  `Stalled_Stream_Backpressure`** (shim un-armed → TREADY low).
- Circuit route: TREADY propagates **combinationally end-to-end**; a fully-backpressured
  circuit is EMPTY, not FIFO-full. (EMU models each hop as a fill-first FIFO.)

## The divergence, side by side

| | EMU | HW |
|---|-----|-----|
| pre-drain producer | produces ~2, then **frozen** (deadlock) | produces **12** (stall at iter7, continues) |
| pre-drain of_out (j_cons) buffer | **FULL** (2: 1 in MM2S transfer + 1 waiting) | **EMPTY** (shim starves +13 ⇒ ≤~13 words) |
| pre-drain chain | **total deadlock**, frozen ~6000cy | **flowing equilibrium**, producer stalled but of_out empty |
| at arm | deadlock-full chain **cascades smoothly**, ~14 objects contiguous | of_out empty; first object only at +1103 |
| first starvation | **+1492** | **+13** |

## The crux question (for joint analysis)

**Why does the EMU freeze into deadlock with of_out FULL, while HW reaches a flowing
equilibrium with of_out EMPTY — given identical depth-2 fifos and the same lock init?**

Two sub-questions the data raises:

1. **Does HW's of_out MM2S acquire+hold a buffer pre-drain (like EMU), or not?**
   The EMU MM2S acquires ONE j_cons buffer at cy 353 and holds it (Transferring,
   stalled), and the S2MM fills the 2nd → of_out full. If HW's MM2S instead does
   NOT acquire/read a buffer until the shim's TREADY is live (stays at lock-acquire,
   0 words read), of_out stays empty and the S2MM can only fill within lock credit.
   This is the "of_out empty vs full" axis — worth ~128 words (2 objects) but not the
   whole gap. **We have no HW lock data; this is the key unknown.**

2. **Why does the EMU chain cascade SMOOTHLY at arm (~14 objects contiguous) where HW
   is relay-paced (first object +1103, then bursty)?** Even if of_out started empty,
   the EMU's deadlock-full upstream would still cascade in continuously. HW's relay
   has a per-object cold latency (~1103cy for the first object end-to-end from a
   flowing-but-not-deadlocked state). This is the "EMU inter-tile delivery is too
   smooth at cold-start" axis = known-fidelity-gaps rows 56/57, the residual prior
   sessions found hard.

**Hypothesis to weigh:** the dominant lever is #2 (smooth cascade of a deadlock-full
chain vs HW's relay-paced flow), with #1 a secondary ~128-word contribution. If so,
the fix is NOT a terminal-send gate (proven no-op) but the relay/cascade cadence —
i.e. why the EMU's whole chain deadlock-freezes and then releases as one smooth wave,
where HW never fully deadlocks and meters objects out at the relay rate.

## W3 JOINT PROBE RESULT (2026-07-02) — flow-vs-deadlock RESOLVED on HW

Prong A of the HW probe (`build/experiments/pathA-cntr-spike/w3-joint/`): one kernel,
producer(0,2) + consA(0,3) instrumented on a co-reset (shared drain-relative) tile
timer. ConsA stamped 3×/iter to separate the of_j-acquire block from the of_d wait;
producer stamped post-of_src-acquire. Lock sequences bit-identical to `of_q0_lean`
baseline. Ran clean on real NPU (`host/tsc.bin`, `host/tsp.bin`; decode `decode.py`).

**Decisive signal — consA of_j-acquire block (B−A) is FLAT 38cy, every iteration,
pre- and post-drain, to the cycle.** 38cy is pure instrumentation cost (record_ts +
the acquire when the lock is free); it never spikes. **HW's consA NEVER blocks on
of_j.** Whole consA iteration is flat ~146cy across all 32 iters. If HW had the EMU's
~6000cy chain freeze, B−A (or the of_d wait A−prevC) would explode into the thousands.
It does not. **HW flows; it does not deadlock.**

- Producer: reaches iter12 pre-dispatch with a *single* ~1131cy stall at iter7, then
  settles to ~146cy/iter (= consA's relay rate). Flowing equilibrium, never frozen at 2.
- Perturbation cross-check PASSES: W3's 3-stamp load slowed consA 80→146cy/iter vs
  W2's 1-stamp, yet both show 5 smooth pre-dispatch of_j releases, zero block. A 6000cy
  freeze cannot hide behind 66cy of added stamps → the "no deadlock" finding is robust.

**What this pins:** the divergence is **lever #1** — the EMU's memtile of_out MM2S
*acquires and HOLDS* a full j_cons buffer at cy 353 while the shim is un-armed (of_out
fills to 2 → whole chain hard-freezes ~6000cy). HW's of_out MM2S does NOT hold full
buffers pre-arm (of_out stays empty; consA keeps flowing). So the EMU FIX target is the
of_out MM2S's **buffer acquire/read** behavior under terminal backpressure (NOT the
fabric-delivery gate of fix (a), which was correctly a no-op — it gated delivery, not
the acquire). The doc's "does the of_out MM2S over-acquire?" candidate is now the
primary suspect, promoted above the smooth-cascade lever #2.

**Still open (Prong B, next):** the exact mechanism/where the ~1 pre-dispatch object
goes — does the shim S2MM accept stream *ahead of its DDR-drain START_TASK*? consA's
flow proves the memtile freed ≥1 j_cons slot pre-dispatch (so the shim accepted ~1
object's worth), yet the shim starves at +13 (drained only ~13 words at START_TASK).
Reconciling requires the shim's OWN trace: shim South-slave ingress activity vs
DMA_S2MM_0_START_TASK. That is Prong B (shim-side trace, greenlit).

## What would resolve the unknowns (candidate next data)

- **HW of_out/j_cons occupancy at arm** — the #1 unknown. Options: an in-kernel-timer-
  style probe that stamps the memtile relay's lock-handoff timing (the in-kernel timer
  oracle works); or reason it out from the +13 starvation bound (of_out ≤ ~13 words at
  arm ⇒ HW's MM2S is NOT holding 2 full buffers, unlike EMU).
- **EMU: does the of_out MM2S over-acquire?** Test whether gating the MM2S's *buffer
  acquire/read* (not just the fabric drain) on the terminal S2MM being armed keeps
  of_out empty pre-arm — and whether that alone moves the oracle (isolates #1 from #2).
- **EMU cascade rate vs HW relay rate** at cold-start — compare the per-object relay
  latency (producer→of_out end-to-end) EMU vs HW to size #2.

## Code state (to clean up once direction settles)

- Fix (a) terminal-send gate (`routing.rs` `build_mm2s_terminal_map` /
  `walk_to_terminal_s2mm` + the gate in `route_dma_to_tile_switches`, map field on
  `TileArray`): implemented, compiles, `--lib` 3585/0, **no-op on the oracle**. Revert
  or keep-as-faithfulness pending decision.
- `maybe_sp4a_probe` (env `XDNA_EMU_SP4A_PROBE=<max_cycle>`): temporary data probe,
  strip before any landing.
