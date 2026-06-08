# Known Fidelity Gaps

A single registry of **confirmed** points where the emulator (or the aiesim
oracle) provably disagrees with real NPU hardware. Each is documented in detail
elsewhere; this page is the index, so the pattern is visible in one place and we
don't re-investigate the same gap twice. (That happened once already: a stale
"shim queue is 8-deep" caveat sent a whole session chasing a dead mechanism.)

**Scope:** confirmed HW-disagreement gaps only -- behaviors where we have ground
truth that the model is wrong. This is *not* the list of not-yet-implemented
features (vector compute semantics, stream-switch per-port types, micro-timing);
those live in [`toolchain-sources.md`](toolchain-sources.md) and the
[roadmap](../ROADMAP.md).

**How these surface:** mostly via the aiesim oracle (the XRT-plugin -> aiesim
path). The count rising is the oracle doing its job -- before it, these were
invisible. Each was a deliberate "document, don't chase" call; see the per-gap
rationale.

---

## Emulator (xdna-emu interpreter)

These cluster in **one subsystem: shim DMA finite-resource modeling**. Shared
root posture: our DMA model is *optimistic where silicon is strict* -- it
assumes generous/infinite resources (deep queues, unbounded token buffers,
freely reusable BDs) where the hardware is finite and will stall or wedge. All
are **low real-workload impact**: real compilers don't hit these; they bite
synthetic / pathological kernels (e.g. our own calibration sweeps). If ever
worth closing, they're plausibly **one "finite shim DMA" pass**, not three
separate hunts.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| Shim task queue depth | we model **8**, HW is **4** (aie-rt `XAIE_DMA_MAX_QUEUE_SIZE 4U` / `StartQSizeMax = 4U`) | `src/device/dma/token.rs` (`MAX_TASK_QUEUE_DEPTH`) | **OPEN.** A 1-line correction, but it changes queue behavior / cycle counts broadly -- blast radius with no current payoff (it does not by itself reproduce any known HW wedge; our peak occupancy is 1-2). Batch into a finite-DMA pass, don't do ad-hoc. |
| TCT token buffer | we model **unbounded**; HW has finite token backpressure that stalls the channel (`Stalled_TCT`) | `src/device/dma/token.rs` (`TokenState`; the deliberate simplification is called out in the doc comment) | **OPEN.** Documented in code. No numeric depth in the toolchain (single stall bit only), so faithful modeling needs empirical depth. |
| BD reuse-while-live / BD-pool over-allocation | we re-parse a BD on reuse and complete; HW wedges when a kernel needs > 16 distinct shim BDs (in-flight BD reuse) | [finding: 2026-06-06 shim BD-pool over-allocation](superpowers/findings/2026-06-06-shim-bd-pool-overallocation-nonmonotonic-wedge.md) | **OPEN (won't-fix).** The HW wedge is *non-monotonic* in K (k8 pass, k9 wedge, k12 pass, k16 wedge) -- no clean model. Generator (`gen-shim-chain-sweep.py`) capped at K=8 instead. |

---

## aiesim oracle (AMD's proprietary simulator)

Gaps in the *oracle*, not in our emulator. They bound what aiesim can validate.
Notably, in the first case **our interpreter is the more faithful of the two.**

| Gap | What | Where | Status / rationale |
|-----|------|-------|--------------------|
| AIE2 compute-to-compute shared memory | aiesim does not model neighbour shared-memory / lock handoff between adjacent compute tiles; such kernels deadlock in the sim. Real HW + our interpreter run them. | [finding: 2026-06-06 aiesim AIE2 c2c shared memory](superpowers/findings/2026-06-06-aiesim-aie2-cross-core-shared-memory-limitation.md); [`aiesimulator.md`](aiesimulator.md) | **DOCUMENTED.** AMD's own test suite marks this `XFAIL`. Exclude this kernel class from the aiesim oracle; our interpreter is the oracle here. |
| Control-read response aliasing | aiesim's cluster model reused one control block across all beats of a control-packet read response, corrupting the header -> never route-matched -> S2MM starved | [`aiesimulator.md`](aiesimulator.md) (Known Issues) | **RESOLVED locally** via a distinct-object-per-beat clone patch (`ss_probe`, gated by `XDNA_CLONE_BEATS`). Local-only; never shipped. |
| Trace stall/starvation micro-timing | aiesim trace counts of *timing-sensitive* events diverge from HW: on `vec_mul_trace_distribute_lateral`, `LOCK_STALL` 3 (aiesim) vs 19 (HW), `DMA_S2MM_0_STREAM_STARVATION` 3 vs 2, `PORT_RUNNING_1` 4 vs 5. Deterministic events match exactly (INSTR_EVENT_0/1=4, lock acq/rel=9, PORT_RUNNING_0=6, 4 invocations, 12297 cycles) and the deterministic event *order* is byte-identical. | trace bridge commit `7d93a83`; HW-vs-aiesim comparison artifacts in local `build/experiments/bcast-bridge/` (FINDINGS.md, trace_{aiesim,hw}.json) | **DOCUMENTED / expected.** The cycle-approximate ISS under-models memory/DMA stall latency; from aiesim's ISS, not our trace bridge. aiesim is a faithful oracle for trace *structure/flow*, not stall micro-counts. Matches the ballpark-deterministic calibration stance. |

---

## Maintenance

When a new confirmed HW-disagreement surfaces: add a one-line row here pointing
at the detailed finding, and note whether it's fixed, documented-and-deferred,
or won't-fix. Keep this page an index -- detail lives in the linked findings.
