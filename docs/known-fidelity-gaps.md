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
synthetic / pathological kernels (e.g. our own calibration sweeps). The one
toolchain-derivable member (task queue depth) is now **fixed**; the two
residual gaps both need **HW-empirical data** the open-source toolchain does not
expose, so they stay documented-and-deferred rather than guessed.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| TCT token buffer | we model **unbounded**; HW has finite token backpressure that stalls the channel (`Stalled_TCT`) | `src/device/dma/token.rs` (`TokenState`; the deliberate simplification is called out in the doc comment) | **OPEN.** Documented in code. aie-rt exposes only a 1-bit `STALLED_TCT` flag, **no numeric depth** -- faithful finite modeling needs an empirically-measured HW depth (HW-gated; a guessed depth would violate derive-from-toolchain). |
| BD reuse-while-live / BD-pool over-allocation | we re-parse a BD on reuse and complete; HW wedges when a kernel needs > 16 distinct shim BDs (in-flight BD reuse) | [finding: 2026-06-06 shim BD-pool over-allocation](superpowers/findings/2026-06-06-shim-bd-pool-overallocation-nonmonotonic-wedge.md) | **OPEN (won't-fix).** The HW wedge is *non-monotonic* in K (k8 pass, k9 wedge, k12 pass, k16 wedge) -- no clean model. Generator (`gen-shim-chain-sweep.py`) capped at K=8 instead. |

A second, unrelated cluster: **trace held-level encoding**. After the held-level
re-architecture (LOCK_STALL / PORT_* / DMA stall-starvation / core stall family
emit as B..E spans, decoded by upstream `parse_trace`), the span *structure* is
faithful but two residual gaps remain. Both are **timing/encoding artifacts, not
mechanism bugs**, and both are stable Phoenix->Strix (DMA delivery timing, not
silicon-version-specific). Measured on `vec_mul_trace_distribute_lateral` (peano,
interpreter backend) vs real NPU1.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| Held-level falling edge, close-to-empty | a held span whose deassert lands on a cycle with no *other* concurrent frame cannot encode its falling edge in mode-0 (no empty frame); the close defers to the next frame / end-of-segment. HW closes the span at the deassert cycle. Visible as the mem-trace `DMA_S2MM_*_STREAM_STARVATION` span staying open to EOS. | `src/device/trace_unit/mod.rs` (`set_event_level` / `commit_pending_frame`, the `active==0` defer); [finding: 2026-06-08 falling-edge depends on concurrent levels](superpowers/findings/2026-06-08-lock-stall-falling-edge-depends-on-concurrent-levels.md) | **DOCUMENTED.** Matches how HW closes a level span (no synthetic empty frame); the residual is only the *lone-level* case where no concurrent frame carries the close. With the per-cycle TRUE metronome the deferral is bounded to ~1 cycle. |
| Held-level count under-emission (DDR-fill / bursty delivery) | our stream/DMA delivery is *smoother* (more continuous) than HW's bursty ~1024-cycle bursts, so level events assert fewer times: `LOCK_STALL` ~11 vs 19 spans, `PORT_RUNNING_0` 3 vs 6, `DMA_S2MM_*_STREAM_STARVATION` 2 vs 4. (Counts are span B-phases; the held mechanism itself fires correctly -- see the single 943-cycle hold in the finding.) | DMA/stream delivery timing model; [finding: 2026-06-07 LOCK_STALL over-emission](superpowers/findings/2026-06-07-lock-stall-overemission-interp-vs-hw.md), [finding: 2026-06-08 falling-edge](superpowers/findings/2026-06-08-lock-stall-falling-edge-depends-on-concurrent-levels.md) | **DOCUMENTED.** Held-level re-architecture fixed the gross ~375x LOCK_STALL over-emission; the residual count gap is a DMA fill-latency/burstiness axis, the same axis as the aiesim "trace stall/starvation micro-timing" row below. Calibrate against HW, not chase in the trace encoder. **UPDATE (2026-06-15, #140) -- SUPERSEDED, see 2026-06-16.** The "shim host-DDR burst-delivery jitter" narrative and the seeded-PRNG `AIE2_DDR_PHOENIX` model that "band-matched within 1 sigma" were a **metric artifact**: the cadence tool counted trace *frame-records* (the re-checkpoint frames emitted at every concurrent edge), not *spans*, manufacturing run-to-run variance that does not exist on silicon. **CORRECTED (2026-06-16, commit `ace4c321`+):** measured span-based (`tools/port-span-cadence.py`, oracle B/E, idle-gap>2), NPU1 `add_one_using_dma` `PORT_RUNNING` is **deterministic** (std 0, 15 HW runs): HW `[slot0=1, slot1=5, slot4=3, slot5=4]`. Default EMU (phoenix removed) is `[1, 6, 5, 4]` -- **slot0 and slot5 (the shim-DDR-gated ports) match exactly**; only slot1 (+1) and slot4 (+2), the two **compute-core-gated** ports, diverge. So this is **not** a DDR-delivery gap at all -- it is **compute-core buffer release/acquire phasing**: EMU fragments HW's long opening `PORT_RUNNING` burst into extra short sub-bursts (the core releases/acquires a buffer slightly later than silicon, so the small stream FIFO drains and idles >2cy before the next buffer is ready). Steady-state cadence/gaps already match (~50-56cy). The stochastic DDR (`AIE2_DDR_PHOENIX`) model + its env vars are **removed** (`burst.rs` deleted) -- it modeled a ghost and made slot0 *worse* (2 vs 1). #140 reframed to core-release-phasing (overlaps #135/#139). Spec `superpowers/specs/2026-06-15-ddr-stochastic-delivery-jitter.md` is superseded; finding: `superpowers/findings/2026-06-16-port-cadence-metric-was-frame-records.md`. |

The core stall family conversion (MEMORY_STALL / STREAM_STALL / CASCADE_STALL as
held levels) is **not exercised by `distribute_lateral`** -- its trace config does
not include events 23/24/25 -- so those three have unit coverage but **no HW-trace
validation yet**; pending a cascade/stream-stall kernel capture. Two related
modeling notes (not HW-confirmed disagreements, hence not rows above): mode-1
(EventPc) PC anchoring is dropped for stall held levels (the level path carries no
PC, matching the LOCK_STALL precedent), and the executor-level *intra-bundle*
structural bank conflict stays a bounded MEMORY_STALL pulse (it resolves within one
executor step, so a same-step rising+falling would encode as a zero-width span).

A third cluster: **vector-register result visibility (forwarding network)**. The
AIE2 per-operand bypass/forwarding matrix is implemented and validated against
the chess bridge sweep (89/89 PASS, 0 regressions) **and against real Phoenix
(NPU1) silicon**: `vec_mac_bf16`, `two_col`, and all three
`matrix_multiplication_using_cascade` variants (plain/buffer/cascade) PASS
HW==EMU on the bridge (2026-06-09), with `vec_mac_bf16` and `two_col` also
diffing CLEAN on the trace comparison. The W/X-file forwarding model is therefore
HW-verified for the bf16 matmul class; the remaining gaps below are scoped
deferrals, not unvalidated model.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| Accumulator/CM-domain (`VEC_Bypass`) result visibility | not in the per-operand bypass matrix; modeled via the separate MAC-pipeline-latency path (`ExecutionContext::queue_matmul_accum_write`). The matrix resolves W/X-file results; `VEC_Bypass` (MAC/MUL results into the accumulator file) is deferred. | `src/interpreter/state/registers.rs` (`VectorRegisterFile::visible_at`); `src/interpreter/state/context.rs` (`queue_matmul_accum_write`, carries a `FIXME(bypass-model)`) | **OPEN / deferred.** The MAC-latency path is HW-validated *functionally* — all three `matrix_multiplication_using_cascade` variants PASS HW==EMU on the bridge (2026-06-09), which exercises the accumulator-write path — but not yet at trace granularity (those kernels' TRACE PREP fails for an orthogonal trace-injection reason). Accumulator folding into the per-operand bypass matrix is a noted future extension (see `docs/superpowers/plans/2026-06-09-vector-write-result-latency.md` Scope). **Partially advanced 2026-06-10 (`430d841`):** the `VMOV bml,x` (wide vector->accumulator move) write is now deferred by its def latency via the same `queue_matmul_accum_write` path, after the phase-B `vec_conv_bf16_edge` silicon kernels proved an immediate write corrupts delay-slot `VST.CONV` stores (read freshly-overwritten accumulator -> Inf/NaN garbage). All 10 convert modes EMU==silicon. The full bypass-matrix fold (distinguishing `VEC_Bypass` MAC->MAC from `NoBypass` store consumers) is still the open extension. |
| `source_forward` positional alignment for merged memory operands | `SlotOp::source_forward` aligns exactly for compute ops; may drift for fused memory operands that merge address/data into a single TableGen operand list position | `src/interpreter/bundle/slot.rs` (`FIXME(source-forward-memory-alignment)`); `src/interpreter/decode/operand_extraction.rs` | **DOCUMENTED / benign.** The bypass model only reads vector-register compute sources; memory operands use `NoBypass` or are not vector-reg reads, so this positional drift does not affect result visibility decisions. |

A fourth cluster: **`vadd.f`/`vsub.f` special-value NaN payload** (bf16 *and*
fp32). **RESOLVED 2026-06-11** -- full mechanism and evidence in
[`docs/superpowers/specs/2026-06-11-nan-payload-datapath-regime.md`](superpowers/specs/2026-06-11-nan-payload-datapath-regime.md).

The result NaN payload when an operand is a NaN has **two real silicon
regimes**: a **datapath regime** (dominant) where the payload is the actual
exp-255 mantissa-datapath sum `r`, and a **canonical regime** (rare) that forces
payload 1. The regime is residual float-unit HW state -- global across lanes,
deterministic within a session, but it varies across sessions and survives a
driver reload. The *same dense binary* gave canonical 0xFF81 for all 127 Inf+NaN
payloads in the 12:40 HW capture and datapath payloads (120/127 different) in a
later session. So this is **not** bf16-specific, **not** two-NaN-operand-only,
and **not** in the bf16 narrow -- it is the shared `aie2_acc_fp32_add` datapath
(fp32 shows it identically), and `+Inf + -NaN` is exactly the trigger. The
earlier "two-exp-255-operand significand arithmetic, bf16-only" framing was
wrong; seed 6159 lane 29 is `+Inf + -NaN(0x46)`, not a NaN+NaN pair.

The model now produces the **datapath regime** (Maya's call: it is what the ALU
computes; canonical is a residual-state suppression). One change in
`aie2_acc_fp32_add`: `use_r = (a_nan || b_nan) && !overflow` lets the datapath
sum `r` through instead of zeroing it. Validated **8160/8160** against fresh
dominant-regime NPU1 captures (bf16+fp32, add+sub); `cargo test --lib` 3408/3408;
seed 6159 via EMU now reproduces `0xFF8C`. Seeds 6159/6258 are thereby
re-credited.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| special-value NaN payload: datapath vs canonical regime | HW has two residual-state regimes for `vadd.f`/`vsub.f` NaN-input payloads; we now model the dominant **datapath** regime (payload = exp-255 mantissa sum `r`). HW in the rare **canonical** regime forces payload 1 and would mismatch the payload bits (a HW-state artifact, not a model bug). | `src/interpreter/execute/vector_float.rs` (`aie2_acc_fp32_add`, `use_r` gate). | **RESOLVED (datapath regime) 2026-06-11.** Validated 8160/8160 vs dominant-regime HW. Residual open: *what* flips the regime is uncharacterized (not driver reload); low priority since the payload is functionally dead. See the design note. |

---

## aiesim oracle (AMD's proprietary simulator)

Gaps in the *oracle*, not in our emulator. They bound what aiesim can validate.
Notably, in the first case **our interpreter is the more faithful of the two.**

| Gap | What | Where | Status / rationale |
|-----|------|-------|--------------------|
| AIE2 compute-to-compute shared memory | aiesim does not model neighbour shared-memory / lock handoff between adjacent compute tiles; such kernels deadlock in the sim. Real HW + our interpreter run them. | [finding: 2026-06-06 aiesim AIE2 c2c shared memory](superpowers/findings/2026-06-06-aiesim-aie2-cross-core-shared-memory-limitation.md); [`aiesimulator.md`](aiesimulator.md) | **DOCUMENTED.** AMD's own test suite marks this `XFAIL`. Exclude this kernel class from the aiesim oracle; our interpreter is the oracle here. |
| Control-read response aliasing | aiesim's cluster model reused one control block across all beats of a control-packet read response, corrupting the header -> never route-matched -> S2MM starved | [`aiesimulator.md`](aiesimulator.md) (Known Issues) | **RESOLVED locally** via a distinct-object-per-beat clone patch (`ss_probe`, gated by `XDNA_CLONE_BEATS`). Local-only; never shipped. |
| Trace stall/starvation micro-timing | aiesim trace counts of *timing-sensitive* events diverge from HW on `vec_mul_trace_distribute_lateral` (`LOCK_STALL` 3 vs 19, `DMA_S2MM_0_STREAM_STARVATION` 3 vs 2, `PORT_RUNNING_1` 4 vs 5). Per-event timeline analysis decomposes the gap into **three separable sources** (not one blanket "approximate"): **(1) Compute region is cycle-EXACT** -- the user trace markers `INSTR_EVENT_0`->`INSTR_EVENT_1` span **12297 ns in both worlds, all 4 invocations**, to the ns. **(2) DMA/DDR input fill latency is under-modeled** -- first `EVENT_0` at HW 8326 ns vs aiesim 2195 ns (~6131 ns optimistic); it is a **constant offset, not per-iteration drift** (each invocation stays exactly 12297 wide). **(3) The `LOCK_STALL` count gap is the HW per-lock-transaction arbitration cycle** -- both worlds issue the *identical* lock ops (`LOCK_ACQUIRE_REQ`=9/9, `LOCK_RELEASE_REQ`=9/9), but HW trails **every** request with a 1 ns `LOCK_STALL` pulse (uncontended lock still costs 1 arbitration cycle) while aiesim models uncontended acquires as zero-stall. HW 19 = 1 genuine startup wait + 18 per-transaction pulses; aiesim 3 = genuine blocking waits only. | trace bridge commit `7d93a83`; HW-vs-aiesim comparison artifacts in local `build/experiments/bcast-bridge/` (FINDINGS.md, trace_{aiesim,hw}.json) | **DOCUMENTED / expected.** From aiesim's ISS, not our trace bridge. **Calibration rule** (which knobs aiesim is safe to teach the emulator): compute-region cycles -> **safe** (exact); DMA/DDR fill latency and per-lock-op stall cost -> **calibrate against HW only** (aiesim is optimistic/zero there). aiesim is a faithful oracle for trace *structure/flow* and compute-region timing, not for fill latency or lock-arbitration overhead. **UPDATE (2026-06-14, commit `737f5505`):** the per-lock-transaction arbitration cost (source 3) is now MODELED in core *cycle* timing -- every completed acquire/release charges +1 cycle (`cycle_accurate.rs`), calibrated to HW (tenant-4 core loop 2125->2129 = HW exact; full bridge corpus clean, 0 trace divergence). What remains open is the *trace-event* half: emitting the per-transaction `LOCK_STALL` pulse on the uncontended path (the `LOCK_STALL` count gap of this row and the "Held-level count under-emission" row above). Cycle cost closed; trace-pulse emission pending. |

---

## Maintenance

When a new confirmed HW-disagreement surfaces: add a one-line row here pointing
at the detailed finding, and note whether it's fixed, documented-and-deferred,
or won't-fix. Keep this page an index -- detail lives in the linked findings.
