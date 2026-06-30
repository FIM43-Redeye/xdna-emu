# Cold-start head-start: HW cores warm before the trace baseline; EMU captures cold

**Date:** 2026-06-29  **Issue:** #140 (timer-sync arc, SP-4a)
**Status:** PARTIALLY FALSIFIED -- the head-start is REAL but is NOT the cause of
the SP-4a offset (empirically disproven, see correction below). The actual offset
lever is the **fill-cadence / consumer-pacing residual** (known-fidelity-gaps
row 51).
**Branch:** `sp3-validation-kernel`.
**Working notes + raw evidence:** `build/experiments/sp3-spike-trace/lean/`
(decoded HW + EMU traces, 5 stable HW runs, the cold-start sim probe).

> **CORRECTION (2026-06-29, same session -- the causal claim below was tested and
> FALSIFIED).** This doc concluded the cross-domain offset (-52 EMU vs +2 HW) is
> caused by the missing head-start. That is WRONG. Added a configurable warm-up
> floor (`XDNA_EMU_CORE_HEADSTART`, FFI `backend.rs`) and swept N = 0 .. 2000cy:
> the trace baseline genuinely shifts later (total cycles 117556 -> 119449, shim
> base soc 6779 -> 8672, so the cores DO warm longer) but the prod->consA offset
> stays **pinned at -52** for every N. So the head-start does not move the offset.
> Everything in "What the gap is NOT" + the head-start measurement table below is
> still TRUE (HW cores really do run ~795-1068cy before the baseline, 6 runs) --
> but that is a real, ORTHOGONAL behavior, not the offset's cause.
>
> **The actual cause (re-confirmed):** the pipeline FILL-STATE at the baseline.
> The EMU producer reaches the baseline with FREE buffers (its MM2S drained them
> downstream; the pipeline blocks downstream, not at the producer), so after the
> reset+drain cascade it re-fills 2 buffers before stalling -> +52 after consA.
> HW's producer is coupled to one buffer / blocked at the baseline, so it stalls
> ~immediately with consA -> +2. This IS known-fidelity-gaps **row 51** (send-port
> `[16,16,16,16]` EMU vs `[8,8,14]` HW = same producer-fills-both vs
> coupled-to-one fill-cadence). The lever is the consumer-pacing / drain-rate /
> buffer-coupling axis (device-model audit), NOT cold-start sequencing.

This finding **corrects** the prior SP-4a root-cause framing (the
`build/experiments/sp3-spike-trace/SP4A-LEAN-FINDING.md` "memtile relay cadence"
conclusion). It also **characterizes** the open residual in known-fidelity-gaps
**row 51** (the send-cadence "cold-start transient... needs an HW capture of the
first-buffer lock-handoff timing") -- they are the same root cause, seen from
two observables.

## The substrate

The lean SP-4a kernel `of_q0_lean.py` (Q=0, pure lock/DMA handshake, no compute):
ProdCore(0,2) -> MemTile(0,1) -> ConsA(0,3) objectfifo chain, REPS=32, OBJ=64.
The SP-4a oracle is the within-column cross-domain pair: prod -> consA first
`LOCK_STALL` offset, HW **+2** (range 0 across 20 runs), EMU **-52**.

## What the gap is NOT (each ruled out with evidence)

1. **Not a per-tile clock stagger.** A cold-start sim probe (env-gated
   `XDNA_EMU_COLDPROBE` in `coordinator.rs` step loop) shows both cores execute
   in lockstep from sim-cycle 0 -- identical PC every cycle through the prologue.
   The clock ungate is per-column (`dispatch.rs` `assign_partition_columns`), so
   all compute cores in a column become active the same cycle. No stagger.

2. **Not an instruction-timing / VLIW-bundling gap.** Producer loop period: HW
   ~69 trace-ticks/iter, EMU ~67. ConsA: HW 67, EMU 69. The loops run at HW speed.

3. **Not a producer free-buffer lock bug -- the EMU is toolchain-faithful.**
   The lowered MLIR sets `src_prod_lock` (producer free-slot lock) **init = 2**,
   set in the CDO before core-enable and **never reset** by any DMA-config or
   channel-start path (aie-rt: DMA setup writes only BD/channel-control
   registers, never lock *value* registers; the only lock-value writer is
   `XAie_LockSetValue` in `initLocks`, which runs before core-enable). The
   producer core lowers to `acquire_ge(prod_lock, 1)` -- immediately satisfiable
   against 2. So per the toolchain the producer **should** acquire 2 buffers at
   cold-start without stalling, which is exactly what the EMU does. (Derivation:
   `mlir-aie/lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp`
   :422-451 lock init, :1628-1641 acquire lock/value; `aie-rt/.../dma/xaie_dma.c`
   channel start touches only the Enable bit; `mlir-aie/lib/Targets/AIERT.cpp`
   :533-557 `initLocks` ordering.)

## What the gap IS: HW cores warm before the trace baseline; EMU captures cold

PERF_CNT_2 fires every 1024 core-clocks, so its first occurrence back-dates each
core's counter start relative to the shim trace base (the shim
`DMA_S2MM_0_START_TASK`). Across **5 stable HW runs + the main run** (consistent
to +/-25):

| | producer counter-start (rel. trace base) | consA counter-start | first-LOCK_STALL offset (consA - prod) |
|---|---|---|---|
| **HW** | **-774 ... -822** (BEFORE the base) | **-561 ... -602** | **+2 every run** |
| **EMU** | **+581** (AFTER the base) | +573 | **-52** |

So on HW the cores have already executed ~795cy (producer) / ~580cy (consA)
**before** the trace/timer baseline is established; by the time the trace starts
they are warm and in steady-state phase-lock (+2). On the EMU the cores start
~580cy **after** the baseline -- the trace captures the cold-start transient
(producer fills its 2 free buffers and stalls on its 3rd acquire ~52cy after
consA's 1st-acquire stall) -> -52.

Two HW behaviors the EMU lacks, both feeding the trace-baseline phase:
- the **~795cy CDO-config -> runtime-sequence head-start** (cores enabled at CDO
  time run freely; the runtime sequence that starts the shim DMA / trace baseline
  is dispatched ~795cy later);
- a **~215cy producer-before-consumer core-start stagger** (HW prod -795 vs consA
  -580); the EMU starts both cores in lockstep.

## The +2 is a phase-locked sample, not a steady-state invariant

Sweeping a hypothetical trace-start time T and computing `(first consA stall >= T)
- (first prod stall >= T)` shows the offset is **noisy (+/-~30)**, not a stable
constant, in BOTH worlds. HW reproduces +2 (range 0 across 20 runs) because its
trace-start event is deterministically phase-locked to the dataflow **at a warm
point**; the EMU samples -52 because its trace-start lands **cold** and at a
different dataflow phase. So "byte-match the raw cross-domain offset" is
reproducing a phase-locked sample on a +/-30 jitter floor, not validating a
within-domain timing invariant. SP-4a's premise (within-domain exactness =>
cross-domain +2) does not hold the way it was framed.

## Unification with known-fidelity-gaps row 51 (send-port cadence)

Row 51's open residual is the same mechanism from the send-port observable:
EMU `PORT_RUNNING_4 = [16,16,16,16]` (producer empties the whole transfer before
any consumer stall) vs HW `[8,8,14,...]` (consumer-paced). That is the **cold**
opening (producer fills both ping-pong buffers, un-paced) vs the **warm**
steady-state (consumer stalling regularly, pacing the producer). The head-start
explains both observables. Testable prediction: warming the EMU cores before the
trace baseline should shift the send port toward `[8,8,14,...]` AND the
cross-domain offset toward +2.

## Resume: build the head-start / trace-baseline model

The fix is to model the CDO-config -> runtime-sequence head-start (~795cy, a
HW-measured constant; SP-5 flavor) so the EMU's cores warm before the trace
baseline is established, as on HW -- and likely the ~215cy producer-first
core-start stagger. This is cold-start sequencing that affects every traced
kernel, so it is HW-calibrated territory: gate any change on the offset moving
-52 -> +2 (lean oracle), the send port moving toward HW (row 51 oracle),
`cargo test --lib`, and a bridge-corpus spot-check for no regression.

Evidence on disk: `build/experiments/sp3-spike-trace/lean/{hw,emu,hw_stab/run_*}/
events.json`; the cold-start sim probe output and the analysis scripts used for
the tables above.

## SPAN-CADENCE CONFIRMATION (2026-06-29, session 2): faithful steady-state, real cold-start over-fill

To settle "trace-phase artifact vs real dataflow gap" (Maya: confirm first), the
existing lean `trace.bin` (HW + EMU) was re-decoded to **B/E perfetto spans**
(`parse-trace.py --out-perfetto`; the lean build survives at the ABSOLUTE path
`/home/triple/npu-work/mlir-aie/test/npu-xrt/spike_bringup/build_q0_lean_trace/`
-- note the xdna-emu-relative `mlir-aie/` is a stray near-empty tree). The
perfetto names landed empty but track identity survives: **pid = tile, tid =
port slot**, so held-level spans reconstruct directly from the B/E pairs (this
is the correct unit -- immune to the re-checkpoint frame-record contamination
that makes raw `events.json` PORT_RUNNING event-timing useless for cadence).

Memtile (pid for `memtile(1,1)`) port sub-burst structure (idle-gap>2). Slot
directions decoded from the trace `Stream_Switch_Event_Port_Selection_{0,1}`
registers + the `aie.flow` table + span-count cross-check (of_out drains 32
objects -> slot5 has exactly 32 spans): slots 0/1 are MASTER ports = memtile
S2MM (recv); slots 4/5 are slave ports = memtile MM2S (send):

| port slot | dir | objectfifo (path) | HW | EMU | verdict |
|-----------|-----|-------------------|-----|-----|---------|
| slot0 | recv S2MM0 | of_src prod->memtile (**forward**) | 43 spans `[2,2,...]` | 47 `[2,2,...]` | **identical** |
| slot4 | send MM2S0 | of_d memtile->consA (**forward**) | 49 spans `[2,2,...]` | 51 `[2,2,...]` | **identical** |
| slot1 | recv S2MM1 | of_j consA->memtile (**return**) | 32 spans, first burst **11** | 30, first **17** | steady identical, **cold-start differs** |
| slot5 | send MM2S1 | of_out memtile->**shim drain** (**drain**) | 32 spans, first **14** | 29, first **19** | steady identical, **cold-start differs** |

**The divergence is entirely on the RETURN/DRAIN path.** The forward spine
prod->memtile->consA (slots 0+4) is bit-faithful in BOTH cold-start and
steady-state. Only of_j (consA->memtile) and of_out (memtile->shim DDR drain)
differ, and only at cold-start. The 228cy smoking-gun span is of_out -- the
shim drain.

Initial send-port span timeline (begin relative to baseline, duration), the
decisive view:
- **HW slot5:** `(1103, 48) (1152, 64) (1217, 64) ...` -- gradual ramp into 64cy spans.
- **EMU slot5:** `(534, 228) (763, 64) (828, 64) ...` -- one **228cy** initial dump, then 64cy.
- **HW slot1:** `(1129, 6) (1136, 8) (1145, 34) (1180, 64) ...` -- ramps 6->8->34->64.
- **EMU slot1:** `(569, 48) (618, 64) ...` -- jumps straight to full bursts.

The earliest span begins **exactly at the baseline ts in BOTH worlds** (HW
426455, EMU 6780) -- so HW's trace is NOT truncating a pre-baseline burst. The
HW gradual ramp is genuine; the EMU 228cy initial dump is genuine.

**Verdict -- the fork is not binary, it splits cleanly by phase:**
1. **Steady-state dataflow is FAITHFUL** (case B). Both worlds settle to one
   64cy send span per ~65cy; recv ports (slots 0/4) are structurally identical.
   The EMU dataflow RATE is correct. Changing steady-state ingress depth (the
   broad `accept_awaiting_drain` form) would corrupt a correct model.
2. **Cold-start initial burst is a REAL over-fill** (case A, CONFINED to the
   opening) -- and **localized to the DRAIN path, NOT producer ingress**. The
   228cy dump is slot5 = of_out = the memtile->shim DDR drain (register-decoded,
   span-count-verified). The producer ingress path (of_src, slot0) is faithful.

**Localization REDIRECTS the fix away from `accept_awaiting_drain`.** That gate
is on the producer ingress / forward path, which this trace proves is faithful
(slots 0+4 identical). The resume hypothesis aimed at the wrong mechanism. The
real lever is **shim-drain cold-start pacing**: EMU's memtile dumps ~3.5 objects
to the shim in one 228cy burst because the EMU shim S2MM does not backpressure
the drain at cold-start the way HW does (HW ramps gradually). This is the
DDR-drain side (cf. the shim S2MM cold-start `s2mm:341` knob and the row-51
`788e3d70` "S2MM drains at memory-bus rate" fix -- which addressed the COMPUTE
recv, not the SHIM drain).

**Before/after oracle (any drain-pacing fix must satisfy ALL):** forward slots
0+4 stay `[2,2,...]`; return/drain steady-state stays 64cy@~65cy; only the
cold-start initial burst moves (slot5 228->~64; slot1 17->~11). Perfetto +
reconstruction artifacts at
`build/experiments/sp3-spike-trace/lean/{hw,emu}/span.perfetto.json`.

> **WARM-UP FALSIFIED (2026-06-29, session 2, same day).** The "warm-up
> pre-fills" conclusion below was TESTED via a `XDNA_EMU_WARMUP_CAP` knob (caps
> the warm-up loop; cap=0 = no warm-up) and FALSIFIED: cap=0 on the lean EMU is
> BYTE-IDENTICAL to default (offset still -52, shim starvation still t+1684,
> of_out still a 228cy dump). Total run cycles dropped by ~107 (the skipped
> warm-up) but the TRACE is unchanged. So the warm-up is NOT what fills the
> pipeline. The REAL mechanism (root-cause log dump): all tile DMAs + cores
> start at **cy=0** (CDO config), but the shim drain (tile 1,0 ch0 S2MM, the
> of_out sink) does not `Idle->BdSetup` until **cy=6672** -> `Transferring`
> cy=7202, because the NPU executor processes ~39 of 42 runtime-sequence
> instructions (firmware latency) before the `dma_wait`/shim trigger. That
> **6672cy window of cores-running-with-no-drain** is what fills the pipeline to
> deadlock. HW by contrast is EMPTY at the shim start: its prod engages at t+22
> (AFTER the baseline) and the first of_out object only reaches the shim at
> t+1103 -- i.e. HW's cores engage ~WITH the shim drain, pipeline fills
> concurrently with draining. **Refined candidate root cause: the EMU runs core
> execution + inter-tile dataflow from cy=0, but on HW (this Q=0 trace) the
> cores effectively engage at the runtime-sequence/shim-start, so EMU pre-fills
> while HW starts empty.** This is in TENSION with the warm-up's stated purpose
> (cores must init before NPU mem writes) -- resolving when cores SHOULD start
> relative to the runtime sequence is the open design question. Evidence:
> `build/experiments/sp3-spike-trace/lean/emu_cap0/` (run.log shim-start at
> cy=6672; identical trace metrics).

## ROOT CAUSE CONFIRMED (2026-06-29, session 2): the warm-up pre-fills the pipeline

The coupling check (does the drain divergence move the prod->consA offset?)
RESOLVED: yes -- they are the SAME bug. Cold-start unblock cascade + the
decisive shim-starvation signal:

| signal | HW | EMU |
|--------|-----|-----|
| first shim S2MM STREAM_STARVATION | **t+14** (then every ~65cy, 50 total) | **t+1684** (28 total) |
| pipeline state at trace baseline | **EMPTY** (shim instantly starves) | **FULL** (shim drains backlog ~1684cy before starving) |
| cold-start wake order | shim-starve -> prod stall t+22 -> consA t+24 (**prod-first, +2**) | memtile drain t+534 -> consA t+594 -> prod t+646 (**prod-last, -52**) |

**Single root cause:** the EMU warm-up loop (`backend.rs`, runs the engine until
`all_cores_blocked` BEFORE processing the NPU runtime-sequence that sets the
trace baseline) pre-fills the whole pipeline to deadlock. So at the baseline the
EMU pipeline is FULL: the shim drains a backlog (no starvation until t+1684) and
the cores wake by BACKWARD drain-cascade (memtile -> consA -> prod), making prod
stall LAST (-52). On HW the pipeline is EMPTY at the baseline: the shim instantly
starves (t+14) and the cores engage from empty with prod FIRST (+2). The same
full-pipeline state also produces the 228cy of_out drain dump (draining the
pre-filled backlog). ONE cause, BOTH observables (the offset AND the drain dump).

**This is why the head-start sweep failed -- it tested the WRONG DIRECTION.**
Extending the warm-up floor keeps the pipeline full (cannot over-fill a
deadlock), so the offset stayed -52 for all N. The fix is the OPPOSITE: make the
EMU pipeline EMPTY at the baseline (as HW is), i.e. do NOT pre-fill via warm-up
to deadlock before the trace baseline.

**Decisive confirmation test (next):** reduce/suppress the warm-up pre-fill and
re-run the lean EMU. Predicted: shim starvation appears at ~t+14 (not t+1684),
prod->consA first-LOCK_STALL moves -52 -> +2, and the of_out initial span
shrinks 228 -> ~64 (gradual ramp). Caveat: the warm-up exists for a reason
(other kernels' init ordering) -- the test establishes the root cause; the real
fix must preserve whatever the warm-up was protecting. Understand the warm-up's
purpose (git-blame `backend.rs` warm-up loop) before shipping a change.

## TOOLCHAIN VERDICT + REFINED HYPOTHESIS (2026-06-29, session 2, end-of-session)

NOTE: the "warm-up pre-fills" framing two sections up is SUPERSEDED (warm-up
falsified, see the WARM-UP FALSIFIED quote-block). This is the current state.

**Option 1 (derive from toolchain, Explore agent) -- DONE, authoritative.** For
an AIE2 objectfifo kernel: (a) CORE ENABLE is emitted in the **CDO**
(`mlir-aie AIERT.cpp:815-824 addCoreEnable`, default-on via `enable_cores=true`),
so cores execute their ELF from context-creation. (b) TILE + MEMTILE DMA
CHANNELS are **CDO**-started, not just configured: `AIERT.cpp:494-510
pushToBdQueueAndEnable` emits `XAie_DmaChannelSetStartQueue` + `XAie_DmaChannelEnable`
(live channel-enable bit) inside `addInitConfig`. (c) The RUNTIME SEQUENCE
(insts.bin) is **shim-DMA only**: `npu_dma_memcpy_nd` -> shim BD write + address
patch + queue push (`AIEDmaToNpu.cpp:194-471`); `dma_wait` -> TCT sync. No
core-enable, no tile-channel-start in insts.bin. **Crux: at the instant the shim
drain triggers, the cores + tile/memtile channels have been free-running since
CDO time, gated only by objectfifo locks.** => The EMU's "cores + tile DMAs run
from cy=0" IS toolchain-faithful. **"EMU starts cores too early" is ELIMINATED.**

**Refined hypothesis (the surviving lever).** Since running cores from cy=0 is
faithful AND HW is empty at the shim-drain start (starve t+14 => cores barely ran
before the drain), the only remaining variable is the GAP between core-start and
shim-drain-start: EMU's is **6672cy** (the executor grinding through ~42
runtime-sequence instructions before the `dma_wait` triggers the shim -- shim BD
`Idle->BdSetup` cy=6672 -> Transferring cy=7202); HW's is SHORT (pipeline stays
empty). So the fix locus is the **executor / firmware-latency dispatch timing**
(how long the EMU takes to reach + fire the shim drain), NOT the dataflow and NOT
core-start. **Sharp irony to check:** those ~42 instructions are likely dominated
by TRACE-SETUP register writes (present only in TRACED runs) -- so the act of
tracing may inflate the very cold-start we measure.

**RESUME (next session):**
1. **Decompose the 6672cy** -- dump the 42 insts.bin instructions (trace-setup
   writes vs the memcpy) and the EMU's per-instruction firmware-latency; is it
   HW-calibrated? (Debugging the EMU's OWN model -- legit, not oracle-trusting.)
   Build dir: ABS `/home/triple/npu-work/mlir-aie/test/npu-xrt/spike_bringup/
   build_q0_lean_trace/` (insts.bin); EMU run log
   `build/experiments/sp3-spike-trace/lean/emu_cap0/run.log` (shim start cy=6672).
2. **HW-measure** the real kernel-dispatch -> shim-drain gap (Maya's option 2:
   bypass the trace unit, read AIE registers -- lock values / core PC / timer --
   via the bridge-runner XRT 2.23.0 aie::device path, b7ef416b) to get the
   ground-truth gap the EMU must reproduce. EMU cannot validate itself.
Gate unchanged: lean offset -52->+2 AND send-port -> HW (row 51) AND `--lib` +
bridge no-regress. Inert TEMP env knobs to strip before clean gate:
`XDNA_EMU_WARMUP_CAP` (backend.rs, committed) + `XDNA_EMU_COLDPROBE`
(coordinator.rs, uncommitted).

## 6672cy DECOMPOSED (2026-06-30, session 3): ~all trace infrastructure, amplified by a shared dispatch gate

Step 1 (decompose) DONE. Two source artifacts settle it:

**(A) The runtime sequence is ~90% trace infrastructure.** The lean kernel's
insts.bin lowers to 39 NPU instructions (`aie_traced.mlir` lines 40-78). Decoded
against the AM025 register DB:
- **Lines 40-67 (28x Write32):** ALL trace/perf/timer config -- `Trace_Control0/1`,
  `Trace_Event0/1`, `Timer_Control`, `Performance_Control1/2`,
  `Performance_Counter2_Event_Value` (the PERF_CNT_2 anchor itself!),
  `Stream_Switch_Event_Port_Selection_0/1`, on cores(0,2)(0,3), memtile(0,1), shim(0,0).
- **Lines 68-74 (7 instr):** the **trace-DATA-drain DMA** -- `writebd bd_id=15`,
  `address_patch`, `DMA_S2MM_1_Ctrl` (maskwrite) + `DMA_S2MM_1_Task_Queue`
  (the trace packets' own drain-to-DDR channel), `Event_Broadcast15/14` +
  `Event_Generate` (trace start).
- **Lines 75-76 (2 instr):** the ONLY real data-path ops -- `dma_memcpy_nd`
  (of_out drain, bd_id=0, shim S2MM ch0) + `dma_wait`.
- **Lines 77-78:** trace stop.
So ~35 of 39 instructions exist ONLY because tracing is on. A non-traced run's
runtime sequence is ~4 instructions (the of_out BD write + dispatch + dma_wait).

**(B) The 6672cy breaks into two trace-only contributions** (cost model
`src/npu/cycle_cost.rs::provisional_npu1`, the executor `RetiringInstruction`
charge `executor.rs:583` + the shared dispatch gate `executor.rs:524,541`).
Write32 cost = `cmp_decode(100) + fabric(row*4+3) + register_write(1)` = 104-116cy
each (row-dependent). The shim of_out drain goes `Idle->BdSetup` at cy=6672:
1. **~3100cy** -- the 28 trace-setup Write32s retire serially (one
   `RetiringInstruction` countdown each) before the executor even reaches the
   trace-drain config. Pure trace-setup; absent in a no-trace run.
2. **~3050cy** -- the **shared controller dispatch gate**. At instruction 71 the
   trace-drain `DMA_S2MM_1_Task_Queue` dispatch fires (~cy 3700) and arms the
   executor-wide `controller_next_taskq_cycle = npu_cycle + 3050` (S2MM gate is
   `flat(3050)`). The gate variable is **a single field, not per-channel**
   (`executor.rs:541`), so when the REAL of_out drain dispatch (S2MM ch0, from
   `dma_memcpy_nd`) arrives ~4 instructions later, it STALLS at
   `executor.rs:524` until npu_cycle reaches ~6750. **The real drain is delayed
   3050cy purely because the trace-drain DMA dispatched first and armed the
   shared gate.** Absent in a no-trace run (of_out would be the first S2MM
   dispatch -> gate unarmed -> fires at ~cy 500).

**=> In a non-traced run the of_out shim drain would set up at ~cy 500-700, not
6672. The cold-start over-fill that produces the -52 SP-4a offset is ~ENTIRELY a
trace-measurement artifact:** the free-running cores fill the pipeline during the
6672cy the executor spends on trace setup + the trace-drain-armed dispatch gate.
This is the "tracing inflates the very cold-start we measure" irony, now
quantified.

**Two caveats keeping this from being a clean "EMU bug" verdict:**
- The two dominant constants are CLAIMED HW-calibrated: `write_32 = 100`
  (~100.5 cyc/pkt microbenchmark) and the `dispatch_s2mm = flat(3050)` gate
  (2026-05-27 N=50 HW campaign, g0 2520 / g1 2791). If both are HW-faithful, HW
  ALSO spends thousands of control cycles on the traced runtime sequence -- yet
  HW's cores warm only ~795cy before the trace base (PERF_CNT back-calc) and the
  shim starves at t+14 (empty). The discrepancy is in the **composition** (6672
  EMU vs ~795 HW for the same instructions), not obviously any single constant.
- **Prime suspect = the shared cross-channel dispatch gate.** The 3050cy gate was
  calibrated on `_diag_shim_chain_sweep` (likely SAME-channel chaining), then
  applied as a single executor-wide variable across DIFFERENT channels (trace
  drain S2MM ch1 -> of_out S2MM ch0). If HW's command processor does NOT serialize
  two different channels' first dispatches by 3050cy, the EMU over-serializes ->
  inflates the of_out start by ~3050cy. Testable on HW (option 2): does HW gate two
  different-channel S2MM Task_Queue writes by 3050cy? And the deeper question --
  do HW cores even free-run during control setup, or does the ~795cy say they
  effectively engage at the runtime sequence? -- still needs the register-readback
  measurement. Neither is answerable from the EMU alone.

**Implication for SP-4a (sharpened):** the traced cross-domain offset (+2 HW vs
-52 EMU) is a trace-PERTURBED cold-start sample in BOTH worlds, perturbed
DIFFERENTLY (EMU via shared-gate serialization of trace-drain-then-of_out; HW via
its real CMP). This is the strongest argument yet for Maya's option 2 -- bypass
the trace unit and read AIE registers directly -- because the trace unit is the
confound, not just the readout. Resume = option 2 (HW register-readback of THIS
sequence's real dispatch->drain timing), with the shared-gate faithfulness as the
first thing that measurement settles.

## OPTION-2 SCOPED (2026-06-30, session 3): AIE_RW_ACCESS readback is a DEAD END; trace-microbench is the path

Scoped Maya's option 2 (read AIE registers to get trace-free ground truth).
Result: the **register-readback approach is dead** on Phoenix, on two independent
counts, one of which a prior session already proved:

1. **Writes are root-gated.** A live lean HW run with `bridge-trace-runner
   --read-perf-counter --perf-tile 0:2` got `DRM_IOCTL_AMDXDNA_SET_STATE IOCTL
   failed (err=-13): Permission denied`. The AIE register WRITE ioctl
   (`AMDXDNA_SET_STATE`) is `DRM_ROOT_ONLY` (`xdna-driver
   amdxdna_drm.c:328`); reads (`AMDXDNA_GET_ARRAY`/`AIE_TILE_READ`, line 327) are
   perm 0 (unprivileged). So host-side perf-counter CONFIG needs root.
2. **Readback returns FW artifacts, not real values** -- already established by
   `docs/superpowers/findings/2026-05-26-aie-rw-access-not-a-cycle-probe.md`:
   `read_aie_reg` on Timer_Low advances a fixed ~12,000 ticks/call regardless of
   wall-clock (FW-handler-internal artifact), `write_aie_reg` to Timer_Control
   silently no-ops, and -- decisively -- **the kernel's OWN runtime-sequence
   write32 to a timer/perf reg was INVISIBLE to AIE_RW_ACCESS readback (read 0).**
   That session's verdict: "cycle-accuracy work must remain trace-unit-based."
   So even the root-free "configure-counter-in-kernel, read-back" idea is
   disproven -- the readback can't see kernel-configured registers.

**The sound root-free path is NOT readback -- it is the trace-based dispatch-gap
microbenchmark + in-kernel timer-to-DDR.** The 3050cy gate was ORIGINALLY
measured via the trace unit (START_TASK gaps on `_diag_shim_chain_sweep`,
same-channel; S2MM-direction trace is mailbox-safe, only MM2S level events wedge
it -- `2026-05-27-mm2s-level-trace-event-wedges-mgmt-mailbox`). The "trace is the
confound" concern was specific to trace inflating a dataflow kernel's cold-start
FILL; it does NOT apply to directly timing two dispatch events. Two measurements
partition the SP-4a question:
- **Shim-side (the gate):** fork `_diag_shim_chain_sweep` to dispatch on two
  DIFFERENT S2MM channels (it currently chains ONE), trace both
  `DMA_S2MM_START_TASK` events, measure the gap. EMU's single shared
  `controller_next_taskq_cycle` predicts ~3050; physics predicts ~0 for
  independent channels. Decisive on whether the shared cross-channel gate is the
  bug. (Channel note: shim has only 2 S2MM channels, trace-drain eats one -- a
  2-S2MM-data variant needs the trace drain on another column.)
- **Core-side (the actual SP-4a oracle):** the +2/-52 offset is a CORE LOCK_STALL
  quantity. An in-kernel-timer kernel (each core reads its tile timer, writes
  `(timer, iter, tag)` tuples to a DDR objectfifo) measures the prod->consA
  stagger + first-stall offset TRACE-FREE, read back via reliable BO readback (the
  2026-05-26 finding's untaken "in-kernel instrumentation" option). Requires the
  two cores' timers share a reference -- exactly the faithful timer broadcast this
  arc (SP-1/SP-2) already built; a kernel can set it up in its runtime sequence.
  Feasibility gate: does Peano expose a tile-timer read to core code? (under
  investigation.)

Caveat carried forward: the 6672cy is ~half trace-setup write32 cost (`write_32
= 100`, self-flagged "likely garbage") + ~half the shared gate. The microbench
settles the gate half; full SP-4a closure likely also needs control-path
`write32` recalibration. Evidence: `build/experiments/sp3-spike-trace/lean/
hw_perf/run.log` (the EACCES).
