---
class: core-compute-timing
subsystem: compute-core memory-port / execution cycle timing
posture: mechanism AND per-iteration magnitude modelled + HW-confirmed (bank-arbitration arc); DMA bank-demand cadence HW-derived and fixed; DMA memory-pressure events closed by the staging-FIFO model; producer core-vs-DMA conflict count fixed (C-cycle arbiter, 2026-07-15), residual stall-count phase-average tail documented
status: scalar write-port collisions modelled (fixed 2026-11-03); scalar-side register widths derived and fixed (2026-11-03; status-flag producers open); cascade link modelled (2-word credit window, 2-cycle transit, same-cycle credit return, staggered core starts; 2026-11-04); stall-entry cycle freezes in-flight write-backs (fixed 2026-11-08); core lock requests resolve in front of the fourth bundle after them, freezing the pipeline (fixed 2026-11-08; resume 0-1 cycle early open); `acq.cond` / `rel.cond` perform iff r26 bit 0 is set (fixed 2026-11-08); 2 open (pipeline-fill depth; bank-arbiter resume-cycle hole, dormant/unreachable on corpus), 1 documented residual (producer stall-count phase-average tail below lockstep resolution, not chased), 4 root-caused-and-modelled
---

# Core Compute Timing Gaps

Gaps in the compute core's per-cycle timing model -- specifically memory-port
stalls the hardware charges that the emulator does not. The core's *instruction
issue* timing (bundle-issue rate, lock function-call overhead) is faithful and
HW-validated; these gaps are in the memory-port / stall accounting layered on top.

| Gap | Model vs hardware | Where | Status / rationale | Obs. |
|-----|-------------------|-------|--------------------|------|
| Compute-core MEMORY_STALL: per-access stall not modeled (count + cycle cost) | A memory-heavy compute loop (`eo[i]=ei[i]+1`, 32 elts x 16 reps) on real NPU1 fires **MEMORY_STALL ~220-244x per consumer core**; EMU fires **0**. EMU only models the intra-bundle structural bank-conflict pulse (`event_trace.rs:440`), not the broad per-access stall HW emits. **It is a cycle-cost gap, not cosmetic:** consA HW active span exceeds EMU's by +235cy ~= its un-modeled stall count (220 MEMORY_STALL + 17 extra LOCK_STALL), i.e. ~1 HW cycle per missing stall; the deficit back-pressures upstream (producer span +1235cy). Diagnosed via of_q0_rich EMU-vs-HW trace (PERF_CNT_2 interval 1024 both -> no clock-rate divergence; span deficit == stall count). | EMU core memory-port timing model; intra-bundle pulse `src/interpreter/state/event_trace.rs:440`; evidence `build/experiments/sp3-spike-trace/SP4A-EMU-FINDING.md` | **ROOT-CAUSED + MODELLED (bank-arbitration arc, 2026-07-14); superseded by the two rows below.** The "why" is answered: MEMORY_STALL is the core losing per-physical-bank round-robin arbitration against the tile DMA (AM020 ch.2:166), HW-confirmed cycle-by-cycle (every stall preceded by a bank conflict at t-1, 220/220 and 245/245). The emulator now models it and charges the cycle (`CoreStatus::WaitBank`), and it reproduces HW's magnitude **per pipeline iteration**: 24 stall cycles in a 47-cycle burst, gap 2, banks 0/1 -- identical to silicon (`xdna-emu-priv: research/sdd/coreports-report.md`). Do NOT compare run totals against the `of_q0_rich` capture: it holds ~9 of the kernel's 16 reps, and a total-vs-total reading of it produced a phantom "~1.9x over-production" that was purely 16/9 (retracted 2026-07-14). Lock cycle-cost IS modeled (`f22e301c`); only the lock *pulse* count is cosmetic. | confirmed |
| DMA bank-demand density: EMU DMA requests a bank every cycle, HW bursts 16B per 4 beats | Task 8 validation of the bank-arbitration model against the Phoenix capture (`build/experiments/memory-stall-bankcap/`): the model got the MECHANISM right but charged the **producer** stall cycles silicon does not (102 vs HW 1), and emitted MM2S_0_MEMORY_STARVATION 40/53/128 where HW has 0. Root cause: `words_per_cycle_for` gives a streaming DMA channel 1 word (4B) per cycle, so `peek_bank_demand` declares a bank demand EVERY cycle of a transfer; silicon aggregates stream beats into 128-bit bank accesses (AM020 ch.2:164) and touches the bank once per ~4 beats, i.e. EMU's DMA demand is ~4x too dense and collides ~4x too often. Deliberately NOT fitted (design spec: idealized fast memory, DMA cadence varies with installed silicon). | `src/device/dma/engine/stepping.rs:1357` (`words_per_cycle_for`), `peek_bank_demand`; evidence `xdna-emu-priv: research/findings/2026-07-13-memory-stall-bank-arbitration.md` (Task 8 section), probe `of_q0_rich_bank_arbitration_vs_hw` | **FIXED (2026-07-14, Stage 1 of the granule arc).** The DMA now claims a bank once per 16-byte access granule (`BankLayout::access_granule_bytes`, `word_opens_granule`) instead of once per stream beat, so the other three beats of a granule move through the staging buffer with no bank demand. Effect on the `of_q0_rich` census: **Producer 102 -> 4 stall cycles (HW 1)**. The consumers barely moved, correctly: their contention was never DMA-vs-core (only ~30 of ~406 contended cycles involve a DMA demand at all) but the core's own load and store landing in one physical bank -- which is the real mechanism, HW-confirmed, and already matches silicon per rep. Original analysis retained for the record:  **Root cause now MEASURED on silicon, not hypothesised** (`xdna-emu-priv: research/findings/2026-07-14-dma-bank-access-width.md`, capture `build/experiments/bankwidth/`): a dedicated HW experiment pinned the tile DMA's bank-access width at **16.0 bytes = 4.00 stream beats** (Model B / per-beat predicts 4 B and is off by 3.7x). The DMA lost 112 bank arbitrations in the collide variant and paid 5 cycles for them -- it has memory-side slack and does not claim a bank every cycle. Fix = model the DMA's bank access as a 16-byte burst once per 4 stream beats, which moves DMA timing everywhere else in the emulator -- a separate scoped decision, not a stall-model edit. **Also measured:** `MM2S_0_MEMORY_STARVATION` is 0 on HW even under 112 lost arbitrations, so the emulator's emission of it on arbitration loss is semantically wrong, independent of the cadence fix. **Semantics now MEASURED too** (`xdna-emu-priv: research/findings/2026-07-14-dma-memory-pressure-event-semantics.md`, capture `build/experiments/s2mm-backpressure-semantics/`): both events are FIFO-occupancy signals. `S2MM_0_MEMORY_BACKPRESSURE` is 99.1% contained in `S2MM_0_STALLED_LOCK` windows, asserting exactly 15 cycles after each lock stall begins (zero variance over 39 windows) and deasserting 1 cycle after it ends -- the ingress FIFO fills in 15 beats while the BD cannot write. `BACKPRESSURE INTERSECT STREAM_STARVATION = 0` exactly, confirming both read the same FIFO. Bank contention (205 cyc) cannot explain 757 cyc of backpressure. Fix = model the DMA FIFO occupancy; both events fall out. | confirmed |
| S2MM_MEMORY_BACKPRESSURE semantics: EMU emits it only on bank-arbitration loss | HW asserts `S2MM_0_MEMORY_BACKPRESSURE` for **756 / 853** cycles on the two consumers -- far more than the 232 / 255 cycles in which ANY bank was contended -- so on silicon the event cannot mean only "this channel lost a bank arbitration this cycle". The emulator emits it exactly and only for that (56 / 76 cycles): a strict subset of HW's assertion. The event evidently covers any cycle the S2MM write path cannot drain into memory, of which arbitration loss is one cause. | `src/interpreter/engine/coordinator.rs` Phase E (DMA bank-arbitration pressure block) | **WIRING REMOVED (2026-07-14).** HW disproved the arbitration mechanism outright: `MM2S_MEMORY_STARVATION` stayed at 0 across 112 lost arbitrations. Both events are now emitted NOWHERE rather than on the wrong cause; EMU reads 0 against HW's 756/853, a known gap pending the staging-FIFO model (whose semantics are already measured -- see the row above). Registered in [`dma-stream-resources.md`](dma-stream-resources.md). | confirmed |
| Pipeline-fill depth: EMU free-runs 5 reps before blocking, HW ~7 | On `of_q0_rich` (Q=0, so the cores are released at CDO time and run before the host arms the DDR drain), the emulator completes **5** outer-loop reps before blocking on the un-armed drain; the HW capture shows **~7**. Both then finish the remaining reps once the drain opens. A ~2-rep difference in how deep the pipeline fills. | `src/device/dma/engine/` (egress/ingress fill), the host instruction-stream arm timing | **OPEN (2026-07-14).** A fifo / host-arm timing question, NOT a bank-arbitration one: per rep the stall model is exact (24 stalls in a 47-cycle burst, EMU and HW identical). Evidence: `xdna-emu-priv: research/sdd/coreports-report.md`. Same phenomenon as the emulator's long warmup lock stall noted in `dma-stream-resources.md`. | confirmed |
| Producer core over-collides with its own MM2S: net ~1.6x, from two compensating errors | **RE-CHARACTERISED 2026-07-14 on a clean steady-state probe (`producer_probe`); the "17 vs 1, ~10x" framing is RETRACTED.** On a self-looping MM2S draining logical bank 0 while a dense 1-store/cycle core marches the same logical bank, the producer stall rate is **EMU 30 vs HW 18-20 per run (~1.6x)**, the net of two errors that partly cancel: **(1)** EMU generates **~16x FEWER core-vs-DMA bank conflicts** (68 vs ~1100 CONFLICT_DM_BANK cy/run) -- a deterministic **phase anti-lock**: the DMA's per-granule bank pulse (1-in-4, period 8) and the core's per-store bank run (1/cy, period 8) run at the same rate, and the emulator's fixed step schedule pins them in a stable ~half-period anti-phase (92.5% anti-correlated, 6.7x below chance), so the DMA sails through the dense core's opposite-bank cycles. HW phase-locks the OTHER way (nearly every granule collides). **(2)** EMU's per-bank round-robin then loses **44%** of the few conflicts it has; HW loses only **1.6%** (core wins ~98%, DMA absorbs via egress-FIFO slack, `MM2S_STARVATION=0`). Few-but-lossy vs many-but-rarely-lost cancel to ~1.6x. The old "17" was this anti-lock stacked across of_q0_rich's 16 short-object BD boundaries; the reference-capture "1" was a Q=0 coverage undercount (same 16/9 trap as the width finding). | `src/device/dma/engine/stepping.rs:311` `next_granule_fetch` (DMA demand phase, decoupled from core); `:156` `channel_bank_mask`; `src/interpreter/engine/coordinator.rs` `arbitrate_memory_banks` (~2075-2108, the two phases meet uncoupled); `src/device/bank_arbiter.rs` (round-robin, error #2 site); probe `of_q0_rich_bank_arbitration_vs_hw` (`src/testing/xclbin_suite.rs`) | **CONFLICT COUNT LANDED (2026-07-15, C-cycle fix); residual STALL count DOCUMENTED, not chased.** The two-part fix landed as one change across three commits: core-class-priority arbiter (`c1e404a5`), DMA egress-FIFO urgency override + backoff-on-denial granule fetch (`c01ff205`), and threading that urgency into bank arbitration (`a9c2e134`) -- the backoff drift is what breaks the anti-lock (error #1), and core-CLASS priority with round-robin preserved WITHIN the class is error #2's fix, exactly as designed. FIFO depth unchanged (12); no phase constant, no RNG, no calibration knob. Gate (`xdna-emu-priv: research/sdd/task-4-emu-gate.md`): CONFLICT 68 -> 870 (collide) / 900 (collide_read_dense), within ~20% of HW's 1098-1103 / ~1162, read_dense >= collide (port-occupancy direction correct, per the aiesim sequel finding), apart stays 0 (no false conflicts). STARVATION 0, correct. STALL undershoots: EMU 0 vs HW 18-20 (collide) / 24 (collide_read_dense) -- this is NOT the anti-lock recurring and NOT a calibration miss. These are core-vs-local-SRAM bank conflicts (single-cycle on-chip memory on both HW and EMU; the core never touches DDR); the stall count is a small sub-cycle phase-sensitive TAIL -- on HW the analog jitter along the DMA's egress path (stream switch -> shim -> NoC -> DDR) samples the phase-AVERAGE (~18-20 forced grants), while the deterministic lockstep EMU locks to a single phase and lands in the zero-force regime (the same determinism ceiling the N=10 same-binary bimodality already proved -- see the phase-antilock finding). The CONFLICT count is a large aggregate that averages well under the backoff-drift the fix introduced; the STALL count is the small tail a lockstep model structurally has no way to populate. Registered as its own residual in `docs/known-fidelity-gaps.md` and left for the future variable-latency memory / RAM-simulation model -- the principled place to inject the sub-cycle jitter and de-phase-lock the DMA fetch -- as a SIGNPOST, not a promise it recovers exactly 18-20. Both findings updated to LANDED: `xdna-emu-priv: research/findings/2026-07-14-producer-overcollision-phase-antilock.md`, `xdna-emu-priv: research/findings/2026-07-14-producer-collision-aiesim-bank-model.md`. Plan/spec: `xdna-emu-priv: research/plans/2026-07-15-producer-collision-C-cycle.md`, `xdna-emu-priv: research/specs/2026-07-15-producer-collision-C-cycle-spec.md`. | confirmable: controlled-latency egress |
| Bank-arbiter resume-cycle hole: one un-arbitrated bank touch on lock/DMA/stream resume | The per-physical-bank round-robin arbiter (bank-arbitration arc, `664f824e`) runs in the coordinator's Phase B demand peek (`CoreInterpreter::peek_bank_demand`), which declares nothing for a core in `WaitingLock`/`WaitingDma`/`WaitingStream` (`src/interpreter/core/interpreter.rs` ~198-205). But `try_resume_stall` (same file, ~858) resolves the stall INSIDE `step_internal`, on the commit path, one call after that cycle's peek already ran -- so the exact cycle a stalled core resumes, it decodes and executes its bundle having declared no bank demand for that cycle. The access reaches memory without ever reaching the arbiter: a **missed conflict, never a double grant** -- strictly an under-count of contention, never an over-count, and nothing is corrupted (EMU memory is not physically single-port). Measured unreachable on the validation corpus: disassembling 60 core ELFs (chess + peano, including the objectfifo kernels used for bridge validation) found **zero** bundles pairing a blocking op (`acq`, stream/cascade ops) with a load/store slot -- Peano always emits the acquire as its own out-of-line function (`acq r0, r1` / `ret lr`), a lone bundle with no memory slot, so nothing is ever both resuming AND touching a bank in the same bundle on the current corpus. | `src/interpreter/core/interpreter.rs`: `peek_bank_demand`'s `WaitingLock`/`WaitingDma`/`WaitingStream` exclusion (~198-205), `try_resume_stall` (~858) | **OPEN, dormant (task 6 review, bank-arbitration arc, 2026-07).** Zero bank touches leak on the current corpus (measured, above), so this is not an active fidelity regression today -- but a VLIW bundle CAN legally carry both a blocking op and a memory slot, so the hole is real, just unexercised. Fix is a non-mutating `can_resume_stall()` predicate consulted in the coordinator's Phase B demand peek instead of inside `step_internal`'s commit path; it needs the neighbour-lock snapshot for the `WaitingLock` arm (the only new plumbing) to evaluate the resume condition without mutating state. NOT implemented -- flagged for whoever next touches the stall-resume machinery, or if a kernel corpus ever produces a bundle that trips it. | confirmable: hand-written acq+load bundle under DMA bank load |

## Cascade link (depth, timing, start skew: modelled from silicon 2026-11-04)

Probes: `tools/gen_cascade_probe.py` (smoke, depth) and `tools/gen_cascade_timing_probe.py`
(timing, timer phase, SEMAPHORE partners) through the topology harness
(`python3 -m harness suite build|hw|emu|compare`); data in `build/experiments/harness/cascade/`
(`final/` depth, `t1` timing, `t2`/`t3` timer phase, `sem1` SEMAPHORE partners, `skew1` start skew).
Producer on the north (flow South) or west (flow East) of the consumer, one `aie.cascade_flow`.

The timing probes first pass a **sync word**: the consumer is already held in its first read, so
the release of that read ties the two programs together and everything after it is straight-line
code on both cores (the cores' start skew, below, would otherwise put a run-to-run jitter of
hundreds of cycles on every cross-core placement). Each core stamps its timer at `@ts` markers.

| What | Silicon | Model | Obs. |
|------|---------|-------|------|
| Data path | `vmov MCD, x` / `vmov x, SCD` and the accumulator form (`bml`) carry all 512 bits, in order, South and East (8 distinct words, byte-exact, r1 == r2 == r3) | EMU == HW for both forms since the x-register fix (`cascade.rs`: x is a VectorReg pair) | confirmed |
| **Total buffering** | **2 words**, counted from the write until the consumer takes the word. Depth probe: writes 0 and 1 complete at the issue pitch, write 2 stalls until the consumer reads (27/27 stalling runs, South and East, K = 3/8/24) | `CASCADE_TOTAL_DEPTH = 2`, a credit window over MCD stage + transit + SCD (`Tile::cascade_output_full`) | confirmed |
| **Hop latency** (write to readable) | **2 cycles**: a word written in cycle t is taken by a consumer bundle issuing in t + 2 or later. Family `rel` (data word n bundles after the sync word, consumer read fixed): consumer stall = max(0, n - 4) for n = 0..24, South and East, r1 == r2; the sync release stamp reads the write's stamp + 3 (the stamp bundle after the write) in every South run | `CASCADE_TRANSIT_CYCLES = 2`: two end-of-cycle moves MCD -> transit -> SCD (`route_cascade`) | confirmed |
| **Release latency** (word arrival to held bundle) | **0**: the stall curve above is continuous (no step at the threshold), so a bundle held on an empty SCD issues in the first cycle the word is readable, the cycle an unheld bundle would | the held core resumes in the cycle the word reaches its SCD (`try_resume_stall`, port 254) | confirmed |
| **Credit return** (pop to stalled write) | **same cycle**: family `wrel` (link full, third write n bundles after the second): producer stall = 3 - n, i.e. the stalled write issues in the very cycle the consumer's hold bundle takes a word; family `rt` (consumer reads R words back to back after the link filled and a further write stalled; R = 3, 4, 6, spaced 0, 1, 2 bundles): no read ever waits, the producer's stalled writes issue one per pop. South and East, r1 == r2 | the consumer core steps before its producer in every cycle (`InterpreterEngine::core_step_order`) and the producer's downstream occupancy is refreshed between them (`TileArray::refresh_cascade_occupancy`) | confirmed |
| **Per-side split** of the 2 words | **not observable**: every probe sees exactly the credit window above plus the 2-cycle transit; the offset-free probes (`rel`, `wrel`, `rt`: single-core timer differences) cannot place the two slots, and no cross-core quantity depends on it once the one-way latency is pinned | modelled as the window, not as two FIFOs (the old 1 + 1 split is gone) | equivalent |
| One-way split of the round trip | The offset-free round trip is 2 cycles (pop -> stalled write -> readable); cross-tile stamps read forward 2 / backward 0 if the two tiles' timers agree. **Timer phase probe** (`mf1`/`mr2`, `mf2`/`mr1`): after the sync word one core stores to shared data memory (the producer's South window is the consumer's memory) and the other loads it at a swept distance; the same access kinds in both directions give visibility gaps (neighbour store -> local load 0, local store -> neighbour load 1 cycle) that agree between the two directions only with forward = 2, i.e. the two tiles of a column read the same timer. Two independent pairs agree; r1 == r2 | the emulator's timers are one timebase, so forward 2 / backward 0 reproduce the cross-tile stamps | confirmed |
| **Start skew** | Cores start one at a time, column-major with rows ascending over the cores the design uses, about 1448 cycles apart, jittering: see "Core start order" below. A South pair's producer (row r + 1) starts after its consumer, an East pair's producer (column c) before | staggered starts (`DeviceState::stagger_core_starts`, `CORE_RELEASE_SPACING_CYCLES`) | confirmed |
| Timers across columns | Column c + 1 reads about 140 cycles more than column c at the same instant (East sync release = write + 137..157, a few cycles of jitter per run; skew probe cross-column gaps median 1584 vs 1448 in a column) | **not modelled** (one timebase): East cross-tile stamp differences differ from silicon by this offset; every offset-free East quantity matches | confirmed |
| **Stall freezes in-flight write-backs** | A scalar `lda` in flight when the core stalls lands **7 issued bundles** after it, whatever the stall length: the stall cycles do not advance the pipeline. Probes `tools/gen_cascade_read_probe.py` families `lda_D` (load D = 1..7 bundles after `vmov x0, SCD`, the read's hold at +5 stalling ~15k cycles), `wlda_D` (load before a third `vmov MCD` that stalls on a full link) and `slda_D` (load before a blocking `mov r2, SS` on an empty stream), r1 == r2, no TDR; data `build/experiments/cascade-numerics/{lda1,wlda1,slda1}`. Peano relies on it: `matmul_scalar_cascade` (mlir-aie `matrix_multiplication_using_cascade`, `aie_cascadex4.mlir`) reads the old `r6` six bundles after `lda r6` across the hold | **Fixed 2026-11-08**: the three stall pre-checks in `CycleAccurateExecutor::execute` charged their entry cycle with `record_instruction`, which moves the wall clock without slipping the `pending_writes`/`pending_stores` deadlines, so the load landed at +6 (EMU before: `new_from=+6` for D <= 4 / write or read before +6; the cascadex4 bridge test failed 138-153/256 cells). Now `record_stall` (same cycle count). Tests `scalar_load_in_flight_lands_seven_issued_bundles_later_across_a_{cascade_hold,stream_stall,lock_stall}`. The lock stall freezes the same way (`tools/gen_lock_stall_probe.py` `lk_D`, `same`, next row) | confirmed |
| **Lock request resolution point** | A core `acq` / `rel` does not stop the core where it issues. The next three bundles issue back to back; the fourth waits one arbitration cycle (a granted acquire or any release) or as long as the acquire is refused, and the whole pipeline is frozen meanwhile (`pos_stall` counter-read deltas after an `acq` 2, 1, 1, **14714**, 1, 1, 1; `pos_free` and `relpos` 2, 1, 1, **2**, 1, 1, 1). A scalar load in flight across the stall lands 7 issued bundles after it (`lk_D`, D = 1..8, stall and free), and an `lda ; paddb ; acq` bundle commits its pointer add once (`same`). The held core resumes 7 or 8 cycles after the releasing core's counter read that precedes its `rel`, set by a 3-cycle phase (below). Probes `tools/gen_lock_stall_probe.py` (two cores sharing one lock, harness `SharedLock`), r1 == r2, no TDR; data `build/experiments/lock-stall/{suite,suite2}` | **Fixed 2026-11-08**: a refused `acq` bundle ran to completion (its other slots committed, `instructions`, `bundle_seq` and the write-back clock advanced: a load in flight landed one issued bundle early, `lk_D` `new_from=+6` for D <= 5) and re-ran on resume (`same`: the pointer moved 8); the arbitration cycle was charged at the `acq` itself (window 11 where silicon reads 10 for D >= 6); the core resumed on any count > 0. Now `CycleAccurateExecutor` holds the request (`PendingLockRequest`) and performs it in front of the fourth bundle (`aie2::locks::CORE_REQUEST_BUNDLES_IN_FLIGHT`), holding that bundle; the interpreter polls the held acquire's own condition. Tests `a_lock_request_resolves_in_front_of_the_fourth_bundle_after_it`, `the_bundle_carrying_a_refused_acquire_commits_once`, `a_lock_stall_resumes_only_when_the_acquire_can_be_granted`. Bridge corpus: verdicts unchanged; where an acquire needs a count above 1 (repeat counts) the old resume check re-ran and re-blocked the `acq` every cycle, so LOCK_STALL and shim STREAM_STARVATION toggled per cycle (`objectfifo_repeat_compute_repeat` starvation intervals EMU 449 -> 12, HW 24-32). **Back-to-back requests (2026-11-08, `b2b` family, rel->acq, acq->acq, acq->rel, rel->rel at 1..5 bundles apart, counter read in every bundle, each lock's count read back afterwards; r1 == r2; data `build/experiments/lock-b2b`)**: a request issued 1..3 bundles after another resolves with it, in front of the first's fourth bundle, with one arbitration cycle for both, and both take effect; 4 or more bundles apart each resolves in front of its own fourth bundle (lost cycles in front of +4 and +d+4). The EMU performed the first in front of the second's bundle (+d) and the second at +d+4 (2 cycles where silicon loses 1); now `PendingLockRequests` holds every request of the window (test `a_lock_request_inside_the_window_of_another_resolves_with_it`). The compilers never issue requests that close (llvm-aie II_ACQ / II_REL hold SEMAPHORE over +1..+4), and silicon does something irregular with a *refused* acquire inside a window: the core never waits on it (an acquire refused with a second request at +1..+3, or the refused second request itself, costs 1, 3 or 4 cycles), and the acquire then takes effect late, is dropped or turns into a release (classified by the flag and count read-back, below), has no effect (`rel L ; acq L` at count 0, 1-2 bundles apart; first acquire with an `acq` 3 bundles later), or leaves the lock one higher (refused `acq` followed by a `rel` of another lock: as if it became that release); not modelled: the EMU holds the core on it like any refused acquire (also `rel L ; acq L` 3 bundles apart, which silicon performs in order with 1 cycle, costs the EMU 2: the own-tile release reaches the lock through the tile arbiter a cycle later). **Resume phase**: the held bundle never issues at a cycle I with (I - S) mod 3 == 0, S the cycle it was first held (`aie2::locks::CORE_RESUME_PHASE_CYCLES`): P+8 exactly when (P - S) mod 3 == 2, else P+7, 116 samples over five runs (the same binary gives either, so it is not layout; the old 8-vs-7 split of `lk_1`/`lk_2`/`pos` vs `lk_3..5` was this phase). Test `a_held_bundle_issues_off_a_three_cycle_phase_of_its_stall_start`; EMU end to end P+7, P+7, P+8 for a producer shifted 0, 1, 2 cycles (`lock-b2b/phase`). **Arbitration cycle as an engine step**: the arbitration cycle used to be charged inside the step that issued the bundle, so every lock request moved the core's counter one cycle ahead of the other cores' (EMU P+8 where silicon reads P+7 behind one extra request); it is now its own step (`ExecuteResult`/`StepResult::LockArbitration`, a lock-stall cycle in `Core_Status`; test `every_step_of_a_granted_lock_request_is_one_cycle`)  **Refused acquire inside a window, read back (2026-11-08, `b2b_*` cases of the `frozen` family: the host reads both locks' counts and Locks_Overflow/Underflow over control packets after the window, then sets the refused lock's count to 1 -- a request still waiting takes it, a dropped one leaves 1; window a, d = 1..3, r1 == r2, `build/experiments/lock-frozen/suite4`)**: no case sets a flag (nothing underflows or wraps). Classes: (a) *pending* -- the core is not held and the acquire completes in the background once the count allows (second request refused, `rel L14 ; acq L15` and `acq L14 ; acq L15` on an empty L15, d = 1..3, 1 lost cycle; refused first request followed by an acquire, d = 1, 2, 3 lost cycles); (b) *dropped* (refused first request followed by an acquire at d = 3, 4 lost cycles; `rel L ; acq L` at count 0, d = 1, 2: the release would have let it through and it is still not granted); (c) *turned into a release of +1* on its own lock (refused acquire first, a release of another lock second, d = 1..3, 3/3/4 lost cycles); (d) *performed in order* (`rel L ; acq L`, d = 3). No single rule explains them (order, kinds, d and same-lock all matter), and no compiler emits requests this close, so it is **not modelled**: the EMU still holds the core on the refused acquire (suite4 `e1`: counts equal silicon only for the pending class). | confirmed; classified above, not modelled (needs a rule or a per-case table); a request on a frozen lock: see dma-stream-resources.md |
| **Conditional lock requests (`acq.cond` / `rel.cond`)** | Both take the condition register r26 as a third operand (llvm-aie `AIE2_acquire_cond_inst_alu`; intrinsics `int_aie2_acquire_cond` / `int_aie2_release_cond`, builtins `__builtin_aiev2_acquire_cond` / `_release_cond`). Silicon (`cond` family of `tools/gen_lock_stall_probe.py`, 21 cases, r1 == r2, no TDR; `acq` on a lock holding 1, on an empty lock that the producer releases ~15k cycles later, and `rel` into an empty lock, with r26 = 0, 1, 2, 0x100, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFE): the request is performed iff **bit 0 of r26 is set** (1 and 0xFFFFFFFF; 0, 2, 0x100, 0x80000000, 0xFFFFFFFE are skipped), the same for the acquire and the release. A performed request is the plain one (granted acquire: 1 lost cycle in front of +4 and the count taken; refused acquire: ~14.5-14.8k lost; release: 1 lost cycle and +1). A skipped one is nothing at all: no arbitration cycle, no wait on a lock that would refuse it, count untouched, and it opens no window (`condwin`, 24 cases: a skipped `acq.cond` / `rel.cond` followed by a plain `rel` 1..3 bundles later leaves that `rel` resolving in front of its own fourth bundle, +d+4, where a performed one gives +4; a skipped one after a plain `rel` changes nothing). The condition is read at issue: a `mov r26` in the previous bundle is seen, one in the same bundle is not (`condlat`, 12 cases). Reachability: Peano emits the instructions only from the explicit builtins; mlir-aie, the runtime library and the kernels never use them; data `build/experiments/lock-cond` | **Fixed 2026-11-08**: the EMU ignored r26 (both instructions were plain requests: skipped ones cost the arbitration cycle or held the core on a lock that was empty). `ControlUnit::lock_request` returns no request when the op's third source (r26) has bit 0 clear. Tests `a_conditional_lock_request_is_performed_iff_bit_0_of_r26_is_set`, `a_skipped_conditional_request_opens_no_window`, `test_conditional_lock_requests_carry_r26_as_a_third_source`; EMU == HW on all 57 probe cases | confirmed; unobserved: the INSTR_LOCK_ACQUIRE_REQ trace event of a skipped request (EMU emits none), edge-observable with the trace unit |
| SEMAPHORE partners of `VMOV_CASCADE_READ` | No effect at any distance: see vector-compute.md, "Cascade-read collisions" | nothing to model | confirmed |

EMU vs silicon after the model change (`t1`, 78 cases: `rel` n = 0..24, `wrel` n = 0..8, five `rt`
patterns, South and East): every offset-free quantity and every South cross-tile stamp
difference equal. Before it the offset-free quantities already matched, but the South stamps read
forward 1 / backward 1: the old model moved a word one stage per cycle and freed the producer a
cycle after the pop.

### Core start order (silicon 2026-11-04)

Skew probe (`build/experiments/harness/cascade/skew1`, generator `skewgen.py` there): 2 to 4 cores,
each stamping its timer at its first instruction; layouts: a column (rows 2-5), rows 2+3, rows
2+5, a row (columns 0-3), columns 0+1, a 2x2 square; 6 runs each. The cores start in
column-major order with rows ascending over the cores present (the square: (0,2), (0,3), (1,2),
(1,3)); consecutive starts are one gap apart whatever the distance (rows 2 and 5 alone: one gap).
The gap is jitter, not a function of the layout: 60 same-column gaps span 1103-2018 cycles
(median 1448.5, mean 1473, sd 201), and one layout varies run to run. The order is that of the
cores' enable writes in mlir-aie's CDO (its init section clears RESET, its enable section sets
ENABLE, one write per core, column-major): the launch path applies them one gap apart (off-array).
The emulator starts the first core at its enable write and stages each later start one more
`CORE_RELEASE_SPACING_CYCLES` (1448, the median, a deterministic representative) after it
(`DeviceState::stagger_core_starts`); cores a CDO leaves held in reset take the following slots at
the release seam (`release_core_resets`). A core's `MOV_CNTR` reads its core module timer, so
cores started at different times read one timebase, a host `Timer_Control` reset moves what a
running core reads (silicon, launch probe; the EMU read engine cycles and ignored the reset until
2026-11-08, `cycle_counter_follows_a_reset_of_the_core_timer`), and it writes both halves of its
register pair (the EMU used to leave the high register untouched; silicon writes the timer's high
word, 0 here). Where the start sits relative to the runtime sequence: "Core launch against the
runtime sequence" in [`host-firmware-dispatch.md`](host-firmware-dispatch.md). The launch probe
(2026-11-08) measured the same order with gaps of 946-1188 cycles (24 gaps, median about 1100),
inside this jitter range but under its median; the real-firmware backend gives 1100-1144.

### Cross-tile memory visibility depends on core step order (EMU, open; found 2026-11-04)

The timer-phase probe doubles as a cross-tile memory probe. Silicon, with the timers aligned: a
store through a neighbour window is visible to a local load issued in the same cycle, and a local
store to a neighbour-window load issued one cycle later, in both directions of a column (`t2`/`t3`,
`mf*`/`mr*`, r1 == r2). The emulator gets two of the four right and is one cycle off in the other
two, and which two depends on the order cores step within a cycle (column-major, rows ascending):
a neighbour store by the later-stepping core reaches an earlier-stepping local reader one cycle
late (`mf1`: EMU threshold k = 9, silicon 8), and a local store by the earlier-stepping core
reaches a later-stepping neighbour-window reader one cycle early (`mr1`: EMU 2, silicon 3). Cause:
buffered neighbour stores drain after the writing core's step and neighbour reads snapshot
lazily during the reading core's step (`execute/memory/neighbor.rs`), so same-cycle visibility
follows the step order instead of the pipeline stages. Not changed here (memory subsystem, outside
the cascade scope); the fix is order-independent commit points for cross-tile stores and reads,
checked against these four thresholds.

Probe-design findings (silicon):

- **The "jnz loop hangs cascade probes" finding was a probe bug, now explained.**
  Silicon fetches program memory in aligned 16-byte lines and a taken
  `jnz`/`j`/`jl` (and the `ret` return address) resumes fetch at the start of the
  line holding the target: the low 4 bits of a branch target are ignored. The old
  probes put `mov r0, #N` in the same line before the loop label, so it re-ran on
  every iteration and the loop never ended (core never finished, TDR, Status 8).
  Nothing to do with the cascade. Evidence
  (`build/experiments/cascade-collisions/p1/`): a marker `add r5,r5,#1` at a line
  start and the loop label at line offset R, four iterations: R=0 gives r5=1,
  every R in 4..14 gives r5=4; `j` (fj), `jl` callee entry (fjl_entry2) and the
  `jl` return address (fjl_ret3) truncate the same way. The emulator now matches
  (`PendingBranch` masks targets, `FETCH_LINE_BYTES`). llvm-aie pads every jump
  target and return address to 16 bytes (`AIEMachineAlignment.cpp`), so compiled
  code never notices; hand-written loops need `.p2align 4` before the label.
- **Where a cascade read stalls (derived, EMU matches).** `vmov x/bml, SCD`
  neither stalls at the vmov nor at the first use of its destination. The four
  bundles after it issue normally (a cycle-counter read 1..4 bundles later excludes
  the wait); the fifth bundle after it is held until a cascade word is available
  (a counter read there or later includes the whole wait). The destination is
  written so that it is visible 7 issued bundles after the vmov (`II_VMOV_CASCADE_READ`
  = 7, TableGen), independent of any hold; there is no interlock, so a reader at
  +1..+6 sees the OLD register value even if the core then holds. Same for x and
  bm destinations, with independent vector ops in between; a later `vbcst x0` (+1)
  is overwritten by the SCD word landing at +7; two reads hold at the same point
  relative to the first. Word already buffered: no stall, one bundle per cycle.
  Probe: `tools/gen_cascade_read_probe.py`, data `build/experiments/cascade-collisions/p2/`
  (`hw_r1_sweep1.txt`, 72 cases x absent/present; EMU `emu_e1_sweep1.txt` identical via
  `p2/cmp.py`). EMU: `PendingCascadeRead` (`state/context.rs`), `CascadeOps::{execute_read,
  due_read_would_stall, complete_due_read}`; tests in `execute/cascade.rs`. Before this the
  EMU stalled at the vmov and wrote the destination immediately. The split and the release
  latency were probed 2026-11-04 (table above).
- `vst` clears the low 5 address bits on silicon (a store at +48 landed at +32). Derived for
  the whole load/store family and fixed in the emulator: see "Unaligned accesses" in
  [`addressing.md`](addressing.md).

## Zero-overhead loop set-up: the loop registers are read at fetch (FIXED 2026-11-09, NPU1 silicon)

Hand-written loops (`add.nc lc`, `movxm ls`, `movxm le`, a body of 12 or 40 nop bundles) ran once on
silicon and four times in the emulator. The rule silicon enforces is that the loop registers are read
when a 16-byte program line is *fetched*, and the fetch runs four lines ahead of issue; the emulator
now models that front end (`src/interpreter/state/fetch.rs`).

Probe: `tools/gen_zol_probe.py` (84 loops, one per kernel; each stores its pass count, LC afterwards,
and counters for the bundle after LE, a bundle before LS and the line after LE's to the output buffer).
Silicon and emulator agree on all of them (`build/experiments/front-end/z6`, tags `h1.hw` / `e1.emu`).
Before the change the emulator iterated every ill-formed variant below.

| Observation (silicon) | Rule |
|---|---|
| LE not a multiple of 16: one pass, whatever the distance (offsets 2..14 tried, 100+ bytes from the set-up) | the fetch compares a line address with LE exactly |
| LE a line start, set-up too close: one pass. The last of the three writes to LE, LC or LS, a bundle ending at `end`: loops iff `LE >= line(end) + 80`. Same threshold for 2-byte nops, 6-byte and 16-byte bundles | the writing bundle's lines are already fetched up to four beyond the line of the next PC (`FETCH_AHEAD_LINES`); LE written after its line was fetched is never compared |
| LC written last and late (`add.nc lc`, 4 bytes, `LE < line(end)+80`): five passes with LC left at 1 (stale LC was 0) | the fetch read the stale LC = 0, looped without a real count, then the late write of 4 gave three more |
| LS written last and late: one pass, LC 3, the bundle after LE run twice | the fetch redirected once to the stale LS (here a label inside the LE line) and the comparator then stayed disarmed |
| LC = n (n >= 1): n passes, LC afterwards 1 (not 0). LC = 1: one pass, LC unchanged | the loop-back needs LC != 1 and decrements it |
| LC = 0: loops on, LC counts down through 0 (12 passes, LC = -11, bounded by a `jz` in the body) | LC = 0 is not a zero count |
| The bundle after LE in its line runs on every pass; the next line's first bundle only after the last | the loop-back reaches issue once the last bundle of the LE line has issued |
| LS off a line start (offsets 4..14 tried): two passes, the bundle before LS in its line runs twice, LC 3 | the redirect lands on LS's line start; fetching a line equal to LS exactly re-arms the comparator, which an unaligned LS never gets |
| LS and LE in one line, or a loop of two lines: loop normally | the re-arm happens before the LE compare of the same line |

llvm-aie agrees: `getZOLSupport` demands 7 bundles of 16 bytes (112) between the last set-up bundle
and the loop end (`LoopSetupDistance`), the loop-end bundle is its own aligned region
(`AIEMachineAlignment.cpp`), and LS is a block start, so compiled loops are always well-formed. The
silicon threshold is 70..72 bytes, so the compiler keeps a margin.

State the model does not reach (each reproduced on silicon, deterministic, not modelled):
- An *aborted* ill-formed loop (left by a `jz` mid-body, or one whose redirect went to a stale LS)
  changes how the **next loop in the same kernel** behaves: in the early multi-loop probes the
  following loops lost redirects or ran a fifth pass. The probe therefore runs one loop per kernel.
  Whatever program-control state survives an abnormal loop exit is not identified.
- Where the comparator is armed from reset is not pinned: the model arms it on a change of LE.
- Bundles that straddle a line boundary are treated as belonging to the line they start in.

## Taken branches cost no extra cycles; the "0..7 stall" was line re-execution (silicon 2026-11-09, EMU already agrees)

`trace-encoding.md` recorded that the same 45-bundle `br_j` sites cost 0..7 extra cycles between
consecutive events, cycling with the site's alignment, and concluded that the front end stalls on a
taken branch. Counter reads (`mov r17:r16, cntr`) around the branch show no stall:

- `tools/gen_branch_stall_probe.py` brackets a `j`, a taken or not-taken `jnz`, or a `jl` + `ret`
  with two counter reads and compares the cycles to the number of bundles actually issued (derived
  from the linked ELF's disassembly, target addresses truncated to their 16-byte line). 70 sites on
  silicon: `extra = 0` at every alignment of the branch against the fetch lines (lead 0..7, read to
  branch 0..4 bundles), every distance past the target (0..47 bundles), with 2-byte, 6-byte and
  16-byte bundles around it and in the five delay slots. The emulator's counter deltas are
  identical to silicon's on all of them (`build/experiments/front-end/b1`, tags `h*.hw` / `e*.emu`),
  because it charges no branch penalty (the delay slots are the cost, `cycle_accurate.rs`).
- The 45..52 gaps are real cycles but not stalls: a taken branch resumes at the *start of the line*
  holding the target, so the bundles before the label in that line run again, 0..7 two-byte nops
  depending on where the label sits. The `event #0` / counter-read bundle is 4 bytes, so every site
  shifts the following code by 2 bytes and the count cycles with the site, exactly as measured. The
  chained counter reads (`chain_j`) give 46, 49, 52, 48, 50, 45, 48 on silicon and in the emulator.
- Two `jl` sites showed negative "extras": the return address's line start fell inside a 4-byte
  bundle that straddles the line, so the return resumed in the middle of a bundle (silicon runs
  whatever that decodes to, the emulator stops with an unknown instruction). Not a timing effect;
  the probe now keeps to `jl` placements where no bundle straddles that line.

So there is no taken-branch fetch stall to model. Cycle counts did not move.

### Mode-1 (EventPc) PC around branches: still open

The PC carried by a mode-1 frame is **not** a fetch-line quantity. Straight-line code
(`gen_fetch_pc_probe.py scan`, `scan_wide`, `scan_mid`; PC minus the event bundle's address):

| bundles around the event | PC lead in bytes | in bundles |
|---|---|---|
| 2-byte nops | 8 (10 when the event bundle's line offset is 0 or 4) | 4 |
| 6-byte bundles | 12..16 | ~2.7 |
| 16-byte bundles | 22..36 (36 for most) | ~2.3 |

so it is neither a fixed byte count nor a fixed bundle count, and it moves with the event's offset
in its line. Around a taken `j` the PC minus the issue address in bytes is 8, 20, 10, 20, 26 for the
events 5..1 bundles before it, 12 and 18 in the first two delay slots (both in the skipped
fall-through code), and 32 in delay slots 3..5 (at the target), with the PC one or two lines beyond
the issue line (not the four a line-granular fetch pointer would give). A fetch-stage pointer
(`state/fetch.rs`) does not produce this; the mechanism (a decode-stage pointer whose lead depends on
bundle sizes and on the redirect timing) is not settled by these data, so `trace_pc` keeps its
sequential walk and the `br_j` / `br_jl` rows stay OPEN.

## Scalar write-port collisions: a delayed write dropped or ORed (FIXED, NPU1 silicon 2026-11-03)

`mov.dN dst, src` (N = 1..6) writes `dst` N+1 cycles after issue, from the source read at issue
(`execute_mov`, result latency from the schedule; probe `build/experiments/harness/pairs/probe3c`).
Silicon **drops that write** when another scalar write lands in the same cycle through the same
write port: `mov.d1 r1, r2` then `mov r5, r6` makes `r1` keep its old value for good. The compilers
never emit such schedules (TableGen's resource model forbids them); hand-written assembly, fuzzers
and suite tests do.

**Port model, derived from TableGen** (`llvm-aie/.../aie2/AIE2Schedule.td`): `AIE2Schedule.td:73-96`
names the r/s register write unit `RS_WM_PORT`; `II_MOVd1..6` hold it in cycle N (`:633-638`:
`SimpleCycle<P_RM_PORT>, EmptyCycles<N-1>, SimpleCycle<RS_WM_PORT>`, result latency N+1), the plain
movers (`II_MOV*` `:606-631`, `II_ADD_NC*` `:445-454`) in cycle 0, the vector extract and compare
family (`II_VEXTRACT*`, `II_VCMP`, `II_VEQZ`, `II_VMAX_LT`, ... `:755-968`) in cycle 1. The other
scalar write units are `R_WX_PORT` (ALU, `mul` in cycle 1), `R_WA_PORT` (`lda`/`mova`/`mov SS`, a load
in cycle 6) and the per-file `P_WM_PORT`/`M_WM_PORT`/`DN_WM_PORT`/`DJ_WM_PORT`/`DC_WM_PORT`. An
instruction holds its write unit in the cycle before its result lands (`def - 1`).

**The rule on silicon** (510 trials in three suites, `build/experiments/harness/movd/p{1,2,3}`,
probes `probe{1,2,3}.py`; HW r1 == r2 byte-identical on every case):

1. *Same register file, same port, same landing cycle: the later-issued write survives, every
   earlier one is dropped* (register keeps its old value). The interferer's own write always
   survives. Of two delayed moves landing together the one issued second (shorter delay) wins;
   with three, only the last survives.
2. *Same register through two different ports in one cycle: the bitwise OR of both values*
   (`mov.d3 r1` + `add r1` = `0x12345673 | 0x0f0f0f01`; `mova r1, #5` makes `..77`, not `..78`;
   `lda r1` bundled with `mov.d6 r1`; `mul r1` with `mov.d1 r1`). The same port with the same
   register is a drop, not an OR.
3. *Different register files do not collide even where the schedule lists the same unit.* TableGen
   **overpredicts**: `II_MOV_SCL_{RS(eS),P,M,D*}`, `II_ADD_NC_{P,M,..}` and `II_VEXTRACT_{P,M,..}` also
   hold `RS_WM_PORT`, but `mov s0/p4/m1/dn1/dj1/dc1`, `add.nc` and `vextract` into them never drop a
   `mov.dN` into `r1`. A write is therefore keyed by (unit, destination register file).
4. Collides exactly when `k + (own hold cycle) = (the lead's hold cycle)`: `mov`/`movxm`/`add.nc` `k = N`;
   `vextract`/`vlt`/`veqz`/`vmax_lt`/`vabs_gtz`/`vsub_lt` into `r` `k = N-1`; neighbours (+-1, +-2) never.
   Other ports: `lda r` (`R_WA@6`) is dropped by `mova` at `k = 6`; `lda p/m/dn` by `mov` to the same file
   at `k = 6` (not by `mov m1` for `lda p5`); `mul r` (`R_WX@1`) by `add`/`sub` at `k = 1` (not by
   `movx crSat`, which writes no r register); `add`, `mul`, `lda`, `mova`, `movx`, `vadd` never drop
   a `mov.dN`.

**EMU** (branch `feat/movd-port`): `decode::ports::PortModel::scalar_defs` lists, per scalar result,
the units its resolved itinerary holds in the cycle before it lands and the destination file
(`PortUse::scalar_defs`, no unit named in code); the context logs every such write
(`ExecutionContext::log_scalar_write`); a deferred write (`mov.dN`, `lda`, `mul`, vector extract and
compare masks) asks the pure resolver `state::scalar_collision::resolve` when it lands. Tests:
`scalar_collision` (rules), `decode/scalar_ports_goldens.rs` (derived ports of 30 instructions),
`execute/scalar_port_goldens.rs` (decoded bytes through the executor), ISA suite batches 195-199
(63 test points, declared payload, EMU == HW). Probe suites replay EMU == HW: p1 11/11, p2 6/6, p3 3/3.

Not probed: `mov SS` (`R_WA@6`, needs a stream), `sthb`, collisions of two writes to the S file
among themselves, M vs DN cross-file on a shared unit. Found in passing: the wide (x) forms of
`vabs_gtz`/`vneg_gtz`/`vneg_ltz`/`vsub_lt`/`vsub_ge`/`vmaxdiff_lt` wrote no compare mask in the
emulator (fixed, `execute/wide_cmp_goldens.rs`); shift registers `s0-s3` held 6 bits on silicon
and 32 in the emulator (fixed with every other register width, next section).

## Scalar-side register widths: a write keeps only the register's own bits (FIXED, NPU1 silicon 2026-11-03)

`mov s0, r6` with `r6 = 0x66666666` reads back `0x26` on silicon (6 bits); the emulator kept 32, as
it did for `lr`, `LS`, `LE`, `DP`, `sp`, every control register and every status register.

**Derivation** (`interpreter/state/register_widths.rs`): the width is TableGen's, the narrowest
register class holding the register (`RegInfo.RegSize` of its classes, extracted into
`RegisterClassDef::reg_size_bits`, `RegisterModel::register_width_bits`): the 20-bit classes
(`AIE220BitRegisterClass`: `eP`, `eM`, `eDN`, `eDJ`, `eDC`, `eSpecial20` = LS, DP, lr, LE, SP,
CORE_ID) give 20, the scalar classes 32. TableGen states a CLASS width and puts the shift, control and
status registers in 32-bit scalar classes (`eS`, `mCRm`, `mSRm`), so where silicon keeps fewer bits
`silicon_bits` narrows them (the only hand data; each entry below is a measured row). Reads: `mov r,
<reg>` zero-extends every register except `m` and `dj`, which sign-extend their 20 bits.

Silicon probe: `build/experiments/harness/bugs/widths` (`widths_gen.py`; 17 cases, eight patterns
per register: all ones, `0x66666666`, `0xA5A5A5A5`, `0x5A5A5A5A`, bit 19, bit 31, `0x40`, `1`; HW
r1 == r2 byte-identical; the probe restores every register it writes). Pinned in the ISA suite by
batches 200-205 (68 test points, `ScalarWidthStrategy`, declared payload, EMU == HW).

| Register(s) | Bits | Read-back | TableGen (class: RegSize) | Silicon (all ones -> readback) |
|---|---|---|---|---|
| r0-r31 | 32 | n/a | `eR`: 32 | `r9` `ffffffff` |
| p0-p7 | 20 | zero-extended | `eP`: 20 | `p2..p5` `000fffff` (`0x5a5a5a5a` -> `000a5a5a`) |
| m0-m7 | 20 | sign-extended | `eM`: 20 | `ffffffff` (`0x5a5a5a5a` -> `fffa5a5a`, bit 19 -> `fff80000`) |
| dj0-dj7 | 20 | sign-extended | `eDJ`: 20 | `ffffffff`, same |
| dn0-dn7, dc0-dc7 | 20 | zero-extended | `eDN`, `eDC`: 20 | `000fffff` (`0x5a5a5a5a` -> `000a5a5a`) |
| LS, LE, DP, lr, sp | 20 | zero-extended | `eSpecial20`: 20 | `000fffff` |
| CORE_ID | 20 | read only | `eSpecial20`: 20 | `col << 16 | row` (`00010002` at tile 1,2); writes ignored |
| LC | 32 | n/a | `eRLC`/`mMvSclDst`: 32 only | `ffffffff` |
| s0-s3 | 6 | zero-extended | `eS`: 32 | `0000003f` (`0x66666666` -> `26`) |
| crSat | 2 | zero-extended | `mCRm`: 32 | `3` |
| crRnd | 4 | zero-extended | `mCRm`: 32 | `f` |
| crFPMask, crF2IMask, crF2FMask | 5 | zero-extended | `mCRm`: 32 | `1f` |
| crSRSSign, crUPSSign, crPackSign, crUnpackSign, crVaddSign, crSCDEn, crMCDEn | 1 | zero-extended | `mCRm`: 32 | `1` |
| srSS0 | 2 | zero-extended | `mSRm`: 32 | `3` |
| srFPFlags, srF2IFlags, srF2FFlags | 5 | zero-extended | `mSRm`: 32 | `1f` |
| srCarry, srMS0, srSRS_of, srUPS_of, srCompr_uf, srSparse_of | 1 | zero-extended | `mSRm`: 32 | `1` |

p6/p7, p0/p1 and the m/dn/dj/dc indices not probed take the class width; the m0/m7 and dn/dj/dc 0/7
probes showed the eight registers of a file alike. The pointer and modifier masks were 20 bits in the
emulator already (from the tile address mask), but `dn`/`dc` sign-extended on a scalar read where
silicon zero-extends (fixed), and `lr`/`LS`/`LE`/`DP`/`sp`, the control and the status registers kept
32 bits.

Six status registers were not modelled at all (`mov srCarry, r6` was dropped, a read gave 0); they
are now slots that keep what a move wrote, masked to their width. `srCarry`, `srSRS_of` and
`srUPS_of` start at 0 and have producers (2026-11-04, `src/interpreter/execute/status_flags.rs`):

- **`srCarry`** is written only by the instructions whose TableGen record lists
  `Defs = [srCarry]` (`add` register and immediate, `sub`, `adc`, `sbc`, `abs`, `jnzd`); `add.nc`,
  `mul`, logic, shifts and compares leave it. The add family sets the carry out of the 32-bit
  unsigned sum (an immediate is sign-extended; `jnzd` adds -1, so its carry is "the source was
  nonzero"), the subtract family the **borrow** (1 when the subtrahend plus borrow-in exceeds the
  minuend; `sbc` subtracts the flag as borrow-in: the emulator had ARM's inverted carry), `abs` the
  sign of its source. It is overwritten, not sticky; a reader in the same bundle sees the old
  value, the next bundle the new one (`adc`/`sbc` one bundle after `add`/`sub` take its carry).
  Evidence: `tools/gen_carry_flag_probe.py` (12 cases, 1112 records, edge operands from either
  carry, chains at 1-3 bundles, ten non-definers), HW r1 == r2, EMU == HW, model check
  `--analyze` 1112/1112.
- **`srSRS_of` / `srUPS_of`**: see "Status flags" in [vector-compute](vector-compute.md).

- **`srFPFlags` / `srF2IFlags` / `srF2FFlags`**: modelled 2026-11-05 (five sticky flag bits per register,
  enable masks, per-definer visibility); see "Floating-point status flags" in
  [vector-compute](vector-compute.md).

The pre-existing status slots (`srSS0`, `srMS0`, `srCompr_uf`, `srSparse_of`) keep their producers.

## Related: lock-arbitration cost and its LOCK_STALL pulse (modeled)

The per-lock-transaction arbitration **cycle cost** (HW trails every lock request
with a 1-cycle arbitration stall, even uncontended) is modeled in core cycle
timing (`f22e301c`, `cycle_accurate.rs`); since 2026-11-08 it is charged where silicon
puts it, in front of the fourth bundle after the request ("Lock request resolution
point" above). Its **trace pulse** is modeled since 2026-11-09
(`build/experiments/lock-stall-pulse`, `tools/gen_lock_pulse_probe.py`, silicon r1 == r2):

- Every window of lock requests, acquire or release, uncontended, puts one 1-cycle
  LOCK_STALL in the core trace on its arbitration cycle: request bundle at cycle c,
  INSTR_LOCK_*_REQ at c+1, LOCK_STALL at c+4. Requests inside one window (d = 1..3
  bundles apart) share the one pulse; from d = 4 each window has its own (c+4 and
  c+d+5). Only acquires lacked it in the emulator (releases pulsed already), which is the
  HW/EMU 2:1 LOCK_STALL interval ratio of the trace-gate core triage.
- A refused acquire is one held level from c+4; the arbitration cycle of the grant is
  inside it, no pulse follows (`a_granted_wait_adds_no_pulse_after_the_held_level`).
- A skipped `acq.cond` / `rel.cond` pulses nothing; a performed one pulses like the plain
  request.
- A core held on an acquire since before the trace started shows LOCK_STALL in the first
  frame (one cycle after the start event); the trace unit now tracks level edges from
  before it runs (`level_signals`).

Open (not a lock-arbitration cost, a trace-stamping rule): INSTR_LOCK_ACQUIRE_REQ /
INSTR_LOCK_RELEASE_REQ sit at c+1 on silicon, for a refused acquire too (so a core held
across the trace start has no REQ in the trace for that acquire), where the emulator
stamps them when the request completes (c+4, or the end of the hold). See
[`trace-encoding.md`](trace-encoding.md).
