---
class: core-compute-timing
subsystem: compute-core memory-port / execution cycle timing
posture: mechanism AND per-iteration magnitude modelled + HW-confirmed (bank-arbitration arc); DMA bank-demand cadence HW-derived and fixed; DMA memory-pressure events closed by the staging-FIFO model; producer core-vs-DMA conflict count fixed (C-cycle arbiter, 2026-07-15), residual stall-count phase-average tail documented
status: scalar write-port collisions modelled (fixed 2026-11-03); scalar-side register widths derived and fixed (2026-11-03; status-flag producers open); cascade link modelled (2-word credit window, 2-cycle transit, same-cycle credit return, staggered core starts; 2026-11-04); 2 open (pipeline-fill depth; bank-arbiter resume-cycle hole, dormant/unreachable on corpus), 1 documented residual (producer stall-count phase-average tail below lockstep resolution, not chased), 4 root-caused-and-modelled
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
the release seam (`release_core_resets`). A core's `MOV_CNTR` reads engine cycles, so cores started
at different times read one timebase, and it writes both halves of its register pair (the EMU used
to leave the high register untouched; silicon writes the timer's high word, 0 here). Where the
start sits relative to the runtime sequence is the open `core-launch-timing` question
(host-firmware-dispatch.md); the order and spacing hold wherever it lands.

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

## Related: lock-arbitration cost (modeled) vs pulse emission (pending)

The per-lock-transaction arbitration **cycle cost** (HW trails every lock request
with a 1-cycle arbitration stall, even uncontended) is **modeled** in core cycle
timing (`f22e301c`, `cycle_accurate.rs`; HW-exact on the tenant-4 core loop
2125->2129, full bridge corpus clean). What remains open is the *trace-event* half
-- emitting the per-transaction `LOCK_STALL` pulse on the uncontended path -- which
is a trace-encoding concern, cross-referenced from the aiesim "Trace
stall/starvation micro-timing" row in `xdna-emu-priv: research/aiesim/aiesim-oracle.md` and
the count-under-emission row in [`trace-encoding.md`](trace-encoding.md).
