---
class: core-compute-timing
subsystem: compute-core memory-port / execution cycle timing
posture: needs-HW-empirical -- the disagreement is a real per-cycle cost the model omits, but why HW charges it is uncharacterized and requires HW calibration
status: 2 open (1 own campaign HW-calibration-heavy; 1 dormant/unreachable on corpus)
---

# Core Compute Timing Gaps

Gaps in the compute core's per-cycle timing model -- specifically memory-port
stalls the hardware charges that the emulator does not. The core's *instruction
issue* timing (bundle-issue rate, lock function-call overhead) is faithful and
HW-validated; these gaps are in the memory-port / stall accounting layered on top.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| Compute-core MEMORY_STALL: per-access stall not modeled (count + cycle cost) | A memory-heavy compute loop (`eo[i]=ei[i]+1`, 32 elts x 16 reps) on real NPU1 fires **MEMORY_STALL ~220-244x per consumer core**; EMU fires **0**. EMU only models the intra-bundle structural bank-conflict pulse (`event_trace.rs:440`), not the broad per-access stall HW emits. **It is a cycle-cost gap, not cosmetic:** consA HW active span exceeds EMU's by +235cy ~= its un-modeled stall count (220 MEMORY_STALL + 17 extra LOCK_STALL), i.e. ~1 HW cycle per missing stall; the deficit back-pressures upstream (producer span +1235cy). Diagnosed via of_q0_rich EMU-vs-HW trace (PERF_CNT_2 interval 1024 both -> no clock-rate divergence; span deficit == stall count). | EMU core memory-port timing model; intra-bundle pulse `src/interpreter/state/event_trace.rs:440`; evidence `build/experiments/sp3-spike-trace/SP4A-EMU-FINDING.md` | **OPEN (own campaign, #140-discovered, 2026-06-29).** Why HW stalls ~220x in a simple add loop is unresolved (likely core-port vs DMA bank contention or the compiler's vectorized memory schedule) -- deep memory-port cycle-accuracy work, HW-calibration-heavy, deliberately NOT folded into the timer-sync arc (SP-4a re-scoped to a lean lock/DMA kernel that avoids memory-heavy compute, where within-domain timing is already exact). Lock cycle-cost IS modeled (`737f5505`); only the lock *pulse* count is cosmetic. Bank before Phoenix swap if pursued. |
| Bank-arbiter resume-cycle hole: one un-arbitrated bank touch on lock/DMA/stream resume | The per-physical-bank round-robin arbiter (bank-arbitration arc, `664f824e`) runs in the coordinator's Phase B demand peek (`CoreInterpreter::peek_bank_demand`), which declares nothing for a core in `WaitingLock`/`WaitingDma`/`WaitingStream` (`src/interpreter/core/interpreter.rs` ~198-205). But `try_resume_stall` (same file, ~858) resolves the stall INSIDE `step_internal`, on the commit path, one call after that cycle's peek already ran -- so the exact cycle a stalled core resumes, it decodes and executes its bundle having declared no bank demand for that cycle. The access reaches memory without ever reaching the arbiter: a **missed conflict, never a double grant** -- strictly an under-count of contention, never an over-count, and nothing is corrupted (EMU memory is not physically single-port). Measured unreachable on the validation corpus: disassembling 60 core ELFs (chess + peano, including the objectfifo kernels used for bridge validation) found **zero** bundles pairing a blocking op (`acq`, stream/cascade ops) with a load/store slot -- Peano always emits the acquire as its own out-of-line function (`acq r0, r1` / `ret lr`), a lone bundle with no memory slot, so nothing is ever both resuming AND touching a bank in the same bundle on the current corpus. | `src/interpreter/core/interpreter.rs`: `peek_bank_demand`'s `WaitingLock`/`WaitingDma`/`WaitingStream` exclusion (~198-205), `try_resume_stall` (~858) | **OPEN, dormant (task 6 review, bank-arbitration arc, 2026-07).** Zero bank touches leak on the current corpus (measured, above), so this is not an active fidelity regression today -- but a VLIW bundle CAN legally carry both a blocking op and a memory slot, so the hole is real, just unexercised. Fix is a non-mutating `can_resume_stall()` predicate consulted in the coordinator's Phase B demand peek instead of inside `step_internal`'s commit path; it needs the neighbour-lock snapshot for the `WaitingLock` arm (the only new plumbing) to evaluate the resume condition without mutating state. NOT implemented -- flagged for whoever next touches the stall-resume machinery, or if a kernel corpus ever produces a bundle that trips it. |

## Related: lock-arbitration cost (modeled) vs pulse emission (pending)

The per-lock-transaction arbitration **cycle cost** (HW trails every lock request
with a 1-cycle arbitration stall, even uncontended) is **modeled** in core cycle
timing (`737f5505`, `cycle_accurate.rs`; HW-exact on the tenant-4 core loop
2125->2129, full bridge corpus clean). What remains open is the *trace-event* half
-- emitting the per-transaction `LOCK_STALL` pulse on the uncontended path -- which
is a trace-encoding concern, cross-referenced from the aiesim "Trace
stall/starvation micro-timing" row in [`aiesim-oracle.md`](aiesim-oracle.md) and
the count-under-emission row in [`trace-encoding.md`](trace-encoding.md).
