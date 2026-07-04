---
class: firmware-mmu
subsystem: Xtensa MMU-v3 model (management-firmware interpreter, src/firmware/xtensa/mmu.rs)
posture: needs-HW-empirical -- derived throughout from QEMU's `target/xtensa/mmu_helper.c`/`exc_helper.c` as the Xtensa ISA reference (no open-source AMD-side MMU spec exists); a couple of corners are deliberate simplifications that are unobservable in this model, and one is a genuinely open question
status: 2 deliberate simplifications (inert here, M2b); 1 open question blocking M2c confidence
---

# Firmware MMU Gaps

Xtensa MMU-v3 fidelity gaps in the M2b management-firmware interpreter (TLB
lookup, autorefill page-table walk, privilege/permission faults, double-fault
routing). Ground truth throughout M2b has been QEMU's Xtensa target as the ISA
reference; real-firmware behavior (traced instruction-by-instruction) is the
final oracle where the two could disagree.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| Double-fault writes EPC1 unconditionally | QEMU (`exc_helper.c` `HELPER(exception_cause)`) writes `sregs[DEPC]` on a double fault (PS.EXCM already set) when `config->ndepc` is set, else `EPC1`. This model's `raise_general_exception` always writes EPC1 and does not implement DEPC at all. | `src/firmware/xtensa/interp/mod.rs` (`Cpu::raise_general_exception`) | **Deliberate, inert.** A double fault is fatal to this firmware boot in practice (M0-M2 never reach one on the real boot path); DEPC support deferred until a scenario actually needs it. |
| EXCVADDR not set on successful autorefill | QEMU sets `sregs[EXCVADDR] = vaddr` on a successful autorefill (`mmu_helper.c:831`), even though no fault is raised. This model omits it. | `src/firmware/xtensa/mmu.rs` (`Mmu::translate_inner`, the refill arm) | **Deliberate, unobservable here.** A successful refill enters no exception handler, and EXCVADDR is only ever read from a fault handler, so the omission has no observable effect in this model. |
| varway56 (ways 5/6 fixed vs software-writable) | Model defaults `varway56=false`: MMU ways 5/6 are two hard-wired region-protection entries per way, and `witlb`/`wdtlb`/`iitlb`/`idtlb` targeting them are silent no-ops. Real firmware's own boot prologue issues exactly these calls (AS bits 0-2 = 5 for `witlb`/`wdtlb`, AS bits = 6 for seven `iitlb`/`idtlb` calls) -- confirmed by stepping it. **Open question:** if the real AMD core configures `varway56=true`, those calls install the firmware's own high-region map, which this model cannot represent under the current default at all. | `src/firmware/xtensa/mmu.rs` (`Mmu::new`, `ways_5_and_6_hold_fixed_region_entries`) | **OPEN, central to M2c.** The firmware behaves as if it expects those writes to matter; whether they do on real silicon is unresolved. See [`2026-07-04-m2b-autorefill-characterization.md`](../superpowers/findings/2026-07-04-m2b-autorefill-characterization.md). |
