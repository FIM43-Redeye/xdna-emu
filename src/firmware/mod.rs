//! In-tree base-Xtensa interpreter that runs the real NPU management firmware.
//!
//! Phase M0+M1 scope: load the `$PS1` image and boot it to a command-loop idle.
//! Device/mailbox MMIO routing into `DeviceState` is later (M2).

mod error;
mod image;
mod mmio;
mod psp_map;
mod sysstub;
pub mod xtensa;

pub use error::FirmwareError;
pub use image::FirmwareImage;
pub use mmio::Bus;
pub use sysstub::SysStub;

use std::collections::HashMap;
use std::path::Path;

use xtensa::decode::{self, Op};
use xtensa::interp::{Cpu, Step, WaitReason, CAUSE_WINDOW_OVERFLOW, CAUSE_WINDOW_UNDERFLOW};

/// A loaded firmware ready to run: the Xtensa interpreter core, its routed
/// MMIO bus over the firmware image, and the entry PC boot begins at.
pub struct FirmwareProcessor {
    /// The interpreter core (PC + windowed register file + VECBASE/EPC1).
    pub cpu: Cpu,
    /// The routed memory/MMIO bus over the firmware's base-0 image.
    pub bus: Bus,
    /// The entry PC `boot_to_idle` starts stepping from.
    pub entry: u32,
    /// Recovered `addr -> name` symbol map (empty if `symbols.txt` is absent);
    /// used to name the `call8`/`callx8` targets in [`IdleReport::funcs_entered`].
    symbols: HashMap<u32, String>,
}

/// The outcome of a [`FirmwareProcessor::boot_to_idle`] run: how far the
/// firmware got and why it stopped. The milestone-M1.7 observation record.
#[derive(Debug, Clone)]
pub struct IdleReport {
    /// True iff the run stopped because the firmware reached a stable idle
    /// wait (`Step::Wait` at an unchanging PC) -- the command-loop idle.
    pub reached_idle: bool,
    /// Instructions executed before the run stopped.
    pub instrs_executed: u64,
    /// The wait reason, if the run stopped on a `Step::Wait`.
    pub wait_reason: Option<WaitReason>,
    /// The `(addr, name)` of every `call8`/`callx8` whose target matched the
    /// recovered symbol map, in call order.
    pub funcs_entered: Vec<(u32, String)>,
    /// `Some(addr)` if the run stopped because [`SysStub::spinning`] flagged a
    /// tight poll on an unmodeled system-aperture address.
    pub unresolved_spin: Option<u32>,
    /// `Some((pc, word))` if the run stopped on an unimplemented/undecodable
    /// opcode -- the raw fetched bytes for oracle disassembly.
    pub unknown_op: Option<(u32, u32)>,
    /// Count of window overflow/underflow exceptions raised during the run
    /// (the H1-dormant vs H2-fires signal for M2).
    pub window_exceptions: u64,
    /// The PC at the moment the run stopped.
    pub last_pc: u32,
}

impl FirmwareProcessor {
    /// Load `image` into a fresh bus and point the CPU at `entry`. Attempts to
    /// load the recovered symbol map from the firmware-RE experiment dir (for
    /// naming entered functions); a missing map is not an error.
    pub fn load(image: FirmwareImage, entry: u32) -> Self {
        let bus = Bus::new(image.bytes().to_vec());
        let mut cpu = Cpu::new(entry);

        // PROVISIONAL boot-time identity map for the low ROM region (M2b
        // Task 8, pending M2c). Hardware fact, not a guess: the M1.7 boot
        // observation (this same test, before Task 8 wired fetch through the
        // MMU) proved the reset head + MMU-setup prologue at `entry`
        // (`~0x200..0x399`, all low ROM addresses) executes correctly with
        // vaddr==phys -- `image.rs`'s own doc already establishes this for
        // the base-0 `.text`/`.rodata` segment ("file offset == link
        // address"). Something (the PSP, before this firmware even starts)
        // must establish that identity view on real hardware for the reset
        // vector to be fetchable at all; we don't have that artifact, so we
        // model its OBSERVED EFFECT rather than inventing its mechanism.
        // Covers a full 1MB (way 4's default page size, ITLBCFG/DTLBCFG==0)
        // so it comfortably spans the reset head, the `0x320..0x399`
        // prologue, and its nearby literal pool without needing to chase
        // exact page boundaries.
        //
        // Way 4, not 0-3 or 5/6: ways 0-3 are the hardware autorefill ways
        // (`Mmu::refill`) -- a real page-table walk (once M2c reconstructs
        // one) could silently evict this entry there. Ways 5/6 are the fixed
        // region-protection ways (`Mmu::load_fixed_ways56`) and refuse
        // software writes outright. Way 4 is variable, outside the
        // autorefill round-robin, and -- confirmed by stepping the real
        // firmware's own prologue -- untouched by it: its `witlb`/`wdtlb`
        // (AS bits 0-2 = 5) and seven `iitlb`/`idtlb` calls (AS bits = 6) all
        // target ways 5/6, which are fixed and so are themselves no-ops
        // against the current MMU model (see the M2b Task 8 report for the
        // full trace -- worth a closer look for M2c, since it means the
        // firmware's own high-region mapping attempt currently has no
        // effect at all).
        let low_page = entry & 0xfff0_0000; // way-4 1MB page containing `entry`
        cpu.mmu.write_tlb(false, low_page | 0x1, low_page | 4); // ITLB: R+X
        cpu.mmu.write_tlb(true, low_page | 0x3, low_page | 4); // DTLB: RWX

        let symbols = load_symbols();
        Self { cpu, bus, entry, symbols }
    }

    /// Load `image` for the M2c boot-to-idle path: PSP load-offset, varway56=true,
    /// synthesized code-region page table, starting at the physical reset entry.
    pub fn load_m2c(image: FirmwareImage) -> Self {
        let image_len = image.bytes().len() as u32;
        let mut bus = Bus::new_with_load_offset(image.bytes().to_vec(), PSP_LOAD_OFFSET);
        let mut cpu = Cpu::new(RESET_ENTRY);
        cpu.mmu = xtensa::mmu::Mmu::new_with_varway56(true);

        // NO provisional low-region map here (unlike the M2b `load` path). With
        // varway56=true the reset populates way-6 entry 0 as an identity region
        // 0..0x1fffffff, attr 3 (RWX), which already covers the reset head and
        // prologue's low physical addresses -- and the firmware's own prologue
        // leaves way-6 entry 0 alone (it invalidates entries 1..7 only). Adding a
        // separate provisional entry would MULTI-HIT against way-6 entry 0 and fault
        // (cause 17) on the very first fetch. The way-6 reset identity IS the
        // low-region map the PSP established; we do not re-invent it.

        // The prologue programs PTEVADDR/DTLBCFG itself (to these exact values), but
        // the synth PT install below needs them now to place the region entry (its
        // way-4 page size is read from DTLBCFG) and the PTEs. Setting them early is
        // consistent: the prologue re-writes the identical values.
        cpu.mmu.ptevaddr = 0x3c00_0000;
        cpu.mmu.dtlbcfg = 0x0003_0000;
        psp_map::install(&mut cpu.mmu, &mut bus, PSP_LOAD_OFFSET, image_len);

        let symbols = load_symbols();
        Self { cpu, bus, entry: RESET_ENTRY, symbols }
    }

    /// Step until the CPU reaches the code at file offset `file_target` (returns
    /// true), or `max` instructions pass / the run stops (returns false). Phase-1
    /// coherence probe. The mapping has a load-offset: pre-paging the PC runs at low
    /// physical `file - L`; post-paging it runs in the code region's virtual space,
    /// `virtual = CODE_REGION_BASE + (file - L)`. Match both (the C entry is
    /// post-paging, so `virt_alias` is the one that fires there).
    pub fn reaches_pc(&mut self, file_target: u32, max: u64) -> bool {
        let phys_target = file_target.wrapping_sub(PSP_LOAD_OFFSET);
        let virt_alias = crate::firmware::psp_map::CODE_REGION_BASE.wrapping_add(phys_target);
        for _ in 0..max {
            if self.cpu.pc == phys_target || self.cpu.pc == virt_alias {
                return true;
            }
            match self.cpu.step(&mut self.bus) {
                Step::Ran | Step::Exception { .. } => {}
                Step::Wait(_) | Step::Unknown { .. } => return false,
            }
        }
        false
    }

    /// Step the firmware from its entry until one of four things happens:
    /// (a) a `Step::Wait` at a stable PC (idle -- `reached_idle`),
    /// (b) [`SysStub::spinning`] fires (`unresolved_spin`),
    /// (c) a `Step::Unknown` unimplemented opcode (`unknown_op`), or
    /// (d) `max_instrs` is exceeded.
    ///
    /// Records every `call8`/`callx8` into a named function (per the symbol
    /// map) in `funcs_entered`, and counts window exceptions raised.
    pub fn boot_to_idle(&mut self, max_instrs: u64) -> IdleReport {
        let mut funcs_entered = Vec::new();
        let mut window_exceptions = 0u64;
        let mut instrs_executed = 0u64;
        let mut reached_idle = false;
        let mut wait_reason = None;
        let mut unresolved_spin = None;
        let mut unknown_op = None;

        while instrs_executed < max_instrs {
            let pc = self.cpu.pc;

            // Peek (no side effects) to record a call into a named function
            // before the CPU consumes the instruction.
            let bytes =
                [self.bus.peek8(pc), self.bus.peek8(pc.wrapping_add(1)), self.bus.peek8(pc.wrapping_add(2))];
            let call_target = match decode::decode(&bytes, pc).op {
                Op::Call8 { target } => Some(target),
                Op::Callx8 { s } => Some(self.cpu.regs.read_ar(s)),
                _ => None,
            };

            let step = self.cpu.step(&mut self.bus);

            match step {
                // Executed instructions (including a raised fault) count; an
                // Unknown did not execute (pc is left unchanged), so it is a
                // stop reason, not an executed instruction.
                Step::Ran | Step::Wait(_) | Step::Exception { .. } => instrs_executed += 1,
                Step::Unknown { .. } => {}
            }

            if let Some(target) = call_target {
                if let Some(name) = self.symbols.get(&target) {
                    funcs_entered.push((target, name.clone()));
                }
            }

            match step {
                Step::Ran => {}
                Step::Wait(reason) => {
                    // A wait that doesn't move the PC is a stable idle.
                    if self.cpu.pc == pc {
                        reached_idle = true;
                        wait_reason = Some(reason);
                        break;
                    }
                }
                Step::Exception { cause, .. } => {
                    // Only a REAL window overflow/underflow counts here --
                    // `Step::Exception` is also the general-exception/MMU-
                    // fault channel (M2a Task 9 / M2b Task 7-8), so an
                    // unrelated cause (e.g. an ITLB miss past the mapped
                    // region) must not inflate this counter; see
                    // `IdleReport::window_exceptions`'s own doc.
                    if cause == CAUSE_WINDOW_OVERFLOW || cause == CAUSE_WINDOW_UNDERFLOW {
                        window_exceptions += 1;
                    }
                }
                Step::Unknown { pc, word } => {
                    unknown_op = Some((pc, word));
                    break;
                }
            }

            // A tight poll on an unmodeled system register: the firmware is
            // waiting on hardware state this phase does not simulate.
            if let Some(addr) = self.bus.sysstub().spinning() {
                unresolved_spin = Some(addr);
                break;
            }
        }

        IdleReport {
            reached_idle,
            instrs_executed,
            wait_reason,
            funcs_entered,
            unresolved_spin,
            unknown_op,
            window_exceptions,
            last_pc: self.cpu.pc,
        }
    }
}

/// The PSP load-offset: physical `P` in the ROM aperture reads image byte
/// `P + PSP_LOAD_OFFSET` (`phys = file - PSP_LOAD_OFFSET`). Pinned by the M2c
/// Phase 1 coherence gate (`m2c_boot_reaches_c_entry`): the value that makes the
/// `jx 0x20000340` target land on the coherent continuation at file 0x39c.
/// Candidate 0x5c (0x39c - 0x340); confirmed by the gate. Hardware fact: the
/// x86 PSP loads the firmware body at this physical base before start.
const PSP_LOAD_OFFSET: u32 = 0x5c;

/// The physical reset entry: the reset vector at file 0x200 sits at physical
/// `0x200 - PSP_LOAD_OFFSET`. Boot begins here and the reset head `j`-es to the
/// MMU prologue.
const RESET_ENTRY: u32 = 0x200 - PSP_LOAD_OFFSET;

/// Load the recovered symbol map (`0xADDR\tNAME` per line) from the firmware-RE
/// experiment dir, if present. Absent file or unparsable lines yield an empty
/// (or partial) map rather than an error -- symbol names are a diagnostic aid.
fn load_symbols() -> HashMap<u32, String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("build/experiments/firmware-re/symbols.txt");
    let mut map = HashMap::new();
    let Ok(text) = std::fs::read_to_string(&path) else {
        return map;
    };
    for line in text.lines() {
        let mut cols = line.split('\t');
        let (Some(addr), Some(name)) = (cols.next(), cols.next()) else {
            continue;
        };
        let addr = addr.trim().strip_prefix("0x").unwrap_or(addr.trim());
        if let Ok(a) = u32::from_str_radix(addr, 16) {
            map.insert(a, name.trim().to_string());
        }
    }
    map
}

/// Locate the real firmware binary for firmware-gated tests: an
/// `XDNA_FIRMWARE` env override first, then the known repo-relative download
/// location. `None` if neither exists -- the binary is not checked into the
/// repo, so callers skip cleanly rather than failing. Shared by
/// `boot_tests` (below) and `xtensa::coverage_scan` (M2a Task 10).
#[cfg(test)]
pub(crate) fn firmware_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("XDNA_FIRMWARE") {
        let p = std::path::PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin");
    p.exists().then_some(p)
}

/// The pinned boot entry: the first instruction of the MMU-init reset
/// routine (`movi.n a2,0`), derived by coherence in M1.7 (see the
/// module-level M1.7 report for how it was pinned). Shared by `boot_tests`
/// (below) and `xtensa::coverage_scan`'s boot-prologue scan (M2a Task 10),
/// which independently confirms (via `objdump` on the raw image) that this
/// entry runs exactly 42 instructions before the `jx` into virtual space at
/// `0x399`.
#[cfg(test)]
pub(crate) const BOOT_ENTRY: u32 = 0x320;

#[cfg(test)]
mod boot_tests {
    use super::*;

    #[test]
    fn boots_real_firmware_from_pinned_entry() {
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");

        let mut proc = FirmwareProcessor::load(img, BOOT_ENTRY);
        let report = proc.boot_to_idle(5_000_000);

        eprintln!("=== M1.7 boot observation ===");
        eprintln!("entry            = {:#x}", proc.entry);
        eprintln!("instrs_executed  = {}", report.instrs_executed);
        eprintln!("last_pc          = {:#x}", report.last_pc);
        eprintln!("reached_idle     = {}", report.reached_idle);
        eprintln!("wait_reason      = {:?}", report.wait_reason);
        eprintln!("unresolved_spin  = {:?}", report.unresolved_spin);
        eprintln!("unknown_op       = {:?}", report.unknown_op.map(|(p, w)| format!("{p:#x}: {w:#08x}")));
        eprintln!("window_exceptions= {}", report.window_exceptions);
        eprintln!("funcs_entered    = {:?}", report.funcs_entered);

        // Coherence assertion (the entry-pinning check): from BOOT_ENTRY the
        // interpreter decodes and runs a coherent MMU-setup stream -- it does
        // NOT desync into Unknown within the first handful of instructions.
        // The prologue is 0x320..0x399 (movi.n/wsr/witlb/wdtlb/iitlb/idtlb/or/
        // dsync/isync/l32r) before the `jx` into virtual space at 0x399.
        assert!(
            report.instrs_executed > 20,
            "entry {BOOT_ENTRY:#x} desynced early: only {} instrs, last_pc={:#x}, unknown={:?}",
            report.instrs_executed,
            report.last_pc,
            report.unknown_op,
        );

        // The prologue is straight-line MMU setup: no windowed calls, so H1
        // (window overflow dormant) holds across everything observable this
        // phase -- H2 (overflow fires) cannot be reached before the MMU wall.
        assert_eq!(report.window_exceptions, 0, "no window exception in the boot prologue");
    }

    /// M2b Task 10 (#140): an OBSERVATION run, not a pass/fail correctness
    /// test. Boots the real firmware with the now-live MMU (M2b Tasks 1-9)
    /// and records what the autorefill mechanism actually computes at the
    /// `jx` target, plus the operands of every `witlb`/`wdtlb`/`iitlb`/
    /// `idtlb` the boot prologue issues -- the empirical starting point for
    /// M2c's page-table-data reconstruction. See
    /// `docs/superpowers/findings/2026-07-04-m2b-autorefill-characterization.md`
    /// for the write-up this test's output feeds.
    #[test]
    fn characterize_real_firmware_autorefill() {
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load(img, BOOT_ENTRY);

        // One recorded `witlb`/`wdtlb`/`iitlb`/`idtlb`: the AS operand (way
        // index + VPN) and, for the two install ops, the AT operand
        // (paddr|attr) -- the firmware's own INTENDED region map. This is
        // the concrete artifact M2c needs (Task 8 already found these all
        // target fixed ways 5/6, so they're currently no-ops against this
        // MMU model -- the central M2c "varway56" question).
        struct TlbOp {
            pc: u32,
            mnemonic: &'static str,
            way: u32,
            as_: u32,
            at: Option<u32>,
        }
        let mut tlb_ops: Vec<TlbOp> = Vec::new();

        const MAX_STEPS: u32 = 200;
        let mut n = 0u32;
        let stop_reason = loop {
            if n >= MAX_STEPS {
                break format!("step cap ({MAX_STEPS}) reached, stuck at pc={:#x}", proc.cpu.pc);
            }

            let pc = proc.cpu.pc;
            // Peek (no side effects), same pattern as
            // `coverage_scan::zero_unknown_in_boot_prologue`'s `is_jx` check
            // -- witlb/wdtlb/iitlb/idtlb only READ their AR operands, so
            // recording them before the step executes is equivalent to
            // after, but keeps the established "peek, then step" shape.
            let peek =
                [proc.bus.peek8(pc), proc.bus.peek8(pc.wrapping_add(1)), proc.bus.peek8(pc.wrapping_add(2))];
            match decode::decode(&peek, pc).op {
                Op::Witlb { t, s } => tlb_ops.push(TlbOp {
                    pc,
                    mnemonic: "witlb",
                    way: proc.cpu.regs.read_ar(s) & 0x7,
                    as_: proc.cpu.regs.read_ar(s),
                    at: Some(proc.cpu.regs.read_ar(t)),
                }),
                Op::Wdtlb { t, s } => tlb_ops.push(TlbOp {
                    pc,
                    mnemonic: "wdtlb",
                    way: proc.cpu.regs.read_ar(s) & 0xf,
                    as_: proc.cpu.regs.read_ar(s),
                    at: Some(proc.cpu.regs.read_ar(t)),
                }),
                Op::Iitlb { s } => tlb_ops.push(TlbOp {
                    pc,
                    mnemonic: "iitlb",
                    way: proc.cpu.regs.read_ar(s) & 0x7,
                    as_: proc.cpu.regs.read_ar(s),
                    at: None,
                }),
                Op::Idtlb { s } => tlb_ops.push(TlbOp {
                    pc,
                    mnemonic: "idtlb",
                    way: proc.cpu.regs.read_ar(s) & 0xf,
                    as_: proc.cpu.regs.read_ar(s),
                    at: None,
                }),
                _ => {}
            }

            let step = proc.cpu.step(&mut proc.bus);

            // Same counting convention as `FirmwareProcessor::boot_to_idle`:
            // an executed instruction (including one that raises a fault)
            // counts; `Step::Unknown` did not execute (pc unchanged), so it's
            // a stop reason, not an executed instruction.
            match step {
                Step::Ran => {
                    n += 1;
                }
                Step::Exception { cause, pc: vector_pc } => {
                    n += 1;
                    break format!("Exception cause={cause} vector_pc={vector_pc:#x}");
                }
                Step::Wait(reason) => {
                    n += 1;
                    break format!("Wait({reason:?})");
                }
                Step::Unknown { pc, word } => break format!("Unknown pc={pc:#x} word={word:#010x}"),
            }
        };

        // The boot must reach the wall via the live MMU: an ITLB-miss
        // Exception (cause 16, INST_TLB_MISS) raised by the `jx` target's
        // fetch fault. Checked BEFORE reading `excvaddr` below, since that
        // field is only meaningful as "the jx target" once this specific
        // fault path is confirmed to be what actually happened -- a
        // step-cap timeout or a Wait would leave `excvaddr` holding
        // something else (or its zeroed reset value).
        assert!(
            stop_reason.starts_with("Exception cause=16"),
            "expected the boot to stop at the jx target's ITLB miss (cause 16, INST_TLB_MISS) -- a \
             different outcome means cpu.excvaddr below would not be the jx-target vaddr this \
             characterization assumes: {stop_reason}",
        );

        // The autorefill anchor numbers (`get_pte`, `mmu.rs`). PTEVADDR and
        // the faulting vaddr are both LIVE-READ off the CPU -- `mmu.ptevaddr`
        // (programmed by the prologue's own `wsr.ptevaddr`) and
        // `cpu.excvaddr` (set by `Cpu::translate`'s fault path, interp/mod.rs,
        // to whatever vaddr actually faulted -- NOT assumed from static
        // analysis of the prologue's `jx` operand). `pt_vaddr` is then
        // computed from those two live values by the same formula production
        // code uses (`get_pte`). `pt_lookup` is a read-only probe of whether
        // anything in the DTLB actually covers the computed address (it
        // doesn't -- the firmware's own high-region witlb targeted fixed
        // ways 5/6, which never took per the loop above).
        let ptevaddr = proc.cpu.mmu.ptevaddr;
        let jx_target = proc.cpu.excvaddr;
        let pt_vaddr = (ptevaddr | (jx_target >> 10)) & !3;
        let pt_lookup = proc.cpu.mmu.lookup(pt_vaddr, true);

        eprintln!("=== M2b Task 10: real-firmware autorefill characterization ===");
        eprintln!("instructions executed = {n}");
        eprintln!("stop reason           = {stop_reason}");
        eprintln!("last_pc               = {:#x}", proc.cpu.pc);
        eprintln!("PTEVADDR              = {ptevaddr:#x}");
        eprintln!("jx target (excvaddr)  = {jx_target:#x}");
        eprintln!("computed pt_vaddr     = {pt_vaddr:#x}");
        eprintln!("pt_vaddr DTLB lookup  = {pt_lookup:?}");
        eprintln!("firmware TLB-setup operands during the prologue:");
        for op in &tlb_ops {
            eprintln!("  {:#x}: {} way={} AS={:#x} AT={:?}", op.pc, op.mnemonic, op.way, op.as_, op.at);
        }

        // Sanity checks only -- NOT a boot-success assertion. M2b is not
        // expected to get past the wall (M2c supplies the missing
        // page-table data); these confirm the *mechanism* ran faithfully.
        assert_eq!(
            ptevaddr, 0x3c00_0000,
            "boot prologue should program PTEVADDR via wsr.ptevaddr to 0x3c000000 -- if this drifts, the \
             M2c pt_vaddr derivation (docs/superpowers/specs/2026-07-04-m2b-mmu-mechanism-design.md) needs \
             re-deriving",
        );
        assert_eq!(
            jx_target, 0x2000_0340,
            "the jx target (live-read from cpu.excvaddr, not a literal) drifted from the known boot-\
             prologue jx destination -- re-derive the M2c pt_vaddr math (docs/superpowers/specs/2026-07-04-\
             m2b-mmu-mechanism-design.md) if this genuinely changed",
        );
        assert!(
            !tlb_ops.is_empty(),
            "boot prologue issued no witlb/wdtlb/iitlb/idtlb -- expected several (already observed during \
             M2a/M2b); an empty list means the boot desynced before reaching them",
        );
        assert!(
            tlb_ops.iter().all(|op| op.way == 5 || op.way == 6),
            "expected every boot TLB-setup op to target fixed ways 5/6 (Task 8 finding); a different way \
             changes the M2c varway56 framing entirely: {:#?}",
            tlb_ops.iter().map(|op| (op.pc, op.mnemonic, op.way)).collect::<Vec<_>>(),
        );
        assert!(
            pt_lookup.is_err(),
            "expected the PTE address {pt_vaddr:#x} to be unmapped (the firmware's own high-region map \
             never took) -- a hit here would mean something now covers it and the wall should have moved",
        );
    }

    /// M2c Phase 2 boot observation harness: boots from the reset entry via
    /// `load_m2c` and prints the full `IdleReport` -- the instrument each Phase 2
    /// walk-and-stub iteration reports against (the current wall's PC / stop
    /// reason). It is NOT yet the idle gate (Phase 2 is not complete); the only
    /// assertion is a regression guard that the boot still advances at least into
    /// the C runtime (past the C entry at virtual 0x2000e024), so a change that
    /// regresses the Phase 1 map is caught here. When Phase 2 reaches idle, this
    /// hardens into the `reached_idle` gate.
    #[test]
    fn m2c_boot_advances_into_c_runtime() {
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        let report = proc.boot_to_idle(2_000_000);
        eprintln!("=== M2c Phase 2 boot observation ===");
        eprintln!("instrs_executed  = {}", report.instrs_executed);
        eprintln!("last_pc          = {:#x}", report.last_pc);
        eprintln!("reached_idle     = {}", report.reached_idle);
        eprintln!("wait_reason      = {:?}", report.wait_reason);
        eprintln!("unresolved_spin  = {:?}", report.unresolved_spin.map(|a| format!("{a:#x}")));
        eprintln!("unknown_op       = {:?}", report.unknown_op.map(|(p, w)| format!("{p:#x}: {w:#010x}")));
        eprintln!("window_exceptions= {}", report.window_exceptions);
        eprintln!("funcs_entered    = {:?}", report.funcs_entered);

        // Regression guard: the boot must still execute through all of Phase 1
        // into the C runtime. Reaching the C entry (virtual 0x2000e024) takes
        // ~2013 instructions; a Phase 1 map regression walls in the low hundreds.
        // Use the instruction count, not last_pc: as Phase 2 walls clear, the
        // furthest PC can jump to a NUMERICALLY LOWER address (e.g. an indirect
        // call into another code view), but instrs_executed only ever grows,
        // since clearing a wall means strictly more instructions run before the
        // next stop.
        assert!(
            report.instrs_executed > 1900,
            "boot regressed to only {} instrs (short of the ~2013 to reach the C runtime) -- Phase 1 map broke",
            report.instrs_executed,
        );
    }

    /// M2c Phase 1 coherence gate: with the load-offset, varway56, and the synth
    /// PT in place, the real firmware boots from the reset entry past the MMU wall,
    /// through the way-5 teardown and data-copy, to the C entry (`call0 0xe080`).
    /// This test PINS the load-offset L: it passes iff L makes the continuation
    /// coherent. If it fails, `last_pc` / `funcs_entered` localize the correct L.
    #[test]
    fn m2c_boot_reaches_c_entry() {
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        // Record whether the boot reaches the C-entry call (file 0xe080). The C
        // entry is reached via the continuation after the way-5 teardown, so
        // reaching it proves the whole code-region map is coherent.
        let reached = proc.reaches_pc(0xe080, 200_000);
        eprintln!("m2c boot: reached C entry (0xe080) = {reached}, last_pc = {:#x}", proc.cpu.pc);
        assert!(
            reached,
            "boot did not reach the C entry; last_pc={:#x} -- L or the map is wrong",
            proc.cpu.pc
        );
    }
}
