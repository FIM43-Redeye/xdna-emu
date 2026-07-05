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
        let segments = psp_load_map(image_len);

        // Segment A (low .text): served read-only by the ROM aperture at its offset.
        let mut bus = Bus::new_with_load_offset(image.bytes().to_vec(), segments[0].rom_load_offset());
        // Segment B (.rodata/.data/.text-tail): PSP-pre-loaded into the writable
        // 0x08b00000 data RAM. The firmware runs code and reads data here via
        // absolute 0x08b0xxxx pointers; it never copies these bytes at runtime.
        let seg_b = &segments[1];
        bus.preload_ram(seg_b.phys_base, &image.bytes()[seg_b.file_range()]);

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

/// Physical base and file start of PSP load segment B (the relocated
/// `.rodata`/`.data`/`.text`-tail). `phys = file + D` where
/// `D = SEG_B_PHYS_BASE - SEG_B_FILE_START = 0x08ad2f00` -- the same relocation
/// base the RE uses for `.rodata`/string-pointer recovery, found in M2c Phase 2
/// to be the code-relocation base too (25/25 indirect-call targets decode as
/// `entry` prologues under it). The PSP pre-loads this segment; the firmware
/// never copies it at runtime (zero writes to 0x08b00000 across the whole boot).
const SEG_B_PHYS_BASE: u32 = 0x08b0_0000;
const SEG_B_FILE_START: u32 = 0x0002_d100;

/// One placement in the PSP's multi-segment load of the firmware image.
struct PspSegment {
    /// Physical base the PSP places this segment at.
    phys_base: u32,
    /// File offset in the image where this segment's bytes begin.
    file_start: u32,
    /// Byte length of the segment.
    len: u32,
}

impl PspSegment {
    /// The ROM-aperture load-offset for a segment served read-only from the
    /// image directly: physical `P` reads image byte `P + offset`.
    fn rom_load_offset(&self) -> u32 {
        self.file_start.wrapping_sub(self.phys_base)
    }
    /// The image byte range this segment places (`file_start .. file_start+len`).
    fn file_range(&self) -> std::ops::Range<usize> {
        self.file_start as usize..(self.file_start + self.len) as usize
    }
}

/// The PSP load map: where the PSP places each part of the firmware image before
/// start. Segment A is the low `.text` (served read-only by the ROM aperture at
/// offset `L`); segment B is the relocated `.rodata`/`.data`/`.text`-tail,
/// pre-loaded into the writable `0x08b00000` data RAM. See the M2c design spec's
/// "Phase 2 amendment: the PSP loads the firmware as multiple segments".
fn psp_load_map(image_len: u32) -> [PspSegment; 2] {
    [
        PspSegment { phys_base: 0, file_start: PSP_LOAD_OFFSET, len: image_len - PSP_LOAD_OFFSET },
        PspSegment {
            phys_base: SEG_B_PHYS_BASE,
            file_start: SEG_B_FILE_START,
            len: image_len - SEG_B_FILE_START,
        },
    ]
}

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
    use crate::firmware::mmio::Region;

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
        let report = proc.boot_to_idle(200_000);
        eprintln!("=== M2c Phase 2 boot observation ===");
        eprintln!("instrs_executed  = {}", report.instrs_executed);
        eprintln!("last_pc          = {:#x}", report.last_pc);
        eprintln!("reached_idle     = {}", report.reached_idle);
        eprintln!("wait_reason      = {:?}", report.wait_reason);
        eprintln!("unresolved_spin  = {:?}", report.unresolved_spin.map(|a| format!("{a:#x}")));
        eprintln!("unknown_op       = {:?}", report.unknown_op.map(|(p, w)| format!("{p:#x}: {w:#010x}")));
        eprintln!("window_exceptions= {}", report.window_exceptions);
        eprintln!("funcs_entered    = {:?}", report.funcs_entered);

        // With the fill-loop fast-path the boot no longer grinds the 128 MiB
        // boot memset; it advances well past the region-zeroing routine (which
        // sits ~instr 23k). The exact next wall is under active investigation
        // (iter6); this stays an OBSERVATION test, asserting only that the boot
        // clears the region-init stretch, not a fixed idle gate.
        assert!(
            report.instrs_executed > 20_000 || report.reached_idle,
            "boot regressed short of the region-init routine ({} instrs) -- a map/fast-path regression",
            report.instrs_executed,
        );

        // iter12 (2026-07-05): the firmware's first SYSCALL now dispatches
        // correctly. The kernel exception vector's stub `l32r a3,=0x28b4; jx a3`
        // reads its dispatcher literal from the instruction-stream literal pool
        // (IRAM, via `l32r_load`), NOT the DRAM overlay (`local_data`) that the
        // boot's low-window memset (`fill 0x4..0xff0`) zeroes. So `a3` is the
        // real dispatcher, not 0, and the syscall services and returns instead
        // of jumping to PC=0 (the pre-fix iter12 wall). Pin that: the boot must
        // NOT wall at PC=0.
        //
        // (`main` now returns NORMALLY to the crt0 post-return site 0x2000e035,
        // an undecoded op 0x41f0 -- iter13's wall. That is the *correct* path,
        // unlike iter10's early unwind to the same address, which the >20k
        // instruction floor above already rules out.)
        assert_ne!(
            report.unknown_op.map(|(pc, _)| pc),
            Some(0x0000_0000),
            "boot walls at PC=0 -- the exception-vector l32r read the zeroed DRAM \
             overlay instead of the IRAM literal (last_pc={:#x})",
            report.last_pc,
        );
    }

    /// M2c Phase 2 boot-walk DIAGNOSTIC (not a correctness gate): arm the Bus
    /// stub-access probe and boot, so every Array/Mailbox/System access the
    /// firmware issues is captured with the PC that issued it. The suspected
    /// wrong-path source is a peripheral read that returns the stub 0 which the
    /// firmware then branches on. Prints a per-site summary (deduped by PC) in
    /// access order, plus the raw tail near the boot wall. Ignored unless
    /// XDNA_FW_PROBE is set, so the full suite stays fast.
    #[test]
    fn m2c_probe_peripheral_reads() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the peripheral-read characterization");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        proc.bus.arm_probe();

        // Step the boot manually so we can stamp each access with the issuing
        // PC and record the instruction index at which the run stops. Budget
        // covers past the memset wall (~23105 instrs).
        const MAX: u64 = 40_000;
        let mut n = 0u64;
        let mut stop = String::from("budget reached");
        while n < MAX {
            let pc = proc.cpu.pc;
            proc.bus.set_probe_pc(pc);
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    if proc.cpu.pc == pc {
                        stop = format!("idle Wait({reason:?}) at pc={pc:#x}");
                        break;
                    }
                }
                Step::Unknown { pc, word } => {
                    stop = format!("Unknown pc={pc:#x} word={word:#010x}");
                    break;
                }
            }
            if let Some(addr) = proc.bus.sysstub().spinning() {
                stop = format!("sysstub spin at {addr:#x}");
                break;
            }
        }
        let log = proc.bus.take_probe();

        eprintln!("=== M2c peripheral-read characterization ===");
        eprintln!("instrs executed = {n}");
        eprintln!("stop reason     = {stop}");
        eprintln!("total stub accesses = {}", log.len());

        // Per-site summary, deduped by (pc, is_write), in first-seen order. For
        // each site: region, the distinct addresses and values seen, access
        // count, and the seq of first occurrence.
        use std::collections::BTreeMap;
        let mut sites: BTreeMap<(u32, bool), (Region, Vec<u32>, Vec<u32>, u64, u64)> = BTreeMap::new();
        let mut order: Vec<(u32, bool)> = Vec::new();
        for a in &log {
            let key = (a.pc, a.is_write);
            let e = sites.entry(key).or_insert_with(|| {
                order.push(key);
                (a.region, Vec::new(), Vec::new(), a.seq, 0)
            });
            if !e.1.contains(&a.addr) {
                e.1.push(a.addr);
            }
            if !e.2.contains(&a.value) {
                e.2.push(a.value);
            }
            e.4 += 1;
        }
        order.sort_by_key(|k| sites[k].3);
        eprintln!("distinct sites = {}", order.len());
        eprintln!(
            "{:>6} {:<5} {:<7} {:<3} {:<28} {:<28} {}",
            "seq", "rd/wr", "region", "cnt", "addrs", "values", "pc"
        );
        for key in &order {
            let (region, addrs, values, first_seq, count) = &sites[key];
            let addr_s = addrs.iter().take(4).map(|a| format!("{a:#x}")).collect::<Vec<_>>().join(",");
            let val_s = values.iter().take(4).map(|v| format!("{v:#x}")).collect::<Vec<_>>().join(",");
            eprintln!(
                "{:>6} {:<5} {:<7} {:<3} {:<28} {:<28} {:#x}",
                first_seq,
                if key.1 { "wr" } else { "rd" },
                format!("{region:?}"),
                count,
                addr_s,
                val_s,
                key.0,
            );
        }

        // Raw tail: the last 40 accesses, to see what immediately precedes the wall.
        eprintln!("--- last 40 raw accesses (seq: pc region rd/wr addr=value) ---");
        for a in log.iter().rev().take(40).rev() {
            eprintln!(
                "{:>6}: pc={:#x} {:?} {} {:#x}={:#x} w{}",
                a.seq,
                a.pc,
                a.region,
                if a.is_write { "wr" } else { "rd" },
                a.addr,
                a.value,
                a.width,
            );
        }
    }

    /// M2c Phase 2 DIAGNOSTIC: run the boot until it walls (unknown-op / spin /
    /// idle) and dump a ring buffer of the last N instructions leading up to the
    /// stop -- translated disassembly + the full a0..a15 window each step. Finds
    /// the exact control-transfer instruction that lands on a bad target.
    /// Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_trace_to_wall() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the trace-to-wall probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        const MAX: u64 = 200_000;
        const KEEP: usize = 48;
        // Ring buffer of (instr_n, pc, disasm, a0..a15).
        let mut ring: std::collections::VecDeque<(u64, u32, String, [u32; 16])> =
            std::collections::VecDeque::with_capacity(KEEP + 1);
        let mut n = 0u64;
        let mut stop = String::from("budget reached");
        while n < MAX {
            let pc = proc.cpu.pc;
            let disasm = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                Ok(phys) => {
                    // Peek up to 8 bytes so FLIX bundles (op0 0xe/0xf) disassemble
                    // correctly in the trace, not just the first 3.
                    let b: [u8; 8] = std::array::from_fn(|i| proc.bus.peek8(phys.wrapping_add(i as u32)));
                    format!("{:?}", decode::decode(&b, pc).op)
                }
                Err(_) => "<fetch-fault>".to_string(),
            };
            let mut regs = [0u32; 16];
            for (r, slot) in regs.iter_mut().enumerate() {
                *slot = proc.cpu.regs.read_ar(r as u8);
            }
            if ring.len() == KEEP {
                ring.pop_front();
            }
            ring.push_back((n, pc, disasm, regs));

            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    if proc.cpu.pc == pc {
                        stop = format!("idle Wait({reason:?}) at pc={pc:#x}");
                        break;
                    }
                }
                Step::Unknown { pc: upc, word } => {
                    stop = format!("Unknown pc={upc:#x} word={word:#010x}");
                    break;
                }
            }
            if let Some(addr) = proc.bus.sysstub().spinning() {
                stop = format!("sysstub spin at {addr:#x}");
                break;
            }
        }

        eprintln!("=== M2c trace-to-wall (last {KEEP} instrs before the stop) ===");
        eprintln!("instrs executed = {n}");
        eprintln!("stop reason     = {stop}");
        eprintln!("cpu.vecbase     = {:#x}", proc.cpu.vecbase);
        for (i, pc, disasm, regs) in &ring {
            let lo: Vec<String> = (0..8).map(|r| format!("a{r}={:#x}", regs[r])).collect();
            eprintln!("{i:>6} pc={pc:#x} {disasm:<30} | {}", lo.join(" "));
        }
        // The full a0..a15 window of the last few instructions (call/window state).
        eprintln!("--- a8..a15 of the final {} instrs ---", 6.min(ring.len()));
        for (i, pc, _, regs) in ring.iter().rev().take(6).rev() {
            let hi: Vec<String> = (8..16).map(|r| format!("a{r}={:#x}", regs[r])).collect();
            eprintln!("{i:>6} pc={pc:#x} | {}", hi.join(" "));
        }
    }

    /// M2c Phase 2 DIAGNOSTIC: statically disassemble the low-ROM exception
    /// vector entries via our own decoder (which, unlike lx106 objdump, handles
    /// the windowed ops these vectors are built from) and resolve each `l32r`
    /// literal so `jx`-stub targets are readable. This is the tool that derived
    /// the corrected [`KERNEL_EXCEPTION_VECTOR_OFFSET`] (see the finding
    /// `docs/superpowers/findings/2026-07-05-iter7-exception-vector-offset.md`).
    ///
    /// The entries below were pinned from the firmware image (vecbase=0x800,
    /// confirmed from the prologue's own `wsr.vecbase` literal):
    /// - 0xae0 (vecbase+0x2e0): the Kernel/general-exception vector -- a stub
    ///   `wsr.excsave1 a3; l32r a3,=0x28b4; jx a3` that jumps to the real
    ///   exception dispatcher at runtime 0x28b4.
    /// - 0xb1c (vecbase+0x31c): the DoubleException handler -- inline
    ///   `wsr.excsave1/2/5/6; rsr.exccause; ...; rfde` (surfaces as `Unknown`
    ///   at the `rfde`, which our windowed-firmware decoder doesn't carry).
    ///
    /// Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_vector_table() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the vector-table probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let proc = FirmwareProcessor::load_m2c(img);

        const VECBASE: u32 = 0x800;
        const MAX_INSTRS: usize = 24;

        let peek3 = |phys: u32| {
            [proc.bus.peek8(phys), proc.bus.peek8(phys.wrapping_add(1)), proc.bus.peek8(phys.wrapping_add(2))]
        };
        let peek32 = |phys: u32| {
            u32::from_le_bytes([
                proc.bus.peek8(phys),
                proc.bus.peek8(phys.wrapping_add(1)),
                proc.bus.peek8(phys.wrapping_add(2)),
                proc.bus.peek8(phys.wrapping_add(3)),
            ])
        };

        eprintln!("=== M2c vector-table static disasm (vecbase {VECBASE:#x}) ===");
        // (entry, label). Extend when characterizing more of the vector table.
        for &(entry, label) in &[(0xae0u32, "kernel/general exc stub"), (0xb1c, "double exc handler")] {
            eprintln!("--- entry {entry:#x} (vecbase+{:#x}) -- {label} ---", entry - VECBASE);
            let mut pc = entry;
            for _ in 0..MAX_INSTRS {
                let b = peek3(pc);
                let d = decode::decode(&b, pc);
                // Resolve an L32r's literal address + value so `jx`-stub targets
                // are readable (l32r literal = ((pc+3)&~3) + sext(imm16)<<2).
                let extra = if let Op::L32r { target, .. } = d.op {
                    format!("  [lit@{target:#x} = {:#x}]", peek32(target))
                } else {
                    String::new()
                };
                eprintln!("  {pc:#06x}: {:02x} {:02x} {:02x}   {:?}{extra}", b[0], b[1], b[2], d.op);
                let terminal =
                    matches!(d.op, Op::Jx { .. } | Op::RetN | Op::Retw | Op::RetwN | Op::Unknown { .. });
                if terminal {
                    break;
                }
                pc = pc.wrapping_add(d.len as u32);
            }
        }
    }

    /// M2c Phase 2 DIAGNOSTIC: stop at the big memset's entry (phys 0x08b0e290)
    /// and dump its windowed arguments (a2=dest, a3=fill, a4=count) plus a ring
    /// buffer of the recent call8/callx8 history, to see how the boot reaches it
    /// and with what count. Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_memset_entry() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the memset-entry probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        const MEMSET_ENTRY: u32 = 0x08b0_e290;
        const MAX: u64 = 300_000;
        // Every memset entry: (instr_n, return_a0, dest, fill, count).
        let mut hits: Vec<(u64, u32, u32, u32, u32)> = Vec::new();
        let mut n = 0u64;
        let mut runaway: Option<(u64, u32, u32, u32, u32)> = None;
        while n < MAX {
            let pc = proc.cpu.pc;
            if pc == MEMSET_ENTRY {
                let rec = (
                    n,
                    proc.cpu.regs.read_ar(0),
                    proc.cpu.regs.read_ar(2),
                    proc.cpu.regs.read_ar(3),
                    proc.cpu.regs.read_ar(4),
                );
                hits.push(rec);
                // A count over 1 MiB is the runaway; stop so we don't grind it.
                if rec.4 > 0x10_0000 {
                    runaway = Some(rec);
                    break;
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(_) | Step::Unknown { .. } => break,
            }
        }

        eprintln!("=== M2c memset-entry probe (all invocations) ===");
        eprintln!("total memset entries seen = {}", hits.len());
        eprintln!("{:>8} {:>12} {:>12} {:>12} {:>12}", "instr", "a0(ret)", "dest", "fill", "count");
        for (i, a0, dest, fill, count) in &hits {
            eprintln!("{i:>8} {a0:>#12x} {dest:>#12x} {fill:>#12x} {count:>#12x}");
        }
        if let Some((i, a0, dest, fill, count)) = runaway {
            eprintln!(
                "--- RUNAWAY memset at instr {i}: dest={dest:#x} fill={fill:#x} count={count:#x} (ret a0={a0:#x}) ---"
            );
        } else {
            eprintln!("no runaway (>1MiB) memset within {MAX} instrs; last_pc={:#x}", proc.cpu.pc);
        }
    }

    /// M2c Phase 2 DIAGNOSTIC: trace the spin loop around instr 23089+ that
    /// repeatedly calls memset(0xe740, 0x1b8, 0). Runs to just before the spin,
    /// then single-steps ~60 instructions printing pc + decoded op + the
    /// current-window a-registers, to reveal what the loop branches on and why
    /// it never exits. Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_spin_loop() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the spin-loop trace");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        // Run to just before the spin sets in.
        const WARMUP: u64 = 23_060;
        for _ in 0..WARMUP {
            proc.cpu.step(&mut proc.bus);
        }
        eprintln!("=== M2c spin-loop trace (from instr {WARMUP}) ===");
        for i in 0..70u64 {
            let pc = proc.cpu.pc;
            // Translate the virtual PC to physical so peek reads the real
            // instruction bytes (peek8 on a virtual code address in the System
            // region would otherwise return 0 -> a phantom Unknown{word:0}).
            let disasm = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                Ok(phys) => {
                    let bytes = [
                        proc.bus.peek8(phys),
                        proc.bus.peek8(phys.wrapping_add(1)),
                        proc.bus.peek8(phys.wrapping_add(2)),
                    ];
                    format!("{:?}", decode::decode(&bytes, pc).op)
                }
                Err(_) => "<fetch-fault>".to_string(),
            };
            // Dump the low 8 window registers alongside each instruction.
            let a: Vec<String> = (0..8).map(|r| format!("a{r}={:#x}", proc.cpu.regs.read_ar(r))).collect();
            eprintln!("{:>5} pc={:#x} {:<30} | {}", WARMUP + i, pc, disasm, a.join(" "));
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => {}
                other => {
                    eprintln!("stop: {other:?}");
                    break;
                }
            }
        }
    }

    /// M2c Phase 2 iter 2: the PSP load map places segment B (the relocated
    /// `.rodata`/`.data`/`.text`-tail) at physical `0x08b00000`, pre-loaded (not
    /// copied at runtime). This asserts the load map places segment B's bytes at
    /// the right physical addresses.
    #[test]
    fn m2c_load_map_places_segment_b() {
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        // Segment B: phys 0x08b041f0 (= file 0x312f0) is the callx8 target that
        // walled iter 1; it must now hold `entry a1,0x60` (36 c1 00 -> 0x4c00c136).
        assert_eq!(proc.bus.load32(0x08b0_41f0), 0x4c00_c136, "segment B callx8 target not placed");
        // And memset's entry at phys 0x08b0e290 (= file 0x3b390): 36 41 00 -> 0x8c004136.
        assert_eq!(proc.bus.load32(0x08b0_e290), 0x8c00_4136, "segment B memset entry not placed");
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
