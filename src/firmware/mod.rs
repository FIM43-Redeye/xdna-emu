//! In-tree base-Xtensa interpreter that runs the real NPU management firmware.
//!
//! Phase M0+M1 scope: load the `$PS1` image and boot it to a command-loop idle.
//! Device/mailbox MMIO routing into `DeviceState` is later (M2).

mod error;
mod host_mailbox;
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

use host_mailbox::HostMailbox;
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
    /// Host-side mailbox model (Task-completion). Disabled by default; ticked by
    /// `boot_to_idle`. Enable with `enable_host_mailbox` for the real boot path.
    host_mailbox: HostMailbox,
}

/// The outcome of a [`FirmwareProcessor::boot_to_idle`] run: how far the
/// firmware got and why it stopped. The milestone-M1.7 observation record.
#[derive(Debug, Clone)]
pub struct IdleReport {
    /// True iff the run stopped because the firmware reached its idle wait --
    /// any `Step::Wait` (the command-loop `waiti`; with `waiti` now retiring
    /// and interrupt delivery checked ahead of execution, a returned `Wait`
    /// is itself proof nothing was deliverable, so no PC-stability check is
    /// needed).
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
        Self { cpu, bus, entry, symbols, host_mailbox: HostMailbox::new() }
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

        // Low-VMA sections stored at file = vaddr + 0x100, not the base +0x5c
        // (see LOW_VMA_FILE_OFFSET docs). Registered as vaddr-keyed fetch overlays
        // so only low-window fetches are remapped, not the code region's alias of
        // the same physical bytes. The dispatch-function block (iter16) and the
        // window-exception vector table (iter17).
        bus.add_rom_overlay(LOW_TEXT_BLOCK_LO, LOW_TEXT_BLOCK_HI, LOW_VMA_FILE_OFFSET);
        bus.add_rom_overlay(WINDOW_VECTOR_LO, WINDOW_VECTOR_HI, LOW_VMA_FILE_OFFSET);

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
        Self { cpu, bus, entry: RESET_ENTRY, symbols, host_mailbox: HostMailbox::new() }
    }

    /// Enable the host-mailbox completion model for the boot-to-idle run. Off by
    /// default so existing observation tests are unaffected.
    pub fn enable_host_mailbox(&mut self) {
        self.host_mailbox.enable();
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
    /// (a) a `Step::Wait` (idle -- `reached_idle`),
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
            // Faithful task-completion: on the firmware's mailbox POST, the host
            // model consumes the descriptor and the completion agent writes the
            // task done-flag (no-op until enabled).
            self.host_mailbox.tick(&mut self.bus);

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
                    // Interrupt delivery is checked ahead of execution
                    // (Task 4), so a returned Wait means nothing was
                    // deliverable -- the CPU is genuinely idle in its
                    // command-loop waiti. (With waiti now retiring, keying on
                    // PC-stability would miss the first idle step.)
                    reached_idle = true;
                    wait_reason = Some(reason);
                    break;
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

/// M2c iter16/iter17: the firmware `.text` is not a single uniform file offset.
/// Several sections are linked at a low VMA but stored later in the file, so
/// `file = vaddr + LOW_VMA_FILE_OFFSET` (0x100) for them instead of the base
/// `+0x5c`. Two such sections are modeled as vaddr-keyed fetch overlays (see
/// [`crate::firmware::mmio::Bus::add_rom_overlay`]):
///
/// - **The dispatch-function block** `[LOW_TEXT_BLOCK_LO, HI)` (iter16). The
///   firmware calls into it via compiled-in function pointers (e.g. `0x581c`,
///   `0x5858`, `0x588c`, registered as a dispatch table); under the base offset
///   those pointers fetch mid-instruction garbage (the iter16 `Unknown 0x588c`
///   wall). Proven: at `+0x100` all three targets are clean `entry a1,a1,0x20`
///   prologues with coherent bodies; at `+0x5c` they are mid-instruction.
/// - **The window-exception vector table** `[WINDOW_VECTOR_LO, HI)` (iter17).
///   The six VECBASE-relative window vectors (Overflow/Underflow 4/8/12, 0x40
///   apart from VECBASE=0x800) hold the register-window spill/fill handlers
///   (`s32e`/`l32e`/`rfwo`/`rfwu`). At `+0x5c` this region reads as zeros (the
///   `Unknown 0x880` wall); at `+0x100` it decodes as the real spill handlers.
///
/// The `[LO, HI)` bounds are empirically determined (walk-and-stub) -- the seams
/// are code-to-code with no padding marker, and the `$PS1` container has no
/// segment table to derive exact section extents from.
/// FIXME(iter16): reconstruct the firmware's full piecewise VMA/LMA layout
/// (every seam) rather than these hand-bounded sections.
const LOW_TEXT_BLOCK_LO: u32 = 0x0000_581c;
const LOW_TEXT_BLOCK_HI: u32 = 0x0000_5d30;
const WINDOW_VECTOR_LO: u32 = 0x0000_0800;
const WINDOW_VECTOR_HI: u32 = 0x0000_0980;
const LOW_VMA_FILE_OFFSET: u32 = 0x100;

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
    let mut map = HashMap::new();
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    // Two layers, read in order so the second OVERRIDES the first for a shared
    // address:
    //   1. base   -- the Ghidra export (FUN_ placeholders + recovered library
    //      names). Lives under gitignored `build/`; regenerable, not tracked.
    //   2. overlay -- our git-TRACKED semantic names discovered during RE
    //      (e.g. `task_dispatcher`). Persists across `git clean`, versions with
    //      the code, visible to teammates. THIS is where new RE names go.
    // Same format for both: `0xADDR<TAB>name`; lines without a hex first column
    // (blank lines, `#` comments) are skipped.
    let base = manifest.join("build/experiments/firmware-re/symbols.txt");
    let overlay = manifest.join("src/firmware/firmware-symbols.txt");
    for path in [base, overlay] {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
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
        // of jumping to PC=0 (the pre-fix iter12 wall). `main` then returns to
        // the crt0 post-return site 0x2000e035 -- an undecoded op 0x41f0,
        // iter13's wall.
        //
        // This gate is a coarse progress OBSERVATION: it pins only "the syscall
        // dispatch did not collapse to PC=0" (the specific iter12 regression).
        // It does NOT prove the syscall was serviced correctly -- the
        // pre-Harvard iter10 state also reached 0x2000e035 (~47.5k instrs, same
        // 0x41f0 wall) despite the stack-store-drop bug, because
        // window_exceptions=0 keeps the crt0->main return chain in the register
        // file, immune to lost stack data. The precise regression guard for the
        // l32r-reads-IRAM fix is the unit test
        // `low_window_l32r_reads_image_not_clobbered_local_data`.
        assert_ne!(
            report.unknown_op.map(|(pc, _)| pc),
            Some(0x0000_0000),
            "boot walls at PC=0 -- the exception-vector l32r read the zeroed DRAM \
             overlay instead of the IRAM literal (last_pc={:#x})",
            report.last_pc,
        );

        // iter13 (2026-07-05): the boot's one user-mode SYSCALL now routes to
        // the firmware's unified general-exception handler (0x2958) instead of
        // the mislabeled 0x28b4 dispatcher, so it is SERVICED rather than
        // falling through -- `main` no longer returns to the crt0 trap. The
        // wall advances past 0x2000e035 (to the handler's own next frontier, an
        // undecoded `rur` at ~0x2a09). This pins that the main-return wall
        // stays cleared: a regression that re-broke syscall routing would send
        // the boot back to 0x2000e035.
        assert_ne!(
            report.unknown_op.map(|(pc, _)| pc),
            Some(0x2000_e035),
            "boot walls at 0x2000e035 again -- the user-mode syscall was not \
             serviced (general-exception routing regressed; last_pc={:#x})",
            report.last_pc,
        );

        // iter16 (2026-07-06): a dispatch table the firmware builds holds
        // compiled-in function pointers into the low `.text` (0x581c/0x5858/
        // 0x588c). Those functions are stored in the file at `vaddr + 0x100`, not
        // the base `+0x5c` -- a piecewise VMA/LMA layout. The ROM fetch overlay
        // (`LOW_TEXT_BLOCK_*`) now serves them, so `callx8 0x588c` fetches the real
        // `entry` prologue instead of mid-instruction garbage. This pins that the
        // block wall stays cleared: a regression in the overlay would send the boot
        // back to an `Unknown` at 0x588c.
        assert_ne!(
            report.unknown_op.map(|(pc, _)| pc),
            Some(0x0000_588c),
            "boot walls at 0x588c again -- the low-.text +0x100 fetch overlay \
             regressed (last_pc={:#x})",
            report.last_pc,
        );

        // iter17 (2026-07-06): the windowed-register ABI is now fully modeled.
        // The firmware nests calls until the register window fills (8 packed
        // call8 frames, WINDOWSTART=0xaaaa); the next high-register write then
        // must spill the oldest frame (WindowOverflow) BEFORE it clobbers that
        // frame's saved a0. Without the general `window_check` (run before any
        // instruction whose `max_ar` reaches a8..a15), the return address was
        // written into the to-be-spilled slot and lost, so a later `retw.n`
        // read a0=0 and walled at PC=0 (~instr 48215). With the spill firing at
        // the right time, the window overflow/underflow handlers (s32e/l32e +
        // rfwo/rfwu) round-trip correctly and the boot runs the full budget.
        //
        // Two independent regression guards: (a) window exceptions actually
        // fire -- a broken `window_check`/`max_ar` would drop this to 0; and
        // (b) the boot advances past the old PC=0 window-ABI wall.
        assert!(
            report.window_exceptions > 0,
            "the windowed-register ABI is no longer exercised ({} window exceptions) \
             -- window_check/max_ar regressed",
            report.window_exceptions,
        );
        assert!(
            report.instrs_executed > 48_215 || report.reached_idle,
            "boot regressed to the window-ABI wall ({} instrs, last_pc={:#x}) -- the \
             full-window return-address spill (window_check) regressed",
            report.instrs_executed,
            report.last_pc,
        );
    }

    /// M2c iter18: with the faithful completion model enabled, boot advances past
    /// the `task_dispatcher` (0xd7f0) recursion along the REAL path (the task is
    /// picked from real scheduler state, not force-done's artificial switch). The
    /// completion delivers the done-flag `[0x9070]`; boot then runs to its next
    /// genuine stop, which this test records for the follow-through task.
    #[test]
    fn m2c_boot_completion_advances_past_recursion() {
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        proc.enable_host_mailbox();
        let report = proc.boot_to_idle(2_000_000);

        eprintln!("=== M2c completion-model boot ===");
        eprintln!("reached_idle    = {}", report.reached_idle);
        eprintln!("instrs_executed = {}", report.instrs_executed);
        eprintln!(
            "last_pc         = {:#x}  {}",
            report.last_pc,
            nearest_symbol(&proc.symbols, report.last_pc)
        );
        eprintln!("wait_reason     = {:?}", report.wait_reason);
        eprintln!("unknown_op      = {:?}", report.unknown_op.map(|(p, w)| format!("{p:#x}: {w:#010x}")));
        eprintln!("unresolved_spin = {:?}", report.unresolved_spin);
        eprintln!("done-flag[0x9070] = {:#x}", proc.bus.load_local32(0x9070));

        // The completion fired: the local done-flag is set.
        assert_ne!(proc.bus.load_local32(0x9070), 0, "completion delivered the done-flag");
        // Boot progressed OUT of the dispatcher recursion (0xd7f0..0xd848): it
        // either reached idle, hit a new decode/opcode wall, or a spin elsewhere,
        // but it is no longer looping in the scheduler.
        let in_recursion = (0xd7f0..=0xd848).contains(&report.last_pc);
        assert!(
            !in_recursion || report.reached_idle,
            "boot left the task_dispatcher recursion (last_pc={:#x})",
            report.last_pc
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
        let mut ring: std::collections::VecDeque<(u64, u32, String, [u32; 16], u32, u32)> =
            std::collections::VecDeque::with_capacity(KEEP + 1);
        let mut n = 0u64;
        let mut stop = String::from("budget reached");
        // XDNA_FW_CALLS: record ONLY call-family / entry / retw events in the
        // ring, so the tail shows the call/return structure (e.g. an unbounded
        // recursion cycle) instead of the inner-loop or overflow-handler grind.
        let calls_only = std::env::var("XDNA_FW_CALLS").is_ok();
        // XDNA_FW_STOP_PC=<hex>: stop the probe the first time execution reaches
        // this PC, so the ring shows the call-path INTO it (e.g. the top-level
        // caller that first enters a loop we later see recurse).
        let stop_pc = std::env::var("XDNA_FW_STOP_PC")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok());
        while n < MAX {
            let pc = proc.cpu.pc;
            let disasm = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                Ok(phys) => {
                    // Disassemble via `fetch8` (vaddr-aware), NOT `peek8` (phys):
                    // low-`.text` overlay PCs (0x581c-0x5d30, 0x800-0x980) execute
                    // from file+0x100, so a phys peek shows garbage/misclassified
                    // ops there. Fetch8 matches what `step` actually runs. Peek up
                    // to 8 bytes so FLIX bundles (op0 0xe/0xf) disassemble fully.
                    let b: [u8; 8] = std::array::from_fn(|i| {
                        proc.bus.fetch8(pc.wrapping_add(i as u32), phys.wrapping_add(i as u32))
                    });
                    format!("{:?}", decode::decode(&b, pc).op)
                }
                Err(_) => "<fetch-fault>".to_string(),
            };
            let mut regs = [0u32; 16];
            for (r, slot) in regs.iter_mut().enumerate() {
                *slot = proc.cpu.regs.read_ar(r as u8);
            }
            let record = !calls_only
                || disasm.starts_with("Call")
                || disasm.starts_with("Entry")
                || disasm.starts_with("Retw")
                || disasm.starts_with("RetN");
            if record {
                if ring.len() == KEEP {
                    ring.pop_front();
                }
                ring.push_back((n, pc, disasm, regs, proc.cpu.regs.windowbase, proc.cpu.regs.windowstart));
            }

            if Some(pc) == stop_pc {
                stop = format!("XDNA_FW_STOP_PC {pc:#x} reached at n={n}");
                break;
            }

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
        for (i, pc, disasm, regs, wb, ws) in &ring {
            let lo: Vec<String> = (0..8).map(|r| format!("a{r}={:#x}", regs[r])).collect();
            let sym = nearest_symbol(&proc.symbols, *pc);
            eprintln!("{i:>6} pc={pc:#x} {sym:<24} {disasm:<30} wb={wb} ws={ws:#06x} | {}", lo.join(" "));
        }
        // The full a0..a15 window of the last few instructions (call/window state).
        eprintln!("--- a8..a15 of the final {} instrs ---", 6.min(ring.len()));
        for (i, pc, _, regs, _, _) in ring.iter().rev().take(6).rev() {
            let hi: Vec<String> = (8..16).map(|r| format!("a{r}={:#x}", regs[r])).collect();
            eprintln!("{i:>6} pc={pc:#x} | {}", hi.join(" "));
        }
    }

    /// Nearest symbol at or below `pc`, formatted `name+0xNN` (or bare `name`
    /// at the exact entry), for readable probe output. Empty when no symbol
    /// lies within `MAX_SPAN` below `pc` -- so a gap between symbols reads as
    /// blank rather than getting mislabeled as a distant earlier function.
    /// Names live in `build/experiments/firmware-re/symbols.txt`; add semantic
    /// names there as RE proceeds (e.g. `task_dispatcher`).
    fn nearest_symbol(symbols: &std::collections::HashMap<u32, String>, pc: u32) -> String {
        const MAX_SPAN: u32 = 0x800;
        let mut best: Option<(u32, &str)> = None;
        for (&addr, name) in symbols {
            if addr <= pc && pc - addr < MAX_SPAN && best.map_or(true, |(b, _)| addr > b) {
                best = Some((addr, name.as_str()));
            }
        }
        match best {
            Some((addr, name)) if addr == pc => name.to_string(),
            Some((addr, name)) => format!("{name}+{:#x}", pc - addr),
            None => String::new(),
        }
    }

    /// M2c Phase 0 (iter18) DIAGNOSTIC: image-wide STATIC search for store
    /// instructions with a given byte displacement (default 0x30, the task
    /// done-flag offset the dispatcher reads via `l32i.n [task+0x30]`).
    /// Resolves the shape-(i)-vs-(ii) completion-writer fork: if NO store
    /// anywhere in the firmware's code targets `+0x30`, then only an external
    /// agent (DMA/peripheral) can set a task's done-flag (shape ii); if some
    /// store does, it names the candidate writer for inspection.
    ///
    /// Disassembles every function listed in `symbols.txt` linearly (each entry
    /// up to the next symbol). Reads code via `fetch8(vaddr, vaddr)` -- the
    /// reset's way-6 identity region makes VMA==phys across all code, and
    /// `fetch8` applies the low-window overlays -- so it covers NEVER-EXECUTED
    /// functions too, which is the whole point of a static search (vs. iter18's
    /// runtime store-watch that only saw executed code). Set XDNA_FW_STORE_DISP
    /// to search another offset. Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_store_search() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the static store-search");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let disp = std::env::var("XDNA_FW_STORE_DISP")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .unwrap_or(0x30);
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        // Symbol entries sorted by address; disassemble each [entry, next).
        let mut syms: Vec<(u32, String)> = proc.symbols.iter().map(|(a, n)| (*a, n.clone())).collect();
        syms.sort_by_key(|(a, _)| *a);

        eprintln!("=== M2c static store-search: stores with displacement {disp:#x} ===");
        eprintln!("(each hit: pc  symbol  op -- store address is AR[s]+{disp:#x})");
        let mut hits = 0u32;
        let mut store_disps: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
        for i in 0..syms.len() {
            let start = syms[i].0;
            let end = syms.get(i + 1).map(|(a, _)| *a).unwrap_or(start + 0x400);
            // Cap per-function span so a bogus symbol gap can't run away.
            let end = end.min(start + 0x2000);
            let mut pc = start;
            while pc < end {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, pc + k as u32));
                let d = decode::decode(&b, pc);
                let store_imm = match d.op {
                    decode::Op::S32i { imm, .. }
                    | decode::Op::S32iN { imm, .. }
                    | decode::Op::S8i { imm, .. }
                    | decode::Op::S16i { imm, .. } => Some(imm),
                    _ => None,
                };
                if let Some(imm) = store_imm {
                    *store_disps.entry(imm).or_insert(0) += 1;
                    if imm == disp {
                        let sym = nearest_symbol(&proc.symbols, pc);
                        eprintln!("  pc={pc:#08x}  {sym:<28}  {:?}", d.op);
                        hits += 1;
                    }
                }
                pc += (d.len as u32).max(1);
            }
        }
        eprintln!("--- {hits} store(s) with displacement {disp:#x} across {} functions ---", syms.len());
        eprintln!("--- store-displacement histogram (top offsets) ---");
        let mut by_count: Vec<(u32, u32)> = store_disps.into_iter().collect();
        by_count.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        for (off, c) in by_count.iter().take(16) {
            eprintln!("  disp {off:#06x}: {c}");
        }
    }

    /// M2c iter18 RE TOOL: static disassembly of an arbitrary VMA range, read
    /// via `fetch8` over the reset way-6 identity region (covers never-executed
    /// code and branches not taken -- unlike the trace probes, which only show
    /// the executed path). Set XDNA_FW_DISASM=<start>:<end> (hex VMAs) to pick
    /// the range; each line is `pc symbol op` walked by decoded length. Reading
    /// the actual control flow of a function beats theorizing about it. Ignored
    /// unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_disasm_range() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the range disassembler");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let range = std::env::var("XDNA_FW_DISASM").unwrap_or_else(|_| "0xd7f0:0xd870".to_string());
        let (s, e) = range.split_once(':').expect("XDNA_FW_DISASM must be start:end (hex)");
        let start = u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).expect("start hex");
        let end = u32::from_str_radix(e.trim().trim_start_matches("0x"), 16).expect("end hex");
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        eprintln!("=== M2c static disasm {start:#x}..{end:#x} ===");
        let mut pc = start;
        while pc < end {
            let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, pc + k as u32));
            let d = decode::decode(&b, pc);
            let sym = nearest_symbol(&proc.symbols, pc);
            let raw_hex: String =
                b[..(d.len as usize).max(1).min(8)].iter().map(|x| format!("{x:02x}")).collect();
            eprintln!("  {pc:#08x} {sym:<26} {:<40} [{raw_hex}]", format!("{:?}", d.op));
            pc += (d.len as u32).max(1);
        }
    }

    /// M2c Phase 0 (iter18) DIAGNOSTIC: static DIRECT-call cross-reference.
    /// Scans every symbol function for call-family instructions with an
    /// immediate target (call0/call4/call8/call12 -- NOT register-indirect
    /// callx*, whose target isn't statically known), and lists the call sites
    /// targeting each address in a query set. Traces who reaches the
    /// scheduler-region done-flag-setter candidates the store-search surfaced.
    /// Query set = XDNA_FW_XREF (comma-separated hex) or a built-in default of
    /// those candidates. Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_call_xref() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the call-xref probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let targets: Vec<u32> = std::env::var("XDNA_FW_XREF")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok())
                    .collect()
            })
            .unwrap_or_else(|| {
                // The store-search's scheduler-region candidates + the dispatcher.
                vec![0xd7f0, 0xc9dc, 0xd134, 0xd1e8, 0xd53c, 0xd84c, 0xe098]
            });
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        let mut syms: Vec<(u32, String)> = proc.symbols.iter().map(|(a, n)| (*a, n.clone())).collect();
        syms.sort_by_key(|(a, _)| *a);

        // Collect all DIRECT (immediate-target) call edges: (call_site, target).
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for i in 0..syms.len() {
            let start = syms[i].0;
            let end = syms.get(i + 1).map(|(a, _)| *a).unwrap_or(start + 0x400).min(start + 0x2000);
            let mut pc = start;
            while pc < end {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, pc + k as u32));
                let d = decode::decode(&b, pc);
                let target = match d.op {
                    decode::Op::Call0 { target }
                    | decode::Op::Call4 { target }
                    | decode::Op::Call8 { target }
                    | decode::Op::Call12 { target } => Some(target),
                    _ => None,
                };
                if let Some(t) = target {
                    edges.push((pc, t));
                }
                pc += (d.len as u32).max(1);
            }
        }

        eprintln!("=== M2c call-xref: DIRECT callers of {} target(s) ===", targets.len());
        eprintln!("(register-indirect callx* callers are NOT shown -- target unknown statically)");
        for tgt in targets {
            let tname = nearest_symbol(&proc.symbols, tgt);
            eprintln!("target {tgt:#08x}  {tname}:");
            let mut any = false;
            for (site, _) in edges.iter().filter(|(_, t)| *t == tgt) {
                eprintln!("    called from {site:#08x}  {}", nearest_symbol(&proc.symbols, *site));
                any = true;
            }
            if !any {
                eprintln!("    (no DIRECT callers -- reached only via callx*/table or fall-through)");
            }
        }
    }

    /// M2c Phase 0 (iter18) EXPERIMENT: force the task done-flag and observe.
    /// The `task_dispatcher` recursion spins because `[current_task + 0x30]`
    /// (the done-flag) is never set. This probe force-writes it to 1 right
    /// before the dispatcher's check at `0xd828` (`l32i.n a10, [a4+0x30]`), so
    /// `beqz` falls through to the "task done / unwind" path. Tests the causal
    /// hypothesis directly (does setting the done-flag unwind the recursion to
    /// the `waiti 0` idle loop?), sidestepping who/what writes it on real
    /// silicon. NOT a fix -- a diagnostic. Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_force_done() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the force-done experiment");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        const MAX: u64 = 1_000_000;
        const KEEP: usize = 48;
        // The dispatcher's done-flag read site (`l32i.n a10, [a4+0x30]`): force
        // the flag to 1 just before it, computing the address from the live
        // task pointer in a4 (handles whichever task is current).
        const DONE_CHECK_PC: u32 = 0xd828;
        let mut n = 0u64;
        let mut forces = 0u64;
        let mut forced_addrs: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        let mut stop = String::from("budget reached");
        // Ring of the last KEEP instrs (n, pc, disasm) to see the stop context.
        let mut ring: std::collections::VecDeque<(u64, u32, String)> =
            std::collections::VecDeque::with_capacity(KEEP + 1);
        // Distinct-PC histogram over the FINAL window, to spot a tight new spin.
        while n < MAX {
            let pc = proc.cpu.pc;
            if pc == DONE_CHECK_PC {
                let done_addr = proc.cpu.regs.read_ar(4).wrapping_add(0x30);
                proc.bus.store_local32(done_addr, 1);
                forces += 1;
                forced_addrs.insert(done_addr);
            }
            let disasm = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                Ok(phys) => {
                    let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                    format!("{:?}", decode::decode(&b, pc).op)
                }
                Err(_) => "<fetch-fault>".to_string(),
            };
            if ring.len() == KEEP {
                ring.pop_front();
            }
            ring.push_back((n, pc, disasm));
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?}) at pc={pc:#x} (idle)");
                    break;
                }
                Step::Unknown { pc: upc, word } => {
                    stop = format!("Unknown at pc={upc:#x} word={word:#010x}");
                    break;
                }
            }
            if let Some(addr) = proc.bus.sysstub().spinning() {
                stop = format!("sysstub spin at {addr:#x}");
                break;
            }
        }
        eprintln!("=== M2c force-done experiment ===");
        eprintln!("forced done-flag {forces} time(s) at addrs {forced_addrs:x?}");
        eprintln!("instrs executed = {n}");
        eprintln!("stop reason     = {stop}");
        eprintln!("last pc         = {:#x}  {}", proc.cpu.pc, nearest_symbol(&proc.symbols, proc.cpu.pc));
        // Distinct PCs in the final window: a small set == a tight spin.
        let distinct: std::collections::BTreeSet<u32> = ring.iter().map(|(_, pc, _)| *pc).collect();
        eprintln!("distinct PCs in last {} instrs = {}", ring.len(), distinct.len());
        eprintln!("--- last {} instrs before stop ---", ring.len());
        for (i, pc, disasm) in &ring {
            eprintln!("{i:>7} pc={pc:#08x} {:<24} {disasm}", nearest_symbol(&proc.symbols, *pc));
        }
    }

    /// M2c iter18 DIAGNOSTIC: verify the xdna-driver-derived completion trigger.
    /// The firmware posts a fw->host mailbox message (i2x tail 0x27200170=0xf18)
    /// and its boot task waits on a local done-flag the host's ACK would set.
    /// This performs each candidate host-ack once the post is detected and
    /// watches whether `[task+0x30]` (0x9070) is set NATURALLY and boot advances.
    /// XDNA_FW_ACK selects the candidate:
    ///   head     -> write i2x HEAD 0x27200174 = posted val, intr 0x27200178 = 0
    ///   tail0    -> write i2x tail 0x27200170 = 0        (ring drained)
    ///   tailadv  -> write i2x tail 0x27200170 = posted+8 (host advanced past)
    ///   doorbell -> pend level-1 interrupt bit 0 (the fw's armed doorbell)
    ///   headdb   -> head-ack AND doorbell
    /// XDNA_FW_ACK_RESEED=1 re-applies the ack every step. Ignored unless
    /// XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_force_ack() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the force-ack experiment");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        const TAIL: u32 = 0x2720_0170;
        const HEAD: u32 = 0x2720_0174;
        const INTR: u32 = 0x2720_0178;
        const POSTED: u32 = 0xf18;
        let ack = std::env::var("XDNA_FW_ACK").unwrap_or_else(|_| "head".to_string());
        let reseed = std::env::var("XDNA_FW_ACK_RESEED").is_ok();
        let apply_ack = |proc: &mut FirmwareProcessor, ack: &str| match ack {
            "head" => {
                proc.bus.store32(HEAD, POSTED);
                proc.bus.store32(INTR, 0);
            }
            "tail0" => proc.bus.store32(TAIL, 0),
            "tailadv" => proc.bus.store32(TAIL, POSTED + 8),
            "doorbell" => proc.cpu.interrupt |= 1,
            "headdb" => {
                proc.bus.store32(HEAD, POSTED);
                proc.bus.store32(INTR, 0);
                proc.cpu.interrupt |= 1;
            }
            other => panic!("unknown XDNA_FW_ACK={other}"),
        };

        const DONE_FLAGS: [u32; 2] = [0x9070, 0x10f40];
        const MAX: u64 = 1_000_000;
        const KEEP: usize = 32;
        let mut n = 0u64;
        let mut posted_at: Option<u64> = None;
        let mut acked = false;
        let mut done_set: Vec<(u32, u64)> = Vec::new();
        let mut stop = String::from("budget reached");
        let mut ring: std::collections::VecDeque<(u64, u32, String)> =
            std::collections::VecDeque::with_capacity(KEEP + 1);
        while n < MAX {
            let pc = proc.cpu.pc;
            // Detect the post: i2x tail reads back the posted value.
            if posted_at.is_none() && proc.bus.load32(TAIL) == POSTED {
                posted_at = Some(n);
            }
            // Apply the ack once posted (once, or every step if reseed).
            if posted_at.is_some() && (!acked || reseed) {
                apply_ack(&mut proc, &ack);
                acked = true;
            }
            let disasm = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                Ok(phys) => {
                    let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                    format!("{:?}", decode::decode(&b, pc).op)
                }
                Err(_) => "<fetch-fault>".to_string(),
            };
            if ring.len() == KEEP {
                ring.pop_front();
            }
            ring.push_back((n, pc, disasm));
            // Only watch after the post (pre-boot image bytes at these addresses
            // are nonzero until boot's memset zeroes local memory -- would false-
            // trigger at n=0).
            if posted_at.is_some() {
                for f in DONE_FLAGS {
                    if proc.bus.load_local32(f) != 0 && !done_set.iter().any(|(a, _)| *a == f) {
                        done_set.push((f, n));
                    }
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?}) at pc={pc:#x} (idle!)");
                    break;
                }
                Step::Unknown { pc: upc, word } => {
                    stop = format!("Unknown at pc={upc:#x} word={word:#010x}");
                    break;
                }
            }
            if let Some(addr) = proc.bus.sysstub().spinning() {
                stop = format!("sysstub spin at {addr:#x}");
                break;
            }
        }
        eprintln!("=== M2c force-ack experiment (ack={ack}, reseed={reseed}) ===");
        eprintln!("posted (tail==0xf18) at instr = {posted_at:?}");
        eprintln!("instrs executed = {n}");
        eprintln!("stop reason     = {stop}");
        eprintln!("last pc         = {:#x}  {}", proc.cpu.pc, nearest_symbol(&proc.symbols, proc.cpu.pc));
        eprintln!("done-flags set naturally: {done_set:x?}");
        for f in DONE_FLAGS {
            eprintln!("  [{f:#x}] = {:#x}", proc.bus.load_local32(f));
        }
        eprintln!("--- last {} instrs before stop ---", ring.len());
        for (i, pc, disasm) in &ring {
            eprintln!("{i:>7} pc={pc:#08x} {:<24} {disasm}", nearest_symbol(&proc.symbols, *pc));
        }
    }

    /// M2c iter18 RE TOOL (planning discovery): locate the firmware-LOCAL i2x
    /// ring buffer where the fw writes the mailbox message header. The driver
    /// (xdna-driver) stores HOST-view addresses; the emulator runs the fw's
    /// LOCAL view, so the ring base must be found empirically. Store-watches
    /// every 32-bit store for the header magic (top byte 0x1D == the `id`
    /// field), and every write to the i2x tail reg 0x27200170 (the post).
    /// Reports each magic store's EA (= header.id, base = EA-8), the 16-byte
    /// header decoded, and the memory region it lands in (backed vs the
    /// unmodeled 0x08a00000 gap). Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_i2x_ring_locate() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the i2x-ring locate probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        const MAX: u64 = 60_000;
        const TAIL: u32 = 0x2720_0170;
        const MAGIC: u32 = 0x1D00_0000;
        const MAGIC_MASK: u32 = 0xFF00_0000;
        let region_name = |a: u32| -> &'static str {
            match a {
                _ if a < 0x0400_0000 => "LOCAL(low)",
                _ if a < 0x0800_0000 => "ARRAY",
                _ if (0x0800_0000..0x08b0_0000).contains(&a) => "GAP(unbacked->System)",
                _ if (0x08b0_0000..0x2700_0000).contains(&a) => "RAM",
                _ if (0x2700_0000..0x2800_0000).contains(&a) => "MAILBOX",
                _ => "SYSTEM",
            }
        };

        let _ = (MAGIC, MAGIC_MASK);
        // First store to each buffer/mailbox address (EA >= 0x08000000): reveals
        // where the fw builds the message. Keyed by EA; keeps (n, pc, val, width).
        let mut buf_stores: std::collections::BTreeMap<u32, (u64, u32, u32, u8)> =
            std::collections::BTreeMap::new();
        let mut tail_writes: Vec<(u64, u32, u32)> = Vec::new(); // (n, pc, val)
        let mut n = 0u64;
        let mut stop = String::from("budget reached");
        while n < MAX {
            let pc = proc.cpu.pc;
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                // (t, s, imm, width_bytes) for every store width.
                let store = match decode::decode(&b, pc).op {
                    Op::S32i { t, s, imm } | Op::S32iN { t, s, imm } | Op::S32ri { t, s, imm } => {
                        Some((t, s, imm, 4u8))
                    }
                    Op::S16i { t, s, imm } => Some((t, s, imm, 2)),
                    Op::S8i { t, s, imm } => Some((t, s, imm, 1)),
                    _ => None,
                };
                if let Some((t, s, imm, w)) = store {
                    let ea = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                    let val = proc.cpu.regs.read_ar(t);
                    if ea == TAIL {
                        tail_writes.push((n, pc, val));
                    }
                    // Buffer/mailbox writes only (skip local/ROM/array scratch).
                    if ea >= 0x0800_0000 {
                        buf_stores.entry(ea).or_insert((n, pc, val, w));
                    }
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?})");
                    break;
                }
                Step::Unknown { pc: upc, word } => {
                    stop = format!("Unknown at {upc:#x} word={word:#010x}");
                    break;
                }
            }
        }
        eprintln!("=== M2c i2x-ring locate ===");
        eprintln!("instrs executed = {n}; stop = {stop}");
        eprintln!("--- i2x tail (0x27200170) writes (the post) ---");
        for (n, pc, val) in &tail_writes {
            eprintln!("  n={n:>6} pc={pc:#08x} tail <- {val:#x}");
        }
        eprintln!("--- first store to each buffer/mailbox address (EA >= 0x08000000) ---");
        eprintln!("    (contiguous runs in one region == a message/struct the fw wrote)");
        let mut last_ea = 0u32;
        for (ea, (n, pc, val, w)) in &buf_stores {
            let gap = if *ea != last_ea.wrapping_add(4) && last_ea != 0 {
                "  <-- new block"
            } else {
                ""
            };
            eprintln!(
                "  {ea:#010x} ({:<22}) <- {val:#010x} (w{w}) n={n:>6} pc={pc:#08x}{gap}",
                region_name(*ea)
            );
            last_ea = *ea;
        }
        eprintln!("(total {} distinct buffer/mailbox addresses written)", buf_stores.len());
    }

    /// M2c iter18 RE TOOL: enumerate every load site the stuck boot spins on.
    /// Widen-before-deepen: rather than guess the next gate, run into steady
    /// state then, over a window of the recursion cycle, compute the effective
    /// address of every load (EA = AR[s] + imm; imm is byte-scaled) and count
    /// repetitions. Addresses read many times -- especially in the mailbox
    /// (0x2700_0000..0x2800_0000) or system apertures -- are the poll/wait sites.
    /// Reports top load addresses by count with region tag and a best-effort
    /// peeked value. Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_poll_map() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the poll-map");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        // WARMUP skips the transient to reach steady state; WINDOW records over
        // many cycles. Override via XDNA_FW_POLL_WARMUP / XDNA_FW_POLL_WINDOW to
        // capture the EARLY phase (e.g. WARMUP=0) instead of the steady spin.
        let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        let warmup = env_u64("XDNA_FW_POLL_WARMUP", 300_000);
        let window = env_u64("XDNA_FW_POLL_WINDOW", 200_000);
        // addr -> (count, causing-pc)
        let mut hits: std::collections::HashMap<u32, (u64, u32)> = std::collections::HashMap::new();
        let mut n = 0u64;
        let region = |a: u32| -> &'static str {
            if (0x2700_0000..0x2800_0000).contains(&a) {
                "MBOX"
            } else if (0x0800_0000..0x2700_0000).contains(&a) {
                "RAM"
            } else if a < 0x0080_0000 {
                "IMG/LO"
            } else if (0x3c00_0000..0x3d00_0000).contains(&a) {
                "PGTBL"
            } else {
                "SYS"
            }
        };
        while n < warmup + window {
            let pc = proc.cpu.pc;
            if n >= warmup {
                if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                    let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                    let op = decode::decode(&b, pc).op;
                    let ea = match op {
                        decode::Op::L32i { s, imm, .. }
                        | decode::Op::L32iN { s, imm, .. }
                        | decode::Op::L8ui { s, imm, .. }
                        | decode::Op::L16ui { s, imm, .. }
                        | decode::Op::L16si { s, imm, .. } => {
                            Some(proc.cpu.regs.read_ar(s).wrapping_add(imm))
                        }
                        _ => None,
                    };
                    if let Some(ea) = ea {
                        let e = hits.entry(ea).or_insert((0, pc));
                        e.0 += 1;
                    }
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(_) => break,
                Step::Unknown { .. } => break,
            }
            if proc.bus.sysstub().spinning().is_some() {
                break;
            }
        }
        eprintln!("=== M2c poll-map (loads over instrs [{warmup}, {}]) ===", warmup + window);
        let mut v: Vec<(u32, u64, u32)> = hits.iter().map(|(a, (c, pc))| (*a, *c, *pc)).collect();
        v.sort_by_key(|(_, c, _)| std::cmp::Reverse(*c));
        eprintln!("--- top 30 load addresses by repetition ---");
        eprintln!("   addr        region  count   from-pc(symbol)          peeked");
        for (a, c, pc) in v.iter().take(30) {
            let val = u32::from_le_bytes(std::array::from_fn(|k| proc.bus.peek8(a.wrapping_add(k as u32))));
            eprintln!(
                "  {a:#010x}  {:<6}  {c:>6}  {pc:#08x} {:<20}  {val:#010x}",
                region(*a),
                nearest_symbol(&proc.symbols, *pc)
            );
        }
        // Focus list: only mailbox/system aperture hits (host/hardware writable).
        eprintln!("--- mailbox/system aperture loads (candidate host handshakes) ---");
        for (a, c, pc) in v.iter().filter(|(a, _, _)| region(*a) == "MBOX" || region(*a) == "SYS") {
            let val = u32::from_le_bytes(std::array::from_fn(|k| proc.bus.peek8(a.wrapping_add(k as u32))));
            eprintln!("  {a:#010x}  {:<6}  {c:>6}  {pc:#08x}  {val:#010x}", region(*a));
        }
    }

    /// M2c iter18 DIAGNOSTIC: is the polled event-status bit the TRUE source
    /// whose downstream effect is the task done-flag? force-done proved setting
    /// `[task+0x30]` unwinds the recursion, but that's artificial. The dispatcher
    /// root-cause found the firmware polls event-status pages `0x2727n000`
    /// (FUN_8c68: `l32i a9,[0x27274000]; bbci a9,bit0/bit1`) for a completion bit
    /// nothing sets. This seeds those pages with bit0|bit1 set (the faithful
    /// signal the host/hardware would raise) and observes whether the firmware's
    /// own poll then takes the active path and sets `[task+0x30]` downstream --
    /// i.e. whether the status bit drives the done-flag, or is orthogonal.
    /// XDNA_FW_EVENT_RESEED=1 re-seeds every step (persistently-asserted event)
    /// instead of once. Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_force_event() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the force-event experiment");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        // Event-status pages the poll iterates over (a5=0x27274000, +=0x1000);
        // seed a generous span with bit0|bit1 set.
        const EVENT_PAGES: [u32; 8] = [
            0x2727_1000,
            0x2727_2000,
            0x2727_3000,
            0x2727_4000,
            0x2727_5000,
            0x2727_6000,
            0x2727_7000,
            0x2727_8000,
        ];
        let reseed = std::env::var("XDNA_FW_EVENT_RESEED").is_ok();
        let seed = |proc: &mut FirmwareProcessor| {
            for p in EVENT_PAGES {
                proc.bus.store32(p, 0b11);
            }
        };
        seed(&mut proc);

        // Done-flag addresses of the two known tasks (task+0x30): watch for a
        // natural (firmware-driven, not forced) set.
        const DONE_FLAGS: [u32; 2] = [0x9070, 0x10f40];
        let mut done_set: Vec<(u32, u64)> = Vec::new();

        const MAX: u64 = 1_000_000;
        const KEEP: usize = 40;
        let mut n = 0u64;
        let mut stop = String::from("budget reached");
        // Sentinel: 0x8c9b is the ack store reached ONLY when FUN_8c68 sees bit0
        // SET (fall-through from `bbci a9,bit0,0x8ca5`). If this stays 0 the
        // firmware never observed our seeded bit at the poll -> seed invalid.
        const ACTIVE_PATH_PC: u32 = 0x8c9b;
        let mut active_hits = 0u64;
        let mut ring: std::collections::VecDeque<(u64, u32, String)> =
            std::collections::VecDeque::with_capacity(KEEP + 1);
        while n < MAX {
            let pc = proc.cpu.pc;
            if pc == ACTIVE_PATH_PC {
                active_hits += 1;
            }
            if reseed {
                seed(&mut proc);
            }
            let disasm = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                Ok(phys) => {
                    let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                    format!("{:?}", decode::decode(&b, pc).op)
                }
                Err(_) => "<fetch-fault>".to_string(),
            };
            if ring.len() == KEEP {
                ring.pop_front();
            }
            ring.push_back((n, pc, disasm));
            for f in DONE_FLAGS {
                if proc.bus.load_local32(f) != 0 && !done_set.iter().any(|(a, _)| *a == f) {
                    done_set.push((f, n));
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?}) at pc={pc:#x} (idle)");
                    break;
                }
                Step::Unknown { pc: upc, word } => {
                    stop = format!("Unknown at pc={upc:#x} word={word:#010x}");
                    break;
                }
            }
            if let Some(addr) = proc.bus.sysstub().spinning() {
                stop = format!("sysstub spin at {addr:#x}");
                break;
            }
        }
        eprintln!("=== M2c force-event experiment (reseed={reseed}) ===");
        eprintln!("seeded pages {EVENT_PAGES:x?} = 0b11");
        eprintln!("instrs executed = {n}");
        eprintln!("stop reason     = {stop}");
        eprintln!("last pc         = {:#x}  {}", proc.cpu.pc, nearest_symbol(&proc.symbols, proc.cpu.pc));
        eprintln!("FUN_8c68 active-path (bit0-seen) hits = {active_hits}");
        eprintln!("done-flags set naturally: {done_set:x?}");
        for f in DONE_FLAGS {
            eprintln!("  [{f:#x}] = {:#x}", proc.bus.load_local32(f));
        }
        eprintln!("--- last {} instrs before stop ---", ring.len());
        for (i, pc, disasm) in &ring {
            eprintln!("{i:>7} pc={pc:#08x} {:<24} {disasm}", nearest_symbol(&proc.symbols, *pc));
        }
    }

    /// M2c iter18 Phase 0 DIAGNOSTIC: what interrupt does the firmware actually
    /// ARM? The (C) done-flag mechanism delivers a real async event
    /// (mailbox doorbell -> level-1 interrupt); to inject the RIGHT interrupt we
    /// need the INTENABLE bit(s) the firmware sets during boot. `wsr.intenable`
    /// (SR 0xE4) is internal to the CPU -- invisible to the peripheral probe --
    /// but `Cpu::intenable` is a public field, so we watch it (and `interrupt`,
    /// the pending bits) for changes after each step and log the PC + symbol
    /// that caused each. Answers the "INTENABLE bits NOT yet observed" open item
    /// in the Phase-0 findings. Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_intenable_watch() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the intenable watch");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        const MAX: u64 = 1_000_000;
        let mut n = 0u64;
        let mut stop = String::from("budget reached");
        let mut prev_ie = proc.cpu.intenable;
        let mut prev_ip = proc.cpu.interrupt;
        let mut prev_il = proc.cpu.regs.intlevel();
        // Each transition: (n, causing-pc, symbol, which, old, new).
        let mut changes: Vec<(u64, u32, String, &'static str, u32, u32)> = Vec::new();
        // First instr at which INTLEVEL returns to 0 AFTER intenable is armed --
        // the only window a level-1 doorbell could actually be delivered. If this
        // stays None past the wall, a level-1 IRQ can never set the done-flag
        // (the dispatcher's rsil-2 critical section masks it), which argues the
        // completion is a DMA/DRAM write (shape ii), not a CPU handler (shape i).
        let mut armed_at: Option<u64> = None;
        let mut first_level0_after_arm: Option<(u64, u32)> = None;
        while n < MAX {
            let pc = proc.cpu.pc;
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?}) at pc={pc:#x} (idle)");
                    break;
                }
                Step::Unknown { pc: upc, word } => {
                    stop = format!("Unknown at pc={upc:#x} word={word:#010x}");
                    break;
                }
            }
            if proc.cpu.intenable != prev_ie {
                if prev_ie == 0 && proc.cpu.intenable != 0 {
                    armed_at = Some(n);
                }
                changes.push((
                    n,
                    pc,
                    nearest_symbol(&proc.symbols, pc),
                    "INTENABLE",
                    prev_ie,
                    proc.cpu.intenable,
                ));
                prev_ie = proc.cpu.intenable;
            }
            if proc.cpu.interrupt != prev_ip {
                changes.push((
                    n,
                    pc,
                    nearest_symbol(&proc.symbols, pc),
                    "INTERRUPT",
                    prev_ip,
                    proc.cpu.interrupt,
                ));
                prev_ip = proc.cpu.interrupt;
            }
            let il = proc.cpu.regs.intlevel();
            if il != prev_il {
                changes.push((n, pc, nearest_symbol(&proc.symbols, pc), "INTLEVEL", prev_il, il));
                prev_il = il;
            }
            if armed_at.is_some() && first_level0_after_arm.is_none() && il == 0 {
                first_level0_after_arm = Some((n, pc));
            }
            if let Some(addr) = proc.bus.sysstub().spinning() {
                stop = format!("sysstub spin at {addr:#x}");
                break;
            }
        }
        eprintln!("=== M2c intenable/interrupt/intlevel watch ===");
        eprintln!("instrs executed = {n}");
        eprintln!("stop reason     = {stop}");
        eprintln!(
            "final INTENABLE = {:#010x}  INTERRUPT = {:#010x}  INTLEVEL = {}",
            proc.cpu.intenable,
            proc.cpu.interrupt,
            proc.cpu.regs.intlevel()
        );
        eprintln!("armed_at instr  = {armed_at:?}");
        eprintln!("first INTLEVEL==0 after arm = {first_level0_after_arm:x?} (level-1 doorbell deliverability window)");
        eprintln!("--- {} SR transition(s) ---", changes.len());
        for (i, pc, sym, which, old, new) in &changes {
            eprintln!("{i:>7} pc={pc:#08x} {sym:<24} {which} {old:#010x} -> {new:#010x}");
        }
    }

    /// M2c Phase 2 DIAGNOSTIC (iter13): is the exception dispatcher's entry
    /// `l32r a15` value LOAD-BEARING for the "main returns" wall? The dispatcher at
    /// runtime 0x28b4 loads a15 from a literal whose PC-relative target wraps to
    /// 0xfffe3094 (stubbed to 0, provenance unresolved -- see
    /// `exception-dispatch-pc-verdict.md`). This probe FORCES a15 to a chosen value
    /// right after that l32r executes and reports where boot then walls, plus where
    /// the dispatcher returns. If the wall (instr count / stop / return target) is
    /// INVARIANT across forced a15 values, the 0xfffe3094 read is a red herring; if
    /// it changes, a15 is load-bearing and worth deriving. Set XDNA_FW_FORCE_A15 to
    /// a hex value to force; unset = control (a15 = the stub's 0). Ignored unless
    /// XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_a15_loadbearing() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the a15 load-bearing probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let force: Option<u32> = std::env::var("XDNA_FW_FORCE_A15").ok().map(|s| {
            u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).expect("XDNA_FW_FORCE_A15 must be hex")
        });

        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        const MAX: u64 = 200_000;
        const DISPATCHER_L32R_PC: u32 = 0x28b4; // entry FLIX bundle: l32r a15,<lit>
        const DISPATCHER_RETW_PC: u32 = 0x291c; // retw.n that ends the dispatcher
        let mut n = 0u64;
        let mut stop = String::from("budget reached");
        let mut dispatcher_hits = 0u64;
        let mut forced_events = 0u64;
        let mut retw_returns: Vec<(u64, u32)> = Vec::new(); // (n, return-target pc)
        let mut in_dispatcher = false;

        while n < MAX {
            let pc = proc.cpu.pc;
            if pc == DISPATCHER_L32R_PC {
                dispatcher_hits += 1;
                in_dispatcher = true;
            }
            let at_retw = pc == DISPATCHER_RETW_PC && in_dispatcher;

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

            // Override a15 right after the dispatcher's entry l32r executed (pc has
            // advanced past 0x28b4; the l32r set a15=0). a15 is untouched between the
            // l32r and its consumer `bnez.n a15` at 0x28bf, so this override is seen.
            if pc == DISPATCHER_L32R_PC {
                if let Some(v) = force {
                    proc.cpu.regs.write_ar(15, v);
                    forced_events += 1;
                }
            }
            if at_retw {
                retw_returns.push((n, proc.cpu.pc));
                in_dispatcher = false;
            }

            if let Some(addr) = proc.bus.sysstub().spinning() {
                stop = format!("sysstub spin at {addr:#x}");
                break;
            }
        }

        eprintln!("=== M2c a15 load-bearing probe ===");
        match force {
            Some(v) => eprintln!("forced a15 = {v:#x} ({forced_events}x)"),
            None => eprintln!("control run (a15 = stub 0, not forced)"),
        }
        eprintln!("instrs executed = {n}");
        eprintln!("stop reason     = {stop}");
        eprintln!("dispatcher hits (pc=0x28b4) = {dispatcher_hits}");
        eprintln!("dispatcher retw returns     = {retw_returns:?}");
    }

    /// M2c Phase 2 DIAGNOSTIC (iter13): does the firmware COPY code into the low
    /// window as DATA (which our Harvard model routes to `local_data`), then FETCH
    /// it (which reads the pristine image)? The syscall cause-handler target
    /// (0xe1fc) reads as zeros in the image; if `local_data` holds real code there,
    /// the low window is unified IRAM/DRAM at that address and our fetch/data split
    /// is wrong for it (iter12's deferred "fork (b)"). Boots to the wall (init runs
    /// first) and dumps image vs `local_data` at XDNA_FW_DUMP_ADDR (default 0xe1fc).
    /// Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_low_window_code() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the low-window code probe");
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
        let mut n = 0u64;
        let mut syscall_ps: Option<u32> = None;
        while n < MAX {
            let pc = proc.cpu.pc;
            // Capture PS at the boot syscall (the MERT hand-off): PS.UM (bit 5)
            // decides user- vs kernel-mode exception routing.
            if pc == 0x08b0_43e1 && syscall_ps.is_none() {
                syscall_ps = Some(proc.cpu.regs.ps);
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(_) => {
                    n += 1;
                    if proc.cpu.pc == pc {
                        break;
                    }
                }
                Step::Unknown { .. } => break,
            }
            if proc.bus.sysstub().spinning().is_some() {
                break;
            }
        }
        match syscall_ps {
            Some(ps) => eprintln!(
                "syscall PS = {ps:#x}  INTLEVEL={} EXCM={} UM(user)={} RING={} WOE={}",
                ps & 0xf,
                (ps >> 4) & 1,
                (ps >> 5) & 1,
                (ps >> 6) & 3,
                (ps >> 18) & 1
            ),
            None => eprintln!("syscall PS = <syscall pc 0x8b043e1 not reached>"),
        }

        let base = std::env::var("XDNA_FW_DUMP_ADDR")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .unwrap_or(0xe1fc);
        eprintln!("=== low-window handler-region dump @ {base:#x} (after {n} instrs) ===");
        eprintln!("(image = fetch source / IRAM; local_data = data-write overlay / DRAM)");
        let mut any_differ = false;
        for i in 0..16u32 {
            let a = base + i * 4;
            let img_w = proc.bus.load32(a); // physical Rom path == image (fetch source)
            let loc_w = proc.bus.load_local32(a); // local_data overlay (data writes)
            let flag = if img_w != loc_w {
                any_differ = true;
                "   <-- DIFFER"
            } else {
                ""
            };
            eprintln!("  {a:#010x}: image={img_w:#010x}  local_data={loc_w:#010x}{flag}");
        }
        eprintln!(
            "verdict: {}",
            if any_differ {
                "local_data holds code the image lacks -> low window is unified IRAM/DRAM here (fork b)"
            } else {
                "image and local_data agree -> handler genuinely absent; not a fetch/data-split issue"
            }
        );
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
