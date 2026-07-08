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

    /// Characterization lock (MMU data-path design, 2026-07-06): the
    /// translation-authoritative data path depends on the low DRAM window being
    /// TLB-covered from reset through steady state. Assert the STRUCTURAL facts,
    /// not just a point sample: way-6 ei0 is the reset identity region, the
    /// prologue clears entry 1 (code region) not entry 0, asid resolves ring 0,
    /// and every low-window data probe still translates identity at steady state.
    #[test]
    fn low_window_dram_is_translation_covered_from_reset() {
        use crate::firmware::xtensa::mmu::Mmu;
        // (a) Reset populates way-6 ei0 = VPN 0 -> paddr 0, asid 1, attr 3, variable.
        let fresh = Mmu::new_with_varway56(true);
        let e0 = fresh.dtlb[6][0];
        assert_eq!(e0.vaddr, 0);
        assert_eq!(e0.paddr, 0);
        assert_eq!(e0.asid, 1);
        assert_eq!(e0.attr, 3, "RWX -- grants low-window read AND write");
        assert!(e0.variable);
        // (c) asid 1 always resolves to ring 0 (write_rasid forces the ring-0 byte).
        let mut r = Mmu::new_with_varway56(true);
        r.write_rasid(0x08070605);
        assert_eq!(r.lookup(0x0000_f9e0, true).expect("low window resolves").ring, 0);

        // (b) + (d) need the real boot; skip cleanly without the binary.
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary absent -- structural (a)/(c) still checked");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        for _ in 0..300_000 {
            if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
                break;
            }
        }
        // (b) prologue invalidated way-6 entry 1 (code region), left entry 0 (low window).
        assert_eq!(proc.cpu.mmu.dtlb[6][0].asid, 1, "low-window entry 0 still live");
        assert_eq!(proc.cpu.mmu.dtlb[6][1].asid, 0, "code-region entry 1 invalidated");
        // (d) every low-window data address the firmware touches translates identity.
        for a in [0x0000_f9e0u32, 0x0000_9070, 0x0000_2278, 0x0000_2250, 0x0000_22bc] {
            let t = proc.cpu.mmu.translate(&mut proc.bus, a, 0 /*load*/, 0).expect("resolves");
            assert_eq!(t.paddr, a, "low-window data translates identity");
        }
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
    ///
    /// IGNORED pending the per-task completion redesign. The single-post one-shot
    /// delivery this asserts does NOT unblock the real firmware: the lone mailbox
    /// post fires at instr ~6972 (an early alive-handshake) before the scheduler
    /// is up, and the recursion blocks >=2 distinct tasks (0x10f10 at ~41k, 0x9040
    /// at ~59k) each on its own done-flag -- one post cannot map to both. Deep RE
    /// of the per-task completion causality is in progress; when the faithful
    /// per-task model lands, remove this `ignore` (the assertion itself is still
    /// the right gate). See the iter18 completion-model findings.
    #[test]
    #[ignore = "pending per-task completion redesign (deep RE in progress) -- one-shot post model does not unblock real fw"]
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
        eprintln!("done-flag[0x9070] = {:#x}", proc.cpu.data_read32(&mut proc.bus, 0x9070).unwrap_or(0));

        // The completion fired: the local done-flag is set.
        assert_ne!(
            proc.cpu.data_read32(&mut proc.bus, 0x9070).unwrap_or(0),
            0,
            "completion delivered the done-flag"
        );
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

    /// Like [`nearest_symbol`] but returns the bucket KEY (the symbol entry
    /// address, or the 4KB page when no symbol is near) plus a bare routine
    /// label -- so cycle-accounting aggregates a whole routine into one bucket
    /// instead of splitting it by offset. Every PC maps to exactly one bucket,
    /// which is the "every cycle accounted" invariant the histogram relies on.
    fn routine_bucket(symbols: &std::collections::HashMap<u32, String>, pc: u32) -> (u32, String) {
        const MAX_SPAN: u32 = 0x800;
        let mut best: Option<(u32, &str)> = None;
        for (&addr, name) in symbols {
            if addr <= pc && pc - addr < MAX_SPAN && best.map_or(true, |(b, _)| addr > b) {
                best = Some((addr, name.as_str()));
            }
        }
        match best {
            Some((addr, name)) => (addr, name.to_string()),
            None => {
                let page = pc & !0xfff;
                (page, format!("<no-sym {page:#x}>"))
            }
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

    /// M2c #140 RE TOOL: static scan of every RSR/WSR/XSR in the firmware,
    /// histogrammed by special-register number, with the known names. Answers
    /// "does the firmware use the Xtensa TIMER?" -- if it programs CCOMPARE
    /// (0xF0/F1/F2) it drives a timer-tick interrupt, and since the interp
    /// models CCOUNT/CCOMPARE as no-ops (interp/mod.rs: "every other SR is a
    /// logged no-op"), that tick can NEVER fire in EMU -> the event-poll run-fn
    /// is starved by construction. The CCOMPARE period would then BE the coarse
    /// completion-poll cadence that dominates the fitted 8000-cycle dma_wait.
    /// Walks all code via the reset way-6 identity region (covers never-executed
    /// code). Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_sr_usage() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the SR-usage scan");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        fn sr_name(sr: u16) -> &'static str {
            match sr {
                0x00 => "LBEG",
                0x01 => "LEND",
                0x02 => "LCOUNT",
                0x03 => "SAR",
                0x48 => "WINDOWBASE",
                0x49 => "WINDOWSTART",
                0x53 => "PTEVADDR",
                0x5A => "ITLBCFG",
                0x5B => "DTLBCFG",
                0x60 => "IBREAKENABLE",
                0x83 => "EPC1",
                0x90 => "EPS2",
                0xB0 => "DEPC",
                0xB1 => "EXCSAVE1",
                0xC0 => "EXCCAUSE",
                0xD1 => "EXCVADDR",
                0xE2 => "INTERRUPT/INTSET",
                0xE3 => "INTCLEAR",
                0xE4 => "INTENABLE",
                0xE6 => "PS",
                0xE7 => "VECBASE",
                0xEA => "CCOUNT (TIMER)",
                0xF0 => "CCOMPARE0 (TIMER)",
                0xF1 => "CCOMPARE1 (TIMER)",
                0xF2 => "CCOMPARE2 (TIMER)",
                _ => "",
            }
        }

        let mut syms: Vec<(u32, String)> = proc.symbols.iter().map(|(a, n)| (*a, n.clone())).collect();
        syms.sort_by_key(|(a, _)| *a);

        // sr -> (rsr, wsr, xsr, first_pc)
        let mut usage: std::collections::BTreeMap<u16, (u32, u32, u32, u32)> =
            std::collections::BTreeMap::new();
        for i in 0..syms.len() {
            let start = syms[i].0;
            let end = syms.get(i + 1).map(|(a, _)| *a).unwrap_or(start + 0x400).min(start + 0x2000);
            let mut pc = start;
            while pc < end {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, pc + k as u32));
                let d = decode::decode(&b, pc);
                let hit = match d.op {
                    decode::Op::Rsr { sr, .. } => Some((sr as u16, 0)),
                    decode::Op::Wsr { sr, .. } => Some((sr as u16, 1)),
                    _ => None,
                };
                if let Some((sr, kind)) = hit {
                    let e = usage.entry(sr).or_insert((0, 0, 0, pc));
                    match kind {
                        0 => e.0 += 1,
                        1 => e.1 += 1,
                        _ => e.2 += 1,
                    }
                }
                pc += (d.len as u32).max(1);
            }
        }

        eprintln!("=== M2c static SR-usage scan (#140 timer hunt) ===");
        eprintln!("  sr      rsr wsr xsr  first-pc            name");
        let mut timer_used = false;
        for (&sr, &(r, w, x, first_pc)) in &usage {
            let name = sr_name(sr);
            if matches!(sr, 0xEA | 0xF0 | 0xF1 | 0xF2) {
                timer_used = true;
            }
            eprintln!(
                "  {sr:#04x}   {r:>3} {w:>3} {x:>3}  {first_pc:#08x} {:<20} {name}",
                nearest_symbol(&proc.symbols, first_pc)
            );
        }
        eprintln!(
            "--- TIMER (CCOUNT/CCOMPARE) used by firmware: {} ---",
            if timer_used {
                "YES -- timer-driven tick; interp models it as no-op"
            } else {
                "NO"
            }
        );
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

    /// M2c iter18 (Session-4) DIAGNOSTIC: static L32r-literal cross-reference.
    /// The AIE-completion ISR is the last uncharted link in the completion path
    /// (it reads the AIE interrupt-status register and posts an event message
    /// that `wake_tasks_by_event_mask` turns into a pending-mask bit). It can't
    /// be found dynamically -- no interrupt fires in EMU without an array model.
    /// So scan every symbol function for `l32r` instructions whose resolved
    /// literal VALUE falls in a target range (the AIE mgmt register aperture,
    /// default `0x2701_0000..0x2701_2000`), and report each load site + the
    /// literal value + the containing function. Whoever loads the int-status
    /// constant OUTSIDE `sched_event_poll` is the ISR (or its status helper).
    /// Range overridable via `XDNA_FW_LIT_LO`/`XDNA_FW_LIT_HI` (hex). Ignored
    /// unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_literal_xref() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the literal-xref probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let parse_hex = |k: &str, d: u32| -> u32 {
            std::env::var(k)
                .ok()
                .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
                .unwrap_or(d)
        };
        let lo = parse_hex("XDNA_FW_LIT_LO", 0x2701_0000);
        let hi = parse_hex("XDNA_FW_LIT_HI", 0x2701_2000);
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        let mut syms: Vec<(u32, String)> = proc.symbols.iter().map(|(a, n)| (*a, n.clone())).collect();
        syms.sort_by_key(|(a, _)| *a);

        eprintln!("=== M2c L32r-literal xref: value in {lo:#x}..{hi:#x} ===");
        let mut hits = 0u64;
        for i in 0..syms.len() {
            let start = syms[i].0;
            let end = syms.get(i + 1).map(|(a, _)| *a).unwrap_or(start + 0x400).min(start + 0x2000);
            let fname = syms[i].1.clone();
            let mut pc = start;
            while pc < end {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, pc + k as u32));
                let d = decode::decode(&b, pc);
                if let decode::Op::L32r { target, .. } = d.op {
                    let lit: [u8; 4] =
                        std::array::from_fn(|k| proc.bus.fetch8(target + k as u32, target + k as u32));
                    let val = u32::from_le_bytes(lit);
                    if val >= lo && val < hi {
                        eprintln!("  {pc:#08x} {fname:<26} l32r -> lit@{target:#x} = {val:#010x}");
                        hits += 1;
                    }
                }
                pc += (d.len as u32).max(1);
            }
        }
        eprintln!("total hits = {hits}");
    }

    /// M2c iter18 (Session-4) EXPERIMENT: inject the AIE-completion interrupt.
    /// The completion path is interrupt-driven: on silicon the AIE array raises
    /// an IRQ whose status appears in `0x27010d28`; the level-1 ISR (via the
    /// general-exc handler `0x2958`, EXCCAUSE=4) reads it, posts an event
    /// message, and `wake_tasks_by_event_mask(1<<id)` sets task 0x10f10's pending
    /// mask `[0x10f40]`. Nothing fires that IRQ in EMU (no array model), so boot
    /// wedges. This probe FAITHFULLY drives the interrupt: warm up to steady
    /// state, seed the AIE status reg, set `cpu.interrupt`, and keep stepping --
    /// `step()` delivers the level-1 interrupt as soon as PS.INTLEVEL returns to
    /// 0. Watches whether the interrupt is taken, whether the ISR/event path
    /// runs, whether `[0x10f40]` gets set, and how far boot advances -- pinning
    /// the completion contract end-to-end (the last boundary inch + the first
    /// rail of H-b delivery). Env: XDNA_FW_INT_WARMUP (default 60000),
    /// XDNA_FW_INT_STATUS (hex, seed of 0x27010d28, default 0xffffffff),
    /// XDNA_FW_INT_LINE (hex, cpu.interrupt bits; 0 => use current INTENABLE),
    /// XDNA_FW_INT_RUN (steps after inject, default 400000), XDNA_FW_INT_RESEED
    /// (presence => re-seed status every step). Ignored unless XDNA_FW_PROBE set.
    #[test]
    fn m2c_probe_inject_interrupt() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the interrupt-injection experiment");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        let env_hex = |k: &str, d: u32| -> u32 {
            std::env::var(k)
                .ok()
                .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
                .unwrap_or(d)
        };
        let warmup = env_u64("XDNA_FW_INT_WARMUP", 60_000);
        let status_val = env_hex("XDNA_FW_INT_STATUS", 0xffff_ffff);
        let line = env_hex("XDNA_FW_INT_LINE", 0);
        let run = env_u64("XDNA_FW_INT_RUN", 400_000);
        let reseed = std::env::var("XDNA_FW_INT_RESEED").is_ok();

        const STATUS_REG: u32 = 0x2701_0d28;
        const GEN_EXC: u32 = 0x2958; // GENERAL_EXCEPTION_HANDLER
        const WAKE: u32 = 0xd84c; // wake_tasks_by_event_mask
        const PENDING: u32 = 0x10f40; // task 0x10f10's pending-event mask
        const CUR_TASK: u32 = 0x2278; // scheduler current-task ptr

        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        // Warm up to steady state.
        for _ in 0..warmup {
            if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
                break;
            }
        }
        let intenable = proc.cpu.intenable;
        let fire = if line == 0 { intenable } else { line };
        eprintln!("=== M2c interrupt-injection experiment ===");
        eprintln!(
            "at warmup n={warmup}: pc={:#x} {}",
            proc.cpu.pc,
            nearest_symbol(&proc.symbols, proc.cpu.pc)
        );
        eprintln!(
            "  INTENABLE={intenable:#010x} INTERRUPT={:#010x} intlevel={} excm={}",
            proc.cpu.interrupt,
            proc.cpu.regs.intlevel(),
            proc.cpu.regs.excm()
        );
        eprintln!(
            "  current-task [0x2278]={:#x}  pending [0x10f40]={:#x}",
            proc.cpu.data_read32(&mut proc.bus, CUR_TASK).unwrap_or(0),
            proc.cpu.data_read32(&mut proc.bus, PENDING).unwrap_or(0)
        );
        eprintln!(
            "  seeding {STATUS_REG:#x}={status_val:#x}, setting INTERRUPT |= {fire:#010x}, reseed={reseed}"
        );

        // Inject.
        let _ = proc.cpu.data_write32(&mut proc.bus, STATUS_REG, status_val);
        proc.cpu.interrupt |= fire;

        let mut n = warmup;
        let mut min_intlevel = 15u32;
        let mut lvl0_windows = 0u64;
        let mut first_gen_exc: Option<u64> = None;
        let mut first_wake: Option<u64> = None;
        let mut first_pending: Option<(u64, u32)> = None;
        let mut stop = String::from("run budget reached");
        let end = warmup + run;
        while n < end {
            if reseed {
                let _ = proc.cpu.data_write32(&mut proc.bus, STATUS_REG, status_val);
            }
            let pc = proc.cpu.pc;
            if pc == GEN_EXC && first_gen_exc.is_none() {
                first_gen_exc = Some(n);
            }
            if pc == WAKE && first_wake.is_none() {
                first_wake = Some(n);
            }
            if first_pending.is_none() {
                let p = proc.cpu.data_read32(&mut proc.bus, PENDING).unwrap_or(0);
                if p != 0 {
                    first_pending = Some((n, p));
                }
            }
            let lvl = proc.cpu.regs.intlevel();
            if lvl < min_intlevel {
                min_intlevel = lvl;
            }
            if lvl == 0 && !proc.cpu.regs.excm() {
                lvl0_windows += 1;
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?}) at pc={pc:#x} (idle -- boot reached waiti!)");
                    break;
                }
                Step::Unknown { pc: upc, word } => {
                    stop = format!("Unknown at pc={upc:#x} word={word:#010x}");
                    break;
                }
            }
        }
        eprintln!("--- results ---");
        eprintln!("ran to n={n}  (advanced {} past warmup)", n - warmup);
        eprintln!("stop reason         = {stop}");
        eprintln!("min intlevel seen   = {min_intlevel}   (level-0 deliverable windows = {lvl0_windows})");
        eprintln!("interrupt taken (@0x2958) first at n = {first_gen_exc:?}");
        eprintln!("wake_tasks (@0xd84c)     first at n = {first_wake:?}");
        eprintln!("[0x10f40] pending set    first at   = {first_pending:x?}");
        eprintln!(
            "final INTERRUPT={:#010x} intlevel={} excm={}",
            proc.cpu.interrupt,
            proc.cpu.regs.intlevel(),
            proc.cpu.regs.excm()
        );
        eprintln!("final pc={:#x} {}", proc.cpu.pc, nearest_symbol(&proc.symbols, proc.cpu.pc));
        eprintln!(
            "final current-task [0x2278]={:#x}",
            proc.cpu.data_read32(&mut proc.bus, CUR_TASK).unwrap_or(0)
        );
    }

    /// EXTERNAL-COMPLETE EXPERIMENT (2026-07-08): deliver the column-power
    /// completion the way silicon would -- an external agent writes the task
    /// done-flag `[task+0x30]` AND column `[task+8]` ONCE while the CPU idles in
    /// the poll loop -- then let the NEXT natural dispatch consume it. This
    /// tests two things at once:
    ///   (1) natural-completion contract: does completing the column-power
    ///       worker(s) advance boot to the go-alive run-fn (0x55f8) / publisher
    ///       (0x50e8) / idle `waiti` (0x56e6), dropping INTLEVEL to 0?
    ///   (2) the breach-corruption pivot: is the go-alive record at SCHED+0xd0
    ///       (0x2320) still intact afterward? The prior "corruption" was seen
    ///       forcing REPEATEDLY at the done-check PC (0xd828) deep in the
    ///       dispatch stack; a single external write at the idle loop is the
    ///       faithful delivery and may consume cleanly.
    /// Env: XDNA_FW_EC_WARMUP (default 80_000), XDNA_FW_EC_RUN (default
    /// 800_000), XDNA_FW_EC_TASKS (hex csv, default 0x9040,0x10f10),
    /// XDNA_FW_EC_COL (hex, column to assign, default 0). Ignored unless
    /// XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_external_complete() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the external-complete experiment");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        let env_hex = |k: &str, d: u32| -> u32 {
            std::env::var(k)
                .ok()
                .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
                .unwrap_or(d)
        };
        let warmup = env_u64("XDNA_FW_EC_WARMUP", 80_000);
        let run = env_u64("XDNA_FW_EC_RUN", 800_000);
        let col = env_hex("XDNA_FW_EC_COL", 0);
        let tasks: Vec<u32> = std::env::var("XDNA_FW_EC_TASKS")
            .unwrap_or_else(|_| "0x9040,0x10f10".to_string())
            .split(',')
            .filter_map(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .collect();

        const RECORD: u32 = 0x2320; // go-alive task record (SCHED+0xd0)
        const CUR_TASK: u32 = 0x2278; // scheduler current-task ptr
        const PUBLISH: u32 = 0x50e8; // publish_chann_info
        const GOALIVE: u32 = 0x55f8; // goalive_runfn
        const WAITI: u32 = 0x56e6; // idle waiti 0
        const MAGIC: u32 = 0x5550_4e5f; // FW_ALIVE magic ("_NPU")

        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        let syms = load_symbols();

        let dump4 = |proc: &mut FirmwareProcessor, base: u32| -> [u32; 4] {
            std::array::from_fn(|i| proc.cpu.data_read32(&mut proc.bus, base + (i as u32) * 4).unwrap_or(0))
        };

        // Warm up to the steady idle loop.
        for _ in 0..warmup {
            if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
                break;
            }
        }
        let rec_before = dump4(&mut proc, RECORD);
        let cur = proc.cpu.data_read32(&mut proc.bus, CUR_TASK).unwrap_or(0);
        eprintln!("=== external-complete experiment ===");
        eprintln!(
            "warmup n={warmup}: pc={:#x} {} intlevel={} current-task[0x2278]={cur:#x}",
            proc.cpu.pc,
            nearest_symbol(&syms, proc.cpu.pc),
            proc.cpu.regs.intlevel(),
        );
        eprintln!("[0x2320] BEFORE = {:08x?}", rec_before);

        // Deliver the completion ONCE for each whitelisted task: flag + column.
        for &t in &tasks {
            let _ = proc.cpu.data_write32(&mut proc.bus, t.wrapping_add(0x30), 1);
            let _ = proc.cpu.data_write32(&mut proc.bus, t.wrapping_add(8), col);
        }
        eprintln!("delivered flag+col={col:#x} to tasks {:x?} (once, at idle)", tasks);

        let mut n = warmup;
        let end = warmup + run;
        let mut min_intlevel = 15u32;
        let mut first: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        let mut magic_seen = false;
        let mut stop = String::from("run budget reached");
        while n < end {
            let pc = proc.cpu.pc;
            for &t in &[PUBLISH, GOALIVE, WAITI] {
                if pc == t {
                    first.entry(t).or_insert(n);
                }
            }
            let lvl = proc.cpu.regs.intlevel();
            if lvl < min_intlevel {
                min_intlevel = lvl;
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?}) at pc={pc:#x} (IDLE -- reached waiti!)");
                    break;
                }
                Step::Unknown { pc: upc, word } => {
                    stop = format!("Unknown at pc={upc:#x} word={word:#010x}");
                    break;
                }
            }
            // Cheap alive-magic detector: watch the publisher's struct buffer as
            // it is built (magic stored into [buf+0x20]). Scanned lazily below.
            if !magic_seen && first.contains_key(&PUBLISH) {
                magic_seen = true; // reaching the publisher IS the alive signal
            }
        }

        let rec_after = dump4(&mut proc, RECORD);
        eprintln!("--- results ---");
        eprintln!("ran to n={n} (advanced {} past warmup)  stop={stop}", n - warmup);
        eprintln!("min intlevel seen  = {min_intlevel}  (0 == receive-ready idle)");
        eprintln!(
            "reached: publisher(0x50e8)={:?} goalive(0x55f8)={:?} waiti(0x56e6)={:?}",
            first.get(&PUBLISH),
            first.get(&GOALIVE),
            first.get(&WAITI),
        );
        eprintln!("alive-path entered = {magic_seen} (magic {MAGIC:#x} published if publisher ran)");
        eprintln!("[0x2320] AFTER  = {:08x?}", rec_after);
        eprintln!("record intact = {} (run-fn word still {:#x})", rec_after[0] == GOALIVE, rec_after[0],);
        eprintln!(
            "final pc={:#x} {} intlevel={}",
            proc.cpu.pc,
            nearest_symbol(&syms, proc.cpu.pc),
            proc.cpu.regs.intlevel(),
        );
    }

    /// M4.1 EXPERIMENT: discover the x2i (host->firmware) mailbox receive
    /// address. Boot to idle via the completion-agent bootstrap (`waiti`), then
    /// inject the mailbox doorbell (Xtensa INTERRUPT bit0) and capture every
    /// Mailbox-aperture access the interrupt handler issues. The FIRST mailbox
    /// READ after the doorbell is the firmware's x2i-tail poll -- the address the
    /// host producer must write to post a request. Discovered by observation, not
    /// guessed (the driver leaves x2i offsets firmware-defined). Ignored unless
    /// XDNA_FW_PROBE is set. Env: XDNA_FW_MB_BOOT (idle budget, default 2_000_000),
    /// XDNA_FW_MB_STEPS (post-doorbell steps, default 4000).
    #[test]
    fn m2c_probe_mailbox_receive() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the mailbox-receive discovery probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        let boot_budget = env_u64("XDNA_FW_MB_BOOT", 2_000_000);
        let steps = env_u64("XDNA_FW_MB_STEPS", 4000);

        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        // Boot to the command-loop idle (completion-agent bootstrap gets past the
        // task-B wall to `waiti`).
        proc.enable_host_mailbox();
        let report = proc.boot_to_idle(boot_budget);
        eprintln!("=== M4.1 mailbox-receive discovery ===");
        eprintln!(
            "boot: reached_idle={} instrs={} last_pc={:#x} {} INTENABLE={:#010x}",
            report.reached_idle,
            report.instrs_executed,
            report.last_pc,
            nearest_symbol(&proc.symbols, report.last_pc),
            proc.cpu.intenable,
        );
        if !report.reached_idle {
            eprintln!("did NOT reach idle -- cannot exercise the receive path; aborting");
            return;
        }

        // Inject the mailbox doorbell: ensure bit0 is enabled and raise it.
        proc.cpu.intenable |= 1;
        proc.cpu.interrupt |= 1;
        proc.bus.arm_probe();

        let mut n = 0u64;
        let mut stop = String::from("steps budget reached");
        while n < steps {
            proc.bus.set_probe_pc(proc.cpu.pc);
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?}) at pc={:#x} (returned to idle)", proc.cpu.pc);
                    break;
                }
                Step::Unknown { pc, word } => {
                    stop = format!("Unknown at pc={pc:#x} word={word:#010x}");
                    break;
                }
            }
        }
        let log = proc.bus.take_probe();
        eprintln!("post-doorbell: ran {n} steps, stop={stop}");
        eprintln!("mailbox/array/system accesses after doorbell = {}", log.len());
        eprintln!("--- accesses in order (seq: pc[sym] region rd/wr addr=value wN) ---");
        for a in &log {
            eprintln!(
                "{:>5}: pc={:#x}[{}] {:?} {} {:#x}={:#x} w{}",
                a.seq,
                a.pc,
                nearest_symbol(&proc.symbols, a.pc),
                a.region,
                if a.is_write { "wr" } else { "rd" },
                a.addr,
                a.value,
                a.width,
            );
        }
    }

    /// M3 EXPERIMENT (the lever M1 unlocked): boot the firmware with a REAL AIE2
    /// array attached to its bus, vs the stub, and compare. Until M1 the Array
    /// aperture was a discard stub -- every firmware array READ returned 0. With
    /// a `DeviceState` attached, array reads return real register values, so if
    /// the firmware branches on array state during boot the control flow can
    /// diverge. The banked wall was "the completion contract lives in the array,
    /// not derivable from firmware alone" -- this is the first test of that
    /// hypothesis with an actual array. Runs the same boot twice (stub, attached),
    /// reports where each stops and every Array-aperture access. Ignored unless
    /// XDNA_FW_PROBE set. Env: XDNA_FW_MAX (budget, default 700_000).
    #[test]
    fn m2c_probe_boot_with_array() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the array-attached boot experiment");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(700_000);
        let raw = std::fs::read(&path).expect("read firmware");

        // Run one boot; return (instrs, last_pc, stop, array-access log).
        let run = |attach: bool| -> (u64, u32, String, Vec<mmio::StubAccess>) {
            let img = FirmwareImage::parse(&raw).expect("parse");
            let mut proc = FirmwareProcessor::load_m2c(img);
            if attach {
                proc.bus.attach_device(crate::device::DeviceState::new_npu1());
            }
            proc.bus.arm_probe();
            let mut n = 0u64;
            let mut stop = String::from("budget reached");
            while n < max {
                proc.bus.set_probe_pc(proc.cpu.pc);
                let pc = proc.cpu.pc;
                match proc.cpu.step(&mut proc.bus) {
                    Step::Ran | Step::Exception { .. } => n += 1,
                    Step::Wait(reason) => {
                        n += 1;
                        stop = format!("Wait({reason:?}) at pc={pc:#x} (IDLE)");
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
            let last_pc = proc.cpu.pc;
            (n, last_pc, stop, proc.bus.take_probe())
        };

        let syms = load_symbols();
        for (label, attach) in [("STUB", false), ("ATTACHED", true)] {
            let (n, last_pc, stop, log) = run(attach);
            let array: Vec<_> = log.iter().filter(|a| a.region == Region::Array).collect();
            let reads = array.iter().filter(|a| !a.is_write).count();
            let writes = array.iter().filter(|a| a.is_write).count();
            eprintln!("=== boot [{label}] ===");
            eprintln!("  instrs={n} last_pc={last_pc:#x} {}", nearest_symbol(&syms, last_pc));
            eprintln!("  stop={stop}");
            eprintln!("  array accesses: {} (rd={reads} wr={writes})", array.len());
            // Distinct array sites (pc, addr, is_write) -> values seen.
            use std::collections::BTreeMap;
            let mut sites: BTreeMap<(u32, u32, bool), Vec<u32>> = BTreeMap::new();
            for a in &array {
                sites.entry((a.pc, a.addr, a.is_write)).or_default().push(a.value);
            }
            for ((pc, addr, is_wr), vals) in sites.iter().take(30) {
                let mut v = vals.clone();
                v.sort_unstable();
                v.dedup();
                let vs = v.iter().take(4).map(|x| format!("{x:#x}")).collect::<Vec<_>>().join(",");
                eprintln!(
                    "    pc={pc:#x} {} {addr:#x} vals=[{vs}] (n={})",
                    if *is_wr { "wr" } else { "rd" },
                    vals.len()
                );
            }
        }
    }

    /// FAITHFUL stub SMU (2026-07-07, Maya): the crude `m2c_probe_stub_smu_boot`
    /// forces EVERY dispatcher done-flag, which (post-decoder-fix) corrupts
    /// scheduler state once boot runs far enough to hit non-column tasks' checks
    /// -> Wall C panic. This one completes ONLY the real column-power worker tasks
    /// -- the ones that post the `0xfae0` colmask-0xf descriptor and wait on a
    /// per-column completion event the RE pinned to SMU/PSP bring-up (Session-4/6
    /// of the completion-causality finding). It sets the pending mask `[task+0x30]`
    /// at the dispatcher check (`0xd828`, task ptr in a4) ONLY for a whitelist of
    /// such tasks, at most once each, and lets every other task run for real.
    ///
    /// Setting the pending mask IS the faithful stand-in for the completion event:
    /// on silicon `wake_tasks_by_event_mask(1<<id)` sets that same bit and the
    /// dispatcher's own `deliver_pending_events` (0xd82c) consumes it -- we just
    /// supply the bit the (unmodeled) interrupt/ISR would have produced. The poll
    /// path (`FUN_00008c68` RAM bytes / `0x2727_n000` pages) was DEFINITIVELY
    /// falsified as the gate (Session-3), so this is NOT a register stub.
    ///
    /// Decisive question: does completing just the whitelist reach idle, or does a
    /// THIRD blocked task appear? The probe records, per task pointer seen at
    /// `0xd828` with flag==0, how many times it re-checks -- a non-whitelisted task
    /// with a high re-check count is a new column-power waiter (extend the list).
    /// Env: XDNA_FW_SMU_TASKS=hex,hex (whitelist, default 0x10f10,0x9040),
    /// XDNA_FW_MAX (budget, default 3_000_000), XDNA_FW_NOARRAY (skip array attach).
    /// Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_faithful_smu_boot() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the faithful-SMU boot experiment");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        let attach = std::env::var("XDNA_FW_NOARRAY").is_err();
        if attach {
            proc.bus.attach_device(crate::device::DeviceState::new_npu1());
        }

        // The column-power worker tasks whose completion the SMU/PSP would deliver.
        // These are task BASE pointers; the pending mask lives at base+0x30.
        let whitelist: std::collections::BTreeSet<u32> = std::env::var("XDNA_FW_SMU_TASKS")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok())
                    .collect()
            })
            .unwrap_or_else(|| [0x10f10, 0x9040].into_iter().collect());
        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3_000_000);
        // Optional: stop the first time execution reaches this PC, so the ring
        // shows the path INTO it (e.g. the entry to the 0x7fec panic spin).
        let stop_pc = std::env::var("XDNA_FW_STOP_PC")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok());
        const DONE_CHECK_PC: u32 = 0xd828; // dispatcher `l32i.n a10,[a4+0x30]`
        const KEEP: usize = 44;

        let mut n = 0u64;
        let mut stop = String::from("budget reached");
        // Which whitelisted tasks we've already completed (once each).
        let mut completed: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        let mut completed_order: Vec<(u32, u64)> = Vec::new();
        // Per task ptr seen at 0xd828 with flag==0: (recheck count, first n). A
        // non-whitelisted task with a large count is spinning = a new blocker.
        let mut blocked: std::collections::BTreeMap<u32, (u64, u64)> = std::collections::BTreeMap::new();
        let mut ring: std::collections::VecDeque<(u64, u32, String)> =
            std::collections::VecDeque::with_capacity(KEEP + 1);
        while n < max {
            let pc = proc.cpu.pc;
            if pc == DONE_CHECK_PC {
                let task = proc.cpu.regs.read_ar(4);
                let mask_addr = task.wrapping_add(0x30);
                let flag = proc.cpu.data_read32(&mut proc.bus, mask_addr).unwrap_or(0);
                if flag == 0 {
                    let e = blocked.entry(task).or_insert((0, n));
                    e.0 += 1;
                    // Faithful completion: only the whitelisted column-power tasks,
                    // once each. Everything else runs for real.
                    if whitelist.contains(&task) && completed.insert(task) {
                        let _ = proc.cpu.data_write32(&mut proc.bus, mask_addr, 1);
                        completed_order.push((task, n));
                    }
                }
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
            if Some(pc) == stop_pc {
                stop = format!("XDNA_FW_STOP_PC {pc:#x} reached at n={n}");
                break;
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?}) at pc={pc:#x} (IDLE!)");
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

        eprintln!("=== faithful-SMU boot (array attached: {attach}) ===");
        eprintln!("whitelist = {:x?}", whitelist.iter().map(|t| format!("{t:#x}")).collect::<Vec<_>>());
        eprintln!("instrs  = {n}");
        eprintln!("stop    = {stop}");
        eprintln!("last_pc = {:#x}  {}", proc.cpu.pc, nearest_symbol(&proc.symbols, proc.cpu.pc));
        eprintln!("--- completed (task, instr) in order ---");
        for (task, at) in &completed_order {
            eprintln!("  {task:#x} @ n={at}");
        }
        eprintln!("--- blocked tasks seen at 0xd828 with flag==0 (task: rechecks, first-n) ---");
        eprintln!("    (a non-whitelisted task with many rechecks is a NEW column-power waiter)");
        for (task, (count, first)) in &blocked {
            let tag = if whitelist.contains(task) {
                "whitelisted"
            } else {
                "UN-whitelisted"
            };
            eprintln!("  {task:#x}: {count:>8} rechecks, first@n={first}  [{tag}]");
        }
        eprintln!("--- last {} instrs before stop ---", ring.len());
        for (i, pc, disasm) in &ring {
            eprintln!("{i:>8} pc={pc:#08x} {:<24} {disasm}", nearest_symbol(&proc.symbols, *pc));
        }
    }

    /// COLUMN-COMMAND CONTRACT RE (2026-07-07, Maya "thoroughly understand every
    /// bit"): the boot worker posts per-column commands via `FUN_00007fa0`
    /// (real entry `0x7fc4`), which validates a **command type** in a3
    /// (`Bne a3,{12,15,18,20,257}`) and a **column index** in a7 (`Bgeui a7,6`).
    /// This probe captures EVERY call to `0x7fc4` with its full argument set
    /// (a2..a7), the current task, and the firmware-local descriptor at `0xfae0`
    /// (7 words) + response at `[0x1eb08]` -- so the VALID early calls (0x10f10's
    /// poll ~48k, a7<6, a3 in-set) and the ABORTING call (~623k, a7>=6) can be
    /// diffed. The delta IS the contract: what a correct column command looks
    /// like, which types occur, and what response the columns must produce.
    ///
    /// Faithful completion (whitelist {0x10f10,0x9040}) is applied so boot reaches
    /// the abort. Env: XDNA_FW_MAX (default 3_000_000). Ignored unless
    /// XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_col_cmd_trace() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the column-command contract trace");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        proc.bus.attach_device(crate::device::DeviceState::new_npu1());
        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3_000_000);

        const CMD_ENTRY: u32 = 0x7fc4; // FUN_00007fa0 real entry (per-column command)
        const DONE_CHECK_PC: u32 = 0xd828;
        const SCHED: u32 = 0x2278; // current-task ptr
        const DESC: u32 = 0xfae0; // colmask descriptor
        const RESP: u32 = 0x1eb08; // descriptor response
        let whitelist: [u32; 2] = [0x10f10, 0x9040];

        // (n, task, a2..a7, desc[0..7], resp) captured at each 0x7fc4 entry.
        struct Cmd {
            n: u64,
            task: u32,
            args: [u32; 6],
            desc: [u32; 7],
            resp: u32,
        }
        let mut cmds: Vec<Cmd> = Vec::new();
        let mut completed: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        // Ring of the last KEEP (n, pc, disasm, load/store EA+val), snapshotted at
        // the ABORTING command so we see how the bad col index (a7=0xff) was
        // derived from the (empty) response.
        const KEEP: usize = 140;
        let mut ring: std::collections::VecDeque<(u64, u32, String, String)> =
            std::collections::VecDeque::with_capacity(KEEP + 1);
        let mut aborting_ring: Option<Vec<(u64, u32, String, String)>> = None;
        let mut n = 0u64;
        let mut stop = String::from("budget reached");
        while n < max {
            let pc = proc.cpu.pc;
            // Record every instr into the ring with its data EA + value.
            let (disasm, ea_str) = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch)
            {
                Ok(phys) => {
                    let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                    let d = decode::decode(&b, pc);
                    let ea = match d.op {
                        Op::L32i { s, imm, .. }
                        | Op::L32iN { s, imm, .. }
                        | Op::L8ui { s, imm, .. }
                        | Op::L16ui { s, imm, .. }
                        | Op::L16si { s, imm, .. }
                        | Op::S32i { s, imm, .. }
                        | Op::S32iN { s, imm, .. }
                        | Op::S8i { s, imm, .. }
                        | Op::S16i { s, imm, .. } => {
                            let a = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                            format!("ea={:#x}={:#x}", a, proc.cpu.data_read32(&mut proc.bus, a).unwrap_or(0))
                        }
                        _ => String::new(),
                    };
                    (format!("{:?}", d.op), ea)
                }
                Err(_) => ("<fault>".to_string(), String::new()),
            };
            if ring.len() == KEEP {
                ring.pop_front();
            }
            ring.push_back((n, pc, disasm, ea_str));
            if pc == CMD_ENTRY {
                // If this is the aborting call (a7>=6), snapshot the lead-in ring.
                if proc.cpu.regs.read_ar(7) >= 6 && aborting_ring.is_none() {
                    aborting_ring = Some(ring.iter().cloned().collect());
                }
                let mut args = [0u32; 6];
                for (i, slot) in args.iter_mut().enumerate() {
                    *slot = proc.cpu.regs.read_ar((i + 2) as u8); // a2..a7
                }
                let mut desc = [0u32; 7];
                for (i, slot) in desc.iter_mut().enumerate() {
                    *slot = proc.cpu.data_read32(&mut proc.bus, DESC + (i as u32) * 4).unwrap_or(0);
                }
                cmds.push(Cmd {
                    n,
                    task: proc.cpu.data_read32(&mut proc.bus, SCHED).unwrap_or(0),
                    args,
                    desc,
                    resp: proc.cpu.data_read32(&mut proc.bus, RESP).unwrap_or(0),
                });
            }
            // Faithful completion so boot reaches the interesting later commands.
            if pc == DONE_CHECK_PC {
                let task = proc.cpu.regs.read_ar(4);
                let mask_addr = task.wrapping_add(0x30);
                if proc.cpu.data_read32(&mut proc.bus, mask_addr).unwrap_or(0) == 0
                    && whitelist.contains(&task)
                    && completed.insert(task)
                {
                    let _ = proc.cpu.data_write32(&mut proc.bus, mask_addr, 1);
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?}) at pc={pc:#x} (IDLE!)");
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

        // a3 command-type validity per the decoded 0x7fc4 validator.
        let a3_ok = |a3: u32| matches!(a3, 12 | 15 | 18 | 20 | 257);
        eprintln!("=== column-command contract trace ({} calls to 0x7fc4) ===", cmds.len());
        eprintln!("instrs = {n}; stop = {stop}");
        eprintln!("validators: a7 (col idx) must be <6 ; a3 (cmd type) must be in {{12,15,18,20,257}}");
        eprintln!(
            "{:>8}  {:<9}  {:>4} {:>4} {:>9} {:>9} {:>9} {:>4}  {:<24} {:>10}",
            "n", "task", "a2", "a3", "a4", "a5", "a6", "a7", "desc[0..7]", "resp[1eb08]"
        );
        for c in &cmds {
            let a3 = c.args[1];
            let a7 = c.args[5];
            let bad = if a7 >= 6 || !a3_ok(a3) { "  <== ABORTS" } else { "" };
            let desc_s = c.desc.iter().map(|w| format!("{w:#x}")).collect::<Vec<_>>().join(",");
            eprintln!(
                "{:>8}  {:#09x}  {:>4} {:>4} {:>#9x} {:>#9x} {:>#9x} {:>4}  {desc_s:<24} {:>#10x}{bad}",
                c.n, c.task, c.args[0], a3, c.args[2], c.args[3], c.args[4], a7, c.resp
            );
        }
        // Distinct command types seen (for the toolchain cross-reference).
        let mut types: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        for c in &cmds {
            *types.entry(c.args[1]).or_insert(0) += 1;
        }
        eprintln!("--- distinct a3 command types seen (type: count, valid?) ---");
        for (t, cnt) in &types {
            eprintln!("  {t:>4} ({t:#x}): {cnt:>4}  {}", if a3_ok(*t) { "VALID" } else { "invalid" });
        }
        if let Some(r) = &aborting_ring {
            eprintln!(
                "--- last {} instrs before the ABORTING command (how a7=0xff was derived) ---",
                r.len()
            );
            for (i, pc, disasm, ea) in r {
                eprintln!("{i:>8} pc={pc:#08x} {:<22} {disasm:<30} {ea}", nearest_symbol(&proc.symbols, *pc));
            }
        }
    }

    /// BREACH EXPERIMENT (2026-07-08): the pinned Wall-C contract says the
    /// external completion agent writes BOTH the task's pending flag `[task+0x30]`
    /// AND a valid column index `[task+8]` (the init default is `0xff` ->
    /// out-of-range -> the `Bgeui a7,6` abort at 0x7fec). Every prior force-done
    /// set only the flag and hit Wall C. This sets both (col 0) for the whitelist
    /// at the dispatcher done-check and reports how far boot then advances --
    /// past 623k toward idle == contract confirmed, external-agent-modelable.
    /// Env: XDNA_FW_MAX (default 3_000_000), XDNA_FW_COL (default 0). Ignored
    /// unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_colassign_boot() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the column-assign breach probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        proc.bus.attach_device(crate::device::DeviceState::new_npu1());
        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3_000_000);
        let col: u32 = std::env::var("XDNA_FW_COL")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0);

        const DONE_CHECK_PC: u32 = 0xd828;
        let whitelist: [u32; 2] = [0x10f10, 0x9040];
        let mut completed: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        let mut max_pc = 0u32;
        let mut n = 0u64;
        let mut stop = String::from("budget reached");
        // Region histogram (pc>>20) + a ring to catch the first jump into the
        // high (>=0x100000) window -- pinpoints the transition off firmware text.
        let mut regions: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        let mut ring: std::collections::VecDeque<(u64, u32)> = std::collections::VecDeque::with_capacity(41);
        let mut high_trap: Option<Vec<(u64, u32)>> = None;
        // Alive/idle detection: INTLEVEL floor AFTER the worker completes (0 ==
        // receive-ready idle), and the FW_ALIVE magic (0x55504e5f) publish.
        let mut min_il_post = 15u32;
        let mut alive_seen: Option<(u64, u32)> = None;
        // Third-gate localization: AFTER the 0x10f10 breach, where does boot rest?
        // Symbol-bucketed tail histogram + the load addresses it polls (the word
        // it now waits on names the next gate).
        let mut tail_hist: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
        let mut tail_reads: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        while n < max {
            let pc = proc.cpu.pc;
            if pc > max_pc {
                max_pc = pc;
            }
            if completed.contains(&0x10f10) {
                let il = proc.cpu.regs.intlevel();
                if il < min_il_post {
                    min_il_post = il;
                }
                *tail_hist.entry(nearest_symbol(&proc.symbols, pc & 0x00ff_ffff)).or_insert(0) += 1;
                if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                    let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                    let ea = match decode::decode(&b, pc).op {
                        Op::L32i { s, imm, .. }
                        | Op::L32iN { s, imm, .. }
                        | Op::L8ui { s, imm, .. }
                        | Op::L16ui { s, imm, .. } => Some(proc.cpu.regs.read_ar(s).wrapping_add(imm)),
                        _ => None,
                    };
                    if let Some(ea) = ea {
                        *tail_reads.entry(ea).or_insert(0) += 1;
                    }
                }
            }
            if alive_seen.is_none() {
                if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                    let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                    let sval = match decode::decode(&b, pc).op {
                        Op::S32i { t, .. } | Op::S32iN { t, .. } => Some(proc.cpu.regs.read_ar(t)),
                        _ => None,
                    };
                    if sval == Some(0x5550_4e5f) {
                        alive_seen = Some((n, pc));
                    }
                }
            }
            *regions.entry(pc >> 20).or_insert(0) += 1;
            if ring.len() == 40 {
                ring.pop_front();
            }
            ring.push_back((n, pc));
            if pc >= 0x2000_0000 && pc < 0x2100_0000 && high_trap.is_none() {
                high_trap = Some(ring.iter().cloned().collect());
            }
            if pc == DONE_CHECK_PC {
                let task = proc.cpu.regs.read_ar(4);
                if whitelist.contains(&task)
                    && proc.cpu.data_read32(&mut proc.bus, task.wrapping_add(0x30)).unwrap_or(0) == 0
                    && completed.insert(task)
                {
                    // The pinned contract: flag AND a valid column together.
                    let _ = proc.cpu.data_write32(&mut proc.bus, task.wrapping_add(0x30), 1);
                    let _ = proc.cpu.data_write32(&mut proc.bus, task.wrapping_add(8), col);
                    eprintln!("  n={n}: completed task {task:#x} with col={col} (flag+colidx)");
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?}) at pc={pc:#x} (IDLE!)");
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
        eprintln!("=== column-assign breach probe (col={col}) ===");
        eprintln!("instrs = {n}; max_pc = {max_pc:#x}; stop = {stop}");
        eprintln!("completed tasks: {:x?}", completed);
        eprintln!("min INTLEVEL after worker complete = {min_il_post}  (0 == receive-ready idle)");
        match alive_seen {
            Some((an, apc)) => eprintln!("FW_ALIVE magic 0x55504e5f STORED at n={an} pc={apc:#x}"),
            None => eprintln!("FW_ALIVE magic 0x55504e5f NOT stored"),
        }
        eprintln!("--- pc region histogram (pc>>20 : count) ---");
        for (r, c) in &regions {
            eprintln!("  {:#06x}xxxxx : {c}", r);
        }
        if let Some(ring) = &high_trap {
            eprintln!("--- ring into the first high-pc (>=0x100000) transition ---");
            for (i, pc) in ring {
                eprintln!("  n={i} pc={pc:#x}  {}", nearest_symbol(&proc.symbols, *pc));
            }
        }
        // Where does post-breach boot rest, and on what word? (the third gate)
        eprintln!("--- post-breach tail: top PC buckets (nearest function) ---");
        let mut ranked: Vec<_> = tail_hist.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        for (sym, hits) in ranked.iter().take(12) {
            eprintln!("  {hits:>9}  {sym}");
        }
        eprintln!("--- post-breach tail: top polled load addresses ---");
        let mut rreads: Vec<_> = tail_reads.into_iter().collect();
        rreads.sort_by(|a, b| b.1.cmp(&a.1));
        for (ea, hits) in rreads.iter().take(12) {
            eprintln!("  {ea:#010x}  hits={hits:>8}",);
        }
    }

    /// GO-ALIVE reachability (2026-07-08): the publish-channel-info + waiti-idle
    /// path lives in routine 0x5524, entered via run-fn 0x55f8 -- the run-fn of a
    /// task CREATED at 0x3de9 (`Call8 0xd664`, run-fn=0x55f8, col-sentinel=0xff),
    /// DISTINCT from worker 0x10f10 (run-fn 0x588c, a short returner we
    /// force-complete). With the same flag+column breach as `colassign_boot`,
    /// record the first-hit instruction count of each PC on the go-alive chain:
    /// 0x3de9 (task create), 0x55f8 (go-alive run-fn entry), 0x56b1 (call to the
    /// publisher 0x50e8), 0x50e8 (publisher body -- writes the struct + magic),
    /// 0x56e6 (waiti 0 idle). "NEVER" for 0x55f8 => the go-alive task is created
    /// but never dispatched, so the remaining gate is whatever readies IT (not
    /// 0x10f10). Env: XDNA_FW_MAX (default 3_000_000), XDNA_FW_COL (default 0).
    /// Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_goalive_lifecycle() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the go-alive lifecycle probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        proc.bus.attach_device(crate::device::DeviceState::new_npu1());
        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3_000_000);
        let col: u32 = std::env::var("XDNA_FW_COL")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0);

        // XDNA_FW_NOBREACH=1 runs natural boot (no flag+column breach), to compare
        // the go-alive record's fate with vs without the artificial completion.
        let nobreach = std::env::var("XDNA_FW_NOBREACH").is_ok();
        const DONE_CHECK_PC: u32 = 0xd828;
        let whitelist: [u32; 2] = [0x10f10, 0x9040];
        // The go-alive chain, in causal order.
        let watch: [(u32, &str); 5] = [
            (0x3de9, "task-create(run-fn=0x55f8)"),
            (0x55f8, "go-alive run-fn entry"),
            (0x56b1, "call publisher 0x50e8"),
            (0x50e8, "publisher body"),
            (0x56e6, "waiti 0 (IDLE)"),
        ];
        let mut first_hit: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        let mut hit_count: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        let mut completed: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        let mut n = 0u64;
        let mut stop = String::from("budget reached");
        while n < max {
            let pc = proc.cpu.pc;
            let pcm = pc & 0x00ff_ffff;
            if watch.iter().any(|(w, _)| *w == pcm) {
                first_hit.entry(pcm).or_insert(n);
                *hit_count.entry(pcm).or_insert(0) += 1;
            }
            if !nobreach && pc == DONE_CHECK_PC {
                let task = proc.cpu.regs.read_ar(4);
                if whitelist.contains(&task)
                    && proc.cpu.data_read32(&mut proc.bus, task.wrapping_add(0x30)).unwrap_or(0) == 0
                    && completed.insert(task)
                {
                    let _ = proc.cpu.data_write32(&mut proc.bus, task.wrapping_add(0x30), 1);
                    let _ = proc.cpu.data_write32(&mut proc.bus, task.wrapping_add(8), col);
                    eprintln!("  n={n}: completed task {task:#x} (flag+col={col})");
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?}) at pc={pc:#x} (IDLE!)");
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
        eprintln!("=== go-alive lifecycle (col={col}) ===");
        eprintln!("instrs = {n}; stop = {stop}; completed = {:x?}", completed);
        for (w, label) in &watch {
            match first_hit.get(w) {
                Some(fh) => eprintln!("  {w:#08x} {label:<28} first-hit n={fh}  (x{})", hit_count[w]),
                None => eprintln!("  {w:#08x} {label:<28} NEVER"),
            }
        }
        // Post-breach dump of the scheduler structures the globals point at:
        // state table 0xf308, ready-mask 0x11098, SCHED 0x2250, SCHED2 0x1186c.
        // Override via XDNA_FW_DUMP=base:len,base:len (hex, len in bytes).
        let rd = |p: &mut FirmwareProcessor, a: u32| p.cpu.data_read32(&mut p.bus, a).unwrap_or(0);
        let dumps: Vec<(u32, u32)> = std::env::var("XDNA_FW_DUMP")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|r| {
                        let (a, b) = r.split_once(':')?;
                        Some((
                            u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?,
                            u32::from_str_radix(b.trim().trim_start_matches("0x"), 16).ok()?,
                        ))
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![(0xf308, 0x80), (0x11098, 0x40), (0x2250, 0x40), (0x1186c, 0x40)]);
        for (base, len) in dumps {
            eprintln!("--- dump {base:#x}..{:#x} ---", base + len);
            let mut a = base;
            while a < base + len {
                let w: [u32; 4] = std::array::from_fn(|k| rd(&mut proc, a + (k as u32) * 4));
                eprintln!("  {a:#08x}: {:#010x} {:#010x} {:#010x} {:#010x}", w[0], w[1], w[2], w[3]);
                a += 16;
            }
        }
        // Optional post-boot word scan (XDNA_FW_FIND=hex) over low RAM, to locate
        // a persistent TCB by a known field value (e.g. the go-alive run-fn 0x55f8).
        if let Some(word) = std::env::var("XDNA_FW_FIND")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
        {
            eprintln!("--- scan 0x1000..0x30000 for {word:#x} ---");
            let mut a = 0x1000u32;
            while a < 0x30000 {
                if rd(&mut proc, a) == word {
                    eprintln!("  found at {a:#x}");
                }
                a += 4;
            }
        }
    }

    /// DECISIVE idle-vs-busy-wall test (2026-07-08, Maya): unlike
    /// `m2c_probe_mailbox_wake` (whose warmup wedges at the 58k wall and so tests
    /// the doorbell from the WRONG pre-Wall-C state), this applies the flag+column
    /// breach at the dispatcher done-check (0xd828) so boot reaches the *real*
    /// post-Wall-C scheduler loop, THEN injects the host mailbox doorbell and
    /// watches whether the fw LEAVES the scheduler ready-loop. If it enters new
    /// code (mailbox ISR / array-init HAL 0x35000..0x36000), the loop was
    /// idle-waiting-for-host (= effectively booted to idle). If nothing moves --
    /// especially if INTLEVEL stays pinned >=2 and the IRQ is masked -- it is a
    /// real busy sub-wall, not idle. Env: XDNA_FW_BD_WARMUP (default 300_000,
    /// clears the breach at ~48k and settles), XDNA_FW_BD_STEPS (default 400_000),
    /// XDNA_FW_COL (col to assign, default 0). Ignored unless XDNA_FW_PROBE set.
    #[test]
    fn m2c_probe_breach_doorbell() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the breach+doorbell idle test");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        proc.bus.attach_device(crate::device::DeviceState::new_npu1());
        proc.enable_host_mailbox();
        let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        let warmup = env_u64("XDNA_FW_BD_WARMUP", 300_000);
        let steps = env_u64("XDNA_FW_BD_STEPS", 400_000);
        let col: u32 = std::env::var("XDNA_FW_COL")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0);

        const DONE_CHECK_PC: u32 = 0xd828;
        let whitelist: [u32; 2] = [0x10f10, 0x9040];
        let mut completed: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        // Breach-warmup into the post-Wall-C scheduler loop, recording the last-20k
        // "home" symbol set so post-doorbell NEW territory stands out.
        let mut spin_syms: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for i in 0..warmup {
            let pc = proc.cpu.pc;
            if pc == DONE_CHECK_PC {
                let task = proc.cpu.regs.read_ar(4);
                if whitelist.contains(&task)
                    && proc.cpu.data_read32(&mut proc.bus, task.wrapping_add(0x30)).unwrap_or(0) == 0
                    && completed.insert(task)
                {
                    let _ = proc.cpu.data_write32(&mut proc.bus, task.wrapping_add(0x30), 1);
                    let _ = proc.cpu.data_write32(&mut proc.bus, task.wrapping_add(8), col);
                }
            }
            if i + 20_000 >= warmup {
                spin_syms.insert(nearest_symbol(&proc.symbols, pc & 0x00ff_ffff));
            }
            if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
                break;
            }
        }
        // Optional DMA-HAL bit3 completion shim (the seam FUN_00008c68 polls on the
        // 0x2727_col000 handshake). XDNA_FW_BD_SHIM present => inject before the
        // doorbell; XDNA_FW_BD_SHIM=cont => reassert every post-step.
        let shim_mode = std::env::var("XDNA_FW_BD_SHIM").ok();
        let shim_cont = shim_mode.as_deref() == Some("cont");
        let shim_on = shim_mode.is_some();
        let shim_cols: u32 = std::env::var("XDNA_FW_BD_SHIM_COLS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        let inject_bit3 = |proc: &mut FirmwareProcessor| {
            for c in 0..shim_cols {
                let byte_addr = 0xf9e0 + c * 0x60;
                let cur = proc.cpu.data_read32(&mut proc.bus, byte_addr & !3).unwrap_or(0);
                let byte = ((cur >> ((byte_addr & 3) * 8)) & 0xff) | 0x08;
                let _ = proc.cpu.data_write8(&mut proc.bus, byte_addr, byte);
                let ext = 0x2727_1000 + c * 0x1000;
                let _ = proc.cpu.data_write32(&mut proc.bus, ext, 0x3);
            }
        };

        let il_pre = proc.cpu.regs.intlevel();
        eprintln!(
            "=== breach+doorbell: idle-vs-busy-wall test (col={col}, shim={shim_on}/cont={shim_cont}) ==="
        );
        eprintln!("completed tasks in warmup: {:x?}", completed);
        eprintln!(
            "pre-doorbell: pc={:#x} [{}], INTLEVEL={il_pre}, INTENABLE={:#010x}, home syms={}",
            proc.cpu.pc,
            nearest_symbol(&proc.symbols, proc.cpu.pc & 0x00ff_ffff),
            proc.cpu.intenable,
            spin_syms.len(),
        );

        if shim_on {
            inject_bit3(&mut proc);
        }
        // Inject the mailbox doorbell exactly as the host would: enable + pend bit0.
        // We do NOT force INTLEVEL down -- if the IRQ stays masked, that is itself
        // the answer (post-breach state is not receive-ready).
        proc.cpu.intenable |= 1;
        proc.cpu.interrupt |= 1;
        let irq_pending_before = proc.cpu.interrupt & 1;

        let mut new_syms: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
        let mut hal_entry_at: Option<(u64, u32)> = None;
        let mut irq_taken_at: Option<(u64, u32)> = None;
        let mut max_il = il_pre;
        let mut min_il = il_pre;
        let mut il_drop_at: Option<(u64, u32, u32)> = None; // (n, pc, new level < il_pre)
        let mut stop = String::from("steps budget reached");
        let mut n = 0u64;
        while n < steps {
            if shim_cont {
                inject_bit3(&mut proc);
            }
            let pc = proc.cpu.pc & 0x00ff_ffff;
            let sym = nearest_symbol(&proc.symbols, pc);
            if !spin_syms.contains(&sym) {
                *new_syms.entry(sym).or_insert(0) += 1;
            }
            if hal_entry_at.is_none() && (0x35000..0x36000).contains(&pc) {
                hal_entry_at = Some((n, pc));
            }
            // The IRQ is "taken" when the pending bit clears (dispatched by the ISR).
            if irq_taken_at.is_none() && irq_pending_before != 0 && proc.cpu.interrupt & 1 == 0 {
                irq_taken_at = Some((n, pc));
            }
            let il = proc.cpu.regs.intlevel();
            if il > max_il {
                max_il = il;
            }
            if il < min_il {
                min_il = il;
                if il < il_pre && il_drop_at.is_none() {
                    il_drop_at = Some((n, pc, il));
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?}) at pc={:#x} (idle/waiti)", proc.cpu.pc);
                    break;
                }
                Step::Unknown { pc, word } => {
                    stop = format!("Unknown at pc={pc:#x} word={word:#010x}");
                    break;
                }
            }
        }

        eprintln!("post-doorbell: ran {n} steps, stop={stop}");
        eprintln!(
            "last_pc={:#x} [{}], INTLEVEL post-doorbell: min={min_il} max={max_il} (pre={il_pre})",
            proc.cpu.pc,
            nearest_symbol(&proc.symbols, proc.cpu.pc & 0x00ff_ffff),
        );
        match il_drop_at {
            Some((dn, dpc, dil)) => eprintln!(
                "INTLEVEL DROPPED below pre({il_pre}) -> {dil} at n={dn} pc={dpc:#x}  (receive-ready transition!)"
            ),
            None => eprintln!("INTLEVEL never dropped below pre({il_pre}) -- still not receive-ready"),
        }
        eprintln!("IRQ taken (pending bit0 cleared) at: {irq_taken_at:x?}");
        eprintln!("array-init HAL (0x35000..0x36000) entered: {hal_entry_at:x?}");
        eprintln!("--- NEW symbols visited after doorbell (not in the scheduler home set) ---");
        if new_syms.is_empty() {
            eprintln!("  (none -- doorbell did NOT move the fw; busy sub-wall, not idle)");
        } else {
            let mut ranked: Vec<_> = new_syms.into_iter().collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1));
            for (sym, hits) in ranked.iter().take(30) {
                eprintln!("  {hits:>8}  {sym}");
            }
        }
    }

    /// REGROUP EXPERIMENT (2026-07-07, Maya): stub the SMU/PSP so the boot
    /// worker's column-power request is answered "ready immediately" (force the
    /// done-flag at every dispatcher check) AND attach the real emulated array
    /// (M1), then see how far the firmware gets. The prior force-done faulted at
    /// ~623k because the array was a 0-stub ("columns never produced results");
    /// with M1 the columns respond, so this tests "stub the SMU, emulate the
    /// NPU": can the firmware reach its runtime job loop / idle if we tell it boot
    /// succeeded and hand it a real array? Env: XDNA_FW_MAX (budget, default
    /// 2_000_000), XDNA_FW_NOARRAY (skip attach, for A/B). XDNA_FW_PROBE-gated.
    #[test]
    fn m2c_probe_stub_smu_boot() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the stub-SMU boot experiment");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        let attach = std::env::var("XDNA_FW_NOARRAY").is_err();
        if attach {
            proc.bus.attach_device(crate::device::DeviceState::new_npu1());
        }
        proc.bus.arm_probe();

        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2_000_000);
        const DONE_CHECK_PC: u32 = 0xd828; // dispatcher `l32i.n a10,[a4+0x30]`
        const KEEP: usize = 44;
        let mut n = 0u64;
        let mut forces = 0u64;
        let mut stop = String::from("budget reached");
        let mut ring: std::collections::VecDeque<(u64, u32, String)> =
            std::collections::VecDeque::with_capacity(KEEP + 1);
        while n < max {
            let pc = proc.cpu.pc;
            // Stub SMU "columns ready": whenever the boot worker checks its
            // done-flag, report done (the instant power-up ack).
            if pc == DONE_CHECK_PC {
                let done = proc.cpu.regs.read_ar(4).wrapping_add(0x30);
                if proc.cpu.data_read32(&mut proc.bus, done).unwrap_or(0) == 0 {
                    let _ = proc.cpu.data_write32(&mut proc.bus, done, 1);
                    forces += 1;
                }
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
                    stop = format!("Wait({reason:?}) at pc={pc:#x} (IDLE!)");
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
        let log = proc.bus.take_probe();
        let array: Vec<_> = log.iter().filter(|a| a.region == Region::Array).collect();
        eprintln!("=== stub-SMU boot (array attached: {attach}) ===");
        eprintln!("forced done-flag {forces}x");
        eprintln!("instrs  = {n}");
        eprintln!("stop    = {stop}");
        eprintln!("last_pc = {:#x}  {}", proc.cpu.pc, nearest_symbol(&proc.symbols, proc.cpu.pc));
        eprintln!(
            "array accesses = {} (rd={} wr={})",
            array.len(),
            array.iter().filter(|a| !a.is_write).count(),
            array.iter().filter(|a| a.is_write).count()
        );
        for a in array.iter().take(20) {
            eprintln!(
                "  pc={:#x} {} {:#x}={:#x}",
                a.pc,
                if a.is_write { "wr" } else { "rd" },
                a.addr,
                a.value
            );
        }
        eprintln!("--- last {} instrs before stop ---", ring.len());
        for (i, pc, disasm) in &ring {
            eprintln!("{i:>8} pc={pc:#08x} {:<24} {disasm}", nearest_symbol(&proc.symbols, *pc));
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
                let _ = proc.cpu.data_write32(&mut proc.bus, done_addr, 1);
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

    /// M2c #140 CYCLE-ACCOUNTING: bucket every executed instruction by the
    /// routine (nearest symbol range) its PC falls in, so every executed cycle
    /// lands in exactly one named bucket. This is the "where do the cycles go"
    /// instrument for decomposing the fitted 8000-cycle `dma_wait` mailbox
    /// constant (`src/npu/executor.rs` `DEFAULT_MAILBOX_CYCLES`) into a real
    /// mgmt-firmware instruction budget.
    ///
    /// Grounding: on NPU1 (single Xtensa mgmt processor, NO per-column uC -- the
    /// per-column TCT machinery is VE2/Versal), the mgmt firmware is the only
    /// processor that consumes a shim-DMA TCT and unblocks a `dma_wait`. So its
    /// wait -> notice -> wake -> dispatch instruction cost IS the firmware share
    /// of the 8000. The array->completion propagation is the remaining share and
    /// is modeled by the emulator's DMA/stream engine, not here.
    ///
    /// Modes (env):
    ///   (default)              run to the boot wall; report the full per-routine
    ///                          histogram, and the dispatcher SPIN PERIOD (instrs
    ///                          per completion re-check) = the firmware's
    ///                          completion-poll granularity at the scheduler.
    ///   XDNA_FW_FORCE_DONE=1   force `[task+0x30]=1` at the dispatcher done-flag
    ///                          check (0xd828) so the completion "fires"; report
    ///                          where the downstream wake/dispatch/resume cycles
    ///                          go by routine.
    /// XDNA_FW_MAX=<n> overrides the instruction budget (default 120000).
    /// Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_cycle_accounting() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the cycle-accounting probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(120_000);
        let force = std::env::var("XDNA_FW_FORCE_DONE").is_ok();
        const DONE_CHECK_PC: u32 = 0xd828; // dispatcher `l32i.n a10,[a4+0x30]`

        // Per-routine instruction counts: bucket entry addr -> (label, count).
        let mut hist: std::collections::HashMap<u32, (String, u64)> = std::collections::HashMap::new();
        // Ring of recent (n, pc) for steady-state loop-period detection.
        const TAIL: usize = 4096;
        let mut tail: std::collections::VecDeque<(u64, u32)> =
            std::collections::VecDeque::with_capacity(TAIL + 1);
        let mut n = 0u64;
        let mut forces = 0u64;
        let mut stop = String::from("budget reached");

        // Full-run completion-check periods: for each anchor, the median
        // instruction gap between successive visits across the WHOLE run (the
        // 4096-instr tail window is too short for the coarse event-dispatcher
        // period). Anchors: done-flag check, event dispatcher, dispatcher entry,
        // pending-event delivery.
        const ANCHORS: [u32; 4] = [0xd828, 0x5580, 0xd7f0, 0xcadc];
        let mut anchor_last: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
        let mut anchor_gaps: std::collections::HashMap<u32, Vec<u64>> = std::collections::HashMap::new();

        while n < max {
            let pc = proc.cpu.pc;
            if ANCHORS.contains(&pc) {
                if let Some(&last) = anchor_last.get(&pc) {
                    anchor_gaps.entry(pc).or_default().push(n - last);
                }
                anchor_last.insert(pc, n);
            }
            if force && pc == DONE_CHECK_PC {
                let done_addr = proc.cpu.regs.read_ar(4).wrapping_add(0x30);
                let _ = proc.cpu.data_write32(&mut proc.bus, done_addr, 1);
                forces += 1;
            }
            let (bucket, label) = routine_bucket(&proc.symbols, pc);
            hist.entry(bucket).or_insert((label, 0)).1 += 1;
            if tail.len() == TAIL {
                tail.pop_front();
            }
            tail.push_back((n, pc));
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

        eprintln!("=== M2c cycle-accounting (#140 dma_wait 8000cy decomposition) ===");
        eprintln!(
            "mode            = {}",
            if force {
                "FORCE_DONE continuation"
            } else {
                "wall spin"
            }
        );
        eprintln!("instrs executed = {n}");
        if force {
            eprintln!("forced done-flag= {forces} time(s)");
        }
        eprintln!("stop reason     = {stop}");
        eprintln!("last pc         = {:#x}  {}", proc.cpu.pc, nearest_symbol(&proc.symbols, proc.cpu.pc));

        // Per-routine histogram, sorted by instruction count descending. Every
        // executed cycle is in exactly one row -> the rows sum to `n`.
        let mut rows: Vec<(u32, String, u64)> = hist.into_iter().map(|(a, (l, c))| (a, l, c)).collect();
        rows.sort_by(|a, b| b.2.cmp(&a.2));
        eprintln!("--- per-routine instruction budget (every cycle accounted) ---");
        let show = 24.min(rows.len());
        let mut shown = 0u64;
        for (addr, label, count) in rows.iter().take(show) {
            let pct = 100.0 * *count as f64 / n.max(1) as f64;
            eprintln!("  {count:>9} ({pct:>5.1}%)  {addr:#08x} {label}");
            shown += *count;
        }
        if rows.len() > show {
            let rest = n - shown;
            eprintln!(
                "  {rest:>9} ({:>5.1}%)  (+{} more routines)",
                100.0 * rest as f64 / n.max(1) as f64,
                rows.len() - show
            );
        }

        // Full-run completion-check periods (median gap between anchor visits).
        eprintln!("--- completion-check periods (full run, median instrs between visits) ---");
        for anchor in ANCHORS {
            match anchor_gaps.get(&anchor) {
                Some(gaps) if gaps.len() >= 2 => {
                    let mut g = gaps.clone();
                    g.sort_unstable();
                    let med = g[g.len() / 2];
                    eprintln!(
                        "  {anchor:#08x} {:<24} period ~ {med:>6} instrs  ({} visits)",
                        nearest_symbol(&proc.symbols, anchor),
                        gaps.len() + 1
                    );
                }
                _ => eprintln!(
                    "  {anchor:#08x} {:<24} <2 visits (essentially never reached)",
                    nearest_symbol(&proc.symbols, anchor)
                ),
            }
        }

        // Completion-check granularity: the median instruction gap between
        // successive visits to each anchor over the tail window. The relevant
        // quantum is NOT the scheduler's inner bit-scan loop but the period
        // between successive COMPLETION CHECKS -- the dispatcher's done-flag read
        // (0xd828) and the event dispatcher entry (0x5580). That bounds how fast
        // the firmware can notice a `dma_wait` TCT once the array signals it.
        if tail.len() > 16 {
            let period_of = |anchor: u32| -> Option<(u64, u64)> {
                let visits: Vec<u64> = tail.iter().filter(|(_, pc)| *pc == anchor).map(|(n, _)| *n).collect();
                if visits.len() < 3 {
                    return None;
                }
                let mut gaps: Vec<u64> = visits.windows(2).map(|w| w[1] - w[0]).collect();
                gaps.sort_unstable();
                Some((gaps[gaps.len() / 2], visits.len() as u64))
            };
            let mut freq: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
            for (_, pc) in &tail {
                *freq.entry(*pc).or_insert(0) += 1;
            }
            let mode_pc = freq.iter().max_by_key(|(_, c)| **c).map(|(&pc, _)| pc).unwrap_or(0);
            let distinct: std::collections::BTreeSet<u32> = tail.iter().map(|(_, pc)| *pc).collect();
            eprintln!(
                "--- completion-check granularity (tail {} instrs, {} distinct PCs) ---",
                tail.len(),
                distinct.len()
            );
            // Named completion-check anchors + the hottest inner PC, plus any
            // caller-supplied anchor via XDNA_FW_ANCHOR_PC=<hex>.
            let extra = std::env::var("XDNA_FW_ANCHOR_PC")
                .ok()
                .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok());
            let anchors: Vec<(u32, &str)> = vec![
                (0xd828, "dispatcher done-flag check l32i.n [task+0x30]"),
                (0x5580, "event dispatcher entry (-> wake_tasks_by_event_mask)"),
                (mode_pc, "hottest inner PC (scheduler bit-scan)"),
                (extra.unwrap_or(0), "XDNA_FW_ANCHOR_PC"),
            ];
            for (anchor, what) in anchors {
                if anchor == 0 {
                    continue;
                }
                match period_of(anchor) {
                    Some((period, visits)) => eprintln!(
                        "  {anchor:#08x} {:<20} period = {period:>6} instrs  ({visits} visits)  {what}",
                        nearest_symbol(&proc.symbols, anchor)
                    ),
                    None => eprintln!(
                        "  {anchor:#08x} {:<20} NOT in tail window (<3 visits)  {what}",
                        nearest_symbol(&proc.symbols, anchor)
                    ),
                }
            }
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
                proc.bus.data_store32(HEAD, POSTED);
                proc.bus.data_store32(INTR, 0);
            }
            "tail0" => proc.bus.data_store32(TAIL, 0),
            "tailadv" => proc.bus.data_store32(TAIL, POSTED + 8),
            "doorbell" => proc.cpu.interrupt |= 1,
            "headdb" => {
                proc.bus.data_store32(HEAD, POSTED);
                proc.bus.data_store32(INTR, 0);
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
            if posted_at.is_none() && proc.bus.data_load32(TAIL) == POSTED {
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
                    if proc.cpu.data_read32(&mut proc.bus, f).unwrap_or(0) != 0
                        && !done_set.iter().any(|(a, _)| *a == f)
                    {
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
            eprintln!("  [{f:#x}] = {:#x}", proc.cpu.data_read32(&mut proc.bus, f).unwrap_or(0));
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

    /// M2c iter18 RE TOOL (armed-deferred design): the current-task timeline.
    /// The one mailbox post (~6972) fires before the scheduler is up, so the
    /// faithful completion must ARM on the post and DELIVER once a valid current
    /// task appears. This probe answers: when does the scheduler global
    /// `[0x2278]` (current-task ptr) first become non-zero, what values does it
    /// take, and is `0x9040` (the task the dispatcher blocks on) the FIRST valid
    /// task -- or do others run first (which would mis-target a deliver-on-first
    /// -valid rule)? Also records the post timing and the first done-flag check
    /// (`pc==0xd828`, task ptr in a4). Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_current_task_timeline() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the current-task timeline probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        const MAX: u64 = 1_500_000;
        const SCHED: u32 = 0x2278; // scheduler global -> current-task ptr
        const TAIL: u32 = 0x2720_0170;
        const CHECK_PC: u32 = 0xd828; // dispatcher done-flag check l32i.n a10,[a4+0x30]
                                      // Distinct current-task values in first-seen order (n, value).
        let mut task_changes: Vec<(u64, u32)> = Vec::new();
        let mut last_task = 0u32;
        let mut post_at: Option<u64> = None;
        let mut first_check: Option<(u64, u32, u32)> = None; // (n, a4, [a4+0x30])
        let mut n = 0u64;
        let mut stop = String::from("budget");
        while n < MAX {
            let pc = proc.cpu.pc;
            // Sample the current-task global each step (cheap local read).
            let cur = proc.cpu.data_read32(&mut proc.bus, SCHED).unwrap_or(0);
            if cur != last_task {
                task_changes.push((n, cur));
                last_task = cur;
            }
            // Post detection (tail advance to non-zero).
            if post_at.is_none() && proc.bus.data_load32(TAIL) != 0 {
                post_at = Some(n);
            }
            // First done-flag check: record the task ptr (a4) and its flag.
            if first_check.is_none() && pc == CHECK_PC {
                let a4 = proc.cpu.regs.read_ar(4);
                let flag = proc.cpu.data_read32(&mut proc.bus, a4.wrapping_add(0x30)).unwrap_or(0);
                first_check = Some((n, a4, flag));
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(r) => {
                    n += 1;
                    stop = format!("Wait({r:?})");
                    break;
                }
                Step::Unknown { pc: upc, word } => {
                    stop = format!("Unknown at {upc:#x} word={word:#010x}");
                    break;
                }
            }
        }
        eprintln!("=== M2c current-task timeline ===");
        eprintln!("instrs = {n}; stop = {stop}");
        eprintln!("post (tail!=0) at instr = {post_at:?}");
        eprintln!("first done-flag check (pc=0xd828): {first_check:x?}  (n, a4=task, [a4+0x30])");
        eprintln!("--- current-task [0x2278] change timeline (n, value) ---");
        for (n, v) in &task_changes {
            eprintln!("  n={n:>8}  [0x2278] <- {v:#x}");
        }
        eprintln!("(total {} distinct current-task transitions)", task_changes.len());
    }

    /// M2c iter18 RE TOOL (per-task completion RE): map each blocked task to the
    /// run-function the dispatcher calls for it, and capture what that run-fn
    /// TOUCHES (its load/store effective addresses) -- the poll/request sites that
    /// reveal what each task is waiting on. The dispatcher (`0xd7f0..0xd848`) does
    /// `callx8 a3` (a3 = the current task's run-fn) when the done-flag is 0. This
    /// records every distinct `(current_task, call_target, call_pc)` for calls
    /// made from inside the dispatcher, plus, for the first execution of each
    /// distinct run-fn, the set of distinct data effective-addresses it accesses
    /// (region-tagged) over a bounded instruction window. Ignored unless
    /// XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_task_runfns() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the task-runfn probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        const MAX: u64 = 1_500_000;
        const SCHED: u32 = 0x2278;
        const DISP_LO: u32 = 0xd7f0;
        const DISP_HI: u32 = 0xd848;
        let region_name = |a: u32| -> &'static str {
            match a {
                _ if a < 0x0400_0000 => "LOCAL",
                _ if a < 0x0800_0000 => "ARRAY",
                _ if (0x0800_0000..0x08b0_0000).contains(&a) => "GAP",
                _ if (0x08b0_0000..0x2700_0000).contains(&a) => "RAM",
                _ if (0x2700_0000..0x2800_0000).contains(&a) => "MAILBOX",
                _ => "SYSTEM",
            }
        };
        // (task, run_fn target, call_pc) triples observed at dispatcher calls.
        let mut pairs: std::collections::BTreeSet<(u32, u32, u32)> = std::collections::BTreeSet::new();
        // For each distinct run_fn target, the distinct data EAs it accessed
        // (only the run-fn currently being traced, bounded).
        let mut runfn_eas: std::collections::BTreeMap<u32, std::collections::BTreeSet<u32>> =
            std::collections::BTreeMap::new();
        // Trace window: when we enter a run_fn we haven't fully profiled, record
        // its data EAs for up to TRACE_WIN instrs (following calls out of it too).
        const TRACE_WIN: u32 = 4000;
        let mut tracing: Option<(u32, u32)> = None; // (run_fn, instrs_left)

        let mut n = 0u64;
        let mut stop = String::from("budget");
        while n < MAX {
            let pc = proc.cpu.pc;
            let cur = proc.cpu.data_read32(&mut proc.bus, SCHED).unwrap_or(0);
            let decoded = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                Ok(phys) => {
                    let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                    Some(decode::decode(&b, pc))
                }
                Err(_) => None,
            };
            if let Some(d) = &decoded {
                // Dispatcher-made call -> the run-fn (or scheduler helper) target.
                if (DISP_LO..DISP_HI).contains(&pc) {
                    let target = match d.op {
                        Op::Call4 { target } | Op::Call8 { target } | Op::Call12 { target } => Some(target),
                        Op::Callx4 { s } | Op::Callx8 { s } | Op::Callx12 { s } => {
                            Some(proc.cpu.regs.read_ar(s))
                        }
                        _ => None,
                    };
                    if let Some(t) = target {
                        pairs.insert((cur, t, pc));
                        // Begin tracing this run-fn's data accesses if new.
                        if !runfn_eas.contains_key(&t) {
                            runfn_eas.insert(t, std::collections::BTreeSet::new());
                            tracing = Some((t, TRACE_WIN));
                        }
                    }
                }
                // While tracing, record data EAs of loads/stores.
                if let Some((rf, left)) = tracing {
                    let ea = match d.op {
                        Op::L32i { s, imm, .. }
                        | Op::L32iN { s, imm, .. }
                        | Op::L8ui { s, imm, .. }
                        | Op::L16ui { s, imm, .. }
                        | Op::L16si { s, imm, .. }
                        | Op::S32i { s, imm, .. }
                        | Op::S32iN { s, imm, .. }
                        | Op::S8i { s, imm, .. }
                        | Op::S16i { s, imm, .. }
                        | Op::S32ri { s, imm, .. } => Some(proc.cpu.regs.read_ar(s).wrapping_add(imm)),
                        _ => None,
                    };
                    if let Some(ea) = ea {
                        runfn_eas.get_mut(&rf).unwrap().insert(ea);
                    }
                    tracing = if left <= 1 { None } else { Some((rf, left - 1)) };
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(r) => {
                    n += 1;
                    stop = format!("Wait({r:?})");
                    break;
                }
                Step::Unknown { pc: upc, word } => {
                    stop = format!("Unknown at {upc:#x} word={word:#010x}");
                    break;
                }
            }
        }
        eprintln!("=== M2c task run-fn map ===");
        eprintln!("instrs = {n}; stop = {stop}");
        eprintln!("--- (task, run_fn, call_pc) at dispatcher calls ---");
        for (task, rf, cpc) in &pairs {
            eprintln!("  task={task:#x}  run_fn={rf:#x}  call_pc={cpc:#x}");
        }
        eprintln!(
            "--- distinct data EAs each run_fn touched (first {TRACE_WIN} instrs of its first run) ---"
        );
        for (rf, eas) in &runfn_eas {
            let tagged: Vec<String> = eas.iter().map(|a| format!("{a:#x}({})", region_name(*a))).collect();
            eprintln!("  run_fn={rf:#x}: {}", tagged.join(" "));
        }
    }

    /// M2c iter18 RE TOOL (completion-binding search): does the firmware ever
    /// STORE a task done-flag address (0x9070 / 0x10f40) -- or task base
    /// (0x9040 / 0x10f10) -- as a VALUE? If the fw registers "land my completion
    /// at 0xNNNN" with the hardware/host, that address appears as a stored value
    /// in a request/descriptor. Watches every 32-bit store whose stored value is
    /// one of the tracked pointers and reports (pc, EA, value, region) -- the EA's
    /// region tells whether it went into a mailbox descriptor, a DMA config, or a
    /// local table. Absence means the completion target is IMPLICIT (the agent
    /// knows the task layout), not fw-registered. Ignored unless XDNA_FW_PROBE set.
    #[test]
    fn m2c_probe_completion_binding() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the completion-binding search");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        const MAX: u64 = 1_500_000;
        // Tracked pointers: task bases, done-flags, and the +0x1b state byte area.
        let tracked: [u32; 4] = [0x9040, 0x9070, 0x10f10, 0x10f40];
        let region_name = |a: u32| -> &'static str {
            match a {
                _ if a < 0x0400_0000 => "LOCAL",
                _ if a < 0x0800_0000 => "ARRAY",
                _ if (0x0800_0000..0x08b0_0000).contains(&a) => "GAP",
                _ if (0x08b0_0000..0x2700_0000).contains(&a) => "RAM",
                _ if (0x2700_0000..0x2800_0000).contains(&a) => "MAILBOX",
                _ => "SYSTEM",
            }
        };
        // (n, pc, ea, value) for each store of a tracked value; dedup by (ea,value).
        let mut hits: std::collections::BTreeMap<(u32, u32), (u64, u32)> = std::collections::BTreeMap::new();
        let mut n = 0u64;
        let mut stop = String::from("budget");
        while n < MAX {
            let pc = proc.cpu.pc;
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                let store = match decode::decode(&b, pc).op {
                    Op::S32i { t, s, imm } | Op::S32iN { t, s, imm } | Op::S32ri { t, s, imm } => {
                        Some((t, s, imm))
                    }
                    _ => None,
                };
                if let Some((t, s, imm)) = store {
                    let val = proc.cpu.regs.read_ar(t);
                    if tracked.contains(&val) {
                        let ea = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                        hits.entry((ea, val)).or_insert((n, pc));
                    }
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(r) => {
                    n += 1;
                    stop = format!("Wait({r:?})");
                    break;
                }
                Step::Unknown { pc: upc, word } => {
                    stop = format!("Unknown at {upc:#x} word={word:#010x}");
                    break;
                }
            }
        }
        eprintln!("=== M2c completion-binding search ===");
        eprintln!("instrs = {n}; stop = {stop}");
        eprintln!("--- stores of a tracked pointer value (dedup by ea,value) ---");
        for ((ea, val), (n, pc)) in &hits {
            eprintln!("  n={n:>8} pc={pc:#x}  [{ea:#x} ({})] <- {val:#x}", region_name(*ea));
        }
        eprintln!("(total {} distinct (ea,value) bindings)", hits.len());
    }

    /// M2c iter18 RE TOOL (x2i experiment prep): locate the `mgmt_mbox_chann_info`
    /// struct the fw publishes at boot, by catching the store of its magic
    /// `0x55504e5f` ("_NPU", struct offset 0x20). Dumps the 64-byte struct
    /// (x2i/i2x tail/head/buf/buf_sz device addresses + magic + msi_id +
    /// prot_major/minor) from `magic_ea - 0x20`. Also records where the fw wrote
    /// the struct pointer (the FW_ALIVE_OFF slot: a store whose VALUE equals the
    /// struct base). This gives the x2i register/ring addresses needed to deliver
    /// a host->fw message. Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_alive_struct() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the alive-struct locate");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        const MAX: u64 = 1_500_000;
        const MAGIC: u32 = 0x5550_4e5f; // "_NPU"
        let mut magic_ea: Option<(u64, u32, u32)> = None; // (n, pc, ea)
        let mut ptr_stores: Vec<(u64, u32, u32, u32)> = Vec::new(); // (n, pc, ea, val) where val==struct base
        let mut n = 0u64;
        // First pass: find the magic store.
        while n < MAX {
            let pc = proc.cpu.pc;
            if magic_ea.is_none() {
                if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                    let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                    if let Op::S32i { t, s, imm } | Op::S32iN { t, s, imm } | Op::S32ri { t, s, imm } =
                        decode::decode(&b, pc).op
                    {
                        if proc.cpu.regs.read_ar(t) == MAGIC {
                            let ea = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                            magic_ea = Some((n, pc, ea));
                        }
                    }
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(_) | Step::Unknown { .. } => break,
            }
            if magic_ea.is_some() && n > magic_ea.unwrap().0 + 50_000 {
                break; // struct + pointer publish happen close together
            }
        }
        eprintln!("=== M2c alive-struct locate ===");
        match magic_ea {
            None => eprintln!("magic 0x55504e5f store NOT seen in {n} instrs"),
            Some((mn, mpc, mea)) => {
                let base = mea.wrapping_sub(0x20);
                eprintln!("magic store at n={mn} pc={mpc:#x} -> magic@{mea:#x}, struct base {base:#x}");
                let fields = [
                    "x2i_tail",
                    "x2i_head",
                    "x2i_buf",
                    "x2i_buf_sz",
                    "i2x_tail",
                    "i2x_head",
                    "i2x_buf",
                    "i2x_buf_sz",
                    "magic",
                    "msi_id",
                    "prot_major",
                    "prot_minor",
                ];
                for (i, name) in fields.iter().enumerate() {
                    let a = base.wrapping_add((i * 4) as u32);
                    eprintln!(
                        "  +{:#04x} {name:<11} = {:#x}",
                        i * 4,
                        proc.cpu.data_read32(&mut proc.bus, a).unwrap_or(0)
                    );
                }
                // Re-scan for a store whose VALUE == struct base (the FW_ALIVE_OFF publish).
                let mut proc2 = {
                    let raw = std::fs::read(&path).expect("read");
                    FirmwareProcessor::load_m2c(FirmwareImage::parse(&raw).expect("parse"))
                };
                let mut m = 0u64;
                while m < mn + 50_000 {
                    let pc = proc2.cpu.pc;
                    if let Ok(phys) = proc2.cpu.translate(&mut proc2.bus, pc, xtensa::interp::Access::Fetch) {
                        let b: [u8; 8] =
                            std::array::from_fn(|k| proc2.bus.fetch8(pc + k as u32, phys + k as u32));
                        if let Op::S32i { t, s, imm } | Op::S32iN { t, s, imm } | Op::S32ri { t, s, imm } =
                            decode::decode(&b, pc).op
                        {
                            let v = proc2.cpu.regs.read_ar(t);
                            if v == base {
                                let ea = proc2.cpu.regs.read_ar(s).wrapping_add(imm);
                                ptr_stores.push((m, pc, ea, v));
                            }
                        }
                    }
                    match proc2.cpu.step(&mut proc2.bus) {
                        Step::Ran | Step::Exception { .. } => m += 1,
                        Step::Wait(_) | Step::Unknown { .. } => break,
                    }
                }
                eprintln!(
                    "--- stores whose value == struct base {base:#x} (FW_ALIVE_OFF publish candidates) ---"
                );
                for (m, pc, ea, v) in &ptr_stores {
                    eprintln!("  n={m} pc={pc:#x}  [{ea:#x}] <- {v:#x}");
                }
            }
        }
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
        // The steady-state poll (FUN_00008c68 @0x8c8b) checks BIT3 of each page,
        // not bit0/1; override the seeded bits with XDNA_FW_EVENT_BITS (hex).
        let bits = std::env::var("XDNA_FW_EVENT_BITS")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .unwrap_or(0b11);
        // The poll (FUN_00008c68 @0x8c88) actually reads a BYTE at a RAM struct
        // 0xf9e0 + k*0x60 (base a8), NOT the HW page in a5 (that is the ack
        // target 0x2727n114). Seed the RAM struct pending bytes via the CPU's
        // D-side accessor (data_write8, alias-correct). Keep the HW-page seed too.
        // HW event pages 0x2727n000 are read by the poll handler (FUN_8c68
        // @0x8c93) through the CPU's DTLB, so seed them via the alias-correct
        // translation path -- a raw bus.data_store32 lands on a different backing
        // than the translated read (the alias tax; session-2d translation gap).
        let seed = |proc: &mut FirmwareProcessor| {
            for p in EVENT_PAGES {
                let _ = proc.cpu.data_write32(&mut proc.bus, p, bits);
            }
            for k in 0..8u32 {
                let _ = proc.cpu.data_write8(&mut proc.bus, 0xf9e0 + k * 0x60, bits);
            }
        };
        // Deliver the completion at a warmup point (XDNA_FW_EVENT_SEED_AT) so it
        // lands while task B is polling in steady state -- seeding at n=0 is wiped
        // by early-boot memset of the RAM pending bytes before the poll ever runs.
        let seed_at: u64 = std::env::var("XDNA_FW_EVENT_SEED_AT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        // Done-flag addresses of the two known tasks (task+0x30): watch for a
        // natural (firmware-driven, not forced) set.
        const DONE_FLAGS: [u32; 2] = [0x9070, 0x10f40];
        let mut done_set: Vec<(u32, u64)> = Vec::new();

        const MAX: u64 = 1_000_000;
        const KEEP: usize = 40;
        let mut n = 0u64;
        let mut stop = String::from("budget reached");
        // Sentinel: 0x8c8e is the fall-through from `bbci a9,bit3,0x8cae` at
        // 0x8c8b -- reached ONLY when FUN_8c68 sees BIT3 SET at the polled status
        // page. If this stays 0 with bits=0x8 seeded, the CPU's translated read
        // of 0x2727n000 never sees our seed -> seed invalid (translation gap).
        const ACTIVE_PATH_PC: u32 = 0x8c8e;
        let mut active_hits = 0u64;
        let mut ring: std::collections::VecDeque<(u64, u32, String)> =
            std::collections::VecDeque::with_capacity(KEEP + 1);
        while n < MAX {
            let pc = proc.cpu.pc;
            if pc == ACTIVE_PATH_PC {
                active_hits += 1;
            }
            if n == seed_at {
                seed(&mut proc);
            } else if reseed && n > seed_at {
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
                if proc.cpu.data_read32(&mut proc.bus, f).unwrap_or(0) != 0
                    && !done_set.iter().any(|(a, _)| *a == f)
                {
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
            eprintln!("  [{f:#x}] = {:#x}", proc.cpu.data_read32(&mut proc.bus, f).unwrap_or(0));
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
            let img_w = proc.bus.inst_load32(a); // physical Rom path == image (fetch source)
            let loc_w = proc.bus.data_load32(a); // local_data overlay (data writes)
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
        assert_eq!(proc.bus.inst_load32(0x08b0_41f0), 0x4c00_c136, "segment B callx8 target not placed");
        // And memset's entry at phys 0x08b0e290 (= file 0x3b390): 36 41 00 -> 0x8c004136.
        assert_eq!(proc.bus.inst_load32(0x08b0_e290), 0x8c00_4136, "segment B memset entry not placed");
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

    /// M2c iter18 DIAGNOSTIC: peek 32-bit words (and one level of deref) at a
    /// set of static addresses. Used to resolve literal-pool words (L32r
    /// targets) to the pointers/sentinels/bases they hold -- e.g. the scheduler
    /// event-source pointer at lit 0x3364. XDNA_FW_PEEK=comma-sep hex addresses.
    /// Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_peek() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the peek probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let addrs: Vec<u32> = std::env::var("XDNA_FW_PEEK")
            .expect("set XDNA_FW_PEEK=addr,addr,... (hex)")
            .split(',')
            .filter_map(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok())
            .collect();
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let proc = FirmwareProcessor::load_m2c(img);
        let rd =
            |a: u32| u32::from_le_bytes(std::array::from_fn(|k| proc.bus.peek8(a.wrapping_add(k as u32))));
        eprintln!("=== M2c peek ===");
        for a in addrs {
            let w = rd(a);
            let deref = rd(w);
            eprintln!(
                "  [{a:#010x}] = {w:#010x}  {:<24}  ->deref [{w:#010x}] = {deref:#010x}",
                nearest_symbol(&proc.symbols, w)
            );
        }
    }

    /// M2c iter18 DIAGNOSTIC: does boot ever enter the event-ISR path
    /// (`FUN_00005580`) and read the event-source register `0x27010d28`?
    /// Decides the fork: path never entered -> interrupt-vectored, interrupt
    /// never fires; path entered with sentinel-value -> value problem. Records
    /// pc-hits at the ISR entry / event-loop / dispatch-call / event-dispatcher,
    /// every load whose EA == 0x27010d28 (with the value the fw actually read),
    /// and the final PS.INTLEVEL. XDNA_FW_MAX overrides the budget. Ignored
    /// unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_event_source() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the event-source probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1_500_000);
        const EVENT_SRC: u32 = 0x2701_0d28;
        // pc landmarks in FUN_00005580 / the event dispatcher.
        const ISR_ENTRY: u32 = 0x5580;
        const EVENT_LOOP: u32 = 0x57e0; // Beqi a6,5 -- generic-event decode
        const SRC_LOAD: u32 = 0x57f8; // L32i a2,[a14+0] -- reads the event source
        const DISPATCH_CALL: u32 = 0x5809; // Call8 FUN_0000d84c(mask)
        const DISPATCHER: u32 = 0xd84c; // FUN_0000d84c entry

        let mut pc_hits: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        let mut src_reads: u64 = 0;
        let mut src_samples: Vec<u32> = Vec::new();
        let mut n = 0u64;
        let mut stop = "budget reached";
        while n < max {
            let pc = proc.cpu.pc;
            if matches!(pc, ISR_ENTRY | EVENT_LOOP | SRC_LOAD | DISPATCH_CALL | DISPATCHER) {
                *pc_hits.entry(pc).or_insert(0) += 1;
            }
            // Generic EA watch for loads of the event-source register.
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                let op = decode::decode(&b, pc).op;
                let ea = match op {
                    decode::Op::L32i { s, imm, .. } | decode::Op::L32iN { s, imm, .. } => {
                        Some(proc.cpu.regs.read_ar(s).wrapping_add(imm))
                    }
                    _ => None,
                };
                if ea == Some(EVENT_SRC) {
                    src_reads += 1;
                    // Capture the value the fw reads: step, then read the dest reg.
                    if let decode::Op::L32i { t, .. } | decode::Op::L32iN { t, .. } =
                        decode::decode(&b, pc).op
                    {
                        let _ = proc.cpu.step(&mut proc.bus);
                        n += 1;
                        let v = proc.cpu.regs.read_ar(t);
                        if src_samples.len() < 8 {
                            src_samples.push(v);
                        }
                        continue;
                    }
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(_) => {
                    stop = "waiti";
                    break;
                }
                Step::Unknown { .. } => {
                    stop = "unknown op";
                    break;
                }
            }
            if proc.bus.sysstub().spinning().is_some() {
                stop = "spin detected";
                break;
            }
        }
        eprintln!("=== M2c event-source probe ===");
        eprintln!("instrs = {n}, stop = {stop}, last_pc = {:#x}", proc.cpu.pc);
        eprintln!("PS.INTLEVEL = {}", proc.cpu.regs.intlevel());
        eprintln!("--- pc landmark hits (FUN_00005580 event ISR + dispatcher) ---");
        for (pc, c) in &pc_hits {
            let label = match *pc {
                ISR_ENTRY => "ISR entry (0x5580)",
                EVENT_LOOP => "event-loop decode (0x57e0)",
                SRC_LOAD => "event-source load (0x57f8)",
                DISPATCH_CALL => "dispatch call (0x5809)",
                DISPATCHER => "FUN_0000d84c (0xd84c)",
                _ => "?",
            };
            eprintln!("  {pc:#08x}  {label:<28}  hits={c}");
        }
        if pc_hits.is_empty() {
            eprintln!("  (NONE -- the event ISR path is never entered)");
        }
        eprintln!("--- event-source register 0x27010d28 ---");
        eprintln!("  reads = {src_reads}, sample values = {src_samples:x?}");
    }

    /// M2c iter18 DIAGNOSTIC: scan mapped memory ranges for a target 32-bit LE
    /// word (e.g. a function pointer 0x00005580 to locate the vector/dispatch
    /// slot that reaches the event ISR). XDNA_FW_SCAN=targetword (hex),
    /// XDNA_FW_SCAN_RANGES=start:end,start:end,... (hex; default = low image +
    /// relocated segment). Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_word_scan() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the word-scan probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let target = u32::from_str_radix(
            std::env::var("XDNA_FW_SCAN")
                .expect("set XDNA_FW_SCAN=word (hex)")
                .trim()
                .trim_start_matches("0x"),
            16,
        )
        .expect("target hex");
        let ranges: Vec<(u32, u32)> = std::env::var("XDNA_FW_SCAN_RANGES")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|r| {
                        let (a, b) = r.split_once(':')?;
                        Some((
                            u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?,
                            u32::from_str_radix(b.trim().trim_start_matches("0x"), 16).ok()?,
                        ))
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![(0x0, 0x4_0000), (0x0800_0000, 0x0900_0000)]);
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let proc = FirmwareProcessor::load_m2c(img);
        let rd =
            |a: u32| u32::from_le_bytes(std::array::from_fn(|k| proc.bus.peek8(a.wrapping_add(k as u32))));
        eprintln!("=== M2c word-scan for {target:#010x} ===");
        let mut hits = 0u64;
        for (start, end) in ranges {
            let mut a = start & !3;
            while a < end {
                if rd(a) == target {
                    eprintln!("  {a:#010x}  {}", nearest_symbol(&proc.symbols, a));
                    hits += 1;
                }
                a = a.wrapping_add(4);
            }
        }
        eprintln!("total hits = {hits}");
    }

    /// M2c iter18 DIAGNOSTIC: runtime store-VALUE watch. Runs boot and records
    /// every store (S32i/S32iN/S16i/S8i) whose stored value == XDNA_FW_WATCH_VAL
    /// (hex), with the store address + pc. Locates where a function pointer
    /// (e.g. the event ISR 0x5580) or a magic gets installed into a table at
    /// runtime -- the store address is the dispatch-table slot (which encodes
    /// the IRQ number). XDNA_FW_MAX overrides the budget. Ignored unless
    /// XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_store_value_watch() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the store-value watch");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let target = u32::from_str_radix(
            std::env::var("XDNA_FW_WATCH_VAL")
                .expect("set XDNA_FW_WATCH_VAL=value (hex)")
                .trim()
                .trim_start_matches("0x"),
            16,
        )
        .expect("value hex");
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1_500_000);
        let mut n = 0u64;
        let mut stop = "budget reached";
        let mut hits: Vec<(u64, u32, u32)> = Vec::new();
        while n < max {
            let pc = proc.cpu.pc;
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                let d = decode::decode(&b, pc);
                let sv = match d.op {
                    decode::Op::S32i { t, s, imm }
                    | decode::Op::S32iN { t, s, imm }
                    | decode::Op::S16i { t, s, imm }
                    | decode::Op::S8i { t, s, imm }
                    | decode::Op::S32ri { t, s, imm }
                    | decode::Op::S32c1i { t, s, imm } => {
                        Some((proc.cpu.regs.read_ar(t), proc.cpu.regs.read_ar(s).wrapping_add(imm)))
                    }
                    _ => None,
                };
                if let Some((val, addr)) = sv {
                    if val == target && hits.len() < 64 {
                        hits.push((n, pc, addr));
                    }
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(_) => {
                    stop = "waiti";
                    break;
                }
                Step::Unknown { .. } => {
                    stop = "unknown op";
                    break;
                }
            }
            if proc.bus.sysstub().spinning().is_some() {
                stop = "spin detected";
                break;
            }
        }
        eprintln!("=== M2c store-value watch for {target:#010x} ===");
        eprintln!("instrs = {n}, stop = {stop}");
        for (at, pc, addr) in &hits {
            eprintln!(
                "  n={at:>8}  pc={pc:#08x} {:<26}  store -> [{addr:#010x}]",
                nearest_symbol(&proc.symbols, *pc)
            );
        }
        eprintln!("total stores of {target:#010x} = {}", hits.len());
    }

    /// M2c iter18 DIAGNOSTIC: runtime store-to-ADDRESS watch. Runs boot and
    /// records every store (S32i/S32iN/S16i/S8i) whose target address is in
    /// XDNA_FW_WATCH_ADDR (comma-sep hex), with the stored value + pc + instr.
    /// Directly shows whether/where the pending-event words (0x22bc, task+0x30)
    /// or the current-task pointer (0x2278) get written -- the event-injection
    /// point. XDNA_FW_MAX overrides the budget. Ignored unless XDNA_FW_PROBE set.
    #[test]
    fn m2c_probe_addr_store_watch() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the addr store watch");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let watch: Vec<u32> = std::env::var("XDNA_FW_WATCH_ADDR")
            .expect("set XDNA_FW_WATCH_ADDR=addr,addr,... (hex)")
            .split(',')
            .filter_map(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok())
            .collect();
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1_500_000);
        let mut n = 0u64;
        let mut stop = "budget reached";
        // (n, pc, addr, value); cap per address so a hot store can't flood.
        let mut hits: Vec<(u64, u32, u32, u32)> = Vec::new();
        let mut per_addr: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
        while n < max {
            let pc = proc.cpu.pc;
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                let d = decode::decode(&b, pc);
                let sv = match d.op {
                    decode::Op::S32i { t, s, imm }
                    | decode::Op::S32iN { t, s, imm }
                    | decode::Op::S16i { t, s, imm }
                    | decode::Op::S8i { t, s, imm }
                    | decode::Op::S32ri { t, s, imm }
                    | decode::Op::S32c1i { t, s, imm } => {
                        Some((proc.cpu.regs.read_ar(t), proc.cpu.regs.read_ar(s).wrapping_add(imm)))
                    }
                    _ => None,
                };
                if let Some((val, addr)) = sv {
                    if watch.contains(&addr) {
                        let c = per_addr.entry(addr).or_insert(0);
                        *c += 1;
                        if *c <= 12 {
                            hits.push((n, pc, addr, val));
                        }
                    }
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(_) => {
                    stop = "waiti";
                    break;
                }
                Step::Unknown { .. } => {
                    stop = "unknown op";
                    break;
                }
            }
            if proc.bus.sysstub().spinning().is_some() {
                stop = "spin detected";
                break;
            }
        }
        eprintln!("=== M2c addr store watch {watch:x?} ===");
        eprintln!("instrs = {n}, stop = {stop}");
        for (at, pc, addr, val) in &hits {
            eprintln!(
                "  n={at:>8}  pc={pc:#010x}  [{addr:#010x}] <- {val:#010x}   {}",
                nearest_symbol(&proc.symbols, *pc & 0x00ff_ffff)
            );
        }
        eprintln!("--- store counts per watched address ---");
        let mut counts: Vec<(u32, u64)> = per_addr.into_iter().collect();
        counts.sort_by_key(|(a, _)| *a);
        for (a, c) in counts {
            eprintln!("  [{a:#010x}]: {c}");
        }
    }

    /// M2c iter18 DIAGNOSTIC: POLL-based value watch (reliable). Reads each
    /// XDNA_FW_POLL_ADDR (comma-sep hex) via `bus.data_load32` every step and
    /// records value changes (n, pc, old->new). Unlike the store-EA watches,
    /// this catches writes through ANY addressing/alias (the firmware's windowed
    /// RAM is written via aliased virtual addresses that exact-EA matching
    /// misses -- proven by `0x2278`). Use for the pending-event words / task
    /// state. XDNA_FW_MAX overrides budget. Ignored unless XDNA_FW_PROBE set.
    #[test]
    fn m2c_probe_poll_watch() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the poll watch");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let addrs: Vec<u32> = std::env::var("XDNA_FW_POLL_ADDR")
            .expect("set XDNA_FW_POLL_ADDR=addr,addr,... (hex)")
            .split(',')
            .filter_map(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok())
            .collect();
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1_500_000);
        let mut last: Vec<u32> = addrs.iter().map(|a| proc.bus.data_load32(*a)).collect();
        let mut changes: Vec<(u64, u32, u32, u32, u32)> = Vec::new(); // n, pc, addr, old, new
        let mut per: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
        let mut n = 0u64;
        let mut stop = "budget reached";
        while n < max {
            let pc = proc.cpu.pc;
            for (i, a) in addrs.iter().enumerate() {
                let v = proc.bus.data_load32(*a);
                if v != last[i] {
                    let c = per.entry(*a).or_insert(0);
                    *c += 1;
                    if *c <= 16 {
                        changes.push((n, pc, *a, last[i], v));
                    }
                    last[i] = v;
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(_) => {
                    stop = "waiti";
                    break;
                }
                Step::Unknown { .. } => {
                    stop = "unknown op";
                    break;
                }
            }
            if proc.bus.sysstub().spinning().is_some() {
                stop = "spin detected";
                break;
            }
        }
        eprintln!("=== M2c poll watch {addrs:x?} ===");
        eprintln!("instrs = {n}, stop = {stop}");
        for (at, pc, addr, old, new) in &changes {
            eprintln!("  n={at:>8}  pc={:#08x}  [{addr:#010x}] {old:#010x} -> {new:#010x}", pc & 0x00ff_ffff);
        }
        eprintln!("--- change counts per address ---");
        let mut counts: Vec<(u32, u64)> = per.into_iter().collect();
        counts.sort_by_key(|(a, _)| *a);
        for (a, c) in counts {
            eprintln!("  [{a:#010x}]: {c} change(s)");
        }
    }

    /// M2c iter18 DIAGNOSTIC: clean EXECUTION trace (follows real PCs, so decode
    /// alignment is always correct -- unlike linear disasm). Warms up
    /// XDNA_FW_TRACE_WARMUP instrs, then prints the next XDNA_FW_TRACE_COUNT as
    /// `n pc symbol op | a2..a7 | ea=<load/store EA & value>`. Reveals the
    /// steady-state task work loop and its decision points. Ignored unless
    /// XDNA_FW_PROBE set.
    #[test]
    fn m2c_probe_exec_trace() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the exec trace");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        let warmup = env_u64("XDNA_FW_TRACE_WARMUP", 300_000);
        let count = env_u64("XDNA_FW_TRACE_COUNT", 400);
        // Optional: seed the per-column poll-completion flags each traced step,
        // so the trace shows the work-fn's SATISFIED-poll path (what it reads and
        // branches on AFTER the poll succeeds). XDNA_FW_TRACE_SEEDPOLL=<hex bits>
        // (e.g. 0xb = bit0|bit1|bit3).
        let seedpoll: Option<u32> = std::env::var("XDNA_FW_TRACE_SEEDPOLL")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok());
        for _ in 0..warmup {
            if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
                break;
            }
        }
        eprintln!("=== M2c exec trace (warmup {warmup}, {count} instrs, seedpoll={seedpoll:x?}) ===");
        for i in 0..count {
            if let Some(bits) = seedpoll {
                for k in 0..4u32 {
                    let _ = proc.cpu.data_write32(&mut proc.bus, 0x2727_1000 + k * 0x1000, bits);
                    let _ = proc.cpu.data_write8(&mut proc.bus, 0xf9e0 + k * 0x60, bits);
                }
            }
            let pc = proc.cpu.pc;
            let (op, ea) = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                Ok(phys) => {
                    let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                    let d = decode::decode(&b, pc);
                    let ea = match d.op {
                        decode::Op::L32i { s, imm, .. }
                        | decode::Op::L32iN { s, imm, .. }
                        | decode::Op::L8ui { s, imm, .. }
                        | decode::Op::L16ui { s, imm, .. }
                        | decode::Op::S32i { s, imm, .. }
                        | decode::Op::S32iN { s, imm, .. }
                        | decode::Op::S8i { s, imm, .. } => {
                            let a = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                            format!(" ea={:#x}={:#x}", a, proc.bus.data_load32(a & 0x00ff_ffff))
                        }
                        _ => String::new(),
                    };
                    (format!("{:?}", d.op), ea)
                }
                Err(_) => ("<fault>".to_string(), String::new()),
            };
            let regs: Vec<String> = (2..8).map(|r| format!("a{r}={:#x}", proc.cpu.regs.read_ar(r))).collect();
            eprintln!(
                "{:>7} {:#08x} {:<22} {:<34}{ea} | {}",
                warmup + i,
                pc & 0x00ff_ffff,
                nearest_symbol(&proc.symbols, pc & 0x00ff_ffff),
                op,
                regs.join(" ")
            );
            if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
                eprintln!("stop");
                break;
            }
        }
    }

    /// FAITHFUL-MODEL step 1 (2026-07-07, Maya "read that await-mask"): the boot
    /// worker task `0x10f10` parks at the dispatcher done-check `0xd828` because
    /// no event is delivered. `wake_tasks_by_event_mask` (0xd84c) wakes a task
    /// whose await-mask `[task+0x38]` intersects the delivered event mask. To
    /// deliver the RIGHT event we first need that await-mask. This probe boots to
    /// the first moment `0x10f10` is observed parked (pc==0xd828, a4==task,
    /// pending `[task+0x30]`==0) and dumps its task struct neighborhood
    /// (0x10f10..0x10f50): +0x2c state byte, +0x30 pending/event, +0x38 await-mask,
    /// plus `[sched+108]` (0x22e4, the global pending accumulator). Env:
    /// XDNA_FW_MAX (default 3_000_000). Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_await_mask() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the await-mask read");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        proc.bus.attach_device(crate::device::DeviceState::new_npu1());
        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3_000_000);
        const DONE_CHECK_PC: u32 = 0xd828;
        const TASK: u32 = 0x10f10;

        let mut n = 0u64;
        let mut found_at: Option<u64> = None;
        let mut sched = 0u32;
        while n < max {
            let pc = proc.cpu.pc;
            if pc == DONE_CHECK_PC && proc.cpu.regs.read_ar(4) == TASK {
                let pending = proc.cpu.data_read32(&mut proc.bus, TASK + 0x30).unwrap_or(0);
                if pending == 0 {
                    found_at = Some(n);
                    sched = proc.cpu.regs.read_ar(3); // a3 = scheduler base at 0xd828
                    break;
                }
            }
            if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
                break;
            }
            n += 1;
        }

        eprintln!("=== await-mask read: task {TASK:#x} ===");
        match found_at {
            Some(at) => eprintln!("parked (pending==0) at 0xd828, n={at}, sched(a3)={sched:#x}"),
            None => {
                eprintln!("NOT observed parked within {max} instrs (n={n}); dumping current state anyway");
            }
        }
        eprintln!("--- task struct [0x10f10..0x10f50] ---");
        for w in 0..16u32 {
            let a = TASK + w * 4;
            let v = proc.cpu.data_read32(&mut proc.bus, a).unwrap_or(0);
            let tag = match w * 4 {
                0x2c => "  <- +0x2c state byte (low 8 bits)",
                0x30 => "  <- +0x30 pending/event",
                0x38 => "  <- +0x38 AWAIT-MASK",
                _ => "",
            };
            eprintln!("  [{a:#07x}] (+{:#04x}) = {v:#010x}{tag}", w * 4);
        }
        // The waker (0xd84c) iterates the 9-task table at [sched+56] and wakes any
        // whose await-mask [task+0x38] intersects the delivered event. Dump all 9
        // so we can see whether ANY task is event-driven during boot (nonzero
        // await-mask) or the whole event-wake path is dormant (all zero => boot is
        // poll-gated, not event-gated).
        if sched != 0 {
            let global_pending = proc.cpu.data_read32(&mut proc.bus, sched + 108).unwrap_or(0);
            eprintln!("[sched+108] ({:#x}) global pending = {global_pending:#010x}", sched + 108);
            eprintln!(
                "--- 9-task wake table [sched+56] (task -> +0x38 await-mask, +0x30 pending, +0x2c state) ---"
            );
            for i in 0..9u32 {
                let ent = proc.cpu.data_read32(&mut proc.bus, sched + 56 + i * 4).unwrap_or(0);
                if ent == 0 {
                    eprintln!("  [{}] = 0", i);
                    continue;
                }
                let await_mask = proc.cpu.data_read32(&mut proc.bus, ent + 0x38).unwrap_or(0);
                let pending = proc.cpu.data_read32(&mut proc.bus, ent + 0x30).unwrap_or(0);
                let state = proc.cpu.data_read32(&mut proc.bus, ent + 0x2c).unwrap_or(0) & 0xff;
                let here = if ent == TASK { "  <== 0x10f10" } else { "" };
                eprintln!("  [{i}] task={ent:#08x}  await={await_mask:#010x}  pending={pending:#010x}  state={state:#04x}{here}");
            }
        }
    }

    /// FAITHFUL-MODEL step 2 (2026-07-07, Maya "map the seam"): the event-wake
    /// path is dormant during boot (all await-masks 0), so boot is POLL-gated. To
    /// map the poll seam we need to know, in steady state, (a) where the PC lives
    /// (which loop the fw is stuck in) and (b) whether that loop reads ANY external
    /// MMIO address (>= 0x2000_0000) -- the register the fw polls for
    /// column-power-ready that our shim must answer. This histograms PC by nearest
    /// function over a deep window and tallies every distinct external-read EA with
    /// hit counts + the reading PC. If the steady loop touches no external address,
    /// the gate is internal scheduler state, not a hardware poll. Env:
    /// XDNA_FW_HIST_WARMUP (default 200_000), XDNA_FW_HIST_COUNT (default
    /// 2_000_000). Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_steady_histogram() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the steady-state histogram");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        proc.bus.attach_device(crate::device::DeviceState::new_npu1());
        let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        let warmup = env_u64("XDNA_FW_HIST_WARMUP", 200_000);
        let count = env_u64("XDNA_FW_HIST_COUNT", 2_000_000);

        for _ in 0..warmup {
            if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
                break;
            }
        }

        // PC histogram bucketed by nearest symbol, and distinct external-read EAs.
        let mut pc_hist: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
        // ea -> (hits, first reading pc)
        let mut ext_reads: std::collections::BTreeMap<u32, (u64, u32)> = std::collections::BTreeMap::new();
        let mut stopped = String::from("count reached");
        for _ in 0..count {
            let pc = proc.cpu.pc;
            *pc_hist.entry(nearest_symbol(&proc.symbols, pc & 0x00ff_ffff)).or_insert(0) += 1;
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                let ea = match decode::decode(&b, pc).op {
                    Op::L32i { s, imm, .. }
                    | Op::L32iN { s, imm, .. }
                    | Op::L8ui { s, imm, .. }
                    | Op::L16ui { s, imm, .. } => Some(proc.cpu.regs.read_ar(s).wrapping_add(imm)),
                    _ => None,
                };
                if let Some(ea) = ea {
                    if ea >= 0x2000_0000 {
                        let e = ext_reads.entry(ea).or_insert((0, pc & 0x00ff_ffff));
                        e.0 += 1;
                    }
                }
            }
            if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
                stopped = format!("halted at pc={:#x}", proc.cpu.pc);
                break;
            }
        }

        eprintln!("=== steady-state histogram (warmup {warmup}, {count} samples) ===");
        eprintln!("stop = {stopped}");
        let mut ranked: Vec<_> = pc_hist.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!("--- top PC buckets (by nearest function) ---");
        for (sym, hits) in ranked.iter().take(15) {
            eprintln!("  {hits:>9}  {sym}");
        }
        eprintln!("--- distinct external-read EAs (>= 0x2000_0000) ---");
        if ext_reads.is_empty() {
            eprintln!("  (none -- steady state reads NO external MMIO; gate is internal state)");
        } else {
            for (ea, (hits, pc)) in &ext_reads {
                eprintln!("  {ea:#010x}  hits={hits:>8}  first-read-pc={pc:#x}");
            }
        }
    }

    /// FAITHFUL-MODEL step 3 (2026-07-07, Maya-approved): the DECISIVE idle-vs-
    /// deadlock test. Session-9 cont'd found the per-column pending SETTER
    /// (`FUN_00008c14`) is called only from `FUN_00035444`, an aie-rt array-init
    /// HAL routine DORMANT during boot -- so the 58k scheduler spin is likely the
    /// IDLE wait-for-host-work loop, not a deadlock (the fw programs the array only
    /// in response to a host mailbox command). This boots to the steady spin, then
    /// injects the mailbox DOORBELL (Xtensa interrupt bit0, host mailbox enabled),
    /// and watches whether the fw LEAVES the scheduler loop -- entering the mailbox
    /// receive/ISR path and/or the array-init HAL (`0x35000..0x36000`), and whether
    /// bit3 of the pending struct (`0xf9e0`) finally gets set. If the doorbell wakes
    /// it, the spin was idle-waiting; if nothing changes, it is a real deadlock.
    /// Env: XDNA_FW_MB_WARMUP (default 300_000), XDNA_FW_MB_STEPS (default 300_000),
    /// XDNA_FW_MB_ARRAY (presence => attach array). Ignored unless XDNA_FW_PROBE set.
    #[test]
    fn m2c_probe_mailbox_wake() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the mailbox-wake idle-vs-deadlock test");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        proc.enable_host_mailbox();
        if std::env::var("XDNA_FW_MB_ARRAY").is_ok() {
            proc.bus.attach_device(crate::device::DeviceState::new_npu1());
        }
        let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        let warmup = env_u64("XDNA_FW_MB_WARMUP", 300_000);
        let steps = env_u64("XDNA_FW_MB_STEPS", 300_000);

        // Boot into the steady scheduler spin, recording the set of nearest-symbols
        // the spin visits (its "home" PC set) so we can flag NEW territory after the
        // doorbell.
        let mut spin_syms: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for i in 0..warmup {
            // Sample the spin's home set over the last 20k instrs before the doorbell.
            if i + 20_000 >= warmup {
                spin_syms.insert(nearest_symbol(&proc.symbols, proc.cpu.pc & 0x00ff_ffff));
            }
            if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
                break;
            }
        }
        let bit3_before = proc.cpu.data_read32(&mut proc.bus, 0xf9e0).unwrap_or(0) & 0x08;
        eprintln!("=== mailbox-wake: idle-vs-deadlock test ===");
        eprintln!(
            "pre-doorbell: at pc={:#x} [{}], INTENABLE={:#010x}, spin visits {} distinct syms, 0xf9e0 bit3={}",
            proc.cpu.pc,
            nearest_symbol(&proc.symbols, proc.cpu.pc & 0x00ff_ffff),
            proc.cpu.intenable,
            spin_syms.len(),
            bit3_before != 0,
        );

        // Inject the mailbox doorbell.
        proc.cpu.intenable |= 1;
        proc.cpu.interrupt |= 1;

        // Step post-doorbell, tracking NEW symbols (outside the spin home set),
        // first entry into the HAL region (0x35000..0x36000), first bit3 set, and
        // where it ends up.
        let mut new_syms: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
        let mut hal_entry_at: Option<(u64, u32)> = None;
        let mut bit3_set_at: Option<u64> = None;
        let mut stop = String::from("steps budget reached");
        let mut n = 0u64;
        while n < steps {
            let pc = proc.cpu.pc & 0x00ff_ffff;
            let sym = nearest_symbol(&proc.symbols, pc);
            if !spin_syms.contains(&sym) {
                *new_syms.entry(sym).or_insert(0) += 1;
            }
            if hal_entry_at.is_none() && (0x35000..0x36000).contains(&pc) {
                hal_entry_at = Some((n, pc));
            }
            if bit3_set_at.is_none() && proc.cpu.data_read32(&mut proc.bus, 0xf9e0).unwrap_or(0) & 0x08 != 0 {
                bit3_set_at = Some(n);
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?}) at pc={:#x} (idle/waiti)", proc.cpu.pc);
                    break;
                }
                Step::Unknown { pc, word } => {
                    stop = format!("Unknown at pc={pc:#x} word={word:#010x}");
                    break;
                }
            }
        }

        eprintln!("post-doorbell: ran {n} steps, stop={stop}");
        eprintln!(
            "last_pc={:#x} [{}]",
            proc.cpu.pc,
            nearest_symbol(&proc.symbols, proc.cpu.pc & 0x00ff_ffff)
        );
        eprintln!("array-init HAL (0x35000..0x36000) entered: {hal_entry_at:x?}");
        eprintln!("0xf9e0 bit3 set at: {bit3_set_at:?}");
        eprintln!("--- NEW symbols visited after doorbell (not in the spin home set) ---");
        if new_syms.is_empty() {
            eprintln!("  (none -- doorbell did NOT move the fw out of its spin; looks like a DEADLOCK)");
        } else {
            let mut ranked: Vec<_> = new_syms.into_iter().collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1));
            for (sym, hits) in ranked.iter().take(30) {
                eprintln!("  {hits:>8}  {sym}");
            }
        }
    }

    /// FAITHFUL-MODEL step 4 -- experiment (A), Maya-approved: the PRINCIPLED SHIM
    /// of the now-mapped seam. bit3 of the per-column byte `0xf9e0+col*0x60` = "the
    /// DMA HAL marked this column's BD work pending"; the poll `FUN_00008c68`
    /// services it via the `0x2727_(col)000` handshake (read bit0, write, wait
    /// bit1) then clears bit3. On silicon the DMA HAL sets bit3 synchronously; in
    /// EMU it never runs, so the fw spins. This shim SIMULATES the HAL completion:
    /// set bit3 (local) AND bit0|bit1 on the external page (request-present +
    /// already-acked, so the poll's bit1-wait exits immediately -- setting bit0
    /// WITHOUT bit1 would hang the poll at INTLEVEL 2). Then watch whether boot
    /// walks PAST the spin toward a real INTLEVEL-0 idle (`Step::Wait`), enters new
    /// code, or reaches the array-init HAL. Env: XDNA_FW_SHIM_WARMUP (default
    /// 200_000), XDNA_FW_SHIM_STEPS (default 400_000), XDNA_FW_SHIM_MODE
    /// (`once`=default, set at warmup only; `cont`=reassert every step),
    /// XDNA_FW_SHIM_COLS (default 4), XDNA_FW_SHIM_EXTBITS (hex, default 0x3 =
    /// bit0|bit1). Ignored unless XDNA_FW_PROBE set.
    #[test]
    fn m2c_probe_bit3_shim() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the bit3 completion-shim experiment");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        proc.bus.attach_device(crate::device::DeviceState::new_npu1());
        let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        let env_hex = |k: &str, d: u32| {
            std::env::var(k)
                .ok()
                .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
                .unwrap_or(d)
        };
        let warmup = env_u64("XDNA_FW_SHIM_WARMUP", 200_000);
        let steps = env_u64("XDNA_FW_SHIM_STEPS", 400_000);
        let cont = std::env::var("XDNA_FW_SHIM_MODE").map(|s| s == "cont").unwrap_or(false);
        let cols = env_u64("XDNA_FW_SHIM_COLS", 4) as u32;
        let extbits = env_hex("XDNA_FW_SHIM_EXTBITS", 0x3);
        // The "together" half of the synchronous contract: also deliver the task
        // done-flag [t+0x30]=1 + column [t+8]=SHIM_COL for these tasks (hex csv,
        // default empty = poll-only). Neither poll-alone (this probe's default)
        // nor flag-alone (m2c_probe_external_complete) advances boot; this tests
        // whether the two together retire the column-power worker.
        let flag_tasks: Vec<u32> = std::env::var("XDNA_FW_SHIM_TASKS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .collect();
        let flag_col = env_hex("XDNA_FW_SHIM_COL", 0);

        // Inject the simulated DMA-HAL completion for `cols` columns.
        let inject = |proc: &mut FirmwareProcessor| {
            for col in 0..cols {
                let byte_addr = 0xf9e0 + col * 0x60;
                let cur = proc.cpu.data_read32(&mut proc.bus, byte_addr & !3).unwrap_or(0);
                let byte = ((cur >> ((byte_addr & 3) * 8)) & 0xff) | 0x08;
                let _ = proc.cpu.data_write8(&mut proc.bus, byte_addr, byte);
                let ext = 0x2727_1000 + col * 0x1000;
                let _ = proc.cpu.data_write32(&mut proc.bus, ext, extbits);
            }
            for &t in &flag_tasks {
                let _ = proc.cpu.data_write32(&mut proc.bus, t.wrapping_add(0x30), 1);
                let _ = proc.cpu.data_write32(&mut proc.bus, t.wrapping_add(8), flag_col);
            }
        };

        // Boot into the steady spin, capturing its home symbol set.
        let mut spin_syms: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for i in 0..warmup {
            if i + 20_000 >= warmup {
                spin_syms.insert(nearest_symbol(&proc.symbols, proc.cpu.pc & 0x00ff_ffff));
            }
            if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
                break;
            }
        }
        eprintln!(
            "=== bit3 completion-shim experiment (mode={}, cols={cols}, extbits={extbits:#x}) ===",
            if cont { "cont" } else { "once" }
        );
        let rec_before: [u32; 4] = std::array::from_fn(|i| {
            proc.cpu.data_read32(&mut proc.bus, 0x2320 + (i as u32) * 4).unwrap_or(0)
        });
        eprintln!(
            "pre-shim: pc={:#x} [{}], spin visits {} distinct syms  [0x2320]={:08x?}",
            proc.cpu.pc,
            nearest_symbol(&proc.symbols, proc.cpu.pc & 0x00ff_ffff),
            spin_syms.len(),
            rec_before,
        );
        inject(&mut proc);

        let mut new_syms: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
        let mut hal_entry_at: Option<(u64, u32)> = None;
        // Distinct task entries the scheduler scan evaluates: entry base a4 (at
        // sched_task_scan+0x32, `l8ui a8,[a4+8]`) -> (byte[+4] ready, byte[+8]).
        // Shows which task the post-worker scan is (not) finding ready.
        let mut scan_entries: std::collections::BTreeMap<u32, (u32, u32)> = std::collections::BTreeMap::new();
        let mut stop = String::from("steps budget reached");
        let mut n = 0u64;
        while n < steps {
            if cont {
                inject(&mut proc);
            }
            let pc = proc.cpu.pc & 0x00ff_ffff;
            if pc == 0x7c22 {
                let a4 = proc.cpu.regs.read_ar(4);
                let b4 = proc.cpu.data_read32(&mut proc.bus, a4 + 4).unwrap_or(0);
                let b8 = proc.cpu.data_read32(&mut proc.bus, a4 + 8).unwrap_or(0);
                scan_entries.entry(a4).or_insert((b4, b8));
            }
            let sym = nearest_symbol(&proc.symbols, pc);
            if !spin_syms.contains(&sym) {
                *new_syms.entry(sym).or_insert(0) += 1;
            }
            if hal_entry_at.is_none() && (0x35000..0x36000).contains(&pc) {
                hal_entry_at = Some((n, pc));
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    n += 1;
                    stop = format!("Wait({reason:?}) at pc={:#x} (REACHED IDLE!)", proc.cpu.pc);
                    break;
                }
                Step::Unknown { pc, word } => {
                    stop = format!("Unknown at pc={pc:#x} word={word:#010x}");
                    break;
                }
            }
            if let Some(addr) = proc.bus.sysstub().spinning() {
                stop = format!("sysstub spin at {addr:#x}");
                break;
            }
        }

        eprintln!("post-shim: ran {n} steps, stop={stop}");
        eprintln!(
            "last_pc={:#x} [{}]",
            proc.cpu.pc,
            nearest_symbol(&proc.symbols, proc.cpu.pc & 0x00ff_ffff)
        );
        let rec_after: [u32; 4] = std::array::from_fn(|i| {
            proc.cpu.data_read32(&mut proc.bus, 0x2320 + (i as u32) * 4).unwrap_or(0)
        });
        eprintln!("[0x2320] AFTER={rec_after:08x?}  go-alive record intact={}", rec_after[0] == 0x55f8);
        eprintln!("--- task entries scanned by sched_task_scan (base: [+4]=ready [+8]) ---");
        for (base, (b4, b8)) in &scan_entries {
            eprintln!("  entry={base:#08x}  [+4]={b4:#010x}  [+8]={b8:#010x}");
        }
        eprintln!("array-init HAL (0x35000..0x36000) entered: {hal_entry_at:x?}");
        let bit3_now = proc.cpu.data_read32(&mut proc.bus, 0xf9e0).unwrap_or(0) & 0x08;
        eprintln!(
            "0xf9e0 bit3 after run = {} (cleared by fw => the poll SERVICED the column)",
            bit3_now != 0
        );
        eprintln!("--- NEW symbols visited after shim (not in spin home set) ---");
        if new_syms.is_empty() {
            eprintln!("  (none -- shim did NOT move the fw out of the spin; bit3 is NOT the sole gate)");
        } else {
            let mut ranked: Vec<_> = new_syms.into_iter().collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1));
            for (sym, hits) in ranked.iter().take(30) {
                eprintln!("  {hits:>8}  {sym}");
            }
        }
    }
}
