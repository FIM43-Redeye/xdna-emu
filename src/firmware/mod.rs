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
        bus.add_rom_overlay(SYSCALL_BLOCK_LO, SYSCALL_BLOCK_HI, LOW_VMA_FILE_OFFSET);

        // iter20: the syscall-yield context-switch chain, discovered by boot-driven
        // walk-and-stub past the 0x2630 seam. Each region is a +0x100 section
        // verified by coherent execution (NOT static classification): the syscall
        // handler's jump table dispatches (via PC-relative Call8) to the
        // context-switch routine at VMA 0x2630 (file 0x2730), which calls the IPC
        // critical-section primitive at 0xc48c (file 0xc58c) -- the same function
        // the scheduler reaches; it posts to the [0xfae0] mailbox and jumps into
        // Seg-B. The primitive's literal pools live in separate +0x100 rodata at
        // 0x3424/0x3c74; the callee's at 0x254c. Serving these (fetch AND l32r
        // literals, see Bus::inst_load32_overlay) runs the chain byte-coherently
        // instead of walling on the +0x5c misframe. There is NO firmware
        // relocation (zero stores to any +0x100 VMA in a full boot) and NO
        // dual-execution (0xc530, the +0x5c alias, never runs): each function has
        // one canonical VMA, set by its section's file offset. Full account:
        // docs/superpowers/findings/2026-07-10-boot-to-idle-reached.md.
        bus.add_rom_overlay(CTXSW_CALLEE_LO, CTXSW_CALLEE_HI, LOW_VMA_FILE_OFFSET);
        bus.add_rom_overlay(CTXSW_WINDOW_ROTATE_LO, CTXSW_WINDOW_ROTATE_HI, LOW_VMA_FILE_OFFSET);
        bus.add_rom_overlay(IPC_PRIMITIVE_LO, IPC_PRIMITIVE_HI, LOW_VMA_FILE_OFFSET);
        bus.add_rom_overlay(CTXSW_CALLEE_POOL_LO, CTXSW_CALLEE_POOL_HI, LOW_VMA_FILE_OFFSET);
        bus.add_rom_overlay(IPC_POOL_A_LO, IPC_POOL_A_HI, LOW_VMA_FILE_OFFSET);
        bus.add_rom_overlay(IPC_POOL_B_LO, IPC_POOL_B_HI, LOW_VMA_FILE_OFFSET);
        bus.add_rom_overlay(EXC_RESTORE_LO, EXC_RESTORE_HI, LOW_VMA_FILE_OFFSET);
        // EXC_RESTORE's scattered +0x100 literal pools (values 0xe108/0x2278/0xd900
        // -- code/data ptrs, all garbage at +0x5c). Its Callx4 targets the ISR
        // 0xd900 (already in SYSCALL_BLOCK) once 0x3cc0 reads at +0x100.
        bus.add_rom_overlay(0x0000_e0e0, 0x0000_e0e4, LOW_VMA_FILE_OFFSET);
        bus.add_rom_overlay(0x0000_31dc, 0x0000_31e0, LOW_VMA_FILE_OFFSET);
        bus.add_rom_overlay(0x0000_3cc0, 0x0000_3cc4, LOW_VMA_FILE_OFFSET);

        // iter24: the syscall-return-path ring-scan function (Call8 target from
        // FUN_00005958). +0x100 like the rest of the chain; see SYSRET_SCAN docs.
        bus.add_rom_overlay(SYSRET_SCAN_LO, SYSRET_SCAN_HI, LOW_VMA_FILE_OFFSET);

        // iter25 (2026-07-10): the go-alive publish path -- the sections that carry
        // the firmware from "booted but never alive" to a real published mgmt
        // channel. Same +0x100 walk-and-stub class as the chain above, discovered
        // by boot-driven reproduction (not static classification): with these
        // served, a NATURAL boot pops the enqueued go-alive job, runs its run-fn
        // (0x55f8), reaches publish_chann_info (0x50e8), copies the "_NPU" magic
        // (0x55504e5f) + channel descriptor into host-visible SRAM, and rests at a
        // real `waiti` (0x5645). Every code range begins with a valid `entry`
        // prologue at +0x100 and mid-instruction garbage at +0x5c; every pool word
        // is a live L32r target that reads a sane pointer/mask at +0x100 and junk
        // at +0x5c (e.g. the queue pool-base 0x3c84: 0x00002250 vs 0x06194518; the
        // magic literal 0x3288: 0x55504e5f vs 0x08b0e290). Audited over the full
        // boot: ZERO stores land in any of these VMAs (not firmware-relocated data)
        // and ZERO of their +0x5c aliases are ever executed (no dual-framing). Full
        // per-range byte justification: the iter25 table in
        // docs/superpowers/findings/2026-07-10-boot-to-idle-reached.md.
        for &(lo, hi) in &[
            // queue-pop path: work-fetch launcher, MERT pop, pool-base literal
            (0x0000_c648u32, 0x0000_c6b0u32),
            (0x0000_c6b0, 0x0000_c730),
            (0x0000_cc1c, 0x0000_ccc1),
            (0x0000_3c84, 0x0000_3c90),
            // run-fn + publisher code
            (0x0000_55f8, 0x0000_581c),
            (0x0000_501c, 0x0000_518f),
            // publish helpers (address encoder, bitfield/MMIO, NOC/array, scan)
            (0x0000_4a0c, 0x0000_4a37),
            (0x0000_4a5c, 0x0000_4ade),
            (0x0000_7bd0, 0x0000_7c1e),
            (0x0000_7cf0, 0x0000_7d40),
            (0x0000_86f8, 0x0000_8720),
            (0x0000_8970, 0x0000_89d4),
            (0x0000_8c98, 0x0000_8d52),
            (0x0000_8d88, 0x0000_8db4),
            (0x0000_8f44, 0x0000_9065),
            (0x0000_95ec, 0x0000_9704),
            (0x0000_9704, 0x0000_9777),
            (0x0000_9778, 0x0000_978f),
            // live L32r literal pools on the publish path
            (0x0000_31ac, 0x0000_31b0),
            (0x0000_325c, 0x0000_3298),
            (0x0000_329c, 0x0000_32a0),
            (0x0000_3364, 0x0000_3368),
            (0x0000_33a8, 0x0000_33ac),
            (0x0000_33f4, 0x0000_33fc),
            (0x0000_3474, 0x0000_347c),
            (0x0000_34a0, 0x0000_34a8),
            (0x0000_34dc, 0x0000_34e8),
            (0x0000_3500, 0x0000_3520),
            (0x0000_3530, 0x0000_3534),
            (0x0000_353c, 0x0000_3540),
            (0x0000_354c, 0x0000_3564),
            // go-alive tail frontier extension (past the 0x5645 status gate):
            // status literal, the callx8 Segment-B pointer pool, scheduler
            // helpers, and the syscall/context-switch dispatch prefix.
            (0x0000_31a4, 0x0000_31a8),
            (0x0000_32c8, 0x0000_32cc),
            (0x0000_7c5c, 0x0000_7cee),
            (0x0000_7d4c, 0x0000_7e28),
            (0x0000_d864, 0x0000_d8a7),
        ] {
            bus.add_rom_overlay(lo, hi, LOW_VMA_FILE_OFFSET);
        }

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
/// M2c iter19: the syscall-dispatch / scheduler-primitives block -- a THIRD
/// piecewise-relocated `+0x100` section, discovered when the boot-to-idle wall
/// resolved. The exception handler's syscall path (`0x2a88`) `Callx4`s a
/// compiled-in pointer `0xdac4`; at the base `+0x5c` that fetches mid-instruction
/// garbage (the `0xdad2` "unknown opcode" wall), but at `+0x100` `0xdac4` is a
/// clean `entry a1,0x60` -- the real syscall handler. PROVEN: every pool
/// code-pointer into this block (`0xdac4`, `0xd900`=ISR, `0xd9f0`=sched-fn)
/// decodes as an `entry` prologue at `+0x100` and as mid-instruction at `+0x5c`;
/// the block decodes as a continuous run of clean `entry`/`retw.n` functions at
/// `+0x100`. Bounds by walk-and-stub: `wake_tasks_by_event_mask` (reachable
/// `+0x5c` code) ends at `0xd8a5`, and `FUN_0000dea0` (reachable `+0x5c`) resumes
/// at `0xdea8`; the only "code" the `+0x5c` descent found in between is the
/// mislabeled `FUN_0000dbc4` (really this section's `0xdac4`+`0x100`).
const SYSCALL_BLOCK_LO: u32 = 0x0000_d8a7;
const SYSCALL_BLOCK_HI: u32 = 0x0000_de04;

/// M2c iter20/iter22: the syscall-yield context-switch chain -- more `+0x100`
/// sections reached by walk-and-stub once the `0x2630` seam broke. Each is a
/// single function or literal pool, bounded `entry..retw.n`/`rfe` (code) or by
/// its live L32r targets (pools), and verified by COHERENT EXECUTION (the
/// strongest oracle available -- the PSP's real segment table is inaccessible).
/// The context-switch routine (`0x2630`) is dispatched by the syscall jump
/// table; it calls the IPC critical-section primitive (`0xc48c`) which posts to
/// `[0xfae0]` and jumps into Seg-B. Pools are separate `+0x100` rodata.
///
/// iter22: the region is far larger than the iter20 stub (`..0x26d3`) captured.
/// It flows straight through the ctx-switch routine into the symbol-map fn
/// `FUN_00002730` and continues as one contiguous `+0x100` block -- the full
/// syscall/exception context save-restore + dual-way TLB-swap handler (EPC1-7
/// save at `0x2914`, two `wdtlb` blocks at `0x2ad6`/`0x2b7a`) -- terminating at
/// the `rfe` at `0x2bf2`, after which file `0x2cf5..0x2d100` is the zero desert
/// before Seg-B. `+0x100` decodes coherently across the whole span (every L32r
/// target hits an embedded pool; `0x28ef` jumps to `EXC_RESTORE`=0xe1fc); base
/// `+0x5c` mid-instruction garbage-walls at `0x26d6`. Bound = one past the rfe.
const CTXSW_CALLEE_LO: u32 = 0x0000_2630;
const CTXSW_CALLEE_HI: u32 = 0x0000_2b51;
const CTXSW_CALLEE_POOL_LO: u32 = 0x0000_2540;
const CTXSW_CALLEE_POOL_HI: u32 = 0x0000_2560;
/// The non-windowed register-window transition helper called at the end of
/// `CTXSW_CALLEE`. The caller at 0x2a86 is itself in the +0x100 section, so its
/// PC-relative `call0 0xdf98` must fetch the linked helper from file 0xe098,
/// not the unrelated base-framed `callx8 a7` at file 0xdff4. The helper reads
/// WINDOWBASE/WINDOWSTART, spills live windows with `rotw`, and ends at the
/// `ret.n` at 0xe0af; file 0xe1b1 (VMA 0xe0b1) starts zero padding.
const CTXSW_WINDOW_ROTATE_LO: u32 = 0x0000_df98;
const CTXSW_WINDOW_ROTATE_HI: u32 = 0x0000_e0b1;
const IPC_PRIMITIVE_LO: u32 = 0x0000_c48c;
const IPC_PRIMITIVE_HI: u32 = 0x0000_c4d4;
const IPC_POOL_A_LO: u32 = 0x0000_3420;
const IPC_POOL_A_HI: u32 = 0x0000_3430;
const IPC_POOL_B_LO: u32 = 0x0000_3c70;
const IPC_POOL_B_HI: u32 = 0x0000_3c80;
/// iter20 cont.: the exception-frame RESTORE routine, `Jx`-ed to at VMA 0xe1fc
/// (file 0xe2fc) from the syscall-return path (0xb57). Reloads the register file
/// and ends in `RFE` (file 0xe42d = VMA 0xe32d) back to the restored EPC. +0x5c
/// is all zeros here (file 0xe200-0xe2fb), so it is unambiguously a +0x100 seam.
const EXC_RESTORE_LO: u32 = 0x0000_e1fc;
const EXC_RESTORE_HI: u32 = 0x0000_e334;

/// M2c iter24 (2026-07-09): a +0x100 text section reached by `Call8` from
/// `FUN_00005958` in the serviced-syscall return path (the wall that opened once
/// iter23's faithful exception vector let the boot's syscall complete). At base
/// `+0x5c`, VMA 0x93f0 decodes `L32i` then walls `Unknown 0xfc5d` at 0x93f3; at
/// `+0x100` it is a clean `entry a1,0x20` prologue -- a Call8 target MUST begin
/// with `entry`, so the base framing is disproven. The body is a mod-128 ring
/// scan (circular index wraps at 0x7f, head/tail fields at `+0x200`/`+0x204`),
/// semantic TBD. Ghidra's phantom `FUN_000094f0` (file 0x94f0 = VMA 0x93f0's
/// +0x100 image) decodes it coherently `entry..retw.n` (file 0x9560 = VMA
/// 0x9460) with branch-tail landing pads to file 0x956b; the next `entry` is at
/// file 0x9570 = VMA 0x9470. Bound = one function, [0x93f0, 0x9470).
const SYSRET_SCAN_LO: u32 = 0x0000_93f0;
const SYSRET_SCAN_HI: u32 = 0x0000_9470;

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
mod boot_tests;
