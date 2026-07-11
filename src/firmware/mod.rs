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
const CTXSW_CALLEE_HI: u32 = 0x0000_2bf5;
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

    #[test]
    fn ctxsw_call0_target_uses_matching_plus_100_section() {
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        let caller = 0x2a86;
        let caller_bytes: [u8; 8] =
            std::array::from_fn(|i| proc.bus.fetch8(caller + i as u32, caller + i as u32));
        let target = match decode::decode(&caller_bytes, caller).op {
            Op::Call0 { target } => target,
            op => panic!("expected context-switch Call0 at {caller:#x}, got {op:?}"),
        };
        assert_eq!(target, 0xdf98);

        let target_bytes: [u8; 8] =
            std::array::from_fn(|i| proc.bus.fetch8(target + i as u32, target + i as u32));
        assert!(
            matches!(decode::decode(&target_bytes, target).op, Op::Rsr { sr: 72, t: 2 }),
            "the +0x100 context-switch caller must reach its matching window-rotation helper, \
             not the base-framed bytes at {target:#x}: {:02x?}",
            &target_bytes[..3],
        );
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
        // Sample at END OF STEADY STATE: run until the boot reaches the syscall
        // context-switch entry (CTXSW_CALLEE_LO=0x2630) or walls. The invariant is
        // "low-window identity from reset THROUGH STEADY STATE" -- the context
        // switch is precisely where steady state ends: it reprograms processor/MMU
        // state (PS/EPC restore + address-space swap), so past it the low-window
        // identity mapping is legitimately no longer guaranteed. Before iter20's
        // +0x100 overlays, 0x2630 ran a mis-fetched `call0` and walled one instr
        // later at 0x44a34, so this loop happened to stop here anyway; now 0x2630
        // runs for real, so we stop explicitly at its entry (frontier-independent).
        for _ in 0..300_000 {
            if proc.cpu.pc == CTXSW_CALLEE_LO {
                break;
            }
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

        // iter18 (2026-07-09): EXCSAVE1-7 are now modeled (interp/mod.rs). The
        // general-exception handler (0x2958) stashes EXCCAUSE into EXCSAVE3 on
        // entry and reads it back at 0x2a66 to dispatch syscall-vs-interrupt.
        // Without the register backing that readback was 0, so EVERY syscall
        // mis-routed to the interrupt path: init's cooperative-yield syscall was
        // never serviced, the scheduler livelocked re-dispatching 0x588c, and the
        // unbounded re-dispatch recursion eventually spilled the stack into SCHED
        // (cur-task corruption to 0x9040 at ~n=58.7k). That recursion was ALSO the
        // only source of window exceptions in this boot -- which is why the old
        // iter17 guards (window_exceptions>0, instrs>48_215) held. With the
        // syscall now serviced correctly the recursion is gone: the boot advances
        // cleanly. The 0xdad2 "unmodeled opcode" that followed was NOT an opcode
        // at all -- it was our fetch offset: the syscall-dispatch block (the
        // Callx4 target 0xdac4 and its callees) is a THIRD +0x100 piecewise-
        // relocated section (SYSCALL_BLOCK_LO..HI overlay, iter19). With that
        // mapped, the syscall handler runs its syscall-number jump table and
        // dispatches (via PC-relative Call8) to the context-switch routine at
        // 0x2630 -- itself another +0x100 section (iter20 CTXSW_CALLEE overlay),
        // which calls the IPC primitive 0xc48c (IPC_PRIMITIVE overlay) that posts
        // to [0xfae0] and jumps into Seg-B, returns, and runs the exception-frame
        // restore (0xe1fc, iter20 EXC_RESTORE overlay + its scattered pools). With
        // that mapped the boot NO LONGER WALLS anywhere in the budget: it advances
        // into a steady loop inside the exception handler (FUN_0000e098, last_pc
        // ~0xe297) and spins there for all 200k instrs. So unknown_op is None and
        // instrs_executed == the budget.
        //
        // iter21 (2026-07-09): the "steady loop in the exception handler" that the
        // prior iter20 note described was NOT a real scheduler state -- it was a
        // FRAMING ARTIFACT (CTXSW_CALLEE overlay ended too short at 0x26aa; the
        // routine tail was misframed at base into a fake poison-wdtlb epilogue).
        // Fixed by extending the overlay; details in the iter22 note below.
        //
        // iter22 (2026-07-09): mapping the 0x26d3 seam resolved the WHOLE region.
        // 0x26d3 is +0x100 (the ctx-switch routine's straight-line fall-through),
        // and the +0x100 run is far larger than the iter20/iter21 stubs captured:
        // it flows through the ctx-switch routine into the symbol-map fn
        // FUN_00002730 and continues as ONE contiguous +0x100 block -- the full
        // syscall/exception context save-restore + dual-way TLB-swap handler --
        // terminating at the `rfe` at 0x2bf2 (then the zero desert to Seg-B). The
        // discriminator is unambiguous: every L32r target hits an embedded pool,
        // FUN_00002730 aligns exactly, 0x28ef jumps to EXC_RESTORE (0xe1fc), and
        // the block reads exccause (0x28c3) / dispatches syscall (bnei a3,1 at
        // 0x28dc) / ends in rfe. CTXSW_CALLEE_HI now covers the whole span
        // (0x2630..0x2bf5); a regression sends boot back to the 0x26d6 poison-tail
        // wall or the 0xe098 spin.
        //
        // iter23 (2026-07-09): the faithful exception-vector model let the boot's
        // user-mode `syscall` be serviced (via VECBASE+0x2e0 -> stub -> 0x28b4),
        // advancing boot ~600 instrs into new code that walled at 0x93f3
        // (FUN_000093f0+0x3, Unknown 0x0000fc5d) -- the next +0x100 seam.
        //
        // iter24 (2026-07-09): mapping the 0x93f0 seam (SYSRET_SCAN overlay --
        // FUN_000093f0 is a +0x100 ring-scan fn Call8'd in the syscall-return
        // path; base-framed it walled Unknown 0xfc5d, +0x100 it is a clean
        // `entry a1,0x20`) CLEARED the opcode wall entirely: `unknown_op` is now
        // None and the boot runs the full budget into a fully-coherent steady
        // state -- the real IPC/ISR scheduler loop. That loop (period ~137
        // instrs): the IPC critical-section primitive 0xc48c posts a message to
        // the [0xfae0] mailbox, cache-flushes it (Dhwbi 0xb0e710), then stores a
        // doorbell byte to 0x2500000a (FUN_00007ed0), takes the resulting
        // exception, saves context (0xe098), calls the ISR 0xd900, and loops.
        //
        // [SUPERSEDED by iter25 (fault-cycle broken) then iter42/iter44 -- kept as
        // the changelog of how the frontier moved.] At iter24 the terminal state was
        // a STORE_PROHIBITED (cause 29) fault-cycle on the 0x2500000a doorbell.
        // Root cause (m2c_probe_mmu_at_fault, XDNA_FW_FAULT_PC=0x7f22): the
        // firmware disabled the 0x20000000 identity DTLB entry (asid=0) and
        // switched to page-table autorefill (ptevaddr=0x3c000000); our model's
        // page table there is empty, so autorefill maps the doorbell page to
        // paddr 0 READ-only and the store re-faults forever, re-posting the same
        // message. That is the NEXT frontier (page-table / external-memory-map
        // modeling), tracked in the boot-wake finding -- a different KIND of wall
        // from the +0x100 seams.
        //
        // iter25 (2026-07-09): standing in for the PSP, the synth page table now
        // identity-maps the peripheral gap between the code region and the mailbox
        // (psp_map::install -> synthesize_identity_page_table, attr 2 RW device),
        // which contains the 0x2500000a doorbell. The store SUCCEEDS (routes to the
        // Bus RAM backing), breaking the fault-cycle: boot advances ~1800 instrs
        // into the task-context-restore path (FUN_00002730 reconstructs task
        // 0x10f10's saved context from [0x12048], loads its run-fn 0x2450 from
        // field [0x12064], and Callx8's to it via the trampoline FUN_0000df8c).
        //
        // iter42 (2026-07-10): 0x2a86 lives in the +0x100 CTXSW section, so its
        // PC-relative Call0 target 0xdf98 must use the matching +0x100 bytes too.
        // Those bytes are the non-windowed WINDOWBASE/WINDOWSTART rotation helper,
        // not the base-framed Callx8 a7 that produced the false 0x2450 wall. With
        // CTXSW_WINDOW_ROTATE mapped, the helper returns at n=49553, the incoming
        // context reaches RFE at n=49586 and task entry 0x08b041bc at n=49624.
        // The 200k observation now consumes its full budget in coherent Seg-B task
        // code (unknown_op=None), directly pinning that the 0x2450 wall stays gone.
        assert_eq!(
            report.instrs_executed, 200_000,
            "boot stopped before the full post-context-switch observation budget: {report:?}",
        );
        assert_eq!(
            report.unknown_op, None,
            "boot hit an opcode wall after the context-switch overlay fix: {report:?}",
        );
        assert_eq!(
            report.window_exceptions, 0,
            "boot took a window exception (count={}) -- regressed into the old recursion livelock \
             (unbounded 0x588c re-dispatch spilling the register window). last_pc={:#x}",
            report.window_exceptions, report.last_pc,
        );
        // iter44 (2026-07-10): the terminal state is now CHARACTERIZED and the
        // boot-to-idle mission is REACHED. This 200k run is the first task's coherent
        // scheduler idle-loop: it issues void syscall 0x6c and scans an EMPTY external
        // completion ring (this firmware idles by polling, not by `waiti`, so
        // reached_idle stays false by design). Full account:
        // docs/superpowers/findings/2026-07-10-boot-to-idle-reached.md. This guard now
        // pins that idle-loop coherence -- a reached_idle=true here would mean the
        // firmware took a `waiti` we do not expect, worth re-characterizing.
        assert!(
            !report.reached_idle,
            "boot now reports reached_idle (last_pc={:#x}); this firmware idles by \
             polling, not waiti -- re-characterize.",
            report.last_pc,
        );
    }

    /// Collapse-to-bit3 characterization (2026-07-08): the ONLY verified external
    /// stimulus is the per-column readiness bit3 at `[0xf9e0+col*0x60]` (gated by
    /// `FUN_00008c68`'s `Bbci a9,3` at `0x8c8b`). This test runs boot BOTH ways to
    /// a bounded n and contrasts them, establishing what bit3 alone unblocks -- with
    /// no descriptor/target/done-flag machinery (all deleted as misread TLB/IPC
    /// code; see the collapse-to-bit3 audit in the boot-wake finding).
    ///
    /// Arm A: natural boot (no agent). Arm B: bit3 agent. Records for each the
    /// waypoints reached (`task_dispatcher` 0xd7f0, the col-poll `0x8c88`,
    /// `goalive_runfn` real entry `0x588c`, publisher 0x50e8, idle `waiti` 0x56e6),
    /// the current-task path, and the final PC. The OBSERVED result (iter44): bit3 is
    /// NOT the boot-progress gate -- with the 0x2450 fix both arms are byte-identical
    /// (same final pc 0xb04252, same current-task path init 0x10f10 -> task 0x10dfc),
    /// settling in the SAME coherent idle-loop (the first task polling an empty
    /// external completion ring). bit3 changes nothing; go-alive/`waiti` never run
    /// because progress there is host-triggered. See
    /// docs/superpowers/findings/2026-07-10-boot-to-idle-reached.md.
    /// Override the horizon with `XDNA_FW_MAX` (default 400k).
    #[test]
    fn m2c_bit3_advances_boot_past_natural_wall() {
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");

        const DISPATCH: u32 = 0xd7f0; // task_dispatcher
        const COLSVC: u32 = 0x8c88; // FUN_00008c68's bit3 poll (real exec entry, not 0x8c68 symbol)
        const GOALIVE: u32 = 0x588c; // goalive_runfn real exec Entry (not the 0x55f8 symbol)
        const PUBLISH: u32 = 0x50e8; // publish_chann_info (go-alive)
        const WAITI: u32 = 0x56e6; // idle waiti 0
        const CUR_TASK: u32 = 0x2278; // scheduler current-task pointer
        let waypoints = [DISPATCH, COLSVC, GOALIVE, PUBLISH, WAITI];
        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(400_000);

        // One boot arm; returns (first-hit map, current-task transitions, final pc,
        // reached_idle, bit3 rising-edges).
        let run_arm = |enable_agent: bool| {
            let img = FirmwareImage::parse(&raw).expect("parse");
            let mut proc = FirmwareProcessor::load_m2c(img);
            if enable_agent {
                proc.enable_host_mailbox();
            }
            let mut first: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
            let mut cur_tasks: Vec<(u64, u32)> = Vec::new();
            let mut reached_idle = false;
            let mut n = 0u64;
            while n < max {
                let pc = proc.cpu.pc & 0x00ff_ffff;
                for &t in &waypoints {
                    if pc == t {
                        first.entry(t).or_insert(n);
                    }
                }
                let cur = proc.bus.data_load32(CUR_TASK);
                if cur_tasks.last().map(|&(_, t)| t) != Some(cur) {
                    cur_tasks.push((n, cur));
                }
                let step = proc.cpu.step(&mut proc.bus);
                proc.host_mailbox.tick(&mut proc.bus);
                match step {
                    Step::Ran | Step::Exception { .. } => n += 1,
                    Step::Wait(_) => {
                        reached_idle = true;
                        break;
                    }
                    Step::Unknown { .. } => break,
                }
            }
            let edges = proc.host_mailbox.column_stats().0;
            let final_pc = proc.cpu.pc & 0x00ff_ffff;
            (first, cur_tasks, final_pc, reached_idle, edges, proc.symbols)
        };

        let (nat_first, nat_cur, nat_pc, nat_idle, _nat_edges, nat_syms) = run_arm(false);
        let (b3_first, b3_cur, b3_pc, b3_idle, b3_edges, b3_syms) = run_arm(true);

        eprintln!("=== collapse-to-bit3: natural vs bit3 boot (max n=400k) ===");
        let report = |label: &str,
                      first: &std::collections::BTreeMap<u32, u64>,
                      cur: &[(u64, u32)],
                      pc: u32,
                      idle: bool,
                      syms: &HashMap<u32, String>| {
            eprintln!("-- {label} --");
            eprintln!("  final pc={pc:#x} {}  reached_idle={idle}", nearest_symbol(syms, pc));
            for &w in &waypoints {
                eprintln!("    waypoint {w:#07x} {:<22} first@ {:?}", nearest_symbol(syms, w), first.get(&w));
            }
            eprintln!("    current-task path: {cur:x?}");
        };
        report("NATURAL (no agent)", &nat_first, &nat_cur, nat_pc, nat_idle, &nat_syms);
        report("BIT3 agent", &b3_first, &b3_cur, b3_pc, b3_idle, &b3_syms);
        eprintln!("  bit3 rising-edges asserted = {b3_edges}");

        assert!(b3_edges > 0, "bit3 agent never asserted a readiness bit");
        // KEY FINDING (collapse-to-bit3, updated 2026-07-09 for the EXCSAVE fix):
        // bit3 is NOT the gate for boot progress. Once EXCSAVE1-7 are modeled
        // (interp/mod.rs), the general-exception handler routes init's
        // cooperative-yield SYSCALL to the service path instead of mis-routing it
        // to the interrupt path, so BOTH arms now service the syscall. The 0xdad2
        // wall that followed was our fetch offset, not an opcode: the syscall-
        // dispatch block is a +0x100 section (SYSCALL_BLOCK overlay, iter19). With
        // it mapped both arms run the syscall jump table, the context-switch
        // chain (0x2630 -> IPC primitive 0xc48c, iter20 overlays), and -- with the
        // exception-restore section (0xe1fc) also mapped -- NO LONGER WALL: both
        // advance into the SAME steady loop inside the exception handler
        // (FUN_0000e098, ~0xe297) and spin there to the horizon. LONG before the
        // go-alive / column-power path (goalive_runfn 0x588c) would run. So the
        // finding stands and is stronger: bit3 (and the whole ColumnPowerAgent) is
        // entirely DOWNSTREAM of a loop the boot enters identically either way; the
        // two arms are byte-identical into it. (That loop is NOT yet proven idle --
        // see m2c_boot_advances_into_c_runtime; this test only checks bit3 does not
        // change the frontier.)
        assert_eq!(
            nat_pc, b3_pc,
            "the bit3 agent changed the boot frontier -- it is not supposed to gate progress \
             (natural arm ends at {nat_pc:#x}, bit3 at {b3_pc:#x})"
        );
        // iter44 (2026-07-10): with the +0x100 seams and the 0x2450 fix all in
        // place, both arms boot cleanly into the SAME coherent idle-loop -- the first
        // task (0x10dfc) polling an empty external completion ring (see
        // m2c_boot_advances_into_c_runtime and the boot-to-idle-reached finding). The
        // bit3-is-downstream finding is clean and stronger: both arms are
        // byte-identical (nat_pc == b3_pc above), not walling on divergent frontiers,
        // and go-alive/bit3 never run because progress there is host-triggered. Here
        // we only assert bit3 does not move the frontier and the boot does not regress
        // into the 0x9040 corruption.
        // The pre-fix 0x9040 cur-task corruption (the livelock's stack-overflow spill into SCHED)
        // is gone: neither arm's current-task ever leaves the legit init task 0x10f10.
        for (label, cur) in [("natural", &nat_cur), ("bit3", &b3_cur)] {
            assert!(
                !cur.iter().any(|&(_, t)| t == 0x9040),
                "{label} boot regressed to the 0x9040 stack-overflow corruption -- the syscall \
                 livelock is back (cur-task path: {cur:x?})"
            );
        }
        // Neither takes a `waiti`: this firmware idles by polling (iter44), so a
        // reached_idle here would mean an unexpected waiti worth re-characterizing.
        assert!(!nat_idle && !b3_idle, "a boot arm reached waiti idle -- re-characterize");
    }

    /// TAIL-POLL characterization (iter43/iter44, 2026-07-10): after the iter42
    /// 0x2450 fix the boot no longer walls -- it advances into a coherent scheduler
    /// steady-loop on the first real task (0x10dfc), dispatcher-dominated, no
    /// fault/idle/go-alive. This probe characterizes that loop: over a tail window
    /// [lo,hi) it histograms every data-load EA (EXTERNAL HW-aperture reads
    /// >=0x2500_0000 vs internal RAM) WITH per-site pc+value, the syscall selector
    /// the task re-issues each period, and a dynamic instruction ring of the last K
    /// steps. iter44 conclusion: the task loops on void syscall 0x6c, whose kernel
    /// dispatch scans an EMPTY external completion ring (head/tail at 0x27200330/
    /// 0x2720032c = 0) driven by the configured active-set [0x272003b8]=0x8000 --
    /// the firmware is idling on an empty ring, waiting for host/array events we do
    /// not supply. Env: XDNA_FW_MAX (default 200000), XDNA_FW_WIN=lo:hi (default
    /// 80000:200000), XDNA_FW_RING (default 60). Ignored unless XDNA_FW_PROBE set.
    #[test]
    fn m2c_probe_tail_poll() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the tail-poll probe");
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
        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200_000);
        let (win_lo, win_hi) = std::env::var("XDNA_FW_WIN")
            .ok()
            .and_then(|s| {
                let (a, b) = s.split_once(':')?;
                Some((a.trim().parse::<u64>().ok()?, b.trim().parse::<u64>().ok()?))
            })
            .unwrap_or((80_000, 200_000));
        let ring_depth: usize = std::env::var("XDNA_FW_RING").ok().and_then(|s| s.parse().ok()).unwrap_or(60);

        const EXT: u32 = 0x2500_0000; // HW-aperture threshold (doorbell/mailbox/columns)
        let mut ext: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        // Per external read site: (pc, addr) -> (count, sample loaded value).
        let mut ext_sites: std::collections::BTreeMap<(u32, u32), (u64, u32)> =
            std::collections::BTreeMap::new();
        let mut int: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        let mut ring: std::collections::VecDeque<(u64, u32, Op)> = std::collections::VecDeque::new();
        // Syscall dispatcher (FUN_0000dab0): a4 = [a2] is the syscall selector the
        // task issued (the k/m/l/p = 0x6b/0x6d/0x6c/0x70 comparison tree), live at
        // 0xdae4; a2 is the on-stack syscall arg block. Pinned selector => the task
        // re-issues the same call forever (iter44: 0x6c, the empty-ring event scan);
        // cycling => it walks through calls.
        const STATE_PC: u32 = 0xdae4;
        let mut state_hist: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        let mut state_seq: Vec<(u64, u32)> = Vec::new();

        let mut n = 0u64;
        while n < max {
            let pc = proc.cpu.pc & 0x00ff_ffff;
            if pc == STATE_PC {
                let s = proc.cpu.regs.read_ar(4);
                let addr = proc.cpu.regs.read_ar(2); // state var address = [a2]
                *state_hist.entry(s).or_insert(0) += 1;
                if state_seq.last().map(|&(_, v)| v) != Some(s) {
                    state_seq.push((n, s));
                }
                if state_seq.len() <= 6 {
                    eprintln!("    [syscall@dae4] n={n} argblock={addr:#x} selector={s:#x}");
                }
            }
            let in_win = n >= win_lo && n < win_hi;
            if in_win {
                if let Ok(phys) =
                    proc.cpu.translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch)
                {
                    let b: [u8; 8] =
                        std::array::from_fn(|k| proc.bus.fetch8(proc.cpu.pc + k as u32, phys + k as u32));
                    let op = decode::decode(&b, proc.cpu.pc).op;
                    // Data-load EA = AR[s] + imm (L32r targets are absolute literals).
                    let ea = match op {
                        Op::L32i { s, imm, .. }
                        | Op::L32iN { s, imm, .. }
                        | Op::L8ui { s, imm, .. }
                        | Op::L16ui { s, imm, .. }
                        | Op::L16si { s, imm, .. } => Some(proc.cpu.regs.read_ar(s).wrapping_add(imm)),
                        _ => None,
                    };
                    if let Some(addr) = ea {
                        if addr >= EXT {
                            *ext.entry(addr).or_insert(0) += 1;
                            let e = ext_sites.entry((pc, addr)).or_insert((0, 0));
                            e.0 += 1;
                            e.1 = proc.bus.data_load32(addr); // value the fw is about to read
                        } else {
                            *int.entry(addr).or_insert(0) += 1;
                        }
                    }
                    ring.push_back((n, pc, op));
                    if ring.len() > ring_depth {
                        ring.pop_front();
                    }
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(reason) => {
                    eprintln!("  IDLE Wait({reason:?}) at n={n} pc={pc:#x}");
                    n += 1;
                    break;
                }
                Step::Unknown { pc: upc, word } => {
                    eprintln!("  Unknown at pc={upc:#x} word={word:#010x}");
                    break;
                }
            }
            proc.host_mailbox.tick(&mut proc.bus);
        }

        eprintln!("=== tail-poll probe (natural+agent, n={n}, win=[{win_lo},{win_hi})) ===");
        let dump = |label: &str, m: &std::collections::BTreeMap<u32, u64>| {
            let mut v: Vec<(u32, u64)> = m.iter().map(|(&a, &c)| (a, c)).collect();
            v.sort_by(|a, b| b.1.cmp(&a.1));
            let total: u64 = v.iter().map(|&(_, c)| c).sum();
            eprintln!("--- {label} load EAs (top 16, total {total}) ---");
            for (addr, cnt) in v.into_iter().take(16) {
                eprintln!("  {cnt:>8}  {addr:#010x}");
            }
        };
        dump("EXTERNAL (>=0x25000000)", &ext);
        eprintln!("--- EXTERNAL read SITES (pc, addr) -> count, sample value ---");
        let mut es: Vec<((u32, u32), (u64, u32))> = ext_sites.into_iter().collect();
        es.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        for ((pc, addr), (cnt, val)) in es.into_iter().take(20) {
            eprintln!(
                "  {cnt:>8}  pc={pc:#08x} {:<20} [{addr:#010x}] = {val:#010x}",
                nearest_symbol(&proc.symbols, pc)
            );
        }
        dump("INTERNAL", &int);
        eprintln!("--- FUN_0000dab0 state-code histogram ({} distinct) ---", state_hist.len());
        let mut sv: Vec<(u32, u64)> = state_hist.iter().map(|(&a, &c)| (a, c)).collect();
        sv.sort_by(|a, b| b.1.cmp(&a.1));
        for (val, cnt) in sv.into_iter().take(16) {
            eprintln!("  {cnt:>8}  state={val:#x}");
        }
        eprintln!("--- state transition sequence (first 40 distinct-value changes) ---");
        for (nn, v) in state_seq.iter().take(40) {
            eprintln!("  n={nn:>8} state={v:#x}");
        }
        eprintln!("--- dynamic instruction ring (last {} steps) ---", ring.len());
        for (nn, pc, op) in &ring {
            eprintln!("  n={nn:>8} pc={pc:#08x} {:<22} {op:x?}", nearest_symbol(&proc.symbols, *pc));
        }
    }

    /// task-0x10f10 identity probe (2026-07-09): WHAT is the current task the
    /// dispatcher spins on, and what is it blocked waiting for?  Boots to a
    /// pre-corruption steady state (default n=50000), dumps the task struct at
    /// `XDNA_FW_TASK` (default 0x10f10) word-by-word with pointer annotation, and
    /// the go-alive record 0x2320 for comparison. Then, over a full boot, counts how
    /// many times the task's run-fn (field[0]) actually EXECUTES, and records the
    /// task's state byte (+0x2c), block/wait field (+0x1b), done-flag (+0x30), and
    /// link (+0x38) transitions -- so we can see if the task ever runs, and what
    /// gates it. Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_task_struct() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the task-struct probe");
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
        let task: u32 = std::env::var("XDNA_FW_TASK")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0x10f10);
        let snap: u64 = std::env::var("XDNA_FW_START")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(50_000);
        let total: u64 = std::env::var("XDNA_FW_DUMP_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60_000);

        let mut n = 0u64;
        while n < snap {
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(_) | Step::Unknown { .. } => break,
            }
            proc.host_mailbox.tick(&mut proc.bus);
        }

        let known: [(u32, &str); 3] = [(task, "self"), (0x2320, "goalive-rec"), (0x2250, "SCHED")];
        let annotate = |v: u32| -> String {
            for (a, name) in &known {
                if v >= *a && v < *a + 0x60 {
                    return format!(" <- {name}+{:#x}", v - *a);
                }
            }
            // code region?
            if (0x400..0x30000).contains(&v) {
                return format!(" <- code? {}", nearest_symbol(&proc.symbols, v));
            }
            String::new()
        };
        eprintln!("=== task struct dump @ n={snap} (task {task:#x}) ===");
        for i in 0..0x18u32 {
            let a = task + i * 4;
            let v = proc.bus.data_load32(a);
            eprintln!("  [{a:#07x}] +{:#04x} = {v:#010x}{}", i * 4, annotate(v));
        }
        let runfn = proc.bus.data_load32(task) & 0x00ff_ffff;
        eprintln!("--- go-alive record 0x2320 (comparison) ---");
        for i in 0..6u32 {
            let a = 0x2320 + i * 4;
            let v = proc.bus.data_load32(a);
            eprintln!("  [{a:#07x}] +{:#04x} = {v:#010x}{}", i * 4, annotate(v));
        }

        // Continue to `total`, counting run-fn executions + field transitions.
        let mut runfn_hits = 0u64;
        let mut state_hist: Vec<(u64, u8)> = Vec::new();
        let mut done_hist: Vec<(u64, u32)> = Vec::new();
        while n < total {
            let pc = proc.cpu.pc & 0x00ff_ffff;
            if pc == runfn {
                runfn_hits += 1;
            }
            let st = proc.bus.data_load8(task + 0x2c);
            if state_hist.last().map(|&(_, s)| s) != Some(st) {
                state_hist.push((n, st));
            }
            let df = proc.bus.data_load32(task + 0x30);
            if done_hist.last().map(|&(_, v)| v) != Some(df) {
                done_hist.push((n, df));
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(_) | Step::Unknown { .. } => break,
            }
            proc.host_mailbox.tick(&mut proc.bus);
        }
        eprintln!("--- over boot to n={total} ---");
        eprintln!(
            "  run-fn {runfn:#x} ({}) executed {runfn_hits} times",
            nearest_symbol(&proc.symbols, runfn)
        );
        eprintln!("  state byte [+0x2c] transitions (n,val): {state_hist:x?}");
        eprintln!("  done-flag [+0x30] transitions (n,val): {done_hist:x?}");
    }

    /// goalive-SPIN characterization (2026-07-09, from scratch): what loop is boot
    /// actually stuck in, and what does it poll?  Runs to a steady-state `start` n,
    /// then records the exact repeating instruction cycle -- traces forward until the
    /// PC seen at `start` recurs at the same SP (one period) -- annotating every load
    /// with its effective address + value and every conditional branch with
    /// taken/not-taken.  Reports the cycle's PC span, its length, the distinct
    /// addresses it polls (with values), and the branch that decides the loop. No
    /// assumptions about "retire gate" or "worker 0x9040". `XDNA_FW_START` (default
    /// 200000) sets the steady-state entry; `XDNA_FW_PERIOD_CAP` (default 4000) caps
    /// the traced period. Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_goalive_spin() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the goalive-spin probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        proc.enable_host_mailbox(); // bit3 kernel: study the demoted-but-present state
        let start: u64 = std::env::var("XDNA_FW_START")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200_000);
        let cap: usize = std::env::var("XDNA_FW_PERIOD_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4000);

        // The recursion never repeats SP (it descends 144B/pass), so anchor on a
        // PC (the dispatcher entry by default) and detect PC-only recurrence.
        let anchor_pc: u32 = std::env::var("XDNA_FW_ANCHOR_PC")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0xd7f0);

        // Advance to steady state, then to the first anchor-PC hit at/after `start`.
        let mut n = 0u64;
        while n < start || (proc.cpu.pc & 0x00ff_ffff) != anchor_pc {
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(_) | Step::Unknown { .. } => break,
            }
            proc.host_mailbox.tick(&mut proc.bus);
            if n > start + 200_000 {
                break; // anchor PC not reached -- bail
            }
        }
        let anchor_n = n;
        let anchor_sp = proc.cpu.regs.read_ar(1);
        let mut lines: Vec<String> = Vec::new();
        let mut poll: std::collections::BTreeMap<u32, (u64, u32)> = std::collections::BTreeMap::new(); // addr->(count,last)
        let mut pc_hist: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        let mut steps = 0usize;
        loop {
            let pc = proc.cpu.pc & 0x00ff_ffff;
            let sp = proc.cpu.regs.read_ar(1);
            *pc_hist.entry(pc).or_insert(0) += 1;
            let mut note = String::new();
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch) {
                let b: [u8; 8] =
                    std::array::from_fn(|k| proc.bus.fetch8(proc.cpu.pc + k as u32, phys + k as u32));
                let d = decode::decode(&b, proc.cpu.pc);
                match d.op {
                    Op::L8ui { s, imm, .. } => {
                        let a = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                        let v = proc.bus.data_load8(a) as u32;
                        note = format!("  L8 [{a:#x}]={v:#x}");
                        let e = poll.entry(a).or_insert((0, 0));
                        e.0 += 1;
                        e.1 = v;
                    }
                    Op::L32i { s, imm, .. } | Op::L32iN { s, imm, .. } => {
                        let a = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                        let v = proc.bus.data_load32(a);
                        note = format!("  L32 [{a:#x}]={v:#x}");
                        let e = poll.entry(a).or_insert((0, 0));
                        e.0 += 1;
                        e.1 = v;
                    }
                    _ => {}
                }
                if steps < 400 {
                    lines.push(format!("  {pc:#08x} sp={sp:#x} {:<26}{note}", format!("{:?}", d.op)));
                }
            }
            if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
                break;
            }
            proc.host_mailbox.tick(&mut proc.bus);
            steps += 1;
            let now_pc = proc.cpu.pc & 0x00ff_ffff;
            if now_pc == anchor_pc && steps > 1 {
                break; // anchor PC recurred -- one full dispatcher cycle
            }
            if steps >= cap {
                lines.push(format!(
                    "  ... period exceeded cap={cap} (anchor PC {anchor_pc:#x} never recurred)"
                ));
                break;
            }
        }
        let end_sp = proc.cpu.regs.read_ar(1);

        eprintln!("=== goalive-spin: dispatcher cycle at n={anchor_n} (anchor pc={anchor_pc:#x}) ===");
        eprintln!(
            "anchor sp={anchor_sp:#x} -> end sp={end_sp:#x} (delta={} bytes)",
            anchor_sp as i64 - end_sp as i64
        );
        eprintln!("period length = {steps} steps; distinct PCs in period = {}", pc_hist.len());
        eprintln!("--- polled addresses in the period (addr -> count, last-val) ---");
        for (a, (c, v)) in &poll {
            eprintln!("  {a:#010x}  count={c:>4} last={v:#x}");
        }
        eprintln!("--- first {} instrs of the period ---", lines.len().min(400));
        for l in &lines {
            eprintln!("{l}");
        }
    }

    /// Waypoint-hit probe (2026-07-09, general). Counts hits + first-hit n for a
    /// set of PCs across a full natural boot. `XDNA_FW_WAYPOINTS` = comma-sep hex
    /// (default = the scheduler entry points: picker `0xc980` + its two callers
    /// `0x42c8`/`0xdd7a`, the command dispatcher `0xdbc4`, early-init `0x41b8`, the
    /// linker `0xd4e0`). Answers "does boot ever reach the picker / the early-init
    /// scheduler-start call". Ignored unless XDNA_FW_PROBE.
    #[test]
    fn m2c_probe_waypoint_hits() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the waypoint-hit probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let wps: Vec<u32> = std::env::var("XDNA_FW_WAYPOINTS")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok())
                    .collect()
            })
            .unwrap_or_else(|| vec![0xc980, 0x42c8, 0xdd7a, 0xdbc4, 0x41b8, 0xd4e0]);
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1_500_000);

        let mut first: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        let mut count: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        let mut n = 0u64;
        while n < max {
            let pc = proc.cpu.pc & 0x00ff_ffff;
            if wps.contains(&pc) {
                first.entry(pc).or_insert(n);
                *count.entry(pc).or_insert(0) += 1;
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(_) | Step::Unknown { .. } => break,
            }
        }
        eprintln!("=== waypoint hits (natural boot, n<{max}) ===");
        for w in &wps {
            match first.get(w) {
                Some(f) => eprintln!(
                    "  {w:#08x} ({:<26}) first-hit n={f} (x{})",
                    nearest_symbol(&proc.symbols, *w),
                    count[w]
                ),
                None => eprintln!("  {w:#08x} ({:<26}) NEVER", nearest_symbol(&proc.symbols, *w)),
            }
        }
    }

    /// Kernel-verification probe (2026-07-08, collapse-to-bit3): what does the
    /// firmware ACTUALLY read in the per-column status region during a natural
    /// (no-agent) boot?  Records every byte/half/word load whose address falls in
    /// `[0xf900, 0xfc00)` (covers 8 columns * 0x60 stride), with the issuing PC,
    /// first-seen n, hit count, and last value.  Establishes the verified kernel
    /// params -- base, stride, column count, and the polled bit -- WITHOUT trusting
    /// the deleted descriptor's `colmask=0xf`.  Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_colstatus_poll() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the col-status poll probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        // NATURAL boot by default; XDNA_FW_AGENT=1 enables the (old) host mailbox so
        // the poll is satisfied and boot advances -- reveals any columns polled only
        // AFTER the first wait phase passes.
        if std::env::var("XDNA_FW_AGENT").is_ok() {
            proc.enable_host_mailbox();
        }
        let stop: u64 = std::env::var("XDNA_FW_DUMP_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60_000);
        let lo: u32 = 0xf900;
        let hi: u32 = 0xfc00;

        // addr -> (first_n, count, last_val, width, issuing PCs set)
        let mut hits: std::collections::BTreeMap<u32, (u64, u64, u32, u8, std::collections::BTreeSet<u32>)> =
            std::collections::BTreeMap::new();
        let mut n = 0u64;
        while n < stop {
            let pc = proc.cpu.pc & 0x00ff_ffff;
            // Decode the load about to run; compute its effective address.
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch) {
                let b: [u8; 8] =
                    std::array::from_fn(|k| proc.bus.fetch8(proc.cpu.pc + k as u32, phys + k as u32));
                let d = decode::decode(&b, proc.cpu.pc);
                let load = match d.op {
                    Op::L8ui { s, imm, .. } => Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), 1u8)),
                    Op::L16ui { s, imm, .. } | Op::L16si { s, imm, .. } => {
                        Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), 2))
                    }
                    Op::L32i { s, imm, .. } | Op::L32iN { s, imm, .. } => {
                        Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), 4))
                    }
                    _ => None,
                };
                if let Some((addr, width)) = load {
                    if addr >= lo && addr < hi {
                        let val = match width {
                            1 => proc.bus.data_load8(addr) as u32,
                            2 => proc.bus.data_load32(addr) & 0xffff,
                            _ => proc.bus.data_load32(addr),
                        };
                        let e =
                            hits.entry(addr).or_insert((n, 0, 0, width, std::collections::BTreeSet::new()));
                        e.1 += 1;
                        e.2 = val;
                        e.4.insert(pc);
                    }
                }
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(_) | Step::Unknown { .. } => break,
            }
        }

        eprintln!("=== col-status region reads [{lo:#x},{hi:#x}) over natural boot (n={n}) ===");
        eprintln!("(addr, first-n, count, width, last-val, issuing-PCs)");
        let mut prev: Option<u32> = None;
        for (addr, (fn_, cnt, val, w, pcs)) in &hits {
            let stride = prev.map(|p| addr - p).unwrap_or(0);
            let pcstr: Vec<String> = pcs.iter().map(|p| format!("{p:#x}")).collect();
            eprintln!(
                "  {addr:#07x} (+{stride:#x} from prev)  firstN={fn_:>6} count={cnt:>5} w={w} last={val:#x}  pcs=[{}]",
                pcstr.join(",")
            );
            prev = Some(*addr);
        }
        eprintln!("--- distinct addresses: {} ---", hits.len());
    }

    /// Thread-1 control-flow probe: how does execution ENTER `0x8773` (the
    /// `Srli a6,a6,2` that produces the garbage mask `0x9268`)?  If the
    /// `Movi a6,-1` at `0x876d` immediately precedes it, a6 should be
    /// `0xffffffff` and the mask correct; the probe caught a6=`0x249a0` instead,
    /// so `0x876d` must have been skipped.  Dumps a ring of the last N executed
    /// (pc, op, a6) before the first arrival at a trigger PC (default `0x8773`,
    /// override XDNA_FW_TRIG).  Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_pc_history() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the pc-history probe");
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
        let stop: u64 = std::env::var("XDNA_FW_DUMP_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60_000);
        let trig: u32 = std::env::var("XDNA_FW_TRIG")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0x8773);
        let depth: usize = std::env::var("XDNA_FW_HIST").ok().and_then(|s| s.parse().ok()).unwrap_or(28);

        let mut ring: std::collections::VecDeque<(u64, u32, String, u32, u32)> =
            std::collections::VecDeque::with_capacity(depth + 1);
        let mut n = 0u64;
        let mut fired = false;
        while n < stop {
            let pc = proc.cpu.pc & 0x00ff_ffff;
            let op_str = if let Ok(phys) =
                proc.cpu.translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch)
            {
                let b: [u8; 8] =
                    std::array::from_fn(|k| proc.bus.fetch8(proc.cpu.pc + k as u32, phys + k as u32));
                format!("{:?}", decode::decode(&b, proc.cpu.pc).op)
            } else {
                "<xlate-fail>".to_string()
            };
            let a0 = proc.cpu.regs.read_ar(0);
            let a6 = proc.cpu.regs.read_ar(6);
            ring.push_back((n, pc, op_str, a0, a6));
            if ring.len() > depth {
                ring.pop_front();
            }
            if pc == trig {
                fired = true;
                eprintln!("=== pc-history: last {depth} instrs entering {trig:#x} (n={n}) ===");
                for (nn, p, op, a0, a6) in &ring {
                    eprintln!(
                        "  n={nn:>7} pc={p:#08x} ({:<26}) a0={a0:#x} a6={a6:#x}  {op}",
                        nearest_symbol(&proc.symbols, *p)
                    );
                }
                break;
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(_) | Step::Unknown { .. } => break,
            }
            proc.host_mailbox.tick(&mut proc.bus);
        }
        if !fired {
            eprintln!("trigger {trig:#x} never reached in n<{stop}");
        }
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

        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200_000);
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
        while n < max {
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

        // XDNA_FW_DISASM_OVL=lo:hi -- register a +0x100 fetch overlay over that
        // vaddr range before disassembling, to view an un-mapped region in its
        // piecewise-relocated (LMA=vaddr+0x100) framing. Lets a +0x100-seam
        // candidate be compared against its base framing without editing load_m2c.
        if let Ok(ovl) = std::env::var("XDNA_FW_DISASM_OVL") {
            let (a, b) = ovl.split_once(':').expect("XDNA_FW_DISASM_OVL must be lo:hi (hex)");
            let lo = u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).expect("lo hex");
            let hi = u32::from_str_radix(b.trim().trim_start_matches("0x"), 16).expect("hi hex");
            proc.bus.add_rom_overlay(lo, hi, LOW_VMA_FILE_OFFSET);
            eprintln!("(+0x100 overlay registered over {lo:#x}..{hi:#x})");
        }

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

    /// iter23 (2026-07-09): capture VECBASE + PS at the first `syscall` so the
    /// faithful exception-vector model knows where the CPU actually vectors and in
    /// which mode. Steps to the syscall pc (XDNA_FW_STOP_PC, default 0x8b043e1) and
    /// dumps vecbase/ps/epc1 + the vector bytes at candidate offsets. Self-skips
    /// unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_exc_vector_state() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the exc-vector-state probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        let stop_pc = std::env::var("XDNA_FW_STOP_PC")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0x08b0_43e1);
        let mut vecbase_writes: Vec<(u64, u32)> = Vec::new();
        let mut last_vb = proc.cpu.vecbase;
        let mut n = 0u64;
        while n < 5_000_000 {
            if proc.cpu.pc == stop_pc {
                break;
            }
            if proc.cpu.vecbase != last_vb {
                vecbase_writes.push((n, proc.cpu.vecbase));
                last_vb = proc.cpu.vecbase;
            }
            match proc.cpu.step(&mut proc.bus) {
                crate::firmware::xtensa::interp::Step::Unknown { .. }
                | crate::firmware::xtensa::interp::Step::Wait(_) => break,
                _ => {}
            }
            n += 1;
        }
        eprintln!("=== M2c exc-vector state @ pc={:#x} (n={n}) ===", proc.cpu.pc);
        eprintln!("vecbase        = {:#x}", proc.cpu.vecbase);
        eprintln!("ps             = {:#010x}", proc.cpu.regs.ps);
        eprintln!(
            "  EXCM={} INTLEVEL={} WOE={} UM(bit5)={}",
            proc.cpu.regs.ps & 0x10 != 0,
            proc.cpu.regs.ps & 0xF,
            proc.cpu.regs.ps & (1 << 18) != 0,
            proc.cpu.regs.ps & 0x20 != 0
        );
        eprintln!("epc1           = {:#x}", proc.cpu.epc1);
        eprintln!("vecbase writes = {:x?}", vecbase_writes);
        // Dump candidate general-exception vector offsets (bytes, both framings).
        for off in [0x300u32, 0x340, 0x2c0, 0x280, 0x31c] {
            let v = proc.cpu.vecbase.wrapping_add(off);
            let base: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(v + k as u32, v + k as u32));
            let hex: String = base.iter().map(|x| format!("{x:02x}")).collect();
            eprintln!("  vecbase+{off:#05x}={v:#06x}: {hex}");
        }
    }

    /// M2c GOLD LISTING (2026-07-09): overlay-correct recursive-descent
    /// disassembly of the whole reachable firmware, on OUR ground-truth decoder.
    ///
    /// Motivation: linear-sweep disasm (`m2c_probe_disasm_range` from 0x0)
    /// desyncs the instant it hits a literal pool or the signed header (vaddr <
    /// 0x1a4 is the `$PS1` header, NOT code), producing garbage that masks real
    /// findings. Ghidra's recursive descent is desync-free but (a) misses
    /// indirect/vtable-reached code -- exactly the scheduler core (picker
    /// 0xc980, idle 0xc8e0, FUN_000041b8/dbc4, go-alive 0x55f8) -- and (b) uses
    /// an Xtensa module that mis-decodes this image's FLIX bundles.
    ///
    /// This walks control flow from every known entry (the full symbol map +
    /// the reset vector + the VECBASE=0x800 vector stubs + the indirect
    /// scheduler targets we recovered by hand), decoding through `bus.fetch8`
    /// (so the `+0x5c` base / `+0x100` low-VMA / Seg-B overlays are ALL applied
    /// correctly and identically to execution). Every visited instruction is a
    /// real instruction boundary, so there is no desync. Unvisited bytes are
    /// data (literal pools / padding / .bss) and are marked as gaps, not forced
    /// into instructions.
    ///
    /// THE GATE (the direct answer to "is the overlay faulty anywhere"): any
    /// reachable instruction that decodes to `Op::Unknown` is either an
    /// overlay/mapping fault (wrong bytes at that vaddr) or a genuinely
    /// undecodable op. The probe reports every such site with how it was reached
    /// (seed / fall-through / branch-target) so fall-through Unknowns -- the ones
    /// that signal a real mid-function mapping problem -- stand out from
    /// branch-target ones (which may just be a mis-followed indirect-ish edge).
    /// It also cross-checks inst-fetch vs data-load bytes over the visited set
    /// (a Harvard/overlay split would diverge). Writes the full listing to
    /// `build/experiments/firmware-re/gold-listing.txt`. Extra comma-separated
    /// hex seeds via `XDNA_FW_GOLD_SEEDS`. Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_gold_disasm() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the gold-listing disassembler");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        use std::collections::{BTreeMap, HashSet};
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        // Code regions (vaddr) that may hold instructions: the low .text
        // (base-mapped, plus the two low-VMA overlays) and the PSP-relocated
        // Segment B runtime code-tail at 0x08b00000. Everything else (device
        // apertures, .bss zero-desert) is not code; descent never leaves these.
        // Low .text/.rodata ends ~0xe7xx (largest fn tops ~0xc1e5, code pointers
        // to ~0xe594); file 0x10000..0x2d100 is the entropy-scan zero desert
        // (.bss/pad), so the low region stops at 0x10000 -- a branch computed
        // into the desert is data, not code. Seg-B is the relocated runtime tail.
        let regions: [(u32, u32); 2] = [(0x1a4, 0x0001_0000), (0x08b0_0000, 0x08b0_fa10)];
        let in_code = |a: u32| regions.iter().any(|&(lo, hi)| a >= lo && a < hi);

        // Extract every statically-known control-flow target the descent should
        // follow (direct call/branch/jump/loop). Register-indirect callx*/jx and
        // Syscall have no static target. Returns the list of destination vaddrs.
        fn cf_targets(op: &decode::Op) -> Vec<u32> {
            use decode::Op::*;
            match *op {
                Call0 { target, .. }
                | Call4 { target, .. }
                | Call8 { target, .. }
                | Call12 { target, .. }
                | J { target }
                | Beqz { target, .. }
                | Bnez { target, .. }
                | Bltz { target, .. }
                | Bgez { target, .. }
                | BeqzN { target, .. }
                | BnezN { target, .. }
                | Beq { target, .. }
                | Bne { target, .. }
                | Blt { target, .. }
                | Bltu { target, .. }
                | Bge { target, .. }
                | Bgeu { target, .. }
                | Beqi { target, .. }
                | Bnei { target, .. }
                | Blti { target, .. }
                | Bgei { target, .. }
                | Bltui { target, .. }
                | Bgeui { target, .. }
                | Bbci { target, .. }
                | Bbsi { target, .. }
                | Bbc { target, .. }
                | Bbs { target, .. }
                | Bnone { target, .. }
                | Bany { target, .. }
                | Ball { target, .. }
                | Bnall { target, .. } => vec![target],
                Loop { end, .. } | Loopnez { end, .. } => vec![end],
                _ => vec![],
            }
        }
        // Does this instruction end the fall-through run? (control leaves; the
        // next byte is NOT guaranteed to be an instruction). Unconditional J and
        // indirect Jx transfer away; the ret/rf* family returns; Unknown is a
        // wall. Conditional branches, calls (callee returns), Syscall, and
        // Loop all CONTINUE the fall-through.
        fn is_terminator(op: &decode::Op) -> bool {
            use decode::Op::*;
            matches!(op, J { .. } | Jx { .. } | Retw | RetwN | RetN | Rfe | Rfwo | Rfwu | Unknown { .. })
        }

        // How a visited PC was first reached -- to triage the gate.
        #[derive(Clone, Copy, PartialEq)]
        enum Reach {
            Seed,
            Fall,
            Branch,
        }

        // Seeds. XDNA_FW_GOLD_RESETONLY=1 seeds ONLY the reset/vector/verified
        // entries (below), NOT the Ghidra symbol map -- so every Unknown is
        // reached by pure control-flow closure from a known-good boundary. Any
        // Unknown in THAT run is a real overlay/decoder concern, not a
        // misaligned-symbol-seed artifact (Ghidra FUN_ labels are sometimes a
        // byte or two off a real instruction boundary, which seeds garbage).
        let reset_only = std::env::var("XDNA_FW_GOLD_RESETONLY").is_ok();
        let mut work: Vec<(u32, Reach)> = Vec::new();
        let mut skipped_syms = 0u64;
        if !reset_only {
            // Ghidra `FUN_` labels are sometimes a byte or two off a real
            // instruction boundary; seeding descent there decodes garbage.
            // Validate each symbol seed: decode a short run from it and skip it
            // if an Unknown appears within the first few instructions (a
            // misaligned label garbles almost immediately, a real function start
            // does not).
            for (&a, _) in proc.symbols.iter() {
                if !in_code(a) {
                    continue;
                }
                let mut p = a;
                let mut aligned = true;
                for _ in 0..4 {
                    let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(p + k as u32, p + k as u32));
                    let d = decode::decode(&b, p);
                    if matches!(d.op, decode::Op::Unknown { .. }) {
                        aligned = false;
                        break;
                    }
                    p += (d.len as u32).max(1);
                }
                if aligned {
                    work.push((a, Reach::Seed));
                } else {
                    skipped_syms += 1;
                }
            }
        }
        // Reset vector, the general-exception vector stub (VECBASE=0x800 +
        // 0x2e0 = 0xae0, live-confirmed in exception-dispatch-pc-verdict.md),
        // the whole VECBASE stub table start, and the indirect scheduler core.
        for &s in &[
            0x1a4u32, 0xae0, 0x800, 0xc980, 0xc8e0, 0x41b8, 0xdbc4, 0x55f8, 0xd4e0, 0xd664, 0xd7f0, 0x588c,
            0x50e8, 0x56e6, 0xd84c, 0x2958, 0x28b4,
        ] {
            work.push((s, Reach::Seed));
        }
        if let Ok(extra) = std::env::var("XDNA_FW_GOLD_SEEDS") {
            for t in extra.split(',') {
                if let Ok(a) = u32::from_str_radix(t.trim().trim_start_matches("0x"), 16) {
                    work.push((a, Reach::Seed));
                }
            }
        }

        // Two-pass recursive descent with literal-pool awareness. Xtensa
        // compilers place 4-byte literal words INLINE in the code stream
        // (`l32r` loads from them). Pass 0 collects every `l32r` target as a
        // data word; pass 1 re-descends treating each such word as a hard data
        // boundary, so descent never falls/branches into a pool and mis-decodes
        // it. (Those pool-walks were the bulk of the residual Unknowns; xtdis
        // confirms the bytes are data -- libisa can't decode them either.)
        let seeds = work;
        let mut decoded: BTreeMap<u32, (decode::Op, u8, [u8; 8])> = BTreeMap::new();
        let mut reach: BTreeMap<u32, Reach> = BTreeMap::new();
        let mut data_bytes: HashSet<u32> = HashSet::new();
        for _pass in 0..2 {
            decoded.clear();
            reach.clear();
            let mut visited: HashSet<u32> = HashSet::new();
            let mut work = seeds.clone();
            while let Some((entry, how)) = work.pop() {
                if !in_code(entry) || data_bytes.contains(&entry) {
                    continue;
                }
                let mut pc = entry;
                let mut first = true;
                loop {
                    if !in_code(pc) || visited.contains(&pc) || data_bytes.contains(&pc) {
                        break;
                    }
                    let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, pc + k as u32));
                    let d = decode::decode(&b, pc);
                    visited.insert(pc);
                    reach.insert(pc, if first { how } else { Reach::Fall });
                    decoded.insert(pc, (d.op.clone(), d.len, b));
                    if let decode::Op::L32r { target, .. } = d.op {
                        for k in 0..4 {
                            data_bytes.insert(target.wrapping_add(k));
                        }
                    }
                    for t in cf_targets(&d.op) {
                        if in_code(t) && !visited.contains(&t) && !data_bytes.contains(&t) {
                            work.push((t, Reach::Branch));
                        }
                    }
                    if is_terminator(&d.op) {
                        break;
                    }
                    pc += (d.len as u32).max(1);
                    first = false;
                }
            }
        }
        let visited: HashSet<u32> = decoded.keys().copied().collect();

        // Emit the listing + gaps.
        let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("build/experiments/firmware-re/gold-listing.txt");
        let mut out = String::new();
        out.push_str("=== M2c GOLD LISTING: overlay-correct recursive-descent (our decoder) ===\n");
        let mut unknowns: Vec<(u32, Reach, u32)> = Vec::new();
        let mut prev_end: Option<u32> = None;
        for (&pc, (op, len, b)) in decoded.iter() {
            if let Some(pe) = prev_end {
                if pc > pe && in_code(pe) {
                    // Only note gaps inside a single region (skip the big
                    // low-text -> Seg-B jump).
                    if regions.iter().any(|&(lo, hi)| pe >= lo && pc <= hi) {
                        out.push_str(&format!(
                            "  {:#08x} .. {:#08x}  [gap {} bytes -- data/pool/pad]\n",
                            pe,
                            pc,
                            pc - pe
                        ));
                    }
                }
            }
            let sym = nearest_symbol(&proc.symbols, pc);
            let raw_hex: String =
                b[..(*len as usize).max(1).min(8)].iter().map(|x| format!("{x:02x}")).collect();
            let r = reach[&pc];
            let rc = match r {
                Reach::Seed => 'S',
                Reach::Fall => '.',
                Reach::Branch => 'b',
            };
            out.push_str(&format!("{rc} {pc:#08x} {sym:<26} {:<42} [{raw_hex}]\n", format!("{:?}", op)));
            if let decode::Op::Unknown { word } = op {
                unknowns.push((pc, r, *word));
            }
            prev_end = Some(pc + (*len as u32).max(1));
        }
        std::fs::write(&out_path, &out).expect("write gold listing");

        // Harvard/overlay split check: over the visited set, does inst-fetch
        // agree with data-load byte-for-byte? (A split would diverge.)
        let mut harvard_mismatch = 0u64;
        for &pc in visited.iter() {
            let fetched = proc.bus.fetch8(pc, pc);
            let loaded = proc.cpu.data_read8(&mut proc.bus, pc).unwrap_or(fetched);
            if fetched != loaded {
                harvard_mismatch += 1;
            }
        }

        // Report.
        eprintln!("=== GOLD LISTING SUMMARY ===");
        eprintln!("listing written to {}", out_path.display());
        eprintln!("instructions decoded (reachable) = {}", decoded.len());
        eprintln!("literal-pool data words marked   = {}", data_bytes.len() / 4);
        eprintln!("misaligned symbol seeds skipped  = {skipped_syms}");
        eprintln!("--- THE GATE: Unknown-on-reachable-code = {} ---", unknowns.len());
        let fall_unknowns = unknowns.iter().filter(|(_, r, _)| *r == Reach::Fall).count();
        eprintln!("  of which fall-through (real mid-function mapping suspects) = {}", fall_unknowns);
        for (pc, r, word) in unknowns.iter().take(40) {
            let rc = match r {
                Reach::Seed => "seed",
                Reach::Fall => "FALL",
                Reach::Branch => "bra",
            };
            eprintln!("    {pc:#08x} [{rc}] word={word:#010x}  ({})", nearest_symbol(&proc.symbols, *pc));
        }
        if unknowns.len() > 40 {
            eprintln!("    ... ({} more)", unknowns.len() - 40);
        }
        eprintln!(
            "--- Harvard/overlay split (inst-fetch vs data-load) mismatches = {} ---",
            harvard_mismatch
        );
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
        const GEN_EXC: u32 = 0x28b4; // general-exception handler entry (iter23; via vector 0xae0)
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
                    | decode::Op::S32c1i { t, s, imm }
                    | decode::Op::S32e { t, s, imm } => {
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
                    | decode::Op::S32c1i { t, s, imm }
                    | decode::Op::S32e { t, s, imm } => {
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

    /// M2c MMU-AT-FAULT probe: boot to the steady-state exception livelock, then
    /// introspect why the load of the current-task pointer `0x2278` faults with
    /// cause 28 (LOAD_PROHIBITED). Stops at the faulting `L32iN` (pc=0xe2a9,
    /// a2=0x2278) in the steady loop and dumps the DTLB resolution (hit way/attr
    /// or miss), PTEVADDR/DTLBCFG/RASID, the way-6 low-window identity entries,
    /// and the full autorefill translate result -- pinning whether the fault is a
    /// resident no-read entry or a page-walk to a no-read PTE. Ignored unless
    /// XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_mmu_at_fault() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the mmu-at-fault probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        use crate::firmware::xtensa::mmu::attr_to_access;
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        let max: u64 = std::env::var("XDNA_FW_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(60_000);
        // Default = the legacy 0x2278 load fault (pc 0xe2a9). Override to inspect a
        // different fault site: XDNA_FW_FAULT_PC=<hex> stop pc, XDNA_FW_FAULT_ADDR=<hex>
        // the DTLB vaddr to introspect (e.g. the 0x2500000a doorbell store at 0x7f22).
        let hexenv = |k: &str, d: u32| {
            std::env::var(k)
                .ok()
                .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
                .unwrap_or(d)
        };
        let stop_pc = hexenv("XDNA_FW_FAULT_PC", 0xe2a9);
        let fault_addr = hexenv("XDNA_FW_FAULT_ADDR", 0x2278);

        let mut n = 0u64;
        let mut found = false;
        while n < max {
            if proc.cpu.pc == stop_pc {
                let m = &proc.cpu.mmu;
                eprintln!("=== MMU at fault ({fault_addr:#x}) n={n} ===");
                eprintln!(
                    "  ptevaddr={:#x} dtlbcfg={:#x} itlbcfg={:#x} rasid={:#x}",
                    m.ptevaddr, m.dtlbcfg, m.itlbcfg, m.rasid
                );
                match m.lookup(fault_addr, true) {
                    Ok(hit) => {
                        let e = m.dtlb[hit.wi][hit.ei];
                        eprintln!(
                            "  lookup: HIT way={} ei={} ring={} vaddr={:#x} paddr={:#x} attr={} access={:#x}",
                            hit.wi,
                            hit.ei,
                            hit.ring,
                            e.vaddr,
                            e.paddr,
                            e.attr,
                            attr_to_access(e.attr)
                        );
                    }
                    Err(c) => eprintln!("  lookup: MISS/err cause={c}"),
                }
                eprintln!(
                    "  dtlbcfg={:#x} (way5 psz={}, way6 psz={})",
                    proc.cpu.mmu.dtlbcfg,
                    (proc.cpu.mmu.dtlbcfg >> 20) & 1,
                    (proc.cpu.mmu.dtlbcfg >> 24) & 1
                );
                for wi in [4usize, 5, 6] {
                    for ei in 0..4usize {
                        let e = proc.cpu.mmu.dtlb[wi][ei];
                        if e.asid == 0 && e.vaddr == 0 && e.paddr == 0 {
                            continue;
                        }
                        eprintln!(
                            "  dtlb[{wi}][{ei}] vaddr={:#x} paddr={:#x} asid={} attr={} var={}",
                            e.vaddr, e.paddr, e.asid, e.attr, e.variable
                        );
                    }
                }
                let t = proc.cpu.mmu.translate(&mut proc.bus, fault_addr, 0, 0);
                eprintln!("  translate(load, with autorefill) = {t:?}");
                found = true;
                break;
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                other => {
                    eprintln!("stopped early: {other:?} at n={n} pc={:#x}", proc.cpu.pc);
                    break;
                }
            }
        }
        if !found {
            eprintln!("did not reach the 0xe2a9 fault within {max} instrs (n={n})");
        }
    }

    /// M2c iter24 DISCRIMINATOR (2026-07-09, Maya "discriminate first"): the boot's
    /// terminal state is a STORE_PROHIBITED fault-cycle on the 0x2500000a doorbell,
    /// because the firmware disabled the 0x20000000 identity DTLB entry and switched
    /// to page-table autorefill at ptevaddr=0x3c000000 -- which our model reads as
    /// empty. This probe answers WHOSE gap that is: over a full boot it (a) counts
    /// every store whose EA lands in the page-table region [0x3c000000,0x40000000)
    /// and (b) checks, non-perturbingly (mmu.lookup, no autorefill), whether the
    /// doorbell vaddr ever becomes a RESIDENT WRITABLE mapping. Writes found or the
    /// doorbell going writable => the firmware/our-load owns it (emulator gap to
    /// fix). Neither ever => the PT at 0x3c000000 is populated by the PSP/an external
    /// agent we don't model. Env: XDNA_FW_MAX (default 400_000),
    /// XDNA_FW_DOORBELL (hex, default 0x2500000a). Ignored unless XDNA_FW_PROBE set.
    #[test]
    fn m2c_probe_pt_writes() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the pt-writes discriminator");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        use crate::firmware::xtensa::mmu::{access_granted, attr_to_access};
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        let max: u64 = std::env::var("XDNA_FW_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(400_000);
        let doorbell = std::env::var("XDNA_FW_DOORBELL")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .unwrap_or(0x2500_000a);
        const PT_LO: u32 = 0x3c00_0000;
        const PT_HI: u32 = 0x4000_0000;

        let mut pt_stores = 0u64;
        let mut first_pt: Option<(u64, u32, u32, u32)> = None; // n, pc, ea, val
        let mut first_writable: Option<(u64, u32, u8, u8)> = None; // n, pc, way, attr
                                                                   // Transitions of "doorbell resident-writable": (n, pc, op, now_writable).
        let mut transitions: Vec<(u64, u32, String, bool)> = Vec::new();
        let mut prev_writable = false;
        // Watch dtlb[5][0].asid (the firmware's 0x20000000 128MB RWX install) go 1->0:
        // (n, pc-that-ran, op, old_asid, new_asid). Only invalidate_tlb can zero it.
        let mut way5_events: Vec<(u64, u32, String, u8, u8)> = Vec::new();
        let mut prev_asid5 = proc.cpu.mmu.dtlb[5][0].asid;
        let mut n = 0u64;
        while n < max {
            let pc = proc.cpu.pc;
            // (a) resident-writable check for the doorbell (non-perturbing).
            let cur_writable = match proc.cpu.mmu.lookup(doorbell, true) {
                Ok(hit) => {
                    let e = proc.cpu.mmu.dtlb[hit.wi][hit.ei];
                    let w = access_granted(attr_to_access(e.attr), 1);
                    if w && first_writable.is_none() {
                        first_writable = Some((n, pc & 0x00ff_ffff, hit.wi as u8, e.attr));
                    }
                    w
                }
                Err(_) => false,
            };
            if cur_writable != prev_writable {
                // The instruction at the PREVIOUS step caused the transition; capture
                // the current pc's op as the site we land on / are about to run.
                let op = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                    Ok(phys) => {
                        let b: [u8; 8] =
                            std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                        format!("{:?}", decode::decode(&b, pc).op)
                    }
                    Err(_) => "<fault>".to_string(),
                };
                transitions.push((n, pc & 0x00ff_ffff, op, cur_writable));
                prev_writable = cur_writable;
            }
            // (b) count stores into the page-table region.
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                let d = decode::decode(&b, pc);
                let store = match d.op {
                    Op::S32i { s, t, imm } | Op::S8i { s, t, imm } | Op::S16i { s, t, imm } => {
                        Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), proc.cpu.regs.read_ar(t)))
                    }
                    Op::S32iN { s, t, imm } => {
                        Some((proc.cpu.regs.read_ar(s).wrapping_add(imm), proc.cpu.regs.read_ar(t)))
                    }
                    _ => None,
                };
                if let Some((ea, val)) = store {
                    if (PT_LO..PT_HI).contains(&ea) {
                        pt_stores += 1;
                        if first_pt.is_none() {
                            first_pt = Some((n, pc & 0x00ff_ffff, ea, val));
                        }
                    }
                }
            }
            // Decode the op about to run so we can attribute a way-5 asid change to it.
            let op_here = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                Ok(phys) => {
                    let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                    format!("{:?}", decode::decode(&b, pc).op)
                }
                Err(_) => "<fault>".to_string(),
            };
            if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
                eprintln!("halted at n={n} pc={:#x}", proc.cpu.pc);
                break;
            }
            let asid5 = proc.cpu.mmu.dtlb[5][0].asid;
            if asid5 != prev_asid5 {
                way5_events.push((n, pc & 0x00ff_ffff, op_here, prev_asid5, asid5));
                prev_asid5 = asid5;
            }
            n += 1;
        }
        eprintln!("=== pt-writes discriminator (max {max}, doorbell {doorbell:#x}) ===");
        eprintln!("  ran n={n}");
        eprintln!("  dtlb[5][0].asid transitions (n, pc-that-ran, op, old->new):");
        for e in &way5_events {
            eprintln!("    n={:>8} pc={:#08x} {} : {}->{}", e.0, e.1, e.2, e.3, e.4);
        }
        eprintln!("  page-table-region stores [{PT_LO:#x},{PT_HI:#x}) = {pt_stores}");
        eprintln!("    first = {first_pt:x?}");
        eprintln!("  doorbell first resident-writable = {first_writable:x?}");
        eprintln!("  writable transitions (n, pc, op-landed-on, now_writable):");
        for t in &transitions {
            eprintln!("    n={:>8} pc={:#08x} now_writable={} {}", t.0, t.1, t.3, t.2);
        }
        // Also dump the final resident TLB entry that covers the doorbell, if any.
        match proc.cpu.mmu.lookup(doorbell, true) {
            Ok(hit) => {
                let e = proc.cpu.mmu.dtlb[hit.wi][hit.ei];
                eprintln!(
                    "  final lookup({doorbell:#x}): HIT way={} vaddr={:#x} paddr={:#x} asid={} attr={} access={:#x}",
                    hit.wi, e.vaddr, e.paddr, e.asid, e.attr, attr_to_access(e.attr)
                );
            }
            Err(c) => eprintln!("  final lookup({doorbell:#x}): MISS/err cause={c}"),
        }
    }

    /// M2c POISON-TLB-TRIGGER probe: find the exact instant the low-window no-access
    /// TLB entry (a way 0-3 slot with attr>=12) first appears, and what installs it.
    /// Logs every way-6 idtlb/wdtlb (pc + operands) and every `dtlb[6][0].asid`
    /// liveness transition, and dumps each ctx-switch-tail instruction. Result (iter21):
    /// it is NOT autorefill and the region entry is never invalidated -- the firmware
    /// itself runs `wdtlb a5(=0xdeadbeef sentinel), a7(=0x2250)` at pc=0x26ac because a
    /// task field `[0x121d0]` still holds its create-time uninitialized sentinel.
    /// Ignored unless XDNA_FW_PROBE is set.
    #[test]
    fn m2c_probe_autorefill_trigger() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the autorefill-trigger probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        use crate::firmware::xtensa::mmu::MAX_TLB_WAY_SIZE;
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        let max: u64 = std::env::var("XDNA_FW_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(60_000);

        let find_poison = |mmu: &crate::firmware::xtensa::mmu::Mmu| -> Option<(usize, usize)> {
            for wi in 0..4usize {
                for ei in 0..MAX_TLB_WAY_SIZE {
                    let e = mmu.dtlb[wi][ei];
                    if e.asid != 0 && e.attr >= 12 {
                        return Some((wi, ei));
                    }
                }
            }
            None
        };

        let mut way6_ops: Vec<String> = Vec::new();
        let mut region_transitions: Vec<String> = Vec::new();
        let mut prev_region_live = proc.cpu.mmu.dtlb[6][0].asid != 0;
        let mut prev_pc = proc.cpu.pc;
        let mut n = 0u64;
        let mut found = false;
        while n < max {
            let pc = proc.cpu.pc;
            // Log way-6 D-side TLB management ops (peek before step; they only read ARs).
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                let op = decode::decode(&b, pc).op;
                if (0x2630..0x2700).contains(&pc) {
                    // Decode-agnostic dump of each ctx-switch instruction; at the
                    // task-struct load (0x2647) also resolve mem[a6] to reconcile the
                    // store-watch (0x12048) vs the observed read (0xdeadbeef).
                    let a6 = proc.cpu.regs.read_ar(6);
                    let mem_a6 = proc
                        .cpu
                        .mmu
                        .translate(&mut proc.bus, a6, 0, 0)
                        .map(|t| proc.bus.data_load32(t.paddr));
                    eprintln!(
                        "  [tail] pc={pc:#x} op={op:?} a5={:#x} a6={a6:#x} mem[a6]={:?}",
                        proc.cpu.regs.read_ar(5),
                        mem_a6
                    );
                }
                match op {
                    Op::Wdtlb { s, .. } => {
                        let as_ = proc.cpu.regs.read_ar(s);
                        if (as_ & 0xf) == 6 {
                            way6_ops.push(format!(
                                "n={n} pc={pc:#x} wdtlb as={as_:#x} (way6 vpn={:#x})",
                                as_ & !0xf
                            ));
                        }
                    }
                    Op::Idtlb { s } => {
                        let as_ = proc.cpu.regs.read_ar(s);
                        if (as_ & 0xf) == 6 {
                            way6_ops.push(format!(
                                "n={n} pc={pc:#x} idtlb as={as_:#x} (way6 vpn={:#x})",
                                as_ & !0xf
                            ));
                        }
                    }
                    _ => {}
                }
            }

            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                other => {
                    eprintln!("stopped early: {other:?} at n={n} pc={:#x}", proc.cpu.pc);
                    break;
                }
            }

            let region_live = proc.cpu.mmu.dtlb[6][0].asid != 0;
            if region_live != prev_region_live {
                region_transitions.push(format!(
                    "n={n} pc={pc:#x} dtlb[6][0].asid {} -> {}",
                    if prev_region_live { "live" } else { "dead" },
                    if region_live { "live" } else { "dead" }
                ));
                prev_region_live = region_live;
            }

            if let Some((wi, ei)) = find_poison(&proc.cpu.mmu) {
                let e = proc.cpu.mmu.dtlb[wi][ei];
                let r0 = proc.cpu.mmu.dtlb[6][0];
                eprintln!("=== poison autorefill entry first appeared ===");
                eprintln!("  n={n}  triggering instr (just stepped) = {pc:#x}  (prev instr = {prev_pc:#x})");
                eprintln!(
                    "  dtlb[{wi}][{ei}] vaddr={:#x} paddr={:#x} asid={} attr={} var={}",
                    e.vaddr, e.paddr, e.asid, e.attr, e.variable
                );
                eprintln!(
                    "  region dtlb[6][0] vaddr={:#x} paddr={:#x} asid={} attr={}  (live={})",
                    r0.vaddr,
                    r0.paddr,
                    r0.asid,
                    r0.attr,
                    r0.asid != 0
                );
                eprintln!("  dtlbcfg={:#x} ptevaddr={:#x}", proc.cpu.mmu.dtlbcfg, proc.cpu.mmu.ptevaddr);
                eprintln!("--- way-6 D-side TLB ops so far ({}) ---", way6_ops.len());
                for l in &way6_ops {
                    eprintln!("  {l}");
                }
                eprintln!("--- dtlb[6][0] liveness transitions ({}) ---", region_transitions.len());
                for l in &region_transitions {
                    eprintln!("  {l}");
                }
                found = true;
                break;
            }
            prev_pc = pc;
        }
        if !found {
            eprintln!("no poison autorefill entry appeared within {max} instrs (n={n})");
        }
    }

    /// EXTERNAL-REQUEST OBSERVATION (2026-07-08): the causality-respecting first
    /// step of the external-agent principle. Boots naturally (array attached) and
    /// logs every firmware STORE whose effective address lands in an external
    /// aperture -- the per-column HW/SMU pages (`0x2727xxxx`, whole `0x27xxxxxx`
    /// peripheral/SMN/mailbox band) and device SRAM (`0x03xxxxxx`). It POKES
    /// NOTHING: it only observes whether the firmware ever makes an external
    /// request (an ack to `0x2727n114`, a mailbox/SMU write, an alive-struct
    /// store) during reachable boot. If it writes nothing external, the completion
    /// is not externally-requested in reachable boot -> the divergence is upstream
    /// (the request-generating code never runs) and the agent cannot help until
    /// that is fixed. If it does, those sites ARE the contract to respond to.
    /// Env: XDNA_FW_MAX (budget, default 1_500_000). Ignored unless XDNA_FW_PROBE.
    #[test]
    fn m2c_probe_external_requests() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the external-request observation");
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
            .unwrap_or(1_500_000);

        // External apertures: the 0x27xxxxxx peripheral/SMN/mailbox/per-column band
        // and device SRAM 0x03xxxxxx. Everything else is firmware-internal RAM.
        let is_external =
            |a: u32| (0x2700_0000..0x2800_0000).contains(&a) || (0x0300_0000..0x0320_0000).contains(&a);

        let syms = load_symbols();
        let mut n = 0u64;
        let mut stop = "budget reached";
        // (pc, addr) -> (count, first_n, last_value)
        let mut sites: std::collections::BTreeMap<(u32, u32), (u64, u64, u32)> =
            std::collections::BTreeMap::new();
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
                    | decode::Op::S32c1i { t, s, imm }
                    | decode::Op::S32e { t, s, imm } => {
                        Some((proc.cpu.regs.read_ar(t), proc.cpu.regs.read_ar(s).wrapping_add(imm)))
                    }
                    _ => None,
                };
                if let Some((val, addr)) = sv {
                    if is_external(addr) {
                        let e = sites.entry((pc & 0x00ff_ffff, addr)).or_insert((0, n, val));
                        e.0 += 1;
                        e.2 = val;
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
        }
        eprintln!("=== external-request observation (array attached) ===");
        eprintln!("instrs = {n}, stop = {stop}");
        eprintln!("distinct external store sites = {}", sites.len());
        eprintln!("--- (pc, addr) <- last_val  count  first_n  sym ---");
        for ((pc, addr), (count, first_n, val)) in &sites {
            eprintln!(
                "  pc={pc:#08x}  [{addr:#010x}] <- {val:#010x}  x{count}  first@{first_n}  {}",
                nearest_symbol(&syms, *pc)
            );
        }
    }

    /// EXTERNAL CONVERSATION TRACE (2026-07-08): the write->read request/response
    /// pairing the external-agent model must reproduce. Boots naturally (array
    /// attached, pokes NOTHING) and logs, in temporal order, every firmware load
    /// AND store whose effective address lands in the external band
    /// (`0x27xxxxxx` peripheral/mailbox/per-column, `0x03xxxxxx` device SRAM).
    /// The `m2c_probe_external_requests` sibling captures WRITES only, aggregated;
    /// this one keeps loads too and in sequence, so the handshake protocol is
    /// visible: request write -> status poll read -> ack write -> re-poll, etc.
    /// Window: [XDNA_FW_CONV_START (default 38000), +XDNA_FW_CONV_LEN (default
    /// 16000)]. Caps printed lines at XDNA_FW_CONV_CAP (default 500) but always
    /// prints the full per-(pc,addr,dir) summary. Ignored unless XDNA_FW_PROBE.
    #[test]
    fn m2c_probe_external_conversation() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the external-conversation trace");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        if std::env::var("XDNA_FW_CONV_NOATTACH").is_err() {
            proc.bus.attach_device(crate::device::DeviceState::new_npu1());
        }
        let start: u64 = std::env::var("XDNA_FW_CONV_START")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(38_000);
        let len: u64 = std::env::var("XDNA_FW_CONV_LEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16_000);
        let cap: usize = std::env::var("XDNA_FW_CONV_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(500);
        let end = start + len;

        let is_external =
            |a: u32| (0x2700_0000..0x2800_0000).contains(&a) || (0x0300_0000..0x0320_0000).contains(&a);

        let syms = load_symbols();
        let mut n = 0u64;
        let mut stop = "budget reached";
        let mut printed = 0usize;
        // (pc, addr, is_write) -> (count, first_n, last_value)
        let mut sites: std::collections::BTreeMap<(u32, u32, bool), (u64, u64, u32)> =
            std::collections::BTreeMap::new();
        // Decoupled PC-visit counters: 0x8c88 is FUN_00008c68's poll L8ui -- in
        // steady state its base is the INTERNAL RAM struct 0xf9e0+col*0x60 (bit3),
        // NOT the external 0x2727n114 page (that base only appears in a rarer
        // iteration); 0xc964 is the sched_ready_popcount rest loop.
        let mut hit_8c88 = 0u64;
        let mut hit_c964 = 0u64;
        eprintln!("=== external conversation (temporal, window [{start},{end})) ===");
        while n < end {
            let pc = proc.cpu.pc;
            match pc & 0x00ff_ffff {
                0x8c88 => hit_8c88 += 1,
                0xc964 => hit_c964 += 1,
                _ => {}
            }
            if let Ok(phys) = proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                let d = decode::decode(&b, pc);
                // (is_write, base_reg, imm, store_val_reg)
                let acc: Option<(bool, u8, u32, Option<u8>)> = match d.op {
                    decode::Op::L32i { s, imm, .. }
                    | decode::Op::L32iN { s, imm, .. }
                    | decode::Op::L8ui { s, imm, .. }
                    | decode::Op::L16ui { s, imm, .. }
                    | decode::Op::L16si { s, imm, .. } => Some((false, s, imm, None)),
                    decode::Op::S32i { t, s, imm }
                    | decode::Op::S32iN { t, s, imm }
                    | decode::Op::S16i { t, s, imm }
                    | decode::Op::S8i { t, s, imm } => Some((true, s, imm, Some(t))),
                    _ => None,
                };
                if let Some((is_write, base, imm, vreg)) = acc {
                    let addr = proc.cpu.regs.read_ar(base).wrapping_add(imm);
                    if is_external(addr) && n >= start {
                        let val = match vreg {
                            Some(t) => proc.cpu.regs.read_ar(t),
                            None => proc.cpu.data_read32(&mut proc.bus, addr).unwrap_or(0),
                        };
                        let e = sites.entry((pc & 0x00ff_ffff, addr, is_write)).or_insert((0, n, val));
                        e.0 += 1;
                        e.2 = val;
                        if printed < cap {
                            eprintln!(
                                "  n={n:>7} pc={:#08x} {} [{addr:#010x}] {val:#010x}  {}",
                                pc & 0x00ff_ffff,
                                if is_write { "W" } else { "R" },
                                nearest_symbol(&syms, pc & 0x00ff_ffff)
                            );
                            printed += 1;
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
        }
        eprintln!(
            "--- stop={stop}, instrs={n}, final_pc={:#x} {}, printed={printed}/{} distinct ---",
            proc.cpu.pc,
            nearest_symbol(&syms, proc.cpu.pc & 0x00ff_ffff),
            sites.len()
        );
        eprintln!("--- pc-visits: 0x8c88(0x2727-poll)={hit_8c88}  0xc964(sched_ready)={hit_c964} ---");
        eprintln!("--- summary: (pc, addr, dir) <- last_val  count  first_n  sym ---");
        for ((pc, addr, is_write), (count, first_n, val)) in &sites {
            eprintln!(
                "  pc={pc:#08x} {} [{addr:#010x}] {val:#010x}  x{count}  first@{first_n}  {}",
                if *is_write { "W" } else { "R" },
                nearest_symbol(&syms, *pc)
            );
        }
    }

    /// INTLEVEL SEAM (2026-07-08): the completion the boot waits on is a masked
    /// interrupt (the external side is one-directional programming; both live
    /// gates -- `[0xf9e0+col*0x60]` bit3 and `[task+0x30]` -- are internal flags
    /// an ISR would set). This decides determination (A) vs (B): does INTLEVEL
    /// ever drop low enough for an interrupt to deliver? Boots naturally (array
    /// attached, pokes nothing) and reports, over the whole boot and over the
    /// post-wall tail (n >= XDNA_FW_WALL_N, default 48000): a histogram of
    /// PS.INTLEVEL, the min post-wall INTLEVEL, first/last n at which INTLEVEL==0,
    /// the distinct INTENABLE values seen (which interrupts the fw enabled), the
    /// final INTERRUPT (pending) word, and how many instructions satisfy
    /// `interrupt_deliverable()` (our interp models level-1 delivery only: it
    /// requires INTLEVEL==0). If INTLEVEL never reaches 0 post-wall, a level-1
    /// completion IRQ can never land here -> either the fw should lower it and we
    /// diverge (A), or the real completion is a high-level (>2) interrupt this
    /// interp does not model (a modeling gap on the B side). XDNA_FW_MAX budget
    /// (default 1_500_000). Ignored unless XDNA_FW_PROBE.
    #[test]
    fn m2c_probe_intlevel_seam() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the intlevel-seam probe");
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
            .unwrap_or(1_500_000);
        let wall_n: u64 = std::env::var("XDNA_FW_WALL_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(48_000);

        let syms = load_symbols();
        let mut n = 0u64;
        let mut stop = "budget reached";
        let mut hist: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        let mut post_hist: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        let mut intenables: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        let mut first_il0: Option<(u64, u32)> = None;
        let mut last_il0: Option<(u64, u32)> = None;
        let mut il0_count = 0u64;
        let mut deliverable = 0u64;
        // INTLEVEL transitions: (n, pc, old, new, disasm-of-the-instr-that-changed-it).
        // The instruction that CHANGED INTLEVEL is the one just stepped, so record
        // the pc/disasm from BEFORE the step and pair it with the post-step level.
        let mut transitions: Vec<(u64, u32, u32, u32, String)> = Vec::new();
        let mut prev_il = proc.cpu.regs.intlevel();
        while n < max {
            let il = proc.cpu.regs.intlevel();
            *hist.entry(il).or_insert(0) += 1;
            if n >= wall_n {
                *post_hist.entry(il).or_insert(0) += 1;
            }
            intenables.insert(proc.cpu.intenable);
            if il == 0 {
                il0_count += 1;
                let here = (n, proc.cpu.pc & 0x00ff_ffff);
                first_il0.get_or_insert(here);
                last_il0 = Some(here);
            }
            if proc.cpu.interrupt_deliverable() {
                deliverable += 1;
            }
            let pc = proc.cpu.pc;
            let disasm = match proc.cpu.translate(&mut proc.bus, pc, xtensa::interp::Access::Fetch) {
                Ok(phys) => {
                    let b: [u8; 8] = std::array::from_fn(|k| proc.bus.fetch8(pc + k as u32, phys + k as u32));
                    format!("{:?}", decode::decode(&b, pc).op)
                }
                Err(_) => "<fault>".to_string(),
            };
            let r = proc.cpu.step(&mut proc.bus);
            let new_il = proc.cpu.regs.intlevel();
            if new_il != prev_il {
                transitions.push((n, pc & 0x00ff_ffff, prev_il, new_il, disasm));
                prev_il = new_il;
            }
            match r {
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
        }
        eprintln!("=== intlevel seam (natural boot, array attached) ===");
        eprintln!(
            "instrs={n}, stop={stop}, final_pc={:#x} {}",
            proc.cpu.pc,
            nearest_symbol(&syms, proc.cpu.pc & 0x00ff_ffff)
        );
        eprintln!("--- INTLEVEL histogram (whole boot) ---");
        for (il, c) in &hist {
            eprintln!("  intlevel={il}  x{c}");
        }
        eprintln!("--- INTLEVEL histogram (post-wall, n>={wall_n}) ---");
        for (il, c) in &post_hist {
            eprintln!("  intlevel={il}  x{c}");
        }
        let min_post = post_hist.keys().next().copied();
        eprintln!("min post-wall INTLEVEL = {min_post:?}");
        eprintln!("INTLEVEL==0: count={il0_count}  first={first_il0:?}  last={last_il0:?}");
        eprintln!("interrupt_deliverable() true count = {deliverable}");
        eprintln!(
            "final INTENABLE = {:#010x}  final INTERRUPT(pending) = {:#010x}",
            proc.cpu.intenable, proc.cpu.interrupt
        );
        eprintln!("distinct INTENABLE values seen ({}):", intenables.len());
        for e in &intenables {
            eprintln!("  {e:#010x}");
        }
        eprintln!("--- INTLEVEL transitions ({} total; first 40 + last 20) ---", transitions.len());
        let show: Vec<&(u64, u32, u32, u32, String)> = if transitions.len() <= 60 {
            transitions.iter().collect()
        } else {
            transitions
                .iter()
                .take(40)
                .chain(transitions.iter().skip(transitions.len() - 20))
                .collect()
        };
        for (tn, pc, old, new, dis) in show {
            eprintln!("  n={tn:>8} pc={pc:#08x} {old}->{new}  {:<28} {}", dis, nearest_symbol(&syms, *pc));
        }
        // The pin point: the last transition INTO 2 that is never followed by a drop below 2.
        if let Some((tn, pc, old, new, dis)) =
            transitions.iter().rev().find(|(_, _, _, new, _)| *new <= 2 && *new == 2)
        {
            eprintln!(
                "last transition to INTLEVEL 2: n={tn} pc={pc:#08x} {old}->{new} {dis} {}",
                nearest_symbol(&syms, *pc)
            );
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

    /// (A) ISR OBSERVATION (2026-07-09, Maya-approved single controlled delivery).
    /// The boot livelock never delivers the armed line-0 IRQ (INTLEVEL pinned at
    /// 2), so the ISR's callback-dispatched tail (`Callx4 [lit]` -> trampoline ->
    /// `Callx8 [obj+168]` -> ?) can't be read statically (FLIX-heavy, runtime
    /// pointers) nor traced under natural boot. This warms to steady state, FORCES
    /// ONE faithful delivery (drop PS.INTLEVEL->0 and clear PS.EXCM, set INTERRUPT
    /// |= INTENABLE) and TRACES the ISR path through the interp -- the FLIX ground
    /// truth (its `step()` executes `Op::Flix1` bundles natively). It records every
    /// PC (with nearest symbol), MMIO reads/writes (>= 0x2000_0000 -- the TRUE
    /// interrupt-source register the ISR polls), each Call/Callx target (resolving
    /// the runtime callback chain), reaching `post_event` (0xcf68) / `wake` (0xd84c)
    /// / `deliver_pending_events` (0xcadc), changes to the global event accumulator
    /// `[0x22bc]`, and where `rfe` returns. DIAGNOSTIC observation of the ISR
    /// mechanism -- NOT a livelock break (no array modelled, one hand-forced
    /// delivery). Env: XDNA_FW_ISR_WARMUP (default 300000), XDNA_FW_ISR_STEPS
    /// (default 20000), XDNA_FW_ISR_SRC=addr:val (hex, seed an interrupt-source reg
    /// before delivery, e.g. 0x272003b8:0x1000). Ignored unless XDNA_FW_PROBE set.
    #[test]
    fn m2c_probe_isr_observe() {
        if std::env::var("XDNA_FW_PROBE").is_err() {
            eprintln!("skip: set XDNA_FW_PROBE=1 to run the ISR-observation probe");
            return;
        }
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let env_u64 = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        let warmup = env_u64("XDNA_FW_ISR_WARMUP", 300_000);
        let steps = env_u64("XDNA_FW_ISR_STEPS", 20_000);
        // Optional interrupt-source seed: XDNA_FW_ISR_SRC=addr:val (hex).
        let src: Option<(u32, u32)> = std::env::var("XDNA_FW_ISR_SRC").ok().and_then(|s| {
            let (a, v) = s.split_once(':')?;
            Some((
                u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?,
                u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()?,
            ))
        });
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);

        for _ in 0..warmup {
            if !matches!(proc.cpu.step(&mut proc.bus), Step::Ran | Step::Exception { .. }) {
                break;
            }
        }
        const ACCUM: u32 = 0x22bc; // global event accumulator [sched+108]
        let pre_pc = proc.cpu.pc;
        let pre_accum = proc.bus.data_load32(ACCUM);
        eprintln!("=== M2c ISR observation (warmup {warmup}) ===");
        eprintln!(
            "pre: pc={:#x} {} INTENABLE={:#010x} INTLEVEL={} excm={} [0x22bc]={:#x} cur-task[0x2278]={:#x}",
            pre_pc,
            nearest_symbol(&proc.symbols, pre_pc & 0x00ff_ffff),
            proc.cpu.intenable,
            proc.cpu.regs.intlevel(),
            proc.cpu.regs.excm(),
            pre_accum,
            proc.bus.data_load32(0x2278),
        );
        if let Some((a, v)) = src {
            let _ = proc.cpu.data_write32(&mut proc.bus, a, v);
            eprintln!("seeded interrupt-source [{a:#x}] = {v:#x}");
        }
        // FORCE ONE deliverable window: drop INTLEVEL to 0, clear PS.EXCM, assert
        // the armed line(s). The interp's step() then delivers faithfully next tick
        // (EXCCAUSE=4, EPC1<-pc, vector 0x2958).
        proc.cpu.regs.set_intlevel(0);
        proc.cpu.regs.ps &= !0x10; // clear PS.EXCM (bit 4)
        proc.cpu.interrupt |= proc.cpu.intenable;
        eprintln!("forced delivery: INTERRUPT |= {:#010x}", proc.cpu.intenable);

        let dumpreads = std::env::var("XDNA_FW_ISR_DUMPREADS").is_ok();
        let mut prev_accum = pre_accum;
        let mut delivered = false;
        let mut returned_at: Option<u64> = None;
        let mut n = 0u64;
        while n < steps {
            let pc = proc.cpu.pc & 0x00ff_ffff;
            // Decode for EA + call-target annotation.
            let (op_str, note) =
                match proc.cpu.translate(&mut proc.bus, proc.cpu.pc, xtensa::interp::Access::Fetch) {
                    Ok(phys) => {
                        let b: [u8; 8] =
                            std::array::from_fn(|k| proc.bus.fetch8(proc.cpu.pc + k as u32, phys + k as u32));
                        let d = decode::decode(&b, proc.cpu.pc);
                        let note = match d.op {
                            Op::L32i { s, imm, .. }
                            | Op::L32iN { s, imm, .. }
                            | Op::L8ui { s, imm, .. }
                            | Op::L16ui { s, imm, .. } => {
                                let ea = proc.cpu.regs.read_ar(s).wrapping_add(imm);
                                if ea >= 0x2000_0000 {
                                    format!(" MMIO-read ea={ea:#x}")
                                } else if dumpreads {
                                    let v = proc.bus.data_load32(ea & 0x00ff_ffff);
                                    format!(" read [{:#x}]={:#x}", ea & 0x00ff_ffff, v)
                                } else {
                                    String::new()
                                }
                            }
                            Op::Call8 { target }
                            | Op::Call4 { target }
                            | Op::Call12 { target }
                            | Op::Call0 { target } => {
                                format!(" -> call {:#x} {}", target, nearest_symbol(&proc.symbols, target))
                            }
                            Op::Callx8 { s } | Op::Callx4 { s } | Op::Callx12 { s } => {
                                let t = proc.cpu.regs.read_ar(s);
                                format!(
                                    " -> callx {:#x} {}",
                                    t,
                                    nearest_symbol(&proc.symbols, t & 0x00ff_ffff)
                                )
                            }
                            _ => String::new(),
                        };
                        (format!("{:?}", d.op), note)
                    }
                    Err(_) => ("<fault>".into(), String::new()),
                };
            // Log handler entry, event-machinery hits, and all calls/MMIO.
            let key = matches!(pc, 0x2958 | 0xcf68 | 0xcf5c | 0xd84c | 0xcadc | 0x2911);
            if !delivered && pc == 0x2958 {
                delivered = true;
                eprintln!("--- [{n}] ISR ENTRY @0x2958 (EXCCAUSE={}) ---", proc.cpu.regs.exccause);
            }
            if key || !note.is_empty() {
                eprintln!("[{n:>5}] {pc:#08x} {:<22} {op_str}{note}", nearest_symbol(&proc.symbols, pc));
            }
            match proc.cpu.step(&mut proc.bus) {
                Step::Ran | Step::Exception { .. } => n += 1,
                Step::Wait(r) => {
                    eprintln!("[{n}] WAIT({r:?}) @ {pc:#x}");
                    n += 1;
                    break;
                }
                Step::Unknown { pc: u, word } => {
                    eprintln!("[{n}] UNKNOWN @ {u:#x} word={word:#010x}");
                    break;
                }
            }
            let accum = proc.bus.data_load32(ACCUM);
            if accum != prev_accum {
                eprintln!("      *** [0x22bc] {prev_accum:#x} -> {accum:#x} (event posted) ***");
                prev_accum = accum;
            }
            if delivered
                && returned_at.is_none()
                && !proc.cpu.regs.excm()
                && (proc.cpu.pc & 0x00ff_ffff) == (pre_pc & 0x00ff_ffff)
            {
                returned_at = Some(n);
                eprintln!("--- [{n}] rfe RETURNED to preempted pc {:#x} ---", pre_pc);
                break;
            }
        }
        eprintln!("--- ISR observation summary ---");
        eprintln!("delivered={delivered} returned_at={returned_at:?} steps_run={n}");
        eprintln!(
            "post: [0x22bc]={:#x} (pre {:#x}) cur-task[0x2278]={:#x} pc={:#x}",
            proc.bus.data_load32(ACCUM),
            pre_accum,
            proc.bus.data_load32(0x2278),
            proc.cpu.pc,
        );
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
}
