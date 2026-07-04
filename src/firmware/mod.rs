//! In-tree base-Xtensa interpreter that runs the real NPU management firmware.
//!
//! Phase M0+M1 scope: load the `$PS1` image and boot it to a command-loop idle.
//! Device/mailbox MMIO routing into `DeviceState` is later (M2).

mod error;
mod image;
mod mmio;
mod sysstub;
pub mod xtensa;

pub use error::FirmwareError;
pub use image::FirmwareImage;
pub use mmio::Bus;
pub use sysstub::SysStub;

use std::collections::HashMap;
use std::path::Path;

use xtensa::decode::{self, Op};
use xtensa::interp::{Cpu, Step, WaitReason};

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
        let cpu = Cpu::new(entry);
        let symbols = load_symbols();
        Self { cpu, bus, entry, symbols }
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
                Step::Exception { .. } => {
                    window_exceptions += 1;
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

#[cfg(test)]
mod boot_tests {
    use super::*;

    fn firmware_path() -> Option<std::path::PathBuf> {
        // Env override first, then the known repo-relative download location.
        if let Ok(p) = std::env::var("XDNA_FIRMWARE") {
            let p = std::path::PathBuf::from(p);
            return p.exists().then_some(p);
        }
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin");
        p.exists().then_some(p)
    }

    /// The pinned boot entry: the first instruction of the MMU-init reset
    /// routine (`movi.n a2,0`), derived by coherence in this milestone. See
    /// the module-level M1.7 report for how it was pinned.
    const BOOT_ENTRY: u32 = 0x320;

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
}
