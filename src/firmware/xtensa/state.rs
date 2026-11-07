//! Serializable architectural state of the Xtensa core, so a platform can keep
//! a management processor's state across emulator processes.
//!
//! A real management processor boots once and keeps running while host
//! processes come and go; whatever its firmware sets up lazily (software-filled
//! TLB entries, say) stays set up. An emulator process that boots it again pays
//! those one-time costs again. A platform that saves [`CpuState`] together
//! with its own devices and memory, and restores both into a freshly
//! constructed [`Cpu`], continues the same device lifetime instead.
//!
//! What is state and what is configuration: everything an instruction can
//! change is in [`CpuState`] (program counter, the windowed AR file and its
//! special registers, the exception and interrupt registers, the FP register
//! file, the `waiti` halt flag, both TLBs and the MMU special registers). The
//! build-time configuration of the core (exception-vector layout, the MMU
//! configuration, the fill fast-path switch) comes from the constructor the
//! platform calls before restoring; the MMU configuration is also saved so a
//! state taken under another configuration is refused rather than misread. The
//! per-step instrument outputs (`last_work`, the fetched bytes, the data
//! accesses) describe the step that produced them and are not state.

use serde::{Deserialize, Serialize};

use super::interp::Cpu;
use super::mmu::{Mmu, MmuConfig, TlbEntry};
use super::regfile::RegFile;

/// The windowed register file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegFileState {
    /// The 64 physical AR registers.
    pub ar: Vec<u32>,
    pub windowbase: u32,
    pub windowstart: u32,
    pub sar: u32,
    pub ps: u32,
    pub lbeg: u32,
    pub lend: u32,
    pub lcount: u32,
    pub exccause: u32,
}

/// Both TLBs, the MMU special registers and the refill replacement state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MmuState {
    pub config: MmuConfig,
    /// ITLB entries, way-major.
    pub itlb: Vec<Vec<TlbEntry>>,
    /// DTLB entries, way-major.
    pub dtlb: Vec<Vec<TlbEntry>>,
    pub ptebase: u32,
    pub rasid: u32,
    pub itlbcfg: u32,
    pub dtlbcfg: u32,
    pub refill_victim: [u8; 2],
}

/// Everything a [`Cpu`] carries from one step to the next.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuState {
    pub pc: u32,
    pub regs: RegFileState,
    pub vecbase: u32,
    pub epc1: u32,
    pub depc: u32,
    pub excvaddr: u32,
    pub threadptr: u32,
    pub interrupt: u32,
    pub intenable: u32,
    pub scompare1: u32,
    pub excsave: [u32; 7],
    pub fr: [u32; 16],
    pub halted: bool,
    pub mmu: MmuState,
}

/// Why a state was not restored. The CPU is left unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestoreError {
    /// The state was taken under another MMU configuration.
    MmuConfig { saved: MmuConfig, this: MmuConfig },
    /// A table has another shape than this core's.
    Shape(String),
}

impl std::fmt::Display for RestoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RestoreError::MmuConfig { saved, this } => {
                write!(f, "saved under MMU configuration {saved:?}, this core has {this:?}")
            }
            RestoreError::Shape(e) => write!(f, "{e}"),
        }
    }
}

impl RegFile {
    pub fn state(&self) -> RegFileState {
        let RegFile { ar, windowbase, windowstart, sar, ps, lbeg, lend, lcount, exccause } = self;
        RegFileState {
            ar: ar.to_vec(),
            windowbase: *windowbase,
            windowstart: *windowstart,
            sar: *sar,
            ps: *ps,
            lbeg: *lbeg,
            lend: *lend,
            lcount: *lcount,
            exccause: *exccause,
        }
    }

    /// The register file `state` describes.
    pub fn from_state(state: &RegFileState) -> Result<Self, RestoreError> {
        let ar =
            state.ar.as_slice().try_into().map_err(|_| {
                RestoreError::Shape(format!("{} AR registers, the core has 64", state.ar.len()))
            })?;
        Ok(RegFile {
            ar,
            windowbase: state.windowbase,
            windowstart: state.windowstart,
            sar: state.sar,
            ps: state.ps,
            lbeg: state.lbeg,
            lend: state.lend,
            lcount: state.lcount,
            exccause: state.exccause,
        })
    }
}

fn table_from<const W: usize, const E: usize>(
    name: &str,
    ways: &[Vec<TlbEntry>],
) -> Result<[[TlbEntry; E]; W], RestoreError> {
    let shape =
        || RestoreError::Shape(format!("{name}: saved {} ways, the core has {W} of {E} entries", ways.len()));
    if ways.len() != W {
        return Err(shape());
    }
    let mut out = [[TlbEntry::default(); E]; W];
    for (dst, src) in out.iter_mut().zip(ways) {
        *dst = src.as_slice().try_into().map_err(|_| shape())?;
    }
    Ok(out)
}

impl Mmu {
    pub fn state(&self) -> MmuState {
        let Mmu { config, itlb, dtlb, ptebase, rasid, itlbcfg, dtlbcfg, refill_victim } = self;
        MmuState {
            config: *config,
            itlb: itlb.iter().map(|w| w.to_vec()).collect(),
            dtlb: dtlb.iter().map(|w| w.to_vec()).collect(),
            ptebase: *ptebase,
            rasid: *rasid,
            itlbcfg: *itlbcfg,
            dtlbcfg: *dtlbcfg,
            refill_victim: *refill_victim,
        }
    }

    /// Replace this MMU's state with `state`, which must have been taken under
    /// the same configuration.
    pub fn restore_state(&mut self, state: &MmuState) -> Result<(), RestoreError> {
        if state.config != self.config {
            return Err(RestoreError::MmuConfig { saved: state.config, this: self.config });
        }
        let itlb = table_from("ITLB", &state.itlb)?;
        let dtlb = table_from("DTLB", &state.dtlb)?;
        *self = Mmu {
            config: state.config,
            itlb,
            dtlb,
            ptebase: state.ptebase,
            rasid: state.rasid,
            itlbcfg: state.itlbcfg,
            dtlbcfg: state.dtlbcfg,
            refill_victim: state.refill_victim,
        };
        Ok(())
    }
}

impl Cpu {
    /// The core's architectural state (see the module documentation).
    pub fn state(&self) -> CpuState {
        // Destructured in full so a new field cannot be added without
        // deciding whether it is state.
        let Cpu {
            pc,
            regs,
            vecbase,
            epc1,
            depc,
            mmu,
            excvaddr,
            fastpath_enabled: _,
            threadptr,
            interrupt,
            intenable,
            scompare1,
            excsave,
            fr,
            halted,
            last_work: _,
            work_override: _,
            last_fetch: _,
            last_data: _,
            last_cache_op: _,
            last_fetch_attribute: _,
            vectors: _,
        } = self;
        CpuState {
            pc: *pc,
            regs: regs.state(),
            vecbase: *vecbase,
            epc1: *epc1,
            depc: *depc,
            excvaddr: *excvaddr,
            threadptr: *threadptr,
            interrupt: *interrupt,
            intenable: *intenable,
            scompare1: *scompare1,
            excsave: *excsave,
            fr: *fr,
            halted: *halted,
            mmu: mmu.state(),
        }
    }

    /// Continue from `state`: this CPU (built by the platform's constructor,
    /// which supplies the configuration) takes over the saved architectural
    /// state. On error nothing changes.
    pub fn restore_state(&mut self, state: &CpuState) -> Result<(), RestoreError> {
        let regs = RegFile::from_state(&state.regs)?;
        let mut mmu = self.mmu.clone();
        mmu.restore_state(&state.mmu)?;
        self.pc = state.pc;
        self.regs = regs;
        self.vecbase = state.vecbase;
        self.epc1 = state.epc1;
        self.depc = state.depc;
        self.mmu = mmu;
        self.excvaddr = state.excvaddr;
        self.threadptr = state.threadptr;
        self.interrupt = state.interrupt;
        self.intenable = state.intenable;
        self.scompare1 = state.scompare1;
        self.excsave = state.excsave;
        self.fr = state.fr;
        self.halted = state.halted;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::firmware::xtensa::decode::{decode, Op};
    use crate::firmware::xtensa::flat_bus::FlatBus;
    use crate::firmware::xtensa::interp::{unmapped_mmu, ExceptionVectors};
    use crate::firmware::xtensa::mmu::{MmuConfig, RefillEntries, Tlb};

    const DATA: u32 = 0x0100_0000;

    /// `j 0` placed at `pc`.
    fn j_to_zero(pc: u32) -> [u8; 3] {
        let offset = 0u32.wrapping_sub(pc + 4) & 0x3_ffff;
        let word = (offset << 6) | 0x6;
        let bytes = [word as u8, (word >> 8) as u8, (word >> 16) as u8];
        assert!(matches!(decode(&bytes, pc).op, Op::J { target: 0 }));
        bytes
    }

    /// A loop that loads, increments and stores a data word and writes SAR:
    /// every step goes through both TLBs and the register file.
    fn program() -> Vec<u8> {
        let mut p = vec![0x78, 0x02]; // l32i.n a7, a2, 0
        p.extend([0x72, 0xc7, 0x01]); // addi a7, a7, 1
        p.extend([0x79, 0x02]); // s32i.n a7, a2, 0
        p.extend([0x70, 0x03, 0x13]); // wsr a7, sar
        p.extend(j_to_zero(p.len() as u32));
        p
    }

    fn bus() -> FlatBus {
        let mut bus = FlatBus::new(program());
        bus.write32(DATA, 41);
        bus
    }

    /// A CPU with every state field away from its reset value.
    fn busy_cpu() -> Cpu {
        let mut cpu = Cpu::new(0, unmapped_mmu(), ExceptionVectors::default());
        cpu.mmu.write_entry(Tlb::Instruction, 1, 0x1);
        cpu.mmu.write_entry(Tlb::Data, DATA, DATA | 0x3);
        cpu.mmu.write_entry(Tlb::Data, 0x0300_0000 | 2, 0x0300_0000 | 0x13);
        for r in 0..16 {
            cpu.regs.write_ar(r, 0x1000 + u32::from(r));
        }
        cpu.regs.write_ar(2, DATA);
        cpu.regs.windowstart = 0b1011;
        cpu.regs.lbeg = 0x40;
        cpu.regs.lend = 0x50;
        cpu.regs.exccause = 26;
        cpu.vecbase = 0x4000_0000;
        cpu.epc1 = 0x11;
        cpu.depc = 0x22;
        cpu.excvaddr = 0x33;
        cpu.threadptr = 0x44;
        cpu.intenable = 0x5;
        cpu.scompare1 = 0x66;
        cpu.excsave = [1, 2, 3, 4, 5, 6, 7];
        cpu.fr[3] = 0x3f80_0000;
        cpu
    }

    fn round_trip(state: &CpuState) -> CpuState {
        serde_json::from_str(&serde_json::to_string(state).unwrap()).unwrap()
    }

    #[test]
    fn a_restored_cpu_steps_exactly_like_the_one_it_was_saved_from() {
        let mut a = busy_cpu();
        let mut bus_a = bus();
        for _ in 0..7 {
            a.step(&mut bus_a);
        }
        let saved = round_trip(&a.state());

        let mut b = Cpu::new(0, unmapped_mmu(), ExceptionVectors::default());
        assert_ne!(b.state(), saved, "a fresh core differs");
        b.restore_state(&saved).unwrap();
        assert_eq!(b.state(), a.state());

        let mut bus_b = bus();
        bus_b.write32(DATA, bus_a.read32(DATA));
        for step in 0..40 {
            assert_eq!(a.step(&mut bus_a), b.step(&mut bus_b), "step {step}");
            assert_eq!(a.state(), b.state(), "after step {step}");
        }
        // 47 steps = 9 passes of the five-instruction loop and two steps more.
        assert_eq!(bus_a.read32(DATA), 41 + 9, "the loop ran");
        assert_eq!(a.regs.sar, 41 + 9);
        assert_eq!(bus_b.read32(DATA), bus_a.read32(DATA));
    }

    #[test]
    fn without_the_saved_tlbs_the_same_step_faults() {
        let mut a = busy_cpu();
        let saved = a.state();
        let mut b = Cpu::new(0, unmapped_mmu(), ExceptionVectors::default());
        let mut partial = saved.clone();
        partial.mmu = b.mmu.state();
        b.restore_state(&partial).unwrap();
        assert!(matches!(b.step(&mut bus()), crate::firmware::xtensa::interp::Step::Exception { .. }));
        assert!(matches!(a.step(&mut bus()), crate::firmware::xtensa::interp::Step::Ran));
    }

    #[test]
    fn every_state_field_survives_serialization() {
        let mut cpu = busy_cpu();
        cpu.halted = true;
        cpu.interrupt = 0x80;
        cpu.regs.lcount = 3;
        cpu.regs.windowbase = 5;
        let state = cpu.state();
        assert_eq!(round_trip(&state), state);
        let mut b = Cpu::new(0, unmapped_mmu(), ExceptionVectors::default());
        b.restore_state(&round_trip(&state)).unwrap();
        assert_eq!(b.state(), state);
        assert_eq!(b.regs.read_ar(7), cpu.regs.read_ar(7));
    }

    #[test]
    fn a_state_of_another_mmu_configuration_is_refused_and_changes_nothing() {
        let saved = busy_cpu().state();
        let other = MmuConfig { dtlb_refill_entries: RefillEntries::ThirtyTwo, ..MmuConfig::GENERIC };
        let mut b = Cpu::new(0, crate::firmware::xtensa::mmu::Mmu::new(other), ExceptionVectors::default());
        let before = b.state();
        assert!(matches!(b.restore_state(&saved), Err(RestoreError::MmuConfig { .. })));
        assert_eq!(b.state(), before);
    }

    #[test]
    fn a_malformed_table_is_refused_and_changes_nothing() {
        let mut saved = busy_cpu().state();
        saved.mmu.dtlb.pop();
        let mut b = Cpu::new(0, unmapped_mmu(), ExceptionVectors::default());
        let before = b.state();
        assert!(matches!(b.restore_state(&saved), Err(RestoreError::Shape(_))));
        assert_eq!(b.state(), before);
        let mut saved = busy_cpu().state();
        saved.regs.ar.push(0);
        assert!(matches!(b.restore_state(&saved), Err(RestoreError::Shape(_))));
        assert_eq!(b.state(), before);
    }
}
