//! Register-file port facts of an instruction, derived from the resolved TableGen itinerary.
//!
//! The scheduling model names the units an instruction holds (a register-file write port, a
//! read port, ...) and for how many cycles. Silicon honours those reservations until a schedule
//! breaks them; what happens then is modelled by the collision resolver (`state::write_collision`)
//! and the read-side hooks, which need to know, per instruction:
//!
//! * the unit that writes its result into the vector or accumulator register file, and
//! * the unit(s) it holds in the cycle it reads its accumulator source.
//!
//! No unit name appears in this file: units are indices into `TblgenOutput::functional_units`
//! and every classification below is read off the itinerary tables and the destination register
//! classes. Two properties are *inferred* from the silicon collision probes (documented in
//! `docs/fidelity-gaps/vector-compute.md`) and pinned by the tests at the bottom:
//!
//! * a **mover** is an instruction that reads an accumulator through the main read port and
//!   writes through the last (highest-indexed) write port of a file;
//! * the **secondary read unit** of an instruction is the highest-indexed unit it holds in its
//!   accumulator read cycle when it holds two (the shifter that the SRS family adds to the
//!   read port).

use smallvec::SmallVec;
use std::collections::HashMap;

use xdna_archspec::aie2::isa::decoder_ffi::ResolvedStage;
use xdna_archspec::aie2::isa::reservation::{held_units, reservations};
use xdna_archspec::aie2::isa::TblgenOutput;

use super::register_map::classify_reg_name;
use crate::interpreter::bundle::{AccReadPort, Operand, PortUse, ScalarDefPort, ScalarFile};
use crate::interpreter::state::{
    MOD_BASE_D3D, MOD_BASE_DC, MOD_BASE_DJ, MOD_BASE_DN, NUM_SCALAR_GPRS, NUM_SHIFT_REGS, SHIFT_REG_BASE,
    SP_PTR_INDEX,
};

/// The register files an instruction can write through a collision-modelled port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegFile {
    Vector = 0,
    Accum = 1,
}

impl RegFile {
    fn of_operand(op: &Operand) -> Option<Self> {
        match op {
            Operand::VectorReg(_) => Some(Self::Vector),
            Operand::AccumReg(_) => Some(Self::Accum),
            _ => None,
        }
    }
    fn bit(self) -> u8 {
        1 << self as u8
    }
}

impl ScalarFile {
    /// The scalar register file of a destination operand; `None` for registers outside the
    /// seven files (link/loop registers, status registers, the stack pointer, descriptors).
    pub fn of_operand(op: &Operand) -> Option<Self> {
        match *op {
            Operand::ScalarReg(r) if (r as usize) < NUM_SCALAR_GPRS => Some(Self::R),
            Operand::ScalarReg(r) if (SHIFT_REG_BASE..SHIFT_REG_BASE + NUM_SHIFT_REGS).contains(&r) => {
                Some(Self::S)
            }
            Operand::PointerReg(p) if p != SP_PTR_INDEX => Some(Self::P),
            Operand::ModifierReg(r) if r < MOD_BASE_DN => Some(Self::M),
            Operand::ModifierReg(r) if r < MOD_BASE_DJ => Some(Self::Dn),
            Operand::ModifierReg(r) if r < MOD_BASE_DC => Some(Self::Dj),
            Operand::ModifierReg(r) if r < MOD_BASE_D3D => Some(Self::Dc),
            _ => None,
        }
    }
}

/// Which units write which register file, and each unit's rank among the write ports of that
/// file, derived once from the itineraries.
#[derive(Debug, Clone, Default)]
pub struct PortModel {
    /// Per unit: bit per [`RegFile`] the unit is the result-write port of.
    write_files: Vec<u8>,
    /// Per file: the highest-indexed write port. The last port of a file is the one the moves
    /// and the non-accumulating converts use; the others serve the arithmetic and the loads.
    last: [Option<u16>; 2],
}

impl PortModel {
    /// Derive the write-port table. An itinerary class that holds exactly one unit in the cycle
    /// its result is written (`operand_cycles[0] - 1`) names that unit the write port of the
    /// register files its instructions write; classes with several candidates (a shuffle that
    /// the model reserves in both files) are resolved per instruction by destination file in
    /// [`PortModel::port_use`].
    pub fn derive(tbl: &TblgenOutput) -> Self {
        let n = tbl.functional_units.len();
        let index: HashMap<&str, u16> = tbl
            .functional_units
            .iter()
            .enumerate()
            .map(|(i, u)| (u.as_str(), i as u16))
            .collect();
        let mut write_files = vec![0u8; n];
        for (class, reg_class) in &tbl.itinerary_output_classes {
            let Some(itin) = tbl.itineraries.get(class) else {
                continue;
            };
            let Some(&def) = itin.operand_cycles.first() else {
                continue;
            };
            if def == 0 {
                continue;
            }
            let Some(file) = Self::class_file(tbl, reg_class) else {
                continue;
            };
            let mut at_def: Vec<u16> = reservations(itin)
                .into_iter()
                .filter(|r| r.offset + 1 == def as u32)
                .map(|r| index[r.unit.as_str()])
                .collect();
            at_def.sort_unstable();
            at_def.dedup();
            if let [unit] = at_def[..] {
                write_files[unit as usize] |= file.bit();
            }
        }
        let last = [RegFile::Vector, RegFile::Accum]
            .map(|f| (0..n as u16).filter(|u| write_files[*u as usize] & f.bit() != 0).max());
        Self { write_files, last }
    }

    /// The register file of a register class: that of its first leaf register (classes may
    /// list other classes as members). Composite operand classes that union several files (a
    /// shuffle destination is an x or a bm) have no members and give `None`.
    fn class_file(tbl: &TblgenOutput, reg_class: &str) -> Option<RegFile> {
        let mut name = reg_class;
        for _ in 0..8 {
            match tbl.register_model.classes.get(name) {
                Some(class) => name = class.members.first()?,
                None => return RegFile::of_operand(&classify_reg_name(name)?.operand),
            }
        }
        None
    }

    /// The write ports of each file in unit-index order (tests): `(vector, accumulator)`.
    #[cfg(test)]
    pub(crate) fn write_ports(&self) -> (Vec<u16>, Vec<u16>) {
        let of = |f: RegFile| (0..self.write_files.len() as u16).filter(|u| self.writes(*u, f)).collect();
        (of(RegFile::Vector), of(RegFile::Accum))
    }

    fn writes(&self, unit: u16, file: RegFile) -> bool {
        self.write_files.get(unit as usize).is_some_and(|b| b & file.bit() != 0)
    }

    fn is_last_port(&self, unit: u16) -> bool {
        self.last.contains(&Some(unit))
    }

    /// The scalar-file results of an instruction: for each def operand `(operand, result cycle)`
    /// in a scalar register file, the units held in the cycle before the result lands. Which of
    /// them is the write port is not decided here: two results collide when they share a unit
    /// and a file (`state::scalar_collision`), so the read port a mover also holds is harmless.
    fn scalar_defs(held: &[(u16, u32, bool)], defs: &[(Operand, i16)]) -> SmallVec<[ScalarDefPort; 1]> {
        defs.iter()
            .filter_map(|(operand, cycle)| {
                let file = ScalarFile::of_operand(operand)?;
                let land = u8::try_from(*cycle).ok().filter(|c| *c >= 1)?;
                let mut units: SmallVec<[u16; 4]> = held
                    .iter()
                    .filter(|(_, offset, reserved)| *offset + 1 == land as u32 && !reserved)
                    .map(|(u, _, _)| *u)
                    .collect();
                units.sort_unstable();
                units.dedup();
                (!units.is_empty()).then_some(ScalarDefPort { file, land, units })
            })
            .collect()
    }

    /// Port facts of one decoded instruction.
    ///
    /// `def_cycle` is the resolved itinerary cycle of the first result, `source_forward` the
    /// per-source `(use cycle, bypass)` aligned with `sources`.
    pub fn port_use(
        &self,
        stages: &[ResolvedStage],
        def_cycle: Option<i16>,
        dest: Option<&Operand>,
        sources: &[Operand],
        source_forward: &[(u8, xdna_archspec::aie2::Bypass)],
        defs: &[(Operand, i16)],
    ) -> PortUse {
        let held = held_units(stages);
        let mut out = PortUse::default();
        out.scalar_defs = Self::scalar_defs(&held, defs);

        // Accumulator read: units held (not merely reserved) in the cycle of the first
        // accumulator source's use.
        let acc_src = sources.iter().position(|s| matches!(s, Operand::AccumReg(_)));
        if let Some((pos, &(use_cycle, _))) = acc_src.and_then(|p| source_forward.get(p).map(|f| (p, f))) {
            let _ = pos;
            if use_cycle >= 1 {
                let offset = (use_cycle - 1) as u32;
                let mut units: SmallVec<[u16; 4]> =
                    held.iter().filter(|(_, o, r)| *o == offset && !r).map(|(u, _, _)| *u).collect();
                units.sort_unstable();
                units.dedup();
                if let Some(&primary) = units.first() {
                    out.acc_read = Some(AccReadPort {
                        offset: offset as u8,
                        primary,
                        secondary: (units.len() >= 2).then(|| *units.last().unwrap()),
                    });
                }
            }
        }

        // Result write: the file-matching unit held in the cycle before the result lands.
        if let Some(def) = def_cycle.filter(|d| *d >= 1) {
            let at_def: SmallVec<[u16; 4]> = held
                .iter()
                .filter(|(_, o, _)| *o + 1 == def as u32)
                .map(|(u, _, _)| *u)
                .collect();
            let file = dest.and_then(RegFile::of_operand);
            if let Some(file) = file {
                let mut cand: SmallVec<[u16; 2]> =
                    at_def.iter().copied().filter(|u| self.writes(*u, file)).collect();
                cand.sort_unstable();
                cand.dedup();
                if let [unit] = cand[..] {
                    out.write_port = Some(unit);
                }
            }
            if out.acc_read.is_some() {
                out.mover = at_def.iter().any(|u| self.is_last_port(*u));
            }
        }
        out
    }
}
