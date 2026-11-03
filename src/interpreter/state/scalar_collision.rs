//! Same-cycle collisions of scalar-file register writes (pure resolver).
//!
//! The scalar register files (r, s, p, m, dn, dj, dc) are written through write-port units the
//! TableGen schedule names (`AIE2Schedule.td`): an instruction holds the unit in the cycle
//! before its result lands. The compilers never schedule two instructions onto one unit in one
//! cycle; a hand-written schedule can (`mov.d3 r1, r2` followed two bundles later by
//! `mov r5, r6` lands both in one cycle), and NPU1 silicon is deterministic about it. The rules
//! were read off the probes in `build/experiments/harness/movd` (HW r1 == r2, three suites,
//! 510 trials); the evidence is in `docs/fidelity-gaps/core-compute-timing.md`.
//!
//! # Model
//!
//! Writes that land in one cycle, each with the units it holds, its destination register file
//! and register:
//!
//! 1. **Same port, same file: the latest-issued write wins.** A write is dropped (the register
//!    keeps its old value) when a later-issued write to the same register *file* holds one of
//!    its units. The dropper's own write survives. Writes to different files do not collide
//!    even when the schedule lists the same unit for both: `mov s0, r6` leaves a delayed
//!    `mov.d3 r1, r2` alone although both hold the r/s write unit. This is what the schedule
//!    alone gets wrong.
//! 2. **Different ports, same register: wired OR.** Surviving writes that reach one register
//!    through different ports leave the bitwise OR of their values (`add r1` + `mov.d3 r1`
//!    landing together: `0x12345673 | 0x0f0f0f01`).
//!
//! "Latest-issued" is the issue sequence number of the writing instruction.

use smallvec::SmallVec;

use crate::interpreter::bundle::ScalarFile;

/// One scalar-file write landing in the cycle being resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortedWrite {
    /// Issue sequence number of the writing instruction (larger = issued later).
    pub seq: u64,
    pub file: ScalarFile,
    /// Index of the destination register within its file.
    pub reg: u8,
    /// The units the instruction holds in the cycle before the result lands.
    pub units: SmallVec<[u16; 4]>,
    pub value: u32,
}

/// What happens to one input write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Landing {
    /// Whether the register is written at all.
    pub lands: bool,
    /// The value the register receives when it is.
    pub value: u32,
}

/// Resolve the writes that land in one cycle. `out[i]` describes `writes[i]`.
pub fn resolve(writes: &[PortedWrite]) -> Vec<Landing> {
    let shares_port =
        |a: &PortedWrite, b: &PortedWrite| a.file == b.file && a.units.iter().any(|u| b.units.contains(u));
    let survives: Vec<bool> = writes
        .iter()
        .map(|w| !writes.iter().any(|o| o.seq > w.seq && shares_port(w, o)))
        .collect();
    writes
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let value = writes
                .iter()
                .enumerate()
                .filter(|(j, o)| survives[*j] && o.file == w.file && o.reg == w.reg)
                .fold(0, |acc, (_, o)| acc | o.value);
            Landing { lands: survives[i], value }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RS: u16 = 4; // stand-ins for unit indices; the resolver never names a unit
    const WX: u16 = 1;
    const WA: u16 = 2;

    fn w(seq: u64, file: ScalarFile, reg: u8, units: &[u16], value: u32) -> PortedWrite {
        PortedWrite { seq, file, reg, units: units.iter().copied().collect(), value }
    }

    fn lands(l: &[Landing]) -> Vec<Option<u32>> {
        l.iter().map(|x| x.lands.then_some(x.value)).collect()
    }

    // Silicon (movd p1): mov.d3 r1 issued first, `mov r5, r6` issued 3 bundles later.
    #[test]
    fn the_earlier_issued_write_on_a_shared_port_is_dropped() {
        let r = resolve(&[w(1, ScalarFile::R, 1, &[RS], 0x22), w(2, ScalarFile::R, 5, &[RS], 0x66)]);
        assert_eq!(lands(&r), [None, Some(0x66)]);
    }

    // Silicon (movd p1/p2): the interferer's own write always survives, whichever way round
    // the delays are (the longer delay was issued first).
    #[test]
    fn only_the_latest_issued_of_three_survives() {
        let r = resolve(&[
            w(1, ScalarFile::R, 1, &[RS], 1),
            w(2, ScalarFile::R, 3, &[RS], 2),
            w(3, ScalarFile::R, 5, &[RS], 3),
        ]);
        assert_eq!(lands(&r), [None, None, Some(3)]);
    }

    // Silicon (movd p1): `mov s0, r6` / `mov p4, r6` hold the r/s unit in the schedule but do
    // not drop an r-file write.
    #[test]
    fn a_shared_unit_does_not_collide_across_register_files() {
        for other in [ScalarFile::S, ScalarFile::P, ScalarFile::M, ScalarFile::Dn] {
            let r = resolve(&[w(1, ScalarFile::R, 1, &[RS], 0x22), w(2, other, 0, &[RS], 0x66)]);
            assert_eq!(lands(&r), [Some(0x22), Some(0x66)], "{other:?}");
        }
    }

    // Silicon (movd p1 controls, p2 F4/F5): different ports do not collide.
    #[test]
    fn different_ports_do_not_collide() {
        let r = resolve(&[w(1, ScalarFile::R, 1, &[RS], 0x22), w(2, ScalarFile::R, 5, &[WX], 0x66)]);
        assert_eq!(lands(&r), [Some(0x22), Some(0x66)]);
    }

    // Silicon (movd p2 F4, lda p5 against mov p4): the p file collides on its own port.
    #[test]
    fn writes_to_the_same_file_on_its_own_port_collide() {
        let r = resolve(&[w(1, ScalarFile::P, 5, &[7], 0x77), w(2, ScalarFile::P, 4, &[7, RS], 0x66)]);
        assert_eq!(lands(&r), [None, Some(0x66)]);
    }

    // Silicon (movd p3 G1): the same register reached through two ports takes the bitwise OR.
    #[test]
    fn one_register_reached_through_two_ports_takes_the_or() {
        let r =
            resolve(&[w(1, ScalarFile::R, 1, &[RS], 0x12345673), w(2, ScalarFile::R, 1, &[WX], 0x0f0f0f01)]);
        assert_eq!(lands(&r), [Some(0x1f3f5f73), Some(0x1f3f5f73)]);
        let r = resolve(&[w(1, ScalarFile::R, 1, &[RS], 0x12345673), w(2, ScalarFile::R, 1, &[WA], 5)]);
        assert_eq!(lands(&r), [Some(0x12345677), Some(0x12345677)]);
    }

    // Silicon (movd p3 G1, mov r1 against mov.d3 r1): the same port is a drop, not an OR.
    #[test]
    fn one_register_on_one_port_takes_the_latest_value_alone() {
        let r =
            resolve(&[w(1, ScalarFile::R, 1, &[RS], 0x12345673), w(2, ScalarFile::R, 1, &[RS], 0x0f0f0f00)]);
        assert_eq!(lands(&r), [None, Some(0x0f0f0f00)]);
    }

    // A dropped write contributes nothing to a register that two other ports reach.
    #[test]
    fn a_dropped_write_does_not_enter_the_or() {
        let r = resolve(&[
            w(1, ScalarFile::R, 1, &[RS], 0xf0),
            w(2, ScalarFile::R, 1, &[WX], 0x01),
            w(3, ScalarFile::R, 3, &[RS], 0x66),
        ]);
        assert_eq!(lands(&r), [None, Some(0x01), Some(0x66)]);
    }

    #[test]
    fn a_lone_write_passes_through() {
        assert_eq!(lands(&resolve(&[w(1, ScalarFile::M, 2, &[3], 9)])), [Some(9)]);
    }
}
