//! Resource reservation check over a bundle sequence.
//!
//! The AIE2 scheduling model (`AIE2Schedule.td`) describes every instruction as a list of
//! pipeline stages, each holding a functional unit (a register-file port, a memory unit, ...)
//! for some cycles. Two instructions that hold the same unit in the same cycle violate the
//! model; the compilers never emit such a schedule, so any hit is either a checker bug or a
//! hand-written (probe) schedule that leaves the modelled envelope.
//!
//! Everything here is derived from the extracted [`ItineraryInfo`] records: no unit name and no
//! stage number is spelled out in this file.
//!
//! Stage arithmetic follows LLVM's `InstrStage`: a stage holds its units for `cycles` cycles
//! starting at the running offset, then the offset advances by `time_inc` (or by `cycles` when
//! `time_inc` is negative). `EmptyCycles<n>` is a stage of 0 cycles with `time_inc = n`;
//! `PrefixCycle` has `time_inc = 0`, so the next stage starts in the same cycle.

use std::collections::{BTreeMap, HashMap};

use super::decoder_ffi::ResolvedStage;
use super::types::{ItineraryInfo, PipelineStage};

/// One unit held in one cycle, relative to the issue cycle of the instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reservation {
    pub unit: String,
    /// Cycles after issue.
    pub offset: u32,
    /// Stage kind `Reserved`: the unit is held but this stage does not require it free.
    pub reserved: bool,
}

/// The reservations of a stage list, in stage order.
///
/// A stage that lists several units means "any one of them". Every stage the AIE2 model
/// defines names exactly one unit (asserted by `no_stage_offers_a_choice_of_units`), so a
/// stage with a choice is rejected rather than approximated.
pub fn reservations_of(stages: &[PipelineStage]) -> Vec<Reservation> {
    let mut out = Vec::new();
    let mut start: i32 = 0;
    for stage in stages {
        if stage.cycles > 0 {
            assert!(stage.units.len() == 1, "stage with a choice of units {:?} is not modelled", stage.units);
            for c in 0..stage.cycles as i32 {
                out.push(Reservation {
                    unit: stage.units[0].clone(),
                    offset: (start + c) as u32,
                    reserved: stage.reserved,
                });
            }
        }
        start += if stage.time_inc >= 0 {
            stage.time_inc as i32
        } else {
            stage.cycles as i32
        };
    }
    out
}

/// The reservations of a named itinerary class (its static, register-independent form).
pub fn reservations(itin: &ItineraryInfo) -> Vec<Reservation> {
    reservations_of(&itin.stages)
}

/// Name the units of the decoder's register-aware resolved stages. `units` is
/// `TblgenOutput::functional_units`; a resolved stage's `unit` indexes it.
pub fn stages_from_resolved(units: &[String], resolved: &[ResolvedStage]) -> Vec<PipelineStage> {
    resolved
        .iter()
        .map(|r| PipelineStage {
            cycles: r.cycles,
            units: vec![units[r.unit as usize].clone()],
            time_inc: r.next_cycles as i8,
            reserved: r.reserved,
        })
        .collect()
}

/// One instruction of the checked sequence: a label for reports and what it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Issued {
    pub label: String,
    pub reservations: Vec<Reservation>,
}

/// Where an instruction sits in the checked sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpRef {
    /// Bundle index; one bundle issues per cycle.
    pub bundle: usize,
    /// Position inside the bundle.
    pub op: usize,
    pub label: String,
    /// Cycles after this instruction's issue at which it holds the unit.
    pub offset: u32,
}

/// Two instructions holding one unit in one cycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Collision {
    pub unit: String,
    /// Absolute cycle (bundle index of cycle 0 is the first bundle).
    pub cycle: u32,
    /// The earlier-issued instruction (lower bundle, or lower position in the same bundle).
    pub first: OpRef,
    pub second: OpRef,
}

/// Report every cycle in which two instructions of `bundles` hold the same unit. Two
/// `Reserved` stages never collide with each other.
pub fn find_collisions(bundles: &[Vec<Issued>]) -> Vec<Collision> {
    // (unit, absolute cycle) -> holders in issue order, with their stage kind.
    let mut holders: BTreeMap<(&str, u32), Vec<(OpRef, bool)>> = BTreeMap::new();
    for (b, bundle) in bundles.iter().enumerate() {
        for (o, op) in bundle.iter().enumerate() {
            for r in &op.reservations {
                let who = OpRef { bundle: b, op: o, label: op.label.clone(), offset: r.offset };
                holders
                    .entry((r.unit.as_str(), b as u32 + r.offset))
                    .or_default()
                    .push((who, r.reserved));
            }
        }
    }
    let mut out = Vec::new();
    for ((unit, cycle), hs) in holders {
        for (i, (a, a_res)) in hs.iter().enumerate() {
            for (b, b_res) in &hs[i + 1..] {
                let same_op = a.bundle == b.bundle && a.op == b.op;
                if same_op || (*a_res && *b_res) {
                    continue;
                }
                out.push(Collision { unit: unit.to_string(), cycle, first: a.clone(), second: b.clone() });
            }
        }
    }
    out.sort_by(|x, y| {
        (x.cycle, &x.unit, x.first.bundle, x.second.bundle).cmp(&(
            y.cycle,
            &y.unit,
            y.first.bundle,
            y.second.bundle,
        ))
    });
    out
}

/// For every unit, the classes that hold it and at which stage offsets, from the full model.
pub fn unit_users(
    itineraries: &HashMap<String, ItineraryInfo>,
) -> BTreeMap<String, BTreeMap<String, Vec<u32>>> {
    let mut out: BTreeMap<String, BTreeMap<String, Vec<u32>>> = BTreeMap::new();
    for (class, itin) in itineraries {
        for r in reservations(itin) {
            out.entry(r.unit).or_default().entry(class.clone()).or_default().push(r.offset);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::load_from_generated;
    use super::*;

    fn stage(cycles: u8, unit: &str, time_inc: i8) -> PipelineStage {
        PipelineStage { cycles, units: vec![unit.to_string()], time_inc, reserved: false }
    }

    fn offsets(stages: &[PipelineStage]) -> Vec<(String, u32)> {
        reservations_of(stages).into_iter().map(|r| (r.unit, r.offset)).collect()
    }

    #[test]
    fn stage_arithmetic_follows_llvm_instr_stage() {
        // SimpleCycle advances by cycles, EmptyCycles<n> is 0 cycles advancing n,
        // PrefixCycle advances 0 (the next stage shares its cycle).
        let stages = vec![
            stage(1, "A", -1),
            stage(0, "EMPTY_FU", 2),
            stage(2, "B", -1),
            stage(1, "C", 0),
            stage(1, "D", -1),
        ];
        assert_eq!(
            offsets(&stages),
            vec![("A".into(), 0), ("B".into(), 3), ("B".into(), 4), ("C".into(), 5), ("D".into(), 5)]
        );
    }

    #[test]
    fn multi_cycle_stage_holds_every_cycle() {
        assert_eq!(offsets(&[stage(3, "S", -1)]), vec![("S".into(), 0), ("S".into(), 1), ("S".into(), 2)]);
    }

    #[test]
    fn resolved_unit_indices_are_named_by_list_position() {
        let units: Vec<String> = ["EMPTY_FU", "P", "Q"].iter().map(|s| s.to_string()).collect();
        let resolved = [
            ResolvedStage { unit: 1, cycles: 1, next_cycles: 1, reserved: false },
            ResolvedStage { unit: 0, cycles: 0, next_cycles: 3, reserved: false },
            ResolvedStage { unit: 2, cycles: 1, next_cycles: 1, reserved: true },
        ];
        let stages = stages_from_resolved(&units, &resolved);
        assert_eq!(offsets(&stages), vec![("P".into(), 0), ("Q".into(), 4)]);
        assert!(reservations_of(&stages)[1].reserved);
    }

    fn model() -> HashMap<String, ItineraryInfo> {
        load_from_generated().itineraries
    }

    /// Two classes of the real model, `gap` bundles apart.
    fn two(model: &HashMap<String, ItineraryInfo>, a: &str, b: &str, gap: usize) -> Vec<Collision> {
        let issue = |c: &str| Issued { label: c.into(), reservations: reservations(&model[c]) };
        let mut bundles = vec![vec![]; gap + 1];
        bundles[0].push(issue(a));
        bundles[gap].push(issue(b));
        find_collisions(&bundles)
    }

    fn only(r: &[Collision]) -> &Collision {
        assert_eq!(r.len(), 1, "{r:?}");
        &r[0]
    }

    #[test]
    fn flags_move_source_read_against_accumulator_read() {
        // vadd (II_VACC) reads its accumulator sources through CM_RM_PORT in cycle 2; a vmov
        // cm,cm (II_VMOV_CM) issued two bundles later reads its source there in its cycle 0.
        let m = model();
        let r = two(&m, "II_VACC", "II_VMOV_CM", 2);
        let c = only(&r);
        assert_eq!((c.unit.as_str(), c.cycle), ("CM_RM_PORT", 2));
        assert_eq!((c.first.label.as_str(), c.first.offset), ("II_VACC", 2));
        assert_eq!((c.second.label.as_str(), c.second.offset), ("II_VMOV_CM", 0));
        // One cycle either side is clean.
        assert!(two(&m, "II_VACC", "II_VMOV_CM", 1).is_empty());
        assert!(two(&m, "II_VACC", "II_VMOV_CM", 3).is_empty());
    }

    #[test]
    fn flags_clear_write_against_float_mac_write() {
        // II_VCLR writes through CM_WA_PORT in cycle 4, the float add-mac in cycle 5.
        let m = model();
        let r = two(&m, "II_VADDMACf", "II_VCLR", 1);
        let c = only(&r);
        assert_eq!((c.unit.as_str(), c.cycle), ("CM_WA_PORT", 5));
        assert!(two(&m, "II_VADDMACf", "II_VCLR", 0).is_empty());
        assert!(two(&m, "II_VADDMACf", "II_VCLR", 2).is_empty());
    }

    #[test]
    fn same_register_write_landing_together_is_not_a_unit_collision() {
        // vlda.ups (def 9) and vaddmac.f (def 6) land on one register in the same cycle when
        // the load is issued three bundles before the MAC, but they write through different
        // ports (CM_WM_PORT vs CM_WA_PORT): the reservation tables do not see this.
        assert!(two(&model(), "II_VLDA_UPS", "II_VADDMACf", 3).is_empty());
    }

    #[test]
    fn reserved_stages_do_not_collide_with_each_other() {
        let res = |reserved| Issued {
            label: "x".into(),
            reservations: vec![Reservation { unit: "U".into(), offset: 0, reserved }],
        };
        assert!(find_collisions(&[vec![res(true), res(true)]]).is_empty());
        assert_eq!(find_collisions(&[vec![res(true), res(false)]]).len(), 1);
        assert_eq!(find_collisions(&[vec![res(false), res(false)]]).len(), 1);
    }

    #[test]
    fn an_instruction_never_collides_with_itself() {
        let two_cycles = Issued {
            label: "x".into(),
            reservations: vec![
                Reservation { unit: "U".into(), offset: 0, reserved: false },
                Reservation { unit: "U".into(), offset: 0, reserved: false },
            ],
        };
        assert!(find_collisions(&[vec![two_cycles]]).is_empty());
    }

    #[test]
    fn no_stage_offers_a_choice_of_units() {
        for (class, i) in &model() {
            for s in &i.stages {
                assert!(s.units.len() <= 1, "{class}: {:?}", s.units);
            }
        }
    }

    #[test]
    fn functional_unit_list_names_every_unit_the_itineraries_use() {
        let out = load_from_generated();
        for (class, i) in &out.itineraries {
            for s in &i.stages {
                for u in &s.units {
                    assert!(out.functional_units.contains(u), "{class}: {u}");
                }
            }
        }
    }
}
