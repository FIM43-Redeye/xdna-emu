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

use super::types::ItineraryInfo;

/// One unit held in one cycle, relative to the issue cycle of the instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reservation {
    pub unit: String,
    /// Cycles after issue.
    pub offset: u32,
    /// Stage kind `Reserved`: the unit is held but this stage does not require it free.
    pub reserved: bool,
}

/// The reservations of one itinerary, in stage order.
///
/// A stage that lists several units means "any one of them". Every stage the AIE2 model
/// defines names exactly one unit (asserted by `no_stage_offers_a_choice_of_units`), so a
/// stage with a choice is rejected rather than approximated.
pub fn reservations(itin: &ItineraryInfo) -> Vec<Reservation> {
    let mut out = Vec::new();
    let mut start: i32 = 0;
    for stage in &itin.stages {
        if stage.cycles > 0 {
            assert!(
                stage.units.len() == 1,
                "{}: stage with a choice of units {:?} is not modelled",
                itin.class_name,
                stage.units
            );
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

/// Where an instruction sits in the checked sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpRef {
    /// Bundle index; one bundle issues per cycle.
    pub bundle: usize,
    /// Position inside the bundle.
    pub op: usize,
    pub class: String,
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

#[derive(Debug, Default, PartialEq, Eq)]
pub struct CheckReport {
    pub collisions: Vec<Collision>,
    /// Instructions whose itinerary class has no record, so they reserve nothing.
    pub unmodelled: Vec<(usize, usize, String)>,
}

/// Report every cycle in which two instructions of `bundles` (itinerary class names, one list
/// per bundle) hold the same unit. Two `Reserved` stages never collide with each other.
pub fn find_collisions(itineraries: &HashMap<String, ItineraryInfo>, bundles: &[Vec<String>]) -> CheckReport {
    let mut report = CheckReport::default();
    let mut cache: HashMap<&str, Vec<Reservation>> = HashMap::new();
    // (unit, absolute cycle) -> holders in issue order, with their stage kind.
    let mut holders: BTreeMap<(String, u32), Vec<(OpRef, bool)>> = BTreeMap::new();
    for (b, bundle) in bundles.iter().enumerate() {
        for (o, class) in bundle.iter().enumerate() {
            let Some(itin) = itineraries.get(class) else {
                report.unmodelled.push((b, o, class.clone()));
                continue;
            };
            let res = cache.entry(class.as_str()).or_insert_with(|| reservations(itin));
            for r in res.iter() {
                let cycle = b as u32 + r.offset;
                let who = OpRef { bundle: b, op: o, class: class.clone(), offset: r.offset };
                holders.entry((r.unit.clone(), cycle)).or_default().push((who, r.reserved));
            }
        }
    }
    for ((unit, cycle), hs) in holders {
        for (i, (a, a_res)) in hs.iter().enumerate() {
            for (b, b_res) in &hs[i + 1..] {
                let same_op = a.bundle == b.bundle && a.op == b.op;
                if same_op || (*a_res && *b_res) {
                    continue;
                }
                report.collisions.push(Collision {
                    unit: unit.clone(),
                    cycle,
                    first: a.clone(),
                    second: b.clone(),
                });
            }
        }
    }
    report.collisions.sort_by(|x, y| {
        (x.cycle, &x.unit, x.first.bundle, x.second.bundle).cmp(&(
            y.cycle,
            &y.unit,
            y.first.bundle,
            y.second.bundle,
        ))
    });
    report
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
    use super::super::types::PipelineStage;
    use super::*;

    fn stage(cycles: u8, unit: &str, time_inc: i8) -> PipelineStage {
        PipelineStage { cycles, units: vec![unit.to_string()], time_inc, reserved: false }
    }

    fn itin(class: &str, stages: Vec<PipelineStage>) -> ItineraryInfo {
        ItineraryInfo {
            class_name: class.to_string(),
            total_latency: 0,
            operand_cycles: vec![],
            stages,
            bypasses: vec![],
        }
    }

    fn offsets(i: &ItineraryInfo) -> Vec<(String, u32)> {
        reservations(i).into_iter().map(|r| (r.unit, r.offset)).collect()
    }

    #[test]
    fn stage_arithmetic_follows_llvm_instr_stage() {
        // SimpleCycle advances by cycles, EmptyCycles<n> is 0 cycles advancing n,
        // PrefixCycle advances 0 (the next stage shares its cycle).
        let i = itin(
            "T",
            vec![
                stage(1, "A", -1),
                stage(0, "EMPTY_FU", 2),
                stage(2, "B", -1),
                stage(1, "C", 0),
                stage(1, "D", -1),
            ],
        );
        assert_eq!(
            offsets(&i),
            vec![("A".into(), 0), ("B".into(), 3), ("B".into(), 4), ("C".into(), 5), ("D".into(), 5)]
        );
    }

    fn two(model: &HashMap<String, ItineraryInfo>, a: &str, b: &str, gap: usize) -> CheckReport {
        let mut bundles = vec![vec![]; gap + 1];
        bundles[0].push(a.to_string());
        bundles[gap].push(b.to_string());
        find_collisions(model, &bundles)
    }

    fn model() -> HashMap<String, ItineraryInfo> {
        load_from_generated().itineraries
    }

    fn only(r: &CheckReport) -> &Collision {
        assert_eq!(r.collisions.len(), 1, "{:?}", r.collisions);
        &r.collisions[0]
    }

    #[test]
    fn flags_move_source_read_against_accumulator_read() {
        // vadd (II_VACC) reads its accumulator sources through CM_RM_PORT in cycle 2; a vmov
        // cm,cm (II_VMOV_CM) issued two bundles later reads its source there in its cycle 0.
        let m = model();
        let r = two(&m, "II_VACC", "II_VMOV_CM", 2);
        let c = only(&r);
        assert_eq!((c.unit.as_str(), c.cycle), ("CM_RM_PORT", 2));
        assert_eq!((c.first.class.as_str(), c.first.offset), ("II_VACC", 2));
        assert_eq!((c.second.class.as_str(), c.second.offset), ("II_VMOV_CM", 0));
        // One cycle either side is clean.
        assert!(two(&m, "II_VACC", "II_VMOV_CM", 1).collisions.is_empty());
        assert!(two(&m, "II_VACC", "II_VMOV_CM", 3).collisions.is_empty());
    }

    #[test]
    fn flags_clear_write_against_float_mac_write() {
        // II_VCLR writes through CM_WA_PORT in cycle 4, the float add-mac in cycle 5.
        let m = model();
        let r = two(&m, "II_VADDMACf", "II_VCLR", 1);
        let c = only(&r);
        assert_eq!((c.unit.as_str(), c.cycle), ("CM_WA_PORT", 5));
        assert!(two(&m, "II_VADDMACf", "II_VCLR", 0).collisions.is_empty());
        assert!(two(&m, "II_VADDMACf", "II_VCLR", 2).collisions.is_empty());
    }

    #[test]
    fn same_register_write_landing_together_is_not_a_unit_collision() {
        // vlda.ups (def 9) and vaddmac.f (def 6) land on one register in the same cycle when
        // the load is issued three bundles before the MAC, but they write through different
        // ports (CM_WM_PORT vs CM_WA_PORT): the reservation tables do not see this.
        let m = model();
        assert!(two(&m, "II_VLDA_UPS", "II_VADDMACf", 3).collisions.is_empty());
    }

    #[test]
    fn reserved_stages_do_not_collide_with_each_other() {
        let mut m = HashMap::new();
        let mut s = stage(1, "U", -1);
        s.reserved = true;
        m.insert("R".to_string(), itin("R", vec![s]));
        m.insert("Q".to_string(), itin("Q", vec![stage(1, "U", -1)]));
        let rr = find_collisions(&m, &[vec!["R".into(), "R".into()]]);
        assert!(rr.collisions.is_empty());
        let rq = find_collisions(&m, &[vec!["R".into(), "Q".into()]]);
        assert_eq!(rq.collisions.len(), 1);
    }

    #[test]
    fn unknown_class_is_reported_not_ignored() {
        let r = find_collisions(&model(), &[vec!["II_NOT_A_CLASS".into()]]);
        assert_eq!(r.unmodelled, vec![(0, 0, "II_NOT_A_CLASS".to_string())]);
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
    fn multi_cycle_stage_holds_every_cycle() {
        let i = itin("M", vec![stage(3, "S", -1)]);
        assert_eq!(offsets(&i), vec![("S".into(), 0), ("S".into(), 1), ("S".into(), 2)]);
    }
}
