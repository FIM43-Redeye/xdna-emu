//! Deterministic, target-driven scalar chain generation.
//!
//! [`generate`] turns `(seed, target_key)` into a [`ScalarChain`] guaranteed to
//! exercise the target: an arith/branch target forces a stage of that op at a
//! random slot; a `loop_*` target forces the outer loop style. Operands draw
//! from the input buffer, an earlier stage's register, or a literal -- the
//! `Prior(j)` references are always backward (`j < k`).

use super::chain::{LoopStyle, Operand, ScalarChain, ScalarOp, ScalarStage, StageOp};
use super::table::{parse_key, TargetKind};

/// xorshift64 PRNG (same as the vector/scalar fuzzers'): zero state forbidden.
pub(crate) struct Xorshift64(pub u64);

impl Xorshift64 {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

const REGION_LENS: [usize; 3] = [16, 32, 64];

/// Generate one chain deterministically from `(seed, target_key)`.
pub fn generate(seed: u64, target_key: &str) -> ScalarChain {
    let target = parse_key(target_key);
    let mut rng = Xorshift64(if seed == 0 { 1 } else { seed });

    let region_len = REGION_LENS[(rng.next() % REGION_LENS.len() as u64) as usize];

    let loop_style = match target.kind {
        TargetKind::LoopSimple => LoopStyle::Simple,
        TargetKind::LoopHw => LoopStyle::HardwareLoop,
        TargetKind::Stage(_) => {
            if rng.next() % 2 == 0 {
                LoopStyle::Simple
            } else {
                LoopStyle::HardwareLoop
            }
        }
    };

    let total = 8 + (rng.next() % 9) as usize; // 8-16 stages
    let target_slot = (rng.next() % total as u64) as usize;
    let forced = match target.kind {
        TargetKind::Stage(op) => Some(op),
        _ => None,
    };

    let mut stages = Vec::with_capacity(total);
    for k in 0..total {
        let op = if k == target_slot {
            forced.unwrap_or_else(|| StageOp::Arith(rand_arith(&mut rng)))
        } else {
            StageOp::Arith(rand_arith(&mut rng))
        };
        let a = rand_operand(&mut rng, k, op, false);
        let b = rand_operand(&mut rng, k, op, true);
        let cond = rand_operand(&mut rng, k, op, false);
        stages.push(ScalarStage { op, a, b, cond });
    }

    ScalarChain {
        seed,
        target_key: target_key.to_string(),
        dtype: target.dtype,
        region_len,
        loop_style,
        stages,
    }
}

fn rand_arith(rng: &mut Xorshift64) -> ScalarOp {
    ScalarOp::all()[(rng.next() % 8) as usize]
}

/// Pick an operand for stage `k`. `is_shift_amount` clamps a literal to `0..=7`
/// when the op is a shift and this is the second (amount) operand, so shifts
/// stay defined for every dtype. `Prior` is only offered when `k > 0`.
fn rand_operand(rng: &mut Xorshift64, k: usize, op: StageOp, is_shift_amount: bool) -> Operand {
    let is_shift = matches!(op, StageOp::Arith(ScalarOp::Shl) | StageOp::Arith(ScalarOp::Shr));
    if is_shift_amount && is_shift {
        return Operand::Literal((rng.next() % 8) as i32);
    }
    match rng.next() % 3 {
        0 => Operand::Input,
        1 if k > 0 => Operand::Prior((rng.next() % k as u64) as usize),
        1 => Operand::Input,
        _ => Operand::Literal((rng.next() % 256) as i32 - 128),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::domains::scalar::chain::{Dtype, LoopStyle, Operand, ScalarOp, StageOp};
    use crate::fuzzer::domains::scalar::table::universe_keys;

    #[test]
    fn same_seed_and_key_identical_chain() {
        for key in universe_keys().iter().step_by(5) {
            assert_eq!(generate(7, key), generate(7, key), "key {key}");
        }
    }

    #[test]
    fn stage_count_in_8_to_16() {
        for (seed, key) in (0..400u64).zip(universe_keys().into_iter().cycle()) {
            let c = generate(seed, &key);
            assert!((8..=16).contains(&c.stages.len()), "seed {seed} key {key}: {} stages", c.stages.len());
        }
    }

    #[test]
    fn prior_operands_reference_only_earlier_stages() {
        for (seed, key) in (0..400u64).zip(universe_keys().into_iter().cycle()) {
            let c = generate(seed, &key);
            for (k, s) in c.stages.iter().enumerate() {
                for op in [s.a, s.b, s.cond] {
                    if let Operand::Prior(j) = op {
                        assert!(j < k, "seed {seed} key {key}: stage {k} references Prior({j})");
                    }
                }
            }
        }
    }

    #[test]
    fn arith_target_key_is_present_in_keys() {
        for key in universe_keys().iter().filter(|k| !k.starts_with("loop_")) {
            let c = generate(11, key);
            assert!(c.keys().contains(key), "key {key} not in {:?}", c.keys());
        }
    }

    #[test]
    fn loop_target_sets_the_loop_style() {
        for d in ["I32", "I16", "I8"] {
            let cs = generate(3, &format!("loop_simple/{d}"));
            assert_eq!(cs.loop_style, LoopStyle::Simple);
            let ch = generate(3, &format!("loop_hw/{d}"));
            assert_eq!(ch.loop_style, LoopStyle::HardwareLoop);
        }
    }

    #[test]
    fn dtype_follows_target() {
        assert_eq!(generate(1, "add/I8").dtype, Dtype::I8);
        assert_eq!(generate(1, "branch/I16").dtype, Dtype::I16);
    }

    #[test]
    fn shift_literals_are_bounded() {
        for (seed, key) in (0..400u64).zip(universe_keys().into_iter().cycle()) {
            for s in generate(seed, &key).stages {
                if matches!(s.op, StageOp::Arith(op) if matches!(op, ScalarOp::Shl | ScalarOp::Shr)) {
                    if let Operand::Literal(n) = s.b {
                        assert!((0..=7).contains(&n), "seed {seed} key {key}: shift literal {n}");
                    }
                }
            }
        }
    }
}
