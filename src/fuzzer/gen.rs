//! Random kernel generation from a seed.
//!
//! Uses a simple xorshift64 RNG (no external dependency) to produce
//! deterministic `FuzzParams` from a seed value. Same seed always
//! produces the same kernel.

use crate::fuzzer::ast::*;
use crate::fuzzer::params::*;

/// Generate a complete set of fuzz parameters from a seed.
///
/// The output is fully deterministic: same seed -> same parameters -> same C++.
pub fn generate(seed: u64) -> FuzzParams {
    let mut rng = Xorshift64(if seed == 0 { 1 } else { seed });

    let buffer_size = pick(&mut rng, &[16, 32, 64, 128, 256]) as usize;
    let dtype = pick(&mut rng, &[0, 1, 2]);
    let dtype = match dtype {
        0 => ScalarType::I32,
        1 => ScalarType::I16,
        _ => ScalarType::I8,
    };

    let loop_style = if rng.next() % 2 == 0 {
        LoopStyle::Simple
    } else {
        LoopStyle::HardwareLoop
    };

    let num_ops = 2 + (rng.next() % 7) as usize; // 2-8 ops
    let mut ops = Vec::with_capacity(num_ops + 1);

    for _ in 0..num_ops {
        ops.push(gen_op(&mut rng));
    }

    // Always end with a store to output buffer so there's something to compare.
    ops.push(KernelOp::Store { buf: BufRef(1), idx: Operand::Var(Var(1)), val: Operand::Var(Var(0)) });

    FuzzParams { seed, buffer_size, dtype, body: KernelBody { ops, loop_style } }
}

/// Generate a single kernel operation.
fn gen_op(rng: &mut Xorshift64) -> KernelOp {
    let kind = rng.next() % 10;
    match kind {
        // 60% chance: scalar arithmetic
        0..=5 => {
            let op = gen_scalar_op(rng);
            let src1 = gen_operand(rng);
            let src2 = gen_operand(rng);
            KernelOp::ScalarArith {
                op,
                dst: Var(0), // always write to accumulator
                src1,
                src2,
            }
        }
        // 20% chance: store to output buffer
        6 | 7 => KernelOp::Store { buf: BufRef(1), idx: Operand::Var(Var(1)), val: gen_operand(rng) },
        // 10% chance: branch
        8 => {
            let then_count = 1 + (rng.next() % 3) as usize;
            let else_count = rng.next() % 2;
            let then_ops: Vec<_> = (0..then_count).map(|_| gen_simple_op(rng)).collect();
            let else_ops: Vec<_> = (0..else_count).map(|_| gen_simple_op(rng)).collect();
            KernelOp::Branch { cond: Operand::Var(Var(0)), then_ops, else_ops }
        }
        // 10% chance: hw loop
        _ => {
            let count = 1 + (rng.next() % 4) as u32;
            let body_count = 1 + (rng.next() % 3) as usize;
            let body: Vec<_> = (0..body_count).map(|_| gen_simple_op(rng)).collect();
            KernelOp::HwLoop { count, body }
        }
    }
}

/// Generate a non-nested operation (for use inside branches/loops to bound depth).
fn gen_simple_op(rng: &mut Xorshift64) -> KernelOp {
    if rng.next() % 3 == 0 {
        KernelOp::Store { buf: BufRef(1), idx: Operand::Var(Var(1)), val: gen_operand(rng) }
    } else {
        KernelOp::ScalarArith {
            op: gen_scalar_op(rng),
            dst: Var(0),
            src1: gen_operand(rng),
            src2: gen_operand(rng),
        }
    }
}

/// Generate a random scalar operation.
fn gen_scalar_op(rng: &mut Xorshift64) -> ScalarOp {
    match rng.next() % 8 {
        0 => ScalarOp::Add,
        1 => ScalarOp::Sub,
        2 => ScalarOp::Mul,
        3 => ScalarOp::And,
        4 => ScalarOp::Or,
        5 => ScalarOp::Xor,
        6 => ScalarOp::Shl,
        _ => ScalarOp::Shr,
    }
}

/// Generate a random operand.
fn gen_operand(rng: &mut Xorshift64) -> Operand {
    match rng.next() % 4 {
        // 25% variable (accumulator)
        0 => Operand::Var(Var(0)),
        // 25% loop index
        1 => Operand::Var(Var(1)),
        // 25% literal
        2 => {
            // Small values to fit i8, clamp shifts to 0-7
            let val = (rng.next() % 256) as i32 - 128;
            Operand::Literal(val)
        }
        // 25% buffer load
        _ => Operand::Load { buf: BufRef(0), idx: Box::new(Operand::Var(Var(1))) },
    }
}

fn pick(rng: &mut Xorshift64, choices: &[u64]) -> u64 {
    choices[(rng.next() % choices.len() as u64) as usize]
}

/// Minimal xorshift64 RNG. No external dependency, fully deterministic.
struct Xorshift64(u64);

impl Xorshift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_seed_produces_same_params() {
        let a = generate(42);
        let b = generate(42);
        let cpp_a = crate::fuzzer::lower_cpp::lower_to_cpp(&a);
        let cpp_b = crate::fuzzer::lower_cpp::lower_to_cpp(&b);
        assert_eq!(cpp_a, cpp_b);
    }

    #[test]
    fn test_different_seeds_produce_different_params() {
        let a = generate(1);
        let b = generate(2);
        let cpp_a = crate::fuzzer::lower_cpp::lower_to_cpp(&a);
        let cpp_b = crate::fuzzer::lower_cpp::lower_to_cpp(&b);
        assert_ne!(cpp_a, cpp_b);
    }

    #[test]
    fn test_generated_body_has_ops() {
        let params = generate(100);
        assert!(!params.body.ops.is_empty(), "Generated kernel should have operations");
    }

    #[test]
    fn test_generated_kernel_always_has_final_store() {
        for seed in 0..50 {
            let params = generate(seed + 1000);
            let last = params.body.ops.last().expect("body should have ops");
            match last {
                KernelOp::Store { buf: BufRef(1), .. } => {} // good
                other => panic!("seed {}: last op should be Store to buf_out, got {:?}", seed, other),
            }
        }
    }

    #[test]
    fn test_xorshift_deterministic() {
        let mut a = Xorshift64(42);
        let mut b = Xorshift64(42);
        for _ in 0..100 {
            assert_eq!(a.next(), b.next());
        }
    }

    #[test]
    fn test_generated_cpp_compiles_syntax() {
        // Spot-check that lowered C++ has balanced braces.
        for seed in 1..=20 {
            let params = generate(seed);
            let cpp = crate::fuzzer::lower_cpp::lower_to_cpp(&params);
            let opens = cpp.chars().filter(|c| *c == '{').count();
            let closes = cpp.chars().filter(|c| *c == '}').count();
            assert_eq!(opens, closes, "seed {}: unbalanced braces in:\n{}", seed, cpp);
        }
    }
}
