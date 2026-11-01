//! The accumulator-destination moves the collision probe's alone references showed the emulator
//! getting wrong, checked against NPU1 silicon (`probe3`/`probe7`, run twice, byte-identical):
//!
//! - `vmov am??, q1` copies the 128-bit mask register into the low 128 bits of an `am` quarter
//!   and zero-extends to 256 bits (all four quarters).
//! - `vshuffle bm?`, `vbcstshfl.* bm?` write their 512-bit vector result into a `bm` half as a
//!   raw reinterpretation, the layout `vmov bml, x` uses (u64 lane j = words 2j, 2j+1).
//!
//! Every case decodes the real instruction bytes, so the handler sees the operands the decoder
//! produces. Inputs are the probe's: x0, x1 and q1 from its input table, shuffle mode r1 = 794
//! (mode 26), broadcast scalar r12 = 9320 and shuffle mode r29 = 0. The destination starts as a
//! visible pattern so lanes the instruction must leave alone are checked too.

use super::vector_dispatch::VectorAlu;
use crate::interpreter::bundle::Operand;
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::traits::Decoder;

const Q1: [u32; 4] = [0xc9a4_7f5a, 0x5d38_13ee, 0xf1cc_a782, 0x8560_3b16];
const X0: [u32; 16] = [
    0x0002_0001,
    0x0004_0003,
    0x0006_0005,
    0x0001_0007,
    0x0003_0002,
    0x0005_0004,
    0x0007_0006,
    0x0002_0001,
    0x0004_0003,
    0x0006_0005,
    0x0001_0007,
    0x0003_0002,
    0x0005_0004,
    0x0007_0006,
    0x0002_0001,
    0x0004_0003,
];
const X1: [u32; 16] = [
    0x0003_0002,
    0x0005_0004,
    0x0002_0006,
    0x0004_0003,
    0x0006_0005,
    0x0003_0002,
    0x0005_0004,
    0x0002_0006,
    0x0004_0003,
    0x0006_0005,
    0x0003_0002,
    0x0005_0004,
    0x0002_0006,
    0x0004_0003,
    0x0006_0005,
    0x0003_0002,
];
const SHUFFLE_BM: [u32; 16] = [
    0x0003_0001,
    0x0003_0002,
    0x0004_0002,
    0x0004_0003,
    0x0005_0003,
    0x0005_0004,
    0x0006_0004,
    0x0006_0005,
    0x0007_0005,
    0x0002_0006,
    0x0001_0006,
    0x0003_0002,
    0x0002_0007,
    0x0004_0003,
    0x0003_0001,
    0x0005_0004,
];

/// The destination's starting contents: a pattern per u64 lane.
fn init_lanes() -> [u64; 8] {
    std::array::from_fn(|j| 0xE10 + j as u64)
}

fn bytes(hex: &str) -> Vec<u8> {
    hex.split_whitespace().map(|h| u8::from_str_radix(h, 16).unwrap()).collect()
}

/// Run one decoded instruction on the probe's inputs; the destination accumulator register and the
/// op's own text for failure messages.
fn run(hex: &str) -> (u8, [u64; 8]) {
    let decoder = InstructionDecoder::load_default();
    let bundle = decoder.decode(&bytes(hex), 0).unwrap_or_else(|e| panic!("{hex}: {e:?}"));
    let op = bundle
        .slots()
        .iter()
        .flatten()
        .find(|s| !s.is_nop())
        .unwrap_or_else(|| panic!("{hex}: no op"))
        .clone();
    let mut ctx = ExecutionContext::new();
    ctx.mask.write(1, Q1);
    ctx.scalar.write(1, 794);
    ctx.scalar.write(12, 9320);
    ctx.scalar.write(29, 0);
    let mut wide = [X0, X1].into_iter();
    for src in &op.sources {
        if let Operand::VectorReg(r) = src {
            ctx.vector.write_wide(*r & !1, wide.next().unwrap_or(X0));
        }
    }
    let Some(Operand::AccumReg(d)) = op.dest else {
        panic!("{hex}: destination {:?}", op.dest)
    };
    ctx.accumulator.write(d, init_lanes());
    ctx.result_latency = 2;
    assert!(VectorAlu::execute(&op, &mut ctx), "{hex}: not handled ({op:?})");
    ctx.accumulator.land_all_pending();
    (d, ctx.accumulator.read_committed(d))
}

fn words(lanes: &[u64; 8]) -> [u32; 16] {
    std::array::from_fn(|i| (lanes[i / 2] >> (32 * (i % 2))) as u32)
}

/// `[Q1 low 128 bits, zeros]` in a quarter's four lanes.
fn quarter_with_q1() -> [u64; 4] {
    [Q1[0] as u64 | (Q1[1] as u64) << 32, Q1[2] as u64 | (Q1[3] as u64) << 32, 0, 0]
}

fn check_quarter(hex: &str, base: usize) {
    let (_, got) = run(hex);
    let mut want = init_lanes();
    want[base..base + 4].copy_from_slice(&quarter_with_q1());
    assert_eq!(got, want, "{hex}: silicon {want:x?}, emulator {got:x?}");
}

#[test]
fn vmov_am_quarter_from_q_zero_extends_all_four_quarters() {
    // vmov amll0 / amlh0 / amhl0 / amhh0, q1 (probe7 refs; amll is probe3's)
    check_quarter("59 87 1e 18 37 88 03 00 00 00 00 00 00 00 00 00", 0);
    check_quarter("59 87 3e 18 37 88 03 00 00 00 00 00 00 00 00 00", 4);
    check_quarter("59 87 5e 18 37 88 03 00 00 00 00 00 00 00 00 00", 0);
    check_quarter("59 87 7e 18 37 88 03 00 00 00 00 00 00 00 00 00", 4);
}
