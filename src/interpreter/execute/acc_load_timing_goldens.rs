//! Silicon thresholds of the plain accumulator loads (`vlda amll` / `vlda amlh`), replayed through
//! the cycle-accurate executor.
//!
//! `tools/gen_acc_timing_probe.py` swept the loads as producers against every accumulator reader
//! and as one of two writers of the same register (`--waw`), on NPU1 silicon twice with identical
//! results (`build/experiments/harness/cfloat/ldtiming`). Every threshold follows the accumulator
//! file's one rule with the load's itinerary result cycle (II_VLDA_AM: 7), exactly like the MAC
//! results: a reader with use cycle `u` sees the load from distance `8 - u` on, and of two writes the
//! one landing later wins, the later-issued one on a tie. Each test runs a pair at the silicon
//! threshold and one bundle short of it and classifies the destination's lanes against the two
//! single-instruction references, as the probe's analyzer does.

use super::cycle_accurate::CycleAccurateExecutor;
use crate::device::tile::Tile;
use crate::interpreter::bundle::{SlotOp, VliwBundle};
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::traits::{Decoder, Executor};

const VLDA_AMLL: [u8; 4] = [0x99, 0x80, 0x02, 0x05]; // vlda amll0, [p5, #0]
const VLDA_AMLH: [u8; 4] = [0x99, 0x81, 0x06, 0x05]; // vlda amlh0, [p5, #32]
const VMUL_F: [u8; 4] = [0x49, 0xc0, 0xc8, 0x1f]; // vmul.f bml0, x2, x3, r3
const VMUL_I: [u8; 4] = [0x09, 0x40, 0x80, 0x0f]; // vmul cm0, x0, x1, r1
const VMOV_BM_X: [u8; 4] = [0xd9, 0x56, 0x2c, 0x18]; // vmov bml0, x10
const VLDA_UPS: [u8; 4] = [0x19, 0x41, 0x0f, 0x04]; // vlda.ups.s32.s16 bml0, s3, [p4, #0]
const VMAC_F_ACC1: [u8; 4] = [0x49, 0xd8, 0x08, 0x18]; // vmac.f bml6, bml0, x2, x3, r3
const VMOV_X_BM: [u8; 4] = [0xd9, 0x06, 0x00, 0x1c]; // vmov x8, bml0

/// MAC configuration words of the probe (`gen_acc_timing_probe.CONFIGS`): r1 dense i16 x i16,
/// r3 dense bf16.
const CONF_R1: u32 = 0x31a;
const CONF_R3: u32 = 0x31c;

const LOCAL_BASE: u32 = 0x70000;
const LOAD_AT: u32 = 0x400; // what the loads bring in (p5)
const UPS_AT: u32 = 0x500; // what vlda.ups brings in (p4)

fn op(decoder: &InstructionDecoder, bytes: &[u8]) -> SlotOp {
    decoder
        .decode(bytes, 0)
        .unwrap()
        .slots()
        .iter()
        .flatten()
        .find(|s| !s.is_nop())
        .unwrap()
        .clone()
}

/// Distinct, finite data everywhere a lane can come from: the OLD accumulator, the operands, and
/// both load sources.
fn start_state() -> (ExecutionContext, Tile) {
    let mut ctx = ExecutionContext::new();
    let mut tile = Tile::compute(0, 2);
    let mem = tile.data_memory_mut();
    for k in 0..16u32 {
        let load = (0x4140 + k) << 16; // bf16-exact floats
        mem[(LOAD_AT + 4 * k) as usize..][..4].copy_from_slice(&load.to_le_bytes());
        let ups = 0x0300 + k;
        mem[(UPS_AT + 4 * k) as usize..][..4].copy_from_slice(&ups.to_le_bytes());
    }
    ctx.pointer.write(5, LOCAL_BASE + LOAD_AT);
    ctx.pointer.write(4, LOCAL_BASE + UPS_AT);
    ctx.scalar.write(1, CONF_R1);
    ctx.scalar.write(3, CONF_R3);
    let fill = |base: u32| -> [u32; 8] { std::array::from_fn(|i| base + i as u32) };
    for (reg, base) in [(0u8, 0x0001_0002), (1, 0x0003_0001), (2, 0x0002_0001), (3, 0x0001_0003)] {
        // x<n> = {wl<n>, wh<n>}: small ints that also read as small bf16 pairs
        ctx.vector.write(2 * reg, fill(base));
        ctx.vector.write(2 * reg + 1, fill(base + 0x10));
    }
    let bf16_pair = |v: u32| v << 16 | v;
    ctx.vector.write(4, [bf16_pair(0x3F80); 8]); // x2 = 1.0
    ctx.vector.write(5, [bf16_pair(0x3F80); 8]);
    ctx.vector.write(6, [bf16_pair(0x4000); 8]); // x3 = 2.0
    ctx.vector.write(7, [bf16_pair(0x4000); 8]);
    ctx.vector.write(20, fill(0x4280_0000)); // x10
    ctx.vector.write(21, fill(0x4290_0000));
    for r in [0u8, 1, 12, 13] {
        // bml0/bmh0 and bml6/bmh6 start as OLD
        ctx.accumulator
            .write(r, std::array::from_fn(|i| 0x4300_0000_4310_0000 + ((r as u64) << 40) + i as u64));
    }
    (ctx, tile)
}

/// Run `first` and `second` `s` bundles apart (`s` >= 0: `second` issues `s` after `first`, 0 shares
/// a bundle), drain, and return register `out` (an accumulator index, or a vector index + 100).
fn run(first: Option<&[u8]>, second: Option<&[u8]>, s: usize, out: u8) -> Vec<u64> {
    let decoder = InstructionDecoder::load_default();
    let (mut ctx, mut tile) = start_state();
    let mut executor = CycleAccurateExecutor::new();
    for _ in 0..4 {
        executor.execute(&VliwBundle::empty(), &mut ctx, &mut tile);
    }
    for at in 0..=s {
        let mut bundle = VliwBundle::empty();
        if let (0, Some(f)) = (at, first) {
            bundle.set_slot(op(&decoder, f));
        }
        if let (true, Some(sec)) = (at == s, second) {
            bundle.set_slot(op(&decoder, sec));
        }
        executor.execute(&bundle, &mut ctx, &mut tile);
    }
    for _ in 0..24 {
        executor.execute(&VliwBundle::empty(), &mut ctx, &mut tile);
    }
    if out >= 100 {
        let r = 2 * (out - 100);
        ctx.vector
            .read_store(r)
            .iter()
            .chain(ctx.vector.read_store(r + 1).iter())
            .map(|&w| w as u64)
            .collect()
    } else {
        ctx.accumulator.read_committed(out).to_vec()
    }
}

/// Classify every lane of `got` against the references: `O`, `N`, `.` (they agree) or `X`.
fn classify(got: &[u64], old: &[u64], new: &[u64]) -> String {
    got.iter()
        .zip(old.iter().zip(new))
        .map(|(g, (o, n))| match (o == n, g == o, g == n) {
            (true, _, _) => '.',
            (_, true, _) => 'O',
            (_, _, true) => 'N',
            _ => 'X',
        })
        .collect()
}

/// A WAW pair at signed distance `s` (the load issued `s` bundles after `writer`; the load wins from
/// `t` on): the classification of R's lanes, informative lanes only.
fn waw(load: &[u8], writer: &[u8], s: i32) -> String {
    let at = |s: i32| {
        if s >= 0 {
            run(Some(writer), Some(load), s as usize, 0)
        } else {
            run(Some(load), Some(writer), (-s) as usize, 0)
        }
    };
    // references: the writer landing long after the load (OLD), the load long after it (NEW)
    classify(&at(s), &at(-14), &at(14)).replace('.', "")
}

/// A RAW pair at distance `d`: the reader's destination classified against the reader alone (OLD)
/// and the reader long after the load (NEW).
fn raw(load: &[u8], reader: &[u8], d: usize, out: u8) -> String {
    let old = run(Some(reader), None, 0, out);
    let new = run(Some(load), Some(reader), 14, out);
    classify(&run(Some(load), Some(reader), d, out), &old, &new).replace('.', "")
}

fn all(s: &str, c: char) -> bool {
    !s.is_empty() && s.chars().all(|x| x == c)
}

#[test]
fn a_load_issued_with_a_float_mac_lands_after_it() {
    // the Chess cfloat mac_elem_8 schedule: `vlda amlh0 ; vmul.f bml0` in one bundle
    for load in [VLDA_AMLH, VLDA_AMLL] {
        let s0 = waw(&load, &VMUL_F, 0);
        let s1 = waw(&load, &VMUL_F, -1);
        assert!(all(&s0, 'N'), "s=0 must keep the load: {s0}");
        assert!(all(&s1, 'O'), "s=-1 ties, the later-issued vmul.f wins: {s1}");
    }
}

#[test]
fn load_waw_thresholds_match_silicon() {
    // (writer, t): the load wins from s = t on
    for (writer, t) in [(&VMUL_I, -1), (&VMOV_BM_X, -4), (&VLDA_UPS, 2)] {
        for load in [VLDA_AMLL, VLDA_AMLH] {
            let at = waw(&load, writer, t);
            let short = waw(&load, writer, t - 1);
            assert!(all(&at, 'N') && all(&short, 'O'), "t={t}: at {at} short {short}");
        }
    }
}

#[test]
fn a_mac_reads_a_load_from_distance_five_a_move_from_seven() {
    for load in [VLDA_AMLL, VLDA_AMLH] {
        let (at, short) = (raw(&load, &VMAC_F_ACC1, 5, 12), raw(&load, &VMAC_F_ACC1, 4, 12));
        assert!(all(&at, 'N') && all(&short, 'O'), "vmac.f acc1: at {at} short {short}");
        let (at, short) = (raw(&load, &VMOV_X_BM, 7, 108), raw(&load, &VMOV_X_BM, 6, 108));
        assert!(all(&at, 'N') && all(&short, 'O'), "vmov x,bm: at {at} short {short}");
    }
}
