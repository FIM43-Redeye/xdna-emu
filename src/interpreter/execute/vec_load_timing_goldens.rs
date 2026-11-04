//! Silicon behaviour of the vector-register loads, replayed through the cycle-accurate executor.
//!
//! `tools/gen_vec_load_timing_probe.py` swept every vector load (`vlda`/`vldb` into `wl`/`wh`,
//! post-modify, 2D, `vlda.128`, the fused `vldb.unpack` and the `vldb.4x` gather) against every
//! reader of the loaded register, as one of two writers of it, against writers of other
//! registers landing in the same cycle, and in one bundle with the instructions TableGen keeps
//! apart from it; on NPU1 silicon twice with identical results
//! (`build/experiments/harness/ldvec/{vt2,tie1,hz2}`). What the tests pin:
//!
//! * every reader sees a load from issue distance 7 (all vector reads are use cycle 1, the loads'
//!   result cycle is 7): the vector file's ordinary visibility rule, on the issued-bundle clock;
//! * the B load channel (`vldb` and every form of it: post-modify, 2D/3D, `.unpack`, `.4x`) holds
//!   no write port in the tables. In silicon it shares the bus of the vector file's last write
//!   port: landing in one cycle on the same register bank as a write through that port, the two
//!   are ORed; a write through the first port (the one `vlda` uses) beats it; writes to other
//!   registers or banks are untouched;
//! * a `vldb.4x` gather in one bundle with another reader of the W store-data read port (a `vst`
//!   of a W register, `vpack`) shares that port: both read the OR of the two registers; in one
//!   bundle with an A-channel load (both hold load unit A in one cycle) the load reads memory at
//!   the gather's address instead of its own.

use super::cycle_accurate::CycleAccurateExecutor;
use crate::device::tile::Tile;
use crate::interpreter::bundle::{SlotOp, VliwBundle};
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::traits::{Decoder, Executor};

const VLDA_WL4: [u8; 4] = [0xd9, 0x91, 0x02, 0x05]; // vlda wl4, [p5, #0]
const VLDB_WL4: [u8; 4] = [0x19, 0x08, 0x0d, 0x3d]; // vldb wl4, [p5, #0]
const VLDA_WL4_ALT: [u8; 4] = [0xd9, 0x91, 0x0e, 0x00]; // vlda wl4, [p0, #96]
const VMOV_WL4_WL0: [u8; 4] = [0x59, 0x07, 0x8c, 0x18]; // vmov wl4, wl0
const VMOV_WL8_WL0: [u8; 4] = [0x59, 0x07, 0x0c, 0x19]; // vmov wl8, wl0
const VST_WL4: [u8; 4] = [0xd9, 0x91, 0x02, 0x09]; // vst wl4, [p1, #0]
const VLDB_UNPACK: [u8; 4] = [0x19, 0x30, 0x05, 0x3d]; // vldb.unpack.s16.s8 x4, [p5, dj0]
const VSRS_WL4: [u8; 4] = [0x99, 0x49, 0x1d, 0x0a]; // vsrs.s16.s32 wl4, bml2, s1
const VLDB_4X: [u8; 4] = [0x19, 0x58, 0x8d, 0x3c]; // vldb.4x32.lo wl5, wl9
const LDA_R9: [u8; 4] = [0x59, 0x92, 0x62, 0x00]; // lda r9, [p0, #96]

const WL0: u8 = 0;
const WL4: u8 = 8;
const WH4: u8 = 9;
const WL5: u8 = 10;
const WL8: u8 = 16;
const WL9: u8 = 18;

const LOCAL_BASE: u32 = 0x70000;
const P0_AT: u32 = 0x100; // p0: [p0 + 96] is the alternative load source
const LOAD_AT: u32 = 0x400; // p5: what the loads bring in
const GATHER_AT: u32 = 0x600; // wl9 points here
const OUT_AT: u32 = 0x800; // p1

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

fn word_at(tile: &Tile, at: u32) -> u32 {
    u32::from_le_bytes(tile.data_memory()[at as usize..][..4].try_into().unwrap())
}

fn words_at(tile: &Tile, at: u32) -> [u32; 8] {
    std::array::from_fn(|k| word_at(tile, at + 4 * k as u32))
}

/// Distinct data in every source: memory items, the old registers, the gather's address vector.
fn start_state() -> (ExecutionContext, Tile) {
    let mut ctx = ExecutionContext::new();
    let mut tile = Tile::compute(0, 2);
    let mem = tile.data_memory_mut();
    for k in 0..64u32 {
        let put = |mem: &mut [u8], at: u32, v: u32| mem[at as usize..][..4].copy_from_slice(&v.to_le_bytes());
        put(mem, LOAD_AT + 4 * k, 0x0700_0100 + 0x0003_0003 * k);
        put(mem, P0_AT + 96 + 4 * k, 0x0050_0005 + k);
        put(mem, GATHER_AT + 4 * k, 0x2000_0000 + 0x11 * k);
    }
    ctx.pointer.write(0, LOCAL_BASE + P0_AT);
    ctx.pointer.write(1, LOCAL_BASE + OUT_AT);
    ctx.pointer.write(5, LOCAL_BASE + LOAD_AT);
    let fill = |base: u32| -> [u32; 8] { std::array::from_fn(|i| base + i as u32) };
    ctx.vector.write(WL0, fill(0x0000_0030));
    ctx.vector.write(WL0 + 1, fill(0x0000_0040));
    ctx.vector.write(WL4, fill(0x0000_0064));
    ctx.vector.write(WH4, fill(0x0000_006c));
    ctx.vector.write(WL5, fill(0x0000_0500));
    ctx.vector.write(WL8, fill(0x0000_0800));
    ctx.vector.write(WL9, [LOCAL_BASE + GATHER_AT; 8]);
    // bml2: what vsrs converts
    ctx.accumulator.write(4, std::array::from_fn(|i| 0x3e8 + i as u64));
    (ctx, tile)
}

/// Issue each `(bundle, encoding)` at its bundle (several may share one), drain, and return the
/// final state.
fn run(ops: &[(usize, [u8; 4])]) -> (ExecutionContext, Tile) {
    let decoder = InstructionDecoder::load_default();
    let (mut ctx, mut tile) = start_state();
    let mut executor = CycleAccurateExecutor::new();
    for _ in 0..4 {
        executor.execute(&VliwBundle::empty(), &mut ctx, &mut tile);
    }
    let last = ops.iter().map(|(at, _)| *at).max().unwrap_or(0);
    for at in 0..=last {
        let mut bundle = VliwBundle::empty();
        for (_, bytes) in ops.iter().filter(|(b, _)| *b == at) {
            bundle.set_slot(op(&decoder, bytes));
        }
        executor.execute(&bundle, &mut ctx, &mut tile);
    }
    for _ in 0..24 {
        executor.execute(&VliwBundle::empty(), &mut ctx, &mut tile);
    }
    (ctx, tile)
}

fn reg(ops: &[(usize, [u8; 4])], r: u8) -> [u32; 8] {
    run(ops).0.vector.read_store(r)
}

fn or(a: [u32; 8], b: [u32; 8]) -> [u32; 8] {
    std::array::from_fn(|k| a[k] | b[k])
}

#[test]
fn every_vector_load_is_read_from_distance_seven() {
    for load in [VLDA_WL4, VLDB_WL4] {
        let read_at = |d: usize| {
            let (_, tile) = run(&[(0, load), (d, VST_WL4)]);
            words_at(&tile, OUT_AT)
        };
        let loaded = words_at(&start_state().1, LOAD_AT);
        assert_eq!(read_at(7), loaded, "distance 7 sees the load");
        assert_eq!(read_at(6), start_state().0.vector.read(WL4), "distance 6 still sees OLD");
    }
}

#[test]
fn a_b_channel_load_landing_with_a_move_into_the_same_register_is_ored_with_it() {
    let loaded = words_at(&start_state().1, LOAD_AT);
    let moved = start_state().0.vector.read(WL0);
    // the move (result cycle 2) issued five bundles after the load lands in the same cycle
    assert_eq!(reg(&[(0, VLDB_WL4), (5, VMOV_WL4_WL0)], WL4), or(loaded, moved));
    // through the first write port the load wins the tie
    assert_eq!(reg(&[(0, VLDA_WL4), (5, VMOV_WL4_WL0)], WL4), loaded);
    // one bundle either side the later-landing write wins
    assert_eq!(reg(&[(0, VLDB_WL4), (4, VMOV_WL4_WL0)], WL4), loaded);
    assert_eq!(reg(&[(0, VLDB_WL4), (6, VMOV_WL4_WL0)], WL4), moved);
}

#[test]
fn a_b_channel_load_leaves_a_move_into_another_register_alone() {
    let (ctx, _) = run(&[(0, VLDB_WL4), (5, VMOV_WL8_WL0)]);
    assert_eq!(ctx.vector.read_store(WL4), words_at(&start_state().1, LOAD_AT));
    assert_eq!(ctx.vector.read_store(WL8), start_state().0.vector.read(WL0));
}

#[test]
fn a_vlda_beats_a_vldb_landing_in_the_same_register() {
    let alt = words_at(&start_state().1, P0_AT + 96);
    assert_eq!(reg(&[(0, VLDB_WL4), (0, VLDA_WL4_ALT)], WL4), alt);
}

#[test]
fn the_fused_unpack_lands_through_the_b_channel() {
    let (unpacked, _) = run(&[(0, VLDB_UNPACK)]);
    let (lo, hi) = (unpacked.vector.read_store(WL4), unpacked.vector.read_store(WH4));
    let moved = start_state().0.vector.read(WL0);
    let (ctx, _) = run(&[(0, VLDB_UNPACK), (5, VMOV_WL4_WL0)]);
    assert_eq!(ctx.vector.read_store(WL4), or(lo, moved), "ORed with the move on its bank");
    assert_eq!(ctx.vector.read_store(WH4), hi, "the other bank is the unpack's");
    // vsrs (result cycle 4, the first write port) beats it on its bank
    let srs = reg(&[(0, VSRS_WL4)], WL4);
    let (ctx, _) = run(&[(0, VLDB_UNPACK), (3, VSRS_WL4)]);
    assert_eq!(ctx.vector.read_store(WL4), srs);
    assert_eq!(ctx.vector.read_store(WH4), hi);
}

#[test]
fn a_gather_and_a_store_in_one_bundle_both_read_the_or_of_their_registers() {
    let s = start_state().0;
    let (old, addr) = (s.vector.read(WL4), s.vector.read(WL9));
    let (_, tile) = run(&[(0, VLDB_4X), (0, VST_WL4)]);
    assert_eq!(words_at(&tile, OUT_AT), or(old, addr), "the store writes the OR");
    // the gather follows the ORed addresses: the same as a gather from them alone
    let mut gathered_from_or = None;
    {
        let decoder = InstructionDecoder::load_default();
        let (mut ctx, mut tile) = start_state();
        ctx.vector.write(WL9, or(old, addr));
        let mut executor = CycleAccurateExecutor::new();
        let mut bundle = VliwBundle::empty();
        bundle.set_slot(op(&decoder, &VLDB_4X));
        executor.execute(&bundle, &mut ctx, &mut tile);
        for _ in 0..12 {
            executor.execute(&VliwBundle::empty(), &mut ctx, &mut tile);
        }
        gathered_from_or.replace(ctx.vector.read_store(WL5));
    }
    assert_eq!(reg(&[(0, VLDB_4X), (0, VST_WL4)], WL5), gathered_from_or.unwrap());
}

#[test]
fn an_a_channel_load_beside_a_gather_reads_at_the_gather_address() {
    let at_gather = words_at(&start_state().1, GATHER_AT);
    assert_eq!(reg(&[(0, VLDB_4X), (0, VLDA_WL4_ALT)], WL4), at_gather);
    let (ctx, _) = run(&[(0, VLDB_4X), (0, LDA_R9)]);
    assert_eq!(ctx.scalar.read(9), at_gather[0]);
    // the gather itself is unaffected
    assert_eq!(reg(&[(0, VLDB_4X), (0, VLDA_WL4_ALT)], WL5), reg(&[(0, VLDB_4X)], WL5));
}
