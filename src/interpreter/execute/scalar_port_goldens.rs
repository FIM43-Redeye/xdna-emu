//! Scalar write-port collisions, run end to end: real instruction bytes (assembled from the movd
//! probes, `build/experiments/harness/movd`) through the cycle-accurate executor. Each case is a
//! trial of the silicon probes (HW r1 == r2) with its observed outcome; the rules themselves are
//! pinned in `state::scalar_collision`, the derived ports in `decode::scalar_ports_goldens`.

use crate::device::tile::Tile;
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::traits::{Decoder, Executor};

use super::cycle_accurate::CycleAccurateExecutor;

const NOP: &str = "01 00";
const MOV_D1_R1_R2: &str = "59 0e 41 18";
const MOV_D2_R3_R4: &str = "59 1e c2 18";
const MOV_D3_R1_R2: &str = "59 0e 51 18";
const MOV_D6_R1_R2: &str = "59 1e 61 18";
const MOV_R5_R6: &str = "59 16 43 19";
const MOV_R1_R11: &str = "59 96 45 18";
const MOV_S0_R6: &str = "59 16 b3 18";
const MOV_P4_R6: &str = "59 16 33 1c";
const ADD_R5_R6_R9: &str = "99 90 8a 11";
const ADD_R1_R11_R9: &str = "99 90 c2 12";
const ADD_R3_R6_R9: &str = "99 90 86 11";
const MUL_R5_R6_R9: &str = "99 9f 8a 11";
const MOVA_R1_5: &str = "59 02 05 00";
const MOVXM_R5: &str = "55 aa 8a 52 55 55";
const VEXTRACT_R5_X0_R17: &str = "b9 86 4a 19";
const VLT_R16_X0_X1: &str = "d9 80 0c 18";
/// `mov.d6 r1, r2` and `lda r1, [p0, #8]` in one bundle.
const MOV_D6_WITH_LDA: &str = "4d 1e 61 48 50 01";

fn bytes(hex: &str) -> Vec<u8> {
    hex.split_whitespace().map(|h| u8::from_str_radix(h, 16).unwrap()).collect()
}

/// `r1 = 0x11, r2 = 0x22, r3 = 0x33, r4 = 0x44, r6 = 0x66, r9 = 1, r11 = 0x0f0f0f00` (and `init`
/// over it), word 0x0f0f0f00 at `[p0, #8]`; then the bundles one per issue slot and a drain of
/// nops (every delayed write lands).
fn run_with(init: &[(u8, u32)], program: &[&str]) -> ExecutionContext {
    let decoder = InstructionDecoder::load_default();
    let mut ctx = ExecutionContext::new();
    for (r, v) in [(1, 0x11), (2, 0x22), (3, 0x33), (4, 0x44), (6, 0x66), (9, 1), (11, 0x0f0f0f00)] {
        ctx.scalar.write(r, v);
    }
    for (r, v) in init {
        ctx.scalar.write(*r, *v);
    }
    let mut tile = Tile::compute(0, 2);
    ctx.pointer.write(0, 0x400);
    tile.write_data_u32(0x408, 0x0f0f0f00);
    let mut executor = CycleAccurateExecutor::new();
    for hex in program.iter().copied().chain(std::iter::repeat(NOP).take(12)) {
        let bundle = decoder.decode(&bytes(hex), 0).unwrap_or_else(|e| panic!("{hex}: {e:?}"));
        executor.execute(&bundle, &mut ctx, &mut tile);
    }
    ctx
}

fn run(program: &[&str]) -> ExecutionContext {
    run_with(&[], program)
}

/// `lead`, then `gap - 1` nops, then `second`: the second instruction issues `gap` bundles later.
fn run_pair_with(init: &[(u8, u32)], lead: &str, gap: usize, second: &str) -> ExecutionContext {
    let mut program = vec![lead];
    program.extend(std::iter::repeat(NOP).take(gap - 1));
    program.push(second);
    run_with(init, &program)
}

fn run_pair(lead: &str, gap: usize, second: &str) -> ExecutionContext {
    run_pair_with(&[], lead, gap, second)
}

// Silicon (movd p1): a plain mov issued exactly N bundles after mov.dN drops its write.
#[test]
fn mov_n_bundles_after_a_delayed_move_drops_it() {
    for (lead, n) in [(MOV_D1_R1_R2, 1), (MOV_D3_R1_R2, 3), (MOV_D6_R1_R2, 6)] {
        for second in [MOV_R5_R6, MOVXM_R5] {
            let ctx = run_pair(lead, n, second);
            assert_eq!(ctx.scalar.read(1), 0x11, "N={n}: dropped, r1 keeps its old value");
            assert_ne!(ctx.scalar.read(5), 0, "N={n}: the mover's own write survives");
            for gap in [n.saturating_sub(1).max(1), n + 1] {
                if gap != n {
                    assert_eq!(run_pair(lead, gap, second).scalar.read(1), 0x22, "N={n} k={gap}");
                }
            }
        }
    }
}

// Silicon (movd p1): a vector compare / extract holds the write unit one cycle later, so it
// collides one bundle earlier.
#[test]
fn a_vector_compare_one_bundle_early_drops_it() {
    for second in [VEXTRACT_R5_X0_R17, VLT_R16_X0_X1] {
        assert_eq!(run_pair(MOV_D3_R1_R2, 2, second).scalar.read(1), 0x11);
        assert_eq!(run_pair(MOV_D3_R1_R2, 3, second).scalar.read(1), 0x22);
        assert_eq!(run_pair(MOV_D3_R1_R2, 1, second).scalar.read(1), 0x22);
    }
}

// Silicon (movd p1/p2): other register files and other ports leave the delayed write alone.
#[test]
fn other_files_and_other_ports_do_not_collide() {
    for second in [MOV_S0_R6, MOV_P4_R6, ADD_R5_R6_R9, MUL_R5_R6_R9] {
        assert_eq!(run_pair(MOV_D3_R1_R2, 3, second).scalar.read(1), 0x22, "{second}");
    }
}

// Silicon (movd p2 F2): of two delayed moves landing together the later-issued survives.
#[test]
fn of_two_delayed_moves_the_later_issued_survives() {
    // mov.d3 r1 at 0, mov.d2 r3 at 1: both land in cycle 4
    let ctx = run(&[MOV_D3_R1_R2, MOV_D2_R3_R4]);
    assert_eq!((ctx.scalar.read(1), ctx.scalar.read(3)), (0x11, 0x44));
    // one bundle apart the other way round they land one cycle apart
    let ctx = run(&[MOV_D3_R1_R2, NOP, MOV_D2_R3_R4]);
    assert_eq!((ctx.scalar.read(1), ctx.scalar.read(3)), (0x22, 0x44));
}

// Silicon (movd p2 F5): mul (write unit held one cycle later) is dropped by an add issued a
// bundle after it.
#[test]
fn a_multiply_is_dropped_by_the_next_bundles_add() {
    let init = [(5, 0xaaaa)];
    let ctx = run_with(&init, &[MUL_R5_R6_R9, ADD_R3_R6_R9]);
    assert_eq!((ctx.scalar.read(5), ctx.scalar.read(3)), (0xaaaa, 0x67));
    assert_eq!(run_with(&init, &[MUL_R5_R6_R9, NOP, ADD_R3_R6_R9]).scalar.read(5), 0x66);
}

// Silicon (movd p3 G1, r2 = 0x12345673, r11 = 0x0f0f0f00): one register reached through two ports
// in one cycle holds the OR of both; through one port it holds the later value.
#[test]
fn one_register_through_two_ports_is_the_or_through_one_port_the_later() {
    let init = [(2, 0x12345673)];
    let ctx = run_pair_with(&init, MOV_D3_R1_R2, 3, ADD_R1_R11_R9);
    assert_eq!(ctx.scalar.read(1), 0x1f3f5f73);
    let ctx = run_pair_with(&init, MOV_D3_R1_R2, 3, MOV_R1_R11);
    assert_eq!(ctx.scalar.read(1), 0x0f0f0f00);
    let ctx = run_pair_with(&init, MOV_D3_R1_R2, 3, MOVA_R1_5);
    assert_eq!(ctx.scalar.read(1), 0x12345677);
}

// Silicon (movd p3 G2): lda r1 (lands in cycle 7) bundled with mov.d6 r1 ORs into r1.
#[test]
fn a_load_bundled_with_a_delayed_move_ors_into_the_register() {
    let ctx = run_with(&[(2, 0x12345673)], &[MOV_D6_WITH_LDA]);
    assert_eq!(ctx.scalar.read(1), 0x1f3f5f73);
}
