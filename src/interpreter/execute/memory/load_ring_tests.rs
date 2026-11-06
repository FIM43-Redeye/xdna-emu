//! A core in row 2 reading its south quadrant (no core memory: the memory-tile row), from the
//! NPU1 silicon cases of `tools/gen_port_latch_probe.py` (suite
//! `build/experiments/harness/port-latch/suite`, every case r1 == r2). See
//! [`crate::interpreter::state::load_ring`] for the rule.

use super::*;
use crate::device::DeviceState;
use crate::interpreter::bundle::{ElementType, SlotIndex};
use xdna_archspec::aie2::isa::SemanticOp;

/// The probe core: physical column 2, row 2 (west = column 1, real memory; south = memtile).
const COL: usize = 2;
const ROW: usize = 2;
/// Window offset the tests use in every quadrant.
const BASE: u32 = 0x0800;

/// A neighbour row's bytes name the tile and the row: `tag, row index, byte index...`.
fn row_bytes(tag: u8, addr: u32) -> [u8; 16] {
    std::array::from_fn(|k| match k {
        0 => tag,
        1 => (addr >> 4) as u8,
        _ => k as u8,
    })
}

const OWN: u8 = 0x70;
const WEST: u8 = 0x50;
const NORTH: u8 = 0x60;

fn fill(mem: &mut [u8], tag: u8) {
    for a in (0..0x4000u32).step_by(16) {
        mem[a as usize..a as usize + 16].copy_from_slice(&row_bytes(tag, a));
    }
}

struct Core {
    ctx: ExecutionContext,
    tile: Tile,
    nbr: NeighborMemory,
}

impl Core {
    fn new() -> Self {
        let mut device = DeviceState::new_npu1();
        fill(device.tile_mut(COL - 1, ROW).unwrap().data_memory_mut(), WEST);
        fill(device.tile_mut(COL, ROW + 1).unwrap().data_memory_mut(), NORTH);
        let mut nbr = NeighborMemory::new(COL, ROW);
        for d in [MemoryQuadrant::South, MemoryQuadrant::West, MemoryQuadrant::North] {
            nbr.ensure_snapshot(d, &device);
        }
        let mut tile = Tile::compute(COL as u8, ROW as u8);
        fill(tile.data_memory_mut(), OWN);
        Core { ctx: ExecutionContext::new(), tile, nbr }
    }

    fn run(&mut self, op: &SlotOp) {
        assert!(MemoryUnit::execute(op, &mut self.ctx, &mut self.tile, Some(&mut self.nbr), None));
        self.ctx.flush_pending_writes();
    }

    /// A pointer load into v`dest` (or r`dest`) at `addr`; returns the destination bytes.
    fn load(&mut self, slot: SlotIndex, width: MemWidth, addr: u32) -> Vec<u8> {
        self.ctx.pointer.write(if slot == SlotIndex::LoadA { 0 } else { 4 }, addr);
        let ptr = if slot == SlotIndex::LoadA { 0 } else { 4 };
        let mut op = SlotOp::from_semantic(slot, SemanticOp::Load).with_mem_width(width);
        if matches!(width, MemWidth::Vector256 | MemWidth::QuadWord) {
            op = op.as_vector(ElementType::Int32).with_dest(Operand::VectorReg(2));
            self.ctx.vector.write(2, [0xEEEE_EEEE; 8]);
        } else {
            op = op.with_dest(Operand::ScalarReg(9));
        }
        let op = op.with_source(Operand::PointerReg(ptr));
        self.run(&op);
        if matches!(width, MemWidth::Vector256 | MemWidth::QuadWord) {
            self.ctx.vector.read(2).iter().flat_map(|w| w.to_le_bytes()).collect()
        } else {
            self.ctx.scalar.read(9).to_le_bytes().to_vec()
        }
    }

    fn vldb(&mut self, addr: u32) -> Vec<u8> {
        self.load(SlotIndex::LoadB, MemWidth::Vector256, addr)
    }

    fn vlda(&mut self, addr: u32) -> Vec<u8> {
        self.load(SlotIndex::LoadA, MemWidth::Vector256, addr)
    }

    fn store(&mut self, addr: u32) {
        self.ctx.pointer.write(2, addr);
        self.ctx.vector.write(6, [0x5151_5151; 8]);
        let op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Store)
            .as_vector(ElementType::Int32)
            .with_mem_width(MemWidth::Vector256)
            .with_source(Operand::VectorReg(6))
            .with_source(Operand::PointerReg(2));
        self.run(&op);
    }

    /// `vldb.4x<width>.lo` with the four lane values given; returns the 32 result bytes.
    fn gather(&mut self, width: u32, lanes: [u32; 4]) -> Vec<u8> {
        let mut src = [0u32; 8];
        src[..4].copy_from_slice(&lanes);
        self.ctx.vector.write(4, src);
        self.ctx.vector.write(2, [0xEEEE_EEEE; 8]);
        let mut op = SlotOp::from_semantic(SlotIndex::LoadB, SemanticOp::Load)
            .as_vector(ElementType::Int32)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::VectorReg(4));
        op.encoding_name = Some(format!("vldb.4x{width}.lo"));
        self.run(&op);
        self.ctx.vector.read(2).iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

fn west(off: u32) -> u32 {
    0x5_0000 | (BASE + off)
}
fn north(off: u32) -> u32 {
    0x6_0000 | (BASE + off)
}
fn own(off: u32) -> u32 {
    0x7_0000 | (BASE + off)
}
fn south(off: u32) -> u32 {
    0x4_0000 | (BASE + off)
}

/// The 32 bytes of the line at `off` of a filled tile.
fn line(tag: u8, off: u32) -> Vec<u8> {
    let a = BASE + off;
    [row_bytes(tag, a), row_bytes(tag, a + 16)].concat()
}

fn lines(tag: u8, offs: impl IntoIterator<Item = u32>) -> Vec<Vec<u8>> {
    offs.into_iter().map(|o| line(tag, o)).collect()
}

/// ring_a b0 / sweep ph8: six vldb lines, then south vldb probes return them oldest first, with
/// period six, whatever the probe address.
#[test]
fn south_vldb_probes_return_the_last_six_vldb_lines_oldest_first() {
    let mut c = Core::new();
    for k in 0..6 {
        c.vldb(west(32 * k));
    }
    let got: Vec<Vec<u8>> = (0..7).map(|k| c.vldb(south(0x100 + 0x40 * k))).collect();
    let want = lines(WEST, [0, 32, 64, 96, 128, 160, 0]);
    assert_eq!(got, want);
}

/// ring_a b1 (own vldb writes the ring), b2 / b3 (vlda: the other port), b4 / b5 (stores).
#[test]
fn own_vldb_writes_the_b_ring_and_vlda_and_stores_do_not() {
    let mut c = Core::new();
    for k in 0..6 {
        c.vldb(north(32 * k));
    }
    c.vlda(west(0x400));
    c.store(west(0x420));
    c.store(own(0x440));
    c.vldb(own(0x200));
    let got: Vec<Vec<u8>> = (0..6).map(|_| c.vldb(south(0))).collect();
    let mut want = lines(NORTH, [32, 64, 96, 128, 160]);
    want.push(line(OWN, 0x200));
    assert_eq!(got, want);
}

/// ring_b b2: a south vldb returns its slot and moves the pointer without writing.
#[test]
fn a_south_vldb_moves_the_pointer_without_writing() {
    let mut c = Core::new();
    for k in 0..6 {
        c.vldb(west(32 * k));
    }
    assert_eq!(c.vldb(south(0)), line(WEST, 0));
    let got: Vec<Vec<u8>> = (0..6).map(|_| c.vldb(south(0))).collect();
    assert_eq!(got, lines(WEST, [32, 64, 96, 128, 160, 0]));
}

/// ring_b b0 / b1: a 128-bit vldb writes the row of its half and moves only that half's pointer;
/// ring_b b3: a south 128-bit vldb reads its half's slot into the low 16 bytes.
#[test]
fn a_128_bit_vldb_touches_only_the_half_its_address_selects() {
    let mut c = Core::new();
    for k in 0..6 {
        c.vldb(north(32 * k));
    }
    let q = c.load(SlotIndex::LoadB, MemWidth::QuadWord, west(0x210));
    assert_eq!(&q[..16], &row_bytes(WEST, BASE + 0x210));
    let p1 = c.vldb(south(0));
    // the hi half is one ahead of the lo half now
    assert_eq!(&p1[..16], &line(NORTH, 0)[..16]);
    assert_eq!(&p1[16..], &line(NORTH, 32)[16..]);
    for _ in 0..4 {
        c.vldb(south(0));
    }
    let p6 = c.vldb(south(0));
    assert_eq!(&p6[16..], &row_bytes(WEST, BASE + 0x210));
    let s = c.load(SlotIndex::LoadB, MemWidth::QuadWord, south(0));
    assert_eq!(&s[..16], &line(NORTH, 0)[..16]);
    assert_eq!(&s[16..], &[0u8; 16]);
}

/// ring_d b3 / b4: a scalar load writes its whole 16-byte row into its half;
/// ring_e b4 / b5: a south scalar load reads its word at its in-row offset from that half.
#[test]
fn a_scalar_load_writes_its_whole_row_and_a_south_one_reads_its_word() {
    let mut c = Core::new();
    for k in 0..6 {
        c.vlda(north(32 * k));
    }
    c.load(SlotIndex::LoadA, MemWidth::Word, west(0x314)); // hi half, row 0x310
    let p: Vec<Vec<u8>> = (0..6).map(|_| c.vlda(south(0))).collect();
    assert_eq!(&p[0][..16], &line(NORTH, 0)[..16]);
    assert_eq!(&p[0][16..], &line(NORTH, 32)[16..]);
    assert_eq!(&p[5][..16], &line(NORTH, 160)[..16]);
    assert_eq!(&p[5][16..], &row_bytes(WEST, BASE + 0x310));
    // the A lo pointer is back at the first north line, hi at the second
    let w = c.load(SlotIndex::LoadA, MemWidth::Word, south(4));
    assert_eq!(w, line(NORTH, 0)[4..8].to_vec());
    let w = c.load(SlotIndex::LoadA, MemWidth::Word, south(20));
    assert_eq!(w, line(NORTH, 32)[20..24].to_vec());
}

/// ring_c b4..b7, ring_e b7 / b8: a gather's lanes 0-1 are an access on port A, lanes 2-3 on
/// port B; a south lane reads its half's slot at its in-row offset.
#[test]
fn gather_lanes_zero_one_use_port_a_and_two_three_port_b() {
    let mut c = Core::new();
    for k in 0..6 {
        c.vlda(west(32 * k));
        c.vldb(north(32 * k));
    }
    // lanes: own, south, south, own -> lane 1 from A hi slot 0, lane 2 from B lo slot 0
    let r = c.gather(64, [own(0x600), south(0x600), south(0x620), own(0x640)]);
    assert_eq!(&r[0..8], &row_bytes(OWN, BASE + 0x600)[..8]);
    assert_eq!(&r[8..16], &line(WEST, 0)[16..24]);
    assert_eq!(&r[16..24], &line(NORTH, 0)[..8]);
    assert_eq!(&r[24..32], &row_bytes(OWN, BASE + 0x650)[..8]);
    // each beat wrote / advanced one slot of each half: A = [own 0x600 row, W0 hi]
    let a = c.vlda(south(0));
    assert_eq!(&a[..16], &line(WEST, 32)[..16]);
    let b = c.vldb(south(0));
    assert_eq!(&b[16..], &line(NORTH, 32)[16..]);
    for _ in 0..4 {
        c.vlda(south(0));
    }
    let a = c.vlda(south(0));
    assert_eq!(&a[..16], &row_bytes(OWN, BASE + 0x600));
    assert_eq!(&a[16..], &line(WEST, 0)[16..]);
}

/// A north-of-row-two load still reads its memory and fills the ring (control).
#[test]
fn a_load_with_memory_behind_it_returns_that_memory() {
    let mut c = Core::new();
    assert_eq!(c.vldb(north(0x80)), line(NORTH, 0x80));
    assert_eq!(c.vlda(west(0xa0)), line(WEST, 0xa0));
}
