//! The sparse loads (`vldb.sparse.*`) against NPU1 silicon, record for record.
//!
//! The six captures of `tools/gen_vldb_sparse.py` (`vec_sparseload_{proto,ring}_w{4,8,16}`) are
//! replayed through `MemoryUnit::execute` the way the decoder presents the ops: the pointer is
//! both source and destination, a `pop`/`peek` names a `qwl`/`qwh` register in `extra_dests`.
//! Each chunk is one designed 1 KiB stream placed in data memory; the ring, `DP` and the flags
//! persist from chunk to chunk exactly as on the core, and every chunk starts with the kernel's
//! prologue (a `reset` and nine `fill`s). Every recorded step is compared:
//!
//! - after any step, `srSparse_of` and `srCompr_uf`;
//! - after a `pop`/`peek`, the half register (data) and the half of `q` (mask) it wrote;
//! - after a `dp` step (a peek whose pointer copy is replaced by `DP`), the address `DP` holds.
//!
//! The one mismatch that is not understood is listed in [`KNOWN_RESIDUAL`]; any other difference
//! fails the test, so a new one cannot hide behind it.

use super::memory::MemoryUnit;
use crate::device::tile::Tile;
use crate::interpreter::bundle::{Operand, SlotIndex, SlotOp};
use crate::interpreter::state::{ExecutionContext, COMPR_UF_REG_INDEX, DP_REG_INDEX, SPARSE_OF_REG_INDEX};
use xdna_archspec::aie2::isa::SemanticOp;

/// Where the kernel's input buffer sits: tile-local 0x4000, core address 0x74000.
const STREAM_OFFSET: usize = 0x4000;
const STREAM_ADDR: u32 = 0x70000 + STREAM_OFFSET as u32;
/// The pointer register the replay uses (the kernel's `P`).
const POINTER: u8 = 4;
/// The ring is ten lines; the prologue writes all of them.
const PROLOGUE_FILLS: usize = 9;
/// qwl1/qwh1: vector registers 2 and 3, mask `q1`.
const LO: u8 = 21;
const HI: u8 = 25;
const SCRATCH_LO: u8 = 20;
const LO_VECTOR: u8 = 2;
const HI_VECTOR: u8 = 3;
const MASK: u8 = 1;

/// (stream, script) pairs whose recorded data differ from the model: a `pop` straight after a
/// `reset` on a ring whose second line an earlier script wrote returns the tail of its data from
/// a different slot (width 4 and 8 only). Open question in docs/fidelity-gaps/vector-compute.md.
const KNOWN_RESIDUAL: [(&str, &str); 1] = [("dense40", "unprimed")];

fn load(kernel: &str) -> serde_json::Value {
    let path = format!("{}/tools/golden/silicon/{kernel}.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).unwrap()
}

fn words(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn sparse_op(mnemonic: &str, dest: Option<u8>) -> SlotOp {
    let mut op = SlotOp::from_semantic(SlotIndex::LoadB, SemanticOp::Load)
        .with_dest(Operand::PointerReg(POINTER))
        .with_source(Operand::PointerReg(POINTER));
    op.encoding_name = Some(mnemonic.to_string());
    if let Some(c) = dest {
        op.extra_dests.push(Operand::ControlReg(c));
    }
    op
}

struct Replay {
    ctx: ExecutionContext,
    tile: Tile,
    width: u32,
}

impl Replay {
    fn new(width: u32) -> Self {
        let mut ctx = ExecutionContext::new();
        ctx.bundle_seq = 0;
        Self { ctx, tile: Tile::compute(0, 2), width }
    }

    fn issue(&mut self, kind: &str, dest: Option<u8>) {
        let op = sparse_op(&format!("vldb.sparse.{kind}.{}", self.width), dest);
        assert!(MemoryUnit::execute(&op, &mut self.ctx, &mut self.tile, None, None));
        self.ctx.force_commit_all_pending();
    }

    fn flags(&self) -> (u32, u32) {
        (self.ctx.scalar_read(SPARSE_OF_REG_INDEX), self.ctx.scalar_read(COMPR_UF_REG_INDEX))
    }

    /// The half register and mask half a lo/hi step left: (mask bytes, data bytes).
    fn half(&self, high: bool) -> (Vec<u8>, Vec<u8>) {
        let q = self.ctx.mask.read(MASK);
        let (m, v) = if high {
            ([q[2], q[3]], HI_VECTOR)
        } else {
            ([q[0], q[1]], LO_VECTOR)
        };
        let bytes = |ws: &[u32]| ws.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
        (bytes(&m), bytes(&self.ctx.vector.read(v)))
    }
}

fn bytes(ws: &[u32]) -> Vec<u8> {
    ws.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Replay one capture; returns (reads compared, dp steps compared, flag pairs compared) and the
/// differences found as (stream, script, step index, what).
fn replay(kernel: &str) -> ((usize, usize, usize), Vec<(String, String, usize, &'static str)>) {
    let g = load(kernel);
    let width = g["layout"]["width"].as_u64().unwrap() as u32;
    let rec = g["layout"]["record_words"].as_u64().unwrap() as usize;
    let inputs = words(&g["inputs"][0]);
    let silicon = words(&g["silicon"]);
    let streams: Vec<String> = g["layout"]["streams"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect();
    let scripts: Vec<(String, Vec<String>)> = g["layout"]["scripts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            (
                s["name"].as_str().unwrap().to_string(),
                s["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_str().unwrap().to_string())
                    .collect(),
            )
        })
        .collect();
    let stream_words = 256;
    assert_eq!(inputs.len(), streams.len() * stream_words);

    let mut r = Replay::new(width);
    let (mut reads, mut dps, mut flag_pairs) = (0, 0, 0);
    let mut bad = Vec::new();
    let mut at = 0;
    for (ci, stream) in streams.iter().enumerate() {
        let chunk = &inputs[ci * stream_words..(ci + 1) * stream_words];
        assert!(r.tile.write_data(STREAM_OFFSET, &bytes(chunk)));
        r.ctx.pointer.write(POINTER, STREAM_ADDR);
        r.issue("reset", None);
        for _ in 0..PROLOGUE_FILLS {
            r.issue("fill", None);
        }
        for (name, steps) in &scripts {
            r.ctx.pointer.write(POINTER, STREAM_ADDR);
            for (ki, step) in steps.iter().enumerate() {
                let record = &silicon[at..at + rec];
                at += rec;
                let mut what = None;
                match step.as_str() {
                    "clr" => {
                        r.ctx.scalar.write(SPARSE_OF_REG_INDEX, 0);
                        r.ctx.scalar.write(COMPR_UF_REG_INDEX, 0);
                    }
                    "reset" | "fill" => r.issue(step, None),
                    "peek_lo" | "peek_hi" | "pop_lo" | "pop_hi" => {
                        let (kind, high) = step.split_once('_').map(|(k, h)| (k, h == "hi")).unwrap();
                        r.issue(kind, Some(if high { HI } else { LO }));
                        let (mask, data) = r.half(high);
                        let (want_mask, want_data) = if high {
                            (bytes(&record[10..12]), bytes(&record[24..32]))
                        } else {
                            (bytes(&record[8..10]), bytes(&record[16..24]))
                        };
                        reads += 1;
                        if (&mask, &data) != (&want_mask, &want_data) {
                            what = Some("data");
                        }
                    }
                    "dp" => {
                        r.issue("peek", Some(SCRATCH_LO));
                        dps += 1;
                        if r.ctx.scalar_read(DP_REG_INDEX) != record[0] {
                            what = Some("dp");
                        }
                    }
                    other => panic!("unknown step {other}"),
                }
                flag_pairs += 1;
                if r.flags() != (record[1], record[2]) {
                    what = what.or(Some("flags"));
                }
                if let Some(w) = what {
                    bad.push((stream.clone(), name.clone(), ki, w));
                }
            }
        }
    }
    assert_eq!(at, silicon.len(), "{kernel}: consumed every record");
    ((reads, dps, flag_pairs), bad)
}

fn check(kernel: &str, reads: usize, dps: usize, flag_pairs: usize) {
    let (counts, bad) = replay(kernel);
    assert_eq!(counts, (reads, dps, flag_pairs), "{kernel}: records compared");
    let unexpected: Vec<_> = bad
        .iter()
        .filter(|(s, sc, _, what)| !(*what == "data" && KNOWN_RESIDUAL.contains(&(s.as_str(), sc.as_str()))))
        .collect();
    assert!(
        unexpected.is_empty(),
        "{kernel}: {} differ from silicon, first {:?}",
        unexpected.len(),
        &unexpected[..unexpected.len().min(5)]
    );
    // The residual is exactly the documented one, so a fix that makes it vanish is noticed.
    let width4_or_8 = !kernel.ends_with("w16") && kernel.contains("proto");
    assert_eq!(bad.len(), usize::from(width4_or_8), "{kernel}: residual count");
}

#[test]
fn proto_w4_matches_silicon() {
    check("vec_sparseload_proto_w4", 1820, 595, 4060);
}

#[test]
fn proto_w8_matches_silicon() {
    check("vec_sparseload_proto_w8", 1820, 595, 4060);
}

#[test]
fn proto_w16_matches_silicon() {
    check("vec_sparseload_proto_w16", 1820, 595, 4060);
}

#[test]
fn ring_w4_matches_silicon() {
    check("vec_sparseload_ring_w4", 1680, 140, 3990);
}

#[test]
fn ring_w8_matches_silicon() {
    check("vec_sparseload_ring_w8", 1680, 140, 3990);
}

#[test]
fn ring_w16_matches_silicon() {
    check("vec_sparseload_ring_w16", 1680, 140, 3990);
}
