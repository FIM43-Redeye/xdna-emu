//! Replay of the 2D / 3D addressing captures taken on NPU1 silicon
//! (`tools/gen_agu_2d3d.py`, kernels `vec_agu_*`) through the real instruction handlers.
//!
//! Every 2D / 3D load and store applies one pointer and count update, whatever its width, its
//! register file or its fused conversion. The captures run each TableGen family (scalar,
//! vector, accumulator, q-register, fused ups / conv loads, fused srs / pack / conv stores)
//! with four concurrent walkers over three groups of descriptor sets whose counts start
//! below, at and above their sizes, with negative strides, and record the pointer and both
//! counts after every access.
//!
//! The ops are decoded from the instruction bytes Chess emitted for them (stored in the
//! goldens), so the decoder's classification of the descriptor operand is part of what is
//! replayed. A walker takes the captured encoding of its family at index `walker % n`: the
//! compiler picked `d0..d3` for the concurrent walkers, and the replay exercises each of
//! them with every descriptor set.
//!
//! Checked for every access of every family: the pointer after the access and the counts.
//! Checked for the raw families (scalar, 256- and 128-bit vector): the loaded data, and for
//! the stores the memory image after the walk. The fused families' data paths are not
//! under test here (their sources are accumulators); the EMU bridge run of the same
//! kernels covers them end to end.

use super::memory::MemoryUnit;
use crate::device::tile::Tile;
use crate::interpreter::bundle::{Operand, PostModify, SlotOp};
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::traits::Decoder;
use crate::interpreter::state::{
    ExecutionContext, MOD_BASE_D2D, MOD_BASE_D3D, MOD_BASE_DC, MOD_BASE_DJ, MOD_BASE_DN, MOD_BASE_M,
};

const KERNELS: [&str; 9] = [
    "vec_agu_ldscalar",
    "vec_agu_stscalar",
    "vec_agu_vload",
    "vec_agu_vstore",
    "vec_agu_ups_s",
    "vec_agu_ups_d",
    "vec_agu_srs_s",
    "vec_agu_srs_d",
    "vec_agu_pack",
];

/// Tile-local offset of the replayed memory and the matching core-view address (0x70000 is
/// the local data-memory quadrant). Walker `w` owns `REGION_OFFSET + w * region_bytes`.
const REGION_OFFSET: usize = 0x2000;
const CORE_BASE: u32 = 0x70000 + REGION_OFFSET as u32;

fn load(kernel: &str) -> serde_json::Value {
    let path = format!("{}/tools/golden/silicon/{kernel}.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).unwrap()
}

fn words(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn num(v: &serde_json::Value, key: &str) -> usize {
    v[key].as_u64().unwrap_or_else(|| panic!("layout key {key}")) as usize
}

/// The hex byte pairs of a disassembly line's byte column.
fn hex_bytes(text: &str) -> Vec<u8> {
    text.split_whitespace().map(|b| u8::from_str_radix(b, 16).unwrap()).collect()
}

/// One captured encoding of a mnemonic, decoded to the op the executor sees.
fn decode(decoder: &InstructionDecoder, enc: &serde_json::Value) -> SlotOp {
    let mnemonic = enc["mnemonic"].as_str().unwrap();
    let bytes = hex_bytes(enc["bytes"].as_str().unwrap());
    let bundle = decoder
        .decode(&bytes, 0)
        .unwrap_or_else(|e| panic!("{mnemonic}: {bytes:02x?} does not decode: {e:?}"));
    bundle
        .slots()
        .iter()
        .flatten()
        .find(|op| op.encoding_name.as_deref() == Some(mnemonic))
        .unwrap_or_else(|| panic!("{mnemonic}: no such slot in {}", enc["bundle"]))
        .clone()
}

fn descriptor_of(op: &SlotOp) -> u8 {
    match op.post_modify {
        PostModify::Descriptor(d) => d,
        other => panic!("{:?}: expected a descriptor post-modify, got {other:?}", op.encoding_name),
    }
}

/// The pointer register the access goes through (a store's data may itself be a pointer).
fn pointer_of(op: &SlotOp) -> u8 {
    op.sources
        .iter()
        .rev()
        .find_map(|s| match s {
            Operand::PointerReg(p) => Some(*p),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{:?} has no pointer operand", op.encoding_name))
}

fn start(ctx: &mut ExecutionContext) {
    ctx.bundle_seq = 0;
    ctx.vector.advance_bundle(0);
}

/// Whether an encoding's data operand is in the register file the family's C++ meant. The
/// same mnemonic serves several files (`vst.2d` stores w, x, accumulator and q registers), and
/// the captured disassembly lists each one.
fn operand_fits(family: &str, op: &SlotOp, is_store: bool) -> bool {
    let operand = if is_store {
        op.sources.first()
    } else {
        op.dest.as_ref()
    };
    match family {
        "lda" | "lda_s8" | "lda_u8" | "lda_s16" | "lda_u16" | "st" | "st_s8" | "st_s16" => {
            matches!(operand, Some(Operand::ScalarReg(_)))
        }
        "ldq" | "stq" => matches!(operand, Some(Operand::ControlReg(_))),
        "vld256" | "vld128" | "vst256" | "vst128" => matches!(operand, Some(Operand::VectorReg(_))),
        "vld_am32" | "vld_am64" | "vst_am32" | "vst_am64" => matches!(operand, Some(Operand::AccumReg(_))),
        _ => true,
    }
}

/// Families whose loaded data (or stored memory image) is compared as well.
fn data_checked(family: &str) -> bool {
    matches!(
        family,
        "lda"
            | "lda_s8"
            | "lda_u8"
            | "lda_s16"
            | "lda_u16"
            | "vld256"
            | "vld128"
            | "st"
            | "st_s8"
            | "st_s16"
            | "vst256"
            | "vst128"
    )
}

/// Let the deferred partial-word stores (`st.s8`, `st.s16`: their data register is read seven
/// cycles after issue) commit, the way the cycle-accurate executor does.
fn drain_stores(ctx: &mut ExecutionContext, tile: &mut Tile) {
    for _ in 0..10 {
        if ctx.pending_stores_empty() {
            break;
        }
        ctx.cycles += 1;
        for ps in ctx.drain_ready_stores() {
            let value = MemoryUnit::read_store_register(&ps.source, ctx, ps.width);
            MemoryUnit::write_memory(tile, ps.address, value, ps.width, None, None);
        }
    }
}

/// Write the source register of a raw store with the bytes the kernel stored from.
fn set_store_source(ctx: &mut ExecutionContext, op: &SlotOp, family: &str, bytes: &[u8]) {
    let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    match (family, op.sources.first()) {
        ("st", Some(Operand::ScalarReg(r))) => ctx.scalar.write(*r, word(0)),
        ("st_s8", Some(Operand::ScalarReg(r))) => ctx.scalar.write(*r, bytes[0] as u32),
        ("st_s16", Some(Operand::ScalarReg(r))) => {
            ctx.scalar.write(*r, u16::from_le_bytes([bytes[0], bytes[1]]) as u32)
        }
        ("vst256" | "vst128", Some(Operand::VectorReg(r))) => {
            let mut v = [0u32; 8];
            for (i, w) in v.iter_mut().enumerate() {
                *w = word(i);
            }
            ctx.vector.write(*r, v);
        }
        (f, s) => panic!("{f}: unexpected store source {s:?} in {:?}", op.encoding_name),
    }
}

/// Replays one kernel's capture. Returns (accesses, data words compared, images compared).
fn replay(kernel: &str) -> (usize, usize, usize) {
    let decoder = InstructionDecoder::load_default();
    let g = load(kernel);
    let lay = &g["layout"];
    let silicon = words(&g["silicon"]);
    let inputs = words(&g["inputs"][0]);
    let (walkers, groups, rec) = (num(lay, "walkers"), num(lay, "groups"), num(lay, "record_words"));
    let (region_bytes, region_base, region_word) =
        (num(lay, "region_bytes"), num(lay, "region_base"), num(lay, "region_word"));
    let (max_records, per_chunk, image_words) =
        (num(lay, "max_records"), num(lay, "words_per_chunk"), num(lay, "image_words"));
    let nin = region_word + 4 * region_bytes / 4;
    let encodings = g["encodings"].as_array().expect("goldens carry the Chess encodings");
    let (mut accesses, mut data_words, mut images) = (0, 0, 0);

    for (ci, chunk) in lay["chunks"].as_array().unwrap().iter().enumerate() {
        let family = chunk["family"].as_str().unwrap();
        let dim = num(chunk, "dim");
        let iters = num(
            lay,
            if dim == 2 {
                "iterations_2d"
            } else {
                "iterations_3d"
            },
        );
        let kind = chunk["kind"].as_str().unwrap();
        let mnemonic = chunk["mnemonic"].as_str().unwrap().replace(".2d", &format!(".{dim}d"));
        let ops: Vec<SlotOp> = encodings
            .iter()
            .filter(|e| e["mnemonic"].as_str() == Some(mnemonic.as_str()))
            .map(|e| decode(&decoder, e))
            .filter(|op| operand_fits(family, op, kind == "st"))
            .collect();
        assert!(!ops.is_empty(), "{kernel}: no {family} {mnemonic} in the captured disassembly");
        let sets = chunk["sets"].as_array().unwrap();
        let region_index = "RVBN".find(chunk["source"].as_str().unwrap()).unwrap();
        let region_at = ci * nin + region_word + region_index * region_bytes / 4;
        let region: Vec<u8> = inputs[region_at..region_at + region_bytes / 4]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let out = &silicon[ci * per_chunk..(ci + 1) * per_chunk];

        for w in 0..walkers {
            let op = ops[w % ops.len()].clone();
            let (desc, ptr) = (descriptor_of(&op), pointer_of(&op));
            let k = if dim == 2 {
                desc - MOD_BASE_D2D
            } else {
                desc - MOD_BASE_D3D
            };
            let mut tile = Tile::compute(0, 2);
            let walker_off = REGION_OFFSET + w * region_bytes;
            if kind == "ld" {
                tile.data_memory_mut()[walker_off..walker_off + region_bytes].copy_from_slice(&region);
            }
            let base = CORE_BASE + (w * region_bytes + region_base) as u32;

            for grp in 0..groups {
                let set: Vec<i64> = sets[grp * walkers + w]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect();
                let mut ctx = ExecutionContext::new();
                ctx.pointer.write(ptr, base);
                ctx.modifier.write(MOD_BASE_M + k, set[0] as u32);
                ctx.modifier.write(MOD_BASE_DN + k, set[1] as u32);
                ctx.modifier.write(MOD_BASE_DJ + k, set[2] as u32);
                if dim == 2 {
                    ctx.modifier.write(MOD_BASE_DC + k, set[3] as u32);
                } else {
                    ctx.modifier.write(MOD_BASE_DC + k, set[5] as u32);
                    ctx.modifier.write(MOD_BASE_DN + k + 4, set[3] as u32);
                    ctx.modifier.write(MOD_BASE_DJ + k + 4, set[4] as u32);
                    ctx.modifier.write(MOD_BASE_DC + k + 4, set[6] as u32);
                }
                for i in 0..iters {
                    let r = &out[((grp * iters + i) * walkers + w) * rec..][..rec];
                    let what = format!("{kernel} {mnemonic} {family} walker {w} group {grp} step {i} (d{k})");
                    if kind == "st" && data_checked(family) {
                        // The kernel's access index: `(i * walkers + w) & 31`, scaled by the access size.
                        let (stride, width) = match family {
                            "st" => (4, 4),
                            "st_s8" => (1, 1),
                            "st_s16" => (2, 2),
                            _ => (32, 32),
                        };
                        let a = ((i * walkers + w) & 31) * stride;
                        let mut bytes = region[a..a + width].to_vec();
                        bytes.resize(32, 0);
                        set_store_source(&mut ctx, &op, family, &bytes);
                    }
                    start(&mut ctx);
                    assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None), "{what}: not handled");
                    ctx.flush_pending_writes();
                    drain_stores(&mut ctx, &mut tile);

                    // The record's base word is walker 0's; a store walker has its own region.
                    let shift = if kind == "st" {
                        (w * region_bytes) as u32
                    } else {
                        0
                    };
                    assert_eq!(
                        ctx.pointer.read(ptr).wrapping_sub(base),
                        r[16].wrapping_sub(r[19]).wrapping_sub(shift),
                        "{what}: pointer"
                    );
                    assert_eq!(ctx.modifier_read(MOD_BASE_DC + k), r[17], "{what}: count 1");
                    if dim == 3 {
                        assert_eq!(ctx.modifier_read(MOD_BASE_DC + k + 4), r[18], "{what}: count 2");
                    }
                    if kind == "ld" && data_checked(family) {
                        let got: Vec<u32> = match &op.dest {
                            Some(Operand::ScalarReg(d)) => vec![ctx.scalar_read(*d)],
                            Some(Operand::VectorReg(d)) => {
                                let n = if family == "vld128" { 4 } else { 8 };
                                ctx.vector.read(*d)[..n].to_vec()
                            }
                            other => panic!("{what}: unexpected destination {other:?}"),
                        };
                        assert_eq!(&got[..], &r[..got.len()], "{what}: data");
                        data_words += got.len();
                    }
                    accesses += 1;
                }
            }
            if kind == "st" && data_checked(family) {
                let img_at = max_records * rec + w * image_words;
                let want: Vec<u8> =
                    out[img_at..img_at + image_words].iter().flat_map(|x| x.to_le_bytes()).collect();
                let got = &tile.data_memory()[walker_off..walker_off + region_bytes];
                let first_diff = want.iter().zip(got).position(|(a, b)| a != b);
                assert!(
                    first_diff.is_none(),
                    "{kernel} {mnemonic} {family} walker {w}: image differs from silicon at byte {first_diff:?} \
                     (silicon {:?}, replay {:?})",
                    first_diff.map(|d| want[d]),
                    first_diff.map(|d| got[d])
                );
                images += 1;
            }
        }
    }
    (accesses, data_words, images)
}

#[test]
fn every_family_follows_the_one_pointer_and_count_rule_on_silicon() {
    let mut total = 0;
    for kernel in KERNELS {
        let (accesses, data, images) = replay(kernel);
        eprintln!(
            "{kernel}: {accesses} accesses matched silicon ({data} loaded words, {images} store images)"
        );
        total += accesses;
    }
    // Every captured record is replayed once.
    let captured: usize = KERNELS
        .iter()
        .map(|k| {
            load(k)["layout"]["chunks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| num(c, "records"))
                .sum::<usize>()
        })
        .sum();
    assert_eq!(total, captured);
    eprintln!("{total} accesses matched silicon");
}

/// Every instruction the captures reached is classified the same way: the descriptor is the
/// post-modify of the access, and no descriptor operand is left among the sources.
#[test]
fn every_captured_encoding_decodes_to_a_descriptor_post_modify() {
    let decoder = InstructionDecoder::load_default();
    let mut seen = std::collections::BTreeSet::new();
    for kernel in KERNELS {
        for enc in load(kernel)["encodings"].as_array().unwrap() {
            let op = decode(&decoder, enc);
            let desc = descriptor_of(&op);
            let mnemonic = enc["mnemonic"].as_str().unwrap();
            let range = if mnemonic.contains(".3d") {
                MOD_BASE_D3D..MOD_BASE_D3D + 4
            } else {
                MOD_BASE_D2D..MOD_BASE_D2D + 8
            };
            assert!(range.contains(&desc), "{mnemonic}: descriptor operand {desc}");
            assert!(
                !op.sources.iter().any(|s| matches!(s, Operand::ModifierReg(_))),
                "{mnemonic}: descriptor left among the sources {:?}",
                op.sources
            );
            seen.insert(mnemonic.to_string());
        }
    }
    eprintln!("{} distinct 2D / 3D mnemonics decoded", seen.len());
}
