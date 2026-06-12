//! Deterministic, Stage-A-safe DMA chain generation.
use super::chain::{BdPattern, Direction, DmaChain, DmaTransfer, Dtype, Engine, Feature};
use super::table::{parse_key, supported, Target};

/// Field-width maxima (AM025 aie_registers_aie2.json). Stage A clamps well under
/// these; the binding constraint is footprint <= region.
const SHIM_STEP_MAX: u32 = 0xFFFFF; // 20-bit
const MEMTILE_STEP_MAX: u32 = 0x1FFFF; // 17-bit

struct Xorshift64(u64);
impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Xorshift64(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())]
    }
}

/// Region element counts a chain may use (kept modest for Stage-A safety and fast
/// compiles; all transfers in a chain share the same in-region length).
const REGION_ELEMS: [usize; 3] = [64, 128, 256];

/// Build a Stage-A-safe pattern for `feature` covering `region` data elements.
///
/// `dtype` matters on the shim: the NoC addresses memory in 4-byte words, so
/// every non-innermost BD stride must satisfy `stride * byte_size % 4 == 0`
/// (verified against mlir-aie AIEXDialect.cpp shim BD legalization). We express
/// that as `word = (4 / byte_size)` element-granularity: I32 -> 1 (no
/// constraint, the fix is a no-op), I16 -> 2, I8 -> 4. Memtile is on-chip and
/// element-granular, so `word = 1` there.
fn build_pattern(
    rng: &mut Xorshift64,
    feature: Feature,
    engine: Engine,
    dtype: Dtype,
    region: usize,
) -> BdPattern {
    let step_max = match engine {
        Engine::Shim => SHIM_STEP_MAX,
        Engine::Memtile => MEMTILE_STEP_MAX,
    };
    let word = match engine {
        Engine::Shim => (4 / dtype.byte_size()).max(1),
        Engine::Memtile => 1,
    };
    let split2 = |rng: &mut Xorshift64, n: usize| -> (usize, usize) {
        let mut divs: Vec<usize> = (1..=n).filter(|d| n % d == 0).collect();
        divs.retain(|&d| d > 1 && d < n);
        if divs.is_empty() {
            return (1, n);
        }
        let a = rng.pick(&divs);
        (a, n / a)
    };
    let clamp = |s: u32| -> u32 { s.min(step_max).max(1) };

    match feature {
        Feature::Linear | Feature::Packet => {
            let mut p = BdPattern {
                sizes: vec![region as u32],
                strides: vec![1],
                pad_before: vec![],
                pad_after: vec![],
                packet: None,
            };
            if feature == Feature::Packet {
                p.packet = Some((rng.below(4) as u8, 0));
            }
            p
        }
        Feature::Strided2d => {
            // Non-innermost stride is b; require b to be a multiple of `word`
            // (shim word-addressing). region and word are powers of two, so split
            // `region/word` units then scale: b = (region/a) stays word-aligned.
            let units = (region / word).max(1);
            let (a, bu) = split2(rng, units);
            let b = bu * word;
            BdPattern {
                sizes: vec![a as u32, b as u32],
                strides: vec![clamp(b as u32), 1],
                pad_before: vec![],
                pad_after: vec![],
                packet: None,
            }
        }
        Feature::Transpose => {
            if word > 1 {
                // Word-granular transpose: innermost `word`-sized contiguous run
                // (stride 1, exempt); outer strides `word` and `b*word` are both
                // multiples of `word`. A genuine transpose of 4-byte words that
                // fits the shim's 3-dim cap. b*a*word == region.
                let units = (region / word).max(1);
                let (b, a) = split2(rng, units);
                BdPattern {
                    sizes: vec![b as u32, a as u32, word as u32],
                    strides: vec![clamp(word as u32), clamp((b * word) as u32), 1],
                    pad_before: vec![],
                    pad_after: vec![],
                    packet: None,
                }
            } else {
                let (a, b) = split2(rng, region);
                BdPattern {
                    sizes: vec![b as u32, a as u32],
                    strides: vec![1, clamp(b as u32)],
                    pad_before: vec![],
                    pad_after: vec![],
                    packet: None,
                }
            }
        }
        Feature::Strided3d => {
            // Non-innermost strides are b*c and c; require c (hence b*c) to be a
            // multiple of `word`. Split `region/word` units, then scale c by word.
            let units = (region / word).max(1);
            let (a, bc) = split2(rng, units);
            let (b, _c0) = split2(rng, bc.max(2));
            let a = a.max(1);
            let b = b.max(1);
            let cu = (units / (a * b)).max(1);
            let c = cu * word;
            BdPattern {
                sizes: vec![a as u32, b as u32, c as u32],
                strides: vec![clamp((b * c) as u32), clamp(c as u32), 1],
                pad_before: vec![],
                pad_after: vec![],
                packet: None,
            }
        }
        Feature::Strided4d => {
            let half = (region / 4).max(1);
            let (c, d) = split2(rng, half.max(2));
            BdPattern {
                sizes: vec![2, 2, c as u32, d as u32],
                strides: vec![clamp((2 * c * d) as u32), clamp((c * d) as u32), clamp(d as u32), 1],
                pad_before: vec![],
                pad_after: vec![],
                packet: None,
            }
        }
        Feature::Overlap => {
            // Split `region/word` units so the inner run b is a multiple of
            // `word` and large enough to admit a word-aligned overlapping stride
            // (stride a multiple of `word`, 0 < stride < b). For word>1 this also
            // keeps b's stride contiguity word-granular.
            let units = (region / word).max(1);
            let (a, bu) = split2(rng, units);
            let b = bu * word;
            // Overlapping outer stride: ~b/2 rounded to a multiple of `word`,
            // clamped to [word, b-word] so it stays strictly below b and overlaps.
            let mut stride = ((b / 2) / word).max(1) * word;
            if stride >= b {
                stride = b.saturating_sub(word).max(word);
            }
            BdPattern {
                sizes: vec![a as u32, b as u32],
                strides: vec![clamp(stride as u32), 1],
                pad_before: vec![],
                pad_after: vec![],
                packet: None,
            }
        }
        Feature::PadBefore | Feature::PadAfter | Feature::PadBoth => {
            let (a, b) = split2(rng, region);
            let pad = 2u32;
            let (pb, pa) = match feature {
                Feature::PadBefore => (vec![pad, pad], vec![0, 0]),
                Feature::PadAfter => (vec![0, 0], vec![pad, pad]),
                _ => (vec![pad, pad], vec![pad, pad]),
            };
            BdPattern {
                sizes: vec![a as u32, b as u32],
                strides: vec![clamp(b as u32), 1],
                pad_before: pb,
                pad_after: pa,
                packet: None,
            }
        }
        Feature::Iter | Feature::Chain => BdPattern {
            sizes: vec![region as u32],
            strides: vec![1],
            pad_before: vec![],
            pad_after: vec![],
            packet: None,
        },
    }
}

/// Generate a deterministic Stage-A-safe chain for `(seed, target)`.
pub fn generate(seed: u64, target: &str) -> DmaChain {
    let tgt: Target = parse_key(target).expect("generate target must be a valid 3a key");
    let mut rng = Xorshift64::new(seed);
    let engine = tgt.engine;
    let dtype = tgt.dtype;

    // Packet routing is device-level: a packet transfer routes via aie.packet_flow
    // instead of the circuit aie.flow on the same shim->memtile port. Mixing a
    // packet transfer with circuit transfers on that port would demand two
    // contradictory static routings -> aiecc rejects it. So packet chains must be
    // PURE: exactly one packet transfer, no circuit transfers mixed in.
    // Memtile chains are forced single-transfer (spec "single-transfer mode"):
    // N=1 covers all 48 memtile keys; multi-pattern memtile chains would need N
    // distinct static BDs with intricate lock sequencing, and the shim path
    // already exercises multi-transfer chains. Packet chains are N=1 for purity.
    let n = if tgt.feature == Feature::Packet || tgt.engine == Engine::Memtile {
        1
    } else {
        1 + rng.below(4)
    };
    let target_slot = rng.below(n);
    let region = REGION_ELEMS[rng.below(REGION_ELEMS.len())];

    let mut transfers = Vec::with_capacity(n);
    let mut in_off = 0usize;
    let mut out_off = 0usize;
    for k in 0..n {
        let (feature, dir) = if k == target_slot {
            (tgt.feature, tgt.dir)
        } else {
            loop {
                let f = rng.pick(&Feature::all_3a());
                let d = rng.pick(&Direction::all());
                // Exclude Packet from random fillers: a non-packet chain must
                // never accidentally include a packet transfer (would corrupt
                // device-level routing -- see the n=1 guard above).
                if supported(engine, f, d) && f != Feature::Packet {
                    break (f, d);
                }
            }
        };
        let pattern = build_pattern(&mut rng, feature, engine, dtype, region);
        let in_elems = region;
        let out_elems = region + pattern.pad_elems();
        transfers.push(DmaTransfer { engine, dir, feature, pattern, in_off, out_off, in_elems, out_elems });
        in_off += in_elems;
        out_off += out_elems;
    }

    DmaChain { seed, target_key: target.to_string(), dtype, engine, transfers }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::table::universe_keys;

    #[test]
    fn deterministic() {
        let a = generate(42, "transpose/memtile/mm2s/I16");
        let b = generate(42, "transpose/memtile/mm2s/I16");
        assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&b).unwrap());
    }

    #[test]
    fn target_always_present() {
        for (i, key) in universe_keys().into_iter().enumerate() {
            let c = generate(i as u64, &key);
            assert_eq!(c.target_key, key);
            assert!(c.keys().contains(&key), "seed {i}: target {key} not in chain keys {:?}", c.keys());
            assert!(c.transfers.iter().all(|t| t.engine == c.engine));
        }
    }

    #[test]
    fn packet_chains_are_single_pure() {
        for dt in ["I32", "I16", "I8"] {
            for eng in ["shim", "memtile"] {
                let key = format!("packet/{eng}/mm2s/{dt}");
                let c = generate(99, &key);
                assert_eq!(c.transfers.len(), 1, "packet chain must be single-transfer: {key}");
                assert_eq!(c.transfers[0].pattern.packet.is_some(), true, "{key}: packet info present");
            }
        }
        // non-packet chains never include a packet transfer
        for i in 0..200u64 {
            let c = generate(i, "linear/shim/mm2s/I32");
            assert!(
                c.transfers.iter().all(|t| t.pattern.packet.is_none()),
                "filler leaked a packet transfer"
            );
        }
    }

    #[test]
    fn memtile_chains_are_single() {
        for i in 0..60u64 {
            let c = generate(i, "strided2d/memtile/mm2s/I32");
            assert_eq!(c.transfers.len(), 1, "memtile chains must be single-transfer");
        }
    }

    #[test]
    fn stage_a_safe_footprints() {
        for (i, key) in universe_keys().into_iter().enumerate().cycle().take(2000) {
            let c = generate(i as u64 * 7 + 1, &key);
            for t in &c.transfers {
                let p = &t.pattern;
                let max_addr: usize = p
                    .sizes
                    .iter()
                    .zip(&p.strides)
                    .map(|(&s, &st)| (s.saturating_sub(1) as usize) * st as usize)
                    .sum();
                assert!(
                    max_addr < t.in_elems,
                    "seed{i} key{key}: footprint {max_addr} >= region {}",
                    t.in_elems
                );
                let step_max = if t.engine == Engine::Shim {
                    SHIM_STEP_MAX
                } else {
                    MEMTILE_STEP_MAX
                };
                assert!(
                    p.strides.iter().all(|&s| s <= step_max),
                    "seed{i} key{key}: stride over field width"
                );
                assert!(p.sizes.len() <= 4 && !p.sizes.is_empty());
                if t.engine == Engine::Shim {
                    assert!(p.sizes.len() <= 3, "shim caps at 3 data dims");
                    assert!(p.pad_before.is_empty(), "shim has no padding");
                    // Shim NoC word-addressing: every NON-innermost stride must be
                    // 4-byte aligned. The innermost dim (last) may be any stride
                    // (typically 1, a contiguous run). Catch regressions in Rust.
                    let bs = c.dtype.byte_size();
                    let last = p.strides.len() - 1;
                    for (d, &st) in p.strides.iter().enumerate() {
                        if d == last {
                            continue;
                        }
                        assert_eq!(
                            (st as usize * bs) % 4,
                            0,
                            "seed{i} key{key}: non-innermost stride {st} (dim {d}) * {bs}B not 4-aligned"
                        );
                    }
                }
                assert_eq!(t.out_elems, t.in_elems + p.pad_elems());
            }
        }
    }
}
