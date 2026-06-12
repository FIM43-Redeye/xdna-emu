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
    // `word` = elements per 32-bit word (I32->1, I16->2, I8->4). Two distinct
    // constraints reference it:
    //   - Shim NoC outer-stride alignment (4-byte): strided2d/3d/overlap below
    //     scale their non-innermost strides by `shim_word`, which is `word` on
    //     the shim and 1 on the memtile (element-granular -- those features
    //     already have innermost stride 1 there, so no scaling is needed).
    //   - Memtile sub-32-bit BD rules reference `word` directly on BOTH engines:
    //     transpose's innermost dim must be a stride-1 word-run, and inner-axis
    //     padding counts must be 32-bit-word-aligned.
    let word = (4 / dtype.byte_size()).max(1);
    let shim_word = match engine {
        Engine::Shim => word,
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
            // Non-innermost stride is b; on the shim require b to be a multiple of
            // `shim_word` (NoC word-addressing). region and shim_word are powers of
            // two, so split `region/shim_word` units then scale: b = (region/a)
            // stays word-aligned. On the memtile shim_word == 1 (no constraint).
            let units = (region / shim_word).max(1);
            let (a, bu) = split2(rng, units);
            let b = bu * shim_word;
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
                // Word-granular transpose (I8/I16, BOTH engines): innermost
                // `word`-sized contiguous run (stride 1) -- satisfies the shim's
                // 4-byte rule AND the memtile's sub-32-bit innermost-stride-1
                // rule; outer strides `word` and `b*word` are multiples of `word`.
                // 3 dims fits both the shim's 3-dim and memtile's 4-dim caps.
                // b*a*word == region.
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
            // Non-innermost strides are b*c and c; on the shim require c (hence
            // b*c) to be a multiple of `shim_word`. Split `region/shim_word`
            // units, then scale c by shim_word. Memtile: shim_word == 1 (the
            // innermost stride is already 1, no further constraint).
            let units = (region / shim_word).max(1);
            let (a, bc) = split2(rng, units);
            let (b, _c0) = split2(rng, bc.max(2));
            let a = a.max(1);
            let b = b.max(1);
            let cu = (units / (a * b)).max(1);
            let c = cu * shim_word;
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
            // Split `region/shim_word` units so the inner run b is a multiple of
            // `shim_word` and large enough to admit a word-aligned overlapping
            // stride (a multiple of `shim_word`, 0 < stride < b). On the shim this
            // keeps the outer stride NoC-word-aligned; on the memtile shim_word
            // == 1 (innermost stride is 1, no constraint).
            let units = (region / shim_word).max(1);
            let (a, bu) = split2(rng, units);
            let b = bu * shim_word;
            // Overlapping outer stride: ~b/2 rounded to a multiple of `shim_word`,
            // clamped to [shim_word, b-shim_word] so it stays strictly below b.
            let mut stride = ((b / 2) / shim_word).max(1) * shim_word;
            if stride >= b {
                stride = b.saturating_sub(shim_word).max(shim_word);
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
            // 2D pattern [a, b]; pad lists are [outer, inner]. The memtile BD
            // requires the INNERMOST (last) pad count to be 32-bit-word-aligned:
            // `inner_pad * byte_size % 4 == 0`. `word` (elems per 32-bit word) is
            // I32->1, I16->2, I8->4 -- so `inner_pad = word` gives 4 bytes exactly
            // for I16/I8. For I32 (word==1) keep 2 for parity with the spike;
            // 2 elems * 4B = 8B, still word-aligned. Outer-axis pad is
            // unconstrained, stays 2.
            let outer_pad = 2u32;
            let inner_pad = if word == 1 { 2 } else { word as u32 };
            let (pb, pa) = match feature {
                Feature::PadBefore => (vec![outer_pad, inner_pad], vec![0, 0]),
                Feature::PadAfter => (vec![0, 0], vec![outer_pad, inner_pad]),
                _ => (vec![outer_pad, inner_pad], vec![outer_pad, inner_pad]),
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
                let bs = c.dtype.byte_size();
                // Memtile sub-32-bit rule: the innermost (last) data-dim stride
                // must be 1 (contiguous word-run); aiecc rejects otherwise.
                if t.engine == Engine::Memtile && bs < 4 {
                    assert_eq!(
                        *p.strides.last().unwrap(),
                        1,
                        "seed{i} key{key}: memtile sub-32b innermost stride must be 1"
                    );
                }
                // Padding rule (memtile MM2S): each innermost pad count must be
                // 32-bit-word-aligned in bytes.
                if !p.pad_before.is_empty() {
                    let pin_b = *p.pad_before.last().unwrap() as usize;
                    let pin_a = *p.pad_after.last().unwrap() as usize;
                    assert_eq!(
                        (pin_b * bs) % 4,
                        0,
                        "seed{i} key{key}: inner pad_before {pin_b} * {bs}B not 4-aligned"
                    );
                    assert_eq!(
                        (pin_a * bs) % 4,
                        0,
                        "seed{i} key{key}: inner pad_after {pin_a} * {bs}B not 4-aligned"
                    );
                }
                assert_eq!(t.out_elems, t.in_elems + p.pad_elems());
            }
        }
    }
}
