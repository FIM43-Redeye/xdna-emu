//! Deterministic chain generation + edge-weighted input pool.
//!
//! [`generate`] turns `(seed, target_key)` into a [`Chain`]: stage 0 is the
//! target key's entry/mode, then a random type-legal walk of 8-16 stages
//! total. Type changes route through coupler entries (the only entries whose
//! input type differs from output), so legality of `in_types[0]` against the
//! previous stage's `out_type` is the single invariant. Bf16 chains stay
//! bf16-only because the table has no bf16-int bridge.
//!
//! [`pool_bytes`] fills 64-byte pool slots with class-weighted 16-bit units --
//! denormals, NaN/Inf, sign extremes, zeros, uniform random -- the input
//! classes that found the phase B/C convert bugs.

use super::chain::{Chain, Stage};
use super::table::table;

/// xorshift64 PRNG (same as the scalar fuzzer's): zero state forbidden.
pub(crate) struct Xorshift64(pub u64);

impl Xorshift64 {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Generate one chain deterministically from `(seed, target_key)`.
///
/// Panics on a malformed/unknown key or out-of-range mode -- callers feed keys
/// from [`super::table::universe_keys`].
pub fn generate(seed: u64, target_key: &str) -> Chain {
    let t = table();
    let (target_idx, target_mode) = parse_key(target_key);
    let mut rng = Xorshift64(if seed == 0 { 1 } else { seed });

    let total = 8 + (rng.next() % 9) as usize; // 8-16 stages
    let mut stages = Vec::with_capacity(total);
    let mut next_slot = 1usize; // slot 0 feeds stage 0's first operand

    let mut push = |stages: &mut Vec<Stage>, idx: usize, mode: u8| {
        let second = if t[idx].in_types.len() == 2 {
            let slot = next_slot;
            next_slot += 1;
            Some(slot)
        } else {
            None
        };
        stages.push(Stage { entry_idx: idx, mode, second_pool_slot: second });
    };

    push(&mut stages, target_idx, target_mode);
    let mut cur = t[target_idx].out_type;

    while stages.len() < total {
        let candidates: Vec<usize> = (0..t.len()).filter(|&i| t[i].in_types[0] == cur).collect();
        assert!(!candidates.is_empty(), "type dead-end at {cur:?} (table invariant broken)");
        let idx = candidates[(rng.next() % candidates.len() as u64) as usize];
        let mode = (rng.next() % t[idx].modes as u64) as u8;
        push(&mut stages, idx, mode);
        cur = t[idx].out_type;
    }

    let slots = 1 + stages.iter().filter(|s| s.second_pool_slot.is_some()).count();
    let for_float = t[target_idx].in_types[0].is_float();
    let pool = pool_bytes(&mut rng, slots, for_float);

    Chain { seed, target_key: target_key.to_string(), stages, pool }
}

/// Resolve `{name}/{out_type:?}/m{mode}` to `(table index, mode)`.
fn parse_key(key: &str) -> (usize, u8) {
    let mut parts = key.rsplitn(3, '/');
    let mode_part = parts.next().expect("key has mode part");
    let out_part = parts.next().expect("key has out_type part");
    let name = parts.next().expect("key has name part");
    let mode: u8 = mode_part
        .strip_prefix('m')
        .and_then(|m| m.parse().ok())
        .unwrap_or_else(|| panic!("bad mode in key {key:?}"));
    let idx = table()
        .iter()
        .position(|e| e.name == name && format!("{:?}", e.out_type) == out_part)
        .unwrap_or_else(|| panic!("unknown key {key:?}"));
    assert!(mode < table()[idx].modes, "mode {mode} out of range for {key:?}");
    (idx, mode)
}

/// Fill `slots` 64-byte pool slots with class-weighted 16-bit units:
/// 25% denormal-pattern, 15% NaN/Inf, 20% sign extremes, 15% zeros,
/// 25% uniform random. Int chains get the analogous integer edge values;
/// the 32-bit int extremes 0x7FFFFFFF/0x80000000 occupy two adjacent units.
pub(crate) fn pool_bytes(rng: &mut Xorshift64, slots: usize, for_float: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(slots * 64);
    let units = slots * 32;
    while out.len() < units * 2 {
        let class = rng.next() % 100;
        let unit: u16 = match class {
            0..=24 => {
                // Denormal-pattern: bf16 exponent 0, mantissa nonzero; int small.
                let sign = ((rng.next() & 1) as u16) << 15;
                if for_float {
                    sign | (1 + (rng.next() % 0x7F) as u16)
                } else {
                    let mag = 1 + (rng.next() % 8) as u16;
                    if sign != 0 {
                        mag.wrapping_neg()
                    } else {
                        mag
                    }
                }
            }
            25..=39 => {
                if for_float {
                    if rng.next() & 1 == 0 {
                        0x7FC0
                    } else {
                        0x7F80
                    } // NaN / +Inf
                } else {
                    // 32-bit INT_MAX/INT_MIN little-endian across two units.
                    let (lo, hi): (u16, u16) = if rng.next() & 1 == 0 {
                        (0xFFFF, 0x7FFF)
                    } else {
                        (0x0000, 0x8000)
                    };
                    out.extend_from_slice(&lo.to_le_bytes());
                    if out.len() >= units * 2 {
                        break;
                    }
                    hi
                }
            }
            40..=59 => {
                // Sign extremes: largest-magnitude finite values of each sign.
                if for_float {
                    if rng.next() & 1 == 0 {
                        0x7F7F
                    } else {
                        0xFF7F
                    }
                } else if rng.next() & 1 == 0 {
                    0x7FFF
                } else {
                    0x8000
                }
            }
            60..=74 => 0,
            _ => rng.next() as u16,
        };
        out.extend_from_slice(&unit.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::vector::table::{universe_keys, VecType};

    /// Pick a deterministic pseudo-random target key for a seed.
    fn key_for_seed(universe: &[String], seed: u64) -> &str {
        let mut rng = Xorshift64(seed.wrapping_mul(0x9E3779B97F4A7C15).max(1));
        &universe[(rng.next() % universe.len() as u64) as usize]
    }

    #[test]
    fn thousand_seeds_type_legal_8_to_16_stages() {
        let t = table();
        let universe = universe_keys();
        for seed in 0..1000u64 {
            let key = key_for_seed(&universe, seed);
            let chain = generate(seed, key);
            assert!(
                (8..=16).contains(&chain.stages.len()),
                "seed {seed} key {key}: {} stages",
                chain.stages.len()
            );
            let mut cur = t[chain.stages[0].entry_idx].in_types[0]; // pool type
            for (k, s) in chain.stages.iter().enumerate() {
                assert_eq!(
                    t[s.entry_idx].in_types[0], cur,
                    "seed {seed} key {key}: stage {k} input mismatch"
                );
                cur = t[s.entry_idx].out_type;
            }
            let binary = chain.stages.iter().filter(|s| s.second_pool_slot.is_some()).count();
            assert_eq!(chain.pool_slots(), binary + 1, "seed {seed} key {key}");
            assert_eq!(chain.pool.len(), chain.pool_slots() * 64, "seed {seed} key {key}");
            assert_eq!(chain.out_bytes(), chain.stages.len() * 64, "seed {seed} key {key}");
        }
    }

    #[test]
    fn same_seed_and_key_identical_chain_and_pool() {
        let universe = universe_keys();
        for seed in [0u64, 1, 42, 999] {
            let key = key_for_seed(&universe, seed);
            assert_eq!(generate(seed, key), generate(seed, key), "seed {seed} key {key}");
        }
    }

    #[test]
    fn bf16_target_keeps_all_stages_bf16() {
        let t = table();
        for (seed, key) in
            (0..200u64).zip(universe_keys().into_iter().filter(|k| k.contains("Bf16x32")).cycle())
        {
            let chain = generate(seed, &key);
            for (k, s) in chain.stages.iter().enumerate() {
                assert_eq!(
                    t[s.entry_idx].out_type,
                    VecType::Bf16x32,
                    "seed {seed} key {key}: stage {k} left bf16"
                );
            }
        }
    }

    #[test]
    fn stage0_is_target_entry_with_target_mode() {
        let t = table();
        for key in universe_keys().iter().step_by(7) {
            let chain = generate(7, key);
            let s0 = &chain.stages[0];
            let e = &t[s0.entry_idx];
            assert_eq!(&format!("{}/{:?}/m{}", e.name, e.out_type, s0.mode), key);
            assert!(chain.keys().contains(key), "key {key} missing from keys()");
        }
    }

    #[test]
    fn i32x16_target_key_is_covered_in_keys() {
        for (seed, key) in
            (0..200u64).zip(universe_keys().into_iter().filter(|k| k.contains("I32x16")).cycle())
        {
            let chain = generate(seed, &key);
            assert!(chain.keys().contains(&key), "seed {seed}: {key} not in {:?}", chain.keys());
        }
    }

    #[test]
    fn float_pool_has_zero_and_naninf_units_per_window() {
        for seed in 1..=50u64 {
            let mut rng = Xorshift64(seed);
            let pool = pool_bytes(&mut rng, 8, true);
            assert_eq!(pool.len(), 8 * 64);
            for (w, window) in pool.chunks(4 * 64).enumerate() {
                let units: Vec<u16> = window.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                assert!(units.contains(&0), "seed {seed} window {w}: no zero unit");
                assert!(
                    units.iter().any(|&u| u == 0x7F80 || u == 0x7FC0),
                    "seed {seed} window {w}: no Inf/NaN unit"
                );
            }
        }
    }

    #[test]
    fn int_pool_contains_int32_extremes_and_denormal_smalls() {
        let mut rng = Xorshift64(1);
        let pool = pool_bytes(&mut rng, 16, false);
        let words: Vec<u32> = pool.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert!(words.contains(&0x7FFF_FFFF), "no INT_MAX in int pool");
        assert!(words.contains(&0x8000_0000), "no INT_MIN in int pool");
        let units: Vec<u16> = pool.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        assert!(units.iter().any(|&u| (1..=8).contains(&u)), "no small positive");
        assert!(units.contains(&0), "no zero unit");
    }
}
