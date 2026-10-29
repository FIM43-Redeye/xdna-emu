//! bf16 x bf16 -> fp32 multiply-accumulate lane (`vmul.f`, `vmac.f`, `vmsc.f`,
//! `vaddmac.f`, ... and the 16-lane elementwise form).
//!
//! Derived from NPU1 silicon only (`tools/golden/silicon/vec_mac_bf16*.json`);
//! evidence and the rejected alternatives are in
//! `docs/fidelity-gaps/vector-compute.md`.
//!
//! One lane is the sum of its bf16 products, the accumulator inputs (acc1, and
//! acc2 for the `add*` forms), each with an optional negation:
//!
//! 1. Every term is an exact value `sig * 2^e` with an IEEE exponent `E`
//!    (product: `ea + eb - 127`, significand in [1, 4); accumulator: its own).
//!    Exponent fields of 255 (Inf/NaN) and 0 (denormal, field read as 1) are
//!    ordinary numbers here; zero terms do not take part.
//! 2. The reference exponent is the largest `E` of any term.
//! 3. Each term is aligned to the reference and rounded to nearest-even onto a
//!    grid of weight `2^(E_ref - 23)`. The largest term is exact; the others lose
//!    their low bits (a term below half a grid step vanishes).
//! 4. The grid integers are summed exactly (two's complement), so summation
//!    order and K position do not matter.
//! 5. The sum is rounded once to fp32 (nearest-even). A sum below 2^-126 in
//!    magnitude flushes to a signed zero BEFORE rounding (it does not round up
//!    to the smallest normal); an exact zero is +0; overflow gives Inf.
//! 6. Specials: any NaN input or Inf*0 makes the lane NaN, otherwise Inf terms
//!    of one sign make that Inf (both signs: NaN). A NaN lane is the ordinary
//!    numeric result of step 5 (Inf/NaN exponent fields taken at face value)
//!    with its exponent forced to 255 and mantissa bit 0 set; the mantissa is
//!    zero when that numeric result overflowed or flushed, which is why most
//!    NaNs read `0x?f800001` and only cancelling special terms carry a payload.

/// One accumulator operand: the fp32 bits and whether the instruction negates it.
pub type AccTerm = (u32, bool);

#[derive(Clone, Copy)]
struct Term {
    neg: bool,
    sig: u64,
    /// Weight exponent of the significand LSB: value = sig * 2^lsb.
    lsb: i32,
    /// IEEE exponent of the term.
    ieee: i32,
}

#[derive(Clone, Copy, PartialEq)]
enum Flag {
    None,
    Nan,
    InfPos,
    InfNeg,
}

fn bf16_term(a: u16, b: u16, neg_prod: bool) -> (Option<Term>, Flag) {
    let (ea, eb) = (((a >> 7) & 0xff) as i32, ((b >> 7) & 0xff) as i32);
    let (ma, mb) = ((a & 0x7f) as u64, (b & 0x7f) as u64);
    let (za, zb) = (a & 0x7fff == 0, b & 0x7fff == 0);
    let neg = (((a ^ b) >> 15) & 1 != 0) ^ neg_prod;
    let nan = |e: i32, m: u64| e == 255 && m != 0;
    let inf = |e: i32, m: u64| e == 255 && m == 0;
    let flag = if nan(ea, ma) || nan(eb, mb) || (inf(ea, ma) && zb) || (inf(eb, mb) && za) {
        Flag::Nan
    } else if inf(ea, ma) || inf(eb, mb) {
        if neg {
            Flag::InfNeg
        } else {
            Flag::InfPos
        }
    } else {
        Flag::None
    };
    if za || zb {
        return (None, flag);
    }
    let sig_a = if ea != 0 { 128 + ma } else { ma };
    let sig_b = if eb != 0 { 128 + mb } else { mb };
    let lsb = (ea.max(1) - 127 - 7) + (eb.max(1) - 127 - 7);
    (Some(Term { neg, sig: sig_a * sig_b, lsb, ieee: lsb + 14 }), flag)
}

fn acc_term(bits: u32, negate: bool) -> (Option<Term>, Flag) {
    let neg = (bits >> 31 != 0) ^ negate;
    let (ef, m) = (((bits >> 23) & 0xff) as i32, (bits & 0x7f_ffff) as u64);
    let flag = match (ef == 255, m != 0) {
        (true, true) => Flag::Nan,
        (true, false) if neg => Flag::InfNeg,
        (true, false) => Flag::InfPos,
        _ => Flag::None,
    };
    let term = if ef == 0 {
        // Denormal accumulator: used as is, reference exponent of field 1.
        (m != 0).then_some(Term { neg, sig: m, lsb: -149, ieee: -126 })
    } else {
        Some(Term { neg, sig: (1 << 23) | m, lsb: ef - 127 - 23, ieee: ef - 127 })
    };
    (term, flag)
}

/// Right-shift `v` by `n` bits rounding to nearest-even.
fn shr_rne(v: u64, n: u32) -> u64 {
    if n == 0 {
        return v;
    }
    if n > 63 {
        return 0;
    }
    let q = v >> n;
    let rem = v & ((1u64 << n) - 1);
    let half = 1u64 << (n - 1);
    if rem > half || (rem == half && q & 1 == 1) {
        q + 1
    } else {
        q
    }
}

/// Sum of the terms as (sign, biased exponent before clamping, 23-bit mantissa,
/// flushed), or `None` for an exact zero.
fn numeric_sum(terms: &[Term]) -> Option<(bool, i32, u32, bool)> {
    let e_ref = terms.iter().map(|t| t.ieee).max()?;
    let mut total: i128 = 0;
    for t in terms {
        let shift = t.lsb - (e_ref - 23);
        let q = if shift >= 0 {
            (t.sig as u128) << shift
        } else {
            shr_rne(t.sig, (-shift) as u32) as u128
        } as i128;
        total += if t.neg { -q } else { q };
    }
    if total == 0 {
        return None;
    }
    let neg = total < 0;
    let mag = total.unsigned_abs();
    let len = 128 - mag.leading_zeros() as i32;
    let unb = e_ref - 23 + len - 1;
    if unb < -126 {
        return Some((neg, unb + 127, 0, true));
    }
    let (mut sig, mut unb) = if len > 24 {
        let n = (len - 24) as u32;
        let q = shr_rne_wide(mag, n);
        (q, unb)
    } else {
        (mag << (24 - len), unb)
    };
    if sig >= 1 << 24 {
        sig >>= 1;
        unb += 1;
    }
    Some((neg, unb + 127, (sig & 0x7f_ffff) as u32, false))
}

fn shr_rne_wide(v: u128, n: u32) -> u128 {
    let q = v >> n;
    let rem = v & ((1u128 << n) - 1);
    let half = 1u128 << (n - 1);
    if rem > half || (rem == half && q & 1 == 1) {
        q + 1
    } else {
        q
    }
}

/// One output lane: `products` are the (A, B) bf16 pairs, `accs` the accumulator
/// operands, `neg_prod` negates every product. Returns the fp32 bits.
pub fn mac_lane(products: &[(u16, u16)], accs: &[AccTerm], neg_prod: bool) -> u32 {
    let mut terms = Vec::with_capacity(products.len() + accs.len());
    let mut flags = Vec::with_capacity(products.len() + accs.len());
    for &(a, b) in products {
        let (t, f) = bf16_term(a, b, neg_prod);
        terms.extend(t);
        flags.push(f);
    }
    for &(bits, negate) in accs {
        let (t, f) = acc_term(bits, negate);
        terms.extend(t);
        flags.push(f);
    }
    let any_nan = flags.contains(&Flag::Nan);
    let inf_pos = flags.contains(&Flag::InfPos);
    let inf_neg = flags.contains(&Flag::InfNeg);
    let sum = numeric_sum(&terms);
    let sign_bit = |neg: bool| (neg as u32) << 31;

    if !any_nan && (inf_pos != inf_neg) {
        return sign_bit(inf_neg) | 0x7f80_0000;
    }
    if any_nan || (inf_pos && inf_neg) {
        let (neg, mant) = match sum {
            None => (false, 0),
            Some((neg, biased, mant, flushed)) => (neg, if flushed || biased >= 255 { 0 } else { mant }),
        };
        return sign_bit(neg) | 0x7f80_0000 | mant | 1;
    }
    match sum {
        None => 0,
        Some((neg, _, _, true)) => sign_bit(neg),
        Some((neg, biased, _, _)) if biased >= 255 => sign_bit(neg) | 0x7f80_0000,
        Some((neg, biased, mant, false)) => sign_bit(neg) | ((biased as u32) << 23) | mant,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf(sign: u16, exp: u16, mant: u16) -> u16 {
        (sign << 15) | (exp << 7) | mant
    }

    #[test]
    fn exact_small_integers() {
        // 2*3 + 4*5 = 26
        let r = mac_lane(&[(bf(0, 128, 0), bf(0, 128, 64)), (bf(0, 129, 0), bf(0, 129, 32))], &[], false);
        assert_eq!(f32::from_bits(r), 26.0);
    }

    #[test]
    fn small_term_rounds_onto_the_grid_of_the_large_one() {
        // 1.0 + 1.5 * 2^-23 : the small term is 1.5 grid steps, ties to even -> 2 steps.
        let big = (bf(0, 127, 0), bf(0, 127, 0));
        let small = (bf(0, 127 - 12, 64), bf(0, 127 - 11, 0));
        assert_eq!(mac_lane(&[big, small], &[], false), 0x3f80_0002);
        // 1.0 + 0.5 * 2^-23 : tie, even -> 1.0.
        let tie = (bf(0, 127 - 12, 0), bf(0, 127 - 12, 0));
        assert_eq!(mac_lane(&[big, tie], &[], false), 0x3f80_0000);
    }

    /// Pair whose exact product is 2^exp (sign optional), both factors normal.
    fn pow2(exp: i32, sign: u16) -> (u16, u16) {
        let ea = exp.div_euclid(2);
        (bf(sign, (127 + ea) as u16, 0), bf(0, (127 + exp - ea) as u16, 0))
    }

    #[test]
    fn flush_precedes_rounding() {
        // 2^-126 - 2^-150 is a tie that RNE would round up to the smallest
        // normal; silicon flushes it (NPU1, vec_mac_bf16_forms edge tiles).
        let r = mac_lane(&[pow2(-127, 0), pow2(-127, 0), pow2(-150, 1)], &[], false);
        assert_eq!(r, 0);
        let r = mac_lane(&[pow2(-127, 1), pow2(-127, 1), pow2(-150, 0)], &[], false);
        assert_eq!(r, 0x8000_0000, "negative underflow keeps its sign");
        assert_eq!(mac_lane(&[pow2(-127, 0), pow2(-127, 0)], &[], false), 0x0080_0000);
    }

    #[test]
    fn nan_encoding_and_invalid_sign() {
        let nan = bf(0, 255, 0x40);
        let one = bf(0, 127, 0);
        assert_eq!(mac_lane(&[(nan, one)], &[], false), 0x7f80_0001);
        assert_eq!(mac_lane(&[(nan, bf(1, 127, 0))], &[], false), 0xff80_0001);
        // 0 * Inf is invalid; sign is positive whatever the operand signs.
        assert_eq!(mac_lane(&[(bf(1, 255, 0), 0)], &[], false), 0x7f80_0001);
        // +Inf alone is Inf; +Inf and -Inf is NaN.
        assert_eq!(mac_lane(&[(bf(0, 255, 0), one)], &[], false), 0x7f80_0000);
        assert_eq!(mac_lane(&[(bf(0, 255, 0), one), (bf(1, 255, 0), one)], &[], false), 0x7f80_0001);
    }

    #[test]
    fn zero_sums_are_positive() {
        let one = bf(0, 127, 0);
        assert_eq!(mac_lane(&[(bf(1, 0, 0), one); 8], &[], false), 0);
        assert_eq!(mac_lane(&[(one, one), (bf(1, 127, 0), one)], &[], false), 0);
    }

    /// Silicon golden: every lane of every form of `vec_mac_bf16_forms`.
    #[test]
    fn matches_silicon_on_every_form_lane() {
        let path = format!("{}/tools/golden/silicon/vec_mac_bf16_forms.json", env!("CARGO_MANIFEST_DIR"));
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let col = |i: usize| -> Vec<u64> {
            v["inputs"][i].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect()
        };
        let (a, b, c0, c1) = (col(0), col(1), col(2), col(3));
        let silicon: Vec<u32> = v["silicon"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let forms = v["forms"].as_array().unwrap();
        let tiles = v["tile"]["tiles"].as_u64().unwrap() as usize;
        let (mut checked, mut bad) = (0usize, Vec::new());
        for t in 0..tiles {
            for (fi, f) in forms.iter().enumerate() {
                let instr = f["instr"].as_str().unwrap();
                let flag = |k: &str| f[k].as_u64().unwrap() != 0;
                let neg_prod = matches!(instr, "negmul" | "msc" | "addmsc") ^ flag("sub_mul");
                let uses1 = matches!(instr, "mac" | "msc" | "addmac" | "addmsc") && !flag("zero_acc1");
                let uses2 = matches!(instr, "addmac" | "addmsc");
                let elementwise = f["geometry"].as_str().unwrap() == "el";
                for lane in 0..16 {
                    let (i, j) = (lane / 4, lane % 4);
                    let products: Vec<(u16, u16)> = if elementwise {
                        vec![
                            (a[t * 32 + lane] as u16, b[t * 32 + lane] as u16),
                            (a[t * 32 + lane + 16] as u16, b[t * 32 + lane + 16] as u16),
                        ]
                    } else {
                        (0..8)
                            .map(|k| (a[t * 32 + i * 8 + k] as u16, b[t * 32 + k * 4 + j] as u16))
                            .collect()
                    };
                    let mut accs = Vec::new();
                    if uses1 {
                        accs.push((c0[t * 16 + lane] as u32, flag("sub_acc1")));
                    }
                    if uses2 {
                        accs.push((c1[t * 16 + lane] as u32, flag("sub_acc2")));
                    }
                    let got = mac_lane(&products, &accs, neg_prod);
                    let want = silicon[(t * forms.len() + fi) * 16 + lane];
                    checked += 1;
                    if got != want {
                        bad.push((t, f["name"].as_str().unwrap().to_string(), lane, got, want));
                    }
                }
            }
        }
        assert!(checked > 100_000);
        assert!(
            bad.is_empty(),
            "{} of {checked} lanes differ, first: {:?}",
            bad.len(),
            &bad[..bad.len().min(5)]
        );
    }

    /// The two older fresh-multiply captures, now exact (previously within a few ulps).
    #[test]
    fn matches_older_vmul_captures_exactly() {
        for kernel in ["vec_mac_bf16", "vec_mac_bf16_ovf"] {
            let path = format!("{}/tools/golden/silicon/{kernel}.json", env!("CARGO_MANIFEST_DIR"));
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let col = |i: usize| -> Vec<u64> {
                v["inputs"][i].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect()
            };
            let (a, b) = (col(0), col(1));
            let silicon: Vec<u32> = v["silicon"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let mut bad = Vec::new();
            for t in 0..silicon.len() / 16 {
                for lane in 0..16 {
                    let (i, j) = (lane / 4, lane % 4);
                    let products: Vec<(u16, u16)> = (0..8)
                        .map(|k| (a[t * 32 + i * 8 + k] as u16, b[t * 32 + k * 4 + j] as u16))
                        .collect();
                    let got = mac_lane(&products, &[], false);
                    if got != silicon[t * 16 + lane] {
                        bad.push((t, lane, got, silicon[t * 16 + lane]));
                    }
                }
            }
            assert!(
                bad.is_empty(),
                "{kernel}: {} lanes differ, first {:?}",
                bad.len(),
                &bad[..bad.len().min(5)]
            );
        }
    }
}
