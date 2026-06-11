//! Hardware-accurate bf16 MAC pipeline for AIE2.
//!
//! Reimplementation of the AIE2 hardware MAC pipeline for bfloat16 inputs.
//! The hardware uses a biased exponent system with global alignment, matching
//! the vmac pipeline from the architecture's multiplier-accumulator unit.
//!
//! Key hardware behavioral facts (derived from architecture reference):
//!   - Product exponents are biased: cexp = aexp + bexp + 1
//!   - Accumulator exponent is biased: qexp_biased = qexp + 128
//!   - All values (products + accumulator) align to a global maximum exponent
//!   - Products are signed, aligned at float_width=23 with RNE rounding
//!   - Accumulator is unsigned, aligned at float_width=23, then signed
//!   - Wide integer summation preserves all bits (no intermediate rounding)
//!   - Single final normalization: exps = cexpmax + fl - 23 - 128
//!
//! Pipeline stages:
//!   1. Split bf16 inputs, detect NaN/Inf, compute biased product exponents
//!   2. Integer multiply (8-bit mantissa * 8-bit mantissa = 16-bit product)
//!   3. Bias accumulator exponent (+128), find global cexpmax
//!   4. Align signed products to cexpmax at float_width=23 (bfshift, RNE)
//!   5. Align unsigned accumulator to cexpmax at float_width=23 (fpshift, RNE)
//!   6. Sum all aligned values exactly (wide integer arithmetic)
//!   7. Normalize to fp32 via fpcorr (exps = cexpmax + fl - 23 - 128)

use super::super::vector_float::{bf16_split, fp32_split, fp32_make};

/// Shift-and-round with RNE (round-to-nearest-even).
///
/// Shifts `man` right so that the leading bit ends up at position `prec`.
/// `lo_pos` is the current leading bit position.  Returns (rounded_value,
/// exponent_increment_from_overflow).
///
/// Direct translation of `shift_round_rne` from bfloat16.py.
fn shift_round_rne(man: i64, lo_pos: i32, prec: i32, norm_round_overflow: bool) -> (i64, bool) {
    let sh_dn = lo_pos - prec - 1;

    // Guard against shifts exceeding i64 width.
    if sh_dn >= 63 {
        // Shifted completely out -- rounds to zero.
        return (0, false);
    }

    let rmask: i64 = if sh_dn > 0 && sh_dn < 63 {
        (1i64 << sh_dn) - 1
    } else {
        0
    };

    let r: i64 = if sh_dn < 0 {
        let lsh = (-sh_dn) as u32;
        if lsh >= 63 {
            return (0, false);
        } // Would overflow
        man.wrapping_shl(lsh)
    } else {
        man >> sh_dn
    };
    let q = r.wrapping_add(2); // Pre-compute round-up value

    let grd = (r & 1) != 0;
    let lsb = (r & 2) != 0;
    let stk = (man & rmask) != 0;

    // RNE: round up when guard=1 AND (sticky=1 OR lsb=1)
    let rup = grd && (stk || lsb);

    let mut expincr = false;

    let overflow_threshold = if (prec + 2) < 63 {
        1i64 << (prec + 2)
    } else {
        i64::MAX
    };
    if norm_round_overflow && rup && (q >= overflow_threshold) {
        let val = q >> 2;
        expincr = true;
        return (val, expincr);
    }

    let val = if rup { q >> 1 } else { r >> 1 };
    (val, expincr)
}

/// Find the position of the leading one bit (0-indexed from LSB).
/// Returns -1 for input 0.
fn flo(man: i64) -> i32 {
    if man <= 0 {
        return -1;
    }
    63 - man.leading_zeros() as i32
}

/// Hardware-accurate bf16 MAC for one output lane (vmac pipeline).
///
/// Takes the previous accumulator value `q` (fp32 bits), paired bf16
/// input elements `a_elems` and `b_elems`, and whether to subtract.
/// Returns the new fp32 accumulator value (as bits).
///
/// Implements the AIE2 vmac pipeline's biased exponent system with global
/// alignment, matching the hardware multiplier-accumulator unit behavior.
/// Execute one lane of the bf16 MAC pipeline.
///
/// Parameters:
/// - `q`: primary accumulator (acc1) as fp32 bits
/// - `a_elems`/`b_elems`: bf16 input pairs for this lane
/// - `subtract`: negate products (MSC variants)
/// - `acc2`: optional second accumulator for AddMac/SubMac (fp32 bits)
/// - `sub_acc2`: if true, subtract acc2 instead of adding (SubMac)
///
/// Hardware pipeline (from aietools me_inline_primitives.h):
/// 1. bfshiftcompute_hw: align products, acc1, AND acc2 to common cexpmax
/// 2. psal_hw -> split_adder5_hw: products + acc1 + acc2 in 68-bit adder
/// 3. bfnorm_hw: normalize the COMBINED result
///
/// This matches the hardware's single-pass merge, where acc2 participates
/// in the same wide addition as products and acc1, BEFORE normalization.
pub(crate) fn bf16_mac_hw_lane(
    q: u32,
    a_elems: &[u16],
    b_elems: &[u16],
    subtract: bool,
    acc2: Option<u32>,
    sub_acc2: bool,
) -> u32 {
    const FLOAT_WIDTH: i32 = 23; // Internal alignment precision

    let n = a_elems.len();
    debug_assert_eq!(n, b_elems.len());

    // Phase 1: Split inputs, detect NaN/Inf, compute biased product exponents.
    //
    // Hardware biases product exponents by +1: cexp = aexp + bexp + 1.
    // This accounts for the multiplier array's internal representation.
    let mut nan_flag = false;
    let mut inf_flag = false;
    let mut inf_sgn = false;

    struct Product {
        signed_mantissa: i64,
        biased_exp: i32,
    }
    let mut products = Vec::with_capacity(n);

    for i in 0..n {
        let (asgn, mut aexp, aman) = bf16_split(a_elems[i]);
        let (bsgn, mut bexp, bman) = bf16_split(b_elems[i]);

        let csgn = asgn != bsgn; // Product sign = XOR

        // NaN/Inf detection (before denorm handling).
        let ainf = aexp == 255 && aman == 0;
        let binf = bexp == 255 && bman == 0;
        let anan = aexp == 255 && aman != 0;
        let bnan = bexp == 255 && bman != 0;
        let cinf = ainf || binf;
        let cnan = anan || bnan;
        // Inf * zero = NaN (hardware behavior).
        let cex0 = (aexp == 0 && aman == 0) || (bexp == 0 && bman == 0);

        nan_flag = nan_flag || cnan || (inf_flag && cinf && csgn != inf_sgn) || (cinf && cex0);
        inf_flag = (inf_flag || cinf) && !nan_flag;
        if cinf && inf_sgn == false && !nan_flag {
            inf_sgn = csgn;
        }

        // Track whether original exponent was zero BEFORE denorm handling.
        // Hardware uses this to decide implicit leading 1 bit (cex0 flag).
        let aex0 = aexp == 0;
        let bex0 = bexp == 0;

        // Denorm handling: bump exponent so product exponent math works,
        // but mantissa does NOT get implicit 1 (tracked via aex0/bex0).
        if aexp == 0 && aman != 0 {
            aexp = 1;
        }
        if bexp == 0 && bman != 0 {
            bexp = 1;
        }

        // Biased product exponent: hardware adds +1.
        let cexp: i32 = if aexp == 0 || bexp == 0 {
            0 // Zero input -> zero product
        } else {
            (aexp as i32) + (bexp as i32) + 1
        };

        // Unsigned product mantissa. Implicit leading 1 bit is added only
        // when the ORIGINAL exponent was non-zero (hardware cex0 flag).
        // Hardware includes mantissa for NaN/Inf (exp=255) -- the NaN/Inf
        // detection is separate from the mantissa datapath.  The multiplier
        // array always computes products; bfnorm applies NaN/Inf flags at
        // the output.  (See bfshiftcompute_lane oaccman = concat(!exp0, man).)
        let a_man: i64 = if !aex0 {
            0x80 | aman as i64 // Normal, Inf, or NaN: add implicit 1
        } else if aman != 0 {
            aman as i64 // Denorm: raw mantissa, no implicit 1
        } else {
            0 // Zero
        };
        let b_man: i64 = if !bex0 {
            0x80 | bman as i64
        } else if bman != 0 {
            bman as i64
        } else {
            0
        };
        let unsigned_product = a_man * b_man; // Up to 16 bits

        // Sign applied to product BEFORE alignment (matches hardware sge/mpy stages).
        let signed_product = if csgn != subtract {
            -(unsigned_product)
        } else {
            unsigned_product
        };

        products.push(Product { signed_mantissa: signed_product, biased_exp: cexp });
    }

    // Phase 2: Accumulator (acc1) -- extract and bias exponent.
    //
    // Hardware biases accumulator exponent by +128: qexp_biased = qexp + 128.
    // Zero accumulator keeps biased exponent = 0.
    let (qsgn, qexp_raw, qman_raw) = fp32_split(q);
    let qzero = qexp_raw == 0 && qman_raw == 0;
    let qinf = qexp_raw == 255 && qman_raw == 0;
    let qnan = qexp_raw == 255 && qman_raw != 0;

    // Denorm handling for accumulator (same pattern as products).
    let mut qexp = qexp_raw;
    if qexp == 0 && qman_raw != 0 {
        qexp = 1;
    }

    let qexp_biased: i32 = if !qzero { (qexp as i32) + 128 } else { 0 };

    // Unsigned accumulator mantissa.  Hardware bfshiftcompute_lane uses
    // oaccman = concat(!accexp0, accman) -- implicit leading 1 for ALL
    // non-zero exponents (including NaN/Inf at exp=255), raw mantissa
    // without implicit 1 for denorms, zero for true zeros.
    let qman_unsigned: i64 = if qexp_raw > 0 {
        (1i64 << 23) | (qman_raw as i64) // Normal, NaN, or Inf
    } else if qman_raw != 0 {
        qman_raw as i64 // Denorm (no implicit 1)
    } else {
        0 // True zero
    };

    // NaN/Inf from acc1.
    nan_flag = nan_flag || qnan || (inf_flag && qinf && qsgn != inf_sgn);
    inf_flag = (inf_flag || qinf) && !nan_flag;
    if qinf && !nan_flag {
        inf_sgn = inf_sgn || qsgn;
    }

    // Phase 2b: Second accumulator (acc2) for AddMac/SubMac.
    //
    // Hardware (me_inline_primitives.h addmac_bf line 14976) passes acc2
    // as the scd_ parameter into the MAC pipeline. It participates in the
    // same bfshiftcompute_hw -> psal_hw -> bfnorm_hw flow as acc1.
    // The acc2 merge happens in the 68-bit PSA adder BEFORE normalization.
    let (d_sgn, d_exp_biased, d_man_unsigned) = if let Some(d) = acc2 {
        let (dsgn, dexp_raw, dman_raw) = fp32_split(d);
        let dzero = dexp_raw == 0 && dman_raw == 0;
        let dinf = dexp_raw == 255 && dman_raw == 0;
        let dnan = dexp_raw == 255 && dman_raw != 0;

        let mut dexp = dexp_raw;
        if dexp == 0 && dman_raw != 0 {
            dexp = 1;
        }

        let dexp_biased: i32 = if !dzero { (dexp as i32) + 128 } else { 0 };
        // Same mantissa logic as acc1: include for NaN/Inf/denorm.
        let dman: i64 = if dexp_raw > 0 {
            (1i64 << 23) | (dman_raw as i64)
        } else if dman_raw != 0 {
            dman_raw as i64 // Denorm
        } else {
            0
        };

        // NaN/Inf from acc2.
        nan_flag = nan_flag || dnan || (inf_flag && dinf && (dsgn != sub_acc2) != inf_sgn);
        inf_flag = (inf_flag || dinf) && !nan_flag;
        if dinf && !nan_flag {
            inf_sgn = inf_sgn || (dsgn != sub_acc2);
        }

        // Effective sign: XOR with sub_acc2 for SubMac.
        let effective_sgn = dsgn != sub_acc2;
        (effective_sgn, dexp_biased, dman)
    } else {
        (false, 0i32, 0i64)
    };

    // NaN/Inf handling.
    //
    // Hardware does NOT early-exit on NaN.  NaN/Inf mantissas participate
    // in the PSA adder normally (with implicit leading 1).  The nan/inf
    // flags propagate through to bfnorm_lane which applies them AFTER
    // normalization:
    //
    //   out_zeros = !(pinf | minf | overflow | underflow)
    //   rr = out_zeros ? normalized_mantissa : 0
    //   rr[0] |= nan
    //   exp = (nan|pinf|minf|overflow) ? 0xFF : normal_exp
    //
    // This means: when nan_flag is set but no overflow/underflow, the
    // normalized PSA sum is PRESERVED in the NaN mantissa bits (with bit 0
    // OR'd).  Only when overflow also occurs is the mantissa zeroed to
    // produce canonical NaN (0x7F800001).
    //
    // Special case: when NaN comes ONLY from acc1 and all products are zero
    // and no acc2, the hardware passes the accumulator through unchanged
    // (MAC is a no-op).  Verified against real NPU1 hardware.
    if nan_flag {
        let all_products_zero = products.iter().all(|p| p.biased_exp == 0);
        if qnan && !qinf && all_products_zero && acc2.is_none() {
            return q;
        }
        // Fall through: NaN mantissa participates in PSA sum, bfnorm
        // applies NaN logic after normalization (see Phase 7 below).
    }
    if inf_flag && !nan_flag {
        return fp32_make(inf_sgn, 255, 0); // Infinity
    }

    // Phase 3: Global cexpmax across products, acc1, AND acc2.
    let mut cexpmax: i32 = qexp_biased;
    if d_exp_biased > cexpmax {
        cexpmax = d_exp_biased;
    }
    for p in &products {
        if p.biased_exp > cexpmax {
            cexpmax = p.biased_exp;
        }
    }

    if cexpmax == 0 {
        // All contributions have zero exponent, so the PSA sum is zero --
        // but a pending NaN (e.g. Inf * 0, whose product keeps cexp = 0)
        // must still come out as NaN.  bfnorm: zero mantissa, bit 0 |= nan,
        // exp = 0xFF.  Silicon-verified (fuzzer seed 6160, mmul_bf16).
        if nan_flag {
            return fp32_make(false, 255, 1);
        }
        return 0; // All inputs zero
    }

    // Phase 4: Align signed products to cexpmax at float_width=23 (bfshift).
    //
    // Hardware bfshift formula: vk = product << (float_width - 14)
    // then shift_round_rne(vk, 23 + shift, float_width) where shift = cexpmax - cexp.
    // The << 9 scales the 14-bit product to the 23-bit alignment field.
    let mut product_sum: i64 = 0;
    for p in &products {
        if p.biased_exp == 0 {
            continue; // Zero product
        }
        let s = cexpmax - p.biased_exp;
        let vk = p.signed_mantissa << (FLOAT_WIDTH - 14); // << 9
        let (aligned, _) = shift_round_rne(vk, 23 + s, FLOAT_WIDTH, false);
        product_sum += aligned;
    }

    // Phase 5: Align acc1 to cexpmax at float_width=23 (fpshift).
    //
    // Hardware fpshift formula: vk = man << (float_width - 23)
    // then shift_round_rne(vk, 23 + shift, float_width) where shift = cexpmax - qexp_biased.
    // For fp32 mantissa, the << 0 is a no-op.
    let aligned_acc: i64 = if qman_unsigned > 0 && qexp_biased > 0 {
        let s_acc = cexpmax - qexp_biased;
        let vk_acc = qman_unsigned; // << 0 (float_width - 23 = 0)
        let (aligned, _) = shift_round_rne(vk_acc, 23 + s_acc, FLOAT_WIDTH, false);
        // Apply accumulator sign (hardware cry_hw + cpr_hw 2's complement).
        if qsgn {
            -aligned
        } else {
            aligned
        }
    } else {
        0
    };

    // Phase 5b: Align acc2 to cexpmax (same fpshift as acc1).
    //
    // Hardware psal_hw (line 12440) merges products + acc1 + acc2 via
    // split_adder5_hw in a single 68-bit addition.
    let aligned_acc2: i64 = if d_man_unsigned > 0 && d_exp_biased > 0 {
        let s_d = cexpmax - d_exp_biased;
        let vk_d = d_man_unsigned;
        let (aligned, _) = shift_round_rne(vk_d, 23 + s_d, FLOAT_WIDTH, false);
        if d_sgn {
            -aligned
        } else {
            aligned
        }
    } else {
        0
    };

    // Phase 6: Sum in 68-bit PSA adder (products + acc1 + acc2).
    let total = product_sum + aligned_acc + aligned_acc2;

    // Phase 7: fpcorr normalization to fp32.
    //
    // Hardware fpcorr formula: exps = cexpmax + fl - 23 - 128
    // where fl = leading bit position of the unsigned magnitude.
    let sgn = total < 0;
    let man = total.unsigned_abs() as i64;

    if man == 0 {
        // Zero sum.  bfnorm_lane: real_zero=true, exp_zeros=false.
        // If NaN flagged: exp=0xFF, mantissa=1 (out_zeros=true for zero
        // but mantissa is 0, then bit 0 |= nan -> 1).
        if nan_flag {
            return fp32_make(false, 255, 1);
        }
        return 0;
    }

    let fl = flo(man);

    // Shift mantissa to 23-bit fp32 field with RNE rounding.
    let (mans, fl_adj) = if fl <= 23 {
        // Value fits in 23 bits -- left-shift to fill.
        (man << (23 - fl), fl)
    } else {
        // Value exceeds 23 bits -- right-shift with RNE.
        let shft = fl - 23 - 1;
        let mut shifted = man >> shft;
        let rmask = (1i64 << shft) - 1;
        let grd = (shifted & 1) != 0;
        let lsb = (shifted & 2) != 0;
        let stk = (man & rmask) != 0;
        let rup = grd && (stk || lsb);
        let mut fl_out = fl;
        if rup {
            shifted += 2;
            if shifted >= (1i64 << 25) {
                shifted >>= 1;
                fl_out += 1;
            }
        }
        (shifted >> 1, fl_out)
    };

    let exps = cexpmax + fl_adj - 23 - 128;

    // bfnorm_lane NaN logic (me_inline_primitives.h:13557-13569):
    //   overflow  = (exps >= 255) && !real_zero
    //   underflow = (exps <= 0)   && !real_zero
    //   out_zeros = !(pinf | minf | overflow | underflow)
    //   exp_ones  = nan | pinf | minf | overflow
    //   rr = out_zeros ? normalized_mantissa : 0
    //   rr[0] |= nan
    //
    // When nan_flag is set: exp is always 0xFF (from exp_ones).
    // Mantissa depends on whether overflow/underflow also applies.
    if nan_flag {
        let overflow = exps >= 255;
        let underflow = exps <= 0;
        if overflow || underflow {
            // out_zeros = false -> mantissa zeroed, bit 0 = nan
            return fp32_make(sgn, 255, 1);
        }
        // out_zeros = true -> mantissa preserved, bit 0 |= nan
        return fp32_make(sgn, 255, (mans as u32 & 0x7F_FFFF) | 1);
    }

    if exps <= 0 || fl == -1 {
        return 0; // Underflow (FTZ)
    }
    if exps >= 255 {
        return fp32_make(sgn, 255, 0); // Overflow -> Inf
    }

    fp32_make(sgn, exps as u8, mans as u32 & 0x7F_FFFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_mac_hw_lane_basic() {
        // Test 1: 8 * (1.0 * 1.0) + 0 = 8.0
        let a = [0x3F80u16; 8]; // bf16 1.0
        let b = [0x3F80u16; 8];
        let r = bf16_mac_hw_lane(0, &a, &b, false, None, false);
        assert_eq!(r, 0x41000000, "8*1*1+0: expected 0x41000000, got 0x{:08X}", r);

        // Test 2: 8 * (-2.0 * 3.0) + 0 = -48.0
        let a2 = [0xC000u16; 8]; // bf16 -2.0
        let b2 = [0x4040u16; 8]; // bf16 3.0
        let r2 = bf16_mac_hw_lane(0, &a2, &b2, false, None, false);
        assert_eq!(r2, 0xC2400000, "8*-2*3+0: expected 0xC2400000, got 0x{:08X}", r2);

        // Test 3: tiny * tiny = underflow to 0
        let a3 = [0x0080u16; 8]; // bf16 min normal
        let b3 = [0x0080u16; 8];
        let r3 = bf16_mac_hw_lane(0, &a3, &b3, false, None, false);
        assert_eq!(r3, 0x00000000, "tiny*tiny: expected 0, got 0x{:08X}", r3);

        // Test 4: zeros * ones = 0
        let a4 = [0x0000u16; 8];
        let b4 = [0x3F80u16; 8];
        let r4 = bf16_mac_hw_lane(0, &a4, &b4, false, None, false);
        assert_eq!(r4, 0x00000000, "zeros: expected 0, got 0x{:08X}", r4);

        // Test 5: subtract mode: 0 - 8*1*1 = -8.0
        let r5 = bf16_mac_hw_lane(0, &a, &b, true, None, false);
        assert_eq!(r5, 0xC1000000, "0-8*1*1: expected 0xC1000000, got 0x{:08X}", r5);

        // Test 6: random inputs (produces 0x7F mantissa pattern in Python model)
        let a6: [u16; 8] = [0xebd7, 0xb13b, 0x4f6c, 0x8d21, 0x6080, 0x39c6, 0x107e, 0xd6a6];
        let b6: [u16; 8] = [0x0802, 0x7b38, 0x6983, 0x2273, 0x33a8, 0x1130, 0xe493, 0x18ec];
        let r6 = bf16_mac_hw_lane(0, &a6, &b6, false, None, false);
        assert_eq!(r6, 0x797187FF, "random: expected 0x797187FF, got 0x{:08X}", r6);
    }

    /// Inf * 0 = NaN even when every product has zero exponent and the
    /// accumulator is zero (cexpmax == 0 must not swallow the NaN flag).
    ///
    /// Silicon-verified: vector fuzzer seed 6160 (mmul_bf16 lane 15) takes
    /// a = -0 in all rows, b containing +Inf -- NPU1 returns NaN where a
    /// zero short-circuit returns 0.
    #[test]
    fn test_inf_times_zero_with_zero_cexpmax_is_nan() {
        // Exact lane inputs from banked seed 6160, slice 6, acc lane 15.
        let a: [u16; 8] = [0x8000, 0x8036, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000];
        let b: [u16; 8] = [0x7F7F, 0x0000, 0x8037, 0x8004, 0x0029, 0x806A, 0xF349, 0x7F80];
        let r = bf16_mac_hw_lane(0, &a, &b, false, None, false);
        assert_eq!(r, 0x7F800001, "Inf*0 with all-zero cexpmax: expected NaN, got 0x{:08X}", r);

        // Minimal form: 0 * Inf alone.
        let a2 = [0x0000u16];
        let b2 = [0x7F80u16];
        let r2 = bf16_mac_hw_lane(0, &a2, &b2, false, None, false);
        assert_eq!(r2, 0x7F800001, "0*Inf: expected NaN, got 0x{:08X}", r2);
    }
}
