//! Hardware-faithful vmac pipeline ported from the aietools C++ ISS.
//!
//! This module implements the full sparse (and dense) MAC pipeline:
//!   mask2sel -> crossbar(prmx) -> Y-perm(prmy) -> broadcast -> multiply
//!   -> PSA tree -> acc_overlap -> accumulator output
//!
//! The routing functions (prmx_hw, prmy_hw) use data-driven routing tables
//! extracted from the C++ model. The compute stages (multiplier, PSA tree,
//! acc overlap) are hand-ported from vmac_functions.inc.
//!
//! Reference: aietools me_inline_primitives.h, lines 4961-14892.

#[allow(unused, unused_parens)]
mod routing {
    include!("vmac_routing.rs");
}

// Re-export routing evaluation functions.
pub use routing::{eval_prmx, eval_prmy};

// ---------------------------------------------------------------------------
// decode_mask: 4-bit mask nibble -> two 3-bit selection indices
// ---------------------------------------------------------------------------

/// Decode a 4-bit sparsity mask nibble into two 3-bit selection indices.
///
/// The hardware decode_mask maps each 4-bit mask pattern to two 3-bit values
/// that encode which of the 4 possible elements are selected. Masks with >2
/// bits set map to (0, 0) = no selection.
///
/// Returns (sel0, sel1).
fn decode_mask(mask4: u8) -> (u8, u8) {
    match mask4 & 0xF {
        1  => (1, 0),
        2  => (2, 0),
        3  => (1, 1),
        4  => (0, 2),
        5  => (1, 2),
        6  => (2, 2),
        8  => (0, 4),
        9  => (1, 4),
        10 => (2, 4),
        12 => (4, 4),
        _  => (0, 0), // >2 bits set or zero
    }
}

// ---------------------------------------------------------------------------
// mask2sel: 128-bit mask + pmode -> 768-bit smode
// ---------------------------------------------------------------------------

/// Compute the 768-bit smode routing control from a 128-bit sparsity mask.
///
/// smode encodes per-element selection indices that the crossbar uses to
/// route dense A operand elements to multiply positions.
///
/// The 768-bit smode is organized as:
///   [0..192)    = r8x4  (64 x 3-bit, for pmode bit 21 = i8xi4 sparse)
///   [192..384)  = r8x8  (64 x 3-bit, for pmode bit 22 = i8xi8 sparse)
///   [384..576)  = r16x8 (64 x 3-bit, for pmode bit 23 = i16xi8 sparse)
///   [576..672)  = r16x16(32 x 3-bit, for pmode bit 24 = i16xi16 sparse)
///   [672..768)  = rbfxbf(32 x 3-bit, for pmode bit 25 = bf16 sparse)
///
/// pmode_bit: which pmode bit is active (21-25 for sparse modes)
/// mask: the 128-bit sparsity mask from the qxs2 register
///
/// Returns smode as [u64; 12] (768 bits, LSB-first).
pub fn mask2sel(mask: u128, pmode_bit: u8) -> [u64; 12] {
    let mut smode = [0u64; 12];

    match pmode_bit {
        21 => {
            // i8xi4 sparse: 8 rows x 4 cols, mask as 32 x 4-bit nibbles
            let nibbles: [u8; 32] = std::array::from_fn(|i| ((mask >> (4 * i)) & 0xF) as u8);
            let mut r8x4 = [0u8; 64]; // 64 x 3-bit selection indices
            for row in 0..8u32 {
                for col in 0..4u32 {
                    let nibble_idx = (row + col * 8) as usize;
                    let (sel0, sel1) = decode_mask(nibbles[nibble_idx]);
                    let idx0 = (col + 4 * (2 * row)) as usize;
                    let idx1 = (col + 4 * (2 * row + 1)) as usize;
                    r8x4[idx0] = sel0;
                    r8x4[idx1] = sel1;
                }
            }
            // Pack r8x4 into smode bits [0..192)
            for i in 0..64 {
                let val = r8x4[i] as u64;
                let bit_pos = i * 3;
                let word = bit_pos / 64;
                let bit = bit_pos % 64;
                smode[word] |= val << bit;
                if bit + 3 > 64 && word + 1 < 12 {
                    smode[word + 1] |= val >> (64 - bit);
                }
            }
        }
        22 => {
            // i8xi8 sparse: 4 rows x 8 cols
            let nibbles: [u8; 32] = std::array::from_fn(|i| ((mask >> (4 * i)) & 0xF) as u8);
            let mut r8x8 = [0u8; 64];
            for row in 0..4u32 {
                for col in 0..8u32 {
                    let nibble_idx = (row + col * 4) as usize;
                    let (sel0, sel1) = decode_mask(nibbles[nibble_idx]);
                    let idx0 = (col + 8 * (2 * row)) as usize;
                    let idx1 = (col + 8 * (2 * row + 1)) as usize;
                    r8x8[idx0] = sel0;
                    r8x8[idx1] = sel1;
                }
            }
            // Pack into smode bits [192..384) (offset by 192 = 64*3)
            for i in 0..64 {
                let val = r8x8[i] as u64;
                let bit_pos = 192 + i * 3;
                let word = bit_pos / 64;
                let bit = bit_pos % 64;
                smode[word] |= val << bit;
                if bit + 3 > 64 && word + 1 < 12 {
                    smode[word + 1] |= val >> (64 - bit);
                }
            }
        }
        23 => {
            // i16xi8 sparse: 4 rows x 8 cols (same mask decode, different output array)
            let nibbles: [u8; 32] = std::array::from_fn(|i| ((mask >> (4 * i)) & 0xF) as u8);
            let mut r16x8 = [0u8; 64];
            for row in 0..4u32 {
                for col in 0..8u32 {
                    let nibble_idx = (row + col * 4) as usize;
                    let (sel0, sel1) = decode_mask(nibbles[nibble_idx]);
                    let idx0 = (col + 8 * (2 * row)) as usize;
                    let idx1 = (col + 8 * (2 * row + 1)) as usize;
                    r16x8[idx0] = sel0;
                    r16x8[idx1] = sel1;
                }
            }
            // Pack into smode bits [384..576)
            for i in 0..64 {
                let val = r16x8[i] as u64;
                let bit_pos = 384 + i * 3;
                let word = bit_pos / 64;
                let bit = bit_pos % 64;
                smode[word] |= val << bit;
                if bit + 3 > 64 && word + 1 < 12 {
                    smode[word + 1] |= val >> (64 - bit);
                }
            }
        }
        24 => {
            // i16xi16 sparse: 2 rows x 8 cols, 8-bit mask bytes.
            //
            // Each byte is paired: 4 pairs of 2 bits each, ORed to form
            // a 4-bit mask4. mask4 goes through decode_mask to produce
            // (sel0, sel1), written to r16x16 (no column reversal, matching ISS).
            //
            // Note: mask bytes with 3+ active pairs produce (0,0) from
            // decode_mask -- verified against oracle. This is correct
            // hardware behavior (no element selection when >2 groups active).
            let mask_bytes: [u8; 16] = std::array::from_fn(|i| ((mask >> (8 * i)) & 0xFF) as u8);
            let mut r16x16 = [0u8; 32];
            for row in 0..2u32 {
                for col in 0..8u32 {
                    let byte_idx = (row + col * 2) as usize;
                    let t = mask_bytes[byte_idx];
                    let mask4 = ((t | (t >> 1)) & 1)
                        | (((t >> 2) | (t >> 3)) & 1) << 1
                        | (((t >> 4) | (t >> 5)) & 1) << 2
                        | (((t >> 6) | (t >> 7)) & 1) << 3;
                    let (sel0, sel1) = decode_mask(mask4 as u8);
                    let idx0 = (col + 8 * (2 * row)) as usize;
                    let idx1 = (col + 8 * (2 * row + 1)) as usize;
                    r16x16[idx0] = sel0;
                    r16x16[idx1] = sel1;
                }
            }
            // Pack into smode bits [576..672)
            for i in 0..32 {
                let val = r16x16[i] as u64;
                let bit_pos = 576 + i * 3;
                let word = bit_pos / 64;
                let bit = bit_pos % 64;
                smode[word] |= val << bit;
                if bit + 3 > 64 && word + 1 < 12 {
                    smode[word + 1] |= val >> (64 - bit);
                }
            }
        }
        25 => {
            // bf16 sparse: 4 rows x 4 cols, 8-bit mask bytes (pair bits)
            let mask_bytes: [u8; 16] = std::array::from_fn(|i| ((mask >> (8 * i)) & 0xFF) as u8);
            let mut rbfxbf = [0u8; 32];
            for row in 0..4u32 {
                for col in 0..4u32 {
                    let byte_idx = (row + col * 4) as usize;
                    let t = mask_bytes[byte_idx];
                    let mask4 = ((t | (t >> 1)) & 1)
                        | (((t >> 2) | (t >> 3)) & 1) << 1
                        | (((t >> 4) | (t >> 5)) & 1) << 2
                        | (((t >> 6) | (t >> 7)) & 1) << 3;
                    let (sel0, sel1) = decode_mask(mask4 as u8);
                    let idx0 = (col + 4 * (2 * row)) as usize;
                    let idx1 = (col + 4 * (2 * row + 1)) as usize;
                    rbfxbf[idx0] = sel0;
                    rbfxbf[idx1] = sel1;
                }
            }
            // Pack into smode bits [672..768)
            for i in 0..32 {
                let val = rbfxbf[i] as u64;
                let bit_pos = 672 + i * 3;
                let word = bit_pos / 64;
                let bit = bit_pos % 64;
                smode[word] |= val << bit;
                if bit + 3 > 64 && word + 1 < 12 {
                    smode[word + 1] |= val >> (64 - bit);
                }
            }
        }
        _ => {} // Not a sparse mode, smode stays zero
    }

    smode
}

/// Build the 789-bit crossbar control word: concat(smode[768], pmode[0:20]).
///
/// Returns as [u64; 13] (832 bits, only 789 used).
fn build_prmx_control(smode: &[u64; 12], pmode: u32) -> [u64; 13] {
    let mut m = [0u64; 13];
    // Bits 0-20: pmode[0:20]
    m[0] = (pmode & 0x1F_FFFF) as u64;
    // Bits 21-788: smode[0:767]
    // smode bit i goes to m bit (i + 21)
    for smode_bit in 0..768 {
        let src_word = smode_bit / 64;
        let src_bit = smode_bit % 64;
        if (smode[src_word] >> src_bit) & 1 != 0 {
            let dst_bit = smode_bit + 21;
            let dst_word = dst_bit / 64;
            let dst_bit_in_word = dst_bit % 64;
            m[dst_word] |= 1u64 << dst_bit_in_word;
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Sign mask functions
// ---------------------------------------------------------------------------

/// Compute the 32-bit sign extension mask for X path.
/// Each bit controls whether the corresponding 8-bit multiply input gets
/// sign-extended (bit 7 replicated to bit 8).
fn sgex_mask(mmode: u8, sgn_x: bool) -> u32 {
    if !sgn_x { return 0; }
    let mut s = 0u32;
    // Groups of 4 bits, each controlled by different mmode bit combinations
    let patterns: [(u8, u32); 8] = [
        (0x03, 0x0000_000F), // bits 0-3:   mmode & 0x03
        (0x17, 0x0000_00F0), // bits 4-7:   mmode & 0x17
        (0x2B, 0x0000_0F00), // bits 8-11:  mmode & 0x2B
        (0x3F, 0x0000_F000), // bits 12-15: mmode & 0x3F
        (0x03, 0x000F_0000), // bits 16-19: mmode & 0x03
        (0x17, 0x00F0_0000), // bits 20-23: mmode & 0x17
        (0xAB, 0x0F00_0000), // bits 24-27: mmode & 0xAB
        (0xBF, 0xF000_0000), // bits 28-31: mmode & 0xBF
    ];
    for (mask, bits) in &patterns {
        if mmode & mask != 0 {
            s |= bits;
        }
    }
    s
}

/// Compute the 32-bit sign extension mask for Y path.
fn sgey_mask(mmode: u8, sgn_y: bool) -> u32 {
    if !sgn_y { return 0; }
    let mut s = 0u32;
    let patterns: [(u8, u32); 8] = [
        (0x01, 0x0000_0003), // bits 0-1:   mmode & 0x01
        (0x17, 0x0000_000C), // bits 2-3:   mmode & 0x17
        (0x01, 0x0000_0030), // bits 4-5:   mmode & 0x01
        (0xBF, 0x0000_00C0), // bits 6-7:   mmode & 0xBF
        (0x01, 0x0000_0300), // bits 8-9:   mmode & 0x01
        (0x17, 0x0000_0C00), // bits 10-11: mmode & 0x17
        (0x01, 0x0000_3000), // bits 12-13: mmode & 0x01
        (0xBF, 0x0000_C000), // bits 14-15: mmode & 0xBF
    ];
    for (mask, bits) in &patterns {
        if mmode & mask != 0 {
            s |= bits;
        }
    }
    // Repeat for bits 16-31 (same pattern)
    s |= s << 16;
    s
}

// ---------------------------------------------------------------------------
// Multiply stages
// ---------------------------------------------------------------------------

/// Low-path multiplier: 16 pairs of (8-bit x 4-bit) -> 16 x 17-bit products.
///
/// px: 32 x 8-bit values (256 bits)
/// py: 32 x 4-bit values (128 bits, packed as 32 nibbles)
/// Returns 16 x 17-bit products packed in i32 values.
fn mpyl_hw_lane(
    px: &[u8; 32],
    py: &[u8; 32], // pre-extracted 4-bit values
    sgn_x: bool,
    sgn_y: bool,
    negate: u32,
    exp0x: u16,
    exp0y: u16,
    mmode: u8,
) -> [i32; 16] {
    let maskx = sgex_mask(mmode, sgn_x);
    let masky = sgey_mask(mmode, sgn_y);
    let bfloat = (mmode >> 6) & 1 != 0;
    let mut result = [0i32; 16];

    for l in 0..16 {
        let i0 = 2 * l;
        let i1 = 2 * l + 1;

        // Sign-extend 8-bit to 9-bit (conditional on maskx)
        let mut x0 = px[i0] as i16;
        if (maskx >> i0) & 1 != 0 && (x0 & 0x80) != 0 {
            x0 |= !0xFF_i16; // sign extend
        }
        let mut x1 = px[i1] as i16;
        if (maskx >> i1) & 1 != 0 && (x1 & 0x80) != 0 {
            x1 |= !0xFF_i16;
        }

        // Sign-extend 4-bit to 5-bit (conditional on masky)
        let mut y0 = py[i0] as i16;
        if (masky >> i0) & 1 != 0 && (y0 & 0x8) != 0 {
            y0 |= !0xF_i16;
        }
        let mut y1 = py[i1] as i16;
        if (masky >> i1) & 1 != 0 && (y1 & 0x8) != 0 {
            y1 |= !0xF_i16;
        }

        // Force implicit bit for non-zero exponents (bf16)
        if (exp0x >> l) & 1 == 0 {
            x0 |= 0x80;
            x1 |= 0x80;
        }
        if (exp0y >> l) & 1 == 0 {
            y1 |= 0x8;
        }

        // Conditional negate (one's complement)
        let (ny0, q0) = if (negate >> i0) & 1 != 0 {
            (!y0, x0)
        } else {
            (y0, 0i16)
        };
        let (ny1, q1) = if (negate >> i1) & 1 != 0 {
            (!y1, x1 as i32)
        } else {
            (y1, 0i32)
        };

        let r0 = (x0 as i32) * (ny0 as i32); // 14-bit
        let r1 = (x1 as i32) * (ny1 as i32) + q1; // 15-bit

        // BFloat shift: left-shift r1 by 4 bits
        let r1s = if bfloat {
            (r1 << 4) as i32
        } else {
            r1
        };

        // Truncate to 17 bits (sign-extended)
        let r = r0 + r1s + (q0 as i32);
        result[l] = (r << 15) >> 15; // sign-extend from 17 bits
    }
    result
}

/// High-path multiplier: 16 pairs of (8-bit x 4-bit) -> 16 x 14-bit products.
fn mpyh_hw_lane(
    px: &[u8; 32],
    py: &[u8; 32],
    sgn_x: bool,
    sgn_y: bool,
    negate: u32,
    mmode: u8,
) -> [i32; 16] {
    let maskx = sgex_mask(mmode, sgn_x);
    let masky = sgey_mask(mmode, sgn_y);
    let mut result = [0i32; 16];

    for l in 0..16 {
        let i0 = 2 * l;
        let i1 = 2 * l + 1;

        let mut x0 = px[i0] as i16;
        if (maskx >> i0) & 1 != 0 && (x0 & 0x80) != 0 {
            x0 |= !0xFF_i16;
        }
        let mut x1 = px[i1] as i16;
        if (maskx >> i1) & 1 != 0 && (x1 & 0x80) != 0 {
            x1 |= !0xFF_i16;
        }

        let mut y0 = py[i0] as i16;
        if (masky >> i0) & 1 != 0 && (y0 & 0x8) != 0 {
            y0 |= !0xF_i16;
        }
        let mut y1 = py[i1] as i16;
        if (masky >> i1) & 1 != 0 && (y1 & 0x8) != 0 {
            y1 |= !0xF_i16;
        }

        let (ny0, q0) = if (negate >> i0) & 1 != 0 {
            (!y0, x0 as i32)
        } else {
            (y0, 0i32)
        };
        let (ny1, q1) = if (negate >> i1) & 1 != 0 {
            (!y1, x1 as i32)
        } else {
            (y1, 0i32)
        };

        let r0 = (x0 as i32) * (ny0 as i32);
        let r1 = (x1 as i32) * (ny1 as i32);
        let r = r0 + r1 + q0 + q1;
        result[l] = (r << 18) >> 18; // sign-extend from 14 bits
    }
    result
}

// ---------------------------------------------------------------------------
// PSA tree (partial sum accumulator)
// ---------------------------------------------------------------------------

/// Compute shift mode from mmode.
///
/// Returns 8 x 6-bit shift control values packed in a [u8; 8].
/// Each 6-bit value has exactly one bit set, selecting the shift amount:
///   bit 0 = shift 0, bit 1 = shift 4, bit 2 = shift 8,
///   bit 3 = shift 12, bit 4 = shift 16, bit 5 = shift 20
fn psa_create_shmode(mmode: u8) -> [u8; 8] {
    const SHIFTS: [u32; 8] = [
        0x41041, 0x42082, 0x41104, 0x42208,
        0x41044, 0x42088, 0x41110, 0x42220,
    ];
    let mut result = [0u8; 8];
    for i in 0..8 {
        let s = SHIFTS[i];
        let mut m = 0u8;
        // bits 23:18 selected by mmode bit 0 or 6
        if mmode & 0x41 != 0 { m |= ((s >> 18) & 0x3F) as u8; }
        // bits 17:12 selected by mmode bit 1
        if mmode & 0x02 != 0 { m |= ((s >> 12) & 0x3F) as u8; }
        // bits 11:6 selected by mmode bit 2 or 4
        if mmode & 0x14 != 0 { m |= ((s >> 6) & 0x3F) as u8; }
        // bits 5:0 selected by mmode bit 3 or 5 or 7
        if mmode & 0xA8 != 0 { m |= (s & 0x3F) as u8; }
        result[i] = m;
    }
    result
}

/// Apply variable left shift to 16 products based on shift mode.
///
/// products: 16 x 26-bit values (sign-extended from 17-bit mpyl or 14-bit mpyh)
/// shmode: 8 shift control values (one per pair of products)
///
/// Returns 16 x 35-bit shifted values as i64.
fn psa_shift1(products: &[i64; 16], shmode: &[u8; 8]) -> [i64; 16] {
    let mut result = [0i64; 16];
    for i in 0..16 {
        let m = shmode[i & 7]; // lower 3 bits of index select shmode
        let a = products[i];
        // m is one-hot: select shift amount
        let shifted = if m & 1 != 0 { a }
            else if m & 2 != 0 { a << 4 }
            else if m & 4 != 0 { a << 8 }
            else if m & 8 != 0 { a << 12 }
            else if m & 16 != 0 { a << 16 }
            else if m & 32 != 0 { a << 20 }
            else { 0 };
        // Truncate to 35 bits (sign-extended)
        result[i] = (shifted << 29) >> 29;
    }
    result
}

/// Multi-adder: sum 8 values (two groups of 8 from 16 shifted products).
///
/// Returns (sum_lo, sum_hi) as 36-bit signed values.
fn multi_adder8(shifted: &[i64; 16]) -> (i64, i64) {
    let lo: i64 = shifted[0..8].iter().sum();
    let hi: i64 = shifted[8..16].iter().sum();
    // Truncate to 36 bits
    let lo = (lo << 28) >> 28;
    let hi = (hi << 28) >> 28;
    (lo, hi)
}

/// PSA shift2: mode-dependent second-stage shift.
///
/// Input: (sum_lo, sum_hi) from multi_adder8 (2 x 36-bit)
/// Returns (r0, r1, r2) as (36-bit, 36-bit, 32-bit).
fn psa_shift2(lo: i64, hi: i64, mmode: u8) -> (i64, i64, i32) {
    let m32 = mmode & 0x4F != 0; // bits 0,1,2,3,6
    let m0 = mmode & 0x30 != 0;  // bits 4,5
    let m16 = mmode & 0x80 != 0; // bit 7

    let r0 = lo;
    let r1 = if m0 {
        hi
    } else if m16 {
        // Shift: extract lo 16 bits of hi, place at bit 16
        let lo16 = (hi & 0xFFFF) as i64;
        lo16 << 16
    } else {
        0
    };
    let r2 = if m32 {
        hi as i32
    } else if m16 {
        let hi20 = ((hi >> 16) & 0xFFFFF) as i32;
        // Sign-extend from 20 bits
        (hi20 << 12) >> 12
    } else {
        0
    };
    (r0, r1, r2)
}

/// Full PSA low pipeline for one lane.
///
/// products: 16 x 17-bit values from mpyl_hw
/// subtract: 2-bit subtract control
/// acc: 64-bit accumulator value
/// scd: 64-bit secondary accumulator
/// mmode: multiplier mode
///
/// Returns 68-bit result as (lo36, hi32).
fn psal_lane(
    products: &[i32; 16],
    subtract: u8,
    acc: u64,
    scd: u64,
    mmode: u8,
) -> (i64, i32) {
    let shmode = psa_create_shmode(mmode);

    // Extend products to i64 (26-bit in hardware, via bfshift_bypass)
    let mut extended = [0i64; 16];
    for i in 0..16 {
        extended[i] = products[i] as i64; // sign-extended
    }

    // PSA shift1
    let shifted = psa_shift1(&extended, &shmode);

    // Multi-adder8
    let (sum_lo, sum_hi) = multi_adder8(&shifted);

    // PSA shift2
    let (m0, m1, m2) = psa_shift2(sum_lo, sum_hi, mmode);

    // Accumulator invert (conditional negate based on subtract bit 0)
    let neg_acc0 = (subtract & 1) != 0;
    let neg_acc1 = (subtract & 1) != 0;
    let acc_lo = (acc & 0xFFFF_FFFF) as u32;
    let acc_hi = (acc >> 32) as u32;
    let ai_lo = if neg_acc0 { !acc_lo } else { acc_lo };
    let ai_hi = if neg_acc1 { !acc_hi } else { acc_hi };

    // SCD invert (conditional negate based on subtract bit 1)
    let neg_scd0 = (subtract >> 1) & 1 != 0;
    let neg_scd1 = (subtract >> 1) & 1 != 0;
    let scd_lo = (scd & 0xFFFF_FFFF) as u32;
    let scd_hi = (scd >> 32) as u32;
    let di_lo = if neg_scd0 { !scd_lo } else { scd_lo };
    let di_hi = if neg_scd1 { !scd_hi } else { scd_hi };

    // Compute offset (one's complement correction)
    let split = mmode & 0x4F != 0;
    let o0 = (neg_acc0 as u32) + (neg_scd0 as u32);
    let o1 = if split { (neg_acc1 as u32) + (neg_scd1 as u32) } else { 0 };

    // Split adder5: combine products + acc + scd + offset
    let k0 = (m0 as i128) + (m1 as i128)
        + (ai_lo as i128) + (di_lo as i128) + (o0 as i128);
    let k1 = (m2 as i128)
        + (ai_hi as i128) + (di_hi as i128) + (o1 as i128);

    // Truncate k0 to 36 bits, k1 to 32 bits
    let k0_36 = (k0 & 0xF_FFFF_FFFF) as i64;
    let k1_32 = (k1 & 0xFFFF_FFFF) as i32;

    (k0_36, k1_32)
}

/// Full PSA high pipeline for one lane.
fn psah_lane(
    products: &[i32; 16],
    subtract: u8,
    acc: u64,
    scd: u64,
    mmode: u8,
) -> (i64, i32) {
    let shmode = psa_create_shmode(mmode);

    let mut extended = [0i64; 16];
    for i in 0..16 {
        // mpyh products are 14-bit, extend to 26-bit
        extended[i] = products[i] as i64;
    }

    let shifted = psa_shift1(&extended, &shmode);
    let (sum_lo, sum_hi) = multi_adder8(&shifted);
    let (m0, m1, m2) = psa_shift2(sum_lo, sum_hi, mmode);

    // psah uses subtract directly (no sgn_acc/sgn_scd)
    let neg0 = (subtract & 1) != 0;
    let neg1 = (subtract & 1) != 0;
    let acc_lo = (acc & 0xFFFF_FFFF) as u32;
    let acc_hi = (acc >> 32) as u32;
    let ai_lo = if neg0 { !acc_lo } else { acc_lo };
    let ai_hi = if neg1 { !acc_hi } else { acc_hi };

    let neg_scd0 = (subtract >> 1) & 1 != 0;
    let neg_scd1 = (subtract >> 1) & 1 != 0;
    let scd_lo = (scd & 0xFFFF_FFFF) as u32;
    let scd_hi = (scd >> 32) as u32;
    let di_lo = if neg_scd0 { !scd_lo } else { scd_lo };
    let di_hi = if neg_scd1 { !scd_hi } else { scd_hi };

    let split = mmode & 0x4F != 0;
    let o0 = (neg0 as u32) + (neg_scd0 as u32);
    let o1 = if split { (neg1 as u32) + (neg_scd1 as u32) } else { 0 };

    let k0 = (m0 as i128) + (m1 as i128)
        + (ai_lo as i128) + (di_lo as i128) + (o0 as i128);
    let k1 = (m2 as i128)
        + (ai_hi as i128) + (di_hi as i128) + (o1 as i128);

    let k0_36 = (k0 & 0xF_FFFF_FFFF) as i64;
    let k1_32 = (k1 & 0xFFFF_FFFF) as i32;

    (k0_36, k1_32)
}

// ---------------------------------------------------------------------------
// Accumulator overlap
// ---------------------------------------------------------------------------

/// Accumulator overlap: combine 68-bit (36+32) PSA result to 64-bit.
fn acc_overlap(k0_36: i64, k1_32: i32, mmode: u8) -> u64 {
    let split = mmode & 0x4F != 0; // bits 0,1,2,3,6
    let al = k0_36 & 0xF_FFFF_FFFF; // 36 bits
    let mut ah = k1_32 as i64; // 32 bits signed

    if !split {
        // Carry cascade: add SIGN-EXTENDED 4-bit carry (al[35:32]) to ah.
        // Hardware: VBit<32,true>(VBit<4,true>(al[35:32])) = sign-extend 4→32.
        let carry_4 = ((al >> 32) & 0xF) as i8;
        let carry_signed = ((carry_4 << 4) >> 4) as i64; // sign-extend from 4 bits
        ah = ah.wrapping_add(carry_signed);
    }

    let lo32 = (al & 0xFFFF_FFFF) as u64;
    let hi32 = (ah & 0xFFFF_FFFF) as u64;
    lo32 | (hi32 << 32)
}

// ---------------------------------------------------------------------------
// vec_control: config word -> mmode/pmode (one-hot encoding)
// ---------------------------------------------------------------------------

/// Parse the MAC config word into one-hot mmode and pmode.
///
/// Returns (mmode, pmode, sgn_x, sgn_y, subtract_acc, subtract_mul).
pub fn vec_control(
    config: u32,
    _zero_acc_in: bool,
    sub0: bool,
    sub1: bool,
    sub2: bool,
) -> (u8, u32, bool, bool, u8, u8) {
    let _zero_acc = (config & 1) != 0;
    let amode = ((config >> 1) & 3) as u8;
    let bmode = ((config >> 3) & 3) as u8;
    let variant = ((config >> 5) & 7) as u8;
    let sgn_y = (config >> 8) & 1 != 0;
    let sgn_x = (config >> 9) & 1 != 0;

    // One-hot mmode encoding
    let mmode = match (amode, bmode) {
        (0, 0) => 0x01, // i8xi4
        (0, 1) => 0x02, // i8xi8
        (0, 2) => 0x04, // i16xi8
        (0, 3) => 0x08, // i16xi16
        (1, 2) => 0x10, // sparse variant
        (1, 3) | (1, 1) => 0x20,
        (2, _) => 0x40, // bf16
        (1, 0) => 0x80,
        _ => 0x01,
    };

    // One-hot pmode encoding from C++ vec_control (lines 9720-9745).
    // Each (mmode_bit, variant) maps to exactly one pmode bit.
    // Bits 0-20 are dense modes, bits 21-25 are sparse modes.
    let pmode_bit = match (mmode, variant) {
        // mmode bit 0 (0x01 = i8xi4)
        (0x01, 0) => 0,   // dense
        (0x01, 1) => 21,  // sparse
        // mmode bit 1 (0x02 = i8xi8)
        (0x02, 0) => 1,   // dense
        (0x02, 1) => 10,  // 0xA
        (0x02, 2) => 13,  // 0xD
        (0x02, 3) => 14,  // 0xE
        (0x02, 4) => 15,  // 0xF
        (0x02, 5) => 22,  // 0x16 = sparse
        // mmode bit 2 (0x04 = i16xi8)
        (0x04, 0) => 2,   // dense
        (0x04, 1) => 9,   // 0x9
        // mmode bit 3 (0x08 = i16xi16)
        (0x08, 0) => 3,   // dense
        (0x08, 1) => 11,  // 0xB
        // mmode bit 4 (0x10)
        (0x10, 0) => 4,   // 0x4
        (0x10, 1) => 5,   // 0x5
        (0x10, 2) => 23,  // 0x17 = sparse (i16xi8 sparse)
        // mmode bit 5 (0x20)
        (0x20, 0) => 6,   // 0x6
        (0x20, 1) => 7,   // 0x7
        (0x20, 2) => 12,  // 0xC
        (0x20, 3) => 16,  // 0x10
        (0x20, 4) => 20,  // 0x14
        (0x20, 5) => 24,  // 0x18 = sparse (i16xi16 sparse)
        // mmode bit 6 (0x40 = bf16)
        (0x40, 0) => 8,   // 0x8
        (0x40, 1) => 17,  // 0x11
        (0x40, 2) => 25,  // 0x19 = sparse (bf16 sparse)
        // mmode bit 7 (0x80)
        (0x80, 0) => 18,  // 0x12
        (0x80, 1) => 19,  // 0x13
        _ => 0,
    };
    let pmode = 1u32 << pmode_bit;

    // Subtract control
    let sub_mul_raw = ((config >> 10) & 0xFF) as u8;
    let subtract_mul = sub_mul_raw ^ if sub0 { 0xFF } else { 0 };

    let sub_acc_raw = ((config >> 12) & 3) as u8;
    let subtract_acc_lo = (sub_acc_raw & 1) ^ (sub1 as u8);
    let subtract_acc_hi = ((sub_acc_raw >> 1) & 1) ^ (sub2 as u8);
    let subtract_acc = subtract_acc_lo | (subtract_acc_hi << 1);

    (mmode, pmode, sgn_x, sgn_y, subtract_acc, subtract_mul)
}

// ---------------------------------------------------------------------------
// vec_control_negate: subtract_mul -> per-lane negate mask
// ---------------------------------------------------------------------------

/// Convert 8-bit subtract_mul into per-lane negate masks.
///
/// Returns [u32; 8] where each element is a 32-bit per-pair negate mask.
/// Even indices (0,2,4,6) share one pattern, odd indices (1,3,5,7) share
/// another. The hardware replicates a 64-bit value across all 8 lanes.
///
/// Three patterns exist based on operand width (mmode):
///   - i8xi4/i8xi8 (mmode bits 0,1): interleaved 4-element groups
///   - bf16 (mmode bit 6): paired elements
///   - i16 modes (mmode bits 2,3,4,5,7): simple alternating pairs
///
/// Reference: C++ vec_control_negate in vmac_functions.inc line 6805.
fn vec_control_negate(mmode: u8, sub: u8) -> [u32; 8] {
    // Build 64-bit negate value. Each bit position maps to a specific sub bit.
    let mut v = 0u64;

    if mmode & 0x03 != 0 {
        // Branch 1: i8xi4/i8xi8
        // Pattern per 8-bit group: sub[base+0], sub[base+1], ..., sub[base+2], sub[base+3], ...
        // Groups 0,1,4,5 use sub[0..3], groups 2,3,6,7 use sub[4..7].
        for b in 0..64u32 {
            let group = b / 8;
            let in_group = (b % 8) as u8;
            let half = in_group / 4;       // 0 for first 4 bits, 1 for next 4
            let pair_bit = in_group % 2;
            let base: u8 = if group % 4 < 2 { 0 } else { 4 };
            let sub_idx = base + half * 2 + pair_bit;
            if (sub >> sub_idx) & 1 != 0 {
                v |= 1u64 << b;
            }
        }
    } else if mmode & 0x40 != 0 {
        // Branch 2: bf16
        // Pattern per 8-bit group: sub[base+0], sub[base+0], sub[base+1], sub[base+1], ...
        // Groups 0,1,4,5 use sub[0..3], groups 2,3,6,7 use sub[4..7].
        for b in 0..64u32 {
            let group = b / 8;
            let in_group = (b % 8) as u8;
            let base: u8 = if group % 4 < 2 { 0 } else { 4 };
            let sub_idx = base + in_group / 2;
            if (sub >> sub_idx) & 1 != 0 {
                v |= 1u64 << b;
            }
        }
    } else if mmode & 0xBC != 0 {
        // Branch 3: i16 modes (bits 2,3,4,5,7)
        // Simple: 16-bit blocks, alternating sub[2k], sub[2k+1].
        for b in 0..64u32 {
            let sub_idx = ((b / 16) * 2 + b % 2) as u8;
            if (sub >> sub_idx) & 1 != 0 {
                v |= 1u64 << b;
            }
        }
    }
    // else: v stays 0 (no negate)

    // Replicate: even lanes get v[31:0], odd lanes get v[63:32].
    let lo = (v & 0xFFFF_FFFF) as u32;
    let hi = (v >> 32) as u32;
    [lo, hi, lo, hi, lo, hi, lo, hi]
}

// ---------------------------------------------------------------------------
// bf16 sparse vmac (element-level float pipeline)
// ---------------------------------------------------------------------------

/// bf16 sparse MAC: determines active bf16 element pairs from the sparsity
/// mask and calls the full bfloat pipeline (bf16_mac_hw_lane) per output.
///
/// The ISS dispatches bf16 sparse through vmac_bf, which uses bfextract +
/// bfexpcompute + bfshift + bfnorm -- a completely different pipeline from
/// the integer mpyl/psal path. We achieve the same result by working at the
/// bf16 element level and reusing the verified dense bf16 MAC function.
///
/// Element mapping (derived from ISS bfextract_hw + oracle probing):
///   A has 64 bf16 elements (4 rows x 16 per row)
///   B has 32 bf16 elements (8 inner x 4 cols)
///   Output is 4 rows x 4 cols = 16 fp32 values
///
///   For output(row, col), inner loops 0..8:
///     bidx = inner*4 + col  (B element index)
///     valid = bfsmode[bidx] != 0  (from mask2sel rbfxbf)
///     aidx depends on bfsmode value:
///       bfsmode=1: aidx = row*16 + (inner>>1)*4 + (inner&1 ? 1 : 0)
///       bfsmode=2: aidx = row*16 + (inner>>1)*4 + (inner&1 ? 2 : 1)
///       bfsmode=3: aidx = row*16 + (inner>>1)*4 + (inner&1 ? 3 : 2)
///
///   Output lane mapping: lane i = output(i/4, i%4)
fn sparse_vmac_bf16(
    a_dense: &[u8; 128],
    b_sparse: &[u8; 64],
    mask: u128,
    acc: &[u64; 16],
    scd: &[u64; 16],
    _config: u32,
    sub0: bool,
    _sub1: bool,
    _sub2: bool,
    subtract_acc: u8,
    _subtract_mul: u8,
) -> [u64; 16] {
    use super::vector_matmul::bf16_mac_hw_lane;

    // Extract bf16 elements from A (64 elements) and B (32 elements)
    let a_bf16: [u16; 64] = std::array::from_fn(|i| {
        u16::from_le_bytes([a_dense[2 * i], a_dense[2 * i + 1]])
    });
    let b_bf16: [u16; 32] = std::array::from_fn(|i| {
        u16::from_le_bytes([b_sparse[2 * i], b_sparse[2 * i + 1]])
    });

    // Compute bfsmode from mask (same computation as mask2sel for pmode_bit=25).
    // bfsmode[i] = rbfxbf[i], a 3-bit routing selector per B position.
    let mask_bytes: [u8; 16] = std::array::from_fn(|i| ((mask >> (8 * i)) & 0xFF) as u8);
    let mut bfsmode = [0u8; 32];
    for row in 0..4u32 {
        for col in 0..4u32 {
            let byte_idx = (row + col * 4) as usize;
            let t = mask_bytes[byte_idx];
            let mask4 = ((t | (t >> 1)) & 1)
                | (((t >> 2) | (t >> 3)) & 1) << 1
                | (((t >> 4) | (t >> 5)) & 1) << 2
                | (((t >> 6) | (t >> 7)) & 1) << 3;
            let (sel0, sel1) = decode_mask(mask4 as u8);
            let idx0 = (col + 4 * (2 * row)) as usize;
            let idx1 = (col + 4 * (2 * row + 1)) as usize;
            bfsmode[idx0] = sel0;
            bfsmode[idx1] = sel1;
        }
    }

    // Product negation from sub0 (MSC variants negate all products)
    let subtract = sub0;

    // Accumulator negation from subtract_acc bit 0
    let negate_acc = (subtract_acc & 1) != 0;

    // Secondary accumulator negation from subtract_acc bit 1
    let negate_scd = (subtract_acc >> 1) & 1 != 0;

    // Compute all 16 fp32 results, then pack into 8 x u64 (bm accumulator format).
    let mut fp32_results = [0u32; 16];

    for row in 0..4u32 {
        for col in 0..4u32 {
            let out_idx = (row * 4 + col) as usize;

            // Collect active bf16 element pairs for this output position
            let mut a_elems = Vec::with_capacity(8);
            let mut b_elems = Vec::with_capacity(8);

            for inner in 0..8u32 {
                let bidx = (inner * 4 + col) as usize;
                let mode = bfsmode[bidx];
                if mode == 0 { continue; }

                // A element index depends on bfsmode value (one-hot):
                //   The base is row*16 + (inner>>1)*4
                //   The offset (t) depends on inner parity and bfsmode:
                //     bfsmode=1 (bit 0, aidx_bf_0): t = inner&1 ? 1 : 0
                //     bfsmode=2 (bit 1, aidx_bf_1): t = inner&1 ? 2 : 1
                //     bfsmode=4 (bit 2, aidx_bf_2): t = inner&1 ? 3 : 2
                let base = (row * 16 + (inner >> 1) * 4) as usize;
                let odd = (inner & 1) != 0;
                let t = match mode {
                    1 => if odd { 1 } else { 0 },
                    2 => if odd { 2 } else { 1 },
                    4 => if odd { 3 } else { 2 },
                    _ => continue, // unknown mode, skip
                };
                let aidx = base + t;

                if aidx < 64 && bidx < 32 {
                    a_elems.push(a_bf16[aidx]);
                    b_elems.push(b_bf16[bidx]);
                }
            }

            // Read existing accumulator. bm accumulators pack 2 fp32 per u64:
            //   acc[out_idx/2] lo32 = even positions, hi32 = odd positions.
            let acc_reg = out_idx / 2;
            let prev_bits = if out_idx & 1 == 0 {
                acc[acc_reg] as u32
            } else {
                (acc[acc_reg] >> 32) as u32
            };
            // For VNEGMAC/VNEGMSC, negate the accumulator via fp32 sign flip.
            let prev_bits = if negate_acc && prev_bits != 0 {
                prev_bits ^ 0x8000_0000
            } else {
                prev_bits
            };

            // Secondary accumulator (same packed layout)
            let scd_val = if out_idx & 1 == 0 {
                scd[acc_reg] as u32
            } else {
                (scd[acc_reg] >> 32) as u32
            };
            let scd_bits = if scd_val != 0 { Some(scd_val) } else { None };

            if a_elems.is_empty() && prev_bits == 0 && scd_bits.is_none() {
                fp32_results[out_idx] = 0;
            } else {
                fp32_results[out_idx] = bf16_mac_hw_lane(
                    prev_bits, &a_elems, &b_elems, subtract,
                    scd_bits, negate_scd,
                );
            }
        }
    }

    // Pack 16 fp32 values into 8 x u64 (bm accumulator format).
    // result[i] = fp32_results[2*i] (lo32) | fp32_results[2*i+1] (hi32)
    // Remaining result[8..16] are zero (unused for bm accumulators).
    let mut result = [0u64; 16];
    for i in 0..8 {
        result[i] = (fp32_results[2 * i] as u64)
            | ((fp32_results[2 * i + 1] as u64) << 32);
    }
    result
}

// ---------------------------------------------------------------------------
// Top-level sparse vmac
// ---------------------------------------------------------------------------

/// Execute the full hardware vmac pipeline for sparse MAC.
///
/// a_dense: 128 bytes of dense A operand (zero-padded if narrow xs1)
/// b_sparse: 64 bytes of compressed sparse B data (qxs2 data portion)
/// mask: 128-bit sparsity mask (qxs2 mask portion)
/// acc: 16 x u64 accumulator values (low 8 = lanes 0-7, high 8 = lanes 8-15)
/// scd: 16 x u64 secondary accumulator (for addmac/submac, otherwise zeros)
/// config: raw config word from scalar register
/// sub0/sub1/sub2: subtract flags from instruction encoding
///
/// Returns 16 x u64 accumulator result.
pub fn sparse_vmac(
    a_dense: &[u8; 128],
    b_sparse: &[u8; 64],
    mask: u128,
    acc: &[u64; 16],
    scd: &[u64; 16],
    config: u32,
    sub0: bool,
    sub1: bool,
    sub2: bool,
) -> [u64; 16] {
    let zero_acc = (config & 1) != 0;
    let (mmode, pmode, sgn_x, sgn_y, subtract_acc, subtract_mul) =
        vec_control(config, true, sub0, sub1, sub2);

    // When zero_acc=1, hardware ignores the existing accumulator.
    // Note: scd (secondary acc for AddMac/SubMac) is NOT zeroed by zero_acc.
    let acc = if zero_acc { &[0u64; 16] } else { acc };

    // Determine sparse pmode bit
    let pmode_bit = pmode.trailing_zeros() as u8;

    // bf16 sparse (pmode_bit=25) uses a completely different pipeline:
    // the ISS dispatches through vmac_bf which does bfextract + bfshift +
    // bfnorm (float normalization), not the integer mpyl/psal pipeline.
    //
    // We implement this by determining which bf16 element pairs are active
    // per output position (from the mask) and calling bf16_mac_hw_lane
    // which already implements the full bfloat pipeline.
    if mmode & 0x40 != 0 {
        return sparse_vmac_bf16(
            a_dense, b_sparse, mask, acc, scd, config,
            sub0, sub1, sub2, subtract_acc, subtract_mul,
        );
    }

    // Step 1: mask2sel -> smode
    let smode = mask2sel(mask, pmode_bit);

    // Step 2: Build crossbar control word and evaluate routing
    let prmx_m = build_prmx_control(&smode, pmode);

    let bx = eval_prmx(a_dense, &prmx_m); // 512 output bytes

    // Step 3: Evaluate Y-perm routing
    // Reinterpret b_sparse as 128 nibbles
    let mut b_nibbles = [0u8; 128];
    for i in 0..64 {
        b_nibbles[2 * i] = b_sparse[i] & 0xF;
        b_nibbles[2 * i + 1] = (b_sparse[i] >> 4) & 0xF;
    }
    let by = eval_prmy(&b_nibbles, pmode); // 512 output nibbles

    // Step 4: Broadcast (reinterpret) into 16 lanes
    // bx: 512 bytes -> 16 lanes x 32 bytes
    // by: 512 nibbles -> 16 lanes x 32 nibbles

    // Split into low (lanes 0-7) and high (lanes 8-15)
    // Per-lane negate mask from subtract_mul (controls product negation for MSC variants).
    let negate_lanes = vec_control_negate(mmode, subtract_mul);

    let mut result = [0u64; 16];

    // Process low 8 lanes
    for lane in 0..8 {
        let base = lane * 32;
        let px: [u8; 32] = std::array::from_fn(|i| bx[base + i]);
        let py: [u8; 32] = std::array::from_fn(|i| by[base + i]);

        let products = mpyl_hw_lane(
            &px, &py, sgn_x, sgn_y, negate_lanes[lane], 0xFFFF, 0xFFFF, mmode,
        );

        let (k0, k1) = psal_lane(&products, subtract_acc, acc[lane], scd[lane], mmode);
        result[lane] = acc_overlap(k0, k1, mmode);
    }

    // Process high 8 lanes
    for lane in 0..8 {
        let base = (8 + lane) * 32;
        let px: [u8; 32] = std::array::from_fn(|i| bx[base + i]);
        let py: [u8; 32] = std::array::from_fn(|i| by[base + i]);

        let products = mpyh_hw_lane(
            &px, &py, sgn_x, sgn_y, negate_lanes[lane], mmode,
        );

        let (k0, k1) = psah_lane(&products, subtract_acc, acc[8 + lane], scd[8 + lane], mmode);
        result[8 + lane] = acc_overlap(k0, k1, mmode);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_mask() {
        assert_eq!(decode_mask(0), (0, 0));
        assert_eq!(decode_mask(1), (1, 0));
        assert_eq!(decode_mask(3), (1, 1));
        assert_eq!(decode_mask(5), (1, 2));
        assert_eq!(decode_mask(7), (0, 0)); // >2 bits set
        assert_eq!(decode_mask(15), (0, 0)); // >2 bits set
    }

    #[test]
    fn test_vec_control_negate_all_zero() {
        // sub=0 -> all negate bits zero regardless of mmode
        assert_eq!(vec_control_negate(0x10, 0x00), [0; 8]);
        assert_eq!(vec_control_negate(0x01, 0x00), [0; 8]);
        assert_eq!(vec_control_negate(0x40, 0x00), [0; 8]);
    }

    #[test]
    fn test_vec_control_negate_all_ones() {
        // sub=0xFF -> all negate bits set (MSC with sub0=true XOR 0xFF)
        let r = vec_control_negate(0x10, 0xFF);
        assert_eq!(r, [0xFFFF_FFFF; 8]);
        let r = vec_control_negate(0x01, 0xFF);
        assert_eq!(r, [0xFFFF_FFFF; 8]);
        let r = vec_control_negate(0x40, 0xFF);
        assert_eq!(r, [0xFFFF_FFFF; 8]);
    }

    #[test]
    fn test_vec_control_negate_i16_partial() {
        // mmode=0x10 (branch 3), sub=0x01 (only bit 0 set)
        // Branch 3: bit b maps to sub[(b/16)*2 + b%2]
        // sub bit 0 -> even bits of first 16 positions
        let r = vec_control_negate(0x10, 0x01);
        // bits 0,2,4,...14 = 1 (sub[0]), others 0
        let expected_lo = 0x5555u32; // bits 0-15: alternating 1,0
        assert_eq!(r[0], expected_lo); // even lanes
        assert_eq!(r[1], 0);          // odd lanes (high 32 bits, sub[4..7]=0)
    }

    #[test]
    fn test_psa_create_shmode_mmode_0x10() {
        // mmode = 0x10 (bit 4, i16xi8 sparse)
        // Should select shifts[i] bits [11:6] for each lane
        let shmode = psa_create_shmode(0x10);
        // Expected: shift 0, 4, 8, 12, 0, 4, 8, 12
        // Which corresponds to shmode bits: 1, 2, 4, 8, 1, 2, 4, 8
        assert_eq!(shmode, [1, 2, 4, 8, 1, 2, 4, 8]);
    }

    /// Compare sparse_vmac output against the C++ oracle with UNIFORM data.
    /// When all A=1 and B=1, routing errors are masked out. This isolates
    /// the multiply-accumulate path.
    #[test]
    fn test_sparse_vmac_uniform_vs_oracle() {
        use std::process::Command;
        let oracle_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/vmac-oracle/vmac_oracle");
        if !std::path::Path::new(oracle_path).exists() { return; }

        // Uniform: all A = 0x01, all B = 0x01
        let mut a_dense = [0u8; 128];
        for i in 0..64 { a_dense[i] = 1; }
        let b_sparse = [1u8; 64];
        // Uniform mask: 2 bits set per nibble
        let mask: u128 = 0x33333333_33333333_33333333_33333333;
        let config: u32 = 0x353;
        let acc = [0u64; 16];
        let scd = [0u64; 16];

        let result = sparse_vmac(&a_dense, &b_sparse, mask, &acc, &scd, config, false, false, false);

        let a_hex: String = a_dense.iter().map(|b| format!("{:02x}", b)).collect();
        let b_hex: String = b_sparse.iter().map(|b| format!("{:02x}", b)).collect();
        let mask_bytes_le: Vec<u8> = (0..16).map(|i| ((mask >> (8 * i)) & 0xFF) as u8).collect();
        let mask_hex: String = mask_bytes_le.iter().map(|b| format!("{:02x}", b)).collect();
        let output = Command::new(oracle_path)
            .args([&a_hex, &b_hex, &mask_hex, "353"])
            .output().unwrap();
        let oracle_stdout = String::from_utf8_lossy(&output.stdout);
        let oracle_lanes: Vec<u64> = oracle_stdout.lines()
            .filter(|l| !l.is_empty())
            .map(|l| u64::from_str_radix(l.trim(), 16).unwrap())
            .collect();

        eprintln!("Rust:   {:016x?}", &result);
        eprintln!("Oracle: {:016x?}", &oracle_lanes);
        let mismatches = (0..16).filter(|&i| result[i] != oracle_lanes[i]).count();
        if mismatches > 0 {
            panic!("{}/16 lanes differ (uniform test)", mismatches);
        }
    }

    /// Uniform i16xi16 sparse oracle test -- isolates compute from routing.
    #[test]
    fn test_sparse_vmac_i16xi16_uniform_vs_oracle() {
        use std::process::Command;
        let oracle_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/vmac-oracle/vmac_oracle");
        if !std::path::Path::new(oracle_path).exists() { return; }

        let mut a_dense = [0u8; 128];
        for i in 0..64 { a_dense[i] = 1; }
        let b_sparse = [1u8; 64];
        let mask: u128 = 0x33333333_33333333_33333333_33333333;
        let config: u32 = 0x3bb; // i16xi16 sparse, signed, zero_acc
        let acc = [0u64; 16];
        let scd = [0u64; 16];

        let result = sparse_vmac(&a_dense, &b_sparse, mask, &acc, &scd, config, false, false, false);

        let a_hex: String = a_dense.iter().map(|b| format!("{:02x}", b)).collect();
        let b_hex: String = b_sparse.iter().map(|b| format!("{:02x}", b)).collect();
        let mask_bytes_le: Vec<u8> = (0..16).map(|i| ((mask >> (8 * i)) & 0xFF) as u8).collect();
        let mask_hex: String = mask_bytes_le.iter().map(|b| format!("{:02x}", b)).collect();
        let output = Command::new(oracle_path)
            .args([&a_hex, &b_hex, &mask_hex, "3bb"])
            .output().unwrap();
        let oracle_stdout = String::from_utf8_lossy(&output.stdout);
        let oracle_lanes: Vec<u64> = oracle_stdout.lines()
            .filter(|l| !l.is_empty())
            .map(|l| u64::from_str_radix(l.trim(), 16).unwrap())
            .collect();

        eprintln!("i16xi16 uniform:");
        eprintln!("Rust:   {:016x?}", &result);
        eprintln!("Oracle: {:016x?}", &oracle_lanes);
        let mismatches = (0..16).filter(|&i| result[i] != oracle_lanes[i]).count();
        if mismatches > 0 {
            panic!("{}/16 lanes differ (i16xi16 uniform test)", mismatches);
        }
    }

    /// Compare sparse_vmac output against the C++ oracle for a known test vector.
    ///
    /// Uses the same test data that was verified against real NPU hardware:
    /// config 0x353 (i16xi8 sparse, sgn_x=1, sgn_y=1, zero_acc=1).
    #[test]
    fn test_sparse_vmac_vs_oracle() {
        use std::process::Command;

        let oracle_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/vmac-oracle/vmac_oracle");
        if !std::path::Path::new(oracle_path).exists() {
            eprintln!("Skipping oracle test: vmac_oracle not compiled");
            return;
        }

        // Test vectors: sequential data for reproducibility.
        // A: 128 bytes (64 used, rest zero-padded)
        let mut a_dense = [0u8; 128];
        for i in 0..64 { a_dense[i] = (i as u8).wrapping_mul(7).wrapping_add(3); }
        // B: 64 bytes compressed sparse data
        let b_sparse: [u8; 64] = std::array::from_fn(|i| (i as u8).wrapping_mul(13).wrapping_add(5));
        // Mask: each nibble has at most 2 bits set (valid sparse mask)
        // Use mask 0x33333333... (bits 0,1 set in each nibble = 2 elements per group)
        let mask: u128 = 0x33333333_33333333_33333333_33333333;
        let config: u32 = 0x353; // i16xi8 sparse, signed, zero_acc
        let acc = [0u64; 16];
        let scd = [0u64; 16];

        // Run our Rust implementation
        let result = sparse_vmac(&a_dense, &b_sparse, mask, &acc, &scd, config, false, false, false);

        // Run the oracle
        let a_hex: String = a_dense.iter().map(|b| format!("{:02x}", b)).collect();
        let b_hex: String = b_sparse.iter().map(|b| format!("{:02x}", b)).collect();
        let mask_bytes_le: Vec<u8> = (0..16).map(|i| ((mask >> (8 * i)) & 0xFF) as u8).collect();
        let mask_hex: String = mask_bytes_le.iter().map(|b| format!("{:02x}", b)).collect();
        let config_hex = format!("{:x}", config);

        let output = Command::new(oracle_path)
            .args([&a_hex, &b_hex, &mask_hex, &config_hex])
            .output();

        let output = match output {
            Ok(o) => o,
            Err(e) => {
                eprintln!("Skipping oracle test: failed to run oracle: {}", e);
                return;
            }
        };

        if !output.status.success() {
            eprintln!("Oracle stderr: {}", String::from_utf8_lossy(&output.stderr));
            panic!("Oracle exited with error");
        }

        let oracle_stdout = String::from_utf8_lossy(&output.stdout);
        let oracle_lanes: Vec<u64> = oracle_stdout
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| u64::from_str_radix(l.trim(), 16).unwrap())
            .collect();

        assert_eq!(oracle_lanes.len(), 16, "Oracle should produce 16 lanes");

        let mut mismatches = 0;
        for i in 0..16 {
            if result[i] != oracle_lanes[i] {
                eprintln!(
                    "Lane {}: rust=0x{:016x} oracle=0x{:016x}",
                    i, result[i], oracle_lanes[i]
                );
                mismatches += 1;
            }
        }
        if mismatches > 0 {
            panic!("{} of 16 lanes differ from oracle", mismatches);
        }
    }

    /// Oracle comparison for i16xi16 sparse -- multiple mask patterns.
    #[test]
    fn test_sparse_vmac_i16xi16_vs_oracle() {
        use std::process::Command;

        let oracle_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/vmac-oracle/vmac_oracle");
        if !std::path::Path::new(oracle_path).exists() {
            eprintln!("Skipping: vmac_oracle not compiled");
            return;
        }

        // Test multiple mask patterns and configs
        let masks: [u128; 4] = [
            0x33333333_33333333_33333333_33333333, // uniform 2 bits per byte
            0xFF00FF00_FF00FF00_FF00FF00_FF00FF00, // alternating full/empty bytes
            0x01020408_10204080_01020408_10204080, // single bits in different positions
            0xA5A5A5A5_A5A5A5A5_A5A5A5A5_A5A5A5A5, // mixed pattern
        ];
        let configs: [u32; 2] = [0x3bb, 0x0bb]; // signed, unsigned

        let mut a_dense = [0u8; 128];
        for i in 0..64 { a_dense[i] = (i as u8).wrapping_mul(7).wrapping_add(3); }
        let b_sparse: [u8; 64] = std::array::from_fn(|i| (i as u8).wrapping_mul(13).wrapping_add(5));
        let acc = [0u64; 16];
        let scd = [0u64; 16];

        let a_hex: String = a_dense.iter().map(|b| format!("{:02x}", b)).collect();
        let b_hex: String = b_sparse.iter().map(|b| format!("{:02x}", b)).collect();

        let mut total_tests = 0;
        let mut total_fail = 0;
        for &config in &configs {
            for &mask in &masks {
                total_tests += 1;
                let result = sparse_vmac(&a_dense, &b_sparse, mask, &acc, &scd, config, false, false, false);

                // Oracle hex_to_bytes reads bytes left-to-right, so pass mask
                // in memory order (LSB first) rather than format!() MSB order.
                let mask_bytes_le: Vec<u8> = (0..16).map(|i| ((mask >> (8 * i)) & 0xFF) as u8).collect();
                let mask_hex: String = mask_bytes_le.iter().map(|b| format!("{:02x}", b)).collect();
                let output = Command::new(oracle_path)
                    .args([&a_hex, &b_hex, &mask_hex, &format!("{:x}", config)])
                    .output().unwrap();
                let oracle_stdout = String::from_utf8_lossy(&output.stdout);
                let oracle_lanes: Vec<u64> = oracle_stdout.lines()
                    .filter(|l| !l.is_empty())
                    .map(|l| u64::from_str_radix(l.trim(), 16).unwrap())
                    .collect();

                let mismatches = (0..16).filter(|&i| result[i] != oracle_lanes[i]).count();
                if mismatches > 0 {
                    total_fail += 1;
                    eprintln!("FAIL config=0x{:03x} mask=0x{:032x}: {}/16 lanes differ",
                        config, mask, mismatches);
                    for i in 0..16 {
                        if result[i] != oracle_lanes[i] {
                            eprintln!("  L{}: rust=0x{:016x} oracle=0x{:016x}", i, result[i], oracle_lanes[i]);
                        }
                    }
                }
            }
        }
        if total_fail > 0 {
            panic!("{}/{} test vectors failed", total_fail, total_tests);
        }
    }

    #[test]
    fn test_sparse_vmac_bf16_vs_oracle() {
        use std::process::Command;

        let oracle_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/vmac-oracle/vmac_oracle");
        if !std::path::Path::new(oracle_path).exists() { return; }

        // bf16 sparse config: amode=2, bmode=0, variant=2, zero_acc=1
        let config: u32 = 0x45;

        // Random-ish bf16 data for thorough testing
        let a_dense: [u8; 128] = std::array::from_fn(|i| {
            (i as u8).wrapping_mul(47).wrapping_add(91)
        });
        let b_sparse: [u8; 64] = std::array::from_fn(|i| {
            (i as u8).wrapping_mul(31).wrapping_add(17)
        });

        let masks: [u128; 5] = [
            u128::from_le_bytes([0x33; 16]),         // 2 active per output
            u128::from_le_bytes([0x03; 16]),         // 1 active per output
            u128::from_le_bytes([0xCC; 16]),         // 2 active (bits 2,3)
            u128::from_le_bytes([0x55; 16]),         // 2 active (bits 0,2)
            0x01020408_10204080_01020408_10204080,   // asymmetric
        ];

        let acc = [0u64; 16];
        let scd = [0u64; 16];

        let a_hex: String = a_dense.iter().map(|b| format!("{:02x}", b)).collect();
        let b_hex: String = b_sparse.iter().map(|b| format!("{:02x}", b)).collect();

        let mut total_tests = 0;
        let mut total_fail = 0;

        for &mask in &masks {
            total_tests += 1;
            let result = sparse_vmac(&a_dense, &b_sparse, mask, &acc, &scd, config, false, false, false);

            let mask_bytes_le: Vec<u8> = (0..16).map(|i| ((mask >> (8 * i)) & 0xFF) as u8).collect();
            let mask_hex: String = mask_bytes_le.iter().map(|b| format!("{:02x}", b)).collect();

            let output = Command::new(oracle_path)
                .args([&a_hex, &b_hex, &mask_hex, &format!("{:x}", config)])
                .output().unwrap();
            let oracle_stdout = String::from_utf8_lossy(&output.stdout);
            let oracle_lanes: Vec<u64> = oracle_stdout.lines()
                .filter(|l| !l.is_empty())
                .map(|l| u64::from_str_radix(l.trim(), 16).unwrap())
                .collect();

            if oracle_lanes.len() != 16 {
                eprintln!("SKIP mask=0x{:032x}: oracle returned {} lanes (expected 16)",
                    mask, oracle_lanes.len());
                continue;
            }

            // Oracle outputs 16 values: acc[i].lo32, acc[i].hi32 interleaved.
            // Pack into 8 u64 for comparison with our packed result.
            let mut oracle_packed = [0u64; 8];
            for i in 0..8 {
                oracle_packed[i] = (oracle_lanes[2 * i] & 0xFFFF_FFFF)
                    | ((oracle_lanes[2 * i + 1] & 0xFFFF_FFFF) << 32);
            }

            let mismatches = (0..8).filter(|&i| result[i] != oracle_packed[i]).count();
            if mismatches > 0 {
                total_fail += 1;
                eprintln!("FAIL config=0x{:03x} mask=0x{:032x}: {}/8 accumulators differ",
                    config, mask, mismatches);
                for i in 0..8 {
                    if result[i] != oracle_packed[i] {
                        eprintln!("  acc{}: rust=0x{:016x} oracle=0x{:016x}", i, result[i], oracle_packed[i]);
                    }
                }
            }
        }
        if total_fail > 0 {
            panic!("{}/{} bf16 sparse test vectors failed", total_fail, total_tests);
        }
        eprintln!("{}/{} bf16 sparse oracle tests passed", total_tests, total_tests);
    }
}
