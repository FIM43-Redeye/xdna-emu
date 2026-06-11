//! Typed vector-op table for the vector fuzzer.
//!
//! Each [`OpEntry`] describes one stage the chain generator can instantiate:
//! input/output vector types, the number of mode variants (shift amounts,
//! shuffle modes), and an emit function producing the C++ expression. All
//! spellings come from the Peano reach spike
//! (`docs/superpowers/specs/2026-06-10-vector-fuzzer-spike-findings.md`),
//! except bf16 sel_lt/sel_ge/bcast, which extrapolate the int-probe spellings
//! to bf16 and get verified by the compile-clean test in Task 8.

use std::sync::OnceLock;

/// Vector types a chain stage can produce/consume. Full-width types are
/// 512-bit; `I16x16` and `I32x8` are 256-bit half-width coupler types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VecType {
    I8x64,
    I16x32,
    I32x16,
    Bf16x32,
    I16x16,
    I32x8,
}

impl VecType {
    /// Element C type name as spelled in aie_api kernels.
    pub fn ctype(self) -> &'static str {
        match self {
            VecType::I8x64 => "int8_t",
            VecType::I16x32 | VecType::I16x16 => "int16_t",
            VecType::I32x16 | VecType::I32x8 => "int32_t",
            VecType::Bf16x32 => "bfloat16",
        }
    }

    /// Number of lanes.
    pub fn lanes(self) -> usize {
        match self {
            VecType::I8x64 => 64,
            VecType::I16x32 => 32,
            VecType::I32x16 => 16,
            VecType::Bf16x32 => 32,
            VecType::I16x16 => 16,
            VecType::I32x8 => 8,
        }
    }

    /// Total vector size in bytes (64 full-width, 32 half-width).
    pub fn bytes(self) -> usize {
        match self {
            VecType::I16x16 | VecType::I32x8 => 32,
            _ => 64,
        }
    }

    /// True for floating-point element types.
    pub fn is_float(self) -> bool {
        matches!(self, VecType::Bf16x32)
    }

    /// Parse from the `{:?}` spelling embedded in coverage keys (`name/Type/m0`).
    /// Lets replay recover a slice's type from banked keys without the live table.
    pub fn from_debug(s: &str) -> Option<VecType> {
        match s {
            "I8x64" => Some(VecType::I8x64),
            "I16x32" => Some(VecType::I16x32),
            "I32x16" => Some(VecType::I32x16),
            "Bf16x32" => Some(VecType::Bf16x32),
            "I16x16" => Some(VecType::I16x16),
            "I32x8" => Some(VecType::I32x8),
            _ => None,
        }
    }
}

/// One vector op the chain generator can instantiate as a stage.
pub struct OpEntry {
    /// Unique per (name, out_type); coverage key = `{name}/{out_type:?}/m{mode}`.
    pub name: &'static str,
    /// Input vector types, length 1 or 2.
    pub in_types: Vec<VecType>,
    /// Result vector type.
    pub out_type: VecType,
    /// Number of mode variants (1 = no mode dimension).
    pub modes: u8,
    /// Emit the C++ expression for this stage given input expressions.
    pub emit: fn(args: &[String], mode: u8, vt: VecType) -> String,
}

/// The full op table (built once).
pub fn table() -> &'static [OpEntry] {
    static TABLE: OnceLock<Vec<OpEntry>> = OnceLock::new();
    TABLE.get_or_init(build_table)
}

/// Two-input op, same type in and out, no mode dimension.
fn bin(name: &'static str, vt: VecType, emit: fn(&[String], u8, VecType) -> String) -> OpEntry {
    OpEntry { name, in_types: vec![vt, vt], out_type: vt, modes: 1, emit }
}

/// One-input op, same type in and out.
fn un(name: &'static str, vt: VecType, modes: u8, emit: fn(&[String], u8, VecType) -> String) -> OpEntry {
    OpEntry { name, in_types: vec![vt], out_type: vt, modes, emit }
}

/// One-input coupler between different vector types.
fn coupler(
    name: &'static str,
    in_t: VecType,
    out_t: VecType,
    emit: fn(&[String], u8, VecType) -> String,
) -> OpEntry {
    OpEntry { name, in_types: vec![in_t], out_type: out_t, modes: 1, emit }
}

fn build_table() -> Vec<OpEntry> {
    let mut t = Vec::new();

    // Spike-verified spellings (except bf16 sel/bcast, noted below):
    // aie::bit_and/bit_or/bit_xor/bit_not
    // (not band/bor/...), aie::max_reduce (not max_red), raw ::shuffle with the
    // mode as a register operand. down/upshift and the pack couplers route via
    // the UPS/SRS accumulator pipeline -- intended, they exercise SRS semantics.

    let int_full = [VecType::I8x64, VecType::I16x32, VecType::I32x16];
    for vt in int_full {
        t.push(bin("add", vt, |a, _, _| format!("aie::add({}, {})", a[0], a[1])));
        t.push(bin("sub", vt, |a, _, _| format!("aie::sub({}, {})", a[0], a[1])));
        t.push(bin("min", vt, |a, _, _| format!("aie::min({}, {})", a[0], a[1])));
        t.push(bin("max", vt, |a, _, _| format!("aie::max({}, {})", a[0], a[1])));
        t.push(un("neg", vt, 1, |a, _, _| format!("aie::neg({})", a[0])));
        t.push(un("abs", vt, 1, |a, _, _| format!("aie::abs({})", a[0])));
        t.push(bin("bit_and", vt, |a, _, _| format!("aie::bit_and({}, {})", a[0], a[1])));
        t.push(bin("bit_or", vt, |a, _, _| format!("aie::bit_or({}, {})", a[0], a[1])));
        t.push(bin("bit_xor", vt, |a, _, _| format!("aie::bit_xor({}, {})", a[0], a[1])));
        t.push(un("bit_not", vt, 1, |a, _, _| format!("aie::bit_not({})", a[0])));
        t.push(bin("sel_lt", vt, |a, _, _| format!("aie::select({0}, {1}, aie::lt({0}, {1}))", a[0], a[1])));
        t.push(bin("sel_ge", vt, |a, _, _| format!("aie::select({0}, {1}, aie::ge({0}, {1}))", a[0], a[1])));
        t.push(bin("sel_eq", vt, |a, _, _| format!("aie::select({0}, {1}, aie::eq({0}, {1}))", a[0], a[1])));
        t.push(un("bcast", vt, 1, |a, _, vt| {
            format!("aie::broadcast<{}, {}>({}.get(0))", vt.ctype(), vt.lanes(), a[0])
        }));
        // Fill variants: plain shuffle_up/down shift in UNDEFINED lanes (silicon
        // returns prior-kernel register residue -- batch-1 seeds 1027/1472), so
        // they cannot be modeled. The fill operand makes every lane defined.
        t.push(OpEntry {
            name: "shup",
            in_types: vec![vt, vt],
            out_type: vt,
            modes: 4,
            emit: |a, m, _| format!("aie::shuffle_up_fill({}, {}, {})", a[0], a[1], m + 1),
        });
        t.push(OpEntry {
            name: "shdn",
            in_types: vec![vt, vt],
            out_type: vt,
            modes: 4,
            emit: |a, m, _| format!("aie::shuffle_down_fill({}, {}, {})", a[0], a[1], m + 1),
        });
        // Keeps reductions in-chain by re-broadcasting the scalar.
        t.push(un("max_reduce_bcast", vt, 1, |a, _, vt| {
            format!("aie::broadcast<{}, {}>(aie::max_reduce({}))", vt.ctype(), vt.lanes(), a[0])
        }));

        // Tier 1: cmp-flag family. maxdiff = max(a-b, 0); max_cmp/min_cmp also
        // return the lane mask, folded into every lane via broadcast-add so the
        // flag result is silicon-observable (spike: vmax_lt/vmin_ge write GPR
        // masks the emulator must model, not just lanes).
        t.push(bin("maxdiff", vt, |a, _, _| format!("aie::maxdiff({}, {})", a[0], a[1])));
        t.push(bin("max_cmp", vt, |a, _, vt| {
            format!(
                "[&]{{ auto [v, m] = aie::max_cmp({}, {}); return aie::add(v, {}); }}()",
                a[0],
                a[1],
                mask_bcast_expr("m", vt)
            )
        }));
        t.push(bin("min_cmp", vt, |a, _, vt| {
            format!(
                "[&]{{ auto [v, m] = aie::min_cmp({}, {}); return aie::add(v, {}); }}()",
                a[0],
                a[1],
                mask_bcast_expr("m", vt)
            )
        }));
        // Mask producers: broadcast the (XOR-folded) compare mask to all lanes.
        t.push(bin("mask_lt", vt, |a, _, vt| {
            format!("[&]{{ auto m = aie::lt({}, {}); return {}; }}()", a[0], a[1], mask_bcast_expr("m", vt))
        }));
        t.push(bin("mask_ge", vt, |a, _, vt| {
            format!("[&]{{ auto m = aie::ge({}, {}); return {}; }}()", a[0], a[1], mask_bcast_expr("m", vt))
        }));
        t.push(bin("mask_eq", vt, |a, _, vt| {
            format!("[&]{{ auto m = aie::eq({}, {}); return {}; }}()", a[0], a[1], mask_bcast_expr("m", vt))
        }));

        // Tier 1: general shifts, mode = shift class {1, width/2, width-1}.
        // i32 stays on the dedicated downshift/upshift entries below.
        if vt != VecType::I32x16 {
            t.push(un("shl", vt, 3, |a, m, vt| format!("aie::upshift({}, {})", a[0], shift_amount(vt, m))));
            t.push(un("sra", vt, 3, |a, m, vt| format!("aie::downshift({}, {})", a[0], shift_amount(vt, m))));
            t.push(un("srl", vt, 3, |a, m, vt| {
                format!("aie::logical_downshift({}, {})", a[0], shift_amount(vt, m))
            }));
        }

        // Tier 1: vector movement. vexin swaps halves via extract+insert
        // (mode = which half is rewritten); elem_ins copies the top lane into
        // lane mode*lanes/4. elem_ins lowers via select+single-lane mask, NOT
        // vector::set -- set on 8-bit lanes references an undefined
        // ::shuffle(v8DB64) symbol that only fails at link (Peano lib gap).
        t.push(un("vexin", vt, 2, |a, m, vt| {
            let half = vt.lanes() / 2;
            format!(
                "[&]{{ auto t = {0}; t.insert({m}, {0}.extract<{half}>({1})); return t; }}()",
                a[0],
                1 - m
            )
        }));
        t.push(un("elem_ins", vt, 4, elem_ins_emit));
    }

    // Tier 1: matrix engine (aie::mmul tiles). A/B come from the full-width
    // inputs via extract; the accumulator couples back through SRS via
    // to_vector (fixed shift 0). mac variants accumulate a second product on
    // top of mul, exercising the acc-accumulate path with defined state.
    // i8 uses the native 4x8x8 tile (Half-A shape): 4x8x4 i8 compiles but
    // references an undefined runtime-mode ::shuffle(v8DB64) symbol that
    // only fails at LINK time (Peano lib gap; same root as elem_ins set).
    // C is 32 i32; the low 16 couple onward.
    t.push(OpEntry {
        name: "mmul_i8",
        in_types: vec![VecType::I8x64, VecType::I8x64],
        out_type: VecType::I32x16,
        modes: 1,
        emit: |a, _, _| {
            format!(
                "[&]{{ aie::mmul<4, 8, 8, int8, int8, accauto> m; m.mul({0}.extract<32>(0), {1}); return m.to_vector<int32>(0).extract<16>(0); }}()",
                a[0], a[1]
            )
        },
    });
    t.push(OpEntry {
        name: "mmac_i8",
        in_types: vec![VecType::I8x64, VecType::I8x64],
        out_type: VecType::I32x16,
        modes: 1,
        emit: |a, _, _| {
            format!(
                "[&]{{ aie::mmul<4, 8, 8, int8, int8, accauto> m; m.mul({0}.extract<32>(0), {1}); m.mac({0}.extract<32>(1), {1}); return m.to_vector<int32>(0).extract<16>(0); }}()",
                a[0], a[1]
            )
        },
    });
    t.push(OpEntry {
        name: "mmul_i16",
        in_types: vec![VecType::I16x32, VecType::I16x32],
        out_type: VecType::I32x16,
        modes: 1,
        emit: |a, _, _| {
            format!(
                "[&]{{ aie::mmul<4, 4, 4, int16, int16, accauto> m; m.mul({0}.extract<16>(0), {1}.extract<16>(0)); return m.to_vector<int32>(0); }}()",
                a[0], a[1]
            )
        },
    });
    t.push(OpEntry {
        name: "mmac_i16",
        in_types: vec![VecType::I16x32, VecType::I16x32],
        out_type: VecType::I32x16,
        modes: 1,
        emit: |a, _, _| {
            format!(
                "[&]{{ aie::mmul<4, 4, 4, int16, int16, accauto> m; m.mul({0}.extract<16>(0), {1}.extract<16>(0)); m.mac({0}.extract<16>(1), {1}.extract<16>(1)); return m.to_vector<int32>(0); }}()",
                a[0], a[1]
            )
        },
    });

    // int32-only: shift amount = mode (UPS/SRS pipeline), raw vshuffle mode sweep.
    t.push(un("downshift", VecType::I32x16, 8, |a, m, _| format!("aie::downshift({}, {})", a[0], m)));
    t.push(un("upshift", VecType::I32x16, 8, |a, m, _| format!("aie::upshift({}, {})", a[0], m)));
    t.push(OpEntry {
        name: "shuffle",
        in_types: vec![VecType::I32x16, VecType::I32x16],
        out_type: VecType::I32x16,
        modes: 48,
        // Raw ::shuffle returns a native v16int32; wrap it back into an
        // aie::vector so downstream stages (store_v, aie::* calls) type-check.
        emit: |a, m, _| format!("aie::vector<int32_t, 16>(::shuffle({}, {}, {}))", a[0], a[1], m),
    });

    // Couplers between widths/types (pack/unpack go through accum SRS/UPS).
    t.push(coupler("pack16", VecType::I32x16, VecType::I16x16, |a, _, _| format!("aie::pack({})", a[0])));
    t.push(coupler("pack8", VecType::I16x32, VecType::I8x64, |a, _, _| {
        format!("aie::concat(aie::pack({0}), aie::pack({0}))", a[0])
    }));
    t.push(coupler("unpack16", VecType::I8x64, VecType::I16x32, |a, _, _| {
        format!("aie::unpack({}.extract<32>(0))", a[0])
    }));
    t.push(coupler("unpack32", VecType::I16x16, VecType::I32x16, |a, _, _| format!("aie::unpack({})", a[0])));
    t.push(coupler("narrow16", VecType::I16x32, VecType::I16x16, |a, _, _| {
        format!("{}.extract<16>(0)", a[0])
    }));
    t.push(coupler("narrow32", VecType::I32x16, VecType::I32x8, |a, _, _| format!("{}.extract<8>(0)", a[0])));
    t.push(coupler("grow16", VecType::I16x16, VecType::I16x32, |a, _, _| {
        format!("aie::concat({0}, {0})", a[0])
    }));
    t.push(coupler("grow32", VecType::I32x8, VecType::I32x16, |a, _, _| {
        format!("aie::concat({0}, {0})", a[0])
    }));

    // bf16 family -- never couples to int types. add/sub/min/max/neg are
    // spike-probed; sel_lt/sel_ge/bcast extrapolate the int-probe spellings
    // (Task 8 compile-clean test verifies them).
    let bf = VecType::Bf16x32;
    t.push(bin("add", bf, |a, _, _| format!("aie::add({}, {})", a[0], a[1])));
    t.push(bin("sub", bf, |a, _, _| format!("aie::sub({}, {})", a[0], a[1])));
    t.push(bin("min", bf, |a, _, _| format!("aie::min({}, {})", a[0], a[1])));
    t.push(bin("max", bf, |a, _, _| format!("aie::max({}, {})", a[0], a[1])));
    t.push(un("neg", bf, 1, |a, _, _| format!("aie::neg({})", a[0])));
    t.push(bin("sel_lt", bf, |a, _, _| format!("aie::select({0}, {1}, aie::lt({0}, {1}))", a[0], a[1])));
    t.push(bin("sel_ge", bf, |a, _, _| format!("aie::select({0}, {1}, aie::ge({0}, {1}))", a[0], a[1])));
    t.push(un("bcast", bf, 1, |a, _, vt| {
        format!("aie::broadcast<{}, {}>({}.get(0))", vt.ctype(), vt.lanes(), a[0])
    }));

    // Tier 1 bf16 family. mmul tile is 4x8x4 bf16 -> fp32, coupled back to
    // bf16 via the accumulator conversion. The converted vector must be
    // SINGLE-USE: duplicating it (concat(c, c)) trips Peano GlobalISel
    // (selectG_AIE_STORE_CONV "Expected SSA" assertion -- tier-1 probe), so
    // the upper output half takes the b operand's low half. Defined, modeled.
    t.push(OpEntry {
        name: "mmul_bf16",
        in_types: vec![bf, bf],
        out_type: bf,
        modes: 1,
        emit: |a, _, _| {
            format!(
                "[&]{{ aie::mmul<4, 8, 4, bfloat16, bfloat16, accauto> m; m.mul({0}, {1}); return aie::concat(m.to_vector<bfloat16>(), {1}.extract<16>(0)); }}()",
                a[0], a[1]
            )
        },
    });
    t.push(OpEntry {
        name: "mmac_bf16",
        in_types: vec![bf, bf],
        out_type: bf,
        modes: 1,
        emit: |a, _, _| {
            format!(
                "[&]{{ aie::mmul<4, 8, 4, bfloat16, bfloat16, accauto> m; m.mul({0}, {1}); m.mac({0}, {1}); return aie::concat(m.to_vector<bfloat16>(), {1}.extract<16>(0)); }}()",
                a[0], a[1]
            )
        },
    });
    t.push(bin("maxdiff", bf, |a, _, _| format!("aie::maxdiff({}, {})", a[0], a[1])));
    t.push(bin("mask_lt", bf, |a, _, vt| {
        format!("[&]{{ auto m = aie::lt({}, {}); return {}; }}()", a[0], a[1], mask_bcast_expr("m", vt))
    }));
    t.push(bin("mask_ge", bf, |a, _, vt| {
        format!("[&]{{ auto m = aie::ge({}, {}); return {}; }}()", a[0], a[1], mask_bcast_expr("m", vt))
    }));
    t.push(un("vexin", bf, 2, |a, m, vt| {
        let half = vt.lanes() / 2;
        format!("[&]{{ auto t = {0}; t.insert({m}, {0}.extract<{half}>({1})); return t; }}()", a[0], 1 - m)
    }));
    t.push(un("elem_ins", bf, 4, elem_ins_emit));

    t
}

/// elem_ins emit: copy the top lane into lane `mode * lanes/4` via
/// select + single-lane mask (vsel + vbcst), avoiding `vector::set`, which
/// references an undefined `::shuffle(v8DB64)` symbol at link for 8-bit lanes.
fn elem_ins_emit(a: &[String], m: u8, vt: VecType) -> String {
    let lanes = vt.lanes();
    let lane = m as usize * (lanes / 4);
    let mask = if lanes == 64 {
        format!("aie::mask<64>::from_uint64(1ull << {lane})")
    } else {
        format!("aie::mask<{lanes}>::from_uint32(1u << {lane})")
    };
    format!(
        "aie::select({0}, aie::broadcast<{ct}, {lanes}>({0}.get({top})), {mask})",
        a[0],
        ct = vt.ctype(),
        top = lanes - 1
    )
}

/// Broadcast a compare mask (named by `mask`) to all lanes of `vt`. Lane
/// counts wider than the element type XOR-fold so every mask bit affects the
/// result; bf16 routes through float conversion (deterministic on silicon).
fn mask_bcast_expr(mask: &str, vt: VecType) -> String {
    match vt {
        VecType::I32x16 | VecType::I32x8 => {
            format!("aie::broadcast<int32_t, {}>((int32_t){mask}.to_uint32())", vt.lanes())
        }
        VecType::I16x32 | VecType::I16x16 => format!(
            "aie::broadcast<int16_t, {}>((int16_t)({mask}.to_uint32() ^ ({mask}.to_uint32() >> 16)))",
            vt.lanes()
        ),
        VecType::I8x64 => format!(
            "aie::broadcast<int8_t, 64>((int8_t)(({mask}.to_uint64() ^ ({mask}.to_uint64() >> 32) ^ ({mask}.to_uint64() >> 16) ^ ({mask}.to_uint64() >> 8)) & 0xff))"
        ),
        VecType::Bf16x32 => format!("aie::broadcast<bfloat16, 32>(bfloat16((float){mask}.to_uint32()))"),
    }
}

/// Shift amount for the general-shift entries: mode 0 -> 1 bit, mode 1 ->
/// half the element width, mode 2 -> width-1 (sign-replication edge).
fn shift_amount(vt: VecType, mode: u8) -> usize {
    let width = vt.bytes() / vt.lanes() * 8;
    match mode {
        0 => 1,
        1 => width / 2,
        _ => width - 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn dummy_args(n: usize) -> Vec<String> {
        ["a", "b"].iter().take(n).map(|s| s.to_string()).collect()
    }

    #[test]
    fn table_non_empty() {
        assert!(table().len() > 90, "expected >90 entries, got {}", table().len());
    }

    #[test]
    fn tier1_families_present_per_type() {
        let has = |name: &str, vt: VecType| table().iter().any(|e| e.name == name && e.out_type == vt);
        for vt in [VecType::I8x64, VecType::I16x32, VecType::I32x16, VecType::Bf16x32] {
            assert!(has("maxdiff", vt), "maxdiff missing for {vt:?}");
            assert!(has("mask_lt", vt), "mask_lt missing for {vt:?}");
            assert!(has("vexin", vt), "vexin missing for {vt:?}");
            assert!(has("elem_ins", vt), "elem_ins missing for {vt:?}");
        }
        // General shifts only on i8/i16 (i32 keeps the dedicated entries).
        for vt in [VecType::I8x64, VecType::I16x32] {
            for name in ["shl", "sra", "srl"] {
                let e = table().iter().find(|e| e.name == name && e.out_type == vt).expect(name);
                assert_eq!(e.modes, 3, "{name}/{vt:?} mode count");
            }
        }
        assert!(!has("shl", VecType::I32x16), "i32 shl should not exist");
        // Matrix engine: int tiles couple to I32x16, bf16 stays bf16.
        for name in ["mmul_i8", "mmac_i8", "mmul_i16", "mmac_i16"] {
            let e = table().iter().find(|e| e.name == name).expect(name);
            assert_eq!(e.out_type, VecType::I32x16, "{name} out type");
            assert_eq!(e.in_types.len(), 2, "{name} arity");
        }
        for name in ["mmul_bf16", "mmac_bf16"] {
            let e = table().iter().find(|e| e.name == name).expect(name);
            assert_eq!(e.out_type, VecType::Bf16x32, "{name} out type");
        }
    }

    #[test]
    fn shift_amounts_are_1_mid_widthm1() {
        assert_eq!(shift_amount(VecType::I8x64, 0), 1);
        assert_eq!(shift_amount(VecType::I8x64, 1), 4);
        assert_eq!(shift_amount(VecType::I8x64, 2), 7);
        assert_eq!(shift_amount(VecType::I16x32, 0), 1);
        assert_eq!(shift_amount(VecType::I16x32, 1), 8);
        assert_eq!(shift_amount(VecType::I16x32, 2), 15);
    }

    #[test]
    fn mask_bcast_expr_matches_lane_type() {
        assert!(mask_bcast_expr("m", VecType::I32x16).contains("int32_t, 16"));
        assert!(mask_bcast_expr("m", VecType::I16x32).contains("int16_t, 32"));
        assert!(mask_bcast_expr("m", VecType::I8x64).contains("to_uint64"));
        assert!(mask_bcast_expr("m", VecType::Bf16x32).contains("bfloat16"));
    }

    #[test]
    fn coverage_keys_globally_unique() {
        let mut seen = HashSet::new();
        for e in table() {
            for mode in 0..e.modes {
                let key = format!("{}/{:?}/m{}", e.name, e.out_type, mode);
                assert!(seen.insert(key.clone()), "duplicate coverage key {key}");
            }
        }
    }

    #[test]
    fn in_types_len_1_or_2() {
        for e in table() {
            assert!((1..=2).contains(&e.in_types.len()), "{}: in_types len {}", e.name, e.in_types.len());
        }
    }

    #[test]
    fn bf16_never_couples_to_int() {
        for e in table() {
            let touches_bf16 = e.out_type == VecType::Bf16x32 || e.in_types.contains(&VecType::Bf16x32);
            if touches_bf16 {
                assert_eq!(e.out_type, VecType::Bf16x32, "{}: bf16 entry int out", e.name);
                for t in &e.in_types {
                    assert_eq!(*t, VecType::Bf16x32, "{}: bf16 entry int input", e.name);
                }
            }
        }
    }

    #[test]
    fn emit_produces_expression_for_edge_modes() {
        for e in table() {
            let args = dummy_args(e.in_types.len());
            for mode in [0, e.modes - 1] {
                let expr = (e.emit)(&args, mode, e.out_type);
                assert!(!expr.is_empty(), "{} m{mode}: empty emit", e.name);
                assert!(expr.contains('('), "{} m{mode}: no call in {expr:?}", e.name);
            }
        }
    }

    #[test]
    fn half_width_types_are_32_bytes() {
        for (vt, expected) in [
            (VecType::I8x64, 64),
            (VecType::I16x32, 64),
            (VecType::I32x16, 64),
            (VecType::Bf16x32, 64),
            (VecType::I16x16, 32),
            (VecType::I32x8, 32),
        ] {
            assert_eq!(vt.bytes(), expected, "{vt:?}");
        }
    }

    #[test]
    fn int_entry_lanes_match_bytes() {
        for e in table() {
            for vt in e.in_types.iter().copied().chain([e.out_type]) {
                if vt.is_float() {
                    continue;
                }
                let elem = match vt.ctype() {
                    "int8_t" => 1,
                    "int16_t" => 2,
                    "int32_t" => 4,
                    other => panic!("unexpected ctype {other}"),
                };
                assert_eq!(vt.lanes() * elem, vt.bytes(), "{}: {vt:?} lanes*elem != bytes", e.name);
            }
        }
    }
}
