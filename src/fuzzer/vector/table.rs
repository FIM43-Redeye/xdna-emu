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
        t.push(un("shup", vt, 4, |a, m, _| format!("aie::shuffle_up({}, {})", a[0], m + 1)));
        t.push(un("shdn", vt, 4, |a, m, _| format!("aie::shuffle_down({}, {})", a[0], m + 1)));
        // Keeps reductions in-chain by re-broadcasting the scalar.
        t.push(un("max_reduce_bcast", vt, 1, |a, _, vt| {
            format!("aie::broadcast<{}, {}>(aie::max_reduce({}))", vt.ctype(), vt.lanes(), a[0])
        }));
    }

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

    t
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
        assert!(table().len() > 60, "expected >60 entries, got {}", table().len());
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
