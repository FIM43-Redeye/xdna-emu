//! Extraction of AIE2 matrix-multiply geometries from llvm-aie's
//! `aiev2_vmult.h` (Apache-2.0 WITH LLVM-exception).
//!
//! Every matmul-shaped intrinsic in that header is named
//! `<op>_<rows>x<inner>_<inner>x<cols>` and builds its hardware config word
//! with `aiev2_compute_control(sgn_x, sgn_y, amode, bmode, variant, ...)`.
//! The name gives the tile shape, the call gives the (amode, bmode, variant)
//! config-word fields, and the parameter and return vector types give element
//! widths, sparsity and accumulator width. One geometry per config key falls
//! out; the many `mul`/`mac`/`msc`/... flavours of each shape must agree.
//!
//! This file is compiled both into the library (so its tests run) and, via
//! `#[path]`, into `build.rs`, which renders the result as `gen_mac_geometry.rs`.
//! Anything the parser cannot interpret is an error: the build fails rather
//! than guessing.

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// One matmul geometry, keyed by the config-word fields that select it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractedGeometry {
    pub amode: u32,
    pub bmode: u32,
    pub variant: u32,
    pub bits_x: u32,
    pub bits_y: u32,
    pub rows: u32,
    pub inner: u32,
    pub cols: u32,
    /// 1 for 32-bit accumulator lanes (`acc32`, `accfloat`), 2 for 64-bit.
    pub acc_cmb: u32,
    pub bfloat: bool,
    pub sparse: bool,
}

/// Parse every matmul-shaped intrinsic definition in `source`.
///
/// Returns one entry per distinct (amode, bmode, variant), sorted by that key.
pub fn parse_vmult_header(source: &str) -> Result<Vec<ExtractedGeometry>, String> {
    let text = strip_inactive_regions(source)?;
    let mut by_key: BTreeMap<(u32, u32, u32), (String, ExtractedGeometry)> = BTreeMap::new();

    let mut rest = text.as_str();
    while let Some(at) = rest.find("INTRINSIC(") {
        rest = &rest[at + "INTRINSIC(".len()..];
        let (ret_type, after_ret) = take_balanced(rest, ')')
            .ok_or_else(|| format!("unterminated INTRINSIC( near `{}`", snippet(rest)))?;
        let after_ret = after_ret.trim_start();
        let name_len = after_ret.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(0);
        let name = &after_ret[..name_len];
        let after_name = after_ret[name_len..].trim_start();
        let Some(shape) = parse_shape(name) else {
            // Not matmul-shaped (clear, add, elementwise, ...): out of scope.
            rest = after_name;
            continue;
        };
        let params_src = after_name
            .strip_prefix('(')
            .and_then(|s| take_balanced(s, ')'))
            .ok_or_else(|| format!("{name}: no parameter list"))?;
        let (params, after_params) = params_src;
        let body_src = after_params
            .trim_start()
            .strip_prefix('{')
            .ok_or_else(|| format!("{name}: matmul-shaped intrinsic without a body"))?;
        let (body, after_body) =
            take_balanced(body_src, '}').ok_or_else(|| format!("{name}: unterminated body"))?;
        rest = after_body;

        if is_delegating_wrapper(shape, body) {
            continue;
        }
        let entry = geometry_of(name, shape, ret_type, params, body)?;
        let key = (entry.amode, entry.bmode, entry.variant);
        match by_key.get(&key) {
            Some((first, existing)) if *existing != entry => {
                return Err(format!(
                    "conflicting geometries for (amode={}, bmode={}, variant={}): {first} gives {existing:?}, \
                     {name} gives {entry:?}",
                    key.0, key.1, key.2
                ));
            }
            Some(_) => {}
            None => {
                by_key.insert(key, (name.to_string(), entry));
            }
        }
    }

    if by_key.is_empty() {
        return Err("no matmul-shaped intrinsics found (is this aiev2_vmult.h?)".to_string());
    }
    Ok(by_key.into_values().map(|(_, entry)| entry).collect())
}

/// Render the table as Rust source defining `CONFIG_GEOMETRY_TABLE`, to be
/// `include!`d next to `GeometryEntry` in `aie2/matmul.rs`.
pub fn render_config_geometry_table(entries: &[ExtractedGeometry]) -> String {
    let mut out = String::new();
    writeln!(out, "/// (amode, bmode, variant) -> geometry, extracted from aiev2_vmult.h.").unwrap();
    writeln!(out, "pub const CONFIG_GEOMETRY_TABLE: &[(u32, u32, u32, GeometryEntry)] = &[").unwrap();
    for e in entries {
        writeln!(
            out,
            "    ({}, {}, {}, GeometryEntry {{ bits_x: {}, bits_y: {}, rows: {}, inner: {}, cols: {}, \
             acc_cmb: {}, bfloat: {}, sparse: {} }}),",
            e.amode,
            e.bmode,
            e.variant,
            e.bits_x,
            e.bits_y,
            e.rows,
            e.inner,
            e.cols,
            e.acc_cmb,
            e.bfloat,
            e.sparse
        )
        .unwrap();
    }
    writeln!(out, "];").unwrap();
    out
}

/// Convenience overloads that take wider operands, split them with
/// `shuffle`, and call the same-shaped intrinsic twice. They carry no config
/// word of their own, so the intrinsics they delegate to define the geometry.
/// A body that neither builds a config word nor calls a same-shaped intrinsic
/// is not recognised and falls through to an error.
fn is_delegating_wrapper((rows, inner, cols): (u32, u32, u32), body: &str) -> bool {
    let shape = format!("{rows}x{inner}_{inner}x{cols}");
    !body.contains("aiev2_compute_control(") && body.contains(&shape)
}

/// `(rows, inner, cols)` from `<op>_<r>x<k>_<k>x<c>`, or `None` if the name
/// is not matmul-shaped (including names whose two inner dimensions differ).
fn parse_shape(name: &str) -> Option<(u32, u32, u32)> {
    let mut parts = name.rsplitn(3, '_');
    let right = parts.next()?;
    let left = parts.next()?;
    let op = parts.next()?;
    if op.is_empty() || !op.chars().all(|c| c.is_ascii_lowercase()) {
        return None;
    }
    let (rows, inner_l) = left.split_once('x')?;
    let (inner_r, cols) = right.split_once('x')?;
    let num =
        |s: &str| (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse::<u32>().ok())?;
    let (rows, inner_l, inner_r, cols) = (num(rows)?, num(inner_l)?, num(inner_r)?, num(cols)?);
    (inner_l == inner_r).then_some((rows, inner_l, cols))
}

fn geometry_of(
    name: &str,
    (rows, inner, cols): (u32, u32, u32),
    ret_type: &str,
    params: &str,
    body: &str,
) -> Result<ExtractedGeometry, String> {
    let call_at = body
        .find("aiev2_compute_control(")
        .ok_or_else(|| format!("{name}: no aiev2_compute_control call"))?;
    let (call_args, _) = take_balanced(&body[call_at + "aiev2_compute_control(".len()..], ')')
        .ok_or_else(|| format!("{name}: unterminated aiev2_compute_control call"))?;
    let args: Vec<&str> = split_top_level(call_args);
    // (sgn_x, sgn_y, amode, bmode, variant, zero_acc, shift16, sub0, sub1, sub2, sub_mask)
    if args.len() != 11 {
        return Err(format!("{name}: aiev2_compute_control has {} arguments, expected 11", args.len()));
    }
    let field = |index: usize, what: &str| -> Result<u32, String> {
        args[index].trim().parse::<u32>().map_err(|_| {
            format!("{name}: {what} argument `{}` is not an integer literal", args[index].trim())
        })
    };
    let (amode, bmode, variant) = (field(2, "amode")?, field(3, "bmode")?, field(4, "variant")?);

    // The first two non-accumulator vector parameters are A (x) and B (y);
    // scalar sign parameters and accumulator inputs are skipped.
    let mut operands = split_top_level(params).into_iter().filter_map(|param| {
        let ty = param.trim().rsplit_once(char::is_whitespace)?.0.trim();
        (ty.starts_with('v') && !ty.contains("acc")).then_some(ty)
    });
    let x = operands.next().ok_or_else(|| format!("{name}: no A operand type"))?;
    let y = operands.next().ok_or_else(|| format!("{name}: no B operand type"))?;
    let x = parse_vector_type(x).map_err(|e| format!("{name}: A operand: {e}"))?;
    let y = parse_vector_type(y).map_err(|e| format!("{name}: B operand: {e}"))?;
    if x.sparse {
        return Err(format!("{name}: sparse A operand is not a known configuration"));
    }
    if x.bfloat != y.bfloat {
        return Err(format!("{name}: mixed bfloat16 and integer operands"));
    }

    let acc_cmb = match ret_type.trim().rsplit_once("acc").map(|(_, w)| w) {
        Some("32") | Some("float") => 1,
        Some("64") => 2,
        _ => return Err(format!("{name}: unrecognised accumulator return type `{}`", ret_type.trim())),
    };

    Ok(ExtractedGeometry {
        amode,
        bmode,
        variant,
        bits_x: x.bits,
        bits_y: y.bits,
        rows,
        inner,
        cols,
        acc_cmb,
        bfloat: x.bfloat,
        sparse: y.sparse,
    })
}

struct VectorType {
    bits: u32,
    bfloat: bool,
    sparse: bool,
}

/// `v64uint8`, `v128int4`, `v64bfloat16_sparse`, ...
fn parse_vector_type(ty: &str) -> Result<VectorType, String> {
    let (base, sparse) = match ty.strip_suffix("_sparse") {
        Some(base) => (base, true),
        None => (ty, false),
    };
    let after_v = base.strip_prefix('v').ok_or_else(|| format!("`{ty}` is not a vector type"))?;
    let lanes_len = after_v.bytes().take_while(u8::is_ascii_digit).count();
    let elem = &after_v[lanes_len..];
    if lanes_len == 0 {
        return Err(format!("`{ty}` has no lane count"));
    }
    let (bfloat, width) = if let Some(w) = elem.strip_prefix("bfloat") {
        (true, w)
    } else if let Some(w) = elem.strip_prefix("uint").or_else(|| elem.strip_prefix("int")) {
        (false, w)
    } else {
        return Err(format!("`{ty}` has an unrecognised element type"));
    };
    let bits = width
        .parse::<u32>()
        .map_err(|_| format!("`{ty}` has an unparseable element width"))?;
    if bfloat && bits != 16 {
        return Err(format!("`{ty}`: bfloat element width {bits} is not 16"));
    }
    Ok(VectorType { bits, bfloat, sparse })
}

/// Drop `#if 0` regions and comments, and reject any conditional the parser
/// cannot evaluate. The header uses only `#if 0`, `#if 1` and its include
/// guard; anything else would silently change which intrinsics are real.
fn strip_inactive_regions(source: &str) -> Result<String, String> {
    let mut out = String::with_capacity(source.len());
    // One entry per open conditional: is that branch active?
    let mut stack: Vec<bool> = Vec::new();
    for (index, line) in source.lines().enumerate() {
        let trimmed = line.trim_start();
        if let Some(directive) = trimmed.strip_prefix('#') {
            let directive = directive.trim_start();
            let keyword: String = directive.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
            let operand = directive[keyword.len()..].split("//").next().unwrap_or("").trim();
            let operand = operand.split("/*").next().unwrap_or("").trim();
            match keyword.as_str() {
                "ifndef" => stack.push(true),
                "if" => match operand {
                    "0" => stack.push(false),
                    "1" => stack.push(true),
                    other => return Err(format!("line {}: unsupported `#if {other}`", index + 1)),
                },
                "endif" => {
                    stack.pop().ok_or_else(|| format!("line {}: #endif without #if", index + 1))?;
                }
                "define" => {}
                other => return Err(format!("line {}: unsupported directive `#{other}`", index + 1)),
            }
            continue;
        }
        if stack.iter().all(|&active| active) {
            let code = line.split("//").next().unwrap_or("");
            if code.contains("/*") {
                return Err(format!("line {}: block comments are not supported", index + 1));
            }
            out.push_str(code);
            out.push('\n');
        }
    }
    if !stack.is_empty() {
        return Err("unterminated #if".to_string());
    }
    Ok(out)
}

/// Text up to the delimiter that closes an already-opened `(`/`{` group
/// (nesting of both kinds is tracked), and the remainder after it.
fn take_balanced(s: &str, close: char) -> Option<(&str, &str)> {
    let mut depth = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '(' | '{' | '[' => depth += 1,
            ')' | '}' | ']' if depth > 0 => depth -= 1,
            c if c == close && depth == 0 => return Some((&s[..i], &s[i + c.len_utf8()..])),
            _ => {}
        }
    }
    None
}

fn split_top_level(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let (mut depth, mut start) = (0usize, 0usize);
    for (i, c) in s.char_indices() {
        match c {
            '(' | '{' | '[' => depth += 1,
            ')' | '}' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if !s.trim().is_empty() {
        parts.push(&s[start..]);
    }
    parts
}

fn snippet(s: &str) -> &str {
    let end = s.char_indices().nth(60).map_or(s.len(), |(i, _)| i);
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREAMBLE: &str = "#ifndef __GUARD\n#define __GUARD\n";

    fn parse(body: &str) -> Result<Vec<ExtractedGeometry>, String> {
        parse_vmult_header(&format!("{PREAMBLE}{body}\n#endif /*__GUARD*/\n"))
    }

    #[test]
    fn dense_intrinsic_yields_shape_widths_and_key() {
        let entries = parse(
            "INTRINSIC(v32acc32) mac_4x16_16x8(v64uint8 a, v128uint4 b, v32acc32 acc1) {\n\
             int conf = aiev2_compute_control(__SIGN_UNSIGNED, __SIGN_UNSIGNED, 0, 0, 0, 0,\n\
                                              0, 0, 0, 0, 0);\n\
             return __builtin_x(a, b, acc1, conf);\n}",
        )
        .unwrap();
        assert_eq!(
            entries,
            vec![ExtractedGeometry {
                amode: 0,
                bmode: 0,
                variant: 0,
                bits_x: 8,
                bits_y: 4,
                rows: 4,
                inner: 16,
                cols: 8,
                acc_cmb: 1,
                bfloat: false,
                sparse: false,
            }]
        );
    }

    #[test]
    fn sparse_b_operand_marks_sparse_and_64bit_acc_sets_combining() {
        let entries = parse(
            "INTRINSIC(v16acc64) mul_2x16_16x8(v32int16 a, int sgn_x, v128int8_sparse b, int sgn_y) {\n\
             int conf = aiev2_compute_control(sgn_x, sgn_y, 1, 2, 2, 0, 0, 0, 0, 0, 0);\n\
             return 0;\n}",
        )
        .unwrap();
        let e = entries[0];
        assert_eq!((e.amode, e.bmode, e.variant), (1, 2, 2));
        assert_eq!((e.bits_x, e.bits_y, e.acc_cmb), (16, 8, 2));
        assert!(e.sparse && !e.bfloat);
    }

    #[test]
    fn bfloat16_operands_and_accfloat_result() {
        let entries = parse(
            "INTRINSIC(v16accfloat) mul_4x8_8x4(v32bfloat16 a, v32bfloat16 b) {\n\
             int conf = aiev2_compute_control(0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0);\n return 0;\n}",
        )
        .unwrap();
        let e = entries[0];
        assert_eq!((e.bits_x, e.bits_y, e.acc_cmb), (16, 16, 1));
        assert!(e.bfloat && !e.sparse);
    }

    #[test]
    fn duplicate_flavours_collapse_and_non_matmul_names_are_ignored() {
        let entries = parse(
            "INTRINSIC(v32acc32) mul_4x8_8x8(v64int8 a, v64int8 b) {\n\
             int conf = aiev2_compute_control(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);\n return 0;\n}\n\
             INTRINSIC(v32acc32) msc_4x8_8x8(v64int8 a, v64int8 b, v32acc32 acc1) {\n\
             int conf = aiev2_compute_control(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);\n return 0;\n}\n\
             INTRINSIC(v32acc32) clr32() { return 0; }\n\
             INTRINSIC(v32acc32) mul_elem_32_2(v32int16 a, v32int16 b) {\n\
             int conf = aiev2_compute_control(0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0);\n return 0;\n}",
        )
        .unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn inactive_regions_are_skipped() {
        let entries = parse(
            "#if 0 // Complex\n\
             INTRINSIC(v32acc32) mul_4x4_4x4(v64int8 a, v64int8 b) {\n\
             int conf = aiev2_compute_control(0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0);\n return 0;\n}\n\
             #endif\n\
             INTRINSIC(v32acc32) mul_4x8_8x8(v64int8 a, v64int8 b) {\n\
             int conf = aiev2_compute_control(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);\n return 0;\n}",
        )
        .unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].bmode, 1);
    }

    #[test]
    fn non_literal_config_field_is_an_error() {
        let err = parse(
            "INTRINSIC(v32acc32) mul_4x8_8x8(v64int8 a, v64int8 b, int m) {\n\
             int conf = aiev2_compute_control(0, 0, m, 1, 0, 0, 0, 0, 0, 0, 0);\n return 0;\n}",
        )
        .unwrap_err();
        assert!(err.contains("amode") && err.contains("not an integer literal"), "{err}");
    }

    #[test]
    fn conflicting_geometries_for_one_key_are_an_error() {
        let err = parse(
            "INTRINSIC(v32acc32) mul_4x8_8x8(v64int8 a, v64int8 b) {\n\
             int conf = aiev2_compute_control(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);\n return 0;\n}\n\
             INTRINSIC(v32acc32) mul_2x8_8x8(v64int8 a, v64int8 b) {\n\
             int conf = aiev2_compute_control(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);\n return 0;\n}",
        )
        .unwrap_err();
        assert!(err.contains("conflicting"), "{err}");
    }

    #[test]
    fn unrecognised_operand_type_is_an_error() {
        let err = parse(
            "INTRINSIC(v32acc32) mul_4x8_8x8(v64float32 a, v64int8 b) {\n\
             int conf = aiev2_compute_control(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);\n return 0;\n}",
        )
        .unwrap_err();
        assert!(err.contains("unrecognised element type"), "{err}");
    }

    #[test]
    fn matmul_shaped_intrinsic_without_config_call_is_an_error() {
        let err = parse("INTRINSIC(v32acc32) mul_4x8_8x8(v64int8 a, v64int8 b) { return 0; }").unwrap_err();
        assert!(err.contains("no aiev2_compute_control"), "{err}");
    }

    #[test]
    fn delegating_wrapper_overload_is_skipped() {
        let entries = parse(
            "INTRINSIC(v32acc32) mul_4x8_8x8(v64int8 a, v64int8 b) {\n\
             int conf = aiev2_compute_control(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);\n return 0;\n}\n\
             INTRINSIC(v32acc32) mul_4x8_8x8(v32int16 a0, v64int8 b) {\n\
             v32acc32 acc = ::mul_4x8_8x8(hi, b);\n return acc;\n}",
        )
        .unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn unsupported_preprocessor_conditional_is_an_error() {
        let err = parse("#if defined(FOO)\n#endif").unwrap_err();
        assert!(err.contains("unsupported"), "{err}");
    }

    #[test]
    fn header_without_matmul_intrinsics_is_an_error() {
        assert!(parse("INTRINSIC(v32acc32) clr32() { return 0; }").is_err());
    }

    #[test]
    fn rendered_table_names_every_field() {
        let entries = parse(
            "INTRINSIC(v32acc32) mul_4x8_8x8(v64int8 a, v64int8 b) {\n\
             int conf = aiev2_compute_control(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);\n return 0;\n}",
        )
        .unwrap();
        let rendered = render_config_geometry_table(&entries);
        assert!(
            rendered.contains("(0, 1, 0, GeometryEntry { bits_x: 8, bits_y: 8, rows: 4, inner: 8, cols: 8,")
        );
    }
}
