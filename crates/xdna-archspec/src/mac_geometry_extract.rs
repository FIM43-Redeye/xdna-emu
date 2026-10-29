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
//! A second pass ([`parse_vmult_modes`]) is looser about names and keeps every
//! intrinsic that builds a config word, so elementwise, convolution,
//! multi-channel and accumulator-only intrinsics are covered as well: one
//! [`ExtractedMode`] per config-word key with the signedness pairs, operation
//! kinds and representative family name the header gives it.
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

/// Values of the `__SIGN_*` macros the intrinsic headers use for the
/// `sgn_x`/`sgn_y` arguments of `aiev2_compute_control`.
///
/// They are defined in `aiev2intrin.h`, not in `aiev2_vmult.h`, so the build
/// reads them from there rather than assuming 0/1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignMacros {
    pub unsigned: u32,
    pub signed: u32,
}

impl SignMacros {
    /// Read `#define __SIGN_UNSIGNED <n>` and `#define __SIGN_SIGNED <n>`.
    pub fn parse(source: &str) -> Result<Self, String> {
        let find = |name: &str| -> Result<u32, String> {
            source
                .lines()
                .filter_map(|line| line.trim_start().strip_prefix('#'))
                .filter_map(|directive| directive.trim_start().strip_prefix("define"))
                .filter_map(|rest| rest.trim_start().strip_prefix(name))
                .filter(|rest| rest.starts_with(char::is_whitespace))
                .find_map(|rest| rest.split("//").next()?.trim().parse::<u32>().ok())
                .ok_or_else(|| format!("no `#define {name} <integer>` found"))
        };
        Ok(Self { unsigned: find("__SIGN_UNSIGNED")?, signed: find("__SIGN_SIGNED")? })
    }
}

/// Operation kinds, as (builtin stem, `MacOpKind` variant name), in the order
/// `ExtractedMode::ops` and the generated tables list them. The stem is what
/// the builtin name carries (`..._acc32_negmac_conf` -> `negmac`).
const OP_KINDS: &[(&str, &str)] = &[
    ("mul", "Mul"),
    ("negmul", "NegMul"),
    ("mac", "Mac"),
    ("msc", "Msc"),
    ("negmac", "NegMac"),
    ("negmsc", "NegMsc"),
    ("addmac", "AddMac"),
    ("addmsc", "AddMsc"),
    ("submac", "SubMac"),
    ("submsc", "SubMsc"),
    ("add", "Add"),
    ("sub", "Sub"),
    ("negadd", "NegAdd"),
    ("negsub", "NegSub"),
    ("neg", "Neg"),
];

/// Kinds that multiply (the rest only combine accumulators).
const MULTIPLY_KINDS: &[&str] =
    &["mul", "negmul", "mac", "msc", "negmac", "negmsc", "addmac", "addmsc", "submac", "submsc"];

/// One distinct (amode, bmode, variant) config mode and what the header does
/// with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedMode {
    pub amode: u32,
    pub bmode: u32,
    pub variant: u32,
    /// Distinct `(sgn_x, sgn_y)` pairs the header can build, sorted.
    pub signs: Vec<(u32, u32)>,
    /// From the operand vector types of the multiplying intrinsics.
    pub bfloat: bool,
    /// Name of the multiplying intrinsics that build the mode, with the
    /// operation prefix and `_conf` suffix removed (`4x16_16x8`, `elem_16_2`,
    /// `conv_4x4_8ch`). Every multiplying intrinsic of a mode must agree on
    /// it. Overloads on operand type reuse a family name across modes, so
    /// this is a label for humans, not a key.
    pub family: String,
    /// `MacOpKind` variant names, in `OP_KINDS` order.
    pub ops: Vec<&'static str>,
}

/// Parse every intrinsic that builds a config word, whatever its name.
///
/// Returns one entry per distinct (amode, bmode, variant), sorted by that key.
/// Accumulator-only intrinsics (`add`, `sub`, `negadd`, `neg`) pass their
/// bmode in redundant parentheses (`(0)`, `(3)`); these are plain literals,
/// not macros, and are read the same way. Any construct the parser does not
/// understand is an error.
pub fn parse_vmult_modes(source: &str, signs: SignMacros) -> Result<Vec<ExtractedMode>, String> {
    #[derive(Default)]
    struct Accum {
        signs: std::collections::BTreeSet<(u32, u32)>,
        bfloat: Option<bool>,
        family: Option<String>,
        ops: std::collections::BTreeSet<usize>,
    }
    let text = strip_inactive_regions(source)?;
    let mut by_key: BTreeMap<(u32, u32, u32), Accum> = BTreeMap::new();

    let mut rest = text.as_str();
    while let Some(at) = rest.find("INTRINSIC(") {
        rest = &rest[at + "INTRINSIC(".len()..];
        let (_ret_type, after_ret) = take_balanced(rest, ')')
            .ok_or_else(|| format!("unterminated INTRINSIC( near `{}`", snippet(rest)))?;
        let after_ret = after_ret.trim_start();
        let name_len = after_ret.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(0);
        let name = &after_ret[..name_len];
        let after_name = after_ret[name_len..].trim_start();
        let Some((params, after_params)) = after_name.strip_prefix('(').and_then(|s| take_balanced(s, ')'))
        else {
            rest = after_name;
            continue;
        };
        let Some(body_src) = after_params.trim_start().strip_prefix('{') else {
            rest = after_params;
            continue;
        };
        let (body, after_body) =
            take_balanced(body_src, '}').ok_or_else(|| format!("{name}: unterminated body"))?;
        rest = after_body;

        let Some(call_at) = body.find("aiev2_compute_control(") else {
            continue; // delegating wrapper or an intrinsic with no config word
        };
        let after_call = &body[call_at + "aiev2_compute_control(".len()..];
        if after_call.contains("aiev2_compute_control(") {
            return Err(format!("{name}: more than one aiev2_compute_control call"));
        }
        let (call_args, _) = take_balanced(after_call, ')')
            .ok_or_else(|| format!("{name}: unterminated aiev2_compute_control call"))?;
        let args = split_top_level(call_args);
        if args.len() != 11 {
            return Err(format!("{name}: aiev2_compute_control has {} arguments, expected 11", args.len()));
        }
        let literal = |index: usize, what: &str| -> Result<u32, String> {
            let arg = strip_redundant_parens(args[index]);
            arg.parse::<u32>()
                .map_err(|_| format!("{name}: {what} argument `{arg}` is not an integer literal"))
        };
        let key = (literal(2, "amode")?, literal(3, "bmode")?, literal(4, "variant")?);

        let param_names: Vec<&str> = split_top_level(params)
            .into_iter()
            .filter_map(|param| {
                param
                    .trim()
                    .rsplit_once(char::is_whitespace)
                    .map(|(_, n)| n.trim_start_matches(['*', '&']))
            })
            .collect();
        let sign_values = |index: usize, what: &str| -> Result<Vec<u32>, String> {
            let arg = strip_redundant_parens(args[index]);
            match arg {
                "__SIGN_UNSIGNED" => Ok(vec![signs.unsigned]),
                "__SIGN_SIGNED" => Ok(vec![signs.signed]),
                _ if arg.parse::<u32>().is_ok() => Ok(vec![arg.parse().unwrap()]),
                // A runtime parameter can be either signedness.
                _ if param_names.contains(&arg) => Ok(vec![signs.unsigned, signs.signed]),
                _ => {
                    Err(format!("{name}: {what} argument `{arg}` is not a sign macro, literal or parameter"))
                }
            }
        };
        let (xs, ys) = (sign_values(0, "sgn_x")?, sign_values(1, "sgn_y")?);

        let builtins: std::collections::BTreeSet<&str> = body
            .match_indices("__builtin_aiev2_")
            .map(|(i, _)| {
                let tail = &body[i..];
                &tail[..tail
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .unwrap_or(tail.len())]
            })
            .collect();
        let builtin = match builtins.iter().collect::<Vec<_>>()[..] {
            [only] => *only,
            _ => return Err(format!("{name}: expected exactly one builtin call, found {builtins:?}")),
        };
        let stem = op_stem_of_builtin(builtin).map_err(|e| format!("{name}: {e}"))?;
        let kind_index = OP_KINDS
            .iter()
            .position(|(s, _)| *s == stem)
            .expect("op_stem_of_builtin returns known stems");

        let entry = by_key.entry(key).or_default();
        for &x in &xs {
            for &y in &ys {
                entry.signs.insert((x, y));
            }
        }
        entry.ops.insert(kind_index);

        if MULTIPLY_KINDS.contains(&stem) {
            // Operand vectors decide bfloat16; accumulator-only intrinsics have none.
            let mut operands = split_top_level(params).into_iter().filter_map(|param| {
                let ty = param.trim().rsplit_once(char::is_whitespace)?.0.trim();
                (ty.starts_with('v') && !ty.contains("acc")).then_some(ty)
            });
            let x = operands.next().ok_or_else(|| format!("{name}: no A operand type"))?;
            let y = operands.next().ok_or_else(|| format!("{name}: no B operand type"))?;
            let x = parse_vector_type(x).map_err(|e| format!("{name}: A operand: {e}"))?;
            let y = parse_vector_type(y).map_err(|e| format!("{name}: B operand: {e}"))?;
            if x.bfloat != y.bfloat {
                return Err(format!("{name}: mixed bfloat16 and integer operands"));
            }
            match entry.bfloat {
                Some(seen) if seen != x.bfloat => {
                    return Err(format!(
                        "{name}: bfloat16-ness disagrees with an earlier intrinsic of mode {key:?}"
                    ));
                }
                _ => entry.bfloat = Some(x.bfloat),
            }
            let family = name.strip_prefix(&format!("{stem}_")).unwrap_or(name);
            let family = family.strip_suffix("_conf").unwrap_or(family);
            match &entry.family {
                Some(first) if first != family => {
                    return Err(format!(
                        "mode {key:?} is built by two different families: `{first}` and `{family}` ({name})"
                    ));
                }
                Some(_) => {}
                None => entry.family = Some(family.to_string()),
            }
        }
    }

    if by_key.is_empty() {
        return Err("no config-word intrinsics found (is this aiev2_vmult.h?)".to_string());
    }
    by_key
        .into_iter()
        .map(|((amode, bmode, variant), acc)| {
            let missing = |what: &str| {
                format!("mode ({amode}, {bmode}, {variant}) has no multiplying intrinsic ({what})")
            };
            Ok(ExtractedMode {
                amode,
                bmode,
                variant,
                signs: acc.signs.into_iter().collect(),
                bfloat: acc.bfloat.ok_or_else(|| missing("bfloat16-ness"))?,
                family: acc.family.ok_or_else(|| missing("family name"))?,
                ops: acc.ops.into_iter().map(|i| OP_KINDS[i].1).collect(),
            })
        })
        .collect()
}

/// The operation stem a builtin name carries, one of `OP_KINDS`' stems.
///
/// Config-driven builtins end `_<stem>_conf` (`..._acc32_negmac_conf`,
/// `..._bf_addmac_conf`, and the bfloat16 `bf_mul16_conf` family, whose stem
/// carries the lane count); accumulator-only ones end `_<stem>_acc` or
/// `_<stem>_accfloat` (`add_acc`, `negsub_accfloat`).
fn op_stem_of_builtin(builtin: &str) -> Result<&'static str, String> {
    let name = builtin
        .strip_prefix("__builtin_aiev2_")
        .ok_or_else(|| format!("builtin `{builtin}` is not a __builtin_aiev2_ call"))?;
    let stem = if let Some(base) = name.strip_suffix("_conf") {
        let last = base.rsplit('_').next().unwrap_or(base);
        last.strip_suffix("16").unwrap_or(last)
    } else if let Some(base) = name.strip_suffix("_accfloat").or_else(|| name.strip_suffix("_acc")) {
        base
    } else {
        return Err(format!("builtin `{builtin}` has no recognised operation suffix"));
    };
    OP_KINDS
        .iter()
        .find(|(known, _)| *known == stem)
        .map(|(known, _)| *known)
        .ok_or_else(|| format!("builtin `{builtin}`: unknown operation `{stem}`"))
}

/// `(0)` -> `0`, `((3))` -> `3`; anything else unchanged (trimmed).
fn strip_redundant_parens(arg: &str) -> &str {
    let mut arg = arg.trim();
    while let Some(inner) = arg.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        arg = inner.trim();
    }
    arg
}

/// Render the mode table as Rust source defining `MAC_CONFIG_MODES`, to be
/// `include!`d next to `MacConfigMode`/`MacOpKind` in `aie2/matmul.rs`.
/// `geometries` decides each entry's `has_geometry`.
pub fn render_mac_config_modes(modes: &[ExtractedMode], geometries: &[ExtractedGeometry]) -> String {
    let mut out = String::new();
    writeln!(out, "/// Every config mode aiev2_vmult.h builds, sorted by (amode, bmode, variant).").unwrap();
    writeln!(out, "pub const MAC_CONFIG_MODES: &[MacConfigMode] = &[").unwrap();
    for m in modes {
        let signs: Vec<String> = m.signs.iter().map(|(x, y)| format!("({x}, {y})")).collect();
        let ops: Vec<String> = m.ops.iter().map(|op| format!("MacOpKind::{op}")).collect();
        let has_geometry = geometries
            .iter()
            .any(|g| (g.amode, g.bmode, g.variant) == (m.amode, m.bmode, m.variant));
        writeln!(
            out,
            "    MacConfigMode {{ amode: {}, bmode: {}, variant: {}, signs: &[{}], bfloat: {}, \
             family: {:?}, ops: &[{}], has_geometry: {} }},",
            m.amode,
            m.bmode,
            m.variant,
            signs.join(", "),
            m.bfloat,
            m.family,
            ops.join(", "),
            has_geometry
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

    // ---- parse_vmult_modes ----

    const SIGNS: SignMacros = SignMacros { unsigned: 0, signed: 1 };

    fn modes(body: &str) -> Result<Vec<ExtractedMode>, String> {
        parse_vmult_modes(&format!("{PREAMBLE}{body}\n#endif /*__GUARD*/\n"), SIGNS)
    }

    #[test]
    fn sign_macros_are_read_from_their_defines() {
        let m = SignMacros::parse("#define __SIGN_UNSIGNED 0\n#define __SIGN_SIGNED 1 // two's\n").unwrap();
        assert_eq!(m, SignMacros { unsigned: 0, signed: 1 });
        // Not assumed: a header that defined them the other way round is honoured.
        let m = SignMacros::parse("#define __SIGN_SIGNED 0\n#define __SIGN_UNSIGNED 1\n").unwrap();
        assert_eq!(m, SignMacros { unsigned: 1, signed: 0 });
        assert!(SignMacros::parse("#define __SIGN_SIGNED 1\n").is_err());
        // A longer macro name sharing the prefix must not match.
        assert!(SignMacros::parse("#define __SIGN_SIGNED_X 1\n#define __SIGN_UNSIGNED 0\n").is_err());
    }

    #[test]
    fn elementwise_intrinsics_are_modes_without_a_matmul_shape() {
        let m = modes(
            "INTRINSIC(v32acc32) mul_elem_32_2(v32int16 a, v32int16 b) {\n\
             int conf = aiev2_compute_control(__SIGN_SIGNED, __SIGN_UNSIGNED, 0, 1, 1, 0, 0, 0, 0, 0, 0);\n\
             return __builtin_aiev2_I512_I512_ACC1024_acc32_mul_conf(a, b, conf);\n}",
        )
        .unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!((m[0].amode, m[0].bmode, m[0].variant), (0, 1, 1));
        assert_eq!(m[0].signs, vec![(1, 0)]);
        assert_eq!(m[0].family, "elem_32_2");
        assert_eq!(m[0].ops, vec!["Mul"]);
        assert!(!m[0].bfloat);
    }

    #[test]
    fn runtime_sign_parameters_contribute_both_values_and_flavours_merge() {
        let m = modes(
            "INTRINSIC(v32acc32) mul_conv_32x8(v64int8 a, int sgn_x, v64int8 b, int sgn_y) {\n\
             int conf = aiev2_compute_control(sgn_x, sgn_y, 0, 1, 4, 0, 0, 0, 0, 0, 0);\n\
             return __builtin_aiev2_I512_I512_ACC1024_acc32_mul_conf(a, b, conf);\n}\n\
             INTRINSIC(v32acc32) negmul_conv_32x8(v64int8 a, v64int8 b) {\n\
             int conf = aiev2_compute_control(__SIGN_UNSIGNED, __SIGN_UNSIGNED, 0, 1, 4, 0, 0, 0, 0, 0, 0);\n\
             return __builtin_aiev2_I512_I512_ACC1024_acc32_negmul_conf(a, b, conf);\n}\n\
             INTRINSIC(v32acc32) mac_conv_32x8_conf(v64int8 a, v64int8 b, v32acc32 acc1, int zero_acc1, \
             int sub_mul, int sub_acc1, int sub_acc2) {\n\
             int conf = aiev2_compute_control(__SIGN_SIGNED, __SIGN_SIGNED, 0, 1, 4, zero_acc1, 0, sub_mul, \
             sub_acc1, sub_acc2, 0);\n\
             return __builtin_aiev2_I512_I512_ACC1024_acc32_mac_conf(a, b, acc1, conf);\n}",
        )
        .unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].signs, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
        assert_eq!(m[0].family, "conv_32x8");
        // Listed in OP_KINDS order, whatever the header order.
        assert_eq!(m[0].ops, vec!["Mul", "NegMul", "Mac"]);
    }

    #[test]
    fn bfloat16_mode_and_bf_builtin_stems() {
        let m = modes(
            "INTRINSIC(v16accfloat) mul_elem_16_2(v32bfloat16 a, v32bfloat16 b) {\n\
             int conf = aiev2_compute_control(0, 0, 2, 3, 1, 0, 0, 0, 0, 0, 0);\n\
             return __builtin_aiev2_bf_mul16_conf(a, b, conf);\n}\n\
             INTRINSIC(v16accfloat) addmsc_elem_16_2(v32bfloat16 a, v32bfloat16 b, v16accfloat acc1, \
             v16accfloat acc2) {\n\
             int conf = aiev2_compute_control(0, 0, 2, 3, 1, 0, 0, 0, 0, 0, 0);\n\
             return __builtin_aiev2_bf_addmsc16_conf(a, b, acc1, acc2, conf);\n}",
        )
        .unwrap();
        assert!(m[0].bfloat);
        assert_eq!(m[0].signs, vec![(0, 0)]);
        assert_eq!(m[0].ops, vec!["Mul", "AddMsc"]);
        assert_eq!(m[0].family, "elem_16_2");
    }

    #[test]
    fn parenthesised_fields_on_accumulator_only_intrinsics_are_plain_literals() {
        // `add`/`sub`/`neg` wrap their bmode in redundant parentheses and have
        // no operand vectors; they share the config key of the multiplying
        // intrinsics and add their operation kind to it.
        let m = modes(
            "INTRINSIC(v32acc32) mul_4x8_8x8(v64int8 a, v64int8 b) {\n\
             int conf = aiev2_compute_control(0, 0, 0, (0), 0, 0, 0, 0, 0, 0, 0);\n\
             return __builtin_aiev2_I512_I512_ACC1024_acc32_mul_conf(a, b, conf);\n}\n\
             INTRINSIC(v32acc32) add(v32acc32 acc1, v32acc32 acc2) {\n\
             int conf = aiev2_compute_control(0, 0, 0, (0), 0, 0, 0, 0, 0, 0, 0);\n\
             return __builtin_aiev2_add_acc(acc1, acc2, conf);\n}\n\
             INTRINSIC(v32acc32) neg(v32acc32 acc) {\n\
             int conf = aiev2_compute_control(0, 0, 0, ((0)), 0, 0, 0, 0, 0, 0, 0);\n\
             return __builtin_aiev2_ACC1024_acc32_neg_conf(acc, conf);\n}",
        )
        .unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!((m[0].amode, m[0].bmode, m[0].variant), (0, 0, 0));
        assert_eq!(m[0].ops, vec!["Mul", "Add", "Neg"]);
        assert_eq!(m[0].family, "4x8_8x8");
    }

    #[test]
    fn float_accumulator_builtins_map_to_their_kinds() {
        assert_eq!(op_stem_of_builtin("__builtin_aiev2_negsub_accfloat"), Ok("negsub"));
        assert_eq!(op_stem_of_builtin("__builtin_aiev2_sub_accfloat"), Ok("sub"));
        assert_eq!(op_stem_of_builtin("__builtin_aiev2_ACC512_accfloat_neg_conf"), Ok("neg"));
        assert_eq!(
            op_stem_of_builtin("__builtin_aiev2_I1024_I1024_ACC1024_accfloat_bf_addmac_conf"),
            Ok("addmac")
        );
        assert_eq!(op_stem_of_builtin("__builtin_aiev2_bf_negmul16_conf"), Ok("negmul"));
        assert!(op_stem_of_builtin("__builtin_aiev2_I512_acc32_frobnicate_conf")
            .unwrap_err()
            .contains("unknown operation"));
        assert!(op_stem_of_builtin("__builtin_aiev2_vbroadcast_zero")
            .unwrap_err()
            .contains("no recognised"));
    }

    #[test]
    fn intrinsics_without_a_config_word_and_inactive_regions_are_ignored() {
        let m = modes(
            "INTRINSIC(v32acc32) clr32() { return 0; }\n\
             #if 0 // Complex\n\
             INTRINSIC(v8cacc64) mul_elem_8(v8cint32 a, v16cint16 b) {\n\
             int conf = aiev2_compute_control(1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 10);\n\
             return __builtin_aiev2_I512_I512_ACC1024_acc64_mul_conf(a, b, conf);\n}\n\
             #endif\n\
             INTRINSIC(v32acc32) mul_4x8_8x8(v64int8 a, v64int8 b) {\n\
             int conf = aiev2_compute_control(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);\n\
             return __builtin_aiev2_I512_I512_ACC1024_acc32_mul_conf(a, b, conf);\n}",
        )
        .unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].bmode, 1);
    }

    #[test]
    fn mode_parser_rejects_what_it_cannot_interpret() {
        let call = |sgn: &str, field: &str, builtin: &str| {
            modes(&format!(
                "INTRINSIC(v32acc32) mul_4x8_8x8(v64int8 a, v64int8 b) {{\n\
                 int conf = aiev2_compute_control({sgn}, 0, {field}, 1, 0, 0, 0, 0, 0, 0, 0);\n\
                 return {builtin}(a, b, conf);\n}}"
            ))
        };
        let ok_builtin = "__builtin_aiev2_I512_I512_ACC1024_acc32_mul_conf";
        assert!(call("0", "0", ok_builtin).is_ok());
        assert!(call("mystery", "0", ok_builtin).unwrap_err().contains("sgn_x"));
        assert!(call("0", "m", ok_builtin).unwrap_err().contains("not an integer literal"));
        assert!(call("0", "0", "__builtin_aiev2_I512_acc32_frobnicate_conf")
            .unwrap_err()
            .contains("unknown operation"));
        // No builtin at all.
        let err = modes(
            "INTRINSIC(v32acc32) mul_4x8_8x8(v64int8 a, v64int8 b) {\n\
             int conf = aiev2_compute_control(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);\n return 0;\n}",
        )
        .unwrap_err();
        assert!(err.contains("exactly one builtin"), "{err}");
    }

    #[test]
    fn two_families_for_one_mode_are_an_error() {
        let err = modes(
            "INTRINSIC(v32acc32) mul_4x8_8x8(v64int8 a, v64int8 b) {\n\
             int conf = aiev2_compute_control(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);\n\
             return __builtin_aiev2_I512_I512_ACC1024_acc32_mul_conf(a, b, conf);\n}\n\
             INTRINSIC(v32acc32) mul_2x8_8x8(v64int8 a, v64int8 b) {\n\
             int conf = aiev2_compute_control(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);\n\
             return __builtin_aiev2_I512_I512_ACC1024_acc32_mul_conf(a, b, conf);\n}",
        )
        .unwrap_err();
        assert!(err.contains("two different families"), "{err}");
    }

    #[test]
    fn rendered_mode_table_names_every_field_and_geometry_presence() {
        let source = "INTRINSIC(v32acc32) mul_4x8_8x8(v64int8 a, v64int8 b) {\n\
             int conf = aiev2_compute_control(__SIGN_SIGNED, __SIGN_UNSIGNED, 0, 1, 0, 0, 0, 0, 0, 0, 0);\n\
             return __builtin_aiev2_I512_I512_ACC1024_acc32_mul_conf(a, b, conf);\n}\n\
             INTRINSIC(v32acc32) mul_elem_32(v32int16 a, v32int16 b) {\n\
             int conf = aiev2_compute_control(0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0);\n\
             return __builtin_aiev2_I512_I512_ACC1024_acc32_mul_conf(a, b, conf);\n}";
        let geometries = parse(source).unwrap();
        let rendered = render_mac_config_modes(&modes(source).unwrap(), &geometries);
        assert!(
            rendered.contains(
                "MacConfigMode { amode: 0, bmode: 1, variant: 0, signs: &[(1, 0)], bfloat: false, \
             family: \"4x8_8x8\", ops: &[MacOpKind::Mul], has_geometry: true }"
            ),
            "{rendered}"
        );
        assert!(rendered.contains("amode: 0, bmode: 3, variant: 1, signs: &[(0, 0)]"), "{rendered}");
        assert!(
            rendered.contains("family: \"elem_32\", ops: &[MacOpKind::Mul], has_geometry: false }"),
            "{rendered}"
        );
    }
}
