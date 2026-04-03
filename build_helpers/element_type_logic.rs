// Shared element type inference logic.
//
// This file is the SINGLE SOURCE OF TRUTH for instruction name -> element type
// mapping. It is textually included by both:
//   - build_helpers/semantics.rs  (build-time codegen, String output)
//   - src/tablegen/resolver.rs    (runtime decoder, ElementType output)
//
// It uses a self-contained TypeTag enum with no external dependencies so it
// compiles in both contexts. Each consumer provides a thin conversion layer
// from TypeTag to its own output type.
//
// IMPORTANT: when adding patterns here, both build-time and runtime consumers
// automatically pick them up. No more dual maintenance.

/// Lightweight type discriminant shared between build-time and runtime.
///
/// Intentionally has no derives beyond what both contexts need (Copy, Clone,
/// PartialEq, Eq, Debug). No dependency on ElementType or any crate type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeTag {
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    BFloat16,
    Float32,
}

/// Parse a type token from an instruction name component.
///
/// Tokens appear between underscores in encoding names:
/// `VSRS_S16_S32_mv_w_srs` -> "S16", "S32"
pub fn parse_type_tag(token: &str) -> Option<TypeTag> {
    match token {
        "S8" => Some(TypeTag::Int8),
        "D8" => Some(TypeTag::UInt8),
        "S16" => Some(TypeTag::Int16),
        "D16" => Some(TypeTag::UInt16),
        "S32" => Some(TypeTag::Int32),
        "D32" => Some(TypeTag::UInt32),
        "S64" => Some(TypeTag::Int64),
        "D64" => Some(TypeTag::UInt64),
        "FP32" => Some(TypeTag::Float32),
        "BF16" => Some(TypeTag::BFloat16),
        _ => None,
    }
}

/// Infer both element types for dual-type instructions.
///
/// Returns `(output_type, input_type)` based on encoding name patterns.
/// All patterns use `{OUT}_{IN}` ordering (output type first, input second).
///
/// Supported patterns:
/// - `V{SRS|VSRSM|UPS}_{OUT}_{IN}_*`
/// - `VFLOOR_{OUT}_{IN}_*`
/// - `VCONV_{OUT}_{IN}`
/// - `V{LDA|ST}_{2D|3D}_{UPS|SRS}_{OUT}_{IN}*`
/// - `V{LDA|ST}_{UPS|SRS}_{OUT}_{IN}_*`
/// - `V{LDA|ST}_*_CONV_{OUT}_{IN}*` (fused conv, optional 2D/3D after CONV)
pub fn infer_dual_type_tags(name: &str) -> (Option<TypeTag>, Option<TypeTag>) {
    let parts: Vec<&str> = name.split('_').collect();

    // Pattern 1: V{SRS|UPS|FLOOR|CONV}_{OUT}_{IN}_*
    if parts.len() >= 3
        && matches!(
            parts[0],
            "VSRS" | "VSRSM" | "VUPS" | "VFLOOR" | "VCONV"
        )
    {
        if let (Some(out), Some(inp)) = (parse_type_tag(parts[1]), parse_type_tag(parts[2])) {
            return (Some(out), Some(inp));
        }
    }

    // Pattern 2: V{LDA|ST}_{2D|3D}_{UPS|SRS}_{OUT}_{IN}*
    if parts.len() >= 5
        && matches!(parts[0], "VLDA" | "VST")
        && matches!(parts[2], "UPS" | "SRS")
    {
        if let (Some(out), Some(inp)) = (parse_type_tag(parts[3]), parse_type_tag(parts[4])) {
            return (Some(out), Some(inp));
        }
    }

    // Pattern 3: V{LDA|ST}_{UPS|SRS}_{OUT}_{IN}_* (no 2D/3D prefix)
    if parts.len() >= 4
        && matches!(parts[0], "VLDA" | "VST")
        && matches!(parts[1], "UPS" | "SRS")
    {
        if let (Some(out), Some(inp)) = (parse_type_tag(parts[2]), parse_type_tag(parts[3])) {
            return (Some(out), Some(inp));
        }
    }

    // Pattern 4: Fused CONV -- V{LDA|ST}_*_CONV_{OUT}_{IN}*
    if parts.len() >= 4 && matches!(parts[0], "VLDA" | "VST") {
        if let Some(conv_pos) = parts.iter().position(|&p| p == "CONV") {
            // Skip optional 2D/3D after CONV keyword.
            let type_start = if parts
                .get(conv_pos + 1)
                .map_or(false, |p| *p == "2D" || *p == "3D")
            {
                conv_pos + 2
            } else {
                conv_pos + 1
            };
            if type_start + 1 < parts.len() {
                if let (Some(out), Some(inp)) =
                    (parse_type_tag(parts[type_start]), parse_type_tag(parts[type_start + 1]))
                {
                    return (Some(out), Some(inp));
                }
            }
        }
    }

    (None, None)
}

/// Infer element type from a mnemonic string.
///
/// AIE2 mnemonics encode element types via suffixes:
/// - `.s8`, `.s16`, `.s32`, `.s64` -> signed integer
/// - `.d8`, `.d16`, `.d32`, `.d64` -> unsigned integer (data/raw)
/// - `.u8`, `.u16`, `.u32`, `.u64` -> unsigned integer
/// - `.bf16`, `.bf` -> BFloat16
/// - `.f32`, `.f`, `float` -> Float32
pub fn infer_type_tag_from_mnemonic(mnemonic: &str) -> Option<TypeTag> {
    let has_dot_d_digit = mnemonic.contains(".d8")
        || mnemonic.contains(".d16")
        || mnemonic.contains(".d32")
        || mnemonic.contains(".d64")
        || mnemonic.contains(".d4");
    let is_unsigned = mnemonic.contains(".u") || has_dot_d_digit;

    if mnemonic.ends_with('8') || mnemonic.contains(".i8") || mnemonic.contains(".u8") {
        Some(if is_unsigned { TypeTag::UInt8 } else { TypeTag::Int8 })
    } else if mnemonic.ends_with("16") || mnemonic.contains(".i16") || mnemonic.contains(".u16") {
        if mnemonic.contains("bf16") || mnemonic.contains(".bf") {
            return Some(TypeTag::BFloat16);
        }
        Some(if is_unsigned { TypeTag::UInt16 } else { TypeTag::Int16 })
    } else if mnemonic.ends_with("32") || mnemonic.contains(".i32") || mnemonic.contains(".u32") {
        Some(if is_unsigned { TypeTag::UInt32 } else { TypeTag::Int32 })
    } else if mnemonic.ends_with("64") || mnemonic.contains(".i64") || mnemonic.contains(".u64") {
        Some(if is_unsigned { TypeTag::UInt64 } else { TypeTag::Int64 })
    } else if mnemonic.contains("bf16") || mnemonic.contains(".bf") {
        Some(TypeTag::BFloat16)
    } else if mnemonic.contains("f32") || mnemonic.contains("float") || mnemonic.ends_with(".f") {
        Some(TypeTag::Float32)
    } else {
        None
    }
}

#[cfg(test)]
mod shared_type_tests {
    use super::*;

    #[test]
    fn test_parse_type_tag() {
        assert_eq!(parse_type_tag("S16"), Some(TypeTag::Int16));
        assert_eq!(parse_type_tag("D32"), Some(TypeTag::UInt32));
        assert_eq!(parse_type_tag("FP32"), Some(TypeTag::Float32));
        assert_eq!(parse_type_tag("BF16"), Some(TypeTag::BFloat16));
        assert_eq!(parse_type_tag("X"), None);
    }

    #[test]
    fn test_infer_dual_srs_ups() {
        let (et, ft) = infer_dual_type_tags("VSRS_S16_S32_mv_w_srs");
        assert_eq!(et, Some(TypeTag::Int16));
        assert_eq!(ft, Some(TypeTag::Int32));
    }

    #[test]
    fn test_infer_dual_vconv() {
        // VCONV_FP32_BF16 = output fp32, input bf16
        let (et, ft) = infer_dual_type_tags("VCONV_FP32_BF16");
        assert_eq!(et, Some(TypeTag::Float32));
        assert_eq!(ft, Some(TypeTag::BFloat16));

        let (et, ft) = infer_dual_type_tags("VCONV_BF16_FP32");
        assert_eq!(et, Some(TypeTag::BFloat16));
        assert_eq!(ft, Some(TypeTag::Float32));
    }

    #[test]
    fn test_infer_dual_vfloor() {
        let (et, ft) = infer_dual_type_tags("VFLOOR_S32_BF16_mFl2FxSrc_AM");
        assert_eq!(et, Some(TypeTag::Int32));
        assert_eq!(ft, Some(TypeTag::BFloat16));
    }

    #[test]
    fn test_infer_mnemonic() {
        assert_eq!(infer_type_tag_from_mnemonic("vadd.s16"), Some(TypeTag::Int16));
        assert_eq!(infer_type_tag_from_mnemonic("vadd.d32"), Some(TypeTag::UInt32));
        assert_eq!(infer_type_tag_from_mnemonic("vadd.f"), Some(TypeTag::Float32));
        assert_eq!(infer_type_tag_from_mnemonic("vadd.bf16"), Some(TypeTag::BFloat16));
        // Multi-type mnemonics like "vconv.bf16.fp32" are ambiguous for single-type
        // inference -- use infer_dual_type_tags() on the encoding name instead.
    }
}
