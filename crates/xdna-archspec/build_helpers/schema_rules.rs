//! Pure rules of the encoding-schema extraction (`extract.rs::load_encoding_schemas`).
//!
//! These take strings and values already read from TableGen records, so they depend on nothing
//! else in the build script. The crate includes this file under `#[cfg(test)]`
//! (`src/aie2/isa/schema.rs`) so its tests run with `cargo test -p xdna-archspec`; build-script
//! code has no test harness of its own.

/// How an immediate operand decodes, from the operand record's `DecoderMethod`: (encoded bits,
/// step, signed, negative). `field_width` is the width of the encoding field, used when the
/// operand has no decoder method (the LLVM default decoder takes the raw field value).
/// `None` for a decoder method this table does not know: the build fails on it, so a new
/// operand kind cannot slip through undecoded.
pub fn imm_decoder(method: &str, field_width: Option<u8>) -> Option<(u8, u16, bool, bool)> {
    let method = method.trim();
    if method.is_empty() {
        return field_width.map(|w| (w, 1, false, false));
    }
    let (name, args) = method.split_once('<')?;
    let args: Vec<u64> = args
        .strip_suffix('>')?
        .split(',')
        .map(|a| a.trim().parse().ok())
        .collect::<Option<_>>()?;
    match (name, args.as_slice()) {
        ("decodeUImmOperand", [n]) => Some((*n as u8, 1, false, false)),
        ("decodeSImmOperand", [n]) => Some((*n as u8, 1, true, false)),
        ("decodeSImmOperandXStep", [n, step, negative, unsigned]) => {
            if !step.is_power_of_two() {
                return None;
            }
            // `isNegative` overrides `isUnsigned` (AIEBaseDisassembler.h).
            Some((*n as u8, *step as u16, *negative != 0 || *unsigned == 0, *negative != 0))
        }
        _ => None,
    }
}

/// The operand ties of an instruction's `Constraints` string (`"$ptr_out = $ptr, $dst = $src"`),
/// as pairs of DAG names without `$`. Other constraint kinds (`@earlyclobber`) are ignored.
pub fn constraint_ties(constraints: &str) -> Vec<(String, String)> {
    constraints
        .split(',')
        .filter_map(|c| {
            let (a, b) = c.split_once('=')?;
            let a = a.trim().strip_prefix('$')?;
            let b = b.trim().strip_prefix('$')?;
            Some((a.to_string(), b.to_string()))
        })
        .collect()
}

/// The address-generator form of a `*__ag_<form>` class, as the `AgAddressing` variant name.
/// `None` for a class that is not an address-generator form class; panics on a form this table
/// does not know (a new TableGen addressing class must be added here, not dropped).
pub fn ag_addressing(class: &str) -> Option<&'static str> {
    let (_, form) = class.rsplit_once("__ag_")?;
    Some(match form {
        "idx" => "Indexed",
        "idx_imm" => "IndexedImm",
        "pstm_nrm" => "PostModify",
        "pstm_nrm_imm" => "PostModifyImm",
        "pstm_2d" => "TwoDim",
        "pstm_3d" => "ThreeDim",
        "spill" => "Spill",
        "pstm_sp_imm" => "SpPostModifyImm",
        other => panic!("{class}: unknown address-generator form `{other}`"),
    })
}

/// The stream form a derived format class names: (non-blocking, TLAST as the `StreamTlast`
/// variant name or `None` for a read). The llvm-aie format generator names the stream variants
/// `<format>_b` / `<format>_nb` (reads) and `<format>_{b,nb}_{noTlast,doTlast,doTlast_reg}`
/// (writes).
pub fn stream_form(class: &str) -> Option<(bool, Option<&'static str>)> {
    for (suffix, tlast) in
        [("_doTlast_reg", Some("FromRegister")), ("_doTlast", Some("Set")), ("_noTlast", Some("Clear"))]
    {
        if let Some(rest) = class.strip_suffix(suffix) {
            return blocking_suffix(rest).map(|nb| (nb, tlast));
        }
    }
    // A read: `<format>_inst_<slot>` is the format, its `_b` / `_nb` subclass the variant.
    if class.contains("_inst_") {
        return None;
    }
    blocking_suffix(class).map(|nb| (nb, None))
}

fn blocking_suffix(s: &str) -> Option<bool> {
    if s.ends_with("_nb") {
        Some(true)
    } else if s.ends_with("_b") {
        Some(false)
    } else {
        None
    }
}

/// Orders a record's classes so that each comes after every class it derives from (ties by
/// name). `derives(a, b)` says whether class `a` is a subclass of class `b`.
pub fn order_by_derivation(classes: &mut Vec<String>, derives: impl Fn(&str, &str) -> bool) {
    let depth = |c: &str, all: &[String]| all.iter().filter(|o| o.as_str() != c && derives(c, o)).count();
    let snapshot = classes.clone();
    classes.sort_by(|a, b| depth(a, &snapshot).cmp(&depth(b, &snapshot)).then_with(|| a.cmp(b)));
}

/// The values of a format class's template arguments in one record: `arg_bits` lists
/// (argument, argument bit, instruction bit) as traced through the format class's `Inst`, and
/// `literal(inst_bit)` reads the record's fixed bit there. Arguments keep the order of their
/// first entry in `arg_bits`; an argument with any non-literal bit is left out.
pub fn format_arg_values(
    arg_bits: &[(String, usize, usize)],
    literal: impl Fn(usize) -> Option<bool>,
) -> Vec<(String, u64)> {
    let mut order: Vec<&str> = Vec::new();
    for (arg, _, _) in arg_bits {
        if !order.contains(&arg.as_str()) {
            order.push(arg);
        }
    }
    order
        .into_iter()
        .filter_map(|arg| {
            let mut value = 0u64;
            for (_, arg_bit, inst_bit) in arg_bits.iter().filter(|(a, _, _)| a == arg) {
                if literal(*inst_bit)? {
                    value |= 1 << arg_bit;
                }
            }
            Some((arg.to_string(), value))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imm_decoder_reads_the_llvm_decoder_method() {
        assert_eq!(imm_decoder("decodeUImmOperand<2>", None), Some((2, 1, false, false)));
        assert_eq!(imm_decoder("decodeSImmOperand<10>", Some(10)), Some((10, 1, true, false)));
        assert_eq!(imm_decoder("decodeSImmOperandXStep<6, 4, 0, 0>", None), Some((6, 4, true, false)));
        assert_eq!(imm_decoder("decodeSImmOperandXStep<12, 32, 1, 0>", None), Some((12, 32, true, true)));
        assert_eq!(imm_decoder("decodeSImmOperandXStep<3, 32, 0, 1>", None), Some((3, 32, false, false)));
        assert_eq!(imm_decoder("decodeSImmOperandXStep<3, 32, 1, 1>", None), Some((3, 32, true, true)));
        assert_eq!(imm_decoder("", Some(20)), Some((20, 1, false, false)));
        assert_eq!(imm_decoder("", None), None);
        assert_eq!(imm_decoder("decodeSomethingNew<3>", None), None);
        assert_eq!(imm_decoder("decodeSImmOperandXStep<3, 3, 0, 0>", None), None);
    }

    #[test]
    fn constraint_ties_parse_every_pair() {
        assert!(constraint_ties("").is_empty());
        assert_eq!(constraint_ties("$ptr_out = $ptr"), [("ptr_out".to_string(), "ptr".to_string())]);
        assert_eq!(
            constraint_ties("$ptr_out = $ptr, $dst = $src"),
            [("ptr_out".to_string(), "ptr".to_string()), ("dst".to_string(), "src".to_string())]
        );
        assert!(constraint_ties("@earlyclobber $dst").is_empty());
    }

    #[test]
    fn ag_addressing_maps_every_form() {
        assert_eq!(ag_addressing("AIE2_ag_all__ag_idx_imm"), Some("IndexedImm"));
        assert_eq!(ag_addressing("AIE2_agb_standalone__agb_nospill__ag_pstm_2d"), Some("TwoDim"));
        assert_eq!(ag_addressing("AIE2_ag_standalone__ag_pstm_sp_imm"), Some("SpPostModifyImm"));
        assert_eq!(ag_addressing("AIE2_ag_all"), None);
    }

    #[test]
    #[should_panic(expected = "unknown address-generator form")]
    fn ag_addressing_refuses_an_unknown_form() {
        ag_addressing("AIE2_ag_all__ag_pstm_4d");
    }

    #[test]
    fn stream_form_reads_the_variant_suffix() {
        assert_eq!(stream_form("AIE2_mv_scl2ms_nb_doTlast_reg"), Some((true, Some("FromRegister"))));
        assert_eq!(stream_form("AIE2_mv_ph2ms_b_doTlast"), Some((false, Some("Set"))));
        assert_eq!(stream_form("AIE2_mv_cph2ms_nb_noTlast"), Some((true, Some("Clear"))));
        assert_eq!(stream_form("AIE2_mv_ss2scl_nb"), Some((true, None)));
        assert_eq!(stream_form("AIE2_mv_ss2scl_b"), Some((false, None)));
        assert_eq!(stream_form("AIE2_mv_ss2scl_inst_lda"), None);
        assert_eq!(stream_form("vmac_cm_core_dense"), None);
    }

    #[test]
    fn derivation_order_puts_bases_first() {
        let derives = |a: &str, b: &str| matches!((a, b), ("C", "B") | ("C", "A") | ("B", "A"));
        let mut v = vec!["C".to_string(), "Z".to_string(), "A".to_string(), "B".to_string()];
        order_by_derivation(&mut v, derives);
        assert_eq!(v, ["A", "Z", "B", "C"]);
    }

    #[test]
    fn format_args_reassemble_from_literal_bits() {
        let bits = vec![
            ("mdm".to_string(), 0, 3),
            ("op".to_string(), 1, 7),
            ("op".to_string(), 0, 6),
            ("md1".to_string(), 0, 2),
        ];
        let lit = |i: usize| Some([3, 7].contains(&i));
        assert_eq!(
            format_arg_values(&bits, lit),
            [("mdm".to_string(), 1), ("op".to_string(), 2), ("md1".to_string(), 0)]
        );
        let partial = |i: usize| if i == 6 { None } else { Some(true) };
        assert_eq!(format_arg_values(&bits, partial), [("mdm".to_string(), 1), ("md1".to_string(), 1)]);
    }
}
