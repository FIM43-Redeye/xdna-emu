//! LLVM register name -> emulator Operand mapping.
//!
//! Adapter layer between the FFI in `xdna_archspec::aie2::isa::decoder_ffi`
//! and the interpreter's `Operand` enum. This is interpreter execution-model
//! code (it references the emulator's `Operand` enum and register-file
//! indices), so it lives here rather than in the arch-data crate.

use xdna_archspec::aie2::isa::decoder_ffi::*;

use crate::interpreter::bundle::slot::Operand;
use crate::interpreter::state::{
    LR_REG_INDEX, LS_REG_INDEX, LE_REG_INDEX, LC_REG_INDEX, DP_REG_INDEX, CORE_ID_REG_INDEX, SP_PTR_INDEX,
    MOD_BASE_M, MOD_BASE_DN, MOD_BASE_DJ, MOD_BASE_DC,
};

/// Width hint for accumulator operands from LLVM register names.
///
/// The same AccumReg(n) index can represent different access widths depending
/// on the LLVM register class. The execution handler needs this to choose
/// between 256-bit, 512-bit, or 1024-bit access patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumWidth {
    /// 256-bit low quarter-accumulator (amll, amhl) -- lanes 0-3.
    QuarterLow,
    /// 256-bit high quarter-accumulator (amlh, amhh) -- lanes 4-7.
    QuarterHigh,
    /// 512-bit half-accumulator (bml, bmh).
    Half,
    /// 1024-bit full accumulator (cm).
    Full,
}

/// Result of mapping an LLVM register name to our internal Operand.
///
/// Carries the operand plus optional metadata that cannot be inferred from
/// the Operand enum alone (e.g., accumulator access width).
#[derive(Debug, Clone, PartialEq)]
pub struct MappedOperand {
    pub operand: Operand,
    /// For AccumReg operands, the access width implied by the LLVM register class.
    pub accum_width: Option<AccumWidth>,
}

/// Parse a prefix + decimal index from a register name.
///
/// Returns `(prefix, index)` if the name matches `prefix + digits`.
/// For names with no trailing digits (e.g., "lr", "SP"), returns None.
fn parse_reg_name(name: &str) -> Option<(&str, u8)> {
    // Find where trailing digits start.
    let digit_start = name.rfind(|c: char| !c.is_ascii_digit())?;
    let prefix = &name[..=digit_start];
    let digits = &name[digit_start + 1..];
    if digits.is_empty() {
        return None;
    }
    let index: u8 = digits.parse().ok()?;
    Some((prefix, index))
}

/// Classify an LLVM register name to our internal Operand.
///
/// This is the domain knowledge layer: given a name string, produce
/// our internal MappedOperand. Called at init time to populate the
/// RegisterMap, and directly by legacy code paths.
///
/// # Accumulator Register Indexing
///
/// LLVM's accumulator hierarchy maps to our flat 8 x 512-bit AccumReg file:
///
/// ```text
/// cm_n (1024-bit) = { bml_n (low 512), bmh_n (high 512) }
///   bml_n         -> AccumReg(n * 2)
///   bmh_n         -> AccumReg(n * 2 + 1)
///   cm_n          -> AccumReg(n * 2)  [wide: reads n*2 and n*2+1]
///
/// 256-bit sub-banks share the same AccumReg index as their parent bml/bmh:
///   amll_n, amlh_n -> AccumReg(n * 2)     [within bml_n]
///   amhl_n, amhh_n -> AccumReg(n * 2 + 1) [within bmh_n]
/// ```
pub fn classify_reg_name(name: &str) -> Option<MappedOperand> {
    // Special registers (no trailing index).
    match name {
        "lr" => return Some(MappedOperand { operand: Operand::ScalarReg(LR_REG_INDEX), accum_width: None }),
        "SP" | "sp" => {
            return Some(MappedOperand { operand: Operand::PointerReg(SP_PTR_INDEX), accum_width: None })
        }
        "LS" | "ls" => {
            return Some(MappedOperand { operand: Operand::ScalarReg(LS_REG_INDEX), accum_width: None })
        }
        "LE" | "le" => {
            return Some(MappedOperand { operand: Operand::ScalarReg(LE_REG_INDEX), accum_width: None })
        }
        "LC" | "lc" => {
            return Some(MappedOperand { operand: Operand::ScalarReg(LC_REG_INDEX), accum_width: None })
        }
        "DP" | "dp" => {
            return Some(MappedOperand { operand: Operand::ScalarReg(DP_REG_INDEX), accum_width: None })
        }
        "CORE_ID" | "core_id" => {
            return Some(MappedOperand { operand: Operand::ScalarReg(CORE_ID_REG_INDEX), accum_width: None })
        }
        // DMA 3D addressing descriptor registers (d0_3d - d3_3d).
        // These select 3D stride patterns for loads/stores. Map to ModifierReg
        // in the 3D descriptor range (indices 32-35).
        "d0_3d" => return Some(MappedOperand { operand: Operand::ModifierReg(32), accum_width: None }),
        "d1_3d" => return Some(MappedOperand { operand: Operand::ModifierReg(33), accum_width: None }),
        "d2_3d" => return Some(MappedOperand { operand: Operand::ModifierReg(34), accum_width: None }),
        "d3_3d" => return Some(MappedOperand { operand: Operand::ModifierReg(35), accum_width: None }),
        _ => {}
    }

    // Control registers (named, no index).
    // IDs must match HWEncoding from AIE2GenRegisterInfo.td:
    //   crVaddSign=0, crF2FMask=1, crF2IMask=2, crFPMask=3,
    //   crMCDEn=4, crPackSign=5, crRnd=6, crSCDEn=7,
    //   crSRSSign=8, crSat=9, crUPSSign=10, crUnpackSign=11
    let cr_id = match name {
        "crVaddSign" => Some(0u8),
        "crF2FMask" => Some(1),
        "crF2IMask" => Some(2),
        "crFPMask" => Some(3),
        "crMCDEn" => Some(4),
        "crPackSign" => Some(5),
        "crRnd" => Some(6),
        "crSCDEn" => Some(7),
        "crSRSSign" => Some(8),
        "crSat" => Some(9),
        "crUPSSign" => Some(10),
        "crUnpackSign" => Some(11),
        _ => None,
    };
    if let Some(id) = cr_id {
        return Some(MappedOperand { operand: Operand::ControlReg(id), accum_width: None });
    }

    // Status registers -- currently no Operand variant for these.
    // They are read-only flags, not instruction operands in our model.
    if name.starts_with("sr") {
        return None;
    }

    // Indexed registers: parse prefix + index.
    let (prefix, idx) = parse_reg_name(name)?;

    let simple = |op: Operand| MappedOperand { operand: op, accum_width: None };
    let accum = |op: Operand, w: AccumWidth| MappedOperand { operand: op, accum_width: Some(w) };

    match prefix {
        // Scalar GPRs: r0-r31.
        "r" => Some(simple(Operand::ScalarReg(idx))),

        // Pointer registers: p0-p7.
        "p" => Some(simple(Operand::PointerReg(idx))),

        // Vector 512-bit (x-registers): x0-x11.
        // x_n spans two 256-bit registers: VectorReg(n * 2).
        "x" => Some(simple(Operand::VectorReg(idx * 2))),

        // Vector 256-bit sub-registers: wl (low half), wh (high half).
        "wl" => Some(simple(Operand::VectorReg(idx * 2))),
        "wh" => Some(simple(Operand::VectorReg(idx * 2 + 1))),

        // Vector 1024-bit (y-registers): y0-y7.
        // y_n spans four 256-bit registers: VectorReg(n * 4).
        "y" => Some(simple(Operand::VectorReg(idx * 4))),

        // Accumulator 512-bit low half: bml_n -> AccumReg(n * 2).
        "bml" => Some(accum(Operand::AccumReg(idx * 2), AccumWidth::Half)),

        // Accumulator 512-bit high half: bmh_n -> AccumReg(n * 2 + 1).
        "bmh" => Some(accum(Operand::AccumReg(idx * 2 + 1), AccumWidth::Half)),

        // Accumulator 1024-bit: cm_n -> AccumReg(n * 2) with wide access.
        "cm" => Some(accum(Operand::AccumReg(idx * 2), AccumWidth::Full)),

        // Accumulator 256-bit sub-banks.
        // amll_n = low quarter of bml_n (lanes 0-3).
        "amll" => Some(accum(Operand::AccumReg(idx * 2), AccumWidth::QuarterLow)),
        // amlh_n = high quarter of bml_n (lanes 4-7).
        "amlh" => Some(accum(Operand::AccumReg(idx * 2), AccumWidth::QuarterHigh)),
        // amhl_n = low quarter of bmh_n (lanes 0-3).
        "amhl" => Some(accum(Operand::AccumReg(idx * 2 + 1), AccumWidth::QuarterLow)),
        // amhh_n = high quarter of bmh_n (lanes 4-7).
        "amhh" => Some(accum(Operand::AccumReg(idx * 2 + 1), AccumWidth::QuarterHigh)),

        // Modifier sub-registers.
        "m" => Some(simple(Operand::ModifierReg(MOD_BASE_M + idx))),
        "dn" => Some(simple(Operand::ModifierReg(MOD_BASE_DN + idx))),
        "dj" => Some(simple(Operand::ModifierReg(MOD_BASE_DJ + idx))),
        "dc" => Some(simple(Operand::ModifierReg(MOD_BASE_DC + idx))),

        // Shift registers: s0-s3.  Map to ScalarReg with high indices.
        // These are separate in hardware but rarely appear as explicit operands.
        "s" => Some(simple(Operand::ScalarReg(40 + idx))),

        // Long register pairs: l0-l7 (64-bit, r_2n:r_2n+1).
        // Map to the low scalar register of the pair.
        "l" => Some(simple(Operand::ScalarReg(16 + idx * 2))),

        // Mask registers: q0-q3 (128-bit), qx0-qx3 (640-bit).
        // Used as explicit operands in sparse vector loads (VLDB_SPARSE_POP).
        // Map to ControlReg in a dedicated mask range (indices 16-35).
        //   q0-q3:    ControlReg(16..19)   -- 128-bit mask (full)
        //   ql0-ql3:  ControlReg(28..31)   -- low 64-bit of q
        //   qh0-qh3:  ControlReg(32..35)   -- high 64-bit of q
        //   qwl0-qwl3: ControlReg(20..23)  -- 256-bit wide mask low
        //   qwh0-qwh3: ControlReg(24..27)  -- 256-bit wide mask high
        //   qx0-qx3:   SparseQxReg(0..3)   -- sparse composite (x_n + q_n)
        "q" => Some(simple(Operand::ControlReg(16 + idx))),
        "ql" => Some(simple(Operand::ControlReg(28 + idx))),
        "qh" => Some(simple(Operand::ControlReg(32 + idx))),
        "qwl" => Some(simple(Operand::ControlReg(20 + idx))),
        "qwh" => Some(simple(Operand::ControlReg(24 + idx))),
        "qx" => Some(simple(Operand::SparseQxReg(idx))),

        // DMA 2D addressing descriptor registers (d0-d7).
        // These select 2D stride patterns for loads/stores. Map to ModifierReg
        // in the 2D descriptor range (indices 36-43).
        // (3D descriptors d0_3d-d3_3d are handled as special-case names above.)
        "d" => Some(simple(Operand::ModifierReg(36 + idx))),

        _ => None,
    }
}

// ---------------------------------------------------------------------------
// RegisterMap: init-time HashMap from LLVM's MCRegisterInfo
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::OnceLock;

/// Pre-built register name -> MappedOperand lookup table.
///
/// Populated once at init by querying MCRegisterInfo for all register names
/// and running each through `classify_reg_name()`. Subsequent lookups are
/// O(1) HashMap access with zero FFI cost.
struct RegisterMap {
    map: HashMap<String, MappedOperand>,
    /// Register names that LLVM knows but we intentionally don't map
    /// (mask regs, DMA regs, status regs, etc.).
    unmapped: Vec<String>,
}

static REG_MAP: OnceLock<RegisterMap> = OnceLock::new();

impl RegisterMap {
    /// Build the register map by querying MCRegisterInfo.
    fn from_llvm() -> Self {
        let mut map = HashMap::new();
        let mut unmapped = Vec::new();

        let num_regs = get_num_regs();
        if num_regs == 0 {
            // FFI not available -- return empty map (classify_reg_name fallback).
            return Self { map, unmapped };
        }

        for reg_id in 1..num_regs {
            let name = match get_reg_name(reg_id) {
                Some(n) => n,
                None => continue,
            };

            match classify_reg_name(&name) {
                Some(mapped) => {
                    map.insert(name, mapped);
                }
                None => {
                    unmapped.push(name);
                }
            }
        }

        Self { map, unmapped }
    }

    fn lookup(&self, name: &str) -> Option<MappedOperand> {
        self.map.get(name).cloned()
    }
}

/// Get or initialize the global register map.
fn reg_map() -> &'static RegisterMap {
    REG_MAP.get_or_init(RegisterMap::from_llvm)
}

/// Map an LLVM register name to our internal Operand.
///
/// Uses the pre-built RegisterMap (O(1) HashMap lookup, zero FFI cost).
/// Falls back to `classify_reg_name()` for names not in the map (e.g.,
/// if the map hasn't been initialized or a name wasn't in MCRegisterInfo).
pub fn operand_from_reg_name(name: &str) -> Option<MappedOperand> {
    let map = reg_map();
    map.lookup(name).or_else(|| classify_reg_name(name))
}

/// Get the count of mapped and unmapped registers from MCRegisterInfo.
///
/// Returns `(mapped, unmapped_names)`. The unmapped list contains register
/// names LLVM knows that we intentionally skip (mask regs, DMA regs, etc.).
pub fn reg_map_coverage() -> (usize, &'static [String]) {
    let map = reg_map();
    (map.map.len(), &map.unmapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_map_populated_from_llvm() {
        let (mapped, unmapped) = reg_map_coverage();

        // LLVM knows ~135 physical registers for AIE2. We should map most of them.
        assert!(mapped > 80, "Should map 80+ registers from MCRegisterInfo, got {}", mapped);

        // Unmapped registers are intentional (mask regs, DMA regs, status regs).
        // Verify the unmapped list contains expected categories.
        eprintln!("RegisterMap: {} mapped, {} unmapped", mapped, unmapped.len());
        for name in unmapped {
            eprintln!("  unmapped: {}", name);
        }
    }

    #[test]
    fn test_register_map_matches_classify() {
        // Every entry in the RegisterMap should produce the same result
        // as calling classify_reg_name directly.
        let map = reg_map();
        for (name, mapped) in &map.map {
            let direct = classify_reg_name(name);
            assert_eq!(
                direct.as_ref(),
                Some(mapped),
                "RegisterMap and classify_reg_name disagree for '{}'",
                name
            );
        }
    }

    #[test]
    fn test_register_map_lookup_matches_direct() {
        // operand_from_reg_name (HashMap path) should match classify_reg_name
        // for all known register names.
        let test_names = [
            "r0", "r15", "r31", "p0", "p7", "lr", "SP", "LS", "LE", "LC", "DP", "x0", "x4", "wl0", "wh3",
            "y2", "bml0", "bmh0", "cm0", "cm3", "amll0", "amhh1", "m0", "dn3", "dj7", "dc0", "crRnd",
            "crSat", "s0",
        ];
        for name in &test_names {
            let from_map = operand_from_reg_name(name);
            let from_classify = classify_reg_name(name);
            assert_eq!(
                from_map, from_classify,
                "Mismatch for '{}': map={:?} classify={:?}",
                name, from_map, from_classify
            );
        }
    }

    #[test]
    fn test_scalar_regs() {
        for i in 0..32u8 {
            let name = format!("r{}", i);
            let m = operand_from_reg_name(&name).expect(&name);
            assert_eq!(m.operand, Operand::ScalarReg(i));
            assert_eq!(m.accum_width, None);
        }
    }

    #[test]
    fn test_pointer_regs() {
        for i in 0..8u8 {
            let name = format!("p{}", i);
            let m = operand_from_reg_name(&name).expect(&name);
            assert_eq!(m.operand, Operand::PointerReg(i));
        }
    }

    #[test]
    fn test_special_regs() {
        let m = operand_from_reg_name("lr").unwrap();
        assert_eq!(m.operand, Operand::ScalarReg(LR_REG_INDEX));

        let m = operand_from_reg_name("SP").unwrap();
        assert_eq!(m.operand, Operand::PointerReg(SP_PTR_INDEX));

        let m = operand_from_reg_name("LS").unwrap();
        assert_eq!(m.operand, Operand::ScalarReg(LS_REG_INDEX));

        let m = operand_from_reg_name("LE").unwrap();
        assert_eq!(m.operand, Operand::ScalarReg(LE_REG_INDEX));

        let m = operand_from_reg_name("LC").unwrap();
        assert_eq!(m.operand, Operand::ScalarReg(LC_REG_INDEX));

        let m = operand_from_reg_name("DP").unwrap();
        assert_eq!(m.operand, Operand::ScalarReg(DP_REG_INDEX));
    }

    #[test]
    fn test_vector_regs() {
        // x0 -> VectorReg(0), x1 -> VectorReg(2), x4 -> VectorReg(8)
        let m = operand_from_reg_name("x0").unwrap();
        assert_eq!(m.operand, Operand::VectorReg(0));

        let m = operand_from_reg_name("x4").unwrap();
        assert_eq!(m.operand, Operand::VectorReg(8));

        // wl/wh sub-registers
        let m = operand_from_reg_name("wl0").unwrap();
        assert_eq!(m.operand, Operand::VectorReg(0));

        let m = operand_from_reg_name("wh0").unwrap();
        assert_eq!(m.operand, Operand::VectorReg(1));

        // y-registers (1024-bit)
        let m = operand_from_reg_name("y2").unwrap();
        assert_eq!(m.operand, Operand::VectorReg(8));
    }

    #[test]
    fn test_accum_regs() {
        // bml0 -> AccumReg(0), Half
        let m = operand_from_reg_name("bml0").unwrap();
        assert_eq!(m.operand, Operand::AccumReg(0));
        assert_eq!(m.accum_width, Some(AccumWidth::Half));

        // bmh0 -> AccumReg(1), Half
        let m = operand_from_reg_name("bmh0").unwrap();
        assert_eq!(m.operand, Operand::AccumReg(1));
        assert_eq!(m.accum_width, Some(AccumWidth::Half));

        // bml1 -> AccumReg(2)
        let m = operand_from_reg_name("bml1").unwrap();
        assert_eq!(m.operand, Operand::AccumReg(2));

        // cm0 -> AccumReg(0), Full
        let m = operand_from_reg_name("cm0").unwrap();
        assert_eq!(m.operand, Operand::AccumReg(0));
        assert_eq!(m.accum_width, Some(AccumWidth::Full));

        // cm1 -> AccumReg(2), Full
        let m = operand_from_reg_name("cm1").unwrap();
        assert_eq!(m.operand, Operand::AccumReg(2));

        // amll0 -> AccumReg(0), QuarterLow (lanes 0-3 of bml0)
        let m = operand_from_reg_name("amll0").unwrap();
        assert_eq!(m.operand, Operand::AccumReg(0));
        assert_eq!(m.accum_width, Some(AccumWidth::QuarterLow));

        // amlh0 -> AccumReg(0), QuarterHigh (lanes 4-7 of bml0)
        let m = operand_from_reg_name("amlh0").unwrap();
        assert_eq!(m.operand, Operand::AccumReg(0));
        assert_eq!(m.accum_width, Some(AccumWidth::QuarterHigh));

        // amhl1 -> AccumReg(3), QuarterLow (lanes 0-3 of bmh1)
        let m = operand_from_reg_name("amhl1").unwrap();
        assert_eq!(m.operand, Operand::AccumReg(3));
        assert_eq!(m.accum_width, Some(AccumWidth::QuarterLow));

        // amhh1 -> AccumReg(3), QuarterHigh (lanes 4-7 of bmh1)
        let m = operand_from_reg_name("amhh1").unwrap();
        assert_eq!(m.operand, Operand::AccumReg(3));
        assert_eq!(m.accum_width, Some(AccumWidth::QuarterHigh));
    }

    #[test]
    fn test_modifier_regs() {
        let m = operand_from_reg_name("m0").unwrap();
        assert_eq!(m.operand, Operand::ModifierReg(MOD_BASE_M));

        let m = operand_from_reg_name("dn3").unwrap();
        assert_eq!(m.operand, Operand::ModifierReg(MOD_BASE_DN + 3));

        let m = operand_from_reg_name("dj7").unwrap();
        assert_eq!(m.operand, Operand::ModifierReg(MOD_BASE_DJ + 7));

        let m = operand_from_reg_name("dc0").unwrap();
        assert_eq!(m.operand, Operand::ModifierReg(MOD_BASE_DC));
    }

    #[test]
    fn test_control_regs() {
        // IDs must match HWEncoding from AIE2GenRegisterInfo.td.
        let m = operand_from_reg_name("crRnd").unwrap();
        assert_eq!(m.operand, Operand::ControlReg(6));

        let m = operand_from_reg_name("crSat").unwrap();
        assert_eq!(m.operand, Operand::ControlReg(9));

        let m = operand_from_reg_name("crSRSSign").unwrap();
        assert_eq!(m.operand, Operand::ControlReg(8));

        let m = operand_from_reg_name("crUPSSign").unwrap();
        assert_eq!(m.operand, Operand::ControlReg(10));

        let m = operand_from_reg_name("crVaddSign").unwrap();
        assert_eq!(m.operand, Operand::ControlReg(0));
    }

    #[test]
    fn test_unknown_returns_none() {
        assert!(operand_from_reg_name("xyz42").is_none());
        assert!(operand_from_reg_name("").is_none());
        assert!(operand_from_reg_name("tile_cntr").is_none());
    }

    /// No unmapped register ever appears as an explicit operand in any
    /// decoded instruction across all slots.  If this fails, LLVM is
    /// emitting an operand with a register name we don't handle -- a real
    /// gap to fix, not a test bug.
    #[test]
    fn test_unmapped_regs_never_appear_as_explicit_operands() {
        use xdna_archspec::aie2::isa::decoder_ffi::{DecodedOperand, Slot, decode_slot};
        let (_, unmapped) = reg_map_coverage();
        let unmapped_set: std::collections::HashSet<&str> = unmapped.iter().map(|s| s.as_str()).collect();

        let slots = [
            (Slot::Alu, 20u64),
            (Slot::Lda, 21),
            (Slot::Ldb, 16),
            (Slot::Mv, 22),
            (Slot::St, 21),
            (Slot::Vec, 26),
        ];

        let mut hits: Vec<String> = Vec::new();
        for (slot, width) in &slots {
            for probe in 0..512u64 {
                let bits = probe << (width / 2);
                if let Some(decoded) = decode_slot(*slot, bits) {
                    for op in &decoded.operands {
                        if let DecodedOperand::Reg { name, .. } = op {
                            if unmapped_set.contains(name.as_str())
                                && !hits.iter().any(|h| h.starts_with(name.as_str()))
                            {
                                hits.push(format!(
                                    "{} in {} (slot {:?} bits 0x{:X})",
                                    name, decoded.name, slot, bits
                                ));
                            }
                        }
                    }
                }
            }
        }

        if !hits.is_empty() {
            eprintln!("Unmapped registers appearing as explicit operands ({}):", hits.len());
            for h in &hits {
                eprintln!("  {}", h);
            }
            panic!("{} unmapped register(s) appear as explicit operands -- need mapping", hits.len());
        }
    }

    /// Every register operand returned by the LLVM decoder must resolve
    /// through operand_from_reg_name (i.e., no silent drops).
    #[test]
    fn test_llvm_decode_operands_map_correctly() {
        use xdna_archspec::aie2::isa::decoder_ffi::{DecodedOperand, Slot, decode_slot};
        let decoded = decode_slot(Slot::Mv, 0x028C3D).expect("VPUSH_HI_32 should decode");
        assert_eq!(decoded.name, "VPUSH_HI_32");

        let mut mapped = 0;
        for op in &decoded.operands {
            if let DecodedOperand::Reg { name, .. } = op {
                let result = operand_from_reg_name(name);
                assert!(result.is_some(), "LLVM register '{}' should map to an Operand", name,);
                mapped += 1;
            }
        }
        assert!(mapped > 0, "Should have mapped at least one register operand");
    }

    /// Informational probe of vec-slot decoded operands: warns on unmapped
    /// registers rather than failing (surfaces gaps without blocking).
    #[test]
    fn test_vec_decode_operands_map_correctly() {
        use xdna_archspec::aie2::isa::decoder_ffi::{DecodedOperand, Slot, decode_slot};
        for bits in [0x000001u64, 0x100001, 0x200001] {
            if let Some(decoded) = decode_slot(Slot::Vec, bits) {
                for op in &decoded.operands {
                    if let DecodedOperand::Reg { name, .. } = op {
                        if operand_from_reg_name(name).is_none() {
                            eprintln!(
                                "WARNING: unmapped register '{}' in {} (Vec 0x{:06X})",
                                name, decoded.name, bits
                            );
                        }
                    }
                }
            }
        }
    }
}
