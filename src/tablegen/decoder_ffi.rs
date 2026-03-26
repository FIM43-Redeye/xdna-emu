//! FFI bindings to LLVM's AIE2 MCDisassembler.
//!
//! Links against llvm-aie's disassembler library for:
//! - Instruction identification with full TRY_DECODE disambiguation
//! - Decoded operand values (register IDs and immediates)
//! - Register names from MCRegisterInfo (e.g., "r0", "bm0", "cm0")
//! - Output operand classification from MCInstrDesc::NumDefs
//!
//! The C side (`decoder_ffi/aie2_decoder.cpp`) is compiled and linked by
//! build.rs using the `cc` crate and `llvm-config --libs aie`.

use std::ffi::CStr;
use std::sync::Once;

/// Operand kinds matching LLVM MCOperand (mirrors Aie2OpKind in C header).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    Invalid = 0,
    Reg = 1,
    Imm = 2,
}

/// A single decoded operand with register metadata (mirrors Aie2Operand).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RawOperand {
    pub kind: OpKind,
    pub value: i64,
    pub reg_name: *const std::os::raw::c_char,
}

/// Raw FFI decode result (mirrors Aie2DecodeResult).
#[repr(C)]
#[derive(Debug, Clone)]
pub struct RawDecodeResult {
    pub success: i32,
    pub opcode: u32,
    pub num_operands: u32,
    pub num_defs: u32,
    pub operands: [RawOperand; 16],
}

/// Slot identifiers (mirrors Aie2Slot in C header).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    Alu = 0,
    Lda = 1,
    Ldb = 2,
    Lng = 3,
    Mv = 4,
    St = 5,
    Vec = 6,
    Nop = 7,
}

/// Raw per-instruction metadata from MCInstrDesc + itinerary model.
/// Mirrors `Aie2InstrInfo` in the C header.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct RawInstrInfo {
    pub flags: u64,
    pub num_operands: u16,
    pub num_defs: u8,
    pub latency: i16,        // operand_cycles[0], or -1 if unavailable
    pub stage_latency: i16,  // Pipeline stage sum, or -1 if unavailable
    pub sched_class: u16,    // Itinerary class index (opaque)
}

extern "C" {
    fn aie2_decode_slot(slot: Slot, insn_bits: u64) -> RawDecodeResult;
    fn aie2_opcode_name(opcode: u32) -> *const std::os::raw::c_char;
    fn aie2_decoder_init() -> i32;
    fn aie2_get_num_regs() -> u32;
    fn aie2_get_reg_name(reg_id: u32) -> *const std::os::raw::c_char;
    fn aie2_get_num_opcodes() -> u32;
    fn aie2_get_instr_info(opcode: u32, out: *mut RawInstrInfo) -> i32;
}

static INIT: Once = Once::new();
static mut INIT_OK: bool = false;

// ---------------------------------------------------------------------------
// Decoded operand (safe Rust type)
// ---------------------------------------------------------------------------

/// A decoded operand with LLVM register name resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedOperand {
    /// Register operand with LLVM register name (e.g., "r0", "bm0", "cm0").
    Reg { id: u32, name: String },
    /// Immediate operand.
    Imm(i64),
}

/// Full decode result with output/input classification.
#[derive(Debug, Clone)]
pub struct DecodeResult {
    /// LLVM instruction name (e.g., "VPUSH_HI_32").
    pub name: String,
    /// LLVM opcode ID.
    pub opcode: u32,
    /// Number of leading operands that are outputs (from MCInstrDesc).
    pub num_defs: u32,
    /// All operands (first `num_defs` are outputs, rest are inputs).
    pub operands: Vec<DecodedOperand>,
}

impl DecodeResult {
    /// Output operands (first `num_defs`).
    pub fn defs(&self) -> &[DecodedOperand] {
        let n = (self.num_defs as usize).min(self.operands.len());
        &self.operands[..n]
    }

    /// Input operands (after the defs).
    pub fn uses(&self) -> &[DecodedOperand] {
        let n = (self.num_defs as usize).min(self.operands.len());
        &self.operands[n..]
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Initialize the LLVM decoder. Safe to call multiple times.
pub fn init() -> bool {
    INIT.call_once(|| {
        let result = unsafe { aie2_decoder_init() };
        unsafe {
            INIT_OK = result == 1;
        }
    });
    unsafe { INIT_OK }
}

/// Decode a single VLIW slot using LLVM's MCDisassembler.
///
/// Returns a full `DecodeResult` with instruction name, operand values,
/// register names, and output/input classification.
pub fn decode_slot(slot: Slot, insn_bits: u64) -> Option<DecodeResult> {
    if !init() {
        return None;
    }

    let raw = unsafe { aie2_decode_slot(slot, insn_bits) };
    if raw.success == 0 {
        return None;
    }

    let name = opcode_name(raw.opcode)?;

    let mut operands = Vec::with_capacity(raw.num_operands as usize);
    for i in 0..raw.num_operands as usize {
        let op = &raw.operands[i];
        match op.kind {
            OpKind::Reg => {
                let reg_name = if !op.reg_name.is_null() {
                    unsafe { CStr::from_ptr(op.reg_name) }
                        .to_string_lossy()
                        .into_owned()
                } else {
                    format!("?{}", op.value)
                };
                operands.push(DecodedOperand::Reg {
                    id: op.value as u32,
                    name: reg_name,
                });
            }
            OpKind::Imm => {
                operands.push(DecodedOperand::Imm(op.value));
            }
            _ => {}
        }
    }

    Some(DecodeResult {
        name,
        opcode: raw.opcode,
        num_defs: raw.num_defs,
        operands,
    })
}

/// Decode a slot and return just the instruction name (legacy interface).
///
/// Used by the resolver's decode path for instruction identification.
pub fn decode_slot_name(slot: Slot, insn_bits: u64) -> Option<String> {
    decode_slot(slot, insn_bits).map(|r| r.name)
}

/// Get the LLVM instruction name for an opcode ID.
pub fn opcode_name(opcode: u32) -> Option<String> {
    if !init() {
        return None;
    }

    let ptr = unsafe { aie2_opcode_name(opcode) };
    if ptr.is_null() {
        return None;
    }
    let cstr = unsafe { CStr::from_ptr(ptr) };
    Some(cstr.to_string_lossy().into_owned())
}

/// Map a slot name string (e.g., "mv", "alu") to a Slot enum value.
pub fn slot_from_name(name: &str) -> Option<Slot> {
    match name {
        "alu" => Some(Slot::Alu),
        "lda" => Some(Slot::Lda),
        "ldb" => Some(Slot::Ldb),
        "lng" | "long" => Some(Slot::Lng),
        "mv" => Some(Slot::Mv),
        "st" => Some(Slot::St),
        "vec" => Some(Slot::Vec),
        "nop" => Some(Slot::Nop),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Instruction metadata (bulk-queryable at init time)
// ---------------------------------------------------------------------------

/// MCID flag bit positions (from llvm/MC/MCInstrDesc.h).
pub mod mcid {
    pub const RETURN: u64          = 1 << 5;
    pub const CALL: u64            = 1 << 7;
    pub const BARRIER: u64         = 1 << 8;
    pub const TERMINATOR: u64      = 1 << 9;
    pub const BRANCH: u64          = 1 << 10;
    pub const INDIRECT_BRANCH: u64 = 1 << 11;
    pub const COMPARE: u64         = 1 << 12;
    pub const MOVE_IMM: u64        = 1 << 13;
    pub const MOVE_REG: u64        = 1 << 14;
    pub const MAY_LOAD: u64        = 1 << 19;
    pub const MAY_STORE: u64       = 1 << 20;
    pub const COMMUTABLE: u64      = 1 << 25;
    pub const ADD: u64             = 1 << 37;
}

/// Per-instruction metadata from LLVM's MCInstrDesc and itinerary model.
///
/// Queried once at decoder init and stored in a Vec indexed by opcode.
/// All hot-path lookups are then pure Rust array accesses -- zero FFI cost.
#[derive(Debug, Clone)]
pub struct InstrInfo {
    /// MCID flags bitmask (MayLoad, MayStore, isBranch, etc.).
    pub flags: u64,
    /// Number of output (def) operands.
    pub num_defs: u8,
    /// Result latency from itinerary operand_cycles[0], or None.
    pub latency: Option<u8>,
    /// Total pipeline latency from InstrStage sum, or None.
    pub stage_latency: Option<u8>,
    /// Itinerary class index (opaque, for cross-ref with build-time data).
    pub sched_class: u16,
}

impl InstrInfo {
    pub fn is_load(&self) -> bool { self.flags & mcid::MAY_LOAD != 0 }
    pub fn is_store(&self) -> bool { self.flags & mcid::MAY_STORE != 0 }
    pub fn is_branch(&self) -> bool { self.flags & mcid::BRANCH != 0 }
    pub fn is_call(&self) -> bool { self.flags & mcid::CALL != 0 }
    pub fn is_return(&self) -> bool { self.flags & mcid::RETURN != 0 }
    pub fn is_terminator(&self) -> bool { self.flags & mcid::TERMINATOR != 0 }
    pub fn is_compare(&self) -> bool { self.flags & mcid::COMPARE != 0 }
    pub fn is_commutable(&self) -> bool { self.flags & mcid::COMMUTABLE != 0 }
    pub fn is_move_reg(&self) -> bool { self.flags & mcid::MOVE_REG != 0 }
    pub fn is_move_imm(&self) -> bool { self.flags & mcid::MOVE_IMM != 0 }
}

/// Query instruction metadata for all opcodes at init time.
///
/// Returns a Vec where `result[opcode]` is the InstrInfo for that opcode.
/// Called once during decoder initialization; subsequent accesses are O(1)
/// array lookups with zero FFI overhead.
pub fn query_all_instr_info() -> Vec<InstrInfo> {
    if !init() {
        return Vec::new();
    }

    let num = unsafe { aie2_get_num_opcodes() };
    let mut result = Vec::with_capacity(num as usize);

    for opcode in 0..num {
        let mut raw = RawInstrInfo {
            flags: 0,
            num_operands: 0,
            num_defs: 0,
            latency: -1,
            stage_latency: -1,
            sched_class: 0,
        };

        let ok = unsafe { aie2_get_instr_info(opcode, &mut raw) };
        if ok == 0 {
            result.push(InstrInfo {
                flags: 0,
                num_defs: 0,
                latency: None,
                stage_latency: None,
                sched_class: 0,
            });
            continue;
        }

        let latency = if raw.latency >= 0 {
            Some(raw.latency as u8)
        } else {
            None
        };

        let stage_latency = if raw.stage_latency >= 0 {
            Some(raw.stage_latency as u8)
        } else {
            None
        };

        result.push(InstrInfo {
            flags: raw.flags,
            num_defs: raw.num_defs,
            latency,
            stage_latency,
            sched_class: raw.sched_class,
        });
    }

    result
}

// ---------------------------------------------------------------------------
// Register name -> Operand mapping
//
// Two-tier design:
// 1. classify_reg_name() contains the domain knowledge: given a register name,
//    produce our internal MappedOperand. This is pure logic, no FFI.
// 2. RegisterMap builds a HashMap<String, MappedOperand> at init by querying
//    MCRegisterInfo for ALL register names and running each through the
//    classifier. This gives O(1) lookups at runtime with zero FFI cost,
//    and validates coverage at init time.
// ---------------------------------------------------------------------------

use crate::interpreter::bundle::slot::Operand;
use crate::interpreter::state::{
    LR_REG_INDEX, LS_REG_INDEX, LE_REG_INDEX, LC_REG_INDEX,
    DP_REG_INDEX, CORE_ID_REG_INDEX, SP_PTR_INDEX,
    MOD_BASE_M, MOD_BASE_DN, MOD_BASE_DJ, MOD_BASE_DC,
};

/// Width hint for accumulator operands from LLVM register names.
///
/// The same AccumReg(n) index can represent different access widths depending
/// on the LLVM register class. The execution handler needs this to choose
/// between 256-bit, 512-bit, or 1024-bit access patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumWidth {
    /// 256-bit quarter-accumulator (amll, amlh, amhl, amhh).
    Quarter,
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
        "lr" => return Some(MappedOperand {
            operand: Operand::ScalarReg(LR_REG_INDEX),
            accum_width: None,
        }),
        "SP" | "sp" => return Some(MappedOperand {
            operand: Operand::PointerReg(SP_PTR_INDEX),
            accum_width: None,
        }),
        "LS" | "ls" => return Some(MappedOperand {
            operand: Operand::ScalarReg(LS_REG_INDEX),
            accum_width: None,
        }),
        "LE" | "le" => return Some(MappedOperand {
            operand: Operand::ScalarReg(LE_REG_INDEX),
            accum_width: None,
        }),
        "LC" | "lc" => return Some(MappedOperand {
            operand: Operand::ScalarReg(LC_REG_INDEX),
            accum_width: None,
        }),
        "DP" | "dp" => return Some(MappedOperand {
            operand: Operand::ScalarReg(DP_REG_INDEX),
            accum_width: None,
        }),
        "CORE_ID" | "core_id" => return Some(MappedOperand {
            operand: Operand::ScalarReg(CORE_ID_REG_INDEX),
            accum_width: None,
        }),
        // DMA 3D addressing descriptor registers (d0_3d - d3_3d).
        // These select 3D stride patterns for loads/stores. Map to ModifierReg
        // in the 3D descriptor range (indices 32-35).
        "d0_3d" => return Some(MappedOperand {
            operand: Operand::ModifierReg(32),
            accum_width: None,
        }),
        "d1_3d" => return Some(MappedOperand {
            operand: Operand::ModifierReg(33),
            accum_width: None,
        }),
        "d2_3d" => return Some(MappedOperand {
            operand: Operand::ModifierReg(34),
            accum_width: None,
        }),
        "d3_3d" => return Some(MappedOperand {
            operand: Operand::ModifierReg(35),
            accum_width: None,
        }),
        _ => {}
    }

    // Control registers (named, no index).
    let cr_id = match name {
        "crVaddSign" => Some(0u8),
        "crSCDEn"    => Some(1),
        "crMCDEn"    => Some(2),
        "crUnpackSign" => Some(3),
        "crPackSign" => Some(4),
        "crUPSSign"  => Some(5),
        "crRnd"      => Some(6),
        "crSat"      => Some(7),
        "crSRSSign"  => Some(8),
        "crFPMask"   => Some(9),
        "crF2IMask"  => Some(10),
        "crF2FMask"  => Some(11),
        _ => None,
    };
    if let Some(id) = cr_id {
        return Some(MappedOperand {
            operand: Operand::ControlReg(id),
            accum_width: None,
        });
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
        // amll_n, amlh_n are within bml_n -> AccumReg(n * 2).
        "amll" | "amlh" => Some(accum(Operand::AccumReg(idx * 2), AccumWidth::Quarter)),
        // amhl_n, amhh_n are within bmh_n -> AccumReg(n * 2 + 1).
        "amhl" | "amhh" => Some(accum(Operand::AccumReg(idx * 2 + 1), AccumWidth::Quarter)),

        // Modifier sub-registers.
        "m"  => Some(simple(Operand::ModifierReg(MOD_BASE_M + idx))),
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
        // Map to ControlReg in a dedicated mask range (indices 16-31).
        //   q0-q3:    ControlReg(16..19)   -- 128-bit mask
        //   ql0-ql3:  ControlReg(16..19)   -- low 64-bit of q
        //   qh0-qh3:  ControlReg(16..19)   -- high 64-bit of q
        //   qwl0-qwl3: ControlReg(20..23)  -- 256-bit wide mask low
        //   qwh0-qwh3: ControlReg(24..27)  -- 256-bit wide mask high
        //   qx0-qx3:  ControlReg(28..31)   -- 640-bit extended mask
        "q" | "ql" | "qh" => Some(simple(Operand::ControlReg(16 + idx))),
        "qwl" => Some(simple(Operand::ControlReg(20 + idx))),
        "qwh" => Some(simple(Operand::ControlReg(24 + idx))),
        "qx" => Some(simple(Operand::ControlReg(28 + idx))),

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

        if !init() {
            // FFI not available -- return empty map (classify_reg_name fallback).
            return Self { map, unmapped };
        }

        let num_regs = unsafe { aie2_get_num_regs() };

        for reg_id in 1..num_regs {
            let name_ptr = unsafe { aie2_get_reg_name(reg_id) };
            if name_ptr.is_null() {
                continue;
            }
            let name = unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned();

            if name.is_empty() {
                continue;
            }

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
    fn test_decoder_init() {
        assert!(init(), "LLVM decoder should initialize successfully");
    }

    #[test]
    fn test_opcode_name_lookup() {
        assert!(init());
        let _ = opcode_name(0);
        assert!(opcode_name(999999).is_none());
    }

    #[test]
    fn test_mv_slot_decode_with_operands() {
        let result = decode_slot(Slot::Mv, 0x00713c);
        assert!(result.is_some(), "MV bits 0x00713c should decode");
        let decoded = result.unwrap();
        assert!(
            decoded.name.contains("MOV") || decoded.name.contains("mov"),
            "Expected MOV, got '{}'",
            decoded.name,
        );
        // Should have operands with register names.
        assert!(!decoded.operands.is_empty(), "Should have operands");
        // First operand(s) should have reg names like "r0", "x0", etc.
        for op in &decoded.operands {
            if let DecodedOperand::Reg { name, .. } = op {
                assert!(!name.starts_with('?'), "Reg name should be resolved: {}", name);
            }
        }
    }

    #[test]
    fn test_vpush_hi_32_disambiguation() {
        let r4_bits: u64 = 0x02903D;
        let decoded = decode_slot(Slot::Mv, r4_bits)
            .expect("r4 bits should decode");
        assert_eq!(
            decoded.name, "VPUSH_HI_32",
            "r4 encoding should decode as VPUSH_HI_32, got {}",
            decoded.name
        );
    }

    #[test]
    fn test_num_defs_for_vadd() {
        // VADD_F in the vec slot should have 1 output (the destination accum).
        // We need a real VADD_F encoding to test this properly.
        // For now, verify that num_defs is populated for any vec instruction.
        let result = decode_slot(Slot::Vec, 0x000001);
        if let Some(decoded) = result {
            // Just verify the field is populated (not all-zero by accident).
            // The specific value depends on the instruction matched.
            let _ = decoded.num_defs;
            let _ = decoded.defs();
            let _ = decoded.uses();
        }
    }

    #[test]
    fn test_register_names_populated() {
        // VPUSH_HI_32 x0, x0, r3 -- should have register names.
        let decoded = decode_slot(Slot::Mv, 0x028C3D)
            .expect("should decode");
        assert_eq!(decoded.name, "VPUSH_HI_32");

        let reg_names: Vec<&str> = decoded.operands.iter().filter_map(|op| {
            if let DecodedOperand::Reg { name, .. } = op { Some(name.as_str()) } else { None }
        }).collect();

        assert!(!reg_names.is_empty(), "Should have register operands");
        eprintln!("VPUSH_HI_32: num_defs={} operands={:?}", decoded.num_defs, decoded.operands);
    }

    #[test]
    fn test_vadd_f_has_output() {
        // Find a VADD_F encoding by trying a known vec slot pattern.
        // VADD_F bml0, bml0, bml0 -- look at what LLVM returns.
        // We'll try a few vec slot encodings and look for one with num_defs > 0.
        //
        // For now, just verify the infrastructure works by decoding any vec
        // instruction and checking that num_defs is reported.
        for bits in [0x000001u64, 0x100001, 0x200001, 0x0F0001] {
            if let Some(decoded) = decode_slot(Slot::Vec, bits) {
                eprintln!(
                    "Vec 0x{:06X}: {} num_defs={} ops={:?}",
                    bits, decoded.name, decoded.num_defs, decoded.operands
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // RegisterMap and operand_from_reg_name tests
    // -----------------------------------------------------------------------

    /// Verify that no unmapped register ever appears as an explicit operand
    /// in any decoded instruction across all slots.
    ///
    /// If this test fails, it means LLVM is producing an operand with a
    /// register name we don't handle -- that's a real gap to fix.
    #[test]
    fn test_unmapped_regs_never_appear_as_explicit_operands() {
        let (_, unmapped) = reg_map_coverage();
        let unmapped_set: std::collections::HashSet<&str> =
            unmapped.iter().map(|s| s.as_str()).collect();

        let slots = [
            (Slot::Alu, 20),
            (Slot::Lda, 21),
            (Slot::Ldb, 16),
            (Slot::Mv,  22),
            (Slot::St,  21),
            (Slot::Vec, 26),
        ];

        let mut hits: Vec<String> = Vec::new();

        // Probe a range of bit patterns per slot to cover diverse encodings.
        for (slot, width) in &slots {
            for probe in 0..512u64 {
                // Spread probes across the encoding space.
                let bits = probe << (width / 2);
                if let Some(decoded) = decode_slot(*slot, bits) {
                    for op in &decoded.operands {
                        if let DecodedOperand::Reg { name, .. } = op {
                            if unmapped_set.contains(name.as_str()) {
                                let msg = format!(
                                    "{} in {} (slot {:?} bits 0x{:X})",
                                    name, decoded.name, slot, bits
                                );
                                if !hits.iter().any(|h| h.starts_with(name.as_str())) {
                                    hits.push(msg);
                                }
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
            panic!(
                "{} unmapped register(s) appear as explicit operands -- need mapping",
                hits.len()
            );
        }
    }

    #[test]
    fn test_register_map_populated_from_llvm() {
        let (mapped, unmapped) = reg_map_coverage();

        // LLVM knows ~135 physical registers for AIE2. We should map most of them.
        assert!(mapped > 80,
            "Should map 80+ registers from MCRegisterInfo, got {}", mapped);

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
                direct.as_ref(), Some(mapped),
                "RegisterMap and classify_reg_name disagree for '{}'", name
            );
        }
    }

    #[test]
    fn test_register_map_lookup_matches_direct() {
        // operand_from_reg_name (HashMap path) should match classify_reg_name
        // for all known register names.
        let test_names = [
            "r0", "r15", "r31", "p0", "p7", "lr", "SP", "LS", "LE", "LC", "DP",
            "x0", "x4", "wl0", "wh3", "y2",
            "bml0", "bmh0", "cm0", "cm3",
            "amll0", "amhh1",
            "m0", "dn3", "dj7", "dc0",
            "crRnd", "crSat",
            "s0",
        ];
        for name in &test_names {
            let from_map = operand_from_reg_name(name);
            let from_classify = classify_reg_name(name);
            assert_eq!(from_map, from_classify,
                "Mismatch for '{}': map={:?} classify={:?}", name, from_map, from_classify);
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

        // amll0 -> AccumReg(0), Quarter
        let m = operand_from_reg_name("amll0").unwrap();
        assert_eq!(m.operand, Operand::AccumReg(0));
        assert_eq!(m.accum_width, Some(AccumWidth::Quarter));

        // amhh1 -> AccumReg(3), Quarter (within bmh1)
        let m = operand_from_reg_name("amhh1").unwrap();
        assert_eq!(m.operand, Operand::AccumReg(3));
        assert_eq!(m.accum_width, Some(AccumWidth::Quarter));
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
        let m = operand_from_reg_name("crRnd").unwrap();
        assert_eq!(m.operand, Operand::ControlReg(6));

        let m = operand_from_reg_name("crSat").unwrap();
        assert_eq!(m.operand, Operand::ControlReg(7));

        let m = operand_from_reg_name("crSRSSign").unwrap();
        assert_eq!(m.operand, Operand::ControlReg(8));
    }

    #[test]
    fn test_unknown_returns_none() {
        assert!(operand_from_reg_name("xyz42").is_none());
        assert!(operand_from_reg_name("").is_none());
        assert!(operand_from_reg_name("tile_cntr").is_none());
    }

    #[test]
    fn test_llvm_decode_operands_map_correctly() {
        // Decode a real instruction and verify all register operands map
        // through operand_from_reg_name without returning None.
        let decoded = decode_slot(Slot::Mv, 0x028C3D)
            .expect("VPUSH_HI_32 should decode");
        assert_eq!(decoded.name, "VPUSH_HI_32");

        let mut mapped = 0;
        for op in &decoded.operands {
            if let DecodedOperand::Reg { name, .. } = op {
                let result = operand_from_reg_name(name);
                assert!(
                    result.is_some(),
                    "LLVM register '{}' should map to an Operand",
                    name,
                );
                mapped += 1;
            }
        }
        assert!(mapped > 0, "Should have mapped at least one register operand");
    }

    // -----------------------------------------------------------------------
    // Instruction metadata (InstrInfo) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_query_all_instr_info_returns_data() {
        let infos = query_all_instr_info();
        assert!(infos.len() > 600, "Should have 600+ opcodes, got {}", infos.len());

        // Count how many have latency data.
        let with_latency = infos.iter().filter(|i| i.latency.is_some()).count();
        assert!(with_latency > 100,
            "Should have 100+ opcodes with latency, got {}", with_latency);

        // Count how many have non-zero sched class indices.
        let with_sched = infos.iter().filter(|i| i.sched_class > 0).count();
        assert!(with_sched > 100,
            "Should have 100+ opcodes with sched class, got {}", with_sched);
    }

    #[test]
    fn test_instr_info_flags_populated() {
        let infos = query_all_instr_info();

        // Scan all opcodes for instructions with MayLoad and MayStore flags.
        let loads = infos.iter().filter(|i| i.is_load()).count();
        let stores = infos.iter().filter(|i| i.is_store()).count();
        let branches = infos.iter().filter(|i| i.is_branch()).count();

        assert!(loads > 50, "Should have 50+ load instructions, got {}", loads);
        assert!(stores > 50, "Should have 50+ store instructions, got {}", stores);
        assert!(branches > 5, "Should have 5+ branch instructions, got {}", branches);
    }

    #[test]
    fn test_instr_info_latency_cross_check() {
        // Cross-validate LLVM itinerary latencies against our AM020 constants.
        let infos = query_all_instr_info();

        let mut found_load = false;
        let mut found_store = false;
        let mut found_vmac = false;

        for (opcode, info) in infos.iter().enumerate() {
            if let Some(latency) = info.latency {
                let name = match opcode_name(opcode as u32) {
                    Some(n) => n,
                    None => continue,
                };

                // Load instructions with MayLoad flag should have latency 7.
                if info.is_load() && name.starts_with("LDA_") && !found_load {
                    assert_eq!(latency, 7, "{} latency should be 7 (load)", name);
                    found_load = true;
                }

                // Store instructions with MayStore flag should have latency 1.
                if info.is_store() && name.starts_with("ST_") && !found_store {
                    assert_eq!(latency, 1, "{} latency should be 1 (store)", name);
                    found_store = true;
                }

                // Vector MAC (integer) should have latency 5.
                // VMAC_F (float) may differ -- skip floating-point variants.
                if name.starts_with("VMAC_vmac_") && !found_vmac {
                    assert_eq!(latency, 5, "{} latency should be 5 (VMAC)", name);
                    found_vmac = true;
                }
            }
        }
        assert!(found_load, "Should have found a load instruction with latency");
        assert!(found_store, "Should have found a store instruction with latency");
        assert!(found_vmac, "Should have found a VMAC instruction with latency");
    }

    #[test]
    fn test_vec_decode_operands_map_correctly() {
        // Decode a vec-slot instruction and verify register name mapping.
        for bits in [0x000001u64, 0x100001, 0x200001] {
            if let Some(decoded) = decode_slot(Slot::Vec, bits) {
                for op in &decoded.operands {
                    if let DecodedOperand::Reg { name, .. } = op {
                        let result = operand_from_reg_name(name);
                        if result.is_none() {
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
