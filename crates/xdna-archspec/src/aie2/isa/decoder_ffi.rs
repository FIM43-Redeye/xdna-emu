//! FFI bindings to LLVM's AIE2 MCDisassembler.
//!
//! Links against llvm-aie's disassembler library for:
//! - Instruction identification with full TRY_DECODE disambiguation
//! - Decoded operand values (register IDs and immediates)
//! - Register names from MCRegisterInfo (e.g., "r0", "bm0", "cm0")
//! - Output operand classification from MCInstrDesc::NumDefs
//!
//! The C side (`decoder_ffi/aie2_decoder.cpp`) is compiled and linked by
//! **xdna-emu's** build.rs (not this crate's build.rs). The FFI symbols
//! are resolved at link time when xdna-archspec is linked into xdna-emu.
//! Subsystem 6 Tasks 8 and 9 will move the C++ compilation into this
//! crate's build.rs, making archspec self-contained.
//!
//! The interpreter-aware half (MappedOperand, RegisterMap, classify_reg_name)
//! lives in xdna-emu's `tablegen::register_map` (relocates to
//! `interpreter::decode::register_map` in Subsystem 6 Part B).

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
    pub latency: i16,       // operand_cycles[0], or -1 if unavailable
    pub stage_latency: i16, // Pipeline stage sum, or -1 if unavailable
    pub sched_class: u16,   // Itinerary class index (opaque)
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
                    unsafe { CStr::from_ptr(op.reg_name) }.to_string_lossy().into_owned()
                } else {
                    format!("?{}", op.value)
                };
                operands.push(DecodedOperand::Reg { id: op.value as u32, name: reg_name });
            }
            OpKind::Imm => {
                operands.push(DecodedOperand::Imm(op.value));
            }
            _ => {}
        }
    }

    Some(DecodeResult { name, opcode: raw.opcode, num_defs: raw.num_defs, operands })
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

/// Get the total number of physical registers in MCRegisterInfo.
///
/// Returns 0 if the decoder is not initialized. Used by RegisterMap
/// to iterate over all register IDs and build the name-to-Operand table.
pub fn get_num_regs() -> u32 {
    if !init() {
        return 0;
    }
    unsafe { aie2_get_num_regs() }
}

/// Get the LLVM register name for a register ID.
///
/// Returns None if the decoder is not initialized or the name is null/empty.
pub fn get_reg_name(reg_id: u32) -> Option<String> {
    if !init() {
        return None;
    }
    let ptr = unsafe { aie2_get_reg_name(reg_id) };
    if ptr.is_null() {
        return None;
    }
    let name = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
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
    pub const RETURN: u64 = 1 << 5;
    pub const CALL: u64 = 1 << 7;
    pub const BARRIER: u64 = 1 << 8;
    pub const TERMINATOR: u64 = 1 << 9;
    pub const BRANCH: u64 = 1 << 10;
    pub const INDIRECT_BRANCH: u64 = 1 << 11;
    pub const COMPARE: u64 = 1 << 12;
    pub const MOVE_IMM: u64 = 1 << 13;
    pub const MOVE_REG: u64 = 1 << 14;
    pub const MAY_LOAD: u64 = 1 << 19;
    pub const MAY_STORE: u64 = 1 << 20;
    pub const COMMUTABLE: u64 = 1 << 25;
    pub const ADD: u64 = 1 << 37;
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
    pub fn is_load(&self) -> bool {
        self.flags & mcid::MAY_LOAD != 0
    }
    pub fn is_store(&self) -> bool {
        self.flags & mcid::MAY_STORE != 0
    }
    pub fn is_branch(&self) -> bool {
        self.flags & mcid::BRANCH != 0
    }
    pub fn is_call(&self) -> bool {
        self.flags & mcid::CALL != 0
    }
    pub fn is_return(&self) -> bool {
        self.flags & mcid::RETURN != 0
    }
    pub fn is_terminator(&self) -> bool {
        self.flags & mcid::TERMINATOR != 0
    }
    pub fn is_compare(&self) -> bool {
        self.flags & mcid::COMPARE != 0
    }
    pub fn is_commutable(&self) -> bool {
        self.flags & mcid::COMMUTABLE != 0
    }
    pub fn is_move_reg(&self) -> bool {
        self.flags & mcid::MOVE_REG != 0
    }
    pub fn is_move_imm(&self) -> bool {
        self.flags & mcid::MOVE_IMM != 0
    }
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
        assert!(!decoded.operands.is_empty(), "Should have operands");
        for op in &decoded.operands {
            if let DecodedOperand::Reg { name, .. } = op {
                assert!(!name.starts_with('?'), "Reg name should be resolved: {}", name);
            }
        }
    }

    #[test]
    fn test_vpush_hi_32_disambiguation() {
        let r4_bits: u64 = 0x02903D;
        let decoded = decode_slot(Slot::Mv, r4_bits).expect("r4 bits should decode");
        assert_eq!(decoded.name, "VPUSH_HI_32");
    }

    #[test]
    fn test_num_defs_for_vadd() {
        let result = decode_slot(Slot::Vec, 0x000001);
        if let Some(decoded) = result {
            let _ = decoded.num_defs;
            let _ = decoded.defs();
            let _ = decoded.uses();
        }
    }

    #[test]
    fn test_register_names_populated() {
        let decoded = decode_slot(Slot::Mv, 0x028C3D).expect("should decode");
        assert_eq!(decoded.name, "VPUSH_HI_32");
        let reg_names: Vec<&str> = decoded
            .operands
            .iter()
            .filter_map(|op| {
                if let DecodedOperand::Reg { name, .. } = op {
                    Some(name.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(!reg_names.is_empty(), "Should have register operands");
    }

    #[test]
    fn test_vadd_f_decodes_correctly() {
        let decoded = decode_slot(Slot::Vec, 0x2).expect("VADD_F should decode");
        assert_eq!(decoded.name, "VADD_F", "LLVM should identify as VADD_F");
        assert_eq!(decoded.num_defs, 1, "one output (destination accum)");
        for op in &decoded.operands {
            if let DecodedOperand::Reg { name, .. } = op {
                assert!(
                    name.starts_with("bm") || name.starts_with("r"),
                    "VADD_F operand should be accum or scalar, got {}",
                    name,
                );
            }
        }
    }

    #[test]
    fn test_vmac_f_untied_acc_operands() {
        let vec_bits: u64 = 0x0040001;
        let decoded = decode_slot(Slot::Vec, vec_bits).expect("VMAC_F with untied acc should decode");
        assert!(decoded.name.starts_with("VMAC_F"), "Expected VMAC_F, got {}", decoded.name);
        assert_eq!(decoded.num_defs, 1, "one output def");
        let reg_names: Vec<&str> = decoded
            .operands
            .iter()
            .filter_map(|op| {
                if let DecodedOperand::Reg { name, .. } = op {
                    Some(name.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(reg_names.len() >= 2, "Need at least dst + acc1 registers");
        assert_eq!(reg_names[0], "bml0", "dst should be bml0");
        assert_eq!(reg_names[1], "bml2", "acc1 should be bml2");
    }

    // -----------------------------------------------------------------------
    // Instruction metadata (InstrInfo) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_query_all_instr_info_returns_data() {
        let infos = query_all_instr_info();
        assert!(infos.len() > 600, "Should have 600+ opcodes, got {}", infos.len());
        let with_latency = infos.iter().filter(|i| i.latency.is_some()).count();
        assert!(with_latency > 100, "Should have 100+ opcodes with latency, got {}", with_latency);
        let with_sched = infos.iter().filter(|i| i.sched_class > 0).count();
        assert!(with_sched > 100, "Should have 100+ opcodes with sched class, got {}", with_sched);
    }

    #[test]
    fn test_instr_info_flags_populated() {
        let infos = query_all_instr_info();
        let loads = infos.iter().filter(|i| i.is_load()).count();
        let stores = infos.iter().filter(|i| i.is_store()).count();
        let branches = infos.iter().filter(|i| i.is_branch()).count();
        assert!(loads > 50, "Should have 50+ load instructions, got {}", loads);
        assert!(stores > 50, "Should have 50+ store instructions, got {}", stores);
        assert!(branches > 5, "Should have 5+ branch instructions, got {}", branches);
    }

    #[test]
    fn test_instr_info_latency_cross_check() {
        // Cross-validate LLVM itinerary latencies against AM020 constants.
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
                if info.is_load() && name.starts_with("LDA_") && !found_load {
                    assert_eq!(latency, 7, "{} latency should be 7 (load)", name);
                    found_load = true;
                }
                if info.is_store() && name.starts_with("ST_") && !found_store {
                    assert_eq!(latency, 1, "{} latency should be 1 (store)", name);
                    found_store = true;
                }
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
}
