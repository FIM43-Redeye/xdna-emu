//! Per-slot decode diagnostics for corpus sweeps.
//!
//! `InstructionDecoder::decode` folds every failure into an "unknown" slot, so
//! a caller cannot tell *why* a slot was rejected. The probe re-runs the same
//! three stages the production path uses (slot extraction, LLVM FFI
//! identification, encoding lookup by LLVM name) and reports where each slot
//! stopped. It is read-only and shares `try_decode_via_ffi` with `decode`, so
//! the two cannot drift apart.

use xdna_archspec::aie2::isa::decoder_ffi::{self, DecodedOperand};

use super::InstructionDecoder;
use crate::interpreter::bundle::SlotType;

/// Where a slot's decode stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotOutcome {
    /// Decoded end to end; this is what the interpreter executes.
    Decoded(DecodedSlot),
    /// LLVM's MCDisassembler found no instruction for these bits.
    FfiNoMatch,
    /// LLVM returned an instruction name that has no encoding in our index
    /// (for example the pseudo-instruction `PHI`).
    LookupMiss { llvm_name: String },
    /// LLVM and the index agree on a name, but operand mapping rejected it.
    OperandMapFailed { llvm_name: String },
}

/// What the FFI reported for a successfully decoded slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedSlot {
    /// LLVM instruction name, e.g. `PADDB_sp_imm`.
    pub llvm_name: String,
    /// Register operands in MI order (defs first), lowercase LLVM names.
    pub regs: Vec<String>,
    /// Immediate operands in MI order.
    pub imms: Vec<i64>,
}

/// One extracted slot and its outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotProbe {
    pub slot_type: SlotType,
    /// Right-aligned raw slot bits.
    pub bits: u64,
    /// Bit width of the slot.
    pub width: u8,
    pub outcome: SlotOutcome,
}

impl SlotProbe {
    pub fn is_failure(&self) -> bool {
        !matches!(self.outcome, SlotOutcome::Decoded(_))
    }
}

impl InstructionDecoder {
    /// Extract every slot of the bundle starting at `bytes` and report how
    /// each one decodes. NOP-marker slots are omitted.
    pub fn probe_bundle(&self, bytes: &[u8]) -> Vec<SlotProbe> {
        let extracted = self.extract_bundle_slots(bytes);
        extracted
            .slots
            .iter()
            .filter(|s| s.slot_type != SlotType::Nop)
            .map(|s| SlotProbe {
                slot_type: s.slot_type,
                bits: s.bits,
                width: s.width,
                outcome: self.probe_slot(s.bits, s.slot_type),
            })
            .collect()
    }

    fn probe_slot(&self, bits: u64, slot_type: SlotType) -> SlotOutcome {
        let Some(ffi_slot) = Self::slot_type_to_ffi(slot_type) else {
            return SlotOutcome::FfiNoMatch;
        };
        let Some(result) = decoder_ffi::decode_slot(ffi_slot, bits) else {
            return SlotOutcome::FfiNoMatch;
        };
        if self.try_decode_via_ffi(bits, slot_type).is_none() {
            let in_index = ["lda", "ldb", "alu", "mv", "st", "vec", "lng"]
                .iter()
                .any(|s| self.index.encoding_by_name(s, &result.name).is_some());
            return if in_index {
                SlotOutcome::OperandMapFailed { llvm_name: result.name }
            } else {
                SlotOutcome::LookupMiss { llvm_name: result.name }
            };
        }
        let mut regs = Vec::new();
        let mut imms = Vec::new();
        for op in &result.operands {
            match op {
                DecodedOperand::Reg { name, .. } => regs.push(name.to_lowercase()),
                DecodedOperand::Imm(v) => imms.push(*v),
            }
        }
        SlotOutcome::Decoded(DecodedSlot { llvm_name: result.name, regs, imms })
    }
}
