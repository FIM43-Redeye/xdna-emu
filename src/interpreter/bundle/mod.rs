//! VLIW bundle representation.
//!
//! This module defines the core data structures for representing decoded
//! AIE2 VLIW (Very Long Instruction Word) bundles.
//!
//! # AIE2 VLIW Architecture
//!
//! AIE2 uses a 128-bit VLIW format that can encode up to 7 parallel operations:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     128-bit VLIW Bundle                             │
//! ├──────────┬──────────┬────────┬───────┬──────┬───────┬───────────────┤
//! │ Scalar0  │ Scalar1  │ Vector │ Accum │ Load │ Store │    Control    │
//! │  (ALU)   │  (MUL)   │ (SIMD) │ (MAC) │      │       │ (Branch/Lock) │
//! └──────────┴──────────┴────────┴───────┴──────┴───────┴───────────────┘
//! ```
//!
//! Not all combinations of slots are valid - the hardware has resource
//! constraints that the decoder must respect.
//!
//! # Bundle Formats
//!
//! Instructions come in three sizes:
//!
//! - **Short (32-bit)**: Single operation, typically scalar or control
//! - **Medium (64-bit)**: Two operations, scalar + limited vector/memory
//! - **Full (128-bit)**: All 7 slots potentially active
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::interpreter::bundle::{VliwBundle, SlotOp, Operation};
//!
//! let bundle = VliwBundle::from_bytes(&instruction_bytes)?;
//!
//! for slot_op in bundle.active_slots() {
//!     println!("Slot {:?}: {:?}", slot_op.slot, slot_op.op);
//! }
//! ```

pub mod encoding;
pub mod slot;

pub use encoding::{detect_format, is_nop_encoding, BundleFormat, SlotMask, NOP_ENCODINGS};
pub use slot::{
    BranchCondition, ElementType, MemWidth, Operand, Operation, PostModify, Predicate,
    ShufflePattern, SlotIndex, SlotOp,
};

/// A decoded VLIW instruction bundle.
///
/// Contains up to 7 slot operations that execute in parallel within
/// a single cycle. The bundle also tracks the raw bytes and format
/// for debugging and disassembly.
#[derive(Debug, Clone)]
pub struct VliwBundle {
    /// Raw instruction bytes (up to 16 for full VLIW).
    raw: [u8; 16],

    /// Decoded slot operations (None if slot is empty/NOP).
    slots: [Option<SlotOp>; 7],

    /// Size of this bundle in bytes (4, 8, or 16).
    size: u8,

    /// Bundle format.
    format: BundleFormat,

    /// Program counter where this bundle was decoded.
    pc: u32,
}

impl VliwBundle {
    /// Create an empty bundle.
    pub fn empty() -> Self {
        Self {
            raw: [0; 16],
            slots: Default::default(),
            size: 4,
            format: BundleFormat::Short32,
            pc: 0,
        }
    }

    /// Create a NOP bundle.
    pub fn nop() -> Self {
        let mut bundle = Self::empty();
        bundle.slots[0] = Some(SlotOp::nop(SlotIndex::Scalar0));
        bundle
    }

    /// Create a bundle from raw bytes.
    ///
    /// This creates a bundle with the raw bytes but no decoded slots.
    /// Use the decoder to actually decode the instruction.
    pub fn from_raw(bytes: &[u8], pc: u32) -> Self {
        let mut raw = [0u8; 16];
        let len = bytes.len().min(16);
        raw[..len].copy_from_slice(&bytes[..len]);

        let format = detect_format(bytes);

        Self {
            raw,
            slots: Default::default(),
            size: format.size_bytes(),
            format,
            pc,
        }
    }

    /// Get the raw instruction bytes.
    #[inline]
    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw[..self.size as usize]
    }

    /// Get the first instruction word (for quick opcode access).
    #[inline]
    pub fn word0(&self) -> u32 {
        u32::from_le_bytes([self.raw[0], self.raw[1], self.raw[2], self.raw[3]])
    }

    /// Get the size of this bundle in bytes.
    #[inline]
    pub fn size(&self) -> u8 {
        self.size
    }

    /// Get the bundle format.
    #[inline]
    pub fn format(&self) -> BundleFormat {
        self.format
    }

    /// Get the program counter where this bundle was decoded.
    #[inline]
    pub fn pc(&self) -> u32 {
        self.pc
    }

    /// Set the program counter.
    #[inline]
    pub fn set_pc(&mut self, pc: u32) {
        self.pc = pc;
    }

    /// Get a reference to all slots.
    #[inline]
    pub fn slots(&self) -> &[Option<SlotOp>; 7] {
        &self.slots
    }

    /// Get a mutable reference to all slots.
    #[inline]
    pub fn slots_mut(&mut self) -> &mut [Option<SlotOp>; 7] {
        &mut self.slots
    }

    /// Get the operation in a specific slot.
    #[inline]
    pub fn slot(&self, index: SlotIndex) -> Option<&SlotOp> {
        self.slots[index as usize].as_ref()
    }

    /// Set the operation in a specific slot.
    pub fn set_slot(&mut self, op: SlotOp) {
        let slot_index = op.slot as usize;
        self.slots[slot_index] = Some(op);
    }

    /// Clear a slot.
    pub fn clear_slot(&mut self, index: SlotIndex) {
        self.slots[index as usize] = None;
    }

    /// Iterate over active (non-empty) slots.
    pub fn active_slots(&self) -> impl Iterator<Item = &SlotOp> {
        self.slots.iter().filter_map(|s| s.as_ref())
    }

    /// Count the number of active slots.
    pub fn active_slot_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Get the slot mask for this bundle.
    pub fn slot_mask(&self) -> SlotMask {
        let mut mask = SlotMask::EMPTY;
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.is_some() {
                mask.set_active(i as u8);
            }
        }
        mask
    }

    /// Check if this bundle is effectively a NOP.
    pub fn is_nop(&self) -> bool {
        self.active_slots().all(|s| s.op.is_nop())
    }

    /// Check if this bundle contains a control flow operation.
    pub fn has_control_flow(&self) -> bool {
        self.active_slots().any(|s| s.op.is_control_flow())
    }

    /// Check if this bundle contains a memory operation.
    pub fn has_memory_op(&self) -> bool {
        self.active_slots().any(|s| s.op.is_memory())
    }

    /// Check if this bundle contains a synchronization operation.
    pub fn has_sync_op(&self) -> bool {
        self.active_slots().any(|s| s.op.is_sync())
    }

    /// Get the control flow operation if present.
    pub fn control_op(&self) -> Option<&SlotOp> {
        self.slot(SlotIndex::Control)
            .filter(|s| s.op.is_control_flow())
    }

    /// Get a disassembly string for this bundle.
    pub fn disassemble(&self) -> String {
        let mut parts = Vec::new();

        for slot_op in self.active_slots() {
            let op_str = disassemble_op(slot_op);
            parts.push(op_str);
        }

        if parts.is_empty() {
            return "nop".to_string();
        }

        // For single operations, just return the operation
        if parts.len() == 1 {
            return parts.into_iter().next().unwrap();
        }

        // For VLIW bundles, wrap in braces
        format!("{{ {} }}", parts.join(" ; "))
    }
}

impl Default for VliwBundle {
    fn default() -> Self {
        Self::empty()
    }
}

/// Disassemble a single slot operation.
fn disassemble_op(slot_op: &SlotOp) -> String {
    let op_name = match &slot_op.op {
        Operation::ScalarAdd => "add",
        Operation::ScalarSub => "sub",
        Operation::ScalarMul => "mul",
        Operation::ScalarAnd => "and",
        Operation::ScalarOr => "or",
        Operation::ScalarXor => "xor",
        Operation::ScalarShl => "shl",
        Operation::ScalarShr => "shr",
        Operation::ScalarSra => "asr",
        Operation::ScalarMov => "mov",
        Operation::ScalarMovi { value } => return format!("movi #{}", value),
        Operation::ScalarCmp => "cmp",

        Operation::VectorAdd { element_type } => {
            return format!("vadd.{}", element_suffix(*element_type))
        }
        Operation::VectorSub { element_type } => {
            return format!("vsub.{}", element_suffix(*element_type))
        }
        Operation::VectorMul { element_type } => {
            return format!("vmul.{}", element_suffix(*element_type))
        }
        Operation::VectorMac { element_type } => {
            return format!("vmac.{}", element_suffix(*element_type))
        }
        Operation::VectorShuffle { .. } => "vshuffle",
        Operation::VectorPack => "vpack",
        Operation::VectorUnpack => "vunpack",
        Operation::VectorCmp { .. } => "vcmp",
        Operation::VectorMin { .. } => "vmin",
        Operation::VectorMax { .. } => "vmax",

        Operation::Load { width, .. } => return format!("ld.{}", width_suffix(*width)),
        Operation::Store { width, .. } => return format!("st.{}", width_suffix(*width)),

        Operation::Branch { condition } => return format!("b{}", condition_suffix(*condition)),
        Operation::Call => "call",
        Operation::Return => "ret",

        Operation::LockAcquire => "lock.acquire",
        Operation::LockRelease => "lock.release",
        Operation::DmaStart => "dma.start",
        Operation::DmaWait => "dma.wait",

        Operation::Nop => "nop",
        Operation::Halt => "halt",
        Operation::Unknown { opcode } => return format!(".word 0x{:08X}", opcode),
    };

    let mut result = op_name.to_string();

    // Add destination
    if let Some(dest) = &slot_op.dest {
        result.push(' ');
        result.push_str(&operand_str(dest));
    }

    // Add sources
    for (i, src) in slot_op.sources.iter().enumerate() {
        if i == 0 && slot_op.dest.is_some() {
            result.push_str(", ");
        } else if i > 0 {
            result.push_str(", ");
        } else {
            result.push(' ');
        }
        result.push_str(&operand_str(src));
    }

    result
}

fn operand_str(op: &Operand) -> String {
    match op {
        Operand::ScalarReg(r) => format!("r{}", r),
        Operand::VectorReg(r) => format!("v{}", r),
        Operand::AccumReg(r) => format!("acc{}", r),
        Operand::PointerReg(r) => format!("p{}", r),
        Operand::ModifierReg(r) => format!("m{}", r),
        Operand::Immediate(v) => format!("#{}", v),
        Operand::Memory { base, offset } => {
            if *offset == 0 {
                format!("[p{}]", base)
            } else {
                format!("[p{}, #{}]", base, offset)
            }
        }
        Operand::Lock(id) => format!("lock{}", id),
        Operand::DmaChannel(ch) => format!("ch{}", ch),
        Operand::BufferDescriptor(bd) => format!("bd{}", bd),
    }
}

fn element_suffix(et: ElementType) -> &'static str {
    match et {
        ElementType::Int8 => "i8",
        ElementType::UInt8 => "u8",
        ElementType::Int16 => "i16",
        ElementType::UInt16 => "u16",
        ElementType::Int32 => "i32",
        ElementType::UInt32 => "u32",
        ElementType::BFloat16 => "bf16",
        ElementType::Float32 => "f32",
    }
}

fn width_suffix(w: MemWidth) -> &'static str {
    match w {
        MemWidth::Byte => "b",
        MemWidth::HalfWord => "h",
        MemWidth::Word => "w",
        MemWidth::DoubleWord => "d",
        MemWidth::QuadWord => "q",
        MemWidth::Vector256 => "v",
    }
}

fn condition_suffix(c: BranchCondition) -> &'static str {
    match c {
        BranchCondition::Always => "",
        BranchCondition::Equal => "eq",
        BranchCondition::NotEqual => "ne",
        BranchCondition::Less => "lt",
        BranchCondition::GreaterEqual => "ge",
        BranchCondition::LessEqual => "le",
        BranchCondition::Greater => "gt",
        BranchCondition::Negative => "mi",
        BranchCondition::PositiveOrZero => "pl",
        BranchCondition::CarrySet => "cs",
        BranchCondition::CarryClear => "cc",
        BranchCondition::OverflowSet => "vs",
        BranchCondition::OverflowClear => "vc",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_bundle() {
        let bundle = VliwBundle::empty();
        assert_eq!(bundle.size(), 4);
        assert_eq!(bundle.format(), BundleFormat::Short32);
        assert_eq!(bundle.active_slot_count(), 0);
        assert!(bundle.is_nop());
    }

    #[test]
    fn test_nop_bundle() {
        let bundle = VliwBundle::nop();
        assert_eq!(bundle.active_slot_count(), 1);
        assert!(bundle.is_nop());
    }

    #[test]
    fn test_from_raw() {
        let bytes = [0x00u8, 0x00, 0x00, 0x00]; // NOP
        let bundle = VliwBundle::from_raw(&bytes, 0x100);
        assert_eq!(bundle.pc(), 0x100);
        assert_eq!(bundle.word0(), 0);
    }

    #[test]
    fn test_set_slot() {
        let mut bundle = VliwBundle::empty();

        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1))
            .with_source(Operand::ScalarReg(2));

        bundle.set_slot(op);

        assert_eq!(bundle.active_slot_count(), 1);
        assert!(bundle.slot(SlotIndex::Scalar0).is_some());
        assert!(bundle.slot(SlotIndex::Vector).is_none());
    }

    #[test]
    fn test_slot_mask() {
        let mut bundle = VliwBundle::empty();
        bundle.set_slot(SlotOp::nop(SlotIndex::Scalar0));
        bundle.set_slot(SlotOp::nop(SlotIndex::Vector));

        let mask = bundle.slot_mask();
        assert!(mask.is_active(0)); // Scalar0
        assert!(!mask.is_active(1)); // Scalar1
        assert!(mask.is_active(2)); // Vector
        assert_eq!(mask.count(), 2);
    }

    #[test]
    fn test_has_control_flow() {
        let mut bundle = VliwBundle::empty();
        assert!(!bundle.has_control_flow());

        bundle.set_slot(SlotOp::new(
            SlotIndex::Control,
            Operation::Branch {
                condition: BranchCondition::Always,
            },
        ));
        assert!(bundle.has_control_flow());
    }

    #[test]
    fn test_disassemble_scalar() {
        let mut bundle = VliwBundle::empty();
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1))
            .with_source(Operand::ScalarReg(2));
        bundle.set_slot(op);

        assert_eq!(bundle.disassemble(), "add r0, r1, r2");
    }

    #[test]
    fn test_disassemble_vector() {
        let mut bundle = VliwBundle::empty();
        let op = SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorAdd {
                element_type: ElementType::Int32,
            },
        );
        bundle.set_slot(op);

        assert_eq!(bundle.disassemble(), "vadd.i32");
    }

    #[test]
    fn test_disassemble_vliw() {
        let mut bundle = VliwBundle::empty();
        bundle.set_slot(SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd));
        bundle.set_slot(SlotOp::new(
            SlotIndex::Vector,
            Operation::VectorMul {
                element_type: ElementType::Int16,
            },
        ));

        let dis = bundle.disassemble();
        assert!(dis.starts_with("{"));
        assert!(dis.contains("add"));
        assert!(dis.contains("vmul.i16"));
    }
}
