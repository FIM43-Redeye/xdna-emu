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
//! use xdna_emu::interpreter::bundle::{VliwBundle, SlotOp};
//!
//! let bundle = VliwBundle::from_bytes(&instruction_bytes)?;
//!
//! for slot_op in bundle.active_slots() {
//!     println!("Slot {:?}: {:?}", slot_op.slot, slot_op.semantic);
//! }
//! ```

pub mod encoding;
pub mod slot;
pub mod slot_layout;

pub use encoding::{detect_format, is_nop_encoding, BundleFormat, SlotMask, NOP_ENCODINGS};
pub use slot_layout::{extract_slots, ExtractedBundle, ExtractedSlot, FormatTable, SlotType};
pub use slot::{
    BranchCondition, ElementType, MemWidth, Operand, PostModify, Predicate,
    SelectVariant, ShufflePattern, SlotIndex, SlotOp,
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
    slots: [Option<SlotOp>; 8],

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
    pub fn slots(&self) -> &[Option<SlotOp>; 8] {
        &self.slots
    }

    /// Get a mutable reference to all slots.
    #[inline]
    pub fn slots_mut(&mut self) -> &mut [Option<SlotOp>; 8] {
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
        self.active_slots().all(|s| s.is_nop())
    }

    /// Check if this bundle contains a control flow operation.
    pub fn has_control_flow(&self) -> bool {
        self.active_slots().any(|s| s.is_control_flow())
    }

    /// Check if this bundle contains a memory operation.
    pub fn has_memory_op(&self) -> bool {
        self.active_slots().any(|s| s.is_memory())
    }

    /// Check if this bundle contains a synchronization operation.
    pub fn has_sync_op(&self) -> bool {
        self.active_slots().any(|s| s.is_sync())
    }

    /// Get the control flow operation if present.
    pub fn control_op(&self) -> Option<&SlotOp> {
        self.slot(SlotIndex::Control)
            .filter(|s| s.is_control_flow())
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
///
/// Uses `encoding_name` (from TableGen) when available, falling back to
/// a SemanticOp-based mnemonic for test-constructed SlotOps.
fn disassemble_op(slot_op: &SlotOp) -> String {
    // Primary: use the actual encoding name from TableGen
    let op_name = if let Some(name) = &slot_op.encoding_name {
        name.clone()
    } else {
        // Fallback: build mnemonic from SemanticOp + metadata
        disassemble_from_semantic(slot_op)
    };

    let mut result = op_name;

    // Add destination
    if let Some(dest) = &slot_op.dest {
        result.push(' ');
        result.push_str(&operand_str(dest));
    }

    // Add sources
    for (i, src) in slot_op.sources.iter().enumerate() {
        if slot_op.dest.is_some() || i > 0 {
            result.push_str(", ");
        } else {
            result.push(' ');
        }
        result.push_str(&operand_str(src));
    }

    result
}

/// Build a mnemonic string from SemanticOp + SlotOp metadata.
///
/// This is the fallback path for SlotOps constructed in tests without
/// an encoding_name. Real decoded instructions always have encoding_name.
fn disassemble_from_semantic(slot_op: &SlotOp) -> String {
    use crate::tablegen::SemanticOp;

    let Some(semantic) = slot_op.semantic else {
        return "unknown".to_string();
    };

    let et = slot_op.element_type;
    let is_vec = slot_op.is_vector;

    // Helper closures for common patterns
    let vec_et = |prefix: &str| -> String {
        if let Some(e) = et {
            format!("{}.{}", prefix, element_suffix(e))
        } else {
            prefix.to_string()
        }
    };

    let dual_et = |prefix: &str| -> String {
        let from = slot_op.from_type.unwrap_or(crate::interpreter::bundle::ElementType::Int32);
        let to = et.unwrap_or(crate::interpreter::bundle::ElementType::Int32);
        format!("{}.{}.{}", prefix, element_suffix(from), element_suffix(to))
    };

    match semantic {
        // Arithmetic
        SemanticOp::Add => if is_vec { vec_et("vadd") } else { "add".into() },
        SemanticOp::Sub => if is_vec { vec_et("vsub") } else { "sub".into() },
        SemanticOp::Mul => if is_vec { vec_et("vmul") } else { "mul".into() },
        SemanticOp::Adc => "adc".into(),
        SemanticOp::Sbc => "sbc".into(),
        SemanticOp::Abs => "abs".into(),
        SemanticOp::Neg => if is_vec { vec_et("vneg") } else { "neg".into() },
        SemanticOp::SDiv => "divs".into(),
        SemanticOp::UDiv => "divu".into(),
        SemanticOp::SRem | SemanticOp::URem => "mod".into(),

        // Bitwise
        SemanticOp::And => if is_vec { vec_et("vband") } else { "and".into() },
        SemanticOp::Or => if is_vec { vec_et("vbor") } else { "or".into() },
        SemanticOp::Xor => if is_vec { vec_et("vbxor") } else { "xor".into() },
        SemanticOp::Not => if is_vec { vec_et("vbnot") } else { "not".into() },
        SemanticOp::Shl => if is_vec { vec_et("vshl") } else { "shl".into() },
        SemanticOp::Srl => if is_vec { vec_et("vshr") } else { "shr".into() },
        SemanticOp::Sra => if is_vec { vec_et("vasr") } else { "asr".into() },
        SemanticOp::Rotl => "rotl".into(),
        SemanticOp::Rotr => "rotr".into(),

        // Comparison
        SemanticOp::SetLt => if is_vec { vec_et("vlt") } else { "lt".into() },
        SemanticOp::SetLe => "le".into(),
        SemanticOp::SetGt => "gt".into(),
        SemanticOp::SetGe => if is_vec { vec_et("vge") } else { "ge".into() },
        SemanticOp::SetEq => if is_vec { vec_et("veqz") } else { "eq".into() },
        SemanticOp::SetNe => "ne".into(),
        SemanticOp::SetUlt => "ltu".into(),
        SemanticOp::SetUle => "leu".into(),
        SemanticOp::SetUgt => "gtu".into(),
        SemanticOp::SetUge => "geu".into(),
        SemanticOp::Cmp => if is_vec { vec_et("vcmp") } else { "cmp".into() },
        SemanticOp::Clb => "clb".into(),

        // Bit manipulation
        SemanticOp::Ctlz => "clz".into(),
        SemanticOp::Cttz => "ctz".into(),
        SemanticOp::Ctpop => "popcount".into(),
        SemanticOp::Bswap => "bswap".into(),

        // Copy/extend
        SemanticOp::Copy => if is_vec { vec_et("vmov") } else { "mov".into() },
        SemanticOp::SignExtend => "ext.s".into(),
        SemanticOp::ZeroExtend => "ext.u".into(),
        SemanticOp::Truncate => "trunc".into(),
        SemanticOp::Select => "sel".into(),
        SemanticOp::Nop => "nop".into(),

        // Memory
        SemanticOp::Load => {
            if is_vec {
                "vlda".into()
            } else {
                format!("ld.{}", width_suffix(slot_op.mem_width))
            }
        }
        SemanticOp::Store => {
            if is_vec {
                "vst".into()
            } else {
                format!("st.{}", width_suffix(slot_op.mem_width))
            }
        }

        // Control
        SemanticOp::Br => {
            if let Some(bc) = slot_op.branch_condition {
                format!("b{}", condition_suffix(bc))
            } else {
                "b".into()
            }
        }
        SemanticOp::BrCond => {
            if let Some(bc) = slot_op.branch_condition {
                format!("b{}", condition_suffix(bc))
            } else {
                "bcond".into()
            }
        }
        SemanticOp::Call => "call".into(),
        SemanticOp::Ret => "ret".into(),
        SemanticOp::Done | SemanticOp::Halt => "halt".into(),
        SemanticOp::Event => "event".into(),

        // Sync
        SemanticOp::LockAcquire => "lock.acquire".into(),
        SemanticOp::LockRelease => "lock.release".into(),
        SemanticOp::DmaStart => "dma.start".into(),
        SemanticOp::DmaWait => "dma.wait".into(),

        // Vector-specific
        SemanticOp::Mac => vec_et("vmac"),
        SemanticOp::MatMul => vec_et("vmul.dense"),
        SemanticOp::MatMulSub => vec_et("vmsc.dense"),
        SemanticOp::NegMatMul => vec_et("vnegmac"),
        SemanticOp::AddMac => vec_et("vaddmac"),
        SemanticOp::SubMac => vec_et("vsubmac"),
        SemanticOp::Srs => dual_et("vsrs"),
        SemanticOp::Ups => dual_et("vups"),
        SemanticOp::Convert => dual_et("vconv"),
        SemanticOp::Shuffle => "vshuffle".into(),
        SemanticOp::Pack => "vpack".into(),
        SemanticOp::Unpack => "vunpack".into(),
        SemanticOp::Align => vec_et("valign"),
        SemanticOp::VectorBroadcast => vec_et("vbcst"),
        SemanticOp::VectorExtract => vec_et("vext"),
        SemanticOp::VectorInsert => vec_et("vins"),
        SemanticOp::VectorSelect => vec_et("vsel"),
        SemanticOp::VectorClear => "vclr".into(),
        SemanticOp::Min => if is_vec { vec_et("vmin") } else { "min".into() },
        SemanticOp::Max => if is_vec { vec_et("vmax") } else { "max".into() },

        // Conditional vector
        SemanticOp::SubLt => vec_et("vsub_lt"),
        SemanticOp::SubGe => vec_et("vsub_ge"),
        SemanticOp::MaxDiffLt => vec_et("vmaxdiff_lt"),
        SemanticOp::MaxLt => vec_et("vmax_lt"),
        SemanticOp::MinGe => vec_et("vmin_ge"),
        SemanticOp::AbsGtz => vec_et("vabs_gtz"),
        SemanticOp::NegGtz => vec_et("vneg_gtz"),
        SemanticOp::NegLtz => vec_et("vneg_ltz"),
        SemanticOp::NegAdd => vec_et("vnegadd"),
        SemanticOp::NegMul => vec_et("vnegmul"),
        SemanticOp::Accumulate => vec_et("vacc"),

        // Hardware state reads
        SemanticOp::ReadCycleCounter => "mov.cntr".into(),
        // Side-effect
        SemanticOp::CascadeRead => "vmov.scd".into(),
        SemanticOp::CascadeWrite => "vmov.mcd".into(),
        SemanticOp::StreamRead => {
            if slot_op.blocking { "stream.read.scl.blocking".into() }
            else { "stream.read.scl".into() }
        }
        SemanticOp::StreamWrite => {
            if slot_op.blocking { "stream.write.scl.blocking".into() }
            else { "stream.write.scl".into() }
        }
        SemanticOp::StreamWritePacketHeader => {
            if slot_op.blocking { "stream.write.ph.blocking".into() }
            else { "stream.write.ph".into() }
        }

        // Pointer
        SemanticOp::PointerAdd => "padd".into(),
        SemanticOp::PointerMov => "pmov".into(),

        _ => "?unknown".into(),
    }
}

fn operand_str(op: &Operand) -> String {
    match op {
        Operand::ScalarReg(r) => format!("r{}", r),
        Operand::VectorReg(r) => format!("v{}", r),
        Operand::AccumReg(r) => format!("acc{}", r),
        Operand::PointerReg(r) => format!("p{}", r),
        Operand::ModifierReg(r) => match r {
            0..=7 => format!("m{}", r),
            8..=15 => format!("dn{}", r - 8),
            16..=23 => format!("dj{}", r - 16),
            24..=31 => format!("dc{}", r - 24),
            _ => format!("mod{}", r),
        },
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
        Operand::ControlReg(id) => format!("cr{}", id),
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
        ElementType::Int64 => "i64",
        ElementType::UInt64 => "u64",
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
        BranchCondition::Zero => "z",
        BranchCondition::NotZero => "nz",
        BranchCondition::NotZeroDecrement => "nzd",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tablegen::SemanticOp;

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

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
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

        bundle.set_slot(
            SlotOp::from_semantic(SlotIndex::Control, SemanticOp::Br)
                .with_branch_condition(BranchCondition::Always),
        );
        assert!(bundle.has_control_flow());
    }

    #[test]
    fn test_disassemble_scalar() {
        let mut bundle = VliwBundle::empty();
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1))
            .with_source(Operand::ScalarReg(2));
        bundle.set_slot(op);

        assert_eq!(bundle.disassemble(), "add r0, r1, r2");
    }

    #[test]
    fn test_disassemble_vector() {
        let mut bundle = VliwBundle::empty();
        let op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Add)
            .as_vector(ElementType::Int32);
        bundle.set_slot(op);

        assert_eq!(bundle.disassemble(), "vadd.i32");
    }

    #[test]
    fn test_disassemble_vliw() {
        let mut bundle = VliwBundle::empty();
        bundle.set_slot(SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add));
        bundle.set_slot(
            SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Mul)
                .as_vector(ElementType::Int16),
        );

        let dis = bundle.disassemble();
        assert!(dis.starts_with("{"));
        assert!(dis.contains("add"));
        assert!(dis.contains("vmul.i16"));
    }
}
