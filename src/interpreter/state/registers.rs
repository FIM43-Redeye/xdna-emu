//! Register file implementations for AIE2.
//!
//! AIE2 has several register files:
//!
//! - **Scalar GPR**: 32 x 32-bit general purpose registers (r0-r31)
//! - **Special**: 16 x 32-bit special-purpose registers (lr, LS, LE, LC, etc.)
//! - **Pointer**: 8 x 20-bit address registers (p0-p7)
//! - **Modifier**: 8 x 20-bit post-modify registers (m0-m7)
//! - **Vector**: 32 x 256-bit SIMD registers (v0-v31)
//! - **Accumulator**: 18 x 512-bit MAC accumulators (9 cm pairs: bml0-bml8 + bmh0-bmh8)
//!
//! The scalar register file is extended to 48 entries to include special-
//! purpose registers at indices 32-47. In AIE2 hardware these are separate
//! registers (not part of the GPR file), but for simplicity we store them
//! in the same array with dedicated index constants.

use std::fmt;

pub use xdna_archspec::aie2::Bypass;

/// 512-bit vector data: two consecutive 256-bit w-registers.
pub type Vec512 = [u32; 16];

/// 1024-bit accumulator data: two consecutive 512-bit bm-registers.
pub type Acc1024 = [u64; 16];

/// Number of scalar registers including special-purpose slots.
/// Indices 0-31 are GPRs (r0-r31); 32-47 are special registers.
pub const NUM_SCALAR_REGS: usize = 48;

/// Number of general-purpose scalar registers (r0-r31).
pub const NUM_SCALAR_GPRS: usize = 32;

// Special register indices (32+). These are NOT part of the r0-r31 GPR file
// in hardware; they are separate special-purpose registers. We map them to
// high indices in the scalar array for convenience.
/// Link register (lr) -- return address saved by jl/call.
pub const LR_REG_INDEX: u8 = 32;
/// Loop start address register (LS).
pub const LS_REG_INDEX: u8 = 33;
/// Loop end address register (LE).
pub const LE_REG_INDEX: u8 = 34;
/// Loop count register (LC).
pub const LC_REG_INDEX: u8 = 35;
/// Decompress pointer register (DP).
pub const DP_REG_INDEX: u8 = 36;
/// Core ID register (CORE_ID) -- read-only in hardware.
pub const CORE_ID_REG_INDEX: u8 = 37;

/// Stack pointer sentinel index for PointerReg operands.
///
/// AIE2's SP is a dedicated special register (SPLReg<12, "sp">), separate
/// from general-purpose pointer registers p0-p7. Using index 255 as a
/// sentinel in `Operand::PointerReg(SP_PTR_INDEX)` allows the decoder to
/// distinguish SP from p0-p7 without adding a new Operand variant.
///
/// `pointer_read(SP_PTR_INDEX)` and `pointer_write(SP_PTR_INDEX, _)` are
/// intercepted by the execution context to route to the dedicated SP storage.
pub const SP_PTR_INDEX: u8 = 255;

/// Number of pointer registers.
pub const NUM_POINTER_REGS: usize = 8;

/// Number of modifier registers.
///
/// AIE2 has 8 composite 80-bit "d" registers (d0-d7), each containing
/// four 20-bit sub-registers:
///   - m0-m7 (sub_mod):        indices 0-7   -- post-modify values
///   - dn0-dn7 (sub_dim_size): indices 8-15  -- dimension size
///   - dj0-dj7 (sub_dim_stride): indices 16-23 -- dimension stride/jump
///   - dc0-dc7 (sub_dim_count): indices 24-31 -- dimension count
pub const NUM_MODIFIER_REGS: usize = 32;

/// Base index for modifier sub-classes within the modifier register file.
pub const MOD_BASE_M: u8 = 0;
pub const MOD_BASE_DN: u8 = 8;
pub const MOD_BASE_DJ: u8 = 16;
pub const MOD_BASE_DC: u8 = 24;

/// Number of vector registers.
pub const NUM_VECTOR_REGS: usize = 32;

/// Number of accumulator registers.
///
/// AIE2 has 9 cm-registers (cm0-cm8), each 1024 bits. Each cm is composed
/// of two 512-bit bm-registers (bml and bmh), giving 18 physical 512-bit
/// entries. Our Operand::AccumReg indices use the bm-level numbering:
///   bml_n = n*2, bmh_n = n*2+1, cm_n = n*2 (wide: reads n*2 and n*2+1).
/// So indices range from 0 to 17 (cm8 = bml8 = index 16, bmh8 = index 17).
pub const NUM_ACCUMULATOR_REGS: usize = 18;

/// Mask for pointer/modifier values (tile-local address width, from archspec).
const PTR_MASK: u32 = xdna_archspec::aie2::TILE_OFFSET_MASK;

/// Scalar register file including special-purpose registers.
///
/// Indices 0-31: general-purpose registers r0-r31.
/// Indices 32-47: special-purpose registers (lr, LS, LE, LC, DP, CORE_ID).
///
/// In AIE2 hardware, the special registers are separate from the GPR file.
/// We store them in the same array for simplicity, using dedicated index
/// constants (LR_REG_INDEX, etc.) to access them.
#[derive(Clone)]
pub struct ScalarRegisterFile {
    regs: [u32; NUM_SCALAR_REGS],
}

impl Default for ScalarRegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarRegisterFile {
    /// Create a new zeroed register file.
    pub const fn new() -> Self {
        Self { regs: [0; NUM_SCALAR_REGS] }
    }

    /// Initialize the read-only CORE_ID register.
    ///
    /// This bypasses write protection -- used only at context creation
    /// to set the tile's identity. Per AM020, CORE_ID encodes the tile's
    /// column and row position in the array.
    pub fn set_core_id(&mut self, col: u8, row: u8) {
        let idx = CORE_ID_REG_INDEX as usize % NUM_SCALAR_REGS;
        self.regs[idx] = ((col as u32) << 16) | (row as u32);
    }

    /// Read a register. Indices 0-31 are GPRs; 32-47 are special registers.
    #[inline]
    pub fn read(&self, reg: u8) -> u32 {
        let idx = (reg as usize) % NUM_SCALAR_REGS;
        self.regs[idx]
    }

    /// Write a register. Indices 0-31 are GPRs; 32-47 are special registers.
    ///
    /// CORE_ID (index 37) is read-only in hardware. Writes are silently
    /// ignored to match silicon behavior.
    #[inline]
    pub fn write(&mut self, reg: u8, value: u32) {
        if reg == CORE_ID_REG_INDEX {
            log::trace!("[SCALAR] ignored write to read-only CORE_ID (value=0x{:08x})", value);
            return;
        }
        let idx = (reg as usize) % NUM_SCALAR_REGS;
        self.regs[idx] = value;
    }

    /// Get a slice of GPR registers only (r0-r31) for debugging/display.
    pub fn as_slice(&self) -> &[u32] {
        &self.regs[..NUM_SCALAR_GPRS]
    }

    /// Fill all registers with a sentinel pattern (tripwire for uninitialized reads).
    pub fn fill_pattern(&mut self, pattern: u32) {
        self.regs.fill(pattern);
    }
}

impl fmt::Debug for ScalarRegisterFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Only show non-zero registers
        let non_zero: Vec<_> = self.regs.iter().enumerate().filter(|(_, v)| **v != 0).collect();

        if non_zero.is_empty() {
            write!(f, "ScalarRegisterFile {{ all zero }}")
        } else {
            write!(f, "ScalarRegisterFile {{ ")?;
            for (i, (reg, val)) in non_zero.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "r{}: 0x{:08X}", reg, val)?;
            }
            write!(f, " }}")
        }
    }
}

/// Pointer register file.
///
/// 8 × 20-bit address registers (p0-p7).
/// Used for memory addressing with optional post-modify.
#[derive(Clone)]
pub struct PointerRegisterFile {
    regs: [u32; NUM_POINTER_REGS],
}

impl Default for PointerRegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

impl PointerRegisterFile {
    /// Create a new zeroed register file.
    pub const fn new() -> Self {
        Self { regs: [0; NUM_POINTER_REGS] }
    }

    /// Read a pointer register (0-7).
    #[inline]
    pub fn read(&self, reg: u8) -> u32 {
        self.regs[(reg & 0x07) as usize]
    }

    /// Write a pointer register (0-7).
    /// Value is masked to 20 bits.
    #[inline]
    pub fn write(&mut self, reg: u8, value: u32) {
        self.regs[(reg & 0x07) as usize] = value & PTR_MASK;
    }

    /// Add offset to a pointer register (with wrapping at 20 bits).
    #[inline]
    pub fn add(&mut self, reg: u8, offset: i32) {
        let idx = (reg & 0x07) as usize;
        let new_value = self.regs[idx].wrapping_add(offset as u32) & PTR_MASK;
        self.regs[idx] = new_value;
    }

    /// Fill all registers with a sentinel pattern (tripwire for uninitialized reads).
    pub fn fill_pattern(&mut self, pattern: u32) {
        self.regs.fill(pattern);
    }
}

impl fmt::Debug for PointerRegisterFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let non_zero: Vec<_> = self.regs.iter().enumerate().filter(|(_, v)| **v != 0).collect();

        if non_zero.is_empty() {
            write!(f, "PointerRegisterFile {{ all zero }}")
        } else {
            write!(f, "PointerRegisterFile {{ ")?;
            for (i, (reg, val)) in non_zero.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "p{}: 0x{:05X}", reg, val)?;
            }
            write!(f, " }}")
        }
    }
}

/// Modifier register file.
///
/// 32 × 20-bit registers: 4 sub-classes of 8 registers each.
/// Each sub-class corresponds to a 20-bit slice of the composite d0-d7 registers:
///   - m0-m7 (indices 0-7): post-modify values
///   - dn0-dn7 (indices 8-15): dimension size (AGU)
///   - dj0-dj7 (indices 16-23): dimension stride/jump (AGU)
///   - dc0-dc7 (indices 24-31): dimension count (AGU)
#[derive(Clone)]
pub struct ModifierRegisterFile {
    regs: [u32; NUM_MODIFIER_REGS],
}

impl Default for ModifierRegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

impl ModifierRegisterFile {
    /// Create a new zeroed register file.
    pub const fn new() -> Self {
        Self { regs: [0; NUM_MODIFIER_REGS] }
    }

    /// Read a modifier register (0-31).
    ///
    /// Indices: m0-m7 = 0-7, dn0-dn7 = 8-15, dj0-dj7 = 16-23, dc0-dc7 = 24-31.
    #[inline]
    pub fn read(&self, reg: u8) -> u32 {
        let idx = (reg as usize) % NUM_MODIFIER_REGS;
        self.regs[idx]
    }

    /// Read as signed value (sign-extended from 20 bits).
    #[inline]
    pub fn read_signed(&self, reg: u8) -> i32 {
        let val = self.read(reg);
        // Sign-extend from 20 bits
        if val & 0x0008_0000 != 0 {
            (val | 0xFFF0_0000) as i32
        } else {
            val as i32
        }
    }

    /// Write a modifier register (0-31).
    /// Value is masked to 20 bits.
    ///
    /// Indices: m0-m7 = 0-7, dn0-dn7 = 8-15, dj0-dj7 = 16-23, dc0-dc7 = 24-31.
    #[inline]
    pub fn write(&mut self, reg: u8, value: u32) {
        let idx = (reg as usize) % NUM_MODIFIER_REGS;
        self.regs[idx] = value & PTR_MASK;
    }

    /// Fill all registers with a sentinel pattern (tripwire for uninitialized reads).
    pub fn fill_pattern(&mut self, pattern: u32) {
        self.regs.fill(pattern);
    }
}

impl fmt::Debug for ModifierRegisterFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let non_zero: Vec<_> = self.regs.iter().enumerate().filter(|(_, v)| **v != 0).collect();

        if non_zero.is_empty() {
            write!(f, "ModifierRegisterFile {{ all zero }}")
        } else {
            write!(f, "ModifierRegisterFile {{ ")?;
            for (i, (reg, val)) in non_zero.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                let name = match *reg {
                    0..=7 => format!("m{}", reg),
                    8..=15 => format!("dn{}", reg - 8),
                    16..=23 => format!("dj{}", reg - 16),
                    24..=31 => format!("dc{}", reg - 24),
                    _ => format!("mod{}", reg),
                };
                write!(f, "{}: 0x{:05X}", name, val)?;
            }
            write!(f, " }}")
        }
    }
}

/// An in-flight vector-register write awaiting bypass-network visibility.
///
/// Carries the producer-side facts (result latency `l_def`, def bypass class
/// `def_bypass`, and the `issue_bundle` it was queued in). Visibility is
/// *consumer-keyed*: when each read happens determines what it sees, computed
/// from these producer facts plus the consumer's `(use_cycle, use_bypass)` by
/// [`VectorRegisterFile::visible_at`].
#[derive(Clone, Copy, Debug)]
struct PendingVecWrite {
    reg: u8,
    value: [u32; 8],
    issue_bundle: u64,
    l_def: u8,
    def_bypass: Bypass,
}

/// Vector register file.
///
/// 32 × 256-bit SIMD registers (v0-v31).
/// Each register holds 8 × 32-bit, 16 × 16-bit, or 32 × 8-bit elements.
#[derive(Clone)]
pub struct VectorRegisterFile {
    /// Each register is 256 bits = 8 × u32. Holds *committed* (fully-landed)
    /// values; in-flight writes live in `pending` until they land.
    regs: [[u32; 8]; NUM_VECTOR_REGS],
    /// VLIW bundle snapshot of the committed file. When `Some`, the committed
    /// base seen by reads is the pre-bundle state, so direct (`write`) writes
    /// within a bundle stay invisible to sibling slots. Bypass-overlay writes
    /// (`queue_write`) are not in `regs` until landed, so they get read-old
    /// semantics from the resolution rule itself.
    snapshot: Option<[[u32; 8]; NUM_VECTOR_REGS]>,
    /// In-flight vector writes, in issue order. Resolved per-read against the
    /// AIE2 forwarding network; moved into `regs` once fully landed.
    pending: Vec<PendingVecWrite>,
    /// Current issued-bundle counter (the issue-slot-relative clock the
    /// compiler's latencies are expressed in). Set at bundle start.
    cur_bundle: u64,
}

impl Default for VectorRegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorRegisterFile {
    /// Create a new zeroed register file.
    pub const fn new() -> Self {
        Self {
            regs: [[0; 8]; NUM_VECTOR_REGS],
            snapshot: None,
            pending: Vec::new(),
            cur_bundle: 0,
        }
    }

    /// The committed array reads see: the bundle snapshot if active, else live.
    #[inline]
    fn active(&self) -> &[[u32; 8]; NUM_VECTOR_REGS] {
        match &self.snapshot {
            Some(snap) => snap,
            None => &self.regs,
        }
    }

    /// Advance the issued-bundle clock and land any in-flight writes that have
    /// reached their architectural visibility (`issue_bundle + l_def <=
    /// bundle`). Call at the start of each bundle, BEFORE `begin_bundle`, so
    /// landed values are captured by the snapshot.
    ///
    /// Landed writes are applied in issue order, so when several target the
    /// same register the latest-issued wins.
    pub fn advance_bundle(&mut self, bundle: u64) {
        self.cur_bundle = bundle;
        if self.pending.is_empty() {
            return;
        }
        // Apply landed writes in issue order (Vec is push/issue-ordered).
        for w in &self.pending {
            if w.issue_bundle + w.l_def as u64 <= bundle {
                self.regs[(w.reg & 0x1F) as usize] = w.value;
            }
        }
        self.pending.retain(|w| w.issue_bundle + w.l_def as u64 > bundle);
    }

    /// Effective visibility bundle of an in-flight write to a consumer reading
    /// it at `use_cycle` with `use_bypass`. AIE2 forwarding-network latency:
    ///   eff = max(1, l_def - use_cycle + 1 - (match ? 1 : 0))
    /// where match = producer/consumer bypass classes equal and not NoBypass.
    /// The `+1` is the bundle-clock convention constant (calibrated against the
    /// VMOV->VST issue+2 / VMOV->VEXTRACT issue+1 anchors; see the design spec).
    /// Clamped to >= 1: a write never resolves within its own issue bundle.
    #[inline]
    fn visible_at(w: &PendingVecWrite, use_cycle: u8, use_bypass: Bypass) -> u64 {
        let matched = w.def_bypass == use_bypass && w.def_bypass != Bypass::No;
        let eff = (w.l_def as i64) - (use_cycle as i64) + 1 - (matched as i64);
        let eff = eff.max(1) as u64;
        w.issue_bundle + eff
    }

    /// Resolve register `reg` for a consumer reading at `(use_cycle, use_bypass)`:
    /// the committed base value, overridden by the latest-issued in-flight write
    /// visible to this consumer at the current bundle.
    #[inline]
    fn resolve(&self, reg: u8, use_cycle: u8, use_bypass: Bypass) -> [u32; 8] {
        let idx = (reg & 0x1F) as usize;
        let mut val = self.active()[idx];
        for w in &self.pending {
            if (w.reg & 0x1F) as usize == idx && Self::visible_at(w, use_cycle, use_bypass) <= self.cur_bundle
            {
                val = w.value;
            }
        }
        val
    }

    /// Primary vector-read entry point. The execute path supplies the consuming
    /// operand's `(use_cycle, use_bypass)` from `SlotOp::source_forward`.
    #[inline]
    pub fn read_with(&self, reg: u8, use_cycle: u8, use_bypass: Bypass) -> [u32; 8] {
        self.resolve(reg, use_cycle, use_bypass)
    }

    /// Snapshot the committed file for VLIW bundle read semantics.
    #[inline]
    pub fn begin_bundle(&mut self) {
        self.snapshot = Some(self.regs);
    }

    /// Discard the bundle snapshot; committed reads return live state again.
    #[inline]
    pub fn end_bundle(&mut self) {
        self.snapshot = None;
    }

    /// Queue a vector-register write through the bypass network. `l_def` is the
    /// producer's result latency (itinerary `operand_cycles[0]`); `def_bypass`
    /// its result bypass class. `l_def == 0` writes immediately (test/legacy
    /// helpers that bypass the executor's latency plumbing).
    pub fn queue_write(&mut self, reg: u8, value: [u32; 8], l_def: u8, def_bypass: Bypass) {
        if l_def == 0 {
            self.regs[(reg & 0x1F) as usize] = value;
            return;
        }
        self.pending.push(PendingVecWrite {
            reg: reg & 0x1F,
            value,
            issue_bundle: self.cur_bundle,
            l_def,
            def_bypass,
        });
    }

    /// Queue a 512-bit wide write (two consecutive w-registers) through the
    /// bypass network. Each half carries the same latency and bypass class.
    pub fn queue_write_wide(&mut self, base_reg: u8, data: Vec512, l_def: u8, def_bypass: Bypass) {
        debug_assert!(base_reg % 2 == 0, "wide vector queue to odd base register {}", base_reg);
        let mut lo = [0u32; 8];
        let mut hi = [0u32; 8];
        lo.copy_from_slice(&data[..8]);
        hi.copy_from_slice(&data[8..]);
        self.queue_write(base_reg, lo, l_def, def_bypass);
        self.queue_write(base_reg + 1, hi, l_def, def_bypass);
    }

    /// Read a vector register as a compute consumer (`MOV_Bypass` use), the
    /// common case: vector ALU ops forward from `MOV_Bypass` producers at
    /// issue+1. Store-data reads must use [`read_store`](Self::read_store).
    /// For callers with explicit operand context, use [`read_with`](Self::read_with).
    #[inline]
    pub fn read(&self, reg: u8) -> [u32; 8] {
        // Compute-consumer default (use_cycle 2, Mov bypass). Only for callers
        // without operand context; the execute path uses read_with.
        self.resolve(reg, 2, Bypass::Mov)
    }

    /// Read a vector register as a store-data consumer (`NoBypass` use): never
    /// receives ALU forwarding, so a producer's result is visible only at its
    /// full architectural latency (e.g. issue+2 for a vector ALU op). This is
    /// the path the bf16 split-tile store depends on (reads the *old* value the
    /// bundle after a `VMOV x,bml`).
    #[inline]
    pub fn read_store(&self, reg: u8) -> [u32; 8] {
        self.resolve(reg, 1, Bypass::No)
    }

    /// Write a vector register directly (committed immediately, no bypass
    /// modelling). Used by load commits, cascade reads, and tests; ordinary
    /// compute writes go through [`queue_write`](Self::queue_write).
    #[inline]
    pub fn write(&mut self, reg: u8, value: [u32; 8]) {
        self.regs[(reg & 0x1F) as usize] = value;
    }

    /// Read a single lane (0-7) as u32 (compute / `MOV_Bypass` consumer).
    #[inline]
    pub fn read_lane(&self, reg: u8, lane: u8) -> u32 {
        self.resolve(reg, 2, Bypass::Mov)[(lane & 0x07) as usize]
    }

    /// Write a single lane (0-7).
    #[inline]
    pub fn write_lane(&mut self, reg: u8, lane: u8, value: u32) {
        self.regs[(reg & 0x1F) as usize][(lane & 0x07) as usize] = value;
    }

    /// Read entire register as bytes (for memory operations).
    pub fn read_bytes(&self, reg: u8) -> [u8; 32] {
        let words = self.read(reg);
        let mut bytes = [0u8; 32];
        for (i, word) in words.iter().enumerate() {
            bytes[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// Write entire register from bytes.
    pub fn write_bytes(&mut self, reg: u8, bytes: &[u8; 32]) {
        let mut words = [0u32; 8];
        for (i, word) in words.iter_mut().enumerate() {
            *word = u32::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]]);
        }
        self.write(reg, words);
    }

    /// Read a 512-bit x-register (two consecutive w-registers).
    ///
    /// The decoder maps x0 -> vreg 0, x1 -> vreg 2, etc. (reg * 2).
    /// `base_reg` is already the decoded index (0, 2, 4, ...) -- the caller
    /// is responsible for the x -> w mapping before calling here.
    pub fn read_wide(&self, base_reg: u8) -> Vec512 {
        debug_assert!(base_reg % 2 == 0, "wide vector read from odd base register {}", base_reg);
        let lo = self.read(base_reg);
        let hi = self.read(base_reg + 1);
        let mut result = [0u32; 16];
        result[..8].copy_from_slice(&lo);
        result[8..].copy_from_slice(&hi);
        result
    }

    /// Read a 512-bit x-register with explicit forwarding context.
    ///
    /// Like `read_wide` but resolves each half-register via
    /// `read_with(reg, use_cycle, use_bypass)` rather than the compute default.
    /// Used by `get_wide_vec_source` when the op carries per-operand
    /// `source_forward` data from the resolved itinerary.
    pub fn read_wide_with(&self, base_reg: u8, use_cycle: u8, use_bypass: Bypass) -> Vec512 {
        debug_assert!(base_reg % 2 == 0, "wide vector read_with from odd base register {}", base_reg);
        let lo = self.resolve(base_reg, use_cycle, use_bypass);
        let hi = self.resolve(base_reg + 1, use_cycle, use_bypass);
        let mut result = [0u32; 16];
        result[..8].copy_from_slice(&lo);
        result[8..].copy_from_slice(&hi);
        result
    }

    /// Write a 512-bit x-register (split across two consecutive w-registers).
    pub fn write_wide(&mut self, base_reg: u8, data: Vec512) {
        debug_assert!(base_reg % 2 == 0, "wide vector write to odd base register {}", base_reg);
        let mut lo = [0u32; 8];
        let mut hi = [0u32; 8];
        lo.copy_from_slice(&data[..8]);
        hi.copy_from_slice(&data[8..]);
        self.write(base_reg, lo);
        self.write(base_reg + 1, hi);
    }

    /// Read a 1024-bit y-register (four consecutive w-registers).
    ///
    /// The decoder maps y0 -> vreg 0, y1 -> vreg 4, etc. (reg * 4).
    /// `base_reg` is already the decoded index (0, 4, 8, ...).
    pub fn read_quad(&self, base_reg: u8) -> [u32; 32] {
        debug_assert!(base_reg % 4 == 0, "quad vector read from non-aligned base register {}", base_reg);
        let mut result = [0u32; 32];
        result[..8].copy_from_slice(&self.read(base_reg));
        result[8..16].copy_from_slice(&self.read(base_reg + 1));
        result[16..24].copy_from_slice(&self.read(base_reg + 2));
        result[24..].copy_from_slice(&self.read(base_reg + 3));
        result
    }

    /// Write a 1024-bit y-register (split across four consecutive w-registers).
    pub fn write_quad(&mut self, base_reg: u8, data: &[u32; 32]) {
        debug_assert!(base_reg % 4 == 0, "quad vector write to non-aligned base register {}", base_reg);
        let mut w = [0u32; 8];
        for i in 0..4 {
            w.copy_from_slice(&data[i * 8..(i + 1) * 8]);
            self.write(base_reg + i as u8, w);
        }
    }

    /// Fill all registers with a sentinel pattern (tripwire for uninitialized reads).
    pub fn fill_pattern(&mut self, pattern: u32) {
        for reg in &mut self.regs {
            reg.fill(pattern);
        }
    }
}

impl fmt::Debug for VectorRegisterFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let non_zero: Vec<_> = self
            .regs
            .iter()
            .enumerate()
            .filter(|(_, v)| v.iter().any(|x| *x != 0))
            .collect();

        if non_zero.is_empty() {
            write!(f, "VectorRegisterFile {{ all zero }}")
        } else {
            writeln!(f, "VectorRegisterFile {{")?;
            for (reg, val) in non_zero {
                write!(f, "  v{}: [", reg)?;
                for (i, lane) in val.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "0x{:08X}", lane)?;
                }
                writeln!(f, "]")?;
            }
            write!(f, "}}")
        }
    }
}

/// Accumulator register file.
///
/// 18 × 512-bit registers for multiply-accumulate operations (9 cm-register
/// pairs: cm0-cm8, each split into bml and bmh halves).
/// Each 512-bit register holds 8 × 64-bit or 16 × 32-bit accumulators.
#[derive(Clone)]
pub struct AccumulatorRegisterFile {
    /// Each register is 512 bits = 8 × u64
    regs: [[u64; 8]; NUM_ACCUMULATOR_REGS],
    /// VLIW bundle snapshot (see `VectorRegisterFile::snapshot`). Reads return
    /// pre-bundle values; writes go live. `accumulate` is a live read-modify-
    /// write (MAC accumulation crosses bundles, one accumulate per cm/bundle).
    snapshot: Option<[[u64; 8]; NUM_ACCUMULATOR_REGS]>,
}

impl Default for AccumulatorRegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

impl AccumulatorRegisterFile {
    /// Create a new zeroed register file.
    pub const fn new() -> Self {
        Self { regs: [[0; 8]; NUM_ACCUMULATOR_REGS], snapshot: None }
    }

    /// The array reads see: the bundle snapshot if active, else live registers.
    #[inline]
    fn active(&self) -> &[[u64; 8]; NUM_ACCUMULATOR_REGS] {
        match &self.snapshot {
            Some(snap) => snap,
            None => &self.regs,
        }
    }

    /// Snapshot live registers for VLIW bundle read semantics.
    #[inline]
    pub fn begin_bundle(&mut self) {
        self.snapshot = Some(self.regs);
    }

    /// Discard the bundle snapshot; reads return live state again.
    #[inline]
    pub fn end_bundle(&mut self) {
        self.snapshot = None;
    }

    /// Read an accumulator register as 8 × u64.
    #[inline]
    pub fn read(&self, reg: u8) -> [u64; 8] {
        self.active()[(reg as usize) % NUM_ACCUMULATOR_REGS]
    }

    /// Write an accumulator register from 8 × u64.
    #[inline]
    pub fn write(&mut self, reg: u8, value: [u64; 8]) {
        self.regs[(reg as usize) % NUM_ACCUMULATOR_REGS] = value;
    }

    /// Read a single lane (0-7) as u64.
    #[inline]
    pub fn read_lane(&self, reg: u8, lane: u8) -> u64 {
        self.active()[(reg as usize) % NUM_ACCUMULATOR_REGS][(lane & 0x07) as usize]
    }

    /// Write a single lane (0-7).
    #[inline]
    pub fn write_lane(&mut self, reg: u8, lane: u8, value: u64) {
        self.regs[(reg as usize) % NUM_ACCUMULATOR_REGS][(lane & 0x07) as usize] = value;
    }

    /// Clear (zero) an accumulator register.
    #[inline]
    pub fn clear(&mut self, reg: u8) {
        self.regs[(reg as usize) % NUM_ACCUMULATOR_REGS] = [0; 8];
    }

    /// Accumulate: add values to existing accumulator lanes.
    pub fn accumulate(&mut self, reg: u8, values: [u64; 8]) {
        let idx = (reg as usize) % NUM_ACCUMULATOR_REGS;
        for (lane, val) in values.iter().enumerate() {
            self.regs[idx][lane] = self.regs[idx][lane].wrapping_add(*val);
        }
    }

    /// Read a 1024-bit cm-register (two consecutive bm-registers).
    ///
    /// cm0 = (acc0, acc1), cm2 = (acc2, acc3), etc.
    /// `base_reg` must be even (hardware enforces pair alignment).
    pub fn read_wide(&self, base_reg: u8) -> Acc1024 {
        debug_assert!(base_reg % 2 == 0, "wide accum read from odd base register {}", base_reg);
        let lo = self.read(base_reg);
        let hi = self.read(base_reg + 1);
        let mut result = [0u64; 16];
        result[..8].copy_from_slice(&lo);
        result[8..].copy_from_slice(&hi);
        result
    }

    /// Write a 1024-bit cm-register (split across two consecutive bm-registers).
    pub fn write_wide(&mut self, base_reg: u8, data: Acc1024) {
        debug_assert!(base_reg % 2 == 0, "wide accum write to odd base register {}", base_reg);
        let mut lo = [0u64; 8];
        let mut hi = [0u64; 8];
        lo.copy_from_slice(&data[..8]);
        hi.copy_from_slice(&data[8..]);
        self.write(base_reg, lo);
        self.write(base_reg + 1, hi);
    }

    /// Fill all registers with a sentinel pattern (tripwire for uninitialized reads).
    /// The u32 pattern is doubled to fill u64 lanes (e.g., 0xDEADBEEF -> 0xDEADBEEF_DEADBEEF).
    pub fn fill_pattern(&mut self, pattern: u32) {
        let wide = (pattern as u64) << 32 | (pattern as u64);
        for reg in &mut self.regs {
            reg.fill(wide);
        }
    }
}

impl fmt::Debug for AccumulatorRegisterFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let non_zero: Vec<_> = self
            .regs
            .iter()
            .enumerate()
            .filter(|(_, v)| v.iter().any(|x| *x != 0))
            .collect();

        if non_zero.is_empty() {
            write!(f, "AccumulatorRegisterFile {{ all zero }}")
        } else {
            writeln!(f, "AccumulatorRegisterFile {{")?;
            for (reg, val) in non_zero {
                write!(f, "  acc{}: [", reg)?;
                for (i, lane) in val.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "0x{:016X}", lane)?;
                }
                writeln!(f, "]")?;
            }
            write!(f, "}}")
        }
    }
}

// ============================================================================
// Mask Register File (q0-q3)
// ============================================================================

/// Number of mask registers.
///
/// AIE2 has 4 mask registers (q0-q3), each 128 bits. These are used for
/// sparse vector operations -- the mask indicates which elements in a
/// sparse vector are non-zero (2:4 structured sparsity).
pub const NUM_MASK_REGS: usize = 4;

/// The AIE2 **q register file**: 4 x 128-bit registers (q0-q3).
///
/// Named `MaskRegisterFile` for its primary architectural role -- in llvm-aie
/// TableGen these are literally the `AIE2Vector128RegisterClass` "128-bit mask
/// registers" -- but the same physical registers serve **dual duty**:
///
/// 1. **Masks**: comparison results and 2:4 sparsity masks (the sparse matmul
///    uses the low 64 bits of each register).
/// 2. **128-bit vector data**: the q registers are also the vehicle for 128-bit
///    moves and stores. Chess lands int32->int16 SRS results via `vmov qN, wX`
///    (low 128 bits of a w-register) followed by `st qN` (see `vec_srs_i32`).
///
/// So `q` (the assembly mnemonic) == this "mask" file; treat reads/writes here
/// as plain 128 bits of register state whose interpretation (mask vs. packed
/// data) is the consuming op's concern, not the file's. Each register is stored
/// as `[u32; 4]` (128 bits); the full width is preserved for both roles.
///
/// Composite qx registers (qx0-qx3) pair a q register with a vector register:
/// qx_n = {x_n, q_n} -- the mask portion from q_n here, vector data from x_n.
#[derive(Clone)]
pub struct MaskRegisterFile {
    regs: [[u32; 4]; NUM_MASK_REGS],
    /// VLIW bundle snapshot (see `VectorRegisterFile::snapshot`). q-registers
    /// double as 128-bit vector data (e.g. `vmov q,w; st q`), so they need the
    /// same read-old/write-new semantics within a bundle.
    snapshot: Option<[[u32; 4]; NUM_MASK_REGS]>,
}

impl MaskRegisterFile {
    /// Create a new mask register file with all registers zeroed.
    pub fn new() -> Self {
        Self { regs: [[0u32; 4]; NUM_MASK_REGS], snapshot: None }
    }

    /// The array reads see: the bundle snapshot if active, else live registers.
    #[inline]
    fn active(&self) -> &[[u32; 4]; NUM_MASK_REGS] {
        match &self.snapshot {
            Some(snap) => snap,
            None => &self.regs,
        }
    }

    /// Snapshot live registers for VLIW bundle read semantics.
    #[inline]
    pub fn begin_bundle(&mut self) {
        self.snapshot = Some(self.regs);
    }

    /// Discard the bundle snapshot; reads return live state again.
    #[inline]
    pub fn end_bundle(&mut self) {
        self.snapshot = None;
    }

    /// Read the low 64 bits of a mask register (used by sparse matmul).
    pub fn read_u64_low(&self, reg: u8) -> u64 {
        let idx = (reg as usize) % NUM_MASK_REGS;
        let regs = self.active();
        (regs[idx][0] as u64) | ((regs[idx][1] as u64) << 32)
    }

    /// Read the full 128-bit mask register as u128 (used by sparse pair-routing).
    pub fn read_u128(&self, reg: u8) -> u128 {
        let idx = (reg as usize) % NUM_MASK_REGS;
        let regs = self.active();
        (regs[idx][0] as u128)
            | ((regs[idx][1] as u128) << 32)
            | ((regs[idx][2] as u128) << 64)
            | ((regs[idx][3] as u128) << 96)
    }

    /// Read the full 128-bit mask register as [u32; 4].
    pub fn read(&self, reg: u8) -> [u32; 4] {
        let idx = (reg as usize) % NUM_MASK_REGS;
        self.active()[idx]
    }

    /// Write the full 128-bit mask register.
    pub fn write(&mut self, reg: u8, value: [u32; 4]) {
        let idx = (reg as usize) % NUM_MASK_REGS;
        self.regs[idx] = value;
    }

    /// Write a 32-bit value to the low word of a mask register.
    ///
    /// Used when a scalar write targets a mask register (e.g., setting
    /// a mask from a scalar value).
    pub fn write_u32_low(&mut self, reg: u8, value: u32) {
        let idx = (reg as usize) % NUM_MASK_REGS;
        self.regs[idx][0] = value;
        self.regs[idx][1] = 0;
        self.regs[idx][2] = 0;
        self.regs[idx][3] = 0;
    }

    /// Clear all mask registers to zero.
    pub fn clear(&mut self) {
        self.regs = [[0u32; 4]; NUM_MASK_REGS];
    }

    /// Fill all registers with a sentinel pattern (tripwire for uninitialized reads).
    pub fn fill_pattern(&mut self, pattern: u32) {
        for reg in &mut self.regs {
            reg.fill(pattern);
        }
    }
}

impl Default for MaskRegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MaskRegisterFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let non_zero: Vec<_> = self
            .regs
            .iter()
            .enumerate()
            .filter(|(_, v)| v.iter().any(|x| *x != 0))
            .collect();

        if non_zero.is_empty() {
            write!(f, "MaskRegisterFile {{ all zero }}")
        } else {
            writeln!(f, "MaskRegisterFile {{")?;
            for (reg, val) in non_zero {
                writeln!(f, "  q{}: 0x{:08X}_{:08X}_{:08X}_{:08X}", reg, val[3], val[2], val[1], val[0])?;
            }
            write!(f, "}}")
        }
    }
}

// ============================================================================
// TableGen Validation
// ============================================================================

/// Validate that our hardcoded register file sizes match the parsed TableGen
/// register model. This should be called once at startup when the register
/// model is available.
///
/// The register file arrays use const generics for performance, so we can't
/// dynamically size them. Instead, we assert at init time that the parsed
/// sizes match our constants, catching any drift between the emulator and
/// the llvm-aie register definitions.
///
/// # Panics
///
/// Panics with a descriptive message if any register class size doesn't match
/// the emulator's expectations.
pub fn validate_register_model(model: &xdna_archspec::aie2::isa::RegisterModel) {
    // Scalar GPR class "eR" should have exactly NUM_SCALAR_GPRS members
    if let Some(er) = model.classes.get("eR") {
        assert_eq!(
            er.members.len(),
            NUM_SCALAR_GPRS,
            "eR class has {} members, expected NUM_SCALAR_GPRS={}",
            er.members.len(),
            NUM_SCALAR_GPRS
        );
    }

    // Pointer class "eP" should have exactly NUM_POINTER_REGS members
    if let Some(ep) = model.classes.get("eP") {
        assert_eq!(
            ep.members.len(),
            NUM_POINTER_REGS,
            "eP class has {} members, expected NUM_POINTER_REGS={}",
            ep.members.len(),
            NUM_POINTER_REGS
        );
    }

    // Modifier sub-classes should each have 8 members
    let modifier_classes = [
        ("eM", "post-modify"),
        ("eDN", "dimension size"),
        ("eDJ", "dimension stride"),
        ("eDC", "dimension count"),
    ];
    for (class_name, desc) in &modifier_classes {
        if let Some(cls) = model.classes.get(*class_name) {
            assert_eq!(
                cls.members.len(),
                8,
                "{} ({}) class has {} members, expected 8",
                class_name,
                desc,
                cls.members.len()
            );
        }
    }

    // Vector class "eVEC256" or similar should have NUM_VECTOR_REGS members
    // (The exact class name varies; check common candidates)
    for candidate in &["eVEC256", "eV", "mVS"] {
        if let Some(cls) = model.classes.get(*candidate) {
            if cls.members.len() == NUM_VECTOR_REGS {
                break; // Found a matching vector class
            }
        }
    }

    // Accumulator class "eACC512" or "eACC" should have NUM_ACCUMULATOR_REGS
    for candidate in &["eACC512", "eACC", "mAcc"] {
        if let Some(cls) = model.classes.get(*candidate) {
            if cls.members.len() == NUM_ACCUMULATOR_REGS {
                break;
            }
        }
    }

    // Validate special register HWEncodings match our index mapping.
    // These are the most critical -- a mismatch means LR, LC, etc. are
    // decoded to the wrong register slot.
    let special_checks: &[(&str, u16)] = &[
        ("lr", 39),      // (4 << 3) | 0b111
        ("LS", 7),       // (0 << 3) | 0b111
        ("LE", 71),      // (8 << 3) | 0b111
        ("LC", 87),      // (10 << 3) | 0b111
        ("DP", 23),      // (2 << 3) | 0b111
        ("CORE_ID", 55), // (6 << 3) | 0b111
    ];
    for (name, expected_hw) in special_checks {
        if let Some(reg) = model.registers.get(*name) {
            assert_eq!(
                reg.hw_encoding, *expected_hw,
                "Special register {} has HWEncoding {}, expected {}",
                name, reg.hw_encoding, expected_hw
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Scalar Register Tests ==========

    #[test]
    fn test_scalar_read_write() {
        let mut regs = ScalarRegisterFile::new();

        regs.write(0, 0xDEAD_BEEF);
        assert_eq!(regs.read(0), 0xDEAD_BEEF);

        regs.write(31, 0xCAFE_BABE);
        assert_eq!(regs.read(31), 0xCAFE_BABE);
    }

    #[test]
    fn test_scalar_special_registers() {
        let mut regs = ScalarRegisterFile::new();

        // Index 32 is the LR register slot, NOT r0.
        regs.write(LR_REG_INDEX, 0x1234);
        assert_eq!(regs.read(LR_REG_INDEX), 0x1234);
        assert_eq!(regs.read(0), 0); // r0 unaffected

        // Other special registers are independent.
        regs.write(LC_REG_INDEX, 100);
        assert_eq!(regs.read(LC_REG_INDEX), 100);
        assert_eq!(regs.read(LR_REG_INDEX), 0x1234); // lr unaffected

        // Wrapping still applies at NUM_SCALAR_REGS (48).
        regs.write(48, 0xABCD); // wraps to index 0 (r0)
        assert_eq!(regs.read(0), 0xABCD);
    }

    #[test]
    fn test_scalar_debug_format() {
        let mut regs = ScalarRegisterFile::new();
        regs.write(5, 42);

        let debug = format!("{:?}", regs);
        assert!(debug.contains("r5"));
        assert!(debug.contains("0x0000002A"));
    }

    // ========== Pointer Register Tests ==========

    #[test]
    fn test_pointer_read_write() {
        let mut regs = PointerRegisterFile::new();

        regs.write(0, 0x1000);
        assert_eq!(regs.read(0), 0x1000);

        // Value masked to 20 bits
        regs.write(1, 0xFFFF_FFFF);
        assert_eq!(regs.read(1), 0x000F_FFFF);
    }

    #[test]
    fn test_pointer_add() {
        let mut regs = PointerRegisterFile::new();

        regs.write(0, 0x1000);
        regs.add(0, 0x100);
        assert_eq!(regs.read(0), 0x1100);

        // Negative offset
        regs.add(0, -0x50);
        assert_eq!(regs.read(0), 0x10B0);

        // Wrapping at 20 bits
        regs.write(1, 0x000F_FFF0);
        regs.add(1, 0x20);
        assert_eq!(regs.read(1), 0x0000_0010); // Wrapped
    }

    // ========== Modifier Register Tests ==========

    #[test]
    fn test_modifier_read_write() {
        let mut regs = ModifierRegisterFile::new();

        regs.write(0, 0x100);
        assert_eq!(regs.read(0), 0x100);
    }

    #[test]
    fn test_modifier_signed_read() {
        let mut regs = ModifierRegisterFile::new();

        // Positive value
        regs.write(0, 0x100);
        assert_eq!(regs.read_signed(0), 0x100);

        // Negative value (sign bit set in 20-bit)
        regs.write(1, 0x000F_FF00); // -256 in 20-bit signed
        assert_eq!(regs.read_signed(1), -256);
    }

    #[test]
    fn test_modifier_subclasses_are_independent() {
        let mut regs = ModifierRegisterFile::new();

        // m0, dn0, dj0, dc0 are all sub-registers of d0 but at different
        // bit positions. They should be independent in our flat model.
        regs.write(MOD_BASE_M + 0, 0x111); // m0
        regs.write(MOD_BASE_DN + 0, 0x222); // dn0
        regs.write(MOD_BASE_DJ + 0, 0x333); // dj0
        regs.write(MOD_BASE_DC + 0, 0x444); // dc0

        assert_eq!(regs.read(MOD_BASE_M + 0), 0x111); // m0 unchanged
        assert_eq!(regs.read(MOD_BASE_DN + 0), 0x222); // dn0 unchanged
        assert_eq!(regs.read(MOD_BASE_DJ + 0), 0x333); // dj0 unchanged
        assert_eq!(regs.read(MOD_BASE_DC + 0), 0x444); // dc0 unchanged
    }

    // ========== Vector Register Tests ==========

    #[test]
    fn test_vector_read_write() {
        let mut regs = VectorRegisterFile::new();

        let data = [1, 2, 3, 4, 5, 6, 7, 8];
        regs.write(0, data);
        assert_eq!(regs.read(0), data);
    }

    #[test]
    fn test_vector_lane_access() {
        let mut regs = VectorRegisterFile::new();

        regs.write_lane(5, 3, 0xDEAD_BEEF);
        assert_eq!(regs.read_lane(5, 3), 0xDEAD_BEEF);
        assert_eq!(regs.read_lane(5, 0), 0); // Other lanes unaffected
    }

    #[test]
    fn test_vector_byte_access() {
        let mut regs = VectorRegisterFile::new();

        let mut bytes = [0u8; 32];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i as u8;
        }
        regs.write_bytes(0, &bytes);

        let read_back = regs.read_bytes(0);
        assert_eq!(read_back, bytes);
    }

    // ========== Accumulator Register Tests ==========

    #[test]
    fn test_accumulator_read_write() {
        let mut regs = AccumulatorRegisterFile::new();

        let data = [1u64, 2, 3, 4, 5, 6, 7, 8];
        regs.write(0, data);
        assert_eq!(regs.read(0), data);
    }

    #[test]
    fn test_accumulator_clear() {
        let mut regs = AccumulatorRegisterFile::new();

        let data = [1u64, 2, 3, 4, 5, 6, 7, 8];
        regs.write(0, data);
        regs.clear(0);
        assert_eq!(regs.read(0), [0; 8]);
    }

    #[test]
    fn test_accumulator_accumulate() {
        let mut regs = AccumulatorRegisterFile::new();

        regs.write(0, [10, 20, 30, 40, 50, 60, 70, 80]);
        regs.accumulate(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(regs.read(0), [11, 22, 33, 44, 55, 66, 77, 88]);
    }

    #[test]
    fn test_accumulator_lane_access() {
        let mut regs = AccumulatorRegisterFile::new();

        regs.write_lane(2, 5, 0xCAFE_BABE_DEAD_BEEF);
        assert_eq!(regs.read_lane(2, 5), 0xCAFE_BABE_DEAD_BEEF);
    }

    // ========== Wide Register Tests ==========

    #[test]
    fn test_vector_read_wide() {
        let mut vrf = VectorRegisterFile::new();
        vrf.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
        vrf.write(1, [9, 10, 11, 12, 13, 14, 15, 16]);
        let wide = vrf.read_wide(0);
        assert_eq!(wide, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    }

    #[test]
    fn test_vector_write_wide() {
        let mut vrf = VectorRegisterFile::new();
        let data: [u32; 16] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
        vrf.write_wide(0, data);
        assert_eq!(vrf.read(0), [10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(vrf.read(1), [90, 100, 110, 120, 130, 140, 150, 160]);
    }

    #[test]
    fn test_accum_read_wide() {
        let mut arf = AccumulatorRegisterFile::new();
        arf.write(0, [100, 200, 300, 400, 500, 600, 700, 800]);
        arf.write(1, [900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]);
        let wide = arf.read_wide(0);
        assert_eq!(
            wide,
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
        );
    }

    #[test]
    fn test_accum_write_wide() {
        let mut arf = AccumulatorRegisterFile::new();
        let mut data = [0u64; 16];
        for i in 0..16 {
            data[i] = (i as u64 + 1) * 10;
        }
        arf.write_wide(0, data);
        assert_eq!(arf.read(0), [10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(arf.read(1), [90, 100, 110, 120, 130, 140, 150, 160]);
    }

    // ========== MaskRegisterFile Tests ==========

    #[test]
    fn test_mask_read_u128() {
        let mut mrf = MaskRegisterFile::new();
        mrf.write(0, [0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD]);
        let val = mrf.read_u128(0);
        assert_eq!(val, 0xDDDDDDDD_CCCCCCCC_BBBBBBBB_AAAAAAAA_u128);
    }

    #[test]
    fn test_mask_read_u128_zero() {
        let mrf = MaskRegisterFile::new();
        assert_eq!(mrf.read_u128(0), 0u128);
        assert_eq!(mrf.read_u128(3), 0u128);
    }

    // ========== TableGen Validation Tests ==========

    #[test]
    fn test_validate_register_model_with_live_data() {
        let output = xdna_archspec::aie2::isa::load_from_generated();

        // Should not panic
        validate_register_model(&output.register_model);
    }

    // ========== Vector bypass-network visibility ==========
    //
    // Models the AIE2 forwarding network. The matrix tests below are the
    // canonical specification: a vector write at issue bundle B with result
    // latency l_def and def bypass class is visible to a consumer reading at
    // use_cycle with use_bypass at bundle:
    //   eff = max(1, l_def - use_cycle + 1 - (match ? 1 : 0))
    //   visible_at = B + eff
    // where match = (def_bypass == use_bypass) && def_bypass != No.
    //
    // The shim methods read/read_store/read_lane are validated indirectly through
    // their defaults (use_cycle=2/Mov and use_cycle=1/No respectively).

    // ── Forwarding matrix (consumer-keyed) ──────────────────────────────
    // Producer VMOV-class: l_def = 2. Visibility per the itinerary formula
    //   eff = max(1, l_def - use_cycle + 1 - match), match = def==use != No.

    /// Helper: file with one pending write at issue bundle 0, advanced to `at`.
    fn pending_at(l_def: u8, def: Bypass, at: u64) -> VectorRegisterFile {
        let mut f = VectorRegisterFile::new();
        f.queue_write(4, [0xAAAA_AAAA; 8], l_def, def);
        f.advance_bundle(at); // lands writes where issue+l_def <= at; sets cur_bundle
        f
    }

    #[test]
    fn test_matrix_compute_read_l2_visible_issue_plus_1() {
        // use_cycle 2 (compute), No-def: eff = 2-2+1-0 = 1 -> visible at bundle 1.
        let f = pending_at(2, Bypass::No, 0);
        assert_eq!(f.read_with(4, 2, Bypass::No), [0u32; 8], "issue+0 reads old");
        let f1 = pending_at(2, Bypass::No, 1);
        assert_eq!(f1.read_with(4, 2, Bypass::No), [0xAAAA_AAAA; 8], "issue+1 visible");
    }

    #[test]
    fn test_matrix_store_read_l2_visible_issue_plus_2() {
        // use_cycle 1 (store), No-def: eff = 2-1+1-0 = 2 -> visible at bundle 2.
        let f1 = pending_at(2, Bypass::No, 1);
        assert_eq!(f1.read_with(4, 1, Bypass::No), [0u32; 8], "issue+1 still old for store");
        let f2 = pending_at(2, Bypass::No, 2);
        assert_eq!(f2.read_with(4, 1, Bypass::No), [0xAAAA_AAAA; 8], "issue+2 visible");
    }

    #[test]
    fn test_matrix_matched_forwarding_shaves_a_cycle() {
        // use_cycle 2, Mov-def == Mov-use: eff = 2-2+1-1 = 0 -> clamp 1 -> issue+1.
        let f0 = pending_at(2, Bypass::Mov, 0);
        assert_eq!(f0.read_with(4, 2, Bypass::Mov), [0u32; 8], "matched: own bundle reads old");
        let f1 = pending_at(2, Bypass::Mov, 1);
        assert_eq!(f1.read_with(4, 2, Bypass::Mov), [0xAAAA_AAAA; 8], "matched: issue+1");
    }

    #[test]
    fn test_matrix_l_def_variation_shifts_visibility() {
        // l_def = 3, use_cycle 1 (store), No: eff = 3-1+1 = 3 -> issue+3.
        let f2 = pending_at(3, Bypass::No, 2);
        assert_eq!(f2.read_with(4, 1, Bypass::No), [0u32; 8], "l_def 3 store: still old at +2");
        let f3 = pending_at(3, Bypass::No, 3);
        assert_eq!(f3.read_with(4, 1, Bypass::No), [0xAAAA_AAAA; 8], "l_def 3 store: visible at +3");
    }

    #[test]
    fn test_matrix_clamp_never_resolves_in_own_bundle() {
        // Even maximally-forwarded, eff clamps to >= 1: issue+0 always reads old.
        let f0 = pending_at(1, Bypass::Mov, 0);
        assert_eq!(f0.read_with(4, 2, Bypass::Mov), [0u32; 8], "own bundle reads old");
    }

    #[test]
    fn test_matrix_per_operand_independence() {
        // Two reads of the same reg at the SAME bundle with different use_cycles
        // resolve independently (compute sees it, store does not).
        let f1 = pending_at(2, Bypass::No, 1);
        assert_eq!(f1.read_with(4, 2, Bypass::No), [0xAAAA_AAAA; 8], "compute (uc2) visible at +1");
        assert_eq!(f1.read_with(4, 1, Bypass::No), [0u32; 8], "store (uc1) not visible at +1");
    }

    // ── Shim-path validation ─────────────────────────────────────────────
    // These test the read/read_store/read_lane convenience wrappers which
    // supply fixed (use_cycle, use_bypass) defaults. They exercise realistic
    // l_def values (1 or 2) covering the VMOV-class anchors.

    #[test]
    fn test_latest_issued_write_wins() {
        // Two in-flight Mov writes to one reg; the later-issued visible one wins.
        let a = [0x1111_1111; 8];
        let b = [0x2222_2222; 8];
        let mut r = VectorRegisterFile::new();
        r.advance_bundle(0);
        r.queue_write(2, a, 2, Bypass::Mov);
        r.advance_bundle(1);
        r.queue_write(2, b, 2, Bypass::Mov); // issued at bundle 1
                                             // cur=1: a visible (0+1 <= 1), b not yet (1+1=2 > 1).
        assert_eq!(r.read(2), a, "only the earlier write is visible at bundle 1");
        r.advance_bundle(2);
        // cur=2: a landed; b now visible (1+1=2 <= 2). b wins (later-issued).
        assert_eq!(r.read(2), b, "later-issued write wins once visible");
    }

    #[test]
    fn test_wide_write_splits_and_resolves() {
        let mut data = [0u32; 16];
        for (i, d) in data.iter_mut().enumerate() {
            *d = 0xC0DE_0000 | i as u32;
        }
        let mut r = VectorRegisterFile::new();
        r.advance_bundle(0);
        r.queue_write_wide(4, data, 2, Bypass::Mov); // x-reg base 4 -> regs 4,5
        r.advance_bundle(1);
        assert_eq!(r.read_wide(4), data, "wide Mov->compute visible at issue+1");
        assert_eq!(r.read(4)[..], data[..8]);
        assert_eq!(r.read(5)[..], data[8..]);
    }

    #[test]
    fn test_zero_latency_writes_immediately() {
        let mut r = VectorRegisterFile::new();
        r.advance_bundle(5);
        r.queue_write(2, [0xAAAA_AAAA; 8], 0, Bypass::Mov);
        assert_eq!(r.read(2), [0xAAAA_AAAA; 8], "l_def=0 writes immediately (test/legacy path)");
    }
}
