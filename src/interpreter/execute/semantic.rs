//! TableGen-driven semantic execution.
//!
//! This module provides execution handlers based on SemanticOp from TableGen,
//! using ~40 SemanticOp handlers for all instruction dispatch.
//!
//! # Architecture
//!
//! ```text
//! SlotOp { semantic: Some(SemanticOp::Add), sources, dest, implicit_regs }
//!                          |
//!                          v
//!                  execute_semantic()
//!                          |
//!                          v
//!                    execute_add()
//!                          |
//!                          v
//!                  read sources[0], sources[1]
//!                  compute result = a + b
//!                  write to dest
//! ```
//!
//! # Operand Conventions
//!
//! All handlers assume operands are in **canonical order** per TableGen InstrDef.inputs:
//! - sources[0] = first input operand
//! - sources[1] = second input operand (if any)
//! - Implicit registers accessed via `op.implicit_regs` or `op.get_implicit_use()`
//!
//! # CPU Flag Behavior (AIE2 Accurate)
//!
//! AIE2 has a unique flag architecture:
//!
//! | Flag | Hardware? | Storage | Behavior |
//! |------|-----------|---------|----------|
//! | **Carry (C)** | YES | SR bit 0 | Set by ADD, SUB, ADC, SBC, ABS, NEG |
//! | Zero (Z) | NO | Computed | Branch logic tests register == 0 |
//! | Negative (N) | NO | Computed | Branch logic tests register < 0 |
//! | Overflow (V) | NO | Computed | Computed on demand |
//!
//! ## Which SemanticOps Set Flags
//!
//! **Set Carry (from TableGen Defs=[srCarry]):**
//! - `ADD`, `SUB` - arithmetic carry/borrow
//! - `ADC`, `SBC` - add/sub with carry (also Uses=[srCarry])
//! - `ABS` - carry set if input was negative
//! - `NEG` - equivalent to SUB(0, x)
//!
//! **Do NOT set flags (preserve Carry):**
//! - `MUL`, `DIV`, `MOD` - no flag effects
//! - `AND`, `OR`, `XOR`, `NOT` - no flag effects
//! - `SHL`, `SHR`, `SRA` - no flag effects (unlike x86!)
//! - Comparisons (`LT`, `GE`, etc.) - produce 0/1 result, no flags
//! - Moves, loads, stores - no flag effects
//!
//! ## Branch Conditions
//!
//! - `jz`/`jnz` - test **register value** directly (most common)
//! - `jc`/`jnc` - test Carry flag (for ADC/SBC chains)
//! - Other conditions computed from register values, not stored flags

use crate::interpreter::bundle::{Operand, SlotIndex, SlotOp, SelectVariant};
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::traits::Flags;
use crate::tablegen::SemanticOp;

/// Execute a SlotOp using its SemanticOp if available.
///
/// Returns `true` if execution was handled, `false` if the caller should
/// fall back to other execution units.
pub fn execute_semantic(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let Some(semantic) = op.semantic else {
        return false; // No semantic info - not a recognized instruction
    };

    // Vector operations are handled by VectorAlu, not semantic dispatch.
    // Check is_vector flag (set by decoder/infer_from_operation).
    if op.is_vector {
        return false; // Fall back to VectorAlu
    }

    log::trace!("[SEMANTIC] Handling semantic {:?}", semantic);

    match semantic {
        // Arithmetic operations
        SemanticOp::Add => execute_add(op, ctx),
        SemanticOp::Sub => execute_sub(op, ctx),
        SemanticOp::Adc => execute_adc(op, ctx),
        SemanticOp::Sbc => execute_sbc(op, ctx),
        SemanticOp::Mul => execute_mul(op, ctx),
        SemanticOp::SDiv => execute_div(op, ctx, true),  // signed
        SemanticOp::UDiv => execute_div(op, ctx, false), // unsigned
        SemanticOp::DivStep => execute_div_step(op, ctx),
        SemanticOp::SRem => execute_rem(op, ctx, true),  // signed
        SemanticOp::URem => execute_rem(op, ctx, false), // unsigned
        SemanticOp::Abs => execute_abs(op, ctx),
        SemanticOp::Neg => execute_neg(op, ctx),

        // Bitwise operations
        SemanticOp::And => execute_and(op, ctx),
        SemanticOp::Or => execute_or(op, ctx),
        SemanticOp::Xor => execute_xor(op, ctx),
        SemanticOp::Not => execute_not(op, ctx),
        SemanticOp::Shl => execute_shl(op, ctx),
        SemanticOp::Sra => execute_sra(op, ctx), // arithmetic right shift
        SemanticOp::Srl => execute_srl(op, ctx), // logical right shift
        SemanticOp::AshlBidir => execute_ashl_bidir(op, ctx),
        SemanticOp::LshlBidir => execute_lshl_bidir(op, ctx),

        // Comparison operations (produce 0 or 1)
        SemanticOp::SetLt => execute_setcc(op, ctx, CmpOp::Lt, true),
        SemanticOp::SetLe => execute_setcc(op, ctx, CmpOp::Le, true),
        SemanticOp::SetGt => execute_setcc(op, ctx, CmpOp::Gt, true),
        SemanticOp::SetGe => execute_setcc(op, ctx, CmpOp::Ge, true),
        SemanticOp::SetEq => execute_setcc(op, ctx, CmpOp::Eq, true),
        SemanticOp::SetNe => execute_setcc(op, ctx, CmpOp::Ne, true),
        SemanticOp::SetUlt => execute_setcc(op, ctx, CmpOp::Lt, false),
        SemanticOp::SetUle => execute_setcc(op, ctx, CmpOp::Le, false),
        SemanticOp::SetUgt => execute_setcc(op, ctx, CmpOp::Gt, false),
        SemanticOp::SetUge => execute_setcc(op, ctx, CmpOp::Ge, false),

        // Conditional select
        SemanticOp::Select => execute_select(op, ctx),

        // Bit manipulation
        SemanticOp::Ctlz => execute_clz(op, ctx),
        SemanticOp::Cttz => execute_ctz(op, ctx),
        SemanticOp::Ctpop => execute_popcount(op, ctx),
        SemanticOp::Rotl => execute_rotl(op, ctx),
        SemanticOp::Rotr => execute_rotr(op, ctx),
        SemanticOp::Bswap => execute_bswap(op, ctx),

        // Scalar-only: compare (flag-setting, no destination)
        SemanticOp::Cmp => execute_cmp(op, ctx),

        // Scalar-only: count leading bits (sign-extension bits, != CLZ)
        SemanticOp::Clb => execute_clb(op, ctx),

        // Move/copy
        SemanticOp::Copy => execute_mov(op, ctx),

        // Pointer operations (padda, paddb, mova, movb)
        SemanticOp::PointerAdd => execute_pointer_add(op, ctx),
        SemanticOp::PointerMov => execute_pointer_mov(op, ctx),

        // Sign/zero extension
        SemanticOp::SignExtend => execute_sign_extend(op, ctx),
        SemanticOp::ZeroExtend => execute_zero_extend(op, ctx),
        SemanticOp::Truncate => execute_truncate(op, ctx),

        // Hardware state reads
        SemanticOp::ReadCycleCounter => {
            // MOV_CNTR: read the per-core cycle counter into a scalar register.
            // The counter is 32-bit on AIE2 (wraps at 2^32).
            let counter = ctx.cycles as u32;
            write_dest(op, ctx, counter);
            true
        }

        // Nop
        SemanticOp::Nop => true, // Nothing to do

        // Event: generate a user-defined trace event (INSTR_EVENT_0/1)
        SemanticOp::Event => {
            let event_id = match op.sources.first() {
                Some(Operand::Immediate(v)) => *v as u8,
                other => {
                    log::warn!(
                        "[SEMANTIC] Event: expected Immediate operand for event_id, got {:?}, defaulting to 0",
                        other
                    );
                    0
                }
            };
            let pc = ctx.pc();
            let cycle = ctx.cycles;
            ctx.timing_context_mut().record_event(
                cycle,
                crate::interpreter::state::EventType::InstrEvent { pc, id: event_id },
            );
            true
        }

        // Intentionally delegated to specialized execution units:
        // - Call/Ret/Done: Control flow with AIE-specific side effects
        //   (delay slots, LR management, core halt). Handled by ControlUnit.
        // - LockAcquire/LockRelease: Interact with the device lock model,
        //   not pure register-to-register computation.
        // - Intrinsic(_): Target-specific intrinsics (vector ops, etc.)
        //   requiring per-intrinsic dispatch.
        _ => {
            log::trace!("[SEMANTIC] Delegated to operation dispatch: {:?}", semantic);
            false
        }
    }
}

// ============================================================================
// Helper types and functions
// ============================================================================

/// Comparison operation type.
#[derive(Debug, Clone, Copy)]
enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// Read a source operand as a 32-bit value.
pub(super) fn read_source(op: &SlotOp, ctx: &ExecutionContext, index: usize) -> u32 {
    op.sources.get(index).map_or(0, |src| read_operand(src, ctx))
}

/// Read a scalar operand value from context.
///
/// Uses VLIW-safe reads where available (scalar_read for scalar regs).
pub(super) fn read_operand(operand: &Operand, ctx: &ExecutionContext) -> u32 {
    match operand {
        Operand::ScalarReg(r) => ctx.scalar_read(*r),
        Operand::PointerReg(r) => ctx.pointer_read(*r),
        Operand::ModifierReg(r) => ctx.modifier_read(*r),
        Operand::Immediate(v) => *v as u32,
        Operand::VectorReg(r) => {
            // Scalar read of a vector register: extract lane 0 (low 32 bits).
            // This occurs in VEXTRACT-like patterns where scalar context needs
            // one element from a vector.
            ctx.vector.read(*r)[0]
        }
        Operand::AccumReg(r) => {
            // Scalar read of an accumulator: extract lane 0 (low 32 bits).
            ctx.accumulator.read(*r)[0] as u32
        }
        _ => 0,
    }
}

/// Write a result to the destination operand.
pub(super) fn write_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: u32) {
    if let Some(dest) = &op.dest {
        write_operand(dest, ctx, value);
    }
}

/// Write a value to an operand.
///
/// Three categories:
/// - Scalar: immediate write to register file.
/// - Pointer/modifier: deferred write with pipeline latency 1. The value
///   becomes visible in the next sequential bundle. At branch boundaries,
///   `delay_pending_writes()` adds an extra cycle, modeling the loss of
///   pipeline forwarding after a branch flush.
/// - Vector/accum: valid destinations, but writes are handled by VectorAlu.
///   Only reaches here if a scalar semantic op has vector operands (misclassification).
/// - Non-writable (Immediate, Lock, etc.): should never reach here after decoder
///   validation in `extract_ordered_operands`. Debug-assert to catch regressions.
pub(super) fn write_operand(operand: &Operand, ctx: &mut ExecutionContext, value: u32) {
    match operand {
        Operand::ScalarReg(r) => ctx.scalar.write(*r, value),
        Operand::PointerReg(r) if *r == crate::interpreter::state::SP_PTR_INDEX => {
            // Dedicated SP register -- write immediately (no pipeline alias)
            ctx.set_sp(value);
        }
        Operand::PointerReg(r) => ctx.queue_pointer_write(*r, value, 1),
        Operand::ModifierReg(r) => ctx.modifier.write(*r, value),
        Operand::ControlReg(id) => {
            // Control register write: update SRS/UPS config.
            // Hardware register IDs from AIE2GenRegisterInfo.td.
            match id {
                9 => { // crSat
                    ctx.srs_config.saturation_mode = (value & 0x3) as u8;
                    log::trace!("crSat = {} (raw 0x{:X})", value & 0x3, value);
                }
                6 => { // crRnd
                    ctx.srs_config.rounding_mode = (value & 0xF) as u8;
                    log::trace!("crRnd = {} (raw 0x{:X})", value & 0xF, value);
                }
                8 => { // crSRSSign
                    ctx.srs_config.srs_sign = (value & 1) != 0;
                    log::trace!("crSRSSign = {} (raw 0x{:X})", value & 1, value);
                }
                // Mask registers q0-q3: ControlReg(16..19).
                // Scalar write sets the low 32 bits (clears upper bits).
                16..=19 => {
                    let q_idx = (id - 16) as u8;
                    ctx.mask.write_u32_low(q_idx, value);
                    log::trace!("q{} = 0x{:08X} (scalar write)", q_idx, value);
                }
                // ql0-ql3: ControlReg(28..31). Scalar write to low word
                // of low 64-bit half (preserves rest).
                28..=31 => {
                    let q_idx = (id - 28) as u8;
                    let mut cur = ctx.mask.read(q_idx);
                    cur[0] = value;
                    ctx.mask.write(q_idx, cur);
                    log::trace!("ql{} low word = 0x{:08X} (scalar write)", q_idx, value);
                }
                // qh0-qh3: ControlReg(32..35). Scalar write to low word
                // of high 64-bit half (preserves rest).
                32..=35 => {
                    let q_idx = (id - 32) as u8;
                    let mut cur = ctx.mask.read(q_idx);
                    cur[2] = value;
                    ctx.mask.write(q_idx, cur);
                    log::trace!("qh{} low word = 0x{:08X} (scalar write)", q_idx, value);
                }
                _ => {
                    log::trace!("control register write: id={}, value=0x{:X}", id, value);
                }
            }
        }
        Operand::VectorReg(r) => {
            // Scalar write to a vector register: write lane 0, zero-fill rest.
            // This should be rare -- most vector writes go through VectorAlu.
            log::warn!(
                "[SEMANTIC] Scalar write to VectorReg({}): value=0x{:X} -- \
                 if this fires frequently, check decoder classification",
                r, value
            );
            let mut lanes = [0u32; 8];
            lanes[0] = value;
            ctx.vector.write(*r, lanes);
        }
        Operand::AccumReg(r) => {
            // Scalar write to an accumulator: write lane 0, zero-fill rest.
            log::warn!(
                "[SEMANTIC] Scalar write to AccumReg({}): value=0x{:X} -- \
                 if this fires frequently, check decoder classification",
                r, value
            );
            let mut lanes = [0u64; 8];
            lanes[0] = value as u64;
            ctx.accumulator.write(*r, lanes);
        }
        _ => {
            debug_assert!(
                false,
                "write_operand called with non-writable operand: {:?}",
                operand,
            );
        }
    }
}

/// Get an implicit register use value.
fn get_implicit_use(op: &SlotOp, ctx: &ExecutionContext, reg_num: u8) -> Option<u32> {
    // Check if this register is in implicit_regs as a use
    if op.implicit_regs.iter().any(|ir| ir.reg_num == reg_num && ir.is_use) {
        Some(ctx.scalar_read(reg_num))
    } else {
        None
    }
}

// ============================================================================
// Arithmetic handlers
// ============================================================================

fn execute_add(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    // All PADDB/PADDA instructions should reach execute_pointer_add via
    // SemanticOp::PointerAdd (forced by the decoder for is_ptr_arithmetic
    // encodings). If a pointer arithmetic instruction reaches execute_add,
    // that is a decoder classification bug.
    if matches!(op.slot, SlotIndex::LoadA | SlotIndex::LoadB)
        && op.sources.len() == 1
        && matches!(op.dest, Some(Operand::PointerReg(_)) | None)
    {
        log::warn!(
            "[BUG] Pointer arithmetic reached execute_add instead of execute_pointer_add: \
             pc=0x{:03X} slot={:?} dest={:?} srcs={:?} name={:?}",
            ctx.pc(), op.slot, op.dest, op.sources, op.encoding_name
        );
    }

    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let result = a.wrapping_add(b);
    write_dest(op, ctx, result);
    // AIE2: ADD sets the Carry flag (C). Z/N/V computed by branch logic.
    ctx.set_flags(Flags::from_add(a, b, result));
    true
}

fn execute_sub(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let result = a.wrapping_sub(b);
    write_dest(op, ctx, result);
    // AIE2: SUB sets the Carry flag (C). Z/N/V computed by branch logic.
    ctx.set_flags(Flags::from_sub(a, b, result));
    true
}

/// Add with carry: result = a + b + carry_in. Sets C, V, Z, N flags.
fn execute_adc(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let carry_in = if ctx.flags().c { 1u32 } else { 0u32 };
    let (r1, c1) = a.overflowing_add(b);
    let (result, c2) = r1.overflowing_add(carry_in);
    write_dest(op, ctx, result);
    let mut flags = Flags::from_result(result);
    flags.c = c1 || c2;
    let a_sign = (a as i32) < 0;
    let b_sign = (b as i32) < 0;
    let r_sign = (result as i32) < 0;
    flags.v = (a_sign == b_sign) && (a_sign != r_sign);
    ctx.set_flags(flags);
    true
}

/// Subtract with borrow: result = a - b - borrow_in. Sets C, V, Z, N flags.
/// ARM convention: C=1 means no borrow, C=0 means borrow occurred.
fn execute_sbc(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let borrow_in = if ctx.flags().c { 0u32 } else { 1u32 };
    let (r1, b1) = a.overflowing_sub(b);
    let (result, b2) = r1.overflowing_sub(borrow_in);
    write_dest(op, ctx, result);
    let mut flags = Flags::from_result(result);
    flags.c = !(b1 || b2); // C=0 means borrow
    let a_sign = (a as i32) < 0;
    let b_sign = (b as i32) < 0;
    let r_sign = (result as i32) < 0;
    flags.v = (a_sign != b_sign) && (a_sign != r_sign);
    ctx.set_flags(flags);
    true
}

fn execute_mul(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let result = a.wrapping_mul(b);
    write_dest(op, ctx, result);
    // AIE2: MUL does NOT set any flags (no Defs=[srCarry] in TableGen)
    true
}

fn execute_div(op: &SlotOp, ctx: &mut ExecutionContext, signed: bool) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let result = if b == 0 {
        // Division by zero: return saturated value
        if signed { i32::MIN as u32 } else { u32::MAX }
    } else if signed {
        ((a as i32).wrapping_div(b as i32)) as u32
    } else {
        a / b
    };
    write_dest(op, ctx, result);
    // AIE2: DIV does NOT set any flags
    true
}

fn execute_rem(op: &SlotOp, ctx: &mut ExecutionContext, signed: bool) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let result = if b == 0 {
        a // Modulo by zero returns dividend unchanged
    } else if signed {
        ((a as i32).wrapping_rem(b as i32)) as u32
    } else {
        a % b
    };
    write_dest(op, ctx, result);
    // AIE2: REM/MOD does NOT set any flags
    true
}

/// Iterative signed division step (dstep).
///
/// AIE2 `divs d0, r31, s0, s1` performs one step of a non-restoring
/// long-division algorithm.  r31 holds the accumulating partial quotient
/// (pi) and s0 holds the working dividend/accumulator (ai).  s1 is the
/// divisor (b).  Each invocation produces one quotient bit.
///
/// The algorithm (from aietools ISG me_inline_primitives.h:1947-1967):
///
///   pre:  div_shft = {pi[30:0], ai[31]}
///   sub:  (div_tmp, co) = div_shft - b
///   post: pa = {pi, ai} << 1
///         if co == 0:  pa[63:32] = div_tmp; pa[0] = 1
///         po = pa[63:32]  -> r31
///         ao = pa[31:0]   -> d0
fn execute_div_step(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let ai = read_source(op, ctx, 0); // s0: working accumulator
    let b  = read_source(op, ctx, 1); // s1: divisor
    let pi = ctx.scalar.read(31);     // r31: partial quotient (implicit)

    // dstep_pre: {pi[30:0], ai[31]}
    let div_shft = ((pi & 0x7FFF_FFFF) << 1) | (ai >> 31);

    // Trial subtraction: div_shft - b, with carry (borrow) detection.
    let (div_tmp, borrow) = div_shft.overflowing_sub(b);
    // In hardware: co=0 means borrow occurred (div_shft < b as unsigned).

    // dstep_post: pa = {pi, ai} << 1
    let pa: u64 = ((pi as u64) << 32) | (ai as u64);
    let mut new_pa = pa << 1;

    if borrow {
        // co == 0 (borrow): update upper half with subtraction result, set bit 0.
        new_pa = (new_pa & 0x0000_0000_FFFF_FFFE) | ((div_tmp as u64) << 32) | 1;
    }

    let po = (new_pa >> 32) as u32; // -> r31
    let ao = new_pa as u32;          // -> d0

    ctx.scalar.write(31, po);
    write_dest(op, ctx, ao);
    true
}

// ============================================================================
// Bitwise handlers
// ============================================================================

fn execute_and(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    write_dest(op, ctx, a & b);
    // AIE2: AND does NOT set any flags (preserves C)
    true
}

fn execute_or(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    write_dest(op, ctx, a | b);
    // AIE2: OR does NOT set any flags (preserves C)
    true
}

fn execute_xor(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    write_dest(op, ctx, a ^ b);
    // AIE2: XOR does NOT set any flags (preserves C)
    true
}

fn execute_not(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    write_dest(op, ctx, !a);
    // AIE2: NOT does NOT set any flags (preserves C)
    true
}

fn execute_shl(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let shift = b & 0x1F; // Mask to 5 bits for 32-bit shift
    write_dest(op, ctx, a << shift);
    // AIE2: Shifts do NOT set any flags (preserves C)
    true
}

fn execute_sra(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0) as i32;
    let b = read_source(op, ctx, 1);
    let shift = b & 0x1F;
    write_dest(op, ctx, (a >> shift) as u32);
    // AIE2: Shifts do NOT set any flags (preserves C)
    true
}

fn execute_srl(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let shift = b & 0x1F;
    write_dest(op, ctx, a >> shift);
    // AIE2: Shifts do NOT set any flags (preserves C)
    true
}

/// ASHL: bidirectional arithmetic shift.
/// Positive shift = left, negative shift = arithmetic right.
/// Per AIE2InstrPatterns.td section 4.1.11.
fn execute_ashl_bidir(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1) as i32;
    let result = if b >= 0 {
        let shift = (b as u32) & 0x1F;
        a << shift
    } else {
        let shift = ((-b) as u32) & 0x1F;
        ((a as i32) >> shift) as u32
    };
    write_dest(op, ctx, result);
    true
}

/// LSHL: bidirectional logical shift.
/// Positive shift = left, negative shift = logical right.
/// Per AIE2InstrPatterns.td section 4.1.11.
fn execute_lshl_bidir(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1) as i32;
    let result = if b >= 0 {
        let shift = (b as u32) & 0x1F;
        a << shift
    } else {
        let shift = ((-b) as u32) & 0x1F;
        a >> shift
    };
    write_dest(op, ctx, result);
    true
}

// ============================================================================
// Comparison handlers
// ============================================================================

fn execute_setcc(op: &SlotOp, ctx: &mut ExecutionContext, cmp: CmpOp, signed: bool) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);

    let result = if signed {
        let a = a as i32;
        let b = b as i32;
        match cmp {
            CmpOp::Lt => a < b,
            CmpOp::Le => a <= b,
            CmpOp::Gt => a > b,
            CmpOp::Ge => a >= b,
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
        }
    } else {
        match cmp {
            CmpOp::Lt => a < b,
            CmpOp::Le => a <= b,
            CmpOp::Gt => a > b,
            CmpOp::Ge => a >= b,
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
        }
    };

    write_dest(op, ctx, if result { 1 } else { 0 });
    true
}

// ============================================================================
// Conditional select
// ============================================================================

/// Execute conditional select.
///
/// - `sel.eqz` / `ScalarSelEqz`: dest = (test == 0) ? true_val : false_val
/// - `sel.nez` / `ScalarSelNez`: dest = (test != 0) ? true_val : false_val
///
/// For both variants, r27 is implicit (from TableGen eR27 register class).
/// For generic select, the test value is the 3rd source operand.
fn execute_select(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let src_true = read_source(op, ctx, 0);
    let src_false = read_source(op, ctx, 1);

    // Get test value: first check for implicit r27, then fall back to source[2]
    let test = get_implicit_use(op, ctx, 27).unwrap_or_else(|| read_source(op, ctx, 2));

    // AIE2 sel.eqz/sel.nez: conditional select between two source registers.
    //   sel.eqz rd, rs1, rs2, r27: rd = (r27 == 0) ? rs1 : rs2
    //   sel.nez rd, rs1, rs2, r27: rd = (r27 != 0) ? rs1 : rs2
    let condition = match op.select_variant {
        Some(SelectVariant::NotEqualZero) => test != 0,
        _ => test == 0, // sel.eqz and generic select
    };

    let result = if condition { src_true } else { src_false };

    log::trace!(
        "[SEMANTIC SELECT] variant={:?} test={} src_true={} cond={} result={}",
        op.select_variant, test, src_true, condition, result
    );

    write_dest(op, ctx, result);
    true
}

// ============================================================================
// Memory operations
// ============================================================================
// Special operations
// ============================================================================

fn execute_mov(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let value = read_source(op, ctx, 0);
    log::debug!(
        "[SEMANTIC MOV] dest={:?} src={:?} value={}",
        op.dest, op.sources.get(0), value
    );
    write_dest(op, ctx, value);
    true
}

/// Pointer add: pN = pN + offset (padda, paddb, padds).
///
/// The pointer register appears as a tied operand (dest=pN, implicit read of
/// pN for the base address). Uses deferred pipeline write (latency 1) to
/// match hardware pointer write timing.
///
/// The decoder may produce dest=None for PADDB/PADDA when the pointer
/// register is only in the Defs list (implicit). In that case, infer the
/// destination from the first PointerReg source (the tied operand).
fn execute_pointer_add(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    // Resolve the destination pointer register.
    // Priority: explicit dest > inferred from first PointerReg source > SP fallback.
    let ptr_dest = match &op.dest {
        Some(Operand::PointerReg(p)) => Some(*p),
        None => {
            // Infer from first PointerReg source (tied operand pattern).
            op.sources.iter().find_map(|s| match s {
                Operand::PointerReg(p) => Some(*p),
                _ => None,
            })
        }
        other => panic!(
            "execute_pointer_add: destination must be PointerReg, got {:?} (encoding={:?})",
            other, op.encoding_name,
        ),
    };

    match ptr_dest {
        Some(p) if p != crate::interpreter::state::SP_PTR_INDEX => {
            let base = ctx.pointer_read(p);
            let offset = read_source(op, ctx, 0);
            let result = base.wrapping_add(offset);
            ctx.queue_pointer_write(p, result, 1);
        }
        _ => {
            // PADDA_sp_imm / PADDB_sp_imm: operates on the dedicated SP register.
            // AIE2 SP is SPLReg<12>, separate from pointer registers p0-p7.
            let base = ctx.sp();
            let offset = read_source(op, ctx, 0);
            let result = base.wrapping_add(offset);
            ctx.set_sp(result);
        }
    }
    true
}

/// Pointer move: pN = value (mova, movb to pointer register).
///
/// Uses deferred pipeline write (latency 1) to match hardware timing.
fn execute_pointer_mov(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let ptr_idx = match &op.dest {
        Some(Operand::PointerReg(p)) => *p,
        other => panic!(
            "execute_pointer_mov: destination must be PointerReg, got {:?} (encoding={:?})",
            other, op.encoding_name,
        ),
    };
    let value = read_source(op, ctx, 0);
    ctx.queue_pointer_write(ptr_idx, value, 1);
    true
}

fn execute_abs(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let result = (a as i32).wrapping_abs() as u32;
    write_dest(op, ctx, result);
    // AIE2: ABS sets the Carry flag (Defs=[srCarry] in TableGen)
    // Carry is set if the input was negative (negation occurred)
    let was_negative = (a as i32) < 0;
    let mut flags = ctx.flags();
    flags.c = was_negative;
    ctx.set_flags(flags);
    true
}

fn execute_neg(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let result = (a as i32).wrapping_neg() as u32;
    write_dest(op, ctx, result);
    // AIE2: NEG is equivalent to SUB(0, a), sets Carry flag
    // Using SUB semantics: C = (0 >= a) in unsigned, i.e., C = (a == 0)
    ctx.set_flags(Flags::from_sub(0, a, result));
    true
}

fn execute_clz(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    write_dest(op, ctx, a.leading_zeros());
    true
}

/// Compare (flag-setting only, no destination write).
/// Sets flags from subtraction a - b without storing the result.
fn execute_cmp(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let result = a.wrapping_sub(b);
    ctx.set_flags(Flags::from_sub(a, b, result));
    true
}

/// Count leading bits: count of leading bits matching the sign bit.
/// Different from CLZ: CLB counts sign-redundant bits including the sign bit.
/// Per AIE2 hardware: CLB(0x00xxxxxx) = 2 (two leading zeros).
fn execute_clb(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let src = read_source(op, ctx, 0);
    let result = if (src as i32) >= 0 {
        src.leading_zeros()
    } else {
        src.leading_ones()
    };
    write_dest(op, ctx, result);
    true
}

fn execute_ctz(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    write_dest(op, ctx, a.trailing_zeros());
    true
}

fn execute_popcount(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    write_dest(op, ctx, a.count_ones());
    true
}

fn execute_rotl(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1) & 31; // rotate amount mod 32
    write_dest(op, ctx, a.rotate_left(b));
    true
}

fn execute_rotr(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1) & 31;
    write_dest(op, ctx, a.rotate_right(b));
    true
}

fn execute_bswap(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    write_dest(op, ctx, a.swap_bytes());
    true
}

fn execute_sign_extend(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    // Width encoded in mem_width: Byte = 8-bit, HalfWord = 16-bit
    let extended = match op.mem_width {
        crate::interpreter::bundle::MemWidth::Byte => ((a as i8) as i32) as u32,
        _ => ((a as i16) as i32) as u32, // 16-bit and default
    };
    write_dest(op, ctx, extended);
    true
}

fn execute_zero_extend(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    // Width encoded in mem_width: Byte = 8-bit, HalfWord = 16-bit
    let extended = match op.mem_width {
        crate::interpreter::bundle::MemWidth::Byte => a & 0xFF,
        _ => a & 0xFFFF, // 16-bit and default
    };
    write_dest(op, ctx, extended);
    true
}

fn execute_truncate(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    // Truncation just keeps lower bits - identity for 32-bit
    write_dest(op, ctx, a);
    true
}


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;
    use crate::interpreter::bundle::SelectVariant;
    use crate::tablegen::ImplicitReg;
    use smallvec::smallvec;

    fn make_test_context() -> ExecutionContext {
        ExecutionContext::new()
    }

    #[test]
    fn test_execute_add() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 10);
        ctx.scalar.write(2, 20);

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 30);
    }

    #[test]
    fn test_execute_sub() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 50);
        ctx.scalar.write(2, 20);

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Sub);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 30);
    }

    #[test]
    fn test_execute_select_with_implicit_r27() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 100); // true value
        ctx.scalar.write(2, 200); // false value
        ctx.scalar.write(27, 0);  // test value (r27) = 0, so select true

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Select)
            .with_select_variant(SelectVariant::EqualZero);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        op.implicit_regs = smallvec![ImplicitReg {
            reg_class: "eR27".to_string(),
            reg_num: 27,
            is_use: true,
        }];

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 100); // r27==0, so true value selected
    }

    #[test]
    fn test_execute_select_with_implicit_r27_nonzero() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 100); // true value
        ctx.scalar.write(2, 200); // false value
        ctx.scalar.write(27, 1);  // test value (r27) != 0, so select false

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Select)
            .with_select_variant(SelectVariant::EqualZero);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        op.implicit_regs = smallvec![ImplicitReg {
            reg_class: "eR27".to_string(),
            reg_num: 27,
            is_use: true,
        }];

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 200); // r27!=0, eqz condition false -> false value
    }

    #[test]
    fn test_execute_sel_nez_with_zero() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 100); // true value
        ctx.scalar.write(2, 200); // false value
        ctx.scalar.write(27, 0);  // test value (r27) = 0, so sel.nez selects false

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Select)
            .with_select_variant(SelectVariant::NotEqualZero);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        op.implicit_regs = smallvec![ImplicitReg {
            reg_class: "eR27".to_string(),
            reg_num: 27,
            is_use: true,
        }];

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 200); // r27==0, nez condition false -> false value
    }

    #[test]
    fn test_execute_sel_nez_with_nonzero() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 100); // true value
        ctx.scalar.write(2, 200); // false value
        ctx.scalar.write(27, 42); // test value (r27) != 0, so sel.nez selects true

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Select)
            .with_select_variant(SelectVariant::NotEqualZero);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        op.implicit_regs = smallvec![ImplicitReg {
            reg_class: "eR27".to_string(),
            reg_num: 27,
            is_use: true,
        }];

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 100); // r27!=0, nez condition true -> true value
    }

    #[test]
    fn test_execute_setcc_lt_signed() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, (-5i32) as u32); // -5
        ctx.scalar.write(2, 3);               // 3

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::SetLt);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 1); // -5 < 3 is true
    }

    #[test]
    fn test_execute_shift_left() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 0b0001);
        ctx.scalar.write(2, 4);

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Shl);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0b10000); // 1 << 4 = 16
    }

    #[test]
    fn test_no_semantic_returns_false() {
        let ctx = make_test_context();
        let mut op = SlotOp::nop(SlotIndex::Scalar0);
        op.semantic = None; // Simulate unknown instruction with no semantic
        assert!(!execute_semantic(&op, &mut ctx.clone()));
    }

    #[test]
    fn test_control_reg_write_via_semantic_copy() {
        // Verify that movx crRnd, rN works through the semantic Copy path.
        let mut ctx = make_test_context();
        ctx.scalar.write(5, 9); // rounding mode value

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy);
        op.sources = smallvec![Operand::ScalarReg(5)];
        op.dest = Some(Operand::ControlReg(6)); // crRnd

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.srs_config.rounding_mode, 9);
    }

    #[test]
    fn test_control_reg_write_crsat_via_semantic() {
        let mut ctx = make_test_context();

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy);
        op.sources = smallvec![Operand::Immediate(3)];
        op.dest = Some(Operand::ControlReg(9)); // crSat

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.srs_config.saturation_mode, 3);
    }

    #[test]
    fn test_control_reg_write_srssign_via_semantic() {
        let mut ctx = make_test_context();

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy);
        op.sources = smallvec![Operand::Immediate(1)];
        op.dest = Some(Operand::ControlReg(8)); // crSRSSign

        assert!(execute_semantic(&op, &mut ctx));
        assert!(ctx.srs_config.srs_sign);
    }

    #[test]
    fn test_control_reg_write_masks_bits() {
        let mut ctx = make_test_context();

        // crSat only uses lower 2 bits
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy);
        op.sources = smallvec![Operand::Immediate(0xFF)];
        op.dest = Some(Operand::ControlReg(9)); // crSat
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.srs_config.saturation_mode, 3); // 0xFF & 0x3

        // crRnd only uses lower 4 bits
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy);
        op.sources = smallvec![Operand::Immediate(0xFF)];
        op.dest = Some(Operand::ControlReg(6)); // crRnd
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.srs_config.rounding_mode, 15); // 0xFF & 0xF
    }

    #[test]
    fn test_execute_mul() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 7);
        ctx.scalar.write(2, 6);

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar1, SemanticOp::Mul);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 42);
    }

    #[test]
    fn test_execute_and_or_xor_not() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 0xFF00);
        ctx.scalar.write(2, 0x0FF0);

        // AND
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::And);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0x0F00);

        // OR
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Or);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0xFFF0);

        // XOR
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Xor);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0xF0F0);

        // NOT
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Not);
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), !0xFF00u32);
    }

    #[test]
    fn test_execute_shifts() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 0x8000_0000u32); // negative in signed
        ctx.scalar.write(2, 4);

        // SRA (arithmetic right shift, preserves sign)
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Sra);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0xF800_0000u32); // sign-extended

        // SRL (logical right shift, zero-fills)
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Srl);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0x0800_0000u32); // zero-filled
    }

    #[test]
    fn test_execute_abs_sets_carry() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, (-42i32) as u32);

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Abs);
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 42);
        assert!(ctx.flags().c); // Carry set because input was negative
    }

    #[test]
    fn test_execute_neg() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 42);

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Neg);
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), (-42i32) as u32);
    }

    #[test]
    fn test_execute_clz_ctz_popcount() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 0x00F0_0000u32); // 8 leading zeros, 20 trailing zeros, 4 ones

        // CLZ
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Ctlz);
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 8);

        // CTZ
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Cttz);
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 20);

        // POPCOUNT
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Ctpop);
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 4);
    }

    #[test]
    fn test_execute_rotl_rotr() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 0x0000_00FFu32);
        ctx.scalar.write(2, 8);

        // ROTL
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Rotl);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0x0000_FF00u32);

        // ROTR (rotate back)
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Rotr);
        op.sources = smallvec![Operand::ScalarReg(3), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(4));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(4), 0x0000_00FFu32);
    }

    #[test]
    fn test_execute_bswap() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 0x12_34_56_78u32);

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Bswap);
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0x78_56_34_12u32);
    }

    #[test]
    fn test_execute_sign_extend() {
        use crate::interpreter::bundle::MemWidth;
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 0x0000_0080u32); // 128 in unsigned, -128 as i8

        // Sign extend from 8 bits
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::SignExtend);
        op.mem_width = MemWidth::Byte;
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0xFFFF_FF80u32); // -128 sign-extended

        // Sign extend from 16 bits
        ctx.scalar.write(1, 0x0000_8000u32); // -32768 as i16
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::SignExtend);
        op.mem_width = MemWidth::HalfWord;
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0xFFFF_8000u32);
    }

    #[test]
    fn test_execute_zero_extend() {
        use crate::interpreter::bundle::MemWidth;
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 0xFFFF_FFFFu32);

        // Zero extend from 8 bits
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::ZeroExtend);
        op.mem_width = MemWidth::Byte;
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0xFF);

        // Zero extend from 16 bits
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::ZeroExtend);
        op.mem_width = MemWidth::HalfWord;
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0xFFFF);
    }

    #[test]
    fn test_execute_div_signed() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, (-100i32) as u32);
        ctx.scalar.write(2, 7);

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::SDiv);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3) as i32, -14); // -100 / 7 = -14
    }

    #[test]
    fn test_execute_div_by_zero() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 42);
        ctx.scalar.write(2, 0);

        // Signed div by zero returns i32::MIN
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::SDiv);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), i32::MIN as u32);

        // Unsigned div by zero returns u32::MAX
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::UDiv);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), u32::MAX);
    }

    #[test]
    fn test_execute_div_step_basic() {
        // One dstep iteration with pi=0, ai=0x00000008, b=3.
        // After 32 iterations this would compute 8/3=2 r2.
        // First step: div_shft = {0[30:0], 0[31]} = 0
        //   div_tmp = 0 - 3 = wraps, borrow=true
        //   pa = {0, 8} << 1 = {0, 16}
        //   borrow: pa[63:32] = (0u32.wrapping_sub(3)), pa[0] = 1
        //   ao = 16 | 1 = 17, po = 0xFFFFFFFD
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 0x00000008); // ai = s0
        ctx.scalar.write(2, 3);          // b  = s1
        ctx.scalar.write(31, 0);         // pi = r31

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::DivStep);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));

        // ao = (8 << 1) | 1 = 17 = 0x11
        assert_eq!(ctx.scalar_read(3), 0x00000011, "ao (d0)");
        // po = 0u32.wrapping_sub(3) = 0xFFFFFFFD
        assert_eq!(ctx.scalar_read(31), 0xFFFFFFFD, "po (r31)");
    }

    #[test]
    fn test_execute_div_step_no_borrow() {
        // pi=0x40000000, ai=0, b=0.
        // div_shft = {pi[30:0], ai[31]} = {0x40000000, 0} = 0x80000000
        // div_tmp = 0x80000000 - 0 = 0x80000000, no borrow.
        // No borrow: pa = {0x40000000, 0} << 1 = {0x80000000, 0}
        // po = 0x80000000, ao = 0
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 0);           // ai
        ctx.scalar.write(2, 0);           // b
        ctx.scalar.write(31, 0x40000000); // pi

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::DivStep);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));

        assert_eq!(ctx.scalar_read(3), 0, "ao");
        assert_eq!(ctx.scalar_read(31), 0x80000000, "po");
    }

    #[test]
    fn test_execute_rem() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 17);
        ctx.scalar.write(2, 5);

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::SRem);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 2); // 17 % 5 = 2
    }

    #[test]
    fn test_execute_adc_with_carry() {
        let mut ctx = make_test_context();
        // Set carry flag before ADC
        ctx.set_flags(Flags { z: false, n: false, c: true, v: false });
        ctx.scalar.write(1, 10);
        ctx.scalar.write(2, 20);

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Adc);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 31); // 10 + 20 + 1 (carry)
    }

    #[test]
    fn test_execute_sbc_with_borrow() {
        let mut ctx = make_test_context();
        // C=0 means borrow occurred, so borrow_in=1
        ctx.set_flags(Flags { z: false, n: false, c: false, v: false });
        ctx.scalar.write(1, 30);
        ctx.scalar.write(2, 10);

        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Sbc);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 19); // 30 - 10 - 1 (borrow)
    }

    #[test]
    fn test_execute_unsigned_comparisons() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 0xFFFF_FFFFu32); // -1 signed, but huge unsigned
        ctx.scalar.write(2, 1);

        // Signed: -1 < 1 is true
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::SetLt);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 1);

        // Unsigned: 0xFFFFFFFF > 1 is true, so ult is false
        let mut op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::SetUlt);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0); // 0xFFFFFFFF is NOT < 1 unsigned
    }

    #[test]
    fn test_execute_pointer_add() {
        // padda instruction: pN = pN + imm
        let mut ctx = make_test_context();
        ctx.pointer.write(2, 0x1000);

        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::PointerAdd)
            .with_dest(Operand::PointerReg(2))
            .with_source(Operand::Immediate(0x100));

        assert!(execute_semantic(&op, &mut ctx));
        ctx.flush_pending_writes();
        assert_eq!(ctx.pointer.read(2), 0x1100);
    }

    #[test]
    fn test_execute_pointer_mov() {
        let mut ctx = make_test_context();
        ctx.scalar.write(3, 0x2000);

        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::PointerMov)
            .with_dest(Operand::PointerReg(1))
            .with_source(Operand::ScalarReg(3));

        assert!(execute_semantic(&op, &mut ctx));
        ctx.flush_pending_writes();
        assert_eq!(ctx.pointer.read(1), 0x2000);
    }

    #[test]
    fn test_pointer_add_via_add_logs_warning() {
        // After the decoder fix, all PADDB/PADDA instructions should arrive
        // with SemanticOp::PointerAdd. If one arrives as Add, the warning
        // fires but the regular Add path still executes (non-fatal).
        let mut ctx = make_test_context();
        ctx.scalar.write(0, 100);
        ctx.scalar.write(1, 200);

        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(2), 300);
    }

    #[test]
    fn test_execute_nop() {
        let mut ctx = make_test_context();
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Nop);
        assert!(execute_semantic(&op, &mut ctx));
    }

    #[test]
    fn test_delegated_ops_return_false() {
        let mut ctx = make_test_context();

        // Call should delegate to ControlUnit
        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::Call);
        assert!(!execute_semantic(&op, &mut ctx));

        // LockAcquire should delegate
        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockAcquire);
        assert!(!execute_semantic(&op, &mut ctx));

        // Load should delegate to MemoryUnit
        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load);
        assert!(!execute_semantic(&op, &mut ctx));
    }

    // --- Tests migrated from scalar.rs (unique coverage not in above tests) ---

    #[test]
    fn test_add_overflow_sets_carry_and_zero() {
        let mut ctx = make_test_context();
        ctx.scalar.write(0, 0xFFFF_FFFF);
        ctx.scalar.write(1, 1);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.scalar_read(2), 0);
        assert!(ctx.flags().c); // Carry set
        assert!(ctx.flags().z); // Zero set
    }

    #[test]
    fn test_cmp_sets_flags() {
        let mut ctx = make_test_context();
        ctx.scalar.write(0, 10);
        ctx.scalar.write(1, 10);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Cmp)
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        execute_semantic(&op, &mut ctx);
        assert!(ctx.flags().z);

        ctx.scalar.write(1, 20);
        execute_semantic(&op, &mut ctx);
        assert!(!ctx.flags().z);
        assert!(ctx.flags().n);
    }

    #[test]
    fn test_copy_register() {
        let mut ctx = make_test_context();
        ctx.scalar.write(0, 42);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy)
            .with_dest(Operand::ScalarReg(1))
            .with_source(Operand::ScalarReg(0));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.scalar_read(1), 42);
    }

    #[test]
    fn test_copy_immediate() {
        let mut ctx = make_test_context();
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy)
            .with_dest(Operand::ScalarReg(5))
            .with_source(Operand::Immediate(12345));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(5), 12345);
    }

    #[test]
    fn test_pointer_reg_as_scalar_source() {
        let mut ctx = make_test_context();
        ctx.pointer.write(0, 0x1000);
        ctx.scalar.write(0, 0x100);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(1))
            .with_source(Operand::PointerReg(0))
            .with_source(Operand::ScalarReg(0));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.scalar_read(1), 0x1100);
    }
}
