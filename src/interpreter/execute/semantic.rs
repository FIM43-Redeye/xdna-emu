//! TableGen-driven semantic execution.
//!
//! This module provides execution handlers based on SemanticOp from TableGen,
//! replacing the 133+ manual Operation handlers with ~40 semantic handlers.
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
//! ## Which Operations Set Flags
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

use crate::interpreter::bundle::{Operand, Operation, SlotIndex, SlotOp};
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::traits::Flags;
use crate::tablegen::SemanticOp;

/// Execute a SlotOp using its SemanticOp if available.
///
/// Returns `true` if execution was handled, `false` if the caller should
/// fall back to Operation-based dispatch.
pub fn execute_semantic(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let Some(semantic) = op.semantic else {
        return false; // No semantic info - fall back to Operation dispatch
    };

    // Vector operations have VectorReg destinations - let VectorAlu handle them
    if let Some(Operand::VectorReg(_)) = op.dest {
        return false; // Fall back to VectorAlu
    }

    // Also check for explicitly vector Operations
    if matches!(
        &op.op,
        Operation::VectorAdd { .. }
            | Operation::VectorSub { .. }
            | Operation::VectorMul { .. }
            | Operation::VectorAnd { .. }
            | Operation::VectorOr { .. }
            | Operation::VectorXor { .. }
    ) {
        return false; // Fall back to VectorAlu
    }

    log::trace!("[SEMANTIC] Handling {:?} via semantic {:?}", op.op, semantic);

    match semantic {
        // Arithmetic operations
        SemanticOp::Add => execute_add(op, ctx),
        SemanticOp::Sub => execute_sub(op, ctx),
        SemanticOp::Adc => execute_adc(op, ctx),
        SemanticOp::Sbc => execute_sbc(op, ctx),
        SemanticOp::Mul => execute_mul(op, ctx),
        SemanticOp::SDiv => execute_div(op, ctx, true),  // signed
        SemanticOp::UDiv => execute_div(op, ctx, false), // unsigned
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

        // Memory operations
        SemanticOp::Load => execute_load(op, ctx),
        SemanticOp::Store => execute_store(op, ctx),

        // Control flow
        SemanticOp::Br => execute_branch(op, ctx),
        SemanticOp::BrCond => execute_branch_cond(op, ctx),

        // Bit manipulation
        SemanticOp::Ctlz => execute_clz(op, ctx),
        SemanticOp::Cttz => execute_ctz(op, ctx),
        SemanticOp::Ctpop => execute_popcount(op, ctx),
        SemanticOp::Rotl => execute_rotl(op, ctx),
        SemanticOp::Rotr => execute_rotr(op, ctx),
        SemanticOp::Bswap => execute_bswap(op, ctx),

        // Move/copy
        SemanticOp::Copy => execute_mov(op, ctx),

        // Sign/zero extension
        SemanticOp::SignExtend => execute_sign_extend(op, ctx),
        SemanticOp::ZeroExtend => execute_zero_extend(op, ctx),
        SemanticOp::Truncate => execute_truncate(op, ctx),

        // Nop
        SemanticOp::Nop => true, // Nothing to do

        // Event: generate a user-defined trace event (INSTR_EVENT_0/1)
        SemanticOp::Event => {
            let event_id = match op.sources.first() {
                Some(Operand::Immediate(v)) => *v as u8,
                _ => 0, // Default to event 0 if operand missing
            };
            let pc = ctx.pc();
            let cycle = ctx.cycles;
            ctx.timing_context_mut().record_event(
                cycle,
                crate::interpreter::state::EventType::InstrEvent { pc, id: event_id },
            );
            true
        }

        // Intentionally delegated to Operation-based dispatch:
        // - Call/Ret/Done: Control flow with AIE-specific side effects
        //   (delay slots, LR management, core halt). Handled by dedicated
        //   jl/ret/done handlers in execute_operation().
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
fn read_source(op: &SlotOp, ctx: &ExecutionContext, index: usize) -> u32 {
    op.sources.get(index).map_or(0, |src| read_operand(src, ctx))
}

/// Read an operand value from context.
///
/// Uses VLIW-safe reads where available (scalar_read for scalar regs).
fn read_operand(operand: &Operand, ctx: &ExecutionContext) -> u32 {
    match operand {
        Operand::ScalarReg(r) => ctx.scalar_read(*r),
        Operand::PointerReg(r) => ctx.pointer_read(*r),
        Operand::ModifierReg(r) => ctx.modifier_read(*r),
        Operand::Immediate(v) => *v as u32,
        Operand::VectorReg(_) | Operand::AccumReg(_) => {
            // Vector/accum need special handling - return 0 for scalar context
            0
        }
        _ => 0,
    }
}

/// Write a result to the destination operand.
fn write_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: u32) {
    if let Some(dest) = &op.dest {
        write_operand(dest, ctx, value);
    }
}

/// Write a value to an operand.
///
/// Three categories:
/// - Scalar/pointer/modifier: fully implemented, write to register file.
/// - Vector/accum: valid destinations, but writes are handled by VectorAlu.
///   Only reaches here if a scalar semantic op has vector operands (misclassification).
/// - Non-writable (Immediate, Lock, etc.): should never reach here after decoder
///   validation in `extract_ordered_operands`. Debug-assert to catch regressions.
fn write_operand(operand: &Operand, ctx: &mut ExecutionContext, value: u32) {
    match operand {
        Operand::ScalarReg(r) => ctx.scalar.write(*r, value),
        Operand::PointerReg(r) => ctx.pointer.write(*r, value),
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
                _ => {
                    log::trace!("control register write: id={}, value=0x{:X}", id, value);
                }
            }
        }
        Operand::VectorReg(_) | Operand::AccumReg(_) => {
            log::trace!(
                "[SEMANTIC] Vector/accum register write not yet implemented: {:?}",
                operand,
            );
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
    // Pointer arithmetic: padda/paddb [pN], $imm or [pN], djK
    //
    // These instructions compute pN = pN + offset. In TableGen, the pointer
    // register appears as a tied output ($res = $ptr constraint) but the
    // resolver puts it only in output_order, not input_order. So the SlotOp
    // has dest=PointerReg(N) and sources=[Immediate] -- missing the pointer
    // as a source.
    //
    // SP variants (PADDA_sp_imm, PADDB_sp_imm) are even more implicit:
    // Defs=[SP], Uses=[SP] with empty (outs), so dest=None.
    //
    // Detect both cases in LoadA/LoadB slots with a single source.
    if matches!(op.slot, SlotIndex::LoadA | SlotIndex::LoadB)
        && op.sources.len() == 1
    {
        let ptr_idx = match op.dest {
            Some(Operand::PointerReg(p)) => p,
            None => 6, // SP = p6
            _ => return false, // Unexpected dest type
        };
        let base = ctx.pointer.read(ptr_idx);
        let offset = read_source(op, ctx, 0);
        let result = base.wrapping_add(offset);
        ctx.pointer.write(ptr_idx, result);
        return true;
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

    // Determine condition: sel.nez inverts the test
    let condition = match op.op {
        Operation::ScalarSelNez => test != 0,
        _ => test == 0, // sel.eqz and generic select
    };

    log::debug!(
        "[SEMANTIC SELECT] op={:?} sources={:?} implicit_regs={:?} test={} true={} false={} cond={}",
        op.op, op.sources, op.implicit_regs, test, src_true, src_false, condition
    );

    let result = if condition { src_true } else { src_false };

    log::debug!("[SEMANTIC SELECT] result={}", result);

    write_dest(op, ctx, result);
    true
}

// ============================================================================
// Memory operations
// ============================================================================

fn execute_load(_op: &SlotOp, _ctx: &mut ExecutionContext) -> bool {
    // Memory operations need the full memory subsystem - fall back for now
    log::trace!("[SEMANTIC] Load requires memory subsystem - falling back");
    false
}

fn execute_store(_op: &SlotOp, _ctx: &mut ExecutionContext) -> bool {
    log::trace!("[SEMANTIC] Store requires memory subsystem - falling back");
    false
}

// ============================================================================
// Control flow
// ============================================================================

fn execute_branch(_op: &SlotOp, _ctx: &mut ExecutionContext) -> bool {
    log::trace!("[SEMANTIC] Branch requires control flow handling - falling back");
    false
}

fn execute_branch_cond(_op: &SlotOp, _ctx: &mut ExecutionContext) -> bool {
    log::trace!("[SEMANTIC] BranchCond requires control flow handling - falling back");
    false
}


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
    // Width determined by the Operation variant
    let extended = match op.op {
        Operation::ScalarExtendS8 => ((a as i8) as i32) as u32,
        _ => ((a as i16) as i32) as u32, // ScalarExtendS16 and default
    };
    write_dest(op, ctx, extended);
    true
}

fn execute_zero_extend(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    // Width determined by the Operation variant
    let extended = match op.op {
        Operation::ScalarExtendU8 => a & 0xFF,
        _ => a & 0xFFFF, // ScalarExtendU16 and default
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
    use crate::interpreter::bundle::Operation;
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

        let mut op = SlotOp::with_semantic(SlotIndex::Scalar0, Operation::ScalarAdd, SemanticOp::Add);
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

        let mut op = SlotOp::with_semantic(SlotIndex::Scalar0, Operation::ScalarSub, SemanticOp::Sub);
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

        let mut op = SlotOp::with_semantic(SlotIndex::Scalar0, Operation::ScalarSelEqz, SemanticOp::Select);
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

        let mut op = SlotOp::with_semantic(SlotIndex::Scalar0, Operation::ScalarSelEqz, SemanticOp::Select);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        op.implicit_regs = smallvec![ImplicitReg {
            reg_class: "eR27".to_string(),
            reg_num: 27,
            is_use: true,
        }];

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 200); // r27!=0, so false value selected
    }

    #[test]
    fn test_execute_sel_nez_with_zero() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 100); // true value
        ctx.scalar.write(2, 200); // false value
        ctx.scalar.write(27, 0);  // test value (r27) = 0, so sel.nez selects false

        let mut op = SlotOp::with_semantic(SlotIndex::Scalar0, Operation::ScalarSelNez, SemanticOp::Select);
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

        let mut op = SlotOp::with_semantic(SlotIndex::Scalar0, Operation::ScalarSelNez, SemanticOp::Select);
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

        let mut op = SlotOp::with_semantic(SlotIndex::Scalar0, Operation::ScalarLt, SemanticOp::SetLt);
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

        let mut op = SlotOp::with_semantic(SlotIndex::Scalar0, Operation::ScalarShl, SemanticOp::Shl);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0b10000); // 1 << 4 = 16
    }

    #[test]
    fn test_no_semantic_returns_false() {
        let ctx = make_test_context();
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd);
        // No semantic set, should return false
        assert!(!execute_semantic(&op, &mut ctx.clone()));
    }

    #[test]
    fn test_control_reg_write_via_semantic_copy() {
        // Verify that movx crRnd, rN works through the semantic Copy path.
        // Previously, ControlReg was not handled in write_operand, so
        // control register writes were silently dropped when semantic
        // dispatch returned true before ScalarAlu fallback could run.
        let mut ctx = make_test_context();
        ctx.scalar.write(5, 9); // rounding mode value

        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarMov, SemanticOp::Copy,
        );
        op.sources = smallvec![Operand::ScalarReg(5)];
        op.dest = Some(Operand::ControlReg(6)); // crRnd

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.srs_config.rounding_mode, 9);
    }

    #[test]
    fn test_control_reg_write_crsat_via_semantic() {
        let mut ctx = make_test_context();

        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarMov, SemanticOp::Copy,
        );
        op.sources = smallvec![Operand::Immediate(3)];
        op.dest = Some(Operand::ControlReg(9)); // crSat

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.srs_config.saturation_mode, 3);
    }

    #[test]
    fn test_control_reg_write_srssign_via_semantic() {
        let mut ctx = make_test_context();

        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarMov, SemanticOp::Copy,
        );
        op.sources = smallvec![Operand::Immediate(1)];
        op.dest = Some(Operand::ControlReg(8)); // crSRSSign

        assert!(execute_semantic(&op, &mut ctx));
        assert!(ctx.srs_config.srs_sign);
    }

    #[test]
    fn test_control_reg_write_masks_bits() {
        let mut ctx = make_test_context();

        // crSat only uses lower 2 bits
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarMov, SemanticOp::Copy,
        );
        op.sources = smallvec![Operand::Immediate(0xFF)];
        op.dest = Some(Operand::ControlReg(9)); // crSat
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.srs_config.saturation_mode, 3); // 0xFF & 0x3

        // crRnd only uses lower 4 bits
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarMov, SemanticOp::Copy,
        );
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

        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar1, Operation::ScalarMul, SemanticOp::Mul,
        );
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
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarAnd, SemanticOp::And,
        );
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0x0F00);

        // OR
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarOr, SemanticOp::Or,
        );
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0xFFF0);

        // XOR
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarXor, SemanticOp::Xor,
        );
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0xF0F0);

        // NOT
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarMov, SemanticOp::Not,
        );
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
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarSra, SemanticOp::Sra,
        );
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0xF800_0000u32); // sign-extended

        // SRL (logical right shift, zero-fills)
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarShr, SemanticOp::Srl,
        );
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0x0800_0000u32); // zero-filled
    }

    #[test]
    fn test_execute_abs_sets_carry() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, (-42i32) as u32);

        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarAbs, SemanticOp::Abs,
        );
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

        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarMov, SemanticOp::Neg,
        );
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
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarClz, SemanticOp::Ctlz,
        );
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 8);

        // CTZ
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarMov, SemanticOp::Cttz,
        );
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 20);

        // POPCOUNT
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarMov, SemanticOp::Ctpop,
        );
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
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarMov, SemanticOp::Rotl,
        );
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0x0000_FF00u32);

        // ROTR (rotate back)
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarMov, SemanticOp::Rotr,
        );
        op.sources = smallvec![Operand::ScalarReg(3), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(4));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(4), 0x0000_00FFu32);
    }

    #[test]
    fn test_execute_bswap() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 0x12_34_56_78u32);

        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarMov, SemanticOp::Bswap,
        );
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0x78_56_34_12u32);
    }

    #[test]
    fn test_execute_sign_extend() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 0x0000_0080u32); // 128 in unsigned, -128 as i8

        // Sign extend from 8 bits
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarExtendS8, SemanticOp::SignExtend,
        );
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0xFFFF_FF80u32); // -128 sign-extended

        // Sign extend from 16 bits
        ctx.scalar.write(1, 0x0000_8000u32); // -32768 as i16
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarExtendS16, SemanticOp::SignExtend,
        );
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0xFFFF_8000u32);
    }

    #[test]
    fn test_execute_zero_extend() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 0xFFFF_FFFFu32);

        // Zero extend from 8 bits
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarExtendU8, SemanticOp::ZeroExtend,
        );
        op.sources = smallvec![Operand::ScalarReg(1)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0xFF);

        // Zero extend from 16 bits
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarExtendU16, SemanticOp::ZeroExtend,
        );
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

        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarDiv, SemanticOp::SDiv,
        );
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
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarDiv, SemanticOp::SDiv,
        );
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), i32::MIN as u32);

        // Unsigned div by zero returns u32::MAX
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarDivu, SemanticOp::UDiv,
        );
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), u32::MAX);
    }

    #[test]
    fn test_execute_rem() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 17);
        ctx.scalar.write(2, 5);

        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarMod, SemanticOp::SRem,
        );
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

        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarAdc, SemanticOp::Adc,
        );
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

        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarSbc, SemanticOp::Sbc,
        );
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
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarLt, SemanticOp::SetLt,
        );
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 1);

        // Unsigned: 0xFFFFFFFF > 1 is true, so ult is false
        let mut op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::ScalarLtu, SemanticOp::SetUlt,
        );
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0); // 0xFFFFFFFF is NOT < 1 unsigned
    }

    #[test]
    fn test_execute_pointer_add_in_load_slot() {
        // padda instruction: pN = pN + imm, in LoadA/LoadB slot
        let mut ctx = make_test_context();
        ctx.pointer.write(2, 0x1000);

        let mut op = SlotOp::with_semantic(
            SlotIndex::LoadA, Operation::PointerAdd, SemanticOp::Add,
        );
        op.sources = smallvec![Operand::Immediate(0x100)];
        op.dest = Some(Operand::PointerReg(2));

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.pointer.read(2), 0x1100);
    }

    #[test]
    fn test_execute_nop() {
        let mut ctx = make_test_context();
        let op = SlotOp::with_semantic(
            SlotIndex::Scalar0, Operation::Nop, SemanticOp::Nop,
        );
        assert!(execute_semantic(&op, &mut ctx));
    }

    #[test]
    fn test_delegated_ops_return_false() {
        let mut ctx = make_test_context();

        // Call should delegate to ControlUnit
        let op = SlotOp::with_semantic(
            SlotIndex::Control, Operation::Call, SemanticOp::Call,
        );
        assert!(!execute_semantic(&op, &mut ctx));

        // LockAcquire should delegate
        let op = SlotOp::with_semantic(
            SlotIndex::Control, Operation::LockAcquire, SemanticOp::LockAcquire,
        );
        assert!(!execute_semantic(&op, &mut ctx));

        // Load should delegate to MemoryUnit
        let op = SlotOp::with_semantic(
            SlotIndex::LoadA,
            Operation::Load {
                width: crate::interpreter::bundle::MemWidth::Word,
                post_modify: crate::interpreter::bundle::PostModify::None,
            },
            SemanticOp::Load,
        );
        assert!(!execute_semantic(&op, &mut ctx));
    }
}
