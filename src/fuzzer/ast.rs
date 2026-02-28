//! Kernel body AST for fuzz-generated programs.
//!
//! This AST covers only what happens inside a single tile. Program structure
//! (DMA, routing, locks at the array level) is handled by the IRON template.

/// The body of a kernel function: a sequence of operations over tile data.
#[derive(Debug, Clone)]
pub struct KernelBody {
    pub ops: Vec<KernelOp>,
    pub loop_style: LoopStyle,
}

/// How the kernel iterates over buffer elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopStyle {
    /// Simple for loop over buffer elements.
    Simple,
    /// Hardware loop (ZLS/ZLE) over buffer elements.
    HardwareLoop,
}

/// A single operation in the kernel body.
#[derive(Debug, Clone)]
pub enum KernelOp {
    /// `dst = src1 op src2`
    ScalarArith {
        op: ScalarOp,
        dst: Var,
        src1: Operand,
        src2: Operand,
    },
    /// `buf[idx] = val`
    Store {
        buf: BufRef,
        idx: Operand,
        val: Operand,
    },
    /// `if (cond) { then_ops } else { else_ops }`
    Branch {
        cond: Operand,
        then_ops: Vec<KernelOp>,
        else_ops: Vec<KernelOp>,
    },
    /// Hardware loop with fixed count.
    HwLoop {
        count: u32,
        body: Vec<KernelOp>,
    },
}

/// Scalar arithmetic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

/// A reference to a temporary variable in the kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Var(pub u8);

/// An operand: variable, literal, or buffer load.
#[derive(Debug, Clone)]
pub enum Operand {
    Var(Var),
    Literal(i32),
    Load { buf: BufRef, idx: Box<Operand> },
}

/// Reference to input (0) or output (1) buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufRef(pub u8);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_body_empty_is_valid() {
        let body = KernelBody {
            ops: vec![],
            loop_style: LoopStyle::Simple,
        };
        assert!(body.ops.is_empty());
    }

    #[test]
    fn test_scalar_arith_construction() {
        let op = KernelOp::ScalarArith {
            op: ScalarOp::Add,
            dst: Var(0),
            src1: Operand::Load {
                buf: BufRef(0),
                idx: Box::new(Operand::Var(Var(1))),
            },
            src2: Operand::Literal(1),
        };
        let _ = format!("{:?}", op);
    }

    #[test]
    fn test_nested_branch_construction() {
        let op = KernelOp::Branch {
            cond: Operand::Var(Var(0)),
            then_ops: vec![KernelOp::Store {
                buf: BufRef(1),
                idx: Operand::Var(Var(1)),
                val: Operand::Literal(42),
            }],
            else_ops: vec![],
        };
        let _ = format!("{:?}", op);
    }
}
