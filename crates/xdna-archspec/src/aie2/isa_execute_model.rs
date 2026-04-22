//! AIE2 implementation of `IsaExecutor`.
//!
//! Shipped with an empty impl body per the Subsystem 7 audit's
//! Approach A conclusion (see `crate::isa_execute` module docs).

use crate::isa_execute::IsaExecutor;

/// Zero-sized type representing the AIE2 execute model.
#[derive(Debug, Default)]
pub struct Aie2IsaExecutor;

impl IsaExecutor for Aie2IsaExecutor {
    // Empty per audit.
}

/// Process-global singleton used by `ArchConfig::isa_executor()`.
pub static AIE2_ISA_EXECUTOR: Aie2IsaExecutor = Aie2IsaExecutor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aie2_isa_executor_is_zst() {
        assert_eq!(core::mem::size_of::<Aie2IsaExecutor>(), 0);
    }

    #[test]
    fn aie2_isa_executor_impls_trait() {
        // Static assertion: the singleton can be borrowed as
        // &'static dyn IsaExecutor (the shape the accessor will use).
        let _: &'static dyn IsaExecutor = &AIE2_ISA_EXECUTOR;
    }
}
