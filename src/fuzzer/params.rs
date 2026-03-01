//! Fuzz case parameters.
//!
//! Every generated test is fully determined by `FuzzParams` plus the seed.
//! Shrinking works by simplifying these parameters.

use crate::fuzzer::ast::KernelBody;

/// Parameters controlling a single fuzz iteration.
#[derive(Debug, Clone)]
pub struct FuzzParams {
    /// RNG seed that produced this case. Enables reproducibility.
    pub seed: u64,
    /// Number of elements in the input/output buffers.
    pub buffer_size: usize,
    /// Element type for buffers.
    pub dtype: ScalarType,
    /// The kernel body to execute.
    pub body: KernelBody,
}

/// Scalar element types for fuzz buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    I32,
    I16,
    I8,
}

impl ScalarType {
    /// Size of one element in bytes.
    pub fn byte_size(self) -> usize {
        match self {
            ScalarType::I32 => 4,
            ScalarType::I16 => 2,
            ScalarType::I8 => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::ast::*;

    #[test]
    fn test_fuzz_params_debug_contains_seed() {
        let params = FuzzParams {
            seed: 42,
            buffer_size: 64,
            dtype: ScalarType::I32,
            body: KernelBody {
                ops: vec![],
                loop_style: LoopStyle::Simple,
            },
        };
        let debug = format!("{:?}", params);
        assert!(debug.contains("42"));
    }

    #[test]
    fn test_scalar_type_variants() {
        assert_ne!(ScalarType::I32, ScalarType::I16);
        assert_ne!(ScalarType::I16, ScalarType::I8);
    }
}
