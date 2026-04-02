//! Composite register encoding.
//!
//! AIE2 uses "composite encoders" that pack multiple register classes into
//! a single bit field using class-specific discriminant patterns. The actual
//! decoding is handled by `decoder.rs` using the encoder formulas from
//! `AIE2MCCodeEmitterRegOperandDef.h` in llvm-aie.
//!
//! The `CompositeEncoder` enum lives in `crate::tablegen` and is imported
//! directly by the decoder.
