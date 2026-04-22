//! ISA execute model -- per-arch behavioral seam for instruction
//! execution semantics.
//!
//! Subsystem 7 (Phase 1b) of the device-family refactor introduces
//! this module. The `IsaExecutor` trait is the seam by which
//! execute-layer code would reach arch-specific behavior where AIE1,
//! AIE2, and AIE2P have fundamentally different algorithmic shape.
//!
//! The Subsystem 7 audit (docs/arch/subsys7-audit.md, Approach A
//! landing) concluded that zero trait methods are warranted today:
//! every candidate divergence (VMAC crossbar routing, SRS/UPS
//! rounding, cascade-link presence, memory-quadrant decode, etc.)
//! reduces to data that lives in archspec and is read via
//! `arch_handle::*` accessors. The trait ships empty as a landing
//! pad -- future seams, if any, attach methods here.
//!
//! See the per-seam design note at `docs/arch/isa-execute-model.md`.

/// Per-arch execute-layer behavioral seam.
///
/// Ships empty per Subsystem 7's audit (Approach A: zero trait
/// methods). Future divergence surfaced by a second-arch landing
/// (AIE1 or AIE2P) that cannot be expressed as data would add a
/// method here. Until then, the trait exists as a stable anchor
/// for the `ArchConfig::isa_executor()` dispatch method.
pub trait IsaExecutor: Send + Sync + core::fmt::Debug {
    // Intentionally empty per audit. See module docs.
}
