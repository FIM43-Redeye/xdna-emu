//! The scalar fuzzer as a coverage-driven `core::Domain` tenant (framework
//! Step 2). A case is a chain of elementwise scalar stages, each writing its
//! own output region, so a divergence localizes to the exact op -- the scalar
//! analogue of the vector tenant's per-slice localization. See
//! `docs/superpowers/specs/2026-06-11-scalar-coverage-domain.md`.

pub mod chain;
pub mod table;
