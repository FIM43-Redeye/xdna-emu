//! NPU Architecture Specification -- validated hardware model.
//!
//! This module re-exports from the `xdna-archspec` workspace crate, which
//! contains the canonical types, extractors, and register database parser.
//! The separate crate enables use from both runtime code and build.rs for
//! compile-time code generation.

pub use xdna_archspec::device_model;
pub use xdna_archspec::regdb_extractor;
pub use xdna_archspec::types;
