//! NPU Architecture Graph -- queryable hardware model.
//!
//! This module re-exports from the `xdna-graph` workspace crate, which
//! contains the canonical graph types, extractors, and register database
//! parser. The separate crate enables use from both runtime code and
//! build.rs for compile-time code generation.

pub use xdna_graph::device_model;
pub use xdna_graph::regdb_extractor;
pub use xdna_graph::types;
