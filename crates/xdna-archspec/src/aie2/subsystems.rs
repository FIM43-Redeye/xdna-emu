//! Per-tile-type subsystem address ranges (from ArchModel).
//!
//! Submodules: compute, memtile, shim. Each contains `pub mod <subsystem>`
//! with `OFFSET_START` and `OFFSET_END` consts. When a subsystem kind
//! appears in multiple modules within the same tile (e.g., compute has
//! `performance` in both Core and Memory), the module name is prefixed:
//! `core_performance`, `memory_performance`.

include!(concat!(env!("OUT_DIR"), "/gen_subsystems.rs"));
