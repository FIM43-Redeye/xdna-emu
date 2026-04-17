//! Stream switch port ranges and configuration bits (from AM025).
//!
//! Submodules `compute`, `mem_tile`, `shim` each contain
//! `NORTH_MASTER_START/END`, `SOUTH_MASTER_START/END`, etc.
//! `ENABLE_BIT` and `SLAVE_SELECT_MASK` live at the module root.

include!(concat!(env!("OUT_DIR"), "/gen_stream_ranges.rs"));
