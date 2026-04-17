//! Trace event codes extracted from mlir-aie.
//!
//! The mlir-aie Python bridge script (`tools/mlir-aie-bridge.py trace-events`)
//! emits a JSON map of event ID -> name for each tile module type. build.rs
//! parses this into per-module const tables and name-lookup functions.
//!
//! Exposed items (from generated code):
//!   - `pub mod core_events`     -- `CoreEvent` codes as `pub const` u8 values
//!   - `pub mod mem_events`      -- `MemEvent` codes
//!   - `pub mod memtile_events`  -- `MemTileEvent` codes
//!   - `pub mod shim_events`     -- `ShimTileEvent` codes
//!   - `core_event_name(u8) -> &'static str`
//!   - `mem_event_name(u8) -> &'static str`
//!   - `memtile_event_name(u8) -> &'static str`
//!   - `shim_event_name(u8) -> &'static str`
//!
//! Falls back to stub functions returning "UNKNOWN" when mlir-aie is not
//! available at build time.

include!(concat!(env!("OUT_DIR"), "/trace_event_codes.rs"));
