//! Trace comparison visualizer for AMD XDNA NPU emulation.
//!
//! Replaces the old emulator GUI with a trace-focused tool that renders
//! HW vs EMU event timelines side by side, highlights divergences, and
//! supports piecewise alignment for phase-aware comparison.

pub mod alignment;
pub mod data;
pub mod event_detail;
pub mod theme;
pub mod tile_selector;
