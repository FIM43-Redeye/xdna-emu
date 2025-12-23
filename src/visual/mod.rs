//! GUI visualization for AMD XDNA NPU emulation.
//!
//! This module provides an egui-based visual debugger that shows:
//! - Tile array grid with status colors
//! - Core state details (PC, registers, status)
//! - Memory hex view
//! - Lock states
//! - DMA buffer descriptors
//! - Emulation controls (run, step, pause, reset)
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::visual::EmulatorApp;
//!
//! let app = EmulatorApp::default();
//! eframe::run_native("xdna-emu", options, Box::new(|_| Ok(Box::new(app))));
//! ```

mod app;
mod tile_grid;
mod tile_detail;
mod controls;
mod memory_view;

pub use app::EmulatorApp;
