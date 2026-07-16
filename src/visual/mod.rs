//! Live visual debugger (egui). Gated behind the `gui` feature. All logic
//! lives in crate::debugger (egui-free, tested); this layer only renders.
pub mod app;
pub mod controls;
pub mod detail;
pub mod overview;
pub mod theme;

pub use app::DebuggerApp;
