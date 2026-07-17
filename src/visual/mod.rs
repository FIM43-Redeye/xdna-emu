//! Live visual debugger (egui). Gated behind the `gui` feature. All logic
//! lives in crate::debugger (egui-free, tested); this layer only renders.
pub mod app;
pub mod controls;
pub mod detail;
pub mod floorplan;
pub mod overview;
pub mod routes;
pub mod theme;
pub mod tile;

pub use app::DebuggerApp;
