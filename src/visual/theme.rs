//! Visual debugger palette. Functional-first; polish is a later pass.
use eframe::egui::Color32;

pub const BG: Color32 = Color32::from_rgb(24, 26, 30);
pub const TILE_SHIM: Color32 = Color32::from_rgb(90, 70, 120);
pub const TILE_MEM: Color32 = Color32::from_rgb(60, 100, 120);
pub const TILE_CORE: Color32 = Color32::from_rgb(70, 110, 80);
pub const TILE_SELECTED: Color32 = Color32::from_rgb(230, 200, 90);
pub const TILE_LABEL: Color32 = Color32::from_rgb(220, 220, 220);
pub const PORT_ACTIVE: Color32 = Color32::from_rgb(120, 200, 120);
pub const PORT_STALLED: Color32 = Color32::from_rgb(220, 140, 90);
