//! Egui-free view-model and stepper for the visual debugger. Compiles without
//! the `gui` feature so its logic is covered by `cargo test --lib`. The egui
//! rendering layer (src/visual) consumes these types.
pub mod engine_host;
pub mod model;
