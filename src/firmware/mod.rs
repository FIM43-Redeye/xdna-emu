//! Generic management-processor support: firmware container parsing and a
//! base-Xtensa interpreter.
//!
//! The interpreter in [`xtensa`] runs against any memory map implementing
//! [`xtensa::bus::XtensaBus`]. Platform crates supply the map for a specific
//! chip; none ships in this crate.

mod error;
mod image;
pub mod xtensa;

pub use error::FirmwareError;
pub use image::FirmwareImage;
