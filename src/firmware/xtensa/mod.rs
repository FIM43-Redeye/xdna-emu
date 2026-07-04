//! In-tree base-Xtensa interpreter (decoder, register file, execution core).

pub mod decode;
pub mod interp;
pub mod mmu;
pub mod regfile;

#[cfg(test)]
mod coverage_scan;
