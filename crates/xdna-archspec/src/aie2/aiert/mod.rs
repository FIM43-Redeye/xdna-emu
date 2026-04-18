//! aie-rt cross-validated constants.
//!
//! Each submodule wraps a build-time-generated file (gen_aiert_*.rs)
//! containing data extracted from aie-rt headers via C preprocessor.
//! The data is also used for ArchModel cross-validation in archspec's
//! build.rs; this module exposes the same data for xdna-emu's
//! aiert_validation runtime tests.
//!
//! When aie-rt is not present at build time, stub files with hardcoded
//! fallback values are emitted instead (see `write_aiert_stubs` in
//! archspec's build.rs).

/// DMA module constants extracted from aie-rt xaiemlgbl_reginit.c.
///
/// Submodules: `compute_dma`, `memtile_dma`, `shim_dma`.
/// Each provides `BD_BASE`, `BD_STRIDE`, `NUM_BDS`, `NUM_CHANNELS`, etc.
pub mod dma {
    include!(concat!(env!("OUT_DIR"), "/gen_aiert_dma.rs"));
}

/// Lock module constants extracted from aie-rt xaiemlgbl_reginit.c.
///
/// Submodules: `compute_locks`, `memtile_locks`, `shim_locks`.
/// Each provides `BASE`, `NUM_LOCKS`, `SET_VAL_BASE`, `SET_VAL_STRIDE`, etc.
pub mod locks {
    include!(concat!(env!("OUT_DIR"), "/gen_aiert_locks.rs"));
}

/// Stream switch port map constants extracted from aie-rt xaiemlgbl_reginit.c.
///
/// Provides `AieRtPortType` enum and constants:
/// `COMPUTE_MASTER_PORTS`, `COMPUTE_SLAVE_PORTS`,
/// `MEMTILE_MASTER_PORTS`, `MEMTILE_SLAVE_PORTS`,
/// `SHIM_MASTER_PORTS`, `SHIM_SLAVE_PORTS`.
pub mod ports {
    include!(concat!(env!("OUT_DIR"), "/gen_aiert_ports.rs"));
}
