//! DMA model trait: feature flags and timing carrier for per-arch DMA
//! behavior.
//!
//! Subsystem 3 of the device-family refactor introduces this trait as a
//! behavioral seam. AIE2 / AIE2P use the concrete `aie2::dma::Aie2DmaModel`
//! impl, which enables task queue, out-of-order, compression, and BD
//! iteration. AIE1's eventual `aie1::dma::Aie1DmaModel` will disable those
//! and enable interleave + double-buffer instead.
//!
//! Consumers access an impl via `ArchModel::dma_model()`, which returns a
//! `&'static dyn DmaModel`. Because every concrete impl is zero-sized and
//! stateless, the accessor returns a reference to a `static` singleton --
//! no allocation, no lifetime bookkeeping.
//!
//! The trait is intentionally coarse (9 methods, all cold-path). Hot-path
//! FSM dispatch stays tile-type-dispatched in xdna-emu; trait calls happen
//! at DmaEngine construction and at the ~5 call-site boundaries where AIE2
//! features are disabled for other archs.

use crate::types::TileKind;

/// DMA timing configuration (eight per-arch cycle constants).
///
/// Read once at DmaEngine construction via `DmaModel::timing_config()`.
/// AIE2's values come from the build-time-generated `xdna_archspec::aie2::timing`
/// submodule (which is derived from AM025 + aie-rt observation). AIE1's
/// impl will return AIE1-tuned values.
#[derive(Debug, Clone, Copy)]
pub struct DmaTimingConfig {
    /// Cycles to parse and setup a buffer descriptor.
    pub bd_setup_cycles: u8,

    /// Cycles from channel start to first data movement.
    pub channel_start_cycles: u8,

    /// Words (32-bit) transferred per cycle per channel.
    pub words_per_cycle: u8,

    /// Memory access latency in cycles.
    pub memory_latency_cycles: u8,

    /// Cycles to acquire a lock.
    pub lock_acquire_cycles: u8,

    /// Cycles to release a lock.
    pub lock_release_cycles: u8,

    /// Cycles between BD completion and next BD start.
    pub bd_chain_cycles: u8,

    /// Extra pipeline latency for shim tile DDR access (NoC + DDR controller).
    /// Applied once per BD, between MemoryLatency and Transferring, only for
    /// shim tiles with host memory endpoints.
    pub host_memory_latency_cycles: u16,
}

/// Per-arch DMA behavior, consulted at DmaEngine construction and at the
/// handful of call-site boundaries where AIE2 features are disabled for
/// other architectures.
pub trait DmaModel: Send + Sync {
    /// Per-channel 8-deep task queue. AIE2+: true. AIE1: false.
    ///
    /// Evidence: `xaie_dma_aieml.c:1257-1279` defines
    /// `_XAieMl_DmaWaitForBdTaskQueue`; `xaie_dma_aie.h` has no equivalent.
    fn supports_task_queue(&self) -> bool;

    /// Out-of-order S2MM packet completion. AIE2+: true. AIE1: false.
    ///
    /// Evidence: `xaie_dma_aieml.c:313-315, 459-461` sets
    /// `OutofOrderBdId` in BD word layout; AIE1 has no such field.
    fn supports_ooo_mode(&self) -> bool;

    /// Sparsity compression on MM2S / decompression on S2MM. AIE2+: true.
    /// AIE1: false.
    ///
    /// Evidence: `xaie_dma_aieml.c:360-362, 514-516` sets `EnCompression`;
    /// AIE1 BD has no compression field.
    fn supports_compression(&self) -> bool;

    /// Per-BD iteration (stepsize + wrap + current counter). AIE2+: true.
    /// AIE1: false.
    ///
    /// Evidence: `xaie_dma_aie.c:1065-1074` stubs
    /// `_XAie_DmaSetBdIteration` to return `XAIE_FEATURE_NOT_SUPPORTED`.
    fn supports_bd_iteration(&self) -> bool;

    /// Whether acquire and release lock IDs may differ on one BD.
    /// AIE2+: true (independent). AIE1: false (acq.lock_id == rel.lock_id
    /// enforced at BD apply time).
    ///
    /// Evidence: `xaie_dma_aie.c:113-116` hard-errors on mismatch; AIE2
    /// allows any combination.
    fn supports_independent_lock_ids(&self) -> bool;

    /// Interleave + LockDesc_2 double-buffer mode (AIE1-only FSM path).
    /// AIE2+: false. AIE1: true.
    ///
    /// Evidence: `xaie_dma_aie.c:189-198, 477-500, 543-545` configures BD
    /// word 5 with interleave + double-buffer fields; AIE2 has no such
    /// API and removed the BD fields.
    fn supports_interleave_mode(&self) -> bool;

    /// Paired with `supports_interleave_mode`. AIE1-only.
    fn supports_double_buffer(&self) -> bool;

    /// Maximum tensor dimensions by tile kind.
    ///
    /// AIE1: 2 for all tile kinds (2D X/Y addressing).
    /// AIE2 compute / shim: 3.
    /// AIE2 mem: 4 (the extra D3 for memtile).
    fn max_tensor_dims(&self, tile: TileKind) -> u8;

    /// The eight per-arch cycle constants. Read once at DmaEngine
    /// construction; not consulted per-cycle.
    fn timing_config(&self) -> DmaTimingConfig;
}

// `dma_model()` accessor on `ArchModel`.
//
// Placed here rather than `types.rs` because `types.rs` is
// `#[path]`-included by `build.rs`, which prevents it from importing
// the `dma` module it would need to return `&dyn DmaModel` from.
impl crate::types::ArchModel {
    /// Return the DMA model for this arch family.
    ///
    /// Dispatches on `self.arch`. Returns a `&'static dyn DmaModel`
    /// because every concrete impl is zero-sized and stateless; the AIE2
    /// impl is a single `AIE2_DMA_MODEL` static. AIE1 / AIE2P impls follow
    /// the same pattern when they land.
    pub fn dma_model(&self) -> &'static dyn DmaModel {
        match self.arch {
            crate::types::Architecture::Aie2
            | crate::types::Architecture::Aie2p => {
                &crate::aie2::dma::AIE2_DMA_MODEL
            }
            crate::types::Architecture::Aie => {
                unimplemented!(
                    "AIE1 DmaModel not populated until AIE1 support lands \
                     (tracked as post-Subsystem-3 follow-on work; see \
                     docs/arch/dma-model.md's 'What would AIE1 look like?' \
                     section)"
                )
            }
        }
    }
}
