//! Control-path cycle-cost model for NPU instruction execution.
//!
//! Real AIE2 hardware retires control packets through a multi-stage path:
//! the host writes the doorbell, the IPU command processor (CMP) fetches
//! the packet from its queue, decodes the header, and issues one or more
//! AXI writes through the NoC fabric to the target tile. Each stage has
//! its own latency.
//!
//! # Schema
//!
//! AMD's proprietary AIE simulator (aietools) reads a JSON config keyed
//! `AIE_CONTROL_PATH_LATENCY` whose subkeys are:
//!
//! - `AIE_TILE_BD`: BD config write to a compute tile
//! - `MEM_TILE_BD`: BD config write to a memtile
//! - `SHIM_TILE_BD`: BD config write to a shim tile
//! - `WRITE_32`: any other Write32 (the catch-all)
//!
//! Source: strings extracted from
//! `aietools/lib/lnx64.o/libaie2_cluster_msm_v1_0_0.osci.so` -- the schema
//! is documented in the binary's string table even though the numeric
//! defaults are hardcoded inside the proprietary library and require
//! disassembly to recover. This module mirrors the schema so calibration
//! data can be applied directly when it lands.
//!
//! # Structural pieces (open-source-derived)
//!
//! Once a packet leaves the CMP, its propagation cost is decomposable into
//! pieces we *do* know from open-source documentation:
//!
//! - **Stream-switch fabric hop**: 3 cycles for an in-tile path (6-deep
//!   FIFO), 4 cycles for any boundary-crossing path (8-deep FIFO). Source:
//!   AM020 ch.2 "AIE-ML Tile Architecture".
//! - **PLIO bridge** (only relevant if a packet enters via PL): baseline
//!   4 clocks AIE->PL, 3 clocks PL->AIE. Source:
//!   `aietools/.../aie_xtlm/aie_xtlm_v1_0_0/src/aie_xtlm.cpp:202-244`.
//! - **Register write completion** at the target: 1 cycle. Source: AM025.
//!
//! # Phase 1 (this module)
//!
//! The framework lays down all the categories with documented sources.
//! The `legacy_one_per_packet()` profile preserves the prior "1 cycle per
//! packet" behavior so existing tests don't shift. The
//! `with_known_constants()` profile enables the known structural pieces
//! (fabric hops, register write) but keeps unknowns at 1.
//!
//! # Phase 2/3 (#322/#323)
//!
//! Calibrate the unknowns (per-tile-type BD cost, generic Write32 cost,
//! CMP-to-shim NoC entry, MaskPoll iteration cost) against real-NPU
//! traces using the trace-sweep harness. The model API is shaped around
//! AMD's JSON schema so calibrated values can ship as a config file
//! mirroring `AIE_CONTROL_PATH_LATENCY.*`.

use super::parser::NpuInstruction;
use crate::device::registers::TileAddress;
use xdna_archspec::aie2::topology::Aie2Topology;
use xdna_archspec::topology::TileTopology;
use xdna_archspec::types::TileKind;

/// NPU1 (Phoenix) topology -- 5 columns, 6 rows, 1 memtile row.
///
/// Phase 1 of the cycle-cost framework hardcodes the topology. Once
/// AIE2P comes online the model gains a `&dyn TileTopology` field and
/// the executor passes the live arch through.
const NPU1_TOPOLOGY: Aie2Topology = Aie2Topology { columns: 5, rows: 6, num_mem_tile_rows: 1 };

/// Per-component cycle costs that compose the total cost of a control
/// packet on the IPU command-path.
///
/// Each field documents whether the value is **derived** (read from
/// open-source toolchain or AMD architecture documentation) or
/// **calibrated** (placeholder pending #322/#323 empirical work).
#[derive(Debug, Clone)]
pub struct CycleCostModel {
    // === CMP decode + dispatch (calibrated) =========================
    //
    // Mirrors AMD's `AIE_CONTROL_PATH_LATENCY.*` JSON schema discovered
    // via string extraction from libaie2_cluster_msm_v1_0_0.osci.so.
    /// CMP decode + dispatch cost for a BD config write to a compute
    /// (AIE) tile. Schema name: `AIE_TILE_BD`.
    pub aie_tile_bd: u64,
    /// Same, for a memtile. Schema name: `MEM_TILE_BD`.
    pub mem_tile_bd: u64,
    /// Same, for a shim tile. Schema name: `SHIM_TILE_BD`.
    pub shim_tile_bd: u64,
    /// CMP decode + dispatch for any other Write32 (not a tile BD
    /// config). Schema name: `WRITE_32`.
    pub write_32: u64,

    /// MaskWrite is a read-modify-write packet -- two AXI transactions
    /// plus the RMW serialisation cost. Modelled as additive overhead
    /// on top of the corresponding Write32 cost. (calibrated)
    pub mask_write_overhead: u64,

    /// Sync packet retirement cost. (calibrated)
    pub sync: u64,

    /// Each iteration of a MaskPoll loop while waiting for the polled
    /// register to converge. (calibrated)
    pub mask_poll_iter: u64,

    /// Per-payload-word cost on top of the base BlockWrite header cost
    /// -- each word in the payload becomes its own AXI write at the
    /// target. (calibrated; conservative default 0 means current
    /// behavior is preserved)
    pub block_write_per_word: u64,

    /// Cycles for a packet to traverse from the IPU command processor
    /// out to the shim entry point of the array. (calibrated; this is
    /// the part of the path the open-source toolchain doesn't expose)
    pub cmp_to_shim: u64,

    // === Structural pieces (derived) ===============================
    /// Stream-switch hop, in-tile path. Derived from AM020 ch.2
    /// "AIE-ML Tile Architecture" (3 cycles, 6-deep FIFO).
    pub fabric_hop_local: u64,
    /// Stream-switch hop, boundary-crossing path. Derived from AM020
    /// ch.2 (4 cycles, 8-deep FIFO).
    pub fabric_hop_boundary: u64,

    /// PLIO bridge AIE->PL baseline. Derived from
    /// `aie_xtlm.cpp:202` (`l_ap_ft_bli = 4.0`). Modifiers from
    /// AUTOPIPE_LINE / FIFO_TYPE / registered status are not currently
    /// modelled (they're per-instance configuration that our trace
    /// profile doesn't expose yet).
    pub plio_aie_to_pl: u64,
    /// PLIO bridge PL->AIE baseline. Derived from `aie_xtlm.cpp:232`.
    pub plio_pl_to_aie: u64,

    /// Register write completion at the target. Derived from AM025.
    pub register_write: u64,
}

impl CycleCostModel {
    /// Backward-compatible profile: every packet costs exactly one
    /// simulation cycle, with no structural pieces engaged.
    ///
    /// This matches the behavior of the original `cycle_cost()` method
    /// and is the default so existing tests don't shift.
    pub fn legacy_one_per_packet() -> Self {
        Self {
            aie_tile_bd: 1,
            mem_tile_bd: 1,
            shim_tile_bd: 1,
            write_32: 1,
            mask_write_overhead: 0,
            sync: 1,
            mask_poll_iter: 1,
            block_write_per_word: 0,
            cmp_to_shim: 0,
            fabric_hop_local: 0,
            fabric_hop_boundary: 0,
            plio_aie_to_pl: 0,
            plio_pl_to_aie: 0,
            register_write: 0,
        }
    }

    /// Profile that engages the **derived** structural pieces from open
    /// sources, leaving CMP / per-tile-type / sync / mask-poll costs at
    /// the conservative one-cycle placeholder.
    ///
    /// This is the profile to use once #322 calibrates per-category
    /// CMP costs -- structural costs land first because they're already
    /// known.
    pub fn with_known_constants() -> Self {
        Self {
            // Calibrated placeholders -- still 1 cycle until #322 lands.
            aie_tile_bd: 1,
            mem_tile_bd: 1,
            shim_tile_bd: 1,
            write_32: 1,
            mask_write_overhead: 0,
            sync: 1,
            mask_poll_iter: 1,
            block_write_per_word: 0,
            cmp_to_shim: 0,

            // Derived from open sources.
            fabric_hop_local: 3,    // AM020 ch.2
            fabric_hop_boundary: 4, // AM020 ch.2
            plio_aie_to_pl: 4,      // aie_xtlm.cpp:202
            plio_pl_to_aie: 3,      // aie_xtlm.cpp:232
            register_write: 1,      // AM025
        }
    }

    /// Provisional NPU1 calibration -- fast-mode empirical values from the
    /// `tools/calibration/` harness. **The values here are likely garbage**.
    ///
    /// This is a horrible kludge excuse for a control-path / NoC cost model.
    /// AMD has the real numbers (gated behind the un-shipped
    /// `AIE_CONTROL_PATH_LATENCY` JSON that
    /// `libaie2_cluster_msm_v1_0_0.osci.so` reads). The on-NPU readback
    /// path that would produce trace-independent ground truth
    /// (Performance_Counter0 readback via `xrt::hw_context::read_aie_reg`)
    /// **is now functional on Phoenix as of 2026-05-05** -- earlier
    /// "firmware-gated" diagnoses were wrong. Phoenix firmware 1.5.5.391
    /// fully implements `MSG_OP_AIE_RW_ACCESS`; the only obstacle was a
    /// missing entry in the driver's `npu1_regs.c` op-table, which
    /// `aie2_is_supported_msg` consults before letting the message through.
    /// One-line driver fix unblocks the path. See
    /// `docs/superpowers/findings/2026-05-05-aie-rw-access-firmware-actually-supported.md`
    /// for the breakthrough writeup; #356 tracks the integration. A
    /// separate lifecycle bug in `bridge-trace-runner.cpp` still prevents
    /// trace-independent cycle counts from being recorded automatically;
    /// the read path itself is good.
    ///
    /// What's encoded here:
    /// - `aie_tile_bd` / `mem_tile_bd` / `shim_tile_bd` / `write_32`: 100 cyc.
    ///   Empirical write32 cost on real NPU1 is ~100.5 cyc/pkt averaged
    ///   over a period-2 modulation (87 cyc on even N, 114 cyc on odd N --
    ///   a 5-stage CMP pipeline thing, presumably). Per-tile-type variation
    ///   was within 0.5 cyc across shim/mem/compute targets. Distance from
    ///   anchor was within 1.34 cyc across the npu1 4x6 array -- effectively
    ///   free. So one number across all tile types, integer-rounded.
    /// - `mask_write_overhead`: +110 cyc. Empirical maskwrite ~210 cyc/pkt
    ///   (intrinsically noisy at ~50 cyc per-rep spread, period structure
    ///   not resolved). Subtract write32 base for the RMW overhead.
    /// - `block_write_per_word`: 13 cyc. Derived from blockwrite
    ///   payload=8 = 203 cyc/pkt minus the 100 cyc write32 base, divided
    ///   by 8 words. The legitimate per-word cost from AMD's schema is
    ///   what should land here once we get it.
    /// - `cmp_to_shim`: 0. Distance-independent slope confirmed empirically
    ///   across 24 tiles and 6 distances; per-hop fit was -0.05 cyc with
    ///   R^2=0.03 (noise).
    /// - `sync` / `mask_poll_iter`: not calibrated, kept at 1.
    ///
    /// What's NOT modelled (deliberately):
    /// - One-time +91 cyc step at write32 N=23 (and analogous events at
    ///   blockwrite N=12, maskwrite ~N=12)
    /// - Stochastic +2780 cyc slow-mode artifact starting at ~3500 NPU
    ///   cycles into a kernel run, locking to 100% probability by ~8000
    ///   cycles.
    ///
    /// Both artifacts trigger at fixed *cycle* thresholds across packet
    /// kinds, which is the signature of a concurrent cycle-clock-driven
    /// subsystem -- most likely the trace controller flushing on its own
    /// internal timer, *not* real CMP behaviour. Working hypothesis: they
    /// are measurement-side and would not apply to a no-trace kernel.
    /// We can't confirm without on-NPU timing.
    ///
    /// See `docs/superpowers/findings/2026-05-04-control-path-cycle-calibration.md`
    /// for the full methodology, the negative results from diagnostic
    /// tests, and what it would take to ship a real model.
    pub fn provisional_npu1() -> Self {
        Self {
            // Empirical fast-mode values. See doc-comment above.
            aie_tile_bd: 100,
            mem_tile_bd: 100,
            shim_tile_bd: 100,
            write_32: 100,
            mask_write_overhead: 110,
            block_write_per_word: 13,
            cmp_to_shim: 0,

            // Not calibrated -- kept at conservative placeholders.
            sync: 1,
            mask_poll_iter: 1,

            // Derived from open sources (same as with_known_constants).
            fabric_hop_local: 3,    // AM020 ch.2
            fabric_hop_boundary: 4, // AM020 ch.2
            plio_aie_to_pl: 4,      // aie_xtlm.cpp:202
            plio_pl_to_aie: 3,      // aie_xtlm.cpp:232
            register_write: 1,      // AM025
        }
    }

    /// Compute the total retirement cost of an instruction.
    ///
    /// This is the source of truth the executor calls. Decodes the
    /// target address (where applicable) to determine which CMP-decode
    /// category applies, sums in the structural fabric pieces, and
    /// returns a single cycle count.
    pub fn cost_of(&self, instr: &NpuInstruction) -> u64 {
        match instr {
            NpuInstruction::Write32 { reg_off, .. } => {
                self.cmp_decode_cost(*reg_off)
                    + self.cmp_to_shim
                    + self.fabric_cost(*reg_off)
                    + self.register_write
            }
            NpuInstruction::BlockWrite { reg_off, values } => {
                let header = self.cmp_decode_cost(*reg_off);
                let payload = self.block_write_per_word.saturating_mul(values.len() as u64);
                let fabric = self.fabric_cost(*reg_off);
                header + payload + self.cmp_to_shim + fabric + self.register_write
            }
            NpuInstruction::MaskWrite { reg_off, .. } => {
                let base = self.cmp_decode_cost(*reg_off);
                base + self.mask_write_overhead
                    + self.cmp_to_shim
                    + self.fabric_cost(*reg_off)
                    + self.register_write
            }
            NpuInstruction::MaskPoll { .. } => self.mask_poll_iter,
            NpuInstruction::Sync { .. } => self.sync,
            NpuInstruction::DdrPatch { .. } => self.write_32,
            NpuInstruction::Unknown { .. } => 1,
        }
    }

    /// Classify the address into a CMP-decode category.
    ///
    /// AMD's schema distinguishes BD config writes per tile type. We
    /// currently treat any write to a tile as a BD config (the dominant
    /// case in compiled mlir-aie xclbins); a future refinement can
    /// inspect the register offset to pick out non-BD writes that
    /// should fall into the WRITE_32 catch-all.
    fn cmp_decode_cost(&self, reg_off: u32) -> u64 {
        let tile = TileAddress::decode(reg_off);
        match NPU1_TOPOLOGY.classify(tile.col, tile.row) {
            TileKind::ShimNoc | TileKind::ShimPl => self.shim_tile_bd,
            TileKind::Mem => self.mem_tile_bd,
            TileKind::Compute => self.aie_tile_bd,
        }
    }

    /// Cost of getting a packet from the shim entry point to the target
    /// tile through the stream-switch fabric.
    ///
    /// Each row crossed is one boundary hop; staying within the
    /// destination tile is one local hop for the final delivery. This is
    /// a conservative under-count -- horizontal traversal is not yet
    /// modelled because mlir-aie's compiled xclbins for NPU1 only enter
    /// from the column the destination lives in. Cross-column entry can
    /// be added once we calibrate against multi-column traces.
    fn fabric_cost(&self, reg_off: u32) -> u64 {
        let tile = TileAddress::decode(reg_off);
        // Row 0 = shim (entry point). Rows 1..=N cross row boundaries.
        let row_hops = tile.row as u64;
        row_hops.saturating_mul(self.fabric_hop_boundary) + self.fabric_hop_local
    }
}

impl Default for CycleCostModel {
    /// Default profile uses the **provisional NPU1 calibration** -- the
    /// best empirical numbers we have despite the caveats documented on
    /// `CycleCostModel::provisional_npu1()`. Switching the default to
    /// these values gives the executor cycle counts in the right ballpark
    /// (~100 cyc per control packet) instead of the placeholder ~5-15 cyc
    /// you got with `with_known_constants()`.
    ///
    /// To preserve the prior 1-cycle-per-packet behaviour for a specific
    /// caller, opt in explicitly with
    /// `CycleCostModel::legacy_one_per_packet()`.
    fn default() -> Self {
        Self::provisional_npu1()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_profile_matches_one_per_packet_behavior() {
        let m = CycleCostModel::legacy_one_per_packet();
        let w = NpuInstruction::Write32 { reg_off: 0, value: 0 };
        assert_eq!(m.cost_of(&w), 1, "Write32 should cost exactly 1 cycle in legacy profile");

        let bw = NpuInstruction::BlockWrite { reg_off: 0, values: vec![0; 8] };
        assert_eq!(m.cost_of(&bw), 1, "BlockWrite should cost exactly 1 cycle in legacy profile");

        let mp = NpuInstruction::MaskPoll { reg_off: 0, value: 0, mask: 0 };
        assert_eq!(m.cost_of(&mp), 1);

        let sy =
            NpuInstruction::Sync { channel: 0, column: 0, direction: 0, column_num: 1, row: 0, row_num: 1 };
        assert_eq!(m.cost_of(&sy), 1);
    }

    #[test]
    fn known_constants_engages_fabric_hops() {
        let m = CycleCostModel::with_known_constants();

        // Shim tile at (0,0): fabric_cost = 0*4 + 3 = 3, plus shim_tile_bd=1, register_write=1
        let shim_addr = TileAddress::encode(0, 0, 0);
        let w_shim = NpuInstruction::Write32 { reg_off: shim_addr, value: 0 };
        assert_eq!(
            m.cost_of(&w_shim),
            1 + 0 + 3 + 1,
            "Shim tile Write32: shim_tile_bd + cmp_to_shim + fabric_local + register_write"
        );

        // Compute tile at (0,2): fabric_cost = 2*4 + 3 = 11, plus aie_tile_bd=1, register_write=1
        let compute_addr = TileAddress::encode(0, 2, 0);
        let w_compute = NpuInstruction::Write32 { reg_off: compute_addr, value: 0 };
        assert_eq!(
            m.cost_of(&w_compute),
            1 + 0 + 11 + 1,
            "Compute tile Write32 should include 2 boundary hops + 1 local"
        );

        // Memtile at (0,1): fabric_cost = 1*4 + 3 = 7, plus mem_tile_bd=1, register_write=1
        let mem_addr = TileAddress::encode(0, 1, 0);
        let w_mem = NpuInstruction::Write32 { reg_off: mem_addr, value: 0 };
        assert_eq!(m.cost_of(&w_mem), 1 + 0 + 7 + 1);
    }

    #[test]
    fn block_write_payload_scales_with_word_count() {
        let mut m = CycleCostModel::legacy_one_per_packet();
        m.block_write_per_word = 2;
        m.aie_tile_bd = 5;

        let bw = NpuInstruction::BlockWrite { reg_off: TileAddress::encode(0, 2, 0), values: vec![0; 10] };
        // header (5) + payload (10*2=20) + cmp_to_shim (0) + fabric (0) + register_write (0)
        assert_eq!(m.cost_of(&bw), 25);
    }

    #[test]
    fn mask_write_adds_overhead_on_top_of_base() {
        let mut m = CycleCostModel::legacy_one_per_packet();
        m.aie_tile_bd = 5;
        m.mask_write_overhead = 3;

        let mw = NpuInstruction::MaskWrite { reg_off: TileAddress::encode(0, 2, 0), value: 0, mask: 0 };
        // base (5) + mask_overhead (3) + zeros for the rest in legacy
        assert_eq!(m.cost_of(&mw), 8);
    }

    #[test]
    fn default_is_provisional_npu1() {
        let d = CycleCostModel::default();
        let p = CycleCostModel::provisional_npu1();
        let w = NpuInstruction::Write32 { reg_off: 0, value: 0 };
        assert_eq!(d.cost_of(&w), p.cost_of(&w));
        // Sanity: provisional CMP cost should be ~100, far above the
        // 1-cyc placeholder in with_known_constants.
        assert!(p.write_32 >= 50, "provisional write_32 should reflect calibration");
    }
}
