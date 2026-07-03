//! CDO application logic.
//!
//! Stage 8b Half 2: `apply_cdo` iterates `DeviceOp`s produced by
//! `crate::parser::cdo::semantics::lower` instead of matching on
//! `CdoRaw` directly. The parser/state type boundary is now
//! `DeviceOp`; state never sees a raw CDO opcode again.
//!
//! Statistics tracking is retained on the `CdoRaw` pre-lowering pass
//! so the `CdoStats` counters stay meaningful (they are CdoRaw-shaped
//! counters, not DeviceOp-shaped).

use anyhow::Result;

use xdna_archspec::types::TileAddr;

use crate::device::ops::DeviceOp;

use super::*;

impl DeviceState {
    /// Apply a CDO to configure the device.
    ///
    /// For each raw CDO command we lower to one or more `DeviceOp`s and
    /// apply each in turn. Stats are counted per-`CdoRaw` (the unit the
    /// CDO stream delivers), before lowering, so the counter semantics
    /// are unchanged from the pre-refactor implementation.
    pub fn apply_cdo(&mut self, cdo: &Cdo) -> Result<()> {
        self.stats = CdoStats::default();

        for cmd in cdo.commands() {
            self.stats.commands += 1;
            self.count_cdo_stats(&cmd)?;
            for op in crate::parser::cdo::semantics::lower(&cmd) {
                self.apply_device_op(&op)?;
            }
        }

        if self.stats.unknown > 0 {
            log::warn!(
                "CDO application complete: {} commands processed, {} unknown opcodes skipped",
                self.stats.commands,
                self.stats.unknown
            );
        }

        // Firmware kernel-launch anchor (#140 SP-4a, Part 1): the CDO leaves
        // enabled cores held in reset (CORE_CONTROL=0x03); on hardware the
        // firmware deasserts reset to start them. Modeled at config-completion
        // here (behavior-preserving: cores start ~at CDO time, as before Part 1
        // introduced the reset-honoring run-state). Part 2 moves this to the
        // true launch timing.
        self.release_core_resets();

        Ok(())
    }

    /// Bump the CdoRaw-shaped counters for this command.
    ///
    /// Counters are shaped around the raw CDO opcode (writes, mask
    /// writes, DMA writes, NOPs, unknowns), so they must be tallied
    /// before lowering -- a single `CdoRaw::Write` that promotes to
    /// `DeviceOp::CoreEnable` is still one "write" for stats purposes.
    ///
    /// Also preserves the original trace-level logging and the
    /// hard-error on `Unknown` opcodes so behavior outside the CDO
    /// execution remains unchanged.
    fn count_cdo_stats(&mut self, cmd: &CdoRaw) -> Result<()> {
        match cmd {
            CdoRaw::Nop { .. } => {
                self.stats.nops += 1;
            }
            CdoRaw::Write { address, value } => {
                self.stats.writes += 1;
                let tile_addr = TileAddress::decode(*address);
                log::trace!(
                    "CDO Write: addr=0x{:08X} -> tile({},{}) offset=0x{:05X} value=0x{:08X}",
                    address,
                    tile_addr.col,
                    tile_addr.row,
                    tile_addr.offset,
                    value
                );
            }
            CdoRaw::MaskWrite { address, mask, value } => {
                self.stats.mask_writes += 1;
                let tile_addr = TileAddress::decode(*address);
                log::trace!("CDO MaskWrite: addr=0x{:08X} -> tile({},{}) offset=0x{:05X} mask=0x{:08X} value=0x{:08X}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset, mask, value);
            }
            CdoRaw::Write64 { address, value } => {
                self.stats.writes += 1;
                let addr32 = *address as u32;
                let tile_addr = TileAddress::decode(addr32);
                log::trace!(
                    "CDO Write64: addr=0x{:016X} -> tile({},{}) offset=0x{:05X} value=0x{:08X}",
                    address,
                    tile_addr.col,
                    tile_addr.row,
                    tile_addr.offset,
                    value
                );
            }
            CdoRaw::MaskWrite64 { address, mask: _, value: _ } => {
                self.stats.mask_writes += 1;
                let addr32 = *address as u32;
                let tile_addr = TileAddress::decode(addr32);
                log::trace!(
                    "CDO MaskWrite64: addr=0x{:016X} -> tile({},{}) offset=0x{:05X}",
                    address,
                    tile_addr.col,
                    tile_addr.row,
                    tile_addr.offset
                );
            }
            CdoRaw::DmaWrite { address, data } => {
                self.stats.dma_writes += 1;
                let tile_addr = TileAddress::decode(*address);
                let subsystem = subsystem_from_offset(tile_addr.offset, tile_kind_from_row(tile_addr.row));
                log::debug!(
                    "CDO DmaWrite: addr=0x{:08X} -> tile({},{}) offset=0x{:05X} subsystem={:?} len={}",
                    address,
                    tile_addr.col,
                    tile_addr.row,
                    tile_addr.offset,
                    subsystem,
                    data.len()
                );
            }
            CdoRaw::MaskPoll { .. } | CdoRaw::MaskPoll64 { .. } => {
                log::trace!("CDO MaskPoll: skipped (emulator writes are synchronous)");
            }
            CdoRaw::Delay { .. } => {
                log::trace!("CDO Delay: skipped (emulator has no real-time clock)");
            }
            CdoRaw::EndMark | CdoRaw::Marker { .. } => {
                log::trace!("CDO marker/end: skipped");
            }
            CdoRaw::Unknown { opcode, payload } => {
                self.stats.unknown += 1;
                anyhow::bail!(
                    "CDO opcode {:#06x} not implemented ({} payload words) -- unknown hardware config",
                    opcode,
                    payload.len(),
                );
            }
        }
        Ok(())
    }

    /// Apply a single `DeviceOp` to the device.
    ///
    /// Register-level ops (`RegWrite` / `RegMask` / `RegBurst`) rebuild
    /// the encoded 32-bit address from `(tile, offset)` and route
    /// through the existing `write_register` / `mask_write_register` /
    /// `dma_write` dispatchers. That keeps the internal shim/memtile/
    /// compute routing as the single source of truth for raw register
    /// writes and avoids duplicating tile-local effect logic.
    ///
    /// Structured ops (`CoreEnable`, `DmaStart`) also route through
    /// `write_register`: the existing offset-dispatch branches in
    /// `write_core_register` (CORE_CONTROL) and `write_dma_channel`
    /// (Start_Queue) are the single source of truth for the typed
    /// effects, so the CDO and non-CDO paths produce identical
    /// observable state. The typed variants survive on `DeviceOp`
    /// as parser-side semantic markers, not parallel device
    /// handlers (see D.3 spec for the rationale).
    pub(super) fn apply_device_op(&mut self, op: &DeviceOp) -> Result<()> {
        // Rebase every operation's logical tile column to the partition's
        // physical column, matching what the xdna-driver does at allocation
        // time. Defaults to a no-op (`start_col == 0`) until the test
        // runner calls `set_start_col`.
        let shift = self.start_col;
        let physical_tile = |t: TileAddr| TileAddr { col: t.col + shift, row: t.row };
        match op {
            DeviceOp::RegWrite { tile, offset, value } => {
                let addr = encode_addr(physical_tile(*tile), *offset);
                self.write_register(addr, *value)?;
            }
            DeviceOp::RegMask { tile, offset, mask, value } => {
                let addr = encode_addr(physical_tile(*tile), *offset);
                self.mask_write_register(addr, *mask, *value)?;
            }
            DeviceOp::RegBurst { tile, offset, words } => {
                let addr = encode_addr(physical_tile(*tile), *offset);
                // Flatten back to a byte buffer for the existing
                // `dma_write` path. Little-endian matches the
                // round-trip used by `semantics::lower::bytes_to_words_le`.
                let mut bytes: Vec<u8> = Vec::with_capacity(words.len() * 4);
                for w in words.iter() {
                    bytes.extend_from_slice(&w.to_le_bytes());
                }
                self.dma_write(addr, &bytes)?;
            }
            DeviceOp::CoreEnable { tile, enabled: _, value } => {
                // Route through the universal register bus. The
                // typed `enabled` flag is intentionally ignored on
                // apply -- `write_core_register`'s CORE_CONTROL
                // branch derives it from `value & 1`. The variant
                // keeps `enabled` as a parser-side semantic marker
                // (room for option (a) future work; see D.3 spec).
                let addr = encode_addr(physical_tile(*tile), xdna_archspec::aie2::registers::CORE_CONTROL);
                self.write_register(addr, *value)?;
            }
            DeviceOp::DmaStart { tile, channel, dir, bd_id } => {
                // Route through the universal register bus.
                // `start_queue_offset` reconstructs the offset from
                // (row, channel, dir); `write_register` then dispatches
                // to `write_dma_channel` (compute) or
                // `write_memtile_dma_channel` / `write_shim_dma_channel`
                // (memtile / shim), each of which has the existing
                // Start_Queue branch that does the typed effect.
                let offset = start_queue_offset(tile.row, *channel, *dir)?;
                let addr = encode_addr(physical_tile(*tile), offset);
                self.write_register(addr, *bd_id)?;
            }
            DeviceOp::MaskPoll { .. } => {
                // On real hardware MaskPoll blocks until the condition is
                // met; in the emulator register writes are synchronous so
                // the condition is always satisfied by the time we get
                // here. Trace-level log only.
                log::trace!("DeviceOp::MaskPoll: no-op (emulator writes are synchronous)");
            }
            DeviceOp::Delay { .. } => {
                log::trace!("DeviceOp::Delay: no-op (emulator has no real-time clock)");
            }
            DeviceOp::Marker { .. } => {
                log::trace!("DeviceOp::Marker: no-op");
            }
        }
        Ok(())
    }
}

/// Rebuild the 32-bit encoded tile address from a `TileAddr` and
/// tile-local offset. Uses the existing `TileAddress::encode` helper
/// so the shift layout stays centralized.
#[inline]
fn encode_addr(tile: TileAddr, offset: u32) -> u32 {
    TileAddress::encode(tile.col, tile.row, offset)
}

/// Inverse of `match_dma_start_queue` in `parser::cdo::semantics`:
/// reconstruct the Start_Queue / Task_Queue offset for a tile
/// from `(row, channel, dir)`. Used by `apply_device_op::DmaStart`
/// to build the address it routes through `write_register`.
///
/// Returns an error for invalid (channel, dir) combinations on the
/// detected tile kind. The lowering side never produces such
/// combinations (it derives them by matching valid offsets), so
/// this branch is defensive: an error here means an upstream bug.
fn start_queue_offset(row: u8, channel: u8, dir: xdna_archspec::types::DmaDirection) -> Result<u32> {
    use xdna_archspec::aie2::registers::dma::{
        COMPUTE_DMA_MM2S_0_START_QUEUE, COMPUTE_DMA_MM2S_1_START_QUEUE, COMPUTE_DMA_S2MM_0_START_QUEUE,
        COMPUTE_DMA_S2MM_1_START_QUEUE, MEMTILE_CHANNELS_PER_DIR, MEMTILE_CHANNEL_STRIDE,
        MEMTILE_DMA_MM2S_0_START_QUEUE, MEMTILE_DMA_S2MM_0_START_QUEUE, SHIM_CHANNELS_PER_DIR,
        SHIM_CHANNEL_STRIDE, SHIM_DMA_MM2S_0_TASK_QUEUE, SHIM_DMA_S2MM_0_TASK_QUEUE,
    };
    use xdna_archspec::types::DmaDirection;

    match tile_kind_from_row(row) {
        TileKind::Compute => match (dir, channel) {
            (DmaDirection::S2mm, 0) => Ok(COMPUTE_DMA_S2MM_0_START_QUEUE),
            (DmaDirection::S2mm, 1) => Ok(COMPUTE_DMA_S2MM_1_START_QUEUE),
            (DmaDirection::Mm2s, 0) => Ok(COMPUTE_DMA_MM2S_0_START_QUEUE),
            (DmaDirection::Mm2s, 1) => Ok(COMPUTE_DMA_MM2S_1_START_QUEUE),
            _ => {
                anyhow::bail!("start_queue_offset: invalid compute DMA channel dir={:?} ch={}", dir, channel)
            }
        },
        TileKind::Mem => {
            if channel >= MEMTILE_CHANNELS_PER_DIR {
                anyhow::bail!(
                    "start_queue_offset: memtile channel {} >= {}",
                    channel,
                    MEMTILE_CHANNELS_PER_DIR
                );
            }
            let base = match dir {
                DmaDirection::S2mm => MEMTILE_DMA_S2MM_0_START_QUEUE,
                DmaDirection::Mm2s => MEMTILE_DMA_MM2S_0_START_QUEUE,
            };
            Ok(base + (channel as u32) * MEMTILE_CHANNEL_STRIDE)
        }
        TileKind::ShimNoc | TileKind::ShimPl => {
            // tile_kind_from_row currently always returns ShimNoc for row 0;
            // ShimPl is matched here for enum exhaustiveness so the helper
            // stays correct if the row->kind classification ever changes.
            if channel >= SHIM_CHANNELS_PER_DIR {
                anyhow::bail!("start_queue_offset: shim channel {} >= {}", channel, SHIM_CHANNELS_PER_DIR);
            }
            let base = match dir {
                DmaDirection::S2mm => SHIM_DMA_S2MM_0_TASK_QUEUE,
                DmaDirection::Mm2s => SHIM_DMA_MM2S_0_TASK_QUEUE,
            };
            Ok(base + (channel as u32) * SHIM_CHANNEL_STRIDE)
        }
    }
}
