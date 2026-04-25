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
                log::trace!("CDO Write: addr=0x{:08X} -> tile({},{}) offset=0x{:05X} value=0x{:08X}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset, value);
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
                log::trace!("CDO Write64: addr=0x{:016X} -> tile({},{}) offset=0x{:05X} value=0x{:08X}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset, value);
            }
            CdoRaw::MaskWrite64 { address, mask: _, value: _ } => {
                self.stats.mask_writes += 1;
                let addr32 = *address as u32;
                let tile_addr = TileAddress::decode(addr32);
                log::trace!("CDO MaskWrite64: addr=0x{:016X} -> tile({},{}) offset=0x{:05X}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset);
            }
            CdoRaw::DmaWrite { address, data } => {
                self.stats.dma_writes += 1;
                let tile_addr = TileAddress::decode(*address);
                let subsystem = subsystem_from_offset(tile_addr.offset, tile_kind_from_row(tile_addr.row));
                log::debug!("CDO DmaWrite: addr=0x{:08X} -> tile({},{}) offset=0x{:05X} subsystem={:?} len={}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset, subsystem, data.len());
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
    /// Structured ops (`CoreEnable`, `DmaStart`) call dedicated helpers
    /// (`apply_core_enable`, `start_dma_channel`) that
    /// replicate the CDO-specific side effects that used to live
    /// inline in `write_core_register`'s CORE_CONTROL branch and
    /// `write_dma_channel`'s Start_Queue branch.
    pub(super) fn apply_device_op(&mut self, op: &DeviceOp) -> Result<()> {
        match op {
            DeviceOp::RegWrite { tile, offset, value } => {
                let addr = encode_addr(*tile, *offset);
                self.write_register(addr, *value)?;
            }
            DeviceOp::RegMask { tile, offset, mask, value } => {
                let addr = encode_addr(*tile, *offset);
                self.mask_write_register(addr, *mask, *value)?;
            }
            DeviceOp::RegBurst { tile, offset, words } => {
                let addr = encode_addr(*tile, *offset);
                // Flatten back to a byte buffer for the existing
                // `dma_write` path. Little-endian matches the
                // round-trip used by `semantics::lower::bytes_to_words_le`.
                let mut bytes: Vec<u8> = Vec::with_capacity(words.len() * 4);
                for w in words.iter() {
                    bytes.extend_from_slice(&w.to_le_bytes());
                }
                self.dma_write(addr, &bytes)?;
            }
            DeviceOp::CoreEnable { tile, enabled, value } => {
                self.apply_core_enable(tile.col, tile.row, *enabled, *value);
            }
            DeviceOp::DmaStart { tile, channel, dir, bd_id } => {
                self.start_dma_channel(tile.col, tile.row, *channel, *dir, *bd_id);
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
