//! CDO application logic.

use anyhow::Result;

use super::*;

impl DeviceState {
    /// Apply a CDO to configure the device.
    ///
    /// Processes all commands in the CDO and updates the tile array accordingly.
    pub fn apply_cdo(&mut self, cdo: &Cdo) -> Result<()> {
        self.stats = CdoStats::default();

        for cmd in cdo.commands() {
            self.stats.commands += 1;
            self.apply_command(&cmd)?;
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

    /// Apply a single CDO command.
    fn apply_command(&mut self, cmd: &CdoRaw) -> Result<()> {
        match cmd {
            CdoRaw::Nop { .. } => {
                self.stats.nops += 1;
            }

            CdoRaw::Write { address, value } => {
                self.stats.writes += 1;
                let tile_addr = TileAddress::decode(*address);
                log::trace!("CDO Write: addr=0x{:08X} -> tile({},{}) offset=0x{:05X} value=0x{:08X}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset, value);
                self.write_register(*address, *value)?;
            }

            CdoRaw::MaskWrite { address, mask, value } => {
                self.stats.mask_writes += 1;
                let tile_addr = TileAddress::decode(*address);
                log::trace!("CDO MaskWrite: addr=0x{:08X} -> tile({},{}) offset=0x{:05X} mask=0x{:08X} value=0x{:08X}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset, mask, value);
                self.mask_write_register(*address, *mask, *value)?;
            }

            // Write64/MaskWrite64 use 64-bit addresses but AIE tiles are 32-bit addressed.
            // The high 32 bits are always 0 for AIE, so we use the low 32 bits.
            CdoRaw::Write64 { address, value } => {
                self.stats.writes += 1;
                let addr32 = *address as u32;
                let tile_addr = TileAddress::decode(addr32);
                log::trace!("CDO Write64: addr=0x{:016X} -> tile({},{}) offset=0x{:05X} value=0x{:08X}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset, value);
                self.write_register(addr32, *value)?;
            }

            CdoRaw::MaskWrite64 { address, mask, value } => {
                self.stats.mask_writes += 1;
                let addr32 = *address as u32;
                let tile_addr = TileAddress::decode(addr32);
                log::trace!("CDO MaskWrite64: addr=0x{:016X} -> tile({},{}) offset=0x{:05X}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset);
                self.mask_write_register(addr32, *mask, *value)?;
            }

            CdoRaw::DmaWrite { address, data } => {
                self.stats.dma_writes += 1;
                let tile_addr = TileAddress::decode(*address);
                let subsystem = subsystem_from_offset(tile_addr.offset, tile_kind_from_row(tile_addr.row));
                log::debug!("CDO DmaWrite: addr=0x{:08X} -> tile({},{}) offset=0x{:05X} subsystem={:?} len={}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset, subsystem, data.len());
                self.dma_write(*address, data)?;
            }

            // Synchronization/timing commands - no-ops in emulation.
            // MaskPoll waits for a register to match a value on real hardware;
            // in the emulator, configuration writes take effect immediately.
            CdoRaw::MaskPoll { .. } | CdoRaw::MaskPoll64 { .. } => {
                log::trace!("CDO MaskPoll: skipped (emulator writes are synchronous)");
            }

            // Delay inserts a wait on real hardware; no-op in emulation.
            CdoRaw::Delay { .. } => {
                log::trace!("CDO Delay: skipped (emulator has no real-time clock)");
            }

            // Structural markers - no functional effect.
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
}
