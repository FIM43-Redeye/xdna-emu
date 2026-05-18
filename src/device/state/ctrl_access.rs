//! Control-packet register-access decode gate.
//!
//! A control-packet-handler-initiated register access whose offset
//! classifies as `SubsystemKind::Unknown` for the tile kind is the
//! faithful AXI SLVERR case: the tile exists (the packet is routed to
//! a real `(col,row)`) but the offset maps to no register (AM025
//! `aie_aximm_config.txt:9`, bit 2 `SLVERR_Block`, "unmapped
//! registers"). It is distinct from DECERR (no tile at the address),
//! which never arises -- control packets route to real tiles. This is
//! the coordinator-origin counterpart to the reassembler-origin
//! `TileArray::latch_pkt_error`; both converge on
//! `pkt_handler_status |= PktHandlerError::*.bit()` (one bit map, two
//! origins).

use super::*;
use crate::device::control_packets::PktHandlerError;

impl DeviceState {
    /// True iff `offset` decodes to a real subsystem for the tile kind
    /// at `row`. A control-packet access to an offset that classifies
    /// as `SubsystemKind::Unknown` is an AXI SLVERR (the tile exists,
    /// the register does not). `col` is irrelevant to address decode.
    pub fn ctrl_pkt_offset_decodes(&self, row: u8, offset: u32) -> bool {
        subsystem_from_offset(offset, tile_kind_from_row(row)) != SubsystemKind::Unknown
    }

    /// A read of `count` consecutive registers from `offset` decodes
    /// iff every beat decodes. Any undecoded beat is a SLVERR and the
    /// whole response is suppressed.
    pub fn ctrl_pkt_read_range_decodes(&self, row: u8, offset: u32, count: u8) -> bool {
        (0..count.max(1) as u32).all(|i| self.ctrl_pkt_offset_decodes(row, offset + i * 4))
    }

    /// Latch `SLVERR_On_Access` on the tile at `(col,row)`. The
    /// coordinator-origin equivalent of `TileArray::latch_pkt_error`.
    pub fn latch_ctrl_slverr(&mut self, col: u8, row: u8) {
        if let Some(tile) = self.array.get_mut(col, row) {
            tile.pkt_handler_status |= PktHandlerError::Slverr.bit();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::device::DeviceState;

    #[test]
    fn unknown_offset_does_not_decode() {
        let d = DeviceState::new_npu1();
        // 0x1F200 is verified SubsystemKind::Unknown for a compute tile.
        assert!(!d.ctrl_pkt_offset_decodes(2, 0x1F200));
    }

    #[test]
    fn data_memory_offset_decodes() {
        let d = DeviceState::new_npu1();
        // Compute data memory (low SRAM) classifies as DataMemory, not Unknown.
        assert!(d.ctrl_pkt_offset_decodes(2, 0x400));
    }

    #[test]
    fn read_range_fails_if_any_beat_unknown() {
        let d = DeviceState::new_npu1();
        // Start decodable, walk into the 0x1F200 Unknown hole.
        assert!(d.ctrl_pkt_read_range_decodes(2, 0x400, 4));
        assert!(!d.ctrl_pkt_read_range_decodes(2, 0x1F200, 1));
    }

    #[test]
    fn latch_sets_bit_2() {
        let mut d = DeviceState::new_npu1();
        d.latch_ctrl_slverr(0, 2);
        let tile = d.array.get(0, 2).expect("compute tile (0,2)");
        assert_eq!(tile.pkt_handler_status & 0x4, 0x4);
    }
}
