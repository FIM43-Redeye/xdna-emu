//! Control packet handling for the tile array.

use super::*;

impl TileArray {
    /// Drain pending control packet actions produced during stream routing.
    ///
    /// Control packets arrive via the stream switch network at individual tiles.
    /// Rather than writing registers directly (which misses the full module
    /// dispatch in DeviceState), the tile returns actions. The caller drains
    /// these and routes them through `DeviceState::write_tile_register()`.
    pub fn drain_ctrl_packet_actions(&mut self) -> Vec<crate::device::tile::CtrlPacketAction> {
        std::mem::take(&mut self.pending_ctrl_actions)
    }

    /// Handle a control packet OP_READ by reading registers and queuing
    /// a response packet for injection into the tile's TileCtrl slave port.
    ///
    /// The response consists of a stream packet header (with pkt_id =
    /// response_id, packet_type = Data) followed by `count` data words,
    /// with TLAST set on the final word.
    ///
    /// Response words are buffered in `tile.pending_ctrl_response` and
    /// drained into the TileCtrl slave port each routing cycle as FIFO
    /// space permits (same backpressure-aware pattern as trace injection).
    ///
    /// Returns true if the response was successfully queued.
    pub fn handle_read_registers(
        &mut self,
        col: u8,
        row: u8,
        offset: u32,
        count: u8,
        response_id: u8,
    ) -> bool {
        use crate::device::stream_switch::{PacketHeader, PacketType};

        let tile = match self.get_mut(col, row) {
            Some(t) => t,
            None => {
                log::error!("handle_read_registers: tile({},{}) not found", col, row);
                return false;
            }
        };

        // Verify the TileCtrl slave port exists.
        if tile.stream_switch.tile_ctrl_slave_port().is_none() {
            log::error!("handle_read_registers: tile({},{}) has no TileCtrl slave port", col, row,);
            return false;
        }

        // Read the register values (pure reads, no side effects).
        let mut values = Vec::with_capacity(count as usize);
        for i in 0..count as u32 {
            values.push(tile.read_register_pure(offset + i * 4));
        }

        // Build stream packet header: pkt_id = response_id, type = Data,
        // source = this tile's (col, row).
        let header = PacketHeader::new(response_id & 0x1F, col, row).with_type(PacketType::Data);
        let header_word = header.encode();

        // Queue header + data words into pending buffer.
        // TLAST is set on the last data word (or on the header if count=0).
        tile.pending_ctrl_response.push_back((header_word, count == 0));
        for (i, &value) in values.iter().enumerate() {
            let is_last = i == values.len() - 1;
            tile.pending_ctrl_response.push_back((value, is_last));
        }

        log::info!(
            "handle_read_registers: tile({},{}) read {} regs from 0x{:05X}, \
             {} response words queued (resp_id={})",
            col,
            row,
            count,
            offset,
            count as usize + 1,
            response_id,
        );

        true
    }

    /// Drain pending control packet read responses into TileCtrl slave ports.
    ///
    /// Called during each routing cycle. Pushes as many queued response words
    /// as the TileCtrl slave FIFO can accept, respecting backpressure.
    /// Returns the number of words injected.
    pub fn drain_ctrl_responses(&mut self) -> usize {
        let mut words_injected = 0;

        for i in 0..self.tiles.len() {
            if self.tiles[i].pending_ctrl_response.is_empty() {
                continue;
            }

            let slave_idx = match self.tiles[i].stream_switch.tile_ctrl_slave_port() {
                Some(idx) => idx,
                None => continue,
            };

            while !self.tiles[i].pending_ctrl_response.is_empty()
                && self.tiles[i].stream_switch.slaves[slave_idx].can_accept()
            {
                let (word, tlast) = self.tiles[i].pending_ctrl_response.pop_front().unwrap();
                self.tiles[i].stream_switch.slaves[slave_idx].push_with_tlast(word, tlast);
                words_injected += 1;
            }
        }

        words_injected
    }
}
