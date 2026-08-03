//! Control packet handling for the tile array.

use super::*;

fn phoenix_tct_actor(kind: TileKind, channel: u8, s2mm_channels: usize) -> Option<u8> {
    use xdna_archspec::aie2::tct;

    let channel = usize::from(channel);
    let (actors, index) = if channel < s2mm_channels {
        (
            match kind {
                TileKind::Compute => tct::COMPUTE_S2MM_ACTORS,
                TileKind::Mem => tct::MEM_S2MM_ACTORS,
                TileKind::ShimNoc | TileKind::ShimPl => tct::SHIM_S2MM_ACTORS,
            },
            channel,
        )
    } else {
        (
            match kind {
                TileKind::Compute => tct::COMPUTE_MM2S_ACTORS,
                TileKind::Mem => tct::MEM_MM2S_ACTORS,
                TileKind::ShimNoc | TileKind::ShimPl => tct::SHIM_MM2S_ACTORS,
            },
            channel - s2mm_channels,
        )
    };
    actors.get(index).copied()
}

fn phoenix_tct_transport_word(col: u8, row: u8, actor: u8, controller_id: u8) -> u32 {
    use xdna_archspec::aie2::packet;

    // Packet type 6 and actor shift 8 are fixed by the observed Phoenix TCT
    // record; tile shifts and parity position come from the generated packet
    // format. Actor bit 4 intentionally overlaps the type field for actors 16+.
    let mut word = u32::from(col) << packet::SRC_COL_SHIFT
        | u32::from(row) << packet::SRC_ROW_SHIFT
        | 6u32 << packet::TYPE_SHIFT
        | u32::from(actor) << 8
        | u32::from(controller_id);
    word |= u32::from(word.count_ones() % 2 == 0) << packet::PARITY_SHIFT;
    word
}

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

    /// True if a control-register-write packet is still being delivered to any
    /// tile: a word waiting at a TileCtrl master port, a packet mid-reassembly,
    /// or a reassembled action not yet dispatched.
    ///
    /// This is the necessary precondition for the (deleted) inter-instruction
    /// `flush_ctrl_packets` to have had any effect. It is deliberately
    /// CONTROL-specific: it keys on the TileCtrl delivery path and the
    /// per-tile control-packet reassemblers, NOT the broad packet-switched
    /// signal. Trace packets are also packet-switched (trace units -> shim DDR)
    /// but never touch the control path, so they must not trip the detector --
    /// an early broad signal fired ~41x per traced tenant-4 run purely on trace
    /// traffic. The control-packet ordering hazard detector screens on this: if
    /// it is ever true at an NPU instruction boundary, removing the flush may
    /// have changed control-register delivery ordering, and we want to know.
    pub fn has_pending_control_packet(&self) -> bool {
        if !self.pending_ctrl_actions.is_empty() {
            return true;
        }
        if self.ctrl_reassemblers.iter().any(|r| r.is_mid_packet()) {
            return true;
        }
        self.tiles.iter().any(|t| {
            t.stream_switch.masters.iter().any(|p| {
                matches!(p.port_type, crate::device::stream_switch::PortType::TileCtrl) && p.has_data()
            })
        })
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

    /// Move at most one issue-ordered DMA token per tile into its TileControl
    /// packet source. The configured stream switch remains responsible for
    /// whether that packet can reach shim South channel 0.
    pub(crate) fn queue_phoenix_tct_packets(&mut self) -> usize {
        let mut queued = 0;
        for i in 0..self.tiles.len() {
            if !self.tile_present[i] {
                continue;
            }
            let Some(token) = self.dma_engines[i].peek_task_token() else {
                continue;
            };
            let kind = self.tiles[i].tile_kind;
            let Some(actor) =
                phoenix_tct_actor(kind, token.channel_id, self.dma_engines[i].s2mm_channel_count())
            else {
                continue;
            };

            let consumed = self.dma_engines[i].pop_task_token().expect("peeked TCT must remain pending");
            debug_assert_eq!(consumed, token);
            let col = self.tiles[i].col;
            let row = self.tiles[i].row;
            let word = phoenix_tct_transport_word(col, row, actor, token.controller_id);
            self.tiles[i].pending_ctrl_response.push_back((word, true));
            self.phoenix_tct_packets_in_flight[col as usize] += 1;
            queued += 1;
        }
        queued
    }

    /// Drain TCT packets which traversed the configured fabric to shim South
    /// channel 0. Returned words retain transport parity; the firmware landing
    /// boundary owns conversion to its parity-free wait key.
    pub(crate) fn drain_phoenix_tct_egress(&mut self) -> Vec<(u8, u32)> {
        use xdna_archspec::aie2::stream_switch::shim;

        let mut landed = Vec::new();
        for col in 0..self.cols {
            if self.phoenix_tct_packets_in_flight[col as usize] == 0 || !self.arch.is_valid_tile(col, 0) {
                continue;
            }
            let tile_idx = self.tile_index(col, 0);
            if !self.tiles[tile_idx].is_shim() {
                continue;
            }
            let port = &mut self.tiles[tile_idx].stream_switch.masters[shim::SOUTH_MASTER_START as usize];
            while self.phoenix_tct_packets_in_flight[col as usize] > 0 {
                let Some((word, tlast)) = port.pop_with_tlast() else {
                    break;
                };
                self.phoenix_tct_packets_in_flight[col as usize] -= 1;
                if tlast {
                    landed.push((col, word));
                } else {
                    self.fatal_errors.push(format!(
                        "Phoenix TCT egress ({col},0) South:0 received non-TLAST word 0x{word:08x}"
                    ));
                }
            }
        }
        landed
    }

    pub(crate) fn phoenix_tct_packets_in_flight(&self) -> usize {
        self.phoenix_tct_packets_in_flight.iter().sum()
    }

    /// Drain pending control packet read responses into TileCtrl slave ports.
    ///
    /// Called during each routing cycle. Pushes as many queued response words
    /// as the TileCtrl slave FIFO can accept, respecting backpressure.
    /// Returns the number of words injected.
    pub fn drain_ctrl_responses(&mut self) -> usize {
        let mut words_injected = 0;

        for i in 0..self.tiles.len() {
            if !self.tile_present[i] {
                continue;
            }
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
