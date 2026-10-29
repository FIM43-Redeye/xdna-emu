//! Out-of-order S2MM (Ctrl Enable_Out_of_Order).
//!
//! An out-of-order channel is a FIFO of packets, each placed into the BD
//! whose id its header carries (bits 10-5: the sending BD's
//! Out_Of_Order_BD_ID). Per aie-rt and mlir-aie's lowering
//! (`AIERTControl::pushToBdQueueAndEnable`): Start_BD_ID is ignored (aie-rt
//! writes 0), receive BDs do not chain, and a task's Repeat_Count is its
//! packet count - 1. The header is consumed: NPU1 silicon places only the
//! payload (mlir-aie test/npu-xrt/dma_s2mm_ooo_shim_recv). The receive BD's
//! lock is acquired per packet, so a packet whose BD lock is unavailable
//! stalls every packet behind it.
//!
//! Timing is the in-order BD path's (setup, lock, transfer); the header word
//! itself costs the cycle it is popped. Not yet calibrated against silicon.

use super::*;
use crate::device::stream_switch::PacketHeader;

impl DmaEngine {
    /// Start a queued task on an out-of-order S2MM channel: receive
    /// `repeat_count + 1` packets, each into the BD its header names.
    pub(super) fn start_ooo_task(&mut self, channel: u8, repeat_count: u8) {
        let ch = &mut self.channels[channel as usize];
        ch.error_event_reported = false;
        ch.current_bd = None;
        ch.chain_start_bd = None;
        ch.repeat_count = 0;
        ch.ooo_packets_remaining = repeat_count as u32 + 1;
        ch.fsm = ChannelFsm::OooAwaitHeader;
        ch.accept_bd = None;
        ch.accept_words_remaining = 0;
        ch.bd_switch_accept_block = 0;
        ch.accept_awaiting_drain = false;
        ch.accept_ooo_header = true;

        log::info!(
            "DMA tile({},{}) ch{} out-of-order task start: {} packet(s) cy={}",
            self.col,
            self.row,
            channel,
            repeat_count as u32 + 1,
            self.current_cycle,
        );
        if !self.channels[channel as usize].prefetch_start_emitted {
            self.trace(EventType::DmaStartTask { channel });
        }
        self.channels[channel as usize].prefetch_start_emitted = false;
    }

    /// The BD an out-of-order header selects, if this channel can use it.
    fn ooo_header_bd(&self, channel: u8, header: u32) -> Option<u8> {
        let bd = PacketHeader::decode(header).0.ooo_bd_id;
        let usable = self.bd_configs.get(bd as usize).is_some_and(|c| c.valid)
            && self.check_memtile_bd_channel_validity(bd, channel);
        usable.then_some(bd)
    }

    /// `OooAwaitHeader`: pop the next packet header and start its BD.
    pub(super) fn step_ooo_await_header(&mut self, ch_idx: usize) -> ChannelFsm {
        let channel = ch_idx as u8;
        let Some(header) = self.pop_stream_in_for_channel(channel) else {
            return ChannelFsm::OooAwaitHeader;
        };
        let Some(bd) = self.ooo_header_bd(channel, header.data) else {
            log::warn!(
                "DMA tile({},{}) ch{} out-of-order header 0x{:08X} names BD {}, which this channel cannot use",
                self.col,
                self.row,
                channel,
                header.data,
                PacketHeader::decode(header.data).0.ooo_bd_id,
            );
            self.set_error_bd_unavailable(channel);
            return ChannelFsm::Error;
        };

        let mut transfer = match self.create_transfer_from_bd(bd, channel) {
            Ok(t) => t,
            Err(e) => {
                log::warn!(
                    "DMA tile({},{}) ch{} out-of-order BD {}: {:?}",
                    self.col,
                    self.row,
                    channel,
                    bd,
                    e
                );
                self.set_error_bd_unavailable(channel);
                return ChannelFsm::Error;
            }
        };
        // Receive BDs do not chain in out-of-order mode.
        transfer.next_bd = None;
        self.channels[ch_idx].current_bd = Some(bd);
        log::info!(
            "DMA tile({},{}) ch{} out-of-order packet -> BD{} cy={}",
            self.col,
            self.row,
            channel,
            bd,
            self.current_cycle
        );

        match transfer.acquire_lock {
            Some(lock_id) => ChannelFsm::AcquiringLock {
                lock_id,
                cycles_remaining: self.timing_config.lock_acquire_cycles as u16,
                acquired: false,
                transfer: Box::new(transfer),
            },
            None => ChannelFsm::BdSetup {
                cycles_remaining: self.timing_config.bd_setup_cycles as u16,
                transfer: Box::new(transfer),
            },
        }
    }

    /// After a packet's BD completes on an out-of-order channel: wait for the
    /// next header while the task has packets left. `None` when the task is
    /// done (or the channel is not out-of-order), so the caller finishes it.
    pub(super) fn ooo_next_packet(&mut self, ch_idx: usize) -> Option<ChannelFsm> {
        let ch = &mut self.channels[ch_idx];
        if ch.ooo_packets_remaining == 0 {
            return None;
        }
        ch.ooo_packets_remaining -= 1;
        (ch.ooo_packets_remaining > 0).then_some(ChannelFsm::OooAwaitHeader)
    }

    /// Accept side: the word just accepted is a packet header; aim the accept
    /// cursor at the BD it names (so the per-BD TREADY gating applies), or
    /// leave it idle when the header names no usable BD.
    pub(super) fn accept_ooo_header(&mut self, ch: usize, header: u32) {
        let words = self.ooo_header_bd(ch as u8, header).and_then(|bd| {
            self.bd_configs
                .get(bd as usize)
                .map(|c| (bd, c.length.div_ceil(super::stream_io::BYTES_PER_STREAM_WORD)))
        });
        let c = &mut self.channels[ch];
        c.accept_ooo_header = false;
        c.accept_bd = words.map(|(bd, _)| bd);
        c.accept_words_remaining = words.map_or(0, |(_, n)| n);
    }
}
