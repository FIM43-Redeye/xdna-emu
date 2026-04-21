//! Channel status and query methods for DMA engine.

use super::*;

impl DmaEngine {
    /// Get the current status register value for a channel.
    ///
    /// The status register format (AM025):
    /// - Bits 27:24: Cur_BD (current BD being processed)
    /// - Bits 22:20: Task_Queue_Size (current number of tasks in queue)
    /// - Bit 19: Channel_Running
    /// - Bit 18: Task_Queue_Overflow (sticky, write-to-clear)
    /// - Bits 1:0: Status state (00=IDLE, 01=STARTING, 10=RUNNING)
    ///
    /// Additional bits for stall conditions are set based on channel state.
    pub fn get_channel_status(&self, channel: u8) -> u32 {
        let layout = self.status_layout();

        let ch_idx = channel as usize;
        if ch_idx >= self.channels.len() {
            return 0;
        }

        let ch = &self.channels[ch_idx];
        let mut status: u32 = 0;

        // Cur_BD
        if let Some(bd_idx) = ch.current_bd {
            status = layout.cur_bd.insert(status, bd_idx as u32);
        }

        // Task_Queue_Size + Task_Queue_Overflow (AIE2+ only; AIE1
        // status register has StartQSize/Stalled instead).
        if self.dma_model.supports_task_queue() {
            let queue_size = ch.task_queue.len() as u32;
            status = layout.task_queue_size.insert(status, queue_size);

            if ch.task_queue.has_overflow() {
                status = layout.task_queue_overflow.set_bit(status);
            }
        }

        // Error_BD_Unavailable
        if ch.error_bd_unavailable {
            status = layout.error_bd_unavailable.set_bit(status);
        }

        // Derive external state from FSM
        match &ch.fsm {
            ChannelFsm::Idle => {}
            ChannelFsm::AcquiringLock { acquired: false, .. } => {
                // Stalled on lock acquire
                status = layout.status.insert(status, 0b10);
                status = layout.channel_running.set_bit(status);
                status = layout.stalled_lock_acq.set_bit(status);
            }
            ChannelFsm::Paused { .. } => {
                status = layout.channel_running.set_bit(status);
            }
            ChannelFsm::Error => {
                status = layout.error_bd_invalid.set_bit(status);
            }
            _ => {
                // All other active states
                status = layout.status.insert(status, 0b10);
                status = layout.channel_running.set_bit(status);
            }
        }

        // Check for stream stall (S2MM waiting for data in Transferring state)
        if let Some(transfer) = ch.fsm.transfer() {
            if matches!(transfer.direction, TransferDirection::S2MM)
                && !self.has_stream_in_for_channel(channel)
            {
                status = layout.stalled_stream.set_bit(status);
            }
        }

        status
    }

    /// Get the FoT mode for a channel (S2MM only).
    pub fn get_channel_fot_mode(&self, ch_idx: u8) -> u8 {
        self.channels
            .get(ch_idx as usize)
            .map(|ch| ch.task_config.fot_mode)
            .unwrap_or(0)
    }

    /// Get whether compression is enabled for a channel (MM2S).
    pub fn is_compression_enabled(&self, channel: u8) -> bool {
        self.channels
            .get(channel as usize)
            .map(|ch| ch.task_config.compression_enable)
            .unwrap_or(false)
    }

    /// Get whether decompression is enabled for a channel (S2MM).
    pub fn is_decompression_enabled(&self, channel: u8) -> bool {
        self.channels
            .get(channel as usize)
            .map(|ch| ch.task_config.decompression_enable)
            .unwrap_or(false)
    }

    /// Get whether out-of-order mode is enabled for a channel (S2MM).
    pub fn is_out_of_order_enabled(&self, channel: u8) -> bool {
        if !self.dma_model.supports_ooo_mode() {
            return false;
        }
        self.channels
            .get(channel as usize)
            .map(|ch| ch.task_config.out_of_order_enable)
            .unwrap_or(false)
    }

    /// Set channel task configuration (called when channel control is written).
    ///
    /// This sets the persistent channel configuration (controller_id, fot_mode, etc.)
    /// that applies to all tasks on this channel.
    pub fn set_channel_task_config(
        &mut self,
        ch_idx: u8,
        enable_token_issue: bool,
        controller_id: u8,
        fot_mode: u8,
    ) {
        if let Some(ch) = self.channels.get_mut(ch_idx as usize) {
            ch.task_config.enable_token_issue = enable_token_issue;
            ch.task_config.controller_id = controller_id;
            ch.task_config.fot_mode = fot_mode;

            log::trace!(
                "DMA tile({},{}) ch{} set task config: token_issue={} controller_id={} fot_mode={}",
                self.col, self.row, ch_idx, enable_token_issue, controller_id, fot_mode
            );
        }
    }

    /// Set channel compression/decompression and out-of-order configuration.
    pub fn set_channel_compression_config(
        &mut self,
        ch_idx: u8,
        compression_enable: bool,
        decompression_enable: bool,
        out_of_order_enable: bool,
    ) {
        // On archs without compression (AIE1), silently drop the
        // compression bits -- setting them would be a no-op during
        // the transfer path since supports_compression() gates the
        // compression callers in stepping.rs.
        let effective_compress = compression_enable && self.dma_model.supports_compression();
        let effective_decompress = decompression_enable && self.dma_model.supports_compression();
        // Same for OOO.
        let effective_ooo = out_of_order_enable && self.dma_model.supports_ooo_mode();

        if let Some(ch) = self.channels.get_mut(ch_idx as usize) {
            ch.task_config.compression_enable = effective_compress;
            ch.task_config.decompression_enable = effective_decompress;
            ch.task_config.out_of_order_enable = effective_ooo;

            log::trace!(
                "DMA tile({},{}) ch{} set compression config: compress={} decompress={} ooo={} \
                 (requested compress={} decompress={} ooo={})",
                self.col, self.row, ch_idx,
                effective_compress, effective_decompress, effective_ooo,
                compression_enable, decompression_enable, out_of_order_enable,
            );
        }
    }
}
