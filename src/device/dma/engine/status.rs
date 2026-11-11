//! Channel status and query methods for DMA engine.

use super::*;

impl DmaEngine {
    const SHIM_PARKED_RELEASE_STALL: usize = 3;

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
            let queue_size = self.task_queue_size(channel) as u32;
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
                status = layout.channel_running.set_bit(status);
                status = layout.stalled_lock_acq.set_bit(status);
            }
            ChannelFsm::Paused { .. } => {
                status = layout.channel_running.set_bit(status);
            }
            ChannelFsm::Error => {
                status = layout.error_bd_invalid.set_bit(status);
                // A memtile channel on an invalid BD still reads Channel_Running (silicon,
                // lock-leftovers mem_dmaerr); shim and compute channels are unprobed.
                if self.tile_kind.is_mem() {
                    status = layout.channel_running.set_bit(status);
                }
            }
            _ => {
                // All other active states
                status = layout.channel_running.set_bit(status);
            }
        }

        // Cur_BD while the head is stalled names the last BD the channel's pipeline has loaded
        // behind it (see `pipeline.rs`). With releases parked behind a frozen lock the pipeline
        // is shorter and tile-specific (silicon, lock-leftovers): on a memtile the next BD of the
        // chain, on a shim the end of it; compute channels are unprobed there and keep the
        // executing BD.
        if ch.parked_releases.is_empty() {
            if let Some(bd) = self.pipeline_cur_bd(channel) {
                status = layout.cur_bd.insert(status, bd as u32);
            } else if let ChannelFsm::AcquiringLock { acquired: false, transfer, .. } = &ch.fsm {
                status = layout.cur_bd.insert(status, transfer.bd_index as u32);
            }
        } else if let ChannelFsm::AcquiringLock { acquired: false, transfer, .. } = &ch.fsm {
            let lookahead = if self.tile_kind.is_shim() {
                self.bd_configs.len()
            } else if self.tile_kind.is_mem() {
                1
            } else {
                0
            };
            let mut bd = transfer.bd_index;
            for _ in 0..lookahead {
                match self.bd_configs.get(bd as usize).and_then(|c| c.next_bd) {
                    Some(next) => bd = next,
                    None => break,
                }
            }
            status = layout.cur_bd.insert(status, bd as u32);
        }

        // An MM2S whose BDs have all FINISHED but whose words are still undelivered keeps
        // running, back-pressured: a shim's read-ahead window (silicon, lock-leftovers
        // mem_s2mm_rel), a compute tile's or memtile's egress staging (silicon, pipeline-fill
        // chain_mem_L4: Running + Stalled_Stream_Backpressure with every BD finished).
        if matches!(self.channel_type(channel), ChannelType::MM2S)
            && (!ch.read_ahead.is_empty()
                || !ch.egress_staging.is_empty()
                || self.stream_out_len_for_channel(channel) > 0)
        {
            status = layout.channel_running.set_bit(status);
            if !self.can_push_stream_out_for_channel(channel) {
                status = layout.stalled_stream_backpressure.set_bit(status);
            }
        }

        // Shim: three releases parked behind a flagged lock leave the channel running with
        // Stalled_Lock_Rel; up to two read as an idle channel (silicon, lock-leftovers
        // shim_*_rel_c1..c7). Memtile channels stop at the next acquire instead, which the
        // AcquiringLock arm above already reports.
        if self.tile_kind.is_shim() && ch.parked_releases.len() >= Self::SHIM_PARKED_RELEASE_STALL {
            status = layout.channel_running.set_bit(status);
            status = layout.stalled_lock_rel.set_bit(status);
        }

        // Bit 4 of the status register reports a stream-side stall, but
        // AM025 names it differently per direction: S2MM calls it
        // Stalled_Stream_Starvation (upstream has no data), MM2S calls
        // it Stalled_Stream_Backpressure (downstream has no space).
        // Both live at bit 4; we still pick the correct field so a
        // future split would surface here rather than silently misreport.
        if let Some(transfer) = ch.fsm.transfer() {
            match transfer.direction {
                TransferDirection::S2MM => {
                    if !self.has_stream_in_for_channel(channel) {
                        status = layout.stalled_stream_starvation.set_bit(status);
                    }
                }
                TransferDirection::MM2S => {
                    // Per-channel: only this MM2S channel's slave-port FIFO
                    // counts; sibling channels with their own stalls do not
                    // assert backpressure on this channel's status bit.
                    if !self.can_push_stream_out_for_channel(channel) {
                        status = layout.stalled_stream_backpressure.set_bit(status);
                    }
                }
            }
        }

        // Status[1:0]: the pipeline's state, not "running" (silicon reads 0 in most stalls).
        status = layout.status.insert(status, self.status_field(channel));

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
                self.col,
                self.row,
                ch_idx,
                enable_token_issue,
                controller_id,
                fot_mode
            );
        }
    }

    /// The channel's Ctrl register was written: its Controller_ID (and, for
    /// S2MM, FoT mode) apply from now on, including to the token of a task
    /// already queued or running (AM025: the token is sent to the
    /// controller the channel names when the task completes).
    pub fn set_channel_controller(&mut self, ch_idx: u8, controller_id: u8, fot_mode: u8) {
        if let Some(ch) = self.channels.get_mut(ch_idx as usize) {
            ch.task_config.controller_id = controller_id;
            ch.task_config.fot_mode = fot_mode;
        }
    }

    /// The Controller_ID a task-complete token from this channel carries now.
    pub fn channel_controller_id(&self, ch_idx: u8) -> Option<u8> {
        self.channels.get(ch_idx as usize).map(|ch| ch.task_config.controller_id)
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
                self.col,
                self.row,
                ch_idx,
                effective_compress,
                effective_decompress,
                effective_ooo,
                compression_enable,
                decompression_enable,
                out_of_order_enable,
            );
        }
    }
}
