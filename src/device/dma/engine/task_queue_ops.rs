//! Task queue operations for DMA engine.

use super::*;

impl DmaEngine {
    /// Enqueue a task to the channel's task queue.
    ///
    /// Per AM025, each channel has an 8-deep task queue. Writing to Start_Queue
    /// enqueues a new task. If the queue is full, sets Task_Queue_Overflow.
    ///
    /// Returns true if the task was enqueued, false if queue was full.
    pub fn enqueue_task(
        &mut self,
        channel: u8,
        start_bd: u8,
        repeat_count: u8,
        enable_token_issue: bool,
    ) -> bool {
        if !self.dma_model.supports_task_queue() {
            // AIE1 has no task queue; this call silently fails.  In
            // production, AIE1 code paths never reach here (higher
            // layers check the flag), but guard defensively.
            log::trace!(
                "DMA tile({},{}) ch{} enqueue_task ignored: arch has no task queue",
                self.col,
                self.row,
                channel
            );
            return false;
        }

        let ch_idx = channel as usize;
        if ch_idx >= self.channels.len() {
            return false;
        }

        let ch = &mut self.channels[ch_idx];

        // Push to task queue (handles overflow flag internally)
        let entry = TaskQueueEntry::new(start_bd, repeat_count, enable_token_issue);
        if ch.task_queue.push(entry).is_err() {
            log::trace!(
                "DMA tile({},{}) ch{} task queue full (BD {} rejected, queue_len={})",
                self.col,
                self.row,
                channel,
                start_bd,
                ch.task_queue.len()
            );
            return false;
        }

        log::debug!(
            "DMA tile({},{}) ch{} enqueued task: BD={} repeat={} token={} (queue_size={})",
            self.col,
            self.row,
            channel,
            start_bd,
            repeat_count,
            enable_token_issue,
            ch.task_queue.len()
        );

        // If channel is idle, start processing the queue
        if matches!(ch.fsm, ChannelFsm::Idle) {
            self.start_next_queued_task(channel);
        }

        true
    }

    /// Start the next task from the channel's queue.
    pub(super) fn start_next_queued_task(&mut self, channel: u8) {
        let ch_idx = channel as usize;
        if ch_idx >= self.channels.len() {
            return;
        }

        let task = match self.channels[ch_idx].task_queue.pop() {
            Some(t) => t,
            None => return,
        };

        self.channels[ch_idx].task_config.enable_token_issue = task.enable_token_issue;

        if task.start_bd as usize >= self.bd_configs.len() {
            log::error!(
                "DMA tile({},{}) ch{} queued task has invalid BD {} (max={})",
                self.col,
                self.row,
                channel,
                task.start_bd,
                self.bd_configs.len()
            );
            self.channels[ch_idx].fsm = ChannelFsm::Error;
            return;
        }

        log::debug!(
            "DMA tile({},{}) ch{} starting queued task: BD={} repeat={} (remaining={})",
            self.col,
            self.row,
            channel,
            task.start_bd,
            task.repeat_count,
            self.channels[ch_idx].task_queue.len()
        );

        // BD is read from registers at execution time (not snapshotted at enqueue).
        // Per AM025: Start_Queue only stores BD_ID; the hardware fetches the BD
        // during the STARTING state transition.

        let start_bd = task.start_bd;
        if let Err(e) = self.start_channel_with_repeat(channel, start_bd, task.repeat_count) {
            log::error!(
                "DMA tile({},{}) ch{} failed to start queued task BD {}: {}",
                self.col,
                self.row,
                channel,
                start_bd,
                e
            );
            self.channels[ch_idx].fsm = ChannelFsm::Error;
        } else {
            self.channels[ch_idx].chain_start_bd = Some(start_bd);
        }
    }

    /// Get the current task queue size for a channel.
    pub fn task_queue_size(&self, channel: u8) -> usize {
        if !self.dma_model.supports_task_queue() {
            return 0;
        }
        self.channels.get(channel as usize).map(|ch| ch.task_queue.len()).unwrap_or(0)
    }

    /// Check if the task queue overflow flag is set for a channel.
    pub fn task_queue_overflow(&self, channel: u8) -> bool {
        if !self.dma_model.supports_task_queue() {
            return false;
        }
        self.channels
            .get(channel as usize)
            .map(|ch| ch.task_queue.has_overflow())
            .unwrap_or(false)
    }

    /// Clear the task queue overflow flag (write-to-clear per AM025).
    pub fn clear_task_queue_overflow(&mut self, channel: u8) {
        if !self.dma_model.supports_task_queue() {
            return;
        }
        if let Some(ch) = self.channels.get_mut(channel as usize) {
            ch.task_queue.clear_overflow();
        }
    }

    /// Check if the BD unavailable error flag is set for a channel.
    pub fn error_bd_unavailable(&self, channel: u8) -> bool {
        self.channels
            .get(channel as usize)
            .map(|ch| ch.error_bd_unavailable)
            .unwrap_or(false)
    }

    /// Set the BD unavailable error flag (S2MM OOO mode).
    pub fn set_error_bd_unavailable(&mut self, channel: u8) {
        if let Some(ch) = self.channels.get_mut(channel as usize) {
            ch.error_bd_unavailable = true;
            log::warn!(
                "DMA tile({},{}) S2MM ch{} Error_BD_Unavailable: invalid BD in OOO packet header",
                self.col,
                self.row,
                channel
            );
        }
    }

    /// Clear the BD unavailable error flag (write-to-clear per AM025).
    pub fn clear_error_bd_unavailable(&mut self, channel: u8) {
        if let Some(ch) = self.channels.get_mut(channel as usize) {
            ch.error_bd_unavailable = false;
        }
    }

    /// Pop a task complete token from the output buffer.
    ///
    /// Returns None if no tokens are pending.
    pub fn pop_task_token(&mut self) -> Option<Token> {
        self.task_tokens.consume()
    }

    /// Check if any task complete tokens are pending.
    pub fn has_task_token(&self) -> bool {
        self.task_tokens.has_pending()
    }

    /// Get the number of pending task complete tokens.
    pub fn task_token_count(&self) -> usize {
        self.task_tokens.pending_count()
    }
}
