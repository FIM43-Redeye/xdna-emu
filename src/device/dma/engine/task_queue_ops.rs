//! Task queue operations for DMA engine.

use super::pipeline::HeadStall;
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

        // The channel holds `staged_task_slots` tasks of its own before the 4-deep queue
        // starts to fill (silicon, shim-queue probes); past both, a push finds no room.
        if !self.task_queue_has_space(channel) {
            let ch = &mut self.channels[ch_idx];
            ch.task_queue.flag_overflow();
            log::trace!(
                "DMA tile({},{}) ch{} task queue full (BD {} rejected, waiting={})",
                self.col,
                self.row,
                channel,
                start_bd,
                ch.task_queue.len()
            );
            return false;
        }

        let ch = &mut self.channels[ch_idx];
        let entry = TaskQueueEntry::new(start_bd, repeat_count, enable_token_issue);
        if ch.task_queue.push(entry).is_err() {
            return false;
        }

        // Count this dispatch for the controller's occupancy-dependent
        // dispatch gate (Phase 2d.2 Part 2).  Monotonic per session; reset
        // only on channel reset.  Saturates so a pathologically long chain
        // can't wrap (the gate caps at its plateau well before this matters).
        ch.controller_dispatch_index = ch.controller_dispatch_index.saturating_add(1);

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

        // Out-of-order S2MM: Start_BD_ID is ignored; each packet's header
        // names its BD.
        if self.is_out_of_order_enabled(channel) {
            self.start_ooo_task(channel, task.repeat_count);
            return;
        }

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

    /// Get the controller dispatch index for a channel -- the monotonic
    /// count of task dispatches issued since the last channel reset.  Feeds
    /// the controller's occupancy-dependent dispatch gate (Phase 2d.2
    /// Part 2).  Returns 0 on architectures without a task queue.
    pub fn controller_dispatch_index(&self, channel: u8) -> u32 {
        if !self.dma_model.supports_task_queue() {
            return 0;
        }
        self.channels
            .get(channel as usize)
            .map(|ch| ch.controller_dispatch_index)
            .unwrap_or(0)
    }

    /// Tasks waiting behind the one the channel is executing (staged plus queued).
    pub fn pending_tasks(&self, channel: u8) -> usize {
        if !self.dma_model.supports_task_queue() {
            return 0;
        }
        self.channels.get(channel as usize).map(|ch| ch.task_queue.len()).unwrap_or(0)
    }

    /// `Task_Queue_Size` as the status register reports it: the waiting tasks the channel's
    /// staging has no room for.
    pub fn task_queue_size(&self, channel: u8) -> usize {
        self.pending_tasks(channel).saturating_sub(self.staged_task_slots(channel))
    }

    /// Tasks a channel holds past the one executing before `Task_Queue_Size` starts counting.
    ///
    /// Silicon (Phoenix, shim-queue probes, every channel kind: one task parked on a lock
    /// acquire or a starved stream, N more pushed, the status read after each):
    /// - head parked on a lock acquire: 2 staged tasks on every tile kind (the first reads
    ///   Cur_BD = its BD, the next two follow in Cur_BD; the queue counts from the fourth push);
    /// - head parked on its stream (S2MM starved): 4 on a compute tile or memtile, 6 on a shim.
    /// The queue behind the staging is 4 deep (aie-rt StartQSizeMax) in every case: a push that
    /// finds `staged + 4` tasks waiting blocks its writer.
    pub fn staged_task_slots(&self, channel: u8) -> usize {
        match self.head_stall(channel) {
            Some((stall, _)) => self.pipeline_geometry(stall).depth - 1,
            // between BDs or past an error: the data-stage pipeline
            None if self
                .channels
                .get(channel as usize)
                .is_some_and(|c| !matches!(c.fsm, ChannelFsm::Idle)) =>
            {
                self.pipeline_geometry(HeadStall::Data).depth - 1
            }
            // nothing executing: the channel takes the next task as its head
            None => 0,
        }
    }

    /// Whether a Start_Queue write would be accepted now (otherwise the writer blocks).
    pub fn task_queue_has_space(&self, channel: u8) -> bool {
        self.pending_tasks(channel) < self.staged_task_slots(channel) + MAX_TASK_QUEUE_DEPTH
    }

    /// Pending task-queue BD ids in hardware FIFO order.
    pub fn queued_bd_ids(&self, channel: ChannelId) -> Vec<u8> {
        if !self.dma_model.supports_task_queue() {
            return Vec::new();
        }
        self.channels
            .get(channel as usize)
            .map(|ch| ch.task_queue.iter().map(|task| task.start_bd).collect())
            .unwrap_or_default()
    }

    /// Whether each pending task issues a task-complete token, in FIFO order.
    pub fn queued_token_issue(&self, channel: ChannelId) -> Vec<bool> {
        self.channels
            .get(channel as usize)
            .map(|ch| ch.task_queue.iter().map(|task| task.enable_token_issue).collect())
            .unwrap_or_default()
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

    /// Inject a task-completion token for `channel` (absolute index) directly.
    ///
    /// The normal source is `maybe_emit_task_token` on a BD completing with
    /// Enable_Token_Issue set. This entry point lets an external completion
    /// source drive a token in -- used by the NPU sync path's tests today, and
    /// the seam where firmware/array wiring reports completion later.
    pub fn issue_task_token(&mut self, channel: u8, controller_id: u8) {
        self.task_tokens.issue(channel, controller_id);
    }

    /// Pop a task complete token from the output buffer.
    ///
    /// Returns None if no tokens are pending.
    pub fn pop_task_token(&mut self) -> Option<Token> {
        self.task_tokens.consume()
    }

    /// Inspect the oldest task-completion token without consuming it.
    pub fn peek_task_token(&self) -> Option<Token> {
        self.task_tokens.peek()
    }

    /// Pop the oldest pending token issued by `channel` (the absolute channel
    /// index the token was issued with). Used by the sync/`WAIT_TCTS` path to
    /// consume exactly the completing channel's token.
    pub fn pop_task_token_for_channel(&mut self, channel: u8) -> Option<Token> {
        self.task_tokens.consume_for_channel(channel)
    }

    /// Check if any task complete tokens are pending.
    pub fn has_task_token(&self) -> bool {
        self.task_tokens.has_pending()
    }

    /// Check if any pending token was issued by `channel` (absolute index).
    pub fn has_task_token_for_channel(&self, channel: u8) -> bool {
        self.task_tokens.has_pending_for_channel(channel)
    }

    /// Get the number of pending task complete tokens.
    pub fn task_token_count(&self) -> usize {
        self.task_tokens.pending_count()
    }
}
