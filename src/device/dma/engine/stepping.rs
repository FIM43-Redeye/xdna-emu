//! FSM execution: the `step()` method and all helper methods it calls.

use super::*;

impl DmaEngine {
    /// Step the DMA engine by one cycle.
    ///
    /// This processes all active channels, moving data between memory and streams.
    /// Returns the overall result of the step.
    pub fn step(
        &mut self,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        host_memory: &mut HostMemory,
    ) -> DmaResult {
        let mut any_active = false;
        let mut any_waiting = false;

        for ch_idx in 0..self.channels.len() {
            let phase_before = self.channels[ch_idx].fsm.phase_name();

            match &self.channels[ch_idx].fsm {
                ChannelFsm::Idle => {
                    // Check for queued BD from chaining or task queue
                    if let Some(next_bd) = self.channels[ch_idx].queued_bd.take() {
                        let repeat_count = self.channels[ch_idx].repeat_count as u8;
                        log::debug!(
                            "DMA tile({},{}) ch{} starting queued BD {} (repeat={})",
                            self.col,
                            self.row,
                            ch_idx,
                            next_bd,
                            repeat_count
                        );
                        if let Err(e) = self.start_channel_with_repeat(ch_idx as u8, next_bd, repeat_count) {
                            log::warn!(
                                "DMA tile({},{}) ch{} failed to start BD {}: {:?}",
                                self.col,
                                self.row,
                                ch_idx,
                                next_bd,
                                e
                            );
                            self.channels[ch_idx].fsm = ChannelFsm::Error;
                        } else {
                            any_active = true;
                        }
                    }
                }
                ChannelFsm::Paused { .. } | ChannelFsm::Error => {}
                _ => {
                    // Active channel -- run one FSM cycle
                    self.step_channel_fsm(ch_idx, tile, neighbors, host_memory);
                    if matches!(self.channels[ch_idx].fsm, ChannelFsm::AcquiringLock { acquired: false, .. })
                    {
                        any_waiting = true;
                    } else {
                        any_active = true;
                    }
                }
            }

            // Log transitions
            let phase_after = self.channels[ch_idx].fsm.phase_name();
            if phase_before != phase_after {
                log::info!(
                    "DMA({},{}) ch{}: {} -> {} cycle={}",
                    self.col,
                    self.row,
                    ch_idx,
                    phase_before,
                    phase_after,
                    self.current_cycle,
                );
            }
        }

        if any_active {
            DmaResult::InProgress
        } else if any_waiting {
            DmaResult::WaitingForLock(0)
        } else {
            DmaResult::Complete
        }
    }

    /// Step a single channel through one cycle of its unified FSM.
    ///
    /// Each match arm does ONE cycle of work and optionally transitions to
    /// a new state. This replaces the old step_channel() + step_channel_timed() +
    /// complete_transfer() + finish_complete_transfer() chain.
    fn step_channel_fsm(
        &mut self,
        ch_idx: usize,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        host_memory: &mut HostMemory,
    ) {
        // Take the FSM out temporarily so we can match on it while
        // mutating self (for stream buffers, do_transfer, etc.)
        let fsm = std::mem::take(&mut self.channels[ch_idx].fsm);

        let new_fsm = match fsm {
            ChannelFsm::BdSetup { cycles_remaining, mut transfer } => {
                if cycles_remaining <= 1 {
                    // BD setup done. Check if lock acquisition is needed.
                    // Packet header insertion is deferred until AFTER lock
                    // acquisition -- on real hardware, the DMA only emits
                    // the header when it starts transferring data, not before
                    // the lock is acquired. Emitting early would lock the
                    // stream switch arbiter while the DMA waits for locks,
                    // blocking other packets that share the same arbiter.
                    if let Some(lock_id) = transfer.acquire_lock {
                        ChannelFsm::AcquiringLock {
                            lock_id,
                            cycles_remaining: self.timing_config.lock_acquire_cycles as u16,
                            acquired: false,
                            transfer,
                        }
                    } else {
                        // No lock needed -- insert header now, go to MemoryLatency.
                        self.maybe_insert_packet_header_from_transfer(&mut transfer);
                        ChannelFsm::MemoryLatency {
                            cycles_remaining: self.timing_config.memory_latency_cycles as u16,
                            transfer,
                        }
                    }
                } else {
                    ChannelFsm::BdSetup { cycles_remaining: cycles_remaining - 1, transfer }
                }
            }

            ChannelFsm::AcquiringLock { lock_id, cycles_remaining, acquired, mut transfer } => {
                if acquired {
                    // Lock already acquired, counting down latency
                    if cycles_remaining <= 1 {
                        // Lock acquired, latency done -- insert packet header
                        // now that we're about to start the actual transfer.
                        self.maybe_insert_packet_header_from_transfer(&mut transfer);
                        ChannelFsm::MemoryLatency {
                            cycles_remaining: self.timing_config.memory_latency_cycles as u16,
                            transfer,
                        }
                    } else {
                        ChannelFsm::AcquiringLock {
                            lock_id,
                            cycles_remaining: cycles_remaining - 1,
                            acquired: true,
                            transfer,
                        }
                    }
                } else {
                    // Check if arbiter granted the acquire (submitted in pre-step pass)
                    if self.check_acquire_granted(lock_id, tile, neighbors, ch_idx) {
                        ChannelFsm::AcquiringLock { lock_id, cycles_remaining, acquired: true, transfer }
                    } else {
                        self.channels[ch_idx].stats.lock_wait_cycles += 1;
                        self.trace(EventType::DmaStalledLock { channel: ch_idx as u8 });
                        ChannelFsm::AcquiringLock { lock_id, cycles_remaining, acquired: false, transfer }
                    }
                }
            }

            ChannelFsm::MemoryLatency { cycles_remaining, transfer } => {
                if cycles_remaining <= 1 {
                    // For shim tiles accessing host DDR, add NoC+DDR pipeline
                    // latency. This is the extra time for the first word to
                    // traverse the NoC to DDR and back. Once the pipeline
                    // fills, throughput is 1 word/cycle (same as tile memory).
                    let host_lat = self.timing_config.host_memory_latency_cycles;
                    if host_lat > 0 && transfer.involves_host_memory() {
                        ChannelFsm::HostPipelineLatency { cycles_remaining: host_lat, transfer }
                    } else {
                        ChannelFsm::Transferring { transfer }
                    }
                } else {
                    ChannelFsm::MemoryLatency { cycles_remaining: cycles_remaining - 1, transfer }
                }
            }

            ChannelFsm::HostPipelineLatency { cycles_remaining, transfer } => {
                if cycles_remaining <= 1 {
                    ChannelFsm::Transferring { transfer }
                } else {
                    ChannelFsm::HostPipelineLatency { cycles_remaining: cycles_remaining - 1, transfer }
                }
            }

            ChannelFsm::Transferring { mut transfer } => {
                // Move one cycle of data
                let result = self.do_transfer_cycle(ch_idx, &mut transfer, tile, neighbors, host_memory);

                match result {
                    TransferCycleResult::Continue => {
                        // Tick the transfer's cycle counter
                        transfer.tick();
                        // Check if data movement is complete
                        if transfer.remaining_bytes() == 0 {
                            self.begin_completion(ch_idx, transfer)
                        } else {
                            ChannelFsm::Transferring { transfer }
                        }
                    }
                    TransferCycleResult::Stalled => {
                        // S2MM stall: stay in Transferring, don't advance
                        self.trace(EventType::DmaStreamStarvation { channel: ch_idx as u8 });
                        ChannelFsm::Transferring { transfer }
                    }
                    TransferCycleResult::FotFinish => {
                        // Early finish on TLAST (FoT mode)
                        self.begin_completion(ch_idx, transfer)
                    }
                    TransferCycleResult::Error => ChannelFsm::Error,
                }
            }

            ChannelFsm::ReleasingLock { lock_id, release_value, cycles_remaining, completion } => {
                if cycles_remaining <= 1 {
                    // Execute the lock release
                    self.execute_lock_release(lock_id, release_value, tile, neighbors);
                    // Update stats and handle chaining/repeat
                    self.after_transfer_done(ch_idx, completion)
                } else {
                    ChannelFsm::ReleasingLock {
                        lock_id,
                        release_value,
                        cycles_remaining: cycles_remaining - 1,
                        completion,
                    }
                }
            }

            ChannelFsm::BdChaining { cycles_remaining, next_bd } => {
                if cycles_remaining <= 1 {
                    // Load next BD and start transfer
                    let bd_addr = self.bd_configs.get(next_bd as usize).map(|c| c.base_addr).unwrap_or(0);
                    log::info!(
                        "DMA tile({},{}) ch{} BD chain -> BD{} (base_addr=0x{:X})",
                        self.col,
                        self.row,
                        ch_idx,
                        next_bd,
                        bd_addr
                    );
                    match self.create_transfer_from_bd(next_bd, ch_idx as u8) {
                        Ok(transfer) => {
                            self.channels[ch_idx].current_bd = Some(next_bd);
                            ChannelFsm::BdSetup {
                                cycles_remaining: self.timing_config.bd_setup_cycles as u16,
                                transfer: Box::new(transfer),
                            }
                        }
                        Err(e) => {
                            log::warn!(
                                "DMA tile({},{}) ch{} BD chain to {} failed: {:?}",
                                self.col,
                                self.row,
                                ch_idx,
                                next_bd,
                                e
                            );
                            ChannelFsm::Error
                        }
                    }
                } else {
                    ChannelFsm::BdChaining { cycles_remaining: cycles_remaining - 1, next_bd }
                }
            }

            // Idle/Paused/Error handled in step(), not here
            other => other,
        };

        self.channels[ch_idx].fsm = new_fsm;
    }

    /// Create a Transfer from a BD index without starting a channel.
    /// Used by BdChaining to load the next BD in a chain.
    fn create_transfer_from_bd(&self, bd_index: u8, channel: u8) -> Result<Transfer, DmaError> {
        if bd_index as usize >= self.bd_configs.len() {
            return Err(DmaError::InvalidBd(bd_index));
        }
        let bd_config = &self.bd_configs[bd_index as usize];
        let direction = match self.channel_type(channel) {
            ChannelType::S2MM => TransferDirection::S2MM,
            ChannelType::MM2S => TransferDirection::MM2S,
        };
        Transfer::new(bd_config, bd_index, channel, direction, self.col, self.row, self.tile_kind)
    }

    /// Begin the completion sequence after data movement is done.
    ///
    /// If the BD has a release lock, transitions to ReleasingLock.
    /// Otherwise, goes directly to chaining/repeat/idle via after_transfer_done.
    fn begin_completion(&mut self, ch_idx: usize, transfer: Box<Transfer>) -> ChannelFsm {
        let completion = CompletionInfo {
            bd_index: transfer.bd_index,
            next_bd: transfer.next_bd,
            cycles_elapsed: transfer.cycles_elapsed,
            channel: ch_idx as u8,
        };

        if let Some(lock_id) = transfer.release_lock {
            let release_value = transfer.release_value;
            ChannelFsm::ReleasingLock {
                lock_id,
                release_value,
                cycles_remaining: self.timing_config.lock_release_cycles as u16,
                completion,
            }
        } else {
            self.after_transfer_done(ch_idx, completion)
        }
    }

    /// Handle post-transfer completion: stats, chaining, repeat, task queue.
    ///
    /// Returns the next FSM state (BdChaining, Idle, or starts next task).
    fn after_transfer_done(&mut self, ch_idx: usize, completion: CompletionInfo) -> ChannelFsm {
        self.channels[ch_idx].stats.transfers_completed += 1;
        self.channels[ch_idx].stats.cycles_spent += completion.cycles_elapsed;

        // Emit DMA_FINISHED_BD
        self.trace_events
            .push((self.current_cycle, EventType::DmaFinishedBd { channel: ch_idx as u8 }));

        // Check for BD chaining.
        //
        // Hardware follows next_bd unconditionally when Use_Next_BD is set
        // (represented as next_bd = Some(n) in our model). The hardware does
        // NOT detect chain cycles -- a BD chain that loops (BD0->BD1->BD0)
        // runs indefinitely. This is correct for double-buffered DMA patterns.
        //
        // Chain termination only occurs when Use_Next_BD=0 (next_bd = None).
        // The repeat_count then controls how many additional times the chain
        // is re-executed from chain_start_bd.
        if let Some(next_bd) = completion.next_bd {
            log::debug!(
                "DMA tile({},{}) ch{} chaining to BD {} (from BD {})",
                self.col,
                self.row,
                ch_idx,
                next_bd,
                completion.bd_index
            );
            return ChannelFsm::BdChaining {
                cycles_remaining: self.timing_config.bd_chain_cycles as u16,
                next_bd,
            };
        }

        // Chain ended (Use_Next_BD=0). Check repeat_count for re-execution.
        if self.channels[ch_idx].repeat_count > 0 {
            self.channels[ch_idx].repeat_count -= 1;
            if let Some(start_bd) = self.channels[ch_idx].chain_start_bd {
                log::debug!(
                    "DMA tile({},{}) ch{} repeating chain from BD {} ({} remaining)",
                    self.col,
                    self.row,
                    ch_idx,
                    start_bd,
                    self.channels[ch_idx].repeat_count
                );
                return ChannelFsm::BdChaining {
                    cycles_remaining: self.timing_config.bd_chain_cycles as u16,
                    next_bd: start_bd,
                };
            }
        }

        // Task complete (no chaining, no repeats)
        self.trace(EventType::DmaFinishedTask { channel: ch_idx as u8 });
        self.maybe_emit_task_token(ch_idx);

        // Check for more tasks in the queue (AIE2+ only).
        if self.dma_model.supports_task_queue() && !self.channels[ch_idx].task_queue.is_empty() {
            log::debug!(
                "DMA tile({},{}) ch{} task complete, {} tasks remaining in queue",
                self.col,
                self.row,
                ch_idx,
                self.channels[ch_idx].task_queue.len()
            );
            self.start_next_queued_task(ch_idx as u8);
            // start_next_queued_task sets the FSM, return what it set
            return std::mem::take(&mut self.channels[ch_idx].fsm);
        }

        ChannelFsm::Idle
    }

    /// Perform one cycle of data transfer for a channel in the Transferring state.
    ///
    /// Extracts transfer parameters, calls do_transfer, and advances the transfer.
    /// The Transfer is borrowed from the FSM via the caller (not from self.channels).
    fn do_transfer_cycle(
        &mut self,
        ch_idx: usize,
        transfer: &mut Transfer,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        host_memory: &mut HostMemory,
    ) -> TransferCycleResult {
        let words_per_cycle = self.timing_config.words_per_cycle as usize;

        if transfer.has_zero_padding() {
            // Padding-aware path: one word at a time
            let remaining = transfer.remaining_bytes();
            if remaining == 0 {
                return TransferCycleResult::Continue;
            }

            let action = transfer.next_output_action();
            let is_last_word = remaining <= 4;
            let channel = transfer.channel;
            let tlast_suppress = transfer.effective_tlast_suppress();

            match action {
                PadAction::Zero => {
                    let should_assert_tlast = is_last_word && !tlast_suppress;
                    self.stream_out
                        .push_back(StreamData { data: 0, tlast: should_assert_tlast, channel });
                    transfer.advance(4);
                    self.channels[ch_idx].stats.bytes_transferred += 4;
                    TransferCycleResult::Continue
                }
                PadAction::Data(addr) => {
                    let source = transfer.source;
                    let dest = transfer.dest;
                    let result = self.do_transfer(
                        source,
                        dest,
                        addr,
                        4,
                        channel,
                        is_last_word,
                        tlast_suppress,
                        tile,
                        neighbors,
                        host_memory,
                    );

                    if result.stall {
                        return TransferCycleResult::Stalled;
                    }
                    if result.success {
                        transfer.advance(4);
                        self.channels[ch_idx].stats.bytes_transferred += 4;
                        if result.fot_finish {
                            TransferCycleResult::FotFinish
                        } else {
                            TransferCycleResult::Continue
                        }
                    } else {
                        TransferCycleResult::Error
                    }
                }
            }
        } else {
            // Standard path: transfer up to words_per_cycle words this cycle.
            //
            // Each word's address comes from the BD's address generator so
            // multi-dimensional stride/wrap configuration (matrix transpose,
            // nd_memcpy_transforms, etc.) is honored. Reading the chunk
            // linearly from a single base address would produce wrong data
            // for any non-contiguous BD.
            let bytes_remaining = transfer.remaining_bytes() as usize;
            if bytes_remaining == 0 {
                return TransferCycleResult::Continue;
            }
            let words_this_cycle = words_per_cycle.min((bytes_remaining + 3) / 4);

            let source = transfer.source;
            let dest = transfer.dest;
            let channel = transfer.channel;
            let tlast_suppress = transfer.effective_tlast_suppress();
            let mut fot_finished = false;

            for w in 0..words_this_cycle {
                let addr = transfer.current_address();
                let is_last = transfer.remaining_bytes() <= 4;

                let result = self.do_transfer(
                    source,
                    dest,
                    addr,
                    4,
                    channel,
                    is_last,
                    tlast_suppress,
                    tile,
                    neighbors,
                    host_memory,
                );

                if result.stall {
                    // Mid-chunk stall: if any word made it through this
                    // cycle, report Continue so progress is visible; if
                    // we stalled on the very first word, report Stalled.
                    return if w == 0 {
                        TransferCycleResult::Stalled
                    } else {
                        TransferCycleResult::Continue
                    };
                }
                if !result.success {
                    return TransferCycleResult::Error;
                }

                transfer.advance(4);
                self.channels[ch_idx].stats.bytes_transferred += 4;

                if result.fot_finish {
                    fot_finished = true;
                    break;
                }
            }

            if fot_finished {
                TransferCycleResult::FotFinish
            } else {
                TransferCycleResult::Continue
            }
        }
    }

    /// Check if a lock acquire was granted by the arbiter.
    ///
    /// The request was submitted in `submit_lock_requests()` before stepping.
    /// This checks whether the arbiter granted it during resolution.
    fn check_acquire_granted(
        &mut self,
        lock_id: u8,
        tile: &Tile,
        neighbors: &NeighborTiles<'_>,
        ch_idx: usize,
    ) -> bool {
        let lock_target = match self.resolve_lock_id(lock_id) {
            Some(target) => target,
            None => return false,
        };

        let (target_tile, local_id): (&Tile, u8) = match lock_target {
            LockTarget::Own(id) => (tile, id),
            LockTarget::West(id) => match neighbors.west.as_deref() {
                Some(west) => (west, id),
                None => return false,
            },
            LockTarget::East(id) => match neighbors.east.as_deref() {
                Some(east) => (east, id),
                None => return false,
            },
        };

        let requestor = self.channel_requestor(ch_idx as u8);
        let granted = target_tile.lock_was_granted(requestor, local_id as usize);

        log::info!(
            "DMA check_acquire_granted tile({},{}) ch{} bd_lock={} target={:?} local_lock={} granted={}",
            self.col,
            self.row,
            ch_idx,
            lock_id,
            lock_target,
            local_id,
            granted
        );

        if let Some(ref mut timing) = self.lock_timing {
            timing.track_acquire(local_id as usize, granted);
        }

        granted
    }

    /// Execute a lock release operation (post-arbiter).
    ///
    /// The release was already submitted to the arbiter in `submit_lock_requests()`
    /// and resolved before `step()`. This method just logs for debugging.
    /// The lock value was already updated by the arbiter's `resolve()`.
    fn execute_lock_release(
        &mut self,
        lock_id: u8,
        release_value: i8,
        _tile: &mut Tile,
        _neighbors: &mut NeighborTiles<'_>,
    ) {
        log::info!(
            "DMA tile({},{}) lock release bd_lock={} delta={} (applied by arbiter)",
            self.col,
            self.row,
            lock_id,
            release_value
        );
    }

    /// Insert a packet header from a Transfer reference (used during BdSetup).
    ///
    /// Unlike the old maybe_insert_packet_header which accessed self.transfers[],
    /// this takes a Transfer reference directly from the FSM.
    ///
    /// Takes `&mut Transfer` so it can call `mark_packet_header_sent()` after
    /// successful insertion. This prevents double-insertion when the same
    /// Transfer passes through both `start_channel_with_repeat` and `BdSetup`
    /// completion (the no-lock path).
    pub(super) fn maybe_insert_packet_header_from_transfer(&mut self, transfer: &mut Transfer) {
        if !transfer.needs_packet_header() || transfer.direction != TransferDirection::MM2S {
            if transfer.enable_packet && transfer.direction == TransferDirection::MM2S {
                log::warn!(
                    "DMA({},{}) ch{} BD{}: enable_packet=true but needs_packet_header()={} header_sent={}",
                    self.col,
                    self.row,
                    transfer.channel,
                    transfer.bd_index,
                    transfer.needs_packet_header(),
                    transfer.packet_header_sent
                );
            }
            return;
        }

        if let Some(header_word) = transfer.generate_packet_header() {
            self.stream_out.push_back(StreamData {
                data: header_word,
                tlast: false,
                channel: transfer.channel,
            });
            transfer.mark_packet_header_sent();
            let (hdr, _) = crate::device::stream_switch::PacketHeader::decode(header_word);
            log::info!(
                "DMA({},{}) ch{} BD{} packet header: 0x{:08X} (pkt_id={}, type={:?})",
                self.col,
                self.row,
                transfer.channel,
                transfer.bd_index,
                header_word,
                hdr.stream_id,
                hdr.packet_type
            );
        }
    }

    /// Perform a data transfer operation.
    ///
    /// For stream transfers (MM2S/S2MM), data is buffered in stream_out/stream_in
    /// and will be routed by the TileArray's stream router.
    ///
    /// # Arguments
    /// * `tlast_suppress` - If true, TLAST is not asserted even on the last word
    fn do_transfer(
        &mut self,
        source: TransferEndpoint,
        dest: TransferEndpoint,
        addr: u64,
        bytes: usize,
        channel: u8,
        is_last: bool,
        tlast_suppress: bool,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        host_memory: &mut HostMemory,
    ) -> TransferResult {
        match (source, dest) {
            (TransferEndpoint::TileMemory { .. }, TransferEndpoint::Stream { .. }) => {
                // MM2S: Read from tile memory, queue to stream output.
                // Backpressure: stall when the DMA's own output queue has
                // saturated the downstream slave port FIFO depth. Models
                // STALL_STRM_STARV on real silicon, where MM2S blocks
                // until the consumer (next switch hop / S2MM / etc.)
                // pulls a word free.
                if self.stream_out.len() >= self.output_fifo_capacity() {
                    return TransferResult::stalled();
                }
                if self.transfer_mm2s(addr, bytes, channel, is_last, tlast_suppress, tile, neighbors) {
                    TransferResult::success()
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::Stream { .. }, TransferEndpoint::TileMemory { .. }) => {
                // S2MM: Read from stream input, write to tile memory
                let result = self.transfer_s2mm(addr, bytes, channel, tile, neighbors);
                if result.stall {
                    // No stream data available - stall (not an error)
                    return TransferResult::stalled();
                }
                if result.success {
                    // Check FoT mode: if enabled and TLAST received, signal early finish
                    let fot_mode = self.get_channel_fot_mode(channel);
                    let fot_finish = result.tlast_received && fot_mode != 0;

                    if fot_finish {
                        log::debug!(
                            "DMA({},{}) ch{} FoT mode {} triggered by TLAST ({} bytes written)",
                            self.col,
                            self.row,
                            channel,
                            fot_mode,
                            result.bytes_written
                        );
                    }

                    TransferResult { success: true, stall: false, fot_finish }
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::HostMemory, TransferEndpoint::TileMemory { .. }) => {
                if Self::transfer_host_to_tile_static(addr, bytes, tile, host_memory) {
                    TransferResult::success()
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::TileMemory { .. }, TransferEndpoint::HostMemory) => {
                // Record bank access for conflict detection (tile read by DMA)
                tile.record_dma_bank_access(addr as u32, bytes);
                if Self::transfer_tile_to_host_static(addr, bytes, tile, host_memory) {
                    TransferResult::success()
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::HostMemory, TransferEndpoint::Stream { .. }) => {
                // Shim MM2S: Read from host DDR, queue to stream output.
                // Same backpressure as tile MM2S above -- shim's outgoing
                // stream port stalls when the next slave FIFO is full,
                // which is what propagates the consumer drain rate (memtile
                // S2MM lock waits, compute kernel rate) back to the shim
                // and gates DMA_FINISHED_TASK timing.
                if self.stream_out.len() >= self.output_fifo_capacity() {
                    return TransferResult::stalled();
                }
                if self.transfer_host_to_stream(addr, bytes, channel, is_last, tlast_suppress, host_memory) {
                    TransferResult::success()
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::Stream { .. }, TransferEndpoint::HostMemory) => {
                // Shim S2MM: Read from stream input, write to host DDR
                let result = self.transfer_stream_to_host(addr, bytes, channel, host_memory);
                if result.stall {
                    // No stream data available - stall (not an error)
                    return TransferResult::stalled();
                }
                if result.success {
                    // Check FoT mode: if enabled and TLAST received, signal early finish
                    let fot_mode = self.get_channel_fot_mode(channel);
                    let fot_finish = result.tlast_received && fot_mode != 0;

                    if fot_finish {
                        log::debug!(
                            "DMA({},{}) Shim S2MM ch{} FoT mode {} triggered by TLAST ({} bytes written)",
                            self.col,
                            self.row,
                            channel,
                            fot_mode,
                            result.bytes_written
                        );
                    }

                    TransferResult { success: true, stall: false, fot_finish }
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::TileMemory { .. }, TransferEndpoint::TileMemory { .. }) => {
                // Tile to tile: Would need array access, mark as success for now
                TransferResult::success()
            }
            _ => TransferResult::failure(),
        }
    }

    /// Resolve an MM2S byte address to (target_tile, offset within it).
    ///
    /// MemTile DMAs use the windowed (West/Own/East) address space per
    /// `mlir-aie` (`getMemLocalBaseAddress`): West=`[0,mem_size)`,
    /// Own=`[mem_size,2*mem_size)`, East=`[2*mem_size,3*mem_size)`.
    ///
    /// **Missing-neighbour fallback:** mlir-aie's `dma_configure_task` path
    /// emits *flat* addresses on AIE2 (no window offset added) -- a BD with
    /// `addr=0x11000` on a column-0 MemTile is intended as Own[0x11000] even
    /// though it lands in the West window. Real hardware silently aliases
    /// the access to Own when the addressed neighbour does not exist (the
    /// neighbour bus has no other endpoint), so we mirror that: West/East
    /// targets with no neighbour fall back to Own at the same byte offset.
    /// Out-of-window addresses (>= 3*mem_size) are still fatal.
    ///
    /// Compute/shim DMAs are unchanged: they wrap at `mem_size`.
    fn resolve_mm2s_target<'a>(
        &mut self,
        addr: u64,
        mem_size: usize,
        own: &'a Tile,
        neighbors: &'a NeighborTiles<'_>,
    ) -> Option<(&'a Tile, usize)> {
        if !self.tile_kind.is_mem() {
            return Some((own, (addr as usize) % mem_size));
        }
        match MemTileTarget::resolve(addr, mem_size) {
            Ok((MemTileTarget::Own, off)) => Some((own, off)),
            Ok((MemTileTarget::West, off)) => {
                Some(neighbors.west.as_deref().map(|t| (t, off)).unwrap_or((own, off)))
            }
            Ok((MemTileTarget::East, off)) => {
                Some(neighbors.east.as_deref().map(|t| (t, off)).unwrap_or((own, off)))
            }
            Err(e) => {
                let msg = format!(
                    "DMA MemTile({},{}) MM2S addr=0x{:X} outside three-window MemTile space (window_size=0x{:X})",
                    self.col, self.row, e.byte_addr, e.mem_size,
                );
                log::error!("{}", msg);
                self.fatal_errors.push(msg);
                None
            }
        }
    }

    /// Resolve an S2MM byte address to (target_tile, offset within it).
    ///
    /// Mutable counterpart to `resolve_mm2s_target`; same windowing and
    /// missing-neighbour fallback semantics. Returns a `&mut Tile` so the
    /// caller can write through the MemTile-to-MemTile shared-memory bus
    /// (or the local data memory if the West/East target is absent).
    fn resolve_s2mm_target<'a>(
        &mut self,
        addr: u64,
        mem_size: usize,
        own: &'a mut Tile,
        neighbors: &'a mut NeighborTiles<'_>,
    ) -> Option<(&'a mut Tile, usize)> {
        if !self.tile_kind.is_mem() {
            return Some((own, (addr as usize) % mem_size));
        }
        match MemTileTarget::resolve(addr, mem_size) {
            Ok((MemTileTarget::Own, off)) => Some((own, off)),
            Ok((MemTileTarget::West, off)) => Some(match neighbors.west.as_deref_mut() {
                Some(t) => (t, off),
                None => (own, off),
            }),
            Ok((MemTileTarget::East, off)) => Some(match neighbors.east.as_deref_mut() {
                Some(t) => (t, off),
                None => (own, off),
            }),
            Err(e) => {
                let msg = format!(
                    "DMA MemTile({},{}) S2MM addr=0x{:X} outside three-window MemTile space (window_size=0x{:X})",
                    self.col, self.row, e.byte_addr, e.mem_size,
                );
                log::error!("{}", msg);
                self.fatal_errors.push(msg);
                None
            }
        }
    }

    /// MM2S: Read from tile memory and queue to stream output.
    ///
    /// For MemTile DMAs, the byte address is decoded into a West/Own/East
    /// window via `MemTileTarget::resolve()`. Compute/shim DMAs always read
    /// from the local tile (legacy `addr % mem_size` wrap).
    ///
    /// # Arguments
    /// * `tlast_suppress` - If true, TLAST is not asserted even on the last word
    pub(super) fn transfer_mm2s(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        is_last: bool,
        tlast_suppress: bool,
        tile: &Tile,
        neighbors: &NeighborTiles<'_>,
    ) -> bool {
        let mem_size = tile.data_memory().len();

        // Resolve target tile and offset.
        // MemTile uses the windowed (West/Own/East) address space; non-MemTile
        // tiles wrap at mem_size.
        let (target_tile, offset) = match self.resolve_mm2s_target(addr, mem_size, tile, neighbors) {
            Some(r) => r,
            None => return false,
        };

        // Record bank access for conflict detection (offset is local to the target tile)
        self.cycle_dma_banks |=
            crate::device::banking::banks_for_access(offset as u32, bytes, self.num_banks);

        if offset + bytes > mem_size {
            let msg = format!(
                "DMA({},{}) MM2S addr=0x{:X} bytes={} wraps past memory end (size=0x{:X}) -- bus error",
                self.col, self.row, addr, bytes, mem_size,
            );
            log::error!("{}", msg);
            self.fatal_errors.push(msg);
            return false;
        }

        let data = target_tile.data_memory();

        // When compression is enabled, read 32-byte blocks and compress each one.
        // Compressed output (mask + packed non-zero bytes) is pushed as 32-bit stream words.
        if self.is_compression_enabled(channel) {
            return self.transfer_mm2s_compressed(data, offset, bytes, channel, is_last, tlast_suppress);
        }

        // Uncompressed path: read data from tile memory in 32-bit words
        let word_count = (bytes + 3) / 4;

        log::debug!(
            "DMA({},{}) MM2S ch{}: addr=0x{:X} offset=0x{:X} bytes={} words={}",
            self.col,
            self.row,
            channel,
            addr,
            offset,
            bytes,
            word_count
        );

        for i in 0..word_count {
            let word_offset = offset + i * 4;
            let word = if word_offset + 4 <= data.len() {
                u32::from_le_bytes([
                    data[word_offset],
                    data[word_offset + 1],
                    data[word_offset + 2],
                    data[word_offset + 3],
                ])
            } else {
                // Partial word at end
                let mut word_bytes = [0u8; 4];
                for j in 0..4 {
                    if word_offset + j < data.len() {
                        word_bytes[j] = data[word_offset + j];
                    }
                }
                u32::from_le_bytes(word_bytes)
            };

            if i < 2 || i == word_count - 1 {
                log::debug!(
                    "DMA({},{}) MM2S ch{} word[{}]: offset=0x{:X} value=0x{:08X}",
                    self.col,
                    self.row,
                    channel,
                    i,
                    word_offset,
                    word
                );
            }

            if crate::debug::watch::is_watched(word_offset as u64, 4) {
                crate::debug::watch::log_dma_read(
                    self.current_cycle,
                    self.col,
                    self.row,
                    word_offset as u64,
                    word,
                    &format!("MM2S{}", channel),
                );
            }

            // TLAST is asserted on last word unless suppressed
            // AM025: TLAST_Suppress (Word 5, bit 31) prevents TLAST assertion
            let is_last_word = is_last && (i == word_count - 1);
            let should_assert_tlast = is_last_word && !tlast_suppress;
            self.stream_out
                .push_back(StreamData { data: word, tlast: should_assert_tlast, channel });
        }

        true
    }

    /// MM2S compressed path: read 32-byte blocks, compress, push to stream.
    ///
    /// Sparsity compression (AM020 Ch1) operates on 256-bit (32-byte) blocks.
    /// Each block produces a 32-bit mask followed by packed non-zero bytes,
    /// padded to a 4-byte boundary. The compressed output is pushed as 32-bit
    /// stream words.
    fn transfer_mm2s_compressed(
        &mut self,
        data: &[u8],
        offset: usize,
        bytes: usize,
        channel: u8,
        is_last: bool,
        tlast_suppress: bool,
    ) -> bool {
        const BLOCK_SIZE: usize = 32;

        log::debug!(
            "DMA({},{}) MM2S ch{} COMPRESSED: offset=0x{:X} bytes={}",
            self.col,
            self.row,
            channel,
            offset,
            bytes
        );

        let num_blocks = (bytes + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for block_idx in 0..num_blocks {
            let block_start = offset + block_idx * BLOCK_SIZE;
            let block_end = (block_start + BLOCK_SIZE).min(offset + bytes).min(data.len());
            let block_len = block_end - block_start;

            // Pad to 32 bytes if this is a partial final block
            let mut block = [0u8; BLOCK_SIZE];
            block[..block_len].copy_from_slice(&data[block_start..block_end]);

            let compressed = match compression::compress(&block) {
                Some(c) => c,
                None => {
                    let msg = format!(
                        "DMA({},{}) MM2S ch{} compression failed at block {} -- data corruption",
                        self.col, self.row, channel, block_idx,
                    );
                    log::error!("{}", msg);
                    self.fatal_errors.push(msg);
                    return false;
                }
            };

            log::debug!(
                "DMA({},{}) MM2S ch{} block {}: {} bytes -> {} compressed bytes",
                self.col,
                self.row,
                channel,
                block_idx,
                BLOCK_SIZE,
                compressed.len()
            );

            // Push compressed bytes as 32-bit stream words
            let compressed_words = compressed.len() / 4;
            let is_last_block = is_last && block_idx == num_blocks - 1;

            for w in 0..compressed_words {
                let wi = w * 4;
                let word = u32::from_le_bytes([
                    compressed[wi],
                    compressed[wi + 1],
                    compressed[wi + 2],
                    compressed[wi + 3],
                ]);

                let is_last_word = is_last_block && w == compressed_words - 1;
                let should_assert_tlast = is_last_word && !tlast_suppress;
                self.stream_out
                    .push_back(StreamData { data: word, tlast: should_assert_tlast, channel });
            }
        }

        true
    }

    /// S2MM: Read from stream input, write to tile memory.
    ///
    /// Only transfers data that is available in stream_in for the specified channel.
    /// Returns S2mmResult indicating success, TLAST reception, and bytes written.
    pub(super) fn transfer_s2mm(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
    ) -> S2mmResult {
        let mem_size = tile.data_memory().len();

        // Resolve target tile and offset.
        // MemTile uses the windowed (West/Own/East) address space; non-MemTile
        // tiles wrap at mem_size.
        let (target_tile, offset) = match self.resolve_s2mm_target(addr, mem_size, tile, neighbors) {
            Some(r) => r,
            None => {
                return S2mmResult { success: false, stall: false, tlast_received: false, bytes_written: 0 }
            }
        };

        // Record bank access for conflict detection (offset is local to the target tile)
        self.cycle_dma_banks |=
            crate::device::banking::banks_for_access(offset as u32, bytes, self.num_banks);

        if offset + bytes > mem_size {
            let msg = format!(
                "DMA({},{}) S2MM addr=0x{:X} bytes={} wraps past memory end (size=0x{:X}) -- bus error",
                self.col, self.row, addr, bytes, mem_size,
            );
            log::error!("{}", msg);
            self.fatal_errors.push(msg);
            return S2mmResult { success: false, stall: false, tlast_received: false, bytes_written: 0 };
        }

        // When decompression is enabled, consume compressed blocks from stream
        // and write decompressed 32-byte blocks to memory. Decompression has
        // its own word-by-word stall logic so skip the atomic beat check.
        if self.is_decompression_enabled(channel) {
            if !self.has_stream_in_for_channel(channel) {
                return S2mmResult { success: true, stall: true, tlast_received: false, bytes_written: 0 };
            }
            return self.transfer_s2mm_decompressed(offset, bytes, channel, target_tile);
        }

        // Uncompressed path: the DMA bus transfers atomically -- all words
        // for this beat must be available in the stream before we proceed.
        let words_needed = (bytes + 3) / 4;
        let words_available = self.stream_in_count_for_channel(channel);
        if words_available < words_needed {
            return S2mmResult { success: true, stall: true, tlast_received: false, bytes_written: 0 };
        }

        // On real hardware, S2MM ignores TLAST unless Finish-on-TLAST (FoT)
        // mode is enabled. Without FoT, the DMA continues accepting words
        // until the BD's transfer length is satisfied, regardless of TLAST
        // markers on the stream.
        let fot_enabled = self.get_channel_fot_mode(channel) != 0;

        // Write data to tile memory in 32-bit words.
        // For MemTile DMAs this may be a neighbour's memory via the shared bus.
        let data = target_tile.data_memory_mut();
        let mut bytes_written = 0;
        let word_count = (bytes + 3) / 4;
        let mut tlast_received = false;

        log::debug!(
            "DMA({},{}) S2MM ch{}: addr=0x{:X} offset=0x{:X} bytes={} words={}",
            self.col,
            self.row,
            channel,
            addr,
            offset,
            bytes,
            word_count
        );
        for word_idx in 0..word_count {
            // Get data from stream for this specific channel
            if let Some(stream_data) = self.pop_stream_in_for_channel(channel) {
                let word = stream_data.data;
                let word_bytes = word.to_le_bytes();

                if word_idx < 2 || word_idx == word_count - 1 {
                    log::debug!(
                        "DMA({},{}) S2MM ch{} word[{}]: offset=0x{:X} value=0x{:08X}",
                        self.col,
                        self.row,
                        channel,
                        word_idx,
                        offset + bytes_written,
                        word
                    );
                }
                for j in 0..4 {
                    if bytes_written + j < bytes && offset + bytes_written + j < data.len() {
                        data[offset + bytes_written + j] = word_bytes[j];
                    }
                }

                if crate::debug::watch::is_watched((offset + bytes_written) as u64, 4) {
                    crate::debug::watch::log_dma_write(
                        self.current_cycle,
                        self.col,
                        self.row,
                        (offset + bytes_written) as u64,
                        word,
                        &format!("S2MM{}", channel),
                    );
                }

                bytes_written += 4;

                // Track TLAST for FoT mode
                if stream_data.tlast {
                    tlast_received = true;
                    if fot_enabled {
                        break; // FoT: stop receiving at TLAST boundary
                    }
                }
            } else {
                // No more stream data for this channel - transfer partial, continue next step
                break;
            };
        }

        S2mmResult { success: true, stall: false, tlast_received, bytes_written }
    }

    /// S2MM decompressed path: read compressed blocks from stream, decompress, write to memory.
    ///
    /// Sparsity decompression (AM020 Ch1) consumes compressed blocks from the stream.
    /// Each block starts with a 32-bit mask word, followed by ceil(popcount(mask)/4)
    /// data words containing packed non-zero bytes. The decompressed 32-byte output
    /// is written to tile memory.
    ///
    /// `target_tile` is the resolved destination (own or neighbour MemTile);
    /// `offset` is local to that tile's data memory.
    fn transfer_s2mm_decompressed(
        &mut self,
        offset: usize,
        bytes: usize,
        channel: u8,
        target_tile: &mut Tile,
    ) -> S2mmResult {
        const BLOCK_SIZE: usize = 32;

        log::debug!(
            "DMA({},{}) S2MM ch{} DECOMPRESSED: offset=0x{:X} bytes={}",
            self.col,
            self.row,
            channel,
            offset,
            bytes
        );

        let mut mem_bytes_written: usize = 0;
        let mut tlast_received = false;

        // Process compressed blocks until we have written enough decompressed bytes
        while mem_bytes_written < bytes {
            // Read the mask word (first word of compressed block)
            let mask_data = match self.pop_stream_in_for_channel(channel) {
                Some(sd) => sd,
                None => break, // No more stream data; partial transfer
            };

            if mask_data.tlast {
                // TLAST on the mask word itself -- unusual, but handle gracefully.
                // Treat as an empty block (mask only, no data bytes).
                tlast_received = true;
                break;
            }

            let mask = mask_data.data;
            let non_zero_count = mask.count_ones() as usize;
            let data_bytes_needed = non_zero_count;
            let data_words_needed = (data_bytes_needed + 3) / 4;

            // Collect the data words for this compressed block
            let mut compressed_buf = Vec::with_capacity(4 + data_words_needed * 4);
            compressed_buf.extend_from_slice(&mask.to_le_bytes());

            let mut got_tlast = false;
            for _ in 0..data_words_needed {
                match self.pop_stream_in_for_channel(channel) {
                    Some(sd) => {
                        compressed_buf.extend_from_slice(&sd.data.to_le_bytes());
                        if sd.tlast {
                            got_tlast = true;
                            break;
                        }
                    }
                    None => break, // Stream starved mid-block
                }
            }

            // Decompress (tolerates short input: missing bytes decompress as zero)
            match compression::decompress(&compressed_buf) {
                Some(decompressed) => {
                    let data = target_tile.data_memory_mut();
                    let write_len = BLOCK_SIZE.min(bytes - mem_bytes_written);
                    let dest_start = offset + mem_bytes_written;
                    let dest_end = (dest_start + write_len).min(data.len());
                    let actual_write = dest_end - dest_start;

                    data[dest_start..dest_end].copy_from_slice(&decompressed[..actual_write]);
                    mem_bytes_written += actual_write;

                    log::debug!("DMA({},{}) S2MM ch{} decompressed block: mask=0x{:08X} {} non-zero -> {} bytes to mem",
                        self.col, self.row, channel, mask, non_zero_count, actual_write);
                }
                None => {
                    let msg = format!(
                        "DMA({},{}) S2MM ch{} decompression failed (mask=0x{:08X}, buf_len={}) -- data corruption",
                        self.col, self.row, channel, mask, compressed_buf.len(),
                    );
                    log::error!("{}", msg);
                    self.fatal_errors.push(msg);
                    return S2mmResult {
                        success: false,
                        stall: false,
                        tlast_received: false,
                        bytes_written: mem_bytes_written,
                    };
                }
            }

            if got_tlast {
                tlast_received = true;
                break;
            }
        }

        S2mmResult {
            success: mem_bytes_written > 0,
            stall: false,
            tlast_received,
            bytes_written: mem_bytes_written,
        }
    }

    /// Shim MM2S: Read from host DDR and queue to stream output.
    ///
    /// # Arguments
    /// * `tlast_suppress` - If true, TLAST is not asserted even on the last word
    fn transfer_host_to_stream(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        is_last: bool,
        tlast_suppress: bool,
        host_memory: &HostMemory,
    ) -> bool {
        // When compression is enabled, read 32-byte blocks and compress
        if self.is_compression_enabled(channel) {
            return self.transfer_host_to_stream_compressed(
                addr,
                bytes,
                channel,
                is_last,
                tlast_suppress,
                host_memory,
            );
        }

        // Uncompressed path: read data from host memory in 32-bit words
        let word_count = (bytes + 3) / 4;

        log::debug!("MM2S transfer: addr=0x{:X} bytes={} words={}", addr, bytes, word_count);
        for i in 0..word_count {
            let word_addr = addr + (i * 4) as u64;
            let word = host_memory.read_u32(word_addr);

            if crate::debug::watch::is_watched(word_addr, 4) {
                crate::debug::watch::log_dma_read(
                    self.current_cycle,
                    self.col,
                    self.row,
                    word_addr,
                    word,
                    &format!("MM2S{}", channel),
                );
            }

            if i < 4 {
                log::debug!("  MM2S word[{}]: addr=0x{:X} -> 0x{:08X}", i, word_addr, word);
            }

            // TLAST is asserted on last word unless suppressed
            // AM025: TLAST_Suppress (Word 5, bit 31) prevents TLAST assertion
            let is_last_word = is_last && i == word_count - 1;
            let should_assert_tlast = is_last_word && !tlast_suppress;
            self.stream_out
                .push_back(StreamData { data: word, channel, tlast: should_assert_tlast });
        }

        true
    }

    /// Shim MM2S compressed path: read 32-byte blocks from host DDR, compress, push to stream.
    fn transfer_host_to_stream_compressed(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        is_last: bool,
        tlast_suppress: bool,
        host_memory: &HostMemory,
    ) -> bool {
        const BLOCK_SIZE: usize = 32;

        log::debug!("Shim MM2S COMPRESSED: addr=0x{:X} bytes={}", addr, bytes);

        let num_blocks = (bytes + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for block_idx in 0..num_blocks {
            let block_start = addr + (block_idx * BLOCK_SIZE) as u64;
            let block_remaining = bytes - block_idx * BLOCK_SIZE;
            let block_len = BLOCK_SIZE.min(block_remaining);

            // Read 32-byte block from host memory
            let mut block = [0u8; BLOCK_SIZE];
            for i in 0..block_len {
                let byte_addr = block_start + i as u64;
                // Read individual bytes using word-aligned reads
                let word_addr = byte_addr & !3;
                let byte_offset = (byte_addr & 3) as usize;
                let word = host_memory.read_u32(word_addr);
                block[i] = word.to_le_bytes()[byte_offset];
            }

            let compressed = match compression::compress(&block) {
                Some(c) => c,
                None => {
                    log::error!("Shim MM2S ch{} compression failed at block {}", channel, block_idx);
                    return false;
                }
            };

            let compressed_words = compressed.len() / 4;
            let is_last_block = is_last && block_idx == num_blocks - 1;

            for w in 0..compressed_words {
                let wi = w * 4;
                let word = u32::from_le_bytes([
                    compressed[wi],
                    compressed[wi + 1],
                    compressed[wi + 2],
                    compressed[wi + 3],
                ]);

                let is_last_word = is_last_block && w == compressed_words - 1;
                let should_assert_tlast = is_last_word && !tlast_suppress;
                self.stream_out
                    .push_back(StreamData { data: word, channel, tlast: should_assert_tlast });
            }
        }

        true
    }

    /// Shim S2MM: Read from stream input and write to host DDR.
    ///
    /// Returns S2mmResult to properly handle stalls when no stream data is available.
    fn transfer_stream_to_host(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        host_memory: &mut HostMemory,
    ) -> S2mmResult {
        // Atomic beat: stall until stream has enough words for this transfer.
        let words_needed = (bytes + 3) / 4;
        let words_available = self.stream_in_count_for_channel(channel);
        if words_available < words_needed {
            return S2mmResult { success: true, stall: true, tlast_received: false, bytes_written: 0 };
        }

        // When decompression is enabled, consume compressed blocks from stream
        if self.is_decompression_enabled(channel) {
            return self.transfer_stream_to_host_decompressed(addr, bytes, channel, host_memory);
        }

        // On real hardware, S2MM ignores TLAST unless Finish-on-TLAST (FoT)
        // mode is enabled. Without FoT, the DMA continues accepting words
        // until the BD's transfer length is satisfied, regardless of TLAST
        // markers on the stream. Breaking unconditionally on TLAST causes
        // address pointer skips when TLAST falls mid-batch (e.g., 10-word
        // source chunks vs 4-word DMA bus width).
        let fot_enabled = self.get_channel_fot_mode(channel) != 0;

        // Uncompressed path: write data to host memory in 32-bit words
        let mut bytes_written = 0;
        let word_count = (bytes + 3) / 4;
        let mut tlast_received = false;

        for i in 0..word_count {
            let stream_data = if let Some(sd) = self.pop_stream_in_for_channel(channel) {
                sd
            } else {
                // No more stream data for this channel - transfer partial, continue next step
                break;
            };

            let word = stream_data.data;
            let word_addr = addr + (i * 4) as u64;
            log::debug!("Shim S2MM write: addr=0x{:X} word=0x{:08X}", word_addr, word);
            host_memory.write_u32(word_addr, word);

            if crate::debug::watch::is_watched(word_addr, 4) {
                crate::debug::watch::log_dma_write(
                    self.current_cycle,
                    self.col,
                    self.row,
                    word_addr,
                    word,
                    &format!("S2MM{}", channel),
                );
            }

            bytes_written += 4;

            // Track TLAST for FoT mode
            if stream_data.tlast {
                tlast_received = true;
                if fot_enabled {
                    break; // FoT: stop receiving at TLAST boundary
                }
            }
        }

        S2mmResult { success: bytes_written > 0, stall: false, tlast_received, bytes_written }
    }

    /// Shim S2MM decompressed path: read compressed blocks from stream, decompress, write to host DDR.
    fn transfer_stream_to_host_decompressed(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        host_memory: &mut HostMemory,
    ) -> S2mmResult {
        const BLOCK_SIZE: usize = 32;

        log::debug!("Shim S2MM DECOMPRESSED: addr=0x{:X} bytes={}", addr, bytes);

        let mut mem_bytes_written: usize = 0;
        let mut tlast_received = false;

        while mem_bytes_written < bytes {
            // Read the mask word
            let mask_data = match self.pop_stream_in_for_channel(channel) {
                Some(sd) => sd,
                None => break,
            };

            if mask_data.tlast {
                tlast_received = true;
                break;
            }

            let mask = mask_data.data;
            let non_zero_count = mask.count_ones() as usize;
            let data_words_needed = (non_zero_count + 3) / 4;

            let mut compressed_buf = Vec::with_capacity(4 + data_words_needed * 4);
            compressed_buf.extend_from_slice(&mask.to_le_bytes());

            let mut got_tlast = false;
            for _ in 0..data_words_needed {
                match self.pop_stream_in_for_channel(channel) {
                    Some(sd) => {
                        compressed_buf.extend_from_slice(&sd.data.to_le_bytes());
                        if sd.tlast {
                            got_tlast = true;
                            break;
                        }
                    }
                    None => break,
                }
            }

            match compression::decompress(&compressed_buf) {
                Some(decompressed) => {
                    let write_len = BLOCK_SIZE.min(bytes - mem_bytes_written);
                    let dest_addr = addr + mem_bytes_written as u64;

                    // Write decompressed bytes to host memory via word-aligned writes
                    let full_words = write_len / 4;
                    for w in 0..full_words {
                        let wi = w * 4;
                        let word = u32::from_le_bytes([
                            decompressed[wi],
                            decompressed[wi + 1],
                            decompressed[wi + 2],
                            decompressed[wi + 3],
                        ]);
                        host_memory.write_u32(dest_addr + wi as u64, word);
                    }
                    // Handle trailing bytes (partial last word)
                    let trailing = write_len % 4;
                    if trailing > 0 {
                        let wi = full_words * 4;
                        let mut word_bytes = [0u8; 4];
                        word_bytes[..trailing].copy_from_slice(&decompressed[wi..wi + trailing]);
                        host_memory.write_u32(dest_addr + wi as u64, u32::from_le_bytes(word_bytes));
                    }

                    mem_bytes_written += write_len;
                }
                None => {
                    log::error!("Shim S2MM ch{} decompression failed (mask=0x{:08X})", channel, mask);
                    return S2mmResult {
                        success: false,
                        stall: false,
                        tlast_received: false,
                        bytes_written: mem_bytes_written,
                    };
                }
            }

            if got_tlast {
                tlast_received = true;
                break;
            }
        }

        S2mmResult {
            success: mem_bytes_written > 0,
            stall: false,
            tlast_received,
            bytes_written: mem_bytes_written,
        }
    }

    /// Transfer data from host memory to tile memory (static version).
    fn transfer_host_to_tile_static(
        tile_addr: u64,
        bytes: usize,
        tile: &mut Tile,
        host_memory: &mut HostMemory,
    ) -> bool {
        let offset = tile_addr as usize;
        if offset + bytes > tile.data_memory().len() {
            return false;
        }

        // Record bank access for conflict detection
        tile.record_dma_bank_access(offset as u32, bytes);

        // Read from host memory
        let mut buf = vec![0u8; bytes];
        // Note: In a real implementation, we'd track a separate host address
        // For now, we use the tile address as an offset into host memory too
        host_memory.read_bytes(tile_addr, &mut buf);

        // Write to tile memory
        tile.data_memory_mut()[offset..offset + bytes].copy_from_slice(&buf);

        true
    }

    /// Transfer data from tile memory to host memory (static version).
    fn transfer_tile_to_host_static(
        tile_addr: u64,
        bytes: usize,
        tile: &Tile,
        host_memory: &mut HostMemory,
    ) -> bool {
        let offset = tile_addr as usize;
        if offset + bytes > tile.data_memory().len() {
            return false;
        }

        // Read from tile memory
        let data = &tile.data_memory()[offset..offset + bytes];

        // Write to host memory
        // Note: In a real implementation, we'd track a separate host address
        host_memory.write_bytes(tile_addr, data);

        true
    }

    // NOTE: complete_transfer() and finish_complete_transfer() have been
    // replaced by begin_completion() and after_transfer_done() in the FSM.

    /// Emit a task complete token if Enable_Token_Issue is set for this channel.
    fn maybe_emit_task_token(&mut self, ch_idx: usize) {
        let config = &self.channels[ch_idx].task_config;

        if config.enable_token_issue {
            log::debug!(
                "DMA tile({},{}) ch{} emitting task complete token (controller_id={})",
                self.col,
                self.row,
                ch_idx,
                config.controller_id
            );

            self.task_tokens.issue(ch_idx as u8, config.controller_id);

            // Clear enable_token_issue after issuing (it's set per-task via Start_Queue)
            self.channels[ch_idx].task_config.enable_token_issue = false;
        }
    }
}
