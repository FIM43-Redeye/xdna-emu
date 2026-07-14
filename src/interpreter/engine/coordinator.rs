//! Multi-core coordinator implementation.
//!
//! The engine manages multiple core interpreters and coordinates their execution.
//! It also coordinates DMA engines and host memory for data transfers.
//!
//! # Execution Mode
//!
//! All execution is cycle-accurate using `CycleAccurateExecutor` with full
//! AM020 timing model - hazard detection, memory bank conflicts, branch
//! penalties, and event tracing. This ensures consistent, accurate behavior
//! that matches the real hardware.

use crate::device::bank_arbiter::{BankArbiter, Requester};
use crate::device::clock_control::ModuleKind;
use crate::device::dma::ChannelState;
use crate::device::host_memory::HostMemory;
use crate::device::tile::Lock;
use crate::device::DeviceState;
use crate::interpreter::bundle::VliwBundle;
use crate::interpreter::core::{CoreInterpreter, CoreStatus, StepResult};
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::execute::{CycleAccurateExecutor, NeighborLocks, NeighborMemory};
use crate::interpreter::state::ExecutionContext;
use crate::parser::AieElf;
use smallvec::SmallVec;
use xdna_archspec::aie2::SHIM_ROW;
use xdna_archspec::types::TileKind;

/// This cycle's arbitration outcome for one compute tile.
///
/// Produced by the request/arbitrate pass (Phases A-C of `step`) and consumed
/// by the commit pass (Phase D) and the emission pass (Phase E).
#[derive(Default, Clone)]
struct TileArbitration {
    /// DMA channels that lost and must be held this cycle.
    denied_dma: Vec<Requester>,
    /// The core lost at least one bank it needed: its bundle does not commit.
    core_lost: bool,
    /// Banks with more than one requester this cycle (CONFLICT_DM_BANK_n).
    contended_banks: u16,
}

/// Per-direction DMA channel count the bank arbiter ever produces
/// `Requester::S2mm`/`Mm2s` demands for -- 2 on a compute tile. Sizes the
/// `for ch in 0..DMA_BANK_CHANNELS_PER_DIRECTION` sweep below. Derives from
/// the arbiter's own `NUM_DMA_CHANNELS` (not re-derived from `xdna_archspec`
/// directly) so there is one canonical source, not two that happen to agree.
const DMA_BANK_CHANNELS_PER_DIRECTION: u8 = crate::device::bank_arbiter::NUM_DMA_CHANNELS as u8;

/// Build a `NeighborLocks` from the executing tile's isolation byte and
/// per-direction lock slices. A set bit in `isolation` (per
/// [`crate::device::tile::isolation`]) hides the corresponding direction's
/// slice from the executor; `route_lock` then falls back to the own-tile
/// lock array. East/Internal quadrant locks always live on the executing
/// tile, so they're unaffected by isolation.
pub(crate) fn build_neighbor_locks_with_isolation<'a>(
    isolation: u8,
    south: Option<&'a mut [Lock]>,
    west: Option<&'a mut [Lock]>,
    north: Option<&'a mut [Lock]>,
) -> NeighborLocks<'a> {
    use crate::device::tile::isolation as iso;
    NeighborLocks {
        south: if isolation & iso::SOUTH != 0 { None } else { south },
        west: if isolation & iso::WEST != 0 { None } else { west },
        north: if isolation & iso::NORTH != 0 { None } else { north },
    }
}

/// Engine execution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EngineStatus {
    /// Engine is ready to run.
    #[default]
    Ready,
    /// Engine is running (at least one core active).
    Running,
    /// Engine is paused.
    Paused,
    /// All cores have halted.
    Halted,
    /// No monotonic progress for stall_threshold cycles.
    Stalled,
    /// Engine encountered an error.
    Error,
}

/// Type alias for the cycle-accurate interpreter.
type Interpreter = CoreInterpreter<InstructionDecoder, CycleAccurateExecutor>;

/// Per-core state managed by the engine.
struct CoreState {
    /// Core interpreter (cycle-accurate).
    interpreter: Interpreter,
    /// Execution context (registers, PC, flags).
    context: ExecutionContext,
    /// Cross-tile neighbor memory snapshots, hoisted across steps.
    /// `ensure_snapshot` is gen-aware, so persisting this across steps
    /// turns most per-step calls into cheap cache hits (see `NeighborMemory`).
    neighbors: NeighborMemory,
    /// Is this core enabled?
    enabled: bool,
    /// Absolute count of trace events already consumed from the
    /// EventLog. Tracked in EventLog's monotonic `total_recorded`
    /// space, NOT as an index into the current `events()` slice --
    /// the latter shifts down each time the circular buffer drops its
    /// oldest entry, silently stranding the cursor past end-of-slice
    /// once the buffer wraps.
    trace_events_consumed: u64,
    /// True if this core executed an instruction (StepResult::Continue)
    /// during the current cycle's execute phase. Used by the tick phase to
    /// route ACTIVE_CORE-configured perf counters to tick_active_cycles vs
    /// tick_idle_cycles. Reset to false at the start of each cycle.
    active_this_cycle: bool,
    /// Held-level state for MEMORY_STALL (core event 23): true while this core
    /// is in a sustained DMA bank conflict. Drives the rising/falling edges so
    /// a multi-cycle conflict renders as one B..E span rather than one pulse
    /// per conflicting cycle. See `mem_stall_edge`.
    mem_stall_active: bool,
}

/// Held-level edge decision for a per-cycle win/loss boolean against its
/// previous level -- shared by MEMORY_STALL (core event 23, per-core) and
/// DMA_x_MEMORY_BACKPRESSURE/STARVATION (mem events 39-42, per DMA channel).
///
/// Given whether a requester lost its bank arbitration THIS cycle
/// (`conflicting`) and its previous held level (`was_active`), returns the
/// trace edge to emit:
///   `Some(true)`  -- rising edge (conflict begins)
///   `Some(false)` -- falling edge (conflict clears)
///   `None`        -- no transition (sustained conflict, or sustained idle)
///
/// Returning `None` on a sustained conflict is what collapses N consecutive
/// bank-conflict cycles into a single held span instead of N one-cycle pulses,
/// matching how HW samples these events as level signals.
fn mem_stall_edge(conflicting: bool, was_active: bool) -> Option<bool> {
    match (conflicting, was_active) {
        (true, false) => Some(true),
        (false, true) => Some(false),
        _ => None,
    }
}

impl CoreState {
    /// Create a new core state for a specific tile position.
    ///
    /// The tile's column and row are used to initialize the read-only
    /// CORE_ID register, so the core knows its position in the array.
    fn new(col: u8, row: u8) -> Self {
        Self {
            interpreter: CoreInterpreter::new(
                InstructionDecoder::load_default(),
                CycleAccurateExecutor::new(),
            ),
            context: ExecutionContext::new_for_tile(col, row),
            neighbors: NeighborMemory::new(col as usize, row as usize),
            enabled: false,
            trace_events_consumed: 0,
            active_this_cycle: false,
            mem_stall_active: false,
        }
    }
}

/// Multi-core interpreter engine.
///
/// Coordinates execution across all compute cores in the device.
/// Also manages DMA engines and host memory for data transfers.
///
/// All execution is cycle-accurate with AM020-based timing - register hazard
/// detection (RAW, WAW, WAR), memory bank conflict modeling, branch penalties,
/// and event tracing for profiling.
pub struct InterpreterEngine {
    /// Device state (tiles, memory, locks, DMA engines).
    device: DeviceState,
    /// Host memory for DMA transfers to/from external DDR.
    host_memory: HostMemory,
    /// Per-core execution state (indexed by column * max_rows + row).
    cores: Vec<CoreState>,
    /// One set of per-physical-bank round-robin arbiters per tile, indexed by
    /// tile index. AM020 ch.2:166: "Each memory bank has its own arbitrator";
    /// the arbiters live in the tile's memory module and their rotor is
    /// persistent state, so they must survive across cycles. Only compute
    /// tiles ever arbitrate (memtile bank geometry is unvalidated), but the
    /// vector is sized per tile so `tile_index` addresses it directly.
    bank_arbiters: Vec<BankArbiter>,
    /// Number of columns.
    cols: usize,
    /// Number of rows.
    rows: usize,
    /// Compute row start (first row with compute tiles).
    compute_row_start: usize,
    /// Engine status.
    status: EngineStatus,
    /// Total cycles executed.
    total_cycles: u64,
    /// Total instructions executed across all cores.
    total_instructions: u64,
    /// Auto-run mode.
    auto_run: bool,
    /// Stall detection: last observed total DMA bytes transferred.
    last_dma_bytes: u64,
    /// Stall detection: last observed total lock releases.
    last_lock_releases: u64,
    /// Stall detection: last observed total instructions executed.
    last_instructions: u64,
    /// Consecutive cycles with no monotonic progress.
    stall_cycles: u64,
    /// Cycles of no progress before declaring stall. 0 = disabled.
    stall_threshold: u64,
    /// Count of control-packet ordering hazards observed (see
    /// `note_ctrl_packet_ordering_hazard`). Should be 0 for the
    /// executor-direct-config majority of the corpus; non-zero only flags
    /// control-packet kernels where the deleted flush would have force-delivered.
    ctrl_packet_hazard_count: u64,
}

impl InterpreterEngine {
    /// Create a new engine from device state.
    ///
    /// Uses `CycleAccurateExecutor` with full AM020 timing model:
    /// - Register hazard detection (RAW, WAW, WAR)
    /// - Memory bank conflict modeling
    /// - Branch penalty tracking
    /// - Event tracing for profiling
    pub fn new(device: DeviceState) -> Self {
        let cols = device.cols();
        let rows = device.rows();
        let compute_row_start = 2; // Rows 0=shim, 1=memtile, 2+=compute

        // Create core states for all possible positions, each initialized
        // with its tile coordinates so CORE_ID is correct.
        let num_cores = cols * rows;
        let cores = (0..num_cores)
            .map(|idx| {
                let col = (idx / rows) as u8;
                let row = (idx % rows) as u8;
                CoreState::new(col, row)
            })
            .collect();

        let bank_arbiters = (0..device.array.tiles.len()).map(|_| BankArbiter::new()).collect();

        Self {
            device,
            host_memory: HostMemory::new(),
            cores,
            bank_arbiters,
            cols,
            rows,
            compute_row_start,
            status: EngineStatus::Ready,
            total_cycles: 0,
            total_instructions: 0,
            auto_run: false,
            last_dma_bytes: 0,
            last_lock_releases: 0,
            last_instructions: 0,
            stall_cycles: 0,
            stall_threshold: 0,
            ctrl_packet_hazard_count: 0,
        }
    }

    /// Create a new engine with cycle-accurate timing.
    /// Create engine for NPU1 (Phoenix).
    pub fn new_npu1() -> Self {
        Self::new(DeviceState::new_npu1())
    }

    /// Create engine for NPU2 (Strix).
    pub fn new_npu2() -> Self {
        Self::new(DeviceState::new_npu2())
    }

    /// All execution is cycle-accurate.
    pub fn is_cycle_accurate(&self) -> bool {
        true
    }

    /// Get the engine status.
    pub fn status(&self) -> EngineStatus {
        self.status
    }

    /// Set the stall detection threshold. 0 disables stall detection.
    pub fn set_stall_threshold(&mut self, threshold: u64) {
        self.stall_threshold = threshold;
    }

    /// Reset status from Halted to Running.
    ///
    /// The engine halts when no cores are enabled and no DMA is active,
    /// which is correct for its local view. But the NPU instruction
    /// executor may still be configuring and triggering DMA from the
    /// host command stream. The run loop calls this when it knows
    /// external work is still in progress, so `step()` continues
    /// advancing DMA engines and stream switches.
    pub fn force_running(&mut self) {
        if self.status == EngineStatus::Halted {
            self.status = EngineStatus::Running;
        }
    }

    /// Get total cycles executed.
    pub fn total_cycles(&self) -> u64 {
        self.total_cycles
    }

    /// Get total instructions executed across all cores.
    pub fn total_instructions(&self) -> u64 {
        self.total_instructions
    }

    /// Flush trace units and route final trace packets to host DDR.
    ///
    /// Call after execution completes. Flushes all tile trace units so partial
    /// packets get emitted with padding, then runs additional routing passes
    /// to deliver them through the stream switch network to host memory.
    pub fn flush_trace_to_host(&mut self) {
        // Flush all tile trace units so partial packets get emitted with padding.
        // This ensures the final trace data is available for stream routing.
        for col in 0..self.device.array.cols() {
            for row in 0..self.device.array.rows() {
                if let Some(tile) = self.device.array.get_mut(col as u8, row as u8) {
                    tile.core_trace.flush();
                    tile.mem_trace.flush();
                }
            }
        }

        // Route flushed trace packets through the stream switch to host DDR.
        // During normal execution, step_data_movement() runs every step() cycle.
        // But flush() above creates final packets AFTER the execution loop exits,
        // so we need additional routing passes to deliver them through the
        // multi-hop stream switch network.
        //
        // Each word traverses multiple hops (e.g., Compute -> MemTile -> Shim
        // -> DMA = 3-4 hops). Each hop takes one routing pass. Two 8-word
        // packets need ~64 passes in the worst case. Use a generous limit
        // and stop early when nothing moves.
        let mut total_flush_words = 0;
        let mut flush_iters = 0;
        for _ in 0..100 {
            let (_, _streams_moved, words_routed) =
                self.device.array.step_data_movement(&mut self.host_memory);
            total_flush_words += words_routed;
            flush_iters += 1;

            // Check for fatal errors from data movement during flush too
            let fatal_errors = self.device.array.drain_fatal_errors();
            if !fatal_errors.is_empty() {
                for err in &fatal_errors {
                    log::error!("Data movement fatal (flush): {}", err);
                }
                self.status = EngineStatus::Error;
                return;
            }

            // Dispatch any ctrl packet actions produced during flush routing
            {
                let ctrl_actions = self.device.array.drain_ctrl_packet_actions();
                for action in ctrl_actions {
                    self.dispatch_ctrl_action(action);
                }
                self.drain_core_enables();
            }

            // Keep routing as long as trace data is anywhere in the pipeline:
            // pending in trace units, in stream switch slave/master FIFOs, or
            // awaiting inter-tile propagation.
            let trace_in_flight = self.device.array.tiles.iter().any(|t| {
                t.core_trace.has_pending_words()
                    || t.mem_trace.has_pending_words()
                    || t.stream_switch.has_pending_packet()
                    || t.stream_switch.has_pending_data()
            });
            if !trace_in_flight && words_routed == 0 {
                break;
            }
        }
        if total_flush_words > 0 {
            log::info!("Post-flush routing: {} words in {} iterations", total_flush_words, flush_iters);
        }
    }

    /// Build the SP-4b `origin_d.json` sidecar: per-module broadcast
    /// timer-reset arrival (`origin_D`) for the single channel-15 flood
    /// source observed this run, plus the `calibrated` flag that gates the
    /// Python inference engine's `causal_offset` decomposition (design doc
    /// `docs/superpowers/specs/2026-06-30-sp4b-skew-export-design.md` Sec.4).
    ///
    /// Contract:
    /// ```json
    /// { "calibrated": bool, "flood_source": "col|row" | null,
    ///   "modules": { "col|row|<module_kind>": origin_D, ... } }
    /// ```
    ///
    /// Single-source only: `DeviceState::flood_sources` records every
    /// distinct `(col, row)` that fired a channel-15 (timer-reset) broadcast
    /// this run. If that set's size isn't exactly 1, `flood_source` is
    /// `null` and `modules` is left empty -- the engine has no way to
    /// attribute a coherent `T0` to a multi-source or zero-source run, and
    /// must fail loud downstream rather than trust an ambiguous table
    /// (design Sec.4d). Module kinds (`core`/`mem`/`memtile`/`shim`) stay
    /// semantic; the Python loader owns the `module -> pkt_type` translation
    /// (design Sec.4b), so this never emits numeric packet-type codes.
    ///
    /// The timing constants come from `DeviceState::effective_broadcast_timing`
    /// -- the runtime override if set (SP-5b measurement apparatus), else the
    /// same build-time-generated `xdna_archspec::aie2::timing` constants
    /// `DeviceState::propagate_broadcasts` reads when it actually computes
    /// `origin_D` during execution -- so the export reproduces the exact
    /// constants the run used, not an independently-resolved copy.
    pub fn export_origin_d_sidecar(&self) -> serde_json::Value {
        let (d_h, d_v, core_off, mem_off, calibrated) = self.device.effective_broadcast_timing();

        let sources = self.device.flood_sources();
        let single = if sources.len() == 1 {
            sources.iter().next().copied()
        } else {
            None
        };

        let mut modules = serde_json::Map::new();
        if let Some((col, row)) = single {
            for (c, r, kind, d) in self.device.origin_d_table(
                col, row, 15, // BROADCAST channel 15 = timer-reset; matches flood_sources().
                d_h, d_v, core_off, mem_off,
            ) {
                modules.insert(format!("{c}|{r}|{kind}"), serde_json::json!(d));
            }
        }

        serde_json::json!({
            "calibrated": calibrated,
            "flood_source": single.map(|(c, r)| format!("{c}|{r}")),
            "modules": modules,
        })
    }

    /// Detect a control-packet ordering hazard at an NPU instruction boundary.
    ///
    /// Replaces the former `flush_ctrl_packets`, which fast-forwarded the stream
    /// router up to 8x at a frozen simulation cycle to deliver in-flight control
    /// packets before the next instruction read them. That flush was vestigial
    /// under the active `provisional_npu1` firmware-latency model:
    ///   - executor config writes (`execute_write32`) are immediate direct
    ///     register writes, not fabric-routed -- they never needed a delivery
    ///     window;
    ///   - the normal per-cycle `step()` path already delivers and dispatches
    ///     control packets every cycle (Phase 3);
    ///   - every fabric control packet gets its physical 1-hop/cycle delivery
    ///     across the emitting instruction's >=100-cycle retirement window.
    /// Its only measurable effect was a parasitic end-of-stream burst that
    /// over-delivered circuit-switched producer data (the tenant-4 tail-collapse),
    /// because each frozen-cycle routing pass also advanced data streams.
    ///
    /// Removing it relies on the firmware-latency model for correct ordering.
    /// To make that reliance falsifiable at runtime, this records a hazard
    /// whenever a packet-switched control packet is STILL in flight at an
    /// instruction boundary -- the exact precondition under which the old flush
    /// would have force-delivered. For the executor-direct-config majority of
    /// the corpus (tenant-4 included) this is never true. A non-zero count flags
    /// a control-packet kernel to scrutinize: if its bridge output is still
    /// correct the flush was unnecessary there too; if not, the fallback is a
    /// packet-only flush (restrict routing to packet-switched traffic).
    pub fn note_ctrl_packet_ordering_hazard(&mut self) {
        if self.device.array.has_pending_control_packet() {
            self.ctrl_packet_hazard_count += 1;
            log::warn!(
                "control-packet ordering hazard at cycle {}: a packet-switched control \
                 packet is still in flight at an NPU instruction boundary -- the removed \
                 flush_ctrl_packets would have force-delivered here. Verify control-packet \
                 ordering for this kernel.",
                self.total_cycles
            );
        }
    }

    /// Number of control-packet ordering hazards observed this run (see
    /// `note_ctrl_packet_ordering_hazard`). 0 = the deleted flush was never
    /// load-bearing on this workload.
    pub fn ctrl_packet_hazard_count(&self) -> u64 {
        self.ctrl_packet_hazard_count
    }

    /// Get reference to device state.
    pub fn device(&self) -> &DeviceState {
        &self.device
    }

    /// Get mutable reference to device state.
    pub fn device_mut(&mut self) -> &mut DeviceState {
        &mut self.device
    }

    /// Get reference to host memory.
    pub fn host_memory(&self) -> &HostMemory {
        &self.host_memory
    }

    /// Get mutable reference to host memory.
    pub fn host_memory_mut(&mut self) -> &mut HostMemory {
        &mut self.host_memory
    }

    /// Get mutable references to both device state and host memory.
    ///
    /// Needed when NPU instruction execution must interleave DMA stepping
    /// (backpressure on full task queues requires stepping DMA with host memory).
    pub fn device_and_host_memory(&mut self) -> (&mut DeviceState, &mut HostMemory) {
        (&mut self.device, &mut self.host_memory)
    }

    /// Ungate every column and module so a directly-constructed engine
    /// behaves as if a CDO had programmed the clock-control registers.
    ///
    /// On real silicon and in production runs, the CDO writes
    /// `Column_Clock_Control` / `Module_Clock_Control` to bring tiles
    /// out of reset before kernels execute.  Tests that construct an
    /// `InterpreterEngine` directly (no xclbin/CDO) need this helper to
    /// reach the same starting state without introducing a test-only
    /// backdoor on the production path.
    #[cfg(test)]
    pub(crate) fn ungate_all_for_test(&mut self) {
        self.device.array.clock_mut().ungate_all();
    }

    /// Enable a core at (col, row).
    pub fn enable_core(&mut self, col: usize, row: usize) {
        if let Some(core) = self.get_core_mut(col, row) {
            core.enabled = true;
        }
        // Mirror to CoreDebugState via the same register semantics as a
        // CDO Core_Control=0x1 write (sets enabled, clears reset) so the
        // runtime enable path cannot diverge from the CDO write path
        // (Core_Status RESET-bit fidelity, §8 close-out 2026-05-19).
        if let Some(tile) = self.device.tile_mut(col, row) {
            tile.core_debug.enable();
        }
    }

    /// Disable a core at (col, row).
    pub fn disable_core(&mut self, col: usize, row: usize) {
        if let Some(core) = self.get_core_mut(col, row) {
            core.enabled = false;
        }
        if let Some(tile) = self.device.tile_mut(col, row) {
            tile.core_debug.set_enabled(false);
        }
    }

    /// Check if a core is enabled.
    pub fn is_core_enabled(&self, col: usize, row: usize) -> bool {
        self.get_core(col, row).map(|c| c.enabled).unwrap_or(false)
    }

    /// Drain pending core enable/disable events from device state.
    ///
    /// Core_Control register writes (from NPU instructions, CDO, or control
    /// packets) push events to `device.pending_core_enables`. This syncs
    /// those to the engine's internal core state, matching how real hardware
    /// immediately reacts to the register write.
    fn drain_core_enables(&mut self) {
        let events: Vec<_> = self.device.pending_core_enables.drain(..).collect();
        for (col, row, enabled) in events {
            if enabled {
                self.enable_core(col as usize, row as usize);
            } else {
                self.disable_core(col as usize, row as usize);
            }
        }
    }

    /// Get core status at (col, row).
    pub fn core_status(&self, col: usize, row: usize) -> Option<CoreStatus> {
        self.get_core(col, row).map(|c| c.interpreter.status())
    }

    /// Get core execution context at (col, row).
    pub fn core_context(&self, col: usize, row: usize) -> Option<&ExecutionContext> {
        self.get_core(col, row).map(|c| &c.context)
    }

    /// Get mutable core execution context at (col, row).
    pub fn core_context_mut(&mut self, col: usize, row: usize) -> Option<&mut ExecutionContext> {
        self.get_core_mut(col, row).map(|c| &mut c.context)
    }

    /// Set the program counter for a core.
    pub fn set_core_pc(&mut self, col: usize, row: usize, pc: u32) {
        if let Some(core) = self.get_core_mut(col, row) {
            core.context.set_pc(pc);
        }
    }

    /// Set a pointer register for a core.
    ///
    /// AIE2 cores have 8 pointer registers (p0-p7) used for address generation.
    /// This method allows pre-initializing buffer base addresses before kernel
    /// execution, as kernels expect the runtime/CDO to configure these pointers.
    pub fn set_core_pointer(&mut self, col: usize, row: usize, reg: u8, value: u32) {
        if let Some(core) = self.get_core_mut(col, row) {
            core.context.pointer.write(reg, value);
        }
    }

    /// Set a modifier register for a core.
    ///
    /// AIE2 cores have 8 modifier registers (m0-m7) used for addressing modes.
    /// These are typically set to stride values for loop-based memory access.
    pub fn set_core_modifier(&mut self, col: usize, row: usize, reg: u8, value: u32) {
        if let Some(core) = self.get_core_mut(col, row) {
            core.context.modifier.write(reg, value);
        }
    }

    /// Get the last decoded bundle for a core (for debugging).
    ///
    /// Returns the last VLIW bundle that was decoded for the core at (col, row).
    /// Useful for diagnosing decode/execution errors.
    pub fn core_last_bundle(&self, col: usize, row: usize) -> Option<&VliwBundle> {
        self.get_core(col, row).and_then(|c| c.interpreter.last_bundle())
    }

    /// Load an ELF file into a core's memory.
    ///
    /// Parses the ELF, loads program and data segments into the tile's memory,
    /// sets the entry point as the PC, and enables the core.
    ///
    /// Returns the entry point address on success.
    pub fn load_elf_bytes(&mut self, col: usize, row: usize, data: &[u8]) -> Result<u32, String> {
        let elf = AieElf::parse(data).map_err(|e| format!("ELF parse error: {}", e))?;
        let entry = elf.entry_point();

        // Load segments into tile memory
        {
            let tile = self
                .device
                .tile_mut(col, row)
                .ok_or_else(|| format!("Invalid tile coordinates ({}, {})", col, row))?;
            elf.load_into(tile);
        }

        // Set entry point and enable core
        self.set_core_pc(col, row, entry);
        self.enable_core(col, row);

        Ok(entry)
    }

    /// Execute one cycle on all enabled cores and DMA engines.
    ///
    /// The cycle is a REQUEST -> ARBITRATE -> COMMIT loop, because the tile's
    /// data memory is eight single-port physical banks with a round-robin
    /// arbiter in front of each (AM020 ch.2:166) -- who gets to touch a bank
    /// has to be decided BEFORE anyone touches it:
    ///
    ///   Phase 1: sync DMA start requests from tiles to DMA engines.
    ///   Phase A: each DMA channel declares the banks it intends to touch this
    ///            cycle (`DmaEngine::peek_bank_demand`). No transfer.
    ///   Phase B: each core that can issue declares its next bundle's bank
    ///            demand (`CoreInterpreter::peek_bank_demand`). A core stalled
    ///            on a lock/DMA/stream declares nothing. No commit.
    ///   Phase C: per compute tile, per physical bank, round-robin arbitration
    ///            (`BankArbiter::arbitrate`).
    ///   Phase D: commit the winners, withhold the losers --
    ///              core lost -> `stall_for_bank` (1 cycle, PC unchanged, the
    ///                           SAME bundle retries next cycle) and its
    ///                           `step` is skipped ENTIRELY (Phase 2 below);
    ///              core won  -> step the core normally;
    ///              DMA lost  -> the channel's FSM step is skipped, so it
    ///                           re-presents the identical demand next cycle;
    ///              DMA won   -> transfers (Phase 3).
    ///   Phase E: emit MEMORY_STALL (core loss), CONFLICT_DM_BANK_n (any
    ///            contended bank), and DMA_x_MEMORY_BACKPRESSURE/STARVATION
    ///            (DMA loss) for the cycle they actually happened in.
    ///
    /// The arbitration pass runs between Phase 1 and Phase 2/3, and NOTHING
    /// between it and the DMA step mutates a stream FIFO (see
    /// `TileArray::step_data_movement_with_denied`) -- stream routing runs
    /// strictly after the DMA step. That ordering is what makes the peeked
    /// demand and the committed access provably the same set of banks.
    ///
    /// DMA still steps before cores for lock visibility: when a core copies
    /// MemTile locks for memory module access (lock IDs 48-63), it sees the
    /// locks DMA just released. This is critical for producer/consumer sync.
    pub fn step(&mut self) {
        if matches!(self.status, EngineStatus::Halted | EngineStatus::Error) {
            return;
        }

        self.status = EngineStatus::Running;
        let mut any_running = false;
        let mut all_halted = true;

        // Reset per-core active flag. Set to true only for StepResult::Continue
        // in Phase 2; the perf counter tick in Phase 3e reads it to pick
        // tick_active_cycles vs tick_idle_cycles for ACTIVE_CORE-gated counters.
        for core in &mut self.cores {
            core.active_this_cycle = false;
        }

        // Publish the current simulation cycle to the tile array so that
        // register-write side effects (trace unit start/stop, broadcast
        // propagation) can time-stamp events with the right cycle. Also
        // propagates to DMA engines for DMA-side trace events.
        self.device.array.set_dma_cycle(self.total_cycles);

        // Phase 1: Sync DMA start requests from tiles to DMA engines
        self.sync_dma_start_requests();

        // Phases A/B/C: request -> arbitrate. Both agents declare the banks
        // they intend to touch this cycle; the per-physical-bank round-robin
        // arbiters decide who gets them. Nothing commits here EXCEPT the
        // losing core's stall, which is the whole point: a core that loses is
        // stalled IN this cycle and its bundle is not executed below.
        let arbitration = self.arbitrate_memory_banks();

        // Phase 2: Step each enabled core.
        //
        // Core lock releases are DEFERRED: they write to core_lock_deltas
        // instead of modifying lock.value directly. This models the hardware
        // lock arbiter pipeline -- on real AIE2, the arbiter handles one
        // request per clock cycle (AM020), so a core release at cycle N
        // becomes visible to DMA at cycle N+1.
        //
        // The deferred requests are resolved by the arbiter during Phase 3,
        // so the NEXT cycle's DMA operations will see the updated values.
        //
        // Core lock acquires still read lock.value directly (committed state
        // from previous cycles). This is correct: the core stalls if the
        // precondition isn't met, and the acquire decrement is immediate.
        //
        // DMA-to-core lock visibility (the reverse direction) is handled by
        // step_data_movement's internal lock snapshot/commit, which commits
        // DMA lock releases at the end of the data movement phase. Cores
        // read committed lock state, so DMA releases from cycle N-1 are
        // visible to cores at cycle N -- the same 1-cycle latency as the
        // hardware's lock pipeline.
        for col in 0..self.cols {
            for row in self.compute_row_start..self.rows {
                let idx = col * self.rows + row;

                if !self.cores[idx].enabled {
                    continue;
                }

                if !self.device.array.clock().is_column_active(col as u8)
                    || !self
                        .device
                        .array
                        .clock()
                        .is_module_active(col as u8, row as u8, ModuleKind::Core)
                {
                    continue;
                }

                // Phase D (core): this core lost a bank it needs, so its
                // bundle does NOT commit this cycle. `stall_for_bank` already
                // charged the one-cycle stall during arbitration; calling
                // `step` as well would clear WaitBank, execute the bundle
                // anyway, and double-charge the tick -- so the two are
                // strictly exclusive, enforced by this `continue`. The PC is
                // untouched, so the SAME bundle re-arbitrates next cycle
                // (AM020 ch.2:166 retry contract).
                if arbitration[self.device.array.tile_index(col as u8, row as u8)].core_lost {
                    // Mirror the stall into Core_Status the same way every
                    // other stall class does (task 6 review, Important-1): a
                    // WaitBank cycle IS a memory-bank access that lost
                    // arbitration, i.e. exactly the AM025 Memory_Stall_*
                    // condition, so it gets the same `mem` bit as a real
                    // memory stall. Without this, Core_Status read while
                    // bank-stalled would show a running core, and a
                    // Debug_Control2 MEM_STALL_HALT enable would never fire
                    // on a lost bank arbitration. `update_stalls` also
                    // re-evaluates the stall-halt latch, so that enable now
                    // takes effect here exactly as it does for the other
                    // stall classes below.
                    if let Some(tile) = self.device.tile_mut(col, row) {
                        tile.core_debug.update_stalls(true, false, false, false);
                    }
                    all_halted = false;
                    any_running = true;
                    continue;
                }

                // AIE2 lock quadrant routing: copy neighbor locks before stepping,
                // then write changes back after. Per mlir-aie getLockLocalBaseIndex:
                //   South (IDs 0-15):  row-1 neighbor
                //   West  (IDs 16-31): col-1 neighbor
                //   North (IDs 32-47): row+1 neighbor
                //   East  (IDs 48-63): own tile (handled internally)
                let mut south_locks: Option<Vec<crate::device::tile::Lock>> = None;
                let mut west_locks: Option<Vec<crate::device::tile::Lock>> = None;
                let mut north_locks: Option<Vec<crate::device::tile::Lock>> = None;

                if row > SHIM_ROW as usize {
                    if let Some(south) = self.device.tile(col, row - 1) {
                        south_locks = Some(south.locks.clone());
                    }
                }
                if col > 0 {
                    if let Some(west) = self.device.tile(col - 1, row) {
                        west_locks = Some(west.locks.clone());
                    }
                }
                if row + 1 < self.rows {
                    if let Some(north) = self.device.tile(col, row + 1) {
                        north_locks = Some(north.locks.clone());
                    }
                }

                // Split the device so the executing tile and a read-through
                // view of all other tiles come from one borrow. The view
                // is threaded into `step_with_neighbor_locks` so the read
                // sites can `ensure_snapshot` lazily -- only the quadrants
                // the kernel actually accesses get snapshotted, and only
                // on the first read. Both `tile` and `view` drop at the
                // end of this body, freeing `&mut self.device` for
                // `drain_writes` below.
                let Some((tile, view)) = self.device.split_tile_mut(col, row) else {
                    continue;
                };
                if !tile.is_compute() {
                    continue;
                }
                let core = &mut self.cores[idx];

                // Tile_Control isolation: refresh the cached byte on
                // NeighborMemory and gate cross-tile lock slices before
                // they reach the executor. A set bit in tile.isolation
                // blocks every cross-tile transit on that direction.
                // Per-call sites in execute/memory then short-circuit
                // via NeighborMemory's own `is_blocked`. See
                // `crate::device::tile::isolation` for the bit layout.
                core.neighbors.set_isolation(tile.isolation);
                let mut nlocks = build_neighbor_locks_with_isolation(
                    tile.isolation,
                    south_locks.as_mut().map(|v| v.as_mut_slice()),
                    west_locks.as_mut().map(|v| v.as_mut_slice()),
                    north_locks.as_mut().map(|v| v.as_mut_slice()),
                );

                // Pre-execute synchronous PC_Event seam (Phase B Unit 1, spec §5.1).
                //
                // G1 silicon observation (NPU1, 2026-05-18): a synchronous
                // PC-event breakpoint halts BEFORE the trap bundle commits.
                // If the next PC matches an armed PC_Event with PC_Event_Halt,
                // halt the core here WITHOUT executing the bundle. The trap
                // store never lands (before-commit).
                //
                // Async halt paths (host Debug_Control0[0], stall-halt,
                // event-halt) are NOT intercepted by this seam, because
                // has_sync_pc_trap_at() returns false for them:
                // pc_event_halt_enabled() (Debug_Control2[0]) is clear for an
                // async halt. They instead set core_debug.halted = true and
                // are caught by the is_halted() gate *inside*
                // step_with_neighbor_locks (interpreter.rs ~181) -- that gate
                // is reached only after this seam is skipped, not "before
                // step_with_neighbor_locks". The seam is purely additive: it
                // does not change the async-halt code path.
                //
                // Known dual-mechanism interaction (M1, deliberate, NOT
                // unified in this unit): this pre-execute seam is the new
                // before-commit PC_Event path, but the pre-existing
                // post-execute update_pc() -> check_pc_events() PC_Event match
                // (further down this block) still co-fires -- e.g. when the
                // program loops back to TRAP_PC, the post-execute matcher also
                // sees it. The observable result is still correct
                // (before-commit is preserved because this seam runs first and
                // skips the bundle), and the sync_trap_consumed_at latch keeps
                // the two from double-counting across a resume. Unifying to a
                // seam-only PC_Event path is a deliberate future follow-up
                // (G2/spec §5.2), explicitly out of Phase B Unit 1 scope.
                //
                // sync_trap_consumed_at lifecycle: consume_sync_pc_trap() sets
                // it to TRAP_PC when the seam fires; has_sync_pc_trap_at()
                // then returns false while the core is halted at TRAP_PC (so
                // intermediate halted ticks and the first post-resume tick do
                // not re-fire the seam). It is cleared ONLY on
                // StepResult::Continue (the trap bundle retired and PC moved
                // off TRAP_PC) -- never on DebugHalt/WaitLock, which can leave
                // PC pinned at TRAP_PC with the bundle un-retired. Clearing it
                // on those would re-arm the seam and swallow the resume
                // (review S1).
                let next_pc = core.context.pc();
                if tile.core_debug.has_sync_pc_trap_at(next_pc) {
                    tile.core_debug.consume_sync_pc_trap(next_pc);
                    // Core is now debug-halted; the bundle did not execute.
                    // Skip the rest of this tile's step processing.
                    all_halted = false;
                    any_running = true;
                    continue;
                }

                // §5.1 principled split (Maya 2026-05-19): PC-wired event
                // single-step also halts before-commit (arming = PC match,
                // known pre-bundle). Same seam, same skip-the-bundle semantics
                // as the G1 PC_Event_Halt path above.
                if tile.core_debug.has_sync_sstep_pc_trap_at(next_pc) {
                    tile.core_debug.consume_sync_sstep_pc_trap(next_pc);
                    all_halted = false;
                    any_running = true;
                    continue;
                }

                let result = core.interpreter.step_with_neighbor_locks(
                    &mut core.context,
                    tile,
                    &mut nlocks,
                    Some(&mut core.neighbors),
                    Some(&view),
                );

                // NOTE: clear_sync_trap_consumed() is intentionally NOT called
                // here. The latch must only be cleared once the trap bundle
                // has actually retired and PC has moved off TRAP_PC -- that is
                // exactly StepResult::Continue (see the Continue arm below).
                //
                // Clearing it unconditionally here was a resume-swallow bug
                // (review S1): between the pre-execute halt and the host
                // resume, the engine keeps ticking; the still-halted core
                // falls through to step_with_neighbor_locks, the
                // interpreter.rs is_halted() gate returns DebugHalt without
                // executing the bundle, and an unconditional clear here would
                // drop the latch while PC is still pinned at TRAP_PC. The next
                // tick (now resumed) would then re-fire the seam on the same
                // bundle, swallowing the resume. The same pin-at-TRAP_PC
                // hazard applies to WaitLock if the trap bundle itself stalls
                // on a lock acquire. Only Continue guarantees the bundle ran.

                // Update CoreDebugState with current PC and stall info.
                // This mirrors the interpreter state into the register
                // space so host reads of Core_Status/Core_PC return
                // correct values (matching real hardware behavior).
                // update_pc also drives PC_Event0..3 matching, which can
                // set the pending single-step latch through check_event_halt.
                tile.core_debug.update_pc(core.context.pc());
                // Drain the SSTEP_EVENT latch: if any event during this
                // step matched Debug_Control1.SSTEP_EVENT, request halt
                // now (interpretation (a): the triggering bundle is the
                // last to commit before halt).
                tile.core_debug.consume_pending_single_step();

                // Mirror the live PC onto tile.core.pc as well, so any
                // path that reads tile-side state (broadcast
                // propagation, register effects) sees the running PC
                // instead of the stale CDO-loaded entry value. Without
                // this sync, mode-2's Start anchor_pc reads 0
                // regardless of what the core was actually doing when
                // start_event fired.
                tile.core.pc = core.context.pc();
                match result {
                    StepResult::Continue => {
                        // §5.2 count-step: a bundle COMMITTED this tick.
                        // Decrement the live budget (only Continue means a
                        // bundle retired -- WaitLock/DebugHalt/etc. did not,
                        // and must not consume the single-step budget; spec
                        // §5.2 "per-committed-bundle"). Expiry latches halted;
                        // the is_halted gate then blocks bundle N+1
                        // (before-commit of N+1, G2-derived).
                        tile.core_debug.tick_count_step();
                        // The trap bundle (if any) has now retired and PC has
                        // moved off TRAP_PC. Clear the sync-trap-consumed
                        // latch so the next time the core returns to that PC
                        // (e.g. a loop back to TRAP_PC) the seam fires fresh.
                        // This is the ONLY place the latch is cleared (review
                        // S1): DebugHalt/WaitLock can leave PC pinned at
                        // TRAP_PC with the bundle un-retired, so clearing
                        // there would re-arm the seam and swallow the resume.
                        tile.core_debug.clear_sync_trap_consumed();
                        tile.core_debug.update_stalls(false, false, false, false);
                        all_halted = false;
                        any_running = true;
                        self.total_instructions += 1;
                        core.active_this_cycle = true;
                    }
                    StepResult::WaitLock { .. } => {
                        tile.core_debug.update_stalls(false, true, false, false);
                        all_halted = false;
                        any_running = true;
                        // Pipeline-PC adjustment: HW's fetch+decode is
                        // several stages ahead of execute. When ACQ stalls
                        // in execute, the fetch PC pins at the start of
                        // the last delay slot of the next branch (RET in
                        // the acquire stub). Override tile.core.pc so the
                        // trace unit sees the same PC HW would, instead
                        // of the execute PC where ACQ is parked.
                        let stall_pc = core.context.pc();
                        if let Some(prog_mem) = tile.program_memory() {
                            let pipeline_pc = core.interpreter.lock_stall_pipeline_pc(prog_mem, stall_pc);
                            tile.core.pc = pipeline_pc;
                            tile.core_debug.update_pc(pipeline_pc);
                        }
                    }
                    StepResult::WaitDma { .. } => {
                        tile.core_debug.update_stalls(true, false, false, false);
                        all_halted = false;
                        any_running = true;
                    }
                    StepResult::WaitStream { .. } => {
                        tile.core_debug.update_stalls(false, false, true, false);
                        all_halted = false;
                        any_running = true;
                    }
                    StepResult::Halt => {
                        tile.core_debug.set_done(true);
                        tile.core_debug.update_stalls(false, false, false, false);
                    }
                    StepResult::DebugHalt => {
                        // Debug halt is a transient pause -- the program isn't
                        // done. Don't call set_done; just clear stall flags
                        // (the previous instruction has already retired) and
                        // mark the engine as still active. Leaving all_halted
                        // false keeps the engine ticking so DMAs continue and
                        // the host can write Debug_Control0=0 to resume.
                        tile.core_debug.update_stalls(false, false, false, false);
                        all_halted = false;
                    }
                    StepResult::DecodeError(ref e) => {
                        log::error!(
                            "Core({},{}) DecodeError at cycle {}: {:?}",
                            col,
                            row,
                            self.total_cycles,
                            e
                        );
                        self.status = EngineStatus::Error;
                        return;
                    }
                    StepResult::ExecError(ref e) => {
                        log::error!(
                            "Core({},{}) ExecError at cycle {}: {:?}",
                            col,
                            row,
                            self.total_cycles,
                            e
                        );
                        self.status = EngineStatus::Error;
                        return;
                    }
                }

                // Notify core trace unit with any new events since last cycle.
                //
                // Use the coordinator's global cycle (self.total_cycles) rather
                // than the per-core `evt.cycle`. The event log stores the
                // core-local retire counter (ctx.cycles), which freezes during
                // stalls and thus does not reflect the tile clock that real HW
                // trace units observe. Events drained here happened in the
                // current simulation step, so total_cycles is the right stamp.
                let event_log = &core.context.timing_context().events;
                let total = event_log.total_recorded();
                let new_start = core.trace_events_consumed;
                if new_start < total {
                    let cycle = self.total_cycles;
                    for evt in event_log.since(new_start) {
                        // Level-edge events (e.g. LockStallLevel) route to the
                        // trace unit's held-level path; everything else is a
                        // one-cycle pulse via notify_event.
                        if let Some((hw_id, active)) = crate::trace::core_level_edge(&evt.event) {
                            tile.notify_core_trace_level(hw_id, cycle, active);
                        } else if let Some(hw_id) = crate::trace::core_event_to_hw_id(&evt.event) {
                            let pc = crate::trace::event_pc(&evt.event);
                            tile.notify_core_trace_event(hw_id, cycle, pc);
                        }
                    }
                    core.trace_events_consumed = total;
                }

                // Drain buffered cross-tile writes to neighbor tiles.
                // Snapshots stay live; only the writes vec is drained.
                // NLL: `tile` and `view` are no longer in use after this
                // point, so `&mut self.device` is free to be re-borrowed.
                if core.neighbors.has_pending_writes() {
                    core.neighbors.drain_writes(&mut self.device);
                }

                // Submit modified neighbor locks back to neighbor tiles.
                //
                // Instead of writing back directly (which would make core lock
                // releases visible to neighbor DMA in the same cycle), we compute
                // the diff and submit it as a core release to the arbiter. The
                // arbiter resolves in Phase 3, providing the 1-cycle delay.
                fn writeback_locks(
                    device: &mut DeviceState,
                    locks: Option<Vec<crate::device::tile::Lock>>,
                    ncol: usize,
                    nrow: usize,
                ) {
                    if let Some(modified) = locks {
                        if let Some(neighbor) = device.tile_mut(ncol, nrow) {
                            for i in 0..modified.len().min(neighbor.locks.len()) {
                                let delta = (modified[i].value as i16) - (neighbor.locks[i].value as i16);
                                if delta != 0 {
                                    neighbor.defer_core_lock_release(i, delta as i8);
                                }
                            }
                        }
                    }
                }
                // South
                if row > SHIM_ROW as usize {
                    writeback_locks(&mut self.device, south_locks, col, row - 1);
                }
                // West
                if col > 0 {
                    writeback_locks(&mut self.device, west_locks, col - 1, row);
                }
                // North
                if row + 1 < self.rows {
                    writeback_locks(&mut self.device, north_locks, col, row + 1);
                }
            }
        }

        // Phase 3 / Phase D (DMA): Step all DMA engines and stream routing.
        //
        // Core lock releases from Phase 2 are pending in tile arbiters.
        // step_data_movement() does: submit DMA lock requests -> resolve
        // all tile arbiters (core + DMA together) -> step DMA channels
        // checking arbiter results -> route streams.
        //
        // Channels that lost this cycle's BANK arbitration are held: their FSM
        // step is skipped entirely, so they re-present the identical demand
        // next cycle. Note that nothing between the Phase-A peek and this call
        // touches a stream FIFO -- stream routing happens inside, strictly
        // after the DMA step -- so a channel can never transfer on data that
        // arrived after it declared its demand.
        //
        // Set cycle timestamp on DMA engines before stepping so trace
        // events get the correct cycle number.
        self.device.array.set_dma_cycle(self.total_cycles);

        // Borrowed slices, not clones (task 6 review, Minor-4): the arbiter's
        // per-tile `denied_dma` already owns its Vec for this cycle, so a
        // view over it is enough -- no need to allocate and copy a second
        // Vec<Requester> per tile every cycle.
        let denied: Vec<&[Requester]> = arbitration.iter().map(|a| a.denied_dma.as_slice()).collect();
        let (dma_active, streams_moved, _words_routed) =
            self.device.array.step_data_movement_with_denied(&mut self.host_memory, &denied);

        // Check for fatal errors from data movement (impossible-on-hardware
        // conditions like missing packet routes or stream buffer overflows).
        let fatal_errors = self.device.array.drain_fatal_errors();
        if !fatal_errors.is_empty() {
            for err in &fatal_errors {
                log::error!("Data movement fatal: {}", err);
            }
            self.status = EngineStatus::Error;
            return;
        }

        // Dispatch control packet actions through DeviceState for full
        // module dispatch (MemTile DMA BDs, stream switch, etc.).
        let ctrl_actions = self.device.array.drain_ctrl_packet_actions();
        for action in ctrl_actions {
            self.dispatch_ctrl_action(action);
        }
        self.drain_core_enables();

        // Drain memory-module trace events (DMA + locks) into the global
        // trace log and notify trace units so they can produce binary trace
        // packets. Each tile type uses a different event ID namespace:
        // - Compute tiles: mem_event_to_hw_id -> mem_trace
        // - MemTiles: memtile_event_to_hw_id -> mem_trace
        // - Shim tiles: shim_event_to_hw_id -> core_trace (PL module)
        for (col, row, cycle, event) in self.device.array.drain_mem_trace_events() {
            if let Some(tile) = self.device.array.get_mut(col, row) {
                // DMA stall/starvation are held levels (assert/deassert edges);
                // start/finished events are one-cycle pulses. `dma_level_active`
                // returns the edge polarity for the former, None for the latter.
                let level = crate::trace::dma_level_active(&event);
                if tile.is_shim() {
                    if let Some(id) = crate::trace::shim_event_to_hw_id(&event) {
                        // Shim DMA/lock events have no program counter.
                        match level {
                            Some(active) => tile.notify_core_trace_level(id, cycle, active),
                            None => tile.notify_core_trace_event(id, cycle, None),
                        }
                    }
                } else if tile.is_mem() {
                    // Memtile DMA events are gated by the DMA_Event_Channel_Selection
                    // register (0xA06A0). Each SEL slot can map to a different
                    // physical channel and a single channel may fire both SEL slots
                    // when both are aimed at it (e.g., reset default = both at ch0).
                    let sel =
                        crate::trace::MemtileDmaEventSel::from_register(tile.memtile_dma_event_chan_sel);
                    for id in crate::trace::memtile_event_to_hw_ids(&event, sel).into_iter().flatten() {
                        match level {
                            Some(active) => tile.notify_mem_trace_level(id, cycle, active),
                            None => tile.notify_mem_trace_event(id, cycle, None),
                        }
                    }
                } else if let Some(id) = crate::trace::mem_event_to_hw_id(&event) {
                    match level {
                        Some(active) => tile.notify_mem_trace_level(id, cycle, active),
                        None => tile.notify_mem_trace_event(id, cycle, None),
                    }
                }
            }
        }

        // Phase 3b: Generate stream switch port events on signal transitions.
        //
        // Four event types per monitored port, all edge-triggered on real
        // silicon (rising edge of the corresponding port signal):
        // - PORT_RUNNING: fires when port becomes active (idle -> active)
        // - PORT_IDLE: fires when port becomes idle (active -> idle)
        // - PORT_STALLED: fires when stall asserts (rising edge only)
        // - PORT_TLAST: fires when TLAST asserts (rising edge only)
        //
        // step_data_movement() seeds these flags from FIFO state at routing
        // start and updates them as data flows. We compare against
        // `tile.prev_port_state` from the previous cycle and only emit on
        // transitions, matching HW behavior. Emitting every active cycle
        // (the old behavior) floods small trace BDs in milliseconds.
        {
            let cycle = self.total_cycles;
            // Collect (tile_idx, hw_event_id, route, trace_target) tuples.
            // `route`: None = one-cycle pulse; Some(active) = held-level edge
            // (assert/deassert). PORT_RUNNING/IDLE/STALLED are level signals
            // (held while the port is in that state), so they emit both edges
            // through the skip-token held-level path; PORT_TLAST is a true
            // pulse. trace_target selects core_trace vs mem_trace.
            let mut port_events: Vec<(usize, u8, Option<bool>, TileKind)> = Vec::new();
            // Per-tile updates to prev_port_state, applied after collection.
            let mut prev_updates: Vec<(usize, u8, bool, bool, bool)> = Vec::new();
            for idx in 0..self.device.array.tiles.len() {
                let tile = &self.device.array.tiles[idx];
                if !tile.core_trace.is_configured() && !tile.mem_trace.is_configured() {
                    continue;
                }
                let tt = tile.tile_kind;

                let running_hw = |p| match tt {
                    TileKind::Compute => crate::trace::core_port_running_hw_id(p),
                    TileKind::Mem => crate::trace::memtile_port_running_hw_id(p),
                    TileKind::ShimNoc | TileKind::ShimPl => crate::trace::shim_port_running_hw_id(p),
                };
                let idle_hw = |p| match tt {
                    TileKind::Compute => crate::trace::core_port_idle_hw_id(p),
                    TileKind::Mem => crate::trace::memtile_port_idle_hw_id(p),
                    TileKind::ShimNoc | TileKind::ShimPl => crate::trace::shim_port_idle_hw_id(p),
                };
                let stalled_hw = |p| match tt {
                    TileKind::Compute => crate::trace::core_port_stalled_hw_id(p),
                    TileKind::Mem => crate::trace::memtile_port_stalled_hw_id(p),
                    TileKind::ShimNoc | TileKind::ShimPl => crate::trace::shim_port_stalled_hw_id(p),
                };
                let tlast_hw = |p| match tt {
                    TileKind::Compute => crate::trace::core_port_tlast_hw_id(p),
                    TileKind::Mem => crate::trace::memtile_port_tlast_hw_id(p),
                    TileKind::ShimNoc | TileKind::ShimPl => crate::trace::shim_port_tlast_hw_id(p),
                };

                for event_port in 0..8u8 {
                    if let Some((port_idx, is_master)) = tile.event_port_selection[event_port as usize] {
                        let port = if is_master {
                            tile.stream_switch.masters.get(port_idx as usize)
                        } else {
                            tile.stream_switch.slaves.get(port_idx as usize)
                        };
                        let Some(port) = port else { continue };

                        let (prev_active, prev_stalled, prev_tlast) =
                            tile.prev_port_state[event_port as usize];
                        // PORT_RUNNING reflects an actual beat crossing the
                        // port this cycle (`cycle_beat`), not buffered-data
                        // presence (`cycle_active`, which drives clock gating).
                        // A receive port holding residual FIFO data between
                        // upstream bursts is idle-with-data, not running -- HW
                        // toggles PORT_RUNNING per beat-run, so we follow the
                        // beat signal here.
                        let cur_active = port.cycle_beat;
                        let cur_stalled = port.cycle_stalled;
                        let cur_tlast = port.cycle_tlast;

                        // PORT_RUNNING and PORT_IDLE are complementary held
                        // levels: a port is either running or idle. On each
                        // active-edge, assert one and deassert the other. The
                        // trace unit records only the levels mapped to a slot,
                        // so emitting both is harmless when only one is traced.
                        if cur_active != prev_active {
                            // XFORM edge log (#140): the exact PORT_RUNNING level
                            // edges fed to the trace-unit encoder for the memtile.
                            // If a continuously-beating port emits just 1 assert +
                            // 1 deassert here but decodes to N>1 sub-bursts, the
                            // transform is in the encode/decode, not the signal.
                            if tt.is_mem() && std::env::var_os("XDNA_EMU_XFORM_PROBE").is_some() {
                                eprintln!(
                                    "[XEDGE] cyc={cycle} memtile ep{event_port} PORT_RUNNING={}",
                                    cur_active as u8
                                );
                            }
                            port_events.push((idx, running_hw(event_port), Some(cur_active), tt));
                            port_events.push((idx, idle_hw(event_port), Some(!cur_active), tt));
                        }

                        // PORT_STALLED is a held level (asserted while the port
                        // is stalled); emit both edges.
                        if cur_stalled != prev_stalled {
                            if tt.is_mem() && std::env::var_os("XDNA_EMU_XFORM_PROBE").is_some() {
                                eprintln!(
                                    "[XEDGE] cyc={cycle} memtile ep{event_port} PORT_STALLED={}",
                                    cur_stalled as u8
                                );
                            }
                            port_events.push((idx, stalled_hw(event_port), Some(cur_stalled), tt));
                        }

                        // PORT_TLAST is a genuine one-cycle pulse (end of
                        // packet), emitted on its rising edge only.
                        if cur_tlast && !prev_tlast {
                            port_events.push((idx, tlast_hw(event_port), None, tt));
                        }

                        prev_updates.push((idx, event_port, cur_active, cur_stalled, cur_tlast));
                    }
                }
            }
            for (idx, hw_id, route, tt) in port_events {
                let tile = &mut self.device.array.tiles[idx];
                // Compute tiles: core_trace (CoreEvent namespace)
                // Shim tiles: core_trace (PL module, single trace unit)
                // MemTiles: mem_trace (MemTileEvent namespace)
                // Port events have no PC -- they are I/O fabric events.
                match (tt.is_mem(), route) {
                    (true, Some(active)) => tile.notify_mem_trace_level(hw_id, cycle, active),
                    (true, None) => tile.notify_mem_trace_event(hw_id, cycle, None),
                    (false, Some(active)) => tile.notify_core_trace_level(hw_id, cycle, active),
                    (false, None) => tile.notify_core_trace_event(hw_id, cycle, None),
                }
            }
            for (idx, event_port, active, stalled, tlast) in prev_updates {
                self.device.array.tiles[idx].prev_port_state[event_port as usize] = (active, stalled, tlast);
            }
        }

        // XFORM_PROBE (#140 Move B): memtile (0,1) double-buffer transform
        // timeline. Default-off scaffolding (XDNA_EMU_XFORM_PROBE). Per cycle,
        // dump slot0/slot4 port beats + every active memtile DMA channel's
        // progress + the in0 objfifo lock values, so the buffer-fill -> re-emit
        // reshaping is directly observable. Remove when the Move-B fidelity
        // work lands.
        if std::env::var_os("XDNA_EMU_XFORM_PROBE").is_some() {
            let cyc = self.total_cycles;
            // Auto-detect the active memtile column: the FFI/plugin path places
            // the kernel at a different column than the in-process path (the
            // decoded trace shows it at col 1), so scan for the memtile (row 1)
            // whose DMA has a live transfer this run rather than hardcoding col 0.
            let probe_col = (0..self.cols as u8)
                .find(|&c| {
                    self.device
                        .array
                        .dma_engine(c, 1)
                        .map_or(false, |e| (0u8..12).any(|ch| e.get_transfer(ch).is_some()))
                })
                .unwrap_or(0);
            let mt = self.device.array.tile(probe_col, 1);
            let beat = |ep: usize| -> i32 {
                match mt.event_port_selection.get(ep).copied().flatten() {
                    Some((pi, true)) => {
                        mt.stream_switch.masters.get(pi as usize).map_or(-1, |p| p.cycle_beat as i32)
                    }
                    Some((pi, false)) => {
                        mt.stream_switch.slaves.get(pi as usize).map_or(-1, |p| p.cycle_beat as i32)
                    }
                    None => -1,
                }
            };
            // #140: stall readout parallel to beat -- to verify whether a slave
            // port asserts cycle_beat (DMA push/pop) and cycle_stalled
            // (can't-forward) in the SAME cycle, which HW treats as exclusive.
            let stall = |ep: usize| -> i32 {
                match mt.event_port_selection.get(ep).copied().flatten() {
                    Some((pi, true)) => {
                        mt.stream_switch.masters.get(pi as usize).map_or(-1, |p| p.cycle_stalled as i32)
                    }
                    Some((pi, false)) => {
                        mt.stream_switch.slaves.get(pi as usize).map_or(-1, |p| p.cycle_stalled as i32)
                    }
                    None => -1,
                }
            };
            let s0 = beat(0);
            let s4 = beat(4);
            let s4st = stall(4);
            let s0st = stall(0);
            let mut chans = String::new();
            if let Some(eng) = self.device.array.dma_engine(probe_col, 1) {
                for ch in 0u8..12 {
                    if let Some(t) = eng.get_transfer(ch) {
                        chans.push_str(&format!(" ch{}={}/{}", ch, t.bytes_transferred, t.total_bytes));
                    }
                }
            }
            let locks: Vec<i8> = mt.locks.iter().take(8).map(|l| l.value).collect();
            // Compute (0,2) consumer side: input-objfifo locks (prod=lock0,
            // cons=lock1), input S2MM fill, output MM2S drain, and whether the
            // core executed this cycle. The HW-faithful invariant is that the
            // core HOLDS cons_lock during its ~N-cycle add loop (prod_lock not
            // yet ++), so with both buffers full (prod_lock==0) the compute S2MM
            // must stall -- that stall is what backpressures memtile MM2S (slot4).
            let c2 = self.device.array.tile(probe_col, 2);
            let c2_locks: Vec<i8> = c2.locks.iter().take(4).map(|l| l.value).collect();
            let mut c2_chans = String::new();
            if let Some(eng) = self.device.array.dma_engine(probe_col, 2) {
                for ch in 0u8..4 {
                    if let Some(t) = eng.get_transfer(ch) {
                        c2_chans.push_str(&format!(" c2ch{}={}/{}", ch, t.bytes_transferred, t.total_bytes));
                    }
                }
            }
            let c2_idx = probe_col as usize * self.rows + 2;
            let c2_act = self.cores.get(c2_idx).map_or(false, |c| c.active_this_cycle);
            if s0 == 1 || s4 == 1 || !chans.is_empty() || !c2_chans.is_empty() {
                eprintln!(
                    "[XFORM] col={probe_col} cyc={cyc} s0={s0} s0st={s0st} s4={s4} s4st={s4st}{chans} mtlk={locks:?} | c2act={} c2lk={c2_locks:?}{c2_chans}",
                    c2_act as u8
                );
            }
        }

        // STAGE_PROBE (#140 send-cadence): per-cycle occupancy of EVERY buffer
        // along the memtile(row1)->compute(row2) send cascade, so we can see
        // exactly which stage holds the producer's words instead of
        // backpressuring when the consumer S2MM lock-stalls. Default-off
        // (XDNA_EMU_STAGE_PROBE). Stages, producer->consumer:
        //   mt_out (MM2S stream_out) -> mt slave/master port FIFOs
        //   -> inter-tile crossing (in-flight delay line)
        //   -> c2 slave/master port FIFOs -> c2_in (S2MM stream_in ingress, 16).
        if std::env::var_os("XDNA_EMU_STAGE_PROBE").is_some() {
            let cyc = self.total_cycles;
            let col = (0..self.cols as u8)
                .find(|&c| {
                    self.device
                        .array
                        .dma_engine(c, 1)
                        .map_or(false, |e| (0u8..12).any(|ch| e.get_transfer(ch).is_some()))
                })
                .unwrap_or(0);
            // Non-empty port FIFOs as "i:depth" for a tile's switch side.
            let ports = |row: u8, master: bool| -> String {
                let t = self.device.array.tile(col, row);
                let v = if master {
                    &t.stream_switch.masters
                } else {
                    &t.stream_switch.slaves
                };
                v.iter()
                    .enumerate()
                    .filter(|(_, p)| !p.fifo.is_empty())
                    .map(|(i, p)| format!("{i}:{}", p.fifo.len()))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let mt_out = self.device.array.dma_engine(col, 1).map_or(0, |e| e.stream_out_len());
            let (infl, infl_transit) = self.device.array.inflight_to_tile(col, 2);
            let c2_in = self.device.array.dma_engine(col, 2).map_or(0, |e| e.stream_in_len());
            let c2_in0 = self
                .device
                .array
                .dma_engine(col, 2)
                .map_or(0, |e| e.stream_in_count_for_channel(0));
            let c2_phase = self.device.array.dma_engine(col, 2).map_or("?", |e| e.channel_phase(0));
            let c2_locks: Vec<i8> =
                self.device.array.tile(col, 2).locks.iter().take(4).map(|l| l.value).collect();
            // Producer send-port (slot 4) beat/stall via the memtile's trace
            // event-port selection (same mapping the XFORM probe uses).
            let mt = self.device.array.tile(col, 1);
            let sel = |ep: usize, stalled: bool| -> i32 {
                match mt.event_port_selection.get(ep).copied().flatten() {
                    Some((pi, true)) => mt
                        .stream_switch
                        .masters
                        .get(pi as usize)
                        .map_or(-1, |p| (if stalled { p.cycle_stalled } else { p.cycle_beat }) as i32),
                    Some((pi, false)) => mt
                        .stream_switch
                        .slaves
                        .get(pi as usize)
                        .map_or(-1, |p| (if stalled { p.cycle_stalled } else { p.cycle_beat }) as i32),
                    None => -1,
                }
            };
            let (s4b, s4s) = (sel(4, false), sel(4, true));
            let mt_s = ports(1, false);
            let mt_m = ports(1, true);
            let c2_s = ports(2, false);
            let c2_m = ports(2, true);
            let active = mt_out > 0
                || infl > 0
                || c2_in > 0
                || !mt_s.is_empty()
                || !mt_m.is_empty()
                || !c2_s.is_empty()
                || !c2_m.is_empty();
            if active {
                eprintln!(
                    "[STAGE] cyc={cyc} col={col} | mt_out={mt_out} mtS[{mt_s}] mtM[{mt_m}] s4b={s4b} s4st={s4s} \
                     -> xtile(inflight={infl},transit={infl_transit}) -> c2S[{c2_s}] c2M[{c2_m}] \
                     c2_in={c2_in}(ch0={c2_in0}) c2phase={c2_phase} c2lk={c2_locks:?}"
                );
            }
        }

        if dma_active || streams_moved {
            any_running = true;
        }

        // Phase E: emit this cycle's bank-arbitration events.
        //
        // Both events come from the SAME arbitration that already decided who
        // commits (Phases A-C), so they land in the cycle the contention
        // actually happened -- not a cycle late, and not from a second,
        // independently-scoped conflict detector (the executor's old
        // commit-time detector is retired; see `record_memory_access`).
        //
        //   * CONFLICT_DM_BANK_N on the memory-module trace unit, once per
        //     bank that had more than one requester this cycle.
        //   * MEMORY_STALL on the core-module trace unit, as a held LEVEL:
        //     `mem_stall_edge` turns the per-cycle "core lost" boolean into
        //     rising/falling edges, so a sustained stall renders as one B..E
        //     span instead of one pulse per cycle (HW samples MEMORY_STALL as
        //     a level signal). Every compute core is visited, not just the
        //     losers, so a core whose stall just cleared fires its falling edge.
        {
            let cycle = self.total_cycles;
            for col in 0..self.cols {
                for row in self.compute_row_start..self.rows {
                    let core_idx = col * self.rows + row;
                    let tile_idx = self.device.array.tile_index(col as u8, row as u8);
                    let arb = &arbitration[tile_idx];

                    if arb.contended_banks != 0 {
                        crate::interpreter::execute::cycle_accurate::fire_bank_conflict_events(
                            &mut self.device.array.tiles[tile_idx],
                            arb.contended_banks,
                            cycle,
                            None,
                        );
                    }

                    // The core lost at least one bank: it is stalled for this
                    // cycle (AM020 ch.4:69 -- a conflict on any port stalls the
                    // whole datapath). Cycle-stat counter ticks every stalled
                    // cycle, independent of the trace-edge collapsing.
                    let stalled = arb.core_lost;
                    if stalled {
                        self.cores[core_idx].context.timing_context_mut().memory_stalls += 1;
                    }

                    let Some(active) = mem_stall_edge(stalled, self.cores[core_idx].mem_stall_active) else {
                        continue;
                    };
                    self.cores[core_idx].mem_stall_active = active;
                    if let Some(tile) = self.device.array.get_mut(col as u8, row as u8) {
                        // hw_id 23 derived from the toolchain event mapping
                        // rather than hardcoded.
                        let hw_id = crate::trace::core_event_to_hw_id(
                            &crate::interpreter::state::EventType::MemoryStall { cycles: 1, pc: None },
                        );
                        if let Some(id) = hw_id {
                            tile.notify_core_trace_level(id, cycle, active);
                            // Feed the core-module perf-counter bank too, same
                            // as `raise_instr_error` does for INSTR_ERROR: any
                            // core-module event is a legal perf-counter
                            // start/stop/reset trigger on real hardware
                            // (`perf_counters` module docs), and MEMORY_STALL
                            // is no exception -- a counter configured with
                            // start_event=MEMORY_STALL must be able to count
                            // core-side bank-arbitration-loss cycles, exactly
                            // as the DMA-side counterparts below already let a
                            // counter count DMA-side loss cycles. RISING EDGE
                            // ONLY (`if active`): the counter's own
                            // `handle_event` is polarity-blind (it just checks
                            // "does this id match my start_event"), so calling
                            // it on the falling edge too would treat the event
                            // clearing as the event firing again -- see the
                            // `if active` guard on the DMA-side call below for
                            // the same bug this mirrors.
                            if active {
                                tile.core_perf_counters.handle_event(id);
                            }
                        }
                    }
                }
            }
        }

        // Phase E (cont'd): DMA channel bank-arbitration pressure.
        //
        // A denied DMA channel is held for the whole cycle (task 6): the
        // arbiter decided this before the DMA step ran (`arbitrate_memory_banks`,
        // Phase C), so -- exactly like MEMORY_STALL above -- this event comes
        // from the SAME arbitration result that already decided who commits,
        // not from the DMA engine's own internal stall detection (which never
        // runs a step for a denied channel this cycle, so it cannot observe
        // the denial itself; that machinery drives the unrelated
        // DMA_x_STREAM_STARVATION/BACKPRESSURE events, the AXI-stream side of
        // the channel, not the SRAM bank-arbiter side).
        //
        // S2MM losing its local-memory WRITE port fires MEMORY_BACKPRESSURE;
        // MM2S losing its local-memory READ port fires MEMORY_STARVATION.
        // Both are held LEVELs like MEMORY_STALL (same `mem_stall_edge`
        // collapse), tracked per DMA-channel identity rather than per-core:
        // two channels on one tile can independently win/lose bank
        // arbitration in the same cycle. Every identity is visited every
        // cycle, not just this cycle's losers, so a channel whose denial just
        // cleared fires its falling edge.
        //
        // Held-level state (`dma_bank_denied_active`) lives on the `Tile`
        // itself, not on this engine's per-core bookkeeping -- it's
        // memory-module state, same as `mem_trace`/`mem_perf_counters`/
        // `dma_channels`, so one tile borrow covers the read, the edge
        // decision, and the write.
        {
            let cycle = self.total_cycles;
            for col in 0..self.cols {
                for row in self.compute_row_start..self.rows {
                    let tile_idx = self.device.array.tile_index(col as u8, row as u8);
                    let denied = &arbitration[tile_idx].denied_dma;
                    let Some(tile) = self.device.array.get_mut(col as u8, row as u8) else {
                        continue;
                    };

                    for ch in 0..DMA_BANK_CHANNELS_PER_DIRECTION {
                        let identities = [
                            (
                                ch as usize,
                                Requester::S2mm(ch),
                                crate::trace::dma_s2mm_memory_backpressure_hw_id(ch),
                            ),
                            (
                                DMA_BANK_CHANNELS_PER_DIRECTION as usize + ch as usize,
                                Requester::Mm2s(ch),
                                crate::trace::dma_mm2s_memory_starvation_hw_id(ch),
                            ),
                        ];
                        for (edge_idx, requester, hw_id) in identities {
                            let Some(hw_id) = hw_id else { continue };
                            let lost = denied.contains(&requester);
                            let Some(active) = mem_stall_edge(lost, tile.dma_bank_denied_active[edge_idx])
                            else {
                                continue;
                            };
                            tile.dma_bank_denied_active[edge_idx] = active;
                            tile.notify_mem_trace_level(hw_id, cycle, active);
                            // RISING EDGE ONLY: `handle_event` just checks "does
                            // this id match my start/stop/reset event", with no
                            // notion of polarity, so calling it on the falling
                            // edge (backpressure/starvation CLEARING) would arm
                            // a counter configured with this hw_id as its
                            // start_event exactly backwards -- on the level
                            // deasserting, not asserting. `notify_mem_trace_level`
                            // itself already gates its timer/edge-detector/halt
                            // side effects the same way (see its doc comment);
                            // this call must match.
                            if active {
                                tile.mem_perf_counters.handle_event(hw_id);
                            }
                        }
                    }
                }
            }
        }

        // Phase 3c: Fire TRUE (event code 1) on every configured trace unit.
        //
        // Hardware fires TRUE unconditionally every cycle. When a trace
        // unit has TRUE in one of its 8 event slots, it records it --
        // otherwise notify_event() returns immediately (no slot match).
        // This enables the "metronome" alignment strategy for multi-run
        // trace sweeps.
        {
            let cycle = self.total_cycles;
            const TRUE_EVENT: u8 = 1;
            for tile in &mut self.device.array.tiles {
                if tile.core_trace.is_configured() {
                    // Pass the core's current PC so mode-2's Start frame
                    // captures the real anchor PC instead of defaulting to
                    // 0. Mode 0/1 ignore the PC arg here (mode 0 has no
                    // PC field, mode 1 only records PCs on slot fires).
                    let pc = Some(tile.core.pc);
                    tile.core_trace.notify_event(TRUE_EVENT, cycle, pc);
                }
                if tile.mem_trace.is_configured() {
                    tile.mem_trace.notify_event(TRUE_EVENT, cycle, None);
                }
            }
        }

        // Phase 3d: Evaluate edge detectors.
        //
        // After all raw events have been generated (DMA, port, core, bank
        // conflict, TRUE), check edge detectors for rising/falling
        // transitions and fire EDGE_DETECTION_EVENT_0/1 as needed.
        {
            let cycle = self.total_cycles;
            for tile in &mut self.device.array.tiles {
                tile.evaluate_edge_detectors(cycle);
            }
        }

        // Phase 3e: Tick tile timers and performance counters; route firings.
        //
        // Each tile has two 64-bit timers (core module and memory module)
        // that increment every cycle.
        //
        // Core module perf counters are dispatched to tick_active_cycles or
        // tick_idle_cycles based on whether the core executed an instruction
        // (StepResult::Continue) this cycle. Counters configured with
        // ACTIVE_CORE (0x1C) as start_event are level-gated: they tick only
        // when the core is in Execute state, matching hardware behavior.
        //
        // Memory module perf counters are not gated on core Execute state;
        // they always use tick_active_cycles (existing semantics preserved).
        //
        // Tiles are indexed identically to self.cores (col * rows + row), so
        // cores[i] is the CoreState for tiles[i].
        //
        // Per aie-rt xaie_events_aieml.h, PERF_CNT_N hw event id is 5+N in
        // every module type (core 5..8, memmod 5..6, memtile 5..8, shim/PL
        // 5..6). Each counter that crossed its threshold this cycle is fed
        // back through handle_event (so self-resetting configs can recycle)
        // and forwarded to the owning module's trace unit.
        //
        // ORDERING: this phase runs *before* Phase 3f (commit_cycle) so that
        // perfcnt notify_event() calls accumulate into the same cycle's
        // pending_slot_mask as TRUE/edge-detector events. AM020 specifies
        // one frame per cycle: when multiple events fire in cycle N, they
        // must be coalesced into a single Multiple frame, not split across
        // two frames.
        {
            // PERF_CNT_0 has the same numeric ID (5) across every module type
            // per aie-rt xaie_events_aieml.h, so core_events::PERF_CNT_0 is
            // safe to use as the base for both core and memmod perf counters.
            use xdna_archspec::aie2::trace_events::core_events::PERF_CNT_0 as PERF_CNT_BASE;
            // TRUE (event 1) and ACTIVE_CORE (event 0x1C) are level-asserted
            // start events on real hardware: TRUE is always asserted, and
            // ACTIVE_CORE is asserted while the core is in Execute state.
            // A counter configured with either as its start event needs to
            // see the corresponding handle_event() each cycle to transition
            // Idle->Active before the tick logic runs. Without this, the
            // perfcnt anchor scheme emitted by mlir-trace-inject (start =
            // ACTIVE_CORE, reset = PERF_CNT_0, threshold = period) would
            // never fire because the counter stays stuck in Idle (#354).
            const TRUE_EVENT: u8 = 1;
            const ACTIVE_CORE_EVENT: u8 = 0x1C;
            let cycle = self.total_cycles;

            // "No clock, no tick": a clock-gated module has no clock, so its
            // timer and performance counters freeze. Precompute the per-tile
            // gate booleans here -- computing them inside the loop below would
            // borrow the clock controller while the tiles are already borrowed
            // mutably.
            //
            // Bank-to-module mapping by tile kind:
            //   core bank (core_timer + core_perf_counters):
            //     Compute -> Core module gate; Shim -> column gate (shim PL
            //     counters live in this bank and are clocked with the column --
            //     our model has no separate shim-PL module gate); Mem -> column
            //     gate (memtile has 0 core counters, so the choice is moot).
            //   mem bank (mem_timer + mem_perf_counters):
            //     Compute/Mem -> Memory module gate; Shim -> column gate (shim
            //     has 0 mem counters, moot).
            let clock_gates: Vec<(bool, bool)> = {
                use crate::device::clock_control::ModuleKind;
                let clock = self.device.array.clock();
                let rows = self.rows;
                self.device
                    .array
                    .tiles
                    .iter()
                    .enumerate()
                    .map(|(i, tile)| {
                        let col = (i / rows) as u8;
                        let row = (i % rows) as u8;
                        let core_clocked = match tile.tile_kind {
                            TileKind::Compute => clock.is_module_active(col, row, ModuleKind::Core),
                            _ => clock.is_column_active(col),
                        };
                        let mem_clocked = match tile.tile_kind {
                            TileKind::Compute | TileKind::Mem => {
                                clock.is_module_active(col, row, ModuleKind::Memory)
                            }
                            _ => clock.is_column_active(col),
                        };
                        (core_clocked, mem_clocked)
                    })
                    .collect()
            };

            for (i, tile) in self.device.array.tiles.iter_mut().enumerate() {
                let (core_clocked, mem_clocked) = clock_gates[i];

                // Core module (timer + perf counters). Skipped entirely when
                // the module is clock-gated: a frozen clock advances neither
                // the free-running timer nor any active counter.
                if core_clocked {
                    tile.core_timer.tick();

                    let core_active = self.cores.get(i).map_or(false, |c| c.active_this_cycle);

                    // Re-assert TRUE every cycle and ACTIVE_CORE only when the
                    // core is executing. handle_event() is idempotent: an
                    // already-Active counter stays Active; counters whose start
                    // event doesn't match are unaffected. Stop and reset events
                    // still take priority inside handle_event() per its existing
                    // ordering.
                    //
                    // ACTIVE_CORE is an edge-style start event: it pulses on
                    // transition to Execute and arms the counter. Once Active,
                    // the counter ticks every cycle regardless of whether the
                    // core is still in Execute -- HW perf counters are duration
                    // counters started by ACTIVE_CORE and stopped by
                    // DISABLED_CORE, not gated per cycle on Execute state.
                    tile.core_perf_counters.handle_event(TRUE_EVENT);
                    if core_active {
                        tile.core_perf_counters.handle_event(ACTIVE_CORE_EVENT);
                    }

                    let core_fired = tile.core_perf_counters.tick();
                    if !core_fired.is_empty() {
                        // Snapshot the core's pipeline-adjusted PC for trace
                        // stamping. tile.core.pc is updated each cycle from the
                        // core context (Phase 2) and includes the stall-PC
                        // pipeline adjustment, so it matches what HW's trace
                        // controller would sample when the perf-counter
                        // threshold fires.
                        let core_pc = tile.core.pc;
                        for cnt_idx in core_fired {
                            let hw_id = PERF_CNT_BASE + cnt_idx as u8;
                            // Feed back so self-reset configs work, then trace-notify.
                            tile.core_perf_counters.handle_event(hw_id);
                            tile.notify_core_trace_event(hw_id, cycle, Some(core_pc));
                        }
                    }
                }

                // Memory module (timer + perf counters). Same freeze rule.
                if mem_clocked {
                    tile.mem_timer.tick();
                    tile.mem_perf_counters.handle_event(TRUE_EVENT);

                    let mem_fired = tile.mem_perf_counters.tick();
                    for cnt_idx in mem_fired {
                        let hw_id = PERF_CNT_BASE + cnt_idx as u8;
                        tile.mem_perf_counters.handle_event(hw_id);
                        tile.notify_mem_trace_event(hw_id, cycle, None);
                    }
                }
            }
        }

        // Phase 3f: Commit per-cycle trace frames.
        //
        // Per AM020 event-time mode, a trace unit creates at most one frame
        // per cycle. `notify_event` accumulates slot activity into a bitmask;
        // committing here emits one Single/Multiple frame for this cycle.
        // Without this step the mask only commits lazily on the next cycle's
        // event, which can leave the last cycle's frame uncommitted past the
        // simulation's natural end and inflate routing pressure for the
        // uncommitted bytes when they eventually land at flush time.
        //
        // This phase MUST run after all event-emitting phases (3a-3e) so
        // every event for `cycle` is already in pending_slot_mask. Otherwise
        // events that fire after the commit get carried into the next
        // cycle's frame, splitting a HW Multiple frame into two.
        //
        // Mode-2 atoms are NOT emitted here. Per AM020 ch.2 mode 2 records
        // only branches and ZOL LC, not per-cycle execution status.
        // notify_atom / notify_branch_taken fire from the executor at
        // branch resolution; this phase just drains them via commit_cycle.
        {
            let cycle = self.total_cycles;
            for tile in &mut self.device.array.tiles {
                tile.core_trace.commit_cycle(cycle);
                tile.mem_trace.commit_cycle(cycle);
            }
        }

        // Phase F: Update tile DMA channel state from engine state
        self.sync_dma_completion();

        self.total_cycles += 1;

        // Determine if we should halt the engine.
        //
        // The engine halts when all work is done:
        // 1. All cores have halted (or no cores were enabled), AND
        // 2. Either:
        //    a. No DMA activity at all, OR
        //    b. DMA is active but no progress for multiple cycles (deadlock)
        //
        // For DMA-only tests (no cores enabled), NPU instructions from the
        // host command stream trigger DMA transfers. The engine must keep
        // stepping so DMA engines and stream switches advance. The run loop
        // calls force_running() while NPU instructions are still pending;
        // once DMA is triggered, the deadlock detection here handles
        // completion and stall detection normally.
        let any_cores_enabled = self.cores.iter().any(|c| c.enabled);

        // All compute work is done when cores have halted (or none exist).
        let cores_done = if any_cores_enabled {
            all_halted
        } else {
            !any_running
        };

        if cores_done && !dma_active {
            self.status = EngineStatus::Halted;
            return;
        }

        // -- Monotonic progress detection --
        //
        // Three signals count as forward progress, any one of which resets
        // the stall counter:
        //
        // 1. DMA bytes transferred: a DMA channel moved data.
        // 2. Lock releases: an arbiter released a lock to a waiter.
        // 3. Core instructions executed: a core advanced its PC at least
        //    once outside of WaitLock/WaitDma/WaitStream.
        //
        // The third signal exists to avoid false stalls during heavy scalar
        // computation (e.g. byte-by-byte memcpy in compute kernels) that can
        // run for tens of thousands of cycles between any DMA or lock event.
        // True deadlocks are still caught: a deadlocked core is in WaitLock
        // (etc.) and does not advance total_instructions.
        if self.stall_threshold > 0 {
            let dma_bytes = self.device.array.total_dma_bytes_transferred();
            let lock_releases = self.device.array.total_lock_releases();
            let instructions = self.total_instructions;

            if dma_bytes > self.last_dma_bytes
                || lock_releases > self.last_lock_releases
                || instructions > self.last_instructions
            {
                self.stall_cycles = 0;
                self.last_dma_bytes = dma_bytes;
                self.last_lock_releases = lock_releases;
                self.last_instructions = instructions;
            } else {
                self.stall_cycles += 1;
                if self.stall_cycles >= self.stall_threshold {
                    log::warn!(
                        "Stall detected: no progress for {} cycles (dma_bytes={}, lock_releases={}, instructions={})",
                        self.stall_cycles, dma_bytes, lock_releases, instructions,
                    );
                    self.status = EngineStatus::Stalled;
                }
            }
        } else if cores_done && dma_active {
            // Stall detection disabled -- fall back to simple deadlock check.
            // Only runs while cores are done but DMA is still active.
            let dma_bytes = self.device.array.total_dma_bytes_transferred();
            if dma_bytes > self.last_dma_bytes {
                self.stall_cycles = 0;
                self.last_dma_bytes = dma_bytes;
            } else {
                self.stall_cycles += 1;
                if self.stall_cycles >= 50 {
                    log::info!(
                        "Engine halting: all cores done, DMA stalled for {} cycles",
                        self.stall_cycles,
                    );
                    self.status = EngineStatus::Halted;
                }
            }
        } else {
            // Cores still running, stall detection disabled -- reset counter
            // so it doesn't carry over stale counts into the post-halt check.
            self.stall_cycles = 0;
        }
    }

    /// Phases A/B/C of the cycle: collect both agents' memory-bank demands and
    /// run the per-physical-bank round-robin arbiters. Returns the per-tile
    /// outcome the commit (Phase D) and emission (Phase E) passes consume.
    ///
    /// COMPUTE TILES ONLY. MemTile bank geometry is a different, unvalidated
    /// arrangement (`BankLayout::MemTile`) and its DMA has 6+6 channels that
    /// the compute-sized `Requester` ordinals cannot even represent, so
    /// memtiles and shims are never arbitrated -- they keep behaving exactly as
    /// before. `DmaEngine::peek_bank_demand` enforces the same gate itself.
    ///
    /// Clock-gated modules declare nothing: a gated DMA module is skipped by
    /// `step_all_dma` (it cannot touch a bank), and a gated core is skipped by
    /// Phase 2 (it cannot issue). Declaring for them would deny the other agent
    /// a bank against a requester that never runs.
    ///
    /// The core's stall is applied HERE, not in Phase 2, so that `step` and
    /// `stall_for_bank` can never both run for one core in one cycle: Phase 2
    /// skips a core whose `core_lost` is set. `accumulate_bank_grants` runs
    /// after `stall_for_bank` (which asserts the sticky mask is empty on a
    /// FRESH stall) and feeds the ports that already won back into the next
    /// cycle's peek, so a bundle whose own ports collide converges instead of
    /// livelocking. The mask is reset inside `try_resume_stall` when WaitBank
    /// resolves -- the coordinator keeps no second copy.
    fn arbitrate_memory_banks(&mut self) -> Vec<TileArbitration> {
        let mut out = vec![TileArbitration::default(); self.device.array.tiles.len()];

        for col in 0..self.cols {
            if !self.device.array.clock().is_column_active(col as u8) {
                continue;
            }
            for row in self.compute_row_start..self.rows {
                let (c, r) = (col as u8, row as u8);
                let tile_idx = self.device.array.tile_index(c, r);
                if !self.device.array.tiles[tile_idx].is_compute() {
                    continue;
                }
                let layout = self.device.array.tiles[tile_idx].bank_layout();
                let core_idx = col * self.rows + row;

                // Phase A: DMA demand (no transfer).
                //
                // SmallVec, not Vec (task 6 review, Minor-4): this is a hot
                // per-compute-tile-per-cycle allocation. The demand set is
                // bounded -- a compute tile's DMA engine has 2 channels
                // (CLAUDE.md) and a core contributes at most 3 memory ports
                // (LoadA/LoadB/Store, `bank_arbiter::CorePort`) -- so 5 inline
                // slots cover every real case with no heap allocation.
                let mut demands: SmallVec<[(Requester, u16); 5]> = SmallVec::new();
                if self.device.array.clock().is_module_active(c, r, ModuleKind::Dma)
                    && !self.device.array.clock().is_adaptive_dma_engaged(c, r)
                {
                    if let Some(engine) = self.device.array.dma_engine(c, r) {
                        demands.extend(engine.peek_bank_demand(layout));
                    }
                }

                // Phase B: core demand (no commit).
                if self.cores[core_idx].enabled
                    && self.device.array.clock().is_module_active(c, r, ModuleKind::Core)
                {
                    let tile = &self.device.array.tiles[tile_idx];
                    let core = &mut self.cores[core_idx];
                    demands.extend(core.interpreter.peek_bank_demand(&core.context, tile, layout));
                }

                if demands.is_empty() {
                    continue;
                }

                // Phase C: arbitrate.
                let arb = self.bank_arbiters[tile_idx].arbitrate(&demands);

                out[tile_idx].contended_banks = arb.contended_banks;
                out[tile_idx].denied_dma =
                    arb.lost.iter().copied().filter(|r| !matches!(r, Requester::Core(_))).collect();

                if arb.core_lost() {
                    out[tile_idx].core_lost = true;
                    let core = &mut self.cores[core_idx];
                    core.interpreter.stall_for_bank(&mut core.context);
                    core.interpreter.accumulate_bank_grants(arb.granted_core_ports());
                } else {
                    // Constraint: a core that is NOT mid bank-stall must carry
                    // no sticky served-ports mask -- otherwise the next bundle
                    // would silently have those ports dropped from arbitration.
                    // (`try_resume_stall`'s WaitBank arm owns the reset; the
                    // coordinator keeps no second copy of the mask.)
                    debug_assert!(
                        self.cores[core_idx].interpreter.status() == CoreStatus::WaitBank
                            || self.cores[core_idx].interpreter.bank_served_ports().is_empty(),
                        "a freshly-issuing bundle must start with an empty served mask"
                    );
                }
            }
        }

        out
    }

    /// Sync DMA start requests from tile.dma_channels to DmaEngine.
    ///
    /// When a core executes DmaStart, it sets tile.dma_channels[ch].running = true
    /// and tile.dma_channels[ch].start_queue = bd_index. This method reads those
    /// values and calls DmaEngine.start_channel().
    fn sync_dma_start_requests(&mut self) {
        for col in 0..self.cols as u8 {
            for row in 0..self.rows as u8 {
                // Get tile and DMA engine together
                if let Some((tile, engine)) = self.device.array.tile_and_dma(col, row) {
                    for ch in 0..tile.dma_channels.len() {
                        let channel = &mut tile.dma_channels[ch];

                        // Check for pending start request:
                        // - running is true (set by DmaStart instruction)
                        // - engine channel is not already active
                        if channel.running && !engine.channel_active(ch as u8) {
                            let bd_index = channel.start_queue as u8;

                            // Start the channel on the engine
                            if engine.start_channel(ch as u8, bd_index).is_ok() {
                                // Clear start_queue to indicate we processed it
                                channel.start_queue = 0xFF;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Sync DMA completion state from DmaEngine to tile.dma_channels.
    ///
    /// When DmaEngine reports a channel as complete (not active), we clear
    /// tile.dma_channels[ch].running so that DmaWait instructions can proceed.
    fn sync_dma_completion(&mut self) {
        for col in 0..self.cols as u8 {
            for row in 0..self.rows as u8 {
                if let Some((tile, engine)) = self.device.array.tile_and_dma(col, row) {
                    for ch in 0..tile.dma_channels.len() {
                        let channel = &mut tile.dma_channels[ch];

                        // If engine reports channel complete, clear running flag
                        if channel.running {
                            let state = engine.channel_state(ch as u8);
                            if matches!(state, ChannelState::Idle) {
                                channel.running = false;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Run for up to `max_cycles` cycles.
    ///
    /// Stops early if all cores halt or an error occurs.
    /// Returns the number of cycles actually executed.
    pub fn run(&mut self, max_cycles: u64) -> u64 {
        let start = self.total_cycles;

        for _ in 0..max_cycles {
            self.step();

            if matches!(self.status, EngineStatus::Halted | EngineStatus::Stalled | EngineStatus::Error) {
                break;
            }
        }

        self.total_cycles - start
    }

    /// Pause execution.
    pub fn pause(&mut self) {
        if self.status == EngineStatus::Running {
            self.status = EngineStatus::Paused;
        }
    }

    /// Resume execution.
    pub fn resume(&mut self) {
        if self.status == EngineStatus::Paused {
            self.status = EngineStatus::Ready;
        }
    }

    /// Reset all cores.
    pub fn reset(&mut self) {
        let rows = self.rows;
        for (idx, core) in self.cores.iter_mut().enumerate() {
            let col = (idx / rows) as u8;
            let row = (idx % rows) as u8;
            core.interpreter.reset();
            // ExecutionContext::reset() resets via Self::new() which sets
            // CORE_ID to (0,0). Re-tag each core's identity so reads of
            // CORE_ID continue to return this core's actual (col, row)
            // -- kernels routinely branch on it.
            core.context = ExecutionContext::new_for_tile(col, row);
            core.enabled = false;
            core.trace_events_consumed = 0;
            core.active_this_cycle = false;
        }

        // The bank arbiters' rotors are per-tile hardware state; a column reset
        // returns them to their power-on position.
        for arb in &mut self.bank_arbiters {
            *arb = BankArbiter::new();
        }

        self.status = EngineStatus::Ready;
        self.total_cycles = 0;
        self.total_instructions = 0;
        self.last_dma_bytes = 0;
        self.last_lock_releases = 0;
        self.last_instructions = 0;
        self.stall_cycles = 0;
    }

    /// Reset all per-context tile state on hw_context teardown / re-creation.
    ///
    /// Mirrors what real HW does on hw_context destroy: a column reset that
    /// clears locks, DMA channel queues + BD configs, stream switch routing,
    /// and trace state. Host memory (BO contents) is intentionally NOT
    /// touched -- the bridge runner zeroes the trace BO via sync_to_device
    /// before each new run, and input BOs are re-uploaded the same way.
    ///
    /// Called from the FFI on a fresh xrt::hw_context creation so the new
    /// run starts on a clean column. Without this, lock state from a prior
    /// run can leave an acquire stuck (target_value reached but the queued
    /// release never fires for the new BD), and the new run stalls at the
    /// stall threshold without making forward progress. Each reset also
    /// nudges the trace_unit clear via Trace_Control0 rewrites in the new
    /// run's insts.bin, but locks/BDs need this explicit reset because no
    /// existing register write paths reach them.
    pub fn reset_for_new_context(&mut self) {
        self.device.async_errors.clear();
        self.reset();
        self.device.array.reset();
    }

    /// Sync core enabled state and PC from device tiles.
    ///
    /// After applying CDO, the tile.core.enabled and tile.core.pc fields
    /// are set by register writes. This method syncs those values to the
    /// engine's internal core state, enabling execution.
    ///
    /// Call this after `device_mut().apply_cdo()` to start execution.
    ///
    /// Scope: this mirrors `tile.core.enabled`/`pc` into engine bookkeeping
    /// only. It does NOT route through `core_debug.enable()`, so it neither
    /// clears the `core_debug` RESET bit nor updates DEBUG_HALT/enable
    /// state. That is correct in the CDO flow because the preceding
    /// `apply_cdo()` `Core_Control` register write already established
    /// `core_debug` state. Do not use this as a standalone enable path:
    /// without a prior `Core_Control` write (or `Coordinator::enable_core`,
    /// which goes through `core_debug.enable()`), `core_debug` is left
    /// unsynced and `Core_Status` reports stale bits. See §8 close-out
    /// (2026-05-19).
    pub fn sync_cores_from_device(&mut self) {
        for col in 0..self.cols {
            for row in self.compute_row_start..self.rows {
                if let Some(tile) = self.device.array.get(col as u8, row as u8) {
                    if tile.is_compute() {
                        let idx = col * self.rows + row;
                        if let Some(core) = self.cores.get_mut(idx) {
                            // Sync enabled state from tile
                            core.enabled = tile.core.enabled;
                            // Sync PC from tile (if set by CDO)
                            if tile.core.pc != 0 || tile.core.enabled {
                                core.context.set_pc(tile.core.pc);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Get the number of enabled cores.
    pub fn enabled_cores(&self) -> usize {
        self.cores.iter().filter(|c| c.enabled).count()
    }

    /// Get the number of running (non-halted) cores.
    pub fn active_cores(&self) -> usize {
        self.cores.iter().filter(|c| c.enabled && !c.interpreter.is_halted()).count()
    }

    /// Check if all enabled cores are blocked (stalled on lock/DMA or halted).
    ///
    /// Returns true when no enabled core can make forward progress. This is
    /// used by the NPU executor warm-up phase: on real hardware, cores run
    /// for thousands of cycles before NPU instructions arrive through the
    /// NoC, so they always reach their first blocking point (typically a lock
    /// acquire) before any host-issued writes modify tile memory.
    pub fn all_cores_blocked(&self) -> bool {
        let enabled: Vec<_> = self.cores.iter().filter(|c| c.enabled).collect();
        !enabled.is_empty() && enabled.iter().all(|c| c.interpreter.is_stalled() || c.interpreter.is_halted())
    }

    /// Set auto-run mode.
    pub fn set_auto_run(&mut self, auto: bool) {
        self.auto_run = auto;
    }

    /// Check if auto-run is enabled.
    pub fn is_auto_run(&self) -> bool {
        self.auto_run
    }

    /// Get status as string for display.
    pub fn status_string(&self) -> &'static str {
        match self.status {
            EngineStatus::Ready => "Ready",
            EngineStatus::Running => "Running",
            EngineStatus::Paused => "Paused",
            EngineStatus::Halted => "Halted",
            EngineStatus::Stalled => "Stalled",
            EngineStatus::Error => "Error",
        }
    }

    // Private helpers

    fn core_index(&self, col: usize, row: usize) -> Option<usize> {
        if col < self.cols && row < self.rows {
            Some(col * self.rows + row)
        } else {
            None
        }
    }

    fn get_core(&self, col: usize, row: usize) -> Option<&CoreState> {
        self.core_index(col, row).map(|idx| &self.cores[idx])
    }

    fn get_core_mut(&mut self, col: usize, row: usize) -> Option<&mut CoreState> {
        self.core_index(col, row).map(|idx| &mut self.cores[idx])
    }

    /// Single dispatch point for a drained `CtrlPacketAction`. Owns the
    /// SLVERR decode-gate: a control-packet access whose offset does not
    /// decode is a faithful AXI slave error -- latch bit 2, suppress the
    /// access, and continue (poll-only sticky; never engine-fatal). The
    /// structural-rejection `Error` arm is unchanged (out of scope).
    fn dispatch_ctrl_action(&mut self, action: crate::device::tile::CtrlPacketAction) {
        use crate::device::tile::CtrlPacketAction;
        match action {
            CtrlPacketAction::WriteRegister { col, row, offset, value } => {
                if self.device.ctrl_pkt_offset_decodes(row, offset) {
                    self.device.write_tile_register(col, row, offset, value);
                } else {
                    log::error!(
                        "Tile ({},{}) ctrl_pkt SLVERR: write to undecoded offset \
                         0x{:05X} (sets Control_Packet_Handler_Status bit 0x4)",
                        col,
                        row,
                        offset
                    );
                    self.device.latch_ctrl_slverr(col, row);
                }
            }
            CtrlPacketAction::ReadRegisters { col, row, offset, count, response_id } => {
                if self.device.ctrl_pkt_read_range_decodes(row, offset, count) {
                    self.device.array.handle_read_registers(col, row, offset, count, response_id);
                } else {
                    log::error!(
                        "Tile ({},{}) ctrl_pkt SLVERR: read of {} regs from undecoded \
                         offset 0x{:05X} (sets Control_Packet_Handler_Status bit 0x4)",
                        col,
                        row,
                        count,
                        offset
                    );
                    self.device.latch_ctrl_slverr(col, row);
                }
            }
            CtrlPacketAction::Error(msg) => {
                self.device.array.fatal_errors.push(msg);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_stall_edge_collapses_sustained_conflict_to_one_span() {
        // MEMORY_STALL is a held level: only the rising (conflict begins) and
        // falling (conflict clears) transitions produce a trace edge. A
        // sustained conflict produces NO edge -- this is what collapses N
        // consecutive bank-conflict cycles into one B..E span instead of N
        // pulses, and also prevents the old direct-notify + EventLog-drain
        // double emission.
        assert_eq!(mem_stall_edge(true, false), Some(true), "rising on first conflict cycle");
        assert_eq!(mem_stall_edge(true, true), None, "no edge while conflict sustained");
        assert_eq!(mem_stall_edge(false, true), Some(false), "falling when conflict clears");
        assert_eq!(mem_stall_edge(false, false), None, "no edge while idle");
    }

    #[test]
    fn test_engine_npu1() {
        let engine = InterpreterEngine::new_npu1();

        assert_eq!(engine.status(), EngineStatus::Ready);
        assert_eq!(engine.total_cycles(), 0);
        assert_eq!(engine.enabled_cores(), 0);
    }

    #[test]
    fn test_enable_disable_cores() {
        let mut engine = InterpreterEngine::new_npu1();

        // Enable core at (0, 2) - first compute tile
        engine.enable_core(0, 2);
        assert!(engine.is_core_enabled(0, 2));
        assert_eq!(engine.enabled_cores(), 1);

        // Disable it
        engine.disable_core(0, 2);
        assert!(!engine.is_core_enabled(0, 2));
        assert_eq!(engine.enabled_cores(), 0);
    }

    #[test]
    fn wait_bank_core_does_not_count_as_all_cores_blocked() {
        // A bank-stalled core is making BOUNDED progress: the arbiter is
        // round-robin over a fixed requester set and provably
        // starvation-free (see `no_requester_starves_under_the_retry_contract`
        // near the bank arbiter), so the core is guaranteed to win its bank
        // within a bounded number of cycles. That's fundamentally unlike
        // WaitingLock/WaitingDma/WaitingStream, which may block indefinitely
        // (e.g. a lock another core never releases).
        //
        // all_cores_blocked()'s only consumer is the FFI warm-up break
        // (crates/xdna-emu-ffi/src/backend.rs), which means "stop warming up,
        // NO core can make forward progress." If WaitBank counted here, the
        // common warm-up shape -- N-1 cores parked on WaitingLock while the
        // last core is still mid init-loop -- would flip all_cores_blocked()
        // true the instant that last core self-collides on a bank for a
        // single cycle, breaking warm-up before the init loop finishes and
        // letting NPU instructions write tile memory too early.
        let mut engine = InterpreterEngine::new_npu1();
        engine.enable_core(0, 2);

        let idx = engine.core_index(0, 2).expect("(0,2) is a valid compute tile");
        let core_state = &mut engine.cores[idx];
        core_state.interpreter.stall_for_bank(&mut core_state.context);

        assert_eq!(engine.core_status(0, 2), Some(CoreStatus::WaitBank));
        assert!(
            !engine.all_cores_blocked(),
            "a WaitBank core must not count toward all_cores_blocked() -- \
             it is bounded-progress, not indefinitely blocked"
        );
    }

    /// End-to-end: a core storing into physical bank 0 every cycle while an
    /// S2MM DMA channel writes into the SAME bank must actually lose some of
    /// those cycles to the arbiter -- and pay for them.
    ///
    /// This is the wiring test for the request -> arbitrate -> commit loop:
    /// the arbiter, the two peeks and `stall_for_bank` all have their own unit
    /// tests, but only the coordinator can prove that a real bundle competing
    /// with a real DMA transfer over a real bank ends up stalled, that the
    /// stall costs a cycle, and that MEMORY_STALL is emitted for it. Under the
    /// retroactive Phase-4 observer this charged nothing.
    #[test]
    fn core_and_dma_contending_for_one_bank_costs_the_core_cycles() {
        use crate::device::dma::BdConfig;
        use crate::device::dma::StreamData;
        use crate::interpreter::state::MOD_BASE_DJ;

        // `st dj0, [p0, #4]; mov m0, #0xaa` (from the debug_halt_probe ELF).
        // With p0 = 0x70400 the store lands at tile-local 0x404 -- physical
        // bank 0 (BankLayout::Compute).
        const STORE_BUNDLE: [u8; 8] = [0x03, 0x40, 0x57, 0x11, 0x00, 0x20, 0xd4, 0x00];

        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        {
            let tile = engine.device_mut().tile_mut(0, 2).expect("compute tile (0,2)");
            let mut prog = [0u8; 256];
            for chunk in prog.chunks_exact_mut(8) {
                chunk.copy_from_slice(&STORE_BUNDLE);
            }
            tile.write_program(0, &prog);
        }
        engine.set_core_pointer(0, 2, 0, 0x70400);
        engine.set_core_modifier(0, 2, MOD_BASE_DJ, 0x7700);
        engine.enable_core(0, 2);

        // S2MM channel 0 on the same tile, writing 0x400.. -- bank 0 too.
        engine
            .device_mut()
            .array
            .dma_engine_mut(0, 2)
            .unwrap()
            .configure_bd(0, BdConfig::simple_1d(0x400, 128))
            .unwrap();
        {
            let tile = engine.device_mut().tile_mut(0, 2).unwrap();
            tile.dma_channels[0].running = true;
            tile.dma_channels[0].start_queue = 0;
        }

        // Keep the S2MM fed so it is genuinely mid-transfer (a starved channel
        // declares no bank demand -- it issues no memory request on HW either).
        for _ in 0..400 {
            let dma = engine.device_mut().array.dma_engine_mut(0, 2).unwrap();
            while dma.stream_in_len() < 8 {
                dma.push_stream_in(StreamData { data: 0xA5, tlast: false, channel: 0 });
            }
            engine.step();
        }

        let stalls = engine.core_context(0, 2).unwrap().timing_context().memory_stalls;
        assert!(
            stalls > 0,
            "core storing into bank 0 while the DMA writes bank 0 must lose \
             arbitration on some cycle and be charged for it; got {stalls} stall cycles"
        );
        assert_eq!(
            engine.core_status(0, 2),
            Some(CoreStatus::Ready),
            "a bank stall is a one-cycle retry, not a terminal state"
        );
    }

    /// Task 7 review fix: MEMORY_STALL must drive the core-module perf
    /// counter bank, exactly like the sibling DMA_x_MEMORY_BACKPRESSURE/
    /// STARVATION events already drive `mem_perf_counters` a few lines above
    /// this same Phase E code. All three events are co-derived from the
    /// identical bank-arbiter `Arbitration` result via the same
    /// `mem_stall_edge` collapse -- there is no hardware reason core-side
    /// arbitration loss should be perf-counter-invisible while DMA-side loss
    /// isn't (real hardware perf counters accept ANY module event, including
    /// MEMORY_STALL, as a start/stop/reset trigger; see `raise_instr_error`
    /// for the established precedent of a non-TRUE/ACTIVE_CORE core event
    /// doing exactly this).
    ///
    /// Reuses `core_and_dma_contending_for_one_bank_costs_the_core_cycles`'s
    /// contention fixture (whose own `memory_stalls` assertion already
    /// proves the core loses arbitration here), but reads the count off a
    /// core-module perf counter configured with start_event=MEMORY_STALL(23)
    /// instead of the timing-context stat.
    #[test]
    fn core_losing_bank_arbitration_ticks_a_core_perf_counter_armed_by_memory_stall() {
        use crate::device::dma::BdConfig;
        use crate::device::dma::StreamData;
        use crate::interpreter::state::MOD_BASE_DJ;
        use xdna_archspec::aie2::trace_events::core_events;

        const STORE_BUNDLE: [u8; 8] = [0x03, 0x40, 0x57, 0x11, 0x00, 0x20, 0xd4, 0x00];

        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        {
            let tile = engine.device_mut().tile_mut(0, 2).expect("compute tile (0,2)");
            let mut prog = [0u8; 256];
            for chunk in prog.chunks_exact_mut(8) {
                chunk.copy_from_slice(&STORE_BUNDLE);
            }
            tile.write_program(0, &prog);
        }
        engine.set_core_pointer(0, 2, 0, 0x70400);
        engine.set_core_modifier(0, 2, MOD_BASE_DJ, 0x7700);
        engine.enable_core(0, 2);

        engine
            .device_mut()
            .array
            .dma_engine_mut(0, 2)
            .unwrap()
            .configure_bd(0, BdConfig::simple_1d(0x400, 128))
            .unwrap();
        {
            let tile = engine.device_mut().tile_mut(0, 2).unwrap();
            tile.dma_channels[0].running = true;
            tile.dma_channels[0].start_queue = 0;
        }

        // Core-module perf counter 0: start=MEMORY_STALL(23), no stop/reset,
        // no threshold -- a free-running "count every cycle the core loses
        // bank arbitration" counter, the same idiom as the sibling
        // `core_module_gate_freezes_core_counter_but_not_mem_counter` test.
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = core_events::MEMORY_STALL as u32;
            tile.core_perf_counters.write_control_start_stop(ctrl0, 0, 1, 7);
        }

        for _ in 0..400 {
            let dma = engine.device_mut().array.dma_engine_mut(0, 2).unwrap();
            while dma.stream_in_len() < 8 {
                dma.push_stream_in(StreamData { data: 0xA5, tlast: false, channel: 0 });
            }
            engine.step();
        }

        let count = engine.device().array.tile(0, 2).core_perf_counters.read_counter(0);
        assert!(
            count > 0,
            "a perf counter armed by MEMORY_STALL's start event must count core-side \
             bank-arbitration-loss cycles; got {count}"
        );
    }

    /// Task 7: a denied DMA channel must raise its own bank-pressure event --
    /// S2MM losing arbitration fires DMA_S2MM_0_MEMORY_BACKPRESSURE (mem
    /// event 39) -- regardless of the core's own MEMORY_STALL/CONFLICT_DM_BANK
    /// bookkeeping (already covered by
    /// `core_and_dma_contending_for_one_bank_costs_the_core_cycles`, whose
    /// contention setup this test reuses).
    ///
    /// Mirrors the established `test_perfcnt_threshold_routes_to_trace_unit`
    /// idiom: configure `tile.mem_trace` for the hw_id under test (mode=
    /// EventTime, start=TRUE so the coordinator's Phase 3c TRUE tick arms it
    /// with no manual poke), run the real contention scenario, and assert the
    /// trace unit actually encoded more than just its 8-byte Start marker.
    /// Should FAIL before Phase E emits this event (only the Start marker
    /// present, exactly 8 bytes).
    #[test]
    fn dma_s2mm_channel_losing_bank_arbitration_fires_memory_backpressure() {
        use crate::device::dma::BdConfig;
        use crate::device::dma::StreamData;
        use crate::interpreter::state::MOD_BASE_DJ;

        // Same store-vs-S2MM-channel-0 bank-0 contention as
        // `core_and_dma_contending_for_one_bank_costs_the_core_cycles`.
        const STORE_BUNDLE: [u8; 8] = [0x03, 0x40, 0x57, 0x11, 0x00, 0x20, 0xd4, 0x00];

        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        {
            let tile = engine.device_mut().tile_mut(0, 2).expect("compute tile (0,2)");
            let mut prog = [0u8; 256];
            for chunk in prog.chunks_exact_mut(8) {
                chunk.copy_from_slice(&STORE_BUNDLE);
            }
            tile.write_program(0, &prog);
        }
        engine.set_core_pointer(0, 2, 0, 0x70400);
        engine.set_core_modifier(0, 2, MOD_BASE_DJ, 0x7700);
        engine.enable_core(0, 2);

        engine
            .device_mut()
            .array
            .dma_engine_mut(0, 2)
            .unwrap()
            .configure_bd(0, BdConfig::simple_1d(0x400, 128))
            .unwrap();
        {
            let tile = engine.device_mut().tile_mut(0, 2).unwrap();
            tile.dma_channels[0].running = true;
            tile.dma_channels[0].start_queue = 0;
        }

        // Mem-module trace on tile (0,2): mode=EventTime(0), start=TRUE(1),
        // slot 0 = DMA_S2MM_0_MEMORY_BACKPRESSURE (39).
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = (0u32 << 24) | (1u32 << 16) | 0u32;
            tile.mem_trace.write_register(0x00, ctrl0);
            tile.mem_trace.write_register(0x10, 39u32);
        }

        for _ in 0..400 {
            let dma = engine.device_mut().array.dma_engine_mut(0, 2).unwrap();
            while dma.stream_in_len() < 8 {
                dma.push_stream_in(StreamData { data: 0xA5, tlast: false, channel: 0 });
            }
            engine.step();
        }

        let encoded = engine.device().array.tile(0, 2).mem_trace.encoded_bytes_len();
        assert!(
            encoded > 8,
            "S2MM channel 0 contending bank 0 with the core must lose arbitration on some \
             cycle and raise DMA_S2MM_0_MEMORY_BACKPRESSURE; trace has only {encoded} encoded \
             bytes (just the Start marker) -- event never fired"
        );
    }

    /// Task 7: MM2S losing arbitration fires DMA_MM2S_0_MEMORY_STARVATION
    /// (mem event 41). Same idiom as the S2MM backpressure test above, but
    /// contends an MM2S channel (flat channel index 2 = MM2S per-direction 0
    /// on a compute tile) reading bank 0 against the core's store to bank 0.
    #[test]
    fn dma_mm2s_channel_losing_bank_arbitration_fires_memory_starvation() {
        use crate::device::dma::BdConfig;
        use crate::interpreter::state::MOD_BASE_DJ;

        const STORE_BUNDLE: [u8; 8] = [0x03, 0x40, 0x57, 0x11, 0x00, 0x20, 0xd4, 0x00];

        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        {
            let tile = engine.device_mut().tile_mut(0, 2).expect("compute tile (0,2)");
            let mut prog = [0u8; 256];
            for chunk in prog.chunks_exact_mut(8) {
                chunk.copy_from_slice(&STORE_BUNDLE);
            }
            tile.write_program(0, &prog);
        }
        engine.set_core_pointer(0, 2, 0, 0x70400);
        engine.set_core_modifier(0, 2, MOD_BASE_DJ, 0x7700);
        engine.enable_core(0, 2);

        // MM2S channel 0 (flat channel index 2) reading bank 0 (0x400..).
        engine
            .device_mut()
            .array
            .dma_engine_mut(0, 2)
            .unwrap()
            .configure_bd(0, BdConfig::simple_1d(0x400, 128))
            .unwrap();
        {
            let tile = engine.device_mut().tile_mut(0, 2).unwrap();
            tile.dma_channels[2].running = true;
            tile.dma_channels[2].start_queue = 0;
        }

        // Mem-module trace on tile (0,2): mode=EventTime(0), start=TRUE(1),
        // slot 0 = DMA_MM2S_0_MEMORY_STARVATION (41).
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = (0u32 << 24) | (1u32 << 16) | 0u32;
            tile.mem_trace.write_register(0x00, ctrl0);
            tile.mem_trace.write_register(0x10, 41u32);
        }

        // Drain the MM2S output every cycle so it stays genuinely mid-transfer
        // (a backpressured-on-its-own-stream channel declares no bank demand,
        // same rationale as the S2MM stream_in feed above).
        for _ in 0..400 {
            engine.step();
            let dma = engine.device_mut().array.dma_engine_mut(0, 2).unwrap();
            while dma.pop_stream_out().is_some() {}
        }

        let encoded = engine.device().array.tile(0, 2).mem_trace.encoded_bytes_len();
        assert!(
            encoded > 8,
            "MM2S channel 0 contending bank 0 with the core must lose arbitration on some \
             cycle and raise DMA_MM2S_0_MEMORY_STARVATION; trace has only {encoded} encoded \
             bytes (just the Start marker) -- event never fired"
        );
    }

    /// Task 7 review fix: a mem-module perf counter armed by
    /// DMA_S2MM_0_MEMORY_BACKPRESSURE's RISING edge (its configured
    /// start_event) must NOT re-arm on the event's FALLING edge (the
    /// backpressure clearing). `PerfCounterBank::handle_event` is
    /// polarity-blind -- it only checks "does this id match my
    /// start/stop/reset event" -- so the coordinator itself must gate the
    /// call to the rising edge only, exactly like `notify_mem_trace_level`
    /// already gates its own timer/edge-detector/halt side effects (see its
    /// doc comment: "on a falling edge the event is not firing, so only the
    /// trace state changes").
    ///
    /// Reuses the S2MM/core bank-0 contention fixture from the sibling test
    /// above. Configures mem-module counter 0 with start_event=39 (S2MM ch0
    /// backpressure) and stop_event=RSVD_103 (an id nothing in this scenario
    /// ever fires naturally) purely as a test-controlled kill switch: the
    /// moment the counter is first observed Active (proving the real
    /// backpressure event's rising edge fired and armed it), the test
    /// force-stops it directly via `handle_event` -- entirely outside the
    /// coordinator -- so the counter is deterministically Stopped WHILE
    /// hw_id 39's level is still asserted (no new rising edge has fired
    /// since). Edges for one requester identity strictly alternate (rising,
    /// falling, rising, ...), so the NEXT thing that can touch hw_id 39 is
    /// channel 0's OWN falling edge -- it cannot be a fresh rising edge
    /// without a falling edge in between. Before the fix, that falling
    /// edge's unconditional `handle_event(39)` call incorrectly re-arms the
    /// counter (Stopped -> Active, since `handle_event` cannot tell "the
    /// start event cleared" from "the start event fired again"); after the
    /// fix it does nothing, so the counter stays Stopped.
    ///
    /// Also configures `mem_trace` on the SAME hw_id (39 only, in no other
    /// slot) as an independent oracle for "an edge just fired": since
    /// nothing else this trace unit is configured for can grow its buffer,
    /// `encoded_bytes_len()` increasing pins the exact step of the channel's
    /// next edge -- which the alternation argument above guarantees is the
    /// falling edge -- so the assertion can fire at exactly that step,
    /// before a later, legitimately-rearming second rising edge has a
    /// chance to occur and mask the bug.
    #[test]
    fn perf_counter_armed_by_backpressure_start_does_not_rearm_on_falling_edge() {
        use crate::device::dma::BdConfig;
        use crate::device::dma::StreamData;
        use crate::interpreter::state::MOD_BASE_DJ;
        use xdna_archspec::aie2::trace_events::mem_events;

        const STORE_BUNDLE: [u8; 8] = [0x03, 0x40, 0x57, 0x11, 0x00, 0x20, 0xd4, 0x00];
        const BACKPRESSURE: u32 = mem_events::DMA_S2MM_0_MEMORY_BACKPRESSURE as u32;

        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        {
            let tile = engine.device_mut().tile_mut(0, 2).expect("compute tile (0,2)");
            let mut prog = [0u8; 256];
            for chunk in prog.chunks_exact_mut(8) {
                chunk.copy_from_slice(&STORE_BUNDLE);
            }
            tile.write_program(0, &prog);
        }
        engine.set_core_pointer(0, 2, 0, 0x70400);
        engine.set_core_modifier(0, 2, MOD_BASE_DJ, 0x7700);
        engine.enable_core(0, 2);

        engine
            .device_mut()
            .array
            .dma_engine_mut(0, 2)
            .unwrap()
            .configure_bd(0, BdConfig::simple_1d(0x400, 128))
            .unwrap();
        {
            let tile = engine.device_mut().tile_mut(0, 2).unwrap();
            tile.dma_channels[0].running = true;
            tile.dma_channels[0].start_queue = 0;
        }

        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            // Mem-module perf counter 0: start=DMA_S2MM_0_MEMORY_BACKPRESSURE
            // (39), stop=RSVD_103 (inert -- the test's own kill switch;
            // nothing else in this scenario ever fires event 103).
            let ctrl0 = BACKPRESSURE | ((mem_events::RSVD_103 as u32) << 8);
            tile.mem_perf_counters.write_control_start_stop(ctrl0, 0, 1, 7);
            // Mem-module trace: mode=EventTime(0), start=TRUE(1), slot 0 = 39
            // only -- the "edge just fired" oracle described above.
            let trace_ctrl0 = (0u32 << 24) | (1u32 << 16) | 0u32;
            tile.mem_trace.write_register(0x00, trace_ctrl0);
            tile.mem_trace.write_register(0x10, BACKPRESSURE);
        }

        let mut armed = false;
        let mut trace_len_at_arm = 0usize;
        let mut saw_falling_edge = false;
        for _ in 0..400 {
            let dma = engine.device_mut().array.dma_engine_mut(0, 2).unwrap();
            while dma.stream_in_len() < 8 {
                dma.push_stream_in(StreamData { data: 0xA5, tlast: false, channel: 0 });
            }
            engine.step();

            let tile = engine.device_mut().array.tile_mut(0, 2);
            if !armed {
                if tile.mem_perf_counters.is_active(0) {
                    // First real rising edge observed -- hw_id 39's level is
                    // still asserted (no falling edge has fired yet).
                    // Force-stop right here via the inert kill switch: this
                    // is test setup (a direct API call, not the
                    // coordinator), not the code under test.
                    tile.mem_perf_counters.handle_event(mem_events::RSVD_103);
                    assert!(
                        !tile.mem_perf_counters.is_active(0),
                        "test setup: the kill switch must actually stop the counter"
                    );
                    trace_len_at_arm = tile.mem_trace.encoded_bytes_len();
                    armed = true;
                }
            } else if !saw_falling_edge && tile.mem_trace.encoded_bytes_len() > trace_len_at_arm {
                // The trace oracle just saw hw_id 39's next edge -- per the
                // alternation argument, this MUST be the falling edge.
                saw_falling_edge = true;
                assert!(
                    !tile.mem_perf_counters.is_active(0),
                    "a counter stopped mid-span must stay stopped through the channel's own \
                     falling edge -- a falling edge must never re-arm a start-triggered counter"
                );
                break;
            }
        }

        assert!(armed, "S2MM ch0 must lose arbitration on some cycle (same contention as the sibling test)");
        assert!(
            saw_falling_edge,
            "S2MM ch0's own backpressure span must end within the run (the arbiter's \
             anti-starvation guarantee bounds the wait) so the falling-edge check actually runs"
        );
    }

    #[test]
    fn test_set_core_pc() {
        let mut engine = InterpreterEngine::new_npu1();

        engine.set_core_pc(0, 2, 0x1000);
        let ctx = engine.core_context(0, 2).unwrap();
        assert_eq!(ctx.pc(), 0x1000);
    }

    #[test]
    fn test_step_no_enabled_cores() {
        let mut engine = InterpreterEngine::new_npu1();

        engine.step();

        // Should halt since no cores enabled
        assert_eq!(engine.status(), EngineStatus::Halted);
        assert_eq!(engine.total_cycles(), 1);
    }

    #[test]
    fn test_step_with_enabled_core() {
        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        // Enable a core and write NOP to program memory
        engine.enable_core(0, 2);

        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00, 0x00, 0x00, 0x00]);
        }

        engine.step();

        assert_eq!(engine.status(), EngineStatus::Running);
        assert_eq!(engine.total_cycles(), 1);

        // Check that PC advanced
        let ctx = engine.core_context(0, 2).unwrap();
        assert_eq!(ctx.pc(), 4);
    }

    #[test]
    fn test_run_limited_cycles() {
        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        engine.enable_core(0, 2);

        // Write many NOPs
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 1024]);
        }

        let cycles = engine.run(10);

        assert_eq!(cycles, 10);
        assert_eq!(engine.total_cycles(), 10);
    }

    #[test]
    fn test_pause_resume() {
        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        engine.enable_core(0, 2);
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 1024]);
        }

        engine.step();
        assert_eq!(engine.status(), EngineStatus::Running);

        engine.pause();
        assert_eq!(engine.status(), EngineStatus::Paused);

        engine.resume();
        assert_eq!(engine.status(), EngineStatus::Ready);
    }

    #[test]
    fn test_reset() {
        let mut engine = InterpreterEngine::new_npu1();

        engine.enable_core(0, 2);
        engine.set_core_pc(0, 2, 0x1000);

        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 64]);
        }

        engine.run(5);

        engine.reset();

        assert_eq!(engine.status(), EngineStatus::Ready);
        assert_eq!(engine.total_cycles(), 0);
        assert_eq!(engine.enabled_cores(), 0);
    }

    #[test]
    fn test_reset_for_new_context_preserves_core_id() {
        // ExecutionContext::reset() does `*self = Self::new()`, which
        // calls `new_for_tile(0, 0)` and zeros the CORE_ID register for
        // every core regardless of position. Real HW's CORE_ID is wired
        // to the tile's position and never changes -- and AIE kernels
        // routinely branch on it. Forgetting to re-tag the identity
        // after reset turned every post-reset run into a "this is the
        // (0,0) tile" simulation, which suppressed real INSTR_VECTOR /
        // LOCK_STALL events whose firing depended on the kernel's
        // per-tile branch -- the j>=8 sweep caught this when fresh
        // workers (correct CORE_ID) produced different traces than
        // post-reset workers (zeroed CORE_ID).
        use crate::interpreter::state::CORE_ID_REG_INDEX;
        let mut engine = InterpreterEngine::new_npu1();

        let pre = engine
            .core_context(1, 2)
            .expect("core (1,2) exists")
            .scalar
            .read(CORE_ID_REG_INDEX);
        assert_eq!(pre, (1 << 16) | 2, "fresh CORE_ID for (1,2) should be 0x10002");

        engine.reset_for_new_context();

        let post = engine
            .core_context(1, 2)
            .expect("core (1,2) exists")
            .scalar
            .read(CORE_ID_REG_INDEX);
        assert_eq!(post, (1 << 16) | 2, "CORE_ID for (1,2) must survive reset; got 0x{post:08x}");
    }

    #[test]
    fn test_reset_for_new_context_clears_perf_counter_values() {
        // Performance counters (PERF_CNT_N) emit trace events when their
        // running count reaches `event_value`. On real hw_context reset
        // they zero out -- if we leave them at their prior count, the
        // next run hits the threshold N cycles earlier than a fresh run
        // would, emitting phantom PERF_CNT events before the new run's
        // workload has actually generated any counted events. The j1
        // sweep saw this as 3 extra PERF_CNT_2 events in batches that
        // followed a high-activity batch on the same runner session
        // (e.g., j1 batch_01=11 vs j4 batch_01=8 in add_one_using_dma).
        let mut engine = InterpreterEngine::new_npu1();

        let tile = engine.device_mut().tile_mut(0, 2).expect("tile (0,2) exists on NPU1");
        tile.core_perf_counters.write_counter(0, 12345);
        tile.mem_perf_counters.write_counter(1, 67890);

        engine.reset_for_new_context();

        let tile = engine.device_mut().tile_mut(0, 2).expect("tile (0,2) exists on NPU1");
        assert_eq!(
            tile.core_perf_counters.read_counter(0),
            0,
            "core_perf_counters[0] must zero on reset_for_new_context"
        );
        assert_eq!(
            tile.mem_perf_counters.read_counter(1),
            0,
            "mem_perf_counters[1] must zero on reset_for_new_context"
        );
    }

    #[test]
    fn test_reset_for_new_context_clears_pending_mem_trace() {
        // mem_trace_pending buffers memory-module trace events (lock
        // acquire/release, DMA) until the coordinator drains them at
        // each cycle boundary. Events pushed near end-of-run can outlive
        // the final drain; if reset_for_new_context leaves them in place
        // they reappear at cycle 0 of the next hw_context, producing
        // phantom events that bias trace-comparison event counts. This
        // showed up as a 3-event divergence between j=1 and j>1 sweep
        // runs whenever a small batch followed a high-activity batch
        // on the same runner session.
        use crate::interpreter::state::EventType;
        let mut engine = InterpreterEngine::new_npu1();

        let tile = engine.device_mut().tile_mut(0, 2).expect("tile (0,2) exists on NPU1");
        tile.mem_trace_pending.push((1000, EventType::LockAcquire { lock_id: 0 }));
        tile.mem_trace_pending.push((1001, EventType::LockRelease { lock_id: 0 }));

        engine.reset_for_new_context();

        let drained = engine.device_mut().array.drain_mem_trace_events();
        assert!(
            drained.is_empty(),
            "reset_for_new_context must clear pending mem-trace events, but found {drained:?}"
        );
    }

    #[test]
    fn test_multiple_cores() {
        let mut engine = InterpreterEngine::new_npu1();

        // Enable multiple cores
        engine.enable_core(0, 2);
        engine.enable_core(1, 2);
        engine.enable_core(0, 3);

        assert_eq!(engine.enabled_cores(), 3);

        // Write NOPs to all their program memories
        for &(col, row) in &[(0, 2), (1, 2), (0, 3)] {
            if let Some(tile) = engine.device_mut().tile_mut(col, row) {
                tile.write_program(0, &[0x00u8; 64]);
            }
        }

        engine.step();

        assert_eq!(engine.active_cores(), 3);
    }

    #[test]
    fn test_auto_run() {
        let mut engine = InterpreterEngine::new_npu1();

        assert!(!engine.is_auto_run());

        engine.set_auto_run(true);
        assert!(engine.is_auto_run());

        engine.set_auto_run(false);
        assert!(!engine.is_auto_run());
    }

    #[test]
    fn test_status_string() {
        let engine = InterpreterEngine::new_npu1();
        assert_eq!(engine.status_string(), "Ready");
    }

    #[test]
    fn test_dma_integration() {
        use crate::device::dma::BdConfig;

        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        // Configure a BD on tile (1, 2) using MM2S channel (channel 2)
        // Use 64-byte transfer: at 4 bytes/cycle, takes 16 cycles to complete
        // This ensures the transfer is still in progress after the first step
        if let Some(dma) = engine.device_mut().array.dma_engine_mut(1, 2) {
            dma.configure_bd(0, BdConfig::simple_1d(0x100, 64)).unwrap();
        }

        // Simulate DmaStart by setting tile channel state (channel 2 = MM2S)
        if let Some(tile) = engine.device_mut().tile_mut(1, 2) {
            tile.dma_channels[2].running = true;
            tile.dma_channels[2].start_queue = 0; // BD index 0
        }

        // Initial state: DMA should not be active on engine yet
        {
            let dma = engine.device().array.dma_engine(1, 2).unwrap();
            assert!(!dma.channel_active(2));
        }

        // Step once - should sync start request and begin transfer
        engine.step();

        // DMA engine should now be active (transfer in progress)
        {
            let dma = engine.device().array.dma_engine(1, 2).unwrap();
            assert!(dma.channel_active(2), "DMA should be active after first step");
        }

        // Step more - 64 bytes at 4 bytes/cycle = 16 cycles
        // (Currently no setup overhead since timing isn't integrated yet).
        // Drain MM2S output each step; without it the 4-deep slave-port
        // FIFO fills after a few cycles and the DMA stalls indefinitely
        // because no consumer is wired up in this isolated test.
        for i in 0..50 {
            engine.step();
            {
                let dma = engine.device_mut().array.dma_engine_mut(1, 2).unwrap();
                while dma.pop_stream_out().is_some() {}
            }

            let dma = engine.device().array.dma_engine(1, 2).unwrap();
            let state = dma.channel_state(2);

            // Break when complete
            if matches!(state, ChannelState::Idle) {
                break;
            }

            // Safety check - shouldn't take more than ~20 cycles
            if i > 30 {
                let stats = dma.channel_stats(2).unwrap();
                panic!(
                    "DMA taking too long. State: {:?}, bytes transferred: {}, active: {}",
                    state,
                    stats.bytes_transferred,
                    dma.channel_active(2)
                );
            }
        }

        // Tile channel should show complete
        {
            let tile = engine.device().array.tile(1, 2);
            assert!(!tile.dma_channels[2].running, "Tile channel should show complete");
        }
    }

    #[test]
    fn test_host_memory_access() {
        let mut engine = InterpreterEngine::new_npu1();

        // Write some data to host memory
        engine.host_memory_mut().write_bytes(0x1000, &[1, 2, 3, 4]);

        // Read it back
        let mut buf = [0u8; 4];
        engine.host_memory().read_bytes(0x1000, &mut buf);
        assert_eq!(buf, [1, 2, 3, 4]);
    }

    // --- Cycle-accurate mode tests ---

    #[test]
    fn test_engine_is_always_cycle_accurate() {
        let engine = InterpreterEngine::new_npu1();

        assert!(engine.is_cycle_accurate());
        assert_eq!(engine.status(), EngineStatus::Ready);
        assert_eq!(engine.total_cycles(), 0);
    }

    #[test]
    fn test_all_execution_is_cycle_accurate() {
        let engine1 = InterpreterEngine::new_npu1();
        assert!(engine1.is_cycle_accurate());

        let engine2 = InterpreterEngine::new_npu2();
        assert!(engine2.is_cycle_accurate());
    }

    #[test]
    fn test_timing_context_enabled() {
        let engine = InterpreterEngine::new_npu1();

        // Core contexts should have timing enabled
        let ctx = engine.core_context(0, 2).unwrap();
        assert!(ctx.has_timing(), "Cores should have timing context");
    }

    #[test]
    fn test_cycle_accurate_step() {
        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        // Enable a core and write NOP to program memory
        engine.enable_core(0, 2);

        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00, 0x00, 0x00, 0x00]);
        }

        engine.step();

        assert_eq!(engine.status(), EngineStatus::Running);
        assert_eq!(engine.total_cycles(), 1);

        // Check that PC advanced
        let ctx = engine.core_context(0, 2).unwrap();
        assert_eq!(ctx.pc(), 4);
        assert_eq!(ctx.cycles, 1);
    }

    #[test]
    fn test_npu2_is_cycle_accurate() {
        let engine = InterpreterEngine::new_npu2();
        assert!(engine.is_cycle_accurate());
    }

    // --- Perf counter → trace unit wiring test (Task 1, A.2) ---

    /// Verifies that threshold firings from PerfCounterBank are routed to the
    /// tile's trace unit.
    ///
    /// Before Task 1, the coordinator discards the Vec<usize> returned by
    /// tick_*_cycles(). This test catches that gap: it configures a perf
    /// counter to fire at threshold=5, maps PERF_CNT_0 (hw_id=5) to trace
    /// slot 0, and asserts the trace unit has recorded data after running.
    ///
    /// The test should FAIL before the wiring is in place (Step 1.3).
    #[test]
    fn test_perfcnt_threshold_routes_to_trace_unit() {
        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        // Enable core (0,2) with enough NOPs to run 20+ cycles without halting.
        // NOPs advance PC by 4 bytes; 512 bytes -> 128 cycles of runtime.
        engine.enable_core(0, 2);
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 512]);
        }

        // Configure perf counter 0 on tile (0,2) core module:
        //   start_event = TRUE (id=1), stop_event = NONE (id=0), threshold = 5.
        // write_control_start_stop(value, counter_lo, counter_hi, event_width)
        // bits [6:0]=start0, [14:8]=stop0
        //
        // No manual handle_event(1) here: the coordinator's Phase 3e fires
        // TRUE every cycle into perf counter banks, which is what arms a
        // TRUE-started counter from Idle->Active. Verifying the fix for
        // #354 means relying on that wiring rather than poking it manually.
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = 1u32 | (0u32 << 8); // start=TRUE(1), stop=NONE(0)
            tile.core_perf_counters.write_control_start_stop(ctrl0, 0, 1, 7);
            tile.core_perf_counters.write_event_value(0, 5);
        }

        // Configure trace unit on tile (0,2) core module:
        //   mode = EventTime (0), start_event = TRUE (1), stop_event = NONE (0)
        //   slot 0 = PERF_CNT_0 (hw_id = 5)
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            // Trace_Control0: [31:24]=stop=0, [23:16]=start=1, [1:0]=mode=0
            let ctrl0 = (0u32 << 24) | (1u32 << 16) | 0u32;
            tile.core_trace.write_register(0x00, ctrl0);
            // Trace_Event0: slot0 = PERF_CNT_0 = 5
            tile.core_trace.write_register(0x10, 5u32);
        }

        // The coordinator fires TRUE(1) on the trace unit in Phase 3c, which
        // will transition it from Idle to Running on the first step.

        // Run 20 cycles: threshold=5 fires at cycle 5 (counter started at 0),
        // and the wiring should route PERF_CNT_0(5) into the trace unit.
        let _cycles = engine.run(20);

        // After Task 1, the trace unit byte_buffer must be non-empty: at least
        // one frame was encoded for the cycle-5 threshold event.
        // After Task 1 wiring, the trace must hold MORE than just the 8-byte
        // start marker (0xF0 + 7 timer bytes that encode_start() pushes).
        // A threshold event at cycle 5 adds a Single0 frame (1 byte), so
        // wired correctly the byte count will be > 8.
        //
        // Without the wiring (coordinator discards Vec return), only the
        // start marker is in the buffer: exactly 8 bytes, never a slot frame.
        let tile = engine.device().array.tile(0, 2);
        let encoded = tile.core_trace.encoded_bytes_len();
        assert!(
            encoded > 8,
            "trace has only {} encoded bytes (just the start marker): \
             perfcnt threshold was not routed to trace unit",
            encoded
        );
    }

    /// Regression for #354: a perf counter configured with start_event =
    /// ACTIVE_CORE (the perfcnt anchor scheme that mlir-trace-inject emits)
    /// must transition Idle->Active and reach its threshold without any
    /// manual handle_event() poke. Before the Phase 3e wiring fix, the
    /// counter sat in Idle forever and never fired, which is why HW
    /// captured PERF_CNT_0 anchor pulses but EMU captured zero.
    ///
    /// Setup matches the mlir-trace-inject perfcnt config block on a
    /// shorter threshold:
    ///   Performance_Control0 = 0x1C  -> Cnt0_Start_Event = ACTIVE_CORE
    ///   Performance_Control2 = 0x05  -> Cnt0_Reset_Event = PERF_CNT_0 (self-reset)
    ///   Performance_Counter0_Event_Value = 5 (vs production 1024)
    /// Then run 12 cycles and assert the trace unit recorded at least
    /// two PERF_CNT_0 firings (the self-reset cycle gives 5-cycle period:
    /// fires at cycle 5 and cycle 11 -- exact count depends on stop ordering).
    #[test]
    fn test_perfcnt_active_core_start_fires_without_manual_arm() {
        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        engine.enable_core(0, 2);
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 512]);
        }

        // Perf counter 0: start=ACTIVE_CORE(0x1C), reset=PERF_CNT_0(5), threshold=5.
        // No manual handle_event() call -- the coordinator's Phase 3e wiring
        // is what arms the counter. This is the bug reproducer for #354.
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = 0x1Cu32 | (0u32 << 8); // start=ACTIVE_CORE, stop=NONE
            tile.core_perf_counters.write_control_start_stop(ctrl0, 0, 1, 7);
            tile.core_perf_counters.write_event_value(0, 5);
            // reset_event[0] = PERF_CNT_0 (5)
            tile.core_perf_counters.write_control_reset(5u32, 7);
        }

        // Trace unit on tile (0,2) core: mode=0, start=TRUE(1), slot 0 = PERF_CNT_0(5).
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = (0u32 << 24) | (1u32 << 16) | 0u32;
            tile.core_trace.write_register(0x00, ctrl0);
            tile.core_trace.write_register(0x10, 5u32);
        }

        // Run 12 cycles. With period=5 and self-reset, the counter should
        // fire PERF_CNT_0 at cycle 5 and cycle 10 (5-cycle interval after
        // reset).
        let _cycles = engine.run(12);

        // The trace unit should have encoded both PERF_CNT_0 firings as
        // Single0 frames (1 byte each) on top of the 8-byte Start marker.
        // Pre-fix this would be exactly 8 bytes (no firings ever land).
        let tile = engine.device().array.tile(0, 2);
        let encoded = tile.core_trace.encoded_bytes_len();
        assert!(
            encoded >= 10,
            "expected >= 10 encoded bytes (start marker + 2x Single0 PERF_CNT_0 \
             frames); got {} bytes -- ACTIVE_CORE-started counter never fired \
             (#354 regression)",
            encoded
        );
    }

    /// A clock-gated module has no clock, so its performance counters must
    /// freeze: "no clock, no tick". This exercises the column gate (the
    /// dominant tier) -- gating column 0 must stop tile (0,2)'s core perf
    /// counter from advancing, even though the simulation keeps stepping
    /// (a keep-alive core in the ungated column 1 drives cycles forward).
    ///
    /// Pre-fix, Phase 3e ticks every tile's perf counters unconditionally,
    /// so the gated counter keeps counting -- a bug a binary could observe by
    /// gating a column and reading Performance_Counter0.
    #[test]
    fn column_gate_freezes_core_perf_counter() {
        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        // Keep-alive core in column 1: run() steps max_cycles unless Halted,
        // so a running core guarantees cycles advance while column 0 is gated.
        engine.enable_core(1, 2);
        if let Some(tile) = engine.device_mut().tile_mut(1, 2) {
            tile.write_program(0, &[0x00u8; 512]);
        }

        // Core perf counter 0 on (0,2): start=TRUE, event_value=0 (free-running
        // count -- never self-fires or resets). Armed Active up front.
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = 1u32 | (0u32 << 8); // start=TRUE(1), stop=NONE(0)
            tile.core_perf_counters.write_control_start_stop(ctrl0, 0, 1, 7);
            tile.core_perf_counters.write_event_value(0, 0);
            tile.core_perf_counters.handle_event(1); // arm -> Active
        }

        // Sanity: while ungated the counter advances every cycle.
        engine.run(5);
        let after_ungated = engine.device().array.tile(0, 2).core_perf_counters.read_counter(0);
        assert!(after_ungated > 0, "sanity: ungated core perf counter must advance, got {}", after_ungated);

        // Gate column 0 (clear Column_Clock_Control bit 0 on the shim row).
        engine.device_mut().array.clock_mut().write_register(0, 0, 0x000FFF20, 0x0);
        assert!(!engine.device().array.clock().is_column_active(0), "precondition: column 0 gated");

        engine.run(10);
        let after_gated = engine.device().array.tile(0, 2).core_perf_counters.read_counter(0);
        assert_eq!(
            after_gated, after_ungated,
            "clock-gated column must freeze the perf counter (no clock, no tick): \
             advanced from {} to {} across 10 gated cycles",
            after_ungated, after_gated
        );
    }

    /// Module-granular freeze: gating only the Core module (column still
    /// active) must freeze the core perf counter while the Memory module's
    /// counter on the same tile keeps ticking. This proves the gate is
    /// per-module, not just per-column, and that the two banks are independent.
    #[test]
    fn core_module_gate_freezes_core_counter_but_not_mem_counter() {
        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        // Keep-alive core in column 1 so cycles advance.
        engine.enable_core(1, 2);
        if let Some(tile) = engine.device_mut().tile_mut(1, 2) {
            tile.write_program(0, &[0x00u8; 512]);
        }

        // Both the core and memory perf counter 0 on (0,2): start=TRUE,
        // free-running, armed Active.
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = 1u32 | (0u32 << 8); // start=TRUE(1), stop=NONE(0)
            tile.core_perf_counters.write_control_start_stop(ctrl0, 0, 1, 7);
            tile.core_perf_counters.write_event_value(0, 0);
            tile.core_perf_counters.handle_event(1);
            tile.mem_perf_counters.write_control_start_stop(ctrl0, 0, 1, 7);
            tile.mem_perf_counters.write_event_value(0, 0);
            tile.mem_perf_counters.handle_event(1);
        }

        // Sanity: both advance while fully ungated.
        engine.run(5);
        let (core_pre, mem_pre) = {
            let tile = engine.device().array.tile(0, 2);
            (tile.core_perf_counters.read_counter(0), tile.mem_perf_counters.read_counter(0))
        };
        assert!(
            core_pre > 0 && mem_pre > 0,
            "sanity: both counters advance ungated ({}, {})",
            core_pre,
            mem_pre
        );

        // Gate ONLY the Core module on (0,2): MCC bits [SS=1, Mem=1, Core=0]
        // -> 0b011. Column 0 stays active. Compute MCC offset = 0x60000.
        engine.device_mut().array.clock_mut().write_register(0, 2, 0x0006_0000, 0b011);
        assert!(!engine.device().array.clock().is_module_active(
            0,
            2,
            crate::device::clock_control::ModuleKind::Core
        ));
        assert!(engine.device().array.clock().is_module_active(
            0,
            2,
            crate::device::clock_control::ModuleKind::Memory
        ));

        engine.run(10);
        let (core_post, mem_post) = {
            let tile = engine.device().array.tile(0, 2);
            (tile.core_perf_counters.read_counter(0), tile.mem_perf_counters.read_counter(0))
        };
        assert_eq!(
            core_post, core_pre,
            "Core-module gate must freeze core counter ({} -> {})",
            core_pre, core_post
        );
        assert_eq!(
            mem_post,
            mem_pre + 10,
            "Memory module still clocked: mem counter must keep ticking ({} -> {})",
            mem_pre,
            mem_post
        );
    }

    /// Verifies AM020 "one frame per cycle" invariant when a perf counter
    /// fires in the same cycle as another slot event (here, TRUE).
    ///
    /// HW emits a single Multiple frame whose mask carries both slot bits.
    /// The emulator must do the same; if Phase 3e (perfcnt route) ran AFTER
    /// Phase 3f (commit_cycle), the perfcnt notify would land in the next
    /// cycle's pending_slot_mask, splitting one HW frame into two emulator
    /// frames.
    ///
    /// Setup: counter 0 fires at cycle 4 (threshold=5, 5 ticks across cycles
    /// 0..4). Trace slots: slot 0 = PERF_CNT_0 (hw_id=5), slot 1 = TRUE
    /// (hw_id=1). After 6 steps the byte buffer contains an 8-byte Start
    /// marker plus a stream of Single0 TRUE frames, with one Multiple0
    /// frame at cycle 4 carrying mask=0b11.
    ///
    /// Multiple0 byte format (from encode_multiple):
    ///   byte0 = 0xC0 | (mask >> 4)
    ///   byte1 = ((mask & 0x0F) << 4) | (delta & 0x0F)
    /// For mask=0b00000011: byte0=0xC0, byte1=0x30 | delta.
    #[test]
    fn test_perfcnt_and_true_same_cycle_coalesce_into_multiple_frame() {
        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        engine.enable_core(0, 2);
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 256]);
        }

        // Perf counter 0: start=TRUE(1), threshold=5. Fires at cycle 4.
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = 1u32 | (0u32 << 8);
            tile.core_perf_counters.write_control_start_stop(ctrl0, 0, 1, 7);
            tile.core_perf_counters.write_event_value(0, 5);
            tile.core_perf_counters.handle_event(1); // arm
        }

        // Trace: start=TRUE(1), slot 0 = PERF_CNT_0 (5), slot 1 = TRUE (1).
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = (0u32 << 24) | (1u32 << 16) | 0u32;
            tile.core_trace.write_register(0x00, ctrl0);
            // Trace_Event0: slot0=5 (PERF_CNT_0), slot1=1 (TRUE), slot2=0, slot3=0
            let evt0 = 5u32 | (1u32 << 8);
            tile.core_trace.write_register(0x10, evt0);
        }

        // 6 steps covers: trace start at cycle 0, TRUE frames at 1..3, the
        // coalesced frame at cycle 4, plus one more cycle so cycle 4's frame
        // is committed (commit_cycle in same step runs after notify, so the
        // cycle-4 frame is committed at end of cycle 4 itself).
        let _ = engine.run(6);

        let bytes = engine.device().array.tile(0, 2).core_trace.encoded_bytes().to_vec();

        // Skip the 8-byte Start marker (byte 0 = 0xF0 + 7 timer bytes).
        assert_eq!(
            bytes.first().copied(),
            Some(0xF0),
            "expected Start marker as first byte; got buffer {:02X?}",
            bytes
        );
        assert!(bytes.len() > 8, "buffer must hold more than just the Start marker; got {:02X?}", bytes);

        // Walk the post-marker bytes and locate any Multiple0 frame.
        // Multiple0 has top 4 bits = 0b1100, so byte0 & 0xF0 == 0xC0.
        // Single0 has top bit = 0 (byte < 0x80), Single1 = 0x80..0x9F,
        // Single2 = 0xA0..0xBF.
        //
        // We expect exactly one Multiple0 frame (at cycle 4) with mask=0b11.
        let mut multiple_count = 0usize;
        let mut multiple_with_mask_3 = 0usize;
        let mut i = 8; // past Start marker
        while i < bytes.len() {
            let b = bytes[i];
            if b & 0xF0 == 0xC0 {
                // Multiple0: 2 bytes
                multiple_count += 1;
                let mask_hi = b & 0x0F;
                let byte1 = bytes.get(i + 1).copied().unwrap_or(0);
                let mask_lo = (byte1 >> 4) & 0x0F;
                let mask = (mask_hi << 4) | mask_lo;
                if mask == 0b0000_0011 {
                    multiple_with_mask_3 += 1;
                }
                i += 2;
            } else if b & 0xFC == 0xD0 {
                // Multiple1: 3 bytes
                multiple_count += 1;
                i += 3;
            } else if b & 0xFC == 0xD4 {
                // Multiple2: 4 bytes
                multiple_count += 1;
                i += 4;
            } else if b & 0x80 == 0 {
                // Single0: 1 byte
                i += 1;
            } else if b & 0xE0 == 0x80 {
                // Single1: 2 bytes
                i += 2;
            } else if b & 0xE0 == 0xA0 {
                // Single2: 3 bytes
                i += 3;
            } else {
                // Unknown -- stop walking to avoid runaway.
                break;
            }
        }

        assert_eq!(
            multiple_with_mask_3, 1,
            "expected exactly one Multiple frame with mask=0b11 (TRUE+PERF_CNT_0 \
             coalesced at cycle 4); got {} Multiple frames total, {} with mask=0b11. \
             Buffer: {:02X?}",
            multiple_count, multiple_with_mask_3, bytes
        );
    }

    /// Verifies that PC values are threaded from the core EventLog through
    /// the coordinator drain loop into the trace unit's mode-1 (EventPc) encoder.
    ///
    /// Flow under test:
    ///   EventType::InstrVector { pc: 0x100 }
    ///     -> core.context.timing_context().events
    ///     -> coordinator Phase 2 drain
    ///     -> trace::event_pc(&evt.event) = Some(0x100)
    ///     -> notify_core_trace_event(37, cycle, Some(0x100))
    ///     -> TraceUnit::notify_event(37, 0, Some(0x100))
    ///     -> pending_pc = 0x100, pending_slot_mask = 0b1
    ///     -> Phase 3f commit_cycle -> encode_event_pc(mask=1, pc=0x100)
    ///     -> byte_buffer[8..12] = [0xC4, 0x04, 0x01, 0x00]
    ///
    /// The test locks in that PC threading actually flows end-to-end from the
    /// EventType variant through event_pc() to the encoded EventPC frame.
    #[test]
    fn test_instr_vector_pc_threads_into_mode1_event_pc_frame() {
        use crate::interpreter::state::EventType;

        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        // Enable core (0,2) with NOPs so the step doesn't halt.
        engine.enable_core(0, 2);
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 256]);
        }

        // Configure core_trace on tile (0,2) in EventPc mode (mode=1):
        //   start_event = TRUE (hw_id=1), stop_event = NONE (hw_id=0)
        //   slot 0 = INSTR_VECTOR (hw_id=37)
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            // Trace_Control0: stop[31:24]=0, start[23:16]=1, mode[1:0]=1
            let ctrl0 = (0u32 << 24) | (1u32 << 16) | 1u32;
            tile.core_trace.write_register(0x00, ctrl0);
            // Trace_Event0: slot 0 = INSTR_VECTOR = 37
            tile.core_trace.write_register(0x10, 37u32);

            // Pre-arm: fire the start event (TRUE=1) directly so the trace
            // unit transitions Idle->Running before the coordinator's Phase 2.
            // This mirrors what Phase 3c would do one cycle later, but we
            // need it done before Phase 2 drains the synthetic event below.
            //
            // HW pipelines Idle->Running by 1 cycle: notify_event(start, 0)
            // arms the unit; only events from cycle > 0 are recorded. The
            // unmatched-event call at cycle 1 trips the cycle-advance
            // promotion path so state is Running before the drain runs.
            tile.core_trace.notify_event(1, 0, None);
            tile.core_trace.notify_event(0xFF, 1, None);
        }

        // Inject a synthetic InstrVector event at cycle 0 into core (0,2).
        // The coordinator drain loop reads from trace_events_consumed=0,
        // which points to this event, and will call notify_core_trace_event
        // with the pc extracted by trace::event_pc().
        {
            let ctx = engine.core_context_mut(0, 2).unwrap();
            ctx.timing_context_mut().events.record(0, EventType::InstrVector { pc: 0x100 });
        }

        // Run 1 cycle: the drain loop processes the synthetic InstrVector event,
        // calls notify_core_trace_event(37, 0, Some(0x100)), which sets
        // pending_slot_mask=0b1 and pending_pc=0x100. Phase 3f commits the
        // frame: encode_event_pc(mask=1, pc=0x0100) appends 4 bytes.
        let _cycles = engine.run(1);

        let tile = engine.device().array.tile(0, 2);
        let bytes = tile.core_trace.encoded_bytes();

        // Start marker for mode 1 is 0xF1 (not 0xF0).
        assert_eq!(
            bytes.first().copied(),
            Some(0xF1),
            "expected mode-1 Start marker 0xF1 as first byte; got {:02X?}",
            bytes
        );
        assert!(
            bytes.len() >= 12,
            "expected at least 12 bytes (8 start marker + 4 EventPC frame); \
             got {} bytes: {:02X?}",
            bytes.len(),
            bytes
        );

        // bytes 0-7: Start marker (0xF1 + 7 timer bytes in EventPc mode).
        // bytes 8-11: the EventPC frame for the InstrVector event.
        // encode_event_pc(mask=0b00000001, pc14=0x0100):
        //   byte0 = 0b1100_0100 | (mask >> 6 & 0b11) = 0xC4 | 0 = 0xC4
        //   byte1 = (mask & 0b0011_1111) << 2 = (1 & 63) << 2 = 0x04
        //   byte2 = (pc14 >> 8) & 0x3F = (0x100 >> 8) & 0x3F = 0x01
        //   byte3 = pc14 & 0xFF = 0x00
        let frame = &bytes[8..12];
        assert_eq!(
            frame,
            &[0xC4, 0x04, 0x01, 0x00],
            "EventPC frame mismatch: expected [0xC4, 0x04, 0x01, 0x00] \
             (mask=0b1, pc=0x100), got {:02X?}. \
             Full buffer: {:02X?}",
            frame,
            bytes
        );
    }

    /// End-to-end perfcnt -> mode-1 round-trip.
    ///
    /// Drives the full chain that the test_perfcnt_threshold_routes_to_trace_unit
    /// test only covers in mode-0:
    ///   perfcnt threshold fires
    ///     -> coordinator Phase 3e drain
    ///     -> notify_core_trace_event(PERF_CNT_0, cycle, None)  [no PC]
    ///     -> TraceUnit::notify_event in EventPc mode
    ///     -> pending_pc = 0 (sentinel for None)
    ///     -> commit_cycle -> encode_event_pc(mask, pc=0)
    ///     -> 4-byte EventPC frame in byte_buffer
    ///
    /// PERF_CNT_N events stamp the frame with the core's current PC (the
    /// same value HW samples when the threshold fires), so a counter that
    /// crosses threshold mid-execution shows up in mode-1 frames at the
    /// in-flight PC rather than as a PC=0 sentinel. HW emits the same:
    /// real captures of `add_one_using_dma` show PERF_CNT_2 frames stamped
    /// with PCs in the compute body and in the lock-stall pipeline PC
    /// (e.g., 204 and 832), never 0.
    #[test]
    fn perfcnt_threshold_routes_to_mode1_event_pc_frame_with_core_pc() {
        use xdna_archspec::aie2::trace_events::core_events::PERF_CNT_0;

        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        engine.enable_core(0, 2);
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 256]);
        }

        // Perf counter 0: start=TRUE, threshold=5.  Fires at cycle 4 (5 ticks
        // across cycles 0..4 with active-cycle tick).
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = 1u32 | (0u32 << 8); // start=TRUE(1), stop=NONE(0)
            tile.core_perf_counters.write_control_start_stop(ctrl0, 0, 1, 7);
            tile.core_perf_counters.write_event_value(0, 5);
            tile.core_perf_counters.handle_event(1); // arm
        }

        // Trace: mode=EventPc(1), start=TRUE(1), slot 0 = PERF_CNT_0.
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            // Trace_Control0: stop[31:24]=0, start[23:16]=1, mode[1:0]=1
            let ctrl0 = (0u32 << 24) | (1u32 << 16) | 1u32;
            tile.core_trace.write_register(0x00, ctrl0);
            tile.core_trace.write_register(0x10, PERF_CNT_0 as u32);
        }

        // Run enough cycles for the threshold to fire and Phase 3f to commit.
        let _cycles = engine.run(10);

        let tile = engine.device().array.tile(0, 2);
        let bytes = tile.core_trace.encoded_bytes();

        // Start marker = 0xF1 for mode 1.
        assert_eq!(
            bytes.first().copied(),
            Some(0xF1),
            "mode-1 Start marker 0xF1 expected; got {:02X?}",
            bytes
        );

        // After 8-byte start marker, expect at least one EventPC frame.
        // Discriminator: (b & 0b1111_1100) == 0b1100_0100, mask=0b1 (slot 0).
        // The 5th NOP issues at PC=0x14 (each NOP advances PC by 4 bytes,
        // 5 cycles after PC=0 -> PC=0x14), so byte3 = 0x14.
        assert!(
            bytes.len() >= 12,
            "expected >= 12 bytes (8-byte Start + 4-byte EventPC); got {}: {:02X?}",
            bytes.len(),
            bytes
        );
        let frame = &bytes[8..12];
        assert_eq!(
            frame,
            &[0xC4, 0x04, 0x00, 0x14],
            "perfcnt threshold did not produce expected mode-1 EventPC frame \
             (mask=0b1, pc=0x14 at firing cycle); got {:02X?}.  Full buffer: {:02X?}",
            frame,
            bytes
        );
    }

    // ----------------------------------------------------------------------
    // Tile_Control isolation -> NeighborLocks gating
    //
    // The coordinator constructs NeighborLocks by calling
    // build_neighbor_locks_with_isolation. These tests pin the mapping from
    // isolation byte to per-direction Some/None state, which is what the
    // executor's route_lock fallback consumes. East quadrant locks live on
    // the executing tile and are unaffected by isolation.
    // ----------------------------------------------------------------------

    fn lock_vec(n: usize) -> Vec<Lock> {
        (0..n).map(|_| Lock::new(0)).collect()
    }

    #[test]
    fn build_neighbor_locks_no_isolation_passes_all_slices() {
        let mut s = lock_vec(16);
        let mut w = lock_vec(16);
        let mut n = lock_vec(16);
        let nlocks = build_neighbor_locks_with_isolation(
            0,
            Some(s.as_mut_slice()),
            Some(w.as_mut_slice()),
            Some(n.as_mut_slice()),
        );
        assert!(nlocks.south.is_some());
        assert!(nlocks.west.is_some());
        assert!(nlocks.north.is_some());
    }

    #[test]
    fn build_neighbor_locks_each_isolation_bit_hides_only_its_direction() {
        use crate::device::tile::isolation as iso;
        let cases = [(iso::SOUTH, "south"), (iso::WEST, "west"), (iso::NORTH, "north")];
        for (bit, dir) in cases {
            let mut s = lock_vec(16);
            let mut w = lock_vec(16);
            let mut n = lock_vec(16);
            let nlocks = build_neighbor_locks_with_isolation(
                bit,
                Some(s.as_mut_slice()),
                Some(w.as_mut_slice()),
                Some(n.as_mut_slice()),
            );
            // The direction matching the bit must be hidden; others stay
            // available. East/Internal isn't in NeighborLocks at all (own
            // tile), so it's implicitly unaffected.
            match dir {
                "south" => {
                    assert!(nlocks.south.is_none(), "SOUTH bit must hide south slice");
                    assert!(nlocks.west.is_some());
                    assert!(nlocks.north.is_some());
                }
                "west" => {
                    assert!(nlocks.south.is_some());
                    assert!(nlocks.west.is_none(), "WEST bit must hide west slice");
                    assert!(nlocks.north.is_some());
                }
                "north" => {
                    assert!(nlocks.south.is_some());
                    assert!(nlocks.west.is_some());
                    assert!(nlocks.north.is_none(), "NORTH bit must hide north slice");
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn build_neighbor_locks_all_directions_hides_every_slice() {
        use crate::device::tile::isolation as iso;
        let mut s = lock_vec(16);
        let mut w = lock_vec(16);
        let mut n = lock_vec(16);
        let nlocks = build_neighbor_locks_with_isolation(
            iso::ALL_DIRECTIONS,
            Some(s.as_mut_slice()),
            Some(w.as_mut_slice()),
            Some(n.as_mut_slice()),
        );
        assert!(nlocks.south.is_none());
        assert!(nlocks.west.is_none());
        assert!(nlocks.north.is_none());
    }

    #[test]
    fn build_neighbor_locks_passes_through_none_inputs() {
        // When the coordinator has no neighbor at all (edge tile), the
        // input slices are already None. Isolation must not turn None
        // into Some -- it can only hide a present slice.
        let nlocks = build_neighbor_locks_with_isolation(0, None, None, None);
        assert!(nlocks.south.is_none());
        assert!(nlocks.west.is_none());
        assert!(nlocks.north.is_none());
    }

    /// End-to-end: bring up the engine, set tile.isolation = SOUTH on a
    /// compute tile, and verify the gate engages without crashing the
    /// step loop. Pre-isolation behavior would let the executor see the
    /// south neighbor's locks; post-isolation it shouldn't. We can't
    /// directly inspect the in-flight NeighborLocks (the closure owns
    /// it), but a clean step + correct status proves the wiring holds.
    #[test]
    fn engine_step_survives_with_isolation_set() {
        use crate::device::tile::isolation as iso;
        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();
        engine.enable_core(0, 2);
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00, 0x00, 0x00, 0x00]); // NOP
            tile.isolation = iso::ALL_DIRECTIONS;
        }
        engine.step();
        assert_eq!(engine.status(), EngineStatus::Running);
        assert_eq!(engine.total_cycles(), 1);
    }

    #[test]
    fn note_ctrl_packet_ordering_hazard_counts_inflight_packets() {
        // The control-packet ordering hazard detector replaces the deleted
        // flush_ctrl_packets. It records a hazard whenever a packet-switched
        // control packet is still in flight at an NPU instruction boundary --
        // the exact condition under which the removed flush would have
        // force-delivered. With no packet in flight it must stay silent.
        let mut engine = InterpreterEngine::new_npu1();
        engine.note_ctrl_packet_ordering_hazard();
        assert_eq!(engine.ctrl_packet_hazard_count(), 0, "no control packet in flight -> no hazard");

        // Put a control word at a tile's TileCtrl MASTER port -- the control
        // delivery point the reassembler drains (not trace/data traffic).
        let tile_idx = engine.device.array.tile_index(1, 2);
        let ss = &mut engine.device.array.tiles[tile_idx].stream_switch;
        let ctrl_master = ss
            .masters
            .iter()
            .position(|p| matches!(p.port_type, crate::device::stream_switch::PortType::TileCtrl))
            .expect("compute tile must have a TileCtrl master port");
        ss.masters[ctrl_master].push(0x8000_0000);

        engine.note_ctrl_packet_ordering_hazard();
        assert_eq!(
            engine.ctrl_packet_hazard_count(),
            1,
            "a control packet in flight at an instruction boundary must record one hazard"
        );
    }

    #[test]
    fn ctrl_write_to_unknown_offset_sets_slverr_and_suppresses_write() {
        use crate::device::tile::CtrlPacketAction;
        let mut engine = InterpreterEngine::new_npu1();
        // 0x1F200 is verified SubsystemKind::Unknown on a compute tile.
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: 0x1F200,
            value: 0xABCD_1234,
        });
        let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
        assert_eq!(tile.pkt_handler_status & 0x4, 0x4, "SLVERR bit must latch");
        assert!(
            tile.registers_ref().get(&0x1F200).is_none(),
            "undecoded write must be suppressed (not stored)"
        );
        assert!(
            !matches!(engine.status(), EngineStatus::Error),
            "SLVERR is poll-only sticky -- engine must not abort"
        );
    }

    #[test]
    fn ctrl_read_of_unknown_offset_sets_slverr_no_response() {
        use crate::device::tile::CtrlPacketAction;
        let mut engine = InterpreterEngine::new_npu1();
        engine.dispatch_ctrl_action(CtrlPacketAction::ReadRegisters {
            col: 0,
            row: 2,
            offset: 0x1F200,
            count: 2,
            response_id: 7,
        });
        let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
        assert_eq!(tile.pkt_handler_status & 0x4, 0x4, "SLVERR bit must latch");
        assert!(tile.pending_ctrl_response.is_empty(), "undecoded read must not queue a response");
        assert!(!matches!(engine.status(), EngineStatus::Error));
    }

    #[test]
    fn ctrl_write_to_valid_offset_no_slverr_and_applies() {
        use crate::device::tile::CtrlPacketAction;
        let mut engine = InterpreterEngine::new_npu1();
        // 0x400 is compute data memory -- decodes, must NOT SLVERR.
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: 0x400,
            value: 0x0000_0001,
        });
        let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
        assert_eq!(
            tile.pkt_handler_status & 0x4,
            0,
            "decodable offset must NOT set SLVERR (false-positive guard)"
        );
        assert_eq!(
            tile.read_data_u32(0x400),
            Some(0x0000_0001),
            "decodable ctrl write must actually apply (end-to-end, not just no-SLVERR)"
        );
    }

    // -----------------------------------------------------------------------
    // Phase B Unit 1 -- Step 1: routing-gap round-trip tests
    //
    // Spec §5.1 / §5.3: control-packet writes to PC_Event0 (0x32020) and
    // Debug_Control2 (0x32018) must reach core_debug, and a read-back via
    // read_register_pure must return the correct live value. The write side
    // is already wired through apply_tile_local_effects; the read side
    // (read_register_pure) must dispatch to core_debug for live-computed
    // registers (Core_Status, Debug_Status). PC_Event*/Debug_Control* raw
    // storage values are verified to round-trip through the tile.registers
    // HashMap. These tests are the TDD gate: they must fail before the
    // read-side fix and pass after.
    //
    // Register offsets (aie-rt xaiemlgbl_params.h, cross-checked against
    // the AM025 register database aie_registers_aie2.json):
    //   Core_Control    0x32000
    //   Core_Status     0x32004  (read-only; live-computed)
    //   Debug_Control0  0x32010
    //   Debug_Control1  0x32014
    //   Debug_Control2  0x32018
    //   Debug_Status    0x3201C  (read-only; live-computed)
    //   PC_Event0       0x32020  (write → core_debug.pc_event0; raw storage)
    //   PC_Event1       0x32024
    //   PC_Event2       0x32028
    //   PC_Event3       0x3202C
    // -----------------------------------------------------------------------

    /// A ctrl-packet write of PC_Event0 (0x32020: VALID=1 | address=0x184)
    /// must reach core_debug.pc_event0. Verified by reading back through
    /// core_debug.read_register (which is what the pre-execute seam uses)
    /// AND through read_register_pure (which is what OP_READ uses).
    #[test]
    fn ctrl_write_pc_event0_reaches_core_debug() {
        use crate::device::tile::CtrlPacketAction;

        const PC_EVENT0: u32 = 0x32020;
        // VALID (bit 31) | PC_ADDRESS = 0x184 (low 14 bits).
        const VALUE: u32 = 0x8000_0184;

        let mut engine = InterpreterEngine::new_npu1();
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: PC_EVENT0,
            value: VALUE,
        });

        // No SLVERR: the offset decodes.
        let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
        assert_eq!(tile.pkt_handler_status & 0x4, 0, "PC_Event0 write must not set SLVERR");

        // Write side: core_debug must have the armed PC_Event0.
        let read_via_core_debug = tile.core_debug.read_register(PC_EVENT0);
        assert_eq!(
            read_via_core_debug,
            Some(VALUE),
            "core_debug.read_register(0x32020) must return the written value"
        );

        // Read side: read_register_pure must return the same value (raw storage
        // round-trip -- the write is also stored in tile.registers).
        let read_via_pure = tile.read_register_pure(PC_EVENT0);
        assert_eq!(
            read_via_pure, VALUE,
            "read_register_pure(0x32020) must return the written PC_Event0 value"
        );
    }

    /// A ctrl-packet write of Debug_Control2 (0x32018) must reach
    /// core_debug.debug_ctrl2, and read_register_pure must return the value.
    #[test]
    fn ctrl_write_debug_control2_reaches_core_debug() {
        use crate::device::tile::CtrlPacketAction;

        const DEBUG_CONTROL2: u32 = 0x32018;
        // PC_Event_Halt enable (bit 0) -- the arming bit for PC_Event halts.
        const VALUE: u32 = 0x0000_0001;

        let mut engine = InterpreterEngine::new_npu1();
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: DEBUG_CONTROL2,
            value: VALUE,
        });

        let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
        assert_eq!(tile.pkt_handler_status & 0x4, 0, "Debug_Control2 write must not set SLVERR");

        // core_debug must have the PC_Event_Halt enable set.
        let read_via_core_debug = tile.core_debug.read_register(DEBUG_CONTROL2);
        assert_eq!(
            read_via_core_debug,
            Some(VALUE),
            "core_debug.read_register(0x32018) must return the written Debug_Control2 value"
        );

        // read_register_pure round-trip (raw storage).
        let read_via_pure = tile.read_register_pure(DEBUG_CONTROL2);
        assert_eq!(
            read_via_pure, VALUE,
            "read_register_pure(0x32018) must return the written Debug_Control2 value"
        );
    }

    /// Core_Status (0x32004) read via read_register_pure must reflect the live
    /// core_debug state, not stale tile.registers. This is the read-side gap:
    /// Core_Status is a live-computed register (core_debug.read_status()); if
    /// read_register_pure reads tile.registers instead, it returns 0 even when
    /// the core is debug-halted.
    ///
    /// This test arms PC_Event0 + PC_Event_Halt, manually sets the halted flag
    /// (simulating what the pre-execute seam will do), and asserts that
    /// read_register_pure(0x32004) returns a value with DEBUG_HALT (bit 16).
    #[test]
    fn read_register_pure_core_status_reflects_live_debug_halt() {
        const CORE_STATUS: u32 = 0x32004;
        const DEBUG_HALT_BIT: u32 = 1 << 16;

        let mut engine = InterpreterEngine::new_npu1();

        // Enable the core and set the halted flag directly on core_debug.
        // (The pre-execute seam in Step 2 will do this via has_sync_pc_trap_at +
        // request_halt; here we isolate the read-side gap test.)
        {
            let tile = engine.device_mut().tile_mut(0, 2).expect("compute tile (0,2)");
            tile.core_debug.set_enabled(true);
            tile.core_debug.request_halt();
        }

        let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
        assert!(tile.core_debug.is_halted(), "precondition: core must be debug-halted");

        // The live computed Core_Status via core_debug must have DEBUG_HALT set.
        let via_core_debug = tile.core_debug.read_status();
        assert_ne!(
            via_core_debug & DEBUG_HALT_BIT,
            0,
            "core_debug.read_status() must have DEBUG_HALT bit set"
        );

        // read_register_pure must return the same live value, not tile.registers[0x32004]=0.
        let via_pure = tile.read_register_pure(CORE_STATUS);
        assert_ne!(
            via_pure & DEBUG_HALT_BIT,
            0,
            "read_register_pure(Core_Status=0x32004) must reflect live DEBUG_HALT state; \
             got 0x{via_pure:08X} (bit 16 absent -- read-side routing gap)"
        );
    }

    /// Non-debug-reg writes and reads through the same paths are unaffected.
    /// Data memory write at 0x400 still reads back via read_register_pure as
    /// confirmed by the existing `ctrl_write_to_valid_offset_no_slverr_and_applies`
    /// test. This companion test verifies that the core_debug dispatch in
    /// read_register_pure does NOT intercept data-memory offsets.
    #[test]
    fn read_register_pure_non_debug_reg_unaffected() {
        let mut engine = InterpreterEngine::new_npu1();

        // Write a known value to data memory at offset 0x800.
        {
            let tile = engine.device_mut().tile_mut(0, 2).expect("compute tile (0,2)");
            tile.write_data_u32(0x800, 0xDEAD_BEEF);
        }

        let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
        let val = tile.read_register_pure(0x800);
        assert_eq!(
            val, 0xDEAD_BEEF,
            "read_register_pure of data memory must be unaffected by core_debug dispatch"
        );
    }

    // -----------------------------------------------------------------------
    // Phase B Unit 1 -- Step 2: pre-execute PC_Event seam tests
    //
    // Spec §5.1 / §5.3: when a synchronous PC_Event halt is armed (PC_Event0
    // VALID + PC_Event_Halt in Debug_Control2) and the coordinator is about to
    // execute the bundle at that PC, it must halt WITHOUT executing the bundle
    // (before-commit, per G1 silicon observation). The post-execute update_pc
    // must not re-fire the same match on the next step after resume.
    //
    // Async halt paths (host Debug_Control0[0], stall-halt, event-halt) must
    // remain caught at the existing interpreter.rs:181 gate -- they are
    // provably unaffected because the pre-execute seam only consults the
    // core_debug.has_sync_pc_trap_at(pc) query, which returns false for async
    // halt conditions (those set halted = true, caught at :181 before the
    // pre-execute check would even fire).
    // -----------------------------------------------------------------------

    /// PC_Event0 armed at TRAP_PC, PC_Event_Halt enabled: stepping when the
    /// core's next-PC matches TRAP_PC must return DebugHalt WITHOUT advancing
    /// the PC (before-commit). The core must be halted and PC must not advance.
    #[test]
    fn pre_execute_pc_event_halts_before_commit() {
        const TRAP_PC: u32 = 0x00; // arm at address 0 -- the first bundle
        const PC_EVENT0: u32 = 0x32020;
        const DEBUG_CONTROL2: u32 = 0x32018;

        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        // Load 4 NOP bundles so the interpreter has something to execute.
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 16]);
        }
        engine.enable_core(0, 2);

        // Arm PC_Event0 at TRAP_PC via ctrl-packet (the routed path, now fixed in Step 1).
        use crate::device::tile::CtrlPacketAction;
        // VALID (bit 31) | address = TRAP_PC (14 bits).
        let pc_event_value = 0x8000_0000 | (TRAP_PC & 0x3FFF);
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: PC_EVENT0,
            value: pc_event_value,
        });
        // PC_Event_Halt: Debug_Control2 bit 0.
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: DEBUG_CONTROL2,
            value: 0x1,
        });

        // Step once. The pre-execute seam must fire: the next PC (0) matches
        // the armed PC_Event0, so the coordinator must halt before executing.
        engine.step();

        // PC must NOT have advanced (bundle did not execute).
        let ctx = engine.core_context(0, 2).expect("core (0,2)");
        assert_eq!(ctx.pc(), TRAP_PC, "PC must not advance on pre-execute halt (bundle did not commit)");

        // Core must be debug-halted.
        let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
        assert!(tile.core_debug.is_halted(), "core must be debug-halted after PC_Event match");

        // Engine status must not be an error.
        assert!(
            !matches!(engine.status(), EngineStatus::Error),
            "engine must not be in an error state after pre-execute halt"
        );
    }

    /// When PC_Event0 is armed at a different PC than the current one, the
    /// core must run normally (no spurious halt before-commit).
    #[test]
    fn pre_execute_pc_event_armed_but_not_matching_runs_normally() {
        const TRAP_PC: u32 = 0x100; // arm at 0x100 -- far away from start
        const PC_EVENT0: u32 = 0x32020;
        const DEBUG_CONTROL2: u32 = 0x32018;

        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 16]);
        }
        engine.enable_core(0, 2);

        use crate::device::tile::CtrlPacketAction;
        let pc_event_value = 0x8000_0000 | (TRAP_PC & 0x3FFF);
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: PC_EVENT0,
            value: pc_event_value,
        });
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: DEBUG_CONTROL2,
            value: 0x1,
        });

        // Step once from PC=0. PC_Event0 is at 0x100, so no match -> normal run.
        engine.step();

        let ctx = engine.core_context(0, 2).expect("core (0,2)");
        assert_eq!(ctx.pc(), 4, "PC must advance normally when no PC_Event match");

        let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
        assert!(!tile.core_debug.is_halted(), "core must NOT be halted when PC does not match");
    }

    /// After a pre-execute halt (Step 2 seam), resume (clear Debug_Control0[0])
    /// and step again: the same PC_Event must NOT re-fire immediately on resume.
    /// The core must advance past the trap PC without halting again, then halt
    /// at the NEXT match if the trap PC wraps or is hit again.
    ///
    /// This tests the "condition the existing post-execute update_pc so the
    /// same match does not re-fire after resume" requirement from the plan.
    #[test]
    fn pre_execute_pc_event_no_refire_after_resume() {
        const TRAP_PC: u32 = 0x00;
        const PC_EVENT0: u32 = 0x32020;
        const DEBUG_CONTROL2: u32 = 0x32018;

        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 16]);
        }
        engine.enable_core(0, 2);

        use crate::device::tile::CtrlPacketAction;
        let pc_event_value = 0x8000_0000 | (TRAP_PC & 0x3FFF);
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: PC_EVENT0,
            value: pc_event_value,
        });
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: DEBUG_CONTROL2,
            value: 0x1,
        });

        // First step: should halt before-commit at TRAP_PC=0.
        engine.step();
        {
            let tile = engine.device.array.get(0, 2).expect("compute tile");
            assert!(tile.core_debug.is_halted(), "must halt at trap PC");
        }
        {
            let ctx = engine.core_context(0, 2).expect("core");
            assert_eq!(ctx.pc(), TRAP_PC, "PC must not advance");
        }

        // Resume: clear the halt.
        {
            let tile = engine.device_mut().tile_mut(0, 2).expect("compute tile");
            tile.core_debug.request_resume();
        }

        // Second step: must NOT re-fire the pre-execute seam at the same PC;
        // the bundle must execute and the PC must advance.
        engine.step();
        {
            let ctx = engine.core_context(0, 2).expect("core");
            assert_eq!(ctx.pc(), 4, "PC must advance past trap PC after resume -- no re-fire");
            let tile = engine.device.array.get(0, 2).expect("compute tile");
            assert!(!tile.core_debug.is_halted(), "must not re-halt on the same PC after resume");
        }
    }

    /// Async halt (host Debug_Control0[0] write) must still be caught at the
    /// existing interpreter.rs:181 gate and must NOT be affected by the
    /// pre-execute seam. The seam only checks for PC_Event conditions; an async
    /// halt sets core_debug.halted = true through a different path, and the
    /// interpreter's existing is_halted() gate (step_with_neighbor_locks ->
    /// interpreter.rs:181) catches it before the bundle executes anyway.
    ///
    /// This test verifies the async path is unbroken: write Debug_Control0[0]
    /// to halt, then step -- the interpreter gate returns DebugHalt, PC does
    /// not advance, no interaction with the pre-execute seam.
    #[test]
    fn async_halt_unaffected_by_pre_execute_seam() {
        const DEBUG_CONTROL0: u32 = 0x32010;

        let mut engine = InterpreterEngine::new_npu1();

        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 16]);
        }
        engine.enable_core(0, 2);

        // Arm async halt via Debug_Control0[0] (the host halt path).
        use crate::device::tile::CtrlPacketAction;
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: DEBUG_CONTROL0,
            value: 0x1, // halt bit
        });

        {
            let tile = engine.device.array.get(0, 2).expect("compute tile");
            assert!(tile.core_debug.is_halted(), "precondition: async halt must be set");
        }

        // Step: interpreter.rs:181 gate catches it before execution.
        engine.step();

        let ctx = engine.core_context(0, 2).expect("core");
        assert_eq!(ctx.pc(), 0, "async halt must not advance PC");
        let tile = engine.device.array.get(0, 2).expect("compute tile");
        assert!(tile.core_debug.is_halted(), "core must remain halted after step with async halt");
    }

    /// Guarding test: the pre-execute halt must fire BEFORE the bundle commits.
    ///
    /// This test provides the literal store-not-landed assertion for the
    /// before-commit guarantee derived from the G1 hardware finding:
    ///   "silicon halts BEFORE the trap bundle commits -- all marker slots
    ///    zero including the trap slot, DEBUG_HALT=1" (findings doc, 2026-05-18).
    ///
    /// Approach (a) per spec §6: reuse the encoded TRAP bundle bytes from
    /// the debug_halt_probe ELF at PC=0x184:
    ///   bytes: [0x03, 0x40, 0x57, 0x11, 0x00, 0x20, 0xd4, 0x00]
    ///   disasm: st dj0, [p0, #4]; mov m0, #0xaa
    ///   source: mlir-aie/build/test/npu-xrt/debug_halt_probe/chess/
    ///           aie_arch.mlir.prj/main_core_0_2.elf (verified 2026-05-18)
    ///
    /// Setup:
    ///   p0 = 0x70400 -- valid local data memory (CardDir 7, 0x70000-0x7FFFF)
    ///   dj0 (modifier reg 16) = 0x7700 -- distinguishable 20-bit store value
    ///   data_memory[0x404] = 0xDEAD_BEEF -- sentinel (offset = 0x70404 - 0x70000)
    ///
    /// Expected (pre-execute halt):
    ///   data_memory[0x404] == 0xDEAD_BEEF  (store did NOT land)
    ///
    /// Falsified (post-execute halt, would indicate regression):
    ///   data_memory[0x404] == 0x7700       (dj0 was stored, bundle committed)
    #[test]
    fn pre_execute_pc_event_store_not_landed_before_commit() {
        use crate::interpreter::state::MOD_BASE_DJ;

        // TRAP bundle from debug_halt_probe ELF (PC=0x184):
        //   st dj0, [p0, #4]; mov m0, #0xaa
        const TRAP_BUNDLE: [u8; 8] = [0x03, 0x40, 0x57, 0x11, 0x00, 0x20, 0xd4, 0x00];
        const TRAP_PC: u32 = 0x00; // load the bundle at offset 0

        // Store address: p0 + 4 = 0x70400 + 4 = 0x70404
        // Data memory offset: 0x70404 - 0x70000 = 0x404
        const P0_VAL: u32 = 0x70400;
        const DATA_MEM_STORE_OFFSET: usize = 0x404;
        // dj0: 20-bit value written to data_memory if bundle executes
        const DJ0_VAL: u32 = 0x7700;
        const SENTINEL: u32 = 0xDEAD_BEEF;

        const PC_EVENT0: u32 = 0x32020;
        const DEBUG_CONTROL2: u32 = 0x32018;

        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        // Load the TRAP bundle at offset 0 in program memory.
        // Pad to 16 bytes so the interpreter always has a valid bundle
        // following the trap (avoid decode underrun on the second step).
        {
            let tile = engine.device_mut().tile_mut(0, 2).expect("compute tile (0,2)");
            let mut prog = [0x00u8; 16];
            prog[..8].copy_from_slice(&TRAP_BUNDLE);
            tile.write_program(0, &prog);

            // Pre-initialize the sentinel at the store target address.
            tile.write_data_u32(DATA_MEM_STORE_OFFSET, SENTINEL);
        }

        // Pre-initialize p0 and dj0 in the core context BEFORE enabling the
        // core.  engine.enable_core() does not reset the context, so these
        // values survive into the first step.
        //
        // set_core_pointer / set_core_modifier require the core to exist in
        // the engine (they look it up by (col, row)); the core is always
        // constructed at engine creation time, so this is safe pre-enable.
        engine.set_core_pointer(0, 2, 0, P0_VAL); // p0 = 0x70400
        engine.set_core_modifier(0, 2, MOD_BASE_DJ + 0, DJ0_VAL); // dj0 = 0x7700
        engine.enable_core(0, 2);

        // Arm PC_Event0 at TRAP_PC and enable PC_Event_Halt.
        use crate::device::tile::CtrlPacketAction;
        let pc_event_value = 0x8000_0000 | (TRAP_PC & 0x3FFF);
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: PC_EVENT0,
            value: pc_event_value,
        });
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: DEBUG_CONTROL2,
            value: 0x1,
        });

        // Step once: pre-execute seam fires at TRAP_PC, halts before commit.
        engine.step();

        // Core must be halted and PC must not have advanced.
        {
            let ctx = engine.core_context(0, 2).expect("core (0,2)");
            assert_eq!(ctx.pc(), TRAP_PC, "PC must not advance -- pre-execute halt");
        }
        {
            let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
            assert!(tile.core_debug.is_halted(), "core must be debug-halted");

            // Literal store-not-landed assertion (G1-derived before-commit guarantee).
            // If the seam incorrectly fired after-commit, data_memory[0x404] would
            // be DJ0_VAL (0x7700); if it fired correctly before-commit, the
            // SENTINEL (0xDEAD_BEEF) is undisturbed.
            let stored = tile.read_data_u32(DATA_MEM_STORE_OFFSET);
            assert_eq!(
                stored,
                Some(SENTINEL),
                "store must NOT have landed (pre-execute halt is before-commit): \
                 expected sentinel 0x{:08X}, got {:?}",
                SENTINEL,
                stored
            );
        }
    }

    /// S1 regression: the intermediate-halted-tick path must not swallow the
    /// host resume.
    ///
    /// Reproduces the exact resume-swallow bug found in review S1: the engine
    /// keeps ticking between the pre-execute halt and the host resume. The
    /// still-halted core falls through to step_with_neighbor_locks, the
    /// is_halted() gate returns DebugHalt without executing the trap bundle,
    /// and (under the buggy code) clear_sync_trap_consumed() ran
    /// unconditionally there -- dropping the latch while PC was still pinned
    /// at TRAP_PC. The next tick (now resumed) then re-fired the seam on the
    /// same bundle, swallowing the resume.
    ///
    /// Sequence: step (seam halts pre-commit) -> step again WITHOUT resuming
    /// (the intermediate halted tick that cleared the latch under the old
    /// code) -> request_resume() -> step. Expected: PC advances past TRAP_PC
    /// and the core is not halted (resume honored, not swallowed).
    ///
    /// FAILS against pre-S1-fix code (unconditional clear after
    /// step_with_neighbor_locks); PASSES after the fix (clear only on
    /// StepResult::Continue).
    #[test]
    fn pre_execute_pc_event_resume_after_intermediate_halted_tick() {
        const TRAP_PC: u32 = 0x00;
        const PC_EVENT0: u32 = 0x32020;
        const DEBUG_CONTROL2: u32 = 0x32018;

        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 16]);
        }
        engine.enable_core(0, 2);

        use crate::device::tile::CtrlPacketAction;
        let pc_event_value = 0x8000_0000 | (TRAP_PC & 0x3FFF);
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: PC_EVENT0,
            value: pc_event_value,
        });
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: DEBUG_CONTROL2,
            value: 0x1,
        });

        // Step 1: pre-execute seam fires -> halt before-commit at TRAP_PC.
        engine.step();
        {
            let tile = engine.device.array.get(0, 2).expect("compute tile");
            assert!(tile.core_debug.is_halted(), "step 1: must halt at trap PC");
            let ctx = engine.core_context(0, 2).expect("core");
            assert_eq!(ctx.pc(), TRAP_PC, "step 1: PC must not advance");
        }

        // Step 2: NO resume yet. The still-halted core falls through to
        // step_with_neighbor_locks; the is_halted() gate returns DebugHalt
        // without executing the bundle. PC is still pinned at TRAP_PC.
        // This is the tick that cleared the latch under the buggy code.
        engine.step();
        {
            let tile = engine.device.array.get(0, 2).expect("compute tile");
            assert!(tile.core_debug.is_halted(), "step 2: still halted (no resume issued)");
            let ctx = engine.core_context(0, 2).expect("core");
            assert_eq!(ctx.pc(), TRAP_PC, "step 2: PC must still be pinned at trap PC");
        }

        // Now the host resumes.
        {
            let tile = engine.device_mut().tile_mut(0, 2).expect("compute tile");
            tile.core_debug.request_resume();
        }

        // Step 3: the trap bundle must execute exactly once and PC must
        // advance. Under the buggy code the latch was already cleared by
        // step 2, so the seam re-fires here and re-halts -- the resume is
        // swallowed and this assertion fails.
        engine.step();
        {
            let ctx = engine.core_context(0, 2).expect("core");
            assert_eq!(
                ctx.pc(),
                4,
                "step 3: PC must advance past trap PC after resume -- \
                 resume must not be swallowed by the intermediate halted tick"
            );
            let tile = engine.device.array.get(0, 2).expect("compute tile");
            assert!(
                !tile.core_debug.is_halted(),
                "step 3: core must not re-halt on the same PC after resume"
            );
        }
    }

    /// Unit-2 integration: count-step budget decrements ONLY on committed
    /// bundles (StepResult::Continue), not on stall/DebugHalt cycles.
    ///
    /// Spec §5.2 locked modeling decision (2026-05-19): count-step is a
    /// "per-committed-bundle decrement"; WaitLock/DebugHalt/etc. cycles must
    /// not consume the budget. This test is the coordinator-level guard that
    /// enforces the Continue-arm placement of tick_count_step() introduced in
    /// Phase B Unit-2.
    ///
    /// Approach: reuse the pre-execute seam harness (same as
    /// pre_execute_pc_event_resume_after_intermediate_halted_tick). PC_Event0
    /// is armed at PC=0 so the first engine.step() fires the pre-execute seam
    /// and returns DebugHalt without committing a bundle. Two such stall ticks
    /// drive M=2 non-committing cycles against a budget of N=2. After both
    /// stall ticks the core must NOT be count-step-halted (budget intact).
    /// Then resume + 2 committed bundles must exhaust the budget and halt.
    ///
    /// Realization: the DebugHalt stall path is a real non-commit cycle (same
    /// coordinator path as WaitLock/WaitDma for the purpose of tick_count_step
    /// gating). The literal "stall doesn't decrement" guarantee rests on the
    /// Continue-arm placement; the pre-fix unconditional site would have fired
    /// tick_count_step on both DebugHalt ticks, exhausting the N=2 budget
    /// before any bundle committed, causing count-step-halt DURING the stall
    /// phase -- the post-stall "not halted" assertion below would then fail.
    ///
    /// FAILS against pre-fix code (tick_count_step unconditional post-execute);
    /// PASSES after the fix (tick_count_step gated to StepResult::Continue).
    ///
    /// G2 hardware corroboration: LANDED=0 on non-committing cycles confirms
    /// the silicon does not count stall ticks toward the step budget.
    #[test]
    fn count_step_budget_not_consumed_by_stall_cycles() {
        const TRAP_PC: u32 = 0x00;
        const PC_EVENT0: u32 = 0x32020;
        const DEBUG_CONTROL0: u32 = 0x32010;
        const DEBUG_CONTROL2: u32 = 0x32018;

        // count-step budget N=2: Single_Step_Count = 2, bits [5:2] of
        // Debug_Control0, no halt bit (bit 0 = 0) so the core starts running.
        // Encoding: 2 << DBG_CTRL0_SSTEP_COUNT_LSB (=2) = 0x08.
        const COUNT_STEP_N2: u32 = 2 << 2; // = 0x08

        let mut engine = InterpreterEngine::new_npu1();
        engine.ungate_all_for_test();

        // Load 8 NOP bundles (64 bytes of zeros) so the core has room to commit
        // several bundles after resuming.
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 64]);
        }
        engine.enable_core(0, 2);

        use crate::device::tile::CtrlPacketAction;

        // Arm count-step N=2 via Debug_Control0. No halt bit: core starts
        // executing. The budget is now live at 2.
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: DEBUG_CONTROL0,
            value: COUNT_STEP_N2,
        });

        // Arm PC_Event0 at TRAP_PC=0 with PC_Event_Halt enabled. The first
        // engine.step() will fire the pre-execute seam at PC=0, halting before
        // the bundle commits (DebugHalt -- no StepResult::Continue).
        let pc_event_value = 0x8000_0000 | (TRAP_PC & 0x3FFF);
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: PC_EVENT0,
            value: pc_event_value,
        });
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: DEBUG_CONTROL2,
            value: 0x1,
        });

        // Precondition: core is not yet halted (only count-step armed, no
        // immediate halt bit, no seam fired yet).
        {
            let tile = engine.device.array.get(0, 2).expect("compute tile");
            assert!(!tile.core_debug.is_halted(), "precondition: core must not be halted before any step");
        }

        // Stall tick 1: pre-execute seam fires -> DebugHalt, no bundle commit.
        // tick_count_step must NOT fire (Continue arm not taken). Budget: 2.
        engine.step();
        {
            let tile = engine.device.array.get(0, 2).expect("compute tile");
            assert!(
                tile.core_debug.is_halted(),
                "stall tick 1: core must be debug-halted by pre-execute seam"
            );
            // Halt is from PC_Event, NOT from count-step expiry. If the budget
            // had been consumed (pre-fix bug), the debug_control0 read-back
            // would still show halted=1 -- but the key follow-up assertion is
            // that AFTER 2 stall ticks the core was not count-step-halted (the
            // budget is intact and resume+commit sequence correctly fires it).
        }

        // Stall tick 2: still halted (no resume), DebugHalt again. Budget must
        // remain at 2 (the pre-fix bug would have decremented to 1 here, and
        // already decremented to 1 in tick 1 -- but we observe behavior via
        // what happens after resume, not by reading the internal field).
        engine.step();
        {
            let tile = engine.device.array.get(0, 2).expect("compute tile");
            assert!(tile.core_debug.is_halted(), "stall tick 2: core must remain halted (no resume issued)");
        }

        // Resume the core. The PC_Event seam latch was consumed on the first
        // stall tick, so resuming lets the trap bundle execute next step.
        {
            let tile = engine.device_mut().tile_mut(0, 2).expect("compute tile");
            tile.core_debug.request_resume();
        }

        // Commit tick 1: trap bundle at PC=0 executes. StepResult::Continue ->
        // tick_count_step fires. Budget 2 -> 1. Core must NOT halt yet.
        engine.step();
        {
            let tile = engine.device.array.get(0, 2).expect("compute tile");
            assert!(
                !tile.core_debug.is_halted(),
                "commit tick 1: after 1 committed bundle (budget 2->1), core must not halt yet"
            );
        }

        // Commit tick 2: next bundle executes. StepResult::Continue ->
        // tick_count_step fires. Budget 1 -> expiry -> halt. Core MUST halt.
        engine.step();
        {
            let tile = engine.device.array.get(0, 2).expect("compute tile");
            assert!(
                tile.core_debug.is_halted(),
                "commit tick 2: after 2 committed bundles (budget expiry), core must be halted by count-step"
            );
        }
    }

    #[test]
    fn enable_core_clears_core_debug_reset() {
        // §8 close-out: the runtime enable path must clear reset so a
        // halted core reports Core_Status 0x10001, not 0x10003.
        let mut engine = InterpreterEngine::new_npu1();
        engine.enable_core(0, 2);
        let tile = engine.device_mut().tile_mut(0, 2).expect("tile (0,2)");
        assert!(!tile.core_debug.is_reset(), "enable_core must clear reset");
        assert!(tile.core_debug.is_enabled(), "enable_core sets enabled");
    }

    #[test]
    fn export_origin_d_sidecar_matches_contract() {
        // SP-4b Task 10: drive a real single-source channel-15 (timer-reset)
        // flood through the engine the same way production code does
        // (`propagate_broadcasts`, which reads the build-time-generated
        // timing constants), then assert the exported sidecar's shape.
        use crate::device::tile::PendingBroadcast;
        let mut engine = InterpreterEngine::new_npu1();
        {
            let tile = engine.device_mut().array.get_mut(0, 0).expect("shim tile (0,0)");
            tile.pending_broadcasts.push(PendingBroadcast::originated(15));
        }
        engine.device_mut().propagate_broadcasts(0, 0);

        let v = engine.export_origin_d_sidecar();
        // SP-5c calibrated flip (2026-07-02): the shipped consts are now calibrated.
        assert_eq!(v["calibrated"], serde_json::json!(true), "calibrated after the SP-5c flip");
        assert_eq!(v["flood_source"], serde_json::json!("0|0"), "single source recorded as col|row");
        assert_eq!(v["modules"]["0|0|shim"], serde_json::json!(0), "flood source origin_D is 0");
        // The shim row does not forward tile-to-tile E/W: a shim-sourced horizontal
        // broadcast detours through the fabric (N-across-S), so shim(1,0) origin_D
        // = d_h + 2*d_v = 4 + 4 = 8, NOT the direct d_h=4. Locks in both the flip
        // and the shim-E/W fix in the production export path.
        assert_eq!(v["modules"]["1|0|shim"], serde_json::json!(8), "shim E/W detour: d_h + 2*d_v");
    }

    #[test]
    fn export_origin_d_sidecar_reflects_override() {
        use crate::device::tile::PendingBroadcast;
        use xdna_archspec::types::BroadcastTiming;
        let mut engine = InterpreterEngine::new_npu1();
        engine.device_mut().set_broadcast_timing_override(Some(BroadcastTiming {
            per_hop_horizontal: 0,
            per_hop_vertical: 3,
            intra_tile_core_offset: 0,
            intra_tile_mem_offset: 0,
            calibrated: true,
        }));
        {
            let tile = engine.device_mut().array.get_mut(0, 0).expect("shim (0,0)");
            tile.pending_broadcasts.push(PendingBroadcast::originated(15));
        }
        engine.device_mut().propagate_broadcasts(0, 0);
        let v = engine.export_origin_d_sidecar();
        assert_eq!(v["calibrated"], serde_json::json!(true), "override calibrated must surface");
        // A tile one vertical hop from the (0,0) shim source has origin_D d_v = 3.
        assert_eq!(
            v["modules"]["0|1|memtile"],
            serde_json::json!(3),
            "override d_v=3 must drive the sidecar origin_D: {v}"
        );
    }

    #[test]
    fn export_origin_d_sidecar_matches_committed_fixture() {
        // SP-5a Task 3: pin the REAL exported sidecar JSON as a committed
        // read-only golden file. The live export must equal it (drift fails
        // loudly). Regeneration is explicit and env-gated -- the default
        // `cargo test` run never writes the fixture:
        //   UPDATE_FIXTURES=1 cargo test --lib export_origin_d_sidecar_matches_committed_fixture
        // The Python contract test (tools/test_inference_real_sidecar_contract.py)
        // consumes the same committed file to prove the cross-language round-trip.
        use crate::device::tile::PendingBroadcast;
        let mut engine = InterpreterEngine::new_npu1();
        {
            let tile = engine.device_mut().array.get_mut(0, 0).expect("shim tile (0,0)");
            tile.pending_broadcasts.push(PendingBroadcast::originated(15));
        }
        engine.device_mut().propagate_broadcasts(0, 0);
        let live = engine.export_origin_d_sidecar();

        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tools/tests/fixtures/origin_d_real_export.json");

        if std::env::var("UPDATE_FIXTURES").is_ok() {
            std::fs::create_dir_all(fixture_path.parent().unwrap()).unwrap();
            std::fs::write(&fixture_path, serde_json::to_string_pretty(&live).unwrap()).unwrap();
        }

        let committed = std::fs::read_to_string(&fixture_path).unwrap_or_else(|e| {
            panic!(
                "committed fixture {} missing ({e}); regenerate with \
                 UPDATE_FIXTURES=1 cargo test --lib export_origin_d_sidecar_matches_committed_fixture",
                fixture_path.display(),
            )
        });
        let committed_val: serde_json::Value = serde_json::from_str(&committed).unwrap();
        assert_eq!(
            live, committed_val,
            "live origin_D export drifted from the committed fixture; if intended, \
             regenerate with UPDATE_FIXTURES=1 cargo test --lib export_origin_d_sidecar_matches_committed_fixture",
        );
    }

    #[test]
    fn export_origin_d_sidecar_omits_flood_source_with_no_flood() {
        // No channel-15 broadcast fired this run -- zero sources is exactly
        // as ambiguous as multiple, so the export must fail loud (null
        // flood_source, empty modules) rather than guess.
        let engine = InterpreterEngine::new_npu1();
        let v = engine.export_origin_d_sidecar();
        assert!(v["flood_source"].is_null(), "no flood source recorded -> null, not a guess");
        assert!(v["modules"].as_object().unwrap().is_empty(), "no source -> no module table");
    }

    #[test]
    fn export_origin_d_sidecar_omits_flood_source_with_multiple_sources() {
        // Two distinct tiles firing channel 15 makes T0 ambiguous (design
        // Sec.4d) -- the export must omit flood_source rather than pick one
        // of the two sources arbitrarily.
        use crate::device::tile::PendingBroadcast;
        let mut engine = InterpreterEngine::new_npu1();
        for &(col, row) in &[(0u8, 0u8), (1u8, 0u8)] {
            let tile = engine.device_mut().array.get_mut(col, row).expect("shim tile");
            tile.pending_broadcasts.push(PendingBroadcast::originated(15));
            engine.device_mut().propagate_broadcasts(col, row);
        }
        let v = engine.export_origin_d_sidecar();
        assert!(v["flood_source"].is_null(), "multi-source must omit flood_source");
        assert!(v["modules"].as_object().unwrap().is_empty(), "multi-source -> empty module table");
    }
}
