//! AIE tile state representation.
//!
//! Each tile contains:
//! - Data memory (64KB for compute tiles, 512KB for mem tiles)
//! - Program memory (16KB, compute tiles only)
//! - Locks for synchronization (16 for compute tiles, 64 for mem tiles)
//! - DMA engine with buffer descriptors and channels
//! - Core state (PC, registers, status)
//! - Stream switch configuration
//!
//! # Architecture Constants
//!
//! All constants are derived from AMD AM020 (AIE-ML Architecture Manual).
//! See `xdna_archspec::aie2` module for the authoritative constants.
//!
//! # Performance
//!
//! This module is designed for fast emulation:
//! - Fixed-size arrays (no heap allocation during emulation)
//! - Direct field access (no hash maps)
//! - Cache-friendly layout (related data together)

mod params;
mod locks;
mod dma_legacy;
mod core_state;
mod edge;
mod registers;
mod streams;

#[cfg(test)]
mod tests;

// Re-export all public types so external code sees them at `tile::`.
pub use params::{PROGRAM_MEMORY_SIZE, TileParams};
pub use locks::{LockResult, LockRequestor, LockRequest, LockArbiterStats, LockArbiter, Lock};
pub use dma_legacy::{DmaBufferDescriptor, DmaChannel};
pub use core_state::{CoreState, LegacyStreamPort, CtrlPacketAction};
pub use xdna_archspec::types::TileKind;
pub use edge::EdgeDetector;

use super::stream_switch::StreamSwitch as FunctionalStreamSwitch;
use super::trace_unit::TraceUnit;
use crate::interpreter::state::EventType;

#[derive(Debug)]
pub struct Tile {
    /// Tile type
    pub tile_kind: TileKind,

    /// Column index
    pub col: u8,

    /// Row index
    pub row: u8,

    /// Processor bus enabled (Core_Processor_Bus register bit 0).
    /// When set, core loads/stores from addresses >= 0x10000 access tile
    /// configuration registers instead of neighbor data memory.
    pub processor_bus_enabled: bool,

    // === Hot data (accessed every cycle) ===
    /// Core processor state (compute tiles only)
    pub core: CoreState,

    /// Lock states (sized from TileParams: 16 for compute, 64 for mem tile, 0 for shim)
    pub locks: Vec<Lock>,

    /// Round-robin lock arbiter for this tile's memory module.
    ///
    /// Per AM020, the lock arbiter serializes competing lock requests from
    /// the core and DMA channels. Requests are submitted during the cycle
    /// and resolved at end-of-cycle. Core releases submitted in Phase 2
    /// are resolved alongside DMA requests in Phase 3, providing the
    /// 1-cycle visibility delay that matches hardware.
    pub lock_arbiter: LockArbiter,

    // === Warm data (accessed during DMA) ===
    /// DMA buffer descriptors (sized from TileParams: 16 for compute, 48 for mem tile)
    pub dma_bds: Vec<DmaBufferDescriptor>,

    /// DMA channels (sized from TileParams: 4 for compute, 12 for mem tile)
    pub dma_channels: Vec<DmaChannel>,

    // === Stream port buffers (for core direct stream access) ===
    /// Stream input buffer for core direct reads (StreamReadScalar)
    /// Maps port number to queue of incoming data
    pub stream_input: [std::collections::VecDeque<u32>; 8],

    /// Stream output buffer for core direct writes (StreamWriteScalar)
    pub stream_output: [std::collections::VecDeque<u32>; 8],

    // === Cold data (routing configuration) ===
    /// Stream switch configuration (full functional model with FIFOs and local routes)
    pub stream_switch: FunctionalStreamSwitch,

    // === Large data (memory) ===
    /// Data memory (64KB for compute, 512KB for mem tile)
    /// Boxed to avoid huge stack allocation
    data_memory: Box<[u8]>,

    /// Program memory (64KB, compute tiles only)
    /// None for shim and mem tiles
    program_memory: Option<Box<[u8; PROGRAM_MEMORY_SIZE]>>,

    /// Register store for shim tiles (NPU configuration registers).
    /// Shim tiles don't have data memory but need to store DMA/stream config.
    /// Stored as sparse map since most addresses won't be written.
    pub(crate) registers: std::collections::HashMap<u32, u32>,

    // Control packet FSM has moved to control_packets::StreamReassembler
    // owned by TileArray (array.rs). The tile no longer tracks packet state.
    /// Shim Mux: which switchbox South slave port each DMA MM2S channel feeds.
    /// Parsed from Mux_Config register (0x1F000). Index 0 = MM2S ch0, etc.
    /// Value is the switchbox slave port index (e.g., 5 for South3).
    pub shim_mux_mm2s_slaves: Vec<Option<usize>>,

    /// Shim Mux: which switchbox South master port feeds each DMA S2MM channel.
    /// Parsed from Demux_Config register (0x1F004). Index 0 = S2MM ch0, etc.
    /// Value is the switchbox master port index (e.g., 2 for South0).
    pub shim_mux_s2mm_masters: Vec<Option<usize>>,

    // === Trace Units ===
    /// Core module trace unit (compute tiles only).
    ///
    /// Configured by writes to offsets 0x340D0-0x340E4. Monitors core module
    /// events (instructions, stalls) and emits binary trace packets through
    /// the stream switch Trace slave port (slave[23] = AIE_TRACE).
    pub core_trace: TraceUnit,

    /// Memory module trace unit (compute and mem tiles).
    ///
    /// Configured by writes to offsets 0x140D0-0x140E4 (compute) or
    /// 0x940D0-0x940E4 (memtile). Monitors memory module events (DMA, locks)
    /// and emits through Trace slave port (slave[24] = MEM_TRACE for compute,
    /// slave[17] = TRACE for memtile).
    pub mem_trace: TraceUnit,

    /// Pending memory-module trace events from all sources (DMA, locks, etc.).
    ///
    /// On real hardware, the mem trace unit monitors event wires from the
    /// entire memory module -- it doesn't distinguish DMA events from lock
    /// events. This buffer unifies all memory-module event sources. The
    /// coordinator drains it each cycle and routes events to the mem trace
    /// unit via `notify_mem_trace_event()`.
    pub mem_trace_pending: Vec<(u64, crate::interpreter::state::EventType)>,

    // === Event Modules (new -- subsystem will absorb broadcast/port selection) ===
    /// Core module event system (compute and shim tiles only).
    /// None for MemTile (which has no core module).
    pub core_events: Option<super::events::EventModule>,
    /// Memory module event system (compute and mem tiles only).
    /// None for Shim (which has no memory module).
    pub mem_events: Option<super::events::EventModule>,

    /// Stream switch event port selection (8 logical event ports).
    ///
    /// Each entry maps a logical event port (0-7) to a physical stream switch
    /// port. `None` means the port is not configured. `Some((port_idx, is_master))`
    /// identifies the physical port to monitor for PORT_RUNNING/IDLE/STALLED events.
    ///
    /// Configured by Event Port Selection registers:
    /// - Compute/Shim: 0x3FF00 (ports 0-3), 0x3FF04 (ports 4-7)
    /// - MemTile: 0xB0F00 (ports 0-3), 0xB0F04 (ports 4-7)
    ///
    /// Register encoding per 8-bit slot: bit 5 = master (1) or slave (0),
    /// bits 4:0 = port index.
    pub event_port_selection: [Option<(u8, bool)>; 8],

    /// Previous-cycle state per event-port slot, used to edge-trigger
    /// PORT_RUNNING/PORT_IDLE/PORT_STALLED/PORT_TLAST trace events.
    ///
    /// On real silicon these events fire on the rising edge of the
    /// corresponding port signal -- not every cycle the signal is asserted.
    /// We track the previous cycle's value and only emit when the signal
    /// transitions, matching HW behavior.
    ///
    /// Indexed by event-port slot (0..8), not by physical port index.
    /// `(prev_active, prev_stalled, prev_tlast)` per slot.
    pub prev_port_state: [(bool, bool, bool); 8],

    // === Cascade Stream (compute tiles only) ===
    /// Cascade input FIFO (SCD). 384-bit width, depth 1.
    /// Dedicated point-to-point link between adjacent compute tiles,
    /// entirely separate from the stream switch fabric.
    /// Source: aie-rt/driver/src/core/xaie_core.c:993-1046
    pub cascade_input: std::collections::VecDeque<[u64; 6]>,

    /// Cascade output FIFO (MCD). 384-bit width, depth 1.
    pub cascade_output: std::collections::VecDeque<[u64; 6]>,

    /// Cascade input direction: 0=North, 1=West.
    /// From accumulator control register at offset 0x36060 bit 0.
    pub cascade_input_dir: u8,

    /// Cascade output direction: 0=South, 1=East.
    /// From accumulator control register at offset 0x36060 bit 1.
    pub cascade_output_dir: u8,

    // === Performance Counters ===
    /// Core module performance counters (compute: 4 counters, shim/PL: 2).
    ///
    /// Configured by PERFORMANCE_CONTROL/COUNTER/EVENT_VALUE registers:
    /// - Compute core module: 0x31500-0x3158C (4 counters)
    /// - Shim PL module:      0x31000-0x31084 (2 counters)
    ///
    /// TODO: Implement actual counting logic triggered by start/stop/reset
    /// events, and fire PERF_CNT_N events (hw_id 5-8) when counter reaches
    /// the configured event_value threshold.
    pub core_perf_counters: super::perf_counters::PerfCounterBank,

    /// Memory module performance counters (compute: 2 counters, memtile: 4).
    ///
    /// Configured by PERFORMANCE_CONTROL/COUNTER/EVENT_VALUE registers:
    /// - Compute memory module: 0x11000-0x11084 (2 counters)
    /// - MemTile:               0x91000-0x9108C (4 counters)
    pub mem_perf_counters: super::perf_counters::PerfCounterBank,

    // === Timers ===
    /// Core module timer (compute and shim tiles).
    pub core_timer: super::timer::TileTimer,
    /// Memory module timer (compute and mem tiles).
    pub mem_timer: super::timer::TileTimer,

    // === Core Debug ===
    /// Core debug state (compute tiles only -- halt, step, PC/SP/LR reads).
    pub core_debug: super::core_debug::CoreDebugState,

    // === Memory Bank Conflict Detection ===
    /// Bitmask of memory banks accessed by DMA during this cycle.
    /// Bit N set = bank N was accessed. Supports up to 16 banks (MemTile).
    /// Reset at the start of each coordinator step.
    pub cycle_dma_banks: u16,

    // === Edge Detection ===
    /// Core module edge detectors (two independent circuits).
    /// Configured by Edge_Detection_event_control register at 0x34408 (compute).
    /// Monitor core module event signals for rising/falling transitions.
    pub core_edge_detectors: [EdgeDetector; 2],

    /// Memory module edge detectors (two independent circuits).
    /// Configured by Edge_Detection_event_control register at:
    /// - 0x14408 (compute tile memory module)
    /// - 0x94408 (MemTile)
    pub mem_edge_detectors: [EdgeDetector; 2],

    // === Interrupt Controllers (shim tiles only) ===
    /// L1 interrupt controller (shim tiles only).
    pub l1_irq: Option<super::interrupts::L1InterruptController>,
    /// L2 interrupt controller (shim NoC tiles only).
    pub l2_irq: Option<super::interrupts::L2InterruptController>,

    // === Event Broadcast ===
    /// Event broadcast channel mapping (16 channels).
    ///
    /// Each entry stores the local event ID that triggers that broadcast channel.
    /// When Event_Generate fires an event matching channel N's configured ID,
    /// Pending broadcast events to propagate to all tiles in this column.
    ///
    /// When Event_Generate fires an event that matches a broadcast channel,
    /// the BROADCAST_N hw_id (107+N) is pushed here. The caller (state.rs
    /// or NPU executor) drains this after each write and propagates to the
    /// column.
    pub pending_broadcasts: Vec<u8>,

    /// Pending control packet read response words.
    ///
    /// When an OP_READ control packet is processed, the response (stream
    /// header + data words) is queued here. Each routing cycle, words are
    /// drained into the TileCtrl slave port as FIFO space permits, matching
    /// the backpressure-aware pattern used by trace unit injection.
    ///
    /// Each entry is (word, tlast). The header is first, followed by data
    /// words, with TLAST on the final data word.
    pub pending_ctrl_response: std::collections::VecDeque<(u32, bool)>,

    /// Cumulative count of granted lock releases (core + DMA).
    ///
    /// Monotonically increasing. Incremented inside `resolve_lock_requests()`
    /// whenever a release operation is granted by the lock arbiter.
    /// Used by stall detection to distinguish 'slow but working' from
    /// 'stuck in an infinite loop'.
    lock_release_count: u64,

    /// MemTile DMA Event Channel Selection register (offset 0xA06A0).
    ///
    /// Selects which physical DMA channel feeds each of the four memtile
    /// DMA event broadcast lines (S2MM_SEL0/SEL1, MM2S_SEL0/SEL1). Only
    /// meaningful on memtiles; unused on compute and shim. Reset value is 0
    /// (every SEL slot points at channel 0). See `crate::trace::MemtileDmaEventSel`
    /// for the field layout.
    pub memtile_dma_event_chan_sel: u32,
}

// Performance counter types are now in src/device/perf_counters/mod.rs.
// The Tile struct uses perf_counters::PerfCounterBank directly.

impl Tile {
    /// Create a new tile of the specified type with explicit parameters.
    ///
    /// Production code should use the `ArchConfig`-derived params (via
    /// `TileArray::new()`). Test code can use `Tile::compute()` etc. for
    /// convenience with NPU1/AIE2 defaults.
    pub fn new(tile_kind: TileKind, col: u8, row: u8, params: &TileParams) -> Self {
        let program_memory = match tile_kind {
            TileKind::Compute => Some(Box::new([0u8; PROGRAM_MEMORY_SIZE])),
            _ => None,
        };

        Self {
            tile_kind,
            col,
            row,
            processor_bus_enabled: false,
            core: CoreState::default(),
            locks: vec![Lock::default(); params.num_locks],
            lock_arbiter: LockArbiter::new(
                params.num_locks,
                params.dma_s2mm_channels as u8,
                params.dma_mm2s_channels as u8,
            ),
            dma_bds: vec![DmaBufferDescriptor::default(); params.num_bds],
            dma_channels: vec![DmaChannel::default(); params.num_channels],
            stream_input: Default::default(),
            stream_output: Default::default(),
            stream_switch: match tile_kind {
                TileKind::ShimNoc | TileKind::ShimPl => FunctionalStreamSwitch::new_shim_tile(col),
                TileKind::Mem => FunctionalStreamSwitch::new_mem_tile(col, row),
                TileKind::Compute => FunctionalStreamSwitch::new_compute_tile(col, row),
            },
            data_memory: vec![0u8; params.data_memory_size].into_boxed_slice(),
            program_memory,
            registers: std::collections::HashMap::new(),
            shim_mux_mm2s_slaves: vec![None; params.dma_mm2s_channels],
            shim_mux_s2mm_masters: vec![None; params.dma_s2mm_channels],
            cascade_input: std::collections::VecDeque::new(),
            cascade_output: std::collections::VecDeque::new(),
            cascade_input_dir: 0,
            cascade_output_dir: 0,
            core_trace: TraceUnit::new(col, row),
            mem_trace: TraceUnit::new(col, row),
            mem_trace_pending: Vec::new(),
            event_port_selection: [None; 8],
            prev_port_state: [(false, false, false); 8],
            core_perf_counters: match tile_kind {
                TileKind::Compute => super::perf_counters::PerfCounterBank::new(4),
                TileKind::ShimNoc | TileKind::ShimPl => super::perf_counters::PerfCounterBank::new(2),
                TileKind::Mem => super::perf_counters::PerfCounterBank::new(0),
            },
            mem_perf_counters: match tile_kind {
                TileKind::Compute => super::perf_counters::PerfCounterBank::new(2),
                TileKind::Mem => super::perf_counters::PerfCounterBank::new(4),
                TileKind::ShimNoc | TileKind::ShimPl => super::perf_counters::PerfCounterBank::new(0),
            },
            core_timer: super::timer::TileTimer::new(),
            mem_timer: super::timer::TileTimer::new(),
            core_debug: super::core_debug::CoreDebugState::new(),
            core_events: match tile_kind {
                TileKind::Compute => {
                    Some(super::events::EventModule::new(super::events::EventModuleType::Core))
                }
                TileKind::ShimNoc | TileKind::ShimPl => {
                    Some(super::events::EventModule::new(super::events::EventModuleType::Pl))
                }
                TileKind::Mem => None, // MemTile has no core module
            },
            mem_events: match tile_kind {
                TileKind::Compute => {
                    Some(super::events::EventModule::new(super::events::EventModuleType::Memory))
                }
                TileKind::Mem => {
                    Some(super::events::EventModule::new(super::events::EventModuleType::MemTile))
                }
                TileKind::ShimNoc | TileKind::ShimPl => None, // Shim has no memory module
            },
            l1_irq: if tile_kind.is_shim() {
                Some(super::interrupts::L1InterruptController::new())
            } else {
                None
            },
            l2_irq: if tile_kind.is_shim() {
                Some(super::interrupts::L2InterruptController::new())
            } else {
                None
            },
            cycle_dma_banks: 0,
            core_edge_detectors: [EdgeDetector::default(); 2],
            mem_edge_detectors: [EdgeDetector::default(); 2],
            pending_broadcasts: Vec::new(),
            pending_ctrl_response: std::collections::VecDeque::new(),
            lock_release_count: 0,
            memtile_dma_event_chan_sel: 0,
        }
    }

    /// Reset all per-context state to construction defaults, preserving
    /// memory contents and tile identity.
    ///
    /// Mirrors a real-HW column reset on `hw_context` teardown: locks,
    /// DMA channels/BDs, stream switch, trace units, event modules,
    /// timers, perf counters, edge detectors, cascade FIFOs, pending
    /// event/broadcast/ctrl queues, register store, and runtime flags
    /// all return to power-on defaults. Memory cells survive (HW does
    /// not lose memory contents across column reset), and the tile's
    /// (col, row, tile_kind) identity is preserved.
    ///
    /// Without a comprehensive reset, fields that survive the prior run
    /// bias the next batch's simulation -- the parallel sweep saw this
    /// as bit-for-bit divergence between j=1 and j>=8 runs (a fresh
    /// worker process produced different cycle counts than a worker
    /// whose prior batch we'd "reset" by hand-zeroing a subset of state).
    pub fn reset_for_new_context(&mut self, params: &TileParams) {
        // Save memory contents -- HW preserves them across column reset.
        let saved_data = std::mem::take(&mut self.data_memory);
        let saved_program = self.program_memory.take();
        *self = Tile::new(self.tile_kind, self.col, self.row, params);
        self.data_memory = saved_data;
        self.program_memory = saved_program;
    }

    /// Create a compute tile with NPU1/AIE2 default parameters.
    ///
    /// Convenience constructor for tests. Production code should use
    /// `Tile::new()` with ArchConfig-derived params.
    #[inline]
    pub fn compute(col: u8, row: u8) -> Self {
        Self::new(TileKind::Compute, col, row, &TileParams::compute())
    }

    /// Create a memory tile with NPU1/AIE2 default parameters.
    #[inline]
    pub fn mem_tile(col: u8, row: u8) -> Self {
        Self::new(TileKind::Mem, col, row, &TileParams::mem_tile())
    }

    /// Create a shim tile with NPU1/AIE2 default parameters.
    #[inline]
    pub fn shim(col: u8, row: u8) -> Self {
        Self::new(TileKind::ShimNoc, col, row, &TileParams::shim())
    }

    /// Get data memory slice.
    #[inline]
    pub fn data_memory(&self) -> &[u8] {
        &self.data_memory
    }

    /// Get mutable data memory slice.
    #[inline]
    pub fn data_memory_mut(&mut self) -> &mut [u8] {
        &mut self.data_memory
    }

    /// Get program memory (compute tiles only).
    #[inline]
    pub fn program_memory(&self) -> Option<&[u8; PROGRAM_MEMORY_SIZE]> {
        self.program_memory.as_deref()
    }

    /// Get mutable program memory (compute tiles only).
    #[inline]
    pub fn program_memory_mut(&mut self) -> Option<&mut [u8; PROGRAM_MEMORY_SIZE]> {
        self.program_memory.as_deref_mut()
    }

    /// Write to data memory at offset.
    /// Returns false if offset + data would exceed memory bounds.
    #[inline]
    pub fn write_data(&mut self, offset: usize, data: &[u8]) -> bool {
        if offset + data.len() <= self.data_memory.len() {
            self.data_memory[offset..offset + data.len()].copy_from_slice(data);
            true
        } else {
            false
        }
    }

    /// Write to program memory at offset (compute tiles only).
    /// Returns false if not a compute tile or would exceed bounds.
    #[inline]
    pub fn write_program(&mut self, offset: usize, data: &[u8]) -> bool {
        if let Some(ref mut pm) = self.program_memory {
            if offset + data.len() <= PROGRAM_MEMORY_SIZE {
                pm[offset..offset + data.len()].copy_from_slice(data);
                return true;
            }
        }
        false
    }

    /// Read 32-bit word from data memory.
    #[inline]
    pub fn read_data_u32(&self, offset: usize) -> Option<u32> {
        if offset + 4 <= self.data_memory.len() {
            Some(u32::from_le_bytes([
                self.data_memory[offset],
                self.data_memory[offset + 1],
                self.data_memory[offset + 2],
                self.data_memory[offset + 3],
            ]))
        } else {
            None
        }
    }

    /// Write 32-bit word to data memory.
    #[inline]
    pub fn write_data_u32(&mut self, offset: usize, value: u32) -> bool {
        if offset + 4 <= self.data_memory.len() {
            self.data_memory[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            true
        } else {
            false
        }
    }

    /// Check if this is a compute tile.
    #[inline]
    pub fn is_compute(&self) -> bool {
        self.tile_kind.is_compute()
    }

    /// Check if this is a memory tile.
    #[inline]
    pub fn is_mem(&self) -> bool {
        self.tile_kind.is_mem()
    }

    /// Check if this is a shim tile.
    #[inline]
    pub fn is_shim(&self) -> bool {
        self.tile_kind.is_shim()
    }

    /// DMA BD base address and stride for this tile type (from register database).
    #[inline]
    fn bd_layout(&self, rl: &super::regdb::DeviceRegLayout) -> (u32, u32) {
        match self.tile_kind {
            TileKind::Mem => (rl.memtile_bd_base, rl.memtile_bd_stride),
            TileKind::ShimNoc | TileKind::ShimPl => (rl.shim_bd_base, rl.shim_bd_stride),
            TileKind::Compute => (rl.memory_bd_base, rl.memory_bd_stride),
        }
    }

    /// DMA channel control base address and stride for this tile type.
    ///
    /// For memtiles, returns the S2MM base (channels are S2MM then MM2S,
    /// contiguous with the same stride).
    #[inline]
    fn channel_layout(&self, rl: &super::regdb::DeviceRegLayout) -> (u32, u32) {
        match self.tile_kind {
            TileKind::Mem => (rl.memtile_channel_s2mm_base, rl.memtile_channel_stride),
            TileKind::ShimNoc | TileKind::ShimPl => (rl.shim_channel_base, rl.shim_channel_stride),
            TileKind::Compute => (rl.memory_channel_base, rl.memory_channel_stride),
        }
    }

    // === Event Broadcast ===

    /// Drain pending broadcast events generated by Event_Generate.
    ///
    /// Returns the hw_ids of broadcast events (e.g., 107 for BROADCAST_0)
    /// that should be propagated to all tiles in this column.
    pub fn drain_pending_broadcasts(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.pending_broadcasts)
    }

    // === Memory Bank Conflict Detection ===

    /// Number of physical memory banks for this tile type (for conflict detection).
    pub fn num_banks(&self) -> usize {
        match self.tile_kind {
            TileKind::Compute => xdna_archspec::aie2::compute::PHYSICAL_BANKS as usize,
            TileKind::Mem => xdna_archspec::aie2::memtile::PHYSICAL_BANKS as usize,
            TileKind::ShimNoc | TileKind::ShimPl => 0,
        }
    }

    /// Record that DMA accessed the given memory address range this cycle.
    /// Call from DMA transfer methods during Phase 2.
    #[inline]
    pub fn record_dma_bank_access(&mut self, addr: u32, bytes: usize) {
        let nb = self.num_banks();
        if nb > 0 {
            self.cycle_dma_banks |= crate::device::banking::banks_for_access(addr, bytes, nb);
        }
    }

    /// Reset bank tracking for a new cycle. Call at the start of each step.
    #[inline]
    pub fn reset_bank_tracking(&mut self) {
        self.cycle_dma_banks = 0;
    }

    // === Edge Detection ===

    /// Notify a core module event for both tracing and edge detection.
    ///
    /// Forwards the event to `core_trace.notify_event()` and marks it as
    /// active for the core module edge detectors this cycle.
    ///
    /// `pc` should be `Some(addr)` for instruction-class events (InstrVector,
    /// InstrLoad, etc.) and `None` for stalls, synthetic events, and any
    /// event where a meaningful program counter is not available.
    #[inline]
    pub fn notify_core_trace_event(&mut self, hw_id: u8, cycle: u64, pc: Option<u32>) {
        // hw_id 0 is the EVENT_NONE sentinel (used by callers as "no event for
        // this module side", e.g. memtile broadcasts that have no core-side
        // hw_id). It must never propagate -- otherwise disabled edge detectors
        // (input_event=0 default) would falsely activate, and trace units with
        // start/stop_event=0 would mis-trigger.
        if hw_id == 0 {
            return;
        }
        self.core_trace.notify_event(hw_id, cycle, pc);
        // Auto-reset the core-module timer when the configured Reset_Event
        // is observed (XAie_SyncTimer protocol; latched-then-applied at
        // the next tick).
        self.core_timer.notify_event(hw_id);
        for det in &mut self.core_edge_detectors {
            if det.input_event != 0 && det.input_event == hw_id {
                det.curr_active = true;
            }
        }
    }

    /// Notify a memory module event for both tracing and edge detection.
    ///
    /// Forwards the event to `mem_trace.notify_event()` and marks it as
    /// active for the memory module edge detectors this cycle.
    ///
    /// Memory-module events (DMA, lock, port) do not carry a program counter;
    /// always pass `None`.
    #[inline]
    pub fn notify_mem_trace_event(&mut self, hw_id: u8, cycle: u64, pc: Option<u32>) {
        // hw_id 0 is EVENT_NONE; see notify_core_trace_event for rationale.
        if hw_id == 0 {
            return;
        }
        self.mem_trace.notify_event(hw_id, cycle, pc);
        // Auto-reset the memory-module timer when the configured Reset_Event
        // is observed (XAie_SyncTimer protocol; latched-then-applied at
        // the next tick).
        self.mem_timer.notify_event(hw_id);
        for det in &mut self.mem_edge_detectors {
            if det.input_event != 0 && det.input_event == hw_id {
                det.curr_active = true;
            }
        }
    }

    /// Evaluate edge detectors and fire generated events to trace units.
    ///
    /// Call once per cycle after all raw events have been notified.
    /// Compares current vs previous signal state and fires
    /// EDGE_DETECTION_EVENT_0/1 on detected transitions.
    pub fn evaluate_edge_detectors(&mut self, cycle: u64) {
        // Core module / PL module edge detectors -> core_trace
        for i in 0..2 {
            let det = &self.core_edge_detectors[i];
            let fire = (det.trigger_rising && det.curr_active && !det.prev_active)
                || (det.trigger_falling && !det.curr_active && det.prev_active);
            if fire {
                // Shim PL module: IDs 11-12; Core module: IDs 13-14
                let hw_id = if self.is_shim() {
                    crate::trace::shim_edge_detection_event_hw_id(i as u8)
                } else {
                    crate::trace::core_edge_detection_event_hw_id(i as u8)
                };
                self.core_trace.notify_event(hw_id, cycle, None);
            }
        }
        // Memory module edge detectors -> mem_trace
        for i in 0..2 {
            let det = &self.mem_edge_detectors[i];
            let fire = (det.trigger_rising && det.curr_active && !det.prev_active)
                || (det.trigger_falling && !det.curr_active && det.prev_active);
            if fire {
                let hw_id = if self.is_mem() {
                    crate::trace::memtile_edge_detection_event_hw_id(i as u8)
                } else {
                    crate::trace::mem_edge_detection_event_hw_id(i as u8)
                };
                self.mem_trace.notify_event(hw_id, cycle, None);
            }
        }
        // Advance state: current becomes previous, reset current
        for det in &mut self.core_edge_detectors {
            det.prev_active = det.curr_active;
            det.curr_active = false;
        }
        for det in &mut self.mem_edge_detectors {
            det.prev_active = det.curr_active;
            det.curr_active = false;
        }
    }

    /// Configure edge detectors from a register write.
    ///
    /// Parses the Edge_Detection_event_control register value and updates
    /// the specified detector pair. `is_memtile` controls whether event
    /// fields are 7-bit (compute/shim) or 8-bit (MemTile).
    pub(crate) fn configure_edge_detectors(detectors: &mut [EdgeDetector; 2], value: u32, is_memtile: bool) {
        // Event 0: bits [6:0] or [7:0], rising=bit 9, falling=bit 10
        let event_mask_0: u32 = if is_memtile { 0xFF } else { 0x7F };
        detectors[0].input_event = (value & event_mask_0) as u8;
        detectors[0].trigger_rising = (value & (1 << 9)) != 0;
        detectors[0].trigger_falling = (value & (1 << 10)) != 0;

        // Event 1: bits [22:16] or [23:16], rising=bit 25, falling=bit 26
        let event_mask_1: u32 = if is_memtile { 0xFF } else { 0x7F };
        detectors[1].input_event = ((value >> 16) & event_mask_1) as u8;
        detectors[1].trigger_rising = (value & (1 << 25)) != 0;
        detectors[1].trigger_falling = (value & (1 << 26)) != 0;

        log::debug!(
            "Edge detectors configured: det0(event={}, rise={}, fall={}), det1(event={}, rise={}, fall={})",
            detectors[0].input_event,
            detectors[0].trigger_rising,
            detectors[0].trigger_falling,
            detectors[1].input_event,
            detectors[1].trigger_rising,
            detectors[1].trigger_falling,
        );
    }

    // === Lock Arbiter Interface ===
    //
    // All lock operations go through the round-robin arbiter. Requests are
    // submitted during the cycle (core in Phase 2, DMA in Phase 3) and
    // resolved at the end of Phase 3. The arbiter serializes competing
    // requests: one grant per lock per cycle, with round-robin fairness.
    //
    // Core lock releases submitted in Phase 2 are resolved alongside DMA
    // requests in Phase 3, providing the 1-cycle visibility delay that
    // matches hardware (core release at cycle N visible to DMA at cycle N+1).

    /// Submit a lock request to the arbiter.
    ///
    /// Called by core (lock release instructions) and DMA (acquire/release).
    /// The request is queued until `resolve_lock_requests()` is called.
    #[inline]
    pub fn submit_lock_request(&mut self, request: LockRequest) {
        log::debug!(
            "Tile({},{}) submit_lock_request: {:?} lock={} acquire={} expected={} delta={}",
            self.col,
            self.row,
            request.requestor,
            request.lock_id,
            request.is_acquire,
            request.expected,
            request.delta
        );
        self.lock_arbiter.submit(request);
    }

    /// Defer a core lock release through the arbiter.
    ///
    /// Core releases are deferred by 1 cycle: submitted during Phase 2
    /// (core stepping), resolved at end of Phase 3 (data movement).
    /// This matches hardware's lock arbiter pipeline latency.
    #[inline]
    pub fn defer_core_lock_release(&mut self, lock_id: usize, delta: i8) {
        if lock_id < self.locks.len() {
            log::debug!(
                "Tile({},{}) defer_core_lock_release lock {} delta {}",
                self.col,
                self.row,
                lock_id,
                delta
            );
            self.lock_arbiter.submit(LockRequest {
                requestor: LockRequestor::Core,
                lock_id,
                is_acquire: false,
                expected: 0,
                delta,
                equal_mode: false,
            });
        }
    }

    /// Resolve all pending lock requests using round-robin arbitration.
    ///
    /// Call at end of Phase 3, after all requestors have submitted.
    /// Applies granted requests directly to lock values. Returns results
    /// for callers that need to check grant status (e.g., DMA engine).
    ///
    /// Granted lock operations emit trace events into `mem_trace_pending`,
    /// matching real hardware where the memory module trace unit monitors
    /// all lock state changes regardless of source.
    pub fn resolve_lock_requests(&mut self, cycle: u64) -> Vec<(LockRequestor, usize, bool, bool)> {
        let results = self.lock_arbiter.resolve(&mut self.locks);
        // Emit trace events for granted lock operations and track releases.
        for &(_, lock_id, granted, is_acquire) in results {
            if granted {
                if !is_acquire {
                    self.lock_release_count += 1;
                }
                let event = if is_acquire {
                    EventType::LockAcquire { lock_id: lock_id as u8 }
                } else {
                    EventType::LockRelease { lock_id: lock_id as u8 }
                };
                self.mem_trace_pending.push((cycle, event));
            }
        }
        results.to_vec()
    }

    /// Check if a specific requestor was granted a lock in the last resolve.
    #[inline]
    pub fn lock_was_granted(&self, requestor: LockRequestor, lock_id: usize) -> bool {
        self.lock_arbiter.was_granted(requestor, lock_id)
    }

    /// Get the current committed lock value.
    ///
    /// Returns the live lock value. Pending arbiter requests that have
    /// not yet been resolved are NOT reflected.
    #[inline]
    pub fn effective_lock_value(&self, lock_id: usize) -> i8 {
        if lock_id < self.locks.len() {
            self.locks[lock_id].value
        } else {
            0
        }
    }

    /// Cumulative count of granted lock releases on this tile.
    ///
    /// Monotonically increasing -- incremented each time `resolve_lock_requests()`
    /// grants a release operation. Used by stall detection.
    #[inline]
    pub fn lock_release_count(&self) -> u64 {
        self.lock_release_count
    }

    // === Control Packet Handling ===

    /// Parse Shim Mux_Config register to find DMA MM2S South slave mapping.
    ///
    /// The Shim Mux selects which source (PL/DMA/NoC) feeds each switchbox South
    /// slave port. DMA MM2S output enters the switchbox through a South slave.
    ///
    /// Field layout and port mappings are derived from the AM025 register database.
    /// Select values: 0=South/PL, 1=DMA, 2=NoC
    pub(crate) fn parse_shim_mux_config(&mut self, value: u32) {
        let mux = &super::regdb::device_reg_layout().shim_mux;

        // Reset mapping (register may be rewritten with different config)
        self.shim_mux_mm2s_slaves.fill(None);

        let mut dma_ch = 0usize;
        for mf in &mux.mux_fields {
            let select = mf.field.extract(value);
            if select == 1 && dma_ch < self.shim_mux_mm2s_slaves.len() {
                // DMA source -> this South slave gets MM2S output
                self.shim_mux_mm2s_slaves[dma_ch] = Some(mf.port_index);
                log::info!(
                    "Shim Mux ({},{}): MM2S ch{} -> slave[{}] ({})",
                    self.col,
                    self.row,
                    dma_ch,
                    mf.port_index,
                    mf.field.name
                );
                dma_ch += 1;
            }
        }
    }

    /// Parse Shim Demux_Config register to find DMA S2MM South master mapping.
    ///
    /// The Shim Demux selects which destination (PL/DMA/NoC) receives switchbox
    /// South master output. DMA S2MM input comes from a South master.
    ///
    /// Field layout and port mappings are derived from the AM025 register database.
    /// Select values: 0=South/PL, 1=DMA, 2=NoC
    pub(crate) fn parse_shim_demux_config(&mut self, value: u32) {
        let mux = &super::regdb::device_reg_layout().shim_mux;

        // Reset mapping
        self.shim_mux_s2mm_masters.fill(None);

        // Per aie-rt (xaie_plif.c), the AIE2 shim demux has a FIXED mapping
        // from south ports to S2MM DMA channels:
        //   South2 (master[4]) -> S2MM ch0
        //   South3 (master[5]) -> S2MM ch1
        // The demux_fields are ordered South2, South3, South4, South5 and
        // the DMA channel index matches the field's position in the valid
        // S2MM port list (first two fields = ch0 and ch1).
        for (dma_ch, df) in mux.demux_fields.iter().enumerate() {
            let select = df.field.extract(value);
            if select == 1 && dma_ch < self.shim_mux_s2mm_masters.len() {
                self.shim_mux_s2mm_masters[dma_ch] = Some(df.port_index);
                log::info!(
                    "Shim Mux ({},{}): S2MM ch{} <- master[{}] ({})",
                    self.col,
                    self.row,
                    dma_ch,
                    df.port_index,
                    df.field.name
                );
            }
        }
    }

    // Control packet processing has been moved to control_packets::StreamReassembler
    // in array.rs. The tile no longer owns the word-by-word FSM.
}
