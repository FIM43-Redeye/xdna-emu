//! Cycle-accurate executor implementation.
//!
//! The `CycleAccurateExecutor` models pipeline stages, register hazards, and
//! memory bank conflicts to provide accurate cycle counts. This executor
//! provides the accurate timing behavior that matches real hardware.
//!
//! # Timing Model
//!
//! Based on AM020 (AIE-ML Architecture Manual):
//! - Scalar add/sub/logic: 1 cycle
//! - Scalar multiply: 2 cycles
//! - Memory access: 5 cycles
//! - Vector MAC: 4 cycles
//! - Branch taken: 3 cycles
//!
//! # Hazard Detection
//!
//! - **RAW** (Read After Write): Reading a register before its write completes
//! - **WAW** (Write After Write): Writing a register before previous write completes
//!
//! Stalls are inserted automatically to resolve hazards.
//!
//! # Memory Bank Conflicts
//!
//! The AIE2 has 8 memory banks. Accessing the same bank from multiple ports
//! in the same cycle causes a stall.

use crate::device::arch_handle;
use crate::device::tile::Tile;
use crate::interpreter::bundle::{Operand, SlotOp, VliwBundle};
use xdna_archspec::aie2::isa::SemanticOp;
use xdna_archspec::aie2::Bypass;
use crate::interpreter::state::{EventType, ExecutionContext};
use crate::interpreter::timing::{HazardDetector, HazardStats, MemoryQuadrant};
use crate::interpreter::traits::{ExecuteResult, Executor};

use super::cascade::{CascadeOps, CascadeResult};
use super::control::ControlUnit;
use super::memory::{MemoryUnit, NeighborMemory};
use crate::device::state::NeighborView;
use super::semantic::execute_semantic;
use super::stream::{StreamOps, StreamResult};
use super::vector_dispatch::VectorAlu;

/// Cycle-accurate executor that models pipeline timing.
///
/// This executor tracks:
/// - Instruction latencies
/// - Register hazards (RAW, WAW)
/// - Memory bank conflicts
/// - Pipeline stalls
///
/// Use this for:
/// - Performance analysis
/// - Cycle-accurate simulation
/// - Validation against hardware
pub struct CycleAccurateExecutor {
    /// Track call return address.
    pending_call_return_addr: Option<u32>,

    /// Latency lookup table (process-global, from arch_handle).
    latencies: &'static crate::interpreter::timing::LatencyTable,

    /// Register hazard detector.
    hazards: HazardDetector,

    /// Total stall cycles from hazards.
    pub total_hazard_stalls: u64,

    /// Total stall cycles from branch penalties.
    pub total_branch_stalls: u64,

    /// Detailed stall breakdown by reason.
    detailed_stats: HazardStats,
}

impl CycleAccurateExecutor {
    /// Create a new cycle-accurate executor.
    ///
    /// The latency table is sourced from the process-global
    /// `arch_handle::latency_table()` cache, which is populated once from
    /// the LLVM FFI and reused across all executor instances.
    pub fn new() -> Self {
        Self {
            pending_call_return_addr: None,
            latencies: arch_handle::latency_table(),
            hazards: HazardDetector::new(),
            total_hazard_stalls: 0,
            total_branch_stalls: 0,
            detailed_stats: HazardStats::default(),
        }
    }

    /// Calculate the execution cycles for a slot operation.
    ///
    /// Uses the TableGen-derived SemanticOp for latency lookup.
    fn operation_cycles(&self, op: &SlotOp) -> u8 {
        self.latencies.timing_for_slot_op(op).latency
    }

    /// Execute a single slot operation with optional neighbor lock routing.
    fn execute_slot_with_neighbor_locks(
        &mut self,
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        neighbor_locks: &mut super::control::NeighborLocks,
        neighbors: Option<&mut NeighborMemory>,
        view: Option<&NeighborView>,
    ) -> Option<ExecuteResult> {
        // Check for call - save return address
        if matches!(op.semantic, Some(SemanticOp::Call)) {
            self.pending_call_return_addr = Some(ctx.pc());
        }

        // Execute using the functional units.
        // Semantic dispatch handles pure register operations (scalar arithmetic,
        // logic, moves, flags). Falls through for vector, memory, stream,
        // cascade, and control flow ops.
        if execute_semantic(op, ctx) {
            return None;
        }

        if VectorAlu::execute(op, ctx) {
            return None;
        }

        if MemoryUnit::execute(op, ctx, tile, neighbors, view) {
            return None;
        }

        // Cascade operations: dedicated point-to-point link, separate from the
        // stream switch fabric. Stall returns WaitStream with sentinel ports:
        // port 254 = SCD read stall (empty input), port 255 = MCD write stall (full output).
        match CascadeOps::execute(op, ctx, tile) {
            CascadeResult::Completed => return None,
            CascadeResult::StallRead => return Some(ExecuteResult::WaitStream { port: 254 }),
            CascadeResult::StallWrite => return Some(ExecuteResult::WaitStream { port: 255 }),
            CascadeResult::NotCascadeOp => {}
            CascadeResult::Error(msg) => return Some(ExecuteResult::Error { message: msg }),
        }

        // Stream operations: read/write between scalar registers and stream
        // switch ports. Stall on blocking read from empty stream.
        match StreamOps::execute(op, ctx, tile) {
            StreamResult::Completed => return None,
            StreamResult::Stall { port } => return Some(ExecuteResult::WaitStream { port }),
            StreamResult::NotStreamOp => {}
        }

        if let Some(result) = ControlUnit::execute_with_neighbor_locks(op, ctx, tile, neighbor_locks) {
            return Some(result);
        }

        // Defense in depth: instructions without a semantic label have no
        // routing information at all. With 100% semantic coverage this path
        // should be unreachable, but keep it as a safety net.
        if op.semantic.is_none() {
            return Some(ExecuteResult::Error {
                message: format!(
                    "Unknown instruction (no semantic): slot={:?}, name={:?}, opcode={:#X?}",
                    op.slot, op.encoding_name, op.raw_opcode,
                ),
            });
        }

        // Every instruction with a semantic label should be claimed by one
        // of the execution units above. Reaching here means a labeled but
        // unimplemented op -- abort immediately rather than silently
        // producing wrong results by pretending execution succeeded.
        Some(ExecuteResult::Error {
            message: format!(
                "Unhandled instruction: semantic={:?}, slot={:?}, name={:?}",
                op.semantic, op.slot, op.encoding_name,
            ),
        })
    }

    /// Record register writes for hazard tracking.
    fn record_writes(&mut self, op: &SlotOp) {
        let latency = self.operation_cycles(op);
        self.hazards.record_operation(op, latency);
    }

    /// Fire watchpoints for a memory access.
    ///
    /// Bank-conflict detection USED to live here: it ran at bundle-commit
    /// time, saw only this bundle's own scalar slots, charged its stall to a
    /// counter nothing gated on, and emitted its own CONFLICT_DM_BANK_n
    /// events. That model is retired -- conflicts are now arbitrated BEFORE
    /// either agent commits, over the core's real ports AND the DMA channels,
    /// by `crate::device::bank_arbiter` driven from the coordinator's
    /// request -> arbitrate -> commit loop, which is also the single emission
    /// point for MEMORY_STALL and CONFLICT_DM_BANK_n. Leaving the old
    /// detection here would double-emit both events.
    ///
    /// Watchpoints are unrelated to arbitration and stay: the HW comparator
    /// sits at the bank interface and fires on every matching access,
    /// scalar or vector.
    fn record_memory_access(&mut self, op: &SlotOp, ctx: &mut ExecutionContext, tile: &mut Tile) {
        if !matches!(op.semantic, Some(SemanticOp::Load) | Some(SemanticOp::Store)) {
            return;
        }
        // Resolve the effective address using the same helpers the actual
        // load/store path uses. This picks up indexed addressing through
        // modifier registers (`[pN, mK]`), which the previous ad-hoc resolver
        // dropped -- a non-zero modifier shifted the recorded address out
        // from under both bank-conflict tracking and watchpoint matching.
        // Post-modify (`op.post_modify`) is intentionally NOT applied: it
        // updates the base register *after* the access, so the address that
        // hits memory this cycle is the pre-modify value.
        let is_store = matches!(op.semantic, Some(SemanticOp::Store));
        let addr = if is_store {
            MemoryUnit::get_store_address(op, ctx)
        } else {
            MemoryUnit::get_address(op, ctx)
        };
        let cycle = ctx.cycles;
        let pc = ctx.pc();

        // Watchpoints fire on every matching access regardless of issuing
        // engine: HW comparator sits at the bank interface and sees both
        // scalar and vector traffic. Programmed slots compare the access
        // address (low bits masked per HW) and direction filter against each
        // WatchPointN register, then notify the mem-module trace unit and
        // perf counters with WATCHPOINT_N events.
        fire_watchpoint_events(tile, addr, is_store, cycle, pc);
    }

    /// Banks this bundle's memory slots will need, WITHOUT executing it.
    ///
    /// Address generation is a pure function of register state (see
    /// `MemoryUnit::get_address` / `get_store_address`), so the arbiter can be
    /// asked for the banks a bundle will touch before deciding whether to let
    /// it commit. This is the "request" half of request -> arbitrate ->
    /// commit; nothing here advances the PC, the cycle counter, or moves any
    /// data -- `ctx` is taken by shared reference.
    ///
    /// Only LOCAL data-memory accesses count: cross-tile (neighbour)
    /// addresses decode to a non-`Local` `MemoryQuadrant` and are excluded,
    /// mirroring the `quadrant == MemoryQuadrant::Local` gate the load/store
    /// sites already use. A bundle with no local memory ops (or none at all)
    /// returns an empty list and can always issue.
    ///
    /// Unlike `record_memory_access`'s stall counter (scalar-only, see the
    /// comment there), this covers vector ops too: `banks_for_access` already
    /// walks every 16-byte word a wide access spans, and per-bank arbitration
    /// doesn't care about port width.
    ///
    /// Returns ONE entry per core memory PORT that has local demand this
    /// cycle, not one OR'd mask -- a physical bank is genuinely single-port,
    /// so the core's own load and store ports must be able to contend with
    /// each other when they land on the same physical bank (AM020 ch.4:69;
    /// see the design doc this implements). `op.slot` already tells us
    /// exactly which port (`LoadA` / `LoadB` / `Store`) issued each access --
    /// the VLIW bundle format has independent bit fields per port, so a
    /// bundle can never carry more than one Load/Store op per port; there is
    /// no "more slots than ports" case to handle here.
    ///
    /// `served` is the STICKY-GRANT set: core ports that already won their
    /// banks on an earlier cycle of this same stalled bundle. Their accesses
    /// completed and latched in hardware, so on the retry cycle only the
    /// UNSERVED ports re-request (AM020 ch.2:166 -- the losers retry, not the
    /// winners). Passing `&[]` means "fresh bundle, nothing served yet". The
    /// coordinator accumulates this set from `Arbitration::granted_core_ports`
    /// across the stall; re-presenting the full demand every cycle instead
    /// would deterministically livelock any bundle whose own two ports target
    /// one single-port bank.
    pub fn peek_bank_demand(
        &self,
        bundle: &VliwBundle,
        ctx: &ExecutionContext,
        layout: crate::device::banking::BankLayout,
        served: &[crate::device::bank_arbiter::CorePort],
    ) -> Vec<(crate::device::bank_arbiter::Requester, u16)> {
        use crate::device::bank_arbiter::{CorePort, Requester};
        use crate::device::banking::banks_for_access;
        use crate::interpreter::bundle::SlotIndex;

        let mut demand = Vec::new();
        for op in bundle.active_slots() {
            if !matches!(op.semantic, Some(SemanticOp::Load) | Some(SemanticOp::Store)) {
                continue;
            }
            let is_store = matches!(op.semantic, Some(SemanticOp::Store));
            let addr = if is_store {
                MemoryUnit::get_store_address(op, ctx)
            } else {
                MemoryUnit::get_address(op, ctx)
            };
            let (quadrant, local_offset) = super::memory::decode_data_address(addr);
            if quadrant != MemoryQuadrant::Local {
                continue;
            }
            let mask = banks_for_access(local_offset as u32, op.mem_width.bytes() as usize, layout);
            if mask == 0 {
                continue;
            }
            let port = match op.slot {
                SlotIndex::LoadA => CorePort::LoadA,
                SlotIndex::LoadB => CorePort::LoadB,
                SlotIndex::Store => CorePort::Store,
                // This is a per-issuing-core-cycle hot path (task 6 review,
                // Minor-1): a decoder-data regression here must degrade, not
                // panic the emulator. `debug_assert!` still catches the
                // invariant violation loudly in debug/test builds; in
                // release the op is simply skipped from this cycle's bank
                // demand (the same as any other non-memory op).
                other => {
                    debug_assert!(
                        false,
                        "Load/Store semantic decoded into non-memory slot {other:?} -- the VLIW \
                         encoding only ever places Load semantics in LoadA/LoadB and Store \
                         semantics in Store"
                    );
                    continue;
                }
            };
            if served.contains(&port) {
                continue; // won an earlier cycle of this stall -- already latched
            }
            demand.push((Requester::Core(port), mask));
        }
        demand
    }

    /// Reset executor state (for new program).
    pub fn reset(&mut self) {
        self.pending_call_return_addr = None;
        self.hazards.reset();
        self.total_hazard_stalls = 0;
        self.total_branch_stalls = 0;
        self.detailed_stats = HazardStats::default();
    }

    /// Get statistics.
    pub fn stats(&self) -> CycleAccurateStats {
        // Merge detailed stats with hazard detector stats
        let mut combined_stats = self.detailed_stats;
        combined_stats.merge(&self.hazards.stats());

        CycleAccurateStats {
            hazard_stalls: self.total_hazard_stalls,
            branch_stalls: self.total_branch_stalls,
            hazard_stats: combined_stats,
        }
    }

    /// Get detailed stall breakdown.
    pub fn detailed_stats(&self) -> &HazardStats {
        &self.detailed_stats
    }
}

impl Default for CycleAccurateExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Fire one `MEM_CONFLICT_DM_BANK_N` event per bit set in `banks_mask` to the
/// tile's mem-module trace and perf counters. Tile-kind dispatch picks the
/// right event-ID base (compute mem 77, memtile 112) per `xaie_events_aieml.h`.
///
/// Sole caller is the coordinator's arbitration phase (the per-physical-bank
/// round-robin arbiter is what decides a bank is contended); `pc` is `None`
/// there because a memory-module event carries no program counter.
pub(crate) fn fire_bank_conflict_events(tile: &mut Tile, banks_mask: u16, cycle: u64, pc: Option<u32>) {
    use crate::device::tile::TileKind;
    use xdna_archspec::aie2::trace_events::{mem_events, memtile_events};
    let base = match tile.tile_kind {
        TileKind::Compute => mem_events::CONFLICT_DM_BANK_0,
        TileKind::Mem => memtile_events::CONFLICT_DM_BANK_0,
        TileKind::ShimNoc | TileKind::ShimPl => return, // no mem module
    };
    for bank in 0..8u8 {
        if banks_mask & (1 << bank) != 0 {
            let event_id = base + bank;
            // Same routing rationale as fire_watchpoint_events: dispatcher
            // for trace + timer + edge + halt; perf counters separately.
            tile.notify_mem_trace_event(event_id, cycle, pc);
            tile.mem_perf_counters.handle_event(event_id);
        }
    }
}

/// Initiator of a memory access seen by the watchpoint comparator.
///
/// AM025 watchpoint registers separately enable filters for AXI accesses,
/// local DMA accesses, and accesses originating from each neighbour
/// quadrant. `Core` is the implicit fall-through: a core load/store on the
/// executing tile, with no AM025 enable bit of its own. When the slot's
/// origin filter bits ([29:24] compute / [27:24] memtile) are all zero,
/// the slot fires regardless of origin (wildcard); when any are set, the
/// access origin must match one of the enabled bits.
// `Axi` has no consumer yet; `Dma` is wired from the DMA engine (task #68),
// `Neighbour` from the cross-tile MemTile-to-MemTile DMA path (task #69).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessOrigin {
    /// Local core load/store on the executing tile.
    Core,
    /// AXI-mapped access (host-side or NoC).
    Axi,
    /// Local DMA engine on the executing tile.
    Dma,
    /// Cross-tile access from the named neighbour quadrant. `Local` is
    /// equivalent to `Core` for filter purposes (own-tile access).
    Neighbour(MemoryQuadrant),
}

/// Per-tile-kind layout of the WatchPoint registers.
///
/// Compute mem-module exposes 2 slots at 0x14100/4 with direction bits at
/// 31 (Read) / 30 (Write) and a 12-bit address comparator [15:4]
/// (16-byte aligned, covers the full 64KB local memory). Memtile exposes 4
/// slots at 0x94100..0x9410C with direction bits at 29 / 28 and a 15-bit
/// comparator [18:4] (16-byte aligned, covers the full 512KB memtile mem).
/// Both layouts gate the slot on WriteStrobes [23:20] == 0xF, the AM025
/// "active slot" sentinel.
///
/// Origin filter bit positions per AM025:
///   compute: AXI [29], DMA [28], East [27], North [26], West [25], South [24]
///   memtile: AXI [27], DMA [26], East [25], West [24]   (no N/S)
struct WatchpointLayout {
    slot_offsets: &'static [u32],
    addr_mask: u32,
    read_bit: u8,
    write_bit: u8,
    axi_bit: u8,
    dma_bit: u8,
    east_bit: u8,
    west_bit: u8,
    /// Memtile lacks N/S quadrant filter bits.
    north_bit: Option<u8>,
    south_bit: Option<u8>,
    event_base: u8,
}

fn watchpoint_layout(kind: crate::device::tile::TileKind) -> Option<WatchpointLayout> {
    use crate::device::tile::TileKind;
    use xdna_archspec::aie2::trace_events::{mem_events, memtile_events};
    match kind {
        TileKind::Compute => Some(WatchpointLayout {
            slot_offsets: &[0x14100, 0x14104],
            addr_mask: 0xFFF0,
            read_bit: 31,
            write_bit: 30,
            axi_bit: 29,
            dma_bit: 28,
            east_bit: 27,
            north_bit: Some(26),
            west_bit: 25,
            south_bit: Some(24),
            event_base: mem_events::WATCHPOINT_0,
        }),
        TileKind::Mem => Some(WatchpointLayout {
            slot_offsets: &[0x94100, 0x94104, 0x94108, 0x9410C],
            addr_mask: 0x7FFF0,
            read_bit: 29,
            write_bit: 28,
            axi_bit: 27,
            dma_bit: 26,
            east_bit: 25,
            west_bit: 24,
            north_bit: None,
            south_bit: None,
            event_base: memtile_events::WATCHPOINT_0,
        }),
        TileKind::ShimNoc | TileKind::ShimPl => None,
    }
}

/// Build the origin-filter bitmask currently programmed in `value`. Each
/// bit position corresponds to one origin (AXI/DMA/E/N/W/S). When the
/// returned mask is 0 the slot is wildcard (all origins match).
fn slot_origin_filter(value: u32, layout: &WatchpointLayout) -> u32 {
    let mut mask = 0u32;
    mask |= ((value >> layout.axi_bit) & 1) << layout.axi_bit;
    mask |= ((value >> layout.dma_bit) & 1) << layout.dma_bit;
    mask |= ((value >> layout.east_bit) & 1) << layout.east_bit;
    mask |= ((value >> layout.west_bit) & 1) << layout.west_bit;
    if let Some(nb) = layout.north_bit {
        mask |= ((value >> nb) & 1) << nb;
    }
    if let Some(sb) = layout.south_bit {
        mask |= ((value >> sb) & 1) << sb;
    }
    mask
}

/// Map an AccessOrigin to the bit position it requires in the slot's
/// origin filter. Returns None for `Core` and `Neighbour(Local)` -- those
/// origins have no AM025 enable bit, so a non-zero filter mask never
/// matches them.
fn origin_required_bit(origin: AccessOrigin, layout: &WatchpointLayout) -> Option<u8> {
    match origin {
        AccessOrigin::Core | AccessOrigin::Neighbour(MemoryQuadrant::Local) => None,
        AccessOrigin::Axi => Some(layout.axi_bit),
        AccessOrigin::Dma => Some(layout.dma_bit),
        AccessOrigin::Neighbour(MemoryQuadrant::East) => Some(layout.east_bit),
        AccessOrigin::Neighbour(MemoryQuadrant::West) => Some(layout.west_bit),
        AccessOrigin::Neighbour(MemoryQuadrant::North) => layout.north_bit,
        AccessOrigin::Neighbour(MemoryQuadrant::South) => layout.south_bit,
    }
}

/// Pure decision: which WATCHPOINT_N event IDs should fire for a Core
/// access at `addr` with direction `is_write`. Convenience wrapper for
/// the common case; for non-Core origins, see
/// [`matching_watchpoint_events_with_origin`].
#[cfg(test)]
fn matching_watchpoint_events(tile: &Tile, addr: u32, is_write: bool) -> Vec<u8> {
    matching_watchpoint_events_with_origin(tile, addr, is_write, AccessOrigin::Core)
}

/// Pure decision with explicit access origin. The slot fires when:
///   - WriteStrobes [23:20] == 0xF (slot active)
///   - The matching direction bit is set
///   - The address comparator matches (after masking)
///   - AND if any origin filter bits are set in the slot, the access
///     origin must correspond to one of those bits. When all filter
///     bits are zero, the slot is wildcard and any origin (including
///     Core) matches.
fn matching_watchpoint_events_with_origin(
    tile: &Tile,
    addr: u32,
    is_write: bool,
    origin: AccessOrigin,
) -> Vec<u8> {
    let Some(layout) = watchpoint_layout(tile.tile_kind) else {
        return Vec::new();
    };
    let mut hits = Vec::new();
    for (i, &reg_off) in layout.slot_offsets.iter().enumerate() {
        let Some(&value) = tile.registers.get(&reg_off) else {
            continue;
        };
        // WriteStrobes [23:20] == 0xF gates the slot. AM025: "always set this
        // field to 0xF when using this watchpoint" -- we treat it as the
        // active/inactive sentinel.
        if (value >> 20) & 0xF != 0xF {
            continue;
        }
        let dir_match = if is_write {
            (value >> layout.write_bit) & 1 != 0
        } else {
            (value >> layout.read_bit) & 1 != 0
        };
        if !dir_match {
            continue;
        }
        if (addr & layout.addr_mask) != (value & layout.addr_mask) {
            continue;
        }
        // Origin filter: when any of the AM025 origin bits ([29:24] compute,
        // [27:24] memtile) are set, the access must match one of them.
        // Wildcard (all bits 0) admits any origin, including Core.
        let origin_filter = slot_origin_filter(value, &layout);
        if origin_filter != 0 {
            match origin_required_bit(origin, &layout) {
                Some(bit) if origin_filter & (1u32 << bit) != 0 => {}
                _ => continue,
            }
        }
        hits.push(layout.event_base + i as u8);
    }
    hits
}

/// Fire WATCHPOINT_N events into the mem-module trace unit and perf counters
/// for every WatchPoint slot that matches this access. Compute and memtile
/// have different bit layouts (`watchpoint_layout`); shim tiles have no
/// watchpoint registers and short-circuit. Convenience wrapper for the
/// Core-origin path (scalar and vector loads/stores).
fn fire_watchpoint_events(tile: &mut Tile, addr: u32, is_write: bool, cycle: u64, pc: u32) {
    fire_watchpoint_events_with_origin(tile, addr, is_write, cycle, Some(pc), AccessOrigin::Core);
}

/// Same as [`fire_watchpoint_events`] but with an explicit access origin and
/// optional PC. Used by the DMA engine (task #68) to fire WATCHPOINT_N with
/// `AccessOrigin::Dma` and no PC (DMA accesses are external to the core).
pub(crate) fn fire_watchpoint_events_with_origin(
    tile: &mut Tile,
    addr: u32,
    is_write: bool,
    cycle: u64,
    pc: Option<u32>,
    origin: AccessOrigin,
) {
    let events = matching_watchpoint_events_with_origin(tile, addr, is_write, origin);
    for event_id in events {
        // Route through notify_mem_trace_event so trace + timer + edge
        // detector + debug-halt check all see the event. Perf counters
        // aren't on that path so we tick them explicitly.
        tile.notify_mem_trace_event(event_id, cycle, pc);
        tile.mem_perf_counters.handle_event(event_id);
    }
}

impl CycleAccurateExecutor {
    /// Execute a bundle with optional neighbor lock routing and cross-tile memory.
    ///
    /// For compute tiles, pass neighbor locks to enable cross-tile lock
    /// access per AIE2 quadrant mapping (getLockLocalBaseIndex).
    /// Pass `neighbors` to enable cross-tile memory access (quadrants 1-3).
    pub fn execute_with_mem_tile(
        &mut self,
        bundle: &VliwBundle,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        mem_tile_locks: Option<&mut [crate::device::tile::Lock]>,
        neighbors: Option<&mut NeighborMemory>,
        view: Option<&NeighborView>,
    ) -> ExecuteResult {
        let mut nlocks = super::control::NeighborLocks::south_only(mem_tile_locks);
        self.execute_internal(bundle, ctx, tile, &mut nlocks, neighbors, view)
    }

    /// Execute with full neighbor lock routing (all four quadrants).
    pub fn execute_with_neighbor_locks(
        &mut self,
        bundle: &VliwBundle,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        neighbor_locks: &mut super::control::NeighborLocks,
        neighbors: Option<&mut NeighborMemory>,
        view: Option<&NeighborView>,
    ) -> ExecuteResult {
        self.execute_internal(bundle, ctx, tile, neighbor_locks, neighbors, view)
    }

    /// Internal execution with neighbor locks and cross-tile memory.
    fn execute_internal(
        &mut self,
        bundle: &VliwBundle,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        neighbor_locks: &mut super::control::NeighborLocks,
        mut neighbors: Option<&mut NeighborMemory>,
        view: Option<&NeighborView>,
    ) -> ExecuteResult {
        self.pending_call_return_addr = None;

        let pc = ctx.pc();
        let start_cycle = ctx.cycles;

        // Advance timing models to current cycle
        self.hazards.advance_to(ctx.cycles);

        // Phase 1: No scoreboard stalls
        //
        // AIE2 uses a write-back pipeline WITHOUT a hardware scoreboard.
        // The compiler (Chess/Peano) is responsible for scheduling loads
        // far enough ahead that no instruction reads the register before
        // the load result is written back. If code reads a register with
        // a pending load before ready_cycle, it gets the OLD value from
        // the register file -- this is correct hardware behavior.
        //
        // Chess software-pipelined code RELIES on this: prolog stores
        // intentionally read registers within the load latency window
        // to capture the previous iteration's computed values, while the
        // current iteration's load hasn't arrived yet.
        //
        // The forwarding functions (forward_scalar/pointer/modifier) in
        // ExecutionContext enforce the ready_cycle check, returning the
        // old register value when the load hasn't completed.

        // Phase 2: Execute all slot operations
        //
        // VLIW semantics require all reads to happen before any writes within
        // the same instruction word. We achieve this by executing slots in a
        // specific order:
        // 1. Load slots - read from memory
        // 2. Store slots - read from registers (must happen before scalar writes!)
        // 3. Scalar/Vector/Control - may write to registers
        //
        // This ensures that when a bundle contains both "st r7, [p1]" and
        // "add r7, r8, #1", the store captures r7's value BEFORE the add modifies it.
        use crate::interpreter::bundle::SlotIndex;

        let mut final_result = ExecuteResult::Continue;

        // Execution order: LoadA/LoadB first, then Store, then compute slots.
        // Both load ports issue in the same cycle (they share a load unit).
        //
        // When two slots write the same scalar register in one bundle, the
        // last writer (in this order) wins.  All reads use the pre-bundle
        // snapshot, so execution order does not affect read values.
        let execution_order = [
            SlotIndex::LoadA,   // Primary load port (LDA)
            SlotIndex::LoadB,   // Secondary load port (LDB)
            SlotIndex::Store,   // Register reads for stores (before scalar writes!)
            SlotIndex::Scalar0, // Scalar operations
            SlotIndex::Scalar1,
            SlotIndex::Vector,
            SlotIndex::Accumulator,
            SlotIndex::Control,
        ];

        // Commit deferred load results whose latency has elapsed.
        // This must happen BEFORE the VLIW snapshot so that load results
        // become visible at the correct cycle and are captured by begin_bundle().
        ctx.commit_pending_writes();

        // Commit deferred partial-word stores whose data-read cycle has arrived.
        // The data register is read NOW (at ready_cycle), not at issue time.
        // This models the II_STHB RMW pipeline (operand latency 7).
        let ready_stores = ctx.drain_ready_stores();
        for ps in &ready_stores {
            let value = super::memory::MemoryUnit::read_store_register(&ps.source, ctx, ps.width);
            let neighbor_ref = neighbors.as_mut().map(|n| &mut **n);
            super::memory::MemoryUnit::write_memory(tile, ps.address, value, ps.width, neighbor_ref, view);
        }

        // Pre-check stall conditions for cascade and stream slots.
        //
        // VLIW bundles must commit atomically: every slot completes in the
        // same cycle or none does. If a cascade/stream slot stalls partway
        // through the bundle and we keep executing sibling slots, those
        // siblings commit side effects (register writes, post-increments,
        // accumulating ALU ops) that then double-execute when the bundle is
        // retried next cycle. That corrupts running sums and pointer math.
        //
        // Cascade reads occupy the LoadA slot (executes first), so even an
        // unconditional `break` on stall would work for them. But cascade
        // writes occupy the Vector slot (executes after Loads/Stores/Scalars),
        // so by the time the stall is detected, sibling slots have already
        // committed. The only correct fix is to look ahead: scan every
        // active slot for a blocking condition before touching state.
        //
        // See cascade_matmul investigation 2026-04-12 for the failure that
        // motivated this (matmul_using_cascade::cascade was producing wrong
        // accumulator values whenever an `ADD r28, r14, r28` was bundled
        // with a `VMOV bml, SCD` and the SCD ran empty).
        for slot_idx in &execution_order {
            if let Some(ref op) = bundle.slots()[*slot_idx as usize] {
                if let Some(stall) = CascadeOps::would_stall(op, tile) {
                    let port = match stall {
                        CascadeResult::StallRead => 254,
                        CascadeResult::StallWrite => 255,
                        // would_stall only returns Stall* variants; other
                        // variants are unreachable.
                        _ => 254,
                    };
                    let stall_cycle = ctx.cycles;
                    // Rising edge of the cascade stall (held level). CASCADE_STALL
                    // (25) is a distinct event from STREAM_STALL (24); the falling
                    // edge is recorded in try_resume_stall when the cascade unblocks.
                    ctx.timing_context_mut()
                        .record_event(stall_cycle, EventType::CascadeStallLevel { active: true });
                    ctx.record_instruction(1);
                    let timing = ctx.timing_context_mut();
                    timing.hazard_stalls = self.total_hazard_stalls;
                    return ExecuteResult::WaitStream { port };
                }
                if let Some(super::stream::StreamResult::Stall { port }) =
                    super::stream::StreamOps::would_stall(op, tile)
                {
                    let stall_cycle = ctx.cycles;
                    // Rising edge of the stream stall (held level); falling edge in
                    // try_resume_stall when stream data resumes.
                    ctx.timing_context_mut()
                        .record_event(stall_cycle, EventType::StreamStallLevel { active: true });
                    ctx.record_instruction(1);
                    let timing = ctx.timing_context_mut();
                    timing.hazard_stalls = self.total_hazard_stalls;
                    return ExecuteResult::WaitStream { port };
                }
            }
        }

        // AIE2 uses pure VLIW semantics: all reads within a bundle see the
        // pre-execution values. This is confirmed by the llvm-aie scheduling
        // model (AIE2Schedule.td) and hazard recognizer which handle data
        // dependencies only across bundles, not within them.
        //
        // The begin_bundle snapshot ensures that when `lda r13, [p5]` and
        // `or r3, r13, r0` are in the same bundle, the OR reads r13's value
        // from BEFORE the load writes it.
        //
        // First advance the vector file's issued-bundle clock and land any
        // bypass-network writes that have reached architectural visibility, so
        // landed values are captured by the snapshot. This runs only on an
        // actually-issued bundle (stall pre-checks above return earlier), so
        // the clock tracks issued bundles, not stall cycles.
        ctx.advance_vector_bundle();

        // Sample sources for VUNPACKs whose stage-7 read bundle arrived
        // (after the vector clock advance so producer writes landing this
        // bundle are visible, before slot execution).
        ctx.process_pending_unpacks();

        // Complete deferred MAC-family multiplies whose stage-3 accumulator
        // read bundle arrived (after commit_pending_writes so a chained
        // VMUL's write landing this cycle is visible -- hardware forwarding;
        // before acc add/sub sampling so their relative order is preserved).
        super::vector_matmul::process_pending_matmuls(ctx);

        // Sample sources for accumulator add/subs whose stage-3 read bundle
        // arrived (after commit_pending_writes so MAC/conv results that land
        // this cycle are visible).
        ctx.process_pending_acc_adds();

        // Sample the shift register for fused vlda.ups whose stage-7 read
        // bundle arrived and write their converted accumulators.
        ctx.process_pending_ups_loads();

        ctx.begin_bundle();

        // Same-bundle scalar->shift forwarding. AIE2 writes an S register at
        // pipeline stage E1 but UPS/SRS sample their shift operand at E7
        // (llvm-aie AIE2Schedule.td: RAW latency -5), so a shift-setup
        // `MOV sN,#imm` bundled with its consumer must forward the new value.
        // Record those immediate writes before any slot executes; only
        // shift-operand reads consult them (ctx.shift_forward) -- general reads
        // keep pure read-old semantics. (Compiled UPS does exactly this: the
        // first vlda.ups is bundled with its `mov s0,#shift`.)
        for slot_idx in &execution_order {
            if let Some(ref op) = bundle.slots()[*slot_idx as usize] {
                if op.semantic == Some(SemanticOp::Copy) && op.sources.len() == 1 {
                    if let (Some(Operand::ScalarReg(r)), Operand::Immediate(v)) = (&op.dest, &op.sources[0]) {
                        ctx.record_bundle_scalar_imm(*r, *v as u32);
                    }
                }
            }
        }

        for slot_idx in &execution_order {
            if let Some(ref op) = bundle.slots()[*slot_idx as usize] {
                // Expose this op's result latency and result bypass class so a
                // vector-register write routes through the AIE2 forwarding
                // network (visible to ALU consumers at issue+1 when MOV_Bypass,
                // to stores at full latency). See VectorRegisterFile::resolve.
                // Set per slot before dispatch so each write sees its own op.
                //
                // result_bypass is the register-aware resolved bypass from the
                // decoded SlotOp, populated in try_decode_via_ffi via
                // Bypass::from_forwarding_id(ffi_result.resolved_def_bypass()).
                // This correctly handles register-pair-variant opcodes like
                // VMOV_mv_x (X<-BM carries MOV_Bypass; a static per-opcode
                // lookup on the base class would return No).
                ctx.result_latency = self.operation_cycles(op);
                ctx.result_bypass = op.result_bypass;

                // Reborrow neighbors for each slot operation
                let slot_neighbors = neighbors.as_mut().map(|n| &mut **n);
                if let Some(result) =
                    self.execute_slot_with_neighbor_locks(op, ctx, tile, neighbor_locks, slot_neighbors, view)
                {
                    match &result {
                        ExecuteResult::Branch { .. }
                        | ExecuteResult::Call { .. }
                        | ExecuteResult::Halt
                        | ExecuteResult::WaitLock { .. }
                        | ExecuteResult::WaitDma { .. }
                        | ExecuteResult::WaitStream { .. } => {
                            final_result = result;
                        }
                        ExecuteResult::Continue => {}
                        ExecuteResult::Error { .. } => {
                            return result;
                        }
                    }
                }

                // Emit per-slot instruction class event matching hardware trace codes.
                // Each active slot generates its own event, just like hardware where
                // different event types fire for each functional unit.
                //
                // NOPs occupy the slot structurally but don't activate the
                // functional unit on real hardware -- NOPA/NOPB don't dispatch
                // to the load units, NOPS doesn't write memory, NOPV doesn't
                // touch the vector pipeline. So they must NOT emit INSTR_LOAD
                // / INSTR_STORE / INSTR_VECTOR.  Skipping NOPs here matches
                // the AIE2 events module behaviour described in
                // `xaie_events_aieml.h`, where these events count unit
                // activity, not slot presence.
                let event = if op.is_nop() {
                    None
                } else {
                    match slot_idx {
                        SlotIndex::LoadA | SlotIndex::LoadB => Some(EventType::InstrLoad { pc }),
                        SlotIndex::Store => Some(EventType::InstrStore { pc }),
                        SlotIndex::Vector | SlotIndex::Accumulator => Some(EventType::InstrVector { pc }),
                        _ => None, // Scalar/Control events classified below
                    }
                };
                if let Some(evt) = event {
                    ctx.timing_context_mut().record_event(start_cycle, evt);
                }
            }
        }

        // Clear the per-slot result latency/bypass so any write outside bundle
        // dispatch (e.g. pipeline drain) defaults to an immediate write.
        ctx.result_latency = 0;
        ctx.result_bypass = Bypass::No;

        ctx.end_bundle();

        // Phase 3: Record writes and memory accesses for future hazard detection
        for op in bundle.active_slots() {
            self.record_writes(op);
            self.record_memory_access(op, ctx, tile);
        }

        // Convert Branch to Call when this was a jl instruction.
        // LR is NOT set here -- it's deferred until delay slots are exhausted
        // (in ExecutionContext::tick_delay_slots) so that delay slot instructions
        // see the pre-call LR value, matching hardware pipeline behavior.
        if self.pending_call_return_addr.is_some() {
            if let ExecuteResult::Branch { target } = final_result {
                final_result = ExecuteResult::Call { target };
            }
        }

        // Emit Call/Return/Halt events based on execution result.
        // These events map to hardware INSTR_CALL, INSTR_RETURN, DISABLED_CORE.
        match &final_result {
            ExecuteResult::Call { .. } => {
                ctx.timing_context_mut().record_event(start_cycle, EventType::InstrCall { pc });
            }
            // Return is detected by the control unit returning a Branch to LR.
            // We check if the branch target matches the link register value.
            ExecuteResult::Branch { target } if *target == ctx.lr() => {
                ctx.timing_context_mut()
                    .record_event(start_cycle, EventType::InstrReturn { pc });
            }
            ExecuteResult::Halt => {
                ctx.timing_context_mut().record_event(start_cycle, EventType::CoreDisabled);
                // Flush remaining pending stores: advance cycles and drain until empty.
                // When the core halts, partial-word stores queued in the last 7 cycles
                // haven't committed yet. The hardware pipeline drains them before
                // the core truly stops.
                for _ in 0..10 {
                    if ctx.pending_stores_empty() {
                        break;
                    }
                    ctx.cycles += 1;
                    ctx.commit_pending_writes();
                    let stores = ctx.drain_ready_stores();
                    for ps in &stores {
                        let value = super::memory::MemoryUnit::read_store_register(&ps.source, ctx, ps.width);
                        let neighbor_ref = neighbors.as_mut().map(|n| &mut **n);
                        super::memory::MemoryUnit::write_memory(
                            tile,
                            ps.address,
                            value,
                            ps.width,
                            neighbor_ref,
                            view,
                        );
                    }
                }
            }
            ExecuteResult::WaitLock { .. } => {
                // Rising edge of the lock stall (held level). The matching
                // falling edge is recorded in try_resume_stall when the lock
                // becomes available. Replaces the former per-cycle pulse, which
                // over-emitted LOCK_STALL ~375x vs HW.
                ctx.timing_context_mut()
                    .record_event(start_cycle, EventType::LockStallLevel { active: true });
            }
            ExecuteResult::WaitStream { port } => {
                // Rising edge of the stall (held level); falling edge in
                // try_resume_stall. Sentinel ports 254/255 are cascade
                // read/write stalls -> CASCADE_STALL (25), a distinct event
                // from the regular-port STREAM_STALL (24).
                let event = if *port == 254 || *port == 255 {
                    EventType::CascadeStallLevel { active: true }
                } else {
                    EventType::StreamStallLevel { active: true }
                };
                ctx.timing_context_mut().record_event(start_cycle, event);
            }
            _ => {}
        }

        // Emit branch event (no explicit penalty -- delay slots handle it).
        //
        // AIE2 has `processor::BRANCH_DELAY_SLOTS` (= 5) delay slots after
        // every taken branch. The compiler fills them with useful work (or
        // NOPs if it can't). Each delay-slot instruction costs 1 cycle via
        // record_instruction(1) like any other bundle. This naturally produces
        // the correct branch cost: filled slots = 0 extra cycles, unfilled =
        // 1 per NOP.
        //
        // Adding BRANCH_PENALTY_CYCLES on top of delay slots would
        // double-count: the pipeline flush IS the delay slots.
        //
        // Mode-2 trace notifications (notify_atom / notify_branch_taken) are
        // emitted from `control.rs` at branch-resolution time, where the
        // semantic context (conditional vs unconditional, direct vs
        // indirect target) is still available. This event is
        // timing-bookkeeping only.
        let branch_target = match final_result {
            ExecuteResult::Branch { target } | ExecuteResult::Call { target } => Some(target),
            _ => None,
        };
        if let Some(target) = branch_target {
            let branch_cycle = ctx.cycles;
            ctx.timing_context_mut()
                .record_event(branch_cycle, EventType::BranchTaken { from_pc: pc, to_pc: target });
        }

        // AIE2 lock-arbitration latency. Every completed lock transaction
        // (acquire or release) costs one extra core cycle beyond issue, even
        // when uncontended: the lock sits behind the tile memory module's
        // round-robin arbiter, which grants one request per cycle (AM020 ch.2
        // memory-bank arbitration). HW trails every LOCK_ACQUIRE_REQ /
        // LOCK_RELEASE_REQ with a 1-cycle LOCK_STALL pulse; the aiesim ISS (and
        // our prior model) treated uncontended locks as zero-stall (the
        // documented per-lock-transaction known-fidelity gap, 2026-06-07).
        // Charged only when the transaction completes (Continue) -- a blocked
        // acquire pays it when it later resumes and succeeds, not while it
        // stalls. DMA-engine lock transactions are timed separately in the DMA
        // model; this is the compute core's lock path only.
        if matches!(final_result, ExecuteResult::Continue)
            && bundle
                .active_slots()
                .any(|op| matches!(op.semantic, Some(SemanticOp::LockAcquire | SemanticOp::LockRelease)))
        {
            ctx.record_stall(1);
        }

        // Advance cycle counter by 1 (pipelined issue rate).
        // Latency-based deferred writes and hazard stalls handle the rest.
        ctx.record_instruction(1);

        // Count this issued bundle. bundle_seq is the issue-slot-relative clock
        // that drives the vector register file's bypass-network visibility (see
        // VectorRegisterFile::resolve / advance_bundle). Cascade/stream stall
        // pre-checks return before this point, so a stall advances `cycles` but
        // NOT `bundle_seq` -- vector-write latency stays robust to stalls.
        ctx.bundle_seq += 1;

        // Sync timing context. `memory_stalls` is deliberately NOT synced
        // here: bank-conflict stalls are owned by the coordinator's arbiter
        // (they are charged on cycles this bundle does NOT commit), and
        // assigning the executor's now-retired counter over it would zero the
        // coordinator's count on every committed bundle.
        let timing = ctx.timing_context_mut();
        timing.hazard_stalls = self.total_hazard_stalls;

        final_result
    }
}

impl Executor for CycleAccurateExecutor {
    fn execute(&mut self, bundle: &VliwBundle, ctx: &mut ExecutionContext, tile: &mut Tile) -> ExecuteResult {
        // Delegate to internal implementation with no neighbor locks or memory
        self.execute_internal(bundle, ctx, tile, &mut super::control::NeighborLocks::none(), None, None)
    }

    fn is_cycle_accurate(&self) -> bool {
        true
    }
}

/// Statistics from cycle-accurate execution.
#[derive(Debug, Clone)]
pub struct CycleAccurateStats {
    /// Total cycles stalled due to register hazards.
    pub hazard_stalls: u64,
    /// Total cycles stalled due to branch penalties.
    pub branch_stalls: u64,
    /// Detailed hazard statistics.
    pub hazard_stats: crate::interpreter::timing::hazards::HazardStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::{MemWidth, Operand, SlotIndex};
    use xdna_archspec::aie2::isa::SemanticOp;

    fn make_bundle(ops: Vec<SlotOp>) -> VliwBundle {
        let mut bundle = VliwBundle::empty();
        for op in ops {
            bundle.set_slot(op);
        }
        bundle
    }

    #[test]
    fn test_cycle_accurate_creation() {
        let executor = CycleAccurateExecutor::new();
        assert!(executor.is_cycle_accurate());
    }

    /// Same-bundle scalar->shift forwarding: `MOV s0, #4` and
    /// `VLDA.UPS.s32.s16 bml0, s0, [p0]` in one VLIW bundle. The S register is
    /// written at pipeline stage E1; the UPS samples its shift operand at E7
    /// (llvm-aie AIE2Schedule.td gives RAW latency -5), so the UPS must see the
    /// NEW s0=4. A pure read-old snapshot leaves the load unshifted. Surfaced by
    /// the compiled vec_ups_i32 kernel, whose first UPS load is bundled with its
    /// shift-setup MOV (scalar regs are 0 at entry, so without forwarding the
    /// first 16 lanes come out unshifted).
    #[test]
    fn test_same_bundle_scalar_forwards_to_ups_shift() {
        use crate::interpreter::bundle::ElementType;
        use crate::interpreter::decode::register_map::AccumWidth;

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // 16 int16 values at local 0x200.
        let vals: [i16; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        for i in 0..16 {
            let b = (vals[i] as u16).to_le_bytes();
            tile.data_memory_mut()[0x200 + i * 2] = b[0];
            tile.data_memory_mut()[0x200 + i * 2 + 1] = b[1];
        }
        ctx.pointer.write(0, 0x70200);
        ctx.scalar.write(0, 0); // pre-bundle s0 = 0; the same-bundle MOV sets the shift

        // Scalar0: MOV s0, #4   (E1 write of the shift)
        let mov = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::Immediate(4));

        // LoadA: vlda.ups.s32.s16 bml0, s0, [p0]   (E7 read of the shift)
        let mut ups = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Ups)
            .with_dest(Operand::AccumReg(0))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::PointerReg(0));
        ups.from_type = Some(ElementType::Int16);
        ups.element_type = Some(ElementType::Int32);
        ups.accum_width = Some(AccumWidth::Half);
        ups.mem_width = MemWidth::Vector256;
        ups.is_vector = true;

        executor.execute(&make_bundle(vec![mov, ups]), &mut ctx, &mut tile);

        // The UPS samples the shift at issue+6 (II_VLDA_UPS operand cycle 7);
        // issue empty bundles to reach the read bundle.
        for _ in 0..6 {
            executor.execute(&make_bundle(vec![]), &mut ctx, &mut tile);
        }

        // acc32 packs two int32 per u64: lane[2j]=acc[j] low, lane[2j+1]=high.
        // With the same-bundle shift forwarded, every lane = value << 4.
        let acc = ctx.accumulator.read(0);
        for j in 0..8usize {
            let lo = (acc[j] & 0xFFFF_FFFF) as i32;
            let hi = (acc[j] >> 32) as i32;
            assert_eq!(lo, (vals[2 * j] as i32) << 4, "lane {}", 2 * j);
            assert_eq!(hi, (vals[2 * j + 1] as i32) << 4, "lane {}", 2 * j + 1);
        }
    }

    /// Cross-bundle late shift write: the `MOV s0,#1` issued several bundles
    /// AFTER the vlda.ups must still control the conversion. II_VLDA_UPS reads
    /// its shift S-register at operand cycle 7, and Peano schedules the
    /// shift-setup mov in the load's shadow (vector fuzzer seeds 1142-1448,
    /// silicon-verified: issue-time sampling produced the unshifted input for
    /// upshift modes 1-7).
    #[test]
    fn test_late_shift_mov_controls_earlier_ups_load() {
        use crate::interpreter::bundle::ElementType;
        use crate::interpreter::decode::register_map::AccumWidth;

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        let vals: [i16; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        for i in 0..16 {
            let b = (vals[i] as u16).to_le_bytes();
            tile.data_memory_mut()[0x200 + i * 2] = b[0];
            tile.data_memory_mut()[0x200 + i * 2 + 1] = b[1];
        }
        ctx.pointer.write(0, 0x70200);
        ctx.scalar.write(0, 0); // s0 = 0 at issue; the late MOV sets the shift

        let mut ups = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Ups)
            .with_dest(Operand::AccumReg(0))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::PointerReg(0));
        ups.from_type = Some(ElementType::Int16);
        ups.element_type = Some(ElementType::Int32);
        ups.accum_width = Some(AccumWidth::Half);
        ups.mem_width = MemWidth::Vector256;
        ups.is_vector = true;

        // Bundle 0: the UPS load. Bundles 1-4: nops. Bundle 5: MOV s0,#1
        // (commits before the issue+6 sample). Bundle 6: sample lands.
        executor.execute(&make_bundle(vec![ups]), &mut ctx, &mut tile);
        for _ in 0..4 {
            executor.execute(&make_bundle(vec![]), &mut ctx, &mut tile);
        }
        let mov = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::Immediate(1));
        executor.execute(&make_bundle(vec![mov]), &mut ctx, &mut tile);
        executor.execute(&make_bundle(vec![]), &mut ctx, &mut tile);

        let acc = ctx.accumulator.read(0);
        for j in 0..8usize {
            let lo = (acc[j] & 0xFFFF_FFFF) as i32;
            let hi = (acc[j] >> 32) as i32;
            assert_eq!(lo, (vals[2 * j] as i32) << 1, "lane {}", 2 * j);
            assert_eq!(hi, (vals[2 * j + 1] as i32) << 1, "lane {}", 2 * j + 1);
        }
    }

    #[test]
    fn test_empty_bundle() {
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        let bundle = make_bundle(vec![]);
        let result = executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(matches!(result, ExecuteResult::Continue));
    }

    #[test]
    fn test_scalar_add_timing() {
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        ctx.scalar.write(0, 10);
        ctx.scalar.write(1, 20);

        // Scalar add has 1 cycle latency
        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1))]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        assert_eq!(ctx.scalar.read(2), 30);
        assert_eq!(ctx.cycles, 1); // 1 cycle for scalar add
    }

    #[test]
    fn test_scalar_mul_timing() {
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        ctx.scalar.write(0, 5);
        ctx.scalar.write(1, 6);

        // Scalar mul has 2 cycle latency
        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::Scalar1, SemanticOp::Mul)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1))]);

        executor.execute(&bundle, &mut ctx, &mut tile);
        assert_eq!(ctx.cycles, 1); // 1 cycle to issue (pipelined)

        // MUL result has latency 2 -- not yet visible after 1 cycle.
        // Advance one more cycle and commit to see the result.
        ctx.record_instruction(1);
        ctx.commit_pending_writes();
        assert_eq!(ctx.scalar.read(2), 30);
    }

    #[test]
    fn test_lock_acquire_charges_arbitration_cycle() {
        // AIE2 charges one memory-arbitration cycle per lock transaction even
        // when uncontended: the lock sits behind the tile memory module's
        // round-robin arbiter, which grants one request per cycle (AM020 ch.2).
        // HW trails every LOCK_ACQUIRE_REQ with a 1-cycle LOCK_STALL pulse. So a
        // successful acquire costs 2 cycles: 1 issue + 1 arbitration.
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);
        tile.locks[5].value = 1; // available -> uncontended success

        let bundle =
            make_bundle(vec![SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockAcquire)
                .with_source(Operand::Lock(5))]);
        let result = executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(matches!(result, ExecuteResult::Continue));
        assert_eq!(tile.locks[5].value, 0, "lock acquired");
        assert_eq!(ctx.cycles, 2, "1 issue + 1 arbitration cycle");
    }

    #[test]
    fn test_lock_release_charges_arbitration_cycle() {
        // Release is also an arbitrated transaction: 1 issue + 1 arbitration.
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);
        tile.locks[3].value = 0;

        let bundle =
            make_bundle(vec![SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockRelease)
                .with_source(Operand::Lock(3))]);
        executor.execute(&bundle, &mut ctx, &mut tile);

        assert_eq!(ctx.cycles, 2, "1 issue + 1 arbitration cycle");
    }

    #[test]
    fn test_blocked_lock_acquire_no_arbitration_charge() {
        // A blocked acquire (lock unavailable) returns WaitLock and does NOT
        // pay the arbitration cycle -- the core is stalling, not completing a
        // transaction. The arbitration is charged once, later, when the
        // acquire resumes and succeeds.
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);
        tile.locks[5].value = 0; // unavailable -> blocks

        let bundle =
            make_bundle(vec![SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockAcquire)
                .with_source(Operand::Lock(5))]);
        let result = executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(matches!(result, ExecuteResult::WaitLock { .. }));
        assert_eq!(ctx.cycles, 1, "issue only -- no arbitration while stalling");
    }

    #[test]
    fn test_memory_load_timing() {
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        tile.write_data_u32(0x100, 42);
        ctx.pointer.write(0, 0x100);

        // Memory load has 5 cycle latency
        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::Word)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::PointerReg(0))]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        // Load results are deferred -- commit them now that the latency has elapsed.
        // In normal execution, this happens at the start of the next bundle.
        ctx.flush_pending_writes();
        assert_eq!(ctx.scalar.read(0), 42);
        assert_eq!(ctx.cycles, 1); // 1 cycle to issue (pipelined); result deferred by latency
    }

    #[test]
    fn stream_stall_entry_records_rising_level_not_pulse() {
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.timing_context_mut().enable_tracing();
        let mut tile = Tile::compute(0, 2);

        // Blocking stream read with an empty FIFO stalls on entry.
        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::StreamRead)
            .with_blocking(true)
            .with_dest(Operand::ScalarReg(5))]);
        let result = executor.execute(&bundle, &mut ctx, &mut tile);
        assert!(matches!(result, ExecuteResult::WaitStream { .. }));

        let events: Vec<_> = ctx.timing_context().events.events().iter().map(|e| e.event).collect();
        assert!(
            events.iter().any(|e| matches!(e, EventType::StreamStallLevel { active: true })),
            "stream-stall entry must record a rising STREAM_STALL held level; got {events:?}"
        );
        // The held-level path replaces the legacy one-cycle pulse variant.
        assert!(
            !events.iter().any(|e| matches!(e, EventType::StreamStall { .. })),
            "stream stall must no longer emit the legacy StreamStall pulse"
        );
    }

    #[test]
    fn cascade_stall_entry_records_rising_cascade_level_not_stream() {
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.timing_context_mut().enable_tracing();
        let mut tile = Tile::compute(0, 2);

        // Cascade read with an empty SCD stalls on entry (sentinel port 254).
        let bundle =
            make_bundle(vec![SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::CascadeRead)
                .with_dest(Operand::VectorReg(0))]);
        let result = executor.execute(&bundle, &mut ctx, &mut tile);
        assert!(matches!(result, ExecuteResult::WaitStream { port: 254 }));

        let events: Vec<_> = ctx.timing_context().events.events().iter().map(|e| e.event).collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventType::CascadeStallLevel { active: true })),
            "cascade-read stall entry must record a rising CASCADE_STALL held level; got {events:?}"
        );
        // CASCADE_STALL (25) is distinct: a cascade stall must never be
        // recorded as STREAM_STALL (24), pulse or level.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EventType::StreamStall { .. } | EventType::StreamStallLevel { .. })),
            "cascade stall must not be mislabeled as STREAM_STALL; got {events:?}"
        );
    }

    #[test]
    fn test_reset() {
        let mut executor = CycleAccurateExecutor::new();
        executor.total_hazard_stalls = 10;
        executor.total_branch_stalls = 3;

        executor.reset();

        assert_eq!(executor.total_hazard_stalls, 0);
        assert_eq!(executor.total_branch_stalls, 0);
    }

    #[test]
    fn test_stats() {
        let executor = CycleAccurateExecutor::new();
        let stats = executor.stats();

        assert_eq!(stats.hazard_stalls, 0);
        assert_eq!(stats.branch_stalls, 0);
    }

    #[test]
    fn test_branch_no_explicit_penalty() {
        use crate::interpreter::bundle::BranchCondition;

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::Control, SemanticOp::Br)
            .with_branch_condition(BranchCondition::Always)
            .with_source(Operand::Immediate(0x100))]);

        let result = executor.execute(&bundle, &mut ctx, &mut tile);

        // Should return Branch result
        assert!(matches!(result, ExecuteResult::Branch { target: 0x100 }));

        // No explicit branch penalty -- delay slots handle the pipeline cost.
        // Branch instruction itself costs 1 cycle (pipelined issue).
        assert_eq!(executor.total_branch_stalls, 0);
        assert_eq!(ctx.stall_cycles, 0);
        assert_eq!(ctx.cycles, 1); // Just the 1-cycle issue cost
    }

    #[test]
    fn test_detailed_stats_no_branch_penalty() {
        use crate::interpreter::bundle::BranchCondition;

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::Control, SemanticOp::Br)
            .with_branch_condition(BranchCondition::Always)
            .with_source(Operand::Immediate(0x200))]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        let stats = executor.stats();

        // No explicit penalty -- delay slots are the penalty mechanism
        assert_eq!(stats.branch_stalls, 0);
        assert_eq!(stats.hazard_stats.branch_stall_cycles, 0);
        assert_eq!(stats.hazard_stats.total_stall_cycles, 0);
        assert_eq!(stats.hazard_stats.register_stall_cycles, 0);
        assert_eq!(stats.hazard_stats.memory_stall_cycles, 0);
    }

    // --- Event Recording Tests ---

    #[test]
    fn test_event_recording_instruction_cycle() {
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // Execute a simple scalar add
        ctx.scalar.write(0, 10);
        ctx.scalar.write(1, 20);

        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1))]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        // Check events were recorded.
        // Scalar-only bundles don't emit per-class events (scalars have no
        // dedicated hardware trace event), but the event log should still
        // have been populated if any timing sync occurred.
        let timing = ctx.timing_context();
        let events = timing.events.events();

        // Scalar-only: no per-slot instruction events are emitted (scalars
        // don't map to a hardware trace event code). The log may be empty
        // or have timing sync events only.
        // Just verify no crash and timing context is populated.
        assert!(ctx.cycles > 0, "Cycles should advance after execution");
        let _ = events; // events may or may not be present for scalar-only
    }

    #[test]
    fn test_event_recording_branch() {
        use crate::interpreter::bundle::BranchCondition;
        use crate::interpreter::state::EventType;

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // Execute a branch instruction
        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::Control, SemanticOp::Br)
            .with_branch_condition(BranchCondition::Always)
            .with_source(Operand::Immediate(0x200))]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        // Check for branch events
        let timing = ctx.timing_context();
        let events = timing.events.events();

        // Should have BranchTaken event (hardware-aligned)
        let has_branch_taken = events
            .iter()
            .any(|e| matches!(e.event, EventType::BranchTaken { from_pc: 0, to_pc: 0x200 }));

        assert!(has_branch_taken, "Should have BranchTaken event");
    }

    #[test]
    fn test_all_execution_has_timing() {
        // All contexts now have timing enabled
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1))]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        // Should always have timing context
        assert!(ctx.has_timing());
        // Cycles should have advanced (execution happened)
        assert!(ctx.cycles > 0, "Execution should advance cycles");
    }

    // --- Deferred Pointer Write Tests ---

    #[test]
    fn test_pointer_write_deferred_by_one_cycle() {
        // A MOV-to-pointer (via SemanticOp::Copy) should defer the write by 1 cycle.
        // Within the same bundle, the old pointer value should be visible.
        // After commit_pending_writes at the next cycle, the new value is visible.
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        ctx.pointer.write(1, 0xDEAD); // Old p1 value

        // movxm p1, #0x70400
        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::Scalar1, SemanticOp::Copy)
            .with_dest(Operand::PointerReg(1))
            .with_source(Operand::Immediate(0x70400))]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        // After execution, p1 should still be the OLD value (write is deferred).
        assert_eq!(ctx.pointer.read(1), 0xDEAD, "Pointer write should be deferred, not visible immediately");

        // Execute another bundle (any NOP will do). This calls commit_pending_writes.
        let nop = make_bundle(vec![]);
        executor.execute(&nop, &mut ctx, &mut tile);

        // Now p1 should have the new value (committed at start of cycle 2).
        assert_eq!(ctx.pointer.read(1), 0x70400, "Pointer write should be committed after one cycle");
    }

    #[test]
    fn test_delay_pending_writes_at_branch_boundary() {
        // When delay_pending_writes(1) is called at a branch boundary,
        // pointer writes from the last delay slot need an extra cycle.
        // This test verifies the mechanism directly on ExecutionContext.
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(1, 0xDEAD);

        // Simulate movxm p1, #0x70400 at cycle 5 (last delay slot)
        ctx.cycles = 5;
        ctx.queue_pointer_write(1, 0x70400, 1); // ready_cycle = 6

        // Branch taken: delay pending writes by 1 extra cycle
        ctx.delay_pending_writes(1); // ready_cycle becomes 7

        // At cycle 6 (branch target first instruction): commit should NOT apply
        ctx.cycles = 6;
        ctx.commit_pending_writes();
        assert_eq!(
            ctx.pointer.read(1),
            0xDEAD,
            "Pointer write should NOT be visible at branch target (cycle 6)"
        );

        // At cycle 7 (branch target second instruction): commit SHOULD apply
        ctx.cycles = 7;
        ctx.commit_pending_writes();
        assert_eq!(
            ctx.pointer.read(1),
            0x70400,
            "Pointer write should be visible one cycle after branch target"
        );
    }

    #[test]
    fn test_pointer_write_sequential_latency_one() {
        // In sequential (non-branch) code, pointer writes should have latency 1:
        // movxm p1, #addr at cycle C → p1 available at cycle C+1.
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(1, 0);

        ctx.cycles = 10;
        ctx.queue_pointer_write(1, 0x1234, 1); // ready_cycle = 11

        // At cycle 11: commit should apply (no branch delay)
        ctx.cycles = 11;
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(1), 0x1234);
    }

    // --- Deferred accumulator (VMOV bml,x) write tests ---

    /// `VMOV bml0, x` (a wide vector->accumulator cross-file move) must defer
    /// its accumulator write by the AIE2 def latency (2 cycles), not apply it
    /// immediately. The accumulator def operand cycle for II_VMOV_X_BM_XM is 2
    /// (NoBypass) in llvm-aie AIE2Schedule.td, so a fused VST.CONV/VST.SRS that
    /// drains the accumulator in a software-pipelined loop reads the value as it
    /// was two cycles earlier -- one VMOV behind the live register.
    ///
    /// Regression: surfaced by the vec_conv_bf16_edge silicon kernels, where the
    /// f32->bf16 store-converts are packed into a RET's delay slots. With an
    /// immediate (0-cycle) accumulator write, each delay-slot store read the
    /// freshly-overwritten accumulator and produced Inf/NaN-range garbage shifted
    /// one group forward; silicon (and a 2-cycle deferral) read the correct value.
    #[test]
    fn test_vmov_to_accum_deferred_by_def_latency() {
        use crate::interpreter::bundle::ElementType;

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // x0 (wide 512-bit pair x0/x1) holds a recognizable pattern: 16 int32
        // lanes 0x100..0x10F. Accumulator lane j packs lanes 2j (low) | 2j+1 (high).
        let mut wide = [0u32; 16];
        for (i, w) in wide.iter_mut().enumerate() {
            *w = 0x100 + i as u32;
        }
        ctx.vector.write_wide(0, wide);

        // Pre-seed bml0 with a sentinel so we can prove the new value is NOT
        // visible until the latency elapses.
        ctx.accumulator.write(0, [0xDEAD_BEEF_DEAD_BEEF; 8]);

        // vmov bml0, x0  (wide cross-file move into the accumulator)
        let mut vmov = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Copy)
            .with_dest(Operand::AccumReg(0))
            .with_source(Operand::VectorReg(0));
        vmov.is_wide_vector = true;
        vmov.is_vector = true;
        vmov.element_type = Some(ElementType::Int32);

        executor.execute(&make_bundle(vec![vmov]), &mut ctx, &mut tile);

        // Immediately after the issuing bundle, bml0 must still read the sentinel:
        // the def latency (2) has not elapsed, so the write is in flight.
        let acc_now = ctx.accumulator.read(0);
        assert_eq!(
            acc_now[0], 0xDEAD_BEEF_DEAD_BEEF,
            "VMOV bml,x write must be deferred, not visible in the issuing bundle"
        );

        // Run filler bundles until the deferred write commits. With latency 2 the
        // value lands within a couple of cycles; allow a small margin.
        let mut committed = false;
        for _ in 0..4 {
            executor.execute(&make_bundle(vec![]), &mut ctx, &mut tile);
            if ctx.accumulator.read(0)[0] != 0xDEAD_BEEF_DEAD_BEEF {
                committed = true;
                break;
            }
        }
        assert!(committed, "deferred VMOV->accum write never committed");

        // Once committed, the accumulator holds the moved x0 data.
        let acc = ctx.accumulator.read(0);
        for j in 0..8usize {
            let lo = (acc[j] & 0xFFFF_FFFF) as u32;
            let hi = (acc[j] >> 32) as u32;
            assert_eq!(lo, 0x100 + (2 * j) as u32, "acc lane {} low", 2 * j);
            assert_eq!(hi, 0x100 + (2 * j + 1) as u32, "acc lane {} high", 2 * j + 1);
        }
    }

    // --- NOP / slot-based event classification tests ---
    //
    // AIE2 hardware fires INSTR_LOAD / INSTR_STORE / INSTR_VECTOR only when
    // the corresponding functional unit does work.  NOPA / NOPB / NOPS / NOPV
    // occupy their slot in the encoding but don't activate the unit, so no
    // event is emitted on real silicon.  The emulator must match this.

    #[test]
    fn test_nopv_does_not_emit_instr_vector() {
        use crate::interpreter::state::EventType;

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // NOPV occupies the Vector slot structurally but does no work.
        let bundle = make_bundle(vec![SlotOp::nop(SlotIndex::Vector)]);
        executor.execute(&bundle, &mut ctx, &mut tile);

        let events = ctx.timing_context().events.events();
        let has_instr_vector = events.iter().any(|e| matches!(e.event, EventType::InstrVector { .. }));
        assert!(!has_instr_vector, "NOPV must not emit INSTR_VECTOR; events: {:?}", events);
    }

    #[test]
    fn test_nopa_does_not_emit_instr_load() {
        use crate::interpreter::state::EventType;

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // NOPA in the LoadA slot: no load port activity.
        let bundle = make_bundle(vec![SlotOp::nop(SlotIndex::LoadA)]);
        executor.execute(&bundle, &mut ctx, &mut tile);

        let events = ctx.timing_context().events.events();
        let has_instr_load = events.iter().any(|e| matches!(e.event, EventType::InstrLoad { .. }));
        assert!(!has_instr_load, "NOPA must not emit INSTR_LOAD; events: {:?}", events);
    }

    #[test]
    fn test_nops_does_not_emit_instr_store() {
        use crate::interpreter::state::EventType;

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // NOPS in the Store slot: no memory write.
        let bundle = make_bundle(vec![SlotOp::nop(SlotIndex::Store)]);
        executor.execute(&bundle, &mut ctx, &mut tile);

        let events = ctx.timing_context().events.events();
        let has_instr_store = events.iter().any(|e| matches!(e.event, EventType::InstrStore { .. }));
        assert!(!has_instr_store, "NOPS must not emit INSTR_STORE; events: {:?}", events);
    }

    #[test]
    fn test_real_load_does_emit_instr_load() {
        use crate::interpreter::state::EventType;

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        tile.write_data_u32(0x100, 42);
        ctx.pointer.write(0, 0x100);

        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::Word)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::PointerReg(0))]);
        executor.execute(&bundle, &mut ctx, &mut tile);

        let events = ctx.timing_context().events.events();
        let instr_loads: Vec<_> = events
            .iter()
            .filter(|e| matches!(e.event, EventType::InstrLoad { .. }))
            .collect();
        assert_eq!(
            instr_loads.len(),
            1,
            "Real load should emit exactly one INSTR_LOAD; events: {:?}",
            events
        );
    }

    #[test]
    fn test_full_nop_bundle_emits_nothing() {
        use crate::interpreter::state::EventType;

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // A bundle of all NOPs across every functional-unit slot.  This is
        // what padding rows in scalar-kernel disassembly look like
        // (`NOPA; NOPB; NOPS; NOPX; NOPM; NOPV`).
        let bundle = make_bundle(vec![
            SlotOp::nop(SlotIndex::LoadA),
            SlotOp::nop(SlotIndex::LoadB),
            SlotOp::nop(SlotIndex::Store),
            SlotOp::nop(SlotIndex::Scalar0),
            SlotOp::nop(SlotIndex::Scalar1),
            SlotOp::nop(SlotIndex::Vector),
        ]);
        executor.execute(&bundle, &mut ctx, &mut tile);

        let events = ctx.timing_context().events.events();
        let unit_activity: Vec<_> = events
            .iter()
            .filter(|e| {
                matches!(
                    e.event,
                    EventType::InstrLoad { .. }
                        | EventType::InstrStore { .. }
                        | EventType::InstrVector { .. }
                )
            })
            .collect();
        assert!(
            unit_activity.is_empty(),
            "Full-NOP bundle must not emit any unit-activity events; got: {:?}",
            unit_activity
        );
    }

    // ------------------------------------------------------------------
    // Watchpoint hardware: matching_watchpoint_events
    //
    // The watchpoint register layout is per AM025:
    //   compute mem-module: WatchPoint0/1 at 0x14100/4
    //     bit 31 = Read_Access, 30 = Write_Access
    //     [23:20] = WriteStrobes (must be 0xF for active slot)
    //     [15:4]  = Address (16-byte aligned, 12-bit comparator)
    //   memtile: WatchPoint0..3 at 0x94100..0xC
    //     bit 29 = Read_Access, 28 = Write_Access
    //     [23:20] = WriteStrobes (must be 0xF for active slot)
    //     [18:4]  = Address (16-byte aligned, 15-bit comparator)
    // ------------------------------------------------------------------

    /// Build a compute-tile watchpoint register value. `read`/`write` set the
    /// direction filter bits, `addr` is the watch address (low 4 bits ignored
    /// per the 16-byte alignment in HW). WriteStrobes is fixed at 0xF (the
    /// AM025 "active slot" sentinel).
    fn compute_wp(read: bool, write: bool, addr: u32) -> u32 {
        let dir = ((read as u32) << 31) | ((write as u32) << 30);
        let strobes = 0xFu32 << 20;
        dir | strobes | (addr & 0xFFF0)
    }

    fn memtile_wp(read: bool, write: bool, addr: u32) -> u32 {
        let dir = ((read as u32) << 29) | ((write as u32) << 28);
        let strobes = 0xFu32 << 20;
        dir | strobes | (addr & 0x7FFF0)
    }

    #[test]
    fn test_watchpoint_unprogrammed_fires_nothing() {
        let tile = Tile::compute(0, 2);
        // No watchpoint register written. Every access must miss.
        assert!(matching_watchpoint_events(&tile, 0x0000, false).is_empty());
        assert!(matching_watchpoint_events(&tile, 0xFFF0, true).is_empty());
    }

    #[test]
    fn test_watchpoint_inactive_slot_fires_nothing() {
        let mut tile = Tile::compute(0, 2);
        // WriteStrobes != 0xF: slot is inactive even if direction + addr would
        // otherwise match. Hardware treats this as "not in use".
        let inactive = (1u32 << 31) | (0x7u32 << 20) | 0x100;
        tile.registers.insert(0x14100, inactive);
        assert!(matching_watchpoint_events(&tile, 0x100, false).is_empty());
    }

    #[test]
    fn test_watchpoint_compute_read_match() {
        use xdna_archspec::aie2::trace_events::mem_events;
        let mut tile = Tile::compute(0, 2);
        tile.registers.insert(0x14100, compute_wp(true, false, 0x100));
        let hits = matching_watchpoint_events(&tile, 0x100, false);
        assert_eq!(hits, vec![mem_events::WATCHPOINT_0]);
    }

    #[test]
    fn test_watchpoint_compute_read_addr_aligned_to_16_bytes() {
        use xdna_archspec::aie2::trace_events::mem_events;
        let mut tile = Tile::compute(0, 2);
        // Watchpoint address 0x100 covers the whole 16-byte block 0x100..0x10F
        // because the comparator masks the low 4 bits.
        tile.registers.insert(0x14100, compute_wp(true, false, 0x100));
        for off in 0..16 {
            let hits = matching_watchpoint_events(&tile, 0x100 + off, false);
            assert_eq!(hits, vec![mem_events::WATCHPOINT_0], "offset {off:#x}");
        }
        // 0x110 is the next 16-byte block: must miss.
        assert!(matching_watchpoint_events(&tile, 0x110, false).is_empty());
    }

    #[test]
    fn test_watchpoint_compute_wrong_direction_misses() {
        let mut tile = Tile::compute(0, 2);
        // Read-only watchpoint must NOT fire on a store to the same address.
        tile.registers.insert(0x14100, compute_wp(true, false, 0x100));
        assert!(matching_watchpoint_events(&tile, 0x100, true).is_empty());
        // Write-only watchpoint must NOT fire on a load.
        tile.registers.insert(0x14100, compute_wp(false, true, 0x100));
        assert!(matching_watchpoint_events(&tile, 0x100, false).is_empty());
    }

    #[test]
    fn test_watchpoint_compute_both_directions_match_load_and_store() {
        use xdna_archspec::aie2::trace_events::mem_events;
        let mut tile = Tile::compute(0, 2);
        tile.registers.insert(0x14100, compute_wp(true, true, 0x200));
        assert_eq!(matching_watchpoint_events(&tile, 0x200, false), vec![mem_events::WATCHPOINT_0]);
        assert_eq!(matching_watchpoint_events(&tile, 0x200, true), vec![mem_events::WATCHPOINT_0]);
    }

    #[test]
    fn test_watchpoint_compute_two_slots_independent() {
        use xdna_archspec::aie2::trace_events::mem_events;
        let mut tile = Tile::compute(0, 2);
        tile.registers.insert(0x14100, compute_wp(true, false, 0x100));
        tile.registers.insert(0x14104, compute_wp(false, true, 0x200));
        // Load at 0x100 hits slot 0 only.
        assert_eq!(matching_watchpoint_events(&tile, 0x100, false), vec![mem_events::WATCHPOINT_0]);
        // Store at 0x200 hits slot 1 only.
        assert_eq!(matching_watchpoint_events(&tile, 0x200, true), vec![mem_events::WATCHPOINT_1]);
        // Load at 0x200 (store-only slot) misses.
        assert!(matching_watchpoint_events(&tile, 0x200, false).is_empty());
    }

    #[test]
    fn test_watchpoint_compute_two_slots_overlapping_address() {
        use xdna_archspec::aie2::trace_events::mem_events;
        let mut tile = Tile::compute(0, 2);
        // Both slots cover the same address. A load there fires both events
        // in declaration order (slot 0 then slot 1).
        tile.registers.insert(0x14100, compute_wp(true, false, 0x40));
        tile.registers.insert(0x14104, compute_wp(true, true, 0x40));
        assert_eq!(
            matching_watchpoint_events(&tile, 0x40, false),
            vec![mem_events::WATCHPOINT_0, mem_events::WATCHPOINT_1]
        );
    }

    #[test]
    fn test_watchpoint_memtile_uses_memtile_event_ids() {
        use xdna_archspec::aie2::trace_events::memtile_events;
        let mut tile = Tile::mem_tile(0, 1);
        // Slot 2 at 0x94108. Address 0x1000 is within memtile's 512KB span.
        tile.registers.insert(0x94108, memtile_wp(true, false, 0x1000));
        assert_eq!(matching_watchpoint_events(&tile, 0x1000, false), vec![memtile_events::WATCHPOINT_2]);
    }

    #[test]
    fn test_watchpoint_memtile_address_field_is_15_bits() {
        use xdna_archspec::aie2::trace_events::memtile_events;
        let mut tile = Tile::mem_tile(0, 1);
        // Memtile address field [18:4] covers 19-bit address space (512KB).
        // 0x40000 is past the compute tile's 64KB span but well within memtile.
        tile.registers.insert(0x94100, memtile_wp(true, false, 0x40000));
        assert_eq!(matching_watchpoint_events(&tile, 0x40000, false), vec![memtile_events::WATCHPOINT_0]);
        // 0x40010 is the next 16-byte block: must miss.
        assert!(matching_watchpoint_events(&tile, 0x40010, false).is_empty());
    }

    #[test]
    fn test_watchpoint_shim_has_no_watchpoints() {
        let tile = Tile::shim(0, 0);
        // Shim tiles have no watchpoint registers; the dispatch short-circuits.
        assert!(matching_watchpoint_events(&tile, 0, false).is_empty());
        assert!(matching_watchpoint_events(&tile, 0xFFFF, true).is_empty());
    }

    #[test]
    fn test_watchpoint_fire_runs_through_record_memory_access() {
        // End-to-end: a programmed compute watchpoint must fire when a real
        // scalar load executes through the cycle-accurate executor. We can't
        // easily probe the trace_unit (it requires configuration to record),
        // but we can verify fire_watchpoint_events runs without panicking and
        // that the matching function reports the expected hit at the access
        // address used by the executor.
        let mut tile = Tile::compute(0, 2);
        tile.registers.insert(0x14100, compute_wp(true, false, 0x100));
        tile.write_data_u32(0x100, 0xDEADBEEF);

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(0, 0x100);

        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::Word)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::PointerReg(0))]);
        executor.execute(&bundle, &mut ctx, &mut tile);

        // Spot-check the pure decision agrees that this access matches.
        use xdna_archspec::aie2::trace_events::mem_events;
        assert_eq!(matching_watchpoint_events(&tile, 0x100, false), vec![mem_events::WATCHPOINT_0]);
    }

    #[test]
    fn test_watchpoint_writestrobes_zero_inactive() {
        // Default register state has WriteStrobes==0. Even with direction +
        // address bits perfectly aligned, the slot must NOT fire. This is the
        // most common "register cleared / never programmed" case.
        let mut tile = Tile::compute(0, 2);
        let cleared = (1u32 << 31) | 0x100; // Read=1, addr=0x100, WriteStrobes=0x0
        tile.registers.insert(0x14100, cleared);
        assert!(matching_watchpoint_events(&tile, 0x100, false).is_empty());
    }

    #[test]
    fn test_watchpoint_writestrobes_partial_inactive() {
        // Only 0xF (all four strobes set) marks the slot active. AM025 says
        // "always set this field to 0xF when using this watchpoint" -- we treat
        // any other value as inactive. Sweep a few partial values to lock in
        // that interpretation.
        let mut tile = Tile::compute(0, 2);
        for partial in [0x1u32, 0x3, 0x7, 0xE] {
            let value = (1u32 << 31) | (partial << 20) | 0x100;
            tile.registers.insert(0x14100, value);
            assert!(
                matching_watchpoint_events(&tile, 0x100, false).is_empty(),
                "WriteStrobes 0x{partial:X} must leave slot inactive"
            );
        }
        // Sanity: 0xF alone DOES enable.
        tile.registers.insert(0x14100, (1u32 << 31) | (0xFu32 << 20) | 0x100);
        assert_eq!(
            matching_watchpoint_events(&tile, 0x100, false).len(),
            1,
            "WriteStrobes 0xF must enable slot"
        );
    }

    #[test]
    fn test_watchpoint_compute_slot1_alone() {
        // Slot 1 (offset 0x14104) fires WATCHPOINT_1 even when slot 0 is
        // unprogrammed. Locks in that the index->event_id mapping is by slot
        // position, not by which slot is the first non-empty one.
        use xdna_archspec::aie2::trace_events::mem_events;
        let mut tile = Tile::compute(0, 2);
        tile.registers.insert(0x14104, compute_wp(true, false, 0x80));
        assert_eq!(matching_watchpoint_events(&tile, 0x80, false), vec![mem_events::WATCHPOINT_1]);
    }

    #[test]
    fn test_watchpoint_memtile_all_four_slots_independent() {
        // Memtile has 4 slots producing WATCHPOINT_0..3. Program each at a
        // distinct address; each access fires only its own slot.
        use xdna_archspec::aie2::trace_events::memtile_events;
        let mut tile = Tile::mem_tile(0, 1);
        let slots = [
            (0x94100u32, 0x100u32, memtile_events::WATCHPOINT_0),
            (0x94104, 0x200, memtile_events::WATCHPOINT_1),
            (0x94108, 0x300, memtile_events::WATCHPOINT_2),
            (0x9410C, 0x400, memtile_events::WATCHPOINT_3),
        ];
        for (off, addr, _) in slots {
            tile.registers.insert(off, memtile_wp(true, false, addr));
        }
        for (_, addr, expected_event) in slots {
            assert_eq!(
                matching_watchpoint_events(&tile, addr, false),
                vec![expected_event],
                "addr 0x{addr:X} must fire only event {expected_event}"
            );
        }
    }

    #[test]
    fn test_watchpoint_halt_via_debug_control1_event0() {
        // End-to-end: a watchpoint configured to fire WATCHPOINT_0 (mem
        // event 16), with Debug_Control1.Debug_Halt_Core_Event0 = 16,
        // must halt the core when the load executes.
        use xdna_archspec::aie2::trace_events::mem_events;
        let mut tile = Tile::compute(0, 2);
        // Program watchpoint at 0x100 for reads.
        tile.registers.insert(0x14100, compute_wp(true, false, 0x100));
        tile.write_data_u32(0x100, 0xCAFED00D);
        // Configure Debug_Control1.Debug_Halt_Core_Event0 = WATCHPOINT_0 (16).
        // Field at bits [22:16].
        let halt_e0 = (mem_events::WATCHPOINT_0 as u32) << 16;
        tile.core_debug.write_register(0x32014, halt_e0);
        // Core must be enabled for request_halt to take effect.
        tile.core_debug.write_register(0x32000, 1);

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(0, 0x100);
        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::Word)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::PointerReg(0))]);
        executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(
            tile.core_debug.is_halted(),
            "watchpoint event must trigger debug halt via Debug_Control1.Event0"
        );
        // Debug_Status[5] (Event0_Halted) must be latched.
        let status = tile.core_debug.read_register(0x3201C).unwrap();
        assert_ne!(status & (1 << 5), 0, "Debug_Status[5] Event0 cause latch");
    }

    #[test]
    fn test_watchpoint_vector_load_fires_event() {
        // Vector loads now fire watchpoints just like scalar loads do (HW
        // comparator sits at the bank interface, doesn't care which engine
        // issued the access). To probe the fire without configuring the
        // trace pipeline, we wire a perf counter to start on WATCHPOINT_0
        // (event 16) and check the counter activates after a vector load
        // through the executor.
        use crate::interpreter::bundle::ElementType;
        use xdna_archspec::aie2::trace_events::mem_events;

        let mut tile = Tile::compute(0, 2);
        tile.registers.insert(0x14100, compute_wp(true, false, 0x100));
        tile.write_data_u32(0x100, 0xFEEDFACE);
        // Counter 0: start on WATCHPOINT_0 (=16), stop never (=0).
        // write_control_start_stop packs counter_lo at bits [7:0] start /
        // [15:8] stop; using event_width=7 (AIE2 event field width).
        tile.mem_perf_counters
            .write_control_start_stop(mem_events::WATCHPOINT_0 as u32, 0, 1, 7);
        assert!(!tile.mem_perf_counters.is_active(0), "counter must start idle");

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(0, 0x100);

        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .as_vector(ElementType::Int32)
            .with_mem_width(MemWidth::Word)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::PointerReg(0))]);
        executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(
            tile.mem_perf_counters.is_active(0),
            "vector load must fire WATCHPOINT_0; counter would have stayed idle if the watchpoint check skipped vector ops"
        );
    }

    // ------------------------------------------------------------------
    // Effective-address tracking in record_memory_access (task #66)
    //
    // Previously record_memory_access resolved the recorded address from
    // the first PointerReg or Memory operand it saw, which dropped the
    // modifier-register contribution in indexed addressing (`[pN, mK]`).
    // The fix routes record_memory_access through MemoryUnit::get_address /
    // get_store_address -- the same helpers the actual load/store path
    // uses -- so bank-conflict tracking and watchpoint matching see the
    // address that hits memory, not the bare pointer.
    //
    // Post-modify (`op.post_modify`) is intentionally NOT included: the
    // modifier updates the base register *after* the access, so the address
    // the access lands on this cycle is the pre-modify base.
    // ------------------------------------------------------------------

    #[test]
    fn test_record_memory_access_uses_modifier_register_indexed_address() {
        // `lda r0, [p0, m0]` with p0=0x100 and m0=0x40 reads address 0x140.
        // A watchpoint armed at the effective address 0x140 must fire.
        // Under the prior approximation it would have stayed silent because
        // record_memory_access only saw p0 and recorded 0x100.
        use xdna_archspec::aie2::trace_events::mem_events;

        let mut tile = Tile::compute(0, 2);
        tile.write_data_u32(0x140, 0xFEEDFACE);
        tile.registers.insert(0x14100, compute_wp(true, false, 0x140));
        tile.mem_perf_counters
            .write_control_start_stop(mem_events::WATCHPOINT_0 as u32, 0, 1, 7);
        assert!(!tile.mem_perf_counters.is_active(0), "counter must start idle");

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(0, 0x100);
        ctx.modifier.write(0, 0x40);

        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::Word)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::PointerReg(0))
            .with_source(Operand::ModifierReg(0))]);
        executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(
            tile.mem_perf_counters.is_active(0),
            "indexed load must fire watchpoint at base+modifier (0x140), not base alone (0x100)"
        );
    }

    #[test]
    fn test_record_memory_access_modifier_register_does_not_fire_at_base_addr() {
        // Same setup as the previous test, but the watchpoint sits at the
        // BASE pointer (0x100) instead of the effective address (0x140).
        // The pre-fix approximation would have wrongly fired here. After the
        // fix, only the effective address sees a hit and the base stays cold.
        use xdna_archspec::aie2::trace_events::mem_events;

        let mut tile = Tile::compute(0, 2);
        tile.write_data_u32(0x140, 0xFEEDFACE);
        tile.registers.insert(0x14100, compute_wp(true, false, 0x100));
        tile.mem_perf_counters
            .write_control_start_stop(mem_events::WATCHPOINT_0 as u32, 0, 1, 7);

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(0, 0x100);
        ctx.modifier.write(0, 0x40);

        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::Word)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::PointerReg(0))
            .with_source(Operand::ModifierReg(0))]);
        executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(
            !tile.mem_perf_counters.is_active(0),
            "watchpoint at the base pointer must not fire when the access lands at base+modifier"
        );
    }

    #[test]
    fn test_record_memory_access_post_modify_does_not_shift_recorded_address() {
        // `lda r0, [p0], #16` reads from p0 *first*, THEN updates p0 += 16.
        // So the address that hits memory this cycle is the pre-modify base
        // (0x100), and the watchpoint there must fire. A watchpoint at the
        // post-modify destination (0x110) must NOT fire -- that address won't
        // see traffic until the next access.
        use crate::interpreter::bundle::PostModify;
        use xdna_archspec::aie2::trace_events::mem_events;

        let mut tile = Tile::compute(0, 2);
        tile.write_data_u32(0x100, 0xCAFED00D);
        // Slot 0 watches 0x100; slot 1 watches the would-be post-modify destination.
        tile.registers.insert(0x14100, compute_wp(true, false, 0x100));
        tile.registers.insert(0x14104, compute_wp(true, false, 0x110));
        tile.mem_perf_counters
            .write_control_start_stop(mem_events::WATCHPOINT_0 as u32, 0, 1, 7);
        tile.mem_perf_counters
            .write_control_start_stop(mem_events::WATCHPOINT_1 as u32, 0, 1, 7);

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(0, 0x100);

        let mut load = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::Word)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::PointerReg(0));
        load.post_modify = PostModify::Immediate(16);
        let bundle = make_bundle(vec![load]);
        executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(
            tile.mem_perf_counters.is_active(0),
            "watchpoint at the pre-modify address must fire (the access lands there)"
        );
        assert!(
            !tile.mem_perf_counters.is_active(1),
            "watchpoint at the post-modify destination must NOT fire -- the modify happens after the access"
        );
        // And as a sanity check the pointer DID advance, so we know post-modify ran.
        assert_eq!(ctx.pointer.read(0), 0x110, "post-modify must advance the pointer for next access");
    }

    #[test]
    fn test_record_memory_access_store_uses_modifier_register_indexed_address() {
        // Same fix path for stores: record_memory_access dispatches to
        // get_store_address, which honors `[pN, djM]` indexed layout. A store
        // with p1=0x200 and dj1=0x20 lands at 0x220; the watchpoint armed
        // there must fire on writes.
        use xdna_archspec::aie2::trace_events::mem_events;

        let mut tile = Tile::compute(0, 2);
        tile.registers.insert(0x14100, compute_wp(false, true, 0x220));
        tile.mem_perf_counters
            .write_control_start_stop(mem_events::WATCHPOINT_0 as u32, 0, 1, 7);

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(0, 0xABCDEF01);
        ctx.pointer.write(1, 0x200);
        ctx.modifier.write(1, 0x20);

        // Store layout matches the test/legacy path in get_store_address:
        // sources[0]=pointer, sources[1]=modifier (byte offset).
        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Store)
            .with_mem_width(MemWidth::Word)
            .with_source(Operand::PointerReg(1))
            .with_source(Operand::ModifierReg(1))]);
        executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(
            tile.mem_perf_counters.is_active(0),
            "indexed store must fire watchpoint at base+modifier (0x220), not base alone (0x200)"
        );
    }

    // ------------------------------------------------------------------
    // Origin filter semantics (AXI / DMA / quadrant bits)
    // ------------------------------------------------------------------

    /// Build a compute-tile watchpoint register value with explicit
    /// origin-filter bits. `axi`/`dma`/`east`/`north`/`west`/`south` set
    /// bits 29/28/27/26/25/24 respectively.
    fn compute_wp_with_filters(
        read: bool,
        write: bool,
        addr: u32,
        axi: bool,
        dma: bool,
        east: bool,
        north: bool,
        west: bool,
        south: bool,
    ) -> u32 {
        let mut v = compute_wp(read, write, addr);
        if axi {
            v |= 1 << 29;
        }
        if dma {
            v |= 1 << 28;
        }
        if east {
            v |= 1 << 27;
        }
        if north {
            v |= 1 << 26;
        }
        if west {
            v |= 1 << 25;
        }
        if south {
            v |= 1 << 24;
        }
        v
    }

    #[test]
    fn test_watchpoint_origin_filter_zero_is_wildcard_for_core() {
        // No origin bits set -> any origin (including Core) matches.
        // Locks in backwards compatibility: every existing test relies on
        // this implicit wildcard.
        use xdna_archspec::aie2::trace_events::mem_events;
        let mut tile = Tile::compute(0, 2);
        tile.registers.insert(0x14100, compute_wp(true, false, 0x100));
        let hits = matching_watchpoint_events_with_origin(&tile, 0x100, false, AccessOrigin::Core);
        assert_eq!(hits, vec![mem_events::WATCHPOINT_0]);
    }

    #[test]
    fn test_watchpoint_origin_filter_dma_only_blocks_core() {
        // DMA_Access bit set, no other origin bits. Core access must NOT
        // fire (filter is non-zero and Core has no enable bit). DMA
        // access at the same address WOULD fire, but no caller passes
        // AccessOrigin::Dma yet (task #68).
        let mut tile = Tile::compute(0, 2);
        tile.registers.insert(
            0x14100,
            compute_wp_with_filters(true, false, 0x100, false, true, false, false, false, false),
        );
        let hits_core = matching_watchpoint_events_with_origin(&tile, 0x100, false, AccessOrigin::Core);
        assert!(hits_core.is_empty(), "Core access must not fire when only DMA_Access is enabled");
        let hits_dma = matching_watchpoint_events_with_origin(&tile, 0x100, false, AccessOrigin::Dma);
        assert_eq!(hits_dma.len(), 1, "DMA access must fire when DMA_Access is enabled");
    }

    #[test]
    fn test_watchpoint_origin_filter_axi_matches_axi_only() {
        let mut tile = Tile::compute(0, 2);
        tile.registers.insert(
            0x14100,
            compute_wp_with_filters(true, false, 0x100, true, false, false, false, false, false),
        );
        assert!(matching_watchpoint_events_with_origin(&tile, 0x100, false, AccessOrigin::Core).is_empty());
        assert!(matching_watchpoint_events_with_origin(&tile, 0x100, false, AccessOrigin::Dma).is_empty());
        assert_eq!(matching_watchpoint_events_with_origin(&tile, 0x100, false, AccessOrigin::Axi).len(), 1);
    }

    #[test]
    fn test_watchpoint_origin_filter_quadrant_bits_compute() {
        // Each quadrant bit independently enables that quadrant. Sweep
        // all four to confirm the bit-to-quadrant mapping.
        let mut tile = Tile::compute(0, 2);
        let cases = [
            (true, false, false, false, MemoryQuadrant::East),
            (false, true, false, false, MemoryQuadrant::North),
            (false, false, true, false, MemoryQuadrant::West),
            (false, false, false, true, MemoryQuadrant::South),
        ];
        for (e, n, w, s, expected_dir) in cases {
            tile.registers
                .insert(0x14100, compute_wp_with_filters(true, false, 0x100, false, false, e, n, w, s));
            // The matching quadrant fires.
            assert_eq!(
                matching_watchpoint_events_with_origin(
                    &tile,
                    0x100,
                    false,
                    AccessOrigin::Neighbour(expected_dir)
                )
                .len(),
                1,
                "quadrant {expected_dir:?} must fire when its bit is set"
            );
            // Core does not fire when filter is non-zero.
            assert!(
                matching_watchpoint_events_with_origin(&tile, 0x100, false, AccessOrigin::Core).is_empty()
            );
            // Other quadrants don't match.
            for other in
                [MemoryQuadrant::East, MemoryQuadrant::North, MemoryQuadrant::West, MemoryQuadrant::South]
            {
                if other == expected_dir {
                    continue;
                }
                assert!(
                    matching_watchpoint_events_with_origin(
                        &tile,
                        0x100,
                        false,
                        AccessOrigin::Neighbour(other)
                    )
                    .is_empty(),
                    "quadrant {other:?} must not fire when only {expected_dir:?} is enabled"
                );
            }
        }
    }

    #[test]
    fn test_watchpoint_origin_filter_multiple_bits_or_semantics() {
        // AXI_Access | DMA_Access set -> both AXI and DMA accesses fire.
        // (Direction match still required.) Confirms enable bits are OR'd
        // within the origin filter category.
        let mut tile = Tile::compute(0, 2);
        tile.registers.insert(
            0x14100,
            compute_wp_with_filters(true, false, 0x100, true, true, false, false, false, false),
        );
        assert_eq!(matching_watchpoint_events_with_origin(&tile, 0x100, false, AccessOrigin::Axi).len(), 1);
        assert_eq!(matching_watchpoint_events_with_origin(&tile, 0x100, false, AccessOrigin::Dma).len(), 1);
        assert!(matching_watchpoint_events_with_origin(&tile, 0x100, false, AccessOrigin::Core).is_empty());
    }

    #[test]
    fn test_watchpoint_origin_filter_memtile_has_no_north_south_quadrant() {
        // Memtile WatchPoint registers only have East/West neighbour
        // quadrant bits ([25:24]) -- no North/South. AccessOrigin::
        // Neighbour(North) on memtile must never match, regardless of
        // register programming.
        let mut tile = Tile::mem_tile(0, 1);
        // Set every memtile origin filter bit (axi+dma+east+west).
        let mut value = memtile_wp(true, false, 0x100);
        value |= 0xF << 24;
        tile.registers.insert(0x94100, value);
        // East/West/AXI/DMA fire because their bits are set.
        for origin in [
            AccessOrigin::Axi,
            AccessOrigin::Dma,
            AccessOrigin::Neighbour(MemoryQuadrant::East),
            AccessOrigin::Neighbour(MemoryQuadrant::West),
        ] {
            assert_eq!(
                matching_watchpoint_events_with_origin(&tile, 0x100, false, origin).len(),
                1,
                "{origin:?} must fire on memtile"
            );
        }
        // North/South have no enable bit on memtile -> never match when
        // the origin filter is non-zero.
        for origin in
            [AccessOrigin::Neighbour(MemoryQuadrant::North), AccessOrigin::Neighbour(MemoryQuadrant::South)]
        {
            assert!(
                matching_watchpoint_events_with_origin(&tile, 0x100, false, origin).is_empty(),
                "{origin:?} must not fire on memtile (no enable bit)"
            );
        }
    }

    #[test]
    fn test_watchpoint_origin_filter_neighbour_local_treated_as_core() {
        // Neighbour(Local) is the "own-tile" quadrant from the executor's
        // perspective; for filter purposes it behaves like Core (no
        // AM025 enable bit). Set DMA_Access only and confirm neither
        // Core nor Neighbour(Local) fires.
        let mut tile = Tile::compute(0, 2);
        tile.registers.insert(
            0x14100,
            compute_wp_with_filters(true, false, 0x100, false, true, false, false, false, false),
        );
        assert!(matching_watchpoint_events_with_origin(&tile, 0x100, false, AccessOrigin::Core).is_empty());
        assert!(matching_watchpoint_events_with_origin(
            &tile,
            0x100,
            false,
            AccessOrigin::Neighbour(MemoryQuadrant::Local)
        )
        .is_empty());
    }

    #[test]
    fn test_watchpoint_memtile_address_comparator_masks_high_bits() {
        // Memtile comparator is 15-bit [18:4]. An access at 0x80010 has bit 19
        // set -- that bit is OUTSIDE the comparator field, so the masked
        // address is 0x10. A slot programmed for 0x10 should match such an
        // aliased access. This is HW-defined (the comparator is intentionally
        // narrower than the access address) and is what HW would do too.
        use xdna_archspec::aie2::trace_events::memtile_events;
        let mut tile = Tile::mem_tile(0, 1);
        tile.registers.insert(0x94100, memtile_wp(true, false, 0x10));
        // Direct match at 0x10.
        assert_eq!(matching_watchpoint_events(&tile, 0x10, false), vec![memtile_events::WATCHPOINT_0]);
        // Aliased access at 0x80010 (bit 19 set, low 19 bits == 0x10): also
        // matches because the comparator only sees [18:4].
        assert_eq!(matching_watchpoint_events(&tile, 0x80010, false), vec![memtile_events::WATCHPOINT_0]);
    }

    // ------------------------------------------------------------------
    // peek_bank_demand -- non-committing bank-demand query (Task 3)
    // ------------------------------------------------------------------

    /// Builds (executor, bundle, ctx) for a single scalar load from local
    /// address `addr` via pointer register 0. Mirrors the
    /// `test_memory_load_timing` fixture, minus the tile (peek doesn't touch
    /// tile memory, only register state).
    fn fixture_scalar_load_at(addr: u32) -> (CycleAccurateExecutor, VliwBundle, ExecutionContext) {
        let executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(0, addr);

        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::Word)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::PointerReg(0))]);

        (executor, bundle, ctx)
    }

    #[test]
    fn peek_bank_demand_does_not_mutate_context_and_reports_banks() {
        let (exec, bundle, ctx) = fixture_scalar_load_at(0x0410);
        let before_pc = ctx.pc();
        let before_cycles = ctx.cycles;

        let demand = exec.peek_bank_demand(&bundle, &ctx, crate::device::banking::BankLayout::Compute, &[]);

        assert_eq!(
            demand,
            vec![(
                crate::device::bank_arbiter::Requester::Core(crate::device::bank_arbiter::CorePort::LoadA),
                1 << 1
            )],
            "0x410 is physical bank 1, requested by the LoadA port"
        );
        assert_eq!(ctx.pc(), before_pc, "peek must not advance the PC");
        assert_eq!(ctx.cycles, before_cycles, "peek must not advance the clock");
    }

    #[test]
    fn peek_bank_demand_returns_empty_for_bundle_with_no_memory_ops() {
        // A bundle with only scalar ALU work touches no banks and can always issue.
        let executor = CycleAccurateExecutor::new();
        let ctx = ExecutionContext::new();
        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1))]);

        let demand =
            executor.peek_bank_demand(&bundle, &ctx, crate::device::banking::BankLayout::Compute, &[]);
        assert!(demand.is_empty());
    }

    #[test]
    fn peek_bank_demand_covers_wide_vector_access_spanning_two_banks() {
        // A 256-bit (32-byte) vector load at an aligned address spans two
        // consecutive 16-byte words -- two physical banks -- per
        // `banks_for_access`. Bank arbitration is per-port-width-agnostic
        // (every 128-bit word hits its own bank arbiter), so unlike the old
        // scalar-only stall counter in `record_memory_access`, the peek must
        // NOT skip vector ops.
        let executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(0, 0x400);

        let mut load = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::Vector256)
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::PointerReg(0));
        load.is_vector = true;
        let bundle = make_bundle(vec![load]);

        let demand =
            executor.peek_bank_demand(&bundle, &ctx, crate::device::banking::BankLayout::Compute, &[]);
        assert_eq!(
            demand,
            vec![(
                crate::device::bank_arbiter::Requester::Core(crate::device::bank_arbiter::CorePort::LoadA),
                (1 << 0) | (1 << 1)
            )],
            "32-byte access at 0x400 spans banks 0 and 1, requested by LoadA"
        );
    }

    #[test]
    fn peek_bank_demand_excludes_neighbour_quadrant_access() {
        // 0x50100 decodes to the West neighbour quadrant (CardDir 5), not
        // Local. Non-local accesses are out of scope for this task -- the
        // peek must report no demand for them, not the local-offset bank.
        let executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(0, 0x50100);

        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::Word)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::PointerReg(0))]);

        let demand =
            executor.peek_bank_demand(&bundle, &ctx, crate::device::banking::BankLayout::Compute, &[]);
        assert!(demand.is_empty(), "neighbour-quadrant accesses are out of scope, must not appear in demand");
    }

    #[test]
    fn peek_bank_demand_covers_store_ops() {
        let executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(1, 0x0410);
        ctx.scalar.write(0, 0xDEADBEEF);

        let bundle = make_bundle(vec![SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Store)
            .with_mem_width(MemWidth::Word)
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::PointerReg(1))]);

        let demand =
            executor.peek_bank_demand(&bundle, &ctx, crate::device::banking::BankLayout::Compute, &[]);
        assert_eq!(
            demand,
            vec![(
                crate::device::bank_arbiter::Requester::Core(crate::device::bank_arbiter::CorePort::Store),
                1 << 1
            )],
            "store to 0x410 is physical bank 1, requested by the Store port"
        );
    }

    // ------------------------------------------------------------------
    // Core self-collision: the reviewer-identified gap. Before this fix,
    // `peek_bank_demand` OR'd every memory slot into ONE mask and the only
    // requester available was the single unit variant `Requester::Core`, so
    // a same-physical-bank collision between the core's own load and store
    // port was structurally invisible to the arbiter (proven red against
    // the pre-fix code: `arb.arbitrate(&[(Requester::Core, demand)])` never
    // contends because there is only ever one entry). Now `peek_bank_demand`
    // reports one entry per port, so feeding its output straight into the
    // arbiter reproduces the collision faithfully.
    // ------------------------------------------------------------------
    #[test]
    fn core_self_collision_on_shared_physical_bank_is_detected() {
        // LoadA at 0x400, Store at 0x404: both addresses fall in the same
        // 16-byte word (0x400..0x40F) -> physical bank 0 for both.
        let executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(0, 0x400); // load address
        ctx.pointer.write(1, 0x404); // store address -- same physical bank
        ctx.scalar.write(2, 0xDEADBEEF);

        let bundle = make_bundle(vec![
            SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
                .with_mem_width(MemWidth::Word)
                .with_dest(Operand::ScalarReg(0))
                .with_source(Operand::PointerReg(0)),
            SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Store)
                .with_mem_width(MemWidth::Word)
                .with_source(Operand::ScalarReg(2))
                .with_source(Operand::PointerReg(1)),
        ]);

        let demand =
            executor.peek_bank_demand(&bundle, &ctx, crate::device::banking::BankLayout::Compute, &[]);
        assert_eq!(demand.len(), 2, "LoadA and Store are two independent ports, two demand entries");

        let mut arb = crate::device::bank_arbiter::BankArbiter::new();
        let arbitration = arb.arbitrate(&demand);

        assert_ne!(
            arbitration.contended_banks, 0,
            "load+store on the same physical bank must contend with each other"
        );
        assert_eq!(
            arbitration.lost.len(),
            1,
            "exactly one of the two ports loses arbitration on the shared bank"
        );
        assert!(
            arbitration.core_lost(),
            "the core must be reported lost overall (bundle-granularity stall, AM020 ch.4:69)"
        );
    }

    #[test]
    fn core_self_collision_absent_when_ports_land_on_different_banks() {
        // The Peano heuristic's common case: LoadA and Store 16 bytes apart
        // -- the paired interleave puts them on different physical banks
        // (0 and 1), so they must NOT contend even though both are core
        // ports issuing in the same cycle.
        let executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(0, 0x400); // load address -> physical bank 0
        ctx.pointer.write(1, 0x410); // store address -> physical bank 1
        ctx.scalar.write(2, 0xDEADBEEF);

        let bundle = make_bundle(vec![
            SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
                .with_mem_width(MemWidth::Word)
                .with_dest(Operand::ScalarReg(0))
                .with_source(Operand::PointerReg(0)),
            SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Store)
                .with_mem_width(MemWidth::Word)
                .with_source(Operand::ScalarReg(2))
                .with_source(Operand::PointerReg(1)),
        ]);

        let demand =
            executor.peek_bank_demand(&bundle, &ctx, crate::device::banking::BankLayout::Compute, &[]);
        assert_eq!(demand.len(), 2);

        let mut arb = crate::device::bank_arbiter::BankArbiter::new();
        let arbitration = arb.arbitrate(&demand);

        assert_eq!(arbitration.contended_banks, 0, "different physical banks must not contend");
        assert!(arbitration.lost.is_empty());
        assert!(!arbitration.core_lost());
    }

    #[test]
    fn sticky_grants_let_a_self_colliding_bundle_retire_after_one_stall_cycle() {
        // The livelock fix, end to end through `peek_bank_demand`: a bundle
        // whose LoadA and Store hit the SAME single-port physical bank can
        // only have one port granted per cycle. If the retry cycle re-presents
        // BOTH ports the bundle never retires. Hardware retries only the
        // UNSERVED port (the winner's access completed and latched), so the
        // collision costs exactly one stall cycle -- matching the HW capture's
        // roughly 1:1 MEMORY_STALL:CONFLICT ratio, not an unbounded stall.
        let executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(0, 0x400); // load  -> physical bank 0
        ctx.pointer.write(1, 0x404); // store -> physical bank 0 (same 16-byte word)
        ctx.scalar.write(2, 0xDEADBEEF);

        let bundle = make_bundle(vec![
            SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
                .with_mem_width(MemWidth::Word)
                .with_dest(Operand::ScalarReg(0))
                .with_source(Operand::PointerReg(0)),
            SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Store)
                .with_mem_width(MemWidth::Word)
                .with_source(Operand::ScalarReg(2))
                .with_source(Operand::PointerReg(1)),
        ]);

        let layout = crate::device::banking::BankLayout::Compute;
        let mut arb = crate::device::bank_arbiter::BankArbiter::new();
        let mut served: Vec<crate::device::bank_arbiter::CorePort> = Vec::new();
        let mut stall_cycles = 0u32;

        // Drive the coordinator loop the way Task 6 must: re-peek each cycle
        // with the accumulated served set until the bundle can issue.
        for _ in 0..8 {
            let demand = executor.peek_bank_demand(&bundle, &ctx, layout, &served);
            let a = arb.arbitrate(&demand);
            if !a.core_lost() {
                break; // every remaining port got its bank -- the bundle retires
            }
            stall_cycles += 1;
            served.extend(a.granted_core_ports());
        }

        assert_eq!(stall_cycles, 1, "a 2-way core self-collision must cost exactly ONE stall cycle");
        assert_eq!(served.len(), 1, "exactly one port was served during the stall");
        assert!(
            executor.peek_bank_demand(&bundle, &ctx, layout, &served).len() == 1,
            "the served port must not re-request; only the unserved port does"
        );
    }
}
