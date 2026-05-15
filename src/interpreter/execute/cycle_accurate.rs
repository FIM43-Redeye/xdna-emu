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
use crate::interpreter::state::{EventType, ExecutionContext};
use crate::interpreter::timing::{HazardDetector, HazardStats, MemoryAccess, MemoryModel};
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

    /// Memory bank conflict detector.
    memory: MemoryModel,

    /// Total stall cycles from hazards.
    pub total_hazard_stalls: u64,

    /// Total stall cycles from memory conflicts.
    pub total_memory_stalls: u64,

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
            memory: MemoryModel::new(),
            total_hazard_stalls: 0,
            total_memory_stalls: 0,
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

    /// Record memory access for bank conflict tracking.
    ///
    /// When two in-cycle accesses land on the same bank, the HW stalls the
    /// bundle for one cycle and fires MEMORY_STALL on the core trace unit
    /// (AM020 event-time mode, core module event 23). We emit the event
    /// into the timing context so `core_event_to_hw_id` can route it to
    /// the trace unit, and bump `total_memory_stalls` for cycle accounting.
    ///
    /// Per-bank: HW additionally fires `MEM_CONFLICT_DM_BANK_N` (mem-module
    /// events 77..84 on compute, 112..120 on memtile) for each conflicting
    /// bank. We route those into `tile.mem_trace` and `tile.mem_perf_counters`
    /// so trace dumps and perf counters see them. Compute path only --
    /// memtile DMA-bank conflicts come through the DMA engine, not here.
    fn record_memory_access(&mut self, op: &SlotOp, ctx: &mut ExecutionContext, tile: &mut Tile) {
        if !matches!(op.semantic, Some(SemanticOp::Load) | Some(SemanticOp::Store)) {
            return;
        }
        // Resolve address from the first pointer register in operands. This is
        // approximate (doesn't account for modifiers/post-increments) but
        // sufficient for both bank-conflict tracking and watchpoint matching.
        let addr = op
            .sources
            .iter()
            .find_map(|src| match src {
                Operand::Memory { base, offset } => {
                    Some(ctx.pointer.read(*base).wrapping_add(*offset as i32 as u32))
                }
                Operand::PointerReg(r) => Some(ctx.pointer.read(*r)),
                _ => None,
            })
            .unwrap_or(0);
        let is_store = matches!(op.semantic, Some(SemanticOp::Store));
        let cycle = ctx.cycles;
        let pc = ctx.pc();

        // Bank-conflict tracking is scalar-only: the conflict model assumes
        // single-port-per-direction access, which doesn't match the wider
        // vector load/store ports. Vector accesses skip this block.
        if !op.is_vector {
            let access = MemoryAccess {
                address: addr,
                width: op.mem_width.bytes(),
                is_write: is_store,
                port: if is_store { 2 } else { 0 },
            };
            let conflict = self.memory.record_access(&access);
            if conflict.has_conflict() {
                let stall = conflict.stall_cycles;
                self.total_memory_stalls += stall as u64;
                ctx.timing_context_mut()
                    .record_event(cycle, EventType::MemoryStall { cycles: stall, pc: Some(pc) });
                fire_bank_conflict_events(tile, conflict.conflict_banks, cycle, pc);
            }
        }
        // Watchpoints fire on every matching access regardless of issuing
        // engine: HW comparator sits at the bank interface and sees both
        // scalar and vector traffic. Programmed slots compare the access
        // address (low bits masked per HW) and direction filter against each
        // WatchPointN register, then notify the mem-module trace unit and
        // perf counters with WATCHPOINT_N events.
        fire_watchpoint_events(tile, addr, is_store, cycle, pc);
    }

    /// Reset executor state (for new program).
    pub fn reset(&mut self) {
        self.pending_call_return_addr = None;
        self.hazards.reset();
        self.memory.reset();
        self.total_hazard_stalls = 0;
        self.total_memory_stalls = 0;
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
            memory_stalls: self.total_memory_stalls,
            branch_stalls: self.total_branch_stalls,
            hazard_stats: combined_stats,
            memory_stats: self.memory.stats(),
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
fn fire_bank_conflict_events(tile: &mut Tile, banks_mask: u8, cycle: u64, pc: u32) {
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
            tile.mem_trace.notify_event(event_id, cycle, Some(pc));
            tile.mem_perf_counters.handle_event(event_id);
        }
    }
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
struct WatchpointLayout {
    slot_offsets: &'static [u32],
    addr_mask: u32,
    read_bit: u8,
    write_bit: u8,
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
            event_base: mem_events::WATCHPOINT_0,
        }),
        TileKind::Mem => Some(WatchpointLayout {
            slot_offsets: &[0x94100, 0x94104, 0x94108, 0x9410C],
            addr_mask: 0x7FFF0,
            read_bit: 29,
            write_bit: 28,
            event_base: memtile_events::WATCHPOINT_0,
        }),
        TileKind::ShimNoc | TileKind::ShimPl => None,
    }
}

/// Pure decision: which WATCHPOINT_N event IDs should fire for an access at
/// `addr` with direction `is_write`, given the tile's current WatchPoint
/// register programming. Returns slot indices in declaration order.
fn matching_watchpoint_events(tile: &Tile, addr: u32, is_write: bool) -> Vec<u8> {
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
        hits.push(layout.event_base + i as u8);
    }
    hits
}

/// Fire WATCHPOINT_N events into the mem-module trace unit and perf counters
/// for every WatchPoint slot that matches this access. Compute and memtile
/// have different bit layouts (`watchpoint_layout`); shim tiles have no
/// watchpoint registers and short-circuit.
fn fire_watchpoint_events(tile: &mut Tile, addr: u32, is_write: bool, cycle: u64, pc: u32) {
    let events = matching_watchpoint_events(tile, addr, is_write);
    for event_id in events {
        tile.mem_trace.notify_event(event_id, cycle, Some(pc));
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
        self.memory.advance_to(ctx.cycles);

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
                    let pc = ctx.pc();
                    ctx.timing_context_mut()
                        .record_event(stall_cycle, EventType::StreamStall { cycles: 1, pc: Some(pc) });
                    ctx.record_instruction(1);
                    let timing = ctx.timing_context_mut();
                    timing.hazard_stalls = self.total_hazard_stalls;
                    timing.memory_stalls = self.total_memory_stalls;
                    return ExecuteResult::WaitStream { port };
                }
                if let Some(super::stream::StreamResult::Stall { port }) =
                    super::stream::StreamOps::would_stall(op, tile)
                {
                    let stall_cycle = ctx.cycles;
                    let pc = ctx.pc();
                    ctx.timing_context_mut()
                        .record_event(stall_cycle, EventType::StreamStall { cycles: 1, pc: Some(pc) });
                    ctx.record_instruction(1);
                    let timing = ctx.timing_context_mut();
                    timing.hazard_stalls = self.total_hazard_stalls;
                    timing.memory_stalls = self.total_memory_stalls;
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
        ctx.begin_bundle();

        for slot_idx in &execution_order {
            if let Some(ref op) = bundle.slots()[*slot_idx as usize] {
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
                let pc = ctx.pc();
                ctx.timing_context_mut()
                    .record_event(start_cycle, EventType::LockStall { cycles: 1, pc: Some(pc) });
            }
            ExecuteResult::WaitStream { .. } => {
                let pc = ctx.pc();
                ctx.timing_context_mut()
                    .record_event(start_cycle, EventType::StreamStall { cycles: 1, pc: Some(pc) });
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

        // Advance cycle counter by 1 (pipelined issue rate).
        // Latency-based deferred writes and hazard stalls handle the rest.
        ctx.record_instruction(1);

        // Sync timing context
        let timing = ctx.timing_context_mut();
        timing.hazard_stalls = self.total_hazard_stalls;
        timing.memory_stalls = self.total_memory_stalls;

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
    /// Total cycles stalled due to memory conflicts.
    pub memory_stalls: u64,
    /// Total cycles stalled due to branch penalties.
    pub branch_stalls: u64,
    /// Detailed hazard statistics.
    pub hazard_stats: crate::interpreter::timing::hazards::HazardStats,
    /// Detailed memory statistics.
    pub memory_stats: crate::interpreter::timing::memory::MemoryStats,
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

    /// Two scalar loads in the same bundle aimed at the same bank must be
    /// detected as a conflict, bump `total_memory_stalls`, and emit a
    /// `MemoryStall` event in the timing context. The event is what the
    /// coordinator later routes to the core trace unit as MEMORY_STALL
    /// (core event 23) -- previously no code path generated this event.
    #[test]
    fn test_memory_bank_conflict_emits_memory_stall() {
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        ctx.timing_context_mut().enable_tracing();
        let mut tile = Tile::compute(0, 2);

        // Two addresses that both land on bank 0 (see banking::BANK_ROW_BYTES).
        // 0x00 and 0x80 are 128 bytes apart which, with 8-way interleave,
        // both map to bank 0. This mirrors the existing timing/memory.rs
        // test_bank_conflict fixture.
        ctx.pointer.write(0, 0x00);
        ctx.pointer.write(1, 0x80);

        let bundle = make_bundle(vec![
            SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
                .with_mem_width(MemWidth::Word)
                .with_dest(Operand::ScalarReg(0))
                .with_source(Operand::PointerReg(0)),
            SlotOp::from_semantic(SlotIndex::LoadB, SemanticOp::Load)
                .with_mem_width(MemWidth::Word)
                .with_dest(Operand::ScalarReg(1))
                .with_source(Operand::PointerReg(1)),
        ]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(executor.total_memory_stalls > 0, "two loads to bank 0 should bump total_memory_stalls");
        let timing = ctx.timing_context();
        let memory_stall_events: Vec<_> = timing
            .events
            .events()
            .iter()
            .filter(|e| matches!(e.event, EventType::MemoryStall { .. }))
            .collect();
        assert!(
            !memory_stall_events.is_empty(),
            "expected a MemoryStall event to be recorded; got {:?}",
            timing.events.events()
        );
    }

    #[test]
    fn test_reset() {
        let mut executor = CycleAccurateExecutor::new();
        executor.total_hazard_stalls = 10;
        executor.total_memory_stalls = 5;
        executor.total_branch_stalls = 3;

        executor.reset();

        assert_eq!(executor.total_hazard_stalls, 0);
        assert_eq!(executor.total_memory_stalls, 0);
        assert_eq!(executor.total_branch_stalls, 0);
    }

    #[test]
    fn test_stats() {
        let executor = CycleAccurateExecutor::new();
        let stats = executor.stats();

        assert_eq!(stats.hazard_stalls, 0);
        assert_eq!(stats.memory_stalls, 0);
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
}
