//! Trace event codes extracted from mlir-aie.
//!
//! The mlir-aie Python bridge script (`tools/mlir-aie-bridge.py trace-events`)
//! emits a JSON map of event ID -> name for each tile module type. build.rs
//! parses this into per-module const tables and name-lookup functions.
//!
//! Exposed items (from generated code):
//!   - `pub mod core_events`     -- `CoreEvent` codes as `pub const` u8 values
//!   - `pub mod mem_events`      -- `MemEvent` codes
//!   - `pub mod memtile_events`  -- `MemTileEvent` codes
//!   - `pub mod shim_events`     -- `ShimTileEvent` codes
//!   - `core_event_name(u8) -> &'static str`
//!   - `mem_event_name(u8) -> &'static str`
//!   - `memtile_event_name(u8) -> &'static str`
//!   - `shim_event_name(u8) -> &'static str`
//!
//! Hand-written items in this file:
//!   - `TraceModule`             -- tile module discriminator for event classification
//!   - `is_level_event(u8, TraceModule) -> bool` -- hw_id-keyed level-event classifier
//!
//! Falls back to stub functions returning "UNKNOWN" when mlir-aie is not
//! available at build time.

include!(concat!(env!("OUT_DIR"), "/trace_event_codes.rs"));

// ============================================================================
// Level-event classification
// ============================================================================

/// Which tile module a hardware event ID belongs to.
///
/// The event namespace is per-module: the same numeric id means different
/// events in the core module vs the memory/memtile module. This discriminator
/// lets `is_level_event` apply the right lookup table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceModule {
    /// Compute tile core module (CoreEvent ids from mlir-aie).
    Core,
    /// Compute tile memory module (MemEvent ids from mlir-aie). Shim tiles are
    /// not classified here.
    Mem,
    /// Mem tile module (MemTileEvent ids from mlir-aie, distinct from MemEvent).
    MemTile,
}

/// Returns `true` if the given hardware event fires every cycle a condition
/// holds (a *level* signal), `false` if it fires once per occurrence (a
/// *pulse/edge* signal).
///
/// Level events must be compared by interval structure, not by occurrence
/// index. This is the hw_id-keyed authority for that classification; it mirrors
/// the name-based `is_level_event` in `src/trace/compare.rs` but is keyed by
/// `(hw_id, TraceModule)` so the trace unit (which works in hw_event_id, not
/// names) can consult it directly without a string round-trip.
///
/// All id constants are referenced from the generated `core_events`,
/// `mem_events`, and `memtile_events` modules so this function derives from
/// the toolchain, not from magic numbers.
///
/// # Scope: deliberately mirrors `compare.rs`, not hardware-complete
///
/// The LEVEL set here deliberately mirrors `src/trace/compare.rs::is_level_event`
/// *exactly*, so that rerouting `compare.rs` to consume this function is
/// behavior-preserving. Classifying *more* events as level than `compare.rs`
/// does today would silently change trace-comparison results and risk breaking
/// comparison tests.
///
/// Hardware-complete coverage is intentionally deferred. The known omissions
/// (all absent from `compare.rs` today) are:
///   - MemTile stream-switch port states `PORT_RUNNING`/`PORT_IDLE`/`PORT_STALLED`
///     (MemTileEvent ids 79-109) -- level on hardware but unclassified here.
///   - Compute memory bank conflicts `CONFLICT_DM_BANK_4..7` (MemEvent ids 81-84).
///   - The MemTile memory bank conflicts (MemTileEvent `CONFLICT_DM_BANK_*`).
///
/// Expanding the set is future work that must touch *both* this file and
/// `compare.rs` together (with comparison-test updates). Do not add those
/// events here in isolation.
pub fn is_level_event(hw_id: u8, module: TraceModule) -> bool {
    match module {
        TraceModule::Core => is_level_core(hw_id),
        TraceModule::Mem => is_level_mem(hw_id),
        TraceModule::MemTile => is_level_memtile(hw_id),
    }
}

/// Level classifier for CoreEvent ids.
///
/// Level events: stall signals (held while the core cannot issue),
/// ACTIVE/DISABLED (held while the tile is in that power state),
/// PORT_RUNNING/IDLE/STALLED (held while a stream switch port is in that state).
fn is_level_core(hw_id: u8) -> bool {
    matches!(
        hw_id,
        // TRUE: the always-on event, permanently held.
        id if id == core_events::TRUE
          // Core stall signals -- held every cycle the hazard blocks issue.
          || id == core_events::MEMORY_STALL   // memory pipeline stall
          || id == core_events::STREAM_STALL   // stream-FIFO backpressure
          || id == core_events::CASCADE_STALL  // cascade-link stall
          || id == core_events::LOCK_STALL     // lock-acquire blocked
          // Tile power state -- held while the state is active.
          || id == core_events::ACTIVE          // core is running
          || id == core_events::DISABLED        // tile clock-gated
          // Stream switch port states -- held while the port is in that state.
          || id == core_events::PORT_RUNNING_0
          || id == core_events::PORT_RUNNING_1
          || id == core_events::PORT_RUNNING_2
          || id == core_events::PORT_RUNNING_3
          || id == core_events::PORT_RUNNING_4
          || id == core_events::PORT_RUNNING_5
          || id == core_events::PORT_RUNNING_6
          || id == core_events::PORT_RUNNING_7
          || id == core_events::PORT_IDLE_0
          || id == core_events::PORT_IDLE_1
          || id == core_events::PORT_IDLE_2
          || id == core_events::PORT_IDLE_3
          || id == core_events::PORT_IDLE_4
          || id == core_events::PORT_IDLE_5
          || id == core_events::PORT_IDLE_6
          || id == core_events::PORT_IDLE_7
          || id == core_events::PORT_STALLED_0
          || id == core_events::PORT_STALLED_1
          || id == core_events::PORT_STALLED_2
          || id == core_events::PORT_STALLED_3
          || id == core_events::PORT_STALLED_4
          || id == core_events::PORT_STALLED_5
          || id == core_events::PORT_STALLED_6
          || id == core_events::PORT_STALLED_7
    )
}

/// Level classifier for MemEvent ids (compute tile memory module).
///
/// Level events: DMA channel stall conditions (held while the channel cannot
/// make progress) and memory bank conflict (held while two accesses contend).
fn is_level_mem(hw_id: u8) -> bool {
    matches!(
        hw_id,
        id if
          // DMA stalled-lock: held while channel waits for a lock to release.
             id == mem_events::DMA_S2MM_0_STALLED_LOCK
          || id == mem_events::DMA_S2MM_1_STALLED_LOCK
          || id == mem_events::DMA_MM2S_0_STALLED_LOCK
          || id == mem_events::DMA_MM2S_1_STALLED_LOCK
          // DMA S2MM stream starvation: held while the input stream has no data.
          || id == mem_events::DMA_S2MM_0_STREAM_STARVATION
          || id == mem_events::DMA_S2MM_1_STREAM_STARVATION
          // DMA MM2S stream backpressure: held while the output stream is full.
          || id == mem_events::DMA_MM2S_0_STREAM_BACKPRESSURE
          || id == mem_events::DMA_MM2S_1_STREAM_BACKPRESSURE
          // DMA S2MM memory backpressure: held while the local memory write port is busy.
          || id == mem_events::DMA_S2MM_0_MEMORY_BACKPRESSURE
          || id == mem_events::DMA_S2MM_1_MEMORY_BACKPRESSURE
          // DMA MM2S memory starvation: held while the local memory read port is busy.
          || id == mem_events::DMA_MM2S_0_MEMORY_STARVATION
          || id == mem_events::DMA_MM2S_1_MEMORY_STARVATION
          // Memory bank conflict: held while two simultaneous accesses contend for a bank.
          || id == mem_events::CONFLICT_DM_BANK_0
          || id == mem_events::CONFLICT_DM_BANK_1
          || id == mem_events::CONFLICT_DM_BANK_2
          || id == mem_events::CONFLICT_DM_BANK_3
    )
}

/// Level classifier for MemTileEvent ids (mem tile module).
///
/// Level events for the MemTile DMA stalled-lock, stream-starvation, and
/// stream-backpressure conditions (SEL0/SEL1 naming). MemTile uses distinct
/// hw_ids from MemEvent -- the same bit patterns mean different events in the
/// two modules. Narrower than `is_level_mem`: MemTile memory bank conflicts and
/// port states are deliberately omitted (see `is_level_event` scope note).
fn is_level_memtile(hw_id: u8) -> bool {
    matches!(
        hw_id,
        id if
          // DMA stalled-lock: held while channel waits for a lock to release.
             id == memtile_events::DMA_S2MM_SEL0_STALLED_LOCK
          || id == memtile_events::DMA_S2MM_SEL1_STALLED_LOCK
          || id == memtile_events::DMA_MM2S_SEL0_STALLED_LOCK
          || id == memtile_events::DMA_MM2S_SEL1_STALLED_LOCK
          // DMA S2MM stream starvation: held while the input stream has no data.
          || id == memtile_events::DMA_S2MM_SEL0_STREAM_STARVATION
          || id == memtile_events::DMA_S2MM_SEL1_STREAM_STARVATION
          // DMA MM2S stream backpressure: held while the output stream is full.
          || id == memtile_events::DMA_MM2S_SEL0_STREAM_BACKPRESSURE
          || id == memtile_events::DMA_MM2S_SEL1_STREAM_BACKPRESSURE
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod level_classifier_tests {
    use super::*;

    #[test]
    fn level_events_classified_level() {
        // Core stall signals
        assert!(is_level_event(core_events::LOCK_STALL, TraceModule::Core)); // 26
        assert!(is_level_event(core_events::MEMORY_STALL, TraceModule::Core)); // 23
        assert!(is_level_event(core_events::STREAM_STALL, TraceModule::Core)); // 24
        assert!(is_level_event(core_events::CASCADE_STALL, TraceModule::Core)); // 25
                                                                                // Core stream-switch port state
        assert!(is_level_event(core_events::PORT_RUNNING_0, TraceModule::Core)); // 75
                                                                                 // High port index: port hw_ids are non-contiguous (stride 4), so cover
                                                                                 // the top of the range as well as port 0.
        assert!(is_level_event(core_events::PORT_RUNNING_7, TraceModule::Core)); // 103
                                                                                 // Mem DMA starvation
        assert!(is_level_event(mem_events::DMA_S2MM_0_STREAM_STARVATION, TraceModule::Mem)); // 35
                                                                                             // Core power state
        assert!(is_level_event(core_events::ACTIVE, TraceModule::Core)); // 28
        assert!(is_level_event(core_events::DISABLED, TraceModule::Core)); // 29
                                                                           // Core port idle/stalled
        assert!(is_level_event(core_events::PORT_IDLE_0, TraceModule::Core)); // 74
        assert!(is_level_event(core_events::PORT_STALLED_0, TraceModule::Core)); // 76
                                                                                 // Mem stalled-lock
        assert!(is_level_event(mem_events::DMA_S2MM_0_STALLED_LOCK, TraceModule::Mem)); // 31
        assert!(is_level_event(mem_events::DMA_MM2S_0_STALLED_LOCK, TraceModule::Mem)); // 33
                                                                                        // Mem memory backpressure/starvation
        assert!(is_level_event(mem_events::DMA_S2MM_0_MEMORY_BACKPRESSURE, TraceModule::Mem)); // 39
        assert!(is_level_event(mem_events::DMA_MM2S_0_MEMORY_STARVATION, TraceModule::Mem)); // 41
                                                                                             // Mem conflict bank
        assert!(is_level_event(mem_events::CONFLICT_DM_BANK_0, TraceModule::Mem)); // 77
        assert!(is_level_event(mem_events::CONFLICT_DM_BANK_3, TraceModule::Mem)); // 80
                                                                                   // MemTile stall events
        assert!(is_level_event(memtile_events::DMA_S2MM_SEL0_STALLED_LOCK, TraceModule::MemTile)); // 33
        assert!(is_level_event(memtile_events::DMA_MM2S_SEL0_STREAM_BACKPRESSURE, TraceModule::MemTile));
        // 39
    }

    #[test]
    fn pulse_events_classified_pulse() {
        // Instruction events are one-shot pulses.
        assert!(!is_level_event(core_events::INSTR_LOCK_ACQUIRE_REQ, TraceModule::Core)); // 44
        assert!(!is_level_event(core_events::INSTR_LOCK_RELEASE_REQ, TraceModule::Core)); // 45
        assert!(!is_level_event(core_events::INSTR_EVENT_0, TraceModule::Core)); // 33
                                                                                 // DMA start-task is a single-cycle pulse at task launch.
        assert!(!is_level_event(mem_events::DMA_S2MM_0_START_TASK, TraceModule::Mem)); // 19
                                                                                       // PORT_TLAST is a one-cycle pulse at packet boundary.
        assert!(!is_level_event(core_events::PORT_TLAST_0, TraceModule::Core)); // 77
                                                                                // Perf counter events are pulses.
        assert!(!is_level_event(core_events::PERF_CNT_0, TraceModule::Core)); // 5
                                                                              // DMA finished events are pulses.
        assert!(!is_level_event(mem_events::DMA_S2MM_0_FINISHED_TASK, TraceModule::Mem)); // 27
                                                                                          // MemTile start-task is a pulse.
        assert!(!is_level_event(memtile_events::DMA_S2MM_SEL0_START_TASK, TraceModule::MemTile));
        // 21
    }

    #[test]
    fn level_events_are_not_pulse_in_wrong_module() {
        // Core LOCK_STALL (26) is level in Core but not in Mem (id 26 in Mem is
        // DMA_MM2S_1_FINISHED_BD which is a pulse). Ensure module discrimination works.
        assert!(!is_level_event(core_events::LOCK_STALL, TraceModule::Mem));
        // Mem DMA_S2MM_0_STREAM_STARVATION (35) is level in Mem but not in Core
        // (id 35 in Core is INSTR_CALL, a pulse).
        assert!(!is_level_event(mem_events::DMA_S2MM_0_STREAM_STARVATION, TraceModule::Core));
    }
}
