//! Hardware trace unit emulation.
//!
//! Each AIE tile has trace units (Core module trace, Memory module trace)
//! that monitor hardware events and emit compressed binary trace packets.
//! These packets flow through the stream switch to shim DMA and into host
//! DDR, where tools like mlir-aie's `parse.py` decode them.
//!
//! # Register Interface (from AM025)
//!
//! | Register       | Offset   | Fields                                           |
//! |----------------|----------|--------------------------------------------------|
//! | Trace_Control0 | +0x00    | [31:24] stop_event, [23:16] start_event, [1:0] mode |
//! | Trace_Control1 | +0x04    | [14:12] packet_type, [4:0] packet_id             |
//! | Trace_Event0   | +0x10    | [31:24] evt3, [23:16] evt2, [15:8] evt1, [7:0] evt0 |
//! | Trace_Event1   | +0x14    | [31:24] evt7, [23:16] evt6, [15:8] evt5, [7:0] evt4 |
//!
//! # Byte Encoding (from mlir-aie utils/trace/utils.py)
//!
//! Events are compressed into variable-length byte sequences. Per AM020
//! event-time mode tracks up to eight events "on a per-cycle basis" and
//! creates ONE frame per cycle to record the state of the tracked events;
//! when multiple slots fire in the same cycle they share one Multiple frame.
//!
//! | Type      | Pattern                                     | Content                                        |
//! |-----------|---------------------------------------------|------------------------------------------------|
//! | Single0   | `0b0EEETTTT`                                | slot(3), delta(4): 0-15 cycles                 |
//! | Single1   | `0b100EEETT TTTTTTTT`                       | slot(3), delta(10): 0-1023                     |
//! | Single2   | `0b101EEETT TTTTTTTT TTTTTTTT`              | slot(3), delta(18): 0-262143                   |
//! | Multiple0 | `0b1100EEEE EEEETTTT`                       | mask(8), delta(4)                              |
//! | Multiple1 | `0b110100EE EEEEEETT TTTTTTTT`              | mask(8), delta(10)                             |
//! | Multiple2 | `0b110101EE EEEEEETT TTTTTTTT TTTTTTTT`     | mask(8), delta(18)                             |
//! | Start     | `0b11110000 + 7 bytes timer`                | 56-bit timer sync                              |
//! | Pad       | `0xFE`                                      | Fills remainder of packet                      |
//!
//! Discriminators and field placements match mlir-aie's
//! `python/utils/trace/utils.py::convert_to_commands` decoder.
//!
//! # Per-cycle coalescing
//!
//! `notify_event` accumulates fired slots for the current cycle into a
//! bitmask without emitting bytes. Emission happens in `commit_cycle()`
//! (called per cycle by the coordinator) or lazily when a new cycle's
//! event arrives. This matches the HW invariant of "at most one frame per
//! cycle" and avoids the inflation that led to trace undercount when many
//! slots were active (issue #138).
//!
//! # Packet Format
//!
//! Each packet is 8 x 32-bit words (32 bytes):
//! - Word 0: packet header (col, row, packet_type, packet_id)
//! - Words 1-7: compressed byte stream (28 bytes of encoded events)

#[cfg(test)]
mod tests;

use std::collections::VecDeque;

/// Trace operating mode (Trace_Control0 bits [1:0]).
///
/// Per AM025: 00=event-time, 01=event-PC, 10=execution.
/// There is no explicit "off" mode -- the trace unit is idle until the
/// configured start event fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TraceMode {
    /// Event-time mode (00): records event slot + cycle delta.
    /// This is the default mode and what trace-inject.py uses.
    #[default]
    EventTime = 0,
    /// Event-PC mode (01): records event slot + program counter.
    EventPc = 1,
    /// Execution mode (10): records detailed execution trace.
    Execution = 2,
    /// Reserved (11).
    Reserved = 3,
}

impl TraceMode {
    pub(super) fn from_u32(val: u32) -> Self {
        match val & 0x3 {
            0 => TraceMode::EventTime,
            1 => TraceMode::EventPc,
            2 => TraceMode::Execution,
            3 => TraceMode::Reserved,
            _ => unreachable!(),
        }
    }
}

/// Trace unit running state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum TraceState {
    /// Waiting for start event.
    #[default]
    Idle,
    /// Actively recording events.
    Running,
    /// Stop event received, no longer recording.
    Stopped,
}

/// In-flight RLE state for consecutive atoms of the same polarity.
/// Only used in mode-2 (Execution); never set in other modes.
#[derive(Debug, Clone, Copy)]
struct AtomRun {
    /// True for E_atom (executed), false for N_atom (stalled).
    exec: bool,
    /// Number of cycles with this polarity. Always >= 1 when set.
    count: u32,
}

/// A non-atom frame queued for the current cycle (mode-2 only).
/// Drained by drain_mode2_pending() at cycle commit.
#[derive(Debug, Clone, Copy)]
enum PendingMode2Frame {
    NewPc { pc: u16 },
    Lc { flag: u8, count: u32 },
}

// Mode-2 LC frame bit-28 ("flag") is the 28-bit-overflow saturation
// indicator: set iff the trip count loaded into LC at ZOL start has any
// bit at position >= 28. Equivalently, set iff trip_count >= 2^28; the
// 28-bit count field carries trip_count mod 2^28. Single saturated bit,
// not a multi-overflow counter.
//
// Empirical confirmation: Phase 0 (2026-04-30) at N <= 16384 always
// flag=0; LC overflow probe (2026-05-08) flipped flag=1 at exactly
// N >= 2^28 and confirmed the count wraps cleanly modulo 2^28 through
// 2^29+5.
//   docs/archive/findings/2026-04-30-mode2-lc-flag-semantics.md
//   docs/archive/findings/2026-05-08-lc-overflow-empirical.md

/// Hardware trace unit for one tile module (Core or Memory).
///
/// Configured by register writes from CDO/control packets. When active,
/// it maps hardware events through 8 configurable slots, compresses
/// them into byte-encoded packets, and emits 8-word trace packets that
/// flow through the stream switch.
#[derive(Debug)]
pub struct TraceUnit {
    // -- Configuration (from register writes) --
    /// Operating mode.
    pub(super) mode: TraceMode,
    /// Hardware event ID that starts tracing.
    pub(super) start_event: u8,
    /// Hardware event ID that stops tracing.
    pub(super) stop_event: u8,
    /// Slot-to-hardware-event-ID mapping (8 slots).
    pub(super) event_slots: [u8; 8],
    /// Packet type field for the header (Trace_Control1 bits [14:12]).
    /// 0=core, 1=memmod, 2=shim/PL, 3=memtile.
    pub(super) packet_type: u8,
    /// Packet ID field for the header (Trace_Control1 bits [4:0]).
    pub(super) packet_id: u8,

    // -- Runtime state --
    /// Current state (Idle/Running/Stopped).
    pub(super) state: TraceState,
    /// Cycle counter (starts when tracing begins).
    timer: u64,
    /// Cycle of the last *emitted* frame (for delta computation). Updated
    /// when `commit_cycle()` flushes the pending slot mask.
    last_event_cycle: u64,
    /// Cycle that `pending_slot_mask` is being accumulated for. All slot
    /// bits set before this cycle advances are coalesced into one frame.
    pending_cycle: u64,
    /// Bitmask of slots whose events have fired during `pending_cycle`.
    /// Flushed by `commit_cycle()` or lazily on next-cycle event arrival.
    /// These are PULSE (one-cycle) events; the bit lives only for the frame
    /// of `pending_cycle`.
    pub(super) pending_slot_mask: u8,
    /// Bitmask of LEVEL-event slots currently asserted. Unlike
    /// `pending_slot_mask`, these persist across cycles until explicitly
    /// deasserted via `set_event_level`. The emitted frame mask is
    /// `held_mask | pending_slot_mask` (the full asserted snapshot), so a
    /// held level appears in every frame between its assert and deassert.
    /// No per-cycle re-emission occurs during a hold: `commit_cycle`
    /// early-returns when no pulse fired, and `set_event_level` only commits
    /// on an actual assert/deassert edge.
    pub(super) held_mask: u8,
    /// Held-level mask as of the *last emitted frame*. Distinct from
    /// `held_mask` (the live state): this records what was asserted when the
    /// previous frame went out, so `commit_pending_frame` knows whether a level
    /// was held *across the gap* since that frame. If so, the gap must be
    /// carried by skip tokens (`Repeat`) rather than the frame's `cycles`
    /// field -- upstream `parse_trace` deactivates all active events on any
    /// `cycles>0` frame, so a held level only survives a gap encoded as
    /// `cycles==0` + Repeat. See
    /// `docs/superpowers/specs/2026-06-08-skip-token-held-level-encoding.md`.
    frame_held: u8,
    /// True iff the currently-open lone hold was opened across an idle gap
    /// (`gap > 0`), which emits a *two-frame* open: a position frame carrying
    /// the gap, then a separate `cycles=0` arming frame. The extra arming frame
    /// advances the upstream decoder one uncompensated cycle past the level's B
    /// mark, so the closing skip-run for that hold must cover `D - 2` cycles
    /// rather than `D - 1`. A `gap == 0` open folds position+arm into one frame
    /// and needs no correction. Set at the lone-hold open; consumed (and
    /// cleared) at the lone close. Any intervening re-checkpoint
    /// (held-across-gap continuation) clears it -- the correction is scoped to
    /// the pure open->close case, the one with HW evidence (#140 RUN_1 spans).
    /// See `docs/superpowers/specs/2026-06-08-skip-token-held-level-encoding.md`
    /// (the "settle empirically" `-1` offset note).
    hold_opened_with_gap: bool,
    /// PC value associated with the most recent event in `pending_cycle`
    /// (mode 1 only). Mode 0 ignores this field entirely.
    pending_pc: u32,
    /// Rate-limit counter for PC truncation warnings (mode 1).
    pc_truncate_warnings: u32,
    /// Rate-limit counter for missing-PC sentinel warnings (mode 1).
    no_pc_warnings: u32,
    /// Accumulated encoded bytes waiting to be packed into packets.
    pub(super) byte_buffer: Vec<u8>,
    /// Individual words ready for emission to the stream switch.
    /// Each entry is (word, tlast). Words are pushed in order: header first,
    /// then 7 data words, with TLAST=true on the last word. This allows the
    /// routing layer to push one word at a time, respecting FIFO backpressure.
    pending_words: VecDeque<(u32, bool)>,

    /// Bit accumulator for mode-2 frame encoding. Bits are packed
    /// MSB-first into pending_word; when 32 bits are accumulated,
    /// flush_word_if_full() pushes 4 big-endian bytes to byte_buffer
    /// and resets the accumulator.
    pending_word: u32,
    pending_word_bits: u8,

    /// In-flight RLE state for consecutive E/N atoms (mode-2 only).
    pending_atoms_run: Option<AtomRun>,

    /// Non-atom frames (New_PC, LC) queued for the current cycle
    /// (mode-2 only). Drained by drain_mode2_pending() in
    /// commit_pending_frame.
    pending_mode2_frames: Vec<PendingMode2Frame>,

    // Mode-2 only: true while we are inside an active ZOL run (between the
    // first iteration's LE_PC boundary and the iteration where LC reaches 0).
    // HW emits exactly one LC frame per ZOL invocation, at the start, with
    // count = trip count. We use this flag to suppress per-iteration LC
    // emissions in `notify_loop_boundary`.
    mode2_zol_active: bool,

    // -- Tile identity (for packet headers) --
    /// Column of the owning tile.
    col: u8,
    /// Row of the owning tile.
    row: u8,

    /// Whether Control0 has been written (trace unit is configured).
    /// The trace unit only processes events after configuration.
    configured: bool,

    /// Cycle at which `start_event` was observed, if a state transition
    /// to `Running` is pending.
    ///
    /// Real HW's trace controller has a 1-cycle pipeline delay on the
    /// `Idle → Running` transition: when start_event arrives in cycle S,
    /// recording does NOT begin until cycle S+1. Our model originally
    /// transitioned state immediately, which caused mode-2 EMU traces to
    /// emit one extra `New_PC` at the kernel's loop entry compared to HW.
    /// See `docs/coverage/trace-start-stop-latency-gap.md` (2026-05-04).
    armed_start_cycle: Option<u64>,
    /// Anchor PC captured at start-event arming time (mode-2 only).
    /// Used when the deferred state transition activates and we emit the
    /// Start frame.
    armed_start_anchor: u16,
}

impl TraceUnit {
    /// Create a new trace unit for the given tile location.
    pub fn new(col: u8, row: u8) -> Self {
        Self {
            mode: TraceMode::EventTime,
            start_event: 0,
            stop_event: 0,
            event_slots: [0; 8],
            packet_type: 0,
            packet_id: 0,
            state: TraceState::Idle,
            timer: 0,
            last_event_cycle: 0,
            pending_cycle: 0,
            pending_slot_mask: 0,
            held_mask: 0,
            frame_held: 0,
            hold_opened_with_gap: false,
            pending_pc: 0,
            pc_truncate_warnings: 0,
            no_pc_warnings: 0,
            byte_buffer: Vec::with_capacity(64),
            pending_words: VecDeque::new(),
            pending_word: 0,
            pending_word_bits: 0,
            pending_atoms_run: None,
            pending_mode2_frames: Vec::new(),
            mode2_zol_active: false,
            col,
            row,
            configured: false,
            armed_start_cycle: None,
            armed_start_anchor: 0,
        }
    }

    /// Reset all state to constructor defaults, preserving only col/row.
    ///
    /// Called from `array.reset()` on submit_cmd / hw_context teardown so a
    /// new run sees a trace unit indistinguishable from a freshly created
    /// one. Without this, when the post-flush drain hits its iteration cap
    /// with packets still in flight, orphaned payload words remain in
    /// `pending_words` and leak into the next batch's fresh stream switch,
    /// where they get misinterpreted as packet headers (with garbage
    /// pkt_id) and trigger a fatal "no packet route" error.
    pub fn reset(&mut self) {
        *self = Self::new(self.col, self.row);
    }

    // -- Test helpers (pub(crate) for trace_unit::tests and tile::tests) --

    /// True only when this trace unit is configured as a core-module
    /// packet type and may legitimately enter EventPc mode.
    ///
    /// Per regdb (aie_registers_aie2.json), only core's Trace_Control0
    /// has a Mode bitfield; memmod, memtile, and shim trace units don't
    /// have the field, so setting EventPc on them would be a HW-impossible
    /// state. Private: only `apply_mode` consumes this; exposing it would
    /// leak a public API surface with no production caller.
    fn mode_supports_pc(&self) -> bool {
        self.packet_type == 0
    }

    /// Apply a new mode with a config-time guard: EventPc is only valid on
    /// core trace units (packet_type == 0). Non-core units that attempt to
    /// enter EventPc mode are clamped to EventTime with a logged error.
    fn apply_mode(&mut self, new_mode: TraceMode) {
        if matches!(new_mode, TraceMode::EventPc) && !self.mode_supports_pc() {
            log::error!(
                "TraceUnit ({},{}): EventPc mode requested on non-core \
                 packet_type={} (regdb: no Mode bitfield on this trace unit); \
                 clamping to EventTime",
                self.col,
                self.row,
                self.packet_type
            );
            self.mode = TraceMode::EventTime;
        } else {
            self.mode = new_mode;
        }
    }

    /// Return the current operating mode.
    #[cfg(test)]
    pub(crate) fn mode(&self) -> TraceMode {
        self.mode
    }

    /// Set the operating mode (test helper; routes through `apply_mode` guard).
    #[cfg(test)]
    pub(crate) fn set_mode(&mut self, mode: TraceMode) {
        self.apply_mode(mode);
    }

    /// Set the packet type (0=core, 1=memmod, 2=shim/PL, 3=memtile).
    /// Test helper -- the real path comes through `write_register(0x04, ...)`.
    #[cfg(test)]
    pub(crate) fn set_packet_type(&mut self, pkt_type: u8) {
        self.packet_type = pkt_type;
    }

    /// Map hw_event_id into slot `slot_idx` (0-7).
    /// Test helper -- the real path is `write_register(0x10/0x14, ...)`.
    #[cfg(test)]
    pub(crate) fn set_event_slot(&mut self, slot_idx: usize, hw_event_id: u8) {
        assert!(slot_idx < 8, "slot index out of range");
        self.event_slots[slot_idx] = hw_event_id;
    }

    /// Set the start event ID.
    /// Test helper -- the real path is `write_register(0x00, ...)`.
    #[cfg(test)]
    pub(crate) fn set_start_event(&mut self, hw_event_id: u8) {
        self.start_event = hw_event_id;
        self.configured = true;
    }

    /// Read a trace register (offset relative to the trace register block).
    ///
    /// Offsets:
    /// - 0x00: Trace_Control0
    /// - 0x04: Trace_Control1
    /// - 0x08: Trace_Status (state[9:8], mode[2:0])
    /// - 0x10: Trace_Event0
    /// - 0x14: Trace_Event1
    pub fn read_register(&self, offset: u32) -> u32 {
        match offset {
            0x00 => {
                // Trace_Control0: mode[1:0], start_event[23:16], stop_event[31:24]
                (self.mode as u32) | ((self.start_event as u32) << 16) | ((self.stop_event as u32) << 24)
            }
            0x04 => {
                // Trace_Control1: packet_type[14:12], packet_id[4:0]
                ((self.packet_type as u32) << 12) | (self.packet_id as u32)
            }
            0x08 => {
                // Trace_Status: state[9:8], mode[2:0]
                // Per aie-rt xaie_trace.c: XAie_TraceGetState reads STATE,
                // XAie_TraceGetMode reads MODE from this register.
                let state_bits: u32 = match self.state {
                    TraceState::Idle => 0,
                    TraceState::Running => 1,
                    TraceState::Stopped => 2,
                };
                (state_bits << 8) | (self.mode as u32)
            }
            0x10 => {
                // Trace_Event0: evt3[31:24], evt2[23:16], evt1[15:8], evt0[7:0]
                (self.event_slots[0] as u32)
                    | ((self.event_slots[1] as u32) << 8)
                    | ((self.event_slots[2] as u32) << 16)
                    | ((self.event_slots[3] as u32) << 24)
            }
            0x14 => {
                // Trace_Event1: evt7[31:24], evt6[23:16], evt5[15:8], evt4[7:0]
                (self.event_slots[4] as u32)
                    | ((self.event_slots[5] as u32) << 8)
                    | ((self.event_slots[6] as u32) << 16)
                    | ((self.event_slots[7] as u32) << 24)
            }
            _ => 0,
        }
    }

    /// Write a trace register (offset relative to the trace register block).
    ///
    /// Offsets:
    /// - 0x00: Trace_Control0
    /// - 0x04: Trace_Control1
    /// - 0x10: Trace_Event0
    /// - 0x14: Trace_Event1
    pub fn write_register(&mut self, offset: u32, value: u32) {
        match offset {
            0x00 => {
                // Trace_Control0: mode[1:0], start_event[23:16], stop_event[31:24]
                // AM025 claims 7-bit fields [22:16] and [30:24], but mlir-aie
                // uses the full byte (MemTile events go up to 160). Use 8 bits
                // to match real hardware behavior.
                let new_mode = TraceMode::from_u32(value);
                self.apply_mode(new_mode);
                self.start_event = ((value >> 16) & 0xFF) as u8;
                self.stop_event = ((value >> 24) & 0xFF) as u8;
                self.configured = true;

                // Programming Trace_Control0 starts a fresh trace session. On
                // real HW, writing the configuration register naturally moves
                // the trace unit to a clean Idle state -- no sticky Stopped
                // status carries forward across reconfigurations. Without this
                // reset, batch-stdin mode (where Trace_Control0 is rewritten
                // at the start of each batch's insts.bin) would only record
                // the first batch: state would latch in Stopped after the
                // first stop_event and no subsequent start_event could
                // re-arm it. Clear the per-session bookkeeping (state,
                // buffers, mode-2 ZOL tracking) so the next start_event
                // begins recording cleanly.
                self.state = TraceState::Idle;
                self.mode2_zol_active = false;
                self.byte_buffer.clear();
                self.pending_words.clear();
                self.pending_word = 0;
                self.pending_word_bits = 0;
                self.pending_atoms_run = None;
                self.pending_mode2_frames.clear();
                self.pending_slot_mask = 0;
                self.pending_pc = 0;
                self.pending_cycle = 0;
                self.armed_start_cycle = None;
                self.armed_start_anchor = 0;

                // Trace unit waits in Idle state until its start_event fires.
                // The CDO sequence configures broadcast channels, then writes
                // Event_Generate to fire USER_EVENT -> BROADCAST_N -> trace
                // start. The tile's write_register() handles Event_Generate
                // and propagates broadcast events to all tiles in the column,
                // which triggers notify_event() on each trace unit.

                log::debug!(
                    "TraceUnit ({},{}) Control0: mode={:?} start={} stop={} -> {:?}",
                    self.col,
                    self.row,
                    self.mode,
                    self.start_event,
                    self.stop_event,
                    self.state
                );
            }
            0x04 => {
                // Trace_Control1: packet_type[14:12], packet_id[4:0]
                self.packet_type = ((value >> 12) & 0x7) as u8;
                self.packet_id = (value & 0x1F) as u8;
                log::debug!(
                    "TraceUnit ({},{}) Control1: pkt_type={} pkt_id={}",
                    self.col,
                    self.row,
                    self.packet_type,
                    self.packet_id
                );
            }
            0x10 => {
                // Trace_Event0: evt3[31:24], evt2[23:16], evt1[15:8], evt0[7:0]
                self.event_slots[0] = (value & 0xFF) as u8;
                self.event_slots[1] = ((value >> 8) & 0xFF) as u8;
                self.event_slots[2] = ((value >> 16) & 0xFF) as u8;
                self.event_slots[3] = ((value >> 24) & 0xFF) as u8;
                log::debug!(
                    "TraceUnit ({},{}) Event0: slots[0-3]={:?}",
                    self.col,
                    self.row,
                    &self.event_slots[0..4]
                );
            }
            0x14 => {
                // Trace_Event1: evt7[31:24], evt6[23:16], evt5[15:8], evt4[7:0]
                self.event_slots[4] = (value & 0xFF) as u8;
                self.event_slots[5] = ((value >> 8) & 0xFF) as u8;
                self.event_slots[6] = ((value >> 16) & 0xFF) as u8;
                self.event_slots[7] = ((value >> 24) & 0xFF) as u8;
                log::debug!(
                    "TraceUnit ({},{}) Event1: slots[4-7]={:?}",
                    self.col,
                    self.row,
                    &self.event_slots[4..8]
                );
            }
            _ => {
                log::debug!("TraceUnit ({},{}) unknown register offset 0x{:X}", self.col, self.row, offset);
            }
        }
    }

    /// Notify the trace unit of a hardware event at the given cycle.
    ///
    /// `pc` is the program counter at the time of the event, used only
    /// when the trace unit is in EventPc mode (mode 1). Mode 0 ignores it.
    /// Pass `None` when the PC is not known; mode 1 will encode a sentinel
    /// pc=0 and rate-limit-warn.
    ///
    /// If the event matches a configured slot, its bit is set in the
    /// pending cycle mask but not encoded immediately. Bytes are emitted
    /// at cycle boundaries by `commit_cycle()` (or lazily when a new cycle
    /// arrives), which matches HW's "one frame per cycle" behavior.
    ///
    /// Start/stop events control the tracing state machine.
    pub fn notify_event(&mut self, hw_event_id: u8, cycle: u64, pc: Option<u32>) {
        if !self.configured {
            return;
        }

        // If we have a pending cycle frame and the event is for a newer
        // cycle, commit the old frame before accumulating into the new one.
        if cycle != self.pending_cycle && self.pending_slot_mask != 0 {
            self.commit_pending_frame();
        }

        // Promote a previously-armed start to active Running once the cycle
        // counter has advanced past the cycle in which start_event arrived.
        // This models HW's 1-cycle pipelined Idle→Running transition: events
        // that arrive in the same cycle as start_event are NOT recorded, but
        // events from the next cycle onward are. Without this latency, EMU
        // captured an extra New_PC at the kernel's loop-entry address that
        // HW never sees — see docs/coverage/trace-start-stop-latency-gap.md.
        if let Some(arm_cycle) = self.armed_start_cycle {
            if self.state == TraceState::Idle && cycle > arm_cycle {
                self.activate_armed_start(cycle);
            }
        }

        // Detect start/stop events. Start emits the Start marker
        // immediately (so the byte stream's Start frame is at the
        // beginning, matching HW), but the recording window itself is
        // *armed* — only events from the next cycle onward are captured.
        // Stop is immediate and discards same-cycle pending mode-2 frames.
        if self.state == TraceState::Idle && hw_event_id == self.start_event {
            if self.armed_start_cycle.is_none() {
                let anchor_pc = pc.map(|p| (p & 0x3FFF) as u16).unwrap_or(0);
                self.armed_start_cycle = Some(cycle);
                self.armed_start_anchor = anchor_pc;
                // Emit the Start marker now. State stays Idle until
                // cycle advances past `cycle` (HW's pipelined start).
                // last_event_cycle bookkeeping is also updated so that
                // mode-0 delta encoding sees a consistent baseline; no
                // events between now and the activation cycle will
                // accumulate (state is Idle), so this is safe.
                self.timer = cycle;
                self.last_event_cycle = cycle;
                self.pending_cycle = cycle;
                self.pending_slot_mask = 0;
                match self.mode {
                    TraceMode::Execution => self.encode_mode2_start(anchor_pc),
                    _ => self.encode_start(cycle),
                }
                log::debug!(
                    "TraceUnit ({},{}) start armed: cycle={} hw_event={} mode={:?} pc={:?}",
                    self.col,
                    self.row,
                    cycle,
                    hw_event_id,
                    self.mode,
                    pc
                );
            }
            return;
        }

        if self.state == TraceState::Running && hw_event_id == self.stop_event {
            self.state = TraceState::Stopped;
            self.mode2_zol_active = false;
            log::debug!("TraceUnit ({},{}) stopped at cycle {}", self.col, self.row, cycle);
            if matches!(self.mode, TraceMode::Execution) {
                // Discard pending mode-2 frames from the stop cycle. Real HW
                // captures none of the cycle in which stop_event arrives
                // (state goes inactive on the same cycle edge); previously
                // we drained pending here, which emitted one extra New_PC
                // beyond HW's window. Atoms-run and ZOL state are reset
                // inside drain_mode2_pending, but we do NOT want to flush
                // their bytes — clear them too.
                self.pending_mode2_frames.clear();
                self.pending_atoms_run = None;
                self.encode_mode2_stop();
            }
            // Flush remaining data as a final packet
            self.flush();
            return;
        }

        // Only record events while running
        if self.state != TraceState::Running {
            return;
        }

        // Find matching slot (0-7) for this hardware event ID
        let slot = match self.event_slots.iter().position(|&s| s == hw_event_id) {
            Some(idx) => idx as u8,
            None => return, // Event not in any slot
        };

        // Accumulate into the pending-cycle mask.
        self.pending_cycle = cycle;
        self.pending_slot_mask |= 1 << slot;

        // PC tracking for mode 1: record the PC for this pending frame.
        // When multiple slots fire in the same cycle, we use the last PC
        // seen (hardware records one PC per EventPC frame regardless of mask).
        if matches!(self.mode, TraceMode::EventPc) {
            match pc {
                Some(p) => self.pending_pc = p,
                None => {
                    // Last-seen-wins: if a Some(pc) call earlier in this same
                    // cycle had recorded a real PC, this None call overwrites
                    // it with the sentinel. HW records one PC per frame
                    // regardless of how many slots coalesce; we mirror that
                    // by letting the most recent notify win, including the
                    // sentinel.
                    if self.no_pc_warnings < 4 {
                        log::warn!(
                            "TraceUnit ({},{}): EventPc mode received event \
                             hw_id={} with no PC; encoding sentinel pc=0",
                            self.col,
                            self.row,
                            hw_event_id
                        );
                        self.no_pc_warnings += 1;
                    }
                    self.pending_pc = 0;
                }
            }
        }
    }

    /// Assert (`active=true`) or deassert (`active=false`) a LEVEL trace event.
    ///
    /// Unlike `notify_event` (a one-cycle pulse), a level's slot bit persists
    /// in `held_mask` across cycles until deasserted, so the asserted snapshot
    /// (`held_mask | pending_slot_mask`) includes it in every frame between
    /// assert and deassert -- the mode-0 decoder then renders one B..E span of
    /// the real duration instead of one span per cycle.
    ///
    /// A frame is committed at the rising edge (assert). The falling edge is
    /// carried by the next frame whose mask drops the bit -- typically the
    /// event that ends the condition (e.g. the lock acquire that ends a stall),
    /// which fires in the same cycle as the deassert. This matches how real HW
    /// closes a level span (it emits no synthetic empty frame). If a level
    /// deasserts with nothing else asserted and no coincident event, the span
    /// closes at the next frame / end-of-segment -- acceptable, and the same
    /// behavior HW exhibits.
    ///
    /// Callers classify level vs pulse via the archspec `is_level_event`
    /// authority and route accordingly (levels here, pulses to `notify_event`).
    pub fn set_event_level(&mut self, hw_event_id: u8, cycle: u64, active: bool) {
        if !self.configured {
            return;
        }
        // Mirror notify_event's deferred-start promotion so a level asserted
        // as the first post-start event isn't dropped.
        if let Some(arm_cycle) = self.armed_start_cycle {
            if self.state == TraceState::Idle && cycle > arm_cycle {
                self.activate_armed_start(cycle);
            }
        }
        if self.state != TraceState::Running {
            return;
        }
        let slot = match self.event_slots.iter().position(|&s| s == hw_event_id) {
            Some(idx) => idx as u8,
            None => return, // Event not in any slot
        };
        let bit = 1u8 << slot;
        let new_held = if active {
            self.held_mask | bit
        } else {
            self.held_mask & !bit
        };
        if new_held == self.held_mask {
            return; // No edge -- already in the requested state.
        }
        // Flush any pending pulse accumulated for an *earlier* cycle first, so
        // its frame keeps the pre-transition snapshot. A pulse in the *same*
        // cycle (e.g. the acquire that ends a stall) is intentionally folded
        // into this transition's frame.
        if cycle != self.pending_cycle && self.pending_slot_mask != 0 {
            self.commit_pending_frame();
        }
        self.held_mask = new_held;
        self.pending_cycle = cycle;
        self.commit_pending_frame();
    }

    /// Mode-2 only: record the disposition of one conditional branch.
    /// `taken=true` queues an E_atom; `taken=false` queues an N_atom.
    ///
    /// Per AM020 ch.2 (L299-305), mode-2 records "Conditional and
    /// unconditional direct branches, all indirect branches, ZOL LC".
    /// Atoms are emitted once per *conditional branch*, not once per
    /// cycle: E for "condition fired", N for "fall-through".
    /// Unconditional branches emit no atom (they always execute).
    pub fn notify_atom(&mut self, taken: bool) {
        if !self.is_mode2_running() {
            return;
        }
        self.append_atom_to_run(taken);
    }

    /// Mode-2 only: queue a New_PC frame for the current cycle.
    ///
    /// Per AM020, the trace records targets only for branches whose
    /// destination cannot be statically deduced from the ELF: returns
    /// (target = LR), indirect calls/jumps, and indirect-target
    /// conditional taken (e.g. `JNZD r,r,p7`). Direct branches with an
    /// immediate target emit no New_PC -- the offline debugger reads
    /// the target from the instruction encoding.
    pub fn notify_branch_taken(&mut self, _cycle: u64, retire_pc: u32) {
        if !self.is_mode2_running() {
            return;
        }
        let pc = (retire_pc & 0x3FFF) as u16;
        self.pending_mode2_frames.push(PendingMode2Frame::NewPc { pc });
    }

    /// Mode-2 only: called at every ZOL iteration's LE_PC boundary.
    ///
    /// HW emits exactly one LC frame per ZOL invocation -- at the first
    /// iteration -- with `count` = trip_count mod 2^28 (the low 28 bits
    /// of `lc_before` on the first call) and `flag` = 1 iff trip_count
    /// has any bit at position >= 28 (i.e. trip_count >= 2^28). Subsequent
    /// iterations of the same ZOL produce no LC frame; the per-cycle
    /// E_atom / N_atom stream already records the body's execution.
    pub fn notify_loop_boundary(&mut self, _cycle: u64, lc_before: u32, lc_after: u32) {
        if !self.is_mode2_running() {
            return;
        }
        if !self.mode2_zol_active {
            let count = lc_before & 0x0FFF_FFFF;
            let flag = (lc_before >> 28 != 0) as u8;
            self.pending_mode2_frames.push(PendingMode2Frame::Lc { flag, count });
            self.mode2_zol_active = true;
        }
        if lc_after == 0 {
            self.mode2_zol_active = false;
        }
    }

    /// True iff the trace unit is currently running.
    pub fn is_running(&self) -> bool {
        self.state == TraceState::Running
    }

    fn is_mode2_running(&self) -> bool {
        matches!(self.mode, TraceMode::Execution) && self.is_running()
    }

    /// Commit any accumulated slot activity for cycles <= `cycle`.
    ///
    /// Called by the coordinator at the end of every cycle in which one or
    /// more trace events may have fired. Writes exactly one frame (Single
    /// or Multiple encoding) to the byte buffer for each cycle that had
    /// any pending events, matching the HW "one frame per cycle" rule.
    pub fn commit_cycle(&mut self, cycle: u64) {
        if self.state != TraceState::Running {
            return;
        }
        // Mode-2 doesn't use the slot-based mask; it always drains its
        // own per-cycle queues. Mode 0/1 only commit when at least one
        // event fired this cycle.
        if !matches!(self.mode, TraceMode::Execution) {
            if self.pending_slot_mask == 0 {
                return;
            }
            if self.pending_cycle > cycle {
                return;
            }
        }
        self.commit_pending_frame();
    }

    /// Advance the timer. Called each cycle to keep the internal clock in sync.
    pub fn tick(&mut self, cycle: u64) {
        if self.state == TraceState::Running {
            self.timer = cycle;
        }
    }

    /// Pop one complete trace packet (8 x u32 words).
    ///
    /// Returns `None` if no packets are ready. Consumes 8 words from the
    /// pending queue. Callers that need backpressure support should use
    /// `pop_word()` instead.
    pub fn pop_packet(&mut self) -> Option<[u32; 8]> {
        if self.pending_words.len() < 8 {
            return None;
        }
        let mut packet = [0u32; 8];
        for p in &mut packet {
            *p = self.pending_words.pop_front().unwrap().0;
        }
        Some(packet)
    }

    /// Pop a single word with its TLAST flag.
    ///
    /// Returns `None` if no words are pending. The caller should push the
    /// word to the stream switch slave port, checking `can_accept()` first
    /// to respect FIFO backpressure.
    pub fn pop_word(&mut self) -> Option<(u32, bool)> {
        self.pending_words.pop_front()
    }

    /// Check if there are pending words to emit.
    pub fn has_pending_words(&self) -> bool {
        !self.pending_words.is_empty()
    }

    /// Check if there are pending packets to emit (8+ words queued).
    pub fn has_pending_packets(&self) -> bool {
        self.pending_words.len() >= 8
    }

    /// Check if the trace unit is configured (mode != Off).
    pub fn is_configured(&self) -> bool {
        self.configured
    }

    /// Return the number of encoded bytes currently buffered.
    ///
    /// Bytes accumulate as events are committed and are consumed when packed
    /// into 8-word packets. A non-zero value means at least one trace frame
    /// has been encoded since the last flush. Crate-internal accessor used
    /// by integration tests that verify event routing without inspecting
    /// packet content; not part of the external API.
    #[cfg(test)]
    pub(crate) fn encoded_bytes_len(&self) -> usize {
        self.byte_buffer.len()
    }

    /// Crate-internal read access to the encoded byte buffer for tests that
    /// need to decode the exact frame layout (e.g., verifying same-cycle
    /// events coalesce into one Multiple frame). Not part of the external API.
    #[cfg(test)]
    pub(crate) fn encoded_bytes(&self) -> &[u8] {
        &self.byte_buffer
    }

    /// Flush any remaining bytes as a padded final packet.
    ///
    /// Commits any pending same-cycle accumulation first so its frame lands
    /// in the byte buffer before padding is applied.
    pub fn flush(&mut self) {
        if self.pending_slot_mask != 0 {
            self.commit_pending_frame();
        }

        // Mode-2: drain any in-flight atom run / queued frames, then
        // align the pending bit accumulator out to a 32-bit word so
        // the byte buffer captures all encoded data before padding.
        if matches!(self.mode, TraceMode::Execution) {
            self.drain_mode2_pending();
            if self.pending_word_bits > 0 {
                self.align_to_word_via_filler0();
            }
        }

        if self.byte_buffer.is_empty() {
            return;
        }

        log::trace!(
            "TraceUnit ({},{}) flush: {} bytes -> padded packet (packets so far: {})",
            self.col,
            self.row,
            self.byte_buffer.len(),
            self.pending_words.len() / 8
        );

        // Pad to 28 bytes with 0xFE
        while self.byte_buffer.len() < 28 {
            self.byte_buffer.push(0xFE);
        }

        self.emit_packet_from_buffer();
    }

    // -- Internal encoding methods --

    /// Commit the pending slot mask for `pending_cycle` as one HW-accurate
    /// trace frame.
    ///
    /// Mode 0 (EventTime):
    ///   - popcount == 1: Single0/1/2 (1/2/3 bytes)
    ///   - popcount >= 2: Multiple0/1/2 (2/3/4 bytes)
    ///
    /// Mode 1 (EventPc):
    ///   - always 4-byte EventPC frame: 8-bit mask + 14-bit PC
    ///   - cycle deltas are NOT emitted in mode 1
    ///
    /// This matches AM020/AM025 behavior and the mode1.py decoder.
    fn commit_pending_frame(&mut self) {
        // The frame carries the full asserted snapshot: held levels OR'd with
        // this cycle's pulses. Pulses are consumed (cleared) after; held bits
        // persist in `held_mask` until deasserted.
        let active = self.held_mask | self.pending_slot_mask;
        // Pulse-only bits (one-cycle events not held as levels). Inside a hold
        // these need an explicit close frame, since upstream does not
        // deactivate events on a cycles==0 frame.
        let pulses = self.pending_slot_mask & !self.held_mask;
        self.pending_slot_mask = 0;

        match self.mode {
            TraceMode::EventTime => {
                // Skip-token model, matching upstream `aie.utils.trace.parse_trace`
                // (the decoder the regression comparison actually runs). Upstream
                // deactivates ALL active events on any `cycles>0` frame, so a held
                // level only survives a multi-cycle gap if that gap is encoded as a
                // `cycles==0` frame plus `Repeat` tokens. Design + HW reference:
                // docs/superpowers/specs/2026-06-08-skip-token-held-level-encoding.md.
                if active == 0 {
                    // A held level deasserted with nothing coincident. HW emits
                    // the hold's Repeat tokens at the drop (not deferred), then
                    // closes the level at the following frame. Emit the skip run
                    // for the hold's real duration now, advance the anchor to the
                    // drop, and clear frame_held -- so a re-assertion re-opens via
                    // the normal hold-open path, whose `Event(cyc=drop_gap),
                    // Event(cyc=0)` IS the HW re-checkpoint that deactivates the
                    // survivor and re-activates it, splitting a momentary drop
                    // into two Repeat runs. (Previously a bare no-op return, which
                    // merged drop+reassert into one continuous hold and
                    // over-extended a pure-deassert close to the next event.)
                    if self.frame_held != 0 {
                        let hold = self.pending_cycle.saturating_sub(self.last_event_cycle);
                        if hold > 1 {
                            // A gap-opened hold paid an extra arming-frame cycle
                            // at open (see `hold_opened_with_gap`); cover one
                            // fewer cycle here so the decoded span equals `hold`.
                            let extra = self.hold_opened_with_gap as u64;
                            self.emit_skip_run(hold - 1 - extra);
                        }
                        self.hold_opened_with_gap = false;
                        self.last_event_cycle = self.pending_cycle;
                        self.frame_held = 0;
                        self.try_emit_packet();
                    }
                    return;
                }
                let gap = self.pending_cycle.saturating_sub(self.last_event_cycle);
                if self.frame_held != 0 {
                    // A level was held across the gap: cover it with skip tokens so
                    // upstream keeps survivors active, then emit this frame at
                    // cycles=0. The closing frame's implicit +1 supplies the final
                    // hold cycle, so the run is `gap - 1` (matches HW: a 6354-cycle
                    // hold -> 6353 cycles of Repeat).
                    if gap > 1 {
                        self.emit_skip_run(gap - 1);
                    }
                    self.emit_event_frame(active, 0);
                    self.close_pulses_during_hold(pulses);
                    // A re-checkpoint re-anchors the timeline (its own arming
                    // frame is balanced by the `gap - 1` run above); the prior
                    // gap-open correction no longer applies to the eventual
                    // close, so fall back to the uncorrected path.
                    self.hold_opened_with_gap = false;
                } else if self.held_mask != 0 {
                    // Opening a hold (no level was held across the gap). With an
                    // idle gap, position+activate with the gap in the cycles field
                    // first; then an immediate cycles=0 frame arms the skip
                    // mechanism for the upcoming hold (mirrors HW's pulse-then-
                    // cycles=0 hold open). At gap 0 the single cycles=0 frame
                    // suffices.
                    if gap > 0 {
                        self.emit_event_frame(active, gap);
                        // Two-frame open: the separate arming frame below adds
                        // an uncompensated decoder cycle; the lone close must
                        // shorten its skip-run by 1 to keep the span exact.
                        self.hold_opened_with_gap = true;
                    } else {
                        self.hold_opened_with_gap = false;
                    }
                    self.emit_event_frame(active, 0);
                    self.close_pulses_during_hold(pulses);
                } else {
                    // Pure pulse / nothing held across the gap: the gap rides in the
                    // frame's cycles field, as before.
                    self.emit_event_frame(active, gap);
                }
                self.last_event_cycle = self.pending_cycle;
                self.frame_held = self.held_mask;
            }
            TraceMode::EventPc => {
                // EventPc carries a PC per frame; include held levels in the
                // mask so they are not dropped mid-span.
                if active == 0 {
                    return;
                }
                let pc_full = self.pending_pc;
                let pc14 = (pc_full & 0x3FFF) as u16;
                if pc_full > 0x3FFF && self.pc_truncate_warnings < 4 {
                    log::warn!(
                        "TraceUnit ({},{}): PC 0x{:X} truncated to 14 bits (0x{:X})",
                        self.col,
                        self.row,
                        pc_full,
                        pc14
                    );
                    self.pc_truncate_warnings += 1;
                }
                self.encode_event_pc(active, pc14);
            }
            TraceMode::Execution => {
                // Mode-2 doesn't use the slot-based mask. Atoms are
                // queued via notify_core_active/stalled, branches via
                // notify_branch_taken, loop boundaries via
                // notify_loop_boundary. Drain here at cycle commit.
                self.drain_mode2_pending();
            }
            TraceMode::Reserved => {
                // mode 3 not defined per xaie_trace.h; ignore.
            }
        }
        self.try_emit_packet();
    }

    /// Encode a 4-byte EventPC frame: 8-bit event mask + 14-bit PC.
    ///
    /// Layout (MSB-first), per tools/trace_decoder/modes/mode1.py:
    ///   byte0 = 0b1100_0100 | (mask >> 6)   (top 6 bits opcode = 0x31; low 2 = mask[7:6])
    ///   byte1 = (mask & 0x3F) << 2           (mask[5:0] + 2 reserved bits = 0)
    ///   byte2 = (pc >> 8) & 0x3F             (2 reserved bits = 0 + PC[13:8])
    ///   byte3 = pc & 0xFF                    (PC[7:0])
    ///
    /// The decoder checks `(b & 0b11111100) == 0b11000100`, then:
    ///   mask = ((b & 0b11) << 6) | (b1 >> 2)
    ///   pc   = ((b2 & 0b00111111) << 8) | b3
    fn encode_event_pc(&mut self, mask: u8, pc: u16) {
        debug_assert!(pc < (1 << 14), "PC {} exceeds 14-bit range", pc);
        let byte0 = 0b1100_0100u8 | ((mask >> 6) & 0b11);
        let byte1 = (mask & 0b0011_1111) << 2;
        let byte2 = ((pc >> 8) as u8) & 0b0011_1111;
        let byte3 = (pc & 0xFF) as u8;
        self.byte_buffer.push(byte0);
        self.byte_buffer.push(byte1);
        self.byte_buffer.push(byte2);
        self.byte_buffer.push(byte3);
    }

    /// Emit one mode-0 event frame for the asserted snapshot `active` with the
    /// given `cycles` delta: a Single0/1/2 when exactly one slot is set, a
    /// Multiple0/1/2 otherwise. Shared by the skip-token commit paths.
    fn emit_event_frame(&mut self, active: u8, cycles: u64) {
        debug_assert!(active != 0, "emit_event_frame called with empty mask");
        if active.count_ones() == 1 {
            self.encode_single(active.trailing_zeros() as u8, cycles);
        } else {
            self.encode_multiple(active, cycles);
        }
    }

    /// After a `cycles==0` frame that carried one-cycle `pulses` alongside held
    /// levels, emit a follow-up `held_mask, cycles=0` frame to deactivate those
    /// pulses. Upstream `parse_trace` only deactivates an active event when a
    /// later frame's mask omits it; inside a hold every frame is `cycles==0`, so
    /// without this close frame the pulse would stay asserted and merge with the
    /// next same-slot pulse (dropping its count). Mirrors HW's
    /// `Multiple(held|pulse) ; Multiple(held)` pattern (e.g. commands [42]->[43]
    /// of the distribute_lateral core stream). No-op when no pulse fired or no
    /// level remains held -- a lone pulse closes at the next frame as before.
    fn close_pulses_during_hold(&mut self, pulses: u8) {
        if pulses != 0 && self.held_mask != 0 {
            self.emit_event_frame(self.held_mask, 0);
        }
    }

    /// Encode a single event with a cycle delta.
    ///
    /// Chooses Single0 (1 byte), Single1 (2 bytes), or Single2 (3 bytes)
    /// based on the magnitude of the delta.
    ///
    /// Bit layouts (from mlir-aie utils/trace/utils.py decode logic):
    ///
    /// Single0: `0b0EEETTTT` (1 byte)
    ///   - bits [6:4] = slot (event), bits [3:0] = delta (0-15)
    ///
    /// Single1: `0b100EEETT TTTTTTTT` (2 bytes)
    ///   - byte0 bits [7:5] = 100, [4:2] = slot, [1:0] = delta_hi
    ///   - byte1 = delta_lo. Total delta = 10 bits (0-1023).
    ///
    /// Single2: `0b101EEETT TTTTTTTT TTTTTTTT` (3 bytes)
    ///   - byte0 bits [7:5] = 101, [4:2] = slot, [1:0] = delta_hi
    ///   - byte1-2 = delta_mid_lo. Total delta = 18 bits (0-262143).
    fn encode_single(&mut self, slot: u8, delta: u64) {
        debug_assert!(slot < 8);

        if delta <= 15 {
            // Single0: 0b0EEETTTT (1 byte)
            let byte = (slot << 4) | (delta as u8 & 0x0F);
            self.byte_buffer.push(byte);
        } else if delta <= 1023 {
            // Single1: 0b100EEETT TTTTTTTT (2 bytes)
            let d = delta as u16;
            let byte0 = 0x80 | (slot << 2) | ((d >> 8) as u8 & 0x03);
            let byte1 = (d & 0xFF) as u8;
            self.byte_buffer.push(byte0);
            self.byte_buffer.push(byte1);
        } else {
            // Single2: 0b101EEETT TTTTTTTT TTTTTTTT (3 bytes)
            // Clamp to 18 bits (262143)
            let d = delta.min(0x3FFFF) as u32;
            let byte0 = 0xA0 | (slot << 2) | ((d >> 16) as u8 & 0x03);
            let byte1 = ((d >> 8) & 0xFF) as u8;
            let byte2 = (d & 0xFF) as u8;
            self.byte_buffer.push(byte0);
            self.byte_buffer.push(byte1);
            self.byte_buffer.push(byte2);
        }
    }

    /// Encode multiple simultaneous events as one Multiple0/1/2 frame.
    ///
    /// Bit layouts match mlir-aie's `convert_to_commands` decoder:
    ///
    /// Multiple0: `0b1100EEEE EEEETTTT` (2 bytes)
    ///   - byte0 bits [7:4] = 0b1100 (discriminator)
    ///   - byte0 bits [3:0] = mask[7:4] (high nibble of slot bitmask)
    ///   - byte1 bits [7:4] = mask[3:0] (low nibble of slot bitmask)
    ///   - byte1 bits [3:0] = delta (4 bits, 0-15)
    ///
    /// Multiple1: `0b110100EE EEEEEETT TTTTTTTT` (3 bytes)
    ///   - byte0 bits [7:2] = 0b110100 (discriminator)
    ///   - byte0 bits [1:0] = mask[7:6]
    ///   - byte1 bits [7:2] = mask[5:0]
    ///   - byte1 bits [1:0] = delta[9:8]
    ///   - byte2        = delta[7:0] (total 10 bits, 0-1023)
    ///
    /// Multiple2: `0b110101EE EEEEEETT TTTTTTTT TTTTTTTT` (4 bytes)
    ///   - byte0 bits [7:2] = 0b110101 (discriminator)
    ///   - byte0 bits [1:0] = mask[7:6]
    ///   - byte1 bits [7:2] = mask[5:0]
    ///   - byte1 bits [1:0] = delta[17:16]
    ///   - byte2        = delta[15:8]
    ///   - byte3        = delta[7:0] (total 18 bits, 0-262143)
    fn encode_multiple(&mut self, mask: u8, delta: u64) {
        debug_assert!(mask.count_ones() >= 2);

        if delta <= 15 {
            // Multiple0 (2 bytes)
            let byte0 = 0xC0 | (mask >> 4);
            let byte1 = ((mask & 0x0F) << 4) | (delta as u8 & 0x0F);
            self.byte_buffer.push(byte0);
            self.byte_buffer.push(byte1);
        } else if delta <= 1023 {
            // Multiple1 (3 bytes)
            let d = delta as u16;
            let byte0 = 0xD0 | (mask >> 6);
            let byte1 = ((mask & 0x3F) << 2) | ((d >> 8) as u8 & 0x03);
            let byte2 = (d & 0xFF) as u8;
            self.byte_buffer.push(byte0);
            self.byte_buffer.push(byte1);
            self.byte_buffer.push(byte2);
        } else {
            // Multiple2 (4 bytes), clamp delta to 18 bits
            let d = delta.min(0x3FFFF) as u32;
            let byte0 = 0xD4 | (mask >> 6);
            let byte1 = ((mask & 0x3F) << 2) | ((d >> 16) as u8 & 0x03);
            let byte2 = ((d >> 8) & 0xFF) as u8;
            let byte3 = (d & 0xFF) as u8;
            self.byte_buffer.push(byte0);
            self.byte_buffer.push(byte1);
            self.byte_buffer.push(byte2);
            self.byte_buffer.push(byte3);
        }
    }

    /// Mode-0 Repeat0 frame: `0b1110_RRRR` (1 byte), 4-bit repeat count (0..15).
    ///
    /// Emitted byte-aligned into the mode-0 byte buffer -- distinct from the
    /// mode-2 `encode_repeat0`, which packs the same bit pattern through the
    /// nibble accumulator. A Repeat after a `cycles==0` frame extends the
    /// decoder timer linearly without deactivating held events (upstream
    /// `parse.py` `convert_commands_to_json`), which is how a held level's
    /// duration is encoded.
    fn encode_mode0_repeat0(&mut self, n: u8) {
        debug_assert!(n < 16, "Repeat0 count {} exceeds 4 bits", n);
        self.byte_buffer.push(0xE0 | (n & 0x0F));
    }

    /// Mode-0 Repeat1 frame: `0b110110_RR RRRRRRRR` (2 bytes), 10-bit count
    /// (0..1023). Byte-aligned mode-0 counterpart to mode-2 `encode_repeat1`.
    fn encode_mode0_repeat1(&mut self, n: u16) {
        debug_assert!(n < 1024, "Repeat1 count {} exceeds 10 bits", n);
        self.byte_buffer.push(0xD8 | ((n >> 8) as u8 & 0x03));
        self.byte_buffer.push((n & 0xFF) as u8);
    }

    /// Emit a skip-token run covering `cycles` cycles of held-level hold time,
    /// using `Repeat1(1023)` chunks plus a final `Repeat0`/`Repeat1` remainder.
    ///
    /// Matches HW's held-level duration encoding exactly: HW's first long
    /// LOCK_STALL hold (6353 cycles) decodes to six `Repeat1(1023)` + one
    /// `Repeat1(215)`, which this reproduces. The caller must have emitted the
    /// opening `cycles==0` frame first so the decoder treats these as linear
    /// timer extension rather than a deactivate/activate replay.
    fn emit_skip_run(&mut self, mut cycles: u64) {
        while cycles >= 1024 {
            self.encode_mode0_repeat1(1023);
            cycles -= 1023;
        }
        if cycles == 0 {
            return;
        }
        if cycles <= 15 {
            self.encode_mode0_repeat0(cycles as u8);
        } else {
            self.encode_mode0_repeat1(cycles as u16);
        }
    }

    /// Encode a Start marker with 56-bit timer value.
    ///
    /// Mode 0: prefix byte 0xF0 (bit 0 = 0, trace-mode discriminator)
    /// Mode 1: prefix byte 0xF1 (bit 0 = 1, signals EventPc stream)
    ///
    /// The mode1.py decoder matches `(b & 0b11110011) == 0b11110001`,
    /// which catches 0xF1 (segment start) and 0xF5 (mid-stream re-anchor).
    /// Activate a previously-armed start: transition to Running, prime
    /// per-cycle bookkeeping, and emit the Start marker.
    ///
    /// `cycle` is the *current* cycle (i.e. the cycle in which we noticed
    /// the arm cycle had passed) — the trace timer starts here, since that
    /// matches HW's behavior of beginning to record on the cycle following
    /// start_event.
    fn activate_armed_start(&mut self, current_cycle: u64) {
        // Use the arm cycle (when start_event arrived) for the trace
        // timer, matching the encoded "trace start time" in the byte
        // stream. The current cycle is where the *first event* will
        // accumulate. This split keeps mode-0 delta encoding stable:
        // delta = pending_cycle (= current_cycle) - last_event_cycle
        // (= arm_cycle), which equals the old immediate-start behavior
        // for tests where start fires at cycle 0 and the first slot
        // event fires at cycle N.
        let arm_cycle = self.armed_start_cycle.take().unwrap_or(current_cycle);
        let anchor_pc = self.armed_start_anchor;
        self.armed_start_anchor = 0;

        self.state = TraceState::Running;
        self.mode2_zol_active = false;
        // timer / last_event_cycle / pending_cycle were already primed at
        // arm time. The Start marker bytes were also emitted at arm time so
        // the byte stream's Start frame sits at the beginning, matching HW.
        // We just flip state here. The first event captured will be from
        // current_cycle onward, since events from the arm cycle were
        // dropped while state == Idle.
        let _ = arm_cycle; // arm_cycle now lives only in the timer field
        log::debug!(
            "TraceUnit ({},{}) START activated: arm_cycle={} current_cycle={} mode={:?} anchor_pc={:#x}",
            self.col,
            self.row,
            arm_cycle,
            current_cycle,
            self.mode,
            anchor_pc,
        );
    }

    fn encode_start(&mut self, timer: u64) {
        let prefix = match self.mode {
            TraceMode::EventPc => 0xF1u8,
            _ => 0xF0u8,
        };
        self.byte_buffer.push(prefix);
        // 7 bytes of timer, big-endian (56 bits)
        for i in (0..7).rev() {
            self.byte_buffer.push(((timer >> (i * 8)) & 0xFF) as u8);
        }
    }

    /// If the byte buffer has >= 28 bytes, pack them into an 8-word packet.
    fn try_emit_packet(&mut self) {
        while self.byte_buffer.len() >= 28 {
            self.emit_packet_from_buffer();
        }
    }

    /// Encode a single E_atom (executed=true) or N_atom (executed=false).
    /// 4-bit frame: 0001 for E, 0000 for N.
    fn encode_atom(&mut self, executed: bool) {
        let prefix = if executed { 0b0001 } else { 0b0000 };
        self.push_bits(prefix, 4);
    }

    /// Encode a New_PC frame: 2-bit prefix `10` + 14-bit PC.
    fn encode_new_pc(&mut self, pc: u16) {
        debug_assert!(pc < (1 << 14), "PC {:#x} exceeds 14 bits", pc);
        let frame = (0b10u32 << 14) | (pc as u32 & 0x3FFF);
        self.push_bits(frame, 16);
    }

    /// Encode an LC (loop count) long frame: 3-bit prefix 010 +
    /// 1-bit flag + 28-bit count = 32-bit total.
    ///
    /// Aligns to word boundary via Filler0 padding before emitting.
    /// Flag = 1 iff trip count >= 2^28 (28-bit-overflow saturation);
    /// count = trip_count mod 2^28. See
    /// docs/archive/findings/2026-05-08-lc-overflow-empirical.md.
    fn encode_lc(&mut self, flag: u8, count: u32) {
        debug_assert!(flag <= 1, "LC flag must be 0 or 1");
        debug_assert!(count < (1u32 << 28), "LC count exceeds 28 bits");
        let word = (0b010u32 << 29) | ((flag as u32 & 1) << 28) | (count & 0x0FFFFFFF);
        self.emit_long_frame(word);
    }

    /// Encode a Repeat0 frame: 4-bit prefix 1110 + 4-bit count.
    fn encode_repeat0(&mut self, n: u8) {
        debug_assert!(n < 16, "Repeat0 count {} exceeds 4 bits", n);
        let frame = (0b1110u32 << 4) | (n as u32 & 0xF);
        self.push_bits(frame, 8);
    }

    /// Encode a Repeat1 frame: 6-bit prefix 110110 + 10-bit count.
    fn encode_repeat1(&mut self, n: u16) {
        debug_assert!(n < 1024, "Repeat1 count {} exceeds 10 bits", n);
        let frame = (0b110110u32 << 10) | (n as u32 & 0x3FF);
        self.push_bits(frame, 16);
    }

    /// Encode a mode-2 Start frame: 5-bit prefix 11110 + 1-bit flag
    /// (0 = initial start) + 12 reserved bits + 14-bit anchor PC.
    fn encode_mode2_start(&mut self, anchor_pc: u16) {
        debug_assert!(anchor_pc < (1 << 14));
        let word = (0b11110u32 << 27) | (anchor_pc as u32 & 0x3FFF);
        self.emit_long_frame(word);
    }

    /// Encode a mode-2 Stop frame: 6-bit prefix 110111 + 26 reserved
    /// bits.
    fn encode_mode2_stop(&mut self) {
        let word = 0b110111u32 << 26;
        self.emit_long_frame(word);
    }

    /// Append one atom to the in-flight RLE run. If the run's polarity
    /// flips (E to N or vice versa), flush the previous run first.
    ///
    /// The flush itself caps each Repeat1 chunk at 10 bits (1023), so a
    /// run can grow arbitrarily large -- chaining is handled at flush
    /// time, not on append.
    fn append_atom_to_run(&mut self, executed: bool) {
        match &mut self.pending_atoms_run {
            Some(run) if run.exec == executed => {
                run.count += 1;
            }
            Some(_) => {
                // Polarity flipped -- flush previous run, start new.
                self.flush_atoms_run();
                self.pending_atoms_run = Some(AtomRun { exec: executed, count: 1 });
            }
            None => {
                self.pending_atoms_run = Some(AtomRun { exec: executed, count: 1 });
            }
        }
    }

    /// Flush the in-flight RLE run, emitting one base atom plus zero or
    /// more Repeat0/Repeat1 frames sufficient to cover the recorded
    /// count. Each Repeat0(n) / Repeat1(n) represents `n` *additional*
    /// same-polarity atoms beyond the base, so the total run length is
    /// `1 + sum(repeat counts)`.
    ///
    /// Encoding rule:
    ///   count == 1     : just the atom
    ///   count <= 16    : atom + Repeat0(count - 1)   [4-bit field, 0..15]
    ///   count <= 1024  : atom + Repeat1(count - 1)   [10-bit field, 0..1023]
    ///   count >  1024  : atom + chained Repeat1 frames (1023 each, then
    ///                    a final shorter Repeat1 for the remainder)
    fn flush_atoms_run(&mut self) {
        let Some(run) = self.pending_atoms_run.take() else {
            return;
        };
        self.encode_atom(run.exec);
        let mut remaining = run.count.saturating_sub(1);
        // Use Repeat0 only when the entire remaining count fits in 4 bits.
        // Anything larger goes through Repeat1 (possibly chained).
        if remaining > 0 && remaining <= 15 {
            self.encode_repeat0(remaining as u8);
            remaining = 0;
        }
        while remaining > 0 {
            let chunk = remaining.min(1023) as u16;
            self.encode_repeat1(chunk);
            remaining -= chunk as u32;
        }
    }

    /// Mode-2 per-cycle drain: flush the in-flight atom run, then any
    /// queued non-atom frames in the order they were recorded.
    fn drain_mode2_pending(&mut self) {
        // Flush atoms run first (atom precedes PC/LC for current cycle
        // per the spec's ordering rule).
        self.flush_atoms_run();
        for frame in std::mem::take(&mut self.pending_mode2_frames) {
            match frame {
                PendingMode2Frame::NewPc { pc } => self.encode_new_pc(pc),
                PendingMode2Frame::Lc { flag, count } => self.encode_lc(flag, count),
            }
        }
    }

    /// Push `count` bits of `value` MSB-first into the bit accumulator.
    /// Used by mode-2 frame encoders only. Triggers flush_word_if_full
    /// after each push.
    ///
    /// `count` must be <= 32. `value` must fit in `count` bits (caller
    /// invariant; debug_assert enforced).
    fn push_bits(&mut self, value: u32, count: u8) {
        debug_assert!(count <= 32);
        debug_assert!(count == 32 || value < (1u32 << count));
        for i in (0..count).rev() {
            let bit = (value >> i) & 1;
            self.pending_word = (self.pending_word << 1) | bit;
            self.pending_word_bits += 1;
            if self.pending_word_bits == 32 {
                self.flush_word_if_full();
            }
        }
    }

    /// If 32 bits are accumulated, push as 4 big-endian bytes to
    /// byte_buffer and reset the accumulator. No-op otherwise.
    fn flush_word_if_full(&mut self) {
        if self.pending_word_bits < 32 {
            return;
        }
        let w = self.pending_word;
        self.byte_buffer.push((w >> 24) as u8);
        self.byte_buffer.push((w >> 16) as u8);
        self.byte_buffer.push((w >> 8) as u8);
        self.byte_buffer.push(w as u8);
        self.pending_word = 0;
        self.pending_word_bits = 0;
        self.try_emit_packet();
    }

    /// Pad the current word with Filler0 (0010) nibbles until aligned
    /// to a 32-bit boundary. No-op if already aligned.
    ///
    /// Filler0 is a 4-bit frame, so this only works when the partial
    /// word has a multiple of 4 bits already accumulated. That
    /// invariant is upheld by all mode-2 frame encoders, which only
    /// push 4-, 8-, 16-, or 32-bit frames. Debug-asserted.
    fn align_to_word_via_filler0(&mut self) {
        if self.pending_word_bits == 0 {
            return;
        }
        debug_assert!(
            self.pending_word_bits % 4 == 0,
            "align_to_word_via_filler0 called with non-nibble-aligned bits ({}); \
             only 4-bit-multiple frames are supported in mode 2",
            self.pending_word_bits
        );
        let nibbles_remaining = (32 - self.pending_word_bits) / 4;
        for _ in 0..nibbles_remaining {
            self.push_bits(0b0010, 4);
        }
    }

    /// Emit a 32-bit "long frame" (Start, LC, Stop). Aligns to word
    /// boundary first via Filler0 padding, then pushes the whole word.
    fn emit_long_frame(&mut self, word: u32) {
        self.align_to_word_via_filler0();
        debug_assert_eq!(self.pending_word_bits, 0);
        self.push_bits(word, 32);
    }

    /// Consume 28 bytes from the buffer and emit one packet as individual words.
    fn emit_packet_from_buffer(&mut self) {
        // Word 0: packet header
        // Format from mlir-aie utils/trace/utils.py extract_tile():
        //   [4:0]   = packet_id
        //   [11:5]  = reserved (must be 0)
        //   [13:12] = packet_type
        //   [15:14] = reserved
        //   [20:16] = row
        //   [27:21] = col
        //   [30:28] = reserved (must be 0)
        //   [31]    = parity (odd parity)
        let mut header: u32 = 0;
        header |= (self.packet_id as u32) & 0x1F;
        header |= ((self.packet_type as u32) & 0x3) << 12;
        header |= ((self.row as u32) & 0x1F) << 16;
        header |= ((self.col as u32) & 0x7F) << 21;
        // Compute odd parity over bits 30:0
        let ones = header.count_ones();
        if ones % 2 == 0 {
            header |= 1 << 31; // Set parity bit to make total odd
        }
        self.pending_words.push_back((header, false));

        // Words 1-7: 28 bytes of trace data, packed big-endian into u32s
        for word_idx in 0..7 {
            let base = word_idx * 4;
            let mut word: u32 = 0;
            for byte_idx in 0..4 {
                let byte = if base + byte_idx < self.byte_buffer.len() {
                    self.byte_buffer[base + byte_idx]
                } else {
                    0xFE // Pad byte
                };
                word = (word << 8) | (byte as u32);
            }
            let tlast = word_idx == 6; // Last data word (word 7 of 8)
            self.pending_words.push_back((word, tlast));
        }

        // Remove consumed bytes (up to 28)
        let consumed = self.byte_buffer.len().min(28);
        self.byte_buffer.drain(..consumed);
    }
}
