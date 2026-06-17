//! Unified per-channel DMA state machine.
//!
//! This module defines `ChannelFsm` and `ChannelContext`, which together
//! replace the triple-state-machine approach (ChannelState + TransferState +
//! TransferPhase) with a single FSM per channel.
//!
//! `ChannelFsm` variants carry all phase-specific state inline. `ChannelContext`
//! bundles the FSM with per-channel bookkeeping (task queue, repeat count,
//! stats) that was previously spread across 11 parallel Vec<T> arrays in
//! DmaEngine.

use std::fmt;

use super::engine::{ChannelStats, ChannelTaskConfig, TaskQueue};
#[cfg(test)]
use super::engine::TaskQueueEntry;
use super::transfer::Transfer;

/// A deferred LockRelease TRACE event awaiting its BD-slot reuse.
///
/// Tenant-4 mechanism, corrected: the memtile lock-release latency is purely an
/// OBSERVABILITY effect. The FUNCTIONAL semaphore (the lock value a waiting
/// consumer acquires) is released PROMPTLY at BD completion -- it must be, or a
/// DMA->core handoff with no buffer slack deadlocks: the consuming compute core
/// is not a DMA channel, so it can neither trigger a buffer swap nor be reached
/// by the deadlock-break flush. Deferring the functional release would wedge
/// (compute fills, defers the full-sem release awaiting its own next acquire,
/// which needs the core to free the buffer, which needs the full-sem -- a cycle
/// nothing breaks).
///
/// Only the LockRelease TRACE EVENT defers, to the BD slot's NEXT REUSE -- the
/// next acquire grant on the SAME `bd_index`. HW's LOCK_SEL1_REL reflects BD
/// RETIREMENT, which happens when the DMA cycles its descriptor ring back to
/// that slot, not at completion. For a depth-2 objectfifo the reuse is two fills
/// later, so each trace event trails FINISHED_BD by the buffer's reuse interval:
/// ~63 cyc in warmup (the latency floor dominates when the reuse is imminent),
/// ~2 fill-periods in steady state (the reuse acquire is backpressured). The
/// event is emitted at `max(ready_cycle, reuse_cycle)`, or at the `ready_cycle`
/// floor if the slot is never reused before the channel idles (the last fills
/// of a finite task).
#[derive(Debug, Clone, Copy)]
pub struct PendingRelease {
    /// The released lock (resolves to the local lock the trace event names).
    pub lock_id: u8,
    /// The BD slot (descriptor-ring index) of the fill that owns this release.
    /// The trace event defers until this slot is re-acquired (reused).
    pub bd_index: u8,
    /// Trace-event floor: BD-completion cycle plus the memtile lock-release
    /// pipeline latency (0 on non-memtile tiles). The trace event is emitted at
    /// `max(ready_cycle, reuse_cycle)`, or at `ready_cycle` if the slot is never
    /// reused (channel goes idle first).
    pub ready_cycle: u64,
    /// The cycle at which the deferred trace event should be emitted, set once
    /// the BD slot is reused (`max(ready_cycle, reuse_cycle)`) or, for a slot
    /// that is never reused, at the `ready_cycle` floor when the channel idles.
    /// `None` until then.
    pub trace_at: Option<u64>,
}

/// Information carried from a completed transfer into the lock release phase.
/// Extracted from Transfer so the Transfer can be dropped once data movement
/// is done.
#[derive(Debug, Clone)]
pub struct CompletionInfo {
    pub bd_index: u8,
    pub next_bd: Option<u8>,
    pub cycles_elapsed: u64,
    pub channel: u8,
}

/// Unified per-channel DMA state machine.
///
/// Each variant represents one phase of the DMA channel lifecycle. The
/// Transfer (when present) is boxed because it's ~200 bytes -- moving
/// between variants is a pointer swap.
///
/// The key design property: `Transferring` has NO countdown timer. It
/// checks `transfer.remaining_bytes() == 0` to know when data movement is
/// complete. This eliminates the desync that existed between the old
/// ChannelTimingState word counter and the Transfer byte counter.
#[derive(Debug)]
pub enum ChannelFsm {
    /// No active transfer.
    Idle,

    /// Loading BD configuration.
    /// Latency: DmaTimingConfig::bd_setup_cycles (default 4).
    BdSetup {
        cycles_remaining: u16,
        transfer: Box<Transfer>,
    },

    /// Waiting to acquire lock before data movement.
    /// Stalls if lock unavailable. Counts down lock_acquire_cycles once
    /// available.
    AcquiringLock {
        lock_id: u8,
        cycles_remaining: u16,
        acquired: bool,
        transfer: Box<Transfer>,
    },

    /// Memory pipeline warmup.
    /// Latency: DmaTimingConfig::memory_latency_cycles (default 5).
    MemoryLatency {
        cycles_remaining: u16,
        transfer: Box<Transfer>,
    },

    /// NoC + DDR pipeline fill latency for shim tile host memory access.
    /// Fires once per BD, between MemoryLatency and Transferring, only for
    /// shim tiles whose transfer involves host DDR memory.
    /// Latency: DmaTimingConfig::host_memory_latency_cycles (default 100).
    HostPipelineLatency {
        cycles_remaining: u16,
        transfer: Box<Transfer>,
    },

    /// BD-switch bubble: the minimum inter-BD handshake gap on the chained-BD
    /// prefetch fast path. Hardware deasserts PORT_RUNNING for ~1 cycle at each
    /// `next_bd` boundary (next_bd fetch + lock handshake) even when the next
    /// buffer is immediately available -- NPU1 add_one memtile slot0 traces
    /// `on16 off1 x4`. No beat moves during this state, so the downstream port
    /// idles, producing the gap. When a real wait (lock stall, host pipeline)
    /// is longer it dominates and this is absorbed.
    /// Latency: DmaTimingConfig::bd_switch_bubble_cycles (default 1 for AIE2).
    BdSwitchBubble {
        cycles_remaining: u16,
        transfer: Box<Transfer>,
    },

    /// Actively moving data word by word.
    /// Exits when transfer.remaining_bytes() == 0 or FoT TLAST received.
    /// S2MM stalls transparently (stays in this state, no advancement).
    Transferring { transfer: Box<Transfer> },

    /// Post-transfer pipeline-fill hold for a non-shim channel's FIRST BD of a
    /// session (#140).  The data has already moved (input drained, so no
    /// upstream stream backpressure), but the channel's downstream-visible
    /// completion -- FINISHED_BD and the functional lock release -- is held off
    /// for `startup_hold_cycles`.  This delays first-output down the ObjectFifo
    /// chain (widening the shim S2MM) WITHOUT inflating the shim MM2S, which a
    /// pre-transfer MemoryLatency stall would do by starving its input FIFO.
    /// Bounded countdown (always completes), so unlike the deferred-trace
    /// PendingRelease path it cannot deadlock a DMA->core handoff.  Fires once
    /// per channel session; subsequent BDs/tasks are warm.
    StartupHold {
        cycles_remaining: u16,
        transfer: Box<Transfer>,
    },

    /// Releasing lock after all data moved.
    /// Latency: DmaTimingConfig::lock_release_cycles (default 1).
    ReleasingLock {
        lock_id: u8,
        release_value: i8,
        cycles_remaining: u16,
        completion: CompletionInfo,
    },

    /// Transitioning between chained BDs.
    /// Latency: DmaTimingConfig::bd_chain_cycles (default 2).
    BdChaining { cycles_remaining: u16, next_bd: u8 },

    /// Channel paused by host. Resumes to saved state.
    Paused { saved: Box<ChannelFsm> },

    /// Unrecoverable error.
    Error,
}

impl Default for ChannelFsm {
    fn default() -> Self {
        ChannelFsm::Idle
    }
}

impl ChannelFsm {
    /// Short human-readable phase name for logging.
    pub fn phase_name(&self) -> &'static str {
        match self {
            ChannelFsm::Idle => "Idle",
            ChannelFsm::BdSetup { .. } => "BdSetup",
            ChannelFsm::AcquiringLock { .. } => "AcquiringLock",
            ChannelFsm::MemoryLatency { .. } => "MemoryLatency",
            ChannelFsm::HostPipelineLatency { .. } => "HostPipelineLatency",
            ChannelFsm::BdSwitchBubble { .. } => "BdSwitchBubble",
            ChannelFsm::Transferring { .. } => "Transferring",
            ChannelFsm::StartupHold { .. } => "StartupHold",
            ChannelFsm::ReleasingLock { .. } => "ReleasingLock",
            ChannelFsm::BdChaining { .. } => "BdChaining",
            ChannelFsm::Paused { .. } => "Paused",
            ChannelFsm::Error => "Error",
        }
    }

    /// Whether this channel is doing work (not idle, not terminal).
    pub fn is_active(&self) -> bool {
        !matches!(self, ChannelFsm::Idle | ChannelFsm::Error | ChannelFsm::Paused { .. })
    }

    /// Access the in-flight Transfer, if the FSM is in a phase that has one.
    pub fn transfer(&self) -> Option<&Transfer> {
        match self {
            ChannelFsm::BdSetup { transfer, .. }
            | ChannelFsm::AcquiringLock { transfer, .. }
            | ChannelFsm::MemoryLatency { transfer, .. }
            | ChannelFsm::HostPipelineLatency { transfer, .. }
            | ChannelFsm::BdSwitchBubble { transfer, .. }
            | ChannelFsm::Transferring { transfer }
            | ChannelFsm::StartupHold { transfer, .. } => Some(transfer),
            _ => None,
        }
    }

    /// Mutable access to the in-flight Transfer.
    pub fn transfer_mut(&mut self) -> Option<&mut Transfer> {
        match self {
            ChannelFsm::BdSetup { transfer, .. }
            | ChannelFsm::AcquiringLock { transfer, .. }
            | ChannelFsm::MemoryLatency { transfer, .. }
            | ChannelFsm::HostPipelineLatency { transfer, .. }
            | ChannelFsm::BdSwitchBubble { transfer, .. }
            | ChannelFsm::Transferring { transfer }
            | ChannelFsm::StartupHold { transfer, .. } => Some(transfer),
            _ => None,
        }
    }
}

impl fmt::Display for ChannelFsm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChannelFsm::Idle => write!(f, "Idle"),
            ChannelFsm::BdSetup { cycles_remaining, .. } => write!(f, "BdSetup(cycles={})", cycles_remaining),
            ChannelFsm::AcquiringLock { lock_id, acquired, .. } => {
                write!(f, "AcquiringLock(lock={}, acquired={})", lock_id, acquired)
            }
            ChannelFsm::MemoryLatency { cycles_remaining, .. } => {
                write!(f, "MemoryLatency(cycles={})", cycles_remaining)
            }
            ChannelFsm::HostPipelineLatency { cycles_remaining, .. } => {
                write!(f, "HostPipelineLatency(cycles={})", cycles_remaining)
            }
            ChannelFsm::BdSwitchBubble { cycles_remaining, .. } => {
                write!(f, "BdSwitchBubble(cycles={})", cycles_remaining)
            }
            ChannelFsm::Transferring { transfer } => {
                write!(f, "Transferring({} bytes remaining)", transfer.remaining_bytes())
            }
            ChannelFsm::StartupHold { cycles_remaining, .. } => {
                write!(f, "StartupHold(cycles={})", cycles_remaining)
            }
            ChannelFsm::ReleasingLock { lock_id, cycles_remaining, .. } => {
                write!(f, "ReleasingLock(lock={}, cycles={})", lock_id, cycles_remaining)
            }
            ChannelFsm::BdChaining { cycles_remaining, next_bd } => {
                write!(f, "BdChaining(next_bd={}, cycles={})", next_bd, cycles_remaining)
            }
            ChannelFsm::Paused { saved } => write!(f, "Paused(was={})", saved.phase_name()),
            ChannelFsm::Error => write!(f, "Error"),
        }
    }
}

/// All per-channel state in one struct.
///
/// Replaces the 11 parallel Vec<T> arrays that DmaEngine previously used
/// for per-channel state.
#[derive(Debug)]
pub struct ChannelContext {
    /// The unified state machine.
    pub fsm: ChannelFsm,

    /// Which channel index this is (0-based).
    pub index: u8,

    /// Currently active BD index.
    pub current_bd: Option<u8>,

    /// First BD in a chain (for repeat restart).
    pub chain_start_bd: Option<u8>,

    /// Next BD to load after current transfer completes (set by chaining).
    pub queued_bd: Option<u8>,

    /// Repeat count for current task.
    pub repeat_count: u32,

    /// Task queue (8-deep FIFO per AM025).
    pub task_queue: TaskQueue,

    /// Per-task configuration (token issue, FoT mode, compression).
    pub task_config: ChannelTaskConfig,

    /// BD unavailable error flag (out-of-order mode).
    pub error_bd_unavailable: bool,

    /// First-BD-of-task gate: true while the channel is waiting to do its
    /// first data movement out of cold idle. Consumed (cleared to false) on
    /// the first transition into MemoryLatency, where it triggers per-task
    /// timing bonuses (channel_start_cycles always, plus the per-direction
    /// `shim_per_task_overhead_{mm2s,s2mm}_cycles` for shim+host transfers).
    /// Reset to true on Idle re-entry or stop_channel -- so fires once per
    /// task dispatch (Idle->BdSetup transition).
    pub is_first_bd: bool,

    /// Index of the next task within this channel session, used to charge
    /// the geometrically-decaying shim warm-up transient
    /// (`cold_start * (decay/1000)^warm_task_index`).  Increments once per
    /// task (per first-BD bonus on a shim+host transfer); reset to 0 on
    /// stop_channel (channel reset == fresh boot).  Phase 2d.
    pub warm_task_index: u32,

    /// BD-prefetch gate (Phase 2d.2): true once the channel has emitted the
    /// START_TASK event for the *next* queued task ahead of time, while still
    /// transferring the current one (HW loads the next BD into its second slot
    /// during the current transfer).  start_channel suppresses the duplicate
    /// START when that task actually begins, then clears this.  Reset on
    /// channel stop/reset.  Lets START[i+1] precede FINISHED[i].
    pub prefetch_start_emitted: bool,

    /// Controller dispatch index (Phase 2d.2 Part 2): count of task
    /// dispatches (Task_Queue writes) issued to this channel since the
    /// last channel reset.  Feeds the controller's occupancy-dependent
    /// dispatch gate (`CycleCostModel::dispatch_overhead_for`).  Unlike
    /// instantaneous queue occupancy this is *monotonic per session*, so
    /// the gate ramps up to its plateau and stays there instead of
    /// collapsing back to the base rate when a short task drains the
    /// channel between dispatches.  Increments once per `enqueue_task`;
    /// reset to 0 on stop_channel (channel reset == fresh boot), like
    /// `warm_task_index`.
    pub controller_dispatch_index: u32,

    /// Performance counters.
    pub stats: ChannelStats,

    /// Edge-trigger memory for DMA_S2MM_STREAM_STARVATION events.
    /// HW fires the starvation event on the rising edge of the stall
    /// signal -- emitting every stalled cycle would flood the trace BD
    /// in milliseconds. We only emit when `prev_starving=false` and
    /// the channel stalls this cycle. Reset to false on Idle re-entry.
    pub prev_starving: bool,

    /// Edge-trigger memory for DMA_STALLED_LOCK_ACQUIRE events.
    /// Same rationale as `prev_starving` -- fires once per
    /// idle->blocked transition, not every cycle a lock acquire is
    /// pending.
    pub prev_lock_stalled: bool,

    /// Deferred cross-lock release, held until the next chained BD's acquire
    /// is granted (the buffer swap).  `(lock_id, release_value)`.
    ///
    /// Tenant-4 mechanism (HW-validated on NPU1): on a memtile shared-link
    /// producer/consumer, a completing BD's lock RELEASE does not fire at BD
    /// completion -- the DMA controller holds it until it swaps to the next
    /// buffer (its next chained BD's AcquireGE is granted), so release and
    /// next-acquire couple at the chain transition, gated on buffer
    /// availability.  Only CROSS-lock handoff defers (next BD acquires a
    /// different lock than the one released -- a producer->consumer signal);
    /// same-lock self-chains release inline.  Applied at the acquire grant
    /// (the owning channel's swap) or, when two channels are mutually blocked
    /// holding each other's buffer, flushed by the blocked peer's
    /// deadlock-break scan.  See `begin_completion` / the AcquiringLock arm.
    ///
    /// A FIFO of in-flight releases: with the memtile release latency a fast
    /// producer can fill several buffers before the earliest release lands, so
    /// each completion enqueues rather than overwrites.
    pub pending_releases: Vec<PendingRelease>,

    /// Swap-enable watch for the end-of-stream release tail: `(acquire_lock_id,
    /// last_value)` of the producer's acquire lock (FREE) while the channel is
    /// stream-stalled. HW fires a deferred full-release on the SWAP-enable -- the
    /// next buffer becoming FREE (a consumer-free event) -- not on the producer's
    /// actual re-acquire. When stalled, an increment of this lock (the consumer
    /// freeing a buffer) emits the oldest still-pending release, so a trailing
    /// fill whose slot is never re-acquired still lands its trace. `None` when not
    /// stream-stalled (re-baselined on each stall). See `service_pending_releases`.
    pub swap_free_watch: Option<(u8, i8)>,

    /// Pending post-transfer startup hold (#140), in cycles.  Latched once per
    /// channel session by `consume_first_bd_bonus` on the first BD of a
    /// non-shim channel (= the tile-kind `*_first_bd_startup_cycles`), and
    /// consumed at that BD's completion: the channel enters `StartupHold` for
    /// this many cycles before its release/FINISHED_BD fire.  Cleared when the
    /// hold fires; reset on stop/reset.  0 = no hold (default, and all warm
    /// BDs/tasks past the first).
    pub startup_hold_cycles: u16,
}

impl ChannelContext {
    pub fn new(index: u8) -> Self {
        Self {
            fsm: ChannelFsm::Idle,
            index,
            current_bd: None,
            chain_start_bd: None,
            queued_bd: None,
            repeat_count: 0,
            task_queue: TaskQueue::new_default(),
            task_config: ChannelTaskConfig::default(),
            error_bd_unavailable: false,
            is_first_bd: true,
            warm_task_index: 0,
            prefetch_start_emitted: false,
            controller_dispatch_index: 0,
            stats: ChannelStats::default(),
            prev_starving: false,
            prev_lock_stalled: false,
            pending_releases: Vec::new(),
            swap_free_watch: None,
            startup_hold_cycles: 0,
        }
    }

    /// Derive the public ChannelState from the FSM.
    ///
    /// Maps FSM variants to the current ChannelState enum, which includes
    /// WaitingForLock and Complete variants. AcquiringLock maps to
    /// WaitingForLock; all other active phases map to Active.
    pub fn state(&self) -> super::ChannelState {
        match &self.fsm {
            ChannelFsm::Idle => super::ChannelState::Idle,
            ChannelFsm::AcquiringLock { lock_id, acquired, .. } => {
                if *acquired {
                    super::ChannelState::Active
                } else {
                    super::ChannelState::WaitingForLock(*lock_id)
                }
            }
            ChannelFsm::Paused { .. } => super::ChannelState::Paused,
            ChannelFsm::Error => super::ChannelState::Error,
            _ => super::ChannelState::Active,
        }
    }

    /// Detailed FSM state description for diagnostics.
    ///
    /// Unlike `state()` which collapses most variants to Active, this returns
    /// the exact FSM variant name so stall diagnostics can distinguish
    /// Transferring from ReleasingLock from BdChaining etc.
    pub fn fsm_description(&self) -> String {
        match &self.fsm {
            ChannelFsm::Idle => "Idle".to_string(),
            ChannelFsm::BdSetup { cycles_remaining, .. } => {
                format!("BdSetup({})", cycles_remaining)
            }
            ChannelFsm::AcquiringLock { lock_id, acquired, .. } => {
                if *acquired {
                    format!("AcquiringLock({}, acquired)", lock_id)
                } else {
                    format!("AcquiringLock({})", lock_id)
                }
            }
            ChannelFsm::BdSwitchBubble { cycles_remaining, .. } => {
                format!("BdSwitchBubble({})", cycles_remaining)
            }
            ChannelFsm::MemoryLatency { cycles_remaining, .. } => {
                format!("MemoryLatency({})", cycles_remaining)
            }
            ChannelFsm::HostPipelineLatency { cycles_remaining, .. } => {
                format!("HostPipelineLatency({})", cycles_remaining)
            }
            ChannelFsm::Transferring { transfer } => {
                format!("Transferring({}/{})", transfer.bytes_transferred, transfer.total_bytes)
            }
            ChannelFsm::StartupHold { cycles_remaining, .. } => {
                format!("StartupHold({}cyc)", cycles_remaining)
            }
            ChannelFsm::ReleasingLock { lock_id, release_value, cycles_remaining, .. } => {
                format!("ReleasingLock({}, delta={}, {}cyc)", lock_id, release_value, cycles_remaining)
            }
            ChannelFsm::BdChaining { cycles_remaining, next_bd } => {
                format!("BdChaining(bd={}, {}cyc)", next_bd, cycles_remaining)
            }
            ChannelFsm::Paused { .. } => "Paused".to_string(),
            ChannelFsm::Error => "Error".to_string(),
        }
    }

    /// Whether this channel has active work.
    pub fn is_active(&self) -> bool {
        self.fsm.is_active()
    }

    /// Whether this channel has any pending work (active or queued).
    pub fn has_pending_work(&self) -> bool {
        self.fsm.is_active() || self.queued_bd.is_some() || !self.task_queue.is_empty()
    }

    /// Access the in-flight Transfer, if any.
    pub fn transfer(&self) -> Option<&Transfer> {
        self.fsm.transfer()
    }

    /// One-call debug dump.
    pub fn debug_string(&self, col: u8, row: u8) -> String {
        format!(
            "DMA({},{}) ch{}: fsm={}, bd={:?}, repeat={}, queue={}",
            col,
            row,
            self.index,
            self.fsm,
            self.current_bd,
            self.repeat_count,
            self.task_queue.len()
        )
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.fsm = ChannelFsm::Idle;
        self.current_bd = None;
        self.chain_start_bd = None;
        self.queued_bd = None;
        self.repeat_count = 0;
        self.task_queue.reset();
        self.task_config = ChannelTaskConfig::default();
        self.error_bd_unavailable = false;
        self.is_first_bd = true;
        self.warm_task_index = 0;
        self.prefetch_start_emitted = false;
        self.controller_dispatch_index = 0;
        self.stats = ChannelStats::default();
        self.pending_releases.clear();
        self.swap_free_watch = None;
        self.startup_hold_cycles = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsm_default_is_idle() {
        let fsm = ChannelFsm::default();
        assert!(matches!(fsm, ChannelFsm::Idle));
        assert_eq!(fsm.phase_name(), "Idle");
    }

    #[test]
    fn test_fsm_phase_names() {
        assert_eq!(ChannelFsm::Idle.phase_name(), "Idle");
        assert_eq!(ChannelFsm::Error.phase_name(), "Error");
        let fsm = ChannelFsm::BdChaining { cycles_remaining: 2, next_bd: 3 };
        assert_eq!(fsm.phase_name(), "BdChaining");
    }

    #[test]
    fn test_fsm_is_active() {
        assert!(!ChannelFsm::Idle.is_active());
        assert!(!ChannelFsm::Error.is_active());
        let paused = ChannelFsm::Paused { saved: Box::new(ChannelFsm::Idle) };
        assert!(!paused.is_active());

        let chaining = ChannelFsm::BdChaining { cycles_remaining: 1, next_bd: 0 };
        assert!(chaining.is_active());
    }

    #[test]
    fn test_fsm_transfer_accessor_returns_none_for_non_transfer_states() {
        assert!(ChannelFsm::Idle.transfer().is_none());
        assert!(ChannelFsm::Error.transfer().is_none());
        let chaining = ChannelFsm::BdChaining { cycles_remaining: 1, next_bd: 0 };
        assert!(chaining.transfer().is_none());
        let releasing = ChannelFsm::ReleasingLock {
            lock_id: 0,
            release_value: 1,
            cycles_remaining: 1,
            completion: CompletionInfo { bd_index: 0, next_bd: None, cycles_elapsed: 0, channel: 0 },
        };
        assert!(releasing.transfer().is_none());
    }

    #[test]
    fn test_fsm_display_idle() {
        assert_eq!(format!("{}", ChannelFsm::Idle), "Idle");
    }

    #[test]
    fn test_fsm_display_error() {
        assert_eq!(format!("{}", ChannelFsm::Error), "Error");
    }

    #[test]
    fn test_fsm_display_bd_chaining() {
        let fsm = ChannelFsm::BdChaining { cycles_remaining: 2, next_bd: 5 };
        assert_eq!(format!("{}", fsm), "BdChaining(next_bd=5, cycles=2)");
    }

    #[test]
    fn test_fsm_display_releasing_lock() {
        let fsm = ChannelFsm::ReleasingLock {
            lock_id: 3,
            release_value: -1,
            cycles_remaining: 1,
            completion: CompletionInfo { bd_index: 0, next_bd: None, cycles_elapsed: 100, channel: 0 },
        };
        assert_eq!(format!("{}", fsm), "ReleasingLock(lock=3, cycles=1)");
    }

    #[test]
    fn test_fsm_display_paused() {
        let inner = ChannelFsm::BdChaining { cycles_remaining: 1, next_bd: 0 };
        let fsm = ChannelFsm::Paused { saved: Box::new(inner) };
        assert_eq!(format!("{}", fsm), "Paused(was=BdChaining)");
    }

    #[test]
    fn test_channel_context_new() {
        let ctx = ChannelContext::new(2);
        assert_eq!(ctx.index, 2);
        assert!(matches!(ctx.fsm, ChannelFsm::Idle));
        assert!(ctx.current_bd.is_none());
        assert!(ctx.chain_start_bd.is_none());
        assert!(ctx.queued_bd.is_none());
        assert_eq!(ctx.repeat_count, 0);
        assert!(ctx.task_queue.is_empty());
        assert!(!ctx.task_queue.has_overflow());
        assert!(!ctx.error_bd_unavailable);
    }

    #[test]
    fn test_channel_context_state_idle() {
        let ctx = ChannelContext::new(0);
        assert!(matches!(ctx.state(), super::super::ChannelState::Idle));
    }

    #[test]
    fn test_channel_context_state_error() {
        let mut ctx = ChannelContext::new(0);
        ctx.fsm = ChannelFsm::Error;
        assert!(matches!(ctx.state(), super::super::ChannelState::Error));
    }

    #[test]
    fn test_channel_context_state_paused() {
        let mut ctx = ChannelContext::new(0);
        ctx.fsm = ChannelFsm::Paused { saved: Box::new(ChannelFsm::Idle) };
        assert!(matches!(ctx.state(), super::super::ChannelState::Paused));
    }

    #[test]
    fn test_channel_context_state_active_for_chaining() {
        let mut ctx = ChannelContext::new(0);
        ctx.fsm = ChannelFsm::BdChaining { cycles_remaining: 1, next_bd: 0 };
        assert!(matches!(ctx.state(), super::super::ChannelState::Active));
    }

    #[test]
    fn test_channel_context_is_active() {
        let mut ctx = ChannelContext::new(0);
        assert!(!ctx.is_active());
        ctx.fsm = ChannelFsm::BdChaining { cycles_remaining: 1, next_bd: 0 };
        assert!(ctx.is_active());
    }

    #[test]
    fn test_channel_context_has_pending_work() {
        let mut ctx = ChannelContext::new(0);
        assert!(!ctx.has_pending_work());

        ctx.queued_bd = Some(3);
        assert!(ctx.has_pending_work());

        ctx.queued_bd = None;
        let _ = ctx.task_queue.push(TaskQueueEntry::new(0, 0, false));
        assert!(ctx.has_pending_work());
    }

    #[test]
    fn test_channel_context_reset() {
        let mut ctx = ChannelContext::new(1);
        ctx.fsm = ChannelFsm::Error;
        ctx.current_bd = Some(5);
        ctx.chain_start_bd = Some(5);
        ctx.queued_bd = Some(6);
        ctx.repeat_count = 10;
        // Trigger overflow by filling the queue
        for _ in 0..9 {
            let _ = ctx.task_queue.push(TaskQueueEntry::new(0, 0, false));
        }
        ctx.error_bd_unavailable = true;
        ctx.stats.bytes_transferred = 1024;

        ctx.reset();

        assert!(matches!(ctx.fsm, ChannelFsm::Idle));
        assert!(ctx.current_bd.is_none());
        assert!(ctx.chain_start_bd.is_none());
        assert!(ctx.queued_bd.is_none());
        assert_eq!(ctx.repeat_count, 0);
        assert!(ctx.task_queue.is_empty());
        assert!(!ctx.task_queue.has_overflow());
        assert!(!ctx.error_bd_unavailable);
        assert_eq!(ctx.stats.bytes_transferred, 0);
    }

    #[test]
    fn test_channel_context_debug_string() {
        let ctx = ChannelContext::new(0);
        let s = ctx.debug_string(1, 2);
        assert!(s.contains("DMA(1,2) ch0"));
        assert!(s.contains("Idle"));
    }
}
