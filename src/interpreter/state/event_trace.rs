//! Event tracing types for AIE2 profiling and trace export.
//!
//! These types record execution events that can be compared against hardware
//! traces in Perfetto format. Each `EventType` variant maps to a hardware
//! trace event code from the AIE2 tile trace units.

/// Types of events that can be recorded for profiling and trace export.
///
/// Each variant maps to a hardware trace event code from the AIE2 tile trace
/// units (Core module and Memory module). This alignment allows direct
/// comparison between emulator traces and hardware traces in Perfetto.
///
/// See AM025 Trace Event Codes and AIE2Schedule.td CoreEvent definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    // -- Instruction events (Core module trace) --
    // Per-class events matching hardware CoreEvent codes from AIE2Schedule.td.
    /// Vector instruction executed (VMAC, VADD, VCMP, etc.).
    /// Maps to hardware INSTR_VECTOR.
    InstrVector { pc: u32 },
    /// Load instruction executed.
    /// Maps to hardware INSTR_LOAD.
    InstrLoad { pc: u32 },
    /// Store instruction executed.
    /// Maps to hardware INSTR_STORE.
    InstrStore { pc: u32 },
    /// Call instruction executed (jl).
    /// Maps to hardware INSTR_CALL.
    InstrCall { pc: u32 },
    /// Return instruction executed (ret).
    /// Maps to hardware INSTR_RETURN.
    InstrReturn { pc: u32 },
    /// Lock acquire request instruction.
    /// Maps to hardware INSTR_LOCK_ACQUIRE_REQ.
    InstrLockAcquireReq { pc: u32 },
    /// Lock release request instruction.
    /// Maps to hardware INSTR_LOCK_RELEASE_REQ.
    InstrLockReleaseReq { pc: u32 },
    /// Stream get instruction.
    /// Maps to hardware INSTR_STREAM_GET.
    InstrStreamGet { pc: u32 },
    /// Stream put instruction.
    /// Maps to hardware INSTR_STREAM_PUT.
    InstrStreamPut { pc: u32 },
    /// User-defined event instruction (`event #0` or `event #1`).
    /// Maps to hardware INSTR_EVENT_0 (id=0) or INSTR_EVENT_1 (id=1).
    InstrEvent { pc: u32, id: u8 },

    // -- Stall events (Core module trace) --
    /// Memory access stall.
    /// Maps to hardware MEMORY_STALL.
    MemoryStall { cycles: u8 },
    /// Lock acquire stall.
    /// Maps to hardware LOCK_STALL.
    LockStall { cycles: u8 },
    /// Stream interface stall.
    /// Maps to hardware STREAM_STALL.
    StreamStall { cycles: u8 },

    // -- DMA events (Memory module trace) --
    // Channel encodes direction: 0-1 = S2MM (input), 2-3 = MM2S (output)
    // for compute tiles. Shim/memtile may have more channels.
    /// DMA channel started a task.
    /// Maps to hardware DMA_x_START_TASK.
    DmaStartTask { channel: u8 },
    /// DMA channel finished one buffer descriptor.
    /// Maps to hardware DMA_x_FINISHED_BD.
    DmaFinishedBd { channel: u8 },
    /// DMA channel finished an entire task (all BDs and repeats).
    /// Maps to hardware DMA_x_FINISHED_TASK.
    DmaFinishedTask { channel: u8 },
    /// DMA channel stalled waiting for a lock.
    /// Maps to hardware DMA_x_STALLED_LOCK.
    DmaStalledLock { channel: u8 },
    /// DMA channel stalled waiting for stream data.
    /// Maps to hardware DMA_x_STREAM_STARVATION.
    DmaStreamStarvation { channel: u8 },

    // -- Lock events (Memory module trace) --
    /// Lock acquired.
    /// Maps to hardware LOCK_n_ACQ.
    LockAcquire { lock_id: u8 },
    /// Lock released.
    /// Maps to hardware LOCK_n_REL.
    LockRelease { lock_id: u8 },

    // -- Core state events --
    /// Core is actively executing.
    /// Maps to hardware ACTIVE_CORE.
    CoreActive,
    /// Core has halted (done instruction).
    /// Maps to hardware DISABLED_CORE.
    CoreDisabled,

    // -- Stream port events (Core/MemTile/Shim module trace) --
    // Level events sampled every cycle per monitored port.
    /// Port had no data this cycle.
    /// Maps to hardware PORT_IDLE_0 through PORT_IDLE_7.
    PortIdle { port: u8 },
    /// Port was active (data flowing).
    /// Maps to hardware PORT_RUNNING_0 through PORT_RUNNING_7.
    PortRunning { port: u8 },
    /// Port had data but couldn't forward (backpressure).
    /// Maps to hardware PORT_STALLED_0 through PORT_STALLED_7.
    PortStalled { port: u8 },
    /// TLAST seen on this port.
    /// Maps to hardware PORT_TLAST_0 through PORT_TLAST_7.
    PortTlast { port: u8 },

    // -- Branch events (emulator-internal, no direct HW trace event) --
    /// Branch taken with source and target PCs.
    BranchTaken { from_pc: u32, to_pc: u32 },
    /// Zero-overhead-loop boundary fired this cycle. Emitted from
    /// `Context::check_hardware_loop` after LC decrement.
    LoopBoundary {
        lc_before: u32,
        lc_after: u32,
        le_pc: u32,
    },
}

/// A timestamped event for profiling.
#[derive(Debug, Clone, Copy)]
pub struct TimestampedEvent {
    /// Cycle when the event occurred.
    pub cycle: u64,
    /// The event type and details.
    pub event: EventType,
}

/// Event log for recording execution events.
#[derive(Clone)]
pub struct EventLog {
    /// Recorded events.
    events: Vec<TimestampedEvent>,
    /// Maximum events to keep (circular buffer behavior).
    max_events: usize,
    /// Whether tracing is enabled.
    enabled: bool,
}

impl EventLog {
    /// Create a new event log with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(10000)
    }

    /// Create a new event log with specified capacity.
    pub fn with_capacity(max_events: usize) -> Self {
        Self { events: Vec::with_capacity(max_events.min(1000)), max_events, enabled: false }
    }

    /// Enable event recording.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable event recording.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if recording is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record an event at the given cycle.
    #[inline]
    pub fn record(&mut self, cycle: u64, event: EventType) {
        if !self.enabled {
            return;
        }
        if self.events.len() >= self.max_events {
            // Drop oldest events (circular buffer)
            self.events.remove(0);
        }
        self.events.push(TimestampedEvent { cycle, event });
    }

    /// Get all recorded events.
    pub fn events(&self) -> &[TimestampedEvent] {
        &self.events
    }

    /// Clear all events.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Get the number of recorded events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_log_disabled_by_default() {
        let log = EventLog::new();
        assert!(!log.is_enabled());
        assert!(log.is_empty());
    }

    #[test]
    fn test_event_log_record_when_disabled() {
        let mut log = EventLog::new();

        // Recording when disabled should not add events
        log.record(10, EventType::InstrLoad { pc: 0x100 });
        assert!(log.is_empty());
    }

    #[test]
    fn test_event_log_record_when_enabled() {
        let mut log = EventLog::new();
        log.enable();

        log.record(10, EventType::InstrLoad { pc: 0x100 });
        log.record(11, EventType::InstrVector { pc: 0x100 });

        assert_eq!(log.len(), 2);
        assert_eq!(log.events()[0].cycle, 10);
        assert_eq!(log.events()[1].cycle, 11);
    }

    #[test]
    fn test_event_log_clear() {
        let mut log = EventLog::new();
        log.enable();

        log.record(1, EventType::CoreDisabled);
        log.record(2, EventType::CoreDisabled);
        assert_eq!(log.len(), 2);

        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn test_event_log_circular_buffer() {
        let mut log = EventLog::with_capacity(3);
        log.enable();

        log.record(1, EventType::InstrLoad { pc: 0x100 });
        log.record(2, EventType::InstrLoad { pc: 0x104 });
        log.record(3, EventType::InstrLoad { pc: 0x108 });

        // At capacity, next record should drop oldest
        log.record(4, EventType::InstrLoad { pc: 0x10C });

        assert_eq!(log.len(), 3);
        // First event (cycle 1) should be dropped
        assert_eq!(log.events()[0].cycle, 2);
        assert_eq!(log.events()[2].cycle, 4);
    }

    #[test]
    fn test_event_type_variants() {
        // Test that all event variants can be created and recorded.
        // Each maps to a hardware trace event code.
        let events = vec![
            // Instruction events
            EventType::InstrVector { pc: 0x100 },
            EventType::InstrLoad { pc: 0x104 },
            EventType::InstrStore { pc: 0x108 },
            EventType::InstrCall { pc: 0x10C },
            EventType::InstrReturn { pc: 0x110 },
            EventType::InstrLockAcquireReq { pc: 0x114 },
            EventType::InstrLockReleaseReq { pc: 0x118 },
            EventType::InstrStreamGet { pc: 0x11C },
            EventType::InstrStreamPut { pc: 0x120 },
            EventType::InstrEvent { pc: 0x124, id: 0 },
            EventType::InstrEvent { pc: 0x128, id: 1 },
            // Stall events
            EventType::MemoryStall { cycles: 2 },
            EventType::LockStall { cycles: 3 },
            EventType::StreamStall { cycles: 1 },
            // DMA events
            EventType::DmaStartTask { channel: 0 },
            EventType::DmaFinishedBd { channel: 1 },
            EventType::DmaFinishedTask { channel: 2 },
            EventType::DmaStalledLock { channel: 0 },
            EventType::DmaStreamStarvation { channel: 1 },
            // Lock events
            EventType::LockAcquire { lock_id: 5 },
            EventType::LockRelease { lock_id: 5 },
            // Core state
            EventType::CoreActive,
            EventType::CoreDisabled,
            // Branch (emulator-internal)
            EventType::BranchTaken { from_pc: 0x100, to_pc: 0x200 },
            // Loop boundary (emulator-internal)
            EventType::LoopBoundary { lc_before: 5, lc_after: 4, le_pc: 0x300 },
        ];

        let mut log = EventLog::new();
        log.enable();

        for (i, event) in events.into_iter().enumerate() {
            log.record(i as u64, event);
        }

        assert_eq!(log.len(), 25);
    }
}
