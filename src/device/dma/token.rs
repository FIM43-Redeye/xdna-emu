//! DMA task queue and token-based synchronization.
//!
//! This module implements the AIE2 DMA task queue and completion token
//! mechanism. Each DMA channel has a hardware task queue (4 entries deep
//! per aie-rt `StartQSizeMax`) that allows multiple transfers to be enqueued without
//! software intervention between tasks. When a task completes and token
//! issue is enabled, the channel produces a completion token that software
//! can poll or use to trigger events.
//!
//! # Hardware behavior (derived from aie-rt and AM025 register database)
//!
//! ## Task Queue Register Layout
//!
//! The task queue is written via the Start_Queue register (write-only).
//! Each write pushes one entry into the channel's hardware FIFO:
//!
//! | Field            | Bits   | Description                              |
//! |------------------|--------|------------------------------------------|
//! | Enable_Token_Issue | [31]   | Issue token on task completion            |
//! | Repeat_Count     | [23:16]| Repeat count (actual - 1); range [1:256] |
//! | Start_BD_ID      | [3:0] or [5:0] | First BD for this task (in-order only) |
//!
//! The Start_BD_ID width varies by tile type:
//! - Compute/Shim tiles: 4 bits [3:0] (16 BDs max)
//! - MemTile: 6 bits [5:0] (48 BDs max, though field says "BD ID [0-23] only")
//!
//! ## Status Register Layout
//!
//! The channel status register (read-only) reports queue state:
//!
//! | Field              | Bits    | Description                          |
//! |--------------------|---------|--------------------------------------|
//! | Current_BD         | [27:24] | BD the channel is currently executing |
//! | Task_Queue_Size    | [22:20] | Number of tasks currently in queue   |
//! | Channel_Running    | [19]    | Channel active or queue non-empty    |
//! | Task_Queue_Overflow| [18]    | Write-to-clear sticky overflow flag  |
//!
//! ## Token Semantics
//!
//! When a task completes and Enable_Token_Issue was set in its queue entry,
//! the DMA emits a TaskCompleteToken carrying the channel's Controller_ID
//! (configured in the channel control register). The token controller ID
//! allows software to distinguish completion sources when multiple channels
//! share a token consumer.
//!
//! aie-rt reference: `XAie_DmaChannelSetStartQueueGeneric()` in
//! `xaie_dma.c` builds the register value as:
//! ```text
//! Val = StartBd | ((RepeatCount - 1) << 16) | (EnTokenIssue << 31)
//! ```
//! Note the -1 encoding: hardware stores `actual_count - 1`, so a register
//! value of 0 means "execute once".
//!
//! ## Out-of-Order Mode
//!
//! When out-of-order execution is enabled on a channel, the Start_BD_ID
//! field in the queue entry is ignored (set to 0 by aie-rt). Instead, BDs
//! are dispatched based on their Out_Of_Order_BD_ID field and availability.

use std::collections::VecDeque;

// ============================================================================
// Task queue register bit positions
//
// Derived from AM025 register database (aie_registers_aie2.json).
// These are the same across compute, memtile, and shim tile types.
// Only Start_BD_ID width varies (4-bit vs 6-bit).
// ============================================================================

/// Bit position of Enable_Token_Issue in the Start_Queue register.
/// Per AM025: bit [31] across all tile types.
pub const ENABLE_TOKEN_ISSUE_BIT: u32 = 31;

/// LSB of the Repeat_Count field in the Start_Queue register.
/// Per AM025: bits [23:16] across all tile types.
pub const REPEAT_COUNT_LSB: u32 = 16;

/// Width of the Repeat_Count field (8 bits -> range 0-255, representing 1-256).
pub const REPEAT_COUNT_WIDTH: u32 = 8;

/// Mask for the Repeat_Count field.
pub const REPEAT_COUNT_MASK: u32 = ((1 << REPEAT_COUNT_WIDTH) - 1) << REPEAT_COUNT_LSB;

/// LSB of Start_BD_ID in the Start_Queue register (always bit 0).
pub const START_BD_ID_LSB: u32 = 0;

/// Width of Start_BD_ID for compute and shim tiles (4 bits -> BD 0-15).
pub const START_BD_ID_WIDTH_COMPUTE: u32 = 4;

/// Width of Start_BD_ID for memory tiles (6 bits -> BD 0-47).
pub const START_BD_ID_WIDTH_MEMTILE: u32 = 6;

/// Maximum task queue depth per channel.
///
/// The hardware start-queue holds 4 tasks.  aie-rt declares this as
/// `XAIE_DMA_MAX_QUEUE_SIZE 4U` (`xaie_dma.c:45`, returned by
/// `XAie_DmaGetMaxQueueSize`) and as the per-tile-type channel property
/// `StartQSizeMax = 4U` -- uniform across compute, memtile, and shim/NoC
/// modules and across every generation (AIE1, AIE-ML/AIE2, AIE2PS).
/// `_XAieMlDmaSetStartQueue` (`xaie_dma_aieml.c:1125`) rejects a task whose
/// `TaskQSize > StartQSizeMax`.
///
/// The `Task_Queue_Size` status field is 3 bits wide -- that is the register
/// *encoding* width (it can express 0-7), not the queue's actual capacity.
/// The capacity aie-rt enforces is 4, so a 5th outstanding task overflows.
///
/// AIE2+ only: AIE1 has no task queue mechanism.  Consult
/// `DmaModel::supports_task_queue()` before using.
pub const MAX_TASK_QUEUE_DEPTH: usize = 4;

/// Maximum repeat count (8-bit field encodes actual-1, so max actual is 256).
pub const MAX_REPEAT_COUNT: u32 = 256;

// ============================================================================
// Task queue entry
// ============================================================================

/// A single entry in the DMA channel task queue.
///
/// Represents one task that was pushed by writing to the Start_Queue register.
/// The hardware FIFO processes these in order (FIFO).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskQueueEntry {
    /// Starting BD index for this task.
    ///
    /// In-order mode: the first BD to execute (subsequent BDs follow the
    /// next_bd chain). In out-of-order mode: ignored (aie-rt sets to 0).
    pub start_bd: u8,

    /// Repeat count in hardware encoding (actual - 1).
    ///
    /// 0 = execute once, 255 = execute 256 times.
    /// This is the raw register value, not the actual count.
    pub repeat_count: u8,

    /// Whether to issue a completion token when this task finishes.
    pub enable_token_issue: bool,
}

impl TaskQueueEntry {
    /// Create a new task queue entry.
    ///
    /// `repeat_count` is in hardware encoding (actual - 1).
    pub fn new(start_bd: u8, repeat_count: u8, enable_token_issue: bool) -> Self {
        Self { start_bd, repeat_count, enable_token_issue }
    }

    /// Get the actual number of times this task will execute.
    ///
    /// The hardware encodes repeat as (actual - 1), so this returns
    /// `repeat_count + 1`. Range: [1, 256].
    pub fn actual_repeat_count(&self) -> u32 {
        self.repeat_count as u32 + 1
    }

    /// Decode a task queue entry from a Start_Queue register write value.
    ///
    /// The `bd_id_width` parameter controls how many bits are used for
    /// Start_BD_ID (4 for compute/shim, 6 for memtile).
    ///
    /// Register layout (from AM025):
    /// - [31]     Enable_Token_Issue
    /// - [23:16]  Repeat_Count (actual - 1)
    /// - [N:0]    Start_BD_ID (N = bd_id_width - 1)
    pub fn from_register(value: u32, bd_id_width: u32) -> Self {
        let bd_mask = (1u32 << bd_id_width) - 1;
        Self {
            start_bd: (value & bd_mask) as u8,
            repeat_count: ((value & REPEAT_COUNT_MASK) >> REPEAT_COUNT_LSB) as u8,
            enable_token_issue: (value >> ENABLE_TOKEN_ISSUE_BIT) & 1 != 0,
        }
    }

    /// Encode this entry as a Start_Queue register value.
    ///
    /// This is the inverse of `from_register`. Produces the 32-bit value
    /// that would be written to the Start_Queue register.
    pub fn to_register(&self) -> u32 {
        let mut val = self.start_bd as u32;
        val |= (self.repeat_count as u32) << REPEAT_COUNT_LSB;
        if self.enable_token_issue {
            val |= 1 << ENABLE_TOKEN_ISSUE_BIT;
        }
        val
    }
}

// ============================================================================
// Task queue
// ============================================================================

/// Error returned when attempting to push to a full task queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueFull;

impl std::fmt::Display for QueueFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DMA task queue is full ({} entries)", MAX_TASK_QUEUE_DEPTH)
    }
}

impl std::error::Error for QueueFull {}

/// DMA channel task queue.
///
/// A fixed-capacity FIFO that buffers pending tasks for a DMA channel.
/// The hardware queue is 4 entries deep (aie-rt `StartQSizeMax`).
///
/// Tasks are pushed by writing to the Start_Queue register and consumed
/// by the DMA engine as the channel becomes idle.
#[derive(Debug, Clone)]
pub struct TaskQueue {
    entries: VecDeque<TaskQueueEntry>,
    capacity: usize,
    /// Sticky overflow flag, set when a push is rejected due to full queue.
    /// Corresponds to AM025 Task_Queue_Overflow status bit. Write-to-clear.
    overflow: bool,
}

impl TaskQueue {
    /// Create a new task queue with the given capacity.
    ///
    /// The hardware capacity is `MAX_TASK_QUEUE_DEPTH` (4), but this
    /// constructor accepts an arbitrary capacity for testing flexibility.
    pub fn new(capacity: usize) -> Self {
        Self { entries: VecDeque::with_capacity(capacity), capacity, overflow: false }
    }

    /// Create a task queue with the standard hardware capacity (4 entries).
    pub fn new_default() -> Self {
        Self::new(MAX_TASK_QUEUE_DEPTH)
    }

    /// Push a task entry onto the queue.
    ///
    /// Returns `Err(QueueFull)` and sets the overflow flag if the queue
    /// is already at capacity. The overflow flag is sticky (persists until
    /// explicitly cleared), matching the AM025 Task_Queue_Overflow behavior.
    pub fn push(&mut self, entry: TaskQueueEntry) -> Result<(), QueueFull> {
        if self.entries.len() >= self.capacity {
            self.overflow = true;
            return Err(QueueFull);
        }
        self.entries.push_back(entry);
        Ok(())
    }

    /// Pop the oldest task entry from the queue.
    ///
    /// Returns `None` if the queue is empty.
    pub fn pop(&mut self) -> Option<TaskQueueEntry> {
        self.entries.pop_front()
    }

    /// Check whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check whether the queue is full.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    /// Get the current number of entries in the queue.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Get the queue capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Decode a Start_Queue register write and push the resulting entry.
    ///
    /// The `bd_id_width` parameter controls Start_BD_ID decoding (4 or 6).
    /// Returns `true` if the entry was enqueued, `false` if the queue was
    /// full (and the overflow flag is set).
    pub fn write_register(&mut self, value: u32, bd_id_width: u32) -> bool {
        let entry = TaskQueueEntry::from_register(value, bd_id_width);
        self.push(entry).is_ok()
    }

    /// Check if the overflow flag is set.
    ///
    /// The overflow flag is a sticky bit that is set whenever a push
    /// is rejected. It persists until cleared with `clear_overflow()`.
    pub fn has_overflow(&self) -> bool {
        self.overflow
    }

    /// Clear the overflow flag (write-to-clear semantics per AM025).
    pub fn clear_overflow(&mut self) {
        self.overflow = false;
    }

    /// Clear all entries and reset overflow flag.
    pub fn reset(&mut self) {
        self.entries.clear();
        self.overflow = false;
    }
}

// ============================================================================
// Completion token
// ============================================================================

/// A DMA task completion token.
///
/// Emitted when a task completes and Enable_Token_Issue was set in its
/// task queue entry. The token carries the channel index and controller
/// ID so the consumer can identify the completion source.
///
/// The controller_id comes from the channel's Ctrl register (Controller_ID
/// field, bits [15:8]), NOT from the task queue entry. This allows software
/// to set a per-channel identifier that is attached to all tokens from
/// that channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    /// DMA channel index that completed the task.
    pub channel_id: u8,
    /// Controller ID from the channel's Ctrl register.
    /// Used by software to distinguish token sources.
    pub controller_id: u8,
}

impl Token {
    /// Create a new completion token.
    pub fn new(channel_id: u8, controller_id: u8) -> Self {
        Self { channel_id, controller_id }
    }
}

/// Manages pending DMA completion tokens.
///
/// Tokens are issued when tasks complete with Enable_Token_Issue set,
/// and consumed by the host or by event generation logic. The token
/// buffer is unbounded in the emulator (hardware has backpressure that
/// stalls the channel -- see AM025 "Channel stalled due to task complete
/// token backpressure").
#[derive(Debug, Clone)]
pub struct TokenState {
    /// Tokens issued but not yet consumed, in issue order.
    pending: VecDeque<Token>,
}

impl TokenState {
    /// Create a new token state with no pending tokens.
    pub fn new() -> Self {
        Self { pending: VecDeque::new() }
    }

    /// Issue a completion token.
    ///
    /// Called when a task completes and its Enable_Token_Issue flag was set.
    pub fn issue(&mut self, channel_id: u8, controller_id: u8) {
        self.pending.push_back(Token::new(channel_id, controller_id));
    }

    /// Consume the oldest pending token.
    ///
    /// Returns `None` if no tokens are pending.
    pub fn consume(&mut self) -> Option<Token> {
        self.pending.pop_front()
    }

    /// Get the number of pending (unconsumed) tokens.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Check if there are any pending tokens.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Read the token status as a register value.
    ///
    /// Returns a 32-bit value with the pending count in the low bits.
    /// This is a simplified view; the actual hardware status is spread
    /// across per-channel status registers (the DMA_Task_Token_Stall
    /// event indicates backpressure). Here we report the global pending
    /// count for diagnostic purposes.
    pub fn read_status(&self) -> u32 {
        self.pending.len() as u32
    }

    /// Clear all pending tokens.
    pub fn reset(&mut self) {
        self.pending.clear();
    }
}

impl Default for TokenState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // === TaskQueueEntry tests ===

    #[test]
    fn entry_new_basic() {
        let entry = TaskQueueEntry::new(3, 0, false);
        assert_eq!(entry.start_bd, 3);
        assert_eq!(entry.repeat_count, 0);
        assert!(!entry.enable_token_issue);
    }

    #[test]
    fn entry_actual_repeat_count() {
        // Hardware encoding: 0 = run once, 255 = run 256 times
        assert_eq!(TaskQueueEntry::new(0, 0, false).actual_repeat_count(), 1);
        assert_eq!(TaskQueueEntry::new(0, 1, false).actual_repeat_count(), 2);
        assert_eq!(TaskQueueEntry::new(0, 127, false).actual_repeat_count(), 128);
        assert_eq!(TaskQueueEntry::new(0, 255, false).actual_repeat_count(), 256);
    }

    #[test]
    fn entry_from_register_compute_tile() {
        // BD=5, repeat=3 (actual 4), token enabled
        // Bits: [31]=1, [23:16]=3, [3:0]=5
        let val = (1 << 31) | (3 << 16) | 5;
        let entry = TaskQueueEntry::from_register(val, START_BD_ID_WIDTH_COMPUTE);
        assert_eq!(entry.start_bd, 5);
        assert_eq!(entry.repeat_count, 3);
        assert!(entry.enable_token_issue);
        assert_eq!(entry.actual_repeat_count(), 4);
    }

    #[test]
    fn entry_from_register_memtile() {
        // BD=35, repeat=0 (actual 1), no token
        // Bits: [31]=0, [23:16]=0, [5:0]=35
        let val = 35u32;
        let entry = TaskQueueEntry::from_register(val, START_BD_ID_WIDTH_MEMTILE);
        assert_eq!(entry.start_bd, 35);
        assert_eq!(entry.repeat_count, 0);
        assert!(!entry.enable_token_issue);
        assert_eq!(entry.actual_repeat_count(), 1);
    }

    #[test]
    fn entry_from_register_max_repeat() {
        // Max repeat: 255 in register = 256 actual
        let val = 255u32 << 16;
        let entry = TaskQueueEntry::from_register(val, START_BD_ID_WIDTH_COMPUTE);
        assert_eq!(entry.repeat_count, 255);
        assert_eq!(entry.actual_repeat_count(), 256);
    }

    #[test]
    fn entry_to_register_roundtrip() {
        let original = TaskQueueEntry::new(7, 42, true);
        let reg_val = original.to_register();
        let decoded = TaskQueueEntry::from_register(reg_val, START_BD_ID_WIDTH_COMPUTE);
        assert_eq!(decoded, original);
    }

    #[test]
    fn entry_to_register_roundtrip_memtile() {
        let original = TaskQueueEntry::new(23, 100, false);
        let reg_val = original.to_register();
        let decoded = TaskQueueEntry::from_register(reg_val, START_BD_ID_WIDTH_MEMTILE);
        assert_eq!(decoded, original);
    }

    #[test]
    fn entry_to_register_no_token() {
        let entry = TaskQueueEntry::new(0, 0, false);
        assert_eq!(entry.to_register(), 0);
    }

    #[test]
    fn entry_to_register_token_only() {
        let entry = TaskQueueEntry::new(0, 0, true);
        assert_eq!(entry.to_register(), 1 << 31);
    }

    // === TaskQueue tests ===

    #[test]
    fn queue_new_empty() {
        let q = TaskQueue::new(4);
        assert!(q.is_empty());
        assert!(!q.is_full());
        assert_eq!(q.len(), 0);
        assert_eq!(q.capacity(), 4);
        assert!(!q.has_overflow());
    }

    #[test]
    fn queue_default_capacity() {
        let q = TaskQueue::new_default();
        assert_eq!(q.capacity(), MAX_TASK_QUEUE_DEPTH);
        // aie-rt XAIE_DMA_MAX_QUEUE_SIZE / StartQSizeMax = 4U (all tile types).
        assert_eq!(q.capacity(), 4);
    }

    #[test]
    fn queue_push_pop_fifo_order() {
        let mut q = TaskQueue::new(4);
        let e0 = TaskQueueEntry::new(0, 0, false);
        let e1 = TaskQueueEntry::new(1, 5, true);
        let e2 = TaskQueueEntry::new(2, 10, false);

        assert!(q.push(e0).is_ok());
        assert!(q.push(e1).is_ok());
        assert!(q.push(e2).is_ok());
        assert_eq!(q.len(), 3);

        // FIFO: pop returns entries in push order
        assert_eq!(q.pop(), Some(e0));
        assert_eq!(q.pop(), Some(e1));
        assert_eq!(q.pop(), Some(e2));
        assert_eq!(q.pop(), None);
        assert!(q.is_empty());
    }

    #[test]
    fn queue_capacity_enforcement() {
        let mut q = TaskQueue::new(3);
        let entry = TaskQueueEntry::new(0, 0, false);

        assert!(q.push(entry).is_ok());
        assert!(q.push(entry).is_ok());
        assert!(q.push(entry).is_ok());
        assert!(q.is_full());

        // Fourth push should fail
        assert_eq!(q.push(entry), Err(QueueFull));
        assert!(q.has_overflow());

        // Queue contents unchanged
        assert_eq!(q.len(), 3);
    }

    #[test]
    fn queue_overflow_flag_sticky() {
        let mut q = TaskQueue::new(1);
        let entry = TaskQueueEntry::new(0, 0, false);

        q.push(entry).unwrap();
        let _ = q.push(entry); // triggers overflow
        assert!(q.has_overflow());

        // Pop frees space, but overflow flag persists (sticky)
        q.pop();
        assert!(q.has_overflow());

        // Must explicitly clear
        q.clear_overflow();
        assert!(!q.has_overflow());
    }

    #[test]
    fn queue_overflow_does_not_corrupt_data() {
        let mut q = TaskQueue::new(2);
        let e0 = TaskQueueEntry::new(5, 10, true);
        let e1 = TaskQueueEntry::new(7, 20, false);
        let rejected = TaskQueueEntry::new(9, 30, true);

        q.push(e0).unwrap();
        q.push(e1).unwrap();
        let _ = q.push(rejected); // rejected, queue unchanged

        assert_eq!(q.pop(), Some(e0));
        assert_eq!(q.pop(), Some(e1));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn queue_write_register_compute() {
        let mut q = TaskQueue::new(4);
        // BD=3, repeat=7 (actual 8), token enabled
        let val = (1u32 << 31) | (7u32 << 16) | 3u32;
        assert!(q.write_register(val, START_BD_ID_WIDTH_COMPUTE));

        let entry = q.pop().unwrap();
        assert_eq!(entry.start_bd, 3);
        assert_eq!(entry.repeat_count, 7);
        assert!(entry.enable_token_issue);
    }

    #[test]
    fn queue_write_register_memtile() {
        let mut q = TaskQueue::new(4);
        // BD=23, repeat=0 (actual 1), no token
        let val = 23u32;
        assert!(q.write_register(val, START_BD_ID_WIDTH_MEMTILE));

        let entry = q.pop().unwrap();
        assert_eq!(entry.start_bd, 23);
        assert_eq!(entry.repeat_count, 0);
        assert!(!entry.enable_token_issue);
    }

    #[test]
    fn queue_write_register_full_returns_false() {
        let mut q = TaskQueue::new(1);
        assert!(q.write_register(0, START_BD_ID_WIDTH_COMPUTE));
        assert!(!q.write_register(0, START_BD_ID_WIDTH_COMPUTE));
        assert!(q.has_overflow());
    }

    #[test]
    fn queue_reset_clears_everything() {
        let mut q = TaskQueue::new(1);
        let entry = TaskQueueEntry::new(0, 0, false);
        q.push(entry).unwrap();
        let _ = q.push(entry); // overflow
        assert!(q.has_overflow());
        assert_eq!(q.len(), 1);

        q.reset();
        assert!(q.is_empty());
        assert!(!q.has_overflow());
    }

    #[test]
    fn queue_interleaved_push_pop() {
        let mut q = TaskQueue::new(2);
        let e0 = TaskQueueEntry::new(0, 0, false);
        let e1 = TaskQueueEntry::new(1, 1, true);
        let e2 = TaskQueueEntry::new(2, 2, false);

        q.push(e0).unwrap();
        q.push(e1).unwrap();
        assert!(q.is_full());

        // Pop one, push another
        assert_eq!(q.pop(), Some(e0));
        assert!(!q.is_full());
        q.push(e2).unwrap();

        assert_eq!(q.pop(), Some(e1));
        assert_eq!(q.pop(), Some(e2));
        assert!(q.is_empty());
    }

    // === Token tests ===

    #[test]
    fn token_new() {
        let t = Token::new(2, 42);
        assert_eq!(t.channel_id, 2);
        assert_eq!(t.controller_id, 42);
    }

    #[test]
    fn token_state_new_empty() {
        let ts = TokenState::new();
        assert_eq!(ts.pending_count(), 0);
        assert!(!ts.has_pending());
    }

    #[test]
    fn token_state_default() {
        let ts = TokenState::default();
        assert_eq!(ts.pending_count(), 0);
    }

    #[test]
    fn token_issue_and_consume_fifo() {
        let mut ts = TokenState::new();
        ts.issue(0, 10);
        ts.issue(1, 20);
        ts.issue(2, 30);
        assert_eq!(ts.pending_count(), 3);
        assert!(ts.has_pending());

        // Consume returns oldest first (FIFO)
        let t0 = ts.consume().unwrap();
        assert_eq!(t0.channel_id, 0);
        assert_eq!(t0.controller_id, 10);

        let t1 = ts.consume().unwrap();
        assert_eq!(t1.channel_id, 1);
        assert_eq!(t1.controller_id, 20);

        let t2 = ts.consume().unwrap();
        assert_eq!(t2.channel_id, 2);
        assert_eq!(t2.controller_id, 30);

        assert_eq!(ts.consume(), None);
        assert!(!ts.has_pending());
    }

    #[test]
    fn token_consume_empty_returns_none() {
        let mut ts = TokenState::new();
        assert_eq!(ts.consume(), None);
    }

    #[test]
    fn token_read_status_reflects_count() {
        let mut ts = TokenState::new();
        assert_eq!(ts.read_status(), 0);

        ts.issue(0, 0);
        assert_eq!(ts.read_status(), 1);

        ts.issue(1, 0);
        assert_eq!(ts.read_status(), 2);

        ts.consume();
        assert_eq!(ts.read_status(), 1);

        ts.consume();
        assert_eq!(ts.read_status(), 0);
    }

    #[test]
    fn token_reset_clears_all() {
        let mut ts = TokenState::new();
        ts.issue(0, 5);
        ts.issue(1, 10);
        assert_eq!(ts.pending_count(), 2);

        ts.reset();
        assert_eq!(ts.pending_count(), 0);
        assert!(!ts.has_pending());
    }

    #[test]
    fn token_issue_after_consume() {
        let mut ts = TokenState::new();
        ts.issue(0, 1);
        ts.consume();

        ts.issue(1, 2);
        let t = ts.consume().unwrap();
        assert_eq!(t.channel_id, 1);
        assert_eq!(t.controller_id, 2);
    }

    // === Integration-style tests ===

    #[test]
    fn task_queue_drives_token_issue() {
        // Simulate: push 3 tasks, 2 with token issue enabled.
        // As each task "completes", check token emission.
        let mut q = TaskQueue::new_default();
        let mut ts = TokenState::new();

        q.push(TaskQueueEntry::new(0, 0, true)).unwrap(); // token
        q.push(TaskQueueEntry::new(1, 2, false)).unwrap(); // no token
        q.push(TaskQueueEntry::new(2, 0, true)).unwrap(); // token

        // Simulate processing each task
        let controller_id = 7u8;

        let task0 = q.pop().unwrap();
        assert_eq!(task0.start_bd, 0);
        if task0.enable_token_issue {
            ts.issue(0, controller_id);
        }

        let task1 = q.pop().unwrap();
        assert_eq!(task1.start_bd, 1);
        assert_eq!(task1.actual_repeat_count(), 3); // repeat_count=2 -> actual=3
        if task1.enable_token_issue {
            ts.issue(0, controller_id); // should NOT happen
        }

        let task2 = q.pop().unwrap();
        assert_eq!(task2.start_bd, 2);
        if task2.enable_token_issue {
            ts.issue(0, controller_id);
        }

        // Should have exactly 2 tokens (tasks 0 and 2)
        assert_eq!(ts.pending_count(), 2);
        ts.consume().unwrap();
        ts.consume().unwrap();
        assert_eq!(ts.pending_count(), 0);
    }

    #[test]
    fn register_write_matches_aiert_encoding() {
        // Verify our encoding matches aie-rt's XAie_DmaChannelSetStartQueueGeneric:
        //   Val = SetField(StartBd, ...) | SetField(RepeatCount - 1, ...) | SetField(EnTokenIssue, ...)
        //
        // Example: StartBd=5, RepeatCount=10 (actual), EnTokenIssue=true
        // Register value: (1 << 31) | ((10-1) << 16) | 5 = 0x80090005
        let expected = 0x80090005u32;
        let entry = TaskQueueEntry::new(5, 9, true); // 9 = actual 10 - 1
        assert_eq!(entry.to_register(), expected);

        // And decode back
        let decoded = TaskQueueEntry::from_register(expected, START_BD_ID_WIDTH_COMPUTE);
        assert_eq!(decoded, entry);
    }

    #[test]
    fn queue_full_at_hardware_depth() {
        // Verify the hardware start-queue depth (aie-rt StartQSizeMax = 4).
        let mut q = TaskQueue::new_default();
        let entry = TaskQueueEntry::new(0, 0, false);

        for i in 0..MAX_TASK_QUEUE_DEPTH {
            assert!(q.push(entry).is_ok(), "Entry {} should fit", i);
        }
        assert!(q.is_full());
        assert_eq!(q.len(), MAX_TASK_QUEUE_DEPTH);
        assert_eq!(MAX_TASK_QUEUE_DEPTH, 4);

        // One past depth fails
        assert_eq!(q.push(entry), Err(QueueFull));
        assert!(q.has_overflow());
    }

    #[test]
    fn queue_pop_from_empty() {
        let mut q = TaskQueue::new(4);
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn entry_bd_id_masking_compute() {
        // Compute tile: 4-bit BD ID, so only bits [3:0] matter
        // If garbage is in upper bits, they should be masked off
        let val = 0x0000_00FF; // bits [7:0] all set
        let entry = TaskQueueEntry::from_register(val, START_BD_ID_WIDTH_COMPUTE);
        assert_eq!(entry.start_bd, 0x0F); // only [3:0]
    }

    #[test]
    fn entry_bd_id_masking_memtile() {
        // Memtile: 6-bit BD ID, so only bits [5:0] matter
        let val = 0x0000_00FF; // bits [7:0] all set
        let entry = TaskQueueEntry::from_register(val, START_BD_ID_WIDTH_MEMTILE);
        assert_eq!(entry.start_bd, 0x3F); // only [5:0]
    }

    #[test]
    fn multiple_tokens_same_channel() {
        // A channel can issue multiple tokens if multiple tasks complete
        let mut ts = TokenState::new();
        ts.issue(0, 5);
        ts.issue(0, 5);
        ts.issue(0, 5);
        assert_eq!(ts.pending_count(), 3);

        // All three should be consumable
        for _ in 0..3 {
            let t = ts.consume().unwrap();
            assert_eq!(t.channel_id, 0);
            assert_eq!(t.controller_id, 5);
        }
        assert_eq!(ts.consume(), None);
    }
}
