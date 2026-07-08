//! Host-side mailbox model for the firmware boot path: the two hardware agents
//! between the firmware's mailbox POST and the task done-flag write.
//!
//! Faithful completion model (spec 2026-07-06-firmware-faithful-completion-model):
//! the firmware posts a descriptor into the mailbox register block (advancing the
//! i2x tail `0x27200170`), then blocks in the RTOS scheduler recursion polling a
//! LOCAL done-flag `[task+0x30]` that no firmware code writes. On real silicon a
//! local hardware agent writes that flag when the request completes. This models
//! the two agents: the HOST consuming the i2x descriptor (protocol-faithful,
//! inert to the stuck boot) and the LOCAL completion agent writing the done-flag
//! (the functional unblock). Zero modeled latency -- the store-watch proved the
//! firmware never re-clears the flag, so a completion written any time after the
//! post sticks; there is no timing to calibrate.

use super::mmio::Bus;

/// Scheduler global holding the current-task pointer: the dispatcher at `0xd81a`
/// loads the scheduler table at local `0x2250` and reads the current task from
/// `[0x2250 + 0x28]`. Live-read so a task switch is followed, not snapshotted.
const SCHED_CURRENT_TASK: u32 = 0x2250 + 0x28;
/// Done-flag offset within the task struct: the dispatcher checks
/// `l32i.n a10,[task+0x30]` at `0xd828` and re-dispatches while it is zero.
const DONE_FLAG_OFF: u32 = 0x30;
/// Upper bound of a valid firmware-local task pointer (local data window).
const LOCAL_ADDR_END: u32 = 0x0400_0000;

/// i2x tail register the firmware advances to POST (fw->host producer pointer).
const I2X_TAIL_REG: u32 = 0x2720_0170;
/// i2x head register the host advances to acknowledge (consumer pointer).
const I2X_HEAD_REG: u32 = 0x2720_0174;
/// i2x interrupt/status register the host clears on acknowledge
/// (`i2x.mb_head_ptr_reg + 4`, xdna-driver `aie2_pci.c:376-379`).
const I2X_INTR_REG: u32 = 0x2720_0178;
/// Descriptor payload-pointer register (fw writes it before the tail; a zero
/// here means an unexpected/partial post -- the descriptor-sanity guard).
const DESC_PTR_REG: u32 = 0x2720_0180;

/// Agent 2: the NPU local completion hardware that writes a task's done-flag
/// into firmware-local SRAM when its request completes (shape ii). Reads the
/// current task from the scheduler global and writes `[task+0x30] = 1`.
#[derive(Default)]
pub struct CompletionAgent;

impl CompletionAgent {
    pub fn new() -> Self {
        Self
    }

    /// Deliver a completion for the current task. Returns the done-flag address
    /// written, or `None` if there is no valid current task yet (scheduler not
    /// up). Zero latency; the value written is `1` because the dispatcher only
    /// tests the flag with `beqz` (non-zero == done).
    // PROJECTED Layer 2: if a downstream consumer reads the done-flag as a
    // status code or pointer rather than a boolean, write the real token instead
    // of 1.
    pub fn deliver(&self, bus: &mut Bus) -> Option<u32> {
        let task = bus.data_load32(SCHED_CURRENT_TASK);
        if task == 0 || task >= LOCAL_ADDR_END {
            return None;
        }
        let done = task + DONE_FLAG_OFF;
        bus.data_store32(done, 1);
        Some(done)
    }
}

/// Descriptor valid flag: `[0xfae0] == 1` means the worker run-fn (0x588c) has
/// built and flushed a column-power command. Zero (memset default) means none.
const DESC_VALID_REG: u32 = 0xfae0;
/// Column bitmask field of the flushed descriptor (`[0xfae8]`): bit `i` set means
/// "power column `i`". The boot bring-up flushes 0xf (all four NPU1 columns).
const DESC_COLMASK_REG: u32 = 0xfae8;
/// Target-task field of the flushed descriptor (`[0xfaf0]`): the task struct whose
/// done-flag the SMU/PSP writes on completion (0x9040 at boot).
const DESC_TARGET_REG: u32 = 0xfaf0;
/// Per-column status byte base: `FUN_00008c68` busy-polls `[0xf9e0+col*0x60]`
/// bit3 for "column powered/ready".
const COL_STATUS_BASE: u32 = 0xf9e0;
/// Stride between adjacent per-column status structs.
const COL_STATUS_STRIDE: u32 = 0x60;
/// bit3 of the per-column status byte: set == column powered (the poll's success bit).
const COL_READY_BIT: u8 = 0x08;
/// NPU1 column count the descriptor colmask spans.
const NUM_COLS: u32 = 4;

/// Agent 3 (2026-07-08 RE): the SMU/PSP column-power completion. Supersedes the
/// i2x-post trigger for the boot column bring-up wall. The worker run-fn (0x588c)
/// builds a command descriptor at local `0xfae0` -- {valid, _, colmask, _, target}
/// -- and cache-flushes it to shared memory (a `Dhwbi`/`Dsync` loop) with NO
/// external MMIO doorbell. On silicon the SMU/PSP polls that shared region, powers
/// the masked columns, and writes the completion BACK into mgmt SRAM: the
/// per-column status byte `[0xf9e0+col*0x60]` bit3 (which `FUN_00008c68` polls) and
/// the task done-flag `[target+0x30]` (which the dispatcher polls). This models
/// that HW writeback. Faithful (not the old "corrupting shim"): the descriptor is
/// deliberately flushed FOR an external reader, and the target carries await-mask 0
/// precisely because it is completed by a direct external write, not an event.
#[derive(Default)]
pub struct ColumnPowerAgent {
    /// Last `(colmask, target)` completed. A re-flushed identical descriptor during
    /// the firmware's busy-poll is not re-completed, so the agent never fights the
    /// firmware's own retire of the worker.
    // ponytail: content-keyed one-shot. If boot ever submits distinct commands with
    // the same (colmask,target), key on a submission counter instead.
    last: Option<(u32, u32)>,
}

impl ColumnPowerAgent {
    pub fn new() -> Self {
        Self::default()
    }

    /// Poll the flushed descriptor and, while a valid column-power command stands,
    /// keep the HW completion asserted in mgmt SRAM. Returns `Some((colmask,
    /// target))` on the rising edge of a newly-standing descriptor (for the
    /// completion count), `None` on a re-assert / no command / invalid.
    ///
    /// The completion is LEVEL-based, not a one-shot edge: real SMU/PSP status is
    /// "column powered" (bit3 stays set while the column is up), so the agent
    /// re-writes the per-column bit3 and the target done-flag on EVERY tick the
    /// descriptor stands valid. This is required because the firmware's own task
    /// dispatch ZEROES `[task+0x30]` when it makes the target current (observed at
    /// boot: the flag is cleared ~n=58929 after being set at ~n=47809); a one-shot
    /// write would be clobbered and never re-seen. The firmware tears the
    /// descriptor down (valid -> 0) when it acknowledges completion, which re-arms
    /// the agent for the next command and stops the re-assert -- a faithful
    /// handshake, not the old corrupting done-check forcing.
    pub fn tick(&mut self, bus: &mut Bus) -> Option<(u32, u32)> {
        if bus.data_load32(DESC_VALID_REG) == 0 {
            self.last = None; // descriptor torn down: re-arm for the next command
            return None;
        }
        let colmask = bus.data_load32(DESC_COLMASK_REG);
        let target = bus.data_load32(DESC_TARGET_REG);
        if colmask == 0 || target == 0 || target >= LOCAL_ADDR_END {
            return None;
        }
        for col in 0..NUM_COLS {
            if colmask & (1 << col) != 0 {
                let sb = COL_STATUS_BASE + col * COL_STATUS_STRIDE;
                let cur = bus.data_load8(sb);
                bus.data_store8(sb, u32::from(cur | COL_READY_BIT));
            }
        }
        bus.data_store32(target + DONE_FLAG_OFF, 1);
        // Rising edge of this standing descriptor -> report it once for the stat.
        if self.last == Some((colmask, target)) {
            return None;
        }
        self.last = Some((colmask, target));
        Some((colmask, target))
    }
}

/// Outcome of one consumer poll.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PollResult {
    /// The i2x tail did not advance -- no new post.
    NoPost,
    /// A post was consumed (acknowledged). `completable` is false when the
    /// descriptor looked invalid (zero payload ptr): consumed for protocol
    /// fidelity, but no completion is delivered.
    Consumed { completable: bool },
}

/// Agent 1: the host servicing a fw->host (i2x) descriptor post. Detects the
/// tail advance, reads the descriptor from the backed mailbox register block,
/// and acknowledges per the driver (head = tail, intr = 0).
#[derive(Default)]
pub struct HostMailboxConsumer {
    /// Last i2x tail value seen, for edge (advance) detection.
    last_tail: u32,
}

impl HostMailboxConsumer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Poll the i2x tail for a new post. On an advance, read the descriptor,
    /// acknowledge, and report whether it is completable.
    pub fn poll(&mut self, bus: &mut Bus) -> PollResult {
        let tail = bus.data_load32(I2X_TAIL_REG);
        // The boot descriptor tail is monotonic (no wrap) -- any change is a post.
        // PROJECTED Layer 2: ring wrap / TOMBSTONE decrease handling arrives with
        // the data-plane ring protocol.
        if tail == self.last_tail {
            return PollResult::NoPost;
        }
        self.last_tail = tail;

        // Descriptor sanity: a zero payload pointer is not a completable request.
        let desc_ptr = bus.data_load32(DESC_PTR_REG);

        // Acknowledge (protocol-faithful; inert to the stuck boot -- the fw never
        // reads these back in the recursion, but real post-idle paths will).
        bus.data_store32(I2X_HEAD_REG, tail);
        bus.data_store32(I2X_INTR_REG, 0);

        PollResult::Consumed { completable: desc_ptr != 0 }
    }
}

/// The host mailbox model: the two agents plus an enable flag. Ticked once per
/// instruction by the boot loop; a no-op until `enable`d, so it does not perturb
/// firmware tests that step the CPU for other reasons.
#[derive(Default)]
pub struct HostMailbox {
    consumer: HostMailboxConsumer,
    agent: CompletionAgent,
    column_power: ColumnPowerAgent,
    enabled: bool,
    /// Diagnostic: column-power completions delivered so far, and the last one.
    column_completions: u32,
    last_column: Option<(u32, u32)>,
}

impl HostMailbox {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable the model for the boot-to-idle path.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// One step: poll both completion seams. No-op when disabled.
    /// 1. i2x mailbox post -> current-task done-flag (host-request completion).
    /// 2. `0xfae0` column-power descriptor flush -> per-column bit3 + target
    ///    done-flag (the boot column bring-up completion; 2026-07-08 RE).
    pub fn tick(&mut self, bus: &mut Bus) {
        if !self.enabled {
            return;
        }
        if let PollResult::Consumed { completable: true } = self.consumer.poll(bus) {
            self.agent.deliver(bus);
        }
        if let Some(done) = self.column_power.tick(bus) {
            self.column_completions += 1;
            self.last_column = Some(done);
        }
    }

    /// Diagnostic: number of column-power completions delivered and the last
    /// `(colmask, target)` completed (for boot-progress assertions).
    pub fn column_stats(&self) -> (u32, Option<(u32, u32)>) {
        (self.column_completions, self.last_column)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_writes_done_flag_for_valid_task() {
        let mut bus = Bus::new(vec![]);
        // Scheduler global -> current task 0x9040 (as at boot).
        bus.data_store32(SCHED_CURRENT_TASK, 0x9040);
        let agent = CompletionAgent::new();
        assert_eq!(agent.deliver(&mut bus), Some(0x9070));
        assert_eq!(bus.data_load32(0x9070), 1, "done-flag [task+0x30] set to 1");
    }

    #[test]
    fn completion_skips_when_scheduler_not_up() {
        let mut bus = Bus::new(vec![]);
        // Current-task pointer still zero (unwritten): no valid task.
        let agent = CompletionAgent::new();
        assert_eq!(agent.deliver(&mut bus), None);
    }

    #[test]
    fn completion_skips_out_of_range_task_pointer() {
        let mut bus = Bus::new(vec![]);
        // A pointer outside the local window is not a valid task struct.
        bus.data_store32(SCHED_CURRENT_TASK, 0x0500_0000);
        let agent = CompletionAgent::new();
        assert_eq!(agent.deliver(&mut bus), None);
    }

    #[test]
    fn no_post_when_tail_unchanged() {
        let mut bus = Bus::new(vec![]);
        let mut c = HostMailboxConsumer::new();
        // Tail register unwritten (reads 0) == last_tail 0: no post.
        assert_eq!(c.poll(&mut bus), PollResult::NoPost);
    }

    #[test]
    fn tail_advance_with_descriptor_is_completable_and_acked() {
        let mut bus = Bus::new(vec![]);
        let mut c = HostMailboxConsumer::new();
        // Firmware writes the descriptor, then advances the tail (the post).
        bus.data_store32(0x2720_0180, 0x08a0_0ff0); // payload ptr (non-zero)
        bus.data_store32(0x2720_0170, 0xf18); // tail advance
        assert_eq!(c.poll(&mut bus), PollResult::Consumed { completable: true });
        // Acknowledged: head = tail, intr = 0.
        assert_eq!(bus.data_load32(0x2720_0174), 0xf18, "i2x head advanced to tail");
        assert_eq!(bus.data_load32(0x2720_0178), 0, "i2x intr cleared");
        // Tail unchanged on the next poll -> no repeat post.
        assert_eq!(c.poll(&mut bus), PollResult::NoPost);
    }

    #[test]
    fn tail_advance_with_zero_descriptor_ptr_is_consumed_not_completable() {
        let mut bus = Bus::new(vec![]);
        let mut c = HostMailboxConsumer::new();
        // Tail advances but the descriptor payload ptr is zero (partial/unexpected).
        bus.data_store32(0x2720_0170, 0xf18);
        assert_eq!(c.poll(&mut bus), PollResult::Consumed { completable: false });
        // Still acked (protocol fidelity).
        assert_eq!(bus.data_load32(0x2720_0174), 0xf18);
    }

    fn post_descriptor(bus: &mut Bus, tail: u32) {
        bus.data_store32(0x2720_0180, 0x08a0_0ff0); // non-zero payload ptr
        bus.data_store32(0x2720_0170, tail); // tail advance == the post
    }

    #[test]
    fn enabled_tick_completes_the_current_task() {
        let mut bus = Bus::new(vec![]);
        bus.data_store32(SCHED_CURRENT_TASK, 0x9040);
        post_descriptor(&mut bus, 0xf18);
        let mut hm = HostMailbox::new();
        hm.enable();
        hm.tick(&mut bus);
        assert_eq!(bus.data_load32(0x9070), 1, "done-flag set via the full chain");
        assert_eq!(bus.data_load32(0x2720_0174), 0xf18, "consumer acked head");
    }

    #[test]
    fn disabled_tick_is_a_noop() {
        let mut bus = Bus::new(vec![]);
        bus.data_store32(SCHED_CURRENT_TASK, 0x9040);
        post_descriptor(&mut bus, 0xf18);
        let mut hm = HostMailbox::new(); // not enabled
        hm.tick(&mut bus);
        assert_eq!(bus.data_load32(0x9070), 0, "no completion while disabled");
        assert_eq!(bus.data_load32(0x2720_0174), 0, "no ack while disabled");
    }

    /// Build a flushed column-power descriptor at 0xfae0 (as the worker run-fn
    /// 0x588c does): valid, colmask, target.
    fn flush_descriptor(bus: &mut Bus, colmask: u32, target: u32) {
        bus.data_store32(DESC_VALID_REG, 1);
        bus.data_store32(0xfae4, 1);
        bus.data_store32(DESC_COLMASK_REG, colmask);
        bus.data_store32(DESC_TARGET_REG, target);
    }

    #[test]
    fn column_power_writes_bit3_and_done_flag() {
        let mut bus = Bus::new(vec![]);
        flush_descriptor(&mut bus, 0xf, 0x9040); // all 4 columns, task 0x9040
        let mut agent = ColumnPowerAgent::new();
        assert_eq!(agent.tick(&mut bus), Some((0xf, 0x9040)));
        // Every masked column's status byte has bit3 set.
        for col in 0..4u32 {
            let sb = COL_STATUS_BASE + col * COL_STATUS_STRIDE;
            assert_eq!(bus.data_load8(sb) & COL_READY_BIT, COL_READY_BIT, "col {col} bit3 set");
        }
        // The target task's done-flag [target+0x30] is set.
        assert_eq!(bus.data_load32(0x9040 + DONE_FLAG_OFF), 1, "done-flag [0x9070] set");
    }

    #[test]
    fn column_power_is_idempotent_per_descriptor() {
        let mut bus = Bus::new(vec![]);
        flush_descriptor(&mut bus, 0xf, 0x9040);
        let mut agent = ColumnPowerAgent::new();
        assert_eq!(agent.tick(&mut bus), Some((0xf, 0x9040)), "first flush completes");
        // Re-flushed identical descriptor during the busy-poll is not re-completed
        // (the agent must not fight the firmware's own worker retire).
        assert_eq!(agent.tick(&mut bus), None, "same descriptor not re-completed");
    }

    #[test]
    fn column_power_skips_absent_or_invalid_descriptor() {
        let mut bus = Bus::new(vec![]);
        let mut agent = ColumnPowerAgent::new();
        // No descriptor flushed (valid == 0): nothing to do.
        assert_eq!(agent.tick(&mut bus), None);
        // Valid but zero colmask / zero target: not a real command.
        bus.data_store32(DESC_VALID_REG, 1);
        assert_eq!(agent.tick(&mut bus), None, "zero colmask+target skipped");
        // Out-of-range target pointer is not a local task struct.
        bus.data_store32(DESC_COLMASK_REG, 0xf);
        bus.data_store32(DESC_TARGET_REG, 0x0500_0000);
        assert_eq!(agent.tick(&mut bus), None, "out-of-range target skipped");
    }

    #[test]
    fn column_power_honors_partial_colmask() {
        let mut bus = Bus::new(vec![]);
        flush_descriptor(&mut bus, 0x5, 0x9040); // columns 0 and 2 only
        let mut agent = ColumnPowerAgent::new();
        assert_eq!(agent.tick(&mut bus), Some((0x5, 0x9040)));
        for col in 0..4u32 {
            let sb = COL_STATUS_BASE + col * COL_STATUS_STRIDE;
            let want = if col == 0 || col == 2 { COL_READY_BIT } else { 0 };
            assert_eq!(bus.data_load8(sb) & COL_READY_BIT, want, "col {col} bit3");
        }
    }

    #[test]
    fn enabled_tick_completes_column_power_descriptor() {
        // The full HostMailbox seam drives the column-power agent too.
        let mut bus = Bus::new(vec![]);
        flush_descriptor(&mut bus, 0xf, 0x9040);
        let mut hm = HostMailbox::new();
        hm.enable();
        hm.tick(&mut bus);
        assert_eq!(bus.data_load32(0x9070), 1, "done-flag set via HostMailbox tick");
        assert_eq!(bus.data_load8(COL_STATUS_BASE) & COL_READY_BIT, COL_READY_BIT, "col0 bit3 via tick");
    }

    #[test]
    fn second_post_rearms_and_completes_again() {
        let mut bus = Bus::new(vec![]);
        bus.data_store32(SCHED_CURRENT_TASK, 0x9040);
        let mut hm = HostMailbox::new();
        hm.enable();
        post_descriptor(&mut bus, 0xf18);
        hm.tick(&mut bus);
        // A new task blocks and a second post arrives (tail advances again).
        bus.data_store32(SCHED_CURRENT_TASK, 0xa000);
        bus.data_store32(0xa030, 0); // its done-flag starts clear
        post_descriptor(&mut bus, 0x1e30);
        hm.tick(&mut bus);
        assert_eq!(bus.data_load32(0xa030), 1, "second task completed on re-arm");
    }
}
