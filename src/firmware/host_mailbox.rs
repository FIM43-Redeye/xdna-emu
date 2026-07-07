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
    enabled: bool,
}

impl HostMailbox {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable the model for the boot-to-idle path.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// One step: poll for a post; on a completable consume, deliver the
    /// completion. No-op when disabled.
    pub fn tick(&mut self, bus: &mut Bus) {
        if !self.enabled {
            return;
        }
        if let PollResult::Consumed { completable: true } = self.consumer.poll(bus) {
            self.agent.deliver(bus);
        }
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
