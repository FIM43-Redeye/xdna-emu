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
        let task = bus.load_local32(SCHED_CURRENT_TASK);
        if task == 0 || task >= LOCAL_ADDR_END {
            return None;
        }
        let done = task + DONE_FLAG_OFF;
        bus.store_local32(done, 1);
        Some(done)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_writes_done_flag_for_valid_task() {
        let mut bus = Bus::new(vec![]);
        // Scheduler global -> current task 0x9040 (as at boot).
        bus.store_local32(SCHED_CURRENT_TASK, 0x9040);
        let agent = CompletionAgent::new();
        assert_eq!(agent.deliver(&mut bus), Some(0x9070));
        assert_eq!(bus.load_local32(0x9070), 1, "done-flag [task+0x30] set to 1");
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
        bus.store_local32(SCHED_CURRENT_TASK, 0x0500_0000);
        let agent = CompletionAgent::new();
        assert_eq!(agent.deliver(&mut bus), None);
    }
}
