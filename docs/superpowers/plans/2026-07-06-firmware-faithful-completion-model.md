# Faithful Firmware Task-Completion Model Implementation Plan

> **SUPERSEDED (2026-07-10, reaffirmed 2026-07-27).** This plan implemented a
> diagnostic hypothesis around an internal queue, not the host management
> mailbox. `0x27200170/174/178` must not be used for BAR2/BAR4 delivery. See
> `docs/arch/firmware-array-plugin-wiring.md` for the current boundary.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the force-done stub with a faithful, post-triggered task-completion model so `boot_to_idle` advances past the firmware's `task_dispatcher` recursion along the real path.

**Architecture:** Two agents in a new `src/firmware/host_mailbox.rs`, mirroring two hardware blocks. `HostMailboxConsumer` detects the firmware's mailbox POST (an advance of the i2x tail register `0x27200170`), reads the descriptor from the already-backed mailbox register block, and acknowledges (head + intr) per the xdna-driver protocol. `CompletionAgent` then writes the current task's done-flag `[task+0x30]` in firmware-local memory (shape ii) with zero modeled latency. A `HostMailbox` wrapper holds both, is ticked once per instruction by `boot_to_idle`, and is a no-op until explicitly enabled (opt-in, so existing tests are untouched).

**Tech Stack:** Rust; the in-tree Xtensa firmware interpreter (`src/firmware/`); the routed MMIO `Bus` (`src/firmware/mmio.rs`); the real NPU management firmware `.sbin`.

**Reference:** spec `docs/superpowers/specs/2026-07-06-firmware-faithful-completion-model-design.md`. Findings `docs/superpowers/findings/2026-07-06-iter18-phase0-interrupt-wiring.md`.

## Global Constraints

- **No latency calibration / no timing knob.** Completion fires at the post (zero modeled latency). The firmware never re-clears the done-flag, so this is safe. Do not add a delay parameter. (Spec principle 1.)
- **Derive from the toolchain.** Register offsets and the ack sequence come from xdna-driver (`amdxdna_mailbox.c`, `aie2_pci.c`); comment the hardware fact, not the tool. (Spec principle 2.)
- **Two agents stay separate types** even though `CompletionAgent`'s only current action is one write. (Spec principle 3.)
- **Projection markers:** every deferred layer is marked in code as `// PROJECTED Layer N: <what> when <trigger>`. Do not build projected layers. (Spec principle 4.)
- **Completion is opt-in:** `HostMailbox` is disabled by default; only `boot_to_idle` after an explicit `enable_host_mailbox()` runs it. Existing tests must not change behavior.
- **No emoji anywhere.** Commit messages end with a line: `Generated using Claude Code.`
- Always run `cargo test --lib` after changes; a regression is fixed before moving on.

## Derived constants (used across tasks, defined once in Task 1)

| Name | Value | Meaning / source |
|------|-------|------------------|
| `I2X_TAIL_REG` | `0x2720_0170` | fw advances to POST (observed burst + `aie2_pci.c:376`) |
| `I2X_HEAD_REG` | `0x2720_0174` | host advances to ack (`mailbox_set_headptr`) |
| `I2X_INTR_REG` | `0x2720_0178` | host clears on ack (`head_ptr_reg + 4`, `aie2_pci.c:376-379`) |
| `DESC_PTR_REG` | `0x2720_0180` | descriptor payload ptr (observed burst n=6958) |
| `SCHED_CURRENT_TASK` | `0x2250 + 0x28` = `0x2278` | current-task ptr; dispatcher `0xd81a` |
| `DONE_FLAG_OFF` | `0x30` | done-flag within the task struct; dispatcher `0xd828` |
| `LOCAL_ADDR_END` | `0x0400_0000` | upper bound of a valid firmware-local pointer |

## File structure

- **Create** `src/firmware/host_mailbox.rs` -- both agents, the `HostMailbox` wrapper, the constants, and all unit tests. One focused file (~220 lines incl. tests).
- **Modify** `src/firmware/mod.rs` -- declare `mod host_mailbox;`, add a `host_mailbox: HostMailbox` field to `FirmwareProcessor` (init in both `load` and `load_m2c`), add `pub fn enable_host_mailbox`, tick it in `boot_to_idle`, and add the boot integration test in `mod boot_tests`.

The mailbox descriptor is already backed by `Bus.mailbox`; **no `mmio.rs` change is needed.**

---

### Task 1: `CompletionAgent` -- the local done-flag write

**Files:**
- Create: `src/firmware/host_mailbox.rs`
- Test: `src/firmware/host_mailbox.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `super::mmio::Bus` -- `load_local32(u32) -> u32`, `store_local32(u32, u32)`.
- Produces: `pub struct CompletionAgent;` with `pub fn new() -> Self` and `pub fn deliver(&self, bus: &mut Bus) -> Option<u32>` (returns the done-flag address written, or `None` if no valid current task). Constants `SCHED_CURRENT_TASK`, `DONE_FLAG_OFF`, `LOCAL_ADDR_END`.

- [ ] **Step 1: Write the failing tests**

Create `src/firmware/host_mailbox.rs` with only the module doc, the imports, and this test module (the types do not exist yet, so it will not compile -- that is the failing state):

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib firmware::host_mailbox`
Expected: FAIL to compile -- `cannot find type CompletionAgent`. First add `mod host_mailbox;` to `src/firmware/mod.rs` (alphabetically near the other `mod` lines, e.g. after `mod error;`), then re-run; expected: FAIL with `CompletionAgent` unresolved.

- [ ] **Step 3: Implement `CompletionAgent`**

Insert above the `#[cfg(test)]` block:

```rust
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib firmware::host_mailbox`
Expected: PASS (3 tests). Then `cargo test --lib` -- full suite still green.

- [ ] **Step 5: Commit**

```bash
git add src/firmware/host_mailbox.rs src/firmware/mod.rs
git commit -m "feat(#140): CompletionAgent -- faithful local done-flag write

Agent 2 of the completion model: reads the current task from the
scheduler global [0x2278] and writes [task+0x30]=1 in firmware-local
memory. Zero modeled latency; guards a not-yet-initialized scheduler.

Generated using Claude Code."
```

---

### Task 2: `HostMailboxConsumer` -- post detection, descriptor read, acknowledge

**Files:**
- Modify: `src/firmware/host_mailbox.rs`
- Test: `src/firmware/host_mailbox.rs` (`mod tests`)

**Interfaces:**
- Consumes: `Bus::load32(u32) -> u32`, `Bus::store32(u32, u32)`.
- Produces: `pub enum PollResult { NoPost, Consumed { completable: bool } }`; `pub struct HostMailboxConsumer` with `pub fn new() -> Self` and `pub fn poll(&mut self, bus: &mut Bus) -> PollResult`. Constants `I2X_TAIL_REG`, `I2X_HEAD_REG`, `I2X_INTR_REG`, `DESC_PTR_REG`.

- [ ] **Step 1: Write the failing tests**

Add to `mod tests`:

```rust
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
        bus.store32(0x2720_0180, 0x08a0_0ff0); // payload ptr (non-zero)
        bus.store32(0x2720_0170, 0xf18); // tail advance
        assert_eq!(c.poll(&mut bus), PollResult::Consumed { completable: true });
        // Acknowledged: head = tail, intr = 0.
        assert_eq!(bus.load32(0x2720_0174), 0xf18, "i2x head advanced to tail");
        assert_eq!(bus.load32(0x2720_0178), 0, "i2x intr cleared");
        // Tail unchanged on the next poll -> no repeat post.
        assert_eq!(c.poll(&mut bus), PollResult::NoPost);
    }

    #[test]
    fn tail_advance_with_zero_descriptor_ptr_is_consumed_not_completable() {
        let mut bus = Bus::new(vec![]);
        let mut c = HostMailboxConsumer::new();
        // Tail advances but the descriptor payload ptr is zero (partial/unexpected).
        bus.store32(0x2720_0170, 0xf18);
        assert_eq!(c.poll(&mut bus), PollResult::Consumed { completable: false });
        // Still acked (protocol fidelity).
        assert_eq!(bus.load32(0x2720_0174), 0xf18);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib firmware::host_mailbox`
Expected: FAIL to compile -- `cannot find type HostMailboxConsumer` / `PollResult`.

- [ ] **Step 3: Implement the consumer**

Add the constants near `SCHED_CURRENT_TASK`:

```rust
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
```

Add the type above `#[cfg(test)]`:

```rust
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
        let tail = bus.load32(I2X_TAIL_REG);
        // The boot descriptor tail is monotonic (no wrap) -- any change is a post.
        // PROJECTED Layer 2: ring wrap / TOMBSTONE decrease handling arrives with
        // the data-plane ring protocol.
        if tail == self.last_tail {
            return PollResult::NoPost;
        }
        self.last_tail = tail;

        // Descriptor sanity: a zero payload pointer is not a completable request.
        let desc_ptr = bus.load32(DESC_PTR_REG);

        // Acknowledge (protocol-faithful; inert to the stuck boot -- the fw never
        // reads these back in the recursion, but real post-idle paths will).
        bus.store32(I2X_HEAD_REG, tail);
        bus.store32(I2X_INTR_REG, 0);

        PollResult::Consumed { completable: desc_ptr != 0 }
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib firmware::host_mailbox`
Expected: PASS (6 tests total). Then `cargo test --lib` -- full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/firmware/host_mailbox.rs
git commit -m "feat(#140): HostMailboxConsumer -- i2x descriptor post detect + ack

Agent 1: detects the mailbox POST (i2x tail advance at 0x27200170),
reads the descriptor from the backed register block, and acknowledges
per xdna-driver (head = tail, intr = 0). Reports whether the descriptor
is completable (non-zero payload ptr).

Generated using Claude Code."
```

---

### Task 3: `HostMailbox` wrapper + `tick`

**Files:**
- Modify: `src/firmware/host_mailbox.rs`
- Test: `src/firmware/host_mailbox.rs` (`mod tests`)

**Interfaces:**
- Consumes: `HostMailboxConsumer`, `CompletionAgent`, `PollResult` (from Tasks 1-2).
- Produces: `pub struct HostMailbox` with `pub fn new() -> Self`, `pub fn enable(&mut self)`, `pub fn tick(&mut self, bus: &mut Bus)`. Used by `boot_to_idle` in Task 4.

- [ ] **Step 1: Write the failing tests**

Add to `mod tests`:

```rust
    fn post_descriptor(bus: &mut Bus, tail: u32) {
        bus.store32(0x2720_0180, 0x08a0_0ff0); // non-zero payload ptr
        bus.store32(0x2720_0170, tail); // tail advance == the post
    }

    #[test]
    fn enabled_tick_completes_the_current_task() {
        let mut bus = Bus::new(vec![]);
        bus.store_local32(SCHED_CURRENT_TASK, 0x9040);
        post_descriptor(&mut bus, 0xf18);
        let mut hm = HostMailbox::new();
        hm.enable();
        hm.tick(&mut bus);
        assert_eq!(bus.load_local32(0x9070), 1, "done-flag set via the full chain");
        assert_eq!(bus.load32(0x2720_0174), 0xf18, "consumer acked head");
    }

    #[test]
    fn disabled_tick_is_a_noop() {
        let mut bus = Bus::new(vec![]);
        bus.store_local32(SCHED_CURRENT_TASK, 0x9040);
        post_descriptor(&mut bus, 0xf18);
        let mut hm = HostMailbox::new(); // not enabled
        hm.tick(&mut bus);
        assert_eq!(bus.load_local32(0x9070), 0, "no completion while disabled");
        assert_eq!(bus.load32(0x2720_0174), 0, "no ack while disabled");
    }

    #[test]
    fn second_post_rearms_and_completes_again() {
        let mut bus = Bus::new(vec![]);
        bus.store_local32(SCHED_CURRENT_TASK, 0x9040);
        let mut hm = HostMailbox::new();
        hm.enable();
        post_descriptor(&mut bus, 0xf18);
        hm.tick(&mut bus);
        // A new task blocks and a second post arrives (tail advances again).
        bus.store_local32(SCHED_CURRENT_TASK, 0xa000);
        bus.store_local32(0xa030, 0); // its done-flag starts clear
        post_descriptor(&mut bus, 0x1e30);
        hm.tick(&mut bus);
        assert_eq!(bus.load_local32(0xa030), 1, "second task completed on re-arm");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib firmware::host_mailbox`
Expected: FAIL to compile -- `cannot find type HostMailbox`.

- [ ] **Step 3: Implement the wrapper**

Add above `#[cfg(test)]`:

```rust
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib firmware::host_mailbox`
Expected: PASS (9 tests total). Then `cargo test --lib` -- full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/firmware/host_mailbox.rs
git commit -m "feat(#140): HostMailbox wrapper -- tick composes consume + complete

Holds both agents and an enable flag; tick() polls the consumer and, on
a completable post, delivers the completion. No-op when disabled so it
does not perturb other firmware tests. Re-arms per post.

Generated using Claude Code."
```

---

### Task 4: Integrate into `boot_to_idle` and characterize the real path

**Files:**
- Modify: `src/firmware/mod.rs` (struct field + both constructors + `enable_host_mailbox` + tick in `boot_to_idle`); test in `mod boot_tests`.

**Interfaces:**
- Consumes: `host_mailbox::HostMailbox` (Task 3), `FirmwareProcessor::load_m2c`, `boot_to_idle`.
- Produces: `pub fn enable_host_mailbox(&mut self)`; a `host_mailbox` field ticked each instruction in `boot_to_idle`.

- [ ] **Step 1: Add the field and the tick, and a `use`**

In `src/firmware/mod.rs`: add `use host_mailbox::HostMailbox;` near the other `use` lines. Add the field to `FirmwareProcessor` (after `symbols`):

```rust
    /// Host-side mailbox model (Task-completion). Disabled by default; ticked by
    /// `boot_to_idle`. Enable with `enable_host_mailbox` for the real boot path.
    host_mailbox: HostMailbox,
```

Initialize it in BOTH `load` and `load_m2c` -- change each `Self { cpu, bus, entry, symbols }` to `Self { cpu, bus, entry, symbols, host_mailbox: HostMailbox::new() }`.

Add the method inside `impl FirmwareProcessor` (right after `load_m2c`):

```rust
    /// Enable the host-mailbox completion model for the boot-to-idle run. Off by
    /// default so existing observation tests are unaffected.
    pub fn enable_host_mailbox(&mut self) {
        self.host_mailbox.enable();
    }
```

In `boot_to_idle`, tick the model immediately after the CPU step. Change:

```rust
            let step = self.cpu.step(&mut self.bus);
```

to:

```rust
            let step = self.cpu.step(&mut self.bus);
            // Faithful task-completion: on the firmware's mailbox POST, the host
            // model consumes the descriptor and the completion agent writes the
            // task done-flag (no-op until enabled).
            self.host_mailbox.tick(&mut self.bus);
```

- [ ] **Step 2: Verify the opt-in default is inert**

Run: `cargo test --lib`
Expected: PASS -- the full suite is unchanged (the model is disabled by default; tick is a no-op). This confirms the integration does not regress existing tests.

- [ ] **Step 3: Write the boot integration test (characterizes the real path)**

Add to `mod boot_tests` (firmware-gated, runs in the normal suite when the `.sbin` is present -- this is the regression gate for the mechanism):

```rust
    /// M2c iter18: with the faithful completion model enabled, boot advances past
    /// the `task_dispatcher` (0xd7f0) recursion along the REAL path (the task is
    /// picked from real scheduler state, not force-done's artificial switch). The
    /// completion delivers the done-flag `[0x9070]`; boot then runs to its next
    /// genuine stop, which this test records for the follow-through task.
    #[test]
    fn m2c_boot_completion_advances_past_recursion() {
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");
        let mut proc = FirmwareProcessor::load_m2c(img);
        proc.enable_host_mailbox();
        let report = proc.boot_to_idle(2_000_000);

        eprintln!("=== M2c completion-model boot ===");
        eprintln!("reached_idle    = {}", report.reached_idle);
        eprintln!("instrs_executed = {}", report.instrs_executed);
        eprintln!("last_pc         = {:#x}  {}", report.last_pc, nearest_symbol(&proc.symbols, report.last_pc));
        eprintln!("wait_reason     = {:?}", report.wait_reason);
        eprintln!("unknown_op      = {:?}", report.unknown_op.map(|(p, w)| format!("{p:#x}: {w:#010x}")));
        eprintln!("unresolved_spin = {:?}", report.unresolved_spin);
        eprintln!("done-flag[0x9070] = {:#x}", proc.bus.load_local32(0x9070));

        // The completion fired: the local done-flag is set.
        assert_ne!(proc.bus.load_local32(0x9070), 0, "completion delivered the done-flag");
        // Boot progressed OUT of the dispatcher recursion (0xd7f0..0xd848): it
        // either reached idle, hit a new decode/opcode wall, or a spin elsewhere,
        // but it is no longer looping in the scheduler.
        let in_recursion = (0xd7f0..=0xd848).contains(&report.last_pc);
        assert!(
            !in_recursion || report.reached_idle,
            "boot left the task_dispatcher recursion (last_pc={:#x})",
            report.last_pc
        );
    }
```

- [ ] **Step 4: Run the integration test and record where the real path stops**

Run: `cargo test --lib m2c_boot_completion_advances_past_recursion -- --nocapture`
Expected: PASS. Read the printed `last_pc` / `unknown_op` / `reached_idle`. **Record the exact stop** (PC, opcode word if `unknown_op`, symbol) in the commit message and carry it into Task 5 -- it determines the next wall. (If the firmware is absent, the test skips; note that and defer Task 5's characterization to a machine with the `.sbin`.)

**If the assertion FAILS** (boot stays in the `0xd7f0..0xd848` recursion, or the done-flag is not set): this is most likely a genuine FINDING, not a code bug -- the real scheduler state selects a different task than force-done's artificial switch did, so the completion targets the wrong done-flag, OR the real path re-arms differently. Do NOT patch the assertion or guess a fix. Re-run the discovery probe and the `poll_map`/`force_done` probes to see which task the real path blocks on, write the divergence into the findings doc, and STOP for a check-in with the human -- the model may need the completion target derived per-post rather than from the single scheduler global. (This is exactly the "completion-address from message contents" projection the spec flags.)

- [ ] **Step 5: Commit**

```bash
git add src/firmware/mod.rs
git commit -m "feat(#140): wire the completion model into boot_to_idle (opt-in)

FirmwareProcessor gains a default-disabled host_mailbox, ticked each
instruction in boot_to_idle; enable_host_mailbox() turns it on for the
real boot path. Integration test asserts boot leaves the task_dispatcher
recursion and the done-flag is set.

Real-path stop observed: <PC / opcode / reached_idle from Step 4>.

Generated using Claude Code."
```

---

### Task 5: Next-wall follow-through

**Files:**
- Depends on Task 4's recorded stop. Likely `src/firmware/xtensa/decode/` + `interp/` (a decode/opcode gap) or a short note if it reached idle.

**Interfaces:**
- Consumes: the exact stop PC / opcode word recorded in Task 4.

This task is deliberately branch-structured because the real path's next wall is only known after Task 4 runs. Follow superpowers:systematic-debugging: characterize the stop before touching anything.

- [ ] **Step 1: Classify the Task 4 stop**

From Task 4's output, the `last_pc` + `unknown_op` fall into one of:

- **(A) `reached_idle == true`** -- boot reached its command-loop `waiti`. The mechanism is complete; there is no next wall. Skip to Step 4 (document + close).
- **(B) `unknown_op` at an `xt_format1` bundle** -- `unknown_op.1` has op0 nibble `0xE` (e.g. word `0x1d020cfe` at `0xd903`). This is the 3-slot FLIX format, a **separate, already-designed feature** (`docs/superpowers/specs/2026-07-05-m2c-flix-bundle-decode-design.md`). It is out of scope for THIS plan (it is its own ISA feature with its own design). Record it and STOP for a check-in: report to the human that the real path reached the `xt_format1` wall and recommend executing the FLIX-decode plan next.
- **(C) a bounded single-opcode/decode gap** (an `unknown_op` that is a normal single instruction, not a FLIX bundle) -- clear it with a TDD cycle in `decode/` + `interp/` following the existing pattern (see how `S32c1i` was added: `decode/mem.rs`, `interp/mem.rs`, `decode/mod.rs` `Op` + `max_ar`). Derive semantics from `../llvm-aie` TableGen / `xtensa-modules.c`; byte-verify the encoding.
- **(D) an unexpected wall** (a sysstub spin on a new address, an MMU fault, a different recursion) -- do NOT guess a fix. Characterize it (which address, which PC, what the firmware polls) and STOP for a check-in: a new wall is new scope for the human to weigh.

- [ ] **Step 2: Act on the classification**

- (A): no code change.
- (B) or (D): no code change in this plan; write the characterization into the findings doc (Step 3) and surface the check-in.
- (C): implement the decode + interp for the one opcode, TDD (failing decode test -> decode arm -> failing interp test -> interp arm -> re-run the Task 4 integration test to confirm boot advances further). Show the byte-verified encoding in the test, as `decodes_s32c1i` does.

- [ ] **Step 3: Update the findings doc**

Append a short section to `docs/superpowers/findings/2026-07-06-iter18-phase0-interrupt-wiring.md`: the completion model landed, the real-path stop (PC/opcode), and the classification (A/B/C/D) with the next action. This is the durable record of where the real boot path now reaches.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(#140): follow the real boot path past the completion to <stop>

<one line: reached idle / cleared opcode X / characterized wall Y and
handed off>. Findings updated with the real-path stop.

Generated using Claude Code."
```

---

## Notes for the executor

- **Clippy:** the `#[derive(Default)]` on `CompletionAgent`, `HostMailboxConsumer`, and `HostMailbox` plus a `new()` that calls `Self::default()` satisfies `clippy::new_without_default`. Keep both.
- **Do not build projected layers.** If you find yourself modeling the SRAM ring buffer, the `0x1D` magic header, the x2i ring, or backing the `0x08a00000` gap, stop -- those are projection-marked (spec) and no current path needs them.
- **The mailbox descriptor is already backed** by `Bus.mailbox`; you never touch `mmio.rs`.
- **The discovery probe** `m2c_probe_i2x_ring_locate` (already committed) is the tool that pinned the descriptor layout; re-run it (`XDNA_FW_PROBE=1 cargo test --lib m2c_probe_i2x_ring_locate -- --nocapture`) if you need to re-confirm the register burst.
