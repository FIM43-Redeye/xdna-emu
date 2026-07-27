# Phoenix Mailbox / Management-Controller Slice Implementation Plan

> **Required workflow:** Use `superpowers:subagent-driven-development` to
> implement task-by-task with test-driven development. Keep the
> BAR4-to-source-46 connection absent.

**Goal:** Model the proven Phoenix host mailbox registers and the proven
management-controller-to-Xtensa path as two independently testable halves.

**Architecture:** `Bus` owns one concrete BAR4 mailbox-register block and one
concrete management interrupt controller. BAR2 continues to use the existing
firmware-programmed SRAM aliases. `Cpu::step_on` consumes an explicitly queued
controller assertion before its existing architectural interrupt check. No
host tail write selects a controller source.

**Tech stack:** Rust 2021, existing firmware interpreter, Cargo unit and
firmware-gated integration tests.

## Constraints

- RED before GREEN for every behavior change.
- Do not write `cpu.interrupt |= 1` outside the architectural CPU bridge.
- Do not connect `0x030ec000` to source 46.
- Do not interpret `0x272009xx`.
- Do not invent multi-source priority, disabled-source latching, or
  edge/level semantics.
- Do not reuse the AIE tile L1/L2 interrupt controllers.
- Do not add a C ABI or change the XRT plugin.
- Preserve mailbox-access probe logging for the existing RE tools.
- Stop and discuss any real-firmware failure that requires behavior outside
  the approved design.

Use this environment prefix for Cargo commands in the isolated worktree:

```bash
PATH=/home/triple/npu-work/mlir-aie/ironenv/bin:$PATH \
PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python \
MLIR_AIE_PATH=/home/triple/npu-work/mlir-aie \
AIE_RT_PATH=/home/triple/npu-work/aie-rt/driver/src \
LLVM_AIE_PATH=/home/triple/npu-work/llvm-aie \
TABLEGEN_210_PREFIX=/home/triple/npu-work/llvm-aie/build
```

## Task 1: Phoenix Host Mailbox Registers

**Files:**

- Add: `src/firmware/phoenix_mailbox.rs`
- Modify: `src/firmware/mod.rs`
- Modify: `src/firmware/mmio.rs`

**Interface:**

```rust
pub fn host_load32(&self, device_address: u32) -> u32;
pub fn host_store32(&mut self, device_address: u32, value: u32);
```

**BAR4 words:**

- X2I tail: `0x030e_c000`
- X2I head: `0x030e_c004`
- I2X tail: `0x030e_d000`
- I2X head: `0x030e_d004`
- I2X status: `0x030e_d008`

- [ ] Add a RED `Bus` test proving:
  - a host write to configured BAR2 X2I SRAM reaches `local_data`;
  - the five published BAR4 words retain independent values;
  - firmware-side `data_load32` observes a host BAR4 write;
  - host-side `host_load32` observes a firmware BAR4 write.
- [ ] Run:

  ```bash
  cargo test --lib host_device_access_shares_bar2_and_bar4_state
  ```

  Expected RED: `host_load32` / `host_store32` do not exist and BAR4 is a
  `SysStub`.
- [ ] Add a concrete `PhoenixMailboxRegisters` with five `u32` fields and
  exact-address `read32` / `write32` methods. No trait, map, event queue, or
  generalized register framework.
- [ ] Store it on `Bus`. Intercept the five addresses in CPU 32-bit
  region-load/store before `SysStub`; keep access-probe recording.
- [ ] Add `Bus::host_load32` / `host_store32`: route the five BAR4 words first,
  then delegate to existing SRAM-alias access.
- [ ] Run the targeted test GREEN and the existing SRAM-alias tests.
- [ ] Commit:

  ```bash
  git commit -m "feat(firmware): model Phoenix mailbox registers"
  ```

## Task 2: Single-Source Management Controller

**Files:**

- Add: `src/firmware/management_controller.rs`
- Modify: `src/firmware/mod.rs`
- Modify: `src/firmware/mmio.rs`

**Interface:**

```rust
pub(crate) fn assert_management_source(&mut self, source: u8) -> bool;
pub(crate) fn take_management_irq_assertion(&mut self) -> bool;
```

**Controller registers:**

- Enable bank `n`: `0x2720_0300 + 4*n`, for `n = 0..3`
- Status/acknowledgement bank `n`: `0x2720_03b0 + 4*n`, for `n = 0..3`
- Active source: `0x2720_03c4`
- Source mapping: `bank = source >> 5`,
  `bit = 1 << (source & 31)`
- An inactive active-source read returns the existing reset value `0`; this is
  an emulator default, not a claim about a hardware sentinel.
- Sources outside the four-bank range are rejected.

- [ ] Add pure RED tests for:
  - disabled source 46 is rejected;
  - enabled source 46 becomes status bank 1 bit 14 and active source 46;
  - acknowledgement clears the source but not its enable bit;
  - a competing source is rejected while one is active.
- [ ] Add a RED `Bus` routing test proving firmware MMIO at
  `0x27200304`, `0x272003b4`, and `0x272003c4` uses the controller state.
- [ ] Run:

  ```bash
  cargo test --lib management_controller
  ```

  Expected RED: the controller type and Bus seam do not exist.
- [ ] Implement one concrete controller with:
  - four enable words;
  - four status words;
  - `Option<u8>` active source;
  - one queued aggregate assertion.
- [ ] Route 32-bit enable, status/ack, and active-source MMIO through it.
  Preserve raw backing for every other `0x272003xx` register and all
  `0x272009xx` registers.
- [ ] Remove the old blanket `MAILBOX_IRQ_ACK_BASE..END` W1C shortcut and
  replace its test with the source lifecycle test.
- [ ] Add a negative test: `host_store32(0x030ec000, tail)` leaves controller
  state and its aggregate queue untouched.
- [ ] Run targeted tests GREEN, then:

  ```bash
  cargo test --lib firmware::mmio::tests
  ```

- [ ] Commit:

  ```bash
  git commit -m "feat(firmware): model management interrupt controller"
  ```

## Task 3: Controller-to-Xtensa Bridge

**Files:**

- Modify: `src/firmware/mmio.rs`
- Modify: `src/firmware/xtensa/interp/mod.rs`

- [ ] Add a RED CPU test which:
  - enables controller source 46 through MMIO;
  - explicitly asserts it through `Bus`;
  - enables Xtensa bit 0;
  - calls `Cpu::step`;
  - expects the existing level-1 exception path;
  - never mutates `Cpu::interrupt` directly.
- [ ] Run:

  ```bash
  cargo test --lib controller_source_reaches_xtensa_level1
  ```

  Expected RED: the queued controller assertion is not consumed by the CPU.
- [ ] Add one `CpuBus` forwarding method for consuming the queued assertion.
- [ ] At the start of `Cpu::step_on`, consume it and set architectural
  `INTERRUPT` bit 0 before calling `interrupt_deliverable`.
- [ ] Keep exception entry, masking, `waiti`, and `rfe` unchanged.
- [ ] Run the new test and existing `interrupt_`, `waiti`, and `rfe` tests
  GREEN.
- [ ] Commit:

  ```bash
  git commit -m "feat(firmware): route management IRQs to Xtensa"
  ```

## Task 4: Pinned Firmware Source-46 Round Trip

**Files:**

- Modify: `src/firmware/boot_tests/guards.rs`

- [ ] Add a firmware-gated integration guard:
  1. load the pinned image;
  2. boot naturally to idle;
  3. verify source 46 and Xtensa bit 0 are enabled;
  4. explicitly assert controller source 46;
  5. resume until the next natural idle;
  6. verify active source, controller status, and CPU pending bit are clear.
- [ ] Run:

  ```bash
  XDNA_FIRMWARE=/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin \
    cargo test --lib m2c_source_46_returns_to_idle -- --nocapture
  ```

  This guard would fail before Task 3 because there is no controller delivery.
  After Task 3, it should pass without additional production behavior.
- [ ] If the test exposes a mapping/decode/controller semantic outside the
  approved slice, stop and report the exact frontier. Do not force a handler,
  patch firmware state, or connect BAR4 tail.
- [ ] If GREEN, commit the guard:

  ```bash
  git commit -m "test(firmware): validate source 46 handler path"
  ```

## Task 5: Documentation and Full Verification

**Files:**

- Modify: `docs/arch/firmware-array-plugin-wiring.md`
- Modify: `docs/fidelity-gaps/host-firmware-dispatch.md`

- [ ] Record that BAR4 register state and the controller-to-Xtensa half exist,
  while the causal BAR4-tail-to-controller edge remains gated.
- [ ] Do not claim a mailbox round trip, virtual PCI support, driver
  acceptance, MSI-X, PSP, or SMU completion.
- [ ] Run:

  ```bash
  cargo fmt --all --check
  cargo test -p xdna-emu-ffi
  cargo test --lib
  git diff --check
  ```

- [ ] Verify `git status --short` contains only the intended changes.
- [ ] Commit:

  ```bash
  git commit -m "docs(firmware): record disconnected controller seam"
  ```
