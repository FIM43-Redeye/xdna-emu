# Phoenix Direct `EXECUTE_BUFFER_CF` Implementation Plan

**Goal:** Prove the frozen Phoenix `add_one_using_dma` command through the
pinned driver's direct `EXECUTE_BUFFER_CF` (`0x00c`) envelope without changing
the kernel, firmware, array behavior, or expected output.

**Architecture:** Reuse the existing configured-CU firmware guard and
vfio-user frozen runner. Add one explicit direct mode which loads the pinned
driver with `force_cmdlist=N`; add no generic mailbox framework or second
runner.

**Execution:** The primary agent executes serially in the existing
`firmware-priors` worktree. No subagents. Production changes follow a witnessed
acceptance RED. If the in-process direct command already passes, preserve that
as a characterization and make no emulator behavior change.

**Approved design:**
[`2026-07-29-phoenix-direct-execute-buffer-cf-design.md`](../specs/2026-07-29-phoenix-direct-execute-buffer-cf-design.md)

## Test Environment

The linked worktree cannot infer sibling toolchain paths. Rust gates run after
sourcing `/home/triple/npu-work/toolchain-build/activate-npu-env.sh` with:

```bash
MLIR_AIE_PATH=/home/triple/npu-work/mlir-aie
LLVM_AIE_PATH=/home/triple/npu-work/llvm-aie
AIE_RT_PATH=/home/triple/npu-work/aie-rt/driver/src
```

Baseline on 2026-07-29: 4,266 passed, 32 ignored, 0 failed.

## Task 1: Establish RED and the Direct Firmware Guard

**Files:**

- Modify and test: `src/firmware/boot_tests/guards.rs`

### RED

- [ ] Run
  `./scripts/phoenix-vfio-user-qemu.sh --run-frozen-direct chess`.
  Require exit 2 with the current usage error. This is the acceptance RED:
  there is no way to select the pinned driver's direct path.
- [ ] Add one focused in-process test for direct `0x00c` execution using:
  - CU index `0`;
  - the frozen Chess PDI, instruction stream, buffers, and 15-word register
    map;
  - four nonzero don't-care words in the fixed 19-word payload; and
  - the literal one-word success response.
- [ ] Run the new test before changing emulator behavior.
  - If it fails at the firmware/array boundary, retain the exact failure and
    diagnose from the unmodified image before writing a fix.
  - If it passes, record that direct behavior already exists and add no
    production emulator code.

The test catches firmware or emulator code that routes only chained commands,
depends on zeroed unused request words, skips real array work, manufactures a
response, or loses completion.

### GREEN

- [ ] Make the smallest test-only refactor needed to share the existing
  configured-CU setup between chained and direct envelopes.
- [ ] If and only if RED exposed a real emulator gap, implement the minimum
  derived fix and rerun the focused test.
- [ ] Rerun the existing chained Chess and Peano firmware tests.
- [ ] Commit the green guard, and any necessary derived fix, separately if
  production behavior changed.

## Task 2: Add the Explicit Guest Direct Mode

**Files:**

- Modify: `scripts/phoenix-vfio-user-qemu.sh`
- Modify: `tools/phoenix-vfio-user/guest-driver-probe-init.sh`

### GREEN

- [ ] Accept exactly `--run-frozen-direct chess|peano` in addition to the
  existing modes.
- [ ] Reuse every existing frozen-artifact, tuple, initramfs, QEMU, output, and
  evidence path. Do not duplicate the runner.
- [ ] Put the selected `cmdlist` or `direct` mode in the guest image.
- [ ] Load `amdxdna.ko` with `force_cmdlist=N` only for direct mode.
- [ ] Read back the live `force_cmdlist` sysfs value and require `N` for direct
  mode and `Y` for normal frozen mode.
- [ ] For direct mode, require the driver log to show:
  - `opcode 0xc size 80` request;
  - `opcode 0xc size 4` response;
  - no `opcode 0x18 size 24` execution request; and
  - matched `DESTROY_CONTEXT` request and response.
- [ ] Keep the existing ordered `2..65`, X2I, I2X/MSI-X, and clean-shutdown
  checks unchanged.
- [ ] Run `bash -n` on both scripts.
- [ ] Commit the harness change.

## Task 3: Full Driver-Boundary Proof

- [ ] Run direct Chess:

```bash
./scripts/phoenix-vfio-user-qemu.sh --run-frozen-direct chess
```

- [ ] Run direct Peano:

```bash
./scripts/phoenix-vfio-user-qemu.sh --run-frozen-direct peano
```

- [ ] Run the normal frozen Chess mode once as a chained-path regression:

```bash
./scripts/phoenix-vfio-user-qemu.sh --run-frozen chess
```

- [ ] If an unexpected failure appears, stop and apply systematic debugging;
  do not weaken assertions or add a responder.

## Task 4: Close the Slice

**Files:**

- Update: `docs/arch/firmware-array-plugin-wiring.md`
- Update only if its claim changes:
  `docs/fidelity-gaps/host-firmware-dispatch.md`
- Update only if its status changes: `docs/known-fidelity-gaps.md`

- [ ] Record both direct run directories, the unchanged pinned tuple, opcode
  request/response evidence, outputs, interrupts, and teardown.
- [ ] Run:

```bash
cargo fmt --all -- --check
cargo test --lib
git diff --check
```

- [ ] Confirm the worktree contains only intended changes and no generated or
  guest-build artifacts.
- [ ] Commit the evidence update last.
