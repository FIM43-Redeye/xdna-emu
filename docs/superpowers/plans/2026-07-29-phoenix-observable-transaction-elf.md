# Phoenix Observable Transaction-ELF `EXEC_DPU` Implementation Plan

**Goal:** Prove both compiler variants of mlir-aie's
`add_one_objFifo_elf` through normal XRT, the pinned open driver, unmodified
Phoenix firmware, and the shared AIE array, with real runtime relocations and
the ordered output `42..=105`.

**Architecture:** Extend the existing Phoenix vfio-user KVM runner and guest
init with one explicit `--run-pinned-elf chess|peano` mode. Reuse the upstream
host executable and generated transaction ELF; add no custom recipe, ELF
patcher, synthetic command producer, or second runner.

**Execution:** The primary agent executes serially in the existing
`firmware-priors` worktree. No subagents. Harness behavior follows a witnessed
unsupported-mode RED. Emulator behavior changes only after an authentic
failure is reproduced by a focused in-process RED guard.

**Approved design:**
[`2026-07-29-phoenix-observable-transaction-elf-design.md`](../specs/2026-07-29-phoenix-observable-transaction-elf-design.md)

## Test Environment

The linked worktree cannot infer sibling toolchain paths. Rust gates run after
sourcing `/home/triple/npu-work/toolchain-build/activate-npu-env.sh` with:

```bash
MLIR_AIE_PATH=/home/triple/npu-work/mlir-aie
LLVM_AIE_PATH=/home/triple/npu-work/llvm-aie
AIE_RT_PATH=/home/triple/npu-work/aie-rt/driver/src
```

The last verified baseline at the design boundary was 4,269 passed, 32
ignored, and 0 failed. The targeted mlir-aie build and bridge proof run
locally; this single test does not justify remote Halo compilation.

## Task 1: Establish RED and Pin the Upstream Artifact

**Files:**

- Generated outside this repository:
  `../mlir-aie/build/test/npu-xrt/add_one_objFifo_elf/`

### RED

- [ ] Run:

```bash
./scripts/phoenix-vfio-user-qemu.sh --run-pinned-elf chess
```

- [ ] Require exit 2 with the current usage error. This is the acceptance RED:
  the pinned driver boundary has no transaction-ELF data-plane mode.

### Artifact gate

- [ ] Rebuild and functionally revalidate only the upstream ELF test:

```bash
./scripts/emu-bridge-test.sh --compile --no-trace \
  '^add_one_objFifo_elf$'
```

- [ ] Require Chess and Peano compile, physical-NPU, and emulator results to
  pass before pinning the artifacts.
- [ ] Inspect each `insts.elf` with `readelf` and require `.ctrltext` plus
  runtime relocation records for the input and output arguments.
- [ ] Record the exact mlir-aie commit and SHA-256 values for:
  - shared `test.exe`;
  - Chess `aie.xclbin` and `insts.elf`; and
  - Peano `aie.xclbin` and `insts.elf`.
- [ ] Do not modify, clean, or commit the existing intentional mlir-aie
  working-tree changes.

## Task 2: Add the Explicit Pinned-ELF Guest Mode

**Files:**

- Modify: `scripts/phoenix-vfio-user-qemu.sh`
- Modify: `tools/phoenix-vfio-user/guest-driver-probe-init.sh`

### GREEN

- [ ] Accept exactly `--run-pinned-elf chess|peano`.
- [ ] Reuse the existing compiler validation, XRT dependency copying,
  initramfs, QEMU, vfio-user, timeout, cleanup, and evidence paths.
- [ ] Require the committed hashes from Task 1 before constructing the guest.
- [ ] Copy the selected upstream `test.exe`, `aie.xclbin`, and `insts.elf`
  into one `/run-elf` guest directory.
- [ ] Load `amdxdna.ko` with `tdr_timeout_ms=0 force_cmdlist=N`.
- [ ] Read back and require `force_cmdlist=N`.
- [ ] Invoke the unmodified host as:

```bash
/run-elf/test.exe -x /run-elf/aie.xclbin \
  -k MLIR_AIE -i /run-elf/insts.elf
```

- [ ] Require `PASS!` and exactly 64 ordered output lines for `42..105`.
- [ ] Require one matched direct `EXEC_DPU` transaction:
  - 160-byte opcode-`0x10` request;
  - four-byte opcode-`0x10` response with the same message ID;
  - no 24-byte opcode-`0x18` execution request; and
  - no 80-byte opcode-`0x0c` execution request.
- [ ] Preserve the source-37 publication, context MSI-X, and matched
  `DESTROY_CONTEXT` checks.
- [ ] Record the compiler and all artifact hashes in `tuple.txt`.
- [ ] Run:

```bash
bash -n scripts/phoenix-vfio-user-qemu.sh
bash -n tools/phoenix-vfio-user/guest-driver-probe-init.sh
```

- [ ] Commit the green harness change.

## Task 3: Cross the Authentic Firmware Boundary

- [ ] Run Chess:

```bash
./scripts/phoenix-vfio-user-qemu.sh --run-pinned-elf chess
```

- [ ] If the authentic run fails inside emulated behavior:
  1. retain the complete evidence directory;
  2. identify the first firmware-visible divergence;
  3. add the smallest in-process failing guard using the exact generated
     xclbin and patched transaction bytes; and
  4. derive the fix from the open driver, XRT, mlir-aie/aiebu, or observed
     unmodified firmware behavior.
- [ ] Do not weaken the output or firmware-lifecycle assertions and do not add
  a response shim.
- [ ] Once Chess passes, run Peano:

```bash
./scripts/phoenix-vfio-user-qemu.sh --run-pinned-elf peano
```

- [ ] Require both evidence directories to satisfy the complete acceptance
  contract independently.

## Task 4: Close the Slice

**Files:**

- Update:
  `docs/superpowers/specs/2026-07-29-phoenix-observable-transaction-elf-design.md`
- Update: `docs/arch/firmware-array-plugin-wiring.md`
- Update only if its classified state changes:
  `docs/fidelity-gaps/host-firmware-dispatch.md`
- Update only if its headline changes: `docs/known-fidelity-gaps.md`

- [ ] Record both KVM evidence directories, artifact hashes, matched opcode
  traffic, outputs, interrupts, and teardown.
- [ ] Run:

```bash
cargo fmt --all -- --check
cargo test --lib
git diff --check
```

- [ ] Confirm the worktree contains only intended source and documentation
  changes; generated mlir-aie and KVM artifacts remain outside the commit.
- [ ] Commit the evidence update last.
