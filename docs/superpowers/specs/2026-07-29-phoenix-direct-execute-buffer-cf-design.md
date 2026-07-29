# Phoenix Direct `EXECUTE_BUFFER_CF` -- Design

**Date:** 2026-07-29

**Status:** Architecture approved; written-spec review pending

## Purpose

Expand the proven Phoenix firmware contract from the default
`CHAIN_EXEC_NPU` execution envelope to the pinned driver's supported direct
`EXECUTE_BUFFER_CF` path.

The proof must change only the driver module parameter that selects the
envelope. The frozen Chess and Peano `add_one_using_dma` artifacts, normal XRT
userspace, unmodified pinned driver and firmware, guest boundary, array work,
outputs, and teardown remain the same.

## Derived Boundary

The source of this slice is the pinned primary driver commit
`216cefececd74effcd7a88350c71b99f5ef9a215`:

- `aie2_ctx.c` defines the supported `force_cmdlist` module parameter, with
  default `true`.
- A non-chain `ERT_START_CU` command uses `aie2_execbuf()` when
  `force_cmdlist=false`.
- `aie2_init_exec_cu_req()` emits `MSG_OP_EXECUTE_BUFFER_CF` (`0x00c`).
- `struct execute_buffer_req` is exactly 80 bytes: one `cu_idx` word followed
  by 19 payload words.
- The constructor copies only the command length and does not initialize the
  unused fixed-size payload tail. Those bytes are explicitly unconstrained;
  neither firmware nor the emulator may depend on them.
- The response is one four-byte status word.

The frozen xclbin-only XRT flow emits `ERT_START_CU`. With the default module
setting the driver wraps that command in `CHAIN_EXEC_NPU` (`0x018`); with
`force_cmdlist=false` it must emit `EXECUTE_BUFFER_CF` (`0x00c`). Therefore the
existing frozen artifact isolates the firmware envelope without changing
kernel or array behavior.

## Selected Approach

Extend the existing runner with one explicit mode:

```text
./scripts/phoenix-vfio-user-qemu.sh --run-frozen-direct chess
./scripts/phoenix-vfio-user-qemu.sh --run-frozen-direct peano
```

The existing `--run-frozen chess|peano` behavior remains unchanged and proves
the default chained envelope. The new mode reuses its tuple validation,
initramfs construction, QEMU invocation, output checks, and evidence layout.

The guest init script receives the selected mode through the generated
initramfs. Direct mode loads the same `amdxdna.ko` with:

```text
tdr_timeout_ms=0 force_cmdlist=N
```

It then reads `/sys/module/amdxdna/parameters/force_cmdlist` and fails unless
the live value is `N`. Default chained mode continues to require `Y`, so each
result proves which driver path actually ran.

A generic execution-mode configuration and a second standalone runner are
rejected for this slice. They add flexibility or duplicate tuple logic without
expanding the proved hardware contract.

## Proof Order

### 1. In-process RED test

Extend the existing configured-CU firmware integration test with a direct
envelope case derived from the pinned driver layout:

- opcode `0x00c`;
- 80-byte body;
- CU index `0`;
- the same frozen 15-word register map;
- a nonzero sentinel in the remaining four unconstrained payload words, proving
  that firmware ignores the driver-defined don't-care tail; and
- one-word success response.

The test must retain the existing assertions for genuine firmware request
consumption, response publication, configured PDI effects, array execution,
ordered output `2..=65`, and completion-token consumption.

No emulator behavior changes are justified unless this test fails. If it
fails, diagnosis starts at the observed unmodified-firmware transition and any
fix must reproduce that derived behavior rather than bypassing firmware.

### 2. Guest boundary

Run both frozen compilers through:

```text
frozen test.exe
  -> normal XRT XDNA plugin
  -> unmodified pinned amdxdna.ko with force_cmdlist=N
  -> vfio-user PCI function
  -> unmodified pinned npu.dev.sbin
  -> shared array emulator
  -> firmware I2X and MSI-X
  -> driver fence and XRT completion
```

The runner must require all existing frozen-run checks plus:

- guest log records `force_cmdlist=N`;
- driver dynamic-debug log contains an `opcode 0xc size 80` request;
- the matching context channel contains an `opcode 0xc size 4` response;
- that context channel contains no `opcode 0x18` request;
- all 64 outputs are the ordered range `2..65`; and
- `DESTROY_CONTEXT` completes successfully.

Checking both request sizes distinguishes send from response while preserving
the unmodified driver's own transport log as the opcode oracle.

## Failure and Evidence

Every run retains its normal directory under:

```text
build/experiments/phoenix-vfio-user/<run-id>/
```

On failure, the existing bounded guest wait and cleanup remain responsible for
stopping QEMU and the vfio-user server. Logs and tuple hashes remain intact.
The slice does not add physical NPU or GPU access.

Known guest-kernel lockdep warnings remain documented observations, not success
criteria. A new timeout, firmware error, wrong output, missing interrupt, or
failed teardown is a contract failure.

## Acceptance

This slice is complete only when:

1. the direct in-process firmware integration test passes;
2. the frozen Chess guest run passes;
3. the frozen Peano guest run passes;
4. each guest run proves `0x00c` request and response traffic and excludes a
   `0x018` execution request;
5. the full library suite passes;
6. formatting and shell syntax checks pass; and
7. the worktree contains a durable evidence summary identifying both run
   directories and the pinned tuple.

## Explicitly Out of Scope

- `EXEC_DPU` (`0x010`) and ELF/module-style `ERT_START_NPU`;
- timeout, restart, suspend/resume, and other lifecycle recovery;
- telemetry, column-status, debug-BO, and AIE register operations;
- changes to the driver, firmware binary, XRT, QEMU, or frozen artifacts;
- a generic mailbox stimulus framework; and
- physical-hardware validation.

`EXEC_DPU` is the immediate next direct-execution slice, but it requires a
separately pinned ELF/module artifact so that its envelope and expected array
behavior can be proved independently.
