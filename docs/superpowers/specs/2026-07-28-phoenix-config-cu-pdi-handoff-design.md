# Phoenix CONFIG_CU / PDI Handoff -- Design

**Date:** 2026-07-28

**Status:** Implemented and verified

## Purpose

Cross the first authentic Phoenix application-load boundary: give the
unmodified `1502_00` firmware the same CU/PDI state that XRT and the open
driver provide, then prove that the firmware programs the interpreter
engine's sole `DeviceState`.

This slice ends when the registered PDI has populated the assigned physical
array column. Kernel execution, output correctness, and the downstream
firmware completion response remain the immediately following slice unless
they occur naturally during this work.

## Pinned Tuple

- Phoenix open driver:
  `216cefececd74effcd7a88350c71b99f5ef9a215`.
- Phoenix firmware `1502_00/npu.dev.sbin`, version `5.5.391`, SHA-256
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`.
- Frozen Chess `add_one_using_dma` xclbin and instruction stream from
  [`2026-07-27-phoenix-primary-kernel-corpus-freeze.md`](../findings/2026-07-27-phoenix-primary-kernel-corpus-freeze.md).

Changing any member of this tuple is a new compatibility case, not silent
continuation of this proof.

## Root-Cause Correction

The current driver-shaped guard performs:

```text
CREATE_CONTEXT
  -> MAP_HOST_BUFFER
  -> CHAIN_EXEC_NPU
```

It omits the normal CU-configuration step and puts the command slot at the
context-heap base.

The production path is different:

1. XRT extracts the PDI selected by the kernel ID from the xclbin.
2. XRT allocates a device BO, copies the complete PDI into it, and supplies
   that BO plus the kernel's `functional` value through the CU-config ioctl.
3. The open driver sends context opcode `CONFIG_CU` (`0x11`). Its 132-byte
   body contains the CU count and 32 packed entries. For NPU1, the PDI address
   is represented in 32 KiB units; bits `16:0` carry the address and bits
   `24:17` carry the CU function.
4. The firmware stores that table in APP-ERT context state.
5. `CHAIN_EXEC_NPU` resolves the selected `cu_idx` through the table and only
   then enters the application/PDI loader.

The firmware path makes the present failure causal rather than speculative.
Its `CONFIG_CU` handler marks and fills the per-context table. Its later CU
lookup rejects an unconfigured table and returns the literal at virtual
`0x08b00010`, whose value is `0x03000003`,
`AIE2_STATUS_APP_LOAD_PDI_FAIL`.

The observed mode-3 management-DMA operation from context-heap base to local
`0x0007d000` occurs before that lookup. It stages the 16 KiB command-chain
window; it is not a PDI transfer. The existing guard therefore has not tested
the firmware's PDI parser or array-programming path.

## Evidence Anchors

- XRT PDI selection and CU-function ownership:
  `../xdna-driver/src/shim/hwctx.cpp`, `xclbin_parser`.
- XRT PDI BO allocation, copy, and CU-config ioctl:
  `../xdna-driver/src/shim/kmq/hwctx.cpp`, `hwctx_kmq`.
- Production request construction:
  `../xdna-driver/src/driver/amdxdna/aie2_message.c`,
  `aie2_config_cu`.
- Wire fields and request size:
  `../xdna-driver/src/driver/amdxdna/aie2_msg_priv.h`,
  `AIE2_MSG_CFG_CU_*` and `config_cu_req`.
- NPU1 device-memory address unit:
  `../xdna-driver/src/driver/amdxdna/npu1_regs.c`,
  `dev_npu1_info.dev_mem_buf_shift`.
- APP-ERT `CONFIG_CU`, CU lookup, and application-loader path: pinned
  `1502_00/npu.dev.sbin` routines at virtual `0x08b04e40`,
  `0x08b0a204`, `0x08b0e6e4`, and `0x08b04638`.
- Current incomplete stimulus:
  `src/firmware/boot_tests/guards.rs`,
  `m2c_unconfigured_cu_fails_before_pdi_loader`.

## Governing Correctness Rule

Every state transition must be caused by an operation emitted by the
unmodified driver, firmware, PDI, or array. Functional transport and
scheduling abstractions may carry that operation, but may not decide its
meaning or manufacture its completion.

For this slice:

| Surface | Provenance and treatment |
|---|---|
| PDI bytes and CU metadata | Read from the frozen real xclbin |
| Lifecycle and request packing | Recreated from XRT and the pinned open driver |
| PDI parsing and relocation | Executed by the unmodified signed firmware |
| Array writes | Firmware's real MMIO addresses and values routed to `DeviceState` |
| Register behavior | Existing aie-rt / AM025-derived device model |
| Device BO / DDR backing | Functionally equivalent registered `HostMemory` |
| Firmware/AIE scheduling | Existing boundary-driven functional pump; no timing claim |
| Management-DMA timing and controller priority | Existing explicit fidelity deferrals |

The harness must not call `load_pdi`, `apply_cdo`, directly enable a core, or
write a firmware result. A configured positive case and an unconfigured
negative case pin the causal difference.

## Exact Test Data Flow

```text
frozen add_one_using_dma xclbin
  -> Xclbin and AiePartition parsers extract the primary PDI
  -> unchanged PDI bytes copied to a 32 KiB-aligned context-heap address
  -> CONFIG_CU registers the PDI address and xclbin CU function
  -> firmware publishes and host consumes the real CONFIG_CU response
  -> CHAIN_EXEC_NPU is submitted from a separate heap address
  -> firmware stages and parses the command chain
  -> firmware resolves CU 0 and loads the registered PDI
  -> PDI logical column 0 is relocated once to assigned physical column 1
  -> firmware MMIO populates shared array program/data/core state
```

Use the existing xclbin and AIE-partition parsers for the PDI. Read the
fixture's `functional` value from its embedded metadata with the smallest
test-only extraction needed; do not add a general metadata framework for one
field.

The NPU1 32 KiB shift and CU bit fields are wire facts from the pinned open
driver. Keep them adjacent to request construction with their source named.
The PDI address must be checked for alignment and field fit before packing.

## Test Structure

Preserve the current no-`CONFIG_CU` behavior as a negative regression:

- the request completes through real firmware;
- failed-command index remains zero;
- failed-command status remains `APP_LOAD_PDI_FAIL`; and
- no array program is loaded.

Add a positive guard using the same boot, initialization, context, heap, and
command slot, differing only in the required production lifecycle:

1. write the PDI into an aligned, non-overlapping heap range;
2. send and consume `CONFIG_CU`;
3. submit the chain from a different heap range; and
4. pump only real firmware and modeled array progress.

The positive guard proves:

- `CONFIG_CU` receives its genuine status-zero response;
- the PDI bytes remain unchanged in registered host memory;
- the configured chain does not take the unconfigured
  `APP_LOAD_PDI_FAIL` branch;
- program memory, data memory, and core configuration appear in physical
  column 1;
- physical column 0 remains tile-free;
- no unassigned column receives PDI tile writes; and
- no direct loader, synthetic core launch, or synthetic response participates.

Record the resulting downstream stop exactly. A natural context response or
correct kernel output is welcome farther progress, but neither is required to
claim this PDI-handoff slice. An unresolved poll, unknown instruction,
unmodeled peripheral, or engine-state mismatch becomes the next named
boundary rather than a reason to insert a shortcut.

## Implementation Boundary

Start with test stimulus and observation only. It is possible that the
existing firmware, management-DMA, host-memory, and array-MMIO paths already
complete the PDI handoff once supplied the missing lifecycle.

If the positive test fails, trace the failure from the first divergent
firmware operation. Add production behavior only when an open toolchain source
or hardware observation establishes the missing peripheral contract. Put the
fix at the shared hardware seam used by every caller, not in the guard.

Do not modify the XRT plugin, public FFI, direct xclbin loader, or
`NpuExecutor` in this slice. Those paths remain independent controls until the
interior firmware-to-array path is proved.

Do not call `InterpreterEngine::sync_cores_from_device()` merely because the
test observes a loaded PDI. The following core-execution slice must make
engine execution follow causally from the firmware's actual `Core_Control`
writes. If the existing split between tile core state and interpreter
bookkeeping blocks that, design and fix that shared state transition
explicitly.

## Timing Position

The runtime pump's current policy--run firmware to a natural boundary, then
advance one AIE cycle--is a functional scheduler, not a clock model.

One plausible but unverified hypothesis is that the management processor is
clocked from either an array-related clock or FCLK. It is recorded only to
guide later measurement. No management/AIE clock ratio, fixed latency, or
cycle-accuracy claim may be derived from it.

Timing work starts only after an authoritative clock declaration or a hardware
measurement can discriminate candidate relationships. Changing the eventual
cadence must remain local to scheduling and must not change the lifecycle,
PDI bytes, firmware control flow, or array side effects proved here.

## TDD Sequence

1. Rename or split the existing guard so its unconfigured-CU failure is an
   explicit negative contract.
2. Add the smallest stateful context-mailbox test helper needed to send two
   ordered requests and consume their responses.
3. Add the positive test using the real xclbin PDI and metadata, then run it
   before any production change.
4. If it reaches the handoff without a production change, keep the result:
   the bug was incomplete test stimulus.
5. If it fails, capture the exact firmware PC, request state, management-DMA
   state, array accesses, and stop reason before designing a fix.
6. Implement only the smallest source-derived shared-seam correction, rerun
   the positive and negative guards, and preserve the newly exposed boundary.

## Stop Conditions

Pause before implementation or return to design if progress would require:

- parsing or applying the PDI outside firmware;
- inferring CU metadata that is present in the xclbin;
- manufacturing a firmware success or completion event;
- using physical column 0 as an AIE tile column;
- relocating logical PDI coordinates more than once;
- launching cores from a test-only synchronization call;
- assigning a management/AIE clock relationship from the current hunch; or
- changing the plugin before the interior path is independently green.

## Outcome

The configured and unconfigured guards now differ only at the production
`CONFIG_CU` lifecycle. The negative control still returns
`APP_LOAD_PDI_FAIL` before the loader. The configured case runs the signed
firmware's genuine PDI loader, populates program/data/core state only in
physical column 1, and stops with the request retained pending array
execution. The root fixes are shared-seam corrections: `$PS1` signed-body
extent, registered device-heap CPU views, the firmware-programmed translated
host view, both firmware array windows, and aligned management-DMA writes into
the borrowed device. No PDI application or response is synthesized.

The full library and FFI package gates pass. Array execution through firmware
completion is the next separately designed slice.

## Verification

After each test or source change:

```bash
cargo test --lib <focused-config-cu-or-pdi-test> -- --nocapture
cargo fmt --all --check
git diff --check
```

After the slice is green:

```bash
cargo test --lib
cargo test -p xdna-emu-ffi
```

The bridge test remains a later integration gate because the current plugin
still applies PDIs above firmware. It cannot validate this interior handoff
until that architecture changes in a separately approved slice.
