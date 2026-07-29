# Phoenix Firmware / Array Runtime Pump -- Design

**Date:** 2026-07-27

**Status:** Architecture approved; written review pending

> **2026-07-29 topology correction:** The pinned open XRT Phoenix validation
> PDI proved that physical column 0 is not wholly tile-free. It has no shim at
> `(0,0)`, but does contain the DPU-reserved memory tile at row 1 and compute
> tiles at rows 2-5. The compact-array statements below preserve the earlier
> design record and are superseded by `docs/arch/tile-topology.md`.

## Purpose

Join the two independently validated halves of Phoenix execution:

1. unmodified `1502_00` management firmware boots, handles the pinned
   management lifecycle, and programs the interpreter engine's sole
   `DeviceState`; and
2. the interpreter executes the frozen `add_one_using_dma` kernel correctly
   from its existing PDI, ELF, instruction-stream, and host-memory inputs.

The first integrated path must use real firmware to consume the driver-shaped
command and launch the existing array model. It must not use `NpuExecutor` as a
firmware responder, synthesize a completion, clone device state, or invent a
firmware-to-AIE clock ratio.

## Acceptance Path

```text
pinned firmware boot and management initialization
  -> CREATE_CONTEXT and returned context completion queue
  -> MAP_HOST_BUFFER for that firmware context
  -> copy the xclbin-selected PDI into an aligned device-heap BO
  -> CONFIG_CU registers that PDI and the kernel's CU function
  -> driver-shaped CHAIN_EXEC_NPU request
  -> firmware reads the command chain from shared HostMemory
  -> firmware programs and releases the shared array
  -> frozen add_one_using_dma executes
  -> output buffer contains the hardware-validated result
  -> grounded array completion reaches firmware
  -> firmware publishes the context response
  -> request reaches ERT_CMD_STATE_COMPLETED
```

The complete acceptance gate requires both Chess and Peano frozen artifacts
from
[`2026-07-27-phoenix-primary-kernel-corpus-freeze.md`](../findings/2026-07-27-phoenix-primary-kernel-corpus-freeze.md).
An intermediate run may stop after producing the correct output if it exposes
an unmodeled downstream-completion seam. That stop is evidence for the next
hardware measurement, not acceptance and not permission to synthesize the
response.

## Grounded Boundary

Already proved:

- the PSP-visible handoff needed by the Xtensa image;
- natural firmware boot and alive publication;
- host-visible X2I/I2X rings and BAR4 indices;
- explicit management-controller source 46 through the pinned firmware
  handler;
- pinned management initialization through `CREATE_CONTEXT`;
- firmware array MMIO borrowing the engine's sole `DeviceState`;
- the existing engine's `HostMemory`, DMA, stream, lock, core, and TCT models;
- functional hardware/emulator equivalence of the frozen kernel.

Still unproved:

- the causal BAR4-tail-to-source-46 controller transition;
- the corresponding context-X2I publication-to-firmware dispatch transition
  for the queue pair returned by `CREATE_CONTEXT`;
- the management-side landing of an array task-completion token;
- which status transition or interrupt wakes firmware for context-job
  completion;
- the Xtensa-to-AIE clock relationship and dispatch timing.

Management source 46 is the proved host-command handler seam. It must not be
reused for array completion without evidence. The old go-alive line-0
experiments validate CPU interrupt delivery but do not identify the context-job
completion source.

## 1. Correct Phoenix Topology First

The current model conflates two different column domains:

```text
physical control/address envelope:  columns 0..4
real application AIE tile columns:  columns 1..4
```

The five-column envelope is real: firmware reset and clock-control operations
address all five columns. The four-column application array is also real:
mlir-aie's virtualized NPU1 model contains four tile columns, and the pinned
open driver sets Phoenix `first_col = 1`.

Physical column 0 therefore remains addressable for evidenced column-level
management operations, but it contains no modeled AIE shim, memtile, compute
tile, tile DMA engine, or stream-switch column.

### Architecture schema

Extend the generated topology with the physical column at which logical
`tile_map` column 0 is placed. The checked-in architecture artifact remains
self-contained; its generator derives NPU placement from the corresponding
pinned open-driver `first_col` declaration. The existing mlir-aie-to-driver
device-family mapping is explicit and must fail regeneration if the declared
driver placement cannot be found.

`ModelConfig` derives:

- real tile width from the generated `tile_map`;
- physical tile origin from the new placement field; and
- physical control/address extent as `origin + tile_width`.

This removes the current unconditional `topology.columns + 1` rule. That rule
accidentally gives Phoenix the right outer extent while manufacturing a full
column of tiles, and it is wrong for devices such as NPU4 whose driver
`first_col` is zero.

`ArchConfig::columns()` keeps its existing externally visible meaning: the
physical address/control extent. Two small accessors expose the real tile
origin and width. `is_valid_tile` uses that span. `tile_kind` is defined only
for valid tile coordinates and comes from the generated tile map rather than
from a row-only assumption.

### Array storage

`TileArray` stores only real tiles, in physical-column order:

```text
index = (physical_col - tile_col_start) * rows + row
```

`get` and `get_mut` return `None` outside the real tile span. The infallible
internal index helper rejects invalid coordinates instead of wrapping.
Iteration naturally visits only real tiles.

The following parallel storage uses the same compact tile index:

- tile DMA engines;
- control-packet reassemblers;
- bank arbiters; and
- interpreter `CoreState` entries.

The column-clock controller retains state for the complete physical envelope.
Column-level control of physical column 0 remains available. Tile-local module
accesses at column 0 have no target and must not create state.

### Coordinate ownership

There are only two coordinate spaces:

- firmware MMIO and device state use physical coordinates; and
- PDI/CDO/ELF/runtime-sequence inputs use partition-relative coordinates.

`DeviceState::start_col` remains the single logical-to-physical relocation
seam. Firmware MMIO never applies it. Every loader call site must state which
coordinate space it accepts and translate exactly once. In-process xclbin
loading must use Phoenix's physical application origin instead of continuing
to place an NPU1 partition at physical column 0.

No `Absent` `TileKind`, dummy `Tile`, `Vec<Option<Tile>>`, or second shadow
array is introduced.

## 2. Share Host Memory With Firmware

`InterpreterEngine` remains the sole owner of both `DeviceState` and
`HostMemory`. `FirmwareProcessor` continues to own only its Xtensa CPU and
firmware bus.

The existing CPU bus view grows one attached form that borrows all three
objects for a firmware instruction:

```text
Bus + &mut DeviceState + &mut HostMemory
```

Array MMIO keeps routing to `DeviceState`. A management outbound-window target
routes to `HostMemory` only when it falls within a registered host-memory
region. Known system registers and unregistered targets retain the existing
system-register and `SysStub` paths, preserving unresolved-spin detection.

Implement only access widths exercised by the pinned firmware path. The first
tests cover little-endian 32-bit and byte accesses through a configured
outbound window, including a page boundary. Bulk or wider access support waits
for an observed caller.

The acceptance harness registers the context heap, command chain,
instruction stream, input, and output regions in the engine's existing
`HostMemory`. Firmware and array DMA therefore observe the same bytes without a
copy or mirror.

## 3. Boundary-Driven Runtime Pump

Use a small free coordinator function borrowing an existing
`FirmwareProcessor` and `InterpreterEngine`. Do not add an owning runtime type,
trait, factory, thread, mutex, or alternate engine.

One functional pump iteration:

1. run firmware until it reaches a natural `waiti`, an unresolved poll, an
   unknown instruction, or a bounded instruction budget;
2. if the firmware produced the requested response, return it;
3. advance the existing interpreter engine by one AIE cycle;
4. deliver only interrupts or status changes produced by modeled hardware;
5. repeat while either side can make progress.

Firmware work between waits consumes no claimed AIE cycles in this phase. This
is explicitly a functional scheduling policy, not a 1:1 clock model. Later
timing work can change the cadence in this loop without changing ownership,
memory routing, command transport, or architectural side effects.

The pump reports why it stopped:

- context response completed;
- array/output completed but firmware is still waiting;
- unresolved firmware poll;
- unknown firmware instruction;
- engine error or TDR classification; or
- bounded no-progress exhaustion.

These are diagnostic outcomes, not synthetic success states.

### Rejected scheduling alternatives

1. **One Xtensa instruction per AIE cycle.** Rejected because it silently
   invents a 1:1 clock ratio.
2. **Run firmware to completion, then run the array to completion.** Rejected
   because context execution and completion require interleaving.
3. **Build a multi-clock event scheduler now.** Deferred until hardware
   observation supplies a clock relationship. The boundary-driven loop keeps
   that later change local.

## 4. Driver-Shaped Command Transport

Reuse the existing test-only pinned mailbox harness. Extend it only as needed
to:

- retain the context ID and context completion-queue descriptors returned by
  real `CREATE_CONTEXT`;
- send `MAP_HOST_BUFFER`;
- place the xclbin-selected PDI in registered host memory and send the exact
  `CONFIG_CU` request before execution;
- publish the exact driver-shaped `CHAIN_EXEC_NPU` request and command-chain
  bytes; and
- consume the eventual context response.

Wire layouts, opcodes, sizes, and ordering come from pinned open-driver commit
`216cefececd74effcd7a88350c71b99f5ef9a215`. The harness transports bytes; it
does not interpret the command, program the array, decide completion, or become
a public ABI.

The existing explicit source-46 assertion remains visible for host-command
delivery on the management channel until the BAR4 causal edge is grounded.
Context commands use the queue pair returned by `CREATE_CONTEXT`; they must
not be delivered by asserting management source 46. The first context-X2I
publication may therefore expose a separate unmodeled wake transition. That
known management-channel synthetic edge must not be confused with or reused
for context dispatch or downstream array completion.

## 5. Array Launch and Completion

The authentic path replaces these current host-side stand-ins only after real
firmware performs their effects:

- PDI/CDO application at the wrong layer;
- early core-reset release;
- core warm-up;
- `NpuExecutor` runtime-sequence interpretation;
- fixed mailbox latency; and
- host-side declaration of natural completion.

They remain in place for existing non-firmware callers while the new path is
being proven. Delete each stand-in separately after the firmware acceptance
path covers it and the existing corpus remains green.

### Completion evidence gate

The array already emits and consumes toolchain-derived TCT state internally.
What is missing is the Phoenix management-side receiver.

The first integrated execution must observe the unmodified firmware after the
frozen kernel finishes:

- If firmware naturally consumes existing modeled state and publishes the
  context response, record and test that path.
- If firmware waits on a previously unmodeled read, flag, or interrupt, stop
  with the exact address, CPU state, outstanding token state, and command
  state.

Only then perform the smallest targeted hardware capture capable of
identifying that transition. The implementation may connect the completion
path only after the capture or an authoritative open source pins:

- the management-side status representation;
- the acknowledgement behavior; and
- the interrupt or wake source.

No output-buffer correctness check, timeout expiration, `NpuExecutor` token,
source-46 assertion, or direct done-flag write may stand in for that evidence.

## TDD Sequence

### A. Topology foundation

First failing tests:

- Phoenix reports five physical control columns;
- exactly 24 real tiles exist at physical columns 1..4;
- every `get(0, row)` is `None`;
- column 0 clock control remains addressable;
- NPU4 does not inherit Phoenix's extra column; and
- a logical NPU1 tile at column 0 relocates to physical column 1 exactly once.

Then migrate internal loops and parallel indices to the real tile span. Run
the complete library suite and the frozen targeted bridge test before
continuing.

### B. Shared outbound host memory

First failing tests configure one management outbound window and prove that
firmware-side byte and word accesses see the engine's registered
`HostMemory`, while an unregistered target still reaches `SysStub`.

### C. Runtime pump

First failing tests use a tiny firmware/engine fixture to prove:

- firmware runs to `waiti`;
- the array advances while firmware waits;
- a modeled interrupt resumes firmware;
- no-progress and error outcomes stop boundedly; and
- no firmware or array state is cloned.

### D. Frozen firmware/kernel path

Drive the pinned initialization, map the host heap, register the xclbin PDI
through `CONFIG_CU`, publish the real context command, and run the frozen
Chess artifact first. The first interior checkpoint is firmware application
of the PDI to the shared array; the first green functional checkpoint remains
correct output produced by firmware-caused launch. Repeat with Peano.

If firmware completion is still blocked, preserve the exact stop as a
regression test, perform the targeted hardware measurement, and add the
minimal grounded completion model. Final green requires the real context
response and ERT completion for both artifacts.

## Stop Conditions

Pause and return to design or measurement if progress would require:

- inventing physical tile presence at column 0;
- applying partition relocation twice;
- treating an unregistered outbound target as host memory;
- assigning a firmware/AIE clock ratio without hardware evidence;
- connecting TCT completion to an unproved management source;
- synthesizing a firmware response or task done flag;
- weakening an existing bridge or hardware acceptance check; or
- expanding into virtual PCI before the interior firmware/array loop closes.

## Verification

Fast gates after every source change:

```bash
cargo test --lib <new-focused-test>
cargo fmt --all --check
git diff --check
```

Required after each complete foundation slice:

```bash
cargo test --lib
cargo test -p xdna-emu-ffi
```

Required after topology and after the final integrated path:

```bash
./scripts/emu-bridge-test.sh --no-trace -v '^add_one_using_dma$'
```

The targeted bridge gate must pass Chess and Peano on both hardware and the
emulator. Brief builds and unit tests run locally; Halo is reserved for heavy
toolchain compilation.

## Deferred Work

- Xtensa/AIE timing reconciliation and clock-ratio measurement;
- removal of the explicit host-command source-46 seam;
- virtual Phoenix PCI and unmodified kernel-driver attachment;
- complete normal/error/reset/timeout/recovery command coverage;
- older authoritative Phoenix firmware revisions; and
- AIE2P devices.

These follow only after the pinned primary firmware completes the frozen
kernel through the shared simulated array.
