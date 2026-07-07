# Firmware / Array / Plugin Wiring -- Design Record

**Issue:** #140 (firmware-emulation dream).
**Branch:** `feat/m2c-mapping-boot-to-idle` (unmerged; holds the whole arc).
**Status:** Archaeology complete; implementation (M1) not started. Written as
the durable guide across a context compaction -- read this first on resume.

---

## The goal (restated, because it was drifting)

The target is **a complete emulation**: the real AMD XDNA management firmware
(Xtensa, `npu.dev.sbin`) running *live* on the in-tree interp, programming the
*emulated* AIE2 array, driven by the XRT plugin over the real mailbox protocol.
When that exists, hardcoded timings (the 8000-cycle `DEFAULT_MAILBOX_CYCLES`,
`DispatchGate`, etc.) **dissolve** -- they stop being constants we derive and
become emergent behavior of real firmware executing against emulated hardware.

**This was never about deriving the 8000.** The 8000 decomposition (below) was a
*probe* to understand the completion mechanism well enough to wire it. The
probe's verdict: the 8000 is the firmware's completion-notice latency (poll ->
wake -> advance), which the emulator today collapses into a flat fudge because it
models the array only up to "data done" and nothing of the firmware side.

## What the 8000 probe established (banked understanding)

- The emulator models the DMA transfer FSM in real per-cycle detail up to
  `Channel_Running` clearing (~1cy after data done), then charges a flat
  `DEFAULT_MAILBOX_CYCLES = 8000` **after** the array is already idle
  (`src/npu/executor.rs:664`). It models **no** TCT issue/route/notice/wake path;
  the DMA task token is emitted but never consumed by the sync path.
- The 8000 is firmware completion-notice, not array propagation (that's modeled
  and tiny) and not firmware processing-heaviness (the scheduler is cheap: one
  dispatch cycle ~392 instrs, done-flag check each cycle). Poll-based (RE ruled
  out interrupts), per-batch. The firmware programs **no Xtensa timer** (zero
  CCOMPARE writes) -- the poll cadence is emergent from scheduler rotation, not a
  readable constant. So the 8000 cannot be derived firmware-statically; it is a
  real hardware delay best replaced by wiring + a calibration knob (seam A).
- Two committed instruments (this session): `m2c_probe_cycle_accounting` and
  `m2c_probe_sr_usage` in `src/firmware/mod.rs` (XDNA_FW_PROBE-gated).

## The completion mechanism (AIE2, from aie-rt / mlir-aie / AM025)

Two physically distinct completion signals; only one faces the mgmt processor:

1. **In-tile status** -- `DMA_*_Status`, `Channel_Running`, task-queue-size.
   Array aperture (`0x200_0000_0000 + ...`). The firmware does NOT poll this.
   This is what the emulator models today.
2. **TCT (Task Completion Token)** -- on a BD completing with
   `Enable_Token_Issue=1`, the DMA emits a **stream packet** (not a register
   write), header `col<<21 | row<<16 | actor_id(channel,dir)`, routed out the
   shim **South** port toward the mgmt subsystem. The firmware counts N tokens
   (`WAIT_TCTS`). Source + format are documented
   (`mlir-aie/.../AIENpuToCert.cpp:143-183`, aie-rt `xaie_dma.c` EnTokenIssue,
   AM025 shim `DMA_S2MM_0_Task_Queue` bit31). **The MP-side landing register
   where tokens accumulate (`0x2727_n000` pages, ack `+0x114`) is NOT documented
   in aie-rt/AM025** -- it lives in the Xtensa/MP_NPU aperture, known only from
   our firmware-side RE. Latency for the whole path: documented nowhere -> an
   explicit calibration knob (the honest 8000 replacement).

## The three seams

### Seam B -- firmware -> array (control). SMALL.
The emulator already has the full array-programming surface behind one entry
point: `DeviceState::write_tile_register(col,row,offset,value)`
(`src/device/state/dispatch.rs:20`), covering DMA BDs, Task_Queue start, stream
routing, locks, core enable/reset, column reset. The transaction opcodes the
firmware interprets from insts.bin are the SAME set the NPU executor decodes
(`src/npu/parser.rs`, `mod.rs:48-123`), and the executor already routes them via
`write_tile_register` (`executor.rs:935,986`).

**The gap is pure connection.** `firmware/mmio.rs` `Region::Array` is a logged
discard stub (`mod.rs:4`: "routing into DeviceState is later (M2)"). The seam:
thread a `&mut DeviceState` into `Bus`; in the `Region::Array` arms of
`region_store32/8` and `region_load32/8` (`mmio.rs:364,433,330,399`), decode via
`decode_npu_address` (`executor.rs:1527`) and call `write_tile_register` /
tile-register read instead of `record_stub`.

### Seam A -- array -> firmware (completion). MEDIUM. The wall's resolution.
On a shim/memtile/core DMA BD completing with `Enable_Token_Issue`, make the
emulated array increment the per-column TCT accumulator the firmware polls
(`0x2727_n000` family, RE-known bits bit0/1/3, ack `+0x114`, in the firmware's
Mailbox aperture `0x2700_0000-0x2800_0000`). The firmware's `WAIT_TCTS` poll is
then satisfied by the emulated array. Model the array->accumulator latency as a
named calibration knob (replaces `DEFAULT_MAILBOX_CYCLES`). This is exactly the
"array-completion contract" that boot-to-idle was banked on as "not derivable
from firmware alone" -- correct: you don't derive it, you wire it.

### Seam C -- plugin -> firmware (mailbox). MEDIUM.
Rings: x2i (host->fw) and i2x (fw->host) in SRAM, pointer regs in the mbox BAR.
Post = copy 16-byte header (`MAGIC 0x1D000000`, opcode, id, size) + payload into
the ringbuf and advance the tail (the doorbell); firmware polls its x2i tail;
host gets MSI-X on i2x. CHAIN_EXEC_NPU (0x18) points at a buffer of job slots
(`aie2_msg_priv.h`). FFI seam: `crates/xdna-emu-ffi/src/execution.rs:47` (today
calls the in-Rust `NpuExecutor`; replace with post-x2i -> step-firmware ->
read-i2x). `src/firmware/host_mailbox.rs` provides ~20% (the i2x receive/ack half
+ the boot CompletionAgent hack); build the x2i producer, header/opcode decode,
dispatch, and i2x response synthesis.

## The one real unknown -- BOOT (not jobs)

Boot wedges waiting on event source `0x27010d28` (MP aperture, PSP/SMU
neighborhood). Runtime `dma_wait` waits on the TCT pages `0x2727_n000`.
**Different signals.** Seam A un-wedges *runtime* waits. But nothing runs at
boot, so `0x27010d28` at boot is an **init/PSP/SMU/host handshake**, not a DMA
TCT -- and we have not pinned what writes it. The fork:

- **Solve boot properly:** model the init trigger for `0x27010d28` -> firmware
  boots to FW_ALIVE + idle naturally -> fully end-to-end. (Recommended: the
  difference between a complete emulation and one with a manual bootstrap.)
- **Snapshot past boot:** reach alive+idle once, snapshot, start jobs there.
  Sidesteps boot -- but reaching alive once is itself the wall.

Before jobs, the firmware must publish FW_ALIVE (`aie2_pci.c:145`, magic
`0x55504e5f`) then handle CREATE_CONTEXT(0x2) -> MAP_HOST_BUFFER(0x106) ->
CONFIG_CU(0x11) (observed order from the HW mailbox trace this session).

## Milestone order

- **M1 -- Seam B. DONE (`0d4ff4da`).** `Bus` owns an `Option<DeviceState>`;
  32-bit `Region::Array` accesses decode via the tile formula (`decode_array_addr`,
  base `0x0400_0000`, shifts from archspec -- NOT `decode_npu_address`, whose
  base/start_col are the runtime encoding) and route into
  `DeviceState::{read,write}_tile_register`. `None` keeps the pre-M1 stub. Open
  ceiling: the aperture only spans cols 0..=1 (`col<<25` from a `0x0400_0000`
  base collides with RAM at col>=5) -- the true multi-column firmware->column map
  is unresolved, deferred to when real firmware runs a multi-column job (M2/M3).
- **M2 -- Seam A.** Emulated DMA completion increments the firmware's TCT
  accumulator; `WAIT_TCTS` satisfied; latency knob. Unit-testable: trigger a DMA
  completion, assert the polled register updates.
- **M3 -- Boot init trigger (`0x27010d28`).** Resolve the fork; get the firmware
  to alive+idle. Critical-path research.
- **M4 -- Seam C.** Plugin posts x2i job, firmware runs it against the array,
  posts i2x response. First light: real firmware runs add_one, timing emerges.

## Key file:line index (for the implementer)

- Array programming entry: `src/device/state/dispatch.rs:20` `write_tile_register`
- Address decode: `src/npu/executor.rs:1527` `decode_npu_address`
- Firmware MMIO stub to replace: `src/firmware/mmio.rs:364/433/330/399` (Region::Array)
- Firmware interp <-> device gap marker: `src/firmware/mod.rs:4`
- Mailbox model (partial): `src/firmware/host_mailbox.rs`
- FFI job seam: `crates/xdna-emu-ffi/src/execution.rs:47`
- The 8000 fudge: `src/npu/executor.rs:664`
- TCT accumulator (firmware side, RE): `0x2727_n000` pages; see
  `docs/superpowers/findings/2026-07-06-iter18-completion-causality-RE.md`
- Cycle-accounting / SR-usage probes: `src/firmware/mod.rs` (XDNA_FW_PROBE-gated)
