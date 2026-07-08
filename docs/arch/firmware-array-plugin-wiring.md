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

## The one real unknown -- BOOT (not jobs). IDENTITY PINNED: SMU/PSP column power.

Boot wedges on a **per-column power-up / clock-ungate bring-up handshake** that
the mgmt firmware performs with the SMU/PSP -- NOT the AIE array, NOT the mailbox
alive handshake (see M3 below for the full evidence). The event source
`0x27010d28` is an SMU/PSP power-mgmt event register (MpNPUAxiXbar public block,
neighbours `PUB_SEC_INTR`/`PUB_PWRMGMT_INTR`); the per-column status/ack block
`0x2727_n000`/`+0x114` is the platform per-column "powered/done" block. Runtime
`dma_wait` waits on TCT pages that alias the same `0x2727_n000` family but are a
*different signal* (Seam A un-wedges those runtime waits; it does NOT touch this
boot power handshake).

The boot fork, re-framed by the pinned identity:

- **Solve boot properly:** model an SMU/PSP column-power agent that consumes the
  `0xfae0` colmask descriptor, powers the 4 columns, and delivers the completion
  event into the fw's wake path. Heavy (a new power subsystem) AND the delivery
  mechanism is still underivable firmware-only. The "complete emulation" path, but
  orthogonal to the array/timing goal.
- **Reach the runtime path another way:** the dream's payoff (the 8000cy
  `DEFAULT_MAILBOX_CYCLES` dissolving) is a RUNTIME concern where M1/array is the
  right tool. Getting there via boot requires paying the SMU/PSP prerequisite
  first. Whether that is worth it vs. a different bootstrap is the strategic
  checkpoint.

FW_ALIVE (`aie2_pci.c:145`, magic `0x55504e5f`) + CREATE_CONTEXT(0x2) ->
MAP_HOST_BUFFER(0x106) -> CONFIG_CU(0x11) all happen AFTER this wall, in the
post-alive mailbox stage the boot never reaches -- not part of the boot gate.

## Milestone order

- **M1 -- Seam B. DONE (`0d4ff4da`).** `Bus` owns an `Option<DeviceState>`;
  32-bit `Region::Array` accesses decode via the tile formula (`decode_array_addr`,
  base `0x0400_0000`, shifts from archspec -- NOT `decode_npu_address`, whose
  base/start_col are the runtime encoding) and route into
  `DeviceState::{read,write}_tile_register`. `None` keeps the pre-M1 stub. Open
  ceiling: the aperture only spans cols 0..=1 (`col<<25` from a `0x0400_0000`
  base collides with RAM at col>=5) -- the true multi-column firmware->column map
  is unresolved, deferred to when real firmware runs a multi-column job (M2/M3).
- **M2 -- Seam A. REORDERED after M4 (2026-07-07, Maya).** M2 in isolation is
  weakly-testable (a standalone unit test only proves "the seam sets the bits we
  told it to"). Worse, the target register `0x2727_n000` was *definitively
  falsified as the boot gate* by the RE (session-3: fully satisfying it does NOT
  advance boot). So build the completion wire *with* a real job flow (M4), where
  it feeds an actual `WAIT_TCTS` and gets validated -- not before.
- **M3 -- Boot to alive+idle. THE PREREQUISITE (blocks M4). GATE IDENTITY PINNED
  2026-07-07 = SMU/PSP COLUMN POWER-UP, not the array, not FW_ALIVE.** M4 needs
  the firmware alive and polling the mailbox to receive a job; **it does not reach
  idle by any bootstrap** (CompletionAgent is `#[ignore]`d as insufficient;
  force-done advances to ~623k then hits Unknown op `0xd903`, not idle).

  **What the wall is (now identified).** Early task `0x10f10` builds a 7-word
  descriptor at local `0xfae0` `{1,1,colmask=0xf,0,task,0,0}`, cache-flushes it,
  and waits for **per-column completion**. An Explore over xdna-driver + aie-rt
  (2026-07-07) pinned this handshake as an **SMU/PSP column power-up / clock-ungate
  bring-up**, verdict (B), well-constrained (exact block-1 register *name* is
  firmware-private, unproven; the *operation* is solid): (1) ZERO tile-aperture
  writes at the wall -> not an AIE op; (2) the per-column status/ack block
  `0x2727_n000`/`+0x114` (stride 0x1000) is far too compact to be AIE tiles
  (AIE-ML col-shift 25, clock reg `0xFFF20`); (3) the event source `0x27010d28`
  sits in the MpNPUAxiXbar public block whose named tenants are `PUB_SEC_INTR`
  (PSP) + `PUB_PWRMGMT_INTR` (SMU) -- `npu1_regs.c:12-13`; (4) colmask 0xf = all 4
  Phoenix columns = an ungate mask, and AIE2 columns are clock-disabled by default,
  ungated per-column (`xaie_device_aieml.c:128`); NPU1 firmware owns clock gating
  (`NPU1_RT_TYPE_CLOCK_GATING`); (5) the driver's `aie2_smu_start` fires ONE
  maskless SMU POWER_ON then delegates to firmware -- the per-column loop runs
  *inside* the mgmt fw between PSP-start and the scalar FW_ALIVE poll = exactly
  this wall.

  **Hypotheses CLOSED (this + prior session):** (a) the array does NOT gate boot
  (`m2c_probe_boot_with_array`: ZERO Array-aperture accesses, stub==attached
  byte-identical; M1's wiring is inert for boot; the array/TCT path is a *runtime*
  job concern only). (b) `0x2727_n000` poll bits do not gate boot (RE session-3:
  fully satisfying them does NOT advance). (c) **FW_ALIVE is NOT the lever
  (falsified 2026-07-07):** the magic `0x55504e5f` store never executes in 1.5M
  boot instrs (`m2c_probe_alive_struct`), and the wall polls ONLY local memory --
  ZERO host-writable-aperture reads (`m2c_probe_poll_map`). The prior "live lever =
  model the HOST side of FW_ALIVE" note was a banking-session over-extrapolation
  that contradicted the RE; corrected. The wall is PRE-alive; FW_ALIVE is a
  later-stage concern the boot never reaches.

  **Where this leaves boot.** The boot gate is a **platform column-power**
  handshake orthogonal to the array/timing-emergence goal -- the emulator models
  no SMU/PSP power subsystem. Modeling it faithfully is a new subsystem, AND the
  completion-DELIVERY mechanism (how the powered-columns event reaches the fw's
  task-readying wake path `FUN_00005580 -> [0x10f40]`) is still underivable from
  firmware alone (the wake path is unreachable in EMU; only bit0/mailbox is ever
  armed; INTLEVEL held at 2). So boot-to-idle remains blocked on either an SMU/PSP
  power-agent model (heavy, uncertain payoff) or HW-in-the-loop observation.
  **Strategic checkpoint raised to Maya:** does the dream's runtime timing goal
  (the 8000cy `DEFAULT_MAILBOX_CYCLES`, a *runtime* concern where M1/array IS the
  right tool) justify first paying the SMU/PSP-column-power boot prerequisite, or
  reach the runtime path another way. Full RE: the completion-causality finding,
  Session-6 entry.
- **M4 -- Seam C.** Plugin posts x2i job, firmware runs it against the array
  (M1), posts i2x response; M2 completion wire built + validated against a real
  `WAIT_TCTS`. First light: real firmware runs add_one, timing emerges. Blocked on
  M3 (needs alive+idle).

## Mailbox wire protocol (derived from xdna-driver 2026-07-07)

Authoritative source: `xdna-driver/src/driver/amdxdna/` (`aie2_pci.c`,
`aie2_pci.h`, `aie2_msg_priv.h`, `amdxdna_mailbox.c/.h`, `npu1_regs.c`).

- **x2i offsets are firmware-defined, not driver-fixed.** At boot the firmware
  publishes a `struct mgmt_mbox_chann_info { u32 x2i_tail,x2i_head,x2i_buf,
  x2i_buf_sz, i2x_tail,i2x_head,i2x_buf,i2x_buf_sz, magic(0x55504e5f), msi_id,
  prot_major, prot_minor, rsvd[4] }` into SRAM and writes its address into the
  `FW_ALIVE_OFF` slot; the host reads the struct back and learns all four rings'
  registers, then writes 0 to `FW_ALIVE_OFF`. So the x2i poll address must be
  DISCOVERED from our firmware (probe `m2c_probe_mailbox_receive`), not chosen.
- **Known i2x triple (confirmed):** tail `0x27200170` (fw writes to post), head
  `0x27200174` (host writes to ack-consume), intr `= head+4 = 0x27200178`, desc
  ptr `0x27200180`. `head==tail` == consumed.
- **Doorbell:** NO separate register. The host posts by `memcpy` into the ring
  then a single write to `x2i.mb_tail_ptr_reg` -- that write raises the fw's
  Xtensa mailbox IRQ (INTENABLE bit0). In-emu: on a store to the published
  x2i-tail addr, set `cpu.interrupt |= 1`.
- **Ring:** message bytes live in SRAM (`ringbuf_base + rb_start + tail`), byte-
  offset tail advancing by full packet size; wrap writes `TOMBSTONE 0xdeadface`
  and restarts at 0. Slot `CHAN_SLOT_SZ = 0x2000`; `rb_size` power-of-2.
- **Header (16B, `static_assert`):** `{u32 total_size, u32 sz_ver (body|ver<<16,
  ver=1), u32 id (entry[0..255] | magic 0x1D000000), u32 opcode}`.
- **Opcodes** (`aie2_msg_priv.h`): CREATE_CONTEXT `0x2`, EXECUTE_BUFFER_CF `0xC`,
  EXEC_DPU `0x10`, CONFIG_CU `0x11`, CHAIN_EXEC_NPU `0x18`, MAP_HOST_BUFFER
  `0x106`, GET_PROTOCOL_VERSION `0x301`. No FW_ALIVE *opcode* -- aliveness is the
  SRAM handshake above.
- **EXEC_DPU payload** (`exec_dpu_req`, 160B): `u64 inst_buf_addr; u32 inst_size;
  u32 inst_prop_cnt; u32 cu_idx; u32 payload[35]`.

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
- Mailbox-receive discovery probe: `m2c_probe_mailbox_receive` (`src/firmware/mod.rs`)
- Array-vs-boot experiment: `m2c_probe_boot_with_array` (`src/firmware/mod.rs`)
- Boot wall PC: `0xc969` (`FUN_0000c928+0x41`, scheduler bit-scan loop)
- FW_ALIVE magic literal: image offset `0x3388`; i2x post seen at instr ~6972
- Host mailbox model (i2x consume + boot completion agent): `src/firmware/host_mailbox.rs`
