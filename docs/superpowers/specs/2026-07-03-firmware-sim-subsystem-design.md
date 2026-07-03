# Firmware-sim subsystem -- design

**Date:** 2026-07-03
**Status:** approved design, pre-implementation
**Origin:** #140 timer-sync arc / SP-4a. See
`build/experiments/firmware-re/INFODUMP.md` for the reconnaissance this design
rests on, and [[project_firmware_emulation_dream]] in memory.

## 1. Purpose

The emulator currently hardcodes several management-firmware behaviors as magic
numbers: the core reset-deassert at kernel launch (`release_core_resets()`), the
mailbox latency (`DEFAULT_MAILBOX_CYCLES`), the dispatch gate/controller pacing
(`DispatchGate` in `src/npu/cycle_cost.rs`), the core warm-up
(`crates/xdna-emu-ffi/src/backend.rs:225`, premise already falsified), and the
config-register MMIO cost. Each is a guess we have been unable to derive.

This subsystem **runs the real Phoenix management firmware on an in-tree Xtensa
interpreter**, with its MMIO routed into the existing `DeviceState`, so those
timings *emerge* from the firmware executing rather than being hand-set. The
firmware becomes, architecturally, just another agent writing into the device
model the cores already use.

**Win condition (broad):** the full host->firmware dispatch path emerges -- we
delete `DEFAULT_MAILBOX_CYCLES`, `DispatchGate`, the core warm-up, and
`release_core_resets()`, and the dispatch timing they encoded is reproduced by
the running firmware, validated against HW.

This is a large new subsystem; it is built **skeleton-first** (Section 9) even
though the target coverage is broad.

## 2. Grounding facts (from recon; full detail in INFODUMP.md)

- **ISA:** Xtensa, little-endian, 32-bit, with the MMU option. Base ISA only on
  the control plane (no TIE/vector custom ops on the dispatch path). Windowed
  register ABI (`entry`/`retw.n`/`call8`/`callx8`).
- **Binary:** `xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`, 248592 B,
  `$PS1` PSP container at offset 0x10. Signed, **not encrypted, not compressed**.
  We bypass signing and run the plaintext payload.
- **Memory map:** `.text`/`.rodata` at base 0 (code range 0x2730-0x3ca0e);
  `.data`/`.bss` + copied `.rodata` at 0x08b00000 (rodata copy delta
  D = 0x08ad2f00); AIE array at 0x04000000; mailbox/doorbell block at 0x27000000
  (regs 0x27010d00-0x27010d28); SMN/NoC/system apertures at
  0x00/0x02/0x03/0x06/0x08/0x0f/0xf7.
- **Array interface:** `tile(col,row,reg) = 0x04000000 + (col<<25) + (row<<20)
  + reg` -- the exact formula the emulator already implements
  (`src/device/model.rs:235`, `TILE_COL_SHIFT=25 / TILE_ROW_SHIFT=20`).
- **Mailbox:** two ring channels `CHAN_RES_X2I` (host->fw) / `CHAN_RES_I2X`
  (fw->host); 16-byte wire header `{total_size, sz_ver, id, opcode}` + payload;
  msg-id magic 0x1D000000; head/tail pointer regs in the 0x27010dxx block.
  Dispatch = `EXEC_DPU 0x10` / `EXECUTE_BUFFER_CF 0xC`, payload
  `exec_dpu_req { u64 inst_buf_addr; u32 inst_size, inst_prop_cnt, cu_idx; u32
  payload[35]; }`; `inst_buf_addr` -> the runtime-sequence instruction buffer.
- **Clock:** firmware uses **zero `CCOUNT`/`CCOMPARE`** -- event-driven
  (`wsr INTCLEAR`), no cycle-calibrated delay loops. `CALIBRATE_CLOCK 0x11C`
  takes `time_base_ns` (host-calibrated time). So reproducing firmware *ordering*
  needs no exact Xtensa MHz; the clock only sets the instruction-processing rate.
- **HW ground truth (validation target):** the KFG host-firmware-dispatch row --
  ~30k-cy dispatch base + ~112 AIE-cy per runtime-sequence instruction (and the
  ~8000-cy mailbox latency).

## 3. Module layout

```
src/firmware/
  mod.rs         FirmwareProcessor: owns the Xtensa core + boot + run loop;
                 exposes step()/run-until-wait to the co-sim scheduler
  image.rs       parse npu.dev.sbin ($PS1 container), strip sig, lay out
                 .text/.rodata/.data into the interpreter's address space
  xtensa/
    decode.rs    base-ISA decoder (RRR/RRI8/RI16/CALLn/BRI8/... formats)
    regfile.rs   windowed register file: AR[64], WINDOWBASE, WINDOWSTART,
                 rotation on call/return; special registers (SAR, PS, ...)
    interp.rs    fetch/execute loop; entry/retw.n/callN/callxN; window
                 overflow/underflow handling
    exc.rs       minimal exception/interrupt path (INTCLEAR, window exc
                 vectors, WAITI if used)
    mmu.rs       only as much TLB/mapping as the boot sequence sets up
  mmio.rs        address router: dispatch a firmware load/store by aperture
  sysstub.rs     off-array system-aperture stubs (SMN/NoC/system config),
                 with per-address access logging + spin-detection
```

The interpreter is deliberately split: `decode.rs` (pure, table-like) and
`regfile.rs` (the windowed-ABI state) are independently testable; `interp.rs`
wires them. `mmio.rs` is the single seam between firmware and `DeviceState`.

## 4. The Xtensa interpreter

Base ISA subset actually used by the control plane -- loads/stores, ALU,
branches, the windowed call family (`entry`, `retw.n`, `call4/8/12`,
`callx4/8/12`), `l32r` (PC-relative literal loads), `movi`/`addmi`, `wsr`/`rsr`
for the handful of special registers the firmware touches (`SAR`, `PS`,
`WINDOWBASE`, `WINDOWSTART`, `INTCLEAR`, MMU regs). We implement instructions on
first encounter, driven by what the firmware actually executes -- not the whole
ISA up front.

**The load-bearing complexity is the windowed register ABI.** `regfile.rs` owns
the 64-entry physical register file, `WINDOWBASE`/`WINDOWSTART`, rotation on
`entry`/`retw`, and window overflow/underflow (either via the exception vectors
or a modeled spill, decided at M1 from how the firmware is built). This is the
one part that is genuinely fiddly; it is isolated so it can be unit-tested
against hand-assembled sequences (and, later, an optional qemu lockstep oracle).

**MMU/TLB:** the objdump showed a boot TLB setup (`witlb`/`wdtlb`/`iitlb`/
`idtlb`) and a virtual base ~0x40000000, while link/physical addresses sit at
base 0. `mmu.rs` implements only enough to honor the mapping the boot sequence
installs. Whether an identity/flat map suffices is resolved empirically at M1
(instrument the TLB writes, see what mapping the firmware actually depends on).

## 5. The MMIO bridge (`mmio.rs`)

A firmware load/store is routed by aperture:

| Aperture | Target | Direction |
|----------|--------|-----------|
| `0x04000000 + (col<<25)+(row<<20)+reg` | `DeviceState` tile/register model | both |
| `0x27000000` block (0x27010dxx) | mailbox rings + head/tail + doorbell | both |
| `0x08b00000` + `.data`/`.bss` | interpreter data RAM | both |
| base 0 `.text`/`.rodata` | interpreter code/rodata (RO) | read |
| `0x00/02/03/06/08/0f/f7` | `sysstub.rs` (off-array system config) | both, logged |

The array aperture reuses the existing tile-decode -- the firmware's writes land
in exactly the registers the cores/DMA read, so no parallel device model. This
is the seam that makes the firmware "just another writer."

**Early RE bite (M2):** confirm the backend `Write32` path -- follow
`XAie_CoreEnable @0x33244`'s `callx8` (`DevInst+0x28 -> ops -> +0x10`) to the
actual CORE_CONTROL MMIO poke, verifying the array write path and how
`DevInst.BaseAddr` is composed.

## 6. The execution / timing model (instrument-first; hypothesis held loosely)

**We refuse to over-design this.** The firmware's real wait behavior is
observable, and once we see it the model largely picks itself. The spec commits
to the *seam* and the *instrumentation*, not to a wait-accounting rule.

**The observable that resolves it (M3):** *how does the firmware wait?*
- `WAITI` (wait-for-interrupt) -- consistent with `wsr INTCLEAR`. Then: firmware
  yields, device advances to the interrupt condition, firmware resumes.
- Poll-spin on a device status register (read/test/loop). Then: detect the spin,
  advance the device until the polled value can change, resume.

We do not know which yet; the first milestone that runs a dispatch will show us.

**Working hypothesis H1 (to confirm or discard):** cooperative co-simulation on
the emulator's existing timeline, with two kinds of firmware activity:
- **Non-waiting work** costs `instr_count x (Xtensa:AIE cycle ratio)`. This is
  the only place the CPU clock enters, and the source of the ~30k-cy dispatch
  base.
- **Waits** cost whatever the *device* takes: the firmware yields, `DeviceState`
  advances to the next relevant event, elapsed *device* cycles are the wait. The
  firmware's own spin instructions do not inflate time; the array/DMA sets the
  duration.

**The seam the spec commits to:** `FirmwareProcessor` exposes a run-until-wait
step to a co-sim scheduler. The interpreter runs firmware instructions
(counting them) until it either (a) performs a device access that cannot yet
make progress (poll-not-ready / `WAITI`), or (b) writes a completion and returns
to idle. On (a), the scheduler advances `DeviceState` to the next event that
could change the awaited state, then resumes. The *duration accounting* on the
wait path is left open until M3 observation.

**Clock ratio (M4):** pinned by the 112-cy reconciliation -- run one dispatch,
count the firmware's non-waiting instructions on the per-runtime-seq-instruction
path, solve `ratio = 112 / (instrs per runtime-seq-instr)`, then verify the
~30k-cy dispatch base falls out of the fixed prologue instruction count x ratio.
One measured ground truth, one unknown. If it fails to close, H1 is wrong and we
revise from observation -- the seam stays.

## 7. The mailbox seam (host <-> firmware)

Replaces the abstract dispatch. The host side (XRT plugin / FFI) writes a real
mailbox message into the `CHAN_RES_X2I` ring in device memory (16-byte header +
`exec_dpu_req` payload, id tagged 0x1D000000) and bumps the X2I tail pointer
register. The firmware, in its command loop, reads head/tail, pulls the message,
dispatches through its aie-rt path, then writes the `CHAN_RES_I2X` completion and
signals the doorbell/interrupt; the FFI reads the completion.

This deletes `DEFAULT_MAILBOX_CYCLES`: the mailbox latency now = firmware
ring-poll + processing time, emergent.

**Early RE bite (M2):** confirm which 0x27010dxx registers are the X2I/I2X
head/tail pointers, the ring-buffer base, and the doorbell -- correlate the
driver's `xdna_mailbox_chann_res` fields against the firmware's register reads.

## 8. Boot and the stub layer

**Run boot from the entry vector every session** (no snapshot machinery). The
PSP has already done the low-level bring-up (DRAM, security) before the firmware
runs, so the firmware's own boot is high-level: MMU/TLB setup -> `.data`/`.bss`
init -> ops-table population -> mailbox ring config -> command loop. Boot runs
~once per emulator session; its *timing* is irrelevant (only its resulting state
matters), so nothing about boot fidelity affects dispatch timing.

`sysstub.rs` answers reads to the off-array apertures with benign values and
**logs every stubbed access**, so we can see exactly what boot depends on. All
"magic" is confined here, to genuinely off-scope system config.

**Risk + mitigation (the one real risk of run-boot):** boot could spin on an
unmodeled status bit. `sysstub.rs` includes **spin-detection** -- if the firmware
reads the same stub address in a tight loop past a threshold without progressing,
it flags the address as a candidate "waiting on unmodeled state," telling us
precisely which stub needs a real value or a modeled transition. This turns a
potential silent hang into a pinpointed, actionable log line.

## 9. Milestones (skeleton-first)

- **M0 -- image:** parse `npu.dev.sbin` (`$PS1`), strip signature, lay out
  `.text`/`.rodata`/`.data` into the interpreter address space. Deliverable: a
  loadable static memory image with the delta-D rodata copy applied.
- **M1 -- interpreter boots to idle:** `decode.rs` + `regfile.rs` (windowed ABI)
  + `interp.rs`; run boot from the entry vector to a stable command-loop idle,
  with `sysstub.rs` logging. Resolve the MMU/TLB question here. Deliverable:
  firmware alive and idle in-sim, no divergence.
- **M2 -- MMIO bridge + mailbox wiring:** route `0x04..` -> `DeviceState`;
  firmware array writes land in tiles. Wire X2I/I2X rings + head/tail regs.
  RE bites: backend `Write32` path (Section 5), 0x27 register correlation
  (Section 7). Deliverable: firmware can read a mailbox message and write to a
  tile register.
- **M3 -- one EXEC_DPU end-to-end:** host writes X2I ring -> firmware dispatches
  -> `CoreUnreset` fires -> array runs -> I2X completion. **Observe the wait
  mechanism (resolves H1).** Confirm the launch-sequence order against the symbol
  map (`SetupPartitionConfig -> SetPartColClockAfterRst -> CoreUnreset ->
  IpuIO_RunOp`). RE bites (INFODUMP section 9): `CoreUnreset` xref, launch
  sequence reconstruction. Deliverable: a real kernel launch driven by the
  running firmware.
- **M4 -- clock reconciliation:** count instructions, solve the ratio against the
  112-cy ground truth, validate the 30k base. Deliverable: a pinned, HW-anchored
  Xtensa:AIE cycle ratio and a closed reconciliation (or a revised H1).
- **M5 -- delete the stubs:** remove `release_core_resets()`,
  `DEFAULT_MAILBOX_CYCLES`, `DispatchGate`, and the core warm-up **one at a time**,
  each only when its path is covered and the corpus stays green. Deliverable:
  dispatch timing emerges and matches HW; the hardcoded firmware constants are
  gone.

## 10. Validation

- **Primary -- differential vs HW dispatch timing.** The emergent mailbox
  latency, dispatch base, and per-instruction cost must match the KFG
  host-firmware-dispatch measurements (~8000-cy mailbox, ~30k-cy base, ~112
  AIE-cy/instr). This is the reason the subsystem exists.
- **Structural -- symbol-map landmark oracle.** Log which named firmware
  functions the run enters and confirm the launch sequence order matches the
  expected aie-rt path. Independent of timing; catches interpreter/route bugs
  early.
- **Self-consistency -- the 112-cy reconciliation** (Section 6, M4).
- **Regression -- `cargo test --lib` plus the existing bridge/trace corpus.**
  Replacing the stubs must not regress the corpus; the emergent timings are only
  accepted when the corpus stays green.
- **Optional later -- qemu-xtensa lockstep oracle.** Once the interpreter exists,
  a differential instruction-by-instruction cross-check against
  `qemu-system-xtensa` can be bolted on for interpreter correctness. Not built up
  front.

## 11. Scope boundaries (YAGNI)

- **Base ISA only**; no FPU/vector TIE unless the control path actually hits one
  (recon says it does not). If encountered, hand-model the specific op.
- **Phoenix / NPU1 (`1502_00`) only** first; the sibling device firmwares
  (Strix/Halo/Krackan) come later once the mechanism is proven.
- **No boot-timing fidelity** -- boot runs once, its duration does not matter,
  only its resulting state.
- **No PSP model** -- we bypass signing and start from the plaintext payload.
- **Instructions implemented on demand** -- driven by what the firmware executes,
  not the full Xtensa ISA.

## 12. Risks

| Risk | Mitigation |
|------|------------|
| Boot spins on an unmodeled status bit | `sysstub.rs` spin-detection pinpoints the address (Section 8) |
| Windowed-ABI interpreter bugs | isolated `regfile.rs`, unit tests on hand-assembled sequences, optional qemu lockstep oracle |
| TIE custom ops on the control path | recon says none; if hit, identify and hand-model the few used |
| MMU complexity | implement only what the boot TLB setup requires; resolve at M1 |
| 112-cy reconciliation fails to close | means H1 is wrong -> revise the wait-accounting from M3 observation; the seam is unaffected |
