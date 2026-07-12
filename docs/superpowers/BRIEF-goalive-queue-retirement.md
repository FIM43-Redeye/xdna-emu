# Brief: is the go-alive run-fn a periodic worker, or a queue item the emulator fails to retire?

## The question (one sentence)

The emulated Phoenix mgmt firmware boots clean, builds its `_NPU` channel struct in
**local** memory, then **re-dispatches its go-alive run function (`0x55f8`) ~131 times
without ever advancing to the host-visible publish** (copy struct to device SRAM
`0x030bb000` + write the alive pointer to `FW_ALIVE_OFF` `0x030bf000`) — determine
whether that repetition is **intentional firmware behavior** (a periodic MERT worker
genuinely waiting on something) or **an emulator scheduler-fidelity bug** (our model
fails to retire the completed go-alive work item, so the firmware loops instead of
progressing).

## Why this is tractable and in-scope (read this first)

This is a debugging task **in code we own** (`src/firmware/`), NOT a reverse-engineering
wall. Two prior escalations that ARE walled — do NOT re-open them:

- The PSP-loader RE (recover the firmware's physical scatter/placement map): CLOSED
  NEGATIVE this session. The placement logic is not in any extractable off-chip PSP
  firmware (mask ROM or absent). Do not disassemble PSP/BIOS blobs.
- The `0x8cae` "runtime VMA-view collision" / host-SRAM visibility seam: this was
  pinned on the (now-blocked) PSP RE. Set it aside. It is DOWNSTREAM of this question:
  if the firmware never advances past the go-alive re-dispatch, whether the final
  publish path has a VMA collision is moot. Attack the re-dispatch first.

**Decisive new evidence (2026-07-12 driver trace):** the open-source Linux driver
(`xdna-driver`, `aie2_pci.c` / `aie2_psp.c`) does **nothing** between PSP START and
reading `FW_ALIVE_OFF` except poll that pointer — no mailbox message, no completion-ring
write, no doorbell. The firmware publishes `_NPU`/`FW_ALIVE_OFF` **autonomously** as a
consequence of PSP START. The first mailbox message (`SET_RUNTIME_CONFIG 0x10A`) is sent
only AFTER the channel is already alive. Therefore the real firmware advances from
"struct built" to "published to device SRAM" with NO external stimulus. If ours does not,
the prime suspect is our scheduler/queue model failing to retire the go-alive work item —
which is exactly what this trace must confirm or refute.

## The exact observation the last finding requested

From `docs/superpowers/findings/2026-07-11-frontier-extension-past-goalive-tail.md`
(READ THIS FIRST — it is the current-frontier finding; §"Terminal state" and its
"settling observation" paragraph define this task):

> "a queue-ownership trace across the `0xc6b0 -> 0x2630` yield/context-switch path:
> record the work-item head/tail and current-task stores before the first and second
> `0x55f8` entries. That will distinguish an intentional periodic MERT worker from a
> queue item that the emulator fails to retire."

Concretely:
- `goalive_runfn` entry = VMA `0x55f8`; go-alive tail = `0x5645`; the yield/context-switch
  edge is `0xc6b0 -> 0x2630`. `0x55f8` fires 131×, tail 132×.
- Instrument (or use the existing probe harness) to capture, at each `0x55f8` entry and
  across the `0x2630` context switch: the scheduler work-item queue head/tail pointers,
  the current-task pointer, and any stores that add/remove the go-alive work item.
- Compare the state **before the 1st `0x55f8`** vs **before the 2nd**: if the queue item
  is still present/re-enqueued after the run-fn returned successfully, the emulator is
  failing to retire it (bug). If the queue is empty and the firmware re-arms the worker
  by its own logic (a timer, a re-post, a poll on a condition), it is intentional — then
  identify the exact condition it re-checks.

## Code anchors (all in `src/firmware/`)

- `FirmwareProcessor::load_m2c` + the `+0x100` overlay tuples: `src/firmware/mod.rs:120-260`,
  constants `mod.rs:380-512`. `PSP_LOAD_OFFSET=0x5c` (BASE), `LOW_VMA_FILE_OFFSET=0x100` (AT).
- Bus / overlay selector / SRAM-band stores: `src/firmware/mmio.rs` (`add_rom_overlay`,
  the VMA-keyed selector ~`:200-227`, `peek8` BASE-only ~`:462-475`).
- The MERT run-to-completion scheduler / syscall dispatcher lives in the firmware image
  itself (Xtensa); the emulator runs it. The kernel syscall dispatcher tree is around
  `FUN_0000dab0` (codes `0x6b/0x6c/0x6d/0x70`); the go-alive service chain is
  `FUN_00005958 -> 0x93f0 -> 0x9448 -> 0x7e4c -> 0x893c`. The completion-ring scan reads
  `[0x27200330]/[0x2720032c]` (head/tail, both 0 at idle — this empty state is CORRECT
  pre-first-mailbox-msg, per the driver trace; it is NOT the blocker).
- Probe infrastructure and existing waypoint probes:
  `src/firmware/boot_tests/coherence_mapper.rs` (e.g. `m2c_probe_add_31a4_overlay_frontier`
  at `:1170`, `m2c_probe_waypoint_hits`, `m2c_probe_alive_publish`). Extend these rather
  than inventing a new harness.

## Reproduce

```
XDNA_FW_PROBE=1 XDNA_FW_MAX=500000 cargo test --lib \
  m2c_probe_add_31a4_overlay_frontier -- --nocapture
cargo test --lib m2c_boot            # the boot_tests module (image auto-detected)
```

Firmware image auto-detected (`../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`,
248592 bytes, Xtensa LE, Zephyr 3.7.1 + AMD "MERT" dispatcher).

## Deliverables

1. **Verdict:** intentional-periodic-worker vs emulator-fails-to-retire, backed by the
   queue-ownership trace (the head/tail + current-task values before 1st vs 2nd `0x55f8`).
2. If **fails-to-retire**: the precise root cause (which store/edge the emulator models
   wrong, or which retire step it skips), and a proposed emulator-side fix as a diff, with
   the boot test rerun showing the firmware now advances toward the device-SRAM publish
   (a store reaching the `0x030bb000` struct region or `FW_ALIVE_OFF`). Present the diff
   for review — do NOT commit; integration is Opus/Maya's call given the fidelity stakes.
3. If **intentional-periodic**: the exact condition the worker re-checks each period
   (register/memory address + value it awaits), so we know what real firmware waits for
   between "struct built locally" and "published to device SRAM."

## Ground rules (fidelity — important)

- **Match real hardware; do not force/shim firmware state.** Prior arc lesson: injecting
  or forcing scheduler state CORRUPTS the boot ("faithful completion doesn't corrupt, only
  repeated forcing did"; "the SHIM structurally corrupts scheduler state"). Any fix must be
  a faithfulness correction to the EMULATOR's model (scheduler/queue/memory semantics),
  never a firmware-state injection or a hardcoded advance.
- The `_NPU` local build is real and correct (`local_data[0x14820]==0x55504e5f`) — don't
  regress it. Full suite `cargo test --lib` must stay green.
- Honest-negative is valuable: if the trace shows an intentional periodic worker awaiting
  a condition the emulator can't yet satisfy, say so precisely — that's a real result, not
  a failure.
