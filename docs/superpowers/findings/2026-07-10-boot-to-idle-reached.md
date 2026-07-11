# Boot-to-idle: REACHED — real XDNA mgmt firmware boots to idle on the interp

**Date:** 2026-07-10
**Issue:** #140 firmware-emulation dream / boot-to-idle.
**Branch:** `feat/m2c-mapping-boot-to-idle` (unmerged).
**Status:** RESOLVED. The mission goal is met.

## TL;DR

The real AMD XDNA management firmware (`xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`,
Xtensa LE, Zephyr v3.7.1 + AMD "MERT" run-to-completion dispatcher) now **boots to
idle** on the in-tree Xtensa interpreter (`src/firmware/`) against the emulated
array, running **entirely on its own code** — no hardcoded timings, no forced state
writes, no emulator shims in the boot path. The boot advances reset → crt0 → `main`
→ scheduler init → the first context switch (init → the first real task `0x10dfc`)
→ a **coherent scheduler idle-loop**.

"Idle" for this firmware is **not** a `waiti`. It idles by running its first task in
a poll-loop that issues a "process events" syscall and scans an **empty external
completion ring**, finding no host/array work. That is the correct post-boot resting
state: the firmware has booted and is waiting for us to act as the host.

## What unlocked it (two findings)

The arc took 44 iterations (full journey + the two Codex adversarial rounds:
[`2026-07-08-boot-wake-breach-journey.md`](2026-07-08-boot-wake-breach-journey.md)).
Two findings carried the resolution:

### 1. The `0x2450` wall was a misframed trampoline (iter42)

The long-standing wall was **not** a scheduler/wake/callback bug (all the earlier
theories — `force_done`, the a7 "switch-out hook", the event-mask — were artifacts).
The dispatcher epilogue's `0x2a86 Call0 0xdf98` is correct, but VMA `0xdf98` sits in
a gap above the `SYSCALL_BLOCK` overlay, so it was fetched at the base `+0x5c` delta
(file `0xdff4` = bytes `e0 07 00` = a bogus `Callx8 a7`) and walled on the faithful
data residue `a7=0x2450`. The **real** linked bytes are at the `+0x100` delta (file
`0xe098`): a non-windowed WINDOWBASE/WINDOWSTART **rotation helper** that returns
cleanly and never touches `a7`.

**Fix:** one `+0x100` ROM overlay `CTXSW_WINDOW_ROTATE = [0xdf98, 0xe0b1)` in
`load_m2c` — the same walk-and-stub piecewise-relocation class as the other `+0x100`
seams. No Xtensa instruction semantics changed.

This is the core reusable lesson of the whole arc: **this image uses a piecewise
VMA→file layout** — a base `+0x5c` delta plus scattered `+0x100` overlay sections —
and "dense Xtensa decodes plausibly at both offsets," so *coherence*, not decode
density, is the only valid discriminator for framing a section.

### 2. The terminal state is idle, not a wall (iter44)

Past the fix, the boot runs a steady scheduler loop (period ~458 instrs) on task
`0x10dfc`, drilled layer by layer to its root:

- The first task re-issues a **void syscall `0x6c`** each period (wrapper at Seg-B
  `0xb0424c`; `FUN_0000dab0`'s `k`/`m`/`l`/`p` = `0x6b`/`0x6d`/`0x6c`/`0x70`
  comparison tree is the **kernel syscall dispatcher**, and the codes are selectors).
- Kernel dispatch scans a **configured active-set** `[0x272003b8]=0x8000` (written 4×
  at init, then stable — a config value, not a self-clearing HW flag), extracting
  `(0x8000>>12)&0xf = 8 → bit 3 → service index 3`.
- The index-3 service chain (`FUN_00005958 → 93f0 → 9448 → 7e4c → 893c`) runs a
  **ring-scan** (`FUN_000093f0`) that loads the completion ring's head/tail
  (`[0x27200330]`/`[0x2720032c]` = `0`/`0`), finds it **empty**, and returns with no
  work. (`[0x27220000]`, read 786×, is a separate page-mapping read-modify-write
  helper, not a poll.)

No host/array events ever arrive → no work → the task loops. Go-alive (column
power-up / `publish_chann_info`) is **host-triggered**, not autonomous, so the
firmware correctly never runs it unprompted.

## The external contract surface

The apertures the idle loop touches — the host↔firmware / array↔firmware boundary a
future stimulus layer must drive:

| addr | role | idle value |
|------|------|-----------|
| `0x272003b8` | configured active column/channel set | `0x8000` (bit 15 → index 3) |
| `0x27200330` / `0x2720032c` | completion ring head / tail | `0` / `0` (empty) |
| `0x27220000` | page-map config table (RMW, not a poll) | `0` (write dropped, re-maps) |
| `0x25000003` | doorbell/status region | `0x7f` |
| `0x2505b32c` | mailbox/status | `0x100` |

## Reproduce / verify

All in `src/firmware/mod.rs` `mod boot_tests` (firmware image auto-detected):

- `cargo test --lib m2c_boot_advances_into_c_runtime` — boot runs to budget with
  `unknown_op=None`, `window_exceptions=0`, no wall.
- `cargo test --lib m2c_bit3_advances_boot_past_natural_wall` — natural vs bit3-agent
  boot both reach the same idle-loop; current-task settles on `0x10dfc`.
- `cargo test --lib ctxsw_call0_target_uses_matching_plus_100_section` — the iter42
  overlay regression guard (`Call0 @0x2a86 → 0xdf98` decodes as `rsr.windowbase`).
- `XDNA_FW_PROBE=1 cargo test --lib m2c_probe_tail_poll -- --nocapture` — the live
  idle-loop diagnostic: external/internal load-EA histogram with per-site pc+value,
  the syscall-selector histogram, and a dynamic instruction ring for one period.

## Next arc (separate phase, deliberately deferred)

Drive **go-alive** by injecting a host command / completion into the ring
(`0x27200330`/`0x2720032c` head/tail + a ring entry) and watch the firmware pick it
up, power up columns, and `publish_chann_info`. That is the start of building out the
host↔firmware ring/doorbell contract — a fresh arc opened on top of a booted
firmware, not a continuation of this wall-breaking one.
