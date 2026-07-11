# Boot-to-idle: REACHED — real XDNA mgmt firmware boots to idle on the interp

**Date:** 2026-07-10
**Issue:** #140 firmware-emulation dream / boot-to-idle.
**Branch:** `feat/m2c-mapping-boot-to-idle` (unmerged).
**Status:** PARTIALLY RESOLVED — see the 2026-07-10 arc-2 correction below.

> **CORRECTION (2026-07-10, arc-2 re-derivation).** The "idle = booted, waiting for
> the host to act" framing below (iter44) is INCOMPLETE. The steady-state the boot
> reaches is **pre-alive**: the firmware never publishes `chann_info` /
> `FW_ALIVE_OFF`, so the driver's own contract (`aie2_get_mgmt_chann_info`, polled
> immediately after `aie_psp_start` with no host kick) would log "firmware is not
> alive." Proven airtight (`m2c_probe_alive_magic_scan`: 0 runtime copies of the
> `_NPU` magic, any store width) and root-caused: the **go-alive job is orphaned at
> creation** — `task_create` (`0xd664`, from `0x3de9`, n=47362) parks run-fn
> `0x55f8` + col `0xff` into a record at `0x2320`, which is then never enqueued,
> never read, never run. The mission SPIRIT holds (the firmware runs entirely on its
> own code to a coherent scheduler steady-state, no hardcoded timings), but that
> state is earlier in the lifecycle than iter44 claimed. Live gate: **why is the
> go-alive job never enqueued after create?** Full detail in the arc-2 section at
> the bottom.

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
completion ring**, finding no host/array work. (CORRECTED by arc-2 — this is not
the post-alive host-wait state; it is **pre-alive**: the go-alive job that would
publish the channel is orphaned at creation and never runs. See the correction
banner above and the arc-2 section below.)

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

## Arc-2 (2026-07-10): the terminal state is PRE-alive — go-alive orphaned at creation

Opening the "drive go-alive by injecting a host command" arc immediately falsified
its premise: the idle is not the post-alive host-wait, because the firmware never
went alive. Re-derived fresh — the pre-iter42 scheduler map is tainted (it even
concluded a "waiter table" that is actually register-window spill), so it was NOT
trusted; every claim below is from live post-iter42 behavior.

**Proven:**
- **Pre-alive, airtight.** `m2c_probe_alive_magic_scan` scans every backing store
  for the `_NPU` magic before and after a full boot: 2 static copies (`rom@0x3388`,
  `local_data@0x332c`), **0 runtime-new** at any store width. The firmware never
  copies `chann_info` into host-visible memory ⇒ never publishes ⇒ pre-alive. The
  driver's publish is autonomous (`aie2_pci.c`: `aie2_get_mgmt_chann_info` runs
  right after `aie_psp_start`, no host doorbell), so this is a genuine "firmware is
  not alive" state, not a "waiting for a host command" one.
- **Gate = go-alive job orphaned at creation.** `m2c_probe_waypoint_hits`: the job
  is created (`0x3de9`, n=47362) but run-fn `0x55f8` / publisher `0x50e8` are NEVER
  reached. `m2c_probe_goalive_record`: `task_create` (`0xd664`) writes run-fn
  `0x55f8` + col `0xff` into a record at **`0x2320`**; write-once, never mutated
  again over 3M instrs. `m2c_probe_goalive_dispatch`: the record pointer `0x2320`
  is **never enqueued** (no store of that value anywhere) and the record is **never
  read** — nothing references it after creation.
- **The dispatcher** (`FUN_00002730`) registers four Zephyr tasks (`0x10dfc` idle,
  `0x10e58`, `0x10eb4`, `0x10f10`, pointers in the table at `0x2278`/`0x22a0..`) and
  runs only the idle task `0x10dfc` forever. The go-alive record `0x2320` is a
  separate object (a MERT run-to-completion job), not one of those tasks.

**Correction to the (tainted) prior model:** 2026-07-08 claimed go-alive is
*registered and readied only by an event via `wake_tasks_by_event_mask`*. Fresh data
disagrees: there is no registered waiter and nothing polls the record — the job is
**orphaned**, not event-gated. **Delivering an event cannot help** (no waiter to
wake). The gate is that the enqueue/registration step after create never runs.

**Open (the live frontier):** why is the go-alive job never enqueued after create?
Either (i) the create-site (`0x3de9`) is meant to enqueue but a branch there skips
it, or (ii) enqueue is a separate later "launch" pass that itself never runs (gated
upstream). Next: trace the `0x3de9` create-site continuation.

**Probes** (all gated on `XDNA_FW_PROBE=1`, in `src/firmware/boot_tests/`):
`m2c_probe_alive_magic_scan`, `m2c_probe_alive_publish`, `m2c_probe_waypoint_hits`
(`XDNA_FW_WAYPOINTS=0x3de9,0x55f8,0x50e8`), `m2c_probe_goalive_record`,
`m2c_probe_goalive_dispatch`.
