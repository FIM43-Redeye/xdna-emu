# Boot-to-idle: REACHED — real XDNA mgmt firmware boots to idle on the interp

**Date:** 2026-07-10
**Issue:** #140 firmware-emulation dream / boot-to-idle.
**Branch:** `feat/m2c-mapping-boot-to-idle` (unmerged).
**Status:** Firmware builds a valid mgmt channel autonomously and reaches the
**pre-publish `waiti` gate**; host-visible publication (SRAM copy + `FW_ALIVE_OFF`
doorbell) is **gated on an unmodeled `waiti` event** — the live frontier. See the
iter25 section (refined 2026-07-11 by the doorbell trace).

> **RESOLUTION (2026-07-10, iter25; refined 2026-07-11) — supersedes BOTH the iter44
> "waiting for host" AND the arc-2 "pre-alive / orphaned at creation" framings
> below.** The terminal state was never a wall, a host-wait, or an orphaned job: the
> go-alive job IS enqueued, but the publish path that runs it was **+0x100-misframed**
> — the same piecewise-relocation class as every earlier seam, just not yet mapped.
> The MERT queue-pop (`0xcc1c`, via `0xc648`) and its literal pool (`0x3c84`) read
> garbage at the base `+0x5c` delta (pool-base `0x06194518` instead of the SCHED base
> `0x00002250`), so the pop took its empty exit and the run-fn (`0x55f8`) never
> dispatched. Mapping the full publish path (26 `+0x100` overlays: run-fn,
> `publish_chann_info` `0x50e8`, their helpers and literal pools) makes a natural boot
> **pop the job, build a complete, structurally-valid `mgmt_mbox_chann_info`** (magic
> `_NPU` + real SRAM ring addresses + sizes + MSI id + protocol 5.8), stage it in
> local DRAM (`0x14800`), and rest at a real **pre-publish `waiti`** (`0x5645`).
> VERIFIED end to end (reproduction + byte-check + clean audit invariants + struct
> dump). **REFINED by the doorbell trace (2026-07-11, `m2c_probe_alive_sram_path`):
> the copy to the host-SRAM aperture and the `FW_ALIVE_OFF` doorbell do NOT happen
> before the `waiti`** — 0 stores land in the SRAM band, `FW_ALIVE_OFF` (`0x30bf000`)
> stays `0`. So the struct at `0x14800` is a staged copy, not the host-visible one,
> and a real driver would **not yet report alive**: host-visible publication is gated
> on whatever the `waiti` awaits. So "go-alive" is precisely "the channel is built and
> the boot reaches the publication gate," not "the driver sees it alive." The live
> frontier: identify the `waiti` (`0x5645`) wake event, supply that faithful stimulus,
> and let the firmware finish the SRAM copy + doorbell. Full detail in the iter25
> section; the arc-2 "orphaned" section below is kept only as the record of how the
> frontier moved (its conclusion is WRONG — refuted by the enqueue count byte and by
> reproduction).

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

> **SUPERSEDED by iter25 (below). This section's conclusion — "orphaned at
> creation" — is WRONG.** The go-alive job is enqueued (fixed-pool count byte
> `[0x24c4]=1`); the `m2c_probe_goalive_dispatch` here was L32i-only and watched the
> wrong window, so it missed the byte-width enqueue and byte-reads of the record.
> The real gate was a `+0x100` misframe of the queue-pop/publish path. Kept as the
> record of how the frontier moved from "pre-alive" to the misframe root cause.

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

## Arc-2 iter25 (2026-07-10): GO-ALIVE — the publish path was +0x100-misframed; mapped, firmware builds a valid channel

This is the resolution. The arc-2 "pre-alive / orphaned" conclusion above was a
measurement artifact; the real gate was one more instance of the arc's core bug
class (piecewise VMA→file relocation), on the go-alive publish path.

**Root cause.** The go-alive job IS enqueued — the fixed-pool count byte at
`[0x24c4]=1` (MERT keeps a count/index beside its embedded record pool; there is no
pointer-queue link, which is why the L32i-only, pointer-watching arc-2 probe read it
as "orphaned"). What blocked dispatch: the MERT queue-pop at VMA `0xcc1c` (reached
via the work-fetch launcher `0xc648`) and its literal pool `0x3c84` are `+0x100`
sections served at the base `+0x5c` delta. The pool-base literal then reads
`0x06194518` (garbage) instead of `0x00002250` (the SCHED base); the count read at
`[base+0x274]=[0x24c4]` lands on unbacked memory (`0`), the pop takes its empty
exit, and the run-fn `0x55f8` is never dispatched. Downstream, the run-fn section
itself and the whole `publish_chann_info` (`0x50e8`) subtree are `+0x100` too.

**The fix (iter25 in `load_m2c`).** 26 `+0x100` ROM overlays covering the publish
path: the queue-pop trio, the run-fn (`0x55f8`) and publisher (`0x501c`) code, their
called helpers (address encoders, bitfield/MMIO/NOC/array config, scheduler scan),
and their scattered live L32r literal pools. Same `add_rom_overlay(lo,hi,0x100)`
mechanism (covers both fetch and L32r literal reads) as iters16–24.

**How verified (not trusted).**
- **Reproduction** (`m2c_probe_goalive_overlay_repro`, full `XDNA_FW_OVL` set):
  `0x55f8` first-hit n=50107, `0x50e8` covered at n=50116, boot rests at
  `Wait(Waiti)` pc `0x5645` after 52,391 instrs.
- **Byte-check vs the raw sbin** (file = `(VMA & 0xffffff) + delta`): every code
  range begins with a valid `entry` prologue at `+0x100` and mid-instruction garbage
  at `+0x5c` (`0x55f8`: `36 81 00`=`entry a1,64` vs `e6 11 7a`; `0x501c`:
  `36 41 00`=`entry a1,32` vs `0b 60 80`); every pool word is sane at `+0x100`
  (queue-base `0x00002250` vs `0x06194518`; magic literal `0x3288`: `0x55504e5f` vs
  `0x08b0e290`).
- **Audit invariants** (`XDNA_FW_AUDIT=1`, over the full boot): **0** stores land in
  any overlaid VMA (not firmware-relocated data) and **0** `+0x5c` aliases of the
  canonical entries are ever executed (no dual-framing). Every word of the
  `0x325c–0x3294` pool is a live L32r target (real literals, not curve-fit).
- **Natural-boot acceptance** (`m2c_probe_alive_magic_scan`, plain `load_m2c`, NO
  extra overlays): runtime-new `_NPU` at `local_data@0x14820` — the fix is baked
  into the mapping, not the harness.

**The published channel is valid** (`m2c_probe_alive_struct`, struct base
`0x14800`, magic at `+0x20`): a complete, structurally-sound `mgmt_mbox_chann_info`,
not a bare magic stamp:

| field | value | reading |
|-------|-------|---------|
| `x2i_buf` / `x2i_buf_sz` | `0x030bc000` / `0x400` | 1 KB ring, SRAM aperture |
| `x2i_head` / `x2i_tail` | `0x030ec004` / `0x030ec000` | mailbox head/tail regs |
| `i2x_buf` / `i2x_buf_sz` | `0x030bd000` / `0x400` | 1 KB ring |
| `i2x_head` / `i2x_tail` | `0x030ed004` / `0x030ed000` | mailbox regs |
| `magic` | `0x55504e5f` | `_NPU` |
| `msi_id` | `0xe` | MSI-X vector 14 |
| `prot_major.minor` | `5.8` | protocol version |

**VERIFIED:** the firmware runs the entire go-alive publish path on its own code and
autonomously builds a valid mgmt channel descriptor, then reaches a real **pre-publish
`waiti`** (`0x5645`).

**REFINED by the doorbell trace (2026-07-11, `m2c_probe_alive_sram_path`):** the
struct is **staged** in local DRAM (`0x14800`); the host-visible publication has NOT
happened at the `waiti`. Concretely, over the whole boot to `waiti 0x5645`: **0 stores
land in the host-SRAM band** (the struct is never copied to the SRAM aperture the
driver reads), and **`FW_ALIVE_OFF` (`0x30bf000`) stays `0`** (the doorbell the driver
polls is never rung). The struct's ring fields already hold host-SRAM addresses
(`0x30bxxxx`), so the copy-out + doorbell are the *remaining* steps — and they are
**gated on whatever the `waiti` awaits** (an event we do not yet supply), not skipped.
So a real driver would **not yet report alive**. THE LIVE FRONTIER: identify the
`waiti` (`0x5645`) wake condition (which interrupt / `INTENABLE` bit, what the ISR
does), supply that faithful stimulus, and let the firmware finish the SRAM copy +
`FW_ALIVE_OFF` write. (Probe evidence: just before the `waiti`, `pc=0x563e` loads a
pointer `0x030bb000` from `[0]` — an SRAM buffer it is poised to use post-wake.)

**Post-alive driver contract (mapped, for the next arc).** Once `FW_ALIVE_OFF` is
non-zero, `aie2_pci.c` reads the 14-u32 struct, wires the i2x/x2i rings, checks
`magic == 0x55504e5f` and `prot_major/minor`, clears `FW_ALIVE_OFF`, then:
`xdna_mailbox_start_channel` → `aie2_mgmt_fw_init` (`aie2_runtime_cfg`: the
`npu1_default_rt_cfg` — the first host→firmware x2i commands) → `aie2_pm_start` →
`aie2_mgmt_fw_query` → `aie2_error_async_events_alloc` → `dev_status = AIE2_DEV_START`.
So the post-doorbell state should be the *real* host-command wait on the x2i ring.

**Overlay set (the 26 `+0x100` ranges in `load_m2c`, base-vs-coherent evidence).**

| Range `[lo,hi)` | Justification |
|-----------------|---------------|
| `c648:c6b0` | work-fetch launcher. `0xc648`: base `32 48 22`; `+0x100` `36 41 00` (`entry a1,32`) |
| `cc1c:ccb4` | MERT queue-pop. `0xcc1c`: base `62 00 41`; `+0x100` `36 41 00` |
| `3c84:3c88` | queue pool base: `0x06194518` → `0x00002250`, making `[base+0x274]=[0x24c4]` |
| `55f8:581c` | `goalive_runfn`. base `e6 11 7a`; `+0x100` `36 81 00` (`entry a1,64`); ends at `waiti 0x5645` |
| `501c:518f` | publisher block. entries `0x501c/0x5044/0x50d4` → `36 41 00 / 36 41 00 / 36 61 00`; ends `retw.n 0x518d` |
| `4a0c:4a37` | post-publish address encoder. base `23 c1 e1`; `+0x100` `36 41 00`; ends `retw.n 0x4a35` |
| `4a5c:4ade` | publish address/bitfield helpers. `0x4a5c`: base `22 55 59` → `36 41 00` |
| `7bd0:7c1e` | post-publish scheduler scan. base `0c 0a e0` → `36 41 00`; ends `retw.n 0x7c1c` |
| `7cf0:7d40` | post-publish scheduler-state helper. base `23 fe a5` → `36 41 00` |
| `86f8:8720` | interrupt/state helper (called by `0x58dc`). base `bb ff 41` → `36 41 00`; ends `retw.n 0x871e` |
| `8970:89d4` | address lookup (called by `0x9778`). base `64 1b d7` → `36 41 00`; ends `retw.n 0x89d2` |
| `8c98:8d52` | MMIO/config helper (called by `0x8f44`). base `c0 20 00` (`memw`, no entry) → `36 81 00`; ends `retw.n 0x8d50` |
| `8d88:8db4` | four bitfield updates from the publisher. base `d0 2d b0` → `36 41 00`; ends `retw.n 0x8db2` |
| `8f44:9065` | publisher MMIO mapping helper. base `f0 00 00` → `36 81 00`; ends `retw.n 0x9063` |
| `95ec:9704` | NOC/config helper family (`0x95ec/0x9628/0x967c/0x96ac/0x96d8`). base `0x95ec` `1d f0 00` (`retw.n`, impossible entry) → `36 41 00` |
| `9704:9777` | array-programming helper. base `20 50 88` → `36 41 00`; ends `retw.n 0x9775` |
| `9778:978f` | lookup/store wrapper. base `00 00 00` → `36 41 00`; ends `retw.n 0x978d` |
| `31ac:31b0` | Seg-B helper pointer: `0x00000000` → `0x08b041f0` |
| `325c:3298` | dense live publisher pool. e.g. `3288: 0x08b0e290 → 0x55504e5f` (`_NPU`); every word through `0x3294` is an executed L32r target |
| `329c:32a0` | mask/address literal: `0x00002b70 → 0x000fff20` |
| `3364:3368` | config literal: `0x27010d28 → 0x02000001` |
| `33a8:33ac` | mask/data literal: `0x00005a3c → 0xf9e8d7c6` |
| `33f4:33fc` | post-publish scheduler literals: `0x08000600/0x030d1000 → 0x0000f308/0x00011098` |
| `3474:347c` | `0x86f8` helper state bases: `0x27010d04/0x27010d08 → 0x000116f0/0x27200300` |
| `34a0:34a8` | lookup bases: `0x0005b32c/0x18e00050 → 0x00011784/0x27220040` |
| `34dc:34e8` | MMIO/config literals: `0x000084d0/0x08b28000/0xf752d024 → 0x0000fac0/0xc0000003/0x27200904` |
| `3500:3520` | one-hot mask table: 8 live words `0x00020001`..`0x01000001` |
| `3530:3534` | mask: `0xffff0fff → 0xfeffffff` |
| `354c:3564` | NOC/address pool: `0x000fff28, 0x00036030, 0x00096030, 0xfbffffff, 0x0005b340, 0x000117b0` |

**How it was found (cross-model).** Codex (GPT-5.6) did the walk-and-stub grind that
produced the 26-range set and the audit harness; Opus framed the task, reproduced
the result independently, byte-checked the load-bearing overlays against the sbin,
validated the published struct against the driver layout, and split VERIFIED from
next-frontier. Pattern: [[feedback_codex_escape_hatch_for_walls]].

**Reproduce / verify (iter25).**
- `cargo test --lib m2c_boot_advances_into_c_runtime` — natural boot reaches the
  post-alive `waiti` at `0x5645` (`reached_idle`, `wait_reason=Waiti`).
- `XDNA_FW_PROBE=1 cargo test --lib m2c_probe_alive_magic_scan -- --nocapture` —
  natural boot publishes `_NPU` at `local_data@0x14820`.
- `XDNA_FW_PROBE=1 cargo test --lib m2c_probe_alive_struct -- --nocapture` — the
  full valid channel-descriptor dump.
- `XDNA_FW_PROBE=1 XDNA_FW_AUDIT=1 cargo test --lib m2c_probe_goalive_overlay_repro -- --nocapture`
  — the reproduction harness + overlay invariants (add ranges via `XDNA_FW_OVL`).
