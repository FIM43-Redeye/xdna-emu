# Brief: what gates the firmware's alive publication? -- trace the `_NPU` builder backward to a modelable input

## Why this brief exists (the arc pivot)

Two audits this session (verified against source, file:line) reframed the boot
wall. The prior arc chased forced-path artifacts (`a7=6` reject, `0x26d4` view
collision, below-CPU bank). The audits show the firmware never *naturally*
reaches those -- it boots on **entirely defaulted device inputs** and the real
alive path is gated behind that missing input layer. This session finds the gate.

### Verified audit facts (ground truth for this task)

**A. The emulator feeds the firmware defaulted/zero device inputs (emulator-input audit).**
- `LOCAL_DATA_END = 0x0400_0000` (`src/firmware/mmio.rs:65`); `is_local_data(v)=v<LOCAL_DATA_END`
  (`mmio.rs:341-342`). So every DATA read to the host-side apertures BAR0
  (`0x0300_0000`), BAR2 shared SRAM (`0x0308_0000`), BAR4 mailbox (`0x030c_0000`)
  routes into the flat `local_data` Vec -- it round-trips only what firmware
  itself wrote, never an externally-supplied value.
- Mailbox region (`0x2700_0000..0x2800_0000`) is a plain-RAM stub
  (`mmio.rs:405-409`); the `0x272xxxxx` completion-ring / active-set / IRQ regs
  read as whatever was last written (default 0).
- "System" catch-all (SMN/NoC/off-array, incl. SMU) returns **0** always
  (`src/firmware/sysstub.rs:42-51`), with a spin-detector built to flag
  "firmware waiting on state we haven't wired up."
- Scope is acknowledged-incomplete: `mod.rs:3-4` -- "Device/mailbox MMIO routing
  into DeviceState is later (M2)." The input layer is UNBUILT, not mysteriously
  failing.
- The harness's own documented natural terminus (`boot_tests/idle_loop.rs:3-15`):
  the boot reaches a coherent scheduler steady-loop on task `0x10dfc`, then
  "idles on an empty ring, waiting for host/array events we do not supply"
  (completion ring head/tail `0x27200330`/`0x2720032c` = 0, active-set
  `[0x272003b8]=0x8000`). The `a7=6` reject is reached ONLY by the probe's
  `HARNESS_VIEW` forcing (committed finding `12a99780`); the clean run does not
  naturally reach `0x7fc7`.

**B. Alive is autonomous, pre-mailbox, gated on a tiny input surface (host-capture/driver-source audit).**
The driver's `aie2_hw_start` (`../xdna-driver/drivers/accel/amdxdna/aie2_pci.c:336-454`) runs:
1. SMU power off->on (MMIO, `aie_smu.c`; `AIE_SMU_POWER_ON=3`)
2. PSP validate->start/copy-fw, which **launches the firmware** (MMIO, `aie_psp.c`;
   `PSP_VALIDATE=1`, `PSP_START=2`, `PSP_START_COPY_FW`)
3. `aie2_get_mgmt_chann_info` (`aie2_pci.c:71-132`): the driver **polls
   `FW_ALIVE_OFF`** (SRAM BAR, device `0x030bf000`) until the firmware writes a
   nonzero pointer, then reads the firmware-authored `mgmt_mbox_chann_info`
   struct at device `0x030bb000` -- magic `"_NPU" = 0x55504e5f`, ring
   descriptors, `msi_id`, protocol major/minor (`aie2_pci.c:54-69`).
4. only THEN starts the mailbox channel (steps 5-8: SET_RUNTIME_CONFIG,
   ASSIGN_MGMT_PASID, queries, etc.).

So alive publication happens BEFORE any mailbox message exists -- it is
autonomous firmware boot behavior, and the ONLY inputs the firmware sees before
it should publish are the **SMU power state and the PSP launch handoff**, both
of which are fully open-source-documented.

**C. The reconciliation finding (committed `12a99780`)** proved the real
`_NPU`-struct builder / `FW_ALIVE_OFF` publisher is in the service continuation
behind the `a7<6` wall; the clean path only stages the *value* `0x030bb000`
into local word 0 and idles. Zero writes to `0x030bb000..03f` or `0x030bf000`
on any traced path.

## The task: find the gate, classify its input

**Anchor: `0x55504e5f` (`"_NPU"`, little-endian).** Find where the firmware
CODE constructs/writes this magic in `1502_00/npu.dev.sbin` -- that routine is
the `mgmt_mbox_chann_info` builder = the alive-struct producer. (It also writes
the device-absolute struct pointer to `FW_ALIVE_OFF`.) Then:

1. **Locate the builder.** By image byte-scan for `5f 4e 50 55` and/or by
   finding the instruction that materializes `0x55504e5f` (literal pool or
   immediate synthesis), corroborated against the HW-observed struct layout
   (magic, ring descriptors, protocol 5.8) from finding `1a7bda8d`.
2. **Trace the gate backward.** From the builder, walk up the call/branch chain
   to the earliest decision that, in our reconstruction, diverges from the path
   that would reach it. This includes but is NOT limited to the known `a7<6`
   service guard -- go further UP: why is the scheduler in the state it is? What
   seeded the go-alive task's slot/queue with the value that later rejects? Walk
   to the earliest input that steers the boot off the builder path.
3. **Classify each gating input by memory class.** For each load that feeds a
   gating decision, is the value:
   - **(a) a DEFAULTED device read** -- a region the emulator returns 0 / local-
     alias for (SMU/System stub, mailbox `0x272xxxxx` stub, BAR0/BAR2/BAR4
     aliased into local RAM) where real hardware, driven by the open-source
     driver's SMU/PSP handoff, would supply a specific value; OR
   - **(b) genuine firmware logic on correctly-modeled local state** -- the gate
     is real and the missing thing is not a defaulted input; OR
   - **(c) a value requiring real-hardware data we cannot derive** from the
     firmware binary + open-source driver.

## The discriminator / verdict

- **If (a):** name the exact defaulted input (firmware address + our region-
  handler that returns the wrong default), and the open-source driver value that
  should be there (cite `aie_smu.c`/`aie_psp.c`/`npu1_regs.c`/`aie2_pci.c`
  file:line). That input + value is the modelable fix that would let the
  firmware run its own builder and publish alive. This is the target outcome.
- **If (b):** name precisely what correctly-modeled state the builder path needs
  that is absent, and where it would come from.
- **If (c):** name the specific hardware datum and why it is underivable.

Note the address-translation seam: the driver addresses via BAR-relative
`0x0300_0000..0x030c_0000` (`npu1_regs.c`); the firmware sees a NoC/internal
`0x272xxxxx` view. Prior findings partially bridged this (mailbox at
`0x272xxxxx`; BAR0 mgmt `0x03010d7c` in `2026-07-12-0x26d4-mmio-write-timeline.md`).
Where a gating input crosses this seam, state the mapping you use and its basis.

## Deliverables

1. **Builder location** -- the `0x55504e5f` producer routine (PC range, how the
   magic is materialized), corroborated against the HW struct layout.
2. **Backward gate chain** -- ordered, each edge VERIFIED (executed) / CLAIMED
   (inferred), from the builder up to the earliest steering input, with source
   addresses + memory classes.
3. **Gate-input classification + verdict** -- (a)/(b)/(c) per above, with the
   specific defaulted input and the open-source driver value if (a).
4. **Ranked single next step** -- derive-only. If (a): the concrete input-model
   change to attempt (which region handler, what value, from which driver
   source line) -- still derive-only, supply the input, do NOT force a branch or
   inject firmware state.

## What "done" looks like

A written finding
(`docs/superpowers/findings/2026-07-13-alive-gate-provenance.md`) with the four
deliverables. Extend the env-gated probe if you need new observations;
`cargo test --lib` green (4091 baseline). Present for review -- do NOT commit.

## Execution discipline (READ -- prior run hung here)

- A prior job in this arc soft-hung 40+ min in a `wait` loop on a self-
  dispatched independent-review subagent AFTER deliverables were written. Do NOT
  dispatch an independent reviewer and do NOT enter any `wait`/collaboration poll
  loop. Do your own inline verification, write the finding, and TERMINATE. Your
  final message is the deliverable.
- If a probe run is long CPU-bound, background-and-block in ONE shell command
  (`cmd & wait $!`); do not poll in a `check -> sleep` loop.
- Reconstruct context FIRST by reading this brief + the committed reconciliation
  finding + the emulator input files named above.

## Ground rules

- Derive, do NOT calibrate. No fitted constants, no hardcoded slot/branch, no
  "make it pass" shim. If (a), the deliverable is the identified input + its
  open-source-derived value; supplying that input to the firmware's own code is
  derivation, but do NOT force a branch, inject a task/scheduler value, or byte-
  swap a path.
- Read-only observation. Test-only env-gated probe; production
  `load_m2c`/`mod.rs`/`mmio.rs`/`system.rs`/`sysstub.rs` behavior unchanged.
  `cargo test --lib` stays green (4091 baseline).
- Do NOT re-open PSP-loader RE, the CPU-self-modifying-code-to-`0x8cae` path, or
  the below-CPU bank *mechanism* hunt (all closed). This is an input-gate
  data-flow question.
- Ground everything in Phoenix `1502_00`
  (`../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`, SHA-256
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`). The
  `17f1_10` sibling is untrusted different-generation -- no importing its
  semantics without a byte match. The open-source amdxdna driver
  (`../xdna-driver/drivers/accel/amdxdna/`) IS an authoritative spec for the
  SMU/PSP handoff and register layout -- use it freely.
- Present findings for review. Do NOT commit.

## Anchors

- Firmware: `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`.
- Committed predecessor: `docs/superpowers/findings/2026-07-13-alive-publish-reconciliation.md`.
- HW struct oracle: finding `1a7bda8d` (`2026-07-11-alive-state-sram-hw-dump.md`)
  -- struct at device `0x030bb000`, `_NPU` magic, protocol 5.8, `FW_ALIVE_OFF`
  transient `0x030bb000`.
- Driver boot sequence: `../xdna-driver/drivers/accel/amdxdna/aie2_pci.c:336-454`
  (`aie2_hw_start`), `aie2_pci.c:54-132` (`mgmt_mbox_chann_info` +
  `get_mgmt_chann_info`), `aie_smu.c`, `aie_psp.c`, `npu1_regs.c`.
- Emulator input layer: `src/firmware/mmio.rs`, `sysstub.rs`, `host_mailbox.rs`,
  `psp_map.rs`; probe family `m2c_probe_26d4_cache_pageroot_timeline` in
  `src/firmware/boot_tests/coherence_mapper.rs`.
- Base commit: `12a99780`.
