# Brief: trace `a7` backward from the 0x7fc7 reject -- genuine logic or reconstructed-view artifact?

## Why this brief exists (the reframe)

Three probes have now proven a complete NEGATIVE about the `0x26d4` view switch:
it is not stock Zephyr, not host-observable, not firmware-architected (no cache/
MMU/TLB op), and not a firmware MMIO write (commits `dcc6009e`, `108f0a76`,
`67cbacf9`). The mechanism is supplied below CPU-visible state with no firmware
trigger -- a hard wall for recovering the *mechanism*.

So stop asking "what is the mechanism." Ask a different, derivable question. The
thing that actually STOPS the firmware from going alive is not a fetch fault --
it is a **self-reject**: the service sequencer `0x7fc4` evaluates `Bgeui a7,6`
at `0x7fc7` and branches to the reject sink `0x7fec` because `a7=6`. The firmware
runs, reaches a decision, and *chooses* reject over building the alive struct.

Precedent that this reframe is worth it: `0x8cae` looked for many sessions like
an exotic overlay mechanism and dissolved into a framing artifact once framed
correctly. The `a7=6` reject may be the same shape -- a boring wrong register
value with a traceable cause, not a fundamental wall.

**This session traces `a7` backward from the reject and answers ONE fork:**

- **(A) `a7=6` is genuine firmware logic** -- `a7` is computed from firmware
  state that is correct in the reconstruction, and the firmware genuinely expects
  something it is not finding (a mailbox value, a struct field, a DMA/lock/queue
  result). Then name that expected state precisely: it becomes a *derivable*,
  HW-observable target -- what the real firmware sees at that point -- with no
  calibration.
- **(B) `a7=6` is a reconstructed-view artifact** -- `a7`'s provenance runs
  through bytes or state produced by the counterfactual `HARNESS_VIEW` transport
  (the two forced view selections the probes ride), so `a7=6` is a consequence of
  imperfect view reconstruction upstream, not real firmware logic. Then the
  "overlay wall" is a *symptom* of an emulator fidelity bug -- fix the upstream
  and the gate passes on its own.

Both outcomes are progress and both are pure derivation. Report whichever the
evidence supports, with the register/memory provenance that proves it.

## The derive-not-calibrate rule (non-negotiable)

The entire reason to run the firmware is that timing DERIVES from faithful
execution with zero free parameters. Do NOT introduce a calibration, a fitted
constant, a hardcoded `a7`, a forced branch, or a "make it pass" shim. If the
answer is (A), the deliverable is the NAME of the expected state and how it would
be derived/observed -- not a value poked in. If (B), the deliverable is the
identified upstream reconstruction defect. No forcing firmware state, no
per-path byte swap, no `load_m2c` diff.

## Method: backward data-flow from the reject

Start at `0x7fc7` (`Bgeui a7,6 -> 0x7fec`, observed `a7=6`) and walk `a7`'s
provenance backward through the trace-identified critical chain:

- **`0x7fc4`** -- service sequencer; the reject lives here. Establish where `a7`
  is set before `0x7fc7`: a return value from `0x8c6c` / `task_dispatcher` /
  the `0x26d4` call, a load from an object field, or a live-in argument.
- **`0xc530`** -- service wrapper that calls `0x7fc4`. Identify the object fields
  it passes and whether `a7` (or the value that becomes `a7`) originates in the
  object it hands over. Note that `0xc530` runs BOTH before and after the later
  BASE `0x26d4` entry -- compare `a7` (or its source) across the two calls; the
  first call does not reject, the second carries `a7=6`.
- **`0x8c6c`** -- BASE-framed service subroutine (returns at `0x8cba`); the
  worklist flags it updates an object before return. Determine whether that
  update feeds `a7`.
- **`0x26d4` (BASE Entry `a1,0x50`)** -- loads task state, calls `0xc530` at
  `0x2734`. Determine what task-state field it reads and whether that field
  becomes `a7`.
- Wherever `a7`'s source is a memory load, record the effective address and
  classify it: ordinary reconstructed `local_data`, a device/SRAM value, or a
  byte that was supplied by a `HARNESS_VIEW` forced selection. **This
  classification is the (A)-vs-(B) discriminator.**

Instrument this as a read-only, env-gated probe (extend the existing
`m2c_probe_26d4_cache_pageroot_timeline` family or add a sibling in
`src/firmware/boot_tests/`). Emit an ordered provenance log: for each step,
`n / pc / op / how a7 (or its precursor) is produced / source-address+region`.
Keep the window bounded to the service chain through `0x7fec`.

## Deliverables

1. **The `a7` provenance chain** -- an ordered, evidence-backed trace from
   `0x7fc7` back to `a7`'s origin, each edge marked VERIFIED (executed) vs
   CLAIMED (inferred), with source addresses/regions.
2. **The fork verdict** -- (A) genuine or (B) artifact, with the specific
   provenance that decides it. If (A): name the exact state the firmware expects
   and the value it wants (`a7<6`), and state how that state would be derived or
   HW-observed. If (B): name the upstream reconstruction defect and the earliest
   point it corrupts `a7`'s source.
3. **The single next step** -- ranked. For (A): the concrete derivation/HW
   observation that supplies the expected state. For (B): the reconstruction fix
   to attempt (still derive-only, no forcing).

## What "done" looks like

A written finding
(`docs/superpowers/findings/2026-07-12-a7-reject-provenance.md`) with the three
deliverables. Plus the env-gated probe, `cargo test --lib` green (4091
baseline). Present for review -- do NOT commit.

## Execution discipline

If any run is long CPU-bound, background-and-block in ONE shell command
(`cmd & wait $!`); do not poll in a `check -> sleep -> "still running"` loop.

## Ground rules

- Derive, do NOT calibrate. No fitted constants, no hardcoded `a7`, no forced
  branch, no "make it pass" shim.
- Read-only. No forcing firmware state, no `load_m2c` diff, no per-path byte
  swap. Instrument and reason.
- Test-only probe code; production `load_m2c`/`mod.rs`/`mmio.rs`/`system.rs`
  behavior unchanged.
- `cargo test --lib` stays green (4091 baseline) if you touch any probe.
- Do NOT re-open PSP-loader RE, the CPU-self-modifying-code-to-0x8cae path, or
  the below-CPU bank *mechanism* hunt (that negative is closed). This is a
  data-flow question about `a7`, not a mechanism question.
- Ground everything in the Phoenix `1502_00` image. The `17f1_10` sibling is an
  untrusted, different-generation hint source -- do not import its semantics
  without a byte-level match against Phoenix.
- Present findings for review. Do NOT commit.

## Anchors

- Firmware: `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`
  (SHA-256 `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`).
- Reject site + worklist: `2026-07-12-0x26d4-mmio-write-timeline.md`
  (worklist rows `0x7fc4`, `0xc530`, `0x8c6c`, `0x26d4`); the reject is
  `Bgeui a7,6 -> 0x7fec` at `0x7fc7`, observed `a7=6`.
- Prior probe family to extend: `m2c_probe_26d4_cache_pageroot_timeline` in
  `src/firmware/boot_tests/coherence_mapper.rs`.
- Counterfactual transport (the (B) suspect surface): the two `HARNESS_VIEW`
  forced selections described in `2026-07-12-0x26d4-cache-pageroot-timeline.md`.
- Base commit: `67cbacf9`.
