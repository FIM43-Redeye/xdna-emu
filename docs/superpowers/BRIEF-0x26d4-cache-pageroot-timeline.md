# Brief: read-only cache / page-root timeline across the 0x26d4 view transition

## Why this brief exists

Your orientation finding (`2026-07-12-memory-architecture-model.md`, committed
`dcc6009e`) resolved the identity-map paradox and named the strongest remaining
mechanism classes for the low-VMA temporal re-view. Its ranked next step #2 is
the cheapest dynamic discriminator we have, and it also closes two real
fidelity holes independent of the boot arc:

- the interpreter decodes 19 cache operations but models all as no-ops
  (`src/firmware/xtensa/interp/system.rs:1-7,103-168`);
- the existing ITLB probe never inventoried page-root writes.

**This is that probe. Build a read-only instruction/control timeline of what the
FIRMWARE ITSELF does across the `0x26d4` transition, and report what it implies
about the view-switch mechanism.** No forcing, no `load_m2c` diff, no firmware
state injection, no per-path byte swap. Instrumentation and reasoning only.

## The question this answers

At the `0x26d4` VMA the firmware needs the AT view early (the AT stream crosses
it) and the BASE view later (`0x7fe7 Call8 0x26d4` needs the BASE `Entry`). The
discriminator: **does the firmware execute anything at/around that crossing that
could be the view selector?**

- If the firmware fires an I-side cache op, a `PTEVADDR`/`RASID` write, a
  pinned-TLB change, or a page-root switch right at the transition -> that names
  (or strongly narrows) the mechanism, and tells us it is firmware-driven.
- If the firmware does NOTHING of that kind across the crossing -> that RULES OUT
  a firmware-active switch and points hard at an external / HW instruction bank
  or an agent below the CPU. Equally decisive, opposite direction.

Either outcome is a real result. Report the negative as confidently as a hit.

## Deliverables

1. **A read-only timeline probe** (test-only, env-gated, in
   `src/firmware/boot_tests/` -- do NOT touch production `load_m2c`/`mod.rs`/
   `mmio.rs` behavior; observe, don't alter). Between the early AT crossing of
   `0x26d4` and the later BASE `Entry` at `0x26d4` (`n≈53784` in the current
   run), record every:
   - executed cache operation (all 19 kinds), with opcode and effective address,
     flagging I-side vs D-side;
   - `WSR`/`WITLB`/`WDTLB` touching `PTEVADDR`, `RASID`, `ITLBCFG`, `DTLBCFG`,
     or any pinned-TLB entry, with old->new value;
   - ITLB/DTLB way/entry changes (extend the existing ITLB-op inventory to
     summarize ALL page-root-relevant writes, not just count them);
   - call/return boundaries (`Call8`/`Entry`/`Retw`) that bracket the crossing,
     so we can see the control context the switch happens in.
   Emit it as an ordered `n / pc / op / detail` log, gated behind an env var like
   the existing probes. Keep the window bounded (early AT epoch through the BASE
   service sink at `0x7fec`); don't dump the whole 53k-instruction boot.

2. **The verdict.** From the timeline, state which class the evidence supports:
   firmware-driven (cache/page-root/pinned-TLB -- name the exact op) vs
   external/HW bank (firmware does nothing at the crossing). Cite the specific
   `n`/`pc`/op lines. If it is ambiguous, say what single additional observation
   would break the tie.

3. **Fidelity-hole note.** Whatever you find about the 19 no-op cache ops and the
   page-root write inventory, record it as a fidelity observation (these are
   genuine model gaps regardless of the boot arc). If any cache op's no-op
   modeling is now clearly wrong, say so -- but do NOT change production cache
   semantics in this pass; that is a separate, scoped change.

## What "done" looks like

A written finding
(`docs/superpowers/findings/2026-07-12-0x26d4-cache-pageroot-timeline.md`)
with: (a) the bounded timeline (or its salient extract); (b) the mechanism-class
verdict with cited lines; (c) the fidelity-hole note; (d) the single next
observation that would most sharpen the verdict. Plus the probe code itself,
env-gated, `cargo test --lib` green (4090 baseline). Present for review -- do
NOT commit.

## Execution discipline (important)

If any step is a long CPU-bound run (a full boot to `n≈53k`, a sweep), launch it
so it **backgrounds and blocks in ONE shell command** (`cmd & wait $!` or
`a & b & wait`) -- make a single tool call that returns only at completion. Do
NOT poll in a `check -> sleep -> "still running" -> repeat` loop; that burns a
model turn per check. A single boot-to-53k is not long, but batch anything
heavier this way.

## Ground rules

- Read-only. No forcing, no `load_m2c` diff, no firmware-state injection, no
  per-path byte swap. Instrument and reason.
- Test-only probe code; production `load_m2c`/`mod.rs`/`mmio.rs`/`system.rs`
  behavior unchanged. (You may READ all of them; you may ADD a gated probe.)
- `cargo test --lib` stays green (4090 baseline) if you touch any probe.
- Do NOT re-open PSP-loader RE or the CPU-self-modifying-code-to-0x8cae path
  (both closed). A cache/bank/page-root mechanism is a different, in-scope class.
- Present findings for review. Do NOT commit.

## Anchors

- Firmware: `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`
  (SHA-256 `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`).
- Cache ops modeled as no-ops: `src/firmware/xtensa/interp/system.rs:1-7,103-168`.
- Existing ITLB-op probe: `m2c_probe_itlb_code_view_selector` in
  `src/firmware/boot_tests/coherence_mapper.rs`.
- The 0x26d4 transition trace (current-session): finding
  `2026-07-12-memory-architecture-model.md:168-184,227-232`.
- Model + holes just committed: `dcc6009e`.
