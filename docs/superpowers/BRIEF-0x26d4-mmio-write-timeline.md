# Brief: MMIO-write timeline across the 0x26d4 view transition

## Why this brief exists

The cache/page-root timeline (`2026-07-12-0x26d4-cache-pageroot-timeline.md`,
committed `108f0a76`) proved the firmware executes NO architected Xtensa selector
(I-cache op, page-root/config write, ITLB op, ITLB change) across the `0x26d4`
view flip. The surviving mechanism class is a below-CPU / external instruction
bank -- consistent with the AMD-private `sram_alias`/`mpnpu/mmu` seam.

But that timeline had one blind spot by construction: **a plain MMIO store to a
vendor bank-select register would not appear in a cache/TLB timeline** -- it is
just a store to a device address, which the emulator currently treats as an
inert device write. This brief closes that blind spot.

## The question this answers

Does the firmware perform an **MMIO write to a device/vendor register** in the
`0x26d4` transition interval that could be a bank/alias selector?

- **If yes:** that names the selector register and its value. Correlate it with
  the `sram_alias` slot facility (power-of-two slots, index < 64) from
  `2026-07-12-memory-architecture-model.md`. Mechanism found -> the RE of
  `sram_alias.c` then has a concrete register target.
- **If no (empty MMIO timeline):** the switch is triggered by a PURE external
  agent (PSP / DMA-side) with no firmware-side MMIO trigger at all. Also
  decisive -- it means RE of the firmware image alone will not reveal a trigger,
  and the hunt must move to the external/PSP side. Report the negative as
  confidently as a hit.

## Deliverables

1. **A read-only MMIO-write timeline probe** (test-only, env-gated, in
   `src/firmware/boot_tests/`; do NOT change production `load_m2c`/`mod.rs`/
   `mmio.rs` behavior -- observe, don't alter). Over the same bounded interval as
   the cache/page-root probe (early AT crossing of `0x26d4` through the service
   sink at `0x7fec`), record every store whose effective address is **outside
   normal local data/stack** -- i.e. device / MMIO / high-region / Segment-B-
   control addresses, anything that is not ordinary `local_data` RAM. For each:
   `n / pc / store-width / effective-address / value / caller-chain`. Annotate
   the caller chain (the `Call8`/`Entry` frame the store executes in) so a hit
   is attributable to a named function.
   - Reuse the existing counterfactual `HARNESS_VIEW` transport from the
     cache/page-root probe to traverse both epochs; mark those events "not
     firmware" and exclude them from the store counts, exactly as that probe does.
   - Classify each recorded store's target region (device BAR aperture,
     `0x2000xxxx` high alias, `0x08b00000` Segment B, other) so the verdict can
     say WHERE any candidate write lands.

2. **The verdict.** State whether any store in the interval is a plausible
   bank/alias selector (target region, value shape, caller). If one is, name it
   and say how it maps to the `sram_alias` slot model. If none is, state the
   clean negative and its implication (external-agent-only). Cite `n`/`pc`/EA
   lines.

3. **Worklist seed for static RE.** List the named functions on the `0x26d4`
   critical path that the timeline touches (the dispatch chain `0xc530`,
   `0x7fc4`, `0x8c6c`, `0x26d4`, the Segment-B `0x08b0exxx` helpers, etc.) with a
   one-line note on what each appears to do. This seeds the function-by-function
   "what does each function expect" RE track -- so that RE starts from the
   trace-identified critical path, not cold.

## What "done" looks like

A written finding
(`docs/superpowers/findings/2026-07-12-0x26d4-mmio-write-timeline.md`) with:
(a) the MMIO-write timeline (or its salient extract) with region classification;
(b) the selector verdict (named register+value / clean external-agent negative),
cited; (c) the critical-path function worklist seed. Plus the env-gated probe,
`cargo test --lib` green (4091 baseline). Present for review -- do NOT commit.

## Execution discipline

If any run is long CPU-bound, background-and-block in ONE shell command
(`cmd & wait $!`); do not poll in a `check -> sleep -> "still running"` loop.
A single boot-to-53k is not long, but batch anything heavier this way.

## Ground rules

- Read-only. No forcing, no `load_m2c` diff, no firmware-state injection, no
  per-path byte swap. Instrument and reason.
- Test-only probe code; production `load_m2c`/`mod.rs`/`mmio.rs`/`system.rs`
  behavior unchanged. (READ all of them; ADD a gated probe.)
- `cargo test --lib` stays green (4091 baseline) if you touch any probe.
- Do NOT re-open PSP-loader RE or the CPU-self-modifying-code-to-0x8cae path
  (both closed). A device MMIO bank selector is a different, in-scope class.
- Present findings for review. Do NOT commit.

## Anchors

- Firmware: `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`
  (SHA-256 `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`).
- Prior probe to reuse the transport from: `m2c_probe_26d4_cache_pageroot_timeline`
  in `src/firmware/boot_tests/coherence_mapper.rs`.
- Cache/page-root finding: `2026-07-12-0x26d4-cache-pageroot-timeline.md`.
- sram_alias seam + BAR map: `2026-07-12-memory-architecture-model.md`
  (BAR2 = shared SRAM device `0x03080000+`; `sram_alias.c` slots power-of-two,
  index < 64).
- Base commit: `108f0a76`.
