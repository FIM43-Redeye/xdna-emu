# A.2b: Mode-2 (Execution) decoder + EMU encoder (2026-04-28)

## Status: deferred from A.2

A.2 PC-anchored validation landed without EMU support for trace mode 2
("Execution" / `inst_exec` mode).  Mode 2 records the instruction
execution stream itself: per-cycle E_atom / N_atom (executed / stalled)
plus New_PC frames at taken branches and LC frames at zero-overhead-loop
boundaries.  This is the most fine-grained trace HW supports.

A.2 already laid the groundwork:
- HW-only mode-2 baselines are captured by `trace-sweep.py
  --with-mode2-baseline` (default true) and preserved on disk under
  `<sweep-dir>/mode2-baseline/<test>/<batch>.events.json`.
- The HW-side bit-stream decoder lives at
  `tools/trace_decoder/modes/mode2.py` -- recovered from the trace
  decoder library symbols, original implementation.
- `compare_sweep_dir_with_opts` in `src/trace/compare.rs` lists mode-2
  baselines in the report so users know which fixtures are available
  even without comparison wired up.

What's missing is everything on the EMU side and the comparator.

## What A.2b owns

### 1. EMU mode-2 encoder

`src/device/trace_unit/mod.rs::commit_pending_frame` currently no-ops
for `TraceMode::Execution` and `TraceMode::Reserved`:

```rust
TraceMode::Execution | TraceMode::Reserved => {
    // Mode 2 not implemented per A.2 spec; mode 3 is reserved.
    // Skip rather than corrupt the stream.
}
```

Replacement work:
- Per-cycle emit E_atom or N_atom based on whether the core retired an
  instruction (already tracked in `CoreState::active_this_cycle`).
- Detect taken branches and emit New_PC frames with the 14-bit retire
  PC.  Source: `EventLog` already records branch retirements as
  `EventType::Branch { target_pc, .. }` (verify exact variant name;
  may be `EventType::InstrEvent` with a discriminator).
- Detect zero-overhead-loop boundaries and emit LC frames.  Source:
  the LP_START / LP_END event types in the core's event log, which
  the coordinator's Phase 2 drain already sees.
- Emit Filler0 padding to align frames to 32-bit word boundaries (the
  encoder packs several short frames per word with trailing 0010
  filler).
- Emit Start (0xF2 / mode-2 prefix) and Stop opcodes at segment
  boundaries.  Verify the prefix byte from `mode2.py`.

### 2. EMU mode-2 frame format

Recoverable from `tools/trace_decoder/modes/mode2.py`'s frame tree.
Each 32-bit word is bit-packed MSB-first.  Quick reference:

| Prefix     | Frame    | Payload                |
|------------|----------|------------------------|
| 0000       | N_atom   | --                     |
| 0001       | E_atom   | --                     |
| 0010       | Filler0  | --                     |
| 010        | LC       | 1b flag + 28b count    |
| 10         | New_PC   | 14b absolute PC        |
| 1110       | Repeat0  | 4b count               |
| 110110     | Repeat1  | 10b count              |
| 110111     | Stop     | 26b (consumes word)    |
| 11110      | Start    | 1b flag + 14b anchor PC|
| 11111110   | Filler1  | --                     |
| 11111111   | Sync     | --                     |

The encoder needs RLE detection (Repeat0 / Repeat1) for runs of
identical E_atoms / N_atoms to keep trace volume tractable on long
kernels.

### 3. Comparator logic

`src/trace/compare.rs` doesn't currently compare mode-2 streams.
Once both HW and EMU produce mode-2 traces:
- Parse both streams via the mode-2 decoder.
- Compare cycle-by-cycle E/N atoms (a per-cycle execution divergence).
- Compare New_PC sequences (branch divergence -- this is the most
  valuable signal).
- Report PC-by-PC where the EMU stops matching HW, with the run-up
  context.

Mode-2 is fundamentally a per-cycle trace, so divergence reports
will be much denser than mode-0/1; the reporter should probably
collapse runs of agreement and only show divergence windows.

## Entry points when we resume

1. `src/device/trace_unit/mod.rs::commit_pending_frame` -- replace the
   no-op arm.
2. `src/device/trace_unit/mod.rs::encode_event_pc` -- model the new
   `encode_*_atom`, `encode_new_pc`, `encode_lc`, `encode_repeat`
   helpers after this one's structure.
3. New tests in `src/device/trace_unit/tests.rs::DecodedFrame` --
   extend the test enum with `Atom { exec: bool }`, `NewPc`, `Lc`,
   `Repeat` variants and a `decode_mode2_for_test` helper mirroring
   `mode2.py`.
4. `src/interpreter/engine/coordinator.rs::run_n_cycles` -- wire the
   per-cycle E/N atom emission into the existing Phase 3f
   `commit_cycle` call; wire branch / loop event detection into the
   Phase 2 drain.
5. `src/trace/compare.rs` -- new `compare_mode2_for_tile` function and
   its integration into the sweep aggregator.

## Why deferred

A.2's value proposition was the *PC-anchored* divergence signal --
finding "where do HW and EMU disagree" without cycle-perfect alignment.
Mode-2 needs cycle-by-cycle alignment to be meaningful (every cycle
emits an atom), which puts it firmly in the cycle-accuracy regime
that A.5 / cycle-budget testing addresses.  Landing mode-2 before
the cycle accumulator was solid would have produced a torrent of
"divergence" noise from sub-cycle skew.

With A.5 in place and mode-2 baselines already captured, A.2b becomes
worth doing: we have ground truth and we have a stable cycle clock to
compare against.

## References

- A.2 plan: `docs/superpowers/plans/2026-04-25-a2-pc-anchored-validation.md`
  (mode-2 deferral noted lines 1384, 1741, 1765)
- A.2 design: `docs/superpowers/specs/2026-04-25-a2-pc-anchored-validation-design.md`
  (mode-2 out-of-scope at line 51)
- HW decoder: `tools/trace_decoder/modes/mode2.py`
- User doc reference: `docs/trace/pc-anchored.md` (mode-2 baseline section)
