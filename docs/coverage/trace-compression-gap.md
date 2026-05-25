# Trace unit wire-format compression gap

**Status**: OPEN, unscheduled. Carved out from
`cycle-accuracy-mission.md` item #9 on 2026-05-25 after that item's
H1 (perf-counter-driven LOCK_STALL emission) was refuted.

## Summary

HW's trace controller compresses runs of repeated events using
skip-tokens in the trace packet stream. EMU's trace unit emits one
packet per cycle uncompressed. The post-decode events.json is
semantically the same on both sides, so this is purely a wire-format
gap -- but it surfaces as:

- **trace_raw.bin size asymmetry**: HW 160 content words vs EMU 3032
  on `_diag_phase_b_add_one_instrumented.chess` (19x).
- **`ts` field asymmetry** in parse-trace.py output: HW assigns the
  chunk's bookkeeping soc to every event unpacked from a single
  packet block (so 2233 LOCK_STALL events share 2 distinct soc
  values), while EMU emits one packet per event so `ts` increments
  per event. This is the proximate cause of the ts/soc gotcha
  ([finding](../superpowers/findings/2026-05-25-trace-ts-vs-soc-measurement-gotcha.md)).

Closing this lets cross-side comparisons that read `ts` work
naturally. soc-only analysis already works today.

## What needs to change

Trace unit encoder in `src/device/trace_unit/`. The current
implementation appears to emit one event marker per cycle without
inspecting whether the previous cycle's event(s) matched. HW's
encoding uses `0xFE`-byte patterns (visible in the raw bin samples
in the finding doc) to express "N cycles passed without an event
change" or similar; need to identify the exact format from
`tools/trace_decoder/modes/mode0.py` (or upstream mlir-aie's
encoder reference) and reproduce it.

## Why it isn't urgent

- Cycle-accuracy: doesn't move the needle. Post-decode event counts
  and soc values are already comparable.
- Bridge tests: pass either way -- they consume the decoded
  events.json, not the raw bin.
- Disk: 1MB trace buffers don't fill up on normal runs.
- Cross-tool comparisons: works as long as analysts use `soc`, per
  the discipline rule. The persistent stage-decomposition output in
  `trace-compare --stages` already uses soc.

## Why it matters anyway

- `ts` is the trace decoder's *intended* timestamp field for
  in-stream event ordering. Once both sides use compatible
  compression, `ts` becomes trustworthy and the soc-discipline rule
  can be relaxed.
- New analysts will keep tripping over the asymmetry until it
  closes. Each trip wastes a session.
- Future trace-decoder work (sweep tooling, regression matrices,
  perfetto exporters) becomes simpler when both sides emit the same
  compressed format.

## Validation gate

After implementing:

1. EMU trace_raw.bin content words on
   `_diag_phase_b_add_one_instrumented.chess` drop from ~3000 to
   within 2x of HW's ~160 (perfect match unlikely; small
   implementation differences are OK as long as both sides scale
   the same way with stall duration).
2. Post-decode events.json from EMU stays semantically the same
   (same event names, same counts within run-to-run noise).
3. parse-trace.py's `ts` values from EMU now track HW's chunked
   pattern (same events sharing the same ts within a packet block).

## See also

- [`../superpowers/findings/2026-05-25-lock-stall-cadence-h1-refuted.md`](../superpowers/findings/2026-05-25-lock-stall-cadence-h1-refuted.md) -- the finding that made this its own gap
- [`../superpowers/findings/2026-05-25-trace-ts-vs-soc-measurement-gotcha.md`](../superpowers/findings/2026-05-25-trace-ts-vs-soc-measurement-gotcha.md) -- the proximate-cause description
- [`cycle-accuracy-mission.md`](cycle-accuracy-mission.md) item #9 -- where this was carved from
- `src/device/trace_unit/` -- where the fix lands
- `tools/trace_decoder/modes/mode0.py` -- decoder reference for the wire format
