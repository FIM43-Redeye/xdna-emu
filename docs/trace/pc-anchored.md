# PC-anchored trace comparison

Walks through running a mode-1 (EventPC) trace sweep on a compiled bridge
test, then comparing HW vs EMU traces with `trace-compare --pc-anchored`.

## What this is for

Standard mode-0 traces use cycle deltas as the timeline. That works for
catching divergences in *which* events fire, but cycle alignment between
HW and EMU is fragile: a few-cycle skew at start, a different IRQ pattern,
or a DMA pipeline depth difference, and the two traces drift apart.

Mode-1 (EventPC) replaces cycle deltas with the *retire program counter*
of each instruction-class event. PCs are stable across runs in a way
cycle counts aren't -- the same instruction always retires at the same
PC -- so divergence comparison becomes a set / multiset diff over PCs
instead of cycle alignment.

The performance counter (`PERF_CNT_0`) is configured to overflow every
N cycles (default 1024). Each overflow records an event with the PC at
that moment, giving a *cycle clock* anchored to PCs instead of cycles.
With both sides anchored to the same clock, a band of cycles between
two consecutive overflows can still be compared meaningfully.

## Quick start

End-to-end run on `add_one_using_dma`, the smallest reliably-passing
test for both compilers:

```bash
# 1. Compile a traced binary in mode 1 (manual one-off; the bridge gate
#    automates this via --trace=pc-anchored).
cd mlir-aie/build/test/npu-xrt/add_one_using_dma
python3 ~/npu-work/xdna-emu/tools/mlir-trace-inject.py \
  --input ../add_one_using_dma.mlir \
  --out  ../add_one_using_dma.traced.mlir \
  --trace-mode event_pc \
  --core-grounding "PERF_CNT_0,INSTR_EVENT_0,INSTR_EVENT_1" \
  --perfcnt-period 1024
aiecc.py --xchesscc --xbridge ../add_one_using_dma.traced.mlir

# 2. Run the sweep (produces sweep-manifest.json + per-batch traces).
python3 ~/npu-work/xdna-emu/tools/trace-sweep.py \
  --test add_one_using_dma --compiler chess \
  --tiles "0:2:core,0:2:memmod" \
  --out-dir /tmp/claude-1000/pc-anchored-demo \
  --mode event_pc \
  --core-grounding "PERF_CNT_0,INSTR_EVENT_0,INSTR_EVENT_1" \
  --memmod-grounding "PERF_CNT_0" \
  --core-sweep all --memmod-sweep all \
  --with-mode2-baseline \
  --reuse-ctx

# 3. Run the comparator and print the report.
cargo build --release --bin trace-compare
./target/release/trace-compare \
  --sweep /tmp/claude-1000/pc-anchored-demo \
  --pc-anchored \
  -o /tmp/claude-1000/pc-anchored-demo/report.txt
less /tmp/claude-1000/pc-anchored-demo/report.txt
```

The bridge script wraps this:

```bash
./scripts/emu-bridge-test.sh add_one_using_dma --trace=pc-anchored
```

## Reading the report

`trace-compare --pc-anchored` produces three sections per tile, on top
of the standard mode-0 per-batch report.

### Coverage matrix

Lists every event the sweep saw, one row per event, one column per
batch. Cells:

| Cell | Meaning |
|------|---------|
| `grounding` | Reserved in a fixed slot for every batch (anchor). |
| `BNN`       | Sweep slot in batch `NN`. |
| (empty)     | Not covered in that batch. |

`PERF_CNT_0` should appear as `grounding` for every batch -- if any
batch shows it as `BNN`, the grounding contract was violated.

### PC-anchored divergences

Sorted by `set_diff + multiset_diff` magnitude, descending, then
alphabetical. Each row:

```
EVENT_NAME       set_diff: <hw_only>/<emu_only>   multiset: <delta>
```

`set_diff` is "PCs that appear on HW but not EMU" (and vice versa).
`multiset_diff` is "for PCs that appear on both sides, how many more
times one side fires than the other." A pure timing skew shows up as
multiset only; a missing event shows up in set_diff.

### Perfcnt-anchored cycle deltas

For each event, for each PC seen on either side, the report
linearly interpolates a cycle estimate from the surrounding two
`PERF_CNT_0` overflows. `delta_cycles` is `hw_estimate - emu_estimate`
for that PC; `exceeds_tolerance` flags deltas larger than half the
overflow period.

If this section says "no PC-anchored data" or is missing entirely:
either the perfcnt clock isn't symmetric (see below), or there aren't
enough ticks on each side to anchor.

## When the report says `unsafe_for_pc_join`

`sweep-manifest.json` records `unsafe_for_pc_join: true` if the sweep
detected that the grounding event PC drifted between batches. The
report prefixes the PC sections with a warning:

```
WARNING: PC-anchored sections may be unreliable -- grounding event
         PC drift detected (unsafe_for_pc_join).
```

Drift means one of:
- The compiler reordered the kernel between sweeps (shouldn't happen
  with `--reuse-ctx` since the same xclbin runs every batch).
- A divergent control-flow path executed in some batches but not
  others, taking grounding events to different PCs.
- The grounding event itself fires inconsistently (a perf counter
  threshold mis-set, or a rare instruction-class event).

Treat the PC-anchored sections as an indicator only, and fall back to
the standard mode-0 per-batch report for ground-truth divergences.

## When cycle bands are suppressed

Three suppression conditions, all visible in the report's tick counts
(`PCAnchoredReport.hw_perfcnt_tick_count` / `emu_perfcnt_tick_count`):

1. **Asymmetric perfcnt:** one side has zero ticks. The report skips
   cycle bands entirely and prints "asymmetric perfcnt clock --
   suppressing bands."
2. **Single-tick anchor:** either side has exactly one tick. Two
   anchors are needed for any interpolation, so bands are suppressed.
3. **Insufficient ticks for period estimation:** if neither side has
   two consecutive overflows, the comparator falls back to
   `DEFAULT_PERFCNT_PERIOD` (currently 1024). Bands are still emitted
   if both sides cleared the two-tick threshold; only the *period*
   estimation reverts to the default.

If you expected bands and don't see them, the first thing to check is
whether the perfcnt configuration actually committed -- inspect the
xclbin for `Performance_Control2` writes, or the events JSON for
`PERF_CNT_0` rows on both sides.

## Mode-2 baseline

`--with-mode2-baseline` (default true) runs one extra HW-only batch
in mode 2 (instruction execution) at the end of the sweep. Mode 2
records every retiring instruction's PC, giving a ground-truth PC
trace for the kernel as the silicon actually ran it.

This is not yet wired into automatic comparison -- mode-2 EMU support
is deferred to A.2b -- but the on-disk baseline is preserved under
`<sweep-dir>/mode2-baseline/<test>/` so future EMU tooling has a
reference HW trace to validate against.

## Defaults and tuning

Defaults live in two coordinated files; keep them in sync if either
changes:

- `tools/perfcnt_defaults.py` -- `DEFAULT_PERFCNT_PERIOD = 1024`
  (Python tools).
- `src/trace/compare.rs` -- `DEFAULT_PERFCNT_PERIOD: u64 = 1024`
  (Rust comparator fallback).

Tuning the period:

- **Smaller period** (e.g. 256) -- finer cycle bands, more trace
  volume, more `PERF_CNT_0` events. Use when investigating sub-1024
  cycle differences in a tight kernel section.
- **Larger period** (e.g. 4096) -- coarser bands, less trace volume.
  Use for long-running kernels where the per-batch trace buffer
  overflows.

## References

- Plan: `docs/superpowers/plans/2026-04-25-a2-pc-anchored-validation.md`
- Mode-1 byte format: `tools/trace_decoder/modes/mode1.py`
- Encoder: `src/device/trace_unit/mod.rs::encode_event_pc`
- Comparator: `src/trace/compare.rs::compare_pc_anchored_for_tile`
- Bridge integration: `scripts/emu-bridge-test.sh` Phase 5b'
