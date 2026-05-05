# Peano mode-2 trace BO empty — diagnosis was wrong; symptom traces to a contaminated sweep

**Status**: investigation 2026-05-04 → root cause re-evaluated 2026-05-05.
The Peano-vs-Chess differential we documented was an artifact of a
contaminated sweep, not a real Peano-specific HW behavior. We already
emit the upstream-prescribed fix
([mlir-aie #2001](https://github.com/Xilinx/mlir-aie/issues/2001) /
[PR #2058](https://github.com/Xilinx/mlir-aie/pull/2058)). A separate,
unrelated regression is producing empty HW trace BOs in single-run
bridge mode as of 2026-05-04 — that's tracked as a new investigation,
not this entry.

## Original (incorrect) hypotheses, kept for context

**Hypothesis 1, 2026-05-04** — fixed-length window: the Peano build's
ELF was 5.6× smaller than Chess's (`add_one_using_dma`: chess 10,716
bytes, peano 1,916 bytes), so the smaller, faster Peano kernel finished
before a fixed-length trace window in npu-instruction-cycles space could
capture useful events. **Refuted.** The trace window is event-bounded,
not duration-bounded, and the per-batch HW trace pattern was
inconsistent across runs of the same test.

**Hypothesis 2, 2026-05-04** — duplicate of upstream #2001 / #2058: we
need to use packet-routed tracing and add
`trace_utils.gen_trace_done_aie2(ShimTile)` at the end of the runtime
sequence. **Already implemented.** Our injector at
`tools/mlir-trace-inject.py` calls `aied.trace_host_config` +
`aied.trace_start_config`, which lower (via `AIEXInlineTraceConfig`)
into exactly the post-#2058 register sequence. Inspecting the
post-injection `input_with_addresses.mlir` for any traced fixture
shows:

- A packet flow from compute tile Trace-port to shim DMA:
  ```
  aie.packet_flow(1) {
    aie.packet_source<%tile_0_2, Trace : 0>
    aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
  } {keep_pkt_header = true}
  ```
- A trace-done event-fire at the tail of the runtime sequence:
  ```
  aiex.npu.write32 {address = 213064, col=0, row=0, value = 126}  // 0x34048: broadcast 14 ← USER_EVENT_2
  aiex.npu.write32 {address = 213000, col=0, row=0, value = 126}  // 0x34008: Event_Generate USER_EVENT_2
  ```
  This pair is exactly `gen_trace_done_aie2(shim, brdcst_num=14, user_event=USER_EVENT_2)`.

We are not missing the upstream fix.

## What actually produced the empirical pattern

The 2026-05-01 sweep (which seeded the original hypothesis) showed
alternating empty/non-empty per-batch HW trace sizes, e.g. for
`add_one_using_dma.chess.pc-anchored`:

| batch    | HW trace nonzero bytes |
|---------:|-----------------------:|
| batch_00 | 1514 |
| batch_01 | 0 |
| batch_02 | 48 |
| batch_03 | 0 |
| batch_04 | 1459 |
| batch_23 | 0 |

That's not a Peano-vs-Chess thing. It's the trace-BD-reuse
contamination across batches that we tracked under task #311 and the
2026-05-04 trace-sweep contamination fix in `tools/trace-sweep.py`
(separate `RunnerSession` instances for the mode-2 baseline). When we
originally pulled the "tests affected: add_one_using_dma,
add_21_i8_using_dma_op_with_padding,
add_256_using_dma_op_no_double_buffering, add_maskwrite" list, we were
sampling tests that happened to land in empty batches in that
particular run, and the Peano-vs-Chess differential was an
artifact of which compiler-test pair fell into which batch.

## What's actually broken right now (as of 2026-05-04)

Spot-check of `build/bridge-test-results/20260504/` (single-run
bridge mode, post-contamination-fix): only 2 out of ~70 tests have
any non-zero bytes in their HW `trace_raw.bin`:

- `dmabd_task_queue.chess.hw/trace_raw.bin`: 173 nonzero bytes
- `dmabd_task_queue.peano.hw/trace_raw.bin`: 176 nonzero bytes

Every other test, both Chess and Peano, is empty. This is uniform
across compilers, so it isn't a Peano issue and isn't a #2058 issue.
It's some other regression in how single-run-mode bridge tests
copy the HW trace BO out, possibly introduced alongside one of the
2026-05-04 contamination/reset fixes. Tracked separately — see the
"Single-run HW empty-trace regression" task; this entry doesn't try
to explain that one.

## What this means for the cycle-accuracy mission

[cycle-accuracy-mission.md](cycle-accuracy-mission.md) item #6 was
written assuming the "Peano-trace-window" framing was real and
upstream-known. Both halves of that turned out to be wrong, but the
entry stays in tree as a pointer to this doc — future-us will trip
over the original misframing in old findings/notes and need the
correction trail.

## Cross-references

- [cycle-accuracy-mission.md](cycle-accuracy-mission.md) item #6.
- Upstream: mlir-aie #2001, mlir-aie PR #2058 (already in our checkout
  and already in our injector's lowering output — verified
  2026-05-05).
- Task #306 (closed; original framing was wrong, real conclusion above).
- Task #311 (trace_unit reset across batches — closed; the actual fix
  for the contamination that drove the empty-batch pattern).
- The mode-2 upstream issue draft
  (`docs/superpowers/findings/2026-05-05-mlir-aie-mode2-issue-draft.md`)
  references this disposition in its "Existing issues we cross-checked"
  section.

## Public discussion

The #2001 / PR #2058 disposition above was flagged for completeness in
[mlir-aie #3047](https://github.com/Xilinx/mlir-aie/issues/3047)
(posted 2026-05-05) under "Existing issues I cross-checked". Update
this section if the upstream conversation surfaces anything that
revises the conclusion (e.g. a missed step in the post-#2058 pattern,
or a different disposition for the single-run empty-trace symptom we
flagged separately).
