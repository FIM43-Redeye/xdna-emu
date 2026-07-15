# MM2S egress FIFO depth: not directly trace-observable; 48-word egress *pipeline* pinned; constant kept at 12

Date: 2026-07-15
Task: Phoenix-retirement arc, Experiment B (MM2S egress FIFO depth)
Related: [`2026-07-14-dma-memory-pressure-event-semantics.md`](2026-07-14-dma-memory-pressure-event-semantics.md)
(the S2MM ingress mirror this experiment set out to reproduce),
[`2026-07-15-memtile-bank-access-width.md`](2026-07-15-memtile-bank-access-width.md)
(Experiment A: the 128-bit = 4-word beat that reappears here)

## Goal and outcome

**Goal:** HW-pin `DMA_MM2S_EGRESS_FIFO_DEPTH` (`mm2s_egress_fifo_depth`, currently
12, documented as 32-bit words), the dual of the HW-pinned `s2mm_ingress_fifo_depth`.

**Outcome (characterize-only, spec-permitted):** the DMA egress staging FIFO is
**not directly isolable by MM2S trace events on this silicon**, because
`FINISHED_BD` gates on downstream-*accept*, not on fetch. Every trace-event probe
measures the fetch-to-some-downstream-point occupancy, never the DMA FIFO alone.
What the experiment *did* pin, cleanly and reproducibly:

- The **end-to-end MM2S egress pipeline** holds a fixed **48 words = 12 x 128-bit
  beats** (the same 4-word beat Experiment A found).
- `FINISHED_BD` posts at **downstream-accept (drain)**, not at fetch.
- The **emulator models the egress pipeline as ~0-deep** (it posts `FINISHED_BD`
  effectively at fetch) -- a latent simplification with no current observable
  divergence.

**The constant is kept at 12.** It is not refuted: the egress *DMA FIFO* (a subset
of the 48-word pipeline) is ~12-16 words by the ingress mirror (below), the
observable it was justified by (`MM2S_MEMORY_STARVATION = 0`) matches HW, and
AM020 gives no standalone egress-FIFO depth to override the device-model value.

## The escalation ladder (what fired, what did not)

All on real Phoenix NPU1, compute-tile MM2S, source tile decoded as `mem(2,1)`
(the col-0 -> col-1 trace virtualization, HW-confirmed; see the measure's
`SOURCE_TILE_COL`).

1. **`fill_stall` baseline (3 reps):** 16 `STALLED_LOCK` windows, **zero
   `MEMORY_STARVATION`**. The FIFO provably reaches full (`STREAM_BACKPRESSURE`
   fires, n=26) -- the fetch outruns the stream -- but the fetch only lock-stalls
   at *rep boundaries*, where its BD has already drained the FIFO empty. Fetch and
   drain are lock-stepped per rep, so the spec's `STALLED_LOCK -> STARVATION`
   recipe has no empty-with-demand moment to measure. Predicted crux, confirmed.

2. **`burst` credit-small-BD escalation (M in 8..64):** the mechanism that lets a
   completed BD sit in the FIFO (a small BD reports `FINISHED_BD` without waiting
   for the whole payload to drain). Still **zero `MEMORY_STARVATION`** -- in every
   design the fetch leads or matches the 1-word/cycle stream, so the FIFO never
   empties with demand. But it surfaced a clean invariant: **`FINISHED_BD` = M -
   12** across the whole sweep (12 BDs perpetually "in flight" at teardown).

3. **`bw` BD-width discriminator (fixed M=64, BD in {1,2,4,8,16} words):** is the
   backlog a fixed descriptor count or a fixed word count?

   | BD words | backlog (BDs) | backlog x words |
   |----------|---------------|-----------------|
   | 1  | 30 | 30 |
   | 2  | 18 | 36 |
   | 4  | 12 | **48** |
   | 8  | 6  | **48** |
   | 16 | 3  | **48** |

   Backlog-in-BDs is **not** constant, so not descriptors; backlog x words
   **plateaus at 48** for BD >= 4 (W=1,2 undershoot -- tiny BDs cannot fetch fast
   enough to fill the buffer to its ceiling). **48 words = 12 x 128-bit beats.**

4. **EMU cross-check (same xclbins, `XDNA_EMU=1`):** EMU backlog is **~0** across
   all BD widths (63-64 `FINISHED_BD` = M). The emulator posts `FINISHED_BD` at
   fetch and holds nothing in flight. HW's 48-word backlog is therefore the
   fetch-to-accept *lag* -- i.e. the end-to-end pipeline -- not a structure the
   emulator under-models by a factor.

5. **`hfill` held-consumer probe (source(0,2) MM2S -> sink(0,3) S2MM held then
   released):** built to isolate the FIFO by holding the drain. The
   compute-to-compute (row2 -> row3) route **compiles cleanly** -- the earlier
   two-tile failure was purely the acquire-only lock bug (AIERT
   `configureLocksInBdBlock`), never routing. Under the held consumer the source
   posts **zero `FINISHED_BD`**, confirming completion gates on downstream-accept.
   Runs complete normally (ERT state 4). But the probe does not isolate the FIFO
   either: the fill during the hold spans source-FIFO + stream-switch + sink
   ingress up to the hold point -- still a pipeline, still confounded.

## Why the DMA egress FIFO is not trace-observable here

`FINISHED_BD` gates on downstream-accept and `STREAM_BACKPRESSURE` gates on the
downstream refusing -- neither is FIFO-local for MM2S egress. So:

- The **drain-side** read (backlog) measures fetch -> accept = the whole pipeline
  (48 words).
- The **fill-side** read (held consumer) measures fetch -> hold-point = a pipeline
  prefix, confounded by the switch and sink buffering between source and hold.

There is no event that fires when the *DMA staging FIFO alone* fills or empties.
The one regime that would isolate it -- the fetch outpaced by the drain so the
staging underruns -- does not occur, because the memory side out-bandwidths the
32-bit stream (Experiment A's stream-bound result: a single stream drains at 1
word/cycle while the memory side fetches a 128-bit granule per bank slot). The
staging therefore always runs ahead; it never starves; its depth leaves no
trace-event signature.

## AM020 corroboration (derive-from-toolchain)

`docs/xdna/am020-aie-ml/chapter-2-aie-ml-tile-architecture.md` documents the
stream-switch FIFOs but gives the **MM2S DMA egress staging no standalone depth**:

- ch.2:74 -- "The switch has one FIFO that is **16-deep** and 34 bit (32 bit + 1
  bit parity + 1 bit TLAST)."
- ch.2:88 / :92 -- "Local slave to external master / External to external:
  4-cycle latency and **8-deep FIFO**."
- ch.2:143 -- the AIE-ML has one 32-bit output stream with **no** 128-bit FIFO on
  it.

So the 48-word egress pipeline is composed of documented, separately-modelled
buffers -- the 12-word DMA staging (`mm2s_egress_fifo_depth`) plus the 16-deep
switch FIFO, the 8-deep local-slave FIFO, and the shim-side ingress along the
source->shim path -- not a single 48-word egress FIFO. AM020 confirms there is no
manual value that would override the device-model `mm2sChannel.buffer_depth = 12`
for the staging itself.

## Why the constant stays at 12

- **Ingress mirror.** `s2mm_ingress_fifo_depth` derives from the *same*
  device-model `buffer_depth = 12` and was HW-pinned to ~16 words (via the
  FIFO-local "backpressure asserts +15" method). A mirror structure with identical
  device-model depth is ~12-16 words, not 48; the egress *DMA FIFO* is the same
  order, and the 48 is the surrounding pipeline.
- **Observable matches.** `MM2S_MEMORY_STARVATION = 0` on HW under bank contention
  (Experiment A / the granule-cap finding); the emulator reproduces 0. The field
  doc already notes any staging depth >= ~5 satisfies this, so 12 is safe.
- **No better source.** AM020 gives no standalone egress-FIFO depth; the device
  model's 12 is the only derive-from-toolchain value, and it is used as-is (not
  fitted).

## HW facts banked

- MM2S egress **pipeline** (source -> shim, compute tile): **48 words = 12 x
  128-bit beats**, dead flat over BD width for BD >= 4, both reps.
- MM2S `FINISHED_BD` posts at **downstream-accept**, not fetch (held-consumer ->
  0 FINISHED; bw backlog = a 48-word fetch-to-accept lag; EMU-at-fetch -> 0 lag).
- The emulator models the egress pipeline as ~0-deep (FINISHED at fetch) -- a
  latent simplification, no current observable divergence (starvation and steady
  send-cadence already match HW).

## Reproduce

`tools/experiments/mm2s_egress_depth.py` (variants: `fill_stall`, `burst_*`,
`bw_*`, `hfill_*`) + `tools/experiments/b_egress_capture.py` (driver) +
`tools/experiments/mm2s_egress_depth_measure.py` (measure). HW via
`env -u XDNA_EMU XDNA_EMU_RUNTIME=release`; EMU cross-check via `XDNA_EMU=1
XDNA_EMU_RUNTIME=debug` on the same xclbins.
