# Connectivity: Cross-Column Edge Grounding — Design Spec

**Date:** 2026-06-26
**Status:** Approved (design); pending spec review before plan
**Issue:** #140 (trace inference engine — connectivity sub-project)
**Predecessor:** SP1 Multi-Column Trace Capture (merged local `master` @ 8f9c80bc)

## Problem

The trace-inference timeline engine assembles per-tile event tracks into one
integrated, multi-column timeline. SP1 made capture genuinely multi-column, but
the assembled timeline came out **disconnected-but-honest**: every cross-column
coupling — and most intra-column ones — surfaced as a `connectivity_defect:<a>~<b>`
flag instead of a grounded cross-track edge. The headline question for this
sub-project: make the engine *understand how columns and cells talk to each
other*, rather than flag couplings it cannot explain.

Investigation (on the real two_col NPU1 capture under
`build/experiments/two_col_capture/`) established that the engine **already
comprehends** the logical cross-column conversation — the ledger, derived by
traversing the emulator route graph, contains entries like
`1|1|3|PORT_RUNNING_6 -> 2|4|1|DMA_MM2S_0_START_TASK` (memtile col-1 feeds a
compute tile in col-2) and `2|4|1|DMA_S2MM_0_START_TASK -> 1|0|2|DMA_S2MM_0_*`
(compute col-2 feeds the shim). The comprehension exists. Two independent
problems block it from showing up as grounded edges:

### P1 — the connectivity check uses the wrong abstraction (offline bug)

The connectivity **oracle** (`coupling_oracle` in `tools/inference/timeline.py`)
enumerates **physical adjacent-tile hops**: it walks `route_graph.edges` and
emits a tile pair for every edge whose endpoints differ. For two_col those are
`inter_tile` stream-relay hops between physically adjacent tiles
(`1|2~2|2`, `1|4~2|4`, `1|0~2|0`, and all the intra-column neighbours).

The **weave** (`weave` / `ground_edge`) grounds **logical endpoint
conversations**: producer→consumer dataflow such as `memtile→compute`
(`1|1~2|4`). These are different maps of "connectivity" at different abstraction
levels. `connectivity_defects` compares them directly, so *even if every logical
edge grounded perfectly*, the physical-hop oracle pairs would still be flagged —
the oracle is checking transit wiring against endpoint conversations. The
physical hops are *transit* (intermediate stream-switch ports nobody watches);
the logical edges are *endpoints* (DMA events the trace records). The engine is
literally grading itself against an incompatible answer key.

**Definition — the oracle.** The connectivity oracle is *not* the thing that
couples tracks (that is `weave`/`ground_edge`). It is an independent answer key:
"which tiles should be exchanging data," derived from the emulator route graph
(the config), never from the trace. After `weave` couples what it can, the
oracle audits whether every conversation the config implies was accounted for.
**Weave couples; the oracle audits.** (This concerns the *spatial* tile-to-tile
coupling only — it is unrelated to the *temporal* multi-run clustering/frames
machinery.)

### P2 — compute DMA-module trace never reaches the buffer (HW-gated)

Every cross-column conversation in two_col runs through a compute tile's DMA
(memmod, `pkt_type 1`) events — the col-2 side of the boundary is always a
compute tile, and the dataflow uses its DMA. Those events are **configured and
patched** (`patch.json` carries `{row:2, tile_type:"memmod", events:[21,19,11,...]}`;
`probe_slot_capacity` keeps the tiles, so the binary holds the trace-event
register writes) — yet **no `pkt_type 1` event ever appears in the decoded
trace**. Confirmed absent in the SP1 capture, in the prior `two_col_shakedown`
`witness_before`/`witness_after`, and even in its **128-run** `full` capture. The
cause (emission / trace-packet routing / decode) is not yet isolated and is only
diagnosable on hardware.

Because the col-2 endpoints never fire, `weave` skips all 15 cross-column
candidate pairs at its `child not in fired` guard, and 0 cross-column edges
ground — independent of P1.

## Goal

The inference engine models inter-tile / cross-column communication as **logical
dataflow conversations**, grounds them where the trace observes both ends, and
reports the rest with honest, typed status — plus a diagnosis of the compute-DMA
trace gap that currently blocks the hardware proof of a grounded cross-column
edge.

## Scope

**In scope:**
- **P1** (offline): replace the physical-hop connectivity oracle with a logical
  endpoint-to-endpoint oracle; replace the single `connectivity_defect` verdict
  with a three-way honest classification. Fully unit-testable on the existing
  two_col capture, no HW.
- **P2 spike** (HW-gated, runs first): a timeboxed diagnosis isolating why
  compute memmod (`pkt_type 1`) trace never reaches the buffer, ending in a
  findings doc + a decision gate with Maya on whether/how to fix.

**Out of scope (decided after the P2 spike):**
- The actual P2 *fix* and the on-hardware grounded-cross-column-edge
  demonstration. Whether this lands in this sub-project or a follow-on is the
  P2 decision gate.

## Global Constraints

Binding requirements every task inherits (copied verbatim from standing
project rules):

- **No statistical inference. Q=0.** Connectivity is *derived* from the route
  graph and *grounded* from observed trace values; never inferred from timing
  correlation, and never tuned to make a test pass.
- **Derive from the toolchain.** The emulator route graph (itself derived from
  aie-rt / AM025 / the CDO) is the authoritative source for "which tiles talk."
  No hardcoded topology.
- **HW is the cheap fast ground-truth oracle; EMU is the thing under test.** Use
  HW to settle questions, not EMU.
- **HW discipline:** never two HW suites concurrently; no `xrt-smi` during HW
  runs; `dmesg` never via pkexec; privileged ops via `pkexec` (combined into one
  call), never `sudo`; for HW from a poisoned shell use
  `env -u XDNA_EMU XDNA_EMU_RUNTIME=release`; never self-reboot (hand to Maya
  via `!`); reboot-first when the kernel is wedged.
- **Build/test discipline:** never pipe long builds/tests through tail/head/grep
  (redirect to file or run in background; tee OK); run `cargo build`/`cargo test`
  bare; `cargo test --lib` after any Rust change; rebuild the FFI `.so` before
  HW/bridge tests.
- **Never persistent work in `/tmp`** (machine reboots wipe it); experiment
  output under `build/experiments/`.
- **Commits:** no emoji; end messages with the standard Claude Code trailer.
  Internal project — skip commit-message pre-approval; show EXTERNAL posts
  before posting. Push to origin only on Maya's explicit say-so.

## P1 Architecture (the comprehension fix)

### Logical connectivity oracle

Replace the physical-hop enumeration with an **endpoint-to-endpoint**
enumeration derived from the route graph:

- Identify dataflow **endpoints** — ports where data enters/leaves the stream
  fabric: DMA/stream sources (producers) and DMA sinks (consumers).
- For each producer endpoint, traverse **transit** edges (`circuit`, `packet`,
  `inter_tile`) forward to every reachable consumer endpoint, and emit one
  logical coupling `(producer_tile, consumer_tile)` per producer→consumer path.
  This **collapses transit hops** — the multi-hop physical path becomes a single
  logical conversation.
- `dma_buffer_relay` and `lock_pair` edges are already endpoint-level
  (memory/sync handoffs) and pass through directly as logical couplings.

The result is the set of conversations the config implies — the same level at
which `weave` grounds — so oracle and weave are finally comparable.

### Three-way honest classification

Replace the single `connectivity_defect:<a>~<b>` verdict. For each logical
coupling, classify by observability then groundedness:

- **grounded** — both endpoint events observed (fired) **and** `weave` produced a
  `CrossTrackEdge` connecting the tiles → the conversation is measured.
- **observed-but-ungrounded** — both endpoints observed but no grounded edge →
  the genuine defect; **keeps** the `connectivity_defect` flag.
- **unobserved** — an endpoint event isn't traced / didn't fire → honest gap,
  **not** a defect; a new typed annotation (e.g. `connectivity_unobserved:<a>~<b>`).

On the existing two_col capture this is provably correct offline: the
cross-column couplings classify **unobserved** (col-2 DMA never fired); the
col-1 conversations that ground classify **grounded**; the ~11 bogus physical-hop
"defects" disappear.

### Where it lives

- New focused module `tools/inference/connectivity.py`: the route-graph
  endpoint traversal + the three-way classification.
- `tools/inference/timeline.py` stays glue: `assemble_timeline` calls the new
  module and folds the classification into the timeline flags/output. The old
  `coupling_oracle` / `connectivity_defects` physical-hop logic is removed
  (superseded), not left dead.

### Oracle independence (resolved design point)

`coupling_oracle` today deliberately avoids `generate_ledger` to stay an
independent check. The logical traversal shares the port↔event mapping with the
ledger, so it is not fully code-independent. **Decision (approved):** accept the
shared mapping rather than duplicate it. The independence that matters — and is
preserved — is *structure (route-graph reachability) vs grounding result*: the
oracle enumerates expected conversations from the route graph and compares them
against what `weave` actually grounded, which is a different thing than the
ledger's candidate-pair production. The shared low-level port↔event map does not
compromise that audit.

## P2 Architecture (diagnosis spike)

**Question:** why do compute memmod (`pkt_type 1`) trace events never reach the
decoded trace, despite being configured and patched?

**Decisive first cut** — inspect the **raw `trace.bin`** (pre-decode) for any
`pkt_type 1` packets, against a working reference (the memtile's `pkt_type 3`,
which *does* appear):

- raw has pkt-1 but decoded `trace.events.json` doesn't → **H3 decode-layer**
  bug (`parse_trace` / our prep drops them). Cheapest fix.
- raw has no pkt-1 → emission/routing:
  - **H1 routing** — the compute *mem-module* trace stream has no packet route to
    the shim collector. Check traced MLIR / CDO trace-routing for compute memmod
    vs memtile.
  - **H2 control** — the memmod `Trace_Control` (start) isn't armed for compute
    tiles, so the module never traces even though event slots are set.

**Deliverable:** `build/experiments/two_col_p2_spike/P2-FINDINGS.md` naming the root
cause, the evidence, and a recommended fix with a size estimate — then a
**decision gate** with Maya (fix in this sub-project vs split to a follow-on).
Timeboxed to one focused HW session. Single capture; no concurrent suites;
`env -u XDNA_EMU`.

**Sequencing:** the spike runs **first** — cheap, HW-gated, and its outcome tells
us whether the cross-column grounding proof is one fix away or a deeper effort.
P1 does not depend on the spike's result, so P1 build proceeds regardless.

## Testing Strategy

### P1 — fully offline, on the existing two_col capture

No HW. Use `build/experiments/two_col_capture/` (run dirs + `ledger.json` +
`tools/config_extract/fixtures/two_col.config.json`).

- **Characterization (RED first):** assert today's engine emits the bogus
  physical-hop `connectivity_defect` flags — locks current behavior before change.
- **Logical oracle unit test:** the traversal on the two_col route graph yields
  *logical* couplings (`memtile~compute`, `compute~shim`), not physical hops
  (`1|2~2|2`).
- **Three-way classification:** on the real capture, cross-column couplings
  classify **unobserved** (col-2 DMA never fired); col-1 conversations that
  ground classify **grounded**; zero false "defects."
- **Synthetic both-sides-observed fixture:** a small offline timeline where both
  endpoints *are* observed and ground → classifies **grounded**, proving the
  engine *will* ground cross-column dataflow once the trace cooperates. This is
  the offline stand-in for the P2 hardware proof.

### P2 — the spike

Produces the findings doc. If a fix later lands, the regression is a fresh
two_col capture showing `pkt_type 1` events present and a real cross-column edge
classifying **grounded** on hardware.

## Risks & Open Questions

- **P2 depth unknown.** The compute-DMA trace gap is unsolved across a prior
  128-run shakedown. The spike bounds the risk by isolating root cause before any
  fix is committed; the decision gate keeps the sub-project from absorbing an
  open-ended HW debug.
- **Transit-collapse generality.** two_col's inter-tile communication is entirely
  `inter_tile` stream links; `dma_buffer_relay`/`lock_pair` are intra-tile here.
  The traversal is designed to collapse stream transit (the two_col case) and to
  pass endpoint-level relay/lock couplings through directly. Cross-tile
  relay/lock couplings (e.g. memtile shared locks) are handled as endpoint-level
  if/when a fixture exercises them; not over-built now (YAGNI).
- **Classification of partially-observed couplings.** A coupling where exactly
  one endpoint fired classifies **unobserved** (cannot ground without both ends);
  the spike/plan will confirm this is the desired honesty semantics on the real
  capture.
