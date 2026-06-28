# Cycle-accurate AXI4-Stream backpressure for the stream switch fabric

**Status:** design (approved to write, 2026-06-27). Offshoot of the #140
device-model audit (see `docs/device-model-audit.md`).

## 1. Problem

The memtile MM2S (send) port trace cadence diverges from hardware. In
`add_one_using_dma`, the memtile MM2S send port (PORT_RUNNING_4) decodes to:

- **HW:** `[8,8,14,2,14,2,6,8,1]` over 219 cycles, with producer stalls
  (`gap=25,60,66`) that track the consumer's ~60-cycle lock-stalls.
- **EMU:** `[16,16,16,8,7]` over 68 cycles, no stall > 2 cycles -- the producer
  races ~3x faster than silicon, sending clean full 16-word BDs.

The consumer side is faithful: the compute S2MM recv cadence and its ~60-cycle
per-buffer lock-stall gaps already match HW (`[8,8,8,8]` runs, gaps
`1,1,1,13,56,63,56` vs HW `1,1,1,25,61,66,61`). Data is correct (bridge PASS).
The divergence is purely the **producer not being backpressured by the
lock-stalled consumer**.

### Root cause (confirmed)

Stream backpressure does not propagate from a stalled consumer back to the
producer port, because the inter-tile wire model (`inter_tile_pipeline` in
`src/device/array/routing.rs`) is an **unbounded queue**: `propagate_inter_tile`
pops a word from the source tile's master FIFO into the pipeline without
enforcing a depth bound or re-checking the destination slave's capacity. Words
pile up in the pipeline indefinitely when the downstream (lock-stalled) compute
slave can't accept, which drains the memtile master -> slave -> egress and lets
the producer run free. The intra-tile `switch_pipeline` is, by contrast, bounded
(a `latency + fifo_capacity` budget check), so the leak is specifically the
inter-tile wire. A secondary contributor is the consume-before-produce phase
ordering (the S2MM ingress is drained in Phase 3 before being filled in Phase 4,
handing out a free slot each cycle).

## 2. Proof that this is how the hardware works

The fix must reproduce real hardware behavior, not a plausible model. AM020
(AIE2/AIE-ML architecture manual, `docs/xdna/am020-aie-ml/`), chapter 2,
"AXI4-Stream Interconnect":

> "Each AIE-ML tile has an AXI4-Stream interconnect (alternatively called a
> stream switch) that is a fully programmable, 32-bit, AXI4-Stream crossbar...
> **It handles backpressure**..."

and:

> "In all the streaming cases there are **built-in hand-shake and backpressure
> mechanisms.**"

The fabric is an **AXI4-Stream crossbar**. AXI4-Stream is, by protocol
definition, a `TVALID`/`TREADY` (valid/ready) handshake with backpressure. So
ready/valid + backpressure is the hardware's actual mechanism -- proven from the
documentation, not assumed.

### The mechanism is registered (pipelined), not combinational

AM020 documents **register slices** and per-port latency:

- External ports: 2-cycle latency, 4-deep FIFO
- Local slave ports: 2-cycle latency, 4-deep FIFO
- Local master ports: **one register slice**, 1-cycle latency, 2-deep FIFO

So backpressure (`TREADY`) propagates through *registered* stages -- roughly one
stage per cycle -- with the FIFOs providing skid to sustain throughput. The
model is a **pipelined** ready/valid, NOT a single-cycle combinational
ready-propagation across the whole chain.

### Independent confirmation of the measured mechanism

AM020 per-crossing latency + buffering (the spec table for the audit):

| Crossing | Latency | FIFO depth |
|---|---|---|
| local slave -> local master | 3 | 6 |
| local slave -> external master | 4 | 8 |
| external slave -> local master | 3 | 6 |
| external -> external | 4 | 8 |
| explicit switch FIFO port | -- | 16 (32b + parity + TLAST) |

The HW measurement showed the producer stalls at word **14 = 8 (consumer
buffer) + 6 (chain slack)**. The compute receives from the wire via the
*external-slave -> local-master* crossing = **6-deep FIFO**. The documentation's
6 matches the measured 6 exactly. The mechanism is confirmed from two
independent directions (HW trace and AM020).

## 3. Derivation sources & trust model

In priority order (per CLAUDE.md "DERIVE FROM THE TOOLCHAIN"):

1. **AM020 ch2/ch3/ch5** -- the architecture authority for the stream switch
   crossing latencies and FIFO depths. Generic Versal AIE-ML; ch2 is
   compute-tile-centric.
2. **NPU1.json** (`build/experiments/aiesim-device-decrypt/NPU1.json`, our
   Phoenix-specific aiesim-oracle artifact) -- fills what AM020 leaves
   unspecified: per-tile-type crossing params (memtile/shim stream switches that
   may differ from the ch2 compute table), the explicit `fifo_depth=16`
   (cross-confirms AM020 line 74), packet-switch arbitration overhead (AM020
   explicitly excludes it), and any Phoenix deviations.
3. **Hardware** -- the validation oracle. Resolves any AM020-vs-NPU1.json
   conflict and is the acceptance gate.

**Trust caveat:** structural params (FIFO depths, port maps, latencies as
documented) are adopted from AM020/NPU1.json directly. Anything timing-ish that
the docs don't pin exactly is confirmed against HW before trusting (the
device-model-audit trust model).

## 4. The model

A **pipelined, registered AXI4-Stream ready/valid handshake** for every crossing
in the fabric. Each crossing is:

```
slave-side FIFO  ->  registered latency pipeline  ->  master-side FIFO
   (4-deep)          (3 or 4 stages, 1/cycle)         (2 local / 4 external)
```

- `valid` flows forward (a stage offers a word when non-empty).
- `ready` flows backward **one registered stage per cycle** (a stage accepts
  when it has room or is simultaneously forwarding -- skid semantics). The FIFOs
  are the skid that sustains full throughput while backpressure walks back.
- A transfer commits on an edge iff `valid && ready`; all commits in a cycle
  apply together (the clock edge).
- **No stage is unbounded.** A producer stalls exactly when the bounded chain
  downstream of it is full -- which is what the current model fails to do.

This replaces the multi-pass greedy routing (`route_dma_to_tile_switches ->
step_tile_switches -> propagate_inter_tile -> step_tile_switches ->
route_tile_switches_to_dma`) with a settle-then-commit pass over bounded,
latency-accurate crossings.

## 5. The audit (the work)

A documentation-driven audit of **every** crossing EMU implements -- intra-tile
and inter-tile -- against the AM020 table (section 2), filling gaps from
NPU1.json. The inter-tile wire is the dominant correction, but the intra-tile
crossings are verified for exactness too, not assumed correct.

1. **`inter_tile_pipeline` -> bounded delay line with backpressure** (the main
   fix). Today unbounded; must become the documented external-port buffering
   (4-deep each side) as a bounded shift register that (a) carries the
   documented wire latency and (b) deasserts `ready` to the upstream master when
   full, so a stalled consumer's backpressure reaches the producer. This is the
   single change that makes the memtile MM2S cadence track HW.
2. **Verify intra-tile `switch_pipeline` bounds are exactly the table's 6/8-deep
   and 3/4-cycle**, not approximate. If the current budget computes e.g.
   `latency + master_fifo` (= 5) rather than slave 4 + master 2 (= 6), correct
   it.
3. **Per-port FIFO depths** (external 4, local slave 4, local master 2): confirm
   EMU's constants match AM020; derive any that are still hardcoded. Cross-check
   against NPU1.json per tile type.
4. **Per-crossing latencies**: confirm `switch_pipeline` latency and
   `ROUTE_PER_HOP` produce exactly 3 (local/external -> local master) and 4
   (-> external master) per crossing type.
5. **Per-tile-type differences**: AM020 ch2 is the compute tile; check memtile
   (ch5) and shim (ch3) crossings against NPU1.json and correct if they differ.

## 6. Trace semantics

`PORT_RUNNING`/`PORT_STALLED` reporting is already built on `cycle_beat`
(`src/device/stream_switch/ports.rs`): a port asserts RUNNING iff a `ready &&
valid` transfer crosses it in the commit step; `valid && !ready` is STALLED.
This machinery is preserved; the audit only ensures it is driven by the
corrected bounded crossings. The run-length/gap decoders
(`tools/trace-port-spans.py`, and the gap-aware variant under the session
scratch) are the inspection tools.

## 7. Validation & re-baseline

Layered, because the change is fabric-global and trace churn is expected:

1. **Synthetic-crossing unit test (TDD anchor).** A producer emitting 16-word
   BDs + a lock-stalled consumer (8-word buffers, ~N-cycle lock stall) + the
   AM020 external-slave -> local-master crossing (6-deep) must reproduce the
   producer stalling at word 14 (8 + 6). This is the standalone chain-sim
   (validated in scratch) promoted to a Rust unit test, parameterized by the
   AM020 table. RED on the current model (clean `[16,...]`), GREEN after the fix.
2. **Bridge trace sweep, re-baselined against HW.** EMU-vs-HW across the kernels
   that have HW traces (`trace-port-spans.py` + the matrix/regression diff).
   `add_one_using_dma` memtile MM2S `[8,8,14,2,14,2,6,8,1]` is the primary
   acceptance target. HW captures for the validation set are cheap; capture as
   needed.

**Breakage-spotting (explicit goal).** A change this large will move traces.
The matrix diff is the breakage detector. Every moved trace is triaged into:
(a) **win** -- moved toward HW (the re-baseline target), or (b) **regression**
-- moved away from HW or broke a passing data/bridge test. Wins are
re-baselined; regressions are fixed before proceeding. Silent acceptance of
churn is not allowed -- each delta is accounted for.

## 8. Testing & rollout

- **TDD:** synthetic-crossing test first (RED), implement bounded crossings
  (GREEN), refactor.
- **`cargo test --lib` after every change.** The fabric routing is exercised by
  many existing tests; regressions there are the fast early-warning. Baseline is
  the current ~3546 passing.
- **Incremental rollout:** (1) bound the inter-tile wire (dominant fix), run
  `--lib` + a spot `add_one_using_dma` trace capture; (2) tighten intra-tile
  exactness; (3) per-tile-type corrections. Each step attributable, so a
  regression points at one change.
- **Integration gate:** the bridge trace sweep (slow; run once per batch, not to
  "check progress" -- per CLAUDE.md). Never run two HW suites concurrently.

## 9. Risks & open questions

- **Blast radius.** Global backpressure change; many unit tests and all traces
  are affected. Mitigated by incremental rollout + per-delta triage.
- **Phase ordering.** The consume-before-produce ordering (deliberate, enables
  the S2MM ingress-depth fix) is a secondary contributor. The bounded-wire fix
  may suffice; if residual under-backpressure remains after the audit, revisit
  ordering as a follow-on (do not touch it pre-emptively -- it underpins the
  recv-side fidelity that already matches HW).
- **Exact warmup micro-timing.** The acceptance target is the HW cadence; if a
  small residual remains (e.g. the `25` transition gap vs a slightly different
  EMU value), decide case-by-case whether it is within fidelity tolerance or
  needs a further documented parameter from NPU1.json/HW.
- **memtile/shim crossing data.** If AM020 + NPU1.json under-specify a per-tile
  crossing, a targeted HW capture settles it.

## 10. Success criteria

- The synthetic-crossing unit test reproduces the documented `8 + 6 = 14`
  producer stall.
- `add_one_using_dma` memtile MM2S send cadence tracks HW
  (`[8,8,14,2,14,2,6,8,1]`) -- the consumer-driven backpressure now reaches the
  producer.
- Every stream-switch crossing matches the AM020 table (latency + FIFO depth),
  with NPU1.json filling documented gaps.
- The bridge trace sweep is re-baselined against HW with every delta triaged
  (win vs regression); no data/bridge-test regressions.
- `cargo test --lib` green.
