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

### Root cause (confirmed, mechanism corrected per review)

Stream backpressure does not propagate from a stalled consumer back to the
producer port, because the inter-tile wire model (`inter_tile_pipeline` in
`src/device/array/routing.rs`) **over-absorbs words**. `propagate_inter_tile`
*does* gate on the destination slave via `can_accept()` at transfer-collection
time, but `can_accept()` (`ports.rs`) only checks the slave FIFO's **current
occupancy** -- it does **not** count words already in flight in the pipeline
(`cycles_remaining > 0`). So during the `ROUTE_PER_HOP` (4) cycle latency window,
up to ~4 extra words are pushed targeting a slave that *looks* like it has room
but will be filled when the in-flight words land; and the pipeline itself is a
plain `Vec<InFlightWord>` with no depth bound. The net effect: the inter-tile
crossing buffers more than the AM020-documented 8-deep, so a lock-stalled
consumer's backpressure does not reach the producer for several extra words; the
memtile master -> slave -> egress drains and the producer runs free.

The intra-tile `switch_pipeline` is, by contrast, bounded (a `latency +
master_fifo_capacity` budget check) -- and a review fact-check confirmed its
latencies (3/4-cycle) and per-port FIFO depths (slave 4, local master 2,
external master 4) **already match AM020 exactly**. The one nuance (section 5
item 2) is that the route budget for crossings into a *local* master computes
`latency(3) + master_fifo(2) = 5` and does not itself count the slave FIFO's 4
slots; whether the separately-checked slave FIFO makes the total effective
buffering the documented 6-deep is a verify-not-assume item.

A potential **co-cause** (not a mere secondary contributor) is the
consume-before-produce phase ordering: the S2MM ingress is drained in Phase 3
before being filled in Phase 4. That hands out a free ingress slot each cycle --
but only when the consumer is actively draining; during the consumer's
lock-stall the drain is gated by the buffer lock, so no free slot is produced.
Whether this ordering defeats the bounded-wire backpressure is therefore
uncertain and must be settled empirically (section 7), not assumed away.

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
   Phoenix-specific aiesim-oracle artifact). A review fact-check found its
   stream-switch section has **no per-crossing latency or per-port FIFO-depth
   fields** -- so it does **not** fill the per-crossing table; AM020 is the sole
   source for those. Its actual value here is narrower: it carries
   `fifo_depth=16` (cross-confirms AM020's 16-deep switch FIFO), `delay=2` and
   `register_slices=2` (corroborate the 2-cycle / register-slice port latencies),
   and -- importantly -- these scalars are **uniform across compute, memtile, and
   shim**, which resolves the "do tile types differ?" question: per the data they
   do not, so the same archspec constants apply to all tile types.
3. **Hardware** -- the validation oracle. Resolves any AM020-vs-NPU1.json
   conflict and is the acceptance gate. Where AM020 leaves something unspecified
   that NPU1.json also lacks (e.g. packet-switch arbitration overhead, which
   AM020 explicitly excludes), a targeted HW capture settles it.

**Trust caveat:** structural params (FIFO depths, port maps, latencies as
documented) are adopted from AM020/NPU1.json directly. Anything timing-ish that
the docs don't pin exactly is confirmed against HW before trusting (the
device-model-audit trust model).

## 4. The model

The fabric is a **pipelined, registered AXI4-Stream ready/valid** network. Each
crossing is:

```
slave-side FIFO  ->  registered latency pipeline  ->  master-side FIFO
   (4-deep)          (3 or 4 cycles)                  (2 local / 4 external)
```

- `valid` flows forward (a stage offers a word when non-empty).
- `ready` flows backward through registered stages (the register slices) -- the
  FIFOs are the skid that sustains throughput while backpressure walks back.
- A transfer happens on an edge iff `valid && ready`.
- **No stage may be unbounded.** A producer stalls exactly when the bounded
  chain downstream of it is full.

**Scope clarification (per review): this is an audit/tighten, not a rewrite.**
The existing multi-pass routing (`route_dma_to_tile_switches -> step_tile_switches
-> propagate_inter_tile -> step_tile_switches -> route_tile_switches_to_dma`)
**already implements** this pipelined ready/valid model for intra-tile crossings:
a fact-check confirmed the per-port FIFO depths and the per-crossing latencies
match AM020 exactly, and the intra-tile `switch_pipeline` is bounded. So we are
**not** replacing the routing engine with a new settle-commit pass. The work is
to bring the model to AM020-exactness where it falls short -- dominantly,
bounding the inter-tile wire and tightening its capacity check (section 5) -- and
to verify the parts already believed correct. Re-implementing the spec-faithful
intra-tile path is explicitly out of scope.

## 5. The audit (the work)

A documentation-driven audit of **every** crossing EMU implements -- intra-tile
and inter-tile -- against the AM020 table (section 2), filling gaps from
NPU1.json. The inter-tile wire is the dominant correction, but the intra-tile
crossings are verified for exactness too, not assumed correct.

1. **`inter_tile_pipeline` -> bounded delay line with backpressure** (the main
   fix). The bug is precise: `can_accept()` checks only the destination slave
   FIFO's current occupancy and ignores words already in flight in the pipeline,
   so up to `ROUTE_PER_HOP` (4) extra words over-absorb; and the pipeline `Vec`
   has no bound. The fix: model the inter-tile crossing as a bounded structure
   whose admission test counts **in-flight pipeline words + destination slave
   occupancy** against the documented depth, carries the documented wire latency,
   and deasserts `ready` to the upstream master when full. This is the single
   change expected to make the memtile MM2S cadence track HW.
2. **Resolve the intra-tile local-master budget (5 vs 6).** Fact-check found the
   route budget for crossings into a *local* master is `latency(3) +
   master_fifo(2) = 5`; AM020's documented crossing depth is 6 (slave 4 + master
   2). The slave FIFO's 4 slots are checked separately (the slave's own
   `can_accept`), so the *total* effective buffering may already be the
   documented 6. Determine whether the effective slave+route buffering equals the
   AM020 6-deep; correct only if it genuinely under-buffers. (Crossings into an
   *external* master compute `4 + 4 = 8` and already match AM020.)
3. **Per-port FIFO depths** -- fact-check VERIFIED EMU's constants
   (`STREAM_LOCAL_SLAVE_FIFO_DEPTH=4`, `STREAM_LOCAL_MASTER_FIFO_DEPTH=2`,
   `STREAM_EXTERNAL_MASTER_FIFO_DEPTH=4`) match AM020. No change expected;
   keep as a regression check. Minor robustness: the slave-FIFO constructor does
   not branch on `is_external()` (correct only because both are 4-deep) -- making
   the branch explicit guards against a future divergence.
4. **Per-crossing latencies** -- fact-check VERIFIED `switch_pipeline` latencies
   (3 local->local / external->local, 4 ->external) and `ROUTE_PER_HOP=4` match
   AM020. No change expected; keep as a regression check.
5. **Per-tile-type differences** -- RESOLVED by NPU1.json: `fifo_depth`,
   `delay`, `register_slices` are uniform across compute/memtile/shim, so no
   tile-type-specific crossing variation is needed. Document the check; no code
   change unless a future capture refutes the uniformity.
6. **Per-BD accept-block compatibility.** Confirm the #140 per-BD TREADY deassert
   (`bd_switch_accept_block`) composes correctly with the bounded wire -- a
   stalled BD must not deadlock the upstream chain.

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
2. **Phase-ordering decision gate (do this before re-baselining the sweep).**
   After the inter-tile wire fix lands, capture `add_one_using_dma` memtile MM2S
   in isolation and compare to HW `[8,8,14,2,14,2,6,8,1]`. If it tracks HW, the
   consume-before-produce ordering did **not** defeat backpressure (the
   lock-gated drain hypothesis holds) and ordering stays out of scope. If the
   producer still races, ordering is a confirmed co-cause and becomes mandatory
   in-scope work for this pass -- not a follow-on. This gate resolves the
   section 1 / section 9 uncertainty empirically rather than by assumption.
3. **Bridge trace sweep, re-baselined against HW.** EMU-vs-HW across the kernels
   that have HW traces (`trace-port-spans.py` + the matrix/regression diff).
   `add_one_using_dma` memtile MM2S `[8,8,14,2,14,2,6,8,1]` is the primary
   acceptance target. HW captures for the validation set are cheap; capture as
   needed. (Pre-flight: confirm the matrix-diff tooling and the HW-trace kernel
   set are ready before relying on per-delta triage at scale.)

**Decision gate result (2026-06-27).** After the inter-tile wire fix (Task 1,
commit 99539c81), captured `add_one_using_dma` memtile MM2S (`PORT_RUNNING_4`)
HW vs EMU with the FFI `.so` rebuilt to include the fix:

- HW:  `durs=[8, 8, 14, 2, 14, 2, 6, 8, 1]` (throttled from the first BD)
- EMU: `durs=[16, 16, 16, 5, 4, 4, 2]` (was `[16, 16, 16, 8, 7]` pre-fix)

The bounded wire **did** propagate backpressure into the tail -- the trailing
runs fragmented from `[8, 7]` into `[5, 4, 4, 2]`, i.e. real stalls now appear.
But the **head is unchanged**: the producer still races through the first three
full 16-word BDs (`[16, 16, 16, ...]`) while HW throttles immediately
(`[8, 8, 14, ...]`). Per the gate, the producer still races, so the
consume-before-produce phase ordering is a **confirmed co-cause** and Task 3 is
**in-scope** for this pass. The S2MM ingress presents a free slot in the same
cycle it is filled (Phase 3 drains before Phase 4 fills), letting the producer
emit a full BD before backpressure registers -- the head-of-stream race.

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
- **Phase ordering (potential co-cause, not a deferred footnote).** The review
  flagged consume-before-produce as possibly *defeating* the bounded-wire
  backpressure (free ingress slot per cycle), not merely a secondary
  contributor. The counter-argument is that the drain is lock-gated, so no free
  slot is produced during the consumer's lock-stall. This is resolved by the
  section 7 decision gate: measure after the wire fix. If ordering must change,
  note it underpins the recv-side fidelity that already matches HW and the S2MM
  ingress depth (16) -- so re-validate both via the trace sweep after any
  ordering change (the ingress depth may need re-calibration).
- **Multicast / packet-switched paths.** Reviewed: intra-tile multicast already
  has all-or-nothing semantics; inter-tile crossings are point-to-point circuit
  routes only, so the wire fix does not need new multicast handling. No action;
  noted to forestall re-investigation.
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
- Every stream-switch crossing matches the AM020 table (latency + FIFO depth);
  the inter-tile wire is bounded to its documented depth (counting in-flight
  words), and the intra-tile local-master buffering is confirmed to equal the
  documented 6-deep. (NPU1.json confirms the 16-deep switch FIFO and tile-type
  uniformity; it does not supply per-crossing values -- AM020 is the source.)
- The bridge trace sweep is re-baselined against HW with every delta triaged
  (win vs regression); no data/bridge-test regressions.
- `cargo test --lib` green.
