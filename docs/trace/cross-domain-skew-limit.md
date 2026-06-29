# Cross-Domain Trace-Timer Skew: What We Cannot Determine

**This documents a permanent epistemic boundary, not a bug or a backlog item.**
It records what about NPU1 (Phoenix / AIE2) trace timing is *undeterminable from
our vantage*, the physical nature of the quantity we cannot see, why it is
unknowable, and precisely what fidelity we forfeit as a result. The point is so
future work respects the wall instead of re-deriving it -- on #140 the
"cross-domain timer sync" effort hit this wall twice (a co-observed-anchor design
and a skew-decomposition design were both written and then falsified) before the
boundary was characterized. Read this before proposing any cross-domain
*causal*-timing inference.

**Read §7-8 before concluding anything is lost.** The boundary is real for a
*trace in isolation* (§1-6), but an independently-verified compute model recovers
most of it: the broadcast skew becomes a *measured residual*, and the only
genuinely irreducible forfeiture is cycle-deterministic timing across the async
main-memory boundary.

**Sibling caveat:** this doc bounds what a trace tells us *across* timer domains.
For what it tells us *within* one domain when the capturing host is loaded --
including why a phantom "+/-1 HW jitter" was once recorded -- see
[`capture-load-sensitivity.md`](capture-load-sensitivity.md).

Issue: #140 (byte-identical emulator/HW trace reports).
Evidence: `build/experiments/gap140/nondeterminism/add_one_using_dma/` (20 HW
runs); `build/experiments/bcast-bridge/FINDINGS.md` (aiesim broadcast
reconstruction); compiled `aie_traced.mlir`.

## 1. The physical setup

The AIE-ML array is one physical clock domain (1 GHz). Trace timestamps, however,
are **not** measured against that clock directly. Each traced module
(compute-core, memory-module, mem-tile, shim) has its own trace timer, and the
modern IRON trace lowering (`AIEInsertTraceFlows.cpp`) compiles
`Timer_Control.Reset_Event = BROADCAST_15` into every traced array tile (the shim
generates `USER_EVENT_1` and broadcasts it). Each timer resets to 0 when the
BROADCAST_15 flood *arrives at that module*.

So for an event `x` recorded in module D, the trace stores:

```
soc(x) = W(x) - origin_D
```

where `W(x)` is the physical wall-clock cycle of the event and `origin_D` is the
wall-clock cycle at which the broadcast reset reached module D.

## 2. The decomposition

For an event `x` in domain A and `y` in domain B, the recorded offset is:

```
soc(x) - soc(y) = [W(x) - W(y)]  +  [origin_B - origin_A]
                =   Δwall(x, y)    +     skew(A, B)
```

- **`Δwall(x, y)`** -- the real physical separation between the two events. This
  is what we *want*: "y happens N cycles after x." It is workload-dependent.
- **`skew(A, B) = origin_B - origin_A`** -- the difference in *when the broadcast
  reset arrived at each module*. It is a fixed propagation constant of the
  broadcast network and the array geometry, independent of the workload.

The trace gives us only the **sum** of these two terms. Never either one alone.

## 3. What the skew physically is

`skew(A, B)` is a propagation property of the BROADCAST_15 flood: it is set by the
generation point (the shim), the flood topology, the per-hop propagation delay,
and -- for two modules of the *same* tile -- the asymmetry between the core and
memory modules' broadcast-input pipelines. Its components:

- **Inter-tile hop delay** -- the flood takes time to travel from the shim across
  rows/columns; a module further along the route resets later.
- **Intra-tile module asymmetry** -- within one tile, the core and memory modules
  receive the broadcast through different input paths with (apparently) different
  pipeline depths, so they reset a small fixed number of cycles apart.

It is a **fixed, deterministic** quantity: across 20 HW runs of
`add_one_using_dma` the cross-domain raw offsets are stable to range 0, and they
are additively consistent (`core-memtile = +2`, `memmod-memtile = +4`, so
`core-memmod = -2`). That consistency is what a fixed underlying skew structure
plus a stable workload `Δwall` looks like.

## 4. What we *can* determine

- **The raw cross-domain offset (the sum), exactly.** For any pair of events that
  fire deterministically together, `soc(x) - soc(y)` agrees across runs to range 0.
  This is a valid, exactly-measurable **reproduction target** (see §7).
- **That the structure is fixed and additively consistent**, as above.

## 5. What we *cannot* determine -- the loss

*Scope: §5-6 are about the trace **in isolation**. §7-8 show how an
independently-verified compute model recovers most of this.*

- **The decomposition.** We cannot split the measured offset into `Δwall`
  (physical separation) and `skew` (broadcast-arrival difference).
- **Therefore: no cross-domain causal-latency fact.** We can never state "event y
  occurs N causal cycles after event x" when x and y are in different timer
  domains. We know only their recorded-timestamp difference, which is contaminated
  by an unknown skew.
- **No isolated broadcast constants.** We cannot measure the per-hop broadcast
  delay or the intra-tile core/mem asymmetry as independent physical figures.

Note the corollary about the measured numbers: the `-2` between
`core(1,2,INSTR_VECTOR)` and `memmod(1,2,DMA_S2MM_0_START_TASK)` is **not** the
intra-tile skew -- it is `Δwall + skew` for that specific event pair. The
workload separation and the broadcast skew are fused in it.

## 6. Why it is undeterminable *from a trace alone* -- three independent walls

1. **Trace underdetermination.** Each domain pair yields one equation (the sum)
   in two unknowns (`Δwall`, `skew`). There is no second independent equation,
   because no traceable event arrives with a *known* `Δwall` (which would solve
   `skew`) or a *known* `skew` (which would solve `Δwall`).

2. **No globally-simultaneous traceable signal.** The natural idea -- co-observe a
   second broadcast event E in both domains and difference it -- fails, because E
   floods through the same network as the reset, so its arrival delay cancels
   against the reset's arrival delay: `soc_A(E) = soc_B(E)`, difference zero, no
   information. To isolate skew you would need an event that arrives at two
   modules at the *same wall-clock cycle* (so its arrival difference is a known
   zero). No such physically-simultaneous, traceable signal exists on the array;
   every event either floods (delay cancels) or originates locally (no second
   observer).

3. **aiesim is circular here.** aiesim cannot serve as an independent skew oracle,
   because the cluster model we run **wires the inter-tile broadcast topology but
   leaves its propagation dormant** -- the EventBroadcast channel pointers are
   connected (all 50 EBs are shared channels, block masks unblocked) and the
   downstream consumer lists are populated, but the array/compute EventBroadcast
   units never fire: their `generate_outputs` is gated on a credit counter that
   only a tile's *own* events advance, and the sub-model seams (the shim->memtile
   input is a dangling placeholder; memtile->compute is a dropped seam) are left
   unbridged, so 0 array broadcast signal events occur until we intervene. It is
   a partial cluster instantiation that leaves the inter-tile seams for the
   harness to bridge (it exports `CoreModule::event_broadcast_write` for exactly
   that). Our
   `aiesim-bridge/src/bcast_bridge.cpp` *reconstructs* the flood by injecting each
   tile's broadcast from its wired south wire, and that reconstruction "ripples
   one row per posedge" **by construction** (we read the previously-committed wire
   and inject on the next posedge). So any cross-tile skew aiesim reports is our
   injection cadence reflected back -- measuring aiesim measures our own model. The
   intra-tile core/mem split is likewise bridge-driven. (See
   `build/experiments/bcast-bridge/FINDINGS.md`: `ARRAY` broadcast signals 0 -> 112
   appear *only* via our vertical-flood injection.)

These are independent: the first is information-theoretic, the second physical,
the third architectural. All three say the same thing.

## 7. What this costs -- and what in-domain verification recovers

Two tiers, with very different conclusions: the inference engine (working from a
single trace) versus the emulator-plus-verification program (which has an
independently-validated forward model to lean on).

### Inference engine (trace alone): cross-domain edges stay gaps

The engine sees only one trace; by §6 it cannot ground a cross-domain edge as a
causal segment. It records the exact raw offset as a **reproduction target** and
stops there. This is the correct terminal treatment and does not change.
Within-domain edges are unaffected (`skew = 0`, so the offset *is* causal latency
-- this is why within-domain grounding is sound).

### Emulator: the soft-identifiability is mostly dissolved, not permanent

The earlier worry was that fitting the emulator's cross-domain `soc` to HW would
be soft -- a wrong compute-timing hiding behind a compensating wrong broadcast
model, both matching the sum. The escape is that **compute timing is verifiable on
its own, skew-free, within each domain:**

- The emulator runs on **one global clock** and predicts a wall-time `W_sim(e)`
  for every event, so `W_sim(x) − W_sim(y)` is its prediction of `Δwall(x, y)` for
  *any* pair -- cross-domain included.
- Within-domain segments (the merged jitter-grounding rule already extracts them)
  are pure `Δwall` -- skew is identically zero there. Reproducing *all* of them
  exactly validates the compute/DMA model **without skew ever entering the
  picture.**
- Then the cross-domain residual isolates the skew directly:
  ```
  skew(A, B) = [HW raw cross-domain offset] − [W_sim(x) − W_sim(y)]
             =  measured  −  (verified) Δwall
  ```

This is **not** "from traces alone" (which §6 forbids) -- it is traces **plus an
independently-verified compute model**, which is exactly the emulator we are
building. It is non-circular: the `Δwall` predictor is validated where skew cannot
reach (within-domain), then used to extract skew, which is added back as a separate
component.

### The residual ambiguity, and how small it is

Within-domain verification pins the *shape* of each domain's event constellation
perfectly, but not the *alignment* between two constellations. That alignment is
`inter-tile coupling latency` (real wire / stream / DMA propagation when data
flows A→B) `+ skew`. So the degeneracy does not vanish -- it **shrinks** from "the
whole compute model vs skew" to "just the inter-tile propagation constants vs
skew," which is bounded and small, and further constrained two ways:

- **Round-trips constrain the couplings for free.** A path that starts and ends in
  one domain (in `add_one`: shim → memtile → compute → memtile → shim, both ends
  seen by the shim) is itself a within-domain segment, so its total latency is
  skew-free ground truth -- tightly bounding the *sum* of the couplings.
- **Stream-switch and DMA transfer latencies are toolchain-specified** (aie-rt /
  AM025), apart from the irreducible main-memory boundary below. Where they are
  derivable, we subtract them and the skew is **fully isolated** -- cross-domain
  `Δwall` becomes recoverable and the soft-identifiability closes. (Verifying the
  exact specification of these latencies is a concrete, answerable toolchain
  question, not a guess.)

### The DMA / DDR boundary is a gap by design -- not a loss

The async main-memory crossing -- shim ↔ NoC (1 GHz ↔ 960 MHz CDC, AM020) and DDR
access -- is non-deterministic, and we never wanted it otherwise. Cycle-accurate
DDR-transfer timing is impossible and is **not a goal**. The requirement is
narrower and fully met: **a DMA must not poison the grounding of the deterministic
activity around it.**

Because grounding is *local* -- each within-domain segment is measured against its
own domain timer (which does **not** reset at a DMA boundary) by exact pairwise
agreement, never by a cumulative offset chained from the start -- a DMA's
non-determinism is **contained to its own gap** and never propagates. The timeline
is:

```
traced segment (exact) -> DMA (gap) -> traced segment (exact) -> DMA (gap) -> ...
```

and every traced segment grounds independently and exactly, on *both* sides of
every gap. Nothing downstream is "irreversibly skewed" by a DMA: the deterministic
compute is in-domain and stays cycle-accurate; only the DMA waits are gaps, which
is exactly what they have always been. (This is the merged within-domain rule's
"gap + exact segment + gap" model -- §7's in-domain verification builds directly on
it.)

### The honest net

We forfeit **nothing we wanted.** Cross-domain causal latency *through a DMA* was
never cycle-deterministic and is correctly a gap. The deterministic,
cycle-accurate content -- all of it within domains -- stays fully grounded across
every DMA boundary. The broadcast skew (a fixed constant, *not* a DMA effect)
becomes a *measured* residual via in-domain compute verification plus
toolchain-specified inter-tile latencies. The only non-deterministic quantity left
is the DMA/DDR transfer time itself, which is a gap by nature and by intent.

## 8. Routes to the skew (and which we use)

1. **In-domain compute verification + toolchain inter-tile latency (primary,
   non-speculative).** Per §7: validate the compute model skew-free within domains,
   subtract toolchain-specified stream/DMA latencies, read the broadcast skew off
   the cross-domain residual. This is the emulator we are building anyway; no new
   instrument required.
2. **Round-trip closure (corroborating).** Same-domain round-trips bound the
   coupling *sums* independently, cross-checking route 1.
3. **Non-broadcast-reset free-running counter (fallback, HW-gated).** Only if some
   inter-tile coupling latency turns out to be an undocumented micro-constant that
   routes 1+2 cannot pin: timestamp the broadcast's arrival at each module using a
   timer *not* reset by that broadcast, differenced to give `skew(A, B)` directly.
   Speculative (assumes such a counter is readable and synchronized), and only ever
   for a proof on the last sliver -- never required for reproduction.

   **3b. Compute-path timer read, two-source trigger (concrete realization of 3,
   HW-gated).** Route 3's two stated assumptions both have answers, which makes it
   worth recording even though it is still not on the reproduction critical path.

   - *"Readable" is no longer an assumption.* The timer is core-readable from
     compute code: `__builtin_aiev2_read_tm`
     (`llvm-aie/llvm/include/llvm/IR/IntrinsicsAIE2.td:469`) lowers to `LDA_TM`, a
     memory-mapped load of the timer registers (`Timer_Low @ 0x340F8`,
     `Timer_High @ 0x340FC`, per the AM025 register DB) into a core register, from
     where it can be stored to data memory and DMA'd out as ordinary data. It is an
     MMIO load, not a one-cycle register read -- it costs cycles and perturbs the
     compute schedule, so it lives in a dedicated characterization kernel, never the
     trace gate. No existing kernel does this (only host-side timer reads, e.g.
     `mlir-aie/test/benchmarks/14_Timer`), but the capability is confirmed.
   - *"Synchronized" comes free from the reset flood -- the key refinement.* Do
     **not** free-run an un-reset timer (whose per-tile start is set by serialized
     config writes and is therefore unsynchronized). Instead keep the ordinary
     BROADCAST_15 reset flood from source `s1` (it synchronizes every tile's timer,
     `tau_X(W) = (W - T0_1) - D(s1, X)`), and trigger the compute-path reads with a
     **second broadcast from a different corner** `s2`. Each tile reads when `s2`'s
     wavefront arrives, at wall `T0_2 + D(s2, X)`, so:
     ```
     r_X = (T0_2 - T0_1) + D(s2, X) - D(s1, X)
         = const + (Δn_h)·d_h + (Δn_v)·d_v        [Δn known from geometry]
     ```
     Differencing `r_X - r_Y` across tiles drops the const and leaves a linear
     system in `(d_h, d_v)` -- rank-2 if the tile/source geometry is chosen well,
     solving `d_h` and `d_v` **directly on silicon**, with no trace decode and no
     dependence on the within-domain-exact emulator. When `s2 = s1` the two delays
     cancel (`r_X = const`, wall 2 of §6) and there is no signal; the whole trick is
     `s2 ≠ s1`.

   This is the same instrument as route 3 but self-synchronizing, and it is an
   **independent silicon cross-check of route 1** (and a possible direct substitute
   if route 1's toolchain-specified stream/DMA latencies turn out insufficient).
   Caveats, honestly: it needs two independently-configured broadcast floods, the
   core must be event-triggered into the `LDA_TM` with roughly-constant added
   latency, and the read perturbs the kernel. It has been reasoned through, not
   tested. It belongs to **SP-5** (silicon characterization), not to the trace gate
   (SP-3) or the reproduction path (routes 1-2).

The async DDR/NoC boundary (§7) is not on this list: it is not a fixed constant to
measure but a non-deterministic latency to bound, handled gap-only.

## 9. The resulting design rule

- **Engine:** record the exact raw cross-domain offset as a reproduction-target
  annotation on the gap. Never emit a cross-domain causal segment.
- **Emulator:** give `broadcast.rs` a forward per-hop flood model. **Verify the
  compute model skew-free in-domain first** (reproduce every within-domain
  segment), *then* measure the broadcast skew as the cross-domain residual
  (`measured − verified Δwall`, with toolchain-specified stream/DMA latencies
  subtracted); fit any remaining micro-constant to silicon ground truth. Acceptance
  gate is `emu raw offset == hw raw offset` on cross-domain pairs. The only
  gap-only quantity is async DDR/NoC egress timing.
- **Do not** reach for aiesim as a cross-domain skew oracle. It is a faithful
  oracle for trace *structure*, deterministic event *order*, and compute-region
  *cycle count* (see `known-fidelity-gaps.md`), but it has no independent opinion
  about broadcast skew.
