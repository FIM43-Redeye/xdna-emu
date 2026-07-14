# Known Fidelity Gaps

A registry of **confirmed** points where the emulator (or the aiesim oracle)
provably disagrees with real NPU hardware. Each gap is documented in detail in a
linked finding; this page and the [`fidelity-gaps/`](fidelity-gaps/) folder are
the index, so the pattern is visible in one place and we don't re-investigate the
same gap twice. (That happened once already: a stale "shim queue is 8-deep"
caveat sent a whole session chasing a dead mechanism.)

**Scope:** confirmed HW-disagreement gaps only -- behaviors where we have ground
truth that the model is wrong. This is *not* the list of not-yet-implemented
features (vector compute semantics, stream-switch per-port types, micro-timing);
those live in [`toolchain-sources.md`](toolchain-sources.md) and the
[roadmap](../ROADMAP.md).

**How these surface:** mostly via the aiesim oracle (the XRT-plugin -> aiesim
path). The count rising is the oracle doing its job -- before it, these were
invisible. Each was a deliberate "document, don't chase" call; see the per-gap
rationale in the class file.

**How this is organized.** Gaps are classified by **subsystem** (the file
boundary) -- when you're working on a subsystem, its file shows you every known
gap there so you don't re-chase one. Two categories are a genuinely *different
kind* of gap (model more capable than HW; gap in the oracle not us) and get
posture-named files instead. Each class file carries a `posture:` field in its
frontmatter, so the cross-cutting "what kind of wrong" axis stays queryable.

---

## Class directory

| Class | File | Posture | Gaps (status) |
|-------|------|---------|---------------|
| **DMA & stream resources** | [`fidelity-gaps/dma-stream-resources.md`](fidelity-gaps/dma-stream-resources.md) | optimistic-where-strict | TCT token buffer (open), BD reuse/pool (won't-fix), send/recv port cadence (recv exact; send substantially resolved, cold-start residual superseded by core-reset), decompression bank-demand under-claim (open, bounded) |
| **Trace encoding** | [`fidelity-gaps/trace-encoding.md`](fidelity-gaps/trace-encoding.md) | encoding-artifact | held-level falling-edge (documented), count under-emission (**closed** 2026-06-27) |
| **Core compute timing** | [`fidelity-gaps/core-compute-timing.md`](fidelity-gaps/core-compute-timing.md) | needs-HW-empirical | MEMORY_STALL mechanism root-caused + modelled (bank-arbitration arc, HW-confirmed cycle-by-cycle); DMA bank-demand cadence **fixed** 2026-07-14 (16B granule, HW-measured -- producer 102->4 vs HW 1); residual consumer over-production is core-vs-core port conflict (open, new); S2MM_BACKPRESSURE wiring removed as HW-disproven (emits 0, gap registered); lock-arb cost modeled, pulse emission pending; bank-arbiter resume-cycle hole (open, dormant/unreachable on corpus) |
| **Vector compute** | [`fidelity-gaps/vector-compute.md`](fidelity-gaps/vector-compute.md) | mostly-resolved | accumulator bypass fold (deferred), source_forward alignment (benign), NaN payload regime (**resolved**) |
| **Event broadcast** | [`fidelity-gaps/event-broadcast.md`](fidelity-gaps/event-broadcast.md) | fixed | shim-row E/W broadcast edge (**fixed** 2026-07-02) |
| **Host / firmware dispatch** | [`fidelity-gaps/host-firmware-dispatch.md`](fidelity-gaps/host-firmware-dispatch.md) | off-array / firmware seam | dispatch latency (deferred), core reset-deassert (**root-caused**, Part 1 landed `1e9e6700`) |
| **Firmware MMU** | [`fidelity-gaps/firmware-mmu.md`](fidelity-gaps/firmware-mmu.md) | needs-HW-empirical | double-fault EPC1/DEPC (inert), EXCVADDR on autorefill (unobservable), varway56 ways 5/6 (**open**, blocks M2c) |
| **Permissive-vs-HW** (inverse gap) | [`fidelity-gaps/permissive-vs-hw.md`](fidelity-gaps/permissive-vs-hw.md) | permissive-where-broken | task-API memtile-relay TDR (documented) |
| **aiesim oracle** | [`fidelity-gaps/aiesim-oracle.md`](fidelity-gaps/aiesim-oracle.md) | oracle-bound | c2c shared memory, control-read aliasing, cross-domain skew, trace micro-timing |

---

## Maintenance

When a new confirmed HW-disagreement surfaces:

1. Add a one-line row to the appropriate **class file** under
   [`fidelity-gaps/`](fidelity-gaps/), pointing at the detailed finding, and note
   whether it's fixed, documented-and-deferred, or won't-fix.
2. Update that class file's frontmatter `status:` line and this page's class
   directory row if the class's headline status changed.
3. If the gap is a genuinely new *kind* (a new subsystem, or a new posture like
   the inverse `permissive-vs-hw` category), add a new class file and a directory
   row here.

Keep the class files an index too -- current-state plus a pointer; the
blow-by-blow lives in the linked findings under
[`superpowers/findings/`](superpowers/findings/).
