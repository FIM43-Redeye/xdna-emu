# Parameterizable DDR burst-delivery model (task #140, folding in #129/#131)

**Date:** 2026-06-14
**Status:** SCOPING — design for review, no code yet.
**Prereq landed:** comparator shim-labeling fix (`3430dc3f`) — the #140
frontier histogram now carries trustworthy event names.

---

## 1. What we're actually fixing

The honest bridge corpus is **15 CLEAN / 125 DIVERGE**. With shim events now
correctly labeled, the real mode-0 (genuine-cycle) divergence frontier is:

| event (corrected) | module | signal | ~kernels |
|---|---|---|---|
| `DMA_S2MM_{0,1}_STREAM_STARVATION` | shim | EMU under-emits (HW 7222 / EMU 4270 total; HW>EMU in 103/124) | ~120 |
| `DMA_{MM2S,S2MM}_*_{START,FINISHED}_TASK` | shim | relative milestone timing off by 100s of cycles | ~125 |
| `PORT_RUNNING_*` | memtile | EMU under-segments (HW 2054 / EMU 1090; HW>EMU 40/40) | ~40 |
| `CONFLICT_DM_BANK_*` | mem | smaller, **mixed direction** — separate mechanism | ~40 |

The first three share **one root**, already documented as known-fidelity-gaps
**row 50** and aiesim **row 117**:

> EMU stream/DMA delivery is *smoother* (continuous) than HW's bursty
> ~1024-cycle DDR bursts, so level events assert fewer times. DMA/DDR input
> fill latency is under-modeled — a constant offset (~6131 ns optimistic in the
> aiesim leg), not per-iteration drift.

So the model task is: **make host-DDR delivery bursty and latency-gated instead
of uniform**, so the S2MM channel starves (and ports toggle) the way real
silicon does. `CONFLICT_DM_BANK` is explicitly **out of scope** here (mixed
direction → not the smoothness root; its own investigation).

## 2. Design intent (Maya's call, 2026-06-14)

> Loop in #129/#131 for a full DDR latency/bandwidth model BUT don't worry about
> getting it EXACT — the framework should exist so people can pick their own
> params.

So this is a **framework, not a calibration**. The deliverable is a
parameterizable DDR-delivery model with:
- Sensible Phoenix-shaped defaults (ballpark, deterministic — matches the
  "DMA timing target is ballpark+deterministic" calibration guidance).
- Every knob overridable (config + env) so a user targeting a different part,
  DDR speed grade, or workload can dial it in themselves.
- No magic numbers buried in delivery logic — all timing lives in one config
  struct, derived/defaulted in one place.

This subsumes the two deferred DMA timing tasks:
- **#131** (warmup-length / bw=32) — already partly modeled in
  `DmaTimingConfig` (cold-start + geometric warmup decay). The burst model is
  the same structure extended; #131's knobs become the "first-burst latency"
  special case.
- **#129** (DMA iter/chain HW-faithful redesign) — BD iteration re-loops the
  same buffer; under a burst model each iteration re-incurs burst gaps. Folding
  iter into the same delivery loop keeps one timing model, not two.

## 3. Current model (the seam)

Delivery today is **per-cycle uniform**. One injection point:

- **`src/device/dma/engine/stepping.rs:863` `do_transfer_cycle`** — pushes
  `words_per_cycle` (tile-local, 4) or `shim_words_per_cycle` (shim DDR, 1)
  words to `stream_out` every cycle the channel is `Transferring`. No gaps, no
  per-beat latency. This uniformity *is* the "too smooth" bug.
- **`src/device/dma/timing.rs:19` `DmaTimingConfig`** — the existing structured
  knob bag: `bd_setup_cycles`, `channel_start_cycles`, `words_per_cycle`,
  `shim_words_per_cycle`, `shim_ddr_cold_start_*_cycles`,
  `shim_per_task_overhead_*_cycles`, `shim_warmup_decay_*_permille`,
  `memory_latency_cycles`. The burst model **extends this struct** — the right
  home, already plumbed everywhere.
- **`src/device/dma/engine/stepping.rs:583` `step_transferring_cycle`** —
  emits `DmaStreamStarvation{active}` on the S2MM stall rising/falling edge,
  driven by input-FIFO availability (`transfer_s2mm`, stepping.rs:1628). This
  is what the burst model must *drive*: when a burst ends and the next hasn't
  arrived, the consumer FIFO drains → stall edge → starvation event. We do not
  emit starvation directly; we make delivery bursty and let the existing edge
  logic fire naturally. (Physically faithful — matches how HW produces it.)
- **`src/interpreter/engine/coordinator.rs:977` step()** — steps all DMA once
  per `total_cycles` tick; delivery is per-cycle deterministic. Good: a burst
  accumulator keyed on the cycle counter is naturally deterministic and
  resume-safe.

## 4. Proposed framework

### 4.1 Config (extend `DmaTimingConfig`)

New knobs (names indicative), defaulted to Phoenix-ballpark, all overridable:

```
ddr_burst_words:            u16   // words delivered per burst (e.g. 256)
ddr_inter_burst_cycles:     u16   // gap between bursts (e.g. ~1024 HW-shaped)
ddr_first_access_latency:   u16   // latency before first beat (folds #131 cold-start)
ddr_burst_jitter_permille:  u16   // optional deterministic jitter on gap (0 = none)
```

`shim_ddr_cold_start_*` and `shim_warmup_decay_*` (#131) are reframed as the
first-burst special case: first-access latency that decays across a BD chain.
We keep their semantics; the burst model layers the *steady-state* gaps that
#131 never modeled (it only shaped the warm-up transient).

### 4.2 Delivery loop (burst accumulator in `do_transfer_cycle`)

Per-channel state (add to `ChannelContext`, channel.rs):
```
burst_words_remaining:  u16   // words left in the current burst
inter_burst_wait:       u16   // cycles left before next burst opens
```
Logic, only for shim channels touching host memory (`involves_host_memory()`):
- If `inter_burst_wait > 0`: decrement, deliver **0 words** this cycle → the
  downstream consumer FIFO drains → S2MM stalls → starvation edge fires (exactly
  the HW mechanism).
- Else deliver up to `min(shim_words_per_cycle, burst_words_remaining)`; when a
  burst empties, reload `burst_words_remaining = ddr_burst_words` and set
  `inter_burst_wait = ddr_inter_burst_cycles` (± deterministic jitter).

Tile-local (non-host) transfers keep uniform `words_per_cycle` — bursting is a
**DDR/AXI-master property**, not an on-chip stream property. (Bounds the blast
radius: memtile↔compute streams are unaffected.)

### 4.3 Configuration surface (the "pick your own params" requirement)

- A `DmaTimingConfig` constructor with named Phoenix defaults (already the
  pattern). Add a `from_env()` overlay: `XDNA_EMU_DDR_BURST_WORDS`,
  `XDNA_EMU_DDR_INTER_BURST_CYCLES`, `XDNA_EMU_DDR_FIRST_LATENCY`, etc.,
  parsed once at device construction.
- (Stretch) a small TOML/JSON profile file so a user can ship a "Strix DDR5"
  vs "Phoenix LPDDR5" profile without recompiling. Schema-first if we do it.

## 5. Validation strategy

Per the calibration guidance (ballpark + deterministic, not bit-exact):
- **Primary gate:** the bridge trace corpus `TRACE_VERDICT`. Target: move
  STREAM_STARVATION / PORT_RUNNING / DMA-milestone kernels from DIVERGE toward
  CLEAN, or at least shrink the count/timing gaps materially. We will NOT chase
  exact per-event cycle equality — the framework's job is structural fidelity
  (right number of bursts/starvations, right ballpark gaps).
- **Unit tests (TDD):** burst accumulator emits N starvation edges for a feed
  of known size with known burst/gap params; uniform path unchanged for
  tile-local transfers; env overlay parses and overrides.
- **Regression:** full `cargo test --lib`; full bridge corpus verdict re-tally;
  diff against this baseline (15 CLEAN / 125 DIVERGE) — must not regress CLEAN
  kernels, and must improve the DMA-delivery family.
- **Cross-check:** the total-cycle ratio and mode-0 DMA anchors
  (`trace-anchors.py` → `timing-three-way.py`) must stay within their existing
  tolerance — burst gaps add starvations but should not blow up end-to-end
  cycle counts beyond HW.

## 6. Scope boundaries

**In:** shim host-DDR MM2S/S2MM burst+latency delivery; the config framework;
folding #131 warmup as first-burst latency; folding #129 iter into the same
delivery loop; STREAM_STARVATION / PORT_RUNNING / DMA-milestone fidelity.

**Out:** `CONFLICT_DM_BANK` (separate mechanism, mixed direction — own task);
core/EVENT_PC trace events (handled by the mode-aware comparator already);
exact per-event cycle matching; modeling the NoC/memory-controller internals
beyond a burst+latency abstraction.

## 7. Decisions (resolved 2026-06-14)

1. **#129 iter scope** → **burst-delivery first, iter next.** Land and validate
   the burst+latency model, then do #129 iter as an immediate follow-up reusing
   the same delivery loop. Iter does not gate the dominant STREAM_STARVATION fix.
2. **Config surface** → **env-var overlay now, profile file later.** Phoenix
   defaults in `DmaTimingConfig` + `XDNA_EMU_DDR_*` env overrides parsed at
   device construction. TOML/JSON profile file deferred until a real second-part
   use case (e.g. Strix DDR5) lands.
3. **Burst-size default** → derive from the observed ~1024-cycle HW burst cadence
   (row 50/117), documented OBSERVED, user-overridable.
4. **Baseline** → trust the on-disk 20260614 corpus as the before-baseline
   (fresh as of today); HW re-capture only if a result looks suspect.
