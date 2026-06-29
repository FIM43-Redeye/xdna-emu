# SP-2: Trace-Origin Reconciliation -- Design

**Sub-project of the faithful timer-sync arc.** 2026-06-28.

Parent: [`2026-06-28-timer-sync-faithful-broadcast-arc.md`](2026-06-28-timer-sync-faithful-broadcast-arc.md)
(SP-2, arc Sec.4). Builds directly on SP-1 (faithful broadcast flood,
merged to master `f5c63893`).

---

## 1. Goal

Make each traced module's **trace Start-frame absolute origin** reflect its
broadcast-propagation delay `origin_D`, so cross-domain trace timestamps
actually *carry* the modeled timer skew. SP-1 baked `origin_D` into each
tile's hardware timer value; SP-2 makes the trace stream reflect it. This is
the SP-4a prerequisite: once trace timestamps carry the skew, the emulator can
reproduce HW's cross-domain raw offsets.

**Hard invariant: within-domain deltas stay byte-identical.** Only the
Start-frame absolute value shifts. The existing trace sweep is the
origin-invariant regression gate.

---

## 2. Background: how the trace gets its timestamps today

Every trace timestamp comes from the global coordinator cycle
(`self.total_cycles`, `coordinator.rs`), sampled once per cycle and passed as
the `cycle` argument to the trace units. It is identical for every tile, so the
emitted trace carries **zero** cross-tile skew. The trace unit
(`src/device/trace_unit/mod.rs`) seeds its delta-encoder from that `cycle` at
the Start frame (`mod.rs:599`, `encode_start(cycle)` at `:605`) and
delta-encodes subsequent events from the same reference.

SP-1 reset each tile's *hardware* timer (`core_timer` / `mem_timer`,
`src/device/timer.rs`) to `reset_target = max_delay - origin_D` at flood time,
then it free-runs (`tick()` increments 1:1 with the cycle). But the trace
encoder never reads that timer -- it is on a parallel path keyed off the global
cycle. That gap is exactly what SP-2 closes.

On real silicon the trace hardware timestamps each event using the tile's
local timer -- the very timer the BROADCAST_15 flood resets. So the faithful
model is "the trace reflects the tile timer's reset," and the single source of
truth for the per-tile origin shift is the flood's `origin_D` computation
(SP-1's `broadcast_origin_d`, `src/device/state/effects.rs`).

---

## 3. The decisive finding: a live per-site timer read is unsafe

The intuitive realization of "trace reads the tile timer" -- swap the global
`cycle` for `tile.core_timer.value()` at each feed site -- does **not** survive
the per-cycle loop structure (`coordinator.rs`, Phase 3 sub-phases).

Trace-feed sites fire on **both sides of the per-tile tick** (Phase 3e,
`coordinator.rs:1581` core / `:1623` mem):

- **Before the tick:** Phase 3b mem-stall levels (`:1441`), Phase 3c TRUE
  (`:1465`/`:1468`), Phase 3d edge detectors (`tile/mod.rs:958`/`:972`), and
  Phase-2 core events (e.g. `interpreter/core/interpreter.rs:21`).
- **After the tick:** Phase 3e perf-counter fires (`:1616`/`:1630`).

Today all sites stamp from one `cycle = self.total_cycles` sampled once per
cycle, so they agree. If each site instead read `timer.value()` live, the
before-tick sites would read `N` and the after-tick sites `N+1` **within the
same cycle**. The trace unit commits at most one frame per cycle, keyed on the
stamp (`trace_unit/mod.rs:566`: a stamp change with pending slot activity forces
a premature `commit_pending_frame()`). A mismatched mid-cycle stamp therefore
splits one cycle's events across two frames -- silent corruption, not a cosmetic
off-by-one.

Conclusion: the per-tile origin shift must be applied as a **static offset**,
not a live read. The flood's `origin_D` is the source; the trace unit is the
application point.

---

## 4. Design: per-trace-unit origin offset, set by the flood

### 4.1 Mechanism

1. **`TraceUnit` gains `origin_offset: u64`** (default `0`). It is subtracted
   (saturating) from the incoming `cycle` at the trace unit's two public
   entry points -- `notify_event` and `set_event_level` -- before any other
   use. Every downstream consumer (Start-frame absolute, `pending_cycle`,
   `last_event_cycle`, delta encoding) then sees one uniformly-shifted clock.

2. **SP-1's flood sets it.** `propagate_broadcasts_with_timing`
   (`src/device/state/effects.rs:582-586`) already iterates the reached tiles
   and computes, as locals, each one's per-module total delay:
   `core_delay = origin_d + core_off` and `mem_delay = origin_d + mem_off`. SP-2
   adds, alongside the existing `notify_*_trace_event_with_target` calls
   (`:602-603`), a setter call per module reusing those exact locals:
   `tile.core_trace.set_origin_offset(core_delay as u64)` and
   `tile.mem_trace.set_origin_offset(mem_delay as u64)`. One Dijkstra
   computation, one pair of delay locals, fanned out to both the timer
   (`reset_target`) and the trace unit (`origin_offset`) in the same lines --
   they cannot drift, and SP-2 introduces no new arithmetic.

3. **Stamp semantics: `stamp = total_cycles - module_delay`** (saturating),
   where `module_delay = origin_D + intra_offset` (`core_delay` for
   `core_trace`, `mem_delay` for `mem_trace`). A distant tile has a larger
   `module_delay`, hence a smaller timestamp -- exactly the silicon relationship
   (it reset its timer later, so at a common wall cycle its timer reads less).
   The intra-tile core/mem asymmetry falls out for free: `core_off != mem_off`
   makes the two trace units on one tile take different offsets.

### 4.2 Why the trace offset is `module_delay`, not the timer's `reset_target`

The flood derives two scalars from one `module_delay`: the timer
`reset_target = max_delay - module_delay` and the trace
`origin_offset = module_delay` (they sum to `max_delay`). The `max_delay` term
exists only so the *hardware-timer value* stays non-negative as a `u64` after
reset. The trace stamp is `total_cycles - module_delay`; `total_cycles` is
large, so no non-negativity offset is needed, and folding in `max_delay` would
only inject a harmless global constant. SP-2 therefore uses `module_delay`
directly -- exactly the `core_delay`/`mem_delay` SP-1 already computed.

(`total_cycles - module_delay` differs from the literal timer value only by the
global constant `max_delay - F` (F = flood cycle), so it is equivalent for
cross-domain purposes; the direct `module_delay` form is chosen because it gives
exact byte-neutrality at zero constants -- see 4.4.)

### 4.3 Surface

- `src/device/trace_unit/mod.rs`: one field (`origin_offset: u64`), one setter
  (`set_origin_offset`), two saturating subtractions at the `notify_event` and
  `set_event_level` entry points.
- `src/device/state/effects.rs`: two setter calls in
  `propagate_broadcasts_with_timing` (per reached tile, per module).

No coordinator changes, no feed-site changes. The ~11 trace-feed call sites are
untouched.

### 4.4 Invariants

- **Within-domain byte-identity.** A uniform per-stream shift leaves every
  consecutive-event delta unchanged; only the Start-frame absolute moves.
- **Exact behavior-neutrality at zero constants.** With all `BroadcastTiming`
  constants `0`, every `origin_D = 0` and `core_off = mem_off = 0`, so every
  `module_delay = 0`, every `origin_offset = 0`, and `stamp = total_cycles` --
  byte-for-byte identical to current output. (This is strictly stronger than a
  live-timer read, which would shift Start by the flood cycle and be neutral
  only after the sweep's per-tile re-anchoring.)
- **Monotonic within the trace window.** The flood is config-time
  (`dispatch.rs`, before execution/tracing); `origin_offset` is static
  thereafter, so no timestamp jumps backward mid-stream.
- **Trace-unit <-> timer pairing is uniform.** Only `core_trace` and
  `mem_trace` exist (no separate shim/PL trace unit). Shim PL events route to
  `core_trace` (`tile/mod.rs:953-958`), DMA to `mem_trace`. So
  `core_trace` pairs with `core_timer`/`core_delay`, `mem_trace` with
  `mem_timer`/`mem_delay`, on every tile kind.

### 4.5 Saturating subtraction

`module_delay` is small (single-digit to low-tens of cycles); the Start event
fires well after config in real runs, so `total_cycles >= module_delay` holds in
practice. `saturating_sub` is used as a floor-at-zero guard for degenerate
early-trace/unit-test cases; it never distorts a real cross-domain segment
because those start long after `total_cycles` exceeds `module_delay`.

---

## 5. What SP-2 deliberately does NOT do

- **It does not validate cross-domain correctness.** Whether the traced tiles
  are flood-reached and their `origin_D` values match silicon is SP-4a's gate
  (`emu_raw == hw_raw` on the validation kernel). SP-2 only wires the path and
  proves within-domain neutrality. The trace sweep re-anchors each tile to its
  own first shared edge, so it is *blind* to the cross-tile origin shift by
  construction -- it confirms no within-domain regression but cannot positively
  confirm the shift. Positive confirmation is a unit test (Sec.6).
- **It does not touch the inference engine.** That is SP-4b, which reads
  `origin_D` and `W_sim` directly and does not depend on SP-2 (arc Sec.4).
- **It does not change the timer.** SP-1 owns the timer; SP-2 only reads the
  flood's `origin_D` and applies it on the trace side.

---

## 6. Test strategy

1. **Positive (unit).** With nonzero `BroadcastTiming` constants, two tiles at
   different hop distances produce Start-frame absolutes differing by exactly
   `Delta module_delay`, while their inter-event deltas are byte-identical. A
   companion assertion covers the intra-tile case: on one tile with
   `core_off != mem_off`, the `core_trace` and `mem_trace` Start absolutes differ
   by exactly `core_off - mem_off`. This is the test that proves SP-2 did
   something, since the sweep cannot.
2. **Neutrality (unit).** With zero constants, the emitted trace bytes are
   identical to a pre-SP-2 reference (every `origin_offset == 0`).
3. **Within-stream consistency (unit).** Events notified across the
   before-tick / after-tick phase split in one cycle still land in a single
   committed frame (no premature split) -- the regression guard against the
   Sec.3 hazard, proving the static-offset design avoids it.
4. **Integration.** `cargo test --lib` green; the trace sweep verdict stays
   CLEAN on the existing kernels (origin-invariant gate).

---

## 7. Open items / risks

- **Mid-run re-sync.** If a kernel re-issues the timer-sync (BROADCAST_15)
  event during the trace window, the flood re-fires. Topology is unchanged, so
  `module_delay` -- and thus `origin_offset` -- is identical; the offset stays
  static and the invariants hold. Documented assumption; no kernel in the
  current suite does this.
- **Unreached tiles.** A tile never reached by the flood keeps
  `origin_offset = 0` (default) and stamps `total_cycles` as today. Correct for
  within-domain; its cross-domain meaningfulness is an SP-4a concern, not SP-2's.
- **Sign of the intra-tile core/mem asymmetry.** SP-1 models it as a per-module
  offset (`core_off`/`mem_off`); the add_one `+2/+4/-2` signature (arc Sec.7) is
  the reference. SP-2 inherits whatever `core_delay`/`mem_delay` SP-1 produces
  and does not re-derive it -- so if SP-1's sign convention is later corrected,
  SP-2 needs no change.
