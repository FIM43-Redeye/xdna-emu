# SP-2: Trace-Origin Reconciliation -- Design

**Sub-project of the faithful timer-sync arc.** 2026-06-28.

Parent: [`2026-06-28-timer-sync-faithful-broadcast-arc.md`](2026-06-28-timer-sync-faithful-broadcast-arc.md)
(SP-2, arc Sec.4). Builds directly on SP-1 (faithful broadcast flood,
merged to master `f5c63893`).

> **Revision note (post-review).** The first draft subtracted `module_delay`
> from the incoming `cycle` at the trace unit's entry points. An adversarial
> review found that this saturates to zero when the trace start fires at a small
> cycle (erasing the skew, and dropping early events via the `armed_start`
> check), and that it entangles with `commit_cycle`'s raw-vs-reduced comparison.
> The design below instead applies the offset as an **addition of the timer's
> own reset value** at the single absolute-encode point, leaving all cycle
> bookkeeping in the raw frame. This is simpler, has no saturation regime, and
> makes the trace and the SP-1 timer use an identical per-tile term.

---

## 1. Goal

Make each traced module's **trace Start-frame absolute origin** reflect its
broadcast-propagation delay, so cross-domain trace timestamps actually *carry*
the modeled timer skew. SP-1 baked that delay into each tile's hardware timer
value; SP-2 makes the trace stream reflect the same quantity. This is the SP-4a
prerequisite: once trace timestamps carry the skew, the emulator can reproduce
HW's cross-domain raw offsets.

**Hard invariant: within-domain deltas stay byte-identical.** Only the
Start-frame absolute value shifts. The existing trace sweep is the
origin-invariant regression gate.

---

## 2. Background: how the trace gets its timestamps today

The trace byte stream carries **exactly one absolute timestamp** -- the Start
frame, written by `encode_start` (`src/device/trace_unit/mod.rs:1344`, 7 bytes
big-endian, called once at arm time, `:605`). Every later event is encoded as a
**delta** from the previous event's cycle (`commit_pending_frame`, `:958`, via
`pending_cycle - last_event_cycle`). So a decoded event's absolute time is
`Start_absolute + sum(deltas)`.

All of those cycle values come from the global coordinator cycle
(`self.total_cycles`), sampled once per cycle and passed as the `cycle` argument
to the trace units -- identical for every tile, so the emitted trace carries
**zero** cross-tile skew.

SP-1 reset each tile's *hardware* timer (`core_timer` / `mem_timer`,
`src/device/timer.rs`) to `reset_target = max_delay - module_delay` at flood
time (`src/device/state/effects.rs:585-586`, the `core_target`/`mem_target`
locals), then it free-runs. But the trace encoder never reads that timer -- it
is on a parallel path keyed off the global cycle. That gap is what SP-2 closes.

On real silicon the trace hardware timestamps events using the tile's local
timer -- the very timer the BROADCAST_15 flood resets. So the faithful model is
"the trace's absolute origin is the tile timer's value," and the single source
of truth is the flood's already-computed `max_delay - module_delay` per module.

### Definitions

- `origin_d` -- min-latency broadcast arrival delay at a tile, from the flood
  source (`broadcast_origin_d`, Dijkstra; `effects.rs:443`). `0` at zero consts.
- `module_delay` -- per-module total delay: `core_delay = origin_d + core_off`,
  `mem_delay = origin_d + mem_off` (`effects.rs:583-584`). The core/mem split is
  the intra-tile broadcast-pipeline asymmetry.
- `max_delay` -- `max` over reached modules of `origin_d + core_off.max(mem_off)`
  (`effects.rs:580`); guarantees `max_delay - module_delay >= 0`.
- `reset_target = max_delay - module_delay` -- what SP-1 writes to the timer
  (`core_target`/`mem_target`, `effects.rs:585-586`). **SP-2 reuses this exact
  value as the trace `origin_offset`.**

---

## 3. Two rejected approaches, and why the encode-point wins

The intuitive realizations both fail; understanding why is what fixes the
surface to a single, safe point.

**Rejected A -- live per-site timer read.** Swapping the global `cycle` for
`tile.core_timer.value()` at each feed site does not survive the per-cycle loop
(`coordinator.rs`, Phase 3). Trace-feed sites fire on **both sides of the
per-tile tick** (Phase 3e, `:1581` core / `:1623` mem): before it (Phase 3b
mem-stall levels `:1441`, Phase 3c TRUE `:1465`/`:1468`, Phase 3d edge detectors
`tile/mod.rs:958`/`:972`, Phase-2 core events), and after it (Phase 3e
perf-counter fires `:1616`/`:1630`). A live read would stamp before-tick sites
`N` and after-tick sites `N+1` **within one cycle**; the one-frame-per-cycle
commit keyed on the stamp (`trace_unit/mod.rs:566`) would then split a single
cycle's events across two frames -- silent corruption.

**Rejected B -- subtract a static `module_delay` at the entry points.** This
avoids the tick hazard (a static per-tile offset applied uniformly), but the
review found two faults: (i) it **saturates** -- the trace start can fire at a
small `total_cycles`, and `total_cycles - module_delay` floors to 0, erasing the
skew and (via `armed_start`'s `cycle > arm_cycle` collapsing to `0 > 0`,
`:578`) dropping early in-window events; (ii) it **entangles with
`commit_cycle`** (`:830`, `pending_cycle > cycle`), which would receive a raw
cycle while `pending_cycle` is reduced -- a latent fragility, and the "two entry
points" inventory is wrong (`commit_cycle`, `tick`, `notify_branch_taken`,
`notify_loop_boundary` also take cycles).

**Chosen -- add the timer's reset value at the single absolute-encode point.**
The trace has exactly one absolute timestamp (Sec.2). Apply the offset there and
nowhere else; keep every bookkeeping value in the raw frame. Deltas are
differences, so a uniform shift of only the encoded absolute leaves them
byte-identical by construction, and nothing else in the trace unit is touched.

---

## 4. Design: offset the Start absolute by the timer's reset value

### 4.1 Mechanism

1. **`TraceUnit` gains `origin_offset: u64`** (default `0`) and a setter
   `set_origin_offset(&mut self, offset: u64)`.

2. **`encode_start` adds it to the encoded absolute only.** At
   `trace_unit/mod.rs:1344`, the 7 big-endian bytes encode
   `timer.wrapping_add(self.origin_offset)` instead of `timer`. The bookkeeping
   primed at arm time (`self.timer` (dead), `last_event_cycle`, `pending_cycle`
   at `:599-601`) stays **raw** -- untouched. `encode_mode2_start` (`:1407`)
   takes no cycle (it anchors on a PC), so PC-anchored traces are unaffected,
   correctly: they have no cycle origin to skew.

3. **SP-1's flood sets the offset.** In
   `propagate_broadcasts_with_timing` (`effects.rs`), **before** the existing
   `notify_*_trace_event_with_target` calls (`:602-603`), add per module,
   reusing the already-computed reset-target locals (`:585-586`):
   `tile.core_trace.set_origin_offset(core_target as u64)` and
   `tile.mem_trace.set_origin_offset(mem_target as u64)`. One Dijkstra
   computation, one pair of reset-target locals, fanned out to both the timer
   (`reset_target`) and the trace (`origin_offset`) -- they carry the *same
   value*, so they cannot disagree on *what* the offset is. (They can still
   differ on *whether* it is applied: the timer latches its reset only when the
   broadcast event matches the timer's configured `reset_event`, whereas the
   trace offset is set unconditionally on every reached tile. For the
   timer-sync use case the participating tiles all reset on BROADCAST_15, so
   they agree -- but that is an SP-4a precondition, not a property SP-2
   guarantees; see Sec.7.)

A decoded event's absolute time becomes
`(raw_start + reset_target) + sum(raw_deltas) = raw_event + reset_target`, i.e.
every decoded absolute for that module is shifted by `max_delay - module_delay`
-- the skew -- while every delta is unchanged.

### 4.2 Why add `max_delay - module_delay`, not subtract `module_delay`

The two forms differ only by the global constant `max_delay` (and, vs. the live
timer, by the flood cycle `F`). All three are equivalent for cross-domain
purposes, because any global constant cancels in a cross-domain *difference* and
in the trace sweep's per-tile re-anchoring. The `max_delay - module_delay` form
is chosen for three concrete wins:

- **No saturation.** `max_delay >= module_delay` by construction
  (`effects.rs:580`), so the offset is non-negative and the Start absolute is
  `raw_cycle + nonneg` -- never underflows, regardless of how early the start
  fires. This is the fault that killed the subtract form.
- **Representation-identical to the timer.** SP-1 put `max_delay - module_delay`
  in the timer; SP-2 encodes the same value. The trace literally reproduces the
  tile timer's reset baseline (decoded absolute = timer value + global `F`),
  which is exactly "the trace reads the tile timer." SP-4a then compares
  EMU-vs-HW with the trace and timer carrying the same per-tile value -- with
  the application caveat in Sec.4.1 / Sec.7 (the timer reset is conditional on
  the `reset_event` match).
- **DRY.** It is the `core_target`/`mem_target` SP-1 already computed; SP-2 adds
  no arithmetic.

### 4.3 Surface

- `src/device/trace_unit/mod.rs`: one field (`origin_offset: u64`), one setter,
  one `wrapping_add` inside `encode_start`. No change to `notify_event`,
  `set_event_level`, `commit_cycle`, or any other cycle-bearing method.
- `src/device/state/effects.rs`: two setter calls in
  `propagate_broadcasts_with_timing`, placed before the broadcast notify
  (`:602`).

No coordinator changes, no feed-site changes, no entry-point changes. The ~11
trace-feed call sites are untouched.

### 4.4 Invariants

- **Within-domain byte-identity (trivial).** Only the Start frame's 7 absolute
  bytes change; deltas, `pending_cycle`, `last_event_cycle`, `commit_cycle`, and
  `armed_start` all operate on raw cycles exactly as today.
- **Exact behavior-neutrality at zero constants.** With all `BroadcastTiming`
  constants `0`: every `origin_d = 0` (test
  `broadcast_origin_d_reached_set_all_zero_at_zero_delays`, `effects.rs`),
  `core_off = mem_off = 0`, so `max_delay = 0` and every `reset_target = 0`.
  Thus `origin_offset = 0` and `encode_start` adds `0` -- byte-for-byte identical
  output.
- **No saturation regime.** The offset is non-negative; the Start absolute never
  underflows.
- **Monotonic, static offset.** The flood is config-time (`dispatch.rs:219`,
  before execution/tracing); `origin_offset` is set once and never changes
  mid-stream.
- **Uniform trace-unit <-> timer pairing.** Only `core_trace` and `mem_trace`
  exist (no separate shim/PL trace unit). Shim PL events route to `core_trace`
  (`tile/mod.rs:953-958`), DMA to `mem_trace`. So `core_trace` takes
  `core_target`, `mem_trace` takes `mem_target`, on every tile kind.

### 4.5 Ordering requirement (set offset before any start can arm)

`origin_offset` must be set on a tile before that tile's `start_event` can arm
its trace (the arm is where `encode_start` runs). Two cases:

- **Reached tiles (the flood loop).** The setter calls go **before** the
  `notify_*_trace_event_with_target` calls at `:602-603`, so if a reached tile's
  `start_event` happens to equal the broadcast event id, the offset is already
  set when the broadcast notify could arm it. (Mandated, not incidental -- a
  plan that places the setter after the notify reintroduces the bug for that
  edge case.)
- **The flood-source tile's config-time Event_Generate.** The source tile fires
  its generating event locally at `effects.rs:396-397`, inside
  `apply_tile_local_effects` (`dispatch.rs:214`), which runs **before**
  `propagate_broadcasts_fixpoint` (`:219`) computes `origin_d` and sets the
  offset. This is benign in every real trace config because the timer-sync
  *trigger* event written via Event_Generate is never a trace `start_event`
  (start is configured as TRUE or a perf/timer event and arms during execution,
  after config). The source-tile guard test (Sec.6.3) enforces this assumption.

---

## 5. What SP-2 deliberately does NOT do

- **It does not validate cross-domain correctness.** Whether the traced tiles
  are flood-reached and their delays match silicon is SP-4a's gate
  (`emu_raw == hw_raw` on the validation kernel). SP-2 only wires the path and
  proves within-domain neutrality. The trace sweep re-anchors each tile to its
  own first shared edge, so it is *blind* to the cross-tile origin shift by
  construction -- it confirms no within-domain regression but cannot positively
  confirm the shift. Positive confirmation is a unit test (Sec.6).
- **It does not touch the inference engine.** That is SP-4b, which reads the
  delays and `W_sim` directly and does not depend on SP-2 (arc Sec.4).
- **It does not change the timer.** SP-1 owns the timer; SP-2 reuses the flood's
  `reset_target` value and applies it on the trace side only.

---

## 6. Test strategy

1. **Positive -- cross-tile (unit).** With nonzero `BroadcastTiming` constants,
   two reached tiles at different `module_delay` produce Start-frame absolutes
   differing by exactly `Delta(reset_target) = Delta(max_delay - module_delay)`,
   while their inter-event deltas are byte-identical. This is the test that
   proves SP-2 did something, since the sweep cannot.
2. **Positive -- intra-tile (unit).** On one reached tile with
   `core_off != mem_off`, the `core_trace` and `mem_trace` Start absolutes differ
   by exactly `|core_off - mem_off|` (the two reset targets differ by
   `mem_delay - core_delay = mem_off - core_off`).
3. **Source-tile guard (unit).** Trace the **flood-source** tile with
   `core_off != mem_off != 0`; assert its Start carries the offset and its first
   post-Start delta is byte-identical to the zero-const reference -- i.e. the
   offset reached the source tile before its start armed (Sec.4.5).
4. **Neutrality (unit).** With zero constants, emitted trace bytes are identical
   to a pre-SP-2 reference (`origin_offset == 0` everywhere; `encode_start` adds
   `0`).
5. **Integration.** `cargo test --lib` green; the trace sweep verdict stays
   CLEAN on the existing kernels (origin-invariant gate).

---

## 7. Open items / risks

- **Mid-run re-sync.** If a kernel re-issues the timer-sync (BROADCAST_15) event
  during the trace window, the flood re-fires. Topology is unchanged, so every
  `reset_target` -- and thus `origin_offset` -- is identical; the offset stays
  static and the Start is already emitted, so nothing shifts. Documented
  assumption; no kernel in the current suite does this.
- **Unreached tiles.** A tile never reached by the flood keeps
  `origin_offset = 0` (default) and encodes its raw Start as today. Correct for
  within-domain; its cross-domain meaningfulness is an SP-4a concern.
- **Conditional timer vs unconditional trace (SP-4a precondition).** The flood
  hands the *same* `max_delay - module_delay` value to the timer and the trace,
  but applies it asymmetrically: the timer latches its reset only when the
  broadcast event matches the timer's configured `reset_event` (`timer.rs`,
  `notify_event_with_target`), while `set_origin_offset` runs unconditionally on
  every reached tile. So a tile reached by the flood whose timer is *not*
  configured to reset on that broadcast carries the skew in its trace but not in
  its timer. This never breaks an SP-2 invariant (within-domain byte-identity
  and zero-const neutrality hold regardless). But the EMU-vs-HW cross-domain
  match SP-4a validates assumes the participating tiles reset on the broadcast,
  so trace and timer agree -- SP-4a must confirm the validation kernel
  (SP-3) configures the timer reset on every traced cross-domain tile.
- **Sign of the intra-tile core/mem asymmetry.** SP-1 models it as a per-module
  offset (`core_off`/`mem_off`); the add_one `+2/+4/-2` signature (arc Sec.7) is
  the reference. SP-2 inherits whatever `core_target`/`mem_target` SP-1 produces
  and does not re-derive it -- so if SP-1's sign convention is later corrected,
  SP-2 needs no change, and the trace stays consistent with the timer by
  construction.
