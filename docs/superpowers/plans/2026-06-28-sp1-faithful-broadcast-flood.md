# SP-1: Faithful Broadcast Flood Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Model real per-hop BROADCAST_15 timer-reset propagation in the emulator so each tile's timer holds a constant skew baseline (`origin_D = n_h*d_h + n_v*d_v` plus an intra-tile offset) relative to the flood source, instead of all tiles resetting on the same cycle.

**Architecture:** A Dijkstra wavefront over the broadcast adjacency computes each reached tile's `origin_D`; the flood derives `delay = origin_D + intra_tile_offset` per module and latches a reset *target* (`max_delay - delay`) into each timer; the timer's existing `pending_reset` latch is generalized to reset to that target on the first execution tick instead of always to 0. All latency constants live in a new archspec `BroadcastTiming` struct defaulting to **zero**, so the shipped emulator is byte-identical to today and the mechanism is exercised only by unit tests passing explicit constants.

**Tech Stack:** Rust; `xdna-archspec` build-time codegen crate (struct -> model_builder -> build.rs const emission -> `xdna_archspec::aie2::timing::*`); the emulator device-state flood (`src/device/state/effects.rs`) and per-module `TileTimer` (`src/device/timer.rs`).

**Design spec:** [`docs/superpowers/specs/2026-06-28-sp1-faithful-broadcast-flood-design.md`](../specs/2026-06-28-sp1-faithful-broadcast-flood-design.md). Arc context: [`2026-06-28-timer-sync-faithful-broadcast-arc.md`](../specs/2026-06-28-timer-sync-faithful-broadcast-arc.md).

## Global Constraints

- **Behavior-neutral on ship.** All `BroadcastTiming` fields default to 0; with zero constants every `delay = 0`, every reset target = 0, and behavior is byte-identical to today. `cargo test --lib` must stay green at every commit, and the full trace sweep must show no divergence vs. a pre-change baseline (diff `trace.log`, do not trust bridge pass/fail -- per `feedback_verify_against_baseline_trace`).
- **Derive from the toolchain.** Broadcast event IDs come from the aie-rt `xaie_events_aieml.h` values already encoded in `events/mod.rs::EventModuleType::broadcast_event_base` (Core/Mem 107, Pl 110, MemTile 142). Do not re-hardcode them.
- **`u8`, not `i8`, for offsets.** `delay = origin_D + offset` must be non-negative; never introduce a signed delay.
- **No emoji. Commit messages end with the two trailer lines** (`Generated using Claude Code.` / `Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh`).
- **Run `cargo build` / `cargo test` bare** (never piped through head/tail/grep); redirect to a file and Read if long.

---

### Task 1: `BroadcastTiming` constants home (archspec)

Add the per-hop / intra-tile latency constants to the archspec timing model, all defaulting to 0, emitted as `xdna_archspec::aie2::timing::BROADCAST_*` consts. No emulator behavior changes.

**Files:**
- Modify: `crates/xdna-archspec/src/types.rs` (add `BroadcastTiming` struct ~after `StreamSwitchTiming` at :1340; add field to `TimingModel` at :1361-1366)
- Modify: `crates/xdna-archspec/src/model_builder.rs:268-270` (construct `BroadcastTiming` in the NPU1 `TimingModel`)
- Modify: `crates/xdna-archspec/build.rs:553` (emit the four consts before the timing module's closing brace at :555)
- Test: `src/device/state/effects.rs` (a `#[test]` asserting the four consts are emitted and 0)

**Interfaces:**
- Produces: `xdna_archspec::aie2::timing::BROADCAST_PER_HOP_HORIZONTAL: u8`, `BROADCAST_PER_HOP_VERTICAL: u8`, `BROADCAST_INTRA_TILE_CORE_OFFSET: u8`, `BROADCAST_INTRA_TILE_MEM_OFFSET: u8` (all 0 for NPU1).

- [ ] **Step 1: Write the failing test**

In `src/device/state/effects.rs`, inside the existing `#[cfg(test)] mod interrupt_path_tests` block (or a new `#[cfg(test)] mod broadcast_timing_consts_tests`), add:

```rust
#[test]
fn broadcast_timing_consts_default_to_zero() {
    use xdna_archspec::aie2::timing::{
        BROADCAST_INTRA_TILE_CORE_OFFSET, BROADCAST_INTRA_TILE_MEM_OFFSET,
        BROADCAST_PER_HOP_HORIZONTAL, BROADCAST_PER_HOP_VERTICAL,
    };
    // SP-1 ships behavior-neutral: real values arrive in SP-5 (silicon).
    assert_eq!(BROADCAST_PER_HOP_HORIZONTAL, 0);
    assert_eq!(BROADCAST_PER_HOP_VERTICAL, 0);
    assert_eq!(BROADCAST_INTRA_TILE_CORE_OFFSET, 0);
    assert_eq!(BROADCAST_INTRA_TILE_MEM_OFFSET, 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib broadcast_timing_consts_default_to_zero`
Expected: FAIL to compile -- `BROADCAST_PER_HOP_HORIZONTAL` not found in `xdna_archspec::aie2::timing`.

- [ ] **Step 3: Add the `BroadcastTiming` struct and `TimingModel` field**

In `crates/xdna-archspec/src/types.rs`, after the `StreamSwitchTiming` struct (closes at :1340), add:

```rust
/// Broadcast-event timer-reset propagation timing.
///
/// The BROADCAST_15 timer-sync flood (aie-rt `XAie_SyncTimer`) reaches distant
/// tiles later by a per-hop delay; distant tiles' timers reset later and read
/// lower -- the cross-domain skew. These constants are NOT in any machine-readable
/// toolchain source (AM020 Ch2 gives the OR-tree routing but no cycle counts);
/// they are silicon-measured (SP-5). All default to 0 -> the model reproduces the
/// current same-cycle flood (behavior-neutral). The intra-tile offsets carry only
/// the same-tile core-vs-mem asymmetry (the add_one `core-memmod=-2` signature);
/// the cross-tile `+2/+4` terms are carried by `origin_D`, not here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BroadcastTiming {
    /// Per-horizontal-hop (east/west) broadcast propagation delay in cycles.
    pub per_hop_horizontal: u8,
    /// Per-vertical-hop (north/south) broadcast propagation delay in cycles.
    pub per_hop_vertical: u8,
    /// Additive offset for a compute tile's core module (relative to the
    /// earliest-resetting module type = the zero baseline; non-negative).
    pub intra_tile_core_offset: u8,
    /// Additive offset for a compute tile's memory module (same baseline).
    pub intra_tile_mem_offset: u8,
}
```

Then add the field to `TimingModel` (after `pub stream_switch: StreamSwitchTiming,` at :1364):

```rust
    pub broadcast: BroadcastTiming,
```

- [ ] **Step 4: Construct it in the NPU1 model builder**

In `crates/xdna-archspec/src/model_builder.rs`, in the `TimingModel { ... }` literal, after `instruction: InstructionTiming { ... },` (:269) and before `source: src.clone(),` (:270), add:

```rust
        broadcast: BroadcastTiming {
            // All zero: SP-1 is behavior-neutral; SP-5 measures the real
            // per-hop and intra-tile values on Phoenix silicon. See the
            // add_one signature (core-memtile=+2, memmod-memtile=+4,
            // core-memmod=-2) -- only the -2 intra-tile term lives here.
            per_hop_horizontal: 0,
            per_hop_vertical: 0,
            intra_tile_core_offset: 0,
            intra_tile_mem_offset: 0,
        },
```

Ensure `BroadcastTiming` is in scope (it is via the existing `use` of the timing types in this module; if not, add it to the `use crate::types::{...}` import).

- [ ] **Step 5: Emit the consts in build.rs**

In `crates/xdna-archspec/build.rs`, after the `INTER_TILE_HOP_LATENCY` emission block (ends at :553) and before the timing module's closing brace `writeln!(out, "}}\n").unwrap();` (:555), add:

```rust
        writeln!(out).unwrap();
        writeln!(out, "    // Broadcast timer-reset propagation (SP-1; 0 = behavior-neutral).").unwrap();
        writeln!(
            out,
            "    pub const BROADCAST_PER_HOP_HORIZONTAL: u8 = {};",
            t.broadcast.per_hop_horizontal
        )
        .unwrap();
        writeln!(
            out,
            "    pub const BROADCAST_PER_HOP_VERTICAL: u8 = {};",
            t.broadcast.per_hop_vertical
        )
        .unwrap();
        writeln!(
            out,
            "    pub const BROADCAST_INTRA_TILE_CORE_OFFSET: u8 = {};",
            t.broadcast.intra_tile_core_offset
        )
        .unwrap();
        writeln!(
            out,
            "    pub const BROADCAST_INTRA_TILE_MEM_OFFSET: u8 = {};",
            t.broadcast.intra_tile_mem_offset
        )
        .unwrap();
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cargo test --lib broadcast_timing_consts_default_to_zero`
Expected: PASS (the archspec crate rebuilds its codegen, the consts resolve to 0).

- [ ] **Step 7: Full lib test + commit**

Run: `cargo test --lib`
Expected: PASS, no regressions.

```bash
git add crates/xdna-archspec/src/types.rs crates/xdna-archspec/src/model_builder.rs crates/xdna-archspec/build.rs src/device/state/effects.rs
git commit -m "$(cat <<'EOF'
feat(#140): SP-1 BroadcastTiming archspec constants (all zero)

New per-device BroadcastTiming struct (d_h, d_v, intra-tile core/mem offsets),
emitted as xdna_archspec::aie2::timing::BROADCAST_* consts, all 0 for NPU1 so
the emulator stays behavior-neutral. SP-5 measures real values on silicon.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
)"
```

---

### Task 2: `TileTimer` reset-target latch

Generalize the timer's `pending_reset` so it resets to a configurable *target* value (default 0) on the first tick, instead of always 0. This is the mechanism that lets a tile hold a nonzero skew baseline. Pure `TileTimer` change; default target 0 reproduces today exactly.

**Files:**
- Modify: `src/device/timer.rs` (`TileTimer` struct field, `tick`, `notify_event`, new `notify_event_with_target`, `reset`)
- Test: `src/device/timer.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `TileTimer::notify_event_with_target(&mut self, event_id: u8, reset_target: u64)` -- latches a reset to `reset_target` if `event_id == reset_event()`. Existing `notify_event(event_id)` is unchanged in signature and now delegates with target 0.

- [ ] **Step 1: Write the failing tests**

In `src/device/timer.rs` `mod tests`, add:

```rust
#[test]
fn notify_event_with_target_resets_to_target_on_tick() {
    let mut timer = TileTimer::new();
    timer.write_register(reg::CONTROL, 0x0000_0500); // Reset_Event = 5
    for _ in 0..100 {
        timer.tick();
    }
    assert_eq!(timer.value(), 100);

    // Latch a reset to target 7 (the skew baseline a tile would hold).
    timer.notify_event_with_target(5, 7);
    assert!(timer.pending_reset());

    // First tick consumes the latch -> value = target (not 0).
    timer.tick();
    assert_eq!(timer.value(), 7);
    assert!(!timer.pending_reset());

    // Subsequent ticks count up from the target.
    timer.tick();
    assert_eq!(timer.value(), 8);
}

#[test]
fn notify_event_target_zero_matches_plain_notify() {
    // target 0 is byte-identical to the current notify_event behavior.
    let mut timer = TileTimer::new();
    timer.write_register(reg::CONTROL, 0x0000_0500);
    for _ in 0..50 {
        timer.tick();
    }
    timer.notify_event_with_target(5, 0);
    timer.tick();
    assert_eq!(timer.value(), 0);
}

#[test]
fn reset_clears_pending_target() {
    let mut timer = TileTimer::new();
    timer.write_register(reg::CONTROL, 0x0000_0500);
    timer.notify_event_with_target(5, 9);
    assert!(timer.pending_reset());

    // Explicit reset clears the latched target so the next tick increments.
    timer.reset();
    assert!(!timer.pending_reset());
    timer.tick();
    assert_eq!(timer.value(), 1);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib notify_event_with_target_resets_to_target_on_tick reset_clears_pending_target notify_event_target_zero_matches_plain_notify`
Expected: FAIL to compile -- `notify_event_with_target` not found.

- [ ] **Step 3: Add the `reset_target` field**

In `src/device/timer.rs`, in `struct TileTimer` after the `pending_reset: bool` field (:133), add:

```rust
    /// Value the timer is set to when `pending_reset` is consumed by `tick`.
    /// 0 for an ordinary reset event (current behavior); the broadcast flood
    /// sets it to `max_delay - delay` so a tile reset earlier (smaller delay)
    /// holds a higher baseline -- the modeled cross-domain skew (SP-1). Cleared
    /// to 0 whenever the latch is consumed or the timer is reset.
    reset_target: u64,
```

- [ ] **Step 4: Initialize it in `new()`**

In `TileTimer::new()` (:141-149), add `reset_target: 0,` to the struct literal.

- [ ] **Step 5: Consume the target in `tick()`**

Replace the reset branch of `tick()` (:162-164):

```rust
        if self.pending_reset {
            self.value = 0;
            self.pending_reset = false;
        } else {
```

with:

```rust
        if self.pending_reset {
            self.value = self.reset_target;
            self.pending_reset = false;
            self.reset_target = 0;
        } else {
```

- [ ] **Step 6: Add `notify_event_with_target` and delegate `notify_event`**

Replace `notify_event` (:178-185) with:

```rust
    pub fn notify_event(&mut self, event_id: u8) {
        self.notify_event_with_target(event_id, 0);
    }

    /// Latch a "reset to `reset_target` on next tick" if `event_id` matches the
    /// configured `Reset_Event`. The broadcast flood passes `max_delay - delay`
    /// to encode the cross-domain skew baseline (SP-1); ordinary callers use
    /// `notify_event` (target 0). Event id 0 is the unconfigured sentinel and
    /// never triggers a reset.
    pub fn notify_event_with_target(&mut self, event_id: u8, reset_target: u64) {
        if event_id == 0 {
            return;
        }
        if event_id == self.reset_event() {
            self.pending_reset = true;
            self.reset_target = reset_target;
        }
    }
```

- [ ] **Step 7: Clear the target in `reset()`**

In `reset()` (:233-237), add `self.reset_target = 0;` alongside the existing `self.pending_reset = false;`.

- [ ] **Step 8: Run the new + existing timer tests**

Run: `cargo test --lib --  device::timer`
Expected: PASS, including the three new tests AND every existing timer test (e.g. `pending_reset_cleared_by_tick_and_resets_value`, `sync_timer_protocol_aligns_independent_timers`) unchanged.

- [ ] **Step 9: Commit**

```bash
git add src/device/timer.rs
git commit -m "$(cat <<'EOF'
feat(#140): SP-1 TileTimer reset-target latch

Generalize pending_reset to reset to a configurable target value (default 0)
on the first tick. notify_event delegates with target 0 (byte-identical to
today); notify_event_with_target lets the broadcast flood latch the skew
baseline. reset()/control-bit-31 clear the target. Establishes the constant
baseline at the first execution tick -- immune to the config-time-no-ticks
and clock-gating issues a tick-driven countdown would have hit.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
)"
```

---

### Task 3: De-duplicate broadcast event-ID bases

Remove the local `CORE_/SHIM_PL_/MEMTILE_BROADCAST_BASE` consts in `effects.rs` and resolve the per-module base from the single `EventModuleType::broadcast_event_base` accessor. Behavior-neutral (same values: 107/110/142).

**Files:**
- Modify: `src/device/state/effects.rs:480-482` (remove consts) and `:517-519`, `:540` (use the accessor)
- Test: `src/device/state/effects.rs` (assert resolved IDs match the accessor)

**Interfaces:**
- Consumes: `xdna_archspec`/`events` -- `crate::device::events::EventModuleType::broadcast_event_base(self) -> u8` (Core 107, Memory 107, Pl 110, MemTile 142). `EventId = u8`, so `base + channel` stays `u8`.

- [ ] **Step 1: Write the failing test**

In `src/device/state/effects.rs` tests, add:

```rust
#[test]
fn broadcast_bases_resolve_from_event_module_accessor() {
    use crate::device::events::EventModuleType;
    // The de-duped flood must resolve the same per-module bases the single
    // accessor holds (aie-rt xaie_events_aieml.h).
    assert_eq!(EventModuleType::Core.broadcast_event_base(), 107);
    assert_eq!(EventModuleType::Memory.broadcast_event_base(), 107);
    assert_eq!(EventModuleType::Pl.broadcast_event_base(), 110);
    assert_eq!(EventModuleType::MemTile.broadcast_event_base(), 142);
}
```

(This test passes immediately -- it pins the accessor values. The real check is Step 3's edit compiling and the full suite staying green; the test guards against a future drift between flood and accessor.)

- [ ] **Step 2: Run it (guard test, should pass)**

Run: `cargo test --lib broadcast_bases_resolve_from_event_module_accessor`
Expected: PASS.

- [ ] **Step 3: Replace the hardcoded bases with the accessor**

In `propagate_broadcasts` (`src/device/state/effects.rs`):

(a) Delete the three `const ... _BROADCAST_BASE` lines (:480-482) and their comment block (:472-479) is kept but reworded to point at the accessor:

```rust
        // Per-module BROADCAST_N event id = EventModuleType::broadcast_event_base
        // + channel (aie-rt xaie_events_aieml.h, via the single events accessor):
        //   Core/Mem 107, Shim PL 110, MemTile 142.
        // hw_id 0 is the EVENT_NONE sentinel; notify_*_trace_event filters it for
        // tile kinds lacking that module side.
        use crate::device::events::EventModuleType;
```

(b) Replace the match at :516-520 with:

```rust
                    let (core_hw_id, mem_hw_id) = match tile.tile_kind {
                        TileKind::Compute => (
                            EventModuleType::Core.broadcast_event_base() + channel,
                            EventModuleType::Memory.broadcast_event_base() + channel,
                        ),
                        TileKind::ShimNoc | TileKind::ShimPl => {
                            (EventModuleType::Pl.broadcast_event_base() + channel, 0)
                        }
                        TileKind::Mem => (0, EventModuleType::MemTile.broadcast_event_base() + channel),
                    };
```

(c) Replace the L1 tap at :540 (`let ev = SHIM_PL_BROADCAST_BASE + channel;`) with:

```rust
                        let ev = EventModuleType::Pl.broadcast_event_base() + channel;
```

- [ ] **Step 4: Build + run the full lib suite**

Run: `cargo test --lib`
Expected: PASS, no regressions (values unchanged; this is a pure refactor).

- [ ] **Step 5: Commit**

```bash
git add src/device/state/effects.rs
git commit -m "$(cat <<'EOF'
refactor(#140): SP-1 de-dup broadcast event-ID bases

Resolve the per-module BROADCAST_0 base from the single
EventModuleType::broadcast_event_base accessor (aie-rt xaie_events_aieml.h)
across all three effects.rs sites; remove the duplicated local consts. Values
unchanged (107/110/142). Full build-time derivation from the header is a
deferred follow-up.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
)"
```

---

### Task 4: Dijkstra wavefront (behavior-neutral flood refactor)

Replace the LIFO BFS in `propagate_broadcasts` with a min-cost Dijkstra wavefront that computes each reached tile's `origin_D`, then delivers to the reached set exactly as today (at `current_cycle`, no target yet). With `d_h = d_v = 0` every `origin_D = 0` and the reached set is identical, so this is behavior-neutral. The `origin_D` map is computed by a pure helper so it is unit-testable with explicit `d_h != d_v`.

**Files:**
- Modify: `src/device/state/effects.rs` (`propagate_broadcasts`: extract a `broadcast_origin_d` helper, restructure the per-channel walk into compute-then-deliver)
- Test: `src/device/state/effects.rs` (wavefront timing with explicit `d_h`/`d_v`; reached-set identity at `d=0`)

**Interfaces:**
- Consumes: `xdna_archspec::aie2::timing::{BROADCAST_PER_HOP_HORIZONTAL, BROADCAST_PER_HOP_VERTICAL}` (Task 1).
- Produces: `DeviceState::broadcast_origin_d(&self, col: u8, source_row: u8, channel: u8, d_h: u32, d_v: u32) -> Vec<(u8, u8, u32)>` -- for the given channel's broadcast adjacency (block masks honored), returns `(col, row, origin_D)` for every reached tile, `origin_D` = min cumulative cost from the source (edge cost `d_h` horizontal, `d_v` vertical). Source has `origin_D = 0`.

- [ ] **Step 1: Write the failing tests**

In `src/device/state/effects.rs` tests (these need a built `DeviceState`; reuse the existing test-construction pattern in `mod interrupt_path_tests`):

```rust
#[test]
fn broadcast_origin_d_weighted_manhattan_unblocked() {
    // Fresh NPU1: no CDO broadcast block config -> fully connected 5x6 grid.
    let dev = DeviceState::new_npu1();
    let (src_col, src_row) = (0u8, 2u8); // a compute-row source
    let d_h = 2u32;
    let d_v = 3u32;
    let map = dev.broadcast_origin_d(src_col, src_row, 0, d_h, d_v);

    // Source is 0; every tile dc cols, dr rows away costs dc*d_h + dr*d_v
    // (weighted Manhattan, valid because the unblocked grid lets you move
    // monotonically toward any target with non-negative per-axis weights).
    for &(c, r, o) in &map {
        let dc = (c as i32 - src_col as i32).unsigned_abs();
        let dr = (r as i32 - src_row as i32).unsigned_abs();
        assert_eq!(o, dc * d_h + dr * d_v, "origin_D at ({c},{r})");
    }
    assert!(map.iter().any(|&(c, r, o)| c == src_col && r == src_row && o == 0));
}

#[test]
fn broadcast_origin_d_reached_set_all_zero_at_zero_delays() {
    // At d=0 every origin_D is 0; connectivity (reached set) is unchanged --
    // source and the far corner are both reached. (Full reach-equivalence vs.
    // the legacy flood is additionally guarded by interrupt_path_tests.)
    let dev = DeviceState::new_npu1();
    let map = dev.broadcast_origin_d(0, 2, 0, 0, 0);
    assert!(map.iter().all(|&(_, _, o)| o == 0), "d=0 -> all origin_D == 0");
    assert!(map.iter().any(|&(c, r, _)| c == 0 && r == 2), "source reached");
    let (fc, fr) = (dev.array.cols() - 1, dev.array.rows() - 1);
    assert!(map.iter().any(|&(c, r, _)| c == fc && r == fr), "far corner reached");
}
```

(`new_npu1()` is the real constructor used throughout `interrupt_path_tests`. The Manhattan assertion assumes the NPU1 array is a full 5x6 grid with default-unblocked broadcast config -- it is; if a future fixture masks a direction, relax to asserting only monotonic-reachable tiles.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib broadcast_origin_d_weighted_manhattan_unblocked broadcast_origin_d_reached_set_matches_flood_at_zero`
Expected: FAIL to compile -- `broadcast_origin_d` not found.

- [ ] **Step 3: Add the `broadcast_origin_d` helper**

In `src/device/state/effects.rs` `impl DeviceState`, add (near `propagate_broadcasts`):

```rust
    /// Dijkstra wavefront over the broadcast adjacency for one channel.
    ///
    /// Returns `(col, row, origin_D)` for every reached tile, where `origin_D`
    /// is the minimum cumulative propagation delay from the source (edge cost
    /// `d_h` for an east/west hop, `d_v` for north/south). The OR-tree
    /// re-broadcasts on first arrival, so earliest arrival wins = shortest path
    /// (AM020 Ch2). Honors per-tile broadcast block masks (a blocked direction
    /// is a removed edge). Source `origin_D = 0`. With `d_h = d_v = 0` every
    /// `origin_D = 0` and the reached set equals the legacy flood's reach.
    pub(crate) fn broadcast_origin_d(
        &self,
        col: u8,
        source_row: u8,
        channel: u8,
        d_h: u32,
        d_v: u32,
    ) -> Vec<(u8, u8, u32)> {
        use crate::device::events::broadcast::BroadcastDir;
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let cols = self.array.cols();
        let rows = self.array.rows();
        let idx_of = |c: u8, r: u8| (c as usize) * (rows as usize) + (r as usize);

        // best[idx] = settled min cost, or u32::MAX if unreached.
        let mut best = vec![u32::MAX; cols as usize * rows as usize];
        // Min-heap on (cost, col, row). Reverse for min-first.
        let mut heap: BinaryHeap<Reverse<(u32, u8, u8)>> = BinaryHeap::new();

        best[idx_of(col, source_row)] = 0;
        heap.push(Reverse((0, col, source_row)));

        let mut out: Vec<(u8, u8, u32)> = Vec::new();

        while let Some(Reverse((cost, c, r))) = heap.pop() {
            if cost > best[idx_of(c, r)] {
                continue; // stale heap entry
            }
            out.push((c, r, cost));

            // Allowed outbound directions from THIS tile (source side of the hop).
            let bcfg = self
                .array
                .get(c, r)
                .and_then(|t| t.core_events.as_ref().or(t.mem_events.as_ref()))
                .map(|m| &m.broadcast);
            let dirs = match bcfg {
                Some(b) => b.allowed_directions(channel as usize),
                None => BroadcastDir::ALL.to_vec(),
            };

            for dir in dirs {
                let (nc, nr, step) = match dir {
                    BroadcastDir::South if r > 0 => (c, r - 1, d_v),
                    BroadcastDir::North if r + 1 < rows => (c, r + 1, d_v),
                    BroadcastDir::East if c + 1 < cols => (c + 1, r, d_h),
                    BroadcastDir::West if c > 0 => (c - 1, r, d_h),
                    _ => continue,
                };
                let ncost = cost + step;
                let nidx = idx_of(nc, nr);
                if ncost < best[nidx] {
                    best[nidx] = ncost;
                    heap.push(Reverse((ncost, nc, nr)));
                }
            }
        }
        out
    }
```

- [ ] **Step 4: Run the helper tests**

Run: `cargo test --lib broadcast_origin_d_weighted_manhattan_unblocked broadcast_origin_d_reached_set_matches_flood_at_zero`
Expected: PASS.

- [ ] **Step 5: Restructure `propagate_broadcasts` to use the helper (behavior-neutral)**

Replace the per-channel walk (the `let mut visited ...` through the closing of the `while let Some((c, r)) = frontier.pop()` loop, :496-596) so the body of `for &channel in &channels` becomes: compute the reached set via the helper, then deliver to each reached tile with the same per-tile delivery code as today (notify core/mem trace event, L2 signal, L1 tap), still at `current_cycle` and with no reset target yet. Read the constants but do not apply them to the timer in this task:

```rust
            let d_h = xdna_archspec::aie2::timing::BROADCAST_PER_HOP_HORIZONTAL as u32;
            let d_v = xdna_archspec::aie2::timing::BROADCAST_PER_HOP_VERTICAL as u32;
            let reached = self.broadcast_origin_d(col, source_row, channel, d_h, d_v);

            for (c, r, _origin_d) in reached {
                let tile = match self.array.get_mut(c, r) {
                    Some(t) => t,
                    None => continue,
                };
                let core_pc = Some(tile.core.pc);
                let (core_hw_id, mem_hw_id) = match tile.tile_kind {
                    TileKind::Compute => (
                        EventModuleType::Core.broadcast_event_base() + channel,
                        EventModuleType::Memory.broadcast_event_base() + channel,
                    ),
                    TileKind::ShimNoc | TileKind::ShimPl => {
                        (EventModuleType::Pl.broadcast_event_base() + channel, 0)
                    }
                    TileKind::Mem => (0, EventModuleType::MemTile.broadcast_event_base() + channel),
                };
                tile.notify_core_trace_event(core_hw_id, current_cycle, core_pc);
                tile.notify_mem_trace_event(mem_hw_id, current_cycle, None);
                if let Some(ref mut l2) = tile.l2_irq {
                    l2.signal_interrupt(channel);
                }
                if tile.l1_irq.is_some() {
                    let ev = EventModuleType::Pl.broadcast_event_base() + channel;
                    tile.tap_l1_interrupt(ev);
                }
            }
```

Note: this folds in Task 3's accessor edits (the match + L1 tap now live in the delivery loop). The outbound-direction computation that the old walk did inline is now inside `broadcast_origin_d`; do not duplicate it. The `use crate::device::events::EventModuleType;` from Task 3 stays at the top of the function.

- [ ] **Step 6: Run the full lib suite + behavior-neutral check**

Run: `cargo test --lib`
Expected: PASS. Pay attention to the broadcast/interrupt path tests in `mod interrupt_path_tests` (the L1->L2 fixpoint) -- they must stay green, proving the reached-set and interrupt delivery are unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/device/state/effects.rs
git commit -m "$(cat <<'EOF'
refactor(#140): SP-1 Dijkstra broadcast wavefront (behavior-neutral)

Replace the LIFO BFS in propagate_broadcasts with a min-cost Dijkstra wavefront
(broadcast_origin_d) honoring per-channel block masks, then deliver to the
reached set as before (current_cycle, no reset target yet). At d_h=d_v=0 every
origin_D=0 and the reached set is identical -- byte-neutral. The origin_D map is
a pure helper, unit-tested with explicit d_h!=d_v, and is reused by SP-2/SP-4b.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
)"
```

---

### Task 5: Wire delay into the reset target (mechanism active)

Compute each reached module's `delay = origin_D + intra_tile_offset`, find `max_delay` over the reached set, and latch each timer's reset target to `max_delay - delay` via a new target-carrying trace-notify path. Under the zero defaults this is byte-identical; under explicit constants it makes tiles hold the constant skew baseline.

**Files:**
- Modify: `src/device/tile/mod.rs` (add `notify_core_trace_event_with_target` / `notify_mem_trace_event_with_target`; the existing `notify_core_trace_event` / `notify_mem_trace_event` delegate with target 0)
- Modify: `src/device/state/effects.rs` (`propagate_broadcasts` delivery loop: compute `delay`/`max_delay`, deliver with target)
- Test: `src/device/state/effects.rs` (constant baseline through the flood; intra-tile offset; behavior-neutral; shim self-reset)

**Interfaces:**
- Consumes: `TileTimer::notify_event_with_target` (Task 2); `broadcast_origin_d` (Task 4); `xdna_archspec::aie2::timing::{BROADCAST_INTRA_TILE_CORE_OFFSET, BROADCAST_INTRA_TILE_MEM_OFFSET}` (Task 1).
- Produces: `Tile::notify_core_trace_event_with_target(&mut self, hw_id: u8, cycle: u64, pc: Option<u32>, reset_target: u64)` and the `_mem_` analogue. The existing `notify_core_trace_event` / `notify_mem_trace_event` delegate to these with `reset_target = 0`.

- [ ] **Step 1: Write the failing test**

In `src/device/state/effects.rs` tests:

```rust
#[test]
fn flood_sets_constant_skew_baseline_under_explicit_delays() {
    // Drive the flood with explicit d_v and assert two modules hold a constant
    // offset == their delay difference from the first tick. The injection seam
    // is propagate_broadcasts_with_timing (Step 4), which tests call with
    // explicit values since the shipped consts are zero.
    let mut dev = DeviceState::new_npu1();
    let channel = 5u8;
    let bcast_id = EventModuleType::Core.broadcast_event_base() + channel; // 112
    let src = (0u8, 2u8); // compute-row source
    let hop = (0u8, 3u8); // one vertical hop north
    // Configure both core timers to auto-reset on the broadcast event
    // (Timer_Control offset 0x000, Reset_Event in bits [14:8]).
    for &(c, r) in &[src, hop] {
        dev.array.get_mut(c, r).unwrap().core_timer.write_register(0x000, (bcast_id as u32) << 8);
    }
    dev.array.get_mut(src.0, src.1).unwrap().pending_broadcasts.push(channel);
    // d_v = 4: one vertical hop = 4 cy of skew. d_h, offsets = 0.
    dev.propagate_broadcasts_with_timing(src.0, src.1, 0, 4, 0, 0);
    // One tick each consumes the latch -> value = target.
    dev.array.get_mut(src.0, src.1).unwrap().core_timer.tick();
    dev.array.get_mut(hop.0, hop.1).unwrap().core_timer.tick();
    let v_src = dev.array.get(src.0, src.1).unwrap().core_timer.value();
    let v_hop = dev.array.get(hop.0, hop.1).unwrap().core_timer.value();
    // Source reset earlier (delay 0) -> higher baseline by exactly one hop.
    // (v_src - v_hop = (max_delay - 0) - (max_delay - 4) = 4, independent of max_delay.)
    assert_eq!(v_src - v_hop, 4, "one-vertical-hop skew == d_v");
}

#[test]
fn flood_is_behavior_neutral_at_zero_delays() {
    // Shipped (zero) consts: every reached timer resets to 0 on the first tick,
    // exactly as before this change, regardless of prior value.
    let mut dev = DeviceState::new_npu1();
    let channel = 5u8;
    let bcast_id = EventModuleType::Core.broadcast_event_base() + channel;
    let tiles = [(0u8, 2u8), (0u8, 3u8), (1u8, 2u8)];
    for (i, &(c, r)) in tiles.iter().enumerate() {
        dev.array.get_mut(c, r).unwrap().core_timer.write_register(0x000, (bcast_id as u32) << 8);
        for _ in 0..(i * 10 + 5) {
            dev.array.get_mut(c, r).unwrap().core_timer.tick(); // diverge the timers
        }
    }
    dev.array.get_mut(0, 2).unwrap().pending_broadcasts.push(channel);
    dev.propagate_broadcasts(0, 2); // shipped consts = all zero
    for &(c, r) in &tiles {
        dev.array.get_mut(c, r).unwrap().core_timer.tick();
        assert_eq!(dev.array.get(c, r).unwrap().core_timer.value(), 0, "zero consts -> reset to 0");
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib flood_sets_constant_skew_baseline_under_explicit_delays flood_is_behavior_neutral_at_zero_delays`
Expected: FAIL to compile -- `propagate_broadcasts_with_timing` / target methods not found.

- [ ] **Step 3: Add the target-carrying trace-notify methods**

In `src/device/tile/mod.rs`, make the existing `notify_core_trace_event` (:798) delegate, and add the target variant:

```rust
    pub fn notify_core_trace_event(&mut self, hw_id: u8, cycle: u64, pc: Option<u32>) {
        self.notify_core_trace_event_with_target(hw_id, cycle, pc, 0);
    }

    /// As `notify_core_trace_event`, but latches the core timer's reset to
    /// `reset_target` (SP-1 broadcast skew baseline = `max_delay - delay`)
    /// instead of 0. All non-flood callers use the target-0 wrapper above.
    #[inline]
    pub fn notify_core_trace_event_with_target(
        &mut self,
        hw_id: u8,
        cycle: u64,
        pc: Option<u32>,
        reset_target: u64,
    ) {
        if hw_id == 0 {
            return;
        }
        self.core_trace.notify_event(hw_id, cycle, pc);
        self.core_timer.notify_event_with_target(hw_id, reset_target);
        for det in &mut self.core_edge_detectors {
            if det.input_event != 0 && det.input_event == hw_id {
                det.curr_active = true;
            }
        }
        self.core_debug.check_event_halt(hw_id);
    }
```

Do the analogous split for `notify_mem_trace_event` (:857): keep the public name delegating with target 0, add `notify_mem_trace_event_with_target(hw_id, cycle, pc, reset_target)` that calls `self.mem_timer.notify_event_with_target(hw_id, reset_target)` and otherwise mirrors the existing body (mem_trace.notify_event, mem_edge_detectors; no debug halt -- match the current `notify_mem_trace_event` body exactly).

- [ ] **Step 4: Compute delay + target in the flood delivery**

In `src/device/state/effects.rs`, factor `propagate_broadcasts` so the public method reads the consts and delegates:

```rust
    pub(crate) fn propagate_broadcasts(&mut self, col: u8, source_row: u8) {
        let d_h = xdna_archspec::aie2::timing::BROADCAST_PER_HOP_HORIZONTAL as u32;
        let d_v = xdna_archspec::aie2::timing::BROADCAST_PER_HOP_VERTICAL as u32;
        let core_off = xdna_archspec::aie2::timing::BROADCAST_INTRA_TILE_CORE_OFFSET as u32;
        let mem_off = xdna_archspec::aie2::timing::BROADCAST_INTRA_TILE_MEM_OFFSET as u32;
        self.propagate_broadcasts_with_timing(col, source_row, d_h, d_v, core_off, mem_off);
    }

    pub(crate) fn propagate_broadcasts_with_timing(
        &mut self,
        col: u8,
        source_row: u8,
        d_h: u32,
        d_v: u32,
        core_off: u32,
        mem_off: u32,
    ) {
        let current_cycle = self.array.current_cycle;
        let channels = if let Some(tile) = self.array.get_mut(col, source_row) {
            tile.drain_pending_broadcasts()
        } else {
            return;
        };
        if channels.is_empty() {
            return;
        }
        use crate::device::events::EventModuleType;
        for &channel in &channels {
            let reached = self.broadcast_origin_d(col, source_row, channel, d_h, d_v);
            // max_delay over all reached modules (core uses core_off, mem uses mem_off).
            let max_delay = reached
                .iter()
                .map(|&(_, _, o)| o + core_off.max(mem_off))
                .max()
                .unwrap_or(0);
            for (c, r, origin_d) in reached {
                let core_delay = origin_d + core_off;
                let mem_delay = origin_d + mem_off;
                let core_target = (max_delay - core_delay) as u64;
                let mem_target = (max_delay - mem_delay) as u64;
                let tile = match self.array.get_mut(c, r) {
                    Some(t) => t,
                    None => continue,
                };
                let core_pc = Some(tile.core.pc);
                let (core_hw_id, mem_hw_id) = match tile.tile_kind {
                    TileKind::Compute => (
                        EventModuleType::Core.broadcast_event_base() + channel,
                        EventModuleType::Memory.broadcast_event_base() + channel,
                    ),
                    TileKind::ShimNoc | TileKind::ShimPl => {
                        (EventModuleType::Pl.broadcast_event_base() + channel, 0)
                    }
                    TileKind::Mem => (0, EventModuleType::MemTile.broadcast_event_base() + channel),
                };
                tile.notify_core_trace_event_with_target(core_hw_id, current_cycle, core_pc, core_target);
                tile.notify_mem_trace_event_with_target(mem_hw_id, current_cycle, None, mem_target);
                if let Some(ref mut l2) = tile.l2_irq {
                    l2.signal_interrupt(channel);
                }
                if tile.l1_irq.is_some() {
                    let ev = EventModuleType::Pl.broadcast_event_base() + channel;
                    tile.tap_l1_interrupt(ev);
                }
            }
        }
    }
```

`max_delay` uses `core_off.max(mem_off)` for the per-tile worst case so both `max_delay - core_delay` and `max_delay - mem_delay` stay non-negative (the baseline invariant from the spec; assert non-negativity holds for the configured constants). At zero consts, `max_delay = 0`, every target = 0 -> identical to Task 4.

- [ ] **Step 5: Run the new tests + full suite**

Run: `cargo test --lib`
Expected: PASS, including the two new flood tests and every existing test (timer, interrupt path, broadcast).

- [ ] **Step 6: Behavior-neutral regression -- trace sweep baseline diff**

Per `feedback_verify_against_baseline_trace`, do not trust bridge pass/fail. Capture a baseline before this branch and diff after:

Run (one chess-only sweep is enough; the change is trace-relevant): `./scripts/emu-bridge-test.sh --chess-only --sweep -v add_one_using_dma 2>&1 | tee /tmp/claude-1000/sp1-sweep.log` and diff the resulting `trace.log` against a pre-SP-1 baseline.
Expected: zero divergence (all constants are 0; timer baselines are 0; trace timestamps unchanged -- SP-1 does not touch trace Start-frame origins, that is SP-2).

- [ ] **Step 7: Commit**

```bash
git add src/device/tile/mod.rs src/device/state/effects.rs
git commit -m "$(cat <<'EOF'
feat(#140): SP-1 wire broadcast delay into the timer reset target

Compute delay = origin_D + intra_tile_offset per module, find max_delay over the
reached set, and latch each timer's reset target = max_delay - delay via new
target-carrying trace-notify methods (the plain notify_*_trace_event delegate
with target 0). Under the zero consts this is byte-identical; under explicit
constants a tile reset earlier holds a higher baseline -- the modeled cross-domain
skew, present as a constant from execution cycle 0. Completes SP-1.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
)"
```

---

## Notes for the executor

- **Route-1 hazards (spec 3.6) are satisfied by construction, not separate tasks.** (i) clock-gating: the reset-target latch is consumed by the first `tick()` exactly like today's `pending_reset`, so SP-1 introduces no new tick dependence; (ii) flood-before-first-event: the latch persists config->execution and is consumed before that cycle's trace notifies; (iii) shim self-reset: `broadcast_origin_d` includes the source (cost 0), so its own timer is delivered. If a test reveals any of these regressed, stop and revisit -- do not paper over.
- **Coexistence (spec test 8):** if `propagate_broadcasts_with_timing`'s fixpoint re-entry delivers to a tile twice, the later `notify_event_with_target` wins (last-write). That is acceptable for SP-1 (all targets 0). If SP-5 needs min-delay-wins, revisit then.
- **Test fixture:** `DeviceState::new_npu1()` is the real constructor (used throughout `mod interrupt_path_tests`). The flood is driven by `array.get_mut(c, r).unwrap().pending_broadcasts.push(channel)` then `propagate_broadcasts(c, r)`; a timer's auto-reset event is set via `core_timer.write_register(0x000, (event_id as u32) << 8)`; tick/read via `core_timer.tick()` / `.value()`. The plan's test code uses these real APIs.
- **Do not touch trace Start-frame origins.** That is SP-2. SP-1 changes only the timer *value* baseline; trace event timestamps stay at `current_cycle`.
