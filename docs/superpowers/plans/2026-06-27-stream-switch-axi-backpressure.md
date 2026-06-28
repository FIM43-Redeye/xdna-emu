# Stream-Switch AXI4-Stream Backpressure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a lock-stalled stream consumer correctly backpressure its producer across tiles, so the memtile MM2S send-port trace cadence tracks hardware, by bounding the inter-tile wire to its AM020-documented depth.

**Architecture:** The stream fabric is already a pipelined AXI4-Stream ready/valid network with bounded intra-tile crossings whose latencies (3/4) and FIFO depths (slave 4 / local master 2 / external master 4) match AM020 exactly (fact-checked). The one structural defect is the inter-tile wire (`inter_tile_pipeline` in `routing.rs`): its admission check counts only the destination slave's FIFO occupancy, ignoring words already in flight toward that slave, so it over-absorbs ~`ROUTE_PER_HOP` (4) words and breaks the backpressure chain. This is an audit/tighten of that one crossing, plus an empirical decision gate on whether the consume-before-produce phase ordering is a co-cause, plus a verify pass on the intra-tile local-master budget. Full design: `docs/superpowers/specs/2026-06-27-stream-switch-axi-backpressure-design.md`.

**Tech Stack:** Rust (the emulator), `cargo test --lib`, the bridge trace pipeline (`scripts/emu-bridge-test.sh`, `tools/trace-port-spans.py`), AM020 (`docs/xdna/am020-aie-ml/`).

## Global Constraints

- **Derive from the toolchain.** AM020 ch2 is the source for crossing latency + FIFO depth (verbatim values in the spec). NPU1.json has NO per-crossing fields; do not invent. HW is the validation oracle.
- **`cargo test --lib` after every code change.** Baseline before starting: run it and record the pass count (expected ~3546 passing, 0 failing). A pass that regresses is a regression to fix before moving on.
- **Push is HELD** pending Maya's explicit say-so. Commit locally; do not push.
- **No emoji.** End every commit message with:
  `Generated using Claude Code.`
  `Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh`
- **Rebuild the FFI `.so` before EMU captures** (`cargo build -p xdna-emu-ffi`); `emu-bridge-test.sh` does this automatically.
- **Never run two HW test suites concurrently.** A single HW capture is cheap; the full sweep is slow (run once per batch, not to "check progress").
- **Branch:** `device-model-audit` (the current working branch).

---

### Task 1: Bound the inter-tile wire to count in-flight words

**Files:**
- Modify: `src/device/array/routing.rs` — `propagate_inter_tile()` admission checks (the four `slaves[...].can_accept()` calls in the N/S/E/W transfer-collection loops, around lines 1050, 1109, and the two E/W equivalents) and `advance_inter_tile_pipeline()` delivery check (line ~1321); add a private helper.
- Test: `src/device/array/tests.rs` (the existing stream-routing test module).

**Interfaces:**
- Produces: `fn inflight_to(&self, dst_idx: usize, dst_slave: usize) -> usize` on `TileArray` (or the routing impl block) — counts words in `inter_tile_pipeline` already targeting `(dst_idx, dst_slave)`.
- Produces: a bounded admission predicate used in place of bare `can_accept()`: a destination slave admits a new inter-tile word only if `slave.fifo.len() + inflight_to(dst_idx, dst_slave) < slave.fifo_capacity()`.

- [ ] **Step 1: Read the surrounding code.** Read `src/device/array/routing.rs` lines 1000-1340 (`propagate_inter_tile`, `advance_inter_tile_pipeline`, the `InFlightWord` fields `dst_tile_idx`/`dst_slave_idx`/`cycles_remaining`) and `src/device/stream_switch/ports.rs` `can_accept()` and `fifo_capacity` accessor. Read 2-3 existing stream-routing tests in `src/device/array/tests.rs` to learn the setup API (`TileArray::npu1()`, configuring a circuit route, pushing into a master, calling `step_data_movement`/`route_streams`, inspecting `stream_switch.masters[i].fifo.len()`).

- [ ] **Step 2: Write the failing test.**

Add to `src/device/array/tests.rs`. The test sets up an inter-tile circuit route, makes the destination unable to drain (do NOT configure the destination's onward route / DMA, so its slave FIFO fills and stays full), pumps more words than the documented crossing depth from the source master, and asserts the source side backpressures — i.e. the total words admitted downstream (destination slave FIFO occupancy + in-flight pipeline words toward it) never exceeds the destination slave's capacity, and the source master retains the overflow.

```rust
#[test]
fn inter_tile_wire_backpressures_source_when_dest_cannot_drain() {
    // A lock-stalled / unrouted destination slave must not absorb more than its
    // documented FIFO depth (incl. in-flight wire words). Without bounding the
    // inter_tile_pipeline, the source master drains completely and the pipeline
    // over-absorbs ~ROUTE_PER_HOP extra words (the bug). AM020 ch2: external
    // slave port is 4-deep.
    let mut array = TileArray::npu1();
    array.clock_mut().ungate_all();

    // Configure a south-bound circuit route from a compute tile master to the
    // mem tile slave below it, but DO NOT give the mem tile any onward route or
    // DMA drain -- its receiving slave FIFO will fill and stay full.
    // (Use the same route-config API the other stream tests in this file use;
    // confirm the exact port indices from compute::SOUTH_MASTER_START /
    // mem_tile::NORTH_SLAVE_START.)
    let (src_col, src_row) = (0u8, 2u8); // compute
    let src_master = /* compute::SOUTH_MASTER_START */ ;
    configure_inter_tile_circuit_route(&mut array, src_col, src_row, src_master);

    // Seed the source master with more words than the crossing can hold.
    for w in 0..32u32 {
        array.tile_mut(src_col, src_row).stream_switch.masters[src_master]
            .push_with_tlast(w, false);
    }

    let dst_idx = array.tile_index(src_col, src_row - 1);
    let dst_slave = /* mem_tile::NORTH_SLAVE_START */ ;
    let cap = array.tiles[dst_idx].stream_switch.slaves[dst_slave].fifo_capacity();

    // Run many cycles; the destination never drains.
    let mut host = HostMemory::new_for_test();
    for _ in 0..64 {
        array.step_data_movement(&mut host);
        let buffered = array.tiles[dst_idx].stream_switch.slaves[dst_slave].fifo.len();
        let inflight = array.inflight_to(dst_idx, dst_slave);
        assert!(
            buffered + inflight <= cap,
            "inter-tile crossing over-absorbed: buffered={} inflight={} cap={}",
            buffered, inflight, cap
        );
    }

    // And the source must still hold the un-admitted overflow (backpressure).
    assert!(
        array.tile(src_col, src_row).stream_switch.masters[src_master].fifo.len() > 0,
        "source master fully drained -- backpressure did not hold"
    );
}
```

If a `configure_inter_tile_circuit_route` helper or `HostMemory::new_for_test` does not already exist, replace those lines with the equivalent setup used by the neighbouring tests in the same file (adapt to the real API you read in Step 1).

- [ ] **Step 3: Run the test, verify it fails.**

Run: `cargo test --lib inter_tile_wire_backpressures_source_when_dest_cannot_drain`
Expected: FAIL on the `buffered + inflight <= cap` assertion (current code over-absorbs ~`ROUTE_PER_HOP` words), or a compile error on `inflight_to` (not yet defined) — write the helper in Step 4 either way.

- [ ] **Step 4: Implement the bounded admission.**

Add the helper near `advance_inter_tile_pipeline` in `routing.rs`:

```rust
/// Count words already in flight in the inter-tile pipeline that target a
/// given destination slave. Used so inter-tile admission accounts for words
/// that will land in the slave FIFO when they finish traversal -- otherwise
/// the crossing over-absorbs up to ROUTE_PER_HOP words past the documented
/// AM020 external-slave depth, defeating backpressure.
fn inflight_to(&self, dst_idx: usize, dst_slave: usize) -> usize {
    self.inter_tile_pipeline
        .iter()
        .filter(|w| w.dst_tile_idx == dst_idx && w.dst_slave_idx == dst_slave)
        .count()
}
```

In `propagate_inter_tile`, replace each transfer-collection admission check of the form

```rust
if slave_idx < self.tiles[DST].stream_switch.slaves.len()
    && self.tiles[DST].stream_switch.slaves[slave_idx].can_accept()
{
```

with one that also counts in-flight words (DST is `above_idx` / `below_idx` / the E/W destination index; compute the destination tile index once as `dst_idx`):

```rust
let dst_idx = self.tile_index(/* dst col */, /* dst row */);
if slave_idx < self.tiles[dst_idx].stream_switch.slaves.len() && {
    let slave = &self.tiles[dst_idx].stream_switch.slaves[slave_idx];
    slave.fifo.len() + self.inflight_to(dst_idx, slave_idx) < slave.fifo_capacity()
} {
```

Apply the same bound to the delivery check in `advance_inter_tile_pipeline` (line ~1321): a word at `cycles_remaining == 0` delivers only if `slave.fifo.len() < slave.fifo_capacity()` (delivery consumes one in-flight slot and adds one FIFO slot, so the simple occupancy check is correct there — the admission-side `inflight_to` is what enforces the total bound). Keep the existing "stays in pipeline (backpressure)" else-branch.

Confirm `fifo_capacity()` (or the field) is accessible on the slave port; if it is a private field, add a small accessor on the port type in `ports.rs`.

- [ ] **Step 5: Run the test, verify it passes.**

Run: `cargo test --lib inter_tile_wire_backpressures_source_when_dest_cannot_drain`
Expected: PASS.

- [ ] **Step 6: Run the full library suite, verify no regression.**

Run: `cargo test --lib`
Expected: pass count >= the Step-0 baseline, 0 failing. If anything regresses, fix before committing.

- [ ] **Step 7: Commit.**

```bash
git add src/device/array/routing.rs src/device/array/tests.rs src/device/stream_switch/ports.rs
git commit -m "fix(#140): bound inter-tile wire admission by in-flight words

The inter_tile_pipeline admission checked only the destination slave FIFO
occupancy, ignoring words already in flight toward that slave, so it
over-absorbed ~ROUTE_PER_HOP words past the AM020 external-slave depth (4)
and broke the backpressure chain to the producer. Admit a new inter-tile
word only if slave.fifo.len() + inflight_to(dst) < capacity.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 2: Phase-ordering decision gate (empirical measurement)

This task writes no code. It resolves whether the consume-before-produce phase ordering is a co-cause (spec §1/§7/§9). Its deliverable is a decision: Task 3 in-scope or not.

**Files:** none (capture + decode + record).

- [ ] **Step 1: Co-capture HW + EMU and preserve the HW trace.**

```bash
cd /home/triple/npu-work/xdna-emu
SCRATCH=$(mktemp -d)
XDNA_TRACE_MODE=event_time \
XDNA_TRACE_CORE_EVENTS="PORT_RUNNING_0,PORT_RUNNING_1" \
XDNA_TRACE_MEMTILE_EVENTS="PORT_RUNNING_0,PORT_RUNNING_4" \
  ./scripts/emu-bridge-test.sh --chess-only --trace -v add_one_using_dma 2>&1 | tail -20
cp -r build/bridge-test-results/latest/add_one_using_dma.chess.hw "$SCRATCH/hw_saved"
```

(The EMU `.so` is rebuilt automatically by the script, so it reflects Task 1.)

- [ ] **Step 2: Decode the memtile MM2S cadence, HW vs EMU.**

Run: `tools/trace-port-spans.py build/bridge-test-results/latest add_one_using_dma chess hw emu`
Read the `memtile(...) PORT_RUNNING_4` rows.

- [ ] **Step 3: Decide and record the result in the spec.**

Acceptance target: HW `memtile ... PORT_RUNNING_4 durs=[8,8,14,2,14,2,6,8,1]`.
- If EMU now tracks HW (sub-BD splits present, producer throttled to ~the same span): the lock-gated-drain hypothesis held — **phase ordering is OUT of scope**. Skip Task 3.
- If EMU still shows clean `[16,16,16,...]` (producer races): **phase ordering is a confirmed co-cause** — Task 3 is in-scope.

Append a one-paragraph "Decision gate result (YYYY-MM-DD)" note to `docs/superpowers/specs/2026-06-27-stream-switch-axi-backpressure-design.md` §7 recording the EMU cadence and the decision. Commit that doc note.

```bash
git add docs/superpowers/specs/2026-06-27-stream-switch-axi-backpressure-design.md
git commit -m "docs(#140): record phase-ordering decision-gate result

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 3 (CONDITIONAL — only if Task 2 says co-cause): address phase ordering

Only execute if the Task 2 gate showed the producer still races. Do NOT do this pre-emptively — it underpins the recv-side fidelity that already matches HW.

**Files:**
- Modify: `src/device/array/routing.rs` (`step_data_movement` phase order, lines ~128-135) — and/or the S2MM ingress `ready` accounting so the ingress does not present a free slot in the same cycle it is filled.
- Modify (likely re-validate, maybe re-calibrate): `src/device/dma/engine/stream_io.rs` `input_fifo_capacity` (the S2MM ingress depth 16, whose comment depends on the current ordering).
- Test: extend the Task 1 test or add a dedicated cadence-shape test in `src/device/array/tests.rs`.

- [ ] **Step 1: Write a failing test** capturing the desired effect: with a periodically-lock-stalling consumer, the producer's per-window run-length is sub-BD (e.g. the producer cannot send a full 16-word BD into a 6-deep crossing while the consumer is stalled). Model the consumer stall with the existing lock/DMA test scaffolding; assert the producer's emitted run before its first stall equals `consumer_buffer + crossing_depth` (the documented `8 + 6 = 14`, scaled to the test's parameters).

- [ ] **Step 2: Run it, verify it fails** (current ordering hides the backpressure).
  Run: `cargo test --lib <test_name>` — Expected: FAIL.

- [ ] **Step 3: Adjust the ordering / ingress-ready accounting** so a word filled into the S2MM ingress in Phase 4 is not also counted as drained in the same cycle. The minimal change is to make the ingress `ready` (its `can_accept`) reflect post-Phase-3 occupancy without the Phase-4 fill double-counting; only reorder phases if the accounting fix is insufficient. Keep the #140 per-BD `bd_switch_accept_block` semantics intact.

- [ ] **Step 4: Run the test, verify it passes.**
  Run: `cargo test --lib <test_name>` — Expected: PASS.

- [ ] **Step 5: Re-validate the S2MM ingress depth (16).** Re-run the Task 2 capture for `add_one_using_dma` AND a kernel that exercised the original ingress-depth fix (a memtile relay with 16-word recv BDs); confirm the recv cadence `[16,16,16,16]` / `[8,8,8,8]` still matches HW. If it regressed, the ingress depth needs re-calibration under the new ordering — capture, decode, adjust `DMA_S2MM_INGRESS_FIFO_DEPTH` only with a sourced justification.

- [ ] **Step 6: `cargo test --lib`**, verify no regression, then commit.

```bash
git add -A
git commit -m "fix(#140): propagate inter-tile backpressure under corrected ingress ordering

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 4: Verify intra-tile local-master buffering equals AM020 6-deep

Resolves spec §5 item 2. Fact-check found the route budget into a *local* master is `latency(3) + master_fifo(2) = 5`; AM020 documents the crossing as 6-deep. The slave FIFO (4) is checked separately, so the *total* effective buffering may already be 6. Confirm; correct only if it genuinely under-buffers.

**Files:**
- Read/possibly modify: `src/device/stream_switch/mod.rs` (the `budget = peer.latency + masters[..].fifo_capacity` check, ~line 529).
- Test: `src/device/stream_switch/` (the existing stream-switch unit-test module) or `src/device/array/tests.rs`.

- [ ] **Step 1: Write a test that measures effective local-slave→local-master buffering.** Configure a local-slave→local-master route within one tile, make the master undrainable (no onward route), push words into the slave, step the switch, and count the maximum total words the crossing holds before the slave backpressures (slave FIFO occupancy at saturation + master FIFO occupancy + in-pipeline). Assert it equals 6 (AM020 local-slave→local-master).

```rust
#[test]
fn local_to_local_crossing_buffers_exactly_six() {
    // AM020 ch2: local slave -> local master crossing is 3-cycle latency,
    // 6-deep FIFO (slave 4 + master 2). Verify the effective buffering.
    // ... set up an intra-tile local-slave -> local-master route with an
    // undrainable master, saturate it, count total held words at saturation.
    assert_eq!(total_buffered_at_saturation, 6);
}
```

- [ ] **Step 2: Run it.** Run: `cargo test --lib local_to_local_crossing_buffers_exactly_six`
  - If PASS: the effective buffering is already 6 (slave FIFO supplies the missing slot). Skip Step 3; this task is a confirmed regression-guard only.
  - If FAIL (e.g. saturates at 5): proceed to Step 3.

- [ ] **Step 3 (only if FAIL): correct the budget** so the local-master crossing's total buffering is 6, sourced to AM020 (e.g. include the slave FIFO slot in the route budget, or raise the budget by the documented amount). Re-run Step 2 to GREEN.

- [ ] **Step 4: Confirm per-BD accept-block compatibility** (spec §5 item 6). Run the existing #140 tests:
  Run: `cargo test --lib bd_switch` and `cargo test --lib accept`
  Expected: PASS — the bounded wire must not deadlock the per-BD TREADY deassert.

- [ ] **Step 5: `cargo test --lib`**, verify no regression, then commit.

```bash
git add -A
git commit -m "test(#140): verify intra-tile local-master crossing buffers AM020 6-deep

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 5: Trace-sweep re-baseline with per-delta triage

The integration gate. The change is fabric-global; expect trace churn and triage every delta.

**Files:** none (validation); update trace baselines under `build/` / the baseline location the matrix tool uses.

- [ ] **Step 1: Confirm tooling readiness (pre-flight).** Identify which kernels have HW baseline traces and confirm the matrix/regression diff tool runs. Run a dry-run diff on a single kernel to calibrate.

- [ ] **Step 2: Run the trace sweep** (EMU vs HW). Use the established sweep entry point:
  Run: `./scripts/emu-bridge-test.sh --sweep` (or `--trace=pc-anchored` per the kernel set), redirected to a log file; read the log (do not pipe through `tail`/`grep` live).

- [ ] **Step 3: Triage every changed trace** into win (moved toward HW) vs regression (moved away from HW, or broke a data/bridge test). Use `tools/trace-port-spans.py` on the divergent kernels to inspect cadences. Record the triage table (kernel, port, before, after, HW, verdict).

- [ ] **Step 4: Re-baseline the wins, fix the regressions.** Update the trace baselines for traces that moved toward HW. For any regression, root-cause and fix (return to Task 1/3/4 as needed) before accepting. No silent acceptance of churn.

- [ ] **Step 5: Final `cargo test --lib`** (green) and a clean sweep summary. Commit the re-baselined traces + the triage record.

```bash
git add -A
git commit -m "test(#140): re-baseline stream traces after backpressure fix; triage recorded

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Self-Review

**Spec coverage:** §1 root cause → Task 1. §2 proof / §3 sources → embedded as constraints (AM020 values used in tests). §4 model (audit not rewrite) → Task 1 + Task 4 verify-don't-rewrite. §5 audit items: item 1 (bound wire) → Task 1; item 2 (5 vs 6) → Task 4; items 3/4 (verified depths/latencies) → regression guard via `cargo test --lib`; item 5 (tile-type uniformity, resolved) → no task needed; item 6 (per-BD accept-block) → Task 4 Step 4. §6 trace semantics → preserved, exercised in Task 2/5. §7 validation incl. decision gate → Task 2 (gate) + Task 5 (sweep). §9 phase ordering → Task 2 gate + conditional Task 3; §9 ingress-depth dependency → Task 3 Step 5. §10 success criteria → Tasks 1,2,4,5. Covered.

**Placeholder scan:** Task 1 Step 2 and Task 4 Step 1 contain `/* ... */` markers for port indices and one route-config helper that must be read from the existing test module (Step 1 of Task 1 mandates that read). These are deliberate "adapt to the real local API" points, not unspecified logic — the test's assertions and the fix code are concrete and complete. Task 3 is intentionally conditional and lighter (gated on an empirical result that may make it unnecessary).

**Type consistency:** `inflight_to(dst_idx, dst_slave) -> usize` is defined in Task 1 and reused in its test; `fifo_capacity()` accessor referenced consistently; `step_data_movement(&mut host)` and `stream_switch.{masters,slaves}[i].fifo.len()` used consistently across Task 1 and Task 4.
