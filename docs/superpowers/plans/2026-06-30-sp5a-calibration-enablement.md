# SP-5a Calibration Enablement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the two HW-free prerequisites that de-risk the eventual `calibrated` flip -- a broadcast-provenance fix making the timer-reset single-source guard config-independent, and a cross-language contract test proving a real emulator-produced `origin_d.json` round-trips through the Python inference engine.

**Architecture:** Two independent deliverables, both pure software, both fully testable without Phoenix. Deliverable 1 (Tasks 1-2): tag every queued broadcast `Originated` vs `Relayed` at the two producer functions and record channel-15 flood sources only for `Originated` entries (relays still propagate -- the tag gates only source recording). Deliverable 2 (Tasks 3-4): a committed real-export fixture (guarded against staleness by a read-only Rust liveness assertion + env-gated regen) plus a Python test that consumes it through `run_engine`.

**Tech Stack:** Rust (emulator core, `src/device/`, `src/interpreter/`), Python 3.13 + pytest (inference engine, `tools/inference/`).

**Spec:** `docs/superpowers/specs/2026-06-30-sp5a-calibration-enablement-design.md` (rev2).

## Global Constraints

- **No emoji anywhere** (code, comments, commit messages).
- **Derive from the toolchain / match hardware.** A relay re-emitting channel 15 as L1-interrupt transport is not a timer-reset origin; the fix encodes that hardware distinction, it does not invent a workaround.
- **`calibrated` stays `false`.** SP-5a does NOT flip it and does NOT set the four timing constants. These three regression guards must stay green and unchanged: `crates/xdna-archspec/src/runtime.rs:799` (`broadcast_timing_defaults_uncalibrated`), `src/interpreter/engine/coordinator.rs:4077` (sidecar `calibrated == false`), `src/device/state/effects.rs:1248` (`broadcast_timing_consts_default_to_zero`).
- **Provenance gates only source recording.** Propagation, the fixpoint, L2/trace notifications, and `broadcast_origin_d` are behaviorally unchanged.
- **Python tests use the flat convention** `tools/test_inference_*.py` -- NOT `tools/inference/tests/`.
- **`cargo test --lib` green after every Rust task; Python inference suite green after the Python task. No HW / bridge run required.**
- **Commit message footer:** every commit message ends with these two trailer lines (use `git commit -F -` with a heredoc):
  ```
  Generated using Claude Code.
  Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
  ```

---

### Task 1: Broadcast provenance type + producer tagging (behavior-neutral)

Introduce the `PendingBroadcast`/`BroadcastProvenance` types, change `pending_broadcasts` to carry them, tag the two producers, and mechanically update every existing consumer and test site. The recorder is NOT yet provenance-gated in this task -- behavior stays identical (all channel-15 drains still recorded), so the entire existing suite stays green. This is the skeleton; Task 2 fills it.

**Files:**
- Modify: `src/device/tile/mod.rs` (types + field `:310` + `drain` `:691-693` + producers `seed_broadcasts_for_event` `:743-758`, `tap_l1_interrupt` `:708-722`)
- Modify: `src/device/state/effects.rs` (consumer loop `:568-666`; test direct-pushes/asserts)
- Modify: `src/interpreter/engine/coordinator.rs` (test direct-pushes `:4072`, `:4104`)
- Modify: `src/interpreter/core/interpreter.rs` (test assert `:1157`)
- Test: new unit test in the `effects.rs` test module (drain preserves provenance)

**Interfaces:**
- Produces (consumed by Task 2 and Task 3):
  - `pub enum BroadcastProvenance { Originated, Relayed }` (in `src/device/tile/mod.rs`)
  - `pub struct PendingBroadcast { pub channel: u8, pub provenance: BroadcastProvenance }`
  - `impl PendingBroadcast { pub fn originated(channel: u8) -> Self; pub fn relayed(channel: u8) -> Self; }`
  - `Tile::pending_broadcasts: Vec<PendingBroadcast>`
  - `Tile::drain_pending_broadcasts(&mut self) -> Vec<PendingBroadcast>`

- [ ] **Step 1: Define the types in `src/device/tile/mod.rs`**

Add near the top of the file (after the existing `use` block, before `struct Tile`):

```rust
/// How a queued broadcast entered the network.
///
/// A flood "source" is a tile that ORIGINATES a channel -- the Event_Generate
/// register path or the hardware-error path, both via
/// `seed_broadcasts_for_event`. A tile that re-emits a channel because its L1
/// controller latched and drove its configured IRQ_NO broadcast line
/// (`tap_l1_interrupt`) is a RELAY (interrupt-routing transport reusing a
/// broadcast line), not a source. Channel-15 (timer-reset) flood-source
/// recording counts only `Originated` entries; relays still propagate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BroadcastProvenance {
    Originated,
    Relayed,
}

/// A broadcast channel queued for propagation, tagged with how it entered.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PendingBroadcast {
    pub channel: u8,
    pub provenance: BroadcastProvenance,
}

impl PendingBroadcast {
    pub fn originated(channel: u8) -> Self {
        Self { channel, provenance: BroadcastProvenance::Originated }
    }
    pub fn relayed(channel: u8) -> Self {
        Self { channel, provenance: BroadcastProvenance::Relayed }
    }
}
```

- [ ] **Step 2: Change the field and drain types**

In `src/device/tile/mod.rs`, change the field (`:310`):

```rust
    pub pending_broadcasts: Vec<PendingBroadcast>,
```

The initializer at `:500` stays `pending_broadcasts: Vec::new(),` (unchanged). Change `drain_pending_broadcasts` (`:691-693`):

```rust
    pub fn drain_pending_broadcasts(&mut self) -> Vec<PendingBroadcast> {
        std::mem::take(&mut self.pending_broadcasts)
    }
```

- [ ] **Step 3: Tag the two producers**

In `tap_l1_interrupt` (`src/device/tile/mod.rs:719-721`), tag relayed:

```rust
        for irq_no in latched.into_iter().flatten() {
            self.pending_broadcasts.push(PendingBroadcast::relayed(irq_no));
        }
```

In `seed_broadcasts_for_event` (`src/device/tile/mod.rs:749-757`), tag originated. Change the local `hits` to hold tagged entries:

```rust
        let mut hits = Vec::new();
        for ch in 0..16u8 {
            let ch_event = em.broadcast.read_channel(ch as usize) as u8;
            if event_id != 0 && ch_event == event_id {
                log::info!("Tile({},{}) event {} -> BROADCAST channel {}", self.col, self.row, event_id, ch,);
                hits.push(PendingBroadcast::originated(ch));
            }
        }
        self.pending_broadcasts.extend(hits);
```

- [ ] **Step 4: Update the consumer loop in `effects.rs` (behavior-neutral)**

In `propagate_broadcasts_with_timing` (`src/device/state/effects.rs:596-664`), change the loop variable from `&channel` to `&pb` and replace every `channel` use with `pb.channel`. The recorder stays unconditional on channel 15 in THIS task (provenance gate is added in Task 2). The loop header and recorder become:

```rust
        for &pb in &channels {
            // SP-4b: record the distinct flood SOURCE tiles for channel 15
            // (the timer-reset broadcast). Task 2 adds the provenance gate so
            // L1-relay re-emissions of channel 15 are not miscounted as sources.
            if pb.channel == 15 {
                self.channel15_flood_sources.insert((col, source_row));
            }

            log::info!(
                "Propagating BROADCAST channel {} from tile ({},{}) at cycle {}",
                pb.channel,
                col,
                source_row,
                current_cycle,
            );

            let reached = self.broadcast_origin_d(col, source_row, pb.channel, d_h, d_v);
```

Then inside the `for (c, r, origin_d) in reached` block, replace the two `channel` uses (`:636-642` per-module hw_id and `:661` L1 tap):

```rust
                let (core_hw_id, mem_hw_id) = match tile.tile_kind {
                    TileKind::Compute => (
                        EventModuleType::Core.broadcast_event_base() + pb.channel,
                        EventModuleType::Memory.broadcast_event_base() + pb.channel,
                    ),
                    TileKind::ShimNoc | TileKind::ShimPl => {
                        (EventModuleType::Pl.broadcast_event_base() + pb.channel, 0)
                    }
                    TileKind::Mem => (0, EventModuleType::MemTile.broadcast_event_base() + pb.channel),
                };
```

and:

```rust
                if tile.l1_irq.is_some() {
                    let ev = EventModuleType::Pl.broadcast_event_base() + pb.channel;
                    tile.tap_l1_interrupt(ev);
                }
```

- [ ] **Step 5: Update existing test direct-pushes and assertions (compiler-guided)**

Every `pending_broadcasts.push(<n>)` in test code becomes `pending_broadcasts.push(PendingBroadcast::originated(<n>))`, and every `pending_broadcasts.contains(&<n>)` becomes `pending_broadcasts.iter().any(|pb| pb.channel == <n>)`. The exact sites (the compiler will flag any missed one):

Pushes in `src/device/state/effects.rs`: `:775`, `:790`, `:845`, `:876`, `:898`, `:1165`, `:1196`, `:1215`, `:1226`, `:1238`, `:1240`, `:1276`, `:1290`, `:1314`.
Pushes in `src/interpreter/engine/coordinator.rs`: `:4072`, `:4104`.
`contains` asserts in `src/device/state/effects.rs`: `:760`, `:983`.
`contains` assert in `src/interpreter/core/interpreter.rs`: `:1157`.

Example (effects.rs:775 / coordinator.rs:4072 pattern):
```rust
        dev.array.get_mut(col, row).unwrap().pending_broadcasts.push(PendingBroadcast::originated(5));
```
Example (effects.rs:760 / interpreter.rs:1157 pattern):
```rust
        assert!(t.pending_broadcasts.iter().any(|pb| pb.channel == 5), "IRQ_NO 5 must be queued");
```

Bring `PendingBroadcast` into scope where needed. In `effects.rs` and `coordinator.rs` and `interpreter.rs`, add `use crate::device::tile::{BroadcastProvenance, PendingBroadcast};` to the relevant test module (or the file's existing `use` block if the production code references the type -- production `effects.rs` references `PendingBroadcast` via the `drain` return, so import it at module scope in `effects.rs`).

- [ ] **Step 6: Write the new unit test (drain preserves provenance)**

Add to the `effects.rs` test module:

```rust
    #[test]
    fn drain_pending_broadcasts_preserves_provenance() {
        let mut dev = DeviceState::new_npu1();
        let t = dev.array.get_mut(0, 0).unwrap();
        t.pending_broadcasts.push(PendingBroadcast::originated(15));
        t.pending_broadcasts.push(PendingBroadcast::relayed(7));
        let drained = t.drain_pending_broadcasts();
        assert_eq!(
            drained.iter().map(|pb| (pb.channel, pb.provenance)).collect::<Vec<_>>(),
            vec![
                (15, BroadcastProvenance::Originated),
                (7, BroadcastProvenance::Relayed),
            ],
        );
        assert!(t.pending_broadcasts.is_empty(), "drain must empty the queue");
    }
```

- [ ] **Step 7: Build and run the full lib suite (behavior-neutral check)**

Run: `cargo test --lib`
Expected: PASS, all tests including the new `drain_pending_broadcasts_preserves_provenance`. No existing test changes behavior -- the recorder still records every channel-15 drain. The three `calibrated`/zero-constant guards stay green.

- [ ] **Step 8: Commit**

```bash
git add src/device/tile/mod.rs src/device/state/effects.rs src/interpreter/engine/coordinator.rs src/interpreter/core/interpreter.rs
git commit -F - <<'EOF'
feat(#140): SP-5a Task 1 -- broadcast provenance type (behavior-neutral)

Introduce PendingBroadcast { channel, provenance: Originated|Relayed }, change
pending_broadcasts to Vec<PendingBroadcast>, and tag the two producers
(seed_broadcasts_for_event -> Originated, tap_l1_interrupt -> Relayed). The
channel-15 flood-source recorder is not yet provenance-gated, so behavior is
identical and the full suite stays green; Task 2 adds the gate.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
```

---

### Task 2: Provenance-gated recorder + fixpoint channel-15 RED -> GREEN test

Add the failing test that reproduces the defect (a single timer-reset flood records a spurious second source via L1 relay), then gate the recorder on `Originated` so only genuine origins count.

**Files:**
- Modify: `src/device/state/effects.rs:603-605` (recorder guard)
- Test: new test in the `effects.rs` test module

**Interfaces:**
- Consumes: `PendingBroadcast`, `BroadcastProvenance`, `Tile::pending_broadcasts`, `DeviceState::propagate_broadcasts_fixpoint`, `DeviceState::flood_sources() -> &HashSet<(u8,u8)>` (all from Task 1 / existing).

- [ ] **Step 1: Write the failing test**

Add to the `effects.rs` test module:

```rust
    #[test]
    fn fixpoint_channel15_relay_does_not_record_a_second_flood_source() {
        use crate::device::interrupts::{L1_REG_ENABLE_A, L1_REG_IRQ_NO_A, SwitchId};
        // A single genuine channel-15 (timer-reset) flood originates at shim
        // (0,0). The reached shim (1,0) has its L1 configured to latch the
        // channel-15 broadcast event (Pl base 110 + 15 = 125) and drive
        // IRQ_NO 15 -- so the fixpoint re-floods channel 15 *from (1,0)* as
        // L1-interrupt transport (a relay, not a timer reset). Pre-fix the
        // recorder inserts (1,0) as a spurious second source; post-fix the
        // relay is skipped and only the genuine origin (0,0) counts.
        //
        // This config self-feeds (the relay flood re-taps (1,0)'s own L1), so
        // propagate_broadcasts_fixpoint runs to its MAX_ITERS cap and logs a
        // warning -- expected under this pathological config. Assert only on
        // flood_sources(), never on log output or iteration count.
        let mut dev = DeviceState::new_npu1();
        {
            let l1 = dev.array.get_mut(1, 0).unwrap().l1_irq.as_mut().unwrap();
            l1.set_irq_event_slot(SwitchId::A, 0, 110 + 15); // event 125
            l1.write_register(L1_REG_ENABLE_A, 1 << 16);
            l1.write_register(L1_REG_IRQ_NO_A, 15);
        }
        dev.array
            .get_mut(0, 0)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(15));
        dev.propagate_broadcasts_fixpoint(0, 0);

        let sources = dev.flood_sources();
        assert_eq!(
            sources.len(),
            1,
            "a single genuine timer-reset origin must record exactly one flood source; got {sources:?}",
        );
        assert!(
            sources.contains(&(0, 0)),
            "the genuine origin (0,0) must be the single recorded source; got {sources:?}",
        );
    }
```

- [ ] **Step 2: Run the test to verify it FAILS**

Run: `cargo test --lib fixpoint_channel15_relay_does_not_record_a_second_flood_source`
Expected: FAIL -- `assertion failed: sources.len() == 1` (actual `2`): the relay shim (1,0) is recorded as a spurious second source because the recorder does not yet check provenance.

- [ ] **Step 3: Add the provenance gate to the recorder**

In `src/device/state/effects.rs:603`, change the recorder to require `Originated`:

```rust
            // SP-5a: record a channel-15 flood SOURCE only for genuine
            // originations. An L1-relay re-emission of channel 15 (interrupt
            // transport reusing the timer-reset broadcast line) is not a
            // timer-reset source and must not pollute the single-source guard.
            if pb.channel == 15 && pb.provenance == BroadcastProvenance::Originated {
                self.channel15_flood_sources.insert((col, source_row));
            }
```

Ensure `BroadcastProvenance` is in scope in `effects.rs` (added in Task 1 Step 5).

- [ ] **Step 4: Run the new test, the multi-origin regression, and the termination test**

Run: `cargo test --lib fixpoint_channel15_relay_does_not_record_a_second_flood_source`
Expected: PASS.

Run: `cargo test --lib export_origin_d_sidecar_omits_flood_source_with_multiple_sources`
Expected: PASS -- two GENUINE `Originated` channel-15 pushes still yield `len() == 2` -> `flood_source: null`. The fix must not mask real multi-origin floods.

Run: `cargo test --lib fixpoint_propagation_terminates_under_self_feeding_config`
Expected: PASS -- provenance does not affect termination.

- [ ] **Step 5: Run the full lib suite**

Run: `cargo test --lib`
Expected: PASS, all tests. The three `calibrated`/zero-constant guards stay green (SP-5a does not touch them).

- [ ] **Step 6: Commit**

```bash
git add src/device/state/effects.rs
git commit -F - <<'EOF'
fix(#140): SP-5a Task 2 -- record channel-15 flood source only for originations

The recorder gated on `channel == 15` alone: a reached shim whose L1 latches
event 125 with IRQ_NO==15 re-emits channel 15 via the fixpoint, and the relay
tile was wrongly inserted as a second flood source, degrading a single-origin
timer-reset flood to the multi-source failure path. Gate on
provenance == Originated so relays propagate but do not count as sources.
Genuine multi-origin detection is preserved (two Originated channel-15 floods
still trip the guard).

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
```

---

### Task 3: Rust real-export fixture with read-only liveness guard

Pin the real exported sidecar JSON as a committed read-only golden file, asserted equal to the live export (drift fails loudly), regenerated only via an explicit env gate.

**Files:**
- Modify: `src/interpreter/engine/coordinator.rs` (new test alongside `export_origin_d_sidecar_matches_contract` `:4063`)
- Create: `tools/tests/fixtures/origin_d_real_export.json` (generated via the regen path, then committed)

**Interfaces:**
- Consumes: `InterpreterEngine::new_npu1`, `InterpreterEngine::export_origin_d_sidecar`, `PendingBroadcast::originated` (Task 1).
- Produces (consumed by Task 4): the committed fixture `tools/tests/fixtures/origin_d_real_export.json` -- a real Rust export with `calibrated: false`.

- [ ] **Step 1: Write the fixture test (read-only compare + env-gated regen)**

Add to the `coordinator.rs` test module (after `export_origin_d_sidecar_matches_contract`):

```rust
    #[test]
    fn export_origin_d_sidecar_matches_committed_fixture() {
        // SP-5a Task 3: pin the REAL exported sidecar JSON as a committed
        // read-only golden file. The live export must equal it (drift fails
        // loudly). Regeneration is explicit and env-gated -- the default
        // `cargo test` run never writes the fixture:
        //   UPDATE_FIXTURES=1 cargo test --lib export_origin_d_sidecar_matches_committed_fixture
        // The Python contract test (tools/test_inference_real_sidecar_contract.py)
        // consumes the same committed file to prove the cross-language round-trip.
        let mut engine = InterpreterEngine::new_npu1();
        {
            let tile = engine.device_mut().array.get_mut(0, 0).expect("shim tile (0,0)");
            tile.pending_broadcasts.push(PendingBroadcast::originated(15));
        }
        engine.device_mut().propagate_broadcasts(0, 0);
        let live = engine.export_origin_d_sidecar();

        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tools/tests/fixtures/origin_d_real_export.json");

        if std::env::var("UPDATE_FIXTURES").is_ok() {
            std::fs::create_dir_all(fixture_path.parent().unwrap()).unwrap();
            std::fs::write(&fixture_path, serde_json::to_string_pretty(&live).unwrap()).unwrap();
        }

        let committed = std::fs::read_to_string(&fixture_path).unwrap_or_else(|e| {
            panic!(
                "committed fixture {} missing ({e}); regenerate with \
                 UPDATE_FIXTURES=1 cargo test --lib export_origin_d_sidecar_matches_committed_fixture",
                fixture_path.display(),
            )
        });
        let committed_val: serde_json::Value = serde_json::from_str(&committed).unwrap();
        assert_eq!(
            live, committed_val,
            "live origin_D export drifted from the committed fixture; if intended, \
             regenerate with UPDATE_FIXTURES=1 cargo test --lib export_origin_d_sidecar_matches_committed_fixture",
        );
    }
```

- [ ] **Step 2: Generate the committed fixture via the env gate**

Run: `UPDATE_FIXTURES=1 cargo test --lib export_origin_d_sidecar_matches_committed_fixture`
Expected: PASS. The file `tools/tests/fixtures/origin_d_real_export.json` now exists, containing the real export (`"calibrated": false`, `"flood_source": "0|0"`, a `modules` table keyed `"col|row|kind"`).

- [ ] **Step 3: Verify the read-only path passes WITHOUT the env gate**

Run: `cargo test --lib export_origin_d_sidecar_matches_committed_fixture`
Expected: PASS -- the live export equals the just-committed fixture, and the test did not rewrite the file.

- [ ] **Step 4: Run the full lib suite**

Run: `cargo test --lib`
Expected: PASS, all tests.

- [ ] **Step 5: Commit (test + generated fixture together)**

```bash
git add src/interpreter/engine/coordinator.rs tools/tests/fixtures/origin_d_real_export.json
git commit -F - <<'EOF'
test(#140): SP-5a Task 3 -- committed real-export sidecar fixture + liveness guard

Pin the real export_origin_d_sidecar() JSON (single-source channel-15 flood,
calibrated:false) as a committed read-only golden file. The live export is
asserted equal to it so any schema drift fails loudly; regeneration is explicit
and env-gated (UPDATE_FIXTURES=1), so the default test run never overwrites the
golden. The Python contract test consumes this same file.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
```

---

### Task 4: Python cross-language contract test

Prove the real Rust-produced `origin_d.json` round-trips through `load_model` + `run_engine`: the uncalibrated fixture is a clean no-op, and the fixture's schema can drive a causal emission when calibrated.

**Files:**
- Create: `tools/test_inference_real_sidecar_contract.py`
- Consumes (read-only): `tools/tests/fixtures/origin_d_real_export.json` (Task 3)

**Interfaces:**
- Consumes: `inference.engine.run_engine(run_dirs, ledger_path, candidate_pairs, model_path=...)` returning a dict with keys `causal` and `provenance_ok`.

- [ ] **Step 1: Write the contract test**

Create `tools/test_inference_real_sidecar_contract.py`:

```python
# tools/test_inference_real_sidecar_contract.py
"""Cross-language contract (SP-5a, #140): a REAL Rust-produced origin_d.json
round-trips through the Python inference engine.

SP-4b only ever fed hand-written synthetic sidecars to run_engine. This test
closes that gap by consuming the committed real export
(tools/tests/fixtures/origin_d_real_export.json, produced by the Rust test
export_origin_d_sidecar_matches_committed_fixture). load_model re-keys the
real "col|row|kind" module strings via to_domain_key on load -- so loading the
real fixture at all exercises the real key/translation path and catches
schema/key/type drift between the two language sides.

The fixture is committed (not a gitignored capture), so it is always present --
no skipif gate needed.

Two properties:
  (a) The real fixture carries calibrated:false (SP-5a does not flip it), so it
      is a clean no-op end-to-end even where a cross-domain gap exists.
  (b) The real fixture's SCHEMA can drive a causal emission once calibrated:
      flip calibrated:true and populate modules for a cross-domain pair's two
      domains. Note: the cross-domain emission gate (inference.grounding)
      checks both domains are present in `modules`, NOT flood_source
      reachability, so this test asserts nothing about flood_source.
"""
import json
from pathlib import Path

from inference.engine import run_engine

_HERE = Path(__file__).resolve().parent
_FIXTURE = _HERE / "tests" / "fixtures" / "origin_d_real_export.json"


def _ev(col, row, name, soc, pkt_type=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


def _synthetic_cross_domain_fixture(tmp_path):
    """A minimal 2-run capture with one cross-domain pair (shim "1|0|2" MM2S ->
    core "1|2|0" CORE), exact raw offset 40 -- the shape try_causal decomposes.
    Self-contained (mirrors test_inference_sp4b_e2e.py)."""
    dirs = []
    for i, row in enumerate([
        {"1|0|2|MM2S": 0, "1|2|0|CORE": 40},
        {"1|0|2|MM2S": 9, "1|2|0|CORE": 49},
    ]):
        rd = tmp_path / f"syn_run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for key, delta in row.items():
            col, r, pkt, name = key.split("|")
            evs.append(_ev(int(col), int(r), name, 1000 + delta, pkt_type=int(pkt)))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    ledger_path = tmp_path / "syn_led.json"
    ledger_path.write_text(json.dumps({"entries": [
        {"cite": "program:x", "a": "1|0|2|MM2S", "b": "1|2|0|CORE",
         "kind": "program"}]}))
    return dirs, str(ledger_path)


def test_real_fixture_loads_and_is_noop_pre_calibration(tmp_path):
    real = json.loads(_FIXTURE.read_text())
    assert real["calibrated"] is False, "SP-5a must not flip calibrated"
    syn_dirs, syn_ledger = _synthetic_cross_domain_fixture(tmp_path)
    sidecar = tmp_path / "origin_d_real.json"
    sidecar.write_text(json.dumps(real))
    # Feeding the real fixture exercises load_model's re-keying of its real
    # "col|row|kind" module strings; calibrated:false then short-circuits
    # causal emission, so causal stays empty even with a cross-domain gap.
    rep = run_engine(syn_dirs, syn_ledger, [("1|2|0|CORE", "1|0|2|MM2S")],
                     model_path=str(sidecar))
    assert rep["provenance_ok"] is True
    assert rep["causal"] == []


def test_real_fixture_schema_drives_causal_when_calibrated(tmp_path):
    real = json.loads(_FIXTURE.read_text())
    syn_dirs, syn_ledger = _synthetic_cross_domain_fixture(tmp_path)
    # Start from the real fixture's schema; flip calibrated and set modules to
    # cover the synthetic cross-domain pair's two domains
    # (model_io.MODULE_PKT_TYPE: core=0, shim=2). skew(shim 2 - core 5 = -3),
    # raw 40 - (-3) = 43.
    calibrated = dict(real)
    calibrated["calibrated"] = True
    calibrated["modules"] = {"1|2|core": 5, "1|0|shim": 2}
    sidecar = tmp_path / "origin_d_calibrated.json"
    sidecar.write_text(json.dumps(calibrated))
    rep = run_engine(syn_dirs, syn_ledger, [("1|2|0|CORE", "1|0|2|MM2S")],
                     model_path=str(sidecar))
    assert rep["provenance_ok"] is True
    assert ("1|2|0|CORE", "1|0|2|MM2S", 43) in rep["causal"]
```

- [ ] **Step 2: Run the contract test**

Run: `PYTHONPATH=tools python -m pytest tools/test_inference_real_sidecar_contract.py -v`
Expected: PASS, both tests. (a) confirms the real fixture loads and is a no-op; (b) confirms the schema drives the Delta_wall=43 causal triple when calibrated.

- [ ] **Step 3: Run the inference suite**

Run: `PYTHONPATH=tools python -m pytest tools/test_inference_*.py -q`
Expected: PASS (no regressions in the existing inference tests).

- [ ] **Step 4: Commit**

```bash
git add tools/test_inference_real_sidecar_contract.py
git commit -F - <<'EOF'
test(#140): SP-5a Task 4 -- cross-language real-sidecar contract test

Prove a real Rust-produced origin_d.json round-trips through load_model +
run_engine. (a) the committed uncalibrated fixture loads (exercising the real
col|row|kind -> to_domain_key translation) and is a clean no-op even with a
cross-domain gap; (b) the same schema, flipped calibrated with modules covering
a cross-domain pair, drives the Delta_wall=43 causal emission. Asserts nothing
about flood_source (the emission gate checks domains-present, not flood_source
reachability).

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
```

---

## Self-Review

**1. Spec coverage.**
- §2 provenance fix -> Tasks 1 (type + tagging + behavior-neutral) and 2 (recorder gate + RED->GREEN fixpoint test). Producer surface (two functions) tagged; recorder gated on `Originated`; propagation unchanged (only the `insert` is conditioned). Covered.
- §2f tests -> Task 1 Step 6 (drain preserves provenance), Task 2 (fixpoint+ch15 relay RED->GREEN; multi-origin regression; termination). Covered.
- §3 contract verification -> Task 3 (Rust read-only fixture + liveness + env-gated regen, CARGO_MANIFEST_DIR path) and Task 4 (Python consumer: no-op pre-calibration + calibrated emission, no flood_source assertion). Covered.
- Non-goals respected: `calibrated` never flipped; the four constants untouched; the three regression guards explicitly re-checked green (Task 1 Step 7, Task 2 Step 5); no in-sweep integration; no HW run. Covered.

**2. Placeholder scan.** No TBD/TODO. Every code step shows complete code; every run step shows the exact command and expected result; commit steps show full heredocs with the required footer. The Task 1 Step 5 site list enumerates exact line numbers and gives the two transform patterns with concrete examples (not "similar to").

**3. Type consistency.** `PendingBroadcast`/`BroadcastProvenance` and the constructors `originated`/`relayed` are defined in Task 1 and used identically in Tasks 2-3. `drain_pending_broadcasts -> Vec<PendingBroadcast>`, `flood_sources() -> &HashSet<(u8,u8)>` (`.len()`/`.contains(&(0,0))`), and `run_engine(run_dirs, ledger_path, candidate_pairs, model_path=...)` with `rep["causal"]`/`rep["provenance_ok"]` match their real signatures. The synthetic-fixture helper and the Delta_wall=43 arithmetic match the verified SP-4b pattern.

Plan complete.
