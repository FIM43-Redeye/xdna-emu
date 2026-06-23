# Instruction-Event Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the core lock `INSTR_LOCK_ACQUIRE_REQ -> INSTR_LOCK_RELEASE_REQ` edge as an oriented within-domain candidate pair in the inference engine's ledger, derived from the Chess ELF lock-order decode.

**Architecture:** A Rust helper computes the aggregate lock-order fact (`min acquire PC < min release PC`) from `core_relay`'s ELF decode; `build_dump` re-runs that decode per compute tile and serialises a `lock_order` field; the Python `dump_model` loads it; the generator adds a non-port emission path that emits a `kind="program"` ledger edge (with a `program_order:` cite) for the two lock trace events, with a matching audit branch; `selfmodel`'s menu gains the two events so they are enumerated and flow into `candidate_pairs`.

**Tech Stack:** Rust (lib + `examples/dump_config_json.rs`, serde), Python 3 (`tools/config_extract/`, `tools/inference/`, pytest).

## Global Constraints

- DERIVE FROM THE TOOLCHAIN: lock-order orientation comes from the Chess ELF via `core_relay`, never a hardcoded "acquire precedes release" truism. Event names come from the aie-rt header (`INSTR_LOCK_ACQUIRE_REQ`=44, `INSTR_LOCK_RELEASE_REQ`=45, `aie-rt/.../xaie_events_aieml.h:79-80`).
- Reuse `kind="program"` (-> `program_path` predicate). Do NOT introduce a new ledger kind: `tools/inference/ledger.py:27` `_KINDS` hard-rejects unknown kinds and `try_derives` (`rules.py:40`) orients only on `config_path`/`program_path`.
- Orientation fact: `min(acquire PC) < min(release PC)` (first acquire before first release). Aggregate across lock IDs (the trace events are lock-ID-agnostic). Emit nothing if not satisfied (safe false-negative).
- The deliverable is "the oriented candidate pair is PRODUCED"; nothing grounds it (offset/falsifier/report) -- that is the next plan.
- Menu addition (Task 4) and the generator emission path (Task 4) land in the SAME task: adding the events to the menu without the emission path would make them always-unresolved and regress the HW loop's convergence.
- Run `cargo test --lib` after Rust changes; run the offline Python suite (`cd tools && python -m pytest test_config_extract_generator.py test_selfmodel.py -q`) after Python changes. No HW needed for any task in this plan.

---

### Task 1: Rust -- aggregate lock-order helper in `core_relay`

**Files:**
- Modify: `src/device/stream_switch/core_relay.rs` (add `pub fn aggregate_lock_order`; add tests in the existing `#[cfg(test)] mod`)

**Interfaces:**
- Consumes: `CoreLockUsage { locks: Vec<CoreLockOp>, .. }`, `CoreLockOp { lock_id: u8, kind: CoreLockKind, pc: u32 }`, `CoreLockKind::{Acquire, Release}` (all already defined in this file, lines 74-94).
- Produces: `pub fn aggregate_lock_order(usage: &CoreLockUsage) -> Option<(u32, u32)>` returning `(min_acquire_pc, min_release_pc)` iff `min_acquire_pc < min_release_pc`, else `None`.

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block in `src/device/stream_switch/core_relay.rs` (it already constructs `CoreLockUsage` literals, see ~line 1085):

```rust
#[test]
fn aggregate_lock_order_acquire_before_release() {
    let usage = CoreLockUsage {
        locks: vec![
            CoreLockOp { lock_id: 1, kind: CoreLockKind::Acquire, pc: 0x134 },
            CoreLockOp { lock_id: 0, kind: CoreLockKind::Release, pc: 0x1b4 },
        ],
        accesses: vec![],
        fn_end: 0x300,
        bundle_pcs: vec![],
    };
    assert_eq!(aggregate_lock_order(&usage), Some((0x134, 0x1b4)));
}

#[test]
fn aggregate_lock_order_uses_min_pcs_across_lock_ids() {
    // Aggregate: min acquire (0x134 on lock 1) precedes min release (0x1b4 on lock 0),
    // even though no single lock has both an acquire and a release.
    let usage = CoreLockUsage {
        locks: vec![
            CoreLockOp { lock_id: 2, kind: CoreLockKind::Acquire, pc: 0x150 },
            CoreLockOp { lock_id: 1, kind: CoreLockKind::Acquire, pc: 0x134 },
            CoreLockOp { lock_id: 3, kind: CoreLockKind::Release, pc: 0x2c0 },
            CoreLockOp { lock_id: 0, kind: CoreLockKind::Release, pc: 0x1b4 },
        ],
        accesses: vec![],
        fn_end: 0x300,
        bundle_pcs: vec![],
    };
    assert_eq!(aggregate_lock_order(&usage), Some((0x134, 0x1b4)));
}

#[test]
fn aggregate_lock_order_none_when_release_first() {
    let usage = CoreLockUsage {
        locks: vec![
            CoreLockOp { lock_id: 0, kind: CoreLockKind::Release, pc: 0x100 },
            CoreLockOp { lock_id: 1, kind: CoreLockKind::Acquire, pc: 0x200 },
        ],
        accesses: vec![],
        fn_end: 0x300,
        bundle_pcs: vec![],
    };
    assert_eq!(aggregate_lock_order(&usage), None);
}

#[test]
fn aggregate_lock_order_none_when_kind_missing() {
    let usage = CoreLockUsage {
        locks: vec![CoreLockOp { lock_id: 1, kind: CoreLockKind::Acquire, pc: 0x100 }],
        accesses: vec![],
        fn_end: 0x300,
        bundle_pcs: vec![],
    };
    assert_eq!(aggregate_lock_order(&usage), None);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p xdna-emu --lib aggregate_lock_order`
Expected: FAIL -- `cannot find function aggregate_lock_order`.

- [ ] **Step 3: Implement the helper**

Add near the other lock helpers in `src/device/stream_switch/core_relay.rs` (e.g. after `analyze_core_program`):

```rust
/// Aggregate lock-order fact for the instruction-event layer.
///
/// Returns `(min acquire PC, min release PC)` iff the first acquire precedes the
/// first release in program order, else `None` (safe false-negative). Aggregate
/// across lock IDs because the `INSTR_LOCK_ACQUIRE_REQ` / `INSTR_LOCK_RELEASE_REQ`
/// trace events are lock-ID-agnostic (they fire on every acquire / release). The
/// orientation is DERIVED from the ELF-decoded PCs, not assumed.
pub fn aggregate_lock_order(usage: &CoreLockUsage) -> Option<(u32, u32)> {
    let min_acq = usage
        .locks
        .iter()
        .filter(|o| o.kind == CoreLockKind::Acquire)
        .map(|o| o.pc)
        .min()?;
    let min_rel = usage
        .locks
        .iter()
        .filter(|o| o.kind == CoreLockKind::Release)
        .map(|o| o.pc)
        .min()?;
    (min_acq < min_rel).then_some((min_acq, min_rel))
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p xdna-emu --lib aggregate_lock_order`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/device/stream_switch/core_relay.rs
git commit -m "feat(#140): aggregate_lock_order helper (min-acquire < min-release)"
```

---

### Task 2: Rust -- serialise `lock_order` in the dump + regenerate fixtures

**Files:**
- Modify: `examples/dump_config_json.rs` (add `LockOrderDump` struct; add `lock_order` field to `TileDump`; populate it in `build_dump`; add a test)
- Modify (regenerate): `tools/config_extract/fixtures/add_one_using_dma.config.json`, `add_one_objFifo.config.json`, `vector_scalar_using_dma.config.json`

**Interfaces:**
- Consumes: `xdna_emu::device::stream_switch::core_relay::{analyze_core_program, aggregate_lock_order}`, `xdna_emu::interpreter::InstructionDecoder`, `tile.is_compute()`, `tile.program_memory()` (all public).
- Produces: a per-tile JSON object `"lock_order": {"acq_pc": <u32>, "rel_pc": <u32>}` (absent when `None`, via `skip_serializing_if`).

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `examples/dump_config_json.rs` (mirror the existing `dump_produces_route_graph_and_event_bindings_for_add_one` early-return-if-unloadable pattern):

```rust
#[test]
fn dump_emits_lock_order_for_add_one_compute_tile() {
    let xclbin_path =
        "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin";
    let insts_path =
        "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/insts.bin";
    let Ok(state) = load_state_from_xclbin(xclbin_path, Some(insts_path)) else {
        return; // build artifacts absent in this environment -- skip
    };
    let dump = build_dump(&state);
    let json = serde_json::to_value(&dump).unwrap();
    // The compute tile (row 2) must carry an oriented lock_order with acq < rel.
    let tiles = json["tiles"].as_array().unwrap();
    let compute = tiles
        .iter()
        .find(|t| t["kind"] == "compute")
        .expect("a compute tile");
    let lo = &compute["lock_order"];
    assert!(!lo.is_null(), "compute tile must have lock_order");
    assert!(lo["acq_pc"].as_u64().unwrap() < lo["rel_pc"].as_u64().unwrap());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xdna-emu --example dump_config_json dump_emits_lock_order`
Expected: FAIL -- `lock_order` key absent (`is_null()` true) or compilation error (field missing).

- [ ] **Step 3: Add the `LockOrderDump` struct**

In `examples/dump_config_json.rs`, near the other `*Dump` structs (after `TileDump`):

```rust
/// Aggregate core lock-order fact: first acquire precedes first release.
#[derive(Debug, Serialize)]
pub struct LockOrderDump {
    /// Min acquire PC (first acquire in program order).
    pub acq_pc: u32,
    /// Min release PC (first release in program order).
    pub rel_pc: u32,
}
```

- [ ] **Step 4: Add the field to `TileDump`**

In the `TileDump` struct (around line 89), add as the last field:

```rust
    /// Aggregate core lock-order fact (compute tiles with a recoverable
    /// acquire-before-release program; absent otherwise).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lock_order: Option<LockOrderDump>,
```

- [ ] **Step 5: Populate it in `build_dump`**

In the `build_dump` tile closure (around line 379-525), before the `TileDump { .. }` literal (e.g. right after the `shim_mux` block), add. This mirrors the load-bearing non-empty guard from `route_graph.rs:850-858` so idle compute tiles (program memory all-zero) are skipped:

```rust
            // --- lock_order (compute tiles with a recoverable ELF lock program) ---
            // build_dump has no handle to the CoreLockUsage that resolve_route_graph
            // computes and discards, so re-run the decode here. The non-empty guard
            // is load-bearing: program_memory() returns Some(&[0u8; N]) for compute
            // tiles with no ELF loaded; without it we would linearly decode ~16KB of
            // zeros per idle tile.
            let lock_order = if tile.is_compute() {
                tile.program_memory().and_then(|prog| {
                    if prog.iter().any(|&b| b != 0) {
                        let dec = xdna_emu::interpreter::InstructionDecoder::load_cached();
                        let usage = xdna_emu::device::stream_switch::core_relay::analyze_core_program(
                            &prog[..],
                            0,
                            &dec,
                        );
                        xdna_emu::device::stream_switch::core_relay::aggregate_lock_order(&usage)
                            .map(|(acq_pc, rel_pc)| LockOrderDump { acq_pc, rel_pc })
                    } else {
                        None
                    }
                })
            } else {
                None
            };
```

Then add `lock_order,` to the `TileDump { .. }` literal (after `shim_mux,`).

- [ ] **Step 6: Run the test + full lib tests to verify pass + no regression**

Run: `cargo test -p xdna-emu --example dump_config_json dump_emits_lock_order`
Expected: PASS (or skip if artifacts absent -- if it skips, verify manually with Step 7's regen command that the JSON contains `lock_order`).
Run: `cargo test --lib`
Expected: PASS, no regressions.

- [ ] **Step 7: Regenerate the three suite fixtures**

```bash
cd /home/triple/npu-work/xdna-emu
for k in add_one_using_dma add_one_objFifo vector_scalar_using_dma; do
  X=/home/triple/npu-work/mlir-aie/build/test/npu-xrt/$k/chess/aie.xclbin
  I=/home/triple/npu-work/mlir-aie/build/test/npu-xrt/$k/chess/insts.bin
  cargo run -q --example dump_config_json -- "$X" "$I" > tools/config_extract/fixtures/$k.config.json
done
```

Verify the add_one fixture now carries the fact (the compute tile is row 2):

```bash
python3 -c "import json; d=json.load(open('tools/config_extract/fixtures/add_one_using_dma.config.json')); print([t.get('lock_order') for t in d['tiles'] if t['kind']=='compute'])"
```
Expected: a non-empty `[{'acq_pc': ..., 'rel_pc': ...}]` with `acq_pc < rel_pc`.

- [ ] **Step 8: Commit**

```bash
git add examples/dump_config_json.rs tools/config_extract/fixtures/
git commit -m "feat(#140): serialise lock_order in dump + regenerate suite fixtures"
```

---

### Task 3: Python -- load `lock_order` in `dump_model`

**Files:**
- Modify: `tools/config_extract/dump_model.py` (add `LockOrder` dataclass; add `lock_order` field to `TileDump`; load it backward-compatibly)
- Test: `tools/config_extract/test_dump_model_lock_order.py` (new)

**Interfaces:**
- Consumes: the regenerated fixture JSON with `"lock_order": {"acq_pc", "rel_pc"}` (Task 2).
- Produces: `TileDump.lock_order: Optional[LockOrder]` where `LockOrder` has `.acq_pc: int`, `.rel_pc: int`.

- [ ] **Step 1: Write the failing test**

Create `tools/config_extract/test_dump_model_lock_order.py`:

```python
from pathlib import Path
from config_extract.dump_model import load_dump, LockOrder

_FIX = (Path(__file__).resolve().parent / "fixtures"
        / "add_one_using_dma.config.json")


def test_compute_tile_lock_order_loaded():
    dump = load_dump(str(_FIX))
    compute = [t for t in dump.tiles if t.kind == "compute"]
    assert compute, "fixture must have a compute tile"
    los = [t.lock_order for t in compute if t.lock_order is not None]
    assert los, "compute tile must carry a lock_order fact"
    lo = los[0]
    assert isinstance(lo, LockOrder)
    assert lo.acq_pc < lo.rel_pc


def test_lock_order_absent_is_none():
    # A tile dict without lock_order loads with lock_order=None (backward-compat).
    from config_extract.dump_model import _load_tile
    minimal = {"col": 0, "row": 0, "kind": "shim", "ports": [],
               "event_port_selection": [None] * 8, "dma_channels": [],
               "bds": [], "locks": []}
    t = _load_tile(minimal)
    assert t.lock_order is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools && python -m pytest config_extract/test_dump_model_lock_order.py -q`
Expected: FAIL -- `cannot import name 'LockOrder'`.

- [ ] **Step 3: Add the `LockOrder` dataclass + field + loader**

In `tools/config_extract/dump_model.py`, add the dataclass near `ShimMux` (after line 107):

```python
@dataclass(frozen=True)
class LockOrder:
    """Aggregate core lock-order fact: first acquire precedes first release."""
    acq_pc: int
    rel_pc: int
```

Add the field to `TileDump` (after `shim_mux`, line 125):

```python
    lock_order: Optional[LockOrder] = None
```

Add a loader helper near `_load_shim_mux` (before `_load_tile`):

```python
def _load_lock_order(d: Optional[dict]) -> Optional[LockOrder]:
    if d is None:
        return None
    return LockOrder(acq_pc=d["acq_pc"], rel_pc=d["rel_pc"])
```

Wire it into `_load_tile` (add as the last kwarg, line 231):

```python
        lock_order=_load_lock_order(d.get("lock_order")),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tools && python -m pytest config_extract/test_dump_model_lock_order.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/config_extract/dump_model.py tools/config_extract/test_dump_model_lock_order.py
git commit -m "feat(#140): load lock_order in dump_model (backward-compatible)"
```

---

### Task 4: Python -- menu + generator emission + audit branch (deliverable)

**Files:**
- Modify: `tools/inference/selfmodel.py` (add the two events to `_MENU[0]`)
- Modify: `tools/config_extract/generator.py` (lock-order emission path in `generate_ledger`; audit branch in `audit_ledger`)
- Test: `tools/config_extract/test_generator_lock_order.py` (new)

**Interfaces:**
- Consumes: `TileDump.lock_order` (Task 3); `generate_ledger(dump, fired_event_keys, start_col)` and `audit_ledger(led, dump, start_col)` (existing); dump tiles are RELATIVE-col, fired keys are ABSOLUTE-col (so an event key is `f"{tile.col + start_col}|{tile.row}|0|<name>"`).
- Produces: a ledger entry `{cite: "program_order:<acq>--core-locks-->{rel}", a: <acq_key>, b: <rel_key>, kind: "program"}`; `candidate_pairs_from_dump` returns `(rel_key, acq_key)`.

- [ ] **Step 1: Write the failing tests**

Create `tools/config_extract/test_generator_lock_order.py`:

```python
from pathlib import Path
from config_extract.dump_model import load_dump
from config_extract.generator import generate_ledger, audit_ledger
from inference.selfmodel import (
    enumerate_configured_events, candidate_pairs_from_dump)

_FIX = (Path(__file__).resolve().parent / "fixtures"
        / "add_one_using_dma.config.json")
_ACQ = "1|2|0|INSTR_LOCK_ACQUIRE_REQ"
_REL = "1|2|0|INSTR_LOCK_RELEASE_REQ"


def test_lock_order_edge_emitted_and_oriented():
    dump = load_dump(str(_FIX))
    led = generate_ledger(dump, [_ACQ, _REL], start_col=1)
    # Filter to the lock pair -- the generator emits many program entries for
    # add_one from the existing through-core DMA fan-out; assert on the lock one.
    lock = [e for e in led["entries"]
            if e["a"].endswith("INSTR_LOCK_ACQUIRE_REQ")
            and e["b"].endswith("INSTR_LOCK_RELEASE_REQ")]
    assert len(lock) == 1
    e = lock[0]
    assert e["a"] == _ACQ and e["b"] == _REL and e["kind"] == "program"
    assert e["cite"].startswith("program_order:")
    # The portless lock entry must pass audit.
    assert audit_ledger(led, dump, start_col=1) == []


def test_menu_enumerates_lock_events():
    dump = load_dump(str(_FIX))
    keys = enumerate_configured_events(dump, start_col=1)
    assert _ACQ in keys and _REL in keys


def test_candidate_pairs_includes_lock_order():
    dump = load_dump(str(_FIX))
    configured = enumerate_configured_events(dump, start_col=1)
    pairs = candidate_pairs_from_dump(dump, configured, start_col=1)
    assert (_REL, _ACQ) in pairs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools && python -m pytest config_extract/test_generator_lock_order.py -q`
Expected: FAIL -- no lock entry emitted (`len(lock) == 0`) and `_ACQ` not in enumerated keys.

- [ ] **Step 3: Add the two events to the menu**

In `tools/inference/selfmodel.py`, change `_MENU[0]` (the core entry, lines 28-29) to:

```python
    0: ["PERF_CNT_2", "INSTR_VECTOR", "LOCK_STALL",               # core
        "MEMORY_STALL", "STREAM_STALL",
        "INSTR_LOCK_ACQUIRE_REQ", "INSTR_LOCK_RELEASE_REQ"],
```

- [ ] **Step 4: Add the lock-order emission path to `generate_ledger`**

In `tools/config_extract/generator.py`, in `generate_ledger`, immediately before the final `return {...}` (line 248), add:

```python
    # Instruction-event layer (#140): emit the oriented core lock-order edge for
    # each compute tile whose ELF has acquire-before-release, when both lock trace
    # events fired. Bypasses port resolution -- lock events have no port, so the
    # permutations loop above skips them; this is a separate non-port path.
    fired_set = set(fired_event_keys)
    for tile in dump.tiles:
        if tile.lock_order is None:
            continue
        acq_key = f"{tile.col + start_col}|{tile.row}|0|INSTR_LOCK_ACQUIRE_REQ"
        rel_key = f"{tile.col + start_col}|{tile.row}|0|INSTR_LOCK_RELEASE_REQ"
        if acq_key in fired_set and rel_key in fired_set:
            entries.append({
                "cite": f"program_order:{acq_key}--core-locks-->{rel_key}",
                "a": acq_key,   # parent = acquire (first acquire precedes first release)
                "b": rel_key,   # child = release
                "kind": "program",
            })
```

- [ ] **Step 5: Add the audit branch to `audit_ledger`**

In `tools/config_extract/generator.py`, in the per-entry loop of `audit_ledger`, immediately after Check 1 (the `if kind not in ("route", "program")` block, which ends ~line 363) and before the `if kind == "route":` block, add:

```python
        # Instruction-event layer: lock-order entries reuse kind="program" but are
        # oriented by the ELF lock decode, not route-graph reachability. Their
        # endpoints are portless core instruction events, so the standard
        # port-resolution + reachability checks do not apply -- validate the cite
        # structure and the ACQUIRE->RELEASE orientation instead.
        if cite.startswith("program_order:"):
            expected = f"program_order:{a}--core-locks-->{b}"
            if cite != expected:
                failures.append(
                    f"{label}: lock-order cite malformed (expected {expected!r})")
            if not (a.endswith("INSTR_LOCK_ACQUIRE_REQ")
                    and b.endswith("INSTR_LOCK_RELEASE_REQ")):
                failures.append(
                    f"{label}: lock-order entry must be ACQUIRE(a)->RELEASE(b)")
            continue
```

- [ ] **Step 6: Run tests to verify they pass + no regression**

Run: `cd tools && python -m pytest config_extract/test_generator_lock_order.py -q`
Expected: PASS (3 tests).
Run: `cd tools && python -m pytest test_config_extract_generator.py test_selfmodel.py -q`
Expected: PASS, no regressions (existing route/program edges unchanged; the new path is additive).

- [ ] **Step 7: Commit**

```bash
git add tools/inference/selfmodel.py tools/config_extract/generator.py tools/config_extract/test_generator_lock_order.py
git commit -m "feat(#140): emit oriented core lock-order candidate pair (instruction-event layer)"
```

---

## Self-Review

**1. Spec coverage:**
- Component 1 (core_relay helper) -> Task 1. ✓
- Component 2 (TileDump + build_dump re-decode + fixture regen) -> Task 2. ✓
- Component 3 (dump_model loader) -> Task 3. ✓
- Component 4 (fixture regen) -> Task 2 Step 7. ✓
- Component 5 (menu) + Component 6 (generator emission + event_map + audit) -> Task 4 (menu + generator land together per Global Constraints). ✓ Note: `event_map` needs NO change -- lock events already return `None` from `resolve_event_port` and the new path bypasses it; the spec's event_map bullet is "keep current behavior," so there is no code step (correct, not a gap).
- Testing (filtered lock-pair assertion, candidate_pairs, enumerate, audit) -> Task 4 tests. ✓
- Reuse `kind="program"` (not a new kind) -> enforced in Task 4 Step 4. ✓
- Orientation `min(acq) < min(rel)`, derived -> Task 1. ✓

**2. Placeholder scan:** No TBD/TODO/"similar to"/"handle edge cases". All code blocks are complete; the build_dump guard reproduces the real pattern verbatim. ✓

**3. Type consistency:** `aggregate_lock_order -> Option<(u32,u32)>` (Task 1) -> `LockOrderDump{acq_pc,rel_pc}` (Task 2) -> JSON `{acq_pc,rel_pc}` -> `LockOrder{acq_pc,rel_pc}` (Task 3) -> consumed by event-key construction (Task 4). Key format `col|row|0|NAME` with `col = tile.col + start_col` consistent between emission (Step 4) and tests. `kind="program"`, cite prefix `program_order:` consistent between emission (Step 4) and audit (Step 5) and tests. ✓
