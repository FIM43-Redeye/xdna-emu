# Trace Planner v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the heuristic trace feasibility planner with a pathfinder-driven planner that tries all (shim_col, channel) candidates in parallel, validates existing flows are unchanged, and picks the least-interference winner. Enables cross-column trace routing.

**Architecture:** The planner clones the MLIR module N times (one per candidate), injects trace flows via a shared helper, runs the pathfinder pass on each, extracts switch connections, compares to baseline, and picks the winner. The injector consumes the plan without making routing decisions. See `docs/plans/2026-03-05-trace-planner-v2-design.md` for full design.

**Tech Stack:** Python 3.13, mlir-aie Python API (Module, PassManager, find_ops, packetflow, tile, WireBundle), concurrent.futures for parallelism.

**Key file:** `tools/trace-inject.py` -- all changes are in this single file.

---

### Task 1: Add New Data Model

Replace `StrictTracePlan` and `ShimChannelState` with `TracePlan` and `CandidateResult`.

**Files:**
- Modify: `tools/trace-inject.py:690-714` (current dataclass region)

**Step 1: Replace the dataclasses**

Remove `ShimChannelState` (lines 691-702) and `StrictTracePlan` (lines 705-714).
Add at the same location:

```python
@dataclass
class CandidateResult:
    """Pathfinder result for one (shim_col, channel) candidate."""
    shim_col: int
    channel: int
    success: bool                      # pathfinder found a valid route
    existing_flows_intact: bool        # baseline connections unchanged
    failure_reason: str | None = None
    # Tiebreaker metrics (set when success=True)
    trace_connections_on_test_cols: int = 0
    total_trace_connections: int = 0


@dataclass
class TracePlan:
    """Result of trace route planning."""
    feasible: bool
    reason: str
    shim_col: int | None = None
    trace_channel: int | None = None
    candidates: list[CandidateResult] | None = None

    def to_dict(self) -> dict:
        """Serialize for JSON manifest output."""
        d = {
            "feasible": self.feasible,
            "reason": self.reason,
            "shim_col": self.shim_col,
            "trace_channel": self.trace_channel,
        }
        if self.candidates:
            d["candidates"] = [
                {
                    "shim_col": c.shim_col,
                    "channel": c.channel,
                    "success": c.success,
                    "existing_flows_intact": c.existing_flows_intact,
                    "failure_reason": c.failure_reason,
                    "trace_on_test_cols": c.trace_connections_on_test_cols,
                    "total_trace_connections": c.total_trace_connections,
                }
                for c in self.candidates
            ]
        return d
```

**Step 2: Verify no import breakage**

Run: `python3 -c "import sys; sys.path.insert(0, 'tools'); from importlib import import_module"`

This just verifies the file parses. The old `StrictTracePlan` references will
break -- that is expected, we fix them in Task 5.

**Step 3: Commit**

```
git add tools/trace-inject.py
git commit -m "refactor: replace StrictTracePlan with TracePlan + CandidateResult"
```

---

### Task 2: IR-Walking Helpers (Replace Regex)

Replace regex-based `find_used_packet_ids()` with IR-walking version. Add
`find_tile_op()` and `find_device_with_sequence()` helpers that both the
planner and injector will share.

**Files:**
- Modify: `tools/trace-inject.py`

**Step 1: Add IR-walking helpers**

Insert after the new dataclasses (after `TracePlan`), before
`strict_analyze_feasibility`:

```python
def find_device_with_sequence(module):
    """Find the DeviceOp containing a runtime_sequence.

    Multi-device modules (e.g. ctrl_packet_reconfig) have @base (empty)
    and @main (full design).  Returns the device with runtime_sequence,
    or the first device if none has one.
    """
    from aie.extras.util import find_ops  # type: ignore
    import aie.dialects.aie as aiedialect  # type: ignore

    device_ops = find_ops(
        module.operation,
        lambda o: isinstance(o.opview, aiedialect.DeviceOp),
    )
    if not device_ops:
        raise RuntimeError("No aie.device op found in MLIR")

    for dop in device_ops:
        rs = find_ops(
            dop,
            lambda o: isinstance(o.opview, aiedialect.RuntimeSequenceOp),
        )
        if rs:
            return dop.opview
    return device_ops[0].opview


def find_tile_op(device_op, col: int, row: int):
    """Find an existing TileOp for (col, row) within a device, or None."""
    from aie.extras.util import find_ops  # type: ignore
    import aie.dialects.aie as aiedialect  # type: ignore

    tile_ops = find_ops(
        device_op.operation,
        lambda o: isinstance(o.opview, aiedialect.TileOp),
    )
    for t in tile_ops:
        top = t.opview
        if top.col.value == col and top.row.value == row:
            return top
    return None


def find_used_packet_ids_ir(device_op) -> set[int]:
    """Scan a parsed device op for all packet IDs in use.

    Walks PacketFlowOp operations in the IR instead of regex on text.
    """
    from aie.extras.util import find_ops  # type: ignore
    import aie.dialects.aie as aiedialect  # type: ignore

    ids: set[int] = set()
    pkt_ops = find_ops(
        device_op.operation,
        lambda o: isinstance(o.opview, aiedialect.PacketFlowOp),
    )
    for pfop in pkt_ops:
        ids.add(pfop.opview.ID.value)
    return ids


def classify_tiles(device_op) -> tuple[list[int], list[tuple], set[int]]:
    """Classify tiles in a device into shim columns, trace targets, and test columns.

    Returns:
        shim_cols: list of column indices with shim tiles (row 0)
        trace_targets: list of (col, row, trace_port) tuples for traceable tiles
            Core tiles produce two entries: (col, row, 0) for Core, (col, row, 1) for Mem
        test_cols: set of column indices used by the test (non-shim tiles with uses)
    """
    from aie.extras.util import find_ops  # type: ignore
    import aie.dialects.aie as aiedialect  # type: ignore
    from aie.dialects.aie import get_target_model as _gtm  # type: ignore

    tm = _gtm(device_op.operation.attributes["device"])

    tile_ops = find_ops(
        device_op.operation,
        lambda o: isinstance(o.opview, aiedialect.TileOp),
    )

    shim_cols = []
    trace_targets = []
    test_cols = set()

    for t in tile_ops:
        top = t.opview
        col, row = top.col.value, top.row.value
        if row == 0:
            if col not in shim_cols:
                shim_cols.append(col)
            continue
        if len(list(top.result.uses)) == 0:
            continue
        test_cols.add(col)
        if tm.is_core_tile(col, row):
            trace_targets.append((col, row, 0))  # Core trace port
            trace_targets.append((col, row, 1))  # Mem trace port
        elif tm.is_mem_tile(col, row):
            trace_targets.append((col, row, 0))  # MemTile trace port

    return shim_cols, trace_targets, test_cols
```

**Step 2: Verify the helpers parse**

Run: `python3 -c "exec(open('tools/trace-inject.py').read())"`

Expected: no syntax errors.

**Step 3: Commit**

```
git add tools/trace-inject.py
git commit -m "refactor: add IR-walking helpers (find_device, find_tile, classify_tiles)"
```

---

### Task 3: Shared Flow Helper `add_trace_flows()`

The function both the planner and injector will call to declare trace packet
flows on a parsed module.

**Files:**
- Modify: `tools/trace-inject.py`

**Step 1: Add `add_trace_flows()`**

Insert after `classify_tiles()`:

```python
def add_trace_flows(
    device_op,
    shim_col: int,
    trace_channel: int,
    trace_targets: list[tuple[int, int, int]],
    used_packet_ids: set[int],
) -> tuple[int, object]:
    """Declare trace packet flows from tiles to a target shim.

    Injects:
    - aie.tile declaration for the shim (if not already present)
    - aie.packet_flow ops routing each tile's Trace ports to the shim DMA

    Does NOT configure trace registers, timers, DMA, or buffers.

    Args:
        device_op: the DeviceOp to modify
        shim_col: target shim tile column
        trace_channel: S2MM DMA channel on the shim
        trace_targets: list of (col, row, trace_port) -- explicit, no guessing
        used_packet_ids: existing packet IDs to avoid

    Returns:
        (trace_id_start, shim_tile_op) -- first packet ID used and the shim TileOp
    """
    from aie.ir import InsertionPoint  # type: ignore
    from aie.dialects.aie import packetflow, tile, WireBundle  # type: ignore

    device_block = device_op.body_region.blocks[0]

    # Find or create shim tile
    shim_tile = find_tile_op(device_op, shim_col, 0)
    if shim_tile is None:
        with InsertionPoint.at_block_begin(device_block):
            shim_tile = tile(shim_col, 0)

    # Allocate packet IDs
    trace_id_start = choose_trace_id_start(used_packet_ids, len(trace_targets))

    # Build a lookup from (col, row) to existing TileOp
    from aie.extras.util import find_ops  # type: ignore
    import aie.dialects.aie as aiedialect  # type: ignore
    tile_ops = find_ops(
        device_op.operation,
        lambda o: isinstance(o.opview, aiedialect.TileOp),
    )
    tile_lookup = {}
    for t in tile_ops:
        top = t.opview
        tile_lookup[(top.col.value, top.row.value)] = top

    # Declare packet flows
    with InsertionPoint.at_block_terminator(device_block):
        for i, (col, row, trace_port) in enumerate(trace_targets):
            p_id = trace_id_start + i
            src_tile = tile_lookup.get((col, row))
            if src_tile is None:
                raise RuntimeError(
                    f"Tile ({col},{row}) referenced in trace_targets "
                    f"but not found in MLIR"
                )
            packetflow(
                p_id, src_tile, WireBundle.Trace, trace_port,
                dests={"dest": shim_tile, "port": WireBundle.DMA,
                       "channel": trace_channel},
                keep_pkt_header=True,
            )

    return trace_id_start, shim_tile
```

**Step 2: Commit**

```
git add tools/trace-inject.py
git commit -m "feat: add_trace_flows() shared helper for planner and injector"
```

---

### Task 4: Connection Extractor and Pathfinder Runner

Functions to run the pathfinder on a module and extract switch connections
for baseline comparison.

**Files:**
- Modify: `tools/trace-inject.py`

**Step 1: Add `run_pathfinder()` and `extract_connections()`**

Insert after `add_trace_flows()`:

```python
def run_pathfinder(module) -> bool:
    """Run the pathfinder pass on a module. Returns True if successful.

    Mutates the module in place: FlowOps and PacketFlowOps are replaced
    with concrete SwitchboxOp/ConnectOp routing.
    """
    from aie.passmanager import PassManager  # type: ignore

    pipeline = "builtin.module(aie.device(aie-create-pathfinder-flows))"
    try:
        pm = PassManager.parse(pipeline)
        pm.run(module.operation)
        return True
    except Exception:
        return False


def extract_connections(module) -> set[tuple]:
    """Extract all switch connections from a routed module.

    After the pathfinder runs, the module contains SwitchboxOp with
    ConnectOp inside them. This extracts each connection as a tuple:
        (col, row, src_bundle, src_channel, dst_bundle, dst_channel)

    Returns a set for easy comparison via issubset().
    """
    from aie.extras.util import find_ops  # type: ignore
    import aie.dialects.aie as aiedialect  # type: ignore

    connections = set()

    switchbox_ops = find_ops(
        module.operation,
        lambda o: isinstance(o.opview, aiedialect.SwitchboxOp),
    )

    for sb_op in switchbox_ops:
        sb = sb_op.opview
        col = sb.col.value
        row = sb.row.value

        connect_ops = find_ops(
            sb_op,
            lambda o: isinstance(o.opview, aiedialect.ConnectOp),
        )
        for cop in connect_ops:
            c = cop.opview
            connections.add((
                col, row,
                int(c.source_bundle), int(c.source_channel),
                int(c.dest_bundle), int(c.dest_channel),
            ))

    return connections
```

**Step 2: Commit**

```
git add tools/trace-inject.py
git commit -m "feat: run_pathfinder() and extract_connections() for routing comparison"
```

---

### Task 5: The Planner -- `plan_trace_route()`

The core algorithm: tries all candidates in parallel, compares to baseline,
picks winner.

**Files:**
- Modify: `tools/trace-inject.py`

**Step 1: Add the candidate evaluation worker**

Insert after `extract_connections()`. This function runs in a subprocess
(MLIR Context is not thread-safe):

```python
def _evaluate_candidate(
    mlir_asm: str,
    shim_col: int,
    channel: int,
    trace_targets: list[tuple[int, int, int]],
    baseline_connections: set[tuple],
    test_cols: set[int],
) -> dict:
    """Evaluate one (shim_col, channel) candidate. Runs in subprocess.

    Returns a dict (not CandidateResult) for pickling across process boundary.
    """
    from aie.ir import Context, Location, Module  # type: ignore

    result = {
        "shim_col": shim_col,
        "channel": channel,
        "success": False,
        "existing_flows_intact": False,
        "failure_reason": None,
        "trace_connections_on_test_cols": 0,
        "total_trace_connections": 0,
    }

    try:
        with Context(), Location.unknown():
            module = Module.parse(mlir_asm)
            device_op = find_device_with_sequence(module)
            used_ids = find_used_packet_ids_ir(device_op)

            add_trace_flows(
                device_op, shim_col, channel, trace_targets, used_ids,
            )

            if not run_pathfinder(module):
                result["failure_reason"] = "pathfinder_failed"
                return result

            result["success"] = True
            candidate_conns = extract_connections(module)

            # Check baseline preservation
            result["existing_flows_intact"] = baseline_connections.issubset(
                candidate_conns
            )
            if not result["existing_flows_intact"]:
                result["failure_reason"] = "existing_flows_rerouted"

            # Score: trace connections added
            trace_additions = candidate_conns - baseline_connections
            result["trace_connections_on_test_cols"] = sum(
                1 for (col, *_rest) in trace_additions if col in test_cols
            )
            result["total_trace_connections"] = len(trace_additions)

    except Exception as e:
        result["failure_reason"] = f"error: {str(e)[:120]}"

    return result
```

**Step 2: Add `plan_trace_route()`**

Insert after `_evaluate_candidate()`:

```python
def plan_trace_route(mlir_text: str) -> TracePlan:
    """Plan trace routing by trialing all (shim, channel) candidates.

    Uses the pathfinder as oracle: clones the module for each candidate,
    injects trace flows, runs pathfinder, and verifies existing flows
    are unchanged. Picks the candidate with least interference.
    """
    from aie.ir import Context, Location, Module  # type: ignore
    from aie.dialects.aie import get_target_model as _gtm  # type: ignore
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with Context(), Location.unknown():
        try:
            module = Module.parse(mlir_text)
        except Exception as e:
            return TracePlan(
                False,
                f"MLIR parse error: {str(e).splitlines()[0][:120]}",
            )

        device_op = find_device_with_sequence(module)
        tm = _gtm(device_op.operation.attributes["device"])

        shim_cols, trace_targets, test_cols = classify_tiles(device_op)

        if not trace_targets:
            return TracePlan(False, "no used tiles to trace")

        # Run pathfinder on unmodified module to get baseline connections.
        baseline_asm = module.operation.get_asm(enable_debug_info=True)

        baseline_module = Module.parse(baseline_asm)
        if not run_pathfinder(baseline_module):
            return TracePlan(False, "pathfinder failed on unmodified MLIR")
        baseline_connections = extract_connections(baseline_module)

    # Generate candidates: all device columns x 2 channels.
    num_cols = tm.columns()
    candidates_to_try = []
    for col in range(num_cols):
        for ch in range(2):  # S2MM channels 0 and 1
            candidates_to_try.append((col, ch))

    # Evaluate all candidates in parallel.
    results = []
    with ProcessPoolExecutor(max_workers=min(len(candidates_to_try), 10)) as pool:
        futures = {
            pool.submit(
                _evaluate_candidate,
                baseline_asm,
                col, ch,
                trace_targets,
                baseline_connections,
                test_cols,
            ): (col, ch)
            for col, ch in candidates_to_try
        }
        for future in as_completed(futures):
            rd = future.result()
            results.append(CandidateResult(
                shim_col=rd["shim_col"],
                channel=rd["channel"],
                success=rd["success"],
                existing_flows_intact=rd["existing_flows_intact"],
                failure_reason=rd["failure_reason"],
                trace_connections_on_test_cols=rd["trace_connections_on_test_cols"],
                total_trace_connections=rd["total_trace_connections"],
            ))

    # Pick winner: must succeed with existing flows intact.
    winners = [
        c for c in results
        if c.success and c.existing_flows_intact
    ]

    if not winners:
        reasons = set(c.failure_reason or "unknown" for c in results)
        return TracePlan(
            feasible=False,
            reason=f"no valid candidate: {', '.join(sorted(reasons))}",
            candidates=results,
        )

    # Tiebreaker: least trace on test cols, then least total connections.
    best = min(winners, key=lambda c: (
        c.trace_connections_on_test_cols,
        c.total_trace_connections,
    ))

    return TracePlan(
        feasible=True,
        reason="ok",
        shim_col=best.shim_col,
        trace_channel=best.channel,
        candidates=results,
    )
```

**Step 3: Verify the planner parses**

Run: `python3 -c "exec(open('tools/trace-inject.py').read()); print('ok')"`

Expected: prints "ok" (no syntax errors).

**Step 4: Commit**

```
git add tools/trace-inject.py
git commit -m "feat: plan_trace_route() -- pathfinder-driven candidate evaluation"
```

---

### Task 6: Rewire `inject_trace()` to Consume `TracePlan`

Remove shim/channel selection logic from `inject_trace()`. It now receives
the plan and calls `add_trace_flows()` for the real module.

**Files:**
- Modify: `tools/trace-inject.py:1103-1304` (inject_trace flow creation section)

**Step 1: Simplify `inject_trace()` signature**

Change the signature to accept a `TracePlan`:

```python
def inject_trace(
    mlir_text: str,
    trace_size: int,
    plan: TracePlan,
    events_config: dict | None = None,
    tile_filter: list[tuple[int, int, str]] | None = None,
) -> tuple[str, dict]:
```

**Step 2: Replace shim/channel selection with plan consumption**

Inside `inject_trace()`, after parsing the module and finding the device op:

- Remove the shim selection block (lines 1191-1213 in current code --
  `shim_tiles` list, `min(shim_tiles, ...)`)
- Use `plan.shim_col` and `plan.trace_channel` instead
- Replace the manual packet flow insertion block (lines 1264-1303) with
  a call to `add_trace_flows()`

**Step 3: Update tile classification to use explicit targets**

Replace the implicit "first visit = Core, second = Mem" pattern with
explicit `(tile_op, trace_port)` pairs. The register config loop needs
TileOp references, so build a parallel list:

```python
trace_targets = []
tiles_for_config = []  # parallel: (TileOp, trace_port) for register config

for t in tile_ops_to_trace:
    col, row = t.opview.col.value, t.opview.row.value
    if tm.is_core_tile(col, row):
        trace_targets.append((col, row, 0))
        tiles_for_config.append((t.opview, 0))
        trace_targets.append((col, row, 1))
        tiles_for_config.append((t.opview, 1))
    elif tm.is_mem_tile(col, row):
        trace_targets.append((col, row, 0))
        tiles_for_config.append((t.opview, 0))
```

**Step 4: Update the register config loop**

Replace the implicit visit-counting with the explicit list:

```python
with InsertionPoint.at_block_begin(seq_block):
    for i, (tile_op, trace_port) in enumerate(tiles_for_config):
        p_id = trace_id_start + i
        tc, tr = int(tile_op.col), int(tile_op.row)

        if tm.is_core_tile(tc, tr) and trace_port == 0:
            configure_coretile_tracing_aie2(...)
        elif tm.is_core_tile(tc, tr) and trace_port == 1:
            configure_coremem_tracing_aie2(...)
        elif tm.is_mem_tile(tc, tr):
            configure_memtile_tracing_aie2(...)
        elif tm.is_shim_noc_or_pl_tile(tc, tr):
            configure_shimtile_tracing_aie2(...)
```

**Step 5: Commit**

```
git add tools/trace-inject.py
git commit -m "refactor: inject_trace() consumes TracePlan, uses add_trace_flows()"
```

---

### Task 7: Rewire `main()` -- Planner + Injector Integration

Update `main()` to use the new planner and pass the plan to the injector.

**Files:**
- Modify: `tools/trace-inject.py:2060-2304` (main function)

**Step 1: Update `--plan-only` to use new planner**

Replace the `plan_trace_passes` call (line 2125):

```python
if args.plan_only:
    mlir_text = get_mlir_text(test_dir, source_type, device)
    plan = plan_trace_route(mlir_text)
    print(json.dumps(plan.to_dict(), indent=2))
    sys.exit(0)
```

`--plan-only` skips quarantine check by design -- it is for diagnostics.

**Step 2: Replace `strict_analyze_feasibility()` with `plan_trace_route()`**

Replace lines 2185-2233 with the new planner call and diagnostic output.
Print full candidate table to stderr showing scores and winner.

**Step 3: Pass plan to `inject_trace()`**

```python
traced_mlir, manifest_partial = inject_trace(
    mlir_text, args.trace_size, plan,
    events_config=events_config,
    tile_filter=tile_filter,
)
```

**Step 4: Update manifest**

```python
manifest["planner"] = plan.to_dict()
```

**Step 5: Commit**

```
git add tools/trace-inject.py
git commit -m "feat: wire plan_trace_route() into main pipeline"
```

---

### Task 8: Remove Dead Code

Clean up the old planner and regex helpers that are no longer used.

**Files:**
- Modify: `tools/trace-inject.py`

**Step 1: Remove old code**

Delete these functions entirely:
- `strict_analyze_feasibility()` (approx lines 717-1032)
- `find_used_packet_ids()` (regex version, lines 135-155) -- replaced by
  `find_used_packet_ids_ir()`
- `plan_trace_passes()` (line 578) if no longer referenced

Check `has_existing_trace()` (line 1094) -- this is different from
`has_existing_trace_flows()` and is still used by `main()`. Keep it.

Remove unused stream switch constants (lines 26-32) if nothing references
them:
```python
SS_NUM_ARBITERS = 6
SS_MSELS_PER_ARBITER = 4
SS_SLOTS_PER_SLAVE_PORT = 4
SS_PACKET_ID_BITS = 5
```

Search each name before deleting.

**Step 2: Verify**

Run: `python3 tools/trace-inject.py --help`

Expected: prints usage without errors.

**Step 3: Commit**

```
git add tools/trace-inject.py
git commit -m "cleanup: remove old heuristic planner and regex helpers"
```

---

### Task 9: End-to-End Smoke Test

Verify the new planner works on real tests.

**Files:**
- No code changes -- testing only

**Step 1: Run `--plan-only` on a simple test**

```
python3 tools/trace-inject.py \
    ../mlir-aie/test/npu-xrt/add_one_using_dma/ \
    --output /tmp/claude-1000/trace-plan-test \
    --plan-only
```

Expected: JSON with `"feasible": true`, all 10 candidates evaluated,
winner has low `trace_on_test_cols`.

**Step 2: Run full injection**

```
python3 tools/trace-inject.py \
    ../mlir-aie/test/npu-xrt/add_one_using_dma/ \
    --output /tmp/claude-1000/trace-plan-test
```

Expected: writes `aie_traced.mlir` and `manifest.json` with planner data.

**Step 3: Run `--plan-only` on a quarantined test**

```
python3 tools/trace-inject.py \
    ../mlir-aie/test/npu-xrt/ctrl_packet_reconfig/ \
    --output /tmp/claude-1000/trace-plan-quarantine \
    --plan-only
```

Expected: shows whether new planner can handle it or reports infeasible
with candidate details.

**Step 4: Verify Rust tests unaffected**

```
TMPDIR=/tmp/claude-1000 cargo test --lib
```

Expected: all tests pass (Python-only changes).

---

### Task 10: Investigate Parallelism (Thread vs Process)

The design uses `ProcessPoolExecutor` because MLIR Contexts may not be
thread-safe. This needs verification.

**Files:**
- Modify: `tools/trace-inject.py` (only if threads work)

**Step 1: Test thread safety**

Write a quick test creating separate MLIR Contexts in threads. If it
works without crashes or wrong results, switch to `ThreadPoolExecutor`
in `plan_trace_route()` for better performance (no pickling overhead).

**Step 2: If threads work, update and commit**

```
git add tools/trace-inject.py
git commit -m "perf: use ThreadPoolExecutor for parallel pathfinder evaluation"
```

---

## Task Dependency Graph

```
Task 1 (data model)
    |
Task 2 (IR helpers)
    |
Task 3 (add_trace_flows)
    |
Task 4 (extract_connections + run_pathfinder)
    |
Task 5 (plan_trace_route)
    |
Task 6 (rewire inject_trace)
    |
Task 7 (rewire main)
    |
Task 8 (remove dead code)
    |
Task 9 (smoke test)
    |
Task 10 (parallelism investigation)
```

All tasks are strictly sequential -- each builds on the previous.

## Risk Notes

- **PassManager API**: Confirmed via research but untested live. Task 9 is
  the first real exercise. If the pass name or API differs, adjust Task 4.

- **SwitchboxOp / ConnectOp**: `extract_connections()` assumes these op types
  exist after pathfinder runs. If the pathfinder produces different IR (e.g.,
  `AMSelOp`, `MasterSetOp`), inspect the IR in Task 9 and adjust extraction.

- **Process pickling**: `_evaluate_candidate()` receives and returns plain
  dicts and sets (no MLIR objects) to avoid pickling issues. The MLIR module
  is passed as an ASM string and re-parsed in the subprocess.

- **inject_trace_per_column()**: Has similar shim/flow logic that should
  eventually also use `add_trace_flows()` and `TracePlan`. Deferred -- it
  is a separate code path, refactor after the main path is validated.
