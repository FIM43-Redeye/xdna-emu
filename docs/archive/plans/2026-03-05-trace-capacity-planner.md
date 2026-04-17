# Trace Capacity Planner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automatically determine how many trace injection passes are needed for a given binary, partition tiles into routable groups, and execute multi-pass sweeps when a single pass can't fit everything.

**Architecture:** A static analysis function in `trace-inject.py` parses the MLIR, extracts existing flows and tile topology, then computes per-switchbox routing capacity using device model data and hardware constants. The planner outputs a list of tile groups, each guaranteed to fit within the stream switch constraints. `trace-sweep.py` calls the planner once, then runs one injection+compile cycle per group.

**Tech Stack:** Python (mlir-aie API + regex MLIR parsing), JSON device model (`tools/aie-device-models.json`), existing trace-inject/sweep infrastructure.

---

## Background

### Hardware Constraints (AIE2 Stream Switch)

Per switchbox (all tile types):
- **6 arbiters x 4 MSEL = 24 amsel slots** total
- **4 rule slots per slave port** (each with 5-bit mask, 5-bit value, 2-bit MSEL, 3-bit arbiter)
- **5-bit packet ID space** (0-31)
- Circuit-switched flows consume a channel exclusively (`MAX_CIRCUIT_STREAM_CAPACITY = 1`)
- Packet-switched flows share channels (`MAX_PACKET_STREAM_CAPACITY = 32`)

Per tile type (from `aie-device-models.json`):
- Core tile: South has 4 master / 6 slave ports; Trace has 2 slave ports (Core + Mem)
- MemTile: South has 4 master / 6 slave ports; Trace has 1 slave port
- Shim: North has 6 master / 4 slave ports; shim_mux DMA has 2 master / 2 slave

### The Bottleneck

Trace packets flow south from each tile to the destination shim. At each
intermediate switchbox, trace packets enter via a South slave port and exit
via a South master port. The constraints:

1. **Master port limit**: each tile has a fixed number of South master ports
   (4 for core/memtile). A master port can carry multiple packet-switched
   flows but only one circuit-switched flow.
2. **Slave rule slots**: 4 slots per slave port. If existing circuit/packet
   flows already consume slots, fewer remain for trace.
3. **Arbiter exhaustion**: 24 amsel slots total per switchbox. Each distinct
   routing decision (different set of master destinations) needs its own slot.
4. **Packet ID false matches**: mask/value rules on the same slave port must
   not accidentally match unintended packet IDs (including implicit ID 0 from
   circuit-switched flows).

### Existing Infrastructure

- `find_used_packet_ids(mlir_text)` -- scans for existing packet flow IDs
- `choose_trace_id_start(used_ids, num_trace_ids)` -- picks collision-avoiding IDs
- `inject_trace(mlir_text, ...)` -- injects routing + config for a set of tiles
- `compile_base_trace(test_dir, ...)` in trace-sweep.py -- injection + compilation
- Device model at `tools/aie-device-models.json` with per-device port counts

---

## Task 1: Flow Extraction from MLIR

Extract all existing circuit-switched and packet-switched flows from MLIR text
so the planner knows what routing resources are already consumed.

**Files:**
- Modify: `tools/trace-inject.py` (add `extract_flows()` after `find_used_packet_ids()`)

**Step 1: Write the test**

Add a test block at the bottom of `trace-inject.py` (guarded by `if __name__ == "__main__"`)
or create a small test script. For now, we'll verify via manual invocation.

Define the data structures and parsing function:

```python
@dataclass
class FlowInfo:
    """A single flow (circuit or packet) extracted from MLIR."""
    flow_type: str          # "circuit" or "packet"
    src_col: int
    src_row: int
    src_bundle: str         # "DMA", "Trace", "Core", "TileControl", etc.
    src_channel: int
    dst_col: int
    dst_row: int
    dst_bundle: str
    dst_channel: int
    packet_id: int | None   # None for circuit flows


def extract_flows(mlir_text: str) -> list[FlowInfo]:
    """Extract all circuit and packet flows from MLIR text.

    Parses:
    - aie.flow(%tile_C_R, Bundle : Ch, %tile_C2_R2, Bundle2 : Ch2)
    - aie.packet_flow(ID) { aie.packet_source<...> aie.packet_dest<...> }

    Returns a list of FlowInfo describing each flow.
    """
```

The regex patterns:
- Circuit: `aie.flow\(%\w+,\s*(\w+)\s*:\s*(\d+)\s*,\s*%\w+,\s*(\w+)\s*:\s*(\d+)\)`
  with tile names resolved via a `%name -> (col, row)` map built from `aie.tile` declarations.
- Packet: multi-line `aie.packet_flow(ID) { ... source ... dest ... }`

**Step 2: Run to verify parsing works**

```bash
python3 -c "
from tools import ...  # or inline test
text = open('.../add_one_ctrl_packet_4_cores/aie.mlir').read()
flows = extract_flows(text)
for f in flows: print(f)
"
```

Expected: 4 circuit flows + 4 packet flows for the 4_cores test.

**Step 3: Commit**

```bash
git add tools/trace-inject.py
git commit -m "feat: extract_flows() parses circuit and packet flows from MLIR"
```

---

## Task 2: Switchbox Capacity Model

Build a model of each switchbox's routing capacity by walking the device model
and subtracting resources consumed by existing flows.

**Files:**
- Modify: `tools/trace-inject.py` (add `SwitchboxCapacity` class and `build_capacity_map()`)

**Step 1: Define the capacity model**

```python
@dataclass
class SwitchboxCapacity:
    """Routing capacity for one switchbox after existing flows are accounted for."""
    col: int
    row: int
    tile_type: str                          # "core", "mem_tile", "shim_noc"
    south_master_total: int                 # from device model
    south_master_circuit: int               # consumed by circuit flows
    south_master_available: int             # total - circuit (packet flows share)
    south_slave_slots_total: int            # 4 per port * num south slave ports
    south_slave_slots_used: int             # consumed by existing packet rules
    amsel_total: int                        # 24 (6 arbiters * 4 MSEL)
    amsel_used: int                         # consumed by existing flows
    trace_slave_ports: int                  # 2 for core, 1 for memtile/shim


def build_capacity_map(
    flows: list[FlowInfo],
    device_name: str,
    device_model_path: Path = Path("tools/aie-device-models.json"),
) -> dict[tuple[int, int], SwitchboxCapacity]:
    """Build a per-tile capacity map by subtracting existing flow usage.

    For each tile in the device, counts how many South master ports and
    slave rule slots are consumed by the extracted flows, then computes
    remaining capacity.
    """
```

Key logic:
- Load device model JSON, get port counts per tile type
- For each flow, trace the path from source to destination (southward for
  most data flows) and decrement capacity at each intermediate switchbox
- Path tracing is simplified: flows go south within a column, then east/west
  at row 0 to reach a different column's shim. For each hop, consume one
  South master port (circuit) or one slave rule slot (packet).

Hardware constants (configurable, not hardcoded):

```python
SS_NUM_ARBITERS = 6
SS_MSELS_PER_ARBITER = 4
SS_SLOTS_PER_SLAVE_PORT = 4
SS_PACKET_ID_BITS = 5
```

**Step 2: Test against known test cases**

Verify that `build_capacity_map` for `add_one_ctrl_packet_4_cores` shows
reduced South capacity at tiles (0,2)-(0,5) due to the 4 circuit flows
and 4 packet flows.

**Step 3: Commit**

```bash
git commit -m "feat: switchbox capacity model from device model + existing flows"
```

---

## Task 3: Trace Route Feasibility Check

Given a set of tiles to trace and a destination shim, determine whether the
trace flows can be routed without exceeding switchbox capacity.

**Files:**
- Modify: `tools/trace-inject.py` (add `check_trace_feasibility()`)

**Step 1: Implement feasibility check**

```python
def check_trace_feasibility(
    tiles_to_trace: list[tuple[int, int, str]],  # (col, row, "core"/"mem"/"memtile")
    shim_col: int,
    capacity_map: dict[tuple[int, int], SwitchboxCapacity],
    num_existing_packet_ids: int,
) -> tuple[bool, str]:
    """Check whether the given trace tiles can be routed to the shim.

    Walks the path from each tile southward to the shim, checking:
    1. South master port availability at each hop
    2. Slave rule slot availability (4 slots per port, shared by packet flows)
    3. Total packet ID count fits in 5-bit space
    4. Amsel slot availability

    Returns (feasible, reason) where reason explains the bottleneck if infeasible.
    """
```

The walk: for a trace tile at (col, row), the trace packet goes:
- South through (col, row-1), (col, row-2), ..., (col, 1), (col, 0)
- If shim_col != col, then east/west through (col+/-1, 0), ..., (shim_col, 0)

At each hop, one additional packet-switched flow uses:
- 1 slave rule slot on the South (or East/West) slave port
- 1 amsel slot (shared with other trace flows going to the same master)

All trace flows share the same destination (shim DMA:1), so they can share
master ports and amsel entries via mask/value grouping -- but each needs a
distinguishable packet ID at any slave port where they converge.

The key check: at the most congested intermediate tile, how many trace flows
pass through its South port? That count plus existing flows must fit in 4 slots.

But trace flows with the same destination can be GROUPED under one mask/value
rule if their IDs share enough bit structure. So the real limit is:

```
max_trace_flows_per_slave = (4 - existing_rules) * flows_per_rule
```

Where `flows_per_rule` depends on mask wildcarding. Conservatively, assume
each rule can cover up to 4 IDs (2 wildcard bits). So:

```
conservative_capacity = (4 - existing_rules) * 4
```

This is conservative but avoids needing to simulate the pathfinder.

**Step 2: Test**

```python
# 4_cores: 4 existing packet flows through (0,0) South slave
# Available slots at (0,0): 4 - existing = depends on how many rules
# Trace adds 8 flows (4 core + 4 mem) through column 0
# Should report: infeasible if going to shim col 0, feasible if col 2
```

**Step 3: Commit**

```bash
git commit -m "feat: trace route feasibility checker with conservative capacity estimate"
```

---

## Task 4: Tile Partitioner (The Planner)

The main planning function: given all traceable tiles, partition them into
groups where each group fits in a single injection pass.

**Files:**
- Modify: `tools/trace-inject.py` (add `plan_trace_passes()`)

**Step 1: Implement the planner**

```python
@dataclass
class TracePlan:
    """Plan for multi-pass trace injection."""
    passes: list[TracePass]
    total_tiles: int
    reason: str              # "single_pass" or "multi_pass: <explanation>"


@dataclass
class TracePass:
    """One injection pass: a set of tiles routed to a specific shim."""
    tiles: list[tuple[int, int, str]]   # (col, row, module_type)
    shim_col: int
    packet_id_start: int
    estimated_flows: int


def plan_trace_passes(
    mlir_text: str,
    device_name: str = "auto",
    device_model_path: Path = Path("tools/aie-device-models.json"),
) -> TracePlan:
    """Analyze MLIR and plan how many trace injection passes are needed.

    Algorithm:
    1. Extract tiles and existing flows from MLIR
    2. Build switchbox capacity map
    3. For each available shim (sorted by fewest existing uses):
       a. Try fitting ALL tiles -> check feasibility
       b. If infeasible, try core-only (no Mem module)
       c. If still infeasible, binary-partition by tile type:
          - First pass: core tiles (Core module only)
          - Second pass: core tiles (Mem module only)
          - Third pass: memtiles
       d. If a single tile can't route, report error
    4. Return the partition as a TracePlan
    """
```

Grouping strategy (by tile type, as requested):
1. **Core modules** first (most important for instruction-level analysis)
2. **Mem modules** second (DMA/lock stall visibility)
3. **MemTile modules** third (inter-tile data flow)
4. **Shim modules** last (host interface timing)

Within each group, pack as many tiles as the capacity allows. If a group
doesn't fit, split it in half (by column or by row).

**Step 2: Test on known cases**

| Test | Expected |
|------|----------|
| `add_one_using_dma` (1 core tile, no packet flows) | 1 pass, all tiles |
| `add_one_ctrl_packet_4_cores` (4 cores, 4 ctrl pkts) | 1 pass via shim col 2 |
| `add_one_ctrl_packet_col_overlay` (5 tiles, 3 busy shims) | 2+ passes |
| `packet_flow_fanout` (multi-column) | 1 pass (spread across columns) |

**Step 3: Commit**

```bash
git commit -m "feat: plan_trace_passes() partitions tiles into routable groups"
```

---

## Task 5: Integrate Planner into trace-inject.py CLI

Add a `--plan-only` flag that outputs the plan as JSON without modifying MLIR.
Add a `--tiles` flag that accepts a subset of tiles (for multi-pass execution).

**Files:**
- Modify: `tools/trace-inject.py` (update `main()` and `inject_trace()`)

**Step 1: Add CLI flags**

```python
parser.add_argument("--plan-only", action="store_true",
    help="Output a trace plan JSON without injecting. "
         "Shows how many passes are needed and which tiles per pass.")
parser.add_argument("--tiles", type=str, default=None,
    help="Comma-separated list of tiles to trace (col,row:module). "
         "Example: '0,2:core,0,2:mem,0,3:core'. "
         "When omitted, traces all used tiles.")
```

When `--plan-only`:
- Call `plan_trace_passes()`
- Print JSON plan to stdout
- Exit without modifying MLIR

When `--tiles` is specified:
- Filter `tiles_to_trace` to only the listed tiles
- Use the specified module types (core vs mem) instead of auto-duplication

**Step 2: Test**

```bash
python3 tools/trace-inject.py \
    /path/to/add_one_ctrl_packet_col_overlay \
    --plan-only 2>/dev/null | python3 -m json.tool
```

Expected: JSON with 2+ passes.

**Step 3: Commit**

```bash
git commit -m "feat: --plan-only and --tiles flags for multi-pass trace injection"
```

---

## Task 6: Multi-Pass Sweep Orchestration

Update `trace-sweep.py` to call the planner and execute multiple
injection+compile cycles when needed.

**Files:**
- Modify: `tools/trace-sweep.py` (update `compile_base_trace()` and `main()`)

**Step 1: Add plan-aware compilation**

Replace the single `compile_base_trace()` call with:

```python
def compile_trace_passes(
    test_dir: Path,
    output_dir: Path,
    trace_size: int,
) -> list[tuple[Path, Path]]:
    """Plan and compile trace passes.

    1. Run trace-inject.py --plan-only to get the plan
    2. For each pass in the plan:
       a. Run trace-inject.py --tiles=<pass_tiles> --output=<pass_dir>
       b. Compile the traced MLIR
    3. Return list of (traced_dir, manifest_path) per pass

    Falls back to single-pass if planner reports single_pass.
    """
```

Each pass gets its own subdirectory under `output_dir`:
```
sweep/
  pass_0/        # first group of tiles
    base/        # injection + compile
    batch_00/    # event variations
    batch_01/
  pass_1/        # second group (if needed)
    base/
    batch_00/
```

**Step 2: Update batch preparation and result merging**

Batches within a pass work exactly as before (event variation on a shared
xclbin). The trace comparison at the end needs to merge results across passes
-- each pass covers different tiles, so their results are disjoint and can be
concatenated.

**Step 3: Test**

Run the full sweep on `add_one_ctrl_packet_col_overlay`:
```bash
./scripts/emu-bridge-test.sh --trace=sweep-all add_one_ctrl_packet_col_overlay
```

Expected: previously SKIP (routing_conflict), now runs with 2+ passes.

**Step 4: Commit**

```bash
git commit -m "feat: multi-pass trace sweep with automatic capacity planning"
```

---

## Task 7: Clear Error Reporting for Untraceable Tests

When the planner determines that even a single tile can't be traced (zero
available capacity on all paths to all shims), report a clear error with
diagnostic information.

**Files:**
- Modify: `tools/trace-inject.py` (update `plan_trace_passes()`)
- Modify: `tools/trace-sweep.py` (update skip manifest)

**Step 1: Add diagnostic output**

When `plan_trace_passes()` returns an empty plan:
```
TRACE CAPACITY ERROR: cannot route any trace flows for test <name>
  Bottleneck: tile (0,3) South port has 0 available rule slots
  Existing flows consuming capacity:
    - 4 circuit flows (South master)
    - 4 packet flows (slave rule slots)
  Suggestion: reduce test complexity or trace a subset of tiles manually
```

In the sweep, record this as a new skip reason: `"capacity_exceeded"`.

**Step 2: Test**

Verify the error message appears for a hypothetical fully-saturated test.

**Step 3: Commit**

```bash
git commit -m "feat: clear diagnostic error when trace capacity is exceeded"
```

---

## Design Notes

### Device Model Portability

All hardware constants are loaded from `aie-device-models.json` or defined
as module-level constants (not buried in logic). When AIE2P support arrives:
- Port counts come from the device model (already has npu2 entries)
- Arbiter/MSEL counts are in named constants at module top
- The pathfinding heuristic (southward + east/west at row 0) works for all
  current NPU layouts since they're rectangular arrays with shims at row 0

### Conservative vs Exact

The planner is deliberately conservative. It may occasionally split into
2 passes when 1 would have worked, because:
- The pathfinder uses backtracking and creative routing we don't simulate
- False-match analysis requires full mask/value simulation
- Being conservative means fewer pathfinder failures = faster total runtime

If the conservative estimate says "1 pass", it will almost certainly work.
If it says "2 passes", 1 might have worked but 2 definitely will.

### Future: Pathfinder Fallback

If the conservative planner says 1 pass but pathfinder still fails (edge
case), the sweep can catch the `routing_conflict` error and automatically
retry with a 2-pass split. This is the existing behavior, just better
informed.
