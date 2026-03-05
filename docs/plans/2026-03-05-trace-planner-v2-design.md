# Trace Planner v2: Pathfinder-Driven Cross-Column Routing

Date: 2026-03-05

## Problem

The current trace planner (`strict_analyze_feasibility()`) approximates routing
feasibility using heuristics: shim channel occupancy, packet rule density at
intermediate tiles, control packet detection. These heuristics miss cases
(6/11 quarantined tests need the quarantine file as safety net) and can't
reason about cross-column routing at all.

Meanwhile, mlir-aie's pathfinder is a full Dijkstra-based router with
congestion-aware demand weighting that handles cross-column routing natively.
We're manually reimplementing what the compiler already does.

## Core Insight

The pathfinder can be invoked directly from Python via `PassManager`. Running
it takes ~1-2 seconds per variant (just the routing solve, no CDO/ELF). This
means we can trial multiple routing candidates cheaply and pick the best one,
using the pathfinder as its own oracle rather than approximating its behavior.

The golden rule: **if adding trace flows causes the pathfinder to reroute any
existing test flow, that candidate is invalid.** Trace must be invisible to
the test's own dataflow.

## Architecture

### Data Model

```python
@dataclass
class TracePlan:
    """Result of trace route planning."""
    feasible: bool
    reason: str
    shim_col: int | None = None
    trace_channel: int | None = None
    candidates: list[CandidateResult] | None = None

@dataclass
class CandidateResult:
    """Pathfinder result for one (shim_col, channel) candidate."""
    shim_col: int
    channel: int
    success: bool                      # pathfinder found a valid route
    existing_flows_intact: bool        # baseline connections unchanged
    failure_reason: str | None = None
    # Tiebreaker metrics (only meaningful when success=True)
    trace_connections_on_test_cols: int = 0
    total_trace_connections: int = 0
```

### Components

#### 1. Shared Flow Helper: `add_trace_flows()`

Declares trace packet flows from tiles to a target shim. Used by both the
planner (testing candidates on clones) and the injector (committing the
chosen plan).

```python
def add_trace_flows(
    module: Module,
    shim_col: int,
    trace_channel: int,
    trace_targets: list[tuple[TileOp, int]],  # (tile, trace_port) -- explicit
    used_packet_ids: set[int],
) -> int:
```

Responsibilities:
- Find or create shim tile declaration via `aiedialect.tile()`
- Allocate packet IDs above existing ones (scanned from IR, not regex)
- Declare `packetflow()` ops for each (tile, trace_port) -> (shim, DMA:channel)
- Returns first packet ID used

Does NOT configure trace registers, timers, DMA, or buffers.

#### 2. Connection Extractor: `extract_connections()`

Extracts all switch connections from a routed module as a set of tuples.

```python
def extract_connections(module) -> set[tuple]:
    """Returns set of (col, row, src_bundle, src_channel, dst_bundle, dst_channel)."""
```

Used for baseline comparison: `baseline.issubset(candidate)` verifies that
all original routing decisions are preserved.

#### 3. Planner: `plan_trace_route()`

The core algorithm.

```
plan_trace_route(mlir_text) -> TracePlan
    |
    Parse module, classify tiles, build trace_targets
    |
    Run pathfinder on UNMODIFIED module -> baseline_connections
    |
    Generate candidates: all device columns x 2 channels
    |
    In parallel, for each candidate:
        Clone module via Module.parse(module.get_asm())
        add_trace_flows(clone, shim_col, channel, ...)
        Run pathfinder: PassManager.parse(
            "builtin.module(aie.device(aie-create-pathfinder-flows))"
        ).run(clone.operation)
        Extract connections, compare to baseline
        Score: (trace_on_test_cols, total_trace_connections)
    |
    Pick winner: existing_flows_intact, then lowest score
    |
    Return TracePlan
```

Parallelism: each candidate runs in a separate process (MLIR Contexts are
not thread-safe). `concurrent.futures.ProcessPoolExecutor` with one worker
per candidate. For NPU1 (5 cols x 2 channels = 10 candidates), this
completes in ~2 seconds wall time.

#### 4. Injector: `inject_trace()` (simplified)

Receives a `TracePlan` and executes it. No longer selects shim or channel.

Responsibilities (unchanged):
- Call `add_trace_flows()` with plan's shim_col and trace_channel
- Configure trace registers (`configure_*_tracing_aie2`)
- Configure timers (`configure_timer_ctrl_*`)
- Configure shim DMA (`configure_shimtile_dma_aie2`)
- Patch buffer argument via text regex (Python API limitation)
- Serialize module to text

### Winner Selection

When multiple candidates succeed with existing flows intact:

1. **Least trace activity on test columns** -- `trace_connections_on_test_cols`.
   Zero means trace escaped entirely to non-test columns. A single-column
   test on col 0 would prefer col 1 (one East hop) over col 4 (four hops).

2. **Least total data movement** -- `total_trace_connections`. Among equal
   candidates, prefer fewer total switch connections (shorter routes).

```python
winners = [c for c in candidates if c.success and c.existing_flows_intact]
best = min(winners, key=lambda c: (c.trace_connections_on_test_cols,
                                    c.total_trace_connections))
```

### CLI Integration

New flag: `--plan-only`

- Ignores quarantine list
- Runs planner on all candidates
- Prints full candidate results table (success, intact, scores)
- Exits without writing any output files
- Useful for debugging quarantined tests and understanding routing pressure

### Pipeline Flow

```
main()
    |
    Quarantine check (skip if --plan-only)
    |
    plan_trace_route(mlir_text) -> TracePlan
    |
    If --plan-only: print results table, exit
    |
    If infeasible: write skip manifest with reasons, exit
    |
    inject_trace(mlir_text, plan) -> modified_mlir_text
    |
    Write output files + manifest with plan diagnostics
```

## What Gets Removed

| Current code | Replacement |
|-------------|-------------|
| `strict_analyze_feasibility()` | `plan_trace_route()` |
| `ShimChannelState` dataclass | Pathfinder decides directly |
| `StrictTracePlan` dataclass | `TracePlan` + `CandidateResult` |
| `find_used_packet_ids()` (regex) | IR-walking via `find_ops()` |
| `has_existing_trace_flows()` (regex) | IR-walking via `find_ops()` |
| Shim selection logic in `inject_trace()` | Plan provides shim_col |
| Channel selection logic in `inject_trace()` | Plan provides trace_channel |
| Packet rule density heuristic | Pathfinder handles natively |

## What Stays

- Quarantine file (`scripts/trace-quarantine.txt`) -- safety net for
  pathologically unroutable tests. May shrink as the planner handles more
  cases.
- Event configuration and resolution
- Trace register programming (all `configure_*` calls)
- Buffer argument text patching (mlir-aie API limitation)
- `--plan-only` respects quarantine skip for normal runs

## Stream Switch Topology (Reference)

| Tile Type | East/West Ports | Key Constraint |
|-----------|----------------|----------------|
| Compute | 4 master + 4 slave each | Full cross-column support |
| MemTile | NONE | No lateral routing at row 1 |
| Shim | 4 master + 4 slave each | Full cross-column support |

Cross-column trace routing must hop East/West at compute rows (2-5) or shim
row (0). The MemTile row is a hard barrier.

## Verification Strategy

1. Run `--plan-only` on all 37 bridge tests, compare to current planner's
   decisions
2. Run `--plan-only` on all 11 quarantined tests to see which ones the new
   planner can handle
3. End-to-end: inject + compile + run on `add_one_using_dma` with the new
   planner, verify traces match baseline
4. Verify a single-column test routes trace to adjacent column (not same)
5. Verify a multi-column test picks the least-interference corridor
