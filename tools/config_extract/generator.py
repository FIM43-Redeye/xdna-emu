"""config_path generator: SS route-graph reachability -> ledger entries.

This is C3 in the inference-engine Plan 2 (#140).

Given a ConfigDump (the tile + route-graph snapshot of a kernel's configuration)
and the set of event keys that fired in a captured run, the generator produces a
structural ledger of config_path facts.  Each fact asserts that the route graph
carries data FROM a parent event's SS port TO a child event's SS port, so the
inference engine can orient timing relationships.

Soundness contract
------------------
The generator emits config_path(a=child, b=parent, cite) if and only if:
  1. Both the parent event key AND the child event key resolve to distinct
     PortRef nodes via resolve_event_port.
  2. The parent node REACHES the child node via directed BFS on the SS route
     graph (Reachability).
  3. The two are not the same event key (no self-loops).

It does NOT emit edges for:
  - Events that do not correspond to a routable SS port (LOCK_STALL, PERF_CNT_*,
    etc.) -- these return None from resolve_event_port.
  - Co-firing pairs (e.g. broadcast PORT_RUNNING events on the same broadcast
    bus) where there is no actual SS dataflow path.  Being captured together is
    not a structural justification.

This means the generator may produce ZERO edges for a kernel whose dataflow
crosses DMA memory-buffer relays (the SS-only graph does not model those).  A
zero-edge result is a valid sound result -- it is the input signal that Tier E
(DMA-relay modeling) needs to extend the graph.

Col-offset reconciliation
-------------------------
Trace event keys use "decoder space" column numbering: when a kernel is loaded
at hardware column offset C (start_col), the emulator/decoder tags events with
col = hardware_col + start_col.  The config dump, however, records tiles at
their hardware column (0-based from the array boundary).

  dump_col = event_key_col - start_col

The generator accepts a start_col parameter (default=0) and applies this
subtraction before looking up tiles.  For the add_one_using_dma fixture the
kernel runs at start_col=1, so event keys at col=1 map to dump tiles at col=0.

API
---
generate_ledger(dump, fired_event_keys, start_col=0) -> dict
    Returns {"_comment": str, "entries": [...]}.
    Each entry: {"cite": str, "a": child_key, "b": parent_key, "kind": "route"}.

fired_keys_from_run(run_dir) -> list[str]
    Helper: extract the set of event keys that fired in a captured run, by
    loading trace.events.json from each batch via inference.loader helpers.

main()
    CLI: python -m config_extract.generator <config.json> <run_dir> -o <ledger.json>
"""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path
from typing import Optional

from config_extract.dump_model import ConfigDump, PortRef, TileDump, load_dump
from config_extract.event_map import resolve_event_port
from config_extract.reachability import Reachability


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_event_key(key: str) -> tuple[int, int, int, str]:
    """Parse 'col|row|pkt|name' -> (col, row, pkt, name)."""
    parts = key.split("|", 3)
    return int(parts[0]), int(parts[1]), int(parts[2]), parts[3]


def _build_tile_index(dump: ConfigDump) -> dict[tuple[int, int], TileDump]:
    """Build a (col, row) -> TileDump lookup from the dump."""
    return {(t.col, t.row): t for t in dump.tiles}


def _resolve_key(
    key: str,
    dump: ConfigDump,
    tile_idx: dict[tuple[int, int], TileDump],
    start_col: int,
) -> Optional[PortRef]:
    """Resolve one event key to its PortRef using the dump + col offset.

    Returns None if:
    - The tile at (event_col - start_col, event_row) is not in the dump.
    - The event does not correspond to a routable SS port (resolve_event_port).
    """
    fcol, frow, _fpkt, name = _parse_event_key(key)
    dump_col = fcol - start_col
    tile = tile_idx.get((dump_col, frow))
    if tile is None:
        return None
    return resolve_event_port(tile, name, dump)


def _make_cite(parent_key: str, child_key: str, hop_count: Optional[int] = None) -> str:
    """Build a human-auditable cite string for a route edge.

    Format: route:<parent_key>--reaches-->{child_key}
    If hop_count is known, appended as @Nhops.
    """
    if hop_count is not None:
        return f"route:{parent_key}--reaches-->{child_key}@{hop_count}hops"
    return f"route:{parent_key}--reaches-->{child_key}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_ledger(
    dump: ConfigDump,
    fired_event_keys: list[str],
    start_col: int = 0,
) -> dict:
    """Generate a structural ledger of config_path facts from route reachability.

    For every ordered pair (parent_key, child_key) from fired_event_keys
    (where parent != child), resolve both to PortRef nodes and emit a ledger
    entry if parent's node REACHES child's node in the SS route graph.

    Parameters
    ----------
    dump:             The ConfigDump loaded from the Rust extractor's JSON.
    fired_event_keys: Event keys (col|row|pkt|name) that fired in the run.
                      Keys use decoder-space col (before start_col offset).
    start_col:        Hardware column offset applied to the kernel placement.
                      Subtracted from each key's col before tile lookup.
                      Default 0 (keys already in hardware col space).

    Returns
    -------
    A dict with "_comment" and "entries" keys.  entries is a list of dicts
    {"cite", "a", "b", "kind"} compatible with inference/ledger.py's schema
    where "a" is the child (downstream) and "b" is the parent (upstream).
    """
    if not fired_event_keys:
        return {
            "_comment": "generated by config_extract.generator (C3); no fired events",
            "entries": [],
        }

    tile_idx = _build_tile_index(dump)
    reach = Reachability(dump.route_graph.edges)

    # Resolve every fired key to its PortRef (or None for non-route events).
    resolved: dict[str, Optional[PortRef]] = {
        k: _resolve_key(k, dump, tile_idx, start_col)
        for k in fired_event_keys
    }

    entries: list[dict] = []

    # Enumerate every ordered pair; emit when parent reaches child.
    for parent_key, child_key in permutations(fired_event_keys, 2):
        parent_node = resolved[parent_key]
        child_node = resolved[child_key]

        # Skip non-route events (either end unresolvable).
        if parent_node is None or child_node is None:
            continue

        # Core soundness check: structural route reachability.
        if not reach.reachable(parent_node, child_node):
            continue

        cite = _make_cite(parent_key, child_key)
        entries.append({
            "cite": cite,
            "a": child_key,   # child = downstream event
            "b": parent_key,  # parent = upstream event
            "kind": "route",
        })

    return {
        "_comment": (
            "generated by config_extract.generator (C3); "
            "route-graph reachability only (SS-only, no DMA-relay edges yet)"
        ),
        "entries": entries,
    }


def fired_keys_from_run(run_dir: str) -> list[str]:
    """Extract the set of event keys that fired in a captured run.

    Loads trace.events.json from each batch in the run directory via
    inference.loader (which uses trace_join's batch helpers).

    Parameters
    ----------
    run_dir: Path to the run directory (contains batch_NN/ subdirs with
             hw/ and emu/ sub-subdirs and trace.events.json files).

    Returns
    -------
    A list of unique event keys that fired in at least one batch of the run.
    """
    from inference.loader import _first_firsts  # type: ignore[import]

    # _first_firsts returns {event_key: anchored_ts}; we want only the keys.
    anchor = "1|2|0|PERF_CNT_2"  # default anchor from inference.loader
    firsts = _first_firsts(run_dir, anchor)
    return list(firsts.keys())


def main() -> None:
    """CLI: python -m config_extract.generator <config.json> <run_dir> -o <ledger.json>

    If run_dir is not available or you want to pass fired keys inline, use
    generate_ledger() directly.

    Example
    -------
    python -m config_extract.generator \\
        config_extract/fixtures/add_one_using_dma.config.json \\
        /path/to/run \\
        -o build/ledger.json \\
        --start-col 1
    """
    parser = argparse.ArgumentParser(
        description="Generate a config_path ledger from SS route-graph reachability."
    )
    parser.add_argument("config_json", type=Path,
                        help="Path to the config dump JSON file.")
    parser.add_argument("run_dir", type=Path,
                        help="Path to the captured run directory (for fired event keys).")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output ledger JSON path (default: print to stdout).")
    parser.add_argument("--start-col", type=int, default=1,
                        help="Hardware column offset for event key col->dump tile col "
                             "mapping (default: 1 for NPU1 single-kernel placement).")
    args = parser.parse_args()

    dump = load_dump(args.config_json)
    fired_keys = fired_keys_from_run(str(args.run_dir))

    ledger = generate_ledger(dump, fired_keys, start_col=args.start_col)

    serialised = json.dumps(ledger, indent=2)
    if args.output is None:
        print(serialised)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialised, encoding="utf-8")
        print(f"Wrote {len(ledger['entries'])} entries to {args.output}")


if __name__ == "__main__":
    main()
