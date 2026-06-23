"""config_path generator: SS route-graph reachability -> ledger entries.

This is C3 in the inference-engine Plan 2 (#140).

Given a ConfigDump (the tile + route-graph snapshot of a kernel's configuration)
and the set of event keys that fired in a captured run, the generator produces a
structural ledger of config_path facts.  Each fact asserts that the route graph
carries data FROM a parent event's SS port TO a child event's SS port, so the
inference engine can orient timing relationships.

Soundness contract
------------------
The generator emits config_path(a=parent, b=child, cite) if and only if:
  1. Both the parent event key AND the child event key resolve to distinct
     PortRef nodes via resolve_event_port.
  2. The parent node REACHES the child node via directed BFS on the SS route
     graph (Reachability).
  3. The two are not the same event key (no self-loops).

Ledger orientation: a = parent (upstream), b = child (downstream).
This matches the hand-authored ledger schema and the inference engine's contract:
  inference/ledger.py: "config_path(a, b): routes a's producer to b's consumer"
  inference/rules.py:  config_path(args[0]=parent, args[1]=child) -> derives

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
    Each entry: {"cite": str, "a": parent_key, "b": child_key, "kind": "route"}.

fired_keys_from_run(run_dir, anchor) -> list[str]
    Helper: extract the set of event keys that fired in a captured run, by
    loading trace.events.json from each batch via inference.loader helpers.
    The anchor is REQUIRED and explicit (kernel-and-column specific); raises
    ValueError if it resolves zero keys rather than returning silently empty.

main()
    CLI: python -m config_extract.generator <config.json> <run_dir> -o <ledger.json>
"""

from __future__ import annotations

import argparse
import json
import re
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


# Canonical cite shape produced by _make_cite: route:<parent>--reaches-->{child}.
# audit_ledger enforces this FULL structure (not just the "route:" prefix)
# because the cite is the trust anchor and Tier E (DMA-relay modeling) will
# parse it.  Non-greedy parent so the FIRST "--reaches-->" separates the
# endpoints; both sides must be non-empty.
_RE_CITE = re.compile(r"^route:(?P<parent>.+?)--reaches-->(?P<child>.+)$")


def _make_cite(parent_key: str, child_key: str) -> str:
    """Build a human-auditable cite string for a route edge.

    Format (stable -- C4 may parse this): route:<parent_key>--reaches-->{child_key}

    The cite names the structural justification: parent's SS port reaches
    child's SS port in the route graph.  It is intentionally hop-count-free --
    the existence of *a* directed path is the justification; the specific path
    length is not part of the soundness claim, so we do not encode it.  C4
    re-derives reachability from the dump to audit, so the cite only needs to
    name the two endpoints unambiguously.
    """
    return f"route:{parent_key}--reaches-->{child_key}"


# Canonical cite shape produced by _make_program_cite:
# program:<parent>--via-core-->{child}.
# Distinct from the route cite to make the structural claim explicit: this path
# passes through the compute-core (core_lock_relay edge), not purely SS config.
# audit_ledger enforces this FULL structure for program-kind entries.
_RE_PROGRAM_CITE = re.compile(
    r"^program:(?P<parent>.+?)--via-core-->(?P<child>.+)$"
)


def _make_program_cite(parent_key: str, child_key: str) -> str:
    """Build a human-auditable cite string for a program_path (through-core) edge.

    Format: program:<parent_key>--via-core--><child_key>

    The 'via-core' separator names the structural justification: the path from
    parent's SS port to child's SS port passes through the compute-core relay
    (core_lock_relay edge).  This edge is NOT a stream-switch config artifact --
    it requires the ELF program to relay data through the core's lock/buffer
    handshake.  Hence the distinct predicate (program_path vs config_path).
    """
    return f"program:{parent_key}--via-core-->{child_key}"


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
    where "a" is the parent (upstream) and "b" is the child (downstream).
    This matches the engine contract: config_path(args[0]=parent, args[1]=child).
    """
    if not fired_event_keys:
        return {
            "_comment": "generated by config_extract.generator (C3); no fired events",
            "entries": [],
        }

    tile_idx = _build_tile_index(dump)
    all_edges = dump.route_graph.edges

    # Build two reachability objects:
    #   full -- all edges including core_lock_relay (program + config paths).
    #   cfg  -- config-only edges (core_lock_relay excluded).
    # A pair reachable in full but not cfg is "program-only" (through-core relay).
    full = Reachability(all_edges)
    cfg  = Reachability([e for e in all_edges if e.kind != "core_lock_relay"])

    # Resolve every fired key to its PortRef (or None for non-route events).
    resolved: dict[str, Optional[PortRef]] = {
        k: _resolve_key(k, dump, tile_idx, start_col)
        for k in fired_event_keys
    }

    entries: list[dict] = []

    # Enumerate every ordered pair; classify and emit based on reachability.
    for parent_key, child_key in permutations(fired_event_keys, 2):
        parent_node = resolved[parent_key]
        child_node = resolved[child_key]

        # Skip non-route events (either end unresolvable).
        if parent_node is None or child_node is None:
            continue

        if cfg.reachable(parent_node, child_node):
            # Config-reachable: stream-switch config alone justifies the path.
            cite = _make_cite(parent_key, child_key)
            entries.append({
                "cite": cite,
                "a": parent_key,  # parent = upstream event (a = parent per ledger schema)
                "b": child_key,   # child = downstream event (b = child per ledger schema)
                "kind": "route",
            })
        elif full.reachable(parent_node, child_node):
            # Program-only: reachable only via core_lock_relay (through-core relay).
            # The path requires the ELF program to relay data through the core;
            # SS config alone is insufficient.
            cite = _make_program_cite(parent_key, child_key)
            entries.append({
                "cite": cite,
                "a": parent_key,  # parent = upstream (a = parent per ledger schema)
                "b": child_key,   # child = downstream (b = child per ledger schema)
                "kind": "program",
            })
        # else: not reachable in either graph -- decline (safe false-negative).

    return {
        "_comment": (
            "generated by config_extract.generator (C3); "
            "route-graph reachability: 'route' for config-reachable pairs, "
            "'program' for through-core-only pairs (core_lock_relay)"
        ),
        "entries": entries,
    }


# Default anchor for add_one_using_dma at NPU1 start_col=1.  This is
# kernel-and-column specific (PERF_CNT_2 on the compute tile at col=1, row=2);
# other kernels or placements need a different anchor key, so callers must
# supply it explicitly -- it is NOT a sane universal default.
DEFAULT_ANCHOR = "1|2|0|PERF_CNT_2"


def fired_keys_from_run(run_dir: str, anchor: str) -> list[str]:
    """Extract the set of event keys that fired in a captured run.

    Loads trace.events.json from each batch in the run directory via
    inference.loader (which uses trace_join's batch helpers).

    Parameters
    ----------
    run_dir: Path to the run directory (contains batch_NN/ subdirs with
             hw/ and emu/ sub-subdirs and trace.events.json files).
    anchor:  The anchor event key used to time-align batches.  REQUIRED and
             explicit: the anchor is kernel-and-column specific (e.g.
             "1|2|0|PERF_CNT_2" for add_one at start_col=1).  An anchor that
             does not appear in the run yields no aligned events.

    Returns
    -------
    A list of unique event keys that fired in at least one batch of the run.

    Raises
    ------
    ValueError: if no event keys resolve (empty result).  This is almost always
                a wrong/absent anchor or an empty run dir -- failing loudly is
                safer than silently producing an empty ledger.
    """
    from inference.loader import _first_firsts  # type: ignore[import]

    # _first_firsts returns {event_key: anchored_ts}; we want only the keys.
    firsts = _first_firsts(run_dir, anchor)
    keys = list(firsts.keys())
    if not keys:
        raise ValueError(
            f"fired_keys_from_run resolved zero event keys from {run_dir!r} "
            f"with anchor {anchor!r}: the anchor may not appear in this run, "
            f"or the run dir has no traced batches.  Pass the correct anchor "
            f"for this kernel/column placement."
        )
    return keys


def audit_ledger(
    led: dict,
    dump: ConfigDump,
    start_col: int = 0,
) -> list[str]:
    """Audit every entry in a ledger against the route graph (the trust anchor).

    For each entry the following properties are verified:

    1. ``kind`` is ``"route"`` or ``"program"``; any other value is flagged.
    2. ``cite`` matches the FULL canonical structure for its kind AND its
       parent/child portions agree with the entry's ``a``/``b`` keys.
       - ``route`` entries: ``route:<parent>--reaches-->{child}``
       - ``program`` entries: ``program:<parent>--via-core--><child>``
       Prefix-only is not enough -- the cite is the trust anchor that Tier E
       parses, so a malformed payload is caught here.
    3. Both ``a`` (parent key) and ``b`` (child key) resolve to PortRef nodes
       via ``resolve_event_port`` with the supplied ``start_col`` offset.
    4. The cited structural path actually holds in the appropriate graph:
       - ``route`` entries: reachable in the config-only graph (no core_lock_relay).
       - ``program`` entries: reachable in the full graph (including core_lock_relay).

    Parameters
    ----------
    led:       The ledger dict (as returned by ``generate_ledger``), with an
               ``"entries"`` list of dicts.
    dump:      The ``ConfigDump`` that was used to generate the ledger.  The
               audit re-derives the route graph and tile index from it.
    start_col: Column offset applied when resolving event keys to dump tiles
               (same value passed to ``generate_ledger``).  Default 0.

    Returns
    -------
    A list of human-readable failure strings.  An empty list means the ledger
    is internally consistent with the route graph -- it passes audit.
    """
    tile_idx = _build_tile_index(dump)
    failures: list[str] = []

    # Build two reachability objects for audit:
    #   full_reach -- all edges (including core_lock_relay) for 'program' entries.
    #   cfg_reach  -- config-only edges (no core_lock_relay) for 'route' entries.
    all_edges = dump.route_graph.edges
    full_reach = Reachability(all_edges)
    cfg_reach  = Reachability([e for e in all_edges if e.kind != "core_lock_relay"])

    for i, entry in enumerate(led.get("entries", [])):
        cite = entry.get("cite", "")
        a = entry.get("a", "")   # parent key (upstream)
        b = entry.get("b", "")   # child key (downstream)
        kind = entry.get("kind", "")
        label = f"entry[{i}] cite={cite!r}"

        # Check 1: kind must be "route" or "program".
        if kind not in ("route", "program"):
            failures.append(
                f"{label}: expected kind='route' or kind='program', got {kind!r}"
            )
            continue  # remaining checks assume route/program semantics

        if kind == "route":
            # Check 2a (route): cite must match the FULL canonical structure
            # route:<parent>--reaches-->{child}, and its parent/child portions must
            # agree with the entry's a/b keys.  Prefix-only ("route:...") is NOT
            # enough -- the cite is the trust anchor that Tier E parses, so a
            # malformed payload like "route:" or "route:x--reaches-->" must be
            # caught.  Do not skip the remaining checks -- the path check is still
            # meaningful even when the cite is malformed.
            m = _RE_CITE.match(cite)
            if m is None:
                failures.append(
                    f"{label}: cite does not match canonical structure "
                    f"'route:<parent>--reaches-->{{child}}' (got {cite!r})"
                )
            else:
                if m.group("parent") != a:
                    failures.append(
                        f"{label}: cite parent {m.group('parent')!r} does not match "
                        f"entry 'a'={a!r}"
                    )
                if m.group("child") != b:
                    failures.append(
                        f"{label}: cite child {m.group('child')!r} does not match "
                        f"entry 'b'={b!r}"
                    )
        else:
            # kind == "program"
            # Check 2b (program): cite must match the FULL canonical structure
            # program:<parent>--via-core--><child>, and its parent/child portions
            # must agree with the entry's a/b keys.
            m = _RE_PROGRAM_CITE.match(cite)
            if m is None:
                failures.append(
                    f"{label}: cite does not match canonical structure "
                    f"'program:<parent>--via-core--><child>' (got {cite!r})"
                )
            else:
                if m.group("parent") != a:
                    failures.append(
                        f"{label}: cite parent {m.group('parent')!r} does not match "
                        f"entry 'a'={a!r}"
                    )
                if m.group("child") != b:
                    failures.append(
                        f"{label}: cite child {m.group('child')!r} does not match "
                        f"entry 'b'={b!r}"
                    )

        # Check 3: both keys must resolve to PortRef nodes.
        parent_node = _resolve_key(a, dump, tile_idx, start_col)
        child_node = _resolve_key(b, dump, tile_idx, start_col)

        if parent_node is None:
            failures.append(
                f"{label}: 'a'={a!r} does not resolve to a route-graph node "
                f"(non-routable event or unknown tile at start_col={start_col})"
            )
        if child_node is None:
            failures.append(
                f"{label}: 'b'={b!r} does not resolve to a route-graph node "
                f"(non-routable event or unknown tile at start_col={start_col})"
            )

        if parent_node is None or child_node is None:
            continue  # cannot check reachability without both nodes

        # Check 4: the graph must actually reach child from parent.
        # For 'route' entries: config-only reachability (no core_lock_relay).
        # For 'program' entries: full reachability (including core_lock_relay).
        if kind == "route":
            ok = cfg_reach.reachable(parent_node, child_node)
            graph_desc = "config-only SS route graph"
        else:
            ok = full_reach.reachable(parent_node, child_node)
            graph_desc = "full route graph (including core_lock_relay)"

        if not ok:
            failures.append(
                f"{label}: claimed path a={a!r} (parent) -> b={b!r} (child) "
                f"is NOT reachable in the {graph_desc} -- "
                f"parent_node={parent_node}, child_node={child_node}"
            )

    return failures


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
                             "mapping (default: 1 for NPU1 single-kernel placement). "
                             "NOTE: the generate_ledger() API defaults start_col=0 "
                             "(keys already in hardware col space); this CLI defaults "
                             "to 1 because NPU1 kernels run at start_col=1.")
    parser.add_argument("--anchor", type=str, default=DEFAULT_ANCHOR,
                        help=f"Anchor event key for batch time-alignment "
                             f"(default: {DEFAULT_ANCHOR!r}, the add_one/start_col=1 "
                             f"anchor). Kernel-and-column specific -- override for "
                             f"other kernels or placements.")
    args = parser.parse_args()

    dump = load_dump(args.config_json)
    fired_keys = fired_keys_from_run(str(args.run_dir), anchor=args.anchor)

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
