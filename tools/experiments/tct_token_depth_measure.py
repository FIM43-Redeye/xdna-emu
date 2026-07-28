#!/usr/bin/env python3
"""Measure the TCT completion-token buffer depth (Experiment C) from a
tct_token_depth capture.

Algorithm (per the spec's Experiment C section): count `FINISHED_TASK`
rising edges -- a discrete event, so its perfetto `B`-phase timestamp IS
the edge -- that occur strictly BEFORE the first `DMA_TASK_TOKEN_STALL`
onset -- a level event, so its onset is its first interval's `B`-phase
timestamp. That count is the number of completed-but-undrained tasks
(tokens) outstanding at the moment the completion-token buffer first
saturates -- the buffer depth. If TOKEN_STALL never asserts in the
capture, report that explicitly (`stall_observed=False`), never a bogus
count.

`DMA_TASK_TOKEN_STALL` is a single tile-wide event (not indexed per
channel/direction, unlike `FINISHED_TASK`/`START_TASK`/`STALLED_LOCK`;
confirmed against mlir-aie's aie2.py ShimTileEvent/MemEvent/MemTileEvent
enums, each of which has exactly one `DMA_TASK_TOKEN_STALL` member),
consistent with the completion-token queue being one shared per-tile
resource fed by every channel on that tile (docs/device-model-audit.md's
`task_complete_queue_size` row lists it uniformly "(C,M,S)" -- per tile,
not per channel). So `_finished_task_edges` below sums every
`*_FINISHED_TASK`-named slot present for the tile under test, not just
the one channel tct_token_depth.py's wave design happens to stress.

Tile-aware keying. Following the mm2s_egress_depth_measure.py /
bankdisc_measure.py precedent (and the exact lesson from Task 3, which
shipped a Critical bug doing this wrong): a real tct_token_depth capture
traces at minimum the shim tile under test (module="shim", tile (0,0) --
SHIM_ROW/SHIM_COL below) and the compute-tile passthrough (module="mem",
tile (2,0)) that lets the shim MM2S/S2MM channels actually drain/fill.
Those two are already disambiguated by MODULE alone ("shim" vs "mem"), so
they cannot collide under a module-type-only key. The real risk is a
SECOND shim tile: an XCLBIN with more than one shim column, or a trace
harness that instruments every shim tile in the design by default, would
give both an identical module="shim" entry with the same
`DMA_MM2S_0_FINISHED_TASK` / `DMA_TASK_TOKEN_STALL` event names -- a
tile-blind loader (keyed by (module_type, event_name) alone, e.g.
bankdisc_measure.load_intervals) would silently merge them, corrupting the
depth reading exactly as it did for mm2s_egress_depth's two-tile variants.
`_load_tile_intervals` below is a local tile-aware loader (recovering
(row, col) from the perfetto module string, same regex/convention as
mm2s_egress_depth_measure.py's `_load_source_tile_intervals`) that filters
every measurement down to SHIM_ROW/SHIM_COL before it ever reaches the
onset/count computation.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# aie.tile(0, 0) in tct_token_depth.py: the shim DMA under test in every
# variant. The compute-tile passthrough at (2, 0) is module="mem", already
# distinct by module string, but a SECOND shim tile (module="shim") is the
# real collision risk this filter guards against.
#
# Column virtualization: a declared aie.tile(0, N) (logical col 0) decodes in
# the trace at PHYSICAL col 1 -- Phoenix physical col 0 carries a memtile + 4
# cores but NO shim (project_phoenix_col0_tiles_rewritten_inaccessible), so
# the npu1_2col device maps onto physical cols 1-2 and the trace reports
# physical coordinates. Experiment B confirmed the +1 offset for a col-0
# compute tile (declared (0,2) -> decoded mem(2,1)); the same applies to this
# shim, so the trace tile is shim(0,1), i.e. col 1. CONFIRM at capture: if the
# measure finds 0 FINISHED_TASK edges, re-check this against the real trace's
# process_name strings and adjust.
SHIM_ROW, SHIM_COL = 0, 1

# Real capture module strings look like "shim(0,0)": pt_name then (row, col)
# -- see trace_decoder.decode.rebuild_perfetto_mode0 (process_name is built
# as f"{pt_name}({row},{col})"; pt_name for pkt_type 2 is "shim" per
# trace_decoder.decode._PT_CODE_TO_NAME).
_TILE_RE = re.compile(r"^(\w+)\((\d+),(\d+)\)$")

STALL_EVENT = "DMA_TASK_TOKEN_STALL"


def _load_tile_intervals(perfetto_path: Path, config_path: Path):
    """Tile-aware sibling of bankdisc_measure.load_intervals /
    mm2s_egress_depth_measure._load_source_tile_intervals.

    -> {event_name: [(start, end), ...]}, restricted to (SHIM_ROW, SHIM_COL).
    Events from any other tile (a second shim column, the compute-tile
    passthrough) never reach the caller, instead of being merged in under
    a module-type-only key.
    """
    slots = {}
    for t in json.load(config_path.open())["tiles_traced"]:
        # Shim tiles carry NO "module" key in the real trace_config.json (only
        # compute tiles do) -- fall back to "kind", so the shim's event slot
        # table is registered under the same string the decoded process_name
        # uses ("shim"). Same fallback Task 5 added to load_intervals for the
        # memtile. Without it, this line KeyErrors on every real shim capture.
        mod = t.get("module") or t.get("kind")
        slots[mod] = t["events"]

    ev = json.load(perfetto_path.open())
    pid_tile = {}
    for e in ev:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            m = _TILE_RE.match(e["args"]["name"])
            if m:
                mod, row, col = m.group(1), int(m.group(2)), int(m.group(3))
                pid_tile[e["pid"]] = (mod, row, col)

    open_b = {}
    out = defaultdict(list)
    for e in ev:
        ph = e.get("ph")
        if ph not in ("B", "E"):
            continue
        tile = pid_tile.get(e["pid"])
        if tile is None:
            continue
        mod, row, col = tile
        if (row, col) != (SHIM_ROW, SHIM_COL):
            continue
        names = slots.get(mod, [])
        tid = e["tid"]
        if tid >= len(names):
            continue
        key = names[tid]
        if ph == "B":
            open_b[key] = e["ts"]
        elif key in open_b:
            out[key].append((open_b.pop(key), e["ts"]))
    return out


def _finished_task_edges(iv: dict) -> list:
    """Rising-edge (B-phase) timestamps of every *_FINISHED_TASK-named slot
    present for the tile under test, pooled across channels/directions --
    DMA_TASK_TOKEN_STALL is tile-wide (see module docstring), so every
    channel's completions contribute to the same outstanding-token count.
    """
    edges = []
    for name, intervals in iv.items():
        if name.endswith("FINISHED_TASK"):
            edges.extend(start for start, _ in intervals)
    return sorted(edges)


def measure(build_dir: Path, rep: int) -> dict:
    """-> dict with:
      stall_observed: bool
      onset: first DMA_TASK_TOKEN_STALL interval's start ts, or None
      tasks_outstanding: FINISHED_TASK edges strictly before onset (the
        buffer-depth reading), or None if no stall was observed
      n_finished_task: total FINISHED_TASK edges in the capture (context)
    """
    iv = _load_tile_intervals(build_dir / f"perfetto_r{rep}.json",
                               build_dir / "trace_config.json")
    edges = _finished_task_edges(iv)
    stalls = sorted(iv.get(STALL_EVENT, []))
    if not stalls:
        return {
            "stall_observed": False,
            "onset": None,
            "tasks_outstanding": None,
            "n_finished_task": len(edges),
        }
    onset = stalls[0][0]
    outstanding = sum(1 for ts in edges if ts < onset)
    return {
        "stall_observed": True,
        "onset": onset,
        "tasks_outstanding": outstanding,
        "n_finished_task": len(edges),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="build/experiments/tct-token-depth")
    ap.add_argument("--reps", type=int, default=3, help="HW run repeats to pool")
    args = ap.parse_args()
    root = Path(args.root)

    from tct_token_depth import VARIANTS  # noqa: E402

    print(f"{'variant':26} {'run':>3} {'stall':>6} {'onset':>8} {'outstanding':>11}")
    print("-" * 60)
    depths = []
    for v in VARIANTS:
        d = root / f"build_tct_token_depth_{v}"
        if not d.is_dir():
            continue
        for r in range(1, args.reps + 1):
            if not (d / f"perfetto_r{r}.json").exists():
                continue
            m = measure(d, r)
            print(f"{v:26} {r:>3} {str(m['stall_observed']):>6} "
                  f"{str(m['onset']):>8} {str(m['tasks_outstanding']):>11}")
            if m["tasks_outstanding"] is not None:
                depths.append(m["tasks_outstanding"])
        print()

    if depths:
        print(f"DEPTH readings observed (min..max across stalling variants/reps): "
              f"{min(depths)}..{max(depths)}")
    else:
        print("No TOKEN_STALL observed in any capture -- mechanism not triggered "
              "(document, do not force a number).")
