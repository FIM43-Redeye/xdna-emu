#!/usr/bin/env python3
"""Run the full trace-event sweep across every (test, compiler, traced tile).

Harness around tools/trace-sweep.py for the regression-verification use case:
after any emulator change, run this once and diff its output (via
scripts/show-sweep-matrix.py) against a pre-change baseline. Cells that
flip from MATCH to DRIFT/MISS point at exactly which events regressed
and on which tile.

Discovery
---------
- Tests: auto-discovered from
  <mlir-aie>/build/test/npu-xrt/<test>/aie-hw-cycles-traced.mlir. Any test
  that emu-bridge-test.sh --with-hw-cycles has already compiled trace
  routing for is eligible. Pass --test NAME to narrow.
- Traced tiles: parsed from the same MLIR, looking for
  `aie.trace @sym(%tile_C_R)` blocks and the packet `type =` inside
  them. type maps: core->core, mem->memmod.

Output layout
-------------
  build/trace-sweep-results/YYYYMMDD[-N]/
    <test>.<compiler>.<tile>.json          # per-event matrix
    <test>.<compiler>.<tile>.merged.json   # grounded timeline
    <test>.<compiler>.<tile>.work/         # per-batch artifacts
    summary.json                           # all combos + run time + status

A `latest` symlink inside build/trace-sweep-results/ points at the newest
run. show-sweep-matrix.py defaults to reading it.

Cost
----
Per (test, tile, compiler): ~17 batches * ~20s EMU + ~10s HW = ~8 min
with grounding enabled (7 sweep events / batch). For 9 Phase-B tests
with one traced tile each, ~75 min. Scale up accordingly for tests with
multiple tiles. HW+EMU runs serially within one trace-sweep.py
invocation; combos run serially at the harness level to keep NPU
contention zero.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
MLIR_AIE_ROOT = REPO_ROOT.parent / "mlir-aie"
BUILD_BASE = MLIR_AIE_ROOT / "build" / "test" / "npu-xrt"
TRACE_SWEEP = REPO_ROOT / "tools" / "trace-sweep.py"
RESULTS_BASE = REPO_ROOT / "build" / "trace-sweep-results"


# Reuse trace-sweep.py's discover_test_config helper so the two tools
# always agree on "what file does this test produce." Importing a module
# whose filename contains a hyphen requires the importlib dance -- and
# dataclass() needs the module registered in sys.modules *before*
# exec_module runs, or its __module__ lookup breaks.
_TRACE_SWEEP_MOD = None


def _load_trace_sweep_module():
    global _TRACE_SWEEP_MOD
    if _TRACE_SWEEP_MOD is not None:
        return _TRACE_SWEEP_MOD
    import importlib.util as _iu
    spec = _iu.spec_from_file_location("trace_sweep", TRACE_SWEEP)
    mod = _iu.module_from_spec(spec)
    sys.modules["trace_sweep"] = mod
    spec.loader.exec_module(mod)
    _TRACE_SWEEP_MOD = mod
    return mod


def _test_config(test_name: str):
    return _load_trace_sweep_module().discover_test_config(test_name)


_TRACE_PATCH_MOD = None


def _load_trace_patch_module():
    global _TRACE_PATCH_MOD
    if _TRACE_PATCH_MOD is not None:
        return _TRACE_PATCH_MOD
    import importlib.util as _iu
    path = REPO_ROOT / "tools" / "trace-patch-events.py"
    spec = _iu.spec_from_file_location("trace_patch_events", path)
    mod = _iu.module_from_spec(spec)
    sys.modules["trace_patch_events"] = mod
    spec.loader.exec_module(mod)
    _TRACE_PATCH_MOD = mod
    return mod


# ---------------------------------------------------------------------------
# Tile discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TracedTile:
    col: int
    row: int
    tile_type: str  # core / memmod / memtile / shim

    @property
    def label(self) -> str:
        return f"{self.tile_type}_c{self.col}r{self.row}"


_TRACE_BLOCK_RE = re.compile(
    r"aie\.trace\s+@\S+\s*\(\s*%tile_(\d+)_(\d+)\s*\)\s*\{(.*?)\}",
    re.DOTALL,
)
_TYPE_RE = re.compile(r"type\s*=\s*(\w+)")

_PACKET_TYPE_TO_TILE_TYPE = {
    "core": "core",
    "mem": "memmod",
    # memtile + shim not observed in current Phase B tests, but keep here
    # so a future kernel that traces them doesn't silently get skipped.
    "memtile": "memtile",
    "shim": "shim",
}


def discover_traced_tiles(traced_mlir: Path) -> List[TracedTile]:
    """Parse an aie-hw-cycles-traced.mlir and return traced tiles.

    The MLIR is emitted by the --with-hw-cycles trace injection pass and
    is the source of truth for which tiles actually have trace units
    enabled. Each `aie.trace @sym(%tile_C_R) { ... type = X ... }` block
    yields one TracedTile.
    """
    if not traced_mlir.is_file():
        return []
    text = traced_mlir.read_text()
    tiles: List[TracedTile] = []
    for m in _TRACE_BLOCK_RE.finditer(text):
        col, row = int(m.group(1)), int(m.group(2))
        body = m.group(3)
        tm = _TYPE_RE.search(body)
        if not tm:
            # A trace block without a packet type is malformed; skip
            # rather than assume, so the failure mode is a missing matrix
            # row, not a silently mis-typed sweep.
            continue
        ptype = tm.group(1)
        tile_type = _PACKET_TYPE_TO_TILE_TYPE.get(ptype)
        if tile_type is None:
            continue
        tiles.append(TracedTile(col=col, row=row, tile_type=tile_type))
    return tiles


# ---------------------------------------------------------------------------
# Test discovery
# ---------------------------------------------------------------------------

def discover_tests(test_filter: Optional[str]) -> List[str]:
    """Find all tests that have a hw-cycles traced MLIR compiled.

    Filter is a substring match (so --test vector_scalar picks up
    vector_scalar_using_dma). Returns a sorted list.
    """
    if not BUILD_BASE.is_dir():
        raise FileNotFoundError(f"no mlir-aie build tree at {BUILD_BASE}")
    candidates: List[str] = []
    for path in BUILD_BASE.glob("*/aie-hw-cycles-traced.mlir"):
        name = path.parent.name
        if test_filter and test_filter not in name:
            continue
        candidates.append(name)
    return sorted(candidates)


# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------

def allocate_results_dir(label: Optional[str]) -> Path:
    """Create a dated results directory and refresh the `latest` symlink.

    If label is given, use it verbatim (allows side-by-side baseline vs
    refactor runs). Otherwise date the directory, adding -N suffixes if
    today already has runs.
    """
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    if label:
        out = RESULTS_BASE / label
        out.mkdir(parents=True, exist_ok=False)
    else:
        stem = date.today().strftime("%Y%m%d")
        out = RESULTS_BASE / stem
        n = 1
        while out.exists():
            n += 1
            out = RESULTS_BASE / f"{stem}-{n}"
        out.mkdir(parents=True, exist_ok=False)
    latest = RESULTS_BASE / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out.name)
    return out


# ---------------------------------------------------------------------------
# Sweep invocation
# ---------------------------------------------------------------------------

def run_sweep(
    test: str,
    compiler: str,
    tiles: List[TracedTile],
    out_dir: Path,
    ground_event: Optional[str],
    no_hw: bool,
    no_emu: bool,
    events: Optional[str],
    timeout_sec: Optional[float],
) -> dict:
    """Call tools/trace-sweep.py in multi-tile mode for one (test, compiler)
    with its full tile list. One kernel invocation covers every tile; the
    per-tile output JSONs land in out_dir.

    Keeps the subprocess boundary so a trace-sweep crash on one combo
    doesn't take down the whole harness run.  timeout_sec bounds wall time
    so a wedged EMU can't hold up the whole sweep.
    """
    tiles_arg = ",".join(f"{t.col}:{t.row}:{t.tile_type}" for t in tiles)
    cmd = [
        sys.executable, str(TRACE_SWEEP),
        "--test", test,
        "--compiler", compiler,
        "--tiles", tiles_arg,
        "--out-dir", str(out_dir),
    ]
    if ground_event:
        cmd += ["--ground-event", ground_event]
    if no_hw:
        cmd.append("--no-hw")
    if no_emu:
        cmd.append("--no-emu")
    if events:
        cmd += ["--events", events]

    log_path = out_dir / f"{test}.{compiler}.log"
    t0 = time.time()
    rc: int
    timed_out = False
    with log_path.open("w") as lf:
        try:
            rc = subprocess.run(
                cmd, stdout=lf, stderr=subprocess.STDOUT,
                timeout=timeout_sec,
            ).returncode
        except subprocess.TimeoutExpired:
            # subprocess.run already killed the child when timeout fires.
            # Tag the combo as timed-out so the summary surfaces it.
            rc = 124  # timeout convention (GNU `timeout` uses 124)
            timed_out = True
    dt = round(time.time() - t0, 1)
    # Count which per-tile outputs actually landed. Each tile should
    # produce <test>.<compiler>.<tile.label>.json; missing outputs mean
    # the sweep crashed before that tile was written.
    produced = []
    missing = []
    for tile in tiles:
        p = out_dir / f"{test}.{compiler}.{tile.label}.json"
        (produced if p.is_file() else missing).append(tile.label)
    return {
        "test": test,
        "compiler": compiler,
        "tiles": [{"col": t.col, "row": t.row, "type": t.tile_type} for t in tiles],
        "outputs_produced": produced,
        "outputs_missing": missing,
        "log": log_path.name,
        "rc": rc,
        "timed_out": timed_out,
        "elapsed_sec": dt,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--test", help="test name or substring filter "
                                    "(default: all tests with hw-cycles built)")
    ap.add_argument("--compiler", default="chess",
                    choices=["chess", "peano", "both"],
                    help="which compiler(s) to sweep (default: chess)")
    ap.add_argument("--ground-event", default="USER_EVENT_1",
                    help="grounding event for cross-batch merge; "
                         "set to 'none' to disable grounding")
    ap.add_argument("--label", help="results dir name (default: date-stamped)")
    ap.add_argument("--events",
                    help="restrict sweep to a comma-separated event list "
                         "(forwarded to trace-sweep.py)")
    ap.add_argument("--no-hw", action="store_true")
    ap.add_argument("--no-emu", action="store_true")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel combos (EMU-only). --jobs >1 requires "
                         "--no-hw because HW contention serialises through "
                         "the NPU. Default: 1.")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="per-combo wall-time limit in seconds; a timed-out "
                         "combo is killed and recorded rc=124 in summary.json "
                         "so the harness stays responsive when one EMU wedges "
                         "(default: 1800 = 30 min)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list combos without running")
    args = ap.parse_args()

    if args.jobs < 1:
        print("--jobs must be >=1", file=sys.stderr)
        return 2
    if args.jobs > 1 and not args.no_hw:
        print("--jobs >1 requires --no-hw (NPU access is serial). Either "
              "run HW serially first with --no-emu, then EMU in parallel "
              "with --no-hw --jobs N.", file=sys.stderr)
        return 2

    tests = discover_tests(args.test)
    if not tests:
        filter_desc = f" matching {args.test!r}" if args.test else ""
        print(f"no tests{filter_desc} found under {BUILD_BASE}",
              file=sys.stderr)
        return 1

    if args.compiler == "both":
        compilers = ["chess", "peano"]
    else:
        compilers = [args.compiler]

    ground = None if (args.ground_event or "").lower() == "none" else args.ground_event

    # Build the full combo list up front so dry-run shows the plan and we
    # can write a summary even if the run is interrupted partway.
    combos = []
    for test in tests:
        traced = BUILD_BASE / test / "aie-hw-cycles-traced.mlir"
        tiles = discover_traced_tiles(traced)
        if not tiles:
            print(f"[skip] {test}: no traced tiles in {traced}",
                  file=sys.stderr)
            continue
        # Ask trace-sweep.py what this test actually produces. That way
        # control-packet tests (which use aie_run_seq.bin, not insts.bin)
        # come in too -- the sweep can patch any runtime-sequence file the
        # test emits, as long as the compiled xclbin has trace routing on
        # the target tile.
        cfg = _test_config(test)
        for comp in compilers:
            comp_dir = BUILD_BASE / test / comp
            if not (comp_dir / "aie.xclbin").is_file():
                print(f"[skip] {test}/{comp}: no xclbin; run "
                      f"emu-bridge-test.sh --with-hw-cycles --{comp}-only "
                      f"-v {test}", file=sys.stderr)
                continue
            insts_path = comp_dir / cfg.insts_name
            if not insts_path.is_file():
                print(f"[skip] {test}/{comp}: runtime-sequence file "
                      f"{cfg.insts_name!r} missing from build dir "
                      f"(discovered from {test}/run.lit)", file=sys.stderr)
                continue
            # Patch-based sweep needs Trace_Event writes in the runtime
            # sequence. Some tests (e.g. ctrl_packet_reconfig_1x4_cores)
            # program the trace config via control-packet DMA writes that
            # live in ctrlpkt.bin, not via Write32 ops in the runtime
            # sequence. Probe here so the first batch doesn't have to
            # discover that combo-by-combo.
            tp = _load_trace_patch_module()
            data = insts_path.read_bytes()
            schedulable_tiles = [
                tile for tile in tiles
                if tp.probe_slot_capacity(data, tile.col, tile.row, tile.tile_type) > 0
            ]
            if not schedulable_tiles:
                print(f"[skip] {test}/{comp}: no Trace_Event writes in "
                      f"{cfg.insts_name} for any traced tile (trace config "
                      f"likely lives in ctrlpkt.bin -- patching those is "
                      f"not yet implemented)", file=sys.stderr)
                continue
            # Multi-tile grouping: one combo per (test, comp) carrying
            # the full list of schedulable tiles. trace-sweep.py's
            # multi-tile mode runs the kernel once and splits the trace
            # per-tile on the way out, so we don't waste kernel runs.
            combos.append((test, comp, schedulable_tiles))

    print(f"[sweep-all] {len(combos)} combo(s) "
          f"({sum(len(c[2]) for c in combos)} tiles) over "
          f"{len(tests)} tests / {len(compilers)} compiler(s)")
    for test, comp, tiles in combos:
        tile_labels = ",".join(t.label for t in tiles)
        print(f"  {test:40s} {comp:6s} {tile_labels}")
    if args.dry_run:
        return 0

    out_dir = allocate_results_dir(args.label)
    print(f"[sweep-all] writing to {out_dir}")

    def _submit(combo):
        test, comp, tiles = combo
        return run_sweep(
            test=test, compiler=comp, tiles=tiles, out_dir=out_dir,
            ground_event=ground, no_hw=args.no_hw, no_emu=args.no_emu,
            events=args.events, timeout_sec=args.timeout,
        )

    statuses = []
    t_total = time.time()
    if args.jobs == 1:
        for i, combo in enumerate(combos, 1):
            test, comp, tiles = combo
            tile_labels = ",".join(t.label for t in tiles)
            print(f"[sweep-all] ({i}/{len(combos)}) {test} {comp} [{tile_labels}]",
                  flush=True)
            status = _submit(combo)
            if status["rc"] != 0:
                tag = "TIMEOUT" if status.get("timed_out") else f"rc={status['rc']}"
                print(f"  !! {tag}, see {status['log']}", file=sys.stderr)
            statuses.append(status)
    else:
        # Parallel EMU: we only hit here when --no-hw is set (enforced
        # above). Each worker spawns its own trace-sweep.py subprocess
        # which runs its own emulator inside bridge-trace-runner, so the
        # fan-out is filesystem-isolated. Shared state: the results dir,
        # but every combo writes per-tile to unique filenames.
        print(f"[sweep-all] running {len(combos)} combo(s) with {args.jobs} "
              f"parallel workers (EMU only)", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_submit, c): c for c in combos}
            done = 0
            for fut in as_completed(futures):
                done += 1
                status = fut.result()
                statuses.append(status)
                test = status["test"]
                comp = status["compiler"]
                tile_labels = ",".join(
                    f"{t['type']}_c{t['col']}r{t['row']}" for t in status["tiles"]
                )
                if status["rc"] == 0:
                    print(f"[sweep-all] ({done}/{len(combos)}) OK {test} {comp} "
                          f"[{tile_labels}] ({status['elapsed_sec']}s)", flush=True)
                else:
                    tag = "TIMEOUT" if status.get("timed_out") else f"rc={status['rc']}"
                    print(f"[sweep-all] ({done}/{len(combos)}) !! {tag} "
                          f"{test} {comp} [{tile_labels}], see {status['log']}",
                          file=sys.stderr, flush=True)

    # Post-hoc data-quality pass. trace-sweep.py exits 0 even when every
    # batch errored (errors go into the per-event JSON so one bad batch
    # can't kill a sweep), so rc=0 doesn't mean the combo produced usable
    # data. Re-read each produced per-tile JSON and count HW/EMU errors.
    # Each combo now covers multiple tiles, so aggregate per-tile rows.
    want_hw = not args.no_hw
    want_emu = not args.no_emu
    for status in statuses:
        total_rows = hw_errors = emu_errors = 0
        per_tile_health = []
        for tile in status["tiles"]:
            label = f"{tile['type']}_c{tile['col']}r{tile['row']}"
            out_path = out_dir / f"{status['test']}.{status['compiler']}.{label}.json"
            tile_rows = tile_hw_err = tile_emu_err = 0
            if out_path.is_file():
                try:
                    d = json.loads(out_path.read_text())
                    for row in d.get("events", []):
                        tile_rows += 1
                        if want_hw and row.get("hw", {}).get("error"):
                            tile_hw_err += 1
                        if want_emu and row.get("emu", {}).get("error"):
                            tile_emu_err += 1
                except Exception:
                    pass
            total_rows += tile_rows
            hw_errors += tile_hw_err
            emu_errors += tile_emu_err
            per_tile_health.append({
                "tile": label,
                "rows": tile_rows,
                "hw_errors": tile_hw_err,
                "emu_errors": tile_emu_err,
            })
        status["rows_total"] = total_rows
        status["rows_hw_error"] = hw_errors
        status["rows_emu_error"] = emu_errors
        status["per_tile_health"] = per_tile_health

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ground_event": ground,
        "compilers": compilers,
        "tests": tests,
        "total_elapsed_sec": round(time.time() - t_total, 1),
        "combos": statuses,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    orchestrator_ok = sum(1 for s in statuses if s["rc"] == 0)
    # A combo is "clean" iff every event row on every requested side
    # has data. Partial / all-bad combos are surfaced separately so the
    # user sees "orchestrator said OK but HW errored on every batch"
    # rather than the hidden success-by-exit-code reading.
    def _clean(s):
        if s["rc"] != 0:
            return False
        if s.get("rows_total", 0) == 0:
            return False
        if want_hw and s.get("rows_hw_error", 0) > 0:
            return False
        if want_emu and s.get("rows_emu_error", 0) > 0:
            return False
        return True
    clean = sum(1 for s in statuses if _clean(s))
    partial = [s for s in statuses if s["rc"] == 0 and not _clean(s)]
    print(f"[sweep-all] done: {clean}/{len(statuses)} combos with clean data "
          f"({orchestrator_ok}/{len(statuses)} orchestrator rc=0), "
          f"total {summary['total_elapsed_sec']}s")
    for s in partial:
        tile_labels = ",".join(
            f"{t['type']}_c{t['col']}r{t['row']}" for t in s["tiles"]
        )
        bits = []
        if want_hw and s.get("rows_hw_error", 0):
            bits.append(f"HW errors on {s['rows_hw_error']}/{s['rows_total']} rows")
        if want_emu and s.get("rows_emu_error", 0):
            bits.append(f"EMU errors on {s['rows_emu_error']}/{s['rows_total']} rows")
        print(f"  partial: {s['test']} [{tile_labels}]: {'; '.join(bits)}",
              file=sys.stderr)
    return 0 if clean == len(statuses) else 2


if __name__ == "__main__":
    sys.exit(main())
