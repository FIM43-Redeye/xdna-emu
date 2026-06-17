#!/usr/bin/env python3
"""Self-owned NPU trace-capture engine for #140.

Takes a batch plan and produces correctly-labeled events.json per batch per run
on real hardware, reusing three audited primitives (register patcher, XRT
runner, in-tree decoder) and owning column-free exact labeling + N-run coverage.

See docs/superpowers/specs/2026-06-17-trace-capture-engine-design.md.
"""
import json
import math
from pathlib import Path
from shlex import quote
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
_EVENTS_HEADER = (_REPO.parent / "mlir-aie/build/include/xaienginecdo_static/"
                  "xaiengine/xaie_events_aieml.h")
_MOD_PREFIX = {"core": "CORE", "memmod": "MEM", "memtile": "MEM_TILE", "shim": "PL"}


def load_event_ids(tile_type: str) -> Dict[str, int]:
    """{event_name: numeric_id} for a tile-type, from the aie-rt events header."""
    full = f"XAIEML_EVENTS_{_MOD_PREFIX[tile_type]}_"
    exclude = "XAIEML_EVENTS_MEM_TILE_" if tile_type == "memmod" else None
    out: Dict[str, int] = {}
    for line in _EVENTS_HEADER.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "#define" and parts[1].startswith(full):
            if exclude and parts[1].startswith(exclude):
                continue
            name = parts[1][len(full):]
            val = parts[2].rstrip("U")
            if val.isdigit():
                out.setdefault(name, int(val))   # first definition wins (stable)
    return out


PKT_TO_TILE_TYPE = {0: "core", 1: "memmod", 2: "shim", 3: "memtile"}


class CaptureError(Exception):
    pass


def _get(ev, attr):
    return ev[attr] if isinstance(ev, dict) else getattr(ev, attr)


def label_events(raw_events, label_map, traced_col: int) -> List[dict]:
    """Apply label_map to raw decoded events, with hard-error guards.

    Each raw event (dict or object with col,row,pkt_type,slot,ts,soc,mode)
    becomes a record {col,row,pkt_type,name,slot,ts,soc,mode}.

    Raises CaptureError if:
    - (pkt_type,row,slot) not in label_map (unconfigured slot)
    - col != traced_col (foreign column / start_col mismatch)
    """
    out = []
    for ev in raw_events:
        col = _get(ev, "col"); row = _get(ev, "row"); pkt = _get(ev, "pkt_type")
        slot = _get(ev, "slot")
        if col != traced_col:
            raise CaptureError(
                f"foreign column {col} (traced {traced_col}); start_col mismatch")
        key = (pkt, row, slot)
        if key not in label_map:
            raise CaptureError(f"event at unconfigured (pkt,row,slot)={key}")
        out.append({"col": col, "row": row, "pkt_type": pkt,
                    "name": label_map[key], "slot": slot,
                    "ts": _get(ev, "ts"), "soc": _get(ev, "soc"),
                    "mode": _get(ev, "mode")})
    return out


def configure_batch(batch: Dict[str, List[str]], anchor: str = "PERF_CNT_2",
                    mode: int = 0):
    """batch {"col|row|pkt": [names]} -> (patch_spec, label_map).

    label_map is keyed (pkt_type, row, slot) -- column-free by design.

    Each patch-spec entry carries an explicit "mode" so the patcher rewrites the
    tile's Trace_Control0 (bits[1:0]) to the trace mode we decode with. This is
    NOT optional: kernels compile the core trace unit to EVENT_PC (mode 1) by
    default (add_one_using_dma core Trace_Control0 = 0x797a0001), and decoding a
    mode-1 stream with the mode-0 decoder misreads PC bytes as slot bitmasks ->
    phantom fires on slots configured NONE. mode=0 (EVENT_TIME) matches our
    milestone decode (parse_trace EVENT_TIME).
    """
    patch_spec = []
    label_map: Dict[tuple, str] = {}
    for tile_key, names in batch.items():
        col, row, pkt = (int(x) for x in tile_key.split("|"))
        tile_type = PKT_TO_TILE_TYPE[pkt]
        # anchor first (slot 0) if present, then the rest in plan order
        ordered = ([anchor] if anchor in names else []) + [n for n in names if n != anchor]
        if len(ordered) > 8:
            raise ValueError(f"tile {tile_key} has {len(ordered)} events > 8 slots")
        ids = load_event_ids(tile_type)
        event_ids = []
        for slot, name in enumerate(ordered):
            if name not in ids:
                raise ValueError(f"event {name!r} not in {tile_type} table")
            event_ids.append(ids[name])
            label_map[(pkt, row, slot)] = name
        # Pad to 8 slots with NONE (event id 0). Each tile has 8 trace slots;
        # the patcher overwrites only the slots we supply, so any slot we leave
        # short keeps the kernel's compile-time trace event -- which fires on HW
        # and surfaces as an unconfigured-slot hard error at label time. Writing
        # NONE disables those slots. (Matches trace-sweep, which always sends 8.)
        event_ids += [0] * (8 - len(event_ids))
        patch_spec.append({"col": col, "row": row, "tile_type": tile_type,
                           "events": event_ids, "mode": mode})
    return patch_spec, label_map


def coverage_report(configured: Dict[tuple, str], observed_per_run: List[set]) -> dict:
    """Union observed slots across N runs and report gaps.

    Args:
        configured: {(pkt_type, row, slot): name}
        observed_per_run: list (one per run) of sets of (pkt_type, row, slot) that fired

    Returns:
        {
            "n_configured": total configured slots,
            "n_covered": slots observed in at least one run,
            "gaps": [{"pkt_type", "row", "slot", "name"}, ...] sorted by (pkt, row, slot)
        }
    """
    seen = set().union(*observed_per_run) if observed_per_run else set()
    gaps = [{"pkt_type": p, "row": r, "slot": s, "name": configured[(p, r, s)]}
            for (p, r, s) in sorted(configured) if (p, r, s) not in seen]
    return {"n_configured": len(configured),
            "n_covered": len(configured) - len(gaps), "gaps": gaps}


TRACE_SIZE_DEFAULT = 1 << 21


def write_patch_spec(patch_spec, path) -> Path:
    """Write patch spec JSON to file.

    Args:
        patch_spec: list of {col, row, tile_type, events}
        path: output file path

    Returns:
        Path object pointing to the written file
    """
    path = Path(path)
    path.write_text(json.dumps(patch_spec))
    return path


def runner_command(instr, trace_out, trace_size, inputs, outputs) -> str:
    """Build the stdin command line for bridge-trace-runner.

    Args:
        instr: path to instruction binary
        trace_out: output trace file path
        trace_size: maximum trace size in bytes
        inputs: list of input file paths
        outputs: list of output file paths

    Returns:
        Space-joined command line with shell-safe quoting
    """
    parts = ["--instr", str(instr), "--trace-out", str(trace_out),
             "--trace-size", str(trace_size)]
    for p in inputs:
        parts += ["--input", str(p)]
    for p in outputs:
        parts += ["--output", str(p)]
    return " ".join(quote(p) for p in parts)


import subprocess
import sys
from trace_decoder import parse_trace          # in-tree decoder
from trace_decoder.frame import TraceMode

_PATCH_TOOL = _REPO / "tools" / "trace-patch-events.py"


def _read_trace_words(trace_bin: Path):
    import numpy as np
    return np.fromfile(str(trace_bin), dtype="<u4")


def build_active_plan(active, anchor="PERF_CNT_2",
                      anchor_tile="1|2|0", slots=8):
    """{"col|row|pkt": set[names]} -> {"batches": [{"col|row|pkt": [names]}]}.

    Packs each module's active events into batches; the anchor rides slot 0 of
    the anchor tile in every batch (reserving one slot there, 8 elsewhere).
    """
    per_mod = {t: sorted(n for n in names if not (t == anchor_tile and n == anchor))
               for t, names in active.items()}

    def cap(t):
        return slots - 1 if t == anchor_tile else slots

    nb = max([1] + [math.ceil(len(ev) / cap(t)) for t, ev in per_mod.items() if ev])
    batches = []
    for i in range(nb):
        b = {}
        for t, ev in per_mod.items():
            chunk = ev[i * cap(t):(i + 1) * cap(t)]
            names = ([anchor] if t == anchor_tile else []) + chunk
            if names:
                b[t] = names
        b.setdefault(anchor_tile, [anchor])   # anchor present every batch
        batches.append(b)
    return {"batches": batches}


def capture(plan, runner, *, test, out_dir, traced_col=1,
            trace_size=TRACE_SIZE_DEFAULT, instr=None, inputs=(), outputs=()):
    """Per-batch orchestrator: RESET -> patch -> run -> decode -> label -> write.

    For each batch in plan["batches"]:
    - RESET the runner (always, before every run)
    - Write a patch spec via write_patch_spec
    - Invoke the patcher CLI (trace-patch-events.py) via subprocess
    - Run the kernel via runner.run_one(cmd)
    - Decode the resulting trace.bin via parse_trace
    - Label events via label_events
    - Write out_dir/batch_MM/hw/trace.events.json

    Args:
        plan:        {"batches": [batch, ...]} where each batch is {"col|row|pkt": [names]}
        runner:      object with reset() and run_one(cmd_line) -> status_dict
        test:        test name (for the runner command, informational)
        out_dir:     root directory for per-batch output subtrees
        traced_col:  the column whose trace data is expected (default 1)
        trace_size:  maximum trace buffer size in bytes
        instr:       path to the base instruction binary (patched per batch)
        inputs:      list of input file paths forwarded to runner_command
        outputs:     list of output file paths forwarded to runner_command

    Returns:
        List of label_maps, one per batch (each is {(pkt_type, row, slot): name}).

    Raises:
        CaptureError: if runner reports truncation or failure, or if labeling fails.
    """
    out_dir = Path(out_dir)
    label_maps = []
    for i, batch in enumerate(plan["batches"]):
        spec, lmap = configure_batch(batch)
        label_maps.append(lmap)
        bdir = out_dir / f"batch_{i:02d}" / "hw"
        bdir.mkdir(parents=True, exist_ok=True)
        spec_path = write_patch_spec(spec, bdir / "patch.json")
        patched = bdir / "insts.patched.bin"
        subprocess.run([sys.executable, str(_PATCH_TOOL), "--multi-tile",
                        str(spec_path), str(instr), "--output", str(patched)],
                       check=True, capture_output=True)
        trace_bin = bdir / "trace.bin"
        runner.reset()
        status = runner.run_one(runner_command(
            patched, trace_bin, trace_size, list(inputs), list(outputs)))
        if status.get("truncated") or status.get("ok") is False:
            raise CaptureError(f"batch {i}: runner status {status}")
        words = _read_trace_words(trace_bin)
        raw = parse_trace(words, slot_names=None, mode=TraceMode.EVENT_TIME)
        events = label_events(raw, lmap, traced_col)
        (bdir / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": events, "slot_names": {}}))
    return label_maps


# ---------------------------------------------------------------------------
# HW loop driver (Task 9) -- real-NPU wiring + cross-run validation.
#
# This is the one place the engine borrows trace-sweep's proven XRT plumbing
# (RunnerSession: process startup, ready handshake, RESET ack, wedge-snapshot)
# rather than re-deriving it. RunnerSession is imported lazily so the engine's
# pure-logic core (above) stays import-clean and HW coupling only activates
# when HwRunner is actually constructed.
# ---------------------------------------------------------------------------

_MLIR_AIE = _REPO.parent / "mlir-aie"

# add_one_using_dma seed active set, derived from the #140 characterization
# capture (build/experiments/gap140/nondeterminism/add_one_using_dma): the
# events that actually fire, per traced tile-module, on real NPU1.
#
# COLUMN SPACES (the start_col reconcile). Two coexist in this engine:
#   - RELATIVE col (the MLIR/insts.bin logical column) -- what the PATCHER
#     consumes. add_one_using_dma's single column is relative col 0.
#   - ABSOLUTE col (the hardware partition column) -- what the DECODER reports.
#     The driver places this 1-col partition at absolute col 1.
# This seed (patcher input) is therefore keyed on RELATIVE col 0; the decoded
# events land at ABSOLUTE col 1 (traced_col), and the join's anchor_key is in
# absolute space ("1|2|0|PERF_CNT_2"). The label_map is column-free precisely
# so it bridges the two without re-deriving start_col.
SEED_ACTIVE_PLAN = {
    "0|0|2": {                      # shim (rel col 0, row 0) -- DMA milestones
        "DMA_MM2S_0_FINISHED_TASK", "DMA_MM2S_0_START_TASK",
        "DMA_S2MM_0_FINISHED_TASK", "DMA_S2MM_0_START_TASK",
        "DMA_S2MM_0_STREAM_STARVATION", "DMA_S2MM_1_FINISHED_TASK",
        "DMA_S2MM_1_START_TASK",
    },
    "0|1|3": {                      # memtile (rel col 0, row 1) -- stream ports
        "PORT_RUNNING_0", "PORT_RUNNING_1", "PORT_RUNNING_2", "PORT_RUNNING_3",
        "PORT_RUNNING_4", "PORT_RUNNING_5", "PORT_RUNNING_6",
    },
    "0|2|0": {                      # core (rel col 0, row 2) -- anchor lives here
        "INSTR_VECTOR", "LOCK_STALL", "MEMORY_STALL", "PERF_CNT_2", "STREAM_STALL",
    },
    "0|2|1": {                      # memmod (rel col 0, row 2)
        "CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1", "CONFLICT_DM_BANK_2",
        "CONFLICT_DM_BANK_3", "DMA_MM2S_0_START_TASK", "DMA_S2MM_0_START_TASK",
        "EDGE_DETECTION_EVENT_0",
    },
}


def _discover_xclbin_insts(test, compiler="chess"):
    """Locate the built (xclbin, insts.bin) for a test, same layout as trace-sweep.

    add_one_using_dma's run.lit emits --npu-insts-name=insts.bin; we hardcode
    that filename here since this cycle seeds add_one specifically (generalizing
    to discover_test_config is deferred -- see the plan's deferred section).
    """
    build_dir = _MLIR_AIE / "build" / "test" / "npu-xrt" / test / compiler
    xclbin = build_dir / "aie.xclbin"
    insts = build_dir / "insts.bin"
    if not xclbin.is_file():
        raise CaptureError(f"xclbin not found: {xclbin} (is the kernel built?)")
    if not insts.is_file():
        raise CaptureError(f"insts.bin not found: {insts}")
    return xclbin, insts


class HwRunner:
    """Adapts trace-sweep's RunnerSession to capture()'s string-line contract.

    capture() builds a single command line via runner_command() and calls
    runner.run_one(cmd_line). RunnerSession instead takes structured args and
    builds its own line (adding wedge-snapshot/forensics). We bridge by parsing
    the line we ourselves produced (safe -- we control runner_command's format)
    back into RunnerSession.run_one's keyword args, so the engine gets all of
    RunnerSession's hardened plumbing while honoring the reviewed capture()
    interface (reset() + run_one(cmd_line) -> status_dict).
    """

    def __init__(self, xclbin, stderr_log, side="HW"):
        import importlib.util
        sweep_path = _REPO / "tools" / "trace-sweep.py"
        spec = importlib.util.spec_from_file_location("_trace_sweep_mod", str(sweep_path))
        mod = importlib.util.module_from_spec(spec)
        # Register before exec_module: trace-sweep.py uses @dataclass, and on
        # Python 3.13 dataclass processing does sys.modules.get(cls.__module__),
        # which returns None (AttributeError) unless the synthetic module name
        # is already registered.
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        self._RunnerSession = mod.RunnerSession
        stderr_log = Path(stderr_log)
        stderr_log.parent.mkdir(parents=True, exist_ok=True)
        # hw_runner_env = {} (HW inherits os.environ; run under env -u XDNA_EMU
        # so the bridge targets the real NPU, not the emulator).
        self._session = mod.RunnerSession(
            xclbin=Path(xclbin), runner_env={}, side=side, stderr_log=stderr_log)

    def reset(self):
        self._session.reset()

    def run_one(self, cmd_line):
        import shlex
        toks = shlex.split(cmd_line)
        instr = trace_out = None
        trace_size = TRACE_SIZE_DEFAULT
        inputs, outputs = [], []
        i = 0
        while i < len(toks):
            t = toks[i]
            if t == "--instr":
                instr = toks[i + 1]; i += 2
            elif t == "--trace-out":
                trace_out = toks[i + 1]; i += 2
            elif t == "--trace-size":
                trace_size = int(toks[i + 1]); i += 2
            elif t == "--input":
                inputs.append(toks[i + 1]); i += 2
            elif t == "--output":
                outputs.append(toks[i + 1]); i += 2
            else:
                i += 1
        return self._session.run_one(
            instr=Path(instr), trace_out=Path(trace_out),
            inputs=[Path(p) for p in inputs] or None,
            outputs=[Path(p) for p in outputs] or None,
            trace_size=trace_size)

    def close(self):
        self._session.close()


def run_loop(test, active_plan, n_runs, out, compiler="chess", traced_col=1,
             anchor_tile="0|2|0"):
    """Capture the active plan on HW across N runs, then cross-run validate.

    Per run: a fresh HwRunner captures every batch of build_active_plan(active)
    to out/run_NN/batch_MM/hw/trace.events.json. After all runs: union coverage
    (named gaps), then the cross-run derivability graph (build_derivability_graph
    measures std across the runs -- the deterministic backbone gets no edge, the
    DMA milestones surface as stochastic_roots), the synthesized plan, and a
    joined record stream for one run. Writes graph.json/coverage.json/
    joined.perfetto.json under out and returns a summary dict.
    """
    import trace_join
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    xclbin, insts = _discover_xclbin_insts(test, compiler)
    # active_plan is in relative-col space (patcher input); anchor_tile matches.
    plan = build_active_plan(active_plan, anchor_tile=anchor_tile)
    print(f"[run_loop] {test}: {len(plan['batches'])} batch(es), {n_runs} runs; "
          f"xclbin={xclbin}", flush=True)

    run_dirs, configured, observed_per_run = [], {}, []
    for i in range(n_runs):
        run_dir = out / f"run_{i:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        runner = HwRunner(xclbin, stderr_log=run_dir / "hw.runner.log")
        try:
            label_maps = capture(plan, runner, test=test, out_dir=run_dir,
                                 traced_col=traced_col, instr=insts)
        finally:
            runner.close()
        run_dirs.append(str(run_dir))
        for lm in label_maps:
            configured.update(lm)
        obs = set()
        for ev_path in sorted(run_dir.glob("batch_*/hw/trace.events.json")):
            for e in json.loads(ev_path.read_text())["events"]:
                obs.add((e["pkt_type"], e["row"], e["slot"]))
        observed_per_run.append(obs)
        print(f"[run_loop] run {i + 1}/{n_runs}: {len(obs)} slots fired", flush=True)

    cov = coverage_report(configured, observed_per_run)
    print(f"[run_loop] coverage: {cov['n_covered']}/{cov['n_configured']} "
          f"covered, {len(cov['gaps'])} gap(s)", flush=True)
    for g in cov["gaps"]:
        print(f"  GAP: {g}", flush=True)

    graph = trace_join.build_derivability_graph(run_dirs)
    print(f"[run_loop] graph: {len(graph['nodes'])} nodes, {len(graph['roots'])} "
          f"roots, {len(graph['stochastic_roots'])} stochastic root(s)", flush=True)
    for sr in graph["stochastic_roots"]:
        print(f"  STOCHASTIC_ROOT: {sr}", flush=True)
    plan2 = trace_join.synthesize_plan(graph)
    print(f"[run_loop] synthesized plan: {plan2['n_batches']} batch(es)", flush=True)
    records = trace_join.join_run(run_dirs[0], graph)

    (out / "graph.json").write_text(json.dumps(graph, indent=2))
    (out / "coverage.json").write_text(json.dumps(cov, indent=2))
    (out / "joined.perfetto.json").write_text(
        json.dumps(trace_join.to_perfetto(records)))
    print(f"[run_loop] wrote graph.json, coverage.json, joined.perfetto.json "
          f"({len(records)} records) under {out}", flush=True)
    return {"graph": graph, "plan": plan2, "coverage": cov,
            "n_records": len(records)}
