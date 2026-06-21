"""Shared bridge-trace run+parse core, extracted from trace-sweep.py.

RunnerSession (long-lived bridge-trace-runner in --batch-stdin mode), ParseSession
(long-lived parse-trace decode server), and the single-batch run+parse cycle
_run_one_side. The sweep orchestration (sweep_multi/sweep_lockstep) stays in
trace-sweep.py and imports these. trace_capture.HwRunner adapts RunnerSession here.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
MLIR_AIE_ROOT = REPO_ROOT.parent / "mlir-aie"
EVENTS_HEADER = (
    MLIR_AIE_ROOT / "build" / "include" / "xaienginecdo_static"
    / "xaiengine" / "xaie_events_aieml.h"
)
RUNNER = Path(os.environ.get(
    "BRIDGE_TRACE_RUNNER",
    REPO_ROOT / "bridge-runner" / "build" / "bridge-trace-runner",
))
PATCH_TOOL = REPO_ROOT / "tools" / "trace-patch-events.py"
PARSE_TOOL = REPO_ROOT / "tools" / "parse-trace.py"

# xaie_events_aieml.h uses module prefixes that map to our tile types.
_MOD_TO_TILE_TYPE = {
    "CORE": "core",
    "MEM": "memmod",
    "MEM_TILE": "memtile",
    "PL": "shim",
}
_TILE_TYPE_TO_MOD = {v: k for k, v in _MOD_TO_TILE_TYPE.items()}

_GROUNDING_BY_TILE_TYPE = {
    "core":    "PERF_CNT_2,INSTR_EVENT_0,INSTR_EVENT_1",
    "memmod":  "PERF_CNT_2",
    "memtile": "PERF_CNT_2",
    "shim":    "PERF_CNT_2",
}

_MODE_INT = {
    "event_time": 0,
    "event_pc":   1,
    "inst_exec":  2,
}


# ---------------------------------------------------------------------------
# Runner wrappers
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    ok: bool
    cycles: Optional[int]
    events_count: Optional[int]
    per_event_count: Dict[str, int] = None  # populated by _relabel_events
    error: Optional[str] = None

    def __post_init__(self):
        if self.per_event_count is None:
            self.per_event_count = {}


def _run_patch(
    original_insts: Path,
    patched_insts: Path,
    col: int,
    row: int,
    tile_type: str,
    event_ids: List[int],
) -> None:
    spec = ",".join(str(e) for e in event_ids)
    cmd = [
        sys.executable, str(PATCH_TOOL), str(original_insts),
        "--col", str(col), "--row", str(row), "--tile-type", tile_type,
        "--events", spec, "--output", str(patched_insts),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _run_patch_multi(
    original_insts: Path,
    patched_insts: Path,
    patches: List[Tuple[int, int, str, List[int]]],
) -> None:
    """Apply multiple tile patches to one insts.bin in sequence.

    Each entry is (col, row, tile_type, event_ids). The patches are
    independent (they target disjoint Trace_Event registers because the
    NPU address encoding includes (col, row)), so chaining them -- read
    original, patch tile A, patch tile B over the intermediate, write
    final -- produces the same bytes as a single atomic multi-tile
    patch would.
    """
    if not patches:
        # Copy through unchanged -- downstream runner still needs the file.
        patched_insts.write_bytes(original_insts.read_bytes())
        return
    # Start from the original; chain patches through a scratch file so we
    # can invoke the existing single-tile patcher subprocess unchanged.
    scratch = patched_insts.parent / f"{patched_insts.name}.scratch"
    src = original_insts
    for i, (col, row, tile_type, event_ids) in enumerate(patches):
        dst = patched_insts if i == len(patches) - 1 else scratch
        _run_patch(src, dst, col, row, tile_type, event_ids)
        src = dst
    if scratch.exists():
        scratch.unlink()


def _relabel_events(
    events_json: Path,
    col: int, row: int,
    tile_type: str,
    batch: List,
) -> Tuple[Dict[str, int], List[Dict]]:
    """Rewrite slot_names and per-event `name` fields in an events.json so
    they reflect what the patched insts.bin actually programmed, and
    return (per-event fire counts, filtered event list for this tile).

    parse-trace.py reads slot names from the original (pre-patch) MLIR --
    the trace unit itself only records slot indices, so the raw data is
    already correct. This function fixes only the labelling layer so
    downstream consumers see the right event names.

    The returned filtered list is the subset of events whose (row,
    pkt_type) match this tile, with the `name` field overwritten to the
    current batch's label. It's ready to feed into _anchor_events.
    """
    if not events_json.exists():
        return {}, []
    try:
        doc = json.loads(events_json.read_text())
    except Exception:
        return {}, []
    # tile_type -> key in slot_names dict. parse-trace uses "mem" for
    # memmod; everything else matches our naming.
    slot_key = "mem" if tile_type == "memmod" else tile_type
    names = [e.name for e in batch] + [""] * (8 - len(batch))
    if "slot_names" in doc:
        doc["slot_names"].setdefault(slot_key, [""] * 8)
        doc["slot_names"][slot_key] = names

    # Column-axis caveat: HW trace records absolute columns (the runtime
    # allocator picks a start_col at launch), while EMU uses relative
    # columns starting at 0. Filtering strictly by col would drop every HW
    # event when start_col != 0. Since the sweep only enables one tile at
    # a time, we filter by (row, expected packet type) and let the slot
    # index carry identity. This works as long as stray events from
    # adjacent tiles don't share the same (row, slot) -- which they
    # wouldn't, because adjacent tiles aren't routed into this trace BO.
    pkt_for_tile = 1 if tile_type == "memmod" else 0
    per_slot: Dict[int, int] = {}
    filtered: List[Dict] = []
    for ev in doc.get("events", []):
        if ev.get("row") != row:
            continue
        if ev.get("pkt_type") != pkt_for_tile:
            continue
        slot = ev.get("slot")
        if slot is None or slot >= len(names) or not names[slot]:
            continue
        ev["name"] = names[slot]
        per_slot[slot] = per_slot.get(slot, 0) + 1
        filtered.append(ev)

    events_json.write_text(json.dumps(doc, indent=2) + "\n")
    # Map slot -> event name
    counts = {names[s]: n for s, n in per_slot.items() if names[s]}
    return counts, filtered


class RunnerSession:
    """Long-lived bridge-trace-runner process in --batch-stdin mode.

    Holds a single XRT device + xclbin across many patched-instr runs
    so per-launch overhead drops from ~228 ms (fresh process) to
    ~90 ms (shared device, fresh hw_context per run). The hw_context
    is rebuilt per launch inside the runner -- see the runner's
    ``reuse_context_across_runs`` docstring for the reason the outer
    process can't safely hold a single hw_context across runs yet.

    Use as a context manager or call close() explicitly. If the
    subprocess dies or blocks, run_one() raises RuntimeError so the
    caller can restart the session rather than silently hang.
    """

    def __init__(self, xclbin: Path, runner_env: Dict[str, str],
                 side: str, stderr_log: Path, verbose: bool = False,
                 cdo_preambles: Optional[List[Path]] = None,
                 trace_buf_idx: Optional[int] = None,
                 reuse_ctx: bool = False):
        self.side = side
        self.stderr_log = stderr_log
        self._stderr_fh = stderr_log.open("w")
        # Wedge forensics: tell the runner to dump an AIE_RW_ACCESS
        # register snapshot of the perf tile when run.wait() returns
        # non-COMPLETED. Captures core PC/status/control, timer, lock
        # values, and DMA channel state -- the on-tile evidence we lose
        # to driver recovery once the snapshot window closes. See
        # docs/superpowers/findings/2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md
        # for what we'd want to inspect after a CHAIN_EXEC_NPU drop.
        # Auto-created per session under the runner-log directory; if a
        # wedge fires, look for "wedge snapshot written to ..." lines in
        # the stderr log to find the file.
        self._snapshot_dir = stderr_log.parent / "wedge-snapshots"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        # Per-call CLI fragments common to every run on this session.
        # Folded into run_one() rather than the outer cmd because the
        # runner's batch-stdin protocol re-parses every line through the
        # same parser, so they must live on the per-line side.
        self._cdo_preambles = list(cdo_preambles or [])
        self._trace_buf_idx = trace_buf_idx
        cmd = [str(RUNNER), "--batch-stdin", "--xclbin", str(xclbin)]
        if verbose:
            cmd.append("-v")
        env = os.environ.copy()
        env.update(runner_env)
        # Reuse-ctx mode enables BRIDGE_RUNNER_REUSE_CONTEXT in the
        # subprocess. Combined with --cdo-preamble that re-applies the
        # init/enable CDOs each launch, this lets the runner skip
        # hw_context teardown between runs (saving ~90 ms per launch on
        # Phoenix). Without the preamble, this mode hits the alternating
        # state=8/state=6 timeout pattern.
        if reuse_ctx:
            env["BRIDGE_RUNNER_REUSE_CONTEXT"] = "1"
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_fh,
            env=env,
            text=True,
            bufsize=1,
        )
        ready_line = self.proc.stdout.readline().strip()
        if not ready_line:
            self.close()
            raise RuntimeError(f"{side} runner died before ready")
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as e:
            self.close()
            raise RuntimeError(
                f"{side} runner first line not JSON: {ready_line!r}") from e
        if ready.get("event") != "ready":
            self.close()
            raise RuntimeError(
                f"{side} runner first line not 'ready': {ready_line!r}")

    def run_one(
        self,
        instr: Path,
        trace_out: Path,
        inputs: Optional[List[Path]] = None,
        outputs: Optional[List[Path]] = None,
        ctrlpkts: Optional[List[Path]] = None,
        trace_size: int = 1 << 20,
    ) -> dict:
        """Dispatch one run. Returns the parsed JSON status dict.

        The status dict has keys {"ok", "trace_out", "elapsed_ms"} on
        success and an additional "error" string on failure; parse
        errors (invalid CLI line) surface as ok=false with a "parse:"
        error prefix.
        """
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError(f"{self.side} runner has exited")
        parts = [
            "--instr", str(instr),
            "--trace-out", str(trace_out),
            "--trace-size", str(trace_size),
        ]
        for p in (inputs or []):
            parts += ["--input", str(p)]
        for p in (outputs or []):
            parts += ["--output", str(p)]
        for p in (ctrlpkts or []):
            parts += ["--ctrlpkt", str(p)]
        for p in self._cdo_preambles:
            parts += ["--cdo-preamble", str(p)]
        if self._trace_buf_idx is not None:
            parts += ["--trace-buf-idx", str(self._trace_buf_idx)]
        # Always opt into the wedge-snapshot path -- the runner only
        # writes a file when run.wait() actually times out, so this is
        # a no-op on healthy runs.
        parts += ["--snapshot-on-timeout", str(self._snapshot_dir)]
        # Our argument values never contain spaces in this codebase,
        # but quote paths defensively so a future path with spaces
        # doesn't silently corrupt tokenisation on the C++ side.
        def quote(s: str) -> str:
            return f'"{s}"' if (" " in s or "\t" in s) else s
        line = " ".join(quote(p) for p in parts)
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError as e:
            raise RuntimeError(f"{self.side} runner stdin closed") from e
        resp = self.proc.stdout.readline()
        if not resp:
            raise RuntimeError(f"{self.side} runner produced no response")
        try:
            return json.loads(resp)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"{self.side} runner non-JSON response: {resp!r}") from e

    def reset(self) -> None:
        """Tell the runner to drop its cached hw_context/kernel/BO pool.

        The next run_one() will rebuild from scratch, re-allocating the
        NPU partition and zeroing the shim DMA BD write counters. Used
        between batches in the parallel EMU sweep so pooled sessions
        produce identical results regardless of which session services a
        given batch (otherwise the trace shim DMA's cumulative offset
        leaks across the pool and per-batch event counts diverge).
        """
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError(f"{self.side} runner has exited")
        try:
            self.proc.stdin.write("RESET\n")
            self.proc.stdin.flush()
        except BrokenPipeError as e:
            raise RuntimeError(f"{self.side} runner stdin closed") from e
        resp = self.proc.stdout.readline()
        if not resp:
            raise RuntimeError(
                f"{self.side} runner produced no reset response")
        try:
            ack = json.loads(resp)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"{self.side} runner non-JSON reset response: {resp!r}"
            ) from e
        if ack.get("event") != "reset":
            raise RuntimeError(
                f"{self.side} runner unexpected reset response: {ack!r}")

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None
        if self._stderr_fh is not None and not self._stderr_fh.closed:
            self._stderr_fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class ParseSession:
    """Long-lived parse-trace.py decode server.

    Why this exists: parse-trace.py imports mlir-aie + numpy, which
    costs ~620 ms per Python startup. A 32-batch sweep that decodes
    once per side per batch pays that cost 32-64 times -- about 20-40
    seconds purely on imports, or 75% of the total sweep wall clock.
    Spawning one decoder process per sweep amortizes the import to a
    single ~430 ms startup, dropping subsequent decodes to ~100 ms each
    (~6x per-decode speedup).

    Protocol mirrors RunnerSession: the subprocess prints a "ready"
    event on startup, then accepts one JSON request per stdin line and
    emits one JSON response per stdout line.
    """

    def __init__(self, side: str, stderr_log: Path,
                 env_for_parse: Optional[Dict[str, str]] = None):
        self.side = side
        self.stderr_log = stderr_log
        self._stderr_fh = stderr_log.open("w")
        env = os.environ.copy()
        if env_for_parse:
            env.update(env_for_parse)
        # Always inject the mlir-aie install path -- the same way
        # _parse_trace_bin does in fallback mode -- so we don't depend
        # on the caller having activated ironenv.
        env["PYTHONPATH"] = (
            str(MLIR_AIE_ROOT / "install" / "python")
            + os.pathsep + env.get("PYTHONPATH", "")
        ).rstrip(os.pathsep)
        cmd = [sys.executable, str(PARSE_TOOL), "--server"]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_fh,
            env=env,
            text=True,
            bufsize=1,
        )
        ready_line = self.proc.stdout.readline().strip()
        if not ready_line:
            self.close()
            raise RuntimeError(f"{side} parser died before ready "
                               f"(see {stderr_log})")
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as e:
            self.close()
            raise RuntimeError(
                f"{side} parser first line not JSON: {ready_line!r}") from e
        if ready.get("event") != "ready":
            self.close()
            raise RuntimeError(
                f"{side} parser first line not 'ready': {ready_line!r}")

    def parse_one(
        self,
        trace_bin: Path,
        xclbin_mlir: Path,
        out_events: Optional[Path] = None,
        out_cycles: Optional[Path] = None,
        out_perfetto: Optional[Path] = None,
        out_commands: Optional[Path] = None,
        trace_mode: str = "event_time",
    ) -> dict:
        """Send one decode request, return parsed response dict.

        ``trace_mode`` selects the per-tile decoder used by parse-trace's
        ``--decoder=ours`` backend (server default). Supported values:
        ``"event_time"`` (mode 0), ``"event_pc"`` (mode 1), ``"inst_exec"``
        (mode 2). Defaulting to ``"event_time"`` preserves the existing
        sweep_multi behavior so older callers don't have to thread it.

        The response shape is what parse-trace.py's server_loop emits:
            {"ok": True, "events_count": N, "cycles": <span or None>,
             "empty": <bool>, "elapsed_ms": M}
          or {"ok": False, "error": "..."}.
        """
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError(f"{self.side} parser has exited")
        req = {"trace_bin": str(trace_bin), "xclbin_mlir": str(xclbin_mlir)}
        if out_events:   req["out_events"]   = str(out_events)
        if out_cycles:   req["out_cycles"]   = str(out_cycles)
        if out_perfetto: req["out_perfetto"] = str(out_perfetto)
        if out_commands: req["out_commands"] = str(out_commands)
        # Always include trace_mode -- the server defaults to "event_time"
        # on absence, but being explicit keeps the request self-describing
        # and makes mode-2 baseline batches debuggable from the request log.
        req["trace_mode"] = trace_mode
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, IOError) as e:
            raise RuntimeError(f"{self.side} parser write failed: {e}") from e
        resp = self.proc.stdout.readline().strip()
        if not resp:
            raise RuntimeError(f"{self.side} parser closed before response")
        try:
            return json.loads(resp)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"{self.side} parser response not JSON: {resp!r}") from e

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None
        if self._stderr_fh is not None and not self._stderr_fh.closed:
            self._stderr_fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _parse_trace_bin(
    trace_bin: Path,
    mlir: Path,
    events_out: Path,
    cycles_out: Optional[Path],
    parse_log: Path,
    env_for_parse: Dict[str, str],
    parser_session: Optional[ParseSession] = None,
    trace_mode: str = "event_time",
) -> Tuple[bool, Optional[str], Optional[int], Optional[int]]:
    """Run parse-trace on a trace binary. Returns (ok, error,
    cycles, events_count). ok=True with cycles=0/events_count=0 means
    the trace parsed as empty (kernel ran, no events fired); that's
    not a sweep failure.

    When ``parser_session`` is provided, the decode is dispatched
    through the long-lived parse-trace --server subprocess, avoiding
    a fresh Python startup (~620 ms -> ~100 ms per call). When None,
    falls back to the old subprocess.run() path so this helper still
    works outside a session-managed sweep.

    ``trace_mode`` selects the decoder mode (event_time / event_pc /
    inst_exec). Mode-2 baseline batches must pass ``"inst_exec"`` here
    or the server silently decodes them as event_time and emits
    structurally wrong events.
    """
    if parser_session is not None:
        try:
            resp = parser_session.parse_one(
                trace_bin=trace_bin,
                xclbin_mlir=mlir,
                out_events=events_out,
                out_cycles=cycles_out,
                trace_mode=trace_mode,
            )
        except RuntimeError as e:
            return False, f"parse-trace session: {e}", None, None
        if not resp.get("ok"):
            err = resp.get("error", "unknown")
            # The server reports "empty" via flags rather than a
            # special error message, but errno-style failures still
            # arrive as ok=false. Treat empty-trace exactly the same
            # way the fallback path does.
            return False, f"parse-trace: {err}", None, None
        if resp.get("empty"):
            # Server still wrote whatever output files were requested
            # (with cycles=0); make doubly sure the events file is
            # well-formed for downstream tools that didn't pass
            # out_events to the server.
            if not events_out.exists():
                events_out.write_text(
                    '{"schema_version":1,"events":[],"slot_names":{}}\n')
            return True, None, 0, 0
        return True, None, int(resp.get("cycles") or 0), \
               int(resp.get("events_count") or 0)

    # Fallback path: spawn a fresh interpreter per call. Slow but kept
    # for callers that don't want to manage a ParseSession.
    parse_cmd = [
        sys.executable, str(PARSE_TOOL),
        "--trace-bin", str(trace_bin),
        "--xclbin-mlir", str(mlir),
        "--out-events", str(events_out),
        "--trace-mode", trace_mode,
    ]
    if cycles_out is not None:
        parse_cmd.extend(["--out-cycles", str(cycles_out)])
    parse_env = env_for_parse.copy()
    parse_env["PYTHONPATH"] = str(MLIR_AIE_ROOT / "install" / "python")
    with parse_log.open("w") as lf:
        rc = subprocess.run(parse_cmd, env=parse_env,
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        log_text = parse_log.read_text(errors="replace") if parse_log.exists() else ""
        empty_markers = ("no timestamped events", "empty or all zeros")
        if any(m in log_text for m in empty_markers):
            events_out.write_text('{"schema_version":1,"events":[],"slot_names":{}}\n')
            if cycles_out is not None:
                cycles_out.write_text("0\n")
            return True, None, 0, 0
        return False, f"parse-trace exit {rc}", None, None

    cycles = 0
    if cycles_out is not None:
        try:
            cycles = int(cycles_out.read_text().strip() or "0")
        except (ValueError, FileNotFoundError):
            cycles = 0
    events_count = 0
    if events_out.exists():
        try:
            events_count = len(json.loads(events_out.read_text()))
        except Exception:
            events_count = 0
    return True, None, cycles, events_count


def _run_one_side(
    side: str,                   # "HW" or "EMU"
    session: Optional["RunnerSession"],
    runner_env: Dict[str, str],  # used to build parse_env (not for the runner)
    instr: Path,
    trace_bin: Path,
    mlir: Path,
    events_out: Path,
    cycles_out: Path,
    parse_log: Path,
    ctrlpkt: Optional[Path],
    parser_session: Optional["ParseSession"] = None,
    trace_mode: str = "event_time",
) -> RunResult:
    """One (run -> parse) cycle.

    The runner session is shared across all batches of a sweep; this
    function just dispatches a single run_one() and then parses. The
    trace_bin is written by the runner and read back by parse-trace.

    ``trace_mode`` is forwarded to the parse-trace decoder so mode-1 /
    mode-2 traces decode through the right per-tile decoder. Defaults
    to ``"event_time"`` to preserve the legacy single/multi-tile sweep
    behavior.

    Failures are recorded in the RunResult, not raised -- a single
    batch failure must not kill the sweep.
    """
    if session is None:
        return RunResult(ok=False, cycles=None, events_count=None,
                         error=f"{side} session is not open")
    try:
        status = session.run_one(
            instr=instr,
            trace_out=trace_bin,
            ctrlpkts=[ctrlpkt] if (ctrlpkt and ctrlpkt.is_file()) else None,
            trace_size=1 << 20,
        )
    except RuntimeError as e:
        return RunResult(ok=False, cycles=None, events_count=None,
                         error=f"{side} runner session: {e}")
    if not status.get("ok", False):
        return RunResult(ok=False, cycles=None, events_count=None,
                         error=f"{side} runner: {status.get('error', 'unknown')}")

    env_for_parse = os.environ.copy()
    env_for_parse.update(runner_env)
    # Mode 2 (inst_exec) has no cycle scalar -- parse-trace.py rejects
    # --out-cycles for it. Suppress cycles_out so the parse session
    # doesn't fail before writing events_out.
    cycles_arg = None if trace_mode == "inst_exec" else cycles_out
    ok, err, cycles, events_count = _parse_trace_bin(
        trace_bin, mlir, events_out, cycles_arg, parse_log, env_for_parse,
        parser_session=parser_session,
        trace_mode=trace_mode,
    )
    if not ok:
        return RunResult(ok=False, cycles=None, events_count=None,
                         error=f"{side} {err}")
    return RunResult(ok=True, cycles=cycles, events_count=events_count)


# ---------------------------------------------------------------------------
# insts.bin inspection helpers
# ---------------------------------------------------------------------------

_INSTS_HEADER_BYTES = 16
_INSTS_OPCODE_WRITE32   = 0x00
_INSTS_OPCODE_BLOCKWRITE = 0x01
_INSTS_OPCODE_MASKWRITE = 0x03
_INSTS_OPCODE_DDR_PATCH = 0x81


def _discover_trace_buf_idx(insts: Path) -> Optional[int]:
    """Compute the trace BO's 0-indexed position among data buffers.

    Method: walk insts.bin's DdrPatch ops and take the max ``arg_idx``.
    DdrPatch ops fill in BO addresses for shim-DMA BDs at runtime, with
    one patch per data buffer the kernel uses. Trace is added last by
    every build flow we support (mlir-aie's --with-hw-cycles puts it at
    arg_idx=N+1 where N is the user-buffer count; the trace-inject flow
    in xdna-emu's traced-tests adds it as the next runtime_sequence
    arg). In both cases the trace BO ends up with the highest arg_idx
    among DdrPatch targets, so taking the max is robust.

    Returns None if the file isn't a recognisable insts.bin or has no
    DdrPatch ops -- in that case the caller falls back to the runner's
    legacy "last buffer kernarg = trace" heuristic.

    DdrPatch layout (from src/npu/parser.rs lines 313-358):
      48-byte instruction; payload offsets 16/24/32 hold reg_addr,
      arg_idx (one byte at offset 24), and arg_plus respectively.
    """
    try:
        data = insts.read_bytes()
    except OSError:
        return None
    if len(data) < _INSTS_HEADER_BYTES:
        return None
    magic = int.from_bytes(data[0:4], "little")
    if magic != 0x06030100:
        return None
    total_size = int.from_bytes(data[12:16], "little")
    end = min(len(data), total_size)
    off = _INSTS_HEADER_BYTES
    max_arg_idx: Optional[int] = None
    while off + 4 <= end:
        opcode = data[off] & 0xFF
        if opcode == _INSTS_OPCODE_WRITE32:
            size = 24
        elif opcode == _INSTS_OPCODE_BLOCKWRITE:
            if off + 16 > end:
                break
            size = int.from_bytes(data[off + 12:off + 16], "little")
        elif opcode == _INSTS_OPCODE_MASKWRITE:
            size = 28
        elif opcode == _INSTS_OPCODE_DDR_PATCH:
            size = 48
            if off + 36 <= end:
                arg_idx = data[off + 8 + 24]   # payload[24] inside the op
                if max_arg_idx is None or arg_idx > max_arg_idx:
                    max_arg_idx = arg_idx
        else:
            # Unknown opcode -- stop walking; any answer we'd derive
            # past this point is unreliable.
            break
        off += size
    return max_arg_idx


def _find_cdo_preambles(mlir: Path) -> List[Path]:
    """Locate main_aie_cdo_init.bin + main_aie_cdo_enable.bin alongside
    the lowered MLIR. They live in the same .mlir.prj/ directory the
    aiecc.py compile produced. Returns [] if either is missing -- the
    caller treats that as "no preamble injection available."
    """
    prj = mlir.parent
    init = prj / "main_aie_cdo_init.bin"
    enable = prj / "main_aie_cdo_enable.bin"
    if init.is_file() and enable.is_file():
        return [init, enable]
    return []


def _find_post_lowering_mlir(build_dir: Path) -> Optional[Path]:
    """Locate the aiecc-lowered MLIR (input_with_addresses.mlir) inside
    build_dir/*.prj/. Same discovery shape as emu-bridge-test.sh so parse
    behaves identically.
    """
    for prj in build_dir.glob("*.mlir.prj"):
        cand = prj / "input_with_addresses.mlir"
        if cand.exists():
            return cand
    # Fallback: one-level-deep search for robustness against future naming.
    for cand in build_dir.glob("*/input_with_addresses.mlir"):
        return cand
    return None
