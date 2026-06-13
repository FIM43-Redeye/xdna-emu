#!/usr/bin/env python3
"""Run a calibration sweep on real NPU hardware.

For each (kind, target, count) configuration in the sweep, this script:
  1. Generates a calibration kernel via gen_kernel.py
  2. Compiles it via aiecc.py (Peano)
  3. Runs it on the real NPU via bridge-trace-runner
  4. Parses the trace via parse-trace.py
  5. Records anchor cycle delta in the measurements JSON

Compile is parallelized; HW runs are serial (NPU is single-tenant);
trace parsing is parallel.

Usage:
  python3 run_sweep.py --out build/calib/sweep-001.json [--reps 3]
  python3 run_sweep.py --config calib_config.json --out sweep.json

Default sweep covers write32 -> compute, write32 -> mem, write32 -> shim
at N in {16, 64, 256, 1024, 4096} with 3 reps each (45 measurements).

Output JSON format:
  {
    "schema_version": 1,
    "device": "npu1",
    "compiler": "peano",
    "measurements": [
      {"kind": "write32", "target": "compute", "count": 1024,
       "payload": 0, "rep": 0, "hw_cycles": 116397, "anchor_a_ts": ...,
       "anchor_b_ts": ..., "core_ticks": 11981}
    ]
  }
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GEN_KERNEL = REPO_ROOT / "tools" / "calibration" / "gen_kernel.py"
PARSE_TRACE = REPO_ROOT / "tools" / "parse-trace.py"
TRACE_INJECT = REPO_ROOT / "tools" / "mlir-trace-inject.py"
BRIDGE_RUNNER = REPO_ROOT / "bridge-runner" / "build" / "bridge-trace-runner"

# Memtile lock/DMA events swept for the dma_passthrough kind, plus the core
# grounding tick. Same set the v2_core tenant-4 spike proved decodable on NPU1
# (the memtile trace arms via the North broadcast pulled up by the core).
PASSTHROUGH_MEMTILE_SWEEP = (
    "LOCK_SEL0_ACQ_GE,LOCK_SEL0_REL,LOCK_SEL1_ACQ_GE,LOCK_SEL1_REL,"
    "DMA_S2MM_SEL0_START_TASK,DMA_MM2S_SEL0_START_TASK"
)
PASSTHROUGH_CORE_GROUNDING = "PERF_CNT_2,INSTR_EVENT_0,INSTR_EVENT_1"
# Depth-2 objectfifo: the first ~2 buffer hand-offs run at full speed before the
# pipeline backs up to steady state. Drop this many leading release-deltas as
# warmup before taking the steady-state median.
PASSTHROUGH_WARMUP_DELTAS = 2
# Match the proven v2_core trace BO size (steady cadence is tiny vs this).
PASSTHROUGH_TRACE_SIZE = 1048576

DEFAULT_SWEEP = [
    {"kind": "write32", "target": "compute", "counts": [16, 64, 256, 1024, 4096]},
    {"kind": "write32", "target": "mem",     "counts": [16, 64, 256, 1024, 4096]},
    {"kind": "write32", "target": "shim",    "counts": [16, 64, 256, 1024, 4096]},
]

# Buffer-size sweep for DDR-delivery burst characterization. The swept axis is
# the per-iteration objectfifo buffer size; the iteration count is fixed high
# enough to leave several steady-state samples after warmup. A flat steady
# period across sizes => fixed per-buffer-exchange overhead; a period that
# scales with size => per-word transfer rate; a staircase => true bursting.
PASSTHROUGH_SWEEP = [
    {"kind": "dma_passthrough", "device": "npu1_1col",
     "count": 12, "buffer_words_list": [32, 64, 128, 256]},
]


def run(cmd: list, cwd: Path = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def gen_one(work_dir: Path, task: dict) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)

    # DMA passthrough: distinct generator path (buffer-size param, no
    # target/anchor/payload). Trace is injected separately in inject_one().
    if task["kind"] == "dma_passthrough":
        cmd = [
            sys.executable, str(GEN_KERNEL),
            "--kind", "dma_passthrough",
            "--count", str(task["count"]),
            "--buffer-words", str(task["buffer_words"]),
            "--out", str(work_dir),
        ]
        if "device" in task:
            cmd += ["--device", task["device"]]
        run(cmd)
        return

    cmd = [
        sys.executable, str(GEN_KERNEL),
        "--kind", task["kind"],
        "--count", str(task["count"]),
        "--payload", str(task["payload"]),
        "--out", str(work_dir),
    ]
    # Either explicit target_col/target_row or the legacy --target shorthand.
    if "target_col" in task or "target_row" in task:
        cmd += ["--target-col", str(task.get("target_col", 0))]
        cmd += ["--target-row", str(task.get("target_row", 2))]
    elif "target" in task:
        cmd += ["--target", task["target"]]
    if "anchor_col" in task:
        cmd += ["--anchor-col", str(task["anchor_col"])]
    if "anchor_row" in task:
        cmd += ["--anchor-row", str(task["anchor_row"])]
    if "device" in task:
        cmd += ["--device", task["device"]]
    if "ticker_period" in task:
        cmd += ["--ticker-period", str(task["ticker_period"])]
    run(cmd)


def inject_one(work_dir: Path, task: dict) -> None:
    """Author memtile LOCK_SEL trace into a dma_passthrough kernel.

    Runs tools/mlir-trace-inject.py (the proven v2_core memtile-sweep path):
    aie.mlir -> aie_traced.mlir, plus trace_config.json. The compile step then
    builds aie_traced.mlir.
    """
    log_path = work_dir / "inject.log"
    with open(log_path, "w") as logf:
        subprocess.run([
            sys.executable, str(TRACE_INJECT),
            "--input", "aie.mlir",
            "--out", "aie_traced.mlir",
            "--trace-mode", "event_time",
            "--core-grounding", PASSTHROUGH_CORE_GROUNDING,
            "--memtile-sweep-events", PASSTHROUGH_MEMTILE_SWEEP,
            "--trace-config-out", "trace_config.json",
            "--config-test-name", task_tag(task),
        ], cwd=work_dir, stdout=logf, stderr=subprocess.STDOUT, check=True)


def compile_one(work_dir: Path, src_mlir: str = "aie.mlir",
                make_dummies: bool = True) -> bool:
    """Compile `src_mlir` in `work_dir`. Returns True on success."""
    if make_dummies:
        # Create dummy input buffers (4 unused kernargs) for control-packet
        # kernels. The dma_passthrough kernel lets bridge-trace-runner
        # auto-allocate its data BOs, so it skips this.
        for i in range(4):
            (work_dir / f"dummy{i}.bin").write_bytes(b"\0" * 64)

    log_path = work_dir / "compile.log"
    with open(log_path, "w") as logf:
        proc = subprocess.run([
            "nice", "-n", "19",
            "aiecc.py",
            "--no-aiesim",
            "--no-xchesscc", "--no-xbridge",
            "--aie-generate-xclbin",
            "--aie-generate-npu-insts",
            "--no-compile-host",
            "--xclbin-name=final.xclbin",
            "--npu-insts-name=insts.bin",
            src_mlir,
        ], cwd=work_dir, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode == 0 and (work_dir / "final.xclbin").exists()


def run_on_hw(work_dir: Path, task: dict = None, trace_size: int = 262144) -> bool:
    """Run the kernel on real NPU. Returns True on success."""
    log_path = work_dir / "run.log"
    if task and task["kind"] == "dma_passthrough":
        # bridge-trace-runner auto-allocates the in/out data BOs and identifies
        # the trace BO as the last kernarg; no --kernel/--input needed (proven
        # by the v2_core baseline run).
        cmd = [
            str(BRIDGE_RUNNER),
            "--xclbin", "final.xclbin",
            "--instr", "insts.bin",
            "--trace-out", "trace.bin",
            "--trace-size", str(PASSTHROUGH_TRACE_SIZE),
        ]
    else:
        cmd = [
            str(BRIDGE_RUNNER),
            "--xclbin", "final.xclbin",
            "--instr", "insts.bin",
            "--kernel", "MLIR_AIE",
            "--input", "dummy0.bin", "--input", "dummy1.bin",
            "--input", "dummy2.bin", "--input", "dummy3.bin",
            "--trace-out", "trace.bin",
            "--trace-size", str(trace_size),
        ]
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=work_dir, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode == 0 and (work_dir / "trace.bin").exists()


def extract_passthrough_period(events: list) -> dict:
    """Steady-state memtile buffer-handoff period from LOCK_SEL0_REL cadence.

    One SEL0_REL fires per objectfifo buffer hand-off on the memtile (row 1).
    The inter-release deltas, after dropping the depth-2 double-buffer warmup,
    give the steady-state per-buffer delivery cost -- the burst-model signal.
    `hw_cycles` is set to the steady median so the sweep summary counts it.
    """
    rels = sorted(e["ts"] for e in events
                  if e.get("row") == 1 and e.get("name") == "LOCK_SEL0_REL")
    if len(rels) < PASSTHROUGH_WARMUP_DELTAS + 3:
        return {"hw_cycles": None,
                "error": f"too few SEL0_REL ({len(rels)}) for a steady median"}
    deltas = [b - a for a, b in zip(rels, rels[1:])]
    steady = deltas[PASSTHROUGH_WARMUP_DELTAS:]
    steady_sorted = sorted(steady)
    median = steady_sorted[len(steady_sorted) // 2]
    return {
        "hw_cycles": median,
        "period_median": median,
        "period_min": min(steady),
        "period_max": max(steady),
        "n_releases": len(rels),
        "all_deltas": deltas,
        "steady_deltas": steady,
        "first_rel_ts": rels[0],
        "last_rel_ts": rels[-1],
    }


def parse_one(work_dir: Path, task: dict = None) -> dict:
    """Parse the trace into events.json and extract the measurement.

    For control-packet kinds: the anchor A->B cycle delta. For dma_passthrough:
    the steady-state memtile buffer-handoff period.
    """
    is_passthrough = task and task["kind"] == "dma_passthrough"
    # The passthrough builds aie_traced.mlir, so its lowered MLIR lives in a
    # differently-named project dir.
    xclbin_mlir = ("aie_traced.mlir.prj/input_with_addresses.mlir"
                   if is_passthrough else "aie.mlir.prj/input_with_addresses.mlir")
    log_path = work_dir / "parse.log"
    with open(log_path, "w") as logf:
        proc = subprocess.run([
            sys.executable, str(PARSE_TRACE),
            "--trace-bin", "trace.bin",
            "--xclbin-mlir", xclbin_mlir,
            "--trace-mode", "event_time",
            "--decoder", "ours",
            "--out-events", "events.json",
        ], cwd=work_dir, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        return {"hw_cycles": None, "error": "parse-trace failed"}

    try:
        events = json.loads((work_dir / "events.json").read_text())["events"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        return {"hw_cycles": None, "error": f"events.json read: {e}"}

    if is_passthrough:
        return extract_passthrough_period(events)

    # Slot indices match positions in the aie.trace.event<...> declarations
    # in gen_kernel.py. For compute anchor we declare three events
    # (INSTR_EVENT_0, USER_EVENT_0, USER_EVENT_1) so anchors are at slots 1/2.
    # For shim/memtile anchor the INSTR_EVENT_0 slot is dropped, so anchors
    # are at slots 0/1.
    try:
        params = json.loads((work_dir / "params.json").read_text())
        anchor_row = params.get("anchor_row", 2)
    except (FileNotFoundError, json.JSONDecodeError):
        anchor_row = 2
    anchor_is_compute = anchor_row >= 2
    if anchor_is_compute:
        slot_a, slot_b, slot_tick = 1, 2, 0
    else:
        slot_a, slot_b, slot_tick = 0, 1, None

    anchors_a = [e for e in events if e["slot"] == slot_a]
    anchors_b = [e for e in events if e["slot"] == slot_b]
    ticks = [e for e in events if slot_tick is not None and e["slot"] == slot_tick]

    if not anchors_a or not anchors_b:
        return {
            "hw_cycles": None,
            "error": f"missing anchors: a={len(anchors_a)} b={len(anchors_b)}",
        }
    return {
        "hw_cycles": anchors_b[0]["ts"] - anchors_a[0]["ts"],
        "anchor_a_ts": anchors_a[0]["ts"],
        "anchor_b_ts": anchors_b[0]["ts"],
        "core_ticks": len(ticks),
    }


def expand_sweep(sweep_groups: list, reps: int) -> list:
    """Expand sweep groups into individual (kind, target, count, rep) tasks.

    Each group may specify any of: kind, target (shorthand), target_col,
    target_row, anchor_col, anchor_row, device, payload, counts.
    """
    tasks = []
    for grp in sweep_groups:
        # dma_passthrough sweeps buffer_words at a fixed iteration count.
        if grp.get("kind") == "dma_passthrough":
            base = {k: v for k, v in grp.items() if k != "buffer_words_list"}
            base.setdefault("payload", 0)
            for bw in grp["buffer_words_list"]:
                for rep in range(reps):
                    t = dict(base)
                    t["buffer_words"] = bw
                    t["rep"] = rep
                    tasks.append(t)
            continue
        base = {k: v for k, v in grp.items() if k != "counts"}
        base.setdefault("payload", 0)
        for count in grp["counts"]:
            for rep in range(reps):
                t = dict(base)
                t["count"] = count
                t["rep"] = rep
                tasks.append(t)
    return tasks


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True,
                   help="Output measurements JSON path")
    p.add_argument("--workdir", type=Path,
                   default=REPO_ROOT / "build" / "calibration" / "sweep",
                   help="Per-task working directory root")
    p.add_argument("--reps", type=int, default=3,
                   help="Repetitions per (kind, target, count)")
    p.add_argument("--compile-jobs", type=int, default=4,
                   help="Parallel compile workers")
    p.add_argument("--config", type=Path,
                   help="Optional sweep config JSON; overrides default sweep")
    p.add_argument("--passthrough", action="store_true",
                   help="Run the DMA-passthrough buffer-size sweep (DDR-delivery "
                        "burst characterization) instead of the default "
                        "control-packet sweep.")
    args = p.parse_args()

    sweep_groups = DEFAULT_SWEEP
    if args.passthrough:
        sweep_groups = PASSTHROUGH_SWEEP
    if args.config:
        sweep_groups = json.loads(args.config.read_text())

    tasks = expand_sweep(sweep_groups, args.reps)
    n_configs = sum(len(g.get("counts", g.get("buffer_words_list", [])))
                    for g in sweep_groups)
    print(f"Sweep: {len(tasks)} tasks across {n_configs} configurations.",
          file=sys.stderr)

    args.workdir.mkdir(parents=True, exist_ok=True)

    # Phase 1: generate + compile in parallel.
    print("[phase 1/3] generating + compiling kernels...", file=sys.stderr)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.compile_jobs) as pool:
        futs = {}
        for t in tasks:
            # rep=0 generates+compiles; reps 1..N reuse rep=0's xclbin.
            if t["rep"] != 0:
                continue
            tag = task_tag(t)
            wd = args.workdir / tag
            t["work_dir"] = wd
            futs[pool.submit(_gen_compile, wd, t)] = tag
        compile_failures = []
        for fut in as_completed(futs):
            tag = futs[fut]
            ok = fut.result()
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {tag}", file=sys.stderr)
            if not ok:
                compile_failures.append(tag)
    print(f"[phase 1/3] done in {time.time()-t0:.1f}s, "
          f"{len(compile_failures)} compile failures", file=sys.stderr)

    # Wire up rep > 0 tasks to share rep=0's work_dir.
    rep0_dirs = {task_tag(t): t["work_dir"] for t in tasks if t["rep"] == 0}
    for t in tasks:
        if t["rep"] != 0:
            t["work_dir"] = rep0_dirs[task_tag(t)]

    # Phase 2: HW runs (SERIAL -- NPU is single tenant).
    print("[phase 2/3] running on NPU (serial)...", file=sys.stderr)
    t0 = time.time()
    for i, t in enumerate(tasks):
        wd = t["work_dir"]
        if not (wd / "final.xclbin").exists():
            t["hw_status"] = "compile_skipped"
            continue
        # Each rep gets its own trace.bin; bridge-runner overwrites by default
        # so we run, then immediately parse, then move trace aside.
        ok = run_on_hw(wd, t)
        t["hw_status"] = "ok" if ok else "fail"
        # Snapshot trace + events files per rep.
        if ok:
            (wd / f"trace_rep{t['rep']}.bin").write_bytes((wd / "trace.bin").read_bytes())
        if (i + 1) % 10 == 0 or i + 1 == len(tasks):
            print(f"  [{i+1}/{len(tasks)}] elapsed {time.time()-t0:.1f}s",
                  file=sys.stderr)
    print(f"[phase 2/3] done in {time.time()-t0:.1f}s", file=sys.stderr)

    # Phase 3: parse traces in parallel (parse-trace.py reads trace.bin in
    # the work dir, so we need to swap each rep's trace.bin in turn -- do
    # this serially per work-dir but parallel across work-dirs).
    print("[phase 3/3] parsing traces...", file=sys.stderr)
    t0 = time.time()
    measurements = []
    measurement_keys = (
        "kind", "target", "target_col", "target_row",
        "anchor_col", "anchor_row", "device", "ticker_period",
        "count", "payload", "buffer_words", "rep",
    )
    for t in tasks:
        wd = t["work_dir"]
        record = {k: t[k] for k in measurement_keys if k in t}
        if t.get("hw_status") != "ok":
            record["hw_cycles"] = None
            record["error"] = t.get("hw_status", "unknown")
            measurements.append(record)
            continue
        # Swap in this rep's trace and parse.
        rep_trace = wd / f"trace_rep{t['rep']}.bin"
        if rep_trace.exists():
            (wd / "trace.bin").write_bytes(rep_trace.read_bytes())
        result = parse_one(wd, t)
        record.update(result)
        measurements.append(record)
    print(f"[phase 3/3] done in {time.time()-t0:.1f}s", file=sys.stderr)

    # Write the measurements JSON.
    out_doc = {
        "schema_version": 1,
        "device": "npu1",
        "compiler": "peano",
        "sweep_groups": sweep_groups,
        "reps": args.reps,
        "measurements": measurements,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2) + "\n")

    n_ok = sum(1 for m in measurements if m.get("hw_cycles") is not None)
    print(f"\nWrote {args.out} ({n_ok}/{len(measurements)} successful measurements)",
          file=sys.stderr)
    return 0 if n_ok > 0 else 1


def task_tag(task: dict) -> str:
    """Stable identifier per (kind, target, anchor, device, count, payload).

    Two tasks with the same tag share a build directory (so reps reuse the
    rep=0 xclbin). The tag must distinguish every dimension that affects
    the generated kernel.
    """
    if task["kind"] == "dma_passthrough":
        device_part = ""
        if task.get("device") and task["device"] != "npu1_1col":
            device_part = f"_{task['device']}"
        return f"dma_passthrough_w{task['buffer_words']}_n{task['count']}{device_part}"
    if "target_col" in task or "target_row" in task:
        target_part = f"c{task.get('target_col', 0)}r{task.get('target_row', 2)}"
    else:
        target_part = task.get("target", "compute")
    anchor_part = ""
    if "anchor_col" in task or "anchor_row" in task:
        anchor_part = f"_a{task.get('anchor_col', 0)}r{task.get('anchor_row', 2)}"
    device_part = ""
    if task.get("device") and task["device"] != "npu1_1col":
        device_part = f"_{task['device']}"
    ticker_part = ""
    if "ticker_period" in task and task["ticker_period"] != 256:
        ticker_part = f"_t{task['ticker_period']}"
    return (f"{task['kind']}_{target_part}{anchor_part}"
            f"_n{task['count']}_p{task['payload']}{device_part}{ticker_part}")


def _gen_compile(work_dir: Path, task: dict) -> bool:
    try:
        gen_one(work_dir, task)
        if task["kind"] == "dma_passthrough":
            inject_one(work_dir, task)
            return compile_one(work_dir, src_mlir="aie_traced.mlir",
                               make_dummies=False)
        return compile_one(work_dir)
    except subprocess.CalledProcessError:
        return False


if __name__ == "__main__":
    sys.exit(main())
