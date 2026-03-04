#!/usr/bin/env python3
"""Multi-run trace sweep with TRUE metronome alignment.

Pins slot 0 to TRUE (fires every cycle) and rotates the other 7 slots
through all events of interest across multiple runs.  After all runs
complete, invokes trace-merge.py to stitch them into a single unified
trace using TRUE timestamps as the alignment anchor.

Usage:
    trace-sweep.py <test-source-dir> --output <dir> [--no-hw] [--no-emu]
    trace-sweep.py add_one_using_dma -o /tmp/sweep-results

The test name is resolved as a path or substring of mlir-aie/test/npu-xrt/.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Event sets: the "interesting" events for each tile type, excluding TRUE
# (which is always in slot 0) and internal/reserved events.
# ---------------------------------------------------------------------------

# Core module events worth sweeping.  PORT_* events need PortEvent wrappers
# in the mlir-aie API, so we handle them separately.
CORE_EVENTS = [
    # Stalls
    "MEMORY_STALL",       # 23
    "STREAM_STALL",       # 24
    "CASCADE_STALL",      # 25
    "LOCK_STALL",         # 26
    # Core state
    "ACTIVE",             # 28
    "DISABLED",           # 29
    # Program flow
    "INSTR_EVENT_0",      # 33
    "INSTR_EVENT_1",      # 34
    "INSTR_CALL",         # 35
    "INSTR_RETURN",       # 36
    "INSTR_VECTOR",       # 37
    "INSTR_LOAD",         # 38
    "INSTR_STORE",        # 39
    "INSTR_STREAM_GET",   # 40
    "INSTR_STREAM_PUT",   # 41
    "INSTR_CASCADE_GET",  # 42
    "INSTR_CASCADE_PUT",  # 43
    "INSTR_LOCK_ACQUIRE_REQ",  # 44
    "INSTR_LOCK_RELEASE_REQ",  # 45
]

# Memory module events (compute tile).
MEM_EVENTS = [
    # DMA activity
    "DMA_S2MM_0_START_TASK",
    "DMA_S2MM_1_START_TASK",
    "DMA_MM2S_0_START_TASK",
    "DMA_MM2S_1_START_TASK",
    "DMA_S2MM_0_FINISHED_BD",
    "DMA_S2MM_1_FINISHED_BD",
    "DMA_MM2S_0_FINISHED_BD",
    "DMA_MM2S_1_FINISHED_BD",
    "DMA_S2MM_0_FINISHED_TASK",
    "DMA_S2MM_1_FINISHED_TASK",
    "DMA_MM2S_0_FINISHED_TASK",
    "DMA_MM2S_1_FINISHED_TASK",
    # DMA stalls
    "DMA_S2MM_0_STALLED_LOCK",
    "DMA_S2MM_1_STALLED_LOCK",
    "DMA_MM2S_0_STALLED_LOCK",
    "DMA_MM2S_1_STALLED_LOCK",
    "DMA_S2MM_0_STREAM_STARVATION",
    "DMA_S2MM_1_STREAM_STARVATION",
    "DMA_MM2S_0_STREAM_BACKPRESSURE",
    "DMA_MM2S_1_STREAM_BACKPRESSURE",
    # Bank conflicts
    "CONFLICT_DM_BANK_0",
    "CONFLICT_DM_BANK_1",
    "CONFLICT_DM_BANK_2",
    "CONFLICT_DM_BANK_3",
    # Edge/combo
    "EDGE_DETECTION_EVENT_0",
    "EDGE_DETECTION_EVENT_1",
    # Locks
    "LOCK_SEL0_ACQ_GE",
    "LOCK_0_REL",
]


def batch_events(events: list[str], batch_size: int = 7) -> list[list[str]]:
    """Split events into batches of batch_size, each prefixed with TRUE."""
    batches = []
    for i in range(0, len(events), batch_size):
        chunk = events[i:i + batch_size]
        # Pad with NONE if fewer than batch_size
        while len(chunk) < batch_size:
            chunk.append("NONE")
        batches.append(["TRUE"] + chunk)
    return batches


def resolve_test_dir(name_or_path: str) -> Path:
    """Resolve a test name or path to a test source directory."""
    p = Path(name_or_path)
    if p.is_dir():
        return p.resolve()

    # Search in mlir-aie/test/npu-xrt/
    script_dir = Path(__file__).parent
    test_base = script_dir.parent.parent / "mlir-aie" / "test" / "npu-xrt"
    if not test_base.is_dir():
        print(f"Error: test base not found: {test_base}", file=sys.stderr)
        sys.exit(1)

    matches = [d for d in test_base.iterdir() if d.is_dir() and name_or_path in d.name]
    if len(matches) == 0:
        print(f"Error: no test matching '{name_or_path}' in {test_base}", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        print(f"Error: ambiguous '{name_or_path}': {names}", file=sys.stderr)
        sys.exit(1)
    return matches[0]


def compile_base_trace(
    test_dir: Path,
    output_dir: Path,
    trace_size: int,
) -> tuple[Path | None, Path | None]:
    """Inject tracing with default events and compile ONCE.

    Returns (traced_dir, manifest_path) on success, (None, None) on failure.
    All batches will reuse this compiled xclbin + patched insts.bin.
    """
    script_dir = Path(__file__).parent
    traced_dir = output_dir / "base"
    traced_dir.mkdir(parents=True, exist_ok=True)

    # Inject with default events (no --events-json = defaults)
    print("  Base: injecting trace routing...")
    result = subprocess.run(
        [
            sys.executable, str(script_dir / "trace-inject.py"),
            str(test_dir),
            "--output", str(traced_dir),
            "--trace-size", str(trace_size),
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"    Inject failed: {result.stderr}", file=sys.stderr)
        return None, None

    manifest_path = traced_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("skipped"):
        print(f"    Skipped: {manifest.get('reason', 'unknown')}")
        return None, None

    # Compile once
    print("  Base: compiling traced xclbin (one-time)...")
    compile_result = subprocess.run(
        [
            "aiecc.py",
            "--no-aiesim", "--aie-generate-xclbin", "--aie-generate-npu-insts",
            "--no-compile-host", "--alloc-scheme=basic-sequential",
            "--no-xchesscc",
            "--xclbin-name=aie.xclbin", "--npu-insts-name=insts.bin",
            "./aie_traced.mlir",
        ],
        capture_output=True, text=True, timeout=600,
        cwd=str(traced_dir),
    )
    if compile_result.returncode != 0:
        (output_dir / "compile.log").write_text(
            compile_result.stdout + compile_result.stderr
        )
        print("    Compile failed (see compile.log)", file=sys.stderr)
        return None, None

    print("  Base: compile OK")
    return traced_dir, manifest_path


def prepare_batch(
    batch_idx: int,
    core_batch: list[str],
    mem_batch: list[str],
    base_dir: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict:
    """Prepare one batch directory: copy+patch insts.bin, write manifest.

    Returns a batch info dict with paths. Does NOT run anything -- the caller
    decides whether to run HW (serial) and/or EMU (parallel) separately.
    """
    import shutil

    script_dir = Path(__file__).parent
    batch_dir = output_dir / f"batch_{batch_idx:02d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Write events config JSON
    events_config = {
        "core_events": core_batch,
        "mem_events": mem_batch,
    }
    events_json = batch_dir / "events.json"
    events_json.write_text(json.dumps(events_config, indent=2) + "\n")

    # Set up batch traced directory: symlink xclbin, copy+patch insts.bin
    traced_dir = batch_dir / "traced"
    traced_dir.mkdir(parents=True, exist_ok=True)

    base_xclbin = base_dir / "aie.xclbin"
    base_insts = base_dir / "insts.bin"
    batch_xclbin = traced_dir / "aie.xclbin"
    batch_insts = traced_dir / "insts.bin"

    if not base_xclbin.exists() or not base_insts.exists():
        return {"batch": batch_idx, "status": "missing_base"}

    # Symlink xclbin (identical across batches), copy insts for patching
    if not batch_xclbin.exists():
        batch_xclbin.symlink_to(base_xclbin)
    shutil.copy2(base_insts, batch_insts)

    # Copy manifest (paths stay relative -- xclbin and insts.bin are in traced_dir)
    base_manifest = json.loads(manifest_path.read_text())
    base_manifest["insts"] = "insts.bin"
    base_manifest["xclbin"] = "aie.xclbin"
    batch_manifest = traced_dir / "manifest.json"
    batch_manifest.write_text(json.dumps(base_manifest, indent=2) + "\n")

    # Patch event registers in insts.bin
    print(f"  Batch {batch_idx}: patching events "
          f"(core={core_batch[:3]}..., mem={mem_batch[:3]}...)")
    patch_result = subprocess.run(
        [
            sys.executable, str(script_dir / "trace-patch-events.py"),
            str(batch_insts),
            "--manifest", str(batch_manifest),
            "--events-json", str(events_json),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if patch_result.returncode != 0:
        print(f"    Patch failed: {patch_result.stderr}", file=sys.stderr)
        print(f"    (will run with base events instead)")

    return {
        "batch": batch_idx,
        "status": "ok",
        "events": events_config,
        "manifest": str(batch_manifest),
        "batch_dir": str(batch_dir),
    }


def trim_trace_dir(path: Path):
    """Trim all trace_raw.bin files under a directory."""
    script_dir = Path(__file__).parent
    trim_script = script_dir / "trace-trim.py"
    if trim_script.exists():
        subprocess.run(
            [sys.executable, str(trim_script), "--dir", str(path)],
            capture_output=True, text=True, timeout=30,
        )


def run_batch_hw(batch_info: dict, hw_cooldown: float = 2.0) -> dict:
    """Run one batch on hardware (serial, with cooldown)."""
    import time

    script_dir = Path(__file__).parent
    trace_run = script_dir / "trace-run.py"
    batch_idx = batch_info["batch"]
    batch_dir = Path(batch_info["batch_dir"])
    manifest = batch_info["manifest"]

    hw_dir = batch_dir / "hw"
    print(f"  Batch {batch_idx}: running on hardware...")
    hw_result = subprocess.run(
        [sys.executable, str(trace_run), str(manifest), "-o", str(hw_dir)],
        capture_output=True, text=True, timeout=300,
    )
    if hw_result.returncode != 0:
        batch_info["hw_status"] = "failed"
        (batch_dir / "hw-run.log").write_text(hw_result.stdout + hw_result.stderr)
    else:
        batch_info["hw_status"] = "ok"
        batch_info["hw_trace"] = str(hw_dir / "trace.json")
        trim_trace_dir(hw_dir)

    # Cooldown: let the driver settle before the next context creation
    if hw_cooldown > 0:
        time.sleep(hw_cooldown)

    return batch_info


def run_batch_emu(batch_info: dict) -> dict:
    """Run one batch on the emulator (safe to parallelize)."""
    script_dir = Path(__file__).parent
    trace_run = script_dir / "trace-run.py"
    batch_idx = batch_info["batch"]
    batch_dir = Path(batch_info["batch_dir"])
    manifest = batch_info["manifest"]

    emu_dir = batch_dir / "emu"
    print(f"  Batch {batch_idx}: running on emulator...")
    env = os.environ.copy()
    env["XDNA_EMU"] = "1"
    emu_result = subprocess.run(
        [sys.executable, str(trace_run), str(manifest), "-o", str(emu_dir)],
        capture_output=True, text=True, timeout=300,
        env=env,
    )
    if emu_result.returncode != 0:
        batch_info["emu_status"] = "failed"
        (batch_dir / "emu-run.log").write_text(emu_result.stdout + emu_result.stderr)
    else:
        batch_info["emu_status"] = "ok"
        batch_info["emu_trace"] = str(emu_dir / "trace.json")
        trim_trace_dir(emu_dir)

    return batch_info


def main():
    parser = argparse.ArgumentParser(
        description="Multi-run trace sweep with TRUE metronome alignment",
    )
    parser.add_argument(
        "test",
        help="Test name (substring match) or path to test source directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for sweep results",
    )
    parser.add_argument("--no-hw", action="store_true", help="Skip hardware runs")
    parser.add_argument("--no-emu", action="store_true", help="Skip emulator runs")
    parser.add_argument(
        "--trace-size", type=int, default=1048576,
        help="Trace buffer size per batch (default: 1MB)",
    )
    parser.add_argument(
        "--hw-cooldown", type=float, default=2.0,
        help="Seconds between hardware runs to let the driver settle (default: 2)",
    )
    parser.add_argument(
        "--emu-jobs", type=int, default=0,
        help="Parallel emulator jobs (default: nproc)",
    )
    parser.add_argument(
        "--core-only", action="store_true",
        help="Only sweep core events (skip memory events)",
    )
    parser.add_argument(
        "--mem-only", action="store_true",
        help="Only sweep memory events (skip core events)",
    )
    args = parser.parse_args()

    test_dir = resolve_test_dir(args.test)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    test_name = test_dir.name
    print(f"Trace sweep: {test_name}")
    print(f"  Test dir:  {test_dir}")
    print(f"  Output:    {output_dir}")
    print(f"  HW:        {'yes' if not args.no_hw else 'no'}")
    print(f"  EMU:       {'yes' if not args.no_emu else 'no'}")

    # Build batches: core events and mem events independently.
    # Each batch has TRUE in slot 0 + 7 events in slots 1-7.
    # Core and mem events are separate trace units, so we can sweep them
    # in the same run (core_batch[i] paired with mem_batch[i]).
    core_batches = batch_events(CORE_EVENTS) if not args.mem_only else [["TRUE"] + ["NONE"] * 7]
    mem_batches = batch_events(MEM_EVENTS) if not args.core_only else [["TRUE"] + ["NONE"] * 7]

    # Pair core and mem batches. If counts differ, extend the shorter with NONEs.
    num_batches = max(len(core_batches), len(mem_batches))
    noop_batch = ["TRUE"] + ["NONE"] * 7
    while len(core_batches) < num_batches:
        core_batches.append(noop_batch)
    while len(mem_batches) < num_batches:
        mem_batches.append(noop_batch)

    print(f"  Batches:   {num_batches}")
    print()

    # Compile base trace once (all batches share xclbin, patch insts.bin)
    base_dir, base_manifest = compile_base_trace(
        test_dir, output_dir, args.trace_size,
    )
    if base_dir is None:
        print("Failed to compile base trace. Aborting sweep.", file=sys.stderr)
        sys.exit(1)
    print()

    # Prepare all batches (patch insts.bin, no execution yet)
    batches = []
    for i in range(num_batches):
        info = prepare_batch(
            i, core_batches[i], mem_batches[i],
            base_dir, base_manifest, output_dir,
        )
        batches.append(info)
    print()

    # Run HW (serial + cooldown) and EMU (parallel) simultaneously.
    # HW runs are I/O-bound (waiting on NPU), EMU runs are CPU-bound.
    # No conflicts: each writes to its own hw/ or emu/ subdirectory.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    runnable = [b for b in batches if b["status"] == "ok"]
    emu_jobs = args.emu_jobs if args.emu_jobs > 0 else os.cpu_count() or 4

    def hw_serial_runner():
        """Run all HW batches serially with cooldown."""
        for b in runnable:
            run_batch_hw(b, hw_cooldown=args.hw_cooldown)

    with ThreadPoolExecutor(max_workers=emu_jobs + 1) as pool:
        # One thread drives all serial HW runs
        hw_future = None
        if not args.no_hw:
            print(f"  HW:  {len(runnable)} batches (serial, {args.hw_cooldown}s cooldown)")
            hw_future = pool.submit(hw_serial_runner)

        # All EMU batches run in parallel across remaining threads
        emu_futures = []
        if not args.no_emu:
            print(f"  EMU: {len(runnable)} batches (parallel, -j{emu_jobs})")
            emu_futures = [pool.submit(run_batch_emu, b) for b in runnable]

        # Wait for everything
        for f in emu_futures:
            f.result()
        if hw_future:
            hw_future.result()

    print()
    results = batches

    # Write sweep manifest
    sweep_manifest = {
        "test_name": test_name,
        "num_batches": num_batches,
        "batches": results,
    }
    manifest_path = output_dir / "sweep-manifest.json"
    manifest_path.write_text(json.dumps(sweep_manifest, indent=2) + "\n")

    # Collect trace files for merging
    hw_traces = [r["hw_trace"] for r in results if r.get("hw_trace")]
    emu_traces = [r["emu_trace"] for r in results if r.get("emu_trace")]

    script_dir = Path(__file__).parent
    merge_script = script_dir / "trace-merge.py"

    if hw_traces and merge_script.exists():
        print(f"\nMerging {len(hw_traces)} hardware traces...")
        hw_merged = output_dir / "hw-merged.json"
        subprocess.run(
            [sys.executable, str(merge_script)] + hw_traces + ["-o", str(hw_merged)],
            timeout=120,
        )

    if emu_traces and merge_script.exists():
        print(f"\nMerging {len(emu_traces)} emulator traces...")
        emu_merged = output_dir / "emu-merged.json"
        subprocess.run(
            [sys.executable, str(merge_script)] + emu_traces + ["-o", str(emu_merged)],
            timeout=120,
        )

    # Cross-platform comparison (if both HW and EMU ran)
    compare_script = script_dir / "trace-compare.py"
    hw_merged = output_dir / "hw-merged.json"
    emu_merged = output_dir / "emu-merged.json"
    report_path = output_dir / "comparison-report.txt"

    if hw_merged.exists() and emu_merged.exists() and compare_script.exists():
        print(f"\nComparing HW vs EMU (boot-normalized)...")
        subprocess.run(
            [
                sys.executable, str(compare_script),
                str(hw_merged), str(emu_merged),
                "-o", str(report_path),
            ],
            timeout=120,
        )

    # Summary
    print(f"\nSweep complete: {test_name}")
    print(f"  Batches:  {num_batches}")
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"  Success:  {ok}/{num_batches}")
    print(f"  Manifest: {manifest_path}")
    if hw_merged.exists():
        print(f"  HW merged:  {hw_merged}")
    if emu_merged.exists():
        print(f"  EMU merged: {emu_merged}")
    if report_path.exists():
        print(f"  Report:     {report_path}")
    print(f"\nView in Perfetto: https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
