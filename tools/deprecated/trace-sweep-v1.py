#!/usr/bin/env python3
"""Multi-run trace sweep with TRUE metronome alignment.

Pins slot 0 to TRUE (fires every cycle) and rotates the other 7 slots
through all events of interest across multiple runs.  After all runs
complete, invokes trace-merge.py to stitch them into a single unified
trace using TRUE timestamps as the alignment anchor.

Operates on pre-compiled artifacts from the bridge script.  Does NOT
compile anything -- that is the bridge script's responsibility.

Usage:
    trace-sweep.py --build-dir <path> --test-exe <path> -o <dir>
    trace-sweep.py --build-dir build/test/npu-xrt/add_one/chess \
                   --test-exe build/test/npu-xrt/add_one/test.exe \
                   -o /tmp/sweep-results
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
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


# MemTile events worth sweeping.  Uses SEL0/SEL1 naming (MemTileEvent enum).
MEMTILE_EVENTS = [
    # DMA activity
    "DMA_S2MM_SEL0_START_TASK",
    "DMA_S2MM_SEL1_START_TASK",
    "DMA_MM2S_SEL0_START_TASK",
    "DMA_MM2S_SEL1_START_TASK",
    "DMA_S2MM_SEL0_FINISHED_BD",
    "DMA_S2MM_SEL1_FINISHED_BD",
    "DMA_MM2S_SEL0_FINISHED_BD",
    "DMA_MM2S_SEL1_FINISHED_BD",
    "DMA_S2MM_SEL0_FINISHED_TASK",
    "DMA_S2MM_SEL1_FINISHED_TASK",
    "DMA_MM2S_SEL0_FINISHED_TASK",
    "DMA_MM2S_SEL1_FINISHED_TASK",
    # DMA stalls
    "DMA_S2MM_SEL0_STALLED_LOCK",
    "DMA_S2MM_SEL1_STALLED_LOCK",
    "DMA_MM2S_SEL0_STALLED_LOCK",
    "DMA_MM2S_SEL1_STALLED_LOCK",
    "DMA_S2MM_SEL0_STREAM_STARVATION",
    "DMA_S2MM_SEL1_STREAM_STARVATION",
    "DMA_MM2S_SEL0_STREAM_BACKPRESSURE",
    "DMA_MM2S_SEL1_STREAM_BACKPRESSURE",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dmesg_count(pattern: str) -> int:
    """Count dmesg lines matching pattern."""
    try:
        result = subprocess.run(
            ["dmesg"], capture_output=True, text=True, timeout=5,
        )
        return sum(1 for line in result.stdout.splitlines() if pattern in line)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0


def wait_npu_idle(timeout: float = 10.0, poll_interval: float = 0.1) -> bool:
    """Poll xrt-smi until no hardware contexts are running."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["xrt-smi", "examine", "-r", "aie-partitions"],
                capture_output=True, text=True, timeout=5,
            )
            if "No hardware contexts running" in result.stdout:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        time.sleep(poll_interval)
    return False


def _write_skip_manifest(output_dir: Path, reason: str, detail: str = ""):
    """Write a sweep-manifest.json marking this test as skipped."""
    manifest = {
        "skipped": True,
        "reason": reason,
        "detail": detail[:500] if detail else "",
    }
    (output_dir / "sweep-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------

def prepare_batch(
    batch_idx: int,
    core_batch: list[str],
    mem_batch: list[str],
    base_insts: Path,
    manifest: dict,
    output_dir: Path,
    memtile_batch: list[str] | None = None,
) -> dict:
    """Prepare one batch: copy insts.bin and patch event registers.

    Args:
        batch_idx: Batch number (0-indexed).
        core_batch: 8-element list of core event names.
        mem_batch: 8-element list of mem event names.
        base_insts: Path to the base (unpatched) insts.bin.
        manifest: Trace manifest dict (needs "tiles_traced" for event targeting).
        output_dir: Root sweep output directory.
        memtile_batch: Optional 8-element memtile event list.

    Returns:
        Batch info dict with keys: batch, status, events, batch_dir, insts.
    """
    script_dir = Path(__file__).parent
    batch_dir = output_dir / f"batch_{batch_idx:02d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Write events.json for this batch.
    events_config = {
        "core_events": core_batch,
        "mem_events": mem_batch,
    }
    if memtile_batch is not None:
        events_config["memtile_events"] = memtile_batch
    events_path = batch_dir / "events.json"
    events_path.write_text(json.dumps(events_config, indent=2) + "\n")

    # Copy base insts.bin into batch dir for patching.
    batch_insts = batch_dir / "insts.bin"
    shutil.copy2(base_insts, batch_insts)

    # Write a minimal manifest for trace-patch-events.py.
    # It needs "tiles_traced" to know which tile addresses to patch.
    batch_manifest = batch_dir / "manifest.json"
    batch_manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    # Patch event registers in copied insts.bin.
    patch_result = subprocess.run(
        [
            sys.executable, str(script_dir / "trace-patch-events.py"),
            str(batch_insts),
            "--manifest", str(batch_manifest),
            "--events-json", str(events_path),
        ],
        capture_output=True, text=True, timeout=30,
    )

    if patch_result.returncode != 0:
        return {
            "batch": batch_idx,
            "status": "patch_failed",
            "error": patch_result.stderr.strip(),
            "events": events_config,
            "batch_dir": str(batch_dir),
        }

    return {
        "batch": batch_idx,
        "status": "ok",
        "events": events_config,
        "batch_dir": str(batch_dir),
        "insts": str(batch_insts),
    }


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------

def run_batch(
    batch_info: dict,
    build_dir: Path,
    run_cmd: str,
    mode: str,
    hw_cooldown: float = 2.0,
) -> None:
    """Run one batch on HW or EMU by invoking test.exe.

    Modifies batch_info in place to add status and trace paths.

    Args:
        batch_info: Batch metadata dict (mutated).
        build_dir: Directory containing aie.xclbin (test.exe runs from here).
        run_cmd: Run command template (e.g., "./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin").
        mode: "hw" or "emu".
        hw_cooldown: Seconds between HW runs (ignored for EMU).
    """
    batch_idx = batch_info["batch"]
    batch_dir = Path(batch_info["batch_dir"])
    batch_insts = batch_dir / "insts.bin"
    trace_out = batch_dir / mode
    trace_out.mkdir(parents=True, exist_ok=True)

    # Substitute the trailing insts.bin argument in the run command.
    # Use -i flag replacement first (most test.exe invocations), fall back
    # to bare trailing replacement.
    if " -i insts.bin" in run_cmd:
        cmd = run_cmd.replace(" -i insts.bin", f" -i {batch_insts}")
    else:
        cmd = run_cmd.replace("insts.bin", str(batch_insts))

    env = os.environ.copy()
    env["XDNA_TRACE_DIR"] = str(trace_out)

    if mode == "emu":
        env["XDNA_EMU"] = "1"
        env["XDNA_EMU_LOG_LEVEL"] = env.get("XDNA_EMU_LOG_LEVEL", "info")
        env["XRT_DEVICE_BDF"] = "ffff:ff:1f.0"
    # For HW, XRT_DEVICE_BDF should already be set in the environment.

    status_key = f"{mode}_status"
    trace_key = f"{mode}_trace_raw"

    # Monitor for TDR/IOMMU (HW only).
    tdr_before = _dmesg_count("aie2_tdr_work") if mode == "hw" else 0
    iommu_before = _dmesg_count("IO_PAGE_FAULT") if mode == "hw" else 0

    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True, errors="replace",
            timeout=300, cwd=str(build_dir), env=env,
        )
    except subprocess.TimeoutExpired:
        batch_info[status_key] = "timeout"
        return

    # Check for HW faults.
    if mode == "hw":
        tdr_after = _dmesg_count("aie2_tdr_work")
        iommu_after = _dmesg_count("IO_PAGE_FAULT")
        if iommu_after - iommu_before > 0:
            batch_info[status_key] = "iommu_fault"
            (batch_dir / f"{mode}-run.log").write_text(result.stdout + result.stderr)
            return
        if tdr_after - tdr_before > 0:
            batch_info[status_key] = "tdr"
            (batch_dir / f"{mode}-run.log").write_text(result.stdout + result.stderr)
            return

    if result.returncode != 0:
        batch_info[status_key] = "failed"
        (batch_dir / f"{mode}-run.log").write_text(result.stdout + result.stderr)
    else:
        batch_info[status_key] = "ok"
        trace_raw = trace_out / "trace_raw.bin"
        if trace_raw.exists():
            batch_info[trace_key] = str(trace_raw)

    # Trim trace.
    trim_script = Path(__file__).parent / "trace-trim.py"
    trace_raw = trace_out / "trace_raw.bin"
    if trace_raw.exists():
        subprocess.run(
            [sys.executable, str(trim_script), str(trace_raw)],
            capture_output=True, timeout=30,
        )

    # Convert raw trace to Perfetto JSON for merge.
    trace_raw = trace_out / "trace_raw.bin"
    if trace_raw.exists():
        _convert_trace_to_json(trace_raw, build_dir, trace_out)

    # HW cooldown.
    if mode == "hw" and hw_cooldown > 0:
        if not wait_npu_idle(timeout=hw_cooldown):
            import time
            time.sleep(0.5)


def _convert_trace_to_json(
    trace_raw_path: Path,
    build_dir: Path,
    output_dir: Path,
) -> None:
    """Convert trace_raw.bin to Perfetto JSON using mlir-aie parse_trace.

    Looks for aie_traced.mlir (or aie_arch.mlir) in the build dir or
    its traced/ sibling. Writes trace.json alongside trace_raw.bin.
    """
    import numpy as np

    trace_data = np.fromfile(str(trace_raw_path), dtype=np.uint32)
    if len(trace_data) == 0:
        return

    # Find traced MLIR for parse_trace.
    mlir_candidates = [
        build_dir / "aie_arch.mlir",
        build_dir.parent / "traced" / "aie_traced.mlir",
        build_dir / "aie_traced.mlir",
    ]
    mlir_path = None
    for candidate in mlir_candidates:
        if candidate.exists():
            mlir_path = candidate
            break

    if mlir_path is None:
        return

    try:
        from aie.utils.trace import parse_trace  # type: ignore
        mlir_text = mlir_path.read_text()
        events = parse_trace(trace_data, mlir_text)
        trace_json = output_dir / "trace.json"
        with open(trace_json, "w") as f:
            json.dump(events, f, indent=2)
    except Exception:
        pass  # Non-fatal: merge will skip batches without trace.json


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------

def run_sweep(
    build_dir: Path,
    test_exe: Path,
    output_dir: Path,
    run_cmd: str,
    core_batches: list[list[str]],
    mem_batches: list[list[str]],
    memtile_batches: list[list[str]] | None,
    num_batches: int,
    manifest: dict,
    run_hw: bool = True,
    run_emu: bool = True,
    hw_cooldown: float = 2.0,
) -> list[dict]:
    """Execute all sweep batches and merge traces.

    Returns list of batch_info dicts with status and trace paths.
    """
    script_dir = Path(__file__).parent
    base_insts = build_dir / "insts.bin"

    # Prepare all batches (patch only, no execution).
    batches = []
    for i in range(num_batches):
        info = prepare_batch(
            i,
            core_batches[i],
            mem_batches[i],
            base_insts,
            manifest,
            output_dir,
            memtile_batches[i] if memtile_batches else None,
        )
        batches.append(info)
        if info["status"] != "ok":
            print(f"  Batch {i}: PATCH FAILED -- {info.get('error', '?')}")

    runnable = [b for b in batches if b["status"] == "ok"]
    print(f"  Prepared {len(runnable)}/{num_batches} batches")

    # Run HW batches serially.
    if run_hw and runnable:
        print(f"\n  Running {len(runnable)} HW batches (serial)...")
        for b in runnable:
            print(f"    Batch {b['batch']}...", end=" ", flush=True)
            run_batch(b, build_dir, run_cmd, "hw", hw_cooldown)
            print(b.get("hw_status", "?"))

    # Run EMU batches in parallel.
    if run_emu and runnable:
        emu_jobs = os.cpu_count() or 4
        print(f"\n  Running {len(runnable)} EMU batches (-j{emu_jobs})...")
        with ThreadPoolExecutor(max_workers=emu_jobs) as pool:
            futures = [
                pool.submit(run_batch, b, build_dir, run_cmd, "emu")
                for b in runnable
            ]
            for f in futures:
                f.result()
        for b in runnable:
            print(f"    Batch {b['batch']}: {b.get('emu_status', '?')}")

    # Merge traces.
    merge_script = script_dir / "trace-merge.py"
    if merge_script.exists():
        for mode in ("hw", "emu"):
            traces = [
                str(Path(b["batch_dir"]) / mode / "trace.json")
                for b in runnable
                if (Path(b["batch_dir"]) / mode / "trace.json").exists()
            ]
            if traces:
                merged = output_dir / f"{mode}-merged.json"
                print(f"\n  Merging {len(traces)} {mode} traces...")
                subprocess.run(
                    [sys.executable, str(merge_script)] + traces + ["-o", str(merged)],
                    timeout=120,
                )

    # Write sweep manifest.
    sweep_manifest = {
        "num_batches": num_batches,
        "batches": batches,
    }
    (output_dir / "sweep-manifest.json").write_text(
        json.dumps(sweep_manifest, indent=2) + "\n"
    )

    return batches


def main():
    parser = argparse.ArgumentParser(
        description="Multi-run trace sweep with TRUE metronome alignment. "
                    "Operates on pre-compiled artifacts from the bridge script.",
    )
    parser.add_argument(
        "--build-dir", type=Path, required=True,
        help="Build directory containing aie.xclbin and insts.bin",
    )
    parser.add_argument(
        "--test-exe", type=Path, required=True,
        help="Path to compiled test.exe",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output directory for sweep results",
    )
    parser.add_argument(
        "--run-cmd", type=str, default=None,
        help="Run command template (default: auto from test.exe path)",
    )
    parser.add_argument("--no-hw", action="store_true", help="Skip hardware runs")
    parser.add_argument("--no-emu", action="store_true", help="Skip emulator runs")
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--mem-only", action="store_true")
    parser.add_argument(
        "--hw-cooldown", type=float, default=2.0,
        help="Seconds between HW runs (default: 2)",
    )
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    test_exe = args.test_exe.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs.
    if not (build_dir / "insts.bin").exists():
        print(f"Error: {build_dir}/insts.bin not found", file=sys.stderr)
        sys.exit(1)
    if not test_exe.exists():
        print(f"Error: {test_exe} not found", file=sys.stderr)
        sys.exit(1)

    # Load trace manifest for tile targeting.
    manifest_candidates = [
        build_dir.parent / "traced" / "manifest.json",
        build_dir / "manifest.json",
        build_dir.parent / "traced" / "events.json",
    ]
    manifest = {}
    for candidate in manifest_candidates:
        if candidate.exists():
            manifest = json.loads(candidate.read_text())
            break

    # Determine if memtiles are present.
    tiles = manifest.get("tiles_traced", [])
    has_memtile = any(t.get("tile_type") == "memtile" for t in tiles)

    # Build run command.
    if args.run_cmd:
        run_cmd = args.run_cmd
    else:
        run_cmd = f"{test_exe} -x aie.xclbin -k MLIR_AIE -i insts.bin"

    # Generate event batches.
    noop_batch = ["TRUE"] + ["NONE"] * 7
    core_batches = batch_events(CORE_EVENTS) if not args.mem_only else [noop_batch]
    mem_batches = batch_events(MEM_EVENTS) if not args.core_only else [noop_batch]
    memtile_batches = batch_events(MEMTILE_EVENTS) if has_memtile else None

    num_batches = max(len(core_batches), len(mem_batches))
    if memtile_batches:
        num_batches = max(num_batches, len(memtile_batches))

    # Pad to same length.
    while len(core_batches) < num_batches:
        core_batches.append(noop_batch[:])
    while len(mem_batches) < num_batches:
        mem_batches.append(noop_batch[:])
    if memtile_batches:
        while len(memtile_batches) < num_batches:
            memtile_batches.append(noop_batch[:])

    print(f"Trace sweep: {build_dir.name}")
    print(f"  Build:     {build_dir}")
    print(f"  Test exe:  {test_exe}")
    print(f"  Output:    {output_dir}")
    print(f"  HW:        {'yes' if not args.no_hw else 'no'}")
    print(f"  EMU:       {'yes' if not args.no_emu else 'no'}")
    print(f"  Batches:   {num_batches}")
    print()

    # Execute sweep.
    batches = run_sweep(
        build_dir=build_dir,
        test_exe=test_exe,
        output_dir=output_dir,
        run_cmd=run_cmd,
        core_batches=core_batches,
        mem_batches=mem_batches,
        memtile_batches=memtile_batches,
        num_batches=num_batches,
        manifest=manifest,
        run_hw=not args.no_hw,
        run_emu=not args.no_emu,
        hw_cooldown=args.hw_cooldown,
    )

    # Summary.
    ok = sum(1 for b in batches if b["status"] == "ok")
    hw_ok = sum(1 for b in batches if b.get("hw_status") == "ok")
    emu_ok = sum(1 for b in batches if b.get("emu_status") == "ok")
    print(f"\nSweep complete: {ok}/{num_batches} patched")
    if not args.no_hw:
        print(f"  HW:  {hw_ok}/{ok} OK")
    if not args.no_emu:
        print(f"  EMU: {emu_ok}/{ok} OK")


if __name__ == "__main__":
    main()
