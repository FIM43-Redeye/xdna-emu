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


def compile_kernel_objects(test_dir: Path, build_dir: Path) -> tuple[bool, str]:
    """Compile kernel object files referenced in run.lit.

    Extracts kernel compile commands (xchesscc_wrapper, clang with -c) from
    run.lit, resolves lit-style variables, and runs them in build_dir.

    Returns (success, error_detail).
    """
    lit_file = test_dir / "run.lit"
    if not lit_file.exists():
        return True, ""  # No run.lit = no kernel objects needed

    # Environment for variable resolution (mirrors bridge script's apply_lit_subs)
    peano_dir = os.environ.get("PEANO_INSTALL_DIR", "")
    aietools_dir = os.environ.get("AIETOOLS_DIR",
                                   str(test_dir.parent.parent.parent / "aietools"))
    aie_runtime_lib = os.environ.get("AIE_RUNTIME_LIB", "")

    commands = []
    for line in lit_file.read_text().splitlines():
        if "RUN:" not in line:
            continue
        cmd = line.split("RUN:", 1)[1].strip()

        # Only kernel compile commands: xchesscc or clang/clang++ with -c producing .o
        is_kernel_compile = False
        if cmd.startswith("xchesscc_wrapper ") and " -c " in cmd:
            is_kernel_compile = True
        elif ("%cxx" in cmd or "clang" in cmd) and " -c " in cmd and ".o" in cmd:
            # Peano kernel compile (not host test.cpp)
            if "test.cpp" not in cmd:
                is_kernel_compile = True

        if not is_kernel_compile:
            continue

        # Resolve lit variables
        cmd = cmd.replace("%S", str(test_dir))
        cmd = cmd.replace("%aietools", aietools_dir)
        cmd = cmd.replace("%aie_runtime_lib%", aie_runtime_lib)
        if "%cxx" in cmd:
            cxx = f"{peano_dir}/bin/clang++" if peano_dir else "clang++"
            cmd = cmd.replace("%cxx", cxx)

        commands.append(cmd)

    if not commands:
        return True, ""  # No kernel objects to compile

    for cmd in commands:
        print(f"    Kernel: {cmd.split('/')[-1] if '/' in cmd else cmd[:60]}")
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=120, cwd=str(build_dir),
        )
        if result.returncode != 0:
            detail = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown"
            return False, detail

    return True, ""


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
    tile_filter: str | None = None,
) -> tuple[Path | None, Path | None]:
    """Inject tracing with default events and compile ONCE.

    Returns (traced_dir, manifest_path) on success, (None, None) on failure.
    All batches will reuse this compiled xclbin + patched insts.bin.

    If tile_filter is set, passes --tiles to trace-inject.py to limit
    which tiles are traced.  Format: "col.row:module,col.row:module,...".
    """
    script_dir = Path(__file__).parent
    traced_dir = output_dir / "base"
    traced_dir.mkdir(parents=True, exist_ok=True)

    # Inject with default events (no --events-json = defaults)
    label = "Base" if tile_filter is None else f"Pass (tiles={tile_filter})"
    print(f"  {label}: injecting trace routing...")
    cmd = [
        sys.executable, str(script_dir / "trace-inject.py"),
        str(test_dir),
        "--output", str(traced_dir),
        "--trace-size", str(trace_size),
    ]
    if tile_filter:
        cmd.extend(["--tiles", tile_filter])
    result = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"    Inject failed: {result.stderr}", file=sys.stderr)
        _write_skip_manifest(traced_dir, "inject_failed", result.stderr)
        return None, None

    manifest_path = traced_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("skipped"):
        print(f"    Skipped: {manifest.get('reason', 'unknown')}")
        return None, None

    # Compile kernel objects referenced in run.lit (scale.o, etc.)
    print(f"  {label}: compiling kernel objects...")
    kernel_ok, kernel_err = compile_kernel_objects(test_dir, traced_dir)
    if not kernel_ok:
        print(f"    Kernel compile failed: {kernel_err}", file=sys.stderr)
        _write_skip_manifest(traced_dir, "kernel_compile_failed", kernel_err)
        return None, None

    # Compile traced xclbin
    print(f"  {label}: compiling traced xclbin...")
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
        stderr = compile_result.stderr
        (output_dir / "compile.log").write_text(
            compile_result.stdout + stderr
        )
        # Distinguish routing conflicts from generic compile failures
        if "pathfinder-flows" in stderr or "could not route" in stderr.lower():
            reason = "routing_conflict"
        else:
            reason = "compile_failed"
        print(f"    Compile failed: {reason} (see compile.log)", file=sys.stderr)
        _write_skip_manifest(traced_dir, reason, stderr.strip().split("\n")[-1])
        return None, None

    print(f"  {label}: compile OK")
    return traced_dir, manifest_path


def prepare_batch(
    batch_idx: int,
    core_batch: list[str],
    mem_batch: list[str],
    base_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    memtile_batch: list[str] | None = None,
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
    if memtile_batch is not None:
        events_config["memtile_events"] = memtile_batch
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


def _dmesg_count(pattern: str) -> int:
    """Count lines matching pattern in dmesg output."""
    try:
        result = subprocess.run(
            ["dmesg"], capture_output=True, text=True, timeout=5,
        )
        return sum(1 for line in result.stdout.splitlines() if pattern in line)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0


def wait_npu_idle(timeout: float = 10.0, poll_interval: float = 0.1) -> bool:
    """Wait until the NPU has no active hardware contexts.

    Polls `xrt-smi examine -r aie-partitions` for the "No hardware contexts"
    message. Returns True if idle detected within timeout, False if timed out.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["xrt-smi", "examine", "-r", "aie-partitions"],
            capture_output=True, text=True, timeout=5,
        )
        if "No hardware contexts running" in result.stdout:
            return True
        time.sleep(poll_interval)
    return False


def run_batch_hw(
    batch_info: dict,
    hw_cooldown: float = 2.0,
    output_subdir: str = "hw",
) -> dict:
    """Run one batch on hardware.

    Args:
        batch_info: Batch metadata dict (mutated in place).
        hw_cooldown: Seconds to wait for NPU idle after run.
        output_subdir: Directory name under batch_dir for output
            (default "hw"; use "hw-serial"/"hw-parallel" for comparison).
    """
    script_dir = Path(__file__).parent
    trace_run = script_dir / "trace-run.py"
    batch_idx = batch_info["batch"]
    batch_dir = Path(batch_info["batch_dir"])
    manifest = batch_info["manifest"]

    hw_dir = batch_dir / output_subdir
    print(f"  Batch {batch_idx}: running on hardware ({output_subdir})...")

    # Snapshot TDR and IOMMU fault counts for post-run detection.
    tdr_before = _dmesg_count("aie2_tdr_work")
    iommu_before = _dmesg_count("IO_PAGE_FAULT")

    hw_result = subprocess.run(
        [sys.executable, str(trace_run), str(manifest), "-o", str(hw_dir)],
        capture_output=True, text=True, timeout=300,
    )
    # Use namespaced keys for non-default output dirs, standard keys for "hw".
    status_key = "hw_status" if output_subdir == "hw" else f"{output_subdir}_status"
    trace_key = "hw_trace" if output_subdir == "hw" else f"{output_subdir}_trace"

    # Check for TDR or IOMMU faults.
    tdr_after = _dmesg_count("aie2_tdr_work")
    iommu_after = _dmesg_count("IO_PAGE_FAULT")
    tdr_new = tdr_after - tdr_before
    iommu_new = iommu_after - iommu_before

    if iommu_new > 0:
        batch_info[status_key] = "iommu_fault"
        print(f"  Batch {batch_idx}: IOMMU FAULT ({iommu_new} page faults)")
        (batch_dir / f"{output_subdir}-run.log").write_text(
            hw_result.stdout + hw_result.stderr
            + f"\nIOMMU FAULT: {iommu_new} new page faults\n"
        )
    elif tdr_new > 0:
        batch_info[status_key] = "tdr"
        print(f"  Batch {batch_idx}: TDR ({tdr_new} events)")
        (batch_dir / f"{output_subdir}-run.log").write_text(
            hw_result.stdout + hw_result.stderr
            + f"\nTDR: {tdr_new} new aie2_tdr_work events\n"
        )
    elif hw_result.returncode != 0:
        batch_info[status_key] = "failed"
        (batch_dir / f"{output_subdir}-run.log").write_text(
            hw_result.stdout + hw_result.stderr
        )
    else:
        batch_info[status_key] = "ok"
        batch_info[trace_key] = str(hw_dir / "trace.json")
        trim_trace_dir(hw_dir)

    # Wait for NPU to be idle before next run.
    if hw_cooldown > 0 and not wait_npu_idle(timeout=hw_cooldown):
        import time
        time.sleep(0.5)  # Brief fallback if poll failed

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
        "--hw-jobs", type=int, default=1,
        help="Parallel hardware batch jobs (default: 1 = serial). "
             "NPU1 supports up to 5 concurrent contexts.",
    )
    parser.add_argument(
        "--emu-jobs", type=int, default=0,
        help="Parallel emulator jobs (default: nproc)",
    )
    parser.add_argument(
        "--compare-parallel", action="store_true",
        help="Run HW batches twice (serial then parallel at --hw-jobs), "
             "compare traces to measure determinism under concurrent load.",
    )
    parser.add_argument(
        "--core-only", action="store_true",
        help="Only sweep core events (skip memory events)",
    )
    parser.add_argument(
        "--mem-only", action="store_true",
        help="Only sweep memory events (skip core events)",
    )
    parser.add_argument(
        "--compile-only", action="store_true",
        help="Only inject and compile traced xclbin, skip HW/EMU runs",
    )
    parser.add_argument(
        "--use-base", type=Path, default=None,
        help="Use pre-compiled base dir (skip inject+compile)",
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

    # Memtile batches are generated only if the design contains memtiles.
    # Detection happens after base compilation (manifest has tile list),
    # so we initialize to None here and populate after compile_base_trace.
    memtile_batches: list[list[str]] | None = None

    # Pair core and mem batches. If counts differ, extend the shorter with NONEs.
    num_batches = max(len(core_batches), len(mem_batches))
    noop_batch = ["TRUE"] + ["NONE"] * 7
    while len(core_batches) < num_batches:
        core_batches.append(noop_batch)
    while len(mem_batches) < num_batches:
        mem_batches.append(noop_batch)

    print(f"  Batches:   {num_batches} (before memtile detection)")
    print()

    # --- Plan trace passes ------------------------------------------------
    # Call the planner to determine whether all tiles fit through one shim
    # (single pass) or need multiple injection+compile passes.
    script_dir = Path(__file__).parent
    plan = None
    print("  Planning trace routing...")
    try:
        plan_result = subprocess.run(
            [
                sys.executable, str(script_dir / "trace-inject.py"),
                str(test_dir),
                "--output", str(output_dir / "plan_tmp"),
                "--plan-only",
            ],
            capture_output=True, text=True, timeout=60,
        )
        if plan_result.returncode != 0:
            print(f"  Planner failed: {plan_result.stderr.strip()}", file=sys.stderr)
            print("  Falling back to single-pass (existing behavior)")
        else:
            plan = json.loads(plan_result.stdout)
            num_passes = plan["num_passes"]
            print(f"  Plan: {plan['reason']} "
                  f"({num_passes} pass{'es' if num_passes != 1 else ''}, "
                  f"{plan['total_tiles']} tiles)")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
        print(f"  Planner error: {e}", file=sys.stderr)
        print("  Falling back to single-pass (existing behavior)")

    # Task 7: clear error when tracing is impossible (zero passes).
    if plan and plan["num_passes"] == 0:
        reason = plan.get("reason", "unknown")
        print(f"\n  TRACE CAPACITY ERROR: {reason}")
        _write_skip_manifest(output_dir, "capacity_exceeded", reason)
        sweep_manifest = {
            "test_name": test_name,
            "skipped": True,
            "reason": "capacity_exceeded",
            "detail": reason,
            "num_batches": 0,
            "batches": [],
        }
        (output_dir / "sweep-manifest.json").write_text(
            json.dumps(sweep_manifest, indent=2) + "\n"
        )
        print(f"Sweep skipped: capacity_exceeded")
        sys.exit(0)

    # --- Compile-only mode: compile base(s) and exit -----------------------

    if args.compile_only:
        status_file = output_dir / "compile-status.txt"
        try:
            if plan and plan["num_passes"] > 1:
                # Multi-pass: compile each pass's base into its own subdir.
                all_ok = True
                for pass_idx, trace_pass in enumerate(plan["passes"]):
                    tile_specs = []
                    for t in trace_pass["tiles"]:
                        tile_specs.append(f"{t['col']}.{t['row']}:{t['module']}")
                    tile_filter = ",".join(tile_specs)

                    pass_dir = output_dir / f"pass_{pass_idx:02d}"
                    pass_dir.mkdir(parents=True, exist_ok=True)
                    print(f"\n  Pass {pass_idx}: compiling "
                          f"({len(trace_pass['tiles'])} tiles)")

                    base_dir, base_manifest = compile_base_trace(
                        test_dir, pass_dir, args.trace_size,
                        tile_filter=tile_filter,
                    )
                    if base_dir is None:
                        print(f"    Pass {pass_idx} compile failed")
                        all_ok = False

                if all_ok:
                    status_file.write_text("OK\n")
                    print(f"\nCompile-only: all passes compiled successfully")
                else:
                    status_file.write_text("FAIL some passes failed to compile\n")
                    print(f"\nCompile-only: some passes failed")
                    sys.exit(1)
            else:
                # Single pass: compile one base.
                base_dir, base_manifest = compile_base_trace(
                    test_dir, output_dir, args.trace_size,
                )
                if base_dir is not None:
                    status_file.write_text("OK\n")
                    print(f"\nCompile-only: OK")
                else:
                    status_file.write_text("FAIL compile_base_trace failed\n")
                    print(f"\nCompile-only: FAIL")
                    sys.exit(1)
        except Exception as e:
            status_file.write_text(f"FAIL {e}\n")
            print(f"\nCompile-only: FAIL ({e})")
            sys.exit(1)
        return

    # --- Multi-pass vs single-pass execution ------------------------------

    if plan and plan["num_passes"] > 1:
        # Multi-pass: each pass compiles and sweeps independently in its own
        # subdirectory, then results are aggregated into the top-level manifest.
        _run_multi_pass(
            plan, test_dir, output_dir, test_name, args,
            core_batches, mem_batches, memtile_batches,
            num_batches, noop_batch, script_dir,
        )
    else:
        # Single pass: existing behavior (compile once, sweep all batches).
        _run_single_pass(
            test_dir, output_dir, test_name, args,
            core_batches, mem_batches, memtile_batches,
            num_batches, noop_batch, script_dir,
        )


def _run_single_pass(
    test_dir: Path,
    output_dir: Path,
    test_name: str,
    args,
    core_batches: list[list[str]],
    mem_batches: list[list[str]],
    memtile_batches: list[list[str]] | None,
    num_batches: int,
    noop_batch: list[str],
    script_dir: Path,
):
    """Run the standard single-pass sweep (original behavior)."""
    trace_size = args.trace_size

    # Use pre-compiled base if provided, otherwise compile fresh.
    if getattr(args, "use_base", None) is not None:
        use_base = args.use_base.resolve()
        base_manifest_path = use_base / "manifest.json"
        if not use_base.is_dir() or not base_manifest_path.exists():
            print(f"Error: --use-base dir missing or lacks manifest.json: {use_base}",
                  file=sys.stderr)
            sys.exit(1)
        base_dir = use_base
        base_manifest = base_manifest_path
        print(f"  Using pre-compiled base: {use_base}")
    else:
        # Compile base trace once (all batches share xclbin, patch insts.bin)
        base_dir, base_manifest = compile_base_trace(
            test_dir, output_dir, trace_size,
        )
    if base_dir is None:
        # Check if a skip manifest was written with a specific reason
        skip_manifest = output_dir / "base" / "sweep-manifest.json"
        if skip_manifest.exists():
            skip_info = json.loads(skip_manifest.read_text())
            reason = skip_info.get("reason", "unknown")
            # Copy to top-level sweep manifest so the bridge script can parse it
            sweep_manifest = {
                "test_name": test_name,
                "skipped": True,
                "reason": reason,
                "detail": skip_info.get("detail", ""),
                "num_batches": 0,
                "batches": [],
            }
            (output_dir / "sweep-manifest.json").write_text(
                json.dumps(sweep_manifest, indent=2) + "\n"
            )
            print(f"Sweep skipped: {reason}")
            sys.exit(0)
        print("Failed to compile base trace. Aborting sweep.", file=sys.stderr)
        sys.exit(1)
    print()

    # Detect memtiles from manifest and generate memtile event batches
    manifest_data = json.loads(base_manifest.read_text())
    has_memtiles = any(
        t.get("tile_type") == "memtile" or t.get("row") == 1
        for t in manifest_data.get("tiles_traced", [])
    )
    if has_memtiles:
        memtile_batches = batch_events(MEMTILE_EVENTS)
        # Extend core/mem batch lists to cover memtile batches too
        new_total = max(num_batches, len(memtile_batches))
        while len(core_batches) < new_total:
            core_batches.append(noop_batch)
        while len(mem_batches) < new_total:
            mem_batches.append(noop_batch)
        while len(memtile_batches) < new_total:
            memtile_batches.append(noop_batch)
        num_batches = new_total
        print(f"  Memtiles detected: adding {len(batch_events(MEMTILE_EVENTS))} memtile batches")
        print(f"  Total batches: {num_batches}")
    print()

    # Prepare all batches (patch insts.bin, no execution yet)
    batches = []
    for i in range(num_batches):
        mt_batch = memtile_batches[i] if memtile_batches is not None else None
        info = prepare_batch(
            i, core_batches[i], mem_batches[i],
            base_dir, base_manifest, output_dir,
            memtile_batch=mt_batch,
        )
        batches.append(info)
    print()

    # Execute batches and write results.
    results = _execute_and_merge(
        batches, output_dir, test_name, args, script_dir,
    )

    _write_single_pass_manifest(output_dir, test_name, num_batches, results)
    _print_summary(output_dir, test_name, num_batches, results)


def _run_multi_pass(
    plan: dict,
    test_dir: Path,
    output_dir: Path,
    test_name: str,
    args,
    core_batches: list[list[str]],
    mem_batches: list[list[str]],
    memtile_batches: list[list[str]] | None,
    num_batches: int,
    noop_batch: list[str],
    script_dir: Path,
):
    """Run a multi-pass sweep: each pass is an independent sub-sweep.

    Each pass compiles with a tile filter, then runs the full batch+execute
    flow in its own pass_NN/ subdirectory.  The top-level sweep manifest
    aggregates all pass results.
    """
    trace_size = args.trace_size
    pass_manifests = []
    total_batches_across_passes = 0

    for pass_idx, trace_pass in enumerate(plan["passes"]):
        # Build tile filter string: "col.row:module,col.row:module,..."
        tile_specs = []
        for t in trace_pass["tiles"]:
            tile_specs.append(f"{t['col']}.{t['row']}:{t['module']}")
        tile_filter = ",".join(tile_specs)

        pass_dir = output_dir / f"pass_{pass_idx:02d}"
        pass_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Pass {pass_idx}: {len(trace_pass['tiles'])} tiles "
              f"-> shim col {trace_pass['shim_col']}")

        # Use pre-compiled base if provided, otherwise compile fresh.
        if getattr(args, "use_base", None) is not None:
            use_base = args.use_base.resolve()
            pre_pass_dir = use_base / f"pass_{pass_idx:02d}" / "base"
            pre_manifest = pre_pass_dir / "manifest.json"
            if pre_pass_dir.is_dir() and pre_manifest.exists():
                base_dir = pre_pass_dir
                base_manifest = pre_manifest
                print(f"    Using pre-compiled base: {pre_pass_dir}")
            else:
                # Fall back to compiling if pre-compiled pass not found.
                print(f"    Pre-compiled pass {pass_idx} not found, compiling...")
                base_dir, base_manifest = compile_base_trace(
                    test_dir, pass_dir, trace_size, tile_filter=tile_filter,
                )
        else:
            base_dir, base_manifest = compile_base_trace(
                test_dir, pass_dir, trace_size, tile_filter=tile_filter,
            )
        if base_dir is None:
            print(f"    Pass {pass_idx} compile failed, skipping")
            pass_manifests.append({
                "pass_idx": pass_idx,
                "tiles": trace_pass["tiles"],
                "shim_col": trace_pass["shim_col"],
                "status": "compile_failed",
                "num_batches": 0,
                "batches": [],
            })
            continue

        # Detect memtiles in this pass and build batches accordingly.
        manifest_data = json.loads(base_manifest.read_text())
        pass_has_memtiles = any(
            t.get("tile_type") == "memtile" or t.get("row") == 1
            for t in manifest_data.get("tiles_traced", [])
        )

        # Copy batch lists so per-pass extensions don't mutate the originals.
        pass_core = list(core_batches)
        pass_mem = list(mem_batches)
        pass_mt: list[list[str]] | None = (
            list(memtile_batches) if memtile_batches is not None else None
        )
        pass_num_batches = num_batches

        if pass_has_memtiles:
            if pass_mt is None:
                pass_mt = batch_events(MEMTILE_EVENTS)
            new_total = max(pass_num_batches, len(pass_mt))
            while len(pass_core) < new_total:
                pass_core.append(noop_batch)
            while len(pass_mem) < new_total:
                pass_mem.append(noop_batch)
            while len(pass_mt) < new_total:
                pass_mt.append(noop_batch)
            pass_num_batches = new_total

        # Prepare batches for this pass.
        batches = []
        for i in range(pass_num_batches):
            mt_batch = pass_mt[i] if pass_mt is not None else None
            info = prepare_batch(
                i, pass_core[i], pass_mem[i],
                base_dir, base_manifest, pass_dir,
                memtile_batch=mt_batch,
            )
            batches.append(info)

        # Execute and merge within this pass's directory.
        results = _execute_and_merge(
            batches, pass_dir, test_name, args, script_dir,
        )

        total_batches_across_passes += pass_num_batches
        pass_manifests.append({
            "pass_idx": pass_idx,
            "tiles": trace_pass["tiles"],
            "shim_col": trace_pass["shim_col"],
            "status": "ok",
            "num_batches": pass_num_batches,
            "batches": results,
        })

    # Check if all passes failed.
    if all(p["status"] != "ok" for p in pass_manifests):
        _write_skip_manifest(output_dir, "all_passes_failed",
                             "Multi-pass trace: all injection passes failed to compile")
        sweep_manifest = {
            "test_name": test_name,
            "skipped": True,
            "reason": "all_passes_failed",
            "detail": "Multi-pass trace: all injection passes failed to compile",
            "num_batches": 0,
            "batches": [],
        }
        (output_dir / "sweep-manifest.json").write_text(
            json.dumps(sweep_manifest, indent=2) + "\n"
        )
        print(f"\nSweep skipped: all passes failed")
        sys.exit(0)

    # Write top-level multi-pass sweep manifest.
    sweep_manifest = {
        "test_name": test_name,
        "num_passes": len(pass_manifests),
        "passes": pass_manifests,
        "num_batches": total_batches_across_passes,
    }
    manifest_path = output_dir / "sweep-manifest.json"
    manifest_path.write_text(json.dumps(sweep_manifest, indent=2) + "\n")

    # Summary.
    ok_passes = sum(1 for p in pass_manifests if p["status"] == "ok")
    ok_batches = sum(
        sum(1 for b in p.get("batches", [])
            if isinstance(b, dict) and b.get("status") == "ok")
        for p in pass_manifests
    )
    print(f"\nSweep complete: {test_name}")
    print(f"  Passes:   {ok_passes}/{len(pass_manifests)}")
    print(f"  Batches:  {ok_batches}/{total_batches_across_passes}")
    print(f"  Manifest: {manifest_path}")
    print(f"\nView in Perfetto: https://ui.perfetto.dev/")


def _run_hw_batches(
    runnable: list[dict],
    hw_jobs: int,
    hw_cooldown: float,
    output_subdir: str = "hw",
):
    """Run HW batches, serial or parallel depending on hw_jobs."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if hw_jobs <= 1:
        # Serial: one at a time with cooldown between runs.
        label = f"serial, {hw_cooldown}s cooldown"
        print(f"  HW:  {len(runnable)} batches ({label}, -> {output_subdir}/)")
        for b in runnable:
            run_batch_hw(b, hw_cooldown=hw_cooldown, output_subdir=output_subdir)
    else:
        # Parallel: up to hw_jobs concurrent NPU contexts.
        label = f"parallel, -j{hw_jobs}"
        print(f"  HW:  {len(runnable)} batches ({label}, -> {output_subdir}/)")
        with ThreadPoolExecutor(max_workers=hw_jobs) as pool:
            futures = {
                pool.submit(
                    run_batch_hw, b,
                    hw_cooldown=0.2,
                    output_subdir=output_subdir,
                ): b
                for b in runnable
            }
            for f in as_completed(futures):
                f.result()  # propagate exceptions


def _compare_serial_vs_parallel(
    batches: list[dict],
    output_dir: Path,
    script_dir: Path,
):
    """Compare serial vs parallel HW traces per-batch using trace-compare."""
    # Prefer Rust binary, fall back to Python.
    rust_bin = script_dir.parent / "target" / "release" / "trace-compare"
    compare_cmd = (
        [str(rust_bin)] if rust_bin.exists()
        else [sys.executable, str(script_dir / "trace-compare.py")]
    )

    report_path = output_dir / "serial-vs-parallel-report.txt"
    lines = ["=== Serial vs Parallel Trace Comparison ===", ""]

    total = 0
    identical = 0
    diverged = 0
    errors = 0

    for b in batches:
        batch_idx = b["batch"]
        batch_dir = Path(b["batch_dir"])
        serial_raw = batch_dir / "hw-serial" / "trace_raw.bin"
        parallel_raw = batch_dir / "hw-parallel" / "trace_raw.bin"

        if not serial_raw.exists() or not parallel_raw.exists():
            lines.append(f"Batch {batch_idx}: SKIP (missing trace data)")
            errors += 1
            continue

        # Reject all-zeros traces -- they indicate a failed/TDR'd run.
        def _is_all_zeros(path: Path) -> bool:
            with open(path, "rb") as f:
                chunk = f.read(4096)
                return len(chunk) > 0 and chunk == b"\x00" * len(chunk)

        serial_zeros = _is_all_zeros(serial_raw)
        parallel_zeros = _is_all_zeros(parallel_raw)
        if serial_zeros and parallel_zeros:
            lines.append(f"Batch {batch_idx}: EMPTY (both traces all-zeros -- TDR/fault?)")
            errors += 1
            continue
        if serial_zeros:
            lines.append(f"Batch {batch_idx}: SERIAL_EMPTY (serial trace all-zeros)")
            diverged += 1
            continue
        if parallel_zeros:
            lines.append(f"Batch {batch_idx}: PARALLEL_EMPTY (parallel trace all-zeros -- TDR?)")
            diverged += 1
            continue

        total += 1
        batch_report = batch_dir / "serial-vs-parallel.txt"

        # Compare with --remap-columns: the NPU driver assigns different
        # physical columns to each run, so we normalize to logical columns.
        result = subprocess.run(
            compare_cmd + [
                "--remap-columns",
                "--hw", str(serial_raw),
                "--emu", str(parallel_raw),
                "-o", str(batch_report),
            ],
            capture_output=True, text=True, timeout=60,
        )

        # Check if raw bytes are identical first (fast path).
        import filecmp
        if filecmp.cmp(str(serial_raw), str(parallel_raw), shallow=False):
            lines.append(f"Batch {batch_idx}: IDENTICAL (byte-for-byte)")
            identical += 1
        elif result.returncode == 0 and batch_report.exists():
            report_text = batch_report.read_text()

            # For serial-vs-parallel HW comparison, the right question is:
            # "do both runs observe the same event types?"
            #
            # Expected differences between any two HW runs:
            # - Absolute timestamps differ (different NPU clock start)
            # - Total event counts differ (trace buffer fills at different
            #   rates under different load)
            # - Timing drift accumulates (DMA startup jitter, NoC latency)
            #
            # Real non-determinism would be:
            # - An event type present in one trace but completely absent
            #   in the other (count N/0 or 0/N with no overlap at all)
            # - Events in fundamentally different order (not just shifted)

            # Parse the summary line for clean/diverged/count-mismatch.
            import re
            edge_m = re.search(
                r"Edge event types:\s+(\d+) clean, (\d+) diverged, (\d+) count mismatch",
                report_text,
            )
            if edge_m:
                n_clean = int(edge_m.group(1))
                n_diverged = int(edge_m.group(2))
                n_count_mm = int(edge_m.group(3))
                n_total_types = n_clean + n_diverged + n_count_mm

                # Check for missing event types: N/0 or 0/N in paired counts.
                # This would indicate a real non-determinism.
                missing_types = 0
                for line_text in report_text.splitlines():
                    m = re.search(r"\[edge\]\s+\S+\s+(\d+)/(\d+)", line_text)
                    if m:
                        hw_n, emu_n = int(m.group(1)), int(m.group(2))
                        if (hw_n == 0) != (emu_n == 0):
                            missing_types += 1

                if missing_types > 0:
                    lines.append(
                        f"Batch {batch_idx}: MISSING_EVENTS "
                        f"({missing_types} types absent in one run)"
                    )
                    diverged += 1
                else:
                    # All event types present in both runs.  Timing drift
                    # and count differences are expected between HW runs.
                    lines.append(
                        f"Batch {batch_idx}: DETERMINISTIC "
                        f"({n_total_types} event types, "
                        f"{n_clean} timing-clean, "
                        f"{n_diverged} timing-shifted, "
                        f"{n_count_mm} count-differ)"
                    )
                    identical += 1
            else:
                # Couldn't parse summary -- fall back to old heuristic.
                lines.append(f"Batch {batch_idx}: UNKNOWN (see {batch_report})")
                errors += 1
        else:
            lines.append(f"Batch {batch_idx}: ERROR (compare failed)")
            errors += 1

    lines.append("")
    lines.append(f"Summary: {total} batches compared")
    lines.append(f"  Identical/timing-only: {identical}")
    lines.append(f"  Diverged:              {diverged}")
    lines.append(f"  Errors:                {errors}")

    if diverged == 0 and errors == 0:
        lines.append("")
        lines.append("DETERMINISTIC: Parallel execution does not affect trace event sequences.")
    elif diverged > 0:
        lines.append("")
        lines.append(
            f"NON-DETERMINISTIC: {diverged} batch(es) show different events under parallel load."
        )

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)
    print(report_text)


def _execute_and_merge(
    batches: list[dict],
    output_dir: Path,
    test_name: str,
    args,
    script_dir: Path,
) -> list[dict]:
    """Execute batches (HW + EMU) and merge traces.

    Returns the batch results list (mutated in place with hw/emu status).
    """
    from concurrent.futures import ThreadPoolExecutor

    runnable = [b for b in batches if b["status"] == "ok"]
    emu_jobs = args.emu_jobs if args.emu_jobs > 0 else os.cpu_count() or 4
    hw_jobs = getattr(args, "hw_jobs", 1)

    compare_parallel = getattr(args, "compare_parallel", False)

    if compare_parallel and not args.no_hw:
        # --- Compare mode: run HW serial, then HW parallel, then compare ---
        print("  === Serial vs Parallel comparison mode ===")

        # Pass 1: serial
        _run_hw_batches(runnable, hw_jobs=1, hw_cooldown=args.hw_cooldown,
                        output_subdir="hw-serial")

        # Pass 2: parallel
        parallel_jobs = hw_jobs if hw_jobs > 1 else 5
        _run_hw_batches(runnable, hw_jobs=parallel_jobs,
                        hw_cooldown=0.2, output_subdir="hw-parallel")

        # Copy serial results into default "hw" keys for merge compatibility.
        for b in runnable:
            b["hw_status"] = b.get("hw-serial_status", "failed")
            if "hw-serial_trace" in b:
                b["hw_trace"] = b["hw-serial_trace"]

        # Run EMU concurrently (already done by the time we get here, or now).
        with ThreadPoolExecutor(max_workers=emu_jobs) as pool:
            if not args.no_emu:
                print(f"  EMU: {len(runnable)} batches (parallel, -j{emu_jobs})")
                emu_futures = [pool.submit(run_batch_emu, b) for b in runnable]
                for f in emu_futures:
                    f.result()

        # Compare serial vs parallel per-batch.
        print()
        _compare_serial_vs_parallel(runnable, output_dir, script_dir)

    else:
        # --- Normal mode: single HW pass + EMU ---
        with ThreadPoolExecutor(max_workers=emu_jobs + 1) as pool:
            hw_future = None
            if not args.no_hw:
                hw_future = pool.submit(
                    _run_hw_batches, runnable, hw_jobs, args.hw_cooldown
                )

            emu_futures = []
            if not args.no_emu:
                print(f"  EMU: {len(runnable)} batches (parallel, -j{emu_jobs})")
                emu_futures = [pool.submit(run_batch_emu, b) for b in runnable]

            for f in emu_futures:
                f.result()
            if hw_future:
                hw_future.result()

    print()

    # Merge traces.
    merge_script = script_dir / "trace-merge.py"
    hw_traces = [r["hw_trace"] for r in batches if r.get("hw_trace")]
    emu_traces = [r["emu_trace"] for r in batches if r.get("emu_trace")]

    if hw_traces and merge_script.exists():
        print(f"  Merging {len(hw_traces)} hardware traces...")
        hw_merged = output_dir / "hw-merged.json"
        subprocess.run(
            [sys.executable, str(merge_script)] + hw_traces + ["-o", str(hw_merged)],
            timeout=120,
        )

    if emu_traces and merge_script.exists():
        print(f"  Merging {len(emu_traces)} emulator traces...")
        emu_merged = output_dir / "emu-merged.json"
        subprocess.run(
            [sys.executable, str(merge_script)] + emu_traces + ["-o", str(emu_merged)],
            timeout=120,
        )

    # Cross-platform comparison (if both HW and EMU ran).
    compare_script = script_dir / "trace-compare.py"
    hw_merged = output_dir / "hw-merged.json"
    emu_merged = output_dir / "emu-merged.json"
    report_path = output_dir / "comparison-report.txt"

    if hw_merged.exists() and emu_merged.exists() and compare_script.exists():
        print(f"  Comparing HW vs EMU (boot-normalized)...")
        subprocess.run(
            [
                sys.executable, str(compare_script),
                str(hw_merged), str(emu_merged),
                "-o", str(report_path),
            ],
            timeout=120,
        )

    return batches


def _write_single_pass_manifest(
    output_dir: Path,
    test_name: str,
    num_batches: int,
    results: list[dict],
):
    """Write the sweep-manifest.json for a single-pass sweep."""
    sweep_manifest = {
        "test_name": test_name,
        "num_batches": num_batches,
        "batches": results,
    }
    manifest_path = output_dir / "sweep-manifest.json"
    manifest_path.write_text(json.dumps(sweep_manifest, indent=2) + "\n")


def _print_summary(
    output_dir: Path,
    test_name: str,
    num_batches: int,
    results: list[dict],
):
    """Print end-of-sweep summary for a single-pass sweep."""
    manifest_path = output_dir / "sweep-manifest.json"
    hw_merged = output_dir / "hw-merged.json"
    emu_merged = output_dir / "emu-merged.json"
    report_path = output_dir / "comparison-report.txt"

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
