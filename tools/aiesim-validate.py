#!/usr/bin/env python3
"""Validate xdna-emu against aiesimulator using VCD comparison.

Compiles a test kernel targeting xcve2802 (Versal AIE2), runs it through
aiesimulator to get reference VCD output, runs the same kernel through
the emulator, and compares the two VCDs.

Architecture mapping
--------------------
xcve2802 (Versal AIE2, 38x11) tiles are architecturally IDENTICAL to NPU1
(Phoenix, 4x6) tiles -- same ISA, registers, DMA, locks. Only array
topology differs:

    NPU1:     row 0 = shim,  row 1 = memtile,  rows 2-5 = core
    xcve2802: row 0 = shim,  rows 1-2 = memtile, rows 3-10 = core

Core row offset when comparing: NPU1 core row N = xcve2802 core row N+1.

Usage:
    python3 tools/aiesim-validate.py <mlir-test-dir> [options]

Examples:
    # Validate add_one_using_dma kernel
    python3 tools/aiesim-validate.py \\
        ../mlir-aie/test/npu-xrt/add_one_using_dma/ \\
        --output /tmp/aiesim-validate

    # Skip compilation (use existing sim package)
    python3 tools/aiesim-validate.py \\
        --pkg-dir /path/to/existing.prj/sim \\
        --output /tmp/aiesim-validate

    # Just run aiesim (no emulator comparison)
    python3 tools/aiesim-validate.py \\
        ../mlir-aie/test/npu-xrt/add_one_using_dma/ \\
        --aiesim-only
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Core row offset: NPU1 core rows start at 2, xcve2802 at 3.
CORE_ROW_OFFSET_NPU1_TO_XCVE2802 = 1

# Default simulation cycle timeout.
DEFAULT_TIMEOUT_CYCLES = 10000

# aiesimulator process timeout in seconds (wall clock).
AIESIM_WALL_TIMEOUT_SECONDS = 300

# The device target we compile to for aiesim validation.
XCVE2802_DEVICE = "xcve2802"

# The device placeholder used in mlir-aie npu-xrt test MLIR files.
NPUDEVICE_PLACEHOLDER = "NPUDEVICE"


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------

def find_aiesimulator() -> str:
    """Discover the aiesimulator binary path.

    Search order:
      1. AIETOOLS_DIR/bin/aiesimulator (our standard env var)
      2. XILINX_VITIS_AIETOOLS/bin/aiesimulator
      3. PATH lookup via shutil.which()

    Returns the absolute path, or exits with an error if not found.
    """
    # Strategy 1: AIETOOLS_DIR
    aietools_dir = os.environ.get("AIETOOLS_DIR")
    if aietools_dir:
        candidate = os.path.join(aietools_dir, "bin", "aiesimulator")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)

    # Strategy 2: XILINX_VITIS_AIETOOLS
    vitis_aietools = os.environ.get("XILINX_VITIS_AIETOOLS")
    if vitis_aietools:
        candidate = os.path.join(vitis_aietools, "bin", "aiesimulator")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)

    # Strategy 3: PATH
    which_result = shutil.which("aiesimulator")
    if which_result:
        return os.path.abspath(which_result)

    return ""


def find_aiecc() -> str:
    """Find the aiecc.py compiler driver.

    Checks PATH via shutil.which(). Returns empty string if not found.
    """
    result = shutil.which("aiecc.py")
    if result:
        return os.path.abspath(result)
    return ""


def find_vcd_compare() -> str:
    """Find the vcd_compare (vcd-compare) binary.

    Prefers release build, falls back to debug.
    """
    script_dir = Path(__file__).resolve().parent
    emu_root = script_dir.parent

    for profile in ("release", "debug"):
        candidate = emu_root / "target" / profile / "vcd_compare"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    # Fall back to PATH.
    result = shutil.which("vcd-compare")
    if result:
        return os.path.abspath(result)
    return ""


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def find_mlir_source(test_dir: Path) -> Path:
    """Locate the MLIR source file in a test directory.

    Looks for aie.mlir first, then any .mlir file. Exits with error if
    nothing is found.
    """
    # Prefer aie.mlir (standard name in npu-xrt tests).
    aie_mlir = test_dir / "aie.mlir"
    if aie_mlir.is_file():
        return aie_mlir

    # Fall back to any .mlir file.
    mlir_files = sorted(test_dir.glob("*.mlir"))
    if mlir_files:
        return mlir_files[0]

    print(f"ERROR: No .mlir source file found in {test_dir}", file=sys.stderr)
    sys.exit(1)


def prepare_mlir_for_xcve2802(mlir_source: Path, work_dir: Path) -> Path:
    """Prepare an MLIR file for xcve2802 compilation.

    Copies the source MLIR and replaces the device target:
      - NPUDEVICE (placeholder) -> xcve2802
      - npu1_1col / npu1_Ncol  -> xcve2802
      - npu1                    -> xcve2802

    NOTE on tile coordinates: NPU1 and xcve2802 do NOT share a row layout.
    NPU1 is shim(row0) / memtile(row1) / cores(rows 2-5); xcve2802 is
    shim(row0) / memtiles(rows 1-2) / cores(rows 3-10). So an NPU1 source with a
    core at row 2 (any kernel that uses a compute tile -- i.e. nearly all of
    them) places `aie.core` on what is a MEMTILE in xcve2802, and aiecc fails
    verification with "'aie.core' op failed to verify that op exists in a core
    tile". This prep does NOT remap rows; it only swaps the device string, so it
    is limited to kernels whose explicit tile rows happen to be valid on
    xcve2802. We detect the row-2-core collision below and fail early with a
    clear message rather than the cryptic aiecc error. For memtile/compute
    kernels, use the unified driver path (XDNA_BACKEND=aiesim through the XRT
    plugin), which runs natively on NPU1 geometry.

    Returns the path to the prepared MLIR file.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    dest = work_dir / "aie_xcve2802.mlir"

    content = mlir_source.read_text()

    # Guard: NPU1 compute tiles live at rows >= 2, which collide with xcve2802
    # memtiles (rows 1-2). Without a row remap (not done here), aiecc fails an
    # obscure verifier check. Detect and explain instead.
    core_rows = {int(r) for r in re.findall(r'aie\.tile\(\s*\d+\s*,\s*(\d+)\s*\)', content)}
    if any(r >= 2 for r in core_rows):
        print(
            "ERROR: this kernel declares a compute tile at row >= 2 "
            f"(rows seen: {sorted(core_rows)}).",
            file=sys.stderr,
        )
        print(
            "  NPU1 cores (rows 2-5) map to xcve2802 memtiles (rows 1-2) without a\n"
            "  row remap, so aiecc rejects 'aie.core' on a memtile. aiesim-validate's\n"
            "  device-string swap can't handle compute/memtile kernels.\n"
            "  Use the unified driver path instead: run through the XRT plugin with\n"
            "  XDNA_BACKEND=aiesim (native NPU1 geometry, no remap).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Replace device target. Order matters: match longer patterns first.
    content = re.sub(
        r'aie\.device\(\s*NPUDEVICE\s*\)',
        f'aie.device({XCVE2802_DEVICE})',
        content,
    )
    content = re.sub(
        r'aie\.device\(\s*npu1_\d+col\s*\)',
        f'aie.device({XCVE2802_DEVICE})',
        content,
    )
    content = re.sub(
        r'aie\.device\(\s*npu1\s*\)',
        f'aie.device({XCVE2802_DEVICE})',
        content,
    )

    # Strip runtime_sequence and shim_dma_allocation -- these are NPU-specific
    # host instructions that do not apply to xcve2802 aiesim.  aiesimulator
    # uses ps.so (compiled from aie_inc.cpp) to drive shim DMA, not the NPU
    # instruction stream.  Leaving them in causes aie-translate failures
    # because xcve2802 does not support the AIEX NPU dialect operations.
    #
    # We remove:
    #   - aie.shim_dma_allocation lines
    #   - aie.runtime_sequence { ... } blocks (multi-line, brace-matched)

    # Remove shim_dma_allocation lines.
    content = re.sub(
        r'^\s*aie\.shim_dma_allocation\b[^\n]*\n',
        '',
        content,
        flags=re.MULTILINE,
    )

    # Remove runtime_sequence blocks. These are brace-delimited and may span
    # many lines. Use a simple brace counter rather than a regex for
    # robustness.
    content = _remove_runtime_sequence_blocks(content)

    dest.write_text(content)
    return dest


def _remove_runtime_sequence_blocks(content: str) -> str:
    """Remove all aie.runtime_sequence { ... } blocks from MLIR text.

    Uses brace counting to handle nested braces correctly.
    """
    result = []
    i = 0
    marker = "aie.runtime_sequence"

    while i < len(content):
        idx = content.find(marker, i)
        if idx == -1:
            result.append(content[i:])
            break

        # Find the start of this statement -- back up to the start of the line
        # to capture any leading whitespace.
        line_start = content.rfind('\n', 0, idx)
        line_start = line_start + 1 if line_start != -1 else 0

        result.append(content[i:line_start])

        # Find the opening brace.
        brace_start = content.find('{', idx)
        if brace_start == -1:
            # Malformed -- just skip the line.
            next_nl = content.find('\n', idx)
            i = next_nl + 1 if next_nl != -1 else len(content)
            continue

        # Count braces to find the matching close.
        depth = 0
        j = brace_start
        while j < len(content):
            if content[j] == '{':
                depth += 1
            elif content[j] == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1

        # Skip past the closing brace and any trailing newline.
        j += 1
        if j < len(content) and content[j] == '\n':
            j += 1

        i = j

    return ''.join(result)


def compile_for_xcve2802(test_dir: Path, output_dir: Path,
                         verbose: bool = False) -> Path:
    """Compile a test for xcve2802 to produce an aiesim package.

    Generates a .prj directory with a sim/ subdirectory that aiesimulator
    can consume via --pkg-dir.

    The compilation uses aiecc.py with --aiesim --xchesscc --xbridge flags,
    targeting xcve2802 instead of the test's original NPU device.

    Returns the path to the .prj directory (containing sim/).
    """
    aiecc = find_aiecc()
    if not aiecc:
        print("ERROR: aiecc.py not found in PATH.", file=sys.stderr)
        print("  Activate the mlir-aie environment first:", file=sys.stderr)
        print("    source toolchain-build/activate-npu-env.sh", file=sys.stderr)
        sys.exit(1)

    mlir_source = find_mlir_source(test_dir)
    print(f"  Source MLIR: {mlir_source}")

    # Prepare the MLIR for xcve2802.
    work_dir = output_dir / "compile"
    prepared_mlir = prepare_mlir_for_xcve2802(mlir_source, work_dir)
    print(f"  Prepared MLIR: {prepared_mlir}")

    # The .prj directory name is derived from the MLIR filename.
    prj_name = prepared_mlir.stem + ".prj"
    prj_dir = work_dir / prj_name

    # Build the aiecc.py command.
    # --aiesim generates the sim/ directory inside the .prj
    # --xchesscc uses the Chess compiler (required for aiesim)
    # --xbridge uses xbridge for linking (required by --aiesim)
    # --no-compile-host skips host compilation
    # --no-aie-generate-xclbin skips xclbin (we only need sim artifacts)
    # --no-aie-generate-npu-insts skips NPU instruction generation
    cmd = [
        aiecc,
        "--aiesim",
        "--xchesscc",
        "--xbridge",
        "--no-compile-host",
        str(prepared_mlir),
    ]

    print(f"  Running: {' '.join(cmd)}")
    t0 = time.monotonic()

    result = subprocess.run(
        cmd,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=600,
    )

    elapsed = time.monotonic() - t0

    # Save logs.
    (work_dir / "aiecc.stdout.log").write_text(result.stdout)
    (work_dir / "aiecc.stderr.log").write_text(result.stderr)

    if result.returncode != 0:
        print(f"ERROR: aiecc.py failed (exit {result.returncode}, "
              f"{elapsed:.1f}s)", file=sys.stderr)
        print("  stdout: " + str(work_dir / "aiecc.stdout.log"),
              file=sys.stderr)
        print("  stderr: " + str(work_dir / "aiecc.stderr.log"),
              file=sys.stderr)
        # Print last 20 lines of stderr for quick diagnosis.
        stderr_lines = result.stderr.strip().splitlines()
        for line in stderr_lines[-20:]:
            print(f"    {line}", file=sys.stderr)
        sys.exit(1)

    print(f"  Compilation succeeded ({elapsed:.1f}s)")

    # Find the .prj directory. aiecc may create it with a different name
    # than we predicted. Search for any .prj directory.
    prj_dirs = sorted(work_dir.glob("*.prj"))
    if not prj_dirs:
        print("ERROR: No .prj directory created by aiecc.py", file=sys.stderr)
        sys.exit(1)

    prj_dir = prj_dirs[0]
    sim_dir = prj_dir / "sim"
    if not sim_dir.is_dir():
        print(f"ERROR: {prj_dir} has no sim/ subdirectory", file=sys.stderr)
        print("  aiecc.py may not have run with --aiesim successfully.",
              file=sys.stderr)
        sys.exit(1)

    print(f"  Sim package: {sim_dir}")
    return prj_dir


# ---------------------------------------------------------------------------
# aiesimulator execution
# ---------------------------------------------------------------------------

def run_aiesim(pkg_dir: Path, output_dir: Path,
               timeout_cycles: int = DEFAULT_TIMEOUT_CYCLES,
               verbose: bool = False) -> dict:
    """Run aiesimulator on a compiled sim package.

    Launches aiesimulator as a subprocess with --dump-vcd and a cycle
    timeout. The simulator writes output into an aiesimulator_output/
    directory relative to the working directory.

    Args:
        pkg_dir: Path to the sim/ directory inside a .prj package.
        output_dir: Directory for logs and output.
        timeout_cycles: Simulation cycle limit (--simulation-cycle-timeout).
        verbose: Print aiesimulator stdout/stderr in real time.

    Returns:
        Dict with keys: vcd_path (Path or None), exit_code (int),
        duration_seconds (float), stdout (str), stderr (str).
    """
    aiesim_bin = find_aiesimulator()
    if not aiesim_bin:
        print("ERROR: aiesimulator not found.", file=sys.stderr)
        print("  Set AIETOOLS_DIR or XILINX_VITIS_AIETOOLS, or add "
              "aiesimulator to PATH.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # aiesimulator expects --pkg-dir to point to the sim/ directory inside
    # the .prj package. If the caller passed the .prj directory, adjust.
    sim_dir = pkg_dir
    if (pkg_dir / "sim").is_dir():
        sim_dir = pkg_dir / "sim"

    # Run from the output directory so aiesimulator_output/ lands there.
    run_dir = output_dir / "aiesim_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    vcd_name = "aiesim_trace"
    cmd = [
        "nice", "-n", "19",
        aiesim_bin,
        f"--pkg-dir={sim_dir}",
        f"--dump-vcd={vcd_name}",
        f"--simulation-cycle-timeout={timeout_cycles}",
    ]

    print(f"  Running: {' '.join(cmd)}")
    print(f"  Working dir: {run_dir}")
    t0 = time.monotonic()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            timeout=AIESIM_WALL_TIMEOUT_SECONDS,
        )
        exit_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - t0
        print(f"  WARNING: aiesimulator timed out after {elapsed:.1f}s "
              f"(wall-clock limit: {AIESIM_WALL_TIMEOUT_SECONDS}s)")
        exit_code = -1
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

    elapsed = time.monotonic() - t0

    # Save logs.
    (output_dir / "aiesim.stdout.log").write_text(stdout)
    (output_dir / "aiesim.stderr.log").write_text(stderr)

    # Find the VCD file.
    vcd_path = None
    sim_output_dir = run_dir / "aiesimulator_output"
    search_dirs = [sim_output_dir, run_dir]

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        vcd_files = sorted(search_dir.glob("*.vcd"))
        if vcd_files:
            vcd_path = vcd_files[0]
            break

    return {
        "vcd_path": vcd_path,
        "exit_code": exit_code,
        "duration_seconds": elapsed,
        "stdout": stdout,
        "stderr": stderr,
    }


# ---------------------------------------------------------------------------
# Emulator execution (stub)
# ---------------------------------------------------------------------------

def run_emulator(test_dir: Path, output_dir: Path,
                 verbose: bool = False) -> dict:
    """Run the xdna-emu emulator on a test kernel and produce VCD output.

    TODO: Wire up emulator VCD recording via:
        cargo run --release -- --vcd-record <path> <xclbin>

    This requires the emulator's VCD emission feature (behind the
    vcd-recording cargo feature flag). The emulator would load the same
    kernel, execute it, and write a VCD file with the same signal hierarchy
    that aiesimulator produces.

    For now, returns None to indicate that emulator comparison is not yet
    available.
    """
    return None


# ---------------------------------------------------------------------------
# VCD comparison
# ---------------------------------------------------------------------------

def compare_vcds(aiesim_vcd: Path, emu_vcd: Path, output_dir: Path,
                 row_offset: int = CORE_ROW_OFFSET_NPU1_TO_XCVE2802,
                 verbose: bool = False) -> dict:
    """Compare aiesimulator and emulator VCD files.

    Invokes the vcd_compare (vcd-compare) binary to perform signal-level
    comparison between the two VCDs.

    The row_offset parameter accounts for the core row difference between
    NPU1 and xcve2802: NPU1 core row N maps to xcve2802 core row N+1.
    This offset is applied when matching signal paths between the two VCDs.

    Args:
        aiesim_vcd: Path to aiesimulator VCD.
        emu_vcd: Path to emulator VCD.
        output_dir: Directory for comparison report output.
        row_offset: Core row offset (default 1 for NPU1 vs xcve2802).
        verbose: Print detailed comparison output.

    Returns:
        Dict with keys: report_path (Path or None), match_count (int),
        mismatch_count (int), exit_code (int).
    """
    vcd_compare = find_vcd_compare()
    if not vcd_compare:
        print("  WARNING: vcd_compare binary not found. "
              "Build with: cargo build --release --bin vcd_compare",
              file=sys.stderr)
        return {
            "report_path": None,
            "match_count": 0,
            "mismatch_count": 0,
            "exit_code": -1,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "vcd_comparison_report.txt"

    # The vcd_compare binary takes --emu and --sim flags.
    cmd = [
        vcd_compare,
        "--emu", str(emu_vcd),
        "--sim", str(aiesim_vcd),
        "-o", str(report_path),
    ]

    print(f"  Running: {' '.join(cmd)}")
    t0 = time.monotonic()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )

    elapsed = time.monotonic() - t0
    print(f"  Comparison completed ({elapsed:.1f}s, exit {result.returncode})")

    # Save raw output.
    (output_dir / "vcd_compare.stdout.log").write_text(result.stdout)
    (output_dir / "vcd_compare.stderr.log").write_text(result.stderr)

    if result.returncode != 0 and verbose:
        stderr_lines = result.stderr.strip().splitlines()
        for line in stderr_lines[-10:]:
            print(f"    {line}", file=sys.stderr)

    # TODO: Parse match/mismatch counts from the report output once the
    # row_offset remapping is implemented in vcd_compare.
    return {
        "report_path": report_path if report_path.is_file() else None,
        "match_count": 0,
        "mismatch_count": 0,
        "exit_code": result.returncode,
    }


def run_coverage_audit(vcd_path: Path, output_dir: Path,
                       verbose: bool = False) -> dict:
    """Run a VCD coverage audit on a single VCD file.

    Uses vcd_compare --coverage to walk every signal and report which ones
    map to known hardware state paths.

    Returns:
        Dict with keys: report_path (Path or None), exit_code (int),
        summary (str).
    """
    vcd_compare = find_vcd_compare()
    if not vcd_compare:
        print("  WARNING: vcd_compare binary not found.", file=sys.stderr)
        return {"report_path": None, "exit_code": -1, "summary": ""}

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "coverage_report.txt"

    cmd = [vcd_compare, "--coverage", str(vcd_path)]

    print(f"  Running: {' '.join(cmd)}")
    t0 = time.monotonic()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )

    elapsed = time.monotonic() - t0

    report_path.write_text(result.stdout + result.stderr)

    # Extract summary line.
    summary = ""
    for line in result.stdout.splitlines():
        if "signal" in line.lower():
            summary = line.strip()
            break

    return {
        "report_path": report_path,
        "exit_code": result.returncode,
        "summary": summary or f"exit {result.returncode} ({elapsed:.1f}s)",
    }


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate xdna-emu against aiesimulator using VCD "
                    "comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "test_dir",
        nargs="?",
        type=Path,
        help="Path to an mlir-aie test directory (containing aie.mlir). "
             "Required unless --pkg-dir is given.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for all artifacts (logs, VCDs, reports).",
    )
    parser.add_argument(
        "--pkg-dir",
        type=Path,
        default=None,
        help="Skip compilation; use an existing .prj or sim/ directory.",
    )
    parser.add_argument(
        "--aiesim-only",
        action="store_true",
        help="Only run aiesimulator (skip emulator and comparison).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip compilation. Requires --pkg-dir or auto-detects .prj "
             "in the output directory.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_CYCLES,
        help=f"Simulation cycle timeout "
             f"(default: {DEFAULT_TIMEOUT_CYCLES}).",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run VCD coverage audit after aiesim completes.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output from subprocesses.",
    )

    args = parser.parse_args()

    # Validate argument combinations.
    if args.pkg_dir is None and not args.no_compile and args.test_dir is None:
        parser.error("Either provide a test directory (positional argument) "
                     "or use --pkg-dir / --no-compile.")

    return args


def main():
    """Orchestrate the aiesim validation pipeline."""
    args = parse_args()

    print("=" * 72)
    print("aiesim-validate: xdna-emu accuracy validation via aiesimulator")
    print("=" * 72)

    # Gate check: verify aiesimulator is available.
    aiesim_bin = find_aiesimulator()
    if not aiesim_bin:
        print("ERROR: aiesimulator not found.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Set one of these environment variables:", file=sys.stderr)
        print("  AIETOOLS_DIR=/path/to/aietools", file=sys.stderr)
        print("  XILINX_VITIS_AIETOOLS=/path/to/aietools", file=sys.stderr)
        print("Or add aietools/bin to your PATH.", file=sys.stderr)
        sys.exit(1)
    print(f"aiesimulator: {aiesim_bin}")

    # Create output directory. Resolve to absolute FIRST: compile_for_xcve2802
    # runs aiecc.py with cwd=<output>/compile, so any path derived from a
    # *relative* --output (e.g. the prepared-MLIR input passed to aiecc) would
    # not resolve from that cwd -- aiecc fails with a confusing "could not open
    # input file" even though the file exists. Absolutizing here fixes every
    # downstream path in one place.
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output}")
    print()

    # ---- Phase 1: Compilation ----

    prj_dir = None

    if args.pkg_dir is not None:
        # User supplied an existing sim package.
        prj_dir = args.pkg_dir.resolve()
        if not prj_dir.is_dir():
            print(f"ERROR: --pkg-dir does not exist: {prj_dir}",
                  file=sys.stderr)
            sys.exit(1)
        # If they pointed at the .prj, check for sim/ inside.
        if (prj_dir / "sim").is_dir():
            print(f"Phase 1: SKIP (using existing package: {prj_dir})")
        elif prj_dir.name == "sim":
            # They pointed directly at the sim/ directory; go up one level.
            prj_dir = prj_dir.parent
            print(f"Phase 1: SKIP (using existing package: {prj_dir})")
        else:
            print(f"Phase 1: SKIP (using {prj_dir}, sim/ not found -- "
                  f"aiesimulator may fail)")
    elif args.no_compile:
        # Auto-detect .prj in output directory.
        prj_dirs = sorted(args.output.glob("compile/*.prj"))
        if prj_dirs:
            prj_dir = prj_dirs[0]
            print(f"Phase 1: SKIP (auto-detected: {prj_dir})")
        else:
            print("ERROR: --no-compile but no .prj found in "
                  f"{args.output / 'compile'}", file=sys.stderr)
            sys.exit(1)
    else:
        # Compile from source.
        print(f"Phase 1: Compiling {args.test_dir.name} for {XCVE2802_DEVICE}")
        prj_dir = compile_for_xcve2802(
            args.test_dir.resolve(),
            args.output,
            verbose=args.verbose,
        )

    print()

    # ---- Phase 2: Run aiesimulator ----

    print(f"Phase 2: Running aiesimulator (timeout={args.timeout} cycles)")
    sim_result = run_aiesim(
        prj_dir,
        args.output,
        timeout_cycles=args.timeout,
        verbose=args.verbose,
    )

    if sim_result["exit_code"] != 0:
        print(f"  aiesimulator FAILED (exit {sim_result['exit_code']}, "
              f"{sim_result['duration_seconds']:.1f}s)")
        # Print last few lines of stderr for diagnosis.
        stderr_lines = sim_result["stderr"].strip().splitlines()
        for line in stderr_lines[-10:]:
            print(f"    {line}")
        sys.exit(1)

    print(f"  aiesimulator completed ({sim_result['duration_seconds']:.1f}s)")

    if sim_result["vcd_path"]:
        vcd_size = sim_result["vcd_path"].stat().st_size
        vcd_mb = vcd_size / (1024 * 1024)
        print(f"  VCD output: {sim_result['vcd_path']} ({vcd_mb:.1f} MB)")
    else:
        print("  WARNING: No VCD file produced by aiesimulator.")
    print()

    # ---- Phase 2b: Coverage audit (optional) ----

    if args.coverage and sim_result["vcd_path"]:
        print("Phase 2b: VCD coverage audit")
        cov = run_coverage_audit(
            sim_result["vcd_path"],
            args.output / "coverage",
            verbose=args.verbose,
        )
        if cov["summary"]:
            print(f"  {cov['summary']}")
        if cov["report_path"]:
            print(f"  Report: {cov['report_path']}")
        print()

    # ---- Phase 3: Run emulator ----

    if args.aiesim_only:
        print("Phase 3: SKIP (--aiesim-only)")
        print()
        print("=" * 72)
        print("DONE (aiesim only)")
        if sim_result["vcd_path"]:
            print(f"  VCD: {sim_result['vcd_path']}")
        print("=" * 72)
        return

    print("Phase 3: Running emulator")
    emu_result = run_emulator(
        args.test_dir if args.test_dir else prj_dir,
        args.output,
        verbose=args.verbose,
    )

    if emu_result is None:
        print("  SKIP -- emulator VCD recording not yet implemented.")
        print("  TODO: Wire up via cargo run -- --vcd-record")
        print()
        print("=" * 72)
        print("DONE (emulator comparison not yet available)")
        if sim_result["vcd_path"]:
            print(f"  aiesim VCD: {sim_result['vcd_path']}")
        print("=" * 72)
        return

    print()

    # ---- Phase 4: Compare VCDs ----

    print("Phase 4: Comparing VCDs")
    if sim_result["vcd_path"] and emu_result and emu_result.get("vcd_path"):
        cmp_result = compare_vcds(
            sim_result["vcd_path"],
            emu_result["vcd_path"],
            args.output / "comparison",
            verbose=args.verbose,
        )

        if cmp_result["report_path"]:
            print(f"  Report: {cmp_result['report_path']}")
        if cmp_result["exit_code"] == 0:
            print("  PASS")
        else:
            print(f"  FAIL (exit {cmp_result['exit_code']})")
    else:
        print("  SKIP -- missing VCD file(s)")

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
