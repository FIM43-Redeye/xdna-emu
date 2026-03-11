#!/usr/bin/env python3
"""Unified bridge to mlir-aie's Python API.

Single entry point for all mlir-aie queries from xdna-emu's Rust code.
Each subcommand emits JSON to stdout. Replaces aie-device-dump.py.

Subcommands:
    device-model     Topology, tile types, port counts, memory sizes
    platform-detect  Hardware detection, tool availability, feature set
    test-manifest    Scan tests/examples for target device + build feasibility
    trace-events     Export hardware event enums (CoreEvent, MemEvent, etc.)
    build-manifest   Dry-run make to discover xclbin/insts output pairs

Usage:
    python3 tools/mlir-aie-bridge.py device-model [--device NAME]
    python3 tools/mlir-aie-bridge.py platform-detect
    python3 tools/mlir-aie-bridge.py test-manifest --npu-xrt-dir PATH --examples-dir PATH
    python3 tools/mlir-aie-bridge.py trace-events [--arch aie2]
    python3 tools/mlir-aie-bridge.py build-manifest --dir PATH [--chess]

If --mlir-aie-path is not specified, tries ../mlir-aie relative to this
script's location (matching the standard npu-work layout). Looks for the
Python bindings in build/python/ and install/python/ underneath.
"""

import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# mlir-aie Python path resolution (shared across all subcommands)
# ---------------------------------------------------------------------------

def resolve_mlir_aie_python_path(mlir_aie_path=None):
    """Find and return the mlir-aie Python module directory.

    Search order:
      1. Explicit --mlir-aie-path argument (looks for build/python/ underneath)
      2. PYTHONPATH already containing aie modules (ironenv activated)
      3. ../mlir-aie/build/python relative to the repo root
      4. ../mlir-aie/install/python relative to the repo root
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    mlir_aie_root = os.path.join(os.path.dirname(repo_root), "mlir-aie")

    candidates = []

    if mlir_aie_path:
        candidates.append(os.path.join(mlir_aie_path, "build", "python"))
        candidates.append(os.path.join(mlir_aie_path, "install", "python"))

    # Check if already on PYTHONPATH (ironenv activated)
    try:
        from aie._mlir_libs._aie import get_target_model  # noqa: F401
        return None  # Already importable, no path insertion needed
    except ImportError:
        pass

    candidates.append(os.path.join(mlir_aie_root, "build", "python"))
    candidates.append(os.path.join(mlir_aie_root, "install", "python"))

    for path in candidates:
        if os.path.isdir(path):
            return path

    return None


def setup_mlir_aie(mlir_aie_path=None):
    """Insert mlir-aie Python path and verify import works.

    Returns the resolved python path (or None if already on PYTHONPATH).
    Exits with error if mlir-aie is not available.
    """
    python_path = resolve_mlir_aie_python_path(mlir_aie_path)
    if python_path is not None:
        sys.path.insert(0, python_path)

    try:
        from aie._mlir_libs._aie import get_target_model  # noqa: F401
    except ImportError as e:
        print(f"Error: could not import mlir-aie Python bindings: {e}",
              file=sys.stderr)
        if python_path:
            print(f"Searched in: {python_path}", file=sys.stderr)
        print("Use --mlir-aie-path to specify the mlir-aie root directory.",
              file=sys.stderr)
        sys.exit(1)

    return python_path


# ---------------------------------------------------------------------------
# AIEDevice enum from AIEAttrs.td -- maps names to integer IDs used by
# get_target_model(). Only NPU devices are included; Versal FPGAs (IDs 1-3)
# are out of scope.
#
# Naming conventions across projects:
#
#   mlir-aie    xdna-driver    Architecture    Hardware
#   --------    -----------    ------------    --------
#   npu1        NPU1           AIE2 (XDNA)     Phoenix / Hawk Point
#   npu2        NPU4           AIE2P (XDNA2)   Strix Point
#   (none)      NPU5           AIE2P (XDNA2)   Strix Halo
#   (none)      NPU6           AIE2P (XDNA2)   Krackan
#
# The _Ncol variants are column-count subsets of the base device, used by
# the compiler to map workloads onto fewer columns.
# ---------------------------------------------------------------------------
DEVICES = {
    "npu1":      4,
    "npu1_1col": 5,
    "npu1_2col": 6,
    "npu1_3col": 7,
    "npu2":      8,
    "npu2_1col": 9,
    "npu2_2col": 10,
    "npu2_3col": 11,
    "npu2_4col": 12,
    "npu2_5col": 13,
    "npu2_6col": 14,
    "npu2_7col": 15,
}


# ===========================================================================
# Subcommand: device-model
# ===========================================================================

def classify_tile(model, col, row):
    """Return tile type string for the given (col, row) position."""
    if model.is_core_tile(col, row):
        return "core"
    if model.is_mem_tile(col, row):
        return "mem_tile"
    if model.is_shim_noc_tile(col, row):
        return "shim_noc"
    if model.is_shim_pl_tile(col, row):
        return "shim_pl"
    return "unknown"


def dump_device(model, device_id):
    """Extract all queryable parameters from a single device model."""
    cols = model.columns()
    rows = model.rows()

    device = {
        "device_id": device_id,
        "columns": cols,
        "rows": rows,
        "is_npu": model.is_npu(),
        "local_memory_size": model.get_local_memory_size(),
        "mem_tile_size": model.get_mem_tile_size(),
        "num_mem_tile_rows": model.get_num_mem_tile_rows(),
        "column_shift": model.get_column_shift(),
        "row_shift": model.get_row_shift(),
        "mem_base_addresses": {
            "south": model.get_mem_south_base_address(),
            "west": model.get_mem_west_base_address(),
            "north": model.get_mem_north_base_address(),
            "east": model.get_mem_east_base_address(),
        },
    }

    # These methods were removed in mlir-aie after LLVM 23 update.
    # Populate them only if the API still exposes them.
    if hasattr(model, "get_max_lock_value"):
        device["max_lock_value"] = model.get_max_lock_value()
    if hasattr(model, "get_address_gen_granularity"):
        device["address_gen_granularity"] = model.get_address_gen_granularity()

    # Per-tile data: type classification, edge info.
    tile_map = []
    tiles_by_type = {}

    for col in range(cols):
        for row in range(rows):
            tile_type = classify_tile(model, col, row)
            tile_entry = {
                "col": col,
                "row": row,
                "type": tile_type,
                "is_internal": not (
                    col == 0 or col == cols - 1
                    or row == 0 or row == rows - 1
                ),
                "edges": {
                    "north": row == rows - 1,
                    "south": row == 0,
                    "east": col == cols - 1,
                    "west": col == 0,
                },
            }

            # Memory affinity for core tiles.
            # is_mem_south(src_col, src_row, dst_col, dst_row) checks if
            # tile (dst) is the south memory neighbor of tile (src).
            if tile_type == "core":
                affinity = {}
                for direction, dr, dc, method_name in [
                    ("south", -1, 0, "is_mem_south"),
                    ("west", 0, -1, "is_mem_west"),
                    ("north", 1, 0, "is_mem_north"),
                    ("east", 0, 1, "is_mem_east"),
                ]:
                    neighbor_col = col + dc
                    neighbor_row = row + dr
                    method = getattr(model, method_name, None)
                    if (method
                            and 0 <= neighbor_col < cols
                            and 0 <= neighbor_row < rows):
                        try:
                            affinity[direction] = method(
                                col, row, neighbor_col, neighbor_row
                            )
                        except (TypeError, Exception):
                            affinity[direction] = False
                    else:
                        affinity[direction] = False
                tile_entry["mem_affinity"] = affinity

            tile_map.append(tile_entry)
            tiles_by_type.setdefault(tile_type, []).append((col, row))

    # Pick representative tile per type: prefer interior column.
    type_configs = {}
    for type_name, positions in tiles_by_type.items():
        interior = [(c, r) for c, r in positions if 0 < c < cols - 1]
        rep_col, rep_row = interior[0] if interior else positions[0]

        local_mem = model.get_local_memory_size()
        if type_name == "mem_tile":
            local_mem = model.get_mem_tile_size()

        config = {
            "num_locks": model.get_num_locks(rep_col, rep_row),
            "num_bds": model.get_num_bds(rep_col, rep_row),
            "representative": [rep_col, rep_row],
        }

        # get_num_banks was removed in mlir-aie after LLVM 23 update.
        if hasattr(model, "get_num_banks"):
            num_banks = model.get_num_banks(rep_col, rep_row)
            config["num_banks"] = num_banks
            if num_banks > 0:
                config["bank_size"] = local_mem // num_banks

        # Program memory size is not queryable from the API.
        if type_name == "core":
            config["program_memory_size"] = 16384  # 16KB

        type_configs[type_name] = config

    device["tile_types"] = type_configs
    device["tile_map"] = tile_map

    # Stream switch port counts per tile type (by representative).
    # WireBundle enum values from AIEEnums.td.
    bundle_names = {
        0: "Core", 1: "DMA", 2: "FIFO", 3: "South",
        4: "West", 5: "North", 6: "East",
        9: "Trace", 10: "Ctrl",
    }

    for type_name, config in type_configs.items():
        rep_col, rep_row = config["representative"]
        switchbox_ports = {}
        for bundle_id, bundle_name in bundle_names.items():
            dest = model.get_num_dest_switchbox_connections(
                rep_col, rep_row, bundle_id
            )
            src = model.get_num_source_switchbox_connections(
                rep_col, rep_row, bundle_id
            )
            if dest > 0 or src > 0:
                switchbox_ports[bundle_name] = {
                    "master": dest,
                    "slave": src,
                }
        config["switchbox_ports"] = switchbox_ports

        # Shim mux ports (only meaningful for shim tiles).
        if "shim" in type_name:
            shim_mux_ports = {}
            for bundle_id, bundle_name in bundle_names.items():
                try:
                    dest = model.get_num_dest_shim_mux_connections(
                        rep_col, rep_row, bundle_id
                    )
                    src = model.get_num_source_shim_mux_connections(
                        rep_col, rep_row, bundle_id
                    )
                    if dest > 0 or src > 0:
                        shim_mux_ports[bundle_name] = {
                            "master": dest,
                            "slave": src,
                        }
                except Exception:
                    pass  # Not all bundles valid for shim mux
            if shim_mux_ports:
                config["shim_mux_ports"] = shim_mux_ports

    return device


def cmd_device_model(args):
    """device-model subcommand: dump device topology and parameters."""
    python_path = setup_mlir_aie(args.mlir_aie_path)
    from aie._mlir_libs._aie import get_target_model

    if args.device:
        if args.device not in DEVICES:
            print(
                f"Error: unknown device '{args.device}'. "
                f"Available: {', '.join(sorted(DEVICES.keys()))}",
                file=sys.stderr,
            )
            sys.exit(1)
        devices_to_dump = {args.device: DEVICES[args.device]}
    else:
        devices_to_dump = DEVICES

    result = {
        "generator": "mlir-aie-bridge.py device-model",
        "mlir_aie_python_path": python_path or "(on PYTHONPATH)",
        "devices": {},
    }

    for name, device_id in sorted(devices_to_dump.items(), key=lambda x: x[1]):
        try:
            model = get_target_model(device_id)
            result["devices"][name] = dump_device(model, device_id)
        except Exception as e:
            print(
                f"Warning: failed to dump device '{name}' (id={device_id}): {e}",
                file=sys.stderr,
            )

    json.dump(result, sys.stdout, indent=2)
    print()


# ===========================================================================
# Subcommand: platform-detect
# ===========================================================================

def detect_xrt():
    """Detect NPU hardware via xrt-smi, mirroring LitConfigHelper logic."""
    import re
    import shutil
    import subprocess

    xrt_smi = shutil.which("xrt-smi")
    if not xrt_smi:
        return {"xrt_found": False}

    try:
        output = subprocess.check_output(
            [xrt_smi, "examine"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode("utf-8", errors="replace")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return {"xrt_found": True, "npu_model": None}

    # Pattern from mlir-aie's lit_config_helpers.py.
    pattern = r"[\|]?(\[.+:.+:.+\]).+\|(RyzenAI-(npu\d)|NPU ([\w ]+?))\s*\|"
    match = re.search(pattern, output)
    if not match:
        return {"xrt_found": True, "npu_model": None}

    device_id = match.group(1)  # e.g. [0000:c6:00.1]
    npu_model_str = match.group(3) or match.group(4)  # e.g. "npu1" or "Phoenix"

    # NPU_MODELS mapping from lit_config_helpers.py.
    npu_models = {
        "npu1": ["npu1", "Phoenix"],
        "npu2": ["npu4", "Strix", "npu5", "Strix Halo", "npu6", "Krackan"],
    }

    npu_generation = None
    npu_model = None
    for gen, names in npu_models.items():
        if npu_model_str in names:
            npu_generation = gen
            npu_model = gen
            break

    arch_map = {"npu1": "AIE2", "npu2": "AIE2P"}

    return {
        "xrt_found": True,
        "npu_model": npu_model,
        "npu_generation": npu_model_str,
        "arch": arch_map.get(npu_model),
        "device_id": device_id,
    }


def detect_tools():
    """Detect compiler and simulator availability."""
    import shutil
    import subprocess

    tools = {}

    # Peano: find llc with AIE target registered.
    # The llc on PATH may be mlir-aie's bundled LLVM (no AIE target).
    # Check known Peano install locations first, then fall back to PATH.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    peano_candidates = [
        os.path.join(os.path.dirname(repo_root), "llvm-aie", "build", "bin", "llc"),
        os.path.join(os.path.dirname(repo_root), "llvm-aie", "install", "bin", "llc"),
    ]
    # Also check PATH, but after specific locations.
    path_llc = shutil.which("llc")
    if path_llc:
        peano_candidates.append(path_llc)

    tools["peano"] = {"found": False, "path": None}
    for llc in peano_candidates:
        if not os.path.isfile(llc):
            continue
        try:
            output = subprocess.check_output(
                [llc, "-mtriple=aie", "--version"],
                stderr=subprocess.STDOUT,
                timeout=5,
            ).decode("utf-8", errors="replace")
            if "Xilinx AI Engine" in output:
                tools["peano"] = {"found": True, "path": llc}
                break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            pass

    # Chess compiler.
    xchesscc = shutil.which("xchesscc")
    tools["chess"] = {"found": xchesscc is not None, "path": xchesscc}

    # aiesimulator.
    aiesim = shutil.which("aiesimulator")
    tools["aiesimulator"] = {"found": aiesim is not None, "path": aiesim}

    return tools


def cmd_platform_detect(args):
    """platform-detect subcommand: detect hardware and tools."""
    hardware = detect_xrt()
    tools = detect_tools()

    # Build feature list matching mlir-aie lit conventions.
    features = []
    if hardware.get("npu_model"):
        features.append("ryzen_ai")
        features.append(f"ryzen_ai_{hardware['npu_model']}")
    if tools.get("peano", {}).get("found"):
        features.append("peano")
    if tools.get("chess", {}).get("found"):
        features.append("chess")
        features.append("valid_xchess_license")
    if tools.get("aiesimulator", {}).get("found"):
        features.append("aiesimulator")

    # Check for XRT Python bindings (pyxrt).
    try:
        import pyxrt  # noqa: F401
        features.append("xrt_python_bindings")
    except ImportError:
        pass

    result = {
        "hardware": hardware,
        "tools": tools,
        "features": features,
    }

    json.dump(result, sys.stdout, indent=2)
    print()


# ===========================================================================
# Subcommand: trace-events
# ===========================================================================

def cmd_trace_events(args):
    """trace-events subcommand: export hardware event enums as JSON."""
    python_path = setup_mlir_aie(args.mlir_aie_path)

    try:
        from aie.utils.trace_events import CoreEvent, MemEvent, MemTileEvent, ShimTileEvent
    except ImportError:
        try:
            from aie.utils.trace.events.aie2 import (
                CoreEvent, MemEvent, MemTileEvent, ShimTileEvent
            )
        except ImportError as e:
            print(f"Error: could not import trace event enums: {e}",
                  file=sys.stderr)
            print("Tried: aie.utils.trace_events, aie.utils.trace.events.aie2",
                  file=sys.stderr)
            sys.exit(1)

    def dump_enum(enum_class):
        return {member.name: member.value for member in enum_class}

    result = {
        "arch": args.arch if hasattr(args, "arch") and args.arch else "aie2",
        "enums": {
            "CoreEvent": dump_enum(CoreEvent),
            "MemEvent": dump_enum(MemEvent),
            "MemTileEvent": dump_enum(MemTileEvent),
            "ShimTileEvent": dump_enum(ShimTileEvent),
        },
        "trace_registers": {
            "core": {
                "control0": "0x340D0",
                "control1": "0x340D4",
                "event_group1": "0x340E0",
                "event_group2": "0x340E4",
            },
            "mem": {
                "control0": "0x140D0",
                "control1": "0x140D4",
                "event_group1": "0x140E0",
                "event_group2": "0x140E4",
            },
            "memtile": {
                "control0": "0x940D0",
                "control1": "0x940D4",
                "event_group1": "0x940E0",
                "event_group2": "0x940E4",
            },
        },
    }

    json.dump(result, sys.stdout, indent=2)
    print()


# ===========================================================================
# Subcommand: test-manifest
# ===========================================================================

def scan_test_directory(test_dir):
    """Extract metadata from a single test directory."""
    import glob
    import re

    name = os.path.basename(test_dir)

    # Target device detection (priority order).
    target_device = "npu1"  # default

    # 1. Scan Python generators for AIEDevice usage.
    py_files = glob.glob(os.path.join(test_dir, "*.py"))
    for py_file in py_files:
        try:
            content = open(py_file).read()
            m = re.search(r"AIEDevice\.(npu\d[\w]*)", content)
            if m:
                target_device = m.group(1)
                break
        except (OSError, UnicodeDecodeError):
            pass

    # 2. Scan MLIR files for aie.device.
    if target_device == "npu1":
        mlir_files = glob.glob(os.path.join(test_dir, "*.mlir"))
        for mlir_file in mlir_files:
            try:
                content = open(mlir_file).read()
                m = re.search(r"aie\.device\((npu\d[\w]*)\)", content)
                if m:
                    target_device = m.group(1)
                    break
            except (OSError, UnicodeDecodeError):
                pass

    # 3. Check Makefile for NPU2 flag.
    if target_device == "npu1":
        makefile = os.path.join(test_dir, "Makefile")
        if os.path.isfile(makefile):
            try:
                content = open(makefile).read()
                if re.search(r"NPU2\s*[?:]?=\s*1", content):
                    target_device = "npu2"
                if re.search(r"devicename\s*[?:]?=\s*(npu\d[\w]*)", content):
                    m = re.search(
                        r"devicename\s*[?:]?=\s*(npu\d[\w]*)", content
                    )
                    if m:
                        target_device = m.group(1)
            except (OSError, UnicodeDecodeError):
                pass

    # Map device to architecture.
    arch_map = {"npu1": "AIE2", "npu2": "AIE2P"}
    target_arch = arch_map.get(
        target_device.split("_")[0],  # strip _Ncol suffix
        "AIE2",
    )

    # REQUIRES features (from Python files and run.lit).
    requires = []
    scan_files = py_files + glob.glob(os.path.join(test_dir, "run.lit"))
    for f in scan_files:
        try:
            for line in open(f):
                m = re.match(r"#\s*REQUIRES:\s*(.+)", line)
                if m:
                    requires.extend(
                        feat.strip() for feat in m.group(1).split(",")
                    )
        except (OSError, UnicodeDecodeError):
            pass

    # Compiler requirements: check for Chess-only markers.
    compilers = ["peano"]  # default
    all_files = py_files + [os.path.join(test_dir, "Makefile")]
    for f in all_files:
        if not os.path.isfile(f):
            continue
        try:
            content = open(f).read()
            if "--xchesscc" in content or "CHESS=1" in content:
                if "chess" not in compilers:
                    compilers.append("chess")
        except (OSError, UnicodeDecodeError):
            pass

    # Build feasibility.
    makefile_exists = os.path.isfile(os.path.join(test_dir, "Makefile"))
    kernel_sources = (
        glob.glob(os.path.join(test_dir, "*.cc"))
        + glob.glob(os.path.join(test_dir, "*.cpp"))
    )
    python_generators = [
        f for f in py_files if os.path.basename(f) not in ("test.py",)
    ]

    missing = []
    if not makefile_exists:
        missing.append("Makefile")
    if not kernel_sources and not python_generators:
        missing.append("kernel sources or Python generator")

    # Skip reason from XFAIL/SKIP annotations.
    skip_reason = None
    for f in scan_files:
        try:
            content = open(f).read()
            m = re.search(r"#\s*XFAIL:\s*(.+)", content)
            if m:
                skip_reason = f"XFAIL: {m.group(1).strip()}"
                break
            m = re.search(r"#\s*UNSUPPORTED:\s*(.+)", content)
            if m:
                skip_reason = f"UNSUPPORTED: {m.group(1).strip()}"
                break
        except (OSError, UnicodeDecodeError):
            pass

    return {
        "name": name,
        "path": test_dir,
        "target_device": target_device,
        "target_arch": target_arch,
        "requires": sorted(set(requires)),
        "compilers": compilers,
        "build_feasibility": {
            "makefile_exists": makefile_exists,
            "kernel_sources_exist": len(kernel_sources) > 0,
            "python_generator_exists": len(python_generators) > 0,
            "missing_dependencies": missing,
        },
        "skip_reason": skip_reason,
    }


def scan_directory_tree(base_dir):
    """Recursively find test directories (those containing a Makefile or *.py)."""
    import glob

    tests = []
    if not os.path.isdir(base_dir):
        return tests

    for root, dirs, files in os.walk(base_dir):
        # A test directory has a Makefile or Python files.
        has_makefile = "Makefile" in files
        has_python = any(f.endswith(".py") for f in files)
        if has_makefile or has_python:
            # Don't recurse into subdirectories of test dirs.
            dirs.clear()
            tests.append(scan_test_directory(root))
        # Skip hidden directories and build artifacts.
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in ("build", "__pycache__")
        ]

    return tests


def cmd_test_manifest(args):
    """test-manifest subcommand: scan tests and extract metadata."""
    from datetime import datetime

    npu_xrt_tests = []
    examples = []

    if args.npu_xrt_dir:
        npu_xrt_tests = scan_directory_tree(args.npu_xrt_dir)
    if args.examples_dir:
        examples = scan_directory_tree(args.examples_dir)

    all_tests = npu_xrt_tests + examples

    # Summary statistics.
    npu1_count = sum(1 for t in all_tests if t["target_device"].startswith("npu1"))
    npu2_count = sum(1 for t in all_tests if t["target_device"].startswith("npu2"))
    buildable = sum(
        1 for t in all_tests
        if t["build_feasibility"]["makefile_exists"]
        and not t["build_feasibility"]["missing_dependencies"]
    )

    result = {
        "scan_time": datetime.now().isoformat(),
        "npu_xrt_tests": npu_xrt_tests,
        "examples": examples,
        "summary": {
            "total": len(all_tests),
            "npu1_only": npu1_count,
            "npu2_only": npu2_count,
            "buildable": buildable,
            "not_buildable": len(all_tests) - buildable,
        },
    }

    json.dump(result, sys.stdout, indent=2)
    print()


# ===========================================================================
# Subcommand: build-manifest
# ===========================================================================

def cmd_build_manifest(args):
    """build-manifest subcommand: dry-run make to discover xclbin outputs."""
    import re
    import subprocess

    example_dir = os.path.abspath(args.dir)
    if not os.path.isdir(example_dir):
        json.dump({"dir": example_dir, "designs": []}, sys.stdout)
        print()
        return

    chess_flag = "true" if args.chess else "false"
    try:
        result = subprocess.run(
            ["make", "-nBs", "-C", example_dir, f"CHESS={chess_flag}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"Warning: make dry-run failed: {e}", file=sys.stderr)
        json.dump({"dir": example_dir, "designs": []}, sys.stdout)
        print()
        return

    # Join backslash-continuation lines before parsing
    output = output.replace("\\\n", " ")

    # Parse aiecc invocations from make dry-run output.
    # Match both the old Python wrapper (aiecc.py) and the new C++ binary
    # (aiecc) that replaced it in the LLVM 23 rewrite.
    designs = []
    for line in output.splitlines():
        if "aiecc" not in line:
            continue

        xclbin_name = None
        insts_name = None

        # Extract --xclbin-name=VALUE
        m = re.search(r"--xclbin-name[=\s]+(\S+)", line)
        if m:
            xclbin_name = m.group(1)

        # Extract --npu-insts-name=VALUE
        m = re.search(r"--npu-insts-name[=\s]+(\S+)", line)
        if m:
            insts_name = m.group(1)

        if xclbin_name:
            designs.append({
                "xclbin": xclbin_name,
                "insts": insts_name,
            })

    json.dump({"dir": example_dir, "designs": designs}, sys.stdout)
    print()


# ===========================================================================
# Main: argument parser with subcommands
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified bridge to mlir-aie Python API for xdna-emu",
    )
    parser.add_argument(
        "--mlir-aie-path",
        default=None,
        help="Path to mlir-aie root (looks for build/python/ underneath)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # device-model
    p_dm = subparsers.add_parser(
        "device-model",
        help="Dump device topology, tile types, port counts, memory sizes",
    )
    p_dm.add_argument(
        "--device", default=None,
        help="Dump only this device (e.g. 'npu1'). Default: dump all.",
    )

    # platform-detect
    subparsers.add_parser(
        "platform-detect",
        help="Detect NPU hardware, compilers, simulators",
    )

    # test-manifest
    p_tm = subparsers.add_parser(
        "test-manifest",
        help="Scan test directories for target device and build feasibility",
    )
    p_tm.add_argument(
        "--npu-xrt-dir", default=None,
        help="Path to mlir-aie/test/npu-xrt/",
    )
    p_tm.add_argument(
        "--examples-dir", default=None,
        help="Path to mlir-aie/programming_examples/",
    )

    # trace-events
    p_te = subparsers.add_parser(
        "trace-events",
        help="Export hardware event enums (CoreEvent, MemEvent, etc.)",
    )
    p_te.add_argument(
        "--arch", default="aie2",
        help="Architecture (default: aie2)",
    )

    # build-manifest
    p_bm = subparsers.add_parser(
        "build-manifest",
        help="Dry-run make to discover xclbin/insts output pairs",
    )
    p_bm.add_argument(
        "--dir", required=True,
        help="Path to the example source directory",
    )
    p_bm.add_argument(
        "--chess", action="store_true",
        help="Use Chess compiler (CHESS=true)",
    )

    args = parser.parse_args()

    if args.command == "device-model":
        cmd_device_model(args)
    elif args.command == "platform-detect":
        cmd_platform_detect(args)
    elif args.command == "test-manifest":
        cmd_test_manifest(args)
    elif args.command == "trace-events":
        cmd_trace_events(args)
    elif args.command == "build-manifest":
        cmd_build_manifest(args)


if __name__ == "__main__":
    main()
