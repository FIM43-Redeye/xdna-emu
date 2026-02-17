#!/usr/bin/env python3
"""Extract device model parameters from mlir-aie's AIETargetModel.

Queries the mlir-aie Python bindings (aie._mlir_libs._aie) to dump
structured device configuration for all supported NPU variants.
Output is JSON to stdout, suitable for consumption by xdna-emu's Rust parser.

Usage:
    python3 tools/aie-device-dump.py [--mlir-aie-path PATH]

If --mlir-aie-path is not specified, tries ../mlir-aie/build/python relative
to this script's location (matching the standard npu-work layout).
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# AIEDevice enum from AIEAttrs.td -- maps names to integer IDs used by
# get_target_model(). Only NPU devices are included; Versal FPGAs (IDs 1-3)
# are out of scope.
#
# Naming conventions across projects (a source of much confusion):
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

    # ---------------------------------------------------------------
    # Global parameters (not per-tile)
    # ---------------------------------------------------------------
    device = {
        "device_id": device_id,
        "columns": cols,
        "rows": rows,
        "is_npu": model.is_npu(),
        "local_memory_size": model.get_local_memory_size(),
        "mem_tile_size": model.get_mem_tile_size(),
        "num_mem_tile_rows": model.get_num_mem_tile_rows(),
        "max_lock_value": model.get_max_lock_value(),
        "address_gen_granularity": model.get_address_gen_granularity(),
        "column_shift": model.get_column_shift(),
        "row_shift": model.get_row_shift(),
        "mem_base_addresses": {
            "south": model.get_mem_south_base_address(),
            "west": model.get_mem_west_base_address(),
            "north": model.get_mem_north_base_address(),
            "east": model.get_mem_east_base_address(),
        },
    }

    # ---------------------------------------------------------------
    # Per-tile data: type classification, locks, BDs
    # ---------------------------------------------------------------
    tile_map = []
    # Collect tiles by type, then pick the best representative.
    # "Best" = an interior tile (not edge column) so switchbox port
    # counts reflect maximum hardware capability, not topology limits.
    tiles_by_type = {}  # type_name -> [(col, row), ...]

    for col in range(cols):
        for row in range(rows):
            tile_type = classify_tile(model, col, row)
            tile_map.append({"col": col, "row": row, "type": tile_type})
            tiles_by_type.setdefault(tile_type, []).append((col, row))

    # Pick representative: prefer interior column (not 0 or cols-1)
    type_configs = {}
    for type_name, positions in tiles_by_type.items():
        interior = [(c, r) for c, r in positions if 0 < c < cols - 1]
        rep_col, rep_row = interior[0] if interior else positions[0]
        type_configs[type_name] = {
            "num_locks": model.get_num_locks(rep_col, rep_row),
            "num_bds": model.get_num_bds(rep_col, rep_row),
            "num_banks": model.get_num_banks(rep_col, rep_row),
            "representative": [rep_col, rep_row],
        }

    device["tile_types"] = type_configs
    device["tile_map"] = tile_map

    # ---------------------------------------------------------------
    # Stream switch port counts per tile type (by representative)
    # ---------------------------------------------------------------
    # WireBundle enum values (from AIEEnums.td):
    #   Core=0, DMA=1, FIFO=2, South=3, West=4, North=5, East=6,
    #   PLIO=7, NOC=8, Trace=9, Ctrl=10
    bundle_names = {
        0: "Core",
        1: "DMA",
        2: "FIFO",
        3: "South",
        4: "West",  # note: reversed from geography
        5: "North",
        6: "East",  # note: reversed from geography
        9: "Trace",
        10: "Ctrl",
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

        # Shim mux ports (only meaningful for shim tiles)
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


def main():
    parser = argparse.ArgumentParser(
        description="Dump mlir-aie device model parameters as JSON"
    )
    parser.add_argument(
        "--mlir-aie-path",
        default=None,
        help="Path to mlir-aie root (looks for build/python/ underneath)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Dump only this device (e.g. 'npu1'). Default: dump all.",
    )
    args = parser.parse_args()

    # Resolve the Python module path
    if args.mlir_aie_path:
        python_path = os.path.join(args.mlir_aie_path, "build", "python")
    else:
        # Default: ../mlir-aie relative to the repo root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        python_path = os.path.join(
            os.path.dirname(repo_root), "mlir-aie", "build", "python"
        )

    if not os.path.isdir(python_path):
        print(
            f"Error: mlir-aie Python path not found: {python_path}",
            file=sys.stderr,
        )
        print(
            "Use --mlir-aie-path to specify the mlir-aie root directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.path.insert(0, python_path)

    try:
        from aie._mlir_libs._aie import get_target_model
    except ImportError as e:
        print(f"Error: could not import mlir-aie Python bindings: {e}", file=sys.stderr)
        print(f"Searched in: {python_path}", file=sys.stderr)
        sys.exit(1)

    # Determine which devices to dump
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

    # Dump each device
    result = {
        "generator": "aie-device-dump.py",
        "mlir_aie_python_path": python_path,
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
    print()  # trailing newline


if __name__ == "__main__":
    main()
