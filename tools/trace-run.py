#!/usr/bin/env python3
"""Execute a trace-enabled xclbin on the NPU and collect traces.

Reads a manifest.json (produced by trace-inject.py) describing the test
layout, allocates XRT buffer objects (instructions, data, trace), runs
the kernel, and writes output data + raw trace binary + Perfetto JSON.

Usage:
    trace-run.py <manifest.json> --output-dir <dir>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_manifest(manifest_path: Path) -> dict:
    """Load and validate a trace manifest JSON file."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest.get("skipped"):
        print(f"Test {manifest.get('test_name', '?')} was skipped: "
              f"{manifest.get('reason', 'unknown')}")
        sys.exit(0)

    required = ["test_name", "xclbin", "insts", "trace_size",
                 "trace_ddr_id", "kernel_name"]
    for key in required:
        if key not in manifest:
            print(f"Error: manifest missing required key '{key}'",
                  file=sys.stderr)
            sys.exit(1)

    return manifest


def generate_input_data(buf_spec: dict) -> np.ndarray:
    """Generate input data for a buffer based on its spec.

    Uses a deterministic RNG seeded on buffer name for reproducibility.
    """
    dtype_map = {
        "int8": np.int8, "uint8": np.uint8,
        "int16": np.int16, "uint16": np.uint16,
        "int32": np.int32, "uint32": np.uint32,
        "int64": np.int64,
        "float16": np.float16, "float32": np.float32,
    }

    np_dtype = dtype_map.get(buf_spec["dtype"], np.int32)
    count = buf_spec.get("elements", buf_spec["size_bytes"] // np.dtype(np_dtype).itemsize)

    pattern = buf_spec.get("pattern", "incrementing")
    if pattern == "incrementing":
        if np.issubdtype(np_dtype, np.integer):
            return np.arange(count, dtype=np_dtype)
        else:
            return np.arange(count, dtype=np.float32).astype(np_dtype)
    elif pattern == "zeros":
        return np.zeros(count, dtype=np_dtype)
    elif pattern == "ones":
        return np.ones(count, dtype=np_dtype)
    else:
        # Random with deterministic seed
        rng = np.random.default_rng(seed=hash(buf_spec.get("name", "")) & 0xFFFFFFFF)
        if np.issubdtype(np_dtype, np.integer):
            info = np.iinfo(np_dtype)
            # Use a modest range to avoid overflow in computations
            return rng.integers(0, min(100, info.max), size=count, dtype=np_dtype)
        else:
            return rng.random(count).astype(np_dtype)


def run_on_npu(manifest_dir: Path, manifest: dict, output_dir: Path) -> bool:
    """Execute the kernel on the NPU and collect trace data.

    Returns True on success, False on failure.
    """
    try:
        import pyxrt  # type: ignore
    except ImportError:
        print("Error: pyxrt not available. Install XRT Python bindings.",
              file=sys.stderr)
        print("  Ensure /opt/xilinx/xrt is installed and PYTHONPATH includes "
              "the xrt Python package.", file=sys.stderr)
        return False

    xclbin_path = manifest_dir / manifest["xclbin"]
    insts_path = manifest_dir / manifest["insts"]
    kernel_name = manifest["kernel_name"]
    trace_size = manifest["trace_size"]
    trace_ddr_id = manifest["trace_ddr_id"]
    buffers = manifest.get("buffers", [])

    if not xclbin_path.exists():
        print(f"Error: xclbin not found: {xclbin_path}", file=sys.stderr)
        return False
    if not insts_path.exists():
        print(f"Error: insts.bin not found: {insts_path}", file=sys.stderr)
        return False

    # Load instruction binary
    insts_data = np.fromfile(str(insts_path), dtype=np.uint32)

    # Open device and load xclbin
    device = pyxrt.device(0)
    xclbin = pyxrt.xclbin(str(xclbin_path))
    device.register_xclbin(xclbin)

    hw_ctx = pyxrt.hw_context(device, xclbin.get_uuid())
    kernel = pyxrt.kernel(hw_ctx, kernel_name)

    # Allocate instruction buffer (group_id 1)
    instr_bo = pyxrt.bo(
        device,
        len(insts_data) * 4,
        pyxrt.bo.cacheable,
        kernel.group_id(1),
    )
    instr_bo.write(insts_data.tobytes(), 0)
    instr_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Allocate data buffers (group_id 3+)
    data_bos = []
    input_arrays = []
    for i, buf_spec in enumerate(buffers):
        size = buf_spec["size_bytes"]
        bo = pyxrt.bo(
            device,
            size,
            pyxrt.bo.host_only,
            kernel.group_id(3 + i),
        )

        if buf_spec["direction"] == "input":
            arr = generate_input_data(buf_spec)
            bo.write(arr.tobytes(), 0)
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            input_arrays.append(arr)
        else:
            input_arrays.append(None)

        data_bos.append(bo)

    # Allocate trace buffer (group_id 3 + trace_ddr_id)
    trace_bo = pyxrt.bo(
        device,
        trace_size,
        pyxrt.bo.host_only,
        kernel.group_id(3 + trace_ddr_id),
    )
    # Zero the trace buffer so we can detect actual data
    trace_bo.write(bytes(trace_size), 0)
    trace_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Run the kernel
    # The XRT kernel call convention for mlir-aie:
    #   kernel(opcode, instr_bo, instr_len, *data_bos, trace_bo)
    all_bos = data_bos + [trace_bo]
    run = kernel(3, instr_bo, len(insts_data), *all_bos)
    run.wait()

    # Sync outputs from device
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, buf_spec in enumerate(buffers):
        if buf_spec["direction"] == "output":
            data_bos[i].sync(
                pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
            )
            out_data = np.frombuffer(
                data_bos[i].read(buf_spec["size_bytes"], 0),
                dtype=np.int32,  # Default; ideally use buf_spec dtype
            )
            out_path = output_dir / f"output_{buf_spec['name']}.bin"
            out_data.tofile(str(out_path))
            print(f"  Output: {out_path} ({len(out_data)} elements)")

    # Sync and save trace buffer
    trace_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    trace_raw = np.frombuffer(
        trace_bo.read(trace_size, 0),
        dtype=np.uint32,
    )

    trace_raw_path = output_dir / "trace_raw.bin"
    trace_raw.tofile(str(trace_raw_path))
    print(f"  Trace:  {trace_raw_path} ({len(trace_raw)} words)")

    # Check if trace buffer has any data
    nonzero = np.count_nonzero(trace_raw)
    if nonzero == 0:
        print("  Warning: trace buffer is all zeros (no trace data captured)")

    # Parse trace to Perfetto JSON
    mlir_path = manifest_dir / manifest.get("source_mlir", "aie_traced.mlir")
    if mlir_path.exists():
        try:
            from aie.utils.trace import parse_trace  # type: ignore

            mlir_str = mlir_path.read_text()
            trace_events = parse_trace(trace_raw, mlir_str)

            trace_json_path = output_dir / "trace.json"
            with open(trace_json_path, "w") as f:
                json.dump(trace_events, f, indent=2)
            print(f"  JSON:   {trace_json_path} ({len(trace_events)} events)")

        except Exception as e:
            print(f"  Warning: trace parsing failed: {e}", file=sys.stderr)
    else:
        print(f"  Warning: MLIR not found at {mlir_path}, skipping trace parse")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Execute trace-enabled xclbin on NPU and collect traces",
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to manifest.json from trace-inject.py",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=True,
        help="Output directory for trace data",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = load_manifest(manifest_path)
    manifest_dir = manifest_path.parent

    print(f"Running {manifest['test_name']} on NPU...")

    success = run_on_npu(manifest_dir, manifest, args.output_dir.resolve())

    if success:
        print(f"Trace collection complete for {manifest['test_name']}")
    else:
        print(f"Failed to collect trace for {manifest['test_name']}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
