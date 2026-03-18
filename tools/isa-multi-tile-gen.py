#!/usr/bin/env python3
"""Multi-tile MLIR generator for ISA validation harness.

Assigns instruction test batches to NPU tiles (up to 4 per phase, one per
column) and generates one aie.mlir per phase with per-tile objectfifos,
link_with references, and DMA configuration for parallel execution.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def assign_phases(batches: list[dict], tiles_per_phase: int = 4) -> list[list[dict]]:
    """Group batches into phases, each using up to tiles_per_phase tiles.

    Simple chunking -- first N batches go to phase 0, next N to phase 1, etc.
    """
    phases = []
    for i in range(0, len(batches), tiles_per_phase):
        phases.append(batches[i : i + tiles_per_phase])
    return phases


def compute_phase_buffer_layout(batches: list[dict]) -> dict:
    """Compute buffer offsets for a single phase.

    Each batch has in_size and out_size in bytes.  We pack them contiguously
    into shared input and output buffers, tracking both byte and i32-element
    offsets.

    Returns dict with:
        in_offsets_elems  -- list of i32-element offsets into input buffer
        out_offsets_elems -- list of i32-element offsets into output buffer
        in_offsets_bytes  -- list of byte offsets
        out_offsets_bytes -- list of byte offsets
        total_in_elems, total_out_elems  -- total sizes in i32 elements
        total_in_bytes, total_out_bytes  -- total sizes in bytes
    """
    in_off_bytes = []
    out_off_bytes = []
    in_off_elems = []
    out_off_elems = []

    cur_in_bytes = 0
    cur_out_bytes = 0

    for b in batches:
        in_sz = b["in_size"]
        out_sz = b["out_size"]
        # Element counts (i32), minimum 1
        in_elems = max(1, in_sz // 4)
        out_elems = max(1, out_sz // 4)

        in_off_bytes.append(cur_in_bytes)
        out_off_bytes.append(cur_out_bytes)
        in_off_elems.append(cur_in_bytes // 4)
        out_off_elems.append(cur_out_bytes // 4)

        cur_in_bytes += in_elems * 4
        cur_out_bytes += out_elems * 4

    return {
        "in_offsets_elems": in_off_elems,
        "out_offsets_elems": out_off_elems,
        "in_offsets_bytes": in_off_bytes,
        "out_offsets_bytes": out_off_bytes,
        "total_in_elems": cur_in_bytes // 4,
        "total_out_elems": cur_out_bytes // 4,
        "total_in_bytes": cur_in_bytes,
        "total_out_bytes": cur_out_bytes,
    }


def _obj_filename(batch: dict) -> str:
    """Derive .o filename from batch filename (replace .s or .ll with .o)."""
    name = batch.get("filename", batch.get("name", "unknown"))
    base = re.sub(r"\.(s|ll)$", "", name)
    if not base.endswith(".o"):
        base += ".o"
    return base


def generate_phase_mlir(batches: list[dict], phase_idx: int) -> str:
    """Generate MLIR for one phase with up to 4 tiles.

    Each batch occupies one column (0..n-1).  The MLIR uses objectfifos for
    data movement and link_with for linking pre-compiled .o files.

    Args:
        batches: list of batch dicts with at least in_size, out_size, filename.
        phase_idx: phase index (for comments only).

    Returns:
        Complete MLIR module as a string.
    """
    n_tiles = len(batches)
    if n_tiles == 0:
        raise ValueError("generate_phase_mlir called with empty batch list")
    if n_tiles > 4:
        raise ValueError(f"At most 4 tiles per phase, got {n_tiles}")

    # Device target: npu1 for 4 columns, npu1_Ncol for fewer
    if n_tiles == 4:
        device = "npu1"
    else:
        device = f"npu1_{n_tiles}col"

    layout = compute_phase_buffer_layout(batches)
    total_in = layout["total_in_elems"]
    total_out = layout["total_out_elems"]

    lines = []
    lines.append(f"// Phase {phase_idx}: {n_tiles} tile(s)")
    lines.append("module {")
    lines.append(f"  aie.device({device}) {{")

    # -- Tile declarations --
    for col in range(n_tiles):
        lines.append(f"    %tile_{col}_0 = aie.tile({col}, 0)")
    for col in range(n_tiles):
        lines.append(f"    %tile_{col}_2 = aie.tile({col}, 2)")
    lines.append("")

    # -- Objectfifos and function declarations per tile --
    for col, batch in enumerate(batches):
        in_elems = max(1, batch["in_size"] // 4)
        out_elems = max(1, batch["out_size"] // 4)
        obj_file = _obj_filename(batch)

        lines.append(
            f"    aie.objectfifo @of_in_{col}(%tile_{col}_0, "
            f"{{%tile_{col}_2}}, 2 : i32) "
            f": !aie.objectfifo<memref<{in_elems}xi32>>"
        )
        lines.append(
            f"    aie.objectfifo @of_out_{col}(%tile_{col}_2, "
            f"{{%tile_{col}_0}}, 2 : i32) "
            f": !aie.objectfifo<memref<{out_elems}xi32>>"
        )
        lines.append("")
        lines.append(
            f'    func.func private @test_kernel_{col}'
            f"(memref<{in_elems}xi32>, memref<{out_elems}xi32>) "
            f'attributes {{link_with = "{obj_file}"}}'
        )
        lines.append("")

    # -- Core blocks per tile --
    for col, batch in enumerate(batches):
        in_elems = max(1, batch["in_size"] // 4)
        out_elems = max(1, batch["out_size"] // 4)

        lines.append(f"    aie.core(%tile_{col}_2) {{")
        lines.append(
            f"      %sub_in = aie.objectfifo.acquire @of_in_{col}(Consume, 1) "
            f": !aie.objectfifosubview<memref<{in_elems}xi32>>"
        )
        lines.append(
            f"      %elem_in = aie.objectfifo.subview.access %sub_in[0] "
            f": !aie.objectfifosubview<memref<{in_elems}xi32>> -> memref<{in_elems}xi32>"
        )
        lines.append(
            f"      %sub_out = aie.objectfifo.acquire @of_out_{col}(Produce, 1) "
            f": !aie.objectfifosubview<memref<{out_elems}xi32>>"
        )
        lines.append(
            f"      %elem_out = aie.objectfifo.subview.access %sub_out[0] "
            f": !aie.objectfifosubview<memref<{out_elems}xi32>> -> memref<{out_elems}xi32>"
        )
        lines.append(
            f"      func.call @test_kernel_{col}(%elem_in, %elem_out) "
            f": (memref<{in_elems}xi32>, memref<{out_elems}xi32>) -> ()"
        )
        lines.append(f"      aie.objectfifo.release @of_in_{col}(Consume, 1)")
        lines.append(f"      aie.objectfifo.release @of_out_{col}(Produce, 1)")
        lines.append("      aie.end")
        lines.append("    }")
        lines.append("")

    # -- Runtime sequence --
    lines.append(
        f"    aie.runtime_sequence("
        f"%in : memref<{total_in}xi32>, "
        f"%buf : memref<{total_in}xi32>, "
        f"%out : memref<{total_out}xi32>) {{"
    )

    # Common constants
    lines.append("      %c0 = arith.constant 0 : i64")
    lines.append("      %c1 = arith.constant 1 : i64")

    # Per-tile constants for offsets and lengths
    for col, batch in enumerate(batches):
        in_elems = max(1, batch["in_size"] // 4)
        out_elems = max(1, batch["out_size"] // 4)
        in_off = layout["in_offsets_elems"][col]
        out_off = layout["out_offsets_elems"][col]

        lines.append(
            f"      %c_in_off_{col} = arith.constant {in_off} : i64"
        )
        lines.append(
            f"      %c_in_len_{col} = arith.constant {in_elems} : i64"
        )
        lines.append(
            f"      %c_out_off_{col} = arith.constant {out_off} : i64"
        )
        lines.append(
            f"      %c_out_len_{col} = arith.constant {out_elems} : i64"
        )

    lines.append("")
    lines.append("      // Output DMAs first (set up receive before sending)")

    # Output DMAs with issue_token
    for col in range(n_tiles):
        out_elems = max(1, batches[col]["out_size"] // 4)
        dma_id = col + n_tiles
        lines.append(
            f"      aiex.npu.dma_memcpy_nd("
            f"%out[%c0, %c0, %c0, %c_out_off_{col}]"
            f"[%c1, %c1, %c1, %c_out_len_{col}]"
            f"[%c0, %c0, %c0, %c1]) "
            f"{{metadata = @of_out_{col}, id = {dma_id} : i64, "
            f"issue_token = true}} : memref<{total_out}xi32>"
        )

    lines.append("")
    lines.append("      // Input DMAs")

    # Input DMAs without issue_token
    for col in range(n_tiles):
        in_elems = max(1, batches[col]["in_size"] // 4)
        lines.append(
            f"      aiex.npu.dma_memcpy_nd("
            f"%in[%c0, %c0, %c0, %c_in_off_{col}]"
            f"[%c1, %c1, %c1, %c_in_len_{col}]"
            f"[%c0, %c0, %c0, %c1]) "
            f"{{metadata = @of_in_{col}, id = {col} : i64}} "
            f": memref<{total_in}xi32>"
        )

    lines.append("")
    lines.append("      // Wait for all outputs")

    # dma_wait per output
    for col in range(n_tiles):
        lines.append(
            f"      aiex.npu.dma_wait {{symbol = @of_out_{col}}}"
        )

    lines.append("    }")
    lines.append("  }")
    lines.append("}")
    lines.append("")  # trailing newline

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-tile MLIR for ISA validation phases."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.json containing batch definitions.",
    )
    parser.add_argument(
        "--batches",
        required=True,
        help="Comma-separated batch indices to include in this phase.",
    )
    parser.add_argument(
        "--phase-idx",
        type=int,
        default=0,
        help="Phase index (for labeling, default 0).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for aie.mlir.",
    )
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    all_batches = manifest.get("batches", manifest)
    if isinstance(all_batches, dict):
        # If manifest is a dict with a "batches" key that is also a dict,
        # convert to list.  Handle both list and dict shapes.
        all_batches = list(all_batches.values())

    indices = [int(x.strip()) for x in args.batches.split(",")]
    selected = [all_batches[i] for i in indices]

    if len(selected) > 4:
        print(
            f"Error: at most 4 batches per phase, got {len(selected)}",
            file=sys.stderr,
        )
        sys.exit(1)

    mlir = generate_phase_mlir(selected, args.phase_idx)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aie.mlir"
    out_path.write_text(mlir)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
