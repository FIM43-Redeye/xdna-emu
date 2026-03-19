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
import shutil
import subprocess
import sys
from pathlib import Path

# llvm-objcopy from Peano toolchain -- handles AIE2 ELF format.
# Check both install/ (preferred) and build/ (fallback) locations.
def _find_llvm_objcopy() -> str:
    """Return path to llvm-objcopy, preferring PEANO_INSTALL_DIR if set."""
    peano_install = os.environ.get(
        "PEANO_INSTALL_DIR",
        os.path.expanduser("~/npu-work/llvm-aie/install"),
    )
    candidates = [
        os.path.join(peano_install, "bin", "llvm-objcopy"),
        os.path.expanduser("~/npu-work/llvm-aie/build/bin/llvm-objcopy"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    # Return first candidate even if missing -- caller will get a clear error.
    return candidates[0]

LLVM_OBJCOPY = _find_llvm_objcopy()


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


def _obj_filename(batch: dict, key: str = "filename") -> str:
    """Derive .o filename from batch filename (replace .s or .ll with .o).

    Args:
        batch: Batch dict.
        key: Dict key to read the source filename from (default "filename").
    """
    name = batch.get(key, batch.get("filename", batch.get("name", "unknown")))
    base = re.sub(r"\.(s|ll)$", "", name)
    if not base.endswith(".o"):
        base += ".o"
    return base


def prepare_phase_objects(
    batches: list[dict],
    phase_dir: str,
    obj_dir: str,
) -> list[str]:
    """Copy and rename batch .o files for multi-tile linking.

    Every batch .o file exports a symbol named ``test_kernel``.  The
    multi-tile MLIR declares per-tile functions as ``test_kernel_0``,
    ``test_kernel_1``, etc.  This function makes the two agree:

    1. Copies ``batch_NNN.o`` from *obj_dir* to *phase_dir* under the
       name that ``link_with`` will reference (derived from the batch's
       ``filename`` field).
    2. Runs ``llvm-objcopy --redefine-sym test_kernel=test_kernel_{col}``
       on the copy so the linker resolves the per-tile symbol.

    The column index ``col`` is the batch's position in *batches* (0-based),
    NOT the ``batch_index`` field.  This is intentional: the column assignment
    is purely a property of the phase layout.

    Args:
        batches:   Batch dicts.  Each must have ``batch_index`` and
                   ``filename`` fields.
        phase_dir: Directory to write renamed .o files into (created if
                   absent).
        obj_dir:   Directory containing the original ``batch_NNN.o`` files
                   produced by the compile step.

    Returns:
        List of renamed .o basenames in column order, matching what the
        MLIR ``link_with`` attributes expect.

    Raises:
        subprocess.CalledProcessError: if llvm-objcopy fails on any file.
        FileNotFoundError: if a source .o does not exist.
    """
    os.makedirs(phase_dir, exist_ok=True)
    result: list[str] = []
    for col, batch in enumerate(batches):
        if _is_cascade(batch):
            # Cascade pairs have two .o files (producer + consumer).
            for suffix, sym_suffix in [("producer", "prod"),
                                       ("consumer", "cons")]:
                src_name = batch.get(
                    f"{suffix}_filename", ""
                ).replace(".s", ".o")
                src = os.path.join(obj_dir, src_name)
                dst = os.path.join(phase_dir, src_name)
                shutil.copy2(src, dst)
                subprocess.run(
                    [
                        LLVM_OBJCOPY,
                        "--redefine-sym",
                        f"test_kernel=test_kernel_{col}_{sym_suffix}",
                        dst,
                    ],
                    check=True,
                )
                result.append(src_name)
        else:
            # Normal batch: single .o file.
            src = os.path.join(
                obj_dir, f"batch_{batch['batch_index']}.o"
            )
            o_name = _obj_filename(batch)
            dst = os.path.join(phase_dir, o_name)
            shutil.copy2(src, dst)
            subprocess.run(
                [
                    LLVM_OBJCOPY,
                    "--redefine-sym",
                    f"test_kernel=test_kernel_{col}",
                    dst,
                ],
                check=True,
            )
            result.append(o_name)
    return result


def _is_cascade(batch: dict) -> bool:
    """Return True if batch is a cascade_pair."""
    return batch.get("source_type") == "cascade_pair"


def generate_phase_mlir(batches: list[dict], phase_idx: int) -> str:
    """Generate MLIR for one phase with up to 4 tiles.

    Each batch occupies one column (0..n-1).  Normal (assembly) batches use a
    single compute tile at row 2.  Cascade_pair batches use two tiles in the
    same column: producer at row 3, consumer at row 2, connected by
    aie.cascade_flow.  Producer data routes through a memtile (row 1) via
    two-hop objectfifos with objectfifo.link.

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

    # Identify cascade columns
    cascade_cols = [col for col, b in enumerate(batches) if _is_cascade(b)]

    lines = []
    lines.append(f"// Phase {phase_idx}: {n_tiles} tile(s)")
    lines.append("module {")
    lines.append(f"  aie.device({device}) {{")

    # -- Tile declarations --
    # Shim tiles (row 0)
    for col in range(n_tiles):
        lines.append(f"    %tile_{col}_0 = aie.tile({col}, 0)")
    # Consumer / normal compute tiles (row 2)
    for col in range(n_tiles):
        lines.append(f"    %tile_{col}_2 = aie.tile({col}, 2)")
    # Memtiles (row 1) for cascade columns
    for col in cascade_cols:
        lines.append(f"    %tile_{col}_1 = aie.tile({col}, 1)")
    # Producer tiles (row 3) for cascade columns
    for col in cascade_cols:
        lines.append(f"    %tile_{col}_3 = aie.tile({col}, 3)")
    lines.append("")

    # -- Cascade flow declarations --
    for col in cascade_cols:
        lines.append(f"    aie.cascade_flow(%tile_{col}_3, %tile_{col}_2)")
    if cascade_cols:
        lines.append("")

    # -- Objectfifos and function declarations per tile --
    for col, batch in enumerate(batches):
        if _is_cascade(batch):
            _emit_cascade_objectfifos(lines, col, batch)
        else:
            _emit_normal_objectfifos(lines, col, batch)

    # -- Core blocks per tile --
    for col, batch in enumerate(batches):
        if _is_cascade(batch):
            _emit_cascade_cores(lines, col, batch)
        else:
            _emit_normal_core(lines, col, batch)

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

    # Per-tile constants for offsets and lengths.
    # Cascade pairs need sub-offsets for producer and consumer outputs.
    for col, batch in enumerate(batches):
        in_off = layout["in_offsets_elems"][col]
        out_off = layout["out_offsets_elems"][col]

        if _is_cascade(batch):
            prod_in_elems = max(1, batch["producer_in_size"] // 4)
            prod_out_elems = max(1, batch["producer_out_size"] // 4)
            cons_out_elems = max(1, batch["consumer_out_size"] // 4)

            lines.append(
                f"      %c_in_off_{col} = arith.constant {in_off} : i64"
            )
            lines.append(
                f"      %c_in_len_{col} = arith.constant {prod_in_elems} : i64"
            )
            # Producer output sub-region
            lines.append(
                f"      %c_prod_out_off_{col} = arith.constant {out_off} : i64"
            )
            lines.append(
                f"      %c_prod_out_len_{col} = arith.constant {prod_out_elems} : i64"
            )
            # Consumer output sub-region (follows producer output)
            cons_out_off = out_off + prod_out_elems
            lines.append(
                f"      %c_cons_out_off_{col} = arith.constant {cons_out_off} : i64"
            )
            lines.append(
                f"      %c_cons_out_len_{col} = arith.constant {cons_out_elems} : i64"
            )
        else:
            in_elems = max(1, batch["in_size"] // 4)
            out_elems = max(1, batch["out_size"] // 4)

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

    # Output DMAs with issue_token.
    # Cascade pairs emit two output DMAs (producer + consumer); normal emit one.
    dma_id = n_tiles  # start output IDs after input IDs
    for col, batch in enumerate(batches):
        if _is_cascade(batch):
            prod_out_elems = max(1, batch["producer_out_size"] // 4)
            cons_out_elems = max(1, batch["consumer_out_size"] // 4)
            # Producer output DMA (via memtile, shim-side name)
            lines.append(
                f"      aiex.npu.dma_memcpy_nd("
                f"%out[%c0, %c0, %c0, %c_prod_out_off_{col}]"
                f"[%c1, %c1, %c1, %c_prod_out_len_{col}]"
                f"[%c0, %c0, %c0, %c1]) "
                f"{{metadata = @of_prod_out_{col}_1, id = {dma_id} : i64, "
                f"issue_token = true}} : memref<{total_out}xi32>"
            )
            dma_id += 1
            # Consumer output DMA (direct row2->shim)
            lines.append(
                f"      aiex.npu.dma_memcpy_nd("
                f"%out[%c0, %c0, %c0, %c_cons_out_off_{col}]"
                f"[%c1, %c1, %c1, %c_cons_out_len_{col}]"
                f"[%c0, %c0, %c0, %c1]) "
                f"{{metadata = @of_cons_out_{col}, id = {dma_id} : i64, "
                f"issue_token = true}} : memref<{total_out}xi32>"
            )
            dma_id += 1
        else:
            out_elems = max(1, batch["out_size"] // 4)
            lines.append(
                f"      aiex.npu.dma_memcpy_nd("
                f"%out[%c0, %c0, %c0, %c_out_off_{col}]"
                f"[%c1, %c1, %c1, %c_out_len_{col}]"
                f"[%c0, %c0, %c0, %c1]) "
                f"{{metadata = @of_out_{col}, id = {dma_id} : i64, "
                f"issue_token = true}} : memref<{total_out}xi32>"
            )
            dma_id += 1

    lines.append("")
    lines.append("      // Input DMAs")

    # Input DMAs without issue_token
    input_dma_id = 0
    for col, batch in enumerate(batches):
        if _is_cascade(batch):
            # Producer input (via memtile, shim-side name)
            lines.append(
                f"      aiex.npu.dma_memcpy_nd("
                f"%in[%c0, %c0, %c0, %c_in_off_{col}]"
                f"[%c1, %c1, %c1, %c_in_len_{col}]"
                f"[%c0, %c0, %c0, %c1]) "
                f"{{metadata = @of_prod_in_{col}_0, id = {input_dma_id} : i64}} "
                f": memref<{total_in}xi32>"
            )
        else:
            lines.append(
                f"      aiex.npu.dma_memcpy_nd("
                f"%in[%c0, %c0, %c0, %c_in_off_{col}]"
                f"[%c1, %c1, %c1, %c_in_len_{col}]"
                f"[%c0, %c0, %c0, %c1]) "
                f"{{metadata = @of_in_{col}, id = {input_dma_id} : i64}} "
                f": memref<{total_in}xi32>"
            )
        input_dma_id += 1

    lines.append("")
    lines.append("      // Wait for all outputs")

    # dma_wait per output
    for col, batch in enumerate(batches):
        if _is_cascade(batch):
            lines.append(
                f"      aiex.npu.dma_wait {{symbol = @of_prod_out_{col}_1}}"
            )
            lines.append(
                f"      aiex.npu.dma_wait {{symbol = @of_cons_out_{col}}}"
            )
        else:
            lines.append(
                f"      aiex.npu.dma_wait {{symbol = @of_out_{col}}}"
            )

    lines.append("    }")
    lines.append("  }")
    lines.append("}")
    lines.append("")  # trailing newline

    return "\n".join(lines)


def _emit_normal_objectfifos(lines: list[str], col: int, batch: dict) -> None:
    """Emit objectfifos and function declaration for a normal (assembly) batch."""
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


def _emit_cascade_objectfifos(lines: list[str], col: int, batch: dict) -> None:
    """Emit objectfifos, links, and function declarations for a cascade_pair.

    Data flow:
      Producer input:  shim -> memtile -> row3 (two-hop with link)
      Producer output: row3 -> memtile -> shim (two-hop with link)
      Consumer output: row2 -> shim (direct, same as normal)
    """
    prod_in_elems = max(1, batch["producer_in_size"] // 4)
    prod_out_elems = max(1, batch["producer_out_size"] // 4)
    cons_out_elems = max(1, batch["consumer_out_size"] // 4)
    prod_obj = _obj_filename(batch, "producer_filename")
    cons_obj = _obj_filename(batch, "consumer_filename")

    # Producer input: shim(row0) -> memtile(row1) -> producer(row3)
    lines.append(
        f"    aie.objectfifo @of_prod_in_{col}_0(%tile_{col}_0, "
        f"{{%tile_{col}_1}}, 1 : i32) "
        f": !aie.objectfifo<memref<{prod_in_elems}xi32>>"
    )
    lines.append(
        f"    aie.objectfifo @of_prod_in_{col}_1(%tile_{col}_1, "
        f"{{%tile_{col}_3}}, 1 : i32) "
        f": !aie.objectfifo<memref<{prod_in_elems}xi32>>"
    )
    lines.append(
        f"    aie.objectfifo.link [@of_prod_in_{col}_0] -> "
        f"[@of_prod_in_{col}_1] ([] [])"
    )
    lines.append("")

    # Producer output: producer(row3) -> memtile(row1) -> shim(row0)
    lines.append(
        f"    aie.objectfifo @of_prod_out_{col}_0(%tile_{col}_3, "
        f"{{%tile_{col}_1}}, 1 : i32) "
        f": !aie.objectfifo<memref<{prod_out_elems}xi32>>"
    )
    lines.append(
        f"    aie.objectfifo @of_prod_out_{col}_1(%tile_{col}_1, "
        f"{{%tile_{col}_0}}, 1 : i32) "
        f": !aie.objectfifo<memref<{prod_out_elems}xi32>>"
    )
    lines.append(
        f"    aie.objectfifo.link [@of_prod_out_{col}_0] -> "
        f"[@of_prod_out_{col}_1] ([] [])"
    )
    lines.append("")

    # Consumer output: consumer(row2) -> shim(row0) (direct)
    lines.append(
        f"    aie.objectfifo @of_cons_out_{col}(%tile_{col}_2, "
        f"{{%tile_{col}_0}}, 2 : i32) "
        f": !aie.objectfifo<memref<{cons_out_elems}xi32>>"
    )
    lines.append("")

    # Function declarations -- producer takes (in, out), consumer takes (out)
    lines.append(
        f'    func.func private @test_kernel_{col}_prod'
        f"(memref<{prod_in_elems}xi32>, memref<{prod_out_elems}xi32>) "
        f'attributes {{link_with = "{prod_obj}"}}'
    )
    lines.append(
        f'    func.func private @test_kernel_{col}_cons'
        f"(memref<{cons_out_elems}xi32>) "
        f'attributes {{link_with = "{cons_obj}"}}'
    )
    lines.append("")


def _emit_normal_core(lines: list[str], col: int, batch: dict) -> None:
    """Emit core block for a normal (assembly) batch."""
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


def _emit_cascade_cores(lines: list[str], col: int, batch: dict) -> None:
    """Emit core blocks for a cascade_pair (producer at row 3, consumer at row 2)."""
    prod_in_elems = max(1, batch["producer_in_size"] // 4)
    prod_out_elems = max(1, batch["producer_out_size"] // 4)
    cons_out_elems = max(1, batch["consumer_out_size"] // 4)

    # Producer core (row 3): acquires input from memtile + output to memtile
    lines.append(f"    aie.core(%tile_{col}_3) {{")
    lines.append(
        f"      %sub_in = aie.objectfifo.acquire @of_prod_in_{col}_1(Consume, 1) "
        f": !aie.objectfifosubview<memref<{prod_in_elems}xi32>>"
    )
    lines.append(
        f"      %elem_in = aie.objectfifo.subview.access %sub_in[0] "
        f": !aie.objectfifosubview<memref<{prod_in_elems}xi32>> -> memref<{prod_in_elems}xi32>"
    )
    lines.append(
        f"      %sub_out = aie.objectfifo.acquire @of_prod_out_{col}_0(Produce, 1) "
        f": !aie.objectfifosubview<memref<{prod_out_elems}xi32>>"
    )
    lines.append(
        f"      %elem_out = aie.objectfifo.subview.access %sub_out[0] "
        f": !aie.objectfifosubview<memref<{prod_out_elems}xi32>> -> memref<{prod_out_elems}xi32>"
    )
    lines.append(
        f"      func.call @test_kernel_{col}_prod(%elem_in, %elem_out) "
        f": (memref<{prod_in_elems}xi32>, memref<{prod_out_elems}xi32>) -> ()"
    )
    lines.append(f"      aie.objectfifo.release @of_prod_in_{col}_1(Consume, 1)")
    lines.append(f"      aie.objectfifo.release @of_prod_out_{col}_0(Produce, 1)")
    lines.append("      aie.end")
    lines.append("    }")
    lines.append("")

    # Consumer core (row 2): acquires output buffer, reads cascade implicitly
    lines.append(f"    aie.core(%tile_{col}_2) {{")
    lines.append(
        f"      %sub_out = aie.objectfifo.acquire @of_cons_out_{col}(Produce, 1) "
        f": !aie.objectfifosubview<memref<{cons_out_elems}xi32>>"
    )
    lines.append(
        f"      %elem_out = aie.objectfifo.subview.access %sub_out[0] "
        f": !aie.objectfifosubview<memref<{cons_out_elems}xi32>> -> memref<{cons_out_elems}xi32>"
    )
    lines.append(
        f"      func.call @test_kernel_{col}_cons(%elem_out) "
        f": (memref<{cons_out_elems}xi32>) -> ()"
    )
    lines.append(f"      aie.objectfifo.release @of_cons_out_{col}(Produce, 1)")
    lines.append("      aie.end")
    lines.append("    }")
    lines.append("")


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
        help="Output directory for aie.mlir and renamed .o files.",
    )
    parser.add_argument(
        "--obj-dir",
        default=None,
        help=(
            "Directory containing compiled batch_NNN.o files.  When provided, "
            "each .o is copied into --out-dir and its test_kernel symbol is "
            "renamed to test_kernel_N before the MLIR is generated."
        ),
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rename batch .o symbols before generating MLIR so that link_with
    # targets exist with the correct per-tile symbol names.
    if args.obj_dir is not None:
        renamed = prepare_phase_objects(selected, str(out_dir), args.obj_dir)
        print(f"Prepared {len(renamed)} object file(s) in {out_dir}")

    mlir = generate_phase_mlir(selected, args.phase_idx)

    out_path = out_dir / "aie.mlir"
    out_path.write_text(mlir)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
