#!/usr/bin/env python3
"""Unit tests for isa-multi-tile-gen.py."""

import os
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

# Allow importing from the same tools/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module

# The module has a hyphenated filename, so use importlib.
_mod = import_module("isa-multi-tile-gen")
assign_phases = _mod.assign_phases
compute_phase_buffer_layout = _mod.compute_phase_buffer_layout
generate_phase_mlir = _mod.generate_phase_mlir
prepare_phase_objects = _mod.prepare_phase_objects


def _batch(in_size: int = 256, out_size: int = 64, filename: str = "batch_000.s"):
    """Helper to create a minimal batch dict."""
    return {"in_size": in_size, "out_size": out_size, "filename": filename}


# ---------------------------------------------------------------------------
# Phase assignment
# ---------------------------------------------------------------------------


class TestPhaseAssignment:
    def test_4_batches_one_phase(self):
        batches = [_batch() for _ in range(4)]
        phases = assign_phases(batches)
        assert len(phases) == 1
        assert len(phases[0]) == 4

    def test_8_batches_two_phases(self):
        batches = [_batch() for _ in range(8)]
        phases = assign_phases(batches)
        assert len(phases) == 2
        assert all(len(p) == 4 for p in phases)

    def test_5_batches_partial_second_phase(self):
        batches = [_batch() for _ in range(5)]
        phases = assign_phases(batches)
        assert len(phases) == 2
        assert len(phases[0]) == 4
        assert len(phases[1]) == 1

    def test_empty_input(self):
        phases = assign_phases([])
        assert phases == []


# ---------------------------------------------------------------------------
# Buffer layout
# ---------------------------------------------------------------------------


class TestBufferLayout:
    def test_single_batch_offsets(self):
        batches = [_batch(in_size=256, out_size=64)]
        layout = compute_phase_buffer_layout(batches)
        assert layout["in_offsets_elems"] == [0]
        assert layout["out_offsets_elems"] == [0]
        assert layout["total_in_elems"] == 64  # 256 / 4
        assert layout["total_out_elems"] == 16  # 64 / 4

    def test_two_batches_contiguous(self):
        batches = [
            _batch(in_size=256, out_size=64),
            _batch(in_size=128, out_size=32),
        ]
        layout = compute_phase_buffer_layout(batches)
        # First batch: offset 0
        assert layout["in_offsets_elems"][0] == 0
        assert layout["out_offsets_elems"][0] == 0
        # Second batch: offset = first batch's element count
        assert layout["in_offsets_elems"][1] == 64  # 256/4
        assert layout["out_offsets_elems"][1] == 16  # 64/4
        # Totals
        assert layout["total_in_elems"] == 64 + 32  # (256+128)/4
        assert layout["total_out_elems"] == 16 + 8  # (64+32)/4

    def test_byte_and_elem_offsets_consistent(self):
        batches = [
            _batch(in_size=256, out_size=64),
            _batch(in_size=128, out_size=32),
        ]
        layout = compute_phase_buffer_layout(batches)
        for i in range(len(batches)):
            assert layout["in_offsets_bytes"][i] == layout["in_offsets_elems"][i] * 4
            assert layout["out_offsets_bytes"][i] == layout["out_offsets_elems"][i] * 4
        assert layout["total_in_bytes"] == layout["total_in_elems"] * 4
        assert layout["total_out_bytes"] == layout["total_out_elems"] * 4


# ---------------------------------------------------------------------------
# MLIR generation
# ---------------------------------------------------------------------------


class TestGeneratePhaseMlir:
    def _single_tile_mlir(self):
        return generate_phase_mlir([_batch()], phase_idx=0)

    def _four_tile_mlir(self):
        batches = [
            _batch(filename=f"batch_{i:03d}.s") for i in range(4)
        ]
        return generate_phase_mlir(batches, phase_idx=0)

    def test_single_tile_has_device(self):
        mlir = self._single_tile_mlir()
        assert "aie.device(npu1_1col)" in mlir

    def test_single_tile_has_objectfifo(self):
        mlir = self._single_tile_mlir()
        assert "@of_in_0" in mlir
        assert "@of_out_0" in mlir
        assert "aie.objectfifo @of_in_0" in mlir

    def test_single_tile_has_link_with(self):
        mlir = self._single_tile_mlir()
        assert 'link_with = "batch_000.o"' in mlir

    def test_single_tile_has_core(self):
        mlir = self._single_tile_mlir()
        assert "aie.core(%tile_0_2)" in mlir
        assert "func.call @test_kernel_0" in mlir

    def test_single_tile_has_runtime_sequence(self):
        mlir = self._single_tile_mlir()
        assert "aie.runtime_sequence" in mlir
        assert "aiex.npu.dma_memcpy_nd" in mlir
        assert "aiex.npu.dma_wait" in mlir

    def test_four_tiles_all_columns(self):
        mlir = self._four_tile_mlir()
        for col in range(4):
            assert f"aie.tile({col}, 0)" in mlir
            assert f"aie.tile({col}, 2)" in mlir
            assert f"aie.core(%tile_{col}_2)" in mlir
            assert f"@of_in_{col}" in mlir
            assert f"@of_out_{col}" in mlir

    def test_four_tiles_four_link_withs(self):
        mlir = self._four_tile_mlir()
        for i in range(4):
            assert f'link_with = "batch_{i:03d}.o"' in mlir

    def test_partial_phase_uses_ncol_device(self):
        batches = [_batch() for _ in range(2)]
        mlir = generate_phase_mlir(batches, phase_idx=0)
        assert "aie.device(npu1_2col)" in mlir

    def test_full_phase_uses_npu1_device(self):
        mlir = self._four_tile_mlir()
        assert "aie.device(npu1)" in mlir
        # Make sure it is not npu1_4col
        assert "npu1_4col" not in mlir

    def test_output_dma_has_issue_token(self):
        mlir = self._single_tile_mlir()
        # Find lines with of_out in dma_memcpy_nd
        for line in mlir.splitlines():
            if "dma_memcpy_nd" in line and "@of_out_" in line:
                assert "issue_token = true" in line, (
                    f"Output DMA missing issue_token: {line}"
                )

    def test_input_dma_no_issue_token(self):
        mlir = self._single_tile_mlir()
        for line in mlir.splitlines():
            if "dma_memcpy_nd" in line and "@of_in_" in line:
                assert "issue_token" not in line, (
                    f"Input DMA should not have issue_token: {line}"
                )

    def test_dma_wait_per_output(self):
        mlir = self._four_tile_mlir()
        for col in range(4):
            assert f"symbol = @of_out_{col}" in mlir

    def test_dma_ids_unique(self):
        mlir = self._four_tile_mlir()
        ids = []
        for line in mlir.splitlines():
            if "dma_memcpy_nd" in line:
                # Extract id = N
                import re

                m = re.search(r"id = (\d+)", line)
                assert m, f"No id found in DMA line: {line}"
                ids.append(int(m.group(1)))
        # All IDs should be unique
        assert len(ids) == len(set(ids)), f"Duplicate DMA IDs: {ids}"

    def test_buffer_sizes_in_runtime_sequence(self):
        """Runtime sequence memref types should use total element counts."""
        batches = [
            _batch(in_size=256, out_size=64),
            _batch(in_size=128, out_size=32),
        ]
        mlir = generate_phase_mlir(batches, phase_idx=0)
        # Total in = (256+128)/4 = 96, total out = (64+32)/4 = 24
        assert "memref<96xi32>" in mlir
        assert "memref<24xi32>" in mlir


# ---------------------------------------------------------------------------
# prepare_phase_objects
# ---------------------------------------------------------------------------


def _batch_indexed(
    idx: int,
    in_size: int = 256,
    out_size: int = 64,
    filename: str = None,
):
    """Batch dict with a batch_index field, as produced by the manifest."""
    if filename is None:
        filename = f"batch_{idx:03d}.s"
    return {
        "batch_index": idx,
        "in_size": in_size,
        "out_size": out_size,
        "filename": filename,
    }


class TestPreparePhaseObjects:
    """Verify .o copy and symbol rename behaviour (no real toolchain needed)."""

    def _run(self, batches, phase_dir, obj_dir):
        """Call prepare_phase_objects with mocked copy2 and subprocess.run."""
        with (
            mock.patch("shutil.copy2") as m_copy,
            mock.patch("subprocess.run") as m_run,
            mock.patch("os.makedirs"),
        ):
            result = prepare_phase_objects(batches, phase_dir, obj_dir)
        return result, m_copy, m_run

    def test_single_batch_copy_path(self):
        """copy2 is called with batch_000.o as source and the derived dst."""
        batches = [_batch_indexed(0)]
        result, m_copy, _ = self._run(batches, "/phase/0", "/objs")
        assert m_copy.call_count == 1
        src, dst = m_copy.call_args[0]
        assert src == "/objs/batch_000.o"
        assert dst == "/phase/0/batch_000.o"

    def test_single_batch_redefine_sym(self):
        """llvm-objcopy is called with test_kernel=test_kernel_0 for col 0."""
        batches = [_batch_indexed(0)]
        _, _, m_run = self._run(batches, "/phase/0", "/objs")
        assert m_run.call_count == 1
        cmd = m_run.call_args[0][0]
        assert "--redefine-sym" in cmd
        assert "test_kernel=test_kernel_0" in cmd

    def test_column_index_not_batch_index(self):
        """Symbol rename uses column position, not the batch_index field.

        Batch 7 assigned to column 1 of a phase should get test_kernel_1,
        not test_kernel_7.
        """
        # Two batches: batch_005 at col 0, batch_007 at col 1
        batches = [_batch_indexed(5), _batch_indexed(7)]
        _, _, m_run = self._run(batches, "/phase/1", "/objs")
        assert m_run.call_count == 2
        calls = [m_run.call_args_list[i][0][0] for i in range(2)]
        # Column 0
        assert "test_kernel=test_kernel_0" in calls[0]
        # Column 1 (not test_kernel_7)
        assert "test_kernel=test_kernel_1" in calls[1]
        assert "test_kernel=test_kernel_7" not in calls[1]

    def test_four_batches_four_calls(self):
        """One copy and one objcopy call per batch."""
        batches = [_batch_indexed(i) for i in range(4)]
        result, m_copy, m_run = self._run(batches, "/phase/2", "/objs")
        assert m_copy.call_count == 4
        assert m_run.call_count == 4
        assert len(result) == 4

    def test_returned_filenames_match_obj_filename(self):
        """Returned list entries are .o names derived from batch filenames."""
        batches = [
            _batch_indexed(0, filename="batch_000.s"),
            _batch_indexed(1, filename="batch_001.ll"),
        ]
        result, _, _ = self._run(batches, "/phase/0", "/objs")
        assert result[0] == "batch_000.o"
        assert result[1] == "batch_001.o"

    def test_src_path_uses_batch_index_not_col(self):
        """Source .o is addressed by batch_index, ensuring correct file is read."""
        # col 0 = batch 3, col 1 = batch 9
        batches = [_batch_indexed(3), _batch_indexed(9)]
        _, m_copy, _ = self._run(batches, "/p", "/objs")
        srcs = [m_copy.call_args_list[i][0][0] for i in range(2)]
        assert srcs[0] == "/objs/batch_003.o"
        assert srcs[1] == "/objs/batch_009.o"

    def test_dst_path_inside_phase_dir(self):
        """Destination files land in phase_dir, not obj_dir."""
        batches = [_batch_indexed(0), _batch_indexed(1)]
        _, m_copy, _ = self._run(batches, "/my/phase", "/my/objs")
        for call in m_copy.call_args_list:
            _, dst = call[0]
            assert dst.startswith("/my/phase/")

    def test_subprocess_check_true(self):
        """subprocess.run is called with check=True for error propagation."""
        batches = [_batch_indexed(0)]
        _, _, m_run = self._run(batches, "/p", "/o")
        kwargs = m_run.call_args[1]
        assert kwargs.get("check") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
