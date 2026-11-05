"""The runtime-sequence op-cost probe: case matrix, op sequences, kernel and MLIR text, analysis."""
import os
import shutil
import struct
import subprocess

import pytest

import gen_fw_op_cost_probe as g


def case(name):
    return next(c for c in g.all_cases() if c.name == name)


# -- case matrix ------------------------------------------------------------------------------

def test_matrix_covers_every_op_kind_and_sweep():
    names = {c.name for c in g.all_cases()}
    for n in (8, 16, 32, 64):
        assert f"w32_n{n}" in names and f"mw32_n{n}" in names
    for l in (1, 4, 16, 64):
        assert f"bw_l{l}" in names
    assert "bw_l16_last" in names
    assert {"control", "spin_a", "spin_b"} <= names
    assert any(c.filler for c in g.all_cases())
    assert len(names) == len(g.all_cases())


def test_every_case_fits_the_poll_buffer_and_results_buffer():
    for c in g.all_cases():
        idx = g.polled_indices(c)
        assert idx and max(idx) < c.words <= g.POLL_WORDS_MAX, c.name
        assert g.results_words(c) == (4 if c.spin_iters else 2 * len(idx))


# -- host op sequence -------------------------------------------------------------------------

def test_write32_sequence_is_one_op_per_word_in_order():
    ops = g.sequence_ops(case("w32_n8"))
    assert ops == [("write32", "poll", k, g.marker(k)) for k in range(8)]


def test_markers_are_nonzero_and_distinct():
    vals = [g.marker(k) for k in range(g.POLL_WORDS_MAX)]
    assert 0 not in vals and len(set(vals)) == len(vals)


def test_blockwrite_sequence_groups_words_into_blocks():
    ops = g.sequence_ops(case("bw_l4"))
    assert len(ops) == case("bw_l4").words // 4
    kind, target, off, words = ops[2]
    assert (kind, target, off) == ("blockwrite", "poll", 8)
    assert words == [g.marker(k) for k in range(8, 12)]


def test_maskwrite_sequence_uses_a_partial_mask_that_keeps_the_marker():
    (kind, target, off, value, mask) = g.sequence_ops(case("mw32_n8"))[3]
    assert (kind, target, off) == ("maskwrite32", "poll", 3)
    assert value & mask == g.marker(3) and mask != 0xFFFFFFFF


def test_interleave_puts_the_fillers_before_each_polled_write():
    c = next(c for c in g.all_cases() if c.filler)
    ops = g.sequence_ops(c)
    per = c.filler.count + 1
    assert len(ops) == per * c.words
    for k in range(c.words):
        group = ops[k * per:(k + 1) * per]
        assert [o[1] for o in group[:-1]] == [c.filler.target] * c.filler.count
        assert group[-1][1] == "poll" and group[-1][3] == g.marker(k)


def test_control_has_no_runtime_writes_and_presets_the_words():
    c = case("control")
    assert g.sequence_ops(c) == [] and c.preset
    assert "dense<0>" not in g.mlir_text(c)


def test_spin_case_releases_the_core_with_one_write():
    ops = g.sequence_ops(case("spin_a"))
    assert ops == [("write32", "poll", 0, g.marker(0))]


def test_polled_indices_follow_start_and_stride():
    c = case("bw_l16_last")
    assert g.polled_indices(c) == list(range(15, c.words, 16))


# -- kernel text ------------------------------------------------------------------------------

def test_kernel_uses_the_tile_counter_and_a_bounded_poll():
    text = g.kernel_text(case("w32_n8"))
    assert "mov r25:r24, cntr" in text
    assert ".globl probe" in text
    assert "#" + str(g.TIMEOUT_FIRST) in text and "#" + str(g.TIMEOUT_NEXT) in text


def test_kernel_clobbers_only_caller_saved_registers():
    import re
    for c in g.all_cases():
        regs = {int(m) for m in re.findall(r"\br(\d+)\b", g.kernel_text(c))}
        assert not regs & set(range(16, 24)), c.name
        assert not {"p6", "p7"} & set(re.findall(r"\bp\d\b", g.kernel_text(c))), c.name


@pytest.mark.skipif(not os.environ.get("PEANO_INSTALL_DIR"), reason="Peano not activated")
def test_every_kernel_assembles(tmp_path):
    mc = os.path.join(os.environ["PEANO_INSTALL_DIR"], "bin", "llvm-mc")
    if not os.path.exists(mc):
        pytest.skip("llvm-mc missing")
    for c in g.all_cases():
        src = tmp_path / f"{c.name}.s"
        src.write_text(g.kernel_text(c))
        r = subprocess.run([mc, "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / "k.o"),
                            str(src)], capture_output=True, text=True)
        assert r.returncode == 0, (c.name, r.stderr)


# -- MLIR -------------------------------------------------------------------------------------

def test_mlir_names_the_buffers_the_sequence_writes_to():
    text = g.mlir_text(case("w32_n8"))
    assert "aie.device(npu1)" in text
    assert 'sym_name = "poll"' in text and "{buffer = @poll}" in text
    assert text.count("aiex.npu.write32") == 8
    assert "aiex.npu.dma_wait {symbol = @of_out}" in text


def test_mlir_places_the_output_dma_before_the_timed_writes():
    text = g.mlir_text(case("w32_n8"))
    assert text.index("dma_memcpy_nd") < text.index("aiex.npu.write32")


def test_mlir_declares_scratch_tiles_only_for_cases_that_write_them():
    plain = g.mlir_text(case("w32_n8"))
    assert "mt_scratch" not in plain and "c1_scratch" not in plain
    targets = {c.filler.target for c in g.all_cases() if c.filler}
    assert targets == {"memtile", "column1"}
    for c in g.all_cases():
        if c.filler:
            assert {"memtile": "mt_scratch", "column1": "c1_scratch"}[c.filler.target] in g.mlir_text(c)


def test_blockwrite_mlir_carries_one_global_per_block():
    text = g.mlir_text(case("bw_l16"))
    assert text.count("memref.global") == case("bw_l16").words // 16
    assert text.count("aiex.npu.blockwrite") == case("bw_l16").words // 16


def test_poll_buffer_is_pinned_above_the_allocator_range_in_its_own_bank():
    # The output fifo is allocated from the bottom of data memory; the polled words are pinned
    # far above it (bank 0 holds the fifo under either 8 KiB or 16 KiB banks).
    assert g.POLL_ADDR >= 2 * g.BANK_BYTES
    assert f"address = {g.POLL_ADDR} : i32" in g.mlir_text(case("w32_n8"))


# -- analysis ---------------------------------------------------------------------------------

def make_out(c, stamps, values=None):
    idx = g.polled_indices(c)
    values = values or [g.marker(k) for k in idx]
    return struct.pack(f"<{2 * len(idx)}I", *stamps, *values)


def test_intervals_are_successive_differences_modulo_2_32():
    c = case("w32_n8")
    stamps = [4294967290, 20, 50, 51, 100, 140, 141, 142]
    r = g.analyze(c, make_out(c, stamps))
    assert r.intervals == [26, 30, 1, 49, 40, 1, 1]
    assert r.ok and not r.missing


def test_analysis_flags_timeouts_and_wrong_values():
    c = case("w32_n8")
    vals = [g.marker(k) for k in range(8)]
    vals[3] = 0
    vals[5] = 0xDEAD
    r = g.analyze(c, make_out(c, list(range(0, 80, 10)), vals))
    assert r.missing == [3] and r.wrong == [5] and not r.ok


def test_summary_reports_min_median_max():
    s = g.summarize([5, 1, 9, 3, 7])
    assert (s["min"], s["median"], s["max"], s["n"]) == (1, 5, 9, 5)


def test_spin_analysis_returns_cycles_between_the_two_stamps():
    c = case("spin_a")
    out = struct.pack("<4I", 1000, 1000 + 123456, g.marker(0), c.spin_iters)
    assert g.analyze_spin(c, out) == 123456
