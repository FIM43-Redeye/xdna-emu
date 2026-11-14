"""Pure-logic tests for tools/gen_launch_initvals_probe.py: the case list and what each case changes, the
prefix write list, the generated design text (what is in it and what is not), the host-buffer arithmetic the
bounds rest on, the session order and the table logic. No compiler, runner or device."""
import re
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen_launch_initvals_probe as g  # noqa: E402
import gen_launch_probe as launch  # noqa: E402


def design(name):
    return g.design_mlir(g.case_from_name(name))


def sequence_ops(text):
    body = text[text.index("aie.runtime_sequence"):]
    return re.findall(r"aiex\.npu\.(\w+)", body)


# -- cases ----------------------------------------------------------------------------------------------

def test_five_cases_each_one_factor_from_base_and_all_three():
    cs = {c.name: c for c in g.default_cases()}
    assert list(cs) == ["base", "initvals", "bigbuf", "prefix", "all"]
    flags = {n: (c.initvals, c.bigbuf, c.prefix) for n, c in cs.items()}
    assert flags == {"base": (False, False, False), "initvals": (True, False, False),
                     "bigbuf": (False, True, False), "prefix": (False, False, True),
                     "all": (True, True, True)}


def test_case_from_name_round_trips_and_rejects_unknown():
    for c in g.default_cases():
        assert g.case_from_name(c.name) == c
    with pytest.raises(ValueError):
        g.case_from_name("nope")


def test_every_case_has_a_termination_line():
    assert {c.name for c in g.default_cases()} == set(g.TERMINATION)


def test_session_is_three_rounds_base_first_with_a_tag_per_round():
    order = g.session_order()
    assert len(order) == 15
    for r in range(3):
        rnd = order[5 * r:5 * r + 5]
        assert [n for n, _ in rnd] == ["base", "initvals", "bigbuf", "prefix", "all"]
        assert {t for _, t in rnd} == {f"h{r + 1}"}


# -- prefix writes --------------------------------------------------------------------------------------

def test_prefix_is_38_writes_of_the_setup_registers_only():
    w = g.prefix_writes()
    assert len(w) == g.PREFIX_OPS == 38
    assert set(w) == set(g.PREFIX_SETUP)
    assert w[:len(g.PREFIX_SETUP)] == g.PREFIX_SETUP and w[len(g.PREFIX_SETUP):] == g.PREFIX_SETUP[:18]


def test_prefix_touches_only_the_shim_and_the_measured_compute_tile():
    assert {row for row, _, _ in g.prefix_writes()} == {0, 2}


def test_prefix_starts_nothing():
    """No software event generator, no DMA task queue, no shim BD register (0x1D000..0x1D1FF)."""
    event_generate = 0x34008
    for row, addr, _ in g.prefix_writes():
        assert addr != event_generate
        assert not (row == 0 and 0x1D000 <= addr < 0x1D400)


def test_prefix_does_not_reset_the_measured_timer():
    """No prefix write sets Timer_Control's reset bit; only the measured operation does."""
    for row, addr, val in g.prefix_writes():
        if addr == launch.TIMER_CONTROL:
            assert not val & launch.TIMER_RESET


# -- design text ----------------------------------------------------------------------------------------

def test_only_initvals_cases_have_a_memtile_buffer_task_and_sink():
    for c in g.default_cases():
        t = g.design_mlir(c)
        assert ("aie.memtile_dma" in t) == c.initvals
        assert ("initial_value" in t) == c.initvals
        assert ("aie.mem(" in t) == c.initvals
        assert ("aie.flow(" in t) == c.initvals
        assert ("aie.tile(0, 1)" in t) == c.initvals


def test_initvals_task_and_sink_have_the_same_fixed_length_and_the_values_count_up():
    t = design("initvals")
    lens = re.findall(r"aie\.dma_bd\([^)]*len = (\d+)\)", t)
    assert lens == [str(g.INIT_WORDS)] * 2
    values = re.search(r"initial_value = dense<\[([^\]]*)\]>", t).group(1).split(", ")
    assert values == [str(i) for i in range(1, g.INIT_WORDS + 1)]
    assert t.count("aie.dma_start") == 2 and "repeat_count" not in t


def test_initvals_dmas_use_no_locks_and_the_sink_tile_has_no_core():
    t = design("initvals")
    assert "use_lock" not in t
    assert t.count("aie.core(") == 1


def test_one_core_one_wait_and_the_reset_is_the_first_op_of_a_case_without_prefix():
    for c in g.default_cases():
        t = g.design_mlir(c)
        ops = sequence_ops(t)
        assert ops.count("dma_wait") == 1 and ops.count("dma_memcpy_nd") == 1
        assert ops[-2:] == ["dma_memcpy_nd", "dma_wait"]
        assert ops[:-2] == ["write32"] * (c.write_ops + 1)
        assert t.count("aie.core(") == 1


def test_the_measured_reset_follows_the_prefix_and_targets_the_core_timer():
    t = design("prefix")
    writes = re.findall(r"aiex\.npu\.write32\([^)]*\) \{column = (\d+) : i32, row = (\d+) : i32\}", t)
    assert len(writes) == 39
    assert writes[-1] == ("0", "2")
    assert f"= arith.constant {launch.TIMER_CONTROL} : i32" in t.split("%w38_addr")[1].split("\n")[0]
    assert f"arith.constant {launch.TIMER_RESET - (1 << 32)} : i32" in t


def test_prefix_registers_are_the_listed_ones_in_order():
    t = design("prefix")
    addrs = [int(a) for a in re.findall(r"%w\d+_addr = arith\.constant (-?\d+) : i32", t)]
    assert addrs[:-1] == [a for _, a, _ in g.prefix_writes()] and addrs[-1] == launch.TIMER_CONTROL


def test_bigbuf_sizes_and_the_drain_offset():
    t = design("bigbuf")
    assert f"%out : memref<{g.BIG_OUT_WORDS}xi32>" in t
    assert f"%second : memref<{g.BIG_SECOND_BYTES}xi8>" in t
    assert f"%coff = arith.constant {g.BIG_OUT_WORDS - g.OUT_WORDS} : i64" in t
    small = design("base")
    assert f"%out : memref<{g.OUT_WORDS}xi32>" in small and "%coff" not in small


def test_no_dma_references_the_second_buffer():
    for c in g.default_cases():
        body = g.design_mlir(c)
        assert body.count("%second") == 1          # the declaration only
        assert "%out[" in body


# -- host buffers ---------------------------------------------------------------------------------------

def test_expected_runner_sizes_reach_the_end_of_the_drained_words():
    assert g.expected_arg_sizes(g.case_from_name("base")) == {0: 32}
    assert g.expected_arg_sizes(g.case_from_name("bigbuf")) == {0: 65536}
    assert g.expected_arg_sizes(g.case_from_name("all")) == {0: 65536}


def test_allocated_sizes_add_the_second_buffer_and_contain_the_drain():
    for c in g.default_cases():
        alloc = g.allocated_arg_sizes(c)
        assert alloc[g.SECOND_ARG] == c.second_bytes
        drain_end = (c.drain_offset_words + g.OUT_WORDS) * g.WORD
        assert drain_end <= alloc[0]


def test_big_buffers_match_the_real_test():
    assert g.BIG_OUT_WORDS * g.WORD == 64 * 1024 and g.BIG_SECOND_BYTES == 1 << 20


# -- results --------------------------------------------------------------------------------------------

def timer_words(t0, before, jumped, left=5):
    return (t0, 0, before, 0, 0, jumped, left, 0)


def test_drain_words_reads_the_end_of_a_big_buffer():
    c = g.case_from_name("bigbuf")
    buf = bytearray(c.out_words * g.WORD)
    buf[-32:] = struct.pack("<8I", *timer_words(10, 30010, 1))
    assert g.drain_words(bytes(buf), c)[0] == 10
    assert g.result_of(g.drain_words(bytes(buf), c)).lead == 30000


def test_result_of_an_empty_drain_has_no_lead():
    assert g.result_of((0,) * 8).lead is None


def test_lead_is_the_launch_probes_lead():
    w = timer_words(100, 28700, 1)
    assert g.result_of(w).lead == launch.lead(w) == 28600
    w = timer_words(500, 0, 0)
    assert g.result_of(w).lead == -500


def test_table_gives_each_case_median_and_the_difference_to_base():
    rows = [("base", "h1.hw", g.Result(28500, 1, 5, 3)), ("base", "h2.hw", g.Result(28600, 1, 5, 3)),
            ("base", "h3.hw", g.Result(24000, 1, 5, 3)),
            ("prefix", "h1.hw", g.Result(32700, 1, 5, 3)), ("prefix", "h2.hw", g.Result(32800, 1, 5, 3)),
            ("prefix", "h3.hw", g.Result(None, 0, 0, 0))]
    out = g.format_table(rows)
    assert "base      median 28500 min 24000 max 28600 n 3" in out
    assert "prefix    median 32750 min 32700 max 32800 n 2  vs base +4250" in out
    assert " -" in out.splitlines()[6]          # the run with nothing stored shows a dash, not a number
