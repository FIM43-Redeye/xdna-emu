"""Tests for tools/gen_dma_lock_probe.py: predictions, programs, design text."""
import struct
import subprocess

import pytest

import gen_dma_lock_probe as g
from harness.build import _peano_tool


def peano_mc():
    try:
        return _peano_tool("llvm-mc")
    except Exception:
        return None


def test_every_case_tells_the_rules_apart():
    for case in g.CASES.values():
        assert (g.grant_step(case, "eq"), g.taken(case, "eq")) != \
            (g.grant_step(case, "eqdec"), g.taken(case, "eqdec"))


def test_grant_steps():
    assert g.grant_step(g.CASES["c2v1"], "eq") == 1
    assert g.grant_step(g.CASES["c0v0"], "eq") == 0
    assert g.grant_step(g.CASES["c0v0"], "eqdec") == 1


def test_design_uses_the_equal_form_on_bd0():
    text = g.design_mlir(g.CASES["c2v1"])
    assert "%a0 = arith.constant 2 : i32\n      aie.use_lock(%lA, Acquire, %a0)" in text
    assert "aie.flow(%t02, DMA : 0, %t02, DMA : 0)" in text


def test_analyze_out_matches_a_rule():
    meta = {"steps": 3, "grant_step": {"eq": 1, "eqdec": 1}, "taken": {"eq": 0, "eqdec": 2}}
    w = [100, 3700, 7300, 10900, 10905, 10910] + [0] * 9 + [1]
    r = g.analyze_out(meta, struct.pack("<16I", *w), 4000.0)
    assert r["match"] == ["eq"] and r["marker"] == 1


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("name", list(g.CASES))
def test_asm_assembles(name, tmp_path):
    for k, text in (("core", g.core_asm(g.CASES[name], 50)), ("rescuer", g.rescuer_asm(g.CASES[name], 50))):
        s = tmp_path / f"{k}.s"
        s.write_text(text)
        r = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / f"{k}.o"),
                            str(s)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


# -- lock-to-fetch probe ------------------------------------------------------------

import trace_decoder as td                                          # noqa: E402
from trace_decoder.frame import EventCmd, RepeatCmd, StartCmd       # noqa: E402


def phys_bank(addr: int) -> int:
    """Physical data-memory bank: a 16 KB logical bank is two banks interleaved every 16 bytes."""
    return 2 * (addr >> 14) + ((addr >> 4) & 1)


def bundles_of(text: str) -> list[str]:
    """The kernel body between the frame's register saves and restores, one entry per bundle."""
    lines = [ln.strip() for ln in text.splitlines()]
    start = max(i for i, ln in enumerate(lines) if ln.startswith("st r31,")) + 1
    end = min(i for i, ln in enumerate(lines) if ln.startswith("lda r16,"))
    return [ln for ln in lines[start:end] if ln and not ln.startswith((".", "#")) and not ln.endswith(":")]


@pytest.mark.parametrize("k", [0, 1, 3, 7, 15])
def test_fetch_core_release_nops_loads(k):
    body = bundles_of(g.fetch_core_asm(k))
    i = body.index("event #0")
    assert body[i + 1] == f"rel #{g.OWN + g.FETCH_LOCK_A}, r2"          # the marker is just before it
    assert body[i + 2:i + 2 + k] == ["nop"] * k                          # exactly k nops after the release
    assert body[i + 2 + k] == "lda r26, [p0, #0]"                        # physical bank 0
    assert body[i + 3 + k] == "lda r28, [p0, #16]"                       # physical bank 1, one bundle later
    assert sum(b.startswith("rel ") for b in body) == 1                  # the window has one release


def test_fetch_core_waits_for_go_before_the_marker_and_for_b_after_the_loads():
    body = bundles_of(g.fetch_core_asm(3))
    go = body.index(f"acq #{g.OWN + g.FETCH_LOCK_GO}, r3")
    assert go < body.index("event #0")
    done = body.index(f"acq #{g.OWN + g.FETCH_LOCK_B}, r3")
    assert done > body.index("lda r28, [p0, #16]")
    # these are the only acquires: nothing waits on a host-fed lock
    assert sum(b.startswith("acq ") for b in body) == 2


def test_fetch_core_rejects_k_out_of_range():
    for k in (-1, 16):
        with pytest.raises(ValueError):
            g.fetch_core_asm(k)


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("k", [0, 3, 7, 15])
def test_fetch_core_assembles(k, tmp_path):
    s = tmp_path / "core.s"
    s.write_text(g.fetch_core_asm(k))
    r = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / "core.o"), str(s)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_fetch_case_names_cover_k_0_to_15():
    assert g.FETCH_KS == tuple(range(16))
    assert [g.fetch_case_name(k) for k in (0, 3, 15)] == ["lfk00", "lfk03", "lfk15"]


def test_fetch_design_geometry():
    assert phys_bank(g.FETCH_SRC_ADDR) == 0 and g.FETCH_SRC_ADDR >> 14 == 0     # bank 0
    assert [phys_bank(g.FETCH_SRC_ADDR + 16 * i) for i in range(4)] == [0, 1, 0, 1]   # granule parity
    assert g.FETCH_DST_ADDR >> 14 == 2                                           # the S2MM stays in bank 2
    assert g.FETCH_WORDS == 64
    assert (phys_bank(g.FETCH_LD_ADDR), phys_bank(g.FETCH_LD_ADDR + 16)) == (0, 1)   # what the two loads hit
    assert g.FETCH_SCRATCH_ADDR >> 14 not in (0, 1)                              # pad writes clear of the window banks


def test_fetch_design_text():
    t = g.fetch_design_mlir()
    assert "aie.device(npu1_1col)" in t
    assert 'aie.lock(%t02, 0) {init = 0 : i32, sym_name = "lA"}' in t           # lock id 0: reset selector
    assert "aie.flow(%t02, DMA : 0, %t02, DMA : 0)" in t
    send = t.split("^send:")[1].split("^recv_start:")[0]
    assert send.index("use_lock(%lA, AcquireGreaterEqual") < send.index("dma_bd(%src") < send.index("use_lock(%lB, Release")
    recv = t.split("^recv:")[1].split("^end:")[0]
    assert "use_lock" not in recv and "dma_bd(%dst" in recv                      # the S2MM has no lock to wait on
    seq = t.split("aie.runtime_sequence")[1]
    assert "aiex.set_lock(%lGo" in seq
    for forbidden in ("dma_await", "dma_wait", "dma_memcpy_nd", "dma_start_task", "npu.sync"):
        assert forbidden not in seq                                              # nothing in it blocks on anything
    assert "aie.shim_dma_allocation" not in t


def test_fetch_event_lists():
    assert g.FETCH_MEMMOD_EVENTS == [
        "LOCK_0_REL", "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_0_FINISHED_BD", "DMA_MM2S_0_MEMORY_STARVATION",
        "CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1", "DMA_S2MM_0_FINISHED_BD"]
    assert g.FETCH_CORE_EVENTS == ["INSTR_EVENT_0", "LOCK_STALL"]
    assert g.FETCH_CORE_GROUNDING == "PERF_CNT_2"
    args = g.fetch_inject_args()
    assert args[args.index("--memmod-sweep-events") + 1] == ",".join(g.FETCH_MEMMOD_EVENTS)
    assert args[args.index("--core-sweep-events") + 1] == "INSTR_EVENT_0,LOCK_STALL"
    assert args[args.index("--core-grounding") + 1] == "PERF_CNT_2"
    assert "--memtile-sweep-events" not in args


# -- the table ----------------------------------------------------------------------

def ev(name, ts, pkt=1, row=2, col=1):
    return td.Event(col=col, row=row, pkt_type=pkt, slot=0, name=name, ts=ts, soc=ts)


def held(name, first, last, **kw):
    """A level asserted on every cycle first..last, as the level API reports it."""
    return [ev(name, t, **kw) for t in range(first, last + 1)]


def synthetic(rel=1000, grant_after=2, conflict_bank="CONFLICT_DM_BANK_1", conflict_at=5):
    e = held("DMA_MM2S_0_STALLED_LOCK", 200, rel + grant_after - 1)    # held for hundreds of cycles
    e += [ev("LOCK_0_REL", rel)]
    e += held("DMA_MM2S_0_MEMORY_STARVATION", rel + grant_after, rel + grant_after + 3)
    e += [ev(conflict_bank, rel + conflict_at)]
    e += [ev("DMA_MM2S_0_FINISHED_BD", rel + 90), ev("DMA_S2MM_0_FINISHED_BD", rel + 97)]
    e += [ev("INSTR_EVENT_0", rel + 4, pkt=0), ev("LOCK_STALL", rel + 5, pkt=0),
          ev("LOCK_STALL", rel + 6, pkt=0), ev("LOCK_STALL", rel + 7, pkt=0)]
    return e


def test_fetch_row_reads_grant_from_the_falling_edge_of_the_held_level():
    row = g.fetch_row(synthetic(rel=1000, grant_after=2))
    assert row["release"] == 1000
    assert row["grant"] == 2                       # last asserted cycle is release + 1, the grant is the next
    assert row["starvation"] == [(2, 5)]
    assert row["conflicts"] == [("CONFLICT_DM_BANK_1", 5)]
    assert row["mm2s_finished"] == [90] and row["s2mm_finished"] == [97]
    assert row["core_event0"] == 4                 # core timer minus memory timer: cross-domain
    assert row["core_stalls"] == [(1, 3)]          # relative to the core's own INSTR_EVENT_0


def test_fetch_row_grant_zero_when_the_level_ends_just_before_the_release():
    row = g.fetch_row(synthetic(rel=1000, grant_after=0))
    assert row["grant"] == 0


def test_fetch_row_picks_the_interval_that_spans_the_release_not_an_earlier_one():
    e = held("DMA_MM2S_0_STALLED_LOCK", 10, 20) + synthetic(rel=1000, grant_after=3)
    assert g.fetch_row(e)["grant"] == 3


def test_fetch_row_groups_adjacent_conflict_cycles_and_keeps_both_banks():
    e = synthetic(conflict_at=5) + [ev("CONFLICT_DM_BANK_0", 1000 + 9), ev("CONFLICT_DM_BANK_0", 1000 + 10)]
    row = g.fetch_row(e)
    assert row["conflicts"] == [("CONFLICT_DM_BANK_1", 5), ("CONFLICT_DM_BANK_0", 9), ("CONFLICT_DM_BANK_0", 10)]


def test_fetch_row_without_a_release_event_says_so():
    row = g.fetch_row([ev("DMA_MM2S_0_FINISHED_BD", 5)])
    assert row["release"] is None and "no LOCK_0_REL" in g.format_fetch_table([(3, "h1.hw", row)])


def test_format_fetch_table_columns_and_order():
    rows = [(3, "e1.emu", g.fetch_row(synthetic())), (0, "e1.emu", g.fetch_row(synthetic(conflict_at=9)))]
    text = g.format_fetch_table(rows)
    lines = text.splitlines()
    assert lines[0].split()[:3] == ["k", "run", "release"]
    assert lines[1].startswith(" 0 ") and lines[2].startswith(" 3 ")     # sorted by k
    assert "grant +2" in lines[1] and "CONFLICT_DM_BANK_1@+9" in lines[1]
    assert "starv [+2..+5]" in lines[1]
    assert "mm2s_fin [+90]" in lines[1] and "s2mm_fin [+97]" in lines[1]
    assert "core(xtimer) ev0 +4" in lines[1]


def test_levels_from_commands_expands_a_held_level():
    # one frame raising slot 0 (a level), a zero-cycle Repeat of 9 more cycles holding it, then it drops
    cmds = {(1, 2, 1): [StartCmd(500), EventCmd(event_bits=0b1, cycles=0), RepeatCmd(9),
                        EventCmd(event_bits=0, cycles=0)]}
    names = {1: {"2,1": ["DMA_MM2S_0_STALLED_LOCK"] + [""] * 7}}
    events = g.level_events(cmds, names)
    ts = sorted(e.ts for e in events if e.name == "DMA_MM2S_0_STALLED_LOCK")
    assert ts == list(range(ts[0], ts[0] + 10))
