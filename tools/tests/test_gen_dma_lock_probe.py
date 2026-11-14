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
    for k in (-1, 65):
        with pytest.raises(ValueError):
            g.fetch_core_asm(k)
    g.fetch_core_asm(64)                                                 # the last valid k


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("k", [0, 3, 7, 15])
def test_fetch_core_assembles(k, tmp_path):
    s = tmp_path / "core.s"
    s.write_text(g.fetch_core_asm(k))
    r = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / "core.o"), str(s)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_fetch_case_names_cover_k_0_to_15():
    assert g.FETCH_KS == tuple(range(65))                                # valid k; the default build is the first 16
    assert g.FETCH_DEFAULT_KS == tuple(range(16))
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


# -- chained variant ----------------------------------------------------------------

def test_chained_case_names():
    assert g.fetch_case_name(3, chained=True) == "lfc03" and g.fetch_case_name(3) == "lfk03"


def test_chained_design_bd_order_chain_and_sizes():
    t = g.fetch_design_mlir(chained=True)
    assert "aie.dma_start(MM2S, 0, ^pre, ^recv_start)" in t               # the task starts on the short BD
    pre = t.split("^pre:")[1].split("^send:")[0]
    assert "use_lock" not in pre                                          # lock-free
    assert "dma_bd(%pre : memref<4xi32>)" in pre and "aie.next_bd ^send" in pre
    send = t.split("^send:")[1].split("^recv_start:")[0]
    assert send.index("use_lock(%lA, AcquireGreaterEqual") < send.index("dma_bd(%src") < send.index("use_lock(%lB, Release")
    recv = t.split("^recv:")[1].split("^end:")[0]
    assert "use_lock" not in recv and "dma_bd(%dst : memref<68xi32>)" in recv   # room for both BDs, no lock
    assert "memref<68xi32>" in t.split("%dst = ")[1].split("\n")[0]


def test_chained_pre_buffer_is_clear_of_the_measured_banks():
    assert g.FETCH_PRE_ADDR >> 14 == 1                                    # logical bank 1: not src's, not the loads'
    assert g.FETCH_PRE_ADDR >> 14 not in (g.FETCH_SRC_ADDR >> 14, g.FETCH_LD_ADDR >> 14, g.FETCH_DST_ADDR >> 14)
    assert 'sym_name = "pre"' in g.fetch_design_mlir(chained=True)


def test_first_bd_design_is_unchanged_by_the_chained_option():
    t = g.fetch_design_mlir()
    assert "^pre" not in t and "memref<64xi32>" in t.split("^recv:")[1] and "%pre" not in t
    assert g.fetch_design_mlir(chained=False) == t


def test_chained_design_keeps_the_safety_properties():
    t = g.fetch_design_mlir(chained=True)
    seq = t.split("aie.runtime_sequence")[1]
    assert "aiex.set_lock(%lGo" in seq
    for forbidden in ("dma_await", "dma_wait", "dma_memcpy_nd", "dma_start_task", "npu.sync"):
        assert forbidden not in seq
    assert "aie.shim_dma_allocation" not in t


# -- k beyond 15 and the single-bank variants -------------------------------------------

@pytest.mark.parametrize("k", [16, 40, 64])
def test_long_nop_runs_keep_the_shape(k):
    body = bundles_of(g.fetch_core_asm(k))
    i = body.index("event #0")
    assert body[i + 2:i + 2 + k] == ["nop"] * k
    assert body[i + 2 + k] == "lda r26, [p0, #0]" and body[i + 3 + k] == "lda r28, [p0, #16]"
    assert sum(b.startswith("acq ") for b in body) == 2


@pytest.mark.parametrize("bank,load", [("0", "lda r26, [p0, #0]"), ("1", "lda r28, [p0, #16]")])
@pytest.mark.parametrize("k", [0, 5, 40])
def test_single_bank_core_has_one_access_k_nops_after_the_release(bank, load, k):
    body = bundles_of(g.fetch_core_asm(k, bank=bank))
    i = body.index("event #0")
    assert body[i + 2:i + 2 + k] == ["nop"] * k
    assert body[i + 2 + k] == load                                       # the access sits where bank 0's does
    assert sum(b.startswith("lda ") for b in body) == 1                  # and it is the only one
    assert g.fetch_core_asm(k, bank=bank) != g.fetch_core_asm(k)


def test_fetch_core_rejects_an_unknown_bank():
    with pytest.raises(ValueError):
        g.fetch_core_asm(3, bank="2")


def test_default_core_is_the_two_load_core():
    assert g.fetch_core_asm(3) == g.fetch_core_asm(3, bank="both")


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("bank", ["0", "1"])
def test_single_bank_core_assembles(bank, tmp_path):
    s = tmp_path / "core.s"
    s.write_text(g.fetch_core_asm(50, bank=bank))
    r = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / "core.o"), str(s)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_case_names_are_distinct_per_variant_and_clear_of_lfc_lfk():
    names = {(c, b): g.fetch_case_name(7, chained=c, bank=b) for c in (False, True) for b in ("both", "0", "1")}
    assert names[(False, "both")] == "lfk07" and names[(True, "both")] == "lfc07"
    assert len(set(names.values())) == 6
    for key, n in names.items():
        if key[1] != "both":
            assert not n.startswith(("lfc", "lfk")) and n.endswith("07")


def test_the_design_does_not_depend_on_the_bank():
    assert g.fetch_design_mlir(chained=True) == g.fetch_design_mlir(chained=True)   # k and bank live in the core only


# -- the per-bank map ---------------------------------------------------------------

def row(rel=1000, grant=2, conflicts=()):
    return {"release": rel, "grant": grant, "conflicts": [(b, t) for b, t in conflicts], "starvation": [],
            "mm2s_finished": [], "s2mm_finished": [], "core_event0": None, "core_stalls": []}


B0, B1 = "CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1"


def cells_of(entries):
    return g.fetch_access_cells(entries)


def test_platform_comes_from_the_run_suffix():
    assert g.run_platform("h3.hw") == "hw" and g.run_platform("e2.emu") == "emu"


def test_single_bank_conflict_marks_dma_and_clean_rows_mark_absent_relative_to_the_grant():
    # silicon offset 1: bank-0 access at rel + k + 1; grant 2 or 3
    entries = [("lfa03", 3, "0", "h3.hw", row(grant=2, conflicts=[(B0, 4)])),     # access at +4, grant-relative +2
               ("lfa04", 4, "0", "h3.hw", row(grant=3)),                           # access at +5, grant-relative +2: clean
               ("lfa06", 6, "0", "h3.hw", row(grant=2))]                           # access at +7, grant-relative +5: clean
    cells = cells_of(entries)["hw"]
    assert cells[(0, 2)]["dma"] == ["lfa03.h3.hw"] and cells[(0, 2)]["none"] == ["lfa04.h3.hw"]
    assert cells[(0, 5)] == {"dma": [], "none": ["lfa06.h3.hw"]}


def test_both_variant_second_load_is_one_cycle_behind_and_unknown_after_a_bank0_conflict():
    clean = ("lfc08", 8, "both", "h3.hw", row(grant=2))
    c = cells_of([clean])["hw"]
    assert c[(0, 7)]["none"] and c[(1, 8)]["none"]                              # b0 at +9, b1 at +10; grant 2
    hit = ("lfc03", 3, "both", "h3.hw", row(grant=2, conflicts=[(B0, 4)]))
    c = cells_of([hit])["hw"]
    assert c[(0, 2)]["dma"] and (1, 3) not in c                                  # the b1 cell is not trusted


def test_default_offsets_per_platform_and_inference_from_conflicts():
    assert g.infer_access_offsets([("c", 3, "0", "h.hw", row(conflicts=[(B0, 4)]))]) == 1
    assert g.infer_access_offsets([("c", 3, "0", "e.emu", row(conflicts=[(B0, 1)]))]) == -2
    # a both-variant bank-1-only conflict is one cycle behind the first load
    assert g.infer_access_offsets([("c", 11, "both", "h.hw", row(conflicts=[(B1, 13)]))]) == 1
    # no conflicts: the platform default
    assert g.infer_access_offsets([("c", 3, "0", "h.hw", row())]) == g.DEFAULT_ACCESS_OFFSET["hw"]
    assert g.infer_access_offsets([("c", 3, "0", "e.emu", row())]) == g.DEFAULT_ACCESS_OFFSET["emu"]


def test_contradictions_name_cell_case_and_runs():
    entries = [("lfa03", 3, "0", "h3.hw", row(grant=2, conflicts=[(B0, 4)])),
               ("lfa04", 4, "0", "h4.hw", row(grant=3)),
               ("lfa07", 7, "0", "h3.hw", row(grant=2))]
    text = g.format_fetch_map(entries)
    assert "CONTRADICTIONS" in text and "bank 0 g+2" in text
    assert "lfa03.h3.hw" in text.split("CONTRADICTIONS")[1] and "lfa04.h4.hw" in text.split("CONTRADICTIONS")[1]


def test_no_contradictions_is_stated_and_platforms_are_kept_apart():
    entries = [("lfa03", 3, "0", "h3.hw", row(grant=2, conflicts=[(B0, 4)])),
               ("lfa03", 3, "0", "e2.emu", row(grant=0))]                          # emulator: access at +1, clean
    text = g.format_fetch_map(entries)
    assert "silicon" in text and "emulator" in text
    assert "contradictions: none" in text.lower()


def test_map_lists_dma_absent_and_unknown_cycles_per_bank():
    entries = [("lfb03", 3, "1", "h3.hw", row(grant=2, conflicts=[(B1, 4)])),
               ("lfb05", 5, "1", "h3.hw", row(grant=2))]
    text = g.format_fetch_map(entries)
    assert "bank 1 DMA access proven: g+2" in text
    assert "bank 1 no DMA access: g+4" in text
    assert "bank 0 unknown" in text or "bank 0: no data" in text


# -- lock-synchronised variant (--sync): the core's release a controlled d cycles after the pre BD --------

def sync_case(d=12, bank="none", k=0, free=False):
    return g.SyncCase(d=d, bank=bank, k=k, free=free)


def test_sync_case_names_are_distinct_and_clear_of_the_other_variants():
    names = {sync_case(d).name for d in g.SYNC_DEFAULT_DS}
    names |= {sync_case(12, b, k).name for b in "01" for k in range(16)}
    names |= {sync_case(g.FREE_D, free=True).name}
    assert len(names) == len(g.SYNC_DEFAULT_DS) + 32 + 1
    other = {g.fetch_case_name(k, c, b) for k in range(65) for (c, b) in g.FETCH_PREFIX}
    assert not names & other
    assert sync_case(7).name == "ls07" and sync_case(12, "0", 5).name == "lsa12k05"
    assert sync_case(12, "1", 5).name == "lsb12k05" and sync_case(g.FREE_D, free=True).name == "lsfree"


def test_sync_default_d_sweep_is_0_to_24_then_steps_of_two_to_40():
    assert g.SYNC_DEFAULT_DS == tuple(range(25)) + tuple(range(26, 41, 2))


def test_sync_specs_cover_the_requested_grid():
    main = g.sync_specs(ds=[0, 4], ks=[], banks=["none"], free=True)
    assert [s.name for s in main] == ["ls00", "ls04", "lsfree"]
    conf = g.sync_specs(ds=[4, 30], ks=[0, 1], banks=["0", "1"], free=False)
    assert [s.name for s in conf] == ["lsa04k00", "lsa04k01", "lsa30k00", "lsa30k01",
                                      "lsb04k00", "lsb04k01", "lsb30k00", "lsb30k01"]
    with pytest.raises(ValueError):
        g.sync_specs(ds=[41], ks=[], banks=["none"], free=False)


def test_sync_core_acquires_sync_then_event_then_d_nops_then_release_a():
    body = bundles_of(g.sync_core_asm(sync_case(d=9)))
    acq = body.index(f"acq #{g.OWN + g.SYNC_LOCK_SYNC}, r3")
    ev0 = body.index("event #0")
    rel = body.index(f"rel #{g.OWN + g.SYNC_LOCK_A}, r2")
    assert acq < ev0 < rel
    assert body[ev0 + 1:rel] == ["nop"] * 9                       # exactly d nops between the marker and the release
    assert sum(b.startswith("acq ") for b in body) == 1           # the only wait is the pre BD's release
    assert not any(b.startswith("lda r2") for b in body)          # no core memory access in the main sweep


@pytest.mark.parametrize("bank,load", [("0", "lda r26, [p0, #0]"), ("1", "lda r28, [p0, #16]")])
def test_sync_core_conflict_case_loads_k_nops_after_the_release(bank, load):
    body = bundles_of(g.sync_core_asm(sync_case(d=3, bank=bank, k=6)))
    rel = body.index(f"rel #{g.OWN + g.SYNC_LOCK_A}, r2")
    assert body[rel + 1:rel + 7] == ["nop"] * 6 and body[rel + 7] == load
    assert sum(b.startswith("lda r2") for b in body) == 1


def test_sync_core_rejects_out_of_range():
    for bad in (sync_case(d=-1), sync_case(d=41), sync_case(bank="0", k=-1), sync_case(bank="0", k=65)):
        with pytest.raises(ValueError):
            g.sync_core_asm(bad)


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("c", [sync_case(0), sync_case(40), sync_case(5, "0", 3), sync_case(5, "1", 64)])
def test_sync_core_assembles(c, tmp_path):
    s = tmp_path / "c.s"
    s.write_text(g.sync_core_asm(c))
    r = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / "c.o"), str(s)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_sync_design_pre_bd_waits_on_start_releases_sync_and_chains_to_the_waiting_bd():
    t = g.sync_design_mlir(sync_case())
    pre = t.split("^pre:")[1].split("^send:")[0]
    assert pre.index("use_lock(%lStart, AcquireGreaterEqual") < pre.index("dma_bd(%pre") < pre.index("use_lock(%lSync, Release")
    assert "next_bd ^send" in pre
    send = t.split("^send:")[1].split("^recv_start:")[0]
    assert send.index("use_lock(%lA, AcquireGreaterEqual") < send.index("dma_bd(%src")
    assert "use_lock(%lEnd, Release" in send and "use_lock(%lA, Release" not in send   # a BD needs both lock fields
    assert 'aie.lock(%t02, 0) {init = 0 : i32, sym_name = "lA"}' in t          # lock id 0: LOCK_0_REL follows it
    assert "dma_start(MM2S, 0, ^pre," in t
    recv = t.split("^recv:")[1].split("^end:")[0]
    assert "use_lock" not in recv and "memref<68xi32>" in recv                  # both BDs' words, no lock


def test_sync_design_sequence_only_sets_the_start_lock():
    t = g.sync_design_mlir(sync_case())
    seq = t.split("aie.runtime_sequence")[1]
    assert "aiex.set_lock(%lStart" in seq and "aiex.set_lock(%lSync" not in seq and "aiex.set_lock(%lA" not in seq
    assert seq.count("aiex.set_lock") == 1
    for forbidden in ("dma_await", "dma_wait", "dma_memcpy_nd", "dma_start_task", "npu.sync"):
        assert forbidden not in seq
    assert phys_bank(g.FETCH_PRE_ADDR) == 2 and g.FETCH_PRE_ADDR >> 14 == 1       # clear of the measured banks


def test_sync_free_case_starts_lock_a_at_one():
    assert 'sym_name = "lA"' in g.sync_design_mlir(sync_case())
    assert 'aie.lock(%t02, 0) {init = 1 : i32, sym_name = "lA"}' in g.sync_design_mlir(sync_case(g.FREE_D, free=True))


def test_sync_event_lists_per_variant():
    main = g.sync_inject_args("none")
    ev_main = main[main.index("--memmod-sweep-events") + 1].split(",")
    assert ev_main == ["LOCK_0_REL", "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_0_FINISHED_BD", "DMA_S2MM_0_FINISHED_BD"]
    conf = g.sync_inject_args("0")
    assert conf[conf.index("--memmod-sweep-events") + 1].split(",")[-2:] == ["CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1"]
    assert main[main.index("--core-sweep-events") + 1] == "INSTR_EVENT_0,LOCK_STALL"


def sync_synthetic(rel=1000, pre=960, park=6, grant_after=2, fin_after_grant=52, s2mm_after_grant=78,
                   conflict_at=None, waits=True):
    e = held("DMA_MM2S_0_STALLED_LOCK", 100, 940)                   # the pre BD's wait on the start lock
    if waits:
        e += held("DMA_MM2S_0_STALLED_LOCK", pre + park, rel + grant_after - 1)
    grant = rel + grant_after
    e += [ev("DMA_MM2S_0_FINISHED_BD", pre), ev("LOCK_0_REL", rel),
          ev("DMA_MM2S_0_FINISHED_BD", grant + fin_after_grant), ev("DMA_S2MM_0_FINISHED_BD", grant + s2mm_after_grant),
          ev("INSTR_EVENT_0", rel - 12, pkt=0)]
    if conflict_at is not None:
        e.append(ev("CONFLICT_DM_BANK_1", grant + conflict_at))
    return e


def test_sync_row_numbers():
    r = g.sync_row(sync_synthetic(rel=1000, pre=960, park=6, grant_after=3))
    assert r["pre_fin"] == 960 and r["release"] == 1000
    assert r["d"] == 40                                             # release minus the pre BD's finish
    assert r["park"] == 6                                           # the channel's STALLED_LOCK rise after the pre finish
    assert r["grant"] == 3
    assert r["mm2s_after_grant"] == 52 and r["s2mm_after_grant"] == 78
    assert r["fin_after_pre"] == 40 + 3 + 52


def test_sync_row_ignores_the_start_lock_interval():
    r = g.sync_row(sync_synthetic())
    assert r["park"] == 6 and r["grant"] == 2                       # not the interval that ended at 940


def test_sync_row_lock_free_on_request_has_no_wait():
    r = g.sync_row(sync_synthetic(waits=False))
    assert r["park"] is None and r["grant"] is None
    assert r["mm2s_after_grant"] is None and r["fin_after_release"] == 2 + 52   # falls back to the release anchor


def test_sync_row_conflicts_relative_to_the_grant():
    r = g.sync_row(sync_synthetic(grant_after=3, conflict_at=7))
    assert r["conflicts"] == [("CONFLICT_DM_BANK_1", 7)]


def test_sync_row_without_release_or_second_finish_says_so():
    assert g.sync_row([ev("DMA_MM2S_0_FINISHED_BD", 5)])["release"] is None
    r = g.sync_row([ev("LOCK_0_REL", 50), ev("DMA_MM2S_0_FINISHED_BD", 5)])
    assert r["release"] == 50 and r["mm2s_after_grant"] is None and r["fin_after_pre"] is None


def test_format_sync_table_and_summary():
    rows = [(12, "s2.hw", g.sync_row(sync_synthetic(grant_after=3))), (4, "s1.hw", g.sync_row(sync_synthetic())),
            (12, "s1.hw", g.sync_row(sync_synthetic(grant_after=2)))]
    t = g.format_sync_table(rows).splitlines()
    assert t[0].split()[:3] == ["d", "run", "d_meas"]
    assert t[1].split()[:2] == ["4", "s1.hw"] and t[2].split()[:2] == ["12", "s1.hw"] and t[3].split()[:2] == ["12", "s2.hw"]
    assert "grant +3" in t[3] and "mm2s-g +52" in t[3] and "s2mm-g +78" in t[3]
    s = g.format_sync_summary(rows)
    assert "d=12" in s and "s1.hw g+2/f52" in s and "s2.hw g+3/f52" in s


def test_free_case_row_sorts_after_the_numbered_ds_and_does_not_clash_with_the_same_d():
    assert sync_case(g.FREE_D, free=True).label == "free" and sync_case(7).label == 7
    rows = [("free", "s1.hw", g.sync_row(sync_synthetic(waits=False))), (g.FREE_D, "s1.hw", g.sync_row(sync_synthetic())),
            (3, "s1.hw", g.sync_row(sync_synthetic()))]
    t = g.format_sync_table(rows).splitlines()
    assert [ln.split()[0] for ln in t[1:]] == ["3", str(g.FREE_D), "free"]
    assert "lock free on request" in t[3]
    s = g.format_sync_summary(rows)
    assert s.splitlines()[-1].startswith("d=free") and "s1.hw nowait" in s.splitlines()[-1]


def test_sync_row_park_can_precede_the_pre_finish_stamp():
    r = g.sync_row(sync_synthetic(pre=960, park=-1))
    assert r["park"] == -1 and r["grant"] == 2


def test_sync_row_a_stall_that_ended_before_the_release_is_not_a_wait():
    e = sync_synthetic(waits=False) + held("DMA_MM2S_0_STALLED_LOCK", 962, 970)
    assert g.sync_row(e)["grant"] is None


def test_sync_selection_free_only():
    import argparse
    a = argparse.Namespace(free_only=True, ds=[3], ks=[], banks=None, free=False, pre2=0)
    assert [c.name for c in g.sync_selection(a)] == ["lsfree"]
    a = argparse.Namespace(free_only=False, ds=[3], ks=[], banks=None, free=True, pre2=0)
    assert [c.name for c in g.sync_selection(a)] == ["ls03", "lsfree"]


def test_pre2_names_design_and_specs():
    c = g.SyncCase(d=5, pre2=16)
    assert c.name == "lt05" and g.SyncCase(5, "0", 3, pre2=16).name == "lta05k03"
    assert g.SyncCase(g.FREE_D, free=True, pre2=16).name == "ltfree"
    assert [s.name for s in g.sync_specs([1, 2], [], ["none"], True, pre2=16)] == ["lt01", "lt02", "ltfree"]
    t = g.sync_design_mlir(c)
    pre = t.split("^pre:")[1].split("^pre2:")[0]
    assert "use_lock(%lSync, Release" in pre and "next_bd ^pre2" in pre
    pre2 = t.split("^pre2:")[1].split("^send:")[0]
    assert "dma_bd(%pre2 : memref<16xi32>)" in pre2 and "use_lock" not in pre2 and "next_bd ^send" in pre2
    assert "memref<84xi32>" in t.split("^recv:")[1]                        # 64 + 4 + 16 words
    assert g.SYNC_PRE2_ADDR >> 14 == 1 and phys_bank(g.SYNC_PRE2_ADDR) == 2
    assert "^pre2" not in g.sync_design_mlir(g.SyncCase(d=5))              # the single-pre design is unchanged
    with pytest.raises(ValueError):
        g.sync_validate(g.SyncCase(d=1, pre2=g.SYNC_MAX_PRE2 + 1))


@pytest.mark.parametrize("q", [1, 5, 33, 40])
def test_pre2_lengths_beyond_32_and_off_the_granule_are_accepted(q):
    c = g.SyncCase(d=5, pre2=q)
    g.sync_validate(c)
    t = g.sync_design_mlir(c)
    assert f"dma_bd(%pre2 : memref<{q}xi32>)" in t and f"memref<{68 + q}xi32>" in t.split("^recv:")[1]
    # the longest pre2 stays inside its logical bank, clear of the 0x8000 destination
    assert g.SYNC_PRE2_ADDR + 4 * g.SYNC_MAX_PRE2 < g.FETCH_DST_ADDR


def test_sync_row_with_two_pre_bds_measures_d_from_the_last_one():
    e = sync_synthetic(rel=1000, pre=990, park=2, grant_after=2) + [ev("DMA_MM2S_0_FINISHED_BD", 975)]
    r = g.sync_row(e)
    assert r["d"] == 10 and r["d1"] == 25 and r["pre_fin"] == 990
    assert r["grant"] == 2 and r["mm2s_after_grant"] == 52
    e = sync_synthetic(rel=1000, pre=1010, waits=False) + [ev("DMA_MM2S_0_FINISHED_BD", 975)]
    r = g.sync_row(e)
    assert r["d"] == -10 and r["grant"] is None                            # released before the channel asked


def test_load_case_labels_distinguish_k_and_bank():
    assert sync_case(11, "0", 5).label == "a11k05" and g.SyncCase(11, "1", 12, pre2=32).label == "b11k12"
    rows = [(sync_case(11, "1", 2).label, "s1.hw", g.sync_row(sync_synthetic())),
            (sync_case(11, "0", 10).label, "s1.hw", g.sync_row(sync_synthetic()))]
    assert [ln.split()[0] for ln in g.format_sync_table(rows).splitlines()[1:]] == ["a11k10", "b11k02"]


# -- lock request/grant timing family (chain-*) -------------------------------------

def chain(mids="", **kw):
    return g.ChainCase(mids=g.parse_chain_mids(mids.split()), **kw)


def test_chain_mid_specs_parse_words_and_lock_kind():
    assert g.parse_chain_mids(["8", "4c", "2f"]) == (g.ChainBd(8), g.ChainBd(4, "core"), g.ChainBd(2, "free"))
    assert g.format_chain_mids(g.parse_chain_mids(["8", "4c", "2f"])) == "8 4c 2f"
    for bad in (["0"], ["4x"], ["c"], ["4c", "4c"]):
        with pytest.raises(ValueError):
            g.chain_validate(chain(" ".join(bad)))


def test_chain_case_names_carry_the_core_delays_and_stay_clear_of_the_other_families():
    assert chain("4c 8", d1=3, d2=7).name == "q03_07"
    assert chain("8", d2=12).name == "q00_12"
    assert chain("8", releaser="s2mm").name == "qs"
    other = {g.fetch_case_name(k, c, b) for k in range(65) for (c, b) in g.FETCH_PREFIX}
    other |= {g.SyncCase(d).name for d in range(41)} | {g.SyncCase(d, pre2=4).name for d in range(41)}
    assert not {chain("4c", d1=a, d2=b).name for a in range(41) for b in range(41)} & other


def test_chain_validate_limits():
    for bad in (chain("4", d1=1), chain("4c", d1=41), chain("4", d2=-1), chain(big=0), chain(big=257),
                chain(head=0), chain(" ".join(["1f"] * 11)), chain(" ".join(["1"] * 14)),
                chain(" ".join(["1"] * 13), releaser="s2mm"),
                chain("4c", releaser="s2mm"), chain("4", releaser="x")):
        with pytest.raises(ValueError):
            g.chain_validate(bad)
    g.chain_validate(chain(" ".join(["1f"] * 10)))
    g.chain_validate(chain(" ".join(["1"] * 13)))                # head + 13 + send + recv = the 16 BDs


def bd_blocks(text):
    """{block name: block text} of the aie.mem body."""
    mem = text.split("%mem = aie.mem(%t02) {")[1].split("aie.runtime_sequence")[0]
    out = {}
    for part in mem.split("\n    ^")[1:]:
        name, body = part.split(":", 1)
        out[name] = body
    return out


def test_chain_design_bd_order_locks_and_sizes():
    t = g.chain_design_mlir(chain("4c 8 2f", big=16))
    b = bd_blocks(t)
    assert "aie.dma_start(MM2S, 0, ^head, ^recv_start)" in t
    assert "aie.use_lock(%lStart, AcquireGreaterEqual" in b["head"] and "aie.use_lock(%lSync, Release" in b["head"]
    assert "memref<4xi32>" in b["head"] and "aie.next_bd ^m0" in b["head"]
    assert "aie.use_lock(%lB, AcquireGreaterEqual" in b["m0"] and "aie.use_lock(%lB, Release" in b["m0"]
    assert "memref<4xi32>" in b["m0"] and "aie.next_bd ^m1" in b["m0"]
    assert "use_lock" not in b["m1"] and "memref<8xi32>" in b["m1"] and "aie.next_bd ^m2" in b["m1"]
    assert "aie.use_lock(%lF0, AcquireGreaterEqual" in b["m2"] and "aie.use_lock(%lF0, Release" in b["m2"]
    assert "aie.next_bd ^send" in b["m2"]
    assert "aie.use_lock(%lA, AcquireGreaterEqual" in b["send"] and "memref<16xi32>" in b["send"]
    assert "aie.use_lock(%lEnd, Release" in b["send"]
    assert "memref<34xi32>" in b["recv"] and "use_lock" not in b["recv"]          # 4 + 4 + 8 + 2 + 16
    assert f'aie.lock(%t02, {g.CHAIN_LOCK_B}) {{init = 0 : i32, sym_name = "lB"}}' in t
    assert f'aie.lock(%t02, {g.CHAIN_LOCK_FREE0}) {{init = 1 : i32, sym_name = "lF0"}}' in t
    assert 'aie.lock(%t02, 0) {init = 0 : i32, sym_name = "lA"}' in t


def test_chain_design_buffers_are_disjoint_and_clear_of_banks_0_1_and_the_destination():
    c = chain("4c 40 33 2f 1f")
    spans = g.chain_buffers(c)
    flat = sorted((a, a + 4 * w) for a, w in spans)
    assert all(e <= s for (_, e), (s, _) in zip(flat, flat[1:]))
    assert all(a % 32 == 0 and 0x4000 <= a and a + 4 * w <= 0x8000 for a, w in spans)


def test_chain_design_s2mm_releaser_splits_the_receive_and_releases_a():
    t = g.chain_design_mlir(chain("8 4", releaser="s2mm"))
    b = bd_blocks(t)
    assert "aie.use_lock(%lRx, AcquireGreaterEqual" in b["recv"] and "aie.use_lock(%lA, Release" in b["recv"]
    assert "memref<16xi32>" in b["recv"] and "aie.next_bd ^recv2" in b["recv"]     # head 4 + 8 + 4
    assert "memref<64xi32>" in b["recv2"] and "use_lock" not in b["recv2"]
    assert f'aie.lock(%t02, {g.CHAIN_LOCK_RX}) {{init = 1 : i32, sym_name = "lRx"}}' in t


def test_chain_bd_ids_pin_each_mm2s_bd_and_leave_the_default_design_unchanged():
    c = chain("1 1", bd_ids=(0, 2, 3, 7))
    b = bd_blocks(g.chain_design_mlir(c))
    for name, i in (("head", 0), ("m0", 2), ("m1", 3), ("send", 7)):
        assert f"{{bd_id = {i} : i32}}" in b[name], name
    assert "bd_id" not in b["recv"]                                  # the S2MM BD stays aiecc's
    assert c.design()["bd_ids"] == [0, 2, 3, 7]
    plain = chain("1 1")
    assert "bd_id" not in g.chain_design_mlir(plain) and "bd_ids" not in plain.design()


def test_chain_bd_ids_must_name_every_mm2s_bd_once_within_the_tile():
    for bad in ((0, 1, 2), (0, 1, 2, 3, 4), (0, 1, 1, 2), (0, 1, 2, 16), (0, 1, 2, -1)):
        with pytest.raises(ValueError):
            g.chain_validate(chain("1 1", bd_ids=bad))
    g.chain_validate(chain("1 1", bd_ids=(9, 0, 15, 4)))


def test_chain_bd_ids_parse_from_the_command_line():
    args = type("A", (), {"mids": ["1", "1"], "head": 4, "big": 64, "d1s": [], "d2s": [0, 1], "releaser": "core",
                          "bd_ids": [0, 2, 3, 7]})()
    cases = g.chain_selection(args)
    assert [c.bd_ids for c in cases] == [(0, 2, 3, 7)] * 2 and [c.d2 for c in cases] == [0, 1]


def test_chain_design_sequence_only_sets_the_start_lock():
    for c in (chain("4c 8"), chain("8", releaser="s2mm")):
        seq = g.chain_design_mlir(c).split("aie.runtime_sequence")[1]
        assert seq.count("aiex.set_lock") == 1 and "aiex.set_lock(%lStart" in seq
        for forbidden in ("dma_await", "dma_wait", "dma_memcpy_nd", "dma_start_task", "npu.sync"):
            assert forbidden not in seq


def test_chain_core_releases_b_then_a_after_the_sync_lock():
    body = bundles_of(g.chain_core_asm(chain("4c 8", d1=3, d2=5)))
    acq = body.index(f"acq #{g.OWN + g.SYNC_LOCK_SYNC}, r3")
    e0, e1 = body.index("event #0"), body.index("event #1")
    rb, ra = body.index(f"rel #{g.OWN + g.CHAIN_LOCK_B}, r2"), body.index(f"rel #{g.OWN + g.SYNC_LOCK_A}, r2")
    assert acq < e0 < rb < e1 < ra
    assert body[e0 + 1:rb] == ["nop"] * 3 and rb + 1 == e1 and body[e1 + 1:ra] == ["nop"] * 5


def test_chain_core_without_a_core_bd_releases_only_a_and_s2mm_releaser_none():
    body = bundles_of(g.chain_core_asm(chain("8", d2=4)))
    assert not any(x.startswith(f"rel #{g.OWN + g.CHAIN_LOCK_B}") for x in body)
    e1, ra = body.index("event #1"), body.index(f"rel #{g.OWN + g.SYNC_LOCK_A}, r2")
    assert body.index("event #0") + 1 == e1 and body[e1 + 1:ra] == ["nop"] * 4
    s = bundles_of(g.chain_core_asm(chain("8", releaser="s2mm")))
    assert not any(x.startswith("rel #") for x in s) and "event #0" in s


def test_chain_fill_releases_a_after_the_tail_and_leaves_the_default_unchanged():
    rel_a = f"rel #{g.OWN + g.SYNC_LOCK_A}, r2"
    plain = [ln.strip() for ln in g.chain_core_asm(chain("1 1", d2=3)).splitlines()]
    filled = [ln.strip() for ln in g.chain_core_asm(chain("1 1", d2=3, fill=5)).splitlines()]
    assert plain.count(rel_a) == 1 and filled.count(rel_a) == 6
    rels = [i for i, x in enumerate(filled) if x == rel_a]
    tail, tail2 = filled.index(".Ltail:"), filled.index(".Ltail2:")
    assert rels[0] < tail and all(tail < i < tail2 for i in rels[1:])
    assert ".Ltail2:" not in plain
    gaps = [b - a - 1 for a, b in zip(rels[1:], rels[2:])]                # nops between fill releases
    assert gaps == [g.CHAIN_FILL_GAP_NOPS + i for i in range(1, 5)]       # growing: no two fill frames alike
    assert g.chain_design_mlir(chain("1 1")) == g.chain_design_mlir(chain("1 1", fill=5))
    assert "fill" not in chain("1 1").design() and chain("1 1", fill=5).design()["fill"] == 5


def test_chain_fill_limits_and_command_line():
    with pytest.raises(ValueError):
        g.chain_validate(chain("1", fill=g.CHAIN_MAX_FILL + 1))
    with pytest.raises(ValueError):
        g.chain_validate(chain("1", releaser="s2mm", fill=1))
    args = type("A", (), {"mids": ["1"], "head": 8, "big": 64, "d1s": [], "d2s": [0], "releaser": "core",
                          "bd_ids": [], "fill": 24})()
    assert [(c.fill, c.head) for c in g.chain_selection(args)] == [(24, 8)]


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("c", [chain("4c 8", d1=0, d2=0), chain("4c", d1=40, d2=40), chain("8", releaser="s2mm"),
                               chain("1 1 1 1 1 1 1 1", fill=g.CHAIN_MAX_FILL)])
def test_chain_core_assembles(c, tmp_path):
    s = tmp_path / "c.s"
    s.write_text(g.chain_core_asm(c))
    r = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / "c.o"), str(s)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_chain_event_lists():
    a = g.chain_inject_args()
    assert a[a.index("--memmod-sweep-events") + 1].split(",") == [
        "LOCK_0_REL", "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_0_FINISHED_BD", "DMA_S2MM_0_FINISHED_BD", "LOCK_SEL0_ACQ_GE"]
    assert a[a.index("--core-sweep-events") + 1] == "INSTR_EVENT_0,INSTR_EVENT_1,LOCK_STALL"


def test_chain_row_is_relative_to_the_head_grant_and_keeps_every_stall_run():
    e = held("DMA_MM2S_0_STALLED_LOCK", 100, 499)                  # head's wait for START: grant 500
    e += held("DMA_MM2S_0_STALLED_LOCK", 504, 515)                 # a core BD's wait
    e += held("DMA_MM2S_0_STALLED_LOCK", 530, 530)                 # a one-cycle blip
    e += [ev("DMA_MM2S_0_FINISHED_BD", t) for t in (503, 519, 590)] + [ev("LOCK_0_REL", 527)]
    e += [ev("LOCK_SEL0_ACQ_GE", 531), ev("DMA_S2MM_0_FINISHED_BD", 620)]
    e += [ev("INSTR_EVENT_0", 509, pkt=0), ev("INSTR_EVENT_1", 514, pkt=0), ev("INSTR_EVENT_0", 900, pkt=0)]
    r = g.chain_row(e)
    assert r["g0"] == 500
    assert r["stalls"] == [(4, 15), (30, 30)] and r["fins"] == [3, 19, 90] and r["rel_a"] == [27]
    assert r["acq_a"] == [31] and r["s2mm_fins"] == [120] and r["ev0"] == 9 and r["ev1"] == 14


def test_chain_row_without_a_head_wait_says_so():
    assert g.chain_row([ev("LOCK_0_REL", 10)]) == {"g0": None}


def test_format_chain_table_one_line_per_case_and_run():
    r = {"g0": 500, "stalls": [(4, 15)], "fins": [3, 19, 90], "rel_a": [27], "acq_a": [], "s2mm_fins": [120],
         "ev0": 9, "ev1": 14}
    text = g.format_chain_table([(chain("4c", d1=1, d2=2), "p1.hw", r), (chain("4c", d1=0, d2=2), "p1.hw", {"g0": None})])
    lines = text.splitlines()
    assert lines[1].startswith("q00_02 p1.hw") and "no head wait" in lines[1]
    assert lines[2].startswith("q01_02 p1.hw") and "stalls [+4..+15]" in lines[2] and "fins [+3 +19 +90]" in lines[2]
    assert "relA [+27]" in lines[2] and "ev0 +9 ev1 +14" in lines[2]


def test_memmod_event_add_repeats_in_order_and_keeps_the_single_form():
    ns = g.build_parser().parse_args(["chain-build", "d", "--memmod-event-add", "DMA_S2MM_0_STREAM_STARVATION",
                                      "--memmod-event-add", "DMA_MM2S_0_MEMORY_STARVATION"])
    assert ns.memmod_event_add == ["DMA_S2MM_0_STREAM_STARVATION", "DMA_MM2S_0_MEMORY_STARVATION"]
    ns = g.build_parser().parse_args(["fetch-build", "d", "--memmod-event-add", "LOCK_1_REL"])
    assert ns.memmod_event_add == "LOCK_1_REL"
    base = list(g.CHAIN_MEM_EVENTS)
    two = g.edit_memmod_events(base, None, ns2 := ["DMA_S2MM_0_STREAM_STARVATION", "DMA_MM2S_0_MEMORY_STARVATION"])
    assert two == [*base, *ns2]
    assert g.edit_memmod_events(base, None, "LOCK_1_REL") == [*base, "LOCK_1_REL"]
    with pytest.raises(ValueError):
        g.edit_memmod_events(base, None, ["LOCK_1_REL", "LOCK_1_REL"])
    with pytest.raises(ValueError):
        g.edit_memmod_events(base, None, ["LOCK_1_REL", "LOCK_2_REL", "LOCK_3_REL", "LOCK_4_REL"])   # 9 > 8 slots
    a = g.chain_inject_args(None, ns2)
    assert a[a.index("--memmod-sweep-events") + 1].split(",") == [*base, *ns2]


def test_chain_remote_receiver_moves_only_the_s2mm_to_the_tile_above():
    own, north = chain("1 1"), chain("1 1", receiver="north")
    assert own.design() == chain("1 1").design() and "receiver" not in own.design()
    assert north.design()["receiver"] == "north" and north.name == own.name
    m = g.chain_design_mlir(north)
    assert "%t03 = aie.tile(0, 3)" in m
    assert "aie.flow(%t02, DMA : 0, %t03, DMA : 0)" in m and "aie.flow(%t02, DMA : 0, %t02, DMA : 0)" not in m
    mem02 = m.split("%mem = aie.mem(%t02)")[1].split("aie.end")[0]
    mem03 = m.split("aie.mem(%t03)")[1].split("aie.end")[0]
    assert "S2MM" not in mem02 and "MM2S" in mem02
    assert "aie.dma_start(S2MM, 0" in mem03 and "%dst : memref<70xi32>" in mem03     # head 4 + two 1-word mids + 64
    assert 'aie.buffer(%t03) {address = ' in m and 'sym_name = "dst"' in m
    # the MM2S side is the same program
    assert g.chain_core_asm(north) == g.chain_core_asm(own)
    with pytest.raises(ValueError):
        g.chain_validate(chain("1", receiver="north", releaser="s2mm"))
    with pytest.raises(ValueError):
        g.chain_validate(chain("1", receiver="east"))
    args = type("A", (), {"mids": ["1"], "head": 4, "big": 64, "d1s": [], "d2s": [0], "releaser": "core",
                          "bd_ids": [], "fill": 0, "receiver": "north"})()
    assert [c.receiver for c in g.chain_selection(args)] == ["north"]


def test_chain_remote_receiver_fill_also_releases_a_lock_of_the_tile_above():
    rel_a, rel_n = f"rel #{g.OWN + g.SYNC_LOCK_A}, r2", f"rel #{g.NORTH + g.CHAIN_LOCK_NORTH_FILL}, r2"
    own = [ln.strip() for ln in g.chain_core_asm(chain("1 1", fill=3)).splitlines()]
    north = [ln.strip() for ln in g.chain_core_asm(chain("1 1", fill=3, receiver="north")).splitlines()]
    assert rel_n not in own and own.count(rel_a) == 4
    assert north.count(rel_a) == 4 and north.count(rel_n) == 3
    assert all(north[i + 1] == rel_n for i, x in enumerate(north) if x == rel_a and i > north.index(".Ltail:"))
    m = g.chain_design_mlir(chain("1 1", fill=3, receiver="north"))
    assert f"aie.lock(%t03, {g.CHAIN_LOCK_NORTH_FILL})" in m and "aie.lock(%t03" not in g.chain_design_mlir(
        chain("1 1", fill=3))
    assert "aie.lock(%t03" not in g.chain_design_mlir(chain("1 1", receiver="north"))      # no fill, no lock


def test_chain_receiver_two_and_three_tiles_north():
    for rcv, row in (("north2", 4), ("north3", 5)):
        c = chain("1 1", receiver=rcv, fill=3)
        m = g.chain_design_mlir(c)
        t = f"%t0{row}"
        assert f"{t} = aie.tile(0, {row})" in m and f"aie.flow(%t02, DMA : 0, {t}, DMA : 0)" in m
        assert f"aie.buffer({t}) {{address = " in m and f"aie.mem({t})" in m
        assert "aie.lock(%t03" not in m                                   # only an adjacent tile's lock is filled
        assert f"rel #{g.NORTH + g.CHAIN_LOCK_NORTH_FILL}, r2" not in g.chain_core_asm(c)
        assert c.design()["receiver"] == rcv
    e = held("DMA_MM2S_0_STALLED_LOCK", 100, 499) + [ev("DMA_MM2S_0_FINISHED_BD", 503)]
    r = g.chain_row(e + [ev("DMA_S2MM_0_FINISHED_BD", 650, row=5)])
    assert r["s2mm_fins"] == [150]


def test_chain_receiver_south_is_the_memtile_s2mm():
    c = chain("1 1", receiver="south", fill=3)
    m = g.chain_design_mlir(c)
    assert "%t01 = aie.tile(0, 1)" in m and "aie.flow(%t02, DMA : 0, %t01, DMA : 0)" in m
    mt = m.split("aie.memtile_dma(%t01)")[1].split("aie.end")[0]
    assert "aie.dma_start(S2MM, 0" in mt and "%dst : memref<70xi32>" in mt
    assert "aie.buffer(%t01)" in m and "S2MM" not in m.split("%mem = aie.mem(%t02)")[1].split("aie.end")[0]
    assert f"rel #{g.NORTH + g.CHAIN_LOCK_NORTH_FILL}, r2" not in g.chain_core_asm(c)    # no fill on the memtile
    a = g.chain_inject_args(None, None, receiver="south")
    assert a[a.index("--memtile-sweep-events") + 1].split(",") == g.CHAIN_MEMTILE_EVENTS
    assert "--memtile-sweep-events" not in g.chain_inject_args()
    e = held("DMA_MM2S_0_STALLED_LOCK", 100, 499) + [ev("DMA_MM2S_0_FINISHED_BD", 503)]
    e += [ev("DMA_S2MM_SEL0_FINISHED_BD", 640, pkt=3, row=1)] + held("DMA_S2MM_SEL0_STREAM_STARVATION", 600, 610, pkt=3, row=1)
    r = g.chain_row(e)
    assert r["s2mm_fins"] == [140] and r["s2mm_starv_remote"] == [(100, 110)]


def test_chain_row_takes_a_remote_s2mm_and_the_starvation_levels():
    e = held("DMA_MM2S_0_STALLED_LOCK", 100, 499)
    e += [ev("DMA_MM2S_0_FINISHED_BD", 503), ev("LOCK_0_REL", 515), ev("LOCK_SEL0_ACQ_GE", 516)]
    e += held("DMA_S2MM_0_STREAM_STARVATION", 505, 507) + held("DMA_S2MM_0_STREAM_STARVATION", 512, 512)
    e += held("DMA_MM2S_0_MEMORY_STARVATION", 509, 510)
    r = g.chain_row(e)
    assert r["s2mm_starv"] == [(5, 7), (12, 12)] and r["mm2s_mstarv"] == [(9, 10)] and r["s2mm_fins"] == []
    remote = e + [ev("DMA_S2MM_0_FINISHED_BD", 640, row=3)] + held("DMA_S2MM_0_STREAM_STARVATION", 620, 621, row=3)
    r = g.chain_row(remote)
    assert r["s2mm_fins"] == [140] and r["s2mm_starv_remote"] == [(120, 121)]
    plain = held("DMA_MM2S_0_STALLED_LOCK", 100, 499) + [ev("DMA_MM2S_0_FINISHED_BD", 503)]
    assert "s2mm_starv" not in g.chain_row(plain)


def test_format_chain_table_shows_the_starvation_levels_when_traced():
    r = {"g0": 500, "stalls": [], "fins": [3], "rel_a": [27], "acq_a": [], "s2mm_fins": [120], "ev0": 9, "ev1": 14,
         "s2mm_starv": [(5, 7)], "mm2s_mstarv": [(9, 10)]}
    line = g.format_chain_table([(chain("1"), "e1.emu", r)]).splitlines()[1]
    assert "s2starv [+5..+7]" in line and "mstarv [+9..+10]" in line


def test_chain_build_refuses_a_directory_built_for_another_design(tmp_path):
    (tmp_path / "design.json").write_text('{"head": 4, "mids": "8", "big": 64, "releaser": "core"}\n')
    with pytest.raises(ValueError):
        g.chain_check_design_dir(tmp_path, chain("4c 8"))
    g.chain_check_design_dir(tmp_path, chain("8"))


# -- bank arbitration probe (arb-*) ---------------------------------------------------

def arb(bank=0, op="l", n=1, k=16, m=0):
    return g.ArbCase(bank, op, n, k, m)


@pytest.mark.parametrize("bank,op,n,k,m", [(0, "l", 1, 8, 0), (0, "l", 4, 16, 2), (1, "s", 6, 28, 0),
                                           (0, "s", 3, 0, 4)])
def test_arb_core_order_settle_loads_marker_release_nops_accesses_marker(bank, op, n, k, m):
    body = bundles_of(g.arb_core_asm(arb(bank, op, n, k, m)))
    i = body.index("event #0")
    assert body[i + 1] == f"rel #{g.OWN + g.FETCH_LOCK_A}, r2"
    assert body[i + 2:i + 2 + k] == ["nop"] * k
    acc = body[i + 2 + k:i + 2 + k + n]
    verb = "lda" if op == "l" else "st r2,"
    assert all(a.startswith(verb) for a in acc) and len(acc) == n
    offs = [int(a.split("#")[1].rstrip("]")) for a in acc]
    assert [phys_bank(g.FETCH_LD_ADDR + o) for o in offs] == [bank] * n      # all on the one physical bank
    j = i + 2 + k + n
    assert body[j:j + g.ARB_EVENT1_NOPS] == ["nop"] * g.ARB_EVENT1_NOPS and body[j + g.ARB_EVENT1_NOPS] == "event #1"
    settle = [b for b in body[:i] if b.startswith("lda r25,")]
    assert len(settle) == m
    assert all(phys_bank(g.FETCH_LD_ADDR + int(s.split("#")[1].rstrip("]"))) == bank for s in settle)
    assert body.index(f"acq #{g.OWN + g.FETCH_LOCK_GO}, r3") < i                # settle loads after the go lock
    assert sum(b.startswith("acq ") for b in body) == 2                         # go and B only
    assert sum(b.startswith("rel ") for b in body) == 1
    # nothing else touches data memory between the go lock and the wait for B
    go, done = body.index(f"acq #{g.OWN + g.FETCH_LOCK_GO}, r3"), body.index(f"acq #{g.OWN + g.FETCH_LOCK_B}, r3")
    mem = [b for b in body[go:done] if b.startswith(("lda", "st "))]
    assert len(mem) == m + n


def test_arb_rejects_out_of_range():
    for c in (arb(bank=2), arb(op="x"), arb(n=0), arb(n=g.ARB_MAX_N + 1), arb(k=65), arb(m=g.ARB_MAX_M + 1)):
        with pytest.raises(ValueError):
            g.arb_core_asm(c)


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("c", [arb(0, "l", 6, 16, 4), arb(1, "s", 6, 28, 0)])
def test_arb_core_assembles(c, tmp_path):
    s = tmp_path / "core.s"
    s.write_text(g.arb_core_asm(c))
    r = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / "core.o"), str(s)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_arb_plan_names_are_distinct_valid_and_clear_of_other_variants():
    plan = g.arb_plan()
    names = [c.name for c in plan]
    assert len(names) == len(set(names)) == 56
    for c in plan:
        g.arb_validate(c)
    assert all(n.startswith("ar") and not n.startswith(("lf", "ls", "lt")) for n in names)
    assert arb(1, "s", 3, 28, 2).name == "ar1s3k28m2"


def test_arb_inject_args_trace_both_core_markers_and_memory_stall():
    args = g.arb_inject_args()
    assert args[args.index("--core-sweep-events") + 1] == "INSTR_EVENT_0,INSTR_EVENT_1,MEMORY_STALL"
    assert args[args.index("--memmod-sweep-events") + 1] == ",".join(g.FETCH_MEMMOD_EVENTS)


# The rotation rule against the silicon rows it was read from (lock-fetch, single core loads).
@pytest.mark.parametrize("bank,cell,lost", [
    (0, 2, False), (0, 4, False), (0, 7, True), (0, 15, False), (0, 23, False), (0, 31, False), (0, 39, False),
    (1, 3, False), (1, 5, False), (1, 11, False), (1, 19, False), (1, 27, True), (1, 35, False), (1, 43, False),
    (1, 51, False)])
def test_rotation_rule_reproduces_the_lock_fetch_rows(bank, cell, lost):
    p = g.rotation_prediction(arb(bank=bank), cell)
    assert p["contended"] == [cell]
    assert p["core_lost"] == ([cell] if lost else [])


def test_rotation_rule_predicts_a_bank0_loss_at_g47():
    assert g.rotation_prediction(arb(bank=0), 47)["core_lost"] == [47]


def test_rotation_rule_bursts_against_g15_the_dma_waits_at_most_four_cycles():
    for n in range(1, 5):
        p = g.rotation_prediction(arb(n=n), 15)
        assert p["contended"] == list(range(15, 15 + n)) and p["core_lost"] == []
        assert (15, 15 + n) in p["served"]                       # the DMA gets the bank when the core is done
    p = g.rotation_prediction(arb(n=5), 15)
    assert p["contended"] == [15, 16, 17, 18, 19] and p["core_lost"] == [19]
    p = g.rotation_prediction(arb(n=6), 15)
    assert p["core_lost"] == [19] and p["contended"] == [15, 16, 17, 18, 19]   # sixth access runs alone at 20


def test_rotation_rule_history_moves_the_losing_access():
    assert g.rotation_prediction(arb(m=4), 15)["core_lost"] == [15]
    assert all(g.rotation_prediction(arb(m=m), 15)["core_lost"] == [] for m in range(4))
    assert all(g.rotation_prediction(arb(m=m), 7)["core_lost"] == [] for m in range(1, 5))


def test_rotation_rule_uncontended_core_accesses_count():
    # starting one cycle before the DMA's g+15 access: the first load runs alone and advances the count
    p = g.rotation_prediction(arb(n=5), 14)
    assert p["contended"] == [15, 16, 17, 18] and p["core_lost"] == [18]


def test_arb_row_reads_the_core_window_and_memory_stalls():
    e = synthetic(rel=1000, grant_after=2, conflict_bank="CONFLICT_DM_BANK_0", conflict_at=16)
    e += [ev("INSTR_EVENT_1", 1000 + 4 + 24, pkt=0), ev("MEMORY_STALL", 1000 + 4 + 17, pkt=0)]
    r = g.arb_row(e)
    assert r["grant"] == 2 and r["window"] == 24 and r["memory_stalls"] == [17]
    c = arb(n=1, k=16)
    body = bundles_of(g.arb_core_asm(c))
    assert g.arb_clean_window(c) == body.index("event #1") - body.index("event #0")
    assert g.arb_start(c, 2) == 15 and g.arb_start(c, 3) == 14
    text = g.format_arb_table([(c, "a1.hw", r)])
    assert "ar0l1k16m0" in text and "b0@14" in text and "(+1)" in text


def test_arb_matches_compares_contended_cycles_and_core_stall_count():
    r = {"grant": 2, "conflicts": [("CONFLICT_DM_BANK_0", 9)], "memory_stalls": [15]}
    assert g.arb_matches(r, {"contended": [7], "core_lost": [7]})
    assert not g.arb_matches(r, {"contended": [7], "core_lost": []})
    assert not g.arb_matches(r, {"contended": [7, 8], "core_lost": [7]})


def test_rotation_rule_bank1_phases_in_the_plan():
    assert g.rotation_prediction(arb(bank=1, n=1, k=20), 19)["core_lost"] == []
    assert g.rotation_prediction(arb(bank=1, n=2, k=20), 19)["core_lost"] == [20]
    assert g.rotation_prediction(arb(bank=1, n=4, k=36), 35)["core_lost"] == []
    assert g.rotation_prediction(arb(bank=1, n=5, k=36), 35)["core_lost"] == [39]


# -- ring-order probe (ring-*) ----------------------------------------------------------

def ring(variant="m0", **kw):
    return g.RingCase(variant, **kw)


def test_ring_core_camps_on_one_bank_from_the_release_then_fills_the_trace():
    c = ring(n=10, k=0)
    body = bundles_of(g.ring_core_asm(c))
    i = body.index("event #0")
    assert body[i + 1] == f"rel #{g.OWN + g.FETCH_LOCK_A}, r4" and "mov r4, #1" in body
    camp = body[i + 2:i + 12]
    assert all(b.startswith("lda ") for b in camp)
    assert {phys_bank(g.FETCH_LD_ADDR + int(b.split("#")[1].rstrip("]"))) for b in camp} == {0}
    assert body[i + 12:i + 12 + g.ARB_EVENT1_NOPS] == ["nop"] * g.ARB_EVENT1_NOPS
    done = body.index(f"acq #{g.OWN + g.FETCH_LOCK_B}, r3")
    assert sum(b == f"rel #{g.OWN + g.FETCH_LOCK_A}, r2" for b in body[done:]) == g.RING_FILL_RELEASES
    assert sum(b.startswith("acq ") for b in body) == 2
    go = body.index(f"acq #{g.OWN + g.FETCH_LOCK_GO}, r3")
    assert not [b for b in body[go:i] if b.startswith(("lda", "st "))]       # no history access before the camp


def test_ring_bank1_camp_offsets():
    body = bundles_of(g.ring_core_asm(ring(bank=1, n=9)))
    loads = [b for b in body if b.startswith("lda r2") or b.startswith("lda r3")]
    assert {phys_bank(g.FETCH_LD_ADDR + int(b.split("#")[1].rstrip("]"))) for b in loads} == {1}


def test_ring_rejects_out_of_range():
    for c in (ring("x"), ring(bank=2), ring(n=0), ring(n=g.RING_MAX_N + 1), ring(pad=g.RING_MAX_PAD + 1)):
        with pytest.raises(ValueError):
            g.ring_core_asm(c)


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
def test_ring_core_assembles(tmp_path):
    s = tmp_path / "core.s"
    s.write_text(g.ring_core_asm(ring(n=g.RING_MAX_N)))
    r = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / "core.o"), str(s)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_ring_geometry():
    assert {phys_bank(g.RING_SDST_ADDR + 16 * i) for i in range(16)} == {0, 1}      # S2MM target: banks 0/1
    assert {phys_bank(g.RING_FEED_ADDR + 16 * i) for i in range(16)} == {2, 3}      # feed: clear of 0/1
    assert all(phys_bank(a + 4 * j) == 0 for a in g.RING_PAD_ADDRS for j in range(4))
    assert g.FETCH_LD_ADDR + 64 <= g.RING_PAD_ADDRS[0] and g.RING_PAD_ADDRS[1] + 16 <= g.RING_SDST_ADDR


@pytest.mark.parametrize("variant,mm2s,s2mm,source,sdst", [("m0", 0, 0, "%src", False), ("m1", 1, 0, "%src", False),
                                                           ("s0", 0, 0, "%feed", True), ("s1", 0, 1, "%feed", True)])
def test_ring_design_puts_one_channel_on_banks_0_1(variant, mm2s, s2mm, source, sdst):
    t = g.ring_design_mlir(ring(variant))
    assert f"aie.flow(%t02, DMA : {mm2s}, %t02, DMA : {s2mm})" in t
    assert f"aie.dma_start(MM2S, {mm2s}, ^pre" in t and f"aie.dma_start(S2MM, {s2mm}, ^recv" in t
    send = t.split("^send:")[1].split("^recv_start:")[0]
    assert f"aie.dma_bd({source} :" in send
    recv = t.split("^recv:")[1].split("^end:")[0]
    assert ("dma_bd(%sdst" in recv) == sdst
    assert "use_lock" not in recv                                               # the S2MM never waits
    # the S2MM takes exactly what the MM2S sends: 4 + 64 words
    import re
    sizes = [int(x) for x in re.findall(r"dma_bd\(%\w+ : memref<\d+xi32> offset = \d+ len = (\d+)\)", recv)]
    assert sizes
    assert sum(sizes) == g.FETCH_PRE_WORDS + g.FETCH_WORDS
    seq = t.split("aie.runtime_sequence")[1]
    for forbidden in ("dma_wait", "dma_memcpy_nd", "npu.sync"):
        assert forbidden not in seq


def test_ring_variants_share_buffers_and_initial_data():
    def bufs(t):
        return [ln.strip() for ln in t.splitlines() if "aie.buffer" in ln]
    base = bufs(g.ring_design_mlir(ring("m0")))
    for v in ("m1", "s0", "s1"):
        assert bufs(g.ring_design_mlir(ring(v))) == base
    addrs = [int(ln.split("address = ")[1].split(" ")[0]) for ln in base]
    assert addrs == sorted(addrs)                                               # ascending for the allocator


@pytest.mark.parametrize("pad,words", [(0, []), (3, [3]), (4, [4]), (5, [4, 1]), (8, [4, 4])])
def test_ring_pad_words_in_bank0(pad, words):
    t = g.ring_design_mlir(ring(pad=pad))
    found = [int(ln.split("memref<")[1].split("x")[0]) for ln in t.splitlines() if 'sym_name = "pad' in ln]
    assert found == words


def test_ring_event_lists():
    assert g.ring_memmod_events(ring("m1")) == ["LOCK_0_REL", "DMA_MM2S_1_STALLED_LOCK", "DMA_MM2S_1_FINISHED_BD",
                                                "DMA_S2MM_0_FINISHED_BD", "CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1",
                                                "DMA_MM2S_1_MEMORY_STARVATION"]
    s1 = g.ring_memmod_events(ring("s1"))
    assert "DMA_S2MM_1_MEMORY_BACKPRESSURE" in s1 and "DMA_S2MM_1_FINISHED_BD" in s1 and len(s1) == 8
    assert all(len(g.ring_memmod_events(ring(v))) <= 8 for v in g.RING_VARIANTS)


def test_ring_row_and_residues():
    rel = 1000
    e = held("DMA_MM2S_0_STALLED_LOCK", 200, rel + 1) + [ev("LOCK_0_REL", rel), ev("LOCK_0_REL", rel + 500)]
    # camp starts at rel + 1; runs on bank 0 at camp+3..+5 (DMA let in at +5) and +10..+11 (in at +11)
    e += held("CONFLICT_DM_BANK_0", rel + 4, rel + 6) + held("CONFLICT_DM_BANK_0", rel + 11, rel + 12)
    e += [ev("MEMORY_STALL", rel + 1 + 5 + 1, pkt=0), ev("MEMORY_STALL", rel + 1 + 11 + 1, pkt=0)]
    e += [ev("INSTR_EVENT_0", rel - 1, pkt=0), ev("INSTR_EVENT_1", rel + 40, pkt=0)]
    c = ring(n=20)
    r = g.ring_row(e, c)
    assert r["grant"] == 2 and r["lost"] == [5, 11] and r["runs"][0] == [(3, 5), (10, 11)]
    res = g.ring_residues(r, c)
    assert res["dma_wins"] == [0, 1] and res["dma_refused"] == [0, 3, 4] and res["withdrew"] == []
    assert "rgm0b0n020k00p0" in g.format_ring_table([(c, "r1.hw", r)])


def test_ring_plan():
    names = [c.name for c in g.ring_plan()]
    assert len(names) == len(set(names)) == 15 and all(n.startswith("rg") for n in names)


def test_ring_two_channel_variant():
    c = ring("ms")
    t = g.ring_design_mlir(c)
    assert "aie.flow(%t02, DMA : 0, %t02, DMA : 0)" in t and "aie.flow(%t02, DMA : 1, %t02, DMA : 1)" in t
    assert t.count("aie.use_lock(%lA, AcquireGreaterEqual") == 2                 # both MM2S wait on A
    assert "aie.dma_start(MM2S, 1, ^send1" in t and "aie.dma_bd(%feed" in t.split("^send1:")[1].split("^s1_start")[0]
    assert "dma_bd(%sdst" in t.split("^recv1:")[1] and "dma_bd(%sdst" not in t.split("^recv:")[1].split("^m1_start")[0]
    body = bundles_of(g.ring_core_asm(c))
    assert "mov r4, #2" in body                                                # A released by 2: one grant each
    assert len(g.ring_memmod_events(c)) == 8 and "DMA_S2MM_1_FINISHED_BD" in g.ring_memmod_events(c)


def test_ring_pad_one_buffer_spans_banks():
    c = ring(pad=12, pad_one=True)
    t = g.ring_design_mlir(c)
    pads = [ln for ln in t.splitlines() if 'sym_name = "pad' in ln]
    assert len(pads) == 1 and "memref<12xi32>" in pads[0]
    assert [phys_bank(g.RING_PAD_ADDRS[0] + 16 * i) for i in range(3)] == [0, 1, 0]
    assert c.name.endswith("p12o")
    with pytest.raises(ValueError):
        g.ring_validate(ring(pad=13, pad_one=True))


def test_ring_two_mm2s_on_one_schedule():
    t = g.ring_design_mlir(ring("mm", n=120))
    assert "aie.dma_start(MM2S, 1, ^pre1" in t and t.count("aie.dma_bd(%src : memref<64xi32>)") == 2
    assert t.count("aie.dma_bd(%pre : memref<4xi32>)") == 2                     # same chained path for both
    assert "dma_bd(%dst2 : memref<68xi32> offset = 0 len = 68)" in t           # S2MM 1 takes all 68 words
    assert phys_bank(g.RING_DST2_ADDR) in (6, 7) and g.RING_DST2_ADDR > g.FETCH_SCRATCH_ADDR
    assert "mov r4, #2" in bundles_of(g.ring_core_asm(ring("mm", n=120)))
    ev = g.ring_memmod_events(ring("mm"))
    assert len(ev) == 8 and "DMA_MM2S_1_MEMORY_STARVATION" in ev


def test_ring_opposite_parity_variant():
    t = g.ring_design_mlir(ring("mx", n=1))
    send1 = t.split("^send1:")[1].split("^s1_start")[0]
    assert "aie.dma_bd(%src : memref<64xi32> offset = 4 len = 60)" in send1
    assert "dma_bd(%dst2 : memref<68xi32> offset = 0 len = 64)" in t          # 4 pre words + 60
    assert phys_bank(g.FETCH_SRC_ADDR + 16) == 1                               # MM2S 1 starts on bank 1
    ev = g.ring_memmod_events(ring("mx"))
    assert len(ev) == 8 and "DMA_MM2S_1_STALLED_LOCK" in ev and "DMA_MM2S_0_STALLED_LOCK" in ev


# -- DMA-side stamp cases (fetch-build --start-task) ------------------------------------

def test_start_task_names_pre_nops_and_event_list():
    assert [g.fetch_case_name(p, False, "none", True) for p in (0, 5)] == ["lfs00", "lfs05"]
    assert [g.fetch_case_name(p, True, "none", True) for p in (0, 5)] == ["lfq00", "lfq05"]
    assert g.FETCH_PRE_NOPS == (0, 1, 2, 3, 4, 5)
    others = {g.fetch_case_name(k, c, b) for k in g.FETCH_KS for c, b in g.FETCH_PREFIX}
    assert not others & {g.fetch_case_name(p, c, "none", True) for p in g.FETCH_PRE_NOPS for c in (False, True)}
    args = g.fetch_inject_args(True)
    events = args[args.index("--memmod-sweep-events") + 1].split(",")
    assert events == g.FETCH_MEMMOD_EVENTS + ["DMA_MM2S_0_START_TASK"] and len(events) == 8
    assert "DMA_MM2S_0_START_TASK" not in g.fetch_inject_args()                  # the plain cases are unchanged


@pytest.mark.parametrize("pre", [0, 1, 5])
def test_start_task_core_has_pre_nops_before_the_marker_and_no_loads(pre):
    body = bundles_of(g.fetch_core_asm(0, "none", pre=pre))
    i = body.index("event #0")
    base = bundles_of(g.fetch_core_asm(0, "none"))
    assert i == base.index("event #0") + pre                                     # only the pre nops are added
    assert body[i + 1] == f"rel #{g.OWN + g.FETCH_LOCK_A}, r2"
    assert not [b for b in body if b.startswith(("lda r26", "lda r28"))]
    assert sum(b.startswith("acq ") for b in body) == 2 and sum(b.startswith("rel ") for b in body) == 1


def test_start_task_core_rejects_out_of_range_pre():
    for pre in (-1, g.FETCH_MAX_PRE + 1):
        with pytest.raises(ValueError):
            g.fetch_core_asm(0, "none", pre=pre)


@pytest.mark.parametrize("chained", [False, True])
def test_start_task_design_starts_the_task_from_the_sequence_after_the_go_lock_is_still_unset(chained):
    t = g.fetch_design_mlir(chained, task=True)
    assert "aie.dma_start(MM2S" not in t and "aie.dma_start(S2MM, 0, ^recv, ^end)" in t
    seq = t.split("aie.runtime_sequence")[1]
    assert seq.index("aiex.dma_configure_task(%t02, MM2S, 0)") < seq.index("aiex.dma_start_task(%task)") \
        < seq.index("aiex.set_lock(%lGo")                                       # the go lock comes after the start
    task = seq.split("aiex.dma_configure_task")[1].split("aiex.dma_start_task")[0]
    assert task.index("use_lock(%lA, AcquireGreaterEqual") < task.index("dma_bd(%src") < task.index("use_lock(%lB, Release")
    assert ("dma_bd(%pre" in task) == chained
    assert ("aie.next_bd ^send" in task) == chained
    for forbidden in ("dma_await", "dma_wait", "dma_memcpy_nd", "npu.sync"):
        assert forbidden not in seq                                              # still nothing in it blocks


def test_start_task_false_leaves_the_boot_design_alone():
    for chained in (False, True):
        assert g.fetch_design_mlir(chained, task=False) == g.fetch_design_mlir(chained)


def test_start_task_row_and_finish_check():
    rel = 1000
    e = held("DMA_MM2S_0_STALLED_LOCK", 600, rel + 1) + [ev("LOCK_0_REL", rel), ev("DMA_MM2S_0_START_TASK", rel - 400)]
    e += [ev("DMA_MM2S_0_FINISHED_BD", rel + 54), ev("DMA_S2MM_0_FINISHED_BD", rel + 76)]
    r = g.fetch_row(e)
    assert r["mm2s_start"] == [-400] and r["grant"] == 2
    assert g.start_missing(r) == []
    assert g.start_missing(r, chained=True) == ["MM2S FINISHED_BD x1 (want 2)"]   # the pre BD's finish is absent
    r2 = g.fetch_row([x for x in e if x.name != "DMA_MM2S_0_START_TASK"])
    assert g.start_missing(r2) == ["DMA_MM2S_0_START_TASK"]
    assert "start_task [-400]" in g.format_fetch_table([(0, "e1.emu", dict(r, check=[]))])
    assert "finish ok" in g.format_fetch_table([(0, "e1.emu", dict(r, check=[]))])


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
def test_start_task_core_assembles(tmp_path):
    s = tmp_path / "core.s"
    s.write_text(g.fetch_core_asm(0, "none", pre=5))
    r = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / "core.o"), str(s)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# -- refused-access probe (refuse-*) ----------------------------------------------------

def rcase(r=2, pos="e", s=35):
    return g.RefuseCase(r, pos, s)


def test_refuse_plan_is_48_distinct_valid_cases_clear_of_other_variants():
    plan = g.refuse_plan()
    names = [c.name for c in plan]
    assert len(names) == len(set(names)) == 48
    assert {(c.r, c.pos) for c in plan} == {(r, p) for r in (0, 1, 2, 3) for p in "el"}
    for c in plan:
        g.refuse_validate(c)
    for r in (0, 1, 2, 3):
        for pos in "el":
            ss = sorted(c.s for c in plan if (c.r, c.pos) == (r, pos))
            assert ss == list(range(g.REFUSE_S0[pos], g.REFUSE_S0[pos] + 6))      # six consecutive: each residue mod 3 twice
    assert all(n.startswith("rf") and not n.startswith(("rg", "lf", "ls", "lt", "ar")) for n in names)
    assert rcase(3, "l", 8).name == "rfl3s08"
    others = {g.fetch_case_name(k, c, b) for k in g.FETCH_KS for c, b in g.FETCH_PREFIX}
    assert not others & set(names)


def test_refuse_rejects_out_of_range():
    for c in (rcase(r=4), rcase(pos="x"), rcase(s=-1), rcase(s=g.REFUSE_MAX_S + 1)):
        with pytest.raises(ValueError):
            g.refuse_core_asm(c)


@pytest.mark.parametrize("cell", [15, 47])
def test_refuse_rule_refuses_r_accesses_with_the_chosen_history(cell):
    m = g.refuse_settle_loads(cell)
    assert [g.refuse_refused(r, cell, m) for r in g.REFUSE_RS] == [0, 1, 2, 3]
    assert m == {15: 0, 47: 1}[cell]                       # derived from the rotation rule, pinned here as a check


@pytest.mark.parametrize("r,pos,s", [(0, "e", 35), (1, "e", 36), (3, "l", 8), (2, "l", 3)])
def test_refuse_core_order_camp_marker_release_b_wait_end(r, pos, s):
    c = rcase(r, pos, s)
    body = bundles_of(g.refuse_core_asm(c))
    i = body.index("event #0")
    assert body[i + 1] == f"rel #{g.OWN + g.REFUSE_LOCK_A}, r2"
    k = c.cell + 1
    assert body[i + 2:i + 2 + k] == ["nop"] * k                          # first load `cell` cycles after a grant at +2
    camp = body[i + 2 + k:i + 2 + k + g.REFUSE_CAMP_SLOTS]
    assert [b.startswith("lda") for b in camp] == [True] * r + [False] * (g.REFUSE_CAMP_SLOTS - r)
    assert camp[r:] == ["nop"] * (g.REFUSE_CAMP_SLOTS - r)               # every r has the same camp length
    offs = [int(b.split("#")[1].rstrip("]")) for b in camp[:r]]
    assert [phys_bank(g.FETCH_LD_ADDR + o) for o in offs] == [0] * r      # all on physical bank 0
    j = i + 2 + k + g.REFUSE_CAMP_SLOTS
    assert body[j] == "event #1" and body[j + 1:j + 1 + s] == ["nop"] * s
    assert body[j + 1 + s] == f"rel #{g.OWN + g.REFUSE_LOCK_B}, r2"
    assert body.index(f"acq #{g.OWN + g.REFUSE_LOCK_END}, r3") > j + 1 + s
    assert body.index(f"acq #{g.OWN + g.REFUSE_LOCK_GO}, r3") < i
    assert sum(b.startswith("acq ") for b in body) == 2                  # go and END: nothing host-fed
    assert sum(b.startswith("rel ") for b in body) == 2                  # A, then B
    history = [b for b in body[:i] if b.startswith("lda r25,")]
    assert len(history) == g.refuse_settle_loads(c.cell)


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("c", [rcase(0, "e", 35), rcase(3, "l", 8)])
def test_refuse_core_assembles(c, tmp_path):
    s = tmp_path / "core.s"
    s.write_text(g.refuse_core_asm(c))
    r = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / "core.o"), str(s)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_refuse_design_chain_locks_words_and_safety():
    t = g.refuse_design_mlir()
    mm2s = t.split("aie.dma_start(MM2S, 0, ^pre")[1].split("^recv_start:")[0]
    assert mm2s.index("^pre:") < mm2s.index("^send:") < mm2s.index("^second:")
    pre, send, second = (mm2s.split(f"^{n}:")[1].split("\n    ^")[0] for n in ("pre", "send", "second"))
    assert "use_lock" not in pre and "memref<4xi32>" in pre and "aie.next_bd ^send" in pre
    assert send.index("use_lock(%lA, AcquireGreaterEqual") < send.index("dma_bd(%src") < send.index("use_lock(%lDone, Release")
    assert second.index("use_lock(%lB, AcquireGreaterEqual") < second.index("dma_bd(%tail") < second.index("use_lock(%lEnd, Release")
    assert "aie.next_bd ^second" in send and "aie.next_bd ^end" in second
    assert 'aie.lock(%t02, 0) {init = 0 : i32, sym_name = "lB"}' in t           # B is lock 0: the reset selector follows it
    assert "memref<84xi32>" in t.split("^recv:")[1].split("^end:")[0]          # the S2MM takes 4 + 64 + 16 words
    assert "use_lock" not in t.split("^recv:")[1].split("^end:")[0]
    seq = t.split("aie.runtime_sequence")[1]
    assert "aiex.set_lock(%lGo" in seq
    for forbidden in ("dma_await", "dma_wait", "dma_memcpy_nd", "dma_start_task", "npu.sync"):
        assert forbidden not in seq
    assert phys_bank(g.FETCH_PRE_ADDR) // 2 == 1 and phys_bank(g.REFUSE_TAIL_ADDR) // 2 == 1   # logical bank 1
    assert g.REFUSE_TAIL_ADDR % 32 == 0 and g.REFUSE_TAIL_ADDR >= g.FETCH_PRE_ADDR + 4 * g.FETCH_PRE_WORDS


def test_refuse_event_lists():
    assert g.REFUSE_MEMMOD_EVENTS == [
        "LOCK_0_REL", "LOCK_SEL0_ACQ_GE", "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_0_FINISHED_BD",
        "DMA_S2MM_0_FINISHED_BD", "CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1", "DMA_MM2S_0_FINISHED_TASK"]
    args = g.refuse_inject_args()
    assert args[args.index("--memmod-sweep-events") + 1] == ",".join(g.REFUSE_MEMMOD_EVENTS)
    assert args[args.index("--core-sweep-events") + 1] == "INSTR_EVENT_0,INSTR_EVENT_1,MEMORY_STALL"


def refuse_synthetic(rb_after_fk=8, grant_after=3, camp=(15, 16, 17), stalls=0, drop=()):
    g_k = 1000
    fk = g_k + 52
    rb = fk + rb_after_fk
    e = held("DMA_MM2S_0_STALLED_LOCK", 400, g_k - 1) + held("DMA_MM2S_0_STALLED_LOCK", fk + 1, rb + grant_after - 1)
    e += [ev("LOCK_0_REL", rb), ev("LOCK_SEL0_ACQ_GE", rb + grant_after - 1)]
    e += [ev("DMA_MM2S_0_FINISHED_BD", fk), ev("DMA_MM2S_0_FINISHED_BD", rb + grant_after + 16),
          ev("DMA_MM2S_0_FINISHED_TASK", rb + grant_after + 16), ev("DMA_S2MM_0_FINISHED_BD", rb + grant_after + 40)]
    e += [ev("CONFLICT_DM_BANK_0", g_k + t) for t in camp]
    e += [ev("INSTR_EVENT_0", g_k - 3, pkt=0), ev("INSTR_EVENT_1", g_k + 25, pkt=0)]
    e += [ev("MEMORY_STALL", g_k + 16 + i, pkt=0) for i in range(stalls)]
    return [x for x in e if x.name not in drop]


def test_refuse_row_numbers_and_finish_check():
    r = g.refuse_row(refuse_synthetic())
    assert r["g_k"] == 1000 and r["fins"][0] == 52 and r["r_b"] - r["fins"][0] == 8
    assert r["grant_b"] == 3 and r["acq_b"] == [2]                       # STALLED_LOCK falls one cycle after ACQ_GE
    assert r["refused"] == 3 and r["memory_stalls"] == 0
    assert g.refuse_missing(r, rcase(3)) == []
    r2 = g.refuse_row(refuse_synthetic(camp=(15, 16, 17, 18), stalls=1))
    assert r2["refused"] == 3 and r2["memory_stalls"] == 1               # conflicts minus the core's own stalls
    assert "ok" in g.format_refuse_table([(rcase(3), "e1.emu", r)]).splitlines()[1]


def test_refuse_missing_names_what_the_design_needs():
    no_conflict = g.refuse_row(refuse_synthetic(camp=()))
    assert g.refuse_missing(no_conflict, rcase(0)) == []                 # r = 0 has no camp, so no conflict
    assert g.refuse_missing(no_conflict, rcase(2)) == ["CONFLICT_DM_BANK_0"]
    r = g.refuse_row(refuse_synthetic(drop=("DMA_MM2S_0_FINISHED_TASK", "LOCK_SEL0_ACQ_GE")))
    assert set(g.refuse_missing(r, rcase(3))) == {"DMA_MM2S_0_FINISHED_TASK", "LOCK_SEL0_ACQ_GE",
                                                  "MM2S FINISHED_TASK x0 (want 1)"}
    assert g.refuse_missing({"g_k": None}, rcase()) == ["no STALLED_LOCK run or no LOCK_0_REL"]


def test_refuse_row_b_free_when_asked_has_no_grant():
    e = [x for x in refuse_synthetic() if x.name != "DMA_MM2S_0_STALLED_LOCK" or x.ts < 1000]
    assert g.refuse_row(e)["grant_b"] is None


# -- memory-module event swap (--memmod-event-swap) ---------------------------------------------

def test_parse_event_swap_splits_old_and_new():
    assert g.parse_event_swap("DMA_S2MM_0_FINISHED_BD=LOCK_1_REL") == ("DMA_S2MM_0_FINISHED_BD", "LOCK_1_REL")


@pytest.mark.parametrize("bad", ["", "LOCK_1_REL", "A=B=C", "=LOCK_1_REL", "LOCK_0_REL="])
def test_parse_event_swap_rejects_malformed_text(bad):
    with pytest.raises(ValueError, match="OLD=NEW"):
        g.parse_event_swap(bad)


def test_swap_replaces_in_place_so_the_slot_count_is_unchanged():
    out = g.apply_event_swap(g.SYNC_MAIN_EVENTS, ("DMA_S2MM_0_FINISHED_BD", "LOCK_1_REL"))
    assert out == ["LOCK_0_REL", "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_0_FINISHED_BD", "LOCK_1_REL"]
    assert len(out) == len(g.SYNC_MAIN_EVENTS)
    assert g.SYNC_MAIN_EVENTS[-1] == "DMA_S2MM_0_FINISHED_BD"           # the module constant is untouched


def test_swap_none_is_the_identity():
    assert g.apply_event_swap(g.SYNC_MAIN_EVENTS, None) == g.SYNC_MAIN_EVENTS


def test_swap_rejects_an_old_event_that_is_not_in_the_list():
    with pytest.raises(ValueError, match="not in the memory-module list"):
        g.apply_event_swap(g.SYNC_MAIN_EVENTS, ("DMA_MM2S_0_MEMORY_STARVATION", "LOCK_1_REL"))


def test_swap_rejects_an_unknown_new_event_name():
    pytest.importorskip("aie.utils.trace.events")
    with pytest.raises(ValueError, match="unknown memory-module event"):
        g.apply_event_swap(g.SYNC_MAIN_EVENTS, ("DMA_S2MM_0_FINISHED_BD", "LOCK_1_RELL"))


def test_swap_rejects_a_new_event_already_in_the_list():
    with pytest.raises(ValueError, match="already in"):
        g.apply_event_swap(g.SYNC_MAIN_EVENTS, ("DMA_S2MM_0_FINISHED_BD", "LOCK_0_REL"))


def test_sync_inject_args_carry_the_swap_and_default_is_unchanged():
    swap = ("DMA_S2MM_0_FINISHED_BD", "LOCK_1_REL")
    a = g.sync_inject_args("none", swap)
    assert a[a.index("--memmod-sweep-events") + 1] == "LOCK_0_REL,DMA_MM2S_0_STALLED_LOCK,DMA_MM2S_0_FINISHED_BD,LOCK_1_REL"
    assert g.sync_inject_args("none") == g.sync_inject_args("none", None)
    b = g.sync_inject_args("0", swap)                                    # the conflict events still follow
    assert b[b.index("--memmod-sweep-events") + 1].split(",")[-2:] == ["CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1"]


def test_fetch_and_chain_inject_args_carry_the_swap():
    f = g.fetch_inject_args(False, ("DMA_MM2S_0_MEMORY_STARVATION", "LOCK_1_REL"))
    assert "LOCK_1_REL" in f[f.index("--memmod-sweep-events") + 1].split(",")
    c = g.chain_inject_args(("LOCK_SEL0_ACQ_GE", "LOCK_1_REL"))
    assert c[c.index("--memmod-sweep-events") + 1].split(",")[-1] == "LOCK_1_REL"


def test_cli_option_is_on_the_sync_fetch_and_chain_builders():
    for cmd in (["fetch-build", "d", "--sync"], ["fetch-build", "d"], ["chain-build", "d"]):
        ns = g.build_parser().parse_args(cmd + ["--memmod-event-swap", "A=B"])
        assert ns.memmod_event_swap == ("A", "B")


# -- memory-module event add (--memmod-event-add) and lock event selectors (--lock-event-select) -----------

def test_add_appends_one_event_and_keeps_the_rest():
    out = g.apply_event_add(g.SYNC_MAIN_EVENTS, "LOCK_1_REL")
    assert out == g.SYNC_MAIN_EVENTS + ["LOCK_1_REL"]
    assert g.apply_event_add(g.SYNC_MAIN_EVENTS, None) == g.SYNC_MAIN_EVENTS


def test_add_rejects_a_traced_name_a_full_list_and_an_unknown_name():
    with pytest.raises(ValueError, match="already"):
        g.apply_event_add(g.SYNC_MAIN_EVENTS, "LOCK_0_REL")
    full = g.SYNC_MAIN_EVENTS + ["CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1", "LOCK_1_REL", "LOCK_2_REL"]
    assert len(full) == g.MEMMOD_TRACE_SLOTS
    with pytest.raises(ValueError, match="full"):
        g.apply_event_add(full, "LOCK_3_REL")
    pytest.importorskip("aie.utils.trace.events")
    with pytest.raises(ValueError, match="unknown memory-module event"):
        g.apply_event_add(g.SYNC_MAIN_EVENTS, "LOCK_1_RELL")


def test_inject_args_apply_swap_then_add():
    a = g.sync_inject_args("0", ("DMA_S2MM_0_FINISHED_BD", "LOCK_2_REL"), "LOCK_1_REL")
    ev = a[a.index("--memmod-sweep-events") + 1].split(",")
    assert ev == ["LOCK_0_REL", "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_0_FINISHED_BD", "LOCK_2_REL",
                  "CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1", "LOCK_1_REL"]
    assert g.sync_inject_args("none", None, None) == g.sync_inject_args("none")


def test_parse_lock_event_select():
    assert g.parse_lock_event_select("1=1") == (1, 1)
    for bad in ("1", "a=1", "1=", "1=2=3", ""):
        with pytest.raises(ValueError, match="N=LOCK_ID"):
            g.parse_lock_event_select(bad)


def regdb_or_skip():
    try:
        return g.compute_memory_registers()
    except Exception as e:                                                  # no mlir-aie checkout
        pytest.skip(f"register database not available: {e}")


def test_lock_event_select_write_comes_from_the_register_database():
    regdb_or_skip()
    (addr, val), = g.lock_event_select_writes([(1, 1)])
    assert addr == 0x1F104                                   # Locks_Event_Selection_1, memory module
    assert val == (1 << 16) | 0x3F                           # Lock_Select = 1, Lock_Value left at its reset
    (addr7, val7), = g.lock_event_select_writes([(7, 15)])
    assert (addr7, val7) == (0x1F11C, (15 << 16) | 0x3F)


def test_lock_event_select_rejects_selector_zero_and_out_of_range():
    regdb_or_skip()
    with pytest.raises(ValueError, match="selector 0"):
        g.lock_event_select_writes([(0, 1)])
    with pytest.raises(ValueError, match="selector"):
        g.lock_event_select_writes([(8, 1)])
    with pytest.raises(ValueError, match="lock"):
        g.lock_event_select_writes([(1, 16)])
    with pytest.raises(ValueError, match="twice"):
        g.lock_event_select_writes([(1, 1), (1, 2)])


def test_sync_design_programs_selectors_before_the_start_lock_and_default_is_unchanged():
    regdb_or_skip()
    base = g.sync_design_mlir(sync_case())
    assert "%c_sa0" not in base and g.sync_design_mlir(sync_case(), ()) == base
    text = g.sync_design_mlir(sync_case(), [(1, 1)])
    w = ("%c_sa0 = arith.constant 127236 : i32\n      %c_sv0 = arith.constant 65599 : i32\n"
         "      aiex.npu.write32(%c_sa0, %c_sv0) {column = 0 : i32, row = 2 : i32} : i32, i32")
    assert w in text
    assert text.index(w) < text.index("aiex.set_lock(%lStart")


def test_cli_has_the_add_and_select_options_on_fetch_build():
    ns = g.build_parser().parse_args(["fetch-build", "d", "--sync", "--memmod-event-add", "LOCK_1_REL",
                                      "--lock-event-select", "1=1", "--lock-event-select", "2=3"])
    assert ns.memmod_event_add == "LOCK_1_REL" and ns.lock_event_select == [(1, 1), (2, 3)]
    ns = g.build_parser().parse_args(["chain-build", "d", "--memmod-event-add", "LOCK_1_REL"])
    assert ns.memmod_event_add == "LOCK_1_REL"


# -- lock-held S2MM probe (sl-*): release -> S2MM grant, and what follows it ----------------------------

def sl(head=True, d=12, kind="none", k=0, p=0, n=g.FETCH_WORDS):
    return g.SlCase(head=head, d=d, kind=kind, k=k, p=p, n=n)


def test_sl_case_names_are_distinct_and_clear_of_the_other_variants():
    plan = g.sl_plan()
    names = [c.name for c in plan]
    assert len(names) == len(set(names))
    assert sl().name == "slh12" and sl(False, 10).name == "slf10"
    assert sl(True, 12, "load", 3, 1).name == "slh12k03p1" and sl(False, 12, "camp").name == "slf12camp"
    other = {g.fetch_case_name(k, c, b) for k in range(65) for (c, b) in g.FETCH_PREFIX}
    other |= {sync_case(d).name for d in g.SYNC_DEFAULT_DS} | {c.name for c in g.ring_plan()}
    assert not set(names) & other


def test_sl_plan_covers_the_specified_sweeps():
    plan = g.sl_plan()
    assert len(plan) == 71
    ds = tuple(range(g.SL_FIRST_D, g.SL_FIRST_D + 6))
    for head in (True, False):                              # the release sweep: six consecutive d, both variants
        assert [c.d for c in plan if c.head == head and c.kind == "none" and c.n == g.FETCH_WORDS] == list(ds)
        assert sum(c.kind == "camp" and c.head == head and c.ch == 0 for c in plan) == 1
    loads = [c for c in plan if c.kind == "load"]
    assert {c.d for c in loads} == {g.SL_LOAD_D} and all(c.head for c in loads)   # one d, the chained variant
    assert {(c.k, c.p) for c in loads} == {(k, p) for k in range(25) for p in (0, 1)}
    assert g.SL_LOAD_D in ds
    ns = [c for c in plan if c.n != g.FETCH_WORDS]                  # the held word count, first-BD variant only
    assert [(c.head, c.n, c.kind) for c in ns] == [(False, n, "none") for n in (4, 8, 12, 16, 20)]
    assert all(c.d == g.SL_LOAD_D for c in ns)
    ch1 = [c for c in plan if c.ch == 1]                            # the camp on S2MM 1, both variants
    assert sorted((c.head, c.kind) for c in ch1) == [(False, "camp"), (True, "camp")]


def test_sl_rejects_out_of_range():
    for c in (sl(d=-1), sl(d=g.SYNC_MAX_D + 1), sl(kind="x"), sl(kind="load", k=-1), sl(kind="load", k=65),
              sl(kind="load", p=2)):
        with pytest.raises(ValueError):
            g.sl_validate(c)
    g.sl_validate(sl(kind="load", k=0, p=1))


def test_sl_core_releases_a_after_d_nops_then_one_load_k_nops_later():
    body = bundles_of(g.sl_core_asm(sl(d=7, kind="load", k=3, p=1)))
    i = body.index("event #0")
    assert body[i + 1:i + 8] == ["nop"] * 7
    assert body[i + 8] == f"rel #{g.OWN + g.FETCH_LOCK_A}, r2"
    assert body[i + 9:i + 12] == ["nop"] * 3
    assert body[i + 12] == "lda r26, [p0, #16]"                       # parity 1 = the odd granule's physical bank
    assert phys_bank(g.FETCH_LD_ADDR + 16) == 1 and phys_bank(g.FETCH_LD_ADDR) == 0
    assert sum(b.startswith("lda r2") or b.startswith("lda r3") for b in body) == 1
    assert "lda r26, [p0, #0]" in [b for b in bundles_of(g.sl_core_asm(sl(kind="load", p=0)))]


def test_sl_core_waits_for_go_first_and_for_the_s2mm_release_only():
    body = bundles_of(g.sl_core_asm(sl()))
    go = body.index(f"acq #{g.OWN + g.FETCH_LOCK_GO}, r3")
    assert go < body.index("event #0")
    done = body.index(f"acq #{g.OWN + g.FETCH_LOCK_B}, r3")
    assert done > body.index(f"rel #{g.OWN + g.FETCH_LOCK_A}, r2")
    assert sum(b.startswith("acq ") for b in body) == 2                    # nothing waits on a host-fed lock
    assert sum(b == f"rel #{g.OWN + g.FETCH_LOCK_A}, r2" for b in body[done:]) == g.RING_FILL_RELEASES
    assert not [b for b in body[go:body.index("event #0")] if b.startswith(("lda", "st "))]


def test_sl_core_camp_is_the_existing_90_loads_on_bank_0():
    body = bundles_of(g.sl_core_asm(sl(kind="camp")))
    i = body.index("event #0")
    camp = body[i + 14:i + 14 + 90]
    assert len(camp) == 90 and all(b.startswith("lda ") for b in camp)
    assert {phys_bank(g.FETCH_LD_ADDR + int(b.split("#")[1].rstrip("]"))) for b in camp} == {0}
    assert body[i + 14 + 90] == "nop"


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("c", [sl(), sl(kind="load", k=24, p=1), sl(False, 15, "camp")])
def test_sl_core_assembles(c, tmp_path):
    s = tmp_path / "core.s"
    s.write_text(g.sl_core_asm(c))
    r = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / "core.o"), str(s)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_sl_design_chained_s2mm_head_then_the_lock_bd():
    t = g.sl_design_mlir(sl(head=True))
    assert 'aie.lock(%t02, 0) {init = 0 : i32, sym_name = "lA"}' in t                 # lock 0: the reset selector
    assert "aie.flow(%t02, DMA : 0, %t02, DMA : 0)" in t
    send = t.split("^send:")[1].split("^recv_start:")[0]
    assert "use_lock" not in send and "dma_bd(%feed : memref<68xi32>)" in send         # lock-free, 4 + 64 words
    assert "aie.dma_start(S2MM, 0, ^head, ^end)" in t
    head = t.split("^head:")[1].split("^recv:")[0]
    assert "use_lock" not in head and "dma_bd(%dst : memref<4xi32>)" in head and "next_bd ^recv" in head
    recv = t.split("^recv:")[1].split("^end:")[0]
    assert recv.index("use_lock(%lA, AcquireGreaterEqual") < recv.index("dma_bd(%sdst : memref<64xi32>") \
        < recv.index("use_lock(%lB, Release")
    assert phys_bank(g.RING_SDST_ADDR) == 0 and phys_bank(g.RING_FEED_ADDR) == 2


def test_sl_design_first_bd_variant_has_no_head_and_the_mm2s_sends_64_words():
    t = g.sl_design_mlir(sl(head=False))
    assert "aie.dma_start(S2MM, 0, ^recv, ^end)" in t and "^head:" not in t
    assert "dma_bd(%feed : memref<64xi32>)" in t
    recv = t.split("^recv:")[1].split("^end:")[0]
    assert "use_lock(%lA, AcquireGreaterEqual" in recv and "dma_bd(%sdst : memref<64xi32>" in recv


def test_sl_variants_share_buffers_and_the_sequence_only_sets_go():
    def bufs(t):
        return [ln.strip() for ln in t.splitlines() if "aie.buffer" in ln and "feed" not in ln]
    assert bufs(g.sl_design_mlir(sl(True))) == bufs(g.sl_design_mlir(sl(False)))
    addrs = [int(ln.split("address = ")[1].split(" ")[0]) for ln in bufs(g.sl_design_mlir(sl()))]
    assert addrs == sorted(addrs)
    seq = g.sl_design_mlir(sl()).split("aie.runtime_sequence")[1]
    assert "aiex.set_lock(%lGo" in seq
    for forbidden in ("dma_wait", "dma_memcpy_nd", "npu.sync", "dma_start_task"):
        assert forbidden not in seq


def test_sl_event_lists_fit_the_eight_slots_and_name_the_specified_events():
    assert g.SL_MEMMOD_EVENTS == ["LOCK_0_REL", "DMA_S2MM_0_STALLED_LOCK", "DMA_S2MM_0_FINISHED_BD",
                                  "DMA_MM2S_0_STREAM_BACKPRESSURE", "CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1",
                                  "DMA_S2MM_0_STREAM_STARVATION", "DMA_S2MM_0_MEMORY_BACKPRESSURE"]
    assert len(g.SL_MEMMOD_EVENTS) == g.MEMMOD_TRACE_SLOTS
    a = g.sl_inject_args(sl())
    assert a[a.index("--memmod-sweep-events") + 1].split(",") == g.SL_MEMMOD_EVENTS
    assert a[a.index("--core-sweep-events") + 1] == "INSTR_EVENT_0,LOCK_STALL"
    pytest.importorskip("aie.utils.trace.events")
    for name in g.SL_MEMMOD_EVENTS:
        g._check_memmod_event_name(name)


def sl_synthetic(rel=1000, grant_after=4, fin=76, bp_fall=6, conflicts=(), starv=(), fills=g.RING_FILL_RELEASES,
                 s2mm_fins=1, bp=True):
    G = rel + grant_after
    e = held("DMA_S2MM_0_STALLED_LOCK", 100, G - 1)                      # parked since boot
    e += [ev("LOCK_0_REL", rel)] + [ev("LOCK_0_REL", rel + 500 + 30 * i) for i in range(fills)]
    if bp:
        e += held("DMA_MM2S_0_STREAM_BACKPRESSURE", 150, G + bp_fall - 1)
    e += [ev("DMA_S2MM_0_FINISHED_BD", G + fin + 7 * i) for i in range(s2mm_fins)]
    for bank, at in conflicts:
        e.append(ev(f"CONFLICT_DM_BANK_{bank}", G + at))
    for a, z in starv:
        e += held("DMA_S2MM_0_STREAM_STARVATION", G + a, G + z)
    return e


def test_sl_row_numbers_relative_to_the_grant():
    r = g.sl_row(sl_synthetic(rel=1000, grant_after=4, fin=76, bp_fall=6, conflicts=[(0, 17), (1, 25), (0, 33)],
                              starv=[(3, 3)]))
    assert r["release"] == 1000 and r["grant"] == 4
    assert r["s2mm_fin"] == [76]
    assert r["bp_fall"] == 6
    assert r["starvation"] == [(3, 3)]
    assert r["conflicts"] == {0: [17, 33], 1: [25]}
    assert r["releases"] == 1 + g.RING_FILL_RELEASES
    assert g.sl_missing(r, sl()) == []


def test_sl_row_ignores_the_boot_time_head_finish_and_the_early_backpressure():
    e = sl_synthetic() + [ev("DMA_S2MM_0_FINISHED_BD", 50)]
    r = g.sl_row(e)
    assert r["s2mm_fin"] == [76]                                        # only finishes after the release count


def test_sl_row_no_release_or_no_grant_says_so():
    assert g.sl_row([])["release"] is None
    assert g.sl_missing(g.sl_row([]), sl()) == ["no LOCK_0_REL"]
    e = [x for x in sl_synthetic() if x.name != "DMA_S2MM_0_STALLED_LOCK"]
    r = g.sl_row(e)
    assert r["grant"] is None and "grant" in g.sl_missing(r, sl())


def test_sl_missing_flags_an_unfinished_task_and_a_core_that_never_ended():
    r = g.sl_row(sl_synthetic(s2mm_fins=0, bp=False, fills=3))
    miss = g.sl_missing(r, sl())
    assert any("S2MM FINISHED_BD" in m for m in miss)
    assert any("core" in m for m in miss)


def test_format_sl_table_has_one_line_per_run_with_the_decided_columns():
    r = g.sl_row(sl_synthetic(conflicts=[(0, 17)], starv=[(3, 3)]))
    r["check"] = g.sl_missing(r, sl())
    text = g.format_sl_table([(sl(), "e1.emu", r)])
    line = [ln for ln in text.splitlines() if ln.startswith("slh12")][0]
    for want in ("e1.emu", "1000", "grant +4", "fin +76", "bp +6", "starv [+3]", "B0 [+17]", "finish ok"):
        assert want in line
    assert "no LOCK_0_REL" in g.format_sl_table([(sl(), "e1.emu", {"release": None})])


def test_cli_has_the_sl_subcommands():
    ns = g.build_parser().parse_args(["sl-build", "d", "--names", "slh12"])
    assert ns.cmd == "sl-build" and ns.names == ["slh12"]
    ns = g.build_parser().parse_args(["sl-table", "d", "e1.emu"])
    assert ns.runs == ["e1.emu"]
    ns = g.build_parser().parse_args(["sl-run", "d", "e1", "--backend", "emu"])
    assert ns.backend == "emu"


def test_sl_held_word_count_means_the_mm2s_sends_exactly_what_the_s2mm_takes():
    c = sl(False, 12, n=8)
    assert c.name == "slf12n08" and c.words == 8
    t = g.sl_design_mlir(c)
    assert "dma_bd(%feed : memref<8xi32>)" in t                                 # no surplus word
    assert "dma_bd(%sdst : memref<64xi32> offset = 0 len = 8)" in t
    assert sl(True, 12, n=8).words == 12
    for bad in (0, 65):
        with pytest.raises(ValueError):
            g.sl_validate(sl(n=bad))


def test_sl_channel_1_routes_the_loopback_to_s2mm_1_and_traces_its_events():
    c = g.SlCase(head=False, d=12, kind="camp", ch=1)
    assert c.name == "slf12camps1"
    t = g.sl_design_mlir(c)
    assert "aie.flow(%t02, DMA : 0, %t02, DMA : 1)" in t and "aie.dma_start(S2MM, 1, ^recv, ^end)" in t
    ev = g.sl_memmod_events(c)
    assert ev == [e.replace("S2MM_0", "S2MM_1") for e in g.SL_MEMMOD_EVENTS] and len(ev) == 8
    assert "DMA_S2MM_1_STREAM_STARVATION" in ev and "DMA_MM2S_0_STREAM_BACKPRESSURE" in ev
    assert all("DMA_S2MM_0_STREAM_STARVATION" in g.sl_memmod_events(c0) for c0 in g.sl_plan() if c0.ch == 0)
    with pytest.raises(ValueError):
        g.sl_validate(g.SlCase(True, 12, ch=2))


def test_sl_row_reads_the_channel_it_is_given_and_reports_the_trace_tail():
    e = [x for x in sl_synthetic()]
    swapped = [ev(x.name.replace("S2MM_0", "S2MM_1"), x.ts) for x in e]
    assert g.sl_row(swapped, 1)["grant"] == 4 and g.sl_row(swapped, 0)["grant"] is None
    r = g.sl_row(e)
    assert r["tail"] > 0 and g.sl_missing(r, sl()) == []
    early = [x for x in e if not (x.name == "LOCK_0_REL" and x.ts > 1000)] + [ev("LOCK_0_REL", 1005)]
    assert any("trace ends" in m for m in g.sl_missing(g.sl_row(early), sl()))


# -- memtile-fed lock-held S2MM probe (mq-*): the parked memtile feed's restart after a quiet stretch ------

def mq(kind="q", t=0, k=None):
    return g.MqCase(kind=kind, t=t, k=k)


def test_mq_plan_is_the_specified_seventeen_cases_with_distinct_names():
    plan = g.mq_plan()
    names = [c.name for c in plan]
    assert len(plan) == 17 and len(set(names)) == 17
    assert {"mqq", "mqm"} <= set(names)
    assert {f"mq{v}k{k}" for v in "qm" for k in g.MQ_KS} <= set(names)
    assert {f"mqt{t:03d}" for t in g.MQ_TS} <= set(names)
    assert not any(n.startswith(("sl", "rf", "rg")) for n in names)
    assert mq("t", 8).name == "mqt008" and mq("m", k=20).name == "mqmk20"


def test_mq_fill_lengths_follow_the_variant():
    assert g.mq_fill_words(mq("q")) == 0
    assert g.mq_fill_words(mq("q", k=19)) == 0
    assert g.mq_fill_words(mq("m")) == g.MQ_M_FILL_WORDS
    # a longer quiet stretch is a shorter fill, one word a cycle
    assert g.mq_fill_words(mq("t", 8)) - g.mq_fill_words(mq("t", 64)) == 56
    assert g.mq_fill_words(mq("t", 8)) == g.MQ_FILL_END_AT_RESUME - 8
    for c in g.mq_plan():
        assert 0 <= g.mq_fill_words(c) <= g.MQ_M_FILL_WORDS


def test_mq_rejects_out_of_range():
    for c in (mq("x"), mq("t", 0), mq("t", g.MQ_FILL_END_AT_RESUME), mq("q", t=8), mq("q", k=-1),
              mq("q", k=65), mq("t", 64, k=20)):
        with pytest.raises(ValueError):
            g.mq_validate(c)
    g.mq_validate(mq("m", k=21))


def test_mq_core_waits_for_go_settles_releases_a_then_one_bank0_load_k_nops_later():
    body = bundles_of(g.mq_core_asm(mq("m", k=20)))
    rel = [i for i, b in enumerate(body) if b.startswith(f"rel #{g.OWN + g.FETCH_LOCK_A}")]
    ev0 = body.index("event #0")
    assert rel[0] == ev0 + g.MQ_D + 1                         # d nops after event #0
    load = body.index("lda r26, [p0, #0]")                    # bank 0: granule 6's bank
    assert load == rel[0] + 20 + 1
    assert body.index(f"acq #{g.OWN + g.FETCH_LOCK_GO}, r3") < ev0
    assert any(f"acq #{g.OWN + g.FETCH_LOCK_B}" in b for b in body[load:])
    assert len(rel) == 1 + g.RING_FILL_RELEASES               # the first fill releases grant the trailing BDs
    assert "lda r26, [p0" not in g.mq_core_asm(mq("q"))
    assert g.mq_core_asm(mq("q")).count(".Lsettle") >= 1


def test_mq_design_q_has_no_fill_and_m_fills_memtile_s2mm_1_from_the_shim():
    q, m = g.mq_design_mlir(mq("q")), g.mq_design_mlir(mq("m"))
    for t in (q, m):
        assert "aie.flow(%t01, DMA : 0, %t02, DMA : 0)" in t                 # the feed
        assert "aie.memtile_dma(%t01)" in t and "aie.dma_start(MM2S, 0, ^f1" in t
        assert "aie.dma_bd(%feed : memref<64xi32>)" in t
        assert "aie.use_lock(%lA, AcquireGreaterEqual" in t and "aiex.set_lock(%lGo" in t
        assert t.count("aie.dma_bd(%sdst : memref<64xi32>)") == 1
        # every trailing BD the S2MM takes, the MM2S sends: no word left over, no BD left waiting for one
        nt = g.MQ_TRAILING_BDS
        assert t.count("aie.dma_bd(%tdst : memref<64xi32>)") == nt == t.count("aie.dma_bd(%trail : memref<64xi32>)")
        assert t.count("aie.use_lock(%lA, AcquireGreaterEqual") == 1 + nt <= g.RING_FILL_RELEASES + 1
    assert "DMA : 1)" not in q and "shim_dma_allocation" not in q and "memref<4xi32>" in q
    n = g.MQ_M_FILL_WORDS
    assert "aie.flow(%t00, DMA : 0, %t01, DMA : 1)" in m
    assert "aie.shim_dma_allocation @fill (%t00, MM2S, 0)" in m
    assert f"aie.dma_bd(%fbuf : memref<{n}xi32>)" in m and "aie.dma_start(S2MM, 1, ^w" in m
    assert f"runtime_sequence(%fill: memref<{n}xi32>)" in m
    assert "issue_token = true" in m and "aiex.npu.dma_wait {symbol = @fill}" in m
    # the fill is pushed before the go lock, and waited for after the pad writes
    assert m.index("dma_memcpy_nd") < m.index("aiex.set_lock(%lGo") < m.index("dma_wait")


def test_mq_fill_task_dims_cover_exactly_its_words():
    for c in g.mq_plan():
        w = g.mq_fill_words(c)
        if not w:
            continue
        outer, inner = g.mq_fill_dims(w)
        assert outer * inner == w and inner <= 1023 and outer <= 1023
        assert f"[1, 1, {outer}, {inner}][0, 0, {inner}, 1]" in g.mq_design_mlir(c)


def test_mq_routes_read_the_feed_and_trace_ports_from_the_routed_design():
    text = """
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<DMA : 0, North : 1> loc(#loc42)
      aie.connect<South : 4, DMA : 1> loc(#loc43)
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(South : 0, %1) loc(#loc5)
      %3 = aie.masterset(South : 3, %0) loc(#loc5)
      aie.packet_rules(North : 3) {
        aie.rule(29, 1, %0) loc(#loc5)
      }
      aie.packet_rules(Trace : 0) {
        aie.rule(31, 2, %1) loc(#loc5)
      }
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0> loc(#loc42)
    }"""
    assert g.mq_routes(text) == {"feed_north": 1, "fill_south": 4, "trace_south": 3}    # not the memtile's own
    with pytest.raises(ValueError):
        g.mq_routes("%switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {\n}")


def test_mq_inject_args_name_the_ports_and_fit_the_slots():
    routes = {"feed_north": 1, "fill_south": 4, "trace_south": 0}
    a = g.mq_inject_args(routes)
    mt = a[a.index("--memtile-sweep-events") + 1].split(",")
    assert len(mt) == 8 and mt[:5] == ["PORT_RUNNING_0", "PORT_RUNNING_1", "PORT_STALLED_1", "PORT_RUNNING_2",
                                       "PORT_RUNNING_3"]
    assert a[a.index("--port-slots") + 1] == "0=DMA:0:slave,1=North:1:master,2=DMA:1:master,3=South:0:master"
    assert "S2MM_SEL0:1" in a[a.index("--memtile-sel-channels") + 1]
    mm = a[a.index("--memmod-sweep-events") + 1].split(",")
    assert len(mm) == g.MEMMOD_TRACE_SLOTS and mm[0] == "LOCK_0_REL"
    assert {"DMA_S2MM_0_STREAM_STARVATION", "CONFLICT_DM_BANK_0", "DMA_S2MM_0_FINISHED_BD"} <= set(mm)
    assert a[a.index("--memmod-sel-locks") + 1] == "LOCK_SEL1:1"
    pytest.importorskip("aie.utils.trace.events")
    for name in mm:
        g._check_memmod_event_name(name)


def mq_synthetic(rel=5000, grant_after=3, fin=55, starv=((0, 0),), conflicts=(), resume=200, fill_end=None,
                 fills=g.RING_FILL_RELEASES):
    """Compute-tile events on one timer (rel..), memtile events on another (resume.. on the memtile timer)."""
    G = rel + grant_after
    e = held("DMA_S2MM_0_STALLED_LOCK", 100, G - 1)
    e += [ev("LOCK_0_REL", rel)] + [ev("LOCK_0_REL", rel + 600 + 40 * i) for i in range(fills)]
    e += [ev("DMA_S2MM_0_FINISHED_BD", G + fin)]
    e += [ev("DMA_S2MM_0_FINISHED_BD", G + 1500 + 200 * i) for i in range(g.MQ_TRAILING_BDS)]
    for a, z in starv:
        e += held("DMA_S2MM_0_STREAM_STARVATION", G + a, G + z)
    for bank, at in conflicts:
        e.append(ev(f"CONFLICT_DM_BANK_{bank}", G + at))
    mt = dict(pkt=3, row=1)
    e += held("PORT_RUNNING_1", resume - 1, resume + 62, **mt) + held("PORT_RUNNING_0", resume, resume + 33, **mt)
    e += held("PORT_RUNNING_0", resume + 35, resume + 63, **mt)
    e += held("PORT_RUNNING_0", resume + 900, resume + 963, **mt)          # the trailing BD, much later
    e += held("PORT_RUNNING_3", resume - 400, resume - 393, **mt)         # a trace packet crossing
    e += [ev("DMA_MM2S_SEL0_FINISHED_BD", resume + 70 + 200 * i, **mt) for i in range(1 + g.MQ_TRAILING_BDS)]
    if fill_end is not None:
        e += held("PORT_RUNNING_2", fill_end - 300, fill_end, **mt)
        e += [ev("DMA_S2MM_SEL0_FINISHED_BD", fill_end + 1, **mt)]
    return e


def test_mq_row_compute_numbers_relative_to_the_grant_and_memtile_numbers_on_their_own_timer():
    r = g.mq_row(mq_synthetic(fin=56, starv=((0, 0), (20, 21)), conflicts=[(0, 19)], fill_end=150))
    assert r["release"] == 5000 and r["grant"] == 3 and r["fin"] == 56
    assert r["starvation"] == [(0, 0), (20, 21)] and r["starv_after_g2"] == [(20, 21)]
    assert r["conflicts"] == {0: [19], 1: []}
    assert r["resume"] == 200 and r["north_first"] == 199
    # quiet stretches end at the feed's first memtile beat (North master, 199): fill beats 151..198 absent
    assert r["quiet_fill"] == 48 and r["quiet_any"] == 48           # the trace packet ended long before
    assert g.mq_missing(r, fill=True) == []
    q = g.mq_row(mq_synthetic())
    assert q["quiet_fill"] is None and q["quiet_any"] == 199 - (200 - 393) - 1 == q["quiet_lb"]
    lone = g.mq_row([x for x in mq_synthetic() if x.name != "PORT_RUNNING_3"])
    assert lone["quiet_any"] is None and lone["quiet_lb"] == 0      # the North master's own first beat starts it


def test_mq_missing_flags_an_absent_grant_an_unfinished_bd_and_an_unfinished_core():
    assert g.mq_missing(g.mq_row([])) == ["no LOCK_0_REL"]
    e = [x for x in mq_synthetic() if x.name != "DMA_S2MM_0_STALLED_LOCK"]
    assert "grant" in g.mq_missing(g.mq_row(e))
    e = [x for x in mq_synthetic() if not (x.name == "DMA_S2MM_0_FINISHED_BD" and x.ts > 6400)]
    assert any("trailing" in m for m in g.mq_missing(g.mq_row(e)))
    assert any("core" in m for m in g.mq_missing(g.mq_row(mq_synthetic(fills=3))))
    e = [x for x in mq_synthetic() if x.name != "PORT_RUNNING_0"]
    assert any("memtile feed" in m for m in g.mq_missing(g.mq_row(e)))
    e = [x for x in mq_synthetic() if not (x.name == "DMA_MM2S_SEL0_FINISHED_BD" and x.ts > 500)]
    assert any("memtile MM2S 0" in m for m in g.mq_missing(g.mq_row(e)))
    assert g.mq_missing(g.mq_row(mq_synthetic(fill_end=150)), fill=True) == []
    assert any("fill" in m for m in g.mq_missing(g.mq_row(mq_synthetic()), fill=True))


def test_format_mq_table_has_the_decided_columns():
    r = g.mq_row(mq_synthetic(fin=56, conflicts=[(0, 19)], fill_end=150))
    r["check"] = g.mq_missing(r)
    line = [ln for ln in g.format_mq_table([(mq("t", 64), "e1.emu", r)]).splitlines() if ln.startswith("mqt064")][0]
    for want in ("e1.emu", "grant +3", "fin +56", "B0 [+19]", "quiet fill 48", "finish ok"):
        assert want in line


def test_cli_has_the_mq_subcommands():
    p = g.build_parser()
    assert p.parse_args(["mq-build", "d", "--names", "mqq"]).names == ["mqq"]
    assert p.parse_args(["mq-run", "d", "e1", "--backend", "emu"]).backend == "emu"
    assert p.parse_args(["mq-table", "d", "e1.emu"]).runs == ["e1.emu"]
    assert p.parse_args(["mq-packets", "d", "e1.emu"]).runs == ["e1.emu"]


# -- mr (round 2): the quiet stretch made on the array by a core-timed waker -------------------------------

def mr(w="c", t=128):
    return g.MrCase(waker=w, t=t)


def test_mr_plan_covers_each_waker_over_its_grid():
    plan = g.mr_plan()
    names = [c.name for c in plan]
    assert len(names) == len(set(names)) == sum(len(v) for v in g.MR_TS.values())
    assert set(g.MR_TS) == {"c", "b", "m"} and mr("b", 8).name == "mrb008"
    assert max(g.MR_TS["c"]) == 600 and min(g.MR_TS["b"]) <= 8
    assert not set(names) & {c.name for c in g.mq_plan()}


def test_mr_rejects_out_of_range():
    for c in (mr("x"), mr("c", -1), mr("c", g.MR_MAX_T + 1)):
        with pytest.raises(ValueError):
            g.mr_validate(c)
    g.mr_validate(mr("m", 0))


def test_mr_core_releases_w_then_a_exactly_d_nops_later():
    c = mr("b", 96)
    body = bundles_of(g.mr_core_asm(c))
    rw = body.index(f"rel #{g.OWN + g.MR_LOCK_W}, r2")
    ra = [i for i, b in enumerate(body) if b == f"rel #{g.OWN + g.FETCH_LOCK_A}, r2"][0]
    assert body.index("event #0") == rw + g.mr_delay_nops(c) + 1
    assert all(b == "nop" for b in body[rw + 1:rw + 1 + g.mr_delay_nops(c)])
    assert ra == body.index("event #0") + g.MQ_D + 1
    assert g.mr_delay_nops(mr("b", 97)) - g.mr_delay_nops(c) == 1         # one nop per cycle of quiet
    assert body.index(f"acq #{g.OWN + g.FETCH_LOCK_GO}, r3") < rw


def test_mr_designs_route_each_waker_where_it_belongs():
    c, b, m = (g.mr_design_mlir(mr(w)) for w in "cbm")
    for t in (c, b, m):
        assert "aie.flow(%t01, DMA : 0, %t02, DMA : 0)" in t                 # the feed, as in mq
        assert "aie.use_lock(%lW, AcquireGreaterEqual" in t and "aie.dma_start(MM2S, 1, ^w0" in t
        assert "shim_dma_allocation" not in t and "dma_memcpy_nd" not in t   # nothing timed by the host
    assert "aie.flow(%t02, DMA : 1, %t02, DMA : 1)" in c and "%t01, DMA : 1)" not in c
    assert "aie.flow(%t02, DMA : 1, %t01, DMA : 1)" in b
    assert "aie.flow(%t02, DMA : 1, %t01, DMA : 2)" in m and "aie.flow(%t01, DMA : 1, %t01, DMA : 1)" in m
    assert "aie.use_lock(%lX, Release" in m and "aie.use_lock(%lX, AcquireGreaterEqual" in m
    assert f"memref<{g.MR_MT_LOOP_WORDS}xi32>" in m and f"memref<{g.MR_WAKE_WORDS}xi32>" in b


def test_mr_hardware_builds_do_not_trace_the_waker_and_calibration_builds_do():
    routes = {"feed_north": 1, "fill_south": None, "trace_south": 3}
    waker = {"DMA_MM2S_1_STALLED_LOCK", "DMA_MM2S_1_FINISHED_BD", "DMA_S2MM_1_FINISHED_BD", "PORT_RUNNING_2",
             "PORT_RUNNING_4", "DMA_S2MM_SEL0_FINISHED_BD"}
    hw = g.mr_inject_args(routes)
    for flag in ("--memmod-sweep-events", "--memtile-sweep-events"):
        assert not set(hw[hw.index(flag) + 1].split(",")) & waker
    assert "PORT_RUNNING_3" in hw[hw.index("--memtile-sweep-events") + 1]           # trace crossings stay visible
    cal = g.mr_inject_args(routes, trace_waker=True)
    mm = cal[cal.index("--memmod-sweep-events") + 1].split(",")
    assert len(mm) == 8 and {"DMA_MM2S_1_STALLED_LOCK", "DMA_MM2S_1_FINISHED_BD", "DMA_S2MM_1_FINISHED_BD"} <= set(mm)
    assert cal[cal.index("--memmod-sel-locks") + 1] == f"LOCK_SEL1:{g.MR_LOCK_W}"
    assert len(cal[cal.index("--memtile-sweep-events") + 1].split(",")) == 8
    assert cal[cal.index("--port-slots") + 1].endswith(",4=DMA:1:slave")


def test_mr_core_waits_for_the_waker_to_finish_after_b():
    for w in "cbm":
        body = bundles_of(g.mr_core_asm(mr(w)))
        b = body.index(f"acq #{g.OWN + g.FETCH_LOCK_B}, r3")
        assert body.index(f"acq #{g.OWN + g.MR_LOCK_WD}, r3") > b
        assert (f"acq #{g.OWN + g.MR_LOCK_CD}, r3" in body) == (w == "c")
    t = g.mr_design_mlir(mr("c"))
    assert "aie.use_lock(%lWd, Release" in t and "aie.use_lock(%lCd, Release" in t


def test_mr_row_splits_the_memtile_quiet_into_data_and_trace_and_dates_the_waker():
    e = mq_synthetic(fill_end=150)
    e += [ev("LOCK_1_REL", 4000), ev("DMA_MM2S_1_FINISHED_BD", 4030)]
    e += held("DMA_MM2S_1_STALLED_LOCK", 100, 4002)
    r = g.mr_row(e)
    assert r["w_rel"] == 4000 and r["w_grant"] == 3 and r["w_fin"] == 4030 - (5000 + 3)
    assert r["quiet_data"] == 48 and r["quiet_trace"] == 199 - (200 - 393) - 1


# -- sw: the sl design with mr's compute-only waker (where does the compute switch's restart cycle sit?) ------

def test_sw_plan_is_the_baseline_and_the_five_quiet_stretches():
    names = [c.name for c in g.sw_plan()]
    assert names == ["swnone"] + [f"swc{t:03d}" for t in g.SW_TS] and g.SW_TS == (16, 64, 96, 200, 400)


def test_sw_core_releases_w_d_nops_before_a_or_after_b_for_the_baseline():
    c = g.SwCase(96)
    body = bundles_of(g.sw_core_asm(c))
    rw = body.index(f"rel #{g.OWN + g.MR_LOCK_W}, r2")
    ra = [i for i, b in enumerate(body) if b == f"rel #{g.OWN + g.FETCH_LOCK_A}, r2"][0]
    assert body.index("event #0") == rw + g.sw_delay_nops(c) + 1 and ra == body.index("event #0") + g.SL_LOAD_D + 1
    base = bundles_of(g.sw_core_asm(g.SwCase(None)))
    b = base.index(f"acq #{g.OWN + g.FETCH_LOCK_B}, r3")
    assert base.index(f"rel #{g.OWN + g.MR_LOCK_W}, r2") > b                   # the waker runs after the measured BD
    for t in (base, body):
        assert t.index(f"acq #{g.OWN + g.MR_LOCK_CD}, r3") > t.index(f"acq #{g.OWN + g.FETCH_LOCK_B}, r3")
        assert t.count(f"rel #{g.OWN + g.FETCH_LOCK_A}, r2") == 1 + g.SW_FILL_RELEASES


def test_sw_design_is_sl_plus_the_compute_local_waker():
    t = g.sw_design_mlir(g.SwCase(64))
    sl = g.sl_design_mlir(g.SlCase(False, g.SL_LOAD_D))
    assert "aie.flow(%t02, DMA : 0, %t02, DMA : 0)" in t and "aie.flow(%t02, DMA : 1, %t02, DMA : 1)" in t
    assert "aie.use_lock(%lW, AcquireGreaterEqual" in t and "aie.use_lock(%lCd, Release" in t
    for line in sl.splitlines():
        if "aie.dma_bd" in line or "aie.use_lock" in line or "aie.flow" in line:
            assert line in t                                              # sl's own design untouched
    assert g.sw_design_mlir(g.SwCase(None)) == t                             # only the core differs between cases
