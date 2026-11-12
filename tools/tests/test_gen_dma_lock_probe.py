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
