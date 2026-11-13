"""The core load-port probe generator: bank arithmetic, case plan, kernel text, loser model, table rows."""
import pytest

import gen_loadport_probe as g


# -- bank arithmetic (a mirror of the emulator's BankLayout::Compute) --------------------------------

def test_physical_banks_interleave_16_byte_words_inside_16k_logical_banks():
    assert [g.physical_bank(a) for a in (0x0, 0x10, 0x20, 0x30)] == [0, 1, 0, 1]
    assert g.physical_bank(0x4000) == 2 and g.physical_bank(0x4010) == 3
    assert g.physical_bank(0x8000) == 4 and g.physical_bank(0xC010) == 7


def test_accesses_round_down_to_their_width_and_span_the_words_they_cover():
    assert g.access_banks(0x800 + 4, 4) == {0}          # a word load stays in its 16-byte word
    assert g.access_banks(0x810, 4) == {1}
    assert g.access_banks(0x800 + 48, 32) == {0, 1}     # a 256-bit access, rounded down to 32 bytes
    assert g.access_banks(0x800 + 16, 32) == {0, 1}     # the rounding puts the first word on bank 0
    assert g.access_banks(0x800 + 32, 16) == {0}
    assert g.access_banks(0x4800, 32) == {2, 3}


def test_a_vector_loads_first_word_is_always_the_even_bank_and_its_second_the_odd_one():
    for base in (0x800, 0x4800):
        for off in (0, 32, 64, 96):
            first, second = g.vector_banks_in_order(base + off)
            assert first % 2 == 0 and second == first + 1


# -- the case plan ------------------------------------------------------------------------------------

def test_every_case_has_a_unique_name_and_every_buffer_in_logical_banks_0_and_1():
    plan = g.plan()
    assert len({c.name for c in plan}) == len(plan)
    for c in plan:
        for acc in (c.first, c.second):
            assert g.access_banks(acc.address, acc.width) <= {0, 1, 2, 3}, c.name


def test_cases_that_must_meet_share_a_bank_and_negative_controls_share_none():
    for c in g.plan():
        shared = g.meeting_banks(c)
        assert bool(shared) == c.meets, c.name


def test_the_two_accesses_of_a_case_use_two_different_ports_of_one_bundle():
    for c in g.plan():
        assert {c.first.port, c.second.port} <= {"A", "B", "S"}
        assert c.first.port != c.second.port, c.name


def test_scalar_loads_only_exist_on_port_a_so_two_scalar_loads_cannot_meet():
    for c in g.plan():
        for acc in (c.first, c.second):
            if acc.kind == "scalar":
                assert acc.port == "A", c.name
    assert not any(c.first.kind == "scalar" and c.second.kind == "scalar" for c in g.plan())
    assert "scalar load" in g.TWO_SCALAR_LOADS_IMPOSSIBLE and "slot" in g.TWO_SCALAR_LOADS_IMPOSSIBLE


def test_the_plan_has_the_families_the_readings_need():
    names = {c.name for c in g.plan()}
    # a: shared bank is the vector's first bank, b: its second, primed twins flip the loser
    assert {"la0", "lb0", "la1", "lb1"} <= names
    first = next(c for c in g.plan() if c.name == "la0")
    second = next(c for c in g.plan() if c.name == "lb0")
    assert g.meeting_banks(first) == {0} and g.vector_banks_in_order(first.second.address)[0] == 0
    assert g.meeting_banks(second) == {1} and g.vector_banks_in_order(second.second.address)[1] == 1
    assert any(c.family == "c" for c in g.plan())
    assert any(c.family == "d" for c in g.plan())
    assert any(c.family == "e" and not c.meets for c in g.plan())


# -- kernel text --------------------------------------------------------------------------------------

def test_the_meeting_is_one_bundle_of_two_slots():
    la = next(c for c in g.plan() if c.name == "la0")
    assert g.meeting_bundle(la) == "  lda r26, [p0, #0] ; vldb wl4, [p0, #32]"
    d1 = next(c for c in g.plan() if c.name == "d1_0")
    assert g.meeting_bundle(d1).split(";")[1].strip().startswith("st r2, [p0, #")


def test_the_128_bit_vector_loads_use_the_index_form_the_assembler_has():
    c3 = next(c for c in g.plan() if c.name == "c3")
    text = g.meeting_bundle(c3)
    assert "vlda.128 wl2, [p0]" in text and "vldb.128 wl6, [p2, dj0]" in text


def test_the_kernel_marks_the_window_and_primes_only_when_asked():
    plain = g.core_asm(next(c for c in g.plan() if c.name == "la0"))
    primed = g.core_asm(next(c for c in g.plan() if c.name == "la1"))
    for text in (plain, primed):
        assert text.count("event #0") == 1 and text.count("event #1") == 1
        assert "event #2" not in text and "event #3" not in text      # `event` reaches only 0 and 1
        assert f"acq #{g.OWN + g.LOCK_GO}" in text
        assert text.count(f"rel #{g.OWN + g.LOCK_A}") == 2       # the anchor, and the one that ends the kernel
    # the prime is a load and a store of one bank in one bundle, before event #0
    assert "lda r27, [p0" not in plain
    assert primed.index("lda r27, [p0") < primed.index("event #0")
    assert primed.count("lda r27, [p0") == 1 and primed.count("st r2, [p0, #0]") >= 1


def test_the_meeting_bundle_sits_k_nops_after_the_release():
    c = next(c for c in g.plan() if c.name == "lb0")
    lines = g.core_asm(c).splitlines()
    rel = next(i for i, l in enumerate(lines) if f"rel #{g.OWN + g.LOCK_A}" in l)   # the first
    bundle = next(i for i, l in enumerate(lines) if l == g.meeting_bundle(c))
    assert lines[rel + 1:bundle] == [g.N] * g.K_NOPS


# -- the emulator's loser (round-robin rotor of bank_arbiter.rs) -------------------------------------

def test_a_fresh_bank_gives_priority_to_load_a_then_load_b_then_the_store():
    assert g.rotor_loser(("A", "B"), 0) == "B"
    assert g.rotor_loser(("A", "S"), 0) == "S"
    assert g.rotor_loser(("B", "S"), 0) == "S"


def test_one_prior_contended_cycle_moves_the_priority_on_by_one_port():
    assert g.rotor_loser(("A", "B"), 1) == "A"
    assert g.rotor_loser(("A", "S"), 1) == "A"       # the store now beats load A
    assert g.rotor_loser(("B", "S"), 1) == "S"
    assert g.rotor_loser(("B", "S"), 2) == "B"
    assert g.rotor_loser(("A", "B"), 2) == "B"
    assert g.rotor_loser(("A", "B"), 7) == g.rotor_loser(("A", "B"), 0)   # a period of 7 requesters


def test_primed_twins_make_the_other_port_lose_where_the_plan_says_so():
    by_name = {c.name: c for c in g.plan()}
    assert g.predicted_losers(by_name["la0"]) == {0: "B"}
    assert g.predicted_losers(by_name["la1"]) == {0: "A"}
    assert g.predicted_losers(by_name["lb0"]) == {1: "B"}
    assert g.predicted_losers(by_name["lb1"]) == {1: "A"}
    assert g.predicted_losers(by_name["d1_0"]) == {0: "S"}
    assert g.predicted_losers(by_name["d1_1"]) == {0: "A"}
    assert g.predicted_losers(by_name["e1"]) == {}


# -- what each reading predicts -----------------------------------------------------------------------

def test_every_reading_agrees_the_core_stalls_and_the_positive_controls_stamp():
    for c in g.plan():
        if c.family == "d":
            for r in ("R1", "R2", "R3"):
                assert g.predicted_stamps(c, r) == g.meeting_banks(c), (c.name, r)
        if not c.meets:
            for r in ("R0", "R1", "R2", "R3"):
                assert g.predicted_stamps(c, r) == set()


def test_r1_never_stamps_load_against_load_r2_drops_the_vectors_second_bank():
    by_name = {c.name: c for c in g.plan()}
    assert g.predicted_stamps(by_name["la0"], "R0") == {0}      # the emulator: every contended bank
    assert g.predicted_stamps(by_name["la0"], "R1") == set()
    assert g.predicted_stamps(by_name["la0"], "R2") == {0}      # first bank stamped
    assert g.predicted_stamps(by_name["lb0"], "R2") == set()    # second bank not stamped
    assert g.predicted_stamps(by_name["c1"], "R2") == {0}       # two vectors: bank 1 is a second bank
    assert g.predicted_stamps(by_name["c1"], "R0") == {0, 1}


def test_r3_stamps_only_when_load_a_loses():
    by_name = {c.name: c for c in g.plan()}
    assert g.predicted_stamps(by_name["la0"], "R3") == set()    # load B loses: the seed's picture
    assert g.predicted_stamps(by_name["la1"], "R3") == {0}      # load A loses once the rotor has moved


# -- decoding a run -----------------------------------------------------------------------------------

class Ev:
    def __init__(self, name, ts, pkt_type, row=2):
        self.name, self.ts, self.pkt_type, self.row = name, ts, pkt_type, row


def synth(core, mem):
    return ([Ev(n, t, g.PT_CORE) for n, ts in core.items() for t in ts]
            + [Ev(n, t, g.PT_MEM) for n, ts in mem.items() for t in ts])


def test_a_row_reads_stalls_from_event_0_and_conflicts_from_the_release():
    ev = synth({"INSTR_EVENT_0": [1000], "INSTR_EVENT_1": [1020], "MEMORY_STALL": [1013]},
               {"LOCK_0_REL": [500, 1900], "CONFLICT_DM_BANK_0": [511]})
    r = g.row(ev)
    assert r["memory_stalls"] == [13]
    assert r["conflicts"] == [("CONFLICT_DM_BANK_0", 11)]
    assert r["window"] == 20 and r["release"] == 500 and r["releases"] == 2


def test_a_primed_run_keeps_the_prime_stamps_apart_from_the_measured_ones():
    ev = synth({"INSTR_EVENT_0": [1000], "INSTR_EVENT_1": [1020], "MEMORY_STALL": [920, 1013]},
               {"LOCK_0_REL": [500], "CONFLICT_DM_BANK_0": [431, 511]})
    r = g.row(ev)
    assert r["memory_stalls"] == [13] and r["prime_stalls"] == [-80]
    assert r["conflicts"] == [("CONFLICT_DM_BANK_0", 11)]
    assert r["prime_conflicts"] == [("CONFLICT_DM_BANK_0", -69)]


def test_the_finish_check_names_what_is_missing():
    c = next(c for c in g.plan() if c.name == "la1")
    ok = g.row(synth({"INSTR_EVENT_0": [1000], "INSTR_EVENT_1": [1020]}, {"LOCK_0_REL": [500, 1900]}))
    assert g.finish_missing(ok, c, halt="completed") == []
    bad = g.row(synth({"INSTR_EVENT_0": [1000]}, {"LOCK_0_REL": [500]}))
    missing = g.finish_missing(bad, c, halt="completed")
    assert any("INSTR_EVENT_1" in m for m in missing) and any("LOCK_0_REL x1" in m for m in missing)
    assert g.finish_missing(ok, c, halt="max_cycles")[0].startswith("halt_reason")


def test_the_table_line_carries_stall_conflicts_loser_and_finish():
    c = next(c for c in g.plan() if c.name == "la0")
    r = g.row(synth({"INSTR_EVENT_0": [1000], "INSTR_EVENT_1": [1020], "MEMORY_STALL": [1010]},
                    {"LOCK_0_REL": [500, 1900], "CONFLICT_DM_BANK_0": [510]}))
    r["check"] = []
    text = g.format_table([(c, "e1.emu", r)])
    line = [l for l in text.splitlines() if l.startswith("la0")][0]
    assert "stall +10" in line and "B0@+10" in line and "loser B" in line and "finish ok" in line


def test_the_design_has_no_dma_and_takes_its_go_from_the_sequence():
    text = g.design_mlir()
    assert "aie.dma_start" not in text and "aie.mem" not in text
    assert "aiex.set_lock(%lGo" in text and text.count("aiex.npu.write32") == g.PAD_WRITES
    assert "link_with = \"core.o\"" in text
