"""The DMA task-queue / BD-pipeline probe generator: case tables, designs and status decoding."""
import gen_dma_queue_probe as g


def test_every_queue_and_stall_case_has_a_unique_name_and_a_design():
    table = {**g.cases(), **g.stall_cases()}
    assert len(table) == len(g.cases()) + len(g.stall_cases())
    for name, case in table.items():
        assert case.name == name
        text, steps, layout = g.design(case)
        assert "aie.runtime_sequence" in text
        # every read has a label and every label is a read
        assert len(layout.labels) == len(set(layout.labels)), name


def test_a_chain_declares_one_gate_lock_per_bd_and_opens_all_but_the_closed_one():
    case = g.QueueCase("mem", "s2mm", 1, True, 2, 3, 1)
    text, steps, _ = g.design(case)
    for j in range(3):
        assert f'sym_name = "{g.gate_symbol(j)}"' in text
    opened = [s for s in steps if getattr(s, "label", "").startswith("open gate ")]
    assert [s.label for s in opened] == ["open gate 0", "open gate 2"]


def test_an_ungated_case_takes_no_locks_and_a_partnerless_case_starts_no_partner():
    text, _, _ = g.design(g.QueueCase("core", "s2mm", 1, False, 4, 2, 0, False))
    assert "use_lock" not in text
    assert "%tp" not in text


def test_the_status_decoder_reads_the_fields_the_probe_reports():
    word = 0x06480015     # Cur_BD 6, queue 4, running, starved, acquire none, STARTING
    f = g.decode_status(word, "shim")
    assert (f["cur_bd"], f["queue"], f["running"], f["starve"], f["fsm"]) == (6, 4, 1, 1, 1)
    assert g.decode_status(0x3F000000, "mem")["cur_bd"] == 0x3F


def test_absorb_cases_cover_both_modes_and_name_their_length():
    table = g.absorb_cases((16, 80))
    assert set(table) == {"absorb_16", "absorb_80", "absorb_idle_16", "absorb_idle_80",
                          "absorb_mem_16", "absorb_mem_80", "absorb_mem_idle_16", "absorb_mem_idle_80"}
    text, _, _ = g.absorb_design(table["absorb_80"])
    assert "len = 80" in text


def test_absorb_mem_sends_the_shim_stream_north_to_a_memtile_s2mm_parked_on_a_memtile_lock():
    text, st, _ = g.absorb_design(g.AbsorbCase(72, True, "mem"))
    assert "aie.flow(%t00, DMA : 1, %t01, DMA : 0)" in text
    assert "aiex.dma_configure_task(%t01, S2MM, 0)" in text
    assert f"aie.lock(%t01, {g.GATE_LOCK['mem']})" in text
    assert "@din" not in text                      # the shim S2MM 1 is not used
    reads = {s.label: s.tile for s in st if isinstance(s, g.cp.Read)}
    assert reads["a1 mm2s"] == "shim" and reads["a1 s2mm"] == "mem" and reads["a1 C"] == "mem"
