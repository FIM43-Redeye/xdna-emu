"""Tests for tools/gen_trace_wake_probe.py: plan, core program, design text, table logic."""
import re
import subprocess

import pytest

import gen_trace_wake_probe as g
from harness.build import _peano_tool


def peano_mc():
    try:
        return _peano_tool("llvm-mc")
    except Exception:
        return None


def body_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines()]


# -- the plan and the frame accounting ----------------------------------------------

def plan_bytes(p: g.PacketPlan) -> int:
    """Payload bytes of one planned packet under the frame rules in the module doc."""
    if p.kind == "head":
        return 8 + 3 + (p.burst - 1)                 # Start, Single2, Single0s
    if p.kind == "dense":
        return p.burst                               # Single0s
    return p.burst + 2                               # Single0s and the completing Single1


def test_every_planned_packet_is_exactly_one_packet():
    plan = g.default_plan()
    g.validate_plan(plan)
    assert all(plan_bytes(p) == g.PACKET_BYTES for p in plan)


def test_plan_covers_the_probe_spacings():
    idles = {p.idle for p in g.default_plan() if p.idle is not None}
    assert {100, 120, 135, 150} <= idles and max(idles) >= 300
    assert sum(p.kind == "dense" for p in g.default_plan()) >= 5     # a dense baseline
    assert [p.kind for p in g.default_plan()][0] == "head"


def test_completing_delta_is_a_single1_for_every_spaced_packet():
    for p in g.default_plan():
        if p.kind == "spaced":
            assert g.SINGLE0_MAX_DELTA < g.completion_delta(p) <= g.SINGLE1_MAX_DELTA


def test_validate_plan_rejects_a_gap_too_short_for_a_single1_and_a_missing_head():
    with pytest.raises(ValueError):
        g.validate_plan([g.PacketPlan("head"), g.PacketPlan("spaced", 40)])
    with pytest.raises(ValueError):
        g.validate_plan([g.PacketPlan("dense")])


def test_head_delay_forces_a_single2():
    assert g.HEAD_DELAY > g.SINGLE1_MAX_DELTA


# -- the core program ---------------------------------------------------------------

def kernel_body(text: str) -> list[str]:
    lines = body_lines(text)
    start = max(i for i, ln in enumerate(lines) if ln.startswith("st r31,")) + 1
    end = min(i for i, ln in enumerate(lines) if ln.startswith("lda r16,"))
    return [ln for ln in lines[start:end] if ln]


def test_core_program_event_count_and_alternation():
    plan = g.default_plan()
    body = kernel_body(g.core_asm(plan))
    planned = [ln for ln in body[:body.index(".Ltail:")] if ln.startswith("event")]
    assert len(planned) == sum(p.burst + (p.kind == "spaced") for p in plan)
    assert [int(ln[-1]) for ln in planned] == [i % 2 for i in range(len(planned))]   # ids alternate
    tail = [ln for ln in body[body.index(".Ltail:"):] if ln.startswith("event")]
    assert len(tail) == g.PACKET_BYTES                                       # one packet per loop iteration
    assert int(tail[0][-1]) == len(planned) % 2                              # the alternation carries on


def test_core_only_waits_on_the_go_lock_and_touches_no_other_lock():
    body = kernel_body(g.core_asm())
    acquires = [ln for ln in body if ln.startswith("acq")]
    assert acquires == [f"acq #{g.OWN + g.LOCK_GO}, r3"]
    assert not any(ln.startswith(("rel", "vld", "vst", "lda r", "st r")) for ln in body)
    assert body.index(acquires[0]) < min(i for i, ln in enumerate(body) if ln.startswith("event"))


def test_first_event_follows_the_long_head_delay():
    body = kernel_body(g.core_asm())
    first = min(i for i, ln in enumerate(body) if ln.startswith("event"))
    loop = [ln for ln in body[:first] if ln.startswith("movxm")]
    iters = int(re.search(r"#(\d+)", loop[0]).group(1))
    assert iters * g.LOOP_CYCLES >= g.HEAD_DELAY - 2 * g.LOOP_CYCLES


def test_wait_bundles_is_nops_when_short_and_a_loop_when_long():
    assert g.wait_bundles(0, ".L") == [] and g.wait_bundles(3, ".L") == ["  nop"] * 3
    long = g.wait_bundles(400, ".L")
    assert any(ln.strip().startswith("movxm") for ln in long) and any("jnz" in ln for ln in long)
    iters = int(re.search(r"#(\d+)", [ln for ln in long if ln.strip().startswith("movxm")][0]).group(1))
    assert abs(iters * g.LOOP_CYCLES + g.LOOP_FIXED - 400) < g.LOOP_CYCLES      # nops make up the remainder


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
def test_core_assembles(tmp_path):
    s = tmp_path / "core.s"
    s.write_text(g.core_asm())
    r = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / "core.o"), str(s)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# -- the design text ----------------------------------------------------------------

def test_design_has_no_data_traffic_and_no_waits():
    t = g.design_mlir()
    assert "aie.device(npu1_1col)" in t
    assert "aie.flow" not in t and "aie.mem(" not in t and "aie.dma" not in t
    assert "aie.shim_dma_allocation" not in t
    seq = t.split("aie.runtime_sequence")[1]
    assert "aiex.set_lock(%lGo" in seq
    for forbidden in ("dma_await", "dma_wait", "dma_memcpy_nd", "dma_start_task", "npu.sync"):
        assert forbidden not in t
    ops = re.findall(r"aiex?\.[a-z_.0-9]+", seq)
    assert set(ops) == {"aiex.set_lock", "aiex.npu.write32"}                 # only plain writes and the lock set
    assert 'aie.lock(%t02, 0) {init = 0 : i32, sym_name = "lGo"}' in t


def test_inject_traces_two_core_events_and_one_shim_event_without_grounding():
    args = g.inject_args()
    assert args[args.index("--core-sweep-events") + 1] == "INSTR_EVENT_0,INSTR_EVENT_1"
    assert args[args.index("--core-grounding") + 1] == "INSTR_EVENT_0,INSTR_EVENT_1"
    assert args[args.index("--shim-sweep-events") + 1] == "DMA_S2MM_1_STREAM_STARVATION"
    assert args[args.index("--shim-grounding") + 1] == ""
    assert "PERF_CNT_2" not in " ".join(args)


# -- frames and packets -------------------------------------------------------------

def single0(slot: int, delta: int) -> list[int]:
    return [(slot << 4) | delta]


def single1(slot: int, delta: int) -> list[int]:
    return [0x80 | (slot << 2) | (delta >> 8), delta & 0xFF]


def single2(slot: int, delta: int) -> list[int]:
    return [0xA0 | (slot << 2) | (delta >> 16), (delta >> 8) & 0xFF, delta & 0xFF]


def start(timer: int) -> list[int]:
    return [0xF0] + list(timer.to_bytes(7, "big"))


def header(pkt_type: int, row: int = 2, col: int = 1, pid: int = 1) -> int:
    w = (col << 21) | (row << 16) | (pkt_type << 12) | pid
    return w | (0x80000000 if bin(w).count("1") % 2 == 0 else 0)


def packetize(stream: list[int], pkt_type: int) -> list[list[int]]:
    """Whole packets (header word + 7 words) of a byte stream; the stream must be 28-byte multiples."""
    assert len(stream) % 28 == 0
    out = []
    for i in range(0, len(stream), 28):
        words = [int.from_bytes(bytes(stream[i + 4 * j:i + 4 * j + 4]), "big") for j in range(7)]
        out.append([header(pkt_type)] + words)
    return out


def test_frame_len_and_type_match_the_decoder_table():
    assert (g.frame_len(0x11), g.frame_type(0x11)) == (1, "Single0")
    assert (g.frame_len(0x85), g.frame_type(0x85)) == (2, "Single1")
    assert (g.frame_len(0xA0), g.frame_type(0xA0)) == (3, "Single2")
    assert (g.frame_len(0xC1), g.frame_type(0xC1)) == (2, "Multiple0")
    assert g.frame_len(0xF0) == 8 and g.frame_type(0xF0) == "Start"
    assert (g.frame_len(0xD1), g.frame_len(0xD5), g.frame_len(0xD9)) == (3, 4, 2)
    assert g.frame_len(0xE3) == 1 and g.frame_type(0xE3) == "Repeat0"


def test_split_packets_keeps_arrival_order_and_sources():
    a = packetize(start(100) + single2(0, 1900) + single0(1, 1) * 17, 0)[0]
    b = packetize(single0(0, 1) * 28, 2)[0]
    c = packetize(single0(1, 1) * 28, 0)[0]
    pk = g.split_packets(a + b + c)
    assert [(p["order"], p["pkt_type"]) for p in pk] == [(0, 0), (1, 2), (2, 0)]
    assert len(pk[0]["bytes"]) == 28


def synthetic_stream(gaps):
    """Core packets whose completing events come `gaps` cycles apart: a head packet, then per gap a
    26-Single0 burst (delta 1) and a Single1 completing frame, or 28 Single0 for gap == 56."""
    stream = start(1000) + single2(0, 100) + single0(1, 1) * 17     # head: completes at 1000 + 101 + 17*2
    t = 1000 + 101 + 17 * 2
    times = [t]
    for gap in gaps:
        if gap == 56:
            stream += single0(0, 1) * 28
        else:
            stream += single0(0, 1) * 26 + single1(1, gap - 52 - 1)
        t += gap
        times.append(t)
    return stream, times


def test_completing_frames_dates_each_packet_by_its_last_byte():
    stream, times = synthetic_stream([56, 408, 56])
    words = [w for p in packetize(stream, 0) for w in p]
    done = g.completing_frames(g.split_packets(words))
    assert [done[i][0] for i in range(4)] == times
    assert [done[i][1] for i in range(4)] == ["Single0", "Single0", "Single1", "Single0"]


def test_completing_frame_that_runs_into_the_next_packet_is_starred():
    # 27 Single0 and a Single1: its first byte is the 28th byte of the packet, its second opens the next
    stream = single0(0, 1) * 27 + single1(1, 300) + single0(0, 1) * 27
    words = [w for p in packetize(stream, 0) for w in p]
    done = g.completing_frames(g.split_packets(words))
    assert done[0][1] == "Single1*"


def test_arrival_slots_from_a_starvation_level():
    starved = list(range(0, 100)) + list(range(108, 140)) + list(range(172, 200))   # 8, then 32 free cycles
    assert g.arrival_slots(starved) == [100, 140, 148, 156, 164]
    assert g.arrival_slots([1, 2, 3, 4]) == []                                     # no gap, no packet


def test_packet_rows_delay_idle_and_contamination():
    stream, times = synthetic_stream([56, 408, 56])
    core = packetize(stream, 0)
    shim = packetize(single0(0, 1) * 28, 2)
    order = [core[0], core[1], core[2], shim[0], core[3]]
    words = [w for p in order for w in p]
    packets = g.split_packets(words)
    # arrival = completion + 11; the shim packet lands 30 cycles after core[2]'s
    arrivals = [times[0] + 11, times[1] + 11, times[2] + 11, times[2] + 41, times[3] + 11]
    starved, prev_end = list(range(arrivals[0] - 30, arrivals[0])), None
    for a in arrivals:
        if prev_end is not None:
            starved += list(range(prev_end, a))
        prev_end = a + g.PACKET_CYCLES
    starved += list(range(prev_end, prev_end + 5))
    rows = g.packet_rows(packets, starved)
    assert [r["order"] for r in rows] == [0, 1, 2, 4]                     # core packets only, raw order
    assert rows[1]["idle"] == 56 - g.PACKET_CYCLES and rows[1]["phase"] == "dense"
    assert rows[2]["idle"] == 408 - g.PACKET_CYCLES and rows[2]["phase"] == "gated"
    assert [r["delay"] for r in rows] == [11, 11, 11, 11]
    assert rows[3]["prev_src"] == "shim" and rows[3]["since_any"] == 18 and rows[3]["contaminated"]
    assert not rows[2]["contaminated"] and rows[2]["since_any"] == 400
    text = g.format_rows(rows)
    assert text.splitlines()[0].split()[:3] == ["pkt", "phase", "idle"]
    assert "gated" in text


def test_phase_boundaries():
    assert [g.phase_of(i) for i in (48, 63, 64, 100, 127, 128, 400)] == \
        ["dense", "dense", "near", "near", "near", "gated", "gated"]


def row(order, phase, idle, delay, contaminated=False, ftype="Single0"):
    return {"order": order, "phase": phase, "idle": idle, "delay": delay, "contaminated": contaminated,
            "ftype": ftype, "completion": 0, "arrival": 0, "since_any": 100, "prev_src": "core"}


def test_summary_compares_gated_rows_with_the_dense_median_and_skips_contaminated_ones():
    rows = [row(i, "dense", 48, 11) for i in range(1, 6)]
    rows += [row(6, "dense", 48, 99, contaminated=True)]                   # ignored
    rows += [row(7, "gated", 400, 13), row(8, "gated", 400, 13), row(9, "near", 100, 11),
             row(10, "gated", 135, 12, ftype="Single1*")]                  # odd frame: ignored
    text = g.summarize(rows)
    assert "dense baseline: median delay 11" in text
    assert "200+: n=2 median delay 13" in text
    assert "64..109: n=1" in text and "128..142" not in text


def test_summary_without_dense_rows_says_so():
    assert "no baseline" in g.summarize([row(1, "gated", 400, 12)])


# -- the compiled sequence ----------------------------------------------------------

def txn(*ops: bytes) -> bytes:
    import struct
    return bytes([0, 1, 3, 6, 1, 1, 0, 0]) + struct.pack("<II", len(ops), 16 + sum(map(len, ops))) + b"".join(ops)


def op(code: int, size: int) -> bytes:
    import struct
    b = bytearray(size)
    b[0] = code
    if code == 1:
        struct.pack_into("<I", b, 12, size)
    if code == 0x81:
        struct.pack_into("<I", b, 4, size)
    return bytes(b)


def test_sequence_check_accepts_plain_writes_and_the_address_patch():
    data = txn(op(0, 24), op(1, 40), op(3, 28), op(0x81, 48), op(0, 24))
    assert g.check_sequence_is_plain_writes(data) == {0: 2, 1: 1, 3: 1, 0x81: 1}


@pytest.mark.parametrize("code", [4, 7, 0x80])
def test_sequence_check_refuses_anything_else(code):
    with pytest.raises(ValueError):
        g.check_sequence_is_plain_writes(txn(op(0, 24), op(code, 28)))


def test_sequence_check_refuses_a_stream_that_does_not_parse_to_its_length():
    with pytest.raises(ValueError):
        g.check_sequence_is_plain_writes(txn(op(0, 24)) + bytes(4))
