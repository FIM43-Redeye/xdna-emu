"""Tests for tools/gen_stream_absorb_probe.py: asm, topologies, summary analysis."""
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

import gen_stream_absorb_probe as g
from harness.build import _peano_tool
from harness.topology import CoreStream, load_topology
from harness.validate import validate_topology
from harness import suite


def count(text, needle):
    return sum(1 for l in text.splitlines() if needle in l.split("//")[0])


def test_producer_writes_k_blocking_words_and_stamps_each():
    asm = g.producer_asm(7)
    assert count(asm, "mov ms, r1") == 7
    assert count(asm, "mov.nb") == 0, "blocking writes: the producer must stall, not drop"
    assert count(asm, "mov r17:r16, cntr") == 8      # t_start and one stamp per write


def test_consumer_reads_k_words_after_its_delay():
    asm = g.consumer_asm(5, 100, delay_iters=20)
    assert count(asm, ", SS") == 5
    assert count(asm, "jnz") == 1
    assert asm.index("jnz") < asm.index("mov r2, SS"), "the delay comes before the first read"


def test_delay_bounds():
    with pytest.raises(ValueError):
        g.consumer_asm(4, 0)


@pytest.mark.parametrize("via_fifo", [False, True])
def test_case_topology_routes_the_stream_with_or_without_the_switch_fifo(via_fifo, tmp_path):
    topo, files = g.case(8, 100, via_fifo)
    assert topo.links == (CoreStream("producer", "consumer", via_fifo=via_fifo),)
    assert set(files) == {"producer.s", "consumer.s"}
    assert ("fifo" in topo.name) == via_fifo
    suite.write_case(tmp_path, topo.name, topo, files=files)
    assert validate_topology(load_topology(tmp_path / "cases" / topo.name / "topology.json")) == []


def test_buffer_sizes_hold_the_layouts():
    assert g.ts_bytes(96) == 400                    # 97 words rounded to 16 bytes
    assert g.log_bytes(96) == 400 + 96 * 4          # (96 + 2) words rounded to 16, then the data


def test_summarize_finds_the_first_stalled_write_and_checks_the_words():
    k = 40
    stamps, t = [1000], 1000
    for i in range(k):
        t += 6 if i != 18 else 5000                 # write 18 stalls
        stamps.append(t)
    ts = struct.pack(f"<{k + 1}I", *stamps).ljust(g.ts_bytes(k), b"\0")
    log = bytes(g.log_bytes(k))
    good = log[:16 * ((k + 2 + 3) // 4)] + struct.pack(f"<{k}I", *g.expected_words(k))
    bad = bytearray(good)
    bad[-4:] = struct.pack("<I", 0)
    r = g.summarize(ts, bytes(good), k)
    assert (r["absorbed"], r["all_words_in_order"]) == (18, True)
    assert g.summarize(ts, bytes(bad), k)["all_words_in_order"] is False


@pytest.mark.skipif(shutil.which("llvm-mc") is None and _peano_tool is None, reason="no assembler")
def test_kernels_assemble():
    try:
        mc = _peano_tool("llvm-mc")
    except Exception:
        pytest.skip("Peano llvm-mc not available")
    for text in (g.producer_asm(4), g.consumer_asm(4, 10)):
        r = subprocess.run([str(mc), "--triple=aie2", "--filetype=obj", "-o", "/dev/null"],
                           input=text, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
