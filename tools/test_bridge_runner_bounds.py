#!/usr/bin/env python3
"""bridge-trace-runner's host-buffer bounds: buffer sizing and the pre-run refusal.

The runner decodes the submitted instruction stream completely (bridge-runner/insts_bounds.h),
evaluates every shim DMA task it pushes to a byte range in the host buffer its BD is patched onto,
sizes each data buffer to the largest end, and refuses a run whose tasks leave the buffers it
allocates. These tests drive the runner's device-free modes:

  --print-arg-sizes INSTS             "IDX BYTES" per data buffer
  --check-bounds INSTS [IDX=BYTES..]  one line per task; exit 0 inside, 3 outside, 2 undecodable

Expected values are written out from the register semantics, not computed by a Python copy of the
decoder: a mirror that shares the decoder's assumptions also shares its blind spots.
"""
import json
import os
import re
import struct
import subprocess
from pathlib import Path

import pytest

MAGIC = 0x06030100
BD_BASE, BD_STRIDE = 0x1D000, 0x20
QUEUE = {"S2MM0": 0x1D204, "S2MM1": 0x1D20C, "MM2S0": 0x1D214, "MM2S1": 0x1D21C}
DDR_AIE_ADDR_OFFSET = 0x80000000

REPO = Path(__file__).resolve().parents[1]


def _runner():
    path = os.environ.get("BRIDGE_TRACE_RUNNER", str(REPO / "bridge-runner" / "build" / "bridge-trace-runner"))
    if not os.path.exists(path):
        pytest.skip("bridge-trace-runner not built (set BRIDGE_TRACE_RUNNER)")
    return path


def addr(col: int, row: int, reg: int) -> int:
    return (col << 25) | (row << 20) | reg


class Stream:
    """An instruction stream in the insts.bin transaction format."""

    def __init__(self):
        self.ops: list[bytes] = []

    def write32(self, reg: int, value: int, col: int = 0, row: int = 0):
        self.ops.append(struct.pack("<IIIIII", 0, 0, addr(col, row, reg), 0, value, 24))
        return self

    def blockwrite(self, reg: int, words: list[int], col: int = 0, row: int = 0):
        self.ops.append(struct.pack("<IIII", 1, 0, addr(col, row, reg), 16 + 4 * len(words))
                        + struct.pack(f"<{len(words)}I", *words))
        return self

    def maskwrite(self, reg: int, value: int, mask: int, col: int = 0, row: int = 0):
        self.ops.append(struct.pack("<IIIIIII", 3, 0, addr(col, row, reg), 0, value, mask, 28))
        return self

    def maskpoll(self, reg: int, value: int, mask: int, col: int = 0, row: int = 0):
        self.ops.append(struct.pack("<IIIIIII", 4, 0, addr(col, row, reg), 0, value, mask, 28))
        return self

    def sync(self, col: int = 0, channel: int = 0):
        # XAIE_IO_CUSTOM_OP_TCT: opcode, size, then the sync descriptor.
        self.ops.append(struct.pack("<IIII", 0x80, 16, col, 0x10100 | channel))
        return self

    def bd(self, bd: int, length_words: int, d0=(0, 0), d1=(0, 0), d2_step=0, iteration=(0, 0, 0),
           next_bd=None, col: int = 0):
        """BD words: dims as (wrap, actual step); iteration (current, actual wrap, actual step)."""
        def step(s):
            return max(s - 1, 0) & 0xFFFFF
        w = [0] * 8
        w[0] = length_words
        w[3] = (d0[0] << 20) | step(d0[1])
        w[4] = (d1[0] << 20) | step(d1[1])
        w[5] = step(d2_step)
        cur, wrap, st = iteration
        w[6] = (cur << 26) | (max(wrap - 1, 0) << 20) | step(st)
        w[7] = (1 << 25) | ((1 << 26) | (next_bd << 27) if next_bd is not None else 0)
        return self.blockwrite(BD_BASE + bd * BD_STRIDE, w, col=col)

    def patch(self, bd: int, arg_idx: int, arg_plus: int, col: int = 0, reg: int | None = None):
        op = bytearray(48)
        struct.pack_into("<II", op, 0, 0x81, 48)
        struct.pack_into("<I", op, 24, addr(col, 0, BD_BASE + bd * BD_STRIDE + 4 if reg is None else reg))
        op[32] = arg_idx
        struct.pack_into("<Q", op, 40, arg_plus)
        self.ops.append(bytes(op))
        return self

    def push(self, channel: str, bd: int, token: bool = True, col: int = 0):
        return self.write32(QUEUE[channel], (0x80000000 if token else 0) | bd, col=col)

    def task(self, channel: str, bd: int, arg_idx: int, arg_plus: int, length_words: int, **kw):
        return self.bd(bd, length_words, **kw).patch(bd, arg_idx, arg_plus).push(channel, bd)

    def raw(self, op: bytes):
        self.ops.append(op)
        return self

    def bytes(self, num_ops: int | None = None, total: int | None = None) -> bytes:
        body = b"".join(self.ops)
        return struct.pack("<IIII", MAGIC, 0, len(self.ops) if num_ops is None else num_ops,
                           16 + len(body) if total is None else total) + body


def run(tmp_path, data: bytes, *args):
    f = tmp_path / "insts.bin"
    f.write_bytes(data)
    return subprocess.run([_runner(), *args[:1], str(f), *args[1:]], capture_output=True, text=True)


def sizes(tmp_path, data: bytes) -> dict[int, int]:
    p = run(tmp_path, data, "--print-arg-sizes")
    assert p.returncode == 0, p.stderr
    return {int(a): int(b) for a, b in (ln.split() for ln in p.stdout.splitlines() if ln.strip())}


# -- sizing ---------------------------------------------------------------------------------------

def test_a_task_needs_its_patch_offset_plus_its_length(tmp_path):
    s = Stream().task("S2MM0", 0, 0, 4096, 16)
    assert sizes(tmp_path, s.bytes()) == {0: 4096 + 64}


def test_every_task_counts_not_only_those_before_the_first_sync(tmp_path):
    # The defect this guards: a walker that stops at the first op it does not size (the
    # task-complete-token sync, a mask poll) never sees the tasks after it.
    s = (Stream().task("MM2S0", 0, 0, 0, 100).task("S2MM0", 1, 0, 400, 100).sync()
         .maskpoll(0x1D228, 0, 0x7)
         .task("MM2S0", 2, 0, 800, 8).task("S2MM0", 3, 0, 832, 8).sync())
    assert sizes(tmp_path, s.bytes()) == {0: 864}


def test_a_patch_binds_to_the_bd_it_names_not_to_the_last_bd_written(tmp_path):
    s = (Stream().bd(0, 16).bd(1, 1000)
         .patch(0, 0, 0).patch(1, 0, 4096)
         .push("MM2S0", 0).push("S2MM0", 1))
    assert sizes(tmp_path, s.bytes()) == {0: 4096 + 4000}


def test_a_chained_bd_counts_for_the_task_that_starts_its_chain(tmp_path):
    s = (Stream().bd(0, 16, next_bd=1).patch(0, 0, 0)
         .bd(1, 16).patch(1, 0, 8192)
         .push("S2MM0", 0))
    assert sizes(tmp_path, s.bytes()) == {0: 8192 + 64}


def test_dimensions_bound_the_reach_not_the_length(tmp_path):
    # 64 words, d0: wrap 8, step 1; d1: step 16, no wrap -> last word at 7*1 + 7*16 = 119.
    s = Stream().task("MM2S0", 0, 0, 0, 64, d0=(8, 1), d1=(0, 16))
    assert sizes(tmp_path, s.bytes()) == {0: 120 * 4}
    # 3-D: 2*3*4 words, d0 wrap 2 step 1, d1 wrap 3 step 10, d2 step 100 -> 1 + 20 + 300 = 321.
    s = Stream().task("MM2S0", 0, 0, 0, 24, d0=(2, 1), d1=(3, 10), d2_step=100)
    assert sizes(tmp_path, s.bytes()) == {0: 322 * 4}
    # A transposing walk whose last word is not its farthest: d0 wrap 4 step 8, d1 step 1, 6 words
    # -> words at 0, 8, 16, 24, 1, 9: the reach is word 24.
    s = Stream().task("MM2S0", 0, 0, 0, 6, d0=(4, 8), d1=(0, 1))
    assert sizes(tmp_path, s.bytes()) == {0: 25 * 4}


def test_the_iteration_offset_is_bounded_at_its_wrap(tmp_path):
    # 16 words per execution, iteration wrap 4, step 32 words: up to 3*32 + 16 words.
    s = Stream().task("S2MM0", 0, 0, 0, 16, iteration=(0, 4, 32))
    assert sizes(tmp_path, s.bytes()) == {0: (96 + 16) * 4}


def test_a_folded_aperture_offset_is_not_extent(tmp_path):
    s = Stream().task("S2MM0", 0, 5, DDR_AIE_ADDR_OFFSET + 128, 16)
    assert sizes(tmp_path, s.bytes()) == {5: 128 + 64}


def test_tasks_on_two_columns_and_buffers(tmp_path):
    s = (Stream().bd(15, 4096, col=1).patch(15, 1, 0, col=1).push("S2MM1", 15, col=1)
         .task("S2MM0", 0, 0, 0, 2048))
    assert sizes(tmp_path, s.bytes()) == {0: 8192, 1: 16384}


def test_a_masked_write_to_a_bd_word_is_applied(tmp_path):
    s = Stream().bd(0, 16).maskwrite(BD_BASE, 64, 0xFFFFFFFF).patch(0, 0, 0).push("S2MM0", 0)
    assert sizes(tmp_path, s.bytes()) == {0: 256}


# -- what cannot be bounded is refused ----------------------------------------------------------

def _undecodable(tmp_path, data: bytes, needle: str):
    for mode in ("--print-arg-sizes", "--check-bounds"):
        p = run(tmp_path, data, mode)
        assert p.returncode == 2, (mode, p.stdout, p.stderr)
        assert needle in p.stderr, p.stderr


def test_an_unknown_opcode_is_refused(tmp_path):
    s = Stream().task("S2MM0", 0, 0, 0, 16).raw(struct.pack("<II", 0x02, 16) + b"\0" * 8)
    _undecodable(tmp_path, s.bytes(), "opcode 0x02 is not decoded")


def test_a_stream_that_does_not_end_where_its_header_says_is_refused(tmp_path):
    s = Stream().task("S2MM0", 0, 0, 0, 16)
    _undecodable(tmp_path, s.bytes(num_ops=len(s.ops) - 1), "ops end at byte")
    _undecodable(tmp_path, s.bytes(total=16), "header says")


def test_a_task_on_an_unpatched_bd_is_refused(tmp_path):
    _undecodable(tmp_path, Stream().bd(0, 16).push("S2MM0", 0).bytes(), "carries no host-buffer patch")


def test_rewriting_a_bd_address_unbinds_its_patch(tmp_path):
    s = Stream().bd(0, 16).patch(0, 0, 0).write32(BD_BASE + 4, 0x1000).push("S2MM0", 0)
    _undecodable(tmp_path, s.bytes(), "carries no host-buffer patch")


def test_a_patch_of_a_register_that_is_not_a_bd_address_is_refused(tmp_path):
    s = Stream().bd(0, 16).patch(0, 0, 0, reg=BD_BASE).push("S2MM0", 0)
    _undecodable(tmp_path, s.bytes(), "not a shim BD's base address")


def test_a_bad_magic_is_refused(tmp_path):
    _undecodable(tmp_path, b"\0" * 64, "bad magic")


# -- the check against allocated sizes ----------------------------------------------------------

def test_check_bounds_passes_tasks_inside_their_buffers(tmp_path):
    s = Stream().task("MM2S0", 0, 0, 0, 16).task("S2MM0", 1, 0, 64, 16)
    p = run(tmp_path, s.bytes(), "--check-bounds", "0=128")
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines() == ["op 2 col 0 MM2S0 BD 0 arg 0 0x0 0x40",
                                     "op 5 col 0 S2MM0 BD 1 arg 0 0x40 0x80"]


def test_check_bounds_names_every_task_that_leaves_its_buffer(tmp_path):
    s = Stream().task("MM2S0", 0, 0, 0, 16).task("S2MM0", 1, 0, 64, 16).sync().task("S2MM1", 2, 1, 0, 8)
    p = run(tmp_path, s.bytes(), "--check-bounds", "0=100", "1=32")
    assert p.returncode == 3
    assert p.stderr.splitlines() == [
        "outside: op 5: shim column 0 S2MM0 BD 1 reaches bytes [0x40, 0x80) of buffer 0, which holds 0x64 bytes"]
    p = run(tmp_path, s.bytes(), "--check-bounds", "0=128")
    assert p.returncode == 3
    assert "BD 2 reaches bytes [0x0, 0x20) of buffer 1, which is not allocated" in p.stderr


def test_a_run_with_an_undecodable_stream_is_refused_before_any_device_is_opened(tmp_path):
    # Runs under the emulator (XDNA_EMU=1) so that a runner which failed to refuse could reach only
    # the emulator plugin, never hardware; the refusal must come before either is opened, so the
    # emulator never starts and the xclbin need not exist.
    data = Stream().bd(0, 16).push("S2MM0", 0).bytes()
    f = tmp_path / "insts.bin"
    f.write_bytes(data)
    env = dict(os.environ, XDNA_EMU="1")
    p = subprocess.run([_runner(), "--xclbin", str(tmp_path / "none.xclbin"), "--instr", str(f),
                        "--trace-out", str(tmp_path / "t.bin")], capture_output=True, text=True, env=env,
                       timeout=60)
    assert p.returncode == 2, p.stderr
    assert "refusing to run" in p.stderr


# -- the constants are the register database's -------------------------------------------------

def _regdb():
    for root in (os.environ.get("MLIR_AIE_PATH"), os.environ.get("NPU_WORK_DIR") and
                 os.path.join(os.environ["NPU_WORK_DIR"], "mlir-aie"), str(REPO.parent / "mlir-aie")):
        if root:
            p = Path(root) / "lib" / "Dialect" / "AIE" / "Util" / "aie_registers_aie2.json"
            if p.exists():
                return json.loads(p.read_text())
    pytest.skip("AM025 register database (mlir-aie aie_registers_aie2.json) not found")


def test_the_decoder_constants_match_the_register_database():
    src = (REPO / "bridge-runner" / "insts_bounds.h").read_text()
    const = {m[0]: int(m[1], 0) for m in re.findall(r"constexpr uint32_t (\w+) = (0x[0-9A-Fa-f]+|\d+);", src)}
    fields = {m[0]: tuple(int(x) for x in m[1:]) for m in
              re.findall(r"constexpr Field (\w+)\{(\d+), (\d+), (\d+)\};", src)}
    shim = _regdb()["modules"]["shim"]
    regs = {r["name"]: r for r in (shim["registers"] if isinstance(shim, dict) else shim)}
    off = lambda name: int(regs[name]["offset"], 16)
    assert const["SHIM_BD_BASE"] == off("DMA_BD0_0")
    assert const["SHIM_BD_STRIDE"] == off("DMA_BD1_0") - off("DMA_BD0_0")
    assert const["SHIM_BD_WORDS"] * 4 == const["SHIM_BD_STRIDE"]
    assert f"DMA_BD{const['SHIM_NUM_BDS'] - 1}_7" in regs and f"DMA_BD{const['SHIM_NUM_BDS']}_0" not in regs
    for ch in ("S2MM_0", "S2MM_1", "MM2S_0", "MM2S_1"):
        assert const[f"SHIM_{ch}_TASK_QUEUE"] == off(f"DMA_{ch}_Task_Queue")
    assert const["BASE_ADDRESS_LOW_WORD"] == (off("DMA_BD0_1") - off("DMA_BD0_0")) // 4
    assert const["BASE_ADDRESS_HIGH_WORD"] == (off("DMA_BD0_2") - off("DMA_BD0_0")) // 4

    def bits(reg, name):
        f = next(f for f in regs[reg]["bit_fields"] if f["name"] == name)
        lo, hi = f["bit_range"]
        return lo, hi - lo + 1

    names = {"BUFFER_LENGTH": "Buffer_Length", "D0_WRAP": "D0_Wrap", "D0_STEPSIZE": "D0_Stepsize",
             "D1_WRAP": "D1_Wrap", "D1_STEPSIZE": "D1_Stepsize", "D2_STEPSIZE": "D2_Stepsize",
             "ITERATION_WRAP": "Iteration_Wrap", "ITERATION_STEPSIZE": "Iteration_Stepsize",
             "USE_NEXT_BD": "Use_Next_BD", "NEXT_BD": "Next_BD"}
    for const_name, reg_field in names.items():
        word, shift, width = fields[const_name]
        assert (shift, width) == bits(f"DMA_BD0_{word}", reg_field), const_name
    assert fields["START_BD_ID"][1:] == bits("DMA_S2MM_0_Task_Queue", "Start_BD_ID")


def test_a_real_compiled_stream_if_present(tmp_path):
    # mlir-aie's sp5_skew_r1 build: an 8192-byte output drain (the buffer an 8-byte pointer-sized
    # allocation once under-ran) and a 16384-byte trace buffer.
    path = REPO.parent / "mlir-aie" / "build" / "test" / "npu-xrt" / "sp5_skew_r1" / "chess" / "insts.bin"
    if not path.exists():
        pytest.skip("sp5_skew_r1 build artifact not present")
    assert sizes(tmp_path, path.read_bytes()) == {0: 8192, 1: 16384}
