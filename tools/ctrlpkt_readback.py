"""Host-driven tile register access over control packets, for probes.

A core cannot read its own tile's configuration registers, but the host can: a control
packet sent from shim MM2S 0 to a tile's TileControl port reads (opcode 1, the response
returns through a packet flow to shim S2MM 0 and lands in a host buffer) or writes
(opcode 0) any register of that tile. This is the read path the lock overflow/underflow
probes use (the same plumbing as the 2026-11-08 lock-range probe,
build/experiments/objectfifo-wedges/lockrange-probe, which ran on silicon and the EMU).

The packets are laid out in one host buffer (`ctrlin`), one packet per shim BD with a
`dma_wait` after each, and the read responses fill `ctrlout` in packet order. A `Wait` step
makes the host wait for a shim S2MM channel's token (the design's own data, for instance a
core's results) before sending the next packet, so a read can follow a core's work.

Packet header (AM025 / mlir-aie `aiex.control_packet`): bit 31 = odd parity over the word,
[27:24] response stream id (reads), [23:22] opcode (0 write, 1 read), [21:20] beats - 1,
[19:0] tile-local register address. A write is `[header, data...]`, a read `[header]`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

def _memory_module_registers() -> dict[str, int]:
    """Offsets of the compute tile's memory-module registers, from the AM025 database."""
    import json
    import toolchain_paths
    path = toolchain_paths.resolve_mlir_aie(Path(__file__)) / toolchain_paths.MLIR_AIE_SENTINEL
    regs = json.loads(path.read_text())["modules"]["memory"]["registers"]
    return {r["name"]: int(r["offset"], 16) for r in regs}


# One bit per lock (16 locks) in Locks_Overflow / Locks_Underflow, write 1 to clear;
# Lock<n>_value at a 0x10 stride; Event_Status3 holds memory-module events 96..127
# (LOCK_ERROR is 101: bit 5).
_REGS = _memory_module_registers()
LOCK_VALUE_BASE = _REGS["Lock0_value"]
LOCK_STRIDE = _REGS["Lock1_value"] - _REGS["Lock0_value"]
LOCKS_OVERFLOW = _REGS["Locks_Overflow"]
LOCKS_UNDERFLOW = _REGS["Locks_Underflow"]
MEM_EVENT_STATUS3 = _REGS["Event_Status3"]

CTRL_PKT_ID = 5                    # stream id of the packet flow into the tile's TileControl
RESPONSE_ID = 2                    # stream id of the packet flow back to shim S2MM 0
# A BD id per packet, cycled: the shim has 16 and the last ones belong to the data transfers.
BD_IDS = 4


def lock_value_addr(lock: int) -> int:
    return LOCK_VALUE_BASE + LOCK_STRIDE * lock


def parity_bit(word: int) -> int:
    """The bit 31 that makes the whole header word odd parity."""
    return 0 if bin(word & 0x7FFFFFFF).count("1") % 2 else 1


def header(addr: int, opcode: int, response_id: int = 0, beats: int = 1) -> int:
    if not 0 <= addr < 1 << 20:
        raise ValueError(f"register address {addr:#x} does not fit the 20-bit field")
    if opcode not in (0, 1) or not 1 <= beats <= 4:
        raise ValueError("opcode is 0 (write) or 1 (read); a packet carries 1..4 beats")
    word = (response_id << 24) | (opcode << 22) | ((beats - 1) << 20) | addr
    return word | (parity_bit(word) << 31)


@dataclass(frozen=True)
class Write:
    addr: int
    value: int
    label: str = ""


@dataclass(frozen=True)
class Read:
    addr: int
    label: str = ""


@dataclass(frozen=True)
class Wait:
    """Wait for the next completion token of a data transfer (`transfer` = its symbol)."""
    transfer: str


Step = Write | Read | Wait


@dataclass(frozen=True)
class Layout:
    words: tuple[int, ...]                    # the ctrlin buffer
    packets: tuple[tuple[int, int, int], ...]  # (step index, word offset, word count) per packet
    labels: tuple[str, ...]                   # one per read, in response order


def encode(steps: list[Step]) -> Layout:
    words: list[int] = []
    packets: list[tuple[int, int, int]] = []
    labels: list[str] = []
    for i, s in enumerate(steps):
        if isinstance(s, Write):
            pkt = [header(s.addr, 0), s.value & 0xFFFFFFFF]
        elif isinstance(s, Read):
            pkt = [header(s.addr, 1, RESPONSE_ID)]
            labels.append(s.label or f"{s.addr:#x}")
        else:
            continue
        packets.append((i, len(words), len(pkt)))
        words += pkt
    return Layout(tuple(words), tuple(packets), tuple(labels))


def sequence_lines(steps: list[Step], layout: Layout, ctrlout: str, ctrlin: str,
                   wait_symbols: dict[str, str] | None = None) -> list[str]:
    """The runtime-sequence ops: arm the response buffer, then each step in order.

    `ctrlout` / `ctrlin` are the sequence argument names. Shim allocations `ctrlin0` (MM2S 0)
    and `ctrl0` (S2MM 0) must be declared by the design."""
    n_out, n_in = len(layout.labels), len(layout.words)
    out = [f"      aiex.npu.dma_memcpy_nd(%{ctrlout}[0, 0, 0, 0][1, 1, 1, {n_out}][0, 0, 0, 1]) "
           f"{{id = 15 : i64, issue_token = true, metadata = @ctrl0}} : memref<{n_out}xi32>"]
    by_step = {i: (off, n) for i, off, n in layout.packets}
    sent = 0
    for i, s in enumerate(steps):
        if isinstance(s, Wait):
            out.append(f"      aiex.npu.dma_wait {{symbol = @{s.transfer}}}")
            continue
        off, n = by_step[i]
        out.append(f"      aiex.npu.dma_memcpy_nd(%{ctrlin}[0, 0, 0, {off}][1, 1, 1, {n}][0, 0, 0, 1], "
                   f"packet = <pkt_id = {CTRL_PKT_ID}, pkt_type = 0>) "
                   f"{{id = {sent % BD_IDS} : i64, issue_token = true, metadata = @ctrlin0}} "
                   f": memref<{n_in}xi32>")
        out.append("      aiex.npu.dma_wait {symbol = @ctrlin0}")
        sent += 1
    out.append("      aiex.npu.dma_wait {symbol = @ctrl0}")
    return out


def flow_lines(tile: str, shim: str) -> list[str]:
    """Packet flows and shim allocations: ctrlin into `tile`'s TileControl, responses back."""
    return [
        f"    aie.packet_flow({CTRL_PKT_ID}) {{",
        f"      aie.packet_source<{shim}, DMA : 0>",
        f"      aie.packet_dest<{tile}, TileControl : 0>",
        "    }",
        f"    aie.packet_flow({RESPONSE_ID}) {{",
        f"      aie.packet_source<{tile}, TileControl : 0>",
        f"      aie.packet_dest<{shim}, DMA : 0>",
        "    }",
        f"    aie.shim_dma_allocation @ctrlin0 ({shim}, MM2S, 0)",
        f"    aie.shim_dma_allocation @ctrl0 ({shim}, S2MM, 0)",
    ]


def pack_words(words) -> bytes:
    return b"".join(int(w).to_bytes(4, "little") for w in words)


# The host: runs the xclbin with (ctrlout, ctrlin, data) as the kernel's three buffers and
# writes ctrlout then data to a file. A response not delivered stays 0xEEEEEEEE.
HOST_CPP = r'''// ctrlpkt_readback host: usage host xclbin insts.bin pkts.bin n_ctrlout n_data out.bin
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

static std::vector<uint32_t> load(const char *p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { std::cerr << "cannot open " << p << "\n"; exit(2); }
  size_t n = f.tellg(); f.seekg(0);
  std::vector<uint32_t> v(n / 4);
  f.read(reinterpret_cast<char *>(v.data()), n);
  return v;
}

int main(int argc, char **argv) {
  if (argc < 7) { std::cerr << "usage: host xclbin insts.bin pkts.bin n_ctrlout n_data out.bin\n"; return 2; }
  int n_ctrl = std::stoi(argv[4]), n_data = std::stoi(argv[5]);
  auto instr = load(argv[2]);
  auto pkts = load(argv[3]);
  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(std::string(argv[1]));
  std::string name;
  for (auto &k : xclbin.get_kernels())
    if (k.get_name().rfind("MLIR_AIE", 0) == 0) name = k.get_name();
  device.register_xclbin(xclbin);
  xrt::hw_context ctx(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(ctx, name);
  auto bo_instr = xrt::bo(device, instr.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_ctrl = xrt::bo(device, n_ctrl * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_in = xrt::bo(device, pkts.size() * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_data = xrt::bo(device, n_data * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  memcpy(bo_instr.map<void *>(), instr.data(), instr.size() * 4);
  memcpy(bo_in.map<void *>(), pkts.data(), pkts.size() * 4);
  memset(bo_ctrl.map<void *>(), 0xEE, n_ctrl * 4);
  memset(bo_data.map<void *>(), 0xEE, n_data * 4);
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_ctrl.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto run = kernel(3u, bo_instr, (unsigned)instr.size(), bo_ctrl, bo_in, bo_data);
  auto r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) { std::cout << "STATUS " << r << "\n"; return 1; }
  bo_ctrl.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_data.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::ofstream o(argv[6], std::ios::binary);
  o.write(reinterpret_cast<const char *>(bo_ctrl.map<uint32_t *>()), n_ctrl * 4);
  o.write(reinterpret_cast<const char *>(bo_data.map<uint32_t *>()), n_data * 4);
  return 0;
}
'''


def build_host(dest_dir: Path) -> Path:
    """Compile HOST_CPP into `dest_dir` (rebuilt only when the source changes)."""
    import os
    import subprocess
    from harness.run import XRT, RunError, workspace
    mlir_aie = Path(os.environ.get("MLIR_AIE") or workspace() / "mlir-aie")
    test_lib = mlir_aie / "build" / "runtime_lib" / "x86_64" / "test_lib"
    cpp, exe = dest_dir / "ctrlpkt_host.cpp", dest_dir / "ctrlpkt_host"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not cpp.is_file() or cpp.read_text() != HOST_CPP:
        cpp.write_text(HOST_CPP)
    if not exe.exists() or exe.stat().st_mtime < cpp.stat().st_mtime:
        tmp = exe.with_name(f".ctrlpkt_host.{os.getpid()}")
        r = subprocess.run(["clang++", str(cpp), "-o", str(tmp), "-std=c++17", "-Wall",
                            "-I", f"{test_lib}/include", "-I", f"{XRT}/include",
                            "-L", f"{XRT}/lib", "-lxrt_coreutil", "-lrt", "-lstdc++"],
                           capture_output=True, text=True)
        if r.returncode:
            tmp.unlink(missing_ok=True)
            raise RunError(f"building the ctrlpkt host failed:\n{r.stderr[-1500:]}")
        tmp.replace(exe)
    return exe
