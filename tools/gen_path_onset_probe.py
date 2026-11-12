#!/usr/bin/env python3
"""Path onset probe: where does a memtile-sourced path hold its extra 4 words?

The pipeline-fill race (`gen_pipeline_fill_probe.py race`) measured every memtile-sourced path
4 words larger on silicon than in the emulator (memtile -> own S2MM 52 vs 48, memtile -> (0,2)
60 vs 56), while the compute-sourced paths match (48, 56). A race counts words read before an
overwrite, so it cannot say whether the 4 words are in the memtile MM2S's egress staging (the
emulator's 25) or downstream of it, in the stream path.

The trace can. One MM2S BD of N_WORDS words runs into an S2MM whose BD waits on a gate lock. The
staging fills faster than the port drains, so the MM2S port runs at one word a cycle until the
path downstream of the staging is full; STREAM_BACKPRESSURE then rises. Its onset after the
MM2S START_TASK measures the downstream capacity alone: 4 more words downstream move it 4
cycles later, a bigger staging leaves it where it is. The gate opens after the source settled,
the S2MM takes all N_WORDS, and the run ends.

Cases (src -> dst; mem = memtile (0,1), prod = compute (0,2)): mem_own, prod_own, mem_prod,
prod_mem. Each traced tile's DMA events go through the injector (memtile trace unit, or the
compute tile's memory-module trace unit), event-time mode.

A case name may carry a length, `prod_own_n23`: the BD then moves 23 words, so a sweep over N reads
where the MM2S port boundary sits (sustained backpressure until the gate opens from the first N
that does not fit downstream of the staging).

    python3 tools/gen_path_onset_probe.py build OUT_DIR [--cases NAME ...]
    python3 tools/gen_path_onset_probe.py run OUT_DIR TAG --backend hw|emu [--only NAME ...]
    python3 tools/gen_path_onset_probe.py show OUT_DIR TAG [TAG ...] [--only NAME ...]
    python3 tools/gen_path_onset_probe.py ports OUT_DIR TAG [TAG ...] [--only NAME ...]

Port-watching cases (`mem_own_ports`, `prod_own_ports`, `core_own_ports`, optionally `_nN`) trace
PORT_RUNNING and PORT_STALLED of the stream-switch ports instead of the DMA's wait states, to
count the words in each stage of the path while the S2MM is held: the cycles PORT_RUNNING was
high before the sustained PORT_STALLED are the words that passed that port. `core_own_ports`
sources the words from the core (put_stream into Core:0), released by a lock after the trace is
armed. `ports` prints the per-port counts. TAG is the run directory name, e.g. `e1.emu`.

Clock-gate configurations (`build --clock K1`): the runtime sequence first writes the tile clock
controls of the case's data tiles (compute tile, memtile) with a masked write per register, so the
same case runs with the stream-switch and DMA adaptive clock gates changed. K0 writes nothing;
K1 clears Stream_Switch_Adaptive_Clock_Gate, K2 DMA_Adaptive_Clock_Gate, K3 both, K4 sets the
stream-switch Abort_Period to its minimum (the gate engages after the shortest idle time). Register
offsets, fields and the supported period range are read from the AM025 database (mlir-aie), not
written here. Use one OUT_DIR per configuration.

Proof that a clock write lands (`proof-build`, `proof-run`, `proof-show`): the trace routing and
a control-packet readback cannot share one design (the trace injector owns the shim S2MM and the
runtime-sequence arguments the bridge runner binds; the readback needs shim MM2S 0 / S2MM 0 for its
packets and its own host with three buffers), so the proof is a separate, trace-free design on the
same tile: read lock 0 (seeded to a known value, the control read), Module_Clock_Control, the
stream-switch Abort_Period and the gate Status register over control packets, do the config's
masked writes, read the three again, read lock 0 again, then the control-packet handler status.
The sequence does not wait on the replies: the host collects them after a fixed pause, and a read
that was never answered shows as "no reply". `proof-show` compares the clock reads with what the
AM025 database says a landed write must give: LANDED, DROPPED, NO-REPLY (clock reads unanswered,
control reads answered), BROKEN (control reads unanswered), and prints the handler status bits.

Gate pre-wake probe (`prewake-build`, `prewake-run`, `prewake-table`). A tile's stream switch is
adaptively clock-gated once it has been idle for 2^Abort_Period = 128 cycles, and a word offered to
a gated switch is refused for one cycle: a one-cycle DMA MM2S STREAM_BACKPRESSURE "blip" on the
first word of a transfer, which then arrives one cycle later. If that is the mechanism, a dummy
word that crossed the same switch D cycles before the measured transfer's first word hides the
blip exactly when D is below the idle threshold. One compute tile, two loopbacks through its
switch:
  MM2S 1 -> S2MM 1: a repeating one-word BD behind lock Dm (the dummy word, and later the filler).
  MM2S 0 -> S2MM 0: N_WORDS = 16 words behind lock M (the measured transfer).
  core: wait for the go lock (set by the runtime sequence once the trace is armed), settle (the
    switch has seen no traffic since boot), release Dm, D nops, release M; then, long after the
    measured transfer, PW_FILLERS more releases of Dm. Case `pw_dNNN` puts NNN nops between the
    two releases (the table reports the D the trace measured: MM2S 1 grant to MM2S 0 grant, and
    MM2S 1 FINISHED_BD to MM2S 0 grant); `pw_ctl` skips the first release.
Trace: the memory module only (no core-module trace unit, so no core packet ever leaves the tile),
five events: MM2S 0 STALLED_LOCK (its fall is the start: the BDs are started at boot) and
STREAM_BACKPRESSURE, MM2S 1 STALLED_LOCK and FINISHED_BD, S2MM 0
STREAM_STARVATION (it falls when the measured first word arrives).
The confound: a trace packet leaving this tile crosses its switch and wakes it. A packet leaves
when it is full, and the unit's last, partly filled packet never leaves without a stop event, so
the design inverts the usual problem: the window's few frames (about ten edges, well under one
packet) sit in the first packet, which is completed by the filler's MM2S 1 events AFTER the
measured transfer, and the filler also leaves the last packet full. The table counts the trace
edges up to the measured first word (`frames`) and marks `PACKET?` when that reaches PW_FRAME_GUARD
(a packet could have left inside the window); a case so marked must not be read.
Why it always completes. Nothing waits on a host buffer, host DMA or host signal. The runtime sequence
holds only plain writes: one lock set (the go lock) and padding writes to a scratch word; it waits
on nothing. The core waits on one thing, the go lock, which that sequence always sets.
Both S2MMs run from boot on a plain BD and take any word offered; the MM2S BDs wait on locks
that only the core releases, and it always reaches those releases after the go lock.

    python3 tools/gen_path_onset_probe.py prewake-build OUT_DIR [--ds 20 50 ...] [--no-control]
    python3 tools/gen_path_onset_probe.py prewake-run OUT_DIR TAG --backend hw|emu [--ds ...]
    python3 tools/gen_path_onset_probe.py prewake-table OUT_DIR TAG [TAG ...] [--ds ...]

`run --backend hw` takes the NPU lock (`build/experiments/.npu-hw.lock`).
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from functools import lru_cache
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_lock_pulse_probe as lp  # noqa: E402  (build/run plumbing: sh, run_case)
import toolchain_paths  # noqa: E402

N_WORDS = 96
SRC_ADDR, DST_ADDR, SCRATCH_ADDR = 0x1000, 0x2000, 0x3000
SETTLE_WRITES = 300       # command-processor writes between the source start and the gate opening
TILES = {"mem": (0, 1), "prod": (0, 2)}
VAR = {"mem": "%t_mem", "prod": "%t_prod"}
# "core" as a source is the compute tile's core pushing words with put_stream into the stream
# switch's Core:0 port; it lives on the `prod` tile.
TILE_OF = {"mem": "mem", "prod": "prod", "core": "prod"}
CASES = {"mem_own": ("mem", "mem"), "prod_own": ("prod", "prod"),
         "mem_prod": ("mem", "prod"), "prod_mem": ("prod", "mem"),
         # port-watching variants: PORT_RUNNING/PORT_STALLED on the stream-switch ports
         "mem_own_ports": ("mem", "mem"), "prod_own_ports": ("prod", "prod"),
         "core_own_ports": ("core", "prod")}
PLAIN_CASES = [c for c in CASES if not c.endswith("_ports")]
# Gate and done locks of the S2MM (free in both lock spaces: memtile 0..63, compute 0..15).
GATE_LOCK, DONE_LOCK = 4, 5
# The core source waits on this lock until the runtime sequence has armed the trace: a core
# starts at array configuration, before the trace unit, and would push its words unobserved.
GO_LOCK = 6
# A PORT_STALLED run this long (or still high at the end) is the held-consumer stall, not a blip.
SUSTAINED_STALL_CYCLES = 16
PORT_EVENTS = "PORT_RUNNING_{a},PORT_STALLED_{a},PORT_RUNNING_{b},PORT_STALLED_{b}"
# memtile slots: 0 = DMA:0 master (S2MM side), 4 = DMA:0 slave (MM2S side)
MEMTILE_PORT_EVENTS = ("DMA_MM2S_SEL0_START_TASK,DMA_MM2S_SEL0_STREAM_BACKPRESSURE,"
                       "DMA_S2MM_SEL0_STALLED_LOCK," + PORT_EVENTS.format(a=0, b=4))
# compute-tile core module slots: 0 = DMA:0 master (S2MM), 1 = DMA:0 slave (MM2S), or, for a
# core source, Core:0 slave (see --port-slots)
CORE_PORT_EVENTS = PORT_EVENTS.format(a=0, b=1)

MEMTILE_EVENTS = ("DMA_MM2S_SEL0_START_TASK,DMA_MM2S_SEL0_STREAM_BACKPRESSURE,DMA_MM2S_SEL0_FINISHED_BD,"
                  "DMA_MM2S_SEL0_MEMORY_STARVATION,DMA_S2MM_SEL0_START_TASK,DMA_S2MM_SEL0_STALLED_LOCK,"
                  "DMA_S2MM_SEL0_FINISHED_BD,DMA_S2MM_SEL0_STREAM_STARVATION")
MEMMOD_EVENTS = ("DMA_MM2S_0_START_TASK,DMA_MM2S_0_STREAM_BACKPRESSURE,DMA_MM2S_0_FINISHED_BD,"
                 "DMA_MM2S_0_MEMORY_STARVATION,DMA_S2MM_0_START_TASK,DMA_S2MM_0_STALLED_LOCK,"
                 "DMA_S2MM_0_FINISHED_BD,DMA_S2MM_0_STREAM_STARVATION")


# Clock-gate configurations: which adaptive gates a case's data tiles run with (see the module doc).
CLOCK_CONFIGS = {"K0": (), "K1": ("ss_off",), "K2": ("dma_off",), "K3": ("ss_off", "dma_off"),
                 "K4": ("ss_period_min",)}
# regdb module holding each tile's clock-control registers
REGDB_MODULE = {"mem": "memory_tile", "prod": "core"}
ABORT_PERIOD_REG = "Stream_Switch_Adaptive_Clock_Gate_Abort_Period"


@lru_cache(maxsize=1)
def _regdb_modules() -> dict:
    path = toolchain_paths.resolve_mlir_aie(start=REPO) / toolchain_paths.MLIR_AIE_SENTINEL
    return json.loads(path.read_text())["modules"]


def _field(module: str, register: str, field: str) -> tuple[int, int, int]:
    """(register offset, field mask, supported-range-min) from the AM025 database."""
    reg = next(r for r in _regdb_modules()[module]["registers"] if r["name"] == register)
    f = next(f for f in reg["bit_fields"] if f["name"] == field)
    lo, hi = f["bit_range"]
    mask = ((1 << (hi - lo + 1)) - 1) << lo
    rng = re.search(r"range for field: \[(\d+)-(\d+)\]", f["description"])
    return int(reg["offset"], 16), mask, int(rng.group(1)) if rng else 0


def clock_writes(config: str, tile: str) -> list[tuple[int, int, int]]:
    """(offset, value, mask) masked writes that put `tile` ('mem' or 'prod') in clock config `config`."""
    module = REGDB_MODULE[tile]
    out = []
    clears = [name for key, name in (("ss_off", "Stream_Switch_Adaptive_Clock_Gate"),
                                     ("dma_off", "DMA_Adaptive_Clock_Gate"))
              if key in CLOCK_CONFIGS[config]]
    if clears:
        offset, mask = None, 0
        for name in clears:
            offset, m, _ = _field(module, "Module_Clock_Control", name)
            mask |= m
        out.append((offset, 0, mask))
    if "ss_period_min" in CLOCK_CONFIGS[config]:
        offset, mask, lowest = _field(module, ABORT_PERIOD_REG, "Abort_Period")
        out.append((offset, lowest, mask))
    return out


def clock_sequence(config: str, tiles: list[str]) -> str:
    """Runtime-sequence text of the clock writes for the case's data tiles."""
    lines = []
    for t in tiles:
        for i, (offset, value, mask) in enumerate(clock_writes(config, t)):
            n = f"{t}{i}"
            lines.append(f"      %clk_a_{n} = arith.constant {offset} : i32\n"
                         f"      %clk_v_{n} = arith.constant {value} : i32\n"
                         f"      %clk_m_{n} = arith.constant {mask} : i32\n"
                         f"      aiex.npu.maskwrite32(%clk_a_{n}, %clk_v_{n}, %clk_m_{n}) "
                         f"{{column = {TILES[t][0]} : i32, row = {TILES[t][1]} : i32}} : i32, i32, i32")
    return "".join(line + "\n" for line in lines)


# -- clock-write proof: the same masked writes, read back over control packets --------------------

PROOF_TARGET = {"prod": "core", "mem": "mem"}      # ctrlpkt_readback target of each data tile
PROOF_VAR = {"prod": "%t_core", "mem": "%t_mem"}
PROOF_PKT_ID = {"core": 5, "mem": 6}
PROOF_READS = ("Module_Clock_Control", ABORT_PERIOD_REG, "Stream_Switch_Adaptive_Clock_Gate_Status")
# AM025 module holding each tile's lock registers (the control read's register) and its handler status
PROOF_LOCK_MODULE = {"prod": "memory", "mem": "memory_tile"}
HANDLER_STATUS_REG = "Tile_Control_Packet_Handler_Status"
SLVERR_FIELD = "SLVERR_On_Access"
PROOF_LOCK_VALUE = 42               # lock 0's initial value: a reply that cannot be mistaken for a clock word
PROOF_DATA_WORDS = 4                # the sequence's unused third buffer
PROOF_SETTLE_MS = 500               # the host waits this long after the run before collecting replies
NO_REPLY = 0xEEEEEEEE               # what the host leaves in a reply slot that was never written


def _register(module: str, name: str) -> dict:
    return next(r for r in _regdb_modules()[module]["registers"] if r["name"] == name)


def _offset(module: str, name: str) -> int:
    return int(_register(module, name)["offset"], 16)


def clock_proof_steps(config: str, tile: str) -> list:
    """ctrlpkt_readback steps: a control read (lock 0), the three clock reads, the config's masked
    writes, the three clock reads again, the control read again, then the handler status."""
    import ctrlpkt_readback as cp
    module, target = REGDB_MODULE[tile], PROOF_TARGET[tile]
    control = _offset(PROOF_LOCK_MODULE[tile], "Lock0_value")
    reads = [(name, _offset(module, name)) for name in PROOF_READS]
    steps: list = [cp.Read(control, "pre control Lock0_value", target)]
    steps += [cp.Read(off, f"pre {name}", target) for name, off in reads]
    lines = []
    for i, (offset, value, mask) in enumerate(clock_writes(config, tile)):
        lines += [f"%pf_a{i} = arith.constant {offset} : i32", f"%pf_v{i} = arith.constant {value} : i32",
                  f"%pf_m{i} = arith.constant {mask} : i32",
                  f"aiex.npu.maskwrite32(%pf_a{i}, %pf_v{i}, %pf_m{i}) "
                  f"{{column = {TILES[tile][0]} : i32, row = {TILES[tile][1]} : i32}} : i32, i32, i32"]
    if lines:
        steps.append(cp.Raw(tuple(lines)))
    steps += [cp.Read(off, f"post {name}", target) for name, off in reads]
    steps += [cp.Read(control, "post control Lock0_value", target),
              cp.Read(_offset(module, HANDLER_STATUS_REG), f"handler {HANDLER_STATUS_REG}", target)]
    return steps


def proof_sequence_lines(steps: list, layout, ctrlout: str, ctrlin: str) -> list[str]:
    """ctrlpkt_readback.sequence_lines without the waits on the responses: the reply buffer is armed
    without a completion token and the sequence ends after the last packet has left the shim; the
    host collects the replies after PROOF_SETTLE_MS. A read that is never answered leaves its slot
    at NO_REPLY."""
    import ctrlpkt_readback as cp
    n_out, n_in = len(layout.labels), len(layout.words)
    out = [f"      aiex.npu.dma_memcpy_nd(%{ctrlout}[0, 0, 0, 0][1, 1, 1, {n_out}][0, 0, 0, 1]) "
           f"{{id = 15 : i64, issue_token = false, metadata = @ctrl0}} : memref<{n_out}xi32>"]
    by_step = {i: (off, n) for i, off, n in layout.packets}
    sent = 0
    for i, s in enumerate(steps):
        if isinstance(s, cp.Raw):
            out += [f"      {line}" for line in s.lines]
            continue
        off, n = by_step[i]
        out.append(f"      aiex.npu.dma_memcpy_nd(%{ctrlin}[0, 0, 0, {off}][1, 1, 1, {n}][0, 0, 0, 1], "
                   f"packet = <pkt_id = {cp.TARGETS[s.tile].pkt_id}, pkt_type = 0>) "
                   f"{{id = {sent % cp.BD_IDS} : i64, issue_token = true, metadata = @ctrlin0}} "
                   f": memref<{n_in}xi32>")
        out.append("      aiex.npu.dma_wait {symbol = @ctrlin0}")
        sent += 1
    return out


def clock_proof_design(config: str, tile: str) -> tuple[str, list, object]:
    import ctrlpkt_readback as cp
    steps = clock_proof_steps(config, tile)
    layout = cp.encode(steps)
    target, var = PROOF_TARGET[tile], PROOF_VAR[tile]
    col, row = TILES[tile]
    lines = ["module {", "  aie.device(npu1_1col) {",
             "    %t00 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}",
             f"    {var} = aie.tile({col}, {row}) "
             f"{{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = {PROOF_PKT_ID[target]}>}}",
             f'    %l_ctl = aie.lock({var}, 0) {{init = {PROOF_LOCK_VALUE} : i32, sym_name = "l_ctl"}}',
             *cp.packet_flows(var, "%t00", target), *cp.shim_allocations("%t00"),
             f"    aie.runtime_sequence @seq(%ctrlout: memref<{len(layout.labels)}xi32>, "
             f"%ctrlin: memref<{len(layout.words)}xi32>, %data: memref<{PROOF_DATA_WORDS}xi32>) {{",
             *proof_sequence_lines(steps, layout, "ctrlout", "ctrlin"), "    }", "  }", "}", ""]
    return "\n".join(lines), steps, layout


def proof_expectation(config: str, tile: str) -> dict[str, int]:
    """What a landed write leaves in each clock register: the reset value with the written fields
    applied (pre reads are the reset values). The gate Status register is only reported."""
    module = REGDB_MODULE[tile]
    reset = {n: int(_register(module, n)["reset"], 16) for n in PROOF_READS[:2]}
    post = dict(reset)
    for offset, value, mask in clock_writes(config, tile):
        for n in PROOF_READS[:2]:
            if _offset(module, n) == offset:
                post[n] = (reset[n] & ~mask) | (value & mask)
    return {**{f"pre {n}": v for n, v in reset.items()}, **{f"post {n}": v for n, v in post.items()}}


def proof_decode(labels: list[str], words: list[int]) -> dict[str, int | None]:
    """Reply words (NO_REPLY where nothing arrived) -> {label: value or None}.

    Replies fill the buffer in arrival order, so a read that is never answered lets the next reply
    slide into its slot. The control reads return PROOF_LOCK_VALUE, which no clock word equals: a
    clock or status read that finds that value next while a control read is still to come was not
    answered."""
    out: dict[str, int | None] = {}
    p = 0
    for i, label in enumerate(labels):
        control = "control" in label
        control_ahead = any("control" in l for l in labels[i + 1:])
        w = words[p] if p < len(words) else NO_REPLY
        if w == NO_REPLY or (not control and w == PROOF_LOCK_VALUE and control_ahead):
            out[label] = None
            continue
        out[label] = w
        p += 1
    return out


def proof_verdict(config: str, tile: str, readback: dict[str, int | None]) -> str:
    """BROKEN: a control read got no (or the wrong) reply, so control packets are not working.
    NO-REPLY: the clock registers did not answer while control did. LANDED / DROPPED: the post
    read has / has not changed as written. NO-WRITE: K0. UNEXPECTED: any other values."""
    ctl = [readback.get(f"{w} control Lock0_value") for w in ("pre", "post")]
    if any(v != PROOF_LOCK_VALUE for v in ctl):
        return "BROKEN"
    want = proof_expectation(config, tile)
    keys = [f"{w} {n}" for w in ("pre", "post") for n in PROOF_READS[:2]]
    if any(readback.get(k) is None for k in keys):
        return "NO-REPLY"
    changed = [n for n in PROOF_READS[:2] if want[f"post {n}"] != want[f"pre {n}"]]
    if all(readback[k] == want[k] for k in keys):
        return "LANDED" if changed else "NO-WRITE"
    if changed and all(readback[f"post {n}"] == readback[f"pre {n}"] for n in PROOF_READS[:2]):
        return "DROPPED"
    return "UNEXPECTED"


CASE_NAME = re.compile(r"(.+?)(?:_n(\d+)|_([ct])(\d+)x(\d+))?")


@dataclass(frozen=True)
class Shape:
    """A case's route and BD layout: `count` BDs of `length` words, chained in one task
    (`chain`, suffix `_cLxK`) or as back-to-back tasks (`tasks`, `_tLxK`); `single` is one BD
    (`BASE` or `BASE_nN`). The S2MM always takes the whole stream in one BD."""
    src: str
    dst: str
    length: int
    count: int
    mode: str

    @property
    def total(self) -> int:
        return self.length * self.count

    def design_args(self) -> tuple:
        return (self.src, self.dst, self.total, self.length, self.mode)


def case_shape(name: str) -> Shape:
    m = CASE_NAME.fullmatch(name)
    src, dst = CASES[m.group(1)]
    if m.group(3):
        mode = "chain" if m.group(3) == "c" else "tasks"
        return Shape(src, dst, int(m.group(4)), int(m.group(5)), mode)
    return Shape(src, dst, int(m.group(2)) if m.group(2) else N_WORDS, 1, "single")


def case_spec(name: str) -> tuple[str, str, int]:
    """(source, destination, total words) of a case name (see `case_shape`)."""
    s = case_shape(name)
    return s.src, s.dst, s.total


def is_ports_case(name: str) -> bool:
    return CASE_NAME.fullmatch(name).group(1).endswith("_ports")


def design_mlir(src: str, dst: str, n: int = N_WORDS, bd_len: int | None = None,
               mode: str = "single", clock: str = "K0") -> str:
    core_src = src == "core"
    srct = TILE_OF[src]
    tiles = list(dict.fromkeys((srct, dst)))
    decls = "\n".join(f"    {VAR[t]} = aie.tile({TILES[t][0]}, {TILES[t][1]})" for t in tiles)
    words = [0x0A000000 | j for j in range(n)]
    pad = "\n".join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32"
                    for _ in range(SETTLE_WRITES))
    sv, dv = VAR[srct], VAR[dst]
    clock_ops = clock_sequence(clock, list(dict.fromkeys((srct, dst))))
    if core_src:
        src_decl = f"    %lGo = aie.lock({sv}, {GO_LOCK}) {{init = 0 : i32, sym_name = \"lGo\"}}"
        puts = "\n".join(f"      %w{j} = arith.constant {w} : i32\n      aie.put_stream(%ch : i32, %w{j} : i32)"
                         for j, w in enumerate(words))
        src_decl += f"""
    %core_src = aie.core({sv}) {{
      %c_go = arith.constant 1 : i32
      aie.use_lock(%lGo, AcquireGreaterEqual, %c_go)
      %ch = arith.constant 0 : i32
{puts}
      aie.end
    }}"""
        flow = f"aie.flow({sv}, Core : 0, {dv}, DMA : 0)"
        start_src = "      aiex.set_lock(%lGo, %c_one)"
        await_src = ""
    else:
        data = ", ".join(map(str, words))
        src_decl = (f"    %src = aie.buffer({sv}) {{address = {SRC_ADDR} : i32, sym_name = \"src\", "
                    f"initial_value = dense<[{data}]> : tensor<{n}xi32>}} : memref<{n}xi32>")
        flow = f"aie.flow({sv}, DMA : 0, {dv}, DMA : 0)"
        start_src, await_src = mm2s_tasks(sv, n, bd_len or n, mode)
    return f"""module {{
  aie.device(npu1_1col) {{
    %t_shim = aie.tile(0, 0)
{decls}
{src_decl}
    %dst = aie.buffer({dv}) {{address = {DST_ADDR} : i32, sym_name = "dst"}} : memref<{n}xi32>
    %scratch = aie.buffer({sv}) {{address = {SCRATCH_ADDR} : i32, sym_name = "scratch"}} : memref<4xi32>
    %lG = aie.lock({dv}, {GATE_LOCK}) {{init = 0 : i32, sym_name = "lG"}}
    %lD = aie.lock({dv}, {DONE_LOCK}) {{init = 0 : i32, sym_name = "lD"}}
    {flow}
    aie.runtime_sequence(%dummy: memref<4xi32>) {{
      %c_zero = arith.constant 0 : i32
      %c_one = arith.constant 1 : i32
{clock_ops}      %tr = aiex.dma_configure_task({dv}, S2MM, 0) {{
        %c_a = arith.constant 1 : i32
        aie.use_lock(%lG, AcquireGreaterEqual, %c_a)
        aie.dma_bd(%dst : memref<{n}xi32> offset = 0 len = {n}) {{bd_id = 0 : i32}}
        %c_r = arith.constant 1 : i32
        aie.use_lock(%lD, Release, %c_r)
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tr)
{start_src}
{pad}
      aiex.set_lock(%lG, %c_one)
{await_src}      aiex.dma_await_task(%tr)
    }}
  }}
}}
"""


def mm2s_tasks(sv: str, n: int, bd_len: int, mode: str) -> tuple[str, str]:
    """Runtime-sequence text starting the MM2S side and awaiting it: one task of one BD, one task
    of chained BDs, or back-to-back single-BD tasks, each BD `bd_len` words of `src` in order."""
    k = n // bd_len

    def bd(j: int) -> str:
        return (f"        aie.dma_bd(%src : memref<{n}xi32> offset = {j * bd_len} len = {bd_len}) "
                f"{{bd_id = {1 + j} : i32}}")

    if mode == "tasks":
        start = "\n".join(f"""      %ts{j} = aiex.dma_configure_task({sv}, MM2S, 0) {{
{bd(j)}
        aie.end
      }} {{issue_token = true}}""" for j in range(k))
        start += "\n" + "\n".join(f"      aiex.dma_start_task(%ts{j})" for j in range(k))
        return start, "".join(f"      aiex.dma_await_task(%ts{j})\n" for j in range(k))
    body = []
    for j in range(k):
        if j:
            body.append(f"      ^bd{j}:")
        body.append(bd(j))
        body.append(f"        aie.next_bd ^bd{j + 1}" if j + 1 < k else "        aie.end")
    start = (f"      %ts = aiex.dma_configure_task({sv}, MM2S, 0) {{\n" + "\n".join(body)
             + "\n      } {issue_token = true}\n      aiex.dma_start_task(%ts)")
    return start, "      aiex.dma_await_task(%ts)\n"


def inject_args(name: str) -> list[str]:
    """The trace injector's event arguments for a case."""
    src, dst, _ = case_spec(name)
    traced = {TILE_OF[src], dst}
    ports = is_ports_case(name)
    a: list[str] = []
    if "mem" in traced:
        a += ["--memtile-sweep-events", MEMTILE_PORT_EVENTS if ports else MEMTILE_EVENTS]
    if "prod" in traced and src != "core":
        a += ["--memmod-sweep-events", MEMMOD_EVENTS]
    if ports and "prod" in traced:
        a += ["--core-grounding", "", "--core-sweep-events", CORE_PORT_EVENTS]
        if src == "core":
            a += ["--port-slots", "1=Core:0:slave"]
    return a


def build_case(name: str, out: Path, clock: str = "K0") -> None:
    shape = case_shape(name)
    src, dst, n = shape.src, shape.dst, shape.total
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "base.mlir").write_text(design_mlir(*shape.design_args(), clock=clock))
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time",
           *inject_args(name),
           "--trace-config-out", "trace_config.json", "--config-test-name", name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    (d / "case.json").write_text(json.dumps({"case": name, "src": src, "dst": dst, "n_words": n, "clock": clock}) + "\n")


def tile_frames(path: Path, row: int, pkt_type: int | None = None) -> list[tuple[int, frozenset[str]]]:
    """The trace unit's frames for one tile row: (cycle, events active). An event-time frame is
    emitted when the active set changes, so an event is high from a frame naming it to the next
    frame that does not."""
    by_ts: dict[int, set[str]] = {}
    for e in json.loads(path.read_text())["events"]:
        if e["row"] == row and (pkt_type is None or e["pkt_type"] == pkt_type):
            by_ts.setdefault(e["ts"], set()).add(e["name"])
    return [(t, frozenset(n)) for t, n in sorted(by_ts.items())]


def intervals(frames, name: str) -> list[tuple[int, int]]:
    """[start, end) cycles where `name` is high (end None if still high at the last frame)."""
    out, start = [], None
    for t, active in frames:
        if name in active and start is None:
            start = t
        elif name not in active and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, None))
    return out


def port_report(frames, running: str, stalled: str) -> dict:
    """Words a stream-switch port passed before the consumer's hold stalled it.

    PORT_RUNNING is high in each cycle a word transfers, PORT_STALLED while the port holds data
    it cannot pass. The sustained stall is the first PORT_STALLED run of SUSTAINED_STALL_CYCLES
    cycles or more (or one still high at the last frame); the RUNNING cycles that began before it
    are the words that went through. Without a sustained stall every RUNNING cycle counts."""
    run, stall = intervals(frames, running), intervals(frames, stalled)
    onset = next((s for s, e in stall if e is None or e - s >= SUSTAINED_STALL_CYCLES), None)
    counted = [(s, e) for s, e in run if onset is None or s < onset]
    return {"running": run, "stalled": stall, "stall_onset": onset,
            "running_before_stall": sum((e if e is not None else s + 1) - s for s, e in counted)}


# Trace units of a case's tiles: (label, row, pkt_type, port slots as (running, stalled, name)).
# pkt_type: 0 compute core module, 1 compute memory module, 3 memtile.
def port_units(src: str, dst: str):
    if src == "mem":
        return [("memtile", 1, 3, [("PORT_RUNNING_0", "PORT_STALLED_0", "DMA:0 master (S2MM side)"),
                                   ("PORT_RUNNING_4", "PORT_STALLED_4", "DMA:0 slave (MM2S side)")])]
    second = "Core:0 slave (core put_stream)" if src == "core" else "DMA:0 slave (MM2S side)"
    units = [("compute core module", 2, 0, [("PORT_RUNNING_0", "PORT_STALLED_0", "DMA:0 master (S2MM side)"),
                                            ("PORT_RUNNING_1", "PORT_STALLED_1", second)])]
    if src != "core":
        units.insert(0, ("compute memory module", 2, 1, []))
    return units


def ports_summary(path: Path, src: str, dst: str) -> str:
    lines = []
    for label, row, pkt, slots in port_units(src, dst):
        frames = tile_frames(path, row, pkt)
        if not frames:
            lines.append(f"  {label}: no events")
            continue
        names = sorted(set().union(*(a for _, a in frames)))
        starts = [s for n in names if n.endswith(("MM2S_0_START_TASK", "MM2S_SEL0_START_TASK"))
                  for s, _ in intervals(frames, n)]
        t0 = min(starts) if starts else frames[0][0]
        lines.append(f"  {label} (cycles from {'the MM2S START_TASK' if starts else 'its first frame'}"
                     f" at absolute {t0}):")
        for n in names:
            if "BACKPRESSURE" in n:
                iv = [(s - t0, None if e is None else e - t0) for s, e in intervals(frames, n)]
                lines.append(f"    {n}: first rise at {iv[0][0] if iv else None}, intervals {iv[:6]}")
        for running, stalled, what in slots:
            r = port_report(frames, running, stalled)
            shift = lambda iv: [(s - t0, None if e is None else e - t0) for s, e in iv]
            onset = None if r["stall_onset"] is None else r["stall_onset"] - t0
            lines.append(f"    {what}: RUNNING cycles before sustained stall = {r['running_before_stall']}, "
                         f"sustained STALLED onset = {onset}")
            lines.append(f"      {running} {shift(r['running'])}")
            lines.append(f"      {stalled} {shift(r['stalled'])}")
    return "\n".join(lines)


def summarize(path: Path, src: str, dst: str) -> str:
    lines = []
    for t in dict.fromkeys((src, dst)):
        # trace rows are physical; the application column 0 is physical column 1 on NPU1
        frames = tile_frames(path, TILES[t][1])
        if not frames:
            lines.append(f"  {t}: no events")
            continue
        names = sorted(set().union(*(a for _, a in frames)))
        starts = [s for n in names if n.endswith(("MM2S_0_START_TASK", "MM2S_SEL0_START_TASK"))
                  for s, _ in intervals(frames, n)]
        t0 = min(starts) if starts else frames[0][0]
        parts = []
        for n in names:
            iv = [(s - t0, None if e is None else e - t0) for s, e in intervals(frames, n)]
            parts.append(f"{n.split('_', 1)[1]:<26} {iv[:6]}{' ...' if len(iv) > 6 else ''}")
        lines.append(f"  {t} (cycles from the {'MM2S START_TASK' if starts else 'first frame'}):\n    "
                     + "\n    ".join(parts))
    return "\n".join(lines)


def build_proof_host(dest: Path) -> Path:
    """The ctrlpkt_readback host with a PROOF_SETTLE_MS pause between the end of the run and the
    collection of the replies (replies are not awaited by the sequence)."""
    import ctrlpkt_readback as cp
    src = cp.HOST_CPP
    src = src.replace("#include <vector>", "#include <vector>\n#include <chrono>\n#include <thread>")
    src = src.replace("  bo_ctrl.sync(XCL_BO_SYNC_BO_FROM_DEVICE);",
                      f"  std::this_thread::sleep_for(std::chrono::milliseconds({PROOF_SETTLE_MS}));\n"
                      "  bo_ctrl.sync(XCL_BO_SYNC_BO_FROM_DEVICE);", 1)
    assert "sleep_for" in src
    exe = dest / "proof_host"
    cpp = dest / "proof_host.cpp"
    dest.mkdir(parents=True, exist_ok=True)
    if not cpp.is_file() or cpp.read_text() != src:
        cpp.write_text(src)
        exe.unlink(missing_ok=True)
    if not exe.exists():
        # reuse the ctrlpkt host build (same flags) on this source
        saved = cp.HOST_CPP
        cp.HOST_CPP = src
        try:
            built = cp.build_host(dest)
        finally:
            cp.HOST_CPP = saved
        built.replace(exe)
    return exe


def proof_build(out: Path, config: str, tile: str) -> Path:
    import ctrlpkt_readback as cp
    from harness.build import aiecc_command, aiecc_path
    text, _, layout = clock_proof_design(config, tile)
    d = out / f"proof_{tile}_{config}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "aie.mlir").write_text(text)
    (d / "pkts.bin").write_bytes(cp.pack_words(layout.words))
    (d / "case.json").write_text(json.dumps({"case": d.name, "config": config, "tile": tile,
                                             "labels": list(layout.labels), "n_words": len(layout.words),
                                             "n_data": PROOF_DATA_WORDS}, indent=1) + "\n")
    lp.sh(aiecc_command("peano", aiecc_path()), cwd=d, log=d / "aiecc.log")
    return d


def proof_run(out: Path, tag: str, backend: str, only: list[str]) -> bool:
    """Run the built proof designs; `hw` takes the NPU lock itself (do not wrap it in flock)."""
    import fcntl
    import subprocess
    from harness.run import _env, build_ffi
    host = build_proof_host(out)
    if backend == "emu":
        build_ffi("debug")
    env = _env(backend, "debug")
    ok = True
    for d in sorted(p for p in out.iterdir() if (p / "aie.xclbin").exists() and (p / "case.json").exists()):
        if only and d.name not in only:
            continue
        meta = json.loads((d / "case.json").read_text())
        rd = d / "runs" / tag
        rd.mkdir(parents=True, exist_ok=False)
        cmd = [str(host), str(d / "aie.xclbin"), str(d / "insts.bin"), str(d / "pkts.bin"),
               str(len(meta["labels"])), str(meta["n_data"]), str(rd / "out.bin")]
        lock = None
        if backend == "hw":
            lock = open(lp.HW_LOCK, "w")
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60 if backend == "hw" else 1800)
            rc, text = r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            rc, text = -1, "TIMEOUT"
        finally:
            if lock:
                lock.close()
        (rd / "log.txt").write_text(text)
        print(f"{d.name} {tag}.{backend}: rc {rc}", flush=True)
        ok &= rc == 0
    return ok


def proof_show(out: Path, tag: str, only: list[str]) -> str:
    import struct
    lines = []
    for d in sorted(p for p in out.iterdir() if (p / "case.json").exists()):
        meta = json.loads((d / "case.json").read_text())
        f = d / "runs" / tag / "out.bin"
        if (only and d.name not in only) or not f.exists():
            continue
        n = len(meta["labels"])
        words = list(struct.unpack(f"<{n}I", f.read_bytes()[:4 * n]))
        rb = proof_decode(meta["labels"], words)
        want = proof_expectation(meta["config"], meta["tile"])
        lines.append(f"{d.name} {tag}: {proof_verdict(meta['config'], meta['tile'], rb)}")
        for k, v in rb.items():
            shown = "no reply" if v is None else f"{v:#010x}"
            lines.append(f"   {k:56s} {shown}" + (f"   (landed-write prediction {want[k]:#x})" if k in want else ""))
        h = rb.get(f"handler {HANDLER_STATUS_REG}")
        if h is not None:
            module = REGDB_MODULE[meta["tile"]]
            bits = {f["name"]: (h >> f["bit_range"][0]) & 1 for f in _register(module, HANDLER_STATUS_REG)["bit_fields"]}
            lines.append(f"   handler status {h:#x}: " + " ".join(f"{k}={v}" for k, v in bits.items())
                         + (f"   <- {SLVERR_FIELD} set" if bits.get(SLVERR_FIELD) else ""))
    return "\n".join(lines)


# -- gate pre-wake probe (see the module doc) ------------------------------------------------------

# D is the number of nops between the dummy's and the measured release; the D the trace measures
# (see prewake_row) is a few cycles more. The short cases come first: the idle threshold is unknown
# (128 is the register-database guess), so it is searched from the bottom too. At D = 1..2 the
# dummy word can still be in the switch when the measured word is offered ("switch busy").
PREWAKE_SHORT_DS = (1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 64)
PREWAKE_DS = PREWAKE_SHORT_DS + (20, 50, 100, 120, 126, 127, 128, 129, 130, 140, 200)
PW_WORDS = 16                      # the measured transfer: the row where silicon shows the blip
PW_LOCK_DUMMY, PW_LOCK_MEAS, PW_LOCK_GO = 1, 2, 3
PW_LOCK_DUMMY_DONE, PW_LOCK_MEAS_DONE = 4, 5   # a BD with an acquire also carries a release
OWN_LOCKS = 48                     # raw lock id base of the compute tile's own memory module
PW_SETTLE_ITERS = 60               # about 10 cycles each: the switch idles long before the window
PW_AFTER_MEASURED_NOPS = 150       # the measured transfer is long over before the first filler
PW_FILLERS = 16                    # filler words: their MM2S 1 events complete the trace packet
PW_FILLER_SPACING = 24
PW_FILL_WORDS = 32                 # S2MM 1 BD length: the dummy word plus every filler
PW_SRC, PW_DUMMY_SRC, PW_DST, PW_DUMMY_DST, PW_SCRATCH = 0x0400, 0x0800, 0x8000, 0x8400, 0xC000
PW_PAD_WRITES = 600
PW_BLIP_WINDOW = 40                # BACKPRESSURE this soon after the MM2S 0 start is the onset blip
PW_FRAME_GUARD = 14                # trace edges before the first word at which a packet may have left
# The BDs are started at boot, so START_TASK never fires; the grant (end of STALLED_LOCK) dates a start.
# The trace config needs one core event: one the core program never raises (no `event #1`), so the
# core-module unit records nothing and no core packet ever leaves the tile.
PW_CORE_NEVER = "INSTR_EVENT_1"
PW_EVENTS = ["DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_0_STREAM_BACKPRESSURE", "DMA_MM2S_1_STALLED_LOCK",
             "DMA_MM2S_1_FINISHED_BD", "DMA_S2MM_0_STREAM_STARVATION"]


def prewake_case_name(d: int | None) -> str:
    return "pw_ctl" if d is None else f"pw_d{d:03d}"


def prewake_case_d(name: str) -> int | None:
    return None if name == "pw_ctl" else int(name[len("pw_d"):])


def prewake_core_asm(d: int | None) -> str:
    """go, settle, [release Dm, d nops,] release M, the filler's releases of Dm."""
    n = lp.nops
    body = ["  mov r2, #1", "  mov r3, #-1", *n(2),
            f"  acq #{OWN_LOCKS + PW_LOCK_GO}, r3", *n(4),
            *lp.delay_loop(PW_SETTLE_ITERS, ".Lsettle"), *n(8)]
    if d is None:
        body += n(8)
    else:
        body += [f"  rel #{OWN_LOCKS + PW_LOCK_DUMMY}, r2", *n(d)]
    body += [f"  rel #{OWN_LOCKS + PW_LOCK_MEAS}, r2", *n(PW_AFTER_MEASURED_NOPS)]
    for _ in range(PW_FILLERS):
        body += [f"  rel #{OWN_LOCKS + PW_LOCK_DUMMY}, r2", *n(PW_FILLER_SPACING)]
    body += n(8)
    return lp.frame(body)


def prewake_design_mlir() -> str:
    w, f = PW_WORDS, PW_FILL_WORDS
    ramp = ", ".join(str(0x0A000000 | j) for j in range(w))
    one = str(0x0D000000)
    pad = "\n".join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32"
                    for _ in range(PW_PAD_WRITES))
    return f"""module {{
  aie.device(npu1_1col) {{
    %t_shim = aie.tile(0, 0)
    %t_prod = aie.tile(0, 2)
    %src = aie.buffer(%t_prod) {{address = {PW_SRC} : i32, sym_name = "src", initial_value = dense<[{ramp}]> : tensor<{w}xi32>}} : memref<{w}xi32>
    %dsrc = aie.buffer(%t_prod) {{address = {PW_DUMMY_SRC} : i32, sym_name = "dsrc", initial_value = dense<[{one}]> : tensor<1xi32>}} : memref<1xi32>
    %dst = aie.buffer(%t_prod) {{address = {PW_DST} : i32, sym_name = "dst"}} : memref<{w}xi32>
    %ddst = aie.buffer(%t_prod) {{address = {PW_DUMMY_DST} : i32, sym_name = "ddst"}} : memref<{f}xi32>
    %scratch = aie.buffer(%t_prod) {{address = {PW_SCRATCH} : i32, sym_name = "scratch"}} : memref<4xi32>
    %lDm = aie.lock(%t_prod, {PW_LOCK_DUMMY}) {{init = 0 : i32, sym_name = "lDm"}}
    %lM = aie.lock(%t_prod, {PW_LOCK_MEAS}) {{init = 0 : i32, sym_name = "lM"}}
    %lGo = aie.lock(%t_prod, {PW_LOCK_GO}) {{init = 0 : i32, sym_name = "lGo"}}
    %lDd = aie.lock(%t_prod, {PW_LOCK_DUMMY_DONE}) {{init = 0 : i32, sym_name = "lDd"}}
    %lMd = aie.lock(%t_prod, {PW_LOCK_MEAS_DONE}) {{init = 0 : i32, sym_name = "lMd"}}
    aie.flow(%t_prod, DMA : 0, %t_prod, DMA : 0)
    aie.flow(%t_prod, DMA : 1, %t_prod, DMA : 1)
    func.func private @k_core(memref<4xi32>) attributes {{link_with = "core.o"}}
    aie.core(%t_prod) {{
      func.call @k_core(%scratch) : (memref<4xi32>) -> ()
      aie.end
    }}
    %mem = aie.mem(%t_prod) {{
      %s0 = aie.dma_start(MM2S, 0, ^send0, ^start1)
    ^send0:
      %ca = arith.constant 1 : i32
      aie.use_lock(%lM, AcquireGreaterEqual, %ca)
      aie.dma_bd(%src : memref<{w}xi32>)
      %cr0 = arith.constant 1 : i32
      aie.use_lock(%lMd, Release, %cr0)
      aie.next_bd ^end
    ^start1:
      %s1 = aie.dma_start(MM2S, 1, ^send1, ^start2)
    ^send1:
      %cb = arith.constant 1 : i32
      aie.use_lock(%lDm, AcquireGreaterEqual, %cb)
      aie.dma_bd(%dsrc : memref<1xi32>)
      %cr1 = arith.constant 1 : i32
      aie.use_lock(%lDd, Release, %cr1)
      aie.next_bd ^send1
    ^start2:
      %s2 = aie.dma_start(S2MM, 0, ^recv0, ^start3)
    ^recv0:
      aie.dma_bd(%dst : memref<{w}xi32>)
      aie.next_bd ^end
    ^start3:
      %s3 = aie.dma_start(S2MM, 1, ^recv1, ^end)
    ^recv1:
      aie.dma_bd(%ddst : memref<{f}xi32>)
      aie.next_bd ^end
    ^end:
      aie.end
    }}
    aie.runtime_sequence(%dummy: memref<4xi32>) {{
      %c_zero = arith.constant 0 : i32
      %c_one = arith.constant 1 : i32
      aiex.set_lock(%lGo, %c_one)
{pad}
    }}
  }}
}}
"""


def prewake_inject_args() -> list[str]:
    return ["--memmod-sweep-events", ",".join(PW_EVENTS), "--core-grounding", PW_CORE_NEVER, "--core-sweep-events", ""]


def prewake_build_case(d: int | None, out: Path) -> None:
    name = prewake_case_name(d)
    dd = out / name
    dd.mkdir(parents=True, exist_ok=True)
    lp.assemble(prewake_core_asm(d), dd / "core.o", "k_core")
    (dd / "base.mlir").write_text(prewake_design_mlir())
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time", *prewake_inject_args(),
           "--trace-config-out", "trace_config.json", "--config-test-name", name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=dd, log=dd / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=dd, log=dd / "aiecc.log")
    (dd / "case.json").write_text(json.dumps({"case": name, "d": d}) + "\n")


def prewake_row(events) -> dict:
    """One run of the tile's memory module as numbers. `events` are level-expanded (an event on
    every asserted cycle) objects with name, ts, row, pkt_type.

    d_start = MM2S 0 start minus MM2S 1 start (a start is the grant: the cycle after the end of its
    STALLED_LOCK level); d_fin the same against MM2S 1 FINISHED_BD
    (the dummy word has left the DMA by then). blip = STREAM_BACKPRESSURE within PW_BLIP_WINDOW
    cycles of the MM2S 0 start, at blip_at. latency = cycles from the MM2S 0 start to the first
    word at S2MM 0 (the end of its STREAM_STARVATION level that spans the start). frames = trace
    edges up to the first word; packet_risk when a trace packet could have left in the window."""
    stamps: dict[str, list[int]] = {}
    core_seen = sum(1 for e in events if e.row == 2 and e.pkt_type == 0)
    for e in events:
        if e.row == 2 and e.pkt_type == 1:
            stamps.setdefault(e.name, []).append(e.ts)
    grant = lambda name: next(iter(b + 1 for _, b in _level_runs(stamps.get(name, []))), None)
    s0 = grant("DMA_MM2S_0_STALLED_LOCK")
    if s0 is None:
        return {"start0": None}
    start1 = grant("DMA_MM2S_1_STALLED_LOCK")
    fin1 = min(stamps.get("DMA_MM2S_1_FINISHED_BD", [None]), default=None)
    if start1 is not None and start1 > s0 + PW_BLIP_WINDOW:
        start1 = fin1 = None           # the control: MM2S 1 only starts later, as the filler
    row = {"start0": s0, "d_start": None if start1 is None else s0 - start1,
           "d_fin": None if fin1 is None else s0 - fin1}
    bp = [t - s0 for t in stamps.get("DMA_MM2S_0_STREAM_BACKPRESSURE", []) if 0 <= t - s0 <= PW_BLIP_WINDOW]
    row["blip"], row["blip_at"], row["blip_cycles"] = bool(bp), (min(bp) if bp else None), len(bp)
    run = next(((a, b) for a, b in _level_runs(stamps.get("DMA_S2MM_0_STREAM_STARVATION", []))
                if a <= s0 <= b), None)
    row["latency"] = None if run is None else run[1] + 1 - s0
    arrival = s0 + row["latency"] if row["latency"] is not None else s0 + PW_BLIP_WINDOW
    row["frames"] = sum((a <= arrival) + (b + 1 <= arrival)
                        for ts in stamps.values() for a, b in _level_runs(ts))
    row["core_events"] = core_seen        # any core-module event means core packets left the tile
    row["packet_risk"] = row["frames"] >= PW_FRAME_GUARD or core_seen > 0
    return row


def _level_runs(ts) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for t in sorted(set(ts)):
        if out and t == out[-1][1] + 1:
            out[-1] = (out[-1][0], t)
        else:
            out.append((t, t))
    return out


def prewake_threshold(rows, key: str = "blip"):
    """(largest D without `key`, smallest D with it) over rows that have a d_start; None when
    `key` is not a monotone function of D. The upper is None if no row has it."""
    pts = sorted((r["d_start"], r[key]) for r in rows if r.get("d_start") is not None)
    if any(a and not b for (_, a), (_, b) in zip(pts, pts[1:])):
        return None
    no = [d for d, b in pts if not b]
    yes = [d for d, b in pts if b]
    return (max(no) if no else None, min(yes) if yes else None)


def prewake_flip(points, key: str):
    """Where `key` flips with D over pooled rows. points: (d_start, row, label). Returns
    (largest D without, smallest D with, offending labels); the labels are empty unless the rows
    interleave (a row with the feature at a D no larger than a row without it)."""
    no = [(d, lab) for d, r, lab in points if not r[key]]
    yes = [(d, lab) for d, r, lab in points if r[key]]
    no_max = max((d for d, _ in no), default=None)
    yes_min = min((d for d, _ in yes), default=None)
    if no_max is None or yes_min is None or no_max < yes_min:
        return no_max, yes_min, []
    bad = [f"{lab} D={d}" for d, lab in sorted(yes) if d <= no_max]
    bad += [f"{lab} D={d}" for d, lab in sorted(no) if d >= yes_min]
    return no_max, yes_min, bad


def _platform(run: str) -> str:
    """The backend suffix of a run tag (`h1.hw` -> `hw`); a tag without one is its own platform."""
    return run.rsplit(".", 1)[1] if "." in run else run


def format_prewake_table(rows) -> str:
    """rows: (case, run, prewake_row dict). Cases a run did not capture are listed once per run;
    the flip summaries pool all runs of a platform."""
    lines = [f"{'case':<8} {'run':<8} {'D(start)':>8} {'D(fin)':>7} {'blip':>5} {'blip@':>6} {'1st word':>8} "
             f"{'frames':>6}  note"]
    ok: dict[str, list] = {}
    absent: dict[str, list[str]] = {}
    for case, run, r in sorted(rows, key=lambda x: (x[1], prewake_case_d(x[0]) if x[0] != "pw_ctl" else -1)):
        if r["start0"] is None:
            absent.setdefault(run, []).append(case)
            continue
        fmt = lambda v: "-" if v is None else str(v)
        note = "PACKET? do not read" if r["packet_risk"] else ""
        lines.append(f"{case:<8} {run:<8} {fmt(r['d_start']):>8} {fmt(r['d_fin']):>7} "
                     f"{'yes' if r['blip'] else 'no':>5} {fmt(r['blip_at']):>6} {fmt(r['latency']):>8} "
                     f"{r['frames']:>6}  {note}")
        if not r["packet_risk"] and r["d_start"] is not None:
            ok.setdefault(_platform(run), []).append((case, run, r))
    for run, cases in absent.items():
        lines.append(f"not in this run ({run}, no MM2S 0 grant captured): {' '.join(cases)}")
    for platform, items in ok.items():
        lats = [r["latency"] for _, _, r in items if r["latency"] is not None]
        for _, _, r in items:
            r["late"] = bool(lats) and r["latency"] is not None and r["latency"] > min(lats)
        points = [(r["d_start"], r, f"{run} {case}") for case, run, r in items]
        for key, what in (("blip", "the BACKPRESSURE blip"), ("late", "a first word one cycle late")):
            no, yes, bad = prewake_flip(points, key)
            if bad:
                verdict = f"overlap, rows interleave: {'; '.join(bad)}"
            elif yes is None:
                verdict = f"{no} (absent), none present"
            else:
                verdict = f"{no} (absent) and {yes} (present)"
            lines.append(f"{platform}: {what} flips between D(start) = {verdict}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out_dir")
    b.add_argument("--cases", nargs="*", default=PLAIN_CASES)
    b.add_argument("--clock", choices=sorted(CLOCK_CONFIGS), default="K0",
                   help="adaptive clock-gate configuration of the data tiles (see the module doc)")
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("tag")
    r.add_argument("--backend", choices=("hw", "emu"), required=True)
    r.add_argument("--only", nargs="*", default=PLAIN_CASES)
    s = sub.add_parser("show")
    s.add_argument("out_dir")
    s.add_argument("tags", nargs="+")
    s.add_argument("--only", nargs="*", default=PLAIN_CASES)
    po = sub.add_parser("ports", help="per-port RUNNING counts and stall onsets of port-watching cases")
    po.add_argument("out_dir")
    po.add_argument("tags", nargs="+")
    po.add_argument("--only", nargs="*", default=[c for c in CASES if c.endswith("_ports")])
    pb = sub.add_parser("proof-build", help="trace-free design: read, masked write, read back (control packets)")
    pb.add_argument("out_dir")
    pb.add_argument("--clock", nargs="+", choices=sorted(CLOCK_CONFIGS), default=["K1"])
    pb.add_argument("--tile", nargs="+", choices=sorted(PROOF_TARGET), default=["prod"])
    pr = sub.add_parser("proof-run")
    pr.add_argument("out_dir")
    pr.add_argument("tag")
    pr.add_argument("--backend", choices=("hw", "emu"), required=True)
    pr.add_argument("--only", nargs="*", default=[])
    ps = sub.add_parser("proof-show")
    ps.add_argument("out_dir")
    ps.add_argument("tag")
    ps.add_argument("--only", nargs="*", default=[])
    wb = sub.add_parser("prewake-build", help="gate pre-wake probe: build the D cases and the control")
    wb.add_argument("out_dir")
    wb.add_argument("--ds", nargs="*", type=int, default=list(PREWAKE_DS))
    wb.add_argument("--no-control", action="store_true")
    wr = sub.add_parser("prewake-run", help="gate pre-wake probe: run the built cases to TAG.BACKEND/")
    wr.add_argument("out_dir")
    wr.add_argument("tag")
    wr.add_argument("--backend", choices=("hw", "emu"), required=True)
    wr.add_argument("--ds", nargs="*", type=int, default=list(PREWAKE_DS))
    wr.add_argument("--no-control", action="store_true")
    wt = sub.add_parser("prewake-table", help="gate pre-wake probe: D, blip and first-word latency per case")
    wt.add_argument("out_dir")
    wt.add_argument("tags", nargs="+", help="run directory names, e.g. h1.hw e1.emu")
    wt.add_argument("--ds", nargs="*", type=int, default=list(PREWAKE_DS))
    wt.add_argument("--no-control", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    if a.cmd.startswith("prewake-"):
        cases = ([] if a.no_control else [None]) + list(a.ds)
        if a.cmd == "prewake-build":
            for d in cases:
                prewake_build_case(d, out)
                print("built", prewake_case_name(d))
            return 0
        if a.cmd == "prewake-run":
            for d in [c for c in cases if (out / prewake_case_name(c)).is_dir()]:
                if not lp.run_case(out, prewake_case_name(d), a.tag, a.backend, output=False):
                    print(f"{prewake_case_name(d)}: run did not complete; stop")
                    return 1
            return 0
        import gen_dma_lock_probe as dl
        rows = []
        for d in cases:
            for tag in a.tags:
                run_dir = out / prewake_case_name(d) / tag
                if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
                    rows.append((prewake_case_name(d), tag, {"start0": None}))
                    continue
                rows.append((prewake_case_name(d), tag, prewake_row(dl.load_level_events(run_dir))))
        print(format_prewake_table(rows))
        return 0
    if a.cmd == "proof-build":
        for c in a.clock:
            for t in a.tile:
                print(f"built {proof_build(out, c, t)}")
        return 0
    if a.cmd == "proof-run":
        return 0 if proof_run(out, a.tag, a.backend, a.only) else 1
    if a.cmd == "proof-show":
        print(proof_show(out, a.tag, a.only))
        return 0
    if a.cmd == "build":
        for n in a.cases:
            build_case(n, out, a.clock)
            print(f"built {n}")
        return 0
    if a.cmd == "run":
        ok = True
        for n in a.only:
            ok &= lp.run_case(out, n, a.tag, a.backend, output=False)
        return 0 if ok else 1
    for n in a.only:
        src, dst, _ = case_spec(n)
        for tag in a.tags:
            p = out / n / tag / "events.json"
            if not p.exists():
                print(f"{n} {tag}: no events.json")
                continue
            body = ports_summary(p, src, dst) if a.cmd == "ports" else summarize(p, src, dst)
            print(f"{n} {tag} ({src} -> {dst}):\n{body}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
