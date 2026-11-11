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

`run --backend hw` takes the NPU lock (`build/experiments/.npu-hw.lock`).
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_lock_pulse_probe as lp  # noqa: E402  (build/run plumbing: sh, run_case)

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
               mode: str = "single") -> str:
    core_src = src == "core"
    srct = TILE_OF[src]
    tiles = list(dict.fromkeys((srct, dst)))
    decls = "\n".join(f"    {VAR[t]} = aie.tile({TILES[t][0]}, {TILES[t][1]})" for t in tiles)
    words = [0x0A000000 | j for j in range(n)]
    pad = "\n".join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32"
                    for _ in range(SETTLE_WRITES))
    sv, dv = VAR[srct], VAR[dst]
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
      %tr = aiex.dma_configure_task({dv}, S2MM, 0) {{
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


def build_case(name: str, out: Path) -> None:
    shape = case_shape(name)
    src, dst, n = shape.src, shape.dst, shape.total
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "base.mlir").write_text(design_mlir(*shape.design_args()))
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time",
           *inject_args(name),
           "--trace-config-out", "trace_config.json", "--config-test-name", name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    (d / "case.json").write_text(json.dumps({"case": name, "src": src, "dst": dst, "n_words": n}) + "\n")


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out_dir")
    b.add_argument("--cases", nargs="*", default=PLAIN_CASES)
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
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    if a.cmd == "build":
        for n in a.cases:
            build_case(n, out)
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
