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

`run --backend hw` takes the NPU lock (`build/experiments/.npu-hw.lock`).
"""
from __future__ import annotations

import argparse
import json
import re
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
CASES = {"mem_own": ("mem", "mem"), "prod_own": ("prod", "prod"),
         "mem_prod": ("mem", "prod"), "prod_mem": ("prod", "mem")}
# Gate and done locks of the S2MM (free in both lock spaces: memtile 0..63, compute 0..15).
GATE_LOCK, DONE_LOCK = 4, 5

MEMTILE_EVENTS = ("DMA_MM2S_SEL0_START_TASK,DMA_MM2S_SEL0_STREAM_BACKPRESSURE,DMA_MM2S_SEL0_FINISHED_BD,"
                  "DMA_MM2S_SEL0_MEMORY_STARVATION,DMA_S2MM_SEL0_START_TASK,DMA_S2MM_SEL0_STALLED_LOCK,"
                  "DMA_S2MM_SEL0_FINISHED_BD,DMA_S2MM_SEL0_STREAM_STARVATION")
MEMMOD_EVENTS = ("DMA_MM2S_0_START_TASK,DMA_MM2S_0_STREAM_BACKPRESSURE,DMA_MM2S_0_FINISHED_BD,"
                 "DMA_MM2S_0_MEMORY_STARVATION,DMA_S2MM_0_START_TASK,DMA_S2MM_0_STALLED_LOCK,"
                 "DMA_S2MM_0_FINISHED_BD,DMA_S2MM_0_STREAM_STARVATION")


def case_spec(name: str) -> tuple[str, str, int]:
    """(source, destination, words) of a case name, `BASE` or `BASE_nN`."""
    m = re.fullmatch(r"(.+?)(?:_n(\d+))?", name)
    src, dst = CASES[m.group(1)]
    return src, dst, int(m.group(2)) if m.group(2) else N_WORDS


def design_mlir(src: str, dst: str, n: int = N_WORDS) -> str:
    tiles = [src] + ([dst] if dst != src else [])
    decls = "\n".join(f"    {VAR[t]} = aie.tile({TILES[t][0]}, {TILES[t][1]})" for t in tiles)
    data = ", ".join(str(0x0A000000 | j) for j in range(n))
    pad = "\n".join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32"
                    for _ in range(SETTLE_WRITES))
    return f"""module {{
  aie.device(npu1_1col) {{
    %t_shim = aie.tile(0, 0)
{decls}
    %src = aie.buffer({VAR[src]}) {{address = {SRC_ADDR} : i32, sym_name = "src", initial_value = dense<[{data}]> : tensor<{n}xi32>}} : memref<{n}xi32>
    %dst = aie.buffer({VAR[dst]}) {{address = {DST_ADDR} : i32, sym_name = "dst"}} : memref<{n}xi32>
    %scratch = aie.buffer({VAR[src]}) {{address = {SCRATCH_ADDR} : i32, sym_name = "scratch"}} : memref<4xi32>
    %lG = aie.lock({VAR[dst]}, {GATE_LOCK}) {{init = 0 : i32, sym_name = "lG"}}
    %lD = aie.lock({VAR[dst]}, {DONE_LOCK}) {{init = 0 : i32, sym_name = "lD"}}
    aie.flow({VAR[src]}, DMA : 0, {VAR[dst]}, DMA : 0)
    aie.runtime_sequence(%dummy: memref<4xi32>) {{
      %c_zero = arith.constant 0 : i32
      %c_one = arith.constant 1 : i32
      %tr = aiex.dma_configure_task({VAR[dst]}, S2MM, 0) {{
        %c_a = arith.constant 1 : i32
        aie.use_lock(%lG, AcquireGreaterEqual, %c_a)
        aie.dma_bd(%dst : memref<{n}xi32> offset = 0 len = {n}) {{bd_id = 0 : i32}}
        %c_r = arith.constant 1 : i32
        aie.use_lock(%lD, Release, %c_r)
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tr)
      %ts = aiex.dma_configure_task({VAR[src]}, MM2S, 0) {{
        aie.dma_bd(%src : memref<{n}xi32> offset = 0 len = {n}) {{bd_id = 1 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%ts)
{pad}
      aiex.set_lock(%lG, %c_one)
      aiex.dma_await_task(%ts)
      aiex.dma_await_task(%tr)
    }}
  }}
}}
"""


def build_case(name: str, out: Path) -> None:
    src, dst, n = case_spec(name)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "base.mlir").write_text(design_mlir(src, dst, n))
    traced = {src, dst}
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time",
           *(["--memtile-sweep-events", MEMTILE_EVENTS] if "mem" in traced else []),
           *(["--memmod-sweep-events", MEMMOD_EVENTS] if "prod" in traced else []),
           "--trace-config-out", "trace_config.json", "--config-test-name", name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    (d / "case.json").write_text(json.dumps({"case": name, "src": src, "dst": dst, "n_words": n}) + "\n")


def tile_frames(path: Path, row: int) -> list[tuple[int, frozenset[str]]]:
    """The trace unit's frames for one tile row: (cycle, events active). An event-time frame is
    emitted when the active set changes, so an event is high from a frame naming it to the next
    frame that does not."""
    by_ts: dict[int, set[str]] = {}
    for e in json.loads(path.read_text())["events"]:
        if e["row"] == row:
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
    b.add_argument("--cases", nargs="*", default=list(CASES))
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("tag")
    r.add_argument("--backend", choices=("hw", "emu"), required=True)
    r.add_argument("--only", nargs="*", default=list(CASES))
    s = sub.add_parser("show")
    s.add_argument("out_dir")
    s.add_argument("tags", nargs="+")
    s.add_argument("--only", nargs="*", default=list(CASES))
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
            print(f"{n} {tag} ({src} -> {dst}):\n{summarize(p, src, dst)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
