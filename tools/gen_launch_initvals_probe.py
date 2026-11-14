#!/usr/bin/env python3
"""Launch probe with the configuration of a real design added one factor at a time.

The launch probe (gen_launch_probe.py) dates a core's start against the first operation of the host
runtime sequence: the core stamps its timer at entry, spins until the timer runs backwards, and stores
both stamps; the sequence's measured operation is a reset of that core's timer (Timer_Control, bit 31),
and the lead = (last stamp before the reset) - (stamp at entry) is how many cycles the core started
ahead of that operation. Here the measurement is the same kernel, the same reset operation and the same
lead; only what surrounds it changes. A design traced on silicon implies a gap from core enable to trace
start larger than the launch probe's number plus the operations ahead of the trace start. These cases
say which part of such a design, if any, lengthens the gap.

Cases (each differs from `base` in one factor; `all` is the three together):
  base      one core at (0,2), no padding: the launch probe's single-core case, re-measured.
  initvals  + a memtile buffer of 4096 words with initial values and a memtile MM2S task started by the
            configuration, toward a declared sink: a compute-tile S2MM channel at (0,3) with one BD of
            exactly 4096 words (no core there, nothing in the sequence waits on it).
  bigbuf    + host buffers sized like the real test: a 64 KiB output buffer (the 8 timer words are drained
            to its last 32 bytes) and a 1 MiB second buffer. The second buffer is passed to the kernel as
            an argument but no shim DMA task references it.
  prefix    + 38 register writes ahead of the measured operation, from the register classes a traced run's
            trace setup writes (see PREFIX_SETUP): trace unit, performance counter, timer and event
            broadcast registers of the compute tile and the shim, with the values traced designs write and
            without the writes that start anything (no software event, no DMA task queue).
  all       initvals + bigbuf + prefix.

Why every case ends, whatever silicon does (one line per case in `termination`). The core stores its
timer and returns after a bounded spin (SPIN_ITERS), so it halts whether or not the reset is seen. The
only wait in the runtime sequence is the one the launch probe has: on the 8-word timer drain, which
completes once the core has stored them. The memtile MM2S task and its sink are one BD each of exactly
4096 words and nothing waits on either: a task that stalls or finishes late only stays unfinished. The
prefix writes configure registers and start no task.

Host memory. Every shim DMA BD is patched to argument 0 (the output buffer) at the same 32 bytes the
launch probe drains; the runner sizes that argument from the instruction stream (4 KiB when the drain is
at the start, 64 KiB when it is at the end) and refuses a run whose tasks leave their buffers. `bounds`
prints the runner's sizes and the BD ranges without a device.

    python3 tools/gen_launch_initvals_probe.py cases
    python3 tools/gen_launch_initvals_probe.py build OUT_DIR [--cases base initvals ...]
    python3 tools/gen_launch_initvals_probe.py run OUT_DIR TAG --backend hw|emu [--cases ...]
    python3 tools/gen_launch_initvals_probe.py table OUT_DIR RUN [RUN ...] [--cases ...]
    python3 tools/gen_launch_initvals_probe.py bounds OUT_DIR [--cases ...]
    python3 tools/gen_launch_initvals_probe.py session      # the hardware order: three rounds over the set

The emulator backend and library are the caller's choice (environment). `run --backend hw` removes every
emulator variable from the runner's environment.
"""
from __future__ import annotations

import argparse
import fcntl
import statistics
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_launch_probe as launch                                   # noqa: E402
import gen_lock_pulse_probe as lp                                   # noqa: E402
import gen_packet_stage_probe as ps                                 # noqa: E402

# -- constants ---------------------------------------------------------------------------

WORD = 4
OUT_WORDS = launch.OUT_WORDS               # the core's 8 timer words
INIT_WORDS = 4096                          # memtile buffer with initial values, MM2S task and sink BD length
BIG_OUT_WORDS = 16384                      # 64 KiB output buffer of the real test
BIG_SECOND_BYTES = 1 << 20                 # 1 MiB second buffer of the real test
SMALL_SECOND_BYTES = 4096                  # second buffer of the other cases (one page)
PREFIX_OPS = 38
ROUNDS = 3
SECOND_ARG = 1                             # runtime sequence arguments: 0 = output, 1 = second buffer
CORE = (0, 2)                              # the measured core (logical coordinates)
MEMTILE = (0, 1)
SINK = (0, 3)

# Registers the traced designs' trace setup writes, as (row, tile-local address, value): the compute
# tile's core and memory modules (trace unit, performance counter 2, timers) and the shim (trace unit,
# timer, event broadcasts). Values are the ones traced designs write. None of them starts anything: the
# trace units start on broadcast events nobody generates here, and the writes that would start (the
# software event generator, the DMA task queue) and every BD register are not in the list. A memtile
# register is not written, because cases without a configured memtile must not touch one.
PREFIX_SETUP: tuple[tuple[int, int, int], ...] = (
    (2, 0x340D0, 0x797A0000),   # core Trace_Control0
    (2, 0x340D4, 0x00000001),   # core Trace_Control1
    (2, 0x340E0, 0x25222107),   # core Trace_Event0
    (2, 0x340E4, 0x2C1A1817),   # core Trace_Event1
    (2, 0x140D4, 0x00001004),   # memory module Trace_Control1
    (2, 0x140D0, 0x797A0000),   # memory module Trace_Control0
    (2, 0x140E0, 0x4E4D1513),   # memory module Trace_Event0
    (2, 0x140E4, 0x0C0B504F),   # memory module Trace_Event1
    (2, 0x31504, 0x0000001C),   # core Performance_Control1
    (2, 0x31508, 0x00070000),   # core Performance_Control2
    (2, 0x31588, 0x00000400),   # core Performance_Counter2_Event_Value
    (2, 0x34000, 0x00007A00),   # core Timer_Control (reset event select; the measured reset comes later)
    (2, 0x14000, 0x00007A00),   # memory module Timer_Control
    (0, 0x340D4, 0x00002002),   # shim Trace_Control1
    (0, 0x340D0, 0x7E7F0000),   # shim Trace_Control0
    (0, 0x340E0, 0x16100F0E),   # shim Trace_Event0
    (0, 0x340E4, 0x1F1E1817),   # shim Trace_Event1
    (0, 0x34000, 0x00007F00),   # shim Timer_Control
    (0, 0x3404C, 0x0000007F),   # shim Event_Broadcast15_A
    (0, 0x34048, 0x0000007E),   # shim Event_Broadcast14_A
)


def prefix_writes() -> tuple[tuple[int, int, int], ...]:
    """The 38 prefix operations: the setup once, then its first 18 writes again (same registers, same
    values), so the mix stays the one a traced run writes and no register gets a value it never gets."""
    return (PREFIX_SETUP + PREFIX_SETUP)[:PREFIX_OPS]


# -- cases -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    name: str
    initvals: bool = False
    bigbuf: bool = False
    prefix: bool = False

    @property
    def out_words(self) -> int:
        return BIG_OUT_WORDS if self.bigbuf else OUT_WORDS

    @property
    def drain_offset_words(self) -> int:
        """Where the 8 timer words land in the output buffer: the end of a big buffer, the start of a small one."""
        return self.out_words - OUT_WORDS

    @property
    def second_bytes(self) -> int:
        return BIG_SECOND_BYTES if self.bigbuf else SMALL_SECOND_BYTES

    @property
    def write_ops(self) -> int:
        """Register writes ahead of the measured reset."""
        return PREFIX_OPS if self.prefix else 0


def default_cases() -> list[Case]:
    return [Case("base"),
            Case("initvals", initvals=True),
            Case("bigbuf", bigbuf=True),
            Case("prefix", prefix=True),
            Case("all", initvals=True, bigbuf=True, prefix=True)]


def case_from_name(name: str) -> Case:
    for c in default_cases():
        if c.name == name:
            return c
    raise ValueError(f"unknown case {name!r}")


def session_order() -> list[tuple[str, str]]:
    """(case, tag) per hardware run: base first in every round, tags h1..h3 per round."""
    return [(c.name, f"h{r}") for r in range(1, ROUNDS + 1) for c in default_cases()]


TERMINATION = {
    "base": "the core stores its timer and returns after a bounded spin; the only wait is on the 8-word "
            "timer drain, which the stores complete",
    "initvals": "as base; the memtile MM2S task and its sink are one BD of 4096 words each and nothing waits "
                "on either (a stalled task only stays unfinished)",
    "bigbuf": "as base; the second buffer is referenced by no DMA task and the drain is the same 8 words",
    "prefix": "as base; the prefix writes configure registers and start no task",
    "all": "as initvals, bigbuf and prefix together: no wait but the timer drain",
}


# -- design ------------------------------------------------------------------------------

def _i32(v: int) -> int:
    return v - (1 << 32) if v >= 1 << 31 else v


def design_mlir(case: Case) -> str:
    """The case's design. The core, its output fifo and the measured reset are the launch probe's base
    case as the harness emits it; the rest is what the factors add."""
    col, row = CORE
    tiles = ["    %tile_0_0 = aie.tile(0, 0)"]
    if case.initvals:
        tiles.append(f"    %tile_{MEMTILE[0]}_{MEMTILE[1]} = aie.tile({MEMTILE[0]}, {MEMTILE[1]})")
    tiles.append(f"    %tile_{col}_{row} = aie.tile({col}, {row})")
    if case.initvals:
        tiles.append(f"    %tile_{SINK[0]}_{SINK[1]} = aie.tile({SINK[0]}, {SINK[1]})")
    body = ""
    if case.initvals:
        mt, sk = f"%tile_{MEMTILE[0]}_{MEMTILE[1]}", f"%tile_{SINK[0]}_{SINK[1]}"
        values = ", ".join(str(i) for i in range(1, INIT_WORDS + 1))
        body += f"""
    %mt_buf = aie.buffer({mt}) {{address = 0 : i32, sym_name = "mt_buf", initial_value = dense<[{values}]> : tensor<{INIT_WORDS}xi32>}} : memref<{INIT_WORDS}xi32>
    %sink_buf = aie.buffer({sk}) {{address = 0 : i32, sym_name = "sink_buf"}} : memref<{INIT_WORDS}xi32>
    aie.flow({mt}, DMA : 0, {sk}, DMA : 0)
    %mt_dma = aie.memtile_dma({mt}) {{
      %m0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%mt_buf : memref<{INIT_WORDS}xi32> offset = 0 len = {INIT_WORDS})
      aie.next_bd ^end
    ^end:
      aie.end
    }}
    %sink_dma = aie.mem({sk}) {{
      %s0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%sink_buf : memref<{INIT_WORDS}xi32> offset = 0 len = {INIT_WORDS})
      aie.next_bd ^end
    ^end:
      aie.end
    }}
"""
    ops = []
    k = 0
    for r, addr, val in (prefix_writes() if case.prefix else ()):
        ops += [f"      %w{k}_addr = arith.constant {_i32(addr)} : i32",
                f"      %w{k}_val = arith.constant {_i32(val)} : i32",
                f"      aiex.npu.write32(%w{k}_addr, %w{k}_val) {{column = 0 : i32, row = {r} : i32}} : i32, i32"]
        k += 1
    ops += [f"      %w{k}_addr = arith.constant {_i32(launch.TIMER_CONTROL)} : i32",
            f"      %w{k}_val = arith.constant {_i32(launch.TIMER_RESET)} : i32",
            f"      aiex.npu.write32(%w{k}_addr, %w{k}_val) {{column = {col} : i32, row = {row} : i32}} : i32, i32"]
    ops_text = "\n".join(ops)
    consts = ["      %c0 = arith.constant 0 : i64", "      %c1 = arith.constant 1 : i64",
              f"      %c8 = arith.constant {OUT_WORDS} : i64"]
    offset = "%c0"
    if case.drain_offset_words:
        consts.append(f"      %coff = arith.constant {case.drain_offset_words} : i64")
        offset = "%coff"
    consts_text = "\n".join(consts)
    return f"""module {{
  aie.device(npu1) {{
{chr(10).join(tiles)}

    aie.objectfifo @of_ts(%tile_{col}_{row}, {{%tile_0_0}}, 1 : i32) : !aie.objectfifo<memref<{OUT_WORDS}xi32>>

    func.func private @k_core(memref<{OUT_WORDS}xi32>) attributes {{link_with = "core.o"}}

    aie.core(%tile_{col}_{row}) {{
      %a0 = aie.objectfifo.acquire @of_ts(Produce, 1) : memref<{OUT_WORDS}xi32>
      func.call @k_core(%a0) : (memref<{OUT_WORDS}xi32>) -> ()
      aie.objectfifo.release @of_ts(Produce, 1)
      aie.end
    }}
{body}
    aie.runtime_sequence(%out : memref<{case.out_words}xi32>, %second : memref<{case.second_bytes}xi8>) {{
{consts_text}
{ops_text}
      aiex.npu.dma_memcpy_nd(%out[%c0, %c0, %c0, {offset}][%c1, %c1, %c1, %c8][%c0, %c0, %c0, %c1]) {{metadata = @of_ts, id = 0 : i64, issue_token = true}} : memref<{case.out_words}xi32>
      aiex.npu.dma_wait {{symbol = @of_ts}}
    }}
  }}
}}
"""


# -- host buffers ------------------------------------------------------------------------

def expected_arg_sizes(case: Case) -> dict[int, int]:
    """What the runner must derive from the instruction stream: argument 0 to the end of the drained
    words. The second buffer has no task, so the runner derives nothing for it."""
    return {0: (case.drain_offset_words + OUT_WORDS) * WORD}


def allocated_arg_sizes(case: Case) -> dict[int, int]:
    """Every host buffer the run allocates: the output buffer (sized from the stream) and the second one
    (the runner's trace-size option)."""
    return {**expected_arg_sizes(case), SECOND_ARG: case.second_bytes}


def drain_words(out: bytes, case: Case) -> tuple[int, ...]:
    """The 8 timer words of a run's output buffer."""
    start = case.drain_offset_words * WORD
    return struct.unpack("<8I", out[start:start + OUT_WORDS * WORD])


def runner_arg_sizes(insts: Path) -> dict[int, int]:
    p = subprocess.run([str(lp.RUNNER), "--print-arg-sizes", str(insts)], capture_output=True, text=True, check=True)
    return {int(a): int(b) for a, b in (ln.split() for ln in p.stdout.splitlines() if ln.strip())}


def bounds_lines(insts: Path, sizes: dict[int, int]) -> tuple[int, str]:
    """The runner's device-free bounds check: (exit status, its report), every shim DMA task named."""
    args = [str(lp.RUNNER), "--check-bounds", str(insts)] + [f"{i}={b}" for i, b in sorted(sizes.items())]
    p = subprocess.run(args, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


# -- results -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Result:
    lead: int | None        # cycles the core started ahead of the reset write; None = nothing stored
    jumped: int
    t0: int
    iterations_left: int


def result_of(words: tuple[int, ...]) -> Result:
    if not any(words):
        return Result(None, 0, 0, 0)
    return Result(launch.lead(words), words[5], words[0], words[6])


def summarize(leads: list[int]) -> str:
    return f"median {statistics.median(leads):.0f} min {min(leads)} max {max(leads)} n {len(leads)}"


def format_table(rows: list[tuple[str, str, Result]]) -> str:
    """One line per run, then per case the median over its runs and the difference to the base median.
    A lead is the core's start ahead of the measured operation; the difference is what the factor adds."""
    out = [f"{'case':9} {'run':8} {'lead':>8} {'jumped':>6} {'t0':>10} {'iters_left':>10}"]
    for case, run, r in rows:
        out.append(f"{case:9} {run:8} {'-' if r.lead is None else r.lead:>8} {r.jumped:>6} {r.t0:>10} {r.iterations_left:>10}")
    by_case: dict[str, list[int]] = {}
    for case, _, r in rows:
        if r.lead is not None:
            by_case.setdefault(case, []).append(r.lead)
    out.append("")
    base = by_case.get("base")
    base_med = statistics.median(base) if base else None
    for case, leads in by_case.items():
        delta = "" if base_med is None else f"  vs base {statistics.median(leads) - base_med:+.0f}"
        out.append(f"{case:9} {summarize(leads)}{delta}")
    return "\n".join(out)


# -- build / run / table CLI -------------------------------------------------------------

def build_case(case: Case, out: Path) -> None:
    d = out / case.name
    d.mkdir(parents=True, exist_ok=True)
    lp.assemble(launch.kernel_asm(), d / "core.o", "k_core")
    (d / "aie.mlir").write_text(design_mlir(case))
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")


def select(out: Path, names) -> list[Case]:
    cases = [case_from_name(n) for n in names] if names else default_cases()
    return [c for c in cases if (out / c.name).is_dir()]


def cmd_build(args) -> int:
    out = Path(args.out_dir).resolve()
    for case in ([case_from_name(n) for n in args.cases] if args.cases else default_cases()):
        build_case(case, out)
        print("built", case.name, flush=True)
    return 0


def runner_command(d: Path, r: Path, case: Case) -> list[str]:
    return [str(lp.RUNNER), "--xclbin", str(d / "aie.xclbin"), "--instr", str(d / "insts.bin"),
            "--trace-out", str(r / "second.bin"), "--trace-size", str(case.second_bytes),
            "--trace-buf-idx", str(SECOND_ARG), "--output", str(r / "out.bin"), *lp.runner_extra_args()]


def run_case(out: Path, case: Case, tag: str, backend: str) -> bool:
    d = out / case.name
    r = d / f"{tag}.{backend}"
    r.mkdir(parents=True, exist_ok=True)
    lockfile = None
    if backend == "hw":
        lockfile = open(lp.HW_LOCK, "w")
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{case.name}: waiting for {lp.HW_LOCK} (do not wrap this tool in flock)", file=sys.stderr, flush=True)
            fcntl.flock(lockfile, fcntl.LOCK_EX)
    try:
        p = subprocess.run(runner_command(d, r, case), env=ps.run_env(backend), capture_output=True, text=True,
                           timeout=120 if backend == "hw" else 1800)
    except subprocess.TimeoutExpired:
        print(f"{case.name} {backend}: runner timed out")
        return False
    finally:
        if lockfile:
            lockfile.close()
    (r / "runner.log").write_text(p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        print(f"{case.name} {backend}: runner failed rc={p.returncode}")
        return False
    fault = next((ln for ln in p.stderr.splitlines() if ln.startswith(lp.EMU_HOST_FAULT_MARKER)), None)
    if backend != "hw" and fault:
        print(f"{case.name} {backend}: shim DMA access outside every host buffer: {fault}")
        return False
    return True


def cmd_run(args) -> int:
    """Stops at the first run that does not complete; never retries."""
    out = Path(args.out_dir).resolve()
    for case in select(out, args.cases):
        if not run_case(out, case, args.tag, args.backend):
            print(f"{case.name}: run did not complete; stop")
            return 1
        print(f"{case.name}: ok", flush=True)
    return 0


def cmd_table(args) -> int:
    out = Path(args.out_dir).resolve()
    rows, absent = [], {}
    for case in select(out, args.cases):
        for run in args.runs:
            f = out / case.name / run / "out.bin"
            if not f.exists():
                absent.setdefault(run, []).append(case.name)
                continue
            rows.append((case.name, run, result_of(drain_words(f.read_bytes(), case))))
    print(format_table(rows))
    for run, names in absent.items():
        print(f"not in {run}: {' '.join(names)}")
    return 0


def cmd_bounds(args) -> int:
    """Device-free pre-flight of built cases: the runner's own sizing of the host buffers must equal what the
    case needs and every shim DMA task must lie inside its buffer. Exit status 1 if any case fails."""
    out = Path(args.out_dir).resolve()
    bad = 0
    for case in ([case_from_name(n) for n in args.cases] if args.cases else default_cases()):
        insts = out / case.name / "insts.bin"
        if not insts.exists():
            print(f"{case.name}: NOT BUILT")
            bad += 1
            continue
        want, got = expected_arg_sizes(case), runner_arg_sizes(insts)
        rc, report = bounds_lines(insts, allocated_arg_sizes(case))
        ok = got == want and rc == 0
        bad += not ok
        print(f"{case.name}: {'ok' if ok else 'FAIL'} allocated {allocated_arg_sizes(case)}"
              + ("" if got == want else f" (runner sizes {got}, want {want})"))
        print("".join(f"    {ln}\n" for ln in report.splitlines()), end="")
    print("ALL OK" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


def cmd_cases(args) -> int:
    for c in default_cases():
        print(f"{c.name:9} initvals={int(c.initvals)} bigbuf={int(c.bigbuf)} prefix={int(c.prefix)} "
              f"writes_ahead={c.write_ops} out_bytes={c.out_words * WORD} second_bytes={c.second_bytes}")
        print(f"          ends because: {TERMINATION[c.name]}")
    return 0


def cmd_session(args) -> int:
    for name, tag in session_order():
        print(name, tag)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cases")
    b = sub.add_parser("build")
    b.add_argument("out_dir")
    b.add_argument("--cases", nargs="*", default=[])
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("tag")
    r.add_argument("--backend", choices=("hw", "emu"), required=True)
    r.add_argument("--cases", nargs="*", default=[])
    t = sub.add_parser("table")
    t.add_argument("out_dir")
    t.add_argument("runs", nargs="+", help="run directory names, e.g. h1.hw r1.emu")
    t.add_argument("--cases", nargs="*", default=[])
    bd = sub.add_parser("bounds")
    bd.add_argument("out_dir")
    bd.add_argument("--cases", nargs="*", default=[])
    sub.add_parser("session")
    args = ap.parse_args(argv)
    return {"cases": cmd_cases, "build": cmd_build, "run": cmd_run, "table": cmd_table, "bounds": cmd_bounds,
            "session": cmd_session}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
