#!/usr/bin/env python3
"""Tile-DMA BD acquires with a change >= 0 (`aie.use_lock(%l, Acquire, c)` in a BD).

The harness topologies have no hand-written BDs, so this probe writes its own design per
case and builds and runs it with the harness's tools (llvm-mc, aiecc, the shared
test_host). One column; nothing here can hang the NPU under any candidate rule.

Design (tile (0,2) holds every lock and buffer):
  MM2S 0 -> S2MM 0 of the same tile (a local loopback, so nothing waits on the host):
    MM2S 0 bd: acquire A with change c (equal form) -> send `buf` -> release D +1
    S2MM 0 bd: acquire E >= 1 (init 1)              -> fill `copy` -> release F +1
  MM2S 1 -> shim: acquire C >= 1 -> send `stamps` -> release D +1 (the host output).
  core (0,2): writes a step marker into buf[0] before each of its ladder steps (one
    release of A +1 every `step_iters` loop iterations), so the marker that reaches
    `copy` is the step at which the DMA's acquire was granted. It then waits for the
    loopback (acquire F) and probes A's count with a greater-or-equal acquire of X, the
    count A holds if the DMA left it unchanged. It stores its counter stamps and the
    marker into `stamps` and releases C.
  core (0,3): a rescuer. Long after everything else it releases A by the amount a
    decrementing DMA acquire would have taken, so the probe completes either way; when
    the probe waits for it, the DMA acquire decremented the count.

Output (16 words): 0.. = counter stamps (start, before each ladder step, after the
loopback, after the probe), 15 = the marker.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_cascade_probe as base                                    # noqa: E402
from gen_lock_stall_probe import TS_PAIRS, _frame, _ts  # noqa: E402

LOCK_A, LOCK_F, LOCK_C, LOCK_D, LOCK_E = 14, 13, 12, 11, 10
OWN = 48                           # raw ID base of the core's own memory module
WORDS = 16
N = "  nop"


@dataclass(frozen=True)
class DmaCase:
    init: int                      # A's initial count
    change: int                    # bd0's acquire change (>= 0: the equal form)
    steps: int                     # core releases of A +1, one per step


CASES: dict[str, DmaCase] = {
    "c1v1": DmaCase(1, 1, 2),      # count == change at once
    "c2v1": DmaCase(1, 2, 3),      # count reaches the change at step 1
    "c0v0": DmaCase(0, 0, 2),      # change 0 on a free lock
}


def grant_step(case: DmaCase, rule: str) -> int:
    """Ladder step at which bd0's acquire is granted (0: at once).

    eq     count == change (silicon's host Lock_Request rule)
    eqdec  the EMU DMA path: count == change for change > 0, count >= 1 for change 0
    """
    for step in range(case.steps + 1):
        count = case.init + step
        if rule == "eq" and count == case.change:
            return step
        if rule == "eqdec" and (count == case.change if case.change > 0 else count >= 1):
            return step
    raise ValueError(f"bd0 never granted under {rule}")


def taken(case: DmaCase, rule: str) -> int:
    """What bd0's acquire subtracts from A under `rule`."""
    if rule == "eq":
        return 0
    return case.change if case.change > 0 else 1


def probe_value(case: DmaCase) -> int:
    """A's count after the ladder if the DMA left it unchanged."""
    return case.init + case.steps


def core_asm(case: DmaCase, step_iters: int) -> str:
    """p0 = buf, p1 = copy, p2 = stamps."""
    if case.steps + 3 > len(TS_PAIRS):
        raise ValueError("too many steps")
    x = probe_value(case)
    body = ["  mov r6, #0", "  mov r1, #1", "  mov r2, #-1", f"  mov r3, #{-x}", *[N] * 2,
            "  st r6, [p0, #0]", *[N] * 8, _ts(0)]
    for k in range(1, case.steps + 1):
        body += base.delay_loop(step_iters, f".Lstep{k}")
        body += [f"  mov r6, #{k}", *[N] * 2, "  st r6, [p0, #0]", *[N] * 8, _ts(k),
                 f"  rel #{OWN + LOCK_A}, r1"]
    n = case.steps + 1
    body += [f"  acq #{OWN + LOCK_F}, r2", _ts(n), f"  acq #{OWN + LOCK_A}, r3", _ts(n + 1), *[N] * 8]
    body += [f"  st r{TS_PAIRS[i][1]}, [p2, #{4 * i}]" for i in range(n + 2)]
    body += ["  lda r7, [p1, #0]", *[N] * 8, f"  st r7, [p2, #{4 * (WORDS - 1)}]"]
    body += [*[N] * 8, f"  rel #{OWN + LOCK_C}, r1"]
    return _frame(body)


def rescuer_asm(case: DmaCase, step_iters: int) -> str:
    amount = taken(case, "eqdec")
    body = [f"  mov r1, #{amount}", *[N] * 2,
            *base.delay_loop(step_iters * (case.steps + 4), ".Lrescue"),
            f"  rel #{LOCK_A}, r1", *[N] * 8]
    return _frame(body)


def design_mlir(case: DmaCase) -> str:
    m = f"memref<{WORDS}xi32>"

    def bd(acq_lock: str, action: str, acq: int, buf: str, rel_lock: str, nxt: str, tag: str) -> str:
        return f"""      %a{tag} = arith.constant {acq} : i32
      aie.use_lock({acq_lock}, {action}, %a{tag})
      aie.dma_bd({buf} : {m})
      %r{tag} = arith.constant 1 : i32
      aie.use_lock({rel_lock}, Release, %r{tag})
      aie.next_bd {nxt}
"""
    return f"""module {{
  aie.device(npu1) {{
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %buf = aie.buffer(%t02) {{sym_name = "buf"}} : {m}
    %copy = aie.buffer(%t02) {{sym_name = "copy"}} : {m}
    %stamps = aie.buffer(%t02) {{sym_name = "stamps"}} : {m}
    %lA = aie.lock(%t02, {LOCK_A}) {{init = {case.init} : i32, sym_name = "lA"}}
    %lF = aie.lock(%t02, {LOCK_F}) {{init = 0 : i32, sym_name = "lF"}}
    %lC = aie.lock(%t02, {LOCK_C}) {{init = 0 : i32, sym_name = "lC"}}
    %lD = aie.lock(%t02, {LOCK_D}) {{init = 0 : i32, sym_name = "lD"}}
    %lE = aie.lock(%t02, {LOCK_E}) {{init = 1 : i32, sym_name = "lE"}}
    aie.flow(%t02, DMA : 0, %t02, DMA : 0)
    aie.flow(%t02, DMA : 1, %t00, DMA : 0)
    func.func private @k_core({m}, {m}, {m}) attributes {{link_with = "core.o"}}
    func.func private @k_rescuer() attributes {{link_with = "rescuer.o"}}
    aie.core(%t02) {{
      func.call @k_core(%buf, %copy, %stamps) : ({m}, {m}, {m}) -> ()
      aie.end
    }}
    aie.core(%t03) {{
      func.call @k_rescuer() : () -> ()
      aie.end
    }}
    %mem = aie.mem(%t02) {{
      %s0 = aie.dma_start(MM2S, 0, ^send, ^recv_start)
    ^send:
{bd("%lA", "Acquire", case.change, "%buf", "%lD", "^end", "0")}    ^recv_start:
      %s1 = aie.dma_start(S2MM, 0, ^recv, ^out_start)
    ^recv:
{bd("%lE", "AcquireGreaterEqual", 1, "%copy", "%lF", "^end", "1")}    ^out_start:
      %s2 = aie.dma_start(MM2S, 1, ^out, ^end)
    ^out:
{bd("%lC", "AcquireGreaterEqual", 1, "%stamps", "%lD", "^end", "2")}    ^end:
      aie.end
    }}
    aie.shim_dma_allocation @out(%t00, S2MM, 0)
    aie.runtime_sequence(%in : memref<1xi32>, %b : memref<1xi32>, %o : memref<{WORDS}xi32>) {{
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %cn = arith.constant {WORDS} : i64
      aiex.npu.dma_memcpy_nd(%o[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %cn][%c0, %c0, %c0, %c1]) {{metadata = @out, id = 0 : i64, issue_token = true}} : memref<{WORDS}xi32>
      aiex.npu.dma_wait {{symbol = @out}}
    }}
  }}
}}
"""


# -- build / run / analyze --------------------------------------------------------

def cmd_generate(args) -> int:
    out = Path(args.out_dir)
    for name in args.cases:
        case = CASES[name]
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "aie.mlir").write_text(design_mlir(case))
        (d / "core.s").write_text(core_asm(case, args.step_iters))
        (d / "rescuer.s").write_text(rescuer_asm(case, args.step_iters))
        meta = {"case": name, "init": case.init, "change": case.change, "steps": case.steps,
                "step_iters": args.step_iters, "probe_value": probe_value(case),
                "grant_step": {r: grant_step(case, r) for r in ("eq", "eqdec")},
                "taken": {r: taken(case, r) for r in ("eq", "eqdec")}}
        (d / "case.json").write_text(json.dumps(meta, indent=1) + "\n")
    return 0


def cmd_build(args) -> int:
    from harness.build import _peano_tool, aiecc_command, aiecc_path
    for d in sorted(p for p in Path(args.out_dir).iterdir() if (p / "aie.mlir").exists()):
        for k in ("core", "rescuer"):
            subprocess.run([_peano_tool("llvm-mc"), "--triple=aie2", "--filetype=obj",
                            "-o", str(d / f"{k}.o"), str(d / f"{k}.s")], check=True)
            subprocess.run([_peano_tool("llvm-objcopy"), "--redefine-sym", f"test_kernel=k_{k}",
                            str(d / f"{k}.o")], check=True)
        r = subprocess.run(aiecc_command("peano", aiecc_path()), cwd=d, capture_output=True, text=True)
        (d / "aiecc.log").write_text(r.stdout + r.stderr)
        print(f"{d.name}: aiecc exit {r.returncode}")
    return 0


def cmd_run(args) -> int:
    from harness.run import KERNEL_NAME, _env, build_ffi, build_test_host
    root = Path(args.out_dir)
    host = build_test_host(root)
    if args.backend == "emu":
        build_ffi("debug")
    env = _env(args.backend, "debug")
    for d in sorted(p for p in root.iterdir() if (p / "aie.xclbin").exists()):
        rd = d / "runs" / args.tag
        if rd.exists():
            raise SystemExit(f"{rd} exists; pick another tag")
        rd.mkdir(parents=True)
        (rd / "in.bin").write_bytes(bytes(4))
        cmd = [str(host), "-x", str(d / "aie.xclbin"), "-k", KERNEL_NAME, "-i", str(d / "insts.bin"),
               "--in-size", "4", "--out-size", str(4 * WORDS), "--in-file", str(rd / "in.bin"),
               "--out-file", str(rd / "out.bin")]
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                               timeout=60 if args.backend == "hw" else 1800)
            rc, text = r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            rc, text = -1, "TIMEOUT"
        (rd / "log.txt").write_text(text)
        print(f"{d.name}: rc {rc}")
        if rc != 0:
            return 1
    return 0


def analyze_out(meta: dict, out: bytes, step_cycles: float) -> dict:
    w = struct.unpack_from(f"<{WORDS}I", out)
    n = meta["steps"] + 3
    ts = w[:n]
    probe_wait = (ts[-1] - ts[-2]) & 0xFFFFFFFF
    loopback_done = (ts[-2] - ts[0]) & 0xFFFFFFFF
    grant = w[WORDS - 1]
    rules = [r for r in ("eq", "eqdec")
             if meta["grant_step"][r] == grant and (meta["taken"][r] == 0) == (probe_wait < step_cycles)]
    return {"marker": grant, "loopback_done_after": loopback_done, "probe_wait": probe_wait,
            "stamps": [(t - ts[0]) & 0xFFFFFFFF for t in ts], "match": rules}


def cmd_analyze(args) -> int:
    for d in sorted(p for p in Path(args.out_dir).iterdir() if (p / "case.json").exists()):
        meta = json.loads((d / "case.json").read_text())
        out = d / "runs" / args.tag / "out.bin"
        if not out.exists():
            print(f"{d.name:6s} (no output)")
            continue
        r = analyze_out(meta, out.read_bytes(), 10.0 * meta["step_iters"])
        print(f"{d.name:6s} marker={r['marker']} stamps={r['stamps']} "
              f"probe_wait={r['probe_wait']} match={r['match']} "
              f"predicted grant={meta['grant_step']} taken={meta['taken']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("out_dir")
    g.add_argument("--cases", nargs="+", default=list(CASES))
    g.add_argument("--step-iters", type=int, default=400)
    b = sub.add_parser("build")
    b.add_argument("out_dir")
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("tag")
    r.add_argument("--backend", choices=("hw", "emu"), required=True)
    a = sub.add_parser("analyze")
    a.add_argument("out_dir")
    a.add_argument("tag")
    args = ap.parse_args(argv)
    return {"generate": cmd_generate, "build": cmd_build, "run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
