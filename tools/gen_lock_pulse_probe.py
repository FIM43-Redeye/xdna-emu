#!/usr/bin/env python3
"""Core LOCK_STALL trace probes: the trace of a few lock requests, silicon against emulator.

Where `gen_lock_stall_probe.py` times lock requests with counter reads, this one captures
what the core trace unit reports: one consumer core (0,2) runs a short window of lock
requests between two `event` markers, an optional producer core (0,3) releases a lock it
shares with the consumer late, and the consumer's trace (INSTR_EVENT_0/1, LOCK_STALL,
INSTR_LOCK_ACQUIRE_REQ, INSTR_LOCK_RELEASE_REQ, event-time mode) is decoded to
`events.json` per backend. The probe never reads the cycle counter, so the trace is the
only observation and the window's code is exactly the listed bundles.

    python3 tools/gen_lock_pulse_probe.py build OUT_DIR [--cases NAME ...]
    python3 tools/gen_lock_pulse_probe.py run OUT_DIR TAG --backend hw|emu [--only NAME ...]
    python3 tools/gen_lock_pulse_probe.py show OUT_DIR TAG [--only NAME ...]

`run --backend hw` takes the NPU lock (`build/experiments/.npu-hw.lock`). The kernels
preserve r16/r17 (the aiecc main keeps them live; clobbering them hangs the core).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

TRACE_EVENTS = "INSTR_LOCK_ACQUIRE_REQ,INSTR_LOCK_RELEASE_REQ,LOCK_STALL"
RUNNER = REPO.parent.parent / "bridge-runner" / "build" / "bridge-trace-runner"
HW_LOCK = REPO.parent.parent / "build" / "experiments" / ".npu-hw.lock"
N = "  nop"
OWN, SOUTH = 48, 0          # raw lock id bases: the consumer's own tile, the producer's south quadrant
A, B = 14, 15               # the two locks of the consumer tile the probes use
SPACING_ITERS = 400         # producer wait before its release: about 10 cycles per iteration
# After the window the consumer keeps running so the periodic PERF_CNT_2 events fill the
# trace unit's last packet: it only leaves the tile when full or on the stop event, which
# this short design never raises (a window-only trace came back all zeros on silicon).
TAIL_ITERS = 1500
# The cores start at their enable writes, long before the runtime sequence raises the trace
# start event, so the consumer first acquires an input buffer the sequence sends after the
# trace is armed; the window follows after a short settle.
HEAD_ITERS = 100


def nops(n: int) -> list[str]:
    return [N] * n


def delay_loop(iters: int, label: str) -> list[str]:
    # see gen_cascade_probe.delay_loop: the label sits on a fetch line of its own
    return [f"  movxm r0, #{iters}", *nops(2), ".p2align 4", f"{label}:",
            "  add r0, r0, #-1", *nops(2), f"  jnz r0, #{label}", *nops(5)]


def frame(body: list[str]) -> str:
    """A returning leaf function saving r16-r31 (the aiecc main keeps r16/r17 live)."""
    pro = [".text", ".globl test_kernel", "test_kernel:", "  paddb [sp], #64"]
    pro += [f"  st r{r}, [sp, #{-4 * (r - 15)}]" for r in range(16, 32)]
    epi = [f"  lda r{r}, [sp, #{-4 * (r - 15)}]" for r in range(16, 32)]
    epi += [*nops(7), "  paddb [sp], #-64", "  ret lr", *nops(5)]
    return "\n".join(pro + body + epi) + "\n"


def bundles(spec: dict[int, str], last: int) -> list[str]:
    """Bundles 0..last, `spec[i]` the slots of bundle i (nop where absent)."""
    return [("  " + spec[i]) if i in spec else N for i in range(last + 1)]


@dataclass(frozen=True)
class Case:
    name: str
    window: list[str]                 # consumer bundles between the markers
    inits: dict[int, int] = field(default_factory=dict)       # lock id -> initial count
    producer_release: int | None = None   # lock the producer releases after SPACING_ITERS
    end_gap: int = 12                 # nops between the window and the closing marker

    def consumer_asm(self) -> str:
        setup = ["  mov r2, #1", "  mov r3, #-1", *nops(2)]
        # markers: event 0 opens the window, event 1 closes it; padding keeps the window's
        # bundles away from the setup and the epilogue
        return frame([*setup, *delay_loop(HEAD_ITERS, ".Lhead"), *nops(8), "  event #0", *self.window, *nops(self.end_gap), "  event #1", *nops(8),
                       *delay_loop(TAIL_ITERS, ".Ltail")])

    def producer_asm(self) -> str | None:
        if self.producer_release is None:
            return None
        body = ["  mov r1, #1", *nops(2), *delay_loop(SPACING_ITERS, ".Lprod"),
                f"  rel #{SOUTH + self.producer_release}, r1", *nops(8)]
        return frame(body)


def req(kind: str, lock: int, cond: bool = False) -> str:
    reg = {"acq": "r3", "rel": "r2"}[kind]
    return f"{kind}{'.cond' if cond else ''} #{OWN + lock}, {reg}{', r26' if cond else ''}"


def single(name: str, kind: str, init: int) -> Case:
    return Case(name, bundles({0: req(kind, A)}, 0), {A: init})


def pair(name: str, k1: str, k2: str, d: int, inits: dict[int, int],
         release: int | None = None) -> Case:
    return Case(name, bundles({0: req(k1, A), d: req(k2, B)}, d), inits, release)


def cases() -> dict[str, Case]:
    out: dict[str, Case] = {}

    def add(c: Case) -> None:
        out[c.name] = c

    # EMU cycle accounting of the markers themselves: counter reads around `event`
    add(Case("cnt_event", ["  mov r17:r16, cntr", "  event #0", "  mov r19:r18, cntr", "  event #1",
                           "  mov r21:r20, cntr", "  nop", "  mov r23:r22, cntr", *nops(3),
                           "  st r16, [p0, #0]", "  st r18, [p0, #4]", "  st r20, [p0, #8]",
                           "  st r22, [p0, #12]"]))
    add(Case("none", [N]))
    for gap in (20, 100, 300, 1500):
        add(Case(f"none_g{gap}", [N], end_gap=gap))
    for gap in range(0, 7):
        add(Case(f"none_g{gap}", [N], end_gap=gap))
        add(Case(f"acq_g{gap}", bundles({0: req("acq", A)}, 0), {A: 1}, end_gap=gap))
        add(Case(f"rel_g{gap}", bundles({0: req("rel", A)}, 0), {A: 0}, end_gap=gap))
    add(single("acq_free", "acq", 1))
    add(single("rel_free", "rel", 0))
    add(Case("acq_x2", bundles({0: req("acq", A), 8: req("acq", A)}, 8), {A: 2}))
    add(Case("rel_x2", bundles({0: req("rel", A), 8: req("rel", A)}, 8), {A: 0}))
    add(Case("rel_acq_far", bundles({0: req("rel", A), 8: req("acq", A)}, 8), {A: 0}))
    add(Case("acq_rel_far", bundles({0: req("acq", A), 8: req("rel", A)}, 8), {A: 1}))
    for d in (1, 2, 3, 4, 5):
        add(pair(f"acqacq_d{d}", "acq", "acq", d, {A: 1, B: 1}))
        add(pair(f"relrel_d{d}", "rel", "rel", d, {A: 0, B: 0}))
        add(pair(f"relacq_d{d}", "rel", "acq", d, {A: 0, B: 1}))
    # a refused acquire granted when the producer releases, with a trailing request
    add(Case("acq_refused", bundles({0: req("acq", A)}, 0), {A: 0}, A))
    add(Case("acq_refused_rel", bundles({0: req("acq", A), 8: req("rel", B)}, 8), {A: 0, B: 0}, A))
    # conditional requests: skipped (r26 bit 0 clear) and performed
    for cond, tag in ((0, "skipped"), (1, "performed")):
        for kind, init in (("acq", 1), ("rel", 0)):
            out[f"cond_{kind}_{tag}"] = Case(
                f"cond_{kind}_{tag}",
                [f"  mov r26, #{cond}", *nops(2), "  " + req(kind, A, cond=True)], {A: init})
    return out


# -- MLIR -----------------------------------------------------------------------

def design_mlir(case: Case) -> str:
    has_prod = case.producer_release is not None
    locks = "\n".join(
        f'    %lock_0_2_{i} = aie.lock(%tile_0_2, {i}) {{init = {case.inits.get(i, 0)} : i32, sym_name = "lock_0_2_{i}"}}'
        for i in (A, B))
    prod_decl = ('    func.func private @k_producer() attributes {link_with = "producer.o"}\n'
                 if has_prod else "")
    prod_core = ("    aie.core(%tile_0_3) {\n"
                 "      %p0 = aie.objectfifo.acquire @of_pin(Consume, 1) : memref<16xi32>\n"
                 "      func.call @k_producer() : () -> ()\n"
                 "      aie.objectfifo.release @of_pin(Consume, 1)\n      aie.end\n    }\n"
                 if has_prod else "")
    pin_fifo = ("    aie.objectfifo @of_pin(%tile_0_0, {%tile_0_3}, 1 : i32) : !aie.objectfifo<memref<16xi32>>\n"
                if has_prod else "")
    pin_send = ("      aiex.npu.dma_memcpy_nd(%in[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c16][%c0, %c0, %c0, %c1]) "
                "{metadata = @of_pin, id = 2 : i64} : memref<16xi32>\n" if has_prod else "")
    tile3 = "    %tile_0_3 = aie.tile(0, 3)\n" if has_prod else ""
    return f"""module {{
  aie.device(npu1) {{
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
{tile3}
    aie.objectfifo @of_in(%tile_0_0, {{%tile_0_2}}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
{pin_fifo}    aie.objectfifo @of_log(%tile_0_2, {{%tile_0_0}}, 1 : i32) : !aie.objectfifo<memref<16xi32>>

{locks}

    func.func private @k_consumer(memref<16xi32>) attributes {{link_with = "consumer.o"}}
{prod_decl}
    aie.core(%tile_0_2) {{
      %i0 = aie.objectfifo.acquire @of_in(Consume, 1) : memref<16xi32>
      %a0 = aie.objectfifo.acquire @of_log(Produce, 1) : memref<16xi32>
      func.call @k_consumer(%a0) : (memref<16xi32>) -> ()
      aie.objectfifo.release @of_log(Produce, 1)
      aie.objectfifo.release @of_in(Consume, 1)
      aie.end
    }}
{prod_core}
    aie.runtime_sequence(%in : memref<16xi32>, %buf : memref<1xi32>, %out : memref<16xi32>) {{
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%in[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c16][%c0, %c0, %c0, %c1]) {{metadata = @of_in, id = 1 : i64}} : memref<16xi32>
{pin_send}      aiex.npu.dma_memcpy_nd(%out[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c16][%c0, %c0, %c0, %c1]) {{metadata = @of_log, id = 0 : i64, issue_token = true}} : memref<16xi32>
      aiex.npu.dma_wait {{symbol = @of_log}}
    }}
  }}
}}
"""


# -- build / run ----------------------------------------------------------------

def peano(tool: str) -> str:
    base = os.environ.get("PEANO_INSTALL_DIR")
    if base and (Path(base) / "bin" / tool).exists():
        return str(Path(base) / "bin" / tool)
    found = shutil.which(tool)
    if not found:
        raise SystemExit(f"{tool} not found")
    return found


def sh(cmd: list[str], cwd: Path | None = None, log: Path | None = None) -> None:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if log is not None:
        log.write_text(f"$ {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    if r.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def assemble(src: str, dest: Path, symbol: str) -> None:
    s = dest.with_suffix(".s")
    s.write_text(src)
    sh([peano("llvm-mc"), "--triple=aie2", "--filetype=obj", "-o", str(dest), str(s)])
    sh([peano("llvm-objcopy"), "--redefine-sym", f"test_kernel={symbol}", str(dest)])


def build_case(case: Case, out: Path) -> None:
    d = out / case.name
    d.mkdir(parents=True, exist_ok=True)
    assemble(case.consumer_asm(), d / "consumer.o", "k_consumer")
    if (p := case.producer_asm()) is not None:
        assemble(p, d / "producer.o", "k_producer")
    (d / "base.mlir").write_text(design_mlir(case))
    sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
        "--out", "aie.mlir", "--trace-mode", "event_time", "--core-sweep-events", TRACE_EVENTS,
        "--trace-config-out", "trace_config.json", "--config-test-name", case.name,
        "--config-src-mlir", "base.mlir", "--buffer-size", "65536"], cwd=d, log=d / "inject.log")
    sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
        "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"], cwd=d, log=d / "aiecc.log")


def run_case(out: Path, name: str, tag: str, backend: str) -> bool:
    d = out / name
    r = d / f"{tag}.{backend}"
    r.mkdir(parents=True, exist_ok=True)
    cmd = [str(RUNNER), "--xclbin", str(d / "aie.xclbin"), "--instr", str(d / "insts.bin"),
           "--trace-out", str(r / "trace_raw.bin"), "--trace-size", "65536",
           "--output", str(r / "out.bin")]
    env = dict(os.environ)
    if backend == "hw":
        env.pop("XDNA_EMU", None)
    else:
        env["XDNA_EMU"] = "1"
        env["XDNA_EMU_DIR"] = str(REPO)       # this checkout's libxdna_emu.so, not the main one
        env.setdefault("XDNA_EMU_RUNTIME", "debug")
    lockfile = None
    if backend == "hw":
        lockfile = open(HW_LOCK, "w")
        fcntl.flock(lockfile, fcntl.LOCK_EX)
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print(f"{name} {backend}: runner timed out")
        return False
    finally:
        if lockfile:
            lockfile.close()
    (r / "runner.log").write_text(p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        print(f"{name} {backend}: runner failed rc={p.returncode}")
        return False
    sh([sys.executable, str(REPO / "tools" / "parse-trace.py"), "--trace-bin", str(r / "trace_raw.bin"),
        "--xclbin-mlir", str(d / "input_with_addresses.mlir"),
        "--trace-mode", "event_time", "--out-events", str(r / "events.json")], log=r / "parse.log")
    return True


def core_events(path: Path, row: int = 2) -> list[dict]:
    ev = json.loads(path.read_text())["events"]
    return [e for e in ev if e["pkt_type"] == 0 and e["row"] == row]


# The cycle each decoded event happened at. The decoder's `ts` adds 1 + the frame's cycles
# field per frame. Silicon writes the idle cycles between frames there (probe: INSTR_EVENT_0
# and _1 a known 2..1500 bundles apart decode 2..1502 apart in `ts`), so `ts` is exact for
# silicon; the emulator writes the full elapsed cycles (docs/fidelity-gaps/trace-encoding.md),
# so its `ts` is one late per frame and its `soc` (= ts minus frames) is exact.
TRUE_CYCLE = {"hw": "ts", "emu": "soc"}


def levels(events: list[dict], name: str, field: str) -> list[tuple[int, int]]:
    """(start, end) cycles of each run of consecutive-cycle occurrences of `name`."""
    ts = sorted({e[field] for e in events if e["name"] == name})
    out: list[tuple[int, int]] = []
    for t in ts:
        if out and t == out[-1][1] + 1:
            out[-1] = (out[-1][0], t)
        else:
            out.append((t, t))
    return out


def summarize(path: Path, backend: str) -> str:
    field = TRUE_CYCLE[backend]
    ev = core_events(path)
    base = next((e[field] for e in ev if e["name"] == "INSTR_EVENT_0"), None)
    if base is None:
        return "no INSTR_EVENT_0"
    parts = []
    for name in ("INSTR_EVENT_0", "INSTR_LOCK_ACQUIRE_REQ", "INSTR_LOCK_RELEASE_REQ", "LOCK_STALL", "INSTR_EVENT_1"):
        iv = [(a - base, b - base) for a, b in levels(ev, name, field)]
        parts.append(f"{name}={iv}")
    return " ".join(parts)


def window_stalls(path: Path, backend: str, lo: int = -20, hi: int = 60) -> list[int]:
    """LOCK_STALL cycles within [lo, hi] of INSTR_EVENT_0: the window's own pulses, clear of
    the frame-boundary samples of a long hold."""
    field = TRUE_CYCLE[backend]
    ev = core_events(path)
    base = next((e[field] for e in ev if e["name"] == "INSTR_EVENT_0"), None)
    if base is None:
        return []
    return sorted({e[field] - base for e in ev if e["name"] == "LOCK_STALL" and lo <= e[field] - base <= hi})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out_dir")
    b.add_argument("--cases", nargs="*", default=[])
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("tag")
    r.add_argument("--backend", choices=("hw", "emu"), required=True)
    r.add_argument("--only", nargs="*", default=[])
    c = sub.add_parser("compare", help="LOCK_STALL cycles around the window, silicon against emulator")
    c.add_argument("out_dir")
    c.add_argument("hw_tag", help="first run directory, TAG.hw or TAG.emu")
    c.add_argument("emu_tag", help="second run directory")
    c.add_argument("--only", nargs="*", default=[])
    s = sub.add_parser("show")
    s.add_argument("out_dir")
    s.add_argument("hw_tag")
    s.add_argument("emu_tag")
    s.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args(argv)
    out = Path(a.out_dir).resolve()
    table = cases()
    if a.cmd == "build":
        for name in a.cases or table:
            build_case(table[name], out)
            print("built", name)
        return 0
    names = [n for n in table if (not a.only or n in a.only) and (out / n).is_dir()]
    if a.cmd == "compare":
        bad = 0
        for n in names:
            # the two runs are directory names (`TAG.hw` / `TAG.emu`); either may be silicon
            ha, hb = out / n / a.hw_tag / "events.json", out / n / a.emu_tag / "events.json"
            if not (ha.exists() and hb.exists()):
                continue
            h = window_stalls(ha, a.hw_tag.rsplit(".", 1)[1])
            e = window_stalls(hb, a.emu_tag.rsplit(".", 1)[1])
            verdict = "same" if h == e else "DIFF"
            bad += h != e
            print(f"{verdict:4} {n:20} hw {h} emu {e}")
        print(f"{bad} differing")
        return 1 if bad else 0
    if a.cmd == "run":
        ok = all([run_case(out, n, a.tag, a.backend) for n in names])
        return 0 if ok else 1
    for n in names:
        print(f"== {n}")
        for backend, tag in (("hw", a.hw_tag), ("emu", a.emu_tag)):
            f = out / n / f"{tag}.{backend}" / "events.json"
            print(f"  {backend:3} {summarize(f, backend) if f.exists() else '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
