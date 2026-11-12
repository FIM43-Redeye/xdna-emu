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

Lock-to-fetch probe (`fetch-build`, `fetch-run`, `fetch-table`; a separate design, one
case per k = 0..15). How many cycles pass between a core releasing a lock and the first
memory fetch of an MM2S buffer descriptor that was waiting on it, where the fetches of the
following granules fall, and where the core's own next accesses fall against them? The
emulator's value comes from one of two paths and no silicon run has pinned either.

Two variants (case names lfkNN and lfcNN):
  first-BD (default, lfk): the waiting BD is the first BD of MM2S 0's task, parked on its lock
    from boot. The emulator pays its first-BD start-up after the lock grant: first fetch 9 cycles
    after the release took effect, then granules g0..g5 one per cycle (banks 0,1,0,1,0,1), g6 two
    cycles later, then one per 4 cycles.
  chained (`--chained`, lfc): the task starts at boot with a short lock-free BD (4 words from
    `pre`, 0x4000, logical bank 1, which neither `src` nor the loads use) chained to the BD that
    waits on lock A, so the waiting BD is reached through next-BD chaining. The emulator charges
    a chained BD no start-up: the cycle count between the release and the first fetch is
    reported by the table (see the hardware notes).
The S2MM takes both BDs' words in one lock-free BD (68 words in the chained variant).

  tile (0,2): MM2S 0 bd (started at boot, so it is parked on its lock): acquire A (lock 0,
    init 0), send `src` (64 words, 0x400, data-memory bank 0, so granule g sits in physical
    bank g & 1), release B. The stream switch loops it back to S2MM 0 of the same tile,
    which writes `dst` (0x8000, bank 2): the S2MM never touches banks 0 and 1.
  core: wait on a go lock, settle, `event #0`, release A, k nops, then two adjacent loads
    from `ldbuf` (0x800, bank 0): offset 0 is physical bank 0, offset 16 is bank 1. For
    some k the core's load meets a DMA fetch of the same physical bank and CONFLICT_DM_BANK_0
    or _1 rises. Then it waits for B and ends.

Why it cannot hang. The cores start when the array is configured, before the trace unit is
armed, so the core first waits on a go lock that the runtime sequence sets (`aiex.set_lock`,
a command-processor write) after the trace is armed. Nothing in the design waits on a host
buffer, a host DMA or any host signal: the only things the core waits on are that sequence
write, which the sequence always reaches, and lock B, released by MM2S 0 after it moved its
64 words into an S2MM that has no lock and runs from boot. The sequence waits on nothing in
the tile: it writes the go lock, pads with command-processor writes to a scratch word in
bank 6/7 (clear of the measured banks), and ends.

Trace slots, memory module: LOCK_0_REL, DMA_MM2S_0_STALLED_LOCK, DMA_MM2S_0_FINISHED_BD,
DMA_MM2S_0_MEMORY_STARVATION, CONFLICT_DM_BANK_0, CONFLICT_DM_BANK_1, DMA_S2MM_0_FINISHED_BD;
core module: INSTR_EVENT_0, LOCK_STALL and, grounded, PERF_CNT_2 (the periodic event that
fills the last core packet, see gen_lock_pulse_probe). The memory-module lock events are
selector events (LOCK_n_REL follows Locks_Event_Selection_n), but selector 0 resets to
lock 0, so lock A is lock 0 and nothing is programmed.

    python3 tools/gen_dma_lock_probe.py fetch-build OUT_DIR [--ks 0 3 7] [--chained]
    python3 tools/gen_dma_lock_probe.py fetch-run OUT_DIR TAG --backend hw|emu [--ks ...]
    python3 tools/gen_dma_lock_probe.py fetch-table OUT_DIR RUN [RUN ...] [--ks ...]
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
import gen_lock_pulse_probe as lp                                   # noqa: E402
from gen_lock_stall_probe import LOCK_SETTLE_BUNDLES, TS_PAIRS, _frame, _ts  # noqa: E402

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
    settle = [N] * LOCK_SETTLE_BUNDLES
    body += [f"  acq #{OWN + LOCK_F}, r2", *settle, _ts(n), f"  acq #{OWN + LOCK_A}, r3", *settle, _ts(n + 1),
             *[N] * 8]
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


# -- lock-to-fetch probe ------------------------------------------------------------

FETCH_LOCK_A, FETCH_LOCK_B, FETCH_LOCK_GO = 0, 1, 2   # A is lock 0: LOCK_0_REL follows it at reset
FETCH_KS = tuple(range(16))
FETCH_WORDS = 64
FETCH_SRC_ADDR = 0x0400            # logical bank 0; granule g (16 bytes) in physical bank g & 1
FETCH_LD_ADDR = 0x0800             # logical bank 0; the core's two loads, offsets 0 and 16
FETCH_DST_ADDR = 0x8000            # logical bank 2: the loopback S2MM never meets banks 0 and 1
FETCH_PRE_ADDR = 0x4000            # chained variant: the lock-free first BD's 4 words, logical bank 1
FETCH_PRE_WORDS = 4
FETCH_SCRATCH_ADDR = 0xC000        # target of the sequence's pad writes, banks 6/7
FETCH_SETTLE_ITERS = 40            # about 10 cycles each; the DMA has been parked since boot
FETCH_TAIL_ITERS = 100             # keeps the core running so PERF_CNT_2 fills its last packet
FETCH_PAD_WRITES = 600             # command-processor writes after the go lock: outlast the window
FETCH_MEMMOD_EVENTS = ["LOCK_0_REL", "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_0_FINISHED_BD",
                       "DMA_MM2S_0_MEMORY_STARVATION", "CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1",
                       "DMA_S2MM_0_FINISHED_BD"]
FETCH_CORE_EVENTS = ["INSTR_EVENT_0", "LOCK_STALL"]
FETCH_CORE_GROUNDING = "PERF_CNT_2"
# tile (0,2) is physical column 1; trace pkt_type 0 = core module, 1 = memory module
FETCH_ROW, PT_CORE, PT_MEM = 2, 0, 1


def fetch_case_name(k: int, chained: bool = False) -> str:
    return f"{'lfc' if chained else 'lfk'}{k:02d}"


def fetch_core_asm(k: int) -> str:
    """The core of case k: go, settle, `event #0`, release A, k nops, a bank-0 load, a bank-1 load.

    r2 = +1 releases, r3 = -1 acquires (as gen_lock_pulse_probe). p0 = `ldbuf`."""
    if k not in FETCH_KS:
        raise ValueError(f"k must be 0..{FETCH_KS[-1]}")
    n = lp.nops
    body = ["  mov r2, #1", "  mov r3, #-1", *n(2),
            f"  acq #{OWN + FETCH_LOCK_GO}, r3", *n(4),
            *lp.delay_loop(FETCH_SETTLE_ITERS, ".Lsettle"), *n(8),
            "  event #0", f"  rel #{OWN + FETCH_LOCK_A}, r2", *n(k),
            "  lda r26, [p0, #0]", "  lda r28, [p0, #16]", *n(8),
            f"  acq #{OWN + FETCH_LOCK_B}, r3", *n(8),
            *lp.delay_loop(FETCH_TAIL_ITERS, ".Ltail")]
    return lp.frame(body)


def fetch_design_mlir(chained: bool = False) -> str:
    """Case-independent design: the k lives in the core program only.

    chained: MM2S 0's task starts with a lock-free BD of `pre` that chains to the lock-waiting BD."""
    w = FETCH_WORDS
    dw = w + (FETCH_PRE_WORDS if chained else 0)
    p = FETCH_PRE_WORDS
    pre_buf = (f"    %pre = aie.buffer(%t02) {{address = {FETCH_PRE_ADDR} : i32, sym_name = \"pre\", "
               f"initial_value = dense<[{', '.join(str(0x0C000000 | j) for j in range(p))}]> : tensor<{p}xi32>}} "
               f": memref<{p}xi32>\n") if chained else ""
    first = "^pre" if chained else "^send"
    pre_bd = (f"    ^pre:\n      aie.dma_bd(%pre : memref<{p}xi32>)\n      aie.next_bd ^send\n") if chained else ""
    ramp = ", ".join(str(0x0A000000 | j) for j in range(w))
    ld = ", ".join(str(0x0B000000 | j) for j in range(16))
    pad = "\n".join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32"
                    for _ in range(FETCH_PAD_WRITES))
    return f"""module {{
  aie.device(npu1_1col) {{
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %src = aie.buffer(%t02) {{address = {FETCH_SRC_ADDR} : i32, sym_name = "src", initial_value = dense<[{ramp}]> : tensor<{w}xi32>}} : memref<{w}xi32>
    %ldbuf = aie.buffer(%t02) {{address = {FETCH_LD_ADDR} : i32, sym_name = "ldbuf", initial_value = dense<[{ld}]> : tensor<16xi32>}} : memref<16xi32>
{pre_buf}    %dst = aie.buffer(%t02) {{address = {FETCH_DST_ADDR} : i32, sym_name = "dst"}} : memref<{dw}xi32>
    %scratch = aie.buffer(%t02) {{address = {FETCH_SCRATCH_ADDR} : i32, sym_name = "scratch"}} : memref<4xi32>
    %lA = aie.lock(%t02, {FETCH_LOCK_A}) {{init = 0 : i32, sym_name = "lA"}}
    %lB = aie.lock(%t02, {FETCH_LOCK_B}) {{init = 0 : i32, sym_name = "lB"}}
    %lGo = aie.lock(%t02, {FETCH_LOCK_GO}) {{init = 0 : i32, sym_name = "lGo"}}
    aie.flow(%t02, DMA : 0, %t02, DMA : 0)
    func.func private @k_core(memref<16xi32>) attributes {{link_with = "core.o"}}
    aie.core(%t02) {{
      func.call @k_core(%ldbuf) : (memref<16xi32>) -> ()
      aie.end
    }}
    %mem = aie.mem(%t02) {{
      %s0 = aie.dma_start(MM2S, 0, {first}, ^recv_start)
{pre_bd}    ^send:
      %ca = arith.constant 1 : i32
      aie.use_lock(%lA, AcquireGreaterEqual, %ca)
      aie.dma_bd(%src : memref<{w}xi32>)
      %cb = arith.constant 1 : i32
      aie.use_lock(%lB, Release, %cb)
      aie.next_bd ^end
    ^recv_start:
      %s1 = aie.dma_start(S2MM, 0, ^recv, ^end)
    ^recv:
      aie.dma_bd(%dst : memref<{dw}xi32>)
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


def fetch_inject_args() -> list[str]:
    return ["--memmod-sweep-events", ",".join(FETCH_MEMMOD_EVENTS),
            "--core-grounding", FETCH_CORE_GROUNDING,
            "--core-sweep-events", ",".join(FETCH_CORE_EVENTS)]


# -- lock-to-fetch: the table -------------------------------------------------------

def _runs(stamps) -> list[tuple[int, int]]:
    """(first, last) of each run of consecutive cycles."""
    out: list[tuple[int, int]] = []
    for t in sorted(set(stamps)):
        if out and t == out[-1][1] + 1:
            out[-1] = (out[-1][0], t)
        else:
            out.append((t, t))
    return out


def fetch_row(events) -> dict:
    """One run of tile (0,2) as numbers relative to the lock A release (memory-module timer).

    `events` are level-expanded (`level_events`): a level has an event on every cycle it is
    asserted. grant = the cycle after the last asserted cycle of the MM2S STALLED_LOCK interval
    that holds the release. core_event0 is the core's INSTR_EVENT_0 minus the release: two
    different trace timers, so only good to the cross-domain skew; core_stalls are relative to
    the core's own INSTR_EVENT_0."""
    tile = [e for e in events if e.row == FETCH_ROW]
    mem = {}
    core = {}
    for e in tile:
        (mem if e.pkt_type == PT_MEM else core).setdefault(e.name, []).append(e.ts)
    rel_stamps = sorted(mem.get("LOCK_0_REL", []))
    if not rel_stamps:
        return {"release": None}
    rel = rel_stamps[0]
    row: dict = {"release": rel, "grant": None}
    for first, last in _runs(mem.get("DMA_MM2S_0_STALLED_LOCK", [])):
        if first <= rel <= last + 1:
            row["grant"] = last + 1 - rel
    row["starvation"] = [(a - rel, b - rel) for a, b in _runs(mem.get("DMA_MM2S_0_MEMORY_STARVATION", []))
                         if b >= rel]
    conflicts = []
    for bank in ("CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1"):
        conflicts += [(bank, t - rel) for t in sorted(set(mem.get(bank, [])))]
    row["conflicts"] = sorted(conflicts, key=lambda c: (c[1], c[0]))
    row["mm2s_finished"] = [t - rel for t in sorted(set(mem.get("DMA_MM2S_0_FINISHED_BD", []))) if t >= rel]
    row["s2mm_finished"] = [t - rel for t in sorted(set(mem.get("DMA_S2MM_0_FINISHED_BD", []))) if t >= rel]
    ev0 = sorted(core.get("INSTR_EVENT_0", []))
    row["core_event0"] = ev0[0] - rel if ev0 else None
    row["core_stalls"] = ([(a - ev0[0], b - ev0[0]) for a, b in _runs(core.get("LOCK_STALL", []))
                           if a >= ev0[0] - 2 and a - ev0[0] <= 64] if ev0 else [])
    return row


def _interval(a: int, b: int) -> str:
    return f"+{a}" if a == b else f"+{a}..+{b}"


def format_fetch_table(rows) -> str:
    """rows: (k, run name, fetch_row dict). Sorted by k, then run."""
    lines = [f"{'k':>2} {'run':<8} {'release':>9}  (release: memory-module timer; every other number is cycles "
             "after it; core columns are another timer, marked core)"]
    for k, run, r in sorted(rows, key=lambda x: (x[0], x[1])):
        if r["release"] is None:
            lines.append(f"{k:>2} {run:<8} no LOCK_0_REL event")
            continue
        grant = "grant ?" if r["grant"] is None else f"grant +{r['grant']}"
        starv = "starv [" + ", ".join(_interval(a, b) for a, b in r["starvation"][:6]) + "]"
        conf = ("conflict " + " ".join(f"{b}@+{t}" for b, t in r["conflicts"])) if r["conflicts"] else "no conflict"
        ev0 = "core(xtimer) ev0 n/a" if r["core_event0"] is None else f"core(xtimer) ev0 {r['core_event0']:+d}"
        stalls = "core stall(from ev0) [" + ", ".join(_interval(a, b) for a, b in r["core_stalls"][:6]) + "]"
        fin = (f"mm2s_fin [{', '.join(f'+{t}' for t in r['mm2s_finished'])}] "
               f"s2mm_fin [{', '.join(f'+{t}' for t in r['s2mm_finished'])}]")
        lines.append(f"{k:>2} {run:<8} {r['release']:>9}  {grant}  {starv}  {conf}  {fin}  {ev0}  {stalls}")
    return "\n".join(lines)


def level_events(commands, slot_names):
    """Level-expanded events of per-tile decoded commands (a held level has an event per cycle)."""
    from trace_decoder.decode import rebuild_timeline_mode0_levels
    return rebuild_timeline_mode0_levels(commands, slot_names)


def load_level_events(run_dir: Path):
    """Level-expanded events of a run directory (trace_raw.bin; events.json supplies the slot names)."""
    import trace_decoder as td
    doc = json.loads((run_dir / "events.json").read_text())
    raw = (run_dir / "trace_raw.bin").read_bytes()
    commands = td.decode_words(list(struct.unpack(f"<{len(raw) // 4}I", raw)), mode=None)
    kind = {0: "core", 1: "mem", 2: "shim", 3: "memtile"}
    names: dict = {}
    for e in doc["events"]:
        names.setdefault(e["pkt_type"], {}).setdefault(f"{e['row']},{e['col']}", [""] * 8)[e["slot"]] = e["name"]
    for pkt, row, col in commands:
        table = names.setdefault(pkt, {}).setdefault(f"{row},{col}", [""] * 8)
        defaults = doc.get("slot_names", {}).get(kind[pkt], [""] * 8)
        for slot in range(8):
            table[slot] = table[slot] or defaults[slot]
    return level_events(commands, names)


# -- lock-to-fetch: build / run / table ---------------------------------------------

def fetch_build_case(k: int, out: Path, chained: bool = False) -> None:
    d = out / fetch_case_name(k, chained)
    d.mkdir(parents=True, exist_ok=True)
    lp.assemble(fetch_core_asm(k), d / "core.o", "k_core")
    (d / "base.mlir").write_text(fetch_design_mlir(chained))
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time", *fetch_inject_args(),
           "--trace-config-out", "trace_config.json", "--config-test-name", fetch_case_name(k, chained),
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    (d / "case.json").write_text(json.dumps({"case": fetch_case_name(k, chained), "k": k, "chained": chained}) + "\n")


def fetch_cases_in(out: Path, ks, chained: bool = False) -> list[int]:
    return [k for k in (ks or FETCH_KS) if (out / fetch_case_name(k, chained)).is_dir()]


def cmd_fetch_build(args) -> int:
    out = Path(args.out_dir).resolve()
    for k in args.ks or FETCH_KS:
        fetch_build_case(k, out, args.chained)
        print("built", fetch_case_name(k, args.chained))
    return 0


def cmd_fetch_run(args) -> int:
    out = Path(args.out_dir).resolve()
    ok = True
    for k in fetch_cases_in(out, args.ks, args.chained):
        name = fetch_case_name(k, args.chained)
        ok = lp.run_case(out, name, args.tag, args.backend, output=False) and ok
        if not ok:
            print(f"{name}: run did not complete; stop")
            return 1
    return 0


def cmd_fetch_table(args) -> int:
    out = Path(args.out_dir).resolve()
    rows = []
    for k in fetch_cases_in(out, args.ks, args.chained):
        for run in args.runs:
            run_dir = out / fetch_case_name(k, args.chained) / run
            if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
                rows.append((k, run, {"release": None}))
                continue
            rows.append((k, run, fetch_row(load_level_events(run_dir))))
    print(format_fetch_table(rows))
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
    fb = sub.add_parser("fetch-build", help="lock-to-fetch probe: build the cases (default k = 0..15)")
    fb.add_argument("out_dir")
    fb.add_argument("--ks", nargs="*", type=int, default=[])
    fb.add_argument("--chained", action="store_true", help="the chained variant (cases lfcNN)")
    fr = sub.add_parser("fetch-run", help="lock-to-fetch probe: run the built cases to TAG.BACKEND/")
    fr.add_argument("out_dir")
    fr.add_argument("tag")
    fr.add_argument("--backend", choices=("hw", "emu"), required=True)
    fr.add_argument("--ks", nargs="*", type=int, default=[])
    fr.add_argument("--chained", action="store_true", help="the chained variant (cases lfcNN)")
    ft = sub.add_parser("fetch-table", help="lock-to-fetch probe: per k and run, events relative to the release")
    ft.add_argument("out_dir")
    ft.add_argument("runs", nargs="+", help="run directory names, e.g. h1.hw e1.emu")
    ft.add_argument("--ks", nargs="*", type=int, default=[])
    ft.add_argument("--chained", action="store_true", help="the chained variant (cases lfcNN)")
    args = ap.parse_args(argv)
    return {"generate": cmd_generate, "build": cmd_build, "run": cmd_run, "analyze": cmd_analyze,
            "fetch-build": cmd_fetch_build, "fetch-run": cmd_fetch_run,
            "fetch-table": cmd_fetch_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
