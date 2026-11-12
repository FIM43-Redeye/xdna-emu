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

Single-bank cases (`--banks 0` / `--banks 1`): the core makes one load only, k nops after the release,
so one bank is probed alone and a bank-0 conflict cannot disturb a bank-1 access. Names (chained /
first-BD): bank 0 lfa / lfd, bank 1 lfb / lfe; the two-load cases stay lfc / lfk. k runs 0..64.
fetch-table prints, per platform, the DMA's per-bank map relative to the lock grant pooled over every
run given (cells: DMA access proven / proven absent / unknown) and lists contradictions.

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

Lock-synchronised variant (`--sync`, cases lsDD / lsfree / lsaDDkKK / lsbDDkKK). The chained task's pre BD
waits for a start lock that the runtime sequence sets after the trace is armed, and releases a sync lock
when it finishes; the core blocks on the sync lock, runs `event #0`, d nops, then releases lock A, on which
the 64-word BD is already (or is about to be) parked. So the release comes a controlled number of cycles after
the channel left the pre BD: d sweeps from "lock free when the BD asks" up to a long-parked channel. The table
prints per case and run: d (measured: release minus the pre BD's FINISHED_BD), the channel's STALLED_LOCK rise
after that finish, grant minus release, MM2S and S2MM finish minus grant, and conflicts relative to the grant.
`--free` adds lsfree (lock A starts at 1). Bank cases (lsa/lsb) add one core load k nops after the release.

    python3 tools/gen_dma_lock_probe.py fetch-build OUT_DIR --sync [--ds 0 4 8] [--banks none|0|1] [--ks ..] [--free]
    python3 tools/gen_dma_lock_probe.py fetch-run OUT_DIR TAG --backend hw|emu --sync [same selection]
    python3 tools/gen_dma_lock_probe.py fetch-table OUT_DIR RUN [RUN ...] --sync [same selection]

    python3 tools/gen_dma_lock_probe.py fetch-build OUT_DIR [--ks 0 3 7] [--chained] [--banks both 0 1]
    python3 tools/gen_dma_lock_probe.py fetch-run OUT_DIR TAG --backend hw|emu [--ks ...] [--chained] [--banks ...]
    python3 tools/gen_dma_lock_probe.py fetch-table OUT_DIR RUN [RUN ...] [--ks ...] [--chained] [--banks ...]
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
FETCH_KS = tuple(range(65))        # valid k: nops between the release and the access(es)
FETCH_DEFAULT_KS = tuple(range(16))  # what fetch-build makes when no --ks is given
FETCH_BANKS = ("both", "0", "1")   # both = a bank-0 load then a bank-1 load one bundle later; else one load
# case-name prefix per (chained, bank): lfk / lfc are the two-load cases; the single-bank cases get their own
FETCH_PREFIX = {(False, "both"): "lfk", (True, "both"): "lfc", (True, "0"): "lfa", (True, "1"): "lfb",
                (False, "0"): "lfd", (False, "1"): "lfe"}
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


def fetch_case_name(k: int, chained: bool = False, bank: str = "both") -> str:
    return f"{FETCH_PREFIX[(chained, bank)]}{k:02d}"


FETCH_LOADS = {"0": ["  lda r26, [p0, #0]"], "1": ["  lda r28, [p0, #16]"],
               "both": ["  lda r26, [p0, #0]", "  lda r28, [p0, #16]"]}


def fetch_core_asm(k: int, bank: str = "both") -> str:
    """The core of case k: go, settle, `event #0`, release A, k nops, then the load(s).

    bank "both": a bank-0 load and, one bundle later, a bank-1 load. "0" / "1": that one load alone,
    at the same place (k nops after the release), so the other bank is not touched and a bank-0
    conflict cannot disturb a bank-1 access. r2 = +1 releases, r3 = -1 acquires (as
    gen_lock_pulse_probe). p0 = `ldbuf`."""
    if k not in FETCH_KS:
        raise ValueError(f"k must be 0..{FETCH_KS[-1]}")
    if bank not in FETCH_LOADS:
        raise ValueError(f"bank must be one of {FETCH_BANKS}")
    n = lp.nops
    body = ["  mov r2, #1", "  mov r3, #-1", *n(2),
            f"  acq #{OWN + FETCH_LOCK_GO}, r3", *n(4),
            *lp.delay_loop(FETCH_SETTLE_ITERS, ".Lsettle"), *n(8),
            "  event #0", f"  rel #{OWN + FETCH_LOCK_A}, r2", *n(k),
            *FETCH_LOADS[bank], *n(8),
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


# -- lock-to-fetch: the per-bank map relative to the grant ---------------------------

# Core access offset from the release stamp (cycles): the first load sits at k + offset. Silicon and
# emulator differ (see the analysis notes); the table infers it from the conflicts when any exist.
DEFAULT_ACCESS_OFFSET = {"hw": 1, "emu": -2}
PLATFORM_NAMES = {"hw": "silicon", "emu": "emulator"}


def run_platform(run: str) -> str:
    """'hw' or 'emu' from a run directory name such as h3.hw."""
    return run.rsplit(".", 1)[-1]


def _entry_offsets(entry) -> list[int]:
    """Offsets implied by the conflicts of one row: the stamp of the first conflict minus where the
    first load sits relative to k. Empty when the row has no conflict."""
    _case, k, bank, _run, r = entry
    if r.get("release") is None or not r["conflicts"]:
        return []
    out = []
    first = {}
    for b, t in r["conflicts"]:
        first.setdefault(b, t)
    for b, t in first.items():
        second = bank == "both" and b == "CONFLICT_DM_BANK_1"
        if second and "CONFLICT_DM_BANK_0" in first:
            continue                    # a retried load: its stamp has moved
        if bank in ("0", "1") and b != f"CONFLICT_DM_BANK_{bank}":
            continue
        out.append(t - k - (1 if second else 0))
    return out


def infer_access_offsets(entries) -> int:
    """The release-to-first-load offset of one platform: the most common value implied by its
    conflicts, else the platform default."""
    from collections import Counter
    seen = Counter(o for e in entries for o in _entry_offsets(e))
    if seen:
        return seen.most_common(1)[0][0]
    return DEFAULT_ACCESS_OFFSET[run_platform(entries[0][3])]


def _row_accesses(k: int, bank: str, offset: int) -> list[tuple[int, int]]:
    """(bank, cycle after the release) of each load of a case."""
    if bank == "both":
        return [(0, k + offset), (1, k + offset + 1)]
    return [(int(bank), k + offset)]


def fetch_access_cells(entries) -> dict:
    """Per platform {(bank, cycle after the grant): {'dma': [labels], 'none': [labels]}}.

    A conflict event is a DMA access to that bank in that cycle. A load that ran without a conflict
    event in its cycle shows the DMA absent from that bank then. The second load of a two-load row is
    not trusted once the first conflicted (the retry moves it). Rows without a release or a grant
    say nothing."""
    by_platform: dict = {}
    for e in entries:
        by_platform.setdefault(run_platform(e[3]), []).append(e)
    cells: dict = {}
    for plat, ents in by_platform.items():
        offset = infer_access_offsets(ents)
        c = cells.setdefault(plat, {})
        for case, k, bank, run, r in ents:
            if r.get("release") is None or r.get("grant") is None:
                continue
            label = f"{case}.{run}"
            g = r["grant"]
            hit = {(int(b[-1]), t) for b, t in r["conflicts"]}
            for b, t in hit:
                c.setdefault((b, t - g), {"dma": [], "none": []})["dma"].append(label)
            b0_hit = any(b == "CONFLICT_DM_BANK_0" for b, _ in r["conflicts"])
            for i, (b, t) in enumerate(_row_accesses(k, bank, offset)):
                if (b, t) in hit or (bank == "both" and i == 1 and b0_hit):
                    continue
                c.setdefault((b, t - g), {"dma": [], "none": []})["none"].append(label)
    return cells


def _spans(values) -> str:
    """'g+2, g+4..g+6' from a set of cycles."""
    vs = sorted(values)
    if not vs:
        return "none"
    runs = []
    for v in vs:
        if runs and v == runs[-1][1] + 1:
            runs[-1][1] = v
        else:
            runs.append([v, v])
    return ", ".join(f"g{a:+d}" if a == b else f"g{a:+d}..g{b:+d}" for a, b in runs)


def format_fetch_map(entries) -> str:
    """Per platform: the DMA's per-bank access map relative to the grant, pooled over every run given,
    and the cells where a conflict and a clean access disagree."""
    cells = fetch_access_cells(entries)
    by_platform: dict = {}
    for e in entries:
        by_platform.setdefault(run_platform(e[3]), []).append(e)
    lines = []
    for plat in sorted(cells):
        c = cells[plat]
        lines.append(f"== per-bank map relative to the grant: {PLATFORM_NAMES.get(plat, plat)} "
                     f"(first-load offset from the release {infer_access_offsets(by_platform[plat]):+d}) ==")
        if not c:
            lines.append("  no data")
            continue
        lo, hi = min(g for _, g in c), max(g for _, g in c)
        lines.append("cycle  bank0            bank1       (DMA(n): n rows saw a conflict; none(n): n clean "
                     "loads; ?: no load there)")
        for g in range(lo, hi + 1):
            cols = []
            for b in (0, 1):
                cell = c.get((b, g))
                if cell is None:
                    cols.append("?")
                else:
                    cols.append(" ".join(f"{kind}({len(cell[kind])})" for kind in ("dma", "none") if cell[kind])
                                .replace("dma", "DMA"))
            lines.append(f"g{g:+3d}   {cols[0]:<16} {cols[1]}")
        for b in (0, 1):
            dma = {g for (bb, g), cell in c.items() if bb == b and cell["dma"]}
            absent = {g for (bb, g), cell in c.items() if bb == b and cell["none"] and not cell["dma"]}
            seen = {g for (bb, g) in c if bb == b}
            if not seen:
                lines.append(f"bank {b}: no data")
                continue
            unknown = {g for g in range(lo, hi + 1)} - seen
            lines.append(f"bank {b} DMA access proven: {_spans(dma)}")
            lines.append(f"bank {b} no DMA access: {_spans(absent)}")
            lines.append(f"bank {b} unknown (no load there, within g{lo:+d}..g{hi:+d}): {_spans(unknown)}")
        bad = [((b, g), cell) for (b, g), cell in sorted(c.items()) if cell["dma"] and cell["none"]]
        lines.append("CONTRADICTIONS (same cycle after the grant, same bank, one row with a conflict and one without):")
        if not bad:
            lines.append("  contradictions: none")
        for (b, g), cell in bad:
            lines.append(f"  bank {b} g{g:+d}: conflict in {', '.join(cell['dma'])}; clean in {', '.join(cell['none'])}")
        lines.append("")
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

def fetch_build_case(k: int, out: Path, chained: bool = False, bank: str = "both") -> None:
    name = fetch_case_name(k, chained, bank)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    lp.assemble(fetch_core_asm(k, bank), d / "core.o", "k_core")
    (d / "base.mlir").write_text(fetch_design_mlir(chained))
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time", *fetch_inject_args(),
           "--trace-config-out", "trace_config.json", "--config-test-name", name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    (d / "case.json").write_text(json.dumps({"case": name, "k": k, "chained": chained, "bank": bank}) + "\n")


def fetch_cases_in(out: Path, ks, chained: bool = False, bank: str = "both") -> list[int]:
    return [k for k in (ks or FETCH_KS) if (out / fetch_case_name(k, chained, bank)).is_dir()]


def cmd_fetch_build(args) -> int:
    out = Path(args.out_dir).resolve()
    for bank in args.banks:
        for k in args.ks or FETCH_DEFAULT_KS:
            fetch_build_case(k, out, args.chained, bank)
            print("built", fetch_case_name(k, args.chained, bank))
    return 0


def cmd_fetch_run(args) -> int:
    out = Path(args.out_dir).resolve()
    ok = True
    for bank in args.banks:
        for k in fetch_cases_in(out, args.ks, args.chained, bank):
            name = fetch_case_name(k, args.chained, bank)
            ok = lp.run_case(out, name, args.tag, args.backend, output=False) and ok
            if not ok:
                print(f"{name}: run did not complete; stop")
                return 1
    return 0


def cmd_fetch_table(args) -> int:
    out = Path(args.out_dir).resolve()
    entries = []
    for bank in args.banks:
        rows = []
        for k in fetch_cases_in(out, args.ks, args.chained, bank):
            name = fetch_case_name(k, args.chained, bank)
            for run in args.runs:
                run_dir = out / name / run
                if not run_dir.is_dir():
                    continue                    # this case was not run under this tag
                if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
                    rows.append((k, run, {"release": None}))
                    continue
                r = fetch_row(load_level_events(run_dir))
                rows.append((k, run, r))
                entries.append((name, k, bank, run, r))
        print(f"== bank {bank} ({FETCH_PREFIX[(args.chained, bank)]}NN) ==")
        print(format_fetch_table(rows))
        print()
    print(format_fetch_map(entries))
    return 0


# -- lock-synchronised variant (--sync): the release comes a controlled d cycles after the pre BD ------

SYNC_LOCK_A, SYNC_LOCK_SYNC, SYNC_LOCK_START, SYNC_LOCK_END = 0, 1, 2, 3   # A: the lock the 64-word BD waits on (LOCK_0_REL)
SYNC_PRE2_ADDR = 0x4040            # the second pre BD's words (--pre2), logical bank 1 like `pre`
SYNC_DEFAULT_DS = tuple(range(25)) + tuple(range(26, 41, 2))
SYNC_MAX_D = 40
SYNC_MAX_PRE2 = 40                 # words; the buffer (0x4040 + 4 * words) must stay in its logical bank
FREE_D = 20                        # the lock-free-on-request case's core delay (lock A starts at 1)
SYNC_MAIN_EVENTS = ["LOCK_0_REL", "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_0_FINISHED_BD", "DMA_S2MM_0_FINISHED_BD"]
SYNC_CONFLICT_EVENTS = ["CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1"]
SYNC_BANK_LETTER = {"0": "a", "1": "b"}


@dataclass(frozen=True)
class SyncCase:
    d: int                         # nops between the core's marker and its release of lock A
    bank: str = "none"             # "none": no core memory access; "0"/"1": one load k nops after the release
    k: int = 0
    free: bool = False             # lock A starts at 1: the BD finds it free when it asks
    pre2: int = 0                  # words of a second lock-free pre BD between the sync pre BD and the waiting
                                   # BD (0: none). It delays the moment the channel asks for lock A, which
                                   # moves the core's release from "long after" to "before" the request.

    @property
    def label(self):
        """Row label in the tables: the nominal d; "free"; or "a11k05" (bank letter, d, k) for a load case."""
        if self.free:
            return "free"
        return self.d if self.bank == "none" else self.name[2:]

    @property
    def name(self) -> str:
        stem = "lt" if self.pre2 else "ls"
        if self.free:
            return f"{stem}free"
        if self.bank == "none":
            return f"{stem}{self.d:02d}"
        return f"{stem}{SYNC_BANK_LETTER[self.bank]}{self.d:02d}k{self.k:02d}"


def sync_specs(ds, ks, banks, free: bool, pre2: int = 0) -> list[SyncCase]:
    """The cases of a --sync selection: every d for bank none; every (bank, d, k) for a load bank."""
    out = []
    for bank in banks:
        for d in ds:
            for k in ([0] if bank == "none" else ks or FETCH_DEFAULT_KS):
                c = SyncCase(d, bank, k, pre2=pre2)
                sync_validate(c)
                out.append(c)
    if free:
        out.append(SyncCase(FREE_D, free=True, pre2=pre2))
    return out


def sync_validate(c: SyncCase) -> None:
    if not 0 <= c.d <= SYNC_MAX_D:
        raise ValueError(f"d must be 0..{SYNC_MAX_D}")
    if not 0 <= c.pre2 <= SYNC_MAX_PRE2:
        raise ValueError(f"pre2 must be 0..{SYNC_MAX_PRE2} words")
    if c.bank not in ("none", "0", "1"):
        raise ValueError("bank must be none, 0 or 1")
    if c.bank != "none" and c.k not in FETCH_KS:
        raise ValueError(f"k must be 0..{FETCH_KS[-1]}")


def sync_core_asm(c: SyncCase) -> str:
    """Block on the SYNC lock (released by the pre BD when it finishes), `event #0`, d nops, release A,
    then optionally one load k nops later. r2 = +1 releases, r3 = -1 acquires. p0 = `ldbuf`.
    The core waits on nothing else, so it cannot hang unless the pre BD never finishes."""
    sync_validate(c)
    n = lp.nops
    loads = [*n(c.k), *FETCH_LOADS[c.bank]] if c.bank != "none" else []
    body = ["  mov r2, #1", "  mov r3, #-1", *n(2),
            f"  acq #{OWN + SYNC_LOCK_SYNC}, r3", *n(LOCK_SETTLE_BUNDLES),
            "  event #0", *n(c.d), f"  rel #{OWN + SYNC_LOCK_A}, r2", *loads, *n(8),
            *lp.delay_loop(FETCH_TAIL_ITERS, ".Ltail")]
    return lp.frame(body)


def sync_design_mlir(c: SyncCase) -> str:
    """MM2S 0: pre BD (acquire START, 4 words of `pre`, release SYNC) chained to the BD that waits on A.

    The runtime sequence's only action is setting START, after the trace is armed, so the pre BD runs
    (and releases SYNC for the core) inside the trace window. The S2MM takes both BDs' words."""
    w, p = FETCH_WORDS, FETCH_PRE_WORDS
    q = c.pre2
    dw = w + p + q
    ramp = ", ".join(str(0x0A000000 | j) for j in range(w))
    ld = ", ".join(str(0x0B000000 | j) for j in range(16))
    pre_init = ", ".join(str(0x0C000000 | j) for j in range(p))
    pad = "\n".join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32"
                    for _ in range(FETCH_PAD_WRITES))
    pre2_init = ", ".join(str(0x0D000000 | j) for j in range(q))
    pre2_buf = (f"    %pre2 = aie.buffer(%t02) {{address = {SYNC_PRE2_ADDR} : i32, sym_name = \"pre2\", "
                f"initial_value = dense<[{pre2_init}]> : tensor<{q}xi32>}} : memref<{q}xi32>\n") if q else ""
    pre2_bd = (f"    ^pre2:\n      aie.dma_bd(%pre2 : memref<{q}xi32>)\n      aie.next_bd ^send\n") if q else ""
    return f"""module {{
  aie.device(npu1_1col) {{
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %src = aie.buffer(%t02) {{address = {FETCH_SRC_ADDR} : i32, sym_name = "src", initial_value = dense<[{ramp}]> : tensor<{w}xi32>}} : memref<{w}xi32>
    %ldbuf = aie.buffer(%t02) {{address = {FETCH_LD_ADDR} : i32, sym_name = "ldbuf", initial_value = dense<[{ld}]> : tensor<16xi32>}} : memref<16xi32>
    %pre = aie.buffer(%t02) {{address = {FETCH_PRE_ADDR} : i32, sym_name = "pre", initial_value = dense<[{pre_init}]> : tensor<{p}xi32>}} : memref<{p}xi32>
{pre2_buf}    %dst = aie.buffer(%t02) {{address = {FETCH_DST_ADDR} : i32, sym_name = "dst"}} : memref<{dw}xi32>
    %scratch = aie.buffer(%t02) {{address = {FETCH_SCRATCH_ADDR} : i32, sym_name = "scratch"}} : memref<4xi32>
    %lA = aie.lock(%t02, {SYNC_LOCK_A}) {{init = {1 if c.free else 0} : i32, sym_name = "lA"}}
    %lSync = aie.lock(%t02, {SYNC_LOCK_SYNC}) {{init = 0 : i32, sym_name = "lSync"}}
    %lStart = aie.lock(%t02, {SYNC_LOCK_START}) {{init = 0 : i32, sym_name = "lStart"}}
    %lEnd = aie.lock(%t02, {SYNC_LOCK_END}) {{init = 0 : i32, sym_name = "lEnd"}}
    aie.flow(%t02, DMA : 0, %t02, DMA : 0)
    func.func private @k_core(memref<16xi32>) attributes {{link_with = "core.o"}}
    aie.core(%t02) {{
      func.call @k_core(%ldbuf) : (memref<16xi32>) -> ()
      aie.end
    }}
    %mem = aie.mem(%t02) {{
      %s0 = aie.dma_start(MM2S, 0, ^pre, ^recv_start)
    ^pre:
      %cs = arith.constant 1 : i32
      aie.use_lock(%lStart, AcquireGreaterEqual, %cs)
      aie.dma_bd(%pre : memref<{p}xi32>)
      %cy = arith.constant 1 : i32
      aie.use_lock(%lSync, Release, %cy)
      aie.next_bd {"^pre2" if q else "^send"}
{pre2_bd}    ^send:
      %ca = arith.constant 1 : i32
      aie.use_lock(%lA, AcquireGreaterEqual, %ca)
      aie.dma_bd(%src : memref<{w}xi32>)
      %ce = arith.constant 1 : i32
      aie.use_lock(%lEnd, Release, %ce)
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
      aiex.set_lock(%lStart, %c_one)
{pad}
    }}
  }}
}}
"""


def sync_inject_args(bank: str) -> list[str]:
    events = SYNC_MAIN_EVENTS + (SYNC_CONFLICT_EVENTS if bank != "none" else [])
    return ["--memmod-sweep-events", ",".join(events),
            "--core-grounding", FETCH_CORE_GROUNDING,
            "--core-sweep-events", ",".join(FETCH_CORE_EVENTS)]


def sync_row(events) -> dict:
    """One run as numbers around the lock-A release and the lock grant (memory-module timer).

    The MM2S FINISHED_BD events are the pre BDs' and the 64-word BD's (last); "pre_fin" is the one just
    before it (the last pre BD, whose finish is when the channel moves to the waiting BD) and d is
    measured from it; d1 is measured from the first (the sync BD, which released the core). The channel's
    STALLED_LOCK interval of the waiting BD is the one that holds the release (it may rise a cycle
    before the pre BD's finish stamp, so park can be -1; the earlier interval is the pre BD's wait for
    the start lock). grant = the cycle after it falls; no interval holding the release means the lock
    was free when the BD asked."""
    mem = {}
    core = {}
    for e in events:
        if e.row == FETCH_ROW:
            (mem if e.pkt_type == PT_MEM else core).setdefault(e.name, []).append(e.ts)
    rels = sorted(mem.get("LOCK_0_REL", []))
    if not rels:
        return {"release": None}
    rel = rels[0]
    fins = sorted(set(mem.get("DMA_MM2S_0_FINISHED_BD", [])))
    r: dict = {"release": rel, "pre_fin": None, "d": None, "park": None, "grant": None,
               "mm2s_after_grant": None, "s2mm_after_grant": None, "fin_after_pre": None,
               "fin_after_release": None, "d1": None, "conflicts": [], "core_event0": None}
    if len(fins) < 2:
        return r
    big = fins[-1]
    pre = fins[-2]
    r["pre_fin"], r["d"], r["d1"] = pre, rel - pre, rel - fins[0]
    r["fin_after_pre"] = big - pre
    stalls = [run for run in _runs(mem.get("DMA_MM2S_0_STALLED_LOCK", [])) if run[0] <= rel <= run[1] + 1]
    anchor = rel
    if stalls:
        first, last = stalls[0]
        r["park"] = first - pre
        r["grant"] = last + 1 - rel
        anchor = last + 1
        r["mm2s_after_grant"] = big - anchor
    r["fin_after_release"] = big - rel
    s2 = [t for t in sorted(set(mem.get("DMA_S2MM_0_FINISHED_BD", []))) if t > pre]
    if s2 and stalls:
        r["s2mm_after_grant"] = s2[-1] - anchor
    conflicts = []
    for bank in ("CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1"):
        conflicts += [(bank, t - anchor) for t in sorted(set(mem.get(bank, []))) if t >= rel]
    r["conflicts"] = sorted(conflicts, key=lambda c: (c[1], c[0]))
    ev0 = sorted(core.get("INSTR_EVENT_0", []))
    r["core_event0"] = ev0[0] - rel if ev0 else None
    return r


def _label_key(label) -> tuple:
    """Sort key of a row label: a nominal d (int) first, then names such as "free"."""
    return (0, label, "") if isinstance(label, int) else (1, 0, label)


def _num(v, fmt="+d") -> str:
    return "?" if v is None else format(v, fmt)


def format_sync_table(rows) -> str:
    """rows: (nominal d, run, sync_row dict), sorted by d then run. Cycle columns: d_meas = release minus
    the last pre BD's finish (d1: the first pre BD's); park = STALLED_LOCK rise after the pre finish; grant = after the release;
    mm2s-g / s2mm-g = finish after the grant; fin-pre = the 64-word BD's finish after the pre finish;
    conflicts are cycles after the grant."""
    lines = [f"{'d':>4} {'run':<8} {'d_meas':>6} {'d1':>4}  park  grant  mm2s-g  s2mm-g  fin-pre  conflicts"]
    for d, run, r in sorted(rows, key=lambda x: (_label_key(x[0]), x[1])):
        d = f"{d:>4}"
        if r["release"] is None:
            lines.append(f"{d} {run:<8} no LOCK_0_REL event")
            continue
        if r["pre_fin"] is None:
            lines.append(f"{d} {run:<8} release seen, fewer than two MM2S FINISHED_BD events")
            continue
        if r["grant"] is None:
            wait = f"park n/a (lock free on request)  fin-release +{r['fin_after_release']}"
            grant = ""
        else:
            wait = f"park {r['park']:>3}  grant {r['grant']:+d}  mm2s-g {_num(r['mm2s_after_grant'])}  " \
                   f"s2mm-g {_num(r['s2mm_after_grant'])}"
        conf = " ".join(f"{b.replace('CONFLICT_DM_', '')}@+{t}" for b, t in r["conflicts"]) or "-"
        lines.append(f"{d} {run:<8} {r['d']:>6} {r['d1']:>4}  {wait}  fin-pre {r['fin_after_pre']}  conflicts {conf}")
    return "\n".join(lines)


def format_sync_summary(rows) -> str:
    """One line per d: every run's grant-minus-release and finish-minus-grant side by side."""
    by_d: dict = {}
    for d, run, r in rows:
        by_d.setdefault(d, []).append((run, r))
    lines = ["side by side per d (run g+<grant minus release>/f<MM2S finish minus grant>; d_meas in brackets):"]
    for d in sorted(by_d, key=_label_key):
        cells = []
        for run, r in sorted(by_d[d], key=lambda x: x[0]):
            if r.get("grant") is None:
                cells.append(f"{run} nowait" if r.get("pre_fin") is not None else f"{run} n/a")
            else:
                cells.append(f"{run} g+{r['grant']}/f{r['mm2s_after_grant']}[{r['d']}]")
        lines.append(f"d={d}  " + "  ".join(cells))
    return "\n".join(lines)


def sync_build_case(c: SyncCase, out: Path) -> None:
    d = out / c.name
    d.mkdir(parents=True, exist_ok=True)
    lp.assemble(sync_core_asm(c), d / "core.o", "k_core")
    (d / "base.mlir").write_text(sync_design_mlir(c))
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time", *sync_inject_args(c.bank),
           "--trace-config-out", "trace_config.json", "--config-test-name", c.name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    (d / "case.json").write_text(json.dumps({"case": c.name, "d": c.d, "bank": c.bank, "k": c.k,
                                             "free": c.free, "pre2": c.pre2, "sync": True}) + "\n")


def sync_selection(args) -> list[SyncCase]:
    if args.free_only:
        return [SyncCase(FREE_D, free=True, pre2=args.pre2)]
    return sync_specs(args.ds or SYNC_DEFAULT_DS, args.ks, args.banks or ["none"], args.free, args.pre2)


def cmd_sync_build(args) -> int:
    out = Path(args.out_dir).resolve()
    for c in sync_selection(args):
        sync_build_case(c, out)
        print("built", c.name)
    return 0


def cmd_sync_run(args) -> int:
    out = Path(args.out_dir).resolve()
    for c in sync_selection(args):
        if not (out / c.name).is_dir():
            continue
        if not lp.run_case(out, c.name, args.tag, args.backend, output=False):
            print(f"{c.name}: run did not complete; stop")
            return 1
    return 0


def cmd_sync_table(args) -> int:
    out = Path(args.out_dir).resolve()
    rows = []
    for c in sync_selection(args):
        for run in args.runs:
            run_dir = out / c.name / run
            if not run_dir.is_dir():
                continue
            if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
                rows.append((c.label, run, {"release": None}))
                continue
            rows.append((c.label, run, sync_row(load_level_events(run_dir))))
    print(format_sync_table(rows))
    print()
    print(format_sync_summary(rows))
    return 0


# -- bank arbitration probe (arb-*): who wins a core-vs-DMA bank conflict --------------------------
#
# The chained lock-fetch design, with a core program that makes n consecutive same-bank accesses (loads
# or stores) k nops after releasing lock A, optionally after m extra same-bank loads in the settle phase
# (before the release, while the MM2S is parked and the bank is otherwise idle). The core marks the
# window with `event #0` (just before the release) and `event #1` (ARB_EVENT1_NOPS bundles after the
# accesses), so a lost core access shows as one extra cycle between the two core events and as a
# MEMORY_STALL event, independently of the lock-B stall (which is blind once B is released first).
#
# Candidate rule under test (the "rotation" rule): each physical bank keeps a count of the cycles in
# which it served an access (any requester); in a cycle where the core and the DMA both want the bank,
# the DMA wins when that count is at one residue modulo ARB_PERIOD and the core wins otherwise.
# `rotation_prediction` simulates it for one case so the table can print the prediction next to the
# measurement; the residue is fixed per bank by ARB_DMA_WIN_ORDINAL (the ordinal, counted from 0, of
# the MM2S access on that bank at which a lone core access loses when nothing else touched the bank).

ARB_PERIOD = 5
ARB_DMA_WIN_ORDINAL = {0: 2, 1: 4}       # bank 0: the 3rd access (g+7); bank 1: the 5th (g+27)
# MM2S accesses of the 64-word lock-started BD per physical bank, cycles after the lock grant
# (silicon map of the chained lock-fetch probe; bank = granule parity).
ARB_DMA_ACCESSES = {0: (2, 4, 7, 15, 23, 31, 39, 47), 1: (3, 5, 11, 19, 27, 35, 43, 51)}
ARB_OPS = ("l", "s")                     # load, store
ARB_MAX_N = 6
ARB_MAX_M = 4
ARB_EVENT1_NOPS = 4
# Same-physical-bank offsets in `ldbuf` (0x800, 64 bytes): bit 4 of the address picks the bank.
ARB_OFFSETS = {0: (0, 4, 8, 12, 32, 36), 1: (16, 20, 24, 28, 48, 52)}
ARB_LOAD_REGS = ("r26", "r27", "r28", "r29", "r30", "r31")
# No LOCK_STALL: its four edges overflow the core's first trace packet, which then pushes the
# memory-module packet out of the capture (silicon, first arb run). The window replaces it.
ARB_CORE_EVENTS = ["INSTR_EVENT_0", "INSTR_EVENT_1", "MEMORY_STALL"]


@dataclass(frozen=True)
class ArbCase:
    bank: int                      # physical bank 0 or 1
    op: str                        # "l" load, "s" store
    n: int                         # consecutive same-bank accesses after the release
    k: int                         # nops between the release and the first access
    m: int = 0                     # extra same-bank loads in the settle phase

    @property
    def name(self) -> str:
        return f"ar{self.bank}{self.op}{self.n}k{self.k:02d}m{self.m}"


def arb_validate(c: ArbCase) -> None:
    if c.bank not in (0, 1):
        raise ValueError("bank must be 0 or 1")
    if c.op not in ARB_OPS:
        raise ValueError(f"op must be one of {ARB_OPS}")
    if not 1 <= c.n <= ARB_MAX_N:
        raise ValueError(f"n must be 1..{ARB_MAX_N}")
    if c.k not in FETCH_KS:
        raise ValueError(f"k must be 0..{FETCH_KS[-1]}")
    if not 0 <= c.m <= ARB_MAX_M:
        raise ValueError(f"m must be 0..{ARB_MAX_M}")


def arb_accesses(c: ArbCase) -> list[str]:
    """The n same-bank accesses, one bundle each."""
    offs = ARB_OFFSETS[c.bank]
    if c.op == "l":
        return [f"  lda {ARB_LOAD_REGS[i]}, [p0, #{offs[i]}]" for i in range(c.n)]
    return [f"  st r2, [p0, #{offs[i]}]" for i in range(c.n)]


def arb_core_asm(c: ArbCase) -> str:
    """go, settle, m same-bank loads, `event #0`, release A, k nops, n accesses, `event #1`, wait B.

    Waits only on the go lock (set by the runtime sequence) and on B (released by the MM2S), as the
    lock-fetch core. r2 = +1 releases (and is the stored value), r3 = -1 acquires. p0 = `ldbuf`."""
    arb_validate(c)
    n = lp.nops
    settle_loads = [f"  lda r25, [p0, #{ARB_OFFSETS[c.bank][0]}]"] * c.m
    body = ["  mov r2, #1", "  mov r3, #-1", *n(2),
            f"  acq #{OWN + FETCH_LOCK_GO}, r3", *n(4),
            *lp.delay_loop(FETCH_SETTLE_ITERS, ".Lsettle"), *settle_loads, *n(8),
            "  event #0", f"  rel #{OWN + FETCH_LOCK_A}, r2", *n(c.k),
            *arb_accesses(c), *n(ARB_EVENT1_NOPS), "  event #1", *n(8),
            f"  acq #{OWN + FETCH_LOCK_B}, r3", *n(8),
            *lp.delay_loop(FETCH_TAIL_ITERS, ".Ltail")]
    return lp.frame(body)


def arb_plan() -> list[ArbCase]:
    """The probe grid. k = cell + 1 puts the first access on `cell` after a grant of release + 2 and
    one cycle before it after release + 3 (the two grant phases silicon shows)."""
    cases = []
    # repeat the two measured core losses and the cell the rotation rule says loses next on bank 0
    for bank, cell in ((0, 7), (1, 27), (0, 47)):
        for k in (cell + 1, cell + 2):
            cases.append(ArbCase(bank, "l", 1, k))
    # bursts of 1..6 against bank-0 accesses at different rotation phases, loads and stores
    for op in ARB_OPS:
        for cell in (15, 23, 31):
            for n in range(1, ARB_MAX_N + 1):
                cases.append(ArbCase(0, op, n, cell + 1))
    # history: m extra bank-0 loads before the release move the losing access
    for cell in (7, 15):
        for m in range(1, ARB_MAX_M + 1):
            cases.append(ArbCase(0, "l", 1, cell + 1, m))
    # bank 1 bursts at two other phases: g+19 (DMA due on the 2nd contended cycle), g+35 (on the 5th)
    for n in (1, 2, 3):
        cases.append(ArbCase(1, "l", n, 20))
    for n in (4, 5, 6):
        cases.append(ArbCase(1, "l", n, 36))
    return cases


def rotation_prediction(c: ArbCase, start: int, period: int = ARB_PERIOD,
                        win_ordinal: dict | None = None) -> dict:
    """The rotation rule's outcome for case c when its first access lands `start` cycles after the grant.

    The bank's served-access count starts (at the grant) at m - win_ordinal: the DMA wins a contended
    cycle when the count is 0 modulo `period`. A denied DMA access retries every cycle; a denied core
    access stalls the core one cycle and is repeated. Returns the contended cycles, the cycles the core
    lost (each one core stall), and when the DMA's accesses on this bank were served."""
    win = (win_ordinal or ARB_DMA_WIN_ORDINAL)[c.bank]
    count = c.m - win
    dma = list(ARB_DMA_ACCESSES[c.bank])
    core_left, t_core = c.n, start
    pending = []                   # DMA accesses requested and not yet served
    contended, core_lost, served = [], [], []
    t = min(dma[0], start)
    while dma or pending or core_left:
        while dma and dma[0] <= t:
            pending.append(dma.pop(0))
        core_wants = core_left > 0 and t >= t_core
        if pending and core_wants:
            contended.append(t)
            if count % period == 0:
                core_lost.append(t)
                served.append((pending.pop(0), t))
            else:
                core_left -= 1
            count += 1
        elif pending:
            served.append((pending.pop(0), t))
            count += 1
        elif core_wants:
            core_left -= 1
            count += 1
        t += 1
    return {"contended": contended, "core_lost": core_lost, "served": served}


def arb_row(events) -> dict:
    """fetch_row plus the core window: event #1 minus event #0 (core timer) and the MEMORY_STALL cycles
    relative to event #0."""
    r = fetch_row(events)
    if r.get("release") is None:
        return r
    core: dict = {}
    for e in events:
        if e.row == FETCH_ROW and e.pkt_type == PT_CORE:
            core.setdefault(e.name, []).append(e.ts)
    ev0 = sorted(core.get("INSTR_EVENT_0", []))
    ev1 = sorted(core.get("INSTR_EVENT_1", []))
    r["window"] = ev1[0] - ev0[0] if ev0 and ev1 else None
    r["memory_stalls"] = ([t - ev0[0] for t in sorted(set(core.get("MEMORY_STALL", []))) if t >= ev0[0]]
                          if ev0 else [])
    return r


def arb_clean_window(c: ArbCase) -> int:
    """Bundles from `event #0` to `event #1`: the release, k nops, n accesses, ARB_EVENT1_NOPS nops, then
    the marker. Silicon adds a constant to this (calibrated from conflict-free rows), plus one per core
    stall."""
    return c.k + c.n + ARB_EVENT1_NOPS + 2


def arb_start(c: ArbCase, grant: int) -> int:
    """Grant-relative cycle of the case's first access: release + k + 1 on silicon (lock-fetch fit)."""
    return c.k + 1 - grant


def format_arb_table(rows) -> str:
    """rows: (ArbCase, run, arb_row dict)."""
    lines = ["case          run      grant  conflicts(g+)                      core_lost  window(+stall)  "
             "mm2s_fin s2mm_fin | rotation rule: start contended core_lost"]
    for c, run, r in sorted(rows, key=lambda x: (x[0].name, x[1])):
        if r.get("release") is None or r.get("grant") is None:
            lines.append(f"{c.name:<13} {run:<8} no release or grant")
            continue
        g = r["grant"]
        conf = " ".join(f"b{b[-1]}@{t - g}" for b, t in r["conflicts"]) or "-"
        win = r.get("window")
        extra = "?" if win is None else f"{win}({win - arb_clean_window(c):+d})"
        mstall = ",".join(str(t) for t in r.get("memory_stalls", [])) or "-"
        start = arb_start(c, g)
        p = rotation_prediction(c, start)
        verdict = "MATCH" if arb_matches(r, p) else "DIFFER"
        lines.append(f"{c.name:<13} {run:<8} +{g}     {conf:<34} {mstall:<10} {extra:<15} "
                     f"{','.join(f'+{t - g}' for t in r['mm2s_finished']):<8} "
                     f"{','.join(f'+{t - g}' for t in r['s2mm_finished']):<8} | "
                     f"g+{start} {','.join(map(str, p['contended'])) or '-'} {','.join(map(str, p['core_lost'])) or '-'} "
                     f"{verdict}")
    return "\n".join(lines)


def arb_matches(r: dict, p: dict) -> bool:
    """The measured row agrees with a prediction: the same contended cycles (conflict events, grant-
    relative, the probed bank) and as many core stalls (MEMORY_STALL events) as predicted core losses."""
    g = r["grant"]
    seen = sorted(t - g for _b, t in r["conflicts"])
    return seen == p["contended"] and len(r.get("memory_stalls", [])) == len(p["core_lost"])


def arb_selection(args) -> list[ArbCase]:
    plan = arb_plan()
    return [c for c in plan if not args.names or c.name in args.names]


def arb_build_case(c: ArbCase, out: Path) -> None:
    d = out / c.name
    d.mkdir(parents=True, exist_ok=True)
    lp.assemble(arb_core_asm(c), d / "core.o", "k_core")
    (d / "base.mlir").write_text(fetch_design_mlir(chained=True))
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time", *arb_inject_args(),
           "--trace-config-out", "trace_config.json", "--config-test-name", c.name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    (d / "case.json").write_text(json.dumps({"case": c.name, "bank": c.bank, "op": c.op, "n": c.n,
                                             "k": c.k, "m": c.m, "arb": True}) + "\n")


def arb_inject_args() -> list[str]:
    return ["--memmod-sweep-events", ",".join(FETCH_MEMMOD_EVENTS),
            "--core-grounding", FETCH_CORE_GROUNDING,
            "--core-sweep-events", ",".join(ARB_CORE_EVENTS)]


def cmd_arb_build(args) -> int:
    out = Path(args.out_dir).resolve()
    for c in arb_selection(args):
        arb_build_case(c, out)
        print("built", c.name)
    return 0


def cmd_arb_run(args) -> int:
    out = Path(args.out_dir).resolve()
    for c in arb_selection(args):
        if not (out / c.name / "aie.xclbin").exists():
            continue
        if not lp.run_case(out, c.name, args.tag, args.backend, output=False):
            print(f"{c.name}: run did not complete; stop")
            return 1
    return 0


def cmd_arb_table(args) -> int:
    out = Path(args.out_dir).resolve()
    rows = []
    for c in arb_selection(args):
        for run in args.runs:
            run_dir = out / c.name / run
            if not run_dir.is_dir():
                continue
            if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
                rows.append((c, run, {"release": None}))
                continue
            rows.append((c, run, arb_row(load_level_events(run_dir))))
    print(format_arb_table(rows))
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
    fb.add_argument("--banks", nargs="+", choices=FETCH_BANKS + ("none",), default=None,
                    help="both = two loads (lfk/lfc); 0 or 1 = one load on that bank (lfa/lfb chained, lfd/lfe first-BD)")
    fr = sub.add_parser("fetch-run", help="lock-to-fetch probe: run the built cases to TAG.BACKEND/")
    fr.add_argument("out_dir")
    fr.add_argument("tag")
    fr.add_argument("--backend", choices=("hw", "emu"), required=True)
    fr.add_argument("--ks", nargs="*", type=int, default=[])
    fr.add_argument("--chained", action="store_true", help="the chained variant (cases lfcNN)")
    fr.add_argument("--banks", nargs="+", choices=FETCH_BANKS + ("none",), default=None, help="as for fetch-build")
    ft = sub.add_parser("fetch-table", help="lock-to-fetch probe: per k and run, events relative to the release")
    ft.add_argument("out_dir")
    ft.add_argument("runs", nargs="+", help="run directory names, e.g. h1.hw e1.emu")
    ft.add_argument("--ks", nargs="*", type=int, default=[])
    ft.add_argument("--chained", action="store_true", help="the chained variant (cases lfcNN)")
    ft.add_argument("--banks", nargs="+", choices=FETCH_BANKS + ("none",), default=None, help="as for fetch-build")
    for sp in (fb, fr, ft):
        sp.add_argument("--sync", action="store_true",
                        help="the lock-synchronised variant (cases lsDD, lsfree, lsaDDkKK/lsbDDkKK): --ds are core delays, "
                             "--banks none|0|1 (default none), --ks the load offsets of the bank cases")
        sp.add_argument("--ds", nargs="*", type=int, default=[], help="--sync: core delays d (default 0..24, 26..40 by 2)")
        sp.add_argument("--free", action="store_true", help="--sync: also the lock-free-on-request case (lsfree)")
        sp.add_argument("--pre2", type=int, default=0, help="--sync: words of a second lock-free pre BD (cases ltDD...)")
        sp.add_argument("--free-only", action="store_true", help="--sync: only lsfree")
    ab = sub.add_parser("arb-build", help="bank arbitration probe: build the plan's cases (or --names)")
    ab.add_argument("out_dir")
    ar = sub.add_parser("arb-run", help="bank arbitration probe: run the built cases to TAG.BACKEND/")
    ar.add_argument("out_dir")
    ar.add_argument("tag")
    ar.add_argument("--backend", choices=("hw", "emu"), required=True)
    at = sub.add_parser("arb-table", help="bank arbitration probe: measurement next to the rotation rule")
    at.add_argument("out_dir")
    at.add_argument("runs", nargs="+")
    for sp in (ab, ar, at):
        sp.add_argument("--names", nargs="*", default=[], help="only these case names")
    args = ap.parse_args(argv)
    if args.cmd.startswith("arb-"):
        return {"arb-build": cmd_arb_build, "arb-run": cmd_arb_run, "arb-table": cmd_arb_table}[args.cmd](args)
    if args.cmd.startswith("fetch-"):
        if args.sync:
            return {"fetch-build": cmd_sync_build, "fetch-run": cmd_sync_run, "fetch-table": cmd_sync_table}[args.cmd](args)
        args.banks = args.banks or ["both"]
        if any(b == "none" for b in args.banks):
            ap.error("--banks none needs --sync")
    return {"generate": cmd_generate, "build": cmd_build, "run": cmd_run, "analyze": cmd_analyze,
            "fetch-build": cmd_fetch_build, "fetch-run": cmd_fetch_run,
            "fetch-table": cmd_fetch_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
