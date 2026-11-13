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
`--memmod-event-swap OLD=NEW` (fetch-build and chain-build too) exchanges one traced memory-module event for another in
the same slot, e.g. DMA_S2MM_0_FINISHED_BD=DMA_MM2S_0_MEMORY_STARVATION; `--memmod-event-add NAME` appends one to a list with a free
slot. LOCK_n_REL follows Locks_Event_Selection_n, which resets to lock 0 for every n: `--sync --lock-event-select 1=1` writes
selector 1 (offset and field from the register database) in the runtime sequence before the start lock is set, so with
`--memmod-event-add LOCK_1_REL` the head BD's release of the sync lock is stamped. Each extra register write or traced
event in the sequence shifts every stamp of a run by the same amount (112 cycles in the emulator) and can move the phase
of the core against the DMA by a cycle, so compare cycle offsets inside one run, not absolute stamps.

    python3 tools/gen_dma_lock_probe.py fetch-build OUT_DIR --sync [--ds 0 4 8] [--banks none|0|1] [--ks ..] [--free]
    python3 tools/gen_dma_lock_probe.py fetch-run OUT_DIR TAG --backend hw|emu --sync [same selection]
    python3 tools/gen_dma_lock_probe.py fetch-table OUT_DIR RUN [RUN ...] --sync [same selection]

DMA-side stamp cases (`fetch-build --start-task`, cases lfsNN first-BD / lfqNN with `--chained`; NN = nops before the
release, `--pre-nops`, default 0..5). The runtime sequence configures and starts MM2S 0's task
(`aiex.dma_configure_task` / `dma_start_task`) before it sets the go lock, so DMA_MM2S_0_START_TASK (the 8th
memory-module slot) falls in the trace window; with a boot-started task it fires before the trace is armed. The core
makes no loads. The table adds the start stamp (cycles from the release) and a finish check.

Refused-access probe (`refuse-build`, `refuse-run`, `refuse-table`; cases rf{e|l}{r}sSS, 48 in all): a core camp
refuses r = 0..3 accesses of a chained BD, EARLY or LATE in its read, and a second lock BD follows it, whose lock the
core releases s nops after the camp (s swept over six values). See the section comment above `refuse_design_mlir`.

    python3 tools/gen_dma_lock_probe.py fetch-build OUT_DIR --start-task [--chained] [--pre-nops 0 1 2 3 4 5]
    python3 tools/gen_dma_lock_probe.py refuse-build OUT_DIR [--names rfe3s35 ...]
    python3 tools/gen_dma_lock_probe.py refuse-run OUT_DIR TAG --backend hw|emu [--names ...]
    python3 tools/gen_dma_lock_probe.py refuse-table OUT_DIR RUN [RUN ...] [--names ...]

Lock-held S2MM probe (`sl-build`, `sl-run`, `sl-table`; cases slhDD / slfDD, slhDDkKKpP, slhDDcamp). The S2MM side of
the lock-grant timing: an MM2S lock-free BD feeds the tile's own S2MM 0, whose 64-word BD waits on lock A (a lock-free
4-word head BD chained in front of it in `slh`, the task's first BD in `slf`). The core releases A d nops after
`event #0` and then does nothing, makes one load k nops later at granule parity p, or camps on bank 0. The table reads
only the memory-module unit: the grant (STALLED_LOCK fall) after the release, the S2MM FINISHED_BD, the MM2S
backpressure fall, STREAM_STARVATION and the conflict stamps, all relative to the grant. See the section comment above
`SlCase`.

    python3 tools/gen_dma_lock_probe.py sl-build OUT_DIR [--names slh12 ...]
    python3 tools/gen_dma_lock_probe.py sl-run OUT_DIR TAG --backend hw|emu [--names ...]
    python3 tools/gen_dma_lock_probe.py sl-table OUT_DIR RUN [RUN ...] [--names ...]

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
# --start-task cases: the DMA-side stamp of the task start, then a release `pre` nops later against it; no core loads.
# The NN of the name is the pre-release nop count. First-BD (lfsNN) and chained (lfqNN) forms.
FETCH_START_PREFIX = {False: "lfs", True: "lfq"}
FETCH_START_EVENT = "DMA_MM2S_0_START_TASK"        # the 8th memory-module slot
FETCH_PRE_NOPS = tuple(range(6))                   # default pre-release nop sweep: every residue mod 3 twice
FETCH_MAX_PRE = 64
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


MEMMOD_ADD_HELP = ("append one memory-module trace event to the list (NAME, e.g. LOCK_1_REL); errors when the "
                   "trace unit's eight slots are full or NAME is already traced. Applied after --memmod-event-swap")
LOCK_SELECT_HELP = ("--sync: program lock event selector N to watch lock LOCK_ID (repeatable): a runtime-sequence "
                    "register write placed before the start lock is set. LOCK_n_REL / LOCK_SELn_* events follow "
                    "selector n, which resets to lock 0; selector 0 is left at its reset (LOCK_0_REL is lock 0)")
COMPUTE_MEM_MODULE = "memory"      # the register database's name for the compute tile memory module
LOCK_SELECT_REGISTER = "Locks_Event_Selection_{n}"


def parse_lock_event_select(text: str) -> tuple[int, int]:
    """`N=LOCK_ID` -> (N, LOCK_ID); the argparse type of --lock-event-select."""
    n, sep, lock = text.partition("=")
    try:
        if not sep:
            raise ValueError
        return int(n, 0), int(lock, 0)
    except ValueError:
        raise ValueError(f"expected N=LOCK_ID, got {text!r}") from None


def compute_memory_registers() -> dict:
    """The compute tile memory module of mlir-aie's AM025 register database as {register name: entry}."""
    import toolchain_paths
    path = toolchain_paths.resolve_mlir_aie() / "lib" / "Dialect" / "AIE" / "Util" / "aie_registers_aie2.json"
    return {r["name"]: r for r in json.loads(path.read_text())["modules"][COMPUTE_MEM_MODULE]["registers"]}


def lock_event_select_writes(selects) -> list[tuple[int, int]]:
    """(tile-local address, value) of the register write for each (selector n, lock id).

    Offset, the Lock_Select field and the Lock_Value reset come from the register database; Lock_Value
    (the trigger value of the EQUAL_TO_VALUE events, which the probe does not use) is written at its reset."""
    import re
    regs = compute_memory_registers()
    count = sum(1 for name in regs if re.fullmatch(r"Locks_Event_Selection_\d+", name))
    seen = set()
    out = []
    for n, lock in selects:
        if n == 0:
            raise ValueError("selector 0 stays at its reset (lock 0): LOCK_0_REL is the core's release of lock 0")
        if not 0 < n < count:
            raise ValueError(f"selector {n} out of range (1..{count - 1})")
        if n in seen:
            raise ValueError(f"selector {n} given twice")
        seen.add(n)
        reg = regs[LOCK_SELECT_REGISTER.format(n=n)]
        fields = {f["name"]: f for f in reg["bit_fields"]}
        sel, val = fields["Lock_Select"], fields["Lock_Value"]
        lo, hi = sel["bit_range"]
        if not 0 <= lock < 1 << (hi - lo + 1):
            raise ValueError(f"lock {lock} out of range (0..{(1 << (hi - lo + 1)) - 1})")
        out.append((int(reg["offset"], 0), (lock << lo) | int(val["reset"], 0)))
    return out


MEMMOD_SWAP_HELP = ("replace one memory-module trace event by another (OLD=NEW, e.g. "
                    "DMA_S2MM_0_FINISHED_BD=DMA_MM2S_0_MEMORY_STARVATION); the trace unit has a fixed number of slots, "
                    "so it is a swap, in the same slot. LOCK_n_REL / LOCK_SELn_* events follow "
                    "Locks_Event_Selection_n, which resets to lock 0 for every n. Default: the variant's own list")


def parse_event_swap(text: str) -> tuple[str, str]:
    """`OLD=NEW` -> (OLD, NEW); the argparse type of --memmod-event-swap."""
    old, sep, new = text.partition("=")
    if not sep or not old.strip() or not new.strip() or "=" in new:
        raise ValueError(f"expected OLD=NEW, got {text!r}")
    return old.strip(), new.strip()


MEMMOD_TRACE_SLOTS = 8             # event slots of one trace unit


def _check_memmod_event_name(name: str) -> None:
    try:
        from aie.utils.trace.events import MemEventAIE2
    except ImportError:               # no mlir-aie python on the path: the injector rejects a bad name itself
        return
    if name not in MemEventAIE2.__members__:
        raise ValueError(f"unknown memory-module event {name}")


def apply_event_add(events: list[str], name: str | None) -> list[str]:
    """The memory-module event list with NAME appended (None: unchanged); the list must have a free slot."""
    if name is None:
        return list(events)
    if name in events:
        raise ValueError(f"{name} is already in the memory-module list")
    if len(events) >= MEMMOD_TRACE_SLOTS:
        raise ValueError(f"the memory-module list is full ({MEMMOD_TRACE_SLOTS} slots); use --memmod-event-swap")
    _check_memmod_event_name(name)
    return [*events, name]


def edit_memmod_events(events: list[str], swap: tuple[str, str] | None = None,
                       add: str | None = None) -> list[str]:
    """The variant's list after --memmod-event-swap, then --memmod-event-add."""
    return apply_event_add(apply_event_swap(events, swap), add)


def apply_event_swap(events: list[str], swap: tuple[str, str] | None) -> list[str]:
    """The memory-module event list with OLD replaced in place by NEW (None: unchanged).

    The trace unit has eight slots and a list is capped there, so an event can only be exchanged for
    another, never added past a full list. NEW must be a real memory-module event of the target
    (checked against mlir-aie's AIE2 table) and not already traced."""
    if swap is None:
        return list(events)
    old, new = swap
    if old not in events:
        raise ValueError(f"{old} is not in the memory-module list ({', '.join(events)})")
    if new in events:
        raise ValueError(f"{new} is already in the memory-module list")
    _check_memmod_event_name(new)
    return [new if e == old else e for e in events]
FETCH_CORE_GROUNDING = "PERF_CNT_2"
# tile (0,2) is physical column 1; trace pkt_type 0 = core module, 1 = memory module
FETCH_ROW, PT_CORE, PT_MEM = 2, 0, 1


def fetch_case_name(k: int, chained: bool = False, bank: str = "both", start: bool = False) -> str:
    """lfkNN / lfcNN / ... by k; with `start`, lfsNN / lfqNN where NN is the pre-release nop count (k is 0)."""
    if start:
        return f"{FETCH_START_PREFIX[chained]}{k:02d}"
    return f"{FETCH_PREFIX[(chained, bank)]}{k:02d}"


FETCH_LOADS = {"0": ["  lda r26, [p0, #0]"], "1": ["  lda r28, [p0, #16]"],
               "both": ["  lda r26, [p0, #0]", "  lda r28, [p0, #16]"], "none": []}


def fetch_core_asm(k: int, bank: str = "both", pre: int = 0) -> str:
    """The core of case k: go, settle, `event #0`, release A, k nops, then the load(s).

    bank "both": a bank-0 load and, one bundle later, a bank-1 load. "0" / "1": that one load alone,
    at the same place (k nops after the release), so the other bank is not touched and a bank-0
    conflict cannot disturb a bank-1 access. bank "none": no load (the --start-task cases). `pre` extra
    nops go before `event #0`, which moves the release by single cycles against everything the DMA
    started before it (the settle loop's length is fixed; k only moves what follows the release).
    r2 = +1 releases, r3 = -1 acquires (as gen_lock_pulse_probe). p0 = `ldbuf`."""
    if k not in FETCH_KS:
        raise ValueError(f"k must be 0..{FETCH_KS[-1]}")
    if bank not in FETCH_LOADS:
        raise ValueError(f"bank must be one of {FETCH_BANKS} or none")
    if not 0 <= pre <= FETCH_MAX_PRE:
        raise ValueError(f"pre must be 0..{FETCH_MAX_PRE}")
    n = lp.nops
    body = ["  mov r2, #1", "  mov r3, #-1", *n(2),
            f"  acq #{OWN + FETCH_LOCK_GO}, r3", *n(4),
            *lp.delay_loop(FETCH_SETTLE_ITERS, ".Lsettle"), *n(8), *n(pre),
            "  event #0", f"  rel #{OWN + FETCH_LOCK_A}, r2", *n(k),
            *FETCH_LOADS[bank], *n(8),
            f"  acq #{OWN + FETCH_LOCK_B}, r3", *n(8),
            *lp.delay_loop(FETCH_TAIL_ITERS, ".Ltail")]
    return lp.frame(body)


def fetch_design_mlir(chained: bool = False, task: bool = False) -> str:
    """Case-independent design: the k lives in the core program only.

    chained: MM2S 0's task starts with a lock-free BD of `pre` that chains to the lock-waiting BD.
    task: MM2S 0's task is configured and started by the runtime sequence (`aiex.dma_configure_task` and
    `dma_start_task`, after the trace is armed) instead of at boot, so its DMA_MM2S_0_START_TASK falls in the
    trace window; the BDs and locks are the same, and the S2MM stays in `aie.mem`."""
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
    if task:
        # the task's BDs are in the sequence: `pre` (when chained) and the lock-A BD, one task
        pre_task = (f"        aie.dma_bd(%pre : memref<{p}xi32>)\n        aie.next_bd ^send\n      ^send:\n") if chained else ""
        seq_task = f"""      %c1 = arith.constant 1 : i32
      %task = aiex.dma_configure_task(%t02, MM2S, 0) {{
{pre_task}        aie.use_lock(%lA, AcquireGreaterEqual, %c1)
        aie.dma_bd(%src : memref<{w}xi32>)
        aie.use_lock(%lB, Release, %c1)
        aie.end
      }}
      aiex.dma_start_task(%task)
"""
        mem_head = ""
    else:
        seq_task = ""
        mem_head = f"""      %s0 = aie.dma_start(MM2S, 0, {first}, ^recv_start)
{pre_bd}    ^send:
      %ca = arith.constant 1 : i32
      aie.use_lock(%lA, AcquireGreaterEqual, %ca)
      aie.dma_bd(%src : memref<{w}xi32>)
      %cb = arith.constant 1 : i32
      aie.use_lock(%lB, Release, %cb)
      aie.next_bd ^end
    ^recv_start:
"""
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
{mem_head}      %s1 = aie.dma_start(S2MM, 0, ^recv, ^end)
    ^recv:
      aie.dma_bd(%dst : memref<{dw}xi32>)
      aie.next_bd ^end
    ^end:
      aie.end
    }}
    aie.runtime_sequence(%dummy: memref<4xi32>) {{
      %c_zero = arith.constant 0 : i32
      %c_one = arith.constant 1 : i32
{seq_task}      aiex.set_lock(%lGo, %c_one)
{pad}
    }}
  }}
}}
"""


def fetch_inject_args(start: bool = False, swap: tuple[str, str] | None = None,
                      add: str | None = None) -> list[str]:
    events = edit_memmod_events(FETCH_MEMMOD_EVENTS + ([FETCH_START_EVENT] if start else []), swap, add)
    return ["--memmod-sweep-events", ",".join(events),
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
    row["mm2s_finished_all"] = [t - rel for t in sorted(set(mem.get("DMA_MM2S_0_FINISHED_BD", [])))]
    row["s2mm_finished"] = [t - rel for t in sorted(set(mem.get(f"DMA_S2MM_0_FINISHED_BD", []))) if t >= rel]
    row["mm2s_start"] = [t - rel for t in sorted(set(mem.get(FETCH_START_EVENT, [])))]
    ev0 = sorted(core.get("INSTR_EVENT_0", []))
    row["core_event0"] = ev0[0] - rel if ev0 else None
    row["core_stalls"] = ([(a - ev0[0], b - ev0[0]) for a, b in _runs(core.get("LOCK_STALL", []))
                           if a >= ev0[0] - 2 and a - ev0[0] <= 64] if ev0 else [])
    return row


def start_missing(r: dict, chained: bool = False) -> list[str]:
    """What a --start-task run (fetch_row dict) lacks: the release, the task-start stamp before the grant, the
    grant, and every DMA task finished (the lock BD's FINISHED_BD, plus the pre BD's when chained, and the
    S2MM's)."""
    if r.get("release") is None:
        return ["no LOCK_0_REL"]
    out = []
    if not r.get("mm2s_start"):
        out.append(FETCH_START_EVENT)
    if r.get("grant") is None:
        out.append("grant")
    want = 2 if chained else 1
    fins = len(r.get("mm2s_finished_all", r["mm2s_finished"]))
    if fins != want:
        out.append(f"MM2S FINISHED_BD x{fins} (want {want})")
    if len(r["s2mm_finished"]) != 1:
        out.append(f"S2MM FINISHED_BD x{len(r['s2mm_finished'])} (want 1)")
    return out


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
        start = (f"  start_task [{', '.join(f'{t:+d}' for t in r['mm2s_start'])}]" if r.get("mm2s_start") else "")
        if r.get("check") is not None:
            start += "  finish " + ("ok" if not r["check"] else "MISSING " + "; ".join(r["check"]))
        lines.append(f"{k:>2} {run:<8} {r['release']:>9}  {grant}{start}  {starv}  {conf}  {fin}  {ev0}  {stalls}")
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

def fetch_build_case(k: int, out: Path, chained: bool = False, bank: str = "both", start: bool = False,
                     swap: tuple[str, str] | None = None, add: str | None = None) -> None:
    """Case k of a lock-to-fetch variant. With `start` (the --start-task cases) the number is the pre-release
    nop count, there is no core load, and the task-start event joins the memory-module list."""
    name = fetch_case_name(k, chained, bank, start)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    asm = fetch_core_asm(0, "none", pre=k) if start else fetch_core_asm(k, bank)
    lp.assemble(asm, d / "core.o", "k_core")
    (d / "base.mlir").write_text(fetch_design_mlir(chained, task=start))
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time", *fetch_inject_args(start, swap, add),
           "--trace-config-out", "trace_config.json", "--config-test-name", name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    meta = {"case": name, "k": k, "chained": chained, "bank": bank}
    if start:
        meta.update({"start_task": True, "pre_nops": k, "k": 0, "bank": "none"})
    (d / "case.json").write_text(json.dumps(meta) + "\n")


def fetch_values(args) -> tuple:
    """The case numbers of a fetch selection: the pre-release nop counts with --start-task, else the ks."""
    if getattr(args, "start_task", False):
        return tuple(args.pre_nops or FETCH_PRE_NOPS)
    return tuple(args.ks or FETCH_DEFAULT_KS)


def fetch_cases_in(out: Path, ks, chained: bool = False, bank: str = "both", start: bool = False) -> list[int]:
    return [k for k in (ks or FETCH_KS) if (out / fetch_case_name(k, chained, bank, start)).is_dir()]


def fetch_banks(args) -> list[str]:
    """--start-task cases have no core loads: one pseudo bank, none."""
    return ["none"] if getattr(args, "start_task", False) else args.banks


def cmd_fetch_build(args) -> int:
    out = Path(args.out_dir).resolve()
    start = getattr(args, "start_task", False)
    for bank in fetch_banks(args):
        for k in fetch_values(args):
            fetch_build_case(k, out, args.chained, bank, start, getattr(args, "memmod_event_swap", None),
                             getattr(args, "memmod_event_add", None))
            print("built", fetch_case_name(k, args.chained, bank, start))
    return 0


def cmd_fetch_run(args) -> int:
    out = Path(args.out_dir).resolve()
    start = getattr(args, "start_task", False)
    ok = True
    for bank in fetch_banks(args):
        for k in fetch_cases_in(out, fetch_values(args) if start else args.ks, args.chained, bank, start):
            name = fetch_case_name(k, args.chained, bank, start)
            ok = lp.run_case(out, name, args.tag, args.backend, output=False) and ok
            if not ok:
                print(f"{name}: run did not complete; stop")
                return 1
    return 0


def cmd_fetch_table(args) -> int:
    out = Path(args.out_dir).resolve()
    start = getattr(args, "start_task", False)
    entries = []
    for bank in fetch_banks(args):
        rows = []
        for k in fetch_cases_in(out, fetch_values(args) if start else args.ks, args.chained, bank, start):
            name = fetch_case_name(k, args.chained, bank, start)
            for run in args.runs:
                run_dir = out / name / run
                if not run_dir.is_dir():
                    continue                    # this case was not run under this tag
                if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
                    rows.append((k, run, {"release": None}))
                    continue
                r = fetch_row(load_level_events(run_dir))
                if start:
                    r["check"] = start_missing(r, args.chained)
                rows.append((k, run, r))
                entries.append((name, k, bank, run, r))
        label = f"{FETCH_START_PREFIX[args.chained]}NN, NN = pre-release nops" if start else \
            f"{FETCH_PREFIX[(args.chained, bank)]}NN"
        print(f"== bank {bank} ({label}) ==")
        print(format_fetch_table(rows))
        print()
    if not start:
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


def sync_design_mlir(c: SyncCase, selects=()) -> str:
    """MM2S 0: pre BD (acquire START, 4 words of `pre`, release SYNC) chained to the BD that waits on A.

    The runtime sequence's only action is setting START, after the trace is armed, so the pre BD runs
    (and releases SYNC for the core) inside the trace window. The S2MM takes both BDs' words."""
    w, p = FETCH_WORDS, FETCH_PRE_WORDS
    q = c.pre2
    dw = w + p + q
    sel = "".join(f"      %c_sa{i} = arith.constant {a} : i32\n      %c_sv{i} = arith.constant {v} : i32\n"
                  f"      aiex.npu.write32(%c_sa{i}, %c_sv{i}) {{column = 0 : i32, row = {FETCH_ROW} : i32}} : i32, i32\n"
                  for i, (a, v) in enumerate(lock_event_select_writes(selects))) if selects else ""
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
{sel}      aiex.set_lock(%lStart, %c_one)
{pad}
    }}
  }}
}}
"""


def sync_inject_args(bank: str, swap: tuple[str, str] | None = None, add: str | None = None) -> list[str]:
    events = edit_memmod_events(SYNC_MAIN_EVENTS + (SYNC_CONFLICT_EVENTS if bank != "none" else []), swap, add)
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


def sync_build_case(c: SyncCase, out: Path, swap: tuple[str, str] | None = None, add: str | None = None,
                    selects=()) -> None:
    d = out / c.name
    d.mkdir(parents=True, exist_ok=True)
    lp.assemble(sync_core_asm(c), d / "core.o", "k_core")
    (d / "base.mlir").write_text(sync_design_mlir(c, selects))
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time", *sync_inject_args(c.bank, swap, add),
           "--trace-config-out", "trace_config.json", "--config-test-name", c.name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    (d / "case.json").write_text(json.dumps({"case": c.name, "d": c.d, "bank": c.bank, "k": c.k,
                                             "free": c.free, "pre2": c.pre2, "sync": True,
                                             **({"memmod_event_swap": list(swap)} if swap else {}),
                                             **({"memmod_event_add": add} if add else {}),
                                             **({"lock_event_select": [list(x) for x in selects]} if selects else {})}) + "\n")


def sync_selection(args) -> list[SyncCase]:
    if args.free_only:
        return [SyncCase(FREE_D, free=True, pre2=args.pre2)]
    return sync_specs(args.ds or SYNC_DEFAULT_DS, args.ks, args.banks or ["none"], args.free, args.pre2)


def cmd_sync_build(args) -> int:
    out = Path(args.out_dir).resolve()
    for c in sync_selection(args):
        sync_build_case(c, out, getattr(args, "memmod_event_swap", None), getattr(args, "memmod_event_add", None),
                        getattr(args, "lock_event_select", None) or ())
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


# -- lock request/grant timing family (chain-*): any chain of BDs between the head and the waiting BD ---------
#
# MM2S 0 runs one chained task: the head BD (acquires START, which the runtime sequence sets after the trace is
# armed, sends `head` words, releases SYNC), then the mid BDs in order, then the waiting BD (acquires lock A = lock 0,
# sends `big` words of `src`, releases END). A mid BD is lock-free ("8"), takes its own lock that starts at 1 ("8f":
# free when asked), or takes lock B that the core releases ("8c": at most one). The core blocks on SYNC, runs
# `event #0`, d1 nops, releases B (only with a core BD), `event #1`, d2 nops, releases A. With `releaser = "s2mm"`
# the core releases nothing: the loopback S2MM's first BD (all the words before the waiting BD) releases A when it
# finishes, and a second S2MM BD takes the waiting BD's words.
# Why it cannot hang: every lock a BD waits on is either free at start (the "f" locks, the S2MM's lRx), set by the
# sequence (START), or released by a party whose own waits are earlier in the same chain: the core waits only on SYNC
# (the head's release) and then releases B and A unconditionally; the S2MM's first BD needs only the words the MM2S
# sends before the waiting BD. Nothing waits on the host.
# With `fill` > 0 the core, after its tail (long after every DMA task of the chain has finished), releases lock A
# `fill` more times, the i-th after CHAIN_FILL_LOOP_ITERS + i delay-loop iterations, then runs the tail again: the
# memory-module LOCK_0_REL frames fill the stream's last packets, so no measured frame sits in a packet silicon may
# not deliver (silicon does not write a stream's final partial packet). The spacing grows so that no two fill frames
# are equal: the trace encoder folds a run of equal (event, delta) frames into one frame and a Repeat. Nothing acquires A then; its count stays far below the
# 6-bit limit (at most CHAIN_MAX_FILL = 48).

CHAIN_LOCK_B, CHAIN_LOCK_RX, CHAIN_LOCK_FREE0 = 4, 5, 6   # B: the core-released mid BD's lock; RX: the S2MM's
CHAIN_MAX_FREE = 16 - CHAIN_LOCK_FREE0                     # free locks 6..15
CHAIN_MAX_BDS = 16                                         # BD descriptors of a compute tile's DMA
CHAIN_BUF_BASE, CHAIN_BUF_END = 0x4000, 0x8000             # head and mid buffers: logical bank 1
CHAIN_MAX_BIG = (FETCH_LD_ADDR - FETCH_SRC_ADDR) // 4      # `src` must end before `ldbuf`
CHAIN_MAX_D = 40
CHAIN_MAX_FILL = 48                # fill releases of A: its count (0 by then) stays below the 6-bit limit 63
CHAIN_FILL_LOOP_ITERS = 4          # delay before the first fill release; each later one waits one iteration more
CHAIN_LOCK_KINDS = {"": "none", "f": "free", "c": "core"}
CHAIN_MEM_EVENTS = ["LOCK_0_REL", "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_0_FINISHED_BD", "DMA_S2MM_0_FINISHED_BD",
                    "LOCK_SEL0_ACQ_GE"]
CHAIN_CORE_EVENTS = ["INSTR_EVENT_0", "INSTR_EVENT_1", "LOCK_STALL"]


@dataclass(frozen=True)
class ChainBd:
    words: int
    lock: str = "none"             # "none" | "free" | "core"


@dataclass(frozen=True)
class ChainCase:
    mids: tuple = ()               # ChainBd, in chain order between the head and the waiting BD
    head: int = 4                  # words of the head BD (acquires START, releases SYNC)
    big: int = 64                  # words of the waiting BD (acquires A)
    d1: int = 0                    # nops between `event #0` and the core's release of B (needs a core BD)
    d2: int = 0                    # nops between `event #1` and the core's release of A
    releaser: str = "core"         # who releases A: "core" or "s2mm"
    bd_ids: tuple = ()             # bd_id of every MM2S BD (head, mids, waiting BD); () = aiecc assigns
    fill: int = 0                  # core releases of A after the tail (trace packet fillers; core releaser only)

    @property
    def name(self) -> str:
        """q<d1>_<d2>; one design per output directory (design.json), so the name only carries the delays."""
        return "qs" if self.releaser == "s2mm" else f"q{self.d1:02d}_{self.d2:02d}"

    def design(self) -> dict:
        d = {"head": self.head, "mids": format_chain_mids(self.mids), "big": self.big, "releaser": self.releaser}
        if self.bd_ids:
            d["bd_ids"] = list(self.bd_ids)
        if self.fill:
            d["fill"] = self.fill
        return d


def parse_chain_mids(tokens) -> tuple:
    """["8", "4c", "2f"] -> ChainBd tuple. Validation (counts, sizes) is chain_validate's."""
    out = []
    for t in tokens:
        digits = t.rstrip("cf")
        if not digits.isdigit() or t[len(digits):] not in CHAIN_LOCK_KINDS:
            raise ValueError(f"mid BD spec {t!r}: words then optional c (core lock) or f (free lock)")
        out.append(ChainBd(int(digits), CHAIN_LOCK_KINDS[t[len(digits):]]))
    return tuple(out)


def format_chain_mids(mids) -> str:
    back = {v: k for k, v in CHAIN_LOCK_KINDS.items()}
    return " ".join(f"{m.words}{back[m.lock]}" for m in mids)


def chain_buffers(c: ChainCase) -> list:
    """(address, words) of the head buffer then each mid BD's, packed from 0x4000 on 32-byte boundaries (aiecc
    requires 256-bit aligned buffers)."""
    out, a = [], CHAIN_BUF_BASE
    for w in (c.head, *(m.words for m in c.mids)):
        out.append((a, w))
        a += -(-4 * w // 32) * 32
    return out


def chain_validate(c: ChainCase) -> None:
    if c.releaser not in ("core", "s2mm"):
        raise ValueError("releaser must be core or s2mm")
    if any(m.words < 1 for m in c.mids) or c.head < 1 or not 1 <= c.big <= CHAIN_MAX_BIG:
        raise ValueError(f"every BD needs >= 1 word; big at most {CHAIN_MAX_BIG}")
    cores = sum(m.lock == "core" for m in c.mids)
    if cores > 1 or (cores and c.releaser == "s2mm"):
        raise ValueError("at most one core-released mid BD, and none with the S2MM releaser")
    if sum(m.lock == "free" for m in c.mids) > CHAIN_MAX_FREE:
        raise ValueError(f"at most {CHAIN_MAX_FREE} free-lock mid BDs")
    if 1 + len(c.mids) + 1 + (2 if c.releaser == "s2mm" else 1) > CHAIN_MAX_BDS:
        raise ValueError(f"at most {CHAIN_MAX_BDS} BDs in the tile")
    if not (0 <= c.d1 <= CHAIN_MAX_D and 0 <= c.d2 <= CHAIN_MAX_D) or (c.d1 and not cores):
        raise ValueError(f"d1, d2 in 0..{CHAIN_MAX_D}; d1 only with a core BD")
    if c.bd_ids and (len(c.bd_ids) != len(c.mids) + 2 or len(set(c.bd_ids)) != len(c.bd_ids)
                     or not all(0 <= i < CHAIN_MAX_BDS for i in c.bd_ids)):
        raise ValueError(f"bd_ids: one distinct id in 0..{CHAIN_MAX_BDS - 1} per MM2S BD (head, mids, waiting BD)")
    if not 0 <= c.fill <= CHAIN_MAX_FILL or (c.fill and c.releaser != "core"):
        raise ValueError(f"fill in 0..{CHAIN_MAX_FILL}, and only with the core releaser")
    a, w = chain_buffers(c)[-1]
    if a + 4 * w > CHAIN_BUF_END:
        raise ValueError("head and mid buffers exceed logical bank 1")


def chain_core_asm(c: ChainCase) -> str:
    """Block on SYNC, `event #0`, [d1 nops, release B], `event #1`, d2 nops, release A (core releaser only), the
    tail, then `fill` spaced releases of A and the tail again (trace packet fillers, only with `fill`)."""
    chain_validate(c)
    n = lp.nops
    fill = []
    for i in range(c.fill):
        fill += [*lp.delay_loop(CHAIN_FILL_LOOP_ITERS + i, f".Lfill{i}"), f"  rel #{OWN + SYNC_LOCK_A}, r2"]
    if fill:
        fill += lp.delay_loop(FETCH_TAIL_ITERS, ".Ltail2")
    rel_b = [*n(c.d1), f"  rel #{OWN + CHAIN_LOCK_B}, r2"] if any(m.lock == "core" for m in c.mids) else []
    rel_a = [*n(c.d2), f"  rel #{OWN + SYNC_LOCK_A}, r2"] if c.releaser == "core" else []
    body = ["  mov r2, #1", "  mov r3, #-1", *n(2),
            f"  acq #{OWN + SYNC_LOCK_SYNC}, r3", *n(LOCK_SETTLE_BUNDLES),
            "  event #0", *rel_b, "  event #1", *rel_a, *n(8),
            *lp.delay_loop(FETCH_TAIL_ITERS, ".Ltail"), *fill]
    return lp.frame(body)


def chain_design_mlir(c: ChainCase) -> str:
    chain_validate(c)
    w = FETCH_WORDS if c.big == FETCH_WORDS else c.big
    bufs = chain_buffers(c)
    lines = []
    for i, (a, n_) in enumerate(bufs):
        name = "head" if i == 0 else f"m{i - 1}"
        init = ", ".join(str(((0x0C + i) << 24) | j) for j in range(n_))
        lines.append(f"    %{name}_buf = aie.buffer(%t02) {{address = {a} : i32, sym_name = \"{name}_buf\", "
                     f"initial_value = dense<[{init}]> : tensor<{n_}xi32>}} : memref<{n_}xi32>")
    pre_words = sum(n_ for _, n_ in bufs)
    locks = [f'    %lA = aie.lock(%t02, {SYNC_LOCK_A}) {{init = 0 : i32, sym_name = "lA"}}',
             f'    %lSync = aie.lock(%t02, {SYNC_LOCK_SYNC}) {{init = 0 : i32, sym_name = "lSync"}}',
             f'    %lStart = aie.lock(%t02, {SYNC_LOCK_START}) {{init = 0 : i32, sym_name = "lStart"}}',
             f'    %lEnd = aie.lock(%t02, {SYNC_LOCK_END}) {{init = 0 : i32, sym_name = "lEnd"}}']
    if any(m.lock == "core" for m in c.mids):
        locks.append(f'    %lB = aie.lock(%t02, {CHAIN_LOCK_B}) {{init = 0 : i32, sym_name = "lB"}}')
    if c.releaser == "s2mm":
        locks.append(f'    %lRx = aie.lock(%t02, {CHAIN_LOCK_RX}) {{init = 1 : i32, sym_name = "lRx"}}')
    bds, free_i = [], 0
    names = ["head", *(f"m{i}" for i in range(len(c.mids))), "send"]

    def bd(label, buf, n_, acq, rel, nxt, bd_id=None):
        s = f"    ^{label}:\n"
        if acq:
            s += f"      aie.use_lock({acq}, AcquireGreaterEqual, %c1)\n"
        pin = "" if bd_id is None else f" {{bd_id = {bd_id} : i32}}"
        s += f"      aie.dma_bd({buf} : memref<{n_}xi32>){pin}\n"
        if rel:
            s += f"      aie.use_lock({rel}, Release, %c1)\n"
        return s + f"      aie.next_bd ^{nxt}\n"

    ids = list(c.bd_ids) or [None] * (len(c.mids) + 2)
    bds.append(bd("head", "%head_buf", c.head, "%lStart", "%lSync", names[1], ids[0]))
    for i, m in enumerate(c.mids):
        lock = None
        if m.lock == "core":
            lock = "%lB"
        elif m.lock == "free":
            lock = f"%lF{free_i}"
            locks.append(f'    {lock} = aie.lock(%t02, {CHAIN_LOCK_FREE0 + free_i}) {{init = 1 : i32, sym_name = "lF{free_i}"}}')
            free_i += 1
        bds.append(bd(f"m{i}", f"%m{i}_buf", m.words, lock, lock, names[i + 2], ids[i + 1]))
    bds.append(bd("send", "%src", w, "%lA", "%lEnd", "end", ids[-1]))
    if c.releaser == "s2mm":
        recv = bd("recv", "%dst", pre_words, "%lRx", "%lA", "recv2") + \
            bd("recv2", "%dst2", w, None, None, "end")
        dst = (f"    %dst = aie.buffer(%t02) {{address = {FETCH_DST_ADDR} : i32, sym_name = \"dst\"}} : memref<{pre_words}xi32>\n"
               f"    %dst2 = aie.buffer(%t02) {{address = {FETCH_DST_ADDR + 0x2000} : i32, sym_name = \"dst2\"}} "
               f": memref<{w}xi32>")
    else:
        recv = bd("recv", "%dst", pre_words + w, None, None, "end")
        dst = (f"    %dst = aie.buffer(%t02) {{address = {FETCH_DST_ADDR} : i32, sym_name = \"dst\"}} "
               f": memref<{pre_words + w}xi32>")
    ramp = ", ".join(str(0x0A000000 | j) for j in range(w))
    ld = ", ".join(str(0x0B000000 | j) for j in range(16))
    pad = "\n".join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32"
                    for _ in range(FETCH_PAD_WRITES))
    nl = "\n"
    return f"""module {{
  aie.device(npu1_1col) {{
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %src = aie.buffer(%t02) {{address = {FETCH_SRC_ADDR} : i32, sym_name = "src", initial_value = dense<[{ramp}]> : tensor<{w}xi32>}} : memref<{w}xi32>
    %ldbuf = aie.buffer(%t02) {{address = {FETCH_LD_ADDR} : i32, sym_name = "ldbuf", initial_value = dense<[{ld}]> : tensor<16xi32>}} : memref<16xi32>
{nl.join(lines)}
{dst}
    %scratch = aie.buffer(%t02) {{address = {FETCH_SCRATCH_ADDR} : i32, sym_name = "scratch"}} : memref<4xi32>
{nl.join(locks)}
    aie.flow(%t02, DMA : 0, %t02, DMA : 0)
    func.func private @k_core(memref<16xi32>) attributes {{link_with = "core.o"}}
    aie.core(%t02) {{
      func.call @k_core(%ldbuf) : (memref<16xi32>) -> ()
      aie.end
    }}
    %mem = aie.mem(%t02) {{
      %c1 = arith.constant 1 : i32
      %s0 = aie.dma_start(MM2S, 0, ^head, ^recv_start)
{"".join(bds)}    ^recv_start:
      %s1 = aie.dma_start(S2MM, 0, ^recv, ^end)
{recv}    ^end:
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


def chain_inject_args(swap: tuple[str, str] | None = None, add: str | None = None) -> list[str]:
    return ["--memmod-sweep-events", ",".join(edit_memmod_events(CHAIN_MEM_EVENTS, swap, add)),
            "--core-grounding", FETCH_CORE_GROUNDING,
            "--core-sweep-events", ",".join(CHAIN_CORE_EVENTS)]


def chain_row(events) -> dict:
    """One run relative to g0, the head BD's grant (the cycle after its START wait, the first MM2S STALLED_LOCK
    run, falls). Every later STALLED_LOCK run, every MM2S/S2MM FINISHED_BD, the lock-A releases and ACQ_GE events
    (memory-module timer) and the first INSTR_EVENT_0/1 (core timer; one time base on this tile) are kept."""
    mem, core = {}, {}
    for e in events:
        if e.row == FETCH_ROW:
            (mem if e.pkt_type == PT_MEM else core).setdefault(e.name, []).append(e.ts)
    runs = _runs(mem.get("DMA_MM2S_0_STALLED_LOCK", []))
    if not runs:
        return {"g0": None}
    g0 = runs[0][1] + 1

    def rel(name, src=mem):
        return [t - g0 for t in sorted(set(src.get(name, [])))]
    ev0, ev1 = rel("INSTR_EVENT_0", core), rel("INSTR_EVENT_1", core)
    return {"g0": g0, "stalls": [(a - g0, b - g0) for a, b in runs[1:]], "fins": rel("DMA_MM2S_0_FINISHED_BD"),
            "rel_a": rel("LOCK_0_REL"), "acq_a": rel("LOCK_SEL0_ACQ_GE"), "s2mm_fins": rel("DMA_S2MM_0_FINISHED_BD"),
            "ev0": ev0[0] if ev0 else None, "ev1": ev1[0] if ev1 else None}


def format_chain_table(rows) -> str:
    """rows: (ChainCase, run, chain_row dict). Every number is cycles after g0 (the head BD's grant)."""
    def ts(v):
        return "[" + " ".join(f"{t:+d}" for t in v) + "]"
    lines = ["case   run      (cycles after g0, the head BD's grant; stalls = later MM2S STALLED_LOCK runs)"]
    for c, run, r in sorted(rows, key=lambda x: (x[0].name, x[1])):
        if r.get("g0") is None:
            lines.append(f"{c.name} {run} no head wait (no STALLED_LOCK run)")
            continue
        st = "[" + " ".join(_interval(a, b) for a, b in r["stalls"]) + "]"
        lines.append(f"{c.name} {run} stalls {st} fins {ts(r['fins'])} relA {ts(r['rel_a'])} acqA {ts(r['acq_a'])} "
                     f"s2mm {ts(r['s2mm_fins'])} ev0 {_num(r['ev0'])} ev1 {_num(r['ev1'])}")
    return "\n".join(lines)


def chain_check_design_dir(out: Path, c: ChainCase) -> None:
    """One design per directory: refuse to mix designs (case names carry only the delays)."""
    f = out / "design.json"
    if f.exists() and json.loads(f.read_text()) != c.design():
        raise ValueError(f"{out} holds design {f.read_text().strip()}, not {c.design()}")


def chain_selection(args) -> list:
    mids = parse_chain_mids(args.mids)
    pairs = [(0, 0)] if args.releaser == "s2mm" else [(a, b) for a in (args.d1s or [0]) for b in (args.d2s or [0])]
    ids = tuple(getattr(args, "bd_ids", None) or ())
    fill = getattr(args, "fill", 0) or 0
    out = [ChainCase(mids, args.head, args.big, a, b, args.releaser, ids, fill) for a, b in pairs]
    for c in out:
        chain_validate(c)
    return out


def cmd_chain_build(args) -> int:
    out = Path(args.out_dir).resolve()
    cases = chain_selection(args)
    chain_check_design_dir(out, cases[0])
    out.mkdir(parents=True, exist_ok=True)
    (out / "design.json").write_text(json.dumps(cases[0].design()) + "\n")
    for c in cases:
        d = out / c.name
        d.mkdir(parents=True, exist_ok=True)
        lp.assemble(chain_core_asm(c), d / "core.o", "k_core")
        (d / "base.mlir").write_text(chain_design_mlir(c))
        lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
               "--out", "aie.mlir", "--trace-mode", "event_time", *chain_inject_args(getattr(args, "memmod_event_swap", None), getattr(args, "memmod_event_add", None)),
               "--trace-config-out", "trace_config.json", "--config-test-name", c.name,
               "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
              cwd=d, log=d / "inject.log")
        lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
               "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
              cwd=d, log=d / "aiecc.log")
        (d / "case.json").write_text(json.dumps({**c.design(), "case": c.name, "d1": c.d1, "d2": c.d2}) + "\n")
        print("built", c.name)
    return 0


def cmd_chain_run(args) -> int:
    out = Path(args.out_dir).resolve()
    cases = chain_selection(args)
    chain_check_design_dir(out, cases[0])
    for c in cases:
        if not (out / c.name).is_dir():
            continue
        if not lp.run_case(out, c.name, args.tag, args.backend, output=False):
            print(f"{c.name}: run did not complete; stop")
            return 1
    return 0


def cmd_chain_table(args) -> int:
    out = Path(args.out_dir).resolve()
    cases = chain_selection(args)
    chain_check_design_dir(out, cases[0])
    rows = []
    for c in cases:
        for run in args.runs:
            run_dir = out / c.name / run
            if not run_dir.is_dir():
                continue
            if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
                rows.append((c, run, {"g0": None}))
                continue
            rows.append((c, run, chain_row(load_level_events(run_dir))))
    print(format_chain_table(rows))
    return 0


def add_chain_parsers(sub) -> None:
    """chain-build / chain-run / chain-table: the lock request/grant timing family (see chain_design_mlir)."""
    cb = sub.add_parser("chain-build", help="lock timing chain: build one design's cases into OUT_DIR")
    cr = sub.add_parser("chain-run", help="lock timing chain: run the built cases to TAG.BACKEND/")
    ct = sub.add_parser("chain-table", help="lock timing chain: stamps relative to the head BD's grant")
    for sp in (cb, cr, ct):
        sp.add_argument("out_dir")
    cr.add_argument("tag")
    cr.add_argument("--backend", choices=("hw", "emu"), required=True)
    ct.add_argument("runs", nargs="+")
    cb.add_argument("--memmod-event-swap", type=parse_event_swap, default=None, metavar="OLD=NEW",
                    help=MEMMOD_SWAP_HELP)
    cb.add_argument("--memmod-event-add", default=None, metavar="NAME", help=MEMMOD_ADD_HELP)
    for sp in (cb, cr, ct):
        sp.add_argument("--mids", nargs="*", default=[], help="mid BDs: words, then c (core lock) or f (free lock)")
        sp.add_argument("--head", type=int, default=4, help="head BD words")
        sp.add_argument("--big", type=int, default=FETCH_WORDS, help="waiting BD words")
        sp.add_argument("--d1s", nargs="*", type=int, default=[], help="core nops before releasing B")
        sp.add_argument("--d2s", nargs="*", type=int, default=[], help="core nops before releasing A")
        sp.add_argument("--releaser", choices=("core", "s2mm"), default="core")
        sp.add_argument("--bd-ids", nargs="*", type=int, default=[],
                        help="bd_id of every MM2S BD in chain order (head, mids, waiting BD); default: aiecc assigns")
        sp.add_argument("--fill", type=int, default=0,
                        help="core releases of A after the tail: memory-module trace packet fillers (core releaser)")


CHAIN_COMMANDS = {"chain-build": cmd_chain_build, "chain-run": cmd_chain_run, "chain-table": cmd_chain_table}


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


# -- ring-order probe (ring-*): each DMA channel against the core on one bank -------------------------
#
# The core camps on one physical bank: n consecutive same-bank loads from the lock-A release, so every
# cycle of the camp is exactly one served access of that bank (the core's, or a DMA's while the core is
# refused). The bank's served-access count therefore advances one per cycle inside the camp, and the
# cycles at which a DMA channel is let in, taken modulo 5 from the camp start, are that channel's winning
# count residues; the cycles it is refused are the others. One DMA channel works on banks 0/1 per variant:
#   m0 / m1  MM2S 0 / MM2S 1 runs the chained lock-fetch task (pre BD, then the lock-A BD reading `src`,
#            banks 0/1) into S2MM 0, which writes `dst` (banks 4/5).
#   s0 / s1  MM2S 0 runs the same task reading `feed` (banks 2/3) instead of `src`; S2MM 0 / S2MM 1 takes
#            the pre BD's 4 words into `dst` and the 64 words into `sdst` (banks 0/1).
# Every variant declares the same buffers with the same initial data, so the bank's history before the
# camp is the same and the residues of different variants are on one common count. `pad` adds that many
# initialised words to bank 0 (buffers at RING_PAD_ADDRS) to test whether image-load writes advance it.
# After the window the core releases lock A RING_FILL_RELEASES more times (memory-module LOCK_0_REL events)
# and runs a long tail, so both trace units fill their last packets and nothing measured is left in a
# partial packet. Nothing waits on the host: the runtime sequence only sets the go lock; the core waits
# only on go and on B, released by the lock-A BD after its 64 words; the S2MM BDs take exactly the 68 words
# the MM2S sends (4 + 64), so the MM2S cannot be back-pressured forever.

# ms: MM2S 0 reads `src` (banks 0/1) and MM2S 1 reads `feed` into S2MM 1, which writes `sdst` (banks 0/1):
# two channels on banks 0/1 at once, both lock-started on lock A (the core releases it by 2).
# mm: MM2S 0 and MM2S 1 run the same chained task on `src` (same schedule, so they ask for the same bank in
# the same cycles) into S2MM 0 (`dst`) and S2MM 1 (`dst2`, banks 6/7).
# mx: as mm, but MM2S 1 reads `src` from its second 16-byte line (60 words), so its granule parity is the
# opposite of MM2S 0's: a one-cycle grant offset then puts both on the same bank in the same cycles.
RING_VARIANTS = ("m0", "m1", "s0", "s1", "ms", "mm", "mx")
RING_DST2_ADDR = 0xC100
RING_FEED_ADDR = 0x4100             # banks 2/3: the s-variants' MM2S source
RING_SDST_ADDR = 0x1000             # banks 0/1: the s-variants' S2MM destination (granule g in bank g & 1)
RING_PAD_ADDRS = (0x0C00, 0x0C20)   # bank 0 granules (bit 4 clear): up to 4 + 4 initialised words
RING_MAX_PAD = 8
RING_MAX_PAD_ONE = 12               # one contiguous buffer at RING_PAD_ADDRS[0]: granules bank 0, 1, 0
RING_MAX_N = 120
RING_FILL_RELEASES = 16
RING_TAIL_ITERS = 800
RING_CAMP_OFFSETS = {0: (0, 4, 8, 12, 32, 36, 40, 44), 1: (16, 20, 24, 28, 48, 52, 56, 60)}


@dataclass(frozen=True)
class RingCase:
    variant: str                   # m0, m1, s0, s1: which channel works on banks 0/1
    bank: int = 0                  # the camped physical bank
    n: int = 90                    # camp length (consecutive same-bank loads)
    k: int = 0                     # nops between the release and the camp
    pad: int = 0                   # extra initialised words in bank 0
    pad_one: bool = False          # the pad words as one contiguous buffer (crossing into bank 1 after 4)

    @property
    def name(self) -> str:
        return f"rg{self.variant}b{self.bank}n{self.n:03d}k{self.k:02d}p{self.pad}{'o' if self.pad_one else ''}"

    @property
    def mm2s(self) -> int:
        """The lock-started MM2S channel (its STALLED_LOCK gives the grant)."""
        return 1 if self.variant == "m1" else 0

    @property
    def s2mm(self) -> int:
        return 1 if self.variant == "s1" else 0


def ring_validate(c: RingCase) -> None:
    if c.variant not in RING_VARIANTS:
        raise ValueError(f"variant must be one of {RING_VARIANTS}")
    if c.bank not in (0, 1):
        raise ValueError("bank must be 0 or 1")
    if not 1 <= c.n <= RING_MAX_N:
        raise ValueError(f"n must be 1..{RING_MAX_N}")
    if c.k not in FETCH_KS:
        raise ValueError(f"k must be 0..{FETCH_KS[-1]}")
    if not 0 <= c.pad <= (RING_MAX_PAD_ONE if c.pad_one else RING_MAX_PAD):
        raise ValueError(f"pad must be 0..{RING_MAX_PAD_ONE if c.pad_one else RING_MAX_PAD}")


def ring_core_asm(c: RingCase) -> str:
    """go, settle, `event #0`, release A, k nops, n same-bank loads, `event #1`, wait B, then
    RING_FILL_RELEASES spaced releases of A and a long tail (trace packet fillers)."""
    ring_validate(c)
    n = lp.nops
    offs = RING_CAMP_OFFSETS[c.bank]
    camp = [f"  lda {ARB_LOAD_REGS[i % len(ARB_LOAD_REGS)]}, [p0, #{offs[i % len(offs)]}]" for i in range(c.n)]
    fill = []
    for i in range(RING_FILL_RELEASES):
        fill += [*lp.delay_loop(20, f".Lfill{i}"), f"  rel #{OWN + FETCH_LOCK_A}, r2"]
    grants = 2 if c.variant in ("ms", "mm", "mx") else 1          # lock-A BDs waiting for the release
    body = ["  mov r2, #1", "  mov r3, #-1", f"  mov r4, #{grants}", *n(2),
            f"  acq #{OWN + FETCH_LOCK_GO}, r3", *n(4),
            *lp.delay_loop(FETCH_SETTLE_ITERS, ".Lsettle"), *n(8),
            "  event #0", f"  rel #{OWN + FETCH_LOCK_A}, r4", *n(c.k),
            *camp, *n(ARB_EVENT1_NOPS), "  event #1", *n(8),
            f"  acq #{OWN + FETCH_LOCK_B}, r3", *n(8), *fill,
            *lp.delay_loop(RING_TAIL_ITERS, ".Ltail")]
    return lp.frame(body)


def ring_design_mlir(c: RingCase) -> str:
    """The chained lock-fetch design with the channel of variant c on banks 0/1 (see the section note)."""
    ring_validate(c)
    w, p = FETCH_WORDS, FETCH_PRE_WORDS
    ramp = ", ".join(str(0x0A000000 | j) for j in range(w))
    feed = ", ".join(str(0x0E000000 | j) for j in range(w))
    ld = ", ".join(str(0x0B000000 | j) for j in range(16))
    pre_init = ", ".join(str(0x0C000000 | j) for j in range(p))
    pads = []
    left = c.pad
    if c.pad_one and c.pad:
        init = ", ".join(str(0x0F000000 | j) for j in range(c.pad))
        pads.append(f"    %pad0 = aie.buffer(%t02) {{address = {RING_PAD_ADDRS[0]} : i32, sym_name = \"pad0\", "
                    f"initial_value = dense<[{init}]> : tensor<{c.pad}xi32>}} : memref<{c.pad}xi32>\n")
        left = 0
    for i, addr in enumerate(RING_PAD_ADDRS):
        words = min(left, 4)
        if words:
            init = ", ".join(str(0x0F000000 | (4 * i + j)) for j in range(words))
            pads.append(f"    %pad{i} = aie.buffer(%t02) {{address = {addr} : i32, sym_name = \"pad{i}\", "
                        f"initial_value = dense<[{init}]> : tensor<{words}xi32>}} : memref<{words}xi32>\n")
        left -= words
    pad_text = "".join(pads)
    pad = "\n".join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32"
                    for _ in range(FETCH_PAD_WRITES))
    source = "%feed" if c.variant in ("s0", "s1") else "%src"
    if c.variant in ("s0", "s1"):
        recv = (f"    ^recv:\n      aie.dma_bd(%dst : memref<{w + p}xi32> offset = 0 len = {p})\n      aie.next_bd ^recv2\n"
                f"    ^recv2:\n      aie.dma_bd(%sdst : memref<{w}xi32> offset = 0 len = {w})\n      aie.next_bd ^end\n")
    else:
        recv = f"    ^recv:\n      aie.dma_bd(%dst : memref<{w + p}xi32> offset = 0 len = {w + p})\n      aie.next_bd ^end\n"
    second_flow, second_task, second_lock = "", "", ""
    if c.variant == "ms":
        # a BD carries both lock fields or neither: MM2S 1's releases an otherwise unused END lock
        second_lock = f"\n    %lEnd = aie.lock(%t02, {SYNC_LOCK_END}) {{init = 0 : i32, sym_name = \"lEnd\"}}"
        # MM2S 1: one lock-A BD (no pre BD) reading `feed` into S2MM 1, which writes all 64 words to `sdst`
        second_flow = "\n    aie.flow(%t02, DMA : 1, %t02, DMA : 1)"
        second_task = (f"    ^m1_start:\n      %s2 = aie.dma_start(MM2S, 1, ^send1, ^s1_start)\n"
                       f"    ^send1:\n      %ca1 = arith.constant 1 : i32\n      aie.use_lock(%lA, AcquireGreaterEqual, %ca1)\n"
                       f"      aie.dma_bd(%feed : memref<{w}xi32>)\n"
                       f"      %ce1 = arith.constant 1 : i32\n      aie.use_lock(%lEnd, Release, %ce1)\n"
                       f"      aie.next_bd ^end\n"
                       f"    ^s1_start:\n      %s3 = aie.dma_start(S2MM, 1, ^recv1, ^end)\n"
                       f"    ^recv1:\n      aie.dma_bd(%sdst : memref<{w}xi32> offset = 0 len = {w})\n      aie.next_bd ^end\n")
    dst2 = ""
    if c.variant in ("mm", "mx"):
        m1_bd = f"%src : memref<{w}xi32>" if c.variant == "mm" else f"%src : memref<{w}xi32> offset = 4 len = {w - 4}"
        s1_len = w + p if c.variant == "mm" else w
        second_lock = f"\n    %lEnd = aie.lock(%t02, {SYNC_LOCK_END}) {{init = 0 : i32, sym_name = \"lEnd\"}}"
        second_flow = "\n    aie.flow(%t02, DMA : 1, %t02, DMA : 1)"
        dst2 = (f"\n    %dst2 = aie.buffer(%t02) {{address = {RING_DST2_ADDR} : i32, sym_name = \"dst2\"}} "
                f": memref<{w + p}xi32>")
        second_task = (f"    ^m1_start:\n      %s2 = aie.dma_start(MM2S, 1, ^pre1, ^s1_start)\n"
                       f"    ^pre1:\n      aie.dma_bd(%pre : memref<{p}xi32>)\n      aie.next_bd ^send1\n"
                       f"    ^send1:\n      %ca1 = arith.constant 1 : i32\n      aie.use_lock(%lA, AcquireGreaterEqual, %ca1)\n"
                       f"      aie.dma_bd({m1_bd})\n"
                       f"      %ce1 = arith.constant 1 : i32\n      aie.use_lock(%lEnd, Release, %ce1)\n"
                       f"      aie.next_bd ^end\n"
                       f"    ^s1_start:\n      %s3 = aie.dma_start(S2MM, 1, ^recv1, ^end)\n"
                       f"    ^recv1:\n      aie.dma_bd(%dst2 : memref<{w + p}xi32> offset = 0 len = {s1_len})\n"
                       f"      aie.next_bd ^end\n")
    return f"""module {{
  aie.device(npu1_1col) {{
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %src = aie.buffer(%t02) {{address = {FETCH_SRC_ADDR} : i32, sym_name = "src", initial_value = dense<[{ramp}]> : tensor<{w}xi32>}} : memref<{w}xi32>
    %ldbuf = aie.buffer(%t02) {{address = {FETCH_LD_ADDR} : i32, sym_name = "ldbuf", initial_value = dense<[{ld}]> : tensor<16xi32>}} : memref<16xi32>
{pad_text}    %sdst = aie.buffer(%t02) {{address = {RING_SDST_ADDR} : i32, sym_name = "sdst"}} : memref<{w}xi32>
    %pre = aie.buffer(%t02) {{address = {FETCH_PRE_ADDR} : i32, sym_name = "pre", initial_value = dense<[{pre_init}]> : tensor<{p}xi32>}} : memref<{p}xi32>
    %feed = aie.buffer(%t02) {{address = {RING_FEED_ADDR} : i32, sym_name = "feed", initial_value = dense<[{feed}]> : tensor<{w}xi32>}} : memref<{w}xi32>
    %dst = aie.buffer(%t02) {{address = {FETCH_DST_ADDR} : i32, sym_name = "dst"}} : memref<{w + p}xi32>
    %scratch = aie.buffer(%t02) {{address = {FETCH_SCRATCH_ADDR} : i32, sym_name = "scratch"}} : memref<4xi32>{dst2}
    %lA = aie.lock(%t02, {FETCH_LOCK_A}) {{init = 0 : i32, sym_name = "lA"}}
    %lB = aie.lock(%t02, {FETCH_LOCK_B}) {{init = 0 : i32, sym_name = "lB"}}
    %lGo = aie.lock(%t02, {FETCH_LOCK_GO}) {{init = 0 : i32, sym_name = "lGo"}}{second_lock}
    aie.flow(%t02, DMA : {c.mm2s}, %t02, DMA : {c.s2mm}){second_flow}
    func.func private @k_core(memref<16xi32>) attributes {{link_with = "core.o"}}
    aie.core(%t02) {{
      func.call @k_core(%ldbuf) : (memref<16xi32>) -> ()
      aie.end
    }}
    %mem = aie.mem(%t02) {{
      %s0 = aie.dma_start(MM2S, {c.mm2s}, ^pre, ^recv_start)
    ^pre:
      aie.dma_bd(%pre : memref<{p}xi32>)
      aie.next_bd ^send
    ^send:
      %ca = arith.constant 1 : i32
      aie.use_lock(%lA, AcquireGreaterEqual, %ca)
      aie.dma_bd({source} : memref<{w}xi32>)
      %cb = arith.constant 1 : i32
      aie.use_lock(%lB, Release, %cb)
      aie.next_bd ^end
    ^recv_start:
      %s1 = aie.dma_start(S2MM, {c.s2mm}, ^recv, {"^m1_start" if c.variant in ("ms", "mm", "mx") else "^end"})
{recv}{second_task}    ^end:
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


def ring_memmod_events(c: RingCase) -> list[str]:
    if c.variant == "mx":
        return ["LOCK_0_REL", "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_1_STALLED_LOCK", "DMA_MM2S_0_FINISHED_BD",
                "DMA_MM2S_1_FINISHED_BD", "DMA_S2MM_0_FINISHED_BD", "DMA_S2MM_1_FINISHED_BD", "CONFLICT_DM_BANK_0"]
    if c.variant == "mm":
        return ["LOCK_0_REL", "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_0_FINISHED_BD", "DMA_MM2S_1_FINISHED_BD",
                "DMA_MM2S_0_MEMORY_STARVATION", "DMA_MM2S_1_MEMORY_STARVATION", "CONFLICT_DM_BANK_0",
                "CONFLICT_DM_BANK_1"]
    if c.variant == "ms":
        return ["LOCK_0_REL", "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_0_FINISHED_BD", "DMA_MM2S_1_FINISHED_BD",
                "DMA_S2MM_0_FINISHED_BD", "DMA_S2MM_1_FINISHED_BD", "CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1"]
    m, s = c.mm2s, c.s2mm
    ev = ["LOCK_0_REL", f"DMA_MM2S_{m}_STALLED_LOCK", f"DMA_MM2S_{m}_FINISHED_BD", f"DMA_S2MM_{s}_FINISHED_BD",
          "CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1"]
    if c.variant in ("s0", "s1"):
        ev += [f"DMA_S2MM_{s}_MEMORY_BACKPRESSURE", f"DMA_MM2S_{m}_STREAM_BACKPRESSURE"]
    else:
        ev += [f"DMA_MM2S_{m}_MEMORY_STARVATION"]
    return ev


def ring_inject_args(c: RingCase) -> list[str]:
    return ["--memmod-sweep-events", ",".join(ring_memmod_events(c)),
            "--core-grounding", FETCH_CORE_GROUNDING,
            "--core-sweep-events", ",".join(ARB_CORE_EVENTS)]


def ring_row(events, c: RingCase) -> dict:
    """One run: grant (cycles after the release), conflict runs on each bank, the cycles the core lost (from
    MEMORY_STALL: core stamp minus 1, the offset measured in the arb probe), finishes, back-pressure runs.
    All cycles are relative to the camp start (release + k + 1, the first load's cycle)."""
    mem: dict = {}
    core: dict = {}
    for e in events:
        if e.row == FETCH_ROW:
            (mem if e.pkt_type == PT_MEM else core).setdefault(e.name, []).append(e.ts)
    rels = sorted(mem.get("LOCK_0_REL", []))
    if not rels:
        return {"release": None}
    rel = rels[0]
    t0 = rel + c.k + 1
    r: dict = {"release": rel, "grant": None, "camp_start": t0}
    for first, last in _runs(mem.get(f"DMA_MM2S_{c.mm2s}_STALLED_LOCK", [])):
        if first <= rel <= last + 1:
            r["grant"] = last + 1 - rel
    r["lost"] = [t - 1 - t0 for t in sorted(set(core.get("MEMORY_STALL", [])))]
    r["runs"] = {b: [(a - t0, z - t0) for a, z in _runs(mem.get(f"CONFLICT_DM_BANK_{b}", []))] for b in (0, 1)}
    r["mm2s_fin"] = [t - t0 for t in sorted(set(mem.get(f"DMA_MM2S_{c.mm2s}_FINISHED_BD", []))) if t > rel]
    r["s2mm_fin"] = [t - t0 for t in sorted(set(mem.get(f"DMA_S2MM_{c.s2mm}_FINISHED_BD", []))) if t > rel]
    r["finished"] = {name.replace("DMA_", "").replace("_FINISHED_BD", ""): [t - t0 for t in sorted(set(ts)) if t > rel]
                     for name, ts in mem.items() if name.endswith("_FINISHED_BD")}
    r["backpressure"] = {name: [(a - t0, z - t0) for a, z in _runs(mem.get(name, [])) if z >= rel]
                         for name in (f"DMA_S2MM_{c.s2mm}_MEMORY_BACKPRESSURE", f"DMA_MM2S_{c.mm2s}_STREAM_BACKPRESSURE",
                                      "DMA_MM2S_0_MEMORY_STARVATION", "DMA_MM2S_1_MEMORY_STARVATION") if name in mem}
    ev0, ev1 = sorted(core.get("INSTR_EVENT_0", [])), sorted(core.get("INSTR_EVENT_1", []))
    r["window"] = ev1[0] - ev0[0] if ev0 and ev1 else None
    return r


def ring_residues(r: dict, c: RingCase, period: int = ARB_PERIOD) -> dict:
    """Inside the camp (cycles 0..n-1 + losses from the camp start) every cycle is one served access, so the
    count residue of cycle t is t mod period (plus a constant common to all variants). Returns, for the
    camped bank, the residues at which the DMA was let in (a contended cycle the core lost) and at which it
    was refused (a contended cycle the core won), and contended runs that ended without a core loss inside
    the camp (the DMA withdrew)."""
    end = c.n + len(r["lost"])
    lost = set(r["lost"])
    won, refused, withdrew = set(), set(), []
    for a, z in r["runs"][c.bank]:
        if a < 0 or z >= end:
            continue
        for t in range(a, z + 1):
            (won if t in lost else refused).add(t % period)
        if z not in lost:
            withdrew.append((a, z))
    return {"dma_wins": sorted(won), "dma_refused": sorted(refused), "withdrew": withdrew}


def ring_plan() -> list[RingCase]:
    cases = [RingCase(v) for v in ("m0", "m1", "s0", "s1")]
    cases += [RingCase("m0", pad=p) for p in range(1, 6)]
    # second batch: image-write unit (one buffer over two bank-0 granules), two channels with and without
    # the core camping
    cases += [RingCase("m0", pad=12, pad_one=True), RingCase("ms"), RingCase("ms", n=1)]
    # third batch: two MM2S channels on one schedule, core camping and silent
    cases += [RingCase("mm", n=120), RingCase("mm", n=1)]
    # fourth batch: two MM2S channels on opposite granule parity, core silent (forces a same-bank meeting
    # when the two grants are one cycle apart)
    cases += [RingCase("mx", n=1)]
    return cases


def ring_selection(args) -> list[RingCase]:
    return [c for c in ring_plan() if not args.names or c.name in args.names]


def ring_build_case(c: RingCase, out: Path) -> None:
    d = out / c.name
    d.mkdir(parents=True, exist_ok=True)
    lp.assemble(ring_core_asm(c), d / "core.o", "k_core")
    (d / "base.mlir").write_text(ring_design_mlir(c))
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time", *ring_inject_args(c),
           "--trace-config-out", "trace_config.json", "--config-test-name", c.name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    (d / "case.json").write_text(json.dumps({"case": c.name, "variant": c.variant, "bank": c.bank, "n": c.n,
                                             "k": c.k, "pad": c.pad, "ring": True}) + "\n")


def format_ring_table(rows) -> str:
    lines = ["case               run    grant  lost(camp+)                       runs on the camped bank (start..end)"
             "                  residues: DMA in / refused   withdrew  mm2s_fin s2mm_fin backpressure"]
    for c, run, r in sorted(rows, key=lambda x: (x[0].name, x[1])):
        if r.get("release") is None or r.get("grant") is None:
            lines.append(f"{c.name:<18} {run:<6} no release or grant")
            continue
        res = ring_residues(r, c)
        runs = " ".join(f"{a}" if a == z else f"{a}..{z}" for a, z in r["runs"][c.bank])
        bp = " ".join(f"{k.replace('DMA_', '')}:{','.join(f'{a}..{z}' for a, z in v)}"
                      for k, v in r["backpressure"].items() if v)
        lines.append(f"{c.name:<18} {run:<6} +{r['grant']}    {','.join(map(str, r['lost'])) or '-':<33} "
                     f"{runs:<44} {res['dma_wins']} / {res['dma_refused']}  {res['withdrew'] or '-'}  "
                     f"{r['mm2s_fin']} {r['s2mm_fin']} {bp or '-'}"
                     + (f" fins {r['finished']}" if c.variant in ("ms", "mm", "mx") else ""))
    return "\n".join(lines)


def cmd_ring_build(args) -> int:
    out = Path(args.out_dir).resolve()
    for c in ring_selection(args):
        ring_build_case(c, out)
        print("built", c.name)
    return 0


def cmd_ring_run(args) -> int:
    out = Path(args.out_dir).resolve()
    for c in ring_selection(args):
        if not (out / c.name / "aie.xclbin").exists():
            continue
        if not lp.run_case(out, c.name, args.tag, args.backend, output=False):
            print(f"{c.name}: run did not complete; stop")
            return 1
    return 0


def cmd_ring_table(args) -> int:
    out = Path(args.out_dir).resolve()
    rows = []
    for c in ring_selection(args):
        for run in args.runs:
            run_dir = out / c.name / run
            if not run_dir.is_dir():
                continue
            if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
                rows.append((c, run, {"release": None}))
                continue
            rows.append((c, run, ring_row(load_level_events(run_dir), c)))
    print(format_ring_table(rows))
    return 0


# -- lock-held S2MM probe (sl-*): the S2MM grant after a core release ------------------------------------
#
# The compute tile's S2MM side of the lock-grant timing: how many cycles after a core's release of lock A does an
# S2MM BD parked on A get its grant (STALLED_LOCK falls), how long after the grant does it finish, when does the
# MM2S feeding it stop backpressuring, and where do the stream-rate writes fall against a core load or camp on the
# bank the S2MM writes. Everything is read from the memory-module trace unit (one timer, no cross-unit skew).
#
#   MM2S 0: one lock-free BD (the `feed` buffer, banks 2/3), started at boot, into the tile's own S2MM 0 over the
#           local loopback. It sends exactly what the S2MM takes: 68 words with a head BD, 64 without.
#   S2MM 0: with a head (`slh`): a lock-free 4-word head BD (into `dst`) chained to the 64-word BD, so the measured BD
#           is a chained BD; without (`slf`): the 64-word BD is the task's first BD. The 64-word BD acquires lock A
#           (lock 0, init 0, so LOCK_0_REL follows it at the selector's reset), writes `sdst` (0x1000, banks 0/1: granule
#           g in physical bank g & 1) and releases lock B when it finishes.
#   core:   wait on the go lock (set by the runtime sequence after the trace is armed), settle, `event #0`, d nops,
#           release A; then nothing (`none`), one load from `ldbuf` k nops later at granule parity p (`load`: offset
#           16 * p, physical bank p), or the 90-load camp on bank 0 (`camp`); then wait on B (the S2MM's release) and
#           release A RING_FILL_RELEASES more times plus a tail, which fill the last trace packets (as the ring family).
# Variants of the lock BD: `n` = its word count (first-BD variant swept over 4..20; the MM2S then sends exactly the
# BDs' words, so there is no surplus word and no unfinished task), `ch` = the S2MM channel (the loopback routes
# MM2S 0 to it; the camp is also run on S2MM 1, whose events replace S2MM 0's in the memory-module list).
# The trace stops when the runtime sequence ends (the stop broadcast), FETCH_PAD_WRITES command-processor writes after
# the go lock; the core's fill releases and tail run longer than the S2MM's finish, and the table's `tail` column is
# the last LOCK_0_REL minus the S2MM FINISHED_BD (positive = the trace reaches past the finish and the core's
# follow-up). The S2MM BD's own release of lock B is not a traced event (all eight slots are taken, selector 1 is
# unprogrammed); the core getting past its wait on B and running its fill releases proves it happened.
# Why it cannot hang: the core waits only on go (set by the sequence, which always reaches it) and on B, released by the
# S2MM BD after the 64 words the MM2S sends; the MM2S is lock-free and the S2MM takes exactly its words; the sequence
# waits on nothing in the tile.

SL_FIRST_D = 10                    # the release sweep: SL_DS = six consecutive d (the sync family's banked rows)
SL_DS = tuple(range(SL_FIRST_D, SL_FIRST_D + 6))
SL_LOAD_D = 12                     # the d of the k sweep and of the camp
SL_KS = tuple(range(25))
SL_PARITIES = (0, 1)
SL_CAMP_N = 90
SL_KINDS = ("none", "load", "camp")
SL_HEAD_WORDS = 4
SL_HEAD_ADDR = FETCH_DST_ADDR      # the head BD's destination: clear of banks 0..3
SL_MEMMOD_EVENTS = ["LOCK_0_REL", "DMA_S2MM_0_STALLED_LOCK", "DMA_S2MM_0_FINISHED_BD",
                    "DMA_MM2S_0_STREAM_BACKPRESSURE", "CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1",
                    "DMA_S2MM_0_STREAM_STARVATION", "DMA_S2MM_0_MEMORY_BACKPRESSURE"]
SL_CORE_EVENTS = ["INSTR_EVENT_0", "LOCK_STALL"]
SL_NS = (4, 8, 12, 16, 20)        # the first-BD variant's held word counts: no banked silicon row has 5..20


@dataclass(frozen=True)
class SlCase:
    head: bool                     # True: S2MM head BD chained to the lock BD; False: the lock BD is the first BD
    d: int                         # nops between `event #0` and the release of A
    kind: str = "none"             # "none", "load" (one load k nops after the release) or "camp"
    k: int = 0
    p: int = 0                     # the load's granule parity = its physical bank (offset 16 * p)
    n: int = FETCH_WORDS           # words of the lock-held BD
    ch: int = 0                    # the S2MM channel (the loopback routes MM2S 0 to it)

    @property
    def name(self) -> str:
        stem = f"sl{'h' if self.head else 'f'}{self.d:02d}"
        if self.kind == "load":
            stem += f"k{self.k:02d}p{self.p}"
        elif self.kind == "camp":
            stem += "camp"
        return stem + (f"n{self.n:02d}" if self.n != FETCH_WORDS else "") + (f"s{self.ch}" if self.ch else "")

    @property
    def words(self) -> int:
        """What the MM2S sends and the S2MM takes in all: exactly the BDs' words, so no DMA task is left
        unfinished and no surplus word is sent."""
        return self.n + (SL_HEAD_WORDS if self.head else 0)


def sl_validate(c: SlCase) -> None:
    if not 0 <= c.d <= SYNC_MAX_D:
        raise ValueError(f"d must be 0..{SYNC_MAX_D}")
    if not 1 <= c.n <= FETCH_WORDS:
        raise ValueError(f"n must be 1..{FETCH_WORDS}")
    if c.ch not in (0, 1):
        raise ValueError("ch must be 0 or 1")
    if c.kind not in SL_KINDS:
        raise ValueError(f"kind must be one of {SL_KINDS}")
    if c.kind == "load":
        if c.k not in FETCH_KS:
            raise ValueError(f"k must be 0..{FETCH_KS[-1]}")
        if c.p not in SL_PARITIES:
            raise ValueError("p must be 0 or 1")


def sl_core_asm(c: SlCase) -> str:
    sl_validate(c)
    n = lp.nops
    if c.kind == "load":
        after = [*n(c.k), f"  lda r26, [p0, #{16 * c.p}]"]
    elif c.kind == "camp":
        offs = RING_CAMP_OFFSETS[0]
        after = [f"  lda {ARB_LOAD_REGS[i % len(ARB_LOAD_REGS)]}, [p0, #{offs[i % len(offs)]}]"
                 for i in range(SL_CAMP_N)]
    else:
        after = []
    fill = []
    for i in range(RING_FILL_RELEASES):
        fill += [*lp.delay_loop(20, f".Lfill{i}"), f"  rel #{OWN + FETCH_LOCK_A}, r2"]
    body = ["  mov r2, #1", "  mov r3, #-1", *n(2),
            f"  acq #{OWN + FETCH_LOCK_GO}, r3", *n(4),
            *lp.delay_loop(FETCH_SETTLE_ITERS, ".Lsettle"), *n(8),
            "  event #0", *n(c.d), f"  rel #{OWN + FETCH_LOCK_A}, r2", *after, *n(8),
            f"  acq #{OWN + FETCH_LOCK_B}, r3", *n(8), *fill,
            *lp.delay_loop(RING_TAIL_ITERS, ".Ltail")]
    return lp.frame(body)


def sl_design_mlir(c: SlCase) -> str:
    sl_validate(c)
    w, h = FETCH_WORDS, SL_HEAD_WORDS
    feed = ", ".join(str(0x0E000000 | j) for j in range(c.words))
    ld = ", ".join(str(0x0B000000 | j) for j in range(16))
    pad = "\n".join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32"
                    for _ in range(FETCH_PAD_WRITES))
    head_bd = (f"    ^head:\n      aie.dma_bd(%dst : memref<{h}xi32>)\n      aie.next_bd ^recv\n") if c.head else ""
    return f"""module {{
  aie.device(npu1_1col) {{
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %ldbuf = aie.buffer(%t02) {{address = {FETCH_LD_ADDR} : i32, sym_name = "ldbuf", initial_value = dense<[{ld}]> : tensor<16xi32>}} : memref<16xi32>
    %sdst = aie.buffer(%t02) {{address = {RING_SDST_ADDR} : i32, sym_name = "sdst"}} : memref<{w}xi32>
    %feed = aie.buffer(%t02) {{address = {RING_FEED_ADDR} : i32, sym_name = "feed", initial_value = dense<[{feed}]> : tensor<{c.words}xi32>}} : memref<{c.words}xi32>
    %dst = aie.buffer(%t02) {{address = {SL_HEAD_ADDR} : i32, sym_name = "dst"}} : memref<{h}xi32>
    %scratch = aie.buffer(%t02) {{address = {FETCH_SCRATCH_ADDR} : i32, sym_name = "scratch"}} : memref<4xi32>
    %lA = aie.lock(%t02, {FETCH_LOCK_A}) {{init = 0 : i32, sym_name = "lA"}}
    %lB = aie.lock(%t02, {FETCH_LOCK_B}) {{init = 0 : i32, sym_name = "lB"}}
    %lGo = aie.lock(%t02, {FETCH_LOCK_GO}) {{init = 0 : i32, sym_name = "lGo"}}
    aie.flow(%t02, DMA : 0, %t02, DMA : {c.ch})
    func.func private @k_core(memref<16xi32>) attributes {{link_with = "core.o"}}
    aie.core(%t02) {{
      func.call @k_core(%ldbuf) : (memref<16xi32>) -> ()
      aie.end
    }}
    %mem = aie.mem(%t02) {{
      %s0 = aie.dma_start(MM2S, 0, ^send, ^recv_start)
    ^send:
      aie.dma_bd(%feed : memref<{c.words}xi32>)
      aie.next_bd ^end
    ^recv_start:
      %s1 = aie.dma_start(S2MM, {c.ch}, {"^head" if c.head else "^recv"}, ^end)
{head_bd}    ^recv:
      %ca = arith.constant 1 : i32
      aie.use_lock(%lA, AcquireGreaterEqual, %ca)
      aie.dma_bd(%sdst : memref<{w}xi32>{"" if c.n == w else f" offset = 0 len = {c.n}"})
      %cb = arith.constant 1 : i32
      aie.use_lock(%lB, Release, %cb)
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


def sl_memmod_events(c: SlCase) -> list[str]:
    """The memory-module list: SL_MEMMOD_EVENTS with the S2MM events of the case's channel. STREAM_STARVATION
    stays in every case (its fall dates the first word's entry)."""
    return [e.replace("DMA_S2MM_0_", f"DMA_S2MM_{c.ch}_") for e in SL_MEMMOD_EVENTS]


def sl_inject_args(c: SlCase) -> list[str]:
    return ["--memmod-sweep-events", ",".join(sl_memmod_events(c)),
            "--core-grounding", FETCH_CORE_GROUNDING,
            "--core-sweep-events", ",".join(SL_CORE_EVENTS)]


def sl_plan() -> list[SlCase]:
    """71 cases: the release sweep (six d, no core access) on both variants, the k x parity sweep at one d on the
    chained variant, the camp at that d on both variants, the held word count N on the first-BD variant, and the
    camp on S2MM 1 on both variants."""
    cases = [SlCase(head, d) for head in (True, False) for d in SL_DS]
    cases += [SlCase(True, SL_LOAD_D, "load", k, p) for k in SL_KS for p in SL_PARITIES]
    cases += [SlCase(head, SL_LOAD_D, "camp") for head in (True, False)]
    cases += [SlCase(False, SL_LOAD_D, n=n) for n in SL_NS]              # held word count, first-BD variant
    cases += [SlCase(head, SL_LOAD_D, "camp", ch=1) for head in (True, False)]    # the camp on S2MM 1
    return cases


def sl_row(events, ch: int = 0) -> dict:
    """One run as numbers relative to the release and the S2MM grant, memory-module timer only.

    grant = cycles from the first LOCK_0_REL to the cycle after the S2MM STALLED_LOCK interval holding it ends.
    Every other cycle count is relative to that grant cycle G: S2MM FINISHED_BD after the release (the head BD's
    finish is before it), the MM2S STREAM_BACKPRESSURE fall (bp_fall: the first run still asserted at the release,
    ending one cycle before), STREAM_STARVATION and MEMORY_BACKPRESSURE runs, conflict stamps per bank. The MM2S's own
    FINISHED_BD is not traced: its one BD is absorbed by the path (about 60 words) and ends at boot, before the trace
    is armed, so the MM2S finishing shows only as its backpressure falling.
    `releases` counts LOCK_0_REL: the core's one release plus its RING_FILL_RELEASES after it got B."""
    mem: dict = {}
    for e in events:
        if e.row == FETCH_ROW and e.pkt_type == PT_MEM:
            mem.setdefault(e.name, []).append(e.ts)
    rels = sorted(set(mem.get("LOCK_0_REL", [])))
    if not rels:
        return {"release": None}
    rel = rels[0]
    r: dict = {"release": rel, "grant": None, "releases": len(rels), "s2mm_fin": [],
               "bp_fall": None, "starvation": [], "membp": [], "conflicts": {0: [], 1: []}}
    for first, last in _runs(mem.get(f"DMA_S2MM_{ch}_STALLED_LOCK", [])):
        if first <= rel <= last + 1:
            r["grant"] = last + 1 - rel
    if r["grant"] is None:
        return r
    G = rel + r["grant"]
    r["s2mm_fin"] = [t - G for t in sorted(set(mem.get(f"DMA_S2MM_{ch}_FINISHED_BD", []))) if t > rel]
    bp = [(a, z) for a, z in _runs(mem.get("DMA_MM2S_0_STREAM_BACKPRESSURE", [])) if z + 1 > rel]
    r["bp_fall"] = bp[0][1] + 1 - G if bp else None
    r["starvation"] = [(a - G, z - G) for a, z in _runs(mem.get(f"DMA_S2MM_{ch}_STREAM_STARVATION", [])) if z >= rel]
    r["membp"] = [(a - G, z - G) for a, z in _runs(mem.get(f"DMA_S2MM_{ch}_MEMORY_BACKPRESSURE", [])) if z >= rel]
    for bank in (0, 1):
        r["conflicts"][bank] = [t - G for t in sorted(set(mem.get(f"CONFLICT_DM_BANK_{bank}", []))) if t >= rel]
    # the trace's reach past the S2MM's finish: the core's last release of A (the trace stops when the runtime
    # sequence ends, long after the core's fill releases) minus the finish
    r["tail"] = rels[-1] - (G + r["s2mm_fin"][0]) if r["s2mm_fin"] else None
    return r


def sl_missing(r: dict, c: SlCase) -> list[str]:
    """What a run lacks of the finish check: the release, the grant, the S2MM lock BD's one FINISHED_BD after the
    release, and the core's fill releases (it got past the wait on B, so it ended). The MM2S needs no check of
    its own: the S2MM takes exactly the words it sends, so the S2MM's finish means the MM2S's task finished (its
    backpressure only falls when the words outnumber what the path absorbs, so it is a column, not a check)."""
    if r.get("release") is None:
        return ["no LOCK_0_REL"]
    out = []
    if r["grant"] is None:
        return ["grant"]
    if len(r["s2mm_fin"]) != 1:
        out.append(f"S2MM FINISHED_BD x{len(r['s2mm_fin'])} (want 1)")
    if r["tail"] is not None and r["tail"] <= 0:
        out.append("trace ends before the S2MM's lock release was followed by the core")
    if r["releases"] != 1 + RING_FILL_RELEASES:
        out.append(f"core did not reach its end (LOCK_0_REL x{r['releases']}, want {1 + RING_FILL_RELEASES})")
    return out


def _sl_runs(runs) -> str:
    return ",".join(f"+{a}" if a == z else f"+{a}..+{z}" for a, z in runs) or "-"


def format_sl_table(rows) -> str:
    """rows: (SlCase, run name, sl_row dict). Every cycle count is relative to the S2MM grant except `grant` (after
    the release); the release is the memory-module timer stamp."""
    lines = ["case          run      release  grant     S2MM fin  bp fall  starv           B0 conflicts"
             "                B1 conflicts                tail (trace past the finish)  finish"]
    for c, run, r in sorted(rows, key=lambda x: (x[0].name, x[1])):
        if r["release"] is None:
            lines.append(f"{c.name:<13} {run:<8} no LOCK_0_REL event")
            continue
        if r["grant"] is None:
            lines.append(f"{c.name:<13} {run:<8} {r['release']:>7}  no S2MM grant (no STALLED_LOCK interval holds the release)")
            continue
        cf = {b: ",".join(f"+{t}" for t in r["conflicts"][b]) or "-" for b in (0, 1)}
        chk = "finish ok" if not r.get("check") else "finish MISSING " + "; ".join(r["check"])
        lines.append(f"{c.name:<13} {run:<8} {r['release']:>7}  grant +{r['grant']:<2} "
                     f"fin {','.join(f'{t:+d}' for t in r['s2mm_fin']) or '?':<5} "
                     f"bp {_num(r['bp_fall']) if r['bp_fall'] is not None else '?':<4} "
                     f"starv [{_sl_runs(r['starvation'])}]  B0 [{cf[0]}]  B1 [{cf[1]}]  tail {_num(r['tail']) if r['tail'] is not None else '?'}  {chk}")
    return "\n".join(lines)


def sl_selection(args) -> list[SlCase]:
    return [c for c in sl_plan() if not args.names or c.name in args.names]


def sl_build_case(c: SlCase, out: Path) -> None:
    d = out / c.name
    d.mkdir(parents=True, exist_ok=True)
    lp.assemble(sl_core_asm(c), d / "core.o", "k_core")
    (d / "base.mlir").write_text(sl_design_mlir(c))
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time", *sl_inject_args(c),
           "--trace-config-out", "trace_config.json", "--config-test-name", c.name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    (d / "case.json").write_text(json.dumps({"case": c.name, "head": c.head, "d": c.d, "kind": c.kind,
                                             "k": c.k, "p": c.p, "sl": True}) + "\n")


def cmd_sl_build(args) -> int:
    out = Path(args.out_dir).resolve()
    for c in sl_selection(args):
        sl_build_case(c, out)
        print("built", c.name)
    return 0


def cmd_sl_run(args) -> int:
    out = Path(args.out_dir).resolve()
    for c in sl_selection(args):
        if not (out / c.name / "aie.xclbin").exists():
            continue
        if not lp.run_case(out, c.name, args.tag, args.backend, output=False):
            print(f"{c.name}: run did not complete; stop")
            return 1
    return 0


def cmd_sl_table(args) -> int:
    out = Path(args.out_dir).resolve()
    rows = []
    for c in sl_selection(args):
        for run in args.runs:
            run_dir = out / c.name / run
            if not run_dir.is_dir():
                continue
            if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
                rows.append((c, run, {"release": None}))
                continue
            r = sl_row(load_level_events(run_dir), c.ch)
            r["check"] = sl_missing(r, c)
            rows.append((c, run, r))
    print(format_sl_table(rows))
    return 0


# -- refused-access probe (refuse-*): a lock BD after a BD whose accesses a core camp refused -------------
#
# Does the grant phase of a lock BD depend on what happened to the BD before it? Chained task on MM2S 0:
# pre (lock-free, 4 words) -> BD k (acquire A, 64 words of `src`, release DONE) -> BD k+1 (acquire B, 16 words
# of `tail`, release END). The core releases A, then camps on physical bank 0 with `r` loads (r = 0..3) placed
# at a bank-0 granule of BD k's read (EARLY: the 7th granule, grant + 15; LATE: the last bank-0 granule,
# grant + 47), so r of BD k's accesses are refused (the rotation rule fixes how many cycles the DMA waits;
# `refuse_settle_loads` adds the bank-0 history that puts the camp at the phase where r loads refuse r
# accesses). Then `event #1`, s nops and the release of B, and the core waits for END. The sweep of s moves the
# release of B against the end of BD k, so BD k+1 asks for B at a swept phase; its grant (STALLED_LOCK falling,
# one cycle after LOCK_SEL0_ACQ_GE) minus the release of B is the number the open fidelity row is about.
#
# B is lock 0 so the memory module's lock events (selector 0 at reset) follow it: LOCK_0_REL and
# LOCK_SEL0_ACQ_GE stamp B. The release of A is stamped by the core's INSTR_EVENT_0 (a constant before it).
# Hang safety: the core waits only on the go lock (set by the sequence) and on END, which BD k+1 releases
# after the core released B; the S2MM takes exactly the 84 words the MM2S sends (4 + 64 + 16), lock-free.

REFUSE_LOCK_B, REFUSE_LOCK_A, REFUSE_LOCK_GO, REFUSE_LOCK_DONE, REFUSE_LOCK_END = 0, 1, 2, 3, 4
REFUSE_TAIL_ADDR = 0x4040          # BD k+1's 16 words, logical bank 1 beside `pre`
REFUSE_TAIL_WORDS = 16
REFUSE_RS = (0, 1, 2, 3)           # refused accesses of BD k
REFUSE_CELLS = {"e": 15, "l": 47}  # EARLY / LATE: grant-relative cycle of the camp's first load
REFUSE_CAMP_SLOTS = 3              # bundles of the camp: r loads and nops, so every r has the same length
REFUSE_S_WIDTH = 6                 # consecutive s per (r, position): every residue mod 3 twice
REFUSE_S0 = {"e": 35, "l": 3}      # first s of the sweep per position (calibrated on the emulator)
REFUSE_MAX_S = 80
REFUSE_MEMMOD_EVENTS = ["LOCK_0_REL", "LOCK_SEL0_ACQ_GE", "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_0_FINISHED_BD",
                        "DMA_S2MM_0_FINISHED_BD", "CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1",
                        "DMA_MM2S_0_FINISHED_TASK"]
REFUSE_CORE_EVENTS = ["INSTR_EVENT_0", "INSTR_EVENT_1", "MEMORY_STALL"]
REFUSE_MM2S_BDS = 2                # FINISHED_BD in the trace window: BD k, BD k+1 (the pre BD ends at boot, untraced)


def refuse_refused(r: int, cell: int, m: int) -> int:
    """DMA cycles the rotation rule refuses when r camp loads start at `cell` after the grant with m earlier
    bank-0 loads in the history: contended cycles in which the core, not the DMA, was served."""
    p = rotation_prediction(ArbCase(0, "l", max(r, 1), cell + 1, m), cell) if r else \
        {"contended": [], "core_lost": []}
    return len(p["contended"]) - len(p["core_lost"])


def refuse_settle_loads(cell: int) -> int:
    """The smallest number of bank-0 loads before the release for which camps of r = 1, 2, 3 loads at `cell`
    refuse exactly r DMA accesses under the rotation rule (the bank's served-access count picks who wins)."""
    for m in range(ARB_MAX_M + 1):
        if all(refuse_refused(r, cell, m) == r for r in REFUSE_RS):
            return m
    raise ValueError(f"no history of up to {ARB_MAX_M} loads makes the camps at g+{cell} refuse 1..3")


@dataclass(frozen=True)
class RefuseCase:
    r: int                         # camp loads (accesses of BD k the camp is placed to refuse)
    pos: str                       # "e" early or "l" late
    s: int                         # nops between `event #1` (after the camp) and the release of B

    @property
    def cell(self) -> int:
        return REFUSE_CELLS[self.pos]

    @property
    def name(self) -> str:
        return f"rf{self.pos}{self.r}s{self.s:02d}"


def refuse_validate(c: RefuseCase) -> None:
    if c.r not in REFUSE_RS:
        raise ValueError(f"r must be one of {REFUSE_RS}")
    if c.pos not in REFUSE_CELLS:
        raise ValueError(f"pos must be one of {tuple(REFUSE_CELLS)}")
    if not 0 <= c.s <= REFUSE_MAX_S:
        raise ValueError(f"s must be 0..{REFUSE_MAX_S}")


def refuse_plan() -> list[RefuseCase]:
    """The grid: r x position x six consecutive s from the position's first s (48 cases)."""
    return [RefuseCase(r, pos, REFUSE_S0[pos] + i) for pos in REFUSE_CELLS for r in REFUSE_RS
            for i in range(REFUSE_S_WIDTH)]


def refuse_core_asm(c: RefuseCase) -> str:
    """go, settle, the history loads, `event #0`, release A, k nops, the camp (r bank-0 loads then nops to
    REFUSE_CAMP_SLOTS bundles), `event #1`, s nops, release B, wait END. r2 = +1 releases, r3 = -1 acquires,
    p0 = `ldbuf`. k = cell + 1 puts the first load `cell` cycles after a grant of release + 2."""
    refuse_validate(c)
    n = lp.nops
    history = [f"  lda r25, [p0, #{ARB_OFFSETS[0][0]}]"] * refuse_settle_loads(c.cell)
    camp = [f"  lda {ARB_LOAD_REGS[i]}, [p0, #{ARB_OFFSETS[0][i]}]" for i in range(c.r)]
    body = ["  mov r2, #1", "  mov r3, #-1", *n(2),
            f"  acq #{OWN + REFUSE_LOCK_GO}, r3", *n(4),
            *lp.delay_loop(FETCH_SETTLE_ITERS, ".Lsettle"), *history, *n(8),
            "  event #0", f"  rel #{OWN + REFUSE_LOCK_A}, r2", *n(c.cell + 1),
            *camp, *n(REFUSE_CAMP_SLOTS - c.r), "  event #1", *n(c.s),
            f"  rel #{OWN + REFUSE_LOCK_B}, r2", *n(8),
            f"  acq #{OWN + REFUSE_LOCK_END}, r3", *n(8),
            *lp.delay_loop(FETCH_TAIL_ITERS, ".Ltail")]
    return lp.frame(body)


def refuse_design_mlir() -> str:
    """Case-independent design (see the section comment)."""
    w, p, t = FETCH_WORDS, FETCH_PRE_WORDS, REFUSE_TAIL_WORDS
    dw = p + w + t
    ramp = ", ".join(str(0x0A000000 | j) for j in range(w))
    ld = ", ".join(str(0x0B000000 | j) for j in range(16))
    pre_init = ", ".join(str(0x0C000000 | j) for j in range(p))
    tail_init = ", ".join(str(0x0E000000 | j) for j in range(t))
    pad = "\n".join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32"
                    for _ in range(FETCH_PAD_WRITES))
    return f"""module {{
  aie.device(npu1_1col) {{
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %src = aie.buffer(%t02) {{address = {FETCH_SRC_ADDR} : i32, sym_name = "src", initial_value = dense<[{ramp}]> : tensor<{w}xi32>}} : memref<{w}xi32>
    %ldbuf = aie.buffer(%t02) {{address = {FETCH_LD_ADDR} : i32, sym_name = "ldbuf", initial_value = dense<[{ld}]> : tensor<16xi32>}} : memref<16xi32>
    %pre = aie.buffer(%t02) {{address = {FETCH_PRE_ADDR} : i32, sym_name = "pre", initial_value = dense<[{pre_init}]> : tensor<{p}xi32>}} : memref<{p}xi32>
    %tail = aie.buffer(%t02) {{address = {REFUSE_TAIL_ADDR} : i32, sym_name = "tail", initial_value = dense<[{tail_init}]> : tensor<{t}xi32>}} : memref<{t}xi32>
    %dst = aie.buffer(%t02) {{address = {FETCH_DST_ADDR} : i32, sym_name = "dst"}} : memref<{dw}xi32>
    %scratch = aie.buffer(%t02) {{address = {FETCH_SCRATCH_ADDR} : i32, sym_name = "scratch"}} : memref<4xi32>
    %lB = aie.lock(%t02, {REFUSE_LOCK_B}) {{init = 0 : i32, sym_name = "lB"}}
    %lA = aie.lock(%t02, {REFUSE_LOCK_A}) {{init = 0 : i32, sym_name = "lA"}}
    %lGo = aie.lock(%t02, {REFUSE_LOCK_GO}) {{init = 0 : i32, sym_name = "lGo"}}
    %lDone = aie.lock(%t02, {REFUSE_LOCK_DONE}) {{init = 0 : i32, sym_name = "lDone"}}
    %lEnd = aie.lock(%t02, {REFUSE_LOCK_END}) {{init = 0 : i32, sym_name = "lEnd"}}
    aie.flow(%t02, DMA : 0, %t02, DMA : 0)
    func.func private @k_core(memref<16xi32>) attributes {{link_with = "core.o"}}
    aie.core(%t02) {{
      func.call @k_core(%ldbuf) : (memref<16xi32>) -> ()
      aie.end
    }}
    %mem = aie.mem(%t02) {{
      %s0 = aie.dma_start(MM2S, 0, ^pre, ^recv_start)
    ^pre:
      aie.dma_bd(%pre : memref<{p}xi32>)
      aie.next_bd ^send
    ^send:
      %ca = arith.constant 1 : i32
      aie.use_lock(%lA, AcquireGreaterEqual, %ca)
      aie.dma_bd(%src : memref<{w}xi32>)
      %cd = arith.constant 1 : i32
      aie.use_lock(%lDone, Release, %cd)
      aie.next_bd ^second
    ^second:
      %cb = arith.constant 1 : i32
      aie.use_lock(%lB, AcquireGreaterEqual, %cb)
      aie.dma_bd(%tail : memref<{t}xi32>)
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
      aiex.set_lock(%lGo, %c_one)
{pad}
    }}
  }}
}}
"""


def refuse_inject_args() -> list[str]:
    return ["--memmod-sweep-events", ",".join(REFUSE_MEMMOD_EVENTS),
            "--core-grounding", FETCH_CORE_GROUNDING,
            "--core-sweep-events", ",".join(REFUSE_CORE_EVENTS)]


def refuse_row(events) -> dict:
    """One run of tile (0,2). Memory-module stamps: g_k = the cycle after the first MM2S STALLED_LOCK run
    (BD k's wait on A) falls; R_B = the LOCK_0_REL (B's release); G_B = the cycle after the STALLED_LOCK run
    holding R_B falls, minus R_B (None when B was free when BD k+1 asked). `fins` / `s2mm_fins` / `tasks` are
    the FINISHED_BD / FINISHED_TASK stamps, `conflicts` the bank-0 and bank-1 conflict cycles, all relative to
    g_k; `refused` = bank-0 conflict cycles minus the core's MEMORY_STALL cycles (core clock, in the camp)."""
    mem, core = {}, {}
    for e in events:
        if e.row == FETCH_ROW:
            (mem if e.pkt_type == PT_MEM else core).setdefault(e.name, []).append(e.ts)
    runs = _runs(mem.get("DMA_MM2S_0_STALLED_LOCK", []))
    rels = sorted(set(mem.get("LOCK_0_REL", [])))
    if not runs or not rels:
        return {"g_k": None}
    g = runs[0][1] + 1
    rb = rels[0]

    def rel(name, src=mem):
        return [t - g for t in sorted(set(src.get(name, [])))]
    waiting = [run for run in runs[1:] if run[0] <= rb <= run[1] + 1]
    ev0 = sorted(core.get("INSTR_EVENT_0", []))
    stalls = [t for t in sorted(set(core.get("MEMORY_STALL", []))) if ev0 and t >= ev0[0]]
    conflicts = sorted((t - g, b[-1]) for b in ("CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1")
                       for t in set(mem.get(b, [])))
    fins = rel("DMA_MM2S_0_FINISHED_BD")
    b0 = [t for t, b in conflicts if b == "0"]
    return {"g_k": g, "r_b": rb - g, "grant_b": (waiting[0][1] + 1 - rb) if waiting else None,
            "acq_b": [t - rb for t in sorted(set(mem.get("LOCK_SEL0_ACQ_GE", [])))],
            "fins": fins, "s2mm_fins": rel("DMA_S2MM_0_FINISHED_BD"), "tasks": rel("DMA_MM2S_0_FINISHED_TASK"),
            "conflicts": conflicts, "memory_stalls": len(stalls),
            "refused": len(b0) - len(stalls), "present": sorted(mem), "core_present": sorted(core)}


def refuse_missing(r: dict, c: RefuseCase) -> list[str]:
    """What the design needs from a run and did not get: every event of the memory-module list, the core
    markers, and every DMA task finished (BD k and BD k+1 and the MM2S task, the S2MM's one BD). A bank
    conflict is needed only with a camp (r > 0), and only on bank 0: the camp never touches bank 1."""
    if r.get("g_k") is None:
        return ["no STALLED_LOCK run or no LOCK_0_REL"]
    optional = {"CONFLICT_DM_BANK_1"} | ({"CONFLICT_DM_BANK_0"} if c.r == 0 else set())
    out = [e for e in REFUSE_MEMMOD_EVENTS if e not in r["present"] and e not in optional]
    out += [e for e in ("INSTR_EVENT_0", "INSTR_EVENT_1") if e not in r["core_present"]]
    if len(r["fins"]) != REFUSE_MM2S_BDS:
        out.append(f"MM2S FINISHED_BD x{len(r['fins'])} (want {REFUSE_MM2S_BDS})")
    if len(r["s2mm_fins"]) != 1:
        out.append(f"S2MM FINISHED_BD x{len(r['s2mm_fins'])} (want 1)")
    if len(r["tasks"]) != 1:
        out.append(f"MM2S FINISHED_TASK x{len(r['tasks'])} (want 1)")
    return out


def format_refuse_table(rows) -> str:
    """rows: (RefuseCase, run, refuse_row dict). Cycles after g_k (BD k's grant) except grant_b (after R_B)."""
    lines = ["case      run      g_k(abs)  F_k  R_B-F_k  grant_b  refused  mstall  conflicts(b0 cycles after g_k)  "
             "finish check"]
    for c, run, r in sorted(rows, key=lambda x: (x[0].name, x[1])):
        miss = refuse_missing(r, c)
        if r.get("g_k") is None:
            lines.append(f"{c.name:<9} {run:<8} {'; '.join(miss)}")
            continue
        fk = r["fins"][0] if r["fins"] else None
        conf = ",".join(str(t) for t, b in r["conflicts"] if b == "0") or "-"
        lines.append(f"{c.name:<9} {run:<8} {r['g_k']:>8} {_num(fk, 'd'):>4} "
                     f"{_num(None if fk is None else r['r_b'] - fk):>7}  {_num(r['grant_b']):>7}  "
                     f"{r['refused']:>7}  {r['memory_stalls']:>6}  {conf:<30}  " + ("ok" if not miss else "MISSING " + "; ".join(miss)))
    return "\n".join(lines)


def refuse_selection(args) -> list[RefuseCase]:
    return [c for c in refuse_plan() if not args.names or c.name in args.names]


def refuse_build_case(c: RefuseCase, out: Path) -> None:
    d = out / c.name
    d.mkdir(parents=True, exist_ok=True)
    lp.assemble(refuse_core_asm(c), d / "core.o", "k_core")
    (d / "base.mlir").write_text(refuse_design_mlir())
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time", *refuse_inject_args(),
           "--trace-config-out", "trace_config.json", "--config-test-name", c.name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    (d / "case.json").write_text(json.dumps({"case": c.name, "r": c.r, "pos": c.pos, "s": c.s,
                                             "refuse": True}) + "\n")


def cmd_refuse_build(args) -> int:
    out = Path(args.out_dir).resolve()
    for c in refuse_selection(args):
        refuse_build_case(c, out)
        print("built", c.name)
    return 0


def cmd_refuse_run(args) -> int:
    out = Path(args.out_dir).resolve()
    for c in refuse_selection(args):
        if not (out / c.name / "aie.xclbin").exists():
            continue
        if not lp.run_case(out, c.name, args.tag, args.backend, output=False):
            print(f"{c.name}: run did not complete; stop")
            return 1
    return 0


def cmd_refuse_table(args) -> int:
    out = Path(args.out_dir).resolve()
    rows = []
    for c in refuse_selection(args):
        for run in args.runs:
            run_dir = out / c.name / run
            if not run_dir.is_dir():
                continue
            if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
                rows.append((c, run, {"g_k": None}))
                continue
            rows.append((c, run, refuse_row(load_level_events(run_dir))))
    print(format_refuse_table(rows))
    return 0


def build_parser() -> argparse.ArgumentParser:
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
    fb.add_argument("--memmod-event-swap", type=parse_event_swap, default=None, metavar="OLD=NEW",
                    help=MEMMOD_SWAP_HELP)
    fb.add_argument("--memmod-event-add", default=None, metavar="NAME", help=MEMMOD_ADD_HELP)
    fb.add_argument("--lock-event-select", type=parse_lock_event_select, action="append", default=None,
                    metavar="N=LOCK_ID", help=LOCK_SELECT_HELP)
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
        sp.add_argument("--start-task", action="store_true",
                        help="DMA-side stamp cases (lfsNN first-BD, lfqNN with --chained): DMA_MM2S_0_START_TASK joins "
                             "the memory-module events and --pre-nops nops precede the release; no core loads")
        sp.add_argument("--pre-nops", nargs="*", type=int, default=[],
                        help="--start-task: nops before `event #0` and the release (default 0..5)")
        sp.add_argument("--sync", action="store_true",
                        help="the lock-synchronised variant (cases lsDD, lsfree, lsaDDkKK/lsbDDkKK): --ds are core delays, "
                             "--banks none|0|1 (default none), --ks the load offsets of the bank cases")
        sp.add_argument("--ds", nargs="*", type=int, default=[], help="--sync: core delays d (default 0..24, 26..40 by 2)")
        sp.add_argument("--free", action="store_true", help="--sync: also the lock-free-on-request case (lsfree)")
        sp.add_argument("--pre2", type=int, default=0, help="--sync: words of a second lock-free pre BD (cases ltDD...)")
        sp.add_argument("--free-only", action="store_true", help="--sync: only lsfree")
    add_chain_parsers(sub)
    ab = sub.add_parser("arb-build", help="bank arbitration probe: build the plan's cases (or --names)")
    ab.add_argument("out_dir")
    ar = sub.add_parser("arb-run", help="bank arbitration probe: run the built cases to TAG.BACKEND/")
    ar.add_argument("out_dir")
    ar.add_argument("tag")
    ar.add_argument("--backend", choices=("hw", "emu"), required=True)
    at = sub.add_parser("arb-table", help="bank arbitration probe: measurement next to the rotation rule")
    at.add_argument("out_dir")
    at.add_argument("runs", nargs="+")
    rb = sub.add_parser("ring-build", help="ring-order probe: build the plan's cases (or --names)")
    rb.add_argument("out_dir")
    rr = sub.add_parser("ring-run", help="ring-order probe: run the built cases to TAG.BACKEND/")
    rr.add_argument("out_dir")
    rr.add_argument("tag")
    rr.add_argument("--backend", choices=("hw", "emu"), required=True)
    rt = sub.add_parser("ring-table", help="ring-order probe: per-run camp table with count residues")
    rt.add_argument("out_dir")
    rt.add_argument("runs", nargs="+")
    xb = sub.add_parser("refuse-build", help="refused-access probe: build the plan's cases (or --names)")
    xb.add_argument("out_dir")
    xr = sub.add_parser("refuse-run", help="refused-access probe: run the built cases to TAG.BACKEND/")
    xr.add_argument("out_dir")
    xr.add_argument("tag")
    xr.add_argument("--backend", choices=("hw", "emu"), required=True)
    xt = sub.add_parser("refuse-table", help="refused-access probe: per-run grant offsets and the finish check")
    xt.add_argument("out_dir")
    xt.add_argument("runs", nargs="+")
    sb = sub.add_parser("sl-build", help="lock-held S2MM probe: build the plan's cases (or --names)")
    sb.add_argument("out_dir")
    sr = sub.add_parser("sl-run", help="lock-held S2MM probe: run the built cases to TAG.BACKEND/")
    sr.add_argument("out_dir")
    sr.add_argument("tag")
    sr.add_argument("--backend", choices=("hw", "emu"), required=True)
    st = sub.add_parser("sl-table", help="lock-held S2MM probe: release -> grant, finish, back-pressure, conflicts")
    st.add_argument("out_dir")
    st.add_argument("runs", nargs="+")
    for sp in (ab, ar, at, rb, rr, rt, xb, xr, xt, sb, sr, st):
        sp.add_argument("--names", nargs="*", default=[], help="only these case names")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.cmd in CHAIN_COMMANDS:
        return CHAIN_COMMANDS[args.cmd](args)
    if args.cmd.startswith("sl-"):
        return {"sl-build": cmd_sl_build, "sl-run": cmd_sl_run, "sl-table": cmd_sl_table}[args.cmd](args)
    if args.cmd.startswith("refuse-"):
        return {"refuse-build": cmd_refuse_build, "refuse-run": cmd_refuse_run,
                "refuse-table": cmd_refuse_table}[args.cmd](args)
    if args.cmd.startswith("ring-"):
        return {"ring-build": cmd_ring_build, "ring-run": cmd_ring_run, "ring-table": cmd_ring_table}[args.cmd](args)
    if args.cmd.startswith("arb-"):
        return {"arb-build": cmd_arb_build, "arb-run": cmd_arb_run, "arb-table": cmd_arb_table}[args.cmd](args)
    if args.cmd.startswith("fetch-"):
        if getattr(args, "lock_event_select", None) and not args.sync:
            ap.error("--lock-event-select needs --sync")
        if args.start_task and args.sync:
            ap.error("--start-task and --sync are separate variants")
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
