#!/usr/bin/env python3
"""Packet-switch arbiter hand-over: idle cycles between two packets that share a master port.

Question. When two packet-switched sources share one stream-switch master port, how many
idle cycles separate the last word (TLAST) of packet A from the first word of packet B, as a
function of when B became ready relative to A's TLAST?

Design (source tile (0,2), destination tile (0,3); only the trace goes to the shim):
  ab    MM2S 0 sends A (16 words, packet id 1) and MM2S 1 sends B (4 words, packet id 2), both
        `aie.packet_flow`s to S2MM 0 of the tile above, (0,3). The two flows enter the source
        tile's switch on different slave ports (DMA 0, DMA 1) and leave on one master port
        (North 5 as the router numbers it; `build` checks), so they share one arbiter. (The destination cannot be the source tile
        itself: the router connects a DMA slave only to the DMA master of the same channel,
        so B would be routed out through the shim and back.) The destination drops the packet
        header, S2MM 0 stores the 20 data words. Tile (0,3) runs an empty core.
  same  A and B are two chained BDs of MM2S 0 with the same packet id (the same-stream gap).
  solo  B alone (control): gives the lock-grant-to-first-word latency of a BD.
  far   A, then B released about 1000 cycles later (control: B long after A).
  The core releases A's lock, then k nops, then B's lock, so B becomes ready a known number
  of cycles after A. k is swept one cycle per case; `table` converts k to "B ready relative
  to A's TLAST" from the trace itself, so the sweep only has to cover the window.

Measured, in the memory-module timers: S2MM 0 STREAM_STARVATION of the destination tile (the
words reaching it), MM2S 0/1 STREAM_BACKPRESSURE, FINISHED_BD and STALLED_LOCK of the source
tile. The two tiles' timers are two clocks: the gap is read from the destination clock alone,
and the B-ready offset uses the solo control's grant-to-first-word latency, which absorbs
any constant skew between them. The
STALLED_LOCK interval of B's channel ends when B's lock is granted: that is when B becomes
ready. Core module, its own timer: PORT_RUNNING_0 / PORT_TLAST_0 / PORT_IDLE_0 on the shared
master port (stream-switch event port 0 = North 5 master), PORT_RUNNING_1 / _2 on the two
source slave ports, and INSTR_EVENT_0 just before A's release.

Why the trace does not disturb the measurement. The trace units send their packets out of
the source tile on South masters towards the shim; they never use master North 5, so they do
not share the measured arbiter and port. `trace_shares_master` checks the lowered MLIR for
this (master and arbiter), and the table shows a violation as: the solo or far control rows (one packet only)
having an internal starved interval inside the 4-word packet B, or backpressure on a source
when only one source runs.

Why it always completes. The cores start when the array is configured, before the trace is armed,
so the core first waits on a go lock that the runtime sequence sets (a command-processor
write) after the trace is armed. Nothing in the design waits on a host buffer, host DMA or
host signal. The runtime sequence holds only plain writes and a lock set: it waits
on nothing and ends. Every lock a DMA or the core waits on is released by something that
always runs: the go lock by the sequence, A's and B's locks by the core (which runs after the
go lock), and the S2MM has no lock. A BD needs a release field, so each source BD releases a
fifth lock that nothing waits on. Every S2MM that receives has room for all words sent: its
BD length equals the words of all packets sent to it (20, or 4 for solo), and nothing in the
design waits for that BD to finish.

    python3 tools/gen_packet_gap_probe.py build OUT_DIR [--cases ab07 same10 solo far]
    python3 tools/gen_packet_gap_probe.py run OUT_DIR TAG --backend hw|emu [--cases ...]
    python3 tools/gen_packet_gap_probe.py table OUT_DIR RUN [RUN ...] [--cases ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_dma_lock_probe as dl                                     # noqa: E402
import gen_lock_pulse_probe as lp                                   # noqa: E402

LOCK_A, LOCK_B, LOCK_GO, LOCK_DONE = 0, 1, 2, 3
OWN = dl.OWN
WORDS_A, WORDS_B = 16, 4
PKT_A, PKT_B = 1, 2
AB_KS = tuple(range(4, 30))
SAME_KS = tuple(range(4, 30, 2))
SETTLE_ITERS = 40
FAR_ITERS = 100                    # about 10 cycles each: B is released ~1000 cycles after A
TAIL_ITERS = 100                   # keeps the core running so PERF_CNT_2 fills its last packet
PAD_WRITES = dl.FETCH_PAD_WRITES
ADDR_A, ADDR_B, ADDR_DST, ADDR_SCRATCH = 0x0400, 0x4400, 0x8000, 0xC000
MEMMOD_EVENTS = ["DMA_S2MM_0_STREAM_STARVATION", "DMA_MM2S_0_STREAM_BACKPRESSURE",
                 "DMA_MM2S_1_STREAM_BACKPRESSURE", "DMA_MM2S_0_FINISHED_BD", "DMA_MM2S_1_FINISHED_BD",
                 "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_1_STALLED_LOCK", "DMA_S2MM_0_FINISHED_BD"]
CORE_EVENTS = ["PORT_RUNNING_0", "PORT_TLAST_0", "PORT_IDLE_0", "PORT_RUNNING_1", "PORT_RUNNING_2"]
CORE_GROUNDING = "PERF_CNT_2,INSTR_EVENT_0"
PORT_SLOTS = "0=North:5:master,1=DMA:0:slave,2=DMA:1:slave"
HEADER_WORDS = 1                   # B's header crosses the shared port but is dropped at the destination
ROW, DST_ROW, PT_CORE, PT_MEM = 2, 3, 0, 1
MEASURED_MASTER = ("North", 5)   # what the router picks for the shared master; build verifies


@dataclass(frozen=True)
class Case:
    mode: str                      # ab | same | solo | far
    k: int | None = None           # nops between A's release and B's release (ab, same)

    @property
    def name(self) -> str:
        return self.mode if self.k is None else f"{self.mode}{self.k:02d}"

    @property
    def dst_words(self) -> int:
        return WORDS_B if self.mode == "solo" else WORDS_A + WORDS_B

    @property
    def b_channel(self) -> int:
        """MM2S channel that carries B."""
        return 0 if self.mode == "same" else 1


def default_cases() -> list[Case]:
    return ([Case("ab", k) for k in AB_KS] + [Case("same", k) for k in SAME_KS]
            + [Case("solo"), Case("far", None)])


def case_from_name(name: str) -> Case:
    m = re.fullmatch(r"(ab|same)(\d+)", name)
    if m:
        return Case(m.group(1), int(m.group(2)))
    if name in ("solo", "far"):
        return Case(name)
    raise ValueError(f"unknown case {name!r}")


def core_asm(case: Case) -> str:
    """go, settle, `event #0`, release A, then B after k nops (ab, same) or a long delay (far)."""
    n = lp.nops
    body = ["  mov r2, #1", "  mov r3, #-1", *n(2),
            f"  acq #{OWN + LOCK_GO}, r3", *n(4),
            *lp.delay_loop(SETTLE_ITERS, ".Lsettle"), *n(8), "  event #0"]
    if case.mode == "solo":
        body += [f"  rel #{OWN + LOCK_B}, r2"]
    elif case.mode == "far":
        body += [f"  rel #{OWN + LOCK_A}, r2", *n(2), *lp.delay_loop(FAR_ITERS, ".Lfar"), *n(2),
                 f"  rel #{OWN + LOCK_B}, r2"]
    else:
        body += [f"  rel #{OWN + LOCK_A}, r2", *n(case.k), f"  rel #{OWN + LOCK_B}, r2"]
    body += [*n(8), *lp.delay_loop(TAIL_ITERS, ".Ltail")]
    return lp.frame(body)


def _ramp(base: int, words: int) -> str:
    return ", ".join(str(base | j) for j in range(words))


def design_mlir(case: Case) -> str:
    """The case's design; the core program carries k (the design differs per mode only)."""
    wa, wb, wd = WORDS_A, WORDS_B, case.dst_words

    def bd(lock: str, buf: str, words: int, pkt: int, nxt: str, label: str) -> str:
        return f"""    ^{label}:
      %c_{label} = arith.constant 1 : i32
      aie.use_lock({lock}, AcquireGreaterEqual, %c_{label})
      aie.dma_bd({buf} : memref<{words}xi32>) {{packet = #aie.packet_info<pkt_type = 0, pkt_id = {pkt}>}}
      %d_{label} = arith.constant 1 : i32
      aie.use_lock(%lDone, Release, %d_{label})
      aie.next_bd {nxt}
"""
    flows = {"ab": f"""    aie.packet_flow({PKT_A}) {{
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    }}
    aie.packet_flow({PKT_B}) {{
      aie.packet_source<%t02, DMA : 1>
      aie.packet_dest<%t03, DMA : 0>
    }}
""", "same": f"""    aie.packet_flow({PKT_A}) {{
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    }}
""", "far": None, "solo": f"""    aie.packet_flow({PKT_B}) {{
      aie.packet_source<%t02, DMA : 1>
      aie.packet_dest<%t03, DMA : 0>
    }}
"""}
    flows["far"] = flows["ab"]
    mm2s = {"ab": f"""      %s0 = aie.dma_start(MM2S, 0, ^a, ^ch1)
{bd("%lA", "%bufA", wa, PKT_A, "^end", "a")}    ^ch1:
      %s1 = aie.dma_start(MM2S, 1, ^b, ^end)
{bd("%lB", "%bufB", wb, PKT_B, "^end", "b")}""",
            "same": f"""      %s0 = aie.dma_start(MM2S, 0, ^a, ^end)
{bd("%lA", "%bufA", wa, PKT_A, "^b", "a")}{bd("%lB", "%bufB", wb, PKT_A, "^end", "b")}""",
            "solo": f"""      %s1 = aie.dma_start(MM2S, 1, ^b, ^end)
{bd("%lB", "%bufB", wb, PKT_B, "^end", "b")}"""}
    mm2s["far"] = mm2s["ab"]
    return f"""module {{
  aie.device(npu1_1col) {{
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %bufA = aie.buffer(%t02) {{address = {ADDR_A} : i32, sym_name = "bufA", initial_value = dense<[{_ramp(0x0A000000, wa)}]> : tensor<{wa}xi32>}} : memref<{wa}xi32>
    %bufB = aie.buffer(%t02) {{address = {ADDR_B} : i32, sym_name = "bufB", initial_value = dense<[{_ramp(0x0B000000, wb)}]> : tensor<{wb}xi32>}} : memref<{wb}xi32>
    %dst = aie.buffer(%t03) {{address = {ADDR_DST} : i32, sym_name = "dst"}} : memref<{wd}xi32>
    %scratch = aie.buffer(%t02) {{address = {ADDR_SCRATCH} : i32, sym_name = "scratch"}} : memref<4xi32>
    %lA = aie.lock(%t02, {LOCK_A}) {{init = 0 : i32, sym_name = "lA"}}
    %lB = aie.lock(%t02, {LOCK_B}) {{init = 0 : i32, sym_name = "lB"}}
    %lGo = aie.lock(%t02, {LOCK_GO}) {{init = 0 : i32, sym_name = "lGo"}}
    %lDone = aie.lock(%t02, {LOCK_DONE}) {{init = 0 : i32, sym_name = "lDone"}}
{flows[case.mode]}    func.func private @k_core() attributes {{link_with = "core.o"}}
    aie.core(%t02) {{
      func.call @k_core() : () -> ()
      aie.end
    }}
    aie.core(%t03) {{
      aie.end
    }}
    %mem = aie.mem(%t02) {{
{mm2s[case.mode]}    ^end:
      aie.end
    }}
    %mem3 = aie.mem(%t03) {{
      %s2 = aie.dma_start(S2MM, 0, ^recv, ^end)
    ^recv:
      aie.dma_bd(%dst : memref<{wd}xi32>)
      aie.next_bd ^end
    ^end:
      aie.end
    }}
    aie.runtime_sequence(%dummy: memref<4xi32>) {{
      %c_zero = arith.constant 0 : i32
      %c_one = arith.constant 1 : i32
      aiex.set_lock(%lGo, %c_one)
{chr(10).join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32" for _ in range(PAD_WRITES))}
    }}
  }}
}}
"""


def inject_args() -> list[str]:
    return ["--memmod-sweep-events", ",".join(MEMMOD_EVENTS), "--core-grounding", CORE_GROUNDING,
            "--core-sweep-events", ",".join(CORE_EVENTS), "--port-slots", PORT_SLOTS]


_AMSEL = re.compile(r"(%\w+)\s*=\s*aie\.amsel<(\d+)>")
_MASTERSET = re.compile(r"aie\.masterset\((\w+)\s*:\s*(\d+)((?:\s*,\s*%\w+)+)\)")
_RULES = re.compile(r"aie\.packet_rules\((\w+)\s*:\s*(\d+)\)\s*\{(.*?)\}", re.S)
_RULE = re.compile(r"aie\.rule\(\s*\d+\s*,\s*\d+\s*,\s*(%\w+)\)")


def _switchbox(mlir: str, tile: tuple[int, int]):
    m = re.search(rf"aie\.switchbox\(%tile_{tile[0]}_{tile[1]}\)\s*\{{(.*?)\n    \}}", mlir, re.S)
    if not m:
        raise ValueError(f"no switchbox for tile {tile}")
    box = m.group(1)
    arb = {name: int(a) for name, a in _AMSEL.findall(box)}
    feeds: dict[str, set] = {}
    for bundle, ch, names in _MASTERSET.findall(box):
        for name in re.findall(r"%\w+", names):
            feeds.setdefault(name, set()).add((bundle, int(ch)))
    trace, data = set(), set()
    for bundle, ch, body in _RULES.findall(box):
        (data if bundle == "DMA" else trace).update(_RULE.findall(body))
    return arb, feeds, trace, data


def data_masters(mlir: str, tile: tuple[int, int] = (0, 2)) -> set:
    """Master ports the packet rules of the tile's DMA slaves lead to."""
    _, feeds, _, data = _switchbox(mlir, tile)
    return {m for name in data for m in feeds.get(name, ())}


def trace_shares_master(mlir: str, tile: tuple[int, int] = (0, 2)) -> list[str]:
    """What the lowered MLIR's switchbox of `tile` shares between the trace and the data flows.
    A trace packet rule (slave Trace, or a trace of the tile above passing through) must select neither an arbiter that feeds a master of the
    data flows nor the arbiter number of the data flows' rules. Empty means no sharing."""
    arb, feeds, trace, data = _switchbox(mlir, tile)
    masters = data_masters(mlir, tile)
    bad = []
    for name in sorted(trace):
        if feeds.get(name, set()) & masters:
            bad.append(f"trace rule uses {name}, which feeds {sorted(feeds[name] & masters)}")
        elif arb.get(name) in {arb[d] for d in data if d in arb}:
            bad.append(f"trace rule uses arbiter {arb.get(name)} of the data flows")
    return bad


# -- the table ------------------------------------------------------------------------

def _stamps(events, pkt: int, name: str, row: int = ROW) -> list[int]:
    return sorted({e.ts for e in events if e.row == row and e.pkt_type == pkt and e.name == name})


def gap_row(events, case: Case) -> dict:
    """One run as numbers (memory-module timer unless noted).

    first_word  cycle the destination S2MM 0 first leaves starvation (every number is relative;
                the source tile's cycles are on another clock, see the module doc).
    internal    starved intervals after the first word: (start offset, length) each.
    raw_gap     total internal starved cycles at the destination (one boundary only).
    gap         raw_gap minus the one cycle B's header word occupies: the idle cycles A's last
                word to B's header at the shared port (the header is dropped before the S2MM).
    a_last      the last word of A: just before the first internal starved interval, else
                first_word + 15 when A and B are back to back.
    b_grant     end of the last STALLED_LOCK interval of B's channel (B's lock granted).
    port_gap    core timer: idle cycles between the first two PORT_RUNNING_0 runs (3+ cycles each) of
                the shared master port: the same gap read at the port itself, to check `gap`."""
    starv = dl._runs(_stamps(events, PT_MEM, "DMA_S2MM_0_STREAM_STARVATION", DST_ROW))
    if len(starv) == 0:
        return {"first_word": None}
    first = starv[0][1] + 1
    internal = [(a - first, b - a + 1) for a, b in starv[1:]]
    raw_gap = sum(length for _, length in internal)
    gap = max(raw_gap - HEADER_WORDS, 0)
    a_last = first + (internal[0][0] - 1 if internal else WORDS_A - 1)
    ch = case.b_channel
    locks = dl._runs(_stamps(events, PT_MEM, f"DMA_MM2S_{ch}_STALLED_LOCK"))
    b_grant = locks[-1][1] + 1 if locks else None
    bp = {c: sum(b - a + 1 for a, b in dl._runs(_stamps(events, PT_MEM, f"DMA_MM2S_{c}_STREAM_BACKPRESSURE")))
          for c in (0, 1)}
    port_runs = [r for r in dl._runs(_stamps(events, PT_CORE, "PORT_RUNNING_0")) if r[1] - r[0] + 1 >= 3]
    port_gap = port_runs[1][0] - port_runs[0][1] - 1 if len(port_runs) > 1 else (0 if port_runs else None)
    return {"first_word": first, "internal": internal, "gap": gap, "raw_gap": raw_gap, "a_last": a_last - first,
            "b_grant": None if b_grant is None else b_grant - first, "backpressure": bp,
            "port_gap": port_gap,
            "finished": {c: [t - first for t in _stamps(events, PT_MEM, f"DMA_MM2S_{c}_FINISHED_BD")]
                         for c in (0, 1)}}


def solo_latency(row: dict) -> int | None:
    """Cycles from B's lock grant to B's first word at the destination (the solo control)."""
    if row.get("first_word") is None or row.get("b_grant") is None:
        return None
    return -row["b_grant"]


def b_ready(row: dict, latency: int | None) -> int | None:
    """B's first word could have reached the destination at this cycle relative to A's last
    word: its grant plus the grant-to-first-word latency of an uncontended BD."""
    if row.get("b_grant") is None or latency is None:
        return None
    return row["b_grant"] + latency - row["a_last"]


def format_table(rows, latency: int | None) -> str:
    """rows: (case, run name, gap_row dict)."""
    head = (f"{'case':<8} {'run':<8} {'B_ready':>7} {'gap':>4} {'port_gap':>8}  details  "
            f"(B_ready: B's first word could arrive this many cycles after A's last word, from "
            f"B's lock grant + the solo latency {latency}; gap: idle cycles between them at the "
            f"destination, less B's header word; all relative to the first word)")
    lines = [head]
    for case, run, r in rows:
        if r.get("first_word") is None:
            lines.append(f"{case.name:<8} {run:<8} no S2MM starvation event")
            continue
        ready = b_ready(r, latency)
        rd = "?" if ready is None else f"{ready:+d}"
        pg = "?" if r["port_gap"] is None else str(r["port_gap"])
        lines.append(f"{case.name:<8} {run:<8} {rd:>7} {r['gap']:>4} {pg:>8}  grantB {r['b_grant']}  "
                     f"a_last +{r['a_last']}  internal {r['internal']}  backpressure {r['backpressure']}  "
                     f"fin0 {r['finished'][0]} fin1 {r['finished'][1]}")
    return "\n".join(lines)


# -- build / run / table CLI -------------------------------------------------------------

def build_case(case: Case, out: Path) -> None:
    d = out / case.name
    d.mkdir(parents=True, exist_ok=True)
    lp.assemble(core_asm(case), d / "core.o", "k_core")
    (d / "base.mlir").write_text(design_mlir(case))
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time", *inject_args(),
           "--trace-config-out", "trace_config.json", "--config-test-name", case.name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    lowered = (d / "input_with_addresses.mlir").read_text()
    if case.mode != "solo" and data_masters(lowered) != {MEASURED_MASTER}:
        raise SystemExit(f"{case.name}: data flows use {data_masters(lowered)}, not {MEASURED_MASTER}")
    shared = trace_shares_master(lowered)
    if shared:
        raise SystemExit(f"{case.name}: trace shares the measured master port: {shared}")
    (d / "case.json").write_text(json.dumps({"case": case.name, "mode": case.mode, "k": case.k}) + "\n")


def select(out: Path, names) -> list[Case]:
    cases = [case_from_name(n) for n in names] if names else default_cases()
    return [c for c in cases if (out / c.name).is_dir()]


def cmd_build(args) -> int:
    out = Path(args.out_dir).resolve()
    for case in ([case_from_name(n) for n in args.cases] if args.cases else default_cases()):
        build_case(case, out)
        print("built", case.name, flush=True)
    return 0


def cmd_run(args) -> int:
    """Stops at the first run that does not complete; never retries."""
    out = Path(args.out_dir).resolve()
    for case in select(out, args.cases):
        if not lp.run_case(out, case.name, args.tag, args.backend, output=False):
            print(f"{case.name}: run did not complete; stop")
            return 1
    return 0


def cmd_table(args) -> int:
    out = Path(args.out_dir).resolve()
    rows, latency = [], None
    for case in select(out, args.cases):
        for run in args.runs:
            run_dir = out / case.name / run
            if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
                rows.append((case, run, {"first_word": None}))
                continue
            row = gap_row(dl.load_level_events(run_dir), case)
            rows.append((case, run, row))
            if case.mode == "solo" and latency is None:
                latency = solo_latency(row)
    print(format_table(rows, latency))
    return 0


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
    r.add_argument("--cases", nargs="*", default=[])
    t = sub.add_parser("table")
    t.add_argument("out_dir")
    t.add_argument("runs", nargs="+", help="run directory names, e.g. h1.hw e1.emu")
    t.add_argument("--cases", nargs="*", default=[])
    args = ap.parse_args(argv)
    return {"build": cmd_build, "run": cmd_run, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
