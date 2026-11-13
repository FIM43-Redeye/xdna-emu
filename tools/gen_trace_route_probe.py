#!/usr/bin/env python3
"""Which stream switches charge a trace packet the adaptive-gate wake cycle, and when are the
Sync tokens of a long idle stretch written?

Two families of cases, each a no-data design whose only stream traffic is trace packets, timed by
one core program (the event schedule of `gen_trace_wake_probe`: dense packets 48 cycles apart and
spaced ones after 100 / 120 / 135 / 150 / 400 idle cycles). Arrivals are dated from the carrier
shim's own traced DMA_S2MM_1_STREAM_STARVATION level, as in `gen_trace_wake_probe`.

Wake cases (the source trace unit and the route differ; the carrier compares spaced packets with
dense ones of the same source, so the unknown offset between the two timers cancels):
  w0  core (0,2) trace -> memtile (0,1) -> shim (0,0) S2MM 1          (the trace-wake design)
  w1  core (0,2) trace -> ... -> shim (1,0) S2MM 1 (egress column 1; the route is printed by build)
  w2  memtile (0,1) trace -> shim (0,0) S2MM 1. The memtile records BROADCAST_1 / BROADCAST_2,
      which the core module raises from its two instruction events (Event_Broadcast1 / 2 written by
      the runtime sequence), so the memtile's frames follow the same schedule. The core is not traced.
  w3  shim (0,0) trace -> shim (1,0) S2MM 1. Shim (0,0) records BROADCAST_A_1 / BROADCAST_A_2 the
      same way; the carrier is the next column's shim.

Sync cases (core (0,2) -> shim (0,0); a frame's cycles field is 18 bits, so 2^18 idle cycles or more
before a frame are written as Sync tokens):
  sc  head, dense packets, then a packet filled with 27 one-byte frames, a wait of SYNC_SHORT cycles
      and one closing frame (a 3-byte Single2 that does not fit: Skip fill), then dense packets.
  si  the same with a wait of SYNC_LONG > 2^18 cycles: the 28th byte is the Sync. Written one per
      timer wrap, the packet leaves about 2^18 cycles after its last frame; written with the frame
      that ends the stretch, it leaves with the closing frame.
  se  the 27-byte packet, then nothing until the trace stops (SYNC_PAD_WRITES writes later): a Sync
      written at the wrap shows as an arrival before the stop; one written at the stop does not.

Shim data-word cases (a host-fed transfer from the shim's own DMA MM2S). Shim (0,0) MM2S 0 reads
DATA_WORDS words from the host into S2MM 0 of the same shim (d1q) or of memtile (0,1) (d2q); the trace
goes to the next column's shim, so shim (0,0) has both S2MM channels free. Every task finishes before
the sequence ends: each sink BD takes exactly the words its source sends, the memtile BD has no
successor, and the sequence waits for the shim tasks' completion tokens; every BD starts at offset 0
of the one host buffer. Measured in the shim's own trace: MM2S 0 START_TASK and STREAM_BACKPRESSURE
and the destination S2MM's STREAM_STARVATION fall (the first word in; for d2q the memtile's).

    python3 tools/gen_trace_route_probe.py build OUT_DIR [CASE ...]
    python3 tools/gen_trace_route_probe.py run OUT_DIR CASE TAG --backend hw|emu
    python3 tools/gen_trace_route_probe.py table OUT_DIR CASE RUN [RUN ...]

The runtime sequence holds only plain writes and the go-lock set (checked by `build`); nothing
waits on a host buffer, so a run cannot stall on the design (see `gen_trace_wake_probe`).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_lock_pulse_probe as lp                                       # noqa: E402
import gen_trace_wake_probe as tw                                       # noqa: E402

# Event numbers (AM025 event database, aie2): the core module's instruction events and the
# broadcast events they become in the memtile and the shim.
INSTR_EVENT_0, INSTR_EVENT_1 = 33, 34
CORE_EVENT_BROADCAST = 0x34010          # Event_Broadcast0; channel n at + 4n
BCAST_CHANNELS = (1, 2)                 # not 14 / 15, which start and stop the trace
SYNC_FIELD = 1 << 18
SYNC_SHORT = 3000
SYNC_LONG = SYNC_FIELD + 3000
SYNC_PAD_WRITES = 3500                  # outlasts the 2^18 wait (about 103 cycles per write)
SCRATCH_REG = 0x000340F0                # Timer_Trig_Event_Low_Value of the carrier shim, nothing uses it here
DENSE_LEAD = 3                          # dense packets before the 27-byte packet (timer offset)
PT = {"core": 0, "shim": 2, "memtile": 3}


@dataclass(frozen=True)
class Case:
    name: str
    device: str
    source: tuple[str, int, int]        # (kind, col, row) of the traced source
    carrier_col: int
    sync_wait: int | None = None        # sync cases: cycles between the 27-byte packet and the closing frame
    sync_end: bool = False              # se: no closing frame
    pad_writes: int = tw.PAD_WRITES
    notes: str = ""
    data_dest: str | None = None        # data cases: "shim" (S2MM 0 of shim (0,0)) or "memtile"

    @property
    def broadcast(self) -> bool:
        return self.source[0] != "core"


CASES = {c.name: c for c in [
    Case("w0", "npu1_1col", ("core", 0, 2), 0),
    Case("w1", "npu1_2col", ("core", 0, 2), 1),
    Case("w2", "npu1_1col", ("memtile", 0, 1), 0),
    Case("w3", "npu1_2col", ("shim", 0, 0), 1),
    Case("sc", "npu1_1col", ("core", 0, 2), 0, sync_wait=SYNC_SHORT, pad_writes=SYNC_PAD_WRITES),
    Case("si", "npu1_1col", ("core", 0, 2), 0, sync_wait=SYNC_LONG, pad_writes=SYNC_PAD_WRITES),
    Case("se", "npu1_1col", ("core", 0, 2), 0, sync_end=True, pad_writes=SYNC_PAD_WRITES),
    Case("d1q", "npu1_2col", ("shim", 0, 0), 1, data_dest="shim"),
    Case("d2q", "npu1_2col", ("shim", 0, 0), 1, data_dest="memtile"),
]}
DATA_WORDS = 16
HOST_WORDS = 16384                       # as in the runs that finished cleanly
SHIM_DATA_EVENTS = ["DMA_MM2S_0_START_TASK", "DMA_MM2S_0_STREAM_BACKPRESSURE", "DMA_S2MM_0_STREAM_STARVATION",
                    "DMA_MM2S_0_FINISHED_TASK", "DMA_S2MM_0_FINISHED_TASK"]
MEMTILE_DATA_EVENTS = ["DMA_S2MM_SEL0_STREAM_STARVATION", "DMA_S2MM_SEL0_FINISHED_TASK"]


# -- core programs ----------------------------------------------------------------------------

def sync_asm(case: Case) -> str:
    """Head packet, DENSE_LEAD dense packets, 27 one-byte frames, then the wait and the closing
    frame (sc / si) followed by a dense tail, or nothing (se)."""
    e = tw._Emitter()
    e.lines += ["  mov r3, #-1", *lp.nops(2), f"  acq #{tw.OWN + tw.LOCK_GO}, r3", *lp.nops(4)]
    for i in range(18):                              # head: Start + Single2 + 17 Single0 = 28
        e.wait_before_event(tw.HEAD_DELAY if i == 0 else tw.SPACING)
        e.event()
    for _ in range(DENSE_LEAD * tw.PACKET_BYTES + 27):
        e.wait_before_event(tw.SPACING)
        e.event()
    if case.sync_end:
        return lp.frame(e.lines)
    e.wait_before_event(case.sync_wait)
    e.event()
    for _ in range(tw.TAIL_PACKETS * tw.PACKET_BYTES):
        e.wait_before_event(tw.SPACING)
        e.event()
    return lp.frame(e.lines)


def core_program(case: Case) -> str:
    if case.sync_wait is not None or case.sync_end:
        return sync_asm(case)
    return tw.core_asm(tw.default_plan())


# -- design -----------------------------------------------------------------------------------

def _tile_name(col: int, row: int) -> str:
    return f"%t{col}{row}"


def trace_decls(case: Case) -> tuple[list[str], list[str]]:
    """aie.trace declarations and their start_config lines: the source unit, then the carrier."""
    kind, col, row = case.source
    if kind == "core":
        events = ["INSTR_EVENT_0", "INSTR_EVENT_1"]
    elif kind == "memtile":
        events = [f"BROADCAST_{c}" for c in BCAST_CHANNELS]
    else:
        events = [f"BROADCAST_A_{c}" for c in BCAST_CHANNELS]
    ptype = {"core": "core", "memtile": "memtile", "shim": "shimtile"}[kind]
    decl = [f"    aie.trace @trace_src({_tile_name(col, row)}) {{"]
    if kind == "core":
        decl.append('      aie.trace.mode "Event-Time"')
    decl += [f"      aie.trace.packet id = 1 type = {ptype}",
             *[f'      aie.trace.event <"{ev}">' for ev in events],
             "      aie.trace.start broadcast = 15", "      aie.trace.stop broadcast = 14", "    }",
             f"    aie.trace @trace_carrier({_tile_name(case.carrier_col, 0)}) {{",
             "      aie.trace.packet id = 2 type = shimtile",
             f'      aie.trace.event <"{tw.SHIM_EVENT}">',
             "      aie.trace.start broadcast = 15", "      aie.trace.stop broadcast = 14", "    }"]
    return decl, ["      aie.trace.start_config @trace_src", "      aie.trace.start_config @trace_carrier"]


def design_mlir(case: Case) -> str:
    kind, scol, srow = case.source
    tiles = {(0, 2), (0, 1) if kind == "memtile" else None, (scol, srow), (case.carrier_col, 0), (0, 0)}
    tiles.discard(None)
    tile_lines = [f"    {_tile_name(c, r)} = aie.tile({c}, {r})" for c, r in sorted(tiles, key=lambda t: (t[1], t[0]))]
    bcast = []
    if case.broadcast:
        for ch, ev in zip(BCAST_CHANNELS, (INSTR_EVENT_0, INSTR_EVENT_1)):
            bcast += [f"      %c_ba{ch} = arith.constant {CORE_EVENT_BROADCAST + 4 * ch} : i32",
                      f"      %c_bv{ch} = arith.constant {ev} : i32",
                      f"      aiex.npu.write32(%c_ba{ch}, %c_bv{ch}) {{column = 0 : i32, row = 2 : i32}} : i32, i32"]
    decl, starts = trace_decls(case)
    pad = "\n".join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32"
                    for _ in range(case.pad_writes))
    nl = "\n"
    return f"""module {{
  aie.device({case.device}) {{
{nl.join(tile_lines)}
    %scratch = aie.buffer(%t02) {{address = {tw.SCRATCH_ADDR} : i32, sym_name = "scratch"}} : memref<4xi32>
    %lGo = aie.lock(%t02, {tw.LOCK_GO}) {{init = 0 : i32, sym_name = "lGo"}}
    func.func private @k_core() attributes {{link_with = "core.o"}}
    aie.core(%t02) {{
      func.call @k_core() : () -> ()
      aie.end
    }}
{nl.join(decl)}
    aie.runtime_sequence(%dummy: memref<4xi32>) {{
      aie.trace.host_config {{buffer_size = 65536 : i32, egress_shim_col = {case.carrier_col} : i32}}
{nl.join(starts)}
      %c_zero = arith.constant 0 : i32
      %c_one = arith.constant 1 : i32
{nl.join(bcast)}
      aiex.set_lock(%lGo, %c_one)
{pad}
    }}
  }}
}}
"""


def data_design_mlir(case: Case) -> str:
    """Shim (0,0) MM2S 0 -> S2MM 0 of shim (0,0) (d1q) or memtile (0,1) (d2q). Every task finishes:
    the sink takes exactly the source's words, the memtile BD has no successor, and the sequence
    waits for the shim tasks' completion tokens. Every BD starts at offset 0 of the host buffer."""
    mt = case.data_dest == "memtile"
    nl = "\n"
    tiles = ["    %t00 = aie.tile(0, 0)", "    %t10 = aie.tile(1, 0)"] + (["    %t01 = aie.tile(0, 1)"] if mt else [])
    flows = [f"    aie.flow(%t00, DMA : 0, {'%t01' if mt else '%t00'}, DMA : 0)"]
    allocs = ["    aie.shim_dma_allocation @in0 (%t00, MM2S, 0)"]
    if not mt:
        allocs.append("    aie.shim_dma_allocation @out0 (%t00, S2MM, 0)")
    mem = []
    if mt:
        mem = ["    %mbuf = aie.buffer(%t01) {sym_name = \"mbuf\"} : memref<16xi32>",
               "    %mdma = aie.memtile_dma(%t01) {",
               "      %s0 = aie.dma_start(S2MM, 0, ^bb1, ^bb99)",
               "    ^bb1:",
               f"      aie.dma_bd(%mbuf : memref<16xi32> offset = 0 len = {DATA_WORDS})",
               "      aie.next_bd ^bb99",
               "    ^bb99:",
               "      aie.end",
               "    }"]
    decl = ["    aie.trace @trace_src(%t00) {", "      aie.trace.packet id = 1 type = shimtile",
            *[f'      aie.trace.event <"{ev}">' for ev in SHIM_DATA_EVENTS],
            "      aie.trace.start broadcast = 15", "      aie.trace.stop broadcast = 14", "    }"]
    starts = ["      aie.trace.start_config @trace_src"]
    if mt:
        decl += ["    aie.trace @trace_mt(%t01) {", "      aie.trace.packet id = 2 type = memtile",
                 *[f'      aie.trace.event <"{ev}">' for ev in MEMTILE_DATA_EVENTS],
                 "      aie.trace.start broadcast = 15", "      aie.trace.stop broadcast = 14", "    }"]
        starts.append("      aie.trace.start_config @trace_mt")
    m = f"memref<{HOST_WORDS}xi32>"

    def memcpy(n: int, meta: str, ident: int, token: bool) -> str:
        tok = ", issue_token = true" if token else ""
        return (f"      aiex.npu.dma_memcpy_nd(%buf[%c0, %c0, %c0, %c0] [%c1, %c1, %c1, %c{n}] [%c0, %c0, %c0, %c1]) "
                f"{{id = {ident} : i64, metadata = @{meta}{tok}}} : {m}")
    seq, waits = [], []
    if not mt:
        seq.append(memcpy(DATA_WORDS, "out0", 1, True))
        waits.append("      aiex.npu.dma_wait {symbol = @out0}")
    seq.append(memcpy(DATA_WORDS, "in0", 0, mt))
    if mt:
        waits.append("      aiex.npu.dma_wait {symbol = @in0}")
    pad = nl.join("      aiex.npu.write32(%c_zero, %c_one) {column = 1 : i32, row = 0 : i32} : i32, i32"
                  for _ in range(case.pad_writes))
    return f"""module {{
  aie.device({case.device}) {{
{nl.join(tiles)}
{nl.join(flows)}
{nl.join(allocs)}
{nl.join(mem)}
{nl.join(decl)}
    aie.runtime_sequence(%buf: {m}) {{
      aie.trace.host_config {{buffer_size = 65536 : i32, egress_shim_col = {case.carrier_col} : i32}}
{nl.join(starts)}
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c{DATA_WORDS} = arith.constant {DATA_WORDS} : i64
      %c_zero = arith.constant {SCRATCH_REG} : i32
      %c_one = arith.constant 0 : i32
{nl.join(seq)}
{pad}
{nl.join(waits)}
    }}
  }}
}}
"""


def data_rows(case: Case, run_dir: Path) -> dict:
    """Cycles of the measured first word: the MM2S 0 MEMORY_STARVATION fall, any MM2S 0
    STREAM_BACKPRESSURE cycles, the destination's STREAM_STARVATION fall (first word in)."""
    import gen_dma_lock_probe as dl
    ev = dl.load_level_events(run_dir)
    def cycles(pkt, name):
        return sorted({e.ts for e in ev if e.pkt_type == pkt and e.name == name})
    def first_fall(cs, after=0):
        """The cycle after the end of the first asserted run that reaches past `after`."""
        cs = [c for c in cs if c >= after]
        if not cs:
            return None
        for a, b in zip(cs, cs[1:]):
            if b != a + 1:
                return a + 1
        return cs[-1] + 1
    start = cycles(PT["shim"], "DMA_MM2S_0_START_TASK")
    t0 = start[0] if start else 0
    bp = [c for c in cycles(PT["shim"], "DMA_MM2S_0_STREAM_BACKPRESSURE") if c >= t0]
    if case.data_dest == "memtile":
        arr = first_fall(cycles(PT["memtile"], "DMA_S2MM_SEL0_STREAM_STARVATION"))
    else:
        arr = first_fall(cycles(PT["shim"], "DMA_S2MM_0_STREAM_STARVATION"), t0)
    out = {"start": t0, "backpressure": bp[:8], "arrival": arr,
           "arr_minus_offer": None if arr is None or not bp else arr - bp[0]}
    return out


# -- decoding ---------------------------------------------------------------------------------

def stream_timeline(packets: list[dict], key: tuple[int, int, int]) -> dict[int, dict]:
    """order -> {first, last, frames} for the packets of one source: the cycles (that tile's
    timeline) of its first and last event frame and its frame cycles, decoding mode 0 by hand
    (Start sets the timer, an event frame adds 1 + its delta, a Sync adds 2^18, a Repeat right
    after a Sync adds count * 2^18, a Repeat after a frame repeats it)."""
    out: dict[int, dict] = {}
    t, after_sync, prev_delta = 0, False, None
    for p in packets:
        if (p["col"], p["row"], p["pkt_type"]) != key:
            continue
        b, i, frames, toks = p["bytes"], 0, [], []
        while i < len(b):
            x = b[i]
            n = tw.frame_len(x)
            if (x & 0xFB) == 0xF0:
                t = int.from_bytes(bytes(b[i + 1:i + 8]), "big")
                after_sync, prev_delta = False, None
                toks.append("Start")
            elif x == 0xFF:
                t += SYNC_FIELD
                after_sync = True
                toks.append("Sync")
            elif x == 0xFE or (x & 0xFC) == 0xDC:
                toks.append("Skip" if x == 0xFE else "End")
            elif (x & 0xF0) == 0xE0 or (x & 0xFC) == 0xD8:
                count = x & 0x0F if (x & 0xF0) == 0xE0 else ((x & 3) << 8) | b[i + 1]
                if after_sync:
                    t += count * SYNC_FIELD
                elif prev_delta is not None:
                    for _ in range(count):
                        t += 1 + prev_delta
                        frames.append(t)
                toks.append(f"Rep({count})")
            else:
                if x < 0x80:
                    delta = x & 0x0F
                elif (x & 0xE0) == 0x80:
                    delta = ((x & 3) << 8) | b[i + 1]
                elif (x & 0xE0) == 0xA0:
                    delta = ((x & 3) << 16) | (b[i + 1] << 8) | b[i + 2]
                elif (x & 0xF0) == 0xC0:
                    delta = b[i + 1] & 0x0F
                elif (x & 0xFC) == 0xD0:
                    delta = ((b[i + 1] & 3) << 8) | b[i + 2]
                else:
                    delta = ((b[i + 1] & 3) << 16) | (b[i + 2] << 8) | b[i + 3]
                t += 1 + delta
                after_sync, prev_delta = False, delta
                frames.append(t)
                toks.append(tw.frame_type(x))
            i += n
        out[p["order"]] = {"frames": frames, "toks": toks}
    return out


def carrier_starved(run_dir: Path, carrier_col_raw: int) -> list[int]:
    import gen_dma_lock_probe as dl
    return [e.ts for e in dl.load_level_events(run_dir)
            if e.pkt_type == tw.PT_SHIM and e.name == tw.SHIM_EVENT and getattr(e, "col", carrier_col_raw) == carrier_col_raw]


def source_rows(case: Case, run_dir: Path) -> tuple[list[dict], list[dict]]:
    """One row per packet of the source: completion (last frame of the packet, or for a packet
    whose last byte is a Sync the cycle the Sync stretch began), arrival, delay, idle before it."""
    raw = (run_dir / "trace_raw.bin").read_bytes()
    packets = tw.split_packets(list(struct.unpack(f"<{len(raw) // 4}I", raw)))
    kind, col, row = case.source
    keys = {(p["col"], p["row"], p["pkt_type"]) for p in packets if p["pkt_type"] == PT[kind]}
    if kind == "shim":
        keys = {k for k in keys if k[0] != max(k[0] for k in keys)} or keys   # the carrier is the higher column
    if len(keys) != 1:
        raise SystemExit(f"cannot tell the source stream apart: {sorted(keys)}")
    key = keys.pop()
    carrier_raw = key[0] + (case.carrier_col - col)
    slots = tw.arrival_slots(carrier_starved(run_dir, carrier_raw))
    tl = stream_timeline(packets, key)
    rows, prev = [], None
    for p in packets:
        if p["order"] not in tl:
            continue
        info = tl[p["order"]]
        comp = info["frames"][-1] if info["frames"] else None
        r = {"order": p["order"], "completion": comp, "toks": info["toks"], "arrival": None, "delay": None,
             "idle": None if prev is None or comp is None else comp - prev - tw.PACKET_CYCLES,
             "last_tok": next((x for x in reversed(info["toks"]) if x != "Skip"), "?"),
             "prev_src": None, "since_any": None}
        if p["order"] < len(slots):
            r["arrival"] = slots[p["order"]]
            if comp is not None:
                r["delay"] = r["arrival"] - comp
            if p["order"] > 0:
                r["since_any"] = slots[p["order"]] - (slots[p["order"] - 1] + tw.PACKET_CYCLES)
                q = packets[p["order"] - 1]
                r["prev_src"] = "src" if (q["col"], q["row"], q["pkt_type"]) == key else "carrier"
        r["phase"] = None if r["idle"] is None else tw.phase_of(r["idle"])
        r["contaminated"] = r["prev_src"] == "carrier" and r["since_any"] is not None and r["since_any"] < tw.GATE_CYCLES
        prev = comp if comp is not None else prev
        rows.append(r)
    return rows, packets


def format_rows(rows: list[dict]) -> str:
    lines = [f"{'pkt':>3} {'phase':<6} {'idle':>7} {'completion':>10} {'last':<8} {'arrival':>9} {'delay':>7} "
             f"{'since_any':>9} {'prev':<7} note"]
    for r in rows:
        note = "CONTAMINATED" if r["contaminated"] else ("no arrival" if r["arrival"] is None else "")
        lines.append(f"{r['order']:>3} {r['phase'] or '-':<6} {tw._num(r['idle'], 7)} {tw._num(r['completion'], 10)} "
                     f"{r['last_tok']:<8} {tw._num(r['arrival'], 9)} {tw._num(r['delay'], 7)} "
                     f"{tw._num(r['since_any'], 9)} {r['prev_src'] or '-':<7} {note}")
    return "\n".join(lines)


def wake_summary(rows: list[dict]) -> str:
    """Spaced packets against the dense median, completed by Single0 / Single1 frames only."""
    clean = [r for r in rows if r["delay"] is not None and r["idle"] is not None and not r["contaminated"]
             and r["last_tok"] in ("Single0", "Single1")]
    dense = [r["delay"] for r in clean if r["phase"] == "dense"]
    if not dense:
        return "no clean dense rows"
    base = median(dense)
    out = [f"dense baseline: median delay {base} over {len(dense)} packets (hist {tw._hist(dense)})"]
    bins: dict = {}
    for r in clean:
        if r["phase"] != "dense":
            bins.setdefault(tw.idle_bin(r["idle"]), []).append(r["delay"])
    for lo, hi in tw.BINS:
        if (lo, hi) in bins:
            d = bins[(lo, hi)]
            nm = f"{lo}..{hi - 1}" if hi < 10 ** 9 else f"{lo}+"
            out.append(f"idle {nm:>8}: n={len(d)} median {median(d)} = dense {median(d) - base:+.1f}  hist {tw._hist(d)}")
    return "\n".join(out)


def sync_summary(case: Case, rows: list[dict]) -> str:
    """The 27-byte packet (the one after DENSE_LEAD dense packets): its arrival against its last
    frame L, and against the closing frame C, with the timer offset taken from the dense packets."""
    dense = [r for r in rows if r["phase"] == "dense" and r["delay"] is not None and r["last_tok"] == "Single0"]
    if not dense or len(rows) < DENSE_LEAD + 2:
        return "not enough packets"
    off = median(r["delay"] - 11 for r in dense)       # a dense packet arrives 11 after its frame
    tgt = rows[DENSE_LEAD + 1]
    L = tgt["completion"]
    out = [f"timer offset (dense delay - 11): {off} over {len(dense)} packets",
           f"27-byte packet #{tgt['order']}: tokens {' '.join(tgt['toks'][-4:])}; last frame L = {L}"]
    if tgt["arrival"] is None:
        out.append("  no arrival slot before the trace stopped")
    else:
        a = tgt["arrival"] - off
        out.append(f"  arrival (source timeline) = {a}: L + {a - L}")
    return "\n".join(out)


# -- commands ---------------------------------------------------------------------------------

def cmd_build(args) -> int:
    out = Path(args.out_dir).resolve()
    for name in args.cases or list(CASES):
        case = CASES[name]
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        if case.data_dest is None:
            lp.assemble(core_program(case), d / "core.o", "k_core")
        (d / "aie.mlir").write_text(data_design_mlir(case) if case.data_dest else design_mlir(case))
        lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
               "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
              cwd=d, log=d / "aiecc.log")
        if case.data_dest:
            print(f"{name}: built (the sequence waits for its DMA completion tokens)")
        else:
            ops = tw.check_sequence_is_plain_writes((d / "insts.bin").read_bytes())
            print(f"{name}: sequence ops (opcode: count): {ops}")
    return 0


def cmd_run(args) -> int:
    out = Path(args.out_dir).resolve()
    if not lp.run_case(out, args.case, args.tag, args.backend, output=False):
        print(f"{args.case}: run did not complete; stop")
        return 1
    return 0


def cmd_table(args) -> int:
    out = Path(args.out_dir).resolve()
    case = CASES[args.case]
    for run in args.runs:
        run_dir = out / args.case / run
        print(f"== {args.case} {run}")
        if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
            print("no trace_raw.bin / events.json")
            continue
        if case.data_dest:
            print(json.dumps(data_rows(case, run_dir)))
            continue
        rows, packets = source_rows(case, run_dir)
        print(f"{len(packets)} packets in the buffer, {len(rows)} from the source")
        print(format_rows(rows))
        print(sync_summary(case, rows) if case.sync_wait is not None or case.sync_end else wake_summary(rows))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out_dir")
    b.add_argument("cases", nargs="*", default=[])
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("case", choices=list(CASES))
    r.add_argument("tag")
    r.add_argument("--backend", choices=("hw", "emu"), required=True)
    t = sub.add_parser("table")
    t.add_argument("out_dir")
    t.add_argument("case", choices=list(CASES))
    t.add_argument("runs", nargs="+")
    args = ap.parse_args(argv)
    return {"build": cmd_build, "run": cmd_run, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
