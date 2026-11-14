#!/usr/bin/env python3
"""Does a trace packet pay a wake cycle at a stream switch that has been idle for 128 cycles?

Each tile's stream switch is adaptively clock-gated after 2^Abort_Period = 128 idle cycles. On
the data path silicon shows a one-cycle refusal of the first word offered to a gated switch. For
trace packets the captures we have cannot tell: every non-saturated packet in them is completed by
an event that is itself data traffic through the same switch, so the switch is awake by the time
the packet leaves. This probe has no data traffic at all, so the only thing crossing the switches
of the route is trace packets, and the idle time before each of them is set by the core program.

Design (one column, `npu1_1col`, tile (0,2) only):
  core (0,2): raises `event #0` / `event #1` (INSTR_EVENT_0 / _1, the two traced core events) at
    exact cycle spacing. Its trace packets travel core tile -> memtile -> shim -> S2MM channel 1
    (the route the trace injector builds). The shim's own trace unit records one event,
    DMA_S2MM_1_STREAM_STARVATION, a level that is high while the channel waits for data: each
    packet arriving at the channel shows as an 8-cycle gap in it, so arrivals are dated in the
    shim's timeline. Nothing else is traced: no grounding slot, no periodic event.
  Nothing else exists in the design: no DMA channel is programmed except the trace S2MM the
  injector adds, no data buffer, no flow.

Frame accounting (mode 0, see trace_decoder/modes/mode0.py). A trace packet carries 28 payload
bytes. The first packet starts with an 8-byte Start frame. A frame whose cycle delta (time since
the previous frame) is at most 15 and carries one event is a one-byte Single0; a delta up to 1023
makes it a two-byte Single1, up to 262143 a three-byte Single2. The two event ids alternate, so no
two consecutive frames are equal and the encoder never folds them into a Repeat. A packet leaves
the tile when its 28th byte is written, so the event that writes that byte dates the packet.
  - head packet: Start (8) + one Single2 (3) + 17 Single0 = 28 (18 events). The Single2 is guaranteed by a
    first-event delay above 1023 cycles (HEAD_DELAY); the trace unit was armed long before.
  - dense packet: 28 Single0, events SPACING cycles apart: 28 * SPACING = 56 cycles per packet, so
    every switch sees the next packet well inside the 128-cycle gate.
  - spaced packet: 26 Single0 in a burst (SPACING apart), then the completing event as a Single1
    (2 bytes: 26 + 2 = 28) after a delay chosen so that the packet completes `gap` cycles after
    the previous packet completed. The burst fills the packet quickly; the long wait is in the
    delta of the completing frame, so the switches are idle for the whole wait. The idle time at
    the source switch is the completion-to-completion gap minus the PACKET_CYCLES the previous
    packet occupied the switch.
  Spacings in the plan: idle about 100, 120, 135, 150 around the threshold and 400 well beyond it.
  The tail is a loop of dense packets, long enough for the shim's own trace (a few bytes per
  packet arrival) to fill packets and carry the arrival dates of the packets before it out.

The table (`table`) pairs every core packet, in raw buffer order, with the next 8-cycle
non-starved slot of the shim's STREAM_STARVATION level, as gen_dma_lock_probe-style level events.
Per core packet it prints the phase (dense, near: idle below the 128-cycle gate, gated: at or above it), the idle since the
previous core packet, the completing byte's cycle in the core's timeline, the completing frame's
type (anything but Single0/Single1 means the byte accounting above broke), the arrival cycle in
the shim's timeline, delay = arrival - completion, and since_any: the starved cycles between the
previous word at S2MM_1 (whatever its source) and this arrival, with the source of that previous
packet. The two timelines have an unknown constant offset, so absolute delays mean little; the
comparison is the delay of near and gated rows against dense rows (the summary prints it). The shim's own
packets also cross the shim switch: rows whose previous packet is a shim packet within 128 cycles
are marked contaminated and left out of the summary.

Why it cannot stall. The cores start when the array is configured, before the trace unit is armed,
so the core first waits on a go lock that the runtime sequence sets (`aiex.set_lock`, a
command-processor write) after the trace is armed. Nothing in the design waits on a host buffer,
a host DMA or any host signal: the core's only wait is that go lock, which the sequence always
sets, after which it runs a fixed program of events and delays and returns. The runtime sequence
contains only plain writes and the lock set: it waits on nothing. The trace
S2MM channel is a sink that writes into a buffer and has no consumer to wait for.

    python3 tools/gen_trace_wake_probe.py build OUT_DIR
    python3 tools/gen_trace_wake_probe.py run OUT_DIR TAG --backend hw|emu
    python3 tools/gen_trace_wake_probe.py table OUT_DIR RUN [RUN ...]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_dma_lock_probe as dl                                         # noqa: E402
import gen_lock_pulse_probe as lp                                       # noqa: E402

OWN = dl.OWN                       # raw ID base of the core's own memory module
LOCK_GO = 0
CASE = "tw"
SPACING = 2                        # cycles between burst events: one nop between two `event`s
PACKET_BYTES = 28
PACKET_CYCLES = 8                  # cycles a packet (header + 7 words) occupies a switch
GATE_CYCLES = 128                  # 2^Abort_Period
DENSE_MAX_IDLE = 64                # below this a packet is one of the back-to-back ones
SINGLE0_MAX_DELTA = 15
SINGLE1_MAX_DELTA = 1023
HEAD_DELAY = 2000                  # first-event delay: above 1023, so the first frame is a Single2
LOOP_CYCLES = 9                    # bundles per delay-loop iteration (see lp.delay_loop)
LOOP_FIXED = 4                     # loop set-up bundles outside the iterations
TAIL_PACKETS = 40
PAD_WRITES = 1500                  # command-processor writes after the go lock: outlast the program
SCRATCH_ADDR = 0xC000
SHIM_EVENT = "DMA_S2MM_1_STREAM_STARVATION"
CORE_EVENTS = ["INSTR_EVENT_0", "INSTR_EVENT_1"]
PT_CORE, PT_SHIM = 0, 2


@dataclass(frozen=True)
class PacketPlan:
    """One core trace packet. idle None: dense (28 events SPACING apart). Otherwise the idle
    cycles at the source switch before it: a burst of 26 events and a completing event after
    the wait. kind 'head' is the first packet (Start + Single2 + 17 more events: 18 in all)."""
    kind: str                      # head | dense | spaced
    idle: int | None = None

    @property
    def burst(self) -> int:
        return {"head": 18, "dense": 28, "spaced": 26}[self.kind]

    @property
    def gap(self) -> int | None:
        """Completion-to-completion cycles from the previous packet (spaced only)."""
        return None if self.idle is None else self.idle + PACKET_CYCLES


def default_plan() -> list[PacketPlan]:
    plan = [PacketPlan("head")] + [PacketPlan("dense")] * 5
    plan += [PacketPlan("spaced", 400)] * 5 + [PacketPlan("dense")] * 3
    for idle in (100, 120, 135, 150):
        plan += [PacketPlan("spaced", idle)] * 3 + [PacketPlan("dense")] * 2
    plan += [PacketPlan("spaced", 400)] * 3
    return plan


def completion_delta(p: PacketPlan) -> int | None:
    """Cycle delta of the completing frame of a spaced packet (after its burst)."""
    return None if p.gap is None else p.gap - p.burst * SPACING


def validate_plan(plan: list[PacketPlan]) -> None:
    if not plan or plan[0].kind != "head":
        raise ValueError("the plan starts with the head packet")
    for p in plan:
        if p.kind == "spaced":
            d = completion_delta(p)
            if not SINGLE0_MAX_DELTA < d <= SINGLE1_MAX_DELTA:
                raise ValueError(f"idle {p.idle}: completing delta {d} is not a Single1 (16..1023)")
        elif p.kind not in ("head", "dense"):
            raise ValueError(f"unknown packet kind {p.kind}")


def wait_bundles(n: int, label: str) -> list[str]:
    """About n bundles of nothing: nops when short, otherwise a counted loop plus nops."""
    if n <= 0:
        return []
    if n < LOOP_FIXED + 2 * LOOP_CYCLES:
        return lp.nops(n)
    iters = (n - LOOP_FIXED) // LOOP_CYCLES
    rest = n - LOOP_FIXED - iters * LOOP_CYCLES
    return lp.delay_loop(iters, label) + lp.nops(rest)


class _Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.events = 0
        self.waits = 0

    def event(self) -> None:
        self.lines.append(f"  event #{self.events % 2}")
        self.events += 1

    def wait_before_event(self, delta: int) -> None:
        """Pad so the next event lands `delta` cycles after the previous one (one bundle each)."""
        self.waits += 1
        self.lines += wait_bundles(delta - 1, f".Lw{self.waits}")


def core_asm(plan: list[PacketPlan] | None = None, tail_packets: int = TAIL_PACKETS) -> str:
    """Go-lock wait, then the planned events, then a loop of dense packets."""
    plan = plan or default_plan()
    validate_plan(plan)
    e = _Emitter()
    e.lines += ["  mov r3, #-1", *lp.nops(2), f"  acq #{OWN + LOCK_GO}, r3", *lp.nops(4)]
    for p in plan:
        for i in range(p.burst):
            first = p.kind == "head" and i == 0
            e.wait_before_event(HEAD_DELAY if first else SPACING)
            e.event()
        if p.kind == "spaced":
            e.wait_before_event(completion_delta(p))
            e.event()
    # tail: a packet is 28 events, so every iteration completes one packet; the loop-back delta
    # (jnz + 5 delay slots) stays under 16 cycles so every frame is still a Single0
    e.lines += ["  movxm r0, #%d" % tail_packets, *lp.nops(2), ".p2align 4", ".Ltail:", "  add r0, r0, #-1"]
    for i in range(PACKET_BYTES):
        e.lines += [*lp.nops(SPACING - 1), f"  event #{e.events % 2}"]
        e.events += 1
    e.lines += ["  jnz r0, #.Ltail", *lp.nops(5)]
    return lp.frame(e.lines)


def design_mlir(pad_writes: int = PAD_WRITES) -> str:
    pad = "\n".join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32"
                    for _ in range(pad_writes))
    return f"""module {{
  aie.device(npu1_1col) {{
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %scratch = aie.buffer(%t02) {{address = {SCRATCH_ADDR} : i32, sym_name = "scratch"}} : memref<4xi32>
    %lGo = aie.lock(%t02, {LOCK_GO}) {{init = 0 : i32, sym_name = "lGo"}}
    func.func private @k_core() attributes {{link_with = "core.o"}}
    aie.core(%t02) {{
      func.call @k_core() : () -> ()
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


def inject_args() -> list[str]:
    return ["--core-grounding", ",".join(CORE_EVENTS), "--core-sweep-events", ",".join(CORE_EVENTS),
            "--shim-sweep-events", SHIM_EVENT, "--shim-grounding", ""]


# -- the compiled runtime sequence ----------------------------------------------------

# Transaction-binary opcodes the compiled sequence may contain: plain writes (single, block,
# masked), and the host-buffer address patch the trace injector adds. Any other opcode is
# refused, so the sequence's timing is that of its writes alone.
TXN_WRITE, TXN_BLOCKWRITE, TXN_MASKWRITE, TXN_PATCH = 0, 1, 3, 0x81
TXN_FIXED_SIZES = {TXN_WRITE: 24, TXN_MASKWRITE: 28}


def check_sequence_is_plain_writes(insts: bytes) -> dict[int, int]:
    """Walk a compiled instruction stream and return {opcode: count}; raise ValueError on any
    opcode outside the plain-write set, or a stream that does not parse to its stated length."""
    ops: dict[int, int] = {}
    n_ops = struct.unpack_from("<I", insts, 8)[0]
    off = 16
    for _ in range(n_ops):
        op = insts[off]
        if op == TXN_BLOCKWRITE:
            size = struct.unpack_from("<I", insts, off + 12)[0]
        elif op == TXN_PATCH:
            size = struct.unpack_from("<I", insts, off + 4)[0]
        elif op in TXN_FIXED_SIZES:
            size = TXN_FIXED_SIZES[op]
        else:
            raise ValueError(f"opcode {op:#x} at offset {off} is not a plain write")
        ops[op] = ops.get(op, 0) + 1
        off += size
    if off != len(insts):
        raise ValueError(f"stream parses to {off} bytes, file has {len(insts)}")
    return ops


# -- the table ------------------------------------------------------------------------

def frame_len(b: int) -> int:
    """Byte length of the mode-0 frame starting with byte b (opcode table: modes/mode0.py)."""
    if (b & 0b11111011) == 0b11110000:
        return 8                                     # Start
    if b < 0x80:
        return 1                                     # Single0
    if (b & 0xE0) == 0x80:
        return 2                                     # Single1
    if (b & 0xE0) == 0xA0:
        return 3                                     # Single2
    if (b & 0xF0) == 0xC0:
        return 2                                     # Multiple0
    top6 = b & 0xFC
    if top6 in (0xD4, 0xDC):
        return 4                                     # Multiple2, DC
    if top6 == 0xD0:
        return 3                                     # Multiple1
    if top6 == 0xD8:
        return 2                                     # Repeat1
    return 1                                         # Repeat0, Skip, Sync


END_TOKEN = 0xDC                     # end-of-stream token: opcode, then an 18-bit count (trace_unit::END_TOKEN_*)
SKIP = 0xFE


def produces_command(b: int) -> bool:
    """False for the filler frames the decoder drops (Skip, DC)."""
    return b != 0xFE and (b & 0xFC) != 0xDC


def frame_type(b: int) -> str:
    if (b & 0b11111011) == 0b11110000:
        return "Start"
    if b < 0x80:
        return "Single0"
    if (b & 0xE0) == 0x80:
        return "Single1"
    if (b & 0xE0) == 0xA0:
        return "Single2"
    if (b & 0xF0) == 0xC0:
        return "Multiple0"
    if (b & 0xFC) == 0xD0:
        return "Multiple1"
    if (b & 0xFC) == 0xD4:
        return "Multiple2"
    if (b & 0xFC) == 0xD8:
        return "Repeat1"
    if (b & 0xF0) == 0xE0:
        return "Repeat0"
    return "Other"


def split_packets(words: list[int]) -> list[dict]:
    """Packets in raw buffer order: {order, pkt_type, row, col, bytes}. Same stride walk as the
    decoder's de-interleave (a header word, then seven payload words)."""
    from trace_decoder.packet import PACKET_WORDS, parse_packet_header, words_to_bytes
    out, i = [], 0
    while i + PACKET_WORDS <= len(words):
        h = parse_packet_header(words[i])
        if h is None:
            i += 1
            continue
        out.append({"order": len(out), "pkt_type": h.pkt_type, "row": h.row, "col": h.col,
                    "bytes": words_to_bytes(words[i + 1:i + PACKET_WORDS])})
        i += PACKET_WORDS
    return out


def completing_frames(packets: list[dict], pkt_type: int = PT_CORE) -> dict[int, tuple[int, str]]:
    """order -> (cycle, frame type) of the frame holding the last byte of each packet of the
    given source type, in that tile's own timeline.

    Walks the tile's frames in the decoder's timer convention: a Start sets the timer, an event
    frame advances it by 1 + its cycle delta. The cycle delta is read from the frame bytes.

    The end token (0xDC + an 18-bit count: cycles from the last frame to the stop, minus one) moves
    the timer to the stop, and the Skip padding after it keeps that date: a packet closed by the
    end token was sent at the stop, not when its last frame happened. A trace unit's first packet
    holding only its Start frame is such a packet (the unit sends a packet only when 28 bytes are
    filled, or at the stop); dated by its Start it would look like traffic at trace start."""
    from trace_decoder.modes import mode0
    done: dict[int, tuple[int, str]] = {}
    stream, owner = [], []
    for p in packets:
        if p["pkt_type"] == pkt_type:
            stream += p["bytes"]
            owner += [p["order"]] * len(p["bytes"])
    cmds = list(mode0.decode(stream))
    pos, timer, ci, ended = 0, 0, 0, False
    while pos < len(stream):
        b = stream[pos]
        n = frame_len(b)
        if produces_command(b):
            if ci >= len(cmds):
                break
            cmd = cmds[ci]
            ci += 1
            if hasattr(cmd, "timer_value"):
                timer = cmd.timer_value
            elif hasattr(cmd, "event_bits"):
                timer += 1 + cmd.cycles
            elif hasattr(cmd, "count"):
                timer += cmd.count
        elif (b & 0xFC) == END_TOKEN and pos + 3 < len(stream):
            timer += (((stream[pos + 1] & 0x3) << 16) | (stream[pos + 2] << 8) | stream[pos + 3]) + 1
            ended = True
        end = pos + n - 1
        for k in range(pos, min(end, len(stream) - 1) + 1):
            if k + 1 == len(stream) or owner[k + 1] != owner[k]:
                # the frame holding a packet's last byte; '*' marks one that runs into the next packet
                kind = "End" if ended and b in (SKIP, END_TOKEN) or (b & 0xFC) == END_TOKEN else frame_type(b)
                done[owner[k]] = (timer, kind + ("*" if end > k else ""))
        pos += n
    # Skip bytes that only align the end token are written at the stop as well: a packet they complete
    # takes the end token's date.
    for k in range(len(stream) - 1):
        if stream[k] == SKIP and owner[k + 1] != owner[k]:
            j = k + 1
            while j < len(stream) and stream[j] == SKIP:
                j += 1
            if j < len(stream) and (stream[j] & 0xFC) == END_TOKEN and owner[j] in done:
                done[owner[k]] = done[owner[j]]
    return done


def arrival_slots(starved: list[int]) -> list[int]:
    """Start cycles of the 8-cycle non-starved slots of a level (the cycles the level is asserted).
    A run of 8m non-starved cycles between two asserted runs is m back-to-back packets."""
    cycles = sorted(set(starved))
    slots: list[int] = []
    for a, b in zip(cycles, cycles[1:]):
        gap = b - a - 1
        if gap >= PACKET_CYCLES:
            slots += [a + 1 + PACKET_CYCLES * j for j in range(gap // PACKET_CYCLES)]
    return slots


def packet_rows(packets: list[dict], starved: list[int]) -> list[dict]:
    """One row per core packet (see the module doc for the columns). Packets with no slot (they
    arrived beyond the traced window) get no arrival and are listed as unmatched."""
    slots = arrival_slots(starved)
    comp = completing_frames(packets)
    core = [p for p in packets if p["pkt_type"] == PT_CORE]
    rows, prev_c = [], None
    for p in core:
        o = p["order"]
        c, ftype = comp.get(o, (None, "?"))
        row = {"order": o, "completion": c, "ftype": ftype, "arrival": None, "delay": None,
               "idle": None if prev_c is None or c is None else c - prev_c - PACKET_CYCLES,
               "since_any": None, "prev_src": None}
        if o < len(slots):
            row["arrival"] = slots[o]
            if c is not None:
                row["delay"] = slots[o] - c
            if o > 0:
                row["since_any"] = slots[o] - (slots[o - 1] + PACKET_CYCLES)
                row["prev_src"] = "core" if packets[o - 1]["pkt_type"] == PT_CORE else "shim"
        row["phase"] = None if row["idle"] is None else phase_of(row["idle"])
        row["contaminated"] = row["prev_src"] == "shim" and row["since_any"] is not None \
            and row["since_any"] < GATE_CYCLES
        prev_c = c if c is not None else prev_c
        rows.append(row)
    return rows


def phase_of(idle: int) -> str:
    """dense: the back-to-back packets (about 48 idle); near: below the gate but spaced out by the
    plan (100, 120); gated: at or beyond the 128-cycle gate."""
    if idle < DENSE_MAX_IDLE:
        return "dense"
    return "near" if idle < GATE_CYCLES else "gated"


def _num(v, w) -> str:
    return f"{'-' if v is None else v:>{w}}"


def format_rows(rows: list[dict]) -> str:
    lines = [f"{'pkt':>3} {'phase':<6} {'idle':>5} {'completion':>10} {'type':<8} {'arrival':>9} {'delay':>6} "
             f"{'since_any':>9} {'prev':<5} note"]
    for r in rows:
        note = "CONTAMINATED" if r["contaminated"] else ("bad frame type" if r["ftype"] not in ("Single0", "Single1")
                                                         else ("no arrival" if r["arrival"] is None else ""))
        lines.append(f"{r['order']:>3} {r['phase'] or '-':<6} {_num(r['idle'], 5)} {_num(r['completion'], 10)} "
                     f"{r['ftype']:<8} {_num(r['arrival'], 9)} {_num(r['delay'], 6)} "
                     f"{_num(r['since_any'], 9)} {r['prev_src'] or '-':<5} {note}")
    return "\n".join(lines)


BINS = ((64, 110), (110, GATE_CYCLES), (GATE_CYCLES, 143), (143, 200), (200, 10 ** 9))


def idle_bin(idle: int) -> tuple[int, int]:
    return next(b for b in BINS if b[0] <= idle < b[1])


def summarize(rows: list[dict]) -> str:
    """Delay per idle bin over the clean rows, against the median of the dense rows."""
    clean = [r for r in rows if r["delay"] is not None and r["idle"] is not None and not r["contaminated"]
             and r["ftype"] in ("Single0", "Single1")]
    dense = [r["delay"] for r in clean if r["phase"] == "dense"]
    if not dense:
        return "no clean dense rows: no baseline"
    base = median(dense)
    lines = [f"dense baseline: median delay {base} over {len(dense)} packets (hist {_hist(dense)})"]
    bins: dict[tuple[int, int], list[int]] = {}
    for r in clean:
        if r["phase"] != "dense":
            bins.setdefault(idle_bin(r["idle"]), []).append(r["delay"])
    for lo, hi in BINS:
        if (lo, hi) in bins:
            d = bins[(lo, hi)]
            name = f"{lo}..{hi - 1}" if hi < 10 ** 9 else f"{lo}+"
            lines.append(f"idle {name:>8}: n={len(d)} median delay {median(d)} = dense {median(d) - base:+.1f}  hist {_hist(d)}")
    lines.append(f"({len(rows) - len(clean)} rows left out: head packet, contaminated, unmatched or odd frame type)")
    return "\n".join(lines)


def _hist(xs) -> dict:
    h: dict = {}
    for x in xs:
        h[x] = h.get(x, 0) + 1
    return dict(sorted(h.items()))


def starved_cycles(events) -> list[int]:
    return [e.ts for e in events if e.pkt_type == PT_SHIM and e.name == SHIM_EVENT]


def load_run(run_dir: Path) -> tuple[list[dict], list[int]]:
    raw = (run_dir / "trace_raw.bin").read_bytes()
    words = list(struct.unpack(f"<{len(raw) // 4}I", raw))
    return split_packets(words), starved_cycles(dl.load_level_events(run_dir))


# -- build / run / table commands ---------------------------------------------------

def case_dir(out: Path) -> Path:
    return out / CASE


def cmd_build(args) -> int:
    out = Path(args.out_dir).resolve()
    d = case_dir(out)
    d.mkdir(parents=True, exist_ok=True)
    plan = default_plan()
    lp.assemble(core_asm(plan), d / "core.o", "k_core")
    (d / "base.mlir").write_text(design_mlir())
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time", *inject_args(),
           "--trace-config-out", "trace_config.json", "--config-test-name", CASE,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"], cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    ops = check_sequence_is_plain_writes((d / "insts.bin").read_bytes())
    print("sequence ops (opcode: count):", ops)
    (d / "plan.json").write_text(json.dumps([{"kind": p.kind, "idle": p.idle} for p in plan]) + "\n")
    print("built", CASE)
    return 0


def cmd_run(args) -> int:
    out = Path(args.out_dir).resolve()
    if not lp.run_case(out, CASE, args.tag, args.backend, output=False):
        print(f"{CASE}: run did not complete; stop")
        return 1
    return 0


def cmd_table(args) -> int:
    out = Path(args.out_dir).resolve()
    for run in args.runs:
        run_dir = case_dir(out) / run
        print(f"== {run}")
        if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
            print("no trace_raw.bin / events.json")
            continue
        packets, starved = load_run(run_dir)
        rows = packet_rows(packets, starved)
        n_core = len(rows)
        print(f"{len(packets)} packets in the buffer, {n_core} from the core, "
              f"{len(arrival_slots(starved))} arrival slots in the shim level")
        print(format_rows(rows))
        print(summarize(rows))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="assemble the core, inject trace, compile")
    b.add_argument("out_dir")
    r = sub.add_parser("run", help="run the built design to TAG.BACKEND/")
    r.add_argument("out_dir")
    r.add_argument("tag")
    r.add_argument("--backend", choices=("hw", "emu"), required=True)
    t = sub.add_parser("table", help="per core packet: phase, idle, completion, arrival, delay")
    t.add_argument("out_dir")
    t.add_argument("runs", nargs="+", help="run directory names, e.g. h1.hw e1.emu")
    args = ap.parse_args(argv)
    return {"build": cmd_build, "run": cmd_run, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
