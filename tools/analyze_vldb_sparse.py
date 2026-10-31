#!/usr/bin/env python3
"""Reference model of the sparse loads read off the `vec_sparseload_*` captures, and its check.

    python3 tools/analyze_vldb_sparse.py check            # every golden against the model

The model (derived from NPU1 silicon, see docs/fidelity-gaps/vector-compute.md):

  * The unit has a 320-byte ring (ten 32-byte lines) written one line at a time, a hidden
    consume pointer DP (an absolute address) and the sticky flags `srSparse_of` / `srCompr_uf`.
  * `reset [p]`: remember the ring write position as the read origin, DP = p, write the line at p
    into the ring, p += 32. `fill [p]`: write the line at p, p += 32. Neither touches the flags.
  * `pop [p]` and `peek [p]` read an 8-byte mask at the read position (ring offset DP - origin)
    and expand the data bytes that follow it into half a register (below). `pop` then adds
    8 + (ceil4(popcount(mask)) mod 64) to DP and writes the line at p into the ring (after the
    read), p += 32; `peek` changes neither DP nor the ring.
  * Expansion. Width 4 and 8: one mask bit per data byte; 16 groups of four mask bits; a group with
    one set bit takes one stream byte into slot 0 (bit 0 or 1) or slot 1 (bit 2 or 3), with two set
    bits two bytes into slots 0 and 1, with none, three or four set bits is zero. Group g owns
    output bytes 2g (slot 0) and 2g+1 (slot 1). Width 16: the unit is a 16-bit element with two
    mask bits (bit 2e selects its low byte, bit 2e+1 its high byte); 8 groups of four elements (one
    mask byte), the same slot rule on elements, slot s of group j at output bytes 4j+2s, 4j+2s+1,
    one stream byte per set mask bit. Stream bytes are consumed in mask-bit order, only by groups
    that are placed.
  * `srSparse_of` is set by a pop/peek whose mask has a group with more than two selected units;
    `srCompr_uf` by one with -32 < rem <= 64, rem = 32 * (lines written since reset) - (read
    position). `clr` clears both.

Reads past what has been written return what the ring holds, which the model reproduces.
"""

import argparse
import os
import struct
import sys

import gen_vldb_sparse as g
from capture_kernel_render import load_golden

RING_BYTES = 32 * g.RING_LINES
LINE = 32
DATA_BASE = 0x74000     # where the kernel's input buffer sits in the core's data memory

# Mismatches between the model and silicon that are understood only as far as listed here (stream,
# script): reads that cross a ring line boundary soon after a fetch into the same slot see the
# marker byte or line the model has already replaced. Four masks, eight data bytes and four DP
# values in the `proto` captures (none in `ring`, none that involve a 16-bit-only difference).
# These are open questions in the docs, kept exact so that a new difference cannot hide in them.
KNOWN_RESIDUAL = {
    ("onehot_63", "protocol"), ("onehot_63", "carry"),
    ("onehot_64", "protocol"), ("onehot_64", "many"),
    ("dense40", "unprimed"),
}


def nibbles(mask):
    out = []
    for x in mask:
        out += [x & 15, x >> 4]
    return out


def slot_for_group(pattern):
    """Slot index of each selected unit of a group pattern, or None when the group is zero."""
    sel = [i for i in range(4) if pattern >> i & 1]
    if len(sel) == 1:
        return [1 if sel[0] >= 2 else 0]
    if len(sel) == 2:
        return [0, 1]
    return None


def groups(mask, width):
    """[(group index, [(slot, [byte of the unit]) ...] or None)] in data order."""
    out = []
    if width < 16:
        for gi, nib in enumerate(nibbles(mask)):
            slots = slot_for_group(nib)
            out.append((gi, None if slots is None else [(s, [0]) for s in slots]))
        return out
    for j, byte in enumerate(mask):
        pattern = sum(((byte >> (2 * e)) & 3 != 0) << e for e in range(4))
        slots = slot_for_group(pattern)
        if slots is None:
            out.append((j, None))
            continue
        elements = [e for e in range(4) if pattern >> e & 1]
        out.append((j, [(s, [b for b in (0, 1) if byte >> (2 * e + b) & 1]) for s, e in zip(slots, elements)]))
    return out


def overflows(mask, width):
    if width < 16:
        return any(bin(n).count("1") > 2 for n in nibbles(mask))
    return any(
        bin(sum(((byte >> (2 * e)) & 3 != 0) << e for e in range(4))).count("1") > 2 for byte in mask)


def output_offset(width, group, slot, byte):
    return 2 * group + slot if width < 16 else 4 * group + 2 * slot + byte


class Unit:
    """The sparse load unit; state persists across scripts and chunks like the hardware's."""

    def __init__(self, width):
        self.width = width
        self.ring = bytearray(RING_BYTES)
        self.write = 0
        self.origin = 0
        self.lines = 0
        self.dp_base = 0
        self.dp = 0
        self.sparse_of = 0
        self.compr_uf = 0

    def fetch(self, line):
        self.ring[self.write:self.write + LINE] = line
        self.write = (self.write + LINE) % RING_BYTES
        self.lines += 1

    def at(self, offset):
        return self.ring[(self.origin + offset) % RING_BYTES]

    def read(self, advance_dp):
        """(mask, 32 data bytes) at the read position; flags updated; DP advanced for a pop."""
        offset = self.dp - self.dp_base
        mask = bytes(self.at(offset + i) for i in range(8))
        data = [0] * 32
        cursor = offset + 8
        for group, slots in groups(mask, self.width):
            if slots is None:
                continue
            for slot, bytes_ in slots:
                for b in bytes_:
                    data[output_offset(self.width, group, slot, b)] = self.at(cursor)
                    cursor += 1
        if overflows(mask, self.width):
            self.sparse_of = 1
        if -32 < LINE * self.lines - offset <= 64:
            self.compr_uf = 1
        if advance_dp:
            self.dp += 8 + ((sum(bin(x).count("1") for x in mask) + 3) // 4 * 4) % 64
        return mask, bytes(data)


def replay_chunk(unit, stream, scripts):
    """Run the prologue and every script on one stream; yield (script, step, step name, result).

    The result is (mask, data) for pop/peek steps, DP for `dp`, and None otherwise.
    """
    pointer = 0
    unit.origin = unit.write
    unit.dp_base = unit.dp = 0
    unit.lines = 0
    for _ in range(g.RING_LINES):
        unit.fetch(stream[pointer:pointer + LINE])
        pointer += LINE
    for si, (_, script) in enumerate(scripts):
        pointer = 0
        for ki, step in enumerate(g.steps_of(script)):
            result = None
            if step == "clr":
                unit.sparse_of = unit.compr_uf = 0
            elif step == "reset":
                unit.origin = unit.write
                unit.lines = 0
                unit.dp_base = unit.dp = pointer
                unit.fetch(stream[pointer:pointer + LINE])
                pointer += LINE
            elif step == "fill":
                unit.fetch(stream[pointer:pointer + LINE])
                pointer += LINE
            elif step.startswith("pop"):
                result = unit.read(True)
                unit.fetch(stream[pointer:pointer + LINE])
                pointer += LINE
            elif step.startswith("peek"):
                result = unit.read(False)
            elif step == "dp":
                # A peek through a copy of the pointer: it reads like a peek and fetches the line
                # at the pointer without keeping the advanced pointer.
                unit.read(False)
                result = unit.dp
                unit.fetch(stream[pointer:pointer + LINE])
            yield si, ki, step, result, (unit.sparse_of, unit.compr_uf)


def words_of(gold):
    return gold["silicon"]


def record(words, chunk, script_index, step_index, scripts):
    per_chunk = sum(len(g.steps_of(s)) for _, s in scripts) * g.REC
    base = chunk * per_chunk + (sum(len(g.steps_of(s)) for _, s in scripts[:script_index]) + step_index) * g.REC
    r = words[base:base + g.REC]
    pack = lambda xs: b"".join(struct.pack("<I", x) for x in xs)
    return {"dp": r[0], "of": r[1], "uf": r[2], "q": pack(r[8:12]), "wl": pack(r[16:24]), "wh": pack(r[24:32])}


def check_golden(path, group, width):
    """Compare the model with a golden. Returns a dict of counts and the list of mismatches."""
    g.set_group(group)
    gold = load_golden(path)
    words = [w & 0xFFFFFFFF for w in words_of(gold)]
    unit = Unit(width)
    bad = []
    counts = {"pops": 0, "dps": 0, "flags": 0}
    for ci, (name, stream) in enumerate(g.streams()):
        stream = bytes(stream)
        for si, ki, step, result, flags in replay_chunk(unit, stream, g.SCRIPTS):
            rec = record(words, ci, si, ki, g.SCRIPTS)
            counts["flags"] += 1
            if flags != (rec["of"], rec["uf"]):
                bad.append((name, g.SCRIPTS[si][0], ki, step, "flags", flags, (rec["of"], rec["uf"])))
            if step.startswith(("pop", "peek")):
                counts["pops"] += 1
                lo = step.endswith("lo")
                mask, data = result
                got_mask = rec["q"][:8] if lo else rec["q"][8:]
                got_data = rec["wl"] if lo else rec["wh"]
                if (mask, data) != (got_mask, got_data):
                    bad.append((name, g.SCRIPTS[si][0], ki, step, "data"))
            elif step == "dp":
                counts["dps"] += 1
                if result != rec["dp"] - DATA_BASE:
                    bad.append((name, g.SCRIPTS[si][0], ki, step, "dp", result, rec["dp"] - DATA_BASE))
    return counts, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["check"])
    ap.parse_args(argv)
    for group in g.GROUPS:
        for width in g.WIDTHS:
            path = g.golden_path(width, group)
            if not os.path.exists(path):
                print(f"{path}: no golden")
                continue
            counts, bad = check_golden(path, group, width)
            print(f"{g.kernel_name(width, group)}: {counts}, {len(bad)} differ {bad[:3]}")


if __name__ == "__main__":
    main()
