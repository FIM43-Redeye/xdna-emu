#!/usr/bin/env python3
"""Count per-cycle activity of named VCD signals in an aiesimulator dump.

Usage: vcd_probe.py foo.vcd <name-substring> [<name-substring> ...]

For each signal whose full name contains ALL given substrings, prints how many
timesteps it was asserted (value '1') and its number of value changes -- enough
to see whether an event/conflict/stall signal actually fires during the run.
ponytail: linear scan over the VCD once; fine for a ~30MB dump.
"""
import sys, re

def main():
    vcd, subs = sys.argv[1], sys.argv[2:]
    id_name = {}          # vcd id -> full name
    want_ids = {}         # vcd id -> name (matched)
    # Pass 1: header ($var). Format: $var wire N <id> <name> $end
    with open(vcd, 'r', errors='replace') as f:
        for line in f:
            if line.startswith('$var'):
                parts = line.split()
                # parts: $var wire N id name... $end ; name may be idx 4 (single token here)
                vid, name = parts[3], parts[4]
                id_name[vid] = name
                if all(s in name for s in subs):
                    want_ids[vid] = name
            elif line.startswith('$dumpvars') or line.startswith('#'):
                break
    if not want_ids:
        print("no signals matched", subs); return
    high = {v: 0 for v in want_ids}   # timesteps asserted (scalar '1')
    changes = {v: 0 for v in want_ids}
    last = {v: None for v in want_ids}
    cur_high = {v: False for v in want_ids}
    t = 0
    # Pass 2: value-change section
    with open(vcd, 'r', errors='replace') as f:
        indump = False
        for line in f:
            if line and line[0] == '#':
                # advance time: tally currently-high signals for the elapsed step
                for v in want_ids:
                    if cur_high[v]:
                        high[v] += 1
                t += 1
                continue
            c = line[0] if line else ''
            if c in '01':                       # scalar value change: <val><id>
                val = c; vid = line[1:].strip()
                if vid in want_ids:
                    if last[vid] != val:
                        changes[vid] += 1
                    last[vid] = val
                    cur_high[vid] = (val == '1')
            elif c in 'bB':                      # vector: b<bits> <id>
                m = line.split()
                if len(m) == 2 and m[1] in want_ids:
                    if last[m[1]] != m[0]:
                        changes[m[1]] += 1
                    last[m[1]] = m[0]
    print(f"{'signal':70s} {'asserted_steps':>14s} {'changes':>8s}")
    for v, name in sorted(want_ids.items(), key=lambda kv: kv[1]):
        print(f"{name:70s} {high[v]:>14d} {changes[v]:>8d}")
    print(f"\n(total timesteps seen: {t})")

if __name__ == '__main__':
    main()
