#!/usr/bin/env python3
"""Run every already-built r4d3 configuration under the emulator (3 reps each)
and print a summary of stale counts. Usage: XDNA_EMU_DIR=<worktree> r4d3_emu_sweep.py [tag]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import r4d3_capture as c  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "emu"
names = sorted(p.name for p in c.ROOT.iterdir()
               if p.is_dir() and (p / "aie.xclbin").exists() and "_d1200_d220" in p.name
               or p.is_dir() and p.name.startswith("core_") and (p / "aie.xclbin").exists())
for n in names:
    rows = c.run(c.ROOT / n, 2, True, tag)
    print("SUMMARY", n, sorted({r.get("stale") for r in rows}))
