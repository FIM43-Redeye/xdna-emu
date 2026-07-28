#!/usr/bin/env python3
"""EMU check that closes Experiment A: run the SAME a2_stride xclbins on the
emulator and compare the memtile MM2S span to the HW baseline (~251/258).

HW is stream-bound at 1 word/cycle (finding 2026-07-15-memtile-bank-access-width).
The emulator's granule cap in `granule_capped_words` (BankLayout::MemTile) is only
a fidelity error if EMU is NOT also stream-bound. Decisive test:
  EMU span ~= HW span (~251/258, flat over stride) -> EMU already stream-bound,
    cap harmless, gap closes "no fidelity error".
  EMU strided << contiguous, or EMU exceeds 1 word/cycle -> that IS the gap;
    lift the memtile granule cap.

Rep 1 in each build dir is the existing HW capture. This writes EMU into rep 2
(so nothing is clobbered) and also re-decodes HW rep 1 with the identical
parse-trace command as a decode-parity check -- if the re-decode reproduces the
finding's 251/258, the EMU decode is apples-to-apples.
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from memtile_bankwidth_measure import measure  # noqa: E402

XDNA_EMU = Path("/home/triple/npu-work/xdna-emu")
RUNNER = XDNA_EMU / "bridge-runner/build/bridge-trace-runner"
ROOT = XDNA_EMU / "build/experiments/memtile-bankwidth"
STRIDES = (4, 16, 64, 256)  # the strides with HW data in the finding


def run(cmd, env=None):
    print("+ " + " ".join(str(c) for c in cmd), file=sys.stderr)
    subprocess.run(cmd, env=env, check=True)


def emu_run(build_dir: Path, rep: int):
    env = os.environ.copy()
    env["XDNA_EMU"] = "1"
    env["XDNA_EMU_RUNTIME"] = "debug"
    trace_out = build_dir / f"trace_r{rep}.bin"
    run([str(RUNNER),
         "--xclbin", str(build_dir / "aie.xclbin"),
         "--instr", str(build_dir / "insts.bin"),
         "--trace-out", str(trace_out),
         "--trace-size", "16384",
         "--output", str(build_dir / f"out_r{rep}.bin"),
         "-v"], env=env)
    assert trace_out.exists() and trace_out.stat().st_size > 0, \
        f"empty EMU trace_out for {build_dir.name} rep {rep} -- EMU trace broken"


def decode(build_dir: Path, rep: int, out_name: str):
    run([sys.executable, str(XDNA_EMU / "tools/parse-trace.py"),
         "--trace-bin", str(build_dir / f"trace_r{rep}.bin"),
         "--xclbin-mlir", str(build_dir / "aie_traced.mlir"),
         "--out-perfetto", str(build_dir / out_name),
         "--trace-mode", "event_time"])


if __name__ == "__main__":
    rows = []
    for s in STRIDES:
        bd = ROOT / f"build_memtile_bankwidth_a2_stride_{s}"
        if not bd.is_dir():
            print(f"{s:>6}  MISSING build dir", file=sys.stderr)
            continue
        # EMU capture into rep 2, then decode both EMU (r2) and a HW re-decode
        # (r3, from the existing HW trace_r1.bin) with the SAME parse-trace
        # command so the comparison is apples-to-apples.
        emu_run(bd, 2)
        decode(bd, 2, "perfetto_r2.json")
        run(["cp", str(bd / "trace_r1.bin"), str(bd / "trace_r3.bin")])
        decode(bd, 3, "perfetto_r3.json")

        orig = measure(bd, 1)   # original HW perfetto_r1.json (the finding's decode)
        hw = measure(bd, 3)     # HW re-decoded with my command (parity check vs orig)
        emu = measure(bd, 2)
        rows.append((s, orig, hw, emu))

    print(f"\n{'stride':>6} {'HW_orig':>8} {'HW_redec':>9} {'EMU':>8}   "
          f"(n_brk orig/redec/emu)")
    print("-" * 56)
    for s, orig, hw, emu in rows:
        print(f"{s:>6} {str(orig['median']):>8} {str(hw['median']):>9} "
              f"{str(emu['median']):>8}   "
              f"({orig['n_bracketed']}/{hw['n_bracketed']}/{emu['n_bracketed']})")
