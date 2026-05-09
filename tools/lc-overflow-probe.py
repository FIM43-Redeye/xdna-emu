#!/usr/bin/env python3
"""LC bit-28 overflow probe -- run the lc_overflow_probe fixture across trip
counts straddling 2^28 and dump every LC frame the trace controller emits.

Phase 0 (2026-04-30) found bit-28 always 0 for trip counts 1 .. 16384. The
LC frame's count field is 28 bits wide, so any trip count >= 2^28 cannot
fit. Three plausible behaviors:

  (a) Hardware saturates at 2^28-1 and sets bit-28=1 as a "saturated"
      indicator.
  (b) Hardware wraps: count = N mod 2^28, bit-28 stays 0. The compiler
      would need to emit a multi-ZOL split for correctness above 2^28.
  (c) Compiler refuses to lower a single ZOL above 2^28 -- we'd see
      multiple LC frames per kernel call (the loop split).

We probe a battery of N values:
  - small / medium (sanity baseline; should match Phase 0 exactly)
  - just under 2^28
  - exactly 2^28 / just past
  - around 2^29 (well past, to disambiguate wrap vs saturate)

Outputs a summary table to stdout and a JSON dump per N to
``$REPO/build/experiments/lc_overflow/<timestamp>/`` for the findings doc.

Usage:
    # Run after building the fixture:
    #   tools/mode2_capture_fixtures/build_fixture.sh --chess --mode2 \
    #     tools/mode2_capture_fixtures/lc_overflow_probe
    python3 tools/lc-overflow-probe.py [--xclbin PATH] [--insts PATH]
                                       [--trip-counts N,N,...]
                                       [--out-dir DIR]
"""

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

REPO = Path(__file__).resolve().parents[1]

# Default trip counts -- baseline + 28-bit-boundary sweep.
DEFAULT_TRIP_COUNTS: tuple[int, ...] = (
    4,
    64,
    1024,
    65_536,                  # 2^16
    1 << 24,                 # 2^24, well below boundary
    (1 << 28) - 1,           # 268_435_455, max 28-bit value
    1 << 28,                 # 268_435_456, first overflow
    (1 << 28) + 1,
    (1 << 28) + 5,
    (1 << 29) - 1,
    1 << 29,
    (1 << 29) + 5,
)


def passes_for(n: int) -> int:
    """Pick a wrapper-pass count that keeps total runtime under the XRT
    command timeout (~3s on Phoenix). 4 passes at low N gives redundancy;
    1 pass at high N keeps the kernel under TDR threshold."""
    if n >= (1 << 26):
        return 1
    if n >= (1 << 22):
        return 2
    return 4


def write_input_bin(path: Path, n: int, passes: int) -> None:
    """Write a 64-i32 input buffer with in[0]=n, in[1]=passes, rest zero."""
    buf = bytearray(64 * 4)
    struct.pack_into("<i", buf, 0, n & 0xFFFFFFFF)
    struct.pack_into("<i", buf, 4, passes & 0xFFFFFFFF)
    path.write_bytes(buf)


def run_capture(
    *,
    runner: Path,
    xclbin: Path,
    insts: Path,
    input_bin: Path,
    output_bin: Path,
    trace_bin: Path,
    timeout_sec: float,
) -> dict:
    """Invoke bridge-trace-runner once. Returns {"ok": bool, "stderr": str}."""
    cmd = [
        str(runner),
        "--xclbin", str(xclbin),
        "--instr", str(insts),
        "--input", str(input_bin),
        "--output", str(output_bin),
        "--trace-out", str(trace_bin),
    ]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "stderr": f"timeout after {timeout_sec:.1f}s: {exc}",
            "elapsed_sec": time.monotonic() - t0,
        }
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stderr": proc.stderr,
        "stdout": proc.stdout,
        "elapsed_sec": time.monotonic() - t0,
    }


def decode_lc_frames(trace_bin: Path):
    """Decode the trace bin and return the list of LC frames per tile.

    Returns list of dicts: ``{"tile": (pkt_type, row, col), "frames":
    [(flag, count), ...]}``.
    """
    sys.path.insert(0, str(REPO / "tools"))
    from trace_decoder import decode_words, TraceMode  # type: ignore
    from trace_decoder.modes.mode2 import LoopCountCmd  # type: ignore

    raw = trace_bin.read_bytes()
    if not raw:
        return []
    if len(raw) % 4 != 0:
        raw = raw + b"\x00" * (4 - len(raw) % 4)
    words = list(struct.unpack(f"<{len(raw) // 4}I", raw))

    by_tile = decode_words(words, mode=TraceMode.INST_EXEC)
    out = []
    for tile_key, cmds in by_tile.items():
        lc = [(c.flag, c.count) for c in cmds if isinstance(c, LoopCountCmd)]
        out.append({"tile": tile_key, "frames": lc, "total_cmds": len(cmds)})
    return out


def summarize(per_run: list[dict]) -> str:
    """Build a human-readable summary table."""
    lines = []
    header = (
        f"{'N':>14} {'N_hex':>12} {'lc_frames':>10} {'flag_set':>9} "
        f"{'distinct_count':>15} {'first_count':>14} {'first_count_hex':>18} {'status':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in per_run:
        n = r["n"]
        if not r.get("ok"):
            lines.append(
                f"{n:>14} {n:>#12x} {'-':>10} {'-':>9} {'-':>15} "
                f"{'-':>14} {'-':>18} {'FAILED':>10}"
            )
            continue
        all_frames = []
        for tile in r["tiles"]:
            all_frames.extend(tile["frames"])
        lc_frames = len(all_frames)
        flag_set = sum(1 for (f, _) in all_frames if f != 0)
        distinct_counts = sorted({c for (_, c) in all_frames})
        first_count = all_frames[0][1] if all_frames else 0
        status = "OK" if all_frames else "EMPTY"
        lines.append(
            f"{n:>14} {n:>#12x} {lc_frames:>10} {flag_set:>9} "
            f"{len(distinct_counts):>15} {first_count:>14} "
            f"{first_count:>#18x} {status:>10}"
        )
    return "\n".join(lines)


def parse_trip_counts(arg: str) -> list[int]:
    out = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("0x") or tok.startswith("0X"):
            out.append(int(tok, 16))
        else:
            out.append(int(tok))
    return out


def main(argv: Sequence[str]) -> int:
    fixture_default = REPO / "tools/mode2_capture_fixtures/lc_overflow_probe"
    runner_default = REPO / "bridge-runner/build/bridge-trace-runner"
    out_default = REPO / "build/experiments/lc_overflow"

    p = argparse.ArgumentParser(
        description="Probe LC bit-28 flag at trip counts straddling 2^28."
    )
    p.add_argument(
        "--fixture-dir",
        type=Path,
        default=fixture_default,
        help="lc_overflow_probe fixture root (must already be built).",
    )
    p.add_argument(
        "--xclbin",
        type=Path,
        help="traced xclbin (default: <fixture-dir>/build/traced/aie.xclbin).",
    )
    p.add_argument(
        "--insts",
        type=Path,
        help="traced insts.bin (default: <fixture-dir>/build/traced/insts.bin).",
    )
    p.add_argument(
        "--runner",
        type=Path,
        default=runner_default,
        help="bridge-trace-runner binary path.",
    )
    p.add_argument(
        "--trip-counts",
        type=parse_trip_counts,
        default=list(DEFAULT_TRIP_COUNTS),
        help="comma-separated list of trip counts (decimal or 0x hex).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        help="results dir (default: build/experiments/lc_overflow/<ts>/).",
    )
    p.add_argument(
        "--timeout-sec",
        type=float,
        default=120.0,
        help="per-run runner timeout in seconds (default 120).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan, don't execute.",
    )
    args = p.parse_args(argv)

    xclbin = args.xclbin or args.fixture_dir / "build/traced/aie.xclbin"
    insts = args.insts or args.fixture_dir / "build/traced/insts.bin"

    for path, what in [(xclbin, "traced xclbin"), (insts, "traced insts")]:
        if not path.exists():
            print(f"error: {what} not found at {path}", file=sys.stderr)
            print(
                f"hint: build with `tools/mode2_capture_fixtures/build_fixture.sh "
                f"--chess --mode2 {args.fixture_dir}`",
                file=sys.stderr,
            )
            return 64

    if not args.runner.exists():
        print(f"error: bridge-trace-runner not found at {args.runner}", file=sys.stderr)
        return 64

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (out_default / ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"# lc-overflow-probe results dir: {out_dir}")
    print(f"# trip counts: {args.trip_counts}")

    if args.dry_run:
        return 0

    per_run: list[dict] = []
    for n in args.trip_counts:
        run_dir = out_dir / f"n_{n}"
        run_dir.mkdir(exist_ok=True)
        input_bin = run_dir / "input.bin"
        output_bin = run_dir / "output.bin"
        trace_bin = run_dir / "trace.bin"
        passes = passes_for(n)
        write_input_bin(input_bin, n, passes)

        print(f"# N={n} ({n:#x}) passes={passes} -> {run_dir.name}/", flush=True)
        cap = run_capture(
            runner=args.runner,
            xclbin=xclbin,
            insts=insts,
            input_bin=input_bin,
            output_bin=output_bin,
            trace_bin=trace_bin,
            timeout_sec=args.timeout_sec,
        )

        rec: dict = {"n": n, "passes": passes, **cap}
        if cap["ok"] and trace_bin.exists() and trace_bin.stat().st_size > 0:
            tiles = decode_lc_frames(trace_bin)
            rec["tiles"] = tiles
            rec["trace_bytes"] = trace_bin.stat().st_size
        else:
            rec["tiles"] = []
            rec["trace_bytes"] = trace_bin.stat().st_size if trace_bin.exists() else 0

        # Persist per-N record (truncate stderr if huge).
        rec_for_disk = dict(rec)
        if isinstance(rec_for_disk.get("stderr"), str) and len(rec_for_disk["stderr"]) > 4096:
            rec_for_disk["stderr"] = rec_for_disk["stderr"][:4096] + "...[truncated]"
        if isinstance(rec_for_disk.get("stdout"), str) and len(rec_for_disk["stdout"]) > 4096:
            rec_for_disk["stdout"] = rec_for_disk["stdout"][:4096] + "...[truncated]"
        # Convert tile keys to strings for JSON.
        rec_for_disk["tiles"] = [
            {"tile": list(t["tile"]), "frames": [list(f) for f in t["frames"]],
             "total_cmds": t["total_cmds"]}
            for t in rec.get("tiles", [])
        ]
        (run_dir / "record.json").write_text(json.dumps(rec_for_disk, indent=2))
        per_run.append(rec)

        if not cap["ok"]:
            tail = (cap.get("stderr") or "").splitlines()[-3:]
            print(f"  FAILED ({cap.get('returncode')}): {' / '.join(tail)}",
                  flush=True)
            continue

        all_frames = [f for t in rec["tiles"] for f in t["frames"]]
        if not all_frames:
            print(f"  ok in {cap['elapsed_sec']:.2f}s, "
                  f"trace={rec['trace_bytes']}B, NO LC FRAMES", flush=True)
            continue
        first = all_frames[0]
        flags = sum(1 for (f, _) in all_frames if f != 0)
        print(
            f"  ok in {cap['elapsed_sec']:.2f}s, trace={rec['trace_bytes']}B, "
            f"lc={len(all_frames)}, flag_set={flags}, "
            f"first=(flag={first[0]},count={first[1]} / {first[1]:#x})",
            flush=True,
        )

    print()
    print(summarize(per_run))

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summarize(per_run) + "\n")
    print(f"\n# summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
