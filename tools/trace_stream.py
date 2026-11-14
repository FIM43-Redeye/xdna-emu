#!/usr/bin/env python3
"""Build the emulator's whole trace stream from the FFI's S2MM stream sidecar.

A shim S2MM writes host memory in whole bursts, so ``trace_raw.bin`` (the host's trace buffer,
written by the host program) misses the last partial burst of every channel whose trace BD never
completes, on silicon and in the emulator alike. The FFI records every word the shim S2MMs took off
their streams in ``trace_stream.json`` (crates/xdna-emu-ffi/src/stream_sidecar.rs). This tool lays
the trace buffer's words out as the buffer would be had every word been written,
``trace_stream.bin``, so the trace gate can score the whole stream and stay exact, while
``trace_raw.bin`` stays the silicon-faithful capture.

The trace buffer is the sidecar buffer of ``trace_raw.bin``'s size whose stream ``trace_raw.bin``
is consistent with. The check (written to ``trace_stream.check``): in each run of the sidecar the
host buffer holds a prefix of the run that ends on a burst boundary (or is the whole run, a BD that
completed), the rest of the run unwritten (or, in a buffer several dispatches wrote, an earlier
dispatch's bytes: each sidecar run names the pass, one per FFI run, that staged it last), and
nothing outside the runs.

usage: trace_stream.py DIR     (reads DIR/trace_raw.bin and DIR/trace_stream.json)
exit 0: OK, trace_stream.bin written; 1: check failed; 2: inputs missing.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

# The smallest shim burst: Burst_Length 0, 64 bytes. A run may name its own.
DEFAULT_BURST = 64


@dataclass
class Result:
    ok: bool
    message: str
    image: bytes | None = None


def _image(size: int, buffer: dict) -> bytearray:
    out = bytearray(size)
    for run in buffer["runs"]:
        b = bytes.fromhex(run["bytes"])
        out[run["offset"]:run["offset"] + len(b)] = b
    return out


def _check_buffer(raw: bytes, buffer: dict) -> str | None:
    """None when ``raw`` is consistent with ``buffer``'s stream, else why not.

    Per run: ``raw`` agrees with the run up to its first difference, which lies on a burst
    boundary (a partial burst is never written) unless the run is whole; past it ``raw`` holds
    zeros, or, in a buffer that several runs (dispatches) wrote, an earlier run's bytes."""
    rewritten = len({run.get("pass", 0) for run in buffer["runs"]}) > 1
    covered = bytearray(len(raw))
    for run in buffer["runs"]:
        o = run["offset"]
        b = bytes.fromhex(run["bytes"])
        burst = run.get("burst", DEFAULT_BURST)
        end = o + len(b)
        if end > len(raw):
            return f"run at {o:#x} runs past the buffer"
        covered[o:end] = b"\x01" * len(b)
        held = raw[o:end]
        m = next((i for i in range(0, len(b), 4) if held[i:i + 4] != b[i:i + 4]), len(b))
        if m == len(b):
            continue
        past = any(held[m:])
        if past and not rewritten:
            return f"run at {o:#x}: the buffer's bytes are not a prefix of the stream"
        if (o + m) % burst and not past:
            return (f"run at {o:#x}: the buffer holds {m} of {len(b)} bytes, "
                    f"not whole {burst}-byte bursts")
    stray = next((i for i, (v, c) in enumerate(zip(raw, covered)) if v and not c), None)
    if stray is not None:
        return f"byte at {stray:#x} is outside every stream run"
    return None


def build(raw: bytes, sidecar: dict) -> Result:
    candidates = [b for b in sidecar.get("buffers", []) if b["size"] == len(raw)]
    if not candidates:
        return Result(False, f"no sidecar buffer of the capture's size {len(raw)}")
    fits = []
    reasons = []
    for buf in candidates:
        why = _check_buffer(raw, buf)
        if why is None:
            fits.append(buf)
        else:
            reasons.append(f"buffer {buf['base']:#x}: {why}")
    if not fits:
        return Result(False, "; ".join(reasons))
    images = {bytes(_image(len(raw), b)) for b in fits}
    if len(images) > 1:
        return Result(False, f"{len(fits)} sidecar buffers fit the capture with different streams")
    buf = fits[0]
    staged = sum(len(r["bytes"]) // 2 for r in buf["runs"])
    held = sum(1 for v in raw if v)
    return Result(True, f"buffer {buf['base']:#x}: {len(buf['runs'])} run(s), {staged} bytes staged, "
                        f"{held} non-zero bytes in the capture", images.pop())


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    d = Path(argv[1])
    raw_path, side_path = d / "trace_raw.bin", d / "trace_stream.json"
    if not raw_path.exists() or not side_path.exists():
        return 2
    res = build(raw_path.read_bytes(), json.loads(side_path.read_text()))
    (d / "trace_stream.check").write_text(("OK " if res.ok else "FAIL ") + res.message + "\n")
    if res.ok:
        (d / "trace_stream.bin").write_bytes(res.image)
    print(("OK " if res.ok else "FAIL ") + res.message)
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
