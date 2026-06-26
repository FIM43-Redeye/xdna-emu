"""Canary cleanliness witness: certify a capture session clean from a sentinel
kernel's multi-run capture. A known within-domain core-lock span is cycle-exact
under a quiet host and flickers under host-load contamination (see
docs/trace/capture-load-sensitivity.md). The witness grounds that span: a
Segment (range 0) means the session was clean; a within_domain_nonexact Gap
means it was contaminated and any tally taken alongside it is void.

The verdict is the canary's own rule (range <= Q == 0), threshold-free. The
witness does NOT classify load vs genuine HW -- it only flags 'not clean'; a
human re-captures on a quiet host.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from inference.grounding import ground_edge, Segment
from inference.verifier import ANCHOR

# Sentinel: add_one_using_dma compute-tile core-lock span (col 1, row 2, core
# module pkt 0). Verified cycle-exact-under-clean / flicker-under-load.
SENTINEL_ACQ = "1|2|0|INSTR_LOCK_ACQUIRE_REQ"
SENTINEL_REL = "1|2|0|INSTR_LOCK_RELEASE_REQ"

# The sentinel's config dump -- absolute so the witness works from any CWD. The
# live capture needs it to configure the core-lock events (enumerate_configured_
# events yields INSTR_LOCK_ACQUIRE_REQ/RELEASE_REQ from this dump); dump_path=None
# would trace no events and the witness could never ground its span.
SENTINEL_DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "config_extract", "fixtures",
                             "add_one_using_dma.config.json")


@dataclass
class WitnessResult:
    clean: bool
    detail: str
    offset: Optional[int] = None
    reason: Optional[str] = None


def witness_clean(run_dirs: List[str], acq: str = SENTINEL_ACQ,
                  rel: str = SENTINEL_REL, anchor: str = ANCHOR) -> WitnessResult:
    """Certify a capture session clean iff the sentinel core-lock span grounds
    as a cycle-exact Segment across run_dirs."""
    g = ground_edge(run_dirs, rel, acq, anchor)
    if isinstance(g, Segment):
        return WitnessResult(True, f"core-lock cycle-exact (offset {g.offset})",
                             offset=g.offset)
    return WitnessResult(False, f"core-lock NONEXACT (gap reason {g.reason})",
                         reason=g.reason)


def _capture_sentinel_runs(out_root: str, test: str, compiler: str,
                           n_runs: int, dump_path: str = SENTINEL_DUMP) -> List[str]:
    """HW-gated: capture the sentinel kernel n_runs times via the inference
    capture path, return the resulting run dirs. Touches the NPU. The dump is
    required to configure the sentinel's core-lock events."""
    from inference.run_experiment import KernelConfig, run_experiment
    cfg = KernelConfig(test=test, compiler=compiler, dump_path=dump_path,
                       start_col=1, anchor_tile_abs="1|2|0",
                       anchor_event="PERF_CNT_2", traced_col=1, n_runs=n_runs,
                       out_root=out_root)
    run_experiment(cfg)
    return sorted(glob.glob(os.path.join(out_root, "capture_*", "run_*")))


def capture_and_witness(out_root: str, test: str = "add_one_using_dma",
                        compiler: str = "chess", n_runs: int = 20) -> WitnessResult:
    """Capture the sentinel on HW and certify the session clean/dirty."""
    run_dirs = _capture_sentinel_runs(out_root, test, compiler, n_runs)
    if not run_dirs:
        return WitnessResult(False, "no sentinel run dirs captured")
    return witness_clean(run_dirs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Canary capture-cleanliness witness.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs", nargs="+",
                     help="certify already-captured sentinel run dirs")
    src.add_argument("--capture-out",
                     help="HW-gated: capture the sentinel into this dir, then certify")
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--acq", default=SENTINEL_ACQ)
    ap.add_argument("--rel", default=SENTINEL_REL)
    ap.add_argument("--anchor", default=ANCHOR)
    args = ap.parse_args(argv)
    if args.capture_out:
        res = capture_and_witness(args.capture_out, n_runs=args.n_runs)
    else:
        res = witness_clean(args.runs, args.acq, args.rel, args.anchor)
    print(f"WITNESS: {'CLEAN' if res.clean else 'DIRTY'} -- {res.detail}")
    return 0 if res.clean else 1


if __name__ == "__main__":
    sys.exit(main())
