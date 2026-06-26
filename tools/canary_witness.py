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
import sys
from dataclasses import dataclass
from typing import List, Optional

from inference.grounding import ground_edge, Segment
from inference.verifier import ANCHOR

# Sentinel: add_one_using_dma compute-tile core-lock span (col 1, row 2, core
# module pkt 0). Verified cycle-exact-under-clean / flicker-under-load.
SENTINEL_ACQ = "1|2|0|INSTR_LOCK_ACQUIRE_REQ"
SENTINEL_REL = "1|2|0|INSTR_LOCK_RELEASE_REQ"


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Canary capture-cleanliness witness.")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="sentinel capture run dirs (each with batch_00/hw/trace.events.json)")
    ap.add_argument("--acq", default=SENTINEL_ACQ)
    ap.add_argument("--rel", default=SENTINEL_REL)
    ap.add_argument("--anchor", default=ANCHOR)
    args = ap.parse_args(argv)
    res = witness_clean(args.runs, args.acq, args.rel, args.anchor)
    print(f"WITNESS: {'CLEAN' if res.clean else 'DIRTY'} -- {res.detail}")
    return 0 if res.clean else 1


if __name__ == "__main__":
    sys.exit(main())
