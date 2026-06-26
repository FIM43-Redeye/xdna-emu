"""
timeline.py -- Integrated timeline data model for the trace-inference engine.

Invariants
----------
No cross-domain cycle: a raw cross-domain anchored offset is (Δwall + skew) and
must NEVER be emitted as a tile-cycle value.  Any CrossTrackEdge with
reason="cross_domain" carries a wall-time reproduction_offset or None; the
caller must never interpret it as a local cycle count.

Threshold provisionality (Q=0): all numeric threshold constants in this module
(MIN_N_FLOATING, P_C, FALSE_CLUSTER_BOUND, CENSUS_CONTENT_FLOOR) are
provisional placeholders pending corpus jitter-entropy calibration.  They are
NEVER tuned to make a test pass.  Q refers to the calibration corpus size,
which is zero at this writing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from inference.verifier import ANCHOR, Q, anchored_occurrences_per_run  # noqa: F401

# ---------------------------------------------------------------------------
# Provisional thresholds (corpus-calibrated later; Q=0). NEVER tuned to pass
# a test.  MIN_N_FLOATING and (P_C, FALSE_CLUSTER_BOUND) are co-calibrated:
# the estimated false-cluster probability P_C**(N-1) must drop below
# FALSE_CLUSTER_BOUND at N == MIN_N_FLOATING.  Values are provisional
# placeholders pending the corpus jitter-entropy measurement.
# ---------------------------------------------------------------------------
MIN_N_FLOATING = 12         # uncorroborated floating cluster needs >= this many runs
P_C = 0.4                   # provisional per-component jitter collision rate (low-entropy)
FALSE_CLUSTER_BOUND = 1e-3  # estimated coincidence prob must be below this
CENSUS_CONTENT_FLOOR = 0.5  # >= this fraction of events in a deterministic frame

# ---------------------------------------------------------------------------
# Honesty flags -- string constants used in EventRecord.flags,
# NondeterministicPeriod.flags, and IntegratedTimeline.flags.
# ---------------------------------------------------------------------------
F_COUNT_WINDOW = "count_window"
F_COUNT_TRUNCATED = "count_truncated"
F_REORDERABLE = "occurrences_reorderable"
F_PROVISIONAL_LOW_N = "provisional_low_n"
F_UNGROUNDED_TAIL = "ungrounded_tail"
F_RESUMPTION_UNATTESTED = "resumption_unattested"
F_OVERLAPS_FRAME = "overlaps_frame"
F_ANCHOR_DROPOUT = "anchor_dropout"
F_BATCH_FLIP = "batch_flip"
F_CROSS_BATCH_FRAME = "cross_batch_frame"   # frame members span pinned batches w/o batch-invariance


# ---------------------------------------------------------------------------
# Primitive occurrence types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pulse:
    ts: int                       # anchored_ts of one firing


@dataclass(frozen=True)
class Span:
    begin: int                    # anchored begin ts
    length: int                   # end - begin


Occurrence = Union[Pulse, Span]


# ---------------------------------------------------------------------------
# Per-event timing records
# ---------------------------------------------------------------------------

@dataclass
class RigidRun:
    start_index: int              # index of first occurrence in this run
    cycles: List[int]             # per-occurrence exact cycle (Pulse) or begin-cycle (Span)
    lengths: Optional[List[int]] = None  # per-occurrence exact span length, if Span


@dataclass
class JitterPoint:
    index: int
    window: Tuple[int, int]       # [min,max] of this occurrence across runs
    length_window: Optional[Tuple[int, int]] = None  # for spans


@dataclass
class EventRecord:
    key: str
    domain: str                   # "col|row|pkt_type"
    pinned_batch: Optional[str]
    n_eff: int                    # per-event effective N (post dropout/presence)
    rigid_runs: List[RigidRun] = field(default_factory=list)
    jitter_points: List[JitterPoint] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Timeline periods (deterministic and nondeterministic)
# ---------------------------------------------------------------------------

@dataclass
class DeterministicPeriod:
    events: List[str]             # event keys, in local-cycle order
    cycles: Dict[str, int]        # event key -> exact local cycle (zero at grounding event)
    grounding_event: str          # the frame's local zero
    floating: bool                # True if not anchor-rigid
    offset_to_prior_frame: Optional[Tuple[int, int]] = None  # within-track; (x,x) exact or window


@dataclass
class NondeterministicPeriod:
    events: List[str]
    windows: Dict[str, Tuple[int, int]]      # event key -> [min,max] vs upstream frame ref
    reasons: Dict[str, str]                  # event key -> gap reason
    order_edges: List[Tuple[str, str, str]]  # (a, b, "causal"|"stable_position") meaning a<b
    grounding_event: Optional[str]           # None -> ungrounded_tail
    flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Track (per-domain sequence of periods)
# ---------------------------------------------------------------------------

@dataclass
class Track:
    domain: str
    periods: List[Union[DeterministicPeriod, NondeterministicPeriod]]


# ---------------------------------------------------------------------------
# Cross-track causal edge
# ---------------------------------------------------------------------------

@dataclass
class CrossTrackEdge:
    child: str
    parent: str
    reason: str                   # cross_domain | async_cdc
    reproduction_offset: Optional[int]  # None -> existence-only


# ---------------------------------------------------------------------------
# Presence and census summaries
# ---------------------------------------------------------------------------

@dataclass
class PresenceClass:
    appearance: Tuple[int, ...]   # sorted run indices the members fired in
    events: List[str]
    complement_of: Optional[Tuple[int, ...]] = None  # candidate mutual-exclusion


@dataclass
class Census:
    events: Dict[str, int]        # bucket -> count (anchored/floating/nondeterministic/intermittent/excluded)
    edges: Dict[str, int]         # bucket -> count (reproduction/existence_only)
    content_ok: bool              # meets CENSUS_CONTENT_FLOOR


# ---------------------------------------------------------------------------
# Top-level integrated timeline
# ---------------------------------------------------------------------------

@dataclass
class IntegratedTimeline:
    tracks: List[Track]
    cross_track_edges: List[CrossTrackEdge]
    intermittent: List[PresenceClass]
    flags: List[str]
    census: Census
    capture: Dict[str, object]    # {witness, n_runs, input_id}


# ---------------------------------------------------------------------------
# Level-event classification
# ---------------------------------------------------------------------------

LEVEL_EVENTS = {"LOCK_STALL", "MEMORY_STALL", "STREAM_STALL"}  # + PORT_RUNNING_* by prefix


def _domain_of(key: str) -> str:
    return key.rsplit("|", 1)[0]


def _is_level(key: str) -> bool:
    name = key.rsplit("|", 1)[1]
    return name.startswith("PORT_RUNNING") or name in LEVEL_EVENTS


def _pair_spans(occ):
    # consecutive firings alternate begin/end -> (begin, length); drop a dangling begin
    return [(occ[i], occ[i + 1] - occ[i]) for i in range(0, len(occ) - 1, 2)]


# ---------------------------------------------------------------------------
# Occurrence characterization: rigid-run segmentation
# ---------------------------------------------------------------------------

def characterize_event(run_dirs, key, pinned_batch, *, is_span=None,
                       buffer_ceiling=None, anchor_key=ANCHOR) -> EventRecord:
    if is_span is None:
        is_span = _is_level(key)
    per_run = anchored_occurrences_per_run(run_dirs, key, pinned_batch, anchor_key)
    present = [r for r in per_run if r]
    n_eff = len(present)
    flags = []
    if any(r != sorted(r) for r in present):                 # raw firings out of order
        flags.append(F_REORDERABLE)
    if buffer_ceiling is not None and any(len(r) == buffer_ceiling for r in present):
        flags.append(F_COUNT_TRUNCATED)
    er = EventRecord(key=key, domain=_domain_of(key), pinned_batch=pinned_batch,
                     n_eff=n_eff, flags=flags)
    runs = [_pair_spans(r) for r in present] if is_span else present
    if len({len(r) for r in runs}) > 1:
        flags.append(F_COUNT_WINDOW)
    max_k = max((len(r) for r in runs), default=0)
    run = None
    for k in range(max_k):
        items = [r[k] for r in runs if len(r) > k]
        if is_span:
            begins = [it[0] for it in items]; lens = [it[1] for it in items]
            blo, bhi = min(begins), max(begins); llo, lhi = min(lens), max(lens)
            rigid = (blo == bhi and llo == lhi)            # begin AND length range-0
        else:
            blo, bhi = min(items), max(items); llo = lhi = None
            rigid = (blo == bhi)
        if rigid:
            if run is None:
                run = RigidRun(start_index=k, cycles=[], lengths=([] if is_span else None))
            run.cycles.append(blo)
            if is_span:
                run.lengths.append(llo)
        else:
            if run is not None:
                er.rigid_runs.append(run); run = None
            jp = JitterPoint(index=k, window=(blo, bhi),
                             length_window=((llo, lhi) if is_span else None))
            er.jitter_points.append(jp)
    if run is not None:
        er.rigid_runs.append(run)
    return er
