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

from inference.verifier import ANCHOR, Q  # noqa: F401

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
