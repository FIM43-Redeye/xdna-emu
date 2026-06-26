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

import collections as _c
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from inference.verifier import ANCHOR, Q, anchored_occurrences_per_run, additivity_state  # noqa: F401

# trace_join is a TOP-LEVEL module in tools/ (NOT inference.trace_join). Bind the
# names at module level so the tests' monkeypatch.setattr(T, "batch_firsts", ...)
# / setattr(T, "_batch_names", ...) intercept the bare calls below.
from trace_join import batch_firsts, _batch_names

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


# ---------------------------------------------------------------------------
# Eligibility gates (gate order: anchor-dropout, presence, batch-stability)
# ---------------------------------------------------------------------------

@dataclass
class EligibilityResult:
    clusterable: List[str]
    pinned: Dict[str, str]
    intermittent: List[PresenceClass]
    excluded: Dict[str, str]
    dropout_runs: List[int]                      # runs where EVERY batch lost the anchor
    dropout_batches: List[Tuple[int, str]]       # per-batch anchor dropout (capture-health)


def _pinned_batch_index(run_dir, key, anchor_key):
    # Only anchor-present batches are considered (batch_firsts returns {} when the
    # anchor didn't fire) -- so a single anchorless batch never fabricates absence.
    for bn in _batch_names(run_dir):
        if key in batch_firsts(run_dir, bn, anchor_key):
            return bn
    return None


def eligibility(run_dirs, configured, anchor_key=ANCHOR) -> EligibilityResult:
    # Per-batch anchor dropout (spec gate 1): a batch with no anchor firing does not
    # count as event absence. Record each for capture-health; a run is fully-dropout
    # only if ALL its batches lost the anchor.
    dropout_batches = [(i, bn) for i, rd in enumerate(run_dirs)
                       for bn in _batch_names(rd)
                       if not batch_firsts(rd, bn, anchor_key)]
    dropout_runs = [i for i, rd in enumerate(run_dirs)
                    if all(not batch_firsts(rd, bn, anchor_key) for bn in _batch_names(rd))]
    live = [(i, rd) for i, rd in enumerate(run_dirs) if i not in dropout_runs]
    appear, pin = {}, {}
    for key in configured:
        runs_present = []
        batches = set()
        for i, rd in live:
            bn = _pinned_batch_index(rd, key, anchor_key)
            if bn is not None:
                runs_present.append(i); batches.add(bn)
        appear[key] = tuple(sorted(runs_present))
        pin[key] = batches
    all_live = tuple(sorted(i for i, _ in live))
    clusterable, pinned, excluded = [], {}, {}
    by_appearance: Dict[tuple, List[str]] = {}
    for key in configured:
        if not appear[key]:
            continue  # never fired
        if appear[key] != all_live:
            by_appearance.setdefault(appear[key], []).append(key)
            continue
        if len(pin[key]) != 1:
            excluded[key] = F_BATCH_FLIP
            continue
        clusterable.append(key)
        pinned[key] = next(iter(pin[key]))
    intermittent = [PresenceClass(appearance=a, events=evs)
                    for a, evs in sorted(by_appearance.items())]
    # cross-link complementary appearance sets (candidate mutual exclusion)
    aset = {pc.appearance: pc for pc in intermittent}
    for pc in intermittent:
        comp = tuple(i for i in all_live if i not in pc.appearance)
        if comp in aset:
            pc.complement_of = comp
    return EligibilityResult(sorted(clusterable), pinned, intermittent, excluded,
                             dropout_runs, dropout_batches)


# ---------------------------------------------------------------------------
# Jitter-vector clustering (Task 6)
# ---------------------------------------------------------------------------

@dataclass
class ClusterFrame:
    members: List[str]
    floating: bool
    corroborated: bool
    flags: List[str] = field(default_factory=list)


@dataclass
class ClusterResult:
    frames: List[ClusterFrame]
    nondeterministic: List[str]


def _corroborated(members, derives_pairs) -> bool:
    ms = set(members)
    # direct chain edge between two members
    if any((a, b) in derives_pairs for a in ms for b in ms if a != b):
        return True
    # common parent: P with (m, P) for >=2 members
    parents = _c.Counter(p for (c, p) in derives_pairs if c in ms)
    return any(v >= 2 for v in parents.values())


def _false_cluster_ok(members, n_eff) -> bool:
    # BOTH gates (spec step 4): N-floor AND the estimated false-cluster bound.
    n = min(n_eff[m] for m in members)
    if n < MIN_N_FLOATING:
        return False
    pairs = len(members) * (len(members) - 1) // 2
    return pairs * (P_C ** (n - 1)) < FALSE_CLUSTER_BOUND


def rigid_clusters(jitter_vectors: Dict[str, Tuple[int, ...]],
                   n_eff: Dict[str, int],
                   derives_pairs: set) -> ClusterResult:
    groups: Dict[tuple, List[str]] = {}
    for k, jv in jitter_vectors.items():
        groups.setdefault(jv, []).append(k)
    frames, nondet = [], []
    for jv, members in groups.items():
        members = sorted(members)
        if all(v == 0 for v in jv):
            frames.append(ClusterFrame(members, floating=False, corroborated=True))
            continue
        if len(members) < 2:
            nondet.extend(members)
            continue
        corr = _corroborated(members, derives_pairs)
        if corr or _false_cluster_ok(members, n_eff):
            flags = [] if corr and min(n_eff[m] for m in members) >= MIN_N_FLOATING \
                    else ([F_PROVISIONAL_LOW_N] if corr else [])
            frames.append(ClusterFrame(members, floating=True, corroborated=corr, flags=flags))
        else:
            nondet.extend(members)
    return ClusterResult(frames, sorted(nondet))


# ---------------------------------------------------------------------------
# Intra-frame cycle resolution (Task 7)
# ---------------------------------------------------------------------------

class ClusterViolation(Exception):
    """Raised when additivity_state returns 'violation' for a cluster frame."""


def internal_cycles(frame, anchored0, run_dirs=None, anchor_key=ANCHOR):
    """Resolve a single-domain ClusterFrame to (grounding_event, {member: local_cycle}).

    Local zero = the member with the smallest anchored occurrence-0 value; each
    member's cycle = anchored0[member] - anchored0[zero].  This is exact and
    skew-free because frame is single-domain by construction.

    When run_dirs is given and the frame has >= 3 members, calls additivity_state;
    a "violation" raises ClusterViolation (caller should demote the frame).
    "unverifiable" / "vacuous" / "pass" do not raise.

    Invariant: a cross-domain frame would produce (Delta_wall + skew) as a cycle
    value -- the fatal-A bug.  Assert single-domain and fail loud on a wiring slip.
    """
    # Invariant guard: a frame is single-domain by construction; a cross-domain
    # "cycle" would be Delta_wall + skew (fatal-A). Fail loud on a wiring slip.
    assert len({_domain_of(m) for m in frame.members}) == 1, \
        f"cross-domain frame members: {frame.members}"
    members = sorted(frame.members, key=lambda m: anchored0[m])
    zero = members[0]
    cycles = {m: anchored0[m] - anchored0[zero] for m in members}
    if run_dirs is not None and len(members) >= 3:
        if additivity_state(run_dirs, members, anchor_key) == "violation":
            raise ClusterViolation(f"additivity violation in frame {members}")
    return zero, cycles
