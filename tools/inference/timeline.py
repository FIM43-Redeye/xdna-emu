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

from inference.verifier import (ANCHOR, Q, anchored_occurrences_per_run,  # noqa: F401
                                additivity_state, offset_window, cross_batch_range)
from inference.grounding import ground_edge, same_domain

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
    causal_offset: Optional[int] = None  # model-derived Δwall; None pre-calibration


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
    capture: Dict[str, object]    # n_runs, event_records, dropout_batches/runs
                                  # (when anchor-dropout fired), plus any
                                  # caller-supplied capture keys


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


def internal_cycles(frame, anchored0, run_dirs=None, anchor_key=ANCHOR) -> Tuple[str, Dict[str, int]]:
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


# ---------------------------------------------------------------------------
# Per-track period builder (Task 8)
# ---------------------------------------------------------------------------

def _as_window(x):
    return x if isinstance(x, tuple) else (x, x)


def build_track(domain, frames, nondet_windows, mean_pos, derives_pairs) -> Track:
    """Order a domain's frames and nondeterministic events by mean_pos and emit
    a sequence of DeterministicPeriod / NondeterministicPeriod into a Track.

    frames: List[(grounding_event, {member: cycle}, floating, anchor_pos)]
        anchor_pos = int (anchored frame, exact) | (min,max) (floating frame).
    nondet_windows: Dict[str, (int,int)]  event key -> window vs upstream frame.
    mean_pos: Dict[str,float]  sequencing key (mean anchored occurrence-0; used
        only for ordering, never reported).
    derives_pairs: set of (child, parent) pairs used to gate F_RESUMPTION_UNATTESTED.

    offset_to_prior_frame is computed by interval subtraction of anchor_pos values
    (within-domain, skew-free): (this_lo - prior_hi, this_hi - prior_lo).
    Both anchored -> degenerate (x,x); either floats -> a proper window.
    """
    # Build a unified sequence of (sort_key, kind, payload) and order by sort_key.
    items = []
    for (g, cyc, floating, apos) in frames:
        items.append((min(mean_pos[m] for m in cyc), "frame", (g, cyc, floating, apos)))
    for k in nondet_windows:
        items.append((mean_pos[k], "nondet", k))
    items.sort(key=lambda t: t[0])

    periods, pending, prior_apos = [], [], None

    def flush(closing_g):
        nonlocal pending
        if not pending:
            return
        evs = list(pending)
        attested = closing_g is not None and any(
            (k, closing_g) in derives_pairs or (closing_g, k) in derives_pairs
            for k in evs)
        flags = []
        if closing_g is None:
            flags.append(F_UNGROUNDED_TAIL)
        elif not attested:
            flags.append(F_RESUMPTION_UNATTESTED)
        periods.append(NondeterministicPeriod(
            events=evs,
            windows={k: nondet_windows[k] for k in evs},
            reasons={k: "within_domain_nonexact" for k in evs},
            order_edges=[],
            grounding_event=closing_g,
            flags=flags))
        pending = []

    for (_, kind, payload) in items:
        if kind == "nondet":
            pending.append(payload)
            continue
        g, cyc, floating, apos = payload
        flush(g)
        otp = None
        if prior_apos is not None:
            lo1, hi1 = _as_window(apos)
            lo0, hi0 = _as_window(prior_apos)
            otp = (lo1 - hi0, hi1 - lo0)   # interval subtraction; (x,x) exact iff both anchored
        periods.append(DeterministicPeriod(
            events=sorted(cyc, key=cyc.get),
            cycles=cyc,
            grounding_event=g,
            floating=floating,
            offset_to_prior_frame=otp))
        prior_apos = apos

    flush(None)
    return Track(domain=domain, periods=periods)


# ---------------------------------------------------------------------------
# Intra-period partial-order edges (Task 9)
# ---------------------------------------------------------------------------

def order_nondeterministic(events, derives_pairs, stable_before):
    """Produce honest partial-order edges for a NondeterministicPeriod.

    Returns List[Tuple[str,str,str]] of (a, b, tag) meaning a-before-b.
    tag="causal"          when (b,a) in derives_pairs (a is parent of b).
    tag="stable_position" when stable_before[(a,b)] is True and no causal edge.
    Pairs with neither relationship are omitted (concurrent).

    Causal takes precedence over stable_position (elif), so a pair that is
    both causally related and positionally stable is reported as causal only.
    """
    edges = []
    for a in events:
        for b in events:
            if a == b:
                continue
            if (b, a) in derives_pairs:           # parent a -> child b
                edges.append((a, b, "causal"))
            elif stable_before.get((a, b)):
                edges.append((a, b, "stable_position"))
    return edges


# ---------------------------------------------------------------------------
# Task 10: Cross-track weave + logical connectivity classification
# ---------------------------------------------------------------------------

def weave(run_dirs, cross_domain_pairs, anchor_key=ANCHOR, model=None) -> List[CrossTrackEdge]:
    """Ground each cross-domain (child, parent) pair via grounding.ground_edge.

    Skips same-domain pairs (within-domain grounding belongs to build_track).
    Maps a Gap return to a CrossTrackEdge; a Segment return is skipped
    (same-domain exact result -- should not occur for cross-domain inputs, but
    handled defensively).
    """
    from inference.grounding import Gap
    from inference.loader import load_fired
    # A cross-track edge must connect two OBSERVED events. A configured pair whose
    # endpoint never fired on this kernel would otherwise yield a dangling edge --
    # an endpoint present in no track (a real-HW finding: memmod events the dump
    # configures but the kernel never triggers). The never-firing is already
    # recorded upstream as a never_fired constraint, so dropping the edge here
    # loses no information.
    fired = {f.args[0] for f in load_fired(run_dirs, anchor_key)}
    edges = []
    for (child, parent) in cross_domain_pairs:
        if same_domain(child, parent):
            continue
        if child not in fired or parent not in fired:
            continue
        g = ground_edge(run_dirs, child, parent, anchor_key, model=model)
        if isinstance(g, Gap):
            edges.append(CrossTrackEdge(child=child, parent=parent,
                                        reason=g.reason,
                                        reproduction_offset=g.reproduction_offset,
                                        causal_offset=g.causal_offset))
    return edges


# ---------------------------------------------------------------------------
# Task 11: census + assemble_timeline orchestration + renderer
# ---------------------------------------------------------------------------

# count-truncation honesty: a declared best-effort limit when the trace-buffer
# capacity is not derivable from the config dump (G4). NOT a silently-absent flag.
F_COUNT_CEILING_UNKNOWN = "count_ceiling_unknown"


@dataclass
class _EventVectors:
    """Per clusterable event: the run-0-relative jitter vector + the raw values
    used for sequencing, frame grounding, and window fallbacks. Derived ONLY
    from anchored_occurrences_per_run (single source for anchored values)."""
    vec: Dict[str, List[int]]              # key -> per-run occurrence-0 anchored value
    jitter: Dict[str, Tuple[int, ...]]     # key -> per-run (occ0 - run0_occ0)
    anchored0: Dict[str, int]              # key -> run-0 occurrence-0 (frame representative)
    n_eff: Dict[str, int]                  # key -> #runs the event fired in
    mean_pos: Dict[str, float]             # key -> mean occ-0 (sequencing only)


def _event_vectors(run_dirs, clusterable, pinned, anchor_key=ANCHOR) -> _EventVectors:
    vec, jitter, anchored0, n_eff, mean_pos = {}, {}, {}, {}, {}
    for key in clusterable:
        per_run = anchored_occurrences_per_run(run_dirs, key, pinned[key], anchor_key)
        v = [occ[0] for occ in per_run if occ]       # runs where it fired
        if not v:
            continue
        vec[key] = v
        jitter[key] = tuple(x - v[0] for x in v)     # all-zero => anchor-rigid
        anchored0[key] = v[0]
        n_eff[key] = len(v)
        mean_pos[key] = sum(v) / len(v)
    return _EventVectors(vec, jitter, anchored0, n_eff, mean_pos)


def _buffer_ceiling(dump) -> Optional[int]:
    """Derive the count-truncation ceiling (max events a trace buffer can hold)
    from the config dump: trace-buffer bytes / bytes-per-event. Returns None when
    the dump is absent or does not expose the capacity (G4 best-effort)."""
    if dump is None:
        return None
    nbytes = getattr(dump, "trace_buffer_bytes", None)
    per = getattr(dump, "bytes_per_event", None)
    if not nbytes or not per:
        return None
    return int(nbytes) // int(per)


def _build_one_track(domain, domain_keys, ev, run_dirs, derives_pairs, anchor_key):
    """Cluster one domain's clusterable keys into a Track, then attach intra-period
    order edges (G7) and overlaps_frame flags (G2). Returns (track, flags) where
    flags are timeline-level (F_CROSS_BATCH_FRAME per held-out key)."""
    tl_flags = []

    # G1: an event whose anchored value varies across the batches that trace it
    # cannot safely share a frame spanning pinned batches -> held out as a
    # nondeterministic singleton, and the timeline records F_CROSS_BATCH_FRAME.
    held_out = set()
    for key in domain_keys:
        if cross_batch_range(run_dirs, key, anchor_key) > 0:
            held_out.add(key)
            tl_flags.append(F_CROSS_BATCH_FRAME)
    cluster_keys = [k for k in domain_keys if k not in held_out]

    cr = rigid_clusters({k: ev.jitter[k] for k in cluster_keys}, ev.n_eff, derives_pairs)
    nondet_keys = set(cr.nondeterministic) | held_out

    frame_payloads = []
    for frame in cr.frames:
        try:
            g, cyc = internal_cycles(frame, ev.anchored0, run_dirs, anchor_key)
        except ClusterViolation:
            nondet_keys.update(frame.members)     # demote: emit no frame
            continue
        vec_g = ev.vec[g]
        apos = ev.anchored0[g] if not frame.floating else (min(vec_g), max(vec_g))
        frame_payloads.append((g, cyc, frame.floating, apos))

    # nondet windows: relative to an anchored frame's grounding event when one
    # exists, else the event's own per-run vector min/max.
    anchored_frames = [fp for fp in frame_payloads if not fp[2]]
    ref_grounding = anchored_frames[0][0] if anchored_frames else None
    nondet_windows = {}
    for k in sorted(nondet_keys):
        win = offset_window(run_dirs, k, ref_grounding, anchor_key) if ref_grounding else None
        if win is None:
            vk = ev.vec[k]
            win = (min(vk), max(vk))
        nondet_windows[k] = win

    track = build_track(domain, frame_payloads, nondet_windows, ev.mean_pos, derives_pairs)

    _attach_order_edges(track, run_dirs, derives_pairs, anchor_key)        # G7
    _flag_overlaps_frame(track, ref_grounding, ev.anchored0)              # G2
    return track, tl_flags


def _attach_order_edges(track, run_dirs, derives_pairs, anchor_key):
    """G7: per NondeterministicPeriod, derive honest partial-order edges. An event
    a is stable-before b iff offset_window(b, a) min > 0 (b strictly later, every
    run); causal edges come from derives_pairs (handled in order_nondeterministic)."""
    for period in track.periods:
        if not isinstance(period, NondeterministicPeriod):
            continue
        evs = period.events
        stable_before = {}
        for a in evs:
            for b in evs:
                if a == b:
                    continue
                w = offset_window(run_dirs, b, a, anchor_key)
                stable_before[(a, b)] = (w is not None and w[0] > 0)
        period.order_edges = order_nondeterministic(evs, derives_pairs, stable_before)


def _frame_extent_ref(period, ref_grounding, anchored0):
    """Anchored extent of a DeterministicPeriod expressed in ref_grounding
    coordinates (min,max of member offsets), or None for a floating frame / when
    no anchored reference exists."""
    if ref_grounding is None or period.floating:
        return None
    base = anchored0[period.grounding_event] - anchored0[ref_grounding]
    offs = [base + c for c in period.cycles.values()]
    return (min(offs), max(offs))


def _flag_overlaps_frame(track, ref_grounding, anchored0):
    """G2: a nondeterministic event whose window overlaps the anchored extent of an
    adjacent frame is flagged F_OVERLAPS_FRAME on its period (earned order edges
    retained; NOT blanket-marked concurrent)."""
    periods = track.periods
    for i, period in enumerate(periods):
        if not isinstance(period, NondeterministicPeriod):
            continue
        neighbors = []
        if i > 0 and isinstance(periods[i - 1], DeterministicPeriod):
            neighbors.append(periods[i - 1])
        if i + 1 < len(periods) and isinstance(periods[i + 1], DeterministicPeriod):
            neighbors.append(periods[i + 1])
        overlap = False
        for nb in neighbors:
            ext = _frame_extent_ref(nb, ref_grounding, anchored0)
            if ext is None:
                continue
            for k in period.events:
                lo, hi = period.windows[k]
                if lo <= ext[1] and ext[0] <= hi:        # interval overlap
                    overlap = True
        if overlap and F_OVERLAPS_FRAME not in period.flags:
            period.flags.append(F_OVERLAPS_FRAME)


def census_of(tracks, intermittent, excluded, edges) -> Census:
    """Bucket the timeline into a Census. content_ok is True iff the fraction of
    events living in deterministic frames (anchored + floating) meets the floor."""
    anchored = floating = nondeterministic = 0
    for tr in tracks:
        for p in tr.periods:
            if isinstance(p, DeterministicPeriod):
                if p.floating:
                    floating += len(p.events)
                else:
                    anchored += len(p.events)
            else:
                nondeterministic += len(p.events)
    interm = sum(len(pc.events) for pc in intermittent)
    excl = len(excluded)
    event_buckets = {"anchored": anchored, "floating": floating,
                     "nondeterministic": nondeterministic,
                     "intermittent": interm, "excluded": excl}
    edge_buckets = {
        "reproduction": sum(1 for e in edges if e.reproduction_offset is not None),
        "existence_only": sum(1 for e in edges if e.reproduction_offset is None),
    }
    total = sum(event_buckets.values())
    in_frames = anchored + floating
    content_ok = total > 0 and (in_frames / total) >= CENSUS_CONTENT_FLOOR
    return Census(events=event_buckets, edges=edge_buckets, content_ok=content_ok)


def assemble_timeline(run_dirs, configured, derives_pairs, cross_domain_pairs,
                      dump=None, start_col=1, anchor_key=ANCHOR,
                      capture=None, model=None) -> IntegratedTimeline:
    """Orchestrate Tasks 4-10 into one IntegratedTimeline. See the ten-step wiring
    in the task brief. Delegates everywhere; this is glue, not algorithm."""
    flags: List[str] = []

    # (1) eligibility gates -> clusterable / pinned / intermittent / excluded.
    elig = eligibility(run_dirs, configured, anchor_key)

    # (1b) anchor-dropout honesty (capture-health): a load-contaminated capture
    # where the anchor failed to fire in one or more batches/runs must NOT look
    # clean. Surface the timeline-level flag and stash the detail in capture so a
    # consumer can see exactly which runs/batches lost the anchor reference.
    dropout = {}
    if elig.dropout_batches or elig.dropout_runs:
        flags.append(F_ANCHOR_DROPOUT)
        dropout = {"dropout_batches": elig.dropout_batches,
                   "dropout_runs": elig.dropout_runs}

    # (2) count-truncation ceiling (G4): derive from the dump when available,
    # else declare the limit honestly with a timeline flag.
    buffer_ceiling = _buffer_ceiling(dump)
    if buffer_ceiling is None:
        flags.append(F_COUNT_CEILING_UNKNOWN)

    # (3) per-event characterization + jitter/sequencing vectors.
    ev = _event_vectors(run_dirs, elig.clusterable, elig.pinned, anchor_key)
    records = {k: characterize_event(run_dirs, k, elig.pinned[k],
                                     buffer_ceiling=buffer_ceiling, anchor_key=anchor_key)
               for k in ev.vec}

    # (4)-(8) per-domain clustering -> cycles -> track -> order edges -> overlaps.
    by_domain: Dict[str, List[str]] = {}
    for key in ev.vec:                                   # only events that fired
        by_domain.setdefault(_domain_of(key), []).append(key)
    tracks = []
    for domain in sorted(by_domain):
        track, tflags = _build_one_track(domain, sorted(by_domain[domain]), ev,
                                         run_dirs, derives_pairs, anchor_key)
        tracks.append(track)
        flags.extend(tflags)

    # (9) cross-track weave + logical connectivity classification. The conversation
    # set is the cross-domain candidate pairs (route reachability the ledger
    # computes); classify each cross-tile coupling honestly against the trace.
    # weave couples; classify_connectivity audits. No dump needed.
    edges = weave(run_dirs, cross_domain_pairs, anchor_key, model=model)
    from inference.connectivity import (classify_connectivity, OBSERVED_UNGROUNDED,
                                        UNOBSERVED)
    from inference.loader import load_fired
    fired = {f.args[0] for f in load_fired(run_dirs, anchor_key)}
    for (a, b), status in sorted(classify_connectivity(cross_domain_pairs, fired, edges).items()):
        if status == OBSERVED_UNGROUNDED:
            flags.append(f"connectivity_defect:{a}~{b}")
        elif status == UNOBSERVED:
            flags.append(f"connectivity_unobserved:{a}~{b}")

    # (10) census.
    census = census_of(tracks, elig.intermittent, elig.excluded, edges)

    cap = dict(capture) if capture else {}
    cap.setdefault("n_runs", len(run_dirs))
    cap.setdefault("event_records", records)
    cap.setdefault("excluded", elig.excluded)
    if dropout:
        cap["dropout_batches"] = dropout["dropout_batches"]
        cap["dropout_runs"] = dropout["dropout_runs"]
    # (#5) dedupe order-preserving: N held-out keys would otherwise emit N
    # identical F_CROSS_BATCH_FRAME strings; per-pair flags (connectivity_defect)
    # are already unique and survive untouched.
    flags = list(dict.fromkeys(flags))
    return IntegratedTimeline(tracks=tracks, cross_track_edges=edges,
                              intermittent=elig.intermittent, flags=flags,
                              census=census, capture=cap)


def render_timeline(tl: IntegratedTimeline) -> str:
    """Plain-text fully-integrated view: per-track A->B->C local cycles, windows,
    concurrency, cross-track edges, intermittent presence classes, excluded
    events, per-event EventRecord flags, the timeline flags, and the census line."""
    lines: List[str] = []
    # Per-event flags live in capture["event_records"] (key -> EventRecord); show
    # any non-empty EventRecord.flags inline next to the event.
    records = tl.capture.get("event_records", {}) if isinstance(tl.capture, dict) else {}

    def _ev_flags(key):
        rec = records.get(key) if isinstance(records, dict) else None
        f = getattr(rec, "flags", None)
        return f" flags={f}" if f else ""

    for tr in tl.tracks:
        lines.append(f"TRACK {tr.domain}")
        for p in tr.periods:
            if isinstance(p, DeterministicPeriod):
                tag = "DET(floating)" if p.floating else "DET"
                chain = " -> ".join(f"{k}@{p.cycles[k]}{_ev_flags(k)}" for k in p.events)
                otp = ("" if p.offset_to_prior_frame is None
                       else f" prior_offset={p.offset_to_prior_frame}")
                lines.append(f"  [{tag}] {chain}  ground={p.grounding_event}{otp}")
            else:
                wins = ", ".join(f"{k}~{p.windows[k]}{_ev_flags(k)}" for k in p.events)
                ordered = {(a, b) for (a, b, _) in p.order_edges}
                concurrent = [(a, b) for i, a in enumerate(p.events)
                              for b in p.events[i + 1:]
                              if (a, b) not in ordered and (b, a) not in ordered]
                lines.append(f"  [NONDET] {wins}  ground={p.grounding_event}")
                lines.append(f"    order={p.order_edges}  concurrency={concurrent}"
                             f"  flags={p.flags}")
    for e in tl.cross_track_edges:
        causal = getattr(e, "causal_offset", None)
        suffix = "" if causal is None else f", causal_offset={causal} [model-derived]"
        lines.append(f"EDGE {e.child} <- {e.parent} "
                     f"({e.reason}, reproduction_offset={e.reproduction_offset}{suffix})")
    # Intermittent presence classes: appearance run-index tuple + members
    # (+ candidate mutual-exclusion complement when set).
    if tl.intermittent:
        lines.append("INTERMITTENT")
        for pc in tl.intermittent:
            comp = "" if pc.complement_of is None else f" complement_of={pc.complement_of}"
            lines.append(f"  appearance={pc.appearance} events={pc.events}{comp}")
    # Excluded events: event key -> reason (e.g. batch_flip). Sourced from the
    # census bucket count plus the per-key detail when present in capture.
    excluded = tl.capture.get("excluded", {}) if isinstance(tl.capture, dict) else {}
    if excluded:
        lines.append("EXCLUDED")
        for k in sorted(excluded):
            lines.append(f"  {k} -> {excluded[k]}")
    lines.append(f"FLAGS {tl.flags}")
    c = tl.census
    lines.append(f"CENSUS events={c.events} edges={c.edges} content_ok={c.content_ok}")
    return "\n".join(lines)
