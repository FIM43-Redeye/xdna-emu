"""Logical connectivity classification for the integrated timeline.

The connectivity oracle is the set of cross-tile logical CONVERSATIONS the
config implies -- the tile-pair projection of the cross-domain candidate pairs
(route-graph reachability that generate_ledger computes over configured events).
It is NOT the physical adjacent-hop wiring the old coupling_oracle enumerated.

Each conversation is classified honestly against the trace:

  grounded                -- some candidate pair for the tile coupling had both
                             endpoints fire AND weave produced a CrossTrackEdge
                             connecting the two tiles.  Healthy: no flag.
  observed_but_ungrounded -- both endpoints fired for some candidate pair, but
                             weave grounded no edge between the tiles -> the
                             genuine connectivity defect.
  unobserved              -- no candidate pair for this coupling had both
                             endpoints fire -> honest gap (the trace does not
                             watch both ends), NOT a defect.

NOTE: with the present grounding a both-fired cross-domain pair always resolves
to a Gap and therefore always grounds, so the live engine does not currently
produce observed_but_ungrounded; that bucket is reserved for a future grounding
that can fail on observed pairs.  The classifier defines it regardless and it is
unit-tested directly.

Weave couples; this module audits.  Q=0: derived from route reachability +
observed firing, never inferred from timing correlation, never tuned to pass.
"""
from typing import Dict, List, Set, Tuple

GROUNDED = "grounded"
OBSERVED_UNGROUNDED = "observed_but_ungrounded"
UNOBSERVED = "unobserved"


def _tile(key: str) -> str:
    """'col|row|pkt|name' or 'col|row|pkt' -> 'col|row'."""
    parts = key.split("|")
    return f"{parts[0]}|{parts[1]}"


def classify_connectivity(cross_domain_pairs: List[Tuple[str, str]],
                          fired: Set[str],
                          edges) -> Dict[Tuple[str, str], str]:
    """Classify each cross-tile logical conversation into a connectivity status.

    cross_domain_pairs: (child_key, parent_key) candidate pairs from the ledger
        reachability.  Same-tile pairs are ignored -- intra-tile handoffs are not
        cross-track conversations.
    fired: event keys observed to fire across the runs.
    edges: iterable of objects with .child and .parent event-key attributes
        (weave's CrossTrackEdge list).

    Returns {sorted (tileA, tileB): status} for every cross-tile coupling seen
    in the candidate pairs or grounded by weave.
    """
    # A weave edge can be cross-MODULE but same-TILE (core pkt0 -> memmod pkt1);
    # such an edge is not a cross-tile conversation, so drop it here -- otherwise
    # the `all_pairs |= grounded` fold below would reintroduce the same-tile pair.
    grounded: Set[Tuple[str, str]] = set()
    for e in edges:
        ta_e, tb_e = _tile(e.child), _tile(e.parent)
        if ta_e != tb_e:
            grounded.add(tuple(sorted((ta_e, tb_e))))
    all_pairs: Set[Tuple[str, str]] = set()
    observed: Set[Tuple[str, str]] = set()
    for child, parent in cross_domain_pairs:
        ta, tb = _tile(child), _tile(parent)
        if ta == tb:
            continue
        pr = tuple(sorted((ta, tb)))
        all_pairs.add(pr)
        if child in fired and parent in fired:
            observed.add(pr)
    all_pairs |= grounded  # a grounded coupling is reported even if only edges saw it
    out: Dict[Tuple[str, str], str] = {}
    for pr in sorted(all_pairs):
        if pr in grounded:
            out[pr] = GROUNDED
        elif pr in observed:
            out[pr] = OBSERVED_UNGROUNDED
        else:
            out[pr] = UNOBSERVED
    return out
