"""Directed-graph reachability over the xdna-emu route graph.

Physical node identity (matches the Rust BFS in A4/A5):
  A node is identified by (col, row, port, dir) -- NOT by kind.
  The same physical stream-switch port may be labelled with a different kind
  string by different edge producers (e.g., inter_tile vs. intra_tile).
  Keying on kind would break BFS at every inter/intra seam, which is why the
  Rust side also dropped kind from node identity.

API:
  Reachability(edges) -- accepts an iterable of RouteEdge objects OR
                         (PortRef, PortRef) tuples (for test convenience).
  .reachable(src, dst) -> bool  -- BFS from src; True if dst is reachable.
  .reaches_any(src)   -> set[PortRef]  -- all PortRefs reachable from src.

The returned set from reaches_any contains PortRef objects taken from the
edge list (so kind is preserved from the first-seen edge for that physical
port -- callers should use only (col, row, port, dir) for identity checks).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable, Union

from config_extract.dump_model import PortRef, RouteEdge

# Physical key: (col, row, port, dir)
_PhysKey = tuple[int, int, int, str]

_EdgeInput = Union[RouteEdge, tuple[PortRef, PortRef]]


def _phys(p: PortRef) -> _PhysKey:
    return (p.col, p.row, p.port, p.dir)


def _normalize(edge: _EdgeInput) -> tuple[PortRef, PortRef]:
    """Accept either a RouteEdge or a (src, dst) PortRef tuple."""
    if isinstance(edge, RouteEdge):
        return edge.src, edge.dst
    return edge  # already a (PortRef, PortRef) tuple


class Reachability:
    """Directed reachability index over a stream-switch route graph.

    Build once per config dump; query many times.  Internally stores a
    physical-key adjacency mapping so BFS traverses the inter/intra seam
    correctly even when kind labels differ across edges.
    """

    def __init__(self, edges: Iterable[_EdgeInput]) -> None:
        # adj: physical_src_key -> list of (physical_dst_key, canonical PortRef)
        # We keep a canonical PortRef per physical key (first seen) so
        # reaches_any can return PortRef objects.
        self._adj: dict[_PhysKey, list[_PhysKey]] = defaultdict(list)
        self._canonical: dict[_PhysKey, PortRef] = {}

        for edge in edges:
            src, dst = _normalize(edge)
            sk = _phys(src)
            dk = _phys(dst)
            self._adj[sk].append(dk)
            if sk not in self._canonical:
                self._canonical[sk] = src
            if dk not in self._canonical:
                self._canonical[dk] = dst

    def reachable(self, src: PortRef, dst: PortRef) -> bool:
        """Return True if dst is reachable from src by directed BFS.

        Identity is physical (col, row, port, dir); kind is ignored.
        Self-loops are not implied -- src is not considered reachable from
        itself unless there is an explicit self-loop edge.
        """
        sk = _phys(src)
        dk = _phys(dst)
        if sk not in self._adj:
            return False
        visited: set[_PhysKey] = set()
        queue: deque[_PhysKey] = deque(self._adj[sk])
        while queue:
            current = queue.popleft()
            if current == dk:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self._adj.get(current, []))
        return False

    def reaches_any(self, src: PortRef) -> set[PortRef]:
        """Return the set of all PortRefs reachable from src (not including src).

        The returned PortRefs are the canonical first-seen objects from the
        edge list.  Callers comparing identity should use only the physical
        fields (col, row, port, dir).
        """
        sk = _phys(src)
        if sk not in self._adj:
            return set()

        visited: set[_PhysKey] = set()
        queue: deque[_PhysKey] = deque(self._adj[sk])
        result: set[PortRef] = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if current in self._canonical:
                result.add(self._canonical[current])
            queue.extend(self._adj.get(current, []))

        return result
