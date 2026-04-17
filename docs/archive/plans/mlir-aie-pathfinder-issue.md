# GitHub Issue for Xilinx/mlir-aie

File at: https://github.com/Xilinx/mlir-aie/issues/new

## Title

addFixedConnection() only invalidates specific (src, dst) pair, not the full source port

## Labels

bug

## Body

**Problem**: When using `addFixedConnection()` in a two-pass routing scenario (route data flows first, lock them, route additional flows on remaining fabric), the pathfinder can still route packet flows through the same *source port* used by an existing circuit-switched ConnectOp.

`addFixedConnection()` marks only the specific `connectivity[src_port][dst_port]` cell as INVALID. But a circuit-switched ConnectOp monopolizes the entire source port -- no PacketRulesOp can share it. The pathfinder finds a valid route using `(North:2, East:1)` while only `(North:2, South:2)` was invalidated, and the verifier correctly rejects the result:

```
'aie.packet_rules' op packet switched source North2 cannot match another
connect or masterset operation
```

**Expected behavior**: `addFixedConnection()` should mark all `connectivity[i][*]` cells as INVALID for the matched source port, since a circuit ConnectOp prevents any other use of that source port (circuit or packet).

**Use case**: Two-pass routing for non-interfering trace injection -- route test data flows, lock them via `addFixedConnection()`, then route trace packet flows on remaining fabric.

**Proposed fix**: In `addFixedConnection()`, when a ConnectOp source port matches, invalidate the entire row of the connectivity matrix for that source. Additionally, a defensive `continue` on `Connectivity::INVALID` in the Dijkstra relaxation loop would prevent routing through any invalidated edge.
