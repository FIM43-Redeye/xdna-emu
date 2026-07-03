---
class: event-broadcast
subsystem: event-broadcast network topology (block-mask routing)
posture: fixed -- the model over-connected the shim row; corrected to route through the fabric as silicon does
status: 1 fixed (2026-07-02, SP-5c pre-flip)
---

# Event-Broadcast Network Gaps

Gaps in the event-broadcast flood topology -- which tiles can broadcast directly
to which neighbours. Surfaced during the #140 timer-sync arc (SP-5c) when the
broadcast network had to carry a *decomposed causal offset*, not merely reach the
right tiles.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| Shim-row tile-to-tile E/W broadcast edge | model gave **every** tile (incl. the shim row) a direct E/W broadcast edge gated only by block masks; real Phoenix has **no functional tile-to-tile shim-row E/W broadcast** -- a shim-sourced horizontal broadcast detours through the fabric (up to memtile row, across, down), cost `~2*d_v + n*d_h`, not `n*d_h`. Proven on silicon: floods block-forced onto the shim row confine to their source column across 3 kernel variants / 60 clean runs, even with **both** shim broadcast switches' E/W blocks cleared. | `src/device/state/effects.rs` (`broadcast_origin_d`, the `is_shim` E/W guard); `src/device/events/broadcast.rs`; [finding: 2026-07-02 shim-row topology](../superpowers/findings/2026-07-02-sp5c-phase2-shim-row-topology.md) | **FIXED (2026-07-02, SP-5c pre-flip).** `broadcast_origin_d` now removes the E/W edges from shim tiles; Dijkstra routes shim->memtile (N) -> across (E/W) -> shim (S), yielding the real `2*d_v + n*d_h` detour automatically. Behavior-neutral while `calibrated=false` (reached set unchanged, all `origin_D` 0); guarded by `broadcast_origin_d_shim_source_detours_ew_through_fabric`. Was the one required emu fix before the flip. |
