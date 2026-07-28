# Tile Floor-Plan Schematic -- Design

**Date:** 2026-07-16
**Arc:** Visual debugger, architecture view (follows grid prettification)
**Status:** Approved (design), pending implementation

## Problem

At the closest zoom the architecture-view tile is an anonymous square: a
kind-colored frame, a band color, a centered state code, and a scattering of
small unlabeled circles. Those circles are the tile's *local* stream ports
(Core, DMA, TileCtrl, Cascade, FIFO, Trace), painted by `paint_port` at fixed
interior offsets in `src/visual/tile.rs`. They carry no labels and no legend --
they read as pockmarks. A tile is actually a small system (a core, a banked
memory, a DMA engine, a lock pool, a stream switch), and the current render
throws all that structure away.

## Goal

Replace the anonymous close-up tile with a **floor-plan schematic**: a
kind-aware diagram of the tile's internal components that a reader can learn,
with live per-cycle activity layered on top. "Both, layered" -- a static
structural frame that then lights up.

## Non-Goals (deferred)

- **Motion / flowing-data animation.** Activity is shown as static per-cycle
  highlight (a block tints when active), not animated flow. The animation slice
  (pulse/dot/cadence, and the sync-to-music Easter egg) stays a later arc.
- **Faithful die geometry.** We have no toolchain-provided floor-plan
  coordinates; inventing them would violate the derive-from-the-toolchain rule.
- **The closest grid-zoom tier.** The floor-plan is built as a standalone
  reusable widget and landed in the detail panel first. Reusing it as a new
  closest grid tier is a later slice (see Reuse Path).

## Design Principles

- **Hybrid geometry.** Interior blocks are a *schematic* laid out for
  legibility and kept in the same position every time so the eye learns them.
  Stream ports sit on their **real** N/S/E/W edges, because that connectivity is
  genuine data (`PortType` encodes the edge) and it is what the inter-tile route
  lines already connect to. Nothing invented is presented as hardware truth.
- **AMD's own block grouping.** An AIE-ML (AIE2) tile is three top-level groups:
  the **Core** (scalar + vector datapath, address generators, program memory),
  the **Memory Module** (the memory banks *plus* the DMA engine *plus* the
  locks, as one unit), and the **tile interconnect** (the AXI4-Stream switch =
  the edge ports). DMA and locks live *inside* the Memory Module, not as peers
  of it. Reflecting this grouping makes the schematic recognizable to anyone who
  has read the arch docs. (Ref: AMD AI Engine-ML architecture, arXiv 2509.04162.)
- **Redundant coding (WCAG 1.4.1).** Every block is labeled text *and* color,
  never color alone -- keeps it readable and gives the screen reader real
  content. Reuse the constrained-lightness `Palette` so one fixed text color
  clears AA.
- **Reuse over reinvention.** The LOCKS block is the `lock_map` widget built in
  the previous session, embedded. Edge-port drawing reuses the existing
  `port_offset` / route-color logic.

## Layout

Consistent interior; honest edges. Compute tile:

```
            · · N ports · ·
   +------------+----------------------------+
   |            |  MEMORY MODULE             |
 W |  CORE      |  +----------------------+  | E
   |  [state]   |  | banks / size         |  |
   |  scalar    |  +----------+-----------+  |
   |  vector    |  | DMA      | LOCKS     |  |
   |  + AGUs    |  | ch0 ch1  | [lockmap] |  |
   +------------+--+----------+-----------+--+
            · · S ports · ·   (cascade E<->W)
```

- **Compute tile:** Core (left) + Memory Module group (right, bordered,
  containing banks + DMA + lock-map). Cascade shown E<->W between cores.
- **Mem tile:** no Core; the Memory Module fills the tile; DMA shows its extra
  channels.
- **Shim tile:** a DDR / NoC interface block in place of Core + banks, plus
  DMA and locks.

Directional four-way shared-memory arrows are **not** drawn on compute tiles:
that is the AIE1 model; for AIE2 the emulator's own lock model already shows
compute-tile locks are local-only and cross-tile sharing is a mem-tile feature.
Only the edge stream ports carry directional meaning here.

## Layered Activity (static)

Two layers painted in order:

1. **Structure layer** (always on): labeled block outlines. This is the frame
   that is missing today.
2. **Live layer**: each block tinted by its current state, all sourced from the
   existing snapshot -- no new emu accessors.
   - **Core:** `Palette.band(&TileState)` (running / waiting / done / error),
     matching the grid tile's band so the two views agree.
   - **DMA:** per-channel highlight when the channel is active this cycle
     (`dma[i].state` / active), with `ch0`/`ch1` labels and BD/queue readout.
   - **Locks:** the `lock_map` grid (already colors each lock by value).
   - **Edge ports:** the route moving / stalled / idle colors already used by
     `paint_port` and `draw_routes`.

## Components and Interfaces

New module `src/visual/floorplan.rs`, mirroring `tile.rs` as a reusable atom
that knows nothing about the grid or the inspector:

```rust
pub struct FloorplanPresentation<'a> {
    pub palette: &'a Palette,
    // room to grow (e.g. compact vs full) without touching call sites
}

pub fn floorplan(
    ui: &mut egui::Ui,
    rect: Rect,
    snap: &TileSnapshot,
    state: &TileState,
    ports: &[PortWire],
    pres: &FloorplanPresentation<'_>,
);
```

- Kind comes from `snap.kind` (`TileKindDisplay::{Core,Mem,Shim}`) and selects
  the interior layout.
- Interior block rects are computed from `rect` by a small pure layout function
  (unit-testable without egui: given a rect + kind, return the sub-block rects).
- The `lock_map` helper is shared (moved to `floorplan.rs` or a small shared
  spot) so both call it. The floor-plan's LOCKS block draws the lock-map; the
  existing `detail.rs` locks dropdown **stays** for the full expandable view.
  One helper, two call sites -- no duplication, no removal.

## Data Flow

`detail.rs::show` already has `snap` (`tile_snapshot`), `state`
(`tile_state`), and can get `ports` (`tile_ports`). It currently renders a big
`tile()` at Diagram tier at the top, then value rows. Change: **replace that big
`tile()` call with `floorplan(...)`**; keep the value rows below (core/pc, DMA,
locks, memory, ports) as the precise text readout. All inputs already exist.

## Pockmark Cleanup (grid)

Independently, remove the anonymous **local**-port circles from the grid tile's
Diagram tier in `tile.rs`. They are purely cosmetic: only *external* ports
become route anchors (`port.port_type.is_external().then_some(...)`), so local
dots are never load-bearing. Keep the external edge arrows (they anchor routes).
Result: zoomed-in grid cells go clean immediately, without waiting on the
floor-plan.

## Reuse Path (later)

`floorplan()` takes a `rect`, so the closest grid-zoom tier can later call it
per cell with a small rect -- a new `ZoomLevel`/`DetailTier` beyond `Close`,
with inter-tile routes then connecting real sub-block anchors. No widget change
needed; only the composer grows a zoom level. Out of scope for this arc.

## Testing

- **Pure layout function:** given `(rect, kind)`, assert the sub-block rects are
  inside `rect`, non-overlapping, and that Core is absent for Mem/Shim. No egui.
- **`lock_map`:** existing `lock_color` test carries over.
- **Widget smoke:** `eframe::egui::__run_test_ui` renders `floorplan` for each
  tile kind without panic (same idiom as the `tile.rs` widget tests).
- **Gates:** `cargo build` (default / `--features gui` / `--no-default-features`)
  and `cargo test --lib` all green. GUI never launched (compile-only); the
  `src/debugger` model stays egui-free.

## Human-Only Review (Maya)

Layout legibility, whether the AMD grouping reads as recognizable, block label
clarity, live-highlight salience, and the screen-reader/keyboard pass.
