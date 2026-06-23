# Riallto Visualizer — Visual Vocabulary Reference

A reference distilled from evaluating AMD's [Riallto](https://github.com/AMDResearch/Riallto)
(`~/npu-work/Riallto`) as prior art for the xdna-emu visualization layer.

**Conclusion up front:** Riallto is a beginner on-ramp, frozen since July 2024.
Its visualizer (`npu/utils/appviz.py`, `svg.py`, `svg_config.py`) is
**single-column, design-time, and decorative** — it renders the *intended*
dataflow of a described app, with `repeatCount="indefinite"` SMIL loops standing
in for data movement. There is no execution telemetry, no utilization, no
instruction view, no multi-column array. It is *not* a basis to build on.

What it *does* offer is a clean, considered **visual vocabulary** for drawing an
AIE array. This note captures the parts worth borrowing for the egui hand-roll
and the parts to deliberately discard. Riallto is MIT-licensed; attribute any
lifted conventions.

---

## Borrow: spatial grammar

Source: `svg_config.py`. Geometry is a uniform grid — already parameterized on
columns (`RyzenAiColumn(rows=4, cols=1, ...)`), so the grammar generalizes to
multi-column even though Riallto's app model never uses it.

- Tile placement: `tile_x = x_start + (tile_w + gap) * col`,
  `tile_y = y_start + (tile_h + gap) * row`. Tiles 180×140, gap 20, origin (20,20).
- Column topology, top → bottom:
  **AIE compute tiles (×4) → MemTile → Shim/Interface tile → System-memory bar.**
  Matches real Phoenix column structure.
- **Compute tile = core box + local-memory box, side by side.** Vertically
  adjacent compute tiles draw shared-memory connections (the AIE neighbor-sharing
  model — each core can reach its neighbor's local memory).
- **Interconnect = a switchbox glyph** (40×40) inside each tile with N/S/E/W
  stubs; routes are drawn between adjacent switchboxes. This is the AXI-stream
  switch made visible — the right primitive for showing routing.
- MemTile is taller (extra memory region); Shim/Interface tile draws external
  (south) connections to the system-memory bar.

## Borrow: color conventions

The single best thing to lift.

- **Okabe-Ito colorblind-safe palette** (explicitly commented "for color blind
  people"):

  | Name | Hex | | Name | Hex |
  |---|---|---|---|---|
  | blue | `#56B4E9` | | green | `#009E73` |
  | orange | `#E69F00` | | yellow | `#F0E442` |
  | dark_blue | `#0072B2` | | pink | `#CC79A7` |
  | dark_orange | `#D55E00` | | dark_pink | `#DC267F` |
  | red | `#C00000` | | purple | `#4B0092` |
  | lilac | `#E0C2FF` | | | |

  Lights: light_orange `#F7E2B2`, light_red `#C36D6D`, light_blue `#B2D4E8`,
  light_pink `#E6BBD1`.

- **Tile type encoded by background hue** — instant at-a-glance identification:
  - AIE tile bg `#C2E9FF` (light blue)
  - MemTile bg `#E0FFC2` (light green)
  - Shim/Interface bg `#E0C2FF` (lilac)
  - Interconnect color `#DBBCBD` (muted mauve)
- Per-kernel color cycling drawn from the palette, with a generated legend/key.

## Borrow: motion metaphor

A small circle riding a route path = "a data packet flowing"; direction encoded
by the path geometry (N/S/E/W/diagonal). Good metaphor — keep it, but drive it
from real events (below).

---

## Discard: the engine

The reason xdna-emu exists. Riallto's model is the opposite of what we want.

| Riallto | xdna-emu |
|---|---|
| One-shot SVG emit | egui immediate-mode: re-render each frame from emulator state |
| `repeatCount="indefinite"` decorative loops | particles driven by **real trace/sim events** (timestamps, packet counts) |
| Structure only | **utilization heatmaps, instruction/PC, memory occupancy, lock/contention** |
| Single column (app model) | full **multi-column** array |
| Python emitting SVG/SMIL | Rust drawing each frame from live state |
| — | **fault-injection overlay** (cosmic-ray strike: tile/cell highlight + cascade) |

Borrow the vocabulary; throw away the engine. The grammar (grid, tile=core+mem,
switchbox glyph, memtile/shim, sysmem bar, colorblind palette, type-by-hue,
particle=movement) is a solid visual foundation. The *meaning* — binding every
glyph to live execution state — is the half only xdna-emu is building.

---

## Source map (if revisiting)

- `npu/utils/svg_config.py` — all geometry constants + the color palette.
- `npu/utils/svg.py` — `Tile`/`AieTile`/`MemTile`/`IfTile`/`RyzenAiColumn`
  classes; `draw_ic_connections`, `draw_memory_connections`,
  `add_ic_animation` (the SMIL loops), `SystemMemory`.
- `npu/utils/appviz.py` — `AppViz`: maps app metadata (`tloc`, kernels,
  connections) onto the column. Single-column data model.
