# Visual Debugger: Grid Prettification (Architecture View) -- Design

**Status:** Design approved, ready for implementation plan.
**Predecessor:** [`2026-07-15-visual-debugger-design.md`](2026-07-15-visual-debugger-design.md) (v1, shipped @d398b307).
**Arc:** First slice of "make each GUI detail level genuinely pretty, one level at a
time." This slice is the **array / architecture view**: turn the flat colored-box
overview grid into a spatial plot of the NPU where each tile is a small readable
diagram, tiles are connected by their real stream routes, and the whole thing is
legible and accessible at a glance.

---

## Goal

Make the overview grid a faithful, at-a-glance, accessible visual plot of the NPU
architecture: type-distinct tiles that show *what they are* and *what they are doing*
via color + text, connected by their actual configured stream routes, with the
routes that are moving data standing out. Readable at both a whole-array glance and
an up-close port-accurate inspection.

## Design principle (the north star for every decision below)

**Appearance encodes state.** A tile should *look like* what it is doing. Every
decision serves "dead obvious at a glance," and every state cue is delivered
through more than one channel so it survives colorblindness, low vision, and
screen readers.

## Scope

In scope for this arc:
- Tile visual language: type color, state color band, always-on text code, loud
  error treatment, per-zoom detail tiers.
- Accessibility as a first-class, from-the-start requirement (not a later pass).
- Numeric axis headers; topology read faithfully from the device model.
- Port-accurate tile diagrams (up close) and the routing-line layer between tiles.
- Two discrete zoom levels (fit-all / close) plus pan.
- Componentization: the tile is a reusable widget, not a part of the inspector.

## Non-goals / explicitly deferred

- **Animation / movement semantics** -- *how* a moving route looks (pulse, travelling
  dot, brightness, cadence vs run-speed). This arc lays the data + salience
  foundation (`cycle_beat`-driven "which routes are hot"); the actual motion design
  is the next arc. Includes a planned **sync-to-music Easter egg** (later, for fun).
- **Cascade dataflow** and **per-lock waiter back-links** -- additive, post-initial-spec.
  The port/status data they hang off already exists; bolting them on costs nothing
  we would redo.
- **Continuous / semantic zoom** -- rejected. Zoom is discrete named levels only.
- **More than two zoom levels** -- the tile widget is built for a *spectrum* of
  detail tiers, so adding stops later is a trivial mapping change, not a redesign.
  Decide the exact count when animation lands.

---

## Architecture: build it as pieces

The tile is the **reusable atom**; everything else composes it. No piece reaches
upward to its container. Each composes the one below.

| Piece | File | Responsibility |
|-------|------|----------------|
| Tile widget | `src/visual/tile.rs` *(new)* | Draw ONE tile at the allocated size + tier. Returns its `Response` (accessible, clickable) and its **port anchor positions**. Knows nothing of grid/inspector/zoom. |
| `DetailTier` | `src/visual/tile.rs` | `Dot \| ColorCode \| LetterCode \| Diagram`. The widget switches rendering on this. Number of zoom stops = how many tiers the zoom control exposes. |
| Route layer | `src/visual/routes.rs` *(new)* | Take collected anchors + active routes, paint lines with activity salience. Consumes *anchors*, not tiles -- indifferent to port-edge vs tile-center anchors. |
| Array view | `src/visual/overview.rs` *(exists -> composer)* | Grid layout, row-flip, axis headers; owns zoom level + pan; sizes each cell; calls the tile widget; collects anchors; calls the route layer. |
| Inspector | `src/visual/detail.rs` *(exists -> consumer)* | Renders the selected tile through the SAME tile widget (big, `Diagram` tier) + non-spatial rows (locks, memory, PC). Its reuse of the widget is the proof the decoupling worked. |
| Theme / palette | `src/visual/theme.rs` *(exists, extend)* | State->band mapping, code strings, single-letter alphabet, glyphs. Pure data, shared. |
| View-model | `src/debugger/model.rs` *(exists, egui-free, tested)* | The snapshots everyone reads. No piece above touches the engine directly. |

**The boundary that matters:** the tile widget takes **data** (a snapshot) +
**presentation** (tier, size, theme, selected) and returns **paint + anchors**,
with no back-reference to its container. That is "not beholden to the inspector,"
concretely.

---

## The tile visual language

### Tile types (three genuinely different objects)

- **Shim** (row 0): DDR / NoC boundary. DMA + stream switch, no core, no local mem.
- **Mem** (row 1): shared SRAM + DMA + locks + stream switch, no core.
- **Compute** (rows 2-5): VLIW core + local mem + DMA + 64 locks + stream switch.

Type is carried by a base color / framing so the three read apart at a glance.

### Two-tier meaning: color for the glance, text for the read

Color is the **pre-attentive** layer -- you see a patch of amber across the array
before reading a character. The text code is the **precise** layer -- you look
closer and read `LOCK` vs `STR`. Color is not decoration; it carries the glance.

### Core state -> color band (grounded in `CoreStatus`, `core_status(col,row)`)

The real enum (do not invent states): `Ready`, `Running`, `WaitingLock{id}`,
`WaitingDma{ch}`, `WaitingStream{port}`, `WaitBank`, `Halted`, `Error`.

| State | Band color | Code | Notes |
|-------|-----------|------|-------|
| not enabled (gated column / no ELF) | grey | (none / `---`) | *Not* a `CoreStatus`; separate axis (enabled?). "Not in the game." |
| `Ready` | pale | `RDY` | armed, not yet launched |
| `Running` | green | `RUN` | the alive/working default |
| `WaitingLock{id}` | amber | `LOCK` | stall; target (lock id) in the label |
| `WaitingDma{ch}` | amber | `DMA` | stall; target (channel) in the label |
| `WaitingStream{port}` | amber | `STR` | stall; target (port) in the label |
| `WaitBank` | amber (transient) | -- | 1-cycle arbitration loss; a flicker, not a steady code |
| `Halted` | blue | `DONE` | clean termination |
| `Error` | red | `ERR` | fault (raises `INSTR_ERROR`); loud treatment (see below) |

All `Waiting*` states share **one flat amber** (top-level "stalling"); the *code*
carries the sub-reason. Amber is common and normal in AIE (cores stall on locks to
synchronize) -- it must not read as an alarm.

### Shim / mem state (no core, so keyed off DMA + stream activity)

Same code+color treatment, driven by DMA/stream activity: `DMA` / `STR` when a
channel is moving, `IDLE` when parked, grey when the column is gated.

### Error is loud (the one exception to "code + color is enough")

`Error` is the red-green colorblind landmine (green `Running` + red `Error`) and the
state where a miss is worst. Belt and suspenders: red fill + `ERR` + a distinct
heavy / hatched border. The `ERR` glyph must be bold, not subtle -- we lean on the
glyph there, not the hue.

### Detail tiers (per-zoom rendering; the widget switches on `DetailTier`)

- `Diagram` -- full port-accurate little diagram + always-on 3-char code.
- `LetterCode` -- single-letter code (see alphabet note) + color.
- `ColorCode` -- 3-char code + color, no port geometry.
- `Dot` -- color only; for `Error`, a bare `!`. Nothing else.

"Always on when readable" is the rule: 3 chars fit cleanly below where the old
in-cell coordinate ID sat; below a readability floor we drop to `LetterCode` then
`Dot`. **The AccessKit label always carries the full state regardless of what is
painted**, so dropping the visible code never loses information for screen readers.

**Single-letter alphabet must be hand-picked (codes collide on first initial):**
`RUN`/`RDY` both want `R`; `DMA`/`DONE`/`Dot`-fallback all pull toward `D`. Design a
deliberate one-letter mapping (the full code set is `RUN`/`RDY`/`LOCK`/`DMA`/`STR`/
`DONE`/`ERR`/`IDLE`); do not auto-derive from the code.

---

## Color and accessibility (first-class, from the start)

### Text contrast: one fixed text color per theme, constrained-lightness bands

**Rule:** within a theme, all state fills live in a shared lightness band chosen so a
**single fixed text color** clears WCAG AA (4.5:1) on every one. No per-tile
black/white swapping. Verified sufficient -- candidate dark-theme band with one
off-white text (`#EDEDED`):

| state | fill | ratio | AA (>=4.5) |
|-------|------|-------|-----------|
| dead grey | `#44464A` | 8.08 | pass |
| run green | `#2E6B3E` | 5.45 | pass |
| wait amber | `#8A5A18` | 5.04 | pass |
| done blue | `#2A4E7A` | 7.26 | pass |
| err red | `#7A2530` | 8.40 | pass |

Amber is the tight constraint (5.04) -- it sets the band's darkness. Light theme
gets its own paired band (pale fills + one dark text). These hexes are *candidates*,
not final; the band *rule* is the spec.

### Themes

- **Light / dark: follow the system**, free and default. eframe feeds the OS theme
  via `raw_input.system_theme`; egui default `ThemePreference::System` resolves it.
  Palette is defined as named tokens redefined per theme -- design both grounds with
  equal care, do not naively invert.
- **High-contrast: a third selectable theme.** egui does **not** detect an OS
  high-contrast / forced-colors mode -- it is ours to provide. Cheap (one more
  token block).

### Colorblindness

The color+code+glyph redundancy is the mechanism -- state is never encoded by color
alone (WCAG 1.4.1). The corner-badge idea folds into this: the *code* is the
redundant channel, and it doubles as the wait-subtype distinguisher, so no separate
badge widget is needed.

### Screen readers (AccessKit) -- in from the start

- AccessKit is a **default** feature of eframe (already compiled in, live).
- **Gotcha:** hand-painted rectangles are invisible to AccessKit unless labeled. So
  each tile is allocated as a real interactive `Response` and given a label:
  `ui.allocate_exact_size(size, Sense::click())` -> `response.widget_info(|| ...)` ->
  paint inside `rect`. This is the idiom that gives us both custom painting *and*
  focus / keyboard / screen-reader for free.
- Label content is the fullest form of the state, e.g. `Core (2,3): waiting on lock 5`.

### egui-fact summary (verified against pinned 0.31.1)

System light/dark: automatic. AccessKit: default-on. OS high-contrast detection:
none (build our own). Custom-painted interactive+accessible tiles: via
`allocate_exact_size` + `widget_info`.

---

## Coordinates and axis labels

- **Numeric on both axes, with labeled headers** ("col" across the top, "row" down
  the side). Numeric matches the rest of the stack (logs, register dumps,
  `XDNA_EMU_WATCH`, aie-rt, detail panel); a debugger's core loop is cross-referencing
  the GUI against those, so chessboard letters would force constant mental
  translation. Headers kill the col-vs-row ambiguity without leaving the toolchain's
  numbering.
- **Drop the in-cell coordinate ID** -- margin headers replace it and free the exact
  slot the state code wants. 5x6 (and NPU5's 8x6) is small enough that the margin
  stays on screen; no sticky per-tile IDs needed. The detail panel names the selected
  tile explicitly.
- **Verify at implementation time -- which numbering.** This hardware relocates the
  partition (declared col 0 -> physical col 1; physical col 0 is the rewritten-
  inaccessible tile). The GUI's load path (`src/loading`) applies the CDO unrelocated,
  so the partition occupies physical columns `[0, column_width)` and `array` /
  `core_status(col,row)` index by that physical col. Axis labels must match whatever
  `array` indexes, or the grid lies. Confirm before painting a single label.

---

## Topology fidelity

- **Render tile kind from the array** (`kind_from_row` / tile data), never from a
  GUI-local "row 0 = shim" assumption. Treat "no tile here" as a first-class empty
  cell.
- The emu currently models a **uniform** topology (shim in every column) -- which is
  the *declared / virtual* topology the toolchain presents and every xclbin is
  compiled against. So it is correct for execution; nothing addressed physical col 0.
- **Known deferred fidelity gap:** physical col 0 has no shim (mem + 4 cores). Not a
  live execution hole (virtualized away). When the device model grows real per-column
  topology, a renderer that reads kind from the array inherits it for free. Earns a
  `known-fidelity-gaps.md` row when touched. Not this arc.

---

## Port-accurate diagrams and the routing-line layer

**Key finding: port geometry is already fully modeled -- zero new emulator state is
needed for port-accurate diagrams or routing lines.** `PortType` (in
`src/device/stream_switch/ports.rs`) encodes the physical placement:

- `North` / `South` / `East` / `West` -> the four inter-tile edges (top / bottom /
  right / left).
- `Dma(n)` / `Core` / `TileCtrl` / `Cascade` / `Fifo` / `Trace` -> local connections
  (interior stubs).
- `direction` (`Master` = sends / `Slave` = receives) -> arrow direction.
- `route_to: Option<(col, row, port)>` -> exactly what to draw the line to.
- `cycle_beat` / `cycle_active` / `cycle_stalled` -> whether it is moving / stalled ->
  line color + (future) animation.

The only new code is a **view-side** layout function `(PortType, index) -> position on
the tile edge/interior`. Pure geometry, not emulator state.

- **Port-accurate up close** (`Diagram` tier); **center-to-center when zoomed out**.
  The route layer consumes anchors either way (port anchors vs tile-center anchors).
- **Salience by activity -- the overview shows flow, not wiring.** A route moving this
  cycle (`cycle_beat`) pops; an idle-but-configured route recedes to a faint hairline
  or toggles off. Only a few things move per cycle, so the picture is sparse by nature
  -- this is the foundation the animation arc builds on, and it is what keeps a 30-tile
  array from becoming a hairball.

---

## Zoom and pan

- **Two discrete zoom levels + pan-when-close.** Fit-all (whole array, center-to-
  center, codes shrink to letters, activity-salient flow) and Close (a region of tiles
  big enough to be port diagrams, port-to-port lines, pannable).
- This is the minimum the two stated desires require: "watch it move across the whole
  architecture" needs fit-all; "port-accurate up close" needs Close. One fixed zoom
  cannot serve both, because port-to-port lines are inherently multi-tile (a single-
  tile inspector can name a route target in text but cannot draw the line).
- **Not continuous / semantic zoom** -- two named levels plus pan. Level count is a
  `DetailTier` mapping, extensible later without touching the widget.

---

## Data model backbone

Everything the visualizer reads, in one place. Legend: **have** (accessor exists),
**partial** (v1 reads some), **new** (small accessor to add, all in `src/debugger/`).

### A. Topology -- static, per load
- Array dims -- `array.cols()` / `array.rows()` -- have
- Tile kind per cell -- `tile_grid(array) -> Vec<TileCell{col,row,kind}>` -- have (v1)
- Per-tile port set -- iterate `Tile.stream_switch` ports -> `(index, direction, port_type)` -- have on the tile; **new**: one clean per-tile port-list accessor for the view
- Column / module gated -- `device::clock_control` (`ModuleKind`), `is_core_enabled(col,row)` -- have for cores; **new**: thin column-level "ungated?" read

### B. Per-tile dynamic state -- per frame
- Core status -- `core_status(col,row) -> CoreStatus` -- have
- Core PC (detail) -- `core_context()?.pc()` -- have
- DMA per channel -- `dma_engine(col,row)` + `current_bd` / `queued_bd` / `channel_count` -- have (Task 2); state + `queue_len` -- partial (v1 `ChannelSnapshot`); MM2S/S2MM direction derivable from channel<->port mapping -- have
- Stream ports -- `cycle_beat` / `cycle_active` / `cycle_stalled` + `route_to` -- have
- Locks -- `Tile.locks[i].value` (64) -- have (core + mem; shim none)
- Local memory (detail) -- `Tile.data_memory()` size + peek -- have

### C. Connection layer -- derived per frame, zero new emu state
- For each `Master` port with `route_to = Some(dst)`:
  `{ src: (col,row,port,edge), dst: (col,row,port), moving: cycle_beat, stalled: cycle_stalled }`
  -> one line. Fully computed from A + B.

### D. Global -- per frame
- `total_cycles` -- have; engine `status` -- have; `run_state` (EngineHost) -- have

### E. Pure view state -- computed in the GUI, never from the emu
- Tile rects; port positions (the `(PortType,index) -> offset` layout fn); zoom tier;
  selection; theme + palette bands; (future) animation interpolation; AccessKit label
  strings (assembled from B).

### New plumbing (all small, all in `src/debugger/`)
1. Per-tile stream-port list accessor -- type + direction + activity + route (extend `model.rs`).
2. Column-gating read -- thin wrapper over `clock_control`.
3. DMA channel direction surfaced explicitly (or derived view-side from the port mapping).

Everything else already has a public accessor. The heavy-looking parts -- port
diagrams, routing lines, watch-it-move -- are **derived**, not new state.

---

## Build / test constraints (carried from v1, still binding)

- **egui-free logic stays in `src/debugger/`** (compiles under `--no-default-features`,
  unit-tested); egui rendering stays in the gui-gated `src/visual/`.
- **GUI gate is compile-only** (`cargo build --features gui`) + code review. **Never
  launch the egui GUI in automated / subagent commands** (pkill on a live GUI crashes
  the graphics driver). Visual click-through is Maya's to run.
- `cargo test --lib` must pass after every change.
- **No emoji anywhere.** Commit messages end with `Generated using Claude Code.`
- Fixture-dependent tests use the skip-if-absent idiom (early return + `eprintln` when
  the built xclbin is missing).

## Success criteria

- Overview renders tiles type-distinct and state-colored, each with an always-on code,
  legible at both fit-all and close zoom, ERR unmissable.
- Every state cue is redundant (color + code + AccessKit label); all fills clear AA
  against the theme's fixed text color; a high-contrast theme is selectable; each tile
  is a focusable, screen-reader-labeled `Response`.
- Numeric axis headers match the emulator's own `(col,row)` indexing (verified).
- Port-accurate diagrams up close; routing lines drawn port-to-port (close) and
  center-to-center (fit-all); routes that are moving stand out from idle ones.
- The tile widget is a standalone piece the inspector merely consumes.
- `cargo test --lib` green; `--features gui` / `--no-default-features` / default all build.
