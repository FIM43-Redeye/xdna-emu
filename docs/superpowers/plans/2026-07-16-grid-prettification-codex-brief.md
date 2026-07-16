# Grid Prettification (Architecture View) -- Codex Execution Brief

**This is a Codex implementation brief, not a from-scratch spec.** The design is
already decided and approved. Your job is to implement it faithfully. Do not
redesign; where you have a choice, prefer the smallest change that satisfies the
contract below.

## Read first (durable context -- do not re-derive)

- **The design spec (authoritative for rationale + visual language):**
  `docs/superpowers/specs/2026-07-16-visual-debugger-grid-prettification-design.md`
- **Project rules (binding):** `xdna-emu/CLAUDE.md`, `npu-work/CLAUDE.md`.
- **v1 predecessor (the code you are extending):** `src/visual/` (overview.rs,
  detail.rs, theme.rs), `src/debugger/model.rs`, `src/debugger/engine_host.rs`.

## Hard guardrails (violating any of these fails the task)

1. **Never launch the egui GUI.** No `cargo run ... --gui`, no window. The GUI
   gate is **compile-only**: `cargo build --features gui`. Launching it and then
   killing it crashes the graphics driver on this box. Visual verification is the
   human's job, not yours.
2. **egui-free / gui split is load-bearing.** All logic that classifies or
   projects emulator state lives in `src/debugger/` and MUST compile under
   `cargo build --no-default-features` (no egui import there, ever). Only
   rendering lives in `src/visual/` behind the `gui` feature.
3. **Do not invent core states.** The state taxonomy is exactly the `CoreStatus`
   enum (`src/interpreter/core/interpreter.rs`): `Ready`, `Running`,
   `WaitingLock{id}`, `WaitingDma{ch}`, `WaitingStream{port}`, `WaitBank`,
   `Halted`, `Error`. Map these; do not add others.
4. **The palette hexes in the spec are CANDIDATES, not gospel.** Put them in one
   place as clearly-labelled tunable tokens. Do not scatter literal colors through
   the widget code. The human will tune them by eye.
5. **Column numbering: derive labels from the data, do not assume 0..N.** Axis
   headers must be built from the actual `col`/`row` values that `tile_grid`
   reports, so they are correct-by-construction regardless of virtual/physical
   relocation. Do not hardcode a `0..cols` range. (Add a unit test asserting the
   header set equals the distinct cols/rows present.)
6. **No emoji anywhere.** Commit nothing yourself -- the controller commits. Leave
   the tree clean and building.
7. `cargo test --lib` MUST pass. New egui-free logic MUST have unit tests.

## Verified API facts (already checked against the source -- rely on these)

- `src/debugger/model.rs` already has: `TileKindDisplay{Shim,Mem,Core}`,
  `tile_grid(array) -> Vec<TileCell{col,row,kind}>`, `TileSnapshot` (with
  `core_status: Option<String>`, `dma`, `locks`, `mem_*`, and a **flat**
  `ports: Vec<PortSnapshot{label,active,stalled}>` -- this flat port list is the
  gap you replace/augment).
- `StreamPort` (`src/device/stream_switch/ports.rs`) carries everything routing
  needs, **no new emu state required**: `index: u8`, `direction: PortDirection`
  (`Master`=sends/`Slave`=receives), `port_type: PortType`
  (`North|South|East|West|Dma(u8)|Core|TileCtrl|Cascade|Fifo|Trace`),
  `cycle_active: bool`, `cycle_beat: bool` (the PORT_RUNNING "moving this cycle"
  signal), `cycle_stalled: bool`, `route_to: Option<(u8,u8,u8)>` (dst col,row,port).
- `StreamSwitch` (`src/device/stream_switch/mod.rs`) exposes `pub masters:
  Vec<StreamPort>`, `pub slaves: Vec<StreamPort>`; reachable via
  `tile.stream_switch`.
- `PortType` and `PortDirection` are plain egui-free enums -- safe to use inside
  `src/debugger/`.
- `engine.core_status(col,row) -> Option<CoreStatus>`, `is_core_enabled(col,row)`,
  `array.dma_engine(col,row)`, `total_cycles`, `array.cols()/rows()` all exist.

---

## Contracts (implement to these signatures)

### 1. `src/debugger/model.rs` -- new egui-free projections (tested)

**a. A structural state classification -- the single source of truth for
state -> color/code/letter/label.** Replace the "stringify `CoreStatus` with
`{:?}`" approach with a real enum so the widget never parses strings:

```rust
/// What a tile is doing, for display. Egui-free, testable. Derived from
/// CoreStatus for compute tiles, and from DMA/stream activity for shim/mem.
pub enum TileState {
    NotEnabled,     // gated column / no ELF -- "not in the game" (separate axis)
    Ready,          // CoreStatus::Ready
    Running,        // CoreStatus::Running
    WaitLock(u8),   // WaitingLock{id}
    WaitDma(u8),    // WaitingDma{ch}
    WaitStream(u8), // WaitingStream{port}
    WaitBank,       // transient
    Done,           // Halted
    Error,          // Error
    Dma,            // shim/mem: a channel is moving
    Stream,         // shim/mem: a port is moving
    Idle,           // shim/mem: parked
}

impl TileState {
    pub fn code(&self) -> &'static str;   // RUN/RDY/LOCK/DMA/STR/DONE/ERR/IDLE/---
    pub fn letter(&self) -> char;         // hand-picked, NOT first-initial (see note)
    pub fn is_error(&self) -> bool;
}

/// Full screen-reader sentence, e.g. "Core (2,3): waiting on lock 5".
pub fn accessible_label(col: u8, row: u8, kind: TileKindDisplay, st: &TileState) -> String;

/// Classify one tile for display.
pub fn tile_state(engine: &InterpreterEngine, col: u8, row: u8) -> TileState;
```

Single-letter alphabet (codes collide on first initial -- `RUN`/`RDY` -> R,
`DMA`/`DONE` -> D): pick a deliberate, distinct letter per code. Document the
mapping in one comment. Suggested (change if you find better): R=Running,
Y=Ready, L=WaitLock, D=WaitDma, S=WaitStream, O=Done(dOne), E=Error, I=Idle,
M=Dma(Move), T=Stream. Any injective mapping is acceptable; ERR should stay
visually loud regardless.

**b. Rich per-tile port list (replaces the flat `PortSnapshot` for the view).**

```rust
pub struct PortWire {
    pub index: u8,
    pub direction: PortDirection,     // arrow direction
    pub port_type: PortType,          // physical edge (N/S/E/W) or local stub
    pub moving: bool,                 // cycle_beat
    pub active: bool,                 // cycle_active
    pub stalled: bool,                // cycle_stalled
    pub route_to: Option<(u8,u8,u8)>, // dst (col,row,port_index); Master ports only
}

/// Every stream port on one tile, masters then slaves.
pub fn tile_ports(array: &TileArray, col: u8, row: u8) -> Vec<PortWire>;
```

**c. Column-gating read (thin wrapper).**

```rust
/// True if the column is participating (any enabled core / active tile).
pub fn is_column_enabled(engine: &InterpreterEngine, col: u8) -> bool;
```

DMA direction (MM2S/S2MM) is **derivable view-side** from the channel<->port
mapping; do not add new emu state for it. Surface it only if trivial.

Tests (egui-free, in `model.rs`): `tile_state` returns the right variant/code for
a known kernel's cores; `tile_ports` returns a non-empty port list with correct
`port_type`/`direction` for a compute tile; `letter()` is injective across all
`TileState` codes; `is_column_enabled` true for an active column, false for a
gated one.

### 2. `src/visual/theme.rs` -- a `Palette` struct with per-theme instances

Turn the loose consts into a struct so themes are swappable and the fixed-text-
color rule is enforced structurally:

```rust
pub struct Palette {
    pub bg: Color32,
    pub text: Color32,          // ONE fixed text color; every band must clear AA against it
    pub kind_shim: Color32, pub kind_mem: Color32, pub kind_core: Color32,
    pub band_grey: Color32, pub band_pale: Color32, pub band_green: Color32,
    pub band_amber: Color32, pub band_blue: Color32, pub band_red: Color32,
    pub selected: Color32,
    pub route_moving: Color32, pub route_idle: Color32, pub route_stalled: Color32,
    pub err_border: Color32,
}
impl Palette {
    pub fn dark() -> Self;          // seed band hexes from the spec's verified table
    pub fn light() -> Self;         // paired band: pale fills + one dark text
    pub fn high_contrast() -> Self; // our own; egui has no OS high-contrast detection
    pub fn band(&self, st: &TileState) -> Color32; // state -> band color
}
```

Comment above the hexes: `// Candidate band; tuned by eye. AA-verified against
`text` -- keep any change above 4.5:1.` Theme selection follows the system for
light/dark (egui default `ThemePreference::System`); high-contrast is a manual
override the app exposes.

### 3. `src/visual/tile.rs` (new) -- the reusable tile atom

Container-agnostic: takes a target `Rect` + data + presentation, does its own
accessible interaction + painting, returns its `Response` and its **port
anchors** for the route layer. Knows nothing of grid/inspector/zoom.

```rust
pub enum DetailTier { Dot, ColorCode, LetterCode, Diagram }

pub struct TilePresentation<'a> {
    pub tier: DetailTier,
    pub selected: bool,
    pub palette: &'a Palette,
}

pub struct PortAnchor {
    pub index: u8,
    pub port_type: PortType,
    pub direction: PortDirection,
    pub pos: egui::Pos2,   // absolute, on the tile edge (Diagram) or tile center (coarse tiers)
}

pub struct TileOutput {
    pub response: egui::Response,
    pub anchors: Vec<PortAnchor>,   // empty below Diagram tier except a single center anchor
}

pub fn tile(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    snap: &TileSnapshot,      // carries col,row,kind
    state: &TileState,
    ports: &[PortWire],
    pres: &TilePresentation,
) -> TileOutput;
```

Requirements inside `tile()`:
- **Accessibility idiom (mandatory):** obtain an interactive response for `rect`
  via `ui.interact(rect, id, Sense::click())` (a stable id from `(col,row)`), then
  `response.widget_info(|| WidgetInfo::labeled(Role::Button, true,
  accessible_label(...)))`. Hand-painted rects are invisible to AccessKit without
  this. The **label always carries the full state** even when the painted code is
  dropped to a letter/dot.
- **Tier rendering:** `Diagram` = band fill + always-on 3-char code + port stubs
  laid out by a pure `(PortType,index) -> edge offset` helper (N=top, S=bottom,
  E=right, W=left, locals = interior), anchors returned per external port.
  `ColorCode` = band + 3-char code, center anchor only. `LetterCode` = band +
  single letter. `Dot` = band only; `!` for Error.
- **Error is loud** at every tier: `err_border` heavy/hatched stroke + `ERR`/`!`;
  lean on the glyph, not the hue (red-green colorblind landmine).
- **Not-enabled** tiles: `band_grey`, muted; distinct from any active state.
- Base type framing (`kind_*`) distinguishes shim/mem/core.

The `(PortType,index) -> offset` layout helper is pure geometry -- put it in
`tile.rs` (or a small `layout` fn); it is view state, not emu state.

### 4. `src/visual/routes.rs` (new) -- the routing-line layer

```rust
pub struct RouteLine { pub from: egui::Pos2, pub to: egui::Pos2,
                       pub moving: bool, pub stalled: bool }

pub fn draw_routes(painter: &egui::Painter, routes: &[RouteLine], palette: &Palette);
```

Salience by activity: `moving` (from `cycle_beat`) draws bright (`route_moving`);
idle-but-configured draws a faint hairline (`route_idle`) or is omitted;
`stalled` uses `route_stalled`. Consumes anchors -- indifferent to whether they
are port-edge (close) or tile-center (fit-all) anchors.

### 5. `src/visual/overview.rs` -- becomes the composer

Owns grid layout (keep the existing row-flip so shim row 0 sits at the bottom),
**numeric axis headers** ("col" across top, "row" down side; labels from the
actual `tile_grid` col/row values), the **two zoom levels + pan**, per-cell
sizing, and orchestration:
1. pick `DetailTier` from current zoom/cell size (fit-all -> coarse tier; close ->
   `Diagram`);
2. for each cell compute its `Rect` (apply pan offset), call `tile(...)`, handle
   click-select, collect `TileOutput.anchors`;
3. build `RouteLine`s: for each tile's `PortWire` with `direction==Master &&
   route_to==Some(dst)`, resolve `from` = this port's anchor, `to` = the dst
   tile's matching-index anchor (fall back to dst tile center if not at Diagram
   tier), tag `moving`/`stalled`;
4. call `draw_routes(...)`.

Zoom = two named levels (Fit / Close) + pan when Close. Not continuous. Level
count is just a `DetailTier` mapping -- keep it a small match so more stops are a
one-line change later.

### 6. `src/visual/detail.rs` -- becomes a consumer (proves decoupling)

Render the selected tile through the **same `tile()` widget** at a large `Rect`,
`Diagram` tier, then keep the existing non-spatial rows (core status+PC, DMA
channels, 64 locks, memory peek). Its reuse of the widget with zero grid/zoom
knowledge is the proof the atom is not beholden to the inspector.

---

## Task order (each step builds + `cargo test --lib` green before the next)

1. `model.rs`: `TileState` + `code/letter/is_error` + `accessible_label` +
   `tile_state` + `PortWire` + `tile_ports` + `is_column_enabled`, with unit
   tests. (egui-free; must build under `--no-default-features`.)
2. `theme.rs`: `Palette` struct + `dark()/light()/high_contrast()` + `band()`.
3. `tile.rs`: `DetailTier`, `TilePresentation`, `PortAnchor`, `TileOutput`,
   `tile()`, and the port-offset layout helper.
4. `routes.rs`: `RouteLine` + `draw_routes`.
5. `overview.rs`: rewrite as composer (headers, zoom+pan, tile calls, route build).
6. `detail.rs`: consume `tile()` for the selected-tile diagram.
7. Wire module decls (`src/visual/mod.rs`), ensure all three build configs pass.

## Acceptance you can verify yourself (close every one before reporting done)

- [ ] `cargo build` (default) -- clean.
- [ ] `cargo build --features gui` -- clean.
- [ ] `cargo build --no-default-features` -- clean (proves `src/debugger/` stays
      egui-free).
- [ ] `cargo test --lib` -- all green, including your new `model.rs` tests.
- [ ] No egui import anywhere under `src/debugger/`.
- [ ] Every state cue is redundant in code: `band()` (color) + `code()`/`letter()`
      (text) + `accessible_label()` (screen reader). No state encoded by color
      alone.
- [ ] Axis headers derived from `tile_grid` values, not a hardcoded range (a test
      asserts this).
- [ ] No emoji; tree clean (you do not commit).

## What only the human can verify (do NOT block on these; note them for review)

- Whether the palette actually *looks* right / pretty (hexes are candidates).
- Whether the array "reads" at a glance and moving routes pop.
- Live screen-reader / keyboard feel.

Leave a short report: what you implemented per contract, any contract point you
had to bend and why, and anything you flagged for the human's eye.
