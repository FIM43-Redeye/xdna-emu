# Clock-control subsystem design

**Date**: 2026-05-24
**Status**: Approved (Maya, 2026-05-24)
**Coverage gap closed**: `clock_control: STUB` in `docs/coverage/aie2/implementation-gaps.md`

## Problem

`docs/coverage/aie2/implementation-gaps.md` lists `clock_control: STUB`.
No code in `src/` handles `Module_Clock_Control`, `Column_Clock_Control`,
or the adaptive clock-gate registers. Writes are accepted by the generic
register dispatcher and discarded; tiles always execute regardless of
what the CDO programs.

The gap has two costs:

1. **Correctness**: a binary that gates a tile and then probes its state
   gets nonsense from the emulator (live state) where silicon would give
   undefined results. The emulator silently masks a bug pattern.
2. **Performance**: the emulator steps every tile every cycle, even tiles
   the program has not requested clocked. Real programs typically use
   one or two of five columns, so this is a meaningful waste.

## Hardware reality

Authoritative source: comment on `XAie_PmRequestTiles` in
`aie-rt/driver/src/pm/xaie_clock.c`:

> *"As all the tiles are gated when the system boots, this function needs
> to be called after device instance is initialized and before any other
> AI engine operations. Otherwise, the other AI engine functions may
> access gated tiles."*

Three points fall out of this:

1. **Default boot state is fully gated.** The CDO/PDI configuration
   phase explicitly enables clocks before any kernel runs.
2. **Accesses to gated tiles are not blocked by silicon.** The wording
   "may access gated tiles" is deliberate: silicon does not NACK, return
   sticky-zero, or fault. Accesses happen, but produce undefined
   results.
3. **Clock-control registers are privileged.** The AM025 description
   says "Control clock gating of modules (privileged)". Only management
   firmware (via CDO) writes them; user kernels cannot.

### Three tiers of gating

Derived from `aie_registers_aie2.json` and `aie-rt/driver/src/pm/`:

| Tier | Register | Granularity | Typical writer |
|------|----------|-------------|----------------|
| Column | `Column_Clock_Control` | Whole column on/off | Driver at program load |
| Module | `Module_Clock_Control` (compute, memtile); `Module_Clock_Control_0`/`_1` (shim) | Per-module within a tile (core / memory / DMA / SS) | CDO / driver |
| Adaptive | `DMA_Adaptive_Clock_Gate` and `Stream_Switch_Adaptive_Clock_Gate` bits in `Module_Clock_Control`; period in `Stream_Switch_Adaptive_Clock_Gate_Abort_Period` | DMA / SS auto-gate after N idle cycles | Configured by driver, engages autonomously |

Module-control register reset values: compute `0x37`, memtile `0x33`,
shim `_0` `0x3B`. The adaptive-gate bits default to enabled (`1`) in
all of them.

## Decisions

### Default boot state: silicon-accurate (all gated)

Silicon's default is "all gated, ungate via CDO" and the emulator
adopts that literally. A freshly-constructed `ClockController` reports
every column and every module as gated. Execution of a tile does not
begin until the CDO (or a test helper acting as one) programs its
clocks active.

Boot-state behavior is consistent across two layers:

- **Execution layer**: all column and module gates report inactive on
  query. Step paths skip every tile.
- **Register layer**: a read from `Module_Clock_Control` before any
  write returns the AM025 reset value (compute 0x37 / memtile 0x33 /
  shim `_0` 0x3B). These reset bits include enable bits for individual
  modules; the controller decodes them on first read so the execution
  layer agrees with what the register would say.

The "boot active" pragmatic option was considered and rejected: the
project's correctness principle (see `CLAUDE.md`) is "match real
hardware, not invent workarounds." A test that does not configure
clocks is, in fact, broken on real silicon; the emulator should not
hide that.

### Test-surface migration

Existing tests that construct `Device` / `Array` / `Tile` directly
(without loading a real xclbin/CDO) will see every tile gated and
will need to ungate before the tile can do work. Two mechanisms:

- **`ClockController::ungate_all()`** — test helper that internally
  calls `write_register` for every `(col, row)` with the all-active
  bit pattern. Exercises the real register-write path (no test-only
  backdoor that could hide bugs).
- **xclbin-loading tests** (`xclbin_suite`, bridge tests): no change
  required — the loaded CDO already programs clocks. These tests now
  exercise the silicon-accurate path naturally.

Migration cost: one-line addition per direct-construction test site.
The implementation plan will audit and quantify before changes land.

**Bridge-test risk**: if any bridge-test xclbin does not program
clock-control writes (relied historically on the emulator's no-gating
behavior even though silicon needs explicit configuration), it will
break. Treat as a bug exposed by the change, not a regression; address
case by case.

### Access-to-gated-tile policy: serve and warn

Silicon does not block accesses to gated tiles; they produce undefined
results. We model the silicon behavior (access proceeds, returns
current state) but add a warning log to surface the bug pattern that
silicon hides. This makes the emulator *more helpful than silicon*
without making it more restrictive.

`XDNA_EMU_WARN_GATED_ACCESS=0` silences the warning if a test
intentionally exercises gated-access behavior.

### Privilege enforcement: not in v1

Silicon refuses clock-control writes from user-kernel contexts. The
emulator has no privilege model today; adding one for this one feature
is scope creep. v1 accepts writes from any context.

### NoC interaction: separate gap

`noc: STUB` is a separate coverage entry. Clock-control does not
depend on NoC; NoC implementation can proceed independently later.

## Architecture

**Approach A** (selected): dedicated `ClockController` subsystem,
single source of truth. Step paths query it. Mirrors how
`perf_counters/`, `trace_unit/`, and `stream_switch/` are structured.

Approach B (embedded per-tile state, scattered checks) was rejected
for fragmentation risk: every new step site would need to remember
the gate check. Approach C (hybrid cached) was rejected as premature
optimization. If A's lookup cost ever measures as meaningful, B is
the documented fallback.

### Module layout

```
src/device/clock_control/
+- mod.rs              # ClockController, public API, all v1 logic
+- tests.rs            # unit tests
```

Single-file v1. Split into `column_gate.rs` / `module_gate.rs` /
`adaptive_gate.rs` only if `mod.rs` grows past ~400 LoC.

### Types

```rust
pub struct ClockController {
    columns: Vec<bool>,                          // per-column gate; index = col
    tiles: HashMap<(u8, u8), TileGates>,         // module gates per tile
    adaptive: HashMap<(u8, u8), AdaptiveState>,  // idle counters + abort_period
}

pub enum ModuleKind { Core, Memory, Dma, StreamSwitch }

struct TileGates {
    // Bit-field-decoded module enable bits; layout per tile type.
    // Backed by the actual register value so reads return what was written.
    raw_mcc_0: u32,                              // Module_Clock_Control or _0
    raw_mcc_1: Option<u32>,                      // shim _1 only
}

struct AdaptiveState {
    dma_idle_cycles: u32,
    ss_idle_cycles: u32,
    abort_period_2pow: u8,  // engaged when idle >= 2^abort_period
}
```

Bit-field decoding is driven from the `aie_registers_aie2.json` regdb,
not hardcoded constants. This follows the `bd.rs` pattern.

### Public API

```rust
impl ClockController {
    pub fn is_column_active(&self, col: u8) -> bool;
    pub fn is_module_active(&self, col: u8, row: u8, kind: ModuleKind) -> bool;
    pub fn is_tile_active(&self, col: u8, row: u8) -> bool;  // any module
    pub fn write_register(&mut self, col: u8, row: u8, offset: u32, value: u32);
    pub fn read_register(&self, col: u8, row: u8, offset: u32) -> Option<u32>;
    pub fn tick_adaptive(&mut self, col: u8, row: u8,
                          dma_active: bool, ss_active: bool);

    /// Test helper: ungate every column and every module by writing
    /// all-active patterns through `write_register`.  No test-only
    /// backdoor -- this exercises the same register path the CDO uses.
    pub fn ungate_all(&mut self);
}
```

`tick_adaptive` is called after each per-cycle step with whether DMA
and SS did any work that cycle. Activity resets the matching idle
counter; idleness increments it; reaching `2^abort_period` engages
the gate.

### Ownership

`ClockController` is owned by `Device` (alongside the other
subsystem-level state). Reference passed mutably into the step
functions that need it.

## Data flow

### Register write (CDO config phase)

```
CDO config → existing register_dispatcher
  recognize clock_control offset range
  -> ClockController.write_register(col, row, offset, value)
     update columns / tiles / adaptive state from value
```

### Step path (per cycle)

```
array/routing.rs::step_data_movement(col):
    if !clock.is_column_active(col): return        // top-tier skip
    for each tile in column:
        if clock.is_module_active(col, row, Dma):
            dma_active = step_dma(...).is_some()
        if clock.is_module_active(col, row, StreamSwitch):
            ss_active = step_tile_switches(...).is_some()
        if clock.is_module_active(col, row, Core):
            step_core(...)
        clock.tick_adaptive(col, row, dma_active, ss_active)
```

### Access-to-gated-tile

```
register/memory access path:
    if !clock.is_tile_active(col, row):
        warn_once_per_site("access to gated tile (col, row) at offset")
    proceed with access normally  // serve the value
```

`warn_once_per_site` dedups by the tuple `(col, row, offset)` so a
single bad access doesn't flood the log when a kernel hits it
repeatedly. Reset on next `ClockController` construction (i.e.,
per-test, per-run).

## Edge cases

| Case | Behavior |
|------|----------|
| First read from a clock-control register before any write | Return AM025 reset value (compute 0x37 / memtile 0x33 / shim_0 0x3B). The execution-layer "all gated at boot" default decodes this same reset value, so the two views are consistent. |
| Write to a clock-control register on a tile that does not exist for this arch | No-op + warn. (Same as how other invalid-tile accesses behave today.) |
| Adaptive gate engages during an in-flight DMA transfer | Cannot happen: an in-flight transfer is by definition active, so the idle counter is being reset. If the kernel programs an extremely short `abort_period` while DMA is paused, behavior is undefined per hardware. |
| Re-ungate after gating | Idle counters reset to 0 on ungate. Adaptive state starts fresh. |
| Write to gated tile's general (non-clock-control) registers | Warning emitted; write lands. Reads similarly. |

## Testing

### Unit (`src/device/clock_control/tests.rs`)

- Register write/read round-trip on each tile type
- Column gate: write to disable -> `is_column_active(col)` returns false
- Column gate: write to enable -> returns true
- Module bit-field decode: write specific bit patterns; verify each
  `ModuleKind` flag matches the field
- Reset value returned on first read pre-write
- Adaptive: tick with `dma_active=true` resets counter; tick with
  `false` increments; counter reaching `2^abort_period` engages gate
- Adaptive: gate engagement does not occur if abort_period is 0
  (silicon edge case: spec says supported range 3-12)

### Integration (`xclbin_suite`)

- Hand-rolled CDO: gate column N, run a kernel targeting column N,
  verify no output produced (DMA stays in initial state, core does not
  advance PC)
- Hand-rolled CDO: ungate after gating, verify kernel produces correct
  output
- Mixed: gate column A, leave column B ungated; verify column B runs
  normally while column A is dark

### Bridge test

Identify a real mlir-aie test that exercises clock-control during the
implementation-plan phase. Likely candidates: anything in
`mlir-aie/test/npu-xrt/` whose generated CDO contains
`Module_Clock_Control` writes (greppable). If no good candidate
exists, the integration tests above stand alone.

## Out of scope

- NoC implementation (separate STUB; clock-control does not depend on it)
- Privilege enforcement of writes
- GUI/visualizer integration (future once basic implementation lands)
- File-split into column/module/adaptive submodules (only if needed)
