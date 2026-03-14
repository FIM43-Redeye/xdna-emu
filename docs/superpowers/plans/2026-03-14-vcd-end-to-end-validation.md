# VCD End-to-End Validation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the deep VCD comparison pipeline fully functional end-to-end: validate mapping accuracy against real aiesimulator VCDs, add VC2802 device geometry so both emulator and aiesimulator share coordinate space, and integrate into the bridge test suite.

**Architecture:** The deep VCD module (`src/vcd/`) already has all components built (StatePath, mapping tree, extraction, comparison, tolerance, reporting, emission, CLI binary). This plan validates and fixes the pipeline against real data, adds VC2802 device support to eliminate coordinate translation, and wires the comparison into automated testing.

**Tech Stack:** Rust, wellen (VCD reading), vcd (VCD writing, feature-gated), serde_json (reports)

**Prior plan:** `docs/superpowers/plans/2026-03-12-vcd-deep-extraction.md` (produced the current `src/vcd/` module)

---

## File Structure

| File | Responsibility | Phase |
|------|---------------|-------|
| **`src/vcd/mapping.rs`** | Fix `build_aie2_mapping_tree()` to use VC2802 geometry when in aiesim validation mode | A+B |
| **`src/vcd/coverage.rs`** | No changes expected (already complete) | - |
| **`src/vcd/compare.rs`** | Fix any bugs found during end-to-end validation | A |
| **`src/vcd/report.rs`** | No changes expected | - |
| **`src/vcd/tolerance.rs`** | Tune tolerance bands based on real comparison data | A |
| **`src/bin/vcd_compare.rs`** | Add `--device` flag to select mapping tree geometry | B |
| **`tools/aie-device-models.json`** | Add `xcve2802` device entry | B |
| **`src/device/arch_config.rs`** | Add `ModelConfig::xcve2802()` constructor | B |
| **`src/device/array.rs`** | Bump `MAX_COLS` to accommodate VC2802 (17 cols) | B |
| **`scripts/emu-bridge-test.sh`** | Wire `--aiesim` flag to run VCD comparison after aiesim | C |

---

## Phase A: Validate and Fix Against Real VCDs

Real aiesim VCD files exist at:
- `build/unit_tests/03_simple/--simulation-cycle-timeout.vcd`
- `build/unit_tests/08_tile_locks/--simulation-cycle-timeout.vcd`
- `build/unit_tests/01_precompiled_core_function/--simulation-cycle-timeout.vcd`
- `build/unit_tests/03_cascade_core_functions/--simulation-cycle-timeout.vcd`

These are VC2802 VCDs (38x11 array, AIE2 architecture). The current
`build_aie2_mapping_tree()` uses NPU1 geometry (4 cols, 6 rows). Phase A
will first run the coverage audit to see what maps, then fix any mismatches.

### Task 1: Run coverage audit against real VCD and capture baseline

**Files:**
- Read: `src/vcd/coverage.rs`, `src/vcd/mapping.rs`
- Read: `build/unit_tests/08_tile_locks/--simulation-cycle-timeout.vcd` (via tool)

- [ ] **Step 1: Build vcd_compare binary**

```bash
cd /home/triple/npu-work/xdna-emu
cargo build --release --bin vcd_compare
```

Expected: Compiles successfully. If there are compile errors, fix them first.

- [ ] **Step 2: Run coverage audit on 08_tile_locks VCD**

```bash
./target/release/vcd_compare --coverage build/unit_tests/08_tile_locks/--simulation-cycle-timeout.vcd \
  -o /tmp/claude-1000/vcd-coverage-08.txt
```

Expected: A coverage report showing mapped vs unmapped signals, broken down by
subsystem and prefix. This is our baseline. Save the output.

- [ ] **Step 3: Run coverage audit on 03_simple VCD**

```bash
./target/release/vcd_compare --coverage build/unit_tests/03_simple/--simulation-cycle-timeout.vcd \
  -o /tmp/claude-1000/vcd-coverage-03.txt
```

Expected: Same format. Compare with 08_tile_locks to understand signal variation.

- [ ] **Step 4: Analyze coverage gaps**

Read both coverage reports. Identify:
1. Which subsystems have good mapping coverage (locks, DMA, core, streams)
2. Which signals are unmapped and why (wrong tile coordinates? wrong signal names? missing subsystem?)
3. The VC2802 vs NPU1 coordinate mismatch: tile_7_3 in VCD vs tile_0_2 in mapping tree

Document findings as comments in the code or as a summary for the next task.

### Task 2: Fix mapping tree geometry for VC2802 VCDs

**Files:**
- Modify: `src/vcd/mapping.rs:717-764` (`build_aie2_mapping_tree()`)

The current tree uses NPU1 coordinates (cols 0-3, rows 0-5). Real aiesim
VCDs use VC2802 coordinates (up to col 37, rows 0-10). The mapping tree must
cover the tiles that actually appear in the VCD.

- [ ] **Step 1: Write a test that calls the not-yet-existing function**

In `src/vcd/mapping.rs` tests:

```rust
#[test]
fn vc2802_tree_resolves_compute_tile() {
    // 08_tile_locks uses core at tile (7,3) -- VC2802 coordinates.
    let tree = build_vc2802_mapping_tree();
    let segments = ["top", "math_engine", "array", "tile_7_3", "cm", "pc_E1"];
    let result = tree.resolve(&segments);
    assert!(result.is_some(), "Expected to resolve tile_7_3 pc_E1");
    assert_eq!(result.unwrap(), StatePath::CorePc { col: 7, row: 3, stage: 1 });
}
```

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib vcd::mapping::tests::vc2802_tree_resolves_compute_tile`
Expected: Compile error (`build_vc2802_mapping_tree` not found) -- TDD red

- [ ] **Step 2: Add `build_vc2802_mapping_tree()` function**

Create a new tree builder function that covers VC2802 geometry. VC2802 has:
- 38 columns (0-37), but only some are SHIMNOC (even cols 2,3,6,7,...)
- 11 rows: row 0 = shim, rows 1-2 = memtile, rows 3-10 = compute

For practical mapping, we only need tiles that aiesimulator actually uses.
The VCD signal hierarchy reveals which tiles are present. Start with a
generous range covering the test cases we have:

```rust
/// Build mapping tree for VC2802 geometry (aiesimulator validation mode).
///
/// VC2802 is a 38-column x 11-row AIE2 array used by aiesimulator for all
/// AIE2 targets. The mapping covers all possible tile coordinates that
/// aiesimulator might output.
///
/// Column range: 0-37 (38 columns)
/// Row 0: shim tiles (SHIMPL or SHIMNOC depending on column)
/// Rows 1-2: mem tiles
/// Rows 3-10: compute tiles
pub fn build_vc2802_mapping_tree() -> MappingTree {
    use super::core_mapping::core_mapping;
    use super::dma_mapping::{dma_mapping, shim_dma_mapping};
    use super::event_mapping::event_mapping;
    use super::lock_mapping::lock_mapping;
    use super::stream_mapping::{
        compute_stream_mapping, memtile_stream_mapping, shim_stream_mapping,
    };

    // VC2802 dimensions from aiesimulator device model.
    let cols: Vec<u8> = (0..38).collect();
    let rows_memtile: Vec<u8> = vec![1, 2];
    let rows_compute: Vec<u8> = (3..11).collect();

    let shim_tiles: Vec<(u8, u8)> = cols.iter().map(|&c| (c, 0)).collect();
    let mem_tiles: Vec<(u8, u8)> = cols
        .iter()
        .flat_map(|&c| rows_memtile.iter().map(move |&r| (c, r)))
        .collect();
    let compute_tiles: Vec<(u8, u8)> = cols
        .iter()
        .flat_map(|&c| rows_compute.iter().map(move |&r| (c, r)))
        .collect();

    MappingTree::builder()
        .scope("top")
        .scope("math_engine")
        .tile_group("shim", &shim_tiles)
            .mapping(lock_mapping(16))
            .mapping(shim_dma_mapping(2, 2))
            .mapping(shim_stream_mapping())
            .mapping(event_mapping())
            .done_tile_group()
        .tile_group("mem_row", &mem_tiles)
            .mapping(lock_mapping(64))
            .mapping(dma_mapping(6, 6))
            .mapping(memtile_stream_mapping())
            .mapping(event_mapping())
            .done_tile_group()
        .tile_group("array", &compute_tiles)
            .mapping(lock_mapping(16))
            .mapping(NestedScopeMapping::new("mm", Box::new(dma_mapping(2, 2))))
            .mapping(compute_stream_mapping())
            .mapping(core_mapping())
            .mapping(event_mapping())
            .done_tile_group()
        .build()
}
```

- [ ] **Step 3: Run the test again**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib vcd::mapping::tests::vc2802_tree_resolves_compute_tile`
Expected: PASS -- TDD green

- [ ] **Step 4: Run coverage audit with VC2802 tree**

Update `vcd_compare` to accept a `--device vc2802` flag (or temporarily
modify `build_aie2_mapping_tree()` to use VC2802 ranges) and re-run the
coverage audit. Compare mapped signal counts with the Task 1 baseline.

Expected: Significantly more mapped signals (tiles at VC2802 coordinates now resolve).

- [ ] **Step 5: Commit**

```bash
git add src/vcd/mapping.rs
git commit -m "feat(vcd): add VC2802 mapping tree for aiesim validation"
```

### Task 3: Fix signal naming mismatches discovered by coverage audit

**Files:**
- Modify: whichever `*_mapping.rs` files have naming mismatches
- Test: unit tests in each mapping module

This task depends on the coverage audit results from Tasks 1-2. Common
issues to expect:

1. **Scope hierarchy differences**: aiesimulator may use slightly different
   scope nesting than our mapping tree expects. For example, the VCD might
   use `tl.aie_logical.aie_xtlm.math_engine` as a prefix, while our tree
   starts at `top.math_engine`.

2. **Signal name variations**: DMA signals might use different names in the
   actual VCD vs what we mapped from documentation.

3. **Missing `from_` prefix scopes**: The stream switch mapping expects
   `stream_switch.from_sSouth3.data` but the VCD might use a different
   scope structure.

- [ ] **Step 1: For each unmapped signal group, identify the naming pattern**

Read the VCD header to see actual signal names. Compare with what the
mapping tree expects. Fix each mapping function.

- [ ] **Step 2: Write regression tests for each fixed pattern**

For each naming fix, add a test that resolves the corrected signal name.

- [ ] **Step 3: Re-run coverage audit to verify improvement**

Expected: Mapped percentage should increase with each fix. Target: >80%
of signals in active tiles should map.

- [ ] **Step 4: Commit**

```bash
git add src/vcd/*.rs
git commit -m "fix(vcd): align signal names with real aiesim VCD output"
```

### Task 4: End-to-end comparison with synthetic emulator VCD

**Files:**
- Modify: `src/vcd/compare.rs` (if bugs found)
- Test: add integration test

Since we do not yet have VCD emission from the emulator (feature-gated),
create a synthetic emulator VCD from the aiesim VCD itself to prove the
comparison pipeline works end-to-end. Take the aiesim VCD, re-label it as
the "emulator" source, and compare it to itself. Every signal should be
ExactMatch.

- [ ] **Step 1: Write the self-comparison test**

```rust
#[test]
#[ignore] // requires real VCD file
fn self_comparison_all_exact_match() {
    let vcd_path = "build/unit_tests/08_tile_locks/--simulation-cycle-timeout.vcd";
    if !std::path::Path::new(vcd_path).exists() {
        eprintln!("Skipping: {} not found", vcd_path);
        return;
    }
    let tree = build_vc2802_mapping_tree();
    let tolerance = ToleranceConfig::strict();

    // Compare the VCD to itself -- every signal must match exactly.
    let input = load_and_align(vcd_path, vcd_path, &tree).unwrap();
    let result = compare_signals(&input, &tolerance);
    let summary = result.summary();

    assert_eq!(summary.mismatch, 0, "self-comparison should have 0 mismatches");
    assert_eq!(summary.missing_emu, 0);
    assert_eq!(summary.missing_sim, 0);
    // At least some signals should have matched.
    assert!(summary.exact_match > 0 || summary.both_empty > 0,
        "expected some matched signals");
}
```

- [ ] **Step 2: Run the test**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib vcd::compare::tests::self_comparison_all_exact_match -- --ignored
```

Expected: PASS. If it fails, debug the comparison engine.

- [ ] **Step 3: Print a full comparison report for review**

```bash
./target/release/vcd_compare \
  --emu build/unit_tests/08_tile_locks/--simulation-cycle-timeout.vcd \
  --sim build/unit_tests/08_tile_locks/--simulation-cycle-timeout.vcd \
  --tolerance strict \
  -o /tmp/claude-1000/vcd-self-compare.txt
```

Review the report. Confirm 100% pass rate.

- [ ] **Step 4: Commit**

```bash
git add src/vcd/compare.rs
git commit -m "test(vcd): end-to-end self-comparison proves pipeline works"
```

---

## Phase B: VC2802 Device Geometry for Emulator

With the comparison pipeline proven, add VC2802 as a device target so the
emulator can run in the same coordinate space as aiesimulator.

### Task 5: Generate VC2802 device model JSON

**Files:**
- Modify: `tools/aie-device-models.json`
- Run: `tools/mlir-aie-bridge.py` (if available) or construct manually

- [ ] **Step 1: Check if mlir-aie bridge can generate VC2802 model**

```bash
python3 tools/mlir-aie-bridge.py device-model --device xcve2802 2>/dev/null || echo "manual entry needed"
```

If the bridge script supports it, use the generated output. If not, construct
the entry manually from mlir-aie device model data (columns=17, rows=4 per
mlir-aie; but aiesimulator uses 38x11 -- verify which is correct for our use).

Note: mlir-aie reports VC2802 as 17 cols x 4 rows, but aiesimulator simulates
38 cols x 11 rows (the full Versal AIE2 array). For aiesim validation mode,
we need the 38x11 geometry. This **must** be a manual JSON entry because:
1. The mlir-aie bridge would report 17x4, not 38x11
2. `ModelConfig::from_arch_model()` adds +1 to column count (NPU convention
   at `src/device/arch_config.rs:220`), so the JSON `columns` field must
   account for this (set 37 so +1 gives 38, or skip the adjustment for
   Versal devices)

- [ ] **Step 2: Add VC2802 entry to device model JSON**

Manually construct the entry with correct tile type classification:
- Row 0: shim tiles (SHIMPL at cols 0,1,4,5,...; SHIMNOC at cols 2,3,6,7,...)
- Rows 1-2: mem tiles
- Rows 3-10: compute tiles

**Important**: This device config is metadata-only (dimensions, tile type
queries). Do NOT use it to instantiate a full `TileArray` -- that would
allocate 418 tiles with memory, DMA engines, etc. Add a comment noting
this is for VCD validation geometry only.

- [ ] **Step 2b: Verify MAX_COLS/MAX_ROWS are not used for static arrays**

Before bumping the constants in Task 6, grep the codebase:
```bash
grep -rn "MAX_COLS\|MAX_ROWS" src/
```
Confirm they are only used for bounds checks, not `[T; MAX_COLS]` sizing.

- [ ] **Step 3: Commit**

```bash
git add tools/aie-device-models.json
git commit -m "feat(device): add VC2802 device model for aiesim validation"
```

### Task 6: Bump array size limits and add VC2802 config

**Files:**
- Modify: `src/device/array.rs:34-37` (MAX_COLS, MAX_ROWS)
- Modify: `src/device/arch_config.rs` (add `ModelConfig::xcve2802()`)

- [ ] **Step 1: Increase MAX_COLS and MAX_ROWS**

In `src/device/array.rs`:

```rust
// Previous: MAX_COLS = 9, MAX_ROWS = 6
pub const MAX_COLS: u8 = 38;  // VC2802 has 38 columns
pub const MAX_ROWS: u8 = 11;  // VC2802 has 11 rows
```

Verify no code assumes the old limits.

- [ ] **Step 2: Add ModelConfig::xcve2802() constructor**

In `src/device/arch_config.rs`:

```rust
impl ModelConfig {
    pub fn xcve2802() -> Self {
        // Load from JSON, similar to npu2()
        ModelConfig::from_json_device("xcve2802")
    }
}
```

Or use `from_arch_model()` with the JSON entry from Task 5.

- [ ] **Step 3: Write test verifying VC2802 geometry**

```rust
#[test]
fn vc2802_has_correct_dimensions() {
    let config = ModelConfig::xcve2802();
    assert_eq!(config.columns(), 38);
    assert_eq!(config.rows(), 11);
    // Row 0: shim, rows 1-2: memtile, rows 3-10: compute
    assert_eq!(config.tile_type(7, 0), TileType::Shim);
    assert_eq!(config.tile_type(7, 1), TileType::MemTile);
    assert_eq!(config.tile_type(7, 3), TileType::Compute);
}
```

- [ ] **Step 4: Run all tests**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib
```

Expected: All existing tests still pass. New VC2802 test passes.

- [ ] **Step 5: Commit**

```bash
git add src/device/array.rs src/device/arch_config.rs
git commit -m "feat(device): VC2802 geometry support for aiesim validation mode"
```

### Task 7: Add --device flag to vcd_compare binary

**Files:**
- Modify: `src/bin/vcd_compare.rs`
- Modify: `src/vcd/mapping.rs` (export `build_vc2802_mapping_tree`)

- [ ] **Step 1: Add --device argument parsing**

```rust
// In arg parsing loop:
"--device" => {
    i += 1;
    device_name = args.get(i).cloned().unwrap_or_else(|| usage());
}

// In tree selection:
let tree = match device_name.as_str() {
    "npu1" => build_aie2_mapping_tree(),
    "vc2802" => build_vc2802_mapping_tree(),
    other => {
        eprintln!("Unknown device '{}'. Use npu1 or vc2802.", other);
        process::exit(1);
    }
};
```

Default to `vc2802` when running `--coverage` (most common use case with
aiesim VCDs), and `npu1` for `--emu`/`--sim` comparison mode.

- [ ] **Step 2: Test with real VCD**

```bash
./target/release/vcd_compare --coverage \
  build/unit_tests/08_tile_locks/--simulation-cycle-timeout.vcd \
  --device vc2802
```

Expected: Higher mapped percentage than with the NPU1 tree.

- [ ] **Step 3: Commit**

```bash
git add src/bin/vcd_compare.rs src/vcd/mapping.rs
git commit -m "feat(vcd): --device flag for vcd_compare (npu1 or vc2802)"
```

---

## Phase C: Bridge Test Integration

### Task 8: Update existing bridge test Phase 5c with --device flag

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (Phase 5c, lines ~1780-1885)

Phase 5c already exists and is substantially implemented: it discovers
Chess builds with `sim/` artifacts, runs aiesimulator with `--dump-vcd`,
finds the VCD file, runs `vcd_compare --coverage`, and reports results.
The main missing piece is the `--device vc2802` flag from Task 7.

- [ ] **Step 1: Review existing Phase 5c code**

Read `scripts/emu-bridge-test.sh` around lines 1780-1885. Understand the
current VCD coverage invocation at ~line 1867.

- [ ] **Step 2: Add --device vc2802 to the vcd_compare invocation**

Find the `vcd_compare --coverage` call and add `--device vc2802` so
it uses the correct mapping tree geometry for aiesim VCDs.

- [ ] **Step 3: Test manually**

```bash
./scripts/emu-bridge-test.sh --aiesim --chess-only add_one_using_dma
```

Expected: aiesimulator runs, VCD coverage audit runs, results reported.

- [ ] **Step 4: Commit**

```bash
git add scripts/emu-bridge-test.sh
git commit -m "feat(bridge): wire aiesim VCD coverage audit into --aiesim mode"
```

### Task 9: Add VCD comparison mode to bridge tests (future)

This task is for when the emulator can emit VCDs (requires `vcd-recording`
feature and instrumentation of the device state machine). Deferred until
Phase A and B are proven.

When ready:
1. Build emulator with `--features vcd-recording`
2. Run test through emulator, producing emu.vcd
3. Run test through aiesimulator, producing sim.vcd
4. Run `vcd_compare --emu emu.vcd --sim sim.vcd --device vc2802`
5. Report comparison results alongside existing PASS/FAIL

---

## Verification Checklist

After completing Phases A-C:

- [ ] `vcd_compare --coverage <aiesim.vcd> --device vc2802` maps >80% of active tile signals
- [ ] `vcd_compare --emu <vcd> --sim <same-vcd> --tolerance strict` reports 100% ExactMatch
- [ ] `cargo test --lib` passes with no regressions
- [ ] `emu-bridge-test.sh --aiesim` runs end-to-end (when aietools available)
- [ ] VC2802 device config loads without errors

## Risk Notes

1. **VC2802 38x11 vs mlir-aie 17x4**: aiesimulator simulates the full Versal
   array (38x11), but mlir-aie's device model reports 17x4. The mapping tree
   must use aiesimulator's actual geometry, not mlir-aie's. Verify by checking
   tile coordinates in actual VCD files.

2. **Signal naming drift**: aiesimulator VCD signal names may change between
   aietools versions. The coverage audit is the canary -- if mapped percentage
   drops, signal names changed.

3. **Memory and PerfCounter mappings**: These StatePath variants exist but have
   no mapping functions yet. Add them only if the coverage audit shows they
   represent a significant fraction of unmapped signals.

4. **Feature-gated emission**: The emulator cannot yet produce VCDs for
   comparison (requires `vcd-recording` + device state instrumentation).
   Phase A uses self-comparison as a stand-in. Real emu-vs-sim comparison
   is Phase C Task 9, which is deferred.
