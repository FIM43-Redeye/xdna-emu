# Trace Sweep Consolidation Design

## Goal

Refactor trace-sweep.py from a standalone compilation+execution pipeline into
a thin orchestrator that operates on pre-compiled artifacts from the bridge
script. Remove trace-run.py entirely. Eliminate drift between the sweep and
bridge pipelines.

## Architecture

The bridge script owns compilation and test execution. trace-sweep.py owns
event batching and trace merging. The boundary is the compiled build
directory: bridge produces it, sweep consumes it.

```
Bridge Script (emu-bridge-test.sh)
  |
  +-- trace-prepare.py -> traced MLIR + test_traced.cpp
  +-- compile_one_compiler() -> aie.xclbin + insts.bin
  +-- compile_one() -> test.exe
  |
  +-- [--sweep] invokes trace-sweep.py per test:
        |
        +-- trace-patch-events.py -> batch_NN/insts.bin (per batch)
        +-- test.exe -i batch_NN/insts.bin (HW and/or EMU)
        +-- trace-trim.py -> trimmed trace_raw.bin
        +-- trace-merge.py -> unified Perfetto JSON
```

## New trace-sweep.py Interface

```
trace-sweep.py --build-dir <path>   # Contains aie.xclbin, insts.bin
               --test-exe  <path>   # Shared test.exe (compiler-agnostic)
               --output    <path>   # Output directory for sweep results
               [--run-cmd  <cmd>]   # Run command template (default: auto)
               [--no-hw]            # Skip hardware runs
               [--no-emu]           # Skip emulator runs
               [--core-only]        # Only sweep core events
               [--mem-only]         # Only sweep memory events
               [--hw-jobs N]        # Parallel HW batches (default: 1)
               [--hw-cooldown S]    # Seconds between HW runs (default: 2)
```

Required in `--build-dir`:
- `aie.xclbin` -- compiled from traced MLIR
- `insts.bin` -- instruction stream with default trace events

Required adjacent (discovered automatically):
- `../traced/manifest.json` -- trace metadata (tile list for event targeting)

The `--run-cmd` template defaults to:
`./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin`

Sweep substitutes `insts.bin` with the per-batch patched path.

## Per-Batch Execution

For each of N batches (N = ceil(num_events / 7)):

1. **Patch**: Copy `insts.bin` to `$output/batch_NN/insts.bin`. Run
   `trace-patch-events.py` to overwrite Trace_Event0/Event1 register
   values with this batch's event configuration.

2. **Run HW** (if enabled): Execute test.exe from the build dir with:
   - `-i $output/batch_NN/insts.bin` (patched instruction stream)
   - `XDNA_TRACE_DIR=$output/batch_NN/hw/` (trace output)
   - `XRT_DEVICE_BDF=$bdf` (real NPU)
   HW batches run serially (NPU contention, cooldown between runs).

3. **Run EMU** (if enabled): Same as HW but with:
   - `XDNA_EMU=1` and `XRT_DEVICE_BDF=ffff:ff:1f.0`
   EMU batches run in parallel (no hardware constraint).

4. **Trim**: `trace-trim.py` on each `trace_raw.bin`.

After all batches:

5. **Merge**: `trace-merge.py` stitches per-batch traces into unified
   Perfetto JSON, using TRUE anchor alignment.

## Bridge Script --sweep Handler

The existing `--sweep` handler changes from delegating with a test source
dir to passing pre-compiled artifacts:

```bash
for compiler in "${compilers[@]}"; do
  local build_dir="$BUILD_BASE/$name/$compiler"
  local test_exe="$BUILD_BASE/$name/test.exe"
  local sweep_dir="$RESULTS_DIR/${safe}.${compiler}.sweep"
  local run_cmd
  run_cmd="$(get_run_cmd "$src_dir")"

  local sweep_args=(
    --build-dir "$build_dir"
    --test-exe  "$test_exe"
    --output    "$sweep_dir"
    --run-cmd   "$run_cmd"
  )
  $RUN_HW  || sweep_args+=(--no-hw)
  $RUN_EMU || sweep_args+=(--no-emu)

  python3 "$EMU_ROOT/tools/trace-sweep.py" "${sweep_args[@]}"
done
```

Sweep runs per-compiler, inheriting the bridge's `RUN_HW`/`RUN_EMU` flags.

## What Gets Deleted

| Component | Lines | Reason |
|-----------|-------|--------|
| `trace-run.py` | 329 | test.exe replaces it entirely |
| `compile_base_trace()` | ~80 | Bridge compiles |
| `compile_kernel_objects()` | ~60 | Bridge compiles |
| `resolve_test_dir()` | ~25 | Takes build dir directly |
| `run_batch_hw()` | ~70 | Replaced by test.exe subprocess |
| `run_batch_emu()` | ~25 | Replaced by test.exe subprocess |
| `--compile-only` flag | -- | Bridge compiles |
| `--use-base` flag | -- | Build dir is required input |
| `--compare-parallel` flag | -- | Bridge handles determinism testing |

## What Stays (Unchanged)

| Component | Purpose |
|-----------|---------|
| Event sets (CORE/MEM/MEMTILE) | ~85 event names |
| `batch_events()` | Generate 8-slot batches from event list |
| `trace-patch-events.py` | Binary patch insts.bin event registers |
| `trace-merge.py` | Merge per-batch Perfetto JSON |
| `trace-trim.py` | Strip sentinel padding |

## What Gets Rewritten

| Component | Old | New |
|-----------|-----|-----|
| `main()` / CLI | Source dir + full pipeline | Build dir + orchestration only |
| `prepare_batch()` | Manifest rewrite + symlinks | Just patch insts.bin |
| `_run_single_pass()` | Compilation + execution | Execution only |
| Batch runners | trace-run.py via subprocess | test.exe via subprocess |

## Output Structure

```
$output/
  batch_00/
    insts.bin         # Patched instruction stream
    events.json       # Event configuration for this batch
    hw/
      trace_raw.bin   # Trimmed HW trace
    emu/
      trace_raw.bin   # Trimmed EMU trace
  batch_01/
    ...
  hw-merged.json      # Unified HW Perfetto trace
  emu-merged.json     # Unified EMU Perfetto trace
  sweep-manifest.json # Batch status and metadata
```

## Scope Boundaries

### In scope
- Refactor trace-sweep.py CLI and execution logic
- Delete trace-run.py
- Update bridge script --sweep handler
- Preserve event batching and merge logic

### Not in scope
- Changes to trace-patch-events.py
- Changes to trace-merge.py
- Changes to trace-trim.py
- New event types or batching strategies
- Per-column multi-shim sweep (future)
