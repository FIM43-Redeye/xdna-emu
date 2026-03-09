# Always-On Trace Pipeline Design

## Goal

Make hardware trace collection the default for every bridge test run.
Every test produces `trace_raw.bin` alongside its PASS/FAIL result, for
both HW and EMU, with both compilers. Tracing is opt-out (`--no-trace`)
instead of opt-in.

## Architecture

A new standalone tool (`trace-prepare.py`) prepares a traced variant of
each test before compilation. It runs trace-inject.py on the MLIR (existing
two-pass pathfinder, zero-interference guarantee) and patches test.cpp via
tree-sitter to add trace buffer allocation and collection. The bridge script
calls it once per test, then forks to Chess/Peano compilation using run.lit
commands on the traced artifacts.

## Tech Stack

- Python 3.13 (existing tooling)
- tree-sitter + tree-sitter-cpp (C++ AST transforms)
- trace-inject.py (existing, called as library)
- mlir-aie Python API (existing, via trace-inject.py)

---

## Pipeline Flow

```
Test Source Dir (aie.mlir + test.cpp)
        |
  trace-prepare.py          <-- NEW: one-time preparation
        |
        v
  Traced Build Dir:
    aie_traced.mlir          (trace routing injected via trace-inject.py)
    test_traced.cpp          (trace buffer added via tree-sitter)
    events.json              (event name manifest for trace decoding)
        |
        +-- Chess: aiecc.py --xchesscc --xbridge aie_traced.mlir
        |         clang test_traced.cpp -> test.exe
        |
        +-- Peano: aiecc.py --no-xchesscc --no-xbridge aie_traced.mlir
                   clang test_traced.cpp -> test.exe
        |
        v
  Run (HW or EMU):
    XDNA_TRACE_DIR=/path/to/output ./test.exe ...
        |
        v
  Output Dir:
    test result (PASS/FAIL from test.cpp validation logic)
    trace_raw.bin (binary trace data, written by patched test.exe)
    events.json (copied, for trace decoding/comparison)
```

Key properties:
- trace-prepare.py runs **once per test** (compiler-independent)
- The bridge script forks to Chess/Peano **after** preparation
- `XDNA_TRACE_DIR` env var controls collection: set = write traces,
  unset = zero overhead (safe for upstream lit testing)
- Same test.exe validates correctness AND collects traces in one run

---

## trace-prepare.py

Standalone tool. Takes a test source directory, produces a traced build
directory ready for normal compilation.

### Inputs

- `<test_source_dir>` -- e.g., `mlir-aie/test/npu-xrt/add_one_using_dma/`
- `--output <dir>` -- where to write traced artifacts
- `--trace-size <bytes>` -- default 1048576 (1MB)
- `--device <name>` -- default auto-detect from MLIR

### Steps

1. **Copy and prepare MLIR** -- Read `aie.mlir`, apply `NPUDEVICE`
   substitution in Python (same as run.lit sed). Write `aie_arch.mlir`.

2. **Run trace-inject.py** -- Call as a library function (not subprocess).
   Produces `aie_traced.mlir` + manifest with event config. Uses existing
   two-pass pathfinder: route data flows first, then route trace on
   remaining fabric. Zero interference guaranteed.

3. **Patch test.cpp via tree-sitter** -- Parse into AST. Apply three
   transforms (see next section). Write `test_traced.cpp`.

4. **Write events.json** -- Event slot configuration from trace-inject
   manifest, for later trace decoding.

5. **Write prepare-status.txt** -- OK or FAIL with reason.

### Does NOT

- Compile anything (bridge script's job via run.lit)
- Run anything
- Choose Chess vs Peano (compiler-independent)

---

## Tree-Sitter C++ Transform

Three insertion points, found by AST queries on test.cpp:

### 1. Trace buffer allocation

After the last `xrt::bo` declaration:

```cpp
// Trace buffer (injected by trace-prepare.py)
constexpr size_t trace_size = 1048576;  // 1MB
auto bo_trace = xrt::bo(device, trace_size, XRT_BO_FLAGS_HOST_ONLY,
                         kernel.group_id(NEXT_ID));
memset(bo_trace.map<void*>(), 0, trace_size);
bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
```

`NEXT_ID` computed by finding the highest `group_id(N)` across existing
buffer allocations and adding 1.

### 2. Kernel call argument

Find the `kernel(opcode, bo_instr, ...)` call expression. Append `bo_trace`
as the last argument.

### 3. Trace write-out

After `run.wait()`:

```cpp
// Write trace data (injected by trace-prepare.py)
if (const char* trace_dir = std::getenv("XDNA_TRACE_DIR")) {
    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto trace_ptr = bo_trace.map<char*>();
    std::string trace_path = std::string(trace_dir) + "/trace_raw.bin";
    std::ofstream trace_file(trace_path, std::ios::binary);
    trace_file.write(trace_ptr, trace_size);
}
```

Also adds `#include <fstream>` if not already present.

### Edge cases

- Tests with existing trace buffers (detected by `trace_size` variable):
  skip patching
- `xrt::ext::kernel` vs `xrt::kernel`: both patterns matched
- Multiple kernel runs (loops): trace write after the last `run.wait()`
- Multi-device tests: trace-inject handles MLIR; C++ patch is the same

### Failure mode

If tree-sitter cannot locate an insertion point and the test is NOT
quarantined: **hard error**. Loud message, nonzero exit code. No silent
failures.

---

## Quarantine Hierarchy

Three quarantine files, checked in order:

| File | Purpose | Behavior |
|------|---------|----------|
| `test-quarantine.txt` (NEW) | Fundamentally broken tests (segfault, compile fail) | Skip entirely: SKIP (quarantined) |
| `trace-quarantine.txt` (exists) | Trace injection breaks the test (IOMMU, routing) | Compile and run WITHOUT tracing |
| `hw-quarantine.txt` (exists) | Tests that cause TDRs on hardware | Run on HW last, isolated |

If a test is NOT quarantined and trace preparation fails, that is a
**hard error** in the tooling, not an expected condition.

---

## Bridge Script Changes

### New flow in compile_one

```
1. trace-prepare.py <src> --output <build_dir>/traced/
   - If test-quarantined: skip entirely
   - If trace-quarantined: skip trace prep, use original artifacts
   - If prepare fails AND not quarantined: HARD FAIL
   - If prepare OK: use traced artifacts

2. For each compiler (Chess/Peano):
   - Parse run.lit commands (existing)
   - Transform for compiler (existing transform_for_chess/peano)
   - Replace MLIR path: aie_arch.mlir -> traced/aie_traced.mlir
   - Replace test.cpp path: test.cpp -> traced/test_traced.cpp
   - Execute commands

3. Run phase:
   - Set XDNA_TRACE_DIR=<results>/<test>.<compiler>.{hw,emu}/
   - Execute test.exe (same as before)
   - trace_raw.bin appears automatically
```

### Removed

- `compile_trace_base_one()` -- replaced by trace-prepare.py
- `--trace=sweep` / `--trace=sweep-all` modes -- tracing is always on
- `trace_one_test()` function -- no separate trace phase
- Phase 4b entirely -- traces come from normal Phase 3+4 runs
- `trace-sweep.py` invocation path from bridge script

### Kept

- `trace-quarantine.txt` -- respected, those tests skip trace preparation
- Trace comparison in Phase 5 -- runs automatically when both traces exist
- `--no-trace` flag (NEW) -- opt-out for when tracing isn't wanted

### Runtime

- `XDNA_TRACE_DIR` set for both HW and EMU runs
- After each run, `trace_raw.bin` + `events.json` exist alongside test output
- Phase 5 report adds trace column (CLEAN/DIVERGE/ERROR) for every test

---

## Output Structure

```
/tmp/emu-bridge-results-YYYYMMDD/
    <test>.chess.hw/
        trace_raw.bin
        events.json
    <test>.chess.emu/
        trace_raw.bin
        events.json
    <test>.peano.hw/
        trace_raw.bin
        events.json
    <test>.peano.emu/
        trace_raw.bin
        events.json
```

Compatible with the trace visualizer:
```
cargo run -- --trace-view-hw .../<test>.chess.hw \
             --trace-view-emu .../<test>.chess.emu
```

---

## Scope Boundaries

### In scope (v1)

- trace-prepare.py with tree-sitter C++ patching
- test-quarantine.txt (new quarantine tier)
- Bridge script integration (always-on tracing)
- --no-trace opt-out flag
- Automatic trace comparison in Phase 5

### Explicitly not v1

- Removal of trace-sweep.py (keep for standalone use, just not called
  by bridge script)
- Per-column multi-shim tracing (trace-inject.py supports it, but
  single-channel is fine for now)
- Perfetto JSON generation (raw binary is the interchange format)
- Live trace streaming to the GUI visualizer
