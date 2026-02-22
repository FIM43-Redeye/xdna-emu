# Trace Injection Design

Date: 2026-02-21

## Goal

Collect hardware execution traces from every test in the mlir-aie npu-xrt
suite (78 tests). These traces serve two purposes: understanding real NPU
behavior cycle-by-cycle, and validating xdna-emu's emulator against ground
truth.

## Constraints

- Tracing cannot be added after compilation. Packet flow routing is baked
  into the CDO (stream switch config) and trace register writes are baked
  into the instruction stream. The MLIR source must include tracing before
  `aiecc.py` runs.
- The hardware trace unit is passive -- it does not change core execution
  timing, DMA behavior, or functional correctness. Trace packets route via
  the separate Trace port bundle through the stream switch.
- Each tile traces up to 8 events simultaneously. Two trace units per tile
  (core + memory module) can run in parallel for 16 events. The full event
  namespace is ~128 events per tile type.
- The host must allocate a trace buffer as an XRT BO and pass it to the
  kernel. The existing test.cpp files do not do this, so we need our own
  executor.

## Architecture

Three layers, each doing what it is best at:

```
lit-runner (Rust)                    -- orchestration, staleness, emulator comparison
    |
    +-- trace-inject.py (Python)     -- MLIR IR manipulation via mlir-aie API
    |       uses: Module.parse(), InsertionPoint, setup.py trace utilities
    |
    +-- aiecc.py (existing)          -- compiles trace-enabled MLIR to xclbin
    |
    +-- trace-run.py (Python)        -- executes on NPU via XRT Python bindings
    |       allocates trace buffer, collects trace + output data
    |
    +-- parse_trace() (existing)     -- binary trace to Perfetto JSON
    |
    +-- xdna-emu (Rust, optional)    -- emulator run, produces emulator trace
    |
    +-- trace comparison (Rust)      -- diffs hardware vs emulator Perfetto traces
```

### Data flow for a single test

1. trace-inject.py reads upstream MLIR (or runs aie2.py to generate it),
   produces trace-enabled MLIR + manifest JSON
2. aiecc.py compiles to xclbin + insts.bin
3. trace-run.py executes on NPU, writes output data + raw trace binary
4. parse_trace() converts trace binary to Perfetto JSON
5. (Optional) xdna-emu runs same xclbin, produces emulator Perfetto JSON
6. Comparison tool diffs the two traces

### File layout

```
xdna-emu/
  tools/
    trace-inject.py        -- MLIR trace injection tool
    trace-run.py           -- Universal trace-aware NPU executor
  build/
    traced-tests/          -- generated trace-enabled sources (gitignored)
      <test-name>/
        aie_traced.mlir    -- trace-enabled MLIR
        manifest.json      -- buffer layout + trace config
        aie.xclbin         -- compiled binary
        insts.bin           -- instruction stream
        .source-hash       -- SHA256 of upstream source
        .inject-hash       -- SHA256 of trace-inject.py
    traces/                -- collected Perfetto JSON traces (gitignored)
      <test-name>/
        trace_raw.bin      -- raw trace buffer
        trace.json         -- Perfetto JSON
        output.bin         -- output buffer for correctness check
  src/bin/lit_runner.rs    -- extended with --trace flag
  src/testing/lit_trace.rs -- trace pipeline integration, staleness checks
```

## Component 1: trace-inject.py

A Python tool that takes any npu-xrt test source and produces a
trace-enabled MLIR variant using the mlir-aie Python API.

### Unified injection path

Both test types (aie2.py and raw MLIR) converge to the same MLIR-level
injection:

- aie2.py tests: run `python aie2.py npu` to generate MLIR text, then
  inject at IR level
- Raw MLIR tests: read aie.mlir directly, inject at IR level

The injection operates on real IR using real dialect APIs -- zero text
manipulation.

### Injection steps

1. `Module.parse(mlir_text)` in an mlir-aie Context
2. `find_ops()` to locate DeviceOp, TileOps, RuntimeSequenceOp
3. Classify tiles: shim (row 0), memtile (row 1), compute (row >= 2)
4. At device block level (`InsertionPoint.at_block_terminator`):
   - Call `configure_packet_tracing_flow(compute_tiles, shim_tile)`
   - Adds `aie.packet_flow` ops for trace routing
5. At runtime sequence block:
   - `InsertionPoint.at_block_begin`: call
     `configure_packet_tracing_aie2(compute_tiles, shim_tile, trace_size)`
   - Adds trace register writes + shim BD setup
   - Find the last dma_wait/dma_await_task op
   - `InsertionPoint.after`: call `gen_trace_done_aie2(shim_tile)`
   - Adds trace stop event
6. Add trace buffer argument to RuntimeSequenceOp signature
7. `str(module)` to serialize the modified MLIR
8. Write manifest JSON alongside

### Proven API surface

All operations are production-proven in the mlir-aie codebase:

| Operation | Proven by |
|-----------|-----------|
| `Module.parse()` | `parse.py`, `aiecc/main.py` |
| `find_ops()` | `parse.py`, `aiecc/main.py` |
| `InsertionPoint.at_block_terminator()` | `aiecc/main.py` |
| `InsertionPoint.at_block_begin()` | MLIR Python test suite |
| `InsertionPoint.after(op)` | MLIR Python test suite |
| `configure_packet_tracing_flow()` | `setup.py`, `bd_chain_repeat` test |
| `configure_packet_tracing_aie2()` | `setup.py`, `vec_mul_event_trace` |
| `gen_trace_done_aie2()` | `setup.py` |
| `str(module)` round-trip | `python_passes.py`, `parse.py` |

### Event configuration

Default core events (the setup.py standard set):

1. INSTR_EVENT_0 -- kernel begin marker
2. INSTR_EVENT_1 -- kernel end marker
3. INSTR_VECTOR -- vector ALU active
4. PORT_RUNNING_0 (master) -- outbound DMA stream active
5. PORT_RUNNING_1 (slave) -- inbound DMA stream active
6. INSTR_LOCK_ACQUIRE_REQ -- lock acquire
7. INSTR_LOCK_RELEASE_REQ -- lock release
8. LOCK_STALL -- core stalled on lock

Overridable via `--events` flag for custom event sets.

### CLI

```
trace-inject.py <test_source_dir> --output <output_dir> \
    [--trace-size BYTES] [--events core|mem|both]
```

## Component 2: trace-run.py

A universal Python executor that runs any compiled xclbin on the NPU with
trace collection, using XRT Python bindings.

### What it does

1. Reads a manifest JSON (produced by trace-inject.py) describing the
   test's buffer layout
2. Allocates XRT buffer objects: inputs, outputs, instructions, trace
3. Loads the xclbin, writes input data, runs the kernel
4. Syncs output and trace buffers from device
5. Writes output data + raw trace binary
6. Calls parse_trace() to produce Perfetto JSON

### Manifest format

```json
{
    "test_name": "add_one_objFifo",
    "xclbin": "aie.xclbin",
    "insts": "insts.bin",
    "trace_size": 1048576,
    "buffers": [
        {"name": "in", "size_bytes": 4096, "dtype": "int32",
         "direction": "input", "pattern": "incrementing"},
        {"name": "out", "size_bytes": 4096, "dtype": "int32",
         "direction": "output"}
    ],
    "kernel_name": "MLIR_AIE",
    "trace_ddr_id": 4,
    "tiles_traced": [
        {"col": 0, "row": 2, "trace_types": ["core"]}
    ],
    "source_mlir": "aie_traced.mlir"
}
```

Buffer spec is extracted from the RuntimeSequenceOp memref arguments
during injection, which is the authoritative source.

### CLI

```
trace-run.py <manifest.json> --output-dir <dir>
```

Outputs: trace_raw.bin, output.bin, trace.json

## Component 3: lit-runner --trace

Extends the existing Rust lit-runner with trace orchestration.

### Workflow

```
cargo run --bin lit-runner -- --trace [FILTER...]
```

For each test:
1. Check staleness (source hash + inject tool hash)
2. Generate MLIR if aie2.py (run the Python script)
3. Inject traces (call trace-inject.py)
4. Compile (call aiecc.py)
5. Execute + collect (call trace-run.py)
6. Report progress

### New Rust module: src/testing/lit_trace.rs

Handles:
- Staleness checking (SHA256 comparison)
- Subprocess orchestration (inject -> compile -> execute)
- Trace file management (output directory structure)
- Progress reporting integration

## Trace Buffer Sizing

Default: 1MB per compute tile. With 96GB of system RAM, there is no reason
to be conservative. The shim BD buffer_length field supports up to ~1MB per
BD; for larger buffers, BD chaining would be needed.

- 1-tile test: 1MB trace buffer
- 5-tile test (full column): 5MB trace buffer
- Multi-pass full-event sweep: 1MB per pass, 16 passes for all 128 events

## Staleness Tracking

Each test in build/traced-tests/ gets:

- `.source-hash` -- SHA256 of the upstream MLIR/Python source
- `.inject-hash` -- SHA256 of trace-inject.py itself

If both hashes match AND the xclbin exists, skip injection + compilation.
Re-runs go straight to execution, making them near-instant.

## Future: Multi-Pass Full-Event Tracing

The hardware limits each trace unit to 8 events. But the hardware is
deterministic: same inputs produce identical execution every time. By
running the same test 16 times with different 8-event selections, we can
collect all ~128 core events and merge them into a single comprehensive
trace. The Perfetto JSON format handles this naturally since each pass
produces separate pid/tid lanes that merge by aligning on cycle timestamps.

## Future: Emulator Comparison

xdna-emu already produces Perfetto JSON via --trace. A comparison tool
would load both traces (hardware + emulator), align by event name and
relative timing, and report matching events, missing events, and timing
divergences. This is the ultimate validation loop: real silicon tells us
exactly where the emulator diverges.

## Test Coverage

| Test type | Count | Injection method |
|-----------|-------|-----------------|
| aie2.py (IRON Python) | 24 | Run Python to get MLIR, inject at IR level |
| Raw MLIR (run.lit) | 48 | Read aie.mlir, inject at IR level |
| Already traced | 1 | vec_mul_event_trace (skip or re-inject) |
| Special cases | ~5 | loadpdi, ctrl_packet, reconfigure (may need manual handling) |

## Deliverables

| Component | Language | Est. lines | Purpose |
|-----------|----------|-----------|---------|
| tools/trace-inject.py | Python | ~300 | MLIR IR trace injection |
| tools/trace-run.py | Python | ~250 | Universal NPU executor with trace |
| src/bin/lit_runner.rs | Rust | ~100 added | --trace flag, orchestration |
| src/testing/lit_trace.rs | Rust | ~150 | Trace pipeline, staleness |
