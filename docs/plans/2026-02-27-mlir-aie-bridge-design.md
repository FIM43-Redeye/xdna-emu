# mlir-aie Python Bridge Design

## Context

xdna-emu integrates with mlir-aie in many places -- device model extraction,
test discovery, build orchestration, trace export. Some of these are
data-driven (device model JSON, register database), but many are fragile
(regex-based test parsing, hardcoded edge detection, manual path discovery).

mlir-aie has a rich Python API that we're underutilizing. The lit config
helpers provide hardware detection, tool availability checking, and NPU model
mapping. The trace event enums define canonical hardware event codes. The
device model API exposes memory affinity, edge classification, and tile type
queries we currently hardcode.

### Goal

Build a unified Python bridge (`tools/mlir-aie-bridge.py`) that replaces
`aie-device-dump.py` and becomes the single entry point for all mlir-aie
queries. The Rust test runner and build system call it via subprocess, parse
JSON output. Build-time codegen uses it for trace event tables.

## Design Decisions

### Unified script with subcommands

One script, one Python path resolution, one place to maintain. Replaces
`aie-device-dump.py`. Subcommands:

- `device-model` -- topology, tile types, port counts, memory sizes
- `platform-detect` -- xrt-smi hardware detection, tool availability
- `test-manifest` -- scan tests/examples for target device + build feasibility
- `trace-events` -- export hardware event enums (CoreEvent, MemEvent, etc.)

### Subprocess JSON protocol

Rust calls `python3 tools/mlir-aie-bridge.py <subcommand> [args]`, reads
JSON from stdout. Simple, debuggable (run by hand), no FFI. Already the
pattern used by `aie-device-dump.py`.

### Trace events: build-time codegen + runtime validation

The bridge exports event enums as JSON. `build.rs` generates Rust const
tables and name lookup arrays (zero runtime cost). At test-runner startup,
a validation step optionally re-queries the bridge and checks that generated
tables match current mlir-aie (catches version drift).

### Test manifest: scan + build feasibility

One scan per test-runner invocation produces a manifest JSON with per-test
metadata: target device, compiler requirements, REQUIRES features, build
feasibility (Makefile exists, kernel sources present, required tools
available). The Rust runner filters and classifies using this manifest
instead of fragile regex parsing.

## Subcommand Details

### `device-model`

Absorbs current `aie-device-dump.py` functionality plus new fields:

```
mlir-aie-bridge.py device-model [--device NAME]
```

**New fields beyond current aie-device-dump.py:**
- `program_memory_size` -- currently hardcoded as 16KB in aie2_spec.rs
- `bank_size` per tile type -- currently hardcoded (8KB compute, 32KB memtile)
- `memory_affinity` per tile -- `is_mem_east/west/north/south()` results
- `edge_info` per tile -- `is_internal/north/south/east/west()` booleans

**Output**: Same JSON structure as current `aie-device-models.json` with
additional fields. Backward compatible.

### `platform-detect`

Mirrors `LitConfigHelper` from mlir-aie's `lit_config_helpers.py`:

```
mlir-aie-bridge.py platform-detect
```

**Output**:
```json
{
  "hardware": {
    "npu_model": "npu1",
    "npu_generation": "Phoenix",
    "arch": "AIE2",
    "xrt_found": true,
    "xrt_version": "2.18.0"
  },
  "tools": {
    "peano": {"found": true, "path": "/path/to/llc"},
    "chess": {"found": true, "path": "/path/to/xchesscc"},
    "aiesimulator": {"found": true, "path": "/path/to/aiesimulator"}
  },
  "features": ["ryzen_ai", "ryzen_ai_npu1", "peano", "chess", "aiesimulator"]
}
```

Uses `xrt-smi examine` parsing from `LitConfigHelper.detect_xrt()`,
`detect_chess()`, `detect_peano()`, `detect_aiesimulator()`. Reuses their
NPU_MODELS mapping directly.

### `test-manifest`

Scans test and example directories, extracts metadata per test:

```
mlir-aie-bridge.py test-manifest --npu-xrt-dir PATH --examples-dir PATH
```

**Per-test metadata:**
```json
{
  "name": "add_one_using_dma",
  "path": "/path/to/test/dir",
  "target_device": "npu1",
  "target_arch": "AIE2",
  "requires": ["ryzen_ai_npu1"],
  "compilers": ["peano", "chess"],
  "build_feasibility": {
    "makefile_exists": true,
    "kernel_sources_exist": true,
    "python_generator_exists": true,
    "missing_dependencies": []
  },
  "skip_reason": null
}
```

**Target detection strategy** (priority order):
1. Parse Python generator for `AIEDevice.npu1` / `AIEDevice.npu2` usage
2. Parse MLIR files for `aie.device(npu1)` / `aie.device(npu2)` ops
3. Check Makefile for `NPU2=1` flag
4. Default to `npu1` (our hardware)

**Build feasibility checks:**
- Makefile exists in test directory
- Required kernel source files (.cc, .py) exist
- No XFAIL/SKIP annotation that would prevent execution
- Required tools (chess, peano) are available per platform-detect

### `trace-events`

Exports mlir-aie's 4 trace event enum classes:

```
mlir-aie-bridge.py trace-events [--arch aie2]
```

**Output**:
```json
{
  "arch": "aie2",
  "enums": {
    "CoreEvent": {
      "NONE": 0,
      "TRUE": 1,
      "INSTR_VECTOR": 37,
      "INSTR_LOAD": 38,
      "MEMORY_STALL": 22,
      ...
    },
    "MemEvent": {
      "DMA_S2MM_0_START_TASK": 18,
      "DMA_S2MM_0_FINISHED_BD": 20,
      ...
    },
    "MemTileEvent": { ... },
    "ShimTileEvent": { ... }
  },
  "trace_registers": {
    "core": {"control0": "0x340D0", "control1": "0x340D4", ...},
    "mem": {"control0": "0x140D0", ...},
    "memtile": {"control0": "0x940D0", ...}
  }
}
```

**Build-time codegen** (`build.rs`):
- Generates `trace_event_codes.rs` with const arrays per enum
- Generates `event_name_lookup: [&str; 128]` per tile type
- Generates `event_code_from_name(tile_type, name) -> Option<u8>`

**Runtime validation** (optional, test-runner startup):
- Re-query bridge, compare against compiled-in tables
- Warn if mismatch detected (mlir-aie was rebuilt with different events)

## Rust Integration

### Test runner changes (`src/testing/`)

Replace fragile artifact discovery with manifest-driven flow:

```rust
// At startup:
let manifest = bridge::test_manifest(&mlir_aie_path)?;
let platform = bridge::platform_detect()?;

// Per test:
for test in manifest.tests {
    if !platform.supports_arch(&test.target_arch) {
        report_skip(&test, "platform mismatch");
        continue;
    }
    if !test.build_feasibility.is_buildable() {
        report_skip(&test, &test.build_feasibility.reason());
        continue;
    }
    // ... proceed with build + run
}
```

### Bridge invocation module (`src/integration/bridge.rs`)

New module that wraps subprocess calls:

```rust
pub fn device_model(mlir_aie_path: &Path) -> Result<DeviceModels>;
pub fn platform_detect() -> Result<PlatformInfo>;
pub fn test_manifest(config: &ManifestConfig) -> Result<TestManifest>;
// trace-events is build-time only, called from build.rs
```

Each function calls the bridge script, parses JSON, returns typed Rust
structs. Error handling: if bridge fails (mlir-aie not installed, Python
not found), fall back to existing behavior with a warning.

### Device model enrichment (`src/device/`)

The enriched device model JSON replaces hardcoded values:
- `PROGRAM_MEMORY_SIZE` -> from device model
- `COMPUTE_TILE_BANK_SIZE` -> from device model `bank_size`
- Edge detection (`col == 0`) -> from device model `edge_info`
- Memory affinity validation -> from device model `memory_affinity`

### Trace alignment (`src/trace/`)

Generated event code tables enable:
- `EventType` variants carry their hardware code integer
- Perfetto export includes `hw_code` in args for cross-reference
- Port events suffixed correctly: `PORT_IDLE_0` not `PORT_IDLE`
- DMA events fully qualified: `DMA_S2MM_0_START_TASK` not `DMA_START_TASK`
- Future: trace unit simulation filters events by configured slots

## Files

| File | Changes |
|------|---------|
| `tools/mlir-aie-bridge.py` | **NEW** -- unified bridge script with 4 subcommands |
| `tools/aie-device-dump.py` | **DELETED** -- absorbed into bridge |
| `tools/aie-device-models.json` | Regenerated with new fields |
| `build.rs` | Add trace-events codegen step |
| `src/integration/bridge.rs` | **NEW** -- Rust wrapper for bridge subprocess calls |
| `src/integration/mod.rs` | Add `pub mod bridge;` |
| `src/testing/emu_runner.rs` | Use manifest for test filtering |
| `src/testing/artifacts.rs` | Simplify: manifest provides feasibility info |
| `src/trace/mod.rs` | Use generated event code tables |
| `src/trace/store.rs` | Align event names with canonical mlir-aie names |
| `src/device/model.rs` | Parse new device model fields |
| `src/device/aie2_spec.rs` | Replace hardcoded memory sizes with model values |

## Verification

1. `python3 tools/mlir-aie-bridge.py device-model` -- produces valid JSON
2. `python3 tools/mlir-aie-bridge.py platform-detect` -- detects NPU1 + tools
3. `python3 tools/mlir-aie-bridge.py test-manifest --npu-xrt-dir ...` -- scans tests
4. `python3 tools/mlir-aie-bridge.py trace-events` -- exports all 4 enums
5. `cargo test --lib` -- all existing tests pass with new device model
6. `cargo run --bin npu-test -- -v add_one` -- test runner uses bridge
7. Build failures now show "skip: requires npu2 (have npu1)" instead of "exit 2"
