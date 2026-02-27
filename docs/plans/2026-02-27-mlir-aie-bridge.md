# mlir-aie Python Bridge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified Python bridge (`tools/mlir-aie-bridge.py`) that
queries mlir-aie's Python API and emits structured JSON, replacing
`aie-device-dump.py` and adding platform detection, test manifest generation,
and trace event export.

**Architecture:** Single Python script with argparse subcommands, each
outputting JSON to stdout. Rust side invokes via subprocess, parses JSON
with serde. Build-time codegen for trace event tables via `build.rs`.

**Tech Stack:** Python 3 (mlir-aie ironenv), argparse, json; Rust serde_json,
std::process::Command

**Design doc:** `docs/plans/2026-02-27-mlir-aie-bridge-design.md`

---

### Task 1: Bootstrap the bridge script with device-model subcommand

Absorb `tools/aie-device-dump.py` into the new unified bridge. The
device-model subcommand produces identical output to the old script, plus
new fields for memory affinity and edge classification.

**Files:**
- Create: `tools/mlir-aie-bridge.py`
- Delete: `tools/aie-device-dump.py` (after migration)
- Modify: `tools/aie-device-models.json` (regenerated)

**Step 1: Create the bridge script skeleton**

Create `tools/mlir-aie-bridge.py` with:
- Shared mlir-aie Python path bootstrap (copy from aie-device-dump.py lines
  183-228, the `--mlir-aie-path` resolution and import logic)
- argparse with subcommands: `device-model`, `platform-detect`,
  `test-manifest`, `trace-events`
- Only `device-model` implemented initially; others print
  `{"error": "not implemented"}` and exit 1

The device-model subcommand should contain all code from aie-device-dump.py's
`dump_device()` and `classify_tile()` functions, plus these new per-tile
fields:

```python
# In the per-tile loop, add to each tile_map entry:
tile_entry = {
    "col": col, "row": row, "type": tile_type,
    "is_internal": not (col == 0 or col == cols-1 or row == 0 or row == rows-1),
    "edges": {
        "north": row == rows - 1,
        "south": row == 0,
        "east": col == cols - 1,
        "west": col == 0,
    },
}

# In tile_type configs, add memory info:
# For core tiles:
"program_memory_size": 16384,  # 16KB, not queryable from API
# Bank size = local_memory_size / num_banks
"bank_size": model.get_local_memory_size() // max(model.get_num_banks(rep_col, rep_row), 1),
```

Add memory affinity per core tile (which directions have accessible memory):
```python
if tile_type == "core":
    tile_entry["mem_affinity"] = {
        "south": model.is_mem_south(col, row) if hasattr(model, 'is_mem_south') else False,
        "west": model.is_mem_west(col, row) if hasattr(model, 'is_mem_west') else False,
        "north": model.is_mem_north(col, row) if hasattr(model, 'is_mem_north') else False,
        "east": model.is_mem_east(col, row) if hasattr(model, 'is_mem_east') else False,
    }
```

**Step 2: Verify the bridge produces valid output**

```bash
python3 tools/mlir-aie-bridge.py device-model --device npu1 | python3 -m json.tool > /dev/null
echo $?  # should be 0
```

Compare key fields against old output:
```bash
python3 tools/mlir-aie-bridge.py device-model --device npu1 | python3 -c "
import json, sys
d = json.load(sys.stdin)
npu1 = d['devices']['npu1']
assert npu1['columns'] == 5
assert npu1['rows'] == 6
assert npu1['is_npu'] == True
# Check new fields exist
tiles = npu1['tile_map']
core_tile = [t for t in tiles if t['type'] == 'core'][0]
assert 'is_internal' in core_tile
assert 'edges' in core_tile
print('OK: all assertions passed')
"
```

**Step 3: Delete old script, regenerate JSON**

```bash
rm tools/aie-device-dump.py
python3 tools/mlir-aie-bridge.py device-model > tools/aie-device-models.json
```

**Step 4: Verify existing Rust tests still pass**

```bash
cargo test --lib -- device::model
```

The Rust model parser should still work since we only added fields (backward
compatible). If any tests fail, it means the JSON structure changed in a
way the parser doesn't expect -- fix the parser to accept new optional fields.

**Step 5: Commit**

```bash
git add tools/mlir-aie-bridge.py tools/aie-device-models.json
git rm tools/aie-device-dump.py
git commit -m "feat(tools): unified mlir-aie bridge with device-model subcommand

Replaces aie-device-dump.py. Adds per-tile edge classification,
memory affinity, and bank size to device model JSON."
```

---

### Task 2: platform-detect subcommand

Detect available hardware and tools, mirroring mlir-aie's
`lit_config_helpers.py` detection logic.

**Files:**
- Modify: `tools/mlir-aie-bridge.py`

**Step 1: Implement platform-detect**

Add the `platform-detect` subcommand handler. It should:

1. **Detect NPU hardware** via `xrt-smi examine`:
   - Parse output with the same regex as `LitConfigHelper.detect_xrt()`
     (pattern: `r"[\|]?(\[.+:.+:.+\]).+\|(RyzenAI-(npu\d)|NPU ([\w ]+?))\s*\|"`)
   - Map model string to generation using `NPU_MODELS` dict:
     `{"npu1": ["npu1", "Phoenix"], "npu2": ["npu4", "Strix", "npu5", "Strix Halo", "npu6", "Krackan"]}`
   - Map generation to arch: npu1 -> AIE2, npu2 -> AIE2P

2. **Detect tools**:
   - Peano: run `llc -mtriple=aie --version`, check for "Xilinx AI Engine"
   - Chess: `shutil.which("xchesscc")`
   - aiesimulator: `shutil.which("aiesimulator")`

3. **Build feature list** matching mlir-aie lit conventions:
   `["ryzen_ai", "ryzen_ai_npu1", "peano", "chess", "aiesimulator"]`

Output format:
```json
{
  "hardware": {
    "npu_model": "npu1",
    "npu_generation": "Phoenix",
    "arch": "AIE2",
    "device_id": "[0000:c6:00.1]",
    "xrt_found": true
  },
  "tools": {
    "peano": {"found": true, "path": "/path/to/llc"},
    "chess": {"found": true, "path": "/path/to/xchesscc"},
    "aiesimulator": {"found": true, "path": "/path/to/aiesimulator"}
  },
  "features": ["ryzen_ai", "ryzen_ai_npu1", "peano", "chess", "aiesimulator"]
}
```

Handle all error cases gracefully (xrt-smi not found, timeout, no device).

**Step 2: Test**

```bash
python3 tools/mlir-aie-bridge.py platform-detect | python3 -m json.tool
# Should show npu1/Phoenix on our machine
```

**Step 3: Commit**

```bash
git add tools/mlir-aie-bridge.py
git commit -m "feat(tools): add platform-detect subcommand to mlir-aie bridge"
```

---

### Task 3: trace-events subcommand

Export mlir-aie's 4 trace event enum classes as JSON for codegen.

**Files:**
- Modify: `tools/mlir-aie-bridge.py`

**Step 1: Implement trace-events**

Import the event enums from mlir-aie:
```python
from aie.utils.trace.events.aie2 import CoreEvent, MemEvent, MemTileEvent, ShimTileEvent
```

For each enum class, emit `{name: value}` pairs:
```python
def dump_enum(enum_class):
    return {member.name: member.value for member in enum_class}

result = {
    "arch": "aie2",
    "enums": {
        "CoreEvent": dump_enum(CoreEvent),
        "MemEvent": dump_enum(MemEvent),
        "MemTileEvent": dump_enum(MemTileEvent),
        "ShimTileEvent": dump_enum(ShimTileEvent),
    }
}
```

Also extract trace register base addresses from the setup module if
available, or hardcode them from the research:
```python
"trace_registers": {
    "core": {"control0": "0x340D0", "control1": "0x340D4",
             "event_group1": "0x340E0", "event_group2": "0x340E4"},
    "mem":  {"control0": "0x140D0", "control1": "0x140D4",
             "event_group1": "0x140E0", "event_group2": "0x140E4"},
    "memtile": {"control0": "0x940D0", "control1": "0x940D4",
                "event_group1": "0x940E0", "event_group2": "0x940E4"},
}
```

**Step 2: Test**

```bash
python3 tools/mlir-aie-bridge.py trace-events | python3 -c "
import json, sys
d = json.load(sys.stdin)
core = d['enums']['CoreEvent']
assert core['INSTR_VECTOR'] == 37
assert core['MEMORY_STALL'] == 22
assert core['LOCK_STALL'] == 25
mem = d['enums']['MemEvent']
assert 'DMA_S2MM_0_START_TASK' in mem
print(f'OK: {len(core)} core events, {len(mem)} mem events')
"
```

**Step 3: Commit**

```bash
git add tools/mlir-aie-bridge.py
git commit -m "feat(tools): add trace-events subcommand to mlir-aie bridge"
```

---

### Task 4: test-manifest subcommand

Scan test and example directories, extract target device and build
feasibility per test.

**Files:**
- Modify: `tools/mlir-aie-bridge.py`

**Step 1: Implement test-manifest**

Arguments:
```
mlir-aie-bridge.py test-manifest \
    --npu-xrt-dir PATH \    # mlir-aie/test/npu-xrt/
    --examples-dir PATH \   # mlir-aie/programming_examples/
    [--platform FEATURES]   # comma-separated platform features for feasibility
```

For each test directory, extract:

1. **Target device** (priority order):
   - Scan `*.py` for `AIEDevice.npu1` or `AIEDevice.npu2` (regex:
     `r"AIEDevice\.(npu\d[\w]*)"`)
   - Scan `*.mlir` for `aie.device\((npu\d[\w]*)\)` (regex)
   - Check Makefile for `NPU2 ?= 1` or `devicename`
   - Default: `npu1`

2. **REQUIRES features**: scan `*.py` and `run.lit` for
   `# REQUIRES:` lines, parse comma-separated features

3. **Compiler requirements**: check if test uses Chess-only intrinsics
   (presence of `--xchesscc` in build commands or `CHESS=1` in Makefile)

4. **Build feasibility**:
   - `makefile_exists`: Makefile or makefile present
   - `kernel_sources_exist`: `*.cc` or `*.py` generator files present
   - `python_generator_exists`: main Python design file exists
   - `missing_dependencies`: list of what's missing
   - `skip_reason`: from XFAIL/SKIP annotations, or None

Output structure:
```json
{
  "scan_time": "2026-02-27T...",
  "npu_xrt_tests": [...],
  "examples": [...],
  "summary": {
    "total": 150,
    "npu1_only": 120,
    "npu2_only": 15,
    "both": 15,
    "buildable": 100,
    "not_buildable": 50
  }
}
```

**Step 2: Test**

```bash
python3 tools/mlir-aie-bridge.py test-manifest \
    --npu-xrt-dir ../mlir-aie/test/npu-xrt \
    --examples-dir ../mlir-aie/programming_examples \
    | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Tests: {d[\"summary\"][\"total\"]}')
print(f'NPU1: {d[\"summary\"][\"npu1_only\"]}')
print(f'Buildable: {d[\"summary\"][\"buildable\"]}')
# Check a known test
tests = {t['name']: t for t in d['npu_xrt_tests']}
if 'add_one_using_dma' in tests:
    t = tests['add_one_using_dma']
    assert t['target_device'] == 'npu1'
    print(f'add_one_using_dma: device={t[\"target_device\"]}, buildable={t[\"build_feasibility\"][\"makefile_exists\"]}')
print('OK')
"
```

**Step 3: Commit**

```bash
git add tools/mlir-aie-bridge.py
git commit -m "feat(tools): add test-manifest subcommand to mlir-aie bridge"
```

---

### Task 5: Rust bridge invocation module

Create `src/integration/bridge.rs` that wraps subprocess calls to the
Python bridge.

**Files:**
- Create: `src/integration/bridge.rs`
- Modify: `src/integration/mod.rs`

**Step 1: Write tests for bridge invocation**

```rust
// In src/integration/bridge.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invoke_bridge_device_model() {
        // Skip if mlir-aie not available
        let bridge = BridgePath::discover();
        if bridge.is_none() { return; }
        let result = invoke_bridge(&bridge.unwrap(), "device-model", &["--device", "npu1"]);
        assert!(result.is_ok());
        let json = result.unwrap();
        assert!(json["devices"]["npu1"]["columns"].as_u64().unwrap() == 5);
    }

    #[test]
    fn test_invoke_bridge_platform_detect() {
        let bridge = BridgePath::discover();
        if bridge.is_none() { return; }
        let result = invoke_bridge(&bridge.unwrap(), "platform-detect", &[]);
        assert!(result.is_ok());
        let json = result.unwrap();
        assert!(json["features"].is_array());
    }

    #[test]
    fn test_invoke_bridge_trace_events() {
        let bridge = BridgePath::discover();
        if bridge.is_none() { return; }
        let result = invoke_bridge(&bridge.unwrap(), "trace-events", &[]);
        assert!(result.is_ok());
        let json = result.unwrap();
        let core = &json["enums"]["CoreEvent"];
        assert_eq!(core["INSTR_VECTOR"].as_u64().unwrap(), 37);
    }
}
```

**Step 2: Implement bridge invocation**

```rust
//! Subprocess bridge to mlir-aie Python API.
//!
//! Invokes `tools/mlir-aie-bridge.py` and parses JSON output.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Resolved path to the bridge script and its Python interpreter.
pub struct BridgePath {
    pub script: PathBuf,
    pub python: PathBuf,
}

impl BridgePath {
    /// Discover the bridge script relative to the crate root.
    pub fn discover() -> Option<Self> {
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/mlir-aie-bridge.py");
        if !script.exists() { return None; }

        // Try ironenv python first, fall back to system python3
        let ironenv = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .join("mlir-aie/ironenv/bin/python3");
        let python = if ironenv.exists() {
            ironenv
        } else {
            PathBuf::from("python3")
        };

        Some(Self { script, python })
    }
}

/// Invoke a bridge subcommand and parse JSON output.
pub fn invoke_bridge(
    bridge: &BridgePath,
    subcommand: &str,
    args: &[&str],
) -> Result<serde_json::Value, String> {
    let mut cmd = Command::new(&bridge.python);
    cmd.arg(&bridge.script).arg(subcommand);
    for arg in args {
        cmd.arg(arg);
    }

    let output = cmd.output()
        .map_err(|e| format!("Failed to run bridge: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Bridge {} failed: {}", subcommand, stderr));
    }

    serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Bridge {} returned invalid JSON: {}", subcommand, e))
}
```

Add to `src/integration/mod.rs`:
```rust
pub mod bridge;
```

**Step 3: Run tests**

```bash
cargo test --lib -- integration::bridge
```

**Step 4: Commit**

```bash
git add src/integration/bridge.rs src/integration/mod.rs
git commit -m "feat(integration): add Rust bridge module for mlir-aie Python API"
```

---

### Task 6: Build-time trace event codegen

Add `build.rs` step that calls the bridge to generate Rust trace event
code and name lookup tables.

**Files:**
- Modify: `build.rs`
- Generated: `$OUT_DIR/trace_event_codes.rs`
- Modify: `src/trace/mod.rs` (include generated code)

**Step 1: Add codegen to build.rs**

Add a function to build.rs that:
1. Runs `tools/mlir-aie-bridge.py trace-events`
2. Parses the JSON
3. Generates a Rust source file with:
   - `pub mod core_events { pub const INSTR_VECTOR: u8 = 37; ... }`
   - `pub mod mem_events { pub const DMA_S2MM_0_START_TASK: u8 = 18; ... }`
   - `pub fn core_event_name(code: u8) -> &'static str { ... }` (lookup table)
   - `pub fn mem_event_name(code: u8) -> &'static str { ... }`
   - Same for memtile and shim

The function should gracefully skip codegen if the bridge is not available
(mlir-aie not installed), leaving a stub file with empty arrays and a
warning comment.

```rust
fn generate_trace_events(out_dir: &Path) {
    let bridge = Path::new("tools/mlir-aie-bridge.py");
    if !bridge.exists() {
        // Write stub
        let stub = "// Trace event codes not generated (mlir-aie bridge not available)\n\
                     pub fn core_event_name(_code: u8) -> &'static str { \"UNKNOWN\" }\n\
                     pub fn mem_event_name(_code: u8) -> &'static str { \"UNKNOWN\" }\n\
                     pub fn memtile_event_name(_code: u8) -> &'static str { \"UNKNOWN\" }\n\
                     pub fn shim_event_name(_code: u8) -> &'static str { \"UNKNOWN\" }\n";
        std::fs::write(out_dir.join("trace_event_codes.rs"), stub).unwrap();
        return;
    }

    // Run bridge
    let output = Command::new("python3")
        .arg(bridge)
        .arg("trace-events")
        .output();

    // Parse JSON, generate lookup tables...
    // For each enum, generate:
    //   pub const NAME: u8 = VALUE;
    //   pub fn TYPE_event_name(code: u8) -> &'static str { match code { ... } }
}
```

**Step 2: Include generated code in src/trace/mod.rs**

```rust
/// Generated trace event codes from mlir-aie's canonical enums.
pub mod event_codes {
    include!(concat!(env!("OUT_DIR"), "/trace_event_codes.rs"));
}
```

**Step 3: Verify build works**

```bash
cargo build
cargo test --lib -- trace
```

**Step 4: Commit**

```bash
git add build.rs src/trace/mod.rs
git commit -m "feat(trace): build-time codegen for trace event codes from mlir-aie"
```

---

### Task 7: Integrate platform-detect into test runner

Wire the bridge's platform detection into the test runner for pre-flight
hardware/tool checking.

**Files:**
- Modify: `src/testing/emu_runner.rs`
- Modify: `src/integration/bridge.rs` (add typed PlatformInfo struct)

**Step 1: Add PlatformInfo struct to bridge.rs**

```rust
/// Detected platform capabilities from mlir-aie bridge.
#[derive(Debug)]
pub struct PlatformInfo {
    pub npu_model: Option<String>,    // "npu1", "npu2"
    pub arch: Option<String>,         // "AIE2", "AIE2P"
    pub features: Vec<String>,        // ["ryzen_ai", "ryzen_ai_npu1", ...]
    pub has_peano: bool,
    pub has_chess: bool,
    pub has_aiesimulator: bool,
}

impl PlatformInfo {
    pub fn from_bridge(bridge: &BridgePath) -> Result<Self, String> {
        let json = invoke_bridge(bridge, "platform-detect", &[])?;
        // Parse fields from JSON...
        Ok(Self { ... })
    }

    pub fn supports_device(&self, target_device: &str) -> bool {
        match self.npu_model.as_deref() {
            Some("npu1") => target_device.starts_with("npu1"),
            Some("npu2") => target_device.starts_with("npu2"),
            _ => false,
        }
    }
}
```

**Step 2: Use in test runner startup**

In `emu_runner.rs`, at the start of the test run, invoke platform-detect
and print a summary:

```rust
if let Some(bridge) = BridgePath::discover() {
    match PlatformInfo::from_bridge(&bridge) {
        Ok(platform) => {
            println!("Platform: {} ({}) -- features: {}",
                platform.npu_model.as_deref().unwrap_or("unknown"),
                platform.arch.as_deref().unwrap_or("unknown"),
                platform.features.join(", "));
        }
        Err(e) => {
            eprintln!("Warning: platform detection failed: {}", e);
        }
    }
}
```

This is informational for now. Task 8 will use it for filtering.

**Step 3: Test**

```bash
cargo run --bin npu-test -- -v add_one 2>&1 | head -5
# Should show "Platform: npu1 (AIE2) -- features: ..."
```

**Step 4: Commit**

```bash
git add src/integration/bridge.rs src/testing/emu_runner.rs
git commit -m "feat(testing): integrate platform detection from mlir-aie bridge"
```

---

### Task 8: Integrate test-manifest for pre-flight filtering

Use the manifest to skip tests that can't run on this platform or can't
be built with available tools.

**Files:**
- Modify: `src/testing/emu_runner.rs`
- Modify: `src/integration/bridge.rs` (add TestManifest struct)

**Step 1: Add TestManifest types to bridge.rs**

```rust
#[derive(Debug)]
pub struct TestEntry {
    pub name: String,
    pub target_device: String,
    pub target_arch: String,
    pub requires: Vec<String>,
    pub build_feasibility: BuildFeasibility,
    pub skip_reason: Option<String>,
}

#[derive(Debug)]
pub struct BuildFeasibility {
    pub makefile_exists: bool,
    pub kernel_sources_exist: bool,
    pub missing_dependencies: Vec<String>,
}

impl BuildFeasibility {
    pub fn is_buildable(&self) -> bool {
        self.makefile_exists && self.kernel_sources_exist && self.missing_dependencies.is_empty()
    }
}

#[derive(Debug)]
pub struct TestManifest {
    pub tests: Vec<TestEntry>,
}

impl TestManifest {
    pub fn from_bridge(bridge: &BridgePath, npu_xrt_dir: &Path, examples_dir: &Path) -> Result<Self, String> {
        let json = invoke_bridge(bridge, "test-manifest", &[
            "--npu-xrt-dir", &npu_xrt_dir.to_string_lossy(),
            "--examples-dir", &examples_dir.to_string_lossy(),
        ])?;
        // Parse...
        Ok(Self { tests })
    }
}
```

**Step 2: Use manifest in test runner**

Before the build phase, load the manifest and use it to:
1. Skip tests targeting wrong platform (npu2 tests on npu1 hardware)
2. Skip tests with missing build prerequisites
3. Report WHY each test was skipped

```rust
// In the test discovery/filtering phase:
for test in &manifest.tests {
    if !platform.supports_device(&test.target_device) {
        report_skip(&test.name, &format!(
            "requires {} (have {})",
            test.target_device,
            platform.npu_model.as_deref().unwrap_or("none")
        ));
        continue;
    }
    if !test.build_feasibility.is_buildable() {
        report_skip(&test.name, &format!(
            "not buildable: {}",
            test.build_feasibility.missing_dependencies.join(", ")
        ));
        continue;
    }
    // Add to runnable tests...
}
```

**Step 3: Test**

```bash
cargo run --bin npu-test -- --suite=all --compiler=all -v 2>&1 | grep -i skip | head -20
# Should show platform-aware skip reasons instead of cryptic "exit 2"
```

**Step 4: Commit**

```bash
git add src/integration/bridge.rs src/testing/emu_runner.rs
git commit -m "feat(testing): manifest-driven test filtering with skip diagnostics"
```

---

### Task 9: Align trace event names with mlir-aie canonical names

Update our emulator's trace export to use the exact event names from
mlir-aie's enums (e.g., `DMA_S2MM_0_START_TASK` instead of
`DMA_START_TASK`).

**Files:**
- Modify: `src/trace/mod.rs` (event name generation)
- Modify: `src/interpreter/state/context.rs` (if EventType names live here)

**Step 1: Update DMA event names**

Change `DmaStartTask { channel }` to emit channel-qualified names:
- Channel 0 S2MM -> `DMA_S2MM_0_START_TASK`
- Channel 1 S2MM -> `DMA_S2MM_1_START_TASK`
- Channel 0 MM2S -> `DMA_MM2S_0_START_TASK`
- Channel 1 MM2S -> `DMA_MM2S_1_START_TASK`

Same pattern for `DmaFinishedBd`, `DmaFinishedTask`, `DmaStalledLock`,
`DmaStreamStarvation`.

**Step 2: Update port event names**

Change `PortIdle { port }` to emit `PORT_IDLE_0`, `PORT_IDLE_1`, etc.
Same for `PortRunning`, `PortStalled`, `PortTlast`.

**Step 3: Update lock event names**

Change `LockAcquire { lock_id }` to emit `LOCK_SEL{n}_ACQ_EQ` or similar
matching mlir-aie's naming. Check the generated event code tables for
exact names.

**Step 4: Run tests**

```bash
cargo test --lib -- trace
```

Existing tests that check event names will need updating to match the
new canonical names.

**Step 5: Commit**

```bash
git add src/trace/mod.rs src/interpreter/state/context.rs
git commit -m "feat(trace): align event names with mlir-aie canonical enums"
```

---

### Task 10: Runtime trace event validation

Add an optional validation step that re-queries the bridge at test-runner
startup and compares against compiled-in event tables.

**Files:**
- Modify: `src/trace/mod.rs`
- Modify: `src/integration/bridge.rs`

**Step 1: Add validation function**

```rust
/// Check that compiled-in event tables match current mlir-aie.
/// Prints warnings for any mismatches (version drift).
pub fn validate_trace_events(bridge: &BridgePath) -> Result<(), String> {
    let json = invoke_bridge(bridge, "trace-events", &[])?;
    let core_events = &json["enums"]["CoreEvent"];

    // Spot-check key events
    let checks = [
        ("INSTR_VECTOR", event_codes::core_events::INSTR_VECTOR),
        ("MEMORY_STALL", event_codes::core_events::MEMORY_STALL),
        // ... more
    ];

    for (name, expected_code) in &checks {
        if let Some(actual) = core_events[name].as_u64() {
            if actual as u8 != *expected_code {
                eprintln!("WARNING: trace event {} code mismatch: compiled={} mlir-aie={}",
                    name, expected_code, actual);
            }
        }
    }
    Ok(())
}
```

**Step 2: Wire into test runner startup (optional, behind --validate flag)**

**Step 3: Test and commit**

```bash
cargo test --lib -- trace
git add src/trace/mod.rs src/integration/bridge.rs
git commit -m "feat(trace): runtime validation of trace event codes against mlir-aie"
```

---

### Task 11: Verify end-to-end

**Step 1: Run all bridge subcommands manually**

```bash
python3 tools/mlir-aie-bridge.py device-model --device npu1 | python3 -m json.tool | head -20
python3 tools/mlir-aie-bridge.py platform-detect | python3 -m json.tool
python3 tools/mlir-aie-bridge.py trace-events | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['enums']['CoreEvent']), 'core events')"
python3 tools/mlir-aie-bridge.py test-manifest --npu-xrt-dir ../mlir-aie/test/npu-xrt --examples-dir ../mlir-aie/programming_examples | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['summary'])"
```

**Step 2: Run full test suite**

```bash
cargo test --lib
```

**Step 3: Run test runner with verbose output**

```bash
cargo run --bin npu-test -- --suite=all --compiler=all -v add_one 2>&1 | head -20
# Verify platform detection line appears
```

**Step 4: Check that npu2 tests are properly skipped**

```bash
cargo run --bin npu-test -- --suite=all -v 2>&1 | grep "requires npu2"
# Should show skip messages for npu2-targeted tests
```
