# Build Toolchain & PADDB Unification Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the build toolchain so `cargo build` automatically rebuilds and installs the XRT plugin, kill the XDNA_EMU_LIB zombie env var, and unify PADDB execution to a single handler via a decoder fix.

**Architecture:** The C++ plugin doesn't link against the Rust lib (it uses dlopen at runtime), so cmake can run from build.rs without circular dependency. The Rust lib is found at runtime via XDNA_EMU_DIR/target/$profile/. PADDB unification happens in the decoder (build_slot_op) by forcing SemanticOp::PointerAdd when is_ptr_arithmetic is true.

**Tech Stack:** Rust (cargo build.rs), CMake, C++ (XRT plugin), bash (activate-npu-env.sh)

---

## Chunk 1: Infrastructure (Tasks 1-3)

### Task 1: Own /opt/xilinx/xrt/lib/ and kill XDNA_EMU_LIB

**Files:**
- Modify: `xrt-plugin/src/pdev_emu.cpp:17-46`
- Modify: `../toolchain-build/activate-npu-env.sh:55-60`

- [ ] **Step 1: Change ownership of XRT lib directory**

```bash
pkexec chown -R triple:triple /opt/xilinx/xrt/lib/
```

Verify: `ls -la /opt/xilinx/xrt/lib/ | head -3` shows `triple triple`.

- [ ] **Step 2: Remove XDNA_EMU_LIB from plugin**

In `xrt-plugin/src/pdev_emu.cpp`, replace lines 17-46 with:

```cpp
  const std::lock_guard<std::mutex> lock(m_lock);

  // Determine library path.
  //
  // Resolution order:
  //   1. XDNA_EMU_DIR + XDNA_EMU profile -- e.g. $XDNA_EMU_DIR/target/debug/libxdna_emu.so
  //      XDNA_EMU="debug" or "release" selects the Cargo profile.
  //      XDNA_EMU="1" (or any other truthy value) defaults to "debug".
  //   2. Plain dlopen fallback -- "libxdna_emu.so" via ldconfig/LD_LIBRARY_PATH
  std::string lib_path;
  const char* dir_env = std::getenv("XDNA_EMU_DIR");
  const char* emu_env = std::getenv("XDNA_EMU");
  std::string profile = "debug";  // default
  if (emu_env) {
    std::string val(emu_env);
    if (val == "release")
      profile = "release";
    // "debug", "1", or any other truthy value -> debug
  }
  if (dir_env && dir_env[0] != '\0') {
    lib_path = std::string(dir_env) + "/target/" + profile + "/libxdna_emu.so";
  } else {
    lib_path = "libxdna_emu.so";
  }
  EMU_INFO("Loading emulator library: %s (profile=%s)", lib_path.c_str(), profile.c_str());
```

- [ ] **Step 3: Clean up activate-npu-env.sh**

In `../toolchain-build/activate-npu-env.sh`, replace lines 55-60:

```bash
# Plugin finds the Rust emulator library via XDNA_EMU_DIR + XDNA_EMU profile.
# Usage:  XDNA_EMU=debug ./test.exe    (loads debug lib -- default)
#         XDNA_EMU=release ./test.exe   (loads release lib)
#         XDNA_EMU=1 ./test.exe         (loads debug lib)
export XDNA_EMU_DIR="$NPU_WORK_DIR/xdna-emu"
```

Remove the `unset XDNA_EMU_LIB` line and the `# Override:` comment (nothing to override).

- [ ] **Step 4: Rebuild plugin with new code and verify**

```bash
cd xrt-plugin/build && cmake --build . && cp libxrt_driver_emu.so.2.21.0 /opt/xilinx/xrt/lib/ && cp -P libxrt_driver_emu.so.2 /opt/xilinx/xrt/lib/
```

Verify: `strings /opt/xilinx/xrt/lib/libxrt_driver_emu.so.2 | grep XDNA_EMU_LIB` returns nothing.

- [ ] **Step 5: Commit**

```bash
git add xrt-plugin/src/pdev_emu.cpp
git commit -m "$(cat <<'EOF'
fix(plugin): remove XDNA_EMU_LIB, simplify library resolution

XDNA_EMU_LIB was a full-path override that persisted in shells across
sessions, silently loading the wrong (release) library during debug
work. Remove it entirely. Library resolution now uses only:
1. XDNA_EMU_DIR + XDNA_EMU profile (debug/release)
2. Plain dlopen fallback
EOF
)"
```

Note: activate-npu-env.sh is outside the xdna-emu repo. Commit separately in npu-work if desired.

---

### Task 2: Cargo post-build plugin integration

**Files:**
- Modify: `build.rs` (add post-build cmake + install)
- Modify: `scripts/rebuild-plugin.sh` (simplify to thin wrapper)

The C++ plugin loads libxdna_emu.so at runtime via dlopen (no link-time
dependency), so cmake can run from build.rs without circular issues. The
Rust lib is found at runtime via XDNA_EMU_DIR -- no copy needed.

- [ ] **Step 1: Add plugin build to build.rs**

At the end of `fn main()` in `build.rs`, after all code generation, add:

```rust
    // ========================================================================
    // Post-codegen: rebuild and install XRT plugin
    // ========================================================================
    //
    // The C++ plugin (libxrt_driver_emu.so) loads the Rust emulator at
    // runtime via dlopen -- there is no link-time dependency. This lets
    // us build and install the plugin from build.rs without circularity.
    //
    // The plugin .so is installed to /opt/xilinx/xrt/lib/ so XRT can
    // find it. The Rust lib is NOT copied -- the plugin resolves it at
    // runtime via XDNA_EMU_DIR/target/$profile/libxdna_emu.so.

    // Rebuild triggers for plugin C++ sources
    let plugin_src = manifest_dir.join("xrt-plugin/src");
    if plugin_src.exists() {
        for entry in fs::read_dir(&plugin_src).unwrap() {
            let entry = entry.unwrap();
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
        println!("cargo:rerun-if-changed=xrt-plugin/CMakeLists.txt");
    }

    // Only build the plugin if the cmake build directory exists.
    // First-time setup still requires: mkdir -p xrt-plugin/build && cd xrt-plugin/build && cmake ..
    let plugin_build = manifest_dir.join("xrt-plugin/build");
    let xrt_lib = Path::new("/opt/xilinx/xrt/lib");
    if plugin_build.join("CMakeCache.txt").exists() && xrt_lib.exists() {
        // Incremental cmake build (~2s when nothing changed)
        let status = std::process::Command::new("cmake")
            .args(["--build", "."])
            .current_dir(&plugin_build)
            .status();

        match status {
            Ok(s) if s.success() => {
                // Install plugin .so to XRT lib directory
                let src = plugin_build.join("libxrt_driver_emu.so.2.21.0");
                let dst = xrt_lib.join("libxrt_driver_emu.so.2.21.0");
                let link = xrt_lib.join("libxrt_driver_emu.so.2");
                if src.exists() {
                    if let Err(e) = fs::copy(&src, &dst) {
                        println!("cargo:warning=Plugin install failed: {e}");
                    } else {
                        // Create/update symlink
                        let _ = fs::remove_file(&link);
                        #[cfg(unix)]
                        {
                            use std::os::unix::fs::symlink;
                            let _ = symlink("libxrt_driver_emu.so.2.21.0", &link);
                        }
                    }
                }
            }
            Ok(s) => {
                println!("cargo:warning=Plugin cmake build failed (exit {})", s.code().unwrap_or(-1));
            }
            Err(e) => {
                println!("cargo:warning=Plugin cmake build failed: {e}");
            }
        }
    }
```

- [ ] **Step 2: Simplify rebuild-plugin.sh**

Replace the entire script with a thin wrapper:

```bash
#!/usr/bin/env bash
# rebuild-plugin.sh -- Build emulator + XRT plugin.
#
# This is a convenience wrapper around cargo build. The build.rs script
# handles cmake and plugin installation automatically.
#
# Usage:
#   ./scripts/rebuild-plugin.sh             # Debug build (default)
#   ./scripts/rebuild-plugin.sh --release   # Release build
#   ./scripts/rebuild-plugin.sh --reconfigure  # Force cmake reconfigure
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EMU_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$EMU_DIR/xrt-plugin/build"

CARGO_FLAGS=""
for arg in "$@"; do
  case "$arg" in
    --release) CARGO_FLAGS="--release" ;;
    --reconfigure)
      echo ">>> Reconfiguring cmake..."
      rm -rf "$BUILD_DIR/CMakeCache.txt"
      mkdir -p "$BUILD_DIR"
      ( cd "$BUILD_DIR" && cmake .. )
      ;;
  esac
done

# Ensure cmake is configured (first-time setup)
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  echo ">>> First-time cmake configure..."
  mkdir -p "$BUILD_DIR"
  ( cd "$BUILD_DIR" && cmake .. )
fi

echo ">>> Building (cargo build handles plugin automatically)..."
nice -n 19 cargo build $CARGO_FLAGS

PROFILE="debug"
[[ "$CARGO_FLAGS" == *"--release"* ]] && PROFILE="release"
echo ">>> Done. Rust lib: $EMU_DIR/target/$PROFILE/libxdna_emu.so"
```

- [ ] **Step 3: Verify end-to-end**

```bash
# Touch a plugin source file to force rebuild
touch xrt-plugin/src/pdev_emu.cpp
cargo build 2>&1 | grep -E "warning=|Built target|Compiling"
```

Expected: cmake runs as part of cargo build, plugin .so is installed.

```bash
ls -la /opt/xilinx/xrt/lib/libxrt_driver_emu.so.2*
```

Expected: freshly timestamped files.

- [ ] **Step 4: Commit**

```bash
git add build.rs scripts/rebuild-plugin.sh
git commit -m "$(cat <<'EOF'
feat(build): auto-rebuild XRT plugin from cargo build

build.rs now runs cmake --build and installs the plugin .so to
/opt/xilinx/xrt/lib/ as part of every cargo build. This eliminates
the manual rebuild-plugin.sh step that caused stale plugin bugs.

rebuild-plugin.sh simplified to a thin wrapper around cargo build.
EOF
)"
```

---

### Task 3: Fix PADDB in the decoder

**Files:**
- Modify: `src/interpreter/decode/decoder.rs:799-806` (force PointerAdd semantic)
- Modify: `src/interpreter/execute/semantic.rs:296-324` (delete PADDB fallback from execute_add)
- Modify: `src/interpreter/execute/semantic.rs:593-614` (remove debug logging from execute_pointer_add)

- [ ] **Step 1: Force PointerAdd semantic for all ptr arithmetic in decoder**

In `src/interpreter/decode/decoder.rs`, function `build_slot_op`, replace lines 805-806:

```rust
        // Build SlotOp directly from SemanticOp (no Operation bridge)
        let effective_semantic = semantic_override.or(enc.semantic);
```

With:

```rust
        // Build SlotOp directly from SemanticOp (no Operation bridge).
        // Force PointerAdd for all pointer arithmetic instructions.
        // Some PADDB/PADDA variants get SemanticOp::Add from pattern
        // matching instead of PointerAdd. The is_ptr_arithmetic flag
        // (derived from mnemonic "padd*") is the reliable indicator.
        let effective_semantic = if enc.is_ptr_arithmetic {
            Some(SemanticOp::PointerAdd)
        } else {
            semantic_override.or(enc.semantic)
        };
```

- [ ] **Step 2: Delete PADDB fallback from execute_add**

In `src/interpreter/execute/semantic.rs`, replace lines 296-324 (the entire
PADDB detection block at the top of execute_add) with:

```rust
fn execute_add(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    // All PADDB/PADDA instructions should reach execute_pointer_add via
    // SemanticOp::PointerAdd (forced by the decoder for is_ptr_arithmetic
    // encodings). If a pointer arithmetic instruction reaches execute_add,
    // that is a decoder classification bug.
    if matches!(op.slot, SlotIndex::LoadA | SlotIndex::LoadB)
        && op.sources.len() == 1
        && matches!(op.dest, Some(Operand::PointerReg(_)) | None)
    {
        log::warn!(
            "[BUG] Pointer arithmetic reached execute_add instead of execute_pointer_add: \
             pc=0x{:03X} slot={:?} dest={:?} srcs={:?} name={:?}",
            ctx.pc(), op.slot, op.dest, op.sources, op.encoding_name
        );
    }

    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let result = a.wrapping_add(b);
    write_dest(op, ctx, result);
    // AIE2: ADD sets the Carry flag (C). Z/N/V computed by branch logic.
```

This preserves the regular Add handler but replaces the broken PADDB
fallback with a diagnostic warning.

- [ ] **Step 3: Run tests**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -5
```

Expected: All tests pass. The existing PADDB tests in semantic.rs should
still work because they set `semantic = Some(SemanticOp::PointerAdd)`.

- [ ] **Step 4: Commit**

```bash
git add src/interpreter/decode/decoder.rs src/interpreter/execute/semantic.rs
git commit -m "$(cat <<'EOF'
fix(decoder): unify PADDB to single handler via is_ptr_arithmetic

Force SemanticOp::PointerAdd for all encodings with is_ptr_arithmetic
in the decoder's build_slot_op. This ensures every PADDB/PADDA variant
routes through execute_pointer_add regardless of what the TableGen
pattern inference assigned.

Delete the broken PADDB fallback from execute_add that incorrectly
defaulted dest=None to p6 (should be SP). Replace with a diagnostic
warning that fires if a pointer arithmetic instruction ever reaches
execute_add (indicating a decoder classification bug).
EOF
)"
```

---

## Chunk 2: Cleanup (Task 4)

### Task 4: Remove debug instrumentation

**Files:**
- Modify: `src/interpreter/execute/cycle_accurate.rs:118-122`
- Modify: `src/interpreter/execute/semantic.rs:593-614`
- Modify: `src/interpreter/execute/memory.rs:996-1000`
- Modify: `src/npu/executor.rs:632-635`

- [ ] **Step 1: Remove [DISPATCH] logging from cycle_accurate.rs**

Delete the temporary dispatch logging block at lines 118-122
(the `if ctx.cycles < 20 { log::info!("[DISPATCH]" ...` block).

- [ ] **Step 2: Remove [PADDB] logging from execute_pointer_add**

In `src/interpreter/execute/semantic.rs`, remove the info-level logging
from execute_pointer_add (the `log::info!("[PADDB]"` calls). Keep the
function logic intact.

- [ ] **Step 3: Remove [DBG POST-MOD] logging from memory.rs**

In `src/interpreter/execute/memory.rs`, remove the debug p6 tracking
block around line 997-1000 (the `if ptr_reg == Some(6) ...` block).

- [ ] **Step 4: Revert maskwrite logging to debug level**

In `src/npu/executor.rs`, revert the `execute_maskwrite` logging from
`log::info!` back to `log::debug!` (it was elevated during investigation).

- [ ] **Step 5: Run tests and verify clean build**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -5
```

Expected: All tests pass, no warnings about unused variables.

- [ ] **Step 6: Commit**

```bash
git add src/interpreter/execute/cycle_accurate.rs src/interpreter/execute/semantic.rs src/interpreter/execute/memory.rs src/npu/executor.rs
git commit -m "$(cat <<'EOF'
chore: remove debug instrumentation from add_maskwrite investigation

Strip temporary logging added during the PADDB/maskwrite debugging
session: [DISPATCH], [PADDB], [DBG POST-MOD], and elevated maskwrite
info logging. The underlying bugs are now fixed in the decoder and
build toolchain.
EOF
)"
```
