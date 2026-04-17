# Build Toolchain & PADDB Unification Design

Date: 2026-03-10

## Problem Statement

Three infrastructure problems are compounding to waste significant debugging
time:

1. **XDNA_EMU_LIB zombie**: The old environment variable persists in shells
   that sourced activate-npu-env.sh before the unset was added. It silently
   overrides all other library resolution, causing debug sessions to run
   against release builds. This has wasted hours across multiple sessions.

2. **Disconnected build systems**: The Rust library and C++ plugin are
   separate build systems with no dependency link. Changing Rust code
   requires manually running rebuild-plugin.sh AND reinstalling. Forgetting
   any step means debugging the wrong binary.

3. **Two PADDB execution handlers**: `execute_add` (SemanticOp::Add) and
   `execute_pointer_add` (SemanticOp::PointerAdd) both handle PADDB
   instructions. The decoder assigns different semantics depending on the
   encoding variant, so some PADDBs silently go through the wrong handler.
   The `execute_add` handler has a broken `None => 6` fallback that
   assumes SP is p6.

## Design

### 1. Kill XDNA_EMU_LIB

Remove XDNA_EMU_LIB support entirely from the plugin. No deprecation path,
no warnings -- just delete it.

**pdev_emu.cpp**: Remove the XDNA_EMU_LIB check. Library resolution becomes:
1. `$XDNA_EMU_DIR/target/$profile/libxdna_emu.so` (profile from XDNA_EMU)
2. Fallback: `dlopen("libxdna_emu.so")` via system paths

**activate-npu-env.sh**: Remove the `unset XDNA_EMU_LIB` line (nothing to
unset). Keep XDNA_EMU_DIR export.

### 2. Own /opt/xilinx/xrt/lib/

Change ownership of `/opt/xilinx/xrt/lib/` to the dev user. This eliminates
the need for pkexec/sudo during plugin installation. The directory only
contains our plugin and XRT runtime libs installed by us.

```bash
pkexec chown -R triple:triple /opt/xilinx/xrt/lib/
```

### 3. Cargo post-build auto-rebuilds plugin

Add a post-build step to `build.rs` that:

1. Runs `cmake --build xrt-plugin/build` (incremental, ~2s when nothing
   changed in C++ sources)
2. Copies `libxrt_driver_emu.so.2` to `/opt/xilinx/xrt/lib/`
3. Copies the Rust `libxdna_emu.so` to `/opt/xilinx/xrt/lib/` for the
   plain dlopen fallback path

Uses `cargo:rerun-if-changed` on `xrt-plugin/src/` files to avoid
unnecessary cmake invocations when only Rust code changed.

**rebuild-plugin.sh**: Simplified to a thin wrapper around `cargo build`.
Retains `--release` and `--reconfigure` flags.

### 4. Fix PADDB in the decoder

**Root cause**: Some PADDB encoding variants receive `SemanticOp::Add`
instead of `SemanticOp::PointerAdd` from the semantic inference pipeline.
The TableGen data is correct (confirmed by assertion), but not all encoding
variants inherit the semantic through the pseudo-expansion chain.

**Fix**: In the decoder's semantic assignment phase, if an encoding has
`is_ptr_arithmetic = true`, force `semantic = SemanticOp::PointerAdd`.
This field is already populated on InstrDef from TableGen.

**Cleanup**:
- Delete the PADDB fallback from `execute_add` (lines 309-321)
- Add a debug assertion: if `execute_add` sees a LoadA/LoadB slot with
  single source and pointer dest, log a warning (decoder classification bug)
- `execute_pointer_add` becomes the single PADDB handler (already has the
  dest-inference fix from Bug 2)

### 5. Remove debug instrumentation

Strip all temporary logging added during the add_maskwrite investigation:
- `[DISPATCH]`, `[PADDB]`, `[PADDB-via-Add]` in semantic.rs / cycle_accurate.rs
- `[DBG POST-MOD]` in memory.rs
- `[WATCH-LD]` and `[WATCH-ST]` info-level watchpoints in memory.rs
- Debug watchpoints in ffi/mod.rs

## Verification

After all changes:
1. `cargo build` automatically rebuilds and installs the plugin
2. `cargo test --lib` passes (no regressions from PADDB unification)
3. Fresh shell (re-source activate-npu-env.sh): XDNA_EMU_LIB is gone,
   XDNA_EMU=debug loads debug lib, XDNA_EMU=release loads release lib
4. Bridge test `add_maskwrite` loads the correct debug lib and PADDB
   instructions execute through the unified handler
