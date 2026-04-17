# Dual-Compiler Bridge Test Infrastructure

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Compile and run every bridge test with both Chess (ground truth) and Peano, producing a per-compiler comparison matrix.

**Architecture:** Transform run.lit commands per-compiler. Chess builds use xchesscc + xbridge. Peano builds replace xchesscc_wrapper with Peano clang and strip --xchesscc/--xbridge flags. Shared test.exe, separate xclbin builds, parallel results.

**Tech Stack:** Bash (bridge script), Python (trace tools), aiecc.py, Peano clang++, xchesscc_wrapper

---

## Background

### Current State
- Bridge script (`scripts/emu-bridge-test.sh`) runs run.lit commands verbatim
- Tests written for Chess fail when built with Peano (linker incompatibility)
- 5 tests are stuck as `compile_failed` because of this
- No way to compare Chess vs Peano results side-by-side

### Compiler Command Equivalents

| Operation | Chess | Peano |
|-----------|-------|-------|
| Kernel compile | `xchesscc_wrapper aie2 -I $AIETOOLS/include -c k.cc -o k.o` | `$PEANO/bin/clang++ -O2 -std=c++20 --target=aie2-none-unknown-elf -DNDEBUG -I $MLIR_AIE/install/include -c k.cc -o k.o` |
| xclbin build | `aiecc.py --xchesscc --xbridge --no-aiesim --aie-generate-xclbin ...` | `aiecc.py --no-xchesscc --no-aiesim --aie-generate-xclbin ...` |
| Include path | `$AIETOOLS/include` (aie_api lives here) | `$MLIR_AIE/install/include` (aie_api lives here) |

### Peano Warning Flags (from makefile-common)
```
-Wno-parentheses -Wno-attributes -Wno-macro-redefined
-Wno-empty-body -Wno-missing-template-arg-list-after-template-kw
```

---

## Task 1: Add compiler mode CLI flags and environment detection

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (option parsing section, ~lines 60-119)

**Step 1: Add CLI flags**

Add three new options after the existing option parsing:
```bash
COMPILER_MODE="both"  # "both", "chess", "peano"

# In the case statement:
--chess-only)   COMPILER_MODE="chess"; shift ;;
--peano-only)   COMPILER_MODE="peano"; shift ;;
--chess)        COMPILER_MODE="chess"; shift ;;
--peano)        COMPILER_MODE="peano"; shift ;;
```

Add to help text:
```
  --chess-only    Only compile/run with Chess compiler
  --peano-only    Only compile/run with Peano compiler
  (default: both compilers)
```

**Step 2: Add compiler path detection**

After the configuration section (~line 54), add auto-detection:
```bash
# Peano compiler paths (auto-detect)
PEANO_CLANG="${PEANO_INSTALL_DIR:-/home/triple/npu-work/llvm-aie/install}/bin/clang++"
PEANO_INCLUDE="${MLIR_AIE_INSTALL_DIR:-$MLIR_AIE/install}/include"

# Chess compiler paths
CHESS_INCLUDE="${AIETOOLS_DIR}/include"

# Validate compilers are available
if [[ "$COMPILER_MODE" != "peano" ]] && ! command -v xchesscc_wrapper &>/dev/null; then
  echo "Warning: xchesscc_wrapper not found, Chess builds will fail" >&2
fi
if [[ "$COMPILER_MODE" != "chess" ]] && [[ ! -x "$PEANO_CLANG" ]]; then
  echo "Warning: Peano clang not found at $PEANO_CLANG" >&2
fi

# Peano kernel compilation flags
PEANO_KERNEL_FLAGS="-O2 -std=c++20 --target=aie2-none-unknown-elf -DNDEBUG"
PEANO_KERNEL_FLAGS+=" -Wno-parentheses -Wno-attributes -Wno-macro-redefined"
PEANO_KERNEL_FLAGS+=" -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw"
```

**Step 3: Export new variables**

```bash
export COMPILER_MODE PEANO_CLANG PEANO_INCLUDE CHESS_INCLUDE PEANO_KERNEL_FLAGS
```

**Step 4: Verify**

```bash
./scripts/emu-bridge-test.sh --help  # Shows new options
./scripts/emu-bridge-test.sh --list --chess-only  # Lists tests, no crash
```

---

## Task 2: Command transformation functions

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (new functions after extract_build_commands)

**Step 1: Write transform_for_chess()**

This function takes a build command and ensures it uses Chess:
```bash
transform_for_chess() {
  local cmd="$1"

  # xchesscc_wrapper commands: keep as-is (already Chess)
  if [[ "$cmd" == *xchesscc_wrapper* ]]; then
    echo "$cmd"
    return
  fi

  # aiecc.py commands: ensure --xchesscc --xbridge are present
  if [[ "$cmd" == *aiecc.py* ]]; then
    # Strip any existing --no-xchesscc or --no-xbridge
    cmd="${cmd//--no-xchesscc/}"
    cmd="${cmd//--no-xbridge/}"
    # Add --xchesscc --xbridge if not already present
    if [[ "$cmd" != *"--xchesscc"* ]]; then
      cmd="${cmd/aiecc.py/aiecc.py --xchesscc}"
    fi
    if [[ "$cmd" != *"--xbridge"* ]]; then
      cmd="${cmd/aiecc.py/aiecc.py --xbridge}"
    fi
    echo "$cmd"
    return
  fi

  # Other commands: pass through unchanged
  echo "$cmd"
}
```

**Step 2: Write transform_for_peano()**

This function transforms commands for Peano compilation:
```bash
transform_for_peano() {
  local cmd="$1"
  local src_dir="$2"

  # xchesscc_wrapper commands: replace with Peano clang
  if [[ "$cmd" == *xchesscc_wrapper* ]]; then
    # Extract the -c source and -o output from the command
    # Pattern: xchesscc_wrapper aie2 [flags] -c source.cc -o output.o
    local source output
    source="$(echo "$cmd" | grep -oP '(?<=-c\s)\S+')"
    output="$(echo "$cmd" | grep -oP '(?<=-o\s)\S+')"

    if [[ -n "$source" ]] && [[ -n "$output" ]]; then
      # Resolve source path (may be relative to src_dir)
      if [[ "$source" != /* ]] && [[ ! -f "$source" ]]; then
        source="$src_dir/$source"
      fi
      echo "$PEANO_CLANG $PEANO_KERNEL_FLAGS -I$PEANO_INCLUDE -c $source -o $output"
    else
      # Can't parse -- skip this command for Peano
      echo "# SKIP (unparseable xchesscc): $cmd"
    fi
    return
  fi

  # aiecc.py commands: strip Chess flags, ensure Peano
  if [[ "$cmd" == *aiecc.py* ]]; then
    cmd="${cmd//--xchesscc/}"
    cmd="${cmd//--xbridge/}"
    # Explicitly add --no-xchesscc to be safe
    if [[ "$cmd" != *"--no-xchesscc"* ]]; then
      cmd="${cmd/aiecc.py/aiecc.py --no-xchesscc}"
    fi
    echo "$cmd"
    return
  fi

  # Other commands: pass through
  echo "$cmd"
}
```

**Step 3: Export functions**

```bash
export -f transform_for_chess transform_for_peano
```

**Step 4: Verify with dry run**

Test the transformation on a known command:
```bash
# In a shell:
source scripts/emu-bridge-test.sh  # (won't work directly, but test functions)
# Instead, test by adding --list with debug output
```

---

## Task 3: Refactor compile_one() for dual-compiler builds

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (compile_one function, ~lines 300-410)

This is the core change. The current `compile_one()` builds once. We need it
to build for each active compiler, in separate build directories.

**Step 1: Extract compile_one_compiler()**

Create a new inner function that compiles for a single compiler:
```bash
# Compile a test with a specific compiler.
# Args: $1=test_name  $2=compiler ("chess"|"peano")
compile_one_compiler() {
  local name="$1"
  local compiler="$2"
  local safe
  safe="$(sanitize_name "$name")"
  local src_dir="$TEST_SRC/$name"
  local build_dir="$BUILD_BASE/$name/$compiler"
  local log_file="$RESULTS_DIR/${safe}.${compiler}.compile.log"
  local result_file="$RESULTS_DIR/${safe}.${compiler}.compile.result"
  local lit_file="$src_dir/run.lit"

  mkdir -p "$build_dir"
  : > "$log_file"

  if [[ ! -f "$lit_file" ]]; then
    echo "FAIL" > "$result_file"
    echo "No run.lit found" >> "$log_file"
    echo "  COMPILE $name ($compiler): FAIL (no run.lit)"
    return 0
  fi

  # Check cache
  local have_xclbin=false
  if [[ -f "$build_dir/aie.xclbin" ]] || ls "$build_dir"/*.xclbin &>/dev/null; then
    have_xclbin=true
  fi

  if $have_xclbin && [[ "$FORCE_COMPILE" != "true" ]]; then
    echo "  COMPILE $name ($compiler): cached"
    echo "PASS" > "$result_file"
    return 0
  fi

  # Prepare architecture MLIR
  local npu_dev
  npu_dev="$(get_npu_device "$src_dir")"
  if [[ -f "$src_dir/aie.mlir" ]]; then
    cp "$src_dir/aie.mlir" "$build_dir/aie_arch.mlir"
    sed "s/NPUDEVICE/${npu_dev}/g" -i "$build_dir/aie_arch.mlir"
  fi

  local failed=false
  while IFS= read -r cmd; do
    [[ -z "$cmd" ]] && continue
    [[ "$cmd" == *clang*test.cpp* ]] && continue
    [[ "$cmd" == *g++*test.cpp* ]] && continue

    # Fix MLIR path references
    if [[ "$cmd" == *aiecc.py* ]]; then
      cmd="${cmd//$src_dir\/aie.mlir/./aie_arch.mlir}"
      cmd="${cmd//\.\/aie.mlir/./aie_arch.mlir}"
    fi

    # Transform command for this compiler
    if [[ "$compiler" == "chess" ]]; then
      cmd="$(transform_for_chess "$cmd")"
    else
      cmd="$(transform_for_peano "$cmd" "$src_dir")"
    fi

    # Skip commented-out commands (from transform failures)
    [[ "$cmd" == "# SKIP"* ]] && continue

    if ! ( cd "$build_dir" && nice -n 19 bash -c "$cmd" ) >> "$log_file" 2>&1; then
      failed=true
      break
    fi
  done < <(extract_build_commands "$lit_file" "$src_dir")

  if $failed; then
    echo "FAIL" > "$result_file"
    echo "  COMPILE $name ($compiler): FAIL"
    return 0
  fi

  # Verify xclbin produced
  if [[ ! -f "$build_dir/aie.xclbin" ]]; then
    local any_xclbin
    any_xclbin=$(find "$build_dir" -name "*.xclbin" -print -quit 2>/dev/null || true)
    if [[ -z "$any_xclbin" ]]; then
      echo "FAIL" > "$result_file"
      echo "  COMPILE $name ($compiler): FAIL (no xclbin produced)"
      return 0
    fi
  fi

  echo "PASS" > "$result_file"
  echo "  COMPILE $name ($compiler): PASS"
}
```

**Step 2: Rewrite compile_one() to dispatch**

```bash
compile_one() {
  local name="$1"

  # Compile for each active compiler
  if [[ "$COMPILER_MODE" != "peano" ]]; then
    compile_one_compiler "$name" "chess"
  fi
  if [[ "$COMPILER_MODE" != "chess" ]]; then
    compile_one_compiler "$name" "peano"
  fi

  # Build shared test.exe (compiler-agnostic, only needs XRT headers)
  # ... (existing test.exe build logic, unchanged, uses BUILD_BASE/$name/)
}
```

The test.exe build stays in `$BUILD_BASE/$name/` (not per-compiler).

**Step 3: Export new function**

```bash
export -f compile_one_compiler
```

**Step 4: Verify**

```bash
./scripts/emu-bridge-test.sh --compile add_one_using_dma
# Should show:
#   COMPILE add_one_using_dma (chess): PASS
#   COMPILE add_one_using_dma (peano): PASS

./scripts/emu-bridge-test.sh --compile cascade_flows
# Should show:
#   COMPILE cascade_flows (chess): PASS
#   COMPILE cascade_flows (peano): PASS or FAIL (acceptable)
```

---

## Task 4: Refactor run phases for dual-compiler execution

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (run_hw, run_emu functions)

**Step 1: Update run functions to accept compiler parameter**

The existing `run_one_hw()` and `run_one_emu()` need to:
1. Accept a compiler parameter
2. Use the correct build directory (`$BUILD_BASE/$name/$compiler/`)
3. Write results to compiler-specific files

```bash
run_one_hw() {
  local name="$1"
  local compiler="$2"
  local safe
  safe="$(sanitize_name "$name")"
  local build_dir="$BUILD_BASE/$name/$compiler"
  local result_file="$RESULTS_DIR/${safe}.${compiler}.hw.result"
  local log_file="$RESULTS_DIR/${safe}.${compiler}.hw.log"
  # ... rest uses $build_dir for xclbin, shared test.exe from $BUILD_BASE/$name/
}

run_one_emu() {
  local name="$1"
  local compiler="$2"
  # Same pattern as above
}
```

**Step 2: Update phase dispatch**

The phase 3 (HW) and phase 4 (EMU) loops iterate over test names.
Now they need to iterate over (test, compiler) pairs:

```bash
# Phase 3: Hardware runs (serial, one at a time)
for name in "${TESTS[@]}"; do
  for compiler in "${COMPILERS[@]}"; do
    # Skip if compile failed
    local safe=$(sanitize_name "$name")
    [[ "$(cat "$RESULTS_DIR/${safe}.${compiler}.compile.result" 2>/dev/null)" == "PASS" ]] || continue
    run_one_hw "$name" "$compiler"
  done
done

# Phase 4: Emulator runs (parallel)
for name in "${TESTS[@]}"; do
  for compiler in "${COMPILERS[@]}"; do
    local safe=$(sanitize_name "$name")
    [[ "$(cat "$RESULTS_DIR/${safe}.${compiler}.compile.result" 2>/dev/null)" == "PASS" ]] || continue
    echo "$name $compiler"
  done
done | xargs -P"$JOBS" -I{} bash -c 'run_one_emu ${1} ${2}' _ {}
```

Where `COMPILERS` is set based on `$COMPILER_MODE`:
```bash
case "$COMPILER_MODE" in
  chess) COMPILERS=("chess") ;;
  peano) COMPILERS=("peano") ;;
  both)  COMPILERS=("chess" "peano") ;;
esac
```

**Step 3: Verify**

```bash
./scripts/emu-bridge-test.sh --no-hw add_one_using_dma
# Should run EMU for both chess and peano builds
```

---

## Task 5: Update the report phase for dual-compiler matrix

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (report section, Phase 5)

**Step 1: New report format**

```
=== Bridge Test Results (2026-03-05) ===

Test                              Chess              Peano
                              HW     EMU         HW     EMU
add_one_using_dma            PASS   PASS        PASS   PASS
cascade_flows                PASS   PASS        FAIL*  FAIL*
ctrl_packet_reconfig         PASS   PASS        PASS   PASS
tile_mapped_read             PASS   PASS        FAIL*  FAIL*
vector_scalar_using_dma      PASS   PASS        PASS   PASS

* = compile failed (Peano incompatible)

Chess: 39/39 compiled, 35/39 HW pass, 33/39 EMU pass
Peano: 34/39 compiled, 32/34 HW pass, 30/34 EMU pass
```

**Step 2: Implement the new report generator**

Replace the existing report loop with one that reads both compiler result
files and formats the wider table. If a compiler's compile result is FAIL,
display `FAIL*` for both HW and EMU columns (didn't run).

**Step 3: Verify**

```bash
./scripts/emu-bridge-test.sh add_one_using_dma
# Shows dual-compiler report
```

---

## Task 6: Update trace sweep for dual-compiler

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (trace section)
- Modify: `tools/trace-sweep.py` (minor: accept compiler parameter for build dir)

**Step 1: Trace compilation uses aiecc.py**

The trace path (line ~619) already hardcodes `--no-xchesscc`. For
dual-compiler trace:
- Chess trace: `aiecc.py --xchesscc --xbridge ...`
- Peano trace: `aiecc.py --no-xchesscc ...` (current behavior)

**Step 2: Update trace_one() to accept compiler**

The trace function runs per (test, compiler) pair. Results go to:
```
$RESULTS_DIR/$name.trace/$compiler/
```

**Step 3: Verify**

```bash
./scripts/emu-bridge-test.sh --trace=sweep add_one_using_dma
# Runs trace sweep for both Chess and Peano builds
```

---

## Task 7: Backward compatibility and defaults

**Files:**
- Modify: `scripts/emu-bridge-test.sh`

**Step 1: Migration of existing cached builds**

Existing builds live in `$BUILD_BASE/$name/` (no compiler subdirectory).
For backward compatibility during migration:
- If `$BUILD_BASE/$name/aie.xclbin` exists but `$BUILD_BASE/$name/peano/` doesn't,
  treat the old build as a Peano build (move or symlink).
- This is a one-time migration concern.

Actually, simpler: just use `--compile` to force rebuild into the new
directory structure. Old cached builds are ignored.

**Step 2: Default behavior**

- `--compile` is recommended on first run after this change
- Without `--compile`, the cache check looks in the new per-compiler dirs
- Old flat builds won't be found, so they'll be rebuilt automatically

**Step 3: Update documentation**

Update the script header comment and `--help` text to document:
- Default is dual-compiler (both)
- `--chess-only` / `--peano-only` for single-compiler runs
- Results are per-compiler
- Chess is ground truth, Peano failures are informational

---

## Implementation Order

1. Task 1: CLI flags + env detection (foundation)
2. Task 2: Command transformation functions (core logic)
3. Task 3: Refactor compile_one() (use transformations)
4. Task 4: Refactor run phases (per-compiler execution)
5. Task 5: Report phase (dual-column output)
6. Task 6: Trace sweep dual-compiler
7. Task 7: Backward compat + docs

## Verification

After all tasks:
```bash
# Full dual-compiler run on a few tests
./scripts/emu-bridge-test.sh --compile add_one_using_dma cascade_flows

# Should produce:
# - Chess + Peano compilation for both
# - HW + EMU runs for both compilers
# - Dual-column report showing per-compiler results
# - cascade_flows: Chess PASS, Peano PASS or FAIL (informational)
```
