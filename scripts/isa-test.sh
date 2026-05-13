#!/usr/bin/env bash
# scripts/isa-test.sh -- ISA-level validation harness runner.
#
# Generates assembly test batches from aie2-isa.json, assembles with llvm-mc,
# packages with aiecc.py, runs on real NPU and emulator, diffs outputs.
#
# This tests raw ISA instruction behavior (assembly level), complementing
# scripts/instr-test.sh which tests intrinsic-level behavior (C++ level).
#
# Usage:
#   scripts/isa-test.sh [options]
#
# Options:
#   --seed N         PRNG seed (default: 42)
#   --compile        Force recompilation
#   -j N             Parallelism for compile + EMU (default: nproc)
#   --generate-only  Only run the generator, skip compile/run
#   --filter PAT     Only run batches matching PAT (grep -E on batch filename)
#   --multi-tile     Group batches into phases of 4 tiles (one per NPU column)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ISA_JSON="${PROJECT_DIR}/tools/aie2-isa.json"
LLVM_MC="${HOME}/npu-work/llvm-aie/build/bin/llvm-mc"
PEANO_LLC="${HOME}/npu-work/llvm-aie/install/bin/llc"
PEANO_INSTALL_DIR="${HOME}/npu-work/llvm-aie/install"

# mlir-aie paths for host compilation and aiecc.py
MLIR_AIE="${PROJECT_DIR}/../mlir-aie"
TEST_LIB_DIR="${MLIR_AIE}/build/runtime_lib/x86_64/test_lib"
XRT_DIR="/opt/xilinx/xrt"

# Defaults
SEED=42
FORCE_COMPILE=false
JOBS=$(nproc)
GENERATE_ONLY=false
FILTER=""
MULTI_TILE=false
AMDXDNA_TRACE=true   # opt-out via --no-amdxdna-trace; ringbuffer-captures
                     # amdxdna kernel tracepoints around the HW phase so a
                     # mid-suite wedge leaves the host->FW->IRQ->fence chain
                     # snapshotted to $RESULTS_DIR/amdxdna.{trace,dmesg}.

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)             SEED="$2"; shift 2 ;;
        --compile)          FORCE_COMPILE=true; shift ;;
        -j)                 JOBS="$2"; shift 2 ;;
        --generate-only)    GENERATE_ONLY=true; shift ;;
        --filter)           FILTER="$2"; shift 2 ;;
        --multi-tile)       MULTI_TILE=true; shift ;;
        --no-amdxdna-trace) AMDXDNA_TRACE=false; shift ;;
        *)                  echo "Unknown option: $1"; exit 1 ;;
    esac
done

OUT_DIR="${PROJECT_DIR}/build/isa-tests"
# Results under build/ so they survive reboots (unlike /tmp).
# 'latest' is always the working directory. On each new run, any
# existing 'latest' is archived to a timestamped folder first.
RESULTS_BASE="${PROJECT_DIR}/build/isa-test-results"
RESULTS_DIR="${ISA_TEST_RESULTS:-${RESULTS_BASE}/latest}"

# Determine which EMU profile to use and check for stale builds.
EMU_PROFILE="${XDNA_EMU_RUNTIME:-debug}"
EMU_LIB="${PROJECT_DIR}/target/${EMU_PROFILE}/libxdna_emu.so"
if [[ ! -f "$EMU_LIB" ]]; then
    echo "WARNING: EMU lib not found at $EMU_LIB"
    echo "  Run: cargo build $([ "$EMU_PROFILE" = "release" ] && echo "--release")"
    echo "  Or set XDNA_EMU_RUNTIME=release to use the release profile."
fi
# Auto-rebuild if the EMU lib is older than any Rust source file.
# This ensures the plugin always matches the current code -- manual rebuilds
# are the #1 source of "why didn't my change take effect?" confusion.
if [[ -f "$EMU_LIB" ]]; then
    newest_src=$(find "${PROJECT_DIR}/src" -name '*.rs' -newer "$EMU_LIB" 2>/dev/null | head -1)
    if [[ -n "$newest_src" ]]; then
        echo "EMU lib ($EMU_PROFILE) is stale -- rebuilding..."
        CARGO_FLAGS=""
        [[ "$EMU_PROFILE" = "release" ]] && CARGO_FLAGS="--release"
        TMPDIR="${TMPDIR:-/tmp}" nice -n 19 cargo build $CARGO_FLAGS 2>&1
        echo "Rebuild complete."
        echo ""
    fi
elif [[ ! -f "$EMU_LIB" ]]; then
    echo "EMU lib not found -- building..."
    CARGO_FLAGS=""
    [[ "$EMU_PROFILE" = "release" ]] && CARGO_FLAGS="--release"
    TMPDIR="${TMPDIR:-/tmp}" nice -n 19 cargo build $CARGO_FLAGS 2>&1
    echo "Build complete."
    echo ""
fi

echo "=== ISA-Level Validation Harness ==="
echo "ISA JSON: $ISA_JSON"
echo "Out dir:  $OUT_DIR"
echo "Results:  $RESULTS_DIR"
echo "EMU:      $EMU_PROFILE ($EMU_LIB)"
if $MULTI_TILE; then
    echo "Mode:     multi-tile (4 tiles per phase)"
fi
echo ""

# ---- Phase 1: Generate ----
echo "--- Phase 1: Generate ---"

# Skip generation if ALL inputs (ISA JSON + all Python tools) are older
# than the manifest.  Any tool change triggers regeneration -- stale
# builds from missed tool changes have been a recurring pain point.
GENERATOR="${PROJECT_DIR}/tools/isa-test-gen.py"
HOST_GENERATOR="${PROJECT_DIR}/tools/instr-test-gen.py"
MANIFEST_OUT="${OUT_DIR}/manifest.json"
TOOLS_CHANGED=false
if [[ -f "$MANIFEST_OUT" ]]; then
    for tool in "$ISA_JSON" "$GENERATOR" "$HOST_GENERATOR" \
                "${PROJECT_DIR}/tools/"*.py; do
        if [[ "$tool" -nt "$MANIFEST_OUT" ]]; then
            echo "  Input changed: $(basename "$tool")"
            TOOLS_CHANGED=true
            break
        fi
    done
fi
if ! $FORCE_COMPILE && [[ -f "$MANIFEST_OUT" ]] && ! $TOOLS_CHANGED; then
    echo "  Up to date (inputs unchanged). Skipping generation."
else
    python3 "$GENERATOR" \
        --isa-json "$ISA_JSON" \
        --out-dir "$OUT_DIR"
fi
echo ""

if $GENERATE_ONLY; then
    echo "Generate-only mode. Done."
    exit 0
fi

# Read manifest to get batch list.
MANIFEST="${OUT_DIR}/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest.json not found at $MANIFEST"
    exit 1
fi

BATCH_INFO=$(python3 -c "
import json, sys
m = json.load(open('${MANIFEST}'))
for b in m['batches']:
    source_type = b.get('source_type', 'assembly')
    print(b['batch_index'], b['filename'], b['in_size'], b['out_size'], source_type)
")

# Apply filter if specified.
if [[ -n "$FILTER" ]]; then
    BATCH_INFO=$(echo "$BATCH_INFO" | grep -E "$FILTER" || true)
fi

TOTAL=$(echo "$BATCH_INFO" | grep -c . || echo 0)
if [[ "$TOTAL" -eq 0 ]]; then
    echo "No batches match filter. Done."
    exit 0
fi
echo "Batches to process: $TOTAL"
echo ""

# ---- Phase 2: Assemble / Compile ----
echo "--- Phase 2: Assemble / Compile ---"

if [[ ! -x "$LLVM_MC" ]]; then
    echo "ERROR: llvm-mc not found at $LLVM_MC"
    exit 1
fi
if [[ ! -x "$PEANO_LLC" ]]; then
    echo "ERROR: llc not found at $PEANO_LLC"
    exit 1
fi

assemble_file() {
    # Assemble a single .s or .ll file into a .o file (name derived from input).
    local filename="$1"
    local in_path="${OUT_DIR}/${filename}"

    # Derive .o name from the source filename to match manifest convention.
    local o_name
    if [[ "$filename" == *.ll ]]; then
        o_name="${filename%.ll}.o"
    else
        o_name="${filename%.s}.o"
    fi
    local o_path="${OUT_DIR}/${o_name}"

    # Skip if already assembled/compiled (unless --compile).
    if ! $FORCE_COMPILE && [[ -f "$o_path" ]] && [[ "$o_path" -nt "$in_path" ]]; then
        return 0
    fi

    if [[ "$filename" == *.ll ]]; then
        # LLVM IR: compile with llc.
        # --issue-limit=1 prevents VLIW packing so each instruction occupies
        # its own cycle slot, which is required for single-instruction validation.
        nice -n 19 "$PEANO_LLC" -mtriple=aie2 --issue-limit=1 -filetype=obj \
            -o "$o_path" "$in_path" 2>"${in_path%.ll}.llc.log" && \
            echo "  LLC OK: ${filename}" || \
            echo "  LLC FAIL: ${filename} (see ${in_path%.ll}.llc.log)"
    else
        # Assembly (.s): assemble with llvm-mc.
        nice -n 19 "$LLVM_MC" --triple=aie2 --filetype=obj \
            -o "$o_path" "$in_path" 2>"${in_path%.s}.mc.log" && \
            echo "  ASM OK: ${filename}" || \
            echo "  ASM FAIL: ${filename} (see ${in_path%.s}.mc.log)"
    fi
}
export -f assemble_file
export OUT_DIR FORCE_COMPILE LLVM_MC PEANO_LLC

# Build list of all files to assemble (includes producer/consumer for pairs).
ASM_FILES=$(python3 -c "
import json
m = json.load(open('${MANIFEST}'))
for b in m['batches']:
    st = b.get('source_type', 'assembly')
    if st in ('cascade_pair', 'stream_pair'):
        print(b['producer_filename'])
        print(b['consumer_filename'])
    else:
        print(b['filename'])
")

echo "$ASM_FILES" | xargs -P "$JOBS" -n1 bash -c 'assemble_file "$1"' _
echo ""

# ---- Phase 3: Link + Package ----

# Generate shared test_host.cpp from instr-test-gen.py.
# Regenerate if the generator script is newer than the output, or if missing.
HOST_CPP="${OUT_DIR}/test_host.cpp"
if [[ ! -f "$HOST_CPP" ]] || [[ "$HOST_GENERATOR" -nt "$HOST_CPP" ]]; then
    echo "Generating test_host.cpp..."
    python3 -c "
import sys; sys.path.insert(0, '${PROJECT_DIR}/tools')
import importlib
gen = importlib.import_module('instr-test-gen')
print(gen.generate_test_host_cpp())
" > "$HOST_CPP"
fi

# Compile shared host binary (once).
# Rebuild if source changed, or if the generator is newer (implies source changed).
HOST_BIN="${OUT_DIR}/test_host"
if $FORCE_COMPILE || [[ ! -f "$HOST_BIN" ]] || [[ "$HOST_CPP" -nt "$HOST_BIN" ]] \
    || [[ "$HOST_GENERATOR" -nt "$HOST_BIN" ]]; then
    echo "Compiling test_host..."
    clang++ "$HOST_CPP" -o "$HOST_BIN" \
        -std=c++17 -Wall \
        -I "${TEST_LIB_DIR}/include" \
        -I "${XRT_DIR}/include" \
        -L "${TEST_LIB_DIR}/lib" \
        -L "${XRT_DIR}/lib" \
        -ltest_utils -lxrt_coreutil \
        -lrt -lstdc++
fi

if $MULTI_TILE; then
    # ---- Multi-tile mode: group batches into phases of 4 ----

    # Compute phase assignments from BATCH_INFO.
    # For pair batches (cascade_pair, stream_pair), verify both .o files exist.
    # For normal batches, verify the single .o exists.
    PHASE_INFO=""
    phase_idx=0
    batch_indices=""
    count=0
    while IFS=' ' read -r idx filename in_size out_size source_type; do
        # Check that the required .o file(s) exist.
        if [[ "$source_type" == "cascade_pair" ]] || [[ "$source_type" == "stream_pair" ]]; then
            # Pair batches need both producer and consumer .o files.
            prod_o="${filename/_consumer.s/_producer.o}"
            cons_o="${filename%.s}.o"
            if [[ ! -f "${OUT_DIR}/${prod_o}" ]] || [[ ! -f "${OUT_DIR}/${cons_o}" ]]; then
                echo "  SKIP batch_${idx}: pair assembly incomplete (not included in phase)"
                continue
            fi
        else
            # Normal batch: derive .o from filename.
            if [[ "$filename" == *.ll ]]; then
                o_name="${filename%.ll}.o"
            else
                o_name="${filename%.s}.o"
            fi
            if [[ ! -f "${OUT_DIR}/${o_name}" ]]; then
                echo "  SKIP batch_${idx}: assembly failed (not included in phase)"
                continue
            fi
        fi
        if [[ $count -ge 4 ]]; then
            PHASE_INFO="${PHASE_INFO}${phase_idx} ${batch_indices%,}\n"
            phase_idx=$((phase_idx + 1))
            batch_indices=""
            count=0
        fi
        batch_indices="${batch_indices}${idx},"
        count=$((count + 1))
    done <<< "$BATCH_INFO"
    # Flush remaining batches.
    if [[ $count -gt 0 ]]; then
        PHASE_INFO="${PHASE_INFO}${phase_idx} ${batch_indices%,}\n"
    fi

    TOTAL_PHASES=$(printf '%b' "$PHASE_INFO" | grep -c . || echo 0)
    echo "--- Phase 3: Package (multi-tile, $TOTAL_PHASES phases) ---"

    package_phase() {
        local pidx="$1"
        local batch_list="$2"
        local phase_dir="${OUT_DIR}/phase_${pidx}"

        # Skip if already packaged (unless --compile).
        if ! $FORCE_COMPILE && [[ -f "${phase_dir}/aie.xclbin" ]] && [[ -f "${phase_dir}/insts.bin" ]]; then
            # Check if any SOURCE .o (in OUT_DIR) is newer than the phase copy.
            # This catches cases where the test generator changes batch composition.
            local stale=false
            for ofile in "${phase_dir}"/*.o; do
                [[ -f "$ofile" ]] || continue
                local basename=$(basename "$ofile")
                local src_o="${OUT_DIR}/${basename}"
                if [[ -f "$src_o" ]] && [[ "$src_o" -nt "$ofile" ]]; then
                    stale=true
                    break
                fi
                if [[ "$ofile" -nt "${phase_dir}/aie.xclbin" ]]; then
                    stale=true
                    break
                fi
            done
            if ! $stale; then
                return 0
            fi
            echo "  STALE phase_${pidx}: source .o newer than phase copy, repackaging"
        fi

        mkdir -p "$phase_dir"

        # Use isa-multi-tile-gen.py to copy/rename .o files and generate MLIR.
        python3 "${PROJECT_DIR}/tools/isa-multi-tile-gen.py" \
            --manifest "$MANIFEST" \
            --batches "$batch_list" \
            --phase-idx "$pidx" \
            --out-dir "$phase_dir" \
            --obj-dir "$OUT_DIR" 2>"${phase_dir}/gen.log" || {
                echo "  GEN FAIL: phase_${pidx} (see ${phase_dir}/gen.log)"
                return 0
            }

        # Run aiecc.py (Peano mode, no Chess).
        (cd "$phase_dir" && \
            nice -n 19 aiecc.py --no-aiesim --no-xchesscc --no-xbridge \
                --aie-generate-xclbin --xclbin-name=aie.xclbin \
                --aie-generate-npu-insts --npu-insts-name=insts.bin \
                aie.mlir 2>"${phase_dir}/aiecc.log") && \
            echo "  PKG OK: phase_${pidx}" || \
            echo "  PKG FAIL: phase_${pidx} (see ${phase_dir}/aiecc.log)"
    }
    export -f package_phase
    export OUT_DIR FORCE_COMPILE MANIFEST PROJECT_DIR

    printf '%b' "$PHASE_INFO" | while IFS=' ' read -r pidx batch_list; do
        [[ -z "$pidx" ]] && continue
        printf '%s\0%s\0' "$pidx" "$batch_list"
    done | xargs -0 -n2 -P "$JOBS" bash -c 'package_phase "$1" "$2"' _
    echo ""

else
    # ---- Single-tile mode (original behavior) ----
    echo "--- Phase 3: Link + Package (aiecc.py) ---"

    package_one() {
        local batch_idx="$1"
        local filename="$2"
        local in_size="$3"
        local out_size="$4"
        local source_type="$5"
        local batch_dir="${OUT_DIR}/batch_${batch_idx}"

        # Pair batches need two tiles -- skip in single-tile mode.
        if [[ "$source_type" == "cascade_pair" ]] || [[ "$source_type" == "stream_pair" ]]; then
            return 0
        fi

        # Derive .o name from filename (matches manifest naming convention).
        local o_name
        if [[ "$filename" == *.ll ]]; then
            o_name="${filename%.ll}.o"
        else
            o_name="${filename%.s}.o"
        fi
        local o_path="${OUT_DIR}/${o_name}"

        if [[ ! -f "$o_path" ]]; then
            echo "  SKIP batch_${batch_idx}: assembly failed"
            return 0
        fi

        # Skip if already packaged (unless --compile or .o is newer than xclbin).
        if ! $FORCE_COMPILE && [[ -f "${batch_dir}/aie.xclbin" ]] && [[ -f "${batch_dir}/insts.bin" ]]; then
            if [[ ! "$o_path" -nt "${batch_dir}/aie.xclbin" ]]; then
                return 0
            fi
            echo "  STALE batch_${batch_idx}: .o newer than xclbin, repackaging"
        fi

        mkdir -p "$batch_dir"
        cp "$o_path" "${batch_dir}/kernel.o"

        # Generate aie.mlir for this batch's buffer sizes.
        python3 -c "
import sys; sys.path.insert(0, '${PROJECT_DIR}/tools')
import importlib
gen = importlib.import_module('instr-test-gen')
print(gen.generate_aie_mlir(${in_size}, ${out_size}))
" > "${batch_dir}/aie.mlir"

        # Run aiecc.py (Peano mode, no Chess).
        if (cd "$batch_dir" && \
            nice -n 19 aiecc.py --no-aiesim --no-xchesscc --no-xbridge \
                --aie-generate-xclbin --xclbin-name=aie.xclbin \
                --aie-generate-npu-insts --npu-insts-name=insts.bin \
                aie.mlir 2>"${batch_dir}/aiecc.log"); then
            echo "  PKG OK: batch_${batch_idx}"
        else
            echo "  PKG FAIL: batch_${batch_idx} (see ${batch_dir}/aiecc.log)"
            return 1
        fi
    }
    export -f package_one
    export HOST_BIN PROJECT_DIR OUT_DIR FORCE_COMPILE

    echo "$BATCH_INFO" | while IFS=' ' read -r idx filename in_size out_size source_type; do
        printf '%s\0%s\0%s\0%s\0%s\0' "$idx" "$filename" "$in_size" "$out_size" "$source_type"
    done | xargs -0 -n5 -P "$JOBS" bash -c 'package_one "$1" "$2" "$3" "$4" "$5"' _
    echo ""
fi

# ---- Phase 4: Run HW ----
# Archive any existing 'latest' results before starting a new run.
# Uses the modification time of the directory to name the archive,
# with a suffix for same-day disambiguation.
mkdir -p "$RESULTS_BASE"
# Migrate from old symlink-based layout: if 'latest' is a symlink,
# remove it so we can create a real directory.
if [[ -L "$RESULTS_DIR" ]]; then
    rm "$RESULTS_DIR"
fi
if [[ -d "$RESULTS_DIR" ]] && [[ -z "${ISA_TEST_RESULTS:-}" ]]; then
    # Only archive if there are actual result files in latest.
    if ls "${RESULTS_DIR}"/batch_*_hw.bin &>/dev/null || \
       ls "${RESULTS_DIR}"/phase_*_hw.bin &>/dev/null; then
        archive_date=$(date -r "$RESULTS_DIR" +%Y%m%d)
        archive_path="${RESULTS_BASE}/${archive_date}"
        # Add numeric suffix if the date already exists.
        if [[ -d "$archive_path" ]]; then
            suffix=2
            while [[ -d "${archive_path}-${suffix}" ]]; do
                ((suffix++))
            done
            archive_path="${archive_path}-${suffix}"
        fi
        echo "Archiving previous results -> $(basename "$archive_path")"
        mv "$RESULTS_DIR" "$archive_path"
    else
        # Empty or no results -- just clear it.
        rm -rf "$RESULTS_DIR"
    fi
fi
mkdir -p "$RESULTS_DIR"

# Capture amdxdna kernel tracepoints (xdna_job, mailbox, IRQ, fence) for
# the duration of the HW phase.  Single-pkexec daemon design: the daemon
# spawned here holds tracing on for the whole HW phase under one auth,
# and snapshots+disables to $RESULTS_DIR/amdxdna.* when we remove the
# sentinel at Phase 5 boundary.  No second pkexec, no auth-gap delay
# mid-suite.  Daemon's EXIT trap also handles parent-died and signal-
# killed cases.
AMDXDNA_SENTINEL=""
AMDXDNA_DAEMON_PID=""
if $AMDXDNA_TRACE; then
    AMDXDNA_SENTINEL=$(mktemp /tmp/claude-1000/amdxdna-trace.isa.XXXXXX 2>/dev/null \
                       || mktemp /tmp/amdxdna-trace.isa.XXXXXX)
    pkexec "$PROJECT_DIR/tools/amdxdna-trace.sh" daemon \
        "$AMDXDNA_SENTINEL" $$ "$RESULTS_DIR" amdxdna &
    AMDXDNA_DAEMON_PID=$!
    sleep 1   # let daemon set up tracing before HW jobs start
    if ! kill -0 $AMDXDNA_DAEMON_PID 2>/dev/null; then
        echo "WARN: amdxdna-trace daemon failed to start; continuing without it"
        AMDXDNA_TRACE=false
        rm -f "$AMDXDNA_SENTINEL"
    fi
fi

if $MULTI_TILE; then
    # Multi-tile HW execution: one xclbin per phase, serial.
    echo "--- Phase 4: Run HW (multi-tile, serial) ---"
        printf '%b' "$PHASE_INFO" | while IFS=' ' read -r pidx batch_list; do
            [[ -z "$pidx" ]] && continue
            phase_dir="${OUT_DIR}/phase_${pidx}"
            [[ ! -f "${phase_dir}/aie.xclbin" ]] && {
                echo "  SKIP phase_${pidx}: not packaged"
                continue
            }

            # Compute combined buffer sizes from manifest.
            sizes=$(python3 -c "
import json
m = json.load(open('${MANIFEST}'))
indices = [int(x) for x in '${batch_list}'.split(',')]
by_idx = {b['batch_index']: b for b in m['batches']}
total_in = sum(by_idx[i]['in_size'] for i in indices)
total_out = sum(by_idx[i]['out_size'] for i in indices)
# Stream pairs have in_size=0; MLIR uses max(1 elem, total) for memref.
# Match that here so test_host doesn't try to allocate 0 bytes.
total_in = max(4, total_in)
total_out = max(4, total_out)
print(total_in, total_out)
")
            in_size=$(echo "$sizes" | awk '{print $1}')
            out_size=$(echo "$sizes" | awk '{print $2}')

            hw_out="${RESULTS_DIR}/phase_${pidx}_hw.bin"
            hw_log="${RESULTS_DIR}/phase_${pidx}_hw.log"
            rc=0
            # Unset XDNA_EMU* for HW runs so they go through the real NPU.
            env -u XDNA_EMU -u XDNA_EMU_RUNTIME timeout 30 "$HOST_BIN" \
                -x "${phase_dir}/aie.xclbin" \
                -k MLIR_AIE \
                -i "${phase_dir}/insts.bin" \
                --in-size "$in_size" --out-size "$out_size" \
                --seed "$SEED" --out-file "$hw_out" 2>"$hw_log" \
                || rc=$?
            if [[ $rc -eq 0 ]]; then
                echo "  HW OK: phase_${pidx}"
            elif [[ $rc -eq 124 ]]; then
                echo "  HW TIMEOUT: phase_${pidx} (killed after 30s -- NPU may be wedged)"
                echo "FATAL: Hardware timeout. Aborting."
                exit 1
            else
                echo "  HW FAIL: phase_${pidx} (rc=$rc, see $hw_log)"
                echo "FATAL: Hardware run failed. Aborting."
                exit 1
            fi
        done
        echo ""
else
    # Single-tile HW execution (original behavior).
    echo "--- Phase 4: Run HW (serial) ---"
        while IFS=' ' read -r idx filename in_size out_size source_type; do
            # Skip pair batches in single-tile mode.
            [[ "$source_type" == "cascade_pair" ]] && continue
            [[ "$source_type" == "stream_pair" ]] && continue

            batch_dir="${OUT_DIR}/batch_${idx}"
            hw_out="${RESULTS_DIR}/batch_${idx}_hw.bin"

            if [[ ! -f "${batch_dir}/aie.xclbin" ]]; then
                echo "  SKIP batch_${idx}: not packaged"
                continue
            fi

            hw_log="${RESULTS_DIR}/batch_${idx}_hw.log"
            rc=0
            # Unset XDNA_EMU* for HW runs so they go through the real NPU,
            # not the emulator plugin.  Use env -u to fully remove them.
            env -u XDNA_EMU -u XDNA_EMU_RUNTIME timeout 30 "$HOST_BIN" \
                -x "${batch_dir}/aie.xclbin" \
                -k MLIR_AIE \
                -i "${batch_dir}/insts.bin" \
                --in-size "$in_size" --out-size "$out_size" \
                --seed "$SEED" --out-file "$hw_out" 2>"$hw_log" \
                || rc=$?
            if [[ $rc -eq 0 ]]; then
                echo "  HW OK: batch_${idx}"
            elif [[ $rc -eq 124 ]]; then
                echo "  HW TIMEOUT: batch_${idx} (killed after 30s -- NPU may be wedged)"
                echo "FATAL: Hardware timeout. Aborting."
                exit 1
            else
                echo "  HW FAIL: batch_${idx} (rc=$rc, see $hw_log)"
                echo "FATAL: Hardware run failed. Aborting."
                exit 1
            fi
        done <<< "$BATCH_INFO"
    echo ""
fi

if $AMDXDNA_TRACE; then
    # Signal the daemon to wrap up: remove sentinel, wait for snapshot+disable.
    rm -f "$AMDXDNA_SENTINEL"
    wait "$AMDXDNA_DAEMON_PID" 2>/dev/null || true
fi

# ---- Phase 5: Run EMU ----
if $MULTI_TILE; then
    echo "--- Phase 5: Run EMU (multi-tile, j=$JOBS) ---"

        run_emu_phase() {
            local pidx="$1"
            local batch_list="$2"
            local in_size="$3"
            local out_size="$4"
            local phase_dir="${OUT_DIR}/phase_${pidx}"
            local emu_out="${RESULTS_DIR}/phase_${pidx}_emu.bin"

            if [[ ! -f "${phase_dir}/aie.xclbin" ]]; then
                return 0
            fi

            if XDNA_EMU=1 XDNA_EMU_RUNTIME="$EMU_PROFILE" "$HOST_BIN" \
                -x "${phase_dir}/aie.xclbin" \
                -k MLIR_AIE \
                -i "${phase_dir}/insts.bin" \
                --in-size "$in_size" --out-size "$out_size" \
                --seed "$SEED" --out-file "$emu_out" 2>"${RESULTS_DIR}/phase_${pidx}_emu.log"; then
                echo "  EMU OK: phase_${pidx}"
            else
                echo "  EMU FAIL: phase_${pidx} (see ${RESULTS_DIR}/phase_${pidx}_emu.log)"
                return 1
            fi
        }
        export -f run_emu_phase
        export HOST_BIN OUT_DIR RESULTS_DIR SEED EMU_PROFILE MANIFEST
        export XDNA_EMU=1
        export XDNA_EMU_RUNTIME="$EMU_PROFILE"

        # Build argument list: pidx batch_list in_size out_size (null-separated).
        printf '%b' "$PHASE_INFO" | while IFS=' ' read -r pidx batch_list; do
            [[ -z "$pidx" ]] && continue
            sizes=$(python3 -c "
import json
m = json.load(open('${MANIFEST}'))
indices = [int(x) for x in '${batch_list}'.split(',')]
by_idx = {b['batch_index']: b for b in m['batches']}
total_in = sum(by_idx[i]['in_size'] for i in indices)
total_out = sum(by_idx[i]['out_size'] for i in indices)
# Stream pairs have in_size=0; MLIR uses max(1 elem, total) for memref.
# Match that here so test_host doesn't try to allocate 0 bytes.
total_in = max(4, total_in)
total_out = max(4, total_out)
print(total_in, total_out)
")
            in_size=$(echo "$sizes" | awk '{print $1}')
            out_size=$(echo "$sizes" | awk '{print $2}')
            printf '%s\0%s\0%s\0%s\0' "$pidx" "$batch_list" "$in_size" "$out_size"
        done | xargs -0 -n4 -P "$JOBS" bash -c 'run_emu_phase "$1" "$2" "$3" "$4"' _
        echo ""
else
    echo "--- Phase 5: Run EMU (j=$JOBS) ---"

        run_emu_one() {
            local idx="$1"
            local in_size="$2"
            local out_size="$3"
            local batch_dir="${OUT_DIR}/batch_${idx}"
            local emu_out="${RESULTS_DIR}/batch_${idx}_emu.bin"

            if [[ ! -f "${batch_dir}/aie.xclbin" ]]; then
                return 0
            fi

            if XDNA_EMU=1 XDNA_EMU_RUNTIME="$EMU_PROFILE" "$HOST_BIN" \
                -x "${batch_dir}/aie.xclbin" \
                -k MLIR_AIE \
                -i "${batch_dir}/insts.bin" \
                --in-size "$in_size" --out-size "$out_size" \
                --seed "$SEED" --out-file "$emu_out" 2>"${RESULTS_DIR}/batch_${idx}_emu.log"; then
                echo "  EMU OK: batch_${idx}"
            else
                echo "  EMU FAIL: batch_${idx} (see ${RESULTS_DIR}/batch_${idx}_emu.log)"
                return 1
            fi
        }
        export -f run_emu_one
        export HOST_BIN OUT_DIR RESULTS_DIR SEED EMU_PROFILE
        # Export XDNA_EMU* so xargs subprocesses inherit them reliably.
        # The inline XDNA_EMU=1 XDNA_EMU_RUNTIME=... in run_emu_one is
        # belt-and-suspenders, but xargs spawns via bash -c which can miss
        # inline env vars if the shell fast-paths the exec.
        export XDNA_EMU=1
        export XDNA_EMU_RUNTIME="$EMU_PROFILE"

        echo "$BATCH_INFO" | while IFS=' ' read -r idx filename in_size out_size source_type; do
            # Skip pair batches in single-tile mode.
            [[ "$source_type" == "cascade_pair" ]] && continue
            [[ "$source_type" == "stream_pair" ]] && continue
            printf '%s\0%s\0%s\0' "$idx" "$in_size" "$out_size"
        done | xargs -0 -n3 -P "$JOBS" bash -c 'run_emu_one "$1" "$2" "$3"' _
        if [[ ${PIPESTATUS[1]} -ne 0 ]]; then
            echo "FATAL: Emulator run failed. Aborting."
            exit 1
        fi
        echo ""
fi

# ---- Phase 6: Compare ----
if $MULTI_TILE; then
    # Multi-tile: split combined phase outputs into per-batch files, then compare.
    echo "--- Phase 6: Split + Compare ---"

        # Split all phase outputs into per-batch files.
        printf '%b' "$PHASE_INFO" | while IFS=' ' read -r pidx batch_list; do
            [[ -z "$pidx" ]] && continue
            python3 -c "
import json, sys
m = json.load(open('${MANIFEST}'))
indices = [int(x) for x in '${batch_list}'.split(',')]
by_idx = {b['batch_index']: b for b in m['batches']}

for mode in ['hw', 'emu']:
    combined = '${RESULTS_DIR}/phase_${pidx}_' + mode + '.bin'
    try:
        data = open(combined, 'rb').read()
    except FileNotFoundError:
        continue
    offset = 0
    for idx in indices:
        size = by_idx[idx]['out_size']
        chunk = data[offset:offset + size]
        out_path = f'${RESULTS_DIR}/batch_{idx}_' + mode + '.bin'
        open(out_path, 'wb').write(chunk)
        offset += size
"
        done

        # Compare per-batch (same logic as single-tile).
        PASS=0
        FAIL=0
        SKIP=0
        FAIL_LIST=""

        while IFS=' ' read -r idx filename in_size out_size source_type; do
            hw_out="${RESULTS_DIR}/batch_${idx}_hw.bin"
            emu_out="${RESULTS_DIR}/batch_${idx}_emu.bin"

            if [[ ! -f "$hw_out" ]] || [[ ! -f "$emu_out" ]]; then
                SKIP=$((SKIP + 1))
                continue
            fi

            if cmp -s "$hw_out" "$emu_out"; then
                PASS=$((PASS + 1))
            else
                FAIL=$((FAIL + 1))
                FAIL_LIST="${FAIL_LIST}  DIVERGE: batch_${idx}\n"
                echo "  DIVERGE: batch_${idx}"
            fi
        done <<< "$BATCH_INFO"

        echo ""
        echo "=== Results ==="
        echo "PASS: $PASS"
        echo "FAIL: $FAIL"
        echo "SKIP: $SKIP"
        echo "Phases: $TOTAL_PHASES"
        if [[ $FAIL -gt 0 ]]; then
            echo ""
            echo "Divergences:"
            printf '%b' "$FAIL_LIST"
        fi
else
    # Single-tile comparison (original behavior).
    echo "--- Phase 6: Compare ---"
        PASS=0
        FAIL=0
        SKIP=0
        FAIL_LIST=""

        while IFS=' ' read -r idx filename in_size out_size source_type; do
            # Skip pair batches in single-tile mode.
            [[ "$source_type" == "cascade_pair" ]] && continue
            [[ "$source_type" == "stream_pair" ]] && continue

            hw_out="${RESULTS_DIR}/batch_${idx}_hw.bin"
            emu_out="${RESULTS_DIR}/batch_${idx}_emu.bin"

            if [[ ! -f "$hw_out" ]] || [[ ! -f "$emu_out" ]]; then
                SKIP=$((SKIP + 1))
                continue
            fi

            if cmp -s "$hw_out" "$emu_out"; then
                PASS=$((PASS + 1))
            else
                FAIL=$((FAIL + 1))
                FAIL_LIST="${FAIL_LIST}  DIVERGE: batch_${idx}\n"
                echo "  DIVERGE: batch_${idx}"
            fi
        done <<< "$BATCH_INFO"

        echo ""
        echo "=== Results ==="
        echo "PASS: $PASS"
        echo "FAIL: $FAIL"
        echo "SKIP: $SKIP"
        if [[ $FAIL -gt 0 ]]; then
            echo ""
            echo "Divergences:"
            printf '%b' "$FAIL_LIST"
        fi
fi

# ---- Phase 7: Test-Point Analysis ----
# Runs the per-instruction analyzer for fine-grained accuracy metrics.
ANALYZER="${PROJECT_DIR}/tools/isa-test-analyze.py"
ANALYSIS_LOG="${RESULTS_DIR}/analysis.log"
if [[ -f "$ANALYZER" ]] && [[ -f "$MANIFEST" ]]; then
    echo ""
    echo "--- Phase 7: Test-Point Analysis ---"
    python3 "$ANALYZER" \
        --manifest "$MANIFEST" \
        --results-dir "$RESULTS_DIR" \
        --summary 2>&1 | tee "$ANALYSIS_LOG"

    # Also write a failing-only detailed log.
    DETAIL_LOG="${RESULTS_DIR}/analysis-failing.log"
    python3 "$ANALYZER" \
        --manifest "$MANIFEST" \
        --results-dir "$RESULTS_DIR" \
        --failing > "$DETAIL_LOG" 2>&1

    echo ""
    echo "Analysis logs written to:"
    echo "  Summary:  $ANALYSIS_LOG"
    echo "  Detailed: $DETAIL_LOG"
fi
