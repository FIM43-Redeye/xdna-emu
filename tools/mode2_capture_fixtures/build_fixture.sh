#!/usr/bin/env bash
# Build one capture fixture end to end.
#
# For a given fixture directory (containing kernel.cc + aie.mlir):
#   1. Compile kernel.cc with peano clang to kernel.o (target aie2)
#   2. Materialize aie_arch.mlir from aie.mlir with NPUDEVICE -> npu1_1col
#   3. Run aiecc to produce baseline build/baseline/{aie.xclbin, insts.bin}
#   4. Inject trace ops (default mode: event_pc) -> build/aie_traced.mlir
#   5. Run aiecc on traced MLIR -> build/traced/{aie.xclbin, insts.bin}
#
# The two xclbins are independent artifacts. The baseline pair is useful
# for sanity-running the kernel without trace overhead; the traced pair
# is what gets handed to bridge-trace-runner for HW capture.
#
# Usage:
#   build_fixture.sh [--chess] [--mode2] <fixture_dir>
#
# --chess drops --no-xchesscc / --no-xbridge from the aiecc invocations
# AND switches the kernel.cc compile to xchesscc as well. Cross-toolchain
# mixing (Peano kernel.o + Chess linker, or vice versa) crashes the chess
# bridge linker with a NULL-deref; the two toolchains must be consistent
# end to end.
#
# --mode2 switches the trace injector to inst_exec mode (records the
# compressed LC/PC/atom/RLE frame tree). Default is event_pc, which traces
# events keyed by PC -- easiest to ground in the disassembly. Use --mode2
# only for the LC-overflow / ZOL-shape probes that need raw inst_exec
# frames; event_pc is the right default for general-purpose trace work.
#
# IMPORTANT: --mode2 captures are EMPTY for tiny single-pass kernels
# regardless of compiler. Mode-2 records core execution activity; tiny
# kernels generate < 28 bytes -- below the trace controller's packet
# threshold -- so the partial buffer drops on stop. Use a multi-pass
# wrapper (heavy_zol = 64 passes, lc_overflow_probe = 4-pass) for any
# fixture that opts into --mode2. See
# docs/superpowers/findings/2026-05-08-mode2-flush-and-peano-non-bug.md
# for the empirical breakdown.
#
# Environment overrides (rare):
#   PEANO_INSTALL_DIR (default: /home/triple/npu-work/llvm-aie/install)
#   AIECC             (default: prefer the C++ `aiecc` binary if on PATH,
#                      fall back to `aiecc.py` driven by $AIE_PYTHON below)
#   AIE_PYTHON        (default: autodetected Python 3.13 with the aie
#                      bindings + jsonschema available; needed for the
#                      mlir-trace-inject step regardless of which aiecc
#                      we run)
#
# Python autodetect order (handles the system upgrade away from
# /usr/bin/python3.13 -- which dangles ironenv's own python symlink --
# without forcing every dev to repair their venv before they can build):
#   1. $AIE_PYTHON if set
#   2. ironenv's bundled-3.13 side symlink ($MLIR_AIE_DIR/ironenv/bin/python3.13.amd)
#      if a previous setup has placed it
#   3. the AMD aietools bundled python (tps/lnx64/python-3.13.0/bin/python3.13)
#      -- works because mlir-aie's bindings link the cpython-3.13 ABI
#   4. plain `python3.13` from PATH if available
#
# We accept the first candidate that can `import aie.dialects.aie` AND
# `import jsonschema` -- both are required by mlir-trace-inject.py.

set -euo pipefail

PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR:-/home/triple/npu-work/llvm-aie/install}"
INJECT="$(dirname "$(realpath "$0")")/../mlir-trace-inject.py"

# Resolve the working python used to drive the injector.
_aie_python_works() {
    local candidate="$1"
    [[ -n "$candidate" && -x "$candidate" ]] || return 1
    "$candidate" -c 'import aie.dialects.aie, jsonschema' >/dev/null 2>&1
}

if [[ -z "${AIE_PYTHON:-}" ]]; then
    _mlir_aie_dir="${MLIR_AIE_DIR:-/home/triple/npu-work/mlir-aie}"
    _amd_dir="${AMD_UNIFIED_DIR:-/home/triple/npu-work/amd-unified-software}"
    for _candidate in \
        "$_mlir_aie_dir/ironenv/bin/python3.13.amd" \
        "$_amd_dir/tps/lnx64/python-3.13.0/bin/python3.13" \
        "$(command -v python3.13 2>/dev/null || true)"; do
        if _aie_python_works "$_candidate"; then
            AIE_PYTHON="$_candidate"
            break
        fi
    done
    unset _mlir_aie_dir _amd_dir _candidate
fi

if ! _aie_python_works "${AIE_PYTHON:-}"; then
    echo "error: could not find a working Python 3.13 with aie + jsonschema." >&2
    echo "       set AIE_PYTHON or run setup-aie-python.sh to repair the venv." >&2
    exit 70
fi

# Resolve aiecc. Prefer the C++ binary (no Python needed, sidesteps the
# whole venv mess), fall back to the .py driver invoked through AIE_PYTHON.
# Stored as an array so the python + script form is exec'd correctly.
if [[ -n "${AIECC:-}" ]]; then
    # User override: split on whitespace so "python3 aiecc.py" works.
    # shellcheck disable=SC2206
    AIECC_CMD=( $AIECC )
else
    _mlir_aie_dir="${MLIR_AIE_DIR:-/home/triple/npu-work/mlir-aie}"
    if command -v aiecc >/dev/null 2>&1; then
        AIECC_CMD=( "$(command -v aiecc)" )
    elif [[ -x "$_mlir_aie_dir/install/bin/aiecc" ]]; then
        AIECC_CMD=( "$_mlir_aie_dir/install/bin/aiecc" )
    else
        AIECC_CMD=( "$AIE_PYTHON" "$_mlir_aie_dir/install/bin/aiecc.py" )
    fi
    unset _mlir_aie_dir
fi

USE_CHESS=0
TRACE_MODE="event_pc"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --chess) USE_CHESS=1; shift ;;
    --mode2) TRACE_MODE="inst_exec"; shift ;;
    --trace-mode)
      TRACE_MODE="$2"; shift 2
      [[ "$TRACE_MODE" =~ ^(event_time|event_pc|inst_exec)$ ]] || {
        echo "error: --trace-mode must be event_time|event_pc|inst_exec" >&2
        exit 64
      }
      ;;
    -h|--help)
      echo "usage: $0 [--chess] [--mode2|--trace-mode MODE] <fixture_dir>" >&2
      exit 0
      ;;
    --) shift; break ;;
    -*) echo "error: unknown flag: $1" >&2; exit 64 ;;
    *) break ;;
  esac
done

if [[ $# -ne 1 ]]; then
  echo "usage: $0 [--chess] [--mode2|--trace-mode MODE] <fixture_dir>" >&2
  exit 64
fi

if [[ "$USE_CHESS" -eq 1 ]]; then
  AIECC_COMPILER_FLAGS=()
else
  AIECC_COMPILER_FLAGS=(--no-xchesscc --no-xbridge)
fi

FIXTURE_DIR="$(realpath "$1")"
if [[ ! -f "$FIXTURE_DIR/kernel.cc" || ! -f "$FIXTURE_DIR/aie.mlir" ]]; then
  echo "error: $FIXTURE_DIR is missing kernel.cc or aie.mlir" >&2
  exit 65
fi

BUILD_DIR="$FIXTURE_DIR/build"
BASELINE_DIR="$BUILD_DIR/baseline"
TRACED_DIR="$BUILD_DIR/traced"

rm -rf "$BUILD_DIR"
mkdir -p "$BASELINE_DIR" "$TRACED_DIR"

# Compile kernel.cc with the SAME toolchain that aiecc will use to link.
# Peano and Chess emit AIE2 ELFs with subtly different ABIs/ELF features --
# mixing them causes the chess linker (`bridge`) to NULL-deref on a Peano
# kernel.o, while the Peano linker likewise can't ingest a chess kernel.o.
# So compile with whichever the caller picked for aiecc.
_mlir_aie_dir="${MLIR_AIE_DIR:-/home/triple/npu-work/mlir-aie}"
if [[ "$USE_CHESS" -eq 1 ]]; then
  echo "[1/5] xchesscc $FIXTURE_DIR/kernel.cc -> $BUILD_DIR/kernel.o"
  ( cd "$BUILD_DIR" && \
    "$_mlir_aie_dir/install/bin/xchesscc_wrapper" aie2 \
      -c "$FIXTURE_DIR/kernel.cc" \
      -o "$BUILD_DIR/kernel.o" )
else
  echo "[1/5] peano clang $FIXTURE_DIR/kernel.cc -> $BUILD_DIR/kernel.o"
  "$PEANO_INSTALL_DIR/bin/clang" \
    --target=aie2-none-unknown-elf -O2 \
    -c "$FIXTURE_DIR/kernel.cc" \
    -o "$BUILD_DIR/kernel.o"
fi

# aiecc looks for kernel.o relative to its working directory (because the
# link_with attribute in the MLIR is a bare filename). Stage a copy in
# each per-aiecc workdir so both invocations can find it.
cp "$BUILD_DIR/kernel.o" "$BASELINE_DIR/kernel.o"
cp "$BUILD_DIR/kernel.o" "$TRACED_DIR/kernel.o"

echo "[2/5] materialize $BASELINE_DIR/aie_arch.mlir (NPUDEVICE -> npu1_1col)"
sed 's/NPUDEVICE/npu1_1col/g' "$FIXTURE_DIR/aie.mlir" \
  > "$BASELINE_DIR/aie_arch.mlir"

echo "[3/5] aiecc baseline -> $BASELINE_DIR/{aie.xclbin, insts.bin}"
( cd "$BASELINE_DIR" && \
  "${AIECC_CMD[@]}" \
    "${AIECC_COMPILER_FLAGS[@]}" \
    --aie-generate-xclbin --xclbin-name=aie.xclbin \
    --aie-generate-npu-insts --npu-insts-name=insts.bin \
    ./aie_arch.mlir )

echo "[4/5] trace inject ($TRACE_MODE) -> $TRACED_DIR/aie_arch.mlir"
"$AIE_PYTHON" "$INJECT" \
  --trace-mode "$TRACE_MODE" \
  --input "$BASELINE_DIR/aie_arch.mlir" \
  --out   "$TRACED_DIR/aie_arch.mlir" \
  --trace-config-out "$TRACED_DIR/trace_config.json"

echo "[5/5] aiecc traced -> $TRACED_DIR/{aie.xclbin, insts.bin}"
( cd "$TRACED_DIR" && \
  "${AIECC_CMD[@]}" \
    "${AIECC_COMPILER_FLAGS[@]}" \
    --aie-generate-xclbin --xclbin-name=aie.xclbin \
    --aie-generate-npu-insts --npu-insts-name=insts.bin \
    ./aie_arch.mlir )

echo
echo "done. artifacts:"
echo "  baseline xclbin: $BASELINE_DIR/aie.xclbin"
echo "  baseline insts:  $BASELINE_DIR/insts.bin"
echo "  traced xclbin:   $TRACED_DIR/aie.xclbin"
echo "  traced insts:    $TRACED_DIR/insts.bin"
echo "  kernel.o:        $BUILD_DIR/kernel.o (disassemble with llvm-objdump -d)"
