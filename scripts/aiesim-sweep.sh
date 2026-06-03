#!/bin/bash
# In-process breadth sweep of the aiesim backend over the mlir-aie npu-xrt kernel
# corpus. Each kernel runs through the DIRECT FFI path (XDNA_BACKEND=aiesim) -- the
# proven path the e2e harness uses -- in its OWN process under `timeout`, so a hang
# or crash in one kernel is recorded, not fatal to the sweep.
#
# This is the "how broadly does the aiesim backend handle real kernels" signal. It
# does NOT validate numerical correctness for arbitrary kernels (no per-kernel
# golden data); it classifies each kernel as RAN / EXEC_FAIL / RUN_FAIL / LOAD_FAIL
# / HANG / CRASH, and opportunistically checks the add_<N> int32 family (out[i] == i+N).
#
# Env (all have sensible defaults):
#   XDNA_AIESIM_DEVICE_JSON  native NPU1 device file (default: local all-NoC NPU1.json)
#   AIESIM_SWEEP_COMPILER    chess | peano (default chess)
#   AIESIM_SWEEP_TIMEOUT     per-kernel wall timeout seconds (default 120)
#   AIESIM_SWEEP_FILTER      substring filter on kernel name (default: all)
set -u

EMU=/home/triple/npu-work/xdna-emu
KROOT=/home/triple/npu-work/mlir-aie/build/test/npu-xrt
COMPILER="${AIESIM_SWEEP_COMPILER:-chess}"
# Default 300s: packet-switched-routing kernels (packet_flow*) are correct but
# slow to simulate (packet arbitration + per-cycle FIFO modeling), ~140-260s.
# A 120s timeout misclassified them as hangs; 300s covers them with margin.
TIMEOUT="${AIESIM_SWEEP_TIMEOUT:-300}"
FILTER="${AIESIM_SWEEP_FILTER:-}"
DEVJSON="${XDNA_AIESIM_DEVICE_JSON:-$EMU/build/experiments/aiesim-device-decrypt/NPU1.json}"
STAMP="$(date +%Y%m%d-%H%M%S 2>/dev/null || echo run)"
OUTDIR="$EMU/build/experiments/aiesim-sweep/$STAMP"
mkdir -p "$OUTDIR"

source /home/triple/npu-work/toolchain-build/activate-npu-env.sh >/dev/null 2>&1
export LD_LIBRARY_PATH="$XILINX_VITIS_AIETOOLS/lib/lnx64.o:$EMU/aiesim-bridge/build:$LD_LIBRARY_PATH"
export XDNA_BACKEND=aiesim
export XDNA_AIESIM_DEVICE_JSON="$DEVJSON"
export XDNA_AIESIM_BRIDGE="$EMU/aiesim-bridge/build/libxdna_aiesim_bridge.so"
export XDNA_AIESIM_NATIVE_GEOMETRY=1
export XDNA_AIESIM_POLL_MAX_NS="${XDNA_AIESIM_POLL_MAX_NS:-200000}"
export TMPDIR=/tmp/claude-1000
[[ -f "$DEVJSON" ]] || { echo "device json not found: $DEVJSON" >&2; exit 2; }

echo "Building sweep runner..."
( cd "$EMU" && nice -n 19 cargo test -p xdna-emu-ffi --features aiesim --test aiesim_sweep --no-run ) 2>&1 | tail -3
BIN="$(ls -t "$EMU"/target/debug/deps/aiesim_sweep-* 2>/dev/null | grep -v '\.d$' | head -1)"
[[ -x "$BIN" ]] || { echo "sweep test binary not found" >&2; exit 2; }
echo "Runner: $BIN"
echo "Device: $DEVJSON"
echo "Compiler: $COMPILER  Timeout: ${TIMEOUT}s  Out: $OUTDIR"
echo

summary="$OUTDIR/summary.tsv"
printf "kernel\tclass\texec\trun\thalt\tcycles\tcorrect\n" > "$summary"
declare -A tally

for kdir in "$KROOT"/*/"$COMPILER"; do
  [[ -f "$kdir/aie.xclbin" && -f "$kdir/insts.bin" ]] || continue
  name="$(basename "$(dirname "$kdir")")"
  [[ -z "$FILTER" || "$name" == *"$FILTER"* ]] || continue

  log="$OUTDIR/$name.log"
  AIESIM_KERNEL_DIR="$kdir" timeout "$TIMEOUT" "$BIN" --ignored --exact --nocapture sweep_one_kernel_aiesim \
    > "$log" 2>&1
  rc=$?

  exec_v="$(sed -n 's/.*SWEEP exec=\([A-Za-z]*\).*/\1/p' "$log" | head -1)"
  run_v="$(sed -n 's/.*SWEEP run=\([A-Za-z]*\).*/\1/p' "$log" | head -1)"
  halt_v="$(sed -n 's/.*SWEEP halt=\(.*\)/\1/p' "$log" | head -1)"
  cyc_v="$(sed -n 's/.*SWEEP cycles=\([0-9]*\).*/\1/p' "$log" | head -1)"
  done_v="$(sed -n 's/.*SWEEP done=\([A-Za-z_]*\).*/\1/p' "$log" | head -1)"

  # Classify.
  if [[ $rc -eq 124 ]]; then
    class="HANG"
  elif [[ "$done_v" == "load_failed" ]]; then
    class="LOAD_FAIL"
  elif [[ -z "$done_v" ]]; then
    class="CRASH"
  elif [[ "$exec_v" != "Success" ]]; then
    class="EXEC_FAIL"
  elif [[ "$run_v" != "Success" ]]; then
    class="RUN_FAIL"
  else
    class="RAN"
  fi

  # Opportunistic correctness for add_<N> int32: out[i] == i+N in some buffer.
  correct="-"
  if [[ "$name" =~ ^add_([0-9]+)_ && "$name" != *i8* ]]; then
    n="${BASH_REMATCH[1]}"
    want="[$n, $((n+1)), $((n+2)), $((n+3)), $((n+4)), $((n+5)), $((n+6)), $((n+7))]"
    if grep -qF "$want" "$log"; then correct="PASS"; else correct="MISMATCH"; fi
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$name" "$class" "${exec_v:--}" "${run_v:--}" "${halt_v:--}" "${cyc_v:--}" "$correct" >> "$summary"
  tally[$class]=$(( ${tally[$class]:-0} + 1 ))
  printf "  %-52s %-10s exec=%-8s run=%-8s %s\n" "$name" "$class" "${exec_v:--}" "${run_v:--}" \
    "$([[ "$correct" != "-" ]] && echo "correct=$correct")"
done

echo
echo "=== TALLY ($COMPILER) ==="
for k in RAN EXEC_FAIL RUN_FAIL LOAD_FAIL HANG CRASH; do
  printf "  %-10s %d\n" "$k" "${tally[$k]:-0}"
done
echo
echo "Summary: $summary"
column -t -s$'\t' "$summary"
