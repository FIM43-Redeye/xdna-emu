#!/usr/bin/env bash
# Run the shuffle network characterization sweep on real NPU hardware.
#
# Usage: tests/shuffle-sweep/run.sh
#
# Produces:
#   build/shuffle-sweep/hw.bin          -- 3072 bytes of routing data
#   build/shuffle-sweep/routing.json    -- parsed routing table

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build/shuffle-sweep"
MLIR_AIE="${PROJECT_DIR}/../mlir-aie"
TEST_LIB_DIR="${MLIR_AIE}/build/runtime_lib/x86_64/test_lib"
XRT_DIR="/opt/xilinx/xrt"

IN_SIZE=128     # 128-byte identity pattern
OUT_SIZE=3072   # 48 modes x 64 bytes

mkdir -p "$BUILD_DIR"

# ---- Step 1: Generate input ----
echo "--- Generating input ---"
python3 "${SCRIPT_DIR}/gen_input.py" "${BUILD_DIR}/input.bin"

# ---- Step 2: Compile kernel with Chess ----
echo "--- Compiling kernel (Chess) ---"
nice -n 19 xchesscc_wrapper aie2 -c \
    "${SCRIPT_DIR}/shuffle_sweep.cc" \
    -o "${BUILD_DIR}/shuffle_sweep.o" 2>&1
echo "  Compiled OK"

# ---- Step 3: Generate MLIR wrapper ----
echo "--- Generating MLIR ---"
cat > "${BUILD_DIR}/aie.mlir" << 'MLIR_EOF'
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<768xi32>>

    func.func private @test_kernel(memref<32xi32>, memref<768xi32>) attributes {link_with = "shuffle_sweep.o"}

    aie.core(%tile_0_2) {
      %sub_in  = aie.objectfifo.acquire @of_in(Consume, 1)  : !aie.objectfifosubview<memref<32xi32>>
      %elem_in = aie.objectfifo.subview.access %sub_in[0]   : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
      %sub_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<768xi32>>
      %elem_out = aie.objectfifo.subview.access %sub_out[0] : !aie.objectfifosubview<memref<768xi32>> -> memref<768xi32>

      func.call @test_kernel(%elem_in, %elem_out) : (memref<32xi32>, memref<768xi32>) -> ()

      aie.objectfifo.release @of_in(Consume, 1)
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    }

    aie.runtime_sequence(%in : memref<32xi32>, %buf : memref<768xi32>, %out : memref<768xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c_in  = arith.constant 32 : i64
      %c_out = arith.constant 768 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c_out][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<768xi32>
      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c_in][%c0,%c0,%c0,%c1])  {metadata = @of_in,  id = 0 : i64, issue_token = true} : memref<32xi32>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
MLIR_EOF

# ---- Step 4: Package with aiecc.py ----
echo "--- Packaging (aiecc.py) ---"
(cd "$BUILD_DIR" && \
    nice -n 19 aiecc.py --no-aiesim --xchesscc \
        --aie-generate-xclbin --xclbin-name=aie.xclbin \
        --aie-generate-npu-insts --npu-insts-name=insts.bin \
        aie.mlir 2>&1)
echo "  Packaged OK"

# ---- Step 5: Compile test host ----
HOST_CPP="${PROJECT_DIR}/build/isa-tests/test_host.cpp"
HOST_BIN="${BUILD_DIR}/test_host"
if [[ ! -f "$HOST_BIN" ]] || [[ "$HOST_CPP" -nt "$HOST_BIN" ]]; then
    echo "--- Compiling test_host ---"
    clang++ "$HOST_CPP" -o "$HOST_BIN" \
        -std=c++17 -Wall \
        -I "${TEST_LIB_DIR}/include" \
        -I "${XRT_DIR}/include" \
        -L "${TEST_LIB_DIR}/lib" \
        -L "${XRT_DIR}/lib" \
        -ltest_utils -lxrt_coreutil \
        -lrt -lstdc++
fi

# ---- Step 6: Run on hardware ----
echo "--- Running on NPU ---"
env -u XDNA_EMU "$HOST_BIN" \
    -x "${BUILD_DIR}/aie.xclbin" \
    -k MLIR_AIE \
    -i "${BUILD_DIR}/insts.bin" \
    --in-size "$IN_SIZE" --out-size "$OUT_SIZE" \
    --in-file "${BUILD_DIR}/input.bin" \
    --out-file "${BUILD_DIR}/hw.bin" 2>"${BUILD_DIR}/hw.log"
echo "  HW run OK ($(wc -c < "${BUILD_DIR}/hw.bin") bytes)"

# ---- Step 7: Analyze ----
echo ""
echo "--- Analyzing ---"
python3 "${SCRIPT_DIR}/analyze.py" "${BUILD_DIR}/hw.bin" --json "${BUILD_DIR}/routing.json"

echo ""
echo "Output: ${BUILD_DIR}/hw.bin"
echo "Table:  ${BUILD_DIR}/routing.json"
