// Shuffle network characterization kernel.
//
// Sweeps all 48 one-hot shuffle modes on the AIE2 permutation network.
// Input:  128 bytes identity pattern (byte[i] = i) as two 64-byte vectors.
// Output: 48 x 64 bytes = 3072 bytes, one 512-bit result per mode.
//
// Compile with Chess: xchesscc_wrapper aie2 -c shuffle_sweep.cc -o shuffle_sweep.o
// Package with aiecc.py for NPU execution.
//
// Cleanroom: observes hardware behavior of shuffle intrinsic on owned hardware.

#include <stdint.h>

// Chess intrinsics for AIE2 shuffle.
// shuffle(a, b, mode) invokes the VSHUFFLE instruction which feeds
// both vectors through the permutation network with mode as control.
extern v64int8 shuffle(v64int8 a, v64int8 b, unsigned int mode);

extern "C" {

void test_kernel(uint32_t* __restrict in_buf, uint32_t* __restrict out_buf) {
    // Load two 512-bit vectors from input.
    // A = bytes 0-63 (identity: byte[i] = i)
    // B = bytes 64-127 (identity continued: byte[64+i] = 64+i)
    v64int8 a_vec = *(v64int8*)in_buf;
    v64int8 b_vec = *(v64int8*)((uint8_t*)in_buf + 64);

    uint8_t* out = (uint8_t*)out_buf;

    // Sweep all 48 valid one-hot modes.
    // Mode N sets bit N in the 48-bit control mask: mask = 1 << N.
    // Modes 0-47 each produce a unique permutation of the 128 input bytes,
    // selecting 64 output bytes.
    for (int mode = 0; mode < 48; mode++) {
        v64int8 result = shuffle(a_vec, b_vec, mode);
        *(v64int8*)(out + mode * 64) = result;
    }
}

} // extern "C"
