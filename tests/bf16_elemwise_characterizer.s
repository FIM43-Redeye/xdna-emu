// bf16 element-wise MAC characterizer
//
// Determines the A-side and B-side element permutation for variant=1
// (config=0x3d) bf16 multiply: 16x2x1 geometry (16 lanes, inner=2, cols=1).
//
// Uses Sidon set values (all pairwise sums are unique) to identify
// which element indices contribute to each output lane.
//
// Test 1: A = Sidon values, B = all bf16(1.0)
//         Output lane L = Sidon[a0] + Sidon[a1]  (reveals A-side indices)
//
// Test 2: A = all bf16(1.0), B = Sidon values
//         Output lane L = Sidon[b0] + Sidon[b1]  (reveals B-side indices)
//
// Test 3: A = B = Sidon values (same register x0)
//         Cross-check.
//
// Each test stores the full 32-bit fp32 accumulator via TWO SRS passes:
//   SRS shift=16 -> high 16 bits (bf16-like)
//   SRS shift=0  -> low 16 bits
// The analysis script reconstructs the fp32 from both halves.
//
// Input buffer layout (256 bytes):
//   [0:64]    Sidon set as bf16 (32 elements)
//   [64:128]  bf16(1.0) x 32 elements
//   [128:192] bf16(1.0) x 32 elements
//   [192:256] Sidon set as bf16 (32 elements, same as [0:64])
//
// Output buffer layout (192 bytes):
//   [0:32]    Test 1 high (SRS shift=16)
//   [32:64]   Test 1 low  (SRS shift=0)
//   [64:96]   Test 2 high (SRS shift=16)
//   [96:128]  Test 2 low  (SRS shift=0)
//   [128:160] Test 3 high (SRS shift=16)
//   [160:192] Test 3 low  (SRS shift=0)

.text
.globl test_kernel
test_kernel:

  // ==== Prologue ====
  paddb [sp], #32
  st lr, [sp, #-4]
  st p6, [sp, #-8]
  st p7, [sp, #-12]

  // ==== Zero all vector/accumulator state ====
  mov r0, #0
  vbcst.32 x0, r0
  vbcst.32 x2, r0
  vbcst.32 x4, r0
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  vclr cm0
  vclr cm2

  // ==== Test 1: A=Sidon, B=ones ====
  // Load B (ones) into x0: input[64:128]
  mov p6, p0
  mov r14, #64
  mov m0, r14
  paddb [p6], m0          // p6 = p0+64
  vlda wl0, [p6, #0]      // x0 lo = input[64:95]
  vlda wh0, [p6, #32]     // x0 hi = input[96:127]
  // Load A (Sidon) into x2: input[0:64]
  vlda wl2, [p0, #0]      // x2 lo = input[0:31]
  vlda wh2, [p0, #32]     // x2 hi = input[32:63]
  nop
  nop
  nop
  nop
  nop
  nop
  nop

  // Execute: vmul.f bml0, x2(A), x0(B), config=0x3d
  vclr cm0
  mov r0, #61
  vmul.f bml0, x2, x0, r0
  nop
  nop
  nop
  nop
  nop
  nop

  // Store bml0 high half: SRS shift=16
  mov r14, #16
  mov s3, r14
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #0]       // output[0:31]

  // Store bml0 low half: SRS shift=0
  mov r14, #0
  mov s3, r14
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #32]      // output[32:63]

  // ==== Test 2: A=ones, B=Sidon ====
  // Load B (Sidon) into x0: input[192:256]
  mov p6, p0
  mov r14, #64
  mov m0, r14
  paddb [p6], m0          // p6 = p0+64
  paddb [p6], m0          // p6 = p0+128
  paddb [p6], m0          // p6 = p0+192
  vlda wl0, [p6, #0]      // x0 lo = input[192:223]
  vlda wh0, [p6, #32]     // x0 hi = input[224:255]
  // Load A (ones) into x2: input[128:192]
  mov p6, p0
  paddb [p6], m0          // p6 = p0+64
  paddb [p6], m0          // p6 = p0+128
  vlda wl2, [p6, #0]      // x2 lo = input[128:159]
  vlda wh2, [p6, #32]     // x2 hi = input[160:191]
  nop
  nop
  nop
  nop
  nop
  nop
  nop

  // Execute: vmul.f bml0, x2(A), x0(B), config=0x3d
  vclr cm0
  mov r0, #61
  vmul.f bml0, x2, x0, r0
  nop
  nop
  nop
  nop
  nop
  nop

  // Store bml0 high: SRS shift=16
  mov r14, #16
  mov s3, r14
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #64]      // output[64:95]

  // Store bml0 low: SRS shift=0
  mov r14, #0
  mov s3, r14
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #96]      // output[96:127]

  // ==== Test 3: A=B=Sidon (same register) ====
  // Load Sidon into x0: input[0:64]
  vlda wl0, [p0, #0]      // x0 lo = input[0:31]
  vlda wh0, [p0, #32]     // x0 hi = input[32:63]
  nop
  nop
  nop
  nop
  nop
  nop
  nop

  // Execute: vmul.f bml0, x0, x0, config=0x3d  (A=B=Sidon)
  vclr cm0
  mov r0, #61
  vmul.f bml0, x0, x0, r0
  nop
  nop
  nop
  nop
  nop
  nop

  // Store bml0 high: SRS shift=16
  mov r14, #16
  mov s3, r14
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #128]     // output[128:159]

  // Store bml0 low: SRS shift=0
  mov r14, #0
  mov s3, r14
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #160]     // output[160:191]

  // ==== Epilogue ====
  lda lr, [sp, #-4]
  lda p6, [sp, #-8]
  lda p7, [sp, #-12]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  paddb [sp], #-32
  ret lr
  nop
  nop
  nop
  nop
  nop
