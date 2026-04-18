// Sparse MAC probe: minimal isolated test for sparse multiply-accumulate.
//
// Tests i16xi8 sparse (rows=2, inner=16, cols=8, acc_cmb=2).
// Uses SEPARATE registers for A (xs1=x2) and B (qxs2=qx0=x0+q0).
//
// Output register: wl4 (dedicated, no conflict with data registers).
// Timing: 6 NOPs after vmac, 4 NOPs after vsrs (matches characterize kernel).
//
// Test 1: Identity
//   A = all i16 value 1, B = all i8 value 1, mask = 0x33...
//   Expected: each acc lane = 8 (4 inner_groups * 2 active per group)
//   SRS shift=0: lo16 of 8 = 8
//
// Test 2: Counting
//   A = all i16 value 1, B compressed = {1,2,3,4,5,6,7,8,0...}
//   mask = 0x33..., Expected col 0 acc = 36 (1+2+...+8)
//
// p0 = input buffer (scratch, 128 bytes)
// p1 = output buffer (128 bytes: 2 tests * 64 bytes each)

.text
.globl test_kernel
test_kernel:

  // ====== BUILD CONSTANTS ======

  // r1 = 0x00010001 (i16 value 1 repeated)
  mov r1, #1
  mov r3, #1
  add r3, r3, r3
  add r3, r3, r3
  add r3, r3, r3
  add r3, r3, r3
  add r3, r3, r3
  add r3, r3, r3
  add r3, r3, r3
  add r3, r3, r3
  add r3, r3, r3
  add r3, r3, r3
  add r3, r3, r3
  add r3, r3, r3
  add r3, r3, r3
  add r3, r3, r3
  add r3, r3, r3
  add r3, r3, r3
  or r1, r1, r3

  // r4 = 0x01010101 (i8 value 1 repeated)
  mov r4, #1
  mov r5, r4
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  or r4, r4, r5
  mov r5, r4
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  add r5, r5, r5
  or r4, r4, r5

  // r6 = 0x33333333 (mask: bits 0,1 in every nibble)
  mov r6, #3
  mov r7, r6
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  or r6, r6, r7
  mov r7, r6
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  or r6, r6, r7
  mov r7, r6
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  add r7, r7, r7
  or r6, r6, r7

  // r8 = 0 (for zeroing)
  mov r8, #0

  // ====== TEST 1: Identity (A=1 i16, B=1 i8, mask=0x33) ======

  // Write A data (0x00010001) to scratch, load into x2
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  st r1, [p0, #16]
  st r1, [p0, #20]
  st r1, [p0, #24]
  st r1, [p0, #28]
  nop
  vlda wl2, [p0, #0]
  vlda wh2, [p0, #0]

  // Write B data (0x01010101) to scratch, load into x0
  st r4, [p0, #0]
  st r4, [p0, #4]
  st r4, [p0, #8]
  st r4, [p0, #12]
  st r4, [p0, #16]
  st r4, [p0, #20]
  st r4, [p0, #24]
  st r4, [p0, #28]
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]

  // Write mask (0x33333333) to scratch, load into q0
  st r6, [p0, #0]
  st r6, [p0, #4]
  st r6, [p0, #8]
  st r6, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop

  // Set SRS shift = 0
  mov r14, #0
  mov s3, r14

  // Execute sparse MAC
  movxm r0, #0x353
  vmac cm0, cm0, x2, qx0, r0
  nop
  nop
  nop
  nop
  nop
  nop
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #0]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #32]

  // ====== TEST 2: Counting (A=1, B={1,2,...,8,0...}, mask=0x33) ======

  // Build r9 = 0x04030201 (bytes 0-3 of B)
  mov r9, #1
  mov r10, r9
  add r10, r10, r10
  add r10, r10, r10
  add r10, r10, r10
  add r10, r10, r10
  add r10, r10, r10
  add r10, r10, r10
  add r10, r10, r10
  add r10, r10, r10
  or r9, r9, r10
  mov r10, #3
  mov r11, r10
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  or r9, r9, r11
  mov r10, #4
  mov r11, r10
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  or r9, r9, r11

  // Build r10 = 0x08070605 (bytes 4-7 of B)
  mov r10, #5
  mov r11, r10
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  add r11, r11, r11
  or r10, r10, r11
  mov r11, #7
  mov r12, r11
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  or r10, r10, r12
  mov r11, #8
  mov r12, r11
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  add r12, r12, r12
  or r10, r10, r12

  // Store B to scratch: bytes {1,2,3,4,5,6,7,8} then zeros
  st r9,  [p0, #0]
  st r10, [p0, #4]
  st r8,  [p0, #8]
  st r8,  [p0, #12]
  st r8,  [p0, #16]
  st r8,  [p0, #20]
  st r8,  [p0, #24]
  st r8,  [p0, #28]
  nop
  vlda wl0, [p0, #0]

  // Upper half all zeros
  st r8, [p0, #0]
  st r8, [p0, #4]
  st r8, [p0, #8]
  st r8, [p0, #12]
  st r8, [p0, #16]
  st r8, [p0, #20]
  st r8, [p0, #24]
  st r8, [p0, #28]
  nop
  vlda wh0, [p0, #0]

  // Mask still in q0 from test 1
  // A still in x2 from test 1

  nop
  nop
  nop
  nop
  nop
  nop
  nop

  // Execute sparse MAC
  movxm r0, #0x353
  vmac cm0, cm0, x2, qx0, r0
  nop
  nop
  nop
  nop
  nop
  nop
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #64]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #96]

  // Pipeline drain
  ret lr
  nop
  nop
  nop
  nop
  nop
