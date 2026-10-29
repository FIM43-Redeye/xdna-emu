// VLDB_4x Characterizer
//
// Determines how VLDB_4x instructions decode addresses and read data.
// Constructs a source vector with KNOWN addresses (into the input buffer
// at p0), executes each VLDB_4x variant, stores results for analysis.
//
// Input buffer: 256 bytes of PRNG data at p0 (seed 42).
// Output buffer: 256 bytes at p1.
//
// Output layout:
//   [0:32]    Test 1: vldb.4x32.lo result
//   [32:64]   Test 2: vldb.4x32.hi result
//   [64:96]   Test 3: vldb.4x16.lo result
//   [96:128]  Test 4: vldb.4x16.hi result
//   [128:160] Test 5: vldb.4x64.lo result
//   [160:192] Test 6: vldb.4x64.hi result
//   [192:224] Test 7: address vector (verification)
//   [224:256] Test 8: vldb.4x32.lo with all addrs = p0+0x00

.text
.globl test_kernel
test_kernel:

  // ==== Prologue ====
  paddb [sp], #32
  st lr, [sp, #-4]
  st p6, [sp, #-8]
  st p7, [sp, #-12]
  mov p6, p0          // save input base
  mov p7, p1          // save output base

  // ==== Build address vector ====
  // Store 8 addresses to scratch area at [p1, #0..#28].
  // Addresses: p0+0x00, +0x20, +0x40, +0x60, +0x80, +0xA0, +0xC0, +0xE0
  // Build addresses using repeated add (AIE2 immediates are signed 8-bit).
  // Target: [p0+0, p0+32, p0+64, p0+96, p0+128, p0+160, p0+192, p0+224]
  mov r0, p0
  mov r14, #32          // increment

  // word 0: p0+0x00
  mov r1, r0
  st r1, [p1, #0]
  // word 1: p0+0x20
  add r1, r1, r14
  st r1, [p1, #4]
  // word 2: p0+0x40
  add r1, r1, r14
  st r1, [p1, #8]
  // word 3: p0+0x60
  add r1, r1, r14
  st r1, [p1, #12]
  // word 4: p0+0x80
  add r1, r1, r14
  st r1, [p1, #16]
  // word 5: p0+0xA0
  add r1, r1, r14
  st r1, [p1, #20]
  // word 6: p0+0xC0
  add r1, r1, r14
  st r1, [p1, #24]
  // word 7: p0+0xE0
  add r1, r1, r14
  st r1, [p1, #28]
  nop
  nop
  nop
  nop
  nop
  nop
  nop

  // Load address vector
  vlda wl0, [p1, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop

  // ==== Test 1: vldb.4x32.lo ====
  vldb.4x32.lo wl2, wl0
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  vst wl2, [p1, #0]

  // ==== Test 2: vldb.4x32.hi ====
  vldb.4x32.hi wl4, wl0
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  vst wl4, [p1, #32]

  // ==== Test 3: vldb.4x16.lo ====
  vldb.4x16.lo wl6, wl0
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  vst wl6, [p1, #64]

  // ==== Test 4: vldb.4x16.hi ====
  vldb.4x16.hi wl8, wl0
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  vst wl8, [p1, #96]

  // ==== Test 5: vldb.4x64.lo ====
  // Need to use post-modify for offset 128 (>127)
  mov p2, p1
  mov r14, #128
  mov m0, r14
  paddb [p2], m0
  vldb.4x64.lo wl10, wl0
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  vst wl10, [p2, #0]

  // ==== Test 6: vldb.4x64.hi ====
  vldb.4x64.hi wl2, wl0
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  vst wl2, [p2, #32]

  // ==== Test 7: Store address vector for verification ====
  vst wl0, [p2, #64]

  // ==== Test 8: vldb.4x32.lo with all 4 addrs = p0+0x00 ====
  // Build uniform address vector: all words = p0+0x00
  mov r0, p0
  st r0, [p2, #96]
  st r0, [p2, #100]
  st r0, [p2, #104]
  st r0, [p2, #108]
  st r0, [p2, #112]
  st r0, [p2, #116]
  st r0, [p2, #120]
  st r0, [p2, #124]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl2, [p2, #96]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  vldb.4x32.lo wl4, wl2
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  vst wl4, [p2, #96]

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
