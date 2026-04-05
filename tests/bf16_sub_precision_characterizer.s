// bf16_sub_precision_characterizer.s -- Characterize sub-bf16 rounding in MAC
//
// Extracts full fp32 accumulator values via dual SRS (shift=16 + shift=0)
// for specific bf16 input patterns. Compares model vs hardware at bit level.
//
// Test layout (dense mode, config=0x1d, 4x8x4 geometry):
//   Input:  N test vectors, each 128 bytes (64B x0 + 64B x2)
//   Output: N * 128 bytes (64B SRS shift=16 + 64B SRS shift=0)
//
// Test layout (element-wise mode, config=0x3d, 16x2x1 geometry):
//   Same structure, different config word.
//
// Each test:
//   1. Load x0 and x2 from input buffer
//   2. vmul.f bml0, x0, x2, r0  (zero_acc, products only)
//   3. SRS shift=16 -> store upper bf16 (32 bytes)
//   4. SRS shift=0  -> store lower 16 bits (32 bytes)

  .text
  .globl test_kernel
test_kernel:

  // ==== Prologue: save callee-saved r16-r23 ====
  paddb [sp], #160
  st r16, [sp, #-4]
  st r17, [sp, #-8]
  st r18, [sp, #-12]
  st r19, [sp, #-16]
  st r20, [sp, #-20]
  st r21, [sp, #-24]
  st r22, [sp, #-28]
  st r23, [sp, #-32]

  // Zero all registers
  mov r0, #0
  mov r1, r0
  mov r2, r0
  mov r3, r0
  mov r4, r0
  mov r5, r0
  mov r6, r0
  mov r7, r0
  mov r8, r0
  mov r9, r0
  mov r10, r0
  mov r11, r0
  mov r12, r0
  mov r13, r0
  mov r14, r0
  mov r15, r0
  mov r24, r0
  mov r25, r0
  mov r26, r0
  mov r27, r0
  mov r28, r0
  mov r29, r0
  mov r30, r0
  mov r31, r0
  mov p2, r0
  mov p3, r0
  mov p4, r0
  mov p5, r0
  mov m0, r0
  mov m1, r0
  mov m2, r0
  mov m3, r0
  mov m4, r0
  mov m5, r0
  mov m6, r0
  mov m7, r0
  mov dn0, r0
  mov dn1, r0
  mov dn2, r0
  mov dn3, r0
  mov dn4, r0
  mov dn5, r0
  mov dn6, r0
  mov dn7, r0
  mov dj0, r0
  mov dj1, r0
  mov dj2, r0
  mov dj3, r0
  mov dj4, r0
  mov dj5, r0
  mov dj6, r0
  mov dj7, r0
  mov dc0, r0
  mov dc1, r0
  mov dc2, r0
  mov dc3, r0
  mov dc4, r0
  mov dc5, r0
  mov dc6, r0
  mov dc7, r0

  // Save SP for epilogue (r8 reserved from here on)
  mov r8, sp

  // p0 = input buffer base, p1 = output buffer base
  // (set by test_host via DMA before core start)

  // ================================================================
  // TEST 1: Dense mode (config=0x1d), known 0x007F case
  // Input: 128 bytes at offset 0 (64B x0 + 64B x2)
  // Output: 128 bytes at offset 0 (64B srs16 + 64B srs0)
  // ================================================================
  mov p6, p0
  vlda wl0, [p6, #0]
  padda [p6], #32
  vlda wh0, [p6, #0]
  padda [p6], #32
  vlda wl2, [p6, #0]
  padda [p6], #32
  vlda wh2, [p6, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop

  // vmul.f with config=0x1d (dense 4x8x4, zero_acc)
  mov r0, #29
  vclr cm0
  vmul.f bml0, x0, x2, r0
  nop
  nop
  nop
  nop
  nop
  nop

  // SRS shift=16: extract upper 16 bits (bf16 portion)
  mov r14, #16
  mov s3, r14
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #0]

  // SRS shift=0: extract lower 16 bits (sub-bf16 precision)
  mov r14, #0
  mov s3, r14
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #32]

  // ================================================================
  // TEST 2: Same inputs, element-wise mode (config=0x3d)
  // Output: 128 bytes at offset 128
  // ================================================================
  // Reload same data (registers may have been modified by SRS)
  mov p6, p0
  vlda wl0, [p6, #0]
  padda [p6], #32
  vlda wh0, [p6, #0]
  padda [p6], #32
  vlda wl2, [p6, #0]
  padda [p6], #32
  vlda wh2, [p6, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop

  mov r0, #61
  vclr cm0
  vmul.f bml0, x0, x2, r0
  nop
  nop
  nop
  nop
  nop
  nop

  mov r14, #16
  mov s3, r14
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #128]

  mov r14, #0
  mov s3, r14
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #160]

  // ================================================================
  // TEST 3: Second input set (dense mode, config=0x1d)
  // Input: 128 bytes at offset 128
  // Output: 128 bytes at offset 256
  // ================================================================
  mov p6, p0
  padda [p6], #128
  vlda wl0, [p6, #0]
  padda [p6], #32
  vlda wh0, [p6, #0]
  padda [p6], #32
  vlda wl2, [p6, #0]
  padda [p6], #32
  vlda wh2, [p6, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop

  mov r0, #29
  vclr cm0
  vmul.f bml0, x0, x2, r0
  nop
  nop
  nop
  nop
  nop
  nop

  mov r14, #16
  mov s3, r14
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #256]

  mov r14, #0
  mov s3, r14
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #288]

  // ================================================================
  // TEST 4: Third input set (dense mode), config=0x1d
  // Input: 128 bytes at offset 256
  // Output: 128 bytes at offset 384
  // ================================================================
  mov p6, p0
  padda [p6], #256
  vlda wl0, [p6, #0]
  padda [p6], #32
  vlda wh0, [p6, #0]
  padda [p6], #32
  vlda wl2, [p6, #0]
  padda [p6], #32
  vlda wh2, [p6, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop

  mov r0, #29
  vclr cm0
  vmul.f bml0, x0, x2, r0
  nop
  nop
  nop
  nop
  nop
  nop

  mov r14, #16
  mov s3, r14
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #384]

  mov r14, #0
  mov s3, r14
  vsrs.s16.s32 wl4, bml0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #416]

  // ==== Epilogue: restore callee-saved r16-r23 ====
  mov sp, r8
  lda r16, [sp, #-4]
  lda r17, [sp, #-8]
  lda r18, [sp, #-12]
  lda r19, [sp, #-16]
  lda r20, [sp, #-20]
  lda r21, [sp, #-24]
  lda r22, [sp, #-28]
  lda r23, [sp, #-32]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  paddb [sp], #-160
  ret lr
  nop
  nop
  nop
  nop
  nop
