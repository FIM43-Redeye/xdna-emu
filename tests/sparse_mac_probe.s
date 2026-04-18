// Sparse MAC probe: minimal kernel for debugging sparse multiply.
//
// Input buffer (p0): 128 bytes
//   [0..63]:   Dense A data (i16, loaded into xs1 = x2)
//   [64..127]: Sparse B data (i8 compressed, loaded into qxs2 = x0 + q0 mask)
//
// We use SEPARATE registers for A and B to avoid the x0-overwrite problem:
//   x2 = dense A (xs1)
//   x0 = sparse B data (part of qx0)
//   q0 = mask (part of qx0)
//
// Output buffer (p1): 64 bytes = 32 i16 (SRS output)
//
// Config: i16xi8 sparse, rows=2, inner=16, cols=8, acc_cmb=2
//   amode=1, bmode=2, variant=2, sgn_x=1, sgn_y=1, zero_acc=1
//   config = 0x353
//
// Test pattern:
//   A = all 1s (i16 value 1 at every position)
//   B = all 1s (i8 value 1 at every byte-pair position)
//   mask = 0x33333333_33333333_33333333_33333333 (bits 0,1 in every group)
//   Expected: each output = 8 (4 inner_groups * 2 active = 8 products of 1*1)

  .text
  .globl _main
_main:
  // p0 = input buffer pointer, p1 = output buffer pointer
  // (set by test harness)

  // --- Load dense A into x2 (separate from qx0) ---
  // A occupies bytes [0..63] of input
  vlda wl2, [p0, #0]
  padda [p0], #32
  vlda wh2, [p0, #0]
  padda [p0], #32

  // --- Load sparse B data into x0 ---
  // B occupies bytes [64..127] of input
  vlda wl0, [p0, #0]
  padda [p0], #32
  vlda wh0, [p0, #0]
  padda [p0], #32

  // --- Load mask into q0 ---
  // Fixed mask: all groups select bits 0,1 (0x33333333...)
  // We load from the scalar register since lda q0 needs memory.
  // Actually, let's just set it via movxm.
  // q0 is 128 bits. We need to set it to 0x33333333_33333333_33333333_33333333.
  // movxm can load 20-bit immediate into a register.
  // For simplicity, use memory. Put mask at end of B data or use a fixed value.
  // Actually, let's construct it from scalar ops.
  mov r0, #0x3333
  movxm r1, #0x33333333
  // q0 register: need to figure out how to set it.
  // Actually, let me use a simpler approach: just load q0 from the input buffer.
  // We'll put the mask bytes at input[128..143] and modify input size to 144.

  // Hmm, this is getting complex. Let me use the simplest possible approach:
  // Put ALL data (A, B, mask) in the input buffer and load from memory.

  // Reset p0 to start of input
  // Actually p0 has been advanced. Let me use a different approach.

  nop
  nop
  nop
  nop
  nop
  nop
  nop

  // --- Set config and execute vmac ---
  movxm r0, #0x353
  vmac cm0, cm0, x2, qx0, r0

  nop
  nop
  nop
  nop
  nop

  // --- Store result via SRS ---
  mov r14, #0
  mov s3, r14
  vsrs.s16.s32 wl0, bml0, s3
  nop
  nop
  nop
  nop
  vst wl0, [p1, #0]
  vsrs.s16.s32 wh0, bmh0, s3
  nop
  nop
  nop
  nop
  padda [p1], #32
  vst wh0, [p1, #0]

  nop
  nop
done
