// Sparse MAC mask characterization test.
// Self-contained: constructs A=0x10..., B=0x10... via scalar stores + vector loads.
// Tests all 16 mask patterns in bits 0-3 of q0.
//
// Config: i8xi8 sparse narrow (4x16x8), zero_acc=1, signed.
// Output: 16 x 64 bytes = 1024 bytes (bml0 + bmh0 via SRS per pattern).
//
// p0 = input buffer (used as scratch for constructing data)
// p1 = output buffer (1024 bytes)

.text
.globl test_kernel
test_kernel:

  // -- Construct r1 = 0x10101010 --
  // Use 0x10 (16) per byte so products survive SRS bias of 4 bits.
  // SRS with shift=0 still applies BIAS=4, so value must be >= 16.
  // 16*16=256 per active position; 256>>4 = 16 in output.
  // AIE2 MOV scalar has 10-bit signed immediate (-512..511).
  mov r1, #0x10            // r1 = 0x00000010
  mov r2, #0x10            // r2 = 0x00000010
  // Shift r2 left by 8 to get 0x00001000
  add r2, r2, r2           // 0x20
  add r2, r2, r2           // 0x40
  add r2, r2, r2           // 0x80
  add r2, r2, r2           // 0x100
  add r2, r2, r2           // 0x200
  add r2, r2, r2           // 0x400
  add r2, r2, r2           // 0x800
  add r2, r2, r2           // 0x1000
  or r1, r1, r2            // r1 = 0x00001010
  // Now build upper half: shift r1 left by 16
  mov r2, r1               // r2 = 0x00001010
  add r2, r2, r2           // 0x2020
  add r2, r2, r2           // 0x4040
  add r2, r2, r2           // 0x8080
  add r2, r2, r2           // 0x10100
  add r2, r2, r2           // 0x20200
  add r2, r2, r2           // 0x40400
  add r2, r2, r2           // 0x80800
  add r2, r2, r2           // 0x101000
  add r2, r2, r2           // 0x202000
  add r2, r2, r2           // 0x404000
  add r2, r2, r2           // 0x808000
  add r2, r2, r2           // 0x1010000
  add r2, r2, r2           // 0x2020000
  add r2, r2, r2           // 0x4040000
  add r2, r2, r2           // 0x8080000
  add r2, r2, r2           // 0x10100000
  or r1, r1, r2            // r1 = 0x10101010

  // Fill 32 bytes at p0[0..31] with 0x01010101
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  st r1, [p0, #16]
  st r1, [p0, #20]
  st r1, [p0, #24]
  st r1, [p0, #28]
  nop
  nop
  nop
  nop
  nop
  nop

  // Load A into x2 (wl2 + wh2 = 512 bits = 64 bytes)
  vlda wl2, [p0, #0]
  vlda wh2, [p0, #0]

  // -- SRS shift = 0 --
  mov r14, #0
  mov s3, r14

  // -- Config: i8xi8 sparse narrow, zero_acc=1, sgn_x=1, sgn_y=1 --
  // = 1 | (0<<1) | (1<<3) | (5<<5) | (1<<8) | (1<<9) = 937 = 0x3A9
  // MOV scalar has 10-bit signed immediate (-512..511), so 937 doesn't fit.
  // Build from 0xA9 (169) | (3 << 8) = 0x3A9.
  mov r0, #0xA9             // r0 = 169 (low byte)
  mov r3, #3                // r3 = 3
  add r3, r3, r3            // 6
  add r3, r3, r3            // 12
  add r3, r3, r3            // 24
  add r3, r3, r3            // 48
  add r3, r3, r3            // 96
  add r3, r3, r3            // 192
  add r3, r3, r3            // 384
  add r3, r3, r3            // 768 = 0x300
  or r0, r0, r3             // r0 = 0xA9 | 0x300 = 0x3A9 = 937

  nop
  nop
  nop
  nop
  nop
  nop
  nop

  // ============ Macro-like pattern: for each P in 0..15 ============
  // Each block:
  //   1. Write mask P to scratch, load q0
  //   2. Load B from scratch (all 0x01s) into x0
  //   3. vmac cm0, cm0, x2, qx0, r0
  //   4. SRS bml0 -> wl4, store; SRS bmh0 -> wl4, store

  // Pattern 0: mask=0b0000
  mov r3, #0
  st r3, [p0, #0]
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  // Restore scratch to 0x01010101 for B load
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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

  // Pattern 1: mask=0b0001
  mov r3, #1
  st r3, [p0, #0]
  mov r3, #0
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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

  // Pattern 2: mask=0b0010
  mov r3, #2
  st r3, [p0, #0]
  mov r3, #0
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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
  vst wl4, [p1, #128]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #160]

  // Pattern 3: mask=0b0011
  mov r3, #3
  st r3, [p0, #0]
  mov r3, #0
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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
  vst wl4, [p1, #192]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #224]

  // Pattern 4: mask=0b0100
  mov r3, #4
  st r3, [p0, #0]
  mov r3, #0
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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
  vst wl4, [p1, #256]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #288]

  // Pattern 5: mask=0b0101
  mov r3, #5
  st r3, [p0, #0]
  mov r3, #0
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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
  vst wl4, [p1, #320]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #352]

  // Pattern 6: mask=0b0110
  mov r3, #6
  st r3, [p0, #0]
  mov r3, #0
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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
  vst wl4, [p1, #384]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #416]

  // Pattern 7: mask=0b0111
  mov r3, #7
  st r3, [p0, #0]
  mov r3, #0
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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
  vst wl4, [p1, #448]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #480]

  // Pattern 8: mask=0b1000
  mov r3, #8
  st r3, [p0, #0]
  mov r3, #0
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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
  vst wl4, [p1, #512]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #544]

  // Pattern 9: mask=0b1001
  mov r3, #9
  st r3, [p0, #0]
  mov r3, #0
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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
  vst wl4, [p1, #576]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #608]

  // Pattern 10: mask=0b1010
  mov r3, #10
  st r3, [p0, #0]
  mov r3, #0
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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
  vst wl4, [p1, #640]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #672]

  // Pattern 11: mask=0b1011
  mov r3, #11
  st r3, [p0, #0]
  mov r3, #0
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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
  vst wl4, [p1, #704]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #736]

  // Pattern 12: mask=0b1100
  mov r3, #12
  st r3, [p0, #0]
  mov r3, #0
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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
  vst wl4, [p1, #768]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #800]

  // Pattern 13: mask=0b1101
  mov r3, #13
  st r3, [p0, #0]
  mov r3, #0
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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
  vst wl4, [p1, #832]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #864]

  // Pattern 14: mask=0b1110
  mov r3, #14
  st r3, [p0, #0]
  mov r3, #0
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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
  vst wl4, [p1, #896]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #928]

  // Pattern 15: mask=0b1111
  mov r3, #15
  st r3, [p0, #0]
  mov r3, #0
  st r3, [p0, #4]
  st r3, [p0, #8]
  st r3, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  lda q0, [p0, #0]
  st r1, [p0, #0]
  st r1, [p0, #4]
  st r1, [p0, #8]
  st r1, [p0, #12]
  nop
  nop
  nop
  nop
  nop
  nop
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #0]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
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
  vst wl4, [p1, #960]
  vsrs.s16.s32 wl4, bmh0, s3
  nop
  nop
  nop
  nop
  vst wl4, [p1, #992]

  // Return to caller (wrapper releases locks, triggers DMA)
  ret lr
  nop
  nop
  nop
  nop
  nop
