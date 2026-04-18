/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|*  * AIE2 Disassembler                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


#include <assert.h>

namespace llvm {

// Helper functions for extracting fields from encoded instructions.
// InsnType must either be integral or an APInt-like object that must:
// * be default-constructible and copy-constructible
// * be constructible from an APInt (this can be private)
// * Support insertBits(bits, startBit, numBits)
// * Support extractBitsAsZExtValue(numBits, startBit)
// * Support the ~, &, ==, and != operators with other objects of the same type
// * Support the != and bitwise & with uint64_t
// * Support put (<<) to raw_ostream&
template <typename InsnType>
#if defined(_MSC_VER) && !defined(__clang__)
__declspec(noinline)
#endif
static std::enable_if_t<std::is_integral<InsnType>::value, InsnType>
fieldFromInstruction(const InsnType &insn, unsigned startBit,
                     unsigned numBits) {
  assert(startBit + numBits <= 64 && "Cannot support >64-bit extractions!");
  assert(startBit + numBits <= (sizeof(InsnType) * 8) &&
         "Instruction field out of bounds!");
  InsnType fieldMask;
  if (numBits == sizeof(InsnType) * 8)
    fieldMask = (InsnType)(-1LL);
  else
    fieldMask = (((InsnType)1 << numBits) - 1) << startBit;
  return (insn & fieldMask) >> startBit;
}

template <typename InsnType>
static std::enable_if_t<!std::is_integral<InsnType>::value, uint64_t>
fieldFromInstruction(const InsnType &insn, unsigned startBit,
                     unsigned numBits) {
  return insn.extractBitsAsZExtValue(numBits, startBit);
}

// Helper function for inserting bits extracted from an encoded instruction into
// a field.
template <typename InsnType>
static std::enable_if_t<std::is_integral<InsnType>::value>
insertBits(InsnType &field, InsnType bits, unsigned startBit, unsigned numBits) {
  assert(startBit + numBits <= sizeof field * 8);
  field |= (InsnType)bits << startBit;
}

template <typename InsnType>
static std::enable_if_t<!std::is_integral<InsnType>::value>
insertBits(InsnType &field, uint64_t bits, unsigned startBit, unsigned numBits) {
  field.insertBits(bits, startBit, numBits);
}

static bool Check(DecodeStatus &Out, DecodeStatus In) {
  Out = static_cast<DecodeStatus>(Out & In);
  return Out != MCDisassembler::Fail;
}

static const uint8_t DecoderTableAlu32[] = {
/* 0 */       MCD::OPC_ExtractField, 0, 3,  // Inst{2-0} ...
/* 3 */       MCD::OPC_FilterValue, 0, 4, 1, 0, // Skip to: 268
/* 8 */       MCD::OPC_ExtractField, 3, 4,  // Inst{6-3} ...
/* 11 */      MCD::OPC_FilterValue, 0, 11, 0, 0, // Skip to: 27
/* 16 */      MCD::OPC_CheckField, 7, 13, 0, 129, 2, 0, // Skip to: 664
/* 23 */      MCD::OPC_Decode, 188, 5, 0, // Opcode: NOPX
/* 27 */      MCD::OPC_FilterValue, 1, 14, 0, 0, // Skip to: 46
/* 32 */      MCD::OPC_CheckField, 10, 10, 0, 113, 2, 0, // Skip to: 664
/* 39 */      MCD::OPC_TryDecode, 230, 4, 1, 106, 2, 0, // Opcode: J_jump_ind, skip to: 664
/* 46 */      MCD::OPC_FilterValue, 2, 11, 0, 0, // Skip to: 62
/* 51 */      MCD::OPC_CheckField, 7, 13, 0, 94, 2, 0, // Skip to: 664
/* 58 */      MCD::OPC_Decode, 136, 4, 0, // Opcode: DONE
/* 62 */      MCD::OPC_FilterValue, 3, 7, 0, 0, // Skip to: 74
/* 67 */      MCD::OPC_TryDecode, 227, 4, 2, 78, 2, 0, // Opcode: JNZD, skip to: 664
/* 74 */      MCD::OPC_FilterValue, 4, 11, 0, 0, // Skip to: 90
/* 79 */      MCD::OPC_CheckField, 7, 11, 0, 66, 2, 0, // Skip to: 664
/* 86 */      MCD::OPC_Decode, 139, 4, 3, // Opcode: EVENT
/* 90 */      MCD::OPC_FilterValue, 5, 14, 0, 0, // Skip to: 109
/* 95 */      MCD::OPC_CheckField, 10, 10, 0, 50, 2, 0, // Skip to: 664
/* 102 */     MCD::OPC_TryDecode, 225, 4, 1, 43, 2, 0, // Opcode: JL_IND, skip to: 664
/* 109 */     MCD::OPC_FilterValue, 6, 11, 0, 0, // Skip to: 125
/* 114 */     MCD::OPC_CheckField, 7, 13, 0, 31, 2, 0, // Skip to: 664
/* 121 */     MCD::OPC_Decode, 209, 5, 0, // Opcode: RET
/* 125 */     MCD::OPC_FilterValue, 7, 63, 0, 0, // Skip to: 193
/* 130 */     MCD::OPC_ExtractField, 7, 3,  // Inst{9-7} ...
/* 133 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 145
/* 138 */     MCD::OPC_TryDecode, 133, 4, 4, 7, 2, 0, // Opcode: CLB, skip to: 664
/* 145 */     MCD::OPC_FilterValue, 1, 7, 0, 0, // Skip to: 157
/* 150 */     MCD::OPC_TryDecode, 141, 4, 4, 251, 1, 0, // Opcode: EXTENDs8, skip to: 664
/* 157 */     MCD::OPC_FilterValue, 2, 7, 0, 0, // Skip to: 169
/* 162 */     MCD::OPC_TryDecode, 143, 4, 4, 239, 1, 0, // Opcode: EXTENDu8, skip to: 664
/* 169 */     MCD::OPC_FilterValue, 3, 7, 0, 0, // Skip to: 181
/* 174 */     MCD::OPC_TryDecode, 138, 4, 4, 227, 1, 0, // Opcode: EQZ, skip to: 664
/* 181 */     MCD::OPC_FilterValue, 4, 222, 1, 0, // Skip to: 664
/* 186 */     MCD::OPC_TryDecode, 250, 3, 4, 215, 1, 0, // Opcode: ABS, skip to: 664
/* 193 */     MCD::OPC_FilterValue, 8, 14, 0, 0, // Skip to: 212
/* 198 */     MCD::OPC_CheckField, 7, 4, 0, 203, 1, 0, // Skip to: 664
/* 205 */     MCD::OPC_TryDecode, 148, 5, 5, 196, 1, 0, // Opcode: MOVX_mvx_scl, skip to: 664
/* 212 */     MCD::OPC_FilterValue, 15, 191, 1, 0, // Skip to: 664
/* 217 */     MCD::OPC_ExtractField, 7, 3,  // Inst{9-7} ...
/* 220 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 232
/* 225 */     MCD::OPC_TryDecode, 134, 4, 4, 176, 1, 0, // Opcode: CLZ, skip to: 664
/* 232 */     MCD::OPC_FilterValue, 1, 7, 0, 0, // Skip to: 244
/* 237 */     MCD::OPC_TryDecode, 140, 4, 4, 164, 1, 0, // Opcode: EXTENDs16, skip to: 664
/* 244 */     MCD::OPC_FilterValue, 2, 7, 0, 0, // Skip to: 256
/* 249 */     MCD::OPC_TryDecode, 142, 4, 4, 152, 1, 0, // Opcode: EXTENDu16, skip to: 664
/* 256 */     MCD::OPC_FilterValue, 3, 147, 1, 0, // Skip to: 664
/* 261 */     MCD::OPC_TryDecode, 181, 5, 4, 140, 1, 0, // Opcode: NEZ, skip to: 664
/* 268 */     MCD::OPC_FilterValue, 1, 51, 0, 0, // Skip to: 324
/* 273 */     MCD::OPC_ExtractField, 3, 2,  // Inst{4-3} ...
/* 276 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 288
/* 281 */     MCD::OPC_TryDecode, 128, 4, 6, 120, 1, 0, // Opcode: ADD, skip to: 664
/* 288 */     MCD::OPC_FilterValue, 1, 7, 0, 0, // Skip to: 300
/* 293 */     MCD::OPC_TryDecode, 131, 4, 6, 108, 1, 0, // Opcode: AND, skip to: 664
/* 300 */     MCD::OPC_FilterValue, 2, 7, 0, 0, // Skip to: 312
/* 305 */     MCD::OPC_TryDecode, 180, 5, 6, 96, 1, 0, // Opcode: NE, skip to: 664
/* 312 */     MCD::OPC_FilterValue, 3, 91, 1, 0, // Skip to: 664
/* 317 */     MCD::OPC_TryDecode, 144, 5, 6, 84, 1, 0, // Opcode: LTU, skip to: 664
/* 324 */     MCD::OPC_FilterValue, 2, 7, 0, 0, // Skip to: 336
/* 329 */     MCD::OPC_TryDecode, 147, 5, 7, 72, 1, 0, // Opcode: MOVX_alu_cg, skip to: 664
/* 336 */     MCD::OPC_FilterValue, 3, 51, 0, 0, // Skip to: 392
/* 341 */     MCD::OPC_ExtractField, 3, 2,  // Inst{4-3} ...
/* 344 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 356
/* 349 */     MCD::OPC_TryDecode, 240, 5, 6, 52, 1, 0, // Opcode: SUB, skip to: 664
/* 356 */     MCD::OPC_FilterValue, 1, 7, 0, 0, // Skip to: 368
/* 361 */     MCD::OPC_TryDecode, 190, 5, 6, 40, 1, 0, // Opcode: OR, skip to: 664
/* 368 */     MCD::OPC_FilterValue, 2, 7, 0, 0, // Skip to: 380
/* 373 */     MCD::OPC_TryDecode, 144, 4, 6, 28, 1, 0, // Opcode: GE, skip to: 664
/* 380 */     MCD::OPC_FilterValue, 3, 23, 1, 0, // Skip to: 664
/* 385 */     MCD::OPC_TryDecode, 142, 5, 6, 16, 1, 0, // Opcode: LSHL, skip to: 664
/* 392 */     MCD::OPC_FilterValue, 4, 143, 0, 0, // Skip to: 540
/* 397 */     MCD::OPC_ExtractField, 3, 2,  // Inst{4-3} ...
/* 400 */     MCD::OPC_FilterValue, 0, 99, 0, 0, // Skip to: 504
/* 405 */     MCD::OPC_ExtractField, 10, 4,  // Inst{13-10} ...
/* 408 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 420
/* 413 */     MCD::OPC_TryDecode, 207, 5, 8, 244, 0, 0, // Opcode: REL_mLockId_imm, skip to: 664
/* 420 */     MCD::OPC_FilterValue, 1, 7, 0, 0, // Skip to: 432
/* 425 */     MCD::OPC_TryDecode, 253, 3, 8, 232, 0, 0, // Opcode: ACQ_mLockId_imm, skip to: 664
/* 432 */     MCD::OPC_FilterValue, 2, 7, 0, 0, // Skip to: 444
/* 437 */     MCD::OPC_TryDecode, 205, 5, 9, 220, 0, 0, // Opcode: REL_COND_mLockId_imm, skip to: 664
/* 444 */     MCD::OPC_FilterValue, 3, 7, 0, 0, // Skip to: 456
/* 449 */     MCD::OPC_TryDecode, 251, 3, 9, 208, 0, 0, // Opcode: ACQ_COND_mLockId_imm, skip to: 664
/* 456 */     MCD::OPC_FilterValue, 8, 7, 0, 0, // Skip to: 468
/* 461 */     MCD::OPC_TryDecode, 208, 5, 10, 196, 0, 0, // Opcode: REL_mLockId_reg, skip to: 664
/* 468 */     MCD::OPC_FilterValue, 9, 7, 0, 0, // Skip to: 480
/* 473 */     MCD::OPC_TryDecode, 254, 3, 10, 184, 0, 0, // Opcode: ACQ_mLockId_reg, skip to: 664
/* 480 */     MCD::OPC_FilterValue, 10, 7, 0, 0, // Skip to: 492
/* 485 */     MCD::OPC_TryDecode, 206, 5, 11, 172, 0, 0, // Opcode: REL_COND_mLockId_reg, skip to: 664
/* 492 */     MCD::OPC_FilterValue, 11, 167, 0, 0, // Skip to: 664
/* 497 */     MCD::OPC_TryDecode, 252, 3, 11, 160, 0, 0, // Opcode: ACQ_COND_mLockId_reg, skip to: 664
/* 504 */     MCD::OPC_FilterValue, 1, 7, 0, 0, // Skip to: 516
/* 509 */     MCD::OPC_TryDecode, 211, 5, 12, 148, 0, 0, // Opcode: SELEQZ, skip to: 664
/* 516 */     MCD::OPC_FilterValue, 2, 7, 0, 0, // Skip to: 528
/* 521 */     MCD::OPC_TryDecode, 135, 4, 13, 136, 0, 0, // Opcode: DIVS, skip to: 664
/* 528 */     MCD::OPC_FilterValue, 3, 131, 0, 0, // Skip to: 664
/* 533 */     MCD::OPC_TryDecode, 212, 5, 12, 124, 0, 0, // Opcode: SELNEZ, skip to: 664
/* 540 */     MCD::OPC_FilterValue, 5, 51, 0, 0, // Skip to: 596
/* 545 */     MCD::OPC_ExtractField, 3, 2,  // Inst{4-3} ...
/* 548 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 560
/* 553 */     MCD::OPC_TryDecode, 255, 3, 6, 104, 0, 0, // Opcode: ADC, skip to: 664
/* 560 */     MCD::OPC_FilterValue, 1, 7, 0, 0, // Skip to: 572
/* 565 */     MCD::OPC_TryDecode, 167, 9, 6, 92, 0, 0, // Opcode: XOR, skip to: 664
/* 572 */     MCD::OPC_FilterValue, 2, 7, 0, 0, // Skip to: 584
/* 577 */     MCD::OPC_TryDecode, 143, 5, 6, 80, 0, 0, // Opcode: LT, skip to: 664
/* 584 */     MCD::OPC_FilterValue, 3, 75, 0, 0, // Skip to: 664
/* 589 */     MCD::OPC_TryDecode, 132, 4, 6, 68, 0, 0, // Opcode: ASHL, skip to: 664
/* 596 */     MCD::OPC_FilterValue, 6, 7, 0, 0, // Skip to: 608
/* 601 */     MCD::OPC_TryDecode, 130, 4, 14, 56, 0, 0, // Opcode: ADD_add_r_ri, skip to: 664
/* 608 */     MCD::OPC_FilterValue, 7, 51, 0, 0, // Skip to: 664
/* 613 */     MCD::OPC_ExtractField, 3, 2,  // Inst{4-3} ...
/* 616 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 628
/* 621 */     MCD::OPC_TryDecode, 210, 5, 6, 36, 0, 0, // Opcode: SBC, skip to: 664
/* 628 */     MCD::OPC_FilterValue, 1, 7, 0, 0, // Skip to: 640
/* 633 */     MCD::OPC_TryDecode, 137, 4, 6, 24, 0, 0, // Opcode: EQ, skip to: 664
/* 640 */     MCD::OPC_FilterValue, 2, 7, 0, 0, // Skip to: 652
/* 645 */     MCD::OPC_TryDecode, 145, 4, 6, 12, 0, 0, // Opcode: GEU, skip to: 664
/* 652 */     MCD::OPC_FilterValue, 3, 7, 0, 0, // Skip to: 664
/* 657 */     MCD::OPC_TryDecode, 179, 5, 6, 0, 0, 0, // Opcode: MUL_mul_r_rr, skip to: 664
/* 664 */     MCD::OPC_Fail,
  0
};

static const uint8_t DecoderTableFormats16[] = {
/* 0 */       MCD::OPC_CheckField, 0, 15, 1, 4, 0, 0, // Skip to: 11
/* 7 */       MCD::OPC_Decode, 156, 4, 15, // Opcode: I16_NOP
/* 11 */      MCD::OPC_Fail,
  0
};

static const uint8_t DecoderTableFormats32[] = {
/* 0 */       MCD::OPC_ExtractField, 0, 5,  // Inst{4-0} ...
/* 3 */       MCD::OPC_FilterValue, 9, 11, 0, 0, // Skip to: 19
/* 8 */       MCD::OPC_CheckField, 5, 1, 0, 85, 0, 0, // Skip to: 100
/* 15 */      MCD::OPC_Decode, 162, 4, 16, // Opcode: I32_VEC
/* 19 */      MCD::OPC_FilterValue, 25, 76, 0, 0, // Skip to: 100
/* 24 */      MCD::OPC_ExtractField, 27, 5,  // Inst{31-27} ...
/* 27 */      MCD::OPC_FilterValue, 0, 11, 0, 0, // Skip to: 43
/* 32 */      MCD::OPC_CheckField, 5, 1, 0, 61, 0, 0, // Skip to: 100
/* 39 */      MCD::OPC_Decode, 158, 4, 17, // Opcode: I32_LDA
/* 43 */      MCD::OPC_FilterValue, 1, 11, 0, 0, // Skip to: 59
/* 48 */      MCD::OPC_CheckField, 5, 1, 0, 45, 0, 0, // Skip to: 100
/* 55 */      MCD::OPC_Decode, 161, 4, 18, // Opcode: I32_ST
/* 59 */      MCD::OPC_FilterValue, 2, 11, 0, 0, // Skip to: 75
/* 64 */      MCD::OPC_CheckField, 5, 2, 0, 29, 0, 0, // Skip to: 100
/* 71 */      MCD::OPC_Decode, 157, 4, 19, // Opcode: I32_ALU
/* 75 */      MCD::OPC_FilterValue, 3, 4, 0, 0, // Skip to: 84
/* 80 */      MCD::OPC_Decode, 160, 4, 20, // Opcode: I32_MV
/* 84 */      MCD::OPC_FilterValue, 7, 11, 0, 0, // Skip to: 100
/* 89 */      MCD::OPC_CheckField, 5, 6, 0, 4, 0, 0, // Skip to: 100
/* 96 */      MCD::OPC_Decode, 159, 4, 21, // Opcode: I32_LDB
/* 100 */     MCD::OPC_Fail,
  0
};

static const uint8_t DecoderTableFormats48[] = {
/* 0 */       MCD::OPC_ExtractField, 0, 5,  // Inst{4-0} ...
/* 3 */       MCD::OPC_FilterValue, 5, 44, 0, 0, // Skip to: 52
/* 8 */       MCD::OPC_ExtractField, 43, 5,  // Inst{47-43} ...
/* 11 */      MCD::OPC_FilterValue, 0, 4, 0, 0, // Skip to: 20
/* 16 */      MCD::OPC_Decode, 168, 4, 22, // Opcode: I48_LDB_MV
/* 20 */      MCD::OPC_FilterValue, 1, 11, 0, 0, // Skip to: 36
/* 25 */      MCD::OPC_CheckField, 5, 1, 0, 97, 0, 0, // Skip to: 129
/* 32 */      MCD::OPC_Decode, 169, 4, 23, // Opcode: I48_LDB_ST
/* 36 */      MCD::OPC_FilterValue, 2, 88, 0, 0, // Skip to: 129
/* 41 */      MCD::OPC_CheckField, 5, 2, 0, 81, 0, 0, // Skip to: 129
/* 48 */      MCD::OPC_Decode, 167, 4, 24, // Opcode: I48_LDB_ALU
/* 52 */      MCD::OPC_FilterValue, 13, 4, 0, 0, // Skip to: 61
/* 57 */      MCD::OPC_Decode, 165, 4, 25, // Opcode: I48_LDA_MV
/* 61 */      MCD::OPC_FilterValue, 21, 21, 0, 0, // Skip to: 87
/* 66 */      MCD::OPC_ExtractField, 5, 1,  // Inst{5} ...
/* 69 */      MCD::OPC_FilterValue, 0, 4, 0, 0, // Skip to: 78
/* 74 */      MCD::OPC_Decode, 170, 4, 26, // Opcode: I48_LNG
/* 78 */      MCD::OPC_FilterValue, 1, 46, 0, 0, // Skip to: 129
/* 83 */      MCD::OPC_Decode, 166, 4, 27, // Opcode: I48_LDA_ST
/* 87 */      MCD::OPC_FilterValue, 29, 37, 0, 0, // Skip to: 129
/* 92 */      MCD::OPC_ExtractField, 5, 2,  // Inst{6-5} ...
/* 95 */      MCD::OPC_FilterValue, 0, 4, 0, 0, // Skip to: 104
/* 100 */     MCD::OPC_Decode, 163, 4, 28, // Opcode: I48_LDA_ALU
/* 104 */     MCD::OPC_FilterValue, 1, 4, 0, 0, // Skip to: 113
/* 109 */     MCD::OPC_Decode, 171, 4, 29, // Opcode: I48_ST_ALU
/* 113 */     MCD::OPC_FilterValue, 3, 11, 0, 0, // Skip to: 129
/* 118 */     MCD::OPC_CheckField, 7, 4, 0, 4, 0, 0, // Skip to: 129
/* 125 */     MCD::OPC_Decode, 164, 4, 30, // Opcode: I48_LDA_LDB
/* 129 */     MCD::OPC_Fail,
  0
};

static const uint8_t DecoderTableFormats64[] = {
/* 0 */       MCD::OPC_ExtractField, 0, 6,  // Inst{5-0} ...
/* 3 */       MCD::OPC_FilterValue, 3, 51, 0, 0, // Skip to: 59
/* 8 */       MCD::OPC_ExtractField, 6, 6,  // Inst{11-6} ...
/* 11 */      MCD::OPC_FilterValue, 0, 11, 0, 0, // Skip to: 27
/* 16 */      MCD::OPC_CheckField, 34, 9, 0, 168, 0, 0, // Skip to: 191
/* 23 */      MCD::OPC_Decode, 181, 4, 31, // Opcode: I64_ST_MV
/* 27 */      MCD::OPC_FilterValue, 1, 11, 0, 0, // Skip to: 43
/* 32 */      MCD::OPC_CheckField, 54, 9, 0, 152, 0, 0, // Skip to: 191
/* 39 */      MCD::OPC_Decode, 179, 4, 32, // Opcode: I64_NOP_LNG
/* 43 */      MCD::OPC_FilterValue, 33, 143, 0, 0, // Skip to: 191
/* 48 */      MCD::OPC_CheckField, 54, 10, 0, 136, 0, 0, // Skip to: 191
/* 55 */      MCD::OPC_Decode, 172, 4, 33, // Opcode: I64_ALU_MV
/* 59 */      MCD::OPC_FilterValue, 19, 21, 0, 0, // Skip to: 85
/* 64 */      MCD::OPC_ExtractField, 6, 1,  // Inst{6} ...
/* 67 */      MCD::OPC_FilterValue, 0, 4, 0, 0, // Skip to: 76
/* 72 */      MCD::OPC_Decode, 174, 4, 34, // Opcode: I64_LDA_LDB_ALU
/* 76 */      MCD::OPC_FilterValue, 1, 110, 0, 0, // Skip to: 191
/* 81 */      MCD::OPC_Decode, 180, 4, 35, // Opcode: I64_ST_LDB_ALU
/* 85 */      MCD::OPC_FilterValue, 35, 92, 0, 0, // Skip to: 182
/* 90 */      MCD::OPC_ExtractField, 32, 2,  // Inst{33-32} ...
/* 93 */      MCD::OPC_FilterValue, 0, 28, 0, 0, // Skip to: 126
/* 98 */      MCD::OPC_ExtractField, 34, 9,  // Inst{42-34} ...
/* 101 */     MCD::OPC_FilterValue, 0, 11, 0, 0, // Skip to: 117
/* 106 */     MCD::OPC_CheckField, 59, 5, 0, 78, 0, 0, // Skip to: 191
/* 113 */     MCD::OPC_Decode, 177, 4, 36, // Opcode: I64_LDB_VEC
/* 117 */     MCD::OPC_FilterValue, 1, 69, 0, 0, // Skip to: 191
/* 122 */     MCD::OPC_Decode, 176, 4, 37, // Opcode: I64_LDA_VEC
/* 126 */     MCD::OPC_FilterValue, 1, 11, 0, 0, // Skip to: 142
/* 131 */     MCD::OPC_CheckField, 54, 10, 0, 53, 0, 0, // Skip to: 191
/* 138 */     MCD::OPC_Decode, 173, 4, 38, // Opcode: I64_ALU_VEC
/* 142 */     MCD::OPC_FilterValue, 2, 44, 0, 0, // Skip to: 191
/* 147 */     MCD::OPC_ExtractField, 34, 3,  // Inst{36-34} ...
/* 150 */     MCD::OPC_FilterValue, 0, 11, 0, 0, // Skip to: 166
/* 155 */     MCD::OPC_CheckField, 37, 6, 0, 29, 0, 0, // Skip to: 191
/* 162 */     MCD::OPC_Decode, 182, 4, 39, // Opcode: I64_ST_VEC
/* 166 */     MCD::OPC_FilterValue, 1, 20, 0, 0, // Skip to: 191
/* 171 */     MCD::OPC_CheckField, 59, 5, 0, 13, 0, 0, // Skip to: 191
/* 178 */     MCD::OPC_Decode, 178, 4, 40, // Opcode: I64_MV_VEC
/* 182 */     MCD::OPC_FilterValue, 51, 4, 0, 0, // Skip to: 191
/* 187 */     MCD::OPC_Decode, 175, 4, 41, // Opcode: I64_LDA_LDB_ST
/* 191 */     MCD::OPC_Fail,
  0
};

static const uint8_t DecoderTableFormats80[] = {
/* 0 */       MCD::OPC_ExtractField, 0, 6,  // Inst{5-0} ...
/* 3 */       MCD::OPC_FilterValue, 11, 85, 0, 0, // Skip to: 93
/* 8 */       MCD::OPC_ExtractField, 32, 5,  // Inst{36-32} ...
/* 11 */      MCD::OPC_FilterValue, 1, 11, 0, 0, // Skip to: 27
/* 16 */      MCD::OPC_CheckField, 37, 6, 0, 107, 1, 0, // Skip to: 386
/* 23 */      MCD::OPC_Decode, 201, 4, 42, // Opcode: I80_ST_LDB_VEC
/* 27 */      MCD::OPC_FilterValue, 2, 4, 0, 0, // Skip to: 36
/* 32 */      MCD::OPC_Decode, 189, 4, 43, // Opcode: I80_LDA_MV_VEC
/* 36 */      MCD::OPC_FilterValue, 4, 4, 0, 0, // Skip to: 45
/* 41 */      MCD::OPC_Decode, 203, 4, 44, // Opcode: I80_ST_MV_VEC
/* 45 */      MCD::OPC_FilterValue, 6, 11, 0, 0, // Skip to: 61
/* 50 */      MCD::OPC_CheckField, 75, 5, 0, 73, 1, 0, // Skip to: 386
/* 57 */      MCD::OPC_Decode, 196, 4, 45, // Opcode: I80_LDB_MV_VEC
/* 61 */      MCD::OPC_FilterValue, 9, 11, 0, 0, // Skip to: 77
/* 66 */      MCD::OPC_CheckField, 37, 1, 0, 57, 1, 0, // Skip to: 386
/* 73 */      MCD::OPC_Decode, 192, 4, 46, // Opcode: I80_LDA_ST_VEC
/* 77 */      MCD::OPC_FilterValue, 29, 48, 1, 0, // Skip to: 386
/* 82 */      MCD::OPC_CheckField, 37, 6, 0, 41, 1, 0, // Skip to: 386
/* 89 */      MCD::OPC_Decode, 187, 4, 47, // Opcode: I80_LDA_LDB_VEC
/* 93 */      MCD::OPC_FilterValue, 27, 35, 0, 0, // Skip to: 133
/* 98 */      MCD::OPC_ExtractField, 32, 1,  // Inst{32} ...
/* 101 */     MCD::OPC_FilterValue, 0, 11, 0, 0, // Skip to: 117
/* 106 */     MCD::OPC_CheckField, 75, 5, 0, 17, 1, 0, // Skip to: 386
/* 113 */     MCD::OPC_Decode, 197, 4, 48, // Opcode: I80_LNG_VEC
/* 117 */     MCD::OPC_FilterValue, 1, 8, 1, 0, // Skip to: 386
/* 122 */     MCD::OPC_CheckField, 75, 5, 0, 1, 1, 0, // Skip to: 386
/* 129 */     MCD::OPC_Decode, 183, 4, 49, // Opcode: I80_ALU_MV_VEC
/* 133 */     MCD::OPC_FilterValue, 43, 58, 0, 0, // Skip to: 196
/* 138 */     MCD::OPC_ExtractField, 32, 2,  // Inst{33-32} ...
/* 141 */     MCD::OPC_FilterValue, 0, 11, 0, 0, // Skip to: 157
/* 146 */     MCD::OPC_CheckField, 54, 5, 0, 233, 0, 0, // Skip to: 386
/* 153 */     MCD::OPC_Decode, 185, 4, 50, // Opcode: I80_LDA_ALU_VEC
/* 157 */     MCD::OPC_FilterValue, 1, 11, 0, 0, // Skip to: 173
/* 162 */     MCD::OPC_CheckField, 54, 5, 0, 217, 0, 0, // Skip to: 386
/* 169 */     MCD::OPC_Decode, 199, 4, 51, // Opcode: I80_ST_ALU_VEC
/* 173 */     MCD::OPC_FilterValue, 2, 208, 0, 0, // Skip to: 386
/* 178 */     MCD::OPC_CheckField, 75, 5, 0, 201, 0, 0, // Skip to: 386
/* 185 */     MCD::OPC_CheckField, 54, 5, 0, 194, 0, 0, // Skip to: 386
/* 192 */     MCD::OPC_Decode, 194, 4, 52, // Opcode: I80_LDB_ALU_VEC
/* 196 */     MCD::OPC_FilterValue, 59, 185, 0, 0, // Skip to: 386
/* 201 */     MCD::OPC_ExtractField, 6, 1,  // Inst{6} ...
/* 204 */     MCD::OPC_FilterValue, 0, 161, 0, 0, // Skip to: 370
/* 209 */     MCD::OPC_ExtractField, 7, 5,  // Inst{11-7} ...
/* 212 */     MCD::OPC_FilterValue, 0, 18, 0, 0, // Skip to: 235
/* 217 */     MCD::OPC_CheckField, 75, 5, 0, 162, 0, 0, // Skip to: 386
/* 224 */     MCD::OPC_CheckField, 54, 5, 0, 155, 0, 0, // Skip to: 386
/* 231 */     MCD::OPC_Decode, 195, 4, 53, // Opcode: I80_LDB_LNG
/* 235 */     MCD::OPC_FilterValue, 1, 11, 0, 0, // Skip to: 251
/* 240 */     MCD::OPC_CheckField, 54, 5, 0, 139, 0, 0, // Skip to: 386
/* 247 */     MCD::OPC_Decode, 188, 4, 54, // Opcode: I80_LDA_LNG
/* 251 */     MCD::OPC_FilterValue, 2, 11, 0, 0, // Skip to: 267
/* 256 */     MCD::OPC_CheckField, 54, 5, 0, 123, 0, 0, // Skip to: 386
/* 263 */     MCD::OPC_Decode, 202, 4, 55, // Opcode: I80_ST_LNG
/* 267 */     MCD::OPC_FilterValue, 5, 11, 0, 0, // Skip to: 283
/* 272 */     MCD::OPC_CheckField, 34, 4, 0, 107, 0, 0, // Skip to: 386
/* 279 */     MCD::OPC_Decode, 191, 4, 56, // Opcode: I80_LDA_ST_MV
/* 283 */     MCD::OPC_FilterValue, 13, 11, 0, 0, // Skip to: 299
/* 288 */     MCD::OPC_CheckField, 34, 9, 0, 91, 0, 0, // Skip to: 386
/* 295 */     MCD::OPC_Decode, 200, 4, 57, // Opcode: I80_ST_LDB_MV
/* 299 */     MCD::OPC_FilterValue, 16, 18, 0, 0, // Skip to: 322
/* 304 */     MCD::OPC_CheckField, 75, 5, 0, 75, 0, 0, // Skip to: 386
/* 311 */     MCD::OPC_CheckField, 54, 5, 0, 68, 0, 0, // Skip to: 386
/* 318 */     MCD::OPC_Decode, 193, 4, 58, // Opcode: I80_LDB_ALU_MV
/* 322 */     MCD::OPC_FilterValue, 17, 11, 0, 0, // Skip to: 338
/* 327 */     MCD::OPC_CheckField, 54, 5, 0, 52, 0, 0, // Skip to: 386
/* 334 */     MCD::OPC_Decode, 184, 4, 59, // Opcode: I80_LDA_ALU_MV
/* 338 */     MCD::OPC_FilterValue, 18, 11, 0, 0, // Skip to: 354
/* 343 */     MCD::OPC_CheckField, 54, 5, 0, 36, 0, 0, // Skip to: 386
/* 350 */     MCD::OPC_Decode, 198, 4, 60, // Opcode: I80_ST_ALU_MV
/* 354 */     MCD::OPC_FilterValue, 29, 27, 0, 0, // Skip to: 386
/* 359 */     MCD::OPC_CheckField, 34, 9, 0, 20, 0, 0, // Skip to: 386
/* 366 */     MCD::OPC_Decode, 186, 4, 61, // Opcode: I80_LDA_LDB_MV
/* 370 */     MCD::OPC_FilterValue, 1, 11, 0, 0, // Skip to: 386
/* 375 */     MCD::OPC_CheckField, 27, 11, 0, 4, 0, 0, // Skip to: 386
/* 382 */     MCD::OPC_Decode, 190, 4, 62, // Opcode: I80_LDA_ST_ALU
/* 386 */     MCD::OPC_Fail,
  0
};

static const uint8_t DecoderTableFormats96[] = {
/* 0 */       MCD::OPC_ExtractField, 0, 6,  // Inst{5-0} ...
/* 3 */       MCD::OPC_FilterValue, 7, 21, 0, 0, // Skip to: 29
/* 8 */       MCD::OPC_ExtractField, 32, 1,  // Inst{32} ...
/* 11 */      MCD::OPC_FilterValue, 0, 4, 0, 0, // Skip to: 20
/* 16 */      MCD::OPC_Decode, 212, 4, 63, // Opcode: I96_LDA_LNG_VEC
/* 20 */      MCD::OPC_FilterValue, 1, 26, 1, 0, // Skip to: 307
/* 25 */      MCD::OPC_Decode, 204, 4, 64, // Opcode: I96_LDA_ALU_MV_VEC
/* 29 */      MCD::OPC_FilterValue, 23, 21, 0, 0, // Skip to: 55
/* 34 */      MCD::OPC_ExtractField, 32, 1,  // Inst{32} ...
/* 37 */      MCD::OPC_FilterValue, 0, 4, 0, 0, // Skip to: 46
/* 42 */      MCD::OPC_Decode, 223, 4, 65, // Opcode: I96_ST_LNG_VEC
/* 46 */      MCD::OPC_FilterValue, 1, 0, 1, 0, // Skip to: 307
/* 51 */      MCD::OPC_Decode, 218, 4, 66, // Opcode: I96_ST_ALU_MV_VEC
/* 55 */      MCD::OPC_FilterValue, 39, 118, 0, 0, // Skip to: 178
/* 60 */      MCD::OPC_ExtractField, 32, 2,  // Inst{33-32} ...
/* 63 */      MCD::OPC_FilterValue, 0, 4, 0, 0, // Skip to: 72
/* 68 */      MCD::OPC_Decode, 214, 4, 67, // Opcode: I96_LDA_ST_ALU_VEC
/* 72 */      MCD::OPC_FilterValue, 1, 11, 0, 0, // Skip to: 88
/* 77 */      MCD::OPC_CheckField, 54, 5, 0, 223, 0, 0, // Skip to: 307
/* 84 */      MCD::OPC_Decode, 220, 4, 68, // Opcode: I96_ST_LDB_ALU_VEC
/* 88 */      MCD::OPC_FilterValue, 2, 69, 0, 0, // Skip to: 162
/* 93 */      MCD::OPC_ExtractField, 34, 3,  // Inst{36-34} ...
/* 96 */      MCD::OPC_FilterValue, 0, 4, 0, 0, // Skip to: 105
/* 101 */     MCD::OPC_Decode, 209, 4, 69, // Opcode: I96_LDA_LDB_MV_VEC
/* 105 */     MCD::OPC_FilterValue, 1, 4, 0, 0, // Skip to: 114
/* 110 */     MCD::OPC_Decode, 222, 4, 70, // Opcode: I96_ST_LDB_MV_VEC
/* 114 */     MCD::OPC_FilterValue, 2, 11, 0, 0, // Skip to: 130
/* 119 */     MCD::OPC_CheckField, 79, 1, 0, 181, 0, 0, // Skip to: 307
/* 126 */     MCD::OPC_Decode, 217, 4, 71, // Opcode: I96_LDB_LNG_VEC
/* 130 */     MCD::OPC_FilterValue, 3, 11, 0, 0, // Skip to: 146
/* 135 */     MCD::OPC_CheckField, 37, 1, 0, 165, 0, 0, // Skip to: 307
/* 142 */     MCD::OPC_Decode, 211, 4, 72, // Opcode: I96_LDA_LDB_ST_VEC
/* 146 */     MCD::OPC_FilterValue, 6, 156, 0, 0, // Skip to: 307
/* 151 */     MCD::OPC_CheckField, 79, 1, 0, 149, 0, 0, // Skip to: 307
/* 158 */     MCD::OPC_Decode, 216, 4, 73, // Opcode: I96_LDB_ALU_MV_VEC
/* 162 */     MCD::OPC_FilterValue, 3, 140, 0, 0, // Skip to: 307
/* 167 */     MCD::OPC_CheckField, 54, 5, 0, 133, 0, 0, // Skip to: 307
/* 174 */     MCD::OPC_Decode, 207, 4, 74, // Opcode: I96_LDA_LDB_ALU_VEC
/* 178 */     MCD::OPC_FilterValue, 55, 124, 0, 0, // Skip to: 307
/* 183 */     MCD::OPC_ExtractField, 6, 6,  // Inst{11-6} ...
/* 186 */     MCD::OPC_FilterValue, 0, 11, 0, 0, // Skip to: 202
/* 191 */     MCD::OPC_CheckField, 54, 5, 0, 109, 0, 0, // Skip to: 307
/* 198 */     MCD::OPC_Decode, 221, 4, 75, // Opcode: I96_ST_LDB_LNG
/* 202 */     MCD::OPC_FilterValue, 1, 4, 0, 0, // Skip to: 211
/* 207 */     MCD::OPC_Decode, 215, 4, 76, // Opcode: I96_LDA_ST_LNG
/* 211 */     MCD::OPC_FilterValue, 2, 11, 0, 0, // Skip to: 227
/* 216 */     MCD::OPC_CheckField, 54, 5, 0, 84, 0, 0, // Skip to: 307
/* 223 */     MCD::OPC_Decode, 208, 4, 77, // Opcode: I96_LDA_LDB_LNG
/* 227 */     MCD::OPC_FilterValue, 3, 11, 0, 0, // Skip to: 243
/* 232 */     MCD::OPC_CheckField, 34, 4, 0, 68, 0, 0, // Skip to: 307
/* 239 */     MCD::OPC_Decode, 210, 4, 78, // Opcode: I96_LDA_LDB_ST_MV
/* 243 */     MCD::OPC_FilterValue, 4, 18, 0, 0, // Skip to: 266
/* 248 */     MCD::OPC_CheckField, 54, 5, 0, 52, 0, 0, // Skip to: 307
/* 255 */     MCD::OPC_CheckField, 12, 1, 0, 45, 0, 0, // Skip to: 307
/* 262 */     MCD::OPC_Decode, 206, 4, 79, // Opcode: I96_LDA_LDB_ALU_ST
/* 266 */     MCD::OPC_FilterValue, 32, 11, 0, 0, // Skip to: 282
/* 271 */     MCD::OPC_CheckField, 54, 5, 0, 29, 0, 0, // Skip to: 307
/* 278 */     MCD::OPC_Decode, 219, 4, 80, // Opcode: I96_ST_LDB_ALU_MV
/* 282 */     MCD::OPC_FilterValue, 33, 4, 0, 0, // Skip to: 291
/* 287 */     MCD::OPC_Decode, 213, 4, 81, // Opcode: I96_LDA_ST_ALU_MV
/* 291 */     MCD::OPC_FilterValue, 34, 11, 0, 0, // Skip to: 307
/* 296 */     MCD::OPC_CheckField, 54, 5, 0, 4, 0, 0, // Skip to: 307
/* 303 */     MCD::OPC_Decode, 205, 4, 82, // Opcode: I96_LDA_LDB_ALU_MV
/* 307 */     MCD::OPC_Fail,
  0
};

static const uint8_t DecoderTableFormats112[] = {
/* 0 */       MCD::OPC_ExtractField, 0, 6,  // Inst{5-0} ...
/* 3 */       MCD::OPC_FilterValue, 15, 21, 0, 0, // Skip to: 29
/* 8 */       MCD::OPC_ExtractField, 32, 1,  // Inst{32} ...
/* 11 */      MCD::OPC_FilterValue, 0, 4, 0, 0, // Skip to: 20
/* 16 */      MCD::OPC_Decode, 150, 4, 83, // Opcode: I112_LDA_LDB_LNG_VEC
/* 20 */      MCD::OPC_FilterValue, 1, 110, 0, 0, // Skip to: 135
/* 25 */      MCD::OPC_Decode, 147, 4, 84, // Opcode: I112_LDA_LDB_ALU_MV_VEC
/* 29 */      MCD::OPC_FilterValue, 31, 21, 0, 0, // Skip to: 55
/* 34 */      MCD::OPC_ExtractField, 32, 1,  // Inst{32} ...
/* 37 */      MCD::OPC_FilterValue, 0, 4, 0, 0, // Skip to: 46
/* 42 */      MCD::OPC_Decode, 153, 4, 85, // Opcode: I112_ST_LDB_LNG_VEC
/* 46 */      MCD::OPC_FilterValue, 1, 84, 0, 0, // Skip to: 135
/* 51 */      MCD::OPC_Decode, 152, 4, 86, // Opcode: I112_ST_LDB_ALU_MV_VEC
/* 55 */      MCD::OPC_FilterValue, 47, 35, 0, 0, // Skip to: 95
/* 60 */      MCD::OPC_ExtractField, 54, 1,  // Inst{54} ...
/* 63 */      MCD::OPC_FilterValue, 0, 11, 0, 0, // Skip to: 79
/* 68 */      MCD::OPC_CheckField, 53, 1, 0, 60, 0, 0, // Skip to: 135
/* 75 */      MCD::OPC_Decode, 148, 4, 87, // Opcode: I112_LDA_LDB_ALU_ST_VEC
/* 79 */      MCD::OPC_FilterValue, 1, 51, 0, 0, // Skip to: 135
/* 84 */      MCD::OPC_CheckField, 55, 15, 0, 44, 0, 0, // Skip to: 135
/* 91 */      MCD::OPC_Decode, 151, 4, 88, // Opcode: I112_LDA_ST_MV_VEC
/* 95 */      MCD::OPC_FilterValue, 63, 35, 0, 0, // Skip to: 135
/* 100 */     MCD::OPC_ExtractField, 32, 1,  // Inst{32} ...
/* 103 */     MCD::OPC_FilterValue, 0, 11, 0, 0, // Skip to: 119
/* 108 */     MCD::OPC_CheckField, 6, 5, 1, 20, 0, 0, // Skip to: 135
/* 115 */     MCD::OPC_Decode, 149, 4, 89, // Opcode: I112_LDA_LDB_LNG_ST
/* 119 */     MCD::OPC_FilterValue, 1, 11, 0, 0, // Skip to: 135
/* 124 */     MCD::OPC_CheckField, 6, 5, 1, 4, 0, 0, // Skip to: 135
/* 131 */     MCD::OPC_Decode, 146, 4, 90, // Opcode: I112_LDA_LDB_ALU_MV_ST
/* 135 */     MCD::OPC_Fail,
  0
};

static const uint8_t DecoderTableFormats128[] = {
/* 0 */       MCD::OPC_ExtractField, 27, 1,  // Inst{27} ...
/* 3 */       MCD::OPC_FilterValue, 0, 11, 0, 0, // Skip to: 19
/* 8 */       MCD::OPC_CheckField, 0, 1, 0, 20, 0, 0, // Skip to: 35
/* 15 */      MCD::OPC_Decode, 155, 4, 91, // Opcode: I128_LDB_LDA_ST_LNG_VEC
/* 19 */      MCD::OPC_FilterValue, 1, 11, 0, 0, // Skip to: 35
/* 24 */      MCD::OPC_CheckField, 0, 1, 0, 4, 0, 0, // Skip to: 35
/* 31 */      MCD::OPC_Decode, 154, 4, 92, // Opcode: I128_LDB_LDA_ST_ALU_MV_VEC
/* 35 */      MCD::OPC_Fail,
  0
};

static const uint8_t DecoderTableLda32[] = {
/* 0 */       MCD::OPC_ExtractField, 0, 1,  // Inst{0} ...
/* 3 */       MCD::OPC_FilterValue, 0, 25, 7, 0, // Skip to: 1825
/* 8 */       MCD::OPC_ExtractField, 1, 1,  // Inst{1} ...
/* 11 */      MCD::OPC_FilterValue, 0, 16, 6, 0, // Skip to: 1568
/* 16 */      MCD::OPC_ExtractField, 8, 2,  // Inst{9-8} ...
/* 19 */      MCD::OPC_FilterValue, 0, 201, 1, 0, // Skip to: 481
/* 24 */      MCD::OPC_ExtractField, 12, 2,  // Inst{13-12} ...
/* 27 */      MCD::OPC_FilterValue, 0, 151, 0, 0, // Skip to: 183
/* 32 */      MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 35 */      MCD::OPC_FilterValue, 0, 104, 0, 0, // Skip to: 144
/* 40 */      MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 43 */      MCD::OPC_FilterValue, 0, 18, 0, 0, // Skip to: 66
/* 48 */      MCD::OPC_CheckField, 14, 7, 0, 28, 10, 0, // Skip to: 2643
/* 55 */      MCD::OPC_CheckField, 3, 5, 0, 21, 10, 0, // Skip to: 2643
/* 62 */      MCD::OPC_Decode, 183, 5, 0, // Opcode: NOPA
/* 66 */      MCD::OPC_FilterValue, 1, 34, 0, 0, // Skip to: 105
/* 71 */      MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 74 */      MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 93
/* 79 */      MCD::OPC_CheckField, 17, 1, 0, 253, 9, 0, // Skip to: 2643
/* 86 */      MCD::OPC_TryDecode, 218, 5, 93, 246, 9, 0, // Opcode: ST_3D_S8, skip to: 2643
/* 93 */      MCD::OPC_FilterValue, 1, 241, 9, 0, // Skip to: 2643
/* 98 */      MCD::OPC_TryDecode, 225, 5, 94, 234, 9, 0, // Opcode: ST_S8_ag_idx, skip to: 2643
/* 105 */     MCD::OPC_FilterValue, 3, 229, 9, 0, // Skip to: 2643
/* 110 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 113 */     MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 132
/* 118 */     MCD::OPC_CheckField, 17, 1, 0, 214, 9, 0, // Skip to: 2643
/* 125 */     MCD::OPC_TryDecode, 217, 5, 93, 207, 9, 0, // Opcode: ST_3D_S16, skip to: 2643
/* 132 */     MCD::OPC_FilterValue, 1, 202, 9, 0, // Skip to: 2643
/* 137 */     MCD::OPC_TryDecode, 221, 5, 94, 195, 9, 0, // Opcode: ST_S16_ag_idx, skip to: 2643
/* 144 */     MCD::OPC_FilterValue, 1, 190, 9, 0, // Skip to: 2643
/* 149 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 152 */     MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 171
/* 157 */     MCD::OPC_CheckField, 17, 1, 0, 175, 9, 0, // Skip to: 2643
/* 164 */     MCD::OPC_TryDecode, 201, 6, 95, 168, 9, 0, // Opcode: VLDA_3D_UPS_S32_D16, skip to: 2643
/* 171 */     MCD::OPC_FilterValue, 1, 163, 9, 0, // Skip to: 2643
/* 176 */     MCD::OPC_TryDecode, 215, 6, 96, 156, 9, 0, // Opcode: VLDA_UPS_S32_D16_ag_idx, skip to: 2643
/* 183 */     MCD::OPC_FilterValue, 1, 47, 0, 0, // Skip to: 235
/* 188 */     MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 191 */     MCD::OPC_FilterValue, 0, 27, 0, 0, // Skip to: 223
/* 196 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 199 */     MCD::OPC_FilterValue, 1, 7, 0, 0, // Skip to: 211
/* 204 */     MCD::OPC_TryDecode, 228, 5, 97, 128, 9, 0, // Opcode: ST_S8_ag_pstm_nrm_imm, skip to: 2643
/* 211 */     MCD::OPC_FilterValue, 3, 123, 9, 0, // Skip to: 2643
/* 216 */     MCD::OPC_TryDecode, 224, 5, 97, 116, 9, 0, // Opcode: ST_S16_ag_pstm_nrm_imm, skip to: 2643
/* 223 */     MCD::OPC_FilterValue, 1, 111, 9, 0, // Skip to: 2643
/* 228 */     MCD::OPC_TryDecode, 218, 6, 98, 104, 9, 0, // Opcode: VLDA_UPS_S32_D16_ag_pstm_nrm_imm, skip to: 2643
/* 235 */     MCD::OPC_FilterValue, 2, 107, 0, 0, // Skip to: 347
/* 240 */     MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 243 */     MCD::OPC_FilterValue, 0, 67, 0, 0, // Skip to: 315
/* 248 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 251 */     MCD::OPC_FilterValue, 1, 27, 0, 0, // Skip to: 283
/* 256 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 259 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 271
/* 264 */     MCD::OPC_TryDecode, 227, 5, 99, 68, 9, 0, // Opcode: ST_S8_ag_pstm_nrm, skip to: 2643
/* 271 */     MCD::OPC_FilterValue, 1, 63, 9, 0, // Skip to: 2643
/* 276 */     MCD::OPC_TryDecode, 214, 5, 100, 56, 9, 0, // Opcode: ST_2D_S8, skip to: 2643
/* 283 */     MCD::OPC_FilterValue, 3, 51, 9, 0, // Skip to: 2643
/* 288 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 291 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 303
/* 296 */     MCD::OPC_TryDecode, 223, 5, 99, 36, 9, 0, // Opcode: ST_S16_ag_pstm_nrm, skip to: 2643
/* 303 */     MCD::OPC_FilterValue, 1, 31, 9, 0, // Skip to: 2643
/* 308 */     MCD::OPC_TryDecode, 213, 5, 100, 24, 9, 0, // Opcode: ST_2D_S16, skip to: 2643
/* 315 */     MCD::OPC_FilterValue, 1, 19, 9, 0, // Skip to: 2643
/* 320 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 323 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 335
/* 328 */     MCD::OPC_TryDecode, 217, 6, 101, 4, 9, 0, // Opcode: VLDA_UPS_S32_D16_ag_pstm_nrm, skip to: 2643
/* 335 */     MCD::OPC_FilterValue, 1, 255, 8, 0, // Skip to: 2643
/* 340 */     MCD::OPC_TryDecode, 190, 6, 102, 248, 8, 0, // Opcode: VLDA_2D_UPS_S32_D16, skip to: 2643
/* 347 */     MCD::OPC_FilterValue, 3, 243, 8, 0, // Skip to: 2643
/* 352 */     MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 355 */     MCD::OPC_FilterValue, 0, 41, 0, 0, // Skip to: 401
/* 360 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 363 */     MCD::OPC_FilterValue, 1, 14, 0, 0, // Skip to: 382
/* 368 */     MCD::OPC_CheckField, 14, 1, 0, 220, 8, 0, // Skip to: 2643
/* 375 */     MCD::OPC_TryDecode, 226, 5, 103, 213, 8, 0, // Opcode: ST_S8_ag_idx_imm, skip to: 2643
/* 382 */     MCD::OPC_FilterValue, 3, 208, 8, 0, // Skip to: 2643
/* 387 */     MCD::OPC_CheckField, 14, 1, 0, 201, 8, 0, // Skip to: 2643
/* 394 */     MCD::OPC_TryDecode, 222, 5, 103, 194, 8, 0, // Opcode: ST_S16_ag_idx_imm, skip to: 2643
/* 401 */     MCD::OPC_FilterValue, 1, 189, 8, 0, // Skip to: 2643
/* 406 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 409 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 421
/* 414 */     MCD::OPC_TryDecode, 216, 6, 104, 174, 8, 0, // Opcode: VLDA_UPS_S32_D16_ag_idx_imm, skip to: 2643
/* 421 */     MCD::OPC_FilterValue, 1, 169, 8, 0, // Skip to: 2643
/* 426 */     MCD::OPC_ExtractField, 3, 1,  // Inst{3} ...
/* 429 */     MCD::OPC_FilterValue, 0, 21, 0, 0, // Skip to: 455
/* 434 */     MCD::OPC_CheckField, 15, 6, 0, 154, 8, 0, // Skip to: 2643
/* 441 */     MCD::OPC_CheckField, 10, 2, 0, 147, 8, 0, // Skip to: 2643
/* 448 */     MCD::OPC_TryDecode, 212, 7, 105, 140, 8, 0, // Opcode: VMOV_LO, skip to: 2643
/* 455 */     MCD::OPC_FilterValue, 1, 135, 8, 0, // Skip to: 2643
/* 460 */     MCD::OPC_CheckField, 15, 6, 0, 128, 8, 0, // Skip to: 2643
/* 467 */     MCD::OPC_CheckField, 10, 2, 0, 121, 8, 0, // Skip to: 2643
/* 474 */     MCD::OPC_TryDecode, 211, 7, 105, 114, 8, 0, // Opcode: VMOV_HI, skip to: 2643
/* 481 */     MCD::OPC_FilterValue, 1, 117, 1, 0, // Skip to: 859
/* 486 */     MCD::OPC_ExtractField, 12, 2,  // Inst{13-12} ...
/* 489 */     MCD::OPC_FilterValue, 0, 128, 0, 0, // Skip to: 622
/* 494 */     MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 497 */     MCD::OPC_FilterValue, 0, 81, 0, 0, // Skip to: 583
/* 502 */     MCD::OPC_ExtractField, 3, 1,  // Inst{3} ...
/* 505 */     MCD::OPC_FilterValue, 0, 34, 0, 0, // Skip to: 544
/* 510 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 513 */     MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 532
/* 518 */     MCD::OPC_CheckField, 17, 1, 0, 70, 8, 0, // Skip to: 2643
/* 525 */     MCD::OPC_TryDecode, 202, 6, 106, 63, 8, 0, // Opcode: VLDA_3D_UPS_S32_D8, skip to: 2643
/* 532 */     MCD::OPC_FilterValue, 1, 58, 8, 0, // Skip to: 2643
/* 537 */     MCD::OPC_TryDecode, 219, 6, 107, 51, 8, 0, // Opcode: VLDA_UPS_S32_D8_ag_idx, skip to: 2643
/* 544 */     MCD::OPC_FilterValue, 1, 46, 8, 0, // Skip to: 2643
/* 549 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 552 */     MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 571
/* 557 */     MCD::OPC_CheckField, 17, 1, 0, 31, 8, 0, // Skip to: 2643
/* 564 */     MCD::OPC_TryDecode, 204, 6, 106, 24, 8, 0, // Opcode: VLDA_3D_UPS_S32_S8, skip to: 2643
/* 571 */     MCD::OPC_FilterValue, 1, 19, 8, 0, // Skip to: 2643
/* 576 */     MCD::OPC_TryDecode, 227, 6, 107, 12, 8, 0, // Opcode: VLDA_UPS_S32_S8_ag_idx, skip to: 2643
/* 583 */     MCD::OPC_FilterValue, 1, 7, 8, 0, // Skip to: 2643
/* 588 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 591 */     MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 610
/* 596 */     MCD::OPC_CheckField, 17, 1, 0, 248, 7, 0, // Skip to: 2643
/* 603 */     MCD::OPC_TryDecode, 203, 6, 95, 241, 7, 0, // Opcode: VLDA_3D_UPS_S32_S16, skip to: 2643
/* 610 */     MCD::OPC_FilterValue, 1, 236, 7, 0, // Skip to: 2643
/* 615 */     MCD::OPC_TryDecode, 223, 6, 96, 229, 7, 0, // Opcode: VLDA_UPS_S32_S16_ag_idx, skip to: 2643
/* 622 */     MCD::OPC_FilterValue, 1, 47, 0, 0, // Skip to: 674
/* 627 */     MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 630 */     MCD::OPC_FilterValue, 0, 27, 0, 0, // Skip to: 662
/* 635 */     MCD::OPC_ExtractField, 3, 1,  // Inst{3} ...
/* 638 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 650
/* 643 */     MCD::OPC_TryDecode, 222, 6, 108, 201, 7, 0, // Opcode: VLDA_UPS_S32_D8_ag_pstm_nrm_imm, skip to: 2643
/* 650 */     MCD::OPC_FilterValue, 1, 196, 7, 0, // Skip to: 2643
/* 655 */     MCD::OPC_TryDecode, 230, 6, 108, 189, 7, 0, // Opcode: VLDA_UPS_S32_S8_ag_pstm_nrm_imm, skip to: 2643
/* 662 */     MCD::OPC_FilterValue, 1, 184, 7, 0, // Skip to: 2643
/* 667 */     MCD::OPC_TryDecode, 226, 6, 98, 177, 7, 0, // Opcode: VLDA_UPS_S32_S16_ag_pstm_nrm_imm, skip to: 2643
/* 674 */     MCD::OPC_FilterValue, 2, 107, 0, 0, // Skip to: 786
/* 679 */     MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 682 */     MCD::OPC_FilterValue, 0, 67, 0, 0, // Skip to: 754
/* 687 */     MCD::OPC_ExtractField, 3, 1,  // Inst{3} ...
/* 690 */     MCD::OPC_FilterValue, 0, 27, 0, 0, // Skip to: 722
/* 695 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 698 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 710
/* 703 */     MCD::OPC_TryDecode, 221, 6, 109, 141, 7, 0, // Opcode: VLDA_UPS_S32_D8_ag_pstm_nrm, skip to: 2643
/* 710 */     MCD::OPC_FilterValue, 1, 136, 7, 0, // Skip to: 2643
/* 715 */     MCD::OPC_TryDecode, 191, 6, 110, 129, 7, 0, // Opcode: VLDA_2D_UPS_S32_D8, skip to: 2643
/* 722 */     MCD::OPC_FilterValue, 1, 124, 7, 0, // Skip to: 2643
/* 727 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 730 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 742
/* 735 */     MCD::OPC_TryDecode, 229, 6, 109, 109, 7, 0, // Opcode: VLDA_UPS_S32_S8_ag_pstm_nrm, skip to: 2643
/* 742 */     MCD::OPC_FilterValue, 1, 104, 7, 0, // Skip to: 2643
/* 747 */     MCD::OPC_TryDecode, 193, 6, 110, 97, 7, 0, // Opcode: VLDA_2D_UPS_S32_S8, skip to: 2643
/* 754 */     MCD::OPC_FilterValue, 1, 92, 7, 0, // Skip to: 2643
/* 759 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 762 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 774
/* 767 */     MCD::OPC_TryDecode, 225, 6, 101, 77, 7, 0, // Opcode: VLDA_UPS_S32_S16_ag_pstm_nrm, skip to: 2643
/* 774 */     MCD::OPC_FilterValue, 1, 72, 7, 0, // Skip to: 2643
/* 779 */     MCD::OPC_TryDecode, 192, 6, 102, 65, 7, 0, // Opcode: VLDA_2D_UPS_S32_S16, skip to: 2643
/* 786 */     MCD::OPC_FilterValue, 3, 60, 7, 0, // Skip to: 2643
/* 791 */     MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 794 */     MCD::OPC_FilterValue, 0, 41, 0, 0, // Skip to: 840
/* 799 */     MCD::OPC_ExtractField, 3, 1,  // Inst{3} ...
/* 802 */     MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 821
/* 807 */     MCD::OPC_CheckField, 14, 1, 0, 37, 7, 0, // Skip to: 2643
/* 814 */     MCD::OPC_TryDecode, 220, 6, 111, 30, 7, 0, // Opcode: VLDA_UPS_S32_D8_ag_idx_imm, skip to: 2643
/* 821 */     MCD::OPC_FilterValue, 1, 25, 7, 0, // Skip to: 2643
/* 826 */     MCD::OPC_CheckField, 14, 1, 0, 18, 7, 0, // Skip to: 2643
/* 833 */     MCD::OPC_TryDecode, 228, 6, 111, 11, 7, 0, // Opcode: VLDA_UPS_S32_S8_ag_idx_imm, skip to: 2643
/* 840 */     MCD::OPC_FilterValue, 1, 6, 7, 0, // Skip to: 2643
/* 845 */     MCD::OPC_CheckField, 14, 1, 0, 255, 6, 0, // Skip to: 2643
/* 852 */     MCD::OPC_TryDecode, 224, 6, 104, 248, 6, 0, // Opcode: VLDA_UPS_S32_S16_ag_idx_imm, skip to: 2643
/* 859 */     MCD::OPC_FilterValue, 2, 70, 1, 0, // Skip to: 1190
/* 864 */     MCD::OPC_ExtractField, 12, 2,  // Inst{13-12} ...
/* 867 */     MCD::OPC_FilterValue, 0, 140, 0, 0, // Skip to: 1012
/* 872 */     MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 875 */     MCD::OPC_FilterValue, 0, 93, 0, 0, // Skip to: 973
/* 880 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 883 */     MCD::OPC_FilterValue, 0, 34, 0, 0, // Skip to: 922
/* 888 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 891 */     MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 910
/* 896 */     MCD::OPC_CheckField, 17, 1, 0, 204, 6, 0, // Skip to: 2643
/* 903 */     MCD::OPC_TryDecode, 200, 6, 112, 197, 6, 0, // Opcode: VLDA_3D_CONV_FP32_BF16, skip to: 2643
/* 910 */     MCD::OPC_FilterValue, 1, 192, 6, 0, // Skip to: 2643
/* 915 */     MCD::OPC_TryDecode, 211, 6, 113, 185, 6, 0, // Opcode: VLDA_CONV_FP32_BF16_ag_idx, skip to: 2643
/* 922 */     MCD::OPC_FilterValue, 1, 27, 0, 0, // Skip to: 954
/* 927 */     MCD::OPC_ExtractField, 14, 7,  // Inst{20-14} ...
/* 930 */     MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 942
/* 935 */     MCD::OPC_TryDecode, 178, 5, 114, 165, 6, 0, // Opcode: MOV_mv_ss2scl, skip to: 2643
/* 942 */     MCD::OPC_FilterValue, 64, 160, 6, 0, // Skip to: 2643
/* 947 */     MCD::OPC_TryDecode, 165, 5, 114, 153, 6, 0, // Opcode: MOV_NB_mv_ss2scl, skip to: 2643
/* 954 */     MCD::OPC_FilterValue, 3, 148, 6, 0, // Skip to: 2643
/* 959 */     MCD::OPC_CheckField, 14, 4, 0, 141, 6, 0, // Skip to: 2643
/* 966 */     MCD::OPC_TryDecode, 251, 4, 115, 134, 6, 0, // Opcode: LDA_TM, skip to: 2643
/* 973 */     MCD::OPC_FilterValue, 1, 129, 6, 0, // Skip to: 2643
/* 978 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 981 */     MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 1000
/* 986 */     MCD::OPC_CheckField, 17, 1, 0, 114, 6, 0, // Skip to: 2643
/* 993 */     MCD::OPC_TryDecode, 206, 6, 95, 107, 6, 0, // Opcode: VLDA_3D_UPS_S64_D32, skip to: 2643
/* 1000 */    MCD::OPC_FilterValue, 1, 102, 6, 0, // Skip to: 2643
/* 1005 */    MCD::OPC_TryDecode, 235, 6, 96, 95, 6, 0, // Opcode: VLDA_UPS_S64_D32_ag_idx, skip to: 2643
/* 1012 */    MCD::OPC_FilterValue, 1, 34, 0, 0, // Skip to: 1051
/* 1017 */    MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 1020 */    MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 1039
/* 1025 */    MCD::OPC_CheckField, 10, 2, 0, 75, 6, 0, // Skip to: 2643
/* 1032 */    MCD::OPC_TryDecode, 214, 6, 116, 68, 6, 0, // Opcode: VLDA_CONV_FP32_BF16_pstm_nrm_imm, skip to: 2643
/* 1039 */    MCD::OPC_FilterValue, 1, 63, 6, 0, // Skip to: 2643
/* 1044 */    MCD::OPC_TryDecode, 238, 6, 98, 56, 6, 0, // Opcode: VLDA_UPS_S64_D32_ag_pstm_nrm_imm, skip to: 2643
/* 1051 */    MCD::OPC_FilterValue, 2, 81, 0, 0, // Skip to: 1137
/* 1056 */    MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 1059 */    MCD::OPC_FilterValue, 0, 41, 0, 0, // Skip to: 1105
/* 1064 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1067 */    MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 1086
/* 1072 */    MCD::OPC_CheckField, 10, 2, 0, 28, 6, 0, // Skip to: 2643
/* 1079 */    MCD::OPC_TryDecode, 213, 6, 117, 21, 6, 0, // Opcode: VLDA_CONV_FP32_BF16_pstm_nrm, skip to: 2643
/* 1086 */    MCD::OPC_FilterValue, 1, 16, 6, 0, // Skip to: 2643
/* 1091 */    MCD::OPC_CheckField, 10, 2, 0, 9, 6, 0, // Skip to: 2643
/* 1098 */    MCD::OPC_TryDecode, 189, 6, 118, 2, 6, 0, // Opcode: VLDA_2D_CONV_FP32_BF16, skip to: 2643
/* 1105 */    MCD::OPC_FilterValue, 1, 253, 5, 0, // Skip to: 2643
/* 1110 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1113 */    MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 1125
/* 1118 */    MCD::OPC_TryDecode, 237, 6, 101, 238, 5, 0, // Opcode: VLDA_UPS_S64_D32_ag_pstm_nrm, skip to: 2643
/* 1125 */    MCD::OPC_FilterValue, 1, 233, 5, 0, // Skip to: 2643
/* 1130 */    MCD::OPC_TryDecode, 195, 6, 102, 226, 5, 0, // Opcode: VLDA_2D_UPS_S64_D32, skip to: 2643
/* 1137 */    MCD::OPC_FilterValue, 3, 221, 5, 0, // Skip to: 2643
/* 1142 */    MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 1145 */    MCD::OPC_FilterValue, 0, 21, 0, 0, // Skip to: 1171
/* 1150 */    MCD::OPC_CheckField, 14, 1, 0, 206, 5, 0, // Skip to: 2643
/* 1157 */    MCD::OPC_CheckField, 10, 2, 0, 199, 5, 0, // Skip to: 2643
/* 1164 */    MCD::OPC_TryDecode, 212, 6, 119, 192, 5, 0, // Opcode: VLDA_CONV_FP32_BF16_ag_idx_imm, skip to: 2643
/* 1171 */    MCD::OPC_FilterValue, 1, 187, 5, 0, // Skip to: 2643
/* 1176 */    MCD::OPC_CheckField, 14, 1, 0, 180, 5, 0, // Skip to: 2643
/* 1183 */    MCD::OPC_TryDecode, 236, 6, 104, 173, 5, 0, // Opcode: VLDA_UPS_S64_D32_ag_idx_imm, skip to: 2643
/* 1190 */    MCD::OPC_FilterValue, 3, 168, 5, 0, // Skip to: 2643
/* 1195 */    MCD::OPC_ExtractField, 12, 2,  // Inst{13-12} ...
/* 1198 */    MCD::OPC_FilterValue, 0, 128, 0, 0, // Skip to: 1331
/* 1203 */    MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 1206 */    MCD::OPC_FilterValue, 0, 81, 0, 0, // Skip to: 1292
/* 1211 */    MCD::OPC_ExtractField, 3, 1,  // Inst{3} ...
/* 1214 */    MCD::OPC_FilterValue, 0, 34, 0, 0, // Skip to: 1253
/* 1219 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1222 */    MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 1241
/* 1227 */    MCD::OPC_CheckField, 17, 1, 0, 129, 5, 0, // Skip to: 2643
/* 1234 */    MCD::OPC_TryDecode, 205, 6, 106, 122, 5, 0, // Opcode: VLDA_3D_UPS_S64_D16, skip to: 2643
/* 1241 */    MCD::OPC_FilterValue, 1, 117, 5, 0, // Skip to: 2643
/* 1246 */    MCD::OPC_TryDecode, 231, 6, 107, 110, 5, 0, // Opcode: VLDA_UPS_S64_D16_ag_idx, skip to: 2643
/* 1253 */    MCD::OPC_FilterValue, 1, 105, 5, 0, // Skip to: 2643
/* 1258 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1261 */    MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 1280
/* 1266 */    MCD::OPC_CheckField, 17, 1, 0, 90, 5, 0, // Skip to: 2643
/* 1273 */    MCD::OPC_TryDecode, 207, 6, 106, 83, 5, 0, // Opcode: VLDA_3D_UPS_S64_S16, skip to: 2643
/* 1280 */    MCD::OPC_FilterValue, 1, 78, 5, 0, // Skip to: 2643
/* 1285 */    MCD::OPC_TryDecode, 239, 6, 107, 71, 5, 0, // Opcode: VLDA_UPS_S64_S16_ag_idx, skip to: 2643
/* 1292 */    MCD::OPC_FilterValue, 1, 66, 5, 0, // Skip to: 2643
/* 1297 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1300 */    MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 1319
/* 1305 */    MCD::OPC_CheckField, 17, 1, 0, 51, 5, 0, // Skip to: 2643
/* 1312 */    MCD::OPC_TryDecode, 208, 6, 95, 44, 5, 0, // Opcode: VLDA_3D_UPS_S64_S32, skip to: 2643
/* 1319 */    MCD::OPC_FilterValue, 1, 39, 5, 0, // Skip to: 2643
/* 1324 */    MCD::OPC_TryDecode, 243, 6, 96, 32, 5, 0, // Opcode: VLDA_UPS_S64_S32_ag_idx, skip to: 2643
/* 1331 */    MCD::OPC_FilterValue, 1, 47, 0, 0, // Skip to: 1383
/* 1336 */    MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 1339 */    MCD::OPC_FilterValue, 0, 27, 0, 0, // Skip to: 1371
/* 1344 */    MCD::OPC_ExtractField, 3, 1,  // Inst{3} ...
/* 1347 */    MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 1359
/* 1352 */    MCD::OPC_TryDecode, 234, 6, 108, 4, 5, 0, // Opcode: VLDA_UPS_S64_D16_ag_pstm_nrm_imm, skip to: 2643
/* 1359 */    MCD::OPC_FilterValue, 1, 255, 4, 0, // Skip to: 2643
/* 1364 */    MCD::OPC_TryDecode, 242, 6, 108, 248, 4, 0, // Opcode: VLDA_UPS_S64_S16_ag_pstm_nrm_imm, skip to: 2643
/* 1371 */    MCD::OPC_FilterValue, 1, 243, 4, 0, // Skip to: 2643
/* 1376 */    MCD::OPC_TryDecode, 246, 6, 98, 236, 4, 0, // Opcode: VLDA_UPS_S64_S32_ag_pstm_nrm_imm, skip to: 2643
/* 1383 */    MCD::OPC_FilterValue, 2, 107, 0, 0, // Skip to: 1495
/* 1388 */    MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 1391 */    MCD::OPC_FilterValue, 0, 67, 0, 0, // Skip to: 1463
/* 1396 */    MCD::OPC_ExtractField, 3, 1,  // Inst{3} ...
/* 1399 */    MCD::OPC_FilterValue, 0, 27, 0, 0, // Skip to: 1431
/* 1404 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1407 */    MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 1419
/* 1412 */    MCD::OPC_TryDecode, 233, 6, 109, 200, 4, 0, // Opcode: VLDA_UPS_S64_D16_ag_pstm_nrm, skip to: 2643
/* 1419 */    MCD::OPC_FilterValue, 1, 195, 4, 0, // Skip to: 2643
/* 1424 */    MCD::OPC_TryDecode, 194, 6, 110, 188, 4, 0, // Opcode: VLDA_2D_UPS_S64_D16, skip to: 2643
/* 1431 */    MCD::OPC_FilterValue, 1, 183, 4, 0, // Skip to: 2643
/* 1436 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1439 */    MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 1451
/* 1444 */    MCD::OPC_TryDecode, 241, 6, 109, 168, 4, 0, // Opcode: VLDA_UPS_S64_S16_ag_pstm_nrm, skip to: 2643
/* 1451 */    MCD::OPC_FilterValue, 1, 163, 4, 0, // Skip to: 2643
/* 1456 */    MCD::OPC_TryDecode, 196, 6, 110, 156, 4, 0, // Opcode: VLDA_2D_UPS_S64_S16, skip to: 2643
/* 1463 */    MCD::OPC_FilterValue, 1, 151, 4, 0, // Skip to: 2643
/* 1468 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1471 */    MCD::OPC_FilterValue, 0, 7, 0, 0, // Skip to: 1483
/* 1476 */    MCD::OPC_TryDecode, 245, 6, 101, 136, 4, 0, // Opcode: VLDA_UPS_S64_S32_ag_pstm_nrm, skip to: 2643
/* 1483 */    MCD::OPC_FilterValue, 1, 131, 4, 0, // Skip to: 2643
/* 1488 */    MCD::OPC_TryDecode, 197, 6, 102, 124, 4, 0, // Opcode: VLDA_2D_UPS_S64_S32, skip to: 2643
/* 1495 */    MCD::OPC_FilterValue, 3, 119, 4, 0, // Skip to: 2643
/* 1500 */    MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 1503 */    MCD::OPC_FilterValue, 0, 41, 0, 0, // Skip to: 1549
/* 1508 */    MCD::OPC_ExtractField, 3, 1,  // Inst{3} ...
/* 1511 */    MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 1530
/* 1516 */    MCD::OPC_CheckField, 14, 1, 0, 96, 4, 0, // Skip to: 2643
/* 1523 */    MCD::OPC_TryDecode, 232, 6, 111, 89, 4, 0, // Opcode: VLDA_UPS_S64_D16_ag_idx_imm, skip to: 2643
/* 1530 */    MCD::OPC_FilterValue, 1, 84, 4, 0, // Skip to: 2643
/* 1535 */    MCD::OPC_CheckField, 14, 1, 0, 77, 4, 0, // Skip to: 2643
/* 1542 */    MCD::OPC_TryDecode, 240, 6, 111, 70, 4, 0, // Opcode: VLDA_UPS_S64_S16_ag_idx_imm, skip to: 2643
/* 1549 */    MCD::OPC_FilterValue, 1, 65, 4, 0, // Skip to: 2643
/* 1554 */    MCD::OPC_CheckField, 14, 1, 0, 58, 4, 0, // Skip to: 2643
/* 1561 */    MCD::OPC_TryDecode, 244, 6, 104, 51, 4, 0, // Opcode: VLDA_UPS_S64_S32_ag_idx_imm, skip to: 2643
/* 1568 */    MCD::OPC_FilterValue, 1, 46, 4, 0, // Skip to: 2643
/* 1573 */    MCD::OPC_ExtractField, 2, 2,  // Inst{3-2} ...
/* 1576 */    MCD::OPC_FilterValue, 1, 67, 0, 0, // Skip to: 1648
/* 1581 */    MCD::OPC_ExtractField, 12, 3,  // Inst{14-12} ...
/* 1584 */    MCD::OPC_FilterValue, 0, 21, 0, 0, // Skip to: 1610
/* 1589 */    MCD::OPC_CheckField, 17, 1, 0, 87, 0, 0, // Skip to: 1683
/* 1596 */    MCD::OPC_CheckField, 6, 2, 3, 80, 0, 0, // Skip to: 1683
/* 1603 */    MCD::OPC_TryDecode, 192, 5, 120, 73, 0, 0, // Opcode: PADDA_3D, skip to: 1683
/* 1610 */    MCD::OPC_FilterValue, 2, 14, 0, 0, // Skip to: 1629
/* 1615 */    MCD::OPC_CheckField, 6, 2, 3, 61, 0, 0, // Skip to: 1683
/* 1622 */    MCD::OPC_TryDecode, 193, 5, 121, 54, 0, 0, // Opcode: PADDA_lda_ptr_inc_idx, skip to: 1683
/* 1629 */    MCD::OPC_FilterValue, 6, 49, 0, 0, // Skip to: 1683
/* 1634 */    MCD::OPC_CheckField, 6, 2, 3, 42, 0, 0, // Skip to: 1683
/* 1641 */    MCD::OPC_TryDecode, 191, 5, 122, 35, 0, 0, // Opcode: PADDA_2D, skip to: 1683
/* 1648 */    MCD::OPC_FilterValue, 2, 11, 0, 0, // Skip to: 1664
/* 1653 */    MCD::OPC_CheckField, 6, 2, 3, 23, 0, 0, // Skip to: 1683
/* 1660 */    MCD::OPC_Decode, 195, 5, 123, // Opcode: PADDA_sp_imm
/* 1664 */    MCD::OPC_FilterValue, 3, 14, 0, 0, // Skip to: 1683
/* 1669 */    MCD::OPC_CheckField, 6, 2, 3, 7, 0, 0, // Skip to: 1683
/* 1676 */    MCD::OPC_TryDecode, 194, 5, 124, 0, 0, 0, // Opcode: PADDA_lda_ptr_inc_idx_imm, skip to: 1683
/* 1683 */    MCD::OPC_ExtractField, 8, 1,  // Inst{8} ...
/* 1686 */    MCD::OPC_FilterValue, 0, 121, 0, 0, // Skip to: 1812
/* 1691 */    MCD::OPC_ExtractField, 9, 2,  // Inst{10-9} ...
/* 1694 */    MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 1713
/* 1699 */    MCD::OPC_CheckField, 11, 10, 0, 169, 3, 0, // Skip to: 2643
/* 1706 */    MCD::OPC_TryDecode, 215, 7, 125, 162, 3, 0, // Opcode: VMOV_mv_scd, skip to: 2643
/* 1713 */    MCD::OPC_FilterValue, 1, 81, 0, 0, // Skip to: 1799
/* 1718 */    MCD::OPC_ExtractField, 11, 1,  // Inst{11} ...
/* 1721 */    MCD::OPC_FilterValue, 0, 60, 0, 0, // Skip to: 1786
/* 1726 */    MCD::OPC_ExtractField, 12, 3,  // Inst{14-12} ...
/* 1729 */    MCD::OPC_FilterValue, 0, 14, 0, 0, // Skip to: 1748
/* 1734 */    MCD::OPC_CheckField, 17, 1, 0, 134, 3, 0, // Skip to: 2643
/* 1741 */    MCD::OPC_TryDecode, 209, 6, 126, 127, 3, 0, // Opcode: VLDA_3D_dmw_lda_am, skip to: 2643
/* 1748 */    MCD::OPC_FilterValue, 2, 7, 0, 0, // Skip to: 1760
/* 1753 */    MCD::OPC_TryDecode, 249, 6, 127, 115, 3, 0, // Opcode: VLDA_dmw_lda_am_ag_pstm_nrm, skip to: 2643
/* 1760 */    MCD::OPC_FilterValue, 4, 8, 0, 0, // Skip to: 1773
/* 1765 */    MCD::OPC_TryDecode, 247, 6, 128, 1, 102, 3, 0, // Opcode: VLDA_dmw_lda_am_ag_idx, skip to: 2643
/* 1773 */    MCD::OPC_FilterValue, 6, 97, 3, 0, // Skip to: 2643
/* 1778 */    MCD::OPC_TryDecode, 198, 6, 129, 1, 89, 3, 0, // Opcode: VLDA_2D_dmw_lda_am, skip to: 2643
/* 1786 */    MCD::OPC_FilterValue, 1, 84, 3, 0, // Skip to: 2643
/* 1791 */    MCD::OPC_TryDecode, 248, 6, 130, 1, 76, 3, 0, // Opcode: VLDA_dmw_lda_am_ag_idx_imm, skip to: 2643
/* 1799 */    MCD::OPC_FilterValue, 3, 71, 3, 0, // Skip to: 2643
/* 1804 */    MCD::OPC_TryDecode, 250, 6, 131, 1, 63, 3, 0, // Opcode: VLDA_dmw_lda_am_ag_pstm_nrm_imm, skip to: 2643
/* 1812 */    MCD::OPC_FilterValue, 1, 58, 3, 0, // Skip to: 2643
/* 1817 */    MCD::OPC_TryDecode, 251, 6, 132, 1, 50, 3, 0, // Opcode: VLDA_dmw_lda_am_ag_spill, skip to: 2643
/* 1825 */    MCD::OPC_FilterValue, 1, 45, 3, 0, // Skip to: 2643
/* 1830 */    MCD::OPC_ExtractField, 8, 1,  // Inst{8} ...
/* 1833 */    MCD::OPC_FilterValue, 0, 250, 2, 0, // Skip to: 2600
/* 1838 */    MCD::OPC_ExtractField, 9, 1,  // Inst{9} ...
/* 1841 */    MCD::OPC_FilterValue, 0, 208, 1, 0, // Skip to: 2310
/* 1846 */    MCD::OPC_ExtractField, 1, 2,  // Inst{2-1} ...
/* 1849 */    MCD::OPC_FilterValue, 3, 192, 1, 0, // Skip to: 2302
/* 1854 */    MCD::OPC_ExtractField, 10, 4,  // Inst{13-10} ...
/* 1857 */    MCD::OPC_FilterValue, 0, 56, 0, 0, // Skip to: 1918
/* 1862 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1865 */    MCD::OPC_FilterValue, 0, 36, 0, 0, // Skip to: 1906
/* 1870 */    MCD::OPC_ExtractField, 17, 1,  // Inst{17} ...
/* 1873 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 1886
/* 1878 */    MCD::OPC_TryDecode, 238, 4, 133, 1, 160, 1, 0, // Opcode: LDA_3D_S8_dmhb_lda, skip to: 2302
/* 1886 */    MCD::OPC_FilterValue, 1, 155, 1, 0, // Skip to: 2302
/* 1891 */    MCD::OPC_CheckField, 15, 2, 0, 148, 1, 0, // Skip to: 2302
/* 1898 */    MCD::OPC_TryDecode, 188, 6, 134, 1, 140, 1, 0, // Opcode: VLDA_128, skip to: 2302
/* 1906 */    MCD::OPC_FilterValue, 1, 135, 1, 0, // Skip to: 2302
/* 1911 */    MCD::OPC_TryDecode, 247, 4, 94, 128, 1, 0, // Opcode: LDA_S8_ag_idx, skip to: 2302
/* 1918 */    MCD::OPC_FilterValue, 1, 35, 0, 0, // Skip to: 1958
/* 1923 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1926 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 1946
/* 1931 */    MCD::OPC_CheckField, 17, 1, 0, 108, 1, 0, // Skip to: 2302
/* 1938 */    MCD::OPC_TryDecode, 240, 4, 133, 1, 100, 1, 0, // Opcode: LDA_3D_U8_dmhb_lda, skip to: 2302
/* 1946 */    MCD::OPC_FilterValue, 1, 95, 1, 0, // Skip to: 2302
/* 1951 */    MCD::OPC_TryDecode, 128, 5, 94, 88, 1, 0, // Opcode: LDA_U8_ag_idx, skip to: 2302
/* 1958 */    MCD::OPC_FilterValue, 2, 35, 0, 0, // Skip to: 1998
/* 1963 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1966 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 1986
/* 1971 */    MCD::OPC_CheckField, 17, 1, 0, 68, 1, 0, // Skip to: 2302
/* 1978 */    MCD::OPC_TryDecode, 237, 4, 133, 1, 60, 1, 0, // Opcode: LDA_3D_S16_dmhb_lda, skip to: 2302
/* 1986 */    MCD::OPC_FilterValue, 1, 55, 1, 0, // Skip to: 2302
/* 1991 */    MCD::OPC_TryDecode, 243, 4, 94, 48, 1, 0, // Opcode: LDA_S16_ag_idx, skip to: 2302
/* 1998 */    MCD::OPC_FilterValue, 3, 35, 0, 0, // Skip to: 2038
/* 2003 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 2006 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 2026
/* 2011 */    MCD::OPC_CheckField, 17, 1, 0, 28, 1, 0, // Skip to: 2302
/* 2018 */    MCD::OPC_TryDecode, 239, 4, 133, 1, 20, 1, 0, // Opcode: LDA_3D_U16_dmhb_lda, skip to: 2302
/* 2026 */    MCD::OPC_FilterValue, 1, 15, 1, 0, // Skip to: 2302
/* 2031 */    MCD::OPC_TryDecode, 252, 4, 94, 8, 1, 0, // Opcode: LDA_U16_ag_idx, skip to: 2302
/* 2038 */    MCD::OPC_FilterValue, 4, 8, 0, 0, // Skip to: 2051
/* 2043 */    MCD::OPC_TryDecode, 250, 4, 135, 1, 251, 0, 0, // Opcode: LDA_S8_ag_pstm_nrm_imm, skip to: 2302
/* 2051 */    MCD::OPC_FilterValue, 5, 8, 0, 0, // Skip to: 2064
/* 2056 */    MCD::OPC_TryDecode, 131, 5, 135, 1, 238, 0, 0, // Opcode: LDA_U8_ag_pstm_nrm_imm, skip to: 2302
/* 2064 */    MCD::OPC_FilterValue, 6, 8, 0, 0, // Skip to: 2077
/* 2069 */    MCD::OPC_TryDecode, 246, 4, 135, 1, 225, 0, 0, // Opcode: LDA_S16_ag_pstm_nrm_imm, skip to: 2302
/* 2077 */    MCD::OPC_FilterValue, 7, 8, 0, 0, // Skip to: 2090
/* 2082 */    MCD::OPC_TryDecode, 255, 4, 135, 1, 212, 0, 0, // Opcode: LDA_U16_ag_pstm_nrm_imm, skip to: 2302
/* 2090 */    MCD::OPC_FilterValue, 8, 29, 0, 0, // Skip to: 2124
/* 2095 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 2098 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 2111
/* 2103 */    MCD::OPC_TryDecode, 249, 4, 136, 1, 191, 0, 0, // Opcode: LDA_S8_ag_pstm_nrm, skip to: 2302
/* 2111 */    MCD::OPC_FilterValue, 1, 186, 0, 0, // Skip to: 2302
/* 2116 */    MCD::OPC_TryDecode, 232, 4, 137, 1, 178, 0, 0, // Opcode: LDA_2D_S8_dmhb_lda, skip to: 2302
/* 2124 */    MCD::OPC_FilterValue, 9, 29, 0, 0, // Skip to: 2158
/* 2129 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 2132 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 2145
/* 2137 */    MCD::OPC_TryDecode, 130, 5, 136, 1, 157, 0, 0, // Opcode: LDA_U8_ag_pstm_nrm, skip to: 2302
/* 2145 */    MCD::OPC_FilterValue, 1, 152, 0, 0, // Skip to: 2302
/* 2150 */    MCD::OPC_TryDecode, 234, 4, 137, 1, 144, 0, 0, // Opcode: LDA_2D_U8_dmhb_lda, skip to: 2302
/* 2158 */    MCD::OPC_FilterValue, 10, 29, 0, 0, // Skip to: 2192
/* 2163 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 2166 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 2179
/* 2171 */    MCD::OPC_TryDecode, 245, 4, 136, 1, 123, 0, 0, // Opcode: LDA_S16_ag_pstm_nrm, skip to: 2302
/* 2179 */    MCD::OPC_FilterValue, 1, 118, 0, 0, // Skip to: 2302
/* 2184 */    MCD::OPC_TryDecode, 231, 4, 137, 1, 110, 0, 0, // Opcode: LDA_2D_S16_dmhb_lda, skip to: 2302
/* 2192 */    MCD::OPC_FilterValue, 11, 29, 0, 0, // Skip to: 2226
/* 2197 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 2200 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 2213
/* 2205 */    MCD::OPC_TryDecode, 254, 4, 136, 1, 89, 0, 0, // Opcode: LDA_U16_ag_pstm_nrm, skip to: 2302
/* 2213 */    MCD::OPC_FilterValue, 1, 84, 0, 0, // Skip to: 2302
/* 2218 */    MCD::OPC_TryDecode, 233, 4, 137, 1, 76, 0, 0, // Opcode: LDA_2D_U16_dmhb_lda, skip to: 2302
/* 2226 */    MCD::OPC_FilterValue, 12, 14, 0, 0, // Skip to: 2245
/* 2231 */    MCD::OPC_CheckField, 14, 1, 0, 64, 0, 0, // Skip to: 2302
/* 2238 */    MCD::OPC_TryDecode, 248, 4, 103, 57, 0, 0, // Opcode: LDA_S8_ag_idx_imm, skip to: 2302
/* 2245 */    MCD::OPC_FilterValue, 13, 14, 0, 0, // Skip to: 2264
/* 2250 */    MCD::OPC_CheckField, 14, 1, 0, 45, 0, 0, // Skip to: 2302
/* 2257 */    MCD::OPC_TryDecode, 129, 5, 103, 38, 0, 0, // Opcode: LDA_U8_ag_idx_imm, skip to: 2302
/* 2264 */    MCD::OPC_FilterValue, 14, 14, 0, 0, // Skip to: 2283
/* 2269 */    MCD::OPC_CheckField, 14, 1, 0, 26, 0, 0, // Skip to: 2302
/* 2276 */    MCD::OPC_TryDecode, 244, 4, 103, 19, 0, 0, // Opcode: LDA_S16_ag_idx_imm, skip to: 2302
/* 2283 */    MCD::OPC_FilterValue, 15, 14, 0, 0, // Skip to: 2302
/* 2288 */    MCD::OPC_CheckField, 14, 1, 0, 7, 0, 0, // Skip to: 2302
/* 2295 */    MCD::OPC_TryDecode, 253, 4, 103, 0, 0, 0, // Opcode: LDA_U16_ag_idx_imm, skip to: 2302
/* 2302 */    MCD::OPC_TryDecode, 145, 5, 138, 1, 77, 1, 0, // Opcode: MOVA_lda_cg, skip to: 2643
/* 2310 */    MCD::OPC_FilterValue, 1, 72, 1, 0, // Skip to: 2643
/* 2315 */    MCD::OPC_ExtractField, 10, 1,  // Inst{10} ...
/* 2318 */    MCD::OPC_FilterValue, 0, 234, 0, 0, // Skip to: 2557
/* 2323 */    MCD::OPC_ExtractField, 11, 1,  // Inst{11} ...
/* 2326 */    MCD::OPC_FilterValue, 0, 183, 0, 0, // Skip to: 2514
/* 2331 */    MCD::OPC_ExtractField, 12, 3,  // Inst{14-12} ...
/* 2334 */    MCD::OPC_FilterValue, 0, 46, 0, 0, // Skip to: 2385
/* 2339 */    MCD::OPC_ExtractField, 17, 1,  // Inst{17} ...
/* 2342 */    MCD::OPC_FilterValue, 0, 40, 1, 0, // Skip to: 2643
/* 2347 */    MCD::OPC_CheckField, 1, 5, 1, 8, 0, 0, // Skip to: 2362
/* 2354 */    MCD::OPC_TryDecode, 242, 4, 139, 1, 0, 0, 0, // Opcode: LDA_3D_dmv_lda_q, skip to: 2362
/* 2362 */    MCD::OPC_CheckField, 1, 2, 3, 8, 0, 0, // Skip to: 2377
/* 2369 */    MCD::OPC_TryDecode, 210, 6, 140, 1, 0, 0, 0, // Opcode: VLDA_3D_dmw_lda_w, skip to: 2377
/* 2377 */    MCD::OPC_TryDecode, 241, 4, 141, 1, 2, 1, 0, // Opcode: LDA_3D_dms_lda, skip to: 2643
/* 2385 */    MCD::OPC_FilterValue, 2, 38, 0, 0, // Skip to: 2428
/* 2390 */    MCD::OPC_CheckField, 1, 5, 1, 8, 0, 0, // Skip to: 2405
/* 2397 */    MCD::OPC_TryDecode, 139, 5, 142, 1, 0, 0, 0, // Opcode: LDA_dmv_lda_q_ag_pstm_nrm, skip to: 2405
/* 2405 */    MCD::OPC_CheckField, 1, 2, 3, 8, 0, 0, // Skip to: 2420
/* 2412 */    MCD::OPC_TryDecode, 254, 6, 143, 1, 0, 0, 0, // Opcode: VLDA_dmw_lda_w_ag_pstm_nrm, skip to: 2420
/* 2420 */    MCD::OPC_TryDecode, 134, 5, 144, 1, 215, 0, 0, // Opcode: LDA_dms_lda_pstm_nrm, skip to: 2643
/* 2428 */    MCD::OPC_FilterValue, 4, 38, 0, 0, // Skip to: 2471
/* 2433 */    MCD::OPC_CheckField, 1, 5, 1, 8, 0, 0, // Skip to: 2448
/* 2440 */    MCD::OPC_TryDecode, 137, 5, 145, 1, 0, 0, 0, // Opcode: LDA_dmv_lda_q_ag_idx, skip to: 2448
/* 2448 */    MCD::OPC_CheckField, 1, 2, 3, 8, 0, 0, // Skip to: 2463
/* 2455 */    MCD::OPC_TryDecode, 252, 6, 146, 1, 0, 0, 0, // Opcode: VLDA_dmw_lda_w_ag_idx, skip to: 2463
/* 2463 */    MCD::OPC_TryDecode, 132, 5, 147, 1, 172, 0, 0, // Opcode: LDA_dms_lda_idx, skip to: 2643
/* 2471 */    MCD::OPC_FilterValue, 6, 167, 0, 0, // Skip to: 2643
/* 2476 */    MCD::OPC_CheckField, 1, 5, 1, 8, 0, 0, // Skip to: 2491
/* 2483 */    MCD::OPC_TryDecode, 236, 4, 148, 1, 0, 0, 0, // Opcode: LDA_2D_dmv_lda_q, skip to: 2491
/* 2491 */    MCD::OPC_CheckField, 1, 2, 3, 8, 0, 0, // Skip to: 2506
/* 2498 */    MCD::OPC_TryDecode, 199, 6, 149, 1, 0, 0, 0, // Opcode: VLDA_2D_dmw_lda_w, skip to: 2506
/* 2506 */    MCD::OPC_TryDecode, 235, 4, 150, 1, 129, 0, 0, // Opcode: LDA_2D_dms_lda, skip to: 2643
/* 2514 */    MCD::OPC_FilterValue, 1, 124, 0, 0, // Skip to: 2643
/* 2519 */    MCD::OPC_CheckField, 1, 5, 1, 8, 0, 0, // Skip to: 2534
/* 2526 */    MCD::OPC_TryDecode, 138, 5, 151, 1, 0, 0, 0, // Opcode: LDA_dmv_lda_q_ag_idx_imm, skip to: 2534
/* 2534 */    MCD::OPC_CheckField, 1, 2, 3, 8, 0, 0, // Skip to: 2549
/* 2541 */    MCD::OPC_TryDecode, 253, 6, 152, 1, 0, 0, 0, // Opcode: VLDA_dmw_lda_w_ag_idx_imm, skip to: 2549
/* 2549 */    MCD::OPC_TryDecode, 133, 5, 153, 1, 86, 0, 0, // Opcode: LDA_dms_lda_idx_imm, skip to: 2643
/* 2557 */    MCD::OPC_FilterValue, 1, 81, 0, 0, // Skip to: 2643
/* 2562 */    MCD::OPC_CheckField, 1, 5, 1, 8, 0, 0, // Skip to: 2577
/* 2569 */    MCD::OPC_TryDecode, 140, 5, 154, 1, 0, 0, 0, // Opcode: LDA_dmv_lda_q_ag_pstm_nrm_imm, skip to: 2577
/* 2577 */    MCD::OPC_CheckField, 1, 2, 3, 8, 0, 0, // Skip to: 2592
/* 2584 */    MCD::OPC_TryDecode, 255, 6, 155, 1, 0, 0, 0, // Opcode: VLDA_dmw_lda_w_ag_pstm_nrm_imm, skip to: 2592
/* 2592 */    MCD::OPC_TryDecode, 135, 5, 156, 1, 43, 0, 0, // Opcode: LDA_dms_lda_pstm_nrm_imm, skip to: 2643
/* 2600 */    MCD::OPC_FilterValue, 1, 38, 0, 0, // Skip to: 2643
/* 2605 */    MCD::OPC_CheckField, 1, 5, 1, 8, 0, 0, // Skip to: 2620
/* 2612 */    MCD::OPC_TryDecode, 141, 5, 157, 1, 0, 0, 0, // Opcode: LDA_dmv_lda_q_ag_spill, skip to: 2620
/* 2620 */    MCD::OPC_CheckField, 1, 2, 3, 8, 0, 0, // Skip to: 2635
/* 2627 */    MCD::OPC_TryDecode, 128, 7, 158, 1, 0, 0, 0, // Opcode: VLDA_dmw_lda_w_ag_spill, skip to: 2635
/* 2635 */    MCD::OPC_TryDecode, 136, 5, 159, 1, 0, 0, 0, // Opcode: LDA_dms_spill, skip to: 2643
/* 2643 */    MCD::OPC_Fail,
  0
};

static const uint8_t DecoderTableLdb32[] = {
/* 0 */       MCD::OPC_ExtractField, 0, 2,  // Inst{1-0} ...
/* 3 */       MCD::OPC_FilterValue, 0, 171, 1, 0, // Skip to: 435
/* 8 */       MCD::OPC_CheckField, 2, 14, 0, 4, 0, 0, // Skip to: 19
/* 15 */      MCD::OPC_Decode, 184, 5, 0, // Opcode: NOPB
/* 19 */      MCD::OPC_ExtractField, 2, 11,  // Inst{12-2} ...
/* 22 */      MCD::OPC_FilterValue, 64, 8, 0, 0, // Skip to: 35
/* 27 */      MCD::OPC_TryDecode, 149, 7, 160, 1, 98, 0, 0, // Opcode: VLDB_COMPR_FILL, skip to: 133
/* 35 */      MCD::OPC_FilterValue, 192, 1, 8, 0, 0, // Skip to: 49
/* 41 */      MCD::OPC_TryDecode, 154, 7, 160, 1, 84, 0, 0, // Opcode: VLDB_SPARSE_FILL_4, skip to: 133
/* 49 */      MCD::OPC_FilterValue, 192, 2, 8, 0, 0, // Skip to: 63
/* 55 */      MCD::OPC_TryDecode, 155, 7, 160, 1, 70, 0, 0, // Opcode: VLDB_SPARSE_FILL_8, skip to: 133
/* 63 */      MCD::OPC_FilterValue, 192, 3, 8, 0, 0, // Skip to: 77
/* 69 */      MCD::OPC_TryDecode, 153, 7, 160, 1, 56, 0, 0, // Opcode: VLDB_SPARSE_FILL_16, skip to: 133
/* 77 */      MCD::OPC_FilterValue, 192, 8, 8, 0, 0, // Skip to: 91
/* 83 */      MCD::OPC_TryDecode, 152, 7, 160, 1, 42, 0, 0, // Opcode: VLDB_COMPR_RESET, skip to: 133
/* 91 */      MCD::OPC_FilterValue, 192, 9, 8, 0, 0, // Skip to: 105
/* 97 */      MCD::OPC_TryDecode, 163, 7, 160, 1, 28, 0, 0, // Opcode: VLDB_SPARSE_RESET_4, skip to: 133
/* 105 */     MCD::OPC_FilterValue, 192, 10, 8, 0, 0, // Skip to: 119
/* 111 */     MCD::OPC_TryDecode, 164, 7, 160, 1, 14, 0, 0, // Opcode: VLDB_SPARSE_RESET_8, skip to: 133
/* 119 */     MCD::OPC_FilterValue, 192, 11, 8, 0, 0, // Skip to: 133
/* 125 */     MCD::OPC_TryDecode, 162, 7, 160, 1, 0, 0, 0, // Opcode: VLDB_SPARSE_RESET_16, skip to: 133
/* 133 */     MCD::OPC_ExtractField, 5, 8,  // Inst{12-5} ...
/* 136 */     MCD::OPC_FilterValue, 88, 8, 0, 0, // Skip to: 149
/* 141 */     MCD::OPC_TryDecode, 160, 7, 161, 1, 68, 0, 0, // Opcode: VLDB_SPARSE_POP_4, skip to: 217
/* 149 */     MCD::OPC_FilterValue, 104, 8, 0, 0, // Skip to: 162
/* 154 */     MCD::OPC_TryDecode, 161, 7, 161, 1, 55, 0, 0, // Opcode: VLDB_SPARSE_POP_8, skip to: 217
/* 162 */     MCD::OPC_FilterValue, 120, 8, 0, 0, // Skip to: 175
/* 167 */     MCD::OPC_TryDecode, 159, 7, 161, 1, 42, 0, 0, // Opcode: VLDB_SPARSE_POP_16, skip to: 217
/* 175 */     MCD::OPC_FilterValue, 216, 1, 8, 0, 0, // Skip to: 189
/* 181 */     MCD::OPC_TryDecode, 157, 7, 161, 1, 28, 0, 0, // Opcode: VLDB_SPARSE_PEEK_4, skip to: 217
/* 189 */     MCD::OPC_FilterValue, 232, 1, 8, 0, 0, // Skip to: 203
/* 195 */     MCD::OPC_TryDecode, 158, 7, 161, 1, 14, 0, 0, // Opcode: VLDB_SPARSE_PEEK_8, skip to: 217
/* 203 */     MCD::OPC_FilterValue, 248, 1, 8, 0, 0, // Skip to: 217
/* 209 */     MCD::OPC_TryDecode, 156, 7, 161, 1, 0, 0, 0, // Opcode: VLDB_SPARSE_PEEK_16, skip to: 217
/* 217 */     MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 220 */     MCD::OPC_FilterValue, 0, 82, 0, 0, // Skip to: 307
/* 225 */     MCD::OPC_ExtractField, 7, 3,  // Inst{9-7} ...
/* 228 */     MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 241
/* 233 */     MCD::OPC_TryDecode, 167, 7, 162, 1, 153, 0, 0, // Opcode: VLDB_UNPACK_D8_D4_ag_idx, skip to: 394
/* 241 */     MCD::OPC_FilterValue, 3, 8, 0, 0, // Skip to: 254
/* 246 */     MCD::OPC_TryDecode, 168, 7, 163, 1, 140, 0, 0, // Opcode: VLDB_UNPACK_D8_D4_pstm_nrm, skip to: 394
/* 254 */     MCD::OPC_FilterValue, 4, 15, 0, 0, // Skip to: 274
/* 259 */     MCD::OPC_CheckField, 10, 1, 0, 128, 0, 0, // Skip to: 394
/* 266 */     MCD::OPC_TryDecode, 152, 9, 164, 1, 120, 0, 0, // Opcode: VUNPACK_D8_D4, skip to: 394
/* 274 */     MCD::OPC_FilterValue, 5, 8, 0, 0, // Skip to: 287
/* 279 */     MCD::OPC_TryDecode, 134, 7, 165, 1, 107, 0, 0, // Opcode: VLDB_2D_UNPACK_D8_D4, skip to: 394
/* 287 */     MCD::OPC_FilterValue, 7, 102, 0, 0, // Skip to: 394
/* 292 */     MCD::OPC_CheckField, 12, 1, 0, 95, 0, 0, // Skip to: 394
/* 299 */     MCD::OPC_TryDecode, 140, 7, 166, 1, 87, 0, 0, // Opcode: VLDB_3D_UNPACK_D8_D4, skip to: 394
/* 307 */     MCD::OPC_FilterValue, 1, 82, 0, 0, // Skip to: 394
/* 312 */     MCD::OPC_ExtractField, 7, 3,  // Inst{9-7} ...
/* 315 */     MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 328
/* 320 */     MCD::OPC_TryDecode, 171, 7, 162, 1, 66, 0, 0, // Opcode: VLDB_UNPACK_S8_S4_ag_idx, skip to: 394
/* 328 */     MCD::OPC_FilterValue, 3, 8, 0, 0, // Skip to: 341
/* 333 */     MCD::OPC_TryDecode, 172, 7, 163, 1, 53, 0, 0, // Opcode: VLDB_UNPACK_S8_S4_ag_pstm_nrm, skip to: 394
/* 341 */     MCD::OPC_FilterValue, 4, 15, 0, 0, // Skip to: 361
/* 346 */     MCD::OPC_CheckField, 10, 1, 0, 41, 0, 0, // Skip to: 394
/* 353 */     MCD::OPC_TryDecode, 154, 9, 164, 1, 33, 0, 0, // Opcode: VUNPACK_S8_S4, skip to: 394
/* 361 */     MCD::OPC_FilterValue, 5, 8, 0, 0, // Skip to: 374
/* 366 */     MCD::OPC_TryDecode, 136, 7, 165, 1, 20, 0, 0, // Opcode: VLDB_2D_UNPACK_S8_S4, skip to: 394
/* 374 */     MCD::OPC_FilterValue, 7, 15, 0, 0, // Skip to: 394
/* 379 */     MCD::OPC_CheckField, 12, 1, 0, 8, 0, 0, // Skip to: 394
/* 386 */     MCD::OPC_TryDecode, 142, 7, 166, 1, 0, 0, 0, // Opcode: VLDB_3D_UNPACK_S8_S4, skip to: 394
/* 394 */     MCD::OPC_CheckField, 5, 2, 3, 5, 0, 0, // Skip to: 406
/* 401 */     MCD::OPC_Decode, 200, 5, 167, 1, // Opcode: PADDB_sp_imm
/* 406 */     MCD::OPC_ExtractField, 7, 6,  // Inst{12-7} ...
/* 409 */     MCD::OPC_FilterValue, 18, 8, 0, 0, // Skip to: 422
/* 414 */     MCD::OPC_TryDecode, 151, 7, 168, 1, 35, 2, 0, // Opcode: VLDB_COMPR_POP, skip to: 969
/* 422 */     MCD::OPC_FilterValue, 50, 30, 2, 0, // Skip to: 969
/* 427 */     MCD::OPC_TryDecode, 150, 7, 168, 1, 22, 2, 0, // Opcode: VLDB_COMPR_PEEK, skip to: 969
/* 435 */     MCD::OPC_FilterValue, 1, 163, 0, 0, // Skip to: 603
/* 440 */     MCD::OPC_ExtractField, 7, 2,  // Inst{8-7} ...
/* 443 */     MCD::OPC_FilterValue, 0, 51, 0, 0, // Skip to: 499
/* 448 */     MCD::OPC_ExtractField, 9, 1,  // Inst{9} ...
/* 451 */     MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 471
/* 456 */     MCD::OPC_CheckField, 12, 1, 0, 250, 1, 0, // Skip to: 969
/* 463 */     MCD::OPC_TryDecode, 137, 7, 169, 1, 242, 1, 0, // Opcode: VLDB_3D, skip to: 969
/* 471 */     MCD::OPC_FilterValue, 1, 237, 1, 0, // Skip to: 969
/* 476 */     MCD::OPC_CheckField, 5, 2, 3, 8, 0, 0, // Skip to: 491
/* 483 */     MCD::OPC_TryDecode, 196, 5, 170, 1, 0, 0, 0, // Opcode: PADDB_2D, skip to: 491
/* 491 */     MCD::OPC_TryDecode, 173, 7, 171, 1, 214, 1, 0, // Opcode: VLDB_dmw_ldb_ag_idx, skip to: 969
/* 499 */     MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 512
/* 504 */     MCD::OPC_TryDecode, 176, 7, 172, 1, 201, 1, 0, // Opcode: VLDB_dmw_ldb_ag_pstm_nrm_imm, skip to: 969
/* 512 */     MCD::OPC_FilterValue, 2, 66, 0, 0, // Skip to: 583
/* 517 */     MCD::OPC_ExtractField, 9, 1,  // Inst{9} ...
/* 520 */     MCD::OPC_FilterValue, 0, 23, 0, 0, // Skip to: 548
/* 525 */     MCD::OPC_CheckField, 5, 2, 3, 8, 0, 0, // Skip to: 540
/* 532 */     MCD::OPC_TryDecode, 198, 5, 173, 1, 0, 0, 0, // Opcode: PADDB_ldb_ptr_inc_nospill_nrm, skip to: 540
/* 540 */     MCD::OPC_TryDecode, 175, 7, 174, 1, 165, 1, 0, // Opcode: VLDB_dmw_ldb_ag_pstm_nrm, skip to: 969
/* 548 */     MCD::OPC_FilterValue, 1, 160, 1, 0, // Skip to: 969
/* 553 */     MCD::OPC_CheckField, 12, 1, 0, 15, 0, 0, // Skip to: 575
/* 560 */     MCD::OPC_CheckField, 5, 2, 3, 8, 0, 0, // Skip to: 575
/* 567 */     MCD::OPC_TryDecode, 197, 5, 175, 1, 0, 0, 0, // Opcode: PADDB_3D, skip to: 575
/* 575 */     MCD::OPC_TryDecode, 131, 7, 176, 1, 130, 1, 0, // Opcode: VLDB_2D, skip to: 969
/* 583 */     MCD::OPC_FilterValue, 3, 125, 1, 0, // Skip to: 969
/* 588 */     MCD::OPC_CheckField, 9, 1, 0, 118, 1, 0, // Skip to: 969
/* 595 */     MCD::OPC_TryDecode, 174, 7, 177, 1, 110, 1, 0, // Opcode: VLDB_dmw_ldb_ag_idx_imm, skip to: 969
/* 603 */     MCD::OPC_FilterValue, 2, 192, 0, 0, // Skip to: 800
/* 608 */     MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 611 */     MCD::OPC_FilterValue, 0, 82, 0, 0, // Skip to: 698
/* 616 */     MCD::OPC_ExtractField, 7, 3,  // Inst{9-7} ...
/* 619 */     MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 632
/* 624 */     MCD::OPC_TryDecode, 165, 7, 162, 1, 153, 0, 0, // Opcode: VLDB_UNPACK_D16_D8_ag_idx, skip to: 785
/* 632 */     MCD::OPC_FilterValue, 3, 8, 0, 0, // Skip to: 645
/* 637 */     MCD::OPC_TryDecode, 166, 7, 163, 1, 140, 0, 0, // Opcode: VLDB_UNPACK_D16_D8_ag_pstm_nrm, skip to: 785
/* 645 */     MCD::OPC_FilterValue, 4, 15, 0, 0, // Skip to: 665
/* 650 */     MCD::OPC_CheckField, 10, 1, 0, 128, 0, 0, // Skip to: 785
/* 657 */     MCD::OPC_TryDecode, 151, 9, 164, 1, 120, 0, 0, // Opcode: VUNPACK_D16_D8, skip to: 785
/* 665 */     MCD::OPC_FilterValue, 5, 8, 0, 0, // Skip to: 678
/* 670 */     MCD::OPC_TryDecode, 133, 7, 165, 1, 107, 0, 0, // Opcode: VLDB_2D_UNPACK_D16_D8, skip to: 785
/* 678 */     MCD::OPC_FilterValue, 7, 102, 0, 0, // Skip to: 785
/* 683 */     MCD::OPC_CheckField, 12, 1, 0, 95, 0, 0, // Skip to: 785
/* 690 */     MCD::OPC_TryDecode, 139, 7, 166, 1, 87, 0, 0, // Opcode: VLDB_3D_UNPACK_D16_D8, skip to: 785
/* 698 */     MCD::OPC_FilterValue, 1, 82, 0, 0, // Skip to: 785
/* 703 */     MCD::OPC_ExtractField, 7, 3,  // Inst{9-7} ...
/* 706 */     MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 719
/* 711 */     MCD::OPC_TryDecode, 169, 7, 162, 1, 66, 0, 0, // Opcode: VLDB_UNPACK_S16_S8_ag_idx, skip to: 785
/* 719 */     MCD::OPC_FilterValue, 3, 8, 0, 0, // Skip to: 732
/* 724 */     MCD::OPC_TryDecode, 170, 7, 163, 1, 53, 0, 0, // Opcode: VLDB_UNPACK_S16_S8_ag_pstm_nrm, skip to: 785
/* 732 */     MCD::OPC_FilterValue, 4, 15, 0, 0, // Skip to: 752
/* 737 */     MCD::OPC_CheckField, 10, 1, 0, 41, 0, 0, // Skip to: 785
/* 744 */     MCD::OPC_TryDecode, 153, 9, 164, 1, 33, 0, 0, // Opcode: VUNPACK_S16_S8, skip to: 785
/* 752 */     MCD::OPC_FilterValue, 5, 8, 0, 0, // Skip to: 765
/* 757 */     MCD::OPC_TryDecode, 135, 7, 165, 1, 20, 0, 0, // Opcode: VLDB_2D_UNPACK_S16_S8, skip to: 785
/* 765 */     MCD::OPC_FilterValue, 7, 15, 0, 0, // Skip to: 785
/* 770 */     MCD::OPC_CheckField, 12, 1, 0, 8, 0, 0, // Skip to: 785
/* 777 */     MCD::OPC_TryDecode, 141, 7, 166, 1, 0, 0, 0, // Opcode: VLDB_3D_UNPACK_S16_S8, skip to: 785
/* 785 */     MCD::OPC_CheckField, 5, 2, 3, 177, 0, 0, // Skip to: 969
/* 792 */     MCD::OPC_TryDecode, 199, 5, 178, 1, 169, 0, 0, // Opcode: PADDB_ldb_ptr_inc_nrm_imm, skip to: 969
/* 800 */     MCD::OPC_FilterValue, 3, 164, 0, 0, // Skip to: 969
/* 805 */     MCD::OPC_ExtractField, 7, 3,  // Inst{9-7} ...
/* 808 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 821
/* 813 */     MCD::OPC_TryDecode, 129, 7, 171, 1, 148, 0, 0, // Opcode: VLDB_128_ag_idx, skip to: 969
/* 821 */     MCD::OPC_FilterValue, 1, 29, 0, 0, // Skip to: 855
/* 826 */     MCD::OPC_ExtractField, 10, 1,  // Inst{10} ...
/* 829 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 842
/* 834 */     MCD::OPC_TryDecode, 144, 7, 179, 1, 127, 0, 0, // Opcode: VLDB_4x16_LO, skip to: 969
/* 842 */     MCD::OPC_FilterValue, 1, 122, 0, 0, // Skip to: 969
/* 847 */     MCD::OPC_TryDecode, 143, 7, 179, 1, 114, 0, 0, // Opcode: VLDB_4x16_HI, skip to: 969
/* 855 */     MCD::OPC_FilterValue, 2, 8, 0, 0, // Skip to: 868
/* 860 */     MCD::OPC_TryDecode, 130, 7, 174, 1, 101, 0, 0, // Opcode: VLDB_128_ag_pstm_nrm, skip to: 969
/* 868 */     MCD::OPC_FilterValue, 3, 29, 0, 0, // Skip to: 902
/* 873 */     MCD::OPC_ExtractField, 10, 1,  // Inst{10} ...
/* 876 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 889
/* 881 */     MCD::OPC_TryDecode, 146, 7, 179, 1, 80, 0, 0, // Opcode: VLDB_4x32_LO, skip to: 969
/* 889 */     MCD::OPC_FilterValue, 1, 75, 0, 0, // Skip to: 969
/* 894 */     MCD::OPC_TryDecode, 145, 7, 179, 1, 67, 0, 0, // Opcode: VLDB_4x32_HI, skip to: 969
/* 902 */     MCD::OPC_FilterValue, 4, 8, 0, 0, // Skip to: 915
/* 907 */     MCD::OPC_TryDecode, 132, 7, 176, 1, 54, 0, 0, // Opcode: VLDB_2D_128, skip to: 969
/* 915 */     MCD::OPC_FilterValue, 5, 29, 0, 0, // Skip to: 949
/* 920 */     MCD::OPC_ExtractField, 10, 1,  // Inst{10} ...
/* 923 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 936
/* 928 */     MCD::OPC_TryDecode, 148, 7, 179, 1, 33, 0, 0, // Opcode: VLDB_4x64_LO, skip to: 969
/* 936 */     MCD::OPC_FilterValue, 1, 28, 0, 0, // Skip to: 969
/* 941 */     MCD::OPC_TryDecode, 147, 7, 179, 1, 20, 0, 0, // Opcode: VLDB_4x64_HI, skip to: 969
/* 949 */     MCD::OPC_FilterValue, 6, 15, 0, 0, // Skip to: 969
/* 954 */     MCD::OPC_CheckField, 12, 1, 0, 8, 0, 0, // Skip to: 969
/* 961 */     MCD::OPC_TryDecode, 138, 7, 169, 1, 0, 0, 0, // Opcode: VLDB_3D_128, skip to: 969
/* 969 */     MCD::OPC_Fail,
  0
};

static const uint8_t DecoderTableLng48[] = {
/* 0 */       MCD::OPC_ExtractField, 0, 3,  // Inst{2-0} ...
/* 3 */       MCD::OPC_FilterValue, 0, 11, 0, 0, // Skip to: 19
/* 8 */       MCD::OPC_CheckField, 3, 39, 0, 100, 0, 0, // Skip to: 115
/* 15 */      MCD::OPC_Decode, 189, 5, 0, // Opcode: NOPXM
/* 19 */      MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 32
/* 24 */      MCD::OPC_TryDecode, 146, 5, 180, 1, 83, 0, 0, // Opcode: MOVXM_lng_cg, skip to: 115
/* 32 */      MCD::OPC_FilterValue, 2, 19, 0, 0, // Skip to: 56
/* 37 */      MCD::OPC_CheckField, 37, 5, 0, 71, 0, 0, // Skip to: 115
/* 44 */      MCD::OPC_CheckField, 3, 14, 0, 64, 0, 0, // Skip to: 115
/* 51 */      MCD::OPC_Decode, 229, 4, 181, 1, // Opcode: J_jump_imm
/* 56 */      MCD::OPC_FilterValue, 4, 19, 0, 0, // Skip to: 80
/* 61 */      MCD::OPC_CheckField, 37, 5, 0, 47, 0, 0, // Skip to: 115
/* 68 */      MCD::OPC_CheckField, 3, 14, 0, 40, 0, 0, // Skip to: 115
/* 75 */      MCD::OPC_Decode, 224, 4, 182, 1, // Opcode: JL
/* 80 */      MCD::OPC_FilterValue, 6, 30, 0, 0, // Skip to: 115
/* 85 */      MCD::OPC_ExtractField, 3, 14,  // Inst{16-3} ...
/* 88 */      MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 101
/* 93 */      MCD::OPC_TryDecode, 228, 4, 183, 1, 14, 0, 0, // Opcode: JZ, skip to: 115
/* 101 */     MCD::OPC_FilterValue, 128, 64, 8, 0, 0, // Skip to: 115
/* 107 */     MCD::OPC_TryDecode, 226, 4, 183, 1, 0, 0, 0, // Opcode: JNZ, skip to: 115
/* 115 */     MCD::OPC_Fail,
  0
};

static const uint8_t DecoderTableMv32[] = {
/* 0 */       MCD::OPC_ExtractField, 0, 2,  // Inst{1-0} ...
/* 3 */       MCD::OPC_FilterValue, 0, 163, 1, 0, // Skip to: 427
/* 8 */       MCD::OPC_CheckField, 2, 20, 14, 4, 0, 0, // Skip to: 19
/* 15 */      MCD::OPC_Decode, 185, 5, 0, // Opcode: NOPM
/* 19 */      MCD::OPC_CheckField, 4, 2, 3, 15, 0, 0, // Skip to: 41
/* 26 */      MCD::OPC_CheckField, 2, 1, 1, 8, 0, 0, // Skip to: 41
/* 33 */      MCD::OPC_TryDecode, 170, 5, 184, 1, 0, 0, 0, // Opcode: MOV_mv_cg, skip to: 41
/* 41 */      MCD::OPC_ExtractField, 10, 5,  // Inst{14-10} ...
/* 44 */      MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 64
/* 49 */      MCD::OPC_CheckField, 15, 3, 0, 71, 9, 0, // Skip to: 2431
/* 56 */      MCD::OPC_TryDecode, 137, 6, 185, 1, 63, 9, 0, // Opcode: VADD_8, skip to: 2431
/* 64 */      MCD::OPC_FilterValue, 2, 15, 0, 0, // Skip to: 84
/* 69 */      MCD::OPC_CheckField, 15, 3, 0, 51, 9, 0, // Skip to: 2431
/* 76 */      MCD::OPC_TryDecode, 137, 9, 185, 1, 43, 9, 0, // Opcode: VSUB_8, skip to: 2431
/* 84 */      MCD::OPC_FilterValue, 4, 8, 0, 0, // Skip to: 97
/* 89 */      MCD::OPC_TryDecode, 137, 8, 186, 1, 30, 9, 0, // Opcode: VSEL_8, skip to: 2431
/* 97 */      MCD::OPC_FilterValue, 5, 8, 0, 0, // Skip to: 110
/* 102 */     MCD::OPC_TryDecode, 134, 6, 186, 1, 17, 9, 0, // Opcode: VADDSUB_8, skip to: 2431
/* 110 */     MCD::OPC_FilterValue, 6, 15, 0, 0, // Skip to: 130
/* 115 */     MCD::OPC_CheckField, 6, 4, 0, 5, 9, 0, // Skip to: 2431
/* 122 */     MCD::OPC_TryDecode, 243, 5, 187, 1, 253, 8, 0, // Opcode: VABS_GTZ_D8, skip to: 2431
/* 130 */     MCD::OPC_FilterValue, 7, 15, 0, 0, // Skip to: 150
/* 135 */     MCD::OPC_CheckField, 6, 4, 0, 241, 8, 0, // Skip to: 2431
/* 142 */     MCD::OPC_TryDecode, 246, 5, 187, 1, 233, 8, 0, // Opcode: VABS_GTZ_S8, skip to: 2431
/* 150 */     MCD::OPC_FilterValue, 8, 8, 0, 0, // Skip to: 163
/* 155 */     MCD::OPC_TryDecode, 192, 7, 188, 1, 220, 8, 0, // Opcode: VMAXDIFF_LT_D8, skip to: 2431
/* 163 */     MCD::OPC_FilterValue, 9, 8, 0, 0, // Skip to: 176
/* 168 */     MCD::OPC_TryDecode, 195, 7, 188, 1, 207, 8, 0, // Opcode: VMAXDIFF_LT_S8, skip to: 2431
/* 176 */     MCD::OPC_FilterValue, 10, 8, 0, 0, // Skip to: 189
/* 181 */     MCD::OPC_TryDecode, 147, 9, 188, 1, 194, 8, 0, // Opcode: VSUB_LT_D8, skip to: 2431
/* 189 */     MCD::OPC_FilterValue, 11, 8, 0, 0, // Skip to: 202
/* 194 */     MCD::OPC_TryDecode, 150, 9, 188, 1, 181, 8, 0, // Opcode: VSUB_LT_S8, skip to: 2431
/* 202 */     MCD::OPC_FilterValue, 12, 8, 0, 0, // Skip to: 215
/* 207 */     MCD::OPC_TryDecode, 141, 9, 188, 1, 168, 8, 0, // Opcode: VSUB_GE_D8, skip to: 2431
/* 215 */     MCD::OPC_FilterValue, 13, 8, 0, 0, // Skip to: 228
/* 220 */     MCD::OPC_TryDecode, 144, 9, 188, 1, 155, 8, 0, // Opcode: VSUB_GE_S8, skip to: 2431
/* 228 */     MCD::OPC_FilterValue, 14, 15, 0, 0, // Skip to: 248
/* 233 */     MCD::OPC_CheckField, 6, 4, 0, 143, 8, 0, // Skip to: 2431
/* 240 */     MCD::OPC_TryDecode, 150, 6, 187, 1, 135, 8, 0, // Opcode: VBNEG_LTZ_S8, skip to: 2431
/* 248 */     MCD::OPC_FilterValue, 15, 22, 0, 0, // Skip to: 275
/* 253 */     MCD::OPC_CheckField, 18, 4, 0, 123, 8, 0, // Skip to: 2431
/* 260 */     MCD::OPC_CheckField, 6, 4, 0, 116, 8, 0, // Skip to: 2431
/* 267 */     MCD::OPC_TryDecode, 158, 6, 189, 1, 108, 8, 0, // Opcode: VEQZ_8, skip to: 2431
/* 275 */     MCD::OPC_FilterValue, 16, 8, 0, 0, // Skip to: 288
/* 280 */     MCD::OPC_TryDecode, 206, 7, 188, 1, 95, 8, 0, // Opcode: VMIN_GE_D8, skip to: 2431
/* 288 */     MCD::OPC_FilterValue, 17, 8, 0, 0, // Skip to: 301
/* 293 */     MCD::OPC_TryDecode, 209, 7, 188, 1, 82, 8, 0, // Opcode: VMIN_GE_S8, skip to: 2431
/* 301 */     MCD::OPC_FilterValue, 20, 8, 0, 0, // Skip to: 314
/* 306 */     MCD::OPC_TryDecode, 199, 7, 188, 1, 69, 8, 0, // Opcode: VMAX_LT_D8, skip to: 2431
/* 314 */     MCD::OPC_FilterValue, 21, 8, 0, 0, // Skip to: 327
/* 319 */     MCD::OPC_TryDecode, 202, 7, 188, 1, 56, 8, 0, // Opcode: VMAX_LT_S8, skip to: 2431
/* 327 */     MCD::OPC_FilterValue, 24, 15, 0, 0, // Skip to: 347
/* 332 */     MCD::OPC_CheckField, 18, 4, 0, 44, 8, 0, // Skip to: 2431
/* 339 */     MCD::OPC_TryDecode, 180, 7, 190, 1, 36, 8, 0, // Opcode: VLT_D8, skip to: 2431
/* 347 */     MCD::OPC_FilterValue, 25, 15, 0, 0, // Skip to: 367
/* 352 */     MCD::OPC_CheckField, 18, 4, 0, 24, 8, 0, // Skip to: 2431
/* 359 */     MCD::OPC_TryDecode, 183, 7, 190, 1, 16, 8, 0, // Opcode: VLT_S8, skip to: 2431
/* 367 */     MCD::OPC_FilterValue, 27, 15, 0, 0, // Skip to: 387
/* 372 */     MCD::OPC_CheckField, 6, 4, 0, 4, 8, 0, // Skip to: 2431
/* 379 */     MCD::OPC_TryDecode, 250, 7, 187, 1, 252, 7, 0, // Opcode: VNEG_GTZ8, skip to: 2431
/* 387 */     MCD::OPC_FilterValue, 28, 15, 0, 0, // Skip to: 407
/* 392 */     MCD::OPC_CheckField, 18, 4, 0, 240, 7, 0, // Skip to: 2431
/* 399 */     MCD::OPC_TryDecode, 180, 6, 190, 1, 232, 7, 0, // Opcode: VGE_D8, skip to: 2431
/* 407 */     MCD::OPC_FilterValue, 29, 227, 7, 0, // Skip to: 2431
/* 412 */     MCD::OPC_CheckField, 18, 4, 0, 220, 7, 0, // Skip to: 2431
/* 419 */     MCD::OPC_TryDecode, 183, 6, 190, 1, 212, 7, 0, // Opcode: VGE_S8, skip to: 2431
/* 427 */     MCD::OPC_FilterValue, 1, 166, 3, 0, // Skip to: 1366
/* 432 */     MCD::OPC_ExtractField, 6, 9,  // Inst{14-6} ...
/* 435 */     MCD::OPC_FilterValue, 96, 8, 0, 0, // Skip to: 448
/* 440 */     MCD::OPC_TryDecode, 241, 5, 191, 1, 62, 0, 0, // Opcode: VABS_GTZ_D16, skip to: 510
/* 448 */     MCD::OPC_FilterValue, 112, 8, 0, 0, // Skip to: 461
/* 453 */     MCD::OPC_TryDecode, 244, 5, 191, 1, 49, 0, 0, // Opcode: VABS_GTZ_S16, skip to: 510
/* 461 */     MCD::OPC_FilterValue, 224, 1, 8, 0, 0, // Skip to: 475
/* 467 */     MCD::OPC_TryDecode, 148, 6, 191, 1, 35, 0, 0, // Opcode: VBNEG_LTZ_S16, skip to: 510
/* 475 */     MCD::OPC_FilterValue, 240, 1, 15, 0, 0, // Skip to: 496
/* 481 */     MCD::OPC_CheckField, 18, 4, 0, 22, 0, 0, // Skip to: 510
/* 488 */     MCD::OPC_TryDecode, 156, 6, 192, 1, 14, 0, 0, // Opcode: VEQZ_16, skip to: 510
/* 496 */     MCD::OPC_FilterValue, 176, 3, 8, 0, 0, // Skip to: 510
/* 502 */     MCD::OPC_TryDecode, 248, 7, 191, 1, 0, 0, 0, // Opcode: VNEG_GTZ16, skip to: 510
/* 510 */     MCD::OPC_ExtractField, 10, 5,  // Inst{14-10} ...
/* 513 */     MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 533
/* 518 */     MCD::OPC_CheckField, 15, 3, 0, 74, 1, 0, // Skip to: 855
/* 525 */     MCD::OPC_TryDecode, 135, 6, 185, 1, 66, 1, 0, // Opcode: VADD_16, skip to: 855
/* 533 */     MCD::OPC_FilterValue, 2, 15, 0, 0, // Skip to: 553
/* 538 */     MCD::OPC_CheckField, 15, 3, 0, 54, 1, 0, // Skip to: 855
/* 545 */     MCD::OPC_TryDecode, 135, 9, 185, 1, 46, 1, 0, // Opcode: VSUB_16, skip to: 855
/* 553 */     MCD::OPC_FilterValue, 4, 8, 0, 0, // Skip to: 566
/* 558 */     MCD::OPC_TryDecode, 135, 8, 193, 1, 33, 1, 0, // Opcode: VSEL_16, skip to: 855
/* 566 */     MCD::OPC_FilterValue, 5, 8, 0, 0, // Skip to: 579
/* 571 */     MCD::OPC_TryDecode, 132, 6, 193, 1, 20, 1, 0, // Opcode: VADDSUB_16, skip to: 855
/* 579 */     MCD::OPC_FilterValue, 8, 8, 0, 0, // Skip to: 592
/* 584 */     MCD::OPC_TryDecode, 190, 7, 194, 1, 7, 1, 0, // Opcode: VMAXDIFF_LT_D16, skip to: 855
/* 592 */     MCD::OPC_FilterValue, 9, 8, 0, 0, // Skip to: 605
/* 597 */     MCD::OPC_TryDecode, 193, 7, 194, 1, 250, 0, 0, // Opcode: VMAXDIFF_LT_S16, skip to: 855
/* 605 */     MCD::OPC_FilterValue, 10, 8, 0, 0, // Skip to: 618
/* 610 */     MCD::OPC_TryDecode, 145, 9, 194, 1, 237, 0, 0, // Opcode: VSUB_LT_D16, skip to: 855
/* 618 */     MCD::OPC_FilterValue, 11, 8, 0, 0, // Skip to: 631
/* 623 */     MCD::OPC_TryDecode, 148, 9, 194, 1, 224, 0, 0, // Opcode: VSUB_LT_S16, skip to: 855
/* 631 */     MCD::OPC_FilterValue, 12, 8, 0, 0, // Skip to: 644
/* 636 */     MCD::OPC_TryDecode, 139, 9, 194, 1, 211, 0, 0, // Opcode: VSUB_GE_D16, skip to: 855
/* 644 */     MCD::OPC_FilterValue, 13, 8, 0, 0, // Skip to: 657
/* 649 */     MCD::OPC_TryDecode, 142, 9, 194, 1, 198, 0, 0, // Opcode: VSUB_GE_S16, skip to: 855
/* 657 */     MCD::OPC_FilterValue, 16, 8, 0, 0, // Skip to: 670
/* 662 */     MCD::OPC_TryDecode, 204, 7, 194, 1, 185, 0, 0, // Opcode: VMIN_GE_D16, skip to: 855
/* 670 */     MCD::OPC_FilterValue, 17, 8, 0, 0, // Skip to: 683
/* 675 */     MCD::OPC_TryDecode, 207, 7, 194, 1, 172, 0, 0, // Opcode: VMIN_GE_S16, skip to: 855
/* 683 */     MCD::OPC_FilterValue, 18, 8, 0, 0, // Skip to: 696
/* 688 */     MCD::OPC_TryDecode, 203, 7, 194, 1, 159, 0, 0, // Opcode: VMIN_GE_BF16, skip to: 855
/* 696 */     MCD::OPC_FilterValue, 20, 8, 0, 0, // Skip to: 709
/* 701 */     MCD::OPC_TryDecode, 197, 7, 194, 1, 146, 0, 0, // Opcode: VMAX_LT_D16, skip to: 855
/* 709 */     MCD::OPC_FilterValue, 21, 8, 0, 0, // Skip to: 722
/* 714 */     MCD::OPC_TryDecode, 200, 7, 194, 1, 133, 0, 0, // Opcode: VMAX_LT_S16, skip to: 855
/* 722 */     MCD::OPC_FilterValue, 22, 8, 0, 0, // Skip to: 735
/* 727 */     MCD::OPC_TryDecode, 196, 7, 194, 1, 120, 0, 0, // Opcode: VMAX_LT_BF16, skip to: 855
/* 735 */     MCD::OPC_FilterValue, 24, 15, 0, 0, // Skip to: 755
/* 740 */     MCD::OPC_CheckField, 18, 4, 0, 108, 0, 0, // Skip to: 855
/* 747 */     MCD::OPC_TryDecode, 178, 7, 195, 1, 100, 0, 0, // Opcode: VLT_D16, skip to: 855
/* 755 */     MCD::OPC_FilterValue, 25, 15, 0, 0, // Skip to: 775
/* 760 */     MCD::OPC_CheckField, 18, 4, 0, 88, 0, 0, // Skip to: 855
/* 767 */     MCD::OPC_TryDecode, 181, 7, 195, 1, 80, 0, 0, // Opcode: VLT_S16, skip to: 855
/* 775 */     MCD::OPC_FilterValue, 26, 15, 0, 0, // Skip to: 795
/* 780 */     MCD::OPC_CheckField, 18, 4, 0, 68, 0, 0, // Skip to: 855
/* 787 */     MCD::OPC_TryDecode, 177, 7, 195, 1, 60, 0, 0, // Opcode: VLT_BF16, skip to: 855
/* 795 */     MCD::OPC_FilterValue, 28, 15, 0, 0, // Skip to: 815
/* 800 */     MCD::OPC_CheckField, 18, 4, 0, 48, 0, 0, // Skip to: 855
/* 807 */     MCD::OPC_TryDecode, 178, 6, 195, 1, 40, 0, 0, // Opcode: VGE_D16, skip to: 855
/* 815 */     MCD::OPC_FilterValue, 29, 15, 0, 0, // Skip to: 835
/* 820 */     MCD::OPC_CheckField, 18, 4, 0, 28, 0, 0, // Skip to: 855
/* 827 */     MCD::OPC_TryDecode, 181, 6, 195, 1, 20, 0, 0, // Opcode: VGE_S16, skip to: 855
/* 835 */     MCD::OPC_FilterValue, 30, 15, 0, 0, // Skip to: 855
/* 840 */     MCD::OPC_CheckField, 18, 4, 0, 8, 0, 0, // Skip to: 855
/* 847 */     MCD::OPC_TryDecode, 177, 6, 195, 1, 0, 0, 0, // Opcode: VGE_BF16, skip to: 855
/* 855 */     MCD::OPC_ExtractField, 2, 4,  // Inst{5-2} ...
/* 858 */     MCD::OPC_FilterValue, 12, 55, 0, 0, // Skip to: 918
/* 863 */     MCD::OPC_ExtractField, 15, 3,  // Inst{17-15} ...
/* 866 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 879
/* 871 */     MCD::OPC_TryDecode, 187, 6, 196, 1, 16, 6, 0, // Opcode: VINSERT_8, skip to: 2431
/* 879 */     MCD::OPC_FilterValue, 2, 8, 0, 0, // Skip to: 892
/* 884 */     MCD::OPC_TryDecode, 184, 6, 196, 1, 3, 6, 0, // Opcode: VINSERT_16, skip to: 2431
/* 892 */     MCD::OPC_FilterValue, 4, 8, 0, 0, // Skip to: 905
/* 897 */     MCD::OPC_TryDecode, 185, 6, 196, 1, 246, 5, 0, // Opcode: VINSERT_32, skip to: 2431
/* 905 */     MCD::OPC_FilterValue, 6, 241, 5, 0, // Skip to: 2431
/* 910 */     MCD::OPC_TryDecode, 186, 6, 197, 1, 233, 5, 0, // Opcode: VINSERT_64, skip to: 2431
/* 918 */     MCD::OPC_FilterValue, 13, 107, 0, 0, // Skip to: 1030
/* 923 */     MCD::OPC_ExtractField, 12, 3,  // Inst{14-12} ...
/* 926 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 939
/* 931 */     MCD::OPC_TryDecode, 170, 6, 198, 1, 212, 5, 0, // Opcode: VEXTRACT_D8, skip to: 2431
/* 939 */     MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 952
/* 944 */     MCD::OPC_TryDecode, 174, 6, 198, 1, 199, 5, 0, // Opcode: VEXTRACT_S8, skip to: 2431
/* 952 */     MCD::OPC_FilterValue, 2, 8, 0, 0, // Skip to: 965
/* 957 */     MCD::OPC_TryDecode, 167, 6, 198, 1, 186, 5, 0, // Opcode: VEXTRACT_D16, skip to: 2431
/* 965 */     MCD::OPC_FilterValue, 3, 8, 0, 0, // Skip to: 978
/* 970 */     MCD::OPC_TryDecode, 171, 6, 198, 1, 173, 5, 0, // Opcode: VEXTRACT_S16, skip to: 2431
/* 978 */     MCD::OPC_FilterValue, 4, 8, 0, 0, // Skip to: 991
/* 983 */     MCD::OPC_TryDecode, 168, 6, 198, 1, 160, 5, 0, // Opcode: VEXTRACT_D32, skip to: 2431
/* 991 */     MCD::OPC_FilterValue, 5, 8, 0, 0, // Skip to: 1004
/* 996 */     MCD::OPC_TryDecode, 172, 6, 198, 1, 147, 5, 0, // Opcode: VEXTRACT_S32, skip to: 2431
/* 1004 */    MCD::OPC_FilterValue, 6, 8, 0, 0, // Skip to: 1017
/* 1009 */    MCD::OPC_TryDecode, 169, 6, 199, 1, 134, 5, 0, // Opcode: VEXTRACT_D64, skip to: 2431
/* 1017 */    MCD::OPC_FilterValue, 7, 129, 5, 0, // Skip to: 2431
/* 1022 */    MCD::OPC_TryDecode, 173, 6, 199, 1, 121, 5, 0, // Opcode: VEXTRACT_S64, skip to: 2431
/* 1030 */    MCD::OPC_FilterValue, 14, 107, 0, 0, // Skip to: 1142
/* 1035 */    MCD::OPC_ExtractField, 15, 3,  // Inst{17-15} ...
/* 1038 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 1051
/* 1043 */    MCD::OPC_TryDecode, 166, 6, 200, 1, 100, 5, 0, // Opcode: VEXTBCST_8_mRm, skip to: 2431
/* 1051 */    MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 1064
/* 1056 */    MCD::OPC_TryDecode, 165, 6, 201, 1, 87, 5, 0, // Opcode: VEXTBCST_8_mExtractIdxImm, skip to: 2431
/* 1064 */    MCD::OPC_FilterValue, 2, 8, 0, 0, // Skip to: 1077
/* 1069 */    MCD::OPC_TryDecode, 160, 6, 200, 1, 74, 5, 0, // Opcode: VEXTBCST_16_mRm, skip to: 2431
/* 1077 */    MCD::OPC_FilterValue, 3, 8, 0, 0, // Skip to: 1090
/* 1082 */    MCD::OPC_TryDecode, 159, 6, 201, 1, 61, 5, 0, // Opcode: VEXTBCST_16_mExtractIdxImm, skip to: 2431
/* 1090 */    MCD::OPC_FilterValue, 4, 8, 0, 0, // Skip to: 1103
/* 1095 */    MCD::OPC_TryDecode, 162, 6, 200, 1, 48, 5, 0, // Opcode: VEXTBCST_32_mRm, skip to: 2431
/* 1103 */    MCD::OPC_FilterValue, 5, 8, 0, 0, // Skip to: 1116
/* 1108 */    MCD::OPC_TryDecode, 161, 6, 201, 1, 35, 5, 0, // Opcode: VEXTBCST_32_mExtractIdxImm, skip to: 2431
/* 1116 */    MCD::OPC_FilterValue, 6, 8, 0, 0, // Skip to: 1129
/* 1121 */    MCD::OPC_TryDecode, 164, 6, 200, 1, 22, 5, 0, // Opcode: VEXTBCST_64_mRm, skip to: 2431
/* 1129 */    MCD::OPC_FilterValue, 7, 17, 5, 0, // Skip to: 2431
/* 1134 */    MCD::OPC_TryDecode, 163, 6, 201, 1, 9, 5, 0, // Opcode: VEXTBCST_64_mExtractIdxImm, skip to: 2431
/* 1142 */    MCD::OPC_FilterValue, 15, 4, 5, 0, // Skip to: 2431
/* 1147 */    MCD::OPC_ExtractField, 15, 3,  // Inst{17-15} ...
/* 1150 */    MCD::OPC_FilterValue, 0, 36, 0, 0, // Skip to: 1191
/* 1155 */    MCD::OPC_ExtractField, 6, 3,  // Inst{8-6} ...
/* 1158 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 1178
/* 1163 */    MCD::OPC_CheckField, 9, 1, 0, 237, 4, 0, // Skip to: 2431
/* 1170 */    MCD::OPC_TryDecode, 147, 6, 202, 1, 229, 4, 0, // Opcode: VBCST_8, skip to: 2431
/* 1178 */    MCD::OPC_FilterValue, 4, 224, 4, 0, // Skip to: 2431
/* 1183 */    MCD::OPC_TryDecode, 143, 6, 203, 1, 216, 4, 0, // Opcode: VBCSTSHFL_8, skip to: 2431
/* 1191 */    MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 1204
/* 1196 */    MCD::OPC_TryDecode, 130, 8, 200, 1, 203, 4, 0, // Opcode: VPUSH_HI_8, skip to: 2431
/* 1204 */    MCD::OPC_FilterValue, 2, 36, 0, 0, // Skip to: 1245
/* 1209 */    MCD::OPC_ExtractField, 6, 3,  // Inst{8-6} ...
/* 1212 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 1232
/* 1217 */    MCD::OPC_CheckField, 9, 1, 0, 183, 4, 0, // Skip to: 2431
/* 1224 */    MCD::OPC_TryDecode, 144, 6, 202, 1, 175, 4, 0, // Opcode: VBCST_16, skip to: 2431
/* 1232 */    MCD::OPC_FilterValue, 4, 170, 4, 0, // Skip to: 2431
/* 1237 */    MCD::OPC_TryDecode, 140, 6, 203, 1, 162, 4, 0, // Opcode: VBCSTSHFL_16, skip to: 2431
/* 1245 */    MCD::OPC_FilterValue, 3, 8, 0, 0, // Skip to: 1258
/* 1250 */    MCD::OPC_TryDecode, 255, 7, 200, 1, 149, 4, 0, // Opcode: VPUSH_HI_16, skip to: 2431
/* 1258 */    MCD::OPC_FilterValue, 4, 36, 0, 0, // Skip to: 1299
/* 1263 */    MCD::OPC_ExtractField, 6, 3,  // Inst{8-6} ...
/* 1266 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 1286
/* 1271 */    MCD::OPC_CheckField, 9, 1, 0, 129, 4, 0, // Skip to: 2431
/* 1278 */    MCD::OPC_TryDecode, 145, 6, 202, 1, 121, 4, 0, // Opcode: VBCST_32, skip to: 2431
/* 1286 */    MCD::OPC_FilterValue, 4, 116, 4, 0, // Skip to: 2431
/* 1291 */    MCD::OPC_TryDecode, 141, 6, 203, 1, 108, 4, 0, // Opcode: VBCSTSHFL_32, skip to: 2431
/* 1299 */    MCD::OPC_FilterValue, 5, 8, 0, 0, // Skip to: 1312
/* 1304 */    MCD::OPC_TryDecode, 128, 8, 200, 1, 95, 4, 0, // Opcode: VPUSH_HI_32, skip to: 2431
/* 1312 */    MCD::OPC_FilterValue, 6, 36, 0, 0, // Skip to: 1353
/* 1317 */    MCD::OPC_ExtractField, 6, 3,  // Inst{8-6} ...
/* 1320 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 1340
/* 1325 */    MCD::OPC_CheckField, 9, 1, 0, 75, 4, 0, // Skip to: 2431
/* 1332 */    MCD::OPC_TryDecode, 146, 6, 204, 1, 67, 4, 0, // Opcode: VBCST_64, skip to: 2431
/* 1340 */    MCD::OPC_FilterValue, 4, 62, 4, 0, // Skip to: 2431
/* 1345 */    MCD::OPC_TryDecode, 142, 6, 205, 1, 54, 4, 0, // Opcode: VBCSTSHFL_64, skip to: 2431
/* 1353 */    MCD::OPC_FilterValue, 7, 49, 4, 0, // Skip to: 2431
/* 1358 */    MCD::OPC_TryDecode, 129, 8, 206, 1, 41, 4, 0, // Opcode: VPUSH_HI_64, skip to: 2431
/* 1366 */    MCD::OPC_FilterValue, 2, 152, 3, 0, // Skip to: 2291
/* 1371 */    MCD::OPC_CheckField, 18, 4, 0, 15, 0, 0, // Skip to: 1393
/* 1378 */    MCD::OPC_CheckField, 2, 13, 76, 8, 0, 0, // Skip to: 1393
/* 1385 */    MCD::OPC_TryDecode, 149, 5, 207, 1, 0, 0, 0, // Opcode: MOV_CNTR, skip to: 1393
/* 1393 */    MCD::OPC_ExtractField, 2, 6,  // Inst{7-2} ...
/* 1396 */    MCD::OPC_FilterValue, 13, 22, 0, 0, // Skip to: 1423
/* 1401 */    MCD::OPC_CheckField, 15, 3, 1, 182, 0, 0, // Skip to: 1590
/* 1408 */    MCD::OPC_CheckField, 8, 3, 0, 175, 0, 0, // Skip to: 1590
/* 1415 */    MCD::OPC_TryDecode, 213, 7, 208, 1, 167, 0, 0, // Opcode: VMOV_mv_cm, skip to: 1590
/* 1423 */    MCD::OPC_FilterValue, 14, 55, 0, 0, // Skip to: 1483
/* 1428 */    MCD::OPC_ExtractField, 8, 2,  // Inst{9-8} ...
/* 1431 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 1444
/* 1436 */    MCD::OPC_TryDecode, 141, 8, 209, 1, 146, 0, 0, // Opcode: VSRSM_D16_S32, skip to: 1590
/* 1444 */    MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 1457
/* 1449 */    MCD::OPC_TryDecode, 143, 8, 209, 1, 133, 0, 0, // Opcode: VSRSM_S16_S32, skip to: 1590
/* 1457 */    MCD::OPC_FilterValue, 2, 8, 0, 0, // Skip to: 1470
/* 1462 */    MCD::OPC_TryDecode, 142, 8, 209, 1, 120, 0, 0, // Opcode: VSRSM_D32_S64, skip to: 1590
/* 1470 */    MCD::OPC_FilterValue, 3, 115, 0, 0, // Skip to: 1590
/* 1475 */    MCD::OPC_TryDecode, 144, 8, 209, 1, 107, 0, 0, // Opcode: VSRSM_S32_S64, skip to: 1590
/* 1483 */    MCD::OPC_FilterValue, 28, 42, 0, 0, // Skip to: 1530
/* 1488 */    MCD::OPC_ExtractField, 15, 2,  // Inst{16-15} ...
/* 1491 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 1504
/* 1496 */    MCD::OPC_TryDecode, 150, 5, 210, 1, 86, 0, 0, // Opcode: MOV_D1, skip to: 1590
/* 1504 */    MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 1517
/* 1509 */    MCD::OPC_TryDecode, 152, 5, 210, 1, 73, 0, 0, // Opcode: MOV_D3, skip to: 1590
/* 1517 */    MCD::OPC_FilterValue, 2, 68, 0, 0, // Skip to: 1590
/* 1522 */    MCD::OPC_TryDecode, 154, 5, 210, 1, 60, 0, 0, // Opcode: MOV_D5, skip to: 1590
/* 1530 */    MCD::OPC_FilterValue, 44, 8, 0, 0, // Skip to: 1543
/* 1535 */    MCD::OPC_TryDecode, 175, 5, 211, 1, 47, 0, 0, // Opcode: MOV_mv_scl, skip to: 1590
/* 1543 */    MCD::OPC_FilterValue, 60, 42, 0, 0, // Skip to: 1590
/* 1548 */    MCD::OPC_ExtractField, 15, 2,  // Inst{16-15} ...
/* 1551 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 1564
/* 1556 */    MCD::OPC_TryDecode, 151, 5, 210, 1, 26, 0, 0, // Opcode: MOV_D2, skip to: 1590
/* 1564 */    MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 1577
/* 1569 */    MCD::OPC_TryDecode, 153, 5, 210, 1, 13, 0, 0, // Opcode: MOV_D4, skip to: 1590
/* 1577 */    MCD::OPC_FilterValue, 2, 8, 0, 0, // Skip to: 1590
/* 1582 */    MCD::OPC_TryDecode, 155, 5, 210, 1, 0, 0, 0, // Opcode: MOV_D6, skip to: 1590
/* 1590 */    MCD::OPC_ExtractField, 6, 9,  // Inst{14-6} ...
/* 1593 */    MCD::OPC_FilterValue, 96, 8, 0, 0, // Skip to: 1606
/* 1598 */    MCD::OPC_TryDecode, 242, 5, 191, 1, 62, 0, 0, // Opcode: VABS_GTZ_D32, skip to: 1668
/* 1606 */    MCD::OPC_FilterValue, 112, 8, 0, 0, // Skip to: 1619
/* 1611 */    MCD::OPC_TryDecode, 245, 5, 191, 1, 49, 0, 0, // Opcode: VABS_GTZ_S32, skip to: 1668
/* 1619 */    MCD::OPC_FilterValue, 224, 1, 8, 0, 0, // Skip to: 1633
/* 1625 */    MCD::OPC_TryDecode, 149, 6, 191, 1, 35, 0, 0, // Opcode: VBNEG_LTZ_S32, skip to: 1668
/* 1633 */    MCD::OPC_FilterValue, 240, 1, 15, 0, 0, // Skip to: 1654
/* 1639 */    MCD::OPC_CheckField, 18, 4, 0, 22, 0, 0, // Skip to: 1668
/* 1646 */    MCD::OPC_TryDecode, 157, 6, 192, 1, 14, 0, 0, // Opcode: VEQZ_32, skip to: 1668
/* 1654 */    MCD::OPC_FilterValue, 176, 3, 8, 0, 0, // Skip to: 1668
/* 1660 */    MCD::OPC_TryDecode, 249, 7, 191, 1, 0, 0, 0, // Opcode: VNEG_GTZ32, skip to: 1668
/* 1668 */    MCD::OPC_ExtractField, 2, 4,  // Inst{5-2} ...
/* 1671 */    MCD::OPC_FilterValue, 13, 15, 0, 0, // Skip to: 1691
/* 1676 */    MCD::OPC_CheckField, 15, 1, 0, 33, 1, 0, // Skip to: 1972
/* 1683 */    MCD::OPC_TryDecode, 217, 7, 212, 1, 25, 1, 0, // Opcode: VMOV_mv_x, skip to: 1972
/* 1691 */    MCD::OPC_FilterValue, 14, 8, 0, 0, // Skip to: 1704
/* 1696 */    MCD::OPC_TryDecode, 216, 7, 213, 1, 12, 1, 0, // Opcode: VMOV_mv_w, skip to: 1972
/* 1704 */    MCD::OPC_FilterValue, 15, 7, 1, 0, // Skip to: 1972
/* 1709 */    MCD::OPC_ExtractField, 11, 4,  // Inst{14-11} ...
/* 1712 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 1732
/* 1717 */    MCD::OPC_CheckField, 15, 2, 0, 248, 0, 0, // Skip to: 1972
/* 1724 */    MCD::OPC_TryDecode, 155, 6, 214, 1, 240, 0, 0, // Opcode: VCONV_FP32_BF16, skip to: 1972
/* 1732 */    MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 1745
/* 1737 */    MCD::OPC_TryDecode, 155, 9, 215, 1, 227, 0, 0, // Opcode: VUPS_S32_D16_mv_ups_w2b, skip to: 1972
/* 1745 */    MCD::OPC_FilterValue, 2, 22, 0, 0, // Skip to: 1772
/* 1750 */    MCD::OPC_CheckField, 17, 1, 0, 215, 0, 0, // Skip to: 1972
/* 1757 */    MCD::OPC_CheckField, 10, 1, 0, 208, 0, 0, // Skip to: 1972
/* 1764 */    MCD::OPC_TryDecode, 156, 9, 216, 1, 200, 0, 0, // Opcode: VUPS_S32_D16_mv_ups_x2c, skip to: 1972
/* 1772 */    MCD::OPC_FilterValue, 3, 15, 0, 0, // Skip to: 1792
/* 1777 */    MCD::OPC_CheckField, 17, 1, 0, 188, 0, 0, // Skip to: 1972
/* 1784 */    MCD::OPC_TryDecode, 157, 9, 217, 1, 180, 0, 0, // Opcode: VUPS_S32_D8_mv_ups_w2c, skip to: 1972
/* 1792 */    MCD::OPC_FilterValue, 5, 8, 0, 0, // Skip to: 1805
/* 1797 */    MCD::OPC_TryDecode, 158, 9, 215, 1, 167, 0, 0, // Opcode: VUPS_S32_S16_mv_ups_w2b, skip to: 1972
/* 1805 */    MCD::OPC_FilterValue, 6, 22, 0, 0, // Skip to: 1832
/* 1810 */    MCD::OPC_CheckField, 17, 1, 0, 155, 0, 0, // Skip to: 1972
/* 1817 */    MCD::OPC_CheckField, 10, 1, 0, 148, 0, 0, // Skip to: 1972
/* 1824 */    MCD::OPC_TryDecode, 159, 9, 216, 1, 140, 0, 0, // Opcode: VUPS_S32_S16_mv_ups_x2c, skip to: 1972
/* 1832 */    MCD::OPC_FilterValue, 7, 15, 0, 0, // Skip to: 1852
/* 1837 */    MCD::OPC_CheckField, 17, 1, 0, 128, 0, 0, // Skip to: 1972
/* 1844 */    MCD::OPC_TryDecode, 160, 9, 217, 1, 120, 0, 0, // Opcode: VUPS_S32_S8_mv_ups_w2c, skip to: 1972
/* 1852 */    MCD::OPC_FilterValue, 9, 8, 0, 0, // Skip to: 1865
/* 1857 */    MCD::OPC_TryDecode, 162, 9, 215, 1, 107, 0, 0, // Opcode: VUPS_S64_D32_mv_ups_w2b, skip to: 1972
/* 1865 */    MCD::OPC_FilterValue, 10, 22, 0, 0, // Skip to: 1892
/* 1870 */    MCD::OPC_CheckField, 17, 1, 0, 95, 0, 0, // Skip to: 1972
/* 1877 */    MCD::OPC_CheckField, 10, 1, 0, 88, 0, 0, // Skip to: 1972
/* 1884 */    MCD::OPC_TryDecode, 163, 9, 216, 1, 80, 0, 0, // Opcode: VUPS_S64_D32_mv_ups_x2c, skip to: 1972
/* 1892 */    MCD::OPC_FilterValue, 11, 15, 0, 0, // Skip to: 1912
/* 1897 */    MCD::OPC_CheckField, 17, 1, 0, 68, 0, 0, // Skip to: 1972
/* 1904 */    MCD::OPC_TryDecode, 161, 9, 217, 1, 60, 0, 0, // Opcode: VUPS_S64_D16_mv_ups_w2c, skip to: 1972
/* 1912 */    MCD::OPC_FilterValue, 13, 8, 0, 0, // Skip to: 1925
/* 1917 */    MCD::OPC_TryDecode, 165, 9, 215, 1, 47, 0, 0, // Opcode: VUPS_S64_S32_mv_ups_w2b, skip to: 1972
/* 1925 */    MCD::OPC_FilterValue, 14, 22, 0, 0, // Skip to: 1952
/* 1930 */    MCD::OPC_CheckField, 17, 1, 0, 35, 0, 0, // Skip to: 1972
/* 1937 */    MCD::OPC_CheckField, 10, 1, 0, 28, 0, 0, // Skip to: 1972
/* 1944 */    MCD::OPC_TryDecode, 166, 9, 216, 1, 20, 0, 0, // Opcode: VUPS_S64_S32_mv_ups_x2c, skip to: 1972
/* 1952 */    MCD::OPC_FilterValue, 15, 15, 0, 0, // Skip to: 1972
/* 1957 */    MCD::OPC_CheckField, 17, 1, 0, 8, 0, 0, // Skip to: 1972
/* 1964 */    MCD::OPC_TryDecode, 164, 9, 217, 1, 0, 0, 0, // Opcode: VUPS_S64_S16_mv_ups_w2c, skip to: 1972
/* 1972 */    MCD::OPC_ExtractField, 10, 5,  // Inst{14-10} ...
/* 1975 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 1995
/* 1980 */    MCD::OPC_CheckField, 15, 3, 0, 188, 1, 0, // Skip to: 2431
/* 1987 */    MCD::OPC_TryDecode, 136, 6, 185, 1, 180, 1, 0, // Opcode: VADD_32, skip to: 2431
/* 1995 */    MCD::OPC_FilterValue, 2, 15, 0, 0, // Skip to: 2015
/* 2000 */    MCD::OPC_CheckField, 15, 3, 0, 168, 1, 0, // Skip to: 2431
/* 2007 */    MCD::OPC_TryDecode, 136, 9, 185, 1, 160, 1, 0, // Opcode: VSUB_32, skip to: 2431
/* 2015 */    MCD::OPC_FilterValue, 4, 8, 0, 0, // Skip to: 2028
/* 2020 */    MCD::OPC_TryDecode, 136, 8, 193, 1, 147, 1, 0, // Opcode: VSEL_32, skip to: 2431
/* 2028 */    MCD::OPC_FilterValue, 5, 8, 0, 0, // Skip to: 2041
/* 2033 */    MCD::OPC_TryDecode, 133, 6, 193, 1, 134, 1, 0, // Opcode: VADDSUB_32, skip to: 2431
/* 2041 */    MCD::OPC_FilterValue, 8, 8, 0, 0, // Skip to: 2054
/* 2046 */    MCD::OPC_TryDecode, 191, 7, 194, 1, 121, 1, 0, // Opcode: VMAXDIFF_LT_D32, skip to: 2431
/* 2054 */    MCD::OPC_FilterValue, 9, 8, 0, 0, // Skip to: 2067
/* 2059 */    MCD::OPC_TryDecode, 194, 7, 194, 1, 108, 1, 0, // Opcode: VMAXDIFF_LT_S32, skip to: 2431
/* 2067 */    MCD::OPC_FilterValue, 10, 8, 0, 0, // Skip to: 2080
/* 2072 */    MCD::OPC_TryDecode, 146, 9, 194, 1, 95, 1, 0, // Opcode: VSUB_LT_D32, skip to: 2431
/* 2080 */    MCD::OPC_FilterValue, 11, 8, 0, 0, // Skip to: 2093
/* 2085 */    MCD::OPC_TryDecode, 149, 9, 194, 1, 82, 1, 0, // Opcode: VSUB_LT_S32, skip to: 2431
/* 2093 */    MCD::OPC_FilterValue, 12, 8, 0, 0, // Skip to: 2106
/* 2098 */    MCD::OPC_TryDecode, 140, 9, 194, 1, 69, 1, 0, // Opcode: VSUB_GE_D32, skip to: 2431
/* 2106 */    MCD::OPC_FilterValue, 13, 8, 0, 0, // Skip to: 2119
/* 2111 */    MCD::OPC_TryDecode, 143, 9, 194, 1, 56, 1, 0, // Opcode: VSUB_GE_S32, skip to: 2431
/* 2119 */    MCD::OPC_FilterValue, 16, 8, 0, 0, // Skip to: 2132
/* 2124 */    MCD::OPC_TryDecode, 205, 7, 194, 1, 43, 1, 0, // Opcode: VMIN_GE_D32, skip to: 2431
/* 2132 */    MCD::OPC_FilterValue, 17, 8, 0, 0, // Skip to: 2145
/* 2137 */    MCD::OPC_TryDecode, 208, 7, 194, 1, 30, 1, 0, // Opcode: VMIN_GE_S32, skip to: 2431
/* 2145 */    MCD::OPC_FilterValue, 19, 15, 0, 0, // Skip to: 2165
/* 2150 */    MCD::OPC_CheckField, 15, 3, 0, 18, 1, 0, // Skip to: 2431
/* 2157 */    MCD::OPC_TryDecode, 139, 6, 185, 1, 10, 1, 0, // Opcode: VBAND, skip to: 2431
/* 2165 */    MCD::OPC_FilterValue, 20, 8, 0, 0, // Skip to: 2178
/* 2170 */    MCD::OPC_TryDecode, 198, 7, 194, 1, 253, 0, 0, // Opcode: VMAX_LT_D32, skip to: 2431
/* 2178 */    MCD::OPC_FilterValue, 21, 8, 0, 0, // Skip to: 2191
/* 2183 */    MCD::OPC_TryDecode, 201, 7, 194, 1, 240, 0, 0, // Opcode: VMAX_LT_S32, skip to: 2431
/* 2191 */    MCD::OPC_FilterValue, 23, 15, 0, 0, // Skip to: 2211
/* 2196 */    MCD::OPC_CheckField, 15, 3, 0, 228, 0, 0, // Skip to: 2431
/* 2203 */    MCD::OPC_TryDecode, 151, 6, 185, 1, 220, 0, 0, // Opcode: VBOR, skip to: 2431
/* 2211 */    MCD::OPC_FilterValue, 24, 15, 0, 0, // Skip to: 2231
/* 2216 */    MCD::OPC_CheckField, 18, 4, 0, 208, 0, 0, // Skip to: 2431
/* 2223 */    MCD::OPC_TryDecode, 179, 7, 195, 1, 200, 0, 0, // Opcode: VLT_D32, skip to: 2431
/* 2231 */    MCD::OPC_FilterValue, 25, 15, 0, 0, // Skip to: 2251
/* 2236 */    MCD::OPC_CheckField, 18, 4, 0, 188, 0, 0, // Skip to: 2431
/* 2243 */    MCD::OPC_TryDecode, 182, 7, 195, 1, 180, 0, 0, // Opcode: VLT_S32, skip to: 2431
/* 2251 */    MCD::OPC_FilterValue, 28, 15, 0, 0, // Skip to: 2271
/* 2256 */    MCD::OPC_CheckField, 18, 4, 0, 168, 0, 0, // Skip to: 2431
/* 2263 */    MCD::OPC_TryDecode, 179, 6, 195, 1, 160, 0, 0, // Opcode: VGE_D32, skip to: 2431
/* 2271 */    MCD::OPC_FilterValue, 29, 155, 0, 0, // Skip to: 2431
/* 2276 */    MCD::OPC_CheckField, 18, 4, 0, 148, 0, 0, // Skip to: 2431
/* 2283 */    MCD::OPC_TryDecode, 182, 6, 195, 1, 140, 0, 0, // Opcode: VGE_S32, skip to: 2431
/* 2291 */    MCD::OPC_FilterValue, 3, 135, 0, 0, // Skip to: 2431
/* 2296 */    MCD::OPC_CheckField, 4, 2, 3, 8, 0, 0, // Skip to: 2311
/* 2303 */    MCD::OPC_TryDecode, 129, 4, 218, 1, 0, 0, 0, // Opcode: ADD_NC, skip to: 2311
/* 2311 */    MCD::OPC_ExtractField, 8, 2,  // Inst{9-8} ...
/* 2314 */    MCD::OPC_FilterValue, 3, 55, 0, 0, // Skip to: 2374
/* 2319 */    MCD::OPC_ExtractField, 16, 2,  // Inst{17-16} ...
/* 2322 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 2335
/* 2327 */    MCD::OPC_TryDecode, 134, 8, 219, 1, 39, 0, 0, // Opcode: VPUSH_LO_8, skip to: 2374
/* 2335 */    MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 2348
/* 2340 */    MCD::OPC_TryDecode, 131, 8, 219, 1, 26, 0, 0, // Opcode: VPUSH_LO_16, skip to: 2374
/* 2348 */    MCD::OPC_FilterValue, 2, 8, 0, 0, // Skip to: 2361
/* 2353 */    MCD::OPC_TryDecode, 132, 8, 219, 1, 13, 0, 0, // Opcode: VPUSH_LO_32, skip to: 2374
/* 2361 */    MCD::OPC_FilterValue, 3, 8, 0, 0, // Skip to: 2374
/* 2366 */    MCD::OPC_TryDecode, 133, 8, 220, 1, 0, 0, 0, // Opcode: VPUSH_LO_64, skip to: 2374
/* 2374 */    MCD::OPC_ExtractField, 15, 1,  // Inst{15} ...
/* 2377 */    MCD::OPC_FilterValue, 0, 36, 0, 0, // Skip to: 2418
/* 2382 */    MCD::OPC_ExtractField, 16, 1,  // Inst{16} ...
/* 2385 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 2398
/* 2390 */    MCD::OPC_TryDecode, 140, 8, 221, 1, 33, 0, 0, // Opcode: VSHUFFLE, skip to: 2431
/* 2398 */    MCD::OPC_FilterValue, 1, 28, 0, 0, // Skip to: 2431
/* 2403 */    MCD::OPC_CheckField, 17, 1, 0, 21, 0, 0, // Skip to: 2431
/* 2410 */    MCD::OPC_TryDecode, 138, 8, 222, 1, 13, 0, 0, // Opcode: VSHIFT, skip to: 2431
/* 2418 */    MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 2431
/* 2423 */    MCD::OPC_TryDecode, 139, 8, 223, 1, 0, 0, 0, // Opcode: VSHIFT_ALIGN, skip to: 2431
/* 2431 */    MCD::OPC_Fail,
  0
};

static const uint8_t DecoderTableNop16[] = {
/* 0 */       MCD::OPC_CheckField, 0, 1, 0, 4, 0, 0, // Skip to: 11
/* 7 */       MCD::OPC_Decode, 182, 5, 0, // Opcode: NOP
/* 11 */      MCD::OPC_Fail,
  0
};

static const uint8_t DecoderTableSt32[] = {
/* 0 */       MCD::OPC_ExtractField, 0, 1,  // Inst{0} ...
/* 3 */       MCD::OPC_FilterValue, 0, 233, 9, 0, // Skip to: 2545
/* 8 */       MCD::OPC_ExtractField, 8, 1,  // Inst{8} ...
/* 11 */      MCD::OPC_FilterValue, 0, 212, 6, 0, // Skip to: 1764
/* 16 */      MCD::OPC_ExtractField, 9, 1,  // Inst{9} ...
/* 19 */      MCD::OPC_FilterValue, 0, 194, 4, 0, // Skip to: 1242
/* 24 */      MCD::OPC_ExtractField, 10, 11,  // Inst{20-10} ...
/* 27 */      MCD::OPC_FilterValue, 0, 11, 0, 0, // Skip to: 43
/* 32 */      MCD::OPC_CheckField, 1, 7, 0, 106, 0, 0, // Skip to: 145
/* 39 */      MCD::OPC_Decode, 186, 5, 0, // Opcode: NOPS
/* 43 */      MCD::OPC_FilterValue, 30, 8, 0, 0, // Skip to: 56
/* 48 */      MCD::OPC_TryDecode, 176, 5, 224, 1, 89, 0, 0, // Opcode: MOV_mv_scl2ms, skip to: 145
/* 56 */      MCD::OPC_FilterValue, 124, 14, 0, 0, // Skip to: 75
/* 61 */      MCD::OPC_CheckField, 1, 1, 1, 77, 0, 0, // Skip to: 145
/* 68 */      MCD::OPC_TryDecode, 214, 7, 125, 70, 0, 0, // Opcode: VMOV_mv_mcd, skip to: 145
/* 75 */      MCD::OPC_FilterValue, 158, 2, 8, 0, 0, // Skip to: 89
/* 81 */      MCD::OPC_TryDecode, 169, 5, 224, 1, 56, 0, 0, // Opcode: MOV_TLAST_mv_scl2ms, skip to: 145
/* 89 */      MCD::OPC_FilterValue, 158, 4, 8, 0, 0, // Skip to: 103
/* 95 */      MCD::OPC_TryDecode, 177, 5, 225, 1, 42, 0, 0, // Opcode: MOV_mv_scl2ms_doTlast_reg, skip to: 145
/* 103 */     MCD::OPC_FilterValue, 158, 8, 8, 0, 0, // Skip to: 117
/* 109 */     MCD::OPC_TryDecode, 163, 5, 224, 1, 28, 0, 0, // Opcode: MOV_NB_mv_scl2ms, skip to: 145
/* 117 */     MCD::OPC_FilterValue, 158, 10, 8, 0, 0, // Skip to: 131
/* 123 */     MCD::OPC_TryDecode, 158, 5, 224, 1, 14, 0, 0, // Opcode: MOV_NB_TLAST_mv_scl2ms, skip to: 145
/* 131 */     MCD::OPC_FilterValue, 158, 12, 8, 0, 0, // Skip to: 145
/* 137 */     MCD::OPC_TryDecode, 164, 5, 225, 1, 0, 0, 0, // Opcode: MOV_NB_mv_scl2ms_doTlast_reg, skip to: 145
/* 145 */     MCD::OPC_ExtractField, 1, 3,  // Inst{3-1} ...
/* 148 */     MCD::OPC_FilterValue, 1, 153, 0, 0, // Skip to: 306
/* 153 */     MCD::OPC_ExtractField, 12, 2,  // Inst{13-12} ...
/* 156 */     MCD::OPC_FilterValue, 0, 36, 0, 0, // Skip to: 197
/* 161 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 164 */     MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 184
/* 169 */     MCD::OPC_CheckField, 17, 1, 0, 142, 2, 0, // Skip to: 830
/* 176 */     MCD::OPC_TryDecode, 185, 8, 226, 1, 134, 2, 0, // Opcode: VST_3D_SRS_D8_S32, skip to: 830
/* 184 */     MCD::OPC_FilterValue, 1, 129, 2, 0, // Skip to: 830
/* 189 */     MCD::OPC_TryDecode, 226, 8, 227, 1, 121, 2, 0, // Opcode: VST_SRS_D8_S32_ag_idx, skip to: 830
/* 197 */     MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 210
/* 202 */     MCD::OPC_TryDecode, 229, 8, 228, 1, 108, 2, 0, // Opcode: VST_SRS_D8_S32_ag_pstm_nrm_imm, skip to: 830
/* 210 */     MCD::OPC_FilterValue, 2, 29, 0, 0, // Skip to: 244
/* 215 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 218 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 231
/* 223 */     MCD::OPC_TryDecode, 228, 8, 229, 1, 87, 2, 0, // Opcode: VST_SRS_D8_S32_ag_pstm_nrm, skip to: 830
/* 231 */     MCD::OPC_FilterValue, 1, 82, 2, 0, // Skip to: 830
/* 236 */     MCD::OPC_TryDecode, 170, 8, 230, 1, 74, 2, 0, // Opcode: VST_2D_SRS_D8_S32, skip to: 830
/* 244 */     MCD::OPC_FilterValue, 3, 69, 2, 0, // Skip to: 830
/* 249 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 252 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 265
/* 257 */     MCD::OPC_TryDecode, 227, 8, 231, 1, 53, 2, 0, // Opcode: VST_SRS_D8_S32_ag_idx_imm, skip to: 830
/* 265 */     MCD::OPC_FilterValue, 1, 48, 2, 0, // Skip to: 830
/* 270 */     MCD::OPC_ExtractField, 15, 1,  // Inst{15} ...
/* 273 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 286
/* 278 */     MCD::OPC_TryDecode, 150, 8, 232, 1, 32, 2, 0, // Opcode: VSRS_D8_S32_mv_w_srs, skip to: 830
/* 286 */     MCD::OPC_FilterValue, 1, 27, 2, 0, // Skip to: 830
/* 291 */     MCD::OPC_CheckField, 16, 1, 0, 20, 2, 0, // Skip to: 830
/* 298 */     MCD::OPC_TryDecode, 146, 8, 233, 1, 12, 2, 0, // Opcode: VSRS_D16_S32_mv_x_srs, skip to: 830
/* 306 */     MCD::OPC_FilterValue, 5, 7, 2, 0, // Skip to: 830
/* 311 */     MCD::OPC_ExtractField, 10, 4,  // Inst{13-10} ...
/* 314 */     MCD::OPC_FilterValue, 0, 36, 0, 0, // Skip to: 355
/* 319 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 322 */     MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 342
/* 327 */     MCD::OPC_CheckField, 17, 1, 0, 240, 1, 0, // Skip to: 830
/* 334 */     MCD::OPC_TryDecode, 178, 8, 234, 1, 232, 1, 0, // Opcode: VST_3D_PACK_D4_D8, skip to: 830
/* 342 */     MCD::OPC_FilterValue, 1, 227, 1, 0, // Skip to: 830
/* 347 */     MCD::OPC_TryDecode, 198, 8, 235, 1, 219, 1, 0, // Opcode: VST_PACK_D4_D8_ag_idx, skip to: 830
/* 355 */     MCD::OPC_FilterValue, 1, 36, 0, 0, // Skip to: 396
/* 360 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 363 */     MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 383
/* 368 */     MCD::OPC_CheckField, 17, 1, 0, 199, 1, 0, // Skip to: 830
/* 375 */     MCD::OPC_TryDecode, 179, 8, 234, 1, 191, 1, 0, // Opcode: VST_3D_PACK_D8_D16, skip to: 830
/* 383 */     MCD::OPC_FilterValue, 1, 186, 1, 0, // Skip to: 830
/* 388 */     MCD::OPC_TryDecode, 202, 8, 235, 1, 178, 1, 0, // Opcode: VST_PACK_D8_D16_ag_idx, skip to: 830
/* 396 */     MCD::OPC_FilterValue, 2, 36, 0, 0, // Skip to: 437
/* 401 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 404 */     MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 424
/* 409 */     MCD::OPC_CheckField, 17, 1, 0, 158, 1, 0, // Skip to: 830
/* 416 */     MCD::OPC_TryDecode, 180, 8, 234, 1, 150, 1, 0, // Opcode: VST_3D_PACK_S4_S8, skip to: 830
/* 424 */     MCD::OPC_FilterValue, 1, 145, 1, 0, // Skip to: 830
/* 429 */     MCD::OPC_TryDecode, 206, 8, 235, 1, 137, 1, 0, // Opcode: VST_PACK_S4_S8_ag_idx, skip to: 830
/* 437 */     MCD::OPC_FilterValue, 3, 36, 0, 0, // Skip to: 478
/* 442 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 445 */     MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 465
/* 450 */     MCD::OPC_CheckField, 17, 1, 0, 117, 1, 0, // Skip to: 830
/* 457 */     MCD::OPC_TryDecode, 181, 8, 234, 1, 109, 1, 0, // Opcode: VST_3D_PACK_S8_S16, skip to: 830
/* 465 */     MCD::OPC_FilterValue, 1, 104, 1, 0, // Skip to: 830
/* 470 */     MCD::OPC_TryDecode, 210, 8, 235, 1, 96, 1, 0, // Opcode: VST_PACK_S8_S16_ag_idx, skip to: 830
/* 478 */     MCD::OPC_FilterValue, 4, 8, 0, 0, // Skip to: 491
/* 483 */     MCD::OPC_TryDecode, 201, 8, 236, 1, 83, 1, 0, // Opcode: VST_PACK_D4_D8_ag_pstm_nrm_imm, skip to: 830
/* 491 */     MCD::OPC_FilterValue, 5, 8, 0, 0, // Skip to: 504
/* 496 */     MCD::OPC_TryDecode, 205, 8, 236, 1, 70, 1, 0, // Opcode: VST_PACK_D8_D16_ag_pstm_nrm_imm, skip to: 830
/* 504 */     MCD::OPC_FilterValue, 6, 8, 0, 0, // Skip to: 517
/* 509 */     MCD::OPC_TryDecode, 209, 8, 236, 1, 57, 1, 0, // Opcode: VST_PACK_S4_S8_ag_pstm_nrm_imm, skip to: 830
/* 517 */     MCD::OPC_FilterValue, 7, 8, 0, 0, // Skip to: 530
/* 522 */     MCD::OPC_TryDecode, 213, 8, 236, 1, 44, 1, 0, // Opcode: VST_PACK_S8_S16_ag_pstm_nrm_imm, skip to: 830
/* 530 */     MCD::OPC_FilterValue, 8, 29, 0, 0, // Skip to: 564
/* 535 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 538 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 551
/* 543 */     MCD::OPC_TryDecode, 200, 8, 237, 1, 23, 1, 0, // Opcode: VST_PACK_D4_D8_ag_pstm_nrm, skip to: 830
/* 551 */     MCD::OPC_FilterValue, 1, 18, 1, 0, // Skip to: 830
/* 556 */     MCD::OPC_TryDecode, 163, 8, 238, 1, 10, 1, 0, // Opcode: VST_2D_PACK_D4_D8, skip to: 830
/* 564 */     MCD::OPC_FilterValue, 9, 29, 0, 0, // Skip to: 598
/* 569 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 572 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 585
/* 577 */     MCD::OPC_TryDecode, 204, 8, 237, 1, 245, 0, 0, // Opcode: VST_PACK_D8_D16_ag_pstm_nrm, skip to: 830
/* 585 */     MCD::OPC_FilterValue, 1, 240, 0, 0, // Skip to: 830
/* 590 */     MCD::OPC_TryDecode, 164, 8, 238, 1, 232, 0, 0, // Opcode: VST_2D_PACK_D8_D16, skip to: 830
/* 598 */     MCD::OPC_FilterValue, 10, 29, 0, 0, // Skip to: 632
/* 603 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 606 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 619
/* 611 */     MCD::OPC_TryDecode, 208, 8, 237, 1, 211, 0, 0, // Opcode: VST_PACK_S4_S8_ag_pstm_nrm, skip to: 830
/* 619 */     MCD::OPC_FilterValue, 1, 206, 0, 0, // Skip to: 830
/* 624 */     MCD::OPC_TryDecode, 165, 8, 238, 1, 198, 0, 0, // Opcode: VST_2D_PACK_S4_S8, skip to: 830
/* 632 */     MCD::OPC_FilterValue, 11, 29, 0, 0, // Skip to: 666
/* 637 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 640 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 653
/* 645 */     MCD::OPC_TryDecode, 212, 8, 237, 1, 177, 0, 0, // Opcode: VST_PACK_S8_S16_ag_pstm_nrm, skip to: 830
/* 653 */     MCD::OPC_FilterValue, 1, 172, 0, 0, // Skip to: 830
/* 658 */     MCD::OPC_TryDecode, 166, 8, 238, 1, 164, 0, 0, // Opcode: VST_2D_PACK_S8_S16, skip to: 830
/* 666 */     MCD::OPC_FilterValue, 12, 36, 0, 0, // Skip to: 707
/* 671 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 674 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 687
/* 679 */     MCD::OPC_TryDecode, 199, 8, 239, 1, 143, 0, 0, // Opcode: VST_PACK_D4_D8_ag_idx_imm, skip to: 830
/* 687 */     MCD::OPC_FilterValue, 1, 138, 0, 0, // Skip to: 830
/* 692 */     MCD::OPC_CheckField, 15, 1, 1, 131, 0, 0, // Skip to: 830
/* 699 */     MCD::OPC_TryDecode, 251, 7, 240, 1, 123, 0, 0, // Opcode: VPACK_D4_D8, skip to: 830
/* 707 */     MCD::OPC_FilterValue, 13, 36, 0, 0, // Skip to: 748
/* 712 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 715 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 728
/* 720 */     MCD::OPC_TryDecode, 203, 8, 239, 1, 102, 0, 0, // Opcode: VST_PACK_D8_D16_ag_idx_imm, skip to: 830
/* 728 */     MCD::OPC_FilterValue, 1, 97, 0, 0, // Skip to: 830
/* 733 */     MCD::OPC_CheckField, 15, 1, 1, 90, 0, 0, // Skip to: 830
/* 740 */     MCD::OPC_TryDecode, 252, 7, 240, 1, 82, 0, 0, // Opcode: VPACK_D8_D16, skip to: 830
/* 748 */     MCD::OPC_FilterValue, 14, 36, 0, 0, // Skip to: 789
/* 753 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 756 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 769
/* 761 */     MCD::OPC_TryDecode, 207, 8, 239, 1, 61, 0, 0, // Opcode: VST_PACK_S4_S8_ag_idx_imm, skip to: 830
/* 769 */     MCD::OPC_FilterValue, 1, 56, 0, 0, // Skip to: 830
/* 774 */     MCD::OPC_CheckField, 15, 1, 1, 49, 0, 0, // Skip to: 830
/* 781 */     MCD::OPC_TryDecode, 253, 7, 240, 1, 41, 0, 0, // Opcode: VPACK_S4_S8, skip to: 830
/* 789 */     MCD::OPC_FilterValue, 15, 36, 0, 0, // Skip to: 830
/* 794 */     MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 797 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 810
/* 802 */     MCD::OPC_TryDecode, 211, 8, 239, 1, 20, 0, 0, // Opcode: VST_PACK_S8_S16_ag_idx_imm, skip to: 830
/* 810 */     MCD::OPC_FilterValue, 1, 15, 0, 0, // Skip to: 830
/* 815 */     MCD::OPC_CheckField, 15, 1, 1, 8, 0, 0, // Skip to: 830
/* 822 */     MCD::OPC_TryDecode, 254, 7, 240, 1, 0, 0, 0, // Opcode: VPACK_S8_S16, skip to: 830
/* 830 */     MCD::OPC_ExtractField, 1, 2,  // Inst{2-1} ...
/* 833 */     MCD::OPC_FilterValue, 0, 123, 0, 0, // Skip to: 961
/* 838 */     MCD::OPC_ExtractField, 18, 3,  // Inst{20-18} ...
/* 841 */     MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 861
/* 846 */     MCD::OPC_CheckField, 10, 1, 1, 117, 1, 0, // Skip to: 1226
/* 853 */     MCD::OPC_TryDecode, 171, 5, 241, 1, 109, 1, 0, // Opcode: MOV_mv_cph2ms, skip to: 1226
/* 861 */     MCD::OPC_FilterValue, 1, 15, 0, 0, // Skip to: 881
/* 866 */     MCD::OPC_CheckField, 10, 1, 1, 97, 1, 0, // Skip to: 1226
/* 873 */     MCD::OPC_TryDecode, 167, 5, 241, 1, 89, 1, 0, // Opcode: MOV_TLAST_mv_cph2ms, skip to: 1226
/* 881 */     MCD::OPC_FilterValue, 2, 15, 0, 0, // Skip to: 901
/* 886 */     MCD::OPC_CheckField, 10, 1, 1, 77, 1, 0, // Skip to: 1226
/* 893 */     MCD::OPC_TryDecode, 172, 5, 242, 1, 69, 1, 0, // Opcode: MOV_mv_cph2ms_doTlast_reg, skip to: 1226
/* 901 */     MCD::OPC_FilterValue, 4, 15, 0, 0, // Skip to: 921
/* 906 */     MCD::OPC_CheckField, 10, 1, 1, 57, 1, 0, // Skip to: 1226
/* 913 */     MCD::OPC_TryDecode, 159, 5, 241, 1, 49, 1, 0, // Opcode: MOV_NB_mv_cph2ms, skip to: 1226
/* 921 */     MCD::OPC_FilterValue, 5, 15, 0, 0, // Skip to: 941
/* 926 */     MCD::OPC_CheckField, 10, 1, 1, 37, 1, 0, // Skip to: 1226
/* 933 */     MCD::OPC_TryDecode, 156, 5, 241, 1, 29, 1, 0, // Opcode: MOV_NB_TLAST_mv_cph2ms, skip to: 1226
/* 941 */     MCD::OPC_FilterValue, 6, 24, 1, 0, // Skip to: 1226
/* 946 */     MCD::OPC_CheckField, 10, 1, 1, 17, 1, 0, // Skip to: 1226
/* 953 */     MCD::OPC_TryDecode, 160, 5, 242, 1, 9, 1, 0, // Opcode: MOV_NB_mv_cph2ms_doTlast_reg, skip to: 1226
/* 961 */     MCD::OPC_FilterValue, 2, 123, 0, 0, // Skip to: 1089
/* 966 */     MCD::OPC_ExtractField, 15, 6,  // Inst{20-15} ...
/* 969 */     MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 989
/* 974 */     MCD::OPC_CheckField, 10, 2, 1, 245, 0, 0, // Skip to: 1226
/* 981 */     MCD::OPC_TryDecode, 173, 5, 243, 1, 237, 0, 0, // Opcode: MOV_mv_ph2ms, skip to: 1226
/* 989 */     MCD::OPC_FilterValue, 8, 15, 0, 0, // Skip to: 1009
/* 994 */     MCD::OPC_CheckField, 10, 2, 1, 225, 0, 0, // Skip to: 1226
/* 1001 */    MCD::OPC_TryDecode, 168, 5, 243, 1, 217, 0, 0, // Opcode: MOV_TLAST_mv_ph2ms, skip to: 1226
/* 1009 */    MCD::OPC_FilterValue, 16, 15, 0, 0, // Skip to: 1029
/* 1014 */    MCD::OPC_CheckField, 10, 2, 1, 205, 0, 0, // Skip to: 1226
/* 1021 */    MCD::OPC_TryDecode, 174, 5, 244, 1, 197, 0, 0, // Opcode: MOV_mv_ph2ms_doTlast_reg, skip to: 1226
/* 1029 */    MCD::OPC_FilterValue, 32, 15, 0, 0, // Skip to: 1049
/* 1034 */    MCD::OPC_CheckField, 10, 2, 1, 185, 0, 0, // Skip to: 1226
/* 1041 */    MCD::OPC_TryDecode, 161, 5, 243, 1, 177, 0, 0, // Opcode: MOV_NB_mv_ph2ms, skip to: 1226
/* 1049 */    MCD::OPC_FilterValue, 40, 15, 0, 0, // Skip to: 1069
/* 1054 */    MCD::OPC_CheckField, 10, 2, 1, 165, 0, 0, // Skip to: 1226
/* 1061 */    MCD::OPC_TryDecode, 157, 5, 243, 1, 157, 0, 0, // Opcode: MOV_NB_TLAST_mv_ph2ms, skip to: 1226
/* 1069 */    MCD::OPC_FilterValue, 48, 152, 0, 0, // Skip to: 1226
/* 1074 */    MCD::OPC_CheckField, 10, 2, 1, 145, 0, 0, // Skip to: 1226
/* 1081 */    MCD::OPC_TryDecode, 162, 5, 244, 1, 137, 0, 0, // Opcode: MOV_NB_mv_ph2ms_doTlast_reg, skip to: 1226
/* 1089 */    MCD::OPC_FilterValue, 3, 132, 0, 0, // Skip to: 1226
/* 1094 */    MCD::OPC_ExtractField, 12, 2,  // Inst{13-12} ...
/* 1097 */    MCD::OPC_FilterValue, 0, 36, 0, 0, // Skip to: 1138
/* 1102 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1105 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 1125
/* 1110 */    MCD::OPC_CheckField, 17, 1, 0, 109, 0, 0, // Skip to: 1226
/* 1117 */    MCD::OPC_TryDecode, 182, 8, 245, 1, 101, 0, 0, // Opcode: VST_3D_SRS_D16_S32, skip to: 1226
/* 1125 */    MCD::OPC_FilterValue, 1, 96, 0, 0, // Skip to: 1226
/* 1130 */    MCD::OPC_TryDecode, 214, 8, 246, 1, 88, 0, 0, // Opcode: VST_SRS_D16_S32_ag_idx, skip to: 1226
/* 1138 */    MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 1151
/* 1143 */    MCD::OPC_TryDecode, 217, 8, 247, 1, 75, 0, 0, // Opcode: VST_SRS_D16_S32_ag_pstm_nrm_imm, skip to: 1226
/* 1151 */    MCD::OPC_FilterValue, 2, 29, 0, 0, // Skip to: 1185
/* 1156 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1159 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 1172
/* 1164 */    MCD::OPC_TryDecode, 216, 8, 248, 1, 54, 0, 0, // Opcode: VST_SRS_D16_S32_ag_pstm_nrm, skip to: 1226
/* 1172 */    MCD::OPC_FilterValue, 1, 49, 0, 0, // Skip to: 1226
/* 1177 */    MCD::OPC_TryDecode, 167, 8, 249, 1, 41, 0, 0, // Opcode: VST_2D_SRS_D16_S32, skip to: 1226
/* 1185 */    MCD::OPC_FilterValue, 3, 36, 0, 0, // Skip to: 1226
/* 1190 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1193 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 1206
/* 1198 */    MCD::OPC_TryDecode, 215, 8, 250, 1, 20, 0, 0, // Opcode: VST_SRS_D16_S32_ag_idx_imm, skip to: 1226
/* 1206 */    MCD::OPC_FilterValue, 1, 15, 0, 0, // Skip to: 1226
/* 1211 */    MCD::OPC_CheckField, 15, 1, 0, 8, 0, 0, // Skip to: 1226
/* 1218 */    MCD::OPC_TryDecode, 145, 8, 251, 1, 0, 0, 0, // Opcode: VSRS_D16_S32_mv_w_srs, skip to: 1226
/* 1226 */    MCD::OPC_CheckField, 10, 8, 158, 1, 197, 7, 0, // Skip to: 3223
/* 1234 */    MCD::OPC_TryDecode, 229, 5, 252, 1, 189, 7, 0, // Opcode: ST_TM, skip to: 3223
/* 1242 */    MCD::OPC_FilterValue, 1, 184, 7, 0, // Skip to: 3223
/* 1247 */    MCD::OPC_ExtractField, 1, 5,  // Inst{5-1} ...
/* 1250 */    MCD::OPC_FilterValue, 21, 104, 0, 0, // Skip to: 1359
/* 1255 */    MCD::OPC_ExtractField, 10, 1,  // Inst{10} ...
/* 1258 */    MCD::OPC_FilterValue, 0, 83, 0, 0, // Skip to: 1346
/* 1263 */    MCD::OPC_ExtractField, 11, 1,  // Inst{11} ...
/* 1266 */    MCD::OPC_FilterValue, 0, 62, 0, 0, // Skip to: 1333
/* 1271 */    MCD::OPC_ExtractField, 12, 3,  // Inst{14-12} ...
/* 1274 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 1294
/* 1279 */    MCD::OPC_CheckField, 17, 1, 0, 73, 0, 0, // Skip to: 1359
/* 1286 */    MCD::OPC_TryDecode, 220, 5, 253, 1, 65, 0, 0, // Opcode: ST_3D_dmv_sts_q, skip to: 1359
/* 1294 */    MCD::OPC_FilterValue, 2, 8, 0, 0, // Skip to: 1307
/* 1299 */    MCD::OPC_TryDecode, 237, 5, 254, 1, 52, 0, 0, // Opcode: ST_dmv_sts_q_ag_pstm_nrm, skip to: 1359
/* 1307 */    MCD::OPC_FilterValue, 4, 8, 0, 0, // Skip to: 1320
/* 1312 */    MCD::OPC_TryDecode, 235, 5, 145, 1, 39, 0, 0, // Opcode: ST_dmv_sts_q_ag_idx, skip to: 1359
/* 1320 */    MCD::OPC_FilterValue, 6, 34, 0, 0, // Skip to: 1359
/* 1325 */    MCD::OPC_TryDecode, 216, 5, 255, 1, 26, 0, 0, // Opcode: ST_2D_dmv_sts_q, skip to: 1359
/* 1333 */    MCD::OPC_FilterValue, 1, 21, 0, 0, // Skip to: 1359
/* 1338 */    MCD::OPC_TryDecode, 236, 5, 151, 1, 13, 0, 0, // Opcode: ST_dmv_sts_q_ag_idx_imm, skip to: 1359
/* 1346 */    MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 1359
/* 1351 */    MCD::OPC_TryDecode, 238, 5, 128, 2, 0, 0, 0, // Opcode: ST_dmv_sts_q_ag_pstm_nrm_imm, skip to: 1359
/* 1359 */    MCD::OPC_ExtractField, 1, 3,  // Inst{3-1} ...
/* 1362 */    MCD::OPC_FilterValue, 1, 153, 0, 0, // Skip to: 1520
/* 1367 */    MCD::OPC_ExtractField, 12, 2,  // Inst{13-12} ...
/* 1370 */    MCD::OPC_FilterValue, 0, 36, 0, 0, // Skip to: 1411
/* 1375 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1378 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 1398
/* 1383 */    MCD::OPC_CheckField, 17, 1, 0, 130, 0, 0, // Skip to: 1520
/* 1390 */    MCD::OPC_TryDecode, 183, 8, 226, 1, 122, 0, 0, // Opcode: VST_3D_SRS_D16_S64, skip to: 1520
/* 1398 */    MCD::OPC_FilterValue, 1, 117, 0, 0, // Skip to: 1520
/* 1403 */    MCD::OPC_TryDecode, 218, 8, 227, 1, 109, 0, 0, // Opcode: VST_SRS_D16_S64_ag_idx, skip to: 1520
/* 1411 */    MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 1424
/* 1416 */    MCD::OPC_TryDecode, 221, 8, 228, 1, 96, 0, 0, // Opcode: VST_SRS_D16_S64_ag_pstm_nrm_imm, skip to: 1520
/* 1424 */    MCD::OPC_FilterValue, 2, 29, 0, 0, // Skip to: 1458
/* 1429 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1432 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 1445
/* 1437 */    MCD::OPC_TryDecode, 220, 8, 229, 1, 75, 0, 0, // Opcode: VST_SRS_D16_S64_ag_pstm_nrm, skip to: 1520
/* 1445 */    MCD::OPC_FilterValue, 1, 70, 0, 0, // Skip to: 1520
/* 1450 */    MCD::OPC_TryDecode, 168, 8, 230, 1, 62, 0, 0, // Opcode: VST_2D_SRS_D16_S64, skip to: 1520
/* 1458 */    MCD::OPC_FilterValue, 3, 57, 0, 0, // Skip to: 1520
/* 1463 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1466 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 1479
/* 1471 */    MCD::OPC_TryDecode, 219, 8, 231, 1, 41, 0, 0, // Opcode: VST_SRS_D16_S64_ag_idx_imm, skip to: 1520
/* 1479 */    MCD::OPC_FilterValue, 1, 36, 0, 0, // Skip to: 1520
/* 1484 */    MCD::OPC_ExtractField, 15, 1,  // Inst{15} ...
/* 1487 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 1500
/* 1492 */    MCD::OPC_TryDecode, 147, 8, 232, 1, 20, 0, 0, // Opcode: VSRS_D16_S64_mv_w_srs, skip to: 1520
/* 1500 */    MCD::OPC_FilterValue, 1, 15, 0, 0, // Skip to: 1520
/* 1505 */    MCD::OPC_CheckField, 16, 1, 0, 8, 0, 0, // Skip to: 1520
/* 1512 */    MCD::OPC_TryDecode, 149, 8, 233, 1, 0, 0, 0, // Opcode: VSRS_D32_S64_mv_x_srs, skip to: 1520
/* 1520 */    MCD::OPC_ExtractField, 1, 2,  // Inst{2-1} ...
/* 1523 */    MCD::OPC_FilterValue, 3, 132, 0, 0, // Skip to: 1660
/* 1528 */    MCD::OPC_ExtractField, 12, 2,  // Inst{13-12} ...
/* 1531 */    MCD::OPC_FilterValue, 0, 36, 0, 0, // Skip to: 1572
/* 1536 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1539 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 1559
/* 1544 */    MCD::OPC_CheckField, 17, 1, 0, 109, 0, 0, // Skip to: 1660
/* 1551 */    MCD::OPC_TryDecode, 184, 8, 245, 1, 101, 0, 0, // Opcode: VST_3D_SRS_D32_S64, skip to: 1660
/* 1559 */    MCD::OPC_FilterValue, 1, 96, 0, 0, // Skip to: 1660
/* 1564 */    MCD::OPC_TryDecode, 222, 8, 246, 1, 88, 0, 0, // Opcode: VST_SRS_D32_S64_ag_idx, skip to: 1660
/* 1572 */    MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 1585
/* 1577 */    MCD::OPC_TryDecode, 225, 8, 247, 1, 75, 0, 0, // Opcode: VST_SRS_D32_S64_ag_pstm_nrm_imm, skip to: 1660
/* 1585 */    MCD::OPC_FilterValue, 2, 29, 0, 0, // Skip to: 1619
/* 1590 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1593 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 1606
/* 1598 */    MCD::OPC_TryDecode, 224, 8, 248, 1, 54, 0, 0, // Opcode: VST_SRS_D32_S64_ag_pstm_nrm, skip to: 1660
/* 1606 */    MCD::OPC_FilterValue, 1, 49, 0, 0, // Skip to: 1660
/* 1611 */    MCD::OPC_TryDecode, 169, 8, 249, 1, 41, 0, 0, // Opcode: VST_2D_SRS_D32_S64, skip to: 1660
/* 1619 */    MCD::OPC_FilterValue, 3, 36, 0, 0, // Skip to: 1660
/* 1624 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1627 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 1640
/* 1632 */    MCD::OPC_TryDecode, 223, 8, 250, 1, 20, 0, 0, // Opcode: VST_SRS_D32_S64_ag_idx_imm, skip to: 1660
/* 1640 */    MCD::OPC_FilterValue, 1, 15, 0, 0, // Skip to: 1660
/* 1645 */    MCD::OPC_CheckField, 15, 1, 0, 8, 0, 0, // Skip to: 1660
/* 1652 */    MCD::OPC_TryDecode, 148, 8, 251, 1, 0, 0, 0, // Opcode: VSRS_D32_S64_mv_w_srs, skip to: 1660
/* 1660 */    MCD::OPC_ExtractField, 10, 1,  // Inst{10} ...
/* 1663 */    MCD::OPC_FilterValue, 0, 83, 0, 0, // Skip to: 1751
/* 1668 */    MCD::OPC_ExtractField, 11, 1,  // Inst{11} ...
/* 1671 */    MCD::OPC_FilterValue, 0, 62, 0, 0, // Skip to: 1738
/* 1676 */    MCD::OPC_ExtractField, 12, 3,  // Inst{14-12} ...
/* 1679 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 1699
/* 1684 */    MCD::OPC_CheckField, 17, 1, 0, 252, 5, 0, // Skip to: 3223
/* 1691 */    MCD::OPC_TryDecode, 219, 5, 129, 2, 244, 5, 0, // Opcode: ST_3D_dms_sts, skip to: 3223
/* 1699 */    MCD::OPC_FilterValue, 2, 8, 0, 0, // Skip to: 1712
/* 1704 */    MCD::OPC_TryDecode, 233, 5, 130, 2, 231, 5, 0, // Opcode: ST_dms_sts_pstm_nrm, skip to: 3223
/* 1712 */    MCD::OPC_FilterValue, 4, 8, 0, 0, // Skip to: 1725
/* 1717 */    MCD::OPC_TryDecode, 231, 5, 131, 2, 218, 5, 0, // Opcode: ST_dms_sts_idx, skip to: 3223
/* 1725 */    MCD::OPC_FilterValue, 6, 213, 5, 0, // Skip to: 3223
/* 1730 */    MCD::OPC_TryDecode, 215, 5, 132, 2, 205, 5, 0, // Opcode: ST_2D_dms_sts, skip to: 3223
/* 1738 */    MCD::OPC_FilterValue, 1, 200, 5, 0, // Skip to: 3223
/* 1743 */    MCD::OPC_TryDecode, 232, 5, 133, 2, 192, 5, 0, // Opcode: ST_dms_sts_idx_imm, skip to: 3223
/* 1751 */    MCD::OPC_FilterValue, 1, 187, 5, 0, // Skip to: 3223
/* 1756 */    MCD::OPC_TryDecode, 234, 5, 134, 2, 179, 5, 0, // Opcode: ST_dms_sts_pstm_nrm_imm, skip to: 3223
/* 1764 */    MCD::OPC_FilterValue, 1, 174, 5, 0, // Skip to: 3223
/* 1769 */    MCD::OPC_ExtractField, 9, 1,  // Inst{9} ...
/* 1772 */    MCD::OPC_FilterValue, 0, 114, 1, 0, // Skip to: 2147
/* 1777 */    MCD::OPC_ExtractField, 12, 2,  // Inst{13-12} ...
/* 1780 */    MCD::OPC_FilterValue, 0, 99, 0, 0, // Skip to: 1884
/* 1785 */    MCD::OPC_ExtractField, 1, 2,  // Inst{2-1} ...
/* 1788 */    MCD::OPC_FilterValue, 1, 50, 0, 0, // Skip to: 1843
/* 1793 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1796 */    MCD::OPC_FilterValue, 0, 22, 0, 0, // Skip to: 1823
/* 1801 */    MCD::OPC_CheckField, 17, 1, 0, 202, 2, 0, // Skip to: 2522
/* 1808 */    MCD::OPC_CheckField, 3, 1, 0, 195, 2, 0, // Skip to: 2522
/* 1815 */    MCD::OPC_TryDecode, 189, 8, 226, 1, 187, 2, 0, // Opcode: VST_3D_SRS_S8_S32, skip to: 2522
/* 1823 */    MCD::OPC_FilterValue, 1, 182, 2, 0, // Skip to: 2522
/* 1828 */    MCD::OPC_CheckField, 3, 1, 0, 175, 2, 0, // Skip to: 2522
/* 1835 */    MCD::OPC_TryDecode, 242, 8, 227, 1, 167, 2, 0, // Opcode: VST_SRS_S8_S32_ag_idx, skip to: 2522
/* 1843 */    MCD::OPC_FilterValue, 3, 162, 2, 0, // Skip to: 2522
/* 1848 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1851 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 1871
/* 1856 */    MCD::OPC_CheckField, 17, 1, 0, 147, 2, 0, // Skip to: 2522
/* 1863 */    MCD::OPC_TryDecode, 186, 8, 245, 1, 139, 2, 0, // Opcode: VST_3D_SRS_S16_S32, skip to: 2522
/* 1871 */    MCD::OPC_FilterValue, 1, 134, 2, 0, // Skip to: 2522
/* 1876 */    MCD::OPC_TryDecode, 230, 8, 246, 1, 126, 2, 0, // Opcode: VST_SRS_S16_S32_ag_idx, skip to: 2522
/* 1884 */    MCD::OPC_FilterValue, 1, 36, 0, 0, // Skip to: 1925
/* 1889 */    MCD::OPC_ExtractField, 1, 2,  // Inst{2-1} ...
/* 1892 */    MCD::OPC_FilterValue, 1, 15, 0, 0, // Skip to: 1912
/* 1897 */    MCD::OPC_CheckField, 3, 1, 0, 106, 2, 0, // Skip to: 2522
/* 1904 */    MCD::OPC_TryDecode, 245, 8, 228, 1, 98, 2, 0, // Opcode: VST_SRS_S8_S32_ag_pstm_nrm_imm, skip to: 2522
/* 1912 */    MCD::OPC_FilterValue, 3, 93, 2, 0, // Skip to: 2522
/* 1917 */    MCD::OPC_TryDecode, 233, 8, 247, 1, 85, 2, 0, // Opcode: VST_SRS_S16_S32_ag_pstm_nrm_imm, skip to: 2522
/* 1925 */    MCD::OPC_FilterValue, 2, 85, 0, 0, // Skip to: 2015
/* 1930 */    MCD::OPC_ExtractField, 1, 2,  // Inst{2-1} ...
/* 1933 */    MCD::OPC_FilterValue, 1, 43, 0, 0, // Skip to: 1981
/* 1938 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1941 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 1961
/* 1946 */    MCD::OPC_CheckField, 3, 1, 0, 57, 2, 0, // Skip to: 2522
/* 1953 */    MCD::OPC_TryDecode, 244, 8, 229, 1, 49, 2, 0, // Opcode: VST_SRS_S8_S32_ag_pstm_nrm, skip to: 2522
/* 1961 */    MCD::OPC_FilterValue, 1, 44, 2, 0, // Skip to: 2522
/* 1966 */    MCD::OPC_CheckField, 3, 1, 0, 37, 2, 0, // Skip to: 2522
/* 1973 */    MCD::OPC_TryDecode, 174, 8, 230, 1, 29, 2, 0, // Opcode: VST_2D_SRS_S8_S32, skip to: 2522
/* 1981 */    MCD::OPC_FilterValue, 3, 24, 2, 0, // Skip to: 2522
/* 1986 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 1989 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 2002
/* 1994 */    MCD::OPC_TryDecode, 232, 8, 248, 1, 8, 2, 0, // Opcode: VST_SRS_S16_S32_ag_pstm_nrm, skip to: 2522
/* 2002 */    MCD::OPC_FilterValue, 1, 3, 2, 0, // Skip to: 2522
/* 2007 */    MCD::OPC_TryDecode, 171, 8, 249, 1, 251, 1, 0, // Opcode: VST_2D_SRS_S16_S32, skip to: 2522
/* 2015 */    MCD::OPC_FilterValue, 3, 246, 1, 0, // Skip to: 2522
/* 2020 */    MCD::OPC_ExtractField, 1, 2,  // Inst{2-1} ...
/* 2023 */    MCD::OPC_FilterValue, 1, 78, 0, 0, // Skip to: 2106
/* 2028 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 2031 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 2051
/* 2036 */    MCD::OPC_CheckField, 3, 1, 0, 223, 1, 0, // Skip to: 2522
/* 2043 */    MCD::OPC_TryDecode, 243, 8, 231, 1, 215, 1, 0, // Opcode: VST_SRS_S8_S32_ag_idx_imm, skip to: 2522
/* 2051 */    MCD::OPC_FilterValue, 1, 210, 1, 0, // Skip to: 2522
/* 2056 */    MCD::OPC_ExtractField, 15, 1,  // Inst{15} ...
/* 2059 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 2079
/* 2064 */    MCD::OPC_CheckField, 3, 1, 0, 195, 1, 0, // Skip to: 2522
/* 2071 */    MCD::OPC_TryDecode, 156, 8, 232, 1, 187, 1, 0, // Opcode: VSRS_S8_S32_mv_w_srs, skip to: 2522
/* 2079 */    MCD::OPC_FilterValue, 1, 182, 1, 0, // Skip to: 2522
/* 2084 */    MCD::OPC_CheckField, 16, 1, 0, 175, 1, 0, // Skip to: 2522
/* 2091 */    MCD::OPC_CheckField, 3, 1, 0, 168, 1, 0, // Skip to: 2522
/* 2098 */    MCD::OPC_TryDecode, 152, 8, 233, 1, 160, 1, 0, // Opcode: VSRS_S16_S32_mv_x_srs, skip to: 2522
/* 2106 */    MCD::OPC_FilterValue, 3, 155, 1, 0, // Skip to: 2522
/* 2111 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 2114 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 2127
/* 2119 */    MCD::OPC_TryDecode, 231, 8, 250, 1, 139, 1, 0, // Opcode: VST_SRS_S16_S32_ag_idx_imm, skip to: 2522
/* 2127 */    MCD::OPC_FilterValue, 1, 134, 1, 0, // Skip to: 2522
/* 2132 */    MCD::OPC_CheckField, 15, 1, 0, 127, 1, 0, // Skip to: 2522
/* 2139 */    MCD::OPC_TryDecode, 151, 8, 251, 1, 119, 1, 0, // Opcode: VSRS_S16_S32_mv_w_srs, skip to: 2522
/* 2147 */    MCD::OPC_FilterValue, 1, 114, 1, 0, // Skip to: 2522
/* 2152 */    MCD::OPC_ExtractField, 12, 2,  // Inst{13-12} ...
/* 2155 */    MCD::OPC_FilterValue, 0, 99, 0, 0, // Skip to: 2259
/* 2160 */    MCD::OPC_ExtractField, 1, 2,  // Inst{2-1} ...
/* 2163 */    MCD::OPC_FilterValue, 1, 50, 0, 0, // Skip to: 2218
/* 2168 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 2171 */    MCD::OPC_FilterValue, 0, 22, 0, 0, // Skip to: 2198
/* 2176 */    MCD::OPC_CheckField, 17, 1, 0, 83, 1, 0, // Skip to: 2522
/* 2183 */    MCD::OPC_CheckField, 3, 1, 0, 76, 1, 0, // Skip to: 2522
/* 2190 */    MCD::OPC_TryDecode, 187, 8, 226, 1, 68, 1, 0, // Opcode: VST_3D_SRS_S16_S64, skip to: 2522
/* 2198 */    MCD::OPC_FilterValue, 1, 63, 1, 0, // Skip to: 2522
/* 2203 */    MCD::OPC_CheckField, 3, 1, 0, 56, 1, 0, // Skip to: 2522
/* 2210 */    MCD::OPC_TryDecode, 234, 8, 227, 1, 48, 1, 0, // Opcode: VST_SRS_S16_S64_ag_idx, skip to: 2522
/* 2218 */    MCD::OPC_FilterValue, 3, 43, 1, 0, // Skip to: 2522
/* 2223 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 2226 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 2246
/* 2231 */    MCD::OPC_CheckField, 17, 1, 0, 28, 1, 0, // Skip to: 2522
/* 2238 */    MCD::OPC_TryDecode, 188, 8, 245, 1, 20, 1, 0, // Opcode: VST_3D_SRS_S32_S64, skip to: 2522
/* 2246 */    MCD::OPC_FilterValue, 1, 15, 1, 0, // Skip to: 2522
/* 2251 */    MCD::OPC_TryDecode, 238, 8, 246, 1, 7, 1, 0, // Opcode: VST_SRS_S32_S64_ag_idx, skip to: 2522
/* 2259 */    MCD::OPC_FilterValue, 1, 36, 0, 0, // Skip to: 2300
/* 2264 */    MCD::OPC_ExtractField, 1, 2,  // Inst{2-1} ...
/* 2267 */    MCD::OPC_FilterValue, 1, 15, 0, 0, // Skip to: 2287
/* 2272 */    MCD::OPC_CheckField, 3, 1, 0, 243, 0, 0, // Skip to: 2522
/* 2279 */    MCD::OPC_TryDecode, 237, 8, 228, 1, 235, 0, 0, // Opcode: VST_SRS_S16_S64_ag_pstm_nrm_imm, skip to: 2522
/* 2287 */    MCD::OPC_FilterValue, 3, 230, 0, 0, // Skip to: 2522
/* 2292 */    MCD::OPC_TryDecode, 241, 8, 247, 1, 222, 0, 0, // Opcode: VST_SRS_S32_S64_ag_pstm_nrm_imm, skip to: 2522
/* 2300 */    MCD::OPC_FilterValue, 2, 85, 0, 0, // Skip to: 2390
/* 2305 */    MCD::OPC_ExtractField, 1, 2,  // Inst{2-1} ...
/* 2308 */    MCD::OPC_FilterValue, 1, 43, 0, 0, // Skip to: 2356
/* 2313 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 2316 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 2336
/* 2321 */    MCD::OPC_CheckField, 3, 1, 0, 194, 0, 0, // Skip to: 2522
/* 2328 */    MCD::OPC_TryDecode, 236, 8, 229, 1, 186, 0, 0, // Opcode: VST_SRS_S16_S64_ag_pstm_nrm, skip to: 2522
/* 2336 */    MCD::OPC_FilterValue, 1, 181, 0, 0, // Skip to: 2522
/* 2341 */    MCD::OPC_CheckField, 3, 1, 0, 174, 0, 0, // Skip to: 2522
/* 2348 */    MCD::OPC_TryDecode, 172, 8, 230, 1, 166, 0, 0, // Opcode: VST_2D_SRS_S16_S64, skip to: 2522
/* 2356 */    MCD::OPC_FilterValue, 3, 161, 0, 0, // Skip to: 2522
/* 2361 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 2364 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 2377
/* 2369 */    MCD::OPC_TryDecode, 240, 8, 248, 1, 145, 0, 0, // Opcode: VST_SRS_S32_S64_ag_pstm_nrm, skip to: 2522
/* 2377 */    MCD::OPC_FilterValue, 1, 140, 0, 0, // Skip to: 2522
/* 2382 */    MCD::OPC_TryDecode, 173, 8, 249, 1, 132, 0, 0, // Opcode: VST_2D_SRS_S32_S64, skip to: 2522
/* 2390 */    MCD::OPC_FilterValue, 3, 127, 0, 0, // Skip to: 2522
/* 2395 */    MCD::OPC_ExtractField, 1, 2,  // Inst{2-1} ...
/* 2398 */    MCD::OPC_FilterValue, 1, 78, 0, 0, // Skip to: 2481
/* 2403 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 2406 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 2426
/* 2411 */    MCD::OPC_CheckField, 3, 1, 0, 104, 0, 0, // Skip to: 2522
/* 2418 */    MCD::OPC_TryDecode, 235, 8, 231, 1, 96, 0, 0, // Opcode: VST_SRS_S16_S64_ag_idx_imm, skip to: 2522
/* 2426 */    MCD::OPC_FilterValue, 1, 91, 0, 0, // Skip to: 2522
/* 2431 */    MCD::OPC_ExtractField, 15, 1,  // Inst{15} ...
/* 2434 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 2454
/* 2439 */    MCD::OPC_CheckField, 3, 1, 0, 76, 0, 0, // Skip to: 2522
/* 2446 */    MCD::OPC_TryDecode, 153, 8, 232, 1, 68, 0, 0, // Opcode: VSRS_S16_S64_mv_w_srs, skip to: 2522
/* 2454 */    MCD::OPC_FilterValue, 1, 63, 0, 0, // Skip to: 2522
/* 2459 */    MCD::OPC_CheckField, 16, 1, 0, 56, 0, 0, // Skip to: 2522
/* 2466 */    MCD::OPC_CheckField, 3, 1, 0, 49, 0, 0, // Skip to: 2522
/* 2473 */    MCD::OPC_TryDecode, 155, 8, 233, 1, 41, 0, 0, // Opcode: VSRS_S32_S64_mv_x_srs, skip to: 2522
/* 2481 */    MCD::OPC_FilterValue, 3, 36, 0, 0, // Skip to: 2522
/* 2486 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 2489 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 2502
/* 2494 */    MCD::OPC_TryDecode, 239, 8, 250, 1, 20, 0, 0, // Opcode: VST_SRS_S32_S64_ag_idx_imm, skip to: 2522
/* 2502 */    MCD::OPC_FilterValue, 1, 15, 0, 0, // Skip to: 2522
/* 2507 */    MCD::OPC_CheckField, 15, 1, 0, 8, 0, 0, // Skip to: 2522
/* 2514 */    MCD::OPC_TryDecode, 154, 8, 251, 1, 0, 0, 0, // Opcode: VSRS_S32_S64_mv_w_srs, skip to: 2522
/* 2522 */    MCD::OPC_CheckField, 1, 5, 21, 8, 0, 0, // Skip to: 2537
/* 2529 */    MCD::OPC_TryDecode, 239, 5, 157, 1, 0, 0, 0, // Opcode: ST_dmv_sts_q_ag_spill, skip to: 2537
/* 2537 */    MCD::OPC_TryDecode, 230, 5, 135, 2, 166, 2, 0, // Opcode: ST_dms_spill, skip to: 3223
/* 2545 */    MCD::OPC_FilterValue, 1, 161, 2, 0, // Skip to: 3223
/* 2550 */    MCD::OPC_ExtractField, 1, 1,  // Inst{1} ...
/* 2553 */    MCD::OPC_FilterValue, 0, 5, 1, 0, // Skip to: 2819
/* 2558 */    MCD::OPC_ExtractField, 2, 2,  // Inst{3-2} ...
/* 2561 */    MCD::OPC_FilterValue, 1, 67, 0, 0, // Skip to: 2633
/* 2566 */    MCD::OPC_ExtractField, 12, 3,  // Inst{14-12} ...
/* 2569 */    MCD::OPC_FilterValue, 0, 21, 0, 0, // Skip to: 2595
/* 2574 */    MCD::OPC_CheckField, 17, 1, 0, 71, 0, 0, // Skip to: 2652
/* 2581 */    MCD::OPC_CheckField, 6, 2, 3, 64, 0, 0, // Skip to: 2652
/* 2588 */    MCD::OPC_TryDecode, 202, 5, 120, 57, 0, 0, // Opcode: PADDS_3D, skip to: 2652
/* 2595 */    MCD::OPC_FilterValue, 2, 14, 0, 0, // Skip to: 2614
/* 2600 */    MCD::OPC_CheckField, 6, 2, 3, 45, 0, 0, // Skip to: 2652
/* 2607 */    MCD::OPC_TryDecode, 203, 5, 121, 38, 0, 0, // Opcode: PADDS_st_ptr_inc_idx, skip to: 2652
/* 2614 */    MCD::OPC_FilterValue, 6, 33, 0, 0, // Skip to: 2652
/* 2619 */    MCD::OPC_CheckField, 6, 2, 3, 26, 0, 0, // Skip to: 2652
/* 2626 */    MCD::OPC_TryDecode, 201, 5, 122, 19, 0, 0, // Opcode: PADDS_2D, skip to: 2652
/* 2633 */    MCD::OPC_FilterValue, 3, 14, 0, 0, // Skip to: 2652
/* 2638 */    MCD::OPC_CheckField, 6, 2, 3, 7, 0, 0, // Skip to: 2652
/* 2645 */    MCD::OPC_TryDecode, 204, 5, 124, 0, 0, 0, // Opcode: PADDS_st_ptr_inc_idx_imm, skip to: 2652
/* 2652 */    MCD::OPC_ExtractField, 8, 1,  // Inst{8} ...
/* 2655 */    MCD::OPC_FilterValue, 0, 146, 0, 0, // Skip to: 2806
/* 2660 */    MCD::OPC_ExtractField, 9, 1,  // Inst{9} ...
/* 2663 */    MCD::OPC_FilterValue, 0, 29, 0, 0, // Skip to: 2697
/* 2668 */    MCD::OPC_ExtractField, 12, 5,  // Inst{16-12} ...
/* 2671 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 2684
/* 2676 */    MCD::OPC_TryDecode, 176, 6, 136, 2, 27, 2, 0, // Opcode: VFLOOR_S32_BF16_mFl2FxSrc_W, skip to: 3223
/* 2684 */    MCD::OPC_FilterValue, 16, 22, 2, 0, // Skip to: 3223
/* 2689 */    MCD::OPC_TryDecode, 175, 6, 137, 2, 14, 2, 0, // Opcode: VFLOOR_S32_BF16_mFl2FxSrc_AM, skip to: 3223
/* 2697 */    MCD::OPC_FilterValue, 1, 9, 2, 0, // Skip to: 3223
/* 2702 */    MCD::OPC_ExtractField, 10, 1,  // Inst{10} ...
/* 2705 */    MCD::OPC_FilterValue, 0, 83, 0, 0, // Skip to: 2793
/* 2710 */    MCD::OPC_ExtractField, 11, 1,  // Inst{11} ...
/* 2713 */    MCD::OPC_FilterValue, 0, 62, 0, 0, // Skip to: 2780
/* 2718 */    MCD::OPC_ExtractField, 12, 3,  // Inst{14-12} ...
/* 2721 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 2741
/* 2726 */    MCD::OPC_CheckField, 17, 1, 0, 234, 1, 0, // Skip to: 3223
/* 2733 */    MCD::OPC_TryDecode, 190, 8, 138, 2, 226, 1, 0, // Opcode: VST_3D_dmw_sts_am, skip to: 3223
/* 2741 */    MCD::OPC_FilterValue, 2, 8, 0, 0, // Skip to: 2754
/* 2746 */    MCD::OPC_TryDecode, 248, 8, 139, 2, 213, 1, 0, // Opcode: VST_dmw_sts_am_ag_pstm_nrm, skip to: 3223
/* 2754 */    MCD::OPC_FilterValue, 4, 8, 0, 0, // Skip to: 2767
/* 2759 */    MCD::OPC_TryDecode, 246, 8, 140, 2, 200, 1, 0, // Opcode: VST_dmw_sts_am_ag_idx, skip to: 3223
/* 2767 */    MCD::OPC_FilterValue, 6, 195, 1, 0, // Skip to: 3223
/* 2772 */    MCD::OPC_TryDecode, 175, 8, 141, 2, 187, 1, 0, // Opcode: VST_2D_dmw_sts_am, skip to: 3223
/* 2780 */    MCD::OPC_FilterValue, 1, 182, 1, 0, // Skip to: 3223
/* 2785 */    MCD::OPC_TryDecode, 247, 8, 142, 2, 174, 1, 0, // Opcode: VST_dmw_sts_am_ag_idx_imm, skip to: 3223
/* 2793 */    MCD::OPC_FilterValue, 1, 169, 1, 0, // Skip to: 3223
/* 2798 */    MCD::OPC_TryDecode, 249, 8, 143, 2, 161, 1, 0, // Opcode: VST_dmw_sts_am_ag_pstm_nrm_imm, skip to: 3223
/* 2806 */    MCD::OPC_FilterValue, 1, 156, 1, 0, // Skip to: 3223
/* 2811 */    MCD::OPC_TryDecode, 250, 8, 144, 2, 148, 1, 0, // Opcode: VST_dmw_sts_am_ag_spill, skip to: 3223
/* 2819 */    MCD::OPC_FilterValue, 1, 143, 1, 0, // Skip to: 3223
/* 2824 */    MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
/* 2827 */    MCD::OPC_FilterValue, 0, 125, 0, 0, // Skip to: 2957
/* 2832 */    MCD::OPC_ExtractField, 8, 1,  // Inst{8} ...
/* 2835 */    MCD::OPC_FilterValue, 0, 104, 0, 0, // Skip to: 2944
/* 2840 */    MCD::OPC_ExtractField, 9, 2,  // Inst{10-9} ...
/* 2843 */    MCD::OPC_FilterValue, 1, 83, 0, 0, // Skip to: 2931
/* 2848 */    MCD::OPC_ExtractField, 11, 1,  // Inst{11} ...
/* 2851 */    MCD::OPC_FilterValue, 0, 62, 0, 0, // Skip to: 2918
/* 2856 */    MCD::OPC_ExtractField, 12, 3,  // Inst{14-12} ...
/* 2859 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 2879
/* 2864 */    MCD::OPC_CheckField, 17, 1, 0, 96, 1, 0, // Skip to: 3223
/* 2871 */    MCD::OPC_TryDecode, 177, 8, 145, 2, 88, 1, 0, // Opcode: VST_3D_128, skip to: 3223
/* 2879 */    MCD::OPC_FilterValue, 2, 8, 0, 0, // Skip to: 2892
/* 2884 */    MCD::OPC_TryDecode, 159, 8, 146, 2, 75, 1, 0, // Opcode: VST_128_ag_pstm_nrm, skip to: 3223
/* 2892 */    MCD::OPC_FilterValue, 4, 8, 0, 0, // Skip to: 2905
/* 2897 */    MCD::OPC_TryDecode, 157, 8, 147, 2, 62, 1, 0, // Opcode: VST_128_ag_idx, skip to: 3223
/* 2905 */    MCD::OPC_FilterValue, 6, 57, 1, 0, // Skip to: 3223
/* 2910 */    MCD::OPC_TryDecode, 162, 8, 148, 2, 49, 1, 0, // Opcode: VST_2D_128, skip to: 3223
/* 2918 */    MCD::OPC_FilterValue, 1, 44, 1, 0, // Skip to: 3223
/* 2923 */    MCD::OPC_TryDecode, 158, 8, 149, 2, 36, 1, 0, // Opcode: VST_128_ag_idx_imm, skip to: 3223
/* 2931 */    MCD::OPC_FilterValue, 3, 31, 1, 0, // Skip to: 3223
/* 2936 */    MCD::OPC_TryDecode, 160, 8, 150, 2, 23, 1, 0, // Opcode: VST_128_ag_pstm_nrm_imm, skip to: 3223
/* 2944 */    MCD::OPC_FilterValue, 1, 18, 1, 0, // Skip to: 3223
/* 2949 */    MCD::OPC_TryDecode, 161, 8, 151, 2, 10, 1, 0, // Opcode: VST_128_ag_spill, skip to: 3223
/* 2957 */    MCD::OPC_FilterValue, 1, 5, 1, 0, // Skip to: 3223
/* 2962 */    MCD::OPC_ExtractField, 8, 1,  // Inst{8} ...
/* 2965 */    MCD::OPC_FilterValue, 0, 240, 0, 0, // Skip to: 3210
/* 2970 */    MCD::OPC_ExtractField, 9, 2,  // Inst{10-9} ...
/* 2973 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 2993
/* 2978 */    MCD::OPC_CheckField, 11, 5, 0, 238, 0, 0, // Skip to: 3223
/* 2985 */    MCD::OPC_TryDecode, 154, 6, 152, 2, 230, 0, 0, // Opcode: VCONV_BF16_FP32, skip to: 3223
/* 2993 */    MCD::OPC_FilterValue, 1, 83, 0, 0, // Skip to: 3081
/* 2998 */    MCD::OPC_ExtractField, 11, 1,  // Inst{11} ...
/* 3001 */    MCD::OPC_FilterValue, 0, 62, 0, 0, // Skip to: 3068
/* 3006 */    MCD::OPC_ExtractField, 12, 3,  // Inst{14-12} ...
/* 3009 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 3029
/* 3014 */    MCD::OPC_CheckField, 17, 1, 0, 202, 0, 0, // Skip to: 3223
/* 3021 */    MCD::OPC_TryDecode, 191, 8, 145, 2, 194, 0, 0, // Opcode: VST_3D_dmw_sts_w, skip to: 3223
/* 3029 */    MCD::OPC_FilterValue, 2, 8, 0, 0, // Skip to: 3042
/* 3034 */    MCD::OPC_TryDecode, 253, 8, 146, 2, 181, 0, 0, // Opcode: VST_dmw_sts_w_ag_pstm_nrm, skip to: 3223
/* 3042 */    MCD::OPC_FilterValue, 4, 8, 0, 0, // Skip to: 3055
/* 3047 */    MCD::OPC_TryDecode, 251, 8, 147, 2, 168, 0, 0, // Opcode: VST_dmw_sts_w_ag_idx, skip to: 3223
/* 3055 */    MCD::OPC_FilterValue, 6, 163, 0, 0, // Skip to: 3223
/* 3060 */    MCD::OPC_TryDecode, 176, 8, 148, 2, 155, 0, 0, // Opcode: VST_2D_dmw_sts_w, skip to: 3223
/* 3068 */    MCD::OPC_FilterValue, 1, 150, 0, 0, // Skip to: 3223
/* 3073 */    MCD::OPC_TryDecode, 252, 8, 149, 2, 142, 0, 0, // Opcode: VST_dmw_sts_w_ag_idx_imm, skip to: 3223
/* 3081 */    MCD::OPC_FilterValue, 2, 111, 0, 0, // Skip to: 3197
/* 3086 */    MCD::OPC_ExtractField, 11, 3,  // Inst{13-11} ...
/* 3089 */    MCD::OPC_FilterValue, 0, 36, 0, 0, // Skip to: 3130
/* 3094 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 3097 */    MCD::OPC_FilterValue, 0, 15, 0, 0, // Skip to: 3117
/* 3102 */    MCD::OPC_CheckField, 17, 1, 0, 114, 0, 0, // Skip to: 3223
/* 3109 */    MCD::OPC_TryDecode, 193, 8, 153, 2, 106, 0, 0, // Opcode: VST_CONV_3D_BF16_FP32, skip to: 3223
/* 3117 */    MCD::OPC_FilterValue, 1, 101, 0, 0, // Skip to: 3223
/* 3122 */    MCD::OPC_TryDecode, 194, 8, 154, 2, 93, 0, 0, // Opcode: VST_CONV_BF16_FP32_ag_idx, skip to: 3223
/* 3130 */    MCD::OPC_FilterValue, 2, 8, 0, 0, // Skip to: 3143
/* 3135 */    MCD::OPC_TryDecode, 197, 8, 155, 2, 80, 0, 0, // Opcode: VST_CONV_BF16_FP32_ag_pstm_nrm_imm, skip to: 3223
/* 3143 */    MCD::OPC_FilterValue, 4, 29, 0, 0, // Skip to: 3177
/* 3148 */    MCD::OPC_ExtractField, 14, 1,  // Inst{14} ...
/* 3151 */    MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 3164
/* 3156 */    MCD::OPC_TryDecode, 196, 8, 156, 2, 59, 0, 0, // Opcode: VST_CONV_BF16_FP32_ag_pstm_nrm, skip to: 3223
/* 3164 */    MCD::OPC_FilterValue, 1, 54, 0, 0, // Skip to: 3223
/* 3169 */    MCD::OPC_TryDecode, 192, 8, 157, 2, 46, 0, 0, // Opcode: VST_CONV_2D_BF16_FP32, skip to: 3223
/* 3177 */    MCD::OPC_FilterValue, 6, 41, 0, 0, // Skip to: 3223
/* 3182 */    MCD::OPC_CheckField, 14, 1, 0, 34, 0, 0, // Skip to: 3223
/* 3189 */    MCD::OPC_TryDecode, 195, 8, 158, 2, 26, 0, 0, // Opcode: VST_CONV_BF16_FP32_ag_idx_imm, skip to: 3223
/* 3197 */    MCD::OPC_FilterValue, 3, 21, 0, 0, // Skip to: 3223
/* 3202 */    MCD::OPC_TryDecode, 254, 8, 150, 2, 13, 0, 0, // Opcode: VST_dmw_sts_w_ag_pstm_nrm_imm, skip to: 3223
/* 3210 */    MCD::OPC_FilterValue, 1, 8, 0, 0, // Skip to: 3223
/* 3215 */    MCD::OPC_TryDecode, 255, 8, 151, 2, 0, 0, 0, // Opcode: VST_dmw_sts_w_ag_spill, skip to: 3223
/* 3223 */    MCD::OPC_Fail,
  0
};

static const uint8_t DecoderTableVec32[] = {
/* 0 */       MCD::OPC_ExtractField, 0, 3,  // Inst{2-0} ...
/* 3 */       MCD::OPC_FilterValue, 0, 127, 1, 0, // Skip to: 391
/* 8 */       MCD::OPC_ExtractField, 3, 1,  // Inst{3} ...
/* 11 */      MCD::OPC_FilterValue, 0, 194, 0, 0, // Skip to: 210
/* 16 */      MCD::OPC_ExtractField, 16, 1,  // Inst{16} ...
/* 19 */      MCD::OPC_FilterValue, 0, 142, 0, 0, // Skip to: 166
/* 24 */      MCD::OPC_CheckField, 17, 9, 0, 11, 0, 0, // Skip to: 42
/* 31 */      MCD::OPC_CheckField, 4, 12, 30, 4, 0, 0, // Skip to: 42
/* 38 */      MCD::OPC_Decode, 187, 5, 0, // Opcode: NOPV
/* 42 */      MCD::OPC_CheckField, 21, 5, 0, 16, 0, 0, // Skip to: 65
/* 49 */      MCD::OPC_CheckField, 8, 8, 240, 1, 8, 0, 0, // Skip to: 65
/* 57 */      MCD::OPC_TryDecode, 210, 7, 159, 2, 0, 0, 0, // Opcode: VMOV_D, skip to: 65
/* 65 */      MCD::OPC_ExtractField, 17, 4,  // Inst{20-17} ...
/* 68 */      MCD::OPC_FilterValue, 15, 39, 0, 0, // Skip to: 112
/* 73 */      MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 76 */      MCD::OPC_FilterValue, 3, 23, 0, 0, // Skip to: 104
/* 81 */      MCD::OPC_CheckField, 14, 2, 3, 8, 0, 0, // Skip to: 96
/* 88 */      MCD::OPC_TryDecode, 229, 7, 160, 2, 0, 0, 0, // Opcode: VMUL_vmac_cm_core_sparse_wide, skip to: 96
/* 96 */      MCD::OPC_TryDecode, 228, 7, 161, 2, 0, 0, 0, // Opcode: VMUL_vmac_cm_core_sparse_narrow, skip to: 104
/* 104 */     MCD::OPC_TryDecode, 227, 7, 162, 2, 0, 0, 0, // Opcode: VMUL_vmac_cm_core_dense, skip to: 112
/* 112 */     MCD::OPC_CheckField, 12, 4, 14, 8, 0, 0, // Skip to: 127
/* 119 */     MCD::OPC_TryDecode, 247, 5, 163, 2, 0, 0, 0, // Opcode: VADD, skip to: 127
/* 127 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 130 */     MCD::OPC_FilterValue, 3, 23, 0, 0, // Skip to: 158
/* 135 */     MCD::OPC_CheckField, 14, 2, 3, 8, 0, 0, // Skip to: 150
/* 142 */     MCD::OPC_TryDecode, 189, 7, 164, 2, 0, 0, 0, // Opcode: VMAC_vmac_cm_core_sparse_wide, skip to: 150
/* 150 */     MCD::OPC_TryDecode, 188, 7, 165, 2, 0, 0, 0, // Opcode: VMAC_vmac_cm_core_sparse_narrow, skip to: 158
/* 158 */     MCD::OPC_TryDecode, 187, 7, 166, 2, 80, 3, 0, // Opcode: VMAC_vmac_cm_core_dense, skip to: 1014
/* 166 */     MCD::OPC_FilterValue, 1, 75, 3, 0, // Skip to: 1014
/* 171 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 174 */     MCD::OPC_FilterValue, 3, 23, 0, 0, // Skip to: 202
/* 179 */     MCD::OPC_CheckField, 14, 2, 3, 8, 0, 0, // Skip to: 194
/* 186 */     MCD::OPC_TryDecode, 253, 5, 167, 2, 0, 0, 0, // Opcode: VADDMAC_vmac_cm_core_sparse_wide, skip to: 194
/* 194 */     MCD::OPC_TryDecode, 252, 5, 168, 2, 0, 0, 0, // Opcode: VADDMAC_vmac_cm_core_sparse_narrow, skip to: 202
/* 202 */     MCD::OPC_TryDecode, 251, 5, 169, 2, 36, 3, 0, // Opcode: VADDMAC_vmac_bm_core_dense, skip to: 1014
/* 210 */     MCD::OPC_FilterValue, 1, 31, 3, 0, // Skip to: 1014
/* 215 */     MCD::OPC_ExtractField, 16, 1,  // Inst{16} ...
/* 218 */     MCD::OPC_FilterValue, 0, 124, 0, 0, // Skip to: 347
/* 223 */     MCD::OPC_CheckField, 17, 9, 0, 16, 0, 0, // Skip to: 246
/* 230 */     MCD::OPC_CheckField, 8, 8, 240, 1, 8, 0, 0, // Skip to: 246
/* 238 */     MCD::OPC_TryDecode, 152, 6, 170, 2, 0, 0, 0, // Opcode: VCLR_vclr, skip to: 246
/* 246 */     MCD::OPC_ExtractField, 17, 4,  // Inst{20-17} ...
/* 249 */     MCD::OPC_FilterValue, 15, 39, 0, 0, // Skip to: 293
/* 254 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 257 */     MCD::OPC_FilterValue, 3, 23, 0, 0, // Skip to: 285
/* 262 */     MCD::OPC_CheckField, 14, 2, 3, 8, 0, 0, // Skip to: 277
/* 269 */     MCD::OPC_TryDecode, 244, 7, 160, 2, 0, 0, 0, // Opcode: VNEGMUL_vmac_cm_core_sparse_wide, skip to: 277
/* 277 */     MCD::OPC_TryDecode, 243, 7, 161, 2, 0, 0, 0, // Opcode: VNEGMUL_vmac_cm_core_sparse_narrow, skip to: 285
/* 285 */     MCD::OPC_TryDecode, 242, 7, 162, 2, 0, 0, 0, // Opcode: VNEGMUL_vmac_cm_core_dense, skip to: 293
/* 293 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 296 */     MCD::OPC_FilterValue, 3, 23, 0, 0, // Skip to: 324
/* 301 */     MCD::OPC_CheckField, 14, 2, 3, 8, 0, 0, // Skip to: 316
/* 308 */     MCD::OPC_TryDecode, 223, 7, 164, 2, 0, 0, 0, // Opcode: VMSC_vmac_cm_core_sparse_wide, skip to: 316
/* 316 */     MCD::OPC_TryDecode, 222, 7, 165, 2, 0, 0, 0, // Opcode: VMSC_vmac_cm_core_sparse_narrow, skip to: 324
/* 324 */     MCD::OPC_CheckField, 12, 4, 14, 8, 0, 0, // Skip to: 339
/* 331 */     MCD::OPC_TryDecode, 128, 9, 163, 2, 0, 0, 0, // Opcode: VSUB, skip to: 339
/* 339 */     MCD::OPC_TryDecode, 221, 7, 166, 2, 155, 2, 0, // Opcode: VMSC_vmac_cm_core_dense, skip to: 1014
/* 347 */     MCD::OPC_FilterValue, 1, 150, 2, 0, // Skip to: 1014
/* 352 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 355 */     MCD::OPC_FilterValue, 3, 23, 0, 0, // Skip to: 383
/* 360 */     MCD::OPC_CheckField, 14, 2, 3, 8, 0, 0, // Skip to: 375
/* 367 */     MCD::OPC_TryDecode, 131, 9, 167, 2, 0, 0, 0, // Opcode: VSUBMAC_vmac_cm_core_sparse_wide, skip to: 375
/* 375 */     MCD::OPC_TryDecode, 130, 9, 168, 2, 0, 0, 0, // Opcode: VSUBMAC_vmac_cm_core_sparse_narrow, skip to: 383
/* 383 */     MCD::OPC_TryDecode, 129, 9, 169, 2, 111, 2, 0, // Opcode: VSUBMAC_vmac_bm_core_dense, skip to: 1014
/* 391 */     MCD::OPC_FilterValue, 1, 127, 0, 0, // Skip to: 523
/* 396 */     MCD::OPC_CheckField, 8, 18, 240, 63, 8, 0, 0, // Skip to: 412
/* 404 */     MCD::OPC_TryDecode, 153, 6, 171, 2, 0, 0, 0, // Opcode: VCLR_vclr_BF, skip to: 412
/* 412 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 415 */     MCD::OPC_FilterValue, 0, 22, 0, 0, // Skip to: 442
/* 420 */     MCD::OPC_CheckField, 12, 4, 15, 73, 0, 0, // Skip to: 500
/* 427 */     MCD::OPC_CheckField, 8, 2, 0, 66, 0, 0, // Skip to: 500
/* 434 */     MCD::OPC_TryDecode, 247, 7, 172, 2, 58, 0, 0, // Opcode: VNEG_F, skip to: 500
/* 442 */     MCD::OPC_FilterValue, 3, 53, 0, 0, // Skip to: 500
/* 447 */     MCD::OPC_CheckField, 14, 7, 127, 8, 0, 0, // Skip to: 462
/* 454 */     MCD::OPC_TryDecode, 226, 7, 173, 2, 0, 0, 0, // Opcode: VMUL_F_vmac_bm_core_sparse_wide, skip to: 462
/* 462 */     MCD::OPC_CheckField, 14, 2, 3, 8, 0, 0, // Skip to: 477
/* 469 */     MCD::OPC_TryDecode, 186, 7, 174, 2, 0, 0, 0, // Opcode: VMAC_F_vmac_bm_core_sparse_wide, skip to: 477
/* 477 */     MCD::OPC_CheckField, 16, 5, 31, 8, 0, 0, // Skip to: 492
/* 484 */     MCD::OPC_TryDecode, 225, 7, 175, 2, 0, 0, 0, // Opcode: VMUL_F_vmac_bm_core_sparse_narrow, skip to: 492
/* 492 */     MCD::OPC_TryDecode, 185, 7, 176, 2, 0, 0, 0, // Opcode: VMAC_F_vmac_bm_core_sparse_narrow, skip to: 500
/* 500 */     MCD::OPC_CheckField, 16, 5, 31, 8, 0, 0, // Skip to: 515
/* 507 */     MCD::OPC_TryDecode, 224, 7, 177, 2, 0, 0, 0, // Opcode: VMUL_F_vmac_bm_core_dense, skip to: 515
/* 515 */     MCD::OPC_TryDecode, 184, 7, 178, 2, 235, 1, 0, // Opcode: VMAC_F_vmac_bm_core_dense, skip to: 1014
/* 523 */     MCD::OPC_FilterValue, 2, 29, 0, 0, // Skip to: 557
/* 528 */     MCD::OPC_ExtractField, 13, 3,  // Inst{15-13} ...
/* 531 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 544
/* 536 */     MCD::OPC_TryDecode, 138, 6, 179, 2, 214, 1, 0, // Opcode: VADD_F, skip to: 1014
/* 544 */     MCD::OPC_FilterValue, 4, 209, 1, 0, // Skip to: 1014
/* 549 */     MCD::OPC_TryDecode, 246, 7, 179, 2, 201, 1, 0, // Opcode: VNEGSUB_F, skip to: 1014
/* 557 */     MCD::OPC_FilterValue, 3, 39, 0, 0, // Skip to: 601
/* 562 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 565 */     MCD::OPC_FilterValue, 3, 23, 0, 0, // Skip to: 593
/* 570 */     MCD::OPC_CheckField, 14, 2, 3, 8, 0, 0, // Skip to: 585
/* 577 */     MCD::OPC_TryDecode, 250, 5, 180, 2, 0, 0, 0, // Opcode: VADDMAC_F_vmac_bm_core_sparse_wide, skip to: 585
/* 585 */     MCD::OPC_TryDecode, 249, 5, 181, 2, 0, 0, 0, // Opcode: VADDMAC_F_vmac_bm_core_sparse_narrow, skip to: 593
/* 593 */     MCD::OPC_TryDecode, 248, 5, 182, 2, 157, 1, 0, // Opcode: VADDMAC_F_vmac_bm_core_dense, skip to: 1014
/* 601 */     MCD::OPC_FilterValue, 4, 241, 0, 0, // Skip to: 847
/* 606 */     MCD::OPC_ExtractField, 3, 1,  // Inst{3} ...
/* 609 */     MCD::OPC_FilterValue, 0, 122, 0, 0, // Skip to: 736
/* 614 */     MCD::OPC_ExtractField, 16, 1,  // Inst{16} ...
/* 617 */     MCD::OPC_FilterValue, 0, 54, 0, 0, // Skip to: 676
/* 622 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 625 */     MCD::OPC_FilterValue, 3, 23, 0, 0, // Skip to: 653
/* 630 */     MCD::OPC_CheckField, 14, 2, 3, 8, 0, 0, // Skip to: 645
/* 637 */     MCD::OPC_TryDecode, 238, 7, 164, 2, 0, 0, 0, // Opcode: VNEGMSC_vmac_cm_core_sparse_wide, skip to: 645
/* 645 */     MCD::OPC_TryDecode, 237, 7, 165, 2, 0, 0, 0, // Opcode: VNEGMSC_vmac_cm_core_sparse_narrow, skip to: 653
/* 653 */     MCD::OPC_CheckField, 12, 4, 14, 8, 0, 0, // Skip to: 668
/* 660 */     MCD::OPC_TryDecode, 245, 7, 163, 2, 0, 0, 0, // Opcode: VNEGSUB, skip to: 668
/* 668 */     MCD::OPC_TryDecode, 236, 7, 166, 2, 82, 1, 0, // Opcode: VNEGMSC_vmac_cm_core_dense, skip to: 1014
/* 676 */     MCD::OPC_FilterValue, 1, 77, 1, 0, // Skip to: 1014
/* 681 */     MCD::OPC_CheckField, 8, 8, 240, 1, 8, 0, 0, // Skip to: 697
/* 689 */     MCD::OPC_TryDecode, 230, 7, 183, 2, 0, 0, 0, // Opcode: VNEG, skip to: 697
/* 697 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 700 */     MCD::OPC_FilterValue, 3, 23, 0, 0, // Skip to: 728
/* 705 */     MCD::OPC_CheckField, 14, 2, 3, 8, 0, 0, // Skip to: 720
/* 712 */     MCD::OPC_TryDecode, 131, 6, 167, 2, 0, 0, 0, // Opcode: VADDMSC_vmac_cm_core_sparse_wide, skip to: 720
/* 720 */     MCD::OPC_TryDecode, 130, 6, 168, 2, 0, 0, 0, // Opcode: VADDMSC_vmac_cm_core_sparse_narrow, skip to: 728
/* 728 */     MCD::OPC_TryDecode, 129, 6, 169, 2, 22, 1, 0, // Opcode: VADDMSC_vmac_bm_core_dense, skip to: 1014
/* 736 */     MCD::OPC_FilterValue, 1, 17, 1, 0, // Skip to: 1014
/* 741 */     MCD::OPC_ExtractField, 16, 1,  // Inst{16} ...
/* 744 */     MCD::OPC_FilterValue, 0, 54, 0, 0, // Skip to: 803
/* 749 */     MCD::OPC_CheckField, 12, 4, 14, 8, 0, 0, // Skip to: 764
/* 756 */     MCD::OPC_TryDecode, 231, 7, 163, 2, 0, 0, 0, // Opcode: VNEGADD, skip to: 764
/* 764 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 767 */     MCD::OPC_FilterValue, 3, 23, 0, 0, // Skip to: 795
/* 772 */     MCD::OPC_CheckField, 14, 2, 3, 8, 0, 0, // Skip to: 787
/* 779 */     MCD::OPC_TryDecode, 235, 7, 164, 2, 0, 0, 0, // Opcode: VNEGMAC_vmac_cm_core_sparse_wide, skip to: 787
/* 787 */     MCD::OPC_TryDecode, 234, 7, 165, 2, 0, 0, 0, // Opcode: VNEGMAC_vmac_cm_core_sparse_narrow, skip to: 795
/* 795 */     MCD::OPC_TryDecode, 233, 7, 166, 2, 211, 0, 0, // Opcode: VNEGMAC_vmac_cm_core_dense, skip to: 1014
/* 803 */     MCD::OPC_FilterValue, 1, 206, 0, 0, // Skip to: 1014
/* 808 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 811 */     MCD::OPC_FilterValue, 3, 23, 0, 0, // Skip to: 839
/* 816 */     MCD::OPC_CheckField, 14, 2, 3, 8, 0, 0, // Skip to: 831
/* 823 */     MCD::OPC_TryDecode, 134, 9, 167, 2, 0, 0, 0, // Opcode: VSUBMSC_vmac_cm_core_sparse_wide, skip to: 831
/* 831 */     MCD::OPC_TryDecode, 133, 9, 168, 2, 0, 0, 0, // Opcode: VSUBMSC_vmac_cm_core_sparse_narrow, skip to: 839
/* 839 */     MCD::OPC_TryDecode, 132, 9, 169, 2, 167, 0, 0, // Opcode: VSUBMSC_vmac_bm_core_dense, skip to: 1014
/* 847 */     MCD::OPC_FilterValue, 5, 84, 0, 0, // Skip to: 936
/* 852 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 855 */     MCD::OPC_FilterValue, 3, 53, 0, 0, // Skip to: 913
/* 860 */     MCD::OPC_CheckField, 14, 7, 127, 8, 0, 0, // Skip to: 875
/* 867 */     MCD::OPC_TryDecode, 241, 7, 173, 2, 0, 0, 0, // Opcode: VNEGMUL_F_vmac_bm_core_sparse_wide, skip to: 875
/* 875 */     MCD::OPC_CheckField, 14, 2, 3, 8, 0, 0, // Skip to: 890
/* 882 */     MCD::OPC_TryDecode, 220, 7, 174, 2, 0, 0, 0, // Opcode: VMSC_F_vmac_bm_core_sparse_wide, skip to: 890
/* 890 */     MCD::OPC_CheckField, 16, 5, 31, 8, 0, 0, // Skip to: 905
/* 897 */     MCD::OPC_TryDecode, 240, 7, 175, 2, 0, 0, 0, // Opcode: VNEGMUL_F_vmac_bm_core_sparse_narrow, skip to: 905
/* 905 */     MCD::OPC_TryDecode, 219, 7, 176, 2, 0, 0, 0, // Opcode: VMSC_F_vmac_bm_core_sparse_narrow, skip to: 913
/* 913 */     MCD::OPC_CheckField, 16, 5, 31, 8, 0, 0, // Skip to: 928
/* 920 */     MCD::OPC_TryDecode, 239, 7, 177, 2, 0, 0, 0, // Opcode: VNEGMUL_F_vmac_bm_core_dense, skip to: 928
/* 928 */     MCD::OPC_TryDecode, 218, 7, 178, 2, 78, 0, 0, // Opcode: VMSC_F_vmac_bm_core_dense, skip to: 1014
/* 936 */     MCD::OPC_FilterValue, 6, 29, 0, 0, // Skip to: 970
/* 941 */     MCD::OPC_ExtractField, 13, 3,  // Inst{15-13} ...
/* 944 */     MCD::OPC_FilterValue, 0, 8, 0, 0, // Skip to: 957
/* 949 */     MCD::OPC_TryDecode, 138, 9, 179, 2, 57, 0, 0, // Opcode: VSUB_F, skip to: 1014
/* 957 */     MCD::OPC_FilterValue, 4, 52, 0, 0, // Skip to: 1014
/* 962 */     MCD::OPC_TryDecode, 232, 7, 179, 2, 44, 0, 0, // Opcode: VNEGADD_F, skip to: 1014
/* 970 */     MCD::OPC_FilterValue, 7, 39, 0, 0, // Skip to: 1014
/* 975 */     MCD::OPC_ExtractField, 10, 2,  // Inst{11-10} ...
/* 978 */     MCD::OPC_FilterValue, 3, 23, 0, 0, // Skip to: 1006
/* 983 */     MCD::OPC_CheckField, 14, 2, 3, 8, 0, 0, // Skip to: 998
/* 990 */     MCD::OPC_TryDecode, 128, 6, 180, 2, 0, 0, 0, // Opcode: VADDMSC_F_vmac_bm_core_sparse_wide, skip to: 998
/* 998 */     MCD::OPC_TryDecode, 255, 5, 181, 2, 0, 0, 0, // Opcode: VADDMSC_F_vmac_bm_core_sparse_narrow, skip to: 1006
/* 1006 */    MCD::OPC_TryDecode, 254, 5, 182, 2, 0, 0, 0, // Opcode: VADDMSC_F_vmac_bm_core_dense, skip to: 1014
/* 1014 */    MCD::OPC_Fail,
  0
};

static bool checkDecoderPredicate(unsigned Idx, const FeatureBitset &Bits) {
  llvm_unreachable("Invalid index!");
}

template <typename InsnType>
static DecodeStatus decodeToMCInst(DecodeStatus S, unsigned Idx, InsnType insn, MCInst &MI,
                                   uint64_t Address, const MCDisassembler *Decoder, bool &DecodeComplete) {
  DecodeComplete = true;
  using TmpType = std::conditional_t<std::is_integral<InsnType>::value, InsnType, uint64_t>;
  TmpType tmp;
  switch (Idx) {
  default: llvm_unreachable("Invalid index!");
  case 0:
    return S;
  case 1:
    tmp = fieldFromInstruction(insn, 7, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 2:
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 7, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 3:
    tmp = fieldFromInstruction(insn, 18, 2);
    if (!Check(S, decodeUImmOperand<2>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 4:
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 5:
    tmp = fieldFromInstruction(insn, 11, 4);
    if (!Check(S, DecodemCRmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 6:
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 5, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 7:
    tmp = fieldFromInstruction(insn, 9, 6);
    if (!Check(S, DecodemAluCgRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = 0x0;
    insertBits(tmp, fieldFromInstruction(insn, 3, 6), 0, 6);
    insertBits(tmp, fieldFromInstruction(insn, 15, 5), 6, 5);
    if (!Check(S, decodeSImmOperand<11>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 8:
    tmp = fieldFromInstruction(insn, 14, 6);
    if (!Check(S, decodeUImmOperand<6>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 5, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 9:
    tmp = fieldFromInstruction(insn, 14, 6);
    if (!Check(S, decodeUImmOperand<6>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 5, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    if (!Check(S, DecodeeR26RegisterClass(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 10:
    tmp = fieldFromInstruction(insn, 15, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 5, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 11:
    tmp = fieldFromInstruction(insn, 15, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 5, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    if (!Check(S, DecodeeR26RegisterClass(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 12:
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 5, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    if (!Check(S, DecodeeR27RegisterClass(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 13:
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    if (!Check(S, DecodeeR31RegisterClass(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    if (!Check(S, DecodeeR31RegisterClass(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 5, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 14:
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 7);
    if (!Check(S, decodeSImmOperand<7>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 15:
    tmp = fieldFromInstruction(insn, 15, 1);
    if (!Check(S, decodeNopSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 16:
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 17:
    tmp = fieldFromInstruction(insn, 6, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 18:
    tmp = fieldFromInstruction(insn, 6, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 19:
    tmp = fieldFromInstruction(insn, 7, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 20:
    tmp = fieldFromInstruction(insn, 5, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 21:
    tmp = fieldFromInstruction(insn, 11, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 22:
    tmp = fieldFromInstruction(insn, 27, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 5, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 23:
    tmp = fieldFromInstruction(insn, 27, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 24:
    tmp = fieldFromInstruction(insn, 27, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 7, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 25:
    tmp = fieldFromInstruction(insn, 27, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 5, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 26:
    tmp = fieldFromInstruction(insn, 6, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 27:
    tmp = fieldFromInstruction(insn, 27, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 28:
    tmp = fieldFromInstruction(insn, 27, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 7, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 29:
    tmp = fieldFromInstruction(insn, 27, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 7, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 30:
    tmp = fieldFromInstruction(insn, 27, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 31:
    tmp = fieldFromInstruction(insn, 43, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 32:
    tmp = fieldFromInstruction(insn, 63, 1);
    if (!Check(S, decodeNopSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 33:
    tmp = fieldFromInstruction(insn, 34, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 34:
    tmp = fieldFromInstruction(insn, 43, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 27, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 7, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 35:
    tmp = fieldFromInstruction(insn, 43, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 27, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 7, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 36:
    tmp = fieldFromInstruction(insn, 43, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 37:
    tmp = fieldFromInstruction(insn, 43, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 38:
    tmp = fieldFromInstruction(insn, 34, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 39:
    tmp = fieldFromInstruction(insn, 43, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 40:
    tmp = fieldFromInstruction(insn, 37, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 41:
    tmp = fieldFromInstruction(insn, 43, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 27, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 42:
    tmp = fieldFromInstruction(insn, 59, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 43, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 43:
    tmp = fieldFromInstruction(insn, 59, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 37, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 44:
    tmp = fieldFromInstruction(insn, 59, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 37, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 45:
    tmp = fieldFromInstruction(insn, 59, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 37, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 46:
    tmp = fieldFromInstruction(insn, 59, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 38, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 47:
    tmp = fieldFromInstruction(insn, 59, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 43, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 48:
    tmp = fieldFromInstruction(insn, 33, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 49:
    tmp = fieldFromInstruction(insn, 55, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 33, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 50:
    tmp = fieldFromInstruction(insn, 59, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 34, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 51:
    tmp = fieldFromInstruction(insn, 59, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 34, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 52:
    tmp = fieldFromInstruction(insn, 59, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 34, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 53:
    tmp = fieldFromInstruction(insn, 59, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 54:
    tmp = fieldFromInstruction(insn, 59, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 55:
    tmp = fieldFromInstruction(insn, 59, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 56:
    tmp = fieldFromInstruction(insn, 59, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 38, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 57:
    tmp = fieldFromInstruction(insn, 59, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 43, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 58:
    tmp = fieldFromInstruction(insn, 59, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 34, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 59:
    tmp = fieldFromInstruction(insn, 59, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 34, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 60:
    tmp = fieldFromInstruction(insn, 59, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 34, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 61:
    tmp = fieldFromInstruction(insn, 59, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 43, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 62:
    tmp = fieldFromInstruction(insn, 59, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 38, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 7, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 63:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 33, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 64:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 55, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 33, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 65:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 33, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 66:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 55, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 33, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 67:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 54, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 34, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 68:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 59, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 34, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 69:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 59, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 37, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 70:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 59, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 37, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 71:
    tmp = fieldFromInstruction(insn, 80, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 37, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 72:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 59, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 38, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 73:
    tmp = fieldFromInstruction(insn, 80, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 59, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 37, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 74:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 59, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 34, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 75:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 59, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 76:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 54, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 77:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 59, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 78:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 59, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 38, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 79:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 59, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 34, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 80:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 59, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 34, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 81:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 54, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 34, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 82:
    tmp = fieldFromInstruction(insn, 75, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 59, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 34, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 83:
    tmp = fieldFromInstruction(insn, 91, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 75, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 33, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 84:
    tmp = fieldFromInstruction(insn, 91, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 75, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 55, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 33, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 85:
    tmp = fieldFromInstruction(insn, 91, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 75, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 33, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 86:
    tmp = fieldFromInstruction(insn, 91, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 75, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 55, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 33, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 87:
    tmp = fieldFromInstruction(insn, 91, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 75, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 55, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 32, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 88:
    tmp = fieldFromInstruction(insn, 91, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 70, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 32, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 89:
    tmp = fieldFromInstruction(insn, 91, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 75, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 33, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 90:
    tmp = fieldFromInstruction(insn, 91, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 75, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 55, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 33, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 91:
    tmp = fieldFromInstruction(insn, 112, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 91, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 70, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 28, 42);
    if (!Check(S, decodeLngSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 1, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 92:
    tmp = fieldFromInstruction(insn, 112, 16);
    if (!Check(S, decodeLdbSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 91, 21);
    if (!Check(S, decodeLdaSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 70, 21);
    if (!Check(S, decodeStSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 50, 20);
    if (!Check(S, decodeAluSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 28, 22);
    if (!Check(S, decodeMvSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 1, 26);
    if (!Check(S, decodeVecSlot(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 93:
    if (!Check(S, DecodeST_3D_HBInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 94:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 95:
    if (!Check(S, DecodeVLDA_3D_UPS_BMInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 96:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 97:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 14, 4);
    if (!Check(S, decodeSImmOperand<4>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 98:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 14, 4);
    if (!Check(S, decodeSImmOperandXStep<4, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 99:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 100:
    if (!Check(S, DecodeST_2D_HBInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 101:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 102:
    if (!Check(S, DecodeVLDA_2D_UPS_BMInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 103:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, decodeSImmOperand<3>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 104:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, decodeSImmOperandXStep<3, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 105:
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemCMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 106:
    if (!Check(S, DecodeVLDA_3D_UPS_CMInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 107:
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemCMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 108:
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemCMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 14, 4);
    if (!Check(S, decodeSImmOperandXStep<4, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 109:
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemCMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 110:
    if (!Check(S, DecodeVLDA_2D_UPS_CMInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 111:
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemCMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, decodeSImmOperandXStep<3, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 112:
    if (!Check(S, DecodeVLDA_3D_CONV_BMInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 113:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 114:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 115:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 116:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 14, 4);
    if (!Check(S, decodeSImmOperandXStep<4, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 117:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 118:
    if (!Check(S, DecodeVLDA_2D_CONV_BMInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 119:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, decodeSImmOperandXStep<3, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 120:
    if (!Check(S, DecodePADD_3DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 121:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 122:
    if (!Check(S, DecodePADD_2DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 123:
    tmp = fieldFromInstruction(insn, 8, 13);
    if (!Check(S, decodeSImmOperandXStep<13, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 124:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 10);
    if (!Check(S, decodeSImmOperandXStep<10, 4, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 125:
    tmp = fieldFromInstruction(insn, 2, 6);
    if (!Check(S, DecodemMvBMXDstRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 126:
    if (!Check(S, DecodeVLDA_3D_AMInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 127:
    tmp = fieldFromInstruction(insn, 2, 6);
    if (!Check(S, DecodemAMmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 128:
    tmp = fieldFromInstruction(insn, 2, 6);
    if (!Check(S, DecodemAMmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 129:
    if (!Check(S, DecodeVLDA_2D_AMInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 130:
    tmp = fieldFromInstruction(insn, 2, 6);
    if (!Check(S, DecodemAMmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 6);
    if (!Check(S, decodeSImmOperandXStep<6, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 131:
    tmp = fieldFromInstruction(insn, 2, 6);
    if (!Check(S, DecodemAMmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 7);
    if (!Check(S, decodeSImmOperandXStep<7, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 132:
    tmp = fieldFromInstruction(insn, 2, 6);
    if (!Check(S, DecodemAMmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 9, 12);
    if (!Check(S, decodeSImmOperandXStep<12, 32, 1, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 133:
    if (!Check(S, DecodeLDA_3D_HBInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 134:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemWaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 135:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 14, 4);
    if (!Check(S, decodeSImmOperand<4>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 136:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 137:
    if (!Check(S, DecodeLDA_2D_HBInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 138:
    tmp = fieldFromInstruction(insn, 1, 7);
    if (!Check(S, DecodemLdaCgRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 11);
    if (!Check(S, decodeSImmOperand<11>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 139:
    if (!Check(S, DecodeLDA_Q_3DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 140:
    if (!Check(S, DecodeVLDA_3D_WInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 141:
    if (!Check(S, DecodeLDA_3D_Instruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 142:
    tmp = fieldFromInstruction(insn, 6, 2);
    if (!Check(S, DecodemQQaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 143:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemWaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 144:
    tmp = fieldFromInstruction(insn, 1, 7);
    if (!Check(S, DecodemLdaSclRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 145:
    tmp = fieldFromInstruction(insn, 6, 2);
    if (!Check(S, DecodemQQaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 146:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemWaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 147:
    tmp = fieldFromInstruction(insn, 1, 7);
    if (!Check(S, DecodemLdaSclRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 148:
    if (!Check(S, DecodeLDA_Q_2DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 149:
    if (!Check(S, DecodeVLDA_2D_WInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 150:
    if (!Check(S, DecodeLDA_2D_Instruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 151:
    tmp = fieldFromInstruction(insn, 6, 2);
    if (!Check(S, DecodemQQaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 6);
    if (!Check(S, decodeSImmOperandXStep<6, 16, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 152:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemWaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 6);
    if (!Check(S, decodeSImmOperandXStep<6, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 153:
    tmp = fieldFromInstruction(insn, 1, 7);
    if (!Check(S, DecodemLdaSclRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 6);
    if (!Check(S, decodeSImmOperandXStep<6, 4, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 154:
    tmp = fieldFromInstruction(insn, 6, 2);
    if (!Check(S, DecodemQQaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 7);
    if (!Check(S, decodeSImmOperandXStep<7, 16, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 155:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemWaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 7);
    if (!Check(S, decodeSImmOperandXStep<7, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 156:
    tmp = fieldFromInstruction(insn, 1, 7);
    if (!Check(S, DecodemLdaSclRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 7);
    if (!Check(S, decodeSImmOperandXStep<7, 4, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 157:
    tmp = fieldFromInstruction(insn, 6, 2);
    if (!Check(S, DecodemQQaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 9, 12);
    if (!Check(S, decodeSImmOperandXStep<12, 16, 1, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 158:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemWaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 9, 12);
    if (!Check(S, decodeSImmOperandXStep<12, 32, 1, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 159:
    tmp = fieldFromInstruction(insn, 1, 7);
    if (!Check(S, DecodemLdaSclRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 9, 12);
    if (!Check(S, decodeSImmOperandXStep<12, 4, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 160:
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 161:
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 3);
    if (!Check(S, DecodemQXHLbRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 162:
    tmp = fieldFromInstruction(insn, 3, 4);
    if (!Check(S, DecodemXsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 163:
    tmp = fieldFromInstruction(insn, 3, 4);
    if (!Check(S, DecodemXsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 164:
    tmp = fieldFromInstruction(insn, 3, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 5);
    if (!Check(S, DecodemWsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 165:
    if (!Check(S, DecodeVLDB_UNPACK_2DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 166:
    if (!Check(S, DecodeVLDB_UNPACK_3DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 167:
    tmp = 0x0;
    insertBits(tmp, fieldFromInstruction(insn, 2, 3), 0, 3);
    insertBits(tmp, fieldFromInstruction(insn, 7, 9), 3, 9);
    if (!Check(S, decodeSImmOperandXStep<12, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 168:
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 5);
    if (!Check(S, DecodemWbRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 169:
    if (!Check(S, DecodeVLDB_3DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 170:
    if (!Check(S, DecodePADDB_2DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 171:
    tmp = fieldFromInstruction(insn, 2, 5);
    if (!Check(S, DecodemWaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 172:
    tmp = fieldFromInstruction(insn, 2, 5);
    if (!Check(S, DecodemWaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 9, 4);
    if (!Check(S, decodeSImmOperandXStep<4, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 173:
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 174:
    tmp = fieldFromInstruction(insn, 2, 5);
    if (!Check(S, DecodemWaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 175:
    if (!Check(S, DecodePADDB_3DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 176:
    if (!Check(S, DecodeVLDB_2DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 177:
    tmp = fieldFromInstruction(insn, 2, 5);
    if (!Check(S, DecodemWaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 3);
    if (!Check(S, decodeSImmOperandXStep<3, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 178:
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = 0x0;
    insertBits(tmp, fieldFromInstruction(insn, 2, 3), 0, 3);
    insertBits(tmp, fieldFromInstruction(insn, 7, 6), 3, 6);
    if (!Check(S, decodeSImmOperandXStep<9, 4, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 179:
    tmp = fieldFromInstruction(insn, 2, 5);
    if (!Check(S, DecodemWsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 5);
    if (!Check(S, DecodemWbRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 180:
    tmp = fieldFromInstruction(insn, 15, 7);
    if (!Check(S, DecodemMvSclDstCgRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = 0x0;
    insertBits(tmp, fieldFromInstruction(insn, 3, 12), 0, 12);
    insertBits(tmp, fieldFromInstruction(insn, 22, 20), 12, 20);
    if (!Check(S, decodeSImmOperand<32>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 181:
    tmp = fieldFromInstruction(insn, 17, 20);
    MI.addOperand(MCOperand::createImm(tmp));
    return S;
  case 182:
    tmp = fieldFromInstruction(insn, 17, 20);
    if (!Check(S, decodeUImmOperand<20>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 183:
    tmp = fieldFromInstruction(insn, 37, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 17, 20);
    MI.addOperand(MCOperand::createImm(tmp));
    return S;
  case 184:
    tmp = fieldFromInstruction(insn, 15, 7);
    if (!Check(S, DecodemMvSclDstCgRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = 0x0;
    insertBits(tmp, fieldFromInstruction(insn, 3, 1), 0, 1);
    insertBits(tmp, fieldFromInstruction(insn, 6, 9), 1, 9);
    if (!Check(S, decodeSImmOperand<10>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 185:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 186:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeLRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 187:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeLRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 188:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeLRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 189:
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeLRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 190:
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeLRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 191:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeRS8RegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 192:
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeRS8RegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 193:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeRS8RegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 194:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeRS8RegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 195:
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeRS8RegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 196:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    if (!Check(S, DecodeeR29RegisterClass(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 197:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    if (!Check(S, DecodeeR29RegisterClass(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 3);
    if (!Check(S, DecodeeLRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 198:
    tmp = fieldFromInstruction(insn, 15, 7);
    if (!Check(S, DecodemMvSclDstRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodeeRS4RegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 199:
    tmp = fieldFromInstruction(insn, 19, 3);
    if (!Check(S, DecodeeLRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodeeRS4RegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 200:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 201:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, decodeUImmOperand<5>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 202:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 203:
    tmp = 0x0;
    insertBits(tmp, fieldFromInstruction(insn, 9, 1), 0, 1);
    insertBits(tmp, fieldFromInstruction(insn, 18, 4), 1, 4);
    if (!Check(S, DecodemShflDstRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    if (!Check(S, DecodeeR29RegisterClass(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 204:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 3);
    if (!Check(S, DecodeeLRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 205:
    tmp = 0x0;
    insertBits(tmp, fieldFromInstruction(insn, 9, 1), 0, 1);
    insertBits(tmp, fieldFromInstruction(insn, 18, 4), 1, 4);
    if (!Check(S, DecodemShflDstRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 3);
    if (!Check(S, DecodeeLRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    if (!Check(S, DecodeeR29RegisterClass(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 206:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 3);
    if (!Check(S, DecodeeLRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 207:
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeLRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 208:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemCMmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 4);
    if (!Check(S, DecodemCMmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 209:
    tmp = fieldFromInstruction(insn, 17, 5);
    if (!Check(S, DecodemWmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodemBMmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 2);
    if (!Check(S, DecodemSmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 210:
    tmp = fieldFromInstruction(insn, 17, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 7);
    if (!Check(S, DecodemMvSclSrcRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 211:
    tmp = fieldFromInstruction(insn, 15, 7);
    if (!Check(S, DecodemMvSclDstRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 7);
    if (!Check(S, DecodemMvSclSrcRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 212:
    tmp = fieldFromInstruction(insn, 16, 6);
    if (!Check(S, DecodemMvBMXDstRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 9);
    if (!Check(S, DecodemMvBMXSrcRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 213:
    tmp = fieldFromInstruction(insn, 15, 7);
    if (!Check(S, DecodemMvAMWQDstRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 9);
    if (!Check(S, DecodemMvAMWQSrcRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 214:
    tmp = fieldFromInstruction(insn, 17, 5);
    if (!Check(S, DecodemBMmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 5);
    if (!Check(S, DecodemWm_1RegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 215:
    tmp = fieldFromInstruction(insn, 17, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 5);
    if (!Check(S, DecodemWm_1RegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 216:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemCMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 217:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemCMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 5);
    if (!Check(S, DecodemWm_1RegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 218:
    tmp = fieldFromInstruction(insn, 15, 7);
    if (!Check(S, DecodemMvSclDstRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = 0x0;
    insertBits(tmp, fieldFromInstruction(insn, 2, 2), 0, 2);
    insertBits(tmp, fieldFromInstruction(insn, 6, 4), 2, 4);
    if (!Check(S, decodeSImmOperand<6>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 219:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 220:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 3);
    if (!Check(S, DecodeeLRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 221:
    tmp = fieldFromInstruction(insn, 17, 5);
    if (!Check(S, DecodemShflDstRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 222:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 223:
    tmp = fieldFromInstruction(insn, 18, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 4);
    if (!Check(S, DecodemXmRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 4);
    if (!Check(S, DecodemXnRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 224:
    tmp = fieldFromInstruction(insn, 1, 7);
    if (!Check(S, DecodemSclMSRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 225:
    tmp = fieldFromInstruction(insn, 1, 7);
    if (!Check(S, DecodemSclMSRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    if (!Check(S, DecodeeR28RegisterClass(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 226:
    if (!Check(S, DecodeVST_3D_SRS_CMInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 227:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemCMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 228:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 14, 4);
    if (!Check(S, decodeSImmOperandXStep<4, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemCMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 229:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemCMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 230:
    if (!Check(S, DecodeVST_2D_SRS_CMInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 231:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, decodeSImmOperandXStep<3, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemCMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 232:
    tmp = fieldFromInstruction(insn, 16, 5);
    if (!Check(S, DecodemWaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemCMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 233:
    tmp = fieldFromInstruction(insn, 17, 4);
    if (!Check(S, DecodemXaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemCMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 234:
    if (!Check(S, DecodeVST_PACK_3DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 235:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemXsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 236:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 14, 4);
    if (!Check(S, decodeSImmOperandXStep<4, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemXsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 237:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemXsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 238:
    if (!Check(S, DecodeVST_PACK_2DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 239:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, decodeSImmOperandXStep<3, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemXsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 240:
    tmp = fieldFromInstruction(insn, 16, 5);
    if (!Check(S, DecodemWaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemXsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 241:
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 2);
    if (!Check(S, decodeUImmOperand<2>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 2);
    if (!Check(S, decodeUImmOperand<2>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 242:
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 2);
    if (!Check(S, decodeUImmOperand<2>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 13, 2);
    if (!Check(S, decodeUImmOperand<2>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    if (!Check(S, DecodeeR28RegisterClass(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 243:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 3);
    if (!Check(S, decodeUImmOperand<3>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 244:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 3);
    if (!Check(S, decodeUImmOperand<3>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    if (!Check(S, DecodeeR28RegisterClass(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 245:
    if (!Check(S, DecodeVST_3D_SRS_BMInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 246:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 247:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 14, 4);
    if (!Check(S, decodeSImmOperandXStep<4, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 248:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 249:
    if (!Check(S, DecodeVST_2D_SRS_BMInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 250:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, decodeSImmOperandXStep<3, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 251:
    tmp = fieldFromInstruction(insn, 16, 5);
    if (!Check(S, DecodemWaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 252:
    tmp = fieldFromInstruction(insn, 1, 7);
    if (!Check(S, DecodemSclStRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 253:
    if (!Check(S, DecodeST_Q_3DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 254:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 2);
    if (!Check(S, DecodemQQaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 255:
    if (!Check(S, DecodeST_Q_2DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 256:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 6, 2);
    if (!Check(S, DecodemQQaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 7);
    if (!Check(S, decodeSImmOperandXStep<7, 16, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 257:
    if (!Check(S, DecodeST_3D_Instruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 258:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 1, 7);
    if (!Check(S, DecodemSclStRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 259:
    tmp = fieldFromInstruction(insn, 1, 7);
    if (!Check(S, DecodemSclStRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 260:
    if (!Check(S, DecodeST_2D_Instruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 261:
    tmp = fieldFromInstruction(insn, 1, 7);
    if (!Check(S, DecodemSclStRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 6);
    if (!Check(S, decodeSImmOperandXStep<6, 4, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 262:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 1, 7);
    if (!Check(S, DecodemSclStRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 7);
    if (!Check(S, decodeSImmOperandXStep<7, 4, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 263:
    tmp = fieldFromInstruction(insn, 1, 7);
    if (!Check(S, DecodemSclStRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 9, 12);
    if (!Check(S, decodeSImmOperandXStep<12, 4, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 264:
    tmp = fieldFromInstruction(insn, 17, 4);
    if (!Check(S, DecodemXaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemWsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 265:
    tmp = fieldFromInstruction(insn, 17, 4);
    if (!Check(S, DecodemXaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 6);
    if (!Check(S, DecodemAMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 10, 2);
    if (!Check(S, DecodemSsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 266:
    if (!Check(S, DecodeVST_3D_AMInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 267:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 6);
    if (!Check(S, DecodemAMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 268:
    tmp = fieldFromInstruction(insn, 2, 6);
    if (!Check(S, DecodemAMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 269:
    if (!Check(S, DecodeVST_AM_2DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 270:
    tmp = fieldFromInstruction(insn, 2, 6);
    if (!Check(S, DecodemAMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 6);
    if (!Check(S, decodeSImmOperandXStep<6, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 271:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 2, 6);
    if (!Check(S, DecodemAMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 7);
    if (!Check(S, decodeSImmOperandXStep<7, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 272:
    tmp = fieldFromInstruction(insn, 2, 6);
    if (!Check(S, DecodemAMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 9, 12);
    if (!Check(S, decodeSImmOperandXStep<12, 32, 1, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 273:
    if (!Check(S, DecodeVST_3D_WInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 274:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemWsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 275:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemWsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 276:
    if (!Check(S, DecodeVST_W_2DInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 277:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemWsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 6);
    if (!Check(S, decodeSImmOperandXStep<6, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 278:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemWsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 11, 7);
    if (!Check(S, decodeSImmOperandXStep<7, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 279:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemWsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 9, 12);
    if (!Check(S, decodeSImmOperandXStep<12, 32, 1, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    return S;
  case 280:
    tmp = fieldFromInstruction(insn, 16, 5);
    if (!Check(S, DecodemWaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 281:
    if (!Check(S, DecodeVST_3D_CONV_BMInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 282:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeDJRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 283:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 14, 4);
    if (!Check(S, decodeSImmOperandXStep<4, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 284:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, DecodeeMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 285:
    if (!Check(S, DecodeVST_2D_CONV_BMInstruction(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 286:
    tmp = fieldFromInstruction(insn, 18, 3);
    if (!Check(S, DecodeePRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 15, 3);
    if (!Check(S, decodeSImmOperandXStep<3, 32, 0, 0>(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 287:
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemCMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 17, 4);
    if (!Check(S, DecodemCMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 288:
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 2);
    if (!Check(S, DecodeeYsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 2);
    if (!Check(S, DecodemQQXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 289:
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (!Check(S, DecodemXvRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 2);
    if (!Check(S, DecodemQQXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 290:
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (!Check(S, DecodemXvRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 4);
    if (!Check(S, DecodemXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 291:
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 17, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 292:
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 17, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 2);
    if (!Check(S, DecodeeYsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 2);
    if (!Check(S, DecodemQQXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 293:
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 17, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (!Check(S, DecodemXvRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 2);
    if (!Check(S, DecodemQQXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 294:
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 17, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (!Check(S, DecodemXvRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 4);
    if (!Check(S, DecodemXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 295:
    tmp = fieldFromInstruction(insn, 17, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 17, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 2);
    if (!Check(S, DecodeeYsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 2);
    if (!Check(S, DecodemQQXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 296:
    tmp = fieldFromInstruction(insn, 17, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 17, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (!Check(S, DecodemXvRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 2);
    if (!Check(S, DecodemQQXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 297:
    tmp = fieldFromInstruction(insn, 17, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 17, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (!Check(S, DecodemXvRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 4);
    if (!Check(S, DecodemXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 298:
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodeeCMRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 299:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 300:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 301:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 2);
    if (!Check(S, DecodeeYsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 2);
    if (!Check(S, DecodemQQXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 302:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 2);
    if (!Check(S, DecodeeYsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 2);
    if (!Check(S, DecodemQQXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 303:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (!Check(S, DecodemXvRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 2);
    if (!Check(S, DecodemQQXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 304:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (!Check(S, DecodemXvRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 2);
    if (!Check(S, DecodemQQXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 305:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (!Check(S, DecodemXvRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 4);
    if (!Check(S, DecodemXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 306:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (!Check(S, DecodemXvRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 4);
    if (!Check(S, DecodemXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 307:
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 308:
    tmp = fieldFromInstruction(insn, 16, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 2);
    if (!Check(S, DecodeeYsRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 2);
    if (!Check(S, DecodemQQXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 309:
    tmp = fieldFromInstruction(insn, 16, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (!Check(S, DecodemXvRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 2);
    if (!Check(S, DecodemQQXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 310:
    tmp = fieldFromInstruction(insn, 16, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 3, 5);
    if (!Check(S, DecodemBMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (!Check(S, DecodemXvRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 8, 4);
    if (!Check(S, DecodemXwRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  case 311:
    tmp = fieldFromInstruction(insn, 4, 4);
    if (!Check(S, DecodemCMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 17, 4);
    if (!Check(S, DecodemCMaRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 21, 5);
    if (!Check(S, DecodeeRRegisterClass(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
    return S;
  }
}

template <typename InsnType>
static DecodeStatus decodeInstruction(const uint8_t DecodeTable[], MCInst &MI,
                                      InsnType insn, uint64_t Address,
                                      const MCDisassembler *DisAsm,
                                      const MCSubtargetInfo &STI) {
  const FeatureBitset &Bits = STI.getFeatureBits();

  const uint8_t *Ptr = DecodeTable;
  uint64_t CurFieldValue = 0;
  DecodeStatus S = MCDisassembler::Success;
  while (true) {
    ptrdiff_t Loc = Ptr - DecodeTable;
    switch (*Ptr) {
    default:
      errs() << Loc << ": Unexpected decode table opcode!\n";
      return MCDisassembler::Fail;
    case MCD::OPC_ExtractField: {
      // Decode the start value.
      unsigned Start = decodeULEB128AndIncUnsafe(++Ptr);
      unsigned Len = *Ptr++;
      CurFieldValue = fieldFromInstruction(insn, Start, Len);
      LLVM_DEBUG(dbgs() << Loc << ": OPC_ExtractField(" << Start << ", "
                   << Len << "): " << CurFieldValue << "\n");
      break;
    }
    case MCD::OPC_FilterValue: {
      // Decode the field value.
      uint64_t Val = decodeULEB128AndIncUnsafe(++Ptr);
      // NumToSkip is a plain 24-bit integer.
      unsigned NumToSkip = *Ptr++;
      NumToSkip |= (*Ptr++) << 8;
      NumToSkip |= (*Ptr++) << 16;

      // Perform the filter operation.
      if (Val != CurFieldValue)
        Ptr += NumToSkip;
      LLVM_DEBUG(dbgs() << Loc << ": OPC_FilterValue(" << Val << ", " << NumToSkip
                   << "): " << ((Val != CurFieldValue) ? "FAIL:" : "PASS:")
                   << " continuing at " << (Ptr - DecodeTable) << "\n");

      break;
    }
    case MCD::OPC_CheckField: {
      // Decode the start value.
      unsigned Start = decodeULEB128AndIncUnsafe(++Ptr);
      unsigned Len = *Ptr;
      uint64_t FieldValue = fieldFromInstruction(insn, Start, Len);
      // Decode the field value.
      unsigned PtrLen = 0;
      uint64_t ExpectedValue = decodeULEB128(++Ptr, &PtrLen);
      Ptr += PtrLen;
      // NumToSkip is a plain 24-bit integer.
      unsigned NumToSkip = *Ptr++;
      NumToSkip |= (*Ptr++) << 8;
      NumToSkip |= (*Ptr++) << 16;

      // If the actual and expected values don't match, skip.
      if (ExpectedValue != FieldValue)
        Ptr += NumToSkip;
      LLVM_DEBUG(dbgs() << Loc << ": OPC_CheckField(" << Start << ", "
                   << Len << ", " << ExpectedValue << ", " << NumToSkip
                   << "): FieldValue = " << FieldValue << ", ExpectedValue = "
                   << ExpectedValue << ": "
                   << ((ExpectedValue == FieldValue) ? "PASS\n" : "FAIL\n"));
      break;
    }
    case MCD::OPC_CheckPredicate: {
      // Decode the Predicate Index value.
      unsigned PIdx = decodeULEB128AndIncUnsafe(++Ptr);
      // NumToSkip is a plain 24-bit integer.
      unsigned NumToSkip = *Ptr++;
      NumToSkip |= (*Ptr++) << 8;
      NumToSkip |= (*Ptr++) << 16;
      // Check the predicate.
      bool Pred;
      if (!(Pred = checkDecoderPredicate(PIdx, Bits)))
        Ptr += NumToSkip;
      (void)Pred;
      LLVM_DEBUG(dbgs() << Loc << ": OPC_CheckPredicate(" << PIdx << "): "
            << (Pred ? "PASS\n" : "FAIL\n"));

      break;
    }
    case MCD::OPC_Decode: {
      // Decode the Opcode value.
      unsigned Opc = decodeULEB128AndIncUnsafe(++Ptr);
      unsigned DecodeIdx = decodeULEB128AndIncUnsafe(Ptr);

      MI.clear();
      MI.setOpcode(Opc);
      bool DecodeComplete;
      S = decodeToMCInst(S, DecodeIdx, insn, MI, Address, DisAsm, DecodeComplete);
      assert(DecodeComplete);

      LLVM_DEBUG(dbgs() << Loc << ": OPC_Decode: opcode " << Opc
                   << ", using decoder " << DecodeIdx << ": "
                   << (S != MCDisassembler::Fail ? "PASS" : "FAIL") << "\n");
      return S;
    }
    case MCD::OPC_TryDecode: {
      // Decode the Opcode value.
      unsigned Opc = decodeULEB128AndIncUnsafe(++Ptr);
      unsigned DecodeIdx = decodeULEB128AndIncUnsafe(Ptr);
      // NumToSkip is a plain 24-bit integer.
      unsigned NumToSkip = *Ptr++;
      NumToSkip |= (*Ptr++) << 8;
      NumToSkip |= (*Ptr++) << 16;

      // Perform the decode operation.
      MCInst TmpMI;
      TmpMI.setOpcode(Opc);
      bool DecodeComplete;
      S = decodeToMCInst(S, DecodeIdx, insn, TmpMI, Address, DisAsm, DecodeComplete);
      LLVM_DEBUG(dbgs() << Loc << ": OPC_TryDecode: opcode " << Opc
                   << ", using decoder " << DecodeIdx << ": ");

      if (DecodeComplete) {
        // Decoding complete.
        LLVM_DEBUG(dbgs() << (S != MCDisassembler::Fail ? "PASS" : "FAIL") << "\n");
        MI = TmpMI;
        return S;
      } else {
        assert(S == MCDisassembler::Fail);
        // If the decoding was incomplete, skip.
        Ptr += NumToSkip;
        LLVM_DEBUG(dbgs() << "FAIL: continuing at " << (Ptr - DecodeTable) << "\n");
        // Reset decode status. This also drops a SoftFail status that could be
        // set before the decode attempt.
        S = MCDisassembler::Success;
      }
      break;
    }
    case MCD::OPC_SoftFail: {
      // Decode the mask values.
      uint64_t PositiveMask = decodeULEB128AndIncUnsafe(++Ptr);
      uint64_t NegativeMask = decodeULEB128AndIncUnsafe(Ptr);
      bool Fail = (insn & PositiveMask) != 0 || (~insn & NegativeMask) != 0;
      if (Fail)
        S = MCDisassembler::SoftFail;
      LLVM_DEBUG(dbgs() << Loc << ": OPC_SoftFail: " << (Fail ? "FAIL\n" : "PASS\n"));
      break;
    }
    case MCD::OPC_Fail: {
      LLVM_DEBUG(dbgs() << Loc << ": OPC_Fail\n");
      return MCDisassembler::Fail;
    }
    }
  }
  llvm_unreachable("bogosity detected in disassembler state machine!");
}


} // end namespace llvm
