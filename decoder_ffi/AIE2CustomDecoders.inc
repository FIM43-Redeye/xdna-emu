// AIE2CustomDecoders.inc -- Custom register decoder functions.
//
// Extracted from llvm-aie/llvm/lib/Target/AIE/Disassembler/AIE2Disassembler.cpp
// lines 220-305.  Source: llvm-aie (Apache-2.0 WITH LLVM-exception).
//
// These decoders handle register classes with non-trivial encoding schemes
// (compound register classes where the encoding bits select between multiple
// register files based on bit patterns).
//
// Prerequisites: the TABLEBASEDDECODER-generated functions must be defined
// before this file is included, since these custom decoders call them.

static DecodeStatus DecodemAluCgRegisterClass(MCInst &Inst, uint64_t RegNo,
                                              uint64_t Address,
                                              const MCDisassembler *Decoder) {
  if ((RegNo & 0b1) == 0b0 &&
      DecodeeRRegisterClass(Inst, RegNo >> 1, Address, Decoder))
    return DecodeStatus::Success;
  else {
    Inst.addOperand(MCOperand::createReg(LC));
    return DecodeStatus::Success;
  }
  return MCDisassembler::Fail;
}

static DecodeStatus
DecodemMvAMWQSrcRegisterClass(MCInst &Inst, uint64_t RegNo, uint64_t Address,
                              const MCDisassembler *Decoder) {
  if ((RegNo & 0b101) == 0b001 &&
      DecodemAMmRegisterClass(Inst, RegNo >> 3, Address, Decoder))
    return DecodeStatus::Success;
  if ((RegNo >> 6) == 0b110 &&
      DecodemWm_1RegisterClass(Inst, RegNo & 0b11111, Address, Decoder))
    return DecodeStatus::Success;
  if ((RegNo >> 6) == 0b111 &&
      DecodemQQmRegisterClass(Inst, (RegNo >> 4) & 0b11, Address, Decoder))
    return DecodeStatus::Success;
  return MCDisassembler::Fail;
}

static DecodeStatus
DecodemMvAMWQDstRegisterClass(MCInst &Inst, uint64_t RegNo, uint64_t Address,
                              const MCDisassembler *Decoder) {
  if ((RegNo & 0b1) == 0b1 &&
      DecodemAMmRegisterClass(Inst, RegNo >> 1, Address, Decoder))
    return DecodeStatus::Success;
  if ((RegNo & 0b11) == 0b00 &&
      DecodemWm_1RegisterClass(Inst, RegNo >> 2, Address, Decoder))
    return DecodeStatus::Success;
  if ((RegNo & 0b11) == 0b10 &&
      DecodemQQmRegisterClass(Inst, RegNo >> 5, Address, Decoder))
    return DecodeStatus::Success;
  return MCDisassembler::Fail;
}

static DecodeStatus
DecodemMvBMXSrcRegisterClass(MCInst &Inst, uint64_t RegNo, uint64_t Address,
                             const MCDisassembler *Decoder) {
  if (DecodemBMmRegisterClass(Inst, RegNo >> 4, Address, Decoder) ==
      DecodeStatus::Success)
    return DecodeStatus::Success;
  if ((RegNo >> 7) == 0b11 &&
      DecodemXmRegisterClass(Inst, RegNo & 0b1111, Address, Decoder) ==
          DecodeStatus::Success)
    return DecodeStatus::Success;
  return MCDisassembler::Fail;
}

static DecodeStatus
DecodemMvBMXDstRegisterClass(MCInst &Inst, uint64_t RegNo, uint64_t Address,
                             const MCDisassembler *Decoder) {
  if ((RegNo & 0b1) == 0b1 &&
      DecodemBMmRegisterClass(Inst, RegNo >> 1, Address, Decoder) ==
          DecodeStatus::Success)
    return DecodeStatus::Success;
  if ((RegNo & 0b1) == 0b0 &&
      DecodemXmRegisterClass(Inst, RegNo >> 2, Address, Decoder) ==
          DecodeStatus::Success)
    return DecodeStatus::Success;
  return MCDisassembler::Fail;
}

static DecodeStatus DecodemShflDstRegisterClass(MCInst &Inst, uint64_t RegNo,
                                                uint64_t Address,
                                                const MCDisassembler *Decoder) {
  if ((RegNo & 0b1) == 0b1) {
    RegNo = RegNo >> 1;
    if (DecodemBMSmRegisterClass(Inst, RegNo, Address, Decoder) ==
        DecodeStatus::Success) {
      return DecodeStatus::Success;
    }
  }
  if ((RegNo & 0b1) == 0b0 &&
      DecodemXmRegisterClass(Inst, RegNo >> 1, Address, Decoder) ==
          DecodeStatus::Success)
    return DecodeStatus::Success;
  return MCDisassembler::Fail;
}
