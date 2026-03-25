// aie2_decoder.h -- C interface to LLVM's AIE2 instruction decoder.
//
// Provides instruction decoding via LLVM's MCDisassembler, with full
// operand information including register names and output classification.

#ifndef AIE2_DECODER_H
#define AIE2_DECODER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum operands any AIE2 instruction can have.
#define AIE2_MAX_OPERANDS 16

// Operand kinds matching LLVM MCOperand.
enum Aie2OpKind {
    AIE2_OP_INVALID = 0,
    AIE2_OP_REG = 1,
    AIE2_OP_IMM = 2,
};

// A single decoded operand with register metadata.
struct Aie2Operand {
    enum Aie2OpKind kind;
    int64_t value;           // For REG: LLVM register ID.  For IMM: immediate.
    const char *reg_name;    // For REG: e.g. "r0", "bm0", "cm0", "x4".  NULL otherwise.
};

// Result of decoding one slot.
struct Aie2DecodeResult {
    int success;             // 1 = decoded, 0 = failed
    uint32_t opcode;         // LLVM opcode ID (index into MCInstrInfo)
    uint32_t num_operands;
    uint32_t num_defs;       // First num_defs operands are outputs (from MCInstrDesc)
    struct Aie2Operand operands[AIE2_MAX_OPERANDS];
};

// Slot identifiers (matching our slot table order).
enum Aie2Slot {
    AIE2_SLOT_ALU = 0,
    AIE2_SLOT_LDA = 1,
    AIE2_SLOT_LDB = 2,
    AIE2_SLOT_LNG = 3,
    AIE2_SLOT_MV  = 4,
    AIE2_SLOT_ST  = 5,
    AIE2_SLOT_VEC = 6,
    AIE2_SLOT_NOP = 7,
};

// Decode a single VLIW slot.
//
// slot:      which slot's decoder table to use
// insn_bits: the raw bits for this slot (right-aligned)
//
// Returns a Aie2DecodeResult.  Check .success before reading other fields.
// Register operands include reg_name from MCRegisterInfo.
struct Aie2DecodeResult aie2_decode_slot(enum Aie2Slot slot, uint64_t insn_bits);

// Get the LLVM instruction name for an opcode ID (e.g., "VPUSH_HI_32").
// Returns NULL if the opcode is unknown.
const char *aie2_opcode_name(uint32_t opcode);

// Get the assembly mnemonic for an opcode ID (e.g., "vpush.hi.32").
// Returns NULL if the opcode is unknown.
const char *aie2_opcode_mnemonic(uint32_t opcode);

// Initialize the LLVM disassembler.  Must be called before decode/opcode_name.
// Returns 1 on success, 0 on failure.
int aie2_decoder_init(void);

#ifdef __cplusplus
}
#endif

#endif // AIE2_DECODER_H
