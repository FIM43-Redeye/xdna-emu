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

    // ── Register-aware resolved itinerary ──────────────────────────────
    // For register-pair-variant opcodes (e.g. VMOV_mv_x) the correct
    // forwarding/bypass depends on the actual operand register classes, not
    // the static base schedule class. These fields carry the per-operand
    // itinerary RESOLVED via AIE2InstrInfo::getSchedClass(desc, operands),
    // matching what LLVM's scheduler uses. Indexed by MI operand position
    // (defs then uses), valid for i < res_num_operand_cycles.
    uint8_t res_num_defs;                          // = desc.getNumDefs()
    uint8_t res_num_operand_cycles;                // valid entries below
    int16_t res_operand_cycle[AIE2_MAX_OPERANDS];  // resolved operand cycles
    uint16_t res_operand_bypass[AIE2_MAX_OPERANDS]; // resolved forwarding ids
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

// ── Instruction metadata (bulk-queryable at init time) ──────────────────

// Per-instruction metadata from MCInstrDesc + itinerary scheduling model.
struct Aie2InstrInfo {
    uint64_t flags;          // MCID flags bitmask (MayLoad, MayStore, isBranch, etc.)
    uint16_t num_operands;   // Total operand count
    uint8_t num_defs;        // Number of output (def) operands
    int16_t latency;         // Result latency from itinerary operand_cycles[0], or -1
    int16_t stage_latency;   // Total pipeline latency from InstrStage sum, or -1
    uint16_t sched_class;    // Itinerary class index (opaque; for cross-ref with build data)
    uint16_t def_bypass;     // Forwarding-network id of result operand 0 (0 = NoBypass)
    // Per-operand itinerary data, indexed by MI operand position (defs then
    // uses). operand_cycle[i] is the cycle operand i is read/produced;
    // operand_bypass[i] its forwarding id. Valid for i < num_operand_cycles.
    int16_t operand_cycle[AIE2_MAX_OPERANDS];
    uint16_t operand_bypass[AIE2_MAX_OPERANDS];
    uint8_t num_operand_cycles;
};

// MCID flag bit positions (from llvm/MC/MCInstrDesc.h MCID::Flag enum).
// These are bit indices -- test with (flags & (1ULL << BIT)).
#define AIE2_MCID_RETURN          5
#define AIE2_MCID_CALL            7
#define AIE2_MCID_BARRIER         8
#define AIE2_MCID_TERMINATOR      9
#define AIE2_MCID_BRANCH         10
#define AIE2_MCID_INDIRECT_BRANCH 11
#define AIE2_MCID_COMPARE        12
#define AIE2_MCID_MOVE_IMM       13
#define AIE2_MCID_MOVE_REG       14
#define AIE2_MCID_MAY_LOAD       19
#define AIE2_MCID_MAY_STORE      20
#define AIE2_MCID_COMMUTABLE     25
#define AIE2_MCID_ADD            37

// Get the total number of registers known to MCRegisterInfo.
// Register IDs range from 1..num_regs (0 is NoRegister).
uint32_t aie2_get_num_regs(void);

// Get the name of a register by its LLVM register ID.
// Returns NULL if the ID is out of range or the name is empty.
const char *aie2_get_reg_name(uint32_t reg_id);

// Get the total number of opcodes known to LLVM's MCInstrInfo.
uint32_t aie2_get_num_opcodes(void);

// Query instruction metadata for a single opcode.
// Returns 1 on success, 0 if opcode is out of range or decoder not initialized.
int aie2_get_instr_info(uint32_t opcode, struct Aie2InstrInfo *out);

#ifdef __cplusplus
}
#endif

#endif // AIE2_DECODER_H
