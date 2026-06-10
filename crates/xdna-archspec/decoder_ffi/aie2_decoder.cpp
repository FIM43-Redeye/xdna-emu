// aie2_decoder.cpp -- Thin C wrapper around LLVM's AIE2 MCDisassembler.
//
// Links against llvm-aie's disassembler library to get perfect instruction
// decoding, including TRY_DECODE disambiguation for overlapping encodings.
//
// Per-slot decoding works by constructing a minimal synthetic VLIW bundle
// (32-bit or 48-bit) containing only the target slot, then calling LLVM's
// getInstruction() which dispatches through the format tables to the
// per-slot decoder tables (DecoderTableMv32, etc.).  The result is a
// bundle-level MCInst with the slot instruction as a nested operand.
//
// Bundle encoding is derived from AIE2CompositeFormats.td:
//   32-bit: Inst = {instr32[27:0], 0b1001}  (format code 9)
//   48-bit: Inst = {instr48[44:0], 0b101}   (format code 5)
//   16-bit: Inst = {nop16, dontcare[11:1], 0b0001}  (format code 1)

#include "aie2_decoder.h"

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Triple.h"

// CodeGen headers for register-aware schedule-class resolution. These pull in
// AIE2InstrInfo (a TargetInstrInfo subclass) whose getSchedClass(desc,
// operands) applies the TableGen ItineraryRegPairs override -- the same
// resolution LLVM's scheduler uses. Provided by libLLVMAIECodeGen (already on
// the link line via `llvm-config --libs aie`). clangd may flag these as
// missing in-editor; they exist at build time.
#include "AIE2InstrInfo.h"
#include "AIEBaseInstrInfo.h"

#include <cstring>
#include <mutex>

using namespace llvm;

// Lazy-initialized disassembler state (thread-safe via once_flag).
// The decode mutex serializes getInstruction() calls because MCContext's
// bump allocator (used by createMCInst) is not thread-safe.
static std::once_flag g_init_flag;
static std::mutex g_decode_mutex;
static const Target *g_target = nullptr;
static MCSubtargetInfo *g_sti = nullptr;
static MCRegisterInfo *g_mri = nullptr;
static MCAsmInfo *g_mai = nullptr;
static MCContext *g_ctx = nullptr;
static MCDisassembler *g_disasm = nullptr;
static MCInstrInfo *g_mii = nullptr;
static InstrItineraryData g_iid;  // Legacy itinerary data for scheduling queries.
// CodeGen InstrInfo, used only for register-aware schedule-class resolution
// (getSchedClass(desc, operands)). Constructed once; never owns target state.
static AIE2InstrInfo *g_tii = nullptr;

static void init_disassembler() {
    // Register the AIE2 target (the llvm-aie build provides this).
    LLVMInitializeAIETargetInfo();
    LLVMInitializeAIETargetMC();
    LLVMInitializeAIEDisassembler();

    std::string error;
    g_target = TargetRegistry::lookupTarget("aie2", error);
    if (!g_target)
        return;

    Triple triple("aie2-none-unknown-elf");
    g_mri = g_target->createMCRegInfo(triple.str());
    if (!g_mri) return;

    MCTargetOptions opts;
    g_mai = g_target->createMCAsmInfo(*g_mri, triple.str(), opts);
    if (!g_mai) return;

    g_sti = g_target->createMCSubtargetInfo(triple.str(), "", "");
    if (!g_sti) return;

    g_mii = g_target->createMCInstrInfo();
    if (!g_mii) return;

    g_ctx = new MCContext(triple, g_mai, g_mri, g_sti);
    g_disasm = g_target->createMCDisassembler(*g_sti, *g_ctx);

    // Initialize itinerary data for the AIE2 CPU.
    //
    // createMCSubtargetInfo() with empty CPU/features gives a default model.
    // We need to explicitly request the "aie2" CPU's itineraries, which
    // contain per-instruction latencies via operand_cycles[].
    g_iid = g_sti->getInstrItineraryForCPU("aie2");

    // CodeGen InstrInfo for register-aware schedule-class resolution. The
    // no-arg ctor needs no target state; getSchedClass(desc, operands) reads
    // only the generated variant tables and operand register classes.
    g_tii = new AIE2InstrInfo();
}

// Resolve the register-aware schedule class for a decoded MCInst and fill the
// resolved per-operand itinerary (cycles + forwarding ids) into the result.
//
// For register-pair-variant opcodes (e.g. VMOV_mv_x) the static base class
// reports the wrong bypass; getSchedClass(desc, operands) applies the TableGen
// ItineraryRegPairs override based on the operands' register classes, yielding
// the same resolved class the LLVM scheduler uses. Operands are passed as
// physical register ids (OperandRegInfo{Register(physReg)}), matching the MI
// operand order so OperandRCRequirement.OpIdx lines up.
static void fill_resolved_itinerary(const MCInst &mi, Aie2DecodeResult &result) {
    if (!g_mii || !g_tii || mi.getOpcode() >= g_mii->getNumOpcodes())
        return;

    const MCInstrDesc &desc = g_mii->get(mi.getOpcode());
    result.res_num_defs = (uint8_t)desc.getNumDefs();

    // Build OperandRegInfo per operand from physical register ids. Non-register
    // operands become a default OperandRegInfo (no Reg, no RC), which cannot
    // satisfy any RC requirement -- so the variant falls back to the static
    // class exactly as LLVM does.
    SmallVector<OperandRegInfo, 8> ops;
    for (unsigned i = 0; i < mi.getNumOperands(); i++) {
        const MCOperand &op = mi.getOperand(i);
        if (op.isReg() && op.getReg() != 0)
            ops.emplace_back(Register(op.getReg()));
        else
            ops.emplace_back();
    }

    unsigned resolved_class =
        g_tii->getSchedClass(desc, ArrayRef<OperandRegInfo>(ops));

    // Extract resolved per-operand cycles + forwardings exactly as
    // aie2_get_instr_info does for the static class, but indexed at the
    // resolved class's FirstOperandCycle. The resolved class comes from LLVM's
    // own variant tables, so it is a valid itinerary index whenever the
    // itinerary is non-empty (Itineraries != nullptr).
    if (g_iid.isEmpty())
        return;
    const InstrItinerary &itin = g_iid.Itineraries[resolved_class];
    unsigned n = (itin.LastOperandCycle > itin.FirstOperandCycle)
                     ? (itin.LastOperandCycle - itin.FirstOperandCycle)
                     : 0;
    if (n > AIE2_MAX_OPERANDS)
        n = AIE2_MAX_OPERANDS;
    result.res_num_operand_cycles = (uint8_t)n;
    const bool have_cycles = g_iid.OperandCycles != nullptr;
    const bool have_fwd = g_iid.Forwardings != nullptr;
    for (unsigned i = 0; i < n; i++) {
        if (have_cycles)
            result.res_operand_cycle[i] =
                (int16_t)g_iid.OperandCycles[itin.FirstOperandCycle + i];
        if (have_fwd)
            result.res_operand_bypass[i] =
                (uint16_t)g_iid.Forwardings[itin.FirstOperandCycle + i];
    }
}

// Extract the slot-level MCInst from a bundle-level decode result.
// LLVM's format decoders produce a bundle MCInst (e.g., I32_MV) where
// each slot is a nested MCInst attached as an operand.  For single-slot
// bundles there is exactly one nested MCInst operand.
static const MCInst *extract_slot_inst(const MCInst &bundle) {
    for (unsigned i = 0; i < bundle.getNumOperands(); i++) {
        const MCOperand &op = bundle.getOperand(i);
        if (op.isInst() && op.getInst())
            return op.getInst();
    }
    return nullptr;
}

// Convert an MCInst to our C result struct, enriched with register names
// and output count from MCInstrDesc.
static Aie2DecodeResult mcinst_to_result(const MCInst &mi) {
    Aie2DecodeResult result;
    memset(&result, 0, sizeof(result));
    result.success = 1;
    result.opcode = mi.getOpcode();

    // Get MCInstrDesc for output count and operand constraints.
    if (g_mii && mi.getOpcode() < g_mii->getNumOpcodes()) {
        const MCInstrDesc &desc = g_mii->get(mi.getOpcode());
        result.num_defs = desc.getNumDefs();
    }

    result.num_operands = mi.getNumOperands();
    if (result.num_operands > AIE2_MAX_OPERANDS)
        result.num_operands = AIE2_MAX_OPERANDS;

    for (uint32_t i = 0; i < result.num_operands; i++) {
        const MCOperand &op = mi.getOperand(i);
        if (op.isReg()) {
            result.operands[i].kind = AIE2_OP_REG;
            result.operands[i].value = op.getReg();
            // Look up register name from MCRegisterInfo.
            if (g_mri && op.getReg() != 0) {
                result.operands[i].reg_name = g_mri->getName(op.getReg());
            }
        } else if (op.isImm()) {
            result.operands[i].kind = AIE2_OP_IMM;
            result.operands[i].value = op.getImm();
        } else {
            result.operands[i].kind = AIE2_OP_INVALID;
        }
    }

    // Register-aware resolved itinerary (cycles + forwardings). Must run with
    // the slot MCInst's operands available -- resolution is per-instruction.
    fill_resolved_itinerary(mi, result);

    return result;
}

// Build a synthetic 32-bit VLIW bundle for a single slot.
//
// 32-bit bundle layout (from AIE2CompositeFormats.td):
//   Inst[3:0]   = 0b1001 (format code for 32-bit)
//   Inst[31:4]  = instr32[27:0]
//
// Per-slot instr32 layouts:
//   VEC: inst_vec = {vec[25:0], 0b00}
//   ST:  inst_st  = {0b00001, st[20:0], 0b01}
//   MV:  inst_mv  = {0b00011, mv[21:0], 0b1}
//   LDB: inst_ldb = {0b00111, ldb[15:0], 0b0000001}
//   LDA: inst_lda = {0b00000, lda[20:0], 0b01}
//   ALU: inst_alu = {0b00010, alu[19:0], 0b001}
static bool build_bundle_32(Aie2Slot slot, uint64_t insn_bits,
                            uint8_t *buf, size_t *size) {
    uint32_t instr32 = 0;

    switch (slot) {
    case AIE2_SLOT_VEC:
        // inst_vec = {vec[25:0], 0b00}
        instr32 = ((insn_bits & 0x03FFFFFF) << 2) | 0x00;
        break;
    case AIE2_SLOT_ST:
        // inst_st = {0b00001, st[20:0], 0b01}
        instr32 = (0x01 << 23) | ((insn_bits & 0x1FFFFF) << 2) | 0x01;
        break;
    case AIE2_SLOT_MV:
        // inst_mv = {0b00011, mv[21:0], 0b1}
        instr32 = (0x03 << 23) | ((insn_bits & 0x3FFFFF) << 1) | 0x01;
        break;
    case AIE2_SLOT_LDB:
        // inst_ldb = {0b00111, ldb[15:0], 0b0000001}
        instr32 = (0x07 << 23) | ((insn_bits & 0xFFFF) << 7) | 0x01;
        break;
    case AIE2_SLOT_LDA:
        // inst_lda = {0b00000, lda[20:0], 0b01}
        instr32 = (0x00 << 23) | ((insn_bits & 0x1FFFFF) << 2) | 0x01;
        break;
    case AIE2_SLOT_ALU:
        // inst_alu = {0b00010, alu[19:0], 0b001}
        instr32 = (0x02 << 23) | ((insn_bits & 0xFFFFF) << 3) | 0x01;
        break;
    default:
        return false;
    }

    // Assemble full 32-bit bundle: {instr32[27:0], 0b1001}
    uint32_t bundle = (instr32 << 4) | 0x9;

    // Write as little-endian bytes.
    buf[0] = bundle & 0xFF;
    buf[1] = (bundle >> 8) & 0xFF;
    buf[2] = (bundle >> 16) & 0xFF;
    buf[3] = (bundle >> 24) & 0xFF;
    *size = 4;
    return true;
}

// Build a synthetic 48-bit VLIW bundle for LNG slot.
//
// 48-bit bundle layout:
//   Inst[2:0]   = 0b101 (format code for 48-bit)
//   Inst[47:3]  = instr48[44:0]
//
// LNG: inst_lng = {lng[41:0], 0b010}
static bool build_bundle_48_lng(uint64_t insn_bits,
                                uint8_t *buf, size_t *size) {
    // inst_lng = {lng[41:0], 0b010}
    uint64_t instr48 = ((insn_bits & 0x3FFFFFFFFFF) << 3) | 0x02;
    // Full bundle: {instr48[44:0], 0b101}
    uint64_t bundle = (instr48 << 3) | 0x5;

    // Write as 6 bytes little-endian.
    for (int i = 0; i < 6; i++)
        buf[i] = (bundle >> (8 * i)) & 0xFF;
    *size = 6;
    return true;
}

extern "C" {

Aie2DecodeResult aie2_decode_slot(Aie2Slot slot, uint64_t insn_bits) {
    Aie2DecodeResult result;
    memset(&result, 0, sizeof(result));

    std::call_once(g_init_flag, init_disassembler);
    if (!g_disasm)
        return result;

    // Build a synthetic VLIW bundle containing just this slot.
    uint8_t buf[16];
    size_t bundle_size = 0;

    if (slot == AIE2_SLOT_NOP) {
        // NOP is a 16-bit format: {nop16, dontcare[11:1], 0b0001}
        buf[0] = 0x01;  // format code 1 in low nibble
        buf[1] = 0x00;
        bundle_size = 2;
    } else if (slot == AIE2_SLOT_LNG) {
        if (!build_bundle_48_lng(insn_bits, buf, &bundle_size))
            return result;
    } else {
        if (!build_bundle_32(slot, insn_bits, buf, &bundle_size))
            return result;
    }

    // Call LLVM's getInstruction() to decode the bundle.
    // Serialize access: MCContext's allocator is not thread-safe.
    Aie2DecodeResult decoded;
    {
        std::lock_guard<std::mutex> lock(g_decode_mutex);
        MCInst bundle_inst;
        uint64_t decoded_size = 0;
        ArrayRef<uint8_t> bytes(buf, bundle_size);
        raw_null_ostream null_os;

        auto status = g_disasm->getInstruction(bundle_inst, decoded_size,
                                               bytes, 0, null_os);
        if (status == MCDisassembler::Fail)
            return result;

        // Extract the nested slot instruction from the bundle result.
        const MCInst *slot_inst = extract_slot_inst(bundle_inst);
        if (!slot_inst)
            return result;

        decoded = mcinst_to_result(*slot_inst);
    }
    return decoded;
}

const char *aie2_opcode_name(uint32_t opcode) {
    std::call_once(g_init_flag, init_disassembler);
    if (!g_mii || opcode >= g_mii->getNumOpcodes())
        return nullptr;
    return g_mii->getName(opcode).data();
}

const char *aie2_opcode_mnemonic(uint32_t opcode) {
    // MCInstrInfo only stores the TableGen name, not the asm mnemonic.
    // Use aie2_opcode_name() and derive the mnemonic on the Rust side.
    return aie2_opcode_name(opcode);
}

int aie2_decoder_init(void) {
    std::call_once(g_init_flag, init_disassembler);
    return g_disasm ? 1 : 0;
}

uint32_t aie2_get_num_regs(void) {
    std::call_once(g_init_flag, init_disassembler);
    if (!g_mri)
        return 0;
    return g_mri->getNumRegs();
}

const char *aie2_get_reg_name(uint32_t reg_id) {
    std::call_once(g_init_flag, init_disassembler);
    if (!g_mri || reg_id >= g_mri->getNumRegs())
        return nullptr;
    const char *name = g_mri->getName(reg_id);
    if (!name || name[0] == '\0')
        return nullptr;
    return name;
}

uint32_t aie2_get_num_opcodes(void) {
    std::call_once(g_init_flag, init_disassembler);
    if (!g_mii)
        return 0;
    return g_mii->getNumOpcodes();
}

int aie2_get_instr_info(uint32_t opcode, Aie2InstrInfo *out) {
    std::call_once(g_init_flag, init_disassembler);
    if (!g_mii || !g_sti || opcode >= g_mii->getNumOpcodes() || !out)
        return 0;

    memset(out, 0, sizeof(*out));

    const MCInstrDesc &desc = g_mii->get(opcode);
    out->flags = desc.getFlags();
    out->num_operands = desc.getNumOperands();
    out->num_defs = desc.getNumDefs();
    out->latency = -1;
    out->stage_latency = -1;

    // Query the itinerary-based scheduling model.
    unsigned sched_class = desc.getSchedClass();
    out->sched_class = (uint16_t)sched_class;

    // Get operand-level latency from itinerary data.
    // operand_cycles[0] = result availability cycle (the key metric for us).
    if (!g_iid.isEmpty()) {
        auto op0 = g_iid.getOperandCycle(sched_class, 0);
        if (op0.has_value())
            out->latency = (int16_t)op0.value();

        // Also get pipeline stage latency as a fallback.
        unsigned stage_lat = g_iid.getStageLatency(sched_class);
        if (stage_lat > 0)
            out->stage_latency = (int16_t)stage_lat;

        // Per-operand itinerary cycles + forwarding ids. OperandCycles and
        // Forwardings are parallel arrays indexed by FirstOperandCycle + i,
        // matching MI operand order (defs then uses). def_bypass/latency stay
        // as the operand-0 shorthands the producer side already uses.
        //
        // The enclosing !isEmpty() check already proves Itineraries != nullptr,
        // so we don't re-guard it. OperandCycles/Forwardings can still each be
        // null independently, so capture those once outside the loop.
        const InstrItinerary &itin = g_iid.Itineraries[sched_class];
        unsigned n = (itin.LastOperandCycle > itin.FirstOperandCycle)
                         ? (itin.LastOperandCycle - itin.FirstOperandCycle)
                         : 0;
        if (n > AIE2_MAX_OPERANDS)
            n = AIE2_MAX_OPERANDS;
        out->num_operand_cycles = (uint8_t)n;
        const bool have_cycles = g_iid.OperandCycles != nullptr;
        const bool have_fwd = g_iid.Forwardings != nullptr;
        for (unsigned i = 0; i < n; i++) {
            if (have_cycles)
                out->operand_cycle[i] =
                    (int16_t)g_iid.OperandCycles[itin.FirstOperandCycle + i];
            if (have_fwd)
                out->operand_bypass[i] =
                    (uint16_t)g_iid.Forwardings[itin.FirstOperandCycle + i];
        }
        if (n > 0 && have_fwd)
            out->def_bypass = out->operand_bypass[0];
    }

    return 1;
}

} // extern "C"
