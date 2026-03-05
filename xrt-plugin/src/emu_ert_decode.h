// SPDX-License-Identifier: MIT
//
// emu_ert_decode.h -- ERT packet pretty-printer for diagnostic logging.
//
// Header-only.  Decodes ert_start_kernel_cmd packets and logs their
// contents via EMU_DBG.  Understands ERT_START_NPU and ERT_START_CU
// opcodes and their payload layouts.

#pragma once

#include "emu_debug.h"
#include "core/include/ert.h"

#include <cinttypes>

namespace xdna_emu {
namespace detail {

inline const char* ert_opcode_name(unsigned op)
{
    switch (op) {
    case ERT_START_CU:              return "START_CU";
    case ERT_CONFIGURE:             return "CONFIGURE";
    case ERT_START_DPU:             return "START_DPU";
    case ERT_CMD_CHAIN:             return "CMD_CHAIN";
    case ERT_START_NPU:             return "START_NPU";
    case ERT_START_NPU_PREEMPT:     return "START_NPU_PREEMPT";
    case ERT_START_NPU_PREEMPT_ELF: return "START_NPU_PREEMPT_ELF";
    case ERT_SK_START:              return "SK_START";
    case ERT_SK_CONFIG:             return "SK_CONFIG";
    default:                        return "UNKNOWN";
    }
}

inline void emu_log_ert_packet(const struct ert_start_kernel_cmd* pkt,
                               const char* context)
{
    EMU_DBG("%s: ERT packet: opcode=%s(%u) count=%u extra_cu_masks=%u state=%u",
            context,
            ert_opcode_name(pkt->opcode), pkt->opcode,
            pkt->count, pkt->extra_cu_masks, pkt->state);

    // Decode ERT_START_NPU payload.
    if (pkt->opcode == ERT_START_NPU) {
        auto* npu = get_ert_npu_data(
            const_cast<struct ert_start_kernel_cmd*>(pkt));
        if (npu) {
            EMU_DBG("  START_NPU: instr_buf=0x%" PRIx64 " instr_size=%u prop_count=%u",
                    npu->instruction_buffer,
                    npu->instruction_buffer_size,
                    npu->instruction_prop_count);
        }
        return;
    }

    // Decode ERT_START_DPU payload.
    if (pkt->opcode == ERT_START_DPU) {
        auto* dpu = get_ert_dpu_data(
            const_cast<struct ert_start_kernel_cmd*>(pkt));
        if (dpu) {
            EMU_DBG("  START_DPU: instr_buf=0x%" PRIx64 " instr_size=%u"
                    " dtrace=0x%" PRIx64 " chained=%u",
                    dpu->instruction_buffer,
                    dpu->instruction_buffer_size,
                    dpu->dtrace_buffer,
                    dpu->chained);
        }
        return;
    }

    // Decode ERT_START_NPU_PREEMPT / ERT_START_NPU_PREEMPT_ELF payload.
    if (pkt->opcode == ERT_START_NPU_PREEMPT ||
        pkt->opcode == ERT_START_NPU_PREEMPT_ELF) {
        auto* pre = get_ert_npu_preempt_data(
            const_cast<struct ert_start_kernel_cmd*>(pkt));
        if (!pre)
            pre = get_ert_npu_elf_data(
                const_cast<struct ert_start_kernel_cmd*>(pkt));
        if (pre) {
            EMU_DBG("  START_NPU_PREEMPT: instr_buf=0x%" PRIx64
                    " instr_size=%u save=0x%" PRIx64 "/%u"
                    " restore=0x%" PRIx64 "/%u prop_count=%u",
                    pre->instruction_buffer,
                    pre->instruction_buffer_size,
                    pre->save_buffer, pre->save_buffer_size,
                    pre->restore_buffer, pre->restore_buffer_size,
                    pre->instruction_prop_count);
        }
        return;
    }

    // Decode ERT_CMD_CHAIN payload.
    if (pkt->opcode == ERT_CMD_CHAIN) {
        auto* chain = get_ert_cmd_chain_data(
            reinterpret_cast<struct ert_packet*>(
                const_cast<struct ert_start_kernel_cmd*>(pkt)));
        if (chain) {
            EMU_DBG("  CMD_CHAIN: command_count=%u submit_index=%u"
                    " error_index=%u",
                    chain->command_count, chain->submit_index,
                    chain->error_index);
        }
        return;
    }

    // Decode ERT_START_CU register map.
    if (pkt->opcode == ERT_START_CU) {
        auto* regmap = reinterpret_cast<const uint8_t*>(
            pkt->data + pkt->extra_cu_masks);
        // Standard layout: arg0 "opcode" at +0x00, arg1 "instr" at +0x08,
        // arg2 "ninstr" at +0x10, arg3-5 BOs at +0x14/+0x1c/+0x24.
        uint64_t instr_addr = 0;
        uint32_t ninstr = 0;
        std::memcpy(&instr_addr, regmap + 0x08, sizeof(instr_addr));
        std::memcpy(&ninstr, regmap + 0x10, sizeof(ninstr));
        EMU_DBG("  START_CU: instr_addr=0x%" PRIx64 " ninstr=%u (%u bytes)",
                instr_addr, ninstr, ninstr * 4);

        // Log BO addresses if the packet is large enough.
        uint32_t regmap_bytes = (pkt->count - pkt->extra_cu_masks) * 4;
        if (regmap_bytes >= 0x20) {
            uint64_t bo0 = 0, bo1 = 0, bo2 = 0;
            std::memcpy(&bo0, regmap + 0x14, sizeof(bo0));
            if (regmap_bytes >= 0x24)
                std::memcpy(&bo1, regmap + 0x1c, sizeof(bo1));
            if (regmap_bytes >= 0x2c)
                std::memcpy(&bo2, regmap + 0x24, sizeof(bo2));
            EMU_DBG("  START_CU: bo0=0x%" PRIx64 " bo1=0x%" PRIx64
                    " bo2=0x%" PRIx64, bo0, bo1, bo2);
        }
    }
}

} // namespace detail
} // namespace xdna_emu
