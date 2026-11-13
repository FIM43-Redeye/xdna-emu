// insts_bounds.h -- host-buffer bounds of a runtime sequence's shim DMA tasks.
//
// Decodes an insts.bin transaction stream completely and evaluates every shim
// DMA task it pushes to the byte range that task reaches in its host buffer.
// The runner uses the ranges twice: to size each data buffer (the largest end
// over every task patched onto it) and, before anything is submitted, to refuse
// a run whose buffer descriptors leave the buffers it actually allocated.
//
// The decode is strict: an opcode it does not know, an op size that does not
// fit, or a stream whose ops do not end exactly where the header says is an
// error, never a shorter answer. A walker that stops at the first unfamiliar
// op silently ignores every BD after it; that is precisely how a buffer ends
// up smaller than the tasks that write it.
//
// What is modelled (AIE2 shim tile, row 0):
//  - BD registers DMA_BD0_0 .. DMA_BD15_7 (base 0x1D000, 8 words per BD,
//    stride 0x20), written by Write32 / BlockWrite / MaskWrite.
//  - DdrPatch ops, bound to a BD by their register address, which must be a
//    BD's Base_Address_Low word (BD base + 4). The device address becomes
//    host buffer[arg_idx] + arg_plus; a later write to the BD's address words
//    unbinds it.
//  - Task_Queue writes (DMA_S2MM_0/1_Task_Queue at 0x1D204 / 0x1D20C,
//    DMA_MM2S_0/1_Task_Queue at 0x1D214 / 0x1D21C): Start_BD_ID[3:0], then the
//    Use_Next_BD / Next_BD chain. A task whose BD carries no patch is an error.
//  - The BD's address generator: Buffer_Length (words), D0/D1 wrap and step,
//    D2 step, and the iteration step/wrap (bounded conservatively at its full
//    wrap). Steps are encoded actual - 1; a wrap of 0 means "do not wrap".
// Field positions are those of the AM025 register database
// (mlir-aie lib/Dialect/AIE/Util/aie_registers_aie2.json, module "shim");
// tools/test_bridge_runner_bounds.py checks every constant below against it.
// Opcodes: aie-rt driver/src/common/xaie_txn.h (XAie_TxnOpcode).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace insts_bounds {

struct DecodeError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// One shim DMA task's reach into one host buffer: bytes [start, end) from
// the buffer's base. `op_index` is the Task_Queue write that pushed it.
struct BdAccess {
    size_t op_index;
    uint32_t col;
    uint32_t row;
    const char* channel;
    uint32_t bd;
    int arg_idx;
    uint64_t start;
    uint64_t end;
};

namespace detail {

constexpr uint32_t INSTS_MAGIC = 0x06030100u;
constexpr uint32_t HEADER_BYTES = 16;

// XAie_TxnOpcode.
constexpr uint8_t OP_WRITE = 0x00;
constexpr uint8_t OP_BLOCKWRITE = 0x01;
constexpr uint8_t OP_MASKWRITE = 0x03;
constexpr uint8_t OP_MASKPOLL = 0x04;
constexpr uint8_t OP_TCT = 0x80;        // XAIE_IO_CUSTOM_OP_TCT: the task-complete-token sync
constexpr uint8_t OP_DDR_PATCH = 0x81;  // XAIE_IO_CUSTOM_OP_DDR_PATCH
constexpr uint32_t WRITE_BYTES = 24;
constexpr uint32_t MASK_OP_BYTES = 28;
constexpr uint32_t DDR_PATCH_BYTES = 48;

// Shim tile register map (AM025 "shim" module).
constexpr uint32_t SHIM_ROW = 0;
constexpr uint32_t SHIM_BD_BASE = 0x1D000;   // DMA_BD0_0
constexpr uint32_t SHIM_BD_STRIDE = 0x20;    // DMA_BD1_0 - DMA_BD0_0
constexpr uint32_t SHIM_BD_WORDS = 8;
constexpr uint32_t SHIM_NUM_BDS = 16;
constexpr uint32_t SHIM_S2MM_0_TASK_QUEUE = 0x1D204;
constexpr uint32_t SHIM_S2MM_1_TASK_QUEUE = 0x1D20C;
constexpr uint32_t SHIM_MM2S_0_TASK_QUEUE = 0x1D214;
constexpr uint32_t SHIM_MM2S_1_TASK_QUEUE = 0x1D21C;

// Field (shift, width) by BD word.
struct Field { uint32_t word, shift, width; };
constexpr Field BUFFER_LENGTH{0, 0, 32};
constexpr Field D0_WRAP{3, 20, 10};
constexpr Field D0_STEPSIZE{3, 0, 20};
constexpr Field D1_WRAP{4, 20, 10};
constexpr Field D1_STEPSIZE{4, 0, 20};
constexpr Field D2_STEPSIZE{5, 0, 20};
constexpr Field ITERATION_WRAP{6, 20, 6};
constexpr Field ITERATION_STEPSIZE{6, 0, 20};
constexpr Field USE_NEXT_BD{7, 26, 1};
constexpr Field NEXT_BD{7, 27, 4};
constexpr Field START_BD_ID{0, 0, 4};  // in the Task_Queue value
constexpr uint32_t BASE_ADDRESS_LOW_WORD = 1;
constexpr uint32_t BASE_ADDRESS_HIGH_WORD = 2;

// mlir-aie include/aie/Runtime/TxnEncoding.h: patches of args at index >= 5
// may carry the DDR aperture offset folded into arg_plus; it is not extent.
constexpr uint64_t DDR_AIE_ADDR_OFFSET = 0x80000000ull;
constexpr int NUM_FIRMWARE_TRANSLATED_ARGS = 5;

inline uint32_t field(const uint32_t* bd, Field f) {
    uint64_t mask = (f.width >= 32) ? 0xFFFFFFFFull : ((1ull << f.width) - 1);
    return static_cast<uint32_t>((bd[f.word] >> f.shift) & mask);
}

inline const char* queue_channel(uint32_t reg) {
    switch (reg) {
        case SHIM_S2MM_0_TASK_QUEUE: return "S2MM0";
        case SHIM_S2MM_1_TASK_QUEUE: return "S2MM1";
        case SHIM_MM2S_0_TASK_QUEUE: return "MM2S0";
        case SHIM_MM2S_1_TASK_QUEUE: return "MM2S1";
        default: return nullptr;
    }
}

}  // namespace detail

/// Bytes one execution of a shim BD reaches from its base address: the
/// largest word offset its address generator produces, plus one word, plus
/// the iteration offset at the iteration wrap (an upper bound: whether every
/// iteration is reached depends on repeat counts and earlier executions).
inline uint64_t bd_extent_bytes(const uint32_t* bd) {
    using namespace detail;
    const uint64_t len = field(bd, BUFFER_LENGTH);
    if (len == 0) return 0;
    const uint64_t w0 = field(bd, D0_WRAP), s0 = field(bd, D0_STEPSIZE) + 1ull;
    const uint64_t w1 = field(bd, D1_WRAP), s1 = field(bd, D1_STEPSIZE) + 1ull;
    const uint64_t s2 = field(bd, D2_STEPSIZE) + 1ull;
    const uint64_t k = len - 1;  // last word index
    uint64_t best;
    // Word k's offset is i0*s0 + i1*s1 + i2*s2 with (i2, i1, i0) the mixed-radix
    // digits of k. All steps are positive, so the largest offset over words
    // 0..k is at k itself or at the last word before one of k's digits drops.
    if (w0 == 0) {
        best = k * s0;
    } else if (w1 == 0) {
        uint64_t a1 = k / w0, a0 = k % w0;
        best = a1 * s1 + a0 * s0;
        if (a1 > 0) best = std::max(best, (a1 - 1) * s1 + (w0 - 1) * s0);
    } else {
        uint64_t a2 = k / (w0 * w1), r = k % (w0 * w1);
        uint64_t a1 = r / w0, a0 = r % w0;
        best = a2 * s2 + a1 * s1 + a0 * s0;
        if (a1 > 0) best = std::max(best, a2 * s2 + (a1 - 1) * s1 + (w0 - 1) * s0);
        if (a2 > 0) best = std::max(best, (a2 - 1) * s2 + (w1 - 1) * s1 + (w0 - 1) * s0);
    }
    const uint64_t iter_wrap = field(bd, ITERATION_WRAP) + 1ull;
    const uint64_t iter_step = field(bd, ITERATION_STEPSIZE) + 1ull;
    const uint64_t iter = iter_wrap > 1 ? (iter_wrap - 1) * iter_step : 0;
    return (best + iter + 1) * 4;
}

/// Every shim DMA task the stream pushes, as a byte range in the host buffer
/// its BD was patched onto. Throws DecodeError if the stream cannot be decoded
/// completely, or pushes a task whose BD has no host-buffer patch.
inline std::vector<BdAccess> shim_bd_accesses(const uint8_t* data, size_t size) {
    using namespace detail;
    auto u32 = [&](size_t o) -> uint32_t {
        return static_cast<uint32_t>(data[o]) | (static_cast<uint32_t>(data[o + 1]) << 8)
             | (static_cast<uint32_t>(data[o + 2]) << 16) | (static_cast<uint32_t>(data[o + 3]) << 24);
    };
    if (size < HEADER_BYTES || u32(0) != INSTS_MAGIC)
        throw DecodeError("not an instruction stream (bad magic or short header)");
    const uint32_t num_ops = u32(8);
    const uint32_t total = u32(12);
    if (total != size)
        throw DecodeError("header says " + std::to_string(total) + " bytes, stream has " + std::to_string(size));

    struct Patch { int arg_idx; uint64_t arg_plus; };
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> bd_regs;  // (col,row) -> 16*8 words
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Patch> patches;     // (col,row,bd)
    std::vector<BdAccess> out;

    auto tile_bds = [&](uint32_t col, uint32_t row) -> std::vector<uint32_t>& {
        auto& v = bd_regs[{col, row}];
        if (v.empty()) v.assign(SHIM_NUM_BDS * SHIM_BD_WORDS, 0);
        return v;
    };
    auto is_bd_reg = [](uint32_t row, uint32_t reg) {
        return row == SHIM_ROW && reg >= SHIM_BD_BASE && reg < SHIM_BD_BASE + SHIM_NUM_BDS * SHIM_BD_STRIDE;
    };

    // Apply one register write; a Task_Queue write evaluates the task it pushes.
    auto write_reg = [&](size_t op_index, uint32_t addr, uint32_t value, uint32_t mask) {
        const uint32_t col = (addr >> 25) & 0x7F, row = (addr >> 20) & 0x1F, reg = addr & 0xFFFFF;
        if (is_bd_reg(row, reg)) {
            const uint32_t bd = (reg - SHIM_BD_BASE) / SHIM_BD_STRIDE;
            const uint32_t word = ((reg - SHIM_BD_BASE) % SHIM_BD_STRIDE) / 4;
            uint32_t& w = tile_bds(col, row)[bd * SHIM_BD_WORDS + word];
            w = (w & ~mask) | (value & mask);
            if (word == BASE_ADDRESS_LOW_WORD || word == BASE_ADDRESS_HIGH_WORD) {
                // Base_Address_High shares its word with packet fields: only a
                // write that touches the address bits replaces the address.
                if (word == BASE_ADDRESS_LOW_WORD || (mask & 0xFFFFu) != 0) patches.erase({col, row, bd});
            }
            return;
        }
        const char* channel = (row == SHIM_ROW) ? queue_channel(reg) : nullptr;
        if (!channel) return;
        if (mask != 0xFFFFFFFFu)
            throw DecodeError("op " + std::to_string(op_index) + ": masked write to a Task_Queue register");
        const auto& regs = tile_bds(col, row);
        uint32_t bd = value & ((1u << START_BD_ID.width) - 1);
        std::set<uint32_t> seen;
        while (seen.insert(bd).second) {
            const uint32_t* w = &regs[bd * SHIM_BD_WORDS];
            auto it = patches.find({col, row, bd});
            if (it == patches.end())
                throw DecodeError("op " + std::to_string(op_index) + ": " + channel + " of shim column "
                                  + std::to_string(col) + " pushes BD " + std::to_string(bd)
                                  + ", which carries no host-buffer patch");
            out.push_back(BdAccess{op_index, col, row, channel, bd, it->second.arg_idx, it->second.arg_plus,
                                   it->second.arg_plus + bd_extent_bytes(w)});
            if (!field(w, USE_NEXT_BD)) break;
            bd = field(w, NEXT_BD);
        }
    };

    size_t off = HEADER_BYTES;
    for (uint32_t i = 0; i < num_ops; ++i) {
        const std::string where = "op " + std::to_string(i) + " at byte " + std::to_string(off);
        if (off + 8 > total) throw DecodeError(where + ": stream ends inside the op header");
        const uint8_t opcode = data[off];
        uint32_t op_bytes;
        switch (opcode) {
            case OP_WRITE:
                op_bytes = WRITE_BYTES;
                if (off + op_bytes > total) throw DecodeError(where + ": truncated Write32");
                write_reg(i, u32(off + 8), u32(off + 16), 0xFFFFFFFFu);
                break;
            case OP_BLOCKWRITE: {
                if (off + 16 > total) throw DecodeError(where + ": truncated BlockWrite header");
                op_bytes = u32(off + 12);
                if (op_bytes < 16 || op_bytes % 4 != 0 || off + op_bytes > total)
                    throw DecodeError(where + ": BlockWrite size " + std::to_string(op_bytes));
                const uint32_t addr = u32(off + 8);
                for (uint32_t k = 0; k < (op_bytes - 16) / 4; ++k)
                    write_reg(i, addr + 4 * k, u32(off + 16 + 4 * k), 0xFFFFFFFFu);
                break;
            }
            case OP_MASKWRITE:
                op_bytes = MASK_OP_BYTES;
                if (off + op_bytes > total) throw DecodeError(where + ": truncated MaskWrite");
                write_reg(i, u32(off + 8), u32(off + 16), u32(off + 20));
                break;
            case OP_MASKPOLL:
                op_bytes = MASK_OP_BYTES;
                if (off + op_bytes > total) throw DecodeError(where + ": truncated MaskPoll");
                break;
            case OP_TCT:
                op_bytes = u32(off + 4);
                if (op_bytes < 8 || off + op_bytes > total)
                    throw DecodeError(where + ": sync size " + std::to_string(op_bytes));
                break;
            case OP_DDR_PATCH: {
                op_bytes = u32(off + 4);
                if (op_bytes != DDR_PATCH_BYTES || off + op_bytes > total)
                    throw DecodeError(where + ": DdrPatch size " + std::to_string(op_bytes));
                // Payload from off+8: regaddr at +16 (u32 of a u64), arg_idx at +24, arg_plus at +32 (u64).
                const uint32_t addr = u32(off + 24);
                const uint32_t col = (addr >> 25) & 0x7F, row = (addr >> 20) & 0x1F, reg = addr & 0xFFFFF;
                if (!is_bd_reg(row, reg) || (reg - SHIM_BD_BASE) % SHIM_BD_STRIDE != BASE_ADDRESS_LOW_WORD * 4) {
                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "0x%08x", addr);
                    throw DecodeError(where + ": DdrPatch of register " + buf
                                      + ", which is not a shim BD's base address");
                }
                const int arg_idx = data[off + 32];
                uint64_t arg_plus = static_cast<uint64_t>(u32(off + 40)) | (static_cast<uint64_t>(u32(off + 44)) << 32);
                if (arg_idx >= NUM_FIRMWARE_TRANSLATED_ARGS && arg_plus >= DDR_AIE_ADDR_OFFSET)
                    arg_plus -= DDR_AIE_ADDR_OFFSET;
                patches[{col, row, (reg - SHIM_BD_BASE) / SHIM_BD_STRIDE}] = Patch{arg_idx, arg_plus};
                break;
            }
            default: {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "0x%02x", opcode);
                throw DecodeError(where + ": opcode " + buf + " is not decoded");
            }
        }
        off += op_bytes;
    }
    if (off != total)
        throw DecodeError("ops end at byte " + std::to_string(off) + ", header says " + std::to_string(total));
    return out;
}

/// Per data-buffer index, the largest end over every task patched onto it.
inline std::map<int, uint64_t> extents(const std::vector<BdAccess>& accesses) {
    std::map<int, uint64_t> out;
    for (const auto& a : accesses) out[a.arg_idx] = std::max(out[a.arg_idx], a.end);
    return out;
}

/// One line per task that leaves its buffer (or names a buffer that does not
/// exist), given each data buffer's allocated size in bytes.
inline std::vector<std::string> violations(const std::vector<BdAccess>& accesses,
                                           const std::map<int, uint64_t>& buffer_bytes) {
    std::vector<std::string> out;
    for (const auto& a : accesses) {
        auto it = buffer_bytes.find(a.arg_idx);
        if (it != buffer_bytes.end() && a.end <= it->second) continue;
        char buf[256];
        if (it == buffer_bytes.end()) {
            std::snprintf(buf, sizeof(buf),
                "op %zu: shim column %u %s BD %u reaches bytes [0x%llx, 0x%llx) of buffer %d, which is not allocated",
                a.op_index, a.col, a.channel, a.bd, (unsigned long long)a.start, (unsigned long long)a.end, a.arg_idx);
        } else {
            std::snprintf(buf, sizeof(buf),
                "op %zu: shim column %u %s BD %u reaches bytes [0x%llx, 0x%llx) of buffer %d, which holds 0x%llx bytes",
                a.op_index, a.col, a.channel, a.bd, (unsigned long long)a.start, (unsigned long long)a.end,
                a.arg_idx, (unsigned long long)it->second);
        }
        out.emplace_back(buf);
    }
    return out;
}

}  // namespace insts_bounds
