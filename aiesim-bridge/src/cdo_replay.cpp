#include "cdo_replay.h"

#include <systemc.h>

#include <cstdio>

#include "addr_remap.h"
#include "ps_bridge.h"

namespace aiesim {

namespace {

// Wire-format tags -- MUST match crates/xdna-emu-ffi/src/aiesim/backend.rs
// mod cdo_tag exactly. Field layouts (little-endian, one tagged record per op):
enum CdoTag : uint8_t {
    WRITE = 1,         // [addr u32][val u32]
    WRITE64 = 2,       // [addr u64][val u32]
    MASK_WRITE = 3,    // [addr u32][mask u32][val u32]
    MASK_WRITE64 = 4,  // [addr u64][mask u32][val u32]
    DMA_WRITE = 5,     // [addr u32][len u32][bytes...]
    MASK_POLL = 6,     // [addr u32][mask u32][expected u32]
    MASK_POLL64 = 7,   // [addr u64][mask u32][expected u32]
    DELAY = 8,         // [cycles u32]
    MARKER = 9,        // [value u32]
};

// MASK_POLL bound: advance the kernel in quanta, give up after the cap so a
// never-satisfied condition fails instead of hanging. 1 ns clock => 1 cycle/ns.
// 100K cycles is generous for CDO config ready-bits (reset-done, PLL-lock, which
// clear fast); II-B.2b's DMA waits detect channel completion well before this.
// Tunable -- the cap is a runaway backstop, not a normal exit.
constexpr uint64_t kPollQuantumNs = 256;
constexpr uint64_t kPollMaxNs = 100'000;

// Little-endian cursor over the op-stream with bounds checking. Any short read
// sets err; callers check it before acting on decoded fields.
struct Reader {
    const uint8_t* p;
    std::size_t n;
    std::size_t i = 0;
    bool err = false;

    bool need(std::size_t k) {
        if (i + k > n) { err = true; return false; }
        return true;
    }
    uint8_t u8() { return need(1) ? p[i++] : 0; }
    uint32_t u32() {
        if (!need(4)) return 0;
        uint32_t v = uint32_t(p[i]) | (uint32_t(p[i + 1]) << 8) |
                     (uint32_t(p[i + 2]) << 16) | (uint32_t(p[i + 3]) << 24);
        i += 4;
        return v;
    }
    uint64_t u64() {
        uint64_t lo = u32();
        uint64_t hi = u32();
        return lo | (hi << 32);
    }
};

// Read-modify-write a register, preserving bits outside mask (CDO MaskWrite
// semantics: new = (cur & ~mask) | (val & mask)). Timed live access.
void mask_write(ps_bridge* ps, uint64_t addr, uint32_t mask, uint32_t val) {
    uint32_t cur = ps->read32(addr);
    ps->write32(addr, (cur & ~mask) | (val & mask));
}

// Poll (reg & mask) == expected, advancing the kernel between checks. Returns
// false on timeout. Runs on the driver SC_THREAD, so time advances via wait().
bool mask_poll(ps_bridge* ps, uint64_t addr, uint32_t mask, uint32_t expected) {
    uint64_t elapsed = 0;
    for (;;) {
        if ((ps->read32(addr) & mask) == expected) return true;
        if (elapsed >= kPollMaxNs) return false;
        sc_core::wait(sc_core::sc_time(double(kPollQuantumNs), sc_core::SC_NS));
        elapsed += kPollQuantumNs;
    }
}

}  // namespace

int cdo_replay(ps_bridge* ps, const uint8_t* ops, std::size_t len, uint8_t start_col) {
    Reader r{ops, len};
    while (r.i < r.n && !r.err) {
        const uint8_t tag = r.u8();
        switch (tag) {
            case WRITE: {
                uint32_t a = r.u32(), v = r.u32();
                if (r.err) break;
                ps->write32(cluster_addr(a, start_col), v);
                break;
            }
            case WRITE64: {
                uint64_t a = r.u64();
                uint32_t v = r.u32();
                if (r.err) break;
                ps->write32(cluster_addr(a, start_col), v);
                break;
            }
            case MASK_WRITE: {
                uint32_t a = r.u32(), m = r.u32(), v = r.u32();
                if (r.err) break;
                mask_write(ps, cluster_addr(a, start_col), m, v);
                break;
            }
            case MASK_WRITE64: {
                uint64_t a = r.u64();
                uint32_t m = r.u32(), v = r.u32();
                if (r.err) break;
                mask_write(ps, cluster_addr(a, start_col), m, v);
                break;
            }
            case DMA_WRITE: {
                uint32_t a = r.u32(), l = r.u32();
                if (r.err || !r.need(l)) { r.err = true; break; }
                // A contiguous block write to the config/MMIO space: replay it
                // word by word (register data is 32-bit-aligned). Translate the
                // base once; per-word offsets stay within the tile so they add
                // linearly onto the translated address.
                const uint64_t base = cluster_addr(a, start_col);
                const uint8_t* blk = r.p + r.i;
                for (uint32_t off = 0; off + 4 <= l; off += 4) {
                    uint32_t w = uint32_t(blk[off]) | (uint32_t(blk[off + 1]) << 8) |
                                 (uint32_t(blk[off + 2]) << 16) |
                                 (uint32_t(blk[off + 3]) << 24);
                    ps->write32(base + off, w);
                }
                r.i += l;
                break;
            }
            case MASK_POLL: {
                uint32_t a = r.u32(), m = r.u32(), e = r.u32();
                if (r.err) break;
                if (!mask_poll(ps, cluster_addr(a, start_col), m, e)) {
                    std::fprintf(stderr,
                                 "[cdo_replay] MASK_POLL timeout: addr=0x%x mask=0x%x exp=0x%x\n",
                                 a, m, e);
                    return 1;
                }
                break;
            }
            case MASK_POLL64: {
                uint64_t a = r.u64();
                uint32_t m = r.u32(), e = r.u32();
                if (r.err) break;
                if (!mask_poll(ps, cluster_addr(a, start_col), m, e)) {
                    std::fprintf(stderr,
                                 "[cdo_replay] MASK_POLL64 timeout: addr=0x%llx mask=0x%x exp=0x%x\n",
                                 (unsigned long long)a, m, e);
                    return 1;
                }
                break;
            }
            case DELAY: {
                uint32_t cyc = r.u32();
                if (r.err) break;
                sc_core::wait(sc_core::sc_time(double(cyc), sc_core::SC_NS));
                break;
            }
            case MARKER: {
                (void)r.u32();  // debug annotation -- no replay effect
                break;
            }
            default:
                std::fprintf(stderr, "[cdo_replay] unknown tag %u at offset %zu -- drift\n",
                             tag, r.i - 1);
                return 1;
        }
    }
    if (r.err) {
        std::fprintf(stderr, "[cdo_replay] truncated op-stream (len=%zu)\n", len);
        return 1;
    }
    return 0;
}

}  // namespace aiesim
