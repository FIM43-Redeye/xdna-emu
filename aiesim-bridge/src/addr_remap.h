// NPU1 (Phoenix) -> Versal AIE-ML cluster address translation.
//
// WHY THIS EXISTS: the only AIE2 SystemC cluster we can instantiate is Versal
// AIE-ML (VC2802 et al.) -- its device file is an encrypted XbV18.3 blob, so we
// cannot author a native NPU1 5x6 geometry (see the feasibility findings doc,
// 2026-06-02). The xclbins we replay are NPU1/Phoenix-shaped. The AIE2 *core* is
// identical silicon on both, so compute/timing validate faithfully once we map
// coordinates. This is that map -- a pure coordinate transform, applied in the
// bridge (the Versal-cluster adapter); the Rust side stays architecture-neutral.
//
// Three differences between an NPU1 register address and the Versal cluster's:
//   1. BASE: aie-rt forms absolute addrs as BaseAddr + (col<<25) + (row<<20) +
//      offset; for Versal AIE-ML BaseAddr = 0x20000000000 (XAie_Config.BaseAddr,
//      proven by hal_validate). NPU1 xclbin addresses are 32-bit, base-less.
//   2. COLUMN: NPU1 runtime addresses are partition-LOGICAL (column 0 = leftmost
//      partition col); the driver relocates by start_col. We mirror that shift.
//   3. ROW: NPU1 = {shim row 0, 1 memtile row 1, compute rows 2-5}; Versal AIE-ML
//      = {shim row 0, 2 memtile rows 1-2, compute rows 3-10}. So compute rows
//      shift by +1; shim and the (single) memtile row map identity.
//
// Register OFFSETS within a tile are identical across NPU1/Versal (same core IP),
// so the low 20 bits pass through unchanged. DDR/GM addresses are a SEPARATE
// space (the bridge's ddr_target) and must NOT be translated -- only config/MMIO
// tile addresses go through here.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace aiesim {

// Versal AIE-ML config aperture base (aie-rt XAie_Config.BaseAddr for xcve2802).
constexpr uint64_t kAieMlBase = 0x20000000000ULL;
constexpr uint32_t kColShift = 25;
constexpr uint32_t kRowShift = 20;

// NATIVE-GEOMETRY MODE: when the cluster is built from a native NPU1 device file
// (5 cols, 1 memtile row) instead of the Versal-overlay default, the row remap
// must NOT apply -- the addresses are already NPU1-native (shim row 0, memtile
// row 1, compute rows 2-5, with no second memtile row to skip). Gated on
// XDNA_AIESIM_NATIVE_GEOMETRY (presence = native).
inline bool native_geometry() {
    static const bool v = (std::getenv("XDNA_AIESIM_NATIVE_GEOMETRY") != nullptr);
    return v;
}

// Map an NPU1 row to the cluster row. Native NPU1 geometry: identity. Versal
// overlay (Fork A): shim(0)/memtile(1) identity; compute (>=2) shifts +1 to skip
// Versal's second memtile row.
inline uint32_t remap_row(uint32_t npu1_row) {
    if (native_geometry()) return npu1_row;
    return (npu1_row < 2) ? npu1_row : npu1_row + 1;
}

// Translate a 32-bit NPU1 (partition-logical, base-less) tile register address to
// the absolute Versal cluster address the config aximm expects. `start_col` is
// the partition's physical start column (logical col 0 -> physical start_col).
//
// Addresses already at/above the AIE-ML base are assumed Versal-native (e.g. a
// HAL-formed address) and pass through untouched.
inline uint64_t cluster_addr(uint64_t npu_addr, uint8_t start_col) {
    if (npu_addr >= kAieMlBase) return npu_addr;  // already absolute
    const uint32_t col = static_cast<uint32_t>((npu_addr >> kColShift) & 0x7F);
    const uint32_t row = static_cast<uint32_t>((npu_addr >> kRowShift) & 0x1F);
    const uint32_t off = static_cast<uint32_t>(npu_addr & 0xFFFFF);
    const uint64_t pcol = static_cast<uint64_t>(col) + start_col;
    const uint64_t vrow = remap_row(row);
    return kAieMlBase + (pcol << kColShift) + (vrow << kRowShift) + off;
}

}  // namespace aiesim
