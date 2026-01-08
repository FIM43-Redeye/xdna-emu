// SPDX-License-Identifier: Apache-2.0
// Mock XRT memory flags for xdna-emu
// These match the real XRT definitions for API compatibility

#ifndef MOCK_XRT_DETAIL_XRT_MEM_H_
#define MOCK_XRT_DETAIL_XRT_MEM_H_

#include <cstdint>

// Buffer object flags - matches real XRT
// These are used with xrt::bo constructor

/// Cacheable buffer - data is cached on host
#define XCL_BO_FLAGS_CACHEABLE      (1U << 24)

/// Host-only buffer - not mapped to device
#define XRT_BO_FLAGS_HOST_ONLY      (1U << 29)

/// Device-only buffer - not mapped to host
#define XRT_BO_FLAGS_DEVICE_ONLY    (1U << 30)

/// P2P buffer
#define XRT_BO_FLAGS_P2P            (1U << 28)

/// SVM buffer
#define XRT_BO_FLAGS_SVM            (1U << 27)

// Sync directions for bo::sync()
#define XCL_BO_SYNC_BO_TO_DEVICE    0
#define XCL_BO_SYNC_BO_FROM_DEVICE  1

namespace xrt {
namespace bo_flags {

/// Named constants for buffer flags (modern API)
enum class flags : uint32_t {
    normal = 0,
    cacheable = XCL_BO_FLAGS_CACHEABLE,
    host_only = XRT_BO_FLAGS_HOST_ONLY,
    device_only = XRT_BO_FLAGS_DEVICE_ONLY,
    p2p = XRT_BO_FLAGS_P2P,
    svm = XRT_BO_FLAGS_SVM
};

} // namespace bo_flags
} // namespace xrt

#endif // MOCK_XRT_DETAIL_XRT_MEM_H_
