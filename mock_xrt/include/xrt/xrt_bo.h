// SPDX-License-Identifier: Apache-2.0
// Mock XRT Buffer Object header for xdna-emu
// Provides API-compatible interfaces for testing against the emulator

#ifndef MOCK_XRT_BO_H_
#define MOCK_XRT_BO_H_

#include "xrt/detail/xrt_mem.h"
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

namespace xrt {

// Forward declarations
class device;
class kernel;
class hw_context;

/// Buffer object implementation (internal)
class bo_impl;

/// xrt::bo represents a buffer object that can be used as kernel argument
class bo {
public:
    /// Default constructor - creates invalid bo
    bo() = default;

    /// Construct bo with device, size, flags, and memory group
    ///
    /// @param device   Device to allocate buffer on
    /// @param size     Size in bytes
    /// @param flags    Buffer flags (XCL_BO_FLAGS_*, XRT_BO_FLAGS_*)
    /// @param grp      Memory group (from kernel.group_id())
    bo(const device& device, size_t size, uint64_t flags, uint32_t grp);

    /// Construct bo with hw_context, size, and flags
    bo(const hw_context& ctx, size_t size, uint64_t flags);

    /// Copy constructor
    bo(const bo&) = default;

    /// Move constructor
    bo(bo&&) = default;

    /// Destructor
    ~bo();

    /// Copy assignment
    bo& operator=(const bo&) = default;

    /// Move assignment
    bo& operator=(bo&&) = default;

    /// Map buffer to host memory
    ///
    /// @return Pointer to mapped memory
    template<typename T>
    T map() {
        return reinterpret_cast<T>(map_impl());
    }

    /// Write data to buffer
    ///
    /// @param src      Source data pointer
    /// @param size     Size in bytes
    /// @param offset   Offset in buffer
    void write(const void* src, size_t size, size_t offset = 0);

    /// Read data from buffer
    ///
    /// @param dst      Destination data pointer
    /// @param size     Size in bytes
    /// @param offset   Offset in buffer
    void read(void* dst, size_t size, size_t offset = 0) const;

    /// Sync buffer between host and device
    ///
    /// @param direction  XCL_BO_SYNC_BO_TO_DEVICE or XCL_BO_SYNC_BO_FROM_DEVICE
    /// @param size       Size in bytes to sync (0 = entire buffer)
    /// @param offset     Offset in buffer
    void sync(int direction, size_t size = 0, size_t offset = 0);

    /// Get buffer size
    size_t size() const;

    /// Get device address (for device-side access)
    uint64_t address() const;

    /// Check if buffer is valid
    explicit operator bool() const { return m_impl != nullptr; }

private:
    void* map_impl();
    std::shared_ptr<bo_impl> m_impl;
};

} // namespace xrt

#endif // MOCK_XRT_BO_H_
