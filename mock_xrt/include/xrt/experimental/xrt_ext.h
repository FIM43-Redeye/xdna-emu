// SPDX-License-Identifier: Apache-2.0
// Mock XRT Extensions header for xdna-emu
// Provides extended buffer and kernel classes

#ifndef MOCK_XRT_EXPERIMENTAL_EXT_H_
#define MOCK_XRT_EXPERIMENTAL_EXT_H_

#include "xrt/detail/config.h"
#include "xrt/detail/bitmask.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_module.h"

#include <cstdint>

namespace xrt::ext {

/// Extended buffer object with access mode support
///
/// xrt::ext::bo is an extension of xrt::bo with additional functionality
/// for specifying access modes and buffer sharing.
class bo : public xrt::bo {
public:
    /// Buffer access mode flags
    ///
    /// Specifies how the buffer is used by device and process.
    enum class access_mode : uint64_t {
        none       = 0,         ///< Default: read|write|local

        read       = 1 << 0,    ///< Device reads, host writes
        write      = 1 << 1,    ///< Device writes, host reads
        read_write = read | write,

        local      = 0,         ///< Local to process and device
        shared     = 1 << 2,    ///< Shared between devices in process
        process    = 1 << 3,    ///< Shared between processes
        hybrid     = 1 << 4,    ///< Cross-adapter sharing
    };

    friend constexpr access_mode operator&(access_mode lhs, access_mode rhs) {
        return xrt::detail::operator&(lhs, rhs);
    }

    friend constexpr access_mode operator|(access_mode lhs, access_mode rhs) {
        return xrt::detail::operator|(lhs, rhs);
    }

    /// Default constructor
    bo() = default;

    /// Construct with user buffer and access mode
    ///
    /// @param device   Device to allocate on
    /// @param userptr  User buffer (page aligned)
    /// @param sz       Size (multiple of page size)
    /// @param access   Access mode
    XRT_API_EXPORT
    bo(const xrt::device& device, void* userptr, size_t sz, access_mode access);

    /// Construct with user buffer (default access)
    XRT_API_EXPORT
    bo(const xrt::device& device, void* userptr, size_t sz);

    /// Construct with size and access mode
    XRT_API_EXPORT
    bo(const xrt::device& device, size_t sz, access_mode access);

    /// Construct with size (default access)
    XRT_API_EXPORT
    bo(const xrt::device& device, size_t sz);

    /// Import buffer from another process
    XRT_API_EXPORT
    bo(const xrt::device& device, pid_type pid, xrt::bo::export_handle ehdl);

    /// Construct with hardware context and access mode
    XRT_API_EXPORT
    bo(const xrt::hw_context& hwctx, size_t sz, access_mode access);

    /// Construct with hardware context (default access)
    XRT_API_EXPORT
    bo(const xrt::hw_context& hwctx, size_t sz);
};


/// Extended kernel with module support
///
/// xrt::ext::kernel extends xrt::kernel with constructors that accept
/// module objects for ELF-based execution.
class kernel : public xrt::kernel {
public:
    /// Default constructor
    kernel() = default;

    /// Construct from module
    ///
    /// @param ctx    Hardware context
    /// @param mod    Module with ELF binary instructions
    /// @param name   Name of kernel function
    XRT_API_EXPORT
    kernel(const xrt::hw_context& ctx, const xrt::module& mod, const std::string& name);

    /// Construct by searching registered ELFs
    ///
    /// @param ctx    Hardware context
    /// @param name   Name of kernel function
    ///
    /// Searches through all ELF files registered with the context.
    XRT_API_EXPORT
    kernel(const xrt::hw_context& ctx, const std::string& name);
};

} // namespace xrt::ext

#endif // MOCK_XRT_EXPERIMENTAL_EXT_H_
