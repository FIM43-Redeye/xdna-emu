// SPDX-License-Identifier: Apache-2.0
// Mock XRT Module header for xdna-emu
// Provides module class for ELF-based kernel execution

#ifndef MOCK_XRT_EXPERIMENTAL_MODULE_H_
#define MOCK_XRT_EXPERIMENTAL_MODULE_H_

#include "xrt/detail/config.h"
#include "xrt/detail/pimpl.h"
#include "xrt/xrt_uuid.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_elf.h"

#include <cstdint>
#include <memory>
#include <string>

namespace xrt {

/// Module implementation (internal)
class module_impl;

/// xrt::module contains functions an application will execute in hardware
///
/// In AIE the functions are a set of instructions that are run on
/// configured hardware, the instructions are embedded in an elf file,
/// which is parsed for meta data determining how the functions are invoked.
class module : public detail::pimpl<module_impl> {
public:
    /// Default constructor - creates invalid module
    module() = default;

    /// Construct from ELF
    ///
    /// @param elf  An elf binary with functions to execute
    ///
    /// The elf binary contains instructions for functions to be executed
    /// in some hardware context.
    XRT_API_EXPORT
    explicit module(const xrt::elf& elf);

    /// Construct from user pointer
    ///
    /// @param userptr  Pointer to instruction data
    /// @param sz       Size of instruction data in bytes
    /// @param uuid     UUID of the hardware configuration
    XRT_API_EXPORT
    module(void* userptr, size_t sz, const xrt::uuid& uuid);

    /// Construct module associated with hardware context
    ///
    /// @param parent  Parent module with instruction buffer
    /// @param hwctx   Hardware context to associate with module
    XRT_API_EXPORT
    module(const xrt::module& parent, const xrt::hw_context& hwctx);

    /// Get the UUID of the hardware configuration
    ///
    /// @return UUID of matching hardware configuration
    XRT_API_EXPORT
    xrt::uuid get_cfg_uuid() const;

    /// Get the hardware context
    ///
    /// @return Hardware context associated with this module
    XRT_API_EXPORT
    xrt::hw_context get_hw_context() const;
};

} // namespace xrt

#endif // MOCK_XRT_EXPERIMENTAL_MODULE_H_
