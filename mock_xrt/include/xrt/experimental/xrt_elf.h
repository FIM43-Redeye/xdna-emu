// SPDX-License-Identifier: Apache-2.0
// Mock XRT ELF header for xdna-emu
// Provides ELF loading API for tests that use dynamic ELF loading

#ifndef MOCK_XRT_EXPERIMENTAL_ELF_H_
#define MOCK_XRT_EXPERIMENTAL_ELF_H_

#include "xrt/detail/config.h"
#include "xrt/detail/pimpl.h"
#include "xrt/xrt_uuid.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <istream>

namespace xrt {

/// ELF implementation (internal)
class elf_impl;

/// xrt::elf represents an ELF binary for NPU cores
///
/// An elf contains instructions for functions to execute in some
/// pre-configured hardware. The xrt::elf class provides APIs to mine
/// the elf itself for relevant data.
class elf : public detail::pimpl<elf_impl> {
public:
    /// Default constructor - creates invalid elf
    elf() = default;

    /// Construct from file path
    ///
    /// @param fnm  Path to ELF file
    XRT_API_EXPORT
    explicit elf(const std::string& fnm);

    /// Construct from raw data view
    ///
    /// @param data  Raw data view of elf
    /// The raw data can be deleted after calling the constructor.
    XRT_API_EXPORT
    explicit elf(const std::string_view& data);

    /// Construct from C string (avoids ambiguity)
    explicit elf(const char* fnm)
        : elf(std::string(fnm)) {}

    /// Construct from input stream
    ///
    /// @param stream  Raw data stream of elf
    XRT_API_EXPORT
    explicit elf(std::istream& stream);

    /// Construct from raw memory
    ///
    /// @param data  Pointer to ELF data
    /// @param size  Size of ELF data in bytes
    XRT_API_EXPORT
    elf(const void* data, size_t size);

    /// Get the configuration UUID of the elf
    ///
    /// @return The configuration UUID of the elf
    XRT_API_EXPORT
    uuid get_cfg_uuid() const;
};

} // namespace xrt

#endif // MOCK_XRT_EXPERIMENTAL_ELF_H_
