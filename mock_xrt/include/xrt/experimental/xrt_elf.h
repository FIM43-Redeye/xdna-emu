// SPDX-License-Identifier: Apache-2.0
// Mock XRT ELF header for xdna-emu
// Provides stub ELF handling for tests that use dynamic ELF loading

#ifndef MOCK_XRT_EXPERIMENTAL_ELF_H_
#define MOCK_XRT_EXPERIMENTAL_ELF_H_

#include <cstdint>
#include <memory>
#include <string>

namespace xrt {

/// ELF implementation (internal)
class elf_impl;

/// xrt::elf represents an ELF binary for NPU cores
class elf {
public:
    /// Default constructor - creates invalid elf
    elf() = default;

    /// Construct from file path
    ///
    /// @param fnm  Path to ELF file
    explicit elf(const std::string& fnm);

    /// Construct from memory buffer
    ///
    /// @param data  Pointer to ELF data
    /// @param size  Size of ELF data in bytes
    elf(const void* data, size_t size);

    /// Copy constructor
    elf(const elf&) = default;

    /// Move constructor
    elf(elf&&) = default;

    /// Destructor
    ~elf() = default;

    /// Copy assignment
    elf& operator=(const elf&) = default;

    /// Move assignment
    elf& operator=(elf&&) = default;

    /// Check if elf is valid
    explicit operator bool() const { return m_impl != nullptr; }

    /// Get the entry point address
    uint64_t get_entry() const;

private:
    std::shared_ptr<elf_impl> m_impl;
};

} // namespace xrt

#endif // MOCK_XRT_EXPERIMENTAL_ELF_H_
