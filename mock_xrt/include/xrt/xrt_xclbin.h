// SPDX-License-Identifier: Apache-2.0
// Mock XRT XCLBIN header for xdna-emu
// Provides API-compatible interfaces for testing against the emulator

#ifndef MOCK_XRT_XCLBIN_H_
#define MOCK_XRT_XCLBIN_H_

#include "xrt/xrt_uuid.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace xrt {

/// XCLBIN implementation (internal)
class xclbin_impl;

/// xrt::xclbin represents an xclbin container
class xclbin {
public:
    /// Kernel metadata within xclbin
    class kernel {
    public:
        kernel() = default;
        kernel(const std::string& name) : m_name(name) {}

        /// Get kernel name
        std::string get_name() const { return m_name; }

    private:
        std::string m_name;
    };

    /// IP (compute unit) metadata within xclbin
    class ip {
    public:
        ip() = default;
        std::string get_name() const { return m_name; }

    private:
        std::string m_name;
    };

    /// Memory bank metadata within xclbin
    class mem {
    public:
        mem() = default;
    };

    /// Default constructor - creates invalid xclbin
    xclbin() = default;

    /// Construct xclbin from file path
    ///
    /// @param fnm  Path to xclbin file
    explicit xclbin(const std::string& fnm);

    /// Construct xclbin from memory buffer
    ///
    /// @param data  Pointer to xclbin data in memory
    explicit xclbin(const std::vector<char>& data);

    /// Copy constructor
    xclbin(const xclbin&) = default;

    /// Move constructor
    xclbin(xclbin&&) = default;

    /// Destructor
    ~xclbin();

    /// Copy assignment
    xclbin& operator=(const xclbin&) = default;

    /// Move assignment
    xclbin& operator=(xclbin&&) = default;

    /// Get xclbin UUID
    uuid get_uuid() const;

    /// Get list of kernels in xclbin
    std::vector<kernel> get_kernels() const;

    /// Get list of IPs (compute units) in xclbin
    std::vector<ip> get_ips() const;

    /// Get xclbin file path (if loaded from file)
    std::string get_xsa_name() const;

    /// Get target device name
    std::string get_target_vbnv() const;

    /// Check if xclbin is valid
    explicit operator bool() const { return m_impl != nullptr; }

    /// Get implementation (internal use)
    std::shared_ptr<xclbin_impl> get_impl() const { return m_impl; }

private:
    std::shared_ptr<xclbin_impl> m_impl;
};

} // namespace xrt

// For compatibility with experimental header location
namespace xrt { namespace experimental { using xclbin = xrt::xclbin; } }

#endif // MOCK_XRT_XCLBIN_H_
