// SPDX-License-Identifier: Apache-2.0
// Mock XRT Device header for xdna-emu
// Provides API-compatible interfaces for testing against the emulator

#ifndef MOCK_XRT_DEVICE_H_
#define MOCK_XRT_DEVICE_H_

#include "xrt/xrt_uuid.h"
#include "xrt/xrt_xclbin.h"
#include <cstdint>
#include <memory>
#include <string>

namespace xrt {

// Forward declarations
class bo;

/// Device implementation (internal)
class device_impl;

/// xrt::device represents a device used for acceleration
class device {
public:
    /// Default constructor - creates invalid device
    device() = default;

    /// Construct device from device index
    ///
    /// @param didx  Device index (typically 0)
    explicit device(unsigned int didx);

    /// Construct device from int (convenience overload)
    explicit device(int didx) : device(static_cast<unsigned int>(didx)) {}

    /// Construct device from BDF string
    ///
    /// @param bdf  PCIe BDF string (e.g., "0000:c6:00.1")
    explicit device(const std::string& bdf);

    /// Copy constructor
    device(const device&) = default;

    /// Move constructor
    device(device&&) = default;

    /// Destructor
    ~device();

    /// Copy assignment
    device& operator=(const device&) = default;

    /// Move assignment
    device& operator=(device&&) = default;

    /// Register an xclbin with the device
    ///
    /// @param xclbin  xrt::xclbin object to register
    /// @return UUID of the registered xclbin
    ///
    /// This registers the xclbin but does not associate it with hardware.
    /// Use hw_context to create an execution context.
    uuid register_xclbin(const xclbin& xclbin);

    /// Load an xclbin file
    ///
    /// @param xclbin_fnm  Path to xclbin file
    /// @return UUID of loaded xclbin
    uuid load_xclbin(const std::string& xclbin_fnm);

    /// Load an xclbin object
    ///
    /// @param xclbin  xrt::xclbin object to load
    /// @return UUID of loaded xclbin
    uuid load_xclbin(const xclbin& xclbin);

    /// Get UUID of currently loaded xclbin
    uuid get_xclbin_uuid() const;

    /// Get device name (VBNV)
    std::string get_name() const;

    /// Check if device is valid
    explicit operator bool() const { return m_impl != nullptr; }

    /// Get implementation (internal use)
    std::shared_ptr<device_impl> get_impl() const { return m_impl; }

    /// Friend functions for comparison
    friend bool operator==(const device& d1, const device& d2);
    friend bool operator!=(const device& d1, const device& d2);

private:
    std::shared_ptr<device_impl> m_impl;
};

/// Compare two device objects
bool operator==(const device& d1, const device& d2);
bool operator!=(const device& d1, const device& d2);

} // namespace xrt

#endif // MOCK_XRT_DEVICE_H_
