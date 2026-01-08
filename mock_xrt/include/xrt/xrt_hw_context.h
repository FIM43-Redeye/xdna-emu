// SPDX-License-Identifier: Apache-2.0
// Mock XRT Hardware Context header for xdna-emu
// Provides API-compatible interfaces for testing against the emulator

#ifndef MOCK_XRT_HW_CONTEXT_H_
#define MOCK_XRT_HW_CONTEXT_H_

#include "xrt/xrt_uuid.h"
#include "xrt/xrt_device.h"
#include <cstdint>
#include <memory>

namespace xrt {

// Forward declarations
class xclbin;

/// Hardware context implementation (internal)
class hw_context_impl;

/// xrt::hw_context represents a hardware execution context
///
/// A hardware context allocates AIE resources (columns, tiles) for
/// exclusive use by the application. Kernels are run within a context.
class hw_context {
public:
    /// Access mode for context
    enum class access_mode : uint8_t {
        exclusive = 0,  // Exclusive access to resources
        shared = 1      // Shared access (multiple contexts)
    };

    /// QoS (Quality of Service) settings
    struct qos_type {
        uint32_t gops = 0;          // Giga operations per workload
        uint32_t fps = 0;           // Frames per second
        uint32_t dma_bandwidth = 0; // DMA bandwidth
        uint32_t latency = 0;       // Frame response latency
        uint32_t priority = 0x200;  // Request priority (normal = 0x200)
    };

    /// Default constructor - creates invalid context
    hw_context() = default;

    /// Construct context from device and xclbin UUID
    ///
    /// @param device  Device to create context on
    /// @param xclbin_uuid  UUID of registered xclbin
    hw_context(const device& device, const uuid& xclbin_uuid);

    /// Construct context from device, xclbin UUID, and access mode
    ///
    /// @param device  Device to create context on
    /// @param xclbin_uuid  UUID of registered xclbin
    /// @param mode  Access mode (exclusive or shared)
    hw_context(const device& device, const uuid& xclbin_uuid, access_mode mode);

    /// Construct context from device and xclbin
    ///
    /// @param device  Device to create context on
    /// @param xclbin  xrt::xclbin object (must be registered)
    hw_context(const device& device, const xclbin& xclbin);

    /// Construct context with QoS settings
    ///
    /// @param device  Device to create context on
    /// @param xclbin_uuid  UUID of registered xclbin
    /// @param qos  QoS settings
    hw_context(const device& device, const uuid& xclbin_uuid, const qos_type& qos);

    /// Copy constructor
    hw_context(const hw_context&) = default;

    /// Move constructor
    hw_context(hw_context&&) = default;

    /// Destructor
    ~hw_context();

    /// Copy assignment
    hw_context& operator=(const hw_context&) = default;

    /// Move assignment
    hw_context& operator=(hw_context&&) = default;

    /// Get the device associated with this context
    device get_device() const;

    /// Get the xclbin UUID for this context
    uuid get_xclbin_uuid() const;

    /// Check if context is valid
    explicit operator bool() const { return m_impl != nullptr; }

    /// Get implementation (internal use)
    std::shared_ptr<hw_context_impl> get_impl() const { return m_impl; }

private:
    std::shared_ptr<hw_context_impl> m_impl;
};

} // namespace xrt

// For compatibility with experimental header location
namespace xrt { namespace experimental { using hw_context = xrt::hw_context; } }

#endif // MOCK_XRT_HW_CONTEXT_H_
