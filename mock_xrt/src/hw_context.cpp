// SPDX-License-Identifier: Apache-2.0
// Mock XRT hardware context implementation

#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_xclbin.h"
#include "emulator_bridge.h"

#include <iostream>

namespace xrt {

/// Hardware context implementation
class hw_context_impl {
public:
    device m_device;
    uuid m_xclbin_uuid;
    uint32_t m_handle;

    hw_context_impl(const device& dev, const uuid& uuid)
        : m_device(dev), m_xclbin_uuid(uuid) {
        // Create context in emulator
        uint8_t uuid_bytes[16];
        std::memcpy(uuid_bytes, uuid.get(), 16);
        m_handle = xrt_emu::EmulatorBridge::instance().create_context(uuid_bytes);
    }

    ~hw_context_impl() {
        xrt_emu::EmulatorBridge::instance().destroy_context(m_handle);
    }
};

// hw_context constructors
hw_context::hw_context(const device& device, const uuid& xclbin_uuid)
    : m_impl(std::make_shared<hw_context_impl>(device, xclbin_uuid)) {
    std::cerr << "[mock_xrt] hw_context created with UUID: " << xclbin_uuid.to_string() << std::endl;
}

hw_context::hw_context(const device& device, const uuid& xclbin_uuid, access_mode mode)
    : hw_context(device, xclbin_uuid) {
    // TODO: Handle access_mode (exclusive vs shared)
    (void)mode;
}

hw_context::hw_context(const device& device, const xclbin& xclbin)
    : hw_context(device, xclbin.get_uuid()) {
}

hw_context::hw_context(const device& device, const uuid& xclbin_uuid, const qos_type& qos)
    : hw_context(device, xclbin_uuid) {
    // TODO: Apply QoS settings to emulator
}

hw_context::~hw_context() = default;

device hw_context::get_device() const {
    return m_impl ? m_impl->m_device : device();
}

uuid hw_context::get_xclbin_uuid() const {
    return m_impl ? m_impl->m_xclbin_uuid : uuid();
}

} // namespace xrt
