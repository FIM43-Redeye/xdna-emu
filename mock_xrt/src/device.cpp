// SPDX-License-Identifier: Apache-2.0
// Mock XRT device implementation

#include "xrt/xrt_device.h"
#include "xrt/xrt_xclbin.h"
#include "emulator_bridge.h"

#include <iostream>
#include <stdexcept>

namespace xrt {

/// Device implementation
class device_impl {
public:
    unsigned int index;
    std::string name;
    uuid current_xclbin_uuid;
    std::string xclbin_path;

    device_impl(unsigned int idx) : index(idx), name("NPU Phoenix (Emulated)") {
        xrt_emu::EmulatorBridge::instance().initialize(name);
    }
};

// device constructors
device::device(unsigned int didx)
    : m_impl(std::make_shared<device_impl>(didx)) {
    std::cerr << "[mock_xrt] device(" << didx << ") created" << std::endl;
}

device::device(const std::string& bdf)
    : m_impl(std::make_shared<device_impl>(0)) {
    std::cerr << "[mock_xrt] device(\"" << bdf << "\") created" << std::endl;
}

device::~device() = default;

uuid device::register_xclbin(const xclbin& xclbin) {
    if (!xclbin) {
        throw std::runtime_error("Invalid xclbin");
    }

    // Get UUID from xclbin
    uuid id = xclbin.get_uuid();
    m_impl->current_xclbin_uuid = id;

    std::cerr << "[mock_xrt] Registered xclbin with UUID: " << id.to_string() << std::endl;
    return id;
}

uuid device::load_xclbin(const std::string& xclbin_fnm) {
    m_impl->xclbin_path = xclbin_fnm;

    uint8_t uuid_bytes[16];
    if (!xrt_emu::EmulatorBridge::instance().load_xclbin(xclbin_fnm, uuid_bytes)) {
        throw std::runtime_error("Failed to load xclbin: " + xclbin_fnm);
    }

    m_impl->current_xclbin_uuid = uuid(uuid_bytes);
    return m_impl->current_xclbin_uuid;
}

uuid device::load_xclbin(const xclbin& xclbin) {
    return register_xclbin(xclbin);
}

uuid device::get_xclbin_uuid() const {
    return m_impl->current_xclbin_uuid;
}

std::string device::get_name() const {
    return m_impl->name;
}

bool operator==(const device& d1, const device& d2) {
    if (!d1.m_impl || !d2.m_impl) {
        return d1.m_impl == d2.m_impl;
    }
    return d1.m_impl->index == d2.m_impl->index;
}

bool operator!=(const device& d1, const device& d2) {
    return !(d1 == d2);
}

} // namespace xrt
