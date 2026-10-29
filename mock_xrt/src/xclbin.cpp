// SPDX-License-Identifier: Apache-2.0
// Mock XRT xclbin implementation

#include "xrt/xrt_xclbin.h"
#include "emulator_bridge.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace xrt {

/// XCLBIN implementation
class xclbin_impl {
public:
    std::string path;
    std::vector<char> data;
    uuid xclbin_uuid;
    std::vector<xclbin::kernel> kernels;

    xclbin_impl(const std::string& fnm) : path(fnm) {
        // Read xclbin file
        std::ifstream file(fnm, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("Failed to open xclbin: " + fnm);
        }

        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        data.resize(size);
        if (!file.read(data.data(), size)) {
            throw std::runtime_error("Failed to read xclbin: " + fnm);
        }

        // Generate UUID from file content
        uint8_t uuid_bytes[16];
        xrt_emu::EmulatorBridge::instance().load_xclbin(fnm, uuid_bytes);
        xclbin_uuid = uuid(uuid_bytes);

        // TODO: Parse xclbin to extract kernel names
        // For now, add a default kernel name based on mlir-aie convention
        kernels.push_back(xclbin::kernel("MLIR_AIE"));

        std::cerr << "[mock_xrt] Loaded xclbin: " << fnm << " (" << size << " bytes)" << std::endl;
    }

    xclbin_impl(const std::vector<char>& d) : data(d) {
        // Generate UUID from data
        std::hash<std::string> hasher;
        size_t hash = hasher(std::string(d.begin(), d.end()));
        uint8_t uuid_bytes[16] = {0};
        std::memcpy(uuid_bytes, &hash, sizeof(hash));
        xclbin_uuid = uuid(uuid_bytes);

        kernels.push_back(xclbin::kernel("MLIR_AIE"));
    }
};

// xclbin constructors
xclbin::xclbin(const std::string& fnm)
    : m_impl(std::make_shared<xclbin_impl>(fnm)) {
}

xclbin::xclbin(const std::vector<char>& data)
    : m_impl(std::make_shared<xclbin_impl>(data)) {
}

xclbin::~xclbin() = default;

uuid xclbin::get_uuid() const {
    return m_impl ? m_impl->xclbin_uuid : uuid();
}

std::vector<xclbin::kernel> xclbin::get_kernels() const {
    return m_impl ? m_impl->kernels : std::vector<xclbin::kernel>();
}

std::vector<xclbin::ip> xclbin::get_ips() const {
    return {};
}

std::string xclbin::get_xsa_name() const {
    return m_impl ? m_impl->path : "";
}

std::string xclbin::get_target_vbnv() const {
    return "NPU Phoenix";
}

} // namespace xrt
