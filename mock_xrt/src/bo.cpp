// SPDX-License-Identifier: Apache-2.0
// Mock XRT buffer object implementation

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "emulator_bridge.h"

#include <cstring>
#include <iostream>

namespace xrt {

/// Buffer object implementation
class bo_impl {
public:
    xrt_emu::BufferDesc desc;
    uint64_t flags;
    uint32_t group;

    bo_impl(size_t size, uint64_t f, uint32_t g)
        : flags(f), group(g) {
        desc = xrt_emu::EmulatorBridge::instance().allocate_buffer(size, f);
    }

    ~bo_impl() {
        xrt_emu::EmulatorBridge::instance().free_buffer(desc.id);
    }
};

// bo constructors
bo::bo(const device& device, size_t size, uint64_t flags, uint32_t grp)
    : m_impl(std::make_shared<bo_impl>(size, flags, grp)) {
}

bo::bo(const hw_context& ctx, size_t size, uint64_t flags)
    : m_impl(std::make_shared<bo_impl>(size, flags, 0)) {
}

bo::~bo() = default;

void* bo::map_impl() {
    if (!m_impl) {
        return nullptr;
    }
    return m_impl->desc.host_ptr;
}

void bo::write(const void* src, size_t size, size_t offset) {
    if (!m_impl || !m_impl->desc.host_ptr) {
        return;
    }
    char* dst = static_cast<char*>(m_impl->desc.host_ptr) + offset;
    std::memcpy(dst, src, size);
}

void bo::read(void* dst, size_t size, size_t offset) const {
    if (!m_impl || !m_impl->desc.host_ptr) {
        return;
    }
    const char* src = static_cast<const char*>(m_impl->desc.host_ptr) + offset;
    std::memcpy(dst, src, size);
}

void bo::sync(int direction, size_t size, size_t offset) {
    if (!m_impl) {
        return;
    }

    size_t sync_size = size ? size : m_impl->desc.size;

    if (direction == XCL_BO_SYNC_BO_TO_DEVICE) {
        xrt_emu::EmulatorBridge::instance().sync_to_device(
            m_impl->desc.id, offset, sync_size);
    } else {
        xrt_emu::EmulatorBridge::instance().sync_from_device(
            m_impl->desc.id, offset, sync_size);
    }
}

size_t bo::size() const {
    return m_impl ? m_impl->desc.size : 0;
}

uint64_t bo::address() const {
    return m_impl ? m_impl->desc.device_addr : 0;
}

} // namespace xrt
