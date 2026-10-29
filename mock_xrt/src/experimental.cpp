// SPDX-License-Identifier: Apache-2.0
// Mock XRT experimental API implementation

#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/xrt_device.h"

#include <iostream>
#include <fstream>
#include <vector>

namespace xrt {

// ============================================================================
// elf implementation
// ============================================================================

class elf_impl {
public:
    std::vector<char> data;
    std::string filename;
    uuid cfg_uuid;

    elf_impl() = default;
};

elf::elf(const std::string& fnm) : detail::pimpl<elf_impl>(std::make_shared<elf_impl>()) {
    handle->filename = fnm;
    std::cerr << "[mock_xrt] elf(\"" << fnm << "\") created" << std::endl;

    // Load file into memory
    std::ifstream file(fnm, std::ios::binary | std::ios::ate);
    if (file) {
        size_t size = file.tellg();
        file.seekg(0);
        handle->data.resize(size);
        file.read(handle->data.data(), size);
    }
}

elf::elf(const std::string_view& data) : detail::pimpl<elf_impl>(std::make_shared<elf_impl>()) {
    handle->data.assign(data.begin(), data.end());
}

elf::elf(std::istream& stream) : detail::pimpl<elf_impl>(std::make_shared<elf_impl>()) {
    stream.seekg(0, std::ios::end);
    size_t size = stream.tellg();
    stream.seekg(0);
    handle->data.resize(size);
    stream.read(handle->data.data(), size);
}

elf::elf(const void* data, size_t size) : detail::pimpl<elf_impl>(std::make_shared<elf_impl>()) {
    handle->data.assign(static_cast<const char*>(data), static_cast<const char*>(data) + size);
}

uuid elf::get_cfg_uuid() const {
    return handle ? handle->cfg_uuid : uuid();
}

// ============================================================================
// module implementation
// ============================================================================

class module_impl {
public:
    std::shared_ptr<elf_impl> elf;
    hw_context hwctx;
    uuid cfg_uuid;

    module_impl() = default;
};

module::module(const xrt::elf& e) : detail::pimpl<module_impl>(std::make_shared<module_impl>()) {
    handle->elf = e.get_handle();
    std::cerr << "[mock_xrt] module(elf) created" << std::endl;
}

module::module(void* userptr, size_t sz, const uuid& u) : detail::pimpl<module_impl>(std::make_shared<module_impl>()) {
    handle->cfg_uuid = u;
}

module::module(const module& parent, const hw_context& hwctx) : detail::pimpl<module_impl>(std::make_shared<module_impl>()) {
    if (parent.handle) {
        handle->elf = parent.handle->elf;
        handle->cfg_uuid = parent.handle->cfg_uuid;
    }
    handle->hwctx = hwctx;
}

uuid module::get_cfg_uuid() const {
    return handle ? handle->cfg_uuid : uuid();
}

hw_context module::get_hw_context() const {
    return handle ? handle->hwctx : hw_context();
}

// ============================================================================
// ext::bo implementation
// ============================================================================

namespace ext {

bo::bo(const device& dev, void* userptr, size_t sz, access_mode access)
    : xrt::bo(dev, sz, 0, 0) {
}

bo::bo(const device& dev, void* userptr, size_t sz)
    : xrt::bo(dev, sz, 0, 0) {
}

bo::bo(const device& dev, size_t sz, access_mode access)
    : xrt::bo(dev, sz, 0, 0) {
}

bo::bo(const device& dev, size_t sz)
    : xrt::bo(dev, sz, 0, 0) {
}

bo::bo(const device& dev, pid_type pid, xrt::bo::export_handle ehdl)
    : xrt::bo() {
    // Import not implemented
}

bo::bo(const hw_context& hwctx, size_t sz, access_mode access)
    : xrt::bo() {
    // TODO: Get device from hwctx and allocate
}

bo::bo(const hw_context& hwctx, size_t sz)
    : xrt::bo() {
    // TODO: Get device from hwctx and allocate
}

// ============================================================================
// ext::kernel implementation
// ============================================================================

kernel::kernel(const hw_context& ctx, const module& mod, const std::string& name)
    : xrt::kernel(ctx, name) {
    std::cerr << "[mock_xrt] ext::kernel(ctx, module, \"" << name << "\") created" << std::endl;
}

kernel::kernel(const hw_context& ctx, const std::string& name)
    : xrt::kernel(ctx, name) {
}

} // namespace ext

} // namespace xrt
