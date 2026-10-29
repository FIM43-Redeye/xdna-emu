// SPDX-License-Identifier: Apache-2.0
// Mock XRT pimpl header for xdna-emu

#ifndef MOCK_XRT_DETAIL_PIMPL_H_
#define MOCK_XRT_DETAIL_PIMPL_H_

#include <memory>
#include <cstddef>
#include <cstdint>

namespace xrt::detail {

/// Base class for pimpl idiom - provides handle semantics
template<typename ImplType>
class pimpl {
protected:
    std::shared_ptr<ImplType> handle;

public:
    pimpl() = default;

    explicit pimpl(std::shared_ptr<ImplType> impl)
        : handle(std::move(impl)) {}

    /// Check if handle is valid
    explicit operator bool() const noexcept {
        return handle != nullptr;
    }

    /// Get raw handle (for implementation use)
    ImplType* get_handle_raw() const {
        return handle.get();
    }

    /// Get shared handle
    std::shared_ptr<ImplType> get_handle() const {
        return handle;
    }
};

/// Simple span type for data views (C++20 backport)
template<typename T>
class span {
    T* m_data = nullptr;
    size_t m_size = 0;

public:
    span() = default;
    span(T* data, size_t size) : m_data(data), m_size(size) {}

    T* data() const { return m_data; }
    size_t size() const { return m_size; }
    bool empty() const { return m_size == 0; }

    T& operator[](size_t i) { return m_data[i]; }
    const T& operator[](size_t i) const { return m_data[i]; }

    T* begin() { return m_data; }
    T* end() { return m_data + m_size; }
    const T* begin() const { return m_data; }
    const T* end() const { return m_data + m_size; }
};

} // namespace xrt::detail

#endif // MOCK_XRT_DETAIL_PIMPL_H_
