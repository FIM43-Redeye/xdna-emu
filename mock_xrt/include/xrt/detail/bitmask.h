// SPDX-License-Identifier: Apache-2.0
// Mock XRT bitmask utilities for xdna-emu

#ifndef MOCK_XRT_DETAIL_BITMASK_H_
#define MOCK_XRT_DETAIL_BITMASK_H_

#include <type_traits>

namespace xrt::detail {

/// Bitwise AND for enum class types
template<typename E>
constexpr E operator&(E lhs, E rhs) {
    using U = std::underlying_type_t<E>;
    return static_cast<E>(static_cast<U>(lhs) & static_cast<U>(rhs));
}

/// Bitwise OR for enum class types
template<typename E>
constexpr E operator|(E lhs, E rhs) {
    using U = std::underlying_type_t<E>;
    return static_cast<E>(static_cast<U>(lhs) | static_cast<U>(rhs));
}

/// Bitwise XOR for enum class types
template<typename E>
constexpr E operator^(E lhs, E rhs) {
    using U = std::underlying_type_t<E>;
    return static_cast<E>(static_cast<U>(lhs) ^ static_cast<U>(rhs));
}

/// Bitwise NOT for enum class types
template<typename E>
constexpr E operator~(E val) {
    using U = std::underlying_type_t<E>;
    return static_cast<E>(~static_cast<U>(val));
}

} // namespace xrt::detail

#endif // MOCK_XRT_DETAIL_BITMASK_H_
