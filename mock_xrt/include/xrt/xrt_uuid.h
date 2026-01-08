// SPDX-License-Identifier: Apache-2.0
// Mock XRT UUID header for xdna-emu
// This provides API-compatible interfaces for testing against the emulator

#ifndef MOCK_XRT_UUID_H_
#define MOCK_XRT_UUID_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

namespace xrt {

/// UUID class compatible with XRT's xrt::uuid
class uuid {
public:
    uuid() : m_uuid{} {}

    explicit uuid(const std::string& str) {
        // Simple parsing - real XRT does more validation
        // Format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        (void)str; // For now, just zero-initialize
        m_uuid = {};
    }

    explicit uuid(const uint8_t* data) {
        std::memcpy(m_uuid.data(), data, 16);
    }

    const uint8_t* get() const { return m_uuid.data(); }

    std::string to_string() const {
        // Return a placeholder string representation
        return "00000000-0000-0000-0000-000000000000";
    }

    bool operator==(const uuid& rhs) const {
        return m_uuid == rhs.m_uuid;
    }

    bool operator!=(const uuid& rhs) const {
        return !(*this == rhs);
    }

private:
    std::array<uint8_t, 16> m_uuid;
};

} // namespace xrt

// C-style UUID type for compatibility
using xuid_t = uint8_t[16];

#endif // MOCK_XRT_UUID_H_
