// SPDX-License-Identifier: MIT
// Query amdxdna's reported power mode and MP-NPU/H clocks as JSON.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <drm/amdxdna_accel.h>

int main(int argc, char** argv) {
    if (argc > 2) {
        std::fprintf(stderr, "usage: %s [/dev/accel/accel0]\n", argv[0]);
        return 2;
    }
    const char* path = argc == 2 ? argv[1] : "/dev/accel/accel0";
    const int fd = ::open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "open %s: %s\n", path, std::strerror(errno));
        return 1;
    }

    amdxdna_drm_query_clock_metadata clocks{};
    amdxdna_drm_get_info info{
        .param = DRM_AMDXDNA_QUERY_CLOCK_METADATA,
        .buffer_size = sizeof(clocks),
        .buffer = reinterpret_cast<std::uint64_t>(&clocks),
    };
    if (::ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &info) < 0) {
        std::fprintf(stderr, "QUERY_CLOCK_METADATA: %s\n", std::strerror(errno));
        ::close(fd);
        return 1;
    }

    amdxdna_drm_get_power_mode power{};
    info = {
        .param = DRM_AMDXDNA_GET_POWER_MODE,
        .buffer_size = sizeof(power),
        .buffer = reinterpret_cast<std::uint64_t>(&power),
    };
    if (::ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &info) < 0) {
        std::fprintf(stderr, "GET_POWER_MODE: %s\n", std::strerror(errno));
        ::close(fd);
        return 1;
    }
    ::close(fd);

    constexpr const char* modes[] = {
        "default", "powersaver", "balanced", "performance", "turbo",
    };
    const char* mode = power.power_mode < sizeof(modes) / sizeof(modes[0])
        ? modes[power.power_mode] : "unknown";
    std::printf(
        "{\"power_mode\":\"%s\",\"power_mode_id\":%u,"
        "\"mp_npu_mhz\":%u,\"h_mhz\":%u}\n",
        mode, power.power_mode,
        clocks.mp_npu_clock.freq_mhz, clocks.h_clock.freq_mhz);
    return 0;
}
