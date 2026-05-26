// telemetry-probe: directly invoke DRM_AMDXDNA_QUERY_TELEMETRY for all five
// telemetry types and dump the returned buffer. Used to answer: does Phoenix
// firmware actually populate TELEMETRY_TYPE_PROFILING (or any of the other
// types), and if so, what's in it?
//
// Background: DRM_AMDXDNA_QUERY_TELEMETRY ioctl is admin/matching-EUID gated
// per the uapi comment, but aie2_get_telemetry has no visible capable()/
// EUID check in drivers/accel -- so we may just need any opener of accel0.
// MSG_OP_GET_TELEMETRY (0x4) is NOT feature-bit gated in the driver (unlike
// APP_HEALTH), so if Phoenix firmware ignores it we'll see -ETIMEDOUT or a
// nonzero status; if it accepts the op we'll see populated buffer bytes.
//
// Buffer layout (kernel side):
//   [amdxdna_drm_query_telemetry_header (16 bytes fixed)]
//   [u32 map[hwctx_limit]]                  -- 24 bytes on Phoenix (limit=6)
//   [telemetry_data]                        -- whatever FW fills in
//
// We allocate a generous buffer (64KB) and the driver will fill in whatever
// portion FW returns. resp.size tells us how much.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <drm/amdxdna_accel.h>

namespace {

constexpr size_t BUFFER_SIZE = 64 * 1024;
constexpr uint32_t PHOENIX_HWCTX_LIMIT = 6;  // npu1_regs.c

const char* type_name(uint32_t t) {
    switch (t) {
    case 0: return "DISABLED";
    case 1: return "HEALTH";
    case 2: return "ERROR_INFO";
    case 3: return "PROFILING";
    case 4: return "DEBUG";
    default: return "UNKNOWN";
    }
}

void hex_dump(const uint8_t* data, size_t len, size_t max_lines = 16) {
    size_t lines_printed = 0;
    bool found_nonzero = false;

    // First pass: just look for nonzero ranges.
    for (size_t i = 0; i < len; i += 16) {
        bool any_nz = false;
        for (size_t j = 0; j < 16 && i + j < len; ++j) {
            if (data[i + j] != 0) { any_nz = true; break; }
        }
        if (any_nz) {
            if (!found_nonzero) {
                std::printf("  --- nonzero bytes ---\n");
                found_nonzero = true;
            }
            std::printf("  %04zx:", i);
            for (size_t j = 0; j < 16 && i + j < len; ++j) {
                std::printf(" %02x", data[i + j]);
                if (j == 7) std::printf(" ");
            }
            std::printf("\n");
            if (++lines_printed >= max_lines) {
                std::printf("  ... (truncated; %zu bytes total)\n", len);
                return;
            }
        }
    }
    if (!found_nonzero) {
        std::printf("  (all zeros across %zu bytes)\n", len);
    }
}

int probe_type(int fd, uint32_t type) {
    std::vector<uint8_t> buf(BUFFER_SIZE, 0);

    // Stamp the header.type field in the user buffer.
    auto* hdr = reinterpret_cast<amdxdna_drm_query_telemetry_header*>(buf.data());
    hdr->type = type;

    amdxdna_drm_get_info args = {};
    args.param = DRM_AMDXDNA_QUERY_TELEMETRY;
    args.buffer_size = BUFFER_SIZE;
    args.buffer = reinterpret_cast<__u64>(buf.data());

    std::printf("\n=== type=%u (%s) ===\n", type, type_name(type));

    int rc = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &args);
    int e = errno;
    if (rc < 0) {
        std::printf("  ioctl FAILED errno=%d (%s)\n", e, strerror(e));
        return 1;
    }

    // Header is back-populated by driver.
    auto* hdr_out = reinterpret_cast<amdxdna_drm_query_telemetry_header*>(buf.data());
    std::printf("  header: major=%u minor=%u type=%u map_num_elements=%u\n",
                hdr_out->major, hdr_out->minor, hdr_out->type,
                hdr_out->map_num_elements);

    // Map array
    size_t header_sz = sizeof(*hdr_out) + hdr_out->map_num_elements * sizeof(uint32_t);
    std::printf("  map[%u]:", hdr_out->map_num_elements);
    for (uint32_t i = 0; i < hdr_out->map_num_elements; ++i) {
        std::printf(" %u", hdr_out->map[i]);
    }
    std::printf("\n");

    // Telemetry data begins after header_sz. Dump first few nonzero regions.
    size_t data_off = header_sz;
    if (data_off >= BUFFER_SIZE) {
        std::printf("  no data section in buffer\n");
        return 0;
    }
    hex_dump(buf.data() + data_off, BUFFER_SIZE - data_off, /*max_lines*/ 24);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool only_profiling = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--profiling") == 0) {
            only_profiling = true;
        }
    }

    int fd = open("/dev/accel/accel0", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::perror("open /dev/accel/accel0");
        return 2;
    }

    std::printf("Phoenix hwctx_limit=%u (header_sz=%zu bytes)\n",
                PHOENIX_HWCTX_LIMIT,
                sizeof(amdxdna_drm_query_telemetry_header)
                    + PHOENIX_HWCTX_LIMIT * sizeof(uint32_t));

    if (only_profiling) {
        probe_type(fd, 3);
    } else {
        for (uint32_t t = 0; t < 5; ++t) {
            probe_type(fd, t);
        }
    }

    close(fd);
    return 0;
}
