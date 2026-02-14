// npu_runner -- Generic NPU test execution tool.
//
// Runs any xclbin on real AMD NPU hardware via XRT, reading/writing raw
// binary buffer files.  All manifest logic and data generation stays in
// Rust; this tool is a thin XRT wrapper with a simple CLI:
//
//   npu-runner <xclbin> <insts.bin> [options]
//     --in  <group_id> <file> <size_bytes>   Input buffer  (repeatable)
//     --out <group_id> <file> <size_bytes>   Output buffer (repeatable)
//     --timeout <seconds>                    Execution timeout (default: 30)
//     --kernel <name>                        Kernel name   (default: MLIR_AIE)
//
// Exit codes:
//   0  Success
//   1  Runtime error (device/xclbin/kernel failure)
//   2  Timeout (kernel did not complete in time)
//   3  Argument error

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// ---------------------------------------------------------------------------
// Buffer descriptor parsed from CLI --in / --out flags.
// ---------------------------------------------------------------------------
struct BufferDesc {
    int      group_id;
    std::string file_path;
    size_t   size_bytes;
    bool     is_output;   // true = read back after execution
};

// ---------------------------------------------------------------------------
// Read a binary file into a byte vector.
// ---------------------------------------------------------------------------
static std::vector<uint8_t> read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "ERROR: cannot open file: " << path << "\n";
        std::exit(1);
    }
    f.seekg(0, std::ios::end);
    auto len = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(len);
    if (!f.read(reinterpret_cast<char *>(buf.data()), len)) {
        std::cerr << "ERROR: failed to read file: " << path << "\n";
        std::exit(1);
    }
    return buf;
}

// ---------------------------------------------------------------------------
// Write a byte buffer to a file.
// ---------------------------------------------------------------------------
static void write_file(const std::string &path, const void *data, size_t len) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "ERROR: cannot write file: " << path << "\n";
        std::exit(1);
    }
    f.write(reinterpret_cast<const char *>(data), len);
    if (!f) {
        std::cerr << "ERROR: write failed: " << path << "\n";
        std::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Usage message.
// ---------------------------------------------------------------------------
static void usage(const char *prog) {
    std::cerr
        << "Usage: " << prog << " <xclbin> <insts.bin> [options]\n"
        << "\n"
        << "Options:\n"
        << "  --in  <group_id> <file> <size_bytes>   Input buffer (repeatable)\n"
        << "  --out <group_id> <file> <size_bytes>   Output buffer (repeatable)\n"
        << "  --timeout <seconds>                    Execution timeout (default: 30)\n"
        << "  --kernel <name>                        Kernel name (default: MLIR_AIE)\n"
        << "  --device <index>                       Device index (default: 0)\n"
        << "  -h, --help                             Show this help\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    // Check for help flag before positional argument check
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        }
    }

    if (argc < 3) {
        usage(argv[0]);
        return 3;
    }

    // -- Parse positional arguments ------------------------------------------
    std::string xclbin_path = argv[1];
    std::string insts_path  = argv[2];

    // -- Parse optional arguments --------------------------------------------
    std::string kernel_name = "MLIR_AIE";
    unsigned    device_index = 0;
    int         timeout_sec  = 30;
    std::vector<BufferDesc> buffers;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];

        if ((arg == "--in" || arg == "--out") && i + 3 < argc) {
            BufferDesc bd;
            bd.group_id   = std::stoi(argv[i + 1]);
            bd.file_path  = argv[i + 2];
            bd.size_bytes = std::stoull(argv[i + 3]);
            bd.is_output  = (arg == "--out");
            buffers.push_back(bd);
            i += 3;
        } else if (arg == "--timeout" && i + 1 < argc) {
            timeout_sec = std::stoi(argv[++i]);
        } else if (arg == "--kernel" && i + 1 < argc) {
            kernel_name = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            device_index = std::stoul(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "ERROR: unknown option: " << arg << "\n";
            usage(argv[0]);
            return 3;
        }
    }

    // -- Load instruction binary ---------------------------------------------
    auto insts_raw = read_file(insts_path);
    if (insts_raw.size() % 4 != 0) {
        std::cerr << "ERROR: instruction file size is not a multiple of 4 bytes\n";
        return 1;
    }
    size_t instr_count = insts_raw.size() / sizeof(uint32_t);

    // -- Open device and load xclbin -----------------------------------------
    try {
        std::cerr << "Opening device " << device_index << "...\n";
        auto device = xrt::device(device_index);

        std::cerr << "Loading xclbin: " << xclbin_path << "\n";
        auto xclbin = xrt::xclbin(xclbin_path);

        // Find the kernel by prefix match (same logic as mlir-aie tests)
        auto xkernels = xclbin.get_kernels();
        auto it = std::find_if(xkernels.begin(), xkernels.end(),
            [&kernel_name](xrt::xclbin::kernel &k) {
                return k.get_name().rfind(kernel_name, 0) == 0;
            });
        if (it == xkernels.end()) {
            std::cerr << "ERROR: kernel '" << kernel_name
                      << "' not found in xclbin\n";
            std::cerr << "Available kernels:";
            for (auto &k : xkernels)
                std::cerr << " " << k.get_name();
            std::cerr << "\n";
            return 1;
        }
        auto resolved_name = it->get_name();
        std::cerr << "Using kernel: " << resolved_name << "\n";

        // Register xclbin and create hardware context
        device.register_xclbin(xclbin);
        auto context = xrt::hw_context(device, xclbin.get_uuid());
        auto kernel  = xrt::kernel(context, resolved_name);

        // -- Create instruction buffer (group_id 1) --------------------------
        auto bo_instr = xrt::bo(device, insts_raw.size(),
                                XCL_BO_FLAGS_CACHEABLE,
                                kernel.group_id(1));
        std::memcpy(bo_instr.map<void *>(), insts_raw.data(), insts_raw.size());
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // -- Create data buffers (sorted by group_id for kernel args) ---------
        // Sort buffers by group_id so they map to positional kernel args
        std::sort(buffers.begin(), buffers.end(),
            [](const BufferDesc &a, const BufferDesc &b) {
                return a.group_id < b.group_id;
            });

        struct LiveBuffer {
            xrt::bo   bo;
            BufferDesc desc;
        };
        std::vector<LiveBuffer> live_buffers;

        for (auto &bd : buffers) {
            auto bo = xrt::bo(device, bd.size_bytes,
                              XRT_BO_FLAGS_HOST_ONLY,
                              kernel.group_id(bd.group_id));

            if (!bd.is_output) {
                // Input: read file data into the buffer
                auto file_data = read_file(bd.file_path);
                if (file_data.size() > bd.size_bytes) {
                    std::cerr << "WARNING: input file " << bd.file_path
                              << " (" << file_data.size()
                              << " bytes) exceeds buffer size ("
                              << bd.size_bytes << " bytes), truncating\n";
                    file_data.resize(bd.size_bytes);
                }
                auto *mapped = bo.map<uint8_t *>();
                // Zero the full buffer first, then copy file data
                std::memset(mapped, 0, bd.size_bytes);
                std::memcpy(mapped, file_data.data(), file_data.size());
                bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                std::cerr << "  Input  group_id=" << bd.group_id
                          << " size=" << bd.size_bytes
                          << " file=" << bd.file_path << "\n";
            } else {
                // Output: zero-initialize
                auto *mapped = bo.map<uint8_t *>();
                std::memset(mapped, 0, bd.size_bytes);
                bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                std::cerr << "  Output group_id=" << bd.group_id
                          << " size=" << bd.size_bytes
                          << " file=" << bd.file_path << "\n";
            }

            live_buffers.push_back({std::move(bo), bd});
        }

        // -- Set kernel arguments and run ------------------------------------
        // Argument layout (matching mlir-aie test.cpp convention):
        //   arg 0: opcode (always 3)
        //   arg 1: instruction buffer
        //   arg 2: instruction count
        //   arg 3+: data buffers in group_id order
        auto run = xrt::run(kernel);
        run.set_arg(0, static_cast<uint32_t>(3));  // opcode
        run.set_arg(1, bo_instr);
        run.set_arg(2, static_cast<uint32_t>(instr_count));

        // Data buffers: arg index = group_id (since group_ids start at 3)
        for (auto &lb : live_buffers) {
            run.set_arg(lb.desc.group_id, lb.bo);
        }

        std::cerr << "Running kernel (timeout=" << timeout_sec << "s)...\n";
        run.start();

        auto timeout_ms = static_cast<unsigned int>(timeout_sec * 1000);
        auto state = run.wait(std::chrono::milliseconds(timeout_ms));

        if (state == ERT_CMD_STATE_TIMEOUT) {
            std::cerr << "ERROR: kernel execution timed out after "
                      << timeout_sec << " seconds\n";
            return 2;
        }
        if (state != ERT_CMD_STATE_COMPLETED) {
            std::cerr << "ERROR: kernel did not complete, state="
                      << state << "\n";
            return 1;
        }

        std::cerr << "Kernel completed successfully.\n";

        // -- Read back output buffers ----------------------------------------
        for (auto &lb : live_buffers) {
            if (lb.desc.is_output) {
                lb.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                auto *mapped = lb.bo.map<uint8_t *>();
                write_file(lb.desc.file_path, mapped, lb.desc.size_bytes);
                std::cerr << "  Wrote output: " << lb.desc.file_path
                          << " (" << lb.desc.size_bytes << " bytes)\n";
            }
        }

        return 0;

    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
