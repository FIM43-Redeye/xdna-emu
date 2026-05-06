// validate-readback: probe xrt::hw_context::read_aie_reg for ground-truth honesty.
// Throwaway harness; see docs/superpowers/specs/2026-05-06-validate-read-aie-reg-design.md.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

namespace {

constexpr uint32_t TIMER_LOW_OFFSET     = 0x000340F8;
constexpr uint32_t PERF_CTRL0_OFFSET    = 0x00031500;
constexpr uint32_t PERF_COUNTER0_OFFSET = 0x00031520;
constexpr uint8_t  EVENT_ACTIVE_CORE    = 0x1C;

constexpr const char* DEFAULT_XCLBIN =
    "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/peano/aie.xclbin";
constexpr const char* DEFAULT_INSTS =
    "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/peano/insts.bin";

struct Args {
    std::string xclbin = DEFAULT_XCLBIN;
    std::string insts  = DEFAULT_INSTS;
    int col = 0;
    int row = 2;
    bool verbose = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--xclbin" && i + 1 < argc) a.xclbin = argv[++i];
        else if (s == "--insts"  && i + 1 < argc) a.insts  = argv[++i];
        else if (s == "--col"    && i + 1 < argc) a.col    = std::atoi(argv[++i]);
        else if (s == "--row"    && i + 1 < argc) a.row    = std::atoi(argv[++i]);
        else if (s == "-v" || s == "--verbose")   a.verbose = true;
        else {
            std::fprintf(stderr, "unknown arg: %s\n", s.c_str());
            std::exit(2);
        }
    }
    return a;
}

std::vector<uint32_t> load_insts(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open insts: " + path);
    auto size = f.tellg();
    f.seekg(0);
    std::vector<uint32_t> v(size / 4);
    f.read(reinterpret_cast<char*>(v.data()), size);
    return v;
}

} // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::printf("validate-readback: xclbin=%s col=%d row=%d\n",
                args.xclbin.c_str(), args.col, args.row);

    auto device = xrt::device(0);
    auto xclbin = xrt::xclbin(args.xclbin);
    device.register_xclbin(xclbin);
    auto ctx = xrt::hw_context(device, xclbin.get_uuid());
    auto kernels = xclbin.get_kernels();
    if (kernels.empty()) throw std::runtime_error("no kernels in xclbin");
    auto kernel = xrt::kernel(ctx, kernels[0].get_name());

    std::printf("[INFO] loaded xclbin, kernel=%s\n", kernels[0].get_name().c_str());
    return 0;
}
