// Host for the runtime-sequence op cost probe: one hardware context, one launch, with an
// optional context QoS (gops, fps) and a pause before and after the launch so a clock query
// can be taken while the context is alive. Progress lines on stdout, one per event:
//   READY            context created, buffers synced, pre-launch pause over
//   DONE wall_us=N   the launch completed; N is the wall time of the launch in microseconds
//   EXIT             the post-launch hold is over
// Usage: fw_op_cost_host -x aie.xclbin -i insts.bin --out-size BYTES --out-file F
//          [--gops G --fps F] [--pre-ms 200] [--hold-ms 300] [-k MLIR_AIE]
//          [--instr-bo-bytes N]   (allocate the instruction buffer at least N bytes, zero-padded)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

int main(int argc, const char *argv[]) {
    cxxopts::Options options("fw_op_cost_host", "runtime-sequence op cost probe host");
    test_utils::add_default_options(options);
    options.add_options()
        ("out-size", "Output buffer bytes", cxxopts::value<int>())
        ("out-file", "Output file", cxxopts::value<std::string>())
        ("gops", "Context QoS gops (0 = no QoS)", cxxopts::value<uint32_t>()->default_value("0"))
        ("fps", "Context QoS fps", cxxopts::value<uint32_t>()->default_value("0"))
        ("pre-ms", "Pause before the launch", cxxopts::value<int>()->default_value("200"))
        ("hold-ms", "Pause after the launch", cxxopts::value<int>()->default_value("300"))
        ("instr-bo-bytes", "Minimum instruction buffer size (zero-padded)",
         cxxopts::value<size_t>()->default_value("0"));
    cxxopts::ParseResult vm;
    test_utils::parse_options(argc, argv, options, vm);

    int out_bytes = vm["out-size"].as<int>();
    int out_elems = (out_bytes + 3) / 4;
    uint32_t gops = vm["gops"].as<uint32_t>(), fps = vm["fps"].as<uint32_t>();

    std::vector<uint32_t> instr_v = test_utils::load_instr_binary(vm["instr"].as<std::string>());
    auto device = xrt::device(0u);
    auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
    std::string node = vm["kernel"].as<std::string>();
    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(), [node](xrt::xclbin::kernel &k) {
        return k.get_name().rfind(node, 0) == 0;
    });
    device.register_xclbin(xclbin);
    xrt::hw_context context = [&] {
        if (gops == 0) return xrt::hw_context(device, xclbin.get_uuid());
        xrt::hw_context::cfg_param_type qos{{"gops", gops}, {"fps", fps}};
        return xrt::hw_context(device, xclbin.get_uuid(), qos);
    }();
    auto kernel = xrt::kernel(context, xkernel.get_name());

    size_t instr_bytes = instr_v.size() * sizeof(int);
    size_t instr_bo_bytes = std::max(instr_bytes, vm["instr-bo-bytes"].as<size_t>());
    auto bo_instr = xrt::bo(device, instr_bo_bytes, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_in = xrt::bo(device, 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_buf = xrt::bo(device, 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_out = xrt::bo(device, out_elems * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    memset(bo_in.map<uint8_t *>(), 0, 4);
    memset(bo_buf.map<uint8_t *>(), 0, 4);
    memset(bo_out.map<uint8_t *>(), 0, out_elems * sizeof(int32_t));
    memset(bo_instr.map<uint8_t *>(), 0, instr_bo_bytes);
    memcpy(bo_instr.map<void *>(), instr_v.data(), instr_bytes);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::this_thread::sleep_for(std::chrono::milliseconds(vm["pre-ms"].as<int>()));
    std::cout << "READY" << std::endl;

    auto t0 = std::chrono::steady_clock::now();
    auto run = kernel(3u, bo_instr, instr_v.size(), bo_in, bo_buf, bo_out);
    ert_cmd_state r = run.wait();
    auto t1 = std::chrono::steady_clock::now();
    if (r != ERT_CMD_STATE_COMPLETED) {
        std::cerr << "Kernel did not complete. Status: " << r << std::endl;
        return 1;
    }
    std::cout << "DONE wall_us="
              << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << std::endl;
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::ofstream ofs(vm["out-file"].as<std::string>(), std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(bo_out.map<uint8_t *>()), out_bytes);
    ofs.close();

    std::this_thread::sleep_for(std::chrono::milliseconds(vm["hold-ms"].as<int>()));
    std::cout << "EXIT" << std::endl;
    return 0;
}
