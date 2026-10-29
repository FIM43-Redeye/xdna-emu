// Minimal reproducer for MSG_OP_CHAIN_EXEC_NPU (op 0x18) silent-drop
// on AMD Ryzen AI Phoenix (NPU1).
//
// Submits the `add_one_ctrl_packet` kernel (from the mlir-aie test
// suite, https://github.com/Xilinx/mlir-aie/tree/main/test/npu-xrt/add_one_ctrl_packet)
// with the control-packet input buffer (`bo_ctrlIn`) left zero-filled.
//
// The kernel's compute core acquires `input_lock0` and waits for a
// real control packet to release it (a populated `bo_ctrlIn` would
// carry that release).  With `bo_ctrlIn` zero, the core blocks
// indefinitely, no data ever reaches the shim S2MM output channel,
// and the firmware handler for op 0x18 (`MSG_OP_CHAIN_EXEC_NPU`)
// never raises its completion IRQ.  The driver's mailbox RX_TIMEOUT
// and/or TDR fires; `run.wait()` returns a non-COMPLETED state.
//
// Build (against installed XRT 2.23+):
//   g++ -std=c++17 -O2 op0x18_repro.cpp -o op0x18_repro \
//       -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
//       -lxrt_coreutil -lpthread -Wl,-rpath,/opt/xilinx/xrt/lib
//
// Run:
//   ./op0x18_repro <aie.xclbin> <insts.bin> [iterations=10]

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace {

std::vector<uint32_t> load_u32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto sz = static_cast<std::streamsize>(f.tellg());
    f.seekg(0);
    std::vector<uint32_t> out(sz / sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(out.data()), sz);
    return out;
}

int submit_once(xrt::device& device, xrt::xclbin& xclbin,
                const std::vector<uint32_t>& instr, int iter) {
    constexpr int OUT_SIZE = 64;            // i32 words, mirrors mlir-aie test.cpp
    constexpr uint32_t opcode = 3;          // DPU opcode

    // Fresh hw_context per iteration: when the firmware silent-drops,
    // TDR tears down the affected context.  A fresh context isolates
    // each iteration so we measure the bug's per-submission rate, not
    // a cascade effect.
    xrt::hw_context context(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(context, xclbin.get_kernels()[0].get_name());

    auto bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_ctrlOut = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                              XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_ctrlIn  = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                              XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_out     = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                              XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    std::memcpy(bo_instr.map<void*>(), instr.data(),
                instr.size() * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // bo_ctrlIn DELIBERATELY left zero-filled -- this is the trigger.
    std::memset(bo_ctrlIn.map<void*>(), 0, OUT_SIZE * sizeof(int32_t));
    bo_ctrlIn.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(opcode, bo_instr, instr.size(),
                      bo_ctrlOut, bo_ctrlIn, bo_out);
    const auto t0 = std::chrono::steady_clock::now();
    const auto state = run.wait(std::chrono::seconds(15));
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
    const char* tag = (state == ERT_CMD_STATE_COMPLETED) ? "COMPLETED" : "DROP";
    std::cout << "iter " << iter << ": state=" << static_cast<int>(state)
              << " (" << tag << ") elapsed_ms=" << ms << "\n";
    std::cout.flush();
    return (state == ERT_CMD_STATE_COMPLETED) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <aie.xclbin> <insts.bin> [iterations=10]\n";
        return 2;
    }
    const std::string xclbin_path = argv[1];
    const std::string insts_path  = argv[2];
    const int iterations = (argc >= 4) ? std::atoi(argv[3]) : 10;

    const auto instr = load_u32(insts_path);
    auto device = xrt::device(0u);
    auto xclbin = xrt::xclbin(xclbin_path);
    device.register_xclbin(xclbin);

    int drops = 0;
    for (int i = 0; i < iterations; ++i) {
        try {
            drops += submit_once(device, xclbin, instr, i);
        } catch (const std::exception& e) {
            std::cout << "iter " << i << ": exception: " << e.what() << "\n";
            ++drops;
        }
    }
    const double pct = 100.0 * drops / iterations;
    std::cout << "\n==== summary ====\n"
              << drops << "/" << iterations << " non-COMPLETED ("
              << pct << "%)\n";
    return drops > 0 ? 1 : 0;
}
