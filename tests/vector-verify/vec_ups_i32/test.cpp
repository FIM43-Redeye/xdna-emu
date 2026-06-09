//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Host harness for the Half-B vec_ups_i32 capture kernel (GENERATED -- edit the spec
// in vector_kernel_specs.py, not this file). Feeds a baked input batch and
// checks the output against the Half-A golden (ups slice). PASS means the
// vec_ups_i32 datapath ran correctly. Expected values are the genuine
// aietools-model outputs baked from tools/golden/vector_ops.json.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

static constexpr int N = 48;

static const int16_t IN[48] = {
    0, 1, -32768, -32767, 2, 4, 128, -128,
    8, 256, 4096, -4096, 8192, 16, 512, 16384,
    32, 1024, 64, -64, 2048, -2048, -32, -1024,
    -16, -512, -2, -1, -8, -256, -8192, -16384,
    -4, 32766, 32767, -6624, 18483, 24344, 15741, -26765,
    11687, 18556, 6909, 15451, 10180, 0, 0, 0,
};
static const int32_t EXP[48] = {
    0, 16, -524288, -524272, 32, 64, 2048, -2048,
    128, 4096, 65536, -65536, 131072, 256, 8192, 262144,
    512, 16384, 1024, -1024, 32768, -32768, -512, -16384,
    -256, -8192, -32, -16, -128, -4096, -131072, -262144,
    -64, 524256, 524272, -105984, 295728, 389504, 251856, -428240,
    186992, 296896, 110544, 247216, 162880, 0, 0, 0,
};

int main(int argc, const char *argv[]) {
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary("insts.bin");

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  std::string xclbin_name = "aie.xclbin";
  xrt::xclbin xclbin(xclbin_name);
  std::string Node = "MLIR_AIE";

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 return k.get_name().rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in = xrt::bo(device, N * sizeof(int16_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  int16_t *bufIn = bo_in.map<int16_t *>();
  int32_t *bufOut = bo_out.map<int32_t *>();

  std::memcpy(bufIn, IN, N * sizeof(int16_t));
  std::memset(bufOut, 0, N * sizeof(int32_t));

  void *bufInstr = bo_instr.map<void *>();
  std::memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;
  for (int i = 0; i < N; i++) {
    if (bufOut[i] != EXP[i]) {
      if (errors < 10)
        std::cout << "Error [" << i << "]: in=" << IN[i]
                  << " got=" << bufOut[i] << " != exp=" << EXP[i] << "\n";
      errors++;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed (" << errors << " errors).\n\n";
  return 1;
}
