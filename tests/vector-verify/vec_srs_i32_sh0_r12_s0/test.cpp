//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Host harness for the vec_srs_i32_sh0_r12_s0 capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). Feeds a designed input batch
// (tools/vector_inputs.py) and checks the output against the values observed on
// NPU1 (Phoenix) silicon (tools/golden/silicon/vec_srs_i32_sh0_r12_s0.json, captured with
// tools/capture_silicon.py). Class: srs.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

static constexpr int N = 64;

static const int32_t IN[64] = {
    0, 1, -1, 2147483647, -2147483648, 2147483646, -2147483647, 2,
    3, -2, 32767, 32766, -32768, -32767, 32768, 32769,
    -32769, 127, -128, 255, 256, 65535, -65536, 1073741824,
    -1073741824, 1608608141, -25016, 1219034607, 20885, 2122241293, -30636, -423450405,
    -2232, 436728260, -5301, -588844239, -23414, 1564250256, 1148, 506119234,
    -19244, 1538511831, 11344, 2070159529, 5815, -1542987124, -21511, 581617540,
    -20401, 1558333157, 451, -2106028648, 22067, -1572624594, -17926, 1530373564,
    8814, -1162805794, 4527, 1055883644, 20607, -1359057059, -9645, 912961969,
};
static const int16_t EXP[64] = {
    0, 1, -1, -1, 0, -2, 1, 2,
    3, -2, 32767, 32766, -32768, -32767, -32768, -32767,
    32767, 127, -128, 255, 256, -1, 0, 0,
    0, 27021, -25016, -529, 20885, -10995, -30636, -22309,
    -2232, -3644, -5301, -3279, -23414, -28528, 1148, -15294,
    -19244, -11305, 11344, 8361, 5815, -7540, -21511, -14460,
    -20401, 18149, 451, -29288, 22067, -22738, -17926, -23108,
    8814, -546, 4527, -32388, 20607, 28509, -9645, -20047,
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
  auto bo_in = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, N * sizeof(int16_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  int32_t *bufIn = bo_in.map<int32_t *>();
  int16_t *bufOut = bo_out.map<int16_t *>();

  std::memcpy(bufIn, IN, N * sizeof(int32_t));
  std::memset(bufOut, 0, N * sizeof(int16_t));

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

  // Always dump the raw output buffer (one value per line) so a HW run can be
  // captured as a silicon golden (tools/capture_silicon.py).
  {
    std::ofstream dump("out.txt");
    for (int i = 0; i < N; i++)
      dump << (int64_t)bufOut[i] << "\n";
  }

  int errors = 0;
  for (int i = 0; i < N; i++) {
    if (bufOut[i] != EXP[i]) {
      if (errors < 10)
        std::cout << "Error [" << i << "]: in=" << (int)IN[i]
                  << " got=" << (int)bufOut[i] << " != exp=" << (int)EXP[i] << "\n";
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
