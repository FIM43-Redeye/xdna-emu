//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Host harness for the vec_pack_i16_s0 capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). Feeds a designed input batch
// (tools/vector_inputs.py) and checks the output against the values observed on
// NPU1 (Phoenix) silicon (tools/golden/silicon/vec_pack_i16_s0.json, captured with
// tools/capture_silicon.py). Class: pack.
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

static constexpr int N = 96;

static const int16_t IN[96] = {
    0, 1, -1, 126, 127, 128, 129, -126,
    -127, -128, -129, 254, 255, 256, 257, -254,
    -255, -256, -257, 383, 384, 32639, 32640, -32640,
    -32639, 16384, -16384, 32767, -32768, 32766, -32767, 14035,
    312, -4403, 154, -7119, 245, 18300, 153, 6740,
    469, 10683, -404, 17437, 177, -24442, 155, 22531,
    -510, 20326, 175, 3396, -319, 22244, -465, 11984,
    55, -9904, -107, -7840, 484, 815, -212, 26354,
    -424, -6829, -335, 31651, 424, -24211, 327, 11457,
    361, -17351, -144, -13631, -436, -7133, 46, 18492,
    -24, -30873, 168, 3099, -371, 6420, 298, 22492,
    -468, -21188, 125, -8119, -143, -20608, 413, 979,
};
static const int8_t EXP[96] = {
    0, 1, -1, 126, 127, -128, -127, -126,
    -127, -128, 127, -2, -1, 0, 1, 2,
    1, 0, -1, 127, -128, 127, -128, -128,
    -127, 0, 0, -1, 0, -2, 1, -45,
    56, -51, -102, 49, -11, 124, -103, 84,
    -43, -69, 108, 29, -79, -122, -101, 3,
    2, 102, -81, 68, -63, -28, 47, -48,
    55, 80, -107, 96, -28, 47, 44, -14,
    88, 83, -79, -93, -88, 109, 71, -63,
    105, 57, 112, -63, 76, 35, 46, 60,
    -24, 103, -88, 27, -115, 20, 42, -36,
    44, 60, 125, 73, 113, -128, -99, -45,
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
  auto bo_out = xrt::bo(device, N * sizeof(int8_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  int16_t *bufIn = bo_in.map<int16_t *>();
  int8_t *bufOut = bo_out.map<int8_t *>();

  std::memcpy(bufIn, IN, N * sizeof(int16_t));
  std::memset(bufOut, 0, N * sizeof(int8_t));

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
