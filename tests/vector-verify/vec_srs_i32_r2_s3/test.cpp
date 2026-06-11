//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Host harness for the Half-B vec_srs_i32_r2_s3 capture kernel (GENERATED -- edit the spec
// in vector_kernel_specs.py, not this file). Feeds a baked input batch and
// checks the output against the Half-A golden (srs slice). PASS means the
// vec_srs_i32_r2_s3 datapath ran correctly. Expected values are the genuine
// aietools-model outputs baked from tools/golden/vector_ops.json.
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
    0, 1, -255, 256, -256, -524288, -524287, 8388352,
    -524280, 8388353, 255, -8388608, -8388607, -8388480, 134213632, 134213633,
    16, 4096, -4096, -134217728, -134217727, 16777216, -16777216, 268435456,
    -268435456, 1048576, 134215680, -134215680, 65536, -65536, 134213631, -134217729,
    -1048576, 524271, -16, 524272, 524273, 8388351, 524280, -8388609,
    8388480, -1, -524289, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
};
static const int16_t EXP[64] = {
    0, 0, -15, 16, -16, -32767, -32767, 32767,
    -32767, 32767, 15, -32767, -32767, -32767, 32767, 32767,
    1, 256, -256, -32767, -32767, 32767, -32767, 32767,
    -32767, 32767, 32767, -32767, 4096, -4096, 32767, -32767,
    -32767, 32766, -1, 32767, 32767, 32767, 32767, -32767,
    32767, 0, -32767, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
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
  // captured as a silicon golden. Harmless for model-golden kernels.
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
