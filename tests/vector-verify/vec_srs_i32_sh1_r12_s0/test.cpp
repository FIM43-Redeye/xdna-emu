//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Host harness for the vec_srs_i32_sh1_r12_s0 capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). Feeds a designed input batch
// (tools/vector_inputs.py) and checks the output against the values observed on
// NPU1 (Phoenix) silicon (tools/golden/silicon/vec_srs_i32_sh1_r12_s0.json, captured with
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
    3, 4, 5, 6, -2, -4, -3, 65534,
    65535, 65532, 65533, -65536, -65535, -65534, -65533, 65536,
    65537, 65538, -65538, -65537, 1608608141, -50032, 1219034607, 41771,
    2122241293, -61272, -423450405, -4463, 436728260, -10602, -588844239, -46827,
    1564250256, 2297, 506119234, -38487, 1538511831, 22689, 2070159529, 11630,
    -1542987124, -43021, 581617540, -40802, 1558333157, 902, -2106028648, 44135,
    -1572624594, -35852, 1530373564, 17629, -1162805794, 9054, 1055883644, 41215,
};
static const int16_t EXP[64] = {
    0, 0, 0, 0, 0, -1, 0, 1,
    2, 2, 2, 3, -1, -2, -2, 32767,
    -32768, 32766, 32766, -32768, -32768, -32767, -32766, -32768,
    -32768, -32767, 32767, -32768, -19258, -25016, 32504, 20886,
    27270, -30636, 21614, -2232, -1822, -5301, 31128, -23414,
    18504, 1148, 25121, -19244, -5652, 11344, 4180, 5815,
    -3770, -21510, 25538, -20401, 9074, 451, 18124, 22068,
    -11369, -17926, -11554, 8814, 32495, 4527, -16194, 20608,
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
