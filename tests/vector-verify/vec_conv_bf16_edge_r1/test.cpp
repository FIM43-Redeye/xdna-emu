//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Host harness for the vec_conv_bf16_edge_r1 capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). Feeds a designed input batch
// (tools/vector_inputs.py) and checks the output against the values observed on
// NPU1 (Phoenix) silicon (tools/golden/silicon/vec_conv_bf16_edge_r1.json, captured with
// tools/capture_silicon.py). Class: convert.
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

static const uint32_t IN[96] = {
    0, 2147483648, 1, 2147483649, 127, 2147483775, 16384, 2147500032,
    32767, 2147516415, 32768, 2147516416, 32769, 2147516417, 65535, 2147549183,
    65536, 2147549184, 98303, 2147581951, 98304, 2147581952, 98305, 2147581953,
    131071, 2147614719, 131072, 2147614720, 163840, 2147647488, 4194303, 2151677951,
    4194304, 2151677952, 4227072, 2151710720, 8355839, 2155839487, 8355840, 2155839488,
    8355841, 2155839489, 8388607, 2155872255, 8323072, 2155806720, 2139095040, 4286578688,
    2139095041, 4286578689, 2139160575, 4286644223, 2139127808, 4286611456, 2143289343, 4290772991,
    2143289344, 4290772992, 2143289345, 4290772993, 2143322112, 4290805760, 2147483647, 4294967295,
    2141192192, 4288675840, 2139160576, 4286644224, 3493047, 2152267776, 4292104327, 2142707596,
    2139133972, 2153149533, 2149881521, 2141705164, 2155758354, 2151982348, 2151972864, 2149260326,
    2142558975, 2148771571, 2151330883, 2153192949, 5134974, 4290890141, 2152490536, 6947149,
    2343786, 2152566740, 2141129921, 2151257473, 2149198433, 2148301722, 6139556, 6183881,
};
static const uint16_t EXP[96] = {
    0, 32768, 1, 32768, 1, 32768, 1, 32768,
    1, 32768, 1, 32768, 1, 32768, 1, 32768,
    1, 32769, 2, 32769, 2, 32769, 2, 32769,
    2, 32769, 2, 32770, 3, 32770, 64, 32831,
    64, 32832, 65, 32832, 128, 32895, 128, 32895,
    128, 32895, 128, 32895, 127, 32895, 32640, 65408,
    32641, 65409, 32641, 65409, 32641, 65409, 32703, 65471,
    32704, 65472, 32704, 65472, 32704, 65472, 32767, 65535,
    32672, 65440, 32641, 65409, 54, 32841, 65492, 32695,
    32641, 32854, 32804, 32679, 32894, 32836, 32836, 32795,
    32692, 32787, 32826, 32855, 79, 65473, 32844, 107,
    36, 32845, 32671, 32825, 32794, 32780, 94, 95,
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
  auto bo_in = xrt::bo(device, N * sizeof(uint32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, N * sizeof(uint16_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  uint32_t *bufIn = bo_in.map<uint32_t *>();
  uint16_t *bufOut = bo_out.map<uint16_t *>();

  std::memcpy(bufIn, IN, N * sizeof(uint32_t));
  std::memset(bufOut, 0, N * sizeof(uint16_t));

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
