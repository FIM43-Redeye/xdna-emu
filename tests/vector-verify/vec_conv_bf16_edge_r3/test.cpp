//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Host harness for the Half-B vec_conv_bf16_edge_r3 capture kernel (GENERATED -- edit the spec
// in vector_kernel_specs.py, not this file). Feeds a baked input batch and
// checks the output against the Half-A golden (bf16_srs slice). PASS means the
// vec_conv_bf16_edge_r3 datapath ran correctly. Expected values are the genuine
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

static constexpr int N = 96;

static const uint32_t IN[96] = {
    0, 1, 8192, 16384, 16385, 24576, 32767, 32768,
    32769, 40960, 49152, 65535, 65536, 81920, 98304, 114688,
    131072, 196608, 262144, 524287, 4161536, 4194304, 8323072, 8355839,
    8355840, 8355841, 8388607, 2139095040, 2139095041, 2139095104, 2139111424, 2139111425,
    2139127807, 2139127808, 2139127809, 2139160576, 2139193344, 2139619328, 2141192192, 2143256576,
    2143289344, 2147418112, 2147450879, 2147450880, 2147450881, 2147483647, 2147483648, 2147483649,
    2147491840, 2147500032, 2147500033, 2147508224, 2147516415, 2147516416, 2147516417, 2147524608,
    2147532800, 2147549183, 2147549184, 2147565568, 2147581952, 2147598336, 2147614720, 2147680256,
    2147745792, 2148007935, 2151645184, 2151677952, 2155806720, 2155839487, 2155839488, 2155839489,
    2155872255, 4286578688, 4286578689, 4286578752, 4286595072, 4286595073, 4286611455, 4286611456,
    4286611457, 4286644224, 4286676992, 4287102976, 4288675840, 4290740224, 4290772992, 4292342077,
    4293946125, 4294901760, 4294934527, 4294934528, 4294934529, 4294967295, 0, 0,
};
static const uint16_t EXP[96] = {
    0, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 2, 2, 2,
    2, 3, 4, 8, 64, 64, 127, 128,
    128, 128, 128, 32640, 32641, 32641, 32641, 32641,
    32641, 32641, 32641, 32641, 32641, 32648, 32672, 32703,
    32704, 32767, 32767, 32767, 32767, 32767, 32768, 32769,
    32769, 32769, 32769, 32769, 32769, 32769, 32769, 32769,
    32769, 32769, 32769, 32770, 32770, 32770, 32770, 32771,
    32772, 32776, 32832, 32832, 32895, 32896, 32896, 32896,
    32896, 65408, 65409, 65409, 65409, 65409, 65409, 65409,
    65409, 65409, 65409, 65416, 65440, 65471, 65472, 65495,
    65520, 65535, 65535, 65535, 65535, 65535, 0, 0,
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
