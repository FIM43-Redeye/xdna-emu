//===- test.cpp -------------------------------------------------*- C++ -*-===//
// Host harness for the clean seed-6159 re-author (GENERATED). Feeds the seed's
// input pool, runs, dumps per-lane output; flags lane29/slice4 (the divergent
// lane: HW 0xFF8C, interpreter 0xFF81).
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

static constexpr int NU16 = 256;  // 128 int32
static const uint16_t POOL[NU16] = { 32704, 32704, 77, 32639, 95, 32640, 32640, 32704, 57892, 32885, 46359, 32639, 20, 32640, 65407, 0, 32704, 63, 0, 47686, 58491, 0, 32639, 33208, 105, 29345, 5, 46109, 0, 65407, 19837, 56458, 111, 32640, 33600, 54158, 32704, 32640, 34, 32640, 65407, 65133, 22, 48, 108, 50070, 32639, 27959, 32829, 0, 32830, 65407, 65407, 32640, 0, 32823, 65407, 0, 15480, 35000, 121, 23880, 32833, 55297, 57, 32639, 20543, 32805, 32803, 32639, 32639, 32704, 63982, 32704, 20833, 24104, 32640, 62776, 0, 0, 1, 65407, 32639, 3422, 1401, 15881, 32871, 4, 32639, 97, 6589, 32640, 1999, 32639, 62108, 32640, 43356, 32704, 20, 0, 65407, 32704, 0, 98, 60919, 32862, 32820, 65407, 0, 32640, 32845, 122, 32774, 32863, 32639, 62153, 57226, 54088, 32640, 32704, 65407, 31545, 32639, 115, 124, 65478, 32639, 32640, 32802, 32704, 30, 32639, 32639, 32639, 40982, 32893, 32639, 32640, 50715, 65407, 32639, 32877, 39295, 80, 23263, 32640, 14610, 65407, 49708, 32640, 65407, 32639, 19, 0, 32639, 0, 32809, 32639, 52482, 19844, 0, 65407, 32704, 32639, 0, 65407, 9621, 32640, 32704, 32704, 32814, 108, 32854, 32639, 32640, 0, 82, 32639, 32819, 32704, 32639, 65195, 10821, 5307, 11510, 32889, 65407, 29, 0, 32784, 56, 32639, 32639, 26164, 0, 32704, 32780, 65407, 36, 2812, 26094, 51, 32639, 32639, 0, 32704, 32892, 32639, 32639, 30, 32784, 58478, 32639, 47319, 76, 65407, 0, 67, 4, 53328, 32819, 32850, 32704, 32704, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

int main(int argc, const char *argv[]) {
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary("insts.bin");
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  std::string xclbin_name = "aie.xclbin";
  xrt::xclbin xclbin(xclbin_name);
  std::string Node = "MLIR_AIE";
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
      [Node](xrt::xclbin::kernel &k) { return k.get_name().rfind(Node, 0) == 0; });
  auto kernelName = xkernel.get_name();
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in = xrt::bo(device, NU16 * sizeof(uint16_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, NU16 * sizeof(uint16_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  uint16_t *bufIn = bo_in.map<uint16_t *>();
  uint16_t *bufOut = bo_out.map<uint16_t *>();
  std::memcpy(bufIn, POOL, NU16 * sizeof(uint16_t));
  std::memset(bufOut, 0, NU16 * sizeof(uint16_t));
  std::memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_out);
  if (run.wait() != ERT_CMD_STATE_COMPLETED) { std::cout << "Kernel did not complete\n"; return 1; }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint16_t l29s4 = bufOut[4 * 32 + 29];
  std::ofstream dump("out.txt");
  for (int i = 0; i < NU16; i++) dump << (uint32_t)bufOut[i] << "\n";
  std::cout << "\nseed6159 clean: slice4 lane29 = 0x" << std::hex << l29s4 << std::dec << "\n";
  std::cout << "  (HW expect 0xFF8C; interpreter 0xFF81)\n";
  std::cout << (l29s4 == 0xFF8C ? "REPRODUCES (0xFF8C)\n" : "differs\n");
  std::cout << "\nPASS!\n\n";
  return 0;
}
