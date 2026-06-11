//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Host harness for the Half-B vec_convexp_bf16_denorm capture kernel (GENERATED -- edit the spec
// in vector_kernel_specs.py, not this file). Feeds a baked input batch and
// checks the output against the Half-A golden (direct slice). PASS means the
// vec_convexp_bf16_denorm datapath ran correctly. Expected values are the genuine
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

static constexpr int N = 256;

static const uint16_t IN[256] = {
    1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 64,
    65, 66, 67, 68, 69, 70, 71, 72,
    73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86, 87, 88,
    89, 90, 91, 92, 93, 94, 95, 96,
    97, 98, 99, 100, 101, 102, 103, 104,
    105, 106, 107, 108, 109, 110, 111, 112,
    113, 114, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 125, 126, 127, 32769,
    32770, 32771, 32772, 32773, 32774, 32775, 32776, 32777,
    32778, 32779, 32780, 32781, 32782, 32783, 32784, 32785,
    32786, 32787, 32788, 32789, 32790, 32791, 32792, 32793,
    32794, 32795, 32796, 32797, 32798, 32799, 32800, 32801,
    32802, 32803, 32804, 32805, 32806, 32807, 32808, 32809,
    32810, 32811, 32812, 32813, 32814, 32815, 32816, 32817,
    32818, 32819, 32820, 32821, 32822, 32823, 32824, 32825,
    32826, 32827, 32828, 32829, 32830, 32831, 32832, 32833,
    32834, 32835, 32836, 32837, 32838, 32839, 32840, 32841,
    32842, 32843, 32844, 32845, 32846, 32847, 32848, 32849,
    32850, 32851, 32852, 32853, 32854, 32855, 32856, 32857,
    32858, 32859, 32860, 32861, 32862, 32863, 32864, 32865,
    32866, 32867, 32868, 32869, 32870, 32871, 32872, 32873,
    32874, 32875, 32876, 32877, 32878, 32879, 32880, 32881,
    32882, 32883, 32884, 32885, 32886, 32887, 32888, 32889,
    32890, 32891, 32892, 32893, 32894, 32895, 0, 32768,
};
static const uint32_t EXP[256] = {
    65536, 131072, 196608, 262144, 327680, 393216, 458752, 524288,
    589824, 655360, 720896, 786432, 851968, 917504, 983040, 1048576,
    1114112, 1179648, 1245184, 1310720, 1376256, 1441792, 1507328, 1572864,
    1638400, 1703936, 1769472, 1835008, 1900544, 1966080, 2031616, 2097152,
    2162688, 2228224, 2293760, 2359296, 2424832, 2490368, 2555904, 2621440,
    2686976, 2752512, 2818048, 2883584, 2949120, 3014656, 3080192, 3145728,
    3211264, 3276800, 3342336, 3407872, 3473408, 3538944, 3604480, 3670016,
    3735552, 3801088, 3866624, 3932160, 3997696, 4063232, 4128768, 4194304,
    4259840, 4325376, 4390912, 4456448, 4521984, 4587520, 4653056, 4718592,
    4784128, 4849664, 4915200, 4980736, 5046272, 5111808, 5177344, 5242880,
    5308416, 5373952, 5439488, 5505024, 5570560, 5636096, 5701632, 5767168,
    5832704, 5898240, 5963776, 6029312, 6094848, 6160384, 6225920, 6291456,
    6356992, 6422528, 6488064, 6553600, 6619136, 6684672, 6750208, 6815744,
    6881280, 6946816, 7012352, 7077888, 7143424, 7208960, 7274496, 7340032,
    7405568, 7471104, 7536640, 7602176, 7667712, 7733248, 7798784, 7864320,
    7929856, 7995392, 8060928, 8126464, 8192000, 8257536, 8323072, 2147549184,
    2147614720, 2147680256, 2147745792, 2147811328, 2147876864, 2147942400, 2148007936, 2148073472,
    2148139008, 2148204544, 2148270080, 2148335616, 2148401152, 2148466688, 2148532224, 2148597760,
    2148663296, 2148728832, 2148794368, 2148859904, 2148925440, 2148990976, 2149056512, 2149122048,
    2149187584, 2149253120, 2149318656, 2149384192, 2149449728, 2149515264, 2149580800, 2149646336,
    2149711872, 2149777408, 2149842944, 2149908480, 2149974016, 2150039552, 2150105088, 2150170624,
    2150236160, 2150301696, 2150367232, 2150432768, 2150498304, 2150563840, 2150629376, 2150694912,
    2150760448, 2150825984, 2150891520, 2150957056, 2151022592, 2151088128, 2151153664, 2151219200,
    2151284736, 2151350272, 2151415808, 2151481344, 2151546880, 2151612416, 2151677952, 2151743488,
    2151809024, 2151874560, 2151940096, 2152005632, 2152071168, 2152136704, 2152202240, 2152267776,
    2152333312, 2152398848, 2152464384, 2152529920, 2152595456, 2152660992, 2152726528, 2152792064,
    2152857600, 2152923136, 2152988672, 2153054208, 2153119744, 2153185280, 2153250816, 2153316352,
    2153381888, 2153447424, 2153512960, 2153578496, 2153644032, 2153709568, 2153775104, 2153840640,
    2153906176, 2153971712, 2154037248, 2154102784, 2154168320, 2154233856, 2154299392, 2154364928,
    2154430464, 2154496000, 2154561536, 2154627072, 2154692608, 2154758144, 2154823680, 2154889216,
    2154954752, 2155020288, 2155085824, 2155151360, 2155216896, 2155282432, 2155347968, 2155413504,
    2155479040, 2155544576, 2155610112, 2155675648, 2155741184, 2155806720, 0, 2147483648,
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
  auto bo_in = xrt::bo(device, N * sizeof(uint16_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, N * sizeof(uint32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  uint16_t *bufIn = bo_in.map<uint16_t *>();
  uint32_t *bufOut = bo_out.map<uint32_t *>();

  std::memcpy(bufIn, IN, N * sizeof(uint16_t));
  std::memset(bufOut, 0, N * sizeof(uint32_t));

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
