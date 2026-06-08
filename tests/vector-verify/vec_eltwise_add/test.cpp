//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Host harness for the Half-B element-wise vector-add capture kernel. Fills two
// 8xi32 inputs with deterministic values, launches, and checks the output
// against the reference a+b (int32 wraparound -- identical to the Half-A
// `vadd_Int32` golden semantics). PASS means the emulator (or real NPU) ran the
// vector-add datapath correctly.
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

using T = int32_t;
static constexpr int N = 8; // one 256-bit vector register (8 x int32)

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
  auto bo_a = xrt::bo(device, N * sizeof(T), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b = xrt::bo(device, N * sizeof(T), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c = xrt::bo(device, N * sizeof(T), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  T *bufA = bo_a.map<T *>();
  T *bufB = bo_b.map<T *>();
  T *bufC = bo_c.map<T *>();

  std::vector<T> a(N), b(N), ref(N);
  for (int i = 0; i < N; i++) {
    a[i] = (i + 1) * 1000 - 3;       // varied, signed-friendly
    b[i] = -(i * 7) + 42;
    ref[i] = static_cast<T>(static_cast<uint32_t>(a[i]) + static_cast<uint32_t>(b[i])); // wraparound
  }
  std::memcpy(bufA, a.data(), N * sizeof(T));
  std::memcpy(bufB, b.data(), N * sizeof(T));
  std::memset(bufC, 0, N * sizeof(T));

  void *bufInstr = bo_instr.map<void *>();
  std::memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Status: " << r << "\n";
    return 1;
  }

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;
  for (int i = 0; i < N; i++) {
    if (bufC[i] != ref[i]) {
      if (errors < 10)
        std::cout << "Error [" << i << "]: a=" << a[i] << " b=" << b[i]
                  << " got=" << bufC[i] << " != ref=" << ref[i] << "\n";
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
