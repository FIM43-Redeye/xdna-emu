//===- vv_host.cpp ----------------------------------------------*- C++ -*-===//
//
// Shared host harness for the vector-verify capture kernels. The generators
// emit no per-kernel test.cpp; tools/stage_vector_kernel.py copies this file in
// as test.cpp and writes the data it loads:
//
//   in0.bin, in1.bin   raw little-endian input buffers (VV_NIN of them)
//   exp.bin            the expected output (the silicon golden); when absent
//                      the kernel is uncaptured: the harness still runs and
//                      dumps out.txt, then fails (the bootstrap capture run)
//
// Per-kernel knobs arrive as -D flags on the clang line of the kernel's
// run.lit:
//
//   VV_DATA_DIR   string literal, directory holding the .bin files
//   VV_NIN        number of input buffers (1 or 2)
//   VV_OUT_T      C element type of the output (printing and compare width)
//   VV_NOUT       number of output elements
//
// Input and output sizes come from the file lengths and VV_NOUT; the input
// element type is not needed because the inputs move as raw bytes.
//
// The device-selection and trace patches (scripts/emu-bridge-test.sh,
// tools/cpp_trace_patch.py) match the forms used below: `unsigned int
// device_index`, `auto device = xrt::device(device_index)`, a direct
// `auto run = kernel(...)` call, `run.wait()`, and `xrt::bo` declarations.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

#if !defined(VV_DATA_DIR) || !defined(VV_NIN) || !defined(VV_OUT_T) || \
    !defined(VV_NOUT)
#error "vv_host.cpp needs -DVV_DATA_DIR -DVV_NIN -DVV_OUT_T -DVV_NOUT"
#endif

// The kernel call lists its buffers positionally; the input list is fixed per
// shape, so it is spelled by a macro instead of a branch inside main().
#if VV_NIN == 1
#define VV_IN_BOS bo_in[0]
#elif VV_NIN == 2
#define VV_IN_BOS bo_in[0], bo_in[1]
#else
#error "VV_NIN must be 1 or 2"
#endif

using out_t = VV_OUT_T;
static constexpr int NIN = VV_NIN;
static constexpr int NOUT = VV_NOUT;

// Whole file as bytes. Returns false when it cannot be opened.
static bool load_bytes(const std::string &path, std::vector<uint8_t> &bytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return false;
  bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return true;
}

int main(int argc, const char *argv[]) {
  const std::string data_dir = VV_DATA_DIR;

  std::vector<std::vector<uint8_t>> in_data(NIN);
  for (int i = 0; i < NIN; i++) {
    std::string path = data_dir + "/in" + std::to_string(i) + ".bin";
    if (!load_bytes(path, in_data[i]) || in_data[i].empty()) {
      std::cout << "Missing or empty input data: " << path
                << " (run tools/stage_vector_kernel.py)\n";
      return 1;
    }
  }
  // exp.bin is absent for an uncaptured kernel; that is the bootstrap run.
  std::vector<uint8_t> exp_data;
  const bool have_exp = load_bytes(data_dir + "/exp.bin", exp_data);
  if (have_exp && exp_data.size() != NOUT * sizeof(out_t)) {
    std::cout << "exp.bin is " << exp_data.size() << " bytes, expected "
              << NOUT * sizeof(out_t) << "\n";
    return 1;
  }

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

  // Group ids: 1 is the instruction BO, 3.. the data buffers in order (inputs,
  // then the output).
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  std::vector<xrt::bo> bo_in;
  for (int i = 0; i < NIN; i++)
    bo_in.emplace_back(device, in_data[i].size(), XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(3 + i));
  auto bo_out = xrt::bo(device, NOUT * sizeof(out_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3 + NIN));

  for (int i = 0; i < NIN; i++)
    std::memcpy(bo_in[i].map<void *>(), in_data[i].data(), in_data[i].size());
  std::memset(bo_out.map<void *>(), 0, NOUT * sizeof(out_t));

  void *bufInstr = bo_instr.map<void *>();
  std::memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  for (int i = 0; i < NIN; i++)
    bo_in[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), VV_IN_BOS, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Always dump the raw output buffer (one value per line) so a HW run can be
  // captured as a silicon golden (tools/capture_silicon.py and the generators'
  // capture subcommands).
  out_t *bufOut = bo_out.map<out_t *>();
  {
    std::ofstream dump("out.txt");
    for (int i = 0; i < NOUT; i++)
      dump << (int64_t)bufOut[i] << "\n";
  }

  if (!have_exp) {
    std::cout << "\nfailed: no exp.bin (uncaptured kernel; out.txt written).\n\n";
    return 1;
  }

  const out_t *exp = reinterpret_cast<const out_t *>(exp_data.data());
  int errors = 0;
  for (int i = 0; i < NOUT; i++) {
    if (bufOut[i] != exp[i]) {
      // Unary + promotes 8-bit types so they print as numbers.
      if (errors < 10)
        std::cout << "Error [" << i << "]: got=" << +bufOut[i]
                  << " != exp=" << +exp[i] << "\n";
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
