// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace {

std::vector<uint32_t> load_instructions(const std::string &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open " + path);

  const auto bytes = input.tellg();
  if (bytes <= 0 || bytes % sizeof(uint32_t) != 0)
    throw std::runtime_error("invalid instruction binary " + path);

  std::vector<uint32_t> words(static_cast<size_t>(bytes) / sizeof(uint32_t));
  input.seekg(0);
  input.read(reinterpret_cast<char *>(words.data()), bytes);
  if (!input)
    throw std::runtime_error("cannot read " + path);
  return words;
}

class Workload {
public:
  Workload(xrt::device &device, std::string label,
           const std::string &xclbin_path, const std::string &instruction_path,
           size_t count, uint32_t increment)
      : device_(device), label_(std::move(label)), count_(count),
        increment_(increment), instructions_(load_instructions(instruction_path)),
        xclbin_(xclbin_path) {
    device_.register_xclbin(xclbin_);
    context_ = xrt::hw_context(device_, xclbin_.get_uuid());

    const auto kernels = xclbin_.get_kernels();
    if (kernels.empty())
      throw std::runtime_error(label_ + " xclbin has no kernel");
    kernel_ = xrt::kernel(context_, kernels.front().get_name());

    instruction_bo_ =
        xrt::bo(device_, instructions_.size() * sizeof(uint32_t),
                xrt::bo::flags::cacheable, kernel_.group_id(1));
    input_bo_ = xrt::bo(device_, count_ * sizeof(uint32_t),
                        xrt::bo::flags::host_only, kernel_.group_id(3));
    unused_bo_ = xrt::bo(device_, count_ * sizeof(uint32_t),
                         xrt::bo::flags::host_only, kernel_.group_id(4));
    output_bo_ = xrt::bo(device_, count_ * sizeof(uint32_t),
                         xrt::bo::flags::host_only, kernel_.group_id(5));

    std::memcpy(instruction_bo_.map<void *>(), instructions_.data(),
                instructions_.size() * sizeof(uint32_t));
    instruction_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  void run(const char *phase) {
    auto *input = input_bo_.map<uint32_t *>();
    auto *output = output_bo_.map<uint32_t *>();
    for (size_t i = 0; i < count_; ++i) {
      input[i] = static_cast<uint32_t>(i + 1);
      output[i] = 0xdeadbeef;
    }
    input_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    output_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel_(3, instruction_bo_, instructions_.size(), input_bo_,
                       unused_bo_, output_bo_);
    if (run.wait() != ERT_CMD_STATE_COMPLETED)
      throw std::runtime_error(label_ + " did not complete");

    output_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    for (size_t i = 0; i < count_; ++i) {
      const auto expected = static_cast<uint32_t>(i + 1) + increment_;
      if (output[i] != expected)
        throw std::runtime_error(label_ + " output mismatch at " +
                                 std::to_string(i));
    }
    std::cout << "PHOENIX_REPARTITION_" << phase << "_PASS" << std::endl;
  }

private:
  xrt::device &device_;
  std::string label_;
  size_t count_;
  uint32_t increment_;
  std::vector<uint32_t> instructions_;
  xrt::xclbin xclbin_;
  xrt::hw_context context_;
  xrt::kernel kernel_;
  xrt::bo instruction_bo_;
  xrt::bo input_bo_;
  xrt::bo unused_bo_;
  xrt::bo output_bo_;
};

} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: " << argv[0]
              << " A.xclbin A.insts B.xclbin B.insts\n";
    return 2;
  }

  try {
    xrt::device device(0);
    auto a = std::make_unique<Workload>(device, "A", argv[1], argv[2], 64, 1);
    a->run("A1");

    std::cout << "PHOENIX_REPARTITION_B_CONSTRUCT_BEGIN" << std::endl;
    auto b =
        std::make_unique<Workload>(device, "B", argv[3], argv[4], 4096, 0);
    std::cout << "PHOENIX_REPARTITION_B_CONSTRUCT_END" << std::endl;
    b->run("B");
    a->run("A2");

    a.reset();
    std::cout << "PHOENIX_REPARTITION_A_DESTROYED" << std::endl;
    b.reset();
    std::cout << "PHOENIX_REPARTITION_B_DESTROYED" << std::endl;
    std::cout << "PHOENIX_REPARTITION_PASS" << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "PHOENIX_REPARTITION_FAIL: " << error.what() << std::endl;
    return 1;
  }
}
