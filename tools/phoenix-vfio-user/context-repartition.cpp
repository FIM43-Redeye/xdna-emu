// SPDX-License-Identifier: MIT

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "drm/amdxdna_accel.h"
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

amdxdna_drm_query_firmware_version
wait_for_recovery_replay(const std::string &device_path) {
  const int fd = ::open(device_path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    throw std::runtime_error("cannot open " + device_path + ": " +
                             std::strerror(errno));

  amdxdna_drm_query_firmware_version version{};
  amdxdna_drm_get_info query{};
  query.param = DRM_AMDXDNA_QUERY_FIRMWARE_VERSION;
  query.buffer_size = sizeof(version);
  query.buffer = reinterpret_cast<uintptr_t>(&version);

  if (::ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &query) < 0) {
    const int error = errno;
    ::close(fd);
    throw std::runtime_error("firmware-version query failed: " +
                             std::string(std::strerror(error)));
  }
  if (::close(fd) < 0)
    throw std::runtime_error("cannot close " + device_path + ": " +
                             std::strerror(errno));
  return version;
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

  int run_observed(const char *marker, bool announce = true) {
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
    const auto state = run.wait();
    if (state != ERT_CMD_STATE_COMPLETED) {
      if (announce)
        std::cout << "PHOENIX_" << marker << "_STATE_"
                  << static_cast<int>(state) << std::endl;
      return static_cast<int>(state);
    }

    output_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    for (size_t i = 0; i < count_; ++i) {
      const auto expected = static_cast<uint32_t>(i + 1) + increment_;
      if (output[i] != expected)
        throw std::runtime_error(label_ + " output mismatch at " +
                                 std::to_string(i));
    }
    if (announce)
      std::cout << "PHOENIX_" << marker << "_PASS" << std::endl;
    return static_cast<int>(state);
  }

  void run(const char *marker) {
    if (run_observed(marker) != ERT_CMD_STATE_COMPLETED)
      throw std::runtime_error(label_ + " did not complete");
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
  const bool requested_repeat =
      argc > 1 && std::string(argv[1]) == "--same-context-repeat";
  const bool requested_tdr_retry =
      argc > 1 && std::string(argv[1]) == "--immediate-post-tdr-retry";
  const bool requested_post_replay =
      argc > 1 && std::string(argv[1]) == "--post-replay-tdr-retry";
  const bool same_context_repeat = requested_repeat && argc == 4;
  const bool immediate_post_tdr_retry = requested_tdr_retry && argc == 4;
  const bool post_replay_tdr_retry = requested_post_replay && argc == 5;
  const bool single_context_mode =
      same_context_repeat || immediate_post_tdr_retry || post_replay_tdr_retry;
  const bool requested_single_context =
      requested_repeat || requested_tdr_retry || requested_post_replay;
  if ((!requested_single_context && argc != 5) ||
      (requested_single_context && !single_context_mode)) {
    std::cerr << "usage: " << argv[0] << " A.xclbin A.insts B.xclbin B.insts\n"
              << "       " << argv[0]
              << " --same-context-repeat A.xclbin A.insts\n"
              << "       " << argv[0]
              << " --immediate-post-tdr-retry A.xclbin A.insts\n"
              << "       " << argv[0]
              << " --post-replay-tdr-retry DEVICE A.xclbin A.insts\n";
    return 2;
  }

  try {
    const int a_arg = post_replay_tdr_retry ? 3 : single_context_mode ? 2 : 1;
    xrt::device device(0);
    auto a = std::make_unique<Workload>(device, "A", argv[a_arg],
                                        argv[a_arg + 1], 64, 1);
    a->run(post_replay_tdr_retry      ? "POST_REPLAY_A1"
           : immediate_post_tdr_retry ? "TDR_RETRY_A1"
           : same_context_repeat      ? "CONTEXT_REPEAT_A1"
                                      : "REPARTITION_A1");

    if (post_replay_tdr_retry) {
      const int a2_state = a->run_observed("POST_REPLAY_A2", false);
      if (a2_state == ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("A2 unexpectedly completed");

      std::cout << "PHOENIX_POST_REPLAY_A2_STATE_" << a2_state << std::endl;
      const auto firmware = wait_for_recovery_replay(argv[2]);
      std::cout << "PHOENIX_POST_REPLAY_BARRIER_FW_" << firmware.major << '.'
                << firmware.minor << '.' << firmware.patch << '.'
                << firmware.build << std::endl;

      const int a3_state = a->run_observed("POST_REPLAY_A3", false);
      a.reset();
      std::cout << "PHOENIX_POST_REPLAY_A3_STATE_" << a3_state << std::endl;
      std::cout << "PHOENIX_POST_REPLAY_A_DESTROYED" << std::endl;
      if (a3_state == ERT_CMD_STATE_COMPLETED) {
        std::cout << "PHOENIX_POST_REPLAY_RETRY_PASS" << std::endl;
        return 0;
      }
      std::cout << "PHOENIX_POST_REPLAY_A3_NONCOMPLETION" << std::endl;
      return 3;
    }

    if (immediate_post_tdr_retry) {
      const int a2_state = a->run_observed("TDR_RETRY_A2", false);
      if (a2_state == ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("A2 unexpectedly completed");

      std::cout << "PHOENIX_TDR_RETRY_A2_STATE_" << a2_state << '\n';
      const int a3_state = a->run_observed("TDR_RETRY_A3", false);
      a.reset();
      std::cout << "PHOENIX_TDR_RETRY_A3_STATE_" << a3_state << std::endl;
      std::cout << "PHOENIX_TDR_RETRY_A_DESTROYED" << std::endl;
      if (a3_state == ERT_CMD_STATE_COMPLETED) {
        std::cout << "PHOENIX_TDR_RETRY_PASS" << std::endl;
        return 0;
      }
      std::cout << "PHOENIX_TDR_RETRY_A3_NONCOMPLETION" << std::endl;
      return 3;
    }

    if (same_context_repeat) {
      a->run("CONTEXT_REPEAT_A2");
      a.reset();
      std::cout << "PHOENIX_CONTEXT_REPEAT_A_DESTROYED" << std::endl;
      std::cout << "PHOENIX_CONTEXT_REPEAT_PASS" << std::endl;
      return 0;
    }

    std::cout << "PHOENIX_REPARTITION_B_CONSTRUCT_BEGIN" << std::endl;
    auto b =
        std::make_unique<Workload>(device, "B", argv[3], argv[4], 4096, 0);
    std::cout << "PHOENIX_REPARTITION_B_CONSTRUCT_END" << std::endl;
    b->run("REPARTITION_B");
    a->run("REPARTITION_A2");

    a.reset();
    std::cout << "PHOENIX_REPARTITION_A_DESTROYED" << std::endl;
    b.reset();
    std::cout << "PHOENIX_REPARTITION_B_DESTROYED" << std::endl;
    std::cout << "PHOENIX_REPARTITION_PASS" << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << (post_replay_tdr_retry ? "PHOENIX_POST_REPLAY_RETRY_FAIL: "
                  : immediate_post_tdr_retry ? "PHOENIX_TDR_RETRY_FAIL: "
                  : same_context_repeat      ? "PHOENIX_CONTEXT_REPEAT_FAIL: "
                                             : "PHOENIX_REPARTITION_FAIL: ")
              << error.what() << std::endl;
    return 1;
  }
}
