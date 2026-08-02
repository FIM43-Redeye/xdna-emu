// SPDX-License-Identifier: MIT

#define main context_repartition_cli_main
#include "context-repartition.cpp"
#undef main

int main() {
  try {
    static_cast<void>(wait_for_recovery_replay("/dev/null"));
  } catch (const std::runtime_error &error) {
    const std::string message = error.what();
    if (message.find("firmware-version query failed") != std::string::npos) {
      std::cout << "context-repartition barrier error path: PASS\n";
      return 0;
    }
    std::cerr << "unexpected barrier error: " << message << '\n';
    return 1;
  }

  std::cerr << "firmware-version query unexpectedly accepted /dev/null\n";
  return 1;
}
