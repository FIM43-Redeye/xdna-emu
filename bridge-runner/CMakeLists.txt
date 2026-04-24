cmake_minimum_required(VERSION 3.20)
project(bridge-trace-runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Emit compile_commands.json so clangd/other IDE tooling can resolve XRT
# includes without hand-maintained .clangd configuration.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# XRT comes from /opt/xilinx/xrt by default.
set(XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT install root")
include_directories(SYSTEM "${XRT_ROOT}/include")
link_directories("${XRT_ROOT}/lib")

add_executable(bridge-trace-runner bridge-trace-runner.cpp)
# dl is for dlopen/dlsym -- we optionally load libxdna_emu.so at runtime
# for kernarg-role classification. Declared explicitly so the build
# doesn't rely on xrt_coreutil pulling it in transitively.
target_link_libraries(bridge-trace-runner PRIVATE xrt_coreutil uuid dl)
target_compile_options(bridge-trace-runner PRIVATE -Wall -Wextra)

install(TARGETS bridge-trace-runner RUNTIME DESTINATION bin)
