cmake_minimum_required(VERSION 3.20)
project(bridge-trace-runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# XRT comes from /opt/xilinx/xrt by default.
set(XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT install root")
include_directories(SYSTEM "${XRT_ROOT}/include")
link_directories("${XRT_ROOT}/lib")

add_executable(bridge-trace-runner bridge-trace-runner.cpp)
target_link_libraries(bridge-trace-runner PRIVATE xrt_coreutil uuid)
target_compile_options(bridge-trace-runner PRIVATE -Wall -Wextra)

install(TARGETS bridge-trace-runner RUNTIME DESTINATION bin)
