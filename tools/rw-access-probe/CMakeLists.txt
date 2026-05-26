cmake_minimum_required(VERSION 3.20)
project(rw-access-probe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT install root")
include_directories(SYSTEM "${XRT_ROOT}/include")
link_directories("${XRT_ROOT}/lib")

add_executable(rw-access-probe rw-access-probe.cpp)
target_link_libraries(rw-access-probe PRIVATE xrt_coreutil)
target_compile_options(rw-access-probe PRIVATE -Wall -Wextra)
