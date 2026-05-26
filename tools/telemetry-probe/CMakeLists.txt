cmake_minimum_required(VERSION 3.20)
project(telemetry-probe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(XDNA_DRIVER_ROOT "/home/triple/npu-work/xdna-driver"
    CACHE PATH "xdna-driver source root")

add_executable(telemetry-probe telemetry-probe.cpp)
target_include_directories(telemetry-probe SYSTEM PRIVATE
  "${XDNA_DRIVER_ROOT}/include/uapi")
target_compile_options(telemetry-probe PRIVATE -Wall -Wextra)
