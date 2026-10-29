cmake_minimum_required(VERSION 3.16)
project(npu_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# XRT is typically installed at /opt/xilinx/xrt
set(XRT_DIR "/opt/xilinx/xrt" CACHE PATH "Path to XRT installation")

find_path(XRT_INCLUDE_DIR
    NAMES xrt/xrt_device.h
    HINTS ${XRT_DIR}/include
    REQUIRED
)

find_library(XRT_COREUTIL_LIB
    NAMES xrt_coreutil
    HINTS ${XRT_DIR}/lib
    REQUIRED
)

add_executable(npu_runner npu_runner.cpp)
target_include_directories(npu_runner PRIVATE ${XRT_INCLUDE_DIR})
target_link_libraries(npu_runner PRIVATE ${XRT_COREUTIL_LIB})

# Install to the build directory for easy discovery
install(TARGETS npu_runner DESTINATION ${CMAKE_BINARY_DIR})
