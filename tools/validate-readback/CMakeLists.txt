cmake_minimum_required(VERSION 3.20)
project(validate-readback CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT install root")
include_directories(SYSTEM "${XRT_ROOT}/include")
link_directories("${XRT_ROOT}/lib")

add_executable(validate-readback validate-readback.cpp)
target_link_libraries(validate-readback PRIVATE xrt_coreutil uuid)
target_compile_options(validate-readback PRIVATE -Wall -Wextra)
