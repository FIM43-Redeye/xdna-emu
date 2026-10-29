cmake_minimum_required(VERSION 3.20)
project(validate-readback CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT install root")
include_directories(SYSTEM "${XRT_ROOT}/include")
link_directories("${XRT_ROOT}/lib")

# Out-of-tree amdxdna_accel.h is needed for the GET_ARRAY/COREDUMP IOCTL
# numbers and structs, which the in-tree (/usr/include/drm/) header lacks.
set(XDNA_DRIVER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../xdna-driver" CACHE PATH "xdna-driver source root")

set(MLIR_AIE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../mlir-aie"
    CACHE PATH "mlir-aie checkout holding the default test xclbins")

add_executable(validate-readback validate-readback.cpp)
target_compile_definitions(validate-readback PRIVATE MLIR_AIE_ROOT="${MLIR_AIE_ROOT}")
target_include_directories(validate-readback SYSTEM PRIVATE
  "${XDNA_DRIVER_ROOT}/src/include/uapi/drm_local")
target_link_libraries(validate-readback PRIVATE xrt_coreutil uuid)
target_compile_options(validate-readback PRIVATE -Wall -Wextra)
