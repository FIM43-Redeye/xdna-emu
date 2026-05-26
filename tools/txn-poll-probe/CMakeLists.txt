cmake_minimum_required(VERSION 3.20)
project(txn-poll-probe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT install root")
include_directories(SYSTEM "${XRT_ROOT}/include")
link_directories("${XRT_ROOT}/lib")

# drivers/accel uapi -- new tree with amdxdna_drm_hwctx_entry incl txn_op_idx
# and ctx_pc. The older src/include/uapi/drm_local/ header pre-dates these
# fields and the DRM_AMDXDNA_HW_CONTEXT_ALL param value, so we point at the
# current uapi explicitly.
set(XDNA_DRIVER_ROOT "/home/triple/npu-work/xdna-driver"
    CACHE PATH "xdna-driver source root")

add_executable(txn-poll-probe txn-poll-probe.cpp)
target_include_directories(txn-poll-probe SYSTEM PRIVATE
  "${XDNA_DRIVER_ROOT}/include/uapi")
target_link_libraries(txn-poll-probe PRIVATE xrt_coreutil uuid)
target_compile_options(txn-poll-probe PRIVATE -Wall -Wextra)
