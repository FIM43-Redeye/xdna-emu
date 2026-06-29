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

# XRT source tree (xdna-driver's pinned xrt submodule). Needed ONLY to reach
# the internal shim header core/common/shim/hwctx_handle.h, which is not
# shipped with the installed XRT and provides hwctx_handle::get_slotidx().
# Scoped to the isolated hwctx_slot target below (NOT a global include dir)
# so the source-tree xrt_kernel.h/xrt_bo.h -- which differ slightly from the
# installed headers -- never shadow the installed headers in the main runner.
set(XRT_SRC "/home/triple/npu-work/xdna-driver/xrt/src/runtime_src"
    CACHE PATH "XRT source runtime_src root (for internal shim headers)")

# Isolated TU: hw_context slot-id lookup via XRT internal shim header.
add_library(hwctx_slot STATIC hwctx_slot.cpp)
target_include_directories(hwctx_slot SYSTEM PRIVATE
    "${XRT_ROOT}/include"
    "${XRT_SRC}/core/include"
    "${XRT_SRC}")
target_compile_options(hwctx_slot PRIVATE -Wall -Wextra)

add_executable(bridge-trace-runner bridge-trace-runner.cpp)
# dl is for dlopen/dlsym -- we optionally load libxdna_emu.so at runtime
# for kernarg-role classification. Declared explicitly so the build
# doesn't rely on xrt_coreutil pulling it in transitively.
# hwctx_slot is listed before xrt_coreutil so the operator hwctx_handle*
# symbol it references is resolved from coreutil at final link.
target_link_libraries(bridge-trace-runner PRIVATE hwctx_slot xrt_coreutil uuid dl)
target_compile_options(bridge-trace-runner PRIVATE -Wall -Wextra)

install(TARGETS bridge-trace-runner RUNTIME DESTINATION bin)
