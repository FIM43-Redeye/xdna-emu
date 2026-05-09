cmake_minimum_required(VERSION 3.16)
project(xrt_driver_emu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# XRT install root (system-wide).
set(XRT_INSTALL_DIR "/opt/xilinx/xrt" CACHE PATH "XRT install prefix")

# xdna-driver source tree.  Defaults to the sibling directory in npu-work.
# Override with -DXDNA_DRIVER_DIR=/path/to/xdna-driver on the cmake line.
# NOTE: Two patches are required -- see ../patches/ for details.
set(XDNA_DRIVER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../xdna-driver" CACHE PATH
    "Path to the xdna-driver source tree")

# Derived paths inside xdna-driver.
set(XRT_RUNTIME_SRC "${XDNA_DRIVER_DIR}/xrt/src/runtime_src")
set(XDNA_SHIM_DIR   "${XDNA_DRIVER_DIR}/src/shim")
set(XDNA_UAPI_DIR   "${XDNA_DRIVER_DIR}/src/include/uapi")

# ---------------------------------------------------------------------------
# Find XRT libraries
# ---------------------------------------------------------------------------

find_library(XRT_CORE_LIB     xrt_core     PATHS "${XRT_INSTALL_DIR}/lib" NO_DEFAULT_PATH)
find_library(XRT_COREUTIL_LIB xrt_coreutil PATHS "${XRT_INSTALL_DIR}/lib" NO_DEFAULT_PATH)

if(NOT XRT_CORE_LIB)
  message(FATAL_ERROR "libxrt_core.so not found in ${XRT_INSTALL_DIR}/lib")
endif()
if(NOT XRT_COREUTIL_LIB)
  message(FATAL_ERROR "libxrt_coreutil.so not found in ${XRT_INSTALL_DIR}/lib")
endif()

message(STATUS "XRT core:     ${XRT_CORE_LIB}")
message(STATUS "XRT coreutil: ${XRT_COREUTIL_LIB}")

# ---------------------------------------------------------------------------
# Plugin shared library
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shim base sources from xdna-driver.
# We compile the common shim infrastructure plus the KMQ backend (which
# is lightweight and works with our platform_drv_emu).  UMQ, host, and
# virtio backends are NOT compiled -- UMQ stubs satisfy the linker.
# ---------------------------------------------------------------------------

set(SHIM_SOURCES
  ${XDNA_SHIM_DIR}/pcidrv.cpp
  ${XDNA_SHIM_DIR}/pcidev.cpp
  ${XDNA_SHIM_DIR}/platform.cpp
  ${XDNA_SHIM_DIR}/device.cpp
  ${XDNA_SHIM_DIR}/hwctx.cpp
  ${XDNA_SHIM_DIR}/hwq.cpp
  ${XDNA_SHIM_DIR}/buffer.cpp
  ${XDNA_SHIM_DIR}/fence.cpp
  ${XDNA_SHIM_DIR}/smi_xdna.cpp
)

add_library(xrt_driver_emu SHARED
  src/transport_inprocess.cpp
  src/drv_emu.cpp
  src/pdev_emu.cpp
  src/platform_emu.cpp
  src/umq_stubs.cpp
  ${SHIM_SOURCES}
  # KMQ backend -- lightweight, works with platform_drv_emu.
  ${XDNA_SHIM_DIR}/kmq/hwctx.cpp
  ${XDNA_SHIM_DIR}/kmq/hwq.cpp
)

set_target_properties(xrt_driver_emu PROPERTIES
  VERSION   2.21.0
  SOVERSION 2
)

# -Bsymbolic-functions forces the dynamic linker to resolve calls within
# this .so to our own definitions, even when the same global symbols exist
# in another loaded plugin (libxrt_driver_xdna.so).  Without this,
# first-loaded-wins semantics let the real driver's hwctx_kmq replace ours,
# but our config_ctx_cu_config receiver still reads layout fields the
# other plugin's struct never set -- the result is uninitialized memory
# bleeding into start_col.  The flag affects intra-.so binding only; XRT
# can still resolve our exported FFI through the normal global namespace.
target_link_options(xrt_driver_emu PRIVATE -Wl,-Bsymbolic)

target_compile_definitions(xrt_driver_emu PRIVATE
  XRT_ENABLE_AIE
  XRT_AIE_BUILD
  XRT_BUILD
)

target_include_directories(xrt_driver_emu PRIVATE
  # XRT internal headers (source tree) FIRST -- they must shadow the
  # installed XRT headers when the xdna-driver submodule is newer.
  "${XRT_RUNTIME_SRC}"
  "${XRT_RUNTIME_SRC}/core/include"
  # XRT public headers (installed) -- fallback for anything not in source.
  "${XRT_INSTALL_DIR}/include"
  # XDNA shim sources (for local includes within the shim).
  "${XDNA_SHIM_DIR}"
  # XDNA src root (for "shim/foo.h" style includes).
  "${XDNA_DRIVER_DIR}/src"
  # DRM local UAPI headers (for "drm_local/amdxdna_accel.h").
  "${XDNA_UAPI_DIR}"
)

target_link_libraries(xrt_driver_emu PRIVATE
  ${XRT_CORE_LIB}
  ${XRT_COREUTIL_LIB}
  ${CMAKE_DL_LIBS}
)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

install(TARGETS xrt_driver_emu
  LIBRARY DESTINATION "${XRT_INSTALL_DIR}/lib"
  NAMELINK_SKIP
)
