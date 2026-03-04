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

# xdna-driver submodule relative to this CMakeLists.txt.
set(XDNA_DRIVER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../xdna-driver" CACHE PATH
    "Path to the xdna-driver submodule")

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

add_library(xrt_driver_emu SHARED
  src/transport_inprocess.cpp
)

set_target_properties(xrt_driver_emu PROPERTIES
  VERSION   2.21.0
  SOVERSION 2
)

target_compile_definitions(xrt_driver_emu PRIVATE
  XRT_ENABLE_AIE
  XRT_AIE_BUILD
  XRT_BUILD
)

target_include_directories(xrt_driver_emu PRIVATE
  # XRT public headers (installed).
  "${XRT_INSTALL_DIR}/include"
  # XRT internal headers (source tree inside xdna-driver submodule).
  "${XRT_RUNTIME_SRC}"
  "${XRT_RUNTIME_SRC}/core/include"
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
