cmake_minimum_required(VERSION 3.16)
project(xdna_aiesim_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# aietools discovery
# ---------------------------------------------------------------------------
# Prefer the env that activate-npu-env.sh exports; fall back to mlir-aie's
# FindAIETools. The bridge needs aietools for the SystemC headers + the sc_main
# bootstrap (added in Task II.2); Task II.1 builds stubs only and does not yet
# include them, but we resolve the path now so the later tasks are a no-op.
if(DEFINED ENV{XILINX_VITIS_AIETOOLS})
  set(AIETOOLS_DIR "$ENV{XILINX_VITIS_AIETOOLS}")
else()
  list(APPEND CMAKE_MODULE_PATH
       "${CMAKE_CURRENT_SOURCE_DIR}/../../mlir-aie/cmake/modulesXilinx")
  find_package(AIETools REQUIRED)
endif()
if(NOT EXISTS "${AIETOOLS_DIR}")
  message(FATAL_ERROR "aietools not found at '${AIETOOLS_DIR}' -- source toolchain-build/activate-npu-env.sh")
endif()
message(STATUS "aietools: ${AIETOOLS_DIR}")

set(SC_INCLUDE "${AIETOOLS_DIR}/data/osci_systemc/include")
set(SC_MAIN    "${AIETOOLS_DIR}/data/osci_systemc/sc_main")

# ---------------------------------------------------------------------------
# Bridge shared library
# ---------------------------------------------------------------------------
# Sources are added as the spine is built:
#   II.2  src/sc_bootstrap.cpp + ${SC_MAIN}/sc_main.cpp + ${SC_MAIN}/sc_main_main.cpp
#   II.3  src/aiesim_top.cpp
#   II.4  src/ps_bridge.cpp
#   II.5  src/cdo_replay.cpp
#   II.6  src/service_thread.cpp
add_library(xdna_aiesim_bridge SHARED
  src/c_abi.cpp
)

target_include_directories(xdna_aiesim_bridge PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${SC_INCLUDE}"
)

# In-process embedding recipe (findings doc 2026-06-01):
#   -z execstack : aietools' libsystemc requires an executable stack
#   -rdynamic    : export our host globals so the dlopened cluster resolves them
# Harmless for the II.1 stub library; required once SystemC embeds in II.2.
target_link_options(xdna_aiesim_bridge PRIVATE -Wl,-z,execstack -rdynamic)
target_link_libraries(xdna_aiesim_bridge PRIVATE ${CMAKE_DL_LIBS})
