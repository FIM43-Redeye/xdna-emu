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
# xtlm pulls in common_cpp headers (report_handler.h, properties.h) on the bare
# include path, so both dirs are needed.
set(XTLM_INCLUDE
  "${AIETOOLS_DIR}/include/xtlm/include"
  "${AIETOOLS_DIR}/include/common_cpp/common_cpp_v1_0/include")

# MathEngineBase interface header (math_engine_base.h) -- the closed cluster's
# ABI, needed to call its socket accessors when binding ps_bridge. It ships only
# under data/simmodels/<sim>/.../protected/; the OSCI variant matches our
# SystemC 2.3.1 + the .osci cluster lib. Glob so the version path is not pinned.
file(GLOB ME_INCLUDE_CANDIDATES
  "${AIETOOLS_DIR}/../data/simmodels/osci/*/lnx64/*/systemc/protected/aie_cluster_v1_0_0/include")
if(NOT ME_INCLUDE_CANDIDATES)
  message(FATAL_ERROR "MathEngineBase include (osci aie_cluster) not found under data/simmodels")
endif()
list(GET ME_INCLUDE_CANDIDATES 0 ME_INCLUDE)
message(STATUS "MathEngine interface: ${ME_INCLUDE}")

# ---------------------------------------------------------------------------
# Bridge shared library
# ---------------------------------------------------------------------------
# Sources are added as the spine is built:
#   II.4  src/ps_bridge.cpp
#   II.5  src/cdo_replay.cpp
#   II.6  src/service_thread.cpp
#
# Note: we compile sc_main_main.cpp (provides sc_elab_and_sim, the entry the
# bridge calls to start the kernel) but NOT sc_main.cpp -- that only provides
# main(), which a .so never runs.
add_library(xdna_aiesim_bridge SHARED
  src/c_abi.cpp
  src/sc_bootstrap.cpp
  src/aiesim_top.cpp
  src/ps_bridge.cpp
  src/service_thread.cpp
  "${SC_MAIN}/sc_main_main.cpp"
)

target_include_directories(xdna_aiesim_bridge PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${SC_INCLUDE}"
  "${XTLM_INCLUDE}"
  "${ME_INCLUDE}"
)

# In-process embedding recipe (findings doc 2026-06-01):
#   -rdynamic : export our host globals so the dlopened cluster resolves them.
# NOTE: we deliberately do NOT pass -z execstack. aietools' libsystemc ships
# marked executable-stack, but that marker is spurious (QuickThreads .s files
# lacking .note.GNU-stack) -- and marking our .so execstack is exactly what makes
# dlopen fail on this kernel (it can't flip a running process's stack). Instead
# the build script sanitizes a co-located libsystemc copy; see below + findings.
target_link_options(xdna_aiesim_bridge PRIVATE -rdynamic)

# DT_RPATH=$ORIGIN (old dtags, so it is searched BEFORE LD_LIBRARY_PATH): makes
# the loader resolve libsystemc to the marker-cleared copy the build script
# drops next to this .so, ahead of the RWE one in <aietools>/lib/lnx64.o.
set_target_properties(xdna_aiesim_bridge PROPERTIES
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "$ORIGIN"
)
target_link_options(xdna_aiesim_bridge PRIVATE -Wl,--disable-new-dtags)

# SystemC: link aietools' libsystemc (resolved at run time to the sanitized
# copy via the $ORIGIN rpath above). Cluster libs + other deps live in
# lib/lnx64.o, found via LD_LIBRARY_PATH. Use system libstdc++ (default).
find_library(SYSTEMC_LIB systemc PATHS "${AIETOOLS_DIR}/lib/lnx64.o" NO_DEFAULT_PATH)
if(NOT SYSTEMC_LIB)
  message(FATAL_ERROR "libsystemc not found in ${AIETOOLS_DIR}/lib/lnx64.o")
endif()
find_library(XTLM_LIB xtlm PATHS "${AIETOOLS_DIR}/lib/lnx64.o" NO_DEFAULT_PATH)
if(NOT XTLM_LIB)
  message(FATAL_ERROR "libxtlm not found in ${AIETOOLS_DIR}/lib/lnx64.o")
endif()
message(STATUS "systemc: ${SYSTEMC_LIB}")
message(STATUS "xtlm:    ${XTLM_LIB}")

# The service thread (service_thread.cpp) uses std::thread -> needs pthread.
find_package(Threads REQUIRED)

target_link_libraries(xdna_aiesim_bridge PRIVATE
  ${SYSTEMC_LIB} ${XTLM_LIB} ${CMAKE_DL_LIBS} Threads::Threads)
