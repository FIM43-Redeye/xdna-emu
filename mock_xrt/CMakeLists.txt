# SPDX-License-Identifier: Apache-2.0
# CMake build for mock XRT library
# This builds a drop-in replacement for XRT that uses xdna-emu

cmake_minimum_required(VERSION 3.16)
project(mock_xrt VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find the xdna-emu library (our Rust emulator FFI)
# Default to looking in the parent directory's target/release
set(XDNA_EMU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "Path to xdna-emu root")
set(XDNA_EMU_LIB_DIR "${XDNA_EMU_ROOT}/target/release" CACHE PATH "Path to xdna-emu library")
set(XDNA_EMU_INCLUDE_DIR "${XDNA_EMU_ROOT}/include" CACHE PATH "Path to xdna-emu headers")

# Find the library
find_library(XDNA_EMU_LIBRARY
    NAMES xdna_emu
    PATHS ${XDNA_EMU_LIB_DIR}
    NO_DEFAULT_PATH
)

if(NOT XDNA_EMU_LIBRARY)
    message(WARNING "xdna-emu library not found at ${XDNA_EMU_LIB_DIR}")
    message(WARNING "Build xdna-emu first with: cd ${XDNA_EMU_ROOT} && cargo build --release")
    set(XDNA_EMU_LIBRARY "")
endif()

# Source files
set(MOCK_XRT_SOURCES
    src/device.cpp
    src/bo.cpp
    src/xclbin.cpp
    src/hw_context.cpp
    src/kernel.cpp
    src/experimental.cpp
    src/emulator_bridge.cpp
)

# Create shared library
add_library(xrt_mock SHARED ${MOCK_XRT_SOURCES})

target_include_directories(xrt_mock
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${XDNA_EMU_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against xdna-emu if available
if(XDNA_EMU_LIBRARY)
    target_link_libraries(xrt_mock PRIVATE ${XDNA_EMU_LIBRARY})
    target_compile_definitions(xrt_mock PRIVATE HAVE_XDNA_EMU_FFI)
    message(STATUS "Linking against xdna-emu: ${XDNA_EMU_LIBRARY}")
else()
    message(STATUS "Building without xdna-emu FFI (stub mode)")
endif()

# Export symbols
target_compile_definitions(xrt_mock PRIVATE XCL_DRIVER_DLL_EXPORT)

# Create static library as well (for embedding)
add_library(xrt_mock_static STATIC ${MOCK_XRT_SOURCES})

target_include_directories(xrt_mock_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${XDNA_EMU_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(XDNA_EMU_LIBRARY)
    target_link_libraries(xrt_mock_static PRIVATE ${XDNA_EMU_LIBRARY})
    target_compile_definitions(xrt_mock_static PRIVATE HAVE_XDNA_EMU_FFI)
endif()

# Installation
install(TARGETS xrt_mock xrt_mock_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/xrt
    DESTINATION include
)

# Test executable builder
# This function creates a test executable from an mlir-aie test.cpp
function(add_mlir_aie_test TEST_NAME TEST_DIR)
    if(EXISTS "${TEST_DIR}/test.cpp")
        add_executable(${TEST_NAME} "${TEST_DIR}/test.cpp")
        target_link_libraries(${TEST_NAME} PRIVATE xrt_mock)
        target_include_directories(${TEST_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        # Set rpath to find libxdna_emu.so at runtime
        set_target_properties(${TEST_NAME} PROPERTIES
            BUILD_RPATH "${XDNA_EMU_LIB_DIR}"
            INSTALL_RPATH "${XDNA_EMU_LIB_DIR}"
        )
    endif()
endfunction()

# Example: Build tests from mlir-aie if available
option(BUILD_MLIR_AIE_TESTS "Build tests from mlir-aie" OFF)
if(BUILD_MLIR_AIE_TESTS)
    set(MLIR_AIE_TEST_DIR "" CACHE PATH "Path to mlir-aie test directory")
    if(MLIR_AIE_TEST_DIR)
        # Add specific tests here
        # add_mlir_aie_test(add_one_using_dma "${MLIR_AIE_TEST_DIR}/add_one_using_dma")
    endif()
endif()
