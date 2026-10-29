cmake_minimum_required(VERSION 3.16)
project(aie_reg_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(XRT_INSTALL_DIR "/opt/xilinx/xrt" CACHE PATH "XRT install prefix")

find_library(XRT_COREUTIL_LIB xrt_coreutil
             PATHS "${XRT_INSTALL_DIR}/lib" NO_DEFAULT_PATH)
if(NOT XRT_COREUTIL_LIB)
  message(FATAL_ERROR "libxrt_coreutil.so not found in ${XRT_INSTALL_DIR}/lib")
endif()

add_executable(aie-reg-probe main.cpp)

target_include_directories(aie-reg-probe PRIVATE
  "${XRT_INSTALL_DIR}/include"
)

target_link_libraries(aie-reg-probe PRIVATE
  ${XRT_COREUTIL_LIB}
  ${CMAKE_DL_LIBS}
)
