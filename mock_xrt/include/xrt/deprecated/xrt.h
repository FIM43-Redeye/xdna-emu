// SPDX-License-Identifier: Apache-2.0
// Mock XRT deprecated header for xdna-emu
// Contains legacy type definitions for API compatibility

#ifndef MOCK_XRT_DEPRECATED_XRT_H_
#define MOCK_XRT_DEPRECATED_XRT_H_

#ifdef __cplusplus
#include <cstdlib>
#include <cstdint>
#else
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#endif

// DLL export/import macros
#ifdef __linux__
# define XCL_DRIVER_DLLESPEC __attribute__((visibility("default")))
# define XCL_DRIVER_DLLHIDDEN __attribute__((visibility("hidden")))
#elif defined(_WIN32)
# ifdef XCL_DRIVER_DLL_EXPORT
#  define XCL_DRIVER_DLLESPEC __declspec(dllexport)
# else
#  define XCL_DRIVER_DLLESPEC __declspec(dllimport)
# endif
# define XCL_DRIVER_DLLHIDDEN
#else
# define XCL_DRIVER_DLLESPEC
# define XCL_DRIVER_DLLHIDDEN
#endif

// XRT API export macro
#ifndef XRT_API_EXPORT
# define XRT_API_EXPORT XCL_DRIVER_DLLESPEC
#endif

// Deprecated attribute
#ifdef __GNUC__
# define XRT_DEPRECATED __attribute__((deprecated))
#else
# define XRT_DEPRECATED
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque device handle
typedef void* xclDeviceHandle;
#define XRT_NULL_HANDLE NULL

/// Opaque buffer handle
#ifdef _WIN32
typedef void* xclBufferHandle;
# define NULLBO INVALID_HANDLE_VALUE
#else
typedef unsigned int xclBufferHandle;
# define NULLBO 0xffffffff
#endif
#define XRT_NULL_BO NULLBO

/// Exported buffer handle (for sharing between processes)
typedef uint64_t xclBufferExportHandle;
#define XRT_NULL_BO_EXPORT ((xclBufferExportHandle)0xffffffffffffffffULL)

/// xrtXclbinHandle - opaque handle to xclbin
typedef void* xrtXclbinHandle;

#ifdef __cplusplus
}
#endif

#endif // MOCK_XRT_DEPRECATED_XRT_H_
