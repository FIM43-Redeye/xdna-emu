// SPDX-License-Identifier: Apache-2.0
// Mock XRT config header for xdna-emu

#ifndef MOCK_XRT_DETAIL_CONFIG_H_
#define MOCK_XRT_DETAIL_CONFIG_H_

// XRT API export macro - in mock library everything is visible
#define XRT_API_EXPORT

// Process ID type for buffer sharing
#ifdef _WIN32
typedef int pid_type;
#else
#include <sys/types.h>
typedef pid_t pid_type;
#endif

#endif // MOCK_XRT_DETAIL_CONFIG_H_
