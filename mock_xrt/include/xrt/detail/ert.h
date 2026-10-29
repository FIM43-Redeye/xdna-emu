// SPDX-License-Identifier: Apache-2.0
// Mock ERT (Embedded Runtime) definitions for xdna-emu
// These match the real XRT definitions for API compatibility

#ifndef MOCK_XRT_DETAIL_ERT_H_
#define MOCK_XRT_DETAIL_ERT_H_

#include <cstdint>

/// Command state enum - matches real XRT
enum ert_cmd_state {
    ERT_CMD_STATE_NEW = 1,
    ERT_CMD_STATE_QUEUED = 2,
    ERT_CMD_STATE_RUNNING = 3,
    ERT_CMD_STATE_COMPLETED = 4,
    ERT_CMD_STATE_ERROR = 5,
    ERT_CMD_STATE_ABORT = 6,
    ERT_CMD_STATE_SUBMITTED = 7,
    ERT_CMD_STATE_TIMEOUT = 8,
    ERT_CMD_STATE_NORESPONSE = 9,
    ERT_CMD_STATE_SKERROR = 10,
    ERT_CMD_STATE_SKCRASHED = 11,
    ERT_CMD_STATE_MAX = 12
};

/// Command opcodes
enum ert_cmd_opcode {
    ERT_START_CU = 0,
    ERT_START_KERNEL = 0,
    ERT_CONFIGURE = 2,
    ERT_EXIT = 3,
    ERT_ABORT = 4,
    ERT_EXEC_WRITE = 5,
    ERT_CU_STAT = 6,
    ERT_START_COPYBO = 7,
    ERT_SK_CONFIG = 8,
    ERT_SK_START = 9,
    ERT_SK_UNCONFIG = 10,
    ERT_INIT_CU = 11,
    ERT_START_FA = 12,
    ERT_CLK_CALIB = 13,
    ERT_MB_VALIDATE = 14,
    ERT_START_KEY_VAL = 15,
    ERT_ACCESS_TEST_C = 16,
    ERT_ACCESS_TEST = 17,
    ERT_CMD_CHAIN = 18
};

#endif // MOCK_XRT_DETAIL_ERT_H_
