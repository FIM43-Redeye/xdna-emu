/* SPDX-License-Identifier: Apache-2.0 */
/* xdna-emu C FFI Header
 *
 * This header provides C-callable functions for integrating xdna-emu
 * with C/C++ applications (like the mock XRT library).
 */

#ifndef XDNA_EMU_H_
#define XDNA_EMU_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to emulator state */
typedef struct XdnaEmuHandle XdnaEmuHandle;

/* Result codes for FFI operations */
typedef enum {
    XDNA_EMU_SUCCESS = 0,
    XDNA_EMU_INVALID_HANDLE = 1,
    XDNA_EMU_INVALID_PATH = 2,
    XDNA_EMU_PARSE_ERROR = 3,
    XDNA_EMU_EXECUTION_ERROR = 4,
    XDNA_EMU_BUFFER_ERROR = 5,
    XDNA_EMU_NULL_POINTER = 6,
} XdnaEmuResult;

/* Execution status returned by run functions */
typedef struct {
    XdnaEmuResult result;
    uint64_t cycles_executed;
    int halted;  /* bool: 1 if cores halted, 0 if max cycles reached */
} XdnaEmuExecStatus;

/**
 * Create a new emulator instance.
 *
 * @return Non-null handle on success, NULL on failure.
 *         The returned handle must be freed with xdna_emu_destroy().
 */
XdnaEmuHandle* xdna_emu_create(void);

/**
 * Destroy an emulator instance.
 *
 * @param handle Handle returned by xdna_emu_create(), or NULL (no-op).
 */
void xdna_emu_destroy(XdnaEmuHandle* handle);

/**
 * Load an xclbin file into the emulator.
 *
 * This parses the xclbin, extracts the CDO configuration, and applies
 * it to the device state.
 *
 * @param handle Valid emulator handle.
 * @param xclbin_path Path to xclbin file (null-terminated).
 * @param uuid_out Pointer to 16-byte buffer for UUID output, or NULL to skip.
 * @return XDNA_EMU_SUCCESS on success.
 */
XdnaEmuResult xdna_emu_load_xclbin(
    XdnaEmuHandle* handle,
    const char* xclbin_path,
    uint8_t* uuid_out
);

/**
 * Allocate a region in host memory.
 *
 * @param handle Valid emulator handle.
 * @param name Region name (null-terminated), or NULL for auto-generated name.
 * @param address Base address in host memory space.
 * @param size Size of region in bytes.
 * @return XDNA_EMU_SUCCESS on success.
 */
XdnaEmuResult xdna_emu_alloc_host_region(
    XdnaEmuHandle* handle,
    const char* name,
    uint64_t address,
    uint64_t size
);

/**
 * Write data to host memory at a specific address.
 *
 * @param handle Valid emulator handle.
 * @param address Address to write to.
 * @param data Pointer to data to write.
 * @param size Number of bytes to write.
 * @return XDNA_EMU_SUCCESS on success.
 */
XdnaEmuResult xdna_emu_write_host_memory(
    XdnaEmuHandle* handle,
    uint64_t address,
    const uint8_t* data,
    uint64_t size
);

/**
 * Read data from host memory at a specific address.
 *
 * @param handle Valid emulator handle.
 * @param address Address to read from.
 * @param data Pointer to buffer to receive data.
 * @param size Number of bytes to read.
 * @return XDNA_EMU_SUCCESS on success.
 */
XdnaEmuResult xdna_emu_read_host_memory(
    XdnaEmuHandle* handle,
    uint64_t address,
    uint8_t* data,
    uint64_t size
);

/**
 * Clear host buffer list for NPU executor.
 * Call this before adding buffers for a new execution.
 *
 * @param handle Valid emulator handle.
 * @return XDNA_EMU_SUCCESS on success.
 */
XdnaEmuResult xdna_emu_clear_host_buffers(XdnaEmuHandle* handle);

/**
 * Add a host buffer for NPU instruction address patching.
 * Buffers are added in order matching the runtime_sequence arguments.
 * Call this after clear_host_buffers and before execute_npu_instructions.
 *
 * @param handle Valid emulator handle.
 * @param address Buffer address in host memory space.
 * @param size Buffer size in bytes.
 * @return XDNA_EMU_SUCCESS on success.
 */
XdnaEmuResult xdna_emu_add_host_buffer(
    XdnaEmuHandle* handle,
    uint64_t address,
    uint64_t size
);

/**
 * Execute NPU instructions.
 *
 * This executes the instruction buffer which triggers DMA transfers
 * and configures the shim tiles. Call add_host_buffer first to set up
 * the buffer mapping for DdrPatch instructions.
 *
 * @param handle Valid emulator handle.
 * @param instr_data Pointer to instruction data.
 * @param instr_size Size of instruction data in bytes.
 * @return XDNA_EMU_SUCCESS on success.
 */
XdnaEmuResult xdna_emu_execute_npu_instructions(
    XdnaEmuHandle* handle,
    const uint8_t* instr_data,
    uint64_t instr_size
);

/**
 * Load an ELF file for a specific tile.
 *
 * @param handle Valid emulator handle.
 * @param col Tile column.
 * @param row Tile row.
 * @param elf_path Path to ELF file (null-terminated).
 * @return XDNA_EMU_SUCCESS on success.
 */
XdnaEmuResult xdna_emu_load_elf(
    XdnaEmuHandle* handle,
    uint8_t col,
    uint8_t row,
    const char* elf_path
);

/**
 * Sync cores from device state (after CDO/ELF loading).
 *
 * @param handle Valid emulator handle.
 * @return XDNA_EMU_SUCCESS on success.
 */
XdnaEmuResult xdna_emu_sync_cores(XdnaEmuHandle* handle);

/**
 * Set maximum cycles for execution.
 *
 * @param handle Valid emulator handle.
 * @param max_cycles Maximum cycles before timeout.
 * @return XDNA_EMU_SUCCESS on success.
 */
XdnaEmuResult xdna_emu_set_max_cycles(
    XdnaEmuHandle* handle,
    uint64_t max_cycles
);

/**
 * Run the emulator until completion or max cycles.
 *
 * @param handle Valid emulator handle.
 * @return Execution status including cycles run and whether cores halted.
 */
XdnaEmuExecStatus xdna_emu_run(XdnaEmuHandle* handle);

/**
 * Get the last error message (for debugging).
 *
 * @param buffer Buffer to receive error message.
 * @param buffer_size Size of buffer.
 * @return Number of bytes written (excluding null terminator).
 */
uint64_t xdna_emu_get_error(char* buffer, uint64_t buffer_size);

/**
 * Get version information.
 *
 * @return Version as 0x00MMNN00 (major.minor.patch).
 */
uint32_t xdna_emu_version(void);

#ifdef __cplusplus
}
#endif

#endif /* XDNA_EMU_H_ */
