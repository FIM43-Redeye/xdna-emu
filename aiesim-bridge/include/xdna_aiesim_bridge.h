/* C ABI for libxdna_aiesim_bridge.so.
 *
 * The consumer is the Rust DlopenBridge in
 * crates/xdna-emu-ffi/src/aiesim/bridge.rs -- these eleven signatures must match
 * its fn-pointer types exactly, or symbol binding fails at dlopen time. Keep the
 * two in lockstep.
 */
#ifndef XDNA_AIESIM_BRIDGE_H
#define XDNA_AIESIM_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Construct the cluster for `arch` ("aie2"/"aie2ps"/"aie") using the device
 * model JSON at `device_json`. Spawns the SystemC service thread and elaborates
 * once. Returns an opaque handle, or NULL on failure. */
void *aiesim_create(const char *arch, const char *device_json);

/* Replay a CDO config op-stream (tagged LE wire format; see cdo_replay.cpp).
 * 0 = ok, nonzero = error. */
int aiesim_load_cdo(void *h, const uint8_t *ops, size_t len);

/* Replay a runtime-sequence (NPU instruction) op-stream -- same wire format,
 * staged before run (arrives via execute_npu_instructions). 0 = ok. */
int aiesim_exec_npu(void *h, const uint8_t *ops, size_t len);

/* Register a host buffer (DDR addr + size) so the bridge can resolve DdrPatch
 * records during exec_npu replay. 0 = ok. */
int aiesim_add_host_buffer(void *h, uint64_t addr, size_t size);

/* Clear the registered host buffers (before a new submission). 0 = ok. */
int aiesim_clear_host_buffers(void *h);

/* Host (GM/DDR) memory write/read. 0 = ok. */
int aiesim_write_gm(void *h, uint64_t addr, const uint8_t *data, size_t len);
int aiesim_read_gm(void *h, uint64_t addr, uint8_t *out, size_t len);

/* Run to quiescence or `budget` cycles. Sets *cycles_out. Returns
 * 0 = completed, 1 = budget, 2 = error. */
int aiesim_run(void *h, uint64_t budget, uint64_t *cycles_out);

/* Tier-2 zero-time backdoor register read. */
uint32_t aiesim_read_reg(void *h, uint64_t addr);

/* Reset logical state between submissions (CDO re-applied after). 0 = ok. */
int aiesim_reset(void *h);

/* Park the service thread + free logical state. */
void aiesim_destroy(void *h);

#ifdef __cplusplus
}
#endif

#endif /* XDNA_AIESIM_BRIDGE_H */
