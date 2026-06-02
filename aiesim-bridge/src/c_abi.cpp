// C ABI entry points for the aiesim bridge.
//
// Task II.1: stubs that link and export the eleven aiesim_* symbols the Rust
// DlopenBridge binds. Real bodies arrive in later tasks (the service thread in
// II.6 routes each of these onto the SystemC command queue). Until then every
// entry returns an error/NULL so a premature call fails loudly rather than
// silently pretending to work.
#include "xdna_aiesim_bridge.h"

// SystemC-side hand-off (sc_bootstrap.cpp): set arch/device_json, start the
// kernel (which constructs the cluster in sc_main), read back the handle.
// II.6 replaces this synchronous start with the persistent service thread.
extern "C" {
extern const char* g_aiesim_arch;
extern const char* g_aiesim_device_json;
extern void* g_aiesim_top;
int aiesim_bridge_start_systemc();
}

extern "C" {

void *aiesim_create(const char *arch, const char *device_json) {
    // II.3: hand arch/device_json to sc_main and run elaboration, which
    // constructs the E513-free cluster. Returns the aiesim_top handle.
    g_aiesim_arch = arch;
    g_aiesim_device_json = device_json;
    if (aiesim_bridge_start_systemc() != 0) return nullptr;
    return g_aiesim_top;
}
int aiesim_load_cdo(void *, const uint8_t *, size_t) { return 1; }
int aiesim_exec_npu(void *, const uint8_t *, size_t) { return 1; }
int aiesim_add_host_buffer(void *, uint64_t, size_t) { return 1; }
int aiesim_clear_host_buffers(void *) { return 1; }
int aiesim_write_gm(void *, uint64_t, const uint8_t *, size_t) { return 1; }
int aiesim_read_gm(void *, uint64_t, uint8_t *, size_t) { return 1; }
int aiesim_run(void *, uint64_t, uint64_t *) { return 2; }
uint32_t aiesim_read_reg(void *, uint64_t) { return 0; }
int aiesim_reset(void *) { return 1; }
void aiesim_destroy(void *) {}

}  // extern "C"
