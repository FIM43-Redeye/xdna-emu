// C ABI entry points for the aiesim bridge.
//
// Task II.1: stubs that link and export the eleven aiesim_* symbols the Rust
// DlopenBridge binds. Real bodies arrive in later tasks (the service thread in
// II.6 routes each of these onto the SystemC command queue). Until then every
// entry returns an error/NULL so a premature call fails loudly rather than
// silently pretending to work.
#include "xdna_aiesim_bridge.h"

// Temporary II.2 scaffold (sc_bootstrap.cpp): start the SystemC kernel once to
// prove the in-process embed. II.6 replaces this with the service thread.
extern "C" int aiesim_bridge_start_systemc_smoke();

// Non-null sentinel so the II.2 smoke's aiesim_create returns "success" without
// a real handle yet. II.3+ return an actual cluster handle.
static char g_smoke_handle;

extern "C" {

void *aiesim_create(const char *, const char *) {
    // II.2: run the SystemC banner + sc_main from inside the .so.
    if (aiesim_bridge_start_systemc_smoke() != 0) return nullptr;
    return &g_smoke_handle;
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
