// C ABI entry points for the aiesim bridge.
//
// Task II.1: stubs that link and export the eleven aiesim_* symbols the Rust
// DlopenBridge binds. Real bodies arrive in later tasks (the service thread in
// II.6 routes each of these onto the SystemC command queue). Until then every
// entry returns an error/NULL so a premature call fails loudly rather than
// silently pretending to work.
#include "xdna_aiesim_bridge.h"

extern "C" {

void *aiesim_create(const char *, const char *) { return nullptr; }
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
