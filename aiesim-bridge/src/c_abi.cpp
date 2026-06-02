// C ABI entry points for the aiesim bridge.
//
// These are the eleven aiesim_* symbols the Rust DlopenBridge binds. Each call
// arrives on the host (XRT/Rust) thread; SystemC must be driven from its own
// thread, so every entry (except create/destroy lifecycle) marshals a Command
// onto the service thread's queue and blocks for the reply. The `void* handle`
// argument is the aiesim_top* from create; aiesim is a process singleton, so the
// Service drives the one cluster and the handle is not re-dereferenced here.
//
// Return contract (mirrors crates/xdna-emu-ffi/src/aiesim/abi.rs):
//   status entries: 0 = Ok, 1 = Error
//   aiesim_run:     0 = Completed, 1 = Budget, 2 = Error
#include "xdna_aiesim_bridge.h"

#include "service_thread.h"

namespace {
// Submit a command and return its status reply (the C-ABI int).
int submit_status(aiesim::Command& c) {
    aiesim::Service::instance().submit(c);
    return c.reply_int;
}
}  // namespace

extern "C" {

void* aiesim_create(const char* arch, const char* device_json) {
    // First call spawns the service thread, constructs the E513-free cluster
    // during elaboration, and returns the aiesim_top handle. Idempotent.
    return aiesim::Service::instance().start(arch, device_json);
}

int aiesim_load_cdo(void* /*handle*/, const uint8_t* ops, size_t len) {
    aiesim::Command c(aiesim::Command::LOAD_CDO);
    c.in_ptr = ops;
    c.len = len;
    return submit_status(c);
}

int aiesim_exec_npu(void* /*handle*/, const uint8_t* ops, size_t len) {
    aiesim::Command c(aiesim::Command::EXEC_NPU);
    c.in_ptr = ops;
    c.len = len;
    return submit_status(c);
}

int aiesim_add_host_buffer(void* /*handle*/, uint64_t addr, size_t size) {
    aiesim::Command c(aiesim::Command::ADD_HOST_BUF);
    c.addr = addr;
    c.len = size;
    return submit_status(c);
}

int aiesim_clear_host_buffers(void* /*handle*/) {
    aiesim::Command c(aiesim::Command::CLEAR_HOST_BUF);
    return submit_status(c);
}

int aiesim_write_gm(void* /*handle*/, uint64_t addr, const uint8_t* data, size_t len) {
    aiesim::Command c(aiesim::Command::WRITE_GM);
    c.addr = addr;
    c.in_ptr = data;
    c.len = len;
    return submit_status(c);
}

int aiesim_read_gm(void* /*handle*/, uint64_t addr, uint8_t* out, size_t len) {
    aiesim::Command c(aiesim::Command::READ_GM);
    c.addr = addr;
    c.out_ptr = out;
    c.len = len;
    return submit_status(c);
}

int aiesim_run(void* /*handle*/, uint64_t budget, uint64_t* cycles_out) {
    aiesim::Command c(aiesim::Command::RUN);
    c.budget = budget;
    aiesim::Service::instance().submit(c);
    if (cycles_out) *cycles_out = c.reply_cycles;
    return c.reply_int;  // 0=Completed, 1=Budget, 2=Error
}

uint32_t aiesim_read_reg(void* /*handle*/, uint64_t addr) {
    aiesim::Command c(aiesim::Command::READ_REG);
    c.addr = addr;
    aiesim::Service::instance().submit(c);
    return c.reply_u32;
}

int aiesim_reset(void* /*handle*/) {
    aiesim::Command c(aiesim::Command::RESET);
    return submit_status(c);
}

void aiesim_destroy(void* /*handle*/) {
    // Stop the service loop and join the kernel thread. SystemC is not
    // restartable in-process, so this is a one-way teardown.
    aiesim::Service::instance().shutdown_and_join();
}

}  // extern "C"
