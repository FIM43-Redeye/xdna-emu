// SystemC service thread: marshals C-ABI requests onto the one kernel thread.
//
// SystemC's kernel is a process-global singleton and is NOT thread-safe: every
// sc_* / TLM call must happen on the thread that called sc_elab_and_sim. But the
// C ABI (aiesim_*) is invoked from the host's thread (XRT plugin / Rust), which
// is a different thread. This layer bridges that gap:
//
//   * The first aiesim_create spawns ONE OS thread that runs sc_elab_and_sim ->
//     sc_main. sc_main constructs aiesim_top once, publishes the handle, then
//     loops pulling commands and executing them (the only place sc_* runs).
//   * Every C-ABI entry builds a Command, enqueues it, and blocks on a per-
//     command condvar until the service thread fills in the reply.
//
// This file is deliberately free of SystemC and cluster headers: it is pure
// threading + queue mechanics. The command DISPATCH (which touches aiesim_top /
// ps_bridge / sc_start) lives in sc_bootstrap.cpp, on the service thread.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>

namespace aiesim {

// One unit of work handed from a host thread to the service thread. The caller
// owns it on its stack and blocks (in submit) until done, so no heap/lifetime
// dance is needed -- the pointer enqueued stays valid for the whole exchange.
struct Command {
    enum Tag {
        LOAD_CDO,    // in_ptr/len = config op-stream bytes
        EXEC_NPU,    // in_ptr/len = runtime-sequence op-stream bytes
        WRITE_GM,    // addr + in_ptr/len = host->DDR bytes
        READ_GM,     // addr + out_ptr/len = DDR->host bytes
        READ_REG,    // addr -> reply_u32 (zero sim-time backdoor)
        ADD_HOST_BUF,// addr + len (size) = register a host buffer
        CLEAR_HOST_BUF,
        RUN,         // budget cycles -> reply_int (halt) + reply_cycles
        RESET,
        SHUTDOWN,    // stop the service loop; sc_main returns
    };

    explicit Command(Tag t) : tag(t) {}

    Tag tag;

    // Inputs (only the fields relevant to `tag` are read).
    const uint8_t* in_ptr = nullptr;
    uint8_t* out_ptr = nullptr;
    size_t len = 0;
    uint64_t addr = 0;
    uint64_t budget = 0;

    // Outputs (filled by the service thread before signalling done).
    int reply_int = 0;       // status / halt code (mirrors the C-ABI return)
    uint32_t reply_u32 = 0;  // READ_REG value
    uint64_t reply_cycles = 0;

    // Handshake. The service thread sets done under m and notifies cv.
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
};

// Process-singleton owner of the SystemC thread + command queue. Started lazily
// by the first start() call (from aiesim_create).
class Service {
public:
    static Service& instance();

    // Host thread: spawn the SystemC thread (idempotent), block until aiesim_top
    // is elaborated and published, and return its handle (null on failure).
    // arch / device_json are handed to sc_main via the bootstrap globals.
    void* start(const char* arch, const char* device_json);

    // Host thread: enqueue `c` and block until the service thread completes it.
    // If the service thread never came up, marks the command failed and returns.
    void submit(Command& c);

    // Host thread: enqueue SHUTDOWN, then join the service thread. Safe to call
    // once; further submits after this are no-ops. (SystemC cannot be restarted
    // in-process, so there is no symmetric re-start.)
    void shutdown_and_join();

    // --- called ONLY on the service thread (from sc_bootstrap's sc_main) ---

    // Publish the elaboration result: release start()'s waiter. `top` is the
    // aiesim_top* (null + ok=false on construction failure).
    void publish(void* top, bool ok);

    // Block-pop the next command. Returns nullptr only if asked to stop before a
    // command arrives (not currently used; SHUTDOWN flows as a normal Command).
    Command* next();

    // Mark a popped command complete and wake its waiter.
    void complete(Command* c);

private:
    Service() = default;

    std::mutex mtx_;                  // guards queue_ + lifecycle flags
    std::condition_variable q_cv_;    // queue non-empty
    std::condition_variable elab_cv_; // elaboration settled
    std::deque<Command*> queue_;
    std::thread thread_;
    void* top_ = nullptr;
    bool started_ = false;
    bool elaborated_ = false;  // publish() has run (success or failure)
    bool elab_ok_ = false;
    bool joined_ = false;
};

}  // namespace aiesim
