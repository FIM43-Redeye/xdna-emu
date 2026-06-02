// SystemC service: marshals C-ABI requests onto the kernel and drives the
// cluster through ONE in-kernel SC_THREAD.
//
// Register access on this cluster requires TIMED b_transport (the backdoor
// transport_dbg only reaches a shadow store -- see the feasibility findings doc,
// 2026-06-02). b_transport wait()s on the AXI handshake, so it is illegal in
// sc_main and must run inside a SystemC process. The model:
//
//   * OS thread (C-ABI): enqueue a Command, ring the kernel (a Doorbell's
//     async_request_update -- the one thread-safe OS->kernel hook), block for
//     the reply. This file owns that side; it stays free of SystemC headers (the
//     wake is an injected std::function so sc_bootstrap supplies the Doorbell).
//   * Driver SC_THREAD (sc_bootstrap): drains the queue, executing each command
//     with timed b_transport / wait()-based time advance, then sc_pause()s. The
//     kernel is paused between commands -> no time drift -> cycle-precise stepping.
//   * sc_main: for(;;){ sc_start(); if(shutdown) break; wait_for_pending(); } --
//     only advances the kernel while a command is pending.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace aiesim {

// One unit of work. The caller owns it on its stack and blocks in submit() until
// done, so the enqueued pointer stays valid for the whole exchange.
struct Command {
    enum Tag {
        LOAD_CDO, EXEC_NPU, WRITE_GM, READ_GM, READ_REG,
        ADD_HOST_BUF, CLEAR_HOST_BUF, RUN, RESET, SHUTDOWN,
    };
    explicit Command(Tag t) : tag(t) {}
    Tag tag;

    const uint8_t* in_ptr = nullptr;
    uint8_t* out_ptr = nullptr;
    size_t len = 0;
    uint64_t addr = 0;
    uint64_t budget = 0;  // RUN: cycles (0 handled by the caller)

    int reply_int = 0;       // status / halt (mirrors the C-ABI return)
    uint32_t reply_u32 = 0;  // READ_REG value
    uint64_t reply_cycles = 0;

    std::mutex m;
    std::condition_variable cv;
    bool done = false;
};

// Process-singleton owner of the kernel thread + command queue.
class Service {
public:
    static Service& instance();

    // OS thread: install the kernel-wake hook (rings the Doorbell). Call before
    // start(). The hook is invoked from submit() on the OS thread.
    void set_wake(std::function<void()> wake);

    // OS thread: spawn the kernel thread (idempotent), block until the driver
    // publishes the elaborated handle, return it (null on failure).
    void* start(const char* arch, const char* device_json);

    // OS thread: enqueue, ring the kernel, block until the driver completes it.
    void submit(Command& c);

    // OS thread: enqueue SHUTDOWN, then join the kernel thread. One-way (SystemC
    // does not restart in-process).
    void shutdown_and_join();

    // --- driver SC_THREAD side ---
    void publish(void* top, bool ok);  // release start()'s waiter
    Command* try_pop();                // non-blocking; null when the queue is empty
    void complete(Command* c);         // signal the OS-thread waiter
    void mark_shutdown();              // driver saw SHUTDOWN (before sc_stop)

    // --- sc_main side ---
    void wait_for_pending();           // block until the queue is non-empty
    bool is_shutdown();

private:
    Service() = default;

    std::mutex mtx_;
    std::condition_variable q_cv_;     // queue non-empty (sc_main gate)
    std::condition_variable elab_cv_;  // elaboration settled
    std::deque<Command*> queue_;
    std::thread thread_;
    std::function<void()> wake_;
    void* top_ = nullptr;
    bool started_ = false;
    bool elaborated_ = false;
    bool elab_ok_ = false;
    bool joined_ = false;
    bool shutdown_ = false;
};

}  // namespace aiesim
