// In-process SystemC bootstrap for the aiesim bridge.
//
// Recipe: docs/superpowers/findings/2026-06-01-aiesim-inprocess-backend-feasibility.md
//
// A .so does not run its own main(), so we compile sc_main_main.cpp (which
// provides sc_elab_and_sim, the entry the bridge calls) but NOT sc_main.cpp.
// sc_elab_and_sim prints the SystemC banner then invokes our sc_main() below.
//
// Threading model (findings doc, 2026-06-02): register access needs TIMED
// b_transport (the backdoor reaches only a shadow store), and b_transport
// wait()s -> illegal in sc_main. So ONE in-kernel SC_THREAD (driver_proc) owns
// all register access + time advance; sc_main just hosts the kernel in
// pause/resume windows. The kernel is paused between commands (the driver
// sc_pause()s after draining; sc_main only sc_start()s when work is pending) ->
// no time drift -> cycle-precise stepping is first-class, and free-run is just a
// large RUN budget. The OS thread (C-ABI) wakes the kernel via a Doorbell
// (async_request_update), the one thread-safe OS->kernel hook.
#include <systemc.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

#include "aiesim_top.h"
#include "cdo_replay.h"
#include "ddr_target.h"
#include "ps_bridge.h"
#include "service_thread.h"

// Host-executable contract: the dlopened cluster resolves these at load. plio_
// complete is set by the cluster's PLIO models when the workload finishes -- the
// natural-completion signal for RUN.
extern "C" {
bool sc_stop_at_end_of_main = false;
int plio_complete = 0;
}

// Arch/device_json hand-off. Service::start sets these before spawning the
// kernel thread; sc_main reads them to construct the cluster.
extern "C" {
const char* g_aiesim_arch = nullptr;
const char* g_aiesim_device_json = nullptr;
void* g_aiesim_top = nullptr;  // aiesim_top*
}

namespace {

// Registered host buffers (DDR addr, size). ADD/CLEAR_HOST_BUF maintain this;
// II-B.2b's exec_npu (DdrPatch resolution) consumes it. Driver-thread only.
std::vector<std::pair<uint64_t, std::size_t>> g_host_buffers;

// Doorbell: the OS thread rings async_request_update() (thread-safe); the kernel
// runs update() in its next update phase, notifying the driver's command event.
struct Doorbell : sc_core::sc_prim_channel {
    sc_core::sc_event cmd_event;
    void update() override { cmd_event.notify(sc_core::SC_ZERO_TIME); }
};
Doorbell* g_doorbell = nullptr;

// RUN: advance up to `budget` cycles (1 ns clock => 1 cycle/ns) via wait() (we
// are in a process), stepping in quanta so plio_complete can short-circuit.
// Fills reply_int (0=Completed, 1=Budget) + reply_cycles.
void do_run(aiesim::Command* c) {
    constexpr uint64_t kQuantum = 1024;
    const sc_core::sc_time one_ns(1.0, sc_core::SC_NS);
    const sc_core::sc_time t0 = sc_core::sc_time_stamp();

    uint64_t ran = 0;
    int halt = 1;  // Budget unless plio_complete fires
    while (ran < c->budget) {
        const uint64_t step = std::min<uint64_t>(kQuantum, c->budget - ran);
        sc_core::wait(sc_core::sc_time(static_cast<double>(step), sc_core::SC_NS));
        ran += step;
        if (plio_complete) { halt = 0; break; }
    }
    c->reply_cycles = static_cast<uint64_t>((sc_core::sc_time_stamp() - t0) / one_ns);
    c->reply_int = halt;
}

// Execute one command on the driver SC_THREAD (timed b_transport legal here).
void execute(aiesim_top* top, aiesim::Command* c) {
    ps_bridge* ps = top->ps();
    switch (c->tag) {
        case aiesim::Command::READ_REG:
            c->reply_u32 = ps->read32(c->addr);  // TIMED live read
            c->reply_int = 0;
            break;
        case aiesim::Command::RUN:
            do_run(c);
            break;
        case aiesim::Command::WRITE_GM:
            top->ddr()->host_write(c->addr, c->in_ptr, c->len);
            c->reply_int = 0;
            break;
        case aiesim::Command::READ_GM:
            top->ddr()->host_read(c->addr, c->out_ptr, c->len);
            c->reply_int = 0;
            break;
        case aiesim::Command::ADD_HOST_BUF:
            g_host_buffers.emplace_back(c->addr, c->len);
            c->reply_int = 0;
            break;
        case aiesim::Command::CLEAR_HOST_BUF:
            g_host_buffers.clear();
            c->reply_int = 0;
            break;
        case aiesim::Command::LOAD_CDO:
            c->reply_int = aiesim::cdo_replay(ps, c->in_ptr, c->len);
            break;
        // EXEC_NPU (DdrPatch/Sync) lands in II-B.2b; RESET re-applies CDO.
        case aiesim::Command::EXEC_NPU:
        case aiesim::Command::RESET:
        default:
            c->reply_int = 1;  // not yet implemented
            break;
    }
}

// The driver: the single in-kernel process that touches the cluster. Drains the
// queue, then pauses (releasing sc_main's sc_start) and waits for the next ring.
void driver_proc(aiesim_top* top) {
    auto& svc = aiesim::Service::instance();
    sc_core::wait(sc_core::SC_ZERO_TIME);  // settle start_of_simulation
    svc.publish(top, true);                // release create()

    for (;;) {
        // Drain everything queued (the queue is the source of truth; a ring may
        // coalesce, but nothing enqueued is missed).
        for (aiesim::Command* c = svc.try_pop(); c; c = svc.try_pop()) {
            if (c->tag == aiesim::Command::SHUTDOWN) {
                c->reply_int = 0;
                svc.mark_shutdown();
                svc.complete(c);
                sc_core::sc_stop();  // unwind sc_main's sc_start loop
                return;
            }
            execute(top, c);
            svc.complete(c);
        }
        sc_core::sc_pause();              // let sc_main's sc_start() return (kernel idle)
        sc_core::wait(g_doorbell->cmd_event);  // suspend until the next ring
    }
}

}  // namespace

// sc_elab_and_sim (sc_main_main.cpp) calls this with C linkage on the kernel
// thread. Construct the cluster, spawn the driver, then host the kernel in
// pause/resume windows until SHUTDOWN.
extern "C" int sc_main(int /*argc*/, char* /*argv*/[]) {
    aiesim_top* top = nullptr;
    try {
        top = new aiesim_top("aiesim_top", g_aiesim_arch, g_aiesim_device_json);
    } catch (const std::exception& e) {
        std::cerr << "[aiesim-bridge] cluster instantiation failed: " << e.what()
                  << std::endl;
        aiesim::Service::instance().publish(nullptr, false);
        return 1;
    }
    g_aiesim_top = top;

    g_doorbell = new Doorbell();
    auto& svc = aiesim::Service::instance();
    svc.set_wake([] {
        if (g_doorbell) g_doorbell->async_request_update();
    });
    sc_core::sc_spawn(sc_bind(&driver_proc, top), "aiesim_driver");

    // Host loop: advance the kernel only while a command is pending; the driver
    // sc_pause()s after each drain, returning control here.
    for (;;) {
        sc_core::sc_start();           // runs until the driver sc_pause()s / sc_stop
        if (svc.is_shutdown()) break;
        svc.wait_for_pending();        // block (kernel paused) until a command arrives
    }
    return 0;
}

// Start the SystemC kernel once (sc_elab_and_sim -> sc_main). Runs on the kernel
// thread; returns only after SHUTDOWN unwinds sc_main.
extern "C" int aiesim_bridge_start_systemc() {
    char arg0[] = "aiesim-bridge";
    char* argv[] = {arg0, nullptr};
    return sc_core::sc_elab_and_sim(1, argv);
}
