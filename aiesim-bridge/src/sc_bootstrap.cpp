// In-process SystemC bootstrap for the aiesim bridge.
//
// Recipe: docs/superpowers/findings/2026-06-01-aiesim-inprocess-backend-feasibility.md
//
// A .so does not run its own main(), so we do NOT compile aietools'
// sc_main.cpp (which only provides main()). We compile sc_main_main.cpp, which
// provides sc_core::sc_elab_and_sim(argc, argv) -- the entry the bridge calls
// directly to start the SystemC kernel. sc_elab_and_sim prints the SystemC
// banner (pln()) then invokes our sc_main() below.
//
// The service thread (service_thread.{h,cpp}) owns this kernel: sc_main runs on
// that thread, constructs aiesim_top once, publishes it back to aiesim_create,
// then pumps the command queue. EVERY sc_* / TLM call in the dispatch below runs
// on this one thread -- that is the whole point of the marshalling.
#include <systemc.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

#include "aiesim_top.h"
#include "ps_bridge.h"
#include "service_thread.h"

// Host-executable contract: the dlopened cluster resolves these at load
// (RTLD_GLOBAL + -rdynamic export them). Without them, dlopen of the cluster
// lib fails with unresolved symbols. plio_complete is set by the cluster's PLIO
// models when the workload finishes -- the natural-completion signal for RUN.
extern "C" {
bool sc_stop_at_end_of_main = false;
int plio_complete = 0;
}

// Arch/device_json hand-off. Service::start sets these before spawning the
// kernel thread; sc_main reads them to construct the cluster, then publishes the
// aiesim_top handle. Caller-owned, valid for the cluster's lifetime.
extern "C" {
const char* g_aiesim_arch = nullptr;
const char* g_aiesim_device_json = nullptr;
void* g_aiesim_top = nullptr;  // aiesim_top* (kept alive past sc_main's loop)
}

namespace {

// Registered host buffers (DDR addr, size). ADD_HOST_BUF / CLEAR_HOST_BUF
// maintain this; II-B.2's cdo_replay consumes it to resolve DdrPatch records.
// Touched only on the service thread, so no locking is needed.
std::vector<std::pair<uint64_t, size_t>> g_host_buffers;

// Optional bring-up diagnostic: backdoor write+read sweep over candidate AIE2
// array addresses, proving the ps_bridge seam reaches the cluster. Gated on
// AIESIM_BRIDGE_SELFTEST; run once before the service loop. AIE2 array address
// = (col<<25)|(row<<20)|offset.
void bridge_selftest(ps_bridge* ps) {
    if (!ps) { std::cout << "[selftest] no ps_bridge\n"; return; }
    auto tile = [](uint64_t col, uint64_t row, uint64_t off) {
        return (col << 25) | (row << 20) | off;
    };
    const uint64_t addrs[] = {
        tile(1, 2, 0x0),     tile(1, 2, 0x1000),  tile(2, 3, 0x0),
        tile(1, 1, 0x0),     tile(1, 1, 0x40000), 0x02232000ULL,
    };
    int matched = 0;
    for (uint64_t a : addrs) {
        uint32_t want = 0xCAFE0000u | (uint32_t)(a & 0xFFFF);
        ps->write32_backdoor(a, want);
        uint32_t got = ps->read32_backdoor(a);
        bool ok = (got == want);
        matched += ok;
        std::cout << "[selftest] addr=0x" << std::hex << a << " want=0x" << want
                  << " got=0x" << got << (ok ? "  MATCH" : "  --") << std::dec << "\n";
    }
    std::cout << "[selftest] " << matched << "/6 round-tripped\n";
}

// RUN: advance the kernel up to `budget` cycles (1 ns clock => 1 cycle/ns),
// stepping in quanta so plio_complete (natural completion) can short-circuit.
// The first RUN triggers start_of_simulation. Fills reply_int (0=Completed,
// 1=Budget) + reply_cycles.
void do_run(aiesim::Command* c) {
    constexpr uint64_t kQuantum = 1024;  // cycles per sc_start step
    const sc_core::sc_time one_ns(1.0, sc_core::SC_NS);
    const sc_core::sc_time t0 = sc_core::sc_time_stamp();

    uint64_t ran = 0;
    int halt = 1;  // Budget, unless plio_complete fires
    while (ran < c->budget) {
        const uint64_t step = std::min<uint64_t>(kQuantum, c->budget - ran);
        sc_core::sc_start(sc_core::sc_time(static_cast<double>(step), sc_core::SC_NS));
        ran += step;
        if (plio_complete) { halt = 0; break; }  // Completed
    }

    const sc_core::sc_time dt = sc_core::sc_time_stamp() - t0;
    c->reply_cycles = static_cast<uint64_t>(dt / one_ns);
    c->reply_int = halt;
}

// The service loop: pull commands, execute on this (the SystemC) thread, reply.
// Returns when a SHUTDOWN command is serviced, letting sc_main unwind cleanly.
void service_loop(aiesim_top* top) {
    auto& svc = aiesim::Service::instance();
    ps_bridge* ps = top->ps();
    for (;;) {
        aiesim::Command* c = svc.next();  // blocks
        switch (c->tag) {
            case aiesim::Command::READ_REG:
                // Zero sim-time backdoor; serviced without advancing the kernel.
                c->reply_u32 = ps->read32_backdoor(c->addr);
                c->reply_int = 0;
                break;
            case aiesim::Command::RUN:
                do_run(c);
                break;
            case aiesim::Command::ADD_HOST_BUF:
                g_host_buffers.emplace_back(c->addr, c->len);
                c->reply_int = 0;  // staged for II-B.2 DdrPatch resolution
                break;
            case aiesim::Command::CLEAR_HOST_BUF:
                g_host_buffers.clear();
                c->reply_int = 0;
                break;
            case aiesim::Command::SHUTDOWN:
                c->reply_int = 0;
                svc.complete(c);
                return;
            // Data-path commands route through the queue now but their backends
            // land in the next increments: WRITE_GM/READ_GM need the cluster->
            // host DDR target (replacing the ms_aximm stubs); LOAD_CDO/EXEC_NPU
            // need the cdo_replay decoder (II-B.2); RESET re-applies CDO. Until
            // then they fail loudly rather than pretend to work.
            case aiesim::Command::LOAD_CDO:
            case aiesim::Command::EXEC_NPU:
            case aiesim::Command::WRITE_GM:
            case aiesim::Command::READ_GM:
            case aiesim::Command::RESET:
            default:
                c->reply_int = 1;  // not yet implemented
                break;
        }
        svc.complete(c);
    }
}

}  // namespace

// sc_elab_and_sim (from sc_main_main.cpp) calls this with C linkage, on the
// service thread. Construct the cluster, publish it to release aiesim_create,
// then run the service loop until SHUTDOWN.
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
    aiesim::Service::instance().publish(top, true);

    if (std::getenv("AIESIM_BRIDGE_SELFTEST")) {
        bridge_selftest(top->ps());
    }

    service_loop(top);
    return 0;
}

// Start the SystemC kernel once (sc_elab_and_sim -> sc_main). Runs on the
// service thread and does not return until SHUTDOWN unwinds sc_main.
extern "C" int aiesim_bridge_start_systemc() {
    char arg0[] = "aiesim-bridge";
    char* argv[] = {arg0, nullptr};
    return sc_core::sc_elab_and_sim(1, argv);
}
