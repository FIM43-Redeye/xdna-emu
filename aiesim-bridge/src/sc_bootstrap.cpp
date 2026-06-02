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
// Task II.2 scope: prove SystemC embeds + runs from inside the bridge .so. The
// real service-thread loop (construct aiesim_top once, then pump the command
// queue) replaces this banner body in Task II.6.
#include <systemc.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>

#include "aiesim_top.h"
#include "ps_bridge.h"

// Host-executable contract: the dlopened cluster resolves these at load
// (RTLD_GLOBAL + -rdynamic export them). Without them, dlopen of the cluster
// lib fails with unresolved symbols.
extern "C" {
bool sc_stop_at_end_of_main = false;
int plio_complete = 0;
}

// Interim arch/device_json hand-off (replaced by the service thread's command
// queue in II.6). aiesim_create sets these before starting the kernel; sc_main
// reads them to construct the cluster, then publishes the aiesim_top handle.
// The pointers are caller-owned and valid for the synchronous kernel start.
extern "C" {
const char* g_aiesim_arch = nullptr;
const char* g_aiesim_device_json = nullptr;
void* g_aiesim_top = nullptr;  // aiesim_top* (kept alive past sc_main)
}

// sc_elab_and_sim (from sc_main_main.cpp) calls this with C linkage. Constructs
// the cluster during elaboration (probe4's proven path). II.6 extends this to
// keep aiesim_top alive and service the command queue instead of returning.
// II-B.1 gate (temporary, env-gated): sweep candidate AIE2 array addresses with
// a backdoor write+read round-trip to prove the ess_*() seam reaches the cluster.
// AIE2 array address = (col<<25)|(row<<20)|offset. Removed when II-B.2 lands.
static void bridge_selftest(ps_bridge* ps) {
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

extern "C" int sc_main(int /*argc*/, char* /*argv*/[]) {
    try {
        auto* top = new aiesim_top("aiesim_top", g_aiesim_arch, g_aiesim_device_json);
        g_aiesim_top = top;
        if (top->math_engine() && std::getenv("AIESIM_BRIDGE_SELFTEST")) {
            bridge_selftest(top->ps());
        }
        // II-B.3 scoping scaffold: try to advance the kernel. Surfaces the
        // unbound-port set we still need to stub. Env-gated.
        if (top->math_engine() && std::getenv("AIESIM_BRIDGE_RUN")) {
            std::cout << "[run] sc_start(100 ns)...\n";
            sc_core::sc_start(100, sc_core::SC_NS);
            std::cout << "[run] sc_start returned, t=" << sc_core::sc_time_stamp() << "\n";
        }
        return top->math_engine() ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "[aiesim-bridge] cluster instantiation failed: " << e.what()
                  << std::endl;
        return 1;
    }
}

// Start the SystemC kernel once (sc_elab_and_sim -> sc_main). Returns 0 on the
// clean elaboration path. II.6 replaces this with the service thread that owns
// the single elaboration for the process lifetime.
extern "C" int aiesim_bridge_start_systemc() {
    char arg0[] = "aiesim-bridge";
    char* argv[] = {arg0, nullptr};
    return sc_core::sc_elab_and_sim(1, argv);
}
