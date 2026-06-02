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

#include <exception>
#include <iostream>

#include "aiesim_top.h"

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
extern "C" int sc_main(int /*argc*/, char* /*argv*/[]) {
    try {
        auto* top = new aiesim_top("aiesim_top", g_aiesim_arch, g_aiesim_device_json);
        g_aiesim_top = top;
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
