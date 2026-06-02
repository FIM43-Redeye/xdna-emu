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

#include <iostream>

// Host-executable contract: the dlopened cluster resolves these at load
// (RTLD_GLOBAL + -rdynamic export them). Without them, dlopen of the cluster
// lib fails with unresolved symbols.
extern "C" {
bool sc_stop_at_end_of_main = false;
int plio_complete = 0;
}

// sc_elab_and_sim (from sc_main_main.cpp) calls this with C linkage.
extern "C" int sc_main(int /*argc*/, char* /*argv*/[]) {
    std::cout << "[aiesim-bridge] sc_main entered (SystemC embedded in-process)"
              << std::endl;
    // II.6 replaces this with: construct aiesim_top once, then service the
    // command queue until destroy.
    return 0;
}

// Temporary II.2 scaffold: kick the SystemC kernel once so the banner + sc_main
// prove the embed. II.6 replaces this with a service thread that owns the run.
// Returns sc_elab_and_sim's status (0 on the clean banner path).
extern "C" int aiesim_bridge_start_systemc_smoke() {
    char arg0[] = "aiesim-bridge";
    char* argv[] = {arg0, nullptr};
    return sc_core::sc_elab_and_sim(1, argv);
}
