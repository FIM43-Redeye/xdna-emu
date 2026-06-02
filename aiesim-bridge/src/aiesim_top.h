// Cluster instantiation as an sc_module.
//
// Its constructor performs the E513-free create_math_engine call. See
// docs/superpowers/findings/2026-06-01-aiesim-inprocess-backend-feasibility.md
// (RESOLVED section): the factory internally constructs a child sc_module via
// the default (no-name) ctor, which pops an sc_module_name off SystemC's name
// stack. We are inside a parent sc_module's construction and push one name right
// before the call so the internal module consumes it. (Bare from sc_main: E533;
// inside a parent but no push: E513; with the push: clean.)
#pragma once

#include <systemc.h>

#include <vector>

class ps_bridge;

class aiesim_top : public sc_core::sc_module {
public:
    aiesim_top(sc_core::sc_module_name name, const char* arch, const char* device_json);
    ~aiesim_top();

    // Opaque MathEngine* (closed type). Non-null once construction succeeds.
    void* math_engine() const { return me_; }
    // The PS bridge bound to the cluster's config aximm (II-B.1).
    ps_bridge* ps() const { return ps_; }

private:
    // Bind/stub the cluster ports we do not drive so end_of_elaboration passes
    // (aximm stubs, stream/event clocks, dangling sc_in). me_ must be set.
    void stub_unused_ports();

    void* me_ = nullptr;           // MathEngine*
    void* cluster_lib_ = nullptr;  // dlopen handle for the per-arch cluster .so
    ps_bridge* ps_ = nullptr;      // PS-side ess_*() bridge (child sc_module)
    sc_core::sc_clock clock_;      // drives the cluster's clk (II-B.3)

    // Stub storage (kept alive for the cluster's lifetime). xtlm stub modules
    // are held as sc_object* so this header need not pull in xtlm.
    std::vector<sc_core::sc_object*> stubs_;
    sc_core::sc_signal<uint16_t> dummy_event_;  // shared sink for ss_pl_event
};
