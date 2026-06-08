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

#include <string>
#include <vector>

class ps_bridge;
class ddr_target;

class aiesim_top : public sc_core::sc_module {
public:
    aiesim_top(sc_core::sc_module_name name, const char* arch, const char* device_json);
    ~aiesim_top();

    // Opaque MathEngine* (closed type). Non-null once construction succeeds.
    void* math_engine() const { return me_; }
    // The PS bridge bound to the cluster's config aximm (II-B.1).
    ps_bridge* ps() const { return ps_; }
    // The host DDR model bound to the cluster's shim-DMA masters (II-B.3); the
    // GM (host-buffer) path reads/writes it directly.
    ddr_target* ddr() const { return ddr_; }

    // Spawn the shim egress drains (one SC_THREAD per ms_pl_stream and
    // ms_noc_axis port). The cluster emits a Task-Completion-Token packet on the
    // shim output stream for every BD that completes with issue_token set (PL-shim
    // columns -> ms_pl_stream, NoC-shim columns -> ms_noc_axis). Left unconsumed
    // that egress FIFO fills and back-pressures the shim DMA (status
    // Stalled_TCT[5]=1) -> deadlock. Must be called from sc_main (a simulation
    // context where sc_spawn is legal), after the cluster is constructed.
    void spawn_egress_drains();

    // Spawn the shim->memtile event-broadcast mirror (one clock-posedge method;
    // no-op unless XDNA_AIESIM_BCAST_BRIDGE is set and lanes resolved). Lets
    // trace-start broadcast-15 cross from the shim's SystemC world into the array's
    // MSM world. Call from sc_main alongside spawn_egress_drains (sim context).
    void spawn_bcast_bridge();

    // Request an NPU1-native VCD timing dump from the MSM cluster model. Yields a
    // waveform in NPU1 5x6 geometry -- same as the trace BO / HW side, so no
    // vc2802 row remap -- driving the three-way timing calibration's aiesim leg
    // (docs/coverage/three-way-timing-calibration.md). Records `path`; the actual
    // trace registration runs in end_of_elaboration() (below), because the cluster
    // binds its internal ports (e.g. shim_reset_n) during elaboration -- calling
    // add_sc_traces from sc_main before the first sc_start hits those ports while
    // still unbound (E112). Call from sc_main before sc_start. `path` gets a ".vcd"
    // suffix if absent.
    void request_vcd_trace(const char* path);
    // Flush the VCD writer's tail buffer to disk. Call after the sim loop ends:
    // the writer streams each cycle during the run but nobody destroys it, so the
    // final filebuf would otherwise be lost. No-op if tracing was not requested.
    void flush_vcd_trace();

protected:
    // SystemC elaboration hook. After the cluster has bound its internal ports we
    // register the VCD traces here (if requested): create the cluster's OWN trace
    // writer via msm_create_vcd_trace_file -- NOT SystemC's
    // sc_create_vcd_trace_file: MathEngineBase::add_sc_traces reinterprets its
    // sc_trace_file* arg as a msm_trace::vcd_trace_file_writer* (verified by
    // backtrace -- a SystemC trace file segfaults in write_comment).
    void end_of_elaboration() override;

private:
    // Bind/stub the cluster ports we do not drive so end_of_elaboration passes
    // (aximm stubs, stream/event clocks, dangling sc_in). me_ must be set.
    void stub_unused_ports();

    void* me_ = nullptr;           // MathEngine*
    void* cluster_lib_ = nullptr;  // dlopen handle for the per-arch cluster .so
    size_t config_idx_ = 1;        // ss_aximm index for memory-mapped config.
                                   // [0]=NPI (aie_xtlm.cpp:371); [1..]=noc2aie
                                   // memory-mapped interfaces. Overridable via
                                   // XDNA_AIESIM_CONFIG_IDX while we pin which
                                   // NoC interface reaches a given tile.
    ps_bridge* ps_ = nullptr;      // PS-side ess_*() bridge (child sc_module)
    ddr_target* ddr_ = nullptr;    // host DDR bound to the shim-DMA masters
    std::string vcd_path_;         // requested VCD path ("" = disabled); consumed
                                   // in end_of_elaboration()
    void* vcd_writer_ = nullptr;   // msm_trace::vcd_trace_file_writer* (opaque);
                                   // created in end_of_elaboration, flushed at sim end
    sc_core::sc_clock clock_;      // drives the cluster's clk (II-B.3)

    // Stub storage (kept alive for the cluster's lifetime). xtlm stub modules
    // are held as sc_object* so this header need not pull in xtlm.
    std::vector<sc_core::sc_object*> stubs_;
    sc_core::sc_signal<uint16_t> dummy_event_;  // shared sink for ss_pl_event
};
