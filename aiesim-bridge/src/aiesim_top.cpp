#include "aiesim_top.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <utils/xtlm_aximm_initiator_stub.h>

#include "math_engine_base.h"  // the closed cluster ABI (aietools, build-time ref)
#include "cluster_clone_patch.h"
#include "ddr_target.h"
#include "ps_bridge.h"

namespace {

typedef void* (*create_math_engine_fn)(const char*, const char*, bool, bool);

// arch -> cluster library name. Resolved via LD_LIBRARY_PATH (=<aietools>/
// lib/lnx64.o, where the cluster + its deps live). Naming differs per arch --
// note aie2ps carries the variant tag mid-name, unlike aie2/aie.
const char* arch_to_cluster_lib(const std::string& arch) {
    if (arch == "aie2") return "libaie2_cluster_msm_v1_0_0.osci.so";
    if (arch == "aie2ps") return "libaie2ps_cluster_msm_v1_0_0.osci.so";
    if (arch == "aie") return "libaie_cluster_msm_v1_0_0.osci.so";
    return nullptr;
}

}  // namespace

aiesim_top::aiesim_top(sc_core::sc_module_name name, const char* arch, const char* device_json)
    : sc_core::sc_module(name),
      clock_("aie_clk", sc_core::sc_time(1.0, sc_core::SC_NS)) {
    const std::string arch_s = arch ? arch : "";
    // XDNA_AIESIM_CLUSTER_LIB overrides the default per-arch cluster .so (e.g. to
    // select the functional _func variant vs the timed msm model). Resolved via
    // LD_LIBRARY_PATH like the defaults.
    const char* lib = std::getenv("XDNA_AIESIM_CLUSTER_LIB");
    if (!lib || !*lib) lib = arch_to_cluster_lib(arch_s);
    if (!lib) {
        throw std::runtime_error("aiesim_top: unknown arch '" + arch_s + "'");
    }

    // RTLD_GLOBAL so the cluster resolves our exported host globals
    // (sc_stop_at_end_of_main / plio_complete). RTLD_LAZY matches the proven
    // probe4; the cluster references no unprovided symbols at load.
    cluster_lib_ = dlopen(lib, RTLD_LAZY | RTLD_GLOBAL);
    if (!cluster_lib_) {
        throw std::runtime_error(std::string("aiesim_top: dlopen ") + lib + ": " + dlerror());
    }

    // Work around the cluster model's send_response object-reuse bug so control-
    // packet READ-RESPONSES route (see cluster_clone_patch.h). Must run before the
    // first sc_start (send_response only fires during simulation). Fail-safe + arch-
    // gated + default-on; announces on stderr. LOCAL-ONLY runtime patch.
    aiesim::install_clone_patch(cluster_lib_, arch_s.c_str());

    dlerror();  // clear
    auto factory = reinterpret_cast<create_math_engine_fn>(
        dlsym(cluster_lib_, "create_math_engine"));
    if (const char* e = dlerror()) {
        throw std::runtime_error(std::string("aiesim_top: dlsym create_math_engine: ") + e);
    }

    {
        // E513 fix: push one sc_module_name for the factory's internal
        // default-ctor module to consume. Scoped so it is gone after the call.
        sc_core::sc_module_name internal_name("math_engine");
        // is_fast_pm / is_fast_dm = false -> timed (cycle-approximate) model,
        // the oracle we want; matches aie_xtlm's default path.
        me_ = factory("math_engine", device_json, /*is_fast_pm=*/false, /*is_fast_dm=*/false);
    }
    if (!me_) {
        throw std::runtime_error("aiesim_top: create_math_engine returned null");
    }

    // II-B.1: bind the PS bridge's config initiators to the cluster's NPI/config
    // ss_aximm target sockets ([0], per aie_xtlm). ps_bridge is a child module
    // (explicit name -> no name-stack push needed). The cluster->host DDR path
    // (shim_dma_*_socket) is bound in the functional-run lifecycle (II-B.3).
    auto* me = static_cast<MathEngineBase*>(me_);
    ps_ = new ps_bridge("ps_bridge");
    std::vector<xtlm::xtlm_aximm_target_socket*>& ss_rd = me->get_ss_aximm_rd();
    std::vector<xtlm::xtlm_aximm_target_socket*>& ss_wr = me->get_ss_aximm_wr();
    if (ss_rd.empty() || ss_wr.empty()) {
        throw std::runtime_error("aiesim_top: cluster exposes no ss_aximm config sockets");
    }
    // The HAL's memory-mapped register/DM writes route to a noc2aie interface
    // ss_aximm[1..N] (aie_xtlm.cpp:359-383); [0] is the NPI socket. Bind the PS
    // bridge to the memory-mapped config interface, NOT [0].
    if (const char* idx = std::getenv("XDNA_AIESIM_CONFIG_IDX")) {
        config_idx_ = static_cast<size_t>(std::strtoul(idx, nullptr, 10));
    }
    if (config_idx_ >= ss_rd.size()) {
        throw std::runtime_error("aiesim_top: XDNA_AIESIM_CONFIG_IDX out of range");
    }
    ps_->ps_axi_rd.bind(*ss_rd[config_idx_]);
    ps_->ps_axi_wr.bind(*ss_wr[config_idx_]);

    // Drive the cluster clock (aie_xtlm: me_inst->get_clk()(clk)). Required for
    // sc_start to advance the cluster.
    me->get_clk()(clock_);

    if (std::getenv("AIESIM_BRIDGE_PORTS")) {
        fprintf(stderr,
                "[ports] cols=%zu noc_tiles=%zu aximm_ifs=%zu uc_tiles=%zu "
                "ss_aximm_rd=%zu ss_aximm_wr=%zu ms_aximm_rd=%zu ms_aximm_wr=%zu\n"
                "[ports] ms_pl_stream=%zu ss_pl_stream=%zu ms_pl_event=%zu "
                "ss_pl_event=%zu ss_noc_axis=%zu ms_noc_axis=%zu "
                "ms_pl_stream_clk=%zu ss_pl_stream_clk=%zu "
                "ms_pl_event_clk=%zu ss_pl_event_clk=%zu\n",
                me->get_num_cols(), me->get_num_noc_tiles(),
                me->get_num_aximm_interfaces(), me->get_num_tiles_with_uc(),
                me->get_ss_aximm_rd().size(), me->get_ss_aximm_wr().size(),
                me->ms_aximm_rd.size(), me->ms_aximm_wr.size(),
                me->ms_pl_stream.size(), me->ss_pl_stream.size(),
                me->ms_pl_event.size(), me->ss_pl_event.size(),
                me->ss_noc_axis.size(), me->ms_noc_axis.size(),
                me->ms_pl_stream_clk.size(), me->ss_pl_stream_clk.size(),
                me->ms_pl_event_clk.size(), me->ss_pl_event_clk.size());
    }

    // Bind/stub every other cluster port so end_of_elaboration passes.
    stub_unused_ports();
}

// Modeled on aie_xtlm::stub_unused_ports + its clock-binding loops. We drive
// none of the PL/NoC interfaces, so: slave aximm targets get initiator stubs,
// master aximm (incl. shim DMA) get target stubs, every stream/event clock is
// driven by our clock, and dangling sc_in events sink to a shared dummy. The
// real cluster->host DDR target replaces the ms_aximm stubs for GM (later).
void aiesim_top::stub_unused_ports() {
    auto* me = static_cast<MathEngineBase*>(me_);
    auto name = [](const char* p, size_t i) { return std::string(p) + std::to_string(i); };

    // 1. ss_aximm slave targets: initiator stubs for every index EXCEPT the one
    //    the ps_bridge owns (config_idx_). This stubs the NPI socket [0] and the
    //    other noc2aie interfaces.
    std::vector<xtlm::xtlm_aximm_target_socket*>& ss_rd = me->get_ss_aximm_rd();
    std::vector<xtlm::xtlm_aximm_target_socket*>& ss_wr = me->get_ss_aximm_wr();
    for (size_t i = 0; i < ss_rd.size(); ++i) {
        if (i == config_idx_) continue;
        auto* rs = new xtlm::xtlm_aximm_initiator_stub(name("ss_rd_stub_", i).c_str(), 32);
        auto* ws = new xtlm::xtlm_aximm_initiator_stub(name("ss_wr_stub_", i).c_str(), 32);
        ss_rd[i]->bind(rs->initiator_socket);
        ss_wr[i]->bind(ws->initiator_socket);
        stubs_.push_back(rs);
        stubs_.push_back(ws);
    }

    // 2. ms_aximm shim-DMA masters -> host DDR target. On AIE2 these ARE the
    //    shim DMAs (math_engine_base.h ~114, "AXI-MM Masters from the Shim
    //    DMAs"), and shim_dma_rd/wr_socket(col) returns these same sockets, so
    //    binding every ms_aximm master covers the shim-DMA set without needing
    //    the col->index map. One shared DDR store backs them all; the GM path
    //    (ess_WriteGM/ReadGM) pokes that same store via ddr_->host_write/read.
    const size_t n_mm = me->ms_aximm_rd.size();
    ddr_ = new ddr_target("ddr", n_mm);
    for (size_t i = 0; i < n_mm; ++i) {
        ddr_->bind_rd(i, *me->ms_aximm_rd[i]);
        ddr_->bind_wr(i, *me->ms_aximm_wr[i]);
    }

    // 3. stream + event clocks driven by our clock.
    for (size_t i = 0; i < me->ms_pl_event_clk.size(); ++i) me->ms_pl_event_clk[i](clock_);
    for (size_t i = 0; i < me->ss_pl_event_clk.size(); ++i) me->ss_pl_event_clk[i](clock_);
    for (size_t i = 0; i < me->ms_pl_stream_clk.size(); ++i) me->ms_pl_stream_clk[i](clock_);
    for (size_t i = 0; i < me->ss_pl_stream_clk.size(); ++i) me->ss_pl_stream_clk[i](clock_);

    // NOTE: ss_pl_event (sc_in) is bound internally by the AIE2 cluster -- aie_xtlm
    // only binds it for aie2ps (uc tiles). Binding it here double-binds (E109).
    (void)dummy_event_;

    // 4. ss_noc_axis (NoCStream128 get_port) -> tlm_fifo sources (aie_xtlm:
    // fifo_noc_to_me). The get/put exports (pl/noc streams) are cluster-provided
    // and bound internally, so they need no external binding.
    auto* noc_fifo = new sc_core::sc_vector<NoCStream128_fifo>("noc_to_me_fifo");
    noc_fifo->init(me->get_num_noc_tiles());
    me->ss_noc_axis.bind(*noc_fifo);
    stubs_.push_back(noc_fifo);
}

// Drain the shim output-stream egress. The cluster pushes a packet on the shim
// master output stream for every BD that completes with issue_token set (the
// Task Completion Token), plus other controller traffic. These are get_exports
// the cluster fills internally; the only way to drain one is to call ->get() (a
// put-side sink cannot be bound, unlike the ss_noc_axis ingress get_port). The
// reference wrapper (aietools aie_xtlm.cpp) pulls each port the same way. Left
// unconsumed the egress FIFO fills after a bounded number of tokens and back-
// pressures the shim DMA (status Stalled_TCT[5]=1, Channel_Running stuck),
// deadlocking any kernel that issues more issue_token BDs than the FIFO depth
// (e.g. sync_task_complete_token's 256 single-word input transfers wedge at ~155).
//
// We drain BOTH ms_pl_stream and ms_noc_axis. One blocking SC_THREAD per port:
// idle (suspended in get()) until the cluster emits a packet, so it costs nothing
// when quiet and drains concurrently otherwise. An un-drained egress FIFO
// back-pressures the shim DMA (Stalled_TCT) -- not just on the BD that finally
// fills it, but on every issue_token BD -- so draining is mandatory, not cleanup.
//
// What's actually on these ports (task #80 investigation):
//   - Task-completion tokens (TCTs) are REAL silicon: issue_token (BD field) +
//     Controller_ID (DMA channel reg) are real config, and npu.sync is built on
//     them (the host sequence blocks until N tokens arrive). On real NPU1 a TCT
//     does NOT travel an ME data-stream egress; it goes via an internal,
//     Controller_ID-addressed path to the host controller. aie-rt confirms this
//     negatively: there is NO stream-switch route programmed for a token (the
//     shim mux/demux at 0x1F000/0x1F004 route only DATA ports; our CDO programs
//     them to DMA type only -- verified by register dump). So the TCT appearing
//     on a PL egress here is purely a generic-Versal-cluster artifact: lacking
//     the XDNA token transport, the model dumps each token on a PL BLI port.
//   - The cluster surfaces a TCT as a SINGLE 32-bit control word: d0 = token
//     (controller-id/count, bit31 = valid), d1 = NO_STREAM_DATA filler, tlast=1.
//
// Completion is detected from DMA channel status (dma_wait watches
// Channel_Running) and result DATA travels the shim-DMA aximm path into host DDR,
// so the tokens are redundant for correctness -- but they are USEFUL TO SEE when
// debugging (which BD completed, with controller id). So on PL we CLASSIFY: a
// recognized TCT is drained (and traceable); anything else on a PL port is a
// generic-cluster routing artifact -> drained with a one-shot WARNING (opt into
// a hard fail-fast for hunting a real misroute via XDNA_AIESIM_PL_PANIC).
void aiesim_top::spawn_egress_drains() {
    auto* me = static_cast<MathEngineBase*>(me_);
    const bool trace = std::getenv("XDNA_AIESIM_TRACE") != nullptr;

    // shim->NoC egress (NoCStream128). Carries TCT / controller packets on
    // NoC-shim columns.
    const size_t n_noc = me->ms_noc_axis.size();
    for (size_t i = 0; i < n_noc; ++i) {
        sc_core::sc_spawn(
            [me, i, trace] {
                uint64_t pkts = 0;
                for (;;) {
                    NoCStreamData128 d = me->ms_noc_axis[i]->get();
                    ++pkts;
                    if (trace && (pkts % 64 == 0 || pkts <= 2 || d.tlast)) {
                        std::fprintf(stderr,
                                     "[noc-drain %zu] %llu pkts (d0=0x%08x tdest=0x%x tlast=%d)\n",
                                     i, (unsigned long long)pkts, d.data[0], d.tdest,
                                     d.tlast ? 1 : 0);
                    }
                }
            },
            (std::string("noc_drain_") + std::to_string(i)).c_str());
    }

    // shim->PL egress (MEStream64). The legitimate traffic here is the model's
    // TCT artifact (see above): a single 32-bit control word with d1 ==
    // NO_STREAM_DATA and tlast=1, which we recognize and drain (traceable).
    //
    // Anything else on a PL port is a generic-Versal-cluster routing artifact: the
    // model lacks XDNA's all-NoC routing, so traffic XDNA keeps on-chip can surface
    // on a PL/BLI port. Confirmed case (ctrl_packet_reconfig_elf): harness-injected
    // TRACE egress reaches the shim a few ns BEFORE the runtime's demux South-
    // selector write flips the port from its PL reset-default (00) to DMA (01) --
    // 3 trace beats hit PL at 18790ns; demux->DMA lands at 18827ns (muxlog). The
    // demux write IS present and routed by the bridge; the model's timing just lets
    // the trace beats outrun it. Result DATA still travels the shim-DMA aximm path,
    // so these PL beats are redundant for correctness. DEFAULT: drain + one-shot
    // WARNING (surface the fidelity gap without aborting). XDNA_AIESIM_PL_PANIC=1
    // restores a hard fail-fast at the first non-TCT beat, for hunting a genuine
    // misroute. (Faithful PL routing is a separate, deeper investigation.)
    //
    // NO_STREAM_DATA (0x77777777) is the cluster's "this 32-bit lane carries no
    // data" sentinel (aietools me_axi_stream.h); a 64-bit data beat populates d1.
    constexpr uint32_t kNoStreamData = 0x77777777u;
    const bool pl_panic = std::getenv("XDNA_AIESIM_PL_PANIC") != nullptr;
    const size_t n_pl = me->ms_pl_stream.size();
    for (size_t i = 0; i < n_pl; ++i) {
        sc_core::sc_spawn(
            [me, i, trace, pl_panic, kNoStreamData] {
                uint64_t tcts = 0, drained = 0;
                for (;;) {
                    MEStreamData64 d = me->ms_pl_stream[i]->get();
                    // A recognized TCT artifact: single 32-bit control word
                    // (upper lane is filler), end-of-packet on the one beat.
                    const bool is_tct = (d.data[1] == kNoStreamData) && d.tlast;
                    if (is_tct) {
                        ++tcts;
                        if (trace && (tcts <= 2 || tcts % 64 == 0)) {
                            std::fprintf(stderr,
                                         "[pl-tct %zu] tct#%llu d0=0x%08x (token, drained)\n",
                                         i, (unsigned long long)tcts, d.data[0]);
                        }
                        continue;
                    }
                    if (pl_panic) {
                        std::fprintf(stderr,
                            "\n[PL-PANIC] DATA entered shim->PL egress port %zu at %s\n"
                            "[PL-PANIC]   d0=0x%08x d1=0x%08x tlast=%d (after %llu TCTs)\n"
                            "[PL-PANIC] aborting (XDNA_AIESIM_PL_PANIC set; unset to drain+warn).\n\n",
                            i, sc_core::sc_time_stamp().to_string().c_str(),
                            d.data[0], d.data[1], d.tlast ? 1 : 0,
                            (unsigned long long)tcts);
                        std::fflush(stderr);
                        std::abort();
                    }
                    // Drain + warn. Surface the fidelity gap once per port (first
                    // beat), then count silently so a trace-heavy run isn't spammed;
                    // every beat is logged under XDNA_AIESIM_TRACE for debugging.
                    if (++drained == 1 || trace) {
                        std::fprintf(stderr,
                            "[pl-drain %zu] WARNING: non-TCT data on PL egress drained "
                            "(generic-cluster routing artifact) d0=0x%08x d1=0x%08x tlast=%d @%s\n",
                            i, d.data[0], d.data[1], d.tlast ? 1 : 0,
                            sc_core::sc_time_stamp().to_string().c_str());
                    }
                }
            },
            (std::string("pl_drain_") + std::to_string(i)).c_str());
    }

    if (trace) {
        std::fprintf(stderr,
                     "[egress-drain] spawned %zu ms_noc_axis + %zu ms_pl_stream drains\n",
                     n_noc, n_pl);
    }
}

// The MSM cluster model carries its OWN VCD writer (libmsm_cpp.so), distinct from
// SystemC's sc_trace_file. MathEngineBase::add_sc_traces(sc_trace_file*) actually
// reinterprets that pointer as a msm_trace::vcd_trace_file_writer* and calls
// write_comment/dump_partial_vcd_config on it -- passing a real SystemC trace file
// segfaults (confirmed by backtrace). So we create the writer with the MSM factory
// and hand THAT to add_sc_traces.
//
// We resolve the MSM trace API via dlsym(RTLD_DEFAULT) rather than linking it: the
// bridge does not link libmsm_cpp -- it arrives (RTLD_GLOBAL) as a dependency of
// the per-arch cluster .so we dlopen in the ctor. Resolving at call time (well
// after construction) both avoids a bridge load-time dependency AND guarantees we
// share the SAME libmsm instance -- hence the SAME global writer -- that the
// cluster's elaborate()/get_vcd_trace_file() finalizes. Mangled names are
// ABI-pinned to the local aietools; local-only branch, never shipped.
void aiesim_top::request_vcd_trace(const char* path) {
    vcd_path_ = path ? path : "";
}

void aiesim_top::end_of_elaboration() {
    if (vcd_path_.empty()) return;
    // By now the cluster has bound its internal ports (shim_reset_n etc.), so
    // add_sc_traces' get_interface() walk succeeds. msm_trace::msm_create_vcd_trace_file
    // (std::string, bool): type=true -> write a real file (false would mkfifo a
    // pipe). It also registers the writer as libmsm's global, which the model's
    // elaborate() finalizes via finish_trace_registration.
    using create_fn = void* (*)(std::string, bool);
    auto create = reinterpret_cast<create_fn>(dlsym(
        RTLD_DEFAULT,
        "_ZN9msm_trace25msm_create_vcd_trace_fileENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb"));
    if (!create) {
        fprintf(stderr, "[aiesim-bridge] WARNING: msm_create_vcd_trace_file unresolved; VCD disabled\n");
        return;
    }
    void* writer = create(vcd_path_, true);
    vcd_writer_ = writer;
    static_cast<MathEngineBase*>(me_)->add_sc_traces(
        reinterpret_cast<sc_core::sc_trace_file*>(writer));
}

void aiesim_top::flush_vcd_trace() {
    if (!vcd_writer_) return;
    // msm_trace::vcd_trace_file_writer::flush() -- a non-virtual no-arg member, so
    // ABI-callable as a free function taking `this` first.
    using flush_fn = void (*)(void*);
    auto flush = reinterpret_cast<flush_fn>(
        dlsym(RTLD_DEFAULT, "_ZN9msm_trace21vcd_trace_file_writer5flushEv"));
    if (flush) flush(vcd_writer_);
}

aiesim_top::~aiesim_top() {
    if (cluster_lib_) {
        dlclose(cluster_lib_);
    }
}
