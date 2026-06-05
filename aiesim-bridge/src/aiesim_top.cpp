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
// We drain BOTH ms_pl_stream and ms_noc_axis. Real NPU1 is all-NoC (no PL
// fabric), but the cluster model is the generic Versal AIE-ML array, which
// exposes PL stream ports; with our partition placed at Versal start_col 0 the
// shim routes its TCT/control output stream to a PL egress port (ms_pl_stream),
// not the NoC one -- a placement artifact of emulating a NoC-only NPU on the
// generic model, not real-silicon behavior (on hardware that port does not
// exist). Draining both makes the bridge column-type-agnostic and also removes
// a pervasive per-transfer stall (the un-drained FIFO back-pressures every BD,
// not just the one that finally fills it).
//
// We discard the packets: completion is detected from the DMA channel status
// (dma_wait watches Channel_Running), and result DATA travels the shim-DMA aximm
// path into host DDR -- the stream egress carries only the redundant completion-
// token / controller notifications, for which we model no destination. (A future
// TCT-accurate completion model could parse these instead of discarding.) One
// blocking SC_THREAD per port: idle (suspended in get()) until the cluster emits
// a packet, so it costs nothing when quiet and drains concurrently otherwise.
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

    // shim->PL egress (MEStream64). On PL-shim columns the shim output stream
    // (incl. the TCT) routes here; left unconsumed it back-pressures the shim
    // DMA exactly like the NoC egress.
    const size_t n_pl = me->ms_pl_stream.size();
    for (size_t i = 0; i < n_pl; ++i) {
        sc_core::sc_spawn(
            [me, i, trace] {
                uint64_t pkts = 0;
                for (;;) {
                    MEStreamData64 d = me->ms_pl_stream[i]->get();
                    ++pkts;
                    if (trace && (pkts % 64 == 0 || pkts <= 2 || d.tlast)) {
                        std::fprintf(stderr,
                                     "[pl-drain %zu] %llu pkts (d0=0x%08x d1=0x%08x tlast=%d)\n",
                                     i, (unsigned long long)pkts, d.data[0], d.data[1],
                                     d.tlast ? 1 : 0);
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

aiesim_top::~aiesim_top() {
    if (cluster_lib_) {
        dlclose(cluster_lib_);
    }
}
