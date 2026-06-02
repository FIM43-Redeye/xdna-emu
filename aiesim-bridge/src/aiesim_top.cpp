#include "aiesim_top.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <utils/xtlm_aximm_initiator_stub.h>
#include <utils/xtlm_aximm_target_stub.h>

#include "math_engine_base.h"  // the closed cluster ABI (aietools, build-time ref)
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
    const char* lib = arch_to_cluster_lib(arch_s);
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
    ps_->ps_axi_rd.bind(*ss_rd[0]);
    ps_->ps_axi_wr.bind(*ss_wr[0]);

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

    // 1. ss_aximm slave targets [1..]: initiator stubs ([0] is the ps_bridge).
    std::vector<xtlm::xtlm_aximm_target_socket*>& ss_rd = me->get_ss_aximm_rd();
    std::vector<xtlm::xtlm_aximm_target_socket*>& ss_wr = me->get_ss_aximm_wr();
    for (size_t i = 1; i < ss_rd.size(); ++i) {
        auto* rs = new xtlm::xtlm_aximm_initiator_stub(name("ss_rd_stub_", i).c_str(), 32);
        auto* ws = new xtlm::xtlm_aximm_initiator_stub(name("ss_wr_stub_", i).c_str(), 32);
        ss_rd[i]->bind(rs->initiator_socket);
        ss_wr[i]->bind(ws->initiator_socket);
        stubs_.push_back(rs);
        stubs_.push_back(ws);
    }

    // 2. ms_aximm master initiators (incl. shim DMA): target stubs.
    for (size_t i = 0; i < me->ms_aximm_rd.size(); ++i) {
        auto* rs = new xtlm::xtlm_aximm_target_stub(name("ms_rd_stub_", i).c_str(), 32);
        auto* ws = new xtlm::xtlm_aximm_target_stub(name("ms_wr_stub_", i).c_str(), 32);
        (*me->ms_aximm_rd[i])(rs->target_socket);
        (*me->ms_aximm_wr[i])(ws->target_socket);
        stubs_.push_back(rs);
        stubs_.push_back(ws);
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

aiesim_top::~aiesim_top() {
    if (cluster_lib_) {
        dlclose(cluster_lib_);
    }
}
