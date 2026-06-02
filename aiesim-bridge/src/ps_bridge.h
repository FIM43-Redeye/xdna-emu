// PS-side bridge: provides the ess_*() seam as AXI-MM TLM into the cluster.
//
// Modeled on PSIP_ps_i3 (mlir-aie aie_runtime_lib/AIE2/aiesim/genwrapper_for_ps.cpp),
// but a plain sc_module driven externally -- NOT an adf IPBlock with a ps_main()
// SC_THREAD. The service thread (II-B.3) pumps our command queue and calls these
// methods from within a SystemC process context (b_transport may wait()).
//
// Two bindings (done in aiesim_top, II-B.1 Step 4):
//   host -> cluster (config/MMIO): our initiator sockets -> me->get_ss_aximm_*()[0]
//   cluster -> host (DDR):         me->shim_dma_*_socket(col) -> our DDR target
#pragma once

#include <systemc.h>
#include <xtlm.h>

#include <cstdint>

class ps_bridge : public sc_core::sc_module {
public:
    SC_HAS_PROCESS(ps_bridge);
    explicit ps_bridge(sc_core::sc_module_name nm);
    ~ps_bridge() override;

    // Config/MMIO path (PS -> cluster ss_aximm). b_transport-based: must be
    // called from a SystemC process context.
    void write32(uint64_t addr, uint32_t data);
    uint32_t read32(uint64_t addr);
    void write128(uint64_t addr, const uint32_t* data);  // 4 words
    void read128(uint64_t addr, uint32_t* data);
    void writeGM(uint64_t addr, const void* data, uint64_t size);
    void readGM(uint64_t addr, void* data, uint64_t size);

    // Tier-2 zero-time backdoor register read/write (TLM transport_dbg).
    // Callable outside a process context (does not advance sim time).
    uint32_t read32_backdoor(uint64_t addr);
    void write32_backdoor(uint64_t addr, uint32_t data);

    // Initiator sockets to the cluster's config ss_aximm targets (bound in
    // aiesim_top). Public so aiesim_top can bind them.
    xtlm::xtlm_aximm_initiator_socket ps_axi_rd;
    xtlm::xtlm_aximm_initiator_socket ps_axi_wr;

    // Process-singleton, so the ess_*() free functions can reach the instance
    // exactly as genwrapper's PSIP_ps_i3::getInstance() does.
    static ps_bridge* instance() { return s_inst; }

private:
    // Burst transaction used by writeGM/readGM (genwrapper aximm_transaction).
    void aximm_transaction(xtlm::xtlm_command command, uint64_t address,
                           unsigned char* data, unsigned int size_in_bytes);
    void set_payload_attr(xtlm::aximm_payload* trans, size_t bytes);

    xtlm::xtlm_aximm_initiator_rd_socket_util* rd_util_;
    xtlm::xtlm_aximm_initiator_wr_socket_util* wr_util_;
    xtlm::xtlm_aximm_mem_manager* mem_manager_;

    static ps_bridge* s_inst;
};
