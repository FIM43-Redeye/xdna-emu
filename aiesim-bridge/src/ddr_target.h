// Host DDR model: the cluster->host memory the shim-DMA AXI-MM masters target.
//
// On AIE2, MathEngineBase::ms_aximm_rd/wr ARE the shim-DMA masters (see
// math_engine_base.h line ~114, "AXI-MM Masters from the Shim DMAs"); the
// shim_dma_rd/wr_socket(col) accessors return those same sockets. The real
// aiesim flow (aie_xtlm) binds them to a NoC + external DDR; we provide that DDR
// directly. Two access faces onto ONE sparse store:
//   * cluster side  -- per-master AXI-MM target sockets, pumped at TRANSACTION
//     granularity (modeled on axi_bram_memory_imp), service shim-DMA traffic.
//   * host (PS) side -- host_write/host_read, zero sim-time, for ess_WriteGM/
//     ReadGM (the PS poking its own DDR before/after a run).
//
// One host DDR address space, absolute addresses (the CDO programs shim DMA with
// the resolved host-buffer address), so no translation: a write by the PS at A
// is read by a shim DMA at A. Sparse 64 KB pages -- DDR is a 64-bit space.
#pragma once

#include <systemc.h>
#include <xtlm.h>
#include <utils/xtlm_aximm_target_rd_socket_util.h>
#include <utils/xtlm_aximm_target_wr_socket_util.h>

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

class ddr_target : public sc_core::sc_module {
public:
    SC_HAS_PROCESS(ddr_target);
    // n_masters rd + wr target sockets; width_bits = AXI data width (must match
    // the cluster shim-DMA initiator sockets -- 32 binds, per the proven stubs).
    ddr_target(sc_core::sc_module_name nm, std::size_t n_masters,
               unsigned width_bits = 32);
    ~ddr_target() override;

    // Bind target socket i to a cluster shim-DMA initiator (its operator() binds
    // the pair). i in [0, n_masters).
    void bind_rd(std::size_t i, xtlm::xtlm_aximm_initiator_socket& init);
    void bind_wr(std::size_t i, xtlm::xtlm_aximm_initiator_socket& init);

    // Host (PS) direct access -- zero sim-time, same store the masters see.
    void host_write(uint64_t addr, const void* data, uint64_t size);
    void host_read(uint64_t addr, void* data, uint64_t size);

    std::size_t num_masters() const { return n_masters_; }

private:
    // Spawned (per-socket) SC_METHOD pumps; mirror axi_bram_memory_imp.
    void pump_rd(std::size_t i);
    void pump_wr(std::size_t i);

    // Sparse paged backing store (absolute DDR addresses).
    void mem_read(uint64_t addr, unsigned char* dst, uint64_t size);
    void mem_write(uint64_t addr, const unsigned char* src, uint64_t size);

    std::size_t n_masters_;
    std::vector<xtlm::xtlm_aximm_target_rd_socket_util*> rd_util_;
    std::vector<xtlm::xtlm_aximm_target_wr_socket_util*> wr_util_;
    std::vector<std::deque<xtlm::aximm_payload*>> rd_pending_;
    std::vector<std::deque<xtlm::aximm_payload*>> wr_pending_;

    static constexpr uint64_t kPageBits = 16;          // 64 KB pages
    static constexpr uint64_t kPageSize = uint64_t(1) << kPageBits;
    std::unordered_map<uint64_t, std::vector<unsigned char>> pages_;
};
