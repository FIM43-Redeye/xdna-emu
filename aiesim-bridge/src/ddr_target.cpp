#include "ddr_target.h"

#include <algorithm>
#include <cstring>
#include <string>

ddr_target::ddr_target(sc_core::sc_module_name nm, std::size_t n_masters,
                       unsigned width_bits)
    : sc_core::sc_module(nm), n_masters_(n_masters) {
    rd_util_.resize(n_masters_);
    wr_util_.resize(n_masters_);
    rd_pending_.resize(n_masters_);
    wr_pending_.resize(n_masters_);

    for (std::size_t i = 0; i < n_masters_; ++i) {
        std::string rn = "rd_util_" + std::to_string(i);
        std::string wn = "wr_util_" + std::to_string(i);
        rd_util_[i] = new xtlm::xtlm_aximm_target_rd_socket_util(
            rn.c_str(), xtlm::aximm::granularity::TRANSACTION, width_bits);
        wr_util_[i] = new xtlm::xtlm_aximm_target_wr_socket_util(
            wn.c_str(), xtlm::aximm::granularity::TRANSACTION, width_bits);

        // One spawned SC_METHOD per read/write socket, sensitive to the util's
        // request + flow-control events (same set axi_bram_memory_imp uses).
        sc_core::sc_spawn_options ro;
        ro.spawn_method();
        ro.dont_initialize();
        ro.set_sensitivity(&rd_util_[i]->transaction_available);
        ro.set_sensitivity(&rd_util_[i]->data_sampled);
        sc_core::sc_spawn(sc_bind(&ddr_target::pump_rd, this, i),
                          sc_core::sc_gen_unique_name("ddr_rd_pump"), &ro);

        sc_core::sc_spawn_options wo;
        wo.spawn_method();
        wo.dont_initialize();
        wo.set_sensitivity(&wr_util_[i]->transaction_available);
        wo.set_sensitivity(&wr_util_[i]->resp_sampled);
        sc_core::sc_spawn(sc_bind(&ddr_target::pump_wr, this, i),
                          sc_core::sc_gen_unique_name("ddr_wr_pump"), &wo);
    }
}

ddr_target::~ddr_target() {
    for (auto* u : rd_util_) delete u;
    for (auto* u : wr_util_) delete u;
}

void ddr_target::bind_rd(std::size_t i, xtlm::xtlm_aximm_initiator_socket& init) {
    init(rd_util_[i]->rd_socket);  // initiator operator() binds to our target
}

void ddr_target::bind_wr(std::size_t i, xtlm::xtlm_aximm_initiator_socket& init) {
    init(wr_util_[i]->wr_socket);
}

// --- cluster-side pumps (TRANSACTION granularity, axi_bram_memory_imp shape) --

void ddr_target::pump_rd(std::size_t i) {
    auto* util = rd_util_[i];
    auto& pending = rd_pending_[i];
    if (util->is_trans_available()) {
        xtlm::aximm_payload* trans = util->get_transaction();
        trans->set_response_status(xtlm::XTLM_OK_RESPONSE);
        mem_read(trans->get_address(), trans->get_data_ptr(),
                 trans->get_data_length());
        ++dma_txns_;  // shim-DMA activity (read) -- RUN watches this for quiescence
        pending.push_back(trans);
    }
    if (!pending.empty() && util->is_master_ready()) {
        sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
        util->send_data(*pending.front(), delay);
        pending.pop_front();
    }
}

void ddr_target::pump_wr(std::size_t i) {
    auto* util = wr_util_[i];
    auto& pending = wr_pending_[i];
    if (util->is_trans_available()) {
        xtlm::aximm_payload* trans = util->get_transaction();
        trans->set_response_status(xtlm::XTLM_OK_RESPONSE);
        const uint64_t addr = trans->get_address();
        const uint64_t size = trans->get_data_length();
        unsigned char* data = trans->get_data_ptr();
        unsigned char* be = trans->get_byte_enable_ptr();
        const uint64_t be_len = trans->get_byte_enable_length();
        if (be && be_len) {
            // Honor write-strobes byte by byte (sparse/partial writes).
            for (uint64_t k = 0; k < size; ++k) {
                if (be[k % be_len]) mem_write(addr + k, data + k, 1);
            }
        } else {
            mem_write(addr, data, size);
        }
        ++dma_txns_;  // shim-DMA activity (write) -- RUN watches this for quiescence
        pending.push_back(trans);
    }
    if (!pending.empty() && util->is_master_ready()) {
        sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
        util->send_resp(*pending.front(), delay);
        pending.pop_front();
    }
}

// --- host-side (PS) direct access: same store, zero sim-time ----------------

void ddr_target::host_write(uint64_t addr, const void* data, uint64_t size) {
    mem_write(addr, static_cast<const unsigned char*>(data), size);
}

void ddr_target::host_read(uint64_t addr, void* data, uint64_t size) {
    mem_read(addr, static_cast<unsigned char*>(data), size);
}

// --- sparse paged store -----------------------------------------------------

void ddr_target::mem_write(uint64_t addr, const unsigned char* src, uint64_t size) {
    uint64_t off = 0;
    while (off < size) {
        const uint64_t a = addr + off;
        const uint64_t page = a >> kPageBits;
        const uint64_t poff = a & (kPageSize - 1);
        const uint64_t n = std::min(size - off, kPageSize - poff);
        auto& pg = pages_[page];
        if (pg.empty()) pg.resize(kPageSize, 0);
        std::memcpy(pg.data() + poff, src + off, n);
        off += n;
    }
}

void ddr_target::mem_read(uint64_t addr, unsigned char* dst, uint64_t size) {
    uint64_t off = 0;
    while (off < size) {
        const uint64_t a = addr + off;
        const uint64_t page = a >> kPageBits;
        const uint64_t poff = a & (kPageSize - 1);
        const uint64_t n = std::min(size - off, kPageSize - poff);
        auto it = pages_.find(page);
        if (it == pages_.end()) {
            std::memset(dst + off, 0, n);  // unwritten DDR reads as zero
        } else {
            std::memcpy(dst + off, it->second.data() + poff, n);
        }
        off += n;
    }
}
