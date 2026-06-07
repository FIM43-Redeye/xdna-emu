// ps_bridge -- the ess_*() PS seam. TLM bodies reproduce PSIP_ps_i3
// (genwrapper_for_ps.cpp); see ps_bridge.h for the structural differences.
#include "ps_bridge.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
constexpr unsigned BUSWIDTH = 128;  // bits; matches genwrapper PSIP_ps_i3
}

ps_bridge* ps_bridge::s_inst = nullptr;

ps_bridge::ps_bridge(sc_core::sc_module_name nm)
    : sc_core::sc_module(nm),
      ps_axi_rd("ps_axi_rd", BUSWIDTH),
      ps_axi_wr("ps_axi_wr", BUSWIDTH) {
    rd_util_ = new xtlm::xtlm_aximm_initiator_rd_socket_util(
        "ps_axi_rd_util", xtlm::aximm::TRANSACTION, BUSWIDTH);
    wr_util_ = new xtlm::xtlm_aximm_initiator_wr_socket_util(
        "ps_axi_wr_util", xtlm::aximm::TRANSACTION, BUSWIDTH);
    mem_manager_ = new xtlm::xtlm_aximm_mem_manager(this);
    rd_util_->rd_socket.bind(ps_axi_rd);
    wr_util_->wr_socket.bind(ps_axi_wr);
    s_inst = this;
}

ps_bridge::~ps_bridge() {
    delete rd_util_;
    delete wr_util_;
    delete mem_manager_;
    if (s_inst == this) s_inst = nullptr;
}

// Common payload attributes for a single-beat config transaction.
void ps_bridge::set_payload_attr(xtlm::aximm_payload* trans, size_t bytes) {
    trans->create_and_get_data_ptr(bytes);
    unsigned char* be = trans->create_and_get_byte_enable_ptr(bytes);
    for (size_t i = 0; i < bytes; ++i) be[i] = 0xff;
    trans->set_data_length(bytes);
    trans->set_axi_id(0);
    trans->set_burst_length(1);
    trans->set_burst_size(bytes);
    trans->set_burst_type(1);  // INCR -- genwrapper PSIP_ps_i3 sets this on every
                               // beat; without it the config target sees FIXED.
}

void ps_bridge::write32(uint64_t addr, uint32_t data) {
    // DIAG (XDNA_AIESIM_MUXLOG): log shim stream MUX(0x1F000)/DEMUX(0x1F004)
    // writes -- the selectors that route an egress stream to PL(00)/DMA(01)/
    // NoC(10). Trace egress should set the trace stream's South selector to DMA;
    // if it stays at the PL reset default the trace lands on a PL port.
    if (const uint32_t off = static_cast<uint32_t>(addr & 0xFFFFFu);
        off == 0x1F000u || off == 0x1F004u) {
        if (std::getenv("XDNA_AIESIM_MUXLOG")) {
            const uint64_t rel = addr - 0x20000000000ULL;
            std::fprintf(stderr,
                "[muxlog] %s col=%llu row=%llu val=0x%08x @%s\n",
                off == 0x1F004u ? "DEMUX" : "MUX",
                (unsigned long long)((rel >> 25) & 0x7F),
                (unsigned long long)((rel >> 20) & 0x1F),
                data, sc_core::sc_time_stamp().to_string().c_str());
        }
    }
    const size_t n = sizeof(uint32_t);
    xtlm::aximm_payload* trans = mem_manager_->get_payload();
    trans->acquire();
    set_payload_attr(trans, n);
    trans->set_command(xtlm::XTLM_WRITE_COMMAND);
    trans->set_address(addr);
    std::memcpy(trans->get_data_ptr(), &data, n);
    sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
    wr_util_->b_transport(*trans, delay);
    trans->release();
}

uint32_t ps_bridge::read32(uint64_t addr) {
    const size_t n = sizeof(uint32_t);
    xtlm::aximm_payload* trans = mem_manager_->get_payload();
    trans->acquire();
    set_payload_attr(trans, n);
    trans->set_command(xtlm::XTLM_READ_COMMAND);
    trans->set_address(addr);
    sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
    rd_util_->b_transport(*trans, delay);
    uint32_t data = *reinterpret_cast<uint32_t*>(trans->get_data_ptr());
    trans->release();
    return data;
}

void ps_bridge::write128(uint64_t addr, const uint32_t* data) {
    const size_t n = 4 * sizeof(uint32_t);
    xtlm::aximm_payload* trans = mem_manager_->get_payload();
    trans->acquire();
    set_payload_attr(trans, n);
    trans->set_command(xtlm::XTLM_WRITE_COMMAND);
    trans->set_address(addr);
    std::memcpy(trans->get_data_ptr(), data, n);
    sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
    wr_util_->b_transport(*trans, delay);
    trans->release();
}

void ps_bridge::read128(uint64_t addr, uint32_t* data) {
    const size_t n = 4 * sizeof(uint32_t);
    xtlm::aximm_payload* trans = mem_manager_->get_payload();
    trans->acquire();
    set_payload_attr(trans, n);
    trans->set_command(xtlm::XTLM_READ_COMMAND);
    trans->set_address(addr);
    sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
    rd_util_->b_transport(*trans, delay);
    std::memcpy(data, trans->get_data_ptr(), n);
    trans->release();
}

// Burst transaction (AXI max 4096 B/transfer), chunked by writeGM/readGM.
void ps_bridge::aximm_transaction(xtlm::xtlm_command command, uint64_t address,
                                  unsigned char* data, unsigned int size) {
    unsigned int beats = (size + 15) / 16;  // 16-byte (128-bit) beats
    xtlm::aximm_payload* payload = mem_manager_->get_payload();
    sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
    payload->acquire();
    payload->set_response_status(xtlm::XTLM_INCOMPLETE_RESPONSE);
    payload->set_command(command);
    payload->set_data_ptr(data, size);  // caller owns the memory
    payload->set_address(address);
    payload->set_burst_type(1);  // INCR
    payload->set_burst_length(beats);
    payload->set_burst_size(16);  // 16-byte burst size (buswidth)
    if (command == xtlm::XTLM_READ_COMMAND) {
        if (!rd_util_->is_slave_ready()) wait(rd_util_->transaction_sampled);
        rd_util_->send_transaction(*payload, delay);
        wait(rd_util_->data_available);
        payload = rd_util_->get_data();
    } else {
        if (!wr_util_->is_slave_ready()) wait(wr_util_->transaction_sampled);
        wr_util_->send_transaction(*payload, delay);
        wait(wr_util_->resp_available);
        payload = wr_util_->get_resp();
    }
}

void ps_bridge::writeGM(uint64_t addr, const void* data, uint64_t size) {
    uint64_t remaining = size, cur = addr;
    unsigned char* p = const_cast<unsigned char*>(static_cast<const unsigned char*>(data));
    while (remaining >= 4096) {
        aximm_transaction(xtlm::XTLM_WRITE_COMMAND, cur, p, 4096);
        cur += 4096; p += 4096; remaining -= 4096;
    }
    if (remaining > 0) aximm_transaction(xtlm::XTLM_WRITE_COMMAND, cur, p, remaining);
}

void ps_bridge::readGM(uint64_t addr, void* data, uint64_t size) {
    uint64_t remaining = size, cur = addr;
    unsigned char* p = static_cast<unsigned char*>(data);
    while (remaining >= 4096) {
        aximm_transaction(xtlm::XTLM_READ_COMMAND, cur, p, 4096);
        cur += 4096; p += 4096; remaining -= 4096;
    }
    if (remaining > 0) aximm_transaction(xtlm::XTLM_READ_COMMAND, cur, p, remaining);
}

void ps_bridge::write32_backdoor(uint64_t addr, uint32_t data) {
    const size_t n = sizeof(uint32_t);
    xtlm::aximm_payload* trans = mem_manager_->get_payload();
    trans->acquire();
    set_payload_attr(trans, n);
    trans->set_command(xtlm::XTLM_WRITE_COMMAND);
    trans->set_address(addr);
    std::memcpy(trans->get_data_ptr(), &data, n);
    ps_axi_wr->transport_dbg(*trans);
    trans->release();
}

uint32_t ps_bridge::read32_backdoor(uint64_t addr) {
    // TLM debug transport: zero sim-time, no process context required.
    const size_t n = sizeof(uint32_t);
    xtlm::aximm_payload* trans = mem_manager_->get_payload();
    trans->acquire();
    set_payload_attr(trans, n);
    trans->set_command(xtlm::XTLM_READ_COMMAND);
    trans->set_address(addr);
    uint32_t data = 0;
    unsigned int got = ps_axi_rd->transport_dbg(*trans);
    if (got >= n) std::memcpy(&data, trans->get_data_ptr(), n);
    trans->release();
    return data;
}

// ---- the ess_*() seam (free functions the cluster's HAL-equivalent calls) ----
// Exactly the genwrapper set; NO ess_WriteCmd (HAL NPI-command path, not ours).
extern "C" {
void ess_Write32(uint64_t addr, uint32_t data) { ps_bridge::instance()->write32(addr, data); }
uint32_t ess_Read32(uint64_t addr) { return ps_bridge::instance()->read32(addr); }
void ess_Write128(uint64_t addr, uint32_t* data) { ps_bridge::instance()->write128(addr, data); }
void ess_Read128(uint64_t addr, uint32_t* data) { ps_bridge::instance()->read128(addr, data); }
void ess_WriteGM(uint64_t addr, const void* data, uint64_t size) {
    ps_bridge::instance()->writeGM(addr, data, size);
}
void ess_ReadGM(uint64_t addr, void* data, uint64_t size) {
    ps_bridge::instance()->readGM(addr, data, size);
}
}  // extern "C"
