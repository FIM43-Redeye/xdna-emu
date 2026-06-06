// In-process SystemC bootstrap for the aiesim bridge.
//
// Recipe: docs/superpowers/findings/2026-06-01-aiesim-inprocess-backend-feasibility.md
//
// A .so does not run its own main(), so we compile sc_main_main.cpp (which
// provides sc_elab_and_sim, the entry the bridge calls) but NOT sc_main.cpp.
// sc_elab_and_sim prints the SystemC banner then invokes our sc_main() below.
//
// Threading model (findings doc, 2026-06-02): register access needs TIMED
// b_transport (the backdoor reaches only a shadow store), and b_transport
// wait()s -> illegal in sc_main. So ONE in-kernel SC_THREAD (driver_proc) owns
// all register access + time advance; sc_main just hosts the kernel in
// pause/resume windows. The kernel is paused between commands (the driver
// sc_pause()s after draining; sc_main only sc_start()s when work is pending) ->
// no time drift -> cycle-precise stepping is first-class, and free-run is just a
// large RUN budget. The OS thread (C-ABI) wakes the kernel via a Doorbell
// (async_request_update), the one thread-safe OS->kernel hook.
#include <systemc.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

#include "aiesim_top.h"
#include "cdo_replay.h"
#include "ddr_target.h"
#include "npu_replay.h"
#include "ps_bridge.h"
#include "service_thread.h"

// Host-executable contract: the dlopened cluster resolves these at load. plio_
// complete is set by the cluster's PLIO models when the workload finishes -- the
// natural-completion signal for RUN.
extern "C" {
bool sc_stop_at_end_of_main = false;
int plio_complete = 0;
}

// Arch/device_json hand-off. Service::start sets these before spawning the
// kernel thread; sc_main reads them to construct the cluster.
extern "C" {
const char* g_aiesim_arch = nullptr;
const char* g_aiesim_device_json = nullptr;
void* g_aiesim_top = nullptr;  // aiesim_top*
}

namespace {

// Registered host buffers (DDR addr, size). ADD/CLEAR_HOST_BUF maintain this;
// exec_npu (DdrPatch resolution) consumes it. Driver-thread only.
std::vector<std::pair<uint64_t, std::size_t>> g_host_buffers;

// Partition physical start column (SET_START_COL). Drives the NPU1->Versal
// address translation in cdo_replay/npu_replay (logical col 0 -> physical
// start_col). Default 0: a single-partition kernel runs correctly at column 0
// (consistent logical->physical mapping); the plugin sets the real value.
uint8_t g_start_col = 0;

// Doorbell: the OS thread rings async_request_update() (thread-safe); the kernel
// runs update() in its next update phase, notifying the driver's command event.
struct Doorbell : sc_core::sc_prim_channel {
    sc_core::sc_event cmd_event;
    void update() override { cmd_event.notify(sc_core::SC_ZERO_TIME); }
};
Doorbell* g_doorbell = nullptr;

// RUN: advance up to `budget` cycles (1 ns clock => 1 cycle/ns) via wait() (we
// are in a process), stepping in quanta. Completes EARLY on natural quiescence
// rather than grinding the whole budget: exec_npu already replayed the runtime
// sequence to completion (its Sync ops block on the live DMA), so by RUN time
// the shim DMAs have drained -- we detect that by watching the cluster-side
// shim-DMA transaction counter go idle for a short settle window. This mirrors
// the interpreter's natural-completion ("DMAs drained") and avoids the hang the
// XRT-plugin path hit when it passes its large interpreter-oriented max_cycles.
// plio_complete still short-circuits for streaming kernels that fire it.
// Fills reply_int (0=Completed, 1=Budget) + reply_cycles.
void do_run(aiesim_top* top, aiesim::Command* c) {
    constexpr uint64_t kQuantum = 1024;
    // Quanta of no shim-DMA activity before we call it quiescent. 4 quanta
    // (~4096 cycles) clears the tiny settle the timed tests used (2000) with
    // margin while staying negligible against any real budget.
    constexpr uint64_t kSettleQuanta = 4;
    const sc_core::sc_time one_ns(1.0, sc_core::SC_NS);
    const sc_core::sc_time t0 = sc_core::sc_time_stamp();
    ddr_target* ddr = top ? top->ddr() : nullptr;

    uint64_t ran = 0;
    int halt = 1;  // Budget unless we detect quiescence / plio_complete
    uint64_t last_txns = ddr ? ddr->dma_txn_count() : 0;
    uint64_t idle_quanta = 0;
    while (ran < c->budget) {
        const uint64_t step = std::min<uint64_t>(kQuantum, c->budget - ran);
        sc_core::wait(sc_core::sc_time(static_cast<double>(step), sc_core::SC_NS));
        ran += step;
        if (plio_complete) { halt = 0; break; }
        if (ddr) {
            const uint64_t txns = ddr->dma_txn_count();
            if (txns != last_txns) {
                last_txns = txns;
                idle_quanta = 0;
            } else if (++idle_quanta >= kSettleQuanta) {
                halt = 0;  // shim DMAs have drained -> natural completion
                break;
            }
        }
    }
    c->reply_cycles = static_cast<uint64_t>((sc_core::sc_time_stamp() - t0) / one_ns);
    c->reply_int = halt;
}

// Execute one command on the driver SC_THREAD (timed b_transport legal here).
void execute(aiesim_top* top, aiesim::Command* c) {
    ps_bridge* ps = top->ps();
    switch (c->tag) {
        case aiesim::Command::READ_REG:
            c->reply_u32 = ps->read32(c->addr);  // TIMED live read
            c->reply_int = 0;
            break;
        case aiesim::Command::RUN:
            do_run(top, c);
            break;
        case aiesim::Command::WRITE_GM:
            top->ddr()->host_write(c->addr, c->in_ptr, c->len);
            c->reply_int = 0;
            break;
        case aiesim::Command::READ_GM:
            top->ddr()->host_read(c->addr, c->out_ptr, c->len);
            c->reply_int = 0;
            break;
        case aiesim::Command::ADD_HOST_BUF:
            g_host_buffers.emplace_back(c->addr, c->len);
            c->reply_int = 0;
            break;
        case aiesim::Command::CLEAR_HOST_BUF:
            g_host_buffers.clear();
            c->reply_int = 0;
            break;
        case aiesim::Command::LOAD_CDO:
            c->reply_int = aiesim::cdo_replay(ps, c->in_ptr, c->len, g_start_col);
            break;
        case aiesim::Command::EXEC_NPU:
            c->reply_int = aiesim::npu_replay(ps, top->ddr(), c->in_ptr, c->len, g_start_col,
                                              g_host_buffers);
            break;
        case aiesim::Command::SET_START_COL:
            // XDNA_AIESIM_START_COL overrides the xclbin's start_col -- a debug
            // knob for placing the partition on a specific Versal column (e.g. a
            // NoC shim column, since PL-shim columns route the output stream to
            // an unconnected PL port).
            if (const char* e = std::getenv("XDNA_AIESIM_START_COL")) {
                g_start_col = static_cast<uint8_t>(std::strtoul(e, nullptr, 10));
            } else {
                g_start_col = static_cast<uint8_t>(c->addr);
            }
            c->reply_int = 0;
            break;
        case aiesim::Command::RESET:
            // No-op success for now: the cluster cannot be reset in-process
            // (SystemC is single-shot) and single-submission runs do not need
            // it. Multi-submission state reset is a follow-up (destroy+recreate
            // or a CDO re-apply); returning Ok keeps reset_for_new_context from
            // failing the run.
            c->reply_int = 0;
            break;
        default:
            c->reply_int = 1;  // unknown command
            break;
    }
}

// The driver: the single in-kernel process that touches the cluster. Drains the
// queue, then pauses (releasing sc_main's sc_start) and waits for the next ring.
void driver_proc(aiesim_top* top) {
    auto& svc = aiesim::Service::instance();
    sc_core::wait(sc_core::SC_ZERO_TIME);  // settle start_of_simulation
    svc.publish(top, true);                // release create()

    for (;;) {
        // Drain everything queued (the queue is the source of truth; a ring may
        // coalesce, but nothing enqueued is missed).
        for (aiesim::Command* c = svc.try_pop(); c; c = svc.try_pop()) {
            if (c->tag == aiesim::Command::SHUTDOWN) {
                c->reply_int = 0;
                svc.mark_shutdown();
                svc.complete(c);
                sc_core::sc_stop();  // unwind sc_main's sc_start loop
                return;
            }
            execute(top, c);
            svc.complete(c);
        }
        sc_core::sc_pause();              // let sc_main's sc_start() return (kernel idle)
        sc_core::wait(g_doorbell->cmd_event);  // suspend until the next ring
    }
}

}  // namespace

// sc_elab_and_sim (sc_main_main.cpp) calls this with C linkage on the kernel
// thread. Construct the cluster, spawn the driver, then host the kernel in
// pause/resume windows until SHUTDOWN.
extern "C" int sc_main(int /*argc*/, char* /*argv*/[]) {
    aiesim_top* top = nullptr;
    try {
        top = new aiesim_top("aiesim_top", g_aiesim_arch, g_aiesim_device_json);
    } catch (const std::exception& e) {
        std::cerr << "[aiesim-bridge] cluster instantiation failed: " << e.what()
                  << std::endl;
        aiesim::Service::instance().publish(nullptr, false);
        return 1;
    }
    g_aiesim_top = top;

    g_doorbell = new Doorbell();
    auto& svc = aiesim::Service::instance();
    svc.set_wake([] {
        if (g_doorbell) g_doorbell->async_request_update();
    });
    sc_core::sc_spawn(sc_bind(&driver_proc, top), "aiesim_driver");

    // Drain the shim output-stream egress (TCT packets) so it cannot fill and
    // back-pressure the shim DMA. One blocking SC_THREAD per ms_pl_stream /
    // ms_noc_axis port; see aiesim_top::spawn_egress_drains.
    top->spawn_egress_drains();

    // Optional NPU1-native VCD dump for three-way timing calibration. When
    // XDNA_AIESIM_VCD is set, register the cluster's signals BEFORE the first
    // sc_start (trace registration must precede time advance; the model's
    // elaborate() finalizes the header on that first sc_start). The waveform is
    // in NPU1 5x6 geometry -- identical to the trace-BO/HW side -- so no vc2802
    // row remap is needed (docs/coverage/three-way-timing-calibration.md). The
    // pause/resume host loop is invisible to the VCD: it records sim time (which
    // is monotonic across sc_pause), not wall time, so the timeline is continuous.
    // Local-only branch (links proprietary aietools); never shipped.
    bool vcd_enabled = false;
    if (const char* vcd_path = std::getenv("XDNA_AIESIM_VCD")) {
        if (*vcd_path) {
            top->request_vcd_trace(vcd_path);  // registered in end_of_elaboration()
            vcd_enabled = true;
            std::cerr << "[aiesim-bridge] VCD trace -> " << vcd_path << std::endl;
        }
    }

    // Host loop: advance the kernel only while a command is pending; the driver
    // sc_pause()s after each drain, returning control here.
    for (;;) {
        sc_core::sc_start();           // runs until the driver sc_pause()s / sc_stop
        if (svc.is_shutdown()) break;
        svc.wait_for_pending();        // block (kernel paused) until a command arrives
    }

    if (vcd_enabled) {
        top->flush_vcd_trace();        // push the writer's tail buffer to disk
    }
    return 0;
}

// Start the SystemC kernel once (sc_elab_and_sim -> sc_main). Runs on the kernel
// thread; returns only after SHUTDOWN unwinds sc_main.
extern "C" int aiesim_bridge_start_systemc() {
    char arg0[] = "aiesim-bridge";
    char* argv[] = {arg0, nullptr};
    return sc_core::sc_elab_and_sim(1, argv);
}
