// HAL-driven validation: run a REAL xaiengine config+run+check sequence against
// our in-process bridge (ps_bridge ess_*() -> timed b_transport -> the embedded
// cluster), proving the bridge drives the LIVE silicon model end to end.
//
// Kernel: 03_simple (xcve2802, from build/experiments/2026-05-31-aiesim-peano).
// Tile (1,3): the core reads buf_a[3], computes a+a+a+a+a = 5*a, stores buf_b[5],
// gated by input/output locks. Host writes 7 -> expects buf_b[5] == 35.
//
// This is the phase-2 HAL-replay path brought forward as the validation vehicle:
// the Versal aiesim flow configures the array PROCEDURALLY through the HAL (not a
// standalone CDO), so we drive the real HAL (libxaienginecdo, SIM backend) whose
// ess_*() calls land on our cluster. The HAL must run on the driver SC_THREAD --
// ess_*() do timed b_transport (wait()), and lock-polling advances sim time per
// transaction so the core runs concurrently (exactly the genwrapper ps_main model).
#define SC_INCLUDE_DYNAMIC_PROCESSES  // sc_spawn / sc_bind
#include <systemc.h>

extern "C" {
#include <xaiengine.h>
}

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "aiesim_top.h"

// Host-executable contract the dlopened cluster resolves at load (normally in
// sc_bootstrap.cpp, which we do not link -- it carries the bridge's own sc_main).
extern "C" {
bool sc_stop_at_end_of_main = false;
int plio_complete = 0;
}

// --- ess_*() the HAL needs beyond ps_bridge's config 6 -----------------------
// The SIM backend (xaie_sim.c) routes NPI writes through ess_Write32 (at the NPI
// base) and issues NPI *commands* via ess_WriteCmd. The bare ess_Npi{Write,Read}32
// symbols are referenced by OTHER backends compiled into libxaienginecdo but are
// never called on the SIM path -- link-stub them. ess_WriteCmd is the NPI command
// path (e.g. column reset); log + ignore for now and revisit if init needs it.
extern "C" {
void ess_NpiWrite32(uint64_t, uint32_t) {}
uint32_t ess_NpiRead32(uint64_t) { return 0; }
void ess_WriteCmd(unsigned char Command, unsigned char Col, unsigned char Row,
                  uint32_t CmdWd0, uint32_t CmdWd1, const char* CmdStr) {
    fprintf(stderr, "[hal] ess_WriteCmd cmd=%u col=%u row=%u w0=0x%x w1=0x%x %s\n",
            Command, Col, Row, CmdWd0, CmdWd1, CmdStr ? CmdStr : "");
}

// CDO-generation callbacks the lib references from its (unused-here) CDO
// backend. We drive the SIM backend, so these are link-only stubs.
void cdo_Write32(uint64_t, uint32_t) {}
void cdo_MaskWrite32(uint64_t, uint32_t, uint32_t) {}
void cdo_MaskPoll(uint64_t, uint32_t, uint32_t, uint32_t) {}
void cdo_BlockWrite32(uint64_t, const uint32_t*, uint32_t) {}
void cdo_BlockSet32(uint64_t, uint32_t, uint32_t) {}
}

namespace {

const char* g_elf_path = nullptr;  // main_core_1_3.elf

// Buffer offsets + lock ids from the generated aie_inc.cpp.
constexpr uint64_t kAOffset = 1024;    // buf_a base (tile-local data mem)
constexpr uint64_t kBOffset = 16384;   // buf_b base
constexpr uint8_t kInputLock = 3;
constexpr uint8_t kOutputLock = 5;

#define TRY(call)                                                              \
    do {                                                                       \
        AieRC rc__ = (call);                                                   \
        if (rc__ != XAIE_OK) {                                                 \
            fprintf(stderr, "[hal] FAILED: %s -> rc=%d\n", #call, rc__);       \
            sc_core::sc_stop();                                                \
            return;                                                            \
        }                                                                      \
    } while (0)

// The HAL config+run+check sequence, run on the driver SC_THREAD.
void hal_test_proc() {
    sc_core::wait(sc_core::SC_ZERO_TIME);  // settle start_of_simulation

    // --- init the device on the SIM backend (xcve2802 geometry per aie_inc.cpp) ---
    XAie_Config Config;
    memset(&Config, 0, sizeof(Config));
    Config.AieGen = XAIE_DEV_GEN_AIEML;
    Config.BaseAddr = 0x20000000000ULL;
    Config.ColShift = 25;
    Config.RowShift = 20;
    Config.NumRows = 11;
    Config.NumCols = 38;
    Config.ShimRowNum = 0;
    Config.MemTileRowStart = 1;
    Config.MemTileNumRows = 2;
    Config.AieTileRowStart = 3;
    Config.AieTileNumRows = 8;
    Config.Backend = XAIE_IO_BACKEND_SIM;

    XAie_DevInst DevInst;
    memset(&DevInst, 0, sizeof(DevInst));
    TRY(XAie_CfgInitialize(&DevInst, &Config));
    printf("[hal] CfgInitialize OK (backend=%d, SIM=%d)\n", DevInst.Backend->Type,
           XAIE_IO_BACKEND_SIM);
    XAie_TurnEccOff(&DevInst);  // SIM special-case (test_library.cpp:291)

    const XAie_LocType core = XAie_TileLoc(1, 3);

    // Isolation: does a data-memory write reach LIVE state via ess_*? Round-trip
    // a sentinel before the lock dance to separate "path works" from lock issues.
    {
        TRY(XAie_DataMemWrWord(&DevInst, core, 64, 0x12345678));
        u32 chk = 0xdeadbeef;
        TRY(XAie_DataMemRdWord(&DevInst, core, 64, &chk));
        printf("[hal] DM round-trip @64: wrote 0x12345678 read 0x%08x %s\n", chk,
               chk == 0x12345678u ? "(LIVE OK)" : "(MISMATCH)");
    }

    // --- configure_cores: reset/disable, release locks, load the ELF ---
    TRY(XAie_CoreReset(&DevInst, core));
    TRY(XAie_CoreDisable(&DevInst, core));
    // Config lock-release (reset all 16 locks); non-fatal -- locks may not be in
    // a releasable state pre-config, and the real flow tolerates this.
    for (uint8_t l = 0; l < 16; ++l) {
        AieRC lrc = XAie_LockRelease(&DevInst, core, XAie_LockInit(l, 0x0), 0);
        if (lrc != XAIE_OK)
            fprintf(stderr, "[hal] (non-fatal) LockRelease l=%u -> rc=%d\n", l, lrc);
    }
    printf("[hal] loading ELF %s ...\n", g_elf_path);
    TRY(XAie_LoadElf(&DevInst, core, g_elf_path, 0));
    printf("[hal] ELF loaded\n");

    // --- host interaction: acquire input lock, write 7, start core, release ---
    TRY(XAie_LockAcquire(&DevInst, core, XAie_LockInit(kInputLock, 0), 0));
    TRY(XAie_DataMemWrWord(&DevInst, core, kAOffset + 3 * 4, 7));
    {
        u32 chk = 0xdead;
        TRY(XAie_DataMemRdWord(&DevInst, core, kAOffset + 3 * 4, &chk));
        printf("[hal] buf_a[3] readback before run = %u (want 7)\n", chk);
    }

    TRY(XAie_CoreUnreset(&DevInst, core));
    TRY(XAie_CoreEnable(&DevInst, core));
    printf("[hal] core enabled; releasing input lock\n");
    TRY(XAie_LockRelease(&DevInst, core, XAie_LockInit(kInputLock, 1), 0));

    // Acquire the output lock -- blocks (polls, advancing sim time) until the
    // core finishes and releases it.
    printf("[hal] waiting on output lock (core runs)...\n");
    TRY(XAie_LockAcquire(&DevInst, core, XAie_LockInit(kOutputLock, -1), 0xFFFFFFFF));

    u32 result = 0;
    TRY(XAie_DataMemRdWord(&DevInst, core, kBOffset + 5 * 4, &result));
    printf("[hal] buf_b[5] = %u (want 35)\n", result);
    if (result == 35) {
        printf("[hal] PASS! bridge drove the live cluster end-to-end\n");
    } else {
        printf("[hal] FAIL: expected 35, got %u\n", result);
    }
    sc_core::sc_stop();
}

}  // namespace

extern "C" int sc_main(int argc, char* argv[]) {
    const char* device_json = argc > 1 ? argv[1] : nullptr;
    g_elf_path = argc > 2 ? argv[2] : nullptr;
    if (!device_json || !g_elf_path) {
        fprintf(stderr, "usage: hal_validate <VC2802.json> <main_core_1_3.elf>\n");
        return 2;
    }

    aiesim_top* top = nullptr;
    try {
        top = new aiesim_top("aiesim_top", "aie2", device_json);
    } catch (const std::exception& e) {
        fprintf(stderr, "[hal] cluster instantiation failed: %s\n", e.what());
        return 1;
    }
    (void)top;

    sc_core::sc_spawn(sc_bind(&hal_test_proc), "hal_test");
    sc_core::sc_start();  // runs until hal_test_proc sc_stop()s
    return 0;
}
