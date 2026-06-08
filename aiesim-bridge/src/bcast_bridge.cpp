// See bcast_bridge.h for the full rationale. This file implements the three
// phases: (1) capture memtile EventBroadcast objects by GOT-interposing the bare
// ctor, (2) resolve each lane's SOUTH placeholder channel + shim north_m signal,
// (3) drive the mirror once per clock posedge.
#include "bcast_bridge.h"

#define SC_INCLUDE_DYNAMIC_PROCESSES  // sc_spawn / sc_spawn_options
#include <systemc.h>

#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace aiesim {
namespace {

// RVAs / struct offsets within libaie2_cluster_msm_v1_0_0.osci.so (this aietools
// build). Every one is sanity-checked at use; a mismatch self-disables the bridge
// rather than corrupting the model (an updated lib or different arch).
constexpr unsigned long RVA_BARE_CTOR = 0xe33660UL;   // EventBroadcast::EventBroadcast(...)
constexpr unsigned long RVA_GOT_CTOR  = 0x331c0a0UL;  // JUMP_SLOT GOT slot for the ctor
constexpr unsigned long RVA_DISC_SRC  = 0xdcdb20UL;   // vtable[0x18] discriminator of a source channel
constexpr unsigned long OFF_SOUTH     = 0x3b8UL;      // EventBroadcast+0x3b8 = SOUTH source descriptor
constexpr unsigned long OFF_NEWVAL    = 0x8UL;        // msm_prim_channel new_val (producer writes)
constexpr unsigned long OFF_CURVAL    = 0xaUL;        // msm_prim_channel cur_val (consumer reads)
// aie_sc::EventBroadcast+0x1e90 = the shim's COMBINED local broadcast value (uint16):
// output #5 of generate_outputs (shim_genout.asm e7defd-e7df16), the full union of all
// inputs `0x1c38|0x1d00|0x1b70|0x1dc8|0x1ea8` with NO directional block applied -- the
// actual broadcast bits present at this shim, and what the shim's internal event
// consumers read (e7df4f reads 0x1e90). On real HW, with block_north == 0 (resets to 0,
// the design programs no block masks), this is exactly what propagates north to the
// memtile. Set in EVERY column that carries the broadcast.
//
// Why NOT the north output cur_val at 0x1aa8 (0x19f8+0xb0): the model's north output is
// computed as `user | south(0x1d00) | r14` -- it STRUCTURALLY EXCLUDES the lateral
// (east/west) inputs (0x1c38/0x1b70), so a laterally-received broadcast never drives
// north in the model. That made 0x1aa8 == 0x8000 only in the origin column. This is an
// aiesim partial-cluster artifact, NOT real HW: (1) all Event_Broadcast_*_Block_* reset
// to 0 and the design programs none, so HW floods north everywhere; (2) this is a real
// mlir-aie HW test that passes -- the broadcast demonstrably reaches tile(0,2) on
// silicon. So mirroring the combined value 0x1e90 into every column's memtile faithfully
// reproduces the real-HW 2D flood. (Read directly as plain memory -- the sc_signal value
// cannot be reached via a cross-.so dynamic_cast.) Verdict + disassembly: FINDINGS.md.
constexpr unsigned long OFF_SHIM_BCAST = 0x1e90UL;

// The memtile EventBroadcast's OWN embedded "internal south" msm_prim_channel,
// at EB+0x1700. This is the faithful injection point. Per the model's own
// read_inputs() (cluster lib e315ad-e315c5), the south input is consumed as:
//   v = south_source(+0x3b8).cur_val;  internal_south.new_val(+0x1708) = v;
//   request_update(internal_south);
// and generate_outputs() (e3185e/e31842) then ORs internal_south.cur_val(+0x170a)
// into the north output and request_updates it -- propagating the broadcast up the
// array. Unlike the external +0x3b8 placeholder (whose consumer list is unwired in
// our partial cluster -> request_update faults in sync_update's consumer walk), the
// internal_south channel is constructed and wired by the EB itself (consumer =
// generate_outputs), so request_update on it is safe. We replicate exactly the
// read_inputs south tail, bypassing only the model's disconnected-south guard
// (read_inputs e31582) -- an artifact of our shim<->array wiring gap; on real
// silicon the broadcast reaches the memtile and propagates.
constexpr unsigned long OFF_INT_SOUTH     = 0x1700UL;  // internal south msm_prim_channel
constexpr unsigned long OFF_INT_SOUTH_NEW = 0x1708UL;  // its new_val (0x1700+8)
constexpr long OFF_VT_TOP                 = -0x50L;    // vtable top_offset (base subobject adjust)
// generate_outputs is gated on this uint16 pending-update counter (generate_outputs
// entry: cmpw $0,0x192a; je return; and end: subw $1,0x192a). read_inputs credits it
// (addw $3,0x192a at e316a2) whenever it latches inputs. The bare internal_south poke
// skips read_inputs (to bypass the south disconnect guard) so it must replicate this
// credit itself, else generate_outputs no-ops and emits nothing.
constexpr unsigned long OFF_GEN_GATE      = 0x192aUL;  // generate_outputs pending counter
constexpr uint16_t GEN_GATE_CREDIT        = 3;         // matches read_inputs addw $3

const char* const kBareCtorSym =
    "_ZN14EventBroadcastC1EN8msm_core15msm_module_nameERNS0_9msm_clockERKN6me_cfg13"
    "EventTraceCfgERN8me_state4TileE";
const char* const kReqUpdateSym = "_ZN8msm_core16msm_prim_channel14request_updateEv";
// EventBroadcast::read_inputs(this): the model's own south/lateral input-processing
// method. We call it directly at the unwired shim->memtile seam (mode "readinputs",
// default) so the broadcast enters the array via the model's exact state-transition
// path -- see mirror_once.
const char* const kReadInputsSym = "_ZN14EventBroadcast11read_inputsEv";

// The bare ctor takes (this, msm_module_name [by hidden ref], msm_clock&,
// EventTraceCfg const&, me_state::Tile&) -- all five are pointer-sized register
// args, so a plain 5x void* wrapper forwards them verbatim.
using ctor_fn = void (*)(void*, void*, void*, void*, void*);
using update_fn = void (*)(void*);
using read_inputs_fn = void (*)(void*);

struct Lane {
    void* eb = nullptr;       // memtile EventBroadcast object
    void* south = nullptr;    // its SOUTH source msm_prim_channel (+0x3b8)
    void* shim_eb = nullptr;  // shim aie_sc::EventBroadcast (north_m broadcast value @ +0x1aa8)
    uint16_t last = 0;
    bool active = false;
};

ctor_fn g_real_ctor = nullptr;
update_fn g_request_update = nullptr;
read_inputs_fn g_read_inputs = nullptr;
unsigned long g_base = 0;
bool g_installed = false;
std::string g_prefix;                       // e.g. "aiesim_top.math_engine."
std::map<std::pair<int, char>, Lane> g_lanes;  // (col, net 'a'/'b') -> lane
std::map<std::string, void*> g_all_ebs;     // every bare EB by full name (wiring diagnostic)

// Vertical flood: deliver the broadcast up the whole array, one row per cycle, as on real
// HW. The array sub-model does NOT cascade internally (every inter-tile seam is dropped in
// this partial cluster -- compute EBs stay dormant), so each compute EB is driven directly.
// Each compute EB's SOUTH input channel (EB+0x3b8) is already wired to the tile-below's
// output (the model's authoritative topology -- no cm/mm guessing); we read the real
// broadcast value on that wire (channel cur_val +0xa) and complete the propagation the
// dormant EB drops. Because the read is of the previously-committed wire value, the
// broadcast ripples up one row per posedge -- matching real-HW propagation timing.
struct VLane { void* eb; void* south; uint16_t last; std::string name; };
std::vector<VLane> g_vlanes;

bool enabled() {
    const char* e = std::getenv("XDNA_AIESIM_BCAST_BRIDGE");
    return e && *e && std::strcmp(e, "0") != 0;  // default OFF (opt-in)
}

// Parse "<prefix>mem_row.tile_<col>_1.event_broadcast_<net>". Returns false for
// any other object (shim/compute/non-broadcast). prefix is everything up to the
// "mem_row" token (the SystemC hierarchy root for sibling lookups).
bool parse_memtile(const char* nm, int& col, char& net, std::string& prefix) {
    const char* p = std::strstr(nm, "mem_row.tile_");
    if (!p) return false;
    const char* tile = p + std::strlen("mem_row.tile_");
    int c = 0, r = 0;
    if (std::sscanf(tile, "%d_%d", &c, &r) != 2 || r != 1) return false;  // memtile row == 1
    const char* eb = std::strstr(tile, "event_broadcast_");
    if (!eb) return false;
    char nc = eb[std::strlen("event_broadcast_")];
    if (nc != 'a' && nc != 'b') return false;
    col = c;
    net = nc;
    prefix.assign(nm, static_cast<size_t>(p - nm));
    return true;
}

// GOT-interposed bare ctor: construct for real, then record memtiles by name.
extern "C" void wrap_bare_ctor(void* thisp, void* name, void* clk, void* cfg, void* tile) {
    g_real_ctor(thisp, name, clk, cfg, tile);
    if (!name) return;
    const char* nm = *reinterpret_cast<const char* const*>(static_cast<const char*>(name) + 8);
    if (!nm) return;
    // Log EVERY bare-EB ctor name once (gated) -- to see whether compute-tile EBs
    // (embedded in CoreModule at +0x106b0) are constructed/hooked at all.
    if (std::getenv("XDNA_AIESIM_BCAST_CTORLOG"))
        std::fprintf(stderr, "[bcast-ctor] EB @ %p name='%s'\n", thisp, nm);
    // Record every bare EB by name (for the array-wiring diagnostic). Only event
    // broadcast objects matter; skip the rest to keep the map small.
    if (std::strstr(nm, "event_broadcast")) g_all_ebs[nm] = thisp;
    int col;
    char net;
    std::string prefix;
    if (!parse_memtile(nm, col, net, prefix)) return;
    if (g_prefix.empty()) g_prefix = prefix;
    g_lanes[{col, net}].eb = thisp;
}

// The bare EventBroadcast's 4 directional input-source pointer fields and their
// "outer" connection-gate fields (read_inputs e3156b-e315c5 etc.). The inner
// pointer is the msm_prim_channel a neighbor would drive; the outer's +9 byte is
// the model's disconnected guard (read_inputs skips the read when it is nonzero).
// Direction labels are by the internal channel each feeds (generate_outputs).
struct Dir { unsigned long inner_off; unsigned long outer_off; unsigned long internal_off; const char* label; };
constexpr Dir kDirs[] = {
    {0x338, 0x838, 0x1648, "in0(->0x1648)"},
    {0x3b8, 0x6b8, 0x1700, "SOUTH(->0x1700)"},
    {0x2b8, 0x738, 0x1590, "in2(->0x1590)"},
    {0x438, 0x7b8, 0x17b8, "in3(->0x17b8)"},
};

// Report, per captured EB, whether each directional input points INTO another EB
// (wired) or stands alone (placeholder). Decides plan (A): bridge every hop vs fix
// one cascade bug. Read-only; gated by XDNA_AIESIM_BCAST_WIRING.
void diagnose_wiring() {
    constexpr unsigned long EB_SPAN = 0x2000;  // EB object footprint (inputs/outputs within)
    std::fprintf(stderr, "[bcast-wire] %zu bare EBs captured. Checking input-source containment.\n",
                 g_all_ebs.size());
    for (auto& kv : g_all_ebs) {
        const std::string& nm = kv.first;
        char* eb = static_cast<char*>(kv.second);
        // Short name tail (drop the long prefix) for readability.
        const char* tail = nm.c_str();
        if (const char* p = std::strstr(tail, "tile_")) tail = p;
        for (const Dir& d : kDirs) {
            void* inner = *reinterpret_cast<void**>(eb + d.inner_off);
            void* outer = *reinterpret_cast<void**>(eb + d.outer_off);
            unsigned guard = outer ? *reinterpret_cast<unsigned char*>(static_cast<char*>(outer) + 9) : 0xFFu;
            // Which EB (if any) does inner point into?
            const char* hit = "STANDALONE(placeholder)";
            std::string hitbuf;
            if (inner) {
                for (auto& kv2 : g_all_ebs) {
                    char* eb2 = static_cast<char*>(kv2.second);
                    if (eb2 == eb) continue;
                    unsigned long delta = static_cast<unsigned long>(static_cast<char*>(inner) - eb2);
                    if (delta < EB_SPAN) {
                        const char* t2 = kv2.first.c_str();
                        if (const char* p = std::strstr(t2, "tile_")) t2 = p;
                        char tmp[160];
                        std::snprintf(tmp, sizeof(tmp), "-> %s +0x%lx (WIRED)", t2, delta);
                        hitbuf = tmp;
                        hit = hitbuf.c_str();
                        break;
                    }
                }
            } else {
                hit = "NULL";
            }
            // Consumer-list count of the channel `inner` points to. This is the list
            // sync_update walks to wake processes when the channel changes -- i.e. the
            // set of read_inputs/etc. registered to be woken. A WIRED inter-tile channel
            // with an EMPTY list means the neighbor's read_inputs is NOT registered, so
            // request_update won't cascade (the wake gap we are hunting).
            // Consumer list lives on the msm_prim_channel BASE subobject. `inner` may
            // point to a derived start; the base is inner + vtable_top_offset(-0x50).
            // Report consumer count at BOTH inner and the top_offset-adjusted base.
            auto cons_at = [](char* ch) -> long {
                char* cb = *reinterpret_cast<char**>(ch + 0x28);
                char* ce = *reinterpret_cast<char**>(ch + 0x30);
                long span = ce - cb;
                return (span >= 0 && span < 0x100000 && (span % 0x10) == 0) ? span / 0x10 : -2;
            };
            long cons0 = -1, consB = -1, topoff = 0;
            if (inner) {
                char* ch = static_cast<char*>(inner);
                cons0 = cons_at(ch);
                char* vt = *reinterpret_cast<char**>(ch);           // vtable of inner
                topoff = *reinterpret_cast<long*>(vt + OFF_VT_TOP);  // -0x50
                consB = cons_at(ch + topoff);
            }
            std::fprintf(stderr,
                "[bcast-wire]   %-28s %-16s inner=%p guard=0x%02x cons0=%ld topoff=0x%lx consB=%ld  %s\n",
                tail, d.label, inner, guard, cons0, topoff, consB, hit);
        }
    }
}

}  // namespace

void install_bcast_bridge(void* cluster_lib, const char* arch) {
    if (!enabled()) return;
    const std::string arch_s = arch ? arch : "";
    if (arch_s != "aie2") {
        std::fprintf(stderr, "[bcast-bridge] arch '%s' != aie2; not installed\n", arch_s.c_str());
        return;
    }
    void* sym = dlsym(cluster_lib, kBareCtorSym);
    if (!sym) sym = dlsym(RTLD_DEFAULT, kBareCtorSym);
    if (!sym) {
        std::fprintf(stderr, "[bcast-bridge] WARN: bare ctor symbol not found; not installed\n");
        return;
    }
    g_base = reinterpret_cast<unsigned long>(sym) - RVA_BARE_CTOR;
    g_real_ctor = reinterpret_cast<ctor_fn>(sym);

    void** got = reinterpret_cast<void**>(g_base + RVA_GOT_CTOR);
    // The slot must currently hold either the resolved ctor or a PLT resolver
    // stub inside the lib's mapping; if it points wildly elsewhere the RVA is
    // stale -> refuse.
    const unsigned long cur = reinterpret_cast<unsigned long>(*got);
    if (cur != reinterpret_cast<unsigned long>(sym) &&
        (cur < g_base || cur > g_base + 0x4000000UL)) {
        std::fprintf(stderr,
                     "[bcast-bridge] WARN: GOT slot 0x%lx holds 0x%lx (not ctor/PLT in lib); "
                     "RVA stale? not installed\n", g_base + RVA_GOT_CTOR, cur);
        return;
    }
    const long pg = sysconf(_SC_PAGESIZE);
    void* page = reinterpret_cast<void*>(reinterpret_cast<unsigned long>(got) & ~(unsigned long)(pg - 1));
    if (mprotect(page, pg, PROT_READ | PROT_WRITE) != 0) {
        std::fprintf(stderr, "[bcast-bridge] WARN: mprotect(GOT) failed; not installed\n");
        return;
    }
    *got = reinterpret_cast<void*>(&wrap_bare_ctor);
    g_installed = true;
    std::fprintf(stderr,
                 "[bcast-bridge] ctor capture installed (base=0x%lx). Capturing memtile "
                 "EventBroadcast objects during elaboration.\n", g_base);
}

void bcast_bridge_resolve() {
    if (!g_installed) return;
    if (g_lanes.empty()) {
        std::fprintf(stderr, "[bcast-bridge] WARN: no memtile EventBroadcast captured; disabled\n");
        g_installed = false;
        return;
    }
    g_request_update = reinterpret_cast<update_fn>(dlsym(RTLD_DEFAULT, kReqUpdateSym));
    if (!g_request_update) {
        std::fprintf(stderr, "[bcast-bridge] WARN: request_update unresolved; commit relies on "
                             "direct cur_val write only\n");
    }
    g_read_inputs = reinterpret_cast<read_inputs_fn>(dlsym(RTLD_DEFAULT, kReadInputsSym));
    if (!g_read_inputs) {
        std::fprintf(stderr, "[bcast-bridge] WARN: read_inputs unresolved; faithful 'readinputs' "
                             "inject mode unavailable (falls back to internal)\n");
    }
    const unsigned long expect_disc = g_base + RVA_DISC_SRC;
    int active = 0;
    for (auto& kv : g_lanes) {
        const int col = kv.first.first;
        const char net = kv.first.second;
        Lane& lane = kv.second;
        if (!lane.eb) continue;

        void* south = *reinterpret_cast<void**>(static_cast<char*>(lane.eb) + OFF_SOUTH);
        if (!south) {
            std::fprintf(stderr, "[bcast-bridge] WARN: col%d net%c south descriptor null; skipped\n",
                         col, net);
            continue;
        }
        // Verify the descriptor is a source channel (vtable[0x18] discriminator).
        void* vt = *reinterpret_cast<void**>(south);
        void* disc = vt ? *reinterpret_cast<void**>(static_cast<char*>(vt) + 0x18) : nullptr;
        if (reinterpret_cast<unsigned long>(disc) != expect_disc) {
            std::fprintf(stderr,
                         "[bcast-bridge] WARN: col%d net%c south disc 0x%lx != 0x%lx (layout "
                         "changed?); skipped\n", col, net, reinterpret_cast<unsigned long>(disc),
                         expect_disc);
            continue;
        }
        lane.south = south;

        // Shim aie_sc::EventBroadcast object. It is an sc_module, so sc_find_object
        // returns its sc_object subobject at delta 0 == the object base; we read the
        // broadcast value at +0x1ea8 directly (no dynamic_cast -- see OFF_SHIM_BCAST).
        const std::string name = g_prefix + "shim.tile_" + std::to_string(col) +
                                 "_0.event_broadcast_" + net;
        sc_core::sc_object* obj = sc_core::sc_find_object(name.c_str());
        if (!obj) {
            std::fprintf(stderr, "[bcast-bridge] WARN: shim EventBroadcast '%s' not found; "
                                 "skipped\n", name.c_str());
            continue;
        }
        lane.shim_eb = obj;
        lane.active = true;
        ++active;
        if (std::getenv("XDNA_AIESIM_BCAST_DEBUG")) {
            std::fprintf(stderr, "[bcast-dbg] lane col%d net%c: south=%p eb=%p shim_eb=%p\n",
                         col, net, lane.south, lane.eb, lane.shim_eb);
        }
    }
    std::fprintf(stderr, "[bcast-bridge] resolved %d/%zu broadcast lanes (shim north_m -> memtile "
                         "south).\n", active, g_lanes.size());
    if (active == 0) g_installed = false;

    // Plan (A) step 1 diagnostic: is the ARRAY-INTERNAL broadcast network wired
    // (adjacent tiles share channel objects) or unwired (separate placeholders)?
    // For each captured bare EB, dump its 4 directional input-source pointers and
    // the model's own "disconnected" guard flags, and report whether each input
    // points INTO another EB's [base, base+0x2000) range (== wired to that
    // neighbor) or stands alone (== placeholder, unwired).
    if (std::getenv("XDNA_AIESIM_BCAST_WIRING")) diagnose_wiring();

    // Build the vertical-flood lanes: every COMPUTE-tile bare EB (array.tile_*.{cm,mm}.
    // event_broadcast) whose SOUTH input channel is a valid wired source. We drive each one
    // from the real value on its south wire each posedge so the broadcast ripples up the
    // array. The memtile row is already handled by g_lanes (memtile<-shim); these vlanes
    // carry it the rest of the way up every column. XDNA_AIESIM_BCAST_VFLOOD=0 disables
    // (origin-only, for A/B); a non-0/1 value restricts to EB names containing that substring.
    const char* vf = std::getenv("XDNA_AIESIM_BCAST_VFLOOD");
    const bool vflood = !vf || (std::strcmp(vf, "0") != 0);
    const char* vrestrict = (vf && *vf && std::strcmp(vf, "0") != 0 && std::strcmp(vf, "1") != 0) ? vf : nullptr;
    if (vflood) {
        for (auto& kv : g_all_ebs) {
            const std::string& nm = kv.first;
            if (nm.find("array.tile_") == std::string::npos) continue;  // compute tiles only
            if (vrestrict && !std::strstr(nm.c_str(), vrestrict)) continue;
            char* eb = static_cast<char*>(kv.second);
            void* south = *reinterpret_cast<void**>(eb + OFF_SOUTH);
            if (!south) continue;
            // Verify south is a real source channel (vtable[0x18] discriminator), exactly as
            // the memtile-lane resolve does -- guards against an unwired/placeholder south.
            void* vt = *reinterpret_cast<void**>(south);
            void* disc = vt ? *reinterpret_cast<void**>(static_cast<char*>(vt) + 0x18) : nullptr;
            if (reinterpret_cast<unsigned long>(disc) != expect_disc) continue;
            const char* tail = nm.c_str();
            if (const char* p = std::strstr(tail, "tile_")) tail = p;
            g_vlanes.push_back({kv.second, south, 0, std::string(tail)});
        }
        std::fprintf(stderr, "[bcast-bridge] vertical flood: %zu compute EB lanes (south-wire "
                             "driven, ripples up the array).\n", g_vlanes.size());
    }
}

namespace {
// Drive value v into an EB's internal_south exactly as read_inputs' south tail does, then
// credit the generate_outputs gate (as read_inputs' addw $3) and request_update the channel
// base. This is the one faithful primitive used for every seam (shim->memtile and the
// vertical compute lanes): the model's own generate_outputs then floods all outputs.
inline void inject_south(char* eb, uint16_t v) {
    *reinterpret_cast<volatile uint16_t*>(eb + OFF_INT_SOUTH_NEW) = v;
    *reinterpret_cast<volatile uint16_t*>(eb + OFF_GEN_GATE) += GEN_GATE_CREDIT;
    char* ich = eb + OFF_INT_SOUTH;
    const long topoff = *reinterpret_cast<long*>(*reinterpret_cast<char**>(ich) + OFF_VT_TOP);
    g_request_update(ich + topoff);
}

// One mirror step, run every clock posedge: for each active lane, if the shim's
// north broadcast output changed, drive it into the memtile's south placeholder
// channel exactly as a producing neighbor's generate_outputs would.
void mirror_once() {
    // Inject mode (env XDNA_AIESIM_BCAST_INJECT):
    //   "flood"        -> internal_south poke + request_update + credit 0x192a  [THE FIX, default]
    //   "readinputs"   -> write south placeholder cur_val(+0xa), then CALL read_inputs
    //                     [DOES NOT WORK: read_inputs honors the south disconnect guard
    //                      (e31582) and skips our injected south -- kept for the record]
    //   "internal"     -> poke the EB's internal_south + request_update only  [partial; see below]
    //   "cur"          -> write south placeholder cur_val(+0xa) only   [no propagation]
    //   "new"          -> write south placeholder new_val(+8) only     [diagnostic]
    //   "newcur"       -> write south placeholder +8 and +0xa          [diagnostic, no wake]
    //   "req"          -> write +0xa + request_update(placeholder)     [diagnostic: faults]
    //   "full"         -> +8, +0xa, request_update(placeholder)        [diagnostic: faults]
    //
    // Why "readinputs" is the faithful fix (see genout/readin disassembly study):
    // generate_outputs is GATED on a pending-update counter at EB+0x192a -- if it is 0,
    // generate_outputs early-returns and emits NOTHING. read_inputs is what credits that
    // counter (addw $0x3, 0x192a at e316a2) as it latches each input source and
    // request_updates the internal channels. The "internal" mode pokes internal_south +
    // request_update but never credits 0x192a, so generate_outputs no-ops (it only ran
    // opportunistically when the memtile's OWN DMA/lock events happened to credit the
    // counter -- flaky, and never a deterministic lateral flood). Calling the model's own
    // read_inputs runs the exact state transition: it reads our injected south cur_val,
    // credits 0x192a, schedules generate_outputs, which floods ALL five outputs (north +
    // lateral, e31886..e319a3) and request_updates each -> the fully-wired array cascades
    // on its own. We bridge only the one unwired shim->memtile seam; the model does the rest.
    //
    // "req"/"full" drive request_update on the EXTERNAL +0x3b8 placeholder, whose consumer
    // list is unwired in our partial cluster -> sync_update's consumer walk dereferences
    // garbage and SIGSEGVs. read_inputs does NOT request_update +0x3b8 (it only reads its
    // cur_val and request_updates the EB's own wired internal_south), so it is safe.
    static const char* mode_e = std::getenv("XDNA_AIESIM_BCAST_INJECT");
    static const std::string mode = mode_e ? mode_e : "flood";
    static const bool do_readinputs = (mode == "readinputs") && g_read_inputs;
    static const bool do_internal = (mode == "internal" || mode == "flood") && g_request_update;
    static const bool do_credit = (mode == "flood");
    static const bool do_new = (mode == "new" || mode == "newcur" || mode == "full");
    static const bool do_cur = (mode == "cur" || mode == "newcur" || mode == "req" || mode == "full"
                                || do_readinputs);
    static const bool do_req = (mode == "req" || mode == "full") && g_request_update;

    for (auto& kv : g_lanes) {
        Lane& lane = kv.second;
        if (!lane.active) continue;
        const uint16_t v =
            *reinterpret_cast<volatile uint16_t*>(static_cast<char*>(lane.shim_eb) + OFF_SHIM_BCAST);
        if (v == lane.last) continue;
        lane.last = v;

        if (do_internal) {
            // Replicate read_inputs()'s south tail + gate credit (see inject_south): write the
            // memtile EB's internal_south, credit 0x192a, request_update. generate_outputs then
            // floods all outputs and the broadcast climbs the column. ('internal' mode, kept for
            // A/B, skips the credit -- which only propagated opportunistically; 'flood' credits.)
            char* eb = static_cast<char*>(lane.eb);
            if (do_credit) {
                inject_south(eb, v);
            } else {
                *reinterpret_cast<volatile uint16_t*>(eb + OFF_INT_SOUTH_NEW) = v;
                char* ich = eb + OFF_INT_SOUTH;
                const long topoff = *reinterpret_cast<long*>(*reinterpret_cast<char**>(ich) + OFF_VT_TOP);
                g_request_update(ich + topoff);
            }
        }

        char* south = static_cast<char*>(lane.south);
        if (do_new) *reinterpret_cast<volatile uint16_t*>(south + OFF_NEWVAL) = v;
        if (do_cur) *reinterpret_cast<volatile uint16_t*>(south + OFF_CURVAL) = v;
        if (do_req) g_request_update(lane.south);

        // The faithful seam injection: south cur_val is now set (do_cur), so run the
        // model's own read_inputs on the memtile EB. It latches the south value into
        // internal_south, credits the 0x192a generate_outputs gate, and request_updates
        // the wired internal channels -> generate_outputs floods the array, which cascades
        // through the fully-wired inter-tile network with no further bridging.
        if (do_readinputs) g_read_inputs(lane.eb);
    }

    // Vertical flood: carry the broadcast up the array. Each compute EB reads the real
    // broadcast value on its own wired SOUTH input channel (the tile-below's output, set by
    // that tile's generate_outputs) and, when it changes, injects it into its own
    // internal_south. Because the read is of the previously-committed wire value, the
    // broadcast advances one row per posedge -- a faithful ripple matching HW propagation.
    if (do_credit) {  // only meaningful with the gate credit (flood)
        for (VLane& vl : g_vlanes) {
            const uint16_t v =
                *reinterpret_cast<volatile uint16_t*>(static_cast<char*>(vl.south) + OFF_CURVAL);
            if (v == vl.last) continue;
            vl.last = v;
            inject_south(static_cast<char*>(vl.eb), v);
        }
    }
}
}  // namespace

void bcast_bridge_spawn(sc_core::sc_clock& clk) {
    if (!g_installed) return;
    sc_core::sc_spawn_options opt;
    opt.spawn_method();
    opt.set_sensitivity(&clk.posedge_event());
    opt.dont_initialize();
    sc_core::sc_spawn([] { mirror_once(); }, "bcast_mirror", &opt);
    std::fprintf(stderr, "[bcast-bridge] ACTIVE: per-cycle shim->memtile broadcast mirror spawned. "
                         "Unset XDNA_AIESIM_BCAST_BRIDGE to disable.\n");
}

}  // namespace aiesim
