// Shim->memtile event-broadcast bridge for the AIE2 MSM cluster model.
//
// WHY THIS EXISTS
// The cluster model is two simulation worlds: the shim row is aie_sc::* (pure
// SystemC, sc_signal transport) and the array (memtile + compute rows) is the
// MSM cycle model (msm_prim_channel transport). The event-broadcast network is
// physically split at the shim<->memtile boundary -- streams cross via explicit
// adaptor objects, but the broadcast network has NO adaptor. AMD's design leaves
// cross-world event injection to the integrating harness (that is what the
// exported Array::event_broadcast_write API is for). The consequence for us:
// trace-start, which on AIE2 floods broadcast-15 from the shim up into the array
// (Trace_Control0.Trace_Start_Event = BROADCAST_15), never reaches the array, so
// the array tiles' trace FSMs never start. aiesim cannot oracle trace for any
// broadcast-start kernel (e.g. distribute_lateral). See docs/coverage/
// aiesim-failure-triage.md and task #93/#96/#97.
//
// WHAT IT DOES (faithful mirror, not fabrication)
// Each column's memtile EventBroadcast reads its SOUTH input from a standalone
// msm_prim_channel placeholder (the slot a real south neighbor would drive) that
// the shim, living in the sc_signal world, can never write. We supply exactly
// that missing write: each cycle we read the shim's real north broadcast output
// (event_broadcast_a/b.north_m, an sc_signal<uint16>) and drive it into the
// memtile's south placeholder channel the same way a producing neighbor's
// generate_outputs would -- write new_val(+8), request_update() (commits +8->+0xa
// and wakes the memtile's read_inputs). The array then floods onward through its
// own uniform MSM network. One-cycle latency = one broadcast hop = faithful.
//
// Nothing is fabricated: the value is the shim's actual hardware output, on the
// model's own channel, at the boundary the model left for the harness to close.
//
// MECHANISM DETAILS (libaie2_cluster_msm_v1_0_0.osci.so, this aietools build)
//   - Memtile EventBroadcast objects are captured by GOT-interposing the bare
//     EventBroadcast ctor (JUMP_SLOT @ GOT RVA 0x331c0a0; ctor RVA 0xe33660),
//     matching object names "*.mem_row.tile_C_1.event_broadcast_{a,b}".
//   - SOUTH source descriptor is at this+0x3b8 (structural for all bare tiles;
//     proven: a compute tile's +0x3b8 points into its memtile south neighbor).
//   - msm_prim_channel is double-buffered: new_val@+8 (producer writes), cur_val
//     @+0xa (consumer reads); request_update commits new->cur and notifies.
//   - Shim north_m sc_signals are found via sc_core::sc_find_object by name
//     (the shim is a real sc_module, so it IS in the SystemC hierarchy).
//
// LOCAL-ONLY, gated, fail-safe. Off unless XDNA_AIESIM_BCAST_BRIDGE is set; every
// RVA/offset is sanity-checked and the bridge self-disables on any mismatch (an
// updated aietools or a different arch), exactly like cluster_clone_patch. Never
// shipped; the proprietary cluster lib stays untouched on disk.
#pragma once

namespace sc_core {
class sc_clock;
}

namespace aiesim {

// Install the GOT-interpose ctor capture. Call AFTER dlopen of the cluster lib
// and BEFORE create_math_engine (so it catches the memtile EventBroadcast ctors
// during elaboration). No-op unless the bridge is enabled and arch == aie2.
void install_bcast_bridge(void* cluster_lib, const char* arch);

// Resolve the per-lane endpoints: the memtile SOUTH source channel (captured
// EventBroadcast + 0x3b8) and the shim north_m sc_signal (sc_find_object). Call
// from end_of_elaboration (descriptors are wired, shim ports bound). Verifies
// channel type and self-disables on mismatch.
void bcast_bridge_resolve();

// Spawn the per-cycle mirror driver (clock-posedge method). Call from sc_main, a
// simulation context where sc_spawn is legal (alongside spawn_egress_drains).
void bcast_bridge_spawn(sc_core::sc_clock& clk);

}  // namespace aiesim
