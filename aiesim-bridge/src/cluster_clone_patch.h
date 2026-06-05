// Distinct-object-per-beat clone patch for the aiesim cluster model.
//
// libaie2_cluster_msm_v1_0_0.osci.so's TileControl::send_response reuses ONE
// inplace MEStreamData32 control block for every beat of a control-packet
// response, writing each beat's value into the same object and pushing aliased
// shared_ptrs. For a multi-beat READ-RESPONSE the data beats overwrite the id=2
// routing header before the stream-switch router can read it, so the response
// never routes and the shim S2MM starves (the read hangs). This is a genuine
// aiesim model bug -- real silicon streams distinct flits. Full root-cause +
// proof: build/experiments/ctrl-packet-debug/FINDINGS.md (cont.26-27).
//
// The fix: at each send_response push site, clone the beat into a FRESH control
// block holding that beat's value, so each enqueued shared_ptr is distinct and
// the router reads the header correctly. This makes aiesim MATCH hardware.
//
// LOCAL-ONLY: a runtime binary patch over a proprietary .so. Never shipped.
#pragma once

namespace aiesim {

// Install the clone patch over the cluster lib's send_response push sites.
//   cluster_lib : the dlopen handle for the cluster .so (already RTLD_GLOBAL).
//   arch        : must be "aie2" -- the patch RVAs are specific to
//                 libaie2_cluster_msm. Other arches no-op.
// Fail-safe: each push site is verified (e8 rel32 -> ce2a40) before patching;
// any mismatch (wrong/updated lib) logs a warning and skips, leaving the cluster
// unpatched. Gated by XDNA_AIESIM_CLONE (default "1"; set "0" to disable).
// Announces loudly on stderr when it patches -- never a silent binary edit.
void install_clone_patch(void* cluster_lib, const char* arch);

}  // namespace aiesim
