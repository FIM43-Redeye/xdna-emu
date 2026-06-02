#include "aiesim_top.h"

#include <dlfcn.h>

#include <stdexcept>
#include <string>

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
    : sc_core::sc_module(name) {
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

    // Socket binding (ps_bridge -> me's aximm rd/wr + shim DMA) is Task II.4.
}

aiesim_top::~aiesim_top() {
    if (cluster_lib_) {
        dlclose(cluster_lib_);
    }
}
