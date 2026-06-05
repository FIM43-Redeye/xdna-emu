// See cluster_clone_patch.h. Mechanism (proven in the ctrl-packet-debug probe):
// the cluster lib has no -Bsymbolic and calls the deque-push helper ce2a40 from
// send_response via a direct `call rel32`. We overwrite each of those 3 call
// sites to jump to a small trampoline that invokes push_dispatch(), which clones
// the beat into a fresh control block and then performs the real push. The
// trampoline fully emulates the original `call ce2a40` (push_dispatch does the
// push and returns), so control flow is transparent to send_response.
//
// Control block (MEStreamData32 _Sp_counted_ptr_inplace) layout, from the
// send_response allocation disasm (_Znwm(0x18) @d9f747):
//   +0x00 vtable  +0x08 use_count  +0x0c weak_count  +0x10 value(u32)  +0x14 tlast
// shared_ptr = { _M_ptr = block+0x10, _M_pi = block }. ce2a40 copies
// {_M_ptr,_M_pi} into the deque and atomically increments _M_pi->use_count.
#include "cluster_clone_patch.h"

#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

namespace aiesim {
namespace {

// RVAs within libaie2_cluster_msm_v1_0_0.osci.so (objdump file offsets). Specific
// to this aie2 cluster build; the verify step below refuses to patch if they do
// not decode as expected (e.g. an updated aietools or a different arch).
constexpr unsigned long RVA_SEND_RESPONSE = 0xd9f720UL;
constexpr unsigned long RVA_CE2A40 = 0xce2a40UL;
constexpr unsigned long kSiteRva[3] = {0xd9f83aUL, 0xd9f885UL, 0xd9f8d3UL};
const char* const kSendRespSym =
    "_ZN11TileControl13send_responseERKNS_8ResponseERN5boost11coroutines26detail14"
    "push_coroutineIvEE";

using ce2a40_fn = void (*)(void*, void*, void*);
ce2a40_fn g_real_ce2a40 = nullptr;  // the genuine deque-push helper

// Replaces the `call ce2a40` at each push site: clone the beat into a fresh
// distinct control block, then push that. Args are ce2a40's after the trampoline
// shuffle: deque, sp (&shared_ptr), sink. `site` is unused here (kept for parity
// with the diagnostic probe / future logging).
extern "C" void clone_push_dispatch(unsigned long /*site*/, void* deque, void** sp,
                                     void* sink) {
    void* mptr = sp ? sp[0] : nullptr;  // _M_ptr -> block+0x10 (value word)
    if (!mptr) {
        g_real_ce2a40(deque, sp, sink);
        return;
    }
    void* block = static_cast<char*>(mptr) - 0x10;          // control block base
    void* vtbl = *reinterpret_cast<void**>(block);          // +0x00 vtable
    unsigned val = *static_cast<unsigned*>(mptr);           // +0x10 value
    unsigned char tlast = *(static_cast<unsigned char*>(mptr) + 4);  // +0x14 tlast

    void* fresh = ::operator new(0x18);  // same global operator new the cluster uses
    *reinterpret_cast<void**>(fresh) = vtbl;
    *reinterpret_cast<unsigned*>(static_cast<char*>(fresh) + 0x08) = 1u;  // use_count
    *reinterpret_cast<unsigned*>(static_cast<char*>(fresh) + 0x0c) = 1u;  // weak_count
    *reinterpret_cast<unsigned*>(static_cast<char*>(fresh) + 0x10) = val;
    *(static_cast<unsigned char*>(fresh) + 0x14) = tlast;

    void* fresh_sp[2];
    fresh_sp[0] = static_cast<char*>(fresh) + 0x10;  // _M_ptr
    fresh_sp[1] = fresh;                              // _M_pi
    g_real_ce2a40(deque, fresh_sp, sink);            // deque copies it -> use_count 2
    // drop our construction ref; the deque holds the remaining one (use_count -> 1)
    __atomic_sub_fetch(reinterpret_cast<int*>(static_cast<char*>(fresh) + 0x08), 1,
                       __ATOMIC_ACQ_REL);
}

// Emit a trampoline at `t`: 16B-align, shuffle (deque,&sp,sink) into the SysV arg
// slots after a leading `site` arg, call clone_push_dispatch, ret. Returns bytes.
int emit_trampoline(unsigned char* t, unsigned long site, unsigned long dispatch) {
    unsigned char* p = t;
    std::memcpy(p, "\x48\x83\xec\x08", 4); p += 4;  // sub $8,%rsp   (align)
    std::memcpy(p, "\x48\x89\xd1", 3); p += 3;       // mov %rdx,%rcx (sink->a4)
    std::memcpy(p, "\x48\x89\xf2", 3); p += 3;       // mov %rsi,%rdx (&sp->a3)
    std::memcpy(p, "\x48\x89\xfe", 3); p += 3;       // mov %rdi,%rsi (deque->a2)
    *p++ = 0xbf; *reinterpret_cast<unsigned*>(p) = static_cast<unsigned>(site); p += 4;  // mov $site,%edi
    *p++ = 0x48; *p++ = 0xb8; *reinterpret_cast<unsigned long*>(p) = dispatch; p += 8;   // movabs rax,dispatch
    std::memcpy(p, "\xff\xd0", 2); p += 2;           // call *%rax
    std::memcpy(p, "\x48\x83\xc4\x08", 4); p += 4;   // add $8,%rsp
    *p++ = 0xc3;                                       // ret
    return static_cast<int>(p - t);
}

bool clone_enabled() {
    const char* e = std::getenv("XDNA_AIESIM_CLONE");
    return !e || std::strcmp(e, "0") != 0;  // default ON
}

}  // namespace

void install_clone_patch(void* cluster_lib, const char* arch) {
    if (!clone_enabled()) {
        std::fprintf(stderr,
                     "[aiesim-clone] disabled (XDNA_AIESIM_CLONE=0); control-packet "
                     "read-responses will hang on this cluster model\n");
        return;
    }
    const std::string arch_s = arch ? arch : "";
    if (arch_s != "aie2") {
        // RVAs are specific to libaie2_cluster_msm; other arches are not patched.
        std::fprintf(stderr, "[aiesim-clone] arch '%s' != aie2; clone patch not applied\n",
                     arch_s.c_str());
        return;
    }

    void* sr = dlsym(cluster_lib, kSendRespSym);
    if (!sr) sr = dlsym(RTLD_DEFAULT, kSendRespSym);
    if (!sr) {
        std::fprintf(stderr, "[aiesim-clone] WARN: send_response not found; not patched\n");
        return;
    }
    const unsigned long base = reinterpret_cast<unsigned long>(sr) - RVA_SEND_RESPONSE;
    const unsigned long ce2a40_abs = base + RVA_CE2A40;

    // Verify each site is `e8 rel32` decoding to ce2a40 (refuse a moving target).
    unsigned long site[3];
    for (int i = 0; i < 3; ++i) {
        site[i] = base + kSiteRva[i];
        const unsigned char* s = reinterpret_cast<const unsigned char*>(site[i]);
        if (s[0] != 0xe8) {
            std::fprintf(stderr,
                         "[aiesim-clone] WARN: site%d byte0=0x%02x != e8 (lib changed?); "
                         "not patched\n", i, s[0]);
            return;
        }
        const long rel = *reinterpret_cast<const int*>(s + 1);
        if (site[i] + 5 + rel != ce2a40_abs) {
            std::fprintf(stderr,
                         "[aiesim-clone] WARN: site%d call target mismatch (lib changed?); "
                         "not patched\n", i);
            return;
        }
    }

    // RWX trampoline page, hinted near base so the e8 rel32 from each site reaches.
    void* page = mmap(reinterpret_cast<void*>(base & ~0xFFFFFUL), 4096,
                      PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) {
        std::fprintf(stderr, "[aiesim-clone] WARN: trampoline mmap failed; not patched\n");
        return;
    }

    const long pgsz = sysconf(_SC_PAGESIZE);
    const unsigned long lo = site[0] & ~static_cast<unsigned long>(pgsz - 1);
    const unsigned long hi = (site[2] + 8 + pgsz - 1) & ~static_cast<unsigned long>(pgsz - 1);
    if (mprotect(reinterpret_cast<void*>(lo), hi - lo,
                 PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
        std::fprintf(stderr, "[aiesim-clone] WARN: mprotect(.text) failed; not patched\n");
        return;
    }

    g_real_ce2a40 = reinterpret_cast<ce2a40_fn>(ce2a40_abs);
    const unsigned long dispatch = reinterpret_cast<unsigned long>(&clone_push_dispatch);

    unsigned char* tp = reinterpret_cast<unsigned char*>(page);
    for (int i = 0; i < 3; ++i) {
        unsigned char* t = tp;
        tp += emit_trampoline(t, static_cast<unsigned long>(i), dispatch);
        const long d = reinterpret_cast<long>(t) - static_cast<long>(site[i] + 5);
        if (d > 0x7ffffff0L || d < -0x7ffffff0L) {
            std::fprintf(stderr,
                         "[aiesim-clone] WARN: site%d trampoline out of rel32 range; "
                         "not patched\n", i);
            return;
        }
        unsigned char* s = reinterpret_cast<unsigned char*>(site[i]);
        s[0] = 0xe8;
        *reinterpret_cast<int*>(s + 1) = static_cast<int>(d);  // redirect e8 rel32 -> trampoline
    }
    mprotect(reinterpret_cast<void*>(lo), hi - lo, PROT_READ | PROT_EXEC);
    __builtin___clear_cache(reinterpret_cast<char*>(page), reinterpret_cast<char*>(tp));

    std::fprintf(stderr,
                 "[aiesim-clone] ACTIVE: distinct-object-per-beat patch installed on "
                 "send_response (3 sites, base=0x%lx) -- control-packet read-responses "
                 "will route. Set XDNA_AIESIM_CLONE=0 to disable.\n", base);
}

}  // namespace aiesim
