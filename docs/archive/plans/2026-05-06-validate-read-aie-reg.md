# validate-readback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a throwaway C++ binary that probes whether `xrt::hw_context::read_aie_reg` returns real, wall-time-correlated, round-trip-honest data on Phoenix NPU1, and verifies the bridge-runner pre-launch lifecycle bug + same-hwctx workaround.

**Architecture:** Single-file C++17 binary using XRT C++ API directly. Loads a known xclbin (default: peano build of `add_one_using_dma`), runs a sequence of named tests, prints PASS/FAIL with raw values per test, returns count of failures as exit code. Tests share one hwctx and one kernel handle; each test reuses or re-runs the kernel as needed.

**Tech Stack:** C++17, CMake 3.20+, XRT (`xrt_coreutil`, headers from `/opt/xilinx/xrt/include`), Linux. Builds standalone — no dependency on the xdna-emu Rust code or the bridge-runner.

**Spec:** `docs/superpowers/specs/2026-05-06-validate-read-aie-reg-design.md`

---

## File Structure

```
tools/validate-readback/
├── CMakeLists.txt          ~15 lines, mirrors bridge-runner pattern
├── validate-readback.cpp   ~250 LOC single-file binary
└── README.md               short, what each test means
```

No test files — the *binary itself* is the test, and HW behavior is the verification. Each task ends with a HW run that confirms the test code is well-formed and produces interpretable output.

---

## Task 1: Scaffold + minimal main that loads the xclbin

**Files:**
- Create: `tools/validate-readback/CMakeLists.txt`
- Create: `tools/validate-readback/validate-readback.cpp`

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(validate-readback CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT install root")
include_directories(SYSTEM "${XRT_ROOT}/include")
link_directories("${XRT_ROOT}/lib")

add_executable(validate-readback validate-readback.cpp)
target_link_libraries(validate-readback PRIVATE xrt_coreutil uuid)
target_compile_options(validate-readback PRIVATE -Wall -Wextra)
```

- [ ] **Step 2: Create skeleton validate-readback.cpp**

```cpp
// validate-readback: probe xrt::hw_context::read_aie_reg for ground-truth honesty.
// Throwaway harness; see docs/superpowers/specs/2026-05-06-validate-read-aie-reg-design.md.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_xclbin.h"

namespace {

constexpr uint32_t TIMER_LOW_OFFSET       = 0x000340F8;
constexpr uint32_t PERF_CTRL0_OFFSET      = 0x00031500;
constexpr uint32_t PERF_COUNTER0_OFFSET   = 0x00031520;
constexpr uint8_t  EVENT_ACTIVE_CORE      = 0x1C;

constexpr const char* DEFAULT_XCLBIN =
    "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/peano/aie.xclbin";
constexpr const char* DEFAULT_INSTS =
    "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/peano/insts.bin";

struct Args {
    std::string xclbin = DEFAULT_XCLBIN;
    std::string insts  = DEFAULT_INSTS;
    int col = 0;
    int row = 2;
    bool verbose = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--xclbin" && i + 1 < argc) a.xclbin = argv[++i];
        else if (s == "--insts" && i + 1 < argc) a.insts = argv[++i];
        else if (s == "--col" && i + 1 < argc) a.col = std::atoi(argv[++i]);
        else if (s == "--row" && i + 1 < argc) a.row = std::atoi(argv[++i]);
        else if (s == "-v" || s == "--verbose") a.verbose = true;
        else {
            std::fprintf(stderr, "unknown arg: %s\n", s.c_str());
            std::exit(2);
        }
    }
    return a;
}

std::vector<uint32_t> load_insts(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open insts: " + path);
    auto size = f.tellg();
    f.seekg(0);
    std::vector<uint32_t> v(size / 4);
    f.read(reinterpret_cast<char*>(v.data()), size);
    return v;
}

} // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::printf("validate-readback: xclbin=%s col=%d row=%d\n",
                args.xclbin.c_str(), args.col, args.row);

    auto device = xrt::device(0);
    auto xclbin = xrt::xclbin(args.xclbin);
    device.register_xclbin(xclbin);
    auto ctx = xrt::hw_context(device, xclbin.get_uuid());
    auto kernels = xclbin.get_kernels();
    if (kernels.empty()) throw std::runtime_error("no kernels in xclbin");
    auto kernel = xrt::kernel(ctx, kernels[0].get_name());

    std::printf("[INFO] loaded xclbin, kernel=%s\n", kernels[0].get_name().c_str());
    return 0;
}
```

- [ ] **Step 3: Build**

Run from xdna-emu root:
```bash
cmake -S tools/validate-readback -B tools/validate-readback/build
cmake --build tools/validate-readback/build
```

Expected: `validate-readback` executable produced; no compile errors.

- [ ] **Step 4: Run, confirm load works**

```bash
./tools/validate-readback/build/validate-readback
```

Expected output:
```
validate-readback: xclbin=/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/peano/aie.xclbin col=0 row=2
[INFO] loaded xclbin, kernel=MLIR_AIE_<...>
```

If load fails (e.g., NPU not alive), abort plan and recover NPU before proceeding.

- [ ] **Step 5: Commit**

```bash
git add tools/validate-readback/CMakeLists.txt tools/validate-readback/validate-readback.cpp
git commit -m "validate-readback: scaffold, load xclbin and hwctx"
```

---

## Task 2: Add dummy kernel-run helper

**Why now:** L0 needs the hwctx to exist *without* a run; L1 onward need at least one completed run. We add the run helper before any tests so all later tasks can call it.

**Files:**
- Modify: `tools/validate-readback/validate-readback.cpp`

- [ ] **Step 1: Add `run_kernel_once` helper**

Inside the anonymous namespace, after `load_insts`:

```cpp
struct RunResult {
    uint64_t kernel_us = 0;
};

RunResult run_kernel_once(xrt::device& device, xrt::kernel& kernel,
                          const std::vector<uint32_t>& instr_v,
                          bool verbose) {
    // add_one_using_dma kernarg layout:
    //   0: opcode (3 = ELF kernel)
    //   1: instr_bo
    //   2: ninstrs
    //   3: input BO   (group_id 3)
    //   4: middle BO  (group_id 4, unused by runtime_sequence)
    //   5: output BO  (group_id 5)
    constexpr size_t IN_BYTES = 64 * sizeof(int32_t);
    constexpr size_t MID_BYTES = 32 * sizeof(int32_t);
    constexpr size_t OUT_BYTES = 64 * sizeof(int32_t);

    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    std::memcpy(bo_instr.map<void*>(), instr_v.data(),
                instr_v.size() * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto bo_in  = xrt::bo(device, IN_BYTES,  XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_mid = xrt::bo(device, MID_BYTES, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_out = xrt::bo(device, OUT_BYTES, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    std::memset(bo_in.map<void*>(),  0, IN_BYTES);
    std::memset(bo_mid.map<void*>(), 0, MID_BYTES);
    std::memset(bo_out.map<void*>(), 0, OUT_BYTES);
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_mid.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = xrt::run(kernel);
    run.set_arg(0, 3u);
    run.set_arg(1, bo_instr);
    run.set_arg(2, static_cast<uint32_t>(instr_v.size()));
    run.set_arg(3, bo_in);
    run.set_arg(4, bo_mid);
    run.set_arg(5, bo_out);

    auto t0 = std::chrono::steady_clock::now();
    run.start();
    auto state = run.wait(std::chrono::seconds(30));
    auto t1 = std::chrono::steady_clock::now();
    if (state != ERT_CMD_STATE_COMPLETED) {
        throw std::runtime_error("kernel did not complete (state=" +
                                 std::to_string(static_cast<int>(state)) + ")");
    }
    RunResult r;
    r.kernel_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    if (verbose) std::fprintf(stderr, "  run completed in %lu us\n",
                              static_cast<unsigned long>(r.kernel_us));
    return r;
}
```

Add `#include "xrt/xrt_uuid.h"` and `#include "core/common/api/hw_context_int.h"`-equivalent if needed for `ERT_CMD_STATE_COMPLETED`. (XRT exposes this via `xrt/xrt_kernel.h` typically — try without first.)

- [ ] **Step 2: Add a single dummy run in main, before any tests**

Just before the `return 0;` in main:

```cpp
    auto instr_v = load_insts(args.insts);
    std::printf("[INFO] loaded %zu instr words\n", instr_v.size());

    std::printf("[INFO] running dummy kernel to allocate partition...\n");
    auto dummy = run_kernel_once(device, kernel, instr_v, args.verbose);
    std::printf("[INFO] dummy run kernel_us=%lu\n",
                static_cast<unsigned long>(dummy.kernel_us));
```

- [ ] **Step 3: Build**

```bash
cmake --build tools/validate-readback/build
```

Fix any include / API errors before continuing. If `ERT_CMD_STATE_COMPLETED` is undeclared, include `<xrt/experimental/xrt-next.h>` or check `xrt_enqueue.h`.

- [ ] **Step 4: Run, confirm dummy run completes**

```bash
./tools/validate-readback/build/validate-readback -v
```

Expected: dummy run completes, `kernel_us` printed (~hundreds of us). If kernel hangs, check NPU state with `xrt-smi examine`.

- [ ] **Step 5: Commit**

```bash
git add tools/validate-readback/validate-readback.cpp
git commit -m "validate-readback: add dummy kernel run helper"
```

---

## Task 3: L0 — pre-launch read attempt (lifecycle probe)

**Files:** Modify `tools/validate-readback/validate-readback.cpp`

- [ ] **Step 1: Add result struct + verdict printer**

After the namespace declarations, near the top:

```cpp
enum class Verdict { Pass, Fail, Info, Skip };
struct TestResult {
    std::string id;
    Verdict v;
    std::string detail;
};

const char* verdict_str(Verdict v) {
    switch (v) {
        case Verdict::Pass: return "PASS";
        case Verdict::Fail: return "FAIL";
        case Verdict::Info: return "INFO";
        case Verdict::Skip: return "SKIP";
    }
    return "?";
}

void print_result(const TestResult& r) {
    std::printf("[%s] %-4s %s\n", r.id.c_str(), verdict_str(r.v), r.detail.c_str());
}
```

- [ ] **Step 2: Implement L0**

Define this helper just before `main`:

```cpp
TestResult test_L0(xrt::hw_context& ctx, int col, int row) {
    try {
        uint32_t v = ctx.read_aie_reg(col, row, TIMER_LOW_OFFSET);
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "pre-launch read SUCCEEDED (lifecycle bug not present?), value=0x%08x",
                      v);
        return {"L0", Verdict::Info, buf};
    } catch (const std::exception& e) {
        return {"L0", Verdict::Pass,
                std::string("pre-launch read threw as expected: ") + e.what()};
    }
}
```

- [ ] **Step 3: Wire into main, BEFORE the dummy run**

Reorder main: after constructing `ctx` and `kernel`, before calling `run_kernel_once`:

```cpp
    std::vector<TestResult> results;
    results.push_back(test_L0(ctx, args.col, args.row));
    print_result(results.back());

    auto instr_v = load_insts(args.insts);
    // ... dummy run as before ...
```

- [ ] **Step 4: Build + run**

```bash
cmake --build tools/validate-readback/build && ./tools/validate-readback/build/validate-readback
```

Expected: `[L0] PASS  pre-launch read threw as expected: <some EINVAL message>`. If it shows INFO, the lifecycle diagnosis was wrong — note this prominently and continue.

- [ ] **Step 5: Commit**

```bash
git add tools/validate-readback/validate-readback.cpp
git commit -m "validate-readback: L0 pre-launch read probe"
```

---

## Task 4: L1 — same-hwctx warmup unblocks pre-launch

**Files:** Modify `tools/validate-readback/validate-readback.cpp`

- [ ] **Step 1: Implement L1**

```cpp
TestResult test_L1(xrt::hw_context& ctx, int col, int row) {
    // Same hwctx, AFTER the dummy run has completed.
    try {
        uint32_t v = ctx.read_aie_reg(col, row, TIMER_LOW_OFFSET);
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "post-warmup pre-launch read OK, TIMER_LOW=0x%08x", v);
        return {"L1", Verdict::Pass, buf};
    } catch (const std::exception& e) {
        return {"L1", Verdict::Fail,
                std::string("post-warmup read still failed: ") + e.what()};
    }
}
```

- [ ] **Step 2: Wire into main, AFTER the dummy run, before any other tests**

```cpp
    auto dummy = run_kernel_once(device, kernel, instr_v, args.verbose);
    std::printf("[INFO] dummy run kernel_us=%lu\n",
                static_cast<unsigned long>(dummy.kernel_us));

    results.push_back(test_L1(ctx, args.col, args.row));
    print_result(results.back());
```

- [ ] **Step 3: Build + run**

```bash
cmake --build tools/validate-readback/build && ./tools/validate-readback/build/validate-readback
```

Expected: `[L1] PASS  post-warmup pre-launch read OK, TIMER_LOW=0x........`.

- [ ] **Step 4: Commit**

```bash
git add tools/validate-readback/validate-readback.cpp
git commit -m "validate-readback: L1 post-warmup pre-launch unblock probe"
```

---

## Task 5: V0 — TIMER_LOW monotonic across wall time

**Files:** Modify `tools/validate-readback/validate-readback.cpp`

- [ ] **Step 1: Implement V0**

```cpp
TestResult test_V0(xrt::hw_context& ctx, int col, int row) {
    try {
        uint32_t t0 = ctx.read_aie_reg(col, row, TIMER_LOW_OFFSET);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        uint32_t t1 = ctx.read_aie_reg(col, row, TIMER_LOW_OFFSET);
        uint32_t delta = t1 - t0; // wraparound is fine for ~400k cycles
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "timer_lo: 0x%08x -> 0x%08x (delta=%u, expected ~400k)",
                      t0, t1, delta);
        Verdict v = (delta >= 200000u && delta <= 800000u) ? Verdict::Pass
                                                            : Verdict::Fail;
        return {"V0", v, buf};
    } catch (const std::exception& e) {
        return {"V0", Verdict::Fail, std::string("threw: ") + e.what()};
    }
}
```

- [ ] **Step 2: Wire into main, after L1**

```cpp
    results.push_back(test_V0(ctx, args.col, args.row));
    print_result(results.back());
```

- [ ] **Step 3: Build + run**

Expected: `[V0] PASS  timer_lo: 0xXXX -> 0xYYY (delta=~400000, expected ~400k)`.

If delta is way off (say <10k or >5M), investigate before proceeding — the path may not be returning live values.

- [ ] **Step 4: Commit**

```bash
git add tools/validate-readback/validate-readback.cpp
git commit -m "validate-readback: V0 TIMER_LOW monotonic probe"
```

---

## Task 6: V1 — write-and-read-back PERF_COUNTER0

**Files:** Modify `tools/validate-readback/validate-readback.cpp`

- [ ] **Step 1: Implement V1**

```cpp
TestResult test_V1(xrt::hw_context& ctx, int col, int row) {
    constexpr uint32_t MAGIC = 0xDEADBEEFu;
    try {
        // Make sure the counter is not actively counting (clear start_event).
        ctx.write_aie_reg(col, row, PERF_CTRL0_OFFSET, 0);
        ctx.write_aie_reg(col, row, PERF_COUNTER0_OFFSET, MAGIC);
        uint32_t got = ctx.read_aie_reg(col, row, PERF_COUNTER0_OFFSET);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "wrote 0x%08x, read 0x%08x", MAGIC, got);
        // Allow tiny advance if start_event leaked from prior state.
        Verdict v = (got == MAGIC || (got > MAGIC && got - MAGIC < 100))
                    ? Verdict::Pass : Verdict::Fail;
        return {"V1", v, buf};
    } catch (const std::exception& e) {
        return {"V1", Verdict::Fail, std::string("threw: ") + e.what()};
    }
}
```

- [ ] **Step 2: Wire into main**

```cpp
    results.push_back(test_V1(ctx, args.col, args.row));
    print_result(results.back());
```

- [ ] **Step 3: Build + run**

Expected: `[V1] PASS  wrote 0xdeadbeef, read 0xdeadbeef` (or read value within +100 of the magic).

- [ ] **Step 4: Commit**

```bash
git add tools/validate-readback/validate-readback.cpp
git commit -m "validate-readback: V1 write-and-read-back probe"
```

---

## Task 7: V2 — cross-tile distinctness (with real kernel run)

**Files:** Modify `tools/validate-readback/validate-readback.cpp`

- [ ] **Step 1: Implement V2 + helper to configure ACTIVE_CORE counter**

Before `test_V2`, add:

```cpp
void configure_active_core_counter(xrt::hw_context& ctx, int col, int row) {
    // Read PERF_CTRL0, clear start[6:0] and stop[14:8], set start = ACTIVE_CORE.
    uint32_t ctrl = ctx.read_aie_reg(col, row, PERF_CTRL0_OFFSET);
    ctrl &= ~uint32_t(0x7F7Fu);
    ctrl |= static_cast<uint32_t>(EVENT_ACTIVE_CORE);
    ctx.write_aie_reg(col, row, PERF_CTRL0_OFFSET, ctrl);
    ctx.write_aie_reg(col, row, PERF_COUNTER0_OFFSET, 0);
}
```

Then:

```cpp
struct V2Out {
    uint32_t cnt_target;
    uint32_t cnt_neighbor;
    uint64_t kernel_us;
};

TestResult test_V2(xrt::device& device, xrt::hw_context& ctx,
                   xrt::kernel& kernel,
                   const std::vector<uint32_t>& instr_v,
                   int col, int target_row, int neighbor_row,
                   bool verbose, V2Out* out) {
    try {
        configure_active_core_counter(ctx, col, target_row);
        // Disable the neighbor counter and zero it.
        ctx.write_aie_reg(col, neighbor_row, PERF_CTRL0_OFFSET, 0);
        ctx.write_aie_reg(col, neighbor_row, PERF_COUNTER0_OFFSET, 0);

        auto rr = run_kernel_once(device, kernel, instr_v, verbose);

        uint32_t target_v   = ctx.read_aie_reg(col, target_row,   PERF_COUNTER0_OFFSET);
        uint32_t neighbor_v = ctx.read_aie_reg(col, neighbor_row, PERF_COUNTER0_OFFSET);
        if (out) { out->cnt_target = target_v; out->cnt_neighbor = neighbor_v;
                   out->kernel_us = rr.kernel_us; }

        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "(col,%d)=%u (col,%d)=%u kernel_us=%lu",
                      target_row, target_v, neighbor_row, neighbor_v,
                      static_cast<unsigned long>(rr.kernel_us));
        Verdict v = (target_v > 0 && neighbor_v == 0) ? Verdict::Pass : Verdict::Fail;
        return {"V2", v, buf};
    } catch (const std::exception& e) {
        return {"V2", Verdict::Fail, std::string("threw: ") + e.what()};
    }
}
```

- [ ] **Step 2: Wire into main**

```cpp
    V2Out v2_out{};
    results.push_back(test_V2(device, ctx, kernel, instr_v,
                              args.col, args.row, args.row + 1,
                              args.verbose, &v2_out));
    print_result(results.back());
```

- [ ] **Step 3: Build + run**

Expected: `[V2] PASS  (col,2)=NNN (col,3)=0 kernel_us=...`. NNN should be a few thousand cycles. If neighbor is nonzero, indexing is suspect; investigate.

- [ ] **Step 4: Commit**

```bash
git add tools/validate-readback/validate-readback.cpp
git commit -m "validate-readback: V2 cross-tile distinctness with real kernel run"
```

---

## Task 8: V3 — sanity-check magnitude

**Files:** Modify `tools/validate-readback/validate-readback.cpp`

- [ ] **Step 1: Implement V3 (uses V2's measurement)**

```cpp
TestResult test_V3(const V2Out& v2) {
    if (v2.cnt_target == 0 || v2.kernel_us == 0) {
        return {"V3", Verdict::Skip, "V2 did not produce a usable counter or kernel_us"};
    }
    double expected = static_cast<double>(v2.kernel_us) * 400.0; // 400 MHz
    double ratio = static_cast<double>(v2.cnt_target) / expected;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "counter=%u expected~%.0f (kernel_us=%lu * 400MHz) ratio=%.2fx",
                  v2.cnt_target, expected, static_cast<unsigned long>(v2.kernel_us), ratio);
    Verdict v = (ratio >= 0.1 && ratio <= 10.0) ? Verdict::Pass : Verdict::Fail;
    return {"V3", v, buf};
}
```

- [ ] **Step 2: Wire into main, immediately after V2**

```cpp
    results.push_back(test_V3(v2_out));
    print_result(results.back());
```

- [ ] **Step 3: Build + run**

Expected: `[V3] PASS  counter=NNN expected~MMM ... ratio=~0.5-2.0x`. The ratio is usually well under 1.0 because much of `kernel_us` is host-side overhead (DMA setup, lock acquires) where ACTIVE_CORE doesn't count.

- [ ] **Step 4: Commit**

```bash
git add tools/validate-readback/validate-readback.cpp
git commit -m "validate-readback: V3 sanity-check counter magnitude"
```

---

## Task 9: V4 — run-to-run reproducibility

**Files:** Modify `tools/validate-readback/validate-readback.cpp`

- [ ] **Step 1: Implement V4**

```cpp
TestResult test_V4(xrt::device& device, xrt::hw_context& ctx,
                   xrt::kernel& kernel,
                   const std::vector<uint32_t>& instr_v,
                   int col, int row, uint32_t run1_value,
                   bool verbose) {
    if (run1_value == 0) {
        return {"V4", Verdict::Skip, "no run1 value to compare against"};
    }
    try {
        ctx.write_aie_reg(col, row, PERF_COUNTER0_OFFSET, 0);
        run_kernel_once(device, kernel, instr_v, verbose);
        uint32_t run2 = ctx.read_aie_reg(col, row, PERF_COUNTER0_OFFSET);
        double drift = std::abs(static_cast<double>(run2) - run1_value) / run1_value;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "run1=%u run2=%u drift=%.1f%%", run1_value, run2, drift * 100.0);
        Verdict v = (drift < 0.5) ? Verdict::Pass : Verdict::Fail;
        return {"V4", v, buf};
    } catch (const std::exception& e) {
        return {"V4", Verdict::Fail, std::string("threw: ") + e.what()};
    }
}
```

Add `#include <cmath>` to the include block.

- [ ] **Step 2: Wire into main, after V3**

```cpp
    results.push_back(test_V4(device, ctx, kernel, instr_v,
                              args.col, args.row, v2_out.cnt_target,
                              args.verbose));
    print_result(results.back());
```

- [ ] **Step 3: Build + run**

Expected: `[V4] PASS  run1=NNN run2=MMM drift=X%` with drift well under 50%.

- [ ] **Step 4: Commit**

```bash
git add tools/validate-readback/validate-readback.cpp
git commit -m "validate-readback: V4 run-to-run reproducibility"
```

---

## Task 10: Summary, cleanup, README

**Files:**
- Modify: `tools/validate-readback/validate-readback.cpp`
- Create: `tools/validate-readback/README.md`

- [ ] **Step 1: Add summary at end of main + cleanup of counter state**

Just before `return 0;`, replace with:

```cpp
    // Cleanup: disable any counter we left programmed.
    try {
        ctx.write_aie_reg(args.col, args.row, PERF_CTRL0_OFFSET, 0);
    } catch (...) { /* best-effort */ }

    int passes = 0, fails = 0;
    for (const auto& r : results) {
        if (r.v == Verdict::Pass) ++passes;
        else if (r.v == Verdict::Fail) ++fails;
    }
    std::printf("VALIDATION: %d/%zu PASS%s\n",
                passes, results.size(),
                fails == 0 ? "" : " (failures present)");
    return fails;
```

- [ ] **Step 2: Create README.md**

```markdown
# validate-readback

Throwaway harness that probes whether `xrt::hw_context::read_aie_reg`
returns real, wall-time-correlated, round-trip-honest data on
Phoenix NPU1 — a sanity check before building calibration on top.

See: `docs/superpowers/specs/2026-05-06-validate-read-aie-reg-design.md`.

## Build

```bash
cmake -S tools/validate-readback -B tools/validate-readback/build
cmake --build tools/validate-readback/build
```

## Run

```bash
./tools/validate-readback/build/validate-readback [-v]
```

By default, uses the peano build of `add_one_using_dma`. Override
with `--xclbin <path> --insts <path>`.

## Tests

| ID | What it proves |
|----|----------------|
| L0 | Pre-launch `read_aie_reg` fails (lifecycle bug confirmed) |
| L1 | Post-warmup pre-launch read works (option A workaround viable) |
| V0 | TIMER_LOW advances at the expected wall-time rate (~400 MHz) |
| V1 | Write-and-read-back is honest (round-trip exact) |
| V2 | col/row indexing reaches the addressed tile (cross-tile distinct) |
| V3 | Counter magnitude is plausible vs kernel_us * 400 MHz |
| V4 | Same kernel run twice yields close counter values |

Exit code = number of FAIL verdicts. 0 = all good.
```

- [ ] **Step 3: Build + run end-to-end**

```bash
cmake --build tools/validate-readback/build && ./tools/validate-readback/build/validate-readback
```

Expected: 7 verdicts (L0, L1, V0, V1, V2, V3, V4) followed by `VALIDATION: N/7 PASS`. Exit code 0.

- [ ] **Step 4: Commit**

```bash
git add tools/validate-readback/validate-readback.cpp tools/validate-readback/README.md
git commit -m "validate-readback: summary output, cleanup, README"
```

- [ ] **Step 5: Update task #355a / #357 with findings**

Briefly note in the task whether L1 PASS confirms option-A viability, and what V0-V4 say about path honesty. This is the "what did we learn" closeout.

---

## Self-Review

**Spec coverage:**
- L0 (pre-launch fails) → Task 3 ✓
- L1 (warmup unblocks) → Task 4 ✓
- V0 (TIMER_LOW monotonic) → Task 5 ✓
- V1 (write-and-read-back) → Task 6 ✓
- V2 (cross-tile distinctness) → Task 7 ✓
- V3 (sanity magnitude) → Task 8 ✓
- V4 (reproducibility) → Task 9 ✓
- File layout (CMakeLists, .cpp, README) → Tasks 1, 10 ✓
- CLI (--xclbin, --col, --row, -v) → Task 1 ✓
- Cleanup (zero PERF_CTRL0 at end) → Task 10 ✓
- Final summary line + exit code → Task 10 ✓

**Placeholder scan:** No TBDs, no "implement appropriate error handling," all code blocks complete. The one sentence "If `ERT_CMD_STATE_COMPLETED` is undeclared..." in Task 2 Step 3 is acknowledged uncertainty about XRT header layout, with a concrete fallback include — acceptable.

**Type consistency:** `Verdict` enum, `TestResult`, `V2Out`, `RunResult`, `Args`, `configure_active_core_counter` are referenced consistently. `PERF_COUNTER0_OFFSET`/`PERF_CTRL0_OFFSET`/`TIMER_LOW_OFFSET`/`EVENT_ACTIVE_CORE` constants all defined once in Task 1.

**Stretch arm of L1:** Spec mentioned an optional second arm probing partition state across hwctx instances. Plan deliberately omits per user direction ("if it complicates, leave out"). Consistent.
