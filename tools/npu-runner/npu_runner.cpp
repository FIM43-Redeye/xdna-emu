// npu_runner -- Generic NPU test execution tool.
//
// Runs any xclbin on real AMD NPU hardware via XRT, reading/writing raw
// binary buffer files.  All manifest logic and data generation stays in
// Rust; this tool is a thin XRT wrapper.
//
// Two modes of operation:
//
//   1. Single-kernel (backward compatible):
//      npu-runner <xclbin> <insts.bin> [options]
//        --in  <group_id> <file> <size_bytes>   Input buffer  (repeatable)
//        --out <group_id> <file> <size_bytes>   Output buffer (repeatable)
//        --timeout <seconds>                    Execution timeout (default: 30)
//        --kernel <name>                        Kernel name   (default: MLIR_AIE)
//
//   2. Multi-kernel (run spec file):
//      npu-runner --spec <spec_file>
//
//      The spec file is a simple line-oriented format describing multiple
//      kernel invocations with buffer linking.  See RunSpec below.
//
// Exit codes:
//   0  Success
//   1  Runtime error (device/xclbin/kernel failure)
//   2  Timeout (kernel did not complete in time)
//   3  Argument error

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/experimental/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// ---------------------------------------------------------------------------
// Buffer descriptor -- parsed from CLI or spec file.
// ---------------------------------------------------------------------------
struct BufferDesc {
    int         group_id;
    std::string file_path;    // empty for intermediate/linked buffers
    size_t      size_bytes;
    bool        is_output;    // true = read back after execution
    bool        is_link;      // true = linked from another run's buffer
    int         link_run;     // source run index (when is_link)
    int         link_gid;     // source group_id (when is_link)
};

// ---------------------------------------------------------------------------
// A single kernel invocation within a multi-kernel spec.
// ---------------------------------------------------------------------------
struct RunStep {
    std::string kernel_name;
    std::string insts_path;
    std::vector<BufferDesc> buffers;
};

// ---------------------------------------------------------------------------
// Execution mode for multi-kernel tests.
// ---------------------------------------------------------------------------
enum class ExecMode {
    Single,       // Legacy: one kernel, direct CLI args
    Sequential,   // Multi-kernel: run each in order, host-side linking
    Runlist,      // Multi-kernel: submit all via xrt::runlist
};

// ---------------------------------------------------------------------------
// Complete run specification (multi-kernel).
// ---------------------------------------------------------------------------
struct RunSpec {
    std::string xclbin_path;
    ExecMode    mode = ExecMode::Sequential;
    unsigned    device_index = 0;
    int         timeout_sec = 30;
    std::vector<RunStep> steps;
};

// ---------------------------------------------------------------------------
// Read a binary file into a byte vector.
// ---------------------------------------------------------------------------
static std::vector<uint8_t> read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "ERROR: cannot open file: " << path << "\n";
        std::exit(1);
    }
    f.seekg(0, std::ios::end);
    auto len = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(len);
    if (!f.read(reinterpret_cast<char *>(buf.data()), len)) {
        std::cerr << "ERROR: failed to read file: " << path << "\n";
        std::exit(1);
    }
    return buf;
}

// ---------------------------------------------------------------------------
// Write a byte buffer to a file.
// ---------------------------------------------------------------------------
static void write_file(const std::string &path, const void *data, size_t len) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "ERROR: cannot write file: " << path << "\n";
        std::exit(1);
    }
    f.write(reinterpret_cast<const char *>(data), len);
    if (!f) {
        std::cerr << "ERROR: write failed: " << path << "\n";
        std::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Parse a run spec file.
//
// Format (line-oriented, # comments, blank lines ignored):
//
//   xclbin <path>
//   mode <sequential|runlist>
//   device <index>
//   timeout <seconds>
//
//   run <kernel_name> <insts_path>
//   in <group_id> <file_path> <size_bytes>
//   out <group_id> <file_path> <size_bytes>
//   out <group_id> <size_bytes>                  (intermediate, no file)
//   link <group_id> <src_run_index> <src_gid>
//
// "run" starts a new step.  "in", "out", "link" attach to the current step.
// ---------------------------------------------------------------------------
static RunSpec parse_spec(const std::string &path) {
    RunSpec spec;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "ERROR: cannot open spec file: " << path << "\n";
        std::exit(3);
    }

    std::string line;
    while (std::getline(f, line)) {
        // Strip comments and whitespace
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos)
            line = line.substr(0, comment_pos);

        // Trim leading/trailing whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        size_t end = line.find_last_not_of(" \t\r\n");
        line = line.substr(0, end + 1);
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;

        if (keyword == "xclbin") {
            iss >> spec.xclbin_path;
        } else if (keyword == "mode") {
            std::string mode_str;
            iss >> mode_str;
            if (mode_str == "sequential")
                spec.mode = ExecMode::Sequential;
            else if (mode_str == "runlist")
                spec.mode = ExecMode::Runlist;
            else {
                std::cerr << "ERROR: unknown mode: " << mode_str << "\n";
                std::exit(3);
            }
        } else if (keyword == "device") {
            iss >> spec.device_index;
        } else if (keyword == "timeout") {
            iss >> spec.timeout_sec;
        } else if (keyword == "run") {
            RunStep step;
            iss >> step.kernel_name >> step.insts_path;
            spec.steps.push_back(std::move(step));
        } else if (keyword == "in") {
            if (spec.steps.empty()) {
                std::cerr << "ERROR: 'in' before 'run' in spec\n";
                std::exit(3);
            }
            BufferDesc bd{};
            bd.is_output = false;
            bd.is_link = false;
            iss >> bd.group_id >> bd.file_path >> bd.size_bytes;
            spec.steps.back().buffers.push_back(bd);
        } else if (keyword == "out") {
            if (spec.steps.empty()) {
                std::cerr << "ERROR: 'out' before 'run' in spec\n";
                std::exit(3);
            }
            BufferDesc bd{};
            bd.is_output = true;
            bd.is_link = false;
            // out can be: "out <gid> <file> <size>" or "out <gid> <size>"
            std::string token1, token2;
            iss >> bd.group_id >> token1;
            if (iss >> token2) {
                // Three tokens after gid: file_path + size
                bd.file_path = token1;
                bd.size_bytes = std::stoull(token2);
            } else {
                // Two tokens after gid: just size (intermediate buffer)
                bd.size_bytes = std::stoull(token1);
            }
            spec.steps.back().buffers.push_back(bd);
        } else if (keyword == "link") {
            if (spec.steps.empty()) {
                std::cerr << "ERROR: 'link' before 'run' in spec\n";
                std::exit(3);
            }
            BufferDesc bd{};
            bd.is_output = false;
            bd.is_link = true;
            iss >> bd.group_id >> bd.link_run >> bd.link_gid;
            spec.steps.back().buffers.push_back(bd);
        } else {
            std::cerr << "WARNING: unknown spec keyword: " << keyword << "\n";
        }
    }

    if (spec.xclbin_path.empty()) {
        std::cerr << "ERROR: spec file missing 'xclbin' directive\n";
        std::exit(3);
    }
    if (spec.steps.empty()) {
        std::cerr << "ERROR: spec file has no 'run' steps\n";
        std::exit(3);
    }

    return spec;
}

// ---------------------------------------------------------------------------
// Find a kernel by exact name in the xclbin.
// ---------------------------------------------------------------------------
static xrt::xclbin::kernel find_kernel(
    const std::vector<xrt::xclbin::kernel> &xkernels,
    const std::string &name)
{
    auto it = std::find_if(xkernels.begin(), xkernels.end(),
        [&name](const xrt::xclbin::kernel &k) {
            return k.get_name() == name;
        });
    if (it == xkernels.end()) {
        // Fall back to prefix match
        it = std::find_if(xkernels.begin(), xkernels.end(),
            [&name](const xrt::xclbin::kernel &k) {
                return k.get_name().rfind(name, 0) == 0;
            });
    }
    if (it == xkernels.end()) {
        std::cerr << "ERROR: kernel '" << name << "' not found in xclbin\n";
        std::cerr << "Available kernels:";
        for (auto &k : xkernels)
            std::cerr << " " << k.get_name();
        std::cerr << "\n";
        std::exit(1);
    }
    return *it;
}

// ---------------------------------------------------------------------------
// Tracking structure for live buffer objects across runs.
// Key: (run_index, group_id) -> xrt::bo
// ---------------------------------------------------------------------------
struct LiveBufferKey {
    int run_index;
    int group_id;
    bool operator<(const LiveBufferKey &o) const {
        if (run_index != o.run_index) return run_index < o.run_index;
        return group_id < o.group_id;
    }
};

// ---------------------------------------------------------------------------
// Execute multi-kernel spec using sequential mode.
//
// Runs each kernel in order.  After each run completes, linked buffers
// are available for the next run because they share the same xrt::bo.
// For sequential mode, output buffers from one run can be read back
// and re-synced as input for the next (host-side copy).
// ---------------------------------------------------------------------------
static int exec_sequential(
    const RunSpec &spec,
    xrt::device &device,
    xrt::hw_context &context,
    const std::vector<xrt::xclbin::kernel> &xkernels)
{
    // Buffer pool: (run_index, group_id) -> bo
    std::map<LiveBufferKey, xrt::bo> bo_pool;

    // Track which buffers need file write-back
    struct OutputRecord {
        LiveBufferKey key;
        std::string file_path;
        size_t size_bytes;
    };
    std::vector<OutputRecord> output_records;

    for (int run_idx = 0; run_idx < static_cast<int>(spec.steps.size()); ++run_idx) {
        const auto &step = spec.steps[run_idx];

        std::cerr << "  [run " << run_idx << "] kernel=" << step.kernel_name
                  << " insts=" << step.insts_path << "\n";

        auto xk = find_kernel(xkernels, step.kernel_name);
        auto kernel = xrt::kernel(context, xk.get_name());

        // Load instructions
        auto insts_raw = read_file(step.insts_path);
        if (insts_raw.size() % 4 != 0) {
            std::cerr << "ERROR: instruction file size is not 4-byte aligned: "
                      << step.insts_path << "\n";
            return 1;
        }
        size_t instr_count = insts_raw.size() / sizeof(uint32_t);

        auto bo_instr = xrt::bo(device, insts_raw.size(),
                                XCL_BO_FLAGS_CACHEABLE,
                                kernel.group_id(1));
        std::memcpy(bo_instr.map<void *>(), insts_raw.data(), insts_raw.size());
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Create buffers for this run
        for (const auto &bd : step.buffers) {
            LiveBufferKey key{run_idx, bd.group_id};

            if (bd.is_link) {
                // Linked buffer: read from source run's output, create new BO
                LiveBufferKey src_key{bd.link_run, bd.link_gid};
                auto src_it = bo_pool.find(src_key);
                if (src_it == bo_pool.end()) {
                    std::cerr << "ERROR: link references run " << bd.link_run
                              << " gid " << bd.link_gid
                              << " which does not exist\n";
                    return 1;
                }
                // Read data from source BO
                auto &src_bo = src_it->second;
                src_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                size_t src_size = src_bo.size();
                auto *src_data = src_bo.map<uint8_t *>();

                // Create new BO for this kernel and copy data
                auto bo = xrt::bo(device, src_size,
                                  XRT_BO_FLAGS_HOST_ONLY,
                                  kernel.group_id(bd.group_id));
                auto *dst = bo.map<uint8_t *>();
                std::memcpy(dst, src_data, src_size);
                bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                std::cerr << "    link gid=" << bd.group_id
                          << " <- run " << bd.link_run << " gid " << bd.link_gid
                          << " (" << src_size << " bytes)\n";

                bo_pool[key] = std::move(bo);
            } else if (!bd.is_output) {
                // Input buffer: load from file
                auto bo = xrt::bo(device, bd.size_bytes,
                                  XRT_BO_FLAGS_HOST_ONLY,
                                  kernel.group_id(bd.group_id));
                auto file_data = read_file(bd.file_path);
                auto *mapped = bo.map<uint8_t *>();
                std::memset(mapped, 0, bd.size_bytes);
                size_t copy_len = std::min(file_data.size(), bd.size_bytes);
                std::memcpy(mapped, file_data.data(), copy_len);
                bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                std::cerr << "    in  gid=" << bd.group_id
                          << " file=" << bd.file_path
                          << " size=" << bd.size_bytes << "\n";

                bo_pool[key] = std::move(bo);
            } else {
                // Output buffer: zero-initialize
                auto bo = xrt::bo(device, bd.size_bytes,
                                  XRT_BO_FLAGS_HOST_ONLY,
                                  kernel.group_id(bd.group_id));
                auto *mapped = bo.map<uint8_t *>();
                std::memset(mapped, 0, bd.size_bytes);
                bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                std::cerr << "    out gid=" << bd.group_id
                          << " size=" << bd.size_bytes;
                if (!bd.file_path.empty())
                    std::cerr << " file=" << bd.file_path;
                std::cerr << "\n";

                bo_pool[key] = std::move(bo);

                // Track for write-back if file specified
                if (!bd.file_path.empty()) {
                    output_records.push_back({key, bd.file_path, bd.size_bytes});
                }
            }
        }

        // Set kernel arguments and run
        auto run = xrt::run(kernel);
        run.set_arg(0, static_cast<uint32_t>(3));  // opcode
        run.set_arg(1, bo_instr);
        run.set_arg(2, static_cast<uint32_t>(instr_count));

        for (const auto &bd : step.buffers) {
            LiveBufferKey key{run_idx, bd.group_id};
            run.set_arg(bd.group_id, bo_pool[key]);
        }

        run.start();
        auto timeout_ms = static_cast<unsigned int>(spec.timeout_sec * 1000);
        auto state = run.wait(std::chrono::milliseconds(timeout_ms));

        if (state == ERT_CMD_STATE_TIMEOUT) {
            std::cerr << "ERROR: run " << run_idx << " timed out\n";
            return 2;
        }
        if (state != ERT_CMD_STATE_COMPLETED) {
            std::cerr << "ERROR: run " << run_idx
                      << " did not complete, state=" << state << "\n";
            return 1;
        }
        std::cerr << "  [run " << run_idx << "] completed\n";
    }

    // Write back all output files
    for (const auto &rec : output_records) {
        auto &bo = bo_pool[rec.key];
        bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto *mapped = bo.map<uint8_t *>();
        write_file(rec.file_path, mapped, rec.size_bytes);
        std::cerr << "  Wrote output: " << rec.file_path
                  << " (" << rec.size_bytes << " bytes)\n";
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Execute multi-kernel spec using runlist mode.
//
// Creates all runs upfront and submits them together via xrt::runlist.
// Buffer linking uses the same xrt::bo object (shared reference), so the
// hardware handles the data dependency.
// ---------------------------------------------------------------------------
static int exec_runlist(
    const RunSpec &spec,
    xrt::device &device,
    xrt::hw_context &context,
    const std::vector<xrt::xclbin::kernel> &xkernels)
{
    // Buffer pool: (run_index, group_id) -> bo
    std::map<LiveBufferKey, xrt::bo> bo_pool;

    // Track output files for write-back
    struct OutputRecord {
        LiveBufferKey key;
        std::string file_path;
        size_t size_bytes;
    };
    std::vector<OutputRecord> output_records;

    // Keep instruction BOs alive
    std::vector<xrt::bo> instr_bos;

    // Build all runs
    xrt::runlist runlist(context);

    for (int run_idx = 0; run_idx < static_cast<int>(spec.steps.size()); ++run_idx) {
        const auto &step = spec.steps[run_idx];

        std::cerr << "  [run " << run_idx << "] kernel=" << step.kernel_name
                  << " insts=" << step.insts_path << "\n";

        auto xk = find_kernel(xkernels, step.kernel_name);
        auto kernel = xrt::kernel(context, xk.get_name());

        // Load instructions
        auto insts_raw = read_file(step.insts_path);
        if (insts_raw.size() % 4 != 0) {
            std::cerr << "ERROR: instruction file size is not 4-byte aligned: "
                      << step.insts_path << "\n";
            return 1;
        }
        size_t instr_count = insts_raw.size() / sizeof(uint32_t);

        auto bo_instr = xrt::bo(device, insts_raw.size(),
                                XCL_BO_FLAGS_CACHEABLE,
                                kernel.group_id(1));
        std::memcpy(bo_instr.map<void *>(), insts_raw.data(), insts_raw.size());
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        instr_bos.push_back(bo_instr);

        // Create buffers
        for (const auto &bd : step.buffers) {
            LiveBufferKey key{run_idx, bd.group_id};

            if (bd.is_link) {
                // Linked buffer: SHARE the same xrt::bo as the source.
                // This is the key difference from sequential mode -- the
                // hardware sees the same physical buffer.
                LiveBufferKey src_key{bd.link_run, bd.link_gid};
                auto src_it = bo_pool.find(src_key);
                if (src_it == bo_pool.end()) {
                    std::cerr << "ERROR: link references run " << bd.link_run
                              << " gid " << bd.link_gid
                              << " which does not exist\n";
                    return 1;
                }
                bo_pool[key] = src_it->second;  // shared reference
                std::cerr << "    link gid=" << bd.group_id
                          << " <- run " << bd.link_run << " gid " << bd.link_gid
                          << " (shared BO)\n";
            } else if (!bd.is_output) {
                // Input buffer
                auto bo = xrt::bo(device, bd.size_bytes,
                                  XRT_BO_FLAGS_HOST_ONLY,
                                  kernel.group_id(bd.group_id));
                auto file_data = read_file(bd.file_path);
                auto *mapped = bo.map<uint8_t *>();
                std::memset(mapped, 0, bd.size_bytes);
                size_t copy_len = std::min(file_data.size(), bd.size_bytes);
                std::memcpy(mapped, file_data.data(), copy_len);
                bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                std::cerr << "    in  gid=" << bd.group_id
                          << " file=" << bd.file_path
                          << " size=" << bd.size_bytes << "\n";
                bo_pool[key] = std::move(bo);
            } else {
                // Output buffer
                auto bo = xrt::bo(device, bd.size_bytes,
                                  XRT_BO_FLAGS_HOST_ONLY,
                                  kernel.group_id(bd.group_id));
                auto *mapped = bo.map<uint8_t *>();
                std::memset(mapped, 0, bd.size_bytes);
                bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                std::cerr << "    out gid=" << bd.group_id
                          << " size=" << bd.size_bytes;
                if (!bd.file_path.empty())
                    std::cerr << " file=" << bd.file_path;
                std::cerr << "\n";
                bo_pool[key] = std::move(bo);

                if (!bd.file_path.empty()) {
                    output_records.push_back({key, bd.file_path, bd.size_bytes});
                }
            }
        }

        // Build the run
        auto run = xrt::run(kernel);
        run.set_arg(0, static_cast<uint32_t>(3));  // opcode
        run.set_arg(1, bo_instr);
        run.set_arg(2, static_cast<uint32_t>(instr_count));

        for (const auto &bd : step.buffers) {
            LiveBufferKey key{run_idx, bd.group_id};
            run.set_arg(bd.group_id, bo_pool[key]);
        }

        runlist.add(run);
    }

    // Execute the runlist
    std::cerr << "Executing runlist (" << spec.steps.size()
              << " runs, timeout=" << spec.timeout_sec << "s)...\n";

    runlist.execute();
    auto timeout_ms = static_cast<unsigned int>(spec.timeout_sec * 1000);
    auto cv_state = runlist.wait(std::chrono::milliseconds(timeout_ms));

    if (cv_state == std::cv_status::timeout) {
        std::cerr << "ERROR: runlist timed out after "
                  << spec.timeout_sec << " seconds\n";
        return 2;
    }
    std::cerr << "Runlist completed successfully.\n";

    // Write back outputs
    for (const auto &rec : output_records) {
        auto &bo = bo_pool[rec.key];
        bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto *mapped = bo.map<uint8_t *>();
        write_file(rec.file_path, mapped, rec.size_bytes);
        std::cerr << "  Wrote output: " << rec.file_path
                  << " (" << rec.size_bytes << " bytes)\n";
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Execute a multi-kernel spec.
// ---------------------------------------------------------------------------
static int exec_spec(const RunSpec &spec) {
    try {
        std::cerr << "Opening device " << spec.device_index << "...\n";
        auto device = xrt::device(spec.device_index);

        std::cerr << "Loading xclbin: " << spec.xclbin_path << "\n";
        auto xclbin = xrt::xclbin(spec.xclbin_path);
        auto xkernels = xclbin.get_kernels();

        std::cerr << "Kernels in xclbin:";
        for (auto &k : xkernels)
            std::cerr << " " << k.get_name();
        std::cerr << "\n";

        device.register_xclbin(xclbin);
        auto context = xrt::hw_context(device, xclbin.get_uuid());

        switch (spec.mode) {
            case ExecMode::Sequential:
                return exec_sequential(spec, device, context, xkernels);
            case ExecMode::Runlist:
                return exec_runlist(spec, device, context, xkernels);
            default:
                std::cerr << "ERROR: unsupported exec mode\n";
                return 1;
        }
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}

// ---------------------------------------------------------------------------
// Execute a single-kernel run (legacy CLI mode).
// ---------------------------------------------------------------------------
static int exec_single(
    const std::string &xclbin_path,
    const std::string &insts_path,
    const std::string &kernel_name,
    unsigned device_index,
    int timeout_sec,
    std::vector<BufferDesc> &buffers)
{
    auto insts_raw = read_file(insts_path);
    if (insts_raw.size() % 4 != 0) {
        std::cerr << "ERROR: instruction file size is not a multiple of 4 bytes\n";
        return 1;
    }
    size_t instr_count = insts_raw.size() / sizeof(uint32_t);

    try {
        std::cerr << "Opening device " << device_index << "...\n";
        auto device = xrt::device(device_index);

        std::cerr << "Loading xclbin: " << xclbin_path << "\n";
        auto xclbin = xrt::xclbin(xclbin_path);

        auto xkernels = xclbin.get_kernels();
        auto xk = find_kernel(xkernels, kernel_name);
        auto resolved_name = xk.get_name();
        std::cerr << "Using kernel: " << resolved_name << "\n";

        device.register_xclbin(xclbin);
        auto context = xrt::hw_context(device, xclbin.get_uuid());
        auto kernel  = xrt::kernel(context, resolved_name);

        // Create instruction buffer
        auto bo_instr = xrt::bo(device, insts_raw.size(),
                                XCL_BO_FLAGS_CACHEABLE,
                                kernel.group_id(1));
        std::memcpy(bo_instr.map<void *>(), insts_raw.data(), insts_raw.size());
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Sort buffers by group_id
        std::sort(buffers.begin(), buffers.end(),
            [](const BufferDesc &a, const BufferDesc &b) {
                return a.group_id < b.group_id;
            });

        struct LiveBuffer {
            xrt::bo   bo;
            BufferDesc desc;
        };
        std::vector<LiveBuffer> live_buffers;

        for (auto &bd : buffers) {
            auto bo = xrt::bo(device, bd.size_bytes,
                              XRT_BO_FLAGS_HOST_ONLY,
                              kernel.group_id(bd.group_id));

            if (!bd.is_output) {
                auto file_data = read_file(bd.file_path);
                if (file_data.size() > bd.size_bytes) {
                    std::cerr << "WARNING: input file " << bd.file_path
                              << " (" << file_data.size()
                              << " bytes) exceeds buffer size ("
                              << bd.size_bytes << " bytes), truncating\n";
                    file_data.resize(bd.size_bytes);
                }
                auto *mapped = bo.map<uint8_t *>();
                std::memset(mapped, 0, bd.size_bytes);
                std::memcpy(mapped, file_data.data(), file_data.size());
                bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                std::cerr << "  Input  group_id=" << bd.group_id
                          << " size=" << bd.size_bytes
                          << " file=" << bd.file_path << "\n";
            } else {
                auto *mapped = bo.map<uint8_t *>();
                std::memset(mapped, 0, bd.size_bytes);
                bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                std::cerr << "  Output group_id=" << bd.group_id
                          << " size=" << bd.size_bytes
                          << " file=" << bd.file_path << "\n";
            }

            live_buffers.push_back({std::move(bo), bd});
        }

        // Set kernel arguments and run
        auto run = xrt::run(kernel);
        run.set_arg(0, static_cast<uint32_t>(3));  // opcode
        run.set_arg(1, bo_instr);
        run.set_arg(2, static_cast<uint32_t>(instr_count));

        for (auto &lb : live_buffers) {
            run.set_arg(lb.desc.group_id, lb.bo);
        }

        std::cerr << "Running kernel (timeout=" << timeout_sec << "s)...\n";
        run.start();

        auto timeout_ms = static_cast<unsigned int>(timeout_sec * 1000);
        auto state = run.wait(std::chrono::milliseconds(timeout_ms));

        if (state == ERT_CMD_STATE_TIMEOUT) {
            std::cerr << "ERROR: kernel execution timed out after "
                      << timeout_sec << " seconds\n";
            return 2;
        }
        if (state != ERT_CMD_STATE_COMPLETED) {
            std::cerr << "ERROR: kernel did not complete, state="
                      << state << "\n";
            return 1;
        }

        std::cerr << "Kernel completed successfully.\n";

        // Read back output buffers
        for (auto &lb : live_buffers) {
            if (lb.desc.is_output) {
                lb.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                auto *mapped = lb.bo.map<uint8_t *>();
                write_file(lb.desc.file_path, mapped, lb.desc.size_bytes);
                std::cerr << "  Wrote output: " << lb.desc.file_path
                          << " (" << lb.desc.size_bytes << " bytes)\n";
            }
        }

        return 0;

    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}

// ---------------------------------------------------------------------------
// Usage message.
// ---------------------------------------------------------------------------
static void usage(const char *prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " <xclbin> <insts.bin> [options]    Single-kernel mode\n"
        << "  " << prog << " --spec <file>                     Multi-kernel mode\n"
        << "\n"
        << "Single-kernel options:\n"
        << "  --in  <group_id> <file> <size_bytes>   Input buffer (repeatable)\n"
        << "  --out <group_id> <file> <size_bytes>   Output buffer (repeatable)\n"
        << "  --timeout <seconds>                    Execution timeout (default: 30)\n"
        << "  --kernel <name>                        Kernel name (default: MLIR_AIE)\n"
        << "  --device <index>                       Device index (default: 0)\n"
        << "\n"
        << "Multi-kernel spec file format:\n"
        << "  xclbin <path>                          .xclbin file to load\n"
        << "  mode <sequential|runlist>              Execution mode\n"
        << "  timeout <seconds>                      Execution timeout\n"
        << "  run <kernel_name> <insts_path>         Start a new kernel invocation\n"
        << "  in <group_id> <file> <size_bytes>      Input buffer for current run\n"
        << "  out <group_id> <file> <size_bytes>     Output buffer for current run\n"
        << "  out <group_id> <size_bytes>            Intermediate buffer (no file)\n"
        << "  link <group_id> <src_run> <src_gid>    Link to another run's buffer\n"
        << "\n"
        << "  -h, --help                             Show this help\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    // Check for help flag
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        }
    }

    if (argc < 2) {
        usage(argv[0]);
        return 3;
    }

    // Check for --spec mode
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--spec" && i + 1 < argc) {
            auto spec = parse_spec(argv[i + 1]);
            return exec_spec(spec);
        }
    }

    // Legacy single-kernel mode
    if (argc < 3) {
        usage(argv[0]);
        return 3;
    }

    std::string xclbin_path = argv[1];
    std::string insts_path  = argv[2];

    std::string kernel_name = "MLIR_AIE";
    unsigned    device_index = 0;
    int         timeout_sec  = 30;
    std::vector<BufferDesc> buffers;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];

        if ((arg == "--in" || arg == "--out") && i + 3 < argc) {
            BufferDesc bd{};
            bd.group_id   = std::stoi(argv[i + 1]);
            bd.file_path  = argv[i + 2];
            bd.size_bytes = std::stoull(argv[i + 3]);
            bd.is_output  = (arg == "--out");
            bd.is_link    = false;
            buffers.push_back(bd);
            i += 3;
        } else if (arg == "--timeout" && i + 1 < argc) {
            timeout_sec = std::stoi(argv[++i]);
        } else if (arg == "--kernel" && i + 1 < argc) {
            kernel_name = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            device_index = std::stoul(argv[++i]);
        } else {
            std::cerr << "ERROR: unknown option: " << arg << "\n";
            usage(argv[0]);
            return 3;
        }
    }

    return exec_single(xclbin_path, insts_path, kernel_name,
                        device_index, timeout_sec, buffers);
}
