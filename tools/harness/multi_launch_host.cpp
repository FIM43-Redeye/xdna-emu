// Several kernel launches in ONE process (one xrt::device), for state that silicon carries
// from one launch to the next (core registers, load-port rings). The shared test_host runs
// one launch per process; this host runs a list, so the emulator sees them as one device.
//
// Usage: multi_launch_host LAUNCH [LAUNCH ...]
//   LAUNCH = xclbin,insts,in_file,out_file,in_size,out_size[,ctx]
//   ctx    = "new" (default): its own xrt::hw_context, created after the previous launch's
//            context is gone (what a separate process does); "same": a second run in the
//            previous launch's context (same xclbin).
// Buffers as in the shared test_host: group ids 1 (insts), 3 (in), 4 (scratch), 5 (out);
// opcode 3; kernel name prefix MLIR_AIE. Exit status 0 only if every launch completed.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

struct Launch {
    std::string xclbin, insts, in_file, out_file;
    int in_size = 0, out_size = 0;
    bool same_context = false;
};

static bool parse_launch(const std::string &arg, Launch &l) {
    std::vector<std::string> f;
    std::stringstream ss(arg);
    for (std::string item; std::getline(ss, item, ',');)
        f.push_back(item);
    if (f.size() != 6 && f.size() != 7)
        return false;
    l.xclbin = f[0];
    l.insts = f[1];
    l.in_file = f[2];
    l.out_file = f[3];
    l.in_size = std::stoi(f[4]);
    l.out_size = std::stoi(f[5]);
    if (f.size() == 7) {
        if (f[6] == "same")
            l.same_context = true;
        else if (f[6] != "new")
            return false;
    }
    return true;
}

static std::vector<uint32_t> read_words(const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), words.size() * 4);
    return words;
}

struct Context {
    xrt::xclbin xclbin;
    xrt::hw_context hw;
    xrt::kernel kernel;
};

int main(int argc, const char *argv[]) {
    std::vector<Launch> launches;
    for (int i = 1; i < argc; i++) {
        Launch l;
        if (!parse_launch(argv[i], l)) {
            std::cerr << "bad launch: " << argv[i] << std::endl;
            return 2;
        }
        launches.push_back(l);
    }
    if (launches.empty() || launches[0].same_context) {
        std::cerr << "need at least one launch; the first cannot be 'same'" << std::endl;
        return 2;
    }

    auto device = xrt::device(0u);
    std::unique_ptr<Context> ctx;
    for (size_t n = 0; n < launches.size(); n++) {
        const Launch &l = launches[n];
        if (!l.same_context) {
            ctx.reset();   // the previous context goes first, as when its process exits
            auto xclbin = xrt::xclbin(l.xclbin);
            auto xkernels = xclbin.get_kernels();
            auto xk = *std::find_if(xkernels.begin(), xkernels.end(), [](xrt::xclbin::kernel &k) {
                return k.get_name().rfind("MLIR_AIE", 0) == 0;
            });
            device.register_xclbin(xclbin);
            xrt::hw_context hw(device, xclbin.get_uuid());
            xrt::kernel kernel(hw, xk.get_name());
            ctx.reset(new Context{xclbin, hw, kernel});
        }
        auto &kernel = ctx->kernel;
        std::vector<uint32_t> instr = read_words(l.insts);
        int in_elems = (l.in_size + 3) / 4, out_elems = (l.out_size + 3) / 4;
        auto bo_instr = xrt::bo(device, instr.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        auto bo_in = xrt::bo(device, in_elems * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_buf = xrt::bo(device, in_elems * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bo_out = xrt::bo(device, out_elems * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

        uint8_t *in = bo_in.map<uint8_t *>();
        std::memset(in, 0, in_elems * 4);
        std::ifstream ifs(l.in_file, std::ios::binary);
        ifs.read(reinterpret_cast<char *>(in), l.in_size);
        if (ifs.gcount() < l.in_size) {
            std::cerr << "launch " << n << ": input file too short: " << l.in_file << std::endl;
            return 1;
        }
        std::memset(bo_out.map<uint8_t *>(), 0, out_elems * 4);
        std::memset(bo_buf.map<uint8_t *>(), 0, in_elems * 4);
        std::memcpy(bo_instr.map<void *>(), instr.data(), instr.size() * 4);
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run = kernel(3u, bo_instr, (unsigned)instr.size(), bo_in, bo_buf, bo_out);
        ert_cmd_state r = run.wait();
        if (r != ERT_CMD_STATE_COMPLETED) {
            std::cerr << "launch " << n << ": kernel did not complete, status " << r << std::endl;
            return 1;
        }
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::ofstream ofs(l.out_file, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(bo_out.map<uint8_t *>()), l.out_size);
        std::cout << "launch " << n << " ok" << std::endl;
    }
    return 0;
}
