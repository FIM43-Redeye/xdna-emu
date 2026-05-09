// aie-reg-probe -- diagnostic harness for reading AIE tile registers via
// xrt::hw_context::read_aie_reg.  Built to debug bug #6 (memtile_dmas
// family TDRs on HW with --no-trace).
//
// Modes:
//   passive   -- register xclbin, force_connect, read regs once, exit
//   run-dma2  -- same setup + kick off the failing kernel (2-BO shape:
//                opcode, bo_instr, n_instr, bo_in, bo_out), then
//                periodically read regs while it hangs.  Requires
//                amdxdna.tdr_dump_ctx=1 so TDR doesn't recover.
//
// Register offsets are taken from aie-rt's xaiemlgbl_params.h (the
// authoritative AIE-ML/AIE2 register map).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_uuid.h"

namespace MemTile {
constexpr uint32_t LOCK_VALUE(int n)   { return 0x000C0000u + n * 0x10u; }
constexpr uint32_t S2MM_CTRL(int n)    { return 0x000A0600u + n * 0x08u; }
constexpr uint32_t S2MM_QUEUE(int n)   { return 0x000A0604u + n * 0x08u; }
constexpr uint32_t S2MM_STATUS(int n)  { return 0x000A0660u + n * 0x04u; }
constexpr uint32_t MM2S_CTRL(int n)    { return 0x000A0630u + n * 0x08u; }
constexpr uint32_t MM2S_QUEUE(int n)   { return 0x000A0634u + n * 0x08u; }
constexpr uint32_t MM2S_STATUS(int n)  { return 0x000A0680u + n * 0x04u; }
constexpr uint32_t BD_WORD(int bd, int word) {
  return 0x000A0000u + bd * 0x20u + word * 0x04u;
}
}  // namespace MemTile

namespace Shim {
constexpr uint32_t S2MM_CTRL(int n)    { return 0x0001D200u + n * 0x08u; }
constexpr uint32_t S2MM_STATUS(int n)  { return 0x0001D220u + n * 0x04u; }
constexpr uint32_t MM2S_CTRL(int n)    { return 0x0001D210u + n * 0x08u; }
constexpr uint32_t MM2S_STATUS(int n)  { return 0x0001D228u + n * 0x04u; }
}  // namespace Shim

struct RegSpec {
  uint16_t col;
  uint16_t row;
  uint32_t addr;
  std::string name;
};

// Build the register list relevant to dma_configure_task_lock.  Tile
// addressing uses col/row relative to the hw_context partition: row 0
// is shim, row 1 is memtile.  Col 0 = first column of the partition.
static std::vector<RegSpec> build_reg_list() {
  std::vector<RegSpec> r;
  // Memtile (col 0, row 1): the locks the test cares about.
  r.push_back({0, 1, MemTile::LOCK_VALUE(0), "memtile_lock0_value (prod_lock)"});
  r.push_back({0, 1, MemTile::LOCK_VALUE(1), "memtile_lock1_value (cons_lock)"});

  // Memtile S2MM channel 0 (the BD chain 0->1->2->3).
  r.push_back({0, 1, MemTile::S2MM_STATUS(0), "memtile_s2mm0_status"});
  r.push_back({0, 1, MemTile::S2MM_CTRL(0),   "memtile_s2mm0_ctrl"});
  r.push_back({0, 1, MemTile::S2MM_QUEUE(0),  "memtile_s2mm0_queue"});

  // Memtile MM2S channel 0 (the single BD 4 with cons_lock(4) acq).
  r.push_back({0, 1, MemTile::MM2S_STATUS(0), "memtile_mm2s0_status"});
  r.push_back({0, 1, MemTile::MM2S_CTRL(0),   "memtile_mm2s0_ctrl"});
  r.push_back({0, 1, MemTile::MM2S_QUEUE(0),  "memtile_mm2s0_queue"});

  // Memtile BDs 0-4 -- word 0 (addr), word 1 (length), word 7 (lock fields + valid_bd).
  for (int bd = 0; bd <= 4; ++bd) {
    char nm[64];
    std::snprintf(nm, sizeof(nm), "memtile_bd%d_word0", bd);
    r.push_back({0, 1, MemTile::BD_WORD(bd, 0), nm});
    std::snprintf(nm, sizeof(nm), "memtile_bd%d_word1", bd);
    r.push_back({0, 1, MemTile::BD_WORD(bd, 1), nm});
    std::snprintf(nm, sizeof(nm), "memtile_bd%d_word7_locks", bd);
    r.push_back({0, 1, MemTile::BD_WORD(bd, 7), nm});
  }

  // Shim (col 0, row 0): both DMA directions.
  r.push_back({0, 0, Shim::MM2S_STATUS(0), "shim_mm2s0_status"});
  r.push_back({0, 0, Shim::MM2S_CTRL(0),   "shim_mm2s0_ctrl"});
  r.push_back({0, 0, Shim::S2MM_STATUS(0), "shim_s2mm0_status"});
  r.push_back({0, 0, Shim::S2MM_CTRL(0),   "shim_s2mm0_ctrl"});

  return r;
}

// Decode the channel STATUS register (memtile S2MM/MM2S share layout per
// xaiemlgbl_params.h).  Returns a short human-readable summary.
static std::string decode_channel_status(uint32_t v) {
  unsigned status      = (v >> 0)  & 0x3;
  unsigned stall_acq   = (v >> 2)  & 0x1;
  unsigned stall_rel   = (v >> 3)  & 0x1;
  unsigned stall_strm  = (v >> 4)  & 0x1;
  unsigned stall_tct   = (v >> 5)  & 0x1;
  unsigned err_lock    = (v >> 8)  & 0x1;
  unsigned err_dm      = (v >> 9)  & 0x1;
  unsigned err_bd_unav = (v >> 10) & 0x1;
  unsigned err_bd_inv  = (v >> 11) & 0x1;
  unsigned running     = (v >> 19) & 0x1;
  unsigned qsize       = (v >> 20) & 0x7;
  unsigned cur_bd      = (v >> 24) & 0x3F;

  static const char* names[] = {"IDLE", "STARTING", "RUNNING", "STALLED"};
  std::string s = "[" + std::string(names[status]) + "]";
  if (running)     s += " running";
  if (stall_acq)   s += " STALL_LOCK_ACQ";
  if (stall_rel)   s += " STALL_LOCK_REL";
  if (stall_strm)  s += " STALL_STRM_STARV";
  if (stall_tct)   s += " STALL_TCT_FIFO";
  if (err_lock)    s += " ERR_LOCK_ACCESS";
  if (err_dm)      s += " ERR_DM_ACCESS";
  if (err_bd_unav) s += " ERR_BD_UNAVAIL";
  if (err_bd_inv)  s += " ERR_BD_INVAL";
  s += " qsize=" + std::to_string(qsize);
  s += " cur_bd=" + std::to_string(cur_bd);
  return s;
}

static void dump_regs(xrt::hw_context& ctx, const std::vector<RegSpec>& regs,
                      const std::string& tag) {
  std::cout << "===== " << tag << " =====\n";
  for (const auto& r : regs) {
    uint32_t v = 0;
    bool ok = true;
    std::string err;
    try {
      v = ctx.read_aie_reg(r.col, r.row, r.addr);
    } catch (const std::exception& e) {
      ok = false;
      err = e.what();
    }
    char buf[256];
    if (ok) {
      std::snprintf(buf, sizeof(buf), "  c%u r%u @0x%05X = 0x%08X  %s",
                    r.col, r.row, r.addr, v, r.name.c_str());
      std::cout << buf;
      if (r.name.find("status") != std::string::npos)
        std::cout << "  " << decode_channel_status(v);
      std::cout << "\n";
    } else {
      std::snprintf(buf, sizeof(buf), "  c%u r%u @0x%05X  ERR: %s  %s\n",
                    r.col, r.row, r.addr, err.c_str(), r.name.c_str());
      std::cout << buf;
    }
  }
  std::cout.flush();
}

static int usage(const char* prog) {
  std::cerr <<
    "Usage:\n"
    "  " << prog << " passive --xclbin <path>\n"
    "  " << prog << " run-dma2 --xclbin <path> --instr <path> "
    "[--kernel <prefix>] [--length <int>] [--samples <n>] [--interval-ms <n>]\n";
  return 2;
}

static std::string get_arg(int argc, char** argv, const std::string& key) {
  for (int i = 0; i < argc - 1; ++i)
    if (argv[i] == key) return argv[i + 1];
  return {};
}

int main(int argc, char** argv) {
  std::cout << std::unitbuf;  // unbuffer; reads can block 5s+ on mailbox
  std::cerr << std::unitbuf;
  if (argc < 2) return usage(argv[0]);
  std::string mode = argv[1];

  std::string xclbin_path = get_arg(argc, argv, "--xclbin");
  if (xclbin_path.empty()) return usage(argv[0]);

  // Open device + xclbin + ctx.
  std::cout << "Opening xrt::device(0)...\n";
  auto device = xrt::device(0);

  std::cout << "Loading xclbin: " << xclbin_path << "\n";
  auto xclbin = xrt::xclbin(xclbin_path);
  device.register_xclbin(xclbin);

  std::cout << "Creating hw_context...\n";
  xrt::hw_context ctx(device, xclbin.get_uuid());

  std::cout << "force_connect (timeout 5s)...\n";
  ctx.force_connect(5000);
  std::cout << "force_connect OK\n";

  auto regs = build_reg_list();

  if (mode == "passive") {
    dump_regs(ctx, regs, "post-load idle (no kernel run)");
    return 0;
  }

  if (mode != "run-dma2") return usage(argv[0]);

  // ----- run-dma2 mode -----
  std::string instr_path  = get_arg(argc, argv, "--instr");
  std::string kernel_pref = get_arg(argc, argv, "--kernel");
  std::string len_str     = get_arg(argc, argv, "--length");
  std::string samp_str    = get_arg(argc, argv, "--samples");
  std::string iv_str      = get_arg(argc, argv, "--interval-ms");

  if (instr_path.empty()) return usage(argv[0]);
  if (kernel_pref.empty()) kernel_pref = "MLIR_AIE";
  int length     = len_str.empty()  ? 4096 : std::stoi(len_str);
  int n_samples  = samp_str.empty() ? 8    : std::stoi(samp_str);
  int interval   = iv_str.empty()   ? 500  : std::stoi(iv_str);

  // Find kernel by prefix.
  auto kernels = xclbin.get_kernels();
  auto kit = std::find_if(kernels.begin(), kernels.end(),
                          [&](const xrt::xclbin::kernel& k) {
                            return k.get_name().rfind(kernel_pref, 0) == 0;
                          });
  if (kit == kernels.end()) {
    std::cerr << "Kernel with prefix '" << kernel_pref << "' not found\n";
    return 3;
  }
  auto kernel_name = kit->get_name();
  std::cout << "Kernel: " << kernel_name << "\n";

  auto kernel = xrt::kernel(ctx, kernel_name);

  // Load instructions.
  std::ifstream f(instr_path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "Cannot open instr file: " << instr_path << "\n";
    return 4;
  }
  std::streamsize sz = f.tellg();
  f.seekg(0);
  std::vector<uint32_t> instr(static_cast<size_t>(sz / 4));
  f.read(reinterpret_cast<char*>(instr.data()), sz);
  std::cout << "Loaded " << instr.size() << " instr words\n";

  auto bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in    = xrt::bo(device, length * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out   = xrt::bo(device, length * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  auto* p_in = bo_in.map<int32_t*>();
  for (int i = 0; i < length; ++i) p_in[i] = i + 1;
  std::memcpy(bo_instr.map<void*>(), instr.data(),
              instr.size() * sizeof(uint32_t));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  dump_regs(ctx, regs, "pre-run");

  std::cout << "\nLaunching kernel (will hang on memtile DMA).\n";
  std::cout << "Make sure /sys/module/amdxdna/parameters/tdr_dump_ctx == Y\n"
            << "or the kernel will be reset before we can read.\n\n";

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr.size(), bo_in, bo_out);
  // Do NOT wait -- let it hang.

  for (int i = 0; i < n_samples; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    char tag[64];
    std::snprintf(tag, sizeof(tag), "t=%dms", (i + 1) * interval);
    dump_regs(ctx, regs, tag);
  }

  std::cout << "\nDone.  Kernel still hung.  Recover with:\n"
            << "  pkexec modprobe -r amdxdna && pkexec modprobe amdxdna\n";
  return 0;
}
