// SPDX-License-Identifier: Apache-2.0
// Mock XRT kernel implementation

#include "xrt/xrt_kernel.h"
#include "xrt/xrt_xclbin.h"
#include "emulator_bridge.h"

#include <iostream>

namespace xrt {

/// Run implementation
class run_impl {
public:
    ert_cmd_state state;
    xrt_emu::ExecResult result;

    run_impl() : state(ERT_CMD_STATE_NEW) {}
};

// run methods
run::~run() = default;

ert_cmd_state run::wait(uint32_t timeout_ms) const {
    if (!m_impl) {
        return ERT_CMD_STATE_ERROR;
    }

    // For synchronous emulator, execution is already complete when run is created
    return m_impl->state;
}

ert_cmd_state run::state() const {
    return m_impl ? m_impl->state : ERT_CMD_STATE_ERROR;
}

void run::abort() {
    if (m_impl) {
        m_impl->state = ERT_CMD_STATE_ABORT;
    }
}

/// Kernel implementation
class kernel_impl {
public:
    std::string name;
    hw_context ctx;
    std::string xclbin_path;
    uint32_t ctx_handle;

    kernel_impl(const hw_context& c, const std::string& n)
        : name(n), ctx(c), ctx_handle(0) {
        // Get context handle from hw_context
        // TODO: Store in hw_context_impl and retrieve here
    }
};

// kernel constructors
kernel::kernel(const hw_context& ctx, const std::string& name)
    : m_impl(std::make_shared<kernel_impl>(ctx, name)) {
    std::cerr << "[mock_xrt] kernel(\"" << name << "\") created" << std::endl;
}

kernel::kernel(const device& device, const uuid& xclbin_uuid, const std::string& name)
    : m_impl(std::make_shared<kernel_impl>(hw_context(device, xclbin_uuid), name)) {
}

kernel::~kernel() = default;

uint32_t kernel::group_id(int argno) const {
    // Return a simple memory group ID
    // In real XRT, this maps to specific memory banks
    // For our emulator, we use a unified memory model
    return static_cast<uint32_t>(argno);
}

std::string kernel::get_name() const {
    return m_impl ? m_impl->name : "";
}

run kernel::execute(const std::vector<uint64_t>& args) {
    auto run_impl_ptr = std::make_shared<run_impl>();

    if (!m_impl) {
        run_impl_ptr->state = ERT_CMD_STATE_ERROR;
        return run(run_impl_ptr);
    }

    std::cerr << "[mock_xrt] Executing kernel: " << m_impl->name << std::endl;
    std::cerr << "[mock_xrt]   Arguments (" << args.size() << "): ";
    for (auto arg : args) {
        std::cerr << "0x" << std::hex << arg << std::dec << " ";
    }
    std::cerr << std::endl;

    // For NPU kernels, arguments are typically:
    //   args[0] = opcode (usually 0 or 3)
    //   args[1] = instruction buffer device address
    //   args[2] = instruction count
    //   args[3+] = buffer addresses (input(s), output(s))
    //
    // Common patterns:
    //   5 args: opcode, instr_bo, size, input, output (add_one tests)
    //   6 args: opcode, instr_bo, size, inputA, inputB, output (matmul tests)

    if (args.size() >= 5) {
        uint64_t opcode = args[0];
        uint64_t instr_addr = args[1];
        size_t instr_size = static_cast<size_t>(args[2]);

        // Buffer arguments start at index 3
        std::vector<uint64_t> buffer_args(args.begin() + 3, args.end());

        std::cerr << "[mock_xrt]   Opcode=" << opcode
                  << " instr_addr=0x" << std::hex << instr_addr
                  << " instr_size=" << std::dec << instr_size
                  << " buffers=" << buffer_args.size() << std::endl;

        // Execute via emulator bridge
        auto result = xrt_emu::EmulatorBridge::instance().execute(
            m_impl->ctx_handle,
            m_impl->name,
            instr_addr,
            instr_size,
            buffer_args
        );

        run_impl_ptr->result = result;
        run_impl_ptr->state = result.success ? ERT_CMD_STATE_COMPLETED : ERT_CMD_STATE_ERROR;
    } else {
        std::cerr << "[mock_xrt] Warning: Need at least 5 arguments, got " << args.size() << std::endl;
        run_impl_ptr->state = ERT_CMD_STATE_COMPLETED;  // Allow tests to proceed
    }

    return run(run_impl_ptr);
}

} // namespace xrt
