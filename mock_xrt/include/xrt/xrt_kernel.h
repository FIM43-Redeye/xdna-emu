// SPDX-License-Identifier: Apache-2.0
// Mock XRT Kernel header for xdna-emu
// Provides API-compatible interfaces for testing against the emulator

#ifndef MOCK_XRT_KERNEL_H_
#define MOCK_XRT_KERNEL_H_

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/detail/ert.h"
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace xrt {

// Forward declarations
class xclbin;

/// Run implementation (internal)
class run_impl;

// Forward declaration
class kernel;

/// xrt::run represents one execution of a kernel
class run {
public:
    /// Default constructor - creates invalid run
    run() = default;

    /// Construct from implementation
    explicit run(std::shared_ptr<run_impl> impl) : m_impl(std::move(impl)) {}

    /// Construct from kernel (creates a run ready to execute)
    explicit run(const kernel& k);

    /// Copy constructor
    run(const run&) = default;

    /// Move constructor
    run(run&&) = default;

    /// Destructor
    ~run();

    /// Copy assignment
    run& operator=(const run&) = default;

    /// Move assignment
    run& operator=(run&&) = default;

    /// Wait for kernel execution to complete
    ///
    /// @param timeout_ms  Timeout in milliseconds (0 = infinite)
    /// @return Command state after completion
    ert_cmd_state wait(uint32_t timeout_ms = 0) const;

    /// Wait for kernel execution to complete (chrono version)
    ///
    /// @param timeout  Timeout duration
    /// @return Command state after completion
    template<typename Duration>
    ert_cmd_state wait(const Duration& timeout) const {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(timeout);
        return wait(static_cast<uint32_t>(ms.count()));
    }

    /// Get the current command state
    ert_cmd_state state() const;

    /// Abort the running kernel
    void abort();

    /// Wait for completion (alias for wait with no timeout)
    /// Used by some tests instead of wait()
    void wait2() const { wait(static_cast<uint32_t>(0)); }

    /// Set a kernel argument (integer types)
    ///
    /// @param index  Argument index
    /// @param value  Argument value
    template<typename T>
    typename std::enable_if<std::is_integral<typename std::decay<T>::type>::value>::type
    set_arg(int index, T value) {
        set_arg_impl(index, static_cast<uint64_t>(value));
    }

    /// Set buffer argument (lvalue reference)
    void set_arg(int index, const bo& buf);

    /// Set buffer argument (rvalue reference)
    void set_arg(int index, bo&& buf) {
        set_arg(index, static_cast<const bo&>(buf));
    }

    /// Start kernel execution
    void start();

    /// Check if run is valid
    explicit operator bool() const { return m_impl != nullptr; }

private:
    void set_arg_impl(int index, uint64_t value);

    std::shared_ptr<run_impl> m_impl;
};

/// Kernel implementation (internal)
class kernel_impl;

/// xrt::kernel represents a kernel that can be executed
class kernel {
public:
    /// Default constructor - creates invalid kernel
    kernel() = default;

    /// Construct kernel from hw_context and kernel name
    ///
    /// @param ctx   Hardware context
    /// @param name  Kernel name in xclbin
    kernel(const hw_context& ctx, const std::string& name);

    /// Construct kernel from device, UUID, and kernel name (legacy)
    ///
    /// @param device  Device to run kernel on
    /// @param xclbin_uuid  UUID of loaded xclbin
    /// @param name  Kernel name in xclbin
    kernel(const device& device, const uuid& xclbin_uuid, const std::string& name);

    /// Copy constructor
    kernel(const kernel&) = default;

    /// Move constructor
    kernel(kernel&&) = default;

    /// Destructor
    ~kernel();

    /// Copy assignment
    kernel& operator=(const kernel&) = default;

    /// Move assignment
    kernel& operator=(kernel&&) = default;

    /// Get memory group ID for kernel argument
    ///
    /// @param argno  Argument index
    /// @return Memory group ID for buffer allocation
    uint32_t group_id(int argno) const;

    /// Get kernel name
    std::string get_name() const;

    /// Execute kernel with arguments
    ///
    /// This is the variadic template that handles arbitrary arguments.
    /// For NPU kernels, typical signature is:
    ///   kernel(opcode, bo_instr, instr_size, bo_inA, bo_inB, bo_out)
    ///
    /// @param args  Kernel arguments
    /// @return Run object for tracking execution
    template<typename... Args>
    run operator()(Args&&... args) {
        // Pack arguments into a vector for the implementation
        std::vector<uint64_t> arg_values;
        pack_args(arg_values, std::forward<Args>(args)...);
        return execute(arg_values);
    }

    /// Check if kernel is valid
    explicit operator bool() const { return m_impl != nullptr; }

    /// Make run a friend so it can access m_impl
    friend class run;

private:
    // Base case for argument packing
    void pack_args(std::vector<uint64_t>&) {}

    // Pack integer arguments
    template<typename T, typename... Rest>
    typename std::enable_if<std::is_integral<T>::value>::type
    pack_args(std::vector<uint64_t>& args, T val, Rest&&... rest) {
        args.push_back(static_cast<uint64_t>(val));
        pack_args(args, std::forward<Rest>(rest)...);
    }

    // Pack buffer object arguments
    template<typename... Rest>
    void pack_args(std::vector<uint64_t>& args, const bo& buf, Rest&&... rest) {
        args.push_back(buf.address());
        pack_args(args, std::forward<Rest>(rest)...);
    }

    // Execute with packed arguments
    run execute(const std::vector<uint64_t>& args);

    std::shared_ptr<kernel_impl> m_impl;
};

} // namespace xrt

#endif // MOCK_XRT_KERNEL_H_
