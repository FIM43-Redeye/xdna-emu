// SPDX-License-Identifier: Apache-2.0
// Mock XRT Experimental Kernel header for xdna-emu
// Provides runlist class for batched kernel execution

#ifndef MOCK_XRT_EXPERIMENTAL_KERNEL_H_
#define MOCK_XRT_EXPERIMENTAL_KERNEL_H_

#include "xrt/xrt_kernel.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/detail/config.h"
#include "xrt/detail/pimpl.h"
#include "xrt/detail/ert.h"

#include <chrono>
#include <condition_variable>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace xrt {

/// Runlist implementation (internal)
class runlist_impl;

/// xrt::runlist manages a list of xrt::run objects for atomic execution
///
/// Run objects are added using add() and executed atomically in order
/// using execute(). The list can be reused after completion.
class runlist : public detail::pimpl<runlist_impl> {
public:
    /// Exception for abnormal runlist execution
    class command_error_impl;
    class command_error : public detail::pimpl<command_error_impl>, public std::exception {
    public:
        XRT_API_EXPORT
        command_error(const xrt::run& run, ert_cmd_state state, const std::string& what);

        /// Get the run object that failed
        XRT_API_EXPORT
        xrt::run get_run() const;

        /// Get the command state at failure
        XRT_API_EXPORT
        ert_cmd_state get_command_state() const;

        XRT_API_EXPORT
        const char* what() const noexcept override;

    private:
        std::string m_what;
    };

    /// AIE-specific error with health information
    class aie_error : public command_error {
    public:
        using command_error::command_error;

        XRT_API_EXPORT
        aie_error(const xrt::run& run, ert_cmd_state state, const std::string& what);

        /// Get raw context health data
        XRT_API_EXPORT
        detail::span<const uint32_t> data() const;
    };

public:
    /// Default constructor - creates invalid runlist
    runlist() = default;

    /// Construct runlist for a hardware context
    ///
    /// @param hwctx  Hardware context for kernel execution
    ///
    /// All run objects added must use kernels from this context.
    XRT_API_EXPORT
    explicit runlist(const xrt::hw_context& hwctx);

    /// Destructor
    XRT_API_EXPORT
    ~runlist();

    /// Add a run object to the list
    ///
    /// @param run  Run object to add
    ///
    /// The run is added to the end of the list. Cannot add while executing.
    XRT_API_EXPORT
    void add(const xrt::run& run);

    /// Move a run object into the list
    XRT_API_EXPORT
    void add(xrt::run&& run);

    /// Execute the runlist
    ///
    /// Run objects execute atomically in order. Empty list is a no-op.
    XRT_API_EXPORT
    void execute();

    /// Wait for runlist completion
    ///
    /// @param timeout  Wait timeout (0 = infinite)
    /// @return cv_status::no_timeout if completed, timeout otherwise
    ///
    /// Throws command_error if any run fails.
    XRT_API_EXPORT
    std::cv_status wait(const std::chrono::milliseconds& timeout) const;

    /// Wait for runlist completion (infinite timeout)
    void wait() const {
        wait(std::chrono::milliseconds(0));
    }

    /// Get current state of the runlist
    ///
    /// @return Current command state
    XRT_API_EXPORT
    ert_cmd_state state() const;

    /// Poll for completion (deprecated, use state())
    ///
    /// @return 0 if still running, non-zero if complete
    XRT_API_EXPORT
    int poll() const;

    /// Reset the runlist
    ///
    /// Removes all run objects. Cannot reset while executing.
    XRT_API_EXPORT
    void reset();

private:
    std::vector<xrt::run> m_runs;
    xrt::hw_context m_hwctx;
    bool m_executing = false;
};

} // namespace xrt

#endif // MOCK_XRT_EXPERIMENTAL_KERNEL_H_
