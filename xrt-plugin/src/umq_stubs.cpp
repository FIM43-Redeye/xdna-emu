// SPDX-License-Identifier: MIT
//
// umq_stubs.cpp -- Stub constructors for UMQ hardware contexts.
//
// device.cpp references hwctx_umq constructors in its create_hw_context()
// factory method.  We do not compile the UMQ backend (it depends on host
// queues and other hardware infrastructure).  These stubs satisfy the
// linker; the emulator always takes the KMQ path (is_umq() == false).

#include "shim/buffer.h"
#include "shim/hwq.h"
#include "shim/umq/hwctx.h"
#include "shim/shim_debug.h"

namespace shim_xdna {

// -- UMQ stubs ---------------------------------------------------------------

hwctx_umq::
hwctx_umq(const device& dev, const xrt::xclbin& xclbin, const qos_type& qos)
  : hwctx(dev, qos, xclbin, nullptr)
  , m_pdev(dev.get_pdev())
{
  shim_err(ENOTSUP, "hwctx_umq is not available in emulation mode");
}

hwctx_umq::
hwctx_umq(const device& dev, uint32_t partition_size)
  : hwctx(dev, partition_size, nullptr)
  , m_pdev(dev.get_pdev())
{
  shim_err(ENOTSUP, "hwctx_umq is not available in emulation mode");
}

hwctx_umq::~hwctx_umq() = default;

// (Upstream removed the AIE-debugger TCP server: commit "remove cert aie
// debugger support" deleted dbg_hwq_umq + tcp_server and the
// unique_ptr<tcp_server> member of hwctx_umq, so the destructor-chain stubs
// they used to require are gone with them.)

} // namespace shim_xdna
