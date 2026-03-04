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

// tcp_server and dbg_hwq_umq destructor stubs -- hwctx_umq holds a
// unique_ptr<tcp_server> whose destructor chain needs the complete types.
// The real implementations live in umq/ which we do not compile.
dbg_hwq_umq::~dbg_hwq_umq() = default;
tcp_server::~tcp_server() = default;

} // namespace shim_xdna
