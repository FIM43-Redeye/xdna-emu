// SPDX-License-Identifier: MIT
//
// hwctx_stubs.cpp -- Stub constructors for kmq/umq hardware contexts.
//
// device.cpp references hwctx_kmq and hwctx_umq constructors in its
// create_hw_context() factory method.  We do not compile the real
// kmq/ or umq/ backends (they depend on kernel DRM, UMQ host queues,
// and other hardware infrastructure we replace).  These stubs satisfy
// the linker; they will never be called at runtime because the
// emulator intercepts execution at the platform_drv level.

// Include buffer.h first so that shim_xdna::buffer is visible when
// kmq/hwctx.h uses the unqualified name.
#include "shim/buffer.h"
#include "shim/hwq.h"
#include "shim/kmq/hwctx.h"
#include "shim/umq/hwctx.h"
#include "shim/shim_debug.h"

namespace shim_xdna {

// -- KMQ stubs ---------------------------------------------------------------

hwctx_kmq::
hwctx_kmq(const device& dev, const xrt::xclbin& xclbin, const qos_type& qos)
  : hwctx(dev, qos, xclbin, nullptr)
{
  shim_err(ENOTSUP, "hwctx_kmq is not available in emulation mode");
}

hwctx_kmq::
hwctx_kmq(const device& dev, uint32_t partition_size)
  : hwctx(dev, partition_size, nullptr)
{
  shim_err(ENOTSUP, "hwctx_kmq is not available in emulation mode");
}

hwctx_kmq::~hwctx_kmq() = default;

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
