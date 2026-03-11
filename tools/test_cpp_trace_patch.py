"""Tests for cpp_trace_patch -- tree-sitter C++ transform module.

Validates the three transforms applied by patch_test_cpp():
1. Trace buffer allocation (after last xrt::bo declaration)
2. Kernel call argument (append bo_trace)
3. Trace write-out (after last run.wait())

Plus edge cases: already-traced skip, missing insertion points, fstream include.
"""

import pytest
from cpp_trace_patch import patch_test_cpp, PatchError


# ---------------------------------------------------------------------------
# Minimal test.cpp that exercises all three transforms.
# Uses the standard mlir-aie/test/npu-xrt pattern.
# ---------------------------------------------------------------------------
MINIMAL_CPP = """\
#include <cstdint>
#include <iostream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

int main(int argc, const char *argv[]) {
  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin("test.xclbin");
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, "test");

  auto bo_instr = xrt::bo(device, 1024,
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in = xrt::bo(device, 256,
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, 256,
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  void *bufInstr = bo_instr.map<void *>();
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, 256, bo_in, bo_out);
  ert_cmd_state r = run.wait();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  return 0;
}
"""


class TestTraceBufferAllocation:
    """Transform 1: Trace buffer inserted after last xrt::bo declaration."""

    def test_trace_bo_inserted_after_last_bo(self):
        result = patch_test_cpp(MINIMAL_CPP)
        # The trace buffer declaration must appear after the last xrt::bo line
        assert "auto bo_trace = xrt::bo(" in result
        # And it must appear before the map<> call
        trace_pos = result.index("auto bo_trace = xrt::bo(")
        map_pos = result.index("void *bufInstr = bo_instr.map<void *>()")
        assert trace_pos < map_pos

    def test_next_group_id_computed(self):
        result = patch_test_cpp(MINIMAL_CPP)
        # Existing group_ids are 1, 3, 5 -> max is 5, next is 6
        assert "kernel.group_id(6)" in result

    def test_next_group_id_dense(self):
        """Dense group_id allocation (0, 1, 2) -> next is 3."""
        cpp = MINIMAL_CPP.replace("group_id(1)", "group_id(0)")
        cpp = cpp.replace("group_id(3)", "group_id(1)")
        cpp = cpp.replace("group_id(5)", "group_id(2)")
        result = patch_test_cpp(cpp)
        assert "kernel.group_id(3)" in result

    def test_trace_size_default(self):
        result = patch_test_cpp(MINIMAL_CPP)
        assert "constexpr size_t trace_size = 1048576;" in result

    def test_trace_size_custom(self):
        result = patch_test_cpp(MINIMAL_CPP, trace_size=2097152)
        assert "constexpr size_t trace_size = 2097152;" in result

    def test_trace_bo_has_sync(self):
        result = patch_test_cpp(MINIMAL_CPP)
        assert "bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);" in result

    def test_trace_bo_has_memset(self):
        result = patch_test_cpp(MINIMAL_CPP)
        assert 'memset(bo_trace.map<void*>(), 0, trace_size);' in result


class TestKernelCallArgument:
    """Transform 2: bo_trace appended as last argument to kernel() call."""

    def test_bo_trace_appended(self):
        result = patch_test_cpp(MINIMAL_CPP)
        # The kernel call should now end with bo_trace as last arg
        assert "bo_in, bo_out, bo_trace)" in result

    def test_original_args_preserved(self):
        result = patch_test_cpp(MINIMAL_CPP)
        assert "kernel(opcode, bo_instr, 256, bo_in, bo_out, bo_trace)" in result


class TestTraceWriteOut:
    """Transform 3: Trace data written after last run.wait()."""

    def test_trace_writeout_inserted(self):
        result = patch_test_cpp(MINIMAL_CPP)
        assert 'std::getenv("XDNA_TRACE_DIR")' in result
        assert "bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);" in result
        assert "std::ofstream trace_file(" in result

    def test_trace_writeout_after_wait(self):
        result = patch_test_cpp(MINIMAL_CPP)
        wait_pos = result.index("run.wait()")
        trace_pos = result.index('std::getenv("XDNA_TRACE_DIR")')
        assert trace_pos > wait_pos

    def test_trace_writeout_before_sync(self):
        """Trace write-out should appear after wait but before bo_out.sync."""
        result = patch_test_cpp(MINIMAL_CPP)
        trace_pos = result.index('std::getenv("XDNA_TRACE_DIR")')
        # The bo_out.sync should come after the trace block
        # (it was already after run.wait(), the trace block is inserted between)
        out_sync_pos = result.index("bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE)")
        assert trace_pos < out_sync_pos


class TestFstreamInclude:
    """#include <fstream> added when missing."""

    def test_fstream_added_when_missing(self):
        src = MINIMAL_CPP  # does not have <fstream>
        assert "#include <fstream>" not in src
        result = patch_test_cpp(src)
        assert "#include <fstream>" in result

    def test_fstream_not_duplicated(self):
        src = '#include <fstream>\n' + MINIMAL_CPP
        result = patch_test_cpp(src)
        assert result.count("#include <fstream>") == 1


class TestAlreadyTracedSkip:
    """Already-patched files should be returned unchanged."""

    def test_skip_if_already_injected(self):
        """Files containing our injection marker are returned as-is."""
        src = "// injected by trace-prepare.py\n" + MINIMAL_CPP
        result = patch_test_cpp(src)
        assert result == src  # returned unchanged

    def test_legacy_trace_size_still_patched(self):
        """Files with a legacy trace_size variable ARE patched (not skipped).

        The old guard skipped any file containing 'trace_size'. Now that the
        bridge strips --trace_sz from test.exe commands, these tests need our
        injection like any other.
        """
        src = "constexpr size_t trace_size = 1048576;\n" + MINIMAL_CPP
        result = patch_test_cpp(src)
        assert "auto bo_trace = xrt::bo(" in result  # patched, not skipped


class TestPatchError:
    """PatchError raised when insertion points cannot be found."""

    def test_error_no_bo_declarations(self):
        src = """\
#include <iostream>
int main() {
  auto kernel = xrt::kernel(context, "test");
  auto run = kernel(3);
  run.wait();
  return 0;
}
"""
        with pytest.raises(PatchError, match="xrt::bo"):
            patch_test_cpp(src)

    def test_error_no_kernel_call(self):
        src = """\
#include <iostream>
int main() {
  auto device = xrt::device(0);
  auto kernel = xrt::kernel(context, "test");
  auto bo_out = xrt::bo(device, 256,
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  return 0;
}
"""
        # No kernel() invocation and no wait() -- hits wait error first.
        with pytest.raises(PatchError, match="wait"):
            patch_test_cpp(src)

    def test_error_no_run_wait(self):
        src = """\
#include <iostream>
int main() {
  auto device = xrt::device(0);
  auto kernel = xrt::kernel(context, "test");
  auto bo_out = xrt::bo(device, 256,
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto run = kernel(3, bo_out);
  return 0;
}
"""
        with pytest.raises(PatchError, match="run.wait"):
            patch_test_cpp(src)


class TestExtKernelPattern:
    """Tests using xrt::ext::kernel (no group_id on BOs)."""

    EXT_CPP = """\
#include <cstdint>
#include <iostream>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_ext.h"

int main(int argc, const char *argv[]) {
  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin("test.xclbin");
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::ext::kernel(context, mod, "test");

  xrt::bo bo_inA = xrt::ext::bo{device, 256};
  xrt::bo bo_out = xrt::ext::bo{device, 256};

  unsigned int opcode = 3;
  auto run = kernel(opcode, 0, 0, bo_inA, bo_out);
  run.wait2();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  return 0;
}
"""

    def test_ext_kernel_bo_trace_inserted(self):
        result = patch_test_cpp(self.EXT_CPP)
        assert "auto bo_trace = xrt::bo(" in result

    def test_ext_kernel_no_group_id_uses_default(self):
        """When no group_id calls exist, use a reasonable default (1)."""
        result = patch_test_cpp(self.EXT_CPP)
        # ext::bo has no group_id, so the trace bo should use group_id(1)
        # (default fallback since no existing group_ids to max over)
        assert "kernel.group_id(" in result

    def test_ext_kernel_call_patched(self):
        result = patch_test_cpp(self.EXT_CPP)
        assert "bo_out, bo_trace)" in result

    def test_ext_kernel_wait2_handled(self):
        result = patch_test_cpp(self.EXT_CPP)
        # run.wait2() should be found as the wait point
        assert 'std::getenv("XDNA_TRACE_DIR")' in result
