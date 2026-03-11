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

    def test_group_id_from_arg_count(self):
        """Without trace_arg_index, group_id = call arg count (position)."""
        result = patch_test_cpp(MINIMAL_CPP)
        # kernel(opcode, bo_instr, 256, bo_in, bo_out) has 5 args,
        # so bo_trace goes at position 5 -> group_id(5)
        assert "kernel.group_id(5)" in result

    def test_group_id_from_trace_arg_index(self):
        """With trace_arg_index, group_id uses the exact value."""
        result = patch_test_cpp(MINIMAL_CPP, trace_arg_index=6)
        assert "kernel.group_id(6)" in result

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


class TestTraceArgIndex:
    """trace_arg_index from xclbin metadata controls group_id and set_arg."""

    def test_ext_kernel_patched_with_correct_group_id(self):
        """ext::kernel tests work when trace_arg_index is provided.

        Previously these were skipped because the heuristic defaulted to
        group_id(1) (SRAM bank).  With trace_arg_index, group_id matches
        the xclbin connectivity (HOST bank).
        """
        src = MINIMAL_CPP.replace(
            'auto kernel = xrt::kernel(context, "test");',
            'auto kernel = xrt::ext::kernel(context, mod, "test");',
        )
        result = patch_test_cpp(src, trace_arg_index=6)
        assert "kernel.group_id(6)" in result
        assert "auto bo_trace = xrt::bo(" in result

    def test_set_arg_uses_trace_arg_index(self):
        """set_arg pattern uses trace_arg_index instead of max+1."""
        src = """\
#include <iostream>
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
int main() {
  auto device = xrt::device(0);
  auto kernel = xrt::kernel(context, "test");
  auto bo_out = xrt::bo(device, 256,
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  xrt::run run0 = xrt::run(kernel);
  run0.set_arg(0, 3);
  run0.set_arg(3, bo_out);
  run0.set_arg(4, 0);
  run0.set_arg(5, 0);
  run0.set_arg(6, 0);
  run0.set_arg(7, 0);
  run0.wait();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  return 0;
}
"""
        # trace_arg_index=5 means trace goes at set_arg(5), overriding
        # the padding set_arg(5, 0) that came earlier.
        result = patch_test_cpp(src, trace_arg_index=5)
        assert "run0.set_arg(5, bo_trace);" in result
        assert "kernel.group_id(5)" in result


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
    """xrt::ext::kernel tests work with trace_arg_index from xclbin metadata.

    Without trace_arg_index, the fallback uses call arg count which maps to
    the correct HOST bank position.  With trace_arg_index, the exact kernel
    arg slot is used.
    """

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

    def test_ext_kernel_patched_with_arg_index(self):
        """ext::kernel patched correctly with trace_arg_index from xclbin."""
        result = patch_test_cpp(self.EXT_CPP, trace_arg_index=6)
        assert "kernel.group_id(6)" in result
        assert "bo_out, bo_trace)" in result

    def test_ext_kernel_fallback_uses_arg_count(self):
        """Without trace_arg_index, falls back to call arg count."""
        result = patch_test_cpp(self.EXT_CPP)
        # kernel(opcode, 0, 0, bo_inA, bo_out) = 5 args -> group_id(5)
        assert "kernel.group_id(5)" in result
