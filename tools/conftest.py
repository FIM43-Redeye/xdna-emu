"""pytest configuration for the tools/ test suite.

Adds the amd-unified-software Python site-packages to sys.path so that
clang.cindex is importable without requiring PYTHONPATH to be set before
invoking pytest.  The path is appended (not prepended) to avoid shadowing
any system packages that tests may depend on.

The clang Python bindings ship with amd-unified-software and are not
available from PyPI or the system package manager in this environment.
"""

import sys
from pathlib import Path

_CLANG_SITE = Path("/home/triple/npu-work/amd-unified-software/tps/lnx64"
                   "/python-3.13.0/lib/python3.13/site-packages")

if _CLANG_SITE.is_dir() and str(_CLANG_SITE) not in sys.path:
    sys.path.append(str(_CLANG_SITE))
