"""Locate aiecc's post-lowering MLIR (``input_with_addresses.mlir``).

The C++ aiecc builds only the artifacts it is asked for. Without
``--get-input-with-addresses`` the file is not written (the ``<src>.prj``
scratch dir no longer holds it), so a build made with just ``--get-xclbin``
has nothing for parse-trace's ``--xclbin-mlir`` to read. With the flag the
file lands directly in ``--output-dir`` (default: the working directory),
beside the xclbin, not inside ``.prj``.
"""

from pathlib import Path

GET_INPUT_WITH_ADDRESSES = "--get-input-with-addresses"
FILENAME = "input_with_addresses.mlir"


class MissingInputWithAddresses(FileNotFoundError):
    """No input_with_addresses.mlir where a build should have put it."""


def find_input_with_addresses(build_dir, source_name="aie_arch.mlir"):
    """The file for a build directory, or None.

    Checks where the flag puts it (``build_dir`` itself), then the layout of
    older aiecc builds (``build_dir/<source_name>.prj``).
    """
    build_dir = Path(build_dir)
    for candidate in (build_dir / FILENAME,
                      build_dir / f"{source_name}.prj" / FILENAME):
        if candidate.is_file():
            return candidate
    return None


def require_input_with_addresses(build_dir, source_name="aie_arch.mlir"):
    """Like find_input_with_addresses, but raise with the fix instead of None."""
    found = find_input_with_addresses(build_dir, source_name)
    if found is None:
        raise MissingInputWithAddresses(
            f"{FILENAME} not found in {build_dir} (or its {source_name}.prj). "
            f"aiecc writes it only when asked: rebuild with "
            f"`aiecc {GET_INPUT_WITH_ADDRESSES} ...` (it lands in --output-dir "
            f"next to the xclbin).")
    return found
