"""Binary data model shared by the vector-verify generators and the stager.

A vector-verify kernel directory commits no data. The shared host program
(tests/vector-verify/common/vv_host.cpp) loads raw little-endian buffers from
files that tools/stage_vector_kernel.py writes next to the staged kernel:

    in0.bin, in1.bin, ...   the input buffers, byte-for-byte what the host
                            copies into the input BOs
    exp.bin                 the expected output (the silicon golden); absent
                            for a kernel that has not been captured yet

The values come from the generator's designed inputs and from the silicon
golden JSON in tools/golden/silicon/, which stays the single source of truth.
Each generator exposes a function returning a `KernelData`; this module holds
the packing and the knobs the generators put on the run.lit host compile line.
"""

import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence

# C element type -> (struct format, byte width). Little-endian, no padding.
CTYPES = {
    "int8_t": ("b", 1), "uint8_t": ("B", 1),
    "int16_t": ("h", 2), "uint16_t": ("H", 2),
    "int32_t": ("i", 4), "uint32_t": ("I", 4),
}

IN_PREFIX = "in"
EXP_NAME = "exp.bin"


def in_name(index):
    """File name of input buffer `index` (in0.bin, in1.bin, ...)."""
    return f"{IN_PREFIX}{index}.bin"


def pack(values: Sequence[int], ctype: str) -> bytes:
    """Pack integers as a little-endian array of `ctype`.

    Strict: a value outside the type's range raises, so a golden that does not
    fit its declared buffer type fails loudly instead of wrapping.
    """
    fmt, _ = CTYPES[ctype]
    return struct.pack(f"<{len(values)}{fmt}", *values)


def unpack(data: bytes, ctype: str) -> List[int]:
    """Inverse of `pack`."""
    fmt, width = CTYPES[ctype]
    assert len(data) % width == 0, f"{len(data)} bytes is not whole {ctype} elements"
    return list(struct.unpack(f"<{len(data) // width}{fmt}", data))


@dataclass(frozen=True)
class KernelData:
    """The runtime data of one kernel: input buffers and expected output.

    `expected` is None until a silicon capture exists (bootstrap); the host then
    runs, dumps out.txt and fails by design.
    """

    inputs: List[bytes]
    expected: Optional[bytes]


def host_flags(n_inputs: int, out_ctype: str, n_out: int) -> str:
    """The -D knobs vv_host.cpp reads, as they appear on the run.lit clang line.

    VV_DATA_DIR is a string literal; the lit line is run through `bash -c`, so
    the literal's quotes are protected by an outer pair of single quotes. `%S`
    is replaced by the bridge with the staged kernel directory, where the .bin
    files live.
    """
    assert out_ctype in CTYPES, out_ctype
    return (f"-DVV_DATA_DIR='\"%S\"' -DVV_NIN={n_inputs} "
            f"-DVV_OUT_T={out_ctype} -DVV_NOUT={n_out}")


def write_data(directory, data: KernelData):
    """Write `data` into `directory` as in<N>.bin and exp.bin (if captured)."""
    import os
    os.makedirs(directory, exist_ok=True)
    for i, buf in enumerate(data.inputs):
        with open(os.path.join(directory, in_name(i)), "wb") as f:
            f.write(buf)
    exp_path = os.path.join(directory, EXP_NAME)
    if data.expected is not None:
        with open(exp_path, "wb") as f:
            f.write(data.expected)
    elif os.path.exists(exp_path):
        os.remove(exp_path)
