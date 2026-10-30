#!/usr/bin/env python3
"""Spec-driven generator for vector-compute capture kernels.

Each AIE2 vector-compute class (SRS, UPS, Pack, ...) needs a bridge kernel that
exercises the *actual* vector intrinsic end-to-end (shim DMA -> core vector op
-> shim DMA) so a bridge run can confirm the emulator matches real NPU1 silicon.
The files such a kernel needs -- run.lit, aie.mlir, kernel.cc -- are ~80%
boilerplate; the genuinely variable parts are the intrinsic body (the IP,
hand-written once per class) and the input/expected data.

Inputs are designed by us (tools/vector_inputs.py: class-specific edge cases plus
a seeded PRNG fill). Expected values are observed on NPU1 (Phoenix) silicon:
tools/capture_silicon.py stores a HW run's output in
tools/golden/silicon/<kernel>.json. No data is committed in a kernel directory:
`kernel_data` packs the inputs and the silicon golden into raw buffers, and
tools/stage_vector_kernel.py writes them (as in<N>.bin / exp.bin) next to the
shared host program tests/vector-verify/common/vv_host.cpp when it stages the
kernel. Until a capture exists (bootstrap) there is no exp.bin: the first HW run
fails loudly while still dumping out.txt for capture. A few edge kernels have no
designed inputs in their spec: their inputs live only in their silicon JSON.
"""


import string
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import vv_data


@dataclass(frozen=True)
class Buf:
    """One DMA-connected buffer.

    `ctype` is the host-staging C type used for the baked golden array and the
    XRT buffer (so exact bit patterns move byte-for-byte over DMA). `mlir` is
    the MLIR element type. `ktype` is the kernel-signature C type when it
    differs from `ctype` -- e.g. the f32->bf16 conv kernel stages uint32/uint16
    bit patterns on the host but the kernel reads `float`/`bfloat16`. When
    `ktype` is unset, the kernel uses the host type.
    """

    name: str
    ctype: str
    mlir: str
    ktype: Optional[str] = None

    @property
    def kernel_ctype(self):
        """C type seen in the kernel signature (falls back to the host type)."""
        return self.ktype if self.ktype else self.ctype


@dataclass(frozen=True)
class Matmul:
    """Native AIE2 mmul tile geometry + batch for a matmul capture kernel.

    M,K,N are the tile dims (rows x inner x cols), all native AIE2 mmul tiles.
    a_bytes/b_bytes are the element widths of A and B. `batch` independent tiles
    are concatenated into one DMA buffer set so the kernel loops over many
    multiplies. `bfloat` selects raw-bits unpacking (bf16 inputs, fp32 output);
    otherwise A/B unpack as signed integers and C is int32.
    """

    M: int
    K: int
    N: int
    a_bytes: int
    b_bytes: int
    batch: int
    bfloat: bool = False

    @property
    def size_a(self):
        return self.M * self.K

    @property
    def size_b(self):
        return self.K * self.N

    @property
    def size_c(self):
        return self.M * self.N


@dataclass(frozen=True)
class DirectIO:
    """Designed input arrays for a kernel: one tuple per input buffer.

    Elementwise kernels carry one tuple (host-staged bit patterns); matmul
    kernels carry (A, B) flat row-major tile batches. Expected values are not
    part of the spec: they come from the silicon capture.
    """

    inputs: Tuple[Tuple[int, ...], ...]


@dataclass
class KernelSpec:
    """Everything needed to emit one vector-compute capture kernel.

    `io_class` labels the operation class in the silicon JSON ("srs", "ups",
    "pack", "convert", "matmul", ...). `direct` holds the designed inputs; when
    None the inputs are read from the silicon JSON (edge kernels whose inputs
    are kept as originally captured). `body` is the hand-written intrinsic body
    (the IP), wrapped by the generated kernel.cc scaffold.
    """

    name: str
    func: str
    doc: str
    inputs: List[Buf]
    output: Buf
    n: int
    io_class: str
    body: str
    direct: Optional[DirectIO] = None
    defines: List[Tuple[str, object]] = field(default_factory=list)
    stem: Optional[str] = None
    matmul: Optional[Matmul] = None
    silicon_golden: Optional[str] = None

    @property
    def n_out(self):
        """Number of output elements."""
        if self.matmul is not None:
            return self.matmul.batch * self.matmul.size_c
        return self.n

    @property
    def silicon_path(self):
        return self.silicon_golden or _silicon_path(self.name)


# --- Mode sweep --------------------------------------------------------------
#
# AIE2 vector-compute classes that round (SRS, bf16 convert) read crRnd; classes
# that narrow with overflow (SRS, Pack, UPS) read crSat. A single capture kernel
# bakes ONE mode (one set_rounding/set_saturation in the body). The silicon check
# is *mode-exhaustive*: one kernel per reachable mode-point, so every crRnd/crSat
# value is exercised on HW, not just one representative -- the same axis the
# pack-saturation gap hid behind.
#
# The crRnd index -> aie::rounding_mode enum name and crSat index -> enum name
# maps below follow the aie_api saturation_mode / rounding_mode enums
# (saturation none=0, saturate=1, symmetric=3) and match the emulator's
# RoundingMode (crates/xdna-archspec/.../rounding.rs). Which mode a kernel runs
# under is fixed by its name suffix; the observed outputs are what silicon does
# under that mode.

ROUNDING_ENUM = {
    0: "floor", 1: "ceil", 2: "symmetric_floor", 3: "symmetric_ceil",
    8: "negative_inf", 9: "positive_inf", 10: "symmetric_zero",
    11: "symmetric_inf", 12: "conv_even", 13: "conv_odd",
}

SAT_ENUM = {0: "none", 1: "saturate", 3: "symmetric"}

# crRnd values the aie_api enum does not name (llvm-aie leaves them undefined).
# They are driven by casting the raw index, to record what silicon does there.
UNDEFINED_RND = (4, 5, 6, 7, 14, 15)


def rounding_expr(rnd):
    """C++ expression for crRnd index `rnd` (named enumerator, else a raw cast)."""
    if rnd in ROUNDING_ENUM:
        return f"aie::rounding_mode::{ROUNDING_ENUM[rnd]}"
    return f"static_cast<aie::rounding_mode>({rnd})"


def mode_lines(rnd, crsat):
    """Render the set_rounding / set_saturation lines for a mode-point.

    Emits set_rounding only when `rnd` is given (the class rounds) and
    set_saturation only when `crsat` is given (the class can overflow), so each
    class's `$mode` placeholder gets exactly the control-register writes it uses.
    """
    out = []
    if rnd is not None:
        out.append(f"  ::aie::set_rounding({rounding_expr(rnd)});")
    if crsat is not None:
        out.append(f"  ::aie::set_saturation(aie::saturation_mode::{SAT_ENUM[crsat]});")
    return "\n".join(out)


@dataclass
class SweepSpec:
    """A family of capture kernels covering a class's full crRnd/crSat space.

    `expand()` yields one `KernelSpec` per point in the cartesian product of
    `rnds` x `sats`. Each point derives its kernel name suffix (`_r<rnd>_s<sat>`)
    and its body (the mode-setting lines substituted into `body_template`'s
    `$mode` placeholder). All points share the same designed inputs, so results
    across modes are directly comparable. `direct=None` means the inputs come
    from each point's silicon JSON.

    `body_template` is the hand-written intrinsic IP with a `$mode` marker where
    the control-register writes belong (and `event0()`/`event1()` around it).
    `rnds`/`sats` are the crRnd/crSat index lists to sweep (None => that axis is
    fixed/absent for the class).
    """

    prefix: str
    func: str
    doc: str
    inputs: List[Buf]
    output: Buf
    n: int
    io_class: str
    body_template: str
    direct: Optional[DirectIO] = None
    rnds: Optional[List[int]] = None
    sats: Optional[List[int]] = None
    defines: List[Tuple[str, object]] = field(default_factory=list)
    stem: Optional[str] = None
    key: Optional[str] = None   # registry key when the prefix is shared by sweeps

    def expand(self):
        """Materialize the sweep into one KernelSpec per (rnd, crsat) point."""
        rnds = self.rnds if self.rnds is not None else [None]
        sats = self.sats if self.sats is not None else [None]
        specs = []
        for rnd in rnds:
            for crsat in sats:
                suffix = ""
                desc = []
                if rnd is not None:
                    suffix += f"_r{rnd}"
                    desc.append(f"rnd={rnd}({ROUNDING_ENUM.get(rnd, 'undefined')})")
                if crsat is not None:
                    suffix += f"_s{crsat}"
                    desc.append(f"sat={crsat}({SAT_ENUM[crsat]})")
                body = string.Template(self.body_template).substitute(
                    mode=mode_lines(rnd, crsat))
                specs.append(KernelSpec(
                    name=self.prefix + suffix,
                    func=self.func,
                    doc=self.doc + " Mode-sweep point: " + ", ".join(desc) + ".",
                    inputs=self.inputs,
                    output=self.output,
                    n=self.n,
                    io_class=self.io_class,
                    body=body,
                    direct=self.direct,
                    defines=self.defines,
                    stem=self.stem,
                ))
        return specs


def _stem(spec):
    """Basename for the kernel's .cc/.o (e.g. "srs" from func "srs_i32")."""
    return spec.stem if spec.stem else spec.func.split("_")[0]


def _silicon_path(name):
    """Path to the HW-captured silicon golden for a kernel (tools/golden/silicon/)."""
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "golden", "silicon", name + ".json")


def replace_silicon(spec, path):
    """Return a copy of `spec` with `silicon_golden` set (test helper)."""
    import dataclasses
    return dataclasses.replace(spec, silicon_golden=path)


def load_silicon(spec):
    """Load the spec's silicon golden JSON if present, else None (bootstrap)."""
    import os
    import json
    if not os.path.exists(spec.silicon_path):
        return None
    with open(spec.silicon_path) as f:
        return json.load(f)


def spec_inputs(spec):
    """The input arrays (one list per input buffer) the kernel is fed.

    Designed inputs come from `spec.direct`; a spec without them (edge kernels)
    takes them from its silicon JSON. When both exist they must agree, so a
    stale capture can't silently pair old outputs with new inputs.
    """
    sj = load_silicon(spec)
    if spec.direct is not None:
        ins = [list(a) for a in spec.direct.inputs]
        if sj is not None:
            assert sj["inputs"] == ins, \
                f"{spec.name}: silicon golden inputs != designed inputs (recapture)"
    else:
        assert sj is not None, \
            f"{spec.name}: no designed inputs and no silicon golden to read them from"
        ins = sj["inputs"]
    assert len(ins) == len(spec.inputs), f"{spec.name}: {len(ins)} input arrays"
    return ins


def _bake_io(spec):
    """Return (input arrays, expected array or None) for a spec.

    The expected array is the captured silicon output; with no capture yet
    (bootstrap) it is None.
    """
    ins = spec_inputs(spec)
    sizes = ([spec.matmul.batch * spec.matmul.size_a,
              spec.matmul.batch * spec.matmul.size_b] if spec.matmul else [spec.n])
    for arr, sz in zip(ins, sizes):
        assert len(arr) == sz, f"{spec.name}: input length {len(arr)} != {sz}"
    sj = load_silicon(spec)
    exp = None
    if sj is not None:
        exp = sj["silicon"]
        assert len(exp) == spec.n_out, \
            f"{spec.name}: silicon length {len(exp)} != {spec.n_out}"
    return ins, exp


def kernel_data(spec):
    """The kernel's runtime data as raw buffers (see vv_data.KernelData).

    One little-endian buffer per input (in the host C type of that input) and
    the expected output in the output's host type; `expected` is None when the
    kernel has no silicon capture yet.
    """
    ins, exp = _bake_io(spec)
    return vv_data.KernelData(
        inputs=[vv_data.pack(arr, buf.ctype) for arr, buf in zip(ins, spec.inputs)],
        expected=None if exp is None else vv_data.pack(exp, spec.output.ctype),
    )


_MLIR_TMPL = string.Template(
    """//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Half-B $name capture design (GENERATED). Single compute tile (0,2) reads a
// $n-element batch from DDR via shim DMA, runs the $name kernel, writes the
// result back. Direct shim<->core objectfifos (no memtile relay).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @$func(memref<${n}x${in_ty}>, memref<${n}x${out_ty}>)
        attributes {link_with = "$obj"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<${n}x${in_ty}>>
    aie.objectfifo @outC(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<${n}x${out_ty}>>

    %core_0_2 = aie.core(%core) {
      %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<${n}x${in_ty}>
      %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<${n}x${out_ty}>

      func.call @$func(%a, %o) : (memref<${n}x${in_ty}>, memref<${n}x${out_ty}>) -> ()

      aie.objectfifo.release @inA(Consume, 1)
      aie.objectfifo.release @outC(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<${n}x${in_ty}>, %c: memref<${n}x${out_ty}>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, $n][0, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<${n}x${in_ty}>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, 1, $n][0, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<${n}x${out_ty}>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
"""
)


_MLIR_MATMUL_TMPL = string.Template(
    """//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Half-B $name capture design (GENERATED). Single compute tile (0,2) reads
// row-major A ($na elems) and B ($nb elems) from DDR via two shim DMAs, runs a
// batch of native mmul tiles, writes C ($nc elems) back. Direct shim<->core
// objectfifos (no memtile relay).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @$func(memref<${na}x${a_ty}>, memref<${nb}x${b_ty}>, memref<${nc}x${c_ty}>)
        attributes {link_with = "$obj"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<${na}x${a_ty}>>
    aie.objectfifo @inB(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<${nb}x${b_ty}>>
    aie.objectfifo @outC(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<${nc}x${c_ty}>>

    %core_0_2 = aie.core(%core) {
      %a = aie.objectfifo.acquire @inA(Consume, 1) : memref<${na}x${a_ty}>
      %b = aie.objectfifo.acquire @inB(Consume, 1) : memref<${nb}x${b_ty}>
      %o = aie.objectfifo.acquire @outC(Produce, 1) : memref<${nc}x${c_ty}>

      func.call @$func(%a, %b, %o) : (memref<${na}x${a_ty}>, memref<${nb}x${b_ty}>, memref<${nc}x${c_ty}>) -> ()

      aie.objectfifo.release @inA(Consume, 1)
      aie.objectfifo.release @inB(Consume, 1)
      aie.objectfifo.release @outC(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<${na}x${a_ty}>, %b: memref<${nb}x${b_ty}>, %c: memref<${nc}x${c_ty}>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, $na][0, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<${na}x${a_ty}>
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, $nb][0, 0, 0, 1]) {id = 1 : i64, metadata = @inB} : memref<${nb}x${b_ty}>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, 1, $nc][0, 0, 0, 1]) {id = 2 : i64, metadata = @outC} : memref<${nc}x${c_ty}>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
"""
)


def render_mlir(spec):
    """Render the single-tile shim<->core design.

    Matmul kernels (spec.matmul set) use the two-input (A, B) -> one-output (C)
    template; elementwise kernels use the single-input template.
    """
    if spec.matmul is not None:
        mm = spec.matmul
        return _MLIR_MATMUL_TMPL.substitute(
            name=spec.name,
            func=spec.func,
            obj=f"{_stem(spec)}.o",
            na=mm.batch * mm.size_a,
            nb=mm.batch * mm.size_b,
            nc=mm.batch * mm.size_c,
            a_ty=spec.inputs[0].mlir,
            b_ty=spec.inputs[1].mlir,
            c_ty=spec.output.mlir,
        )
    assert len(spec.inputs) == 1, "render_mlir currently models single-input kernels"
    return _MLIR_TMPL.substitute(
        name=spec.name,
        func=spec.func,
        obj=f"{_stem(spec)}.o",
        n=spec.n,
        in_ty=spec.inputs[0].mlir,
        out_ty=spec.output.mlir,
    )


def _cc_header(stem):
    """80-column LLVM-style file ruler for `<stem>.cc`, regardless of stem length."""
    prefix = f"//===- {stem}.cc "
    suffix = "*- C++ -*-===//"
    return prefix + "-" * (80 - len(prefix) - len(suffix)) + suffix


_KERNEL_TMPL = string.Template(
    """$header
//
// Half-B vector-compute capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). $doc
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>
$defines
extern "C" {

void $func($sig) {
$body}

} // extern "C"
"""
)


def _signature(spec):
    """`<ktype> *restrict <name>` for each input then output, comma-joined.

    Uses each buffer's kernel-signature type (which may differ from the host
    staging type), so a bit-pattern-staged buffer is seen as its real type.
    """
    bufs = list(spec.inputs) + [spec.output]
    return ", ".join(f"{b.kernel_ctype} *restrict {b.name}" for b in bufs)


def render_kernel(spec):
    """Wrap the hand-written intrinsic body in the standard kernel.cc scaffold."""
    defines = "".join(f"\n#define {k} {v}" for k, v in spec.defines)
    if defines:
        defines += "\n"
    return _KERNEL_TMPL.substitute(
        header=_cc_header(_stem(spec)),
        doc=spec.doc,
        defines=defines,
        func=spec.func,
        sig=_signature(spec),
        body=spec.body,
    )


_RUN_LIT_TMPL = string.Template(
    """// (c) Copyright 2026 -- xdna-emu Half-B vector-compute verification.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// REQUIRES: ryzen_ai_npu1, chess
//
// RUN: xchesscc_wrapper aie2 -I %aietools/include -c %S/$cc -o ./$obj
// RUN: %aiecc --xchesscc --xbridge --get-xclbin --get-npu-insts --xclbin-name=aie.xclbin --npu-insts-name=insts.bin %S/aie.mlir
// RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags $host_flags
// RUN: %run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin
"""
)


def render_run_lit(spec):
    """Render the lit build/run recipe (Chess compile -> aiecc -> host -> run).

    test.cpp is the shared vv_host.cpp, staged by tools/stage_vector_kernel.py;
    the -D flags tell it the buffer shape.
    """
    stem = _stem(spec)
    return _RUN_LIT_TMPL.substitute(
        cc=f"{stem}.cc", obj=f"{stem}.o",
        host_flags=vv_data.host_flags(len(spec.inputs), spec.output.ctype, spec.n_out))


def generate(spec, outdir):
    """Write the kernel files into outdir/spec.name; return that directory.

    Files: run.lit, aie.mlir, <stem>.cc. No data and no test.cpp: the host
    program is common/vv_host.cpp and the data is staged from `kernel_data`.
    """
    import os

    dest = os.path.join(str(outdir), spec.name)
    os.makedirs(dest, exist_ok=True)
    files = {
        "run.lit": render_run_lit(spec),
        "aie.mlir": render_mlir(spec),
        f"{_stem(spec)}.cc": render_kernel(spec),
    }
    for fn, text in files.items():
        with open(os.path.join(dest, fn), "w") as f:
            f.write(text)
    return dest


def _default_out_dir():
    import os
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tests", "vector-verify")


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Generate vector-compute capture kernels from specs.")
    ap.add_argument("name", help="spec name (e.g. vec_ups_i32), a sweep name "
                                 "(e.g. vec_srs_i32_sweep), 'all', or 'all-sweeps'")
    ap.add_argument("--out", default=_default_out_dir(),
                    help="output root (default: tests/vector-verify)")
    args = ap.parse_args(argv)

    from vector_kernel_specs import SPECS, SWEEPS

    # Resolve the request to a flat list of KernelSpecs. A sweep name expands to
    # one KernelSpec per mode-point; 'all-sweeps' expands every sweep.
    to_gen = []
    if args.name == "all":
        to_gen = [SPECS[n] for n in sorted(SPECS)]
    elif args.name == "all-sweeps":
        for sname in sorted(SWEEPS):
            to_gen.extend(SWEEPS[sname].expand())
    elif args.name in SWEEPS:
        to_gen = SWEEPS[args.name].expand()
    elif args.name in SPECS:
        to_gen = [SPECS[args.name]]
    else:
        ap.error(f"unknown spec/sweep '{args.name}'; specs: {', '.join(sorted(SPECS))}; "
                 f"sweeps: {', '.join(sorted(SWEEPS))}")

    for spec in to_gen:
        dest = generate(spec, args.out)
        print(f"generated {spec.name} -> {dest}")


if __name__ == "__main__":
    main()
