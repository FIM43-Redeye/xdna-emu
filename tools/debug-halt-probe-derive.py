#!/usr/bin/env python3
"""Derive debug_halt_probe magic constants from the compiled core ELF.

OUTBUF_ADDR     = value of the `output_buffer` ELF symbol & 0xFFFF
TRAP_PC         = PC of the bundle storing the 0xBB marker to
                  output_buffer[1] (the `st ..., [p0, #4]` where p0 was
                  materialized with the output_buffer base address)
PC_EVENT0_VALUE = 0x80000000 | (TRAP_PC & 0x3FFF)

Golden-record-preserving: regenerate aie.mlir/test.cpp from .in
templates with the derived values, diff against the committed golden
files. Match -> emit the golden bytes verbatim, exit 0. Drift -> print
committed-vs-derived and exit 1 (fails the build loudly). A missing
symbol or unlocatable trap bundle is a hard error, exit 2 -- never emit
a default.

Derivation strategy (from real llvm-objdump-aie output on the Chess ELF):
  The AIE2 VLIW disassembly uses llvm-objdump from the Peano (llvm-aie)
  toolchain.  Each bundle occupies one line:
    "     17a: <hex bytes>\\t<slot0>;\\t\\t<slot1>;\\t\\t..."
  Addresses are lower-hex with leading spaces; the address column format
  is "^\\s+([0-9a-fA-F]+):".

  The trap bundle sequence:
    0x17a: mova dj0, #0xbb;  movxm p0, #0x70400   (materialize outbuf into p0)
    0x184: st dj0, [p0, #4]; mov m0, #0xaa         (store 0xBB marker)

  TRAP_PC = 0x184 (the store bundle's address, not the movxm bundle).

  parse_trap_pc scans forward through lines:
    1. Find the line containing "movxm pN, #0x<outbuf_full>" -- capture pN.
    2. Find the NEXT line containing "st ..., [pN, #4]" -- its leading
       address column is TRAP_PC.
  In this kernel's compiled schedule the Chess compiler placed the movxm
  and the store in separate VLIW bundles (separate disasm lines), so the
  store carries its own address column.  That is a property of this
  build, not an ISA guarantee -- the AIE2 ISA permits both in one bundle.
  If a future schedule fuses them, the store's address column cannot be
  recovered unambiguously and parse_trap_pc raises DeriveError rather
  than guessing (the "never emit a default" contract).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys


class DeriveError(Exception):
    """Symbol/bundle could not be located -- never emit a default."""


def _tool(preferred, fallback):
    return shutil.which(preferred) or shutil.which(fallback) or fallback


def _objdump_bin():
    """Locate an llvm-objdump that understands the AIE ELF target.

    The Peano compiler (llvm-aie) builds against the AIE backend and is the
    only llvm-objdump that can disassemble AIE ELFs.  The mlir-aie LLVM does
    NOT include the AIE backend, so the first `llvm-objdump` on PATH may be
    the wrong one.

    Search order:
      1. llvm-objdump-aie  (canonical name if the env wires it up)
      2. $NPU_WORK_DIR/llvm-aie/install/bin/llvm-objdump  (Peano, explicit)
      3. llvm-objdump      (PATH fallback -- may fail on AIE ELFs)
    """
    explicit = shutil.which("llvm-objdump-aie")
    if explicit:
        return explicit
    npu = os.environ.get("NPU_WORK_DIR")
    if npu:
        peano = os.path.join(npu, "llvm-aie", "install", "bin", "llvm-objdump")
        if os.path.isfile(peano):
            return peano
    print("warning: debug-halt-probe-derive: no AIE-aware llvm-objdump found "
          "(llvm-objdump-aie / $NPU_WORK_DIR Peano); falling back to PATH "
          "llvm-objdump, which may not disassemble AIE ELFs",
          file=sys.stderr)
    return shutil.which("llvm-objdump") or "llvm-objdump"


def run_nm(elf):
    return subprocess.run([_tool("llvm-nm-aie", "nm"), elf],
                          capture_output=True, text=True, check=True).stdout


def run_objdump(elf):
    return subprocess.run([_objdump_bin(), "-d", elf],
                          capture_output=True, text=True, check=True).stdout


# nm output line: "<hex_value> <type> output_buffer"
# The type field for an absolute symbol (A) is a single character.
_NM_RE = re.compile(r"^\s*([0-9a-fA-F]+)\s+\S+\s+output_buffer\s*$",
                    re.MULTILINE)


def _find_outbuf_symbol(nm_text):
    """Return the FULL (unmasked) output_buffer symbol value.

    Single point of nm parsing -- callers that need the masked tile-local
    value apply `& 0xFFFF` themselves.  Raises DeriveError if the symbol is
    absent -- never returns a default.
    """
    m = _NM_RE.search(nm_text)
    if not m:
        raise DeriveError("output_buffer symbol not found in nm output")
    return int(m.group(1), 16)


def parse_outbuf_addr(nm_text):
    """Return output_buffer symbol value & 0xFFFF (tile-local address).

    Raises DeriveError if the symbol is absent -- never returns a default.
    """
    return _find_outbuf_symbol(nm_text) & 0xFFFF


# Address column at start of a disasm line.
# Real format (llvm-objdump-aie / Peano): "      17a: <hex bytes>\t..."
# Lower-case hex, leading spaces, colon.  Always .search()ed on a single
# line, so no re.MULTILINE flag is needed (^ anchors at string start).
_ADDR_RE = re.compile(r"^\s*([0-9a-fA-F]+):")


def parse_trap_pc(objdump_text, outbuf_full):
    """PC of the bundle that stores to [pN, #4] where pN == output_buffer base.

    Strategy (deterministic given the kernel; see module docstring):
      1. Find the line containing "movxm pN, #0x<outbuf_full>" and capture
         the pointer register name pN.
      2. Find the NEXT line that contains "st ..., [pN, #4]".
      3. Return that line's leading address column as TRAP_PC.

    In this kernel's compiled schedule the Chess compiler placed the movxm
    and the store in separate VLIW bundles (separate disasm lines, distinct
    address columns), so the first-match-after-movxm rule resolves to the
    store's own address.  That separation is a property of this build, not
    an ISA guarantee; if a future schedule fuses them into one bundle the
    store's address column cannot be recovered unambiguously and this
    raises DeriveError rather than returning a guess.

    Raises DeriveError if either anchor is absent.
    """
    hexaddr = f"#0x{outbuf_full:x}"
    lines = objdump_text.splitlines()
    ptr_reg = None
    movxm_line = None
    for i, ln in enumerate(lines):
        if ptr_reg is None:
            # Look for: movxm pN, #0x<outbuf_full>
            m = re.search(r"movxm\s+(p\d+)\s*,\s*" + re.escape(hexaddr), ln)
            if m:
                ptr_reg = m.group(1)
                movxm_line = i
            continue
        # ptr_reg is set; look for the store to [pN, #4].
        # Pattern: "st" mnemonic (word boundary), any non-semicolon chars
        # (one VLIW slot's worth), then "[pN, #4]".
        if re.search(r"\bst\b[^;]*\[\s*" + re.escape(ptr_reg) + r"\s*,\s*#4\s*\]", ln):
            am = _ADDR_RE.search(ln)
            if not am:
                # The store bundle's address may be on the bundle-header line;
                # walk back to the nearest preceding address column -- but
                # never past the movxm line.  If the only address we could
                # recover is at/before the movxm, the store's own address
                # column was not found: that is the fused-bundle case, and
                # returning the movxm PC (or earlier) would be a wrong
                # guess.  Fail loudly instead.
                for back in range(i - 1, movxm_line, -1):
                    am = _ADDR_RE.search(lines[back])
                    if am:
                        break
            if not am:
                raise DeriveError(
                    f"located the [{ptr_reg}, #4] store but could not recover "
                    f"its address column without crossing the movxm "
                    f"{ptr_reg}, #0x{outbuf_full:x} line (fused-bundle "
                    "schedule?) -- refusing to emit an earlier PC")
            return int(am.group(1), 16)
    if ptr_reg is None:
        raise DeriveError(
            f"could not locate the movxm pN, #0x{outbuf_full:x} that "
            "materializes the output_buffer base")
    raise DeriveError(
        f"could not locate the [{ptr_reg}, #4] store after movxm "
        f"{ptr_reg}, #0x{outbuf_full:x}")


def pc_event0_value(trap_pc):
    """PC_Event0 register value encoding TRAP_PC as a breakpoint address."""
    return 0x80000000 | (trap_pc & 0x3FFF)


def _render(template_path, subs):
    text = open(template_path).read()
    for token, value in subs.items():
        text = text.replace(token, value)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Derive debug_halt_probe constants from the compiled ELF "
                    "and verify against committed golden files.")
    ap.add_argument("--elf", required=True,
                    help="compiled core ELF (first-pass scratch build)")
    ap.add_argument("--in-mlir", required=True,
                    help="aie.mlir.in template with @PC_EVENT0_VALUE@/@TRAP_PC@ tokens")
    ap.add_argument("--in-cpp", required=True,
                    help="test.cpp.in template with @OUTBUF_ADDR@ token")
    ap.add_argument("--golden-mlir", required=True,
                    help="committed golden aie.mlir")
    ap.add_argument("--golden-cpp", required=True,
                    help="committed golden test.cpp")
    ap.add_argument("--out-mlir", required=True,
                    help="output aie.mlir (written only on golden match)")
    ap.add_argument("--out-cpp", required=True,
                    help="output test.cpp (written only on golden match)")
    args = ap.parse_args(argv)

    try:
        nm_text = run_nm(args.elf)
        # Single nm parse: full value for pointer-materialization matching;
        # outbuf is the masked tile-local view of the same symbol.
        full = _find_outbuf_symbol(nm_text)
        outbuf = full & 0xFFFF
        trap_pc = parse_trap_pc(run_objdump(args.elf), outbuf_full=full)
    except (DeriveError, subprocess.CalledProcessError, OSError) as e:
        print(f"FATAL: debug_halt_probe derivation failed: {e}",
              file=sys.stderr)
        return 2

    pcev0 = pc_event0_value(trap_pc)
    subs_mlir = {"@PC_EVENT0_VALUE@": f"0x{pcev0:08x}",
                 "@TRAP_PC@": f"0x{trap_pc:x}"}
    subs_cpp = {"@OUTBUF_ADDR@": f"0x{outbuf:04x}"}

    gen_mlir = _render(args.in_mlir, subs_mlir)
    gen_cpp = _render(args.in_cpp, subs_cpp)
    golden_mlir = open(args.golden_mlir).read()
    golden_cpp = open(args.golden_cpp).read()

    drift = []
    if gen_mlir != golden_mlir:
        drift.append("aie.mlir")
    if gen_cpp != golden_cpp:
        drift.append("test.cpp")

    if drift:
        print("FATAL: debug_halt_probe constants have drifted from the "
              f"committed golden record ({', '.join(drift)}).",
              file=sys.stderr)
        print(f"  derived OUTBUF_ADDR=0x{outbuf:04x} "
              f"TRAP_PC=0x{trap_pc:x} PC_EVENT0=0x{pcev0:08x}",
              file=sys.stderr)
        print("  The probe is a permanent Phase A artifact. If the "
              "kernel/allocation change was intentional, commit the "
              "regenerated aie.mlir/test.cpp as the new golden record "
              "and re-run; otherwise revert the kernel change.",
              file=sys.stderr)
        return 1

    # Match: emit the committed golden bytes verbatim (Phase A bytes
    # preserved; G1/G2 integrity intact).
    open(args.out_mlir, "w").write(golden_mlir)
    open(args.out_cpp, "w").write(golden_cpp)
    print(f"debug_halt_probe constants verified: OUTBUF_ADDR=0x{outbuf:04x} "
          f"TRAP_PC=0x{trap_pc:x} PC_EVENT0=0x{pcev0:08x} (golden match)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
