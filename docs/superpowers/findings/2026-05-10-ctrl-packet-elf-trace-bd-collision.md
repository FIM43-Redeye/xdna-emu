---
name: 'ctrl_packet_reconfig_elf trace BD collides with input BO'
description: ELF-mode kernels with control-packet reconfiguration leave the trace BD pointing at a hardcoded DDR address that overlaps user BOs, corrupting input data partway through the run. Trace-prepare adds bo_trace to the kernel call but not to the runtime_sequence args, so XRT has no relocation to patch the trace BD with the real bo_trace address.
type: project
---

# ctrl_packet_reconfig_elf -- trace BD lands on the input BO

## TL;DR

The bridge test `ctrl_packet_reconfig_elf` failed on EMU with HW PASS
because the trace S2MM BD's destination address (0x800000008000) overlaps
the input BO (0x800000009000). The trace BD writes 1MB of trace stream
data to DDR; once it crosses the 0x9000 boundary it overwrites bo_in,
the compute kernel then reads garbage (0x71...), adds 3, and produces
0x74 = 116 -- exactly the "wrong-value 116" pattern we observed.

Quarantining the test from trace injection restores a functional pass;
the trace integration bug itself is deferred (non-trivial cross-tool
fix in trace-prepare / trace-inject).

## How the bug got hidden as an EMU correctness bug

Without instrumentation, the failure looks like the compute kernel is
broken: ~2500 of 4096 output bytes are wrong, most are 116, and column 0
of every "row" is correct (single byte). It reads as an instruction
decoding or VLIW timing issue.

The pattern actually reflects:

- The input BO at 0x9000 is being overwritten by the trace BD during
  execution. Column 0 of each row is the first byte of a 64-byte row,
  which gets read before the trace BD's stripe reaches that offset.
- Subsequent bytes are read after the trace data has already landed on
  top of them.
- 0x71 = 'q' is just trace packet payload (event ids etc.); the kernel
  reads 0x71, adds 3, stores 0x74 = 116. The compute logic is correct.

## Root cause chain

1. `trace-prepare.py` injects `bo_trace = xrt::bo(...)` and appends it to
   the kernel call: `kernel0(opcode, 0, 0, bo_in, bo_out, bo_trace)`.
2. But the MLIR `runtime_sequence` keeps its original signature:
   `@run(%arg0: memref<64x64xi8>, %arg1: memref<64x64xi8>)` -- only two
   args, no slot for bo_trace.
3. For ELF-mode kernels, XRT patches BO addresses into the ctrltext via
   relocations. The ELF's relocation table only has slots for arguments
   the runtime sequence knows about. bo_trace at kernel arg slot 5 has
   no relocation slot.
4. The trace BD (shim ch1 BD15, S2MM, 1MB) gets its source/dest address
   from whatever the MLIR or ELF baked in. In this test that happens to
   be 0x800000008000, which overlaps the input BO at 0x9000.
5. EMU's `DdrPatch` trace shows only `arg_idx=0,1,2` patches -- bo_in,
   bo_out, and (79 patches) the ctrl_pkt BOs. No patch for bo_trace.

## Why HW passes

Real hardware via XRT firmware handles BD address allocation
differently. The trace integration appears to work on HW either because:

- XRT firmware reads the trace_config-supplied BO from a different path
  and patches it server-side, OR
- HW's BD address space mapping does not overlap user BOs the way the
  emulator's flat DDR model does.

We have not characterized the HW path precisely; the symptom (HW pass,
EMU fail with input-BO corruption) is the diagnostic.

## Verification

Re-ran the test with `--no-trace` (skipping trace injection entirely):

```
ctrl_packet_reconfig_elf:  Chess/HW PASS, Chess/EMU PASS
```

Functional correctness on EMU is intact; the bug is strictly in the
trace integration path.

## Fix applied (mitigation)

Added `ctrl_packet_reconfig_elf` to `scripts/trace-quarantine.txt` with
a full reason comment. The bridge runner now skips trace injection for
this test by default; it still runs HW + EMU functionally.

## Proper fix (deferred)

A real fix requires `trace-prepare.py` to:

1. Detect ELF-mode kernels (xrt::ext::kernel with xrt::elf module).
2. Either:
   - Inject bo_trace as a `runtime_sequence` argument so XRT relocations
     include it, OR
   - Skip injection for ELF-mode tests (effectively automating today's
     manual quarantine).

This is a cross-tool change (mlir-trace-inject.py, cpp_trace_patch.py,
possibly emu-bridge-test.sh). Not justified yet; one test affected.
Revisit if more ELF-mode tests show similar symptoms.

## See also

- `tools/cpp_trace_patch.py` -- C++ side of trace-prepare (host).
- `tools/mlir-trace-inject.py` -- MLIR side (device).
- `scripts/trace-quarantine.txt` -- the quarantine list itself.
- The original FAIL evidence: 20260509 bridge results show 2531/4096
  wrong outputs, dominated by value 116.
