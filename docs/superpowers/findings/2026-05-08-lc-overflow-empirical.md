# Mode-2 LC frame bit-28 IS an overflow flag

**Status:** Phase 0 follow-up complete. Hypothesis confirmed empirically.
**Captured:** 2026-05-08 on Phoenix (NPU1, AIE2), Chess-built fixture.

## What we now know

The 32-bit mode-2 LC frame (prefix `010` = 3 bits, then 1-bit flag, then
28-bit count) decodes as:

| Field         | Width | Meaning                                          |
|---------------|-------|--------------------------------------------------|
| prefix        | 3     | `010` (LC frame discriminator)                   |
| **bit-28 flag** | 1   | **Overflow saturation: 1 iff trip count >= 2^28** |
| count         | 28    | `trip_count mod 2^28`                            |

Flag is saturated -- it stays 1 for any trip count >= 2^28, it does not
count "how many overflows happened." The 28-bit count field wraps cleanly
modulo 2^28.

This refutes the original placeholder rule
(`flag = 1 iff lc_after == 0`) and confirms the bit's actual semantics
that no public AMD/Xilinx doc described.

## Method

Built `tools/mode2_capture_fixtures/lc_overflow_probe/` (Chess kernel.cc
+ Chess-aiecc) with a heavy_zol-style outer wrapper of runtime-controlled
length: `in[0]` holds the ZOL trip count N, `in[1]` holds the wrapper
pass count (4 at low N for redundancy, 1 at high N to stay under the
XRT command timeout). Disassembly confirmed Chess emits a clean
`mov lc, r25 (= N)` ZOL when the body is the runtime_loop shape
(`acc += in[i+1]`).

The driver is `tools/lc-overflow-probe.py`. For each N it writes the
input buffer, invokes `bridge-trace-runner` for HW capture, decodes
the trace bytes through the in-tree mode-2 decoder, and records every
LC frame's `(flag, count)` pair.

**Captures:** `build/experiments/lc_overflow/20260508-00472?` (per-N
record.json + raw trace.bin + summary.txt).

## Result

```
             N        N_hex  lc_frames  flag  count        N_hex_anchor
            64         0x40          4    0   64           below 2^14
       16777216    0x1000000        2    0   16777216      2^24
     268435455    0xfffffff         1    0   268435455     2^28 - 1   <- last 28-bit clean
     268435456   0x10000000         1    1   0             2^28       <- first overflow
     268435457   0x10000001         1    1   1             2^28 + 1
     268435461   0x10000005         1    1   5
     536870911   0x1fffffff         1    1   268435455     2^29 - 1   <- count == 2^28 - 1 again
     536870912   0x20000000         1    1   0             2^29       <- count wraps to 0 again
     536870917   0x20000005         1    1   5
     805306368   0x30000000   FAILED                        3 * 2^28   (kernel TDR -- runtime > 5s)
```

Observations:

1. For N < 2^28, flag=0 and count=N exactly. Matches Phase 0
   (16384-frame baseline).
2. At N = 2^28 (the first value that overflows the 28-bit count field),
   flag flips to 1 and count = 0.
3. For N in (2^28, 2^29), count = N - 2^28 (i.e. N mod 2^28); flag
   stays 1.
4. At N = 2^29, count wraps to 0 again; flag stays 1 -- it does **not**
   advance to 2 or any other multi-bit value, because there's only one
   bit there.
5. Above ~3 * 2^28, the kernel exceeds the Phoenix command-timeout
   (~5s on NPU1 default config) and gets TDR'd -- not a finding about
   the LC frame, just a runtime ceiling for this probe shape.

## Why the runtime_loop body is the only one Chess respects

Initial probe attempts used a register-only body (`acc ^= i`) and a
masked-wrapping body (`acc += in[(i+1) & 63]`); both made Chess
constant-fold the LC to a small literal (LC=0 or LC=2) regardless of
N. Disassembly showed `mov lc, #0x0` / `mov lc, #0x2` instead of
`mov lc, r25`.

Only the runtime_loop shape (`acc += in[i+1]`, no mask, no purely-
register body) produced the desired `mov lc, r25 (= N)` lowering.
The kernel comment in `kernel.cc` documents this so future probes
don't repeat the same dead end.

Out-of-bounds reads at high N (`in[i+1]` with i up to 2^29) are
acceptable for this probe: AIE TLM doesn't fault on tile-DM
out-of-range reads; the loaded value is undefined but we discard
the kernel output. Only the trace bytes matter.

## What this changes in xdna-emu

`src/device/trace_unit/mod.rs::compute_lc_flag` was already pinned at
`0` per the 2026-04-30 finding. With this empirical result we can
upgrade it to a real implementation:

```rust
fn compute_lc_flag(trip_count: u32) -> u8 {
    if trip_count >= (1 << 28) { 1 } else { 0 }
}

fn compute_lc_count(trip_count: u32) -> u32 {
    trip_count & 0x0FFF_FFFF  // bits 27..0
}
```

The emit point is the ZOL boundary (one frame per ZOL invocation,
not per iteration), with `trip_count` taken from the value loaded
into LC at ZLS time. This was already the per-invocation rule from
Phase 0; the only delta is that flag is now a function of trip_count
instead of a constant 0.

## Why this matters upstream

The empirical confirmation is worth flagging on
[mlir-aie #3047](https://github.com/Xilinx/mlir-aie/issues/3047)
(the public discussion thread for the LC bit-28 question). The
mlir-aie issue currently closes with "bit-28 appears inert." Update:
"bit-28 is an overflow saturation flag for trip counts >= 2^28; count
field is `N mod 2^28`. Reproducer:
`tools/mode2_capture_fixtures/lc_overflow_probe/`."

Whether to actually post upstream is a judgment call -- the bit fires
for trip counts above 268M, which essentially never happens in real
ML kernels (typical inner ZOL trip counts are 4..1024). It's a
correctness improvement to the decoder, not something users will
notice. Worth posting as a follow-up comment but not urgent.

## Methodology notes for future Phase-0-style probes

- **Chess + chess-built kernel.o is mandatory for mode-2 trace
  capture.** Peano kernel.o + chess linker (`bridge`) NULL-derefs at
  link time -- the cross-toolchain mix is structurally broken.
  `build_fixture.sh --chess` now compiles kernel.cc with
  `xchesscc_wrapper` to keep the ABI consistent.
- **Heavy_zol-style outer wrappers are required for trace flushing.**
  Single-pass tiny kernels never flush the trace shim DMA, regardless
  of how high N goes. The wrapper count needs to be runtime-controlled
  (via in[1]) when probing a wide N range, because high N kernels need
  fewer passes to fit under the XRT command timeout.
- **The runtime_loop body shape is the canonical "Chess emits a clean
  ZOL" pattern.** Any cleverness in the body (register-only,
  bit-masking, etc.) trips Chess's pipelining/folding analysis and
  the LC stops being a function of N.
- **Empirical N range for AIE2 LC probes on Phoenix: ~2^14 to ~2^29.**
  Below 2^14, single-pass kernels don't flush; above ~3 * 2^28, the
  default XRT timeout fires.
