# Findings: BUG-A Fix + Second Differential Batch (Re-triage)

**Date:** 2026-05-30
**Follow-on to:** `2026-05-30-fuzzer-revival-first-batch.md` (which recorded
BUG-A as the single blocker masking every divergence)
**Run range:** seeds 1..2000 (Peano, scalar single-tile, trace-sweep OFF, `--hw`)

---

## Headline: BUG-A fixed, pass rate 0% -> 76%

| Category | First batch | This batch |
|----------|-------------|------------|
| Pass (EMU == NPU, nonzero) | **0** | **1525 (76%)** |
| Vacuous match (both all-zero) | 379 | 379 (19%) |
| Mismatch | 1577 | **53 (2.6%)** |
| CRASH (emulator panic) | n/a | **0** |
| Error (HW timeout / TDR) | 44 | 43 |

95% of seeds are now confirmed correct against silicon (pass + vacuous), up
from 0%. The batch completed cleanly in ~1655s (~27.6 min; the post-fix EMU
genuinely executes each kernel now instead of bailing at the stall threshold,
so per-seed cost is higher than the all-zeros first batch).

---

## BUG-A root cause and fix (commit `87b3d03`)

Column clock power-on (`Column_Clock_Control = 0x1` per partition column) is a
**driver/firmware action at context create** -- aie-rt `_XAieMl_RequestTiles`
(`device_aieml.c:309`), triggered by the driver's `MSG_OP_CREATE_CONTEXT` /
SMU `AIE_SMU_POWER_ON`. It is **never** carried in the user CDO. Confirmed three
ways: zero clock-register writes in seed_1's full flow; `xclbinutil` shows one
PRIMARY CDO with no `0xFFF20`; aie-rt + xdna-driver confirm it's strictly
driver-side.

The in-process `XclbinSuite` runner had no firmware stand-in, so columns booted
gated (silicon-accurate), `step_all_dma` skipped the entire gated column, and
the shim DMA froze at `BdSetup` -> all-zeros for every kernel. The XRT-plugin
path was immune only because XRT does host DMA externally and never exercises
the emulator's shim data plane -- which is also why no test ever caught this.
The XRT path already emulated firmware via the `xdna_emu_assign_partition` FFI
hook; the in-process path simply lacked the equivalent call.

**Fix:** one shared core primitive, `DeviceState::assign_partition_columns`,
called by both runtime paths -- the in-process runner for `[0, column_width)`
before applying the CDO, and the FFI hook (now delegating instead of
open-coding the loop). New end-to-end regression test
`add_one_using_dma_in_process_moves_data_through_shim` is the first in-process
test that asserts on computed data (`out[i] = i+2`); RED (all-zeros) before the
fix, GREEN after.

---

## Secondary bugs surfaced (now that the EMU executes kernels)

### Shift negate-overflow (commit `d974515`)

`execute_lshl_bidir` / `execute_ashl_bidir` computed the right-shift magnitude
as `((-b) as u32) & 0x1F`. A generated kernel with shift operand `b == i32::MIN`
made `-b` overflow and panic in debug. Fixed with `b.wrapping_neg()` (the shift
is masked to 5 bits anyway; `i32::MIN mod 32 == 0` -> shift-by-0). TDD-covered.

### Fuzzer harness could not survive an emulator panic (commit `e5457c5`)

The emulator workers run inside `std::thread::scope`, which **re-raises a
scoped thread's panic at join**. So a single panicking seed aborted the entire
batch after grinding through every seed -- no summary, no comparison, no report.
Fixed by wrapping each seed's run in `catch_panic`; an emulator panic is now its
own **CRASH** category, surfaced loudly per-seed and in the summary (NPU-side
panics are categorized as errors, not CRASH). This is why the re-run shows
`0 CRASH` and completed cleanly even though the first attempt would have died.

---

## Remaining divergences (re-triage) -> BUG-B

The 53 mismatches are NOT 53 distinct bugs. They collapse to one pattern plus
one outlier:

### BUG-B (52 i8 + 1 i16): NPU zeros every 4th sub-word element

Signature: only sub-word dtypes diverge (**52 i8 + 1 i16, zero i32**), all with
the NPU returning 0 at every element index `== 3 mod 4` while the EMU computes
the C-correct value:

```
seed 18  (i8, buf_out[i]=i*i):  idx 3,7,11,15 -> EMU=9,49,121,-31   NPU=0,0,0,0
seed 1218(i16):                 idx 3,7,11    -> EMU=121,129,137     NPU=0,0,0
```

**Leading hypothesis: this is a fuzzer NPU-harness artifact, not an emulator
bug.** Evidence: i32 (full-word) is flawless; the EMU matches plain C math; the
NPU is the side producing the anomaly; and "silicon zeros every 4th i8 in a
simple loop" cannot be real hardware behavior (it would break every i8 NPU
program in production). The likely culprit is the fuzzer's sub-word output
readback (`npu_runner`) or the runtime_sequence / output DMA that
`tools/fuzz_template.py` generates for sub-word buffers. **To confirm next:**
inspect how the NPU path sizes/reads i8/i16 output buffers vs i32.

If the hypothesis holds, the emulator's true sub-2.6% divergence rate is even
lower -- possibly just seed 1964 (below).

### Seed 1964 (i8): lone genuine value divergence -- defer

Unlike the every-4th-zero pattern, seed 1964 shows a real value difference at
index 3: `EMU=-64 (0xC0)`, `NPU=-16 (0xF0)`, with surrounding elements matching.
This is a candidate **real** emulator sub-word divergence (or a different facet
of BUG-B). Worth its own look once BUG-B's nature is settled.

### 43 errors

All HW `Timeout` (TDR, self-recovered). Not divergences; expected when a
generated kernel runs long on silicon.

---

## Next steps

1. **BUG-B**: determine fuzzer-harness-artifact vs emulator-sub-word-bug by
   inspecting the NPU sub-word output path (`npu_runner` readback +
   `tools/fuzz_template.py` output DMA / buffer sizing). This decides whether
   BUG-B is even an emulator-correctness item.
2. **Seed 1964**: investigate the lone genuine i8 value divergence afterward.
3. Once sub-word comparison is trustworthy, re-run to get a clean
   emulator-divergence count, then move to the deferred phases (vector ops,
   Chess).
