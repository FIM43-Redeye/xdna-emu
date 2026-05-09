# lc_overflow_probe

Hardware probe for the mode-2 LC frame's bit-28 flag at trip counts that
straddle the 2^28 boundary. Findings written up in
[`docs/superpowers/findings/2026-05-08-lc-overflow-empirical.md`](../../../docs/superpowers/findings/2026-05-08-lc-overflow-empirical.md):
bit-28 is an overflow saturation flag, count is `N mod 2^28`.

## Build

```bash
source /home/triple/npu-work/toolchain-build/activate-npu-env.sh
tools/mode2_capture_fixtures/build_fixture.sh --chess --mode2 \
    tools/mode2_capture_fixtures/lc_overflow_probe
```

`--chess` is mandatory: Peano-built kernel.o + Chess linker NULL-derefs.
`--mode2` switches the trace injector to inst_exec (the only mode that
emits LC frames; default is event_pc).

## Run

```bash
python3 tools/lc-overflow-probe.py
```

Writes per-N captures + summary to
`build/experiments/lc_overflow/<timestamp>/`. Defaults sweep
`{4, 64, 1024, 65536, 2^24, 2^28-1, 2^28, 2^28+1, 2^28+5, 2^29-1, 2^29, 2^29+5}`.

`--trip-counts N1,N2,...` overrides the default sweep.

## Inputs

The kernel reads two values from the input buffer:

- `in[0]` = N (ZOL trip count)
- `in[1]` = passes (outer wrapper count)

The probe picks `passes` automatically based on N to stay under the
Phoenix XRT command timeout (~5s):

| N range          | passes |
|------------------|-------:|
| < 2^22           | 4      |
| 2^22 .. 2^26 - 1 | 2      |
| >= 2^26          | 1      |

Above ~3 * 2^28 the kernel runtime exceeds the timeout regardless of
pass count (single pass at 2^30 runs ~7s on Phoenix).
