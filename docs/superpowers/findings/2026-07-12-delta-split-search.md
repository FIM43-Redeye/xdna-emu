# Delta/split search: no observed solution; genuine collision remains unproven

Date: 2026-07-12  
Target: Phoenix/NPU1 firmware `1502_00/npu.dev.sbin`  
SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`  
Region: `[0x8c98,0x8d52)`

## Verdict

No tested static delta/split map was observed to satisfy both runtime
landmarks before execution stopped. The search covered every byte split for
both literal framings over two three-delta sets:

- base/overlay plus zero-offset control: `{0, +0x5c, +0x100}`;
- raw-derived set: `{+0x5c, +0x100, +0x244}`.

Each set produces 2,226 canonical assignments. No tracked post-publisher
execution prefix stored any word of the device-SRAM descriptor or the
`FW_ALIVE_OFF` pointer, so there is no candidate `load_m2c` correction to
present.

This does **not** earn a genuine-collision verdict, even for the finite delta
families. The final conservative classifier leaves 738 and 1,426 assignments
inconclusive, respectively. Most stop on `Step::Unknown`, which represents
both invalid encodings and valid Xtensa instructions the interpreter has not
implemented; the rest hit the instruction budget. The brief also does not
define a finite universal delta set. Calling the collision genuine, and
therefore inferring an exotic fetch mechanism, would overstate this result.

## Search construction

`m2c_probe_execution_guided_framing_search` now starts every assignment from
the current production `FirmwareProcessor::load_m2c` map, removes exactly the
`[0x8c98,0x8d52)` overlay, and installs:

```text
[0x8c98, split)  -> file = VMA + delta_lo
[split, 0x8d52)  -> file = VMA + delta_hi
```

The live `L32r` word at `[0x354c,0x3550)` is independently tried at `+0x5c`
and `+0x100`; the rest of `[0x3550,0x3564)` remains at its production `+0x100`
view. Uniform maps and endpoint splits are canonicalized, so for three deltas
the exact count is:

```text
2 literal views * (3 uniform maps + 3*2 ordered pairs * 185 interior splits)
= 2,226 assignments
```

Each assignment runs continuously from reset with ordinary `Cpu::step` calls.
There is no interrupt injection, register/memory initialization, runtime view
switch, or firmware-state substitution. Instruction boundaries are recorded
from the bytes actually fetched and decoded on that run.

After the natural `0x9045 -> 0x8c98` publisher edge, fill fastpaths are disabled
so every retired store remains observable to recurrence tracking. An `L32r`
whose four-byte literal crosses a candidate seam, or a FLIX bundle containing a
store, is classified inconclusive rather than executed under incomplete
instrumentation. `Unknown` is also inconclusive without an independent Xtensa
decode oracle. Publish-store evidence is reset at the natural
`0x7fe1 -> 0x8c6c` service edge and recognizes any overlapping byte range by
either effective or physical address.

The upstream `[0x8c6c,0x8c98)` service prefix is intentionally left at BASE:
its live callback registration and executed direct-call chain pin that view,
and the brief makes expanding the split upstream optional. The `Bbci` target
therefore remains a naturally executed control-flow input, not an asserted
boundary inside the searched interval.

The runtime predicates are:

- publisher: local `[0x14820] == 0x55504e5f` and PC `0x5645` observed;
- service: the exact 16-word hardware descriptor at `0x030bb000` and
  `0x030bb000` stored at `0x030bf000`.

## Results

| Delta set | Maps | Publisher pass | Service entered | Observed solutions | Inconclusive | Unknown | Budget | Any observed descriptor/alive store |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `{0,0x5c,0x100}` | 2,226 | 214 | 214 | 0 | 738 | 710 | 28 | 0 |
| `{0x5c,0x100,0x244}` | 2,226 | 188 | 188 | 0 | 1,426 | 1,406 | 20 | 0 |

For `{0,0x5c,0x100}`, every publisher-valid map remains inconclusive: 176 stop
at `Unknown 0x8cb1`, 22 at `Unknown 0x26d4`, 2 at `Unknown 0xb7`, and 14 hit the
100,000-instruction horizon in the service cone. Replaying those 14 budget
assignments to 1,000,000 instructions still observes zero descriptor stores,
zero alive-pointer stores, zero solutions, and one common service boundary set;
all 14 remain budgets. Another 14 budget exits and 510 `Unknown` exits fail to
reach the publisher landmark. CPU/MMU plus observed-store-state recurrence
closes 1,488 other publisher failures.

For `{0x5c,0x100,0x244}`, every publisher-valid map is likewise inconclusive:
172 stop at `Unknown 0x8cb1` and 16 at `Unknown 0x26d4`. Its 20 budget exits and
1,218 other `Unknown` exits never satisfy the publisher predicate. Recurrence
closes the remaining 800 publisher failures.

The `+0x244` candidate is byte-derived rather than arbitrary: file `0x8edc`
contains the only additional exact eight-byte copy of the AT publisher entry at
file `0x8d98`. It is therefore a concrete byte-derived third framing suggested
by the raw low-image bytes. It produces no observed two-landmark solution.

## Re-derived boundary counterexample

The search confirms the brief's variable-length-instruction warning. This map:

```text
delta_lo=+0x100 delta_hi=0 split=0x8cae literal_delta=+0x5c
```

does preserve the publisher landmark and enters the service without assigning
two values to a shared byte. Its service-side boundaries are re-derived as:

```text
0x8cae/3 L8ui
0x8cb1/2 MoviN
0x8cb3/2 MoviN
0x8cb5/3 Or
0x8cb8/3 S8i
0x8cbb/2 S32iN
0x8cbd/3 Wsr
0x8cc0/2 RetwN
```

All 14 publisher-valid budget assignments converge to exactly that service
boundary set. It executes `RetwN` and continues running, but does not perform
the device-SRAM publish through 1,000,000 steps: every replay records a
descriptor store mask of `0x0000`, no `FW_ALIVE_OFF` store, and final alive
value zero. Thus a different framing can dissolve the *local asserted-boundary
conflict*, but no candidate was observed to satisfy the service landmark.

## Exhaustion boundary

The tables contain a row for every map in both finite families, but neither is
an exhaustion proof: `Unknown` and `budget` rows remain reachability-unknown.
The extended replay is evidence that the one publisher-valid budget class does
not publish promptly; it does not prove non-reachability forever. No final
result produced `split-literal` or `flix-store`, but those guards remain needed
for a wider delta sweep.

A universal genuine-collision result would also need a finite, justified delta
universe. A uniform/full-region map admits 212,415 nonnegative deltas through
`+0x33dbe`; a short `delta_lo` prefix can admit still larger values. Naive
ordered delta-pair/split enumeration is therefore trillions of maps. The
faithful next step, if the stronger quantifier is required, is first to resolve
the reached `Unknown` words with an independent Xtensa decode oracle or missing
semantics, then add execution checkpoints and byte-equivalence pruning in
`src/firmware/`. PSP-loader RE, self-modifying-code speculation, and
firmware-state injection remain closed.

## Reproduction and retained evidence

```bash
XDNA_FW_PROBE=1 XDNA_FW_DELTAS=0,0x5c,0x100 XDNA_FW_MAX=100000 \
  cargo test --lib m2c_probe_execution_guided_framing_search -- --nocapture

XDNA_FW_PROBE=1 XDNA_FW_DELTAS=0x5c,0x100,0x244 XDNA_FW_MAX=100000 \
  cargo test --lib m2c_probe_execution_guided_framing_search -- --nocapture
```

Machine-readable tables are retained under `build/experiments/firmware-re/`:

- `delta-split-search-deltas-0-5c-100.tsv`;
- `delta-split-search-deltas-5c-100-244.tsv`;
- `delta-split-search-publisher-valid-1m.tsv`.

The search probe deliberately exits nonzero after writing its table whenever
any candidate remains inconclusive. This prevents `Unknown`, budget, seam, or
bundle-observation gaps from being mistaken for a formal exhaustion proof.

No production map, MMU behavior, firmware state, or hardware state changed.
