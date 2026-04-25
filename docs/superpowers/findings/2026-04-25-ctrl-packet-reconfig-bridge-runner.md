# A.4 Findings: ctrl_packet_reconfig (2026-04-25)

## Original framing (Phase E, 2026-04-23)

> `ctrl_packet_reconfig`: `insts.bin` not present (test uses
> `aie_run_seq.bin` + `ctrlpkt.bin` convention); cycle pipeline needs a
> code-path for this. **Classification: NO_DATA.**

User's reframe (2026-04-25):
> No fallbacks, no nothing, if there are two binaries they both have to
> be IMPORTANT. Build proper per-test binary discovery.

## Status today: discovery works; bridge-runner integration is the gap

`scripts/emu-bridge-test.sh:629-660` already implements proper binary
discovery via `_discover_aiecc_name`:

- `_discover_instr_binary` parses `--npu-insts-name=<X>` from run.lit
  → returns `aie_run_seq.bin` for ctrl_packet_reconfig.
- `_discover_ctrlpkt_binary` parses `--ctrlpkt-name=<X>` from run.lit
  → returns `ctrlpkt.bin` for ctrl_packet_reconfig.

Both are passed to `bridge-trace-runner` via `--instr` and `--ctrlpkt`
flags respectively (verified at line 769-777 of emu-bridge-test.sh).
The bridge-runner CLI accepts both (`bridge-trace-runner.cpp:165-225`).

So discovery is solved. The actual gap is that the bridge-runner's
HW + EMU execution paths can't replicate what `test.exe` does for
this test class.

## What test.exe does

`test.cpp` for ctrl_packet_reconfig manually orchestrates a two-stage
protocol:

```cpp
auto instr_v = load_instr_binary("aie_run_seq.bin");
auto ctrlPackets = load_instr_binary("ctrlpkt.bin");
// ... allocate BOs, bind to kernel args ...
// submit kernel run with both ctrlpkt and instr in specific arg slots
```

The kernel-arg-slot binding is what the runner's classifier subsystem
is supposed to figure out from the xclbin metadata. Path:
`bridge-trace-runner.cpp:1465-1498`.

## Empirical confirmation

### test.exe direct under EMU: PASSES

```
$ cd .../ctrl_packet_reconfig/chess
$ XDNA_EMU=debug XRT_DEVICE_BDF="ffff:ff:1f.0" ./test.exe
Name: MLIR_AIE
XDNA_EMU_STATUS: halt_reason=completed cycles=46828 max_cycles=0
PASS!
```

`test.exe` correctly applies the ctrlpkt and runs the sequence;
EMU completes in 46828 cycles.

### bridge-trace-runner via cycle pipeline: FAILS both sides

HW side:
```
bridge-trace-runner: xclbin=...chess/aie.xclbin
                     instr=...chess/aie_run_seq.bin
                     trace_out=...trace_hw.chess.bin
bridge-trace-runner: kernel=MLIR_AIE, 8 args
error: kernel did not complete (state=8)
```

EMU side hits a stream-switch routing error on the shim, with
configured slots that don't match the data's pkt_id=0:
```
TileSwitch(0,0): no packet route for pkt_id=0 on slave[5] (South)
  header=0x00000000 -- configured slots:
    slot[0]: en=true id=27 mask=0x1F msel=3 arb=3
    slot[1]: en=true id=26 mask=0x1F msel=3 arb=4
    slot[2]: en=true id=15 mask=0x1F msel=3 arb=5
    slot[3]: en=false ...
[ERROR] Data movement fatal (flush)
XDNA_EMU_STATUS: halt_reason=budget cycles=10000000
```

Diagnosis: the bridge-runner is not actually applying the ctrlpkt
to the device before submitting the run sequence. Without ctrlpkt
applied, the stream-switch routing for pkt_id=0 doesn't get
configured -- the ctrlpkt is what installs that route.

## Root cause hypothesis

The bridge-runner's classifier pipeline (`bridge-trace-runner.cpp:1408-1498`)
binds ctrlpkt and input BOs to specific kernel arg slots. If the
classifier:

- doesn't identify the ctrlpkt slot correctly, OR
- identifies it but the bridge-runner submits the kernel as a
  single kernel-call instead of a ctrlpkt-then-run-seq sequence,

then the device executes the run sequence without first having the
configuration applied. This matches both observations: HW errors out
(state=8 on incomplete kernel) and EMU sees missing stream routes.

The classifier is in `libxdna_emu.so` (per the FFI integration). To
fix, we'd need to either:

1. Have the classifier flag this as "ctrlpkt arg present, must be
   submitted as separate run", and have the bridge-runner honour
   that, OR
2. Have the bridge-runner always submit ctrlpkt as a prior kernel
   call when present.

This is a real bridge-runner bug. Not a discovery problem.

## What this means for thread A

**A.4 closed as miscoded**: the original framing of "build a real
binary discovery layer" does not match the actual bug. Discovery
already works.

**Tracked separately**: bridge-runner ctrlpkt-protocol bug. This is
its own investigation and would be its own commit/PR. Reasonable to
defer until the cycle-budget validation pipeline matures further --
ctrl_packet_reconfig isn't a baseline test (other tests give us the
cycle-diff signal we need).

For now, ctrl_packet_reconfig variants will continue to surface as
NO_DATA / HW_TRACE_BUG in the cycle-diff column. The classifier
correctly reports them as such. No false positives, just a gap in
coverage.
