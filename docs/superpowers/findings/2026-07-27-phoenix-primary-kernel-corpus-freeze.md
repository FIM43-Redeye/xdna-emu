# Phoenix Primary Firmware Acceptance Kernel Freeze

**Date:** 2026-07-27

**Status:** Frozen functional corpus entry

## Purpose

`add_one_using_dma` is the first independently validated kernel admitted to
the pinned Phoenix firmware-equivalence corpus. It is the functional payload
for the first driver-shaped firmware -> array -> firmware round trip.

This entry freezes exact bytes. A rebuilt artifact is a new candidate and must
receive new hashes plus fresh hardware/emulator validation before replacing
this entry.

## Source

mlir-aie commit:
`cce2910aadb181d35ddcaa12ace8b9b46082639b`

The wider mlir-aie worktree contained unrelated submodule and untracked
changes, but `test/npu-xrt/add_one_using_dma/` was clean at validation time.

| Relative path | SHA-256 |
|---|---|
| `test/npu-xrt/add_one_using_dma/aie.mlir` | `3ad5c4d4fe8427644c9121f8ea3a08901c4ea8f278d6053e194fdb7a28edba13` |
| `test/npu-xrt/add_one_using_dma/run.lit` | `745bf9b13ec5ac4d8e827bc8a70d10e9e90c153b4e8ba931f5d1979b136b3931` |
| `test/npu-xrt/add_one_using_dma/test.cpp` | `83540703a41bf52412e8357f7975fd1c6a2e539ecf73cb5d7d293c0d79cfb189` |

Other open toolchain commits:

- llvm-aie: `384c388ee9d0ac7011cdcb8acf01ec1743e56d2d`
- aie-rt: `6ee6a4da5f55bd66d278d5032108f0ebe920a501`

## Frozen Artifacts

Paths are relative to the mlir-aie root.

| Compiler | Artifact | Bytes | SHA-256 |
|---|---|---:|---|
| Chess | `build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin` | 9,671 | `c46198460a07ff2aa03a12b125851a223eeb1e8c315132d60aec18d831453bf6` |
| Peano | `build/test/npu-xrt/add_one_using_dma/peano/aie.xclbin` | 9,062 | `71deb139ac91bba3a50099bfd0c3a4a966f00e1977eab017589ef51a36d63865` |
| Both | `build/test/npu-xrt/add_one_using_dma/{chess,peano}/insts.bin` | 300 | `ee49b0a66c53d3952604460fe83fab879f38f1dad6cb70a994fc4422aa285896` |
| Host | `build/test/npu-xrt/add_one_using_dma/test.exe` | 506,792 | `511d40e38eecf70def29322b5af8ce261bb79dfb793dc0ca45abc8a8f99b8806` |

The Chess xclbin UUID is `de135b5a-0400-75b4-a398-38c755874dae`.

## Contract

The host supplies 64 little-endian `u32` values `1..=64`. Successful execution:

1. reaches `ERT_CMD_STATE_COMPLETED`; and
2. returns 64 `u32` values where output element `i` is `i + 2`.

This is a functional acceptance payload. It does not by itself license any
firmware, DMA, or array timing claim.

## Validation

Emulator commit:
`bf48738096516152afcb5eb629a002a9a85b6a17`

Pinned firmware:

- path: `/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin`
- SHA-256:
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

Hardware:

- PCI function: `c6:00.1`
- PCI ID: `1022:1502`
- kernel: `7.1.5-custom+`
- XRT: `2.23.0`, hash `619107cda022463cff13d718247c2a77106de5e4`

Command:

```bash
./scripts/emu-bridge-test.sh --no-trace -v '^add_one_using_dma$'
```

Result:

| Compiler | Hardware | Emulator |
|---|---|---|
| Chess | PASS | PASS |
| Peano | PASS | PASS |

The run rebuilt `test.exe` from the current lit command, reused the exact
hashed xclbins and instruction stream, rebuilt the worktree FFI from source,
and exercised the normal XRT bridge path on both targets.

