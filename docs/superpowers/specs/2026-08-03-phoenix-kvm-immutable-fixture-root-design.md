# Phoenix KVM Immutable Fixture Root

**Date:** 2026-08-03

## Problem

`scripts/phoenix-vfio-user-qemu.sh` hash-pins its Phoenix workloads but reads
them from `mlir-aie/build`. Equivalent mlir-aie rebuilds generate fresh XCLBIN
and PDI UUIDs, so the live build tree cannot also be an immutable evidence
source. This currently breaks frozen Peano and context-repartition preflight
after otherwise valid toolchain rebuilds.

## Boundary

Keep live compiler validation and historical driver validation separate:

- Rust firmware guards continue to consume current mlir-aie output and prove
  parsed topology plus execution semantics.
- The KVM driver gate consumes one canonical local fixture bundle rooted at
  `$NPU_WORK/fixtures/phoenix-vfio-user/v1`.
- The bundle is machine-local, shared by linked worktrees, and outside ignored
  or regenerable repository build directories.
- The KVM script never creates, refreshes, or repairs the bundle. Missing files
  and hash mismatches remain hard failures.

No research-reserve graph, replica policy, new dependency, or configurable
fallback is introduced.

## Bundle Layout

```text
phoenix-vfio-user/v1/
├── add_one_using_dma/
│   ├── test.exe
│   ├── chess/{aie.xclbin,insts.bin}
│   └── peano/{aie.xclbin,insts.bin}
├── add_one_objFifo_elf/
│   ├── test.exe
│   ├── chess/{aie.xclbin,insts.elf}
│   └── peano/{aie.xclbin,insts.elf}
└── device_width/chess/{final.xclbin,insts.bin}
```

The test executable is stored once per workload because both compiler variants
use the same host program.

## Provenance and Acceptance

Populate the bundle once from bytes already present locally:

- frozen Chess and the current XRT 2.26 host executable from the validated
  add-one build;
- frozen Peano from the surviving KVM receipt whose XCLBIN hash is already
  pinned by the script;
- transaction-ELF Chess and Peano from their successful KVM receipts;
- the current `device_width` candidate and its unchanged instruction stream.

Copying bytes does not establish a new proof. Existing hashes must match before
copying. The combined current-host/old-Peano tuple becomes accepted only after
its KVM mode passes. The newly frozen `device_width` candidate is accepted only
if A1 and B complete and A2 reproduces the pinned physical nonresponse. The
script's hash constants remain the compact tracked manifest; the only expected
hash change is the context-repartition XCLBIN candidate.

The existing `--run-context-repartition` positive A2 oracle was invalidated by
the 2026-08-01 correction. Its nonzero exit after the matching A2 nonresponse is
a characterization result, not a fixture failure. Repairing that KVM lifecycle
mode requires a separate design and is outside this path-only change.

## Script Change

Derive the fixture root from the existing `NPU_WORK` path and redirect
`FROZEN_ROOT`, `ELF_ROOT`, and `REPARTITION_ROOT` to the bundle. Keep mlir-aie
resolution for the register database and provenance reporting. Do not add an
environment override or live-build fallback, because either would let an
operator accidentally bypass the frozen boundary.

## Validation

1. Prove current preflight is red for stale Peano and repartition inputs.
2. Verify every source digest before populating the canonical bundle.
3. Run `bash -n scripts/phoenix-vfio-user-qemu.sh` and `git diff --check`.
4. Require green KVM runs for chained and direct Chess/Peano, both
   transaction-ELF variants, and asynchronous errors.
5. Run context repartition once as characterization. Require A1 and B to pass,
   followed by A2 publication without a response and the existing bounded
   engine-stall stop. Do not require or claim its invalid positive oracle.
6. Run `nice -n 19 cargo test --lib` before completion.

The implementation is complete only if live mlir-aie rebuilds can no longer
change any KVM fixture byte, every valid fixture-backed mode passes its existing
output and lifecycle oracle, and context repartition retains its corrected
externally observable nonresponse boundary.
