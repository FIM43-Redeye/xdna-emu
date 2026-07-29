# Phoenix Direct `EXEC_DPU` -- Design

**Date:** 2026-07-29

**Status:** Implemented and verified

## Purpose

Expand the proven Phoenix firmware contract from xclbin-only
`ERT_START_CU` execution to the ELF/module-style `ERT_START_NPU` path selected
by the pinned driver's direct `EXEC_DPU` envelope.

This first slice proves an authentic no-op control-code submission and its
complete host-visible lifecycle. Observable DPU data-plane work follows in a
separate slice.

## Derived Contract

The pinned primary driver commit remains
`216cefececd74effcd7a88350c71b99f5ef9a215`.

- `ERT_START_NPU` with `force_cmdlist=false` selects `MSG_OP_EXEC_DPU`
  (`0x010`).
- `struct exec_dpu_req` is exactly 160 bytes:
  `u64 inst_buf_addr`, `u32 inst_size`, `u32 inst_prop_cnt`, `u32 cu_idx`,
  then 35 payload words.
- The driver copies only the used payload words, so unused fixed-size tail
  bytes are unconstrained.
- The response is one four-byte status word.

XRT's AIE2 ELF path allocates the ELF control text as an instruction BO,
patches its relocations, and places that BO's address and byte size in
`ert_npu_data`. The non-preempt path sets property count to zero and emits
`ERT_START_NPU`.

## Pinned Producer

Use the installed Phoenix validation archive:

```text
/opt/xilinx/xrt/share/amdxdna/bins/xrt_smi_phx.a
```

It is the archive selected by XRT for Phoenix validation. Its latency recipe
uses `validate.xclbin` plus `nop.elf`; XRT's own `TestNPULatency` extracts
those names and executes them through the runner API.

Pinned SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `xrt_smi_phx.a` | `0970f2038ee7dcf33dbc704c2ac55271b94687b5a17181cdd2c9118ff195c508` |
| `recipe_latency.json` | `ca8b824cec50a8e41fda8c873978f363d6c7f52f728edb603bbc001ee96d8fba` |
| `profile_latency.json` | `b44e2a96c10370461afe34f920d3c7ab0900cb8d08bd7064f42dc2a9769d3639` |
| `validate.xclbin` | `64e41d6bf7ce9668fc75bbbe699df9612056349ff81f17f0df524c6b5016ebf4` |
| `nop.elf` | `00338b532eeeea01611a36c916c07963b8085c34a6910e0ea80582bfa76fe00e` |
| `xrt-runner` | `f39e2399ab4d70f6bd646a2ad2b5a2b339cee2339c4c4597073d93dc7e3e6089` |

The duplicate `validate.xclbin` and `nop.elf` archive members are
byte-identical. The extracted hashes remain the acceptance boundary.

The adjacent open-tree `src/shim_ve2/Runner/latency` fixture is deliberately
not used: its xclbin, ELF, and recipe differ from the installed Phoenix
validation tuple.

## Selected Approach

Extend the existing runner with one explicit mode:

```text
./scripts/phoenix-vfio-user-qemu.sh --run-npu-direct
```

The mode reuses the existing pinned driver build, initramfs, normal XRT
runtime, vfio-user server, QEMU launch, firmware, MSI-X checks, evidence
directory, and cleanup.

The host extracts the four validation members, verifies every pinned hash,
copies `xrt-runner` and its `ldd`-derived libraries into the guest, and records
the complete tuple. The guest loads:

```text
amdxdna tdr_timeout_ms=0 force_cmdlist=N
```

It then runs one iteration of the official latency recipe and profile. No
custom DPU producer or synthetic control code is introduced.

## Proof Order

1. Record the unsupported-mode failure before changing the runner.
2. Add the smallest runner and guest-init branches needed for the new mode.
3. Run the KVM guest boundary.
4. If the authentic run fails inside emulated behavior, capture the exact
   request and add the smallest in-process failing guard before changing that
   behavior.
5. If the authentic run already passes, do not manufacture an emulator code
   change. Retain the driver-boundary regression and durable evidence.

## Guest Acceptance

The run must prove:

- the guest observed `force_cmdlist=N`;
- normal `xrt-runner` completed successfully;
- the driver log contains a 160-byte opcode-`0x10` request;
- the same message ID receives a four-byte opcode-`0x10` response;
- no 24-byte opcode-`0x18` execution request occurred;
- context channel 5 published through management source 37;
- the corresponding MSI-X completion reached the driver; and
- the matched `DESTROY_CONTEXT` request and response completed.

The no-op artifact has no output-data oracle. Success claims only the genuine
command/control lifecycle.

## Explicitly Out of Scope

- observable DPU data-plane work;
- control-code argument relocation beyond what the no-op producer naturally
  exercises;
- preemption and full-ELF command forms;
- default chained `PARTIAL_ELF` execution;
- timeout, restart, suspend/resume, and multiple-job resilience;
- timing equivalence; and
- physical-NPU execution.

Existing chained and direct `EXECUTE_BUFFER_CF` proofs remain regression
boundaries.

## Outcome

The pinned normal-XRT KVM path passes at:

```text
build/experiments/phoenix-vfio-user/20260729T192543Z-3358656
```

The guest used `force_cmdlist=N`; `xrt-runner` completed the official latency
recipe; driver message ID `0x1d000001` paired the single 160-byte opcode-`0x10`
request with its four-byte response; no opcode-`0x18` request occurred; and
message ID `0x1d000010` completed the matched `DESTROY_CONTEXT` lifecycle.

The authentic PDI exposed two earlier wrong priors:

1. Phoenix physical column 0 is DPU-reserved, not wholly control-only. It has
   no shim at `(0,0)`, but does contain a memory tile at `(0,1)` and compute
   tiles at `(0,2..5)`. The validation PDI loads and enables `(0,2)`.
2. Context restore installs a non-spanning way-5 DTLB mapping over a still-live
   spanning way-6 entry. The AMD LX7 treats way 6 as the fallback; applying
   QEMU's uniform multi-hit rule stopped the PDI after command 87.

The in-process guard extracts `validate.xclbin` and `nop.elf` from the pinned
archive, drives the same five-column context and direct request through the
unmodified firmware, and proves the entire PDI, management-DMA staging, DPU
program load, core enable, response, and unchanged no-op output. The 4,301-test
library suite is green (4,269 passed, 32 ignored).
