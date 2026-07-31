# NPU1 Phase 3A Executing-Driver Qualification

**Qualification date:** 2026-07-31

**Status:** **QUALIFIED FOR ONE REVIEWED BOUNDED LOAD**

The candidate was not loaded during the original qualification. Subsequent
campaigns `physical-vertical-20260731-02` and `-03` loaded it successfully but
stopped during pre-traffic instrumentation setup and restored the original
module. Neither submitted a workload command or invoked device recovery.

## Verdict

The signed candidate at:

```text
build/experiments/npu1-firmware-evidence/module-qualification-20260731/artifacts/amdxdna-signed.ko
```

is qualified for the separately approved Phase 3A physical vertical pair. It
matches the running kernel ABI, comes from the exact clean driver-protocol
revision, retains normal 2,000 ms TDR, exposes writable `force_cmdlist`, and
contains the complete statically required lifecycle and request-observation
surface.

The pre-traffic attempts verified the loaded bytes, build ID, parameter
readbacks, device node, and D0 pin. Task 11 must still verify the corrected
tracefs events and compiled dynamic-debug selectors before the first NPU
submission.

## Candidate Identity

| Field | Value |
|---|---|
| Source repository | `https://github.com/FIM43-Redeye/xdna-driver.git` |
| Source revision | `216cefececd74effcd7a88350c71b99f5ef9a215` |
| Full source tree | `3e67cf1bdda94602c822a0c8708e88401f1561b9` |
| Driver subtree | `ccdd88a71e37a32a3c888866945f33e9eca88e88` |
| Unsigned SHA-256 | `e701cfa38ee976f0d6bd61d876d4e4631a7cc199b7ab887048358c700dd3c32d` |
| Signed SHA-256 | `a0f8ae51f67fd0fa43accf9b13f99b2c5165e72c2b3a9f10c08160914e443093` |
| Signed size | 6,196,902 bytes |
| Module version | `0.1` |
| `srcversion` | `EF2F7855FE7AB3600D4DFA7` |
| `vermagic` | `7.1.5-custom+ SMP preempt mod_unload modversions RANDSTRUCT_1cbe4ec027729cf8d79983540e3d54aa62bd1bfb29d09d65b6b0b15622ad626b` |
| Dependencies | `drm,drm_shmem_helper,amd-pmf,hwmon,gpu-sched` |
| Signature | PKCS#7, SHA-512 |
| Signer | `triple-Laptop-16-AMD-Ryzen-7040 Secure Boot Module Signature key` |
| Signature key ID | `29:8B:2E:6E:09:80:D8:B7:49:EA:65:CB:9D:71:24:F0:BC:9A:08:4A` |

The signature key ID matches the currently loaded system module. The public
certificate is enrolled under Secure Boot. The private key was neither copied
nor recorded.

## Source and Build Provenance

A detached worktree was created at:

```text
/home/triple/npu-work/xdna-driver-phase3a-qualification
```

It was clean at exact revision `216cefe...`. Only the tracked primary driver,
compatibility probe, and headers were staged with `git archive`; the dirty
legacy `src/driver` tree and dirty XRT submodule state from the ordinary
checkout did not enter the build. All 62 staged tracked inputs reproduced
their Git blob identities exactly.

The module was built locally against:

| Input | Identity |
|---|---|
| Kernel release | `7.1.5-custom+` |
| Headers | `/usr/src/linux-headers-7.1.5-custom+` |
| Kernel `.config` SHA-256 | `eeae7c8c7586ec679cd9a3f03ae2d37a1738a77ec7d1aad92d43c10c2cdd19b3` |
| `Module.symvers` SHA-256 | `fe3f54b887a8ed3d808ece07a552173e1e4b63afe5162335afebca6697ffb893` |
| Generated compatibility header SHA-256 | `ade944ccb6bb00ee9f3812d8b26fd9e0a0b5088d42884ef5acf8e3f17d23218f` |
| Compiler | Ubuntu Clang 21.1.8 |
| Linker | Ubuntu LLD 21.1.8 |
| LTO mode | Full Clang LTO, matching the kernel configuration |
| GNU Make | 4.4.1 |
| `pahole` | 1.31 |

The exact non-secret recipe is:

```text
build/experiments/npu1-firmware-evidence/module-qualification-20260731/build-recipe.txt
```

Its SHA-256 is
`516f8634c35c5ab73a97755b185bdc9b6e2c9a41be7b8a72558f6778d6f657f3`.
The configuration and build logs have SHA-256 values
`083f9f7eef1fa5b4d3a53d10270196bb1a2f95c0d5fe985720fab18ae7481264`
and
`adf73a05784906a7164a33afc4a4cfe8b2fe972e84bb9d65a2a4b2a95257c583`.

## Required Static Surface

The candidate advertises:

- `force_cmdlist`, defined as mode `0600`;
- `tdr_timeout_ms`, defined as mode `0400` with default `2000`;
- `aie2_max_col`; and
- `force_iova`.

For this kernel compatibility result, the DRM scheduler timeout is derived
from `tdr_timeout_ms`; the standalone legacy-kernel TDR implementation is not
selected.

All required tracepoint names are present in the candidate image:

```text
xdna_job
mbox_set_tail
mbox_set_head
mbox_irq_handle
mbox_rx_worker
mbox_poll_handle
uc_irq_handle
uc_wakeup
```

The exact module image's `__dyndbg` descriptors contain the six callsites
needed for request and response bytes. Their compiled locations, rather than
the macro-invocation lines in source, are:

```text
file aie2_message.c line 1077 +p
file amdxdna_mailbox.c line 192 +p
file amdxdna_mailbox.c line 236 +p
file amdxdna_mailbox.c line 271 +p
file amdxdna_mailbox.c line 461 +p
file amdxdna_mailbox_helper.c line 49 +p
```

The previously listed `aie2_ctx.c:300/356` sites use `XDNA_DBG`/`drm_dbg`.
This kernel has `CONFIG_DYNAMIC_DEBUG=y` but not
`CONFIG_DRM_USE_DYNAMIC_DEBUG`, so those sites do not exist in either the
candidate's `__dyndbg` section or the live dynamic-debug control surface.
They are not needed: the direct response is one status word, and
`cmd_chain_resp` places status first in its three-word response. The common
`resp data:` callsite captures those exact bytes. The other two command-list
success-path words remain explicit unknowns.

## Current Module Re-audit

The current loaded and system-resolved module remains:

```text
/lib/modules/7.1.5-custom+/kernel/drivers/accel/amdxdna/amdxdna.ko
```

Its SHA-256 is
`9b403eb8d34f0a66f385e6918bba1ebf86da5b527393280047588196b2d16297`
and its `srcversion` is `77910A99EDBD0B6C78C8053`. The live `srcversion`
matches the file. The module remains unqualified for the physical pair because
its exact source-to-bytes relationship is unavailable, it contains only four
of the eight required lifecycle events, it has no `tdr_timeout_ms` parameter,
and its in-source scheduler timeout is 60 seconds.

## Autosuspend Correction

The unowned local file:

```text
/etc/modprobe.d/amdxdna-devel.conf
```

contained only:

```text
options amdxdna autosuspend_ms=-1
```

That parameter belongs to the legacy `src/driver` implementation. Neither the
current upstream kernel driver nor the pinned primary candidate exposes it.
Both primary sources implement runtime PM internally with a fixed 5,000 ms
autosuspend delay, call `pm_runtime_use_autosuspend()`, and balance activity
with runtime resume/get and autosuspend put operations.

The physical campaign still deliberately pins the frozen tuple to D0. Because
candidate probe re-enables normal runtime PM, the bounded transaction must
write and read back PCI `power/control=on` after each candidate load. This is a
campaign-state control, not a legacy module option or driver patch. A
pre-traffic rollback restores the original module's recorded `auto` or `on`
policy rather than mistaking normal upstream `auto` for provenance drift.

The stale file, SHA-256
`46f256e0bc11afe79e786b90e51eb067c4c6e2fe92d7eed64a2c830f31b9acff`,
was preserved at:

```text
build/experiments/npu1-firmware-evidence/module-qualification-20260731/removed-amdxdna-devel.conf
```

and then removed from `/etc/modprobe.d`. No package owns it. The independent
`dyndbg=+p` configuration remains; dynamic debug is supported by this kernel
and both module images.

## Restoration Proof

After removing the stale legacy option, ordinary offline dependency resolution
ends in:

```text
insmod /lib/modules/7.1.5-custom+/kernel/drivers/accel/amdxdna/amdxdna.ko dyndbg=+p
```

The resolved bytes still hash to `9b403e...d16297`. The manual normal
restoration path is therefore:

```text
rmmod amdxdna
modprobe amdxdna
```

The first physical attempt disproved the earlier direct-`insmod` rollback.
`modprobe -r amdxdna` also removed the unused `amd-pmf` dependency, so both
candidate loading and direct original-module rollback failed with the kernel's
`Unknown symbol amd_pmf_get_npu_data` error. No workload command was submitted,
and ordinary `modprobe amdxdna` restored the original module and dependency.

The corrected bounded candidate-load path preserves dependencies:

```text
rmmod amdxdna
insmod <qualified-candidate-path> tdr_timeout_ms=2000
```

Before the first NPU submission only, rollback removes any loaded candidate
with `rmmod amdxdna`, restores dependencies and normal configuration with
`modprobe amdxdna`, and verifies the original path, hash, srcversion, and build
ID plus the recorded initial `power/control` policy. No rollback, reload, or
recovery is automatic after traffic.

## Qualification Artifacts

The ignored local qualification root contains:

- the exact staged build inputs and generated compatibility header;
- the configuration and module-build logs;
- unsigned and signed module bytes;
- the non-secret build recipe;
- `qualified-module.json`; and
- the removed one-line legacy configuration for recovery.

`qualified-module.json` carries the signed candidate, original system module,
source, recipe, tracepoint, and dynamic-debug identities needed by the Task 11
preflight. These local artifacts are not committed as repository payloads.
