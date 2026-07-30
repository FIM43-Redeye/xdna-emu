# NPU1 Historical Corpus Census

**Date:** 2026-07-29

**Status:** Complete

**Scope:** Read-only live-tuple freeze and family-level census of the two known
NPU1 research-corpus roots

## Method

Source locations use stable aliases:

- `repo-experiments` is `build/experiments` beneath the xdna-emu checkout used
  for this scan.
- `workspace-experiments` is `experiments` beneath the containing `npu-work`
  workspace.

The scan does not follow symlinks or cross filesystem boundaries. Family
counts include each family path itself. Apparent and allocated sizes use GNU
`du` default hardlink semantics. Modification times are organizational clues,
not provenance.

No family receives replica credit. No corpus file is moved, written, hashed in
bulk, or interpreted beyond small provenance markers during this pass.

Each root was enumerated with `find -xdev`. Each family was measured twice
with `find -xdev -printf`, `du -sx --block-size=1`, and
`du -sx --apparent-size --block-size=1`; separate `find` passes identified
unreadable entries, broken links, multiply-linked regular files, and the
approved provenance-marker names. Time bounds are shown in the host's
`America/New_York` timezone.

## Scan Identity

| Field | Value |
|-------|-------|
| Scan start | `2026-07-29T22:05:35-04:00` |
| xdna-emu branch | `investigate/firmware-priors` |
| xdna-emu commit | `4596c28c04e776179dce11f8cdb9ac94948c5333` |
| xdna-emu preflight state | Clean |
| Pre-scan `repo-experiments` metadata fingerprint | `779de93324c8abde20043d6240c3a6cff6b430bbfecf3e31510fb580cef89149` |
| Pre-scan `workspace-experiments` metadata fingerprint | `6e4109e53c700265bfad4c80b24459b35fd38a0a1e8d85c1ab1e866f581a478a` |

The metadata fingerprint covers relative path, entry type, mode, byte size,
modification time, device/inode/link count, and symlink target. It excludes
access time and file contents.

## Current Live NPU1 Tuple

| Component | Current value | Source | State |
|-----------|---------------|--------|-------|
| Physical device | One NPU1 function at `0000:c6:00.1`, AMD `1022:1502` | `lspci -Dnn -d 1022:1502` | Observed live |
| PCI subsystem | Framework Computer `f111:0005` | `lspci -Dnnvv` | Observed live |
| Bound driver | `amdxdna`; driver initialization reports version `0.8.0` | PCI sysfs link and boot `dmesg` | Observed live |
| IOMMU group | Group `29` | PCI sysfs link and `lspci` | Observed live |
| Host kernel | `7.1.5-custom+` | `uname -r` | Observed live |
| Loaded amdxdna module | In-tree `amdxdna.ko`; SHA-256 `9b403eb8d34f0a66f385e6918bba1ebf86da5b527393280047588196b2d16297`; `srcversion` `77910A99EDBD0B6C78C8053` | `modinfo` and `sha256sum` | Observed live; source commit unknown |
| amdxdna parameters | `force_iova`, `force_cmdlist`, and `aie2_max_col` are mode `0600` and unreadable without privilege | parameter file metadata and denied reads | Unknown values |
| Primary firmware | Logical `amdnpu/1502_00/npu_7.sbin`, packaged as `npu.sbin.1.5.5.391.zst`; compressed SHA-256 `18413016d44a41cbcf3dacf9d4f7cc22a1927f4b1432a4a9e0f953b0918fedb8`; decompressed SHA-256 `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e` | Boot `dmesg`, symlink resolution, `zstdcat`, `sha256sum` | Observed live and content-verified |
| XRT packages | `xrt-base` `2.23.0`; `xrt-npu` `2.23.0`; `xrt_plugin-amdxdna` `2.23.1` | `dpkg-query` | Observed live |
| XRT libraries | `libxrt_core.so.2.23.0` SHA-256 `69d585730b671dfbe6c48fa7000e398803880fac4ce204c9c274e50d47017fdd`; `libxrt_coreutil.so.2.23.0` SHA-256 `461d3a9de0db09080ea1ad6e66476f012f983bc186772f14730d7eb03c356e76`; `libxrt_driver_xdna.so.2.23.0` SHA-256 `4d6ed092a3ed805edd93053561b02946daa1187c3135a39674630b604455fd91` | resolved library files and `sha256sum` | Observed live |
| aie-rt source | `xlnx_rel_v2026.1` at `6ee6a4da5f55bd66d278d5032108f0ebe920a501`, clean | Git | Observed checkout |
| mlir-aie source | `trace-mode1-decode` at `cce2910aadb181d35ddcaa12ace8b9b46082639b`, dirty: 2 modified and 11 untracked entries | Git | Observed checkout; not a clean revision |
| llvm-aie source | `aie-public` at `384c388ee9d0ac7011cdcb8acf01ec1743e56d2d`, clean | Git | Observed checkout |
| AM025 register database | 2,639,535 bytes; SHA-256 `c5a40ea762f70a5d2728d63370a8ad66ae88d3420c5887604491c4cec9b55396` | selected mlir-aie file plus `stat` and `sha256sum` | Content-verified |
| xdna-driver source | `emu-shim-base` at `216cefececd74effcd7a88350c71b99f5ef9a215`, dirty: 4 modified entries | Git | Observed checkout; not proven source of loaded in-tree module |
| Address/IOMMU state | NPU in IOMMU group `29`; kernel default domain `Translated`, lazy TLB invalidation; actual `force_iova` unreadable | sysfs and boot `dmesg` | Partly observed, parameter unknown |
| Reset, power, and clock state | PCI runtime status `active`, power control `on`, link `16.0 GT/s` x16; reset method and internal NPU clock state unavailable | PCI sysfs | Partly observed |

The tuple uses live system and source-control state only. Historical capture
metadata is not used to fill this table.

The boot log also reports that an `autosuspend_ms` amdxdna parameter was
supplied but ignored because the running in-tree module does not define it.
The installed module advertises production names rather than
`npu.dev.sbin`; nevertheless, decompression of the loaded
`npu.sbin.1.5.5.391.zst` yields 248,592 bytes that are byte-identical to the
installed `npu.dev.sbin` and have the pinned `d13ff9...` digest. This is a
verified packaging-name difference, not a firmware-content difference.

The unversioned `libxrt_driver_xdna.so` link is absent, while the versioned
`libxrt_driver_xdna.so.2` link and `2.23.0` file are present. No conclusion
about runtime loader behavior is drawn from that packaging detail.

## Family Census

| Row | Family | Disposition | Entries | Allocated bytes | Apparent bytes | Time bounds | Provenance markers | Hazards and rationale |
|-----|--------|-------------|---------|-----------------|----------------|-------------|--------------------|-----------------------|
| F | `repo-experiments/firmware-post-alive` | `npu1-relevant` | 7: 5 files, 2 dirs, 0 links | 2,740,224 | 2,725,404 | 2026-07-27 14:21:21 to 14:22:41 | 1 `SHA256SUMS` | Post-alive firmware telemetry, perf data, and trigger log. The checksum file uses absolute source paths; no tuple is present. |
| F | `repo-experiments/firmware-re` | `npu1-relevant` | 9: 8 files, 1 dir, 0 links | 9,986,048 | 9,967,848 | 2026-07-27 00:34:54 to 19:30:11 | 0 | Firmware and create-context logs, including a large timeline. No tuple, manifest, or checksum marker. |
| F | `repo-experiments/phoenix-vfio-user` | `npu1-relevant` | 9,388: 6,896 files, 2,113 dirs, 379 links | 5,508,984,832 | 5,493,414,675 | 2026-01-05 08:30:24 to 2026-07-29 17:07:54 | 30 `tuple.txt` | Firmware/KVM/vfio-user captures and traces. The links are non-broken guest-root BusyBox links, not replicas. This dominant family is ignored beneath the linked worktree's build directory. |
| F | `repo-experiments/sp3-spike-trace` | `npu1-relevant` | 2: 1 file, 1 dir, 0 links | 4,096 | 3,625 | 2026-07-26 20:24:14 | 0 | Targeted hardware-gate script, not a completed capture. It contains checkout-specific absolute paths and no provenance marker. |
| F | `repo-experiments/sp5-skew` | `npu1-relevant` | 6: 5 files, 1 dir, 0 links | 40,960 | 28,336 | 2026-07-26 20:24:14 | 0 | NPU timing/skew scripts, tally, and normalization material. Prior interpretations remain unverified and there is no capture tuple. |
| F | `repo-experiments/transaction-elf-debug` | `npu1-relevant` | 4: 3 files, 1 dir, 0 links | 12,288 | 703 | 2026-07-29 16:41:42 to 16:42:00 | 0 | Small before/after transaction and control-code dumps plus XRT configuration. No provenance marker. |
| F | `workspace-experiments/bios-psp` | `mixed` | 9: 8 files, 1 dir, 0 links | 72,388,608 | 72,376,797 | 2026-03-23 07:44:32 to 2026-07-11 12:27:10 | 0 | Whole platform BIOS capsule and PSP extraction material overlap the NPU firmware-staging investigation; the relevant boundary is not safe to infer from names. Proprietary payloads require local-only handling. |
| F | `workspace-experiments/dkms-accel-install` | `npu1-relevant` | 6: 5 files, 1 dir, 0 links | 147,456 | 135,877 | 2026-05-21 00:19:48 to 00:21:51 | 0 | amdxdna DKMS 2.23.0 installation artifacts and log. This is driver/toolchain-environment provenance, not silicon behavior. |
| F | `workspace-experiments/dkms-llvm-1377` | `npu1-relevant` | 6: 5 files, 1 dir, 0 links | 20,480 | 16,770 | 2026-06-01 19:58:50 to 20:02:24 | 0 | amdxdna DKMS compiler-probe artifacts for LLVM issue 1377. Build evidence only; no hardware observation. |
| F | `workspace-experiments/fidelity-snapshot-20260626.md` | `npu1-relevant` | 1: 1 file, 0 dirs, 0 links | 12,288 | 10,886 | 2026-06-26 13:32:52 | 0 | Historical emulator-versus-hardware trace-status snapshot. Its `CLEAN`, `DIVERGE`, and `ERROR` labels are prior results, not findings revalidated by this census. |
| F | `workspace-experiments/firmware` | `npu1-relevant` | 6: 4 files, 2 dirs, 0 links | 753,664 | 748,393 | 2026-05-24 14:55:11 to 14:56:06 | 0 | Phoenix firmware-version archive containing three binary images and a README. The current live hash matches the digest written in that README for 1.5.5.391; the archived blob and the README's other claims were not reverified. Payloads are non-redistributable by default. |
| F | `workspace-experiments/fw16-bios` | `mixed` | 76: 73 files, 3 dirs, 0 links | 84,766,720 | 84,604,732 | 2026-03-23 07:44:32 to 2026-07-12 00:01:20 | 0 | Whole Framework BIOS and extracted PSP firmware plus an NPU/PSP reverse-engineering brief. The brief preserves an older blocker hypothesis, not current truth; proprietary payloads require local-only handling. |
| F | `workspace-experiments/leak-hunt-2026-05-13` | `mixed` | 2,474: 1,604 files, 870 dirs, 0 links | 4,572,200,960 | 4,568,496,220 | 2026-05-13 15:05:10 to 16:02:10 | 0 | Large bridge/emulator trace and leak-diagnostic campaign. It has NPU-facing test context, but sampled hardware directories are empty and cheap metadata cannot separate hardware evidence from emulator-only output. |
| F | `workspace-experiments/npu-memory-bandwidth` | `npu1-relevant` | 2: 1 file, 1 dir, 0 links | 8,192 | 4,996 | 2026-07-19 19:53:48 | 0 | NPU1 bandwidth-campaign design only; it records intended measurements, not completed observations. |
| F | `workspace-experiments/phoenix-survival` | `npu1-relevant` | 933: 838 files, 95 dirs, 0 links | 80,482,304 | 78,369,483 | 2026-06-10 18:17:27 to 2026-06-12 18:24:59 | 0 | High-value vector-fuzzer corpus with NPU outputs, 1 MiB traces, generated kernels, replay artifacts, and a coverage ledger. Archived seeds explicitly lack the input pool needed for regeneration; no approved tuple/manifest marker is present. |
| F | `workspace-experiments/shim-cleanup-2026-05-21` | `npu1-relevant` | 3: 2 files, 1 dir, 0 links | 8,192 | 2,713 | 2026-05-21 00:51:03 | 0 | XRT shim and DKMS compiler-fix diffs. Implementation provenance only, with no hardware result or tuple. |
| F | `workspace-experiments/vector-oracle` | `npu1-relevant` | 28: 25 files, 3 dirs, 0 links | 245,760 | 184,017 | 2026-06-08 13:53:55 to 14:55:02 | 0 | Oracle-comparison spikes plus local copies of proprietary aietools model code. Relevant to vector semantics, but restricted-source material must remain quarantined and must not enter the MIT repository. |

Disposition meanings:

- `npu1-relevant`: evidence about NPU1 or its emulator contract;
- `mixed`: relevant and unrelated material cannot yet be separated safely;
- `excluded-with-reason`: outside NPU1 preservation scope for the stated
  reason; and
- `unknown`: metadata is insufficient for safe classification.

## Preservation Hazards

- Four capture families beneath this linked worktree's build directory are
  ignored: `firmware-post-alive`, `firmware-re`, `phoenix-vfio-user`, and
  `transaction-elf-debug`. Removing the worktree or cleaning ignored build
  output could erase about 5.5 GB, including the selected firmware witness.
  The `sp3-spike-trace` and `sp5-skew` families contain tracked files, so the
  root is not uniformly ignored.
- No location receives replica credit, and Slice A did not prove an independent
  backup for either root.
- Every family summary reported zero unreadable entries, zero broken links,
  and zero multiply-linked regular files.
- Only 31 approved marker files were found across the 17 families: 30
  `tuple.txt` files in `phoenix-vfio-user` and one `SHA256SUMS` file in
  `firmware-post-alive`. Most families therefore lack a self-contained tuple,
  manifest, or checksum record.
- The 379 symlinks in `phoenix-vfio-user` are intact guest-root links. They are
  part of the capture layout and neither independent copies nor broken
  evidence.
- Several scripts and logs use absolute checkout paths. Those paths are clues,
  not durable artifact identities; a later intake must record missing external
  references rather than substitute current namesakes.
- Firmware, BIOS/PSP payloads, and copied aietools model code are proprietary
  or not proven redistributable. They should be preserved outside Git and
  referenced only through derived metadata or hashes.
- Allocated size exceeds apparent size in every family, but only at family
  granularity. The scan records the discrepancy and does not infer that
  individual files are non-sparse or independently stored.

## Unknowns and Contradictions

- The loaded in-tree amdxdna module cannot be tied to the nearby dirty
  `xdna-driver` checkout from the available live metadata.
- The live values of `force_iova`, `force_cmdlist`, and `aie2_max_col` are
  unreadable without privilege. Reset method and internal NPU clock state are
  likewise unavailable.
- The boot command line supplies `autosuspend_ms`, but the loaded amdxdna
  module reports it as an unknown ignored parameter.
- The installed firmware has two names: the live production package selects
  `npu_7.sbin`/1.5.5.391, while `npu.dev.sbin` is a development name.
  Decompression and byte comparison resolve the content question: both are the
  same 248,592-byte image with the pinned digest.
- `mlir-aie` and `xdna-driver` are dirty checkouts. Their named commits do not
  fully describe the local source state.
- Historical result labels, README claims, warning interpretations, and prior
  blocker hypotheses were not promoted. Each remains unverified until a later
  content intake or fresh canonical run supports it.
- Slice A did not content-hash the corpus. Apart from the explicitly selected
  live firmware and register database, corpus integrity remains unknown.

## Selected Deep-Intake Candidate

The approved Pass 2 witness is
`repo-experiments/phoenix-vfio-user/20260729T171244Z-3136359`: 224 regular
files, 70 directories, 12 symlinks, 195,153,920 allocated bytes, and
194,648,370 apparent bytes. Its small provenance and guest-log artifacts
identify the successful frozen Chess command-list path, making it the closest
existing single witness to the normal driver-reachable firmware contract.

The distinct
`repo-experiments/phoenix-vfio-user/20260729T171042Z-3129577` capture records a
successful Peano direct-execution path. It is a companion intake candidate, not
part of the Chess stimulus. Neither historical success marker is treated as a
lifecycle-clean or generalized hardware proof by this census.

## Validation Evidence

| Check | Result |
|-------|--------|
| All top-level families accounted for exactly once | Pass: 17 actual, 17 reported, no difference or duplicate |
| Repeated family summaries stable | Pass: two complete runs matched byte-for-byte; SHA-256 `df7d057de210d3cd99d73f3bdf60d63a269d4b5fda64f6a2e23becfd52ff497b` |
| Post-scan `repo-experiments` metadata fingerprint | Pass: `779de93324c8abde20043d6240c3a6cff6b430bbfecf3e31510fb580cef89149`, unchanged |
| Post-scan `workspace-experiments` metadata fingerprint | Pass: `6e4109e53c700265bfad4c80b24459b35fd38a0a1e8d85c1ab1e866f581a478a`, unchanged |
| Home-directory path leak check | Pass: no expanded home-directory path |
| Repository diff check | Pass: census report only; `git diff --check` clean |
| `cargo test --lib` | Pass: 4,275 passed, 0 failed, 32 ignored |
