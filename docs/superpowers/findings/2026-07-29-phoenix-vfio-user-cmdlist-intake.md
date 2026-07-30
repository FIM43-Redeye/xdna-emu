# Phoenix vfio-user Chess Command-List Witness Intake

**Date:** 2026-07-29

**Status:** Complete; retain as a non-canonical historical regression witness

**Source witness:**
`repo-experiments/phoenix-vfio-user/20260729T171244Z-3136359`

## Scope and Evidence Rules

This report audits one immutable legacy capture of the frozen Chess
command-list path. It does not modify the witness, promote a historical pass
marker into a generalized fact, or substitute current files for missing
historical references.

Material statements are classified as:

- **Observed:** directly present in a named witness artifact;
- **Derived:** a bounded interpretation whose inputs and reasoning are named;
  or
- **Unknown:** not recoverable from this witness.

Absolute machine paths in legacy artifacts are reported as provenance defects,
not copied into the stable identity below.

## Immutable Source Identity

| Field | Value | State |
|-------|-------|-------|
| xdna-emu intake branch | `investigate/firmware-priors` | Observed |
| xdna-emu intake base | `7c1491ac8343dd8eb7d903c01adf62aa1586a2e0` | Observed |
| Pre-intake metadata fingerprint | `4d80663aecf902e12c46fac3fcca95955a5ee04a1ba4aaf0397354dcd52d2299` | Observed |
| Regular files | 224 | Observed |
| Directories | 70 | Observed |
| Symlinks | 12 | Observed |
| Allocated bytes | 195,153,920 | Observed |
| Apparent bytes | 194,648,370 | Observed |

The metadata fingerprint covers root-relative path, entry type, mode, size,
modification time, device/inode/link count, and symlink target. It excludes
access time and file contents.

Before intake, the four ignored NPU1 families were copied to the verified
same-filesystem working snapshot
`npu1-research-reserve/snapshots/2026-07-29-pre-slice-b`. That snapshot protects
against linked-worktree cleanup but is a Btrfs reflink copy and receives no
independent-replica credit.

## Recovered Platform and Stimulus

| Field | Recovered value | Evidence | State |
|-------|-----------------|----------|-------|
| Driver source | Commit `216cefececd74effcd7a88350c71b99f5ef9a215`; all 56 captured tracked files match that commit. The development module is SHA-256 `8e4e9f1c398abde1622c92b8980e7b5b66a99092e75bb2cb4a0d91f9b50766fd`, `srcversion` `EF2F7855FE7AB3600D4DFA7`, with `AMDXDNA_DEVEL` and module string `2.21.0_20260728,216cefe...`. | `tuple.txt`; `driver-build.log`; captured source and module | Observed |
| Guest kernel | `7.1.5-custom+`; recorded kernel-image SHA-256 `4c069ffa4da7a3b9e2ab5b16d514a1f0fd208c059221938a2c30e8aa47347bb4`. The module vermagic names the same kernel. | `tuple.txt`; module metadata in `tuple.txt`; `guest.log:2` | Observed |
| Firmware | `amdnpu/1502_00/npu.dev.sbin`, SHA-256 `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`; the captured version response decodes as `1.5.5.391`. | `tuple.txt`; `dmesg.log:1156,1191-1196`; `driver-source/drivers/accel/amdxdna/aie2_msg_priv.h:318-328` | Observed and derived |
| Runtime | XRT base/NPU `2.23.0`, XDNA plugin package `2.23.1`; the three frozen guest XRT libraries match their recorded host-reference hashes. | `tuple.txt`; `guest-libraries.txt`; `guest-root/opt/xilinx/xrt/lib/` | Observed |
| Transport | QEMU package `1:10.2.1+ds-1ubuntu3.1`; libvfio-user source commit `37491ed9af828fc161238dacd82e83ea35a09f87`. | `tuple.txt`; `qemu-command.txt`; `build.log` | Observed |
| Emulator build | The log records a successful local debug build, but neither the xdna-emu commit nor the vfio-user server binary hash was captured. | `build.log`; absence from `tuple.txt` | Observed gap |
| Compiler | Frozen marker `chess`; mlir-aie commit recorded as `cce2910aadb181d35ddcaa12ace8b9b46082639b`. Exact Chess version, compiler binary hash, source state, and build recipe are absent. | `tuple.txt`; `guest-root/run-frozen/compiler` | Partly observed |
| Execution mode | Command list; guest asserts `force_cmdlist=Y`; TDR deliberately disabled with `tdr_timeout_ms=0`. | `tuple.txt`; `guest-root/run-frozen/execution-mode`; `guest-root/init:66-90`; `guest.log:5-6` | Observed |
| Workload | Frozen Chess `add_one_using_dma`: `test.exe`, `aie.xclbin`, and `insts.bin` hashes are preserved and match the recorded external references. | `tuple.txt`; `guest-root/run-frozen/` | Observed |
| Guest command | `test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin`, using only the frozen guest paths. | `guest-root/init:121-130` | Observed |
| Guest/device shape | Q35/KVM guest, 4 vCPUs, 2 GiB RAM, emulated PCI identity `1022:1502`, four workload columns `1,2,3,4`, and 16 advertised MSI-X vectors. | `qemu-command.txt`; `guest.log:82-120`; `dmesg.log:1356` | Observed in emulation |
| Address mode | No vIOMMU; `force_iova=N`; guest programs and reads back a 256 MiB carveout at `0x60000000`. | `tuple.txt`; `guest-root/init:101-119`; `guest.log:9-10` | Observed |
| Reset/power/clock epoch | Only a fresh QEMU invocation is evident. Device reset history, firmware epoch, modeled clock state, and host power state were not captured. | Artifact absence | Unknown |

The guest PCI identity is produced by the vfio-user model. It is not evidence
that this run touched the owned Phoenix silicon.

## Artifact Integrity

| Check | Result | State |
|-------|--------|-------|
| Existing `tuple.txt` hashes | All 12 recorded hashes verify: 3 contained artifacts and 9 external references | Observed during intake |
| Contained regular files | 224 root-relative SHA-256 entries generated | Observed during intake |
| Checksum-list SHA-256 | `e7aaacefa4c8f3606529dd27980397a656b22099a349db59d1c0df84330811e2` | Observed during intake |
| Captured driver commit | All 56 tracked driver files present at the recorded commit match byte-for-byte; no tracked file is missing | Derived by direct Git-object comparison |
| Driver archive | 58 file members match the corresponding expanded-tree files in type, mode, size, mtime, and bytes. Archive UID/GID metadata differs from the user-owned expansion; 82 additional expanded files are build products. | Observed during intake |
| Initramfs forms | `initramfs.cpio.gz` expands byte-for-byte to `initramfs.cpio` | Observed during intake |
| Extracted log views | `lspci.log`, `msix.log`, and `dmesg.log` exactly equal their marked regions in `guest.log`; they are convenient views, not independent witnesses | Observed during intake |
| Symlinks | 12 non-broken internal guest-root BusyBox links: 11 under `bin/` and `sbin/modprobe` | Observed during intake |
| Unreadable material | None | Observed during intake |
| Missing external references | None among the 9 hashed references. Unhashed host dependencies receive no integrity credit. | Observed during intake |
| Post-intake metadata fingerprint | `4d80663aecf902e12c46fac3fcca95955a5ee04a1ba4aaf0397354dcd52d2299`, exactly equal to pre-intake | Observed after intake |

## Observed Outcomes

**Observed:** The guest init verified its frozen inputs and normal XDNA XRT
plugin, rejected the emulator XRT plugin, loaded the captured `amdxdna` module,
created `/dev/accel/accel0`, and read back the requested address-mode
parameters (`guest-root/init:27-119`; `guest.log:1-10`).

The frozen XRT test then:

1. opened the emulated NPU through the normal kernel-driver path;
2. printed exactly one ordered result for each integer from 2 through 65;
3. printed `PASS!`;
4. returned zero to the init script, which printed
   `PHOENIX_FROZEN_PASS chess`; and
5. reached `PHOENIX_DRIVER_PROBE_PASS` after collecting post-run PCI,
   interrupt, and kernel-log state.

**Derived:** Together these facts preserve one end-to-end regression witness
for
`XRT -> amdxdna -> vfio-user -> simulated firmware/array -> amdxdna -> XRT`.
They do not establish equivalence to physical NPU1.

## Firmware, Interrupt, and Driver Lifecycle

- **Observed:** The management mailbox starts on guest IRQ 38. Opcode `0x108`
  returns five words (`dmesg.log:1157,1191-1196`).
- **Derived:** Using the captured `firmware_version_resp` layout, those words
  decode as success and firmware version `1.5.5.391`
  (`driver-source/drivers/accel/amdxdna/aie2_msg_priv.h:318-328`).
- **Observed and derived:** Context creation sends opcode `0x2` for columns
  `1,2,3,4`. Its 76-byte response decodes, using the captured
  `create_ctx_resp`, as success, context ID 5, MSI-X ID 5, and one CQ pair.
  The driver then starts the context mailbox on guest IRQ 29
  (`dmesg.log:1356-1367`;
  `driver-source/drivers/accel/amdxdna/aie2_msg_priv.h:129-150`).
- **Observed:** Opcode `0x106` maps context 5's 64 MiB window at `0x60000000`.
  Opcode `0x11` configures the CU (`dmesg.log:1368-1384`).
- **Observed:** The command-list submission is opcode `0x18`, which the
  captured driver names `MSG_OP_CHAIN_EXEC_NPU`. The request and 12-byte
  response framing are logged, but the response payload words are not
  (`dmesg.log:2305-2319`;
  `driver-source/drivers/accel/amdxdna/aie2_msg_priv.h:9-48,430-434`).
- **Derived:** The captured response handler marks a job complete only when
  its status field is zero; the observed test completion therefore bounds the
  response status to success, while the failure-index words remain unknown
  (`driver-source/drivers/accel/amdxdna/aie2_ctx.c:329-377`).
- **Observed and derived:** `server.log:55-58` records two context X2I-tail
  assertions from source 37 and two `msix=0x20` services. Because
  `0x20 == 1 << 5`, these match the context's returned MSI-X ID. The log
  separately records the global source-46 / `msix=0x4000` lifecycle.
- **Observed:** Teardown stops the context mailbox, sends destroy-context
  opcode `0x3` for context 5, and logs a zero status response
  (`dmesg.log:2320-2325`). The later interrupt snapshot shows 12
  management-mailbox interrupts; the stopped context IRQ is no longer listed
  (`guest.log:114-141`).

**Unknown:** The guest requests a forced poweroff after `sync`, but the capture
contains no QEMU exit status, vfio-user server exit status, server shutdown
marker, module unload, or post-poweroff host observation. The evidence supports
successful job and context teardown, not a complete process lifecycle.

## Warnings, Anomalies, and Recovery

**Observed:** The pass is not warning-clean:

- the out-of-tree unsigned driver taints the guest kernel
  (`dmesg.log:1154-1155`);
- SVA binding fails with `-19` and the driver reports no PASID
  (`dmesg.log:1228-1229`);
- lockdep records seven `dma_buf_vmap` warnings, seven matching
  `dma_buf_vunmap` warnings, and one possible recursive-locking warning in the
  command-list path (`dmesg.log:1231-2230`);
- the driver logs `aie2_hwctx_cfg_debug_bo: Get bo 4 failed`
  (`dmesg.log:1681`); and
- the emulator logs 17 accesses to modeled gated shim tiles
  (`server.log:29-36,60,75-82`).

**Derived:** The SVA/PASID error is consistent with the deliberately absent
vIOMMU and carveout route. The gated-access statement that silicon behavior
would be undefined is emulator-authored diagnostic text, not a hardware
observation. No BUG, Oops, or kernel panic appears in the captured dmesg. The
run continued through its pass markers, but this does not prove the warnings
harmless.

**Unknown:** Because TDR was disabled, timeout detection, recovery, and
post-fault resilience were not exercised.

## Candidate Facts and Explicit Non-Claims

**Derived candidate:** This witness preserves a concrete driver-reachable
contract candidate:

`firmware query -> context create -> host-buffer map -> CU config ->
CHAIN_EXEC_NPU -> context MSI-X acknowledgement -> context destroy`.

It also preserves candidate values for that run: columns 1-4, context 5, MSI-X
ID 5, one CQ pair, the address window, message framing, and the ordered kernel
result. Those candidates are suitable inputs to a current differential run and
to regression tests.

**Unknown beyond this capture:** These are **not yet NPU1 facts**. This capture
executed the real firmware payload inside the emulator's modeled surroundings,
so any response can still depend on an incorrect modeled register, interrupt,
clock, reset, or array seam. Promotion requires agreement with an independent
physical-hardware witness or an authoritative open-source toolchain
definition.

**Observed but unaudited here:** The separate
`repo-experiments/phoenix-vfio-user/20260729T171042Z-3129577` capture records a
Peano direct-execution pass and remains a companion intake candidate. It was
not audited here and contributes no evidence to this Chess command-list
witness.

This intake explicitly does not claim:

- a physical NPU run or physical PCI/MSI-X topology;
- cycle, clock, latency, or distributional accuracy;
- complete PSP, SMU, firmware, array, reset, power, or interrupt behavior;
- warning-free create/execute/destroy lifecycle;
- TDR, fault recovery, preemption, cancellation, or repeated-run resilience;
- determinism across launches;
- coverage of direct execution, Peano output, other kernels, other firmware
  versions, or undocumented/development operations; or
- reproducibility from source without recovering the missing emulator and
  compiler identities.

## Redistributability

The raw witness remains outside Git.

| Material | Intake classification | Reason |
|----------|-----------------------|--------|
| Phoenix firmware | Do not redistribute | Proprietary firmware payload; no redistribution grant is captured |
| Captured amdxdna source/module | Conditional | Source declares GPL-2.0 and the module declares GPL, but redistribution must preserve the applicable notices and source obligations |
| Frozen XRT/system libraries | Unknown; default to do not redistribute | Mixed packages are copied into the initramfs and no package/license inventory was captured |
| Chess-produced executable/XCLBIN/instructions | Unknown; default to do not redistribute | Compiler provenance and output licensing were not captured |
| Whole initramfs or raw corpus | Do not redistribute as a unit | It combines the firmware and multiple incompletely inventoried packages |
| This report, hashes, and bounded factual summaries | Redistributable project metadata | No proprietary payload is embedded |

This is an engineering reserve classification, not a general legal conclusion.

## Missing Canonical Fields and Rerun Requirements

For a canonical replacement witness, capture:

1. a bundle ID and manifest that name every input, output, command, and
   duplicate representation;
2. exact xdna-emu commit/tree state, server binary hash, libvfio-user binary
   hash, build profile, host kernel, environment, and all process exit statuses;
3. exact Chess executable/version/hash, source state, compiler command, and
   build recipe, plus the same fields for a separate Peano witness;
4. guest kernel configuration and hashes for every loaded out-of-tree or
   capture-supplied module;
5. reset, power, firmware-load, and modeled-clock epoch;
6. complete command-response payloads plus firmware/array seam and interrupt
   acknowledgement traces;
7. a warning-clean normal lifecycle, including module/server/QEMU shutdown;
8. repeated identical launches, then separate TDR, recovery, preemption,
   cancellation, timing, direct-execution, and older-firmware cases;
9. the identical frozen workload on the owned physical NPU1, linked to its
   independently validated array result, for differential promotion; and
10. a package and license inventory suitable for deciding what may leave the
    research reserve.

The historical `PASS` is retained as regression evidence. It must not be used
to fill any of these missing fields.

## Validation Evidence

| Check | Result |
|-------|--------|
| Existing tuple checksums | Pass: 12 of 12 |
| Root-relative checksum appendix | Pass: all 224 entries exactly reproduced |
| Ordered guest result sequence | Pass: outputs 2 through 65 in order and exactly one Chess pass marker |
| Immutable witness fingerprint | Pass: pre/post `4d80663aecf902e12c46fac3fcca95955a5ee04a1ba4aaf0397354dcd52d2299` |
| Corpus-root fingerprints | Pass: repo `779de93324c8abde20043d6240c3a6cff6b430bbfecf3e31510fb580cef89149`; workspace `6e4109e53c700265bfad4c80b24459b35fd38a0a1e8d85c1ab1e866f581a478a` |
| Home-directory path leak check | Pass: no expanded home-directory path |
| Repository diff check | Pass: only this intake report; no whitespace errors |
| `cargo test --lib` | Pass: 4,275 passed, 0 failed, 32 ignored |

## Root-Relative SHA-256 Appendix

<!-- CHECKSUMS-BEGIN -->
```text
63df1b101b9fab02207f81bcac7ca49f197a7fd8679a242c6c70e4a585e560ae  ./build.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  ./depmod.log
8fe98fa237101f18d44e6656977112605068f0ce22992cb30afd69552c14b710  ./dmesg.log
4173084079a5de63c911dfcbe756dd64c0c3a2144e8fa913fb75f2b406692596  ./driver-build.log
e79a4b75c3b488a990b6f723ea59115460f8313a2176cecaaf4a157e7c5f5792  ./driver-source.tar
8ffc9362f14bde0c96d3447b74df6bbcf2c494340bde66b865e6bf251ad9a147  ./driver-source/drivers/accel/amdxdna/..module-common.o.cmd
2db56e24dbb18428c4affa1dcecb6e5c4b7d5ab6328574ba59b20259c040dde3  ./driver-source/drivers/accel/amdxdna/.Module.symvers.cmd
506bb2b045f587737d9cc438e6e5eea3240178e40d102d8f246d2a142d31d91d  ./driver-source/drivers/accel/amdxdna/.aie.o.cmd
0070a93371b5c76a491a8f4bca6b05de666fbf4fb94aea8e584d411932a3a919  ./driver-source/drivers/accel/amdxdna/.aie2_ctx.o.cmd
7a834ba453c7b76337b166dd1c304f105ac14622b8daec87c3d4c20f7f840836  ./driver-source/drivers/accel/amdxdna/.aie2_error.o.cmd
bf0003dd30396f47561c2ba17b48690d866711701c10db80f835913d36d7ca2f  ./driver-source/drivers/accel/amdxdna/.aie2_message.o.cmd
41a3fca286516242d85f1fb1fbb20f499769ff66e132be53f3b6ed8c76d5e0b4  ./driver-source/drivers/accel/amdxdna/.aie2_pci.o.cmd
6b7e6beeca0a5c89019941649b29a4c5bacde174cf2e8f45b72498335548e6ad  ./driver-source/drivers/accel/amdxdna/.aie2_pm.o.cmd
c8270a2176cbdcbde6eb70716074997df0757cb4d8c624887ea08c29fc925497  ./driver-source/drivers/accel/amdxdna/.aie2_solver.o.cmd
e6de55805682c2fc533137bd962a87e1c0f7212f1c6699d05779f0458f40c4fc  ./driver-source/drivers/accel/amdxdna/.aie2_tdr.o.cmd
39e376331fa21a684b21985f24d04245692bfc1a37138037393fafdf4b7e15d5  ./driver-source/drivers/accel/amdxdna/.aie4_ctx.o.cmd
d2187e032f7396700ab167d9c5dde629485f6fe70edb796f4c9bcba63ef0d3b4  ./driver-source/drivers/accel/amdxdna/.aie4_message.o.cmd
235b28ec4c3871d97f7f5f38378c4b7cc796216106c3df7494d1174ede9f0373  ./driver-source/drivers/accel/amdxdna/.aie4_pci.o.cmd
b23df208c802ed41c18667c06bcf5e4c049c3cda16990814c54679cab4db97ca  ./driver-source/drivers/accel/amdxdna/.aie4_sriov.o.cmd
c98599b4344e275671abf719d43268c2d77bb169686166cab9bd278bdad5ae3d  ./driver-source/drivers/accel/amdxdna/.aie_psp.o.cmd
7086914288b871da4ef2a7c85798b44c2dd54574aebc22e35bf4dfa4d3114aab  ./driver-source/drivers/accel/amdxdna/.aie_smu.o.cmd
59c8b0d66bda2ee68228e628797b2103d32313e1b387b3dea26d34d6da003afb  ./driver-source/drivers/accel/amdxdna/.amdxdna.ko.cmd
4b67822dda6a87ea01e84f83de7f815d1d26154a4d9a084855adcf0b7a0dd46f  ./driver-source/drivers/accel/amdxdna/.amdxdna.mod.cmd
8827851cae2a0a51ce6bd463e19535ebf1a9726dcbbe545576b819be5d206409  ./driver-source/drivers/accel/amdxdna/.amdxdna.mod.o.cmd
33d1f6d950e017eb345368b284860801b6fdd224d7846700a2f761fb5dd082e0  ./driver-source/drivers/accel/amdxdna/.amdxdna.o.cmd
bda25e08ec0e0052f087984c7367807caf7b5e7c5c95014f7e01eafd4bff2e44  ./driver-source/drivers/accel/amdxdna/.amdxdna_cbuf.o.cmd
1e829dd2e8bad7e1b9168fd5f0e7e99a1706c19f2337d5ad8957b5d6a82b0f1a  ./driver-source/drivers/accel/amdxdna/.amdxdna_ctx.o.cmd
1bc955d9de425df8606f4cbad7086051726c998b14b9e910e82a975c9f40ffd3  ./driver-source/drivers/accel/amdxdna/.amdxdna_debugfs.o.cmd
7866f4c3baf83a8fd3f3f0af7365768a4b90275a35c9de0cd4a77d2f30ff03a3  ./driver-source/drivers/accel/amdxdna/.amdxdna_dpt.o.cmd
0333057259f3d4126a0f309f97554ed133d3a91197b502d900d60fe51bcf2f34  ./driver-source/drivers/accel/amdxdna/.amdxdna_gem.o.cmd
8e2d2ec2496079ce584b2a1f1d1a070649ec42bfb61bbc4f0a858adf2ca68e47  ./driver-source/drivers/accel/amdxdna/.amdxdna_iommu.o.cmd
2d9f68185780f34c4506130020b6fffecee49256c29c41ed99b96ff9a3f32f43  ./driver-source/drivers/accel/amdxdna/.amdxdna_mailbox.o.cmd
5e92f2b3f9b002ce30885bf08b27594728052d7c8ad82a6b45a83e9e369b9465  ./driver-source/drivers/accel/amdxdna/.amdxdna_mailbox_helper.o.cmd
26a02aa1e4de4b4423082b9842b37454b18a544d82c0ca14d5d59926051fe9cc  ./driver-source/drivers/accel/amdxdna/.amdxdna_pci_drv.o.cmd
d0608e33e11c8ffc3399bdba01dd38903f26dd6911e826111e83c3911621a3d0  ./driver-source/drivers/accel/amdxdna/.amdxdna_pm.o.cmd
fbdf9086019d0895f68ab21ac9b00c69910c05cdef896254cbeb9727d5fde242  ./driver-source/drivers/accel/amdxdna/.amdxdna_sensors.o.cmd
55ece732e54669dd62fa0b4c1b4b24a7b51543d2c037c5fdf9ddd88c831fb9e4  ./driver-source/drivers/accel/amdxdna/.amdxdna_sysfs.o.cmd
bac502fa72a84057770cbd25ce1f2da16c3168c02606ee16bd0105047ea819d6  ./driver-source/drivers/accel/amdxdna/.amdxdna_ubuf.o.cmd
a0bc87dd84292730fe508604d70aa2aeeb4229caf73f389a2e3ee7d550e36ffc  ./driver-source/drivers/accel/amdxdna/.amdxdna_xen.o.cmd
fba66db77a49321db9b3ed0c6905be685ad8beb3635784866f1dfe54aee0dffb  ./driver-source/drivers/accel/amdxdna/.module-common.o
6f62626e884f047e83718227039b95a06f097962ad28dbc9301f5c8b5eccb302  ./driver-source/drivers/accel/amdxdna/.modules.order.cmd
dd89da80bfff8949a1b30c7a244d45d89b3f4074d6c784eac99d19fdc68d4c26  ./driver-source/drivers/accel/amdxdna/.npu1_regs.o.cmd
9aa6ffa8499e68b3ae34c9d4bcd8f70fe9d0e0e2944e8b3da09c9ea5e2e8687c  ./driver-source/drivers/accel/amdxdna/.npu3_regs.o.cmd
fd06b4e4e8c62dfc2a0411019a142b7abfd433775a5f97d14ca5e025d9bf0616  ./driver-source/drivers/accel/amdxdna/.npu4_regs.o.cmd
97373add1f4eced83ec551d69a6ed6e436d7d06bc6eba15d5974840567ecf61c  ./driver-source/drivers/accel/amdxdna/.npu5_regs.o.cmd
986db77928e0e8d871d9d0d9859db58585f068cd04dd48191f04d3307194bd48  ./driver-source/drivers/accel/amdxdna/.npu6_regs.o.cmd
139a3f79455e47731ed2d790acc30439a7bb1b6be5b7f6b784fb44b8e36b36b2  ./driver-source/drivers/accel/amdxdna/Kbuild
fb3ebbeb2837bd54bd6668b8bdbbbb6722bef7f3f81abb4206e7eaa58ed58505  ./driver-source/drivers/accel/amdxdna/Makefile
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  ./driver-source/drivers/accel/amdxdna/Module.symvers
b8c6c75f2e0b3095a14800d89f8f44672e51e9a65622a33992de40c5200eb772  ./driver-source/drivers/accel/amdxdna/aie.c
65a87db3bc3ff28cb8cd4b5114ae38ed2006def5e5d63be0064abeb2dd0566da  ./driver-source/drivers/accel/amdxdna/aie.h
628ffe6d8f3fedb03ca8b8821b38328a38fb87865586a25cbfec289cfc542897  ./driver-source/drivers/accel/amdxdna/aie.o
7930d7649c6242cd523ae63cfbda63cce18012eb85ab5974768d574e39c6f268  ./driver-source/drivers/accel/amdxdna/aie2_ctx.c
9aa8b601fd1bc74e7372b91aa6f08377e3325435d257080ab2d2d672828d2174  ./driver-source/drivers/accel/amdxdna/aie2_ctx.o
b7233246deeaeac9c730d4901060bf844a47d06a56c6a85a4f5b52189676ce20  ./driver-source/drivers/accel/amdxdna/aie2_error.c
ce0a98be14dd9d304f6a4c60115900878880929cf7c0eddf1d4359ee7c86b0e6  ./driver-source/drivers/accel/amdxdna/aie2_error.o
95bdacc7c75b15e4f49a3e074e58ab3b4c4f16b4c8da64b1ca7276d06e2f2641  ./driver-source/drivers/accel/amdxdna/aie2_message.c
ece84bdfe4ec40dbd2c9020f66cdf3147a02c865fc9e922a408aaa653f585bd2  ./driver-source/drivers/accel/amdxdna/aie2_message.o
0207321f78d23f3ea58651c6c3806a5a6053467c793dd9a64a8eb5e72786c6d4  ./driver-source/drivers/accel/amdxdna/aie2_msg_priv.h
b02b9d2a9ac25b8c38e8b77ff8bf18f2ff81f4cbf9b9c5ff0375f28075c390ac  ./driver-source/drivers/accel/amdxdna/aie2_pci.c
ec1d409cdce1e24785a464318413b2cf83fd67e6628cbec9db48ae053976a242  ./driver-source/drivers/accel/amdxdna/aie2_pci.h
7cec097a30f894b503d2b71e3d05fec5cb4abde45c8fb757887ba0f4abc018d5  ./driver-source/drivers/accel/amdxdna/aie2_pci.o
9beb1d4cd011a6600b65ba74a5eef534fdbd3d14a32eb98d308a15eaf6e0bd10  ./driver-source/drivers/accel/amdxdna/aie2_pm.c
9d5c77cec6ec3f6652e0cb180e4a7a37e5594b8c0cc07413c43a444641df602f  ./driver-source/drivers/accel/amdxdna/aie2_pm.o
d08d31b10be644ede582c3192e00bec2925e4acbe385a7f5547a139995596bdf  ./driver-source/drivers/accel/amdxdna/aie2_solver.c
9e622956fe93113eaeec9a1f25b54957b39e6ba80f926cced8b2a1fe910a69e5  ./driver-source/drivers/accel/amdxdna/aie2_solver.h
aee4fa0503de1851365fbee904e5ee6539dec67ea8bf21c65c090490d0d59909  ./driver-source/drivers/accel/amdxdna/aie2_solver.o
793248c7b9edf7428e67fc43995d734599e0219d35e04d6bd8ecbdda669418ca  ./driver-source/drivers/accel/amdxdna/aie2_tdr.c
02cf32910c99bbb883122530eb5b37e7c0fcac268e0b62699c36bff671a3aa1d  ./driver-source/drivers/accel/amdxdna/aie2_tdr.o
37b5a865f4abc4432ddf6cdb011a64c27b231e24898317815c20d15fb4827b00  ./driver-source/drivers/accel/amdxdna/aie4_ctx.c
a758cba9136b69e09677a9188e92235265e9bc4df919ae8555d2736d762e6ffe  ./driver-source/drivers/accel/amdxdna/aie4_ctx.o
db44d9707ccc599db5aeea7cf5193dae4e6c7462d4610cf771ad558558c3f05b  ./driver-source/drivers/accel/amdxdna/aie4_host_queue.h
0db9bf9e1dbfcab993a4a8f911030e41d4d6df4301acd26f135d16c23360f812  ./driver-source/drivers/accel/amdxdna/aie4_message.c
93f6796a92c53776cb6eea6739e9df13aeecf1d9b8b18db59a6f0641d4f08b0f  ./driver-source/drivers/accel/amdxdna/aie4_message.o
541fcb3a3dffebe7ec0a4c88697d74a35a4917a0dc57d89dd99b5e15d30b5af6  ./driver-source/drivers/accel/amdxdna/aie4_msg_priv.h
42aea1152475e5a50d723b069a7aafd9ea31814badfc76bd02ebae9bfab4a8c9  ./driver-source/drivers/accel/amdxdna/aie4_pci.c
42d6c2ec7b8324ae3e6d2faa14640b20c26f01721e445aad565bf1507c6df0a8  ./driver-source/drivers/accel/amdxdna/aie4_pci.h
e9c3a5e3427e418e68533f1fdc7ed7b7b13b2ced51a0a79d0b34f8674994d6f5  ./driver-source/drivers/accel/amdxdna/aie4_pci.o
d081a5a0bb356af205cf46f763e232652419602bbfc1cb8a0a44cc3a4821abed  ./driver-source/drivers/accel/amdxdna/aie4_sriov.c
3f8abdf2adc2cffa7f766e1c600cace316cd42cbfb259fa7bf4379e59a4a1088  ./driver-source/drivers/accel/amdxdna/aie4_sriov.o
406d53256ce22af2b561fcf268f3fdd27f794fba3b9c13109007769945fc7ab2  ./driver-source/drivers/accel/amdxdna/aie_psp.c
6c4574632f093c67a7e7bba41077df183e8c201dc2b302d72f3a1d368246665f  ./driver-source/drivers/accel/amdxdna/aie_psp.o
59d045ec9fc81c43068f91d387129480bb612829629be8ad9f9d1504e37f8934  ./driver-source/drivers/accel/amdxdna/aie_smu.c
e2d40597fab29f7107aceacf3a22c26c5bc4692be3293421af8aae9d15bb3358  ./driver-source/drivers/accel/amdxdna/aie_smu.o
8e4e9f1c398abde1622c92b8980e7b5b66a99092e75bb2cb4a0d91f9b50766fd  ./driver-source/drivers/accel/amdxdna/amdxdna.ko
34f5a7b01880c716a5256ac7c945a58031a887b516f07c87fca4f8df59f7a53c  ./driver-source/drivers/accel/amdxdna/amdxdna.mod
0077be495c3b6077d3ce84f503cd39674d4aef4d89b098dfebb2df9b2c22633a  ./driver-source/drivers/accel/amdxdna/amdxdna.mod.c
ca6025116a2755f3d902969f631cbaa97b8830610f99ec9b3158c59e8007e0ae  ./driver-source/drivers/accel/amdxdna/amdxdna.mod.o
60a47831a379abd56ff1ab6fb7a73da3787550ce15c8c6c365a44d473112faed  ./driver-source/drivers/accel/amdxdna/amdxdna.o
603fba40d62b8d9c2c790756023364cf1a13cfb81588fcff9298c38860c3ab5a  ./driver-source/drivers/accel/amdxdna/amdxdna_cbuf.c
3e22f5a00e7d134b2b2f2b44b78e3d260c55601cbf1dfefbfa58d391e2f645ae  ./driver-source/drivers/accel/amdxdna/amdxdna_cbuf.h
7f36a7dcaed931ccf4441119bd6d908b05fce64c252612fd109f25216cb6fe62  ./driver-source/drivers/accel/amdxdna/amdxdna_cbuf.o
4b223c612750fb6c604c62eb23db64ac876825c739be669d1cb137c0e22c049f  ./driver-source/drivers/accel/amdxdna/amdxdna_ctx.c
54a24fb766d373ef4ab674a951f7281abe0fe5e6cdb27512bf39f01a0063a287  ./driver-source/drivers/accel/amdxdna/amdxdna_ctx.h
3130808c56d71d1c0c2e3d0277eabd140964a1947d5bec7368d72ced9cdd3626  ./driver-source/drivers/accel/amdxdna/amdxdna_ctx.o
ac4a6ea705b3b96b414eec213e668a54cadaad6607e6d33f4af596ce24315853  ./driver-source/drivers/accel/amdxdna/amdxdna_debugfs.c
dabf684018c23a6cb6be7211c5dad35edead84c670e9b37a06eeceb8cfdacd2d  ./driver-source/drivers/accel/amdxdna/amdxdna_debugfs.h
3d8e22445931e8df5dce21a63e4e3d557a8b569d3db0695f9c1f4c41eec5b0df  ./driver-source/drivers/accel/amdxdna/amdxdna_debugfs.o
005abf665373242d7c1fc136104fe681add041e16f67a194a88cf533408a544a  ./driver-source/drivers/accel/amdxdna/amdxdna_dpt.c
985e6ca7224a7f33ac77bfe0dcf1738acec360891d61807b4af9c53f24accdeb  ./driver-source/drivers/accel/amdxdna/amdxdna_dpt.h
c068cfad7e40f9e5c559fb30ba06a25c2016a12fd04ed765ae8cf731b9c4b498  ./driver-source/drivers/accel/amdxdna/amdxdna_dpt.o
7721df90b0ca64e1088b47f0c24133e69c9375fbed41a2de96a15ab9451bb98c  ./driver-source/drivers/accel/amdxdna/amdxdna_error.h
90f528de79ef58e951fe816c626b31c238441a0021d3bf4b868a3398a420ae63  ./driver-source/drivers/accel/amdxdna/amdxdna_gem.c
8df394bc891957a09b71c0470b5d99fd3862495b7f4e7a08b2b258b6e7a64e6b  ./driver-source/drivers/accel/amdxdna/amdxdna_gem.h
29a1d45ab4da8fd6dd19ad00143fe70bdc3619d2903bee95dc04a9dbbaaa5156  ./driver-source/drivers/accel/amdxdna/amdxdna_gem.o
18cb721ec40b9d26e2455c6c99494c6bc86804372fcd7fabf9afa989bd774614  ./driver-source/drivers/accel/amdxdna/amdxdna_iommu.c
ae872fb3e8ba917ca5bf26780bb5c78b030964576c72815cd6153fd922ea6f92  ./driver-source/drivers/accel/amdxdna/amdxdna_iommu.o
745e290d4122832d4a5ed6c848932e0f4f072625e2b0a84397b8cda69961cbed  ./driver-source/drivers/accel/amdxdna/amdxdna_mailbox.c
5bfb3263a012b137e0886c431a8ebf508c35022419042c35be931b9f40aea4b1  ./driver-source/drivers/accel/amdxdna/amdxdna_mailbox.h
c6e77f0136a351c7349ff34c55e184b5fd07e60fb2e28bb8626eff4ce5846810  ./driver-source/drivers/accel/amdxdna/amdxdna_mailbox.o
e8bd1945a493b525aa6c183011890687f80d19f40482ec3ce87fa05b8d22428b  ./driver-source/drivers/accel/amdxdna/amdxdna_mailbox_helper.c
56380f017612f8972856b93f088518416a2700ad47e7fe5fa1d601e7c13d7b6a  ./driver-source/drivers/accel/amdxdna/amdxdna_mailbox_helper.h
6cf081d31ec110d6a9aee447b8af5e437e2831b43d1153473e89d1b03712bc80  ./driver-source/drivers/accel/amdxdna/amdxdna_mailbox_helper.o
7c5db81c3cdeb93ea2ff732bd445fa3f211acd97f36f3716c799c26078c9d1e8  ./driver-source/drivers/accel/amdxdna/amdxdna_pci_drv.c
79ef76eea9c5e7401b061b11c50f9c9e990e3adea973ea16ce50101af7e6ffdb  ./driver-source/drivers/accel/amdxdna/amdxdna_pci_drv.h
631b675b1cc627fa75140e63c091845c25dfaf28d8b7a49fa3e03209e300e0d2  ./driver-source/drivers/accel/amdxdna/amdxdna_pci_drv.o
6831f643a3bffdb28a92033cafb9d1af1c05ab4233a7d6004e7895d33d15f828  ./driver-source/drivers/accel/amdxdna/amdxdna_pm.c
0f3003901909e5a334c60cc1e0040c19b2f7bc3412ad812989405704dc449261  ./driver-source/drivers/accel/amdxdna/amdxdna_pm.h
245bafe6d5938eba9d795bf3e45e50d2fa2143556aa6c754d6f9b58c94b93798  ./driver-source/drivers/accel/amdxdna/amdxdna_pm.o
9d1e9f48753a0200d340ccbc1f2d9af47c45e351a4bc835976f349e102bb2a7c  ./driver-source/drivers/accel/amdxdna/amdxdna_sensors.c
45c909d6fa00ecb0dabe250b01a7d9dc5b4c0a3c19b8444b6537b4c3f4b1ffe0  ./driver-source/drivers/accel/amdxdna/amdxdna_sensors.h
0fd2fe95195fd7944fa87395dd8d0c95e8f91a59f7c3be1e5027ca4810e55191  ./driver-source/drivers/accel/amdxdna/amdxdna_sensors.o
7f818960a86e10cd6e5206b1557df0d688acae7d79ce1eeeb44e523943520ea7  ./driver-source/drivers/accel/amdxdna/amdxdna_sysfs.c
4f7e7129be08cc18d526fe26813bbff91e0d1905667fa5b9621b1bcb589ceea4  ./driver-source/drivers/accel/amdxdna/amdxdna_sysfs.o
f64fe660f6c9f133ac9eac9bf9449c6fa552364c70bb8a9426a8f73c8697c407  ./driver-source/drivers/accel/amdxdna/amdxdna_ubuf.c
fd989251a900bcb9c66f406da0f68c079fad46458ae1d92c813b442f28a7bfd9  ./driver-source/drivers/accel/amdxdna/amdxdna_ubuf.h
4cc12fb51c467b9c6e6f6909efd9cce570de9b7a591b5149d438505cd2562356  ./driver-source/drivers/accel/amdxdna/amdxdna_ubuf.o
0adc137bce2409ec4ff03d295c6f1ab54ece42198e5a27845fd153f623a7a115  ./driver-source/drivers/accel/amdxdna/amdxdna_xen.c
38a5d13fc51ec5e13db219fd019e4f4c601966cec6cf32e42fee6373d3985c7c  ./driver-source/drivers/accel/amdxdna/amdxdna_xen.h
265dc97909b0090285caa170dd07065dbaeb163a43c747b6441fee82345b7aeb  ./driver-source/drivers/accel/amdxdna/amdxdna_xen.o
ade944ccb6bb00ee9f3812d8b26fd9e0a0b5088d42884ef5acf8e3f17d23218f  ./driver-source/drivers/accel/amdxdna/config_kernel.h
eb4292f9d1739221c42ce98608ed62fb2ec1c39e54031d67c692d919467d357e  ./driver-source/drivers/accel/amdxdna/modules.order
56bd0d7ff60d42b108342ad9f3d8c8a87d7934a551033687a7ee44e6f07aa1d5  ./driver-source/drivers/accel/amdxdna/npu1_regs.c
9c34668f30bd729c9b76756c2037361f2592719e3bee61def3317de533e5a384  ./driver-source/drivers/accel/amdxdna/npu1_regs.o
3b7676840fd96365731e0f6b492f7f9c0c16eb902020c6e9ca1a824c44b47d6d  ./driver-source/drivers/accel/amdxdna/npu3_regs.c
ff38dca17ef3b99215ec8b708b4782f3159363b4baba350124f64746a11fb0ed  ./driver-source/drivers/accel/amdxdna/npu3_regs.o
0dd1ff6b041534e3db1df6b99ecefdf5cc04ad2aef9db21f971aecb2403c6f24  ./driver-source/drivers/accel/amdxdna/npu4_regs.c
e6c75f9c0ec2f5b6c7865795e932769cc45b5ddb8e73e42924324ffff51a4734  ./driver-source/drivers/accel/amdxdna/npu4_regs.o
97060d33aa0c3f7eb3babac3efacd409bb358057b61aba2a89096067a6a60c50  ./driver-source/drivers/accel/amdxdna/npu5_regs.c
0f8f61d91110a9e82703aaad416ac2380341095cd407d56ed7c965f6be228f38  ./driver-source/drivers/accel/amdxdna/npu5_regs.o
692af1e6c2928d67878ebb010bb73835d4ecda030c0a2b0b4bc5eda2e458c10e  ./driver-source/drivers/accel/amdxdna/npu6_regs.c
350faf4aeb11c1cd59de08870bd15dfa47718d71dce1cd6bd48dee19f5b024e0  ./driver-source/drivers/accel/amdxdna/npu6_regs.o
5175c5926d8bbf1488c45c1619d2f3dfc582bf45eaecd52a5d99a389acf3e10f  ./driver-source/drivers/accel/tools/configure_kernel.sh
afb884153ad8a5ad346284d3575dd019b2f6b6def8f2b2e0ef266e0ee8dda47d  ./driver-source/include/trace/events/amdxdna.h
f84efefc9fe94e2e8f95b5b7bd5d11bf0b1970884780405fb9a7e1eff0ef0b9a  ./driver-source/include/uapi/drm/amdxdna_accel.h
98ea5a10254e413b695a89735b6984bab8dbb6f4526a108f5d463085ba1657c4  ./guest-build.log
864ceb9bfc6e300b67ad39d9a47c5d3bc25bb3952849679d9d46c50b687d9e44  ./guest-libraries.txt
df12634c17fcdca839ae5dc47d7627b7558511f7645de7c99ccf097a0f28ed5b  ./guest-root/bin/busybox
5f182f711fed06c94664b098e82ac75a3c1a35aebe4d31cf6e6b68f0fefebcd7  ./guest-root/init
d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e  ./guest-root/lib/firmware/amdnpu/1502_00/npu.dev.sbin
8e4e9f1c398abde1622c92b8980e7b5b66a99092e75bb2cb4a0d91f9b50766fd  ./guest-root/lib/modules/7.1.5-custom+/extra/amdxdna.ko
00e09056a16edbc1fc434469bd2d398560722b93c428dc03a59f145ba064b702  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/acpi/button.ko
b6d9946716b92947a8269afd16a7b5c29802d327065d0f1a948465acef246fc7  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/char/hw_random/rng-core.ko
fab71ca02239f8a5d9a04aabc27649b75f09431cea1dbdabd36676a36ec8903e  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/crypto/ccp/ccp.ko
e608a98187f3429dc0b5b359450234f8c4afa8a058bad104ca3eb0b8e6d41bfb  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/gpu/drm/drm.ko
fbd90f6aaa32850cd0f63312a476030a780d5e5cb60455a9403bbab1b2886739  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/gpu/drm/drm_kms_helper.ko
ad7641815eb179dc6d9543a28aaeba261b222e0348fdd2d0df177e6a3bf59108  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/gpu/drm/drm_panel_orientation_quirks.ko
7947e40b583e6c19967962a3045e1744122d016c71a94489fb1d321270248158  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/gpu/drm/drm_shmem_helper.ko
8ae30116ee782d47f12c9b1f5962dcadb68df738c97fed6e425b00681aaaef04  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/gpu/drm/scheduler/gpu-sched.ko
a8dfaa273cac0d3dc745e8b665fa9779f22a910bfc0681dadd6df0851306626b  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/hid/amd-sfh-hid/amd_sfh.ko
79b0f3504bc6f1dbf58266097922772e272740a69d7049c304cd8b0881c5e6b6  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/hwmon/hwmon.ko
9fdc2f8203bfa3fd5b84e449c192ea9c85a6d0a5f9d30d4bcb413c5341ca9f6c  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/i2c/i2c-core.ko
7d75c4861c5d59c15cbceeeb98f9aacbbcabdcb8fa8ea9099d6d3c5b9a174ecd  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/leds/trigger/ledtrig-backlight.ko
612c5ac73a1ad85c9b5720017d814b2c1f8173f25ba70c8e04fd3b11954abba4  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/platform/x86/amd/pmf/amd-pmf.ko
5c4cd26e146dfacda829b91cc8bb1707aef980e15d6e80b2d7ad33f3e757c485  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/tee/amdtee/amdtee.ko
3dcbde84b7ee5802007143bcdb508a073247c672fec15449abfd4420b896c17b  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/tee/tee.ko
83d11920e4c6c286ff522a50840dcb93bd9a26ceaf07646ddd92acd791d7ee8d  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/tty/serial/8250/8250.ko
08584eff55965a5b56b28bd29b18ed57524cac1b637b60a1bbbb292f7fac6ede  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/tty/serial/8250/8250_base.ko
a0639591d1b88db161a33571cbbbd3df23fda3e5189df7c1f86a63318fb701bf  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/tty/serial/serial_base.ko
9a69e6666903d8195d6fd48f61b4bc5e1cca767dbafcfbac0fd15d794958de56  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/tty/serial/serial_mctrl_gpio.ko
14334ee2247e6d42783090fe234bbf8070b62b689fdaae70300c1bb5bda342e2  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/video/backlight/backlight.ko
109fed03f94366f50bfee6ab83ea8027dacd2ee5c221c549378d42de901cebf6  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/video/backlight/lcd.ko
56ae812cc027f82c76c1358574c3830f660cc29f2d427d55ad15dbf0dc6ccf78  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/video/fbdev/core/fb.ko
020e4301b5497f4481b15dd433b831ee1f70d6683d61b3fedc85a70921de65dc  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/video/fbdev/core/fb_sys_fops.ko
771a0cc156cf2aa2401bc70ab5429c746d50da830f00fc5880e2fc03a1597de9  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/video/fbdev/core/syscopyarea.ko
4a3dba4b475df7653ded8efcf97b5eafa111f2ba1b8e7a22f04bba6d3b7af1c3  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/video/fbdev/core/sysfillrect.ko
555ef5bbbb312cf68504e412fcf211adaf4d3f2f426b19217700e9025da7439c  ./guest-root/lib/modules/7.1.5-custom+/kernel/drivers/video/fbdev/core/sysimgblt.ko
8763c969edd902dc1a2ee74322505f4f41324dbe1512e95f289b8dd2c3c665ad  ./guest-root/lib/modules/7.1.5-custom+/kernel/lib/fonts/font.ko
50c78e17a7705092e8c080ffcbd649d2a7a31b44ccba4289ec2d780f75489df4  ./guest-root/lib/modules/7.1.5-custom+/modules.alias
2950f49a675cde1e02d50925103d52dbc55b376e4b24bd30340cc2bd7c39b1cd  ./guest-root/lib/modules/7.1.5-custom+/modules.alias.bin
4ce8ef507ad83dcadb031b555a76d957fe0ea142a3049523bb0f8b8a29b755c7  ./guest-root/lib/modules/7.1.5-custom+/modules.builtin
078ef4f733f5239890a1f8e12354ddaf35709fce167c4bfcb7f430079ed5c08d  ./guest-root/lib/modules/7.1.5-custom+/modules.builtin.alias.bin
b4198c84c4dace7bd113139ada249e06c93fa3f2bc596af0177e15822fe100ae  ./guest-root/lib/modules/7.1.5-custom+/modules.builtin.bin
b387f120f062950f7a284937279bf094ae98be18637675c3e26b610fd1841585  ./guest-root/lib/modules/7.1.5-custom+/modules.builtin.modinfo
b97c7b3fea526312def7304cc38edabba8f18f73a433be34682cbef09ac0efd3  ./guest-root/lib/modules/7.1.5-custom+/modules.dep
bf1787b3f9d2b6236ac52f041bd7df6c376736d5d1b9784a267f069439caedad  ./guest-root/lib/modules/7.1.5-custom+/modules.dep.bin
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  ./guest-root/lib/modules/7.1.5-custom+/modules.devname
92abe7946044c288e7286d367bc83463d7eacca9535b7863f2048246a168753c  ./guest-root/lib/modules/7.1.5-custom+/modules.order
332fbb784b84d0896f3871b254f5ebdddc63b82b0cee4d284b281e5df8bdff03  ./guest-root/lib/modules/7.1.5-custom+/modules.softdep
85f4c186024c14de217ae1b2fa137a4cc737317d9e4851d4028734e2d2f78e82  ./guest-root/lib/modules/7.1.5-custom+/modules.symbols
8695111d4609dbb77f56ee7b9748fa1ab1b5ead57e7e3f25595f5d29df141763  ./guest-root/lib/modules/7.1.5-custom+/modules.symbols.bin
a1fffe1059d8150b5d402b3f284f507025a8d4b5881810cb17b3fda8b8ab9304  ./guest-root/lib/modules/7.1.5-custom+/modules.weakdep
c5e80a563850d6ab5c2f2482e4202d9c1b71fbf44854b8c399e63527202c64e1  ./guest-root/lib64/ld-linux-x86-64.so.2
69d585730b671dfbe6c48fa7000e398803880fac4ce204c9c274e50d47017fdd  ./guest-root/opt/xilinx/xrt/lib/libxrt_core.so.2
461d3a9de0db09080ea1ad6e66476f012f983bc186772f14730d7eb03c356e76  ./guest-root/opt/xilinx/xrt/lib/libxrt_coreutil.so.2
4d6ed092a3ed805edd93053561b02946daa1187c3135a39674630b604455fd91  ./guest-root/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2
c46198460a07ff2aa03a12b125851a223eeb1e8c315132d60aec18d831453bf6  ./guest-root/run-frozen/aie.xclbin
29087f76686c3be32d18f8028a1a416beab95c5ead19f03dfcc2835c44eecacb  ./guest-root/run-frozen/compiler
ecf28f8639a58d53cc061ad8369e22218bf6baa1aad14866a4c16db5d9442fe3  ./guest-root/run-frozen/execution-mode
ee49b0a66c53d3952604460fe83fab879f38f1dad6cb70a994fc4422aa285896  ./guest-root/run-frozen/insts.bin
511d40e38eecf70def29322b5af8ce261bb79dfb793dc0ca45abc8a8f99b8806  ./guest-root/run-frozen/test.exe
449c816419809ec5e14faa816ee9bbc56ee67289fdd7e5c7552f59c1581d91a8  ./guest-root/usr/bin/lspci
a3947513a02831ec692ebf13053c07614882ab54a2101fb91a1b15724062ed0c  ./guest-root/usr/lib/x86_64-linux-gnu/libc.so.6
5b584c0b69159adb7ea0311fce14ddac7b54d57c3782bb41e57640777477d7ad  ./guest-root/usr/lib/x86_64-linux-gnu/libcrypto.so.3
9d339ecb409578d6a5d587e6c537a8f9589b8a13fefba30d167433a4b5758bee  ./guest-root/usr/lib/x86_64-linux-gnu/libgcc_s.so.1
29549e4793a46445bfc98269be8af8e024279b678af61632c74bcc59cb721ef1  ./guest-root/usr/lib/x86_64-linux-gnu/libkmod.so.2
beea4eeacfcfa2cd96011b959a826c97cf4a774017e214f6a34d7eea3d49cd88  ./guest-root/usr/lib/x86_64-linux-gnu/libm.so.6
7639634bc59ffa807cd9c181998c73ba64375beba607827776f03cc32d2d16c1  ./guest-root/usr/lib/x86_64-linux-gnu/libpci.so.3
5bb0d21308f123b6ad46c6f35b42cedfcb8d6d439a53aa3dae04d880aaffdde3  ./guest-root/usr/lib/x86_64-linux-gnu/libstdc++.so.6
b45c00bcc6d89c3e8fcf59dbe1777cfdba3045f28bb3a9bb94286d126dee13d7  ./guest-root/usr/lib/x86_64-linux-gnu/libudev.so.1
64adb2ed9f0f65ab40f58c61aeef43c90c11ce0358b6e3d3a95e44c57e132153  ./guest-root/usr/lib/x86_64-linux-gnu/libuuid.so.1
fbf56b0e59287033b6579bbbeae2f9de2fe86ad5bf2bd44d44aad67a15109318  ./guest-root/usr/lib/x86_64-linux-gnu/libz.so.1
060eeb79531d435306665fd78329d8a9dd579f95876c5960bb67937825362a80  ./guest-root/usr/lib/x86_64-linux-gnu/libzstd.so.1
d57b3524531fcc31701042f09f6dba44f85c951c7cbec5ba74a8724f85b59fd3  ./guest-root/usr/share/misc/pci.ids
c43c7cc1f2c847d13f406055801e0e019a9e71a1e684879be9f9380d87d5669a  ./guest.log
4bf11bd189efd4770954acee795f78fc1228225d46e04706d39c766fcf79a6d3  ./initramfs.cpio
2b9503c506d8b11533822d096bff106572afb991956ad2dd993e5f054ef095ec  ./initramfs.cpio.gz
d5703afcfeb9cce82f84e530d63a94f491db919e416bab0ecf0b9d314c052597  ./lspci.log
7c028d595efd89eb92e44d83b9f3c9d30b84e472043304059ecda81e675723ce  ./module-paths.txt
2d884069d3b99c4e16a3ee58e22299a3a48ecbf4ff59f4f8fe30add8e838b4d6  ./msix.log
0272d169a1e6d54db6a926f178e605fb3e8966e9cffe1bba39c3badd6ef1a097  ./qemu-command.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  ./qemu.log
6290a91e60d55e8245561d2c5820f485a2e8cbb96e7b90c23c64aa856247ddf7  ./server.log
bc6ffdb57daff4c9be5e11fab3bf89dbf53cd2b59d2ea96882a04743c1deee95  ./tuple.txt
```
<!-- CHECKSUMS-END -->
