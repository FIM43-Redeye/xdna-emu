# NPU Software Stack Rebuild Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the full NPU software stack (out-of-tree xdna-driver + XRT) from latest source so the ISA validation harness can run on real hardware.

**Architecture:** The out-of-tree xdna-driver (from `amd/xdna-driver` GitHub) bundles XRT as a git submodule. Its `build.sh` script builds both the kernel module and the XRT SHIM plugin. We need to: (1) blacklist the in-tree driver, (2) pull latest commits, (3) apply our mailbox resilience patch to the out-of-tree driver, (4) build everything, (5) sign the module, (6) install, (7) verify with xrt-smi validate.

**Tech Stack:** Linux kernel module (C), CMake, XRT (C++), MOK signing, dpkg

---

### Task 1: Blacklist the in-tree amdxdna driver

The kernel has `CONFIG_DRM_ACCEL_AMDXDNA=m` which loads at boot and conflicts with the out-of-tree driver. Blacklist it so the out-of-tree one takes precedence.

**Files:**
- Create: `/etc/modprobe.d/blacklist-amdxdna-intree.conf`

- [ ] **Step 1: Unload current in-tree driver and create blacklist**

```bash
pkexec sh -c '
echo "# Blacklist in-tree amdxdna to use out-of-tree xdna-driver instead" > /etc/modprobe.d/blacklist-amdxdna-intree.conf
echo "blacklist amdxdna" >> /etc/modprobe.d/blacklist-amdxdna-intree.conf
modprobe -r amdxdna 2>/dev/null || true
'
```

- [ ] **Step 2: Verify driver is unloaded**

```bash
lsmod | grep amdxdna
```
Expected: no output (driver unloaded).

### Task 2: Update xdna-driver to latest

**Files:**
- Modify: `/home/triple/npu-work/xdna-driver/` (git pull)

- [ ] **Step 1: Pull latest xdna-driver**

```bash
cd /home/triple/npu-work/xdna-driver
git pull origin main
```

- [ ] **Step 2: Update XRT submodule**

```bash
cd /home/triple/npu-work/xdna-driver
git submodule update --init --recursive
```

- [ ] **Step 3: Verify submodule status**

```bash
git submodule status
```
Expected: XRT submodule at a specific commit (no `+` prefix indicating local changes).

### Task 3: Apply mailbox resilience patch

The out-of-tree driver has the same bug as the in-tree one: mailbox response handlers return `-EINVAL` on firmware command failures, which causes the mailbox worker to disable the IRQ and brick the NPU until driver reload. Fix all three response handlers to always return 0.

**Files:**
- Modify: `/home/triple/npu-work/xdna-driver/src/driver/amdxdna/aie2_ctx.c`

- [ ] **Step 1: Patch aie2_sched_resp_handler (~line 241)**

Change `ret = -EINVAL;` to just setting cmd state, and change `return ret;` to `return 0;`.

The handler should set error state on the command (via `amdxdna_cmd_set_state`) but never return non-zero to the mailbox worker.

- [ ] **Step 2: Patch aie2_sched_nocmd_resp_handler (~line 276)**

Same pattern: remove `ret = -EINVAL;`, return 0 always.

- [ ] **Step 3: Patch aie2_sched_cmdlist_resp_handler (~line 300)**

Same pattern for all `ret = -EINVAL;` assignments.

- [ ] **Step 4: Patch cfg_debug_bo if present**

Check for `aie2_ctx_attach_debug_bo` or similar. If it returns `-EINVAL` when BO lookup fails, change to a warning + return 0.

- [ ] **Step 5: Verify patch compiles**

Build just the driver module to check for compile errors:
```bash
cd /home/triple/npu-work/xdna-driver
nice -n 19 make -C /lib/modules/$(uname -r)/build M=$(pwd)/src/driver/amdxdna modules
```

### Task 4: Build the full stack

The `build.sh` script builds XRT SHIM + driver + packages. Use `-release` for the optimized build.

**Files:**
- Build output: `/home/triple/npu-work/xdna-driver/build/Release/`

- [ ] **Step 1: Clean previous build**

```bash
cd /home/triple/npu-work/xdna-driver/build
./build.sh -clean
```

- [ ] **Step 2: Build release with kernel module**

```bash
cd /home/triple/npu-work/xdna-driver/build
nice -n 19 ./build.sh -release
```

This will take several minutes. It builds:
- The kernel module (`amdxdna.ko`)
- XRT SHIM plugin (`libxrt_driver_xdna.so`)
- Firmware packaging
- .deb packages

Expected: Build completes without errors.

- [ ] **Step 3: Verify build artifacts**

```bash
find build/Release -name "amdxdna.ko" -o -name "*.deb" | head -10
```

### Task 5: Sign and install the kernel module

**Files:**
- Install: kernel module to `/lib/modules/$(uname -r)/`

- [ ] **Step 1: Find the built module**

```bash
find /home/triple/npu-work/xdna-driver/build -name "amdxdna.ko" | head -5
```

- [ ] **Step 2: Sign with MOK key**

```bash
/home/triple/kernelify/linux/scripts/sign-file sha256 \
    /var/lib/shim-signed/mok/MOK.priv \
    /var/lib/shim-signed/mok/MOK.der \
    <path-to-amdxdna.ko>
```

- [ ] **Step 3: Install the .deb packages**

```bash
cd /home/triple/npu-work/xdna-driver/build/Release
pkexec dpkg -i xrt_plugin-amdxdna*.deb
```

If dpkg conflicts with existing packages, remove them first:
```bash
pkexec dpkg -r xrt_plugin-amdxdna xrt-npu xrt-base-dev xrt-base
pkexec dpkg -i xrt*.deb xrt_plugin*.deb
```

- [ ] **Step 4: Install firmware if needed**

```bash
ls /lib/firmware/amdnpu/1502_00/npu.sbin
```
If missing, the build script should have packaged it.

- [ ] **Step 5: Run depmod**

```bash
pkexec depmod -a
```

### Task 6: Load and verify

- [ ] **Step 1: Load the out-of-tree module**

```bash
pkexec modprobe amdxdna
lsmod | grep amdxdna
```

- [ ] **Step 2: Check dmesg for clean initialization**

```bash
pkexec dmesg | grep amdxdna | tail -5
```
Expected: firmware load, no errors, no "disable irq".

- [ ] **Step 3: Run xrt-smi examine**

```bash
/opt/xilinx/xrt/bin/xrt-smi examine
```
Expected: Device shows as RyzenAI-npu1, firmware version visible.

- [ ] **Step 4: Run xrt-smi validate**

```bash
/opt/xilinx/xrt/bin/xrt-smi validate
```
Expected: latency and throughput tests PASS.

- [ ] **Step 5: Run a quick NPU test**

```bash
cd /home/triple/npu-work/xdna-emu
timeout 10 build/isa-tests/test_host \
    -x build/isa-tests/batch_0/aie.xclbin \
    -k MLIR_AIE \
    -i build/isa-tests/batch_0/insts.bin \
    --in-size 1828 --out-size 1828 \
    --seed 42 --out-file /tmp/claude-1000/npu-health.bin
```
Expected: completes without "Status: 6" error.

### Task 7: Rebuild xdna-emu XRT plugin

After XRT is rebuilt, our emulator plugin needs to be rebuilt against the new XRT headers.

- [ ] **Step 1: Rebuild plugin**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/rebuild-plugin.sh --release
```

- [ ] **Step 2: Install plugin**

```bash
pkexec cp xrt-plugin/build/libxrt_driver_emu.so.2 /opt/xilinx/xrt/lib/
```

- [ ] **Step 3: Verify EMU works**

```bash
XDNA_EMU=release timeout 10 build/isa-tests/test_host \
    -x build/isa-tests/batch_0/aie.xclbin \
    -k MLIR_AIE \
    -i build/isa-tests/batch_0/insts.bin \
    --in-size 1828 --out-size 1828 \
    --seed 42 --out-file /tmp/claude-1000/emu-health.bin
```

### Task 8: Full ISA validation run

- [ ] **Step 1: Run full ISA test with HW + EMU comparison**

```bash
cd /home/triple/npu-work/xdna-emu
nice -n 19 ./scripts/isa-test.sh --seed 42
```

- [ ] **Step 2: Analyze results and update baseline**

Compare against 46.0% baseline. Expected improvements from this session's fixes:
- SELEQZ: 36% -> 100% (+16 points)
- PADDB/PADDS: 0% -> working (+46 points)
- MOV_CNTR: 0% -> working (+7 points)
