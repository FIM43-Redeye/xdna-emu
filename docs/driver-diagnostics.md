# xdna-driver Diagnostic Module (aie2_diag) -- Notes for Emulator

## What It Is

The xdna-driver now has an `aie2_diag` module that exposes NPU diagnostic
data via debugfs. It lives entirely in kernel space and requires no
userspace tool changes. Three read-only debugfs files are created under
`/sys/kernel/debug/dri/<N>/`:

| File | Contents |
|------|----------|
| `diag_stats` | Cumulative TDR counters: total events, soft recoveries, SBR resets, unrecoverable failures |
| `diag_tdr_history` | Ring buffer (16 entries) of TDR events, newest-first. Each entry has timestamp, escalation level, and per-context snapshots (hwctx_id, submitted/completed counts, ctx_pc, fatal_type) |
| `diag_cert_state` | Live CERT firmware state for each active context, read via MSG_OP_AIE_RW_ACCESS register reads |

## How to Use It (Real Hardware)

No special tools needed. Just `cat` the files:

```bash
# Find the DRI node
ls /sys/kernel/debug/dri/

# Check stats (all zeros = healthy, no TDRs have fired)
cat /sys/kernel/debug/dri/0/diag_stats

# After a hang, check what happened
cat /sys/kernel/debug/dri/0/diag_tdr_history

# Inspect live CERT firmware state while workloads run
cat /sys/kernel/debug/dri/0/diag_cert_state
```

## What the Emulator Does NOT Need to Do

The emulator requires zero changes to benefit from this module.
The diagnostic data flows from real hardware:

```
NPU hardware --> CERT firmware --> MSG_OP_AIE_RW_ACCESS --> driver --> debugfs
```

The emulator sits outside this path entirely -- it simulates tile array
behavior at the instruction level, not firmware protocol messages.

## What the Data Means (Reference for Future Emulator Work)

### CERT Firmware States

The CERT microcontroller on each column runs a state machine. These are
the states exposed by `diag_cert_state`:

| Code | Name | Meaning |
|------|------|---------|
| 0x0 | UNINIT | CERT not yet initialized |
| 0x1 | INIT | CERT initialized, no context loaded |
| 0x2 | CTX_IDLE | Context loaded but not executing (check idle_status for reason) |
| 0x3 | CTX_RUNNING | Context actively executing |
| 0x4 | CTX_PREEMPT_SAVE | Context being preempted (saving state) |
| 0x5 | CTX_PREEMPT_RESTORE | Context being restored after preemption |
| 0x6 | CTX_ERROR | Context hit a fatal error (check misc_status for reason) |
| 0x7 | CTX_DESTROY | Context being torn down |
| 0x8 | CTX_CREATE | Context being created |
| 0x9 | DEBUG | Debug mode active |

### CERT Handshake Registers

Read from the shim tile (row 0) pDM region. Key offsets:

| Offset | Name | Purpose |
|--------|------|---------|
| 0x00 | ALIVE | Heartbeat / firmware alive indicator |
| 0x1C | IDLE_STATUS | Why CERT is idle (bit 0: HSA queue not empty, bit 1: preempt save done, bit 2: truly idle) |
| 0x20 | MISC_STATUS | Why CERT is stuck (bit 0: uC exception, bit 1: control code hang) |
| 0xA0 | FW_STATE | Current CERT state code (see table above) |
| 0xAC | EAR | Exception Address Register (on uC crash) |
| 0xB0 | ESR | Exception Status Register (on uC crash) |
| 0xB4 | EPC | Exception Program Counter (on uC crash) |
| 0xC0 | JOBS_LAUNCHED | Cumulative jobs launched on this column |
| 0xC4 | JOBS_FINISHED | Cumulative jobs finished on this column |

### TDR Escalation Levels

The driver uses a counter that increments each time the device fails to
make progress within the 2-second timeout window:

| Counter | Action | Recovery |
|---------|--------|----------|
| 1 | Soft recovery | Stop and restart all contexts |
| 2 | Secondary Bus Reset | Full PCIe bus reset via upstream bridge |
| 3+ | Unrecoverable | TDR stopped, device is dead until power cycle |

The counter resets to 0 whenever the device resumes making progress.

## Future Emulator Opportunities

If the emulator ever wants to simulate CERT firmware behavior, the
diagnostic module provides a clear specification of what to model:

1. **State machine**: Implement the CERT state transitions listed above
   as part of the tile array model (could live in `src/device/`)

2. **Job counters**: Track JOBS_LAUNCHED / JOBS_FINISHED per column
   during emulated execution runs

3. **Error injection**: Simulate CTX_ERROR states with configurable
   EAR/ESR/EPC values to test how the driver handles failure modes

4. **TDR simulation**: The emulator already has a basic "no progress"
   detector in the test harness. Mapping that to CERT state transitions
   (CTX_RUNNING -> CTX_ERROR) would provide a more realistic model.

None of this is required -- it's future work that would make the
emulator a better test platform for driver diagnostic development.
