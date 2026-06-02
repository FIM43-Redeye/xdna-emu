//! The bridge ABI: a Rust trait the backend calls, plus the raw C-ABI extern
//! declarations the real impl binds to. Keeping the backend behind a trait lets
//! us unit-test all marshalling against an in-memory `MockBridge` -- no aietools.

/// Result codes the bridge C ABI returns (0 = ok).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub(crate) enum BridgeStatus {
    Ok = 0,
    Error = 1,
}

/// How a bridge `run` ended (mirrors the C ABI's enum; maps to HaltKind).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub(crate) enum BridgeHalt {
    Completed = 0,
    Budget = 1,
    Error = 2,
}

/// The operations `AiesimBackend` needs from the bridge. One method per C-ABI
/// entry. Slices/lengths are marshalled to raw pointers only in the real impl.
pub(crate) trait BridgeAbi {
    // `create`, `read_gm`, and `read_reg` are part of the bridge contract but
    // not yet called from compiled (non-test) code: `create` runs inside
    // `DlopenBridge::open`, and `read_gm`/`read_reg` are the tier-2 read-back
    // paths wired in Part II. Allow them now so the feature build is clean.
    /// Construct the cluster for `arch` ("aie2"/"aie2ps"/"aie") using the device
    /// JSON at `device_json`. Called once; the bridge owns the service thread.
    #[allow(dead_code)]
    fn create(&mut self, arch: &str, device_json: &str) -> BridgeStatus;
    /// Replay a config op-stream (already serialized by cdo_replay-side encoding;
    /// here it is the raw bytes our parser produced). Returns Ok/Error.
    fn load_cdo(&mut self, ops: &[u8]) -> BridgeStatus;
    /// Replay a runtime-sequence (NPU instruction) op-stream as register writes.
    /// Same tagged wire format as `load_cdo`; separate entry so the bridge can
    /// stage it distinctly (it arrives via `execute_npu_instructions`, before run).
    fn exec_npu(&mut self, ops: &[u8]) -> BridgeStatus;
    /// Register a host buffer (DDR addr + size) so the bridge can resolve
    /// DdrPatch records during exec_npu replay.
    fn add_host_buffer(&mut self, addr: u64, size: usize) -> BridgeStatus;
    /// Clear the registered host buffers (before a new submission).
    fn clear_host_buffers(&mut self) -> BridgeStatus;
    /// Write host (DDR/GM) memory.
    fn write_gm(&mut self, addr: u64, data: &[u8]) -> BridgeStatus;
    /// Read host (DDR/GM) memory into `out`.
    #[allow(dead_code)]
    fn read_gm(&mut self, addr: u64, out: &mut [u8]) -> BridgeStatus;
    /// Run to quiescence or `budget` cycles. On Ok, `*cycles_out` is set.
    fn run(&mut self, budget: u64, cycles_out: &mut u64) -> BridgeHalt;
    /// Tier-2: zero-time backdoor register read.
    #[allow(dead_code)]
    fn read_reg(&mut self, addr: u64) -> u32;
    /// Reset logical state between submissions (re-apply CDO follows).
    fn reset(&mut self) -> BridgeStatus;
    /// Set the partition's physical start column for the NPU1->Versal address
    /// translation (logical col 0 -> physical start_col).
    fn set_start_col(&mut self, start_col: u8) -> BridgeStatus;
}

#[cfg(test)]
pub(crate) mod mock {
    use super::*;
    use std::collections::HashMap;

    /// In-memory bridge: records calls + models GM as a sparse map so the
    /// backend's write_gm/read_gm round-trips are testable without a cluster.
    #[derive(Default)]
    pub(crate) struct MockBridge {
        pub created: Option<(String, String)>,
        pub cdo_loads: u32,
        pub npu_loads: u32,
        pub runs: u32,
        pub gm: HashMap<u64, u8>,
        pub host_buffers: Vec<(u64, usize)>,
        pub start_col: u8,
        pub next_run_halt: Option<BridgeHalt>,
        pub next_run_cycles: u64,
    }

    impl BridgeAbi for MockBridge {
        fn create(&mut self, arch: &str, device_json: &str) -> BridgeStatus {
            self.created = Some((arch.to_string(), device_json.to_string()));
            BridgeStatus::Ok
        }
        fn load_cdo(&mut self, _ops: &[u8]) -> BridgeStatus {
            self.cdo_loads += 1;
            BridgeStatus::Ok
        }
        fn exec_npu(&mut self, _ops: &[u8]) -> BridgeStatus {
            self.npu_loads += 1;
            BridgeStatus::Ok
        }
        fn add_host_buffer(&mut self, addr: u64, size: usize) -> BridgeStatus {
            self.host_buffers.push((addr, size));
            BridgeStatus::Ok
        }
        fn clear_host_buffers(&mut self) -> BridgeStatus {
            self.host_buffers.clear();
            BridgeStatus::Ok
        }
        fn write_gm(&mut self, addr: u64, data: &[u8]) -> BridgeStatus {
            for (i, b) in data.iter().enumerate() {
                self.gm.insert(addr + i as u64, *b);
            }
            BridgeStatus::Ok
        }
        fn read_gm(&mut self, addr: u64, out: &mut [u8]) -> BridgeStatus {
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = *self.gm.get(&(addr + i as u64)).unwrap_or(&0);
            }
            BridgeStatus::Ok
        }
        fn run(&mut self, _budget: u64, cycles_out: &mut u64) -> BridgeHalt {
            self.runs += 1;
            *cycles_out = self.next_run_cycles;
            self.next_run_halt.unwrap_or(BridgeHalt::Completed)
        }
        fn read_reg(&mut self, _addr: u64) -> u32 {
            0
        }
        fn reset(&mut self) -> BridgeStatus {
            BridgeStatus::Ok
        }
        fn set_start_col(&mut self, start_col: u8) -> BridgeStatus {
            self.start_col = start_col;
            BridgeStatus::Ok
        }
    }
}

#[cfg(test)]
mod tests {
    use super::mock::MockBridge;
    use super::*;

    #[test]
    fn mock_bridge_round_trips_gm() {
        let mut b = MockBridge::default();
        assert_eq!(b.create("aie2", "/dev/null"), BridgeStatus::Ok);
        assert_eq!(b.write_gm(0x1000, &[1, 2, 3, 4]), BridgeStatus::Ok);
        let mut out = [0u8; 4];
        assert_eq!(b.read_gm(0x1000, &mut out), BridgeStatus::Ok);
        assert_eq!(out, [1, 2, 3, 4]);
    }
}
