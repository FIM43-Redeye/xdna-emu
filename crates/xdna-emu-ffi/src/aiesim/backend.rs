//! `AiesimBackend` -- `NpuBackend` backed by the C++ bridge (via `BridgeAbi`).

use crate::aiesim::abi::{BridgeAbi, BridgeHalt, BridgeStatus};
use crate::backend::{HaltKind, NpuBackend, RunObserver, RunOutcome};
use xdna_emu_core::device::context::ContextId;
use xdna_emu_core::device::host_memory::HostMemory;
use xdna_emu_core::npu::NpuInstructionStream;
use xdna_emu_core::parser::Cdo;

/// Cycle budget passed to the bridge when `max_cycles == 0` (unbounded request).
const DEFAULT_CYCLE_BUDGET: u64 = 100_000_000;

pub(crate) struct AiesimBackend {
    bridge: Box<dyn BridgeAbi>,
    /// Host-memory mirror. The bridge owns the authoritative GM model; we keep
    /// a Rust-side `HostMemory` so the FFI's `host_memory_mut()`-based read/write
    /// path is unchanged, and flush writes through to the bridge on run.
    host: HostMemory,
    start_col: u8,
    cols: usize,
    rows: usize,
    arch: String,
    /// Pending host-memory writes to push to the bridge before the next run.
    /// (Populated in apply path; see Step 3 note.)
    dirty: bool,
    /// Registered host-buffer regions (addr, size). The interpreter's executor
    /// uses these for runtime-sequence address patching; the cluster uses real
    /// DDR addresses directly, so aiesim just tracks them for GM/DDR setup.
    host_buffers: Vec<(u64, usize)>,
}

impl AiesimBackend {
    /// Construct from a ready `BridgeAbi` (real or mock). `create` must already
    /// have been called on the bridge by the caller (lib.rs selector).
    pub(crate) fn new(bridge: Box<dyn BridgeAbi>, arch: String, cols: usize, rows: usize) -> Self {
        Self {
            bridge,
            host: HostMemory::default(),
            start_col: 0,
            cols,
            rows,
            arch,
            dirty: false,
            host_buffers: Vec::new(),
        }
    }
}

impl NpuBackend for AiesimBackend {
    fn apply_cdo(&mut self, cdo: &Cdo<'_>) -> Result<(), String> {
        // Parser-driven data path: serialize the CDO op-stream and hand it to
        // the bridge, which replays it as ess_*() writes. The exact serialization
        // is defined in Task I.8 (encode_cdo); for now encode-then-load.
        let ops = encode_cdo(cdo);
        match self.bridge.load_cdo(&ops) {
            BridgeStatus::Ok => Ok(()),
            BridgeStatus::Error => Err("aiesim bridge: load_cdo failed".to_string()),
        }
    }
    fn set_start_col(&mut self, start_col: u8) {
        self.start_col = start_col;
    }
    fn load_elf_bytes(&mut self, _col: usize, _row: usize, _data: &[u8]) -> Result<u32, String> {
        // ELF core images are delivered to the cluster as part of CDO/config in
        // the aiesim path; a standalone load is a no-op here. Return 0 (bytes
        // ack'd) to match the interpreter's contract. (Revisit if the bridge
        // needs an explicit core-image push -- tracked in Task II.6.)
        Ok(0)
    }
    fn host_memory_mut(&mut self) -> &mut HostMemory {
        self.dirty = true;
        &mut self.host
    }
    fn sync_cores_from_device(&mut self) {
        // No Rust-side core mirror for aiesim; the cluster is authoritative.
    }
    fn reset_for_new_context(&mut self) {
        let _ = self.bridge.reset();
        self.dirty = false;
    }
    fn reset_context(&mut self, _cid: ContextId) -> Result<(), ()> {
        match self.bridge.reset() {
            BridgeStatus::Ok => Ok(()),
            BridgeStatus::Error => Err(()),
        }
    }
    fn execute_npu_instructions(&mut self, stream: &NpuInstructionStream) -> Result<(), String> {
        // Encode the runtime-sequence ops into the same tagged wire format and
        // hand them to the bridge for register-write replay. encode_npu is a
        // placeholder until Task I.8; the bridge stages them for the next run.
        let ops = encode_npu(stream);
        match self.bridge.exec_npu(&ops) {
            BridgeStatus::Ok => Ok(()),
            BridgeStatus::Error => Err("aiesim bridge: exec_npu failed".to_string()),
        }
    }
    fn run(&mut self, max_cycles: u64, _observer: &mut dyn RunObserver) -> RunOutcome {
        // Flush any dirty host memory into the bridge GM model first.
        if self.dirty {
            flush_host_to_bridge(&self.host, self.bridge.as_mut());
            self.dirty = false;
        }
        // max_cycles == 0 means unbounded; pass the backend default to the bridge.
        let budget = if max_cycles == 0 {
            DEFAULT_CYCLE_BUDGET
        } else {
            max_cycles
        };
        let mut cycles = 0u64;
        let halt = match self.bridge.run(budget, &mut cycles) {
            BridgeHalt::Completed => HaltKind::Completed,
            BridgeHalt::Budget => HaltKind::Budget,
            BridgeHalt::Error => HaltKind::Error,
        };
        // Async-error surfacing through `observer` is a tier-3 item (Part II):
        // aiesim errors come back via error registers, not a Rust drain. For now
        // the bridge reports none; the observer is intentionally unused here.
        RunOutcome { cycles, halt }
    }
    fn add_host_buffer(&mut self, address: u64, size: usize) {
        // The cluster's shim-DMA uses real DDR addresses (no patching needed);
        // track the region so the GM/DDR model is set up before run.
        self.host_buffers.push((address, size));
    }
    fn clear_host_buffers(&mut self) {
        self.host_buffers.clear();
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn rows(&self) -> usize {
        self.rows
    }
    fn arch_name(&self) -> String {
        self.arch.clone()
    }
    // as_interpreter / as_interpreter_mut: default None -- aiesim is not an
    // interpreter. Tier-3 interpreter-only introspection is correctly absent.
}

/// Serialize a parsed CDO into the byte op-stream the bridge replays.
/// Placeholder until Task I.8 defines the real encoding; returns empty so the
/// mock-driven tests in this task compile.
pub(crate) fn encode_cdo(_cdo: &Cdo<'_>) -> Vec<u8> {
    Vec::new()
}

/// Serialize a runtime-sequence (NPU instruction) stream into the same tagged
/// wire format. Placeholder until Task I.8; returns empty for now.
pub(crate) fn encode_npu(_stream: &NpuInstructionStream) -> Vec<u8> {
    Vec::new()
}

/// Push populated host-memory regions into the bridge GM model.
/// Placeholder until Task I.8; no-op against the mock.
fn flush_host_to_bridge(_host: &HostMemory, _bridge: &mut dyn BridgeAbi) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aiesim::abi::mock::MockBridge;

    fn backend_with_mock() -> AiesimBackend {
        AiesimBackend::new(Box::new(MockBridge::default()), "aie2".to_string(), 5, 6)
    }

    #[test]
    fn topology_and_arch_reported() {
        let b = backend_with_mock();
        assert_eq!(b.cols(), 5);
        assert_eq!(b.rows(), 6);
        assert_eq!(b.arch_name(), "aie2");
        assert!(b.as_interpreter().is_none());
    }

    #[test]
    fn run_reports_outcome() {
        use crate::backend::RunObserver;
        use xdna_emu_core::device::async_errors::AmdxdnaAsyncError;
        struct NullObs;
        impl RunObserver for NullObs {
            fn on_async_errors(&mut self, _r: &[AmdxdnaAsyncError]) {}
        }
        let mut b = backend_with_mock();
        let out = b.run(0, &mut NullObs);
        assert_eq!(out.halt, HaltKind::Completed);
    }
}
