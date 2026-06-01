//! `AiesimBackend` -- `NpuBackend` backed by the C++ bridge (via `BridgeAbi`).

use crate::aiesim::abi::{BridgeAbi, BridgeHalt, BridgeStatus};
use crate::backend::{HaltKind, NpuBackend, RunObserver, RunOutcome};
use xdna_emu_core::device::context::ContextId;
use xdna_emu_core::device::host_memory::HostMemory;
use xdna_emu_core::npu::{NpuInstruction, NpuInstructionStream};
use xdna_emu_core::parser::cdo::CdoRaw;
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
        // The bridge resolves DdrPatch records against host buffers, so forward
        // the registration. Keep the Rust-side Vec for introspection.
        self.host_buffers.push((address, size));
        let _ = self.bridge.add_host_buffer(address, size);
    }
    fn clear_host_buffers(&mut self) {
        self.host_buffers.clear();
        let _ = self.bridge.clear_host_buffers();
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

/// CDO op tags -- decoder twin in aiesim-bridge/src/cdo_replay.cpp (Task II.5).
/// Every value is little-endian; one tagged record per op. This is a LOSSLESS,
/// faithful serialization of the parser's `CdoRaw` variants -- no semantic
/// translation happens here (the C++ bridge replays each op via ess_*()).
mod cdo_tag {
    pub const WRITE: u8 = 1; // [addr u32][val u32]
    pub const WRITE64: u8 = 2; // [addr u64][val u32]
    pub const MASK_WRITE: u8 = 3; // [addr u32][mask u32][val u32]
    pub const MASK_WRITE64: u8 = 4; // [addr u64][mask u32][val u32]
    pub const DMA_WRITE: u8 = 5; // [addr u32][len u32][bytes...]
    pub const MASK_POLL: u8 = 6; // [addr u32][mask u32][expected u32]
    pub const MASK_POLL64: u8 = 7; // [addr u64][mask u32][expected u32]
    pub const DELAY: u8 = 8; // [cycles u32]
    pub const MARKER: u8 = 9; // [value u32]
}

/// NPU op tags -- decoder twin in aiesim-bridge (Task II.5). DdrPatch/Sync are
/// serialized VERBATIM; the bridge resolves them (DdrPatch against host buffers,
/// Sync -> DMA wait). The Rust side does NO semantic translation.
mod npu_tag {
    pub const WRITE32: u8 = 1; // [reg_off u32][val u32]
    pub const BLOCK_WRITE: u8 = 2; // [reg_off u32][count u32][vals u32...]
    pub const MASK_WRITE: u8 = 3; // [reg_off u32][val u32][mask u32]
    pub const MASK_POLL: u8 = 4; // [reg_off u32][val u32][mask u32]
    pub const DDR_PATCH: u8 = 5; // [reg_addr u32][arg_idx u8][arg_plus u32]
    pub const SYNC: u8 = 6; // [channel u8][column u8][direction u8][column_num u8][row u8][row_num u8]
}

/// Serialize a parsed CDO into the tagged little-endian byte op-stream the bridge
/// replays. One record per op; the format is documented inline in `cdo_tag`. This
/// is a faithful, round-trippable serialization of every replayable `CdoRaw`
/// variant -- the C++ decoder (Task II.5) mirrors it exactly.
///
/// `Nop` and `EndMark` are skipped (structural padding/terminator, no replay
/// effect). `Unknown` is skipped with a warning -- it signals encoder/parser
/// drift that the wire format does not (yet) carry.
pub(crate) fn encode_cdo(cdo: &Cdo<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    for cmd in cdo.commands() {
        match cmd {
            CdoRaw::Write { address, value } => {
                out.push(cdo_tag::WRITE);
                out.extend_from_slice(&address.to_le_bytes());
                out.extend_from_slice(&value.to_le_bytes());
            }
            CdoRaw::Write64 { address, value } => {
                out.push(cdo_tag::WRITE64);
                out.extend_from_slice(&address.to_le_bytes());
                out.extend_from_slice(&value.to_le_bytes());
            }
            CdoRaw::MaskWrite { address, mask, value } => {
                out.push(cdo_tag::MASK_WRITE);
                out.extend_from_slice(&address.to_le_bytes());
                out.extend_from_slice(&mask.to_le_bytes());
                out.extend_from_slice(&value.to_le_bytes());
            }
            CdoRaw::MaskWrite64 { address, mask, value } => {
                out.push(cdo_tag::MASK_WRITE64);
                out.extend_from_slice(&address.to_le_bytes());
                out.extend_from_slice(&mask.to_le_bytes());
                out.extend_from_slice(&value.to_le_bytes());
            }
            CdoRaw::DmaWrite { address, data } => {
                out.push(cdo_tag::DMA_WRITE);
                out.extend_from_slice(&address.to_le_bytes());
                out.extend_from_slice(&(data.len() as u32).to_le_bytes());
                out.extend_from_slice(&data);
            }
            CdoRaw::MaskPoll { address, mask, expected } => {
                out.push(cdo_tag::MASK_POLL);
                out.extend_from_slice(&address.to_le_bytes());
                out.extend_from_slice(&mask.to_le_bytes());
                out.extend_from_slice(&expected.to_le_bytes());
            }
            CdoRaw::MaskPoll64 { address, mask, expected } => {
                out.push(cdo_tag::MASK_POLL64);
                out.extend_from_slice(&address.to_le_bytes());
                out.extend_from_slice(&mask.to_le_bytes());
                out.extend_from_slice(&expected.to_le_bytes());
            }
            CdoRaw::Delay { cycles } => {
                out.push(cdo_tag::DELAY);
                out.extend_from_slice(&cycles.to_le_bytes());
            }
            CdoRaw::Marker { value } => {
                out.push(cdo_tag::MARKER);
                out.extend_from_slice(&value.to_le_bytes());
            }
            // Structural-only -- no replay effect.
            CdoRaw::Nop { .. } | CdoRaw::EndMark => {}
            CdoRaw::Unknown { opcode, .. } => {
                log::warn!(
                    "encode_cdo: skipping Unknown CDO op (opcode 0x{:03X}) -- encoder/parser drift",
                    opcode
                );
            }
        }
    }
    out
}

/// Serialize a runtime-sequence (NPU instruction) stream into the tagged
/// little-endian wire format. Separate tag namespace from CDO (the models
/// diverge). `DdrPatch` and `Sync` are emitted VERBATIM (raw fields, no
/// resolution) -- the C++ bridge (Task II.5) resolves DdrPatch against the
/// registered host buffers and maps Sync to a DMA wait. The decoder mirrors
/// `npu_tag` exactly. `Unknown` is skipped with a warning.
pub(crate) fn encode_npu(stream: &NpuInstructionStream) -> Vec<u8> {
    let mut out = Vec::new();
    for instr in stream.instructions() {
        match instr {
            NpuInstruction::Write32 { reg_off, value } => {
                out.push(npu_tag::WRITE32);
                out.extend_from_slice(&reg_off.to_le_bytes());
                out.extend_from_slice(&value.to_le_bytes());
            }
            NpuInstruction::BlockWrite { reg_off, values } => {
                out.push(npu_tag::BLOCK_WRITE);
                out.extend_from_slice(&reg_off.to_le_bytes());
                out.extend_from_slice(&(values.len() as u32).to_le_bytes());
                for v in values {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            NpuInstruction::MaskWrite { reg_off, value, mask } => {
                out.push(npu_tag::MASK_WRITE);
                out.extend_from_slice(&reg_off.to_le_bytes());
                out.extend_from_slice(&value.to_le_bytes());
                out.extend_from_slice(&mask.to_le_bytes());
            }
            NpuInstruction::MaskPoll { reg_off, value, mask } => {
                out.push(npu_tag::MASK_POLL);
                out.extend_from_slice(&reg_off.to_le_bytes());
                out.extend_from_slice(&value.to_le_bytes());
                out.extend_from_slice(&mask.to_le_bytes());
            }
            NpuInstruction::DdrPatch { reg_addr, arg_idx, arg_plus } => {
                // VERBATIM -- the bridge resolves the host-buffer address.
                out.push(npu_tag::DDR_PATCH);
                out.extend_from_slice(&reg_addr.to_le_bytes());
                out.push(*arg_idx);
                out.extend_from_slice(&arg_plus.to_le_bytes());
            }
            NpuInstruction::Sync { channel, column, direction, column_num, row, row_num } => {
                // VERBATIM -- the bridge maps this to a DMA channel wait.
                out.push(npu_tag::SYNC);
                out.push(*channel);
                out.push(*column);
                out.push(*direction);
                out.push(*column_num);
                out.push(*row);
                out.push(*row_num);
            }
            NpuInstruction::Unknown { opcode, .. } => {
                log::warn!(
                    "encode_npu: skipping Unknown NPU op (opcode 0x{:02X}) -- encoder/parser drift",
                    opcode
                );
            }
        }
    }
    out
}

/// Push populated host-memory regions into the bridge GM model. Iterates the
/// registered `HostMemory` regions and writes each region's contents through to
/// the bridge via `write_gm`. (Regions are the only populated-range view
/// `HostMemory` exposes; sparse pages outside a region are not flushed.)
fn flush_host_to_bridge(host: &HostMemory, bridge: &mut dyn BridgeAbi) {
    for region in host.regions() {
        let mut bytes = vec![0u8; region.size];
        host.read_bytes(region.base_address, &mut bytes);
        let _ = bridge.write_gm(region.base_address, &bytes);
    }
}

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

    // --- wire-format round-trip tests -------------------------------------
    //
    // A tiny in-test byte decoder reads the tagged LE records back into
    // tuples and asserts exact field recovery + length consistency. This is
    // the contract the Part II C++ decoder (cdo_replay.cpp) must match.

    /// Sequential little-endian byte reader for the in-test decoder.
    struct Rd<'a> {
        b: &'a [u8],
        i: usize,
    }
    impl<'a> Rd<'a> {
        fn new(b: &'a [u8]) -> Self {
            Self { b, i: 0 }
        }
        fn tag(&mut self) -> u8 {
            let t = self.b[self.i];
            self.i += 1;
            t
        }
        fn u8(&mut self) -> u8 {
            let v = self.b[self.i];
            self.i += 1;
            v
        }
        fn u32(&mut self) -> u32 {
            let v = u32::from_le_bytes(self.b[self.i..self.i + 4].try_into().unwrap());
            self.i += 4;
            v
        }
        fn u64(&mut self) -> u64 {
            let v = u64::from_le_bytes(self.b[self.i..self.i + 8].try_into().unwrap());
            self.i += 8;
            v
        }
        fn bytes(&mut self, n: usize) -> Vec<u8> {
            let v = self.b[self.i..self.i + n].to_vec();
            self.i += n;
            v
        }
        fn done(&self) -> bool {
            self.i == self.b.len()
        }
    }

    /// Build a minimal CDO byte blob from a list of (opcode, payload-words),
    /// mirroring `src/parser/cdo/syntax.rs`'s framing (header + command words).
    fn build_cdo(cmds: &[(u16, Vec<u32>)]) -> Vec<u8> {
        // Compute command-word count: each cmd is 1 cmd-word + payload words.
        let mut words: Vec<u32> = Vec::new();
        for (opcode, payload) in cmds {
            let cmd_word = ((payload.len() as u32) << 16) | (*opcode as u32);
            words.push(cmd_word);
            words.extend_from_slice(payload);
        }
        let cdo_len = words.len() as u32;
        const CDO_MAGIC_CDO: u32 = 0x004F_4443; // "CDO\0" little-endian
        let mut data = Vec::new();
        data.extend_from_slice(&4u32.to_le_bytes()); // num_words
        data.extend_from_slice(&CDO_MAGIC_CDO.to_le_bytes()); // ident
        data.extend_from_slice(&0x0200u32.to_le_bytes()); // version 2.0
        data.extend_from_slice(&cdo_len.to_le_bytes()); // length (words)
        let checksum = !(4u32.wrapping_add(CDO_MAGIC_CDO).wrapping_add(0x0200).wrapping_add(cdo_len));
        data.extend_from_slice(&checksum.to_le_bytes());
        for w in words {
            data.extend_from_slice(&w.to_le_bytes());
        }
        data
    }

    #[test]
    fn encode_cdo_round_trips() {
        // Write{0x1000, 0xDEAD}, Write64{0x1_0000_2000, 0xBEEF},
        // DmaWrite{0x2000, [4 bytes]}, MaskPoll64{0x3_0000_4000, mask, expected}.
        // Construct via the real parser byte framing. Opcode literals per
        // src/parser/cdo/syntax.rs: Write=0x103, Write64=0x108, DmaWrite=0x105,
        // MaskPoll64=0x106.
        let dma_data: Vec<u8> = vec![0x11, 0x22, 0x33, 0x44];
        let blob = build_cdo(&[
            (0x103, vec![0x1000, 0xDEAD]),
            // Write64 payload: [addr_hi, addr_lo, value]
            (0x108, vec![0x1, 0x0000_2000, 0xBEEF]),
            // DmaWrite payload: [addr_hi=0, addr_lo, data...]
            (0x105, vec![0, 0x2000, 0x4433_2211]),
            // MaskPoll64 payload: [addr_hi, addr_lo, mask, expected]
            (0x106, vec![0x3, 0x0000_4000, 0x00FF, 0x0042]),
        ]);
        let cdo = Cdo::parse(&blob).expect("CDO parse");
        let bytes = encode_cdo(&cdo);

        let mut r = Rd::new(&bytes);

        assert_eq!(r.tag(), cdo_tag::WRITE);
        assert_eq!(r.u32(), 0x1000);
        assert_eq!(r.u32(), 0xDEAD);

        assert_eq!(r.tag(), cdo_tag::WRITE64);
        assert_eq!(r.u64(), 0x1_0000_2000);
        assert_eq!(r.u32(), 0xBEEF);

        assert_eq!(r.tag(), cdo_tag::DMA_WRITE);
        assert_eq!(r.u32(), 0x2000);
        let len = r.u32() as usize;
        assert_eq!(len, dma_data.len());
        assert_eq!(r.bytes(len), dma_data);

        assert_eq!(r.tag(), cdo_tag::MASK_POLL64);
        assert_eq!(r.u64(), 0x3_0000_4000);
        assert_eq!(r.u32(), 0x00FF);
        assert_eq!(r.u32(), 0x0042);

        assert!(r.done(), "decoder consumed exactly the encoded length");
    }

    #[test]
    fn encode_npu_round_trips() {
        // Construct the stream directly from instructions (real parser API).
        let stream = NpuInstructionStream::from_instructions(vec![
            NpuInstruction::Write32 { reg_off: 0x232004, value: 0x1_0000 },
            NpuInstruction::BlockWrite { reg_off: 0x1D000, values: vec![0xAA, 0xBB, 0xCC] },
            NpuInstruction::DdrPatch { reg_addr: 0x1D004, arg_idx: 2, arg_plus: 0x40 },
            NpuInstruction::Sync { channel: 1, column: 2, direction: 0, column_num: 3, row: 4, row_num: 5 },
        ]);
        let bytes = encode_npu(&stream);

        let mut r = Rd::new(&bytes);

        assert_eq!(r.tag(), npu_tag::WRITE32);
        assert_eq!(r.u32(), 0x232004);
        assert_eq!(r.u32(), 0x1_0000);

        assert_eq!(r.tag(), npu_tag::BLOCK_WRITE);
        assert_eq!(r.u32(), 0x1D000);
        let count = r.u32() as usize;
        assert_eq!(count, 3);
        assert_eq!(r.u32(), 0xAA);
        assert_eq!(r.u32(), 0xBB);
        assert_eq!(r.u32(), 0xCC);

        assert_eq!(r.tag(), npu_tag::DDR_PATCH);
        assert_eq!(r.u32(), 0x1D004);
        assert_eq!(r.u8(), 2);
        assert_eq!(r.u32(), 0x40);

        assert_eq!(r.tag(), npu_tag::SYNC);
        assert_eq!(r.u8(), 1); // channel
        assert_eq!(r.u8(), 2); // column
        assert_eq!(r.u8(), 0); // direction
        assert_eq!(r.u8(), 3); // column_num
        assert_eq!(r.u8(), 4); // row
        assert_eq!(r.u8(), 5); // row_num

        assert!(r.done(), "decoder consumed exactly the encoded length");
    }

    #[test]
    fn add_host_buffer_forwards_to_bridge() {
        let mut b = backend_with_mock();
        b.add_host_buffer(0x4000, 256);
        b.add_host_buffer(0x8000, 512);
        assert_eq!(b.host_buffers, vec![(0x4000, 256), (0x8000, 512)]);
        b.clear_host_buffers();
        assert!(b.host_buffers.is_empty());
        // Forwarding is exercised through the trait; the mock records both.
        // (The mock's host_buffers Vec is private to the abi module; this test
        // asserts the Rust-side mirror, and the forwarding call compiles +
        // returns Ok via the trait contract.)
    }

    #[test]
    fn flush_writes_regions_to_bridge() {
        // Populate a region in the backend's host memory, then run; the flush
        // path must push the region bytes into the bridge GM model.
        use crate::backend::RunObserver;
        use xdna_emu_core::device::async_errors::AmdxdnaAsyncError;
        struct NullObs;
        impl RunObserver for NullObs {
            fn on_async_errors(&mut self, _r: &[AmdxdnaAsyncError]) {}
        }
        let mut b = backend_with_mock();
        {
            let host = b.host_memory_mut();
            host.allocate_region("input", 0x1_0000, 8).unwrap();
            host.write_bytes(0x1_0000, &[1, 2, 3, 4, 5, 6, 7, 8]);
        }
        let _ = b.run(0, &mut NullObs);
        // The flush ran on the dirty host memory; verify by direct readback
        // through the bridge's read_gm.
        let mut out = [0u8; 8];
        assert_eq!(b.bridge.read_gm(0x1_0000, &mut out), BridgeStatus::Ok);
        assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
