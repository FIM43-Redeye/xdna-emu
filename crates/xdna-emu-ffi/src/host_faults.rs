//! Reporting of shim DMA accesses the host-memory IOMMU model blocked.
//!
//! Silicon: a device access to a page the process has not mapped is blocked by
//! the IOMMU and logged (an IO_PAGE_FAULT line in the kernel log); the run
//! itself still completes. The emulator's `HostMemory` models the blocking
//! (see `HostMemory::set_iommu_guard`); this module is the log line. After
//! every engine step the run loop hands newly blocked accesses here, which
//! names the shim tile, channel and BD that made each one (a read-only look at
//! the shim DMA engines: the active transfer of a channel of the access's
//! direction whose BD base is the nearest one at or below the address), logs
//! it at error level, and at the end of the run prints one always-on summary
//! line on stderr, `XDNA_EMU_HOST_FAULT: ...`, that harnesses check the way a
//! hardware run's kernel log is checked.

use xdna_emu_core::device::dma::ChannelType;
use xdna_emu_core::device::host_memory::BlockedAccess;
use xdna_emu_core::interpreter::engine::InterpreterEngine;

/// Marker that starts the end-of-run summary line on stderr.
pub const HOST_FAULT_MARKER: &str = "XDNA_EMU_HOST_FAULT:";

/// Blocked accesses logged individually per run; the summary counts the rest.
const MAX_LOGGED: usize = 16;

/// Which shim DMA made a blocked access, as far as the engine state shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attribution {
    pub col: u8,
    pub row: u8,
    pub channel: String,
    pub bd: u8,
    pub bd_base: u64,
}

/// Per-run tracker of the host memory's blocked-access log.
#[derive(Debug, Default)]
pub struct HostFaultTracker {
    start_records: usize,
    start_bytes: u64,
    seen: usize,
    logged: usize,
    first: Option<(BlockedAccess, Option<Attribution>)>,
    start_aperture_records: usize,
    start_aperture_bytes: u64,
}

impl HostFaultTracker {
    /// Start a run: accesses already in the log belong to earlier runs.
    pub fn new(engine: &InterpreterEngine) -> Self {
        let mem = engine.host_memory();
        let records = mem.blocked_record_count();
        Self {
            start_records: records,
            start_bytes: mem.blocked_totals().0,
            seen: records,
            start_aperture_records: mem.outside_aperture_record_count(),
            start_aperture_bytes: mem.outside_aperture_bytes(),
            ..Self::default()
        }
    }

    /// Log the accesses blocked since the last call.
    pub fn observe(&mut self, engine: &InterpreterEngine) {
        let mem = engine.host_memory();
        if mem.blocked_record_count() == self.seen {
            return;
        }
        for access in mem.blocked_accesses_since(self.seen) {
            let who = attribute(engine, &access);
            if self.logged < MAX_LOGGED {
                log::error!("{}", describe(&access, who.as_ref()));
                self.logged += 1;
            }
            if self.first.is_none() {
                self.first = Some((access, who));
            }
        }
        self.seen = mem.blocked_record_count();
    }

    /// The end-of-run summary line, if this run had a blocked access.
    pub fn summary(&self, engine: &InterpreterEngine) -> Option<String> {
        let (first, who) = self.first.as_ref()?;
        let (bytes, dropped) = engine.host_memory().blocked_totals();
        Some(format!(
            "{HOST_FAULT_MARKER} the IOMMU model blocked {} bytes of shim DMA access outside every host buffer \
             in {} run(s) of addresses{}; first: {}",
            bytes - self.start_bytes,
            self.seen - self.start_records,
            if dropped > 0 { format!(" (and {dropped} runs not kept)") } else { String::new() },
            describe(first, who.as_ref())
        ))
    }
}

impl HostFaultTracker {
    /// The end-of-run line for shim DMA accesses at AIE addresses below the DDR aperture, if any.
    ///
    /// These are not IOMMU faults (they have no host address), so the line does not carry
    /// `HOST_FAULT_MARKER`; it names the first one so an unpatched BD is easy to find.
    pub fn aperture_summary(&self, engine: &InterpreterEngine) -> Option<String> {
        let mem = engine.host_memory();
        let first = mem
            .outside_aperture_accesses_since(self.start_aperture_records)
            .into_iter()
            .next()?;
        let who = attribute(engine, &first);
        let what = if first.write { "write" } else { "read" };
        let by = match who {
            Some(a) => {
                format!("shim tile ({}, {}) {} BD {} (base 0x{:x})", a.col, a.row, a.channel, a.bd, a.bd_base)
            }
            None => "a shim DMA".to_string(),
        };
        Some(format!(
            "host memory: {} bytes of shim DMA access at AIE addresses below the DDR aperture (0x{:x}) reached \
             no host memory (not an IOMMU fault); first: {by} {what} of {} bytes at 0x{:x}",
            mem.outside_aperture_bytes() - self.start_aperture_bytes,
            xdna_emu_core::device::dma::DDR_AIE_ADDR_OFFSET,
            first.len,
            first.address
        ))
    }
}

/// One line describing a blocked access and who made it.
pub fn describe(access: &BlockedAccess, who: Option<&Attribution>) -> String {
    let what = if access.write { "write" } else { "read" };
    let tail = if access.write { "dropped" } else { "returned zeros" };
    match who {
        Some(a) => format!(
            "host memory: shim tile ({}, {}) {} BD {} (base 0x{:x}) {} of {} bytes at 0x{:x} outside every \
             host buffer blocked ({tail})",
            a.col, a.row, a.channel, a.bd, a.bd_base, what, access.len, access.address
        ),
        None => format!(
            "host memory: shim DMA {} of {} bytes at 0x{:x} outside every host buffer blocked ({tail}); \
             no active shim transfer matches it",
            what, access.len, access.address
        ),
    }
}

/// The shim DMA channel whose active transfer most plausibly made `access`.
pub fn attribute(engine: &InterpreterEngine, access: &BlockedAccess) -> Option<Attribution> {
    let device = engine.device();
    let want = if access.write {
        ChannelType::S2MM
    } else {
        ChannelType::MM2S
    };
    let mut best: Option<Attribution> = None;
    for col in 0..device.array.cols() {
        let Some(dma) = device.array.dma_engine(col, 0) else {
            continue;
        };
        let s2mm = (0..dma.channel_count() as u8)
            .filter(|&c| dma.channel_type(c) == ChannelType::S2MM)
            .count();
        for ch in 0..dma.channel_count() as u8 {
            if dma.channel_type(ch) != want {
                continue;
            }
            let Some(bd) = dma.get_transfer(ch).map(|t| t.bd_index).or_else(|| dma.current_bd(ch)) else {
                continue;
            };
            let Some(cfg) = dma.get_bd(bd) else { continue };
            if cfg.base_addr > access.address || best.as_ref().is_some_and(|b| b.bd_base >= cfg.base_addr) {
                continue;
            }
            let per_dir = if want == ChannelType::S2MM {
                ch as usize
            } else {
                ch as usize - s2mm
            };
            best = Some(Attribution {
                col,
                row: 0,
                channel: format!("{}{}", if access.write { "S2MM" } else { "MM2S" }, per_dir),
                bd,
                bd_base: cfg.base_addr,
            });
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{xdna_emu_create, xdna_emu_destroy};

    /// The FFI's memory runs with the IOMMU model on and keeps allocations apart, so a device
    /// access just past a buffer is blocked (never landing in the next buffer) and the run's
    /// tracker reports it with the summary marker. End-to-end attribution to a shim channel is
    /// exercised by the probe rehearsals, which need a configured array.
    #[test]
    fn a_device_access_past_an_ffi_buffer_is_blocked_and_reported() {
        unsafe {
            let h = xdna_emu_create();
            assert!(!h.is_null());
            let handle = &mut *h;
            let buf = crate::memory::xdna_emu_alloc_buffer(h, 4096);
            let next = crate::memory::xdna_emu_alloc_buffer(h, 4096);
            assert!(buf != 0 && next != 0);
            assert_eq!(next, buf + 4096 + crate::memory::ALLOC_GUARD_GAP, "allocations are kept apart");
            crate::memory::xdna_emu_write_host_memory(h, buf + 4096, [0x55u8; 4].as_ptr(), 4);

            let engine = handle.backend.as_interpreter_mut().unwrap();
            assert!(engine.host_memory().iommu_guard(), "the FFI turns the IOMMU model on");
            let mut tracker = HostFaultTracker::new(engine);
            assert_eq!(engine.host_memory().read_u32(buf + 4096), 0, "a blocked read returns zeros");
            engine.host_memory_mut().write_u32(buf + 4096 + 4, 7);
            engine.host_memory_mut().write_u32(buf + 64, 9);
            assert_eq!(engine.host_memory().read_u32(buf + 64), 9, "inside the buffer lands");
            tracker.observe(engine);
            let line = tracker.summary(engine).expect("a blocked access gives a summary");
            assert!(line.starts_with(HOST_FAULT_MARKER), "{line}");
            assert!(line.contains("blocked 8 bytes"), "{line}");
            assert!(line.contains(&format!("read of 4 bytes at 0x{:x}", buf + 4096)), "{line}");
            assert!(tracker.summary(engine).is_some());
            let fresh = HostFaultTracker::new(engine);
            assert!(fresh.summary(engine).is_none(), "a new run starts with nothing to report");
            xdna_emu_destroy(h);
        }
    }

    /// A shim access below the DDR aperture is reported on its own line, never as a host fault.
    #[test]
    fn an_access_below_the_aperture_is_not_a_host_fault() {
        unsafe {
            let h = xdna_emu_create();
            assert!(!h.is_null());
            let handle = &mut *h;
            let engine = handle.backend.as_interpreter_mut().unwrap();
            let tracker = HostFaultTracker::new(engine);
            assert!(tracker.aperture_summary(engine).is_none());
            engine.host_memory().record_outside_aperture(0x400, 64, true);
            assert!(tracker.summary(engine).is_none(), "no IOMMU fault");
            let line = tracker.aperture_summary(engine).expect("the aperture line");
            assert!(!line.contains(HOST_FAULT_MARKER), "{line}");
            assert!(line.contains("64 bytes"), "{line}");
            assert!(line.contains("write of 64 bytes at 0x400"), "{line}");
            assert!(
                HostFaultTracker::new(engine).aperture_summary(engine).is_none(),
                "a new run starts clean"
            );
            xdna_emu_destroy(h);
        }
    }

    #[test]
    fn describe_names_the_direction_and_what_silicon_does_with_it() {
        let w = BlockedAccess { address: 0x1000, len: 32, write: true };
        let who = Attribution { col: 1, row: 0, channel: "S2MM0".into(), bd: 5, bd_base: 0x800 };
        assert_eq!(
            describe(&w, Some(&who)),
            "host memory: shim tile (1, 0) S2MM0 BD 5 (base 0x800) write of 32 bytes at 0x1000 outside every \
             host buffer blocked (dropped)"
        );
        let r = BlockedAccess { address: 0x2000, len: 4, write: false };
        assert!(describe(&r, None).contains("read of 4 bytes at 0x2000"));
        assert!(describe(&r, None).ends_with("no active shim transfer matches it"));
    }
}
