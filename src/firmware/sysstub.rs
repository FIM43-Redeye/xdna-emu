//! System-aperture stub: answers off-array SMN/NoC/system-config reads with
//! benign zeros, logs every access, and flags a tight consecutive-read spin
//! as the "firmware is waiting on unmodeled state" signal (spec section 8).

use std::collections::HashMap;

/// Default consecutive-read count past which an address is judged "spinning"
/// (spec section 8).
const SPIN_THRESHOLD: u32 = 1024;

/// Stub handler for the off-array system aperture (`Region::System`): SMN,
/// NoC, and system-config registers the firmware polls that this phase does
/// not model. Every access reads as zero and is logged; a tight run of reads
/// to the same address with no intervening different access is flagged by
/// [`SysStub::spinning`] as the firmware waiting on state we haven't wired up.
pub struct SysStub {
    /// Every access in order, as `(addr, value)`.
    log: Vec<(u32, u32)>,
    /// Run-length of the address currently being read consecutively. Holds
    /// at most one entry: the moment a *different* address is read (or any
    /// write happens), the map is cleared and the new run starts at 1 -- so
    /// a nonempty entry here is always the live consecutive-read streak.
    read_counts: HashMap<u32, u32>,
    /// Consecutive-read count past which [`SysStub::spinning`] fires.
    spin_threshold: u32,
}

impl SysStub {
    /// Create an empty stub with the default spin threshold.
    pub fn new() -> Self {
        Self { log: Vec::new(), read_counts: HashMap::new(), spin_threshold: SPIN_THRESHOLD }
    }

    /// The default spin threshold, exposed for tests and callers that want
    /// to reason about the boundary without hardcoding it.
    pub fn new_threshold() -> u32 {
        SPIN_THRESHOLD
    }

    /// Read `addr`: always returns 0, logs the access, and updates the
    /// consecutive-read counter (reset if `addr` differs from the last read).
    pub fn read(&mut self, addr: u32) -> u32 {
        if !self.read_counts.contains_key(&addr) {
            self.read_counts.clear();
        }
        let count = self.read_counts.entry(addr).or_insert(0);
        *count += 1;
        log::debug!("firmware sysstub: read 0x{:08X} -> 0 (consecutive: {})", addr, count);
        self.log.push((addr, 0));
        0
    }

    /// Write `v` to `addr`: no backing state changes, but the access is
    /// logged and breaks any consecutive-read run in progress.
    pub fn write(&mut self, addr: u32, v: u32) {
        self.read_counts.clear();
        log::debug!("firmware sysstub: write 0x{:08X} = 0x{:08X} (stubbed)", addr, v);
        self.log.push((addr, v));
    }

    /// `Some(addr)` if `addr` has just been read more than the spin
    /// threshold times in a row with no intervening different access --
    /// the "waiting on unmodeled state" signal (spec section 8).
    pub fn spinning(&self) -> Option<u32> {
        self.read_counts
            .iter()
            .find(|&(_, &count)| count > self.spin_threshold)
            .map(|(&addr, _)| addr)
    }

    /// The full access log, in order, as `(addr, value)` pairs.
    pub fn accesses(&self) -> &[(u32, u32)] {
        &self.log
    }
}

impl Default for SysStub {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_reads_return_zero_and_log() {
        let mut s = SysStub::new();
        assert_eq!(s.read(0xf7000000), 0);
        assert_eq!(s.accesses().len(), 1);
    }

    #[test]
    fn detects_a_tight_read_spin() {
        let mut s = SysStub::new();
        for _ in 0..(s_threshold() + 1) {
            s.read(0x03001000);
        }
        assert_eq!(s.spinning(), Some(0x03001000));
    }

    #[test]
    fn interleaved_reads_are_not_a_spin() {
        let mut s = SysStub::new();
        for _ in 0..100 {
            s.read(0x03001000);
            s.read(0x03002000);
        }
        assert_eq!(s.spinning(), None);
    }

    fn s_threshold() -> u32 {
        SysStub::new_threshold()
    }
}
