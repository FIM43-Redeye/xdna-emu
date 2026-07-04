//! Xtensa windowed register file: 64 physical AR registers, WINDOWBASE/
//! WINDOWSTART, and the logical->physical rotation the windowed call ABI uses.

/// PS.CALLINC field shift (bits 17:16): the call increment a `callN` records
/// so the matching `entry` knows how far to rotate the window.
const PS_CALLINC_SHIFT: u32 = 16;
/// PS.EXCM bit (bit 4): exception mode. While set, window overflow/underflow
/// detection is masked (the handler runs with EXCM=1 so it can't re-fault).
pub const PS_EXCM: u32 = 1 << 4;
/// PS.WOE bit (bit 18): window-overflow-detection enable. Overflow/underflow
/// exceptions are only raised when WOE=1 (boot sets it once the ABI is live).
pub const PS_WOE: u32 = 1 << 18;

/// Number of window frames (quads) the WINDOWBASE/WINDOWSTART bookkeeping wraps
/// through: 64 physical AR registers / 4 registers per quad.
pub const NUM_FRAMES: u32 = 16;

/// Xtensa windowed register file: 64 physical AR registers with windowed-ABI state.
pub struct RegFile {
    // Physical AR registers; accessed via logical indices through the window mechanism.
    ar: [u32; 64],
    /// Window base register: selects which 16-register block is the logical AR0-AR15.
    pub windowbase: u32,
    /// Window start register: bitmask indicating which window frames contain live state.
    pub windowstart: u32,
    /// Shift amount register: used for bit shifts and field rotation operations.
    pub sar: u32,
    /// Processor state register: current execution state and mode flags.
    pub ps: u32,
}

impl RegFile {
    /// Create a new windowed register file with all registers zeroed and default ABI state.
    pub fn new() -> Self {
        Self { ar: [0; 64], windowbase: 0, windowstart: 1, sar: 0, ps: 0 }
    }

    /// Compute the physical AR index for a logical register number (0-15).
    /// Maps logical to physical via: `(windowbase * 4 + logical) mod 64`.
    pub fn phys(&self, logical: u8) -> usize {
        (((self.windowbase as usize) * 4) + logical as usize) % 64
    }

    /// Read a logical AR register.
    pub fn read_ar(&self, logical: u8) -> u32 {
        self.ar[self.phys(logical)]
    }

    /// Write a logical AR register.
    pub fn write_ar(&mut self, logical: u8, v: u32) {
        let p = self.phys(logical);
        self.ar[p] = v;
    }

    /// Rotate the window by `delta` register-quads (each quad is 4 registers).
    /// `delta` can be positive (forward; used by `entry`) or negative
    /// (backward; used by `retw`).
    pub fn rotate(&mut self, delta: i32) {
        self.windowbase = (self.windowbase as i32 + delta).rem_euclid(NUM_FRAMES as i32) as u32;
    }

    /// The current PS.CALLINC (0-3): how many quads the pending `entry` rotates.
    pub fn callinc(&self) -> u32 {
        (self.ps >> PS_CALLINC_SHIFT) & 0x3
    }

    /// Set PS.CALLINC (low 2 bits of `k`), leaving the rest of PS untouched.
    /// A `callN` records the call size here for the callee's `entry` to read.
    pub fn set_callinc(&mut self, k: u32) {
        self.ps = (self.ps & !(0x3 << PS_CALLINC_SHIFT)) | ((k & 0x3) << PS_CALLINC_SHIFT);
    }

    /// True when window overflow/underflow detection is enabled (PS.WOE=1 and
    /// PS.EXCM=0). Both gates must hold before a window exception can be raised.
    pub fn window_exceptions_enabled(&self) -> bool {
        (self.ps & PS_WOE != 0) && (self.ps & PS_EXCM == 0)
    }

    /// True when PS.EXCM is set (the CPU is in exception mode).
    pub fn excm(&self) -> bool {
        self.ps & PS_EXCM != 0
    }

    /// Enter exception mode (set PS.EXCM), masking further window exceptions.
    pub fn set_excm(&mut self) {
        self.ps |= PS_EXCM;
    }

    /// True if the window frame quad `q` (taken mod [`NUM_FRAMES`]) is marked
    /// live in WINDOWSTART -- i.e. it holds a not-yet-returned frame's registers.
    pub fn frame_live(&self, q: u32) -> bool {
        self.windowstart & (1 << (q % NUM_FRAMES)) != 0
    }

    /// Mark the frame quad `q` (mod [`NUM_FRAMES`]) live in WINDOWSTART.
    pub fn mark_frame_live(&mut self, q: u32) {
        self.windowstart |= 1 << (q % NUM_FRAMES);
    }

    /// Clear the live bit for frame quad `q` (mod [`NUM_FRAMES`]) in WINDOWSTART.
    pub fn clear_frame_live(&mut self, q: u32) {
        self.windowstart &= !(1 << (q % NUM_FRAMES));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_maps_through_windowbase() {
        let mut rf = RegFile::new();
        rf.windowbase = 0; // physical base 0
        rf.write_ar(3, 0xaaaa);
        assert_eq!(rf.phys(3), 3);
        assert_eq!(rf.read_ar(3), 0xaaaa);

        rf.windowbase = 2; // physical base = 2*4 = 8
        rf.write_ar(0, 0xbbbb);
        assert_eq!(rf.phys(0), 8);
        assert_eq!(rf.read_ar(0), 0xbbbb);
    }

    #[test]
    fn rotate_advances_windowbase_mod_16() {
        let mut rf = RegFile::new();
        rf.windowbase = 15;
        rf.rotate(2); // 15 + 2 = 17 -> 1 (mod 16)
        assert_eq!(rf.windowbase, 1);
    }

    #[test]
    fn physical_wraps_mod_64() {
        let mut rf = RegFile::new();
        rf.windowbase = 15; // base = 60
        assert_eq!(rf.phys(7), (60 + 7) % 64); // 3
    }

    #[test]
    fn rotate_negative_delta_retw_decrement() {
        let mut rf = RegFile::new();
        rf.windowbase = 1;
        rf.rotate(-3); // 1 - 3 = -2 -> 14 (mod 16)
        assert_eq!(rf.windowbase, 14);
    }

    #[test]
    fn callinc_round_trips_through_ps() {
        let mut rf = RegFile::new();
        rf.set_callinc(2);
        assert_eq!(rf.callinc(), 2);
        // Only the 2-bit field is written; the rest of PS is untouched.
        rf.ps |= PS_WOE;
        rf.set_callinc(3);
        assert_eq!(rf.callinc(), 3);
        assert_ne!(rf.ps & PS_WOE, 0);
    }

    #[test]
    fn window_exceptions_gated_on_woe_and_not_excm() {
        let mut rf = RegFile::new();
        assert!(!rf.window_exceptions_enabled(), "WOE clear by default");
        rf.ps |= PS_WOE;
        assert!(rf.window_exceptions_enabled());
        rf.set_excm();
        assert!(!rf.window_exceptions_enabled(), "EXCM masks detection");
        assert!(rf.excm());
    }

    #[test]
    fn frame_live_marks_and_clears_mod_frames() {
        let mut rf = RegFile::new();
        rf.windowstart = 0;
        rf.mark_frame_live(2);
        assert!(rf.frame_live(2));
        assert!(!rf.frame_live(3));
        // Indices wrap mod NUM_FRAMES: quad 18 aliases quad 2.
        assert!(rf.frame_live(2 + NUM_FRAMES));
        rf.clear_frame_live(2);
        assert!(!rf.frame_live(2));
    }
}
