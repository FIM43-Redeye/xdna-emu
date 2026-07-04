//! Xtensa windowed register file: 64 physical AR registers, WINDOWBASE/
//! WINDOWSTART, and the logical->physical rotation the windowed call ABI uses.

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
    /// `delta` can be positive (forward; used by `call8`) or negative (backward; used by `retw`).
    pub fn rotate(&mut self, delta: i32) {
        self.windowbase = (self.windowbase as i32 + delta).rem_euclid(16) as u32;
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
}
