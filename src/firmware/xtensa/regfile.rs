//! Xtensa windowed register file: 64 physical AR registers, WINDOWBASE/
//! WINDOWSTART, and the logical->physical rotation the windowed call ABI uses.

pub struct RegFile {
    ar: [u32; 64],
    pub windowbase: u32,
    pub windowstart: u32,
    pub sar: u32,
    pub ps: u32,
}

impl RegFile {
    pub fn new() -> Self {
        Self { ar: [0; 64], windowbase: 0, windowstart: 1, sar: 0, ps: 0 }
    }

    pub fn phys(&self, logical: u8) -> usize {
        (((self.windowbase as usize) * 4) + logical as usize) % 64
    }

    pub fn read_ar(&self, logical: u8) -> u32 {
        self.ar[self.phys(logical)]
    }

    pub fn write_ar(&mut self, logical: u8, v: u32) {
        let p = self.phys(logical);
        self.ar[p] = v;
    }

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
}
