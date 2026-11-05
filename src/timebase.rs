//! Clock domains and simulated time.
//!
//! The NPU runs two clocks while a context is live: the management core
//! (firmware) and the array ("H"). They are coupled here by an exact
//! rational accumulator, never a float, so a run is deterministic and the
//! firmware:array ratio is whatever the configured pair says.
//!
//! The pair comes from this one place. Points are named after where their
//! numbers come from: the driver's DPM table (`npu1_dpm_clk_table` in
//! amdxdna's `npu1_regs.c`) and what the SMU reported on silicon.

/// A firmware clock and an array clock, in MHz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockPair {
    pub firmware_mhz: u32,
    pub array_mhz: u32,
}

impl ClockPair {
    /// Driver DPM table entry 0 (and the SMU report at QoS gops=1, fps=1000).
    pub const DRIVER_DPM0: ClockPair = ClockPair { firmware_mhz: 400, array_mhz: 800 };
    /// Driver DPM table entry 1 as written in the table.
    pub const DRIVER_DPM1: ClockPair = ClockPair { firmware_mhz: 600, array_mhz: 1024 };
    /// What the SMU reports while a context runs at the default DPM point
    /// (silicon, Phoenix): the table's 1024 MHz reads back as 1028.
    pub const SMU_DPM1: ClockPair = ClockPair { firmware_mhz: 600, array_mhz: 1028 };

    /// The pair used when nothing is configured: the silicon-reported default.
    pub const DEFAULT: ClockPair = Self::SMU_DPM1;

    /// The pair for a DPM index as the emulator selects it (`XDNA_EMU_DPM`).
    pub fn for_dpm(index: u32) -> Option<ClockPair> {
        match index {
            0 => Some(Self::DRIVER_DPM0),
            1 => Some(Self::SMU_DPM1),
            _ => None,
        }
    }

    /// The configured pair: `XDNA_EMU_DPM` (0 or 1) if set, else [`Self::DEFAULT`].
    /// An unparsable or unknown value is an error, not a silent default.
    pub fn configured() -> Result<ClockPair, String> {
        match std::env::var("XDNA_EMU_DPM") {
            Err(_) => Ok(Self::DEFAULT),
            Ok(v) => v
                .trim()
                .parse::<u32>()
                .ok()
                .and_then(Self::for_dpm)
                .ok_or_else(|| format!("XDNA_EMU_DPM={v:?}: expected 0 or 1")),
        }
    }

    /// Simulated wall time of `array_cycles`, in picoseconds (truncated).
    pub fn array_cycles_to_ps(&self, array_cycles: u64) -> u64 {
        (u128::from(array_cycles) * 1_000_000 / u128::from(self.array_mhz)) as u64
    }

    /// Simulated wall time of `array_cycles`, in nanoseconds (truncated).
    pub fn array_cycles_to_ns(&self, array_cycles: u64) -> u64 {
        self.array_cycles_to_ps(array_cycles) / 1000
    }
}

/// Grants whole firmware cycles per array cycle from the exact ratio.
#[derive(Debug, Clone)]
pub struct Coupler {
    pair: ClockPair,
    /// Fractional firmware cycles carried between array cycles, in units of
    /// 1/`array_mhz` cycle (always `< array_mhz`).
    remainder: u32,
}

impl Coupler {
    pub fn new(pair: ClockPair) -> Self {
        assert!(pair.array_mhz > 0 && pair.firmware_mhz > 0, "clocks must be nonzero");
        Self { pair, remainder: 0 }
    }

    pub fn pair(&self) -> ClockPair {
        self.pair
    }

    /// One array cycle elapses: the whole firmware cycles it grants.
    pub fn grant(&mut self) -> u32 {
        let total = self.remainder + self.pair.firmware_mhz;
        self.remainder = total % self.pair.array_mhz;
        total / self.pair.array_mhz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grants_converge_to_the_exact_ratio_with_no_drift() {
        let mut c = Coupler::new(ClockPair::SMU_DPM1);
        let granted: u64 = (0..1028).map(|_| u64::from(c.grant())).sum();
        assert_eq!(granted, 600, "1028 array cycles are exactly 600 firmware cycles");
        let granted: u64 = (0..10 * 1028).map(|_| u64::from(c.grant())).sum();
        assert_eq!(granted, 6000);
    }

    #[test]
    fn a_single_grant_never_exceeds_the_ceiling_of_the_ratio() {
        let mut c = Coupler::new(ClockPair { firmware_mhz: 1500, array_mhz: 1000 });
        let g: Vec<u32> = (0..4).map(|_| c.grant()).collect();
        assert_eq!(g, vec![1, 2, 1, 2]);
    }

    #[test]
    fn dpm0_is_a_configuration_not_a_code_change() {
        let mut c = Coupler::new(ClockPair::for_dpm(0).unwrap());
        assert_eq!((0..800).map(|_| u64::from(c.grant())).sum::<u64>(), 400);
    }

    #[test]
    fn wall_time_is_cycles_over_the_array_clock() {
        let p = ClockPair::SMU_DPM1;
        assert_eq!(p.array_cycles_to_ps(1028), 1_000_000);
        assert_eq!(p.array_cycles_to_ns(1028), 1000);
        assert_eq!(ClockPair::DRIVER_DPM0.array_cycles_to_ns(800), 1000);
        assert_eq!(p.array_cycles_to_ps(0), 0);
    }

    #[test]
    fn dpm_index_selects_the_pair() {
        assert_eq!(ClockPair::for_dpm(0), Some(ClockPair::DRIVER_DPM0));
        assert_eq!(ClockPair::for_dpm(1), Some(ClockPair::SMU_DPM1));
        assert_eq!(ClockPair::for_dpm(2), None);
    }
}
