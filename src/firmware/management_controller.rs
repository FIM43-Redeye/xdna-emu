const ENABLE_BASE: u32 = 0x2720_0300;
const STATUS_BASE: u32 = 0x2720_03b0;
const ACTIVE_SOURCE: u32 = 0x2720_03c4;
const BANKS: usize = 4;

#[derive(Default)]
pub(crate) struct ManagementController {
    enable: [u32; BANKS],
    status: [u32; BANKS],
    active_source: Option<u8>,
    irq_assertion_queued: bool,
}

impl ManagementController {
    pub(crate) fn read32(&self, address: u32) -> Option<u32> {
        if let Some(bank) = bank_at(address, ENABLE_BASE) {
            return Some(self.enable[bank]);
        }
        if let Some(bank) = bank_at(address, STATUS_BASE) {
            return Some(self.status[bank]);
        }
        (address == ACTIVE_SOURCE).then_some(self.active_source.unwrap_or(0) as u32)
    }

    pub(crate) fn write32(&mut self, address: u32, value: u32) -> bool {
        if let Some(bank) = bank_at(address, ENABLE_BASE) {
            self.enable[bank] = value;
            return true;
        }
        if let Some(bank) = bank_at(address, STATUS_BASE) {
            self.status[bank] &= !value;
            if let Some(source) = self.active_source {
                let (active_bank, active_bit) = source_location(source);
                if active_bank == bank && value & active_bit != 0 {
                    self.active_source = None;
                }
            }
            return true;
        }
        address == ACTIVE_SOURCE
    }

    // Explicit assertions remain useful for isolated controller tests.
    #[allow(dead_code)]
    pub(crate) fn assert_source(&mut self, source: u8) -> bool {
        let (bank, bit) = source_location(source);
        if bank >= BANKS {
            return false;
        }
        if self.active_source.is_some() || self.enable[bank] & bit == 0 {
            return false;
        }

        self.status[bank] |= bit;
        self.active_source = Some(source);
        self.irq_assertion_queued = true;
        true
    }

    pub(crate) fn take_irq_assertion(&mut self) -> bool {
        std::mem::take(&mut self.irq_assertion_queued)
    }
}

fn bank_at(address: u32, base: u32) -> Option<usize> {
    let offset = address.checked_sub(base)?;
    (offset % 4 == 0).then_some((offset / 4) as usize).filter(|&bank| bank < BANKS)
}

fn source_location(source: u8) -> (usize, u32) {
    ((source >> 5) as usize, 1 << (source & 31))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_source_is_rejected() {
        let mut controller = ManagementController::default();

        assert!(!controller.assert_source(46));
        assert!(!controller.assert_source(128));
        assert_eq!(controller.read32(0x2720_03b4), Some(0));
        assert_eq!(controller.read32(0x2720_03c4), Some(0));
        assert!(!controller.take_irq_assertion());
    }

    #[test]
    fn enabled_source_sets_status_active_source_and_aggregate_assertion() {
        let mut controller = ManagementController::default();
        controller.write32(0x2720_0304, 1 << 14);

        assert!(controller.assert_source(46));
        assert_eq!(controller.read32(0x2720_03b4), Some(1 << 14));
        assert_eq!(controller.read32(0x2720_03c4), Some(46));
        assert!(controller.take_irq_assertion());
        assert!(!controller.take_irq_assertion());
    }

    #[test]
    fn acknowledgement_clears_source_without_clearing_enable() {
        let mut controller = ManagementController::default();
        controller.write32(0x2720_0304, 1 << 14);
        assert!(controller.assert_source(46));

        controller.write32(0x2720_03b4, 1 << 14);

        assert_eq!(controller.read32(0x2720_0304), Some(1 << 14));
        assert_eq!(controller.read32(0x2720_03b4), Some(0));
        assert_eq!(controller.read32(0x2720_03c4), Some(0));
    }

    #[test]
    fn competing_source_is_rejected_while_one_is_active() {
        let mut controller = ManagementController::default();
        controller.write32(0x2720_0304, (1 << 14) | (1 << 15));
        assert!(controller.assert_source(46));

        assert!(!controller.assert_source(47));
        assert_eq!(controller.read32(0x2720_03b4), Some(1 << 14));
        assert_eq!(controller.read32(0x2720_03c4), Some(46));
    }
}
