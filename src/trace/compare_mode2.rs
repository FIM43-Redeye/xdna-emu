//! Mode-2 trace comparator: three-layer HW/EMU comparison.
//!
//! See spec section 8 of `docs/superpowers/specs/2026-04-29-a2b-mode2-design.md`.
//! The Layer 1 + Layer 2 results gate test pass/fail; Layer 3 (atom windows)
//! is informational only -- DMA-stall nondeterminism makes it noisy in
//! aggregate, so it never gates a test.

use crate::trace::compare::TileKey;

/// Outcome of comparing one tile's mode-2 streams.
#[derive(Debug, Clone)]
pub struct Mode2CompareResult {
    pub tile: TileKey,
    pub layer1: PcSequenceLayer,
    pub layer2: LcLayer,
    pub layer3: AtomWindowLayer,
    /// True iff layer1 + layer2 both have zero divergences.
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct PcSequenceLayer {
    pub hw_count: usize,
    pub emu_count: usize,
    /// Index of first divergence; None means streams agreed up through
    /// min(hw_count, emu_count).
    pub first_diverge: Option<usize>,
    pub samples: Vec<PcSampleRow>,
}

#[derive(Debug, Clone)]
pub struct PcSampleRow {
    pub idx: usize,
    pub hw_pc: u16,
    pub emu_pc: u16,
}

#[derive(Debug, Clone)]
pub struct LcLayer {
    pub hw_count: usize,
    pub emu_count: usize,
    pub first_diverge: Option<usize>,
    pub samples: Vec<LcSampleRow>,
}

#[derive(Debug, Clone)]
pub struct LcSampleRow {
    pub idx: usize,
    pub hw_flag: u8,
    pub hw_count_value: u32,
    pub emu_flag: u8,
    pub emu_count_value: u32,
}

#[derive(Debug, Clone, Default)]
pub struct AtomWindowLayer {
    pub windows: Vec<AtomWindow>,
}

#[derive(Debug, Clone)]
pub struct AtomWindow {
    pub from_pc: u16,
    pub to_pc: u16,
    pub hw_atoms: usize,
    pub emu_atoms: usize,
}

/// Compare HW and EMU mode-2 byte streams for one tile.
///
/// Returns a result with all three layers populated. `passed` is `true`
/// iff Layer 1 (PC sequence) and Layer 2 (LC count + flag) both have
/// zero divergences. Layer 3 (atom windows) is informational only.
pub fn compare_mode2_for_tile(_hw_stream: &[u8], _emu_stream: &[u8], tile: TileKey) -> Mode2CompareResult {
    // Layers populated by Tasks 5.2 / 5.3 / 5.4.
    Mode2CompareResult {
        tile,
        layer1: PcSequenceLayer { hw_count: 0, emu_count: 0, first_diverge: None, samples: Vec::new() },
        layer2: LcLayer { hw_count: 0, emu_count: 0, first_diverge: None, samples: Vec::new() },
        layer3: AtomWindowLayer::default(),
        passed: true,
    }
}

#[cfg(test)]
mod tests {
    // Tests added in Tasks 5.2+.
}
