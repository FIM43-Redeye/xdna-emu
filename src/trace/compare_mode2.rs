//! Mode-2 trace comparator: three-layer HW/EMU comparison.
//!
//! See spec section 8 of `docs/superpowers/specs/2026-04-29-a2b-mode2-design.md`.
//! The Layer 1 + Layer 2 results gate test pass/fail; Layer 3 (atom windows)
//! is informational only -- DMA-stall nondeterminism makes it noisy in
//! aggregate, so it never gates a test.

use crate::trace::compare::TileKey;
use crate::trace::mode2_decode::{decode, Mode2Frame};

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

fn extract_pcs(stream: &[u8]) -> Vec<u16> {
    decode(stream)
        .into_iter()
        .filter_map(|f| match f {
            Mode2Frame::NewPc { pc } => Some(pc),
            _ => None,
        })
        .collect()
}

fn compare_layer1(hw: &[u16], emu: &[u16]) -> PcSequenceLayer {
    let mut samples = Vec::new();
    let mut first = None;
    let n = hw.len().min(emu.len());
    for i in 0..n {
        if hw[i] != emu[i] {
            samples.push(PcSampleRow { idx: i, hw_pc: hw[i], emu_pc: emu[i] });
            if first.is_none() {
                first = Some(i);
            }
        }
    }
    if hw.len() != emu.len() && first.is_none() {
        first = Some(n);
    }
    PcSequenceLayer { hw_count: hw.len(), emu_count: emu.len(), first_diverge: first, samples }
}

/// Compare HW and EMU mode-2 byte streams for one tile.
///
/// Returns a result with all three layers populated. `passed` is `true`
/// iff Layer 1 (PC sequence) and Layer 2 (LC count + flag) both have
/// zero divergences. Layer 3 (atom windows) is informational only.
pub fn compare_mode2_for_tile(hw_stream: &[u8], emu_stream: &[u8], tile: TileKey) -> Mode2CompareResult {
    let hw_pcs = extract_pcs(hw_stream);
    let emu_pcs = extract_pcs(emu_stream);
    let layer1 = compare_layer1(&hw_pcs, &emu_pcs);

    let layer2 = LcLayer { hw_count: 0, emu_count: 0, first_diverge: None, samples: Vec::new() }; // populated in Task 5.3
    let layer3 = AtomWindowLayer::default(); // populated in Task 5.4
    let passed = layer1.first_diverge.is_none() && layer2.first_diverge.is_none();
    Mode2CompareResult { tile, layer1, layer2, layer3, passed }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::compare::TileKey;

    /// Encode a sequence of New_PC frames + Filler0 padding.
    fn encode_pcs(pcs: &[u16]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut bit_pos: u32 = 0;
        let mut word: u32 = 0;
        let mut push_bits = |val: u32, count: u32, out: &mut Vec<u8>, word: &mut u32, bit_pos: &mut u32| {
            for i in (0..count).rev() {
                let b = (val >> i) & 1;
                *word = (*word << 1) | b;
                *bit_pos += 1;
                if *bit_pos == 32 {
                    out.push((*word >> 24) as u8);
                    out.push((*word >> 16) as u8);
                    out.push((*word >> 8) as u8);
                    out.push(*word as u8);
                    *word = 0;
                    *bit_pos = 0;
                }
            }
        };
        for &pc in pcs {
            let frame = (0b10u32 << 14) | (pc as u32 & 0x3FFF);
            push_bits(frame, 16, &mut out, &mut word, &mut bit_pos);
        }
        // Pad with Filler0 nibbles
        while bit_pos != 0 {
            push_bits(0b0010, 4, &mut out, &mut word, &mut bit_pos);
        }
        out
    }

    fn key() -> TileKey {
        TileKey { col: 0, row: 2, pkt_type: 0 }
    }

    #[test]
    fn layer1_no_divergence_when_pcs_match() {
        let hw = encode_pcs(&[0x100, 0x200, 0x300]);
        let emu = encode_pcs(&[0x100, 0x200, 0x300]);
        let r = compare_mode2_for_tile(&hw, &emu, key());
        assert!(r.layer1.first_diverge.is_none());
        assert_eq!(r.layer1.hw_count, 3);
        assert_eq!(r.layer1.emu_count, 3);
        assert!(r.passed);
    }

    #[test]
    fn layer1_finds_divergence() {
        let hw = encode_pcs(&[0x100, 0x200, 0x300]);
        let emu = encode_pcs(&[0x100, 0x250, 0x300]);
        let r = compare_mode2_for_tile(&hw, &emu, key());
        assert_eq!(r.layer1.first_diverge, Some(1));
        assert!(!r.passed);
    }

    #[test]
    fn layer1_handles_count_mismatch() {
        let hw = encode_pcs(&[0x100, 0x200, 0x300]);
        let emu = encode_pcs(&[0x100, 0x200]);
        let r = compare_mode2_for_tile(&hw, &emu, key());
        assert_eq!(r.layer1.hw_count, 3);
        assert_eq!(r.layer1.emu_count, 2);
        assert!(r.layer1.first_diverge.is_some() || !r.passed);
    }
}
