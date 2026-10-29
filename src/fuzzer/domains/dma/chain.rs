//! DMA chain AST. A case = N>=1 transfers, each carrying one BD access pattern.
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    I32,
    I16,
    I8,
}

impl Dtype {
    /// MLIR memref element type.
    pub fn mlir_elem(self) -> &'static str {
        match self {
            Dtype::I32 => "i32",
            Dtype::I16 => "i16",
            Dtype::I8 => "i8",
        }
    }
    /// String returned from `Domain::dtype` (unused by the DMA compile override,
    /// but the trait requires it).
    pub fn template_dtype(self) -> &'static str {
        self.mlir_elem()
    }
    pub fn byte_size(self) -> usize {
        match self {
            Dtype::I32 => 4,
            Dtype::I16 => 2,
            Dtype::I8 => 1,
        }
    }
    /// Coverage-key token.
    pub fn key_str(self) -> &'static str {
        match self {
            Dtype::I32 => "I32",
            Dtype::I16 => "I16",
            Dtype::I8 => "I8",
        }
    }
    pub fn all() -> [Dtype; 3] {
        [Dtype::I32, Dtype::I16, Dtype::I8]
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Engine {
    Shim,
    Memtile,
}

impl Engine {
    pub fn key_str(self) -> &'static str {
        match self {
            Engine::Shim => "shim",
            Engine::Memtile => "memtile",
        }
    }
    pub fn all() -> [Engine; 2] {
        [Engine::Shim, Engine::Memtile]
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    Mm2s,
    S2mm,
}

impl Direction {
    pub fn key_str(self) -> &'static str {
        match self {
            Direction::Mm2s => "mm2s",
            Direction::S2mm => "s2mm",
        }
    }
    pub fn all() -> [Direction; 2] {
        [Direction::Mm2s, Direction::S2mm]
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Feature {
    Linear,
    Strided2d,
    Strided3d,
    Strided4d,
    Transpose,
    Overlap,
    Packet,
    PadBefore,
    PadAfter,
    PadBoth,
    /// 3b: shim outermost-dim BD-repeat. In the enum so 3b is a gating flip.
    Iter,
    /// 3b: memtile multi-BD next_bd chain.
    Chain,
}

impl Feature {
    pub fn key_str(self) -> &'static str {
        match self {
            Feature::Linear => "linear",
            Feature::Strided2d => "strided2d",
            Feature::Strided3d => "strided3d",
            Feature::Strided4d => "strided4d",
            Feature::Transpose => "transpose",
            Feature::Overlap => "overlap",
            Feature::Packet => "packet",
            Feature::PadBefore => "padbefore",
            Feature::PadAfter => "padafter",
            Feature::PadBoth => "padboth",
            Feature::Iter => "iter",
            Feature::Chain => "chain",
        }
    }
    /// All 3a features (excludes Iter/Chain). Used as the random-filler pool, so
    /// fillers never structurally become an iter/chain transfer.
    pub fn all_3a() -> [Feature; 10] {
        [
            Feature::Linear,
            Feature::Strided2d,
            Feature::Strided3d,
            Feature::Strided4d,
            Feature::Transpose,
            Feature::Overlap,
            Feature::Packet,
            Feature::PadBefore,
            Feature::PadAfter,
            Feature::PadBoth,
        ]
    }
    /// Every feature (3a + the 3b structural features iter/chain). The coverage
    /// universe enumerates this and lets `table::supported` gate per engine/dir.
    pub fn all() -> [Feature; 12] {
        [
            Feature::Linear,
            Feature::Strided2d,
            Feature::Strided3d,
            Feature::Strided4d,
            Feature::Transpose,
            Feature::Overlap,
            Feature::Packet,
            Feature::PadBefore,
            Feature::PadAfter,
            Feature::PadBoth,
            Feature::Iter,
            Feature::Chain,
        ]
    }
}

/// One BD access pattern, in MLIR list order (sizes[0]/strides[0] = outermost).
/// All values in dtype elements. Stage-A-safe by construction (gen clamps).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BdPattern {
    pub sizes: Vec<u32>,
    pub strides: Vec<u32>,
    /// Empty = no padding; else len == sizes.len() (memtile MM2S only).
    pub pad_before: Vec<u32>,
    pub pad_after: Vec<u32>,
    /// (pkt_id, pkt_type) when packet-switched.
    pub packet: Option<(u8, u8)>,
}

impl BdPattern {
    /// Element count physically moved into the destination (data only; the
    /// product of the access-pattern sizes).
    pub fn data_elems(&self) -> usize {
        self.sizes.iter().map(|&s| s as usize).product::<usize>().max(1)
    }
    /// Total per-dim padding added (memtile MM2S). Padding inserts this many zero
    /// elements into the streamed output beyond the data.
    pub fn pad_elems(&self) -> usize {
        if self.pad_before.is_empty() {
            return 0;
        }
        let padded: usize = self
            .sizes
            .iter()
            .zip(&self.pad_before)
            .zip(&self.pad_after)
            .map(|((&s, &b), &a)| (s + b + a) as usize)
            .product();
        padded.saturating_sub(self.data_elems())
    }
}

/// One transfer: a region reshuffle on a named engine/direction/feature.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DmaTransfer {
    pub engine: Engine,
    pub dir: Direction,
    pub feature: Feature,
    pub pattern: BdPattern,
    /// Element offset into the input DDR buffer.
    pub in_off: usize,
    /// Element offset into the output DDR buffer.
    pub out_off: usize,
    /// Input region element count (data, pre-pad).
    pub in_elems: usize,
    /// Output region element count (= in_elems + pattern.pad_elems()).
    pub out_elems: usize,
}

impl DmaTransfer {
    pub fn key(&self, dtype: Dtype) -> String {
        format!(
            "{}/{}/{}/{}",
            self.feature.key_str(),
            self.engine.key_str(),
            self.dir.key_str(),
            dtype.key_str()
        )
    }
    /// (start, end) byte bounds of this transfer's output region.
    pub fn out_byte_bounds(&self, dtype: Dtype) -> (usize, usize) {
        let bs = dtype.byte_size();
        (self.out_off * bs, (self.out_off + self.out_elems) * bs)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DmaChain {
    pub seed: u64,
    pub target_key: String,
    pub dtype: Dtype,
    /// One engine per case (the target's engine).
    pub engine: Engine,
    pub transfers: Vec<DmaTransfer>,
}

impl DmaChain {
    /// Per-transfer coverage keys, in transfer order.
    pub fn keys(&self) -> Vec<String> {
        self.transfers.iter().map(|t| t.key(self.dtype)).collect()
    }
    /// Total input buffer size in dtype elements.
    pub fn in_words(&self) -> usize {
        self.transfers.iter().map(|t| t.in_elems).sum()
    }
    /// Total output buffer size in dtype elements.
    pub fn out_words(&self) -> usize {
        self.transfers.iter().map(|t| t.out_elems).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_format() {
        let t = DmaTransfer {
            engine: Engine::Shim,
            dir: Direction::Mm2s,
            feature: Feature::Transpose,
            pattern: BdPattern {
                sizes: vec![8, 8],
                strides: vec![1, 8],
                pad_before: vec![],
                pad_after: vec![],
                packet: None,
            },
            in_off: 0,
            out_off: 0,
            in_elems: 64,
            out_elems: 64,
        };
        assert_eq!(t.key(Dtype::I16), "transpose/shim/mm2s/I16");
    }

    #[test]
    fn pad_elems_arithmetic() {
        let p = BdPattern {
            sizes: vec![8],
            strides: vec![1],
            pad_before: vec![4],
            pad_after: vec![4],
            packet: None,
        };
        assert_eq!(p.data_elems(), 8);
        assert_eq!(p.pad_elems(), 8);
    }

    #[test]
    fn footprints_sum_regions() {
        let mk = |in_off, out_off, ie, oe| DmaTransfer {
            engine: Engine::Memtile,
            dir: Direction::Mm2s,
            feature: Feature::Linear,
            pattern: BdPattern {
                sizes: vec![ie as u32],
                strides: vec![1],
                pad_before: vec![],
                pad_after: vec![],
                packet: None,
            },
            in_off,
            out_off,
            in_elems: ie,
            out_elems: oe,
        };
        let c = DmaChain {
            seed: 1,
            target_key: "linear/memtile/mm2s/I32".into(),
            dtype: Dtype::I32,
            engine: Engine::Memtile,
            transfers: vec![mk(0, 0, 64, 64), mk(64, 64, 64, 80)],
        };
        assert_eq!(c.in_words(), 128);
        assert_eq!(c.out_words(), 144);
    }
}
