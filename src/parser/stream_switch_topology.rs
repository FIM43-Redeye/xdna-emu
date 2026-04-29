//! Stream-switch topology reconstructed from CDO register writes.
//!
//! This module replays `DeviceOp::RegWrite` / `DeviceOp::RegMask` ops that
//! target stream-switch configuration registers and materialises the
//! resulting switch state per tile. The reconstructed graph is consumed
//! by several downstream tools:
//!
//! - the kernarg classifier (`src/npu/classify.rs`), which needs to know
//!   which shim DMA channels are packet-routed (ctrlpkt carriers);
//! - the visual debugger (roadmap phase 3), which renders tile-to-tile
//!   packet flows and highlights congested fabric;
//! - the trace-injection planner (`tools/mlir-trace-inject.py` today),
//!   which must avoid routing conflicts with existing flows.
//!
//! The data here is intentionally low-level — we preserve the raw CDO
//! register values so callers can inspect whatever field they need, and
//! we expose a few convenience queries on top. We do *not* try to render
//! an abstract "route graph" in this first pass; slot-arbitration
//! semantics and master-select matrices are captured as raw state and
//! can be walked by callers.
//!
//! Reference: register layout from aie-rt `xaiemlgbl_params.h`
//! (`XAIEMLGBL_{CORE,PL,MEM_TILE}_MODULE_STREAM_SWITCH_*`). The AM025
//! register database JSON carries the same bit positions.

use std::collections::HashMap;

use xdna_archspec::types::TileAddr;

use crate::device::ops::DeviceOp;

// Tile-local register ranges covering the stream-switch config block. The
// same tile-local offsets are used by CORE, PL (shim), and MEM_TILE
// modules; the semantics of a given offset depend on the tile's kind,
// not on the numerical range. Consumers of the topology disambiguate by
// looking at `TileAddr::row` and archspec's tile-kind mapping.
const SS_MASTER_CONFIG_RANGE: std::ops::Range<u32> = 0x3F000..0x3F100;
const SS_SLAVE_CONFIG_RANGE: std::ops::Range<u32> = 0x3F100..0x3F200;
const SS_SLOT_RANGE: std::ops::Range<u32> = 0x3F200..0x3F400;

// Master Port Config fields (from reginit.c `*MasterPortProp`):
//   bit 31: Master_Enable
//   bit 30: Packet_Enable
//   bit 29: Config (pause/credit; not inspected here)
//   bits 0..5: DropHeader + Configuration (Cntrl/DataOnly)
const MASTER_ENABLE_BIT: u32 = 31;
const MASTER_PACKET_ENABLE_BIT: u32 = 30;

// Slave Port Config fields (`*SlavePortProp`):
//   bit 31: Slave_Enable
//   bit 30: Packet_Enable
const SLAVE_ENABLE_BIT: u32 = 31;
const SLAVE_PACKET_ENABLE_BIT: u32 = 30;

// Slot Config fields (`SlaveSlotProp`):
//   bit 31: Valid_ID
//   bits 28..29: Arbiter (2 bits)
//   bits 24..26: Msel (3 bits) -- master-select mask
//   bits 16..20: Mask (5 bits)
//   bits 12..14: ID_Type / Packet_Type (3 bits)
//   bits  0.. 4: ID / Packet_ID (5 bits)
const SLOT_VALID_BIT: u32 = 31;
const SLOT_ARBITER_LSB: u32 = 28;
const SLOT_ARBITER_MASK: u32 = 0b11;
const SLOT_MSEL_LSB: u32 = 24;
const SLOT_MSEL_MASK: u32 = 0b111;
const SLOT_MASK_LSB: u32 = 16;
const SLOT_MASK_MASK: u32 = 0b11111;
const SLOT_PKT_TYPE_LSB: u32 = 12;
const SLOT_PKT_TYPE_MASK: u32 = 0b111;
const SLOT_PKT_ID_LSB: u32 = 0;
const SLOT_PKT_ID_MASK: u32 = 0b11111;

/// Configuration of a single stream-switch master (output) port.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MasterPortConfig {
    pub enabled: bool,
    pub packet_mode: bool,
    /// Raw register value at the last write; lets callers inspect fields
    /// not unpacked above.
    pub raw: u32,
}

/// Configuration of a single stream-switch slave (input) port.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlavePortConfig {
    pub enabled: bool,
    pub packet_mode: bool,
    pub raw: u32,
}

/// One packet-arbitration slot on a slave port. A slave in packet mode
/// has four such slots; the first one whose `(packet_id, packet_type)`
/// matches (after masking) dispatches the packet to the master(s)
/// selected by `master_select`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlotConfig {
    pub valid: bool,
    pub arbiter: u8,
    pub master_select: u8,
    pub mask: u8,
    pub packet_type: u8,
    pub packet_id: u8,
    pub raw: u32,
}

/// Stream-switch configuration for one tile. Indexed by tile-local
/// register offset so callers can look up specific named ports without
/// this module having to carry the full AIE-ML port enum.
#[derive(Debug, Default, Clone)]
pub struct TileSwitchConfig {
    pub masters: HashMap<u32, MasterPortConfig>,
    pub slaves: HashMap<u32, SlavePortConfig>,
    /// Keyed by (slot-register offset). Each slot's offset is
    /// `slave_slot_group_base + slot_index * 4`, where
    /// `slave_slot_group_base` is the 16-byte-aligned base for the
    /// slave (e.g. 0x3F200 for TILE_CTRL slave, 0x3F220 for SOUTH_0).
    pub slots: HashMap<u32, SlotConfig>,
}

impl TileSwitchConfig {
    pub fn packet_masters(&self) -> impl Iterator<Item = (u32, &MasterPortConfig)> {
        self.masters
            .iter()
            .filter(|(_, m)| m.enabled && m.packet_mode)
            .map(|(off, m)| (*off, m))
    }

    pub fn packet_slaves(&self) -> impl Iterator<Item = (u32, &SlavePortConfig)> {
        self.slaves
            .iter()
            .filter(|(_, s)| s.enabled && s.packet_mode)
            .map(|(off, s)| (*off, s))
    }
}

/// Cross-tile stream-switch topology.
#[derive(Debug, Default, Clone)]
pub struct StreamSwitchTopology {
    pub tiles: HashMap<TileAddr, TileSwitchConfig>,
}

impl StreamSwitchTopology {
    /// Replay a sequence of DeviceOps and return the resulting topology.
    pub fn from_device_ops<I>(ops: I) -> Self
    where
        I: IntoIterator<Item = DeviceOp>,
    {
        let mut topo = Self::default();
        for op in ops {
            topo.apply(op);
        }
        topo
    }

    /// Apply one DeviceOp. Non-SS ops are ignored.
    pub fn apply(&mut self, op: DeviceOp) {
        match op {
            DeviceOp::RegWrite { tile, offset, value } => self.apply_write(tile, offset, value),
            DeviceOp::RegMask { tile, offset, mask, value } => {
                let prior = self.read_raw(tile, offset).unwrap_or(0);
                self.apply_write(tile, offset, (prior & !mask) | (value & mask));
            }
            _ => {}
        }
    }

    fn apply_write(&mut self, tile: TileAddr, offset: u32, value: u32) {
        if !touches_ss(offset) {
            return;
        }
        let ts = self.tiles.entry(tile).or_default();
        if SS_MASTER_CONFIG_RANGE.contains(&offset) {
            ts.masters.insert(
                offset,
                MasterPortConfig {
                    enabled: bit(value, MASTER_ENABLE_BIT),
                    packet_mode: bit(value, MASTER_PACKET_ENABLE_BIT),
                    raw: value,
                },
            );
        } else if SS_SLAVE_CONFIG_RANGE.contains(&offset) {
            ts.slaves.insert(
                offset,
                SlavePortConfig {
                    enabled: bit(value, SLAVE_ENABLE_BIT),
                    packet_mode: bit(value, SLAVE_PACKET_ENABLE_BIT),
                    raw: value,
                },
            );
        } else if SS_SLOT_RANGE.contains(&offset) {
            ts.slots.insert(offset, decode_slot(value));
        }
    }

    fn read_raw(&self, tile: TileAddr, offset: u32) -> Option<u32> {
        let ts = self.tiles.get(&tile)?;
        if SS_MASTER_CONFIG_RANGE.contains(&offset) {
            ts.masters.get(&offset).map(|m| m.raw)
        } else if SS_SLAVE_CONFIG_RANGE.contains(&offset) {
            ts.slaves.get(&offset).map(|s| s.raw)
        } else if SS_SLOT_RANGE.contains(&offset) {
            ts.slots.get(&offset).map(|s| s.raw)
        } else {
            None
        }
    }

    pub fn packet_masters(&self) -> impl Iterator<Item = (TileAddr, u32, &MasterPortConfig)> {
        self.tiles
            .iter()
            .flat_map(|(addr, ts)| ts.packet_masters().map(move |(off, m)| (*addr, off, m)))
    }

    pub fn packet_slaves(&self) -> impl Iterator<Item = (TileAddr, u32, &SlavePortConfig)> {
        self.tiles
            .iter()
            .flat_map(|(addr, ts)| ts.packet_slaves().map(move |(off, s)| (*addr, off, s)))
    }
}

fn touches_ss(offset: u32) -> bool {
    SS_MASTER_CONFIG_RANGE.contains(&offset)
        || SS_SLAVE_CONFIG_RANGE.contains(&offset)
        || SS_SLOT_RANGE.contains(&offset)
}

fn bit(v: u32, pos: u32) -> bool {
    (v >> pos) & 1 == 1
}

fn field(v: u32, lsb: u32, mask: u32) -> u32 {
    (v >> lsb) & mask
}

fn decode_slot(value: u32) -> SlotConfig {
    SlotConfig {
        valid: bit(value, SLOT_VALID_BIT),
        arbiter: field(value, SLOT_ARBITER_LSB, SLOT_ARBITER_MASK) as u8,
        master_select: field(value, SLOT_MSEL_LSB, SLOT_MSEL_MASK) as u8,
        mask: field(value, SLOT_MASK_LSB, SLOT_MASK_MASK) as u8,
        packet_type: field(value, SLOT_PKT_TYPE_LSB, SLOT_PKT_TYPE_MASK) as u8,
        packet_id: field(value, SLOT_PKT_ID_LSB, SLOT_PKT_ID_MASK) as u8,
        raw: value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(col: u8, row: u8) -> TileAddr {
        TileAddr::new(col, row)
    }

    #[test]
    fn master_packet_mode_bit_decoded() {
        let ops = vec![DeviceOp::RegWrite {
            tile: tile(0, 0),
            offset: 0x3F034, // SHIM.NORTH1
            value: 0xC0000043,
        }];
        let topo = StreamSwitchTopology::from_device_ops(ops);
        let ts = topo.tiles.get(&tile(0, 0)).unwrap();
        let m = ts.masters.get(&0x3F034).unwrap();
        assert!(m.enabled);
        assert!(m.packet_mode);
    }

    #[test]
    fn slave_packet_mode_bit_decoded() {
        let ops = vec![DeviceOp::RegWrite {
            tile: tile(0, 0),
            offset: 0x3F108, // SHIM.SLAVE_SOUTH_0
            value: 0xC0000000,
        }];
        let topo = StreamSwitchTopology::from_device_ops(ops);
        let ts = topo.tiles.get(&tile(0, 0)).unwrap();
        let s = ts.slaves.get(&0x3F108).unwrap();
        assert!(s.enabled);
        assert!(s.packet_mode);
    }

    #[test]
    fn slot_fields_decoded() {
        // A Valid slot with packet_id=5, packet_type=3, master_select=0b010,
        // arbiter=1, mask=0x1F.
        // Value bit layout: [31:Valid=1][28:29:arb=01][24:26:msel=010]
        //                   [16:20:mask=11111][12:14:ptype=011][0:4:pid=00101]
        let v: u32 = (1 << 31) | (0b01 << 28) | (0b010 << 24) | (0b11111 << 16) | (0b011 << 12) | 0b00101;
        let ops = vec![DeviceOp::RegWrite { tile: tile(0, 2), offset: 0x3F200, value: v }];
        let topo = StreamSwitchTopology::from_device_ops(ops);
        let slot = topo.tiles.get(&tile(0, 2)).unwrap().slots.get(&0x3F200).unwrap();
        assert!(slot.valid);
        assert_eq!(slot.arbiter, 1);
        assert_eq!(slot.master_select, 0b010);
        assert_eq!(slot.mask, 0x1F);
        assert_eq!(slot.packet_type, 3);
        assert_eq!(slot.packet_id, 5);
    }

    #[test]
    fn reg_mask_merges_into_prior_value() {
        let ops = vec![
            DeviceOp::RegWrite {
                tile: tile(0, 0),
                offset: 0x3F034,
                value: 0x80000000, // enable only
            },
            DeviceOp::RegMask {
                tile: tile(0, 0),
                offset: 0x3F034,
                mask: 0x40000000, // Packet_Enable bit
                value: 0x40000000,
            },
        ];
        let topo = StreamSwitchTopology::from_device_ops(ops);
        let m = topo.tiles.get(&tile(0, 0)).unwrap().masters.get(&0x3F034).unwrap();
        assert!(m.enabled);
        assert!(m.packet_mode);
    }

    #[test]
    fn non_ss_writes_ignored() {
        let ops = vec![DeviceOp::RegWrite {
            tile: tile(0, 2),
            offset: 0x1D000, // ShimDMA BD word 0 -- not SS
            value: 0xC0000000,
        }];
        let topo = StreamSwitchTopology::from_device_ops(ops);
        assert!(topo.tiles.is_empty());
    }

    #[test]
    fn packet_masters_iterator_filters_non_packet() {
        let ops = vec![
            DeviceOp::RegWrite {
                tile: tile(0, 0),
                offset: 0x3F008,
                value: 0xC0000042, // enable + packet
            },
            DeviceOp::RegWrite {
                tile: tile(0, 0),
                offset: 0x3F00C,
                value: 0x80000001, // enable only, circuit
            },
        ];
        let topo = StreamSwitchTopology::from_device_ops(ops);
        let pkt_offsets: Vec<u32> = topo.packet_masters().map(|(_, off, _)| off).collect();
        assert_eq!(pkt_offsets, vec![0x3F008]);
    }
}
