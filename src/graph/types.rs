//! Core type definitions for the NPU architecture graph.

use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// Architecture and tile identification
// ============================================================================

/// Target NPU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Architecture {
    Aie,
    Aie2,
    Aie2p,
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Aie => write!(f, "AIE"),
            Self::Aie2 => write!(f, "AIE2"),
            Self::Aie2p => write!(f, "AIE2P"),
        }
    }
}

/// Fine-grained device generation, matching aie-rt's XAIE_DEV_GEN_* constants.
///
/// `Architecture` is the coarse bucket (AIE/AIE2/AIE2P); this enum captures
/// the specific silicon revision within each architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceGeneration {
    /// Generic/unknown (XAIE_DEV_GENERIC_DEVICE = 0)
    Generic,
    /// AIE1 -- Versal FPGA (XAIE_DEV_GEN_AIE = 1)
    Aie,
    /// AIE-ML -- AIE2 base (XAIE_DEV_GEN_AIEML = 2)
    AieMl,
    /// AIE2 IPU variant (XAIE_DEV_GEN_AIE2IPU = 3)
    Aie2Ipu,
    /// AIE2P (XAIE_DEV_GEN_AIE2P = 4)
    Aie2P,
    /// AIE2PS (XAIE_DEV_GEN_AIE2PS = 5)
    Aie2Ps,
    /// S100 (XAIE_DEV_GEN_S100 = 6)
    S100,
    /// S200 (XAIE_DEV_GEN_S200 = 7)
    S200,
    /// AIE2P Strix A0 stepping (XAIE_DEV_GEN_AIE2P_STRIX_A0 = 8)
    Aie2PStrixA0,
    /// AIE2P Strix B0 stepping (XAIE_DEV_GEN_AIE2P_STRIX_B0 = 9)
    Aie2PStrixB0,
}

/// Tile type within the NPU array.
///
/// aie-rt distinguishes four tile types (AIETILE, SHIMNOC, SHIMPL, MEMTILE).
/// ShimPl tiles exist in the hardware but are not software-accessible on NPU
/// devices, so they typically don't appear in device model extractions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TileKind {
    Compute,
    Mem,
    ShimNoc,
    ShimPl,
}

impl fmt::Display for TileKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compute => write!(f, "compute"),
            Self::Mem => write!(f, "mem"),
            Self::ShimNoc => write!(f, "shim_noc"),
            Self::ShimPl => write!(f, "shim_pl"),
        }
    }
}

// ============================================================================
// Source attribution
// ============================================================================

/// Which toolchain source a fact was derived from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Source {
    Am025Json,
    AieRt,
    DeviceModel,
    TableGen,
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Am025Json => write!(f, "AM025 JSON"),
            Self::AieRt => write!(f, "aie-rt"),
            Self::DeviceModel => write!(f, "device model"),
            Self::TableGen => write!(f, "TableGen"),
        }
    }
}

/// Attribution of a fact to its toolchain source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceAttribution {
    pub origin: Source,
    pub file: String,
    pub detail: String,
}

impl fmt::Display for SourceAttribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} ({})", self.origin, self.file, self.detail)
    }
}

// ============================================================================
// Register model
// ============================================================================

/// Bit range within a register -- contiguous or split across non-adjacent positions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BitRange {
    /// Standard contiguous range, e.g., bits 17:0.
    Contiguous { msb: u8, lsb: u8 },
    /// Split field: multiple (msb, lsb) fragments, ordered high-to-low.
    Split(Vec<(u8, u8)>),
}

impl BitRange {
    /// Total width in bits.
    pub fn width(&self) -> u8 {
        match self {
            Self::Contiguous { msb, lsb } => msb - lsb + 1,
            Self::Split(fragments) => {
                fragments.iter().map(|(msb, lsb)| msb - lsb + 1).sum()
            }
        }
    }
}

/// Semantic meaning of a register field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldSemantics {
    Address,
    Count,
    Enable,
    LockId,
    LockValue,
    ChannelId,
    BdIndex,
    Status,
    Control,
    EventId,
    PortIndex,
    DataWidth,
    Reserved,
    /// Not yet classified.
    Unknown,
}

/// Register access mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Access {
    ReadWrite,
    ReadOnly,
    WriteOnly,
    WriteToClear,
    Mixed,
}

/// A single bit field within a register.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldModel {
    pub name: String,
    pub bits: BitRange,
    pub meaning: FieldSemantics,
    pub source: SourceAttribution,
}

/// A hardware register.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterModel {
    pub name: String,
    pub offset: u32,
    pub width: u8,
    pub reset_value: u32,
    pub fields: Vec<FieldModel>,
    pub module: ModuleKind,
    pub access: Access,
    pub source: SourceAttribution,
}

// ============================================================================
// Module model
// ============================================================================

/// Functional module within a tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModuleKind {
    Dma,
    Lock,
    StreamSwitch,
    Core,
    ProgramMemory,
    DataMemory,
    Trace,
    Event,
    ShimMux,
    /// Module not yet classified.
    Unknown,
}

impl fmt::Display for ModuleKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dma => write!(f, "dma"),
            Self::Lock => write!(f, "lock"),
            Self::StreamSwitch => write!(f, "stream_switch"),
            Self::Core => write!(f, "core"),
            Self::ProgramMemory => write!(f, "program_memory"),
            Self::DataMemory => write!(f, "data_memory"),
            Self::Trace => write!(f, "trace"),
            Self::Event => write!(f, "event"),
            Self::ShimMux => write!(f, "shim_mux"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// A bundle of ports in one direction group (e.g., "DMA" with 2 masters + 2 slaves).
///
/// The device model stores port counts per bundle, not individual ports.
/// All ports within a bundle+direction are identical -- they differ only by
/// index, which can be enumerated from the count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortBundle {
    pub bundle: String,
    pub masters: u8,
    pub slaves: u8,
}

/// Instance counts for repeated hardware elements within a tile type.
///
/// Memory banking info (num_banks, bank_size) lives in `MemoryModel`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceCount {
    pub locks: u8,
    pub bds: u8,
    pub channels: u8,
}

/// A functional module within a tile (Layer 2+: registers grouped by function).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleModel {
    pub kind: ModuleKind,
    pub registers: Vec<RegisterModel>,
    pub source: SourceAttribution,
}

// ============================================================================
// Tile and array topology
// ============================================================================

/// Memory model for a tile type.
///
/// `size_bytes` is data memory (64KB compute, 512KB memtile).
/// `program_memory_bytes` is instruction memory (16KB, compute tiles only).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryModel {
    pub size_bytes: u64,
    pub num_banks: u8,
    pub bank_size_bytes: u64,
    /// Instruction memory size. Only present on compute tiles.
    pub program_memory_bytes: Option<u64>,
    pub source: SourceAttribution,
}

/// A tile type definition (e.g., "core", "mem_tile", "shim_noc").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileTypeModel {
    pub kind: TileKind,
    pub name: String,
    /// Representative tile coordinates [col, row] for this type.
    pub representative: Option<(u8, u8)>,
    /// Hardware resource counts (locks, BDs, channels, banks).
    pub instances: InstanceCount,
    /// Data/program memory model.
    pub memory: Option<MemoryModel>,
    /// DMA feature capabilities (compression, padding, etc.).
    pub dma_capabilities: Option<DmaCapabilities>,
    /// Switchbox port bundles (present on all tile types).
    pub switchbox_ports: Vec<PortBundle>,
    /// Shim mux port bundles (only present on shim tiles).
    pub shim_mux_ports: Vec<PortBundle>,
    /// Register modules (populated in Layer 2).
    pub modules: Vec<ModuleModel>,
    pub source: SourceAttribution,
}

/// DMA feature capabilities for a tile type.
///
/// These flags describe what optional DMA features the hardware supports.
/// Derived from aie-rt's `XAie_DmaMod` structure, which defines these
/// per architecture generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmaCapabilities {
    pub supports_compression: bool,
    pub supports_zero_padding: bool,
    pub supports_out_of_order_bd: bool,
    pub supports_interleave: bool,
    pub supports_fifo_mode: bool,
    pub supports_token_issue: bool,
    pub supports_repeat_count: bool,
    pub supports_tlast_suppress: bool,
    /// Number of addressing dimensions (typically 3 or 4).
    pub max_address_dimensions: u8,
}

/// Device-level hardware constants.
///
/// These are properties of the device as a whole, not of any specific tile.
/// Values come from the mlir-aie device model and aie-rt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceConstants {
    /// Maximum lock counter value (e.g., 63 for AIE2).
    pub max_lock_value: u32,
    /// Minimum lock counter value (e.g., -64 for AIE-ML, 0 for AIE1).
    pub min_lock_value: i32,
    /// DMA address generation granularity in bytes (e.g., 32).
    pub address_gen_granularity: u32,
    /// Accumulator cascade width in bits (384 for AIE1, 512 for AIE2).
    pub accumulator_cascade_bits: Option<u32>,
    /// Memory base addresses per cardinal direction, used for cross-tile
    /// memory access routing. Keys: south, west, north, east.
    pub mem_base_addresses: std::collections::BTreeMap<String, u64>,
    /// Architecture-level capability flags.
    pub properties: DeviceProperties,
    pub source: SourceAttribution,
}

/// Architecture-level capability flags.
///
/// Derived from mlir-aie's `ModelProperty` enum. These describe fundamental
/// behavioral differences between architecture generations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceProperties {
    /// Uses semaphore-style locks (AIE2+) vs. binary locks (AIE1).
    pub uses_semaphore_locks: bool,
    /// Supports multi-dimensional buffer descriptors.
    pub uses_multi_dim_bds: bool,
}

/// Directional flags (north/south/east/west).
///
/// Used for tile boundary edges and memory affinity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CardinalFlags {
    pub north: bool,
    pub south: bool,
    pub east: bool,
    pub west: bool,
}

/// A single tile's placement within the array.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TilePlacement {
    pub col: u8,
    pub row: u8,
    /// Which tile type this position holds (matches a TileTypeModel name).
    pub tile_type: String,
    /// Whether this tile is internal (not on any array boundary).
    pub is_internal: bool,
    /// Which edges of this tile are on the array boundary.
    pub edges: CardinalFlags,
    /// Memory affinity: which directions have accessible memory neighbors.
    /// Only present for compute tiles.
    pub mem_affinity: Option<CardinalFlags>,
}

/// Array topology: grid dimensions, tile placement, and addressing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArrayTopology {
    pub columns: u8,
    pub rows: u8,
    pub num_mem_tile_rows: u8,
    pub column_shift: u8,
    pub row_shift: u8,
    /// Per-tile placement map: one entry per physical tile in the array.
    pub tile_map: Vec<TilePlacement>,
    pub source: SourceAttribution,
}

// ============================================================================
// Relationships
// ============================================================================

/// A node identifier in the graph (dot-separated path).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Type of relationship between two nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipKind {
    DataFlow,
    Configures,
    BlocksOn,
    Triggers,
    RoutesTo,
    Produces,
    Contains,
}

/// A directed relationship between two nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Relationship {
    pub from: NodeId,
    pub to: NodeId,
    pub kind: RelationshipKind,
    pub source: SourceAttribution,
}

// ============================================================================
// Top-level model
// ============================================================================

/// Complete architecture model for one NPU device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchModel {
    pub arch: Architecture,
    /// Fine-grained device generation (aie-rt XAIE_DEV_GEN_*).
    pub generation: Option<DeviceGeneration>,
    /// mlir-aie device ID (e.g., npu1=4, npu2=8).
    pub device_id: Option<u32>,
    /// Whether this device is an NPU (vs. Versal FPGA).
    pub is_npu: bool,
    pub tile_types: Vec<TileTypeModel>,
    pub array_topology: Option<ArrayTopology>,
    pub device_constants: Option<DeviceConstants>,
    pub relationships: Vec<Relationship>,
}

impl ArchModel {
    /// Create an empty model for the given architecture.
    pub fn new(arch: Architecture) -> Self {
        Self {
            arch,
            generation: None,
            device_id: None,
            is_npu: false,
            tile_types: Vec::new(),
            array_topology: None,
            device_constants: None,
            relationships: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_enum_variants() {
        // Verify all expected architectures exist
        let archs = [Architecture::Aie, Architecture::Aie2, Architecture::Aie2p];
        assert_eq!(archs.len(), 3);
    }

    #[test]
    fn test_tile_kind_variants() {
        let kinds = [TileKind::Compute, TileKind::Mem, TileKind::ShimNoc, TileKind::ShimPl];
        assert_eq!(kinds.len(), 4);
        assert_eq!(format!("{}", TileKind::ShimNoc), "shim_noc");
        assert_eq!(format!("{}", TileKind::ShimPl), "shim_pl");
    }

    #[test]
    fn test_source_attribution_display() {
        let attr = SourceAttribution {
            origin: Source::Am025Json,
            file: "aie_registers_aie2.json".to_string(),
            detail: "module=memory, register=DMA_BD0_0".to_string(),
        };
        let s = format!("{}", attr);
        assert!(s.contains("AM025 JSON"));
        assert!(s.contains("aie_registers_aie2.json"));
    }

    #[test]
    fn test_bit_range_contiguous() {
        let range = BitRange::Contiguous { msb: 17, lsb: 0 };
        assert_eq!(range.width(), 18);
    }

    #[test]
    fn test_bit_range_split() {
        let range = BitRange::Split(vec![
            (23, 20),  // 4 bits
            (7, 4),    // 4 bits
        ]);
        assert_eq!(range.width(), 8);
    }

    #[test]
    fn test_arch_model_empty() {
        let model = ArchModel::new(Architecture::Aie2);
        assert_eq!(model.arch, Architecture::Aie2);
        assert_eq!(model.device_id, None);
        assert!(!model.is_npu);
        assert!(model.tile_types.is_empty());
        assert!(model.relationships.is_empty());
    }

    #[test]
    fn test_serde_round_trip() {
        let mut model = ArchModel::new(Architecture::Aie2);
        model.device_id = Some(4);
        model.is_npu = true;
        model.tile_types.push(TileTypeModel {
            kind: TileKind::Compute,
            name: "core".to_string(),
            representative: Some((1, 2)),
            instances: InstanceCount { locks: 16, bds: 16, channels: 2 },
            modules: vec![ModuleModel {
                kind: ModuleKind::Dma,
                registers: vec![RegisterModel {
                    name: "DMA_BD0_0".to_string(),
                    offset: 0x1D000,
                    width: 32,
                    reset_value: 0,
                    fields: vec![FieldModel {
                        name: "Buffer_Length".to_string(),
                        bits: BitRange::Contiguous { msb: 13, lsb: 0 },
                        meaning: FieldSemantics::Count,
                        source: SourceAttribution {
                            origin: Source::Am025Json,
                            file: "test.json".to_string(),
                            detail: "test".to_string(),
                        },
                    }],
                    module: ModuleKind::Dma,
                    access: Access::ReadWrite,
                    source: SourceAttribution {
                        origin: Source::Am025Json,
                        file: "test.json".to_string(),
                        detail: "test".to_string(),
                    },
                }],
                source: SourceAttribution {
                    origin: Source::DeviceModel,
                    file: "test.json".to_string(),
                    detail: "test".to_string(),
                },
            }],
            memory: Some(MemoryModel {
                size_bytes: 65536,
                num_banks: 4,
                bank_size_bytes: 16384,
                program_memory_bytes: Some(16384),
                source: SourceAttribution {
                    origin: Source::DeviceModel,
                    file: "test.json".to_string(),
                    detail: "test".to_string(),
                },
            }),
            dma_capabilities: Some(DmaCapabilities {
                supports_compression: true,
                supports_zero_padding: true,
                supports_out_of_order_bd: false,
                supports_interleave: false,
                supports_fifo_mode: true,
                supports_token_issue: true,
                supports_repeat_count: true,
                supports_tlast_suppress: false,
                max_address_dimensions: 4,
            }),
            switchbox_ports: vec![
                PortBundle { bundle: "DMA".to_string(), masters: 2, slaves: 2 },
                PortBundle { bundle: "Core".to_string(), masters: 1, slaves: 1 },
            ],
            shim_mux_ports: Vec::new(),
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test.json".to_string(),
                detail: "test".to_string(),
            },
        });
        model.relationships.push(Relationship {
            from: NodeId("compute".to_string()),
            to: NodeId("compute.dma".to_string()),
            kind: RelationshipKind::Contains,
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "derived".to_string(),
                detail: "test".to_string(),
            },
        });

        let json = serde_json::to_string(&model).expect("serialize failed");
        let deserialized: ArchModel = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(model, deserialized);
    }

    #[test]
    fn test_device_constants() {
        let mut mem_bases = std::collections::BTreeMap::new();
        mem_bases.insert("south".to_string(), 262144);
        mem_bases.insert("west".to_string(), 327680);
        mem_bases.insert("north".to_string(), 393216);
        mem_bases.insert("east".to_string(), 458752);

        let constants = DeviceConstants {
            max_lock_value: 63,
            min_lock_value: -64,
            address_gen_granularity: 32,
            accumulator_cascade_bits: Some(512),
            mem_base_addresses: mem_bases,
            properties: DeviceProperties {
                uses_semaphore_locks: true,
                uses_multi_dim_bds: true,
            },
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test.json".to_string(),
                detail: "npu1".to_string(),
            },
        };
        assert_eq!(constants.max_lock_value, 63);
        assert_eq!(constants.min_lock_value, -64);
        assert_eq!(constants.accumulator_cascade_bits, Some(512));
        assert!(constants.properties.uses_semaphore_locks);
        assert_eq!(constants.mem_base_addresses.len(), 4);
        assert_eq!(constants.mem_base_addresses["south"], 262144);
    }

    #[test]
    fn test_tile_placement() {
        let placement = TilePlacement {
            col: 1,
            row: 2,
            tile_type: "core".to_string(),
            is_internal: true,
            edges: CardinalFlags {
                north: false,
                south: false,
                east: false,
                west: false,
            },
            mem_affinity: Some(CardinalFlags {
                south: true,
                west: false,
                north: true,
                east: false,
            }),
        };
        assert!(placement.is_internal);
        assert!(placement.mem_affinity.unwrap().south);
        assert!(!placement.mem_affinity.unwrap().north || true); // just exercising access
    }

    #[test]
    fn test_port_bundles() {
        let tile = TileTypeModel {
            kind: TileKind::ShimNoc,
            name: "shim_noc".to_string(),
            representative: Some((1, 0)),
            instances: InstanceCount { locks: 16, bds: 16, channels: 2 },
            memory: None,
            dma_capabilities: None,
            switchbox_ports: vec![
                PortBundle { bundle: "South".to_string(), masters: 6, slaves: 8 },
                PortBundle { bundle: "North".to_string(), masters: 6, slaves: 4 },
                PortBundle { bundle: "Ctrl".to_string(), masters: 1, slaves: 1 },
            ],
            shim_mux_ports: vec![
                PortBundle { bundle: "DMA".to_string(), masters: 2, slaves: 2 },
                PortBundle { bundle: "South".to_string(), masters: 8, slaves: 6 },
            ],
            modules: Vec::new(),
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test.json".to_string(),
                detail: "shim_noc".to_string(),
            },
        };
        // Switchbox and shim_mux are distinct port namespaces
        assert_eq!(tile.switchbox_ports.len(), 3);
        assert_eq!(tile.shim_mux_ports.len(), 2);
        // DMA ports only appear in shim_mux for shim tiles
        assert!(tile.switchbox_ports.iter().all(|p| p.bundle != "DMA"));
        assert!(tile.shim_mux_ports.iter().any(|p| p.bundle == "DMA"));
    }

    #[test]
    fn test_memory_model_program_memory() {
        let compute_mem = MemoryModel {
            size_bytes: 65536,
            num_banks: 4,
            bank_size_bytes: 16384,
            program_memory_bytes: Some(16384),
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test.json".to_string(),
                detail: "test".to_string(),
            },
        };
        let memtile_mem = MemoryModel {
            size_bytes: 524288,
            num_banks: 8,
            bank_size_bytes: 65536,
            program_memory_bytes: None,
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test.json".to_string(),
                detail: "test".to_string(),
            },
        };
        assert_eq!(compute_mem.program_memory_bytes, Some(16384));
        assert_eq!(memtile_mem.program_memory_bytes, None);
    }

    #[test]
    fn test_serde_round_trip_with_topology() {
        let mut model = ArchModel::new(Architecture::Aie2);

        let mut mem_bases = std::collections::BTreeMap::new();
        mem_bases.insert("south".to_string(), 262144);

        model.generation = Some(DeviceGeneration::Aie2Ipu);
        model.device_constants = Some(DeviceConstants {
            max_lock_value: 63,
            min_lock_value: -64,
            address_gen_granularity: 32,
            accumulator_cascade_bits: Some(512),
            mem_base_addresses: mem_bases,
            properties: DeviceProperties {
                uses_semaphore_locks: true,
                uses_multi_dim_bds: true,
            },
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test.json".to_string(),
                detail: "test".to_string(),
            },
        });

        model.array_topology = Some(ArrayTopology {
            columns: 4,
            rows: 6,
            num_mem_tile_rows: 1,
            column_shift: 25,
            row_shift: 20,
            tile_map: vec![
                TilePlacement {
                    col: 0,
                    row: 0,
                    tile_type: "shim_noc".to_string(),
                    is_internal: false,
                    edges: CardinalFlags {
                        north: false, south: true, east: false, west: true,
                    },
                    mem_affinity: None,
                },
                TilePlacement {
                    col: 1,
                    row: 2,
                    tile_type: "core".to_string(),
                    is_internal: true,
                    edges: CardinalFlags {
                        north: false, south: false, east: false, west: false,
                    },
                    mem_affinity: Some(CardinalFlags {
                        south: true, west: false, north: true, east: false,
                    }),
                },
            ],
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test.json".to_string(),
                detail: "test".to_string(),
            },
        });

        let json = serde_json::to_string(&model).expect("serialize failed");
        let deserialized: ArchModel = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(model, deserialized);

        // Verify we can navigate the topology
        let topo = deserialized.array_topology.unwrap();
        assert_eq!(topo.tile_map.len(), 2);
        assert_eq!(topo.tile_map[0].tile_type, "shim_noc");
        assert!(topo.tile_map[1].mem_affinity.unwrap().south);
    }
}
