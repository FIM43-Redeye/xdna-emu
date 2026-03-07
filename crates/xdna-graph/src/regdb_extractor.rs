//! Register database extractor for the architecture graph.
//!
//! Consumes the runtime `regdb::RegisterDb` (already parsed from AM025 JSON)
//! and lifts register definitions into graph-level types with pattern
//! recognition and behavioral classification.
//!
//! The four scheduling-relevant register categories:
//! - **State**: Readable resource state (Lock{N}_value, DMA_BD{N}_{M})
//! - **Operations**: Interfaces that trigger actions (Lock_Request)
//! - **Events**: Cross-subsystem trigger configuration (Event_Selection)
//! - **Status**: Exceptional condition tracking (Overflow, Underflow)

use crate::regdb;
use crate::types::{
    Access, BdFieldRole, BdFieldSpec, BdSchema, BitRange, DmaChannelFieldRole,
    DmaChannelFieldSpec, DmaChannelRegKind, DmaChannelSchema, DmaDirection, FieldModel,
    FieldSemantics, ModuleKind, ModuleModel, RegisterModel, Relationship, Source,
    SourceAttribution, SubsystemKind, TileKind,
};

// ============================================================================
// Register classification by hardware properties
// ============================================================================

/// Scheduling-relevant register category, derived from hardware properties.
///
/// Classification is based on access mode and structural patterns, not
/// register names. This keeps the classification valid across architecture
/// revisions where naming conventions may change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisterCategory {
    /// Readable/writable resource state. The register holds a value that
    /// software and hardware both care about (lock counts, BD fields).
    /// Signal: R/W access + part of a numbered group (per-instance state).
    State,
    /// Interface that triggers a hardware action. The register is read
    /// to get a result, or written to initiate an operation.
    /// Signal: ReadOnly access (hardware produces the value).
    Operations,
    /// Exceptional condition tracking. Hardware sets bits to indicate
    /// errors or saturation; software clears them to acknowledge.
    /// Signal: WriteToClear access, or R/W with per-instance status bits
    /// named Overflow/Underflow (the only name hint we use).
    Status,
    /// Not classifiable from properties alone. Needs additional context
    /// (cross-subsystem analysis, name-based refinement, or aie-rt data).
    Unclassified,
}

/// Classify a register based on its hardware properties.
///
/// Uses access mode as the primary signal, structural context (whether
/// the register is part of a numbered group) as secondary, and two
/// universal name patterns (Overflow/Underflow) as tertiary.
pub fn classify_register(reg: &regdb::RegisterDef, is_grouped: bool) -> RegisterCategory {
    // Priority 1: Access mode is the strongest structural signal.
    match reg.access {
        regdb::AccessMode::ReadOnly => return RegisterCategory::Operations,
        regdb::AccessMode::WriteToClear => return RegisterCategory::Status,
        _ => {}
    }

    // Priority 2: Universal name patterns -- Overflow/Underflow are consistent
    // across all modules and architecture revisions.
    let name_lower = reg.name.to_ascii_lowercase();
    if name_lower.contains("overflow") || name_lower.contains("underflow") {
        return RegisterCategory::Status;
    }

    // Priority 3: Structural context -- grouped R/W registers are per-instance
    // state (lock values, BD fields, event selectors).
    if is_grouped && reg.access == regdb::AccessMode::ReadWrite {
        return RegisterCategory::State;
    }

    RegisterCategory::Unclassified
}

// ============================================================================
// Subsystem profile: behavioral properties derived from register structure
// ============================================================================

/// Behavioral profile of a subsystem, derived entirely from register analysis.
///
/// This captures what a subsystem CAN DO, not just what registers it has.
/// All fields are extracted from register properties -- nothing is hardcoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubsystemProfile {
    /// Which subsystem this profile describes (e.g., "DMA", "Lock", "Event").
    pub subsystem: String,
    /// Typed subsystem kind, mapped from the discovered name.
    pub kind: SubsystemKind,
    /// Number of independent instances (derived from State register group count).
    /// e.g., 16 locks in memory module, 64 in memtile.
    pub instance_count: u32,
    /// Categorized register groups: (category, group).
    pub register_groups: Vec<(RegisterCategory, RegisterGroup)>,
    /// Singleton registers that don't form groups: (category, register name).
    pub singletons: Vec<(RegisterCategory, String)>,
}

/// Build a subsystem profile from a filtered set of registers.
///
/// Given all registers belonging to a single subsystem (e.g., all lock
/// registers in a module), groups them, classifies each, and derives
/// behavioral properties.
pub fn build_subsystem_profile(
    subsystem: &str,
    registers: &[regdb::RegisterDef],
) -> SubsystemProfile {
    let groups = group_registers(registers);

    // Build a set of register names that belong to groups, so we can
    // identify singletons (registers not part of any group).
    let grouped_patterns: std::collections::HashSet<String> = registers
        .iter()
        .filter(|r| {
            let pat = name_to_pattern(&r.name);
            pat != r.name && groups.iter().any(|g| g.pattern == pat)
        })
        .map(|r| r.name.clone())
        .collect();

    // Classify each group
    let classified_groups: Vec<(RegisterCategory, RegisterGroup)> = groups
        .into_iter()
        .map(|g| {
            // Use the first register's properties for classification
            let representative = registers
                .iter()
                .find(|r| {
                    let pat = name_to_pattern(&r.name);
                    pat == g.pattern
                })
                .unwrap();
            let category = classify_register(representative, true);
            (category, g)
        })
        .collect();

    // Classify singletons (registers not in any group)
    let singletons: Vec<(RegisterCategory, String)> = registers
        .iter()
        .filter(|r| !grouped_patterns.contains(&r.name))
        .map(|r| (classify_register(r, false), r.name.clone()))
        .collect();

    // Instance count: the largest State group's instance count, or 0.
    // For multi-dimensional groups, instance_count() returns the outermost
    // dimension (e.g., 16 BDs, not 96 BD-words).
    let instance_count = classified_groups
        .iter()
        .filter(|(cat, _)| *cat == RegisterCategory::State)
        .map(|(_, g)| g.instance_count())
        .max()
        .unwrap_or(0);

    SubsystemProfile {
        subsystem: subsystem.to_string(),
        kind: subsystem_name_to_kind(subsystem),
        instance_count,
        register_groups: classified_groups,
        singletons,
    }
}

// ============================================================================
// Register grouping: recognize numbered patterns
// ============================================================================

/// One dimension of a multi-dimensional register group.
///
/// Register groups can be 1D (Lock0..Lock15) or multi-dimensional
/// (DMA_BD0_0..DMA_BD15_5 = 16 BDs x 6 words). Each dimension
/// captures the count and stride along one axis.
///
/// Dimensions are ordered outer-to-inner: for DMA_BD{N}_{M},
/// dimensions[0] is the BD dimension (N), dimensions[1] is the
/// word dimension (M).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexDimension {
    /// Number of instances along this dimension.
    pub count: u32,
    /// Byte distance between consecutive instances.
    pub stride: u32,
}

/// A group of registers sharing a common name pattern with varying indices.
///
/// For single-index patterns like Lock0_value..Lock15_value:
///   dimensions = [{ count: 16, stride: 0x10 }]
///
/// For multi-index patterns like DMA_BD0_0..DMA_BD15_5:
///   dimensions = [{ count: 16, stride: 0x20 }, { count: 6, stride: 0x4 }]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterGroup {
    /// Pattern name with indices replaced by `{}`, e.g., "Lock{}_value".
    pub pattern: String,
    /// Offset of the first register.
    pub base_offset: u32,
    /// Index dimensions, ordered outer-to-inner.
    pub dimensions: Vec<IndexDimension>,
    /// Field names shared by all registers in this group.
    pub field_names: Vec<String>,
}

impl RegisterGroup {
    /// Total number of registers in this group (product of all dimensions).
    pub fn total_count(&self) -> u32 {
        self.dimensions.iter().map(|d| d.count).product()
    }

    /// Number of top-level instances (outermost dimension count).
    /// For 1D groups, this equals total_count.
    /// For 2D groups like DMA BDs, this is the BD count (not BD x words).
    pub fn instance_count(&self) -> u32 {
        self.dimensions.first().map_or(0, |d| d.count)
    }
}

/// Replace all digit runs in a name with `{}` to get a grouping pattern.
///
/// Examples:
/// - "Lock15_value" -> "Lock{}_value"
/// - "Locks_Event_Selection_7" -> "Locks_Event_Selection_{}"
/// - "Lock_Request" -> "Lock_Request" (no digits, unchanged)
fn name_to_pattern(name: &str) -> String {
    let mut result = String::with_capacity(name.len());
    let mut in_digits = false;
    for ch in name.chars() {
        if ch.is_ascii_digit() {
            if !in_digits {
                result.push_str("{}");
                in_digits = true;
            }
            // skip subsequent digits
        } else {
            in_digits = false;
            result.push(ch);
        }
    }
    result
}

/// Identify groups of numbered registers in a list.
///
/// Scans register names for embedded numeric indices, groups registers
/// sharing the same base pattern, and computes the count and stride
/// for each group. Singleton registers (no numeric index or only one
/// instance) are not included.
///
/// Multi-dimensional groups are detected from offset arithmetic: if
/// registers form a regular 2D grid (e.g., 16 BDs x 6 words), the
/// dimensions are decomposed into outer (BD) and inner (word) axes.
pub fn group_registers(registers: &[regdb::RegisterDef]) -> Vec<RegisterGroup> {
    use std::collections::BTreeMap;

    // Group registers by their pattern. BTreeMap for deterministic order.
    let mut by_pattern: BTreeMap<String, Vec<&regdb::RegisterDef>> = BTreeMap::new();

    for reg in registers {
        let pattern = name_to_pattern(&reg.name);
        // Only consider registers whose pattern differs from name (has digits)
        if pattern != reg.name {
            by_pattern.entry(pattern).or_default().push(reg);
        }
    }

    let mut groups = Vec::new();
    for (pattern, mut regs) in by_pattern {
        if regs.len() < 2 {
            continue; // singletons don't form groups
        }

        // Sort by offset to compute stride
        regs.sort_by_key(|r| r.offset);

        let base_offset = regs[0].offset;
        let offsets: Vec<u32> = regs.iter().map(|r| r.offset).collect();

        // Detect dimensionality from offset structure and name indices
        let dimensions = detect_dimensions(&regs, &offsets, &pattern);

        // Collect field names from the first register (shared across group)
        let field_names: Vec<String> = regs[0].fields.iter().map(|f| f.name.clone()).collect();

        groups.push(RegisterGroup {
            pattern,
            base_offset,
            dimensions,
            field_names,
        });
    }

    groups
}

/// Extract numeric index values from a register name given its pattern.
///
/// Walks the name and pattern in parallel, collecting digits at each `{}`
/// position. Returns None if the name doesn't match the pattern structure.
///
/// Example: extract_indices("DMA_BD15_5", "DMA_BD{}_{}") -> Some([15, 5])
fn extract_indices(name: &str, pattern: &str) -> Option<Vec<u32>> {
    let mut indices = Vec::new();
    let mut name_chars = name.chars().peekable();
    let mut pat_chars = pattern.chars().peekable();

    while let Some(&pc) = pat_chars.peek() {
        if pc == '{' {
            // Consume "{}"
            pat_chars.next(); // '{'
            if pat_chars.next() != Some('}') {
                return None;
            }
            // Collect digits from name
            let mut digits = String::new();
            while let Some(&nc) = name_chars.peek() {
                if nc.is_ascii_digit() {
                    digits.push(nc);
                    name_chars.next();
                } else {
                    break;
                }
            }
            if digits.is_empty() {
                return None;
            }
            indices.push(digits.parse::<u32>().ok()?);
        } else {
            // Literal character must match
            pat_chars.next();
            if name_chars.next() != Some(pc) {
                return None;
            }
        }
    }

    // Both should be exhausted
    if name_chars.next().is_some() {
        return None;
    }

    Some(indices)
}

/// Count the number of `{}` placeholders in a pattern.
fn placeholder_count(pattern: &str) -> usize {
    pattern.matches("{}").count()
}

/// Detect index dimensions from register data.
///
/// Uses two strategies:
/// 1. **Stride-break detection** (offset-only): works when inner groups
///    don't fill the entire outer stride (gaps between groups).
/// 2. **Index extraction** (name-based): works for perfectly packed groups
///    where all offsets are uniformly spaced. Uses the number of `{}`
///    placeholders in the pattern to know dimensionality, then extracts
///    actual index values to determine group sizes.
fn detect_dimensions(
    regs: &[&regdb::RegisterDef],
    offsets: &[u32],
    pattern: &str,
) -> Vec<IndexDimension> {
    if offsets.len() < 2 {
        return vec![IndexDimension {
            count: offsets.len() as u32,
            stride: 0,
        }];
    }

    // Compute consecutive deltas
    let deltas: Vec<u32> = offsets.windows(2).map(|w| w[1] - w[0]).collect();
    let inner_stride = *deltas.iter().min().unwrap();

    // Check if all deltas are the same (uniform spacing)
    let uniform = deltas.iter().all(|&d| d == inner_stride);

    if !uniform {
        // Strategy 1: stride-break detection
        if let Some(dims) = detect_dimensions_by_stride_break(offsets, &deltas, inner_stride) {
            return dims;
        }
    }

    // Strategy 2: if pattern has multiple placeholders and offsets are
    // uniform, extract indices from names to find structure.
    let n_placeholders = placeholder_count(pattern);
    if n_placeholders >= 2 {
        if let Some(dims) = detect_dimensions_by_index_extraction(regs, offsets, pattern) {
            return dims;
        }
    }

    // Fallback: flat 1D
    vec![IndexDimension {
        count: offsets.len() as u32,
        stride: inner_stride,
    }]
}

/// Detect 2D structure from stride breaks in the offset sequence.
fn detect_dimensions_by_stride_break(
    offsets: &[u32],
    deltas: &[u32],
    inner_stride: u32,
) -> Option<Vec<IndexDimension>> {
    // Count consecutive inner_stride deltas before each break
    let mut inner_counts = Vec::new();
    let mut run = 1u32;
    for &d in deltas {
        if d == inner_stride {
            run += 1;
        } else {
            inner_counts.push(run);
            run = 1;
        }
    }
    inner_counts.push(run);

    // All inner groups must have the same size
    let inner_count = inner_counts[0];
    if inner_count < 2 || !inner_counts.iter().all(|&c| c == inner_count) {
        return None;
    }

    let outer_stride = offsets[inner_count as usize] - offsets[0];
    let outer_count = inner_counts.len() as u32;

    debug_assert_eq!(outer_count * inner_count, offsets.len() as u32);

    Some(vec![
        IndexDimension {
            count: outer_count,
            stride: outer_stride,
        },
        IndexDimension {
            count: inner_count,
            stride: inner_stride,
        },
    ])
}

/// Detect 2D structure by extracting actual index values from register names.
///
/// Groups registers by their first index. If all groups have the same size,
/// we have a valid 2D decomposition.
fn detect_dimensions_by_index_extraction(
    regs: &[&regdb::RegisterDef],
    offsets: &[u32],
    pattern: &str,
) -> Option<Vec<IndexDimension>> {
    use std::collections::BTreeMap;

    // Extract first index from each register
    let mut by_first_index: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for reg in regs {
        let indices = extract_indices(&reg.name, pattern)?;
        if indices.len() < 2 {
            return None;
        }
        by_first_index
            .entry(indices[0])
            .or_default()
            .push(reg.offset);
    }

    let groups: Vec<&Vec<u32>> = by_first_index.values().collect();
    if groups.len() < 2 {
        return None;
    }

    // All inner groups must have the same size
    let inner_count = groups[0].len() as u32;
    if !groups.iter().all(|g| g.len() as u32 == inner_count) {
        return None;
    }

    let outer_count = groups.len() as u32;
    let inner_stride = if inner_count >= 2 {
        offsets[1] - offsets[0]
    } else {
        0
    };
    let outer_stride = if outer_count >= 2 {
        *groups[1].first().unwrap() - *groups[0].first().unwrap()
    } else {
        0
    };

    Some(vec![
        IndexDimension {
            count: outer_count,
            stride: outer_stride,
        },
        IndexDimension {
            count: inner_count,
            stride: inner_stride,
        },
    ])
}

// ============================================================================
// Register-to-graph conversion: populate typed RegisterModel from regdb
// ============================================================================

/// Convert a regdb access mode to the graph type system's Access enum.
fn convert_access(mode: regdb::AccessMode) -> Access {
    match mode {
        regdb::AccessMode::ReadWrite => Access::ReadWrite,
        regdb::AccessMode::ReadOnly => Access::ReadOnly,
        regdb::AccessMode::WriteOnly => Access::WriteOnly,
        regdb::AccessMode::WriteToClear => Access::WriteToClear,
        regdb::AccessMode::Mixed => Access::Mixed,
    }
}

/// Convert a single regdb register definition to a typed RegisterModel.
///
/// The `module_name` parameter is the AM025 module name (e.g., "memory",
/// "core") used for source attribution.
pub fn register_def_to_model(reg: &regdb::RegisterDef, module_name: &str) -> RegisterModel {
    let subsystem = subsystem_name_to_kind(subsystem_key(&reg.name));

    let fields: Vec<FieldModel> = reg
        .fields
        .iter()
        .map(|f| FieldModel {
            name: f.name.clone(),
            bits: BitRange::Contiguous {
                msb: f.msb,
                lsb: f.lsb,
            },
            meaning: FieldSemantics::Unknown,
            source: SourceAttribution {
                origin: Source::Am025Json,
                file: "aie_registers_aie2.json".into(),
                detail: format!("{}.{}.{}", module_name, reg.name, f.name),
            },
        })
        .collect();

    RegisterModel {
        name: reg.name.clone(),
        offset: reg.offset,
        width: reg.width as u8,
        reset_value: reg.reset_value,
        fields,
        subsystem,
        access: convert_access(reg.access),
        source: SourceAttribution {
            origin: Source::Am025Json,
            file: "aie_registers_aie2.json".into(),
            detail: format!("{}.{}", module_name, reg.name),
        },
    }
}

/// Build a complete ModuleModel from a regdb module definition.
///
/// Converts every register in the AM025 module to a typed RegisterModel
/// with subsystem classification derived from register names.
pub fn build_module_model(kind: ModuleKind, module: &regdb::ModuleDef) -> ModuleModel {
    let registers: Vec<RegisterModel> = module
        .registers
        .iter()
        .map(|reg| register_def_to_model(reg, &module.name))
        .collect();

    ModuleModel {
        kind,
        registers,
        source: SourceAttribution {
            origin: Source::Am025Json,
            file: "aie_registers_aie2.json".into(),
            detail: format!("module={}", module.name),
        },
    }
}

// ============================================================================
// Populate ArchModel with register data from AM025
// ============================================================================

use crate::types::ArchModel;

/// Tile type name -> (ModuleKind, AM025 module name) pairs.
const TILE_MODULE_MAP: &[(&str, &[(ModuleKind, &str)])] = &[
    ("core", &[(ModuleKind::Core, "core"), (ModuleKind::Memory, "memory")]),
    ("mem_tile", &[(ModuleKind::MemTile, "memory_tile")]),
    ("shim_noc", &[(ModuleKind::Shim, "shim")]),
    ("shim_pl", &[(ModuleKind::Shim, "shim")]),
];

/// Populate the `modules` field on each `TileTypeModel` in an `ArchModel`
/// using register definitions from the AM025 register database.
///
/// Also confirms `instances.locks` and `instances.bds` from register group
/// analysis -- the AM025 register patterns independently reveal how many
/// locks and BDs each module has.
pub fn populate_tile_modules(model: &mut ArchModel, db: &regdb::RegisterDb) {
    use crate::types::{NodeId, Relationship, RelationshipKind};

    let mut new_edges: Vec<Relationship> = Vec::new();

    for tile_type in &mut model.tile_types {
        let mapping = TILE_MODULE_MAP
            .iter()
            .find(|(name, _)| *name == tile_type.name);

        if let Some((_, module_pairs)) = mapping {
            for &(kind, am025_name) in *module_pairs {
                if let Some(module_def) = db.module(am025_name) {
                    tile_type.modules.push(build_module_model(kind, module_def));

                    let edge_source = SourceAttribution {
                        origin: Source::Am025Json,
                        file: "aie_registers_aie2.json".into(),
                        detail: format!("module={} structural hierarchy", am025_name),
                    };

                    // Edge: TileType -> Module (Contains)
                    new_edges.push(Relationship {
                        from: NodeId::TileType { kind: tile_type.kind },
                        to: NodeId::Module { tile: tile_type.kind, module: kind },
                        kind: RelationshipKind::Contains,
                        source: edge_source.clone(),
                    });

                    // Edges: Module -> Register (Contains),
                    //        Register -> RegisterField (Contains),
                    //        Register -> Subsystem (BelongsTo)
                    for reg in &module_def.registers {
                        let reg_node = NodeId::Register {
                            tile: tile_type.kind,
                            module: kind,
                            name: reg.name.clone(),
                        };

                        new_edges.push(Relationship {
                            from: NodeId::Module { tile: tile_type.kind, module: kind },
                            to: reg_node.clone(),
                            kind: RelationshipKind::Contains,
                            source: edge_source.clone(),
                        });

                        // Register -> Subsystem (BelongsTo)
                        let subsystem_kind = subsystem_name_to_kind(subsystem_key(&reg.name));
                        new_edges.push(Relationship {
                            from: reg_node.clone(),
                            to: NodeId::Subsystem {
                                tile: tile_type.kind,
                                module: kind,
                                subsystem: subsystem_kind,
                            },
                            kind: RelationshipKind::BelongsTo,
                            source: edge_source.clone(),
                        });

                        // Register -> RegisterField (Contains)
                        for field in &reg.fields {
                            new_edges.push(Relationship {
                                from: reg_node.clone(),
                                to: NodeId::RegisterField {
                                    tile: tile_type.kind,
                                    module: kind,
                                    register: reg.name.clone(),
                                    field: field.name.clone(),
                                },
                                kind: RelationshipKind::Contains,
                                source: edge_source.clone(),
                            });
                        }
                    }

                    // Confirm instance counts from register group analysis.
                    let profiles = extract_module_profiles(&module_def.registers);
                    for profile in &profiles {
                        let am025_source = SourceAttribution {
                            origin: Source::Am025Json,
                            file: "aie_registers_aie2.json".into(),
                            detail: format!(
                                "module={}, subsystem={} register grouping",
                                am025_name, profile.subsystem
                            ),
                        };
                        match profile.kind {
                            SubsystemKind::Lock => {
                                tile_type.instances.locks.confirm(
                                    profile.instance_count as u8,
                                    am025_source,
                                );
                            }
                            SubsystemKind::Dma => {
                                tile_type.instances.bds.confirm(
                                    profile.instance_count as u8,
                                    am025_source,
                                );
                            }
                            _ => {}
                        }
                    }

                    // Extract BD schema from DMA BD registers if this
                    // module contains them and we haven't extracted one yet.
                    if tile_type.bd_schema.is_none() {
                        let (schema, edges) = extract_bd_schema(
                            &module_def.registers,
                            am025_name,
                            tile_type.kind,
                            kind,
                        );
                        tile_type.bd_schema = schema;
                        new_edges.extend(edges);
                    }

                    // Extract DMA channel schema from channel control/status
                    // registers if this module contains them.
                    if tile_type.channel_schema.is_none() {
                        let (schema, edges) = extract_channel_schema(
                            &module_def.registers,
                            am025_name,
                            tile_type.kind,
                            kind,
                        );
                        tile_type.channel_schema = schema;
                        new_edges.extend(edges);
                    }
                }
            }
        }

        // After all modules for this tile type are processed, emit
        // cross-resource References edges between schemas and subsystems.
        // We need to find the module kind that contains the DMA subsystem
        // to correctly address the Lock subsystem node.
        let dma_module_kind = TILE_MODULE_MAP
            .iter()
            .find(|(name, _)| *name == tile_type.name)
            .and_then(|(_, pairs)| {
                // The module containing DMA registers (Memory for compute,
                // MemTile for memtile, Shim for shim).
                pairs.iter().find(|(_, am025)| *am025 != "core").map(|(k, _)| *k)
            });

        if let Some(mod_kind) = dma_module_kind {
            new_edges.extend(emit_cross_reference_edges(
                tile_type.kind,
                mod_kind,
                tile_type.bd_schema.as_ref(),
                tile_type.channel_schema.as_ref(),
            ));
        }
    }

    model.relationships.extend(new_edges);
}

// ============================================================================
// BD schema extraction
// ============================================================================

/// Map an AM025 BD field name to its semantic role.
fn bd_field_name_to_role(name: &str) -> BdFieldRole {
    match name {
        "Buffer_Length" => BdFieldRole::BufferLength,
        "Base_Address" | "Base_Address_Low" => BdFieldRole::BaseAddress,
        "Base_Address_High" => BdFieldRole::BaseAddressHigh,
        "D0_Stepsize" => BdFieldRole::Stepsize(0),
        "D1_Stepsize" => BdFieldRole::Stepsize(1),
        "D2_Stepsize" => BdFieldRole::Stepsize(2),
        "D3_Stepsize" => BdFieldRole::Stepsize(3),
        "D0_Wrap" => BdFieldRole::Wrap(0),
        "D1_Wrap" => BdFieldRole::Wrap(1),
        "D2_Wrap" => BdFieldRole::Wrap(2),
        "D0_Zero_Before" => BdFieldRole::ZeroPadBefore(0),
        "D1_Zero_Before" => BdFieldRole::ZeroPadBefore(1),
        "D2_Zero_Before" => BdFieldRole::ZeroPadBefore(2),
        "D0_Zero_After" => BdFieldRole::ZeroPadAfter(0),
        "D1_Zero_After" => BdFieldRole::ZeroPadAfter(1),
        "D2_Zero_After" => BdFieldRole::ZeroPadAfter(2),
        "Iteration_Stepsize" => BdFieldRole::IterationStepsize,
        "Iteration_Wrap" => BdFieldRole::IterationWrap,
        "Iteration_Current" => BdFieldRole::IterationCurrent,
        "Lock_Acq_ID" => BdFieldRole::LockAcqId,
        "Lock_Acq_Value" => BdFieldRole::LockAcqValue,
        "Lock_Acq_Enable" => BdFieldRole::LockAcqEnable,
        "Lock_Rel_ID" => BdFieldRole::LockRelId,
        "Lock_Rel_Value" => BdFieldRole::LockRelValue,
        "Valid_BD" => BdFieldRole::ValidBd,
        "Use_Next_BD" => BdFieldRole::UseNextBd,
        "Next_BD" => BdFieldRole::NextBd,
        "Enable_Packet" => BdFieldRole::EnablePacket,
        "Packet_ID" => BdFieldRole::PacketId,
        "Packet_Type" => BdFieldRole::PacketType,
        "Out_Of_Order_BD_ID" => BdFieldRole::OutOfOrderBdId,
        "TLAST_Suppress" => BdFieldRole::TlastSuppress,
        "Enable_Compression" => BdFieldRole::EnableCompression,
        "Burst_Length" => BdFieldRole::BurstLength,
        "SMID" => BdFieldRole::Smid,
        "AxCache" => BdFieldRole::AxCache,
        "AxQoS" => BdFieldRole::AxQos,
        "Secure_Access" => BdFieldRole::SecureAccess,
        "Reserved" => BdFieldRole::Reserved,
        _ => BdFieldRole::Reserved,
    }
}

/// Extract a BD schema from a module's registers, with DerivedFrom edges.
///
/// Finds DMA_BD0_* registers (the representative BD), extracts all fields
/// with their bit positions and semantic roles. Returns `(None, [])` if the
/// module has no DMA BD registers.
///
/// For each non-Reserved field, emits a `DerivedFrom` edge linking the
/// schema field node to the source register field node.
fn extract_bd_schema(
    registers: &[regdb::RegisterDef],
    module_name: &str,
    tile_kind: TileKind,
    module_kind: ModuleKind,
) -> (Option<BdSchema>, Vec<Relationship>) {
    use crate::types::{NodeId, Relationship, RelationshipKind};

    // Find all BD0 registers: DMA_BD0_0, DMA_BD0_1, ... sorted by offset.
    let mut bd0_regs: Vec<&regdb::RegisterDef> = registers
        .iter()
        .filter(|r| r.name.starts_with("DMA_BD0_"))
        .collect();

    if bd0_regs.is_empty() {
        return (None, Vec::new());
    }

    bd0_regs.sort_by_key(|r| r.offset);
    let words_per_bd = bd0_regs.len() as u8;

    // Extract fields from each word of BD0, emitting DerivedFrom edges.
    let mut fields = Vec::new();
    let mut edges = Vec::new();
    let edge_source = SourceAttribution {
        origin: Source::Am025Json,
        file: "aie_registers_aie2.json".into(),
        detail: format!("module={} BD schema derivation", module_name),
    };

    for (word_idx, reg) in bd0_regs.iter().enumerate() {
        for field_def in &reg.fields {
            let role = bd_field_name_to_role(&field_def.name);
            let bits = if field_def.msb >= field_def.lsb {
                BitRange::Contiguous {
                    msb: field_def.msb as u8,
                    lsb: field_def.lsb as u8,
                }
            } else {
                BitRange::Contiguous { msb: 0, lsb: 0 }
            };

            fields.push(BdFieldSpec {
                name: field_def.name.clone(),
                word: word_idx as u8,
                bits,
                role: role.clone(),
            });

            // Emit DerivedFrom edge for non-Reserved fields.
            if role != BdFieldRole::Reserved {
                edges.push(Relationship {
                    from: NodeId::BdField {
                        tile: tile_kind,
                        role,
                    },
                    to: NodeId::RegisterField {
                        tile: tile_kind,
                        module: module_kind,
                        register: reg.name.clone(),
                        field: field_def.name.clone(),
                    },
                    kind: RelationshipKind::DerivedFrom,
                    source: edge_source.clone(),
                });
            }
        }
    }

    // Sort by (word, lsb) for deterministic ordering.
    fields.sort_by(|a, b| {
        let a_lsb = match &a.bits {
            BitRange::Contiguous { lsb, .. } => *lsb,
            BitRange::Split(frags) => frags.last().map(|(_, l)| *l).unwrap_or(0),
        };
        let b_lsb = match &b.bits {
            BitRange::Contiguous { lsb, .. } => *lsb,
            BitRange::Split(frags) => frags.last().map(|(_, l)| *l).unwrap_or(0),
        };
        (a.word, a_lsb).cmp(&(b.word, b_lsb))
    });

    // Derive capabilities from field presence.
    let has_field = |role: &BdFieldRole| fields.iter().any(|f| &f.role == role);
    let has_zero_padding = has_field(&BdFieldRole::ZeroPadBefore(0));
    let has_axi_fields = has_field(&BdFieldRole::BurstLength)
        || has_field(&BdFieldRole::Smid)
        || has_field(&BdFieldRole::AxCache);

    // Count addressing dimensions: D0 always present, check D1/D2/D3.
    let num_dimensions = if has_field(&BdFieldRole::Stepsize(3)) {
        4
    } else if has_field(&BdFieldRole::Stepsize(2)) {
        3
    } else if has_field(&BdFieldRole::Stepsize(1)) {
        2
    } else if has_field(&BdFieldRole::Stepsize(0)) {
        1
    } else {
        0
    };

    (
        Some(BdSchema {
            words_per_bd,
            num_dimensions,
            has_zero_padding,
            has_axi_fields,
            fields,
            source: SourceAttribution {
                origin: Source::Am025Json,
                file: "aie_registers_aie2.json".into(),
                detail: format!("module={} DMA_BD0 field layout", module_name),
            },
        }),
        edges,
    )
}

// ============================================================================
// DMA channel schema extraction
// ============================================================================

/// Map an AM025 DMA channel register field name to its semantic role.
fn channel_field_name_to_role(name: &str) -> DmaChannelFieldRole {
    match name {
        // Control register fields
        "Reset" => DmaChannelFieldRole::Reset,
        "Pause_Mem" => DmaChannelFieldRole::PauseMem,
        "Pause_Stream" => DmaChannelFieldRole::PauseStream,
        "Compression_Enable" => DmaChannelFieldRole::CompressionEnable,
        "Decompression_Enable" => DmaChannelFieldRole::DecompressionEnable,
        "Enable_Out_of_Order" => DmaChannelFieldRole::EnableOutOfOrder,
        "Controller_ID" => DmaChannelFieldRole::ControllerId,
        "FoT_Mode" => DmaChannelFieldRole::FotMode,

        // Start/Task Queue fields
        "Start_BD_ID" => DmaChannelFieldRole::StartBdId,
        "Repeat_Count" => DmaChannelFieldRole::RepeatCount,
        "Enable_Token_Issue" => DmaChannelFieldRole::EnableTokenIssue,

        // Status register fields
        "Status" => DmaChannelFieldRole::ChannelStatus,
        "Stalled_Lock_Acq" => DmaChannelFieldRole::StalledLockAcq,
        "Stalled_Lock_Rel" => DmaChannelFieldRole::StalledLockRel,
        "Stalled_Stream_Starvation" => DmaChannelFieldRole::StalledStreamStarvation,
        "Stalled_Stream_Backpressure" => DmaChannelFieldRole::StalledStreamBackpressure,
        "Stalled_TCT" => DmaChannelFieldRole::StalledTct,
        "Stalled_TCT_or_Count_FIFO_Full" => DmaChannelFieldRole::StalledTctOrCountFifoFull,
        "Error_BD_Unavailable" => DmaChannelFieldRole::ErrorBdUnavailable,
        "Error_BD_Invalid" => DmaChannelFieldRole::ErrorBdInvalid,
        "Error_FoT_Length_Exceeded" => DmaChannelFieldRole::ErrorFotLengthExceeded,
        "Error_FoT_BDs_per_Task" => DmaChannelFieldRole::ErrorFotBdsPerTask,
        "Error_Lock_Access_to_Unavailable" => DmaChannelFieldRole::ErrorLockAccessUnavailable,
        "Error_DM_Access_to_Unavailable" => DmaChannelFieldRole::ErrorDmAccessUnavailable,
        "AXI_MM_Decode_Error" => DmaChannelFieldRole::AxiMmDecodeError,
        "AXI_MM_Slave_Error" => DmaChannelFieldRole::AxiMmSlaveError,
        "Task_Queue_Overflow" => DmaChannelFieldRole::TaskQueueOverflow,
        "Channel_Running" => DmaChannelFieldRole::ChannelRunning,
        "Task_Queue_Size" => DmaChannelFieldRole::TaskQueueSize,
        "Cur_BD" => DmaChannelFieldRole::CurrentBdId,

        // Write Count register fields (S2MM only)
        "Current_Write_Count" => DmaChannelFieldRole::CurrentWriteCount,

        // FoT Count FIFO fields (S2MM only)
        "Write_Count" => DmaChannelFieldRole::FotWriteCount,
        "BD_ID" => DmaChannelFieldRole::FotBdId,
        "Last_in_Task" => DmaChannelFieldRole::FotLastInTask,
        "Valid" => DmaChannelFieldRole::FotValid,

        "Reserved" | _ => DmaChannelFieldRole::Reserved,
    }
}

/// Classify a DMA channel register by its name pattern, returning the
/// register kind, direction, and channel index.
///
/// Register naming patterns in AM025:
/// - Control:    `DMA_{S2MM|MM2S}_{ch}_Ctrl`
/// - Queue:      `DMA_{S2MM|MM2S}_{ch}_Start_Queue` or `_Task_Queue`
/// - Status:     `DMA_{S2MM|MM2S}_Status_{ch}`
/// - WriteCount: `DMA_S2MM_Current_Write_Count_{ch}`
/// - FoT FIFO:   `DMA_S2MM_FoT_Count_FIFO_Pop_{ch}`
fn classify_channel_register(
    name: &str,
) -> Option<(DmaChannelRegKind, DmaDirection, u8, &str)> {
    // Must start with DMA_ and not be a BD register
    if !name.starts_with("DMA_") || name.contains("_BD") {
        return None;
    }

    // Determine direction
    let direction = if name.starts_with("DMA_S2MM_") {
        DmaDirection::S2mm
    } else if name.starts_with("DMA_MM2S_") {
        DmaDirection::Mm2s
    } else {
        return None;
    };

    // Try each pattern in order of specificity

    // Write Count: DMA_S2MM_Current_Write_Count_{ch}
    if let Some(rest) = name.strip_prefix("DMA_S2MM_Current_Write_Count_") {
        if let Ok(ch) = rest.parse::<u8>() {
            return Some((DmaChannelRegKind::WriteCount, direction, ch, ""));
        }
    }

    // FoT FIFO: DMA_S2MM_FoT_Count_FIFO_Pop_{ch}
    if let Some(rest) = name.strip_prefix("DMA_S2MM_FoT_Count_FIFO_Pop_") {
        if let Ok(ch) = rest.parse::<u8>() {
            return Some((DmaChannelRegKind::FotCountFifo, direction, ch, ""));
        }
    }

    // Status: DMA_{dir}_Status_{ch}
    let dir_prefix = if direction == DmaDirection::S2mm {
        "DMA_S2MM_Status_"
    } else {
        "DMA_MM2S_Status_"
    };
    if let Some(rest) = name.strip_prefix(dir_prefix) {
        if let Ok(ch) = rest.parse::<u8>() {
            return Some((DmaChannelRegKind::Status, direction, ch, ""));
        }
    }

    // Control and Queue: DMA_{dir}_{ch}_{Ctrl|Start_Queue|Task_Queue}
    let dir_stem = if direction == DmaDirection::S2mm {
        "DMA_S2MM_"
    } else {
        "DMA_MM2S_"
    };
    if let Some(rest) = name.strip_prefix(dir_stem) {
        // rest should be "{ch}_Ctrl" or "{ch}_Start_Queue" or "{ch}_Task_Queue"
        if let Some(underscore_pos) = rest.find('_') {
            let ch_str = &rest[..underscore_pos];
            let suffix = &rest[underscore_pos + 1..];
            if let Ok(ch) = ch_str.parse::<u8>() {
                match suffix {
                    "Ctrl" => return Some((DmaChannelRegKind::Control, direction, ch, "")),
                    "Start_Queue" => {
                        return Some((DmaChannelRegKind::StartQueue, direction, ch, "Start_Queue"))
                    }
                    "Task_Queue" => {
                        return Some((DmaChannelRegKind::StartQueue, direction, ch, "Task_Queue"))
                    }
                    _ => {}
                }
            }
        }
    }

    None
}

/// Extract a DMA channel schema from a module's registers, with DerivedFrom edges.
///
/// Finds the representative channel registers (channel 0 for each direction)
/// and extracts all fields with their bit positions and semantic roles.
/// Also counts total channels per direction from the register set.
///
/// For each non-Reserved field, emits a `DerivedFrom` edge linking the
/// schema field node to the source register field node.
fn extract_channel_schema(
    registers: &[regdb::RegisterDef],
    module_name: &str,
    tile_kind: TileKind,
    module_kind: ModuleKind,
) -> (Option<DmaChannelSchema>, Vec<Relationship>) {
    use crate::types::{NodeId, Relationship, RelationshipKind};

    // Classify all DMA channel registers
    let mut channel_regs: Vec<(DmaChannelRegKind, DmaDirection, u8, &str, &regdb::RegisterDef)> =
        Vec::new();
    for reg in registers {
        if let Some((kind, dir, ch, queue_name)) = classify_channel_register(&reg.name) {
            channel_regs.push((kind, dir, ch, queue_name, reg));
        }
    }

    if channel_regs.is_empty() {
        return (None, Vec::new());
    }

    // Count channels per direction from S2MM Control registers
    let s2mm_ctrl_count = channel_regs
        .iter()
        .filter(|(k, d, _, _, _)| *k == DmaChannelRegKind::Control && *d == DmaDirection::S2mm)
        .count() as u8;

    // Determine queue register name from the first queue register found
    let queue_name = channel_regs
        .iter()
        .find(|(k, _, _, _, _)| *k == DmaChannelRegKind::StartQueue)
        .map(|(_, _, _, qn, _)| qn.to_string())
        .unwrap_or_else(|| "Start_Queue".to_string());

    // Extract fields from representative channel (channel 0) for each
    // (register kind, direction) combination, emitting DerivedFrom edges.
    let mut fields = Vec::new();
    let mut edges = Vec::new();
    let edge_source = SourceAttribution {
        origin: Source::Am025Json,
        file: "aie_registers_aie2.json".into(),
        detail: format!("module={} channel schema derivation", module_name),
    };

    for &(reg_kind, dir, ch, _, reg) in &channel_regs {
        if ch != 0 {
            continue; // Only use channel 0 as representative
        }

        for field_def in &reg.fields {
            let role = channel_field_name_to_role(&field_def.name);
            let bits = if field_def.msb >= field_def.lsb {
                BitRange::Contiguous {
                    msb: field_def.msb as u8,
                    lsb: field_def.lsb as u8,
                }
            } else {
                BitRange::Contiguous { msb: 0, lsb: 0 }
            };

            fields.push(DmaChannelFieldSpec {
                name: field_def.name.clone(),
                register: reg_kind,
                direction: dir,
                bits,
                role: role.clone(),
            });

            // Emit DerivedFrom edge for non-Reserved fields.
            if role != DmaChannelFieldRole::Reserved {
                edges.push(Relationship {
                    from: NodeId::ChannelField {
                        tile: tile_kind,
                        direction: dir,
                        role,
                    },
                    to: NodeId::RegisterField {
                        tile: tile_kind,
                        module: module_kind,
                        register: reg.name.clone(),
                        field: field_def.name.clone(),
                    },
                    kind: RelationshipKind::DerivedFrom,
                    source: edge_source.clone(),
                });
            }
        }
    }

    // Sort by (register, direction, lsb) for deterministic ordering
    fields.sort_by(|a, b| {
        let a_reg = a.register as u8;
        let b_reg = b.register as u8;
        let a_dir = a.direction as u8;
        let b_dir = b.direction as u8;
        let a_lsb = match &a.bits {
            BitRange::Contiguous { lsb, .. } => *lsb,
            BitRange::Split(frags) => frags.last().map(|(_, l)| *l).unwrap_or(0),
        };
        let b_lsb = match &b.bits {
            BitRange::Contiguous { lsb, .. } => *lsb,
            BitRange::Split(frags) => frags.last().map(|(_, l)| *l).unwrap_or(0),
        };
        (a_reg, a_dir, a_lsb).cmp(&(b_reg, b_dir, b_lsb))
    });

    // Derive capability flags from field presence
    let has_role = |role: &DmaChannelFieldRole| fields.iter().any(|f| &f.role == role);

    let has_compression = has_role(&DmaChannelFieldRole::CompressionEnable)
        || has_role(&DmaChannelFieldRole::DecompressionEnable);
    let has_out_of_order = has_role(&DmaChannelFieldRole::EnableOutOfOrder);
    let has_pause =
        has_role(&DmaChannelFieldRole::PauseMem) || has_role(&DmaChannelFieldRole::PauseStream);
    let has_fot = has_role(&DmaChannelFieldRole::FotMode);
    let has_axi_errors = has_role(&DmaChannelFieldRole::AxiMmDecodeError)
        || has_role(&DmaChannelFieldRole::AxiMmSlaveError);
    let has_cross_tile_errors = has_role(&DmaChannelFieldRole::ErrorLockAccessUnavailable)
        || has_role(&DmaChannelFieldRole::ErrorDmAccessUnavailable);

    // Start_BD_ID width from the field itself
    let start_bd_id_width = fields
        .iter()
        .find(|f| f.role == DmaChannelFieldRole::StartBdId)
        .map(|f| f.bits.width())
        .unwrap_or(4);

    (
        Some(DmaChannelSchema {
            channels_per_direction: s2mm_ctrl_count,
            start_bd_id_width,
            has_compression,
            has_out_of_order,
            has_pause,
            has_fot,
            has_axi_errors,
            has_cross_tile_errors,
            queue_name,
            fields,
            source: SourceAttribution {
                origin: Source::Am025Json,
                file: "aie_registers_aie2.json".into(),
                detail: format!("module={} DMA channel registers", module_name),
            },
        }),
        edges,
    )
}

// ============================================================================
// Cross-resource References edges
// ============================================================================

/// Emit References edges for fields that index into other resource spaces.
///
/// These are static architectural facts: a BD's LockAcqId field always
/// references the Lock subsystem, a channel's StartBdId always points into
/// the BD space, etc. They don't vary by register content -- they're
/// inherent to the hardware programming model.
fn emit_cross_reference_edges(
    tile_kind: TileKind,
    module_kind: ModuleKind,
    bd_schema: Option<&BdSchema>,
    channel_schema: Option<&DmaChannelSchema>,
) -> Vec<Relationship> {
    use crate::types::{NodeId, RelationshipKind};

    let mut edges = Vec::new();
    let source = SourceAttribution {
        origin: Source::Am025Json,
        file: "aie_registers_aie2.json".into(),
        detail: "cross-resource field references".into(),
    };

    let lock_subsystem = NodeId::Subsystem {
        tile: tile_kind,
        module: module_kind,
        subsystem: SubsystemKind::Lock,
    };

    // BD field -> ValidBd is the representative node for the BD resource space.
    let bd_space = NodeId::BdField {
        tile: tile_kind,
        role: BdFieldRole::ValidBd,
    };

    // BD fields that reference other resource spaces
    if let Some(schema) = bd_schema {
        let has_role = |role: &BdFieldRole| schema.fields.iter().any(|f| &f.role == role);

        // LockAcqId, LockRelId -> Lock subsystem
        if has_role(&BdFieldRole::LockAcqId) {
            edges.push(Relationship {
                from: NodeId::BdField { tile: tile_kind, role: BdFieldRole::LockAcqId },
                to: lock_subsystem.clone(),
                kind: RelationshipKind::References,
                source: source.clone(),
            });
        }
        if has_role(&BdFieldRole::LockRelId) {
            edges.push(Relationship {
                from: NodeId::BdField { tile: tile_kind, role: BdFieldRole::LockRelId },
                to: lock_subsystem.clone(),
                kind: RelationshipKind::References,
                source: source.clone(),
            });
        }

        // NextBd -> BD space (chains to another BD)
        if has_role(&BdFieldRole::NextBd) {
            edges.push(Relationship {
                from: NodeId::BdField { tile: tile_kind, role: BdFieldRole::NextBd },
                to: bd_space.clone(),
                kind: RelationshipKind::References,
                source: source.clone(),
            });
        }

        // OutOfOrderBdId -> BD space
        if has_role(&BdFieldRole::OutOfOrderBdId) {
            edges.push(Relationship {
                from: NodeId::BdField { tile: tile_kind, role: BdFieldRole::OutOfOrderBdId },
                to: bd_space.clone(),
                kind: RelationshipKind::References,
                source: source.clone(),
            });
        }
    }

    // Channel fields that reference other resource spaces
    if let Some(schema) = channel_schema {
        let has_field = |dir: DmaDirection, role: &DmaChannelFieldRole| {
            schema.fields.iter().any(|f| f.direction == dir && &f.role == role)
        };

        // StartBdId -> BD space (both directions)
        for dir in [DmaDirection::S2mm, DmaDirection::Mm2s] {
            if has_field(dir, &DmaChannelFieldRole::StartBdId) {
                edges.push(Relationship {
                    from: NodeId::ChannelField {
                        tile: tile_kind,
                        direction: dir,
                        role: DmaChannelFieldRole::StartBdId,
                    },
                    to: bd_space.clone(),
                    kind: RelationshipKind::References,
                    source: source.clone(),
                });
            }

            // CurrentBdId -> BD space
            if has_field(dir, &DmaChannelFieldRole::CurrentBdId) {
                edges.push(Relationship {
                    from: NodeId::ChannelField {
                        tile: tile_kind,
                        direction: dir,
                        role: DmaChannelFieldRole::CurrentBdId,
                    },
                    to: bd_space.clone(),
                    kind: RelationshipKind::References,
                    source: source.clone(),
                });
            }
        }
    }

    edges
}

// ============================================================================
// Subsystem key extraction: identify which subsystem a register belongs to
// ============================================================================

/// AM025 naming convention aliases that map to canonical subsystem names.
///
/// These are structural properties of how AMD names registers in the
/// AM025 register reference, not semantic assumptions about hardware.
/// The convention is stable across architecture revisions (AIE2, AIE2P).
const SUBSYSTEM_ALIASES: &[(&str, &str)] = &[
    ("Locks", "Lock"),  // "Locks_Overflow" -> Lock subsystem
    ("Combo", "Event"), // "Combo_event_inputs0" -> Event subsystem
    ("Edge", "Event"),  // "Edge_Detection_event_0" -> Event subsystem
];

/// Map a discovered subsystem name (from `subsystem_key()`) to its typed
/// `SubsystemKind` variant.
///
/// The mapping covers three tiers:
/// - **aie-rt modules**: DMA, Lock, Event, Trace, Performance, Timer, Stream
/// - **Hardware groupings**: WatchPoint, Debug, PC, Interrupt, NoC, DataMemory
/// - **Naming conventions**: Core->Processor, Program->ProgramMemory
///
/// Everything else maps to `Unknown` -- safe for future subsystem names that
/// may appear in new architecture revisions.
pub fn subsystem_name_to_kind(name: &str) -> SubsystemKind {
    match name {
        // Tier 1: aie-rt functional modules
        "DMA" => SubsystemKind::Dma,
        "Lock" => SubsystemKind::Lock,
        "Event" => SubsystemKind::Event,
        "Trace" => SubsystemKind::Trace,
        "Performance" => SubsystemKind::Performance,
        "Timer" => SubsystemKind::Timer,
        "Stream" => SubsystemKind::StreamSwitch,
        // Tier 2: hardware groupings
        "WatchPoint" => SubsystemKind::WatchPoint,
        "Debug" => SubsystemKind::Debug,
        "PC" => SubsystemKind::ProgramCounter,
        "Interrupt" => SubsystemKind::Interrupt,
        "NoC" => SubsystemKind::NoC,
        "DataMemory" => SubsystemKind::DataMemory,
        // Tier 3: naming convention translations
        "Core" => SubsystemKind::Processor,
        "Program" => SubsystemKind::ProgramMemory,
        // Catch-all: naming artifacts and future unknowns
        _ => SubsystemKind::Unknown,
    }
}

/// Extract the subsystem identity from a register name.
///
/// Uses the AM025 naming convention: the first underscore-delimited segment
/// (with trailing digits stripped) identifies the subsystem. A small alias
/// table normalizes known naming irregularities.
///
/// Examples:
/// - "DMA_BD0_0" -> "DMA"
/// - "Lock0_value" -> "Lock"
/// - "Locks_Overflow" -> "Lock" (alias normalization)
/// - "Combo_event_inputs0" -> "Event" (alias normalization)
pub fn subsystem_key(name: &str) -> &str {
    // First underscore-delimited segment
    let first = name.split('_').next().unwrap_or(name);
    // Strip trailing digits (e.g., "Lock0" -> "Lock")
    let stem = first.trim_end_matches(|c: char| c.is_ascii_digit());

    // Check alias table
    for &(alias, canonical) in SUBSYSTEM_ALIASES {
        if stem == alias {
            return canonical;
        }
    }

    stem
}

/// Extract subsystem profiles for all subsystems in a module.
///
/// This is the primary entry point for register analysis. Automatic
/// subsystem discovery from register names:
///
/// 1. Partition registers by `subsystem_key()` (name-stem extraction)
/// 2. Build a `SubsystemProfile` for each partition
/// 3. Return sorted by subsystem name
pub fn extract_module_profiles(
    registers: &[regdb::RegisterDef],
) -> Vec<SubsystemProfile> {
    let segments = segment_by_subsystem(registers);
    let mut profiles: Vec<SubsystemProfile> = segments
        .into_iter()
        .map(|(key, regs)| build_subsystem_profile(&key, &regs))
        .collect();
    profiles.sort_by(|a, b| a.subsystem.cmp(&b.subsystem));
    profiles
}

/// Partition a module's registers into subsystem groups.
///
/// Each register is assigned to a subsystem based on its name (via
/// `subsystem_key`). Returns a map from subsystem name to the registers
/// belonging to that subsystem, sorted by offset within each group.
pub fn segment_by_subsystem(
    registers: &[regdb::RegisterDef],
) -> std::collections::BTreeMap<String, Vec<regdb::RegisterDef>> {
    let mut segments: std::collections::BTreeMap<String, Vec<regdb::RegisterDef>> =
        std::collections::BTreeMap::new();
    for reg in registers {
        let key = subsystem_key(&reg.name).to_string();
        segments.entry(key).or_default().push(reg.clone());
    }
    // Sort each group by offset for consistent downstream processing
    for regs in segments.values_mut() {
        regs.sort_by_key(|r| r.offset);
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regdb::{AccessMode, BitField, RegisterDef};

    /// Helper: create a minimal RegisterDef with given name, offset, and field names.
    fn make_reg(name: &str, offset: u32, field_names: &[&str]) -> RegisterDef {
        RegisterDef {
            name: name.to_string(),
            offset,
            width: 32,
            access: AccessMode::ReadWrite,
            reset_value: 0,
            description: None,
            fields: field_names
                .iter()
                .map(|n| BitField {
                    name: n.to_string(),
                    lsb: 0,
                    msb: 0,
                    width: 1,
                    mask: 1,
                    shift: 0,
                })
                .collect(),
        }
    }

    #[test]
    fn groups_numbered_lock_value_registers() {
        // 16 lock value registers matching the memory module layout
        let mut regs: Vec<RegisterDef> = (0..16)
            .map(|i| make_reg(&format!("Lock{}_value", i), 0x1F000 + i * 0x10, &["Lock_value"]))
            .collect();

        // Add a singleton (non-numbered) register that should NOT be grouped
        regs.push(make_reg("Lock_Request", 0x40000, &["Request_Result"]));

        let groups = group_registers(&regs);

        // Should find exactly one group (Lock{}_value), not Lock_Request
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!(g.pattern, "Lock{}_value");
        assert_eq!(g.dimensions.len(), 1, "1D group");
        assert_eq!(g.dimensions[0].count, 16);
        assert_eq!(g.base_offset, 0x1F000);
        assert_eq!(g.dimensions[0].stride, 0x10);
        assert_eq!(g.field_names, vec!["Lock_value"]);
    }

    #[test]
    fn groups_event_selection_registers() {
        // 8 event selection registers
        let regs: Vec<RegisterDef> = (0..8)
            .map(|i| {
                make_reg(
                    &format!("Locks_Event_Selection_{}", i),
                    0x1F100 + i * 0x4,
                    &["Lock_Select", "Lock_Value"],
                )
            })
            .collect();

        let groups = group_registers(&regs);

        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!(g.pattern, "Locks_Event_Selection_{}");
        assert_eq!(g.dimensions.len(), 1, "1D group");
        assert_eq!(g.dimensions[0].count, 8);
        assert_eq!(g.base_offset, 0x1F100);
        assert_eq!(g.dimensions[0].stride, 0x4);
    }

    #[test]
    fn singletons_not_grouped() {
        let regs = vec![
            make_reg("Lock_Request", 0x40000, &["Request_Result"]),
            make_reg("Locks_Overflow", 0x1F120, &["Lock_Overflow_0"]),
            make_reg("Locks_Underflow", 0x1F128, &["Lock_Underflow_0"]),
        ];

        let groups = group_registers(&regs);
        assert!(groups.is_empty(), "Singletons should not form groups");
    }

    // ====================================================================
    // Classification tests -- property-based
    // ====================================================================

    /// Helper: create a RegisterDef with a specific access mode.
    fn make_reg_with_access(
        name: &str,
        offset: u32,
        access: AccessMode,
        field_names: &[&str],
    ) -> RegisterDef {
        RegisterDef {
            name: name.to_string(),
            offset,
            width: 32,
            access,
            reset_value: 0,
            description: None,
            fields: field_names
                .iter()
                .map(|n| BitField {
                    name: n.to_string(),
                    lsb: 0,
                    msb: 0,
                    width: 1,
                    mask: 1,
                    shift: 0,
                })
                .collect(),
        }
    }

    #[test]
    fn classify_readonly_as_operations() {
        // Lock_Request: read-only, returns operation result in bit 0
        let reg = make_reg_with_access(
            "Lock_Request",
            0x40000,
            AccessMode::ReadOnly,
            &["Request_Result"],
        );
        assert_eq!(classify_register(&reg, false), RegisterCategory::Operations);
    }

    #[test]
    fn classify_write_to_clear_as_status() {
        // Locks_Overflow: hardware sets bits, software clears by writing 1
        let reg = make_reg_with_access(
            "Locks_Overflow",
            0x1F120,
            AccessMode::WriteToClear,
            &["Lock_Overflow_0", "Lock_Overflow_1"],
        );
        assert_eq!(classify_register(&reg, false), RegisterCategory::Status);
    }

    #[test]
    fn classify_rw_grouped_as_state() {
        // Lock0_value: R/W, part of a numbered group -> per-instance state
        let reg = make_reg(
            "Lock0_value",
            0x1F000,
            &["Lock_value"],
        );
        assert_eq!(classify_register(&reg, true), RegisterCategory::State);
    }

    #[test]
    fn classify_rw_singleton_as_unclassified() {
        // A generic R/W singleton needs more context to classify
        let reg = make_reg(
            "Some_Config_Register",
            0x10000,
            &["Some_Field"],
        );
        assert_eq!(
            classify_register(&reg, false),
            RegisterCategory::Unclassified
        );
    }

    #[test]
    fn classify_overflow_name_as_status() {
        // Even if access mode is R/W (not WriteToClear), the universal
        // Overflow/Underflow pattern is strong enough to classify as Status
        let overflow = make_reg("Locks_Overflow", 0x1F120, &["Lock_Overflow_0"]);
        let underflow = make_reg("Locks_Underflow", 0x1F128, &["Lock_Underflow_0"]);
        assert_eq!(classify_register(&overflow, false), RegisterCategory::Status);
        assert_eq!(classify_register(&underflow, false), RegisterCategory::Status);
    }

    // ====================================================================
    // Subsystem profile tests
    // ====================================================================

    #[test]
    fn lock_profile_from_memory_module_registers() {
        // Simulate the full set of lock registers in a memory module
        let mut regs = Vec::new();

        // 16 lock value registers (R/W, grouped -> State)
        for i in 0..16u32 {
            regs.push(make_reg(
                &format!("Lock{}_value", i),
                0x1F000 + i * 0x10,
                &["Lock_value"],
            ));
        }

        // Lock_Request (ReadOnly -> Operations)
        regs.push(make_reg_with_access(
            "Lock_Request",
            0x40000,
            AccessMode::ReadOnly,
            &["Request_Result"],
        ));

        // 8 event selection registers (R/W, grouped -> State)
        for i in 0..8u32 {
            regs.push(make_reg(
                &format!("Locks_Event_Selection_{}", i),
                0x1F100 + i * 0x4,
                &["Lock_Select", "Lock_Value"],
            ));
        }

        // Overflow/Underflow (WriteToClear -> Status)
        regs.push(make_reg_with_access(
            "Locks_Overflow",
            0x1F120,
            AccessMode::WriteToClear,
            &["Lock_Overflow_0"],
        ));
        regs.push(make_reg_with_access(
            "Locks_Underflow",
            0x1F128,
            AccessMode::WriteToClear,
            &["Lock_Underflow_0"],
        ));

        let profile = build_subsystem_profile("lock", &regs);

        assert_eq!(profile.subsystem, "lock");
        // Instance count derived from the largest State group
        assert_eq!(profile.instance_count, 16);

        // Should have 2 register groups: Lock{}_value and Locks_Event_Selection_{}
        assert_eq!(profile.register_groups.len(), 2);

        // Lock values should be classified as State
        let value_group = profile
            .register_groups
            .iter()
            .find(|(_, g)| g.pattern == "Lock{}_value")
            .expect("Lock{}_value group should exist");
        assert_eq!(value_group.0, RegisterCategory::State);
        assert_eq!(value_group.1.instance_count(), 16);

        // Event selections should also be State (R/W grouped)
        let event_group = profile
            .register_groups
            .iter()
            .find(|(_, g)| g.pattern == "Locks_Event_Selection_{}")
            .expect("Event selection group should exist");
        assert_eq!(event_group.0, RegisterCategory::State);

        // Singletons: Lock_Request (Operations), Overflow (Status), Underflow (Status)
        assert_eq!(profile.singletons.len(), 3);
        assert!(profile
            .singletons
            .iter()
            .any(|(cat, name)| *cat == RegisterCategory::Operations
                && name == "Lock_Request"));
        assert!(profile
            .singletons
            .iter()
            .any(|(cat, name)| *cat == RegisterCategory::Status
                && name == "Locks_Overflow"));
    }

    // ====================================================================
    // Subsystem key extraction tests
    // ====================================================================

    #[test]
    fn subsystem_key_extracts_dma() {
        assert_eq!(subsystem_key("DMA_BD0_0"), "DMA");
        assert_eq!(subsystem_key("DMA_S2MM_0_Ctrl"), "DMA");
        assert_eq!(subsystem_key("DMA_MM2S_0_Ctrl"), "DMA");
    }

    #[test]
    fn subsystem_key_extracts_lock() {
        assert_eq!(subsystem_key("Lock0_value"), "Lock");
        assert_eq!(subsystem_key("Lock_Request"), "Lock");
    }

    #[test]
    fn subsystem_key_normalizes_locks_plural() {
        assert_eq!(subsystem_key("Locks_Overflow"), "Lock");
        assert_eq!(subsystem_key("Locks_Underflow"), "Lock");
        assert_eq!(subsystem_key("Locks_Event_Selection_0"), "Lock");
    }

    #[test]
    fn subsystem_key_normalizes_combo_to_event() {
        assert_eq!(subsystem_key("Combo_event_inputs0"), "Event");
    }

    #[test]
    fn subsystem_key_normalizes_edge_to_event() {
        assert_eq!(subsystem_key("Edge_Detection_event_0"), "Event");
    }

    #[test]
    fn subsystem_key_extracts_other_subsystems() {
        assert_eq!(subsystem_key("Performance_Counter0"), "Performance");
        assert_eq!(subsystem_key("Stream_Switch_Master_Config_AIE_Core0"), "Stream");
        assert_eq!(subsystem_key("Event_Broadcast0"), "Event");
        assert_eq!(subsystem_key("Event_Status0"), "Event");
    }

    #[test]
    fn subsystem_key_handles_no_underscore() {
        assert_eq!(subsystem_key("DataMemory"), "DataMemory");
    }

    // ====================================================================
    // Subsystem segmentation tests
    // ====================================================================

    #[test]
    fn segment_groups_by_subsystem_key() {
        let regs = vec![
            make_reg("DMA_BD0_0", 0x1D000, &["field"]),
            make_reg("DMA_BD0_1", 0x1D004, &["field"]),
            make_reg("Lock0_value", 0x1F000, &["Lock_value"]),
            make_reg("Lock1_value", 0x1F010, &["Lock_value"]),
            make_reg("Locks_Overflow", 0x1F120, &["Overflow"]),
        ];

        let segments = segment_by_subsystem(&regs);

        assert_eq!(segments.len(), 2, "DMA and Lock (Locks merged)");
        assert!(segments.contains_key("DMA"));
        assert!(segments.contains_key("Lock"));
        assert_eq!(segments["DMA"].len(), 2);
        assert_eq!(segments["Lock"].len(), 3, "Lock0, Lock1, Locks_Overflow");
    }

    #[test]
    fn segment_merges_event_aliases() {
        let regs = vec![
            make_reg("Event_Broadcast0", 0x14000, &["field"]),
            make_reg("Event_Broadcast1", 0x14004, &["field"]),
            make_reg("Combo_event_inputs0", 0x14400, &["field"]),
            make_reg("Edge_Detection_event_0", 0x14408, &["field"]),
        ];

        let segments = segment_by_subsystem(&regs);

        assert_eq!(segments.len(), 1, "All should merge into Event");
        assert!(segments.contains_key("Event"));
        assert_eq!(segments["Event"].len(), 4);
    }

    // ====================================================================
    // Grouping tests
    // ====================================================================

    #[test]
    fn memtile_64_locks_grouped() {
        // MemTile has 64 lock value registers
        let regs: Vec<RegisterDef> = (0..64)
            .map(|i| make_reg(&format!("Lock{}_value", i), 0xC0000 + i * 0x10, &["Lock_value"]))
            .collect();

        let groups = group_registers(&regs);

        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!(g.pattern, "Lock{}_value");
        assert_eq!(g.dimensions.len(), 1, "1D group");
        assert_eq!(g.dimensions[0].count, 64);
        assert_eq!(g.base_offset, 0xC0000);
        assert_eq!(g.dimensions[0].stride, 0x10);
    }

    // ====================================================================
    // Integration tests -- real AM025 data
    // ====================================================================

    /// Load the real register database, or skip if unavailable.
    fn load_test_db() -> Option<regdb::RegisterDb> {
        // CARGO_MANIFEST_DIR is crates/xdna-graph/, go up to npu-work/
        let json_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..") // up to npu-work/
            .join("mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json");
        regdb::RegisterDb::from_file(&json_path).ok()
    }

    /// Helper: extract one subsystem profile from a module's registers.
    fn profile_from_module(
        module: &regdb::ModuleDef,
        subsystem: &str,
    ) -> Option<SubsystemProfile> {
        let profiles = extract_module_profiles(&module.registers);
        profiles.into_iter().find(|p| p.subsystem == subsystem)
    }

    #[test]
    fn integration_memory_module_lock_profile() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found (set MLIR_AIE_PATH)");
            return;
        };

        let mem = db.module("memory").expect("memory module should exist");
        let profile = profile_from_module(mem, "Lock")
            .expect("Lock profile should be extracted from memory module");

        assert_eq!(profile.kind, SubsystemKind::Lock);
        assert_eq!(profile.instance_count, 16);

        // Lock value registers: grouped, State, stride 0x10
        let value_group = profile
            .register_groups
            .iter()
            .find(|(_, g)| g.pattern == "Lock{}_value")
            .expect("Lock{}_value group should exist in real data");
        assert_eq!(value_group.0, RegisterCategory::State);
        assert_eq!(value_group.1.instance_count(), 16);
        assert_eq!(value_group.1.dimensions[0].stride, 0x10);
        assert_eq!(value_group.1.base_offset, 0x1F000);

        // Event selection registers: grouped, 8 selectors
        let event_group = profile
            .register_groups
            .iter()
            .find(|(_, g)| g.pattern == "Locks_Event_Selection_{}")
            .expect("Event selection group should exist in real data");
        assert_eq!(event_group.1.instance_count(), 8);

        // Lock_Request: singleton, classified as Operations (ReadOnly)
        assert!(
            profile.singletons.iter().any(|(cat, name)| {
                *cat == RegisterCategory::Operations && name == "Lock_Request"
            }),
            "Lock_Request should be a singleton classified as Operations"
        );

        // Overflow and Underflow: singletons, classified as Status
        assert!(
            profile.singletons.iter().any(|(cat, name)| {
                *cat == RegisterCategory::Status && name == "Locks_Overflow"
            }),
            "Locks_Overflow should be Status"
        );
        assert!(
            profile.singletons.iter().any(|(cat, name)| {
                *cat == RegisterCategory::Status && name == "Locks_Underflow"
            }),
            "Locks_Underflow should be Status"
        );
    }

    #[test]
    fn integration_memtile_lock_profile() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let mt = db.module("memory_tile").expect("memory_tile module should exist");
        let profile = profile_from_module(mt, "Lock")
            .expect("Lock profile should be extracted from memtile");

        assert_eq!(profile.kind, SubsystemKind::Lock);
        assert_eq!(profile.instance_count, 64);

        let value_group = profile
            .register_groups
            .iter()
            .find(|(_, g)| g.pattern == "Lock{}_value")
            .expect("Lock{}_value group should exist for memtile");
        assert_eq!(value_group.1.instance_count(), 64);
        assert_eq!(value_group.1.dimensions[0].stride, 0x10);
        assert_eq!(value_group.1.base_offset, 0xC0000);
    }

    #[test]
    fn integration_shim_lock_profile() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let shim = db.module("shim").expect("shim module should exist");
        let profile = profile_from_module(shim, "Lock")
            .expect("Lock profile should be extracted from shim");

        assert_eq!(profile.kind, SubsystemKind::Lock);
        assert_eq!(profile.instance_count, 16);

        // Shim has only 6 event selectors, not 8
        let event_group = profile
            .register_groups
            .iter()
            .find(|(_, g)| g.pattern == "Locks_Event_Selection_{}")
            .expect("Event selection group should exist for shim");
        assert_eq!(event_group.1.instance_count(), 6);
    }

    // ====================================================================
    // DMA pattern tests -- multi-index stress testing
    // ====================================================================

    // ====================================================================
    // Index extraction tests
    // ====================================================================

    #[test]
    fn extract_indices_single_index() {
        assert_eq!(
            extract_indices("Lock15_value", "Lock{}_value"),
            Some(vec![15])
        );
    }

    #[test]
    fn extract_indices_multi_index() {
        assert_eq!(
            extract_indices("DMA_BD15_5", "DMA_BD{}_{}"),
            Some(vec![15, 5])
        );
        assert_eq!(
            extract_indices("DMA_BD0_0", "DMA_BD{}_{}"),
            Some(vec![0, 0])
        );
        assert_eq!(
            extract_indices("DMA_BD47_7", "DMA_BD{}_{}"),
            Some(vec![47, 7])
        );
    }

    #[test]
    fn extract_indices_no_match_returns_none() {
        // Pattern mismatch
        assert_eq!(extract_indices("Lock15_value", "DMA_BD{}_{}"), None);
        // Extra characters in name
        assert_eq!(extract_indices("Lock15_value_extra", "Lock{}_value"), None);
    }

    #[test]
    fn placeholder_count_works() {
        assert_eq!(placeholder_count("Lock{}_value"), 1);
        assert_eq!(placeholder_count("DMA_BD{}_{}"), 2);
        assert_eq!(placeholder_count("Lock_Request"), 0);
        assert_eq!(placeholder_count("A{}B{}C{}"), 3);
    }

    #[test]
    fn detects_2d_packed_structure() {
        // Perfectly packed 2D: 4 BDs x 8 words, stride 4, no gaps.
        // This tests the index extraction fallback (no stride breaks).
        let regs: Vec<RegisterDef> = (0..4)
            .flat_map(|bd| {
                (0..8).map(move |word| {
                    make_reg(
                        &format!("DMA_BD{}_{}", bd, word),
                        0x1D000 + bd * 0x20 + word * 0x4,
                        &["field"],
                    )
                })
            })
            .collect();

        let groups = group_registers(&regs);
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!(g.dimensions.len(), 2, "packed 2D should be detected");
        assert_eq!(g.dimensions[0].count, 4, "4 BDs");
        assert_eq!(g.dimensions[0].stride, 0x20);
        assert_eq!(g.dimensions[1].count, 8, "8 words per BD");
        assert_eq!(g.dimensions[1].stride, 0x4);
    }

    // ====================================================================
    // DMA pattern tests -- multi-index stress testing
    // ====================================================================

    #[test]
    fn name_to_pattern_handles_multi_index_bd_registers() {
        // DMA BD registers have TWO indices: BD number and word number.
        // name_to_pattern replaces ALL digit runs with {}.
        assert_eq!(name_to_pattern("DMA_BD0_0"), "DMA_BD{}_{}");
        assert_eq!(name_to_pattern("DMA_BD15_5"), "DMA_BD{}_{}");
        assert_eq!(name_to_pattern("DMA_BD47_7"), "DMA_BD{}_{}");
    }

    #[test]
    fn name_to_pattern_treats_s2mm_digits_as_index() {
        // The "2" in S2MM is part of the fixed name (Stream-to-Memory-Mapped),
        // NOT an index. But name_to_pattern doesn't know that.
        assert_eq!(name_to_pattern("DMA_S2MM_0_Ctrl"), "DMA_S{}MM_{}_Ctrl");
        assert_eq!(name_to_pattern("DMA_MM2S_0_Ctrl"), "DMA_MM{}S_{}_Ctrl");
    }

    #[test]
    fn detects_2d_structure_in_bd_registers() {
        // 16 BDs x 6 words = 96 BD registers (compute tile layout).
        // Multi-dimensional detection should find the 2D structure:
        //   outer dimension: 16 BDs at stride 0x20
        //   inner dimension: 6 words at stride 0x4
        let regs: Vec<RegisterDef> = (0..16)
            .flat_map(|bd| {
                (0..6).map(move |word| {
                    make_reg(
                        &format!("DMA_BD{}_{}", bd, word),
                        0x1D000 + bd * 0x20 + word * 0x4,
                        &["field"],
                    )
                })
            })
            .collect();

        let groups = group_registers(&regs);

        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!(g.pattern, "DMA_BD{}_{}");

        // Should detect 2 dimensions, not a flat count of 96
        assert_eq!(
            g.dimensions.len(),
            2,
            "BD registers should have 2 dimensions, got {:?}",
            g.dimensions
        );

        // Outer dimension: 16 BDs, stride 0x20
        assert_eq!(g.dimensions[0].count, 16);
        assert_eq!(g.dimensions[0].stride, 0x20);

        // Inner dimension: 6 words per BD, stride 0x4
        assert_eq!(g.dimensions[1].count, 6);
        assert_eq!(g.dimensions[1].stride, 0x4);

        // Total count is product of dimensions
        assert_eq!(g.total_count(), 96);

        // Instance count is the outermost dimension
        assert_eq!(g.instance_count(), 16);
    }

    #[test]
    fn groups_dma_channel_registers() {
        // S2MM channel control registers: 2 channels
        let regs = vec![
            make_reg("DMA_S2MM_0_Ctrl", 0x1DE00, &["Enable", "Reset"]),
            make_reg("DMA_S2MM_1_Ctrl", 0x1DE08, &["Enable", "Reset"]),
            make_reg("DMA_S2MM_0_Start_Queue", 0x1DE04, &["BD_ID"]),
            make_reg("DMA_S2MM_1_Start_Queue", 0x1DE0C, &["BD_ID"]),
            make_reg("DMA_MM2S_0_Ctrl", 0x1DE10, &["Enable", "Reset"]),
            make_reg("DMA_MM2S_1_Ctrl", 0x1DE18, &["Enable", "Reset"]),
        ];

        let groups = group_registers(&regs);

        // "2" in S2MM/MM2S treated as index, creating patterns like
        // DMA_S{}MM_{}_Ctrl. Both S2MM_0 and S2MM_1 share this pattern.
        let ctrl_s2mm = groups.iter().find(|g| g.pattern == "DMA_S{}MM_{}_Ctrl");
        assert!(
            ctrl_s2mm.is_some(),
            "S2MM Ctrl should group under DMA_S{{}}MM_{{}}_Ctrl, got patterns: {:?}",
            groups.iter().map(|g| &g.pattern).collect::<Vec<_>>()
        );
        assert_eq!(ctrl_s2mm.unwrap().total_count(), 2);

        let ctrl_mm2s = groups.iter().find(|g| g.pattern == "DMA_MM{}S_{}_Ctrl");
        assert!(
            ctrl_mm2s.is_some(),
            "MM2S Ctrl should group under DMA_MM{{}}S_{{}}_Ctrl"
        );
        assert_eq!(ctrl_mm2s.unwrap().total_count(), 2);
    }

    // ====================================================================
    // Module-level extraction tests
    // ====================================================================

    #[test]
    fn integration_extract_memory_module_profiles() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let mem = db.module("memory").expect("memory module should exist");
        let profiles = extract_module_profiles(&mem.registers);

        // Should find DMA and Lock subsystems (among others)
        let dma = profiles.iter().find(|p| p.subsystem == "DMA");
        let lock = profiles.iter().find(|p| p.subsystem == "Lock");

        assert!(dma.is_some(), "DMA profile should be extracted");
        assert!(lock.is_some(), "Lock profile should be extracted");

        // DMA: 16 BDs
        assert_eq!(dma.unwrap().instance_count, 16, "DMA: 16 BD instances");

        // Lock: 16 instances (same as old test)
        assert_eq!(lock.unwrap().instance_count, 16, "Lock: 16 lock instances");

        // Lock_Request should be in the Lock profile (not a separate subsystem)
        let lock_p = lock.unwrap();
        assert!(
            lock_p.singletons.iter().any(|(_, name)| name == "Lock_Request"),
            "Lock_Request should be in Lock profile"
        );

        // Locks_Overflow should be in Lock profile (alias merge)
        assert!(
            lock_p.singletons.iter().any(|(_, name)| name == "Locks_Overflow"),
            "Locks_Overflow should be in Lock profile (via alias merge)"
        );

        // Report all discovered subsystems
        eprintln!("=== Memory Module Subsystems ===");
        for p in &profiles {
            eprintln!(
                "  {} -- {} instances, {} groups, {} singletons",
                p.subsystem,
                p.instance_count,
                p.register_groups.len(),
                p.singletons.len()
            );
        }
    }

    #[test]
    fn integration_extract_memtile_profiles() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let mt = db.module("memory_tile").expect("memory_tile module should exist");
        let profiles = extract_module_profiles(&mt.registers);

        let dma = profiles.iter().find(|p| p.subsystem == "DMA");
        let lock = profiles.iter().find(|p| p.subsystem == "Lock");

        assert!(dma.is_some(), "DMA profile should be extracted from memtile");
        assert!(lock.is_some(), "Lock profile should be extracted from memtile");

        // MemTile: 48 BDs, 64 locks
        assert_eq!(dma.unwrap().instance_count, 48, "MemTile DMA: 48 BDs");
        assert_eq!(lock.unwrap().instance_count, 64, "MemTile Lock: 64 instances");

        eprintln!("=== MemTile Subsystems ===");
        for p in &profiles {
            eprintln!(
                "  {} -- {} instances, {} groups, {} singletons",
                p.subsystem,
                p.instance_count,
                p.register_groups.len(),
                p.singletons.len()
            );
        }
    }

    #[test]
    fn integration_extract_core_module_profiles() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let core = db.module("core").expect("core module should exist");
        let profiles = extract_module_profiles(&core.registers);

        // Core should NOT have Lock or DMA subsystems
        let lock = profiles.iter().find(|p| p.subsystem == "Lock");
        let dma = profiles.iter().find(|p| p.subsystem == "DMA");
        assert!(lock.is_none(), "Core should have no Lock subsystem");
        assert!(dma.is_none(), "Core should have no DMA subsystem");

        // Event subsystem should exist and include Combo/Edge aliases
        let event = profiles.iter().find(|p| p.subsystem == "Event");
        assert!(event.is_some(), "Core should have Event subsystem");

        eprintln!("=== Core Module Subsystems ===");
        for p in &profiles {
            eprintln!(
                "  {} -- {} instances, {} groups, {} singletons",
                p.subsystem,
                p.instance_count,
                p.register_groups.len(),
                p.singletons.len()
            );
        }
    }

    #[test]
    fn integration_extract_shim_profiles() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let shim = db.module("shim").expect("shim module should exist");
        let profiles = extract_module_profiles(&shim.registers);

        let dma = profiles.iter().find(|p| p.subsystem == "DMA");
        let lock = profiles.iter().find(|p| p.subsystem == "Lock");

        assert!(dma.is_some(), "Shim DMA profile should be extracted");
        assert!(lock.is_some(), "Shim Lock profile should be extracted");

        // Shim: 16 BDs, 16 locks
        assert_eq!(dma.unwrap().instance_count, 16, "Shim DMA: 16 BDs");
        assert_eq!(lock.unwrap().instance_count, 16, "Shim Lock: 16 instances");

        eprintln!("=== Shim Subsystems ===");
        for p in &profiles {
            eprintln!(
                "  {} -- {} instances, {} groups, {} singletons",
                p.subsystem,
                p.instance_count,
                p.register_groups.len(),
                p.singletons.len()
            );
        }
    }

    // ====================================================================
    // SubsystemKind mapping tests
    // ====================================================================

    #[test]
    fn maps_primary_subsystem_names_to_kind() {
        // These are the core subsystems that aie-rt defines as distinct modules.
        assert_eq!(subsystem_name_to_kind("DMA"), SubsystemKind::Dma);
        assert_eq!(subsystem_name_to_kind("Lock"), SubsystemKind::Lock);
        assert_eq!(subsystem_name_to_kind("Event"), SubsystemKind::Event);
        assert_eq!(subsystem_name_to_kind("Trace"), SubsystemKind::Trace);
        assert_eq!(subsystem_name_to_kind("Performance"), SubsystemKind::Performance);
        assert_eq!(subsystem_name_to_kind("Timer"), SubsystemKind::Timer);
        assert_eq!(subsystem_name_to_kind("Stream"), SubsystemKind::StreamSwitch);
    }

    #[test]
    fn maps_hardware_grouping_names_to_kind() {
        // Real hardware groupings visible in registers but not aie-rt modules.
        assert_eq!(subsystem_name_to_kind("WatchPoint"), SubsystemKind::WatchPoint);
        assert_eq!(subsystem_name_to_kind("Debug"), SubsystemKind::Debug);
        assert_eq!(subsystem_name_to_kind("PC"), SubsystemKind::ProgramCounter);
        assert_eq!(subsystem_name_to_kind("Interrupt"), SubsystemKind::Interrupt);
        assert_eq!(subsystem_name_to_kind("NoC"), SubsystemKind::NoC);
        assert_eq!(subsystem_name_to_kind("DataMemory"), SubsystemKind::DataMemory);
    }

    #[test]
    fn maps_core_processor_names_to_kind() {
        // "Core" in the core module's registers = processor control regs.
        // "Program" = program memory.
        assert_eq!(subsystem_name_to_kind("Core"), SubsystemKind::Processor);
        assert_eq!(subsystem_name_to_kind("Program"), SubsystemKind::ProgramMemory);
    }

    #[test]
    fn profile_carries_subsystem_kind() {
        // extract_module_profiles should populate both the string name
        // and the typed SubsystemKind on each profile.
        let mut regs = Vec::new();
        for i in 0..16u32 {
            regs.push(make_reg(
                &format!("Lock{}_value", i),
                0x1F000 + i * 0x10,
                &["Lock_value"],
            ));
        }
        regs.push(make_reg("DMA_BD0_0", 0x1D000, &["field"]));
        regs.push(make_reg("DMA_BD0_1", 0x1D004, &["field"]));

        let profiles = extract_module_profiles(&regs);
        let lock_p = profiles.iter().find(|p| p.subsystem == "Lock").unwrap();
        let dma_p = profiles.iter().find(|p| p.subsystem == "DMA").unwrap();

        assert_eq!(lock_p.kind, SubsystemKind::Lock);
        assert_eq!(dma_p.kind, SubsystemKind::Dma);
    }

    #[test]
    fn profile_unknown_kind_for_naming_artifacts() {
        // Register name stems that don't map to real subsystems get Unknown.
        let regs = vec![
            make_reg("Tile_Status", 0x0, &["field"]),
            make_reg("Tile_Control", 0x4, &["field"]),
        ];

        let profiles = extract_module_profiles(&regs);
        let tile_p = profiles.iter().find(|p| p.subsystem == "Tile").unwrap();
        assert_eq!(tile_p.kind, SubsystemKind::Unknown);
    }

    #[test]
    fn maps_naming_artifacts_to_unknown() {
        // These stem from register naming conventions, not real subsystems.
        // They get Unknown until a future pass assigns them properly.
        assert_eq!(subsystem_name_to_kind("AIE"), SubsystemKind::Unknown);
        assert_eq!(subsystem_name_to_kind("Column"), SubsystemKind::Unknown);
        assert_eq!(subsystem_name_to_kind("Tile"), SubsystemKind::Unknown);
        assert_eq!(subsystem_name_to_kind("Module"), SubsystemKind::Unknown);
        assert_eq!(subsystem_name_to_kind("Enable"), SubsystemKind::Unknown);
        assert_eq!(subsystem_name_to_kind("Reset"), SubsystemKind::Unknown);
        assert_eq!(subsystem_name_to_kind("SomeFutureSubsystem"), SubsystemKind::Unknown);
    }

    // ====================================================================
    // DMA integration tests -- real AM025 data
    // ====================================================================

    #[test]
    fn integration_memory_module_dma_profile() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let mem = db.module("memory").expect("memory module should exist");
        let profile = profile_from_module(mem, "DMA")
            .expect("DMA profile should be extracted from memory module");

        assert_eq!(profile.kind, SubsystemKind::Dma);

        // BD group should exist with 2D structure
        let bd_group = profile
            .register_groups
            .iter()
            .find(|(_, g)| g.pattern == "DMA_BD{}_{}")
            .expect("BD group should exist");
        assert_eq!(bd_group.0, RegisterCategory::State, "BDs are R/W grouped");
        assert_eq!(bd_group.1.dimensions.len(), 2, "BDs should be 2D");
        assert_eq!(bd_group.1.instance_count(), 16, "16 BDs");
        assert_eq!(bd_group.1.total_count(), 96, "16 BDs x 6 words = 96");

        // Channel groups should exist
        let s2mm_ctrl = profile
            .register_groups
            .iter()
            .find(|(_, g)| g.pattern.contains("S{}MM") && g.pattern.contains("Ctrl"));
        assert!(
            s2mm_ctrl.is_some(),
            "S2MM Ctrl group should exist in DMA profile"
        );

        assert_eq!(profile.instance_count, 16, "DMA instance count = BD count");
    }

    #[test]
    fn integration_memtile_dma_profile() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let mt = db.module("memory_tile").expect("memory_tile module should exist");
        let profile = profile_from_module(mt, "DMA")
            .expect("DMA profile should be extracted from memtile");

        assert_eq!(profile.kind, SubsystemKind::Dma);

        // 48 BDs x 8 words = 384 BD registers, detected as 2D
        let bd_group = profile
            .register_groups
            .iter()
            .find(|(_, g)| g.pattern == "DMA_BD{}_{}")
            .expect("BD group should exist in memtile");
        assert_eq!(bd_group.1.dimensions.len(), 2, "MemTile BDs should be 2D");
        assert_eq!(bd_group.1.instance_count(), 48, "48 BDs");
        assert_eq!(bd_group.1.total_count(), 384, "48 BDs x 8 words = 384");
        assert_eq!(profile.instance_count, 48, "MemTile DMA: 48 BD instances");
    }

    #[test]
    fn integration_shim_dma_profile() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let shim = db.module("shim").expect("shim module should exist");
        let profile = profile_from_module(shim, "DMA")
            .expect("DMA profile should be extracted from shim");

        assert_eq!(profile.kind, SubsystemKind::Dma);

        // Shim: 16 BDs x 8 words = 128 BD registers, 2D
        let bd_group = profile
            .register_groups
            .iter()
            .find(|(_, g)| g.pattern == "DMA_BD{}_{}")
            .expect("BD group should exist in shim");
        assert_eq!(bd_group.1.dimensions.len(), 2, "Shim BDs should be 2D");
        assert_eq!(bd_group.1.instance_count(), 16, "16 BDs");
        assert_eq!(bd_group.1.total_count(), 128, "16 BDs x 8 words = 128");
        assert_eq!(profile.instance_count, 16, "Shim DMA: 16 BD instances");

        // Shim uses "Task_Queue" instead of "Start_Queue"
        let has_task_queue = profile
            .register_groups
            .iter()
            .any(|(_, g)| g.pattern.contains("Task_Queue"));
        assert!(has_task_queue, "Shim should have Task_Queue registers");
    }

    #[test]
    fn integration_all_profiles_have_typed_kinds() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let module_names = ["core", "memory", "memory_tile", "shim"];
        for module_name in &module_names {
            let module = db.module(module_name).unwrap();
            let profiles = extract_module_profiles(&module.registers);

            // Every real subsystem should get a non-Unknown kind.
            // Naming artifacts are allowed to be Unknown.
            let known: Vec<_> = profiles.iter().filter(|p| p.kind != SubsystemKind::Unknown).collect();
            let unknown: Vec<_> = profiles.iter().filter(|p| p.kind == SubsystemKind::Unknown).collect();

            eprintln!("=== {} ===", module_name);
            for p in &profiles {
                eprintln!("  {} -> {:?} ({} instances)", p.subsystem, p.kind, p.instance_count);
            }

            // The major subsystems must all be typed (not Unknown).
            // Each module should have at least Event, Trace, Performance, Timer.
            let required = ["Event", "Trace", "Performance", "Timer"];
            for name in &required {
                let found = profiles.iter().find(|p| p.subsystem == *name);
                assert!(found.is_some(), "{} should have {} subsystem", module_name, name);
                assert_ne!(
                    found.unwrap().kind, SubsystemKind::Unknown,
                    "{}: {} should have a typed kind", module_name, name
                );
            }

            // Unknown profiles should only be naming artifacts (low register count).
            // Real subsystems have at least a few registers.
            for p in &unknown {
                assert!(
                    p.instance_count <= 2 && p.register_groups.len() <= 1,
                    "{}: '{}' is Unknown but has {} instances and {} groups -- should it be mapped?",
                    module_name, p.subsystem, p.instance_count, p.register_groups.len()
                );
            }

            eprintln!(
                "  -> {} known, {} unknown (naming artifacts)",
                known.len(),
                unknown.len()
            );
        }
    }

    #[test]
    fn integration_populate_tile_modules() {
        use crate::device_model;
        use std::path::PathBuf;

        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let json_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/aie-device-models.json");
        let mut model = device_model::extract_device_model(&json_path, "npu1")
            .expect("device model parse failed");

        // Before: modules are empty
        let core_tt = model.tile_types.iter().find(|t| t.name == "core").unwrap();
        assert!(core_tt.modules.is_empty(), "modules should start empty");

        // Populate
        populate_tile_modules(&mut model, &db);

        // Compute tile gets 2 modules (Core + Memory)
        let core_tt = model.tile_types.iter().find(|t| t.name == "core").unwrap();
        assert_eq!(core_tt.modules.len(), 2, "compute tile: Core + Memory");
        assert!(core_tt.modules.iter().any(|m| m.kind == ModuleKind::Core));
        assert!(core_tt.modules.iter().any(|m| m.kind == ModuleKind::Memory));

        // Core module should have registers with Processor subsystem
        let core_mod = core_tt.modules.iter().find(|m| m.kind == ModuleKind::Core).unwrap();
        assert!(core_mod.registers.iter().any(|r| r.subsystem == SubsystemKind::Processor));
        assert!(!core_mod.registers.iter().any(|r| r.subsystem == SubsystemKind::Lock));

        // Memory module should have Lock and DMA
        let mem_mod = core_tt.modules.iter().find(|m| m.kind == ModuleKind::Memory).unwrap();
        assert!(mem_mod.registers.iter().any(|r| r.subsystem == SubsystemKind::Lock));
        assert!(mem_mod.registers.iter().any(|r| r.subsystem == SubsystemKind::Dma));

        // MemTile gets 1 module
        let mt_tt = model.tile_types.iter().find(|t| t.name == "mem_tile").unwrap();
        assert_eq!(mt_tt.modules.len(), 1);
        assert_eq!(mt_tt.modules[0].kind, ModuleKind::MemTile);

        // Shim gets 1 module
        let shim_tt = model.tile_types.iter().find(|t| t.name == "shim_noc").unwrap();
        assert_eq!(shim_tt.modules.len(), 1);
        assert_eq!(shim_tt.modules[0].kind, ModuleKind::Shim);
    }

    #[test]
    fn integration_core_has_no_locks() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let core = db.module("core").expect("core module should exist");

        // Core module has NO lock registers (locks are in the memory module)
        let lock = profile_from_module(core, "Lock");
        assert!(
            lock.is_none(),
            "Core module should have no Lock subsystem"
        );
    }

    // ====================================================================
    // Register-to-graph wiring tests
    // ====================================================================

    #[test]
    fn converts_regdb_register_to_register_model() {
        let reg = make_reg_with_access(
            "Lock0_value",
            0x1F000,
            AccessMode::ReadWrite,
            &["Lock_value"],
        );

        let model = register_def_to_model(&reg, "memory");

        assert_eq!(model.name, "Lock0_value");
        assert_eq!(model.offset, 0x1F000);
        assert_eq!(model.width, 32);
        assert_eq!(model.reset_value, 0);
        assert_eq!(model.access, Access::ReadWrite);
        assert_eq!(model.subsystem, SubsystemKind::Lock);
        assert_eq!(model.source.origin, Source::Am025Json);
        assert_eq!(model.fields.len(), 1);
        assert_eq!(model.fields[0].name, "Lock_value");
    }

    #[test]
    fn converts_readonly_access_mode() {
        let reg = make_reg_with_access(
            "Lock_Request",
            0x40000,
            AccessMode::ReadOnly,
            &["Request_Result"],
        );

        let model = register_def_to_model(&reg, "memory");
        assert_eq!(model.access, crate::types::Access::ReadOnly);
        assert_eq!(model.subsystem, SubsystemKind::Lock);
    }

    #[test]
    fn builds_module_model_from_regdb() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let mem = db.module("memory").expect("memory module should exist");
        let module_model = build_module_model(ModuleKind::Memory, mem);

        assert_eq!(module_model.kind, ModuleKind::Memory);
        assert_eq!(module_model.source.origin, crate::types::Source::Am025Json);

        // Every register in the AM025 memory module should be converted
        assert_eq!(
            module_model.registers.len(),
            mem.registers.len(),
            "all registers should be converted"
        );

        // Spot-check: Lock0_value should have SubsystemKind::Lock
        let lock0 = module_model.registers.iter()
            .find(|r| r.name == "Lock0_value")
            .expect("Lock0_value should exist");
        assert_eq!(lock0.subsystem, SubsystemKind::Lock);
        assert_eq!(lock0.offset, 0x1F000);

        // DMA_BD0_0 should have SubsystemKind::Dma
        let dma0 = module_model.registers.iter()
            .find(|r| r.name == "DMA_BD0_0")
            .expect("DMA_BD0_0 should exist");
        assert_eq!(dma0.subsystem, SubsystemKind::Dma);
    }

    #[test]
    fn integration_populate_confirms_instance_counts() {
        use crate::device_model;
        use crate::types::Source;
        use std::path::PathBuf;

        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let json_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/aie-device-models.json");
        let mut model = device_model::extract_device_model(&json_path, "npu1")
            .expect("device model parse failed");

        // Before populate: instances come from device model only (1 source each)
        let core = model.tile_types.iter().find(|t| t.name == "core").unwrap();
        assert_eq!(core.instances.locks.sources().len(), 1);
        assert_eq!(core.instances.bds.sources().len(), 1);
        assert_eq!(core.instances.channels.sources().len(), 1);

        populate_tile_modules(&mut model, &db);

        // After populate: locks and bds confirmed by AM025 (2 sources),
        // channels has no register-level signal (stays at 1 source)
        let core = model.tile_types.iter().find(|t| t.name == "core").unwrap();
        assert_eq!(core.instances.locks.sources().len(), 2, "locks should be confirmed by AM025");
        assert_eq!(core.instances.bds.sources().len(), 2, "bds should be confirmed by AM025");
        assert_eq!(core.instances.channels.sources().len(), 1, "channels has no register signal");

        // Values should match what device model provided
        assert_eq!(*core.instances.locks.value(), 16);
        assert_eq!(*core.instances.bds.value(), 16);

        // Second source should be AM025
        assert_eq!(core.instances.locks.sources()[1].origin, Source::Am025Json);
        assert_eq!(core.instances.bds.sources()[1].origin, Source::Am025Json);

        // MemTile too
        let mt = model.tile_types.iter().find(|t| t.name == "mem_tile").unwrap();
        assert_eq!(mt.instances.locks.sources().len(), 2);
        assert_eq!(mt.instances.bds.sources().len(), 2);
        assert_eq!(*mt.instances.locks.value(), 64);
        assert_eq!(*mt.instances.bds.value(), 48);

        // Shim too
        let shim = model.tile_types.iter().find(|t| t.name == "shim_noc").unwrap();
        assert_eq!(shim.instances.locks.sources().len(), 2);
        assert_eq!(shim.instances.bds.sources().len(), 2);
        assert_eq!(*shim.instances.locks.value(), 16);
        assert_eq!(*shim.instances.bds.value(), 16);
    }

    // ====================================================================
    // BD schema extraction tests
    // ====================================================================

    #[test]
    fn extract_compute_bd_schema() {
        use crate::types::BdFieldRole;

        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let json_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/aie-device-models.json");
        let mut model =
            crate::device_model::extract_device_model(&json_path, "npu1").unwrap();
        populate_tile_modules(&mut model, &db);

        let core = model.tile_types.iter().find(|t| t.name == "core").unwrap();
        let schema = core.bd_schema.as_ref().expect("compute tile should have BD schema");

        // Compute tile: 6 words per BD, 3 addressing dimensions
        assert_eq!(schema.words_per_bd, 6);
        assert_eq!(schema.num_dimensions, 3);
        assert!(!schema.has_zero_padding, "compute tile has no zero padding");
        assert!(!schema.has_axi_fields, "compute tile has no AXI fields");

        // Check key fields exist with correct bit widths
        let buf_len = schema.field(&BdFieldRole::BufferLength)
            .expect("should have BufferLength");
        assert_eq!(buf_len.word, 0);
        assert_eq!(buf_len.bits.width(), 14);

        let base_addr = schema.field(&BdFieldRole::BaseAddress)
            .expect("should have BaseAddress");
        assert_eq!(base_addr.word, 0);
        assert_eq!(base_addr.bits.width(), 14);

        let d0_step = schema.field(&BdFieldRole::Stepsize(0))
            .expect("should have D0_Stepsize");
        assert_eq!(d0_step.bits.width(), 13);

        let lock_acq_id = schema.field(&BdFieldRole::LockAcqId)
            .expect("should have LockAcqId");
        assert_eq!(lock_acq_id.bits.width(), 4);

        // No shim-only or memtile-only fields
        assert!(schema.field(&BdFieldRole::BaseAddressHigh).is_none());
        assert!(schema.field(&BdFieldRole::ZeroPadBefore(0)).is_none());
        assert!(schema.field(&BdFieldRole::BurstLength).is_none());
    }

    #[test]
    fn extract_memtile_bd_schema() {
        use crate::types::BdFieldRole;

        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let json_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/aie-device-models.json");
        let mut model =
            crate::device_model::extract_device_model(&json_path, "npu1").unwrap();
        populate_tile_modules(&mut model, &db);

        let mt = model.tile_types.iter().find(|t| t.name == "mem_tile").unwrap();
        let schema = mt.bd_schema.as_ref().expect("memtile should have BD schema");

        // MemTile: 8 words per BD, 4 addressing dimensions, zero padding
        assert_eq!(schema.words_per_bd, 8);
        assert_eq!(schema.num_dimensions, 4);
        assert!(schema.has_zero_padding, "memtile has zero padding");
        assert!(!schema.has_axi_fields, "memtile has no AXI fields");

        // Wider fields than compute
        let buf_len = schema.field(&BdFieldRole::BufferLength).unwrap();
        assert_eq!(buf_len.bits.width(), 17);

        let base_addr = schema.field(&BdFieldRole::BaseAddress).unwrap();
        assert_eq!(base_addr.bits.width(), 19);

        let d0_step = schema.field(&BdFieldRole::Stepsize(0)).unwrap();
        assert_eq!(d0_step.bits.width(), 17);

        // Memtile has D3 stepsize (4th dimension)
        assert!(schema.field(&BdFieldRole::Stepsize(3)).is_some());

        // Zero padding fields present
        assert!(schema.field(&BdFieldRole::ZeroPadBefore(0)).is_some());
        assert!(schema.field(&BdFieldRole::ZeroPadAfter(0)).is_some());

        // 8-bit lock IDs (192-entry address space for cross-column)
        let lock_acq_id = schema.field(&BdFieldRole::LockAcqId).unwrap();
        assert_eq!(lock_acq_id.bits.width(), 8);
    }

    #[test]
    fn extract_shim_bd_schema() {
        use crate::types::BdFieldRole;

        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let json_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/aie-device-models.json");
        let mut model =
            crate::device_model::extract_device_model(&json_path, "npu1").unwrap();
        populate_tile_modules(&mut model, &db);

        let shim = model.tile_types.iter().find(|t| t.name == "shim_noc").unwrap();
        let schema = shim.bd_schema.as_ref().expect("shim should have BD schema");

        // Shim: 8 words per BD, 3 addressing dimensions, AXI fields
        assert_eq!(schema.words_per_bd, 8);
        assert_eq!(schema.num_dimensions, 3);
        assert!(!schema.has_zero_padding, "shim has no zero padding");
        assert!(schema.has_axi_fields, "shim has AXI fields");

        // 32-bit buffer length (full word), 46-bit split address
        let buf_len = schema.field(&BdFieldRole::BufferLength).unwrap();
        assert_eq!(buf_len.bits.width(), 32);

        let base_low = schema.field(&BdFieldRole::BaseAddress).unwrap();
        assert_eq!(base_low.bits.width(), 30);

        let base_high = schema.field(&BdFieldRole::BaseAddressHigh)
            .expect("shim has split base address");
        assert_eq!(base_high.bits.width(), 16);

        // AXI-specific fields
        assert!(schema.field(&BdFieldRole::BurstLength).is_some());
        assert!(schema.field(&BdFieldRole::Smid).is_some());
        assert!(schema.field(&BdFieldRole::AxCache).is_some());
        assert!(schema.field(&BdFieldRole::SecureAccess).is_some());

        // 4-bit lock IDs (same as compute)
        let lock_acq_id = schema.field(&BdFieldRole::LockAcqId).unwrap();
        assert_eq!(lock_acq_id.bits.width(), 4);
    }

    // ====================================================================
    // DMA channel schema extraction tests
    // ====================================================================

    #[test]
    fn extract_compute_channel_schema() {
        use crate::types::{DmaChannelFieldRole, DmaChannelRegKind, DmaDirection};

        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let json_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/aie-device-models.json");
        let mut model =
            crate::device_model::extract_device_model(&json_path, "npu1").unwrap();
        populate_tile_modules(&mut model, &db);

        let core = model.tile_types.iter().find(|t| t.name == "core").unwrap();
        let schema = core
            .channel_schema
            .as_ref()
            .expect("compute tile should have channel schema");

        // Compute tile: 2 channels per direction
        assert_eq!(schema.channels_per_direction, 2);

        // 4-bit Start_BD_ID (16 BDs)
        assert_eq!(schema.start_bd_id_width, 4);

        // Compute tile supports compression, OOO, FoT but not pause or AXI errors
        assert!(schema.has_compression, "compute has compression");
        assert!(schema.has_out_of_order, "compute has out-of-order");
        assert!(schema.has_fot, "compute has FoT");
        assert!(!schema.has_pause, "compute has no pause");
        assert!(!schema.has_axi_errors, "compute has no AXI errors");
        assert!(!schema.has_cross_tile_errors, "compute has no cross-tile errors");

        // Queue register named "Start_Queue"
        assert_eq!(schema.queue_name, "Start_Queue");

        // S2MM Control fields
        let s2mm_ctrl = schema.fields_for(DmaChannelRegKind::Control, DmaDirection::S2mm);
        assert!(
            s2mm_ctrl.iter().any(|f| f.role == DmaChannelFieldRole::FotMode),
            "S2MM Ctrl should have FoT_Mode"
        );
        assert!(
            s2mm_ctrl.iter().any(|f| f.role == DmaChannelFieldRole::DecompressionEnable),
            "S2MM Ctrl should have Decompression_Enable"
        );
        assert!(
            s2mm_ctrl.iter().any(|f| f.role == DmaChannelFieldRole::EnableOutOfOrder),
            "S2MM Ctrl should have Enable_Out_of_Order"
        );

        // MM2S Control fields -- has Compression, no FoT/OOO
        let mm2s_ctrl = schema.fields_for(DmaChannelRegKind::Control, DmaDirection::Mm2s);
        assert!(
            mm2s_ctrl.iter().any(|f| f.role == DmaChannelFieldRole::CompressionEnable),
            "MM2S Ctrl should have Compression_Enable"
        );
        assert!(
            !mm2s_ctrl.iter().any(|f| f.role == DmaChannelFieldRole::FotMode),
            "MM2S Ctrl should NOT have FoT_Mode"
        );

        // Start Queue: BD_ID width = 4 bits
        let s2mm_queue = schema.fields_for(DmaChannelRegKind::StartQueue, DmaDirection::S2mm);
        let bd_id_field = s2mm_queue
            .iter()
            .find(|f| f.role == DmaChannelFieldRole::StartBdId)
            .expect("StartQueue should have Start_BD_ID");
        assert_eq!(bd_id_field.bits.width(), 4);

        // Status: has Channel_Running, Cur_BD (4-bit)
        let s2mm_status = schema.fields_for(DmaChannelRegKind::Status, DmaDirection::S2mm);
        assert!(s2mm_status.iter().any(|f| f.role == DmaChannelFieldRole::ChannelRunning));
        let cur_bd = s2mm_status
            .iter()
            .find(|f| f.role == DmaChannelFieldRole::CurrentBdId)
            .expect("Status should have Cur_BD");
        assert_eq!(cur_bd.bits.width(), 4);

        // S2MM-specific stall
        assert!(s2mm_status.iter().any(|f| f.role == DmaChannelFieldRole::StalledStreamStarvation));

        // MM2S-specific stall
        let mm2s_status = schema.fields_for(DmaChannelRegKind::Status, DmaDirection::Mm2s);
        assert!(mm2s_status.iter().any(|f| f.role == DmaChannelFieldRole::StalledStreamBackpressure));
    }

    #[test]
    fn extract_memtile_channel_schema() {
        use crate::types::{DmaChannelFieldRole, DmaChannelRegKind, DmaDirection};

        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let json_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/aie-device-models.json");
        let mut model =
            crate::device_model::extract_device_model(&json_path, "npu1").unwrap();
        populate_tile_modules(&mut model, &db);

        let memtile = model.tile_types.iter().find(|t| t.name == "mem_tile").unwrap();
        let schema = memtile
            .channel_schema
            .as_ref()
            .expect("memtile should have channel schema");

        // MemTile: 6 channels per direction
        assert_eq!(schema.channels_per_direction, 6);

        // 6-bit Start_BD_ID (64 BDs)
        assert_eq!(schema.start_bd_id_width, 6);

        // MemTile has cross-tile errors
        assert!(schema.has_cross_tile_errors, "memtile has cross-tile errors");
        assert!(!schema.has_axi_errors, "memtile has no AXI errors");
        assert!(!schema.has_pause, "memtile has no pause");

        // Queue register named "Start_Queue"
        assert_eq!(schema.queue_name, "Start_Queue");

        // Status Cur_BD should be 6-bit
        let s2mm_status = schema.fields_for(DmaChannelRegKind::Status, DmaDirection::S2mm);
        let cur_bd = s2mm_status
            .iter()
            .find(|f| f.role == DmaChannelFieldRole::CurrentBdId)
            .expect("Status should have Cur_BD");
        assert_eq!(cur_bd.bits.width(), 6, "memtile Cur_BD is 6-bit (64 BDs)");

        // Cross-tile error fields
        assert!(
            s2mm_status.iter().any(|f| f.role == DmaChannelFieldRole::ErrorLockAccessUnavailable),
            "memtile S2MM Status should have Error_Lock_Access_to_Unavailable"
        );
        assert!(
            s2mm_status.iter().any(|f| f.role == DmaChannelFieldRole::ErrorDmAccessUnavailable),
            "memtile S2MM Status should have Error_DM_Access_to_Unavailable"
        );

        // Write Count register (S2MM only) -- 17-bit on memtile
        let wc_fields = schema.fields_for(DmaChannelRegKind::WriteCount, DmaDirection::S2mm);
        assert!(!wc_fields.is_empty(), "memtile should have WriteCount registers");
        let wc = wc_fields
            .iter()
            .find(|f| f.role == DmaChannelFieldRole::CurrentWriteCount)
            .expect("WriteCount should have Current_Write_Count");
        assert_eq!(wc.bits.width(), 17, "memtile write count is 17-bit");
    }

    #[test]
    fn extract_shim_channel_schema() {
        use crate::types::{DmaChannelFieldRole, DmaChannelRegKind, DmaDirection};

        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let json_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/aie-device-models.json");
        let mut model =
            crate::device_model::extract_device_model(&json_path, "npu1").unwrap();
        populate_tile_modules(&mut model, &db);

        let shim = model.tile_types.iter().find(|t| t.name == "shim_noc").unwrap();
        let schema = shim
            .channel_schema
            .as_ref()
            .expect("shim should have channel schema");

        // Shim: 2 channels per direction
        assert_eq!(schema.channels_per_direction, 2);

        // 4-bit Start_BD_ID
        assert_eq!(schema.start_bd_id_width, 4);

        // Shim has pause and AXI errors but not compression or cross-tile errors
        assert!(schema.has_pause, "shim has pause");
        assert!(schema.has_axi_errors, "shim has AXI errors");
        assert!(!schema.has_compression, "shim has no compression");
        assert!(!schema.has_cross_tile_errors, "shim has no cross-tile errors");

        // Queue register named "Task_Queue" (different from compute/memtile!)
        assert_eq!(schema.queue_name, "Task_Queue");

        // S2MM Control: Pause_Mem, Pause_Stream, no Reset
        let s2mm_ctrl = schema.fields_for(DmaChannelRegKind::Control, DmaDirection::S2mm);
        assert!(
            s2mm_ctrl.iter().any(|f| f.role == DmaChannelFieldRole::PauseMem),
            "Shim S2MM Ctrl should have Pause_Mem"
        );
        assert!(
            s2mm_ctrl.iter().any(|f| f.role == DmaChannelFieldRole::PauseStream),
            "Shim S2MM Ctrl should have Pause_Stream"
        );
        assert!(
            !s2mm_ctrl.iter().any(|f| f.role == DmaChannelFieldRole::Reset),
            "Shim S2MM Ctrl should NOT have Reset"
        );

        // Status: AXI error fields
        let s2mm_status = schema.fields_for(DmaChannelRegKind::Status, DmaDirection::S2mm);
        assert!(
            s2mm_status.iter().any(|f| f.role == DmaChannelFieldRole::AxiMmDecodeError),
            "Shim Status should have AXI_MM_Decode_Error"
        );
        assert!(
            s2mm_status.iter().any(|f| f.role == DmaChannelFieldRole::AxiMmSlaveError),
            "Shim Status should have AXI_MM_Slave_Error"
        );
    }

    // ====================================================================
    // Edge (relationship) tests -- DerivedFrom wiring
    // ====================================================================

    #[test]
    fn bd_schema_fields_have_derived_from_edges() {
        use crate::types::{BdFieldRole, NodeId, RelationshipKind, TileKind};

        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let json_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/aie-device-models.json");
        let mut model =
            crate::device_model::extract_device_model(&json_path, "npu1").unwrap();
        populate_tile_modules(&mut model, &db);

        // Every BD field in the compute tile schema should have a DerivedFrom
        // edge pointing to the register field it was extracted from.
        let core = model.tile_types.iter().find(|t| t.name == "core").unwrap();
        let schema = core.bd_schema.as_ref().expect("should have BD schema");

        let derived_edges: Vec<_> = model
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::DerivedFrom)
            .collect();

        // There should be at least one DerivedFrom edge per non-Reserved BD field
        let non_reserved_fields: Vec<_> = schema
            .fields
            .iter()
            .filter(|f| f.role != BdFieldRole::Reserved)
            .collect();
        assert!(
            !non_reserved_fields.is_empty(),
            "BD schema should have non-reserved fields"
        );

        // Check that Buffer_Length has a DerivedFrom edge to a RegisterField
        let buf_len_node = NodeId::BdField {
            tile: TileKind::Compute,
            role: BdFieldRole::BufferLength,
        };
        let buf_len_edge = derived_edges
            .iter()
            .find(|e| e.from == buf_len_node)
            .expect("Buffer_Length BD field should have a DerivedFrom edge");

        // The target should be a RegisterField in DMA_BD0_0
        match &buf_len_edge.to {
            NodeId::RegisterField { register, field, .. } => {
                assert_eq!(register, "DMA_BD0_0");
                assert_eq!(field, "Buffer_Length");
            }
            other => panic!("Expected RegisterField, got {:?}", other),
        }
    }

    #[test]
    fn channel_schema_fields_have_derived_from_edges() {
        use crate::types::{DmaChannelFieldRole, DmaDirection, NodeId, RelationshipKind, TileKind};

        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let json_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/aie-device-models.json");
        let mut model =
            crate::device_model::extract_device_model(&json_path, "npu1").unwrap();
        populate_tile_modules(&mut model, &db);

        let derived_edges: Vec<_> = model
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::DerivedFrom)
            .collect();

        // Check that S2MM FoT_Mode has a DerivedFrom edge
        let fot_node = NodeId::ChannelField {
            tile: TileKind::Compute,
            direction: DmaDirection::S2mm,
            role: DmaChannelFieldRole::FotMode,
        };
        let fot_edge = derived_edges
            .iter()
            .find(|e| e.from == fot_node)
            .expect("FoT_Mode channel field should have a DerivedFrom edge");

        // Target should be a RegisterField in DMA_S2MM_0_Ctrl
        match &fot_edge.to {
            NodeId::RegisterField { register, field, .. } => {
                assert_eq!(register, "DMA_S2MM_0_Ctrl");
                assert_eq!(field, "FoT_Mode");
            }
            other => panic!("Expected RegisterField, got {:?}", other),
        }
    }

    #[test]
    fn contains_edges_form_hierarchy() {
        use crate::types::{NodeId, RelationshipKind, TileKind};

        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let json_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/aie-device-models.json");
        let mut model =
            crate::device_model::extract_device_model(&json_path, "npu1").unwrap();
        populate_tile_modules(&mut model, &db);

        let contains: Vec<_> = model
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::Contains)
            .collect();

        // TileType -> Module: compute tile has Core and Memory modules
        let tile_to_module: Vec<_> = contains
            .iter()
            .filter(|e| matches!(&e.from, NodeId::TileType { kind: TileKind::Compute }))
            .filter(|e| matches!(&e.to, NodeId::Module { .. }))
            .collect();
        assert_eq!(
            tile_to_module.len(),
            2,
            "compute tile should contain 2 modules (Core + Memory)"
        );

        // Module -> Register: memory module should contain many registers
        let module_to_reg: Vec<_> = contains
            .iter()
            .filter(|e| matches!(
                &e.from,
                NodeId::Module { tile: TileKind::Compute, module: ModuleKind::Memory }
            ))
            .filter(|e| matches!(&e.to, NodeId::Register { .. }))
            .collect();
        assert!(
            module_to_reg.len() > 100,
            "memory module should contain hundreds of registers, got {}",
            module_to_reg.len()
        );

        // Register -> RegisterField: DMA_BD0_0 should contain its fields
        let bd0_0_to_fields: Vec<_> = contains
            .iter()
            .filter(|e| matches!(
                &e.from,
                NodeId::Register { name, .. } if name == "DMA_BD0_0"
            ))
            .filter(|e| matches!(&e.to, NodeId::RegisterField { .. }))
            .collect();
        assert!(
            !bd0_0_to_fields.is_empty(),
            "DMA_BD0_0 should contain field edges"
        );

        // Verify a specific field exists
        let has_buf_len = bd0_0_to_fields.iter().any(|e| matches!(
            &e.to,
            NodeId::RegisterField { field, .. } if field == "Buffer_Length"
        ));
        assert!(has_buf_len, "DMA_BD0_0 should contain Buffer_Length field");
    }

    #[test]
    fn belongs_to_edges_classify_registers() {
        use crate::types::{NodeId, RelationshipKind, SubsystemKind};

        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let json_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/aie-device-models.json");
        let mut model =
            crate::device_model::extract_device_model(&json_path, "npu1").unwrap();
        populate_tile_modules(&mut model, &db);

        let belongs_to: Vec<_> = model
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::BelongsTo)
            .collect();

        // DMA registers should belong to DMA subsystem
        let dma_bd0_belongs = belongs_to.iter().find(|e| matches!(
            &e.from,
            NodeId::Register { name, .. } if name == "DMA_BD0_0"
        ));
        assert!(dma_bd0_belongs.is_some(), "DMA_BD0_0 should have a BelongsTo edge");
        match &dma_bd0_belongs.unwrap().to {
            NodeId::Subsystem { subsystem, .. } => {
                assert_eq!(*subsystem, SubsystemKind::Dma);
            }
            other => panic!("Expected Subsystem node, got {:?}", other),
        }

        // Lock registers should belong to Lock subsystem
        let lock_belongs = belongs_to.iter().find(|e| matches!(
            &e.from,
            NodeId::Register { name, .. } if name == "Lock0_value"
        ));
        assert!(lock_belongs.is_some(), "Lock0_value should have a BelongsTo edge");
        match &lock_belongs.unwrap().to {
            NodeId::Subsystem { subsystem, .. } => {
                assert_eq!(*subsystem, SubsystemKind::Lock);
            }
            other => panic!("Expected Subsystem node, got {:?}", other),
        }
    }

    #[test]
    fn references_edges_connect_cross_resource_fields() {
        use crate::types::{
            BdFieldRole, DmaChannelFieldRole, DmaDirection, NodeId, RelationshipKind,
            SubsystemKind,
        };

        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let json_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/aie-device-models.json");
        let mut model =
            crate::device_model::extract_device_model(&json_path, "npu1").unwrap();
        populate_tile_modules(&mut model, &db);

        let refs: Vec<_> = model
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::References)
            .collect();

        assert!(!refs.is_empty(), "should have References edges");

        // BD LockAcqId -> Lock subsystem
        let lock_acq_ref = refs.iter().find(|e| matches!(
            &e.from,
            NodeId::BdField { role: BdFieldRole::LockAcqId, .. }
        ));
        assert!(lock_acq_ref.is_some(), "LockAcqId should reference Lock subsystem");
        match &lock_acq_ref.unwrap().to {
            NodeId::Subsystem { subsystem, .. } => {
                assert_eq!(*subsystem, SubsystemKind::Lock);
            }
            other => panic!("Expected Subsystem, got {:?}", other),
        }

        // BD LockRelId -> Lock subsystem
        let lock_rel_ref = refs.iter().find(|e| matches!(
            &e.from,
            NodeId::BdField { role: BdFieldRole::LockRelId, .. }
        ));
        assert!(lock_rel_ref.is_some(), "LockRelId should reference Lock subsystem");

        // BD NextBd -> BD resource space (ValidBd as representative)
        let next_bd_ref = refs.iter().find(|e| matches!(
            &e.from,
            NodeId::BdField { role: BdFieldRole::NextBd, .. }
        ));
        assert!(next_bd_ref.is_some(), "NextBd should reference BD space");
        match &next_bd_ref.unwrap().to {
            NodeId::BdField { role: BdFieldRole::ValidBd, .. } => {}
            other => panic!("NextBd should reference ValidBd, got {:?}", other),
        }

        // Channel StartBdId -> BD resource space
        let start_bd_ref = refs.iter().find(|e| matches!(
            &e.from,
            NodeId::ChannelField {
                role: DmaChannelFieldRole::StartBdId,
                direction: DmaDirection::S2mm,
                ..
            }
        ));
        assert!(start_bd_ref.is_some(), "StartBdId should reference BD space");
        match &start_bd_ref.unwrap().to {
            NodeId::BdField { role: BdFieldRole::ValidBd, .. } => {}
            other => panic!("StartBdId should reference ValidBd, got {:?}", other),
        }

        // Channel CurrentBdId -> BD resource space
        let cur_bd_ref = refs.iter().find(|e| matches!(
            &e.from,
            NodeId::ChannelField {
                role: DmaChannelFieldRole::CurrentBdId,
                direction: DmaDirection::S2mm,
                ..
            }
        ));
        assert!(cur_bd_ref.is_some(), "CurrentBdId should reference BD space");
    }
}
