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

use crate::device::regdb;

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
    /// Which subsystem this profile describes.
    pub subsystem: &'static str,
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
    subsystem: &'static str,
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
        subsystem,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::regdb::{AccessMode, BitField, RegisterDef};

    /// Helper: create a minimal RegisterDef with given name, offset, and field names.
    fn make_reg(name: &str, offset: u32, field_names: &[&str]) -> RegisterDef {
        RegisterDef {
            name: name.to_string(),
            offset,
            width: 32,
            access: AccessMode::ReadWrite,
            reset_value: 0,
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
        regdb::RegisterDb::load_for_device("aie2").ok()
    }

    /// Filter registers whose name starts with any of the given prefixes.
    fn filter_by_prefix<'a>(
        registers: &'a [regdb::RegisterDef],
        prefixes: &[&str],
    ) -> Vec<regdb::RegisterDef> {
        registers
            .iter()
            .filter(|r| prefixes.iter().any(|p| r.name.starts_with(p)))
            .cloned()
            .collect()
    }

    #[test]
    fn integration_memory_module_lock_profile() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found (set MLIR_AIE_PATH)");
            return;
        };

        let mem = db.module("memory").expect("memory module should exist");
        let lock_regs = filter_by_prefix(&mem.registers, &["Lock", "Locks_"]);

        // We know from exploration: 16 value + 1 request + 8 event sel
        // + 1 overflow + 1 underflow = 27 lock registers
        assert!(
            lock_regs.len() >= 27,
            "Expected at least 27 lock registers in memory module, got {}",
            lock_regs.len()
        );

        let profile = build_subsystem_profile("lock", &lock_regs);

        // Core behavioral property: 16 lock instances
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
        let lock_regs = filter_by_prefix(&mt.registers, &["Lock", "Locks_"]);

        let profile = build_subsystem_profile("lock", &lock_regs);

        // MemTile has 64 locks, not 16
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
        let lock_regs = filter_by_prefix(&shim.registers, &["Lock", "Locks_"]);

        let profile = build_subsystem_profile("lock", &lock_regs);

        // Shim has 16 locks (same as compute memory module)
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
    // DMA integration tests -- real AM025 data
    // ====================================================================

    #[test]
    fn integration_memory_module_dma_profile() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let mem = db.module("memory").expect("memory module should exist");
        let dma_regs = filter_by_prefix(&mem.registers, &["DMA_"]);

        // Memory module has ~112 DMA registers per the AM025 survey
        assert!(
            dma_regs.len() >= 100,
            "Expected at least 100 DMA registers in memory module, got {}",
            dma_regs.len()
        );

        let profile = build_subsystem_profile("dma", &dma_regs);

        // Report what we found
        eprintln!("=== Memory Module DMA Profile ===");
        eprintln!("Instance count: {}", profile.instance_count);
        eprintln!("Groups ({}):", profile.register_groups.len());
        for (cat, g) in &profile.register_groups {
            eprintln!(
                "  {:?} | {} (dims={:?}, base=0x{:X})",
                cat, g.pattern, g.dimensions, g.base_offset
            );
        }
        eprintln!("Singletons ({}):", profile.singletons.len());
        for (cat, name) in &profile.singletons {
            eprintln!("  {:?} | {}", cat, name);
        }

        // BD group should exist with 2D structure
        let bd_group = profile
            .register_groups
            .iter()
            .find(|(_, g)| g.pattern == "DMA_BD{}_{}")
            .expect("BD group should exist");
        assert_eq!(bd_group.0, RegisterCategory::State, "BDs are R/W grouped");
        // 2D: 16 BDs x 6 words
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

        // With 2D BD detection, instance_count should be 16 (not 96)
        assert_eq!(profile.instance_count, 16, "DMA instance count = BD count");
    }

    #[test]
    fn integration_memtile_dma_profile() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let mt = db.module("memory_tile").expect("memory_tile module should exist");
        let dma_regs = filter_by_prefix(&mt.registers, &["DMA_"]);

        // MemTile has ~433 DMA registers (48 BDs x 8 words + 6 channels each dir)
        assert!(
            dma_regs.len() >= 400,
            "Expected at least 400 DMA registers in memtile, got {}",
            dma_regs.len()
        );

        let profile = build_subsystem_profile("dma", &dma_regs);

        eprintln!("=== MemTile DMA Profile ===");
        eprintln!("Instance count: {}", profile.instance_count);
        eprintln!("Groups ({}):", profile.register_groups.len());
        for (cat, g) in &profile.register_groups {
            eprintln!(
                "  {:?} | {} (dims={:?}, base=0x{:X})",
                cat, g.pattern, g.dimensions, g.base_offset
            );
        }
        eprintln!("Singletons ({}):", profile.singletons.len());
        for (cat, name) in &profile.singletons {
            eprintln!("  {:?} | {}", cat, name);
        }

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
        let dma_regs = filter_by_prefix(&shim.registers, &["DMA_"]);

        let profile = build_subsystem_profile("dma", &dma_regs);

        eprintln!("=== Shim DMA Profile ===");
        eprintln!("Instance count: {}", profile.instance_count);
        eprintln!("Groups ({}):", profile.register_groups.len());
        for (cat, g) in &profile.register_groups {
            eprintln!(
                "  {:?} | {} (dims={:?}, base=0x{:X})",
                cat, g.pattern, g.dimensions, g.base_offset
            );
        }
        eprintln!("Singletons ({}):", profile.singletons.len());
        for (cat, name) in &profile.singletons {
            eprintln!("  {:?} | {}", cat, name);
        }

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
    fn integration_core_has_no_locks() {
        let Some(db) = load_test_db() else {
            eprintln!("Skipping: register database not found");
            return;
        };

        let core = db.module("core").expect("core module should exist");
        let lock_regs = filter_by_prefix(&core.registers, &["Lock", "Locks_"]);

        // Core module has NO lock registers (locks are in the memory module)
        assert!(
            lock_regs.is_empty(),
            "Core module should have no lock registers, found: {:?}",
            lock_regs.iter().map(|r| &r.name).collect::<Vec<_>>()
        );
    }
}
