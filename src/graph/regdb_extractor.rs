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

    // Instance count: the largest State group's count, or 0
    let instance_count = classified_groups
        .iter()
        .filter(|(cat, _)| *cat == RegisterCategory::State)
        .map(|(_, g)| g.count)
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

/// A group of registers sharing a common name pattern with varying index.
///
/// For example, Lock0_value through Lock15_value form a group with
/// pattern "Lock{}_value", count=16, base_offset=0x1F000, stride=0x10.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterGroup {
    /// Pattern name with index replaced by `{}`, e.g., "Lock{}_value".
    pub pattern: String,
    /// Number of registers in this group.
    pub count: u32,
    /// Offset of the first register.
    pub base_offset: u32,
    /// Byte distance between consecutive registers (0 if singleton).
    pub stride: u32,
    /// Field names shared by all registers in this group.
    pub field_names: Vec<String>,
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
        let stride = if regs.len() >= 2 {
            regs[1].offset - regs[0].offset
        } else {
            0
        };

        // Collect field names from the first register (shared across group)
        let field_names: Vec<String> = regs[0].fields.iter().map(|f| f.name.clone()).collect();

        groups.push(RegisterGroup {
            pattern,
            count: regs.len() as u32,
            base_offset,
            stride,
            field_names,
        });
    }

    groups
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
        assert_eq!(g.count, 16);
        assert_eq!(g.base_offset, 0x1F000);
        assert_eq!(g.stride, 0x10);
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
        assert_eq!(g.count, 8);
        assert_eq!(g.base_offset, 0x1F100);
        assert_eq!(g.stride, 0x4);
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
        assert_eq!(value_group.1.count, 16);

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
        assert_eq!(g.count, 64);
        assert_eq!(g.base_offset, 0xC0000);
        assert_eq!(g.stride, 0x10);
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
        assert_eq!(value_group.1.count, 16);
        assert_eq!(value_group.1.stride, 0x10);
        assert_eq!(value_group.1.base_offset, 0x1F000);

        // Event selection registers: grouped, 8 selectors
        let event_group = profile
            .register_groups
            .iter()
            .find(|(_, g)| g.pattern == "Locks_Event_Selection_{}")
            .expect("Event selection group should exist in real data");
        assert_eq!(event_group.1.count, 8);

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
        assert_eq!(value_group.1.count, 64);
        assert_eq!(value_group.1.stride, 0x10);
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
        assert_eq!(event_group.1.count, 6);
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
