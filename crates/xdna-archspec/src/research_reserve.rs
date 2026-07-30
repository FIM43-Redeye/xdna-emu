//! Versioned NPU1 research-reserve ledger and retirement report.
//!
//! The open-source architecture coverage model supplies one input. External
//! evidence remains untrusted until a later bundle validator audits it.

use crate::types::Architecture;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    path::{Component, Path},
};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReserveLedger {
    pub schema_version: u32,
    pub tuples: Vec<PinnedTuple>,
    pub inventory: Vec<InventoryEntry>,
    pub facts: Vec<HardwareFact>,
    pub evidence: Vec<EvidenceRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedTuple {
    pub id: String,
    pub title: String,
    #[serde(with = "architecture_serde")]
    pub architecture: Architecture,
    pub device: DevicePin,
    pub firmware: ContentPin,
    pub driver_surface: RevisionPin,
    pub kernel_corpus_evidence_ids: Vec<String>,
    pub identity_state: TupleIdentityState,
    pub inventory_scope: InventoryScope,
    pub live_attestation_evidence_ids: Vec<String>,
    pub offline_rehearsal_evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DevicePin {
    pub vendor_id: String,
    pub device_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContentPin {
    pub logical_name: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevisionPin {
    pub repository: String,
    pub commit: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum TupleIdentityState {
    Open { missing_fields: Vec<String> },
    Complete { evidence_ids: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum InventoryScope {
    Open { remaining_sources: Vec<String> },
    Sealed { evidence_ids: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventoryEntry {
    pub id: String,
    pub title: String,
    pub source_refs: Vec<String>,
    pub tuple_ids: Vec<String>,
    pub coverage_domain_ids: Vec<String>,
    pub dependency_ids: Vec<String>,
    pub disposition: InventoryDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum InventoryDisposition {
    Applicable { fact_ids: Vec<String> },
    ProvenNotApplicable { fact_ids: Vec<String> },
    Deferred { reason: String },
    Unknown { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareFact {
    pub id: String,
    pub statement: String,
    pub tuple_ids: Vec<String>,
    pub dependency_fact_ids: Vec<String>,
    pub preconditions: Vec<String>,
    pub initial_state: Vec<String>,
    pub stimulus: Vec<String>,
    pub external_events: Vec<String>,
    pub expected_transition: String,
    pub expected_outputs: Vec<String>,
    pub ordering: Vec<String>,
    pub timing_bounds: Vec<String>,
    pub supporting_evidence_ids: Vec<String>,
    pub control_evidence_ids: Vec<String>,
    pub counterevidence_ids: Vec<String>,
    pub alternatives_ruled_out: Vec<String>,
    pub remaining_unknowns: Vec<String>,
    pub source_refs: Vec<String>,
    pub implementation_refs: Vec<String>,
    pub test_refs: Vec<String>,
    pub promotion: PromotionState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum PromotionState {
    Observed,
    Derived,
    Verified,
    Encoded,
    RetirementQualified,
    Contested {
        reason: String,
        evidence_ids: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRecord {
    pub id: String,
    pub kind: EvidenceKind,
    pub candidate_tuple_ids: Vec<String>,
    pub location: StableLocation,
    pub intake_refs: Vec<String>,
    pub expected_digests: EvidenceDigests,
    pub provenance_gaps: Vec<String>,
    pub retention: RetentionClass,
    pub redistributability: Redistributability,
    pub expected_replicas: Vec<ExpectedReplica>,
    pub preservation_notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    ToolchainSource,
    HardwareWitness,
    HistoricalEmulatorWitness,
    ImplementationFixture,
    Documentation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StableLocation {
    pub alias: String,
    pub relative_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceDigests {
    pub metadata_fingerprint_sha256: Option<String>,
    pub checksum_index_sha256: Option<String>,
    pub manifest_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionClass {
    WorkingCapture,
    WitnessCapture,
    ImplementationFixture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Redistributability {
    Redistributable,
    Restricted,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedReplica {
    pub id: String,
    pub location: StableLocation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub path: String,
    pub message: String,
}

#[derive(Debug)]
pub struct LedgerError {
    pub issues: Vec<ValidationIssue>,
}

impl ReserveLedger {
    pub fn from_json(json: &str) -> Result<Self, LedgerError> {
        let ledger: Self = serde_json::from_str(json).map_err(|error| LedgerError {
            issues: vec![ValidationIssue { path: "$".into(), message: error.to_string() }],
        })?;
        let mut issues = ledger.validation_issues();
        if ledger.schema_version != SCHEMA_VERSION {
            issues.push(ValidationIssue {
                path: "$.schema_version".into(),
                message: format!("unsupported schema_version {}", ledger.schema_version),
            });
        }
        if issues.is_empty() {
            Ok(ledger)
        } else {
            issues.sort_by(|a, b| (&a.path, &a.message).cmp(&(&b.path, &b.message)));
            Err(LedgerError { issues })
        }
    }

    fn validation_issues(&self) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let tuple_ids = validate_record_ids(
            "tuple",
            "tuples",
            "tuple",
            self.tuples.iter().map(|record| &record.id),
            &mut issues,
        );
        let inventory_ids = validate_record_ids(
            "inventory",
            "inventory",
            "inventory",
            self.inventory.iter().map(|record| &record.id),
            &mut issues,
        );
        let fact_ids = validate_record_ids(
            "fact",
            "facts",
            "fact",
            self.facts.iter().map(|record| &record.id),
            &mut issues,
        );
        let evidence_ids = validate_record_ids(
            "evidence",
            "evidence",
            "evidence",
            self.evidence.iter().map(|record| &record.id),
            &mut issues,
        );

        for (index, tuple) in self.tuples.iter().enumerate() {
            let base = format!("$.tuples[{index}]");
            require_text(&tuple.title, &format!("{base}.title"), "tuple title", &mut issues);
            require_text(
                &tuple.device.vendor_id,
                &format!("{base}.device.vendor_id"),
                "vendor id",
                &mut issues,
            );
            require_text(
                &tuple.device.device_id,
                &format!("{base}.device.device_id"),
                "device id",
                &mut issues,
            );
            require_text(
                &tuple.firmware.logical_name,
                &format!("{base}.firmware.logical_name"),
                "firmware logical name",
                &mut issues,
            );
            validate_sha256(&tuple.firmware.sha256, &format!("{base}.firmware.sha256"), &mut issues);
            require_text(
                &tuple.driver_surface.repository,
                &format!("{base}.driver_surface.repository"),
                "driver repository",
                &mut issues,
            );
            require_text(
                &tuple.driver_surface.commit,
                &format!("{base}.driver_surface.commit"),
                "driver commit",
                &mut issues,
            );
            validate_refs(
                &tuple.kernel_corpus_evidence_ids,
                &evidence_ids,
                &format!("{base}.kernel_corpus_evidence_ids"),
                "evidence",
                &mut issues,
            );
            match &tuple.identity_state {
                TupleIdentityState::Open { missing_fields } => {
                    require_nonempty(
                        missing_fields,
                        &format!("{base}.identity_state.missing_fields"),
                        "open tuple identity",
                        &mut issues,
                    );
                    validate_text_list(
                        missing_fields,
                        &format!("{base}.identity_state.missing_fields"),
                        &mut issues,
                    );
                }
                TupleIdentityState::Complete { evidence_ids: refs } => {
                    require_nonempty(
                        refs,
                        &format!("{base}.identity_state.evidence_ids"),
                        "complete tuple identity",
                        &mut issues,
                    );
                    validate_refs(
                        refs,
                        &evidence_ids,
                        &format!("{base}.identity_state.evidence_ids"),
                        "evidence",
                        &mut issues,
                    );
                }
            }
            match &tuple.inventory_scope {
                InventoryScope::Open { remaining_sources } => {
                    require_nonempty(
                        remaining_sources,
                        &format!("{base}.inventory_scope.remaining_sources"),
                        "open inventory scope",
                        &mut issues,
                    );
                    validate_text_list(
                        remaining_sources,
                        &format!("{base}.inventory_scope.remaining_sources"),
                        &mut issues,
                    );
                }
                InventoryScope::Sealed { evidence_ids: refs } => {
                    require_nonempty(
                        refs,
                        &format!("{base}.inventory_scope.evidence_ids"),
                        "sealed inventory scope",
                        &mut issues,
                    );
                    validate_refs(
                        refs,
                        &evidence_ids,
                        &format!("{base}.inventory_scope.evidence_ids"),
                        "evidence",
                        &mut issues,
                    );
                }
            }
            validate_refs(
                &tuple.live_attestation_evidence_ids,
                &evidence_ids,
                &format!("{base}.live_attestation_evidence_ids"),
                "evidence",
                &mut issues,
            );
            validate_refs(
                &tuple.offline_rehearsal_evidence_ids,
                &evidence_ids,
                &format!("{base}.offline_rehearsal_evidence_ids"),
                "evidence",
                &mut issues,
            );
        }

        for (index, entry) in self.inventory.iter().enumerate() {
            let base = format!("$.inventory[{index}]");
            require_text(&entry.title, &format!("{base}.title"), "inventory title", &mut issues);
            validate_text_list(&entry.source_refs, &format!("{base}.source_refs"), &mut issues);
            require_nonempty(
                &entry.tuple_ids,
                &format!("{base}.tuple_ids"),
                "inventory tuple scope",
                &mut issues,
            );
            validate_refs(&entry.tuple_ids, &tuple_ids, &format!("{base}.tuple_ids"), "tuple", &mut issues);
            for (domain_index, domain) in entry.coverage_domain_ids.iter().enumerate() {
                if !crate::coverage::spine_ids::SPINE_DOMAIN_IDS.contains(&domain.as_str()) {
                    issues.push(ValidationIssue {
                        path: format!("{base}.coverage_domain_ids[{domain_index}]"),
                        message: format!("unknown coverage domain `{domain}`"),
                    });
                }
            }
            validate_refs(
                &entry.dependency_ids,
                &inventory_ids,
                &format!("{base}.dependency_ids"),
                "inventory",
                &mut issues,
            );
            match &entry.disposition {
                InventoryDisposition::Applicable { fact_ids: refs }
                | InventoryDisposition::ProvenNotApplicable { fact_ids: refs } => {
                    if refs.is_empty() {
                        issues.push(ValidationIssue {
                            path: format!("{base}.disposition.fact_ids"),
                            message: "fact-bearing disposition requires at least one fact".into(),
                        });
                    }
                    validate_refs(
                        refs,
                        &fact_ids,
                        &format!("{base}.disposition.fact_ids"),
                        "fact",
                        &mut issues,
                    );
                }
                InventoryDisposition::Deferred { reason } | InventoryDisposition::Unknown { reason } => {
                    require_text(
                        reason,
                        &format!("{base}.disposition.reason"),
                        "disposition reason",
                        &mut issues,
                    );
                }
            }
        }

        for (index, fact) in self.facts.iter().enumerate() {
            let base = format!("$.facts[{index}]");
            require_text(&fact.statement, &format!("{base}.statement"), "fact statement", &mut issues);
            require_nonempty(&fact.tuple_ids, &format!("{base}.tuple_ids"), "fact tuple scope", &mut issues);
            validate_refs(&fact.tuple_ids, &tuple_ids, &format!("{base}.tuple_ids"), "tuple", &mut issues);
            validate_refs(
                &fact.dependency_fact_ids,
                &fact_ids,
                &format!("{base}.dependency_fact_ids"),
                "fact",
                &mut issues,
            );
            for (name, values) in [
                ("preconditions", &fact.preconditions),
                ("initial_state", &fact.initial_state),
                ("stimulus", &fact.stimulus),
                ("external_events", &fact.external_events),
                ("expected_outputs", &fact.expected_outputs),
                ("ordering", &fact.ordering),
                ("timing_bounds", &fact.timing_bounds),
                ("alternatives_ruled_out", &fact.alternatives_ruled_out),
                ("remaining_unknowns", &fact.remaining_unknowns),
                ("source_refs", &fact.source_refs),
                ("implementation_refs", &fact.implementation_refs),
                ("test_refs", &fact.test_refs),
            ] {
                validate_text_list(values, &format!("{base}.{name}"), &mut issues);
            }
            require_text(
                &fact.expected_transition,
                &format!("{base}.expected_transition"),
                "expected transition",
                &mut issues,
            );
            for (name, refs) in [
                ("supporting_evidence_ids", &fact.supporting_evidence_ids),
                ("control_evidence_ids", &fact.control_evidence_ids),
                ("counterevidence_ids", &fact.counterevidence_ids),
            ] {
                validate_refs(refs, &evidence_ids, &format!("{base}.{name}"), "evidence", &mut issues);
            }
            if let PromotionState::Contested { reason, evidence_ids: refs } = &fact.promotion {
                require_text(reason, &format!("{base}.promotion.reason"), "contested reason", &mut issues);
                require_nonempty(
                    refs,
                    &format!("{base}.promotion.evidence_ids"),
                    "contested fact",
                    &mut issues,
                );
                validate_refs(
                    refs,
                    &evidence_ids,
                    &format!("{base}.promotion.evidence_ids"),
                    "evidence",
                    &mut issues,
                );
            }
        }

        for (index, evidence) in self.evidence.iter().enumerate() {
            let base = format!("$.evidence[{index}]");
            require_nonempty(
                &evidence.candidate_tuple_ids,
                &format!("{base}.candidate_tuple_ids"),
                "evidence tuple scope",
                &mut issues,
            );
            validate_refs(
                &evidence.candidate_tuple_ids,
                &tuple_ids,
                &format!("{base}.candidate_tuple_ids"),
                "tuple",
                &mut issues,
            );
            validate_location(&evidence.location, &format!("{base}.location"), &mut issues);
            validate_text_list(&evidence.intake_refs, &format!("{base}.intake_refs"), &mut issues);
            for (name, digest) in [
                (
                    "metadata_fingerprint_sha256",
                    evidence.expected_digests.metadata_fingerprint_sha256.as_deref(),
                ),
                ("checksum_index_sha256", evidence.expected_digests.checksum_index_sha256.as_deref()),
                ("manifest_sha256", evidence.expected_digests.manifest_sha256.as_deref()),
            ] {
                if let Some(digest) = digest {
                    validate_sha256(digest, &format!("{base}.expected_digests.{name}"), &mut issues);
                }
            }
            validate_text_list(&evidence.provenance_gaps, &format!("{base}.provenance_gaps"), &mut issues);
            validate_text_list(
                &evidence.preservation_notes,
                &format!("{base}.preservation_notes"),
                &mut issues,
            );
            let mut replica_ids = BTreeSet::new();
            for (replica_index, replica) in evidence.expected_replicas.iter().enumerate() {
                let replica_base = format!("{base}.expected_replicas[{replica_index}]");
                if !valid_dotted_id(&replica.id) {
                    issues.push(ValidationIssue {
                        path: format!("{replica_base}.id"),
                        message: format!("invalid replica id `{}`", replica.id),
                    });
                } else if !replica_ids.insert(replica.id.as_str()) {
                    issues.push(ValidationIssue {
                        path: format!("{replica_base}.id"),
                        message: format!("duplicate replica id `{}`", replica.id),
                    });
                }
                validate_location(&replica.location, &format!("{replica_base}.location"), &mut issues);
            }
        }

        let inventory_graph = self
            .inventory
            .iter()
            .map(|entry| (entry.id.clone(), entry.dependency_ids.clone()))
            .collect();
        if let Some(cycle) = dependency_cycle(&inventory_graph) {
            issues.push(ValidationIssue {
                path: "$.inventory".into(),
                message: format!("inventory dependency cycle: {}", cycle.join(" -> ")),
            });
        }
        let fact_graph = self
            .facts
            .iter()
            .map(|fact| (fact.id.clone(), fact.dependency_fact_ids.clone()))
            .collect();
        if let Some(cycle) = dependency_cycle(&fact_graph) {
            issues.push(ValidationIssue {
                path: "$.facts".into(),
                message: format!("fact dependency cycle: {}", cycle.join(" -> ")),
            });
        }

        issues
    }
}

fn validate_record_ids<'a>(
    kind: &str,
    collection: &str,
    prefix: &str,
    ids: impl Iterator<Item = &'a String>,
    issues: &mut Vec<ValidationIssue>,
) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    for (index, id) in ids.enumerate() {
        let path = format!("$.{collection}[{index}].id");
        if !id.starts_with(&format!("{prefix}.")) || !valid_dotted_id(id) {
            issues.push(ValidationIssue { path, message: format!("invalid {kind} id `{id}`") });
        } else if !seen.insert(id.clone()) {
            issues.push(ValidationIssue { path, message: format!("duplicate {kind} id `{id}`") });
        }
    }
    seen
}

fn valid_dotted_id(id: &str) -> bool {
    !id.is_empty()
        && id.split('.').all(|segment| {
            let mut bytes = segment.bytes();
            matches!(bytes.next(), Some(first) if first.is_ascii_lowercase() || first.is_ascii_digit())
                && bytes.all(|byte| {
                    byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
                })
        })
}

fn require_text(value: &str, path: &str, label: &str, issues: &mut Vec<ValidationIssue>) {
    if value.trim().is_empty() {
        issues.push(ValidationIssue { path: path.into(), message: format!("{label} must not be blank") });
    }
}

fn require_nonempty<T>(values: &[T], path: &str, label: &str, issues: &mut Vec<ValidationIssue>) {
    if values.is_empty() {
        issues.push(ValidationIssue {
            path: path.into(),
            message: format!("{label} requires at least one value"),
        });
    }
}

fn validate_text_list(values: &[String], path: &str, issues: &mut Vec<ValidationIssue>) {
    for (index, value) in values.iter().enumerate() {
        require_text(value, &format!("{path}[{index}]"), "list entry", issues);
    }
}

fn validate_refs(
    refs: &[String],
    known: &BTreeSet<String>,
    path: &str,
    kind: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    for (index, reference) in refs.iter().enumerate() {
        if !known.contains(reference) {
            issues.push(ValidationIssue {
                path: format!("{path}[{index}]"),
                message: format!("unknown {kind} id `{reference}`"),
            });
        }
    }
}

fn validate_sha256(value: &str, path: &str, issues: &mut Vec<ValidationIssue>) {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f')) {
        issues.push(ValidationIssue {
            path: path.into(),
            message: format!("invalid lowercase SHA-256 `{value}`"),
        });
    }
}

fn validate_location(location: &StableLocation, path: &str, issues: &mut Vec<ValidationIssue>) {
    require_text(&location.alias, &format!("{path}.alias"), "location alias", issues);
    let relative = Path::new(&location.relative_path);
    if location.relative_path.is_empty()
        || relative.is_absolute()
        || !relative.components().all(|component| matches!(component, Component::Normal(_)))
    {
        issues.push(ValidationIssue {
            path: format!("{path}.relative_path"),
            message: format!("invalid relative path `{}`", location.relative_path),
        });
    }
}

fn dependency_cycle(graph: &BTreeMap<String, Vec<String>>) -> Option<Vec<String>> {
    fn visit(
        node: &str,
        graph: &BTreeMap<String, Vec<String>>,
        active: &mut Vec<String>,
        complete: &mut BTreeSet<String>,
    ) -> Option<Vec<String>> {
        if let Some(start) = active.iter().position(|item| item == node) {
            let mut cycle = active[start..].to_vec();
            cycle.push(node.to_string());
            return Some(cycle);
        }
        if complete.contains(node) {
            return None;
        }
        active.push(node.to_string());
        if let Some(dependencies) = graph.get(node) {
            for dependency in dependencies {
                if graph.contains_key(dependency) {
                    if let Some(cycle) = visit(dependency, graph, active, complete) {
                        return Some(cycle);
                    }
                }
            }
        }
        active.pop();
        complete.insert(node.to_string());
        None
    }

    let mut complete = BTreeSet::new();
    for node in graph.keys() {
        if let Some(cycle) = visit(node, graph, &mut Vec::new(), &mut complete) {
            return Some(cycle);
        }
    }
    None
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, issue) in self.issues.iter().enumerate() {
            if index != 0 {
                write!(f, "; ")?;
            }
            write!(f, "{}: {}", issue.path, issue.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for LedgerError {}

mod architecture_serde {
    use crate::types::Architecture;
    use serde::{de::Error, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(architecture: &Architecture, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(match architecture {
            Architecture::Aie => "aie",
            Architecture::Aie2 => "aie2",
            Architecture::Aie2p => "aie2p",
        })
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Architecture, D::Error>
    where
        D: Deserializer<'de>,
    {
        match String::deserialize(deserializer)?.as_str() {
            "aie" => Ok(Architecture::Aie),
            "aie2" => Ok(Architecture::Aie2),
            "aie2p" => Ok(Architecture::Aie2p),
            other => Err(D::Error::custom(format!("unknown architecture `{other}`"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_LEDGER: &str = r#"
    {
      "schema_version": 1,
      "tuples": [{
        "id": "tuple.test.aie2",
        "title": "Test AIE2 tuple",
        "architecture": "aie2",
        "device": {
          "vendor_id": "1022",
          "device_id": "1502"
        },
        "firmware": {
          "logical_name": "firmware.bin",
          "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        },
        "driver_surface": {
          "repository": "amdxdna",
          "commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        },
        "kernel_corpus_evidence_ids": [],
        "identity_state": {
          "state": "open",
          "missing_fields": ["host_kernel"]
        },
        "inventory_scope": {
          "state": "open",
          "remaining_sources": ["driver"]
        },
        "live_attestation_evidence_ids": [],
        "offline_rehearsal_evidence_ids": []
      }],
      "inventory": [],
      "facts": [],
      "evidence": []
    }"#;

    const LINKED_LEDGER: &str = r#"
    {
      "schema_version": 1,
      "tuples": [{
        "id": "tuple.test.aie2",
        "title": "Test AIE2 tuple",
        "architecture": "aie2",
        "device": {
          "vendor_id": "1022",
          "device_id": "1502"
        },
        "firmware": {
          "logical_name": "firmware.bin",
          "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        },
        "driver_surface": {
          "repository": "amdxdna",
          "commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        },
        "kernel_corpus_evidence_ids": ["evidence.test.hw"],
        "identity_state": {
          "state": "complete",
          "evidence_ids": ["evidence.test.hw"]
        },
        "inventory_scope": {
          "state": "sealed",
          "evidence_ids": ["evidence.test.hw"]
        },
        "live_attestation_evidence_ids": ["evidence.test.hw"],
        "offline_rehearsal_evidence_ids": ["evidence.test.hw"]
      }],
      "inventory": [{
        "id": "inventory.test.command",
        "title": "Command execution",
        "source_refs": ["driver:aie2_msg.c"],
        "tuple_ids": ["tuple.test.aie2"],
        "coverage_domain_ids": ["dma"],
        "dependency_ids": [],
        "disposition": {
          "state": "applicable",
          "fact_ids": ["fact.test.command"]
        }
      }],
      "facts": [{
        "id": "fact.test.command",
        "statement": "A controlled command reaches completion.",
        "tuple_ids": ["tuple.test.aie2"],
        "dependency_fact_ids": [],
        "preconditions": ["context configured"],
        "initial_state": ["command idle"],
        "stimulus": ["submit command"],
        "external_events": ["none"],
        "expected_transition": "idle to complete",
        "expected_outputs": ["success"],
        "ordering": ["completion after submission"],
        "timing_bounds": ["finite"],
        "supporting_evidence_ids": ["evidence.test.hw"],
        "control_evidence_ids": ["evidence.test.hw"],
        "counterevidence_ids": [],
        "alternatives_ruled_out": ["host-only completion"],
        "remaining_unknowns": [],
        "source_refs": ["driver:aie2_msg.c"],
        "implementation_refs": ["src/firmware/mod.rs"],
        "test_refs": ["research_reserve::tests::release_closed"],
        "promotion": {
          "state": "retirement_qualified"
        }
      }],
      "evidence": [{
        "id": "evidence.test.hw",
        "kind": "hardware_witness",
        "candidate_tuple_ids": ["tuple.test.aie2"],
        "location": {
          "alias": "reserve",
          "relative_path": "bundles/test-hw"
        },
        "intake_refs": ["docs/evidence/test-hw.md"],
        "expected_digests": {
          "metadata_fingerprint_sha256": null,
          "checksum_index_sha256": null,
          "manifest_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
        },
        "provenance_gaps": [],
        "retention": "witness_capture",
        "redistributability": "restricted",
        "expected_replicas": [{
          "id": "replica.test.one",
          "location": {
            "alias": "replica-one",
            "relative_path": "npu1/test-hw"
          }
        }, {
          "id": "replica.test.two",
          "location": {
            "alias": "replica-two",
            "relative_path": "npu1/test-hw"
          }
        }],
        "preservation_notes": []
      }]
    }"#;

    fn linked_value() -> serde_json::Value {
        serde_json::from_str(LINKED_LEDGER).expect("linked test ledger must be valid JSON")
    }

    fn replace(value: &mut serde_json::Value, pointer: &str, replacement: serde_json::Value) {
        *value
            .pointer_mut(pointer)
            .unwrap_or_else(|| panic!("missing test pointer {pointer}")) = replacement;
    }

    fn assert_validation_error(value: serde_json::Value, expected: &str) {
        let json = serde_json::to_string(&value).expect("test ledger must serialize");
        let err = ReserveLedger::from_json(&json).expect_err("mutated ledger must fail validation");
        assert!(err.to_string().contains(expected), "expected `{expected}`, got `{err}`");
    }

    #[test]
    fn parse_rejects_unsupported_schema_version() {
        let input = MINIMAL_LEDGER.replacen("\"schema_version\": 1", "\"schema_version\": 2", 1);
        let err = ReserveLedger::from_json(&input).expect_err("version 2 must fail closed");
        assert!(err.to_string().contains("unsupported schema_version 2"), "unexpected error: {err}");
    }

    #[test]
    fn parse_rejects_unknown_root_fields() {
        let input =
            MINIMAL_LEDGER.replacen("\"schema_version\": 1,", "\"schema_version\": 1, \"extra\": true,", 1);
        let err = ReserveLedger::from_json(&input).expect_err("unknown root field must fail closed");
        assert!(err.to_string().contains("unknown field `extra`"), "unexpected error: {err}");
    }

    #[test]
    fn parse_round_trips_architecture_as_snake_case() {
        let ledger = ReserveLedger::from_json(MINIMAL_LEDGER).expect("minimal ledger must parse");
        assert_eq!(ledger.tuples[0].architecture, crate::types::Architecture::Aie2);

        let json = serde_json::to_value(&ledger).expect("ledger must serialize");
        assert_eq!(json["tuples"][0]["architecture"], "aie2");
    }

    #[test]
    fn parse_accepts_minimal_open_ledger() {
        let ledger = ReserveLedger::from_json(MINIMAL_LEDGER).expect("minimal ledger must parse");
        assert_eq!(ledger.schema_version, 1);
        assert_eq!(ledger.tuples.len(), 1);
        assert!(ledger.inventory.is_empty());
        assert!(ledger.facts.is_empty());
        assert!(ledger.evidence.is_empty());
    }

    #[test]
    fn validation_rejects_malformed_duplicate_and_wrong_kind_ids() {
        for (pointer, replacement, expected) in [
            ("/tuples/0/id", serde_json::json!(""), "invalid tuple id"),
            ("/tuples/0/id", serde_json::json!("tuple.Test"), "invalid tuple id"),
            ("/facts/0/id", serde_json::json!("inventory.test.command"), "invalid fact id"),
        ] {
            let mut value = linked_value();
            replace(&mut value, pointer, replacement);
            assert_validation_error(value, expected);
        }

        let mut value = linked_value();
        let duplicate = value["tuples"][0].clone();
        value["tuples"].as_array_mut().unwrap().push(duplicate);
        assert_validation_error(value, "duplicate tuple id");
    }

    #[test]
    fn validation_reports_inventory_ids_under_the_inventory_root() {
        let mut value = linked_value();
        replace(&mut value, "/inventory/0/id", serde_json::json!("bad"));
        let json = serde_json::to_string(&value).unwrap();
        let err = ReserveLedger::from_json(&json).expect_err("malformed inventory id must fail");
        assert!(
            err.issues.iter().any(|issue| issue.path == "$.inventory[0].id"),
            "issues used the wrong JSON root: {:?}",
            err.issues
        );
    }

    #[test]
    fn validation_rejects_empty_fact_bearing_dispositions() {
        for state in ["applicable", "proven_not_applicable"] {
            let mut value = linked_value();
            replace(
                &mut value,
                "/inventory/0/disposition",
                serde_json::json!({"state": state, "fact_ids": []}),
            );
            assert_validation_error(value, "requires at least one fact");
        }
    }

    #[test]
    fn validation_rejects_dangling_references() {
        for (pointer, replacement, expected) in [
            ("/inventory/0/tuple_ids/0", serde_json::json!("tuple.missing"), "unknown tuple id"),
            ("/inventory/0/dependency_ids", serde_json::json!(["inventory.missing"]), "unknown inventory id"),
            ("/facts/0/dependency_fact_ids", serde_json::json!(["fact.missing"]), "unknown fact id"),
            (
                "/facts/0/supporting_evidence_ids/0",
                serde_json::json!("evidence.missing"),
                "unknown evidence id",
            ),
            (
                "/tuples/0/kernel_corpus_evidence_ids/0",
                serde_json::json!("evidence.missing"),
                "unknown evidence id",
            ),
            ("/evidence/0/candidate_tuple_ids/0", serde_json::json!("tuple.missing"), "unknown tuple id"),
        ] {
            let mut value = linked_value();
            replace(&mut value, pointer, replacement);
            assert_validation_error(value, expected);
        }
    }

    #[test]
    fn validation_rejects_unknown_coverage_domains() {
        let mut value = linked_value();
        replace(&mut value, "/inventory/0/coverage_domain_ids/0", serde_json::json!("not_a_spine_domain"));
        assert_validation_error(value, "unknown coverage domain");
    }

    #[test]
    fn validation_rejects_malformed_sha256_values() {
        for pointer in ["/tuples/0/firmware/sha256", "/evidence/0/expected_digests/manifest_sha256"] {
            let mut value = linked_value();
            replace(&mut value, pointer, serde_json::json!("ABC123"));
            assert_validation_error(value, "invalid lowercase SHA-256");
        }
    }

    #[test]
    fn validation_rejects_unsafe_external_locations() {
        for path in ["", "/absolute/path", "../escape", "safe/../escape"] {
            let mut value = linked_value();
            replace(&mut value, "/evidence/0/location/relative_path", serde_json::json!(path));
            assert_validation_error(value, "invalid relative path");
        }

        let mut value = linked_value();
        replace(&mut value, "/evidence/0/location/alias", serde_json::json!(" "));
        assert_validation_error(value, "location alias must not be blank");
    }

    #[test]
    fn validation_rejects_empty_required_text() {
        for (pointer, expected) in [
            ("/tuples/0/title", "tuple title must not be blank"),
            ("/inventory/0/title", "inventory title must not be blank"),
            ("/facts/0/statement", "fact statement must not be blank"),
            ("/facts/0/expected_transition", "expected transition must not be blank"),
        ] {
            let mut value = linked_value();
            replace(&mut value, pointer, serde_json::json!("  "));
            assert_validation_error(value, expected);
        }
    }

    #[test]
    fn validation_rejects_inventory_dependency_cycles() {
        let mut self_cycle = linked_value();
        replace(
            &mut self_cycle,
            "/inventory/0/dependency_ids",
            serde_json::json!(["inventory.test.command"]),
        );
        assert_validation_error(self_cycle, "inventory dependency cycle");

        let mut two_node_cycle = linked_value();
        let mut second = two_node_cycle["inventory"][0].clone();
        second["id"] = serde_json::json!("inventory.test.second");
        second["dependency_ids"] = serde_json::json!(["inventory.test.command"]);
        second["disposition"]["fact_ids"] = serde_json::json!(["fact.test.command"]);
        two_node_cycle["inventory"].as_array_mut().unwrap().push(second);
        two_node_cycle["inventory"][0]["dependency_ids"] = serde_json::json!(["inventory.test.second"]);
        assert_validation_error(two_node_cycle, "inventory dependency cycle");
    }

    #[test]
    fn validation_rejects_fact_dependency_cycles() {
        let mut self_cycle = linked_value();
        replace(&mut self_cycle, "/facts/0/dependency_fact_ids", serde_json::json!(["fact.test.command"]));
        assert_validation_error(self_cycle, "fact dependency cycle");

        let mut two_node_cycle = linked_value();
        let mut second = two_node_cycle["facts"][0].clone();
        second["id"] = serde_json::json!("fact.test.second");
        second["dependency_fact_ids"] = serde_json::json!(["fact.test.command"]);
        two_node_cycle["facts"].as_array_mut().unwrap().push(second);
        two_node_cycle["facts"][0]["dependency_fact_ids"] = serde_json::json!(["fact.test.second"]);
        assert_validation_error(two_node_cycle, "fact dependency cycle");
    }
}
